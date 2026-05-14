/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_NPU_TPREFETCH_L2_HPP
#define PTO_NPU_TPREFETCH_L2_HPP

// TPREFETCH_L2 — L2 cache prefetch via SDMA CMO (opcode = 6).
//
// This instruction is logically a *memory access* instruction (it stages data
// from GM/HBM into the on-chip L2 cache so that subsequent TLOADs hit warm
// lines). It happens to *implement* itself by submitting an SDMA CMO SQE from
// the AI Core, which means it has to depend on the SDMA infrastructure that
// also backs TPUT_ASYNC / TGET_ASYNC. We keep the file here, alongside other
// per-arch compute/memory-access instruction headers, so its placement reflects
// what users see at the API surface (a memory-access instruction in `pto::`)
// rather than what the implementation reaches into (the comm SDMA stack).

#include "pto/common/debug.h"
#include "pto/common/type.hpp"
#include "pto/common/constants.hpp"
#include "pto/comm/comm_types.hpp"
#include "pto/comm/async_common/async_types.hpp"
#include "pto/comm/async_common/async_event_impl.hpp"
#include "pto/npu/comm/async/sdma/sdma_async_intrin.hpp"
#include "pto/npu/comm/async/sdma/sdma_cmo_intrin.hpp"

namespace pto {

namespace detail {

// 256B UB scratch tile used to build a transient AsyncSession when the user
// invokes the workspace-based TPREFETCH_L2 overload. Sized to satisfy the
// SDMA UB alignment requirement; one tile per call lives on the AICORE stack
// frame and is destroyed when the IMPL function returns.
using PrefetchL2ScratchTile = pto::Tile<pto::TileType::Vec, uint8_t, 1, comm::sdma::UB_ALIGN_SIZE>;

template <typename GlobalData>
PTO_INTERNAL bool TPrefetchL2IsFlatContiguous1D(GlobalData &globalData)
{
    const int dim0 = globalData.GetShape(GlobalTensorDim::DIM_0);
    const int dim1 = globalData.GetShape(GlobalTensorDim::DIM_1);
    const int dim2 = globalData.GetShape(GlobalTensorDim::DIM_2);
    const int dim3 = globalData.GetShape(GlobalTensorDim::DIM_3);
    const int dim4 = globalData.GetShape(GlobalTensorDim::DIM_4);

    const int pitch0 = globalData.GetStride(GlobalTensorDim::DIM_0);
    const int pitch1 = globalData.GetStride(GlobalTensorDim::DIM_1);
    const int pitch2 = globalData.GetStride(GlobalTensorDim::DIM_2);
    const int pitch3 = globalData.GetStride(GlobalTensorDim::DIM_3);
    const int pitch4 = globalData.GetStride(GlobalTensorDim::DIM_4);

    const bool hasPackedLayout = (pitch4 == 1) && (pitch3 == dim4) && (pitch2 == dim3 * pitch3) &&
                                 (pitch1 == dim2 * pitch2) && (pitch0 == dim1 * pitch1);
    const bool isSingleLine = (dim0 == 1 && dim1 == 1 && dim2 == 1 && dim3 == 1);
    return hasPackedLayout && isSingleLine;
}

template <typename GlobalData>
PTO_INTERNAL uint64_t TPrefetchL2GetTotalBytes(GlobalData &globalData)
{
    const uint64_t d0 = static_cast<uint64_t>(globalData.GetShape(GlobalTensorDim::DIM_0));
    const uint64_t d1 = static_cast<uint64_t>(globalData.GetShape(GlobalTensorDim::DIM_1));
    const uint64_t d2 = static_cast<uint64_t>(globalData.GetShape(GlobalTensorDim::DIM_2));
    const uint64_t d3 = static_cast<uint64_t>(globalData.GetShape(GlobalTensorDim::DIM_3));
    const uint64_t d4 = static_cast<uint64_t>(globalData.GetShape(GlobalTensorDim::DIM_4));
    using T = typename GlobalData::RawDType;
    return (((d0 * d1) * d2) * d3) * d4 * sizeof(T);
}

template <typename GlobalData>
PTO_INTERNAL comm::AsyncEvent TPrefetchL2SdmaImpl(GlobalData &srcGlobalData,
                                                  const comm::sdma::SdmaExecContext &execCtx)
{
    if (srcGlobalData.data() == nullptr) {
        return comm::AsyncEvent(0, comm::DmaEngine::SDMA);
    }

    if (!TPrefetchL2IsFlatContiguous1D(srcGlobalData)) {
        return comm::AsyncEvent(0, comm::DmaEngine::SDMA);
    }

    const uint64_t totalBytes = TPrefetchL2GetTotalBytes(srcGlobalData);
    if (totalBytes == 0) {
        return comm::AsyncEvent(0, comm::DmaEngine::SDMA);
    }

    const uint64_t eventHandle = comm::sdma::__sdma_cmo_prefetch(srcGlobalData.data(), totalBytes, execCtx);
    return comm::AsyncEvent(eventHandle, comm::DmaEngine::SDMA);
}

PTO_INTERNAL comm::AsyncEvent TPrefetchL2RawSdmaImpl(__gm__ void *src, uint64_t bytes,
                                                     const comm::sdma::SdmaExecContext &execCtx)
{
    if (src == nullptr || bytes == 0) {
        return comm::AsyncEvent(0, comm::DmaEngine::SDMA);
    }

    const uint64_t eventHandle =
        comm::sdma::__sdma_cmo_prefetch(reinterpret_cast<__gm__ uint8_t *>(src), bytes, execCtx);
    return comm::AsyncEvent(eventHandle, comm::DmaEngine::SDMA);
}

// Build a transient AsyncSession for one-shot use:
//   * scratch tile lives in the *caller's* stack frame (so the call site must
//     allocate it; we cannot allocate it inside this helper because the caller
//     consumes the resulting `execCtx`/`session` after we return).
//   * channelGroupIdx defaults to get_block_idx() (handled inside BuildSdmaSession)
//   * syncId = 0, baseConfig = {kDefaultSdmaBlockBytes, 0, 1}
template <typename ScratchTile>
PTO_INTERNAL bool BuildTransientPrefetchL2Session(ScratchTile &scratchTile, __gm__ uint8_t *workspace,
                                                  comm::AsyncSession &session)
{
    if (workspace == nullptr) {
        session.valid = false;
        return false;
    }
    return comm::BuildAsyncSession(scratchTile, workspace, session);
}

} // namespace detail

// ============================================================================
// Public TPREFETCH_L2_IMPL — workspace-based (recommended).
//
// Each call builds a transient AsyncSession on the AICORE stack from the
// caller-supplied `workspace` (the same GM region used by all SDMA-backed
// instructions, normally allocated host-side by `SdmaWorkspaceManager::Init`).
// The session has the following fixed parameters:
//   * channelGroupIdx = get_block_idx()
//   * syncId          = 0
//   * baseConfig      = {kDefaultSdmaBlockBytes, 0, 1}
//
// The returned AsyncEvent's `handle` field IS the workspace base address, so
// `evt.Wait()` (no-arg) and `evt.Wait(workspace)` can both reconstruct an
// equivalent session for the wait/test phase without further plumbing.
// ============================================================================

template <typename GlobalData>
PTO_INTERNAL comm::AsyncEvent TPREFETCH_L2_IMPL(GlobalData &srcGlobalData, __gm__ uint8_t *workspace)
{
    detail::PrefetchL2ScratchTile scratchTile;
    // Fully-qualified call to TASSIGN_IMPL avoids the two-phase template-lookup
    // pitfall: at the point this template is *defined*, the public TASSIGN
    // wrapper in `pto/common/pto_instr.hpp` has not been declared yet (this
    // header is pulled in from `pto_instr_impl.hpp`). TASSIGN_IMPL however is
    // already declared (via `pto/npu/a2a3/TAssign.hpp` or its A5 counterpart),
    // so calling it directly works at definition time too.
    TASSIGN_IMPL(scratchTile, 0x0);

    comm::AsyncSession session;
    if (!detail::BuildTransientPrefetchL2Session(scratchTile, workspace, session)) {
        return comm::AsyncEvent(0, comm::DmaEngine::SDMA);
    }
    return detail::TPrefetchL2SdmaImpl(srcGlobalData, session.sdmaSession.execCtx);
}

PTO_INTERNAL comm::AsyncEvent TPREFETCH_L2_IMPL(__gm__ void *src, uint64_t bytes, __gm__ uint8_t *workspace)
{
    if (src == nullptr || bytes == 0 || workspace == nullptr) {
        return comm::AsyncEvent(0, comm::DmaEngine::SDMA);
    }

    detail::PrefetchL2ScratchTile scratchTile;
    TASSIGN_IMPL(scratchTile, 0x0);

    comm::AsyncSession session;
    if (!detail::BuildTransientPrefetchL2Session(scratchTile, workspace, session)) {
        return comm::AsyncEvent(0, comm::DmaEngine::SDMA);
    }
    return detail::TPrefetchL2RawSdmaImpl(src, bytes, session.sdmaSession.execCtx);
}

// ============================================================================
// Power-user TPREFETCH_L2_IMPL — session-based (for multi-call amortization).
//
// Build the AsyncSession once with `comm::BuildAsyncSession` and pass it to
// every TPREFETCH_L2 in the same kernel. Saves the ~hundreds-of-cycles session
// build cost when the same workspace is reused across many prefetch issues.
// ============================================================================

template <typename GlobalData>
PTO_INTERNAL comm::AsyncEvent TPREFETCH_L2_IMPL(GlobalData &srcGlobalData, const comm::AsyncSession &session)
{
    return detail::TPrefetchL2SdmaImpl(srcGlobalData, session.sdmaSession.execCtx);
}

PTO_INTERNAL comm::AsyncEvent TPREFETCH_L2_IMPL(__gm__ void *src, uint64_t bytes, const comm::AsyncSession &session)
{
    return detail::TPrefetchL2RawSdmaImpl(src, bytes, session.sdmaSession.execCtx);
}

template <typename GlobalData>
PTO_INTERNAL comm::AsyncEvent TPREFETCH_L2_IMPL(GlobalData &srcGlobalData, const comm::sdma::SdmaExecContext &execCtx)
{
    return detail::TPrefetchL2SdmaImpl(srcGlobalData, execCtx);
}

PTO_INTERNAL comm::AsyncEvent TPREFETCH_L2_IMPL(__gm__ void *src, uint64_t bytes,
                                                const comm::sdma::SdmaExecContext &execCtx)
{
    return detail::TPrefetchL2RawSdmaImpl(src, bytes, execCtx);
}

} // namespace pto

#endif // PTO_NPU_TPREFETCH_L2_HPP
