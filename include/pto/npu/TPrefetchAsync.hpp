/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_NPU_TPREFETCH_ASYNC_HPP
#define PTO_NPU_TPREFETCH_ASYNC_HPP

// TPREFETCH_ASYNC — L2 cache prefetch via SDMA CMO (opcode = 6).
//
// This instruction is logically a *memory access* instruction (it stages data
// from GM/HBM into the on-chip L2 cache so that subsequent TLOADs hit warm
// lines). It happens to *implement* itself by submitting an SDMA CMO SQE from
// the AI Core, which means it has to depend on the SDMA infrastructure that
// also backs TPUT_ASYNC / TGET_ASYNC. We keep the file here, alongside other
// per-arch compute/memory-access instruction headers, so its placement reflects
// what users see at the API surface (a memory-access instruction in `pto::`)
// rather than what the implementation reaches into (the comm SDMA stack).

#include "pto/common/type.hpp"
#include "pto/common/pto_tile.hpp"
#include "pto/comm/comm_types.hpp"
#include "pto/comm/async_common/async_types.hpp"
#include "pto/comm/async_common/async_event_impl.hpp"
#include "pto/npu/comm/async/sdma/sdma_async_intrin.hpp"
#include "pto/npu/comm/async/sdma/sdma_cmo_intrin.hpp"

namespace pto {

struct PrefetchAsyncContext {
    using ScratchTile = pto::Tile<pto::TileType::Vec, uint8_t, 1, comm::sdma::UB_ALIGN_SIZE>;

    __gm__ uint8_t *workspace{nullptr};
    ScratchTile scratchTile;
    comm::AsyncSession session;

    AICORE PrefetchAsyncContext() = default;
    AICORE explicit PrefetchAsyncContext(__gm__ uint8_t *workspace_) : workspace(workspace_)
    {}
};

namespace detail {

template <typename GlobalData>
PTO_INTERNAL bool TPrefetchAsyncIsFlatContiguous1D(GlobalData &globalData)
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
PTO_INTERNAL uint64_t TPrefetchAsyncGetTotalBytes(GlobalData &globalData)
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
PTO_INTERNAL comm::AsyncEvent TPrefetchAsyncSdmaImpl(GlobalData &srcGlobalData,
                                                     const comm::sdma::SdmaExecContext &execCtx)
{
    if (srcGlobalData.data() == nullptr) {
        return comm::AsyncEvent(0, comm::DmaEngine::SDMA);
    }

    if (!TPrefetchAsyncIsFlatContiguous1D(srcGlobalData)) {
        return comm::AsyncEvent(0, comm::DmaEngine::SDMA);
    }

    const uint64_t totalBytes = TPrefetchAsyncGetTotalBytes(srcGlobalData);
    if (totalBytes == 0) {
        return comm::AsyncEvent(0, comm::DmaEngine::SDMA);
    }

    const uint64_t eventHandle = comm::sdma::__sdma_cmo_prefetch(srcGlobalData.data(), totalBytes, execCtx);
    return comm::AsyncEvent(eventHandle, comm::DmaEngine::SDMA);
}

PTO_INTERNAL bool BuildPrefetchAsyncSession(PrefetchAsyncContext &ctx)
{
    if (ctx.workspace == nullptr) {
        ctx.session.valid = false;
        return false;
    }

    constexpr uint32_t syncId = 0;
    constexpr comm::sdma::SdmaBaseConfig baseConfig{comm::sdma::kDefaultSdmaBlockBytes, 0, 1};
    const uint32_t channelGroupIdx = static_cast<uint32_t>(get_block_idx());
    if (channelGroupIdx >= (comm::sdma::kSdmaMaxChannel / baseConfig.queue_num)) {
        ctx.session.valid = false;
        return false;
    }

    return comm::BuildAsyncSession<comm::DmaEngine::SDMA>(ctx.scratchTile, ctx.workspace, ctx.session, syncId,
                                                          baseConfig, channelGroupIdx);
}

} // namespace detail

template <typename GlobalData>
PTO_INTERNAL comm::AsyncEvent TPREFETCH_ASYNC_IMPL(GlobalData &srcGlobalData, PrefetchAsyncContext &ctx)
{
    // Fully-qualified call to TASSIGN_IMPL avoids the two-phase template-lookup
    // pitfall: at the point this template is *defined*, the public TASSIGN
    // wrapper in `pto/common/pto_instr.hpp` has not been declared yet (this
    // header is pulled in from `pto_instr_impl.hpp`). TASSIGN_IMPL however is
    // already declared (via `pto/npu/a2a3/TAssign.hpp` or its A5 counterpart),
    // so calling it directly works at definition time too.
    TASSIGN_IMPL(ctx.scratchTile, 0x0);

    if (!detail::BuildPrefetchAsyncSession(ctx)) {
        return comm::AsyncEvent(0, comm::DmaEngine::SDMA);
    }
    return detail::TPrefetchAsyncSdmaImpl(srcGlobalData, ctx.session.sdmaSession.execCtx);
}

} // namespace pto

#endif // PTO_NPU_TPREFETCH_ASYNC_HPP
