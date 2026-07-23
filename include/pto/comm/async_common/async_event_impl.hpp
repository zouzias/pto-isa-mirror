/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMM_ASYNC_COMMON_ASYNC_EVENT_IMPL_HPP
#define PTO_COMM_ASYNC_COMMON_ASYNC_EVENT_IMPL_HPP

#include "pto/comm/comm_types.hpp"
#include "pto/comm/async_common/async_types.hpp"
#include "pto/comm/async/sdma/sdma_async_intrin.hpp"
#ifdef PTO_URMA_SUPPORTED
#include "pto/comm/async/urma/urma_async_intrin.hpp"
#endif
#ifdef PTO_RDMA_SUPPORTED
#include "pto/comm/async/rdma/rdma_async_intrin.hpp"
#endif

namespace pto {
namespace comm {

template <DmaEngine engine = DmaEngine::SDMA, typename ScratchTile>
PTO_INTERNAL bool BuildAsyncSession(
    ScratchTile& scratchTile, __gm__ uint8_t* workspace, AsyncSession& session, uint32_t syncId = 0,
    const sdma::SdmaBaseConfig& baseConfig = {sdma::kDefaultSdmaBlockBytes, 0, 1},
    uint32_t channelGroupIdx = sdma::kAutoChannelGroupIdx)
{
    session.engine = engine;
    if constexpr (engine == DmaEngine::SDMA) {
        session.valid =
            sdma::BuildSdmaSession(scratchTile, workspace, session.sdmaSession, syncId, baseConfig, channelGroupIdx);
        return session.valid;
    } else {
        static_assert(
            engine == DmaEngine::SDMA,
            "This overload is for SDMA; use the engine-specific BuildAsyncSession overload for URMA or RDMA");
        return false;
    }
}

#ifdef PTO_URMA_SUPPORTED
template <DmaEngine engine>
PTO_INTERNAL bool BuildAsyncSession(__gm__ uint8_t* workspace, uint32_t destRankId, AsyncSession& session)
{
    static_assert(engine == DmaEngine::URMA, "This overload is for URMA only");
    session.engine = engine;
    session.valid = urma::BuildUrmaSession(workspace, destRankId, session.urmaSession);
    return session.valid;
}
#endif

#ifdef PTO_RDMA_SUPPORTED
// RDMA session builder. The workspace header records the selected NIC backend;
// validation is delegated to the corresponding RDMA backend implementation.
//   scratchTile  — UB/Vec tile used as backend scratch (HNS_1825 requires >= 64B).
//   workspace    — device address of the RdmaInfo header.
//   destRankId   — remote PE this session communicates with.
//   myPe         — local rank id.
//   syncId       — pipe sync event id assigned to this session.
template <DmaEngine engine, typename ScratchTile>
PTO_INTERNAL bool BuildAsyncSession(
    ScratchTile& scratchTile, __gm__ uint8_t* workspace, uint32_t destRankId, uint32_t myPe, AsyncSession& session,
    uint32_t syncId = 0)
{
    static_assert(engine == DmaEngine::RDMA, "This overload is for RDMA only");
    session.engine = engine;
    session.valid = false;
    session.rdmaSession = {};

    sdma::TmpBuffer tmpBuf{};
    if (!sdma::detail::MakeTmpBufferFromTile(scratchTile, tmpBuf) ||
        !rdma::BuildSession(workspace, destRankId, myPe, tmpBuf, syncId, session.rdmaSession)) {
        return false;
    }
    session.valid = true;
    return true;
}
#endif

// ============================================================================
// AsyncEvent::Wait / Test — AsyncSession overloads (primary user API)
// ============================================================================

PTO_INTERNAL bool AsyncEvent::Wait(const AsyncSession& session) const
{
    if (handle == 0) {
        return true;
    }
    if (!session.valid || engine != session.engine) {
        return false;
    }
    switch (session.engine) {
        case DmaEngine::SDMA:
            return sdma::detail::SdmaWaitEvent(handle, session.sdmaSession);
#ifdef PTO_URMA_SUPPORTED
        case DmaEngine::URMA:
            return urma::detail::UrmaWaitEvent(handle, session.urmaSession.eventCtx);
#endif
#ifdef PTO_RDMA_SUPPORTED
        case DmaEngine::RDMA:
            if (!session.rdmaSession.valid) {
                return false;
            }
            return rdma::WaitEvent(handle, session.rdmaSession.eventCtx);
#endif
        default:
            return false;
    }
}

PTO_INTERNAL bool AsyncEvent::Test(const AsyncSession& session) const
{
    if (handle == 0) {
        return true;
    }
    if (!session.valid || engine != session.engine) {
        return false;
    }
    switch (session.engine) {
        case DmaEngine::SDMA:
            return sdma::detail::SdmaTestEvent(handle, session.sdmaSession);
#ifdef PTO_URMA_SUPPORTED
        case DmaEngine::URMA:
            return urma::detail::UrmaTestEvent(handle, session.urmaSession.eventCtx);
#endif
#ifdef PTO_RDMA_SUPPORTED
        case DmaEngine::RDMA:
            if (!session.rdmaSession.valid) {
                return false;
            }
            return rdma::TestEvent(handle, session.rdmaSession.eventCtx);
#endif
        default:
            return false;
    }
}

} // namespace comm
} // namespace pto

#endif // PTO_COMM_ASYNC_COMMON_ASYNC_EVENT_IMPL_HPP
