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
        session.valid = sdma::BuildSdmaSession(scratchTile, workspace, session, syncId, baseConfig, channelGroupIdx);
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
PTO_INTERNAL bool BuildAsyncSession(__gm__ uint8_t* workspace, AsyncSession& session)
{
    static_assert(engine == DmaEngine::URMA, "This overload is for URMA only");
    session = AsyncSession{};
    session.engine = engine;
    session.contextGm = workspace;
    session.destRankId = 0;
    session.qpIdxBase = 0;
    session.qpCount = 1;
    if (workspace == nullptr) {
        return false;
    }
    __gm__ urma::UrmaInfo* info = reinterpret_cast<__gm__ urma::UrmaInfo*>(workspace);
    if (info->layout != urma::UrmaLayout::SHARED_POOL) {
        session.valid = true;
        return session.valid;
    }
    const uint32_t perCore = info->jettiesPerCore;
    if (perCore == 0U || perCore > kUrmaMaxJettiesPerCore) {
        return false;
    }
    session.qpIdxBase = static_cast<uint32_t>(get_block_idx()) * perCore;
    session.qpCount = perCore;
    session.valid = (session.qpIdxBase + perCore <= urma::detail::UrmaJettyIdxBound(info));
    return session.valid;
}

template <DmaEngine engine>
PTO_INTERNAL bool BuildAsyncSession(__gm__ uint8_t* workspace, uint32_t destRankId, AsyncSession& session)
{
    static_assert(engine == DmaEngine::URMA, "This overload is for URMA only");
    if (!BuildAsyncSession<engine>(workspace, session)) {
        return false;
    }
    session.destRankId = destRankId;
    return session.valid;
}
#endif

#ifdef PTO_RDMA_SUPPORTED
// Peer-independent RDMA session builder. The explicit-peer TPUT_ASYNC and
// TGET_ASYNC overloads can reuse this session for multiple remote ranks.
template <DmaEngine engine, typename ScratchTile>
PTO_INTERNAL bool BuildAsyncSession(
    ScratchTile& scratchTile, __gm__ uint8_t* workspace, uint32_t myPe, AsyncSession& session, uint32_t syncId = 0)
{
    static_assert(engine == DmaEngine::RDMA, "This overload is for RDMA only");
    rdma::RdmaTmpBuffer tmpBuf{};
    if (!rdma::detail::MakeTmpBufferFromTile(scratchTile, tmpBuf)) {
        session = AsyncSession{};
        return false;
    }
    return rdma::BuildSession(workspace, myPe, tmpBuf, syncId, session);
}

// Peer-bound RDMA builder retained for callers that use the original async
// overload without an explicit peer.
template <DmaEngine engine, typename ScratchTile>
PTO_INTERNAL bool BuildAsyncSession(
    ScratchTile& scratchTile, __gm__ uint8_t* workspace, uint32_t destRankId, uint32_t myPe, AsyncSession& session,
    uint32_t syncId = 0)
{
    static_assert(engine == DmaEngine::RDMA, "This overload is for RDMA only");
    rdma::RdmaTmpBuffer tmpBuf{};
    if (!rdma::detail::MakeTmpBufferFromTile(scratchTile, tmpBuf)) {
        session = AsyncSession{};
        return false;
    }
    return rdma::BuildSession(workspace, destRankId, myPe, tmpBuf, syncId, session);
}
#endif

// ============================================================================
// AsyncEvent::Wait / Test — AsyncSession overloads (primary user API)
// ============================================================================

#ifdef PTO_URMA_SUPPORTED
// Shared by Wait/Test so the two cannot drift apart on which jetties they drain.
PTO_INTERNAL bool UrmaEventComplete(const AsyncEvent& event, const AsyncSession& session, bool blocking)
{
    if (event.urmaJettyCount == 0U) {
        return urma::detail::UrmaWaitEvent(event.handle, event.urmaTargetCqe, session, blocking);
    }
    uint32_t peer = 0U;
    uint32_t unusedBb = 0U;
    urma::detail::DecodeHandle(event.handle, peer, unusedBb);
    return urma::detail::UrmaWaitEventMultiJetty(
        peer, event.urmaJettyBase, event.urmaJettyCount, event.urmaTargetBbPerJetty, event.urmaTargetCqePerJetty,
        blocking, session);
}
#endif

PTO_INTERNAL bool AsyncEvent::Wait(const AsyncSession& session) const
{
    if (handle == 0) {
        return true;
    }
    switch (session.engine) {
        case DmaEngine::SDMA:
            return sdma::detail::SdmaWaitEvent(handle, session);
#ifdef PTO_URMA_SUPPORTED
        case DmaEngine::URMA:
            return UrmaEventComplete(*this, session, true);
#endif
#ifdef PTO_RDMA_SUPPORTED
        case DmaEngine::RDMA:
            return rdma::WaitEvent(handle, session);
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
    switch (session.engine) {
        case DmaEngine::SDMA:
            return sdma::detail::SdmaTestEvent(handle, session);
#ifdef PTO_URMA_SUPPORTED
        case DmaEngine::URMA:
            return UrmaEventComplete(*this, session, false);
#endif
#ifdef PTO_RDMA_SUPPORTED
        case DmaEngine::RDMA:
            return rdma::TestEvent(handle, session);
#endif
        default:
            return false;
    }
}

} // namespace comm
} // namespace pto

#endif // PTO_COMM_ASYNC_COMMON_ASYNC_EVENT_IMPL_HPP
