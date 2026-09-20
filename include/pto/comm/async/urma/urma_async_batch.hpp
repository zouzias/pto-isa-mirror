/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMM_ASYNC_URMA_BATCH_HPP
#define PTO_COMM_ASYNC_URMA_BATCH_HPP

#ifdef PTO_URMA_SUPPORTED

#include "pto/comm/async_common/TPutAsyncBatchCommonDetail.hpp"
#include "pto/comm/async/urma/urma_async_intrin.hpp"

namespace pto {
namespace comm {
namespace urma {
namespace detail {

AICORE inline bool UrmaBatchWqeCount(uint64_t messageLen, uint32_t& wqeCount)
{
    const uint64_t count =
        messageLen / kUrmaMaxWqeTransferBytes + (messageLen % kUrmaMaxWqeTransferBytes == 0U ? 0U : 1U);
    if (count == 0U || count > UINT32_MAX) {
        wqeCount = 0U;
        return false;
    }
    wqeCount = static_cast<uint32_t>(count);
    return true;
}

AICORE inline bool ResolveUrmaBatchJetty(
    const AsyncSession& session, uint32_t peer, uint32_t jettyIndex, uint32_t& physicalJetty)
{
    if (jettyIndex >= session.qpCount) {
        return false;
    }
    physicalJetty = session.qpIdxBase + jettyIndex;
    return ValidateUrmaSessionAt(session, peer, physicalJetty);
}

AICORE inline bool EnsureUrmaBatchCapacity(
    const AsyncSession& session, uint32_t peer, uint32_t physicalJetty, __gm__ UrmaWQCtx* wq, __gm__ UrmaCqCtx* cq,
    uint32_t startBbProducer, uint32_t startCqeExpected, uint32_t stagedWqeCount, uint32_t newWqeCount)
{
    if (stagedWqeCount > UINT32_MAX - newWqeCount) {
        return false;
    }
    const uint32_t totalWqeCount = stagedWqeCount + newWqeCount;
    if (totalWqeCount > wq->depth || totalWqeCount > cq->depth) {
        return false;
    }
    while (!UrmaRingHasRoom(
        wq, cq, startBbProducer + stagedWqeCount, startCqeExpected + stagedWqeCount, newWqeCount, newWqeCount)) {
        bool completed = false;
        if (!UrmaPollCqAt(session, peer, physicalJetty, startBbProducer, startCqeExpected, false, completed)) {
            return false;
        }
    }
    return true;
}

AICORE inline void ResolveUrmaBatchStart(
    __gm__ UrmaWQCtx* wq, const UrmaRuntimeContext& runtimeCtx, uint32_t& startBbProducer, uint32_t& startCqeExpected)
{
    if (runtimeCtx.batchStagedWqeCount == 0U) {
        startBbProducer = ld_dev(reinterpret_cast<__gm__ uint32_t*>(wq->headAddr), 0);
        startCqeExpected = wq->submittedWqeCount;
    } else {
        startBbProducer = runtimeCtx.batchStartBbProducer;
        startCqeExpected = runtimeCtx.batchStartCqeExpected;
    }
}

AICORE inline void FillUrmaBatchWqes(
    __gm__ UrmaWQCtx* wq, __gm__ UrmaMemInfo* remoteMem, __gm__ uint8_t* remoteAddr, __gm__ uint8_t* localAddr,
    uint64_t messageLen, uint32_t firstProducer, uint32_t newWqeCount)
{
    uint64_t offset = 0U;
    for (uint32_t index = 0U; index < newWqeCount; ++index) {
        const uint64_t remaining = messageLen - offset;
        const uint32_t chunk =
            static_cast<uint32_t>(remaining < kUrmaMaxWqeTransferBytes ? remaining : kUrmaMaxWqeTransferBytes);
        FillTransferWqeAt(
            wq, remoteMem, remoteAddr + offset, localAddr + offset, chunk, UrmaOpcode::WRITE, firstProducer + index);
        offset += chunk;
    }
}

AICORE inline void UrmaDeferAsyncPut(
    __gm__ uint8_t* remoteAddr, __gm__ uint8_t* localAddr, uint64_t messageLen, const AsyncSession& session,
    uint32_t peer, uint32_t jettyIndex)
{
    uint32_t physicalJetty = 0U;
    if (!ResolveUrmaBatchJetty(session, peer, jettyIndex, physicalJetty)) {
        ::pto::comm::detail::DiscardPendingAsyncPutBatch(session);
        PTO_ASSERT(false, "TPUT_ASYNC_DEFER URMA: invalid session, peer, or jetty.");
        return;
    }

    UrmaRuntimeContext& runtimeCtx = session.urmaRuntimeCtx;
    if (runtimeCtx.batchStagedWqeCount != 0U &&
        (runtimeCtx.batchPeer != peer || runtimeCtx.batchJettyIndex != jettyIndex)) {
        ::pto::comm::detail::DiscardPendingAsyncPutBatch(session);
        PTO_ASSERT(false, "TPUT_ASYNC_DEFER URMA: all operations in one batch must use the same peer and jetty.");
        return;
    }

    uint32_t newWqeCount = 0U;
    if (!UrmaBatchWqeCount(messageLen, newWqeCount)) {
        ::pto::comm::detail::DiscardPendingAsyncPutBatch(session);
        PTO_ASSERT(false, "TPUT_ASYNC_DEFER URMA: invalid transfer size or WQE count overflow.");
        return;
    }

    __gm__ UrmaInfo* info = reinterpret_cast<__gm__ UrmaInfo*>(session.contextGm);
    __gm__ UrmaWQCtx* wq = GetWqContextAt(session, peer, physicalJetty);
    __gm__ UrmaCqCtx* cq = GetCqContextAt(session, peer, physicalJetty);
    __gm__ UrmaMemInfo* remoteMem = GetRemoteMemInfo(info, peer, physicalJetty);

    uint32_t startBbProducer = 0U;
    uint32_t startCqeExpected = 0U;
    const uint32_t stagedWqeCount = runtimeCtx.batchStagedWqeCount;
    ResolveUrmaBatchStart(wq, runtimeCtx, startBbProducer, startCqeExpected);
    if (!EnsureUrmaBatchCapacity(
            session, peer, physicalJetty, wq, cq, startBbProducer, startCqeExpected, stagedWqeCount, newWqeCount)) {
        ::pto::comm::detail::DiscardPendingAsyncPutBatch(session);
        PTO_ASSERT(false, "TPUT_ASYNC_DEFER URMA: batch exceeds WQ/CQ capacity or completion polling failed.");
        return;
    }

    FillUrmaBatchWqes(wq, remoteMem, remoteAddr, localAddr, messageLen, startBbProducer + stagedWqeCount, newWqeCount);
    if (stagedWqeCount == 0U) {
        runtimeCtx.batchStartBbProducer = startBbProducer;
        runtimeCtx.batchStartCqeExpected = startCqeExpected;
        runtimeCtx.batchPeer = peer;
        runtimeCtx.batchJettyIndex = jettyIndex;
    }
    runtimeCtx.batchStagedWqeCount = stagedWqeCount + newWqeCount;
}

AICORE inline AsyncEvent UrmaSubmitAsyncPutBatch(const AsyncSession& session, uint32_t peer, uint32_t jettyIndex)
{
    uint32_t physicalJetty = 0U;
    if (!ResolveUrmaBatchJetty(session, peer, jettyIndex, physicalJetty)) {
        ::pto::comm::detail::DiscardPendingAsyncPutBatch(session);
        PTO_ASSERT(false, "SubmitAsyncPutBatch URMA: invalid session, peer, or jetty.");
        return {};
    }
    UrmaRuntimeContext& runtimeCtx = session.urmaRuntimeCtx;
    if (runtimeCtx.batchStagedWqeCount == 0U) {
        ::pto::comm::detail::DiscardPendingAsyncPutBatch(session);
        PTO_ASSERT(false, "SubmitAsyncPutBatch URMA: batch is empty.");
        return {};
    }
    if (runtimeCtx.batchPeer != peer || runtimeCtx.batchJettyIndex != jettyIndex) {
        ::pto::comm::detail::DiscardPendingAsyncPutBatch(session);
        PTO_ASSERT(false, "SubmitAsyncPutBatch URMA: peer and jetty must match the deferred batch.");
        return {};
    }

    __gm__ UrmaWQCtx* wq = GetWqContextAt(session, peer, physicalJetty);
    const uint32_t targetBb = runtimeCtx.batchStartBbProducer + runtimeCtx.batchStagedWqeCount;
    const uint32_t targetCqe = runtimeCtx.batchStartCqeExpected + runtimeCtx.batchStagedWqeCount;
    const uint64_t handle = EncodeHandle(peer, targetBb);
    if (handle == 0U) {
        ::pto::comm::detail::DiscardPendingAsyncPutBatch(session);
        PTO_ASSERT(false, "SubmitAsyncPutBatch URMA: failed to encode a valid completion event.");
        return {};
    }

    CommitPostedWqes(wq, targetBb, targetCqe);
    UrmaMultiPostResult result{};
    result.handle = handle;
    result.jettyBase = physicalJetty;
    result.jettyCount = 1U;
    result.targetBb[0] = targetBb;
    result.targetCqe[0] = targetCqe;
    runtimeCtx = {};
    return ::pto::comm::urma::MakeUrmaMultiJettyEvent(result);
}

} // namespace detail
} // namespace urma
} // namespace comm
} // namespace pto

#endif // PTO_URMA_SUPPORTED
#endif // PTO_COMM_ASYNC_URMA_BATCH_HPP
