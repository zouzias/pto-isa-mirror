/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMM_ASYNC_URMA_INTRIN_HPP
#define PTO_COMM_ASYNC_URMA_INTRIN_HPP

#ifdef PTO_URMA_SUPPORTED

#include "pto/common/debug.h"
#include "pto/comm/async_common/async_types.hpp"
#include "pto/comm/async/urma/urma_types.hpp"

namespace pto {
namespace comm {
namespace urma {

AICORE inline void DcciCachelines(__gm__ uint8_t* addr, uint64_t length)
{
    if (length == 0U) {
        return;
    }
    __gm__ uint8_t* start =
        reinterpret_cast<__gm__ uint8_t*>(reinterpret_cast<uint64_t>(addr) / kCacheLineSize * kCacheLineSize);
    __gm__ uint8_t* end = reinterpret_cast<__gm__ uint8_t*>(
        (reinterpret_cast<uint64_t>(addr) + length - 1U) / kCacheLineSize * kCacheLineSize);
    for (uint64_t offset = 0; offset <= static_cast<uint64_t>(end - start); offset += kCacheLineSize) {
        __asm__ __volatile__("");
        dcci(reinterpret_cast<__gm__ void*>(start + offset), cache_line_t::SINGLE_CACHE_LINE);
        __asm__ __volatile__("");
    }
}

namespace detail {

struct UrmaPostResult {
    uint64_t handle;
    uint32_t targetCqe;
};

struct UrmaMultiPostResult {
    uint64_t handle;
    uint32_t jettyBase;
    uint32_t jettyCount;
    uint32_t targetBb[kUrmaMaxJettiesPerCore];
    uint32_t targetCqe[kUrmaMaxJettiesPerCore];
};

// One jetty's slice completion targets; ok stays false on a drain failure.
struct UrmaSlicePostResult {
    bool ok;
    uint32_t targetBb;
    uint32_t targetCqe;
};

AICORE inline uint64_t EncodeHandle(uint32_t peer, uint32_t targetBb)
{
    return (static_cast<uint64_t>(peer) << 32U) | targetBb;
}

AICORE inline void DecodeHandle(uint64_t handle, uint32_t& peer, uint32_t& targetBb)
{
    peer = static_cast<uint32_t>(handle >> 32U);
    targetBb = static_cast<uint32_t>(handle);
}

AICORE inline bool SequenceReached(uint32_t current, uint32_t target)
{
    return static_cast<int32_t>(current - target) >= 0;
}

AICORE inline void UrmaPostSendUpdateInfo(uint32_t head, __gm__ UrmaWQCtx* wq)
{
    st_dev(head, reinterpret_cast<__gm__ uint32_t*>(wq->dbAddr), 0);
    st_dev(head, reinterpret_cast<__gm__ uint32_t*>(wq->headAddr), 0);
}

AICORE inline uint32_t UrmaCtxRow(__gm__ UrmaInfo* info, uint32_t peer, uint32_t jettyIdx)
{
    const uint32_t peerStride = (info->layout == UrmaLayout::SHARED_POOL) ? 0U : info->rowsPerPeer;
    return peer * peerStride + jettyIdx;
}

AICORE inline uint32_t UrmaTargetRow(__gm__ UrmaInfo* info, uint32_t peer, uint32_t jettyIdx)
{
    const uint32_t jettyStride = (info->layout == UrmaLayout::SHARED_POOL) ? info->rankCount : 0U;
    return jettyIdx * jettyStride + peer;
}

AICORE inline uint32_t UrmaJettyIdxBound(__gm__ UrmaInfo* info)
{
    return (info->layout == UrmaLayout::SHARED_POOL) ? info->jettyCount : info->rowsPerPeer;
}

AICORE inline __gm__ UrmaWQCtx* GetWqContextAt(const AsyncSession& session, uint32_t peer, uint32_t jettyIdx)
{
    __gm__ UrmaInfo* info = reinterpret_cast<__gm__ UrmaInfo*>(session.contextGm);
    return reinterpret_cast<__gm__ UrmaWQCtx*>(info->sqPtr + UrmaCtxRow(info, peer, jettyIdx) * sizeof(UrmaWQCtx));
}

AICORE inline __gm__ UrmaCqCtx* GetCqContextAt(const AsyncSession& session, uint32_t peer, uint32_t jettyIdx)
{
    __gm__ UrmaInfo* info = reinterpret_cast<__gm__ UrmaInfo*>(session.contextGm);
    return reinterpret_cast<__gm__ UrmaCqCtx*>(info->scqPtr + UrmaCtxRow(info, peer, jettyIdx) * sizeof(UrmaCqCtx));
}

AICORE inline __gm__ UrmaWQCtx* GetWqContext(const AsyncSession& session, uint32_t peer)
{
    return GetWqContextAt(session, peer, session.qpIdxBase);
}

AICORE inline __gm__ UrmaCqCtx* GetCqContext(const AsyncSession& session, uint32_t peer)
{
    return GetCqContextAt(session, peer, session.qpIdxBase);
}

AICORE inline __gm__ UrmaMemInfo* GetRemoteMemInfo(__gm__ UrmaInfo* info, uint32_t peer, uint32_t jettyIdx)
{
    return reinterpret_cast<__gm__ UrmaMemInfo*>(
        info->memPtr + UrmaTargetRow(info, peer, jettyIdx) * sizeof(UrmaMemInfo));
}

AICORE inline bool ValidateUrmaSessionAt(const AsyncSession& session, uint32_t peer, uint32_t jettyIdx)
{
    if (!session.valid || session.engine != DmaEngine::URMA || session.contextGm == nullptr) {
        return false;
    }
    __gm__ UrmaInfo* info = reinterpret_cast<__gm__ UrmaInfo*>(session.contextGm);
    if (peer >= info->rankCount) {
        return false;
    }
    if (jettyIdx >= UrmaJettyIdxBound(info)) {
        return false;
    }
    __gm__ UrmaWQCtx* wq = GetWqContextAt(session, peer, jettyIdx);
    __gm__ UrmaCqCtx* cq = GetCqContextAt(session, peer, jettyIdx);

    const uint32_t sqDepth = wq->depth;
    const uint32_t cqDepth = cq->depth;
    return sqDepth != 0U && (sqDepth & (sqDepth - 1U)) == 0U && cqDepth != 0U && (cqDepth & (cqDepth - 1U)) == 0U;
}

AICORE inline bool ValidateUrmaSession(const AsyncSession& session, uint32_t peer)
{
    if (session.qpCount == 0U || session.qpCount > kUrmaMaxJettiesPerCore) {
        return false;
    }
    for (uint32_t j = 0U; j < session.qpCount; ++j) {
        if (!ValidateUrmaSessionAt(session, peer, session.qpIdxBase + j)) {
            return false;
        }
    }
    return true;
}

AICORE inline __gm__ UrmaNotifyResourceRegion* GetUrmaNotifyResourceRegion(const AsyncSession& session, uint32_t peer)
{
    __gm__ UrmaInfo* info = reinterpret_cast<__gm__ UrmaInfo*>(session.contextGm);
    __gm__ UrmaNotifyResourceRegion* regions = reinterpret_cast<__gm__ UrmaNotifyResourceRegion*>(info->notifyPoolPtr);
    return regions + UrmaTargetRow(info, peer, session.qpIdxBase);
}

AICORE inline void PublishCqAndWqTails(
    uint32_t targetBb, uint32_t targetCqe, __gm__ UrmaCqCtx* cq, __gm__ UrmaWQCtx* wq)
{
    st_dev(targetCqe, reinterpret_cast<__gm__ uint32_t*>(cq->tailAddr), 0);
    st_dev(targetCqe & 0xFFFFFFU, reinterpret_cast<__gm__ uint32_t*>(cq->dbAddr), 0);
    st_dev(targetBb, reinterpret_cast<__gm__ uint32_t*>(wq->tailAddr), 0);
}

AICORE inline __gm__ UrmaJfcCqeCtx* GetUrmaCqe(__gm__ UrmaCqCtx* cq, uint32_t sequence, uint32_t cqeSize)
{
    return reinterpret_cast<__gm__ UrmaJfcCqeCtx*>(cq->bufAddr + cqeSize * (sequence & (cq->depth - 1U)));
}

AICORE inline bool IsUrmaCqePrefixReady(__gm__ UrmaCqCtx* cq, uint32_t begin, uint32_t end, uint32_t cqeSize)
{
    for (uint32_t sequence = begin; sequence != end; ++sequence) {
        __gm__ UrmaJfcCqeCtx* cqe = GetUrmaCqe(cq, sequence, cqeSize);
        const bool validOwner = (sequence / cq->depth) & 1U;
        DcciCachelines(reinterpret_cast<__gm__ uint8_t*>(cqe), sizeof(UrmaJfcCqeCtx));
        if ((validOwner ^ cqe->owner) == 0U) {
            return false;
        }
    }
    return true;
}

AICORE inline bool WaitForUrmaCqe(__gm__ UrmaCqCtx* cq, uint32_t sequence, uint32_t cqeSize, __gm__ UrmaJfcCqeCtx*& cqe)
{
    cqe = GetUrmaCqe(cq, sequence, cqeSize);
    const bool validOwner = (sequence / cq->depth) & 1U;
    uint32_t polls = 0U;
    DcciCachelines(reinterpret_cast<__gm__ uint8_t*>(cqe), sizeof(UrmaJfcCqeCtx));
    while ((validOwner ^ cqe->owner) == 0U) {
        if (++polls >= kUrmaMaxPollTimes) {
            return false;
        }
        DcciCachelines(reinterpret_cast<__gm__ uint8_t*>(cqe), sizeof(UrmaJfcCqeCtx));
    }
    return true;
}

AICORE inline bool RetireUrmaCqePrefix(__gm__ UrmaCqCtx* cq, uint32_t beginCqe, uint32_t targetCqe, uint32_t cqeSize)
{
    for (uint32_t sequence = beginCqe; sequence != targetCqe; ++sequence) {
        __gm__ UrmaJfcCqeCtx* cqe = nullptr;
        if (!WaitForUrmaCqe(cq, sequence, cqeSize, cqe)) {
            return false;
        }
        constexpr uint32_t kStatusShift = 8U;
        const uint32_t cqeError =
            (static_cast<uint32_t>(cqe->status) << kStatusShift) | static_cast<uint32_t>(cqe->substatus);
        if (cqeError != 0U) {
            return false;
        }
    }
    return true;
}

AICORE inline bool UrmaPollCqAt(
    const AsyncSession& session, uint32_t peer, uint32_t jettyIdx, uint32_t targetBb, uint32_t targetCqe, bool blocking,
    bool& completed)
{
    completed = false;
    if (!ValidateUrmaSessionAt(session, peer, jettyIdx)) {
        return false;
    }
    __gm__ UrmaCqCtx* cq = GetCqContextAt(session, peer, jettyIdx);
    __gm__ UrmaWQCtx* wq = GetWqContextAt(session, peer, jettyIdx);
    const uint32_t completedCqe = ld_dev(reinterpret_cast<__gm__ uint32_t*>(cq->tailAddr), 0);
    if (SequenceReached(completedCqe, targetCqe)) {
        const uint32_t completedBb = ld_dev(reinterpret_cast<__gm__ uint32_t*>(wq->tailAddr), 0);
        completed = SequenceReached(completedBb, targetBb);
        return true;
    }
    const uint32_t cqeSize = 1U << cq->cqeShiftSize;
    if (!blocking && !IsUrmaCqePrefixReady(cq, completedCqe, targetCqe, cqeSize)) {
        return true;
    }
    if (!RetireUrmaCqePrefix(cq, completedCqe, targetCqe, cqeSize)) {
        return false;
    }
    PublishCqAndWqTails(targetBb, targetCqe, cq, wq);
    completed = true;
    return true;
}

AICORE inline bool UrmaPollCq(
    const AsyncSession& session, uint32_t peer, uint32_t targetBb, uint32_t targetCqe, bool blocking, bool& completed)
{
    return UrmaPollCqAt(session, peer, session.qpIdxBase, targetBb, targetCqe, blocking, completed);
}

#include "pto/comm/async/urma/urma_async_post.hpp"

AICORE inline bool UrmaWaitEvent(uint64_t handle, uint32_t targetCqe, const AsyncSession& session, bool blocking)
{
    uint32_t peer = 0U;
    uint32_t targetBb = 0U;
    DecodeHandle(handle, peer, targetBb);
    if (!ValidateUrmaSession(session, peer)) {
        return false;
    }
    bool completed = false;
    return UrmaPollCq(session, peer, targetBb, targetCqe, blocking, completed) && completed;
}

AICORE inline bool UrmaWaitEventMultiJetty(
    uint32_t peer, uint32_t jettyBase, uint32_t jettyCount, const uint32_t* targetBb, const uint32_t* targetCqe,
    bool blocking, const AsyncSession& session)
{
    for (uint32_t j = 0U; j < jettyCount; ++j) {
        bool completed = false;
        if (!UrmaPollCqAt(session, peer, jettyBase + j, targetBb[j], targetCqe[j], blocking, completed) || !completed) {
            return false;
        }
    }
    return true;
}

} // namespace detail

AICORE inline AsyncEvent MakeUrmaMultiJettyEvent(const detail::UrmaMultiPostResult& result)
{
    AsyncEvent event(result.handle, DmaEngine::URMA);
    event.urmaJettyBase = static_cast<uint16_t>(result.jettyBase);
    event.urmaJettyCount = static_cast<uint16_t>(result.jettyCount);
    for (uint32_t j = 0U; j < result.jettyCount; ++j) {
        event.urmaTargetBbPerJetty[j] = result.targetBb[j];
        event.urmaTargetCqePerJetty[j] = result.targetCqe[j];
    }
    return event;
}

AICORE inline detail::UrmaMultiPostResult __urma_put_async(
    __gm__ uint8_t* dst, __gm__ uint8_t* src, uint64_t transferSize, const AsyncSession& session, uint32_t peer)
{
    return detail::UrmaPostSend(dst, src, transferSize, UrmaOpcode::WRITE, session, peer);
}

AICORE inline detail::UrmaMultiPostResult __urma_get_async(
    __gm__ uint8_t* dst, __gm__ uint8_t* src, uint64_t transferSize, const AsyncSession& session, uint32_t peer)
{
    return detail::UrmaPostSend(src, dst, transferSize, UrmaOpcode::READ, session, peer);
}

AICORE inline detail::UrmaMultiPostResult __urma_put_async(
    __gm__ uint8_t* dst, __gm__ uint8_t* src, uint64_t transferSize, const AsyncSession& session)
{
    return __urma_put_async(dst, src, transferSize, session, session.destRankId);
}

AICORE inline detail::UrmaMultiPostResult __urma_get_async(
    __gm__ uint8_t* dst, __gm__ uint8_t* src, uint64_t transferSize, const AsyncSession& session)
{
    return __urma_get_async(dst, src, transferSize, session, session.destRankId);
}

AICORE inline detail::UrmaPostResult __urma_put_async_notify(
    __gm__ uint8_t* dst, __gm__ uint8_t* src, uint64_t transferSize, __gm__ int32_t* signal, int32_t signalValue,
    NotifyOp notifyOp, const AsyncSession& session, uint32_t peer)
{
    return detail::UrmaPostNotify(dst, src, transferSize, signal, signalValue, notifyOp, session, peer);
}

// Symmetric MR base VA of a peer, or 0 if the peer is out of range — 0 faults on
// first use instead of returning an address read from past the table. addr does
// not vary across jetties, so any jetty's row will do.
AICORE inline uint64_t UrmaPeerMrBaseAddr(__gm__ uint8_t* urmaWorkspace, uint32_t peerRank)
{
    __gm__ UrmaInfo* info = reinterpret_cast<__gm__ UrmaInfo*>(urmaWorkspace);
    if (peerRank >= info->rankCount) {
        return 0U;
    }
    return detail::GetRemoteMemInfo(info, peerRank, 0U)->addr;
}

} // namespace urma
} // namespace comm
} // namespace pto

#endif // PTO_URMA_SUPPORTED
#endif // PTO_COMM_ASYNC_URMA_INTRIN_HPP
