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

// ============================================================================
// DcciCachelines — flush multiple cache lines covering [addr, addr+length)
// ============================================================================
AICORE inline void DcciCachelines(__gm__ uint8_t* addr, uint64_t length)
{
    if (length == 0U) {
        return;
    }
    __gm__ uint8_t* start = (__gm__ uint8_t*)((uint64_t)addr / kCacheLineSize * kCacheLineSize);
    __gm__ uint8_t* end = (__gm__ uint8_t*)(((uint64_t)addr + length - 1U) / kCacheLineSize * kCacheLineSize);
    for (uint64_t i = 0; i <= static_cast<uint64_t>(end - start); i += kCacheLineSize) {
        __asm__ __volatile__("");
        dcci((__gm__ void*)(start + i), cache_line_t::SINGLE_CACHE_LINE);
        __asm__ __volatile__("");
    }
}

namespace detail {

AICORE inline uint64_t EncodeHandle(uint32_t destRankId, uint32_t sequence);

// ============================================================================
// UrmaPollCqUpdateInfo — update CQ/WQ tail and ring CQ doorbell after polling (URMA)
// ============================================================================
AICORE inline void UrmaPollCqUpdateInfo(uint32_t curTail, __gm__ UrmaCqCtx* cqCtxEntry, __gm__ UrmaWQCtx* wqCtxEntry)
{
    __gm__ uint32_t* dbAddr = (__gm__ uint32_t*)cqCtxEntry->dbAddr;
    st_dev(static_cast<uint32_t>(curTail & 0xFFFFFF), dbAddr, 0);

    __gm__ uint32_t* wqTailAddr = (__gm__ uint32_t*)wqCtxEntry->tailAddr;
    st_dev(curTail, wqTailAddr, 0);
}

// ============================================================================
// UrmaPollCq — poll completion queue until idx entries are consumed
// Returns 0 on success, non-zero on error/timeout.
// ============================================================================
AICORE inline uint32_t UrmaPollCq(__gm__ uint8_t* contextGm, uint32_t destRankId, uint32_t qpIdx, uint32_t idx)
{
    if (idx == 0) {
        return 0;
    }

    __gm__ UrmaInfo* urmaInfo = (__gm__ UrmaInfo*)contextGm;
    uint32_t qpNum = urmaInfo->qpNum;

    __gm__ UrmaCqCtx* cqCtxEntry =
        (__gm__ UrmaCqCtx*)(urmaInfo->scqPtr + (destRankId * qpNum + qpIdx) * sizeof(UrmaCqCtx));
    uint64_t cqBaseAddr = cqCtxEntry->bufAddr;
    uint32_t cqeSize = 1U << cqCtxEntry->cqeShiftSize;
    uint32_t depth = cqCtxEntry->depth;

    uint32_t curTail = ld_dev((__gm__ uint32_t*)cqCtxEntry->tailAddr, 0);

    while (curTail != idx) {
        __gm__ UrmaJfcCqeCtx* cqeAddr = (__gm__ UrmaJfcCqeCtx*)(cqBaseAddr + cqeSize * (curTail & (depth - 1)));
        bool validOwner = (curTail / depth) & 1;

        uint32_t times = 0;
        while ((validOwner ^ cqeAddr->owner) == 0 && times < kUrmaMaxPollTimes) {
            DcciCachelines((__gm__ uint8_t*)cqeAddr, sizeof(UrmaJfcCqeCtx));
            times++;
        }
        if (times >= kUrmaMaxPollTimes) {
            trap();
            return 0xFF;
        }

        curTail++;

        uint8_t status = cqeAddr->status & 0xFF;
        uint8_t subStatus = cqeAddr->substatus & 0xFF;
        constexpr uint8_t kStatusShift = 8;
        if (status != 0 || subStatus != 0) {
            return (status << kStatusShift) | subStatus;
        }
    }

    st_dev(curTail, (__gm__ uint32_t*)cqCtxEntry->tailAddr, 0);

    __gm__ UrmaWQCtx* wqCtxEntry =
        (__gm__ UrmaWQCtx*)(urmaInfo->sqPtr + (destRankId * qpNum + qpIdx) * sizeof(UrmaWQCtx));
    UrmaPollCqUpdateInfo(curTail, cqCtxEntry, wqCtxEntry);

    return 0;
}

// ============================================================================
// UrmaPostSendUpdateInfo — ring SQ doorbell and update head
// ============================================================================
AICORE inline void UrmaPostSendUpdateInfo(uint32_t curHead, __gm__ UrmaWQCtx* qpCtxEntry)
{
    __gm__ uint32_t* doorBellAddr = reinterpret_cast<__gm__ uint32_t*>(qpCtxEntry->dbAddr);
    st_dev(curHead, doorBellAddr, 0);
    st_dev(curHead, (__gm__ uint32_t*)qpCtxEntry->headAddr, 0);
}

// ============================================================================
// FillSqeCtx — populate SQE fields from remote memory info and opcode
// ============================================================================
AICORE inline void FillSqeCtx(
    __gm__ UrmaSqeCtx* sqeCtx, __gm__ UrmaMemInfo* remoteMemInfo, __gm__ uint8_t* remoteAddr, UrmaOpcode opcode,
    uint32_t curHead, uint32_t depth)
{
    sqeCtx->sqeBbIdx = static_cast<uint16_t>(curHead % depth);
    sqeCtx->opcode = static_cast<uint32_t>(opcode);
    sqeCtx->flag = kUrmaWqeFlagCqe;
    sqeCtx->rsv0 = 0;
    sqeCtx->nf = 0;
    sqeCtx->tokenEn = remoteMemInfo->tokenValueValid ? 1U : 0U;
    sqeCtx->rmtJettyType = remoteMemInfo->rmtJettyType;
    sqeCtx->owner = (curHead & depth) == 0U ? 1U : 0U;
    sqeCtx->targetHint = remoteMemInfo->targetHint;
    sqeCtx->rsv1 = 0;
    sqeCtx->inlineMsgLen = 0;
    sqeCtx->tpId = remoteMemInfo->tpn;
    sqeCtx->sgeNum = 1;
    sqeCtx->rmtJettyOrSegId = remoteMemInfo->tid;
    sqeCtx->rsv2 = 0;
    sqeCtx->rmtTokenValue = remoteMemInfo->rmtTokenValue;
    sqeCtx->udfType = 0;
    sqeCtx->reduceDataType = 0;
    sqeCtx->reduceOpcode = 0;
    sqeCtx->rsv3 = 0;

    const uint64_t remoteAddrValue = reinterpret_cast<uint64_t>(remoteAddr);
    __gm__ uint8_t* sqeBytes = reinterpret_cast<__gm__ uint8_t*>(sqeCtx);
    __gm__ uint64_t* rmtEid = reinterpret_cast<__gm__ uint64_t*>(remoteMemInfo->eidAddr);
    *reinterpret_cast<__gm__ uint64_t*>(sqeBytes + kUrmaSqeRmtEidLOffset) = rmtEid[0];
    *reinterpret_cast<__gm__ uint64_t*>(sqeBytes + kUrmaSqeRmtEidHOffset) = rmtEid[1];
    *reinterpret_cast<__gm__ uint32_t*>(sqeBytes + kUrmaSqeRmtAddrLOffset) =
        static_cast<uint32_t>(remoteAddrValue & 0xFFFFFFFFU);
    *reinterpret_cast<__gm__ uint32_t*>(sqeBytes + kUrmaSqeRmtAddrHOffset) =
        static_cast<uint32_t>((remoteAddrValue >> 32) & 0xFFFFFFFFU);
}

// ============================================================================
// UrmaPostSend — prepare WQE+SGE, flush cache, ring doorbell
// Returns curHead after post (used to encode AsyncEvent handle).
// ============================================================================
AICORE inline uint32_t UrmaPostSend(
    __gm__ uint8_t* contextGm, __gm__ uint8_t* remoteAddr, __gm__ uint8_t* localAddr, uint32_t destRankId,
    uint32_t qpIdx, UrmaOpcode opcode, uint64_t messageLen)
{
    PTO_ASSERT(
        messageLen > 0 && messageLen <= kUrmaMaxWqeTransferBytes,
        "UrmaPostSend: messageLen must be in (0, 256MB]; larger transfers require caller-side splitting");
    __gm__ UrmaInfo* urmaInfo = (__gm__ UrmaInfo*)contextGm;
    PTO_ASSERT(destRankId < urmaInfo->rankCount, "UrmaPostSend: destRankId out of range");
    uint32_t qpNum = urmaInfo->qpNum;

    __gm__ UrmaWQCtx* qpCtxEntry =
        (__gm__ UrmaWQCtx*)(urmaInfo->sqPtr + (destRankId * qpNum + qpIdx) * sizeof(UrmaWQCtx));
    uint32_t wqeSize = 1U << qpCtxEntry->wqeShiftSize;
    uint32_t depth = qpCtxEntry->depth;

    uint32_t curHead = ld_dev((__gm__ uint32_t*)qpCtxEntry->headAddr, 0);
    uint32_t curTail = ld_dev((__gm__ uint32_t*)qpCtxEntry->tailAddr, 0);

    if ((curHead + kUrmaPollCqThreshold) % depth == curTail % depth) {
        (void)UrmaPollCq(contextGm, destRankId, qpIdx, curTail + kNumCqePerPollCq);
    }

    __gm__ UrmaMemInfo* remoteMemInfo = (__gm__ UrmaMemInfo*)(urmaInfo->memPtr + sizeof(UrmaMemInfo) * destRankId);

    __gm__ uint8_t* wqeAddr = (__gm__ uint8_t*)(qpCtxEntry->bufAddr + wqeSize * (curHead % depth));
    FillSqeCtx((__gm__ UrmaSqeCtx*)wqeAddr, remoteMemInfo, remoteAddr, opcode, curHead, depth);

    __gm__ UrmaSgeCtx* sgeCtx = (__gm__ UrmaSgeCtx*)(wqeAddr + kUrmaSqeSizeBytes);
    sgeCtx->len = static_cast<uint32_t>(messageLen);
    sgeCtx->tokenId = urmaInfo->localTokenId;
    sgeCtx->va = reinterpret_cast<uint64_t>(localAddr);

    pipe_barrier(PIPE_ALL);
    DcciCachelines(wqeAddr, kUrmaSqeSizeBytes + kUrmaSgeSizeBytes);
    pipe_barrier(PIPE_ALL);
    curHead++;
    UrmaPostSendUpdateInfo(curHead, qpCtxEntry);

    return curHead;
}

struct UrmaRetireResult {
    bool completed;
    bool success;
};

AICORE inline bool SequenceReached(uint32_t current, uint32_t target)
{
    return static_cast<int32_t>(current - target) >= 0;
}

AICORE inline bool IsRangeInside(uint64_t address, uint64_t length, uint64_t base, uint64_t size)
{
    return length != 0U && address >= base && length <= size && address - base <= size - length;
}

AICORE inline __gm__ UrmaWQCtx* GetTrackedWq(__gm__ UrmaInfo* info, uint32_t peer, uint32_t qpIdx)
{
    return reinterpret_cast<__gm__ UrmaWQCtx*>(
        info->sqPtr + (static_cast<uint64_t>(peer) * info->qpNum + qpIdx) * sizeof(UrmaWQCtx));
}

AICORE inline __gm__ UrmaCqCtx* GetTrackedCq(__gm__ UrmaInfo* info, uint32_t peer, uint32_t qpIdx)
{
    return reinterpret_cast<__gm__ UrmaCqCtx*>(
        info->scqPtr + (static_cast<uint64_t>(peer) * info->qpNum + qpIdx) * sizeof(UrmaCqCtx));
}

AICORE inline __gm__ UrmaQueueRuntime* GetTrackedRuntime(__gm__ UrmaInfo* info, uint32_t peer, uint32_t qpIdx)
{
    return reinterpret_cast<__gm__ UrmaQueueRuntime*>(
        info->runtimePtr + (static_cast<uint64_t>(peer) * info->qpNum + qpIdx) * sizeof(UrmaQueueRuntime));
}

AICORE inline bool ValidateTrackedSession(const AsyncSession& session, uint32_t peer)
{
    if (!session.valid || session.engine != DmaEngine::URMA || session.contextGm == nullptr) {
        return false;
    }
    __gm__ UrmaInfo* info = reinterpret_cast<__gm__ UrmaInfo*>(session.contextGm);
    return info->runtimePtr != 0U && peer < info->rankCount && peer != info->localRankId && session.qpIdx < info->qpNum;
}

AICORE inline void PublishTrackedTails(
    __gm__ UrmaCqCtx* cq, __gm__ UrmaWQCtx* wq, uint32_t completedCqe, uint32_t completedBb)
{
    st_dev(completedCqe, reinterpret_cast<__gm__ uint32_t*>(cq->tailAddr), 0);
    st_dev(completedCqe & 0xFFFFFFU, reinterpret_cast<__gm__ uint32_t*>(cq->dbAddr), 0);
    st_dev(completedBb, reinterpret_cast<__gm__ uint32_t*>(wq->tailAddr), 0);
}

AICORE inline __gm__ UrmaCompletionRecord* GetCompletionRecord(__gm__ UrmaQueueRuntime* runtime, uint32_t cqeSequence)
{
    return reinterpret_cast<__gm__ UrmaCompletionRecord*>(runtime->completionRecords) +
           (cqeSequence & (runtime->completionDepth - 1U));
}

AICORE inline void RegisterCompletion(__gm__ UrmaQueueRuntime* runtime, uint32_t cqeSequence, uint32_t bbSequence)
{
    __gm__ UrmaCompletionRecord* record = GetCompletionRecord(runtime, cqeSequence);
    record->cqeSequence = cqeSequence;
    record->bbSequence = bbSequence;
}

AICORE inline bool IsCqeReady(__gm__ UrmaCqCtx* cq, uint32_t sequence)
{
    const uint32_t cqeSize = 1U << cq->cqeShiftSize;
    __gm__ UrmaJfcCqeCtx* cqe = reinterpret_cast<__gm__ UrmaJfcCqeCtx*>(
        cq->bufAddr + static_cast<uint64_t>(cqeSize) * (sequence & (cq->depth - 1U)));
    const bool expectedOwner = (sequence / cq->depth) & 1U;
    DcciCachelines(reinterpret_cast<__gm__ uint8_t*>(cqe), sizeof(UrmaJfcCqeCtx));
    return (expectedOwner ^ cqe->owner) != 0U;
}

AICORE inline UrmaRetireResult RetireTrackedCompletions(
    const AsyncSession& session, uint32_t peer, uint32_t targetCqe, bool blocking)
{
    if (!ValidateTrackedSession(session, peer)) {
        return {false, false};
    }
    __gm__ UrmaInfo* info = reinterpret_cast<__gm__ UrmaInfo*>(session.contextGm);
    __gm__ UrmaWQCtx* wq = GetTrackedWq(info, peer, session.qpIdx);
    __gm__ UrmaCqCtx* cq = GetTrackedCq(info, peer, session.qpIdx);
    __gm__ UrmaQueueRuntime* runtime = GetTrackedRuntime(info, peer, session.qpIdx);
    if (!SequenceReached(runtime->submittedCqe, targetCqe)) {
        return {false, false};
    }
    if (SequenceReached(runtime->completedCqe, targetCqe)) {
        const bool success = runtime->firstError == 0U || !SequenceReached(targetCqe, runtime->firstErrorCqe);
        return {true, success};
    }

    if (!blocking) {
        for (uint32_t sequence = runtime->completedCqe; sequence != targetCqe; ++sequence) {
            if (!IsCqeReady(cq, sequence)) {
                return {false, true};
            }
        }
    }

    uint32_t completedCqe = runtime->completedCqe;
    uint32_t completedBb = runtime->completedBb;
    while (completedCqe != targetCqe) {
        const uint32_t sequence = completedCqe;
        uint32_t polls = 0U;
        while (!IsCqeReady(cq, sequence)) {
            if (!blocking || ++polls >= kUrmaMaxPollTimes) {
                PublishTrackedTails(cq, wq, completedCqe, completedBb);
                return {false, !blocking};
            }
        }

        const uint32_t cqeSize = 1U << cq->cqeShiftSize;
        __gm__ UrmaJfcCqeCtx* cqe = reinterpret_cast<__gm__ UrmaJfcCqeCtx*>(
            cq->bufAddr + static_cast<uint64_t>(cqeSize) * (sequence & (cq->depth - 1U)));
        const uint32_t error = (static_cast<uint32_t>(cqe->status) << 8U) | cqe->substatus;
        ++completedCqe;

        __gm__ UrmaCompletionRecord* record = GetCompletionRecord(runtime, completedCqe);
        DcciCachelines(reinterpret_cast<__gm__ uint8_t*>(record), sizeof(UrmaCompletionRecord));
        if (record->cqeSequence != completedCqe) {
            return {false, false};
        }
        completedBb = record->bbSequence;
        runtime->completedCqe = completedCqe;
        runtime->completedBb = completedBb;
        if (error != 0U && runtime->firstError == 0U) {
            runtime->firstError = error;
            runtime->firstErrorCqe = completedCqe;
        }
    }

    PublishTrackedTails(cq, wq, completedCqe, completedBb);
    return {true, runtime->firstError == 0U || !SequenceReached(targetCqe, runtime->firstErrorCqe)};
}

AICORE inline bool EnsureTrackedCapacity(
    const AsyncSession& session, uint32_t peer, uint32_t requiredBb, uint32_t requiredCqe)
{
    __gm__ UrmaInfo* info = reinterpret_cast<__gm__ UrmaInfo*>(session.contextGm);
    __gm__ UrmaWQCtx* wq = GetTrackedWq(info, peer, session.qpIdx);
    __gm__ UrmaCqCtx* cq = GetTrackedCq(info, peer, session.qpIdx);
    __gm__ UrmaQueueRuntime* runtime = GetTrackedRuntime(info, peer, session.qpIdx);
    if (runtime->firstError != 0U || requiredBb >= wq->depth || requiredCqe >= cq->depth) {
        return false;
    }
    if (runtime->submittedBb - runtime->completedBb + requiredBb >= wq->depth ||
        runtime->submittedCqe - runtime->completedCqe + requiredCqe >= cq->depth) {
        if (runtime->submittedCqe == runtime->completedCqe) {
            return false;
        }
        const UrmaRetireResult result = RetireTrackedCompletions(session, peer, runtime->submittedCqe, true);
        if (!result.completed || !result.success) {
            return false;
        }
    }
    return true;
}

AICORE inline void FillTransferWqe(
    __gm__ uint8_t* wqe, __gm__ UrmaMemInfo* remoteMem, __gm__ uint8_t* remoteAddr, __gm__ uint8_t* localAddr,
    uint64_t messageLen, UrmaOpcode opcode, uint32_t bbSequence, uint32_t depth, uint32_t localTokenId)
{
    FillSqeCtx(reinterpret_cast<__gm__ UrmaSqeCtx*>(wqe), remoteMem, remoteAddr, opcode, bbSequence, depth);
    __gm__ UrmaSgeCtx* sge = reinterpret_cast<__gm__ UrmaSgeCtx*>(wqe + kUrmaSqeSizeBytes);
    sge->len = static_cast<uint32_t>(messageLen);
    sge->tokenId = localTokenId;
    sge->va = reinterpret_cast<uint64_t>(localAddr);
}

AICORE inline void FillInlineSetWqe(
    __gm__ uint8_t* wqe, __gm__ UrmaMemInfo* remoteMem, __gm__ int32_t* remoteSignal, int32_t signalValue,
    uint32_t bbSequence, uint32_t depth)
{
    __gm__ UrmaSqeCtx* sqe = reinterpret_cast<__gm__ UrmaSqeCtx*>(wqe);
    FillSqeCtx(sqe, remoteMem, reinterpret_cast<__gm__ uint8_t*>(remoteSignal), UrmaOpcode::WRITE, bbSequence, depth);
    sqe->flag = kUrmaWqeFlagCqe | kUrmaWqeFlagInline;
    sqe->inlineMsgLen = sizeof(int32_t);
    sqe->sgeNum = 0U;
    *reinterpret_cast<__gm__ int32_t*>(wqe + kUrmaSqeSizeBytes) = signalValue;
}

AICORE inline void FillFaaWqe(
    __gm__ uint8_t* firstBb, __gm__ uint8_t* secondBb, __gm__ UrmaMemInfo* remoteMem, __gm__ int32_t* remoteSignal,
    __gm__ int32_t* result, int32_t signalValue, uint32_t bbSequence, uint32_t depth, uint32_t resultTokenId)
{
    __gm__ UrmaSqeCtx* sqe = reinterpret_cast<__gm__ UrmaSqeCtx*>(firstBb);
    FillSqeCtx(sqe, remoteMem, reinterpret_cast<__gm__ uint8_t*>(remoteSignal), UrmaOpcode::FAA, bbSequence, depth);
    sqe->flag = kUrmaWqeFlagCqe | kUrmaWqeFlagExtended;
    __gm__ UrmaSgeCtx* sge = reinterpret_cast<__gm__ UrmaSgeCtx*>(firstBb + kUrmaSqeSizeBytes);
    sge->len = sizeof(int32_t);
    sge->tokenId = resultTokenId;
    sge->va = reinterpret_cast<uint64_t>(result);
    *reinterpret_cast<__gm__ int32_t*>(secondBb) = signalValue;
}

AICORE inline uint64_t UrmaPostTrackedTransfer(
    __gm__ uint8_t* remoteAddr, __gm__ uint8_t* localAddr, uint64_t messageLen, UrmaOpcode opcode,
    const AsyncSession& session, uint32_t peer)
{
    if (!ValidateTrackedSession(session, peer) || messageLen == 0U || messageLen > kUrmaMaxWqeTransferBytes ||
        (opcode != UrmaOpcode::WRITE && opcode != UrmaOpcode::READ)) {
        return 0U;
    }
    __gm__ UrmaInfo* info = reinterpret_cast<__gm__ UrmaInfo*>(session.contextGm);
    __gm__ UrmaMemInfo* remoteMem = reinterpret_cast<__gm__ UrmaMemInfo*>(info->memPtr) + peer;
    if (!IsRangeInside(reinterpret_cast<uint64_t>(remoteAddr), messageLen, remoteMem->addr, remoteMem->len) ||
        !IsRangeInside(reinterpret_cast<uint64_t>(localAddr), messageLen, info->localMemAddr, info->localMemSize) ||
        !EnsureTrackedCapacity(session, peer, 1U, 1U)) {
        return 0U;
    }

    __gm__ UrmaWQCtx* wq = GetTrackedWq(info, peer, session.qpIdx);
    __gm__ UrmaQueueRuntime* runtime = GetTrackedRuntime(info, peer, session.qpIdx);
    const uint32_t bbSize = 1U << wq->wqeShiftSize;
    const uint32_t startBb = runtime->submittedBb;
    __gm__ uint8_t* wqe =
        reinterpret_cast<__gm__ uint8_t*>(wq->bufAddr + static_cast<uint64_t>(bbSize) * (startBb & (wq->depth - 1U)));
    FillTransferWqe(wqe, remoteMem, remoteAddr, localAddr, messageLen, opcode, startBb, wq->depth, info->localTokenId);

    const uint32_t targetBb = startBb + 1U;
    const uint32_t targetCqe = runtime->submittedCqe + 1U;
    RegisterCompletion(runtime, targetCqe, targetBb);
    runtime->submittedBb = targetBb;
    runtime->submittedCqe = targetCqe;
    pipe_barrier(PIPE_ALL);
    DcciCachelines(wqe, kUrmaSqeSizeBytes + kUrmaSgeSizeBytes);
    DcciCachelines(
        reinterpret_cast<__gm__ uint8_t*>(GetCompletionRecord(runtime, targetCqe)), sizeof(UrmaCompletionRecord));
    pipe_barrier(PIPE_ALL);
    UrmaPostSendUpdateInfo(targetBb, wq);
    return EncodeHandle(peer, targetCqe);
}

AICORE inline uint64_t UrmaPostTrackedNotify(
    __gm__ uint8_t* remotePayload, __gm__ uint8_t* localPayload, uint64_t messageLen, __gm__ int32_t* remoteSignal,
    int32_t signalValue, NotifyOp notifyOp, const AsyncSession& session, uint32_t peer)
{
    if (!ValidateTrackedSession(session, peer) || messageLen == 0U || messageLen > kUrmaMaxWqeTransferBytes ||
        (notifyOp != NotifyOp::Set && notifyOp != NotifyOp::AtomicAdd)) {
        return 0U;
    }
    __gm__ UrmaInfo* info = reinterpret_cast<__gm__ UrmaInfo*>(session.contextGm);
    __gm__ UrmaMemInfo* remoteMem = reinterpret_cast<__gm__ UrmaMemInfo*>(info->memPtr) + peer;
    if (!IsRangeInside(reinterpret_cast<uint64_t>(remotePayload), messageLen, remoteMem->addr, remoteMem->len) ||
        !IsRangeInside(reinterpret_cast<uint64_t>(localPayload), messageLen, info->localMemAddr, info->localMemSize) ||
        !IsRangeInside(reinterpret_cast<uint64_t>(remoteSignal), sizeof(int32_t), remoteMem->addr, remoteMem->len) ||
        (reinterpret_cast<uint64_t>(remoteSignal) & (alignof(int32_t) - 1U)) != 0U) {
        return 0U;
    }

    const bool isAdd = notifyOp == NotifyOp::AtomicAdd;
    const uint32_t requiredBb = isAdd ? 3U : 2U;
    constexpr uint32_t requiredCqe = 2U;
    __gm__ UrmaQueueRuntime* runtime = GetTrackedRuntime(info, peer, session.qpIdx);
    uint32_t notifySlot = 0U;
    if (isAdd) {
        notifySlot = runtime->nextNotifySlot % kUrmaNotifyResultSlotCount;
        const uint32_t reusableCqe = runtime->notifyReusableCqe[notifySlot];
        if (!SequenceReached(runtime->completedCqe, reusableCqe)) {
            const UrmaRetireResult result = RetireTrackedCompletions(session, peer, reusableCqe, true);
            if (!result.completed || !result.success) {
                return 0U;
            }
        }
    }
    if (!EnsureTrackedCapacity(session, peer, requiredBb, requiredCqe)) {
        return 0U;
    }

    __gm__ UrmaWQCtx* wq = GetTrackedWq(info, peer, session.qpIdx);
    const uint32_t bbSize = 1U << wq->wqeShiftSize;
    const uint32_t startBb = runtime->submittedBb;
    const uint32_t signalBb = startBb + 1U;
    __gm__ uint8_t* payloadWqe =
        reinterpret_cast<__gm__ uint8_t*>(wq->bufAddr + static_cast<uint64_t>(bbSize) * (startBb & (wq->depth - 1U)));
    __gm__ uint8_t* signalFirstBb =
        reinterpret_cast<__gm__ uint8_t*>(wq->bufAddr + static_cast<uint64_t>(bbSize) * (signalBb & (wq->depth - 1U)));
    __gm__ uint8_t* signalSecondBb = reinterpret_cast<__gm__ uint8_t*>(
        wq->bufAddr + static_cast<uint64_t>(bbSize) * ((signalBb + 1U) & (wq->depth - 1U)));

    FillTransferWqe(
        payloadWqe, remoteMem, remotePayload, localPayload, messageLen, UrmaOpcode::WRITE, startBb, wq->depth,
        info->localTokenId);
    if (isAdd) {
        __gm__ int32_t* result = reinterpret_cast<__gm__ int32_t*>(info->notifyResultPtr) +
                                 static_cast<uint64_t>(peer) * kUrmaNotifyResultSlotCount + notifySlot;
        FillFaaWqe(
            signalFirstBb, signalSecondBb, remoteMem, remoteSignal, result, signalValue, signalBb, wq->depth,
            info->notifyTokenId);
    } else {
        FillInlineSetWqe(signalFirstBb, remoteMem, remoteSignal, signalValue, signalBb, wq->depth);
    }

    const uint32_t payloadCqe = runtime->submittedCqe + 1U;
    const uint32_t targetCqe = payloadCqe + 1U;
    const uint32_t targetBb = startBb + requiredBb;
    RegisterCompletion(runtime, payloadCqe, startBb + 1U);
    RegisterCompletion(runtime, targetCqe, targetBb);
    runtime->submittedBb = targetBb;
    runtime->submittedCqe = targetCqe;
    if (isAdd) {
        runtime->notifyReusableCqe[notifySlot] = targetCqe;
        runtime->nextNotifySlot++;
    }

    pipe_barrier(PIPE_ALL);
    DcciCachelines(payloadWqe, kUrmaSqeSizeBytes + kUrmaSgeSizeBytes);
    DcciCachelines(signalFirstBb, kUrmaSqeSizeBytes + kUrmaSgeSizeBytes);
    if (isAdd) {
        DcciCachelines(signalSecondBb, sizeof(int32_t));
    }
    DcciCachelines(
        reinterpret_cast<__gm__ uint8_t*>(GetCompletionRecord(runtime, payloadCqe)), sizeof(UrmaCompletionRecord));
    DcciCachelines(
        reinterpret_cast<__gm__ uint8_t*>(GetCompletionRecord(runtime, targetCqe)), sizeof(UrmaCompletionRecord));
    pipe_barrier(PIPE_ALL);
    UrmaPostSendUpdateInfo(targetBb, wq);
    return EncodeHandle(peer, targetCqe);
}

// ============================================================================
// Handle encoding/decoding for AsyncEvent
// ============================================================================
constexpr uint32_t kHandleRankIdShift = 32;

AICORE inline uint64_t EncodeHandle(uint32_t destRankId, uint32_t curHead)
{
    return (static_cast<uint64_t>(destRankId) << kHandleRankIdShift) | static_cast<uint64_t>(curHead);
}

AICORE inline void DecodeHandle(uint64_t handle, uint32_t& destRankId, uint32_t& curHead)
{
    destRankId = static_cast<uint32_t>(handle >> kHandleRankIdShift);
    curHead = static_cast<uint32_t>(handle & 0xFFFFFFFF);
}

// ============================================================================
// UrmaWaitEvent — blocking wait for URMA completion (polls CQ)
// ============================================================================
AICORE inline bool UrmaWaitEvent(uint64_t eventHandle, const UrmaEventContext& eventCtx)
{
    uint32_t destRankId = 0;
    uint32_t curHead = 0;
    DecodeHandle(eventHandle, destRankId, curHead);
    uint32_t ret = UrmaPollCq(eventCtx.contextGm, destRankId, 0, curHead);
    return ret == 0;
}

AICORE inline bool UrmaWaitEvent(uint64_t eventHandle, const AsyncSession& session)
{
    uint32_t destRankId = 0;
    uint32_t targetCqe = 0;
    DecodeHandle(eventHandle, destRankId, targetCqe);
    const UrmaRetireResult result = RetireTrackedCompletions(session, destRankId, targetCqe, true);
    return result.completed && result.success;
}

// ============================================================================
// UrmaTestEvent — non-blocking completion check
// ============================================================================
AICORE inline bool UrmaTestEvent(uint64_t eventHandle, const UrmaEventContext& eventCtx)
{
    uint32_t destRankId = 0;
    uint32_t curHead = 0;
    DecodeHandle(eventHandle, destRankId, curHead);

    __gm__ UrmaInfo* urmaInfo = (__gm__ UrmaInfo*)eventCtx.contextGm;
    uint32_t qpNum = urmaInfo->qpNum;
    __gm__ UrmaCqCtx* cqCtxEntry = (__gm__ UrmaCqCtx*)(urmaInfo->scqPtr + (destRankId * qpNum + 0) * sizeof(UrmaCqCtx));
    uint32_t curTail = ld_dev((__gm__ uint32_t*)cqCtxEntry->tailAddr, 0);
    if (static_cast<int32_t>(curTail - curHead) >= 0) {
        return true;
    }

    uint64_t cqBaseAddr = cqCtxEntry->bufAddr;
    uint32_t cqeSize = 1U << cqCtxEntry->cqeShiftSize;
    uint32_t depth = cqCtxEntry->depth;

    uint32_t lastIdx = curHead - 1;
    __gm__ UrmaJfcCqeCtx* lastCqe = (__gm__ UrmaJfcCqeCtx*)(cqBaseAddr + cqeSize * (lastIdx & (depth - 1)));
    bool validOwner = (lastIdx / depth) & 1;

    DcciCachelines((__gm__ uint8_t*)lastCqe, sizeof(UrmaJfcCqeCtx));

    return (validOwner ^ lastCqe->owner) != 0;
}

AICORE inline bool UrmaTestEvent(uint64_t eventHandle, const AsyncSession& session)
{
    uint32_t destRankId = 0;
    uint32_t targetCqe = 0;
    DecodeHandle(eventHandle, destRankId, targetCqe);
    const UrmaRetireResult result = RetireTrackedCompletions(session, destRankId, targetCqe, false);
    return result.completed && result.success;
}

} // namespace detail

// ============================================================================
// Public API: __urma_put_async / __urma_get_async
// peer selects SQ/CQ/MemInfo; preferred over execCtx.destRankId for multi-peer reuse.
// ============================================================================

AICORE inline uint64_t __urma_put_async(
    __gm__ uint8_t* dst, __gm__ uint8_t* src, uint64_t transferSize, const UrmaExecContext& execCtx, uint32_t peer)
{
    uint32_t curHead =
        detail::UrmaPostSend(execCtx.contextGm, dst, src, peer, execCtx.qpIdx, UrmaOpcode::WRITE, transferSize);
    return detail::EncodeHandle(peer, curHead);
}

AICORE inline uint64_t __urma_put_async(
    __gm__ uint8_t* dst, __gm__ uint8_t* src, uint64_t transferSize, const AsyncSession& session, uint32_t peer)
{
    return detail::UrmaPostTrackedTransfer(dst, src, transferSize, UrmaOpcode::WRITE, session, peer);
}

AICORE inline uint64_t __urma_get_async(
    __gm__ uint8_t* dst, __gm__ uint8_t* src, uint64_t transferSize, const UrmaExecContext& execCtx, uint32_t peer)
{
    // RDMA READ: remote addr = src (SQE remote field), local addr = dst (SGE.va)
    uint32_t curHead =
        detail::UrmaPostSend(execCtx.contextGm, src, dst, peer, execCtx.qpIdx, UrmaOpcode::READ, transferSize);
    return detail::EncodeHandle(peer, curHead);
}

AICORE inline uint64_t __urma_get_async(
    __gm__ uint8_t* dst, __gm__ uint8_t* src, uint64_t transferSize, const AsyncSession& session, uint32_t peer)
{
    // RDMA READ: remote addr = src (SQE remote field), local addr = dst (SGE.va)
    return detail::UrmaPostTrackedTransfer(src, dst, transferSize, UrmaOpcode::READ, session, peer);
}

AICORE inline uint64_t __urma_put_async_notify(
    __gm__ uint8_t* dst, __gm__ uint8_t* src, uint64_t transferSize, __gm__ int32_t* signal, int32_t signalValue,
    NotifyOp notifyOp, const AsyncSession& session, uint32_t peer)
{
    return detail::UrmaPostTrackedNotify(dst, src, transferSize, signal, signalValue, notifyOp, session, peer);
}

AICORE inline uint64_t __urma_put_async(
    __gm__ uint8_t* dst, __gm__ uint8_t* src, uint64_t transferSize, const AsyncSession& session)
{
    return __urma_put_async(dst, src, transferSize, session, session.destRankId);
}

AICORE inline uint64_t __urma_get_async(
    __gm__ uint8_t* dst, __gm__ uint8_t* src, uint64_t transferSize, const AsyncSession& session)
{
    return __urma_get_async(dst, src, transferSize, session, session.destRankId);
}

AICORE inline uint64_t __urma_put_async(
    __gm__ uint8_t* dst, __gm__ uint8_t* src, uint64_t transferSize, const UrmaExecContext& execCtx)
{
    return __urma_put_async(dst, src, transferSize, execCtx, execCtx.destRankId);
}

AICORE inline uint64_t __urma_get_async(
    __gm__ uint8_t* dst, __gm__ uint8_t* src, uint64_t transferSize, const UrmaExecContext& execCtx)
{
    return __urma_get_async(dst, src, transferSize, execCtx, execCtx.destRankId);
}

// ============================================================================
// BuildUrmaSession — fill UrmaSession from workspace (peer at TPUT_ASYNC / TGET_ASYNC)
// ============================================================================

AICORE inline bool BuildUrmaSession(__gm__ uint8_t* contextGm, UrmaSession& session)
{
    session.execCtx.contextGm = contextGm;
    session.execCtx.destRankId = 0;
    session.execCtx.qpIdx = 0;
    session.eventCtx.contextGm = contextGm;
    session.valid = (contextGm != nullptr);
    return session.valid;
}

AICORE inline bool BuildUrmaSession(__gm__ uint8_t* contextGm, uint32_t destRankId, UrmaSession& session)
{
    if (!BuildUrmaSession(contextGm, session)) {
        return false;
    }
    session.execCtx.destRankId = destRankId;
    return true;
}

// ============================================================================
// UrmaPeerMrBaseAddr — symmetric MR base (device VA) for peer index peerRank
//
// Indexes into the per-peer UrmaMemInfo array at memPtr + sizeof(UrmaMemInfo) * peerRank.
// peerRank uses UrmaWorkspaceManager allgather order (MPI rank order, 0 .. rankCount-1).
// ============================================================================
AICORE inline uint64_t UrmaPeerMrBaseAddr(__gm__ uint8_t* urmaWorkspace, uint32_t peerRank)
{
    __gm__ UrmaInfo* info = (__gm__ UrmaInfo*)urmaWorkspace;
    PTO_ASSERT(peerRank < info->rankCount, "UrmaPeerMrBaseAddr: peerRank out of range");
    __gm__ UrmaMemInfo* memRow = reinterpret_cast<__gm__ UrmaMemInfo*>(info->memPtr) + peerRank;
    return memRow->addr;
}

} // namespace urma
} // namespace comm
} // namespace pto

#endif // PTO_URMA_SUPPORTED
#endif // PTO_COMM_ASYNC_URMA_INTRIN_HPP
