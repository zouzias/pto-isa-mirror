/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMM_ASYNC_URMA_POST_HPP
#define PTO_COMM_ASYNC_URMA_POST_HPP

// Included from urma_async_intrin.hpp inside namespace detail.
// Posts WRITE/READ and WRITE+notify WQEs (payload first, signal last).

// SQE flag byte = low byte of the UB JfsWrFlag. Bit 5 (completeEnable) requests a
// CQE; bits 0-1 are placeOrder, URMA's ordering hint (0=no order, 1=relax,
// 2=strong). A strong SQE only refuses to overtake preceding *relax* SQEs, so
// ordering payload-before-signal needs BOTH halves of the pair: relax on every
// payload WQE and strong on the signal WQE. placeOrder is scoped to one jetty/SQ.
constexpr uint32_t kUrmaSqeFlagCqe = 0x20U;
constexpr uint32_t kUrmaPlaceOrderNone = 0x00U;
constexpr uint32_t kUrmaPlaceOrderRelax = 0x01U;
constexpr uint32_t kUrmaPlaceOrderStrong = 0x02U;

AICORE inline void FillSqeCtx(
    __gm__ UrmaSqeCtx* sqe, __gm__ UrmaMemInfo* remoteMem, __gm__ uint8_t* remoteAddr, UrmaOpcode opcode,
    uint32_t bbSequence, uint32_t depth, uint32_t placeOrder = kUrmaPlaceOrderNone)
{
    sqe->sqeBbIdx = static_cast<uint16_t>(bbSequence % depth);
    sqe->opcode = static_cast<uint32_t>(opcode);
    sqe->flag = kUrmaSqeFlagCqe | placeOrder;
    sqe->rsv0 = 0U;
    sqe->nf = 0U;
    sqe->tokenEn = remoteMem->tokenValueValid ? 1U : 0U;
    sqe->rmtJettyType = remoteMem->rmtJettyType;
    sqe->owner = (bbSequence & depth) == 0U ? 1U : 0U;
    sqe->targetHint = remoteMem->targetHint;
    sqe->rsv1 = 0U;
    sqe->inlineMsgLen = 0U;
    sqe->tpId = remoteMem->tpn;
    sqe->sgeNum = 1U;
    sqe->rmtJettyOrSegId = remoteMem->tid;
    sqe->rsv2 = 0U;
    sqe->rmtTokenValue = remoteMem->rmtTokenValue;
    sqe->udfType = 0U;
    sqe->reduceDataType = 0U;
    sqe->reduceOpcode = 0U;
    sqe->rsv3 = 0U;

    const uint64_t remoteAddress = reinterpret_cast<uint64_t>(remoteAddr);
    __gm__ uint8_t* bytes = reinterpret_cast<__gm__ uint8_t*>(sqe);
    __gm__ uint64_t* eid = reinterpret_cast<__gm__ uint64_t*>(remoteMem->eidAddr);
    *reinterpret_cast<__gm__ uint64_t*>(bytes + kUrmaSqeRmtEidLOffset) = eid[0];
    *reinterpret_cast<__gm__ uint64_t*>(bytes + kUrmaSqeRmtEidHOffset) = eid[1];
    *reinterpret_cast<__gm__ uint32_t*>(bytes + kUrmaSqeRmtAddrLOffset) = static_cast<uint32_t>(remoteAddress);
    *reinterpret_cast<__gm__ uint32_t*>(bytes + kUrmaSqeRmtAddrHOffset) = static_cast<uint32_t>(remoteAddress >> 32U);
}

AICORE inline void FillTransferWqe(
    __gm__ uint8_t* wqe, __gm__ UrmaMemInfo* remoteMem, __gm__ uint8_t* remoteAddr, __gm__ uint8_t* localAddr,
    uint64_t messageLen, UrmaOpcode opcode, uint32_t bbSequence, uint32_t depth, uint32_t localTokenId,
    uint32_t placeOrder = kUrmaPlaceOrderNone)
{
    FillSqeCtx(reinterpret_cast<__gm__ UrmaSqeCtx*>(wqe), remoteMem, remoteAddr, opcode, bbSequence, depth, placeOrder);
    __gm__ UrmaSgeCtx* sge = reinterpret_cast<__gm__ UrmaSgeCtx*>(wqe + kUrmaSqeSizeBytes);
    sge->len = static_cast<uint32_t>(messageLen);
    sge->tokenId = localTokenId;
    sge->va = reinterpret_cast<uint64_t>(localAddr);
}

AICORE inline void FillFaaWqe(
    __gm__ uint8_t* firstBb, __gm__ uint8_t* secondBb, __gm__ UrmaMemInfo* remoteMem, __gm__ int32_t* remoteSignal,
    __gm__ int32_t* resultSink, int32_t addValue, uint32_t bbSequence, uint32_t depth, uint32_t localTokenId)
{
    // The FAA signal is the last WQE of a notify; mark it strong so it cannot
    // overtake the preceding relax-marked payload writes (0x22 = CQE | strong).
    FillSqeCtx(
        reinterpret_cast<__gm__ UrmaSqeCtx*>(firstBb), remoteMem, reinterpret_cast<__gm__ uint8_t*>(remoteSignal),
        UrmaOpcode::FAA, bbSequence, depth, kUrmaPlaceOrderStrong);
    __gm__ UrmaSgeCtx* sge = reinterpret_cast<__gm__ UrmaSgeCtx*>(firstBb + kUrmaSqeSizeBytes);
    sge->len = sizeof(int32_t);
    sge->tokenId = localTokenId;
    sge->va = reinterpret_cast<uint64_t>(resultSink);
    *reinterpret_cast<__gm__ int32_t*>(secondBb) = addValue;
}

// Non-blocking capacity probe. Occupancy is measured against the local producer
// head, which may already include WQEs filled but not yet announced via the
// doorbell, rather than wq->headAddr. That lets a multi-WQE split be filled
// before a single doorbell without under-counting the slots in use.
AICORE inline bool UrmaRingHasRoom(
    __gm__ UrmaWQCtx* wq, __gm__ UrmaCqCtx* cq, uint32_t head, uint32_t submittedWqeCount, uint32_t requiredBb,
    uint32_t requiredWqeCount)
{
    const uint32_t completedBb = ld_dev(reinterpret_cast<__gm__ uint32_t*>(wq->tailAddr), 0);
    const uint32_t completedCqe = ld_dev(reinterpret_cast<__gm__ uint32_t*>(cq->tailAddr), 0);
    return head - completedBb + requiredBb <= wq->depth &&
           submittedWqeCount - completedCqe + requiredWqeCount <= cq->depth;
}

// Fills one transfer WQE at producer index `head`, but does not advance the head
// or ring the doorbell. The caller batches those so an N-way split needs only a
// single doorbell.
AICORE inline void FillTransferWqeAt(
    __gm__ UrmaWQCtx* wq, __gm__ UrmaMemInfo* remoteMem, __gm__ uint8_t* remoteAddr, __gm__ uint8_t* localAddr,
    uint32_t messageLen, UrmaOpcode opcode, uint32_t head, uint32_t placeOrder = kUrmaPlaceOrderNone)
{
    const uint32_t bbSize = 1U << wq->wqeShiftSize;
    __gm__ uint8_t* wqe = reinterpret_cast<__gm__ uint8_t*>(wq->bufAddr + bbSize * (head % wq->depth));
    FillTransferWqe(
        wqe, remoteMem, remoteAddr, localAddr, messageLen, opcode, head, wq->depth, remoteMem->localTokenId,
        placeOrder);
    DcciCachelines(wqe, kUrmaSqeSizeBytes + kUrmaSgeSizeBytes);
}

// Publishes every WQE filled up to `head` and rings the doorbell exactly once.
AICORE inline void CommitPostedWqes(__gm__ UrmaWQCtx* wq, uint32_t head, uint32_t submittedWqeCount)
{
    wq->submittedWqeCount = submittedWqeCount;
    pipe_barrier(PIPE_ALL);
    DcciCachelines(reinterpret_cast<__gm__ uint8_t*>(&wq->submittedWqeCount), sizeof(wq->submittedWqeCount));
    dsb(DSB_DDR);
    UrmaPostSendUpdateInfo(head, wq);
}

AICORE inline UrmaPostResult UrmaPostSend(
    __gm__ uint8_t* remoteAddr, __gm__ uint8_t* localAddr, uint64_t messageLen, UrmaOpcode opcode,
    const AsyncSession& session, uint32_t peer)
{
    PTO_ASSERT(ValidateUrmaSession(session, peer), "UrmaPostSend: invalid URMA session, peer or QP.");
    PTO_ASSERT(
        opcode == UrmaOpcode::WRITE || opcode == UrmaOpcode::READ, "UrmaPostSend: opcode must be WRITE or READ.");

    // Zero-length transfer is a no-op: handle 0 tells Wait/Test there is nothing
    // outstanding, matching the SDMA/RDMA engines.
    if (messageLen == 0U) {
        return {0U, 0U};
    }

    // A single WQE's sge->len is a uint32_t and the UB protocol caps one WQE at
    // kUrmaMaxWqeTransferBytes (256MB). Instead of truncating an oversized
    // messageLen (which issues a corrupt transfer and can crash the card) or
    // forcing every caller to pre-split, fan the contiguous range out over as many
    // <=256MB WQEs as needed. All the WQEs are filled first and announced with a
    // single doorbell; only when the ring cannot hold the rest of the split do we
    // ring early to let the hardware drain. Returning the last WQE's handle is
    // enough: Wait/Test drain the whole submitted prefix by sequence number.
    __gm__ UrmaInfo* info = reinterpret_cast<__gm__ UrmaInfo*>(session.contextGm);
    __gm__ UrmaWQCtx* wq = GetWqContext(session, peer);
    __gm__ UrmaCqCtx* cq = GetCqContext(session, peer);
    __gm__ UrmaMemInfo* remoteMem = GetRemoteMemInfo(info, peer, session.qpIdx);

    const uint32_t announcedHead =
        ld_dev(reinterpret_cast<__gm__ uint32_t*>(wq->headAddr), 0); // head the hardware already knows
    uint32_t pendingHead = announcedHead; // head of WQEs filled but not yet rung in this call
    uint32_t submittedWqeCount = wq->submittedWqeCount;
    uint64_t offset = 0U;
    while (offset < messageLen) {
        // Ring full: before blocking on completions, announce whatever we have
        // filled but not yet rung (otherwise we would wait on WQEs the hardware was
        // never told about — deadlock). If nothing new is pending, the outstanding
        // WQEs were already rung by earlier calls, so just drain.
        if (!UrmaRingHasRoom(wq, cq, pendingHead, submittedWqeCount, 1U, 1U)) {
            if (pendingHead != announcedHead) {
                CommitPostedWqes(wq, pendingHead, submittedWqeCount);
            }
            bool completed = false;
            const bool drained =
                UrmaPollCq(session, peer, pendingHead, submittedWqeCount, true, completed) && completed;
            PTO_ASSERT(drained, "UrmaPostSend: failed to acquire URMA queue capacity.");
        }

        const uint64_t remaining = messageLen - offset;
        const uint32_t chunk =
            static_cast<uint32_t>(remaining < kUrmaMaxWqeTransferBytes ? remaining : kUrmaMaxWqeTransferBytes);
        FillTransferWqeAt(wq, remoteMem, remoteAddr + offset, localAddr + offset, chunk, opcode, pendingHead);
        pendingHead += 1U;
        submittedWqeCount += 1U;
        offset += chunk;
    }
    CommitPostedWqes(wq, pendingHead, submittedWqeCount);
    return {EncodeHandle(peer, pendingHead), submittedWqeCount};
}

// SET reuses a small ring of value slots inside the notify region; FAA never
// touches it. When we are about to reuse slot 0 while an earlier SET WQE may
// still reference it, drain the outstanding WQEs first so the in-flight transfer
// reads the old value before we overwrite it. Returns the slot this call owns.
AICORE inline bool PrepareNotifySetSlot(
    const AsyncSession& session, uint32_t peer, __gm__ UrmaNotifyResourceRegion* region, bool isAdd,
    uint32_t& setSlotIndex)
{
    PTO_ASSERT(
        isAdd || region->nextSetSlot < kUrmaNotifySetSlotCount, "PrepareNotifySetSlot: next SET slot is out of range.");
    setSlotIndex = isAdd ? 0U : region->nextSetSlot;
    const bool reusingSetSlotZero = !isAdd && region->setRingStarted != 0U && setSlotIndex == 0U;
    if (!reusingSetSlotZero) {
        return true;
    }
    __gm__ UrmaCqCtx* cq = GetCqContext(session, peer);
    __gm__ UrmaWQCtx* wq = GetWqContext(session, peer);
    const uint32_t completedCqe = ld_dev(reinterpret_cast<__gm__ uint32_t*>(cq->tailAddr), 0);
    if (wq->submittedWqeCount == completedCqe) {
        return true;
    }
    bool completed = false;
    const uint32_t submittedBb = ld_dev(reinterpret_cast<__gm__ uint32_t*>(wq->headAddr), 0);
    return UrmaPollCq(session, peer, submittedBb, wq->submittedWqeCount, true, completed) && completed;
}

// Payload WQEs one notify posts. messageLen <= 256MB keeps the historical single
// WQE (and messageLen == 0 still posts one zero-length WQE); larger payloads are
// split so each WQE's uint32_t sge->len stays within the UB 256MB cap.
AICORE inline uint32_t UrmaNotifyPayloadWqeCount(uint64_t messageLen)
{
    if (messageLen <= kUrmaMaxWqeTransferBytes) {
        return 1U;
    }
    return static_cast<uint32_t>((messageLen + kUrmaMaxWqeTransferBytes - 1U) / kUrmaMaxWqeTransferBytes);
}

// Fills the signal WQE at producer index `head` (WRITE for SET, FAA otherwise)
// and flushes its cachelines. The head is not advanced and the doorbell is not
// rung; the caller commits both once, after the payload, so the signal is always
// the last WQE in the batch. Payload uses the symmetric MR token; the signal
// lives in the notify pool and uses notifyTokenId.
AICORE inline void FillNotifySignalAt(
    __gm__ UrmaWQCtx* wq, __gm__ UrmaMemInfo* remoteMem, __gm__ UrmaNotifyResourceRegion* region,
    __gm__ int32_t* remoteSignal, int32_t signalValue, bool isAdd, uint32_t setSlotIndex, uint32_t head)
{
    const uint32_t bbSize = 1U << wq->wqeShiftSize;
    __gm__ uint8_t* firstBb = reinterpret_cast<__gm__ uint8_t*>(wq->bufAddr + bbSize * (head % wq->depth));
    if (isAdd) {
        __gm__ uint8_t* secondBb = reinterpret_cast<__gm__ uint8_t*>(wq->bufAddr + bbSize * ((head + 1U) % wq->depth));
        FillFaaWqe(
            firstBb, secondBb, remoteMem, remoteSignal, &region->faaResult, signalValue, head, wq->depth,
            remoteMem->notifyTokenId);
        DcciCachelines(firstBb, kUrmaSqeSizeBytes + kUrmaSgeSizeBytes);
        DcciCachelines(secondBb, sizeof(int32_t));
        DcciCachelines(reinterpret_cast<__gm__ uint8_t*>(&region->faaResult), sizeof(region->faaResult));
        return;
    }
    region->setValues[setSlotIndex] = signalValue;
    // Strong-mark the SET signal so it cannot overtake the relax-marked payload
    // writes queued ahead of it on this jetty (0x22 = CQE | strong).
    FillTransferWqe(
        firstBb, remoteMem, reinterpret_cast<__gm__ uint8_t*>(remoteSignal),
        reinterpret_cast<__gm__ uint8_t*>(&region->setValues[setSlotIndex]), sizeof(int32_t), UrmaOpcode::WRITE, head,
        wq->depth, remoteMem->notifyTokenId, kUrmaPlaceOrderStrong);
    DcciCachelines(firstBb, kUrmaSqeSizeBytes + kUrmaSgeSizeBytes);
    DcciCachelines(reinterpret_cast<__gm__ uint8_t*>(region), kUrmaNotifyResourceCachelineBytes);
}

AICORE inline UrmaPostResult UrmaPostNotify(
    __gm__ uint8_t* remotePayload, __gm__ uint8_t* localPayload, uint64_t messageLen, __gm__ int32_t* remoteSignal,
    int32_t signalValue, NotifyOp notifyOp, const AsyncSession& session, uint32_t peer)
{
    __gm__ UrmaInfo* info = reinterpret_cast<__gm__ UrmaInfo*>(session.contextGm);
    PTO_ASSERT(info->notifyPoolPtr != 0U, "UrmaPostNotify: notify resource pool is unavailable.");

    const bool isAtomicAdd = notifyOp == NotifyOp::AtomicAdd;
    const uint32_t signalBb = isAtomicAdd ? 2U : 1U; // FAA occupies SQE + immediate BB; SET a single BB.
    __gm__ UrmaNotifyResourceRegion* region = GetUrmaNotifyResourceRegion(session, peer);
    __gm__ UrmaWQCtx* wq = GetWqContext(session, peer);
    __gm__ UrmaCqCtx* cq = GetCqContext(session, peer);
    __gm__ UrmaMemInfo* remoteMem = GetRemoteMemInfo(info, peer, session.qpIdx);

    uint32_t setSlotIndex = 0U;
    const bool slotReady = PrepareNotifySetSlot(session, peer, region, isAtomicAdd, setSlotIndex);
    PTO_ASSERT(slotReady, "UrmaPostNotify: failed to drain the SET slot before reuse.");

    // The payload is auto-split exactly like UrmaPostSend so an oversized
    // messageLen never truncates into a single WQE's uint32_t sge->len (a corrupt
    // transfer that can crash the card). The signal WQE is posted last, so a
    // receiver still observes the signal only after every payload chunk has
    // landed — one ordered SQ guarantees that regardless of doorbell batching.
    // WQEs are filled without ringing; a single doorbell announces the whole
    // notify, and we only ring early to drain when the ring cannot hold the rest.
    const uint32_t payloadWqeCount = UrmaNotifyPayloadWqeCount(messageLen);
    const uint32_t announcedHead = ld_dev(reinterpret_cast<__gm__ uint32_t*>(wq->headAddr), 0);
    uint32_t pendingHead = announcedHead;
    uint32_t submittedWqeCount = wq->submittedWqeCount;
    uint64_t offset = 0U;
    for (uint32_t i = 0U; i < payloadWqeCount; ++i) {
        if (!UrmaRingHasRoom(wq, cq, pendingHead, submittedWqeCount, 1U, 1U)) {
            if (pendingHead != announcedHead) {
                CommitPostedWqes(wq, pendingHead, submittedWqeCount);
            }
            bool completed = false;
            const bool drained =
                UrmaPollCq(session, peer, pendingHead, submittedWqeCount, true, completed) && completed;
            PTO_ASSERT(drained, "UrmaPostNotify: failed to acquire URMA queue capacity for payload.");
        }
        const uint64_t remaining = messageLen - offset;
        const uint32_t chunk =
            static_cast<uint32_t>(remaining < kUrmaMaxWqeTransferBytes ? remaining : kUrmaMaxWqeTransferBytes);
        // Relax-mark every payload WQE so the strong-marked signal WQE below cannot
        // overtake it: strong only refuses to pass preceding *relax* SQEs, so the
        // pairing (relax payload + strong signal) is what keeps the receiver from
        // seeing the signal before every chunk lands, on this one ordered jetty.
        FillTransferWqeAt(
            wq, remoteMem, remotePayload + offset, localPayload + offset, chunk, UrmaOpcode::WRITE, pendingHead,
            kUrmaPlaceOrderRelax);
        pendingHead += 1U;
        submittedWqeCount += 1U;
        offset += chunk;
    }

    // Reserve room for the signal WQE(s). Announce any filled-but-unrung payload
    // before blocking so we never wait on WQEs the hardware was never told about.
    if (!UrmaRingHasRoom(wq, cq, pendingHead, submittedWqeCount, signalBb, 1U)) {
        if (pendingHead != announcedHead) {
            CommitPostedWqes(wq, pendingHead, submittedWqeCount);
        }
        bool completed = false;
        const bool drained = UrmaPollCq(session, peer, pendingHead, submittedWqeCount, true, completed) && completed;
        PTO_ASSERT(drained, "UrmaPostNotify: failed to acquire URMA queue capacity for signal.");
    }

    // Advance the SET ring before filling the signal so the region DCCI inside
    // FillNotifySignalAt carries the updated ring state to DDR (setSlotIndex is
    // already captured, so setValues[setSlotIndex] still targets this call's slot).
    if (!isAtomicAdd) {
        region->nextSetSlot = (region->nextSetSlot + 1U) % kUrmaNotifySetSlotCount;
        region->setRingStarted = 1U;
    }
    FillNotifySignalAt(wq, remoteMem, region, remoteSignal, signalValue, isAtomicAdd, setSlotIndex, pendingHead);
    const uint32_t targetBb = pendingHead + signalBb;
    submittedWqeCount += 1U; // the signal contributes exactly one completion (FAA's second BB carries no CQE)

    // Single doorbell for the whole notify: publishes submittedWqeCount, barriers
    // every filled WQE and the SET/FAA region into DDR, then rings at targetBb.
    CommitPostedWqes(wq, targetBb, submittedWqeCount);
    return {EncodeHandle(peer, targetBb), submittedWqeCount};
}


#endif // PTO_COMM_ASYNC_URMA_POST_HPP
