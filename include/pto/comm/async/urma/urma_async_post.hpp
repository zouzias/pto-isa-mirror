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

// Jetties a message of this size is worth spreading over. Capped by the run the AIV
// owns and by how many whole 256MB WQEs the payload fills. A transfer that still
// fits in one WQE stays on the first jetty; only the second (and later) 256MB
// occupies another queue.
AICORE inline uint32_t UrmaJettiesForMessage(uint32_t qpCount, uint64_t messageLen)
{
    if (qpCount <= 1U) {
        return 1U;
    }
    const uint64_t affordable = messageLen / kUrmaMinSliceBytes;
    if (affordable <= 1U) {
        return 1U;
    }
    return (affordable < static_cast<uint64_t>(qpCount)) ? static_cast<uint32_t>(affordable) : qpCount;
}

// Bytes every slice but the last one carries. Aligning down cannot reach zero: the
// caller only asks for jettiesToUse slices once the payload holds that many whole
// kUrmaMinSliceBytes, and that minimum is itself slice-aligned.
AICORE inline uint64_t UrmaSliceBytes(uint32_t jettiesToUse, uint64_t messageLen)
{
    if (jettiesToUse <= 1U) {
        return messageLen;
    }
    return (messageLen / jettiesToUse) & ~(kUrmaSliceAlignBytes - 1ULL);
}

// Posts one contiguous slice on one jetty, splitting it across as many WQEs as the
// 256MB per-WQE cap needs, and reports the jetty's completion target. Returns false
// if the queue could not be drained to make room; the WQEs already rung stay in
// flight, which is safe because a later post on this jetty resumes from the shared
// head and retires their CQEs.
AICORE inline bool UrmaPostSliceOnJetty(
    __gm__ uint8_t* remoteAddr, __gm__ uint8_t* localAddr, uint64_t sliceLen, UrmaOpcode opcode,
    const AsyncSession& session, uint32_t peer, uint32_t jetty, uint32_t& targetBb, uint32_t& targetCqe)
{
    __gm__ UrmaInfo* info = reinterpret_cast<__gm__ UrmaInfo*>(session.contextGm);
    __gm__ UrmaWQCtx* wq = GetWqContextAt(session, peer, jetty);
    __gm__ UrmaCqCtx* cq = GetCqContextAt(session, peer, jetty);
    // tpn, eidAddr and the local token belong to the jetty-peer connection rather
    // than to the peer, so the remote mem row follows the jetty.
    __gm__ UrmaMemInfo* remoteMem = GetRemoteMemInfo(info, peer, jetty);

    const uint32_t announcedHead =
        ld_dev(reinterpret_cast<__gm__ uint32_t*>(wq->headAddr), 0); // head the hardware already knows
    uint32_t pendingHead = announcedHead; // head of WQEs filled but not yet rung in this call
    uint32_t submittedWqeCount = wq->submittedWqeCount;
    uint64_t offset = 0U;
    while (offset < sliceLen) {
        // Ring full: before blocking on completions, announce whatever we have
        // filled but not yet rung (otherwise we would wait on WQEs the hardware was
        // never told about — deadlock). If nothing new is pending, the outstanding
        // WQEs were already rung by earlier calls, so just drain.
        if (!UrmaRingHasRoom(wq, cq, pendingHead, submittedWqeCount, 1U, 1U)) {
            if (pendingHead != announcedHead) {
                CommitPostedWqes(wq, pendingHead, submittedWqeCount);
            }
            bool completed = false;
            if (!UrmaPollCqAt(session, peer, jetty, pendingHead, submittedWqeCount, true, completed) || !completed) {
                return false;
            }
        }

        const uint64_t remaining = sliceLen - offset;
        const uint32_t chunk =
            static_cast<uint32_t>(remaining < kUrmaMaxWqeTransferBytes ? remaining : kUrmaMaxWqeTransferBytes);
        FillTransferWqeAt(wq, remoteMem, remoteAddr + offset, localAddr + offset, chunk, opcode, pendingHead);
        pendingHead += 1U;
        submittedWqeCount += 1U;
        offset += chunk;
    }
    CommitPostedWqes(wq, pendingHead, submittedWqeCount);
    targetBb = pendingHead;
    targetCqe = submittedWqeCount;
    return true;
}

AICORE inline UrmaMultiPostResult UrmaPostSend(
    __gm__ uint8_t* remoteAddr, __gm__ uint8_t* localAddr, uint64_t messageLen, UrmaOpcode opcode,
    const AsyncSession& session, uint32_t peer)
{
    UrmaMultiPostResult result{};
    result.jettyBase = session.qpIdxBase;

    // Checked on every build, not under PTO_ASSERT: without it a rejected session
    // walks past the context tables and the zeroed self-peer row divides by zero.
    if (!ValidateUrmaSession(session, peer) || (opcode != UrmaOpcode::WRITE && opcode != UrmaOpcode::READ)) {
        return result;
    }

    // Zero-length transfer is a no-op: handle 0 tells Wait/Test there is nothing
    // outstanding, matching the SDMA/RDMA engines.
    if (messageLen == 0U) {
        return result;
    }

    // The payload becomes one contiguous slice per jetty, each posted and rung on
    // its own queue so the hardware drains them in parallel. A slice is itself
    // split across WQEs where it exceeds the 256MB cap, because a single WQE's
    // sge->len is a uint32_t and truncating an oversized length would issue a
    // corrupt transfer that can crash the card.
    const uint32_t jettiesToUse = UrmaJettiesForMessage(session.qpCount, messageLen);
    const uint64_t sliceBytes = UrmaSliceBytes(jettiesToUse, messageLen);

    for (uint32_t slot = 0U; slot < jettiesToUse; ++slot) {
        const uint64_t sliceOffset = static_cast<uint64_t>(slot) * sliceBytes;
        // The last jetty absorbs the remainder left by aligning the slice size down.
        const uint64_t sliceLen = (slot + 1U == jettiesToUse) ? (messageLen - sliceOffset) : sliceBytes;
        if (!UrmaPostSliceOnJetty(
                remoteAddr + sliceOffset, localAddr + sliceOffset, sliceLen, opcode, session, peer,
                session.qpIdxBase + slot, result.targetBb[slot], result.targetCqe[slot])) {
            // Earlier jetties keep draining on their own. Fail the whole put
            // rather than hand back a completion target for half a transfer.
            result.jettyCount = 0U;
            result.handle = 0U;
            return result;
        }
    }
    result.jettyCount = jettiesToUse;
    // targetBb[0] counted at least one WQE, so the handle is never the "nothing
    // outstanding" value 0.
    result.handle = EncodeHandle(peer, result.targetBb[0]);
    return result;
}

// SET reuses a small ring of value slots inside the notify region; FAA never
// touches it. When we are about to reuse slot 0 while an earlier SET WQE may
// still reference it, drain the outstanding WQEs first so the in-flight transfer
// reads the old value before we overwrite it. Returns the slot this call owns.
AICORE inline bool PrepareNotifySetSlot(
    const AsyncSession& session, uint32_t peer, __gm__ UrmaNotifyResourceRegion* region, bool isAdd,
    uint32_t& setSlotIndex)
{
    // Read back from device memory, so bound it rather than trust it.
    if (!isAdd && region->nextSetSlot >= kUrmaNotifySetSlotCount) {
        return false;
    }
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
    UrmaPostResult result{};

    // Notify only touches the run's first jetty, so that is all it validates. This
    // also rejects a null contextGm, so it must run before info is dereferenced.
    if (!ValidateUrmaSessionAt(session, peer, session.qpIdxBase)) {
        return result;
    }
    __gm__ UrmaInfo* info = reinterpret_cast<__gm__ UrmaInfo*>(session.contextGm);
    if (info->notifyPoolPtr == 0U) {
        return result;
    }

    const bool isAtomicAdd = notifyOp == NotifyOp::AtomicAdd;
    const uint32_t signalBb = isAtomicAdd ? 2U : 1U; // FAA occupies SQE + immediate BB; SET a single BB.
    // One TPUT_ASYNC_NOTIFY is one ordered SQ: every payload WQE and the signal
    // stay on session.qpIdxBase. placeOrder cannot order across jetties, so this
    // path never calls UrmaJettiesForMessage. A preceding TPUT_ASYNC that spread
    // onto other jetties is a different post — Wait that event before notifying,
    // or send the whole payload through this function (it will WQE-split past
    // 256MB on this same jetty).
    const uint32_t jetty = session.qpIdxBase;
    __gm__ UrmaNotifyResourceRegion* region = GetUrmaNotifyResourceRegion(session, peer);
    __gm__ UrmaWQCtx* wq = GetWqContextAt(session, peer, jetty);
    __gm__ UrmaCqCtx* cq = GetCqContextAt(session, peer, jetty);
    __gm__ UrmaMemInfo* remoteMem = GetRemoteMemInfo(info, peer, jetty);

    uint32_t setSlotIndex = 0U;
    if (!PrepareNotifySetSlot(session, peer, region, isAtomicAdd, setSlotIndex)) {
        return result;
    }

    // Oversized payloads still split into <=256MB WQEs, but only on this jetty.
    // The signal is the last WQE; one doorbell announces the whole notify. We
    // only ring early to drain when the ring cannot hold the rest.
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
            if (!UrmaPollCq(session, peer, pendingHead, submittedWqeCount, true, completed) || !completed) {
                return result;
            }
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
        if (!UrmaPollCq(session, peer, pendingHead, submittedWqeCount, true, completed) || !completed) {
            return result;
        }
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
    result.handle = EncodeHandle(peer, targetBb);
    result.targetCqe = submittedWqeCount;
    return result;
}

#endif // PTO_COMM_ASYNC_URMA_POST_HPP
