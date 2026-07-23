/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// AIV progress for fused CCU Reduce+Broadcast (one persistent kernel).
// Default: local groupDone → peer TNOTIFY(groupReady) → owner TriggerProgressCke.
// Escape (USE_PROGRESS=0): readyQueue drain + TNOTIFY/TTEST → TriggerProgressCke.

#ifndef PIPE_FIX
#define PIPE_FIX static_cast<pipe_t>(10)
#endif

#include <pto/pto-inst.hpp>
#include <pto/common/pto_tile.hpp>
#include <pto/comm/pto_comm_inst.hpp>
#include "config.h"
#include "comm_context.h"
#include "ready_queue.hpp"
#include "kernel_launchers.h"

using namespace pto;

template <typename T>
AICORE inline __gm__ T *CommRemotePtr(__gm__ CommDeviceContext *ctx, __gm__ T *localPtr, int pe)
{
    uint64_t localBase = ctx->windowsIn[ctx->rankId];
    uint64_t offset = (uint64_t)localPtr - localBase;
    return (__gm__ T *)(ctx->windowsIn[pe] + offset);
}

static constexpr uint64_t kCkeValidBit = 1ULL << 63;
static constexpr uint32_t kPollFenceInterval = 64;
static constexpr uint32_t kSchedMaxTiles = 2048;
static constexpr uint32_t kSchedReadyWords = kSchedMaxTiles / 32;
// Must stay 1: progress CKE ping-pong is depth-1 (edge-triggered).
static constexpr uint32_t kSchedMaxInflight = 1;

// ProgressCtx and ProgressCkeCtx are defined in kernel_launchers.h.

AICORE inline volatile __gm__ uint64_t *CkeItemsDone(ProgressCkeCtx *cke)
{
    return reinterpret_cast<volatile __gm__ uint64_t *>(cke->itemsDoneAddr);
}

AICORE inline void TriggerOneCke(uint64_t ckeVA, uint32_t mask)
{
    // Match v3 progress trigger: scalar store + post-dcci + dsb. Avoid pre-dcci /
    // PIPE_ALL here — they have been observed to break CKE ping-pong reuse
    // (rsItemsDone sticks after the second cycle on the same slot).
    volatile __gm__ uint64_t *ckePtr = reinterpret_cast<volatile __gm__ uint64_t *>(ckeVA);
    *ckePtr = static_cast<uint64_t>(mask) | kCkeValidBit;
    dcci(reinterpret_cast<__gm__ void *>(ckeVA), SINGLE_CACHE_LINE);
    dsb(DSB_DDR);
}

AICORE inline uint64_t SchedGetSysCnt()
{
    uint64_t syscnt = 0;
    asm volatile("MOV %0, SYS_CNT\n" : "+l"(syscnt));
    return syscnt;
}

AICORE inline void SchedTimingAdd(volatile __gm__ uint64_t *slot, uint64_t delta)
{
    if (slot == nullptr || delta == 0)
        return;
    *slot += delta;
    dcci((__gm__ void *)slot, SINGLE_CACHE_LINE);
}

AICORE inline void SchedTimingInc(volatile __gm__ uint64_t *slot)
{
    if (slot == nullptr)
        return;
    *slot += 1;
    dcci((__gm__ void *)slot, SINGLE_CACHE_LINE);
}

AICORE inline int32_t PollAllQueues(volatile __gm__ MultiBlockQueueSet *qset, int numBlocks, int32_t *heads,
                                    int *nextQueueOffset)
{
    for (int pass = 0; pass < numBlocks; ++pass) {
        int q = (*nextQueueOffset + pass) % numBlocks;
        volatile __gm__ PerBlockQueue *pq = GetMyBlockQueue(qset, q);
        int32_t tile = PerBlockQueueTryDequeue(pq, heads[q]);
        if (tile >= 0) {
            heads[q]++;
            *nextQueueOffset = (q + 1) % numBlocks;
            return tile;
        }
    }
    return -1;
}

AICORE inline void MarkSchedTileReady(uint32_t *readyWords, uint32_t tile)
{
    readyWords[tile >> 5] |= 1u << (tile & 31);
}

AICORE inline bool IsSchedTileReady(const uint32_t *readyWords, uint32_t tile)
{
    return (readyWords[tile >> 5] & (1u << (tile & 31))) != 0;
}

AICORE inline uint64_t SchedPackedOffsetHalves(uint32_t tile, uint32_t sub, uint32_t rankSize)
{
    const uint32_t packedTile = CcuPackedTileIndex(tile, G_NUM_TILES, rankSize);
    return static_cast<uint64_t>(packedTile) * G_BASE_M * G_BASE_N +
           static_cast<uint64_t>(sub) * G_COMM_SUB_M * G_BASE_N;
}

AICORE inline uint64_t SchedRowMajorOffsetHalves(uint32_t tile, uint32_t sub)
{
    const uint32_t mi = tile / G_N_TILES;
    const uint32_t ni = tile % G_N_TILES;
    const uint64_t tileBase = static_cast<uint64_t>(mi) * G_BASE_M * G_N + static_cast<uint64_t>(ni) * G_BASE_N;
    return tileBase + static_cast<uint64_t>(sub) * G_COMM_SUB_M * G_N;
}

AICORE inline void TriggerProgressCke(ProgressCkeCtx *cke, uint64_t nextIssued)
{
    uint32_t parity = static_cast<uint32_t>(nextIssued & 1);
    TriggerOneCke(cke->ckeSlotVA[parity], cke->ckeMask[parity]);
}

AICORE inline void WaitItemsDoneGE(volatile __gm__ uint64_t *itemsDonePtr, uint64_t target)
{
    uint32_t spin = 0;
    while (true) {
        dcci((__gm__ void *)itemsDonePtr, SINGLE_CACHE_LINE);
        if (*itemsDonePtr >= target)
            break;
        if ((++spin % kPollFenceInterval) == 0) {
            dsb(DSB_DDR);
            pipe_barrier(PIPE_ALL);
        }
    }
}

AICORE inline void WaitItemsDoneGE(volatile __gm__ uint64_t *itemsDonePtr, uint64_t target,
                                   volatile __gm__ uint64_t *timingSlot, bool timingOn)
{
    const uint64_t t0 = timingOn ? SchedGetSysCnt() : 0;
    WaitItemsDoneGE(itemsDonePtr, target);
    if (timingOn)
        SchedTimingAdd(timingSlot, SchedGetSysCnt() - t0);
}

AICORE inline bool CkeCanPublish(volatile __gm__ uint64_t *itemsDonePtr, uint64_t issued)
{
    dcci((__gm__ void *)itemsDonePtr, SINGLE_CACHE_LINE);
    const uint64_t done = *itemsDonePtr;
    if (done > issued)
        return false;
    return (issued - done) < kSchedMaxInflight;
}

// CCU progress CKEs are edge-triggered on a ping-pong pair. Before publishing
// item N, CCU must have finished items 0..N-1 so it is already back in WaitEvent
// on the next slot. Firing with depth>1 can lose the edge and freeze rsItemsDone.
AICORE inline void WaitCkeCanPublish(volatile __gm__ uint64_t *itemsDonePtr, uint64_t issued,
                                     volatile __gm__ uint64_t *timingSlot, bool timingOn)
{
    const uint64_t t0 = timingOn ? SchedGetSysCnt() : 0;
    uint32_t spin = 0;
    while (!CkeCanPublish(itemsDonePtr, issued)) {
        if ((++spin % kPollFenceInterval) == 0) {
            dsb(DSB_DDR);
            pipe_barrier(PIPE_ALL);
        }
    }
    if (timingOn)
        SchedTimingAdd(timingSlot, SchedGetSysCnt() - t0);
}

// ============================================================================
// readyQueue escape path (USE_PROGRESS=0): drain + TNOTIFY + trigger fused CKE
// ============================================================================

AICORE inline void RunRsScheduler(ProgressCkeCtx *rsCke, volatile __gm__ MultiBlockQueueSet *qset,
                                 volatile __gm__ ProgressCtx *ctx, volatile __gm__ int32_t *signal,
                                 __gm__ CommDeviceContext *commCtx, uint32_t safeRanks, uint32_t numTiles,
                                 uint32_t myRank)
{
    const bool timingOn = (ctx->schedTimingEnable != 0);
    volatile __gm__ ProgressCtx::Timing *timing = &ctx->rsSchedTiming;
    const uint64_t kernelT0 = timingOn ? SchedGetSysCnt() : 0;

    // Pre-cache remote signal base per rank
    __gm__ int32_t *remoteSignalBase[MAX_RANKS];
    {
        __gm__ int32_t *signalBase = const_cast<__gm__ int32_t *>(signal);
        for (uint32_t r = 0; r < safeRanks && r < MAX_RANKS; ++r)
            remoteSignalBase[r] = (r != myRank) ? CommRemotePtr(commCtx, signalBase, static_cast<int>(r)) : signalBase;
    }

    set_st_atomic_cfg(ATOMIC_S32, ATOMIC_SUM);

    const int numBlocks = qset->num_blocks;
    int32_t heads[MAX_COMPUTE_BLOCKS];
    for (int i = 0; i < numBlocks; ++i)
        heads[i] = 0;

    uint32_t localReadyWords[kSchedReadyWords];
    uint32_t notifyDoneWords[kSchedReadyWords];
    uint32_t reduceDoneWords[kSchedReadyWords];
    for (uint32_t i = 0; i < kSchedReadyWords; ++i) {
        localReadyWords[i] = 0;
        notifyDoneWords[i] = 0;
        reduceDoneWords[i] = 0;
    }

    const uint32_t myOwnerTiles = CcuOwnerTileCount(myRank, numTiles, safeRanks);
    const uint32_t myOwnerGroups = CcuOwnerGroupCount(myRank, numTiles, safeRanks);
    uint32_t totalNotifyGroups = 0;
    for (uint32_t r = 0; r < safeRanks; ++r) {
        if (r != myRank)
            totalNotifyGroups += CcuOwnerGroupCount(r, numTiles, safeRanks);
    }

    uint32_t notifyDone = 0;
    uint32_t reduceDone = 0;
    uint64_t rsNextIssued = 0;
    int nextQueueOffset = 0;

    const int32_t peerTarget = static_cast<int32_t>(safeRanks) - 1;

    // Wait until CCU has passed WaitEvent(gate) (rsKernelReady != 0) before progress CKEs.
    if (rsCke->kernelReadyAddr != 0) {
        volatile __gm__ uint64_t *readyPtr = reinterpret_cast<volatile __gm__ uint64_t *>(rsCke->kernelReadyAddr);
        uint32_t readySpin = 0;
        while (*readyPtr == 0) {
            dcci((__gm__ void *)readyPtr, SINGLE_CACHE_LINE);
            if ((++readySpin % kPollFenceInterval) == 0) {
                dsb(DSB_DDR);
                pipe_barrier(PIPE_ALL);
            }
        }
    }

    uint32_t idleSpin = 0;

    // Phase 1: drain + notify + fire fused progress CKE
    while (notifyDone < totalNotifyGroups || reduceDone < myOwnerGroups) {
        // Step 1: drain readyQueue
        int32_t tileId = PollAllQueues(qset, numBlocks, heads, &nextQueueOffset);
        if (tileId >= 0 && static_cast<uint32_t>(tileId) < numTiles)
            MarkSchedTileReady(localReadyWords, static_cast<uint32_t>(tileId));

        bool madeProgress = (tileId >= 0);

        // Step 2: check non-self owner groups, send TNOTIFY.
        // Must run even when step 1 drained a tile: with ccu_group_tiles>1 a rank can
        // TWAIT on peer notify while the peer keeps skipping this step (!madeProgress),
        // deadlocking warmup/benchmark (seen at group=8 with full readyQueue).
        if (notifyDone < totalNotifyGroups) {
            for (uint32_t owner = 0; owner < safeRanks; ++owner) {
                if (owner == myRank)
                    continue;

                const uint32_t ownerTiles = CcuOwnerTileCount(owner, numTiles, safeRanks);
                const uint32_t ownerGroups = CcuOwnerGroupCount(owner, numTiles, safeRanks);
                for (uint32_t g = 0; g < ownerGroups; ++g) {
                    const uint32_t localStart = g * G_COMM_GROUP_TILES;
                    if (localStart >= ownerTiles)
                        continue;

                    const uint32_t firstTile = owner + localStart * safeRanks;
                    if (IsSchedTileReady(notifyDoneWords, firstTile))
                        continue;

                    uint32_t groupTiles = ownerTiles - localStart;
                    if (groupTiles > G_COMM_GROUP_TILES)
                        groupTiles = G_COMM_GROUP_TILES;

                    bool allReady = true;
                    for (uint32_t j = 0; j < groupTiles; ++j) {
                        const uint32_t t = owner + (localStart + j) * safeRanks;
                        if (!IsSchedTileReady(localReadyWords, t)) {
                            allReady = false;
                            break;
                        }
                    }
                    if (!allReady)
                        continue;

                    // All tiles in this group are locally ready — notify the owner.
                    // Use TNOTIFY (same as v3), not raw st_atomic to a remote VA.
                    const uint64_t nt0 = timingOn ? SchedGetSysCnt() : 0;
                    pto::comm::Signal sig(remoteSignalBase[owner] + G_SIGNAL_SUBTILE_READY_OFFSET + localStart);
                    pto::comm::TNOTIFY(sig, 1, pto::comm::NotifyOp::AtomicAdd);
                    if (timingOn)
                        SchedTimingAdd(&timing->notifyCycles, SchedGetSysCnt() - nt0);

                    MarkSchedTileReady(notifyDoneWords, firstTile);
                    notifyDone++;
                    madeProgress = true;
                    break;
                }
            }
        }

        // Step 3: fire next own-owner group in order (required for CCU seqOffset).
        if (reduceDone < myOwnerGroups) {
            const uint32_t g = reduceDone;
            const uint32_t localStart = g * G_COMM_GROUP_TILES;
            const uint32_t firstTile = myRank + localStart * safeRanks;

            if (localStart < myOwnerTiles && !IsSchedTileReady(reduceDoneWords, firstTile)) {
                uint32_t groupTiles = myOwnerTiles - localStart;
                if (groupTiles > G_COMM_GROUP_TILES)
                    groupTiles = G_COMM_GROUP_TILES;

                bool allLocalReady = true;
                for (uint32_t j = 0; j < groupTiles; ++j) {
                    const uint32_t t = myRank + (localStart + j) * safeRanks;
                    if (!IsSchedTileReady(localReadyWords, t)) {
                        allLocalReady = false;
                        break;
                    }
                }

                pto::comm::Signal counterSig(
                    const_cast<__gm__ int32_t *>(signal + G_SIGNAL_SUBTILE_READY_OFFSET + localStart));
                if (allLocalReady && pto::comm::TTEST(counterSig, peerTarget, pto::comm::WaitCmp::GE)) {
                    WaitCkeCanPublish(CkeItemsDone(rsCke), rsNextIssued, nullptr, false);
                    if (rsNextIssued >= 1) {
                        WaitItemsDoneGE(CkeItemsDone(rsCke), rsNextIssued, &timing->backpressureCycles, timingOn);
                    }

                    // Fused CCU uses sequential group offsets (no per-item work-ring).
                    TriggerProgressCke(rsCke, rsNextIssued);
                    if (timingOn)
                        SchedTimingInc(&timing->triggerCount);
                    rsNextIssued++;
                    reduceDone++;

                    MarkSchedTileReady(reduceDoneWords, firstTile);
                    madeProgress = true;
                }
            }

            // Never TWAIT here: both ranks can enter TWAIT for their first owned group
            // simultaneously and deadlock (rsItemsDone=0, full readyQueue). Spin only;
            // the peer rank keeps running step-2 notify on its AIV block.
        }

        if (!madeProgress) {
            // Throttle heavy fences on idle spins — every empty poll used to pay
            // PIPE_ALL and delayed TNOTIFY / CKE fire.
            if ((++idleSpin % kPollFenceInterval) == 0) {
                dsb(DSB_DDR);
                pipe_barrier(PIPE_ALL);
            }
            if (timingOn)
                SchedTimingInc(&timing->idleSpinCycles);
        } else {
            idleSpin = 0;
        }
    }

    // Phase 2: wait for all fused items to complete
    if (rsNextIssued > 0) {
        WaitItemsDoneGE(CkeItemsDone(rsCke), rsNextIssued);
    }

    if (timingOn)
        SchedTimingAdd(&timing->totalCycles, SchedGetSysCnt() - kernelT0);
}

// ============================================================================
// Kernel entry: readyQueue escape path (USE_PROGRESS=0).
// ============================================================================
__global__ AICORE void ccu_gemm_ar_scheduler_kernel(ProgressCkeCtx rsCke, __gm__ uint8_t *readyQueue,
                                                       __gm__ uint8_t *progressCtx, __gm__ uint8_t *signal_matrix,
                                                       __gm__ uint8_t *commCtxRaw, uint32_t rankSize, uint32_t myRank)
{
    if (get_block_idx() != 0)
        return;

    volatile __gm__ MultiBlockQueueSet *qset = reinterpret_cast<volatile __gm__ MultiBlockQueueSet *>(readyQueue);
    volatile __gm__ ProgressCtx *ctx = reinterpret_cast<volatile __gm__ ProgressCtx *>(progressCtx);
    volatile __gm__ int32_t *signal = reinterpret_cast<volatile __gm__ int32_t *>(signal_matrix);
    __gm__ CommDeviceContext *commCtx = reinterpret_cast<__gm__ CommDeviceContext *>(commCtxRaw);

    dcci((__gm__ void *)ctx, SINGLE_CACHE_LINE);
    pipe_barrier(PIPE_ALL);

    const uint32_t numTiles = ctx->numTiles;
    const uint32_t safeRanks = (rankSize > 0) ? rankSize : 1;
    RunRsScheduler(&rsCke, qset, ctx, signal, commCtx, safeRanks, numTiles, myRank);
}

void launchCcuGemmArScheduler(void *stream, ProgressCkeCtx rsCke, uint8_t *readyQueue, uint8_t *progressCtx,
                                uint8_t *signal_matrix, uint8_t *commCtx, uint32_t rankSize, uint32_t myRank)
{
    ccu_gemm_ar_scheduler_kernel<<<1, nullptr, stream>>>(rsCke, readyQueue, progressCtx, signal_matrix, commCtx,
                                                            rankSize, myRank);
}

// ============================================================================
// Progress kernel (single AIV block) — v3-style groupReady TNOTIFY, owner-scoped.
//
// AIC (every rank): AtomicAdd(local groupDone[flat(owner,g)]) after each tile.
// Every rank AIV: for each (owner,g) in owner-major order:
//   1) wait LOCAL groupDone[flat] >= expected
//   2) if peer: TNOTIFY(owner groupReady[flat], +1)
//   3) if owner: wait local groupReady >= P-1, then depth-1 BP + TriggerProgressCke
// ============================================================================

static constexpr uint32_t kGroupDonePollFenceInterval = 256;

AICORE inline void PollGroupDoneFence(uint32_t &spin)
{
    if ((++spin % kGroupDonePollFenceInterval) == 0) {
        dsb(DSB_DDR);
        pipe_barrier(PIPE_ALL);
    }
}

AICORE inline void WaitLocalGroupDone(__gm__ int32_t *groupDoneBase, uint32_t flatGroupIdx, int32_t expected,
                                      volatile __gm__ uint64_t *timingSlot, bool timingOn)
{
    volatile __gm__ int32_t *done = groupDoneBase + flatGroupIdx * kGroupDoneStride;
    const uint64_t t0 = timingOn ? SchedGetSysCnt() : 0;
    uint32_t spin = 0;
    while (true) {
        dcci((__gm__ void *)done, SINGLE_CACHE_LINE);
        if (*done >= expected)
            break;
        PollGroupDoneFence(spin);
    }
    if (timingOn)
        SchedTimingAdd(timingSlot, SchedGetSysCnt() - t0);
}

AICORE inline void NotifyOwnerGroupReady(__gm__ int32_t *remoteSignalBase[], uint32_t owner, uint32_t flatGroupIdx,
                                         volatile __gm__ uint64_t *timingSlot, bool timingOn)
{
    const uint64_t t0 = timingOn ? SchedGetSysCnt() : 0;
    pto::comm::Signal sig(remoteSignalBase[owner] + G_SIGNAL_GROUP_READY_OFFSET + flatGroupIdx * kGroupReadyStride);
    pto::comm::TNOTIFY(sig, 1, pto::comm::NotifyOp::AtomicAdd);
    if (timingOn)
        SchedTimingAdd(timingSlot, SchedGetSysCnt() - t0);
}

// Owner: fuse peer groupReady + depth-1 itemsDone (+ ring). Attribute GD→notifyCycles
// (peer wait), BP→backpressureCycles, ring→idleSpinCycles.
AICORE inline void WaitPeerReadyAndBackpressure(__gm__ int32_t *signalBase, uint32_t flatGroupIdx, int32_t peerTarget,
                                                volatile __gm__ uint64_t *itemsDonePtr, uint64_t issued,
                                                volatile __gm__ ProgressCtx::Timing *timing, bool timingOn)
{
    pto::comm::Signal readySig(signalBase + G_SIGNAL_GROUP_READY_OFFSET + flatGroupIdx * kGroupReadyStride);
    const bool needBp = (issued >= 1);
    const uint64_t bpTarget = needBp ? issued : 0;
    uint32_t spin = 0;

    while (true) {
        const uint64_t t0 = timingOn ? SchedGetSysCnt() : 0;
        const bool peerReady =
            (peerTarget <= 0) || pto::comm::TTEST(readySig, peerTarget, pto::comm::WaitCmp::GE);

        bool bpReady = true;
        if (needBp) {
            dcci((__gm__ void *)itemsDonePtr, SINGLE_CACHE_LINE);
            bpReady = (*itemsDonePtr >= bpTarget);
        }
        const bool ringReady = CkeCanPublish(itemsDonePtr, issued);

        if (peerReady && bpReady && ringReady)
            break;

        if (timingOn) {
            const uint64_t dt = SchedGetSysCnt() - t0;
            if (!peerReady)
                SchedTimingAdd(&timing->notifyCycles, dt);
            else if (!bpReady)
                SchedTimingAdd(&timing->backpressureCycles, dt);
            else
                SchedTimingAdd(&timing->idleSpinCycles, dt);
        }
        PollGroupDoneFence(spin);
    }
}

AICORE inline void ClearProgressTiming(volatile __gm__ ProgressCtx::Timing *timing)
{
    timing->totalCycles = 0;
    timing->notifyCycles = 0;
    timing->backpressureCycles = 0;
    timing->idleSpinCycles = 0;
    timing->triggerCount = 0;
    timing->groupDoneCycles = 0;
    timing->kernelReadyCycles = 0;
    timing->finalWaitCycles = 0;
    dcci((__gm__ void *)timing, ENTIRE_DATA_CACHE);
    pipe_barrier(PIPE_ALL);
}

__global__ AICORE void ccu_gemm_ar_progress_kernel(ProgressCkeCtx rsCke, __gm__ int32_t *groupDone,
                                                      __gm__ uint8_t *progressCtx, __gm__ uint8_t *signalRaw,
                                                      __gm__ uint8_t *commCtxRaw, uint32_t numTiles, uint32_t rankSize,
                                                      uint32_t myRank)
{
    if (get_block_idx() != 0)
        return;

    __gm__ CommDeviceContext *commCtx = reinterpret_cast<__gm__ CommDeviceContext *>(commCtxRaw);
    __gm__ int32_t *signalBase = reinterpret_cast<__gm__ int32_t *>(signalRaw);
    volatile __gm__ ProgressCtx *ctx = reinterpret_cast<volatile __gm__ ProgressCtx *>(progressCtx);
    if (ctx != nullptr) {
        dcci((__gm__ void *)ctx, SINGLE_CACHE_LINE);
        pipe_barrier(PIPE_ALL);
    }
    const bool timingOn = (ctx != nullptr && ctx->schedTimingEnable != 0);
    volatile __gm__ ProgressCtx::Timing *timing = timingOn ? &ctx->rsSchedTiming : nullptr;
    const uint64_t kernelT0 = timingOn ? SchedGetSysCnt() : 0;
    if (timingOn)
        ClearProgressTiming(timing);

    const uint32_t safeRanks = (rankSize > 0) ? rankSize : 1;
    __gm__ int32_t *remoteSignalBase[MAX_RANKS];
    for (uint32_t r = 0; r < safeRanks && r < MAX_RANKS; ++r)
        remoteSignalBase[r] =
            (r != myRank) ? CommRemotePtr(commCtx, signalBase, static_cast<int>(r)) : signalBase;

    if (safeRanks > 1)
        set_st_atomic_cfg(ATOMIC_S32, ATOMIC_SUM);

    uint64_t issued = 0;

    if (rsCke.kernelReadyAddr != 0) {
        volatile __gm__ uint64_t *rp = reinterpret_cast<volatile __gm__ uint64_t *>(rsCke.kernelReadyAddr);
        const uint64_t tReady0 = timingOn ? SchedGetSysCnt() : 0;
        uint32_t readySpin = 0;
        while (*rp == 0) {
            dcci((__gm__ void *)rp, SINGLE_CACHE_LINE);
            if ((++readySpin % kPollFenceInterval) == 0) {
                dsb(DSB_DDR);
                pipe_barrier(PIPE_ALL);
            }
        }
        if (timingOn)
            SchedTimingAdd(&timing->kernelReadyCycles, SchedGetSysCnt() - tReady0);
    }

    // Owner-major walk: peers notify; only the owner fires CKE (Pull-RS mission order).
    const int32_t peerTarget = static_cast<int32_t>(safeRanks) - 1;
    for (uint32_t owner = 0; owner < safeRanks && owner < static_cast<uint32_t>(MAX_RANKS); ++owner) {
        const uint32_t ownerGroups = CcuOwnerGroupCount(owner, numTiles, safeRanks);
        for (uint32_t g = 0; g < ownerGroups; ++g) {
            const int32_t expected =
                static_cast<int32_t>(CcuOwnerGroupTilesInGroup(owner, g, numTiles, safeRanks));
            if (expected <= 0)
                continue;

            const uint32_t flat = CcuOwnerGroupFlatIndex(owner, g, numTiles, safeRanks);
            WaitLocalGroupDone(groupDone, flat, expected, timingOn ? &timing->groupDoneCycles : nullptr, timingOn);

            if (myRank != owner) {
                NotifyOwnerGroupReady(remoteSignalBase, owner, flat, timingOn ? &timing->notifyCycles : nullptr,
                                      timingOn);
                continue;
            }

            WaitPeerReadyAndBackpressure(signalBase, flat, peerTarget, CkeItemsDone(&rsCke), issued, timing, timingOn);
            TriggerProgressCke(&rsCke, issued);
            if (timingOn)
                SchedTimingInc(&timing->triggerCount);
            issued++;
        }
    }

    if (issued > 0)
        WaitItemsDoneGE(CkeItemsDone(&rsCke), issued, timingOn ? &timing->finalWaitCycles : nullptr, timingOn);

    if (timingOn)
        SchedTimingAdd(&timing->totalCycles, SchedGetSysCnt() - kernelT0);
}

void launchCcuGemmArProgress(void *stream, ProgressCkeCtx rsCke, int32_t *groupDone, uint8_t *progressCtx,
                               uint8_t *signal_matrix, uint8_t *commCtx, uint32_t numTiles, uint32_t rankSize,
                               uint32_t myRank)
{
    ccu_gemm_ar_progress_kernel<<<1, nullptr, stream>>>(rsCke, groupDone, progressCtx, signal_matrix, commCtx,
                                                           numTiles, rankSize, myRank);
}

// Unpack kernel: convert packed reduced output back to row-major
using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
using SchedGlobal = pto::GlobalTensor<half, ShapeDyn, StrideDyn, pto::Layout::ND>;
using SchedSubTile = pto::Tile<pto::TileType::Vec, half, G_COMM_SUB_M, G_BASE_N, pto::BLayout::RowMajor, -1, -1>;

__global__ AICORE void ccu_gemm_ar_unpack_kernel(__gm__ uint8_t *packed_output, __gm__ uint8_t *row_output,
                                                    uint32_t subtilesPerTile, uint32_t rankSize)
{
    const uint32_t idx = static_cast<uint32_t>(get_block_idx());
    const uint32_t total = G_NUM_TILES * subtilesPerTile;
    if (idx >= total)
        return;

    const uint32_t tile = idx / subtilesPerTile;
    const uint32_t sub = idx % subtilesPerTile;
    const uint64_t packedOff = SchedPackedOffsetHalves(tile, sub, rankSize);
    const uint64_t rowOff = SchedRowMajorOffsetHalves(tile, sub);

    SchedSubTile tileBuf(G_COMM_SUB_M, G_BASE_N);
    TASSIGN(tileBuf, 0x0);
    ShapeDyn shape(1, 1, 1, G_COMM_SUB_M, G_BASE_N);
    StrideDyn packedStride(static_cast<int>(G_BASE_M * G_BASE_N), static_cast<int>(G_BASE_M * G_BASE_N),
                           static_cast<int>(G_BASE_M * G_BASE_N), static_cast<int>(G_BASE_N), 1);
    StrideDyn rowStride(static_cast<int>(G_BASE_M * G_N), static_cast<int>(G_BASE_M * G_N),
                        static_cast<int>(G_BASE_M * G_N), static_cast<int>(G_N), 1);
    SchedGlobal srcG(reinterpret_cast<__gm__ half *>(packed_output) + packedOff, shape, packedStride);
    SchedGlobal dstG(reinterpret_cast<__gm__ half *>(row_output) + rowOff, shape, rowStride);

    TLOAD(tileBuf, srcG);
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    TSTORE(dstG, tileBuf);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
}

void launchCcuGemmArUnpack(uint8_t *packed_output, uint8_t *row_output, void *stream, uint32_t subtilesPerTile,
                             uint32_t rankSize)
{
    ccu_gemm_ar_unpack_kernel<<<G_NUM_TILES * subtilesPerTile, nullptr, stream>>>(packed_output, row_output,
                                                                                     subtilesPerTile, rankSize);
}

__global__ AICORE void ccu_gemm_ar_gate_trigger_kernel(uint64_t gateCkeVA, uint32_t gateMask)
{
    if (get_block_idx() != 0)
        return;
    // Same light poke as progress CKE — heavy ENTIRE_DATA_CACHE/PIPE_ALL is unnecessary
    // once DieId/Id are published post-Translate.
    TriggerOneCke(gateCkeVA, gateMask);
}

int launchCcuGemmArGateTrigger(void *stream, uint64_t gateCkeVA, uint32_t gateMask)
{
    ccu_gemm_ar_gate_trigger_kernel<<<1, nullptr, stream>>>(gateCkeVA, gateMask);
    return 0;
}
