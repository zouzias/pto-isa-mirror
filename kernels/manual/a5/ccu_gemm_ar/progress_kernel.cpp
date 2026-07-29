/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// AIV progress / Sequential peer-sync for fused CCU Reduce+Broadcast.
// Shared ready-counter TNOTIFY walk; Pipelined also fires progress CKE per slot.

#ifndef PIPE_FIX
#define PIPE_FIX static_cast<pipe_t>(10)
#endif

#include <pto/pto-inst.hpp>
#include <pto/common/pto_tile.hpp>
#include <pto/comm/pto_comm_inst.hpp>
#include "config.h"
#include "comm_context.h"
#include "kernel_launchers.h"

using namespace pto;

template <typename T>
AICORE inline __gm__ T* CommRemotePtr(__gm__ CommDeviceContext* ctx, __gm__ T* localPtr, int pe)
{
    uint64_t localBase = ctx->windowsIn[ctx->rankId];
    uint64_t offset = (uint64_t)localPtr - localBase;
    return (__gm__ T*)(ctx->windowsIn[pe] + offset);
}

static constexpr uint64_t kCkeValidBit = 1ULL << 63;
static constexpr uint32_t kPollFenceInterval = 64;
// One in-flight group per progress CKE slot. A CKE mask bit is edge-triggered and
// saturates, so a slot must not be re-poked before the CCU consumes it; alternating
// over CCU_PROGRESS_SLOTS slots is what makes depth > 1 safe.
static constexpr uint32_t kSchedMaxInflight = CCU_PIPE_DEPTH;
static constexpr uint32_t kProgressSlots = CCU_PROGRESS_SLOTS;
static_assert(kSchedMaxInflight <= kProgressSlots, "each in-flight group needs its own progress CKE slot");

// ProgressCtx and ProgressCkeCtx are defined in kernel_launchers.h.

AICORE inline volatile __gm__ uint64_t* CkeItemsDone(ProgressCkeCtx* cke)
{
    return reinterpret_cast<volatile __gm__ uint64_t*>(cke->itemsDoneAddr);
}

// All CKE pokes (gate / seqGate / progress) use st_dev to bypass DCache.
AICORE inline void TriggerOneCke(uint64_t ckeVA, uint32_t mask)
{
    const uint64_t val = static_cast<uint64_t>(mask) | kCkeValidBit;
    st_dev(val, reinterpret_cast<__gm__ uint64_t*>(ckeVA), 0);
    dsb(DSB_DDR);
}

AICORE inline uint64_t SchedGetSysCnt()
{
    uint64_t syscnt = 0;
    asm volatile("MOV %0, SYS_CNT\n" : "+l"(syscnt));
    return syscnt;
}

AICORE inline void SchedTimingAdd(volatile __gm__ uint64_t* slot, uint64_t delta)
{
    if (slot == nullptr || delta == 0)
        return;
    *slot += delta;
    dcci((__gm__ void*)slot, SINGLE_CACHE_LINE);
}

AICORE inline void SchedTimingInc(volatile __gm__ uint64_t* slot)
{
    if (slot == nullptr)
        return;
    *slot += 1;
    dcci((__gm__ void*)slot, SINGLE_CACHE_LINE);
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

// Round-robin the progress slot so two in-flight groups never share a CKE mask bit.
AICORE inline void TriggerProgressCke(ProgressCkeCtx* cke, uint64_t issued)
{
    const uint32_t slot = static_cast<uint32_t>(issued % kProgressSlots);
    TriggerOneCke(cke->ckeSlotVA[slot], cke->ckeMask[slot]);
}

AICORE inline void WaitItemsDoneGE(volatile __gm__ uint64_t* itemsDonePtr, uint64_t target)
{
    uint32_t spin = 0;
    while (true) {
        dcci((__gm__ void*)itemsDonePtr, SINGLE_CACHE_LINE);
        if (*itemsDonePtr >= target)
            break;
        if ((++spin % kPollFenceInterval) == 0) {
            dsb(DSB_DDR);
            pipe_barrier(PIPE_ALL);
        }
    }
}

AICORE inline void WaitItemsDoneGE(
    volatile __gm__ uint64_t* itemsDonePtr, uint64_t target, volatile __gm__ uint64_t* timingSlot, bool timingOn)
{
    const uint64_t t0 = timingOn ? SchedGetSysCnt() : 0;
    WaitItemsDoneGE(itemsDonePtr, target);
    if (timingOn)
        SchedTimingAdd(timingSlot, SchedGetSysCnt() - t0);
}

AICORE inline bool CkeCanPublish(volatile __gm__ uint64_t* itemsDonePtr, uint64_t issued)
{
    dcci((__gm__ void*)itemsDonePtr, SINGLE_CACHE_LINE);
    const uint64_t done = *itemsDonePtr;
    if (done > issued)
        return false;
    return (issued - done) < kSchedMaxInflight;
}

AICORE inline void FillRemoteSignalBases(
    __gm__ int32_t* remoteSignalBase[], __gm__ CommDeviceContext* commCtx, __gm__ int32_t* signalBase,
    uint32_t safeRanks, uint32_t myRank)
{
    for (uint32_t r = 0; r < safeRanks && r < MAX_RANKS; ++r) {
        remoteSignalBase[r] = (r != myRank) ? CommRemotePtr(commCtx, signalBase, static_cast<int>(r)) : signalBase;
    }
}

// Wait until CCU has passed WaitEvent(gate) (rsKernelReady != 0) before progress CKEs.
AICORE inline void WaitKernelReadyFlag(uint64_t kernelReadyAddr, volatile __gm__ uint64_t* timingSlot, bool timingOn)
{
    if (kernelReadyAddr == 0) {
        return;
    }
    volatile __gm__ uint64_t* readyPtr = reinterpret_cast<volatile __gm__ uint64_t*>(kernelReadyAddr);
    const uint64_t t0 = timingOn ? SchedGetSysCnt() : 0;
    uint32_t readySpin = 0;
    while (*readyPtr == 0) {
        dcci((__gm__ void*)readyPtr, SINGLE_CACHE_LINE);
        if ((++readySpin % kPollFenceInterval) == 0) {
            dsb(DSB_DDR);
            pipe_barrier(PIPE_ALL);
        }
    }
    if (timingOn) {
        SchedTimingAdd(timingSlot, SchedGetSysCnt() - t0);
    }
}

static constexpr uint32_t kGroupDonePollFenceInterval = 256;

AICORE inline void PollGroupDoneFence(uint32_t& spin)
{
    if ((++spin % kGroupDonePollFenceInterval) == 0) {
        dsb(DSB_DDR);
        pipe_barrier(PIPE_ALL);
    }
}

AICORE inline void WaitLocalGroupDone(
    __gm__ int32_t* groupDoneBase, uint32_t flatGroupIdx, int32_t expected, volatile __gm__ uint64_t* timingSlot,
    bool timingOn)
{
    volatile __gm__ int32_t* done = groupDoneBase + flatGroupIdx * kGroupDoneStride;
    const uint64_t t0 = timingOn ? SchedGetSysCnt() : 0;
    uint32_t spin = 0;
    while (true) {
        dcci((__gm__ void*)done, SINGLE_CACHE_LINE);
        if (*done >= expected)
            break;
        PollGroupDoneFence(spin);
    }
    if (timingOn)
        SchedTimingAdd(timingSlot, SchedGetSysCnt() - t0);
}

// Shared ready-counter TNOTIFY (slot layout is G_SIGNAL_GROUP_READY_*; used by both
// Pipelined mid-compute progress and Sequential post-GEMM peer sync).
AICORE inline void NotifyOwnerReady(
    __gm__ int32_t* remoteSignalBase[], uint32_t owner, uint32_t flatReadyIdx, volatile __gm__ uint64_t* timingSlot,
    bool timingOn)
{
    const uint64_t t0 = timingOn ? SchedGetSysCnt() : 0;
    pto::comm::Signal sig(remoteSignalBase[owner] + G_SIGNAL_GROUP_READY_OFFSET + flatReadyIdx * kGroupReadyStride);
    pto::comm::TNOTIFY(sig, 1, pto::comm::NotifyOp::AtomicAdd);
    if (timingOn)
        SchedTimingAdd(timingSlot, SchedGetSysCnt() - t0);
}

AICORE inline bool PeerReadySatisfied(__gm__ int32_t* signalBase, uint32_t flatReadyIdx, int32_t peerTarget)
{
    if (peerTarget <= 0) {
        return true;
    }
    pto::comm::Signal readySig(signalBase + G_SIGNAL_GROUP_READY_OFFSET + flatReadyIdx * kGroupReadyStride);
    return pto::comm::TTEST(readySig, peerTarget, pto::comm::WaitCmp::GE);
}

// Peer-ready only (Sequential post-GEMM sync; also the peer half of Pipelined wait).
AICORE inline void WaitPeerReady(
    __gm__ int32_t* signalBase, uint32_t flatReadyIdx, int32_t peerTarget, volatile __gm__ uint64_t* timingSlot,
    bool timingOn)
{
    uint32_t spin = 0;
    while (!PeerReadySatisfied(signalBase, flatReadyIdx, peerTarget)) {
        if (timingOn) {
            const uint64_t t0 = SchedGetSysCnt();
            PollGroupDoneFence(spin);
            SchedTimingAdd(timingSlot, SchedGetSysCnt() - t0);
        } else {
            PollGroupDoneFence(spin);
        }
    }
}

// Owner (Pipelined): peer ready + depth-D itemsDone (+ ring).
// Group `issued` may enter flight once all but the last (D - 1) issued groups have
// retired. At D = 1 this is `done >= issued`, i.e. strict one-at-a-time.
AICORE inline void WaitPeerReadyAndBackpressure(
    __gm__ int32_t* signalBase, uint32_t flatReadyIdx, int32_t peerTarget, volatile __gm__ uint64_t* itemsDonePtr,
    uint64_t issued, volatile __gm__ ProgressCtx::Timing* timing, bool timingOn)
{
    const bool needBp = (issued >= kSchedMaxInflight);
    const uint64_t bpTarget = needBp ? (issued - kSchedMaxInflight + 1) : 0;
    uint32_t spin = 0;

    while (true) {
        const uint64_t t0 = timingOn ? SchedGetSysCnt() : 0;
        const bool peerReady = PeerReadySatisfied(signalBase, flatReadyIdx, peerTarget);

        bool bpReady = true;
        if (needBp) {
            dcci((__gm__ void*)itemsDonePtr, SINGLE_CACHE_LINE);
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

AICORE inline void ClearProgressTiming(volatile __gm__ ProgressCtx::Timing* timing)
{
    timing->totalCycles = 0;
    timing->notifyCycles = 0;
    timing->backpressureCycles = 0;
    timing->idleSpinCycles = 0;
    timing->triggerCount = 0;
    timing->groupDoneCycles = 0;
    timing->kernelReadyCycles = 0;
    timing->finalWaitCycles = 0;
    dcci((__gm__ void*)timing, ENTIRE_DATA_CACHE);
    pipe_barrier(PIPE_ALL);
}

// Shared owner-major ready walk over ready-counter slots.
// rsCke != nullptr: Pipelined — wait peer+BP, fire progress CKE per slot.
// rsCke == nullptr: Sequential — peer sync only (caller opens one-shot gate once after).
AICORE inline uint64_t RunOwnerMajorReadyWalk(
    ProgressCkeCtx* rsCke, __gm__ int32_t* groupDone, __gm__ int32_t* signalBase, __gm__ int32_t* remoteSignalBase[],
    uint32_t numTiles, uint32_t safeRanks, uint32_t myRank, volatile __gm__ ProgressCtx::Timing* timing, bool timingOn)
{
    uint64_t issued = 0;
    const int32_t peerTarget = static_cast<int32_t>(safeRanks) - 1;
    for (uint32_t owner = 0; owner < safeRanks && owner < static_cast<uint32_t>(MAX_RANKS); ++owner) {
        const uint32_t ownerGroups = CcuOwnerGroupCount(owner, numTiles, safeRanks);
        for (uint32_t g = 0; g < ownerGroups; ++g) {
            const int32_t expected = static_cast<int32_t>(CcuOwnerGroupTilesInGroup(owner, g, numTiles, safeRanks));
            if (expected <= 0) {
                continue;
            }
            const uint32_t flat = CcuOwnerGroupFlatIndex(owner, g, numTiles, safeRanks);
            WaitLocalGroupDone(groupDone, flat, expected, timingOn ? &timing->groupDoneCycles : nullptr, timingOn);
            if (myRank != owner) {
                NotifyOwnerReady(remoteSignalBase, owner, flat, timingOn ? &timing->notifyCycles : nullptr, timingOn);
                continue;
            }
            if (rsCke != nullptr) {
                WaitPeerReadyAndBackpressure(
                    signalBase, flat, peerTarget, CkeItemsDone(rsCke), issued, timing, timingOn);
                TriggerProgressCke(rsCke, issued);
                if (timingOn) {
                    SchedTimingInc(&timing->triggerCount);
                }
                issued++;
            } else {
                WaitPeerReady(signalBase, flat, peerTarget, nullptr, false);
            }
        }
    }
    return issued;
}

__global__ AICORE void ccu_gemm_ar_progress_kernel(
    ProgressCkeCtx rsCke, __gm__ int32_t* groupDone, __gm__ uint8_t* progressCtx, __gm__ uint8_t* signalRaw,
    __gm__ uint8_t* commCtxRaw, uint32_t numTiles, uint32_t rankSize, uint32_t myRank)
{
    if (get_block_idx() != 0) {
        return;
    }

    __gm__ CommDeviceContext* commCtx = reinterpret_cast<__gm__ CommDeviceContext*>(commCtxRaw);
    __gm__ int32_t* signalBase = reinterpret_cast<__gm__ int32_t*>(signalRaw);
    volatile __gm__ ProgressCtx* ctx = reinterpret_cast<volatile __gm__ ProgressCtx*>(progressCtx);
    if (ctx != nullptr) {
        dcci((__gm__ void*)ctx, SINGLE_CACHE_LINE);
        pipe_barrier(PIPE_ALL);
    }
    const bool timingOn = (ctx != nullptr && ctx->schedTimingEnable != 0);
    volatile __gm__ ProgressCtx::Timing* timing = timingOn ? &ctx->rsSchedTiming : nullptr;
    const uint64_t kernelT0 = timingOn ? SchedGetSysCnt() : 0;
    if (timingOn) {
        ClearProgressTiming(timing);
    }

    const uint32_t safeRanks = (rankSize > 0) ? rankSize : 1;
    __gm__ int32_t* remoteSignalBase[MAX_RANKS];
    FillRemoteSignalBases(remoteSignalBase, commCtx, signalBase, safeRanks, myRank);
    if (safeRanks > 1) {
        set_st_atomic_cfg(ATOMIC_S32, ATOMIC_SUM);
    }

    WaitKernelReadyFlag(rsCke.kernelReadyAddr, timingOn ? &timing->kernelReadyCycles : nullptr, timingOn);
    const uint64_t issued = RunOwnerMajorReadyWalk(
        &rsCke, groupDone, signalBase, remoteSignalBase, numTiles, safeRanks, myRank, timing, timingOn);

    if (issued > 0) {
        WaitItemsDoneGE(CkeItemsDone(&rsCke), issued, timingOn ? &timing->finalWaitCycles : nullptr, timingOn);
    }
    if (timingOn) {
        SchedTimingAdd(&timing->totalCycles, SchedGetSysCnt() - kernelT0);
    }
}

void launchCcuGemmArProgress(
    void* stream, ProgressCkeCtx rsCke, int32_t* groupDone, uint8_t* progressCtx, uint8_t* signal_matrix,
    uint8_t* commCtx, uint32_t numTiles, uint32_t rankSize, uint32_t myRank)
{
    ccu_gemm_ar_progress_kernel<<<1, nullptr, stream>>>(
        rsCke, groupDone, progressCtx, signal_matrix, commCtx, numTiles, rankSize, myRank);
}

// Call only AFTER full local GEMM sync. Peer-sync via shared ready counters, then one-shot gate.
__global__ AICORE void ccu_gemm_ar_seq_peer_sync_gate_kernel(
    uint64_t gateCkeVA, uint32_t gateMask, __gm__ int32_t* groupDone, __gm__ uint8_t* signalRaw,
    __gm__ uint8_t* commCtxRaw, uint32_t numTiles, uint32_t rankSize, uint32_t myRank)
{
    if (get_block_idx() != 0) {
        return;
    }

    __gm__ CommDeviceContext* commCtx = reinterpret_cast<__gm__ CommDeviceContext*>(commCtxRaw);
    __gm__ int32_t* signalBase = reinterpret_cast<__gm__ int32_t*>(signalRaw);
    const uint32_t safeRanks = (rankSize > 0) ? rankSize : 1;
    __gm__ int32_t* remoteSignalBase[MAX_RANKS];
    FillRemoteSignalBases(remoteSignalBase, commCtx, signalBase, safeRanks, myRank);
    if (safeRanks > 1) {
        set_st_atomic_cfg(ATOMIC_S32, ATOMIC_SUM);
    }

    (void)RunOwnerMajorReadyWalk(
        nullptr, groupDone, signalBase, remoteSignalBase, numTiles, safeRanks, myRank, nullptr, false);
    TriggerOneCke(gateCkeVA, gateMask);
}

void launchCcuGemmArSeqPeerSyncGate(
    void* stream, uint64_t gateCkeVA, uint32_t gateMask, int32_t* groupDone, uint8_t* signal_matrix, uint8_t* commCtx,
    uint32_t numTiles, uint32_t rankSize, uint32_t myRank)
{
    ccu_gemm_ar_seq_peer_sync_gate_kernel<<<1, nullptr, stream>>>(
        gateCkeVA, gateMask, groupDone, signal_matrix, commCtx, numTiles, rankSize, myRank);
}

// Unpack kernel: convert packed reduced output back to row-major
using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
using SchedGlobal = pto::GlobalTensor<half, ShapeDyn, StrideDyn, pto::Layout::ND>;
using SchedSubTile = pto::Tile<pto::TileType::Vec, half, G_COMM_SUB_M, G_BASE_N, pto::BLayout::RowMajor, -1, -1>;

__global__ AICORE void ccu_gemm_ar_unpack_kernel(
    __gm__ uint8_t* packed_output, __gm__ uint8_t* row_output, uint32_t subtilesPerTile, uint32_t rankSize)
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
    StrideDyn packedStride(
        static_cast<int>(G_BASE_M * G_BASE_N), static_cast<int>(G_BASE_M * G_BASE_N),
        static_cast<int>(G_BASE_M * G_BASE_N), static_cast<int>(G_BASE_N), 1);
    StrideDyn rowStride(
        static_cast<int>(G_BASE_M * G_N), static_cast<int>(G_BASE_M * G_N), static_cast<int>(G_BASE_M * G_N),
        static_cast<int>(G_N), 1);
    SchedGlobal srcG(reinterpret_cast<__gm__ half*>(packed_output) + packedOff, shape, packedStride);
    SchedGlobal dstG(reinterpret_cast<__gm__ half*>(row_output) + rowOff, shape, rowStride);

    TLOAD(tileBuf, srcG);
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    TSTORE(dstG, tileBuf);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
}

void launchCcuGemmArUnpack(
    uint8_t* packed_output, uint8_t* row_output, void* stream, uint32_t subtilesPerTile, uint32_t rankSize)
{
    ccu_gemm_ar_unpack_kernel<<<G_NUM_TILES * subtilesPerTile, nullptr, stream>>>(
        packed_output, row_output, subtilesPerTile, rankSize);
}

__global__ AICORE void ccu_gemm_ar_gate_trigger_kernel(uint64_t gateCkeVA, uint32_t gateMask)
{
    if (get_block_idx() != 0)
        return;
    // Same st_dev poke as progress / seqGate CKE.
    TriggerOneCke(gateCkeVA, gateMask);
}

int launchCcuGemmArGateTrigger(void* stream, uint64_t gateCkeVA, uint32_t gateMask)
{
    ccu_gemm_ar_gate_trigger_kernel<<<1, nullptr, stream>>>(gateCkeVA, gateMask);
    return 0;
}
