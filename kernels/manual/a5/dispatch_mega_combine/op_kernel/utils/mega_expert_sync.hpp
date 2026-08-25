/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef DISPATCH_MEGA_COMBINE_MEGA_EXPERT_SYNC_HPP
#define DISPATCH_MEGA_COMBINE_MEGA_EXPERT_SYNC_HPP

#include "kernel_operator.h"

#include <pto/common/pto_tile.hpp>
#include <pto/pto-inst.hpp>

#include "dispatch_mega_combine_tiling.h"
#include "const_args.hpp"

struct MegaMoeFixedCoreRoleInfo {
    uint32_t role = 0U;
    uint32_t physicalBlockId = 0U;
    uint32_t subblockId = 0U;
    uint32_t groupLocalId = 0U;
    uint32_t groupSize = 0U;
};

AICORE inline MegaMoeFixedCoreRoleInfo FixedCoreRole(const __gm__ MegaMoeTilingData *tilingData)
{
    MegaMoeFixedCoreRoleInfo info;
    const __gm__ MegaMoeFixedGroupTiling &fixed = tilingData->fixedGroupTiling;
    info.physicalBlockId = get_block_idx();
    const bool firstGmmGroup = info.physicalBlockId < fixed.gmm1GroupSize;
#if defined(__DAV_CUBE__)
    {
        if (firstGmmGroup) {
            info.role = kMegaMoeFixedRoleGmm1;
            info.groupLocalId = info.physicalBlockId;
            info.groupSize = fixed.gmm1GroupSize;
        } else {
            info.role = kMegaMoeFixedRoleGmm2;
            info.groupLocalId = info.physicalBlockId - fixed.gmm1GroupSize;
            info.groupSize = fixed.gmm2GroupSize;
        }
        return info;
    }
#elif defined(__DAV_VEC__)
    info.subblockId = get_subblockid();
    if (info.subblockId == 0U && info.physicalBlockId < fixed.dispatchGroupSize) {
        info.role = kMegaMoeFixedRoleDispatch;
        info.groupLocalId = info.physicalBlockId;
        info.groupSize = fixed.dispatchGroupSize;
    } else if (info.subblockId == 1U && firstGmmGroup) {
        info.role = kMegaMoeFixedRoleSwiglu;
        info.groupLocalId = info.physicalBlockId;
        info.groupSize = fixed.physicalAicNum;
    } else {
        info.role = kMegaMoeFixedRoleCombine;
        // AIV1 follows the post-Dispatch GMM2 split. AIV0 follows the wider
        // Dispatch split and uses the trailing coordinates for metadata work.
        if (info.subblockId == 1U) {
            info.groupLocalId = info.physicalBlockId - fixed.gmm1GroupSize;
        } else {
            info.groupLocalId = fixed.gmm2GroupSize + info.physicalBlockId - fixed.dispatchGroupSize;
        }
        info.groupSize = fixed.gmm2GroupSize * kMegaMoeFixedAivSubblocksPerPhysicalBlock;
    }
#endif
    return info;
}

AICORE inline volatile __gm__ int32_t *FixedSyncSlot(GM_ADDR workspaceGM, const __gm__ MegaMoeTilingData *tilingData,
                                                     uint32_t slot)
{
    const __gm__ MegaMoeFixedGroupTiling &fixed = tilingData->fixedGroupTiling;
    return reinterpret_cast<volatile __gm__ int32_t *>(workspaceGM + fixed.syncOffset +
                                                       static_cast<uint64_t>(slot) * kMegaMoeFixedSyncSlotBytes);
}

struct MegaMoeGroupBarrierResult {
    int32_t arriveEpoch = 0;
    int32_t releaseEpoch = 0;
    int32_t observedArrivalMin = 0;
};

// The doorbell is a scalar GM store. The caller drains payload pipelines before publishing.
AICORE inline void PublishScalarEpoch(volatile __gm__ int32_t *slot, int32_t epoch)
{
    *slot = epoch;
    dcci((__gm__ void *)slot, SINGLE_CACHE_LINE);
    dsb(DSB_DDR);
}

AICORE inline void ResetFixedSyncWorkspace(GM_ADDR workspaceGM, const __gm__ MegaMoeTilingData *tilingData)
{
    for (uint32_t slotId = 0U; slotId < kMegaMoeFixedSyncSlotCount; ++slotId) {
        volatile __gm__ int32_t *slot = FixedSyncSlot(workspaceGM, tilingData, slotId);
        dcci((__gm__ void *)slot, SINGLE_CACHE_LINE);
        __asm__ __volatile__("");
        uint32_t value = 0U;
        if (slotId == kMegaMoeFixedSyncHeadCanarySlot) {
            value = kMegaMoeFixedHeadCanary;
        } else if (slotId == kMegaMoeFixedSyncTailCanarySlot) {
            value = kMegaMoeFixedTailCanary;
        }
        *slot = static_cast<int32_t>(value);
    }
    for (uint32_t slotId = 0U; slotId < kMegaMoeFixedSyncSlotCount; ++slotId) {
        dcci((__gm__ void *)FixedSyncSlot(workspaceGM, tilingData, slotId), SINGLE_CACHE_LINE);
    }
    dsb(DSB_DDR);
    pipe_barrier(PIPE_ALL);
}

AICORE inline int32_t ReadScalarEpoch(volatile __gm__ int32_t *slot)
{
    dcci((__gm__ void *)slot, SINGLE_CACHE_LINE);
    dsb(DSB_DDR);
    return *slot;
}

AICORE inline bool TestEpoch(volatile __gm__ int32_t *slot, int32_t epoch, int32_t &observed)
{
    observed = ReadScalarEpoch(slot);
    return observed >= epoch;
}

AICORE inline void EpochPollBackoff(uint32_t = 0U)
{
    constexpr uint32_t kDelayTicks = 3U;
    const uint64_t deadline = get_sys_cnt() + kDelayTicks;
    while (get_sys_cnt() < deadline) {
        __asm__ __volatile__("");
    }
}

AICORE inline int32_t WaitEpochRaw(volatile __gm__ int32_t *slot, int32_t epoch)
{
    while (true) {
        int32_t observed = 0;
        if (TestEpoch(slot, epoch, observed)) {
            return observed;
        }
        EpochPollBackoff();
    }
}

AICORE inline int32_t WaitEpochAcquire(volatile __gm__ int32_t *slot, int32_t epoch)
{
    const int32_t observed = WaitEpochRaw(slot, epoch);
    pipe_barrier(PIPE_ALL);
    dsb(DSB_DDR);
    return observed;
}

AICORE inline void PublishCombineConsumerArmed(GM_ADDR workspaceGM,
                                                const __gm__ MegaMoeTilingData *tilingData,
                                                uint32_t physicalBlockId)
{
    PublishScalarEpoch(
        FixedSyncSlot(workspaceGM, tilingData, kMegaMoeFixedSyncCombineConsumerArmedBase + physicalBlockId),
        kMegaMoeFixedCombineConsumerArmedMarker);
}

AICORE inline void PublishGmm2EntryReady(GM_ADDR workspaceGM,
                                         const __gm__ MegaMoeTilingData *tilingData)
{
    PublishScalarEpoch(
        FixedSyncSlot(workspaceGM, tilingData, kMegaMoeFixedSyncGmm2EntryReadySlot),
        kMegaMoeFixedGmm2EntryReadyMarker);
}

AICORE inline void WaitGmm2EntryReady(GM_ADDR workspaceGM,
                                      const __gm__ MegaMoeTilingData *tilingData)
{
    WaitEpochAcquire(
        FixedSyncSlot(workspaceGM, tilingData, kMegaMoeFixedSyncGmm2EntryReadySlot),
        kMegaMoeFixedGmm2EntryReadyMarker);
}

AICORE inline bool Gmm2EntryReady(GM_ADDR workspaceGM, const __gm__ MegaMoeTilingData *tilingData)
{
    return ReadScalarEpoch(FixedSyncSlot(workspaceGM, tilingData, kMegaMoeFixedSyncGmm2EntryReadySlot)) >=
           kMegaMoeFixedGmm2EntryReadyMarker;
}

AICORE inline void WaitGmm2ConsumerReady(GM_ADDR workspaceGM, const __gm__ MegaMoeTilingData *tilingData,
                                         uint32_t physicalBlockId)
{
    WaitEpochRaw(FixedSyncSlot(workspaceGM, tilingData, kMegaMoeFixedSyncGmm2EntryReadySlot),
                 kMegaMoeFixedGmm2EntryReadyMarker);
    WaitEpochRaw(FixedSyncSlot(workspaceGM, tilingData, kMegaMoeFixedSyncCombineConsumerArmedBase + physicalBlockId),
                 kMegaMoeFixedCombineConsumerArmedMarker);
    pipe_barrier(PIPE_ALL);
    dsb(DSB_DDR);
}

AICORE inline int32_t EncodeGmm1SplitDecision(uint32_t completedWaveCount, bool split)
{
    const uint32_t epoch = completedWaveCount & static_cast<uint32_t>(kMegaMoeFixedGmm1SplitDecisionMask);
    return static_cast<int32_t>(epoch | (split ? static_cast<uint32_t>(kMegaMoeFixedGmm1SplitDecisionBit) : 0U));
}

AICORE inline bool Gmm1SplitDecisionEnabled(int32_t decision, uint32_t completedWaveCount)
{
    const uint32_t splitWave = static_cast<uint32_t>(decision & kMegaMoeFixedGmm1SplitDecisionMask);
    return (decision & kMegaMoeFixedGmm1SplitDecisionBit) != 0 && completedWaveCount >= splitWave;
}

AICORE inline int32_t CoordinateGmm1SplitDecision(GM_ADDR workspaceGM,
                                                   const __gm__ MegaMoeTilingData *tilingData,
                                                   uint32_t localId, uint32_t completedWaveCount)
{
    if (localId == 0U) {
        const bool gmm2Ready =
            ReadScalarEpoch(FixedSyncSlot(workspaceGM, tilingData, kMegaMoeFixedSyncGmm2EntryReadySlot)) >=
            kMegaMoeFixedGmm2EntryReadyMarker;
        PublishScalarEpoch(FixedSyncSlot(workspaceGM, tilingData, kMegaMoeFixedSyncGmm1SplitDecisionSlot),
                           EncodeGmm1SplitDecision(completedWaveCount, gmm2Ready));
    }
    return WaitEpochAcquire(FixedSyncSlot(workspaceGM, tilingData, kMegaMoeFixedSyncGmm1SplitDecisionSlot),
                            static_cast<int32_t>(completedWaveCount));
}

AICORE inline int32_t WaitGmm1SplitDecision(GM_ADDR workspaceGM,
                                             const __gm__ MegaMoeTilingData *tilingData,
                                             uint32_t completedWaveCount)
{
    return WaitEpochAcquire(FixedSyncSlot(workspaceGM, tilingData, kMegaMoeFixedSyncGmm1SplitDecisionSlot),
                            static_cast<int32_t>(completedWaveCount));
}

#if defined(__DAV_VEC__)
// These snapshot helpers require AIV UB and MTE2/MTE3; AIC callers publish scalar arrivals only.
constexpr uint32_t kMegaMoeSyncSnapshotValuesPerSlot = kMegaMoeFixedSyncSlotBytes / sizeof(int32_t);
constexpr uint32_t kMegaMoeSyncSnapshotMaxValues = kMegaMoeFixedPhysicalAicNum * kMegaMoeSyncSnapshotValuesPerSlot;
constexpr uint64_t kMegaMoeSyncSnapshotUbBytes = static_cast<uint64_t>(kMegaMoeSyncSnapshotMaxValues) * sizeof(int32_t);
constexpr uint64_t kMegaMoeSyncSnapshotUbOffset = AtlasA5::UB_SIZE - kMegaMoeSyncSnapshotUbBytes;
constexpr event_t kMegaMoeSyncSnapshotEvent = EVENT_ID0;

static_assert(kMegaMoeFixedSyncSlotBytes % sizeof(int32_t) == 0U);
static_assert(kMegaMoeSyncSnapshotUbBytes <= A5_UB_SYNC_RESERVE_BYTES);
static_assert(kMegaMoeSyncSnapshotUbOffset >= A5_MAIN_UB_SIZE);
static_assert(kMegaMoeSyncSnapshotUbOffset % UB_ALIGN == 0U);

AICORE inline int32_t ReadArrivalMinMte(GM_ADDR workspaceGM, const __gm__ MegaMoeTilingData *tilingData,
                                        uint32_t arrivalBaseSlot, uint32_t producerCount)
{
    if (producerCount == 0U || producerCount > kMegaMoeFixedPhysicalAicNum) {
        return 0;
    }

    using SnapshotShape = pto::Shape<1, 1, 1, 1, pto::DYNAMIC>;
    using SnapshotStride = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, 1>;
    using SnapshotGlobal = pto::GlobalTensor<int32_t, SnapshotShape, SnapshotStride, pto::Layout::ND>;
    using SnapshotTile =
        pto::Tile<pto::TileType::Vec, int32_t, 1, kMegaMoeSyncSnapshotMaxValues, pto::BLayout::RowMajor, -1, -1>;

    const __gm__ MegaMoeFixedGroupTiling &fixed = tilingData->fixedGroupTiling;
    __gm__ int32_t *arrivalBase = reinterpret_cast<__gm__ int32_t *>(
        workspaceGM + fixed.syncOffset + static_cast<uint64_t>(arrivalBaseSlot) * kMegaMoeFixedSyncSlotBytes);
    const uint32_t snapshotValues = producerCount * kMegaMoeSyncSnapshotValuesPerSlot;
    SnapshotShape snapshotShape(snapshotValues);
    SnapshotStride snapshotStride(snapshotValues, snapshotValues, snapshotValues, snapshotValues);
    SnapshotGlobal snapshotGlobal(arrivalBase, snapshotShape, snapshotStride);
    SnapshotTile snapshotTile(1U, snapshotValues);
    pto::TASSIGN(snapshotTile, kMegaMoeSyncSnapshotUbOffset);
    pto::TLOAD(snapshotTile, snapshotGlobal);
    pto::PtoSetWaitFlag<PIPE_MTE2, PIPE_S>(kMegaMoeSyncSnapshotEvent, kMegaMoeSyncSnapshotEvent);

    int32_t observedMin = snapshotTile.GetValue(0U);
    for (uint32_t peer = 1U; peer < producerCount; ++peer) {
        const int32_t observed = snapshotTile.GetValue(peer * kMegaMoeSyncSnapshotValuesPerSlot);
        observedMin = observed < observedMin ? observed : observedMin;
    }
    return observedMin;
}

AICORE inline int32_t WaitArrivalMinMte(GM_ADDR workspaceGM, const __gm__ MegaMoeTilingData *tilingData,
                                        uint32_t arrivalBaseSlot, uint32_t producerCount, int32_t epoch)
{
    while (true) {
        const int32_t observedMin = ReadArrivalMinMte(workspaceGM, tilingData, arrivalBaseSlot, producerCount);
        if (observedMin >= epoch) {
            return observedMin;
        }
        EpochPollBackoff();
    }
}

AICORE inline void PublishEpochRangeMte(GM_ADDR workspaceGM, const __gm__ MegaMoeTilingData *tilingData,
                                        uint32_t baseSlot, uint32_t count, int32_t epoch)
{
    if (count == 0U || count > kMegaMoeFixedPhysicalAicNum) {
        return;
    }

    using SnapshotShape = pto::Shape<1, 1, 1, 1, pto::DYNAMIC>;
    using SnapshotStride = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, 1>;
    using SnapshotGlobal = pto::GlobalTensor<int32_t, SnapshotShape, SnapshotStride, pto::Layout::ND>;
    using SnapshotTile =
        pto::Tile<pto::TileType::Vec, int32_t, 1, kMegaMoeSyncSnapshotMaxValues, pto::BLayout::RowMajor, -1, -1>;

    const __gm__ MegaMoeFixedGroupTiling &fixed = tilingData->fixedGroupTiling;
    __gm__ int32_t *readyBase = reinterpret_cast<__gm__ int32_t *>(
        workspaceGM + fixed.syncOffset + static_cast<uint64_t>(baseSlot) * kMegaMoeFixedSyncSlotBytes);
    const uint32_t snapshotValues = count * kMegaMoeSyncSnapshotValuesPerSlot;
    SnapshotShape snapshotShape(snapshotValues);
    SnapshotStride snapshotStride(snapshotValues, snapshotValues, snapshotValues, snapshotValues);
    SnapshotGlobal snapshotGlobal(readyBase, snapshotShape, snapshotStride);
    SnapshotTile snapshotTile(1U, snapshotValues);
    pto::TASSIGN(snapshotTile, kMegaMoeSyncSnapshotUbOffset);
    for (uint32_t localId = 0U; localId < count; ++localId) {
        snapshotTile.SetValue(localId * kMegaMoeSyncSnapshotValuesPerSlot, epoch);
    }
    pto::PtoSetWaitFlag<PIPE_S, PIPE_MTE3>(kMegaMoeSyncSnapshotEvent, kMegaMoeSyncSnapshotEvent);
    pto::TSTORE(snapshotGlobal, snapshotTile);
    pto::PtoSetWaitFlag<PIPE_MTE3, PIPE_S>(kMegaMoeSyncSnapshotEvent, kMegaMoeSyncSnapshotEvent);
}
#endif

AICORE inline MegaMoeGroupBarrierResult PublishGroupArrival(GM_ADDR workspaceGM,
                                                            const __gm__ MegaMoeTilingData *tilingData,
                                                            uint32_t arrivalBaseSlot, uint32_t localId,
                                                            uint32_t notifyCall)
{
    MegaMoeGroupBarrierResult result;
    result.arriveEpoch = static_cast<int32_t>(notifyCall + 1U);
    result.releaseEpoch = result.arriveEpoch;
    result.observedArrivalMin = result.arriveEpoch;

    pipe_barrier(PIPE_ALL);
    dsb(DSB_DDR);
    PublishScalarEpoch(FixedSyncSlot(workspaceGM, tilingData, arrivalBaseSlot + localId), result.arriveEpoch);
    return result;
}

#if defined(__DAV_VEC__)
AICORE inline MegaMoeGroupBarrierResult CoordinateGroupConsumersMte(GM_ADDR workspaceGM,
                                                                    const __gm__ MegaMoeTilingData *tilingData,
                                                                    uint32_t arrivalBaseSlot, uint32_t readyBaseSlot,
                                                                    uint32_t producerCount, uint32_t consumerCount,
                                                                    uint32_t notifyCall)
{
    MegaMoeGroupBarrierResult result;
    result.arriveEpoch = static_cast<int32_t>(notifyCall + 1U);
    result.releaseEpoch = result.arriveEpoch;
    result.observedArrivalMin =
        WaitArrivalMinMte(workspaceGM, tilingData, arrivalBaseSlot, producerCount, result.arriveEpoch);
    pipe_barrier(PIPE_ALL);
    dsb(DSB_DDR);
    PublishEpochRangeMte(workspaceGM, tilingData, readyBaseSlot, consumerCount, result.releaseEpoch);
    return result;
}

AICORE inline MegaMoeGroupBarrierResult NotifyGroupConsumersMte(
    GM_ADDR workspaceGM, const __gm__ MegaMoeTilingData *tilingData, uint32_t arrivalBaseSlot, uint32_t readyBaseSlot,
    uint32_t producerCount, uint32_t consumerCount, uint32_t localId, uint32_t coordinatorLocalId, uint32_t notifyCall)
{
    MegaMoeGroupBarrierResult result;
    result.arriveEpoch = static_cast<int32_t>(notifyCall + 1U);
    result.releaseEpoch = result.arriveEpoch;
    result.observedArrivalMin = result.arriveEpoch;

    pipe_barrier(PIPE_ALL);
    dsb(DSB_DDR);
    PublishScalarEpoch(FixedSyncSlot(workspaceGM, tilingData, arrivalBaseSlot + localId), result.arriveEpoch);

    if (localId == coordinatorLocalId) {
        result.observedArrivalMin =
            WaitArrivalMinMte(workspaceGM, tilingData, arrivalBaseSlot, producerCount, result.arriveEpoch);
        pipe_barrier(PIPE_ALL);
        dsb(DSB_DDR);
        PublishEpochRangeMte(workspaceGM, tilingData, readyBaseSlot, consumerCount, result.releaseEpoch);
    }
    return result;
}
#endif

#endif // DISPATCH_MEGA_COMBINE_MEGA_EXPERT_SYNC_HPP
