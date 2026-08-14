/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef DISPATCH_MEGA_COMBINE_COMBINE_H
#define DISPATCH_MEGA_COMBINE_COMBINE_H

#include <type_traits>

#include <pto/pto-inst.hpp>

#include "dispatch_mega_combine_tiling.h"
#include "gmm2_combine_cv_pipe.h"
#include "gmm_common.h"
#include "kernel_operator.h"
#include "utils/common_helpers.hpp"
#include "utils/const_args.hpp"
#include "utils/hccl_window.hpp"
#include "utils/mega_expert_sync.hpp"
#include "utils/mega_wave_schedule.hpp"
#include "utils/pto_vector.hpp"

constexpr uint32_t kDirectCombineMetadataMaxElems = kMegaMoeExpertProgressMaxRanks * kMegaMoeFixedMaxExperts;
constexpr uint64_t kDirectCombineCumsumUbOffset = kGmm2CombineMetadataOffset;
constexpr uint64_t kDirectCombinePreSumUbOffset =
    kDirectCombineCumsumUbOffset + static_cast<uint64_t>(kDirectCombineMetadataMaxElems) * sizeof(int32_t);
constexpr uint64_t kDirectCombineMetadataUbEnd =
    kDirectCombinePreSumUbOffset + static_cast<uint64_t>(kDirectCombineMetadataMaxElems) * sizeof(int32_t);

static_assert(kDirectCombineMetadataUbEnd <= kGmm2CombineMetadataOffset +
                                                 kGmm2CombineMetadataBytes);

template <typename CvTile>
__tf__ AICORE inline void DirectCombineStrideStore(__gm__ bfloat16_t *dst, typename CvTile::TileDType __in__ srcTile,
                                                   uint32_t rows, uint32_t cols, uint32_t dstLeadingDim)
{
    __ubuf__ bfloat16_t *src = reinterpret_cast<__ubuf__ bfloat16_t *>(__cce_get_tile_ptr(srcTile));
    copy_ubuf_to_gm_align_v2(dst, src, 0, rows, cols * sizeof(bfloat16_t), 0,
                             static_cast<uint64_t>(dstLeadingDim) * sizeof(bfloat16_t),
                             CvTile::Cols * sizeof(bfloat16_t));
}

template <typename OutputElement>
class Combine {
public:
    AICORE inline void Init(GM_ADDR workspaceGM, const __gm__ MegaMoeTilingData *tilingData);
    AICORE inline void ProcessFixed(uint32_t physicalBlockId);

private:
    static_assert(std::is_same_v<OutputElement, bfloat16_t>, "MXFP8 combine output must be BF16");

    AICORE inline uint32_t CurrentM(uint32_t groupIdx) const
    {
        return MoeCurrentMRaw(cumsumMMPtr_, rankSize_, expertPerRank_, groupIdx);
    }
    AICORE inline uint32_t CoreLoops(uint32_t currentM) const
    {
        return GmmCommonCoreLoops(currentM, problemK_, kMegaMoeGmmTileM, kMegaMoeGmmTileN);
    }
    AICORE inline uint32_t ProgressWorkerCount() const
    {
        return tilingData_->fixedGroupTiling.gmm2GroupSize;
    }
    AICORE inline uint32_t ProgressWorkerIndex() const
    {
        const uint32_t workerBase = tilingData_->fixedGroupTiling.gmm1GroupSize;
        return physicalBlockId_ >= workerBase ? physicalBlockId_ - workerBase : ProgressWorkerCount();
    }
    AICORE inline uint32_t LargeLanesPerRank() const
    {
        if (rankSize_ == 0U) {
            return 1U;
        }
        uint32_t lanes = ProgressWorkerCount() / rankSize_;
        if (lanes == 0U) {
            lanes = 1U;
        }
        return lanes > 1U ? 1U : lanes;
    }
    AICORE inline uint32_t DirectLargeTaskCount() const
    {
        return rankSize_ * LargeLanesPerRank();
    }
    AICORE inline uint32_t DirectLargeWorkerCount() const
    {
        const uint32_t taskCount = DirectLargeTaskCount();
        return ProgressWorkerCount() < taskCount ? ProgressWorkerCount() : taskCount;
    }
    AICORE inline uint32_t ReadyCoordinatorProgressIndex() const
    {
        const uint32_t workerCount = DirectLargeWorkerCount();
        if (workerCount == 0U) {
            return 0U;
        }
        const uint64_t firstLocalTask = static_cast<uint64_t>(rank_) * LargeLanesPerRank();
        return static_cast<uint32_t>(firstLocalTask % workerCount);
    }
    AICORE inline uint32_t ReadyCoordinatorPhysicalBlock() const
    {
        return tilingData_->fixedGroupTiling.gmm1GroupSize + ReadyCoordinatorProgressIndex();
    }
    AICORE inline void ReplayPrimaryTileBalanceBefore(
        uint32_t groupIdx, MegaMoeCoreTileBalancer &tileBalancer) const;
    AICORE inline uint32_t HelperJoinWave() const;
    AICORE inline bool PrimaryJoinsWave(uint32_t waveIdx) const;
    AICORE inline uint32_t Gmm2ProducerCount(uint32_t waveIdx) const;
    AICORE inline void WaitGmm2Ready(uint32_t waveIdx, uint32_t progressWorkerIdx) const;
    AICORE inline void PublishAssignedExpertProgress(uint32_t progressWorkerIdx, uint32_t readyWaveCount,
                                                     uint32_t readyExpertCount) const;
    AICORE inline void FinalizeRankStreamingLane(uint32_t progressWorkerIdx) const;
    AICORE inline void PrefetchDirectMetadata();
    AICORE inline void PrepareDirectExpert(uint32_t groupIdx);
    AICORE inline void StoreDirectTile(const GmmCommonTileInfo &tileInfo, uint32_t tileIndex) const;
    AICORE inline void ConsumeDirectTile(const GmmCommonTileInfo &tileInfo);
    AICORE inline void ProcessImpl(bool helperGroup);

    GM_ADDR workspaceGM_ = nullptr;
    const __gm__ MegaMoeTilingData *tilingData_ = nullptr;
    PtoRemoteWindow remoteWindow_;
    MegaMoePeerMemoryLayout peerMemoryLayout_;
    Gmm2CombineCvPipe cvPipe_;
    __gm__ int32_t *cumsumMMPtr_ = nullptr;
    __gm__ int32_t *preSumBeforeRankPtr_ = nullptr;
    __gm__ OutputElement *remoteOutputBase_[kMegaMoeExpertProgressMaxRanks] = {nullptr};

    uint32_t problemK_ = 0U;
    uint32_t routeElems_ = 0U;
    uint32_t expertPerRank_ = 0U;
    uint32_t rank_ = 0U;
    uint32_t rankSize_ = 0U;
    uint32_t physicalBlockId_ = 0U;
    int32_t dataReadyEpoch_ = 0;
    bool directMetadataCached_ = false;
    uint32_t directRankRowBegin_[kMegaMoeExpertProgressMaxRanks] = {0U};
    uint32_t directRankRowEnd_[kMegaMoeExpertProgressMaxRanks] = {0U};
    uint32_t directCompactRowBegin_[kMegaMoeExpertProgressMaxRanks] = {0U};
};

template <typename OutputElement>
AICORE inline void Combine<OutputElement>::Init(GM_ADDR workspaceGM, const __gm__ MegaMoeTilingData *tilingData)
{
    workspaceGM_ = workspaceGM;
    tilingData_ = tilingData;
    directMetadataCached_ = false;

    problemK_ = tilingData_->megaMoeInfo.K;
    routeElems_ = tilingData_->frontReorderTiling.routeElems;
    expertPerRank_ = tilingData_->megaMoeInfo.expertPerRank;
    rank_ = tilingData_->runtimeInfo.rank;
    rankSize_ = tilingData_->runtimeInfo.rankSize;
    physicalBlockId_ = get_block_idx();

    remoteWindow_.Init(reinterpret_cast<GM_ADDR>(tilingData_->runtimeInfo.remoteWindowContext));
    peerMemoryLayout_.Init(remoteWindow_, tilingData_->frontReorderTiling);
    cumsumMMPtr_ = reinterpret_cast<__gm__ int32_t *>(workspaceGM + tilingData_->frontReorderTiling.cumsumMMOffset);
    preSumBeforeRankPtr_ =
        reinterpret_cast<__gm__ int32_t *>(remoteWindow_.LocalBase() + peerMemoryLayout_.preSumBeforeRank);
    for (uint32_t srcRank = 0U; srcRank < rankSize_ && srcRank < kMegaMoeExpertProgressMaxRanks; ++srcRank) {
        remoteOutputBase_[srcRank] = reinterpret_cast<__gm__ OutputElement *>(
            remoteWindow_.RemoteBase(peerMemoryLayout_.combineOutputByRouteSlot, static_cast<int32_t>(srcRank)));
    }
    dataReadyEpoch_ = remoteWindow_.DataReadyEpoch();
}

template <typename OutputElement>
AICORE inline void Combine<OutputElement>::ReplayPrimaryTileBalanceBefore(
    uint32_t groupIdx, MegaMoeCoreTileBalancer &tileBalancer) const
{
    const __gm__ MegaMoeFixedGroupTiling &fixed = tilingData_->fixedGroupTiling;
    SetCoreTileBalancerRange(tileBalancer, fixed.gmm1GroupSize, fixed.gmm2GroupSize);
    for (uint32_t expert = 0U; expert < groupIdx; ++expert) {
        const uint32_t coreLoops = CoreLoops(CurrentM(expert));
        const uint32_t startCoreIdx = SelectCoreTileStart(tileBalancer, coreLoops);
        CommitCoreTileAssignment(tileBalancer, startCoreIdx, coreLoops);
    }
}

template <typename OutputElement>
AICORE inline uint32_t Combine<OutputElement>::HelperJoinWave() const
{
    const int32_t decision = WaitEpochAcquire(
        FixedSyncSlot(workspaceGM_, tilingData_, kMegaMoeFixedSyncGmm2JoinSlot),
        kMegaMoeFixedGmm2JoinDecisionBit);
    const int32_t encodedWave = decision & kMegaMoeFixedGmm2JoinDecisionMask;
    return encodedWave > 0 ? static_cast<uint32_t>(encodedWave - 1) : tilingData_->fixedGroupTiling.totalWaveCount;
}

template <typename OutputElement>
AICORE inline bool Combine<OutputElement>::PrimaryJoinsWave(uint32_t waveIdx) const
{
    const int32_t decision = WaitEpochAcquire(
        FixedSyncSlot(workspaceGM_, tilingData_, kMegaMoeFixedSyncGmm2JoinSlot),
        static_cast<int32_t>(waveIdx + 1U));
    const uint32_t encodedWave = static_cast<uint32_t>(decision & kMegaMoeFixedGmm2JoinDecisionMask);
    return (decision & kMegaMoeFixedGmm2JoinDecisionBit) != 0 && encodedWave != 0U && waveIdx + 1U >= encodedWave;
}

template <typename OutputElement>
AICORE inline uint32_t Combine<OutputElement>::Gmm2ProducerCount(uint32_t waveIdx) const
{
    const __gm__ MegaMoeFixedGroupTiling &fixed = tilingData_->fixedGroupTiling;
    const int32_t decision = WaitEpochAcquire(
        FixedSyncSlot(workspaceGM_, tilingData_, kMegaMoeFixedSyncGmm2JoinSlot),
        static_cast<int32_t>(waveIdx + 1U));
    const uint32_t encodedWave = static_cast<uint32_t>(decision & kMegaMoeFixedGmm2JoinDecisionMask);
    const bool joined =
        (decision & kMegaMoeFixedGmm2JoinDecisionBit) != 0 && encodedWave != 0U && waveIdx + 1U >= encodedWave;
    return joined ? fixed.physicalAicNum : fixed.gmm2GroupSize;
}

template <typename OutputElement>
AICORE inline void Combine<OutputElement>::WaitGmm2Ready(uint32_t waveIdx, uint32_t progressWorkerIdx) const
{
    const __gm__ MegaMoeFixedGroupTiling &fixed = tilingData_->fixedGroupTiling;
    const uint32_t expectedEpoch = waveIdx + 1U;
    if (physicalBlockId_ == ReadyCoordinatorPhysicalBlock()) {
        const uint32_t producerCount = Gmm2ProducerCount(waveIdx);
        const uint32_t producerBaseOffset = producerCount == fixed.physicalAicNum ? 0U : fixed.gmm1GroupSize;
        CoordinateGroupConsumersMte(
            workspaceGM_, tilingData_, kMegaMoeFixedSyncGmm2ArrivalBase + producerBaseOffset,
            kMegaMoeFixedSyncCombineReadyBase, producerCount, fixed.gmm2GroupSize, waveIdx);
    } else {
        const uint32_t readySlot = kMegaMoeFixedSyncCombineReadyBase + progressWorkerIdx;
        WaitEpochAcquire(
            FixedSyncSlot(workspaceGM_, tilingData_, readySlot),
            static_cast<int32_t>(expectedEpoch));
    }
}

template <typename OutputElement>
AICORE inline void Combine<OutputElement>::PublishAssignedExpertProgress(uint32_t progressWorkerIdx,
                                                                         uint32_t readyWaveCount,
                                                                         uint32_t readyExpertCount) const
{
    const uint32_t lanesPerRank = LargeLanesPerRank();
    const uint32_t taskCount = DirectLargeTaskCount();
    const uint32_t workerCount = ProgressWorkerCount();
    for (uint32_t taskIdx = progressWorkerIdx; taskIdx < taskCount; taskIdx += workerCount) {
        const uint32_t ownerRank = taskIdx / lanesPerRank;
        const event_t eventId = static_cast<event_t>(taskIdx / workerCount);
        remoteWindow_.PublishRankReadyMte(static_cast<int32_t>(ownerRank), readyWaveCount, readyExpertCount,
                                          dataReadyEpoch_, false, eventId);
    }
}

template <typename OutputElement>
AICORE inline void Combine<OutputElement>::FinalizeRankStreamingLane(uint32_t progressWorkerIdx) const
{
    const uint32_t taskCount = DirectLargeTaskCount();
    const uint32_t activeWorkerCount = DirectLargeWorkerCount();
    if (progressWorkerIdx >= activeWorkerCount) {
        return;
    }
    const uint32_t lanesPerRank = LargeLanesPerRank();
    const uint32_t workerCount = ProgressWorkerCount();
    for (uint32_t taskIdx = progressWorkerIdx; taskIdx < taskCount; taskIdx += workerCount) {
        const uint32_t ownerRank = taskIdx / lanesPerRank;
        const event_t eventId = static_cast<event_t>(taskIdx / workerCount);
        remoteWindow_.PublishRankReadyMte(static_cast<int32_t>(ownerRank), tilingData_->fixedGroupTiling.totalWaveCount,
                                          expertPerRank_, dataReadyEpoch_, true, eventId);
    }
    pto::PtoSetWaitFlag<PIPE_MTE3, PIPE_S>();
    remoteWindow_.PublishLocalCombineDone(progressWorkerIdx, dataReadyEpoch_);

    if (progressWorkerIdx != 0U) {
        return;
    }
    remoteWindow_.WaitLocalCombineDoneMte(activeWorkerCount, dataReadyEpoch_);
}

template <typename OutputElement>
AICORE inline void Combine<OutputElement>::PrefetchDirectMetadata()
{
    const uint32_t metadataElems = rankSize_ * expertPerRank_;
    if (metadataElems == 0U || metadataElems > kDirectCombineMetadataMaxElems) {
        return;
    }
    PtoLoadVector<int32_t, kDirectCombineMetadataMaxElems>(kDirectCombineCumsumUbOffset, cumsumMMPtr_, metadataElems);
    PtoLoadVector<int32_t, kDirectCombineMetadataMaxElems>(kDirectCombinePreSumUbOffset, preSumBeforeRankPtr_,
                                                           metadataElems);
    pto::PtoSetWaitFlag<PIPE_MTE2, PIPE_S>();
    directMetadataCached_ = true;
}

template <typename OutputElement>
AICORE inline void Combine<OutputElement>::PrepareDirectExpert(uint32_t groupIdx)
{
    uint32_t rankRowBegin = 0U;
    const uint32_t cachedRankCount =
        rankSize_ < kMegaMoeExpertProgressMaxRanks ? rankSize_ : kMegaMoeExpertProgressMaxRanks;
    for (uint32_t srcRank = 0U; srcRank < cachedRankCount; ++srcRank) {
        const uint32_t metadataIdx = srcRank * expertPerRank_ + groupIdx;
        const uint32_t rankRowEnd = directMetadataCached_ ?
                                        static_cast<uint32_t>(PtoGetValue<int32_t, kDirectCombineMetadataMaxElems>(
                                            kDirectCombineCumsumUbOffset, metadataIdx)) :
                                        static_cast<uint32_t>(cumsumMMPtr_[metadataIdx]);
        directRankRowBegin_[srcRank] = rankRowBegin;
        directRankRowEnd_[srcRank] = rankRowEnd;
        directCompactRowBegin_[srcRank] =
            directMetadataCached_ ? static_cast<uint32_t>(PtoGetValue<int32_t, kDirectCombineMetadataMaxElems>(
                                        kDirectCombinePreSumUbOffset, metadataIdx)) :
                                    static_cast<uint32_t>(preSumBeforeRankPtr_[metadataIdx]);
        rankRowBegin = rankRowEnd;
    }
}

template <typename OutputElement>
AICORE inline void Combine<OutputElement>::StoreDirectTile(const GmmCommonTileInfo &tileInfo, uint32_t tileIndex) const
{
    const uint32_t tileRowBegin = tileInfo.blockRowStart;
    const uint32_t tileRowEnd = tileRowBegin + tileInfo.actualM;
    const uint32_t cachedRankCount =
        rankSize_ < kMegaMoeExpertProgressMaxRanks ? rankSize_ : kMegaMoeExpertProgressMaxRanks;
    for (uint32_t srcRank = 0U; srcRank < cachedRankCount; ++srcRank) {
        const uint32_t rankRowBegin = directRankRowBegin_[srcRank];
        const uint32_t rankRowEnd = directRankRowEnd_[srcRank];
        if (rankRowBegin >= tileRowEnd) {
            break;
        }
        if (rankRowEnd <= tileRowBegin) {
            continue;
        }
        const uint32_t intersectionBegin = rankRowBegin > tileRowBegin ? rankRowBegin : tileRowBegin;
        const uint32_t intersectionEnd = rankRowEnd < tileRowEnd ? rankRowEnd : tileRowEnd;
        if (intersectionEnd <= intersectionBegin) {
            continue;
        }
        const uint32_t rows = intersectionEnd - intersectionBegin;
        const uint32_t srcTileRow = intersectionBegin - tileRowBegin;
        const uint32_t compactRow = directCompactRowBegin_[srcRank] + intersectionBegin - rankRowBegin;
        if (compactRow >= routeElems_ || rows > routeElems_ - compactRow) {
            continue;
        }
        __gm__ OutputElement *dstBase = remoteOutputBase_[srcRank];
        if (dstBase == nullptr) {
            continue;
        }
        __gm__ OutputElement *dst = dstBase + static_cast<uint64_t>(compactRow) * problemK_ + tileInfo.blockColStart;
        const uint64_t srcOffset =
            Gmm2CombineSlotOffset(tileIndex) +
            static_cast<uint64_t>(srcTileRow) * kGmm2CombineCvTileCols * sizeof(bfloat16_t);
        Gmm2CombineCvTile srcTile(rows, tileInfo.actualN);
        pto::TASSIGN(srcTile, srcOffset);
        DirectCombineStrideStore<Gmm2CombineCvTile>(dst, srcTile.data(), rows,
                                                                              tileInfo.actualN, problemK_);
    }
}

template <typename OutputElement>
AICORE inline void Combine<OutputElement>::ConsumeDirectTile(const GmmCommonTileInfo &tileInfo)
{
    Gmm2CombineConsumerWait(cvPipe_);
    const uint32_t tileIndex = cvPipe_.cons.tileIndex;
    StoreDirectTile(tileInfo, tileIndex);
    Gmm2CombineConsumerRelease(cvPipe_);
}

template <typename OutputElement>
AICORE inline void Combine<OutputElement>::ProcessImpl(bool helperGroup)
{
    const __gm__ MegaMoeFixedGroupTiling &fixed = tilingData_->fixedGroupTiling;
    uint32_t firstWaveIdx = 0U;
    uint32_t tileCoreIdx = helperGroup ? physicalBlockId_ : physicalBlockId_ - fixed.gmm1GroupSize;
    uint32_t tileCoreNum = helperGroup ? fixed.physicalAicNum : fixed.gmm2GroupSize;
    MegaMoeCoreTileBalancer tileBalancer;
    SetCoreTileBalancerRange(tileBalancer, fixed.gmm1GroupSize, fixed.gmm2GroupSize);
    bool joined = helperGroup;
    if (helperGroup) {
        firstWaveIdx = HelperJoinWave();
        if (firstWaveIdx >= fixed.totalWaveCount) {
            return;
        }
        const MegaMoeExpertWaveRange firstWave =
            GetExpertWaveRange(firstWaveIdx, expertPerRank_, fixed.fullAicExpertsPerWave,
                                                       fixed.expertsPerWave, kMegaMoeFullAicGmm1WaveCount);
        ReplayPrimaryTileBalanceBefore(firstWave.begin, tileBalancer);
        SetCoreTileBalancerRange(tileBalancer, 0U, tileCoreNum);
    }

    for (uint32_t waveIdx = firstWaveIdx; waveIdx < fixed.totalWaveCount; ++waveIdx) {
        const MegaMoeExpertWaveRange wave = GetExpertWaveRange(
            waveIdx, expertPerRank_, fixed.fullAicExpertsPerWave, fixed.expertsPerWave,
            kMegaMoeFullAicGmm1WaveCount);
        if (!helperGroup && !joined && PrimaryJoinsWave(waveIdx)) {
            joined = true;
            tileCoreIdx = physicalBlockId_;
            tileCoreNum = fixed.physicalAicNum;
            SetCoreTileBalancerRange(tileBalancer, 0U, tileCoreNum);
        }
        for (uint32_t groupIdx = wave.begin; groupIdx < wave.end; ++groupIdx) {
            const uint32_t currentM = CurrentM(groupIdx);
            const uint32_t coreLoops = CoreLoops(currentM);
            const uint32_t startCoreIdx = SelectCoreTileStart(tileBalancer, coreLoops);
            const uint32_t startLoopIdx = GmmCommonStartLoopIdx(tileCoreIdx, tileCoreNum, startCoreIdx);
            if (startLoopIdx < coreLoops) {
                PrepareDirectExpert(groupIdx);
            }
            for (uint32_t loopIdx = startLoopIdx; loopIdx < coreLoops; loopIdx += tileCoreNum) {
                const GmmCommonTileInfo tileInfo = GmmCommonBuildTileInfoWithOffset<kGmm2CombineSwizzleOffset>(
                    currentM, problemK_, kMegaMoeGmmTileM, kMegaMoeGmmTileN, loopIdx);
                ConsumeDirectTile(tileInfo);
            }
            CommitCoreTileAssignment(tileBalancer, startCoreIdx, coreLoops);
        }
        Gmm2CombineConsumerFinishWave(cvPipe_);
        const uint32_t progressWorkerIdx = ProgressWorkerIndex();
        if (progressWorkerIdx < DirectLargeWorkerCount()) {
            WaitGmm2Ready(waveIdx, progressWorkerIdx);
            const uint32_t readyWaveCount = waveIdx + 1U;
            PublishAssignedExpertProgress(progressWorkerIdx, readyWaveCount, wave.end);
            if (readyWaveCount == fixed.totalWaveCount) {
                FinalizeRankStreamingLane(progressWorkerIdx);
            }
        }
    }
}

template <typename OutputElement>
AICORE inline void Combine<OutputElement>::ProcessFixed(uint32_t physicalBlockId)
{
    if ASCEND_IS_AIC {
        return;
    }
    physicalBlockId_ = physicalBlockId;
    // The Group2 AIV0 coordinator publishes the common start marker only
    // after local Dispatch and all direct-route metadata are ready.
    PrefetchDirectMetadata();
    ProcessImpl(physicalBlockId_ < tilingData_->fixedGroupTiling.gmm1GroupSize);
}

#endif // DISPATCH_MEGA_COMBINE_COMBINE_H
