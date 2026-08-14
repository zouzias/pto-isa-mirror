/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef DISPATCH_MEGA_COMBINE_GMM2_H
#define DISPATCH_MEGA_COMBINE_GMM2_H

#include "kernel_operator.h"

#include "dispatch_mega_combine_tiling.h"
#include "gmm2_combine_cv_pipe.h"
#include "gmm_common.h"
#include "utils/common_helpers.hpp"
#include "utils/const_args.hpp"
#include "utils/mega_expert_sync.hpp"
#include "utils/mega_wave_schedule.hpp"

using Gmm2Pipeline = GmmCommonPipeline;

template <typename InputElement>
class Gmm2 {
public:
    AICORE inline void Init(GM_ADDR weight2GM, GM_ADDR weightScale2GM, GM_ADDR expertTokenNumsGM, GM_ADDR workspaceGM,
                            const __gm__ MegaMoeTilingData *tilingData);
    AICORE inline void ProcessFixed(uint32_t groupLocalId, uint32_t groupSize);
    AICORE inline void ProcessFixedHelper(uint32_t groupLocalId);

private:
    AICORE inline uint32_t CoreLoops(uint32_t currentM) const
    {
        return GmmCommonCoreLoops(currentM, outputN_, kMegaMoeGmmTileM, kMegaMoeGmmTileN);
    }
    AICORE inline uint32_t StartLoopIdx(uint32_t startCoreIdx) const
    {
        return GmmCommonStartLoopIdx(coreIdx_, coreNum_, startCoreIdx);
    }
    AICORE inline GmmCommonTileInfo BuildTileInfo(uint32_t currentM, uint32_t loopIdx) const
    {
        return GmmCommonBuildTileInfoWithOffset<kGmm2CombineSwizzleOffset>(
            currentM, outputN_, kMegaMoeGmmTileM, kMegaMoeGmmTileN, loopIdx);
    }
    AICORE inline void RunGmmTile(Gmm2Pipeline &gmmPipeline, Gmm2CombineCvPipe &cvPipe,
                                  uint32_t groupIdx, uint32_t groupBase, const GmmCommonTileInfo &tileInfo) const
    {
        const uint32_t scaleLeadingDim = inputK_ / kMegaMoeMxGroupSize;
        const uint64_t gmOffsetA = static_cast<uint64_t>(groupBase + tileInfo.blockRowStart) * inputK_;
        const uint64_t gmOffsetAScale = static_cast<uint64_t>(groupBase + tileInfo.blockRowStart) * scaleLeadingDim;
        const uint64_t weightExpertStride = static_cast<uint64_t>(outputN_) * inputK_;
        const uint64_t weightScaleExpertStride = static_cast<uint64_t>(outputN_) * scaleLeadingDim;
        const uint64_t gmOffsetB = static_cast<uint64_t>(groupIdx) * weightExpertStride +
                                   static_cast<uint64_t>(tileInfo.blockColStart) * inputK_;
        const uint64_t gmOffsetBScale = static_cast<uint64_t>(groupIdx) * weightScaleExpertStride +
                                        static_cast<uint64_t>(tileInfo.blockColStart) * scaleLeadingDim;
        gmmPipeline.RunDirect(cvPipe, gmSwigluAPtr_ + gmOffsetA, gmSwigluScalePtr_ + gmOffsetAScale,
                              weight2Ptr_ + gmOffsetB, weightScale2Ptr_ + gmOffsetBScale, tileInfo.actualM,
                              tileInfo.actualN, inputK_, inputK_, scaleLeadingDim, inputK_, scaleLeadingDim);
    }
    AICORE inline uint32_t GroupBaseBefore(uint32_t groupIdx) const;
    AICORE inline void ReplayPrimaryTileBalanceBefore(
        uint32_t groupIdx, MegaMoeCoreTileBalancer &tileBalancer) const;
    AICORE inline int32_t PrimaryJoinDecision(uint32_t waveIdx) const;
    AICORE inline uint32_t HelperJoinWave() const;
    AICORE inline void ProcessImpl(bool helperGroup);

    GM_ADDR workspaceGM_ = nullptr;
    const __gm__ MegaMoeTilingData *tilingData_ = nullptr;
    __gm__ float8_e4m3_t *gmSwigluAPtr_ = nullptr;
    __gm__ float8_e8m0_t *gmSwigluScalePtr_ = nullptr;
    __gm__ float8_e4m3_t *weight2Ptr_ = nullptr;
    __gm__ float8_e8m0_t *weightScale2Ptr_ = nullptr;
    __gm__ int32_t *cumsumMMPtr_ = nullptr;
    uint32_t inputK_ = 0;
    uint32_t outputN_ = 0;
    uint32_t expertPerRank_ = 0;
    uint32_t rankSize_ = 0;
    uint32_t coreIdx_ = 0;
    uint32_t coreNum_ = 1;
    uint32_t primaryLocalId_ = 0;
};

template <typename InputElement>
AICORE inline void Gmm2<InputElement>::Init(GM_ADDR weight2GM, GM_ADDR weightScale2GM, GM_ADDR expertTokenNumsGM,
                                            GM_ADDR workspaceGM, const __gm__ MegaMoeTilingData *tilingData)
{
    (void)sizeof(InputElement);
    (void)expertTokenNumsGM;
    workspaceGM_ = workspaceGM;
    tilingData_ = tilingData;
    const uint32_t problemN = tilingData_->megaMoeInfo.N;
    const uint32_t problemK = tilingData_->megaMoeInfo.K;
    inputK_ = problemN / 2U;
    outputN_ = problemK;
    expertPerRank_ = tilingData_->megaMoeInfo.expertPerRank;
    rankSize_ = tilingData_->runtimeInfo.rankSize;
    coreIdx_ = get_block_idx();
    coreNum_ = get_block_num();

    gmSwigluAPtr_ = reinterpret_cast<__gm__ float8_e4m3_t *>(workspaceGM + tilingData_->swigluTiling.gmSwigluAOffset);
    gmSwigluScalePtr_ =
        reinterpret_cast<__gm__ float8_e8m0_t *>(workspaceGM + tilingData_->swigluTiling.gmSwigluScaleOffset);
    weight2Ptr_ = reinterpret_cast<__gm__ float8_e4m3_t *>(weight2GM);
    weightScale2Ptr_ = reinterpret_cast<__gm__ float8_e8m0_t *>(weightScale2GM);
    cumsumMMPtr_ = reinterpret_cast<__gm__ int32_t *>(workspaceGM + tilingData_->frontReorderTiling.cumsumMMOffset);
}

template <typename InputElement>
AICORE inline void Gmm2<InputElement>::ProcessFixed(uint32_t groupLocalId, uint32_t groupSize)
{
    coreIdx_ = groupLocalId;
    coreNum_ = groupSize;
    primaryLocalId_ = groupLocalId;
    ProcessImpl(false);
}

template <typename InputElement>
AICORE inline void Gmm2<InputElement>::ProcessFixedHelper(uint32_t groupLocalId)
{
    coreIdx_ = groupLocalId;
    coreNum_ = tilingData_->fixedGroupTiling.physicalAicNum;
    primaryLocalId_ = groupLocalId;
    ProcessImpl(true);
}

template <typename InputElement>
AICORE inline uint32_t Gmm2<InputElement>::GroupBaseBefore(uint32_t groupIdx) const
{
    uint32_t groupBase = 0U;
    for (uint32_t expert = 0U; expert < groupIdx; ++expert) {
        groupBase += MoeCurrentMRaw(cumsumMMPtr_, rankSize_, expertPerRank_, expert);
    }
    return groupBase;
}

template <typename InputElement>
AICORE inline void Gmm2<InputElement>::ReplayPrimaryTileBalanceBefore(
    uint32_t groupIdx, MegaMoeCoreTileBalancer &tileBalancer) const
{
    const __gm__ MegaMoeFixedGroupTiling &fixed = tilingData_->fixedGroupTiling;
    SetCoreTileBalancerRange(tileBalancer, fixed.gmm1GroupSize, fixed.gmm2GroupSize);
    for (uint32_t expert = 0U; expert < groupIdx; ++expert) {
        const uint32_t currentM = MoeCurrentMRaw(cumsumMMPtr_, rankSize_, expertPerRank_, expert);
        const uint32_t coreLoops = CoreLoops(currentM);
        const uint32_t startCoreIdx = SelectCoreTileStart(tileBalancer, coreLoops);
        CommitCoreTileAssignment(tileBalancer, startCoreIdx, coreLoops);
    }
}

template <typename InputElement>
AICORE inline int32_t Gmm2<InputElement>::PrimaryJoinDecision(uint32_t waveIdx) const
{
    volatile __gm__ int32_t *joinSlot =
        FixedSyncSlot(workspaceGM_, tilingData_, kMegaMoeFixedSyncGmm2JoinSlot);
    const int32_t expectedDecision = static_cast<int32_t>(waveIdx + 1U);
    if (primaryLocalId_ != 0U) {
        return WaitEpochAcquire(joinSlot, expectedDecision);
    }

    int32_t done = ReadScalarEpoch(
        FixedSyncSlot(workspaceGM_, tilingData_, kMegaMoeFixedSyncGmm1DoneSlot));
    if (tilingData_->fixedGroupTiling.totalWaveCount <= kMegaMoeFullAicGmm1WaveCount &&
        done < kMegaMoeFixedGmm1DoneMarker) {
        done = WaitEpochAcquire(
            FixedSyncSlot(workspaceGM_, tilingData_, kMegaMoeFixedSyncGmm1DoneSlot),
            kMegaMoeFixedGmm1DoneMarker);
    }
    const int32_t decision =
        done >= kMegaMoeFixedGmm1DoneMarker ? (kMegaMoeFixedGmm2JoinDecisionBit | expectedDecision) : expectedDecision;
    PublishScalarEpoch(joinSlot, decision);
    return decision;
}

template <typename InputElement>
AICORE inline uint32_t Gmm2<InputElement>::HelperJoinWave() const
{
    const int32_t decision = WaitEpochAcquire(
        FixedSyncSlot(workspaceGM_, tilingData_, kMegaMoeFixedSyncGmm2JoinSlot),
        kMegaMoeFixedGmm2JoinDecisionBit);
    const int32_t encodedWave = decision & kMegaMoeFixedGmm2JoinDecisionMask;
    return encodedWave > 0 ? static_cast<uint32_t>(encodedWave - 1) : tilingData_->fixedGroupTiling.totalWaveCount;
}

template <typename InputElement>
AICORE inline void Gmm2<InputElement>::ProcessImpl(bool helperGroup)
{
    if ASCEND_IS_AIV {
        return;
    }
    Gmm2Pipeline gmmPipeline;
    Gmm2CombineCvPipe cvPipe;
    const __gm__ MegaMoeFixedGroupTiling &fixed = tilingData_->fixedGroupTiling;
    uint32_t groupBase = 0U;
    MegaMoeCoreTileBalancer tileBalancer;
    SetCoreTileBalancerRange(tileBalancer, fixed.gmm1GroupSize, fixed.gmm2GroupSize);
    uint32_t firstWaveIdx = 0U;
    bool joinedGroup = helperGroup;
    if (helperGroup) {
        firstWaveIdx = HelperJoinWave();
        if (firstWaveIdx >= fixed.totalWaveCount) {
            return;
        }
        const MegaMoeExpertWaveRange firstWave =
            GetExpertWaveRange(firstWaveIdx, expertPerRank_, fixed.fullAicExpertsPerWave,
                                                       fixed.expertsPerWave, kMegaMoeFullAicGmm1WaveCount);
        groupBase = GroupBaseBefore(firstWave.begin);
        ReplayPrimaryTileBalanceBefore(firstWave.begin, tileBalancer);
        SetCoreTileBalancerRange(tileBalancer, 0U, fixed.physicalAicNum);
        coreNum_ = fixed.physicalAicNum;
    }
    for (uint32_t waveIdx = firstWaveIdx; waveIdx < fixed.totalWaveCount; ++waveIdx) {
        const MegaMoeExpertWaveRange wave = GetExpertWaveRange(
            waveIdx, expertPerRank_, fixed.fullAicExpertsPerWave, fixed.expertsPerWave,
            kMegaMoeFullAicGmm1WaveCount);
        if (!helperGroup && !joinedGroup) {
            const int32_t decision = PrimaryJoinDecision(waveIdx);
            const uint32_t encodedJoinWave = static_cast<uint32_t>(decision & kMegaMoeFixedGmm2JoinDecisionMask);
            const bool joinThisWave = (decision & kMegaMoeFixedGmm2JoinDecisionBit) != 0 && encodedJoinWave != 0U &&
                                      waveIdx + 1U >= encodedJoinWave;
            if (joinThisWave) {
                joinedGroup = true;
                coreIdx_ = fixed.gmm1GroupSize + primaryLocalId_;
                coreNum_ = fixed.physicalAicNum;
                SetCoreTileBalancerRange(tileBalancer, 0U, coreNum_);
            }
        }
        const uint32_t readyEpoch = waveIdx + 1U;
        const uint32_t readyLocalId = primaryLocalId_ % fixed.gmm2GroupSize;
        WaitEpochAcquire(
            FixedSyncSlot(workspaceGM_, tilingData_,
                                                    kMegaMoeFixedSyncGmm2ReadyBase + readyLocalId),
            static_cast<int32_t>(readyEpoch));
        for (uint32_t groupIdx = wave.begin; groupIdx < wave.end; ++groupIdx) {
            const uint32_t currentM = MoeCurrentMRaw(cumsumMMPtr_, rankSize_, expertPerRank_, groupIdx);
            const uint32_t coreLoops = CoreLoops(currentM);
            const uint32_t startCoreIdx = SelectCoreTileStart(tileBalancer, coreLoops);
            const uint32_t startLoopIdx = GmmCommonStartLoopIdx(coreIdx_, coreNum_, startCoreIdx);
            for (uint32_t loopIdx = startLoopIdx; loopIdx < coreLoops; loopIdx += coreNum_) {
                const GmmCommonTileInfo tileInfo = BuildTileInfo(currentM, loopIdx);
                RunGmmTile(gmmPipeline, cvPipe, groupIdx, groupBase, tileInfo);
            }
            groupBase += currentM;
            CommitCoreTileAssignment(tileBalancer, startCoreIdx, coreLoops);
        }
        gmmPipeline.SynchronizeBlock();
        Gmm2CombineProducerDrainWave(cvPipe);
        const uint32_t arrivalLocalId = joinedGroup ? coreIdx_ : fixed.gmm1GroupSize + primaryLocalId_;
        PublishGroupArrival(
            workspaceGM_, tilingData_, kMegaMoeFixedSyncGmm2ArrivalBase, arrivalLocalId, waveIdx);
    }
    if (!helperGroup && !joinedGroup && primaryLocalId_ == 0U) {
        const int32_t sentinel = kMegaMoeFixedGmm2JoinDecisionBit + static_cast<int32_t>(fixed.totalWaveCount + 1U);
        PublishScalarEpoch(
            FixedSyncSlot(workspaceGM_, tilingData_, kMegaMoeFixedSyncGmm2JoinSlot),
            sentinel);
    }
}

#endif // DISPATCH_MEGA_COMBINE_GMM2_H
