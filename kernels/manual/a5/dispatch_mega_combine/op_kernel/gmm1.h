/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef DISPATCH_MEGA_COMBINE_GMM1_H
#define DISPATCH_MEGA_COMBINE_GMM1_H

#include "kernel_operator.h"

#include "dispatch_mega_combine_tiling.h"
#include "gmm_common.h"
#include "gmm1_swiglu_cv_pipe.h"
#include "utils/common_helpers.hpp"
#include "utils/const_args.hpp"
#include "utils/mega_expert_sync.hpp"
#include "utils/mega_wave_schedule.hpp"

using Gmm1Pipeline = GmmCommonPipeline;

template <typename InputElement>
class Gmm1 {
public:
    AICORE inline void Init(GM_ADDR weight1GM, GM_ADDR weightScale1GM, GM_ADDR expertTokenNumsGM, GM_ADDR workspaceGM,
                            const __gm__ MegaMoeTilingData *tilingData);
    AICORE inline void ProcessFixed(uint32_t groupLocalId, uint32_t groupSize);

private:
    AICORE inline uint32_t PairTileN() const
    {
        return kMegaMoeGmmTileN;
    }

    AICORE inline uint32_t CoreLoops(uint32_t currentM) const
    {
        return GmmCommonCoreLoops(currentM, outputN_, kMegaMoeGmmTileM, PairTileN());
    }
    AICORE inline uint32_t StartLoopIdx(uint32_t startCoreIdx) const
    {
        return GmmCommonStartLoopIdx(coreIdx_, coreNum_, startCoreIdx);
    }
    AICORE inline volatile __gm__ int32_t *DispatchReadyCountSlot(uint32_t groupIdx, uint32_t blockM) const
    {
        const __gm__ MegaMoeDispatchTiling &dispatch = tilingData_->dispatchTiling;
        const uint64_t byteOffset = dispatch.readyCountOffset +
                                    static_cast<uint64_t>(groupIdx) * dispatch.readyCountExpertStrideBytes +
                                    static_cast<uint64_t>(blockM) * kMegaMoeReadyCountSlotBytes;
        return reinterpret_cast<volatile __gm__ int32_t *>(workspaceGM_ + byteOffset);
    }
    AICORE inline void WaitDispatchTileReady(uint32_t groupIdx, const GmmCommonTileInfo &tileInfo) const
    {
        volatile __gm__ int32_t *slot = DispatchReadyCountSlot(groupIdx, tileInfo.blockM);
        const int32_t targetRows = static_cast<int32_t>(tileInfo.actualM);
        int32_t observedRows = 0;
        do {
            dcci((__gm__ void *)slot, SINGLE_CACHE_LINE);
            dsb(DSB_DDR);
            observedRows = *slot;
            if (observedRows != targetRows) {
                EpochPollBackoff();
            }
        } while (observedRows != targetRows);
        dsb(DSB_DDR);
    }
    AICORE inline GmmCommonTileInfo BuildTileInfo(uint32_t currentM, uint32_t loopIdx) const
    {
        return GmmCommonBuildTileInfo(currentM, outputN_, kMegaMoeGmmTileM, PairTileN(), loopIdx);
    }
    AICORE inline void RunGmmPair(Gmm1Pipeline &gmmPipeline, Gmm1SwigluCvPipe &cvPipe,
                                  uint32_t groupIdx, uint32_t groupBase, const GmmCommonTileInfo &tileInfo) const
    {
        const uint32_t scaleLeadingDim = problemK_ / kMegaMoeMxGroupSize;
        const uint64_t gmOffsetA = static_cast<uint64_t>(groupBase + tileInfo.blockRowStart) * problemK_;
        const uint64_t gmOffsetAScale = static_cast<uint64_t>(groupBase + tileInfo.blockRowStart) * scaleLeadingDim;
        const uint64_t weightExpertStride = static_cast<uint64_t>(problemN_) * problemK_;
        const uint64_t weightScaleExpertStride = static_cast<uint64_t>(problemN_) * scaleLeadingDim;
        const uint64_t weightExpertBase = static_cast<uint64_t>(groupIdx) * weightExpertStride;
        const uint64_t weightScaleExpertBase = static_cast<uint64_t>(groupIdx) * weightScaleExpertStride;
        const uint64_t xOffset = weightExpertBase + static_cast<uint64_t>(tileInfo.blockColStart) * problemK_;
        const uint64_t gateOffset =
            weightExpertBase + static_cast<uint64_t>(outputN_ + tileInfo.blockColStart) * problemK_;
        const uint64_t xScaleOffset =
            weightScaleExpertBase + static_cast<uint64_t>(tileInfo.blockColStart) * scaleLeadingDim;
        const uint64_t gateScaleOffset =
            weightScaleExpertBase + static_cast<uint64_t>(outputN_ + tileInfo.blockColStart) * scaleLeadingDim;

        gmmPipeline.RunPairDirect(cvPipe, gmAPtr_ + gmOffsetA, gmAScalePtr_ + gmOffsetAScale, weight1Ptr_ + xOffset,
                                  weightScale1Ptr_ + xScaleOffset, weight1Ptr_ + gateOffset,
                                  weightScale1Ptr_ + gateScaleOffset, tileInfo.actualM, tileInfo.actualN, problemK_,
                                  problemK_, scaleLeadingDim, problemK_, scaleLeadingDim);
    }
    AICORE inline void ProcessImpl();

    GM_ADDR workspaceGM_ = nullptr;
    const __gm__ MegaMoeTilingData *tilingData_ = nullptr;

    __gm__ float8_e4m3_t *gmAPtr_ = nullptr;
    __gm__ float8_e8m0_t *gmAScalePtr_ = nullptr;
    __gm__ float8_e4m3_t *weight1Ptr_ = nullptr;
    __gm__ float8_e8m0_t *weightScale1Ptr_ = nullptr;
    __gm__ int32_t *cumsumMMPtr_ = nullptr;

    uint32_t problemK_ = 0;
    uint32_t problemN_ = 0;
    uint32_t outputN_ = 0;
    uint32_t expertPerRank_ = 0;
    uint32_t rankSize_ = 0;
    uint32_t coreIdx_ = 0;
    uint32_t coreNum_ = 1;
};

template <typename InputElement>
AICORE inline void Gmm1<InputElement>::Init(GM_ADDR weight1GM, GM_ADDR weightScale1GM, GM_ADDR expertTokenNumsGM,
                                            GM_ADDR workspaceGM, const __gm__ MegaMoeTilingData *tilingData)
{
    (void)expertTokenNumsGM;
    workspaceGM_ = workspaceGM;
    tilingData_ = tilingData;

    problemK_ = tilingData_->megaMoeInfo.K;
    problemN_ = tilingData_->megaMoeInfo.N;
    outputN_ = problemN_ / 2U;
    expertPerRank_ = tilingData_->megaMoeInfo.expertPerRank;
    rankSize_ = tilingData_->runtimeInfo.rankSize;
    coreIdx_ = get_block_idx();
    coreNum_ = get_block_num();

    gmAPtr_ = reinterpret_cast<__gm__ float8_e4m3_t *>(workspaceGM + tilingData_->dispatchTiling.gmAOffset);
    gmAScalePtr_ = reinterpret_cast<__gm__ float8_e8m0_t *>(workspaceGM + tilingData_->dispatchTiling.gmAScaleOffset);
    weight1Ptr_ = reinterpret_cast<__gm__ float8_e4m3_t *>(weight1GM);
    weightScale1Ptr_ = reinterpret_cast<__gm__ float8_e8m0_t *>(weightScale1GM);
    cumsumMMPtr_ = reinterpret_cast<__gm__ int32_t *>(workspaceGM + tilingData_->frontReorderTiling.cumsumMMOffset);
}
template <typename InputElement>
AICORE inline void Gmm1<InputElement>::ProcessFixed(uint32_t groupLocalId, uint32_t groupSize)
{
    coreIdx_ = groupLocalId;
    coreNum_ = groupSize;
    ProcessImpl();
}

template <typename InputElement>
AICORE inline void Gmm1<InputElement>::ProcessImpl()
{
    if ASCEND_IS_AIV {
        return;
    }
    WaitEpochAcquire(
        FixedSyncSlot(workspaceGM_, tilingData_, kMegaMoeFixedSyncFrontMetadataReadySlot),
        kMegaMoeFixedFrontMetadataReadyMarker);
    Gmm1SwigluCvPipe cvPipe(nullptr, kGmm1SwigluCvBufferOffset, 0U);
    Gmm1Pipeline gmmPipeline;
    uint32_t groupBase = 0;
    MegaMoeCoreTileBalancer tileBalancer;
    const __gm__ MegaMoeFixedGroupTiling &fixed = tilingData_->fixedGroupTiling;
    const uint32_t minimumFullAicWaveCount =
        kMegaMoeFullAicGmm1WaveCount < fixed.totalWaveCount ? kMegaMoeFullAicGmm1WaveCount : fixed.totalWaveCount;
    bool splitForGmm2 = false;
    for (uint32_t waveIdx = 0U; waveIdx < fixed.totalWaveCount; ++waveIdx) {
        coreNum_ = splitForGmm2 ? fixed.gmm1GroupSize : fixed.physicalAicNum;
        if (coreIdx_ >= coreNum_) {
            break;
        }
        SetCoreTileBalancerRange(tileBalancer, 0U, coreNum_);
        const MegaMoeExpertWaveRange wave = GetExpertWaveRange(
            waveIdx, expertPerRank_, fixed.fullAicExpertsPerWave, fixed.expertsPerWave,
            kMegaMoeFullAicGmm1WaveCount);
        for (uint32_t groupIdx = wave.begin; groupIdx < wave.end; ++groupIdx) {
            const uint32_t currentM = MoeCurrentMRaw(cumsumMMPtr_, rankSize_, expertPerRank_, groupIdx);
            const uint32_t coreLoops = CoreLoops(currentM);
            const uint32_t startCoreIdx = SelectCoreTileStart(tileBalancer, coreLoops);
            const uint32_t startLoopIdx = StartLoopIdx(startCoreIdx);
            for (uint32_t loopIdx = startLoopIdx; loopIdx < coreLoops; loopIdx += coreNum_) {
                const GmmCommonTileInfo tileInfo = BuildTileInfo(currentM, loopIdx);
                WaitDispatchTileReady(groupIdx, tileInfo);
                RunGmmPair(gmmPipeline, cvPipe, groupIdx, groupBase, tileInfo);
            }
            groupBase += currentM;
            CommitCoreTileAssignment(tileBalancer, startCoreIdx, coreLoops);
        }
        // The CV FIFO carries tile completion; retain a per-AIC wave marker for
        // zero-tile participants and the existing wave-to-GMM2 control path.
        pipe_barrier(PIPE_FIX);
        PublishGroupArrival(workspaceGM_, tilingData_, kMegaMoeFixedSyncGmm1ArrivalBase,
                                                      coreIdx_, waveIdx);
        const uint32_t completedWaveCount = waveIdx + 1U;
        const bool canSplitAtBoundary = completedWaveCount >= minimumFullAicWaveCount &&
                                        completedWaveCount < fixed.totalWaveCount;
        if (!splitForGmm2 && canSplitAtBoundary) {
            const int32_t decision = CoordinateGmm1SplitDecision(
                workspaceGM_, tilingData_, coreIdx_, completedWaveCount);
            splitForGmm2 =
                Gmm1SplitDecisionEnabled(decision, completedWaveCount);
        }
    }
    pipe_barrier(PIPE_FIX);
    Gmm1SwigluProducerDrain(cvPipe);
    if (coreIdx_ < fixed.gmm1GroupSize) {
        pipe_barrier(PIPE_ALL);
        PublishScalarEpoch(
            FixedSyncSlot(workspaceGM_, tilingData_,
                                                    kMegaMoeFixedSyncGmm1ArrivalBase + coreIdx_),
            static_cast<int32_t>(fixed.totalWaveCount + kMegaMoeFixedGmm1GroupDoneEpochOffset));
    }
}

#endif // DISPATCH_MEGA_COMBINE_GMM1_H
