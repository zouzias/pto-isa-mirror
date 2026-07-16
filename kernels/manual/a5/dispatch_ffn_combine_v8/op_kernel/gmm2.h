/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef DISPATCH_FFN_COMBINE_V8_GMM2_H
#define DISPATCH_FFN_COMBINE_V8_GMM2_H

#include "kernel_operator.h"

#include "dispatch_ffn_combine_tiling.h"
#include "device_debug.h"
#include "gmm2_combine_cv_pipe.h"
#include "gmm_common.h"
#include "kernel_launch.hpp"
#include "profile_debug_config.h"
#include "profile_expert_wall.hpp"
#include "utils/common_helpers.hpp"
#include "utils/const_args.hpp"
#include "utils/pto_sync_substrate.hpp"

namespace dispatch_ffn_combine_v8 {

constexpr uint32_t kGmm2LayoutVersion = 1U;
constexpr uint32_t kGmm2CommonReuseMode = 1U;
constexpr uint32_t kGmm2WaitSourceV2COnly = 1U;
constexpr uint32_t kGmm2InvalidTask = kGmmCommonInvalidTask;
using Gmm2Pipeline = GmmCommonPipeline;

template <typename InputElement>
class Gmm2 {
public:
    AICORE inline void Init(GM_ADDR weight2GM, GM_ADDR scale2GM, GM_ADDR expertTokenNumsGM, GM_ADDR workspaceGM,
                            const __gm__ DispatchFFNCombineTilingData *tilingData,
                            volatile __gm__ uint64_t *profileEntry = nullptr);
    AICORE inline void Process();

private:
    friend struct DeviceDebug;
    AICORE inline bool CombineReadyEnabled() const
    {
        return stageNum_ >= 13U;
    }
    AICORE inline uint32_t SwigluSegmentNum() const
    {
        return expertPerRank_ <= 1U ? expertPerRank_ : 2U;
    }
    AICORE inline uint32_t SwigluEpilogueGranularity() const
    {
        if (expertPerRank_ <= 1U) {
            return expertPerRank_;
        }
        return expertPerRank_ <= 4U ? expertPerRank_ - 1U : expertPerRank_ - 3U;
    }
    AICORE inline uint32_t SwigluSegmentStartExpert(uint32_t segmentIdx) const
    {
        return segmentIdx == 0U ? 0U : SwigluEpilogueGranularity();
    }
    AICORE inline uint32_t SwigluSegmentEndExpert(uint32_t segmentIdx) const
    {
        return segmentIdx == 0U && SwigluSegmentNum() == 2U ? SwigluEpilogueGranularity() : expertPerRank_;
    }
    AICORE inline uint16_t V2CFlagId(uint32_t segmentIdx) const
    {
        (void)segmentIdx;
        return V5_V2C_HARD_FLAG_BASE;
    }
    AICORE inline void WaitV2CReady(uint32_t segmentIdx) const
    {
        CrossCoreWaitFlag<0x2>(V2CFlagId(segmentIdx));
        pipe_barrier(PIPE_ALL);
    }
    AICORE inline uint32_t CurrentMRaw(uint32_t groupIdx) const
    {
        return static_cast<uint32_t>(cumsumMMPtr_[static_cast<uint64_t>(rankSize_ - 1U) * expertPerRank_ + groupIdx]);
    }
    AICORE inline uint32_t ExpertTokenNums(uint32_t groupIdx) const
    {
        return static_cast<uint32_t>(expertTokenNumsPtr_[groupIdx]);
    }
    AICORE inline uint32_t ClipCurrentM(uint32_t currentMRaw, uint32_t groupBase) const
    {
        if (groupBase >= maxOutputSize_) {
            return 0;
        }
        const uint32_t remaining = maxOutputSize_ - groupBase;
        return currentMRaw > remaining ? remaining : currentMRaw;
    }
    AICORE inline uint32_t TileM(uint32_t currentM) const
    {
        return GmmCommonTileM(currentM, tilingData_->gmm2Tiling.l1TileM);
    }
    AICORE inline uint32_t TileN() const
    {
        return GmmCommonTileN(outputN_, tilingData_->gmm2Tiling.l1TileN);
    }
    AICORE inline uint32_t CoreLoops(uint32_t currentM) const
    {
        return GmmCommonCoreLoops(currentM, outputN_, tilingData_->gmm2Tiling.l1TileM, tilingData_->gmm2Tiling.l1TileN);
    }
    AICORE inline uint32_t StartLoopIdx(uint32_t startCoreIdx) const
    {
        return GmmCommonStartLoopIdx(coreIdx_, coreNum_, startCoreIdx);
    }
    AICORE inline uint32_t AssignedTileCount(uint32_t startLoopIdx, uint32_t coreLoops) const
    {
        return GmmCommonAssignedTileCount(startLoopIdx, coreLoops, coreNum_);
    }
    AICORE inline bool Gmm2CombineCvDirectEnabled() const
    {
        return stageNum_ >= 13U;
    }
    struct Gmm2CvDirectStore {
        Gmm2CombineCvPipe &pipe;
        volatile __gm__ uint64_t *profileEntry;

        AICORE inline Gmm2CvDirectStore(Gmm2CombineCvPipe &cvPipe, volatile __gm__ uint64_t *profile)
            : pipe(cvPipe), profileEntry(profile)
        {}

        AICORE inline uint64_t ProfileStart() const
        {
#if DISPATCH_FFN_COMBINE_V8_ENABLE_DETAIL_PROFILE
            return profileEntry == nullptr ? 0U : get_sys_cnt();
#else
            return 0U;
#endif
        }

        AICORE inline void ProfileEnd(uint64_t start) const
        {
#if DISPATCH_FFN_COMBINE_V8_ENABLE_DETAIL_PROFILE
            if (profileEntry == nullptr || start == 0U) {
                return;
            }
            const uint64_t end = get_sys_cnt();
            const size_t base = kDispatchFFNCombineProfileGmm2CvDetailBase +
                                DISPATCH_FFN_COMBINE_PROFILE_GMM2_CV_DETAIL_STORE *
                                    kDispatchFFNCombineProfileGmm2CvDetailFieldCount;
            if (profileEntry[base + 3U] == 0U) {
                profileEntry[base] = start;
            }
            profileEntry[base + 1U] = end;
            profileEntry[base + 2U] += end - start;
            profileEntry[base + 3U] += 1U;
#else
            (void)start;
#endif
        }

        template <typename ElementA, typename ElementC, typename ElementAccumulator, int Rows, int Cols>
        AICORE inline void Store(uint64_t accOffset, uint64_t scaleOffset, uint32_t validRow, uint32_t validCol) const
        {
            using AccTile = pto::TileAccCompact<ElementAccumulator, Rows, Cols, pto::DYNAMIC, pto::DYNAMIC>;
            using CvTile =
                pto::Tile<pto::TileType::Vec, ElementC, Rows, Cols, pto::BLayout::RowMajor, pto::DYNAMIC,
                          pto::DYNAMIC, pto::SLayout::NoneBox>;
            using ScalingTile = pto::Tile<pto::TileType::Scaling, uint64_t, 1, Cols, pto::BLayout::RowMajor, 1,
                                          pto::DYNAMIC, pto::SLayout::NoneBox>;

            const bool waitFree =
                pipe.prod.getAllocateStatus() && Gmm2CombineCvPipe::shouldWaitFree(pipe.prod.tileIndex);
            if (waitFree) {
                pipe.prod.template allocate<pto::TileSplitAxis::TILE_UP_DOWN>();
            }

            AccTile accTile(validRow, validCol);
            CvTile cvTile(validRow, validCol);
            ScalingTile scalingTile(validCol);
            pto::TASSIGN(accTile, accOffset);
            pto::TASSIGN(cvTile, V8_GMM2_COMBINE_CV_SLOT_OFFSET);
            pto::TASSIGN(scalingTile, scaleOffset);

            if constexpr (std::is_same_v<ElementA, int8_t>) {
                pto::TMOV<CvTile, AccTile, ScalingTile, pto::AccToVecMode::SingleModeVec0>(cvTile, accTile,
                                                                                           scalingTile);
                pto::TMOV<CvTile, AccTile, ScalingTile, pto::AccToVecMode::SingleModeVec1>(cvTile, accTile,
                                                                                           scalingTile);
            } else {
                pto::TMOV<CvTile, AccTile, pto::AccToVecMode::SingleModeVec0>(cvTile, accTile);
                pto::TMOV<CvTile, AccTile, pto::AccToVecMode::SingleModeVec1>(cvTile, accTile);
            }
            pipe_barrier(PIPE_ALL);
            ++pipe.prod.tileIndex;
            if (pipe.prod.getRecordStatus()) {
                pipe.prod.template record<pto::TileSplitAxis::TILE_UP_DOWN>();
            }
        }
    };
    AICORE inline void GetBlockCoordMN(uint32_t loopIdx, uint32_t tileM, uint32_t tileN, uint32_t &blockM,
                                       uint32_t &blockN) const
    {
        GmmCommonGetBlockCoordMN(loopIdx, tileM, tileN, blockM, blockN);
    }
    AICORE inline void GetActualBlockShapeMN(uint32_t blockM, uint32_t blockN, uint32_t tileM, uint32_t tileN,
                                             uint32_t currentM, uint32_t &actualM, uint32_t &actualN) const
    {
        GmmCommonGetActualBlockShapeMN(blockM, blockN, tileM, tileN, currentM, outputN_,
                                       tilingData_->gmm2Tiling.l1TileM, tilingData_->gmm2Tiling.l1TileN, actualM,
                                       actualN);
    }
    AICORE inline void RunGmmTile(Gmm2Pipeline &gmmPipeline, uint32_t groupIdx, uint32_t groupBase, uint32_t currentM,
                                  uint32_t loopIdx) const
    {
        GmmCommonRunTile(gmmPipeline, gmPermutedTokenPtr_, weight2Ptr_, gmm2OutputPtr_, scale2Ptr_, groupIdx, groupBase,
                         currentM, loopIdx, outputN_, inputK_, inputK_, inputK_, outputN_, outputN_,
                         tilingData_->gmm2Tiling.l1TileM, tilingData_->gmm2Tiling.l1TileN);
    }
    AICORE inline void RunGmmTileDirect(Gmm2Pipeline &gmmPipeline, uint32_t groupIdx, uint32_t groupBase,
                                        uint32_t currentM, uint32_t loopIdx, const Gmm2CvDirectStore &cvDirect) const
    {
        GmmCommonRunTileDirect(gmmPipeline, gmPermutedTokenPtr_, weight2Ptr_, gmm2OutputPtr_, scale2Ptr_, groupIdx,
                               groupBase, currentM, loopIdx, outputN_, inputK_, inputK_, inputK_, outputN_, outputN_,
                               tilingData_->gmm2Tiling.l1TileM, tilingData_->gmm2Tiling.l1TileN, cvDirect);
    }
    AICORE inline void BuildSegmentMetadata(uint32_t segmentIdx, uint32_t &segmentStartExpert,
                                            uint32_t &segmentEndExpert, uint32_t &segmentRowBase, uint32_t &segmentRows,
                                            uint32_t &cumsumRows, uint32_t &expertTokenRows) const;
    AICORE inline void RecordReadyStageStart() const
    {
#if DISPATCH_FFN_COMBINE_V8_ENABLE_INNER_PROFILE
        if (profileEntry_ != nullptr) {
            profileEntry_[kDispatchFFNCombineProfileReadyStageBase + DISPATCH_FFN_COMBINE_PROFILE_READY_STAGE_GMM2] =
                get_sys_cnt();
        }
#endif
    }
    AICORE inline void RecordReadyStageStartOnce(bool &readyStageRecorded) const
    {
#if DISPATCH_FFN_COMBINE_V8_ENABLE_INNER_PROFILE
        if (!readyStageRecorded) {
            RecordReadyStageStart();
            readyStageRecorded = true;
        }
#else
        (void)readyStageRecorded;
#endif
    }
    AICORE inline uint64_t Gmm2ProfileNow() const
    {
#if DISPATCH_FFN_COMBINE_V8_ENABLE_DETAIL_PROFILE
        return profileEntry_ == nullptr ? 0U : get_sys_cnt();
#else
        return 0U;
#endif
    }
    AICORE inline uint16_t Gmm2ToCombineFlagId(uint32_t groupIdx) const
    {
        return V5_GMM2_TO_COMBINE_HARD_FLAG_BASE + groupIdx / CROSS_CORE_FLAG_MAX_SET_COUNT;
    }
    AICORE inline size_t Gmm2ToCombineTraceIndex(uint32_t groupIdx) const
    {
#if DISPATCH_FFN_COMBINE_V8_ENABLE_DETAIL_PROFILE
        return groupIdx < kDispatchFFNCombineProfileGmm2ToCombineTraceGroupCount ?
                   DISPATCH_FFN_COMBINE_PROFILE_GMM2_TO_COMBINE_TRACE_GROUP_BASE + static_cast<size_t>(groupIdx) :
                   kDispatchFFNCombineProfileGmm2ToCombineTraceCount;
#else
        (void)groupIdx;
        return 0U;
#endif
    }
    AICORE inline void RecordGmm2ToCombineTrace(size_t event, uint16_t flagId, uint64_t start, uint64_t end) const
    {
#if DISPATCH_FFN_COMBINE_V8_ENABLE_DETAIL_PROFILE
        if (profileEntry_ == nullptr || start == 0U || end < start ||
            event >= kDispatchFFNCombineProfileGmm2ToCombineTraceCount) {
            return;
        }
        const size_t base = kDispatchFFNCombineProfileGmm2ToCombineTraceBase +
                            event * kDispatchFFNCombineProfileGmm2ToCombineTraceFieldCount;
        if (base + kDispatchFFNCombineProfileGmm2ToCombineTraceFieldCount > kDispatchFFNCombineProfileEntryU64Count) {
            return;
        }
        if (profileEntry_[base + 3U] == 0U) {
            profileEntry_[base] = start;
        }
        profileEntry_[base + 1U] = end;
        profileEntry_[base + 2U] = flagId;
        profileEntry_[base + 3U] += 1U;
#else
        (void)event;
        (void)flagId;
        (void)start;
        (void)end;
#endif
    }
    AICORE inline void RecordGmm2CvDetail(size_t detail, uint64_t start, uint64_t end) const
    {
#if DISPATCH_FFN_COMBINE_V8_ENABLE_DETAIL_PROFILE
        if (profileEntry_ == nullptr || start == 0U || end < start ||
            detail >= kDispatchFFNCombineProfileGmm2CvDetailCount) {
            return;
        }
        const size_t base =
            kDispatchFFNCombineProfileGmm2CvDetailBase + detail * kDispatchFFNCombineProfileGmm2CvDetailFieldCount;
        if (base + kDispatchFFNCombineProfileGmm2CvDetailFieldCount > kDispatchFFNCombineProfileEntryU64Count) {
            return;
        }
        if (profileEntry_[base + 3U] == 0U) {
            profileEntry_[base] = start;
        }
        profileEntry_[base + 1U] = end;
        profileEntry_[base + 2U] += end - start;
        profileEntry_[base + 3U] += 1U;
#else
        (void)detail;
        (void)start;
        (void)end;
#endif
    }

    GM_ADDR workspaceGM_ = nullptr;
    const __gm__ DispatchFFNCombineTilingData *tilingData_ = nullptr;
    volatile __gm__ uint64_t *profileEntry_ = nullptr;

    __gm__ int8_t *gmPermutedTokenPtr_ = nullptr;
    __gm__ float *perTokenScale2Ptr_ = nullptr;
    __gm__ half *gmm2OutputPtr_ = nullptr;
    __gm__ int8_t *weight2Ptr_ = nullptr;
    __gm__ uint64_t *scale2Ptr_ = nullptr;
    __gm__ int32_t *cumsumMMPtr_ = nullptr;
    __gm__ int32_t *expertTokenNumsPtr_ = nullptr;

    uint32_t problemK_ = 0;
    uint32_t problemN_ = 0;
    uint32_t inputK_ = 0;
    uint32_t outputN_ = 0;
    uint32_t maxOutputSize_ = 0;
    uint32_t expertPerRank_ = 0;
    uint32_t rank_ = 0;
    uint32_t rankSize_ = 0;
    uint32_t coreIdx_ = 0;
    uint32_t coreNum_ = 1;
    uint32_t stageNum_ = 0;
};

template <typename InputElement>
AICORE inline void Gmm2<InputElement>::Init(GM_ADDR weight2GM, GM_ADDR scale2GM, GM_ADDR expertTokenNumsGM,
                                            GM_ADDR workspaceGM, const __gm__ DispatchFFNCombineTilingData *tilingData,
                                            volatile __gm__ uint64_t *profileEntry)
{
    (void)sizeof(InputElement);
    workspaceGM_ = workspaceGM;
    tilingData_ = tilingData;
    profileEntry_ = profileEntry;

    problemK_ = tilingData_->dispatchFFNCombineInfo.K;
    problemN_ = tilingData_->dispatchFFNCombineInfo.N;
    inputK_ = problemN_ / 2U;
    outputN_ = problemK_;
    maxOutputSize_ = tilingData_->dispatchFFNCombineInfo.maxOutputSize;
    expertPerRank_ = tilingData_->dispatchFFNCombineInfo.expertPerRank;
    rank_ = tilingData_->runtimeInfo.rank;
    rankSize_ = tilingData_->runtimeInfo.rankSize;
    stageNum_ = tilingData_->frontReorderTiling.stageNum;
    coreIdx_ = get_block_idx();
    coreNum_ = get_block_num();

    gmPermutedTokenPtr_ =
        reinterpret_cast<__gm__ int8_t *>(workspaceGM_ + tilingData_->swigluTiling.gmPermutedTokenOffset);
    perTokenScale2Ptr_ =
        reinterpret_cast<__gm__ float *>(workspaceGM_ + tilingData_->swigluTiling.perTokenScale2Offset);
    gmm2OutputPtr_ = reinterpret_cast<__gm__ half *>(workspaceGM_ + tilingData_->gmm2Tiling.gmm2OutputOffset);
    weight2Ptr_ = reinterpret_cast<__gm__ int8_t *>(weight2GM);
    scale2Ptr_ = reinterpret_cast<__gm__ uint64_t *>(scale2GM);
    cumsumMMPtr_ = reinterpret_cast<__gm__ int32_t *>(workspaceGM_ + tilingData_->frontReorderTiling.cumsumMMOffset);
    expertTokenNumsPtr_ = reinterpret_cast<__gm__ int32_t *>(expertTokenNumsGM);
}

template <typename InputElement>
AICORE inline void Gmm2<InputElement>::BuildSegmentMetadata(uint32_t segmentIdx, uint32_t &segmentStartExpert,
                                                            uint32_t &segmentEndExpert, uint32_t &segmentRowBase,
                                                            uint32_t &segmentRows, uint32_t &cumsumRows,
                                                            uint32_t &expertTokenRows) const
{
    segmentStartExpert = SwigluSegmentStartExpert(segmentIdx);
    segmentEndExpert = SwigluSegmentEndExpert(segmentIdx);
    segmentRowBase = 0;
    segmentRows = 0;
    cumsumRows = 0;
    expertTokenRows = 0;

    uint32_t groupBase = 0;
    for (uint32_t groupIdx = 0; groupIdx < segmentEndExpert; ++groupIdx) {
        const uint32_t currentMRaw = CurrentMRaw(groupIdx);
        const uint32_t currentM = ClipCurrentM(currentMRaw, groupBase);
        if (groupIdx == segmentStartExpert) {
            segmentRowBase = groupBase;
        }
        if (groupIdx >= segmentStartExpert) {
            segmentRows += currentM;
            cumsumRows += currentMRaw;
            expertTokenRows += ExpertTokenNums(groupIdx);
        }
        groupBase += currentM;
    }
}
template <typename InputElement>
AICORE inline void Gmm2<InputElement>::Process()
{
    if ASCEND_IS_AIV {
        return;
    }
    const bool debugEnabled = DeviceDebug::Gmm2DebugEnabled(*this);
#if DISPATCH_FFN_COMBINE_V8_ENABLE_INNER_PROFILE
    const bool stageProfileEnabled = profileEntry_ != nullptr;
#else
    const bool stageProfileEnabled = false;
#endif
    if (debugEnabled) {
        DeviceDebug::Gmm2WriteLayoutDebugHeader(*this);
    }

    Gmm2Pipeline gmmPipeline;
    Gmm2CombineCvPipe cvPipe(nullptr, V8_GMM2_COMBINE_CV_SLOT_OFFSET, 0U);
    Gmm2CvDirectStore cvDirect(cvPipe, profileEntry_);
    uint32_t groupBase = 0;
    uint32_t startCoreIdx = 0;
    const uint32_t segmentNum = SwigluSegmentNum();
    bool readyStageRecorded = false;
    for (uint32_t segmentIdx = 0; segmentIdx < segmentNum; ++segmentIdx) {
        WaitV2CReady(segmentIdx);

        uint32_t segmentStartExpert = 0;
        uint32_t segmentEndExpert = 0;
        uint32_t segmentRowBase = 0;
        uint32_t segmentRows = 0;
        uint32_t cumsumRows = 0;
        uint32_t expertTokenRows = 0;
        BuildSegmentMetadata(segmentIdx, segmentStartExpert, segmentEndExpert, segmentRowBase, segmentRows, cumsumRows,
                             expertTokenRows);
        if (debugEnabled) {
            DeviceDebug::Gmm2WriteV2CDebug(*this, segmentIdx);
            DeviceDebug::Gmm2WriteSegmentDebug(*this, segmentIdx, segmentStartExpert, segmentEndExpert, segmentRowBase,
                                               segmentRows, cumsumRows, expertTokenRows);
        }

        for (uint32_t groupIdx = segmentStartExpert; groupIdx < segmentEndExpert; ++groupIdx) {
            const uint32_t currentMRaw = CurrentMRaw(groupIdx);
            const uint32_t currentM = ClipCurrentM(currentMRaw, groupBase);
            const uint32_t expertTokenNums = ExpertTokenNums(groupIdx);
            const uint32_t coreLoops = CoreLoops(currentM);
#if DISPATCH_FFN_COMBINE_V8_ENABLE_DETAIL_PROFILE
            uint64_t expertWallStart = 0U;
#endif
            if (debugEnabled) {
                DeviceDebug::Gmm2WriteTaskDebug(*this, segmentIdx, groupIdx, groupBase, currentMRaw, currentM,
                                                expertTokenNums, startCoreIdx, coreLoops);
            }
            const uint32_t startLoopIdx = StartLoopIdx(startCoreIdx);
            for (uint32_t loopIdx = startLoopIdx; loopIdx < coreLoops; loopIdx += coreNum_) {
                if (stageProfileEnabled) {
                    RecordReadyStageStartOnce(readyStageRecorded);
                }
#if DISPATCH_FFN_COMBINE_V8_ENABLE_DETAIL_PROFILE
                if (expertWallStart == 0U) {
                    expertWallStart = Gmm2ProfileNow();
                }
#endif
                if (Gmm2CombineCvDirectEnabled()) {
                    RunGmmTileDirect(gmmPipeline, groupIdx, groupBase, currentM, loopIdx, cvDirect);
                } else {
                    RunGmmTile(gmmPipeline, groupIdx, groupBase, currentM, loopIdx);
                }
            }
            if (CombineReadyEnabled()) {
                gmmPipeline.SynchronizeBlockDirect(cvDirect);
            }
#if DISPATCH_FFN_COMBINE_V8_ENABLE_DETAIL_PROFILE
            RecordExpertWallClock(profileEntry_, kDispatchFFNCombineProfileGmm2ExpertWallBase, groupIdx,
                                  kDispatchFFNCombineProfileGmm1DetailMaxExperts, expertWallStart, Gmm2ProfileNow());
#endif
            groupBase += currentM;
            startCoreIdx = (startCoreIdx + coreLoops) % coreNum_;
        }
    }
    if (!CombineReadyEnabled()) {
        gmmPipeline.SynchronizeBlock();
    }
    if (debugEnabled && stageNum_ == 12U) {
        DeviceDebug::Gmm2WriteDoneDebug(*this);
        pto::SYNCALL<pto::SyncCoreType::AICOnly>();
        DeviceDebug::Gmm2WriteFinalSyncDebug(*this);
    } else if (debugEnabled && CombineReadyEnabled()) {
        DeviceDebug::Gmm2WriteDoneDebug(*this);
        DeviceDebug::Gmm2WriteFinalSyncDebug(*this);
    }
}

} // namespace dispatch_ffn_combine_v8

#endif // DISPATCH_FFN_COMBINE_V8_GMM2_H
