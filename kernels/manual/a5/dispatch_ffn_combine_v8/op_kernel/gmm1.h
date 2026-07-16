/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef DISPATCH_FFN_COMBINE_V8_GMM1_H
#define DISPATCH_FFN_COMBINE_V8_GMM1_H

#include "kernel_operator.h"

#include "dispatch_ffn_combine_tiling.h"
#include "device_debug.h"
#include "gmm_common.h"
#include "kernel_launch.hpp"
#include "profile_debug_config.h"
#include "utils/common_helpers.hpp"
#include "utils/const_args.hpp"
#include "utils/pto_sync_substrate.hpp"

namespace dispatch_ffn_combine_v8 {

constexpr uint32_t kGmm1SegmentPolicyReferenceEpilogue = 1U;
constexpr uint32_t kGmm1InvalidTask = kGmmCommonInvalidTask;
using Gmm1Pipeline = GmmCommonPipeline;

template <typename InputElement>
class Gmm1 {
public:
    AICORE inline void Init(GM_ADDR weight1GM, GM_ADDR scale1GM, GM_ADDR expertTokenNumsGM, GM_ADDR workspaceGM,
                            const __gm__ DispatchFFNCombineTilingData *tilingData,
                            volatile __gm__ uint64_t *profileEntry = nullptr);
    AICORE inline void Process();

private:
    friend struct DeviceDebug;
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
    AICORE inline uint16_t C2VFlagId(uint32_t segmentIdx) const
    {
        (void)segmentIdx;
        return V5_C2V_HARD_FLAG_BASE;
    }
    AICORE inline uint16_t DispatchV2CFlagId(uint32_t logicalGroupEventIdx) const
    {
        // Logical dispatch V2C group-ready events are sequentially mapped onto reusable FFTS hard flags.
        return static_cast<uint16_t>(V5_DISPATCH_V2C_HARD_FLAG_BASE +
                                     logicalGroupEventIdx / CROSS_CORE_FLAG_MAX_SET_COUNT);
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
        return GmmCommonTileM(currentM, tilingData_->gmm1Tiling.l1TileM);
    }
    AICORE inline uint32_t TileN() const
    {
        return GmmCommonTileN(problemN_, tilingData_->gmm1Tiling.l1TileN);
    }
    AICORE inline uint32_t CoreLoops(uint32_t currentM) const
    {
        return GmmCommonCoreLoops(currentM, problemN_, tilingData_->gmm1Tiling.l1TileM,
                                  tilingData_->gmm1Tiling.l1TileN);
    }
    AICORE inline uint32_t StartLoopIdx(uint32_t startCoreIdx) const
    {
        return GmmCommonStartLoopIdx(coreIdx_, coreNum_, startCoreIdx);
    }
    AICORE inline uint32_t AssignedTileCount(uint32_t startLoopIdx, uint32_t coreLoops) const
    {
        return GmmCommonAssignedTileCount(startLoopIdx, coreLoops, coreNum_);
    }
    AICORE inline bool DispatchGatherModeEnabled() const
    {
        return tilingData_->dispatchTiling.dispatchGatherMode != 0U;
    }
    AICORE inline void GetBlockCoordMN(uint32_t loopIdx, uint32_t tileM, uint32_t tileN, uint32_t &blockM,
                                       uint32_t &blockN) const
    {
        GmmCommonGetBlockCoordMN(loopIdx, tileM, tileN, blockM, blockN);
    }
    AICORE inline void GetActualBlockShapeMN(uint32_t blockM, uint32_t blockN, uint32_t tileM, uint32_t tileN,
                                             uint32_t currentM, uint32_t &actualM, uint32_t &actualN) const
    {
        GmmCommonGetActualBlockShapeMN(blockM, blockN, tileM, tileN, currentM, problemN_,
                                       tilingData_->gmm1Tiling.l1TileM, tilingData_->gmm1Tiling.l1TileN, actualM,
                                       actualN);
    }
    AICORE inline void WaitDispatchGroupReady(uint32_t groupIdx) const
    {
        WaitDispatchV2CFlagForProbe(DispatchV2CFlagId(groupIdx));
    }
    AICORE inline void DebugWait50msBySysCnt() const
    {
        constexpr uint64_t kWaitTicks = 50000000U; // 50 ms at 1 ns per SYS_CNT tick on A5.
        const uint64_t start = static_cast<uint64_t>(get_sys_cnt());
        while (static_cast<uint64_t>(get_sys_cnt()) - start < kWaitTicks) {
            __asm__ __volatile__("");
        }
    }
    AICORE inline void WaitDispatchV2CFlagForProbe(uint16_t flagId) const
    {
        CrossCoreWaitFlag<0x2>(flagId);
#if DISPATCH_FFN_COMBINE_V8_GMM1_V2C_WAIT_PROBE_EXTRA > 0
        for (uint32_t probeWaitIdx = 0U;
             probeWaitIdx < static_cast<uint32_t>(DISPATCH_FFN_COMBINE_V8_GMM1_V2C_WAIT_PROBE_EXTRA); ++probeWaitIdx) {
            CrossCoreWaitFlag<0x2>(flagId);
        }
#endif
        pipe_barrier(PIPE_ALL);
    }
    AICORE inline size_t DispatchV2CGroupTraceIndex(uint32_t groupIdx) const
    {
#if DISPATCH_FFN_COMBINE_V8_ENABLE_DETAIL_PROFILE
        return groupIdx < kDispatchFFNCombineProfileDispatchV2CTraceGroupCount ?
                   DISPATCH_FFN_COMBINE_PROFILE_DISPATCH_V2C_TRACE_GROUP_BASE + static_cast<size_t>(groupIdx) :
                   kDispatchFFNCombineProfileDispatchV2CTraceCount;
#else
        (void)groupIdx;
        return 0U;
#endif
    }
    AICORE inline uint64_t PackedWeightExpertStride() const
    {
        return GmmCommonPackedWeightExpertStride(problemK_, problemN_);
    }
    AICORE inline void RunGmmTile(Gmm1Pipeline &gmmPipeline, uint32_t groupIdx, uint32_t groupBase, uint32_t currentM,
                                  uint32_t loopIdx) const
    {
        GmmCommonRunTile(gmmPipeline, gmAPtr_, weight1Ptr_, gmCPtr_, scale1Ptr_, groupIdx, groupBase, currentM, loopIdx,
                         problemN_, problemK_, problemK_, problemK_, problemN_, problemN_,
                         tilingData_->gmm1Tiling.l1TileM, tilingData_->gmm1Tiling.l1TileN);
    }
    AICORE inline void SetC2VReady(uint32_t segmentIdx) const;
    AICORE inline void RecordReadyStageStart() const
    {
#if DISPATCH_FFN_COMBINE_V8_ENABLE_INNER_PROFILE
        if (profileEntry_ != nullptr) {
            profileEntry_[kDispatchFFNCombineProfileReadyStageBase + DISPATCH_FFN_COMBINE_PROFILE_READY_STAGE_GMM1] =
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
    AICORE inline uint64_t ProfileNow() const
    {
#if DISPATCH_FFN_COMBINE_V8_ENABLE_DETAIL_PROFILE
        return profileEntry_ == nullptr ? 0U : get_sys_cnt();
#else
        return 0U;
#endif
    }
    AICORE inline bool DetailProfileEnabled() const
    {
#if DISPATCH_FFN_COMBINE_V8_ENABLE_DETAIL_PROFILE
        return profileEntry_ != nullptr;
#else
        return false;
#endif
    }
    AICORE inline void RecordGmm1L1TileProfile(uint32_t groupIdx, uint64_t start, uint64_t end,
                                               uint64_t l1TileCount) const
    {
#if DISPATCH_FFN_COMBINE_V8_ENABLE_DETAIL_PROFILE
        if (profileEntry_ == nullptr || start == 0U || end < start ||
            groupIdx >= kDispatchFFNCombineProfileGmm1DetailMaxExperts) {
            return;
        }
        const size_t base = kDispatchFFNCombineProfileGmm1DetailBase +
                            static_cast<size_t>(groupIdx) * kDispatchFFNCombineProfileGmm1DetailFieldCount;
        if (base + kDispatchFFNCombineProfileGmm1DetailFieldCount > kDispatchFFNCombineProfileEntryU64Count) {
            return;
        }
        if (profileEntry_[base + 3U] == 0U) {
            if (l1TileCount == 0U) {
                return;
            }
            profileEntry_[base] = start;
        }
        profileEntry_[base + 1U] = end;
        profileEntry_[base + 2U] += end - start;
        profileEntry_[base + 3U] += l1TileCount;
#else
        (void)groupIdx;
        (void)start;
        (void)end;
        (void)l1TileCount;
#endif
    }
    AICORE inline void RecordGmm1WaitDetailProfile(size_t detail, uint64_t start, uint64_t end) const
    {
#if DISPATCH_FFN_COMBINE_V8_ENABLE_DETAIL_PROFILE
        if (profileEntry_ == nullptr || start == 0U || end < start ||
            detail >= kDispatchFFNCombineProfileGmm1WaitDetailCount) {
            return;
        }
        const size_t base =
            kDispatchFFNCombineProfileGmm1WaitDetailBase + detail * kDispatchFFNCombineProfileGmm1WaitDetailFieldCount;
        if (base + kDispatchFFNCombineProfileGmm1WaitDetailFieldCount > kDispatchFFNCombineProfileEntryU64Count) {
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
    AICORE inline void RecordDispatchV2CTrace(size_t event, uint16_t flagId, uint64_t start, uint64_t end) const
    {
#if DISPATCH_FFN_COMBINE_V8_ENABLE_DETAIL_PROFILE
        if (profileEntry_ == nullptr || start == 0U || end < start ||
            event >= kDispatchFFNCombineProfileDispatchV2CTraceCount) {
            return;
        }
        const size_t base = kDispatchFFNCombineProfileDispatchV2CTraceBase +
                            event * kDispatchFFNCombineProfileDispatchV2CTraceFieldCount;
        if (base + kDispatchFFNCombineProfileDispatchV2CTraceFieldCount > kDispatchFFNCombineProfileEntryU64Count) {
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
    GM_ADDR workspaceGM_ = nullptr;
    const __gm__ DispatchFFNCombineTilingData *tilingData_ = nullptr;
    volatile __gm__ uint64_t *profileEntry_ = nullptr;

    __gm__ int8_t *gmAPtr_ = nullptr;
    __gm__ half *gmCPtr_ = nullptr;
    __gm__ int8_t *weight1Ptr_ = nullptr;
    __gm__ uint64_t *scale1Ptr_ = nullptr;
    __gm__ float *perTokenScalePtr_ = nullptr;
    __gm__ int32_t *cumsumMMPtr_ = nullptr;
    __gm__ int32_t *expertTokenNumsPtr_ = nullptr;

    uint32_t problemK_ = 0;
    uint32_t problemN_ = 0;
    uint32_t maxOutputSize_ = 0;
    uint32_t expertPerRank_ = 0;
    uint32_t rank_ = 0;
    uint32_t rankSize_ = 0;
    uint32_t coreIdx_ = 0;
    uint32_t coreNum_ = 1;
    uint32_t stageNum_ = 0;
};

template <typename InputElement>
AICORE inline void Gmm1<InputElement>::Init(GM_ADDR weight1GM, GM_ADDR scale1GM, GM_ADDR expertTokenNumsGM,
                                            GM_ADDR workspaceGM, const __gm__ DispatchFFNCombineTilingData *tilingData,
                                            volatile __gm__ uint64_t *profileEntry)
{
    workspaceGM_ = workspaceGM;
    tilingData_ = tilingData;
    profileEntry_ = profileEntry;

    problemK_ = tilingData_->dispatchFFNCombineInfo.K;
    problemN_ = tilingData_->dispatchFFNCombineInfo.N;
    maxOutputSize_ = tilingData_->dispatchFFNCombineInfo.maxOutputSize;
    expertPerRank_ = tilingData_->dispatchFFNCombineInfo.expertPerRank;
    rank_ = tilingData_->runtimeInfo.rank;
    rankSize_ = tilingData_->runtimeInfo.rankSize;
    stageNum_ = tilingData_->frontReorderTiling.stageNum;
    coreIdx_ = get_block_idx();
    coreNum_ = get_block_num();

    gmAPtr_ = reinterpret_cast<__gm__ int8_t *>(workspaceGM_ + tilingData_->dispatchTiling.gmAOffset);
    gmCPtr_ = reinterpret_cast<__gm__ half *>(workspaceGM_ + tilingData_->gmm1Tiling.gmCOffset);
    weight1Ptr_ = reinterpret_cast<__gm__ int8_t *>(weight1GM);
    scale1Ptr_ = reinterpret_cast<__gm__ uint64_t *>(scale1GM);
    perTokenScalePtr_ =
        reinterpret_cast<__gm__ float *>(workspaceGM_ + tilingData_->dispatchTiling.perTokenScaleOffset);
    cumsumMMPtr_ = reinterpret_cast<__gm__ int32_t *>(workspaceGM_ + tilingData_->frontReorderTiling.cumsumMMOffset);
    expertTokenNumsPtr_ = reinterpret_cast<__gm__ int32_t *>(expertTokenNumsGM);
}
template <typename InputElement>
AICORE inline void Gmm1<InputElement>::SetC2VReady(uint32_t segmentIdx) const
{
    pipe_barrier(PIPE_ALL);
    CrossCoreSetFlag<0x2, PIPE_FIX>(C2VFlagId(segmentIdx));
}

template <typename InputElement>
AICORE inline void Gmm1<InputElement>::Process()
{
    if ASCEND_IS_AIV {
        return;
    }
    const bool debugEnabled = DeviceDebug::Gmm1DebugEnabled(*this);
#if DISPATCH_FFN_COMBINE_V8_ENABLE_INNER_PROFILE
    const bool stageProfileEnabled = profileEntry_ != nullptr;
#else
    const bool stageProfileEnabled = false;
#endif
    const bool detailProfileEnabled = DetailProfileEnabled();
    if (debugEnabled) {
        DeviceDebug::Gmm1WriteLayoutDebugHeader(*this);
    }
    Gmm1Pipeline gmmPipeline;
    uint32_t groupBase = 0;
    uint32_t startCoreIdx = 0;
    uint32_t segmentIdx = 0;
    uint32_t segmentRowBase = 0;
    uint32_t segmentRows = 0;
    uint32_t lastProfileGroupWithTile = kGmm1InvalidTask;
    bool readyStageRecorded = false;
    for (uint32_t groupIdx = 0; groupIdx < expertPerRank_; ++groupIdx) {
        // DebugWait50msBySysCnt();
#if DISPATCH_FFN_COMBINE_V8_ENABLE_DETAIL_PROFILE
        uint64_t waitStart = detailProfileEnabled ? ProfileNow() : 0U;
#endif
        WaitDispatchGroupReady(groupIdx);
#if DISPATCH_FFN_COMBINE_V8_ENABLE_DETAIL_PROFILE
        if (detailProfileEnabled) {
            const uint64_t waitEnd = ProfileNow();
            RecordGmm1WaitDetailProfile(DISPATCH_FFN_COMBINE_PROFILE_GMM1_WAIT_GROUP_V2C, waitStart, waitEnd);
            RecordDispatchV2CTrace(DispatchV2CGroupTraceIndex(groupIdx), DispatchV2CFlagId(groupIdx), waitStart,
                                   waitEnd);
        }
        const uint64_t metadataReadStart = detailProfileEnabled ? ProfileNow() : 0U;
#endif
        const uint32_t currentMRaw = CurrentMRaw(groupIdx);
        const uint32_t expertTokenNums = ExpertTokenNums(groupIdx);
#if DISPATCH_FFN_COMBINE_V8_ENABLE_DETAIL_PROFILE
        if (detailProfileEnabled) {
            RecordGmm1WaitDetailProfile(DISPATCH_FFN_COMBINE_PROFILE_GMM1_WAIT_METADATA_READ, metadataReadStart,
                                        ProfileNow());
        }
#endif
        const uint32_t currentM = ClipCurrentM(currentMRaw, groupBase); // 当前expert要处理的token的总行数
        if (groupIdx == SwigluSegmentStartExpert(segmentIdx)) {
            segmentRowBase = groupBase;
            segmentRows = 0;
        }
        const uint32_t coreLoops = CoreLoops(currentM); // 对currentM切成coreLoops的L0 c tile
        if (debugEnabled) {
            pipe_barrier(PIPE_ALL);
            DeviceDebug::Gmm1WriteSyncDebug(*this, groupIdx, groupBase, currentM);
            DeviceDebug::Gmm1WriteTaskDebug(*this, groupIdx, groupBase, currentMRaw, currentM, expertTokenNums,
                                            startCoreIdx, coreLoops);
        }
        const uint32_t startLoopIdx = StartLoopIdx(startCoreIdx);
        uint32_t assignedTileCount = 0;
        uint64_t l1TileProfileStart = 0;
        if (detailProfileEnabled) {
            assignedTileCount = AssignedTileCount(startLoopIdx, coreLoops);
            l1TileProfileStart = assignedTileCount == 0U ? 0U : ProfileNow();
        }
        // gmm运算
        for (uint32_t loopIdx = startLoopIdx; loopIdx < coreLoops; loopIdx += coreNum_) { // 多核按照id分配L0 c tile
            if (stageProfileEnabled) {
                RecordReadyStageStartOnce(readyStageRecorded);
            }
            RunGmmTile(gmmPipeline, groupIdx, groupBase, currentM, loopIdx);
        }
        if (detailProfileEnabled && assignedTileCount != 0U) {
            const uint64_t l1TileProfileEnd = ProfileNow();
            RecordGmm1L1TileProfile(groupIdx, l1TileProfileStart, l1TileProfileEnd, assignedTileCount);
            lastProfileGroupWithTile = groupIdx;
        }
        segmentRows += currentM;
        if (groupIdx + 1U == SwigluSegmentEndExpert(segmentIdx)) {
            const uint64_t drainProfileStart =
                (!detailProfileEnabled || lastProfileGroupWithTile == kGmm1InvalidTask) ? 0U : ProfileNow();
            gmmPipeline.SynchronizeBlock();
            if (detailProfileEnabled && lastProfileGroupWithTile != kGmm1InvalidTask) {
                const uint64_t drainProfileEnd = ProfileNow();
                RecordGmm1L1TileProfile(lastProfileGroupWithTile, drainProfileStart, drainProfileEnd, 0U);
                lastProfileGroupWithTile = kGmm1InvalidTask;
            }
            if (debugEnabled) {
                DeviceDebug::Gmm1WriteDoneDebug(*this, segmentIdx, SwigluSegmentStartExpert(segmentIdx),
                                                SwigluSegmentEndExpert(segmentIdx), segmentRowBase, segmentRows);
            }
            if (stageNum_ >= 11U) {
                SetC2VReady(segmentIdx);
            }
            ++segmentIdx;
        }
        groupBase += currentM;
        startCoreIdx = (startCoreIdx + coreLoops) % coreNum_;
    }
}

} // namespace dispatch_ffn_combine_v8

#endif // DISPATCH_FFN_COMBINE_V8_GMM1_H
