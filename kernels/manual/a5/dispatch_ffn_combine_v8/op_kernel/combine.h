/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef DISPATCH_FFN_COMBINE_V8_COMBINE_H
#define DISPATCH_FFN_COMBINE_V8_COMBINE_H

#include <type_traits>

#include <pto/pto-inst.hpp>

#include "dispatch_ffn_combine_tiling.h"
#include "device_debug.h"
#include "gmm2_combine_cv_pipe.h"
#include "gmm_common.h"
#include "kernel_launch.hpp"
#include "kernel_operator.h"
#include "profile_debug_config.h"
#include "profile_expert_wall.hpp"
#include "utils/common_helpers.hpp"
#include "utils/const_args.hpp"
#include "utils/hccl_window.hpp"
#include "utils/peer_memory_layout.hpp"
#include "utils/pto_sync_substrate.hpp"
#include "utils/pto_vector.hpp"

namespace dispatch_ffn_combine_v8 {

constexpr uint32_t kCombineLayoutVersion = 1U;
constexpr uint32_t kCombineVecTileElems = 8192U;
constexpr uint32_t kCombineBufferNum = 2U;
constexpr uint32_t kCombineInvalidTask = 0xFFFFFFFFU;
constexpr uint32_t kCombineSmallTokenThreshold = 4096U;
constexpr uint32_t kCombineSmallTokenSubtileRows = 16U;
constexpr uint32_t kCombineSmallTokenSubtileCols = 256U;
constexpr uint32_t kCombineSmallMaxElems = 8192U;
constexpr uint32_t kCombineSmallScaleElems = kCombineSmallMaxElems / kCombineSmallTokenSubtileCols;
constexpr uint32_t kCombineDirectSmallUbStages = 2U;
static_assert(V8_GMM2_COMBINE_CV_SCALE_CACHE_ELEMS % V8_GMM2_COMBINE_CV_TILE_M == 0U);
constexpr uint32_t kCombineCvScaleMaxRowBlocks =
    V8_GMM2_COMBINE_CV_SCALE_CACHE_ELEMS / V8_GMM2_COMBINE_CV_TILE_M;
static_assert(kCombineSmallTokenThreshold <= V8_GMM2_COMBINE_CV_SCALE_CACHE_ELEMS);
static_assert(kCombineCvScaleMaxRowBlocks <= 64U);

template <typename OutputElement>
class Combine {
public:
    AICORE inline void Init(GM_ADDR workspaceGM, const __gm__ DispatchFFNCombineTilingData *tilingData,
                            volatile __gm__ uint64_t *profileEntry = nullptr);
    AICORE inline void Process();

private:
    friend struct DeviceDebug;
    static_assert(std::is_same_v<OutputElement, half> || std::is_same_v<OutputElement, bfloat16_t>,
                  "combine output must be half or bfloat16");

    using OutputElementType = OutputElement;
    using VectorShape = pto::Shape<1, 1, 1, 1, pto::DYNAMIC>;
    using VectorStride = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, 1>;
    using ScaleGlobal = pto::GlobalTensor<float, VectorShape, VectorStride, pto::Layout::ND>;
    using BlockShape = pto::Shape<1, 1, 1, pto::DYNAMIC, pto::DYNAMIC>;
    using BlockStride = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, 1>;
    using CBlockGlobal = pto::GlobalTensor<half, BlockShape, BlockStride, pto::Layout::ND>;
    using DBlockGlobal = pto::GlobalTensor<OutputElement, BlockShape, BlockStride, pto::Layout::ND>;
    using TileC = pto::Tile<pto::TileType::Vec, half, 1, kCombineVecTileElems, pto::BLayout::RowMajor, -1, -1>;
    using TileFp32 = pto::Tile<pto::TileType::Vec, float, 1, kCombineVecTileElems, pto::BLayout::RowMajor, -1, -1>;
    using TileD = pto::Tile<pto::TileType::Vec, OutputElement, 1, kCombineVecTileElems, pto::BLayout::RowMajor, -1, -1>;
    using SmallTileC = pto::Tile<pto::TileType::Vec, half, kCombineSmallTokenSubtileRows, kCombineSmallTokenSubtileCols,
                                 pto::BLayout::RowMajor, -1, -1>;
    using SmallTileFp32 = pto::Tile<pto::TileType::Vec, float, kCombineSmallTokenSubtileRows,
                                    kCombineSmallTokenSubtileCols, pto::BLayout::RowMajor, -1, -1>;
    using SmallTileD = pto::Tile<pto::TileType::Vec, OutputElement, kCombineSmallTokenSubtileRows,
                                 kCombineSmallTokenSubtileCols, pto::BLayout::RowMajor, -1, -1>;
    using CvTile = pto::Tile<pto::TileType::Vec, half, V8_GMM2_COMBINE_CV_TILE_M, V8_GMM2_COMBINE_CV_TILE_N,
                             pto::BLayout::RowMajor, -1, -1>;
    using CvScaleTile = pto::Tile<pto::TileType::Vec, float, 1, V8_GMM2_COMBINE_CV_TILE_M, pto::BLayout::RowMajor, 1,
                                  pto::DYNAMIC>;
    struct CvScaleBlockInfo {
        uint32_t tileRowOffset;
        uint32_t srcRowOffset;
        uint32_t rowNum;
        uint32_t cacheRowOffset;
        uint32_t rowOffsetInTile;
    };

    AICORE inline uint64_t TokenVolume() const
    {
        return static_cast<uint64_t>(problemM_) * topK_;
    }
    AICORE inline bool IsSmallTokenPath() const
    {
        return TokenVolume() <= kCombineSmallTokenThreshold;
    }
    AICORE inline uint32_t CombineImplMode() const
    {
        return tilingData_->combineTiling.combineImplMode;
    }
    AICORE inline uint32_t TileCols() const
    {
        const uint32_t cols = tilingData_->combineTiling.combineTileCols;
        return cols == 0U ? problemK_ : cols;
    }
    AICORE inline bool DirectLargeEnabled() const
    {
        return !IsSmallTokenPath();
    }
    AICORE inline bool DirectSmallEnabled() const
    {
        return IsSmallTokenPath();
    }
    AICORE inline uint32_t SubtileFirstIndex(uint32_t subtileCount, uint32_t aivSubCoreIdx) const
    {
        const uint32_t firstSubCoreCount = (subtileCount + 1U) / 2U;
        return aivSubCoreIdx == 0U ? 0U : firstSubCoreCount;
    }
    AICORE inline uint32_t SubtileAssignedCount(uint32_t subtileCount, uint32_t aivSubCoreIdx) const
    {
        const uint32_t firstSubCoreCount = (subtileCount + 1U) / 2U;
        return aivSubCoreIdx == 0U ? firstSubCoreCount : subtileCount - firstSubCoreCount;
    }
    AICORE inline uint16_t Gmm2ToCombineFlagId(uint32_t groupIdx) const
    {
        return V5_GMM2_TO_COMBINE_HARD_FLAG_BASE + groupIdx / CROSS_CORE_FLAG_MAX_SET_COUNT;
    }
    AICORE inline uint32_t CurrentM(uint32_t groupIdx) const
    {
        return rowMapper_.CurrentM(groupIdx);
    }
    AICORE inline uint32_t ClipCurrentM(uint32_t currentMRaw, uint32_t groupBase) const
    {
        return rowMapper_.ClipCurrentM(currentMRaw, groupBase);
    }
    AICORE inline uint32_t CumsumBeforeSource(uint32_t srcRank, uint32_t groupIdx) const
    {
        return rowMapper_.CumsumBeforeSource(srcRank, groupIdx);
    }
    AICORE inline uint32_t GlobalExpert(uint32_t groupIdx) const
    {
        return rowMapper_.GlobalExpert(groupIdx);
    }
    AICORE inline uint32_t RowsRaw(uint32_t srcRank, uint32_t groupIdx) const
    {
        return rowMapper_.RowsRaw(srcRank, groupIdx);
    }
    AICORE inline uint32_t DstRowOffset(uint32_t srcRank, uint32_t groupIdx) const
    {
        return rowMapper_.DstRowOffset(srcRank, groupIdx);
    }
    AICORE inline uint32_t RowsClipped(uint32_t srcRowOffset, uint32_t rowsRaw) const
    {
        return rowMapper_.RowsClipped(srcRowOffset, rowsRaw);
    }
    AICORE inline uint32_t CeilDiv(uint32_t value, uint32_t divisor) const
    {
        return divisor == 0U ? 0U : (value + divisor - 1U) / divisor;
    }
    AICORE inline event_t LoadFreeEvent(uint32_t bufferId) const
    {
        return static_cast<event_t>(bufferId);
    }
    AICORE inline event_t LoadReadyEvent(uint32_t bufferId) const
    {
        return static_cast<event_t>(bufferId + 2U);
    }
    AICORE inline event_t StoreFreeEvent(uint32_t bufferId) const
    {
        return static_cast<event_t>(bufferId);
    }
    AICORE inline event_t StoreReadyEvent(uint32_t bufferId) const
    {
        return static_cast<event_t>(bufferId + 2U);
    }
    AICORE inline event_t CvScaleReadyEvent() const
    {
        return EVENT_ID4;
    }
    AICORE inline void InitUbLayout();
    AICORE inline uint64_t CombineProfileNow() const;
    AICORE inline void RecordCombineDetail(size_t detail, uint64_t start, uint64_t end) const;
    AICORE inline size_t Gmm2ToCombineTraceIndex(uint32_t groupIdx) const;
    AICORE inline void RecordGmm2ToCombineTrace(size_t event, uint16_t flagId, uint64_t start, uint64_t end) const;
    AICORE inline void SetInitialFlags() const;
    AICORE inline void FinalizeLocalPipe();
    AICORE inline uint32_t TokenPerExpertResetElems() const;
    AICORE inline bool ResetTokenPerExpert(uint32_t elems) const;
    AICORE inline void ProcessFinalBoundary();
    AICORE inline void AccumulateDoneStats(uint32_t srcRank, uint32_t rows);
    AICORE inline void RecordReadyStageStart() const;
    AICORE inline void RecordReadyStageStartOnce() const;
    AICORE inline void LoadSmallSubtile(uint32_t bufferId, uint32_t srcRowOffset, uint32_t rowNum, uint32_t colBegin,
                                        uint32_t colNum) const;
    AICORE inline void PopCvTile(Gmm2CombineCvPipe &cvPipe, uint32_t actualM, uint32_t actualN) const;
    AICORE inline void IssueCvScaleBlockLoad(uint32_t srcRowOffset, uint32_t rowNum, uint32_t cacheRowOffset) const;
    AICORE inline bool BuildCvAssignedScaleBlocks(CvScaleBlockInfo *scaleBlocks, uint32_t &scaleBlockCount,
                                                  uint32_t groupBase, uint32_t currentM, uint32_t startLoopIdx,
                                                  uint32_t coreLoops, uint32_t aicCoreNum, uint32_t l1TileM,
                                                  uint32_t l1TileN, uint32_t aivSubCoreIdx) const;
    AICORE inline uint32_t FindCvScaleCacheRowOffset(const CvScaleBlockInfo *scaleBlocks, uint32_t scaleBlockCount,
                                                     uint32_t tileRowOffset) const;
    AICORE inline bool IssueCvScaleBlockLoads(const CvScaleBlockInfo *scaleBlocks, uint32_t scaleBlockCount) const;
    AICORE inline void WaitCvScaleBlockLoads() const;
    AICORE inline void PrepareCvSmallSubtile(uint32_t bufferId, uint32_t cvRowInTile, uint32_t scaleRowOffset,
                                             uint32_t rowNum, uint32_t colNum) const;
    AICORE inline void DequantDirectSmallSubtile(uint32_t bufferId, uint32_t rowNum, uint32_t colNum);
    AICORE inline void StoreSmallSubtileIntersection(uint32_t bufferId, __gm__ OutputElement *dstBase,
                                                     uint32_t dstRowOffset, uint32_t ubRowOffset, uint32_t rowNum,
                                                     uint32_t colBegin, uint32_t colNum);
    AICORE inline void ProcessDirectSmallTokenPath();

    GM_ADDR workspaceGM_ = nullptr;
    const __gm__ DispatchFFNCombineTilingData *tilingData_ = nullptr;
    volatile __gm__ uint64_t *profileEntry_ = nullptr;

    PtoRemoteWindow remoteWindow_;
    PeerMemoryLayout peerMemoryLayout_;
    PeerRowMapper rowMapper_;
    __gm__ half *gmm2OutputPtr_ = nullptr;
    __gm__ float *perTokenScale2Ptr_ = nullptr;
    __gm__ int32_t *cumsumMMPtr_ = nullptr;
    __gm__ int32_t *preSumBeforeRankPtr_ = nullptr;
    __gm__ int32_t *tokenPerExpertPtr_ = nullptr;

    uint32_t problemM_ = 0;
    uint32_t problemK_ = 0;
    uint32_t topK_ = 0;
    uint32_t maxOutputSize_ = 0;
    uint32_t expertPerRank_ = 0;
    uint32_t expertNumAligned_ = 0;
    uint32_t rank_ = 0;
    uint32_t rankSize_ = 0;
    uint32_t coreIdx_ = 0;
    uint32_t coreNum_ = 1;
    uint32_t stageNum_ = 0;
    uint32_t pingpongId_ = 0;
    uint32_t localSegmentCount_ = 0;
    uint32_t remoteSegmentCount_ = 0;
    uint32_t localRows_ = 0;
    uint32_t remoteRows_ = 0;
    uint32_t localBytes_ = 0;
    uint32_t remoteBytes_ = 0;
    mutable bool readyStageRecorded_ = false;
    uint64_t ubCOffset_[kCombineBufferNum] = {0, 0};
    uint64_t ubFp32Offset_[kCombineBufferNum] = {0, 0};
    uint64_t ubDOffset_[kCombineBufferNum] = {0, 0};
    uint64_t ubScaleOffset_[kCombineBufferNum] = {0, 0};
    mutable uint32_t smallScaleSourceOffset_[kCombineBufferNum] = {kCombineInvalidTask, kCombineInvalidTask};
};

template <typename OutputElement>
AICORE inline uint64_t Combine<OutputElement>::CombineProfileNow() const
{
#if DISPATCH_FFN_COMBINE_V8_ENABLE_DETAIL_PROFILE
    return profileEntry_ == nullptr ? 0U : get_sys_cnt();
#else
    return 0U;
#endif
}

template <typename OutputElement>
AICORE inline void Combine<OutputElement>::RecordCombineDetail(size_t detail, uint64_t start, uint64_t end) const
{
#if DISPATCH_FFN_COMBINE_V8_ENABLE_DETAIL_PROFILE
    if (profileEntry_ == nullptr || start == 0U || end < start ||
        detail >= kDispatchFFNCombineProfileCombineDetailCount) {
        return;
    }
    const size_t base =
        kDispatchFFNCombineProfileCombineDetailBase + detail * kDispatchFFNCombineProfileCombineDetailFieldCount;
    if (base + kDispatchFFNCombineProfileCombineDetailFieldCount > kDispatchFFNCombineProfileEntryU64Count) {
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

template <typename OutputElement>
AICORE inline size_t Combine<OutputElement>::Gmm2ToCombineTraceIndex(uint32_t groupIdx) const
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

template <typename OutputElement>
AICORE inline void Combine<OutputElement>::RecordGmm2ToCombineTrace(size_t event, uint16_t flagId, uint64_t start,
                                                                    uint64_t end) const
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

template <typename OutputElement>
AICORE inline void Combine<OutputElement>::Init(GM_ADDR workspaceGM,
                                                const __gm__ DispatchFFNCombineTilingData *tilingData,
                                                volatile __gm__ uint64_t *profileEntry)
{
    workspaceGM_ = workspaceGM;
    tilingData_ = tilingData;
    profileEntry_ = profileEntry;
    readyStageRecorded_ = false;
    pingpongId_ = 0;
    localSegmentCount_ = 0;
    remoteSegmentCount_ = 0;
    localRows_ = 0;
    remoteRows_ = 0;
    localBytes_ = 0;
    remoteBytes_ = 0;

    problemM_ = tilingData_->dispatchFFNCombineInfo.M;
    problemK_ = tilingData_->dispatchFFNCombineInfo.K;
    topK_ = tilingData_->dispatchFFNCombineInfo.topK;
    maxOutputSize_ = tilingData_->dispatchFFNCombineInfo.maxOutputSize;
    expertPerRank_ = tilingData_->dispatchFFNCombineInfo.expertPerRank;
    expertNumAligned_ = tilingData_->frontReorderTiling.expertNumAligned;
    rank_ = tilingData_->runtimeInfo.rank;
    rankSize_ = tilingData_->runtimeInfo.rankSize;
    stageNum_ = tilingData_->frontReorderTiling.stageNum;
    coreIdx_ = get_block_idx();
    coreNum_ = get_block_num();
    if ASCEND_IS_AIV {
        coreIdx_ = get_block_idx() + get_subblockid() * get_block_num();
        coreNum_ = get_block_num() * get_subblockdim();
    }

    remoteWindow_.Init(reinterpret_cast<GM_ADDR>(tilingData_->runtimeInfo.remoteWindowContext));
    peerMemoryLayout_.Init(remoteWindow_);

    gmm2OutputPtr_ = reinterpret_cast<__gm__ half *>(workspaceGM_ + tilingData_->combineTiling.gmm2OutputOffset);
    perTokenScale2Ptr_ =
        reinterpret_cast<__gm__ float *>(workspaceGM_ + tilingData_->combineTiling.perTokenScale2Offset);
    cumsumMMPtr_ = reinterpret_cast<__gm__ int32_t *>(workspaceGM_ + tilingData_->frontReorderTiling.cumsumMMOffset);
    preSumBeforeRankPtr_ =
        reinterpret_cast<__gm__ int32_t *>(workspaceGM_ + tilingData_->frontReorderTiling.preSumBeforeRankOffset);
    tokenPerExpertPtr_ =
        reinterpret_cast<__gm__ int32_t *>(remoteWindow_() + peerMemoryLayout_.offsetPeerTokenPerExpert);
    rowMapper_.Init(cumsumMMPtr_, preSumBeforeRankPtr_, tokenPerExpertPtr_, rank_, rankSize_, expertPerRank_,
                    expertNumAligned_, maxOutputSize_);

    InitUbLayout();
}

template <typename OutputElement>
AICORE inline void Combine<OutputElement>::InitUbLayout()
{
    uint64_t ubOffset = 0;
    for (uint32_t i = 0; i < kCombineBufferNum; ++i) {
        ubCOffset_[i] = ubOffset;
        ubOffset += alignUp(static_cast<uint64_t>(kCombineSmallMaxElems) * sizeof(half), UB_ALIGN);
        ubDOffset_[i] = ubOffset;
        ubOffset += alignUp(static_cast<uint64_t>(kCombineSmallMaxElems) * sizeof(OutputElement), UB_ALIGN);
        ubFp32Offset_[i] = ubOffset;
        ubOffset += alignUp(static_cast<uint64_t>(kCombineSmallMaxElems) * sizeof(float), UB_ALIGN);
        ubScaleOffset_[i] = ubOffset;
        ubOffset += alignUp(static_cast<uint64_t>(kCombineSmallScaleElems) * sizeof(float), UB_ALIGN);
        smallScaleSourceOffset_[i] = kCombineInvalidTask;
    }
}

template <typename OutputElement>
AICORE inline void Combine<OutputElement>::SetInitialFlags() const
{
    for (uint32_t i = 0; i < kCombineBufferNum; ++i) {
        set_flag(PIPE_V, PIPE_MTE2, LoadFreeEvent(i));
        set_flag(PIPE_MTE3, PIPE_V, StoreFreeEvent(i));
    }
}

template <typename OutputElement>
AICORE inline void Combine<OutputElement>::FinalizeLocalPipe()
{
    for (uint32_t i = 0; i < kCombineBufferNum; ++i) {
        wait_flag(PIPE_V, PIPE_MTE2, LoadFreeEvent(i));
        wait_flag(PIPE_MTE3, PIPE_V, StoreFreeEvent(i));
    }
}

template <typename OutputElement>
AICORE inline uint32_t Combine<OutputElement>::TokenPerExpertResetElems() const
{
    return rankSize_ * expertNumAligned_;
}

template <typename OutputElement>
AICORE inline bool Combine<OutputElement>::ResetTokenPerExpert(uint32_t elems) const
{
    if (coreIdx_ != coreNum_ - 1U) {
        return false;
    }
    PtoFillUb<int32_t>(0U, 0, elems);
    pipe_barrier(PIPE_ALL);
    PtoStoreVector<int32_t>(tokenPerExpertPtr_, 0U, elems);
    pipe_barrier(PIPE_ALL);
    dsb(DSB_DDR);
    return true;
}

template <typename OutputElement>
AICORE inline void Combine<OutputElement>::ProcessFinalBoundary()
{
    const uint64_t profileStart = CombineProfileNow();
    const uint64_t finalizeWaitStart = get_sys_cnt();
    FinalizeLocalPipe();
    const uint64_t finalizeWaitEnd = get_sys_cnt();
    uint64_t stepStart = CombineProfileNow();
    const uint64_t syncAllStart = get_sys_cnt();
    pto::SYNCALL<pto::SyncCoreType::AIVOnly>();
    const uint64_t syncAllEnd = get_sys_cnt();
    RecordCombineDetail(DISPATCH_FFN_COMBINE_PROFILE_COMBINE_DETAIL_FINAL_SYNC, stepStart, CombineProfileNow());
    const uint64_t resetStart = get_sys_cnt();
    const bool resetWriter = false;
    const uint64_t resetEnd = get_sys_cnt();
    stepStart = CombineProfileNow();
    const uint64_t crossRankSyncStart = get_sys_cnt();
    remoteWindow_.CrossRankSync();
    const uint64_t crossRankSyncEnd = get_sys_cnt();
    RecordCombineDetail(DISPATCH_FFN_COMBINE_PROFILE_COMBINE_DETAIL_FINAL_CROSS_RANK_SYNC, stepStart,
                        CombineProfileNow());
    RecordCombineDetail(DISPATCH_FFN_COMBINE_PROFILE_COMBINE_DETAIL_FINAL_BOUNDARY, profileStart, CombineProfileNow());
    if (DeviceDebug::CombineDebugEnabled(*this)) {
        DeviceDebug::CombineWriteFinalizeDebug(*this, finalizeWaitStart, finalizeWaitEnd, syncAllStart, syncAllEnd,
                                               resetStart, resetEnd, crossRankSyncStart, crossRankSyncEnd,
                                               TokenPerExpertResetElems(), resetWriter);
    }
}

template <typename OutputElement>
AICORE inline void Combine<OutputElement>::AccumulateDoneStats(uint32_t srcRank, uint32_t rows)
{
    if (rows == 0U) {
        return;
    }
    const uint32_t bytes = rows * problemK_ * sizeof(OutputElement);
    if (srcRank == rank_) {
        ++localSegmentCount_;
        localRows_ += rows;
        localBytes_ += bytes;
        return;
    }
    ++remoteSegmentCount_;
    remoteRows_ += rows;
    remoteBytes_ += bytes;
}

template <typename OutputElement>
AICORE inline void Combine<OutputElement>::RecordReadyStageStart() const
{
#if DISPATCH_FFN_COMBINE_V8_ENABLE_INNER_PROFILE
    if (profileEntry_ != nullptr) {
        profileEntry_[kDispatchFFNCombineProfileReadyStageBase + DISPATCH_FFN_COMBINE_PROFILE_READY_STAGE_COMBINE] =
            get_sys_cnt();
    }
#endif
}

template <typename OutputElement>
AICORE inline void Combine<OutputElement>::RecordReadyStageStartOnce() const
{
#if DISPATCH_FFN_COMBINE_V8_ENABLE_INNER_PROFILE
    if (!readyStageRecorded_) {
        RecordReadyStageStart();
        readyStageRecorded_ = true;
    }
#endif
}

template <typename OutputElement>
AICORE inline void Combine<OutputElement>::LoadSmallSubtile(uint32_t bufferId, uint32_t srcRowOffset, uint32_t rowNum,
                                                            uint32_t colBegin, uint32_t colNum) const
{
    wait_flag(PIPE_V, PIPE_MTE2, LoadFreeEvent(bufferId));
    SmallTileC cTile(rowNum, colNum);
    pto::TASSIGN(cTile, ubCOffset_[bufferId]);
    BlockShape cShape(rowNum, colNum);
    BlockStride cStride(static_cast<int64_t>(rowNum) * problemK_, static_cast<int64_t>(rowNum) * problemK_,
                        static_cast<int64_t>(rowNum) * problemK_, problemK_);
    CBlockGlobal cGlobal(gmm2OutputPtr_ + static_cast<uint64_t>(srcRowOffset) * problemK_ + colBegin, cShape, cStride);
    pto::TLOAD(cTile, cGlobal);

    if (smallScaleSourceOffset_[bufferId] != srcRowOffset) {
        SmallTileFp32 scaleTile(1, rowNum);
        pto::TASSIGN(scaleTile, ubScaleOffset_[bufferId]);
        VectorShape scaleShape(rowNum);
        VectorStride scaleStride(rowNum, rowNum, rowNum, rowNum);
        ScaleGlobal scaleGlobal(perTokenScale2Ptr_ + srcRowOffset, scaleShape, scaleStride);
        pto::TLOAD(scaleTile, scaleGlobal);
        smallScaleSourceOffset_[bufferId] = srcRowOffset;
    }
    set_flag(PIPE_MTE2, PIPE_V, LoadReadyEvent(bufferId));
    set_flag(PIPE_MTE2, PIPE_S, LoadReadyEvent(bufferId));
}

template <typename OutputElement>
AICORE inline void Combine<OutputElement>::PopCvTile(Gmm2CombineCvPipe &cvPipe, uint32_t actualM,
                                                     uint32_t actualN) const
{
    const uint64_t popStart = CombineProfileNow();
    CvTile cvTile(actualM, actualN);
    pto::TPOP<Gmm2CombineCvPipe, CvTile, pto::TileSplitAxis::TILE_UP_DOWN>(cvPipe, cvTile);
    pipe_barrier(PIPE_ALL);
    RecordCombineDetail(DISPATCH_FFN_COMBINE_PROFILE_COMBINE_DETAIL_CV_POP_WAIT, popStart, CombineProfileNow());
}

template <typename OutputElement>
AICORE inline void Combine<OutputElement>::IssueCvScaleBlockLoad(uint32_t srcRowOffset, uint32_t rowNum,
                                                                 uint32_t cacheRowOffset) const
{
    if (rowNum != 0U) {
        CvScaleTile scaleTile(rowNum);
        pto::TASSIGN(scaleTile, V8_GMM2_COMBINE_CV_TILE_SCALE_OFFSET +
                                    static_cast<uint64_t>(cacheRowOffset) * sizeof(float));
        VectorShape scaleShape(rowNum);
        VectorStride scaleStride(rowNum, rowNum, rowNum, rowNum);
        ScaleGlobal scaleGlobal(perTokenScale2Ptr_ + srcRowOffset, scaleShape, scaleStride);
        pto::TLOAD(scaleTile, scaleGlobal);
    }
}

template <typename OutputElement>
AICORE inline bool Combine<OutputElement>::BuildCvAssignedScaleBlocks(CvScaleBlockInfo *scaleBlocks,
                                                                       uint32_t &scaleBlockCount, uint32_t groupBase,
                                                                       uint32_t currentM, uint32_t startLoopIdx,
                                                                       uint32_t coreLoops, uint32_t aicCoreNum,
                                                                       uint32_t l1TileM, uint32_t l1TileN,
                                                                       uint32_t aivSubCoreIdx) const
{
    scaleBlockCount = 0U;
    if (aicCoreNum == 0U || startLoopIdx >= coreLoops) {
        return true;
    }
    for (uint32_t scanLoopIdx = startLoopIdx; scanLoopIdx < coreLoops; scanLoopIdx += aicCoreNum) {
        const GmmCommonTileInfo tileInfo = GmmCommonBuildTileInfo(currentM, problemK_, l1TileM, l1TileN, scanLoopIdx);
        const uint32_t subtileCount = CeilDiv(tileInfo.actualM, kCombineSmallTokenSubtileRows);
        const uint32_t firstSubtile = SubtileFirstIndex(subtileCount, aivSubCoreIdx);
        const uint32_t assignedSubtiles = SubtileAssignedCount(subtileCount, aivSubCoreIdx);
        if (assignedSubtiles == 0U) {
            continue;
        }
        bool alreadyLoaded = false;
        for (uint32_t blockIdx = 0U; blockIdx < scaleBlockCount; ++blockIdx) {
            if (scaleBlocks[blockIdx].tileRowOffset == tileInfo.blockRowStart) {
                alreadyLoaded = true;
                break;
            }
        }
        if (alreadyLoaded) {
            continue;
        }
        if (scaleBlockCount >= kCombineCvScaleMaxRowBlocks) {
            return false;
        }
        const uint32_t rowOffsetInTile = firstSubtile * kCombineSmallTokenSubtileRows;
        const uint32_t assignedRowEnd =
            (firstSubtile + assignedSubtiles) * kCombineSmallTokenSubtileRows < tileInfo.actualM ?
                (firstSubtile + assignedSubtiles) * kCombineSmallTokenSubtileRows :
                tileInfo.actualM;
        scaleBlocks[scaleBlockCount].tileRowOffset = tileInfo.blockRowStart;
        scaleBlocks[scaleBlockCount].srcRowOffset = groupBase + tileInfo.blockRowStart + rowOffsetInTile;
        scaleBlocks[scaleBlockCount].rowNum = assignedRowEnd - rowOffsetInTile;
        scaleBlocks[scaleBlockCount].cacheRowOffset = scaleBlockCount * V8_GMM2_COMBINE_CV_TILE_M;
        scaleBlocks[scaleBlockCount].rowOffsetInTile = rowOffsetInTile;
        ++scaleBlockCount;
    }
    return true;
}

template <typename OutputElement>
AICORE inline uint32_t Combine<OutputElement>::FindCvScaleCacheRowOffset(const CvScaleBlockInfo *scaleBlocks,
                                                                          uint32_t scaleBlockCount,
                                                                          uint32_t tileRowOffset) const
{
    for (uint32_t blockIdx = 0U; blockIdx < scaleBlockCount; ++blockIdx) {
        if (scaleBlocks[blockIdx].tileRowOffset == tileRowOffset) {
            return scaleBlocks[blockIdx].cacheRowOffset;
        }
    }
    return 0U;
}

template <typename OutputElement>
AICORE inline bool Combine<OutputElement>::IssueCvScaleBlockLoads(const CvScaleBlockInfo *scaleBlocks,
                                                                  uint32_t scaleBlockCount) const
{
    if (scaleBlockCount == 0U) {
        return false;
    }
    for (uint32_t blockIdx = 0; blockIdx < scaleBlockCount; ++blockIdx) {
        IssueCvScaleBlockLoad(scaleBlocks[blockIdx].srcRowOffset, scaleBlocks[blockIdx].rowNum,
                              scaleBlocks[blockIdx].cacheRowOffset + scaleBlocks[blockIdx].rowOffsetInTile);
    }
    set_flag(PIPE_MTE2, PIPE_S, CvScaleReadyEvent());
    return true;
}

template <typename OutputElement>
AICORE inline void Combine<OutputElement>::WaitCvScaleBlockLoads() const
{
    const uint64_t scaleStart = CombineProfileNow();
    wait_flag(PIPE_MTE2, PIPE_S, CvScaleReadyEvent());
    pipe_barrier(PIPE_ALL);
    RecordCombineDetail(DISPATCH_FFN_COMBINE_PROFILE_COMBINE_DETAIL_CV_SCALE_LOAD, scaleStart, CombineProfileNow());
}

template <typename OutputElement>
AICORE inline void Combine<OutputElement>::PrepareCvSmallSubtile(uint32_t bufferId, uint32_t cvRowInTile,
                                                                 uint32_t scaleRowOffset, uint32_t rowNum,
                                                                 uint32_t colNum) const
{
    const uint64_t profileStart = CombineProfileNow();
    wait_flag(PIPE_V, PIPE_MTE2, LoadFreeEvent(bufferId));

    SmallTileFp32 fp32Tile(rowNum, colNum);
    SmallTileC cTile(rowNum, colNum);
    pto::TASSIGN(fp32Tile, ubFp32Offset_[bufferId]);
    pto::TASSIGN(cTile, V8_GMM2_COMBINE_CV_SLOT_OFFSET +
                            static_cast<uint64_t>(cvRowInTile) * kCombineSmallTokenSubtileCols * sizeof(half));
    pto::TCVT(fp32Tile, cTile, pto::RoundMode::CAST_NONE);
    pipe_barrier(PIPE_ALL);
    set_flag(PIPE_V, PIPE_MTE2, LoadFreeEvent(bufferId));

    const uint64_t scaleMulStart = CombineProfileNow();
    CvScaleTile scaleTile(rowNum);
    pto::TASSIGN(scaleTile, V8_GMM2_COMBINE_CV_TILE_SCALE_OFFSET +
                                static_cast<uint64_t>(scaleRowOffset) * sizeof(float));
    for (uint32_t row = 0; row < rowNum; ++row) {
        const float scale = scaleTile.GetValue(row);
        SmallTileFp32 rowTile(1, colNum);
        pto::TASSIGN(rowTile, ubFp32Offset_[bufferId] +
                                  static_cast<uint64_t>(row) * kCombineSmallTokenSubtileCols * sizeof(float));
        pto::TMULS(rowTile, rowTile, scale);
    }
    pipe_barrier(PIPE_ALL);
    RecordCombineDetail(DISPATCH_FFN_COMBINE_PROFILE_COMBINE_DETAIL_SMALL_DEQUANT_SCALE_MUL, scaleMulStart,
                        CombineProfileNow());

    wait_flag(PIPE_MTE3, PIPE_V, StoreFreeEvent(bufferId));
    SmallTileD dTile(rowNum, colNum);
    pto::TASSIGN(dTile, ubDOffset_[bufferId]);
    pto::TCVT(dTile, fp32Tile, pto::RoundMode::CAST_RINT);
    set_flag(PIPE_V, PIPE_MTE3, StoreReadyEvent(bufferId));
    RecordCombineDetail(DISPATCH_FFN_COMBINE_PROFILE_COMBINE_DETAIL_SMALL_DEQUANT_TOTAL, profileStart,
                        CombineProfileNow());
}

template <typename OutputElement>
AICORE inline void Combine<OutputElement>::DequantDirectSmallSubtile(uint32_t bufferId, uint32_t rowNum,
                                                                     uint32_t colNum)
{
    const uint64_t profileStart = CombineProfileNow();
    wait_flag(PIPE_MTE2, PIPE_V, LoadReadyEvent(bufferId));
    SmallTileFp32 fp32Tile(rowNum, colNum);
    SmallTileC cTile(rowNum, colNum);
    pto::TASSIGN(fp32Tile, ubFp32Offset_[bufferId]);
    pto::TASSIGN(cTile, ubCOffset_[bufferId]);
    pto::TCVT(fp32Tile, cTile, pto::RoundMode::CAST_NONE);
    pipe_barrier(PIPE_ALL);
    set_flag(PIPE_V, PIPE_MTE2, LoadFreeEvent(bufferId));

    wait_flag(PIPE_MTE2, PIPE_S, LoadReadyEvent(bufferId));
    uint64_t stepStart = CombineProfileNow();
    SmallTileFp32 scaleTile(1, rowNum);
    pto::TASSIGN(scaleTile, ubScaleOffset_[bufferId]);
    for (uint32_t row = 0; row < rowNum; ++row) {
        const float scale = scaleTile.GetValue(row);
        SmallTileFp32 rowTile(1, colNum);
        pto::TASSIGN(rowTile, ubFp32Offset_[bufferId] +
                                  static_cast<uint64_t>(row) * kCombineSmallTokenSubtileCols * sizeof(float));
        pto::TMULS(rowTile, rowTile, scale);
    }
    pipe_barrier(PIPE_ALL);
    RecordCombineDetail(DISPATCH_FFN_COMBINE_PROFILE_COMBINE_DETAIL_SMALL_DEQUANT_SCALE_MUL, stepStart,
                        CombineProfileNow());

    wait_flag(PIPE_MTE3, PIPE_V, StoreFreeEvent(bufferId));
    SmallTileD dTile(rowNum, colNum);
    pto::TASSIGN(dTile, ubDOffset_[bufferId]);
    pto::TCVT(dTile, fp32Tile, pto::RoundMode::CAST_RINT);
    set_flag(PIPE_V, PIPE_MTE3, StoreReadyEvent(bufferId));
    RecordCombineDetail(DISPATCH_FFN_COMBINE_PROFILE_COMBINE_DETAIL_SMALL_DEQUANT_TOTAL, profileStart,
                        CombineProfileNow());
}

template <typename OutputElement>
AICORE inline void Combine<OutputElement>::StoreSmallSubtileIntersection(uint32_t bufferId,
                                                                         __gm__ OutputElement *dstBase,
                                                                         uint32_t dstRowOffset, uint32_t ubRowOffset,
                                                                         uint32_t rowNum, uint32_t colBegin,
                                                                         uint32_t colNum)
{
    SmallTileD dTile(rowNum, colNum);
    pto::TASSIGN(dTile, ubDOffset_[bufferId] +
                            static_cast<uint64_t>(ubRowOffset) * kCombineSmallTokenSubtileCols * sizeof(OutputElement));
    BlockShape dShape(rowNum, colNum);
    BlockStride dStride(static_cast<int64_t>(rowNum) * problemK_, static_cast<int64_t>(rowNum) * problemK_,
                        static_cast<int64_t>(rowNum) * problemK_, problemK_);
    DBlockGlobal dGlobal(dstBase + static_cast<uint64_t>(dstRowOffset) * problemK_ + colBegin, dShape, dStride);
    pto::TSTORE(dGlobal, dTile);
}

template <typename OutputElement>
AICORE inline void Combine<OutputElement>::ProcessDirectSmallTokenPath()
{
    const uint64_t profileStart = CombineProfileNow();
    Gmm2CombineCvPipe cvPipe(nullptr, V8_GMM2_COMBINE_CV_SLOT_OFFSET, 0U);
    uint32_t groupBase = 0;
    uint32_t startCoreIdx = 0;
    const uint32_t aicCoreIdx = get_block_idx();
    const uint32_t aicCoreNum = get_block_num();
    const uint32_t aivSubCoreIdx = get_subblockid();
    const bool consumeCv = aivSubCoreIdx < V8_GMM2_COMBINE_CV_MAX_AIV_PER_AIC;
    const uint32_t l1TileM = tilingData_->gmm2Tiling.l1TileM;
    const uint32_t l1TileN = tilingData_->gmm2Tiling.l1TileN;
    for (uint32_t groupIdx = 0; groupIdx < expertPerRank_; ++groupIdx) {
        const uint64_t groupProfileStart = CombineProfileNow();
        const uint32_t currentM = ClipCurrentM(CurrentM(groupIdx), groupBase);
        if (DeviceDebug::CombineDebugEnabled(*this)) {
            DeviceDebug::CombineWriteTaskDebug(*this, groupIdx, groupBase, currentM);
            if (coreIdx_ == 0U) {
                for (uint32_t srcRank = 0; srcRank < rankSize_; ++srcRank) {
                    DeviceDebug::CombineWriteMetaDebug(*this, groupIdx, srcRank, groupBase, currentM);
                }
            }
        }
        const uint32_t coreLoops = GmmCommonCoreLoops(currentM, problemK_, l1TileM, l1TileN); // 按照L1的size做切tile
        const uint32_t startLoopIdx =
            aicCoreNum == 0U ? 0U : GmmCommonStartLoopIdx(aicCoreIdx, aicCoreNum, startCoreIdx);
        CvScaleBlockInfo scaleBlocks[kCombineCvScaleMaxRowBlocks];
        uint32_t scaleBlockCount = 0U;
        bool cvScaleCacheEnabled = false;
        bool cvScaleReady = true;
        if (consumeCv) {
            cvScaleCacheEnabled = BuildCvAssignedScaleBlocks(scaleBlocks, scaleBlockCount, groupBase, currentM,
                                                             startLoopIdx, coreLoops, aicCoreNum, l1TileM, l1TileN,
                                                             aivSubCoreIdx);
            if (cvScaleCacheEnabled) {
                cvScaleReady = !IssueCvScaleBlockLoads(scaleBlocks, scaleBlockCount);
            }
        }
        for (uint32_t loopIdx = startLoopIdx; consumeCv && aicCoreNum != 0U && loopIdx < coreLoops;
             loopIdx += aicCoreNum) {
            const uint64_t tileProfileStart = CombineProfileNow();
            const GmmCommonTileInfo tileInfo = GmmCommonBuildTileInfo(currentM, problemK_, l1TileM, l1TileN, loopIdx);
            const uint32_t subtileCount = CeilDiv(tileInfo.actualM, kCombineSmallTokenSubtileRows);
            const uint32_t firstSubtile = SubtileFirstIndex(subtileCount, aivSubCoreIdx);
            const uint32_t assignedSubtiles = SubtileAssignedCount(subtileCount, aivSubCoreIdx);
            if (!cvScaleCacheEnabled && assignedSubtiles != 0U) {
                const uint32_t rowOffsetInTile = firstSubtile * kCombineSmallTokenSubtileRows;
                const uint32_t assignedRowEnd =
                    (firstSubtile + assignedSubtiles) * kCombineSmallTokenSubtileRows < tileInfo.actualM ?
                        (firstSubtile + assignedSubtiles) * kCombineSmallTokenSubtileRows :
                        tileInfo.actualM;
                const CvScaleBlockInfo tileScaleBlock = {
                    tileInfo.blockRowStart, groupBase + tileInfo.blockRowStart + rowOffsetInTile,
                    assignedRowEnd - rowOffsetInTile, 0U, rowOffsetInTile};
                cvScaleReady = !IssueCvScaleBlockLoads(&tileScaleBlock, 1U);
            }
            PopCvTile(cvPipe, tileInfo.actualM, tileInfo.actualN);
            if (!cvScaleReady) {
                WaitCvScaleBlockLoads();
                cvScaleReady = true;
            }
            const uint32_t scaleRowBase =
                cvScaleCacheEnabled ? FindCvScaleCacheRowOffset(scaleBlocks, scaleBlockCount, tileInfo.blockRowStart) :
                                      0U;
            for (uint32_t subtile = 0; subtile < assignedSubtiles; ++subtile) {
                const uint32_t subtileIdx = firstSubtile + subtile;
                const uint32_t rowInTile = subtileIdx * kCombineSmallTokenSubtileRows;
                if (rowInTile >= tileInfo.actualM) {
                    continue;
                }
                const uint32_t rows = (tileInfo.actualM - rowInTile > kCombineSmallTokenSubtileRows) ?
                                          kCombineSmallTokenSubtileRows :
                                          (tileInfo.actualM - rowInTile);
                const uint32_t tileRowBegin = tileInfo.blockRowStart + rowInTile;
                const uint32_t srcRow = groupBase + tileRowBegin;
                const uint32_t bufferId = pingpongId_;
                pingpongId_ = (pingpongId_ + 1U) % kCombineBufferNum;
                if (rows != 0U && tileInfo.actualN != 0U) {
                    RecordReadyStageStartOnce();
                }
                PrepareCvSmallSubtile(bufferId, rowInTile, scaleRowBase + rowInTile, rows, tileInfo.actualN);

                const uint32_t stTile = tileRowBegin;
                const uint32_t edTile = tileRowBegin + rows;
                uint32_t preSumRankInExpert = 0U;
                uint32_t tileOffset = 0U;
                wait_flag(PIPE_V, PIPE_MTE3, StoreReadyEvent(bufferId));
                const uint64_t rankIntersectProfileStart = CombineProfileNow();
                for (uint32_t srcRank = 0; srcRank < rankSize_; ++srcRank) {
                    const uint32_t lenRankInExpert = RowsRaw(srcRank, groupIdx);
                    const uint32_t dstExpertOffset = DstRowOffset(srcRank, groupIdx);
                    const uint32_t stRankInExpert = preSumRankInExpert;
                    const uint32_t edRankInExpert = stRankInExpert + lenRankInExpert;
                    preSumRankInExpert += lenRankInExpert;
                    if (stRankInExpert >= edTile) {
                        break;
                    }
                    if (edRankInExpert <= stTile) {
                        continue;
                    }
                    const uint32_t stData = stRankInExpert > stTile ? stRankInExpert : stTile;
                    const uint32_t edData = edRankInExpert < edTile ? edRankInExpert : edTile;
                    if (edData <= stData) {
                        continue;
                    }
                    const uint32_t lenData = edData - stData;
                    const uint32_t dstOffsetInExpert = stTile > stRankInExpert ? stTile - stRankInExpert : 0U;
                    const uint32_t dstRow = dstExpertOffset + dstOffsetInExpert;
                    __gm__ OutputElement *dstBase = reinterpret_cast<__gm__ OutputElement *>(
                        remoteWindow_(peerMemoryLayout_.offsetD, static_cast<int32_t>(srcRank)));
                    if (dstBase != nullptr) {
                        StoreSmallSubtileIntersection(bufferId, dstBase, dstRow, tileOffset, lenData,
                                                      tileInfo.blockColStart, tileInfo.actualN);
                        AccumulateDoneStats(srcRank, lenData);
                    }
                    tileOffset += lenData;
                }
                RecordCombineDetail(DISPATCH_FFN_COMBINE_PROFILE_COMBINE_DETAIL_SMALL_RANK_INTERSECT,
                                    rankIntersectProfileStart, CombineProfileNow());
                set_flag(PIPE_MTE3, PIPE_V, StoreFreeEvent(bufferId));
            }
            const uint64_t freeStart = CombineProfileNow();
            wait_flag(PIPE_MTE3, PIPE_V, StoreFreeEvent((pingpongId_ + kCombineBufferNum - 1U) % kCombineBufferNum));
            set_flag(PIPE_MTE3, PIPE_V, StoreFreeEvent((pingpongId_ + kCombineBufferNum - 1U) % kCombineBufferNum));
            pipe_barrier(PIPE_ALL);
            pto::TFREE<Gmm2CombineCvPipe, pto::TileSplitAxis::TILE_UP_DOWN>(cvPipe);
            RecordCombineDetail(DISPATCH_FFN_COMBINE_PROFILE_COMBINE_DETAIL_CV_FREE, freeStart, CombineProfileNow());
            RecordCombineDetail(DISPATCH_FFN_COMBINE_PROFILE_COMBINE_DETAIL_SMALL_TILE, tileProfileStart,
                                CombineProfileNow());
        }
        startCoreIdx = aicCoreNum == 0U ? 0U : (startCoreIdx + coreLoops) % aicCoreNum;
        groupBase += currentM;
        RecordCombineDetail(DISPATCH_FFN_COMBINE_PROFILE_COMBINE_DETAIL_SMALL_GROUP, groupProfileStart,
                            CombineProfileNow());
    }
    RecordCombineDetail(DISPATCH_FFN_COMBINE_PROFILE_COMBINE_DETAIL_DIRECT_SMALL_TOTAL, profileStart,
                        CombineProfileNow());
}

template <typename OutputElement>
AICORE inline void Combine<OutputElement>::Process()
{
    if ASCEND_IS_AIC {
        return;
    }
    SetInitialFlags();
    if (DeviceDebug::CombineDebugEnabled(*this)) {
        DeviceDebug::CombineWriteLayoutDebugHeader(*this);
    }
    ProcessDirectSmallTokenPath();
    ProcessFinalBoundary();
    if (DeviceDebug::CombineDebugEnabled(*this)) {
        DeviceDebug::CombineWriteDoneDebug(*this);
    }
}

} // namespace dispatch_ffn_combine_v8

#endif // DISPATCH_FFN_COMBINE_V8_COMBINE_H
