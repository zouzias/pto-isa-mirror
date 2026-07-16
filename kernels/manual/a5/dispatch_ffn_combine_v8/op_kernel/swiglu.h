/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef DISPATCH_FFN_COMBINE_V8_SWIGLU_H
#define DISPATCH_FFN_COMBINE_V8_SWIGLU_H

#include "kernel_operator.h"

#ifndef PIPE_FIX
#define PIPE_FIX static_cast<pipe_t>(10)
#endif

#include <pto/pto-inst.hpp>

#include "dispatch_ffn_combine_tiling.h"
#include "device_debug.h"
#include "kernel_launch.hpp"
#include "profile_debug_config.h"
#include "utils/common_helpers.hpp"
#include "utils/const_args.hpp"
#include "utils/pto_vector.hpp"
#include "utils/pto_sync_substrate.hpp"

namespace dispatch_ffn_combine_v8 {

constexpr uint32_t kSwigluLayoutVersion = 1U;
constexpr uint32_t kSwigluWaitSourceC2VOnly = 1U;
constexpr uint32_t kSwigluPipelineModeInputOutputSplit = 1U;
constexpr uint32_t kSwigluMetadataModeSharedSegmentMeta = 1U;
constexpr uint32_t kSwigluVecTileElems = 1024U;
constexpr uint32_t kSwigluFullRowIoBlockChunks = 4U;
constexpr uint32_t kSwigluUbStageNum = 2U;
constexpr uint32_t kSwigluScaleTileElems = 128U;
constexpr uint32_t kSwigluScaleChunkBuffers = 2U;
constexpr uint32_t kSwigluScalarScratchBytes = 32U;
constexpr float kSwigluDynamicQuantEps = 1.0e-6f;

template <typename InputElement>
class Swiglu {
public:
    AICORE inline void Init(GM_ADDR expertTokenNumsGM, GM_ADDR workspaceGM,
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
    AICORE inline uint16_t V2CFlagId(uint32_t segmentIdx) const
    {
        (void)segmentIdx;
        return V5_V2C_HARD_FLAG_BASE;
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
    AICORE inline void WaitC2VReady(uint32_t segmentIdx) const
    {
        CrossCoreWaitFlag<0x2>(C2VFlagId(segmentIdx));
        pipe_barrier(PIPE_ALL);
    }
    AICORE inline void BuildSegmentMetadata(uint32_t segmentIdx, uint32_t &segmentStartExpert,
                                            uint32_t &segmentEndExpert, uint32_t &segmentRowBase, uint32_t &segmentRows,
                                            uint32_t &cumsumRows, uint32_t &expertTokenRows) const;
    AICORE inline __gm__ DispatchFFNCombineSwigluSegmentRuntimeMeta *SegmentMetaPtr() const
    {
        return reinterpret_cast<__gm__ DispatchFFNCombineSwigluSegmentRuntimeMeta *>(
            workspaceGM_ + tilingData_->swigluTiling.swigluSegmentMetaOffset);
    }
    AICORE inline void WriteSharedSegmentMetadata(uint32_t segmentIdx) const;
    AICORE inline void ReadSharedSegmentMetadata(uint32_t segmentIdx, uint32_t &segmentStartExpert,
                                                 uint32_t &segmentEndExpert, uint32_t &segmentRowBase,
                                                 uint32_t &segmentRows, uint32_t &cumsumRows, uint32_t &expertTokenRows,
                                                 uint32_t &rowSplitBase, uint32_t &rowSplitRem) const;
    AICORE inline void SetV2CReady(uint32_t segmentIdx) const
    {
        pipe_barrier(PIPE_ALL);
        CrossCoreSetFlag<0x2, PIPE_MTE3>(V2CFlagId(segmentIdx));
    }
    AICORE inline void RecordReadyStageStart() const
    {
#if DISPATCH_FFN_COMBINE_V8_ENABLE_INNER_PROFILE
        if (profileEntry_ != nullptr) {
            profileEntry_[kDispatchFFNCombineProfileReadyStageBase + DISPATCH_FFN_COMBINE_PROFILE_READY_STAGE_SWIGLU] =
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
    AICORE inline uint64_t AlignUbBytes(uint64_t value) const
    {
        return (value + 31U) / 32U * 32U;
    }
    AICORE inline uint64_t SwigluMaxScratchBytes() const
    {
        const uint64_t bytes = static_cast<uint64_t>(outputN_ / 2U) * sizeof(float);
        return bytes < kSwigluScalarScratchBytes ? kSwigluScalarScratchBytes : AlignUbBytes(bytes);
    }
    AICORE inline uint64_t SwigluStageBytes() const;
    AICORE inline uint64_t SwigluCOffset(uint32_t bufferId) const
    {
        if (bufferId == 0U) {
            return 0;
        }
        return static_cast<uint64_t>(bufferId) * SwigluStageBytes();
    }
    AICORE inline uint64_t SwigluDOffset(uint32_t bufferId) const
    {
        return AlignUbBytes(SwigluCOffset(bufferId) + static_cast<uint64_t>(problemN_) * sizeof(half));
    }
    AICORE inline uint64_t SwigluCFp32Offset(uint32_t bufferId) const
    {
        return AlignUbBytes(SwigluDOffset(bufferId) + static_cast<uint64_t>(outputN_) * sizeof(int8_t));
    }
    AICORE inline uint64_t SwigluWorkOffset(uint32_t bufferId) const
    {
        return AlignUbBytes(SwigluCFp32Offset(bufferId) + static_cast<uint64_t>(problemN_) * sizeof(float));
    }
    AICORE inline uint64_t SwigluAbsOffset(uint32_t bufferId) const
    {
        return AlignUbBytes(SwigluWorkOffset(bufferId) + static_cast<uint64_t>(outputN_) * sizeof(float));
    }
    AICORE inline uint64_t SwigluMaxOffset(uint32_t bufferId) const
    {
        return AlignUbBytes(SwigluAbsOffset(bufferId) + static_cast<uint64_t>(outputN_) * sizeof(float));
    }
    AICORE inline uint64_t SwigluScaleOffset(uint32_t bufferId) const
    {
        return AlignUbBytes(SwigluMaxOffset(bufferId) + SwigluMaxScratchBytes());
    }
    AICORE inline uint64_t SwigluScaleOutputBytes() const
    {
        return AlignUbBytes(static_cast<uint64_t>(kSwigluScaleTileElems) * sizeof(float));
    }
    AICORE inline uint64_t SwigluScaleOutputOffset(uint32_t bufferId) const
    {
        return AlignUbBytes(SwigluStageBytes() * kSwigluUbStageNum) +
               static_cast<uint64_t>(bufferId) * SwigluScaleOutputBytes();
    }
    AICORE inline bool FullRowUbFits() const
    {
        return SwigluScaleOutputOffset(kSwigluScaleChunkBuffers - 1U) + SwigluScaleOutputBytes() <= AtlasA5::UB_SIZE;
    }
    AICORE inline event_t LoadFreeEvent(uint32_t bufferId) const
    {
        return static_cast<event_t>(bufferId);
    }
    AICORE inline event_t LoadReadyEvent(uint32_t bufferId) const
    {
        return static_cast<event_t>(bufferId);
    }
    AICORE inline event_t StoreReadyEvent(uint32_t bufferId) const
    {
        return static_cast<event_t>(bufferId);
    }
    AICORE inline event_t StoreDoneEvent(uint32_t bufferId) const
    {
        return static_cast<event_t>(bufferId);
    }
    AICORE inline event_t ScaleStoreEvent(uint32_t bufferId) const
    {
        return bufferId == 0U ? EVENT_ID2 : EVENT_ID3;
    }
    AICORE inline void InitFullRowPipeline() const;
    AICORE inline void FinalizeFullRowPipeline() const;
    AICORE inline void RunFullRowEpilogue(uint32_t localRowStart, uint32_t localRows) const;
    AICORE inline void IssueFullRowLoad(uint32_t rowIdx, uint32_t bufferId) const;
    AICORE inline float PrepareFullRowCompute(uint32_t rowIdx, uint32_t bufferId) const;
    AICORE inline float ComputeAndStorePreparedFullRow(uint32_t rowIdx, uint32_t bufferId, float perTokenScale) const;
    AICORE inline void IssueStoreScale2Chunk(uint32_t rowStart, uint32_t rowCount, uint32_t scaleBufferId) const;

    GM_ADDR workspaceGM_ = nullptr;
    const __gm__ DispatchFFNCombineTilingData *tilingData_ = nullptr;
    volatile __gm__ uint64_t *profileEntry_ = nullptr;
    __gm__ half *gmCPtr_ = nullptr;
    __gm__ float *perTokenScalePtr_ = nullptr;
    __gm__ int8_t *gmPermutedTokenPtr_ = nullptr;
    __gm__ float *perTokenScale2Ptr_ = nullptr;
    __gm__ int32_t *cumsumMMPtr_ = nullptr;
    __gm__ int32_t *expertTokenNumsPtr_ = nullptr;
    uint32_t problemN_ = 0;
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
AICORE inline void Swiglu<InputElement>::Init(GM_ADDR expertTokenNumsGM, GM_ADDR workspaceGM,
                                              const __gm__ DispatchFFNCombineTilingData *tilingData,
                                              volatile __gm__ uint64_t *profileEntry)
{
    (void)sizeof(InputElement);
    workspaceGM_ = workspaceGM;
    tilingData_ = tilingData;
    profileEntry_ = profileEntry;
    problemN_ = tilingData_->dispatchFFNCombineInfo.N;
    outputN_ = problemN_ / 2U;
    maxOutputSize_ = tilingData_->dispatchFFNCombineInfo.maxOutputSize;
    expertPerRank_ = tilingData_->dispatchFFNCombineInfo.expertPerRank;
    rank_ = tilingData_->runtimeInfo.rank;
    rankSize_ = tilingData_->runtimeInfo.rankSize;
    stageNum_ = tilingData_->frontReorderTiling.stageNum;

    coreIdx_ = get_block_idx();
    coreNum_ = get_block_num();
    if ASCEND_IS_AIV {
        coreIdx_ = get_block_idx() + get_subblockid() * get_block_num();
        coreNum_ = get_block_num() * get_subblockdim();
    }

    gmCPtr_ = reinterpret_cast<__gm__ half *>(workspaceGM_ + tilingData_->gmm1Tiling.gmCOffset);
    perTokenScalePtr_ =
        reinterpret_cast<__gm__ float *>(workspaceGM_ + tilingData_->dispatchTiling.perTokenScaleOffset);
    gmPermutedTokenPtr_ =
        reinterpret_cast<__gm__ int8_t *>(workspaceGM_ + tilingData_->swigluTiling.gmPermutedTokenOffset);
    perTokenScale2Ptr_ =
        reinterpret_cast<__gm__ float *>(workspaceGM_ + tilingData_->swigluTiling.perTokenScale2Offset);
    cumsumMMPtr_ = reinterpret_cast<__gm__ int32_t *>(workspaceGM_ + tilingData_->frontReorderTiling.cumsumMMOffset);
    expertTokenNumsPtr_ = reinterpret_cast<__gm__ int32_t *>(expertTokenNumsGM);
}
template <typename InputElement>
AICORE inline void Swiglu<InputElement>::BuildSegmentMetadata(uint32_t segmentIdx, uint32_t &segmentStartExpert,
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
AICORE inline void Swiglu<InputElement>::WriteSharedSegmentMetadata(uint32_t segmentIdx) const
{
    if (coreIdx_ != 0U) {
        return;
    }

    uint32_t segmentStartExpert = 0;
    uint32_t segmentEndExpert = 0;
    uint32_t segmentRowBase = 0;
    uint32_t segmentRows = 0;
    uint32_t cumsumRows = 0;
    uint32_t expertTokenRows = 0;
    BuildSegmentMetadata(segmentIdx, segmentStartExpert, segmentEndExpert, segmentRowBase, segmentRows, cumsumRows,
                         expertTokenRows);
    const uint32_t rowSplitBase = segmentRows / coreNum_;
    const uint32_t rowSplitRem = segmentRows - rowSplitBase * coreNum_;

    volatile __gm__ DispatchFFNCombineSwigluSegmentRuntimeMeta *entry = SegmentMetaPtr() + segmentIdx;
    entry->valid = 0U;
    entry->segmentIdx = segmentIdx;
    entry->segmentStartExpert = segmentStartExpert;
    entry->segmentEndExpert = segmentEndExpert;
    entry->segmentRowBase = segmentRowBase;
    entry->segmentRows = segmentRows;
    entry->cumsumRows = cumsumRows;
    entry->expertTokenRows = expertTokenRows;
    entry->rowSplitBase = rowSplitBase;
    entry->rowSplitRem = rowSplitRem;
    entry->generation = stageNum_;
    entry->producerCoreIdx = coreIdx_;
    entry->metadataMode = kSwigluMetadataModeSharedSegmentMeta;
    entry->segmentNum = SwigluSegmentNum();
    entry->epilogueGranularity = SwigluEpilogueGranularity();
    entry->marker = 1U;
    pipe_barrier(PIPE_ALL);
    entry->valid = 1U;
    pipe_barrier(PIPE_ALL);
    V5DcciGmRange(reinterpret_cast<__gm__ void *>(SegmentMetaPtr() + segmentIdx),
                  sizeof(DispatchFFNCombineSwigluSegmentRuntimeMeta));
}

template <typename InputElement>
AICORE inline void Swiglu<InputElement>::ReadSharedSegmentMetadata(uint32_t segmentIdx, uint32_t &segmentStartExpert,
                                                                   uint32_t &segmentEndExpert, uint32_t &segmentRowBase,
                                                                   uint32_t &segmentRows, uint32_t &cumsumRows,
                                                                   uint32_t &expertTokenRows, uint32_t &rowSplitBase,
                                                                   uint32_t &rowSplitRem) const
{
    volatile __gm__ DispatchFFNCombineSwigluSegmentRuntimeMeta *entry = SegmentMetaPtr() + segmentIdx;
    segmentStartExpert = entry->segmentStartExpert;
    segmentEndExpert = entry->segmentEndExpert;
    segmentRowBase = entry->segmentRowBase;
    segmentRows = entry->segmentRows;
    cumsumRows = entry->cumsumRows;
    expertTokenRows = entry->expertTokenRows;
    rowSplitBase = entry->rowSplitBase;
    rowSplitRem = entry->rowSplitRem;
}
template <typename InputElement>
AICORE inline uint64_t Swiglu<InputElement>::SwigluStageBytes() const
{
    return AlignUbBytes(SwigluScaleOffset(0) + kSwigluScalarScratchBytes);
}

template <typename InputElement>
AICORE inline void Swiglu<InputElement>::InitFullRowPipeline() const
{
    for (uint32_t bufferId = 0; bufferId < kSwigluUbStageNum; ++bufferId) {
        set_flag(PIPE_V, PIPE_MTE2, LoadFreeEvent(bufferId));
        set_flag(PIPE_MTE3, PIPE_V, StoreDoneEvent(bufferId));
    }
    for (uint32_t bufferId = 0; bufferId < kSwigluScaleChunkBuffers; ++bufferId) {
        set_flag(PIPE_MTE3, PIPE_S, ScaleStoreEvent(bufferId));
    }
}

template <typename InputElement>
AICORE inline void Swiglu<InputElement>::FinalizeFullRowPipeline() const
{
    for (uint32_t bufferId = 0; bufferId < kSwigluUbStageNum; ++bufferId) {
        wait_flag(PIPE_V, PIPE_MTE2, LoadFreeEvent(bufferId));
        wait_flag(PIPE_MTE3, PIPE_V, StoreDoneEvent(bufferId));
    }
    for (uint32_t bufferId = 0; bufferId < kSwigluScaleChunkBuffers; ++bufferId) {
        wait_flag(PIPE_MTE3, PIPE_S, ScaleStoreEvent(bufferId));
    }
}

template <typename InputElement>
AICORE inline void Swiglu<InputElement>::RunFullRowEpilogue(uint32_t localRowStart, uint32_t localRows) const
{
    if (localRows == 0U || outputN_ == 0U || !FullRowUbFits()) {
        return;
    }
    InitFullRowPipeline();
    uint32_t scaleChunkStart = 0;
    uint32_t scaleChunkCount = 0;
    uint32_t scaleBufferId = 0;
    IssueFullRowLoad(localRowStart, 0U);
    for (uint32_t localRow = 0; localRow < localRows; ++localRow) {
        const uint32_t bufferId = localRow % kSwigluUbStageNum;
        const uint32_t rowIdx = localRowStart + localRow;
        const float perTokenScale =
            PrepareFullRowCompute(rowIdx, bufferId); //  用来反量化 GMM1 结果，进入 SwiGLU fp32 计算
        if (localRow + 1U < localRows) {
            IssueFullRowLoad(localRowStart + localRow + 1U, (localRow + 1U) % kSwigluUbStageNum);
        }
        // SwiGLU 输出重新量化后的 scale，给后续 GMM2 反量化/scale 使用
        const float scale2 = ComputeAndStorePreparedFullRow(rowIdx, bufferId, perTokenScale);
        if (scaleChunkCount == 0U) {
            wait_flag(PIPE_MTE3, PIPE_S, ScaleStoreEvent(scaleBufferId));
        }
        PtoSetValue<float, kSwigluScaleTileElems>(SwigluScaleOutputOffset(scaleBufferId), scaleChunkCount, scale2);
        ++scaleChunkCount;
        if (scaleChunkCount == kSwigluScaleTileElems) { // 满128行，一起写回GM
            IssueStoreScale2Chunk(localRowStart + scaleChunkStart, scaleChunkCount, scaleBufferId);
            scaleChunkStart += scaleChunkCount;
            scaleChunkCount = 0;
            scaleBufferId = scaleBufferId + 1U == kSwigluScaleChunkBuffers ? 0U : scaleBufferId + 1U;
        }
    }
    if (scaleChunkCount > 0U) {
        IssueStoreScale2Chunk(localRowStart + scaleChunkStart, scaleChunkCount, scaleBufferId);
    }
    FinalizeFullRowPipeline();
}

template <typename InputElement>
AICORE inline void Swiglu<InputElement>::IssueStoreScale2Chunk(uint32_t rowStart, uint32_t rowCount,
                                                               uint32_t scaleBufferId) const
{
    if (rowCount == 0U) {
        return;
    }
    set_flag(PIPE_S, PIPE_MTE3, ScaleStoreEvent(scaleBufferId));
    wait_flag(PIPE_S, PIPE_MTE3, ScaleStoreEvent(scaleBufferId));
    PtoStoreVector<float, kSwigluScaleTileElems>(perTokenScale2Ptr_ + rowStart, SwigluScaleOutputOffset(scaleBufferId),
                                                 rowCount);
    set_flag(PIPE_MTE3, PIPE_S, ScaleStoreEvent(scaleBufferId));
}

template <typename InputElement>
AICORE inline void Swiglu<InputElement>::IssueFullRowLoad(uint32_t rowIdx, uint32_t bufferId) const
{
    using TileC = PtoVecTile<half, kSwigluVecTileElems>;
    using VectorShape = pto::Shape<1, 1, 1, 1, pto::DYNAMIC>;
    using VectorStride = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, 1>;
    using CGlobal = pto::GlobalTensor<half, VectorShape, VectorStride, pto::Layout::ND>;
    using BlockTileC = pto::Tile<pto::TileType::Vec, half, kSwigluFullRowIoBlockChunks, kSwigluVecTileElems,
                                 pto::BLayout::RowMajor, -1, -1>;
    using BlockShape = pto::Shape<1, 1, 1, pto::DYNAMIC, pto::DYNAMIC>;
    using BlockStride = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, 1>;
    using CBlockGlobal = pto::GlobalTensor<half, BlockShape, BlockStride, pto::Layout::ND>;

    const uint64_t ubCOffset = SwigluCOffset(bufferId);
    __gm__ half *gmCRow = gmCPtr_ + static_cast<uint64_t>(rowIdx) * problemN_;

    wait_flag(PIPE_V, PIPE_MTE2, LoadFreeEvent(bufferId));
    uint32_t offset = 0;
    while (problemN_ - offset >= kSwigluVecTileElems) {
        const uint32_t fullChunks = (problemN_ - offset) / kSwigluVecTileElems;
        const uint32_t chunkRows = fullChunks > kSwigluFullRowIoBlockChunks ? kSwigluFullRowIoBlockChunks : fullChunks;
        BlockTileC cTile(chunkRows, kSwigluVecTileElems);
        pto::TASSIGN(cTile, ubCOffset + static_cast<uint64_t>(offset) * sizeof(half));
        BlockShape cShape(chunkRows, kSwigluVecTileElems);
        BlockStride cStride(static_cast<int64_t>(chunkRows) * kSwigluVecTileElems,
                            static_cast<int64_t>(chunkRows) * kSwigluVecTileElems,
                            static_cast<int64_t>(chunkRows) * kSwigluVecTileElems, kSwigluVecTileElems);
        CBlockGlobal cGlobal(gmCRow + offset, cShape, cStride);
        pto::TLOAD(cTile, cGlobal);
        offset += chunkRows * kSwigluVecTileElems;
    }
    if (offset < problemN_) {
        const uint32_t cur = problemN_ - offset;
        TileC cTile(1, cur);
        pto::TASSIGN(cTile, ubCOffset + static_cast<uint64_t>(offset) * sizeof(half));
        VectorShape cShape(cur);
        VectorStride cStride(cur, cur, cur, cur);
        CGlobal cGlobal(gmCRow + offset, cShape, cStride);
        pto::TLOAD(cTile, cGlobal);
    }
    set_flag(PIPE_MTE2, PIPE_V, LoadReadyEvent(bufferId));
}

template <typename InputElement>
AICORE inline float Swiglu<InputElement>::PrepareFullRowCompute(uint32_t rowIdx, uint32_t bufferId) const
{
    using TileC = PtoVecTile<half, kSwigluVecTileElems>;
    using TileFp32 = PtoVecTile<float, kSwigluVecTileElems>;

    const uint64_t ubCOffset = SwigluCOffset(bufferId);
    const uint64_t ubCFp32Offset = SwigluCFp32Offset(bufferId);

    wait_flag(PIPE_MTE2, PIPE_V, LoadReadyEvent(bufferId));
    for (uint32_t offset = 0; offset < problemN_; offset += kSwigluVecTileElems) {
        const uint32_t cur = problemN_ - offset > kSwigluVecTileElems ? kSwigluVecTileElems : problemN_ - offset;
        TileFp32 fp32Tile(1, cur);
        TileC cTile(1, cur);
        const uint64_t elemOffset = static_cast<uint64_t>(offset);
        pto::TASSIGN(fp32Tile, ubCFp32Offset + elemOffset * sizeof(float));
        pto::TASSIGN(cTile, ubCOffset + elemOffset * sizeof(half));
        pto::TCVT(fp32Tile, cTile, pto::RoundMode::CAST_NONE);
    }
    set_flag(PIPE_V, PIPE_MTE2, LoadFreeEvent(bufferId));

    return perTokenScalePtr_[rowIdx];
}

template <typename InputElement>
AICORE inline float Swiglu<InputElement>::ComputeAndStorePreparedFullRow(uint32_t rowIdx, uint32_t bufferId,
                                                                         float perTokenScale) const
{
    using TileFp32 = PtoVecTile<float, kSwigluVecTileElems>;  // fp32 vector tile, one chunk has up to 1024 elems
    using TileHalf = PtoVecTile<half, kSwigluVecTileElems>;   // half scratch for A5-supported fp32->half->int8 cast path
    using TileD = PtoVecTile<int8_t, kSwigluVecTileElems>;    // int8 tile for qswiglu output
    using VectorShape = pto::Shape<1, 1, 1, 1, pto::DYNAMIC>;
    using VectorStride = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, 1>;
    using DGlobal = pto::GlobalTensor<int8_t, VectorShape, VectorStride, pto::Layout::ND>; // tail GM int8 view
    using BlockTileD = pto::Tile<pto::TileType::Vec, int8_t, kSwigluFullRowIoBlockChunks, kSwigluVecTileElems,
                                 pto::BLayout::RowMajor, -1, -1>;
    using BlockShape = pto::Shape<1, 1, 1, pto::DYNAMIC, pto::DYNAMIC>;
    using BlockStride = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, 1>;
    using DBlockGlobal = pto::GlobalTensor<int8_t, BlockShape, BlockStride, pto::Layout::ND>;      // block GM int8 view
    using RowMaxTile = pto::Tile<pto::TileType::Vec, float, 8, 1, pto::BLayout::ColMajor, -1, 1>;  // chunk max
    using ScalarTile = pto::Tile<pto::TileType::Vec, float, 1, 8, pto::BLayout::RowMajor, -1, -1>; // scalar max merge

    const uint64_t ubDOffset = SwigluDOffset(bufferId);         // UB output buffer for qswiglu[int8]
    const uint64_t ubCFp32Offset = SwigluCFp32Offset(bufferId); // UB buffer holding dequant1[problemN] fp32
    const uint64_t ubWorkOffset = SwigluWorkOffset(bufferId);   // UB work buffer for SwiGLU intermediate/output
    const uint64_t ubAbsOffset = SwigluAbsOffset(bufferId);     // UB scratch buffer for abs and quantized values
    const uint64_t ubMaxOffset = SwigluMaxOffset(bufferId);     // UB scalar slot for maxAbs
    __gm__ int8_t *gmDRow = gmPermutedTokenPtr_ + static_cast<uint64_t>(rowIdx) * outputN_; // GM qswiglu row

    pto::PtoSetWaitFlag<PIPE_S, PIPE_V>(); // perTokenScale scalar is ready for vector pipe
    pipe_barrier(PIPE_ALL);
    for (uint32_t offset = 0; offset < problemN_; offset += kSwigluVecTileElems) { // 1024为单位，切分N
        const uint32_t cur = problemN_ - offset > kSwigluVecTileElems ? kSwigluVecTileElems : problemN_ - offset;
        TileFp32 dstTile(1, cur);
        TileFp32 srcTile(1, cur);
        const uint64_t elemOffset = static_cast<uint64_t>(offset) * sizeof(float);
        pto::TASSIGN(dstTile, ubCFp32Offset + elemOffset);
        pto::TASSIGN(srcTile, ubCFp32Offset + elemOffset);
        pto::TMULS(dstTile, srcTile, perTokenScale); // dequant1 = fp32(gmC) * perTokenScale1[row]
    }
    pipe_barrier(PIPE_ALL);

    for (uint32_t offset = 0; offset < outputN_; offset += kSwigluVecTileElems) {
        const uint32_t cur = outputN_ - offset > kSwigluVecTileElems ? kSwigluVecTileElems : outputN_ - offset;
        TileFp32 dstTile(1, cur);
        TileFp32 srcTile(1, cur);
        pto::TASSIGN(dstTile, ubWorkOffset + static_cast<uint64_t>(offset) * sizeof(float));
        pto::TASSIGN(srcTile, ubCFp32Offset + static_cast<uint64_t>(offset) * sizeof(float));
        pto::TMULS(dstTile, srcTile, -1.0f); // work = -x
    }
    pipe_barrier(PIPE_ALL);
    for (uint32_t offset = 0; offset < outputN_; offset += kSwigluVecTileElems) {
        const uint32_t cur = outputN_ - offset > kSwigluVecTileElems ? kSwigluVecTileElems : outputN_ - offset;
        TileFp32 dstTile(1, cur);
        TileFp32 srcTile(1, cur);
        const uint64_t elemOffset = static_cast<uint64_t>(offset) * sizeof(float);
        pto::TASSIGN(dstTile, ubWorkOffset + elemOffset);
        pto::TASSIGN(srcTile, ubWorkOffset + elemOffset);
        pto::TEXP(dstTile, srcTile); // work = exp(-x)
    }
    pipe_barrier(PIPE_ALL);
    for (uint32_t offset = 0; offset < outputN_; offset += kSwigluVecTileElems) {
        const uint32_t cur = outputN_ - offset > kSwigluVecTileElems ? kSwigluVecTileElems : outputN_ - offset;
        TileFp32 dstTile(1, cur);
        TileFp32 srcTile(1, cur);
        const uint64_t elemOffset = static_cast<uint64_t>(offset) * sizeof(float);
        pto::TASSIGN(dstTile, ubWorkOffset + elemOffset);
        pto::TASSIGN(srcTile, ubWorkOffset + elemOffset);
        pto::TADDS(dstTile, srcTile, 1.0f); // work = 1 + exp(-x)
    }
    pipe_barrier(PIPE_ALL);
    for (uint32_t offset = 0; offset < outputN_; offset += kSwigluVecTileElems) {
        const uint32_t cur = outputN_ - offset > kSwigluVecTileElems ? kSwigluVecTileElems : outputN_ - offset;
        TileFp32 dstTile(1, cur);
        TileFp32 xTile(1, cur);
        TileFp32 denomTile(1, cur);
        const uint64_t elemOffset = static_cast<uint64_t>(offset) * sizeof(float);
        pto::TASSIGN(dstTile, ubWorkOffset + elemOffset);
        pto::TASSIGN(xTile, ubCFp32Offset + elemOffset);
        pto::TASSIGN(denomTile, ubWorkOffset + elemOffset);
        pto::TDIV(dstTile, xTile, denomTile); // work = silu(x) = x / (1 + exp(-x))
    }
    pipe_barrier(PIPE_ALL);
    for (uint32_t offset = 0; offset < outputN_; offset += kSwigluVecTileElems) {
        const uint32_t cur = outputN_ - offset > kSwigluVecTileElems ? kSwigluVecTileElems : outputN_ - offset;
        TileFp32 dstTile(1, cur);
        TileFp32 siluTile(1, cur);
        TileFp32 gateTile(1, cur);
        const uint64_t elemOffset = static_cast<uint64_t>(offset) * sizeof(float);
        pto::TASSIGN(dstTile, ubWorkOffset + elemOffset);
        pto::TASSIGN(siluTile, ubWorkOffset + elemOffset);
        pto::TASSIGN(gateTile, ubCFp32Offset + static_cast<uint64_t>(outputN_ + offset) * sizeof(float)); // gate
        pto::TMUL(dstTile, siluTile, gateTile); // swiglu_out = silu(x) * gate
    }
    pipe_barrier(PIPE_ALL);

    for (uint32_t offset = 0; offset < outputN_; offset += kSwigluVecTileElems) {
        const uint32_t cur = outputN_ - offset > kSwigluVecTileElems ? kSwigluVecTileElems : outputN_ - offset;
        TileFp32 absTile(1, cur);
        TileFp32 srcTile(1, cur);
        const uint64_t elemOffset = static_cast<uint64_t>(offset) * sizeof(float);
        pto::TASSIGN(absTile, ubAbsOffset + elemOffset);
        pto::TASSIGN(srcTile, ubWorkOffset + elemOffset);
        pto::TABS(absTile, srcTile); // abs = abs(swiglu_out)
    }
    pipe_barrier(PIPE_ALL);

    bool firstReduceChunk = true;
    for (uint32_t offset = 0; offset < outputN_; offset += kSwigluVecTileElems) {
        const uint32_t cur = outputN_ - offset > kSwigluVecTileElems ? kSwigluVecTileElems : outputN_ - offset;
        TileFp32 absTile(1, cur);
        TileFp32 tmpTile(1, cur);
        RowMaxTile rowMaxTile(1);
        const uint64_t elemOffset = static_cast<uint64_t>(offset) * sizeof(float);
        pto::TASSIGN(absTile, ubAbsOffset + elemOffset);
        pto::TASSIGN(tmpTile, ubAbsOffset + elemOffset);
        pto::TASSIGN(rowMaxTile, firstReduceChunk ? ubMaxOffset : ubAbsOffset); // first max goes to ubMaxOffset
        pto::TROWMAX(rowMaxTile, absTile, tmpTile);                             // reduce current chunk to one max value
        pipe_barrier(PIPE_ALL); // wait before reading or merging this chunk max
        if (!firstReduceChunk) {
            ScalarTile accTile(1, 1);
            ScalarTile newTile(1, 1);
            ScalarTile dstTile(1, 1);
            pto::TASSIGN(accTile, ubMaxOffset);
            pto::TASSIGN(newTile, ubAbsOffset);
            pto::TASSIGN(dstTile, ubMaxOffset);
            pto::TMAX(dstTile, accTile, newTile); // maxAbs = max(previousMaxAbs, currentChunkMax)
            pipe_barrier(PIPE_ALL);                 // wait for merged maxAbs
        }
        firstReduceChunk = false;
    }
    pipe_barrier(PIPE_ALL);

    pto::PtoSetWaitFlag<PIPE_V, PIPE_S>(); // maxAbs is ready for scalar pipe
    TileFp32 maxScalarTile(1, 1);
    pto::TASSIGN(maxScalarTile, ubMaxOffset);
    const float maxAbs = maxScalarTile.GetValue(0); // row-wise max(abs(swiglu_out))
    const float scale2 = maxAbs > 0.0f ? maxAbs / 127.0f : kSwigluDynamicQuantEps / 127.0f; // perTokenScale2
    const float quantScale = maxAbs > 0.0f ? 127.0f / maxAbs : 0.0f;                        // multiplier for int8 quant
    set_flag(PIPE_S, PIPE_V, EVENT_ID0); // quantScale is ready for vector pipe

    wait_flag(PIPE_S, PIPE_V, EVENT_ID0); // vector pipe waits for quantScale
    for (uint32_t offset = 0; offset < outputN_; offset += kSwigluVecTileElems) {
        const uint32_t cur = outputN_ - offset > kSwigluVecTileElems ? kSwigluVecTileElems : outputN_ - offset;
        TileFp32 dstTile(1, cur);
        TileFp32 srcTile(1, cur);
        const uint64_t elemOffset = static_cast<uint64_t>(offset) * sizeof(float);
        pto::TASSIGN(dstTile, ubAbsOffset + elemOffset);
        pto::TASSIGN(srcTile, ubWorkOffset + elemOffset);
        pto::TMULS(dstTile, srcTile, quantScale); // fp32Quant = swiglu_out * 127 / maxAbs
    }
    pipe_barrier(PIPE_ALL);

    for (uint32_t offset = 0; offset < outputN_; offset += kSwigluVecTileElems) {
        const uint32_t cur = outputN_ - offset > kSwigluVecTileElems ? kSwigluVecTileElems : outputN_ - offset;
        TileHalf halfTile(1, cur);
        TileFp32 srcTile(1, cur);
        const uint64_t elemOffset = static_cast<uint64_t>(offset);
        pto::TASSIGN(halfTile, ubWorkOffset + elemOffset * sizeof(half));
        pto::TASSIGN(srcTile, ubAbsOffset + elemOffset * sizeof(float));
        pto::TCVT(halfTile, srcTile, pto::RoundMode::CAST_NONE); // A5 supports fp32 -> half, then half -> int8
    }
    pipe_barrier(PIPE_ALL);

    wait_flag(PIPE_MTE3, PIPE_V,
              StoreDoneEvent(bufferId)); // avoid reusing output buffer before previous GM store finishes
    for (uint32_t offset = 0; offset < outputN_; offset += kSwigluVecTileElems) {
        const uint32_t cur = outputN_ - offset > kSwigluVecTileElems ? kSwigluVecTileElems : outputN_ - offset;
        TileD dTile(1, cur);
        TileHalf halfTile(1, cur);
        const uint64_t elemOffset = static_cast<uint64_t>(offset);
        pto::TASSIGN(dTile, ubDOffset + elemOffset * sizeof(int8_t));
        pto::TASSIGN(halfTile, ubWorkOffset + elemOffset * sizeof(half));
        pto::TCVT(dTile, halfTile, pto::RoundMode::CAST_RINT); // half -> int8, produces qswiglu
    }
    set_flag(PIPE_V, PIPE_MTE3, StoreReadyEvent(bufferId)); // qswiglu UB buffer is ready for GM store

    wait_flag(PIPE_V, PIPE_MTE3, StoreReadyEvent(bufferId)); // MTE3 waits for qswiglu
    uint32_t offset = 0;
    while (outputN_ - offset >= kSwigluVecTileElems) {
        const uint32_t fullChunks = (outputN_ - offset) / kSwigluVecTileElems;
        const uint32_t chunkRows = fullChunks > kSwigluFullRowIoBlockChunks ? kSwigluFullRowIoBlockChunks : fullChunks;
        BlockTileD dTile(chunkRows, kSwigluVecTileElems);
        pto::TASSIGN(dTile, ubDOffset + static_cast<uint64_t>(offset) * sizeof(int8_t));
        BlockShape dShape(chunkRows, kSwigluVecTileElems);
        BlockStride dStride(static_cast<int64_t>(chunkRows) * kSwigluVecTileElems,
                            static_cast<int64_t>(chunkRows) * kSwigluVecTileElems,
                            static_cast<int64_t>(chunkRows) * kSwigluVecTileElems, kSwigluVecTileElems);
        DBlockGlobal dGlobal(gmDRow + offset, dShape, dStride); // GM target for full 1024-element chunks
        pto::TSTORE(dGlobal, dTile);                            // store qswiglu[int8] to gmPermutedToken
        offset += chunkRows * kSwigluVecTileElems;
    }
    if (offset < outputN_) {
        const uint32_t cur = outputN_ - offset;
        TileD dTile(1, cur);
        pto::TASSIGN(dTile, ubDOffset + static_cast<uint64_t>(offset) * sizeof(int8_t));
        VectorShape dShape(cur);
        VectorStride dStride(cur, cur, cur, cur);
        DGlobal dGlobal(gmDRow + offset, dShape, dStride); // GM target for tail chunk
        pto::TSTORE(dGlobal, dTile);                       // store tail qswiglu[int8]
    }
    set_flag(PIPE_MTE3, PIPE_V, StoreDoneEvent(bufferId)); // GM store completed, buffer can be reused
    return scale2;                                         // caller stores this as perTokenScale2[row]
}

template <typename InputElement>
AICORE inline void Swiglu<InputElement>::Process()
{
    if ASCEND_IS_AIC {
        return;
    }
    const bool debugEnabled = DeviceDebug::SwigluDebugEnabled(*this);
#if DISPATCH_FFN_COMBINE_V8_ENABLE_INNER_PROFILE
    const bool stageProfileEnabled = profileEntry_ != nullptr;
#else
    const bool stageProfileEnabled = false;
#endif
    if (debugEnabled) {
        DeviceDebug::SwigluWriteLayoutDebugHeader(*this);
    }

    const uint32_t segmentNum = SwigluSegmentNum();
    bool readyStageRecorded = false;
    for (uint32_t segmentIdx = 0; segmentIdx < segmentNum; ++segmentIdx) {
        WaitC2VReady(segmentIdx);
        WriteSharedSegmentMetadata(segmentIdx);
        pto::SYNCALL<pto::SyncCoreType::AIVOnly>();

        uint32_t segmentStartExpert = 0;
        uint32_t segmentEndExpert = 0;
        uint32_t segmentRowBase = 0;
        uint32_t segmentRows = 0;
        uint32_t cumsumRows = 0;
        uint32_t expertTokenRows = 0;
        uint32_t localRowStart = 0;
        uint32_t localRows = 0;
        uint32_t rowSplitBase = 0;
        uint32_t rowSplitRem = 0;
        ReadSharedSegmentMetadata(segmentIdx, segmentStartExpert, segmentEndExpert, segmentRowBase, segmentRows,
                                  cumsumRows, expertTokenRows, rowSplitBase, rowSplitRem);
        localRows = rowSplitBase + (coreIdx_ < rowSplitRem ? 1U : 0U);
        const uint32_t prefixRows = coreIdx_ * rowSplitBase + (coreIdx_ < rowSplitRem ? coreIdx_ : rowSplitRem);
        localRowStart = segmentRowBase + prefixRows;
        if (stageProfileEnabled && localRows != 0U) {
            RecordReadyStageStartOnce(readyStageRecorded);
        }
        RunFullRowEpilogue(localRowStart, localRows); // swiglue实际处理
        pto::SYNCALL<pto::SyncCoreType::AIVOnly>();

        if (debugEnabled) {
            DeviceDebug::SwigluWriteC2VDebug(*this, segmentIdx);
            DeviceDebug::SwigluWriteSegmentDebug(*this, segmentIdx, segmentStartExpert, segmentEndExpert,
                                                 segmentRowBase, segmentRows, cumsumRows, expertTokenRows);
            DeviceDebug::SwigluWriteTaskDebug(*this, segmentIdx, segmentStartExpert, segmentEndExpert, segmentRowBase,
                                              segmentRows, localRowStart, localRows, rowSplitBase, rowSplitRem);
            DeviceDebug::SwigluWriteDoneDebug(*this, segmentIdx, segmentRowBase, segmentRows, localRowStart, localRows);
        }
        if (stageNum_ >= 12U) {
            SetV2CReady(segmentIdx);
        }
    }
    if (debugEnabled && stageNum_ == 11U) {
        pto::SYNCALL<pto::SyncCoreType::AIVOnly>();
        DeviceDebug::SwigluWriteFinalSyncDebug(*this);
    }
}

} // namespace dispatch_ffn_combine_v8

#endif // DISPATCH_FFN_COMBINE_V8_SWIGLU_H
