/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef MOE_DISPATCH_COMBINE_A8W8_LAYOUT_HPP_
#define MOE_DISPATCH_COMBINE_A8W8_LAYOUT_HPP_

#include "moe_new_dispatch_combine_a8w8_types.hpp"

#include <limits>
#include <stdexcept>
#include <string>

namespace moe_new_dispatch_combine_a8w8 {

inline uint64_t AlignUp(uint64_t value, uint64_t alignment)
{
    if (alignment == 0) {
        throw std::invalid_argument("alignment must be nonzero");
    }
    return ((value + alignment - 1) / alignment) * alignment;
}

inline uint64_t CheckedMul(uint64_t lhs, uint64_t rhs, const char *name)
{
    if (rhs != 0 && lhs > std::numeric_limits<uint64_t>::max() / rhs) {
        throw std::overflow_error(std::string("layout overflow: ") + name);
    }
    return lhs * rhs;
}

inline uint64_t CheckedAdd(uint64_t lhs, uint64_t rhs, const char *name)
{
    if (lhs > std::numeric_limits<uint64_t>::max() - rhs) {
        throw std::overflow_error(std::string("layout overflow: ") + name);
    }
    return lhs + rhs;
}

inline uint64_t ExpandedRows(const ShapeConfig &shape)
{
    return CheckedMul(shape.m, shape.topK, "expanded rows");
}

inline uint64_t LocalExpertRows(const ShapeConfig &shape)
{
    return CheckedMul(shape.maxTokensPerExpert, shape.expertPerRank, "local expert rows");
}

inline uint64_t GlobalExpertNum(const ShapeConfig &shape)
{
    return CheckedMul(shape.rankNum, shape.expertPerRank, "global expert num");
}

inline uint64_t TokenPerExpertMatrixRowStride(const ShapeConfig &shape)
{
    constexpr uint64_t kI32PerCacheLine = kCacheLineBytes / sizeof(int32_t);
    return AlignUp(GlobalExpertNum(shape), kI32PerCacheLine);
}

inline uint64_t TokenPerExpertMatrixStorageElements(const ShapeConfig &shape)
{
    return CheckedMul(shape.rankNum, TokenPerExpertMatrixRowStride(shape), "token matrix storage elems");
}

inline uint64_t DTypeBytes(uint32_t dtype)
{
    if (dtype == static_cast<uint32_t>(DType::kInt8)) {
        return 1;
    }
    if (dtype == static_cast<uint32_t>(DType::kFloat)) {
        return 4;
    }
    return 2;
}

inline uint64_t DispatchPayloadRowBytes(const ShapeConfig &shape)
{
    return AlignUp(CheckedMul(shape.hiddenSize, DTypeBytes(static_cast<uint32_t>(DType::kInt8)), "dispatch row"),
                   kCacheLineBytes);
}

inline uint64_t ReturnPayloadRowBytes(const ShapeConfig &shape)
{
    return AlignUp(CheckedMul(shape.hiddenSize, DTypeBytes(shape.dtypeOut), "return row"), kCacheLineBytes);
}

inline uint64_t CeilDiv(uint64_t value, uint64_t divisor)
{
    if (divisor == 0) {
        throw std::invalid_argument("divisor must be nonzero");
    }
    return (value + divisor - 1) / divisor;
}

inline uint64_t GmmBaseM()
{
    return kGmmBaseM;
}

inline uint64_t GmmBaseN()
{
    return kGmmBaseN;
}

inline uint64_t GmmBaseK()
{
    return kGmmBaseK;
}

inline uint64_t GmmStepK()
{
    return kGmmStepK;
}

inline uint64_t ReturnTileRows()
{
    return kReturnTileRows;
}

inline uint64_t ReturnHiddenChunkCols(const ShapeConfig &shape)
{
    return shape.gmmBlockN == 0 ? GmmBaseN() : shape.gmmBlockN;
}

inline uint64_t GmmTileTaskCapacity(const ShapeConfig &shape)
{
    uint64_t rowTiles = CeilDiv(LocalExpertRows(shape), GmmBaseM());
    uint64_t w1Cols = CheckedMul(shape.intermediateSize, 2, "w1 intermediate cols");
    uint64_t gmm1NTiles = CeilDiv(w1Cols, GmmBaseN());
    uint64_t gmm2NTiles = CeilDiv(shape.hiddenSize, GmmBaseN());
    uint64_t nTiles = gmm1NTiles > gmm2NTiles ? gmm1NTiles : gmm2NTiles;
    return CheckedMul(CheckedMul(shape.expertPerRank, rowTiles, "gmm row tiles"), nTiles, "gmm tile cap");
}

inline uint64_t ReturnSegmentCapacity(const ShapeConfig &shape)
{
    uint64_t rowTiles = CeilDiv(LocalExpertRows(shape), ReturnTileRows());
    uint64_t hiddenChunks = CeilDiv(shape.hiddenSize, ReturnHiddenChunkCols(shape));
    uint64_t perExpert =
        CheckedMul(CheckedMul(rowTiles, hiddenChunks, "return tile chunks"), shape.rankNum, "return owner segments");
    return CheckedMul(shape.expertPerRank, perExpert, "return segment cap");
}

inline uint64_t ReturnSegmentStagingBytes(const ShapeConfig &shape)
{
    return CheckedMul(CheckedMul(ReturnTileRows(), ReturnHiddenChunkCols(shape), "return staging elems"),
                      DTypeBytes(shape.dtypeOut), "return staging bytes");
}

inline FieldLayout AppendField(uint64_t *offset, uint64_t bytes, uint64_t alignment = kCacheLineBytes)
{
    uint64_t aligned = AlignUp(*offset, alignment);
    *offset = CheckedAdd(aligned, bytes, "append field");
    return FieldLayout{aligned, bytes, alignment};
}

inline WorkspaceLayout MakeWorkspaceLayout(const ShapeConfig &shape)
{
    WorkspaceLayout layout;
    uint64_t offset = 0;
    uint64_t expandedRows = ExpandedRows(shape);
    uint64_t localRows = LocalExpertRows(shape);
    uint64_t globalExpertNum = GlobalExpertNum(shape);
    uint64_t matrixCount = TokenPerExpertMatrixStorageElements(shape);
    uint64_t rankExpertCount = CheckedMul(shape.rankNum, shape.expertPerRank, "rank expert elems");
    uint64_t dispatchRowBytes = DispatchPayloadRowBytes(shape);
    uint64_t returnRowBytes = ReturnPayloadRowBytes(shape);
    uint64_t w1Cols = CheckedMul(shape.intermediateSize, 2, "w1 intermediate cols");
    uint64_t syncGroupCap = shape.expertPerRank + 1;
    uint64_t gmmTaskCap = GmmTileTaskCapacity(shape);
    uint64_t subTileCap = ReturnSegmentCapacity(shape);

    layout.tokenPerExpertMatrix = AppendField(&offset, CheckedMul(matrixCount, sizeof(int32_t), "token matrix"));
    layout.blockTokenPerExpert = AppendField(&offset, CheckedMul(globalExpertNum, sizeof(int32_t), "block counts"));
    layout.blockPrefixPerExpert = AppendField(&offset, CheckedMul(globalExpertNum, sizeof(int32_t), "block prefix"));
    layout.expandedRowIdx = AppendField(&offset, CheckedMul(expandedRows, sizeof(int32_t), "expanded row idx"));
    layout.packedRowToRouteIndex =
        AppendField(&offset, CheckedMul(expandedRows, sizeof(int32_t), "packed row route index"));
    layout.dispatchOffset = AppendField(&offset, CheckedMul(expandedRows, sizeof(int32_t), "dispatch offset"));
    layout.cumsumMM = AppendField(&offset, CheckedMul(rankExpertCount, sizeof(int32_t), "cumsumMM"));
    layout.preSumBeforeRank = AppendField(&offset, CheckedMul(rankExpertCount, sizeof(int32_t), "preSumBeforeRank"));
    layout.expertTokenNums = AppendField(&offset, CheckedMul(shape.expertPerRank, sizeof(int32_t), "expert nums"));
    layout.tokenOwnerRankOffsets = AppendField(&offset, CheckedMul(rankExpertCount, sizeof(int32_t), "owner offsets"));
    layout.dispatchedA = AppendField(&offset, CheckedMul(expandedRows, returnRowBytes, "dispatchedA"));
    layout.dispatchedScale = AppendField(&offset, CheckedMul(expandedRows, sizeof(float), "dispatchedScale"));
    layout.gmm1InputInt8 = AppendField(&offset, CheckedMul(localRows, dispatchRowBytes, "gmm1InputInt8"));
    layout.routingPerTokenScale = AppendField(&offset, CheckedMul(localRows, sizeof(float), "routing scale"));
    layout.gmm1WeightInt8 =
        AppendField(&offset, CheckedMul(CheckedMul(globalExpertNum, shape.hiddenSize, "w1 hidden"), w1Cols, "w1"));
    layout.scale1Uint64 = AppendField(&offset, CheckedMul(w1Cols, sizeof(uint64_t), "scale1"));
    layout.gmm1AccInt32 =
        AppendField(&offset, CheckedMul(CheckedMul(localRows, w1Cols, "gmm1 acc elems"), sizeof(int32_t), "gmm1 acc"));
    layout.gmm1Out =
        AppendField(&offset, CheckedMul(localRows, CheckedMul(w1Cols, sizeof(float), "gmm1 out row"), "gmm1 out"));
    layout.swigluOut = AppendField(
        &offset, CheckedMul(localRows, CheckedMul(shape.intermediateSize, sizeof(float), "swiglu row"), "swiglu"));
    layout.gmm2InputInt8 =
        AppendField(&offset, CheckedMul(localRows, AlignUp(shape.intermediateSize, kCacheLineBytes), "gmm2 input"));
    layout.gmm2PerTokenScale = AppendField(&offset, CheckedMul(localRows, sizeof(float), "gmm2 scale"));
    layout.gmm2WeightInt8 = AppendField(
        &offset,
        CheckedMul(CheckedMul(globalExpertNum, shape.intermediateSize, "w2 intermediate"), shape.hiddenSize, "w2"));
    layout.scale2Uint64 = AppendField(&offset, CheckedMul(shape.hiddenSize, sizeof(uint64_t), "scale2"));
    layout.gmm2AccInt32 = AppendField(
        &offset, CheckedMul(CheckedMul(localRows, shape.hiddenSize, "gmm2 acc elems"), sizeof(int32_t), "gmm2 acc"));
    layout.gmm2Out = AppendField(&offset, CheckedMul(localRows, returnRowBytes, "gmm2 out"));
    layout.returnSegmentStaging = AppendField(&offset, ReturnSegmentStagingBytes(shape));
    layout.readyCounters = AppendField(&offset, CheckedMul(16, kCacheLineBytes, "ready counters"));
    layout.dispatchGroupReady =
        AppendField(&offset, CheckedMul(shape.expertPerRank, kCacheLineBytes, "dispatch ready"));
    layout.gmm1SyncGroupReady = AppendField(&offset, CheckedMul(syncGroupCap, kCacheLineBytes, "gmm1 ready"));
    layout.activationSyncGroupReady =
        AppendField(&offset, CheckedMul(syncGroupCap, kCacheLineBytes, "activation ready"));
    layout.gmm2GroupReady = AppendField(&offset, CheckedMul(shape.expertPerRank, kCacheLineBytes, "gmm2 ready"));
    layout.stageStatus = AppendField(&offset, CheckedMul(16, kCacheLineBytes, "stage status"));
    layout.swigluSyncGroups = AppendField(&offset, CheckedMul(syncGroupCap, sizeof(int32_t), "swiglu groups"));
    layout.dequantSum = AppendField(&offset, CheckedMul(syncGroupCap + 1, sizeof(int32_t), "dequant sum"));
    layout.swigluGroupDesc = AppendField(&offset, CheckedMul(syncGroupCap, 8 * sizeof(int32_t), "swiglu desc"));
    layout.gmm1TileTaskPlan = AppendField(&offset, CheckedMul(gmmTaskCap, 8 * sizeof(int32_t), "gmm1 tasks"));
    layout.gmm2TileTaskPlan = AppendField(&offset, CheckedMul(gmmTaskCap, 8 * sizeof(int32_t), "gmm2 tasks"));
    layout.timeoutDump = AppendField(&offset, CheckedMul(shape.expertPerRank, kCacheLineBytes, "timeout dump"));
    layout.subTileReturnPlan = AppendField(&offset, CheckedMul(subTileCap, 8 * sizeof(int32_t), "sub tile plan"));
    layout.subTileOwnerSegments = AppendField(&offset, CheckedMul(subTileCap, 8 * sizeof(int32_t), "owner segments"));
    layout.subTileReady = AppendField(&offset, CheckedMul(subTileCap, kCacheLineBytes, "sub tile ready"));
    layout.timelineScratch = AppendField(&offset, CheckedMul(64, 4 * sizeof(uint64_t), "timeline scratch"));
    layout.totalBytes = AlignUp(offset, kCacheLineBytes);
    return layout;
}

inline PeerWindowLayout MakePeerWindowLayout(const ShapeConfig &shape)
{
    PeerWindowLayout layout;
    uint64_t offset = 0;
    uint64_t expandedRows = ExpandedRows(shape);
    uint64_t matrixCount = TokenPerExpertMatrixStorageElements(shape);
    layout.dispatchPayloadRowBytes = DispatchPayloadRowBytes(shape);
    layout.returnPayloadRowBytes = ReturnPayloadRowBytes(shape);
    layout.header = AppendField(&offset, kPeerWindowHeaderBytes);
    layout.tokenPerExpertMatrix = AppendField(&offset, CheckedMul(matrixCount, sizeof(int32_t), "peer token matrix"));
    layout.countReadySignal = AppendField(&offset, CheckedMul(shape.rankNum, kCacheLineBytes, "count signal"));
    layout.dispatchPayload =
        AppendField(&offset, CheckedMul(expandedRows, layout.dispatchPayloadRowBytes, "dispatch payload"));
    layout.dispatchScale = AppendField(&offset, CheckedMul(expandedRows, sizeof(float), "dispatch scale"));
    layout.returnPayload =
        AppendField(&offset, CheckedMul(expandedRows, layout.returnPayloadRowBytes, "return payload"));
    layout.combineDoneSignal = AppendField(&offset, CheckedMul(shape.rankNum, kCacheLineBytes, "combine signal"));
    layout.returnSegmentCounters =
        AppendField(&offset, CheckedMul(CheckedMul(shape.rankNum, shape.expertPerRank, "segment counters"),
                                        kCacheLineBytes, "segment counters bytes"));
    layout.debugCounters = AppendField(&offset, CheckedMul(256, kCacheLineBytes, "debug counters"));
    layout.timeline = AppendField(&offset, CheckedMul(64, 4 * sizeof(uint64_t), "timeline"));
    layout.totalBytes = AlignUp(offset, kCacheLineBytes);
    return layout;
}

inline PeerWindowHeader MakePeerWindowHeader(const ShapeConfig &shape, const RankConfig &rank,
                                             const PeerWindowLayout &layout)
{
    PeerWindowHeader header;
    header.rankNum = rank.rankNum;
    header.rankId = rank.rankId;
    header.expertPerRank = shape.expertPerRank;
    header.topK = shape.topK;
    header.dtypeIn = shape.dtypeIn;
    header.dtypeOut = shape.dtypeOut;
    header.dispatchPayloadRowBytes = layout.dispatchPayloadRowBytes;
    header.returnPayloadRowBytes = layout.returnPayloadRowBytes;
    return header;
}

} // namespace moe_new_dispatch_combine_a8w8

#endif // MOE_DISPATCH_COMBINE_A8W8_LAYOUT_HPP_
