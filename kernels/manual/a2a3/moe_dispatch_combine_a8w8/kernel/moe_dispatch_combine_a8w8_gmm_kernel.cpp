/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <cstdint>

#include <pto/common/pto_tile.hpp>
#include <pto/pto-inst.hpp>

#include "kernel_launchers.hpp"
#include "moe_dispatch_combine_a8w8_types.hpp"

#ifndef GM_ADDR
#define GM_ADDR __gm__ uint8_t *
#endif

namespace {

constexpr uint32_t kGmmBaseM = moe_dispatch_combine_a8w8::kGmmBaseM;
constexpr uint32_t kGmmBaseN = moe_dispatch_combine_a8w8::kGmmBaseN;
constexpr uint32_t kGmmBaseK = moe_dispatch_combine_a8w8::kGmmBaseK;
constexpr uint32_t kGmmStepK = moe_dispatch_combine_a8w8::kGmmStepK;
constexpr uint32_t kReturnTileRows = moe_dispatch_combine_a8w8::kReturnTileRows;
constexpr uint32_t kM2SwigluGroupFields = 8;
constexpr uint32_t kM2GmmTileTaskFields = 8;
constexpr uint32_t kM2ReturnPlanFields = 8;
constexpr uint32_t kM2OwnerSegmentFields = 8;
constexpr uint64_t kGmmL1ASingleBytes = static_cast<uint64_t>(kGmmBaseM) * kGmmBaseK * kGmmStepK * sizeof(int8_t);
constexpr uint64_t kGmmL1BSingleBytes = static_cast<uint64_t>(kGmmBaseK) * kGmmStepK * kGmmBaseN * sizeof(int8_t);
constexpr uint64_t kGmmL0ASingleBytes = static_cast<uint64_t>(kGmmBaseM) * kGmmBaseK * sizeof(int8_t);
constexpr uint64_t kGmmL0BSingleBytes = static_cast<uint64_t>(kGmmBaseK) * kGmmBaseN * sizeof(int8_t);
constexpr uint64_t kL1APing = 0x0;
constexpr uint64_t kL1APong = kL1APing + kGmmL1ASingleBytes;
constexpr uint64_t kL1BPing = kL1APing + 2U * kGmmL1ASingleBytes;
constexpr uint64_t kL1BPong = kL1BPing + kGmmL1BSingleBytes;
constexpr uint64_t kL0APing = 0x0;
constexpr uint64_t kL0APong = kL0APing + kGmmL0ASingleBytes;
constexpr uint64_t kL0BPing = 0x0;
constexpr uint64_t kL0BPong = kL0BPing + kGmmL0BSingleBytes;
constexpr uint64_t kL0C = 0x0;

using Shape2DDyn = pto::Shape<1, 1, 1, pto::DYNAMIC, pto::DYNAMIC>;
using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;

template <typename T>
using GlobalNd = pto::GlobalTensor<T, Shape2DDyn, StrideDyn, pto::Layout::ND>;

template <typename T>
AICORE inline GlobalNd<T> MakeGlobal2D(__gm__ T *ptr, int32_t rows, int32_t cols, int32_t rowStride)
{
    Shape2DDyn shape(rows, cols);
    StrideDyn stride(rowStride * rows, rowStride * rows, rowStride * rows, rowStride, 1);
    return GlobalNd<T>(ptr, shape, stride);
}

AICORE inline uint64_t Align64Device(uint64_t value)
{
    return ((value + 63U) / 64U) * 64U;
}

AICORE inline uint64_t M2CeilDivDevice(uint64_t value, uint64_t divisor)
{
    if (divisor == 0) {
        return 0;
    }
    return (value + divisor - 1U) / divisor;
}

AICORE inline uint64_t M2DTypeBytes(uint32_t dtype)
{
    if (dtype == static_cast<uint32_t>(moe_dispatch_combine_a8w8::DType::kInt8)) {
        return 1;
    }
    if (dtype == static_cast<uint32_t>(moe_dispatch_combine_a8w8::DType::kFloat)) {
        return 4;
    }
    return 2;
}

AICORE inline uint64_t M2GlobalExpertNum(moe_dispatch_combine_a8w8::ShapeConfig shape)
{
    return static_cast<uint64_t>(shape.rankNum) * shape.expertPerRank;
}

AICORE inline uint64_t M2LocalRows(moe_dispatch_combine_a8w8::ShapeConfig shape)
{
    return static_cast<uint64_t>(shape.maxTokensPerExpert) * shape.expertPerRank;
}

AICORE inline uint64_t M2TokenPerExpertMatrixRowStride(moe_dispatch_combine_a8w8::ShapeConfig shape)
{
    constexpr uint64_t kI32PerCacheLine = 16;
    uint64_t globalExpertNum = M2GlobalExpertNum(shape);
    return ((globalExpertNum + kI32PerCacheLine - 1) / kI32PerCacheLine) * kI32PerCacheLine;
}

AICORE inline uint64_t M2DispatchPayloadRowBytes(moe_dispatch_combine_a8w8::ShapeConfig shape)
{
    return Align64Device(static_cast<uint64_t>(shape.hiddenSize) *
                         M2DTypeBytes(static_cast<uint32_t>(moe_dispatch_combine_a8w8::DType::kInt8)));
}

AICORE inline uint64_t M2ReturnPayloadRowBytes(moe_dispatch_combine_a8w8::ShapeConfig shape)
{
    return Align64Device(static_cast<uint64_t>(shape.hiddenSize) * M2DTypeBytes(shape.dtypeOut));
}

AICORE inline uint64_t M2ReturnHiddenChunkCols(moe_dispatch_combine_a8w8::ShapeConfig shape)
{
    return shape.gmmBlockN == 0 ? kGmmBaseN : shape.gmmBlockN;
}

AICORE inline uint64_t M2GmmTileTaskCapacity(moe_dispatch_combine_a8w8::ShapeConfig shape)
{
    uint64_t rowTiles = M2CeilDivDevice(M2LocalRows(shape), kGmmBaseM);
    uint64_t w1Cols = static_cast<uint64_t>(shape.intermediateSize) * 2U;
    uint64_t gmm1NTiles = M2CeilDivDevice(w1Cols, kGmmBaseN);
    uint64_t gmm2NTiles = M2CeilDivDevice(shape.hiddenSize, kGmmBaseN);
    uint64_t nTiles = gmm1NTiles > gmm2NTiles ? gmm1NTiles : gmm2NTiles;
    return static_cast<uint64_t>(shape.expertPerRank) * rowTiles * nTiles;
}

AICORE inline uint64_t M2ReturnSegmentCapacity(moe_dispatch_combine_a8w8::ShapeConfig shape)
{
    uint64_t rowTiles = M2CeilDivDevice(M2LocalRows(shape), kReturnTileRows);
    uint64_t hiddenChunks = M2CeilDivDevice(shape.hiddenSize, M2ReturnHiddenChunkCols(shape));
    return static_cast<uint64_t>(shape.expertPerRank) * rowTiles * hiddenChunks * shape.rankNum;
}

AICORE inline moe_dispatch_combine_a8w8::FieldLayout M2AppendFieldDevice(uint64_t &offset, uint64_t bytes)
{
    offset = Align64Device(offset);
    moe_dispatch_combine_a8w8::FieldLayout field{offset, bytes, 64};
    offset += bytes;
    return field;
}

AICORE inline moe_dispatch_combine_a8w8::WorkspaceLayout MakeM2WorkspaceLayoutDevice(
    moe_dispatch_combine_a8w8::ShapeConfig shape)
{
    moe_dispatch_combine_a8w8::WorkspaceLayout layout{};
    uint64_t offset = 0;
    uint64_t expandedRows = static_cast<uint64_t>(shape.m) * shape.topK;
    uint64_t localRows = static_cast<uint64_t>(shape.maxTokensPerExpert) * shape.expertPerRank;
    uint64_t globalExpertNum = M2GlobalExpertNum(shape);
    uint64_t matrixCount = static_cast<uint64_t>(shape.rankNum) * M2TokenPerExpertMatrixRowStride(shape);
    uint64_t rankExpertCount = static_cast<uint64_t>(shape.rankNum) * shape.expertPerRank;
    uint64_t dispatchRowBytes = M2DispatchPayloadRowBytes(shape);
    uint64_t returnRowBytes = M2ReturnPayloadRowBytes(shape);
    uint64_t w1Cols = static_cast<uint64_t>(shape.intermediateSize) * 2U;
    uint64_t syncGroupCap = static_cast<uint64_t>(shape.expertPerRank) + 1U;
    uint64_t gmmTaskCap = M2GmmTileTaskCapacity(shape);
    uint64_t subTileCap = M2ReturnSegmentCapacity(shape);

    layout.tokenPerExpertMatrix = M2AppendFieldDevice(offset, matrixCount * sizeof(int32_t));
    layout.blockTokenPerExpert = M2AppendFieldDevice(offset, globalExpertNum * sizeof(int32_t));
    layout.blockPrefixPerExpert = M2AppendFieldDevice(offset, globalExpertNum * sizeof(int32_t));
    layout.expandedRowIdx = M2AppendFieldDevice(offset, expandedRows * sizeof(int32_t));
    layout.dispatchOffset = M2AppendFieldDevice(offset, expandedRows * sizeof(int32_t));
    layout.cumsumMM = M2AppendFieldDevice(offset, rankExpertCount * sizeof(int32_t));
    layout.preSumBeforeRank = M2AppendFieldDevice(offset, rankExpertCount * sizeof(int32_t));
    layout.expertTokenNums = M2AppendFieldDevice(offset, shape.expertPerRank * sizeof(int32_t));
    layout.tokenOwnerRankOffsets = M2AppendFieldDevice(offset, rankExpertCount * sizeof(int32_t));
    layout.dispatchedA = M2AppendFieldDevice(offset, expandedRows * returnRowBytes);
    layout.dispatchedScale = M2AppendFieldDevice(offset, expandedRows * sizeof(float));
    layout.gmm1InputInt8 = M2AppendFieldDevice(offset, localRows * dispatchRowBytes);
    layout.routingPerTokenScale = M2AppendFieldDevice(offset, localRows * sizeof(float));
    layout.gmm1WeightInt8 = M2AppendFieldDevice(offset, globalExpertNum * shape.hiddenSize * w1Cols);
    layout.scale1Uint64 = M2AppendFieldDevice(offset, w1Cols * sizeof(uint64_t));
    layout.gmm1AccInt32 = M2AppendFieldDevice(offset, localRows * w1Cols * sizeof(int32_t));
    layout.gmm1Out = M2AppendFieldDevice(offset, localRows * w1Cols * sizeof(float));
    layout.swigluOut = M2AppendFieldDevice(offset, localRows * shape.intermediateSize * sizeof(float));
    layout.gmm2InputInt8 = M2AppendFieldDevice(offset, localRows * Align64Device(shape.intermediateSize));
    layout.gmm2PerTokenScale = M2AppendFieldDevice(offset, localRows * sizeof(float));
    layout.gmm2WeightInt8 = M2AppendFieldDevice(offset, globalExpertNum * shape.intermediateSize * shape.hiddenSize);
    layout.scale2Uint64 = M2AppendFieldDevice(offset, shape.hiddenSize * sizeof(uint64_t));
    layout.gmm2AccInt32 = M2AppendFieldDevice(offset, localRows * shape.hiddenSize * sizeof(int32_t));
    layout.gmm2Out = M2AppendFieldDevice(offset, localRows * returnRowBytes);
    layout.returnSegmentStaging =
        M2AppendFieldDevice(offset, kReturnTileRows * M2ReturnHiddenChunkCols(shape) * M2DTypeBytes(shape.dtypeOut));
    layout.readyCounters = M2AppendFieldDevice(offset, 16U * 64U);
    layout.dispatchGroupReady = M2AppendFieldDevice(offset, shape.expertPerRank * 64U);
    layout.gmm1SyncGroupReady = M2AppendFieldDevice(offset, syncGroupCap * 64U);
    layout.activationSyncGroupReady = M2AppendFieldDevice(offset, syncGroupCap * 64U);
    layout.gmm2GroupReady = M2AppendFieldDevice(offset, shape.expertPerRank * 64U);
    layout.stageStatus = M2AppendFieldDevice(offset, 16U * 64U);
    layout.swigluSyncGroups = M2AppendFieldDevice(offset, syncGroupCap * sizeof(int32_t));
    layout.dequantSum = M2AppendFieldDevice(offset, (syncGroupCap + 1U) * sizeof(int32_t));
    layout.swigluGroupDesc = M2AppendFieldDevice(offset, syncGroupCap * kM2SwigluGroupFields * sizeof(int32_t));
    layout.gmm1TileTaskPlan = M2AppendFieldDevice(offset, gmmTaskCap * kM2GmmTileTaskFields * sizeof(int32_t));
    layout.gmm2TileTaskPlan = M2AppendFieldDevice(offset, gmmTaskCap * kM2GmmTileTaskFields * sizeof(int32_t));
    layout.timeoutDump = M2AppendFieldDevice(offset, shape.expertPerRank * 64U);
    layout.subTileReturnPlan = M2AppendFieldDevice(offset, subTileCap * kM2ReturnPlanFields * sizeof(int32_t));
    layout.subTileOwnerSegments = M2AppendFieldDevice(offset, subTileCap * kM2OwnerSegmentFields * sizeof(int32_t));
    layout.subTileReady = M2AppendFieldDevice(offset, subTileCap * 64U);
    layout.timelineScratch = M2AppendFieldDevice(offset, 64U * 4U * sizeof(uint64_t));
    layout.totalBytes = Align64Device(offset);
    return layout;
}

AICORE inline int32_t LoadScalarI32(__gm__ int32_t *ptr)
{
    return *ptr;
}

AICORE inline void StoreScalarI32(__gm__ int32_t *ptr, int32_t value)
{
    *ptr = value;
}

AICORE inline uint32_t BuildGmmTileTaskPlan(moe_dispatch_combine_a8w8::ShapeConfig shape,
                                            __gm__ int32_t *dispatchOffset, __gm__ int32_t *expertTokenNums,
                                            __gm__ int32_t *taskPlan, uint32_t nCols, uint32_t kSize, uint32_t stageId)
{
    uint32_t taskId = 0;
    uint32_t maxTasks = static_cast<uint32_t>(M2GmmTileTaskCapacity(shape));
    for (uint32_t localExpert = 0; localExpert < shape.expertPerRank; ++localExpert) {
        int32_t rowBegin = LoadScalarI32(dispatchOffset + localExpert);
        int32_t rowCount = LoadScalarI32(expertTokenNums + localExpert);
        if (rowCount <= 0) {
            continue;
        }
        for (uint32_t rowOffset = 0; rowOffset < static_cast<uint32_t>(rowCount); rowOffset += kGmmBaseM) {
            uint32_t rows = static_cast<uint32_t>(rowCount) - rowOffset;
            if (rows > kGmmBaseM) {
                rows = kGmmBaseM;
            }
            for (uint32_t nBase = 0; nBase < nCols; nBase += kGmmBaseN) {
                uint32_t cols = nCols - nBase;
                if (cols > kGmmBaseN) {
                    cols = kGmmBaseN;
                }
                if (taskId < maxTasks) {
                    __gm__ int32_t *task = taskPlan + taskId * kM2GmmTileTaskFields;
                    StoreScalarI32(task + 0U, static_cast<int32_t>(taskId));
                    StoreScalarI32(task + 1U, static_cast<int32_t>(stageId));
                    StoreScalarI32(task + 2U, static_cast<int32_t>(localExpert));
                    StoreScalarI32(task + 3U, rowBegin + static_cast<int32_t>(rowOffset));
                    StoreScalarI32(task + 4U, static_cast<int32_t>(rows));
                    StoreScalarI32(task + 5U, static_cast<int32_t>(nBase));
                    StoreScalarI32(task + 6U, static_cast<int32_t>(cols));
                    StoreScalarI32(task + 7U, static_cast<int32_t>(kSize));
                }
                ++taskId;
            }
        }
    }
    return taskId;
}

AICORE inline void RunInt8GmmTile(__gm__ int8_t *input, __gm__ int8_t *weight, __gm__ int32_t *output, uint32_t mValid,
                                  uint32_t kSize, uint32_t nValid, uint32_t inputStride, uint32_t weightStride,
                                  uint32_t outputStride)
{
    using L1A = pto::Tile<pto::TileType::Mat, int8_t, kGmmBaseM, kGmmBaseK * kGmmStepK, pto::BLayout::ColMajor,
                          pto::DYNAMIC, pto::DYNAMIC, pto::SLayout::RowMajor>;
    using L1B = pto::Tile<pto::TileType::Mat, int8_t, kGmmBaseK * kGmmStepK, kGmmBaseN, pto::BLayout::ColMajor,
                          pto::DYNAMIC, pto::DYNAMIC, pto::SLayout::RowMajor>;
    using L0A = pto::TileLeftCompact<int8_t, kGmmBaseM, kGmmBaseK, pto::DYNAMIC, pto::DYNAMIC>;
    using L0B = pto::TileRightCompact<int8_t, kGmmBaseK, kGmmBaseN, pto::DYNAMIC, pto::DYNAMIC>;
    using L0C = pto::TileAccCompact<int32_t, kGmmBaseM, kGmmBaseN, pto::DYNAMIC, pto::DYNAMIC>;

    L1A lhsMat[2] = {L1A(mValid, kGmmBaseK * kGmmStepK), L1A(mValid, kGmmBaseK * kGmmStepK)};
    L1B rhsMat[2] = {L1B(kGmmBaseK * kGmmStepK, nValid), L1B(kGmmBaseK * kGmmStepK, nValid)};
    L0A lhsTile[2] = {L0A(mValid, kGmmBaseK), L0A(mValid, kGmmBaseK)};
    L0B rhsTile[2] = {L0B(kGmmBaseK, nValid), L0B(kGmmBaseK, nValid)};
    L0C accTile(mValid, nValid);
    pto::TASSIGN(lhsMat[0], kL1APing);
    pto::TASSIGN(lhsMat[1], kL1APong);
    pto::TASSIGN(rhsMat[0], kL1BPing);
    pto::TASSIGN(rhsMat[1], kL1BPong);
    pto::TASSIGN(lhsTile[0], kL0APing);
    pto::TASSIGN(lhsTile[1], kL0APong);
    pto::TASSIGN(rhsTile[0], kL0BPing);
    pto::TASSIGN(rhsTile[1], kL0BPong);
    pto::TASSIGN(accTile, kL0C);
    lhsTile[0].SetKAligned(false);
    lhsTile[1].SetKAligned(false);

    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);

    uint8_t l1Stage = 0;
    uint8_t l0Stage = 0;
    uint32_t kLoop = kSize / kGmmBaseK;
    for (uint32_t kIter = 0; kIter < kLoop; ++kIter) {
        uint32_t kModStep = kIter % kGmmStepK;
        if (kModStep == 0) {
            wait_flag(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>(l1Stage));
            uint32_t kPanelSlices = kLoop - kIter;
            if (kPanelSlices > kGmmStepK) {
                kPanelSlices = kGmmStepK;
            }
            uint32_t kPanel = kPanelSlices * kGmmBaseK;
            GlobalNd<int8_t> lhsGlobal =
                MakeGlobal2D(input + static_cast<uint64_t>(kIter) * kGmmBaseK, mValid, kPanel, inputStride);
            GlobalNd<int8_t> rhsGlobal = MakeGlobal2D(weight + static_cast<uint64_t>(kIter) * kGmmBaseK * weightStride,
                                                      kPanel, nValid, weightStride);
            lhsMat[l1Stage].SetValidShape(mValid, kPanel);
            rhsMat[l1Stage].SetValidShape(kPanel, nValid);
            pto::TLOAD(lhsMat[l1Stage], lhsGlobal);
            set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
            pto::TLOAD(rhsMat[l1Stage], rhsGlobal);
            set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
            l1Stage = l1Stage == 0 ? 1 : 0;
        }

        uint8_t activeL1 = l1Stage == 0 ? 1 : 0;
        wait_flag(PIPE_M, PIPE_MTE1, static_cast<event_t>(l0Stage));
        if (kModStep == 0) {
            wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
        }
        pto::TEXTRACT(lhsTile[l0Stage], lhsMat[activeL1], 0, kModStep * kGmmBaseK);
        if (kModStep == 0) {
            wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
        }
        pto::TEXTRACT(rhsTile[l0Stage], rhsMat[activeL1], kModStep * kGmmBaseK, 0);
        if (((kIter + 1U) % kGmmStepK) == 0U || kIter + 1U == kLoop) {
            set_flag(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>(activeL1));
        }

        set_flag(PIPE_MTE1, PIPE_M, static_cast<event_t>(l0Stage));
        wait_flag(PIPE_MTE1, PIPE_M, static_cast<event_t>(l0Stage));
        if (kIter == 0) {
            pto::TMATMUL(accTile, lhsTile[l0Stage], rhsTile[l0Stage]);
        } else {
            pto::TMATMUL_ACC(accTile, accTile, lhsTile[l0Stage], rhsTile[l0Stage]);
        }
        set_flag(PIPE_M, PIPE_MTE1, static_cast<event_t>(l0Stage));
        l0Stage = l0Stage == 0 ? 1 : 0;
    }
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);

    GlobalNd<int32_t> outGlobal = MakeGlobal2D(output, mValid, nValid, outputStride);
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    pto::TSTORE(outGlobal, accTile);
    set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
}

} // namespace

#ifndef M2_FUSED_INCLUDE_DEVICE_BODY_ONLY
__global__ AICORE void M2Gmm1Int8(moe_dispatch_combine_a8w8::ShapeConfig shape,
                                  moe_dispatch_combine_a8w8::RankConfig rank, GM_ADDR workspace)
{
    uint32_t blockId = static_cast<uint32_t>(get_block_idx());
    uint32_t blockNum = static_cast<uint32_t>(get_block_num());
    if (shape.rankNum == 0 || shape.expertPerRank == 0 || shape.hiddenSize == 0U || shape.intermediateSize == 0U ||
        shape.gmmBlockM != kGmmBaseM || shape.gmmBlockN != kGmmBaseN || shape.gmmBlockK != kGmmBaseK ||
        shape.hiddenSize % kGmmBaseK != 0U || shape.intermediateSize % kGmmBaseK != 0U) {
        return;
    }
    auto layout = MakeM2WorkspaceLayoutDevice(shape);
    uint32_t rowBytes = static_cast<uint32_t>(M2DispatchPayloadRowBytes(shape));
    uint32_t w1Cols = shape.intermediateSize * 2U;
    __gm__ int8_t *gmm1Input = reinterpret_cast<__gm__ int8_t *>(workspace + layout.gmm1InputInt8.offset);
    __gm__ int8_t *weight1 = reinterpret_cast<__gm__ int8_t *>(workspace + layout.gmm1WeightInt8.offset);
    __gm__ int32_t *gmm1Acc = reinterpret_cast<__gm__ int32_t *>(workspace + layout.gmm1AccInt32.offset);
    __gm__ int32_t *dispatchOffset = reinterpret_cast<__gm__ int32_t *>(workspace + layout.dispatchOffset.offset);
    __gm__ int32_t *expertTokenNums = reinterpret_cast<__gm__ int32_t *>(workspace + layout.expertTokenNums.offset);
    __gm__ int32_t *gmm1TileTaskPlan = reinterpret_cast<__gm__ int32_t *>(workspace + layout.gmm1TileTaskPlan.offset);
    __gm__ int32_t *stageStatus = reinterpret_cast<__gm__ int32_t *>(workspace + layout.stageStatus.offset);
    uint32_t taskCount =
        BuildGmmTileTaskPlan(shape, dispatchOffset, expertTokenNums, gmm1TileTaskPlan, w1Cols, shape.hiddenSize, 1U);
    if (blockId == 0) {
        StoreScalarI32(stageStatus + 4U * 16U, static_cast<int32_t>(taskCount));
        StoreScalarI32(stageStatus + 5U * 16U, static_cast<int32_t>(blockNum));
    }
    if (taskCount == 0) {
        return;
    }

    for (uint32_t taskId = blockId; taskId < taskCount; taskId += blockNum) {
        __gm__ int32_t *task = gmm1TileTaskPlan + taskId * kM2GmmTileTaskFields;
        uint32_t localExpert = static_cast<uint32_t>(LoadScalarI32(task + 2U));
        uint32_t globalExpert = rank.rankId * shape.expertPerRank + localExpert;
        uint32_t rowBegin = static_cast<uint32_t>(LoadScalarI32(task + 3U));
        uint32_t mValid = static_cast<uint32_t>(LoadScalarI32(task + 4U));
        uint32_t nBase = static_cast<uint32_t>(LoadScalarI32(task + 5U));
        uint32_t nValid = static_cast<uint32_t>(LoadScalarI32(task + 6U));
        __gm__ int8_t *expertWeight = weight1 + static_cast<uint64_t>(globalExpert) * shape.hiddenSize * w1Cols;
        __gm__ int8_t *tileInput = gmm1Input + static_cast<uint64_t>(rowBegin) * rowBytes;
        __gm__ int32_t *tileOutput = gmm1Acc + static_cast<uint64_t>(rowBegin) * w1Cols;
        RunInt8GmmTile(tileInput, expertWeight + nBase, tileOutput + nBase, mValid, shape.hiddenSize, nValid, rowBytes,
                       w1Cols, w1Cols);
    }
}

__global__ AICORE void M2Gmm2Int8(moe_dispatch_combine_a8w8::ShapeConfig shape,
                                  moe_dispatch_combine_a8w8::RankConfig rank, GM_ADDR workspace)
{
    uint32_t blockId = static_cast<uint32_t>(get_block_idx());
    uint32_t blockNum = static_cast<uint32_t>(get_block_num());
    if (shape.rankNum == 0 || shape.expertPerRank == 0 || shape.hiddenSize == 0U || shape.intermediateSize == 0U ||
        shape.gmmBlockM != kGmmBaseM || shape.gmmBlockN != kGmmBaseN || shape.gmmBlockK != kGmmBaseK ||
        shape.hiddenSize % kGmmBaseK != 0U || shape.intermediateSize % kGmmBaseK != 0U) {
        return;
    }
    auto layout = MakeM2WorkspaceLayoutDevice(shape);
    uint32_t rowBytes = static_cast<uint32_t>(Align64Device(shape.intermediateSize));
    __gm__ int8_t *gmm2Input = reinterpret_cast<__gm__ int8_t *>(workspace + layout.gmm2InputInt8.offset);
    __gm__ int8_t *weight2 = reinterpret_cast<__gm__ int8_t *>(workspace + layout.gmm2WeightInt8.offset);
    __gm__ int32_t *gmm2Acc = reinterpret_cast<__gm__ int32_t *>(workspace + layout.gmm2AccInt32.offset);
    __gm__ int32_t *dispatchOffset = reinterpret_cast<__gm__ int32_t *>(workspace + layout.dispatchOffset.offset);
    __gm__ int32_t *expertTokenNums = reinterpret_cast<__gm__ int32_t *>(workspace + layout.expertTokenNums.offset);
    __gm__ int32_t *gmm2GroupReady = reinterpret_cast<__gm__ int32_t *>(workspace + layout.gmm2GroupReady.offset);
    __gm__ int32_t *gmm2TileTaskPlan = reinterpret_cast<__gm__ int32_t *>(workspace + layout.gmm2TileTaskPlan.offset);
    __gm__ int32_t *stageStatus = reinterpret_cast<__gm__ int32_t *>(workspace + layout.stageStatus.offset);
    uint32_t taskCount = BuildGmmTileTaskPlan(shape, dispatchOffset, expertTokenNums, gmm2TileTaskPlan,
                                              shape.hiddenSize, shape.intermediateSize, 2U);
    if (blockId == 0) {
        StoreScalarI32(stageStatus + 6U * 16U, static_cast<int32_t>(taskCount));
        StoreScalarI32(stageStatus + 7U * 16U, static_cast<int32_t>(blockNum));
    }

    for (uint32_t localExpert = 0; localExpert < shape.expertPerRank; ++localExpert) {
        int32_t rowCount = LoadScalarI32(expertTokenNums + localExpert);
        if (rowCount <= 0) {
            StoreScalarI32(gmm2GroupReady + localExpert * 16U, 1);
        }
    }
    if (taskCount == 0) {
        return;
    }

    for (uint32_t taskId = blockId; taskId < taskCount; taskId += blockNum) {
        __gm__ int32_t *task = gmm2TileTaskPlan + taskId * kM2GmmTileTaskFields;
        uint32_t localExpert = static_cast<uint32_t>(LoadScalarI32(task + 2U));
        uint32_t globalExpert = rank.rankId * shape.expertPerRank + localExpert;
        uint32_t rowBegin = static_cast<uint32_t>(LoadScalarI32(task + 3U));
        uint32_t mValid = static_cast<uint32_t>(LoadScalarI32(task + 4U));
        uint32_t nBase = static_cast<uint32_t>(LoadScalarI32(task + 5U));
        uint32_t nValid = static_cast<uint32_t>(LoadScalarI32(task + 6U));
        __gm__ int8_t *expertWeight =
            weight2 + static_cast<uint64_t>(globalExpert) * shape.intermediateSize * shape.hiddenSize;
        __gm__ int8_t *tileInput = gmm2Input + static_cast<uint64_t>(rowBegin) * rowBytes;
        __gm__ int32_t *tileOutput = gmm2Acc + static_cast<uint64_t>(rowBegin) * shape.hiddenSize;
        RunInt8GmmTile(tileInput, expertWeight + nBase, tileOutput + nBase, mValid, shape.intermediateSize, nValid,
                       rowBytes, shape.hiddenSize, shape.hiddenSize);
    }
    pipe_barrier(PIPE_ALL);
    dsb(DSB_DDR);
    if (blockId == 0) {
        for (uint32_t localExpert = 0; localExpert < shape.expertPerRank; ++localExpert) {
            StoreScalarI32(gmm2GroupReady + localExpert * 16U, 1);
        }
    }
}

namespace dispatch_combine_tile {

void LaunchM2Gmm1Int8(moe_dispatch_combine_a8w8::ShapeConfig shape, moe_dispatch_combine_a8w8::RankConfig rank,
                      uint8_t *workspace, void *stream, uint32_t launchBlockCount)
{
    M2Gmm1Int8<<<launchBlockCount, nullptr, stream>>>(shape, rank, workspace);
}

void LaunchM2Gmm2Int8(moe_dispatch_combine_a8w8::ShapeConfig shape, moe_dispatch_combine_a8w8::RankConfig rank,
                      uint8_t *workspace, void *stream, uint32_t launchBlockCount)
{
    M2Gmm2Int8<<<launchBlockCount, nullptr, stream>>>(shape, rank, workspace);
}

} // namespace dispatch_combine_tile
#endif // M2_FUSED_INCLUDE_DEVICE_BODY_ONLY
