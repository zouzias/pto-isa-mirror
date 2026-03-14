/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_MOE_GROUPED_FFN_PROJECTION_COMMON_H
#define PTO_MOE_GROUPED_FFN_PROJECTION_COMMON_H

#include "lib/matmul_intf.h"
#include "kernel_operator.h"
#include "../../../../../include/pto/pto-inst.hpp"
#include "moe_grouped_ffn_custom.h"

using namespace pto;

namespace moe_grouped_ffn_projection {

constexpr uint32_t kBufferNum = 2;
constexpr uint32_t kL0PingpongBytes = 32 * 1024;

using ProjectionTileMatA =
    Tile<TileType::Mat, bfloat16_t, kMoeBaseM, kMoeBaseK * kMoeStepKa, BLayout::ColMajor, -1, -1,
         SLayout::RowMajor>;
using ProjectionTileMatB = Tile<TileType::Mat, bfloat16_t, kMoeBaseK * kMoeStepKb, kMoeBaseN, BLayout::RowMajor,
                                kMoeBaseK * kMoeStepKb, kMoeBaseN, SLayout::ColMajor>;
using ProjectionTileMatBView = Tile<TileType::Mat, bfloat16_t, kMoeBaseK * kMoeStepKb, kMoeBaseN, BLayout::ColMajor,
                                    kMoeBaseK * kMoeStepKb, kMoeBaseN, SLayout::RowMajor>;
using ProjectionLeftTile = TileLeftCompact<bfloat16_t, kMoeBaseM, kMoeBaseK, -1, -1>;
using ProjectionRightTile = TileRightCompact<bfloat16_t, kMoeBaseK, kMoeBaseN, -1, -1>;
using ProjectionResTile = TileAcc<float, kMoeBaseM, kMoeBaseN, -1, -1>;
using ProjectionFullTileMatA =
    Tile<TileType::Mat, bfloat16_t, kMoeBaseM, kMoeBaseK * kMoeStepKa, BLayout::ColMajor, kMoeBaseM,
         kMoeBaseK * kMoeStepKa, SLayout::RowMajor>;
using ProjectionFullLeftTile =
    TileLeftCompact<bfloat16_t, kMoeBaseM, kMoeBaseK, kMoeBaseM, kMoeBaseK>;
using ProjectionFullRightTile =
    TileRightCompact<bfloat16_t, kMoeBaseK, kMoeBaseN, kMoeBaseK, kMoeBaseN>;
using ProjectionFullResTile = TileAcc<float, kMoeBaseM, kMoeBaseN, kMoeBaseM, kMoeBaseN>;
constexpr uint32_t kMoeCombinedInterSize = static_cast<uint32_t>(kMoeInterSize * 2);

template <pipe_t srcPipe, pipe_t dstPipe>
AICORE inline void SetFlag(uint32_t id)
{
    set_flag(srcPipe, dstPipe, static_cast<event_t>(id));
}

template <pipe_t srcPipe, pipe_t dstPipe>
AICORE inline void WaitFlag(uint32_t id)
{
    wait_flag(srcPipe, dstPipe, static_cast<event_t>(id));
}

template <typename OutTile, typename LeftTile, typename RightTile>
AICORE inline void MatmulAcc(OutTile cTile, LeftTile aTile, RightTile bTile, uint32_t kIter)
{
    if (kIter == 0) {
        TMATMUL(cTile, aTile, bTile);
    } else {
        TMATMUL_ACC(cTile, cTile, aTile, bTile);
    }
}

template <typename TileMatA, typename TileMatB, typename LeftTile, typename RightTile, typename ResTile>
AICORE inline void InitBuffers(TileMatA aMatTile[kBufferNum], TileMatB bMatTile[kBufferNum], LeftTile aTile[kBufferNum],
                               RightTile bTile[kBufferNum], ResTile &cTile)
{
    TASSIGN(aMatTile[0], 0x0);
    TASSIGN(aMatTile[1], 0x0 + kMoeBaseM * kMoeBaseK * kMoeStepKa * sizeof(bfloat16_t));
    TASSIGN(bMatTile[0], 0x0 + kMoeBaseM * kMoeBaseK * kMoeStepKa * kBufferNum * sizeof(bfloat16_t));
    TASSIGN(bMatTile[1], 0x0 + kMoeBaseM * kMoeBaseK * kMoeStepKa * kBufferNum * sizeof(bfloat16_t) +
                               kMoeBaseK * kMoeBaseN * kMoeStepKb * sizeof(bfloat16_t));

    TASSIGN(aTile[0], 0x0);
    TASSIGN(aTile[1], 0x0 + kL0PingpongBytes);
    TASSIGN(bTile[0], 0x0);
    TASSIGN(bTile[1], 0x0 + kL0PingpongBytes);
    TASSIGN(cTile, 0x0);
}

AICORE inline void InitSyncFlags()
{
    SetFlag<PIPE_MTE1, PIPE_MTE2>(0);
    SetFlag<PIPE_MTE1, PIPE_MTE2>(1);
    SetFlag<PIPE_M, PIPE_MTE1>(0);
    SetFlag<PIPE_M, PIPE_MTE1>(1);
}

AICORE inline void WaitSyncFlags()
{
    WaitFlag<PIPE_M, PIPE_MTE1>(0);
    WaitFlag<PIPE_M, PIPE_MTE1>(1);
    WaitFlag<PIPE_MTE1, PIPE_MTE2>(0);
    WaitFlag<PIPE_MTE1, PIPE_MTE2>(1);
}

template <typename ProjTile>
AICORE inline void StoreProjectionToOutput(ProjTile &projTile, __gm__ float *currentDst, uint32_t rowCount)
{
    using CValidShape = TileShape2D<float, -1, -1, Layout::ND>;
    using CBaseShape = BaseShape2D<float, kMoeBaseM, kMoeInterSize, Layout::ND>;
    using CGlobal = GlobalTensor<float, CValidShape, CBaseShape, Layout::ND>;

    CGlobal dstGlobal(currentDst, CValidShape(rowCount, kMoeBaseN));
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    TSTORE(dstGlobal, projTile);
    set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
}

template <typename ProjTile, typename OutType>
AICORE inline void StoreProjectionToOutputStrided(ProjTile &projTile, __gm__ OutType *currentDst, uint32_t rowCount,
                                                  uint32_t outputWidth)
{
    using CValidShape = TileShape2D<OutType, -1, -1, Layout::ND>;
    using CBaseShape = BaseShape2D<OutType, -1, -1, Layout::ND>;
    using CGlobal = GlobalTensor<OutType, CValidShape, CBaseShape, Layout::ND>;

    CGlobal dstGlobal(currentDst, CValidShape(rowCount, kMoeBaseN), CBaseShape(rowCount, outputWidth));
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    TSTORE(dstGlobal, projTile);
    set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
}

AICORE inline void ProcessKIteration(uint32_t kIter, __gm__ bfloat16_t *currentSrc0, __gm__ bfloat16_t *currentSrc1,
                                     ProjectionTileMatA aMatTile[kBufferNum], ProjectionTileMatB bMatTile[kBufferNum],
                                     ProjectionLeftTile aTile[kBufferNum], ProjectionRightTile bTile[kBufferNum],
                                     ProjectionResTile &cTile, uint8_t &mte2DBFlag, uint8_t &mte1DBFlag,
                                     uint32_t currentM)
{
    using AValidShape = TileShape2D<bfloat16_t, -1, -1, Layout::ND>;
    using ABaseShape = BaseShape2D<bfloat16_t, kMoeBaseM, kMoeHiddenSize, Layout::ND>;
    using AGlobal = GlobalTensor<bfloat16_t, AValidShape, ABaseShape, Layout::ND>;

    using BValidShape = TileShape2D<bfloat16_t, kMoeBaseK * kMoeStepKb, kMoeBaseN, Layout::DN>;
    using BBaseShape = BaseShape2D<bfloat16_t, kMoeHiddenSize, kMoeInterSize, Layout::DN>;
    using BGlobal = GlobalTensor<bfloat16_t, BValidShape, BBaseShape, Layout::DN>;

    const uint32_t kModStepKa = kIter % kMoeStepKa;

    if (kModStepKa == 0) {
        AGlobal gmA(currentSrc0 + kIter * kMoeBaseK, AValidShape(currentM, kMoeBaseK * kMoeStepKa));
        BGlobal gmB(currentSrc1 + kIter * kMoeBaseK);

        WaitFlag<PIPE_MTE1, PIPE_MTE2>(mte2DBFlag);
        TLOAD(aMatTile[mte2DBFlag], gmA);
        SetFlag<PIPE_MTE2, PIPE_MTE1>(0);
        TLOAD(bMatTile[mte2DBFlag], gmB);
        SetFlag<PIPE_MTE2, PIPE_MTE1>(1);
        mte2DBFlag = (mte2DBFlag == 0) ? 1 : 0;
    }

    const uint32_t currMte2Idx = (mte2DBFlag == 0) ? 1 : 0;

    WaitFlag<PIPE_M, PIPE_MTE1>(mte1DBFlag);

    if (kModStepKa == 0) {
        WaitFlag<PIPE_MTE2, PIPE_MTE1>(0);
    }
    TEXTRACT(aTile[mte1DBFlag], aMatTile[currMte2Idx], 0, kModStepKa * kMoeBaseK);

    if (kModStepKa == 0) {
        WaitFlag<PIPE_MTE2, PIPE_MTE1>(1);
    }
    TEXTRACT(bTile[mte1DBFlag], bMatTile[currMte2Idx], (kIter % kMoeStepKb) * kMoeBaseK, 0);

    if ((kIter + 1) % kMoeStepKa == 0) {
        SetFlag<PIPE_MTE1, PIPE_MTE2>(currMte2Idx);
    }

    SetFlag<PIPE_MTE1, PIPE_M>(mte1DBFlag);
    WaitFlag<PIPE_MTE1, PIPE_M>(mte1DBFlag);
    MatmulAcc(cTile, aTile[mte1DBFlag], bTile[mte1DBFlag], kIter);
    SetFlag<PIPE_M, PIPE_MTE1>(mte1DBFlag);
    mte1DBFlag = (mte1DBFlag == 0) ? 1 : 0;
}

AICORE inline void ProcessKIterationFullM(uint32_t kIter, __gm__ bfloat16_t *currentSrc0,
                                          __gm__ bfloat16_t *currentSrc1,
                                          ProjectionFullTileMatA aMatTile[kBufferNum],
                                          ProjectionTileMatB bMatTile[kBufferNum],
                                          ProjectionFullLeftTile aTile[kBufferNum],
                                          ProjectionFullRightTile bTile[kBufferNum], ProjectionFullResTile &cTile,
                                          uint8_t &mte2DBFlag, uint8_t &mte1DBFlag)
{
    using AValidShape = TileShape2D<bfloat16_t, kMoeBaseM, kMoeBaseK * kMoeStepKa, Layout::ND>;
    using ABaseShape = BaseShape2D<bfloat16_t, kMoeBaseM, kMoeHiddenSize, Layout::ND>;
    using AGlobal = GlobalTensor<bfloat16_t, AValidShape, ABaseShape, Layout::ND>;

    using BValidShape = TileShape2D<bfloat16_t, kMoeBaseK * kMoeStepKb, kMoeBaseN, Layout::DN>;
    using BBaseShape = BaseShape2D<bfloat16_t, kMoeHiddenSize, kMoeInterSize, Layout::DN>;
    using BGlobal = GlobalTensor<bfloat16_t, BValidShape, BBaseShape, Layout::DN>;

    const uint32_t kModStepKa = kIter % kMoeStepKa;

    if (kModStepKa == 0) {
        AGlobal gmA(currentSrc0 + kIter * kMoeBaseK);
        BGlobal gmB(currentSrc1 + kIter * kMoeBaseK);

        WaitFlag<PIPE_MTE1, PIPE_MTE2>(mte2DBFlag);
        TLOAD(aMatTile[mte2DBFlag], gmA);
        SetFlag<PIPE_MTE2, PIPE_MTE1>(0);
        TLOAD(bMatTile[mte2DBFlag], gmB);
        SetFlag<PIPE_MTE2, PIPE_MTE1>(1);
        mte2DBFlag = (mte2DBFlag == 0) ? 1 : 0;
    }

    const uint32_t currMte2Idx = (mte2DBFlag == 0) ? 1 : 0;

    WaitFlag<PIPE_M, PIPE_MTE1>(mte1DBFlag);

    if (kModStepKa == 0) {
        WaitFlag<PIPE_MTE2, PIPE_MTE1>(0);
    }
    TEXTRACT(aTile[mte1DBFlag], aMatTile[currMte2Idx], 0, kModStepKa * kMoeBaseK);

    if (kModStepKa == 0) {
        WaitFlag<PIPE_MTE2, PIPE_MTE1>(1);
    }
    TEXTRACT(bTile[mte1DBFlag], bMatTile[currMte2Idx], (kIter % kMoeStepKb) * kMoeBaseK, 0);

    if ((kIter + 1) % kMoeStepKa == 0) {
        SetFlag<PIPE_MTE1, PIPE_MTE2>(currMte2Idx);
    }

    SetFlag<PIPE_MTE1, PIPE_M>(mte1DBFlag);
    WaitFlag<PIPE_MTE1, PIPE_M>(mte1DBFlag);
    MatmulAcc(cTile, aTile[mte1DBFlag], bTile[mte1DBFlag], kIter);
    SetFlag<PIPE_M, PIPE_MTE1>(mte1DBFlag);
    mte1DBFlag = (mte1DBFlag == 0) ? 1 : 0;
}

AICORE inline void RunProjectionKernel(__gm__ bfloat16_t *xPtr, __gm__ bfloat16_t *weightPtr, __gm__ float *projPtr,
                                       uint32_t expertId, uint32_t rowOffset, uint32_t colTile, uint32_t currentM)
{
    if (currentM == 0) {
        return;
    }

    __gm__ bfloat16_t *currentSrc0 = xPtr + rowOffset * kMoeHiddenSize;
    __gm__ float *currentDst = projPtr + static_cast<std::size_t>(rowOffset) * kMoeInterSize;
    const uint32_t colOffset = colTile * static_cast<uint32_t>(kMoeBaseN);
    __gm__ bfloat16_t *currentSrc1 = weightPtr + (expertId * kMoeInterSize + colOffset) * kMoeHiddenSize;

    ProjectionTileMatA aMatTile[kBufferNum] = {ProjectionTileMatA(currentM, kMoeBaseK * kMoeStepKa),
                                               ProjectionTileMatA(currentM, kMoeBaseK * kMoeStepKa)};
    ProjectionTileMatB bMatTile[kBufferNum];
    ProjectionLeftTile aTile[kBufferNum] = {ProjectionLeftTile(currentM, kMoeBaseK),
                                            ProjectionLeftTile(currentM, kMoeBaseK)};
    ProjectionRightTile bTile[kBufferNum] = {ProjectionRightTile(kMoeBaseK, kMoeBaseN),
                                             ProjectionRightTile(kMoeBaseK, kMoeBaseN)};
    ProjectionResTile cTile(currentM, kMoeBaseN);

    InitBuffers(aMatTile, bMatTile, aTile, bTile, cTile);
    InitSyncFlags();

    uint8_t mte2DBFlag = 0;
    uint8_t mte1DBFlag = 0;
    constexpr uint32_t kLoop = kMoeHiddenSize / kMoeBaseK;

    for (uint32_t kIter = 0; kIter < kLoop; ++kIter) {
        ProcessKIteration(kIter, currentSrc0, currentSrc1, aMatTile, bMatTile, aTile, bTile, cTile, mte2DBFlag,
                          mte1DBFlag, currentM);
    }

    WaitSyncFlags();
    StoreProjectionToOutput(cTile, currentDst + colOffset, currentM);
}

AICORE inline void RunProjectionKernel(__gm__ bfloat16_t *xPtr, __gm__ bfloat16_t *weightPtr, __gm__ float *projPtr,
                                       __gm__ int32_t *expertIdsPtr,
                                       __gm__ int32_t *rowOffsetsPtr, __gm__ int32_t *tileMetaPtr)
{
    const uint32_t workIdx = get_block_idx();
    const uint32_t expertId = static_cast<uint32_t>(expertIdsPtr[workIdx]);
    const uint32_t rowOffset = static_cast<uint32_t>(rowOffsetsPtr[workIdx]);
    const int32_t tileMeta = tileMetaPtr[workIdx];
    const uint32_t colTile = static_cast<uint32_t>(tileMeta >> 16);
    const uint32_t currentM = static_cast<uint32_t>(tileMeta & 0xffff);
    RunProjectionKernel(xPtr, weightPtr, projPtr, expertId, rowOffset, colTile, currentM);
}

AICORE inline void DecodeProjectionWorkItem(__gm__ int32_t *groupOffsetsPtr, __gm__ int32_t *tileOffsetsPtr,
                                            uint32_t numExperts, uint32_t workIdx, uint32_t &expertId,
                                            uint32_t &rowOffset, uint32_t &colTile, uint32_t &currentM)
{
    const uint32_t tileIdx = workIdx / static_cast<uint32_t>(kMoeNumNTiles);
    colTile = workIdx % static_cast<uint32_t>(kMoeNumNTiles);

    // `tileOffsets` is a monotonic prefix-sum array of length E+1. Binary search
    // avoids the per-block linear scan across all experts on the compact split path.
    uint32_t left = 0;
    uint32_t right = numExperts;
    while (left + 1 < right) {
        const uint32_t mid = left + (right - left) / 2;
        if (tileIdx < static_cast<uint32_t>(tileOffsetsPtr[mid])) {
            right = mid;
        } else {
            left = mid;
        }
    }
    expertId = left;

    const uint32_t tileIdxInExpert = tileIdx - static_cast<uint32_t>(tileOffsetsPtr[expertId]);
    const uint32_t expertStart = static_cast<uint32_t>(groupOffsetsPtr[expertId]);
    const uint32_t expertEnd = static_cast<uint32_t>(groupOffsetsPtr[expertId + 1]);
    rowOffset = expertStart + tileIdxInExpert * static_cast<uint32_t>(kMoeBaseM);
    if (rowOffset >= expertEnd) {
        currentM = 0;
        return;
    }
    const uint32_t remainingRows = expertEnd - rowOffset;
    currentM = remainingRows < static_cast<uint32_t>(kMoeBaseM) ? remainingRows : static_cast<uint32_t>(kMoeBaseM);
}

AICORE inline void RunProjectionKernelCompact(__gm__ bfloat16_t *xPtr, __gm__ bfloat16_t *weightPtr,
                                              __gm__ float *projPtr, __gm__ int32_t *groupOffsetsPtr,
                                              __gm__ int32_t *tileOffsetsPtr, uint32_t numExperts)
{
    const uint32_t workIdx = get_block_idx();
    uint32_t expertId = 0;
    uint32_t rowOffset = 0;
    uint32_t colTile = 0;
    uint32_t currentM = 0;
    DecodeProjectionWorkItem(groupOffsetsPtr, tileOffsetsPtr, numExperts, workIdx, expertId, rowOffset, colTile,
                             currentM);
    RunProjectionKernel(xPtr, weightPtr, projPtr, expertId, rowOffset, colTile, currentM);
}

template <typename OutType>
AICORE inline void RunProjectionKernelStrided(__gm__ bfloat16_t *xPtr, __gm__ bfloat16_t *weightPtr,
                                              __gm__ OutType *projPtr, uint32_t outputWidth,
                                              uint32_t baseColOffset, uint32_t expertId, uint32_t rowOffset,
                                              uint32_t colTile, uint32_t currentM)
{
    if (currentM == 0) {
        return;
    }

    if (currentM == static_cast<uint32_t>(kMoeBaseM)) {
        __gm__ bfloat16_t *currentSrc0 = xPtr + rowOffset * kMoeHiddenSize;
        __gm__ OutType *currentDst =
            projPtr + static_cast<std::size_t>(rowOffset) * outputWidth + static_cast<std::size_t>(baseColOffset);
        const uint32_t colOffset = colTile * static_cast<uint32_t>(kMoeBaseN);
        __gm__ bfloat16_t *currentSrc1 = weightPtr + (expertId * kMoeInterSize + colOffset) * kMoeHiddenSize;

        ProjectionFullTileMatA aMatTile[kBufferNum];
        ProjectionTileMatB bMatTile[kBufferNum];
        ProjectionFullLeftTile aTile[kBufferNum];
        ProjectionFullRightTile bTile[kBufferNum];
        ProjectionFullResTile cTile;

        InitBuffers(aMatTile, bMatTile, aTile, bTile, cTile);
        InitSyncFlags();

        uint8_t mte2DBFlag = 0;
        uint8_t mte1DBFlag = 0;
        constexpr uint32_t kLoop = kMoeHiddenSize / kMoeBaseK;

        for (uint32_t kIter = 0; kIter < kLoop; ++kIter) {
            ProcessKIterationFullM(kIter, currentSrc0, currentSrc1, aMatTile, bMatTile, aTile, bTile, cTile,
                                   mte2DBFlag, mte1DBFlag);
        }

        WaitSyncFlags();
        StoreProjectionToOutputStrided(cTile, currentDst + colOffset, static_cast<uint32_t>(kMoeBaseM), outputWidth);
        return;
    }

    __gm__ bfloat16_t *currentSrc0 = xPtr + rowOffset * kMoeHiddenSize;
    __gm__ OutType *currentDst =
        projPtr + static_cast<std::size_t>(rowOffset) * outputWidth + static_cast<std::size_t>(baseColOffset);
    const uint32_t colOffset = colTile * static_cast<uint32_t>(kMoeBaseN);
    __gm__ bfloat16_t *currentSrc1 = weightPtr + (expertId * kMoeInterSize + colOffset) * kMoeHiddenSize;

    ProjectionTileMatA aMatTile[kBufferNum] = {ProjectionTileMatA(currentM, kMoeBaseK * kMoeStepKa),
                                               ProjectionTileMatA(currentM, kMoeBaseK * kMoeStepKa)};
    ProjectionTileMatB bMatTile[kBufferNum];
    ProjectionLeftTile aTile[kBufferNum] = {ProjectionLeftTile(currentM, kMoeBaseK),
                                            ProjectionLeftTile(currentM, kMoeBaseK)};
    ProjectionRightTile bTile[kBufferNum] = {ProjectionRightTile(kMoeBaseK, kMoeBaseN),
                                             ProjectionRightTile(kMoeBaseK, kMoeBaseN)};
    ProjectionResTile cTile(currentM, kMoeBaseN);

    InitBuffers(aMatTile, bMatTile, aTile, bTile, cTile);
    InitSyncFlags();

    uint8_t mte2DBFlag = 0;
    uint8_t mte1DBFlag = 0;
    constexpr uint32_t kLoop = kMoeHiddenSize / kMoeBaseK;

    for (uint32_t kIter = 0; kIter < kLoop; ++kIter) {
        ProcessKIteration(kIter, currentSrc0, currentSrc1, aMatTile, bMatTile, aTile, bTile, cTile, mte2DBFlag,
                          mte1DBFlag, currentM);
    }

    WaitSyncFlags();
    StoreProjectionToOutputStrided(cTile, currentDst + colOffset, currentM, outputWidth);
}

template <typename OutType>
AICORE inline void RunProjectionKernelCompactStrided(__gm__ bfloat16_t *xPtr, __gm__ bfloat16_t *weightPtr,
                                                     __gm__ OutType *projPtr, __gm__ int32_t *groupOffsetsPtr,
                                                     __gm__ int32_t *tileOffsetsPtr, uint32_t numExperts,
                                                     uint32_t outputWidth, uint32_t baseColOffset)
{
    const uint32_t workIdx = get_block_idx();
    uint32_t expertId = 0;
    uint32_t rowOffset = 0;
    uint32_t colTile = 0;
    uint32_t currentM = 0;
    DecodeProjectionWorkItem(groupOffsetsPtr, tileOffsetsPtr, numExperts, workIdx, expertId, rowOffset, colTile,
                             currentM);
    RunProjectionKernelStrided(xPtr, weightPtr, projPtr, outputWidth, baseColOffset, expertId, rowOffset, colTile,
                               currentM);
}

AICORE inline void ProcessKIterationCombinedNd(uint32_t kIter, __gm__ bfloat16_t *currentSrc0,
                                               __gm__ bfloat16_t *currentSrc1,
                                               ProjectionTileMatA aMatTile[kBufferNum],
                                               ProjectionTileMatBView bMatTile[kBufferNum],
                                               ProjectionLeftTile aTile[kBufferNum],
                                               ProjectionRightTile bTile[kBufferNum], ProjectionResTile &cTile,
                                               uint8_t &mte2DBFlag, uint8_t &mte1DBFlag, uint32_t currentM)
{
    using AValidShape = TileShape2D<bfloat16_t, -1, -1, Layout::ND>;
    using ABaseShape = BaseShape2D<bfloat16_t, kMoeBaseM, kMoeHiddenSize, Layout::ND>;
    using AGlobal = GlobalTensor<bfloat16_t, AValidShape, ABaseShape, Layout::ND>;

    using BValidShape = TileShape2D<bfloat16_t, kMoeBaseK * kMoeStepKb, kMoeBaseN, Layout::ND>;
    using BBaseShape =
        Stride<kMoeHiddenSize * kMoeCombinedInterSize, kMoeHiddenSize * kMoeCombinedInterSize,
               kMoeHiddenSize * kMoeCombinedInterSize, kMoeCombinedInterSize, 1>;
    using BGlobal = GlobalTensor<bfloat16_t, BValidShape, BBaseShape, Layout::ND>;

    const uint32_t kModStepKa = kIter % kMoeStepKa;

    if (kModStepKa == 0) {
        AGlobal gmA(currentSrc0 + kIter * kMoeBaseK, AValidShape(currentM, kMoeBaseK * kMoeStepKa));
        BGlobal gmB(currentSrc1 + kIter * kMoeBaseK * kMoeCombinedInterSize);

        WaitFlag<PIPE_MTE1, PIPE_MTE2>(mte2DBFlag);
        TLOAD(aMatTile[mte2DBFlag], gmA);
        SetFlag<PIPE_MTE2, PIPE_MTE1>(0);
        TLOAD(bMatTile[mte2DBFlag], gmB);
        SetFlag<PIPE_MTE2, PIPE_MTE1>(1);
        mte2DBFlag = (mte2DBFlag == 0) ? 1 : 0;
    }

    const uint32_t currMte2Idx = (mte2DBFlag == 0) ? 1 : 0;

    WaitFlag<PIPE_M, PIPE_MTE1>(mte1DBFlag);

    if (kModStepKa == 0) {
        WaitFlag<PIPE_MTE2, PIPE_MTE1>(0);
    }
    TEXTRACT(aTile[mte1DBFlag], aMatTile[currMte2Idx], 0, kModStepKa * kMoeBaseK);

    if (kModStepKa == 0) {
        WaitFlag<PIPE_MTE2, PIPE_MTE1>(1);
    }
    TEXTRACT(bTile[mte1DBFlag], bMatTile[currMte2Idx], (kIter % kMoeStepKb) * kMoeBaseK, 0);

    if ((kIter + 1) % kMoeStepKa == 0) {
        SetFlag<PIPE_MTE1, PIPE_MTE2>(currMte2Idx);
    }

    SetFlag<PIPE_MTE1, PIPE_M>(mte1DBFlag);
    WaitFlag<PIPE_MTE1, PIPE_M>(mte1DBFlag);
    MatmulAcc(cTile, aTile[mte1DBFlag], bTile[mte1DBFlag], kIter);
    SetFlag<PIPE_M, PIPE_MTE1>(mte1DBFlag);
    mte1DBFlag = (mte1DBFlag == 0) ? 1 : 0;
}

AICORE inline void ProcessKIterationCombinedDn(uint32_t kIter, __gm__ bfloat16_t *currentSrc0,
                                               __gm__ bfloat16_t *currentSrc1,
                                               ProjectionTileMatA aMatTile[kBufferNum],
                                               ProjectionTileMatB bMatTile[kBufferNum],
                                               ProjectionLeftTile aTile[kBufferNum],
                                               ProjectionRightTile bTile[kBufferNum], ProjectionResTile &cTile,
                                               uint8_t &mte2DBFlag, uint8_t &mte1DBFlag, uint32_t currentM)
{
    using AValidShape = TileShape2D<bfloat16_t, -1, -1, Layout::ND>;
    using ABaseShape = BaseShape2D<bfloat16_t, kMoeBaseM, kMoeHiddenSize, Layout::ND>;
    using AGlobal = GlobalTensor<bfloat16_t, AValidShape, ABaseShape, Layout::ND>;

    using BValidShape = TileShape2D<bfloat16_t, kMoeBaseK * kMoeStepKb, kMoeBaseN, Layout::DN>;
    using BBaseShape = BaseShape2D<bfloat16_t, kMoeHiddenSize, kMoeCombinedInterSize, Layout::DN>;
    using BGlobal = GlobalTensor<bfloat16_t, BValidShape, BBaseShape, Layout::DN>;

    const uint32_t kModStepKa = kIter % kMoeStepKa;

    if (kModStepKa == 0) {
        AGlobal gmA(currentSrc0 + kIter * kMoeBaseK, AValidShape(currentM, kMoeBaseK * kMoeStepKa));
        BGlobal gmB(currentSrc1 + kIter * kMoeBaseK);

        WaitFlag<PIPE_MTE1, PIPE_MTE2>(mte2DBFlag);
        TLOAD(aMatTile[mte2DBFlag], gmA);
        SetFlag<PIPE_MTE2, PIPE_MTE1>(0);
        TLOAD(bMatTile[mte2DBFlag], gmB);
        SetFlag<PIPE_MTE2, PIPE_MTE1>(1);
        mte2DBFlag = (mte2DBFlag == 0) ? 1 : 0;
    }

    const uint32_t currMte2Idx = (mte2DBFlag == 0) ? 1 : 0;

    WaitFlag<PIPE_M, PIPE_MTE1>(mte1DBFlag);

    if (kModStepKa == 0) {
        WaitFlag<PIPE_MTE2, PIPE_MTE1>(0);
    }
    TEXTRACT(aTile[mte1DBFlag], aMatTile[currMte2Idx], 0, kModStepKa * kMoeBaseK);

    if (kModStepKa == 0) {
        WaitFlag<PIPE_MTE2, PIPE_MTE1>(1);
    }
    TEXTRACT(bTile[mte1DBFlag], bMatTile[currMte2Idx], (kIter % kMoeStepKb) * kMoeBaseK, 0);

    if ((kIter + 1) % kMoeStepKa == 0) {
        SetFlag<PIPE_MTE1, PIPE_MTE2>(currMte2Idx);
    }

    SetFlag<PIPE_MTE1, PIPE_M>(mte1DBFlag);
    WaitFlag<PIPE_MTE1, PIPE_M>(mte1DBFlag);
    MatmulAcc(cTile, aTile[mte1DBFlag], bTile[mte1DBFlag], kIter);
    SetFlag<PIPE_M, PIPE_MTE1>(mte1DBFlag);
    mte1DBFlag = (mte1DBFlag == 0) ? 1 : 0;
}

template <typename OutType>
AICORE inline void RunProjectionKernelStridedCombinedNd(__gm__ bfloat16_t *xPtr, __gm__ bfloat16_t *weightPtr,
                                                        __gm__ OutType *projPtr, uint32_t outputWidth,
                                                        uint32_t outputBaseColOffset, uint32_t weightBaseColOffset,
                                                        uint32_t expertId, uint32_t rowOffset, uint32_t colTile,
                                                        uint32_t currentM)
{
    if (currentM == 0) {
        return;
    }

    __gm__ bfloat16_t *currentSrc0 = xPtr + rowOffset * kMoeHiddenSize;
    __gm__ OutType *currentDst =
        projPtr + static_cast<std::size_t>(rowOffset) * outputWidth + static_cast<std::size_t>(outputBaseColOffset);
    const uint32_t colOffset = colTile * static_cast<uint32_t>(kMoeBaseN);
    __gm__ bfloat16_t *currentSrc1 =
        weightPtr + expertId * kMoeHiddenSize * kMoeCombinedInterSize + weightBaseColOffset + colOffset;

    ProjectionTileMatA aMatTile[kBufferNum] = {ProjectionTileMatA(currentM, kMoeBaseK * kMoeStepKa),
                                               ProjectionTileMatA(currentM, kMoeBaseK * kMoeStepKa)};
    ProjectionTileMatBView bMatTile[kBufferNum];
    ProjectionLeftTile aTile[kBufferNum] = {ProjectionLeftTile(currentM, kMoeBaseK),
                                            ProjectionLeftTile(currentM, kMoeBaseK)};
    ProjectionRightTile bTile[kBufferNum] = {ProjectionRightTile(kMoeBaseK, kMoeBaseN),
                                             ProjectionRightTile(kMoeBaseK, kMoeBaseN)};
    ProjectionResTile cTile(currentM, kMoeBaseN);

    InitBuffers(aMatTile, bMatTile, aTile, bTile, cTile);
    InitSyncFlags();

    uint8_t mte2DBFlag = 0;
    uint8_t mte1DBFlag = 0;
    constexpr uint32_t kLoop = kMoeHiddenSize / kMoeBaseK;

    for (uint32_t kIter = 0; kIter < kLoop; ++kIter) {
        ProcessKIterationCombinedNd(kIter, currentSrc0, currentSrc1, aMatTile, bMatTile, aTile, bTile, cTile,
                                    mte2DBFlag, mte1DBFlag, currentM);
    }

    WaitSyncFlags();
    StoreProjectionToOutputStrided(cTile, currentDst + colOffset, currentM, outputWidth);
}

template <typename OutType>
AICORE inline void RunProjectionKernelCompactStridedCombinedNd(__gm__ bfloat16_t *xPtr, __gm__ bfloat16_t *weightPtr,
                                                               __gm__ OutType *projPtr,
                                                               __gm__ int32_t *groupOffsetsPtr,
                                                               __gm__ int32_t *tileOffsetsPtr, uint32_t numExperts,
                                                               uint32_t outputWidth,
                                                               uint32_t outputBaseColOffset,
                                                               uint32_t weightBaseColOffset)
{
    const uint32_t workIdx = get_block_idx();
    uint32_t expertId = 0;
    uint32_t rowOffset = 0;
    uint32_t colTile = 0;
    uint32_t currentM = 0;
    DecodeProjectionWorkItem(groupOffsetsPtr, tileOffsetsPtr, numExperts, workIdx, expertId, rowOffset, colTile,
                             currentM);
    RunProjectionKernelStridedCombinedNd(xPtr, weightPtr, projPtr, outputWidth, outputBaseColOffset,
                                         weightBaseColOffset, expertId, rowOffset, colTile, currentM);
}

template <typename OutType>
AICORE inline void RunProjectionKernelStridedCombinedDn(__gm__ bfloat16_t *xPtr, __gm__ bfloat16_t *weightPtr,
                                                        __gm__ OutType *projPtr, uint32_t outputWidth,
                                                        uint32_t outputBaseColOffset, uint32_t weightBaseColOffset,
                                                        uint32_t expertId, uint32_t rowOffset, uint32_t colTile,
                                                        uint32_t currentM)
{
    if (currentM == 0) {
        return;
    }

    __gm__ bfloat16_t *currentSrc0 = xPtr + rowOffset * kMoeHiddenSize;
    __gm__ OutType *currentDst =
        projPtr + static_cast<std::size_t>(rowOffset) * outputWidth + static_cast<std::size_t>(outputBaseColOffset);
    const uint32_t colOffset = colTile * static_cast<uint32_t>(kMoeBaseN);
    __gm__ bfloat16_t *currentSrc1 =
        weightPtr + (expertId * kMoeCombinedInterSize + weightBaseColOffset + colOffset) * kMoeHiddenSize;

    ProjectionTileMatA aMatTile[kBufferNum] = {ProjectionTileMatA(currentM, kMoeBaseK * kMoeStepKa),
                                               ProjectionTileMatA(currentM, kMoeBaseK * kMoeStepKa)};
    ProjectionTileMatB bMatTile[kBufferNum];
    ProjectionLeftTile aTile[kBufferNum] = {ProjectionLeftTile(currentM, kMoeBaseK),
                                            ProjectionLeftTile(currentM, kMoeBaseK)};
    ProjectionRightTile bTile[kBufferNum] = {ProjectionRightTile(kMoeBaseK, kMoeBaseN),
                                             ProjectionRightTile(kMoeBaseK, kMoeBaseN)};
    ProjectionResTile cTile(currentM, kMoeBaseN);

    InitBuffers(aMatTile, bMatTile, aTile, bTile, cTile);
    InitSyncFlags();

    uint8_t mte2DBFlag = 0;
    uint8_t mte1DBFlag = 0;
    constexpr uint32_t kLoop = kMoeHiddenSize / kMoeBaseK;

    for (uint32_t kIter = 0; kIter < kLoop; ++kIter) {
        ProcessKIterationCombinedDn(kIter, currentSrc0, currentSrc1, aMatTile, bMatTile, aTile, bTile, cTile,
                                    mte2DBFlag, mte1DBFlag, currentM);
    }

    WaitSyncFlags();
    StoreProjectionToOutputStrided(cTile, currentDst + colOffset, currentM, outputWidth);
}

template <typename OutType>
AICORE inline void RunProjectionKernelCompactStridedCombinedDn(__gm__ bfloat16_t *xPtr, __gm__ bfloat16_t *weightPtr,
                                                               __gm__ OutType *projPtr,
                                                               __gm__ int32_t *groupOffsetsPtr,
                                                               __gm__ int32_t *tileOffsetsPtr, uint32_t numExperts,
                                                               uint32_t outputWidth,
                                                               uint32_t outputBaseColOffset,
                                                               uint32_t weightBaseColOffset)
{
    const uint32_t workIdx = get_block_idx();
    uint32_t expertId = 0;
    uint32_t rowOffset = 0;
    uint32_t colTile = 0;
    uint32_t currentM = 0;
    DecodeProjectionWorkItem(groupOffsetsPtr, tileOffsetsPtr, numExperts, workIdx, expertId, rowOffset, colTile,
                             currentM);
    RunProjectionKernelStridedCombinedDn(xPtr, weightPtr, projPtr, outputWidth, outputBaseColOffset,
                                         weightBaseColOffset, expertId, rowOffset, colTile, currentM);
}

} // namespace moe_grouped_ffn_projection

#endif // PTO_MOE_GROUPED_FFN_PROJECTION_COMMON_H
