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
using ProjectionLeftTile = TileLeftCompact<bfloat16_t, kMoeBaseM, kMoeBaseK, -1, -1>;
using ProjectionRightTile = TileRightCompact<bfloat16_t, kMoeBaseK, kMoeBaseN, -1, -1>;
using ProjectionResTile = TileAcc<float, kMoeBaseM, kMoeBaseN, -1, -1>;

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

AICORE inline void StoreProjectionToOutput(ProjectionResTile &projTile, __gm__ float *currentDst, uint32_t rowCount)
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

} // namespace moe_grouped_ffn_projection

#endif // PTO_MOE_GROUPED_FFN_PROJECTION_COMMON_H
