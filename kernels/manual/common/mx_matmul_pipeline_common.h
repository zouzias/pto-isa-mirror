/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under
the terms and conditions of CANN Open Software License Agreement Version 2.0
(the "License"). Please refer to the License for details. You may not use this
file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON AN "AS
IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
PARTICULAR PURPOSE. See LICENSE in the root of the software repository for the
full text of the License.
*/

#ifndef MX_MATMUL_PIPELINE_COMMON_H
#define MX_MATMUL_PIPELINE_COMMON_H

#include <pto/pto-inst.hpp>

namespace pto::manual::common {

template <typename OutTile, typename LeftTile, typename RightTile, typename LeftScaleTile, typename RightScaleTile>
AICORE inline void MxMatmulAcc(OutTile cTile, LeftTile aTile, RightTile bTile, LeftScaleTile aScaleTile,
                               RightScaleTile bScaleTile, uint32_t k)
{
    if (k == 0) {
        TMATMUL_MX(cTile, aTile, aScaleTile, bTile, bScaleTile);
    } else {
        TMATMUL_MX(cTile, cTile, aTile, aScaleTile, bTile, bScaleTile);
    }
}

template <pipe_t srcPipe, pipe_t dstPipe>
AICORE inline void SetPipeFlag(uint32_t id)
{
    set_flag(srcPipe, dstPipe, static_cast<event_t>(id));
}

template <pipe_t srcPipe, pipe_t dstPipe>
AICORE inline void WaitPipeFlag(uint32_t id)
{
    wait_flag(srcPipe, dstPipe, static_cast<event_t>(id));
}

template <typename T, typename U, typename X, uint32_t baseM, uint32_t baseK, uint32_t baseN, uint32_t baseScaleK,
          uint32_t stepKa, uint32_t stepKb, uint32_t stepKscaleA, uint32_t stepKscaleB, typename TileMatA,
          typename TileMatB, typename TileScaleA, typename TileScaleB, typename LeftTile, typename RightTile,
          typename LeftScaleTile, typename RightScaleTile, typename ResTile, uint32_t BufferNum,
          uint32_t L0PingpongBytes, uint32_t ElemDivisor>
AICORE inline void InitMxBuffers(TileMatA (&aMatTile)[BufferNum], TileMatB (&bMatTile)[BufferNum],
                                 TileScaleA (&aScaleMatTile)[BufferNum], TileScaleB (&bScaleMatTile)[BufferNum],
                                 LeftTile (&aTile)[BufferNum], RightTile (&bTile)[BufferNum],
                                 LeftScaleTile (&aScaleTile)[BufferNum], RightScaleTile (&bScaleTile)[BufferNum],
                                 ResTile &cTile)
{
    TASSIGN(aMatTile[0], 0x0);
    TASSIGN(aMatTile[1], 0x0 + baseM * baseK * stepKa * sizeof(U) / ElemDivisor);
    TASSIGN(bMatTile[0], 0x0 + baseM * baseK * stepKa * BufferNum * sizeof(U) / ElemDivisor);
    TASSIGN(bMatTile[1], 0x0 + baseM * baseK * stepKa * BufferNum * sizeof(U) / ElemDivisor +
                             baseK * baseN * stepKb * sizeof(U) / ElemDivisor);

    constexpr uint32_t baseAddr = baseM * baseK * stepKa * BufferNum * sizeof(U) / ElemDivisor +
                                  baseK * baseN * stepKb * sizeof(U) / ElemDivisor * BufferNum;
    TASSIGN(aScaleMatTile[0], baseAddr);
    TASSIGN(aScaleMatTile[1], baseAddr + baseM * baseScaleK * stepKscaleA * sizeof(X));
    TASSIGN(bScaleMatTile[0], baseAddr + baseM * baseScaleK * stepKscaleA * BufferNum * sizeof(X));
    TASSIGN(bScaleMatTile[1], baseAddr + baseM * baseScaleK * stepKscaleA * BufferNum * sizeof(X) +
                                  baseScaleK * baseN * stepKscaleB * sizeof(X));

    TASSIGN(aTile[0], 0x0);
    TASSIGN(aTile[1], 0x0 + L0PingpongBytes);
    TASSIGN(bTile[0], 0x0);
    TASSIGN(bTile[1], 0x0 + L0PingpongBytes);
    TASSIGN(cTile, 0x0);

    TASSIGN(aScaleTile[0], GetScaleAddr(aTile[0].data()));
    TASSIGN(aScaleTile[1], GetScaleAddr(aTile[1].data()));
    TASSIGN(bScaleTile[0], GetScaleAddr(bTile[0].data()));
    TASSIGN(bScaleTile[1], GetScaleAddr(bTile[1].data()));
}

template <uint32_t baseK, uint32_t baseScaleK, uint32_t stepKa, uint32_t stepKb, uint32_t stepKscaleA,
          uint32_t stepKscaleB, typename TileMatA, typename TileMatB, typename TileScaleA, typename TileScaleB,
          typename LeftTile, typename RightTile, typename LeftScaleTile, typename RightScaleTile, typename ResTile,
          uint32_t BufferNum>
AICORE inline void MacroMxMatmul(uint32_t kIter, uint8_t currMte2Idx, uint8_t currMte2mxIdx, uint8_t mte1DBFlag,
                                 TileMatA (&aMatTile)[BufferNum], TileMatB (&bMatTile)[BufferNum],
                                 TileScaleA (&aScaleMatTile)[BufferNum], TileScaleB (&bScaleMatTile)[BufferNum],
                                 LeftTile (&aTile)[BufferNum], RightTile (&bTile)[BufferNum],
                                 LeftScaleTile (&aScaleTile)[BufferNum], RightScaleTile (&bScaleTile)[BufferNum],
                                 ResTile &cTile)
{
    const uint32_t kModstepKa = kIter % stepKa;
    WaitPipeFlag<PIPE_M, PIPE_MTE1>(mte1DBFlag);

    if (kModstepKa == 0) {
        WaitPipeFlag<PIPE_MTE2, PIPE_MTE1>(0);
    }
    TEXTRACT(aTile[mte1DBFlag], aMatTile[currMte2Idx], 0, kModstepKa * baseK);
    TEXTRACT(aScaleTile[mte1DBFlag], aScaleMatTile[currMte2mxIdx], 0, (kIter % stepKscaleA) * baseScaleK);

    if (kModstepKa == 0) {
        WaitPipeFlag<PIPE_MTE2, PIPE_MTE1>(1);
    }
    TEXTRACT(bTile[mte1DBFlag], bMatTile[currMte2Idx], (kIter % stepKb) * baseK, 0);
    TEXTRACT(bScaleTile[mte1DBFlag], bScaleMatTile[currMte2mxIdx], (kIter % stepKscaleB) * baseScaleK, 0);

    if ((kIter + 1) % stepKa == 0) {
        SetPipeFlag<PIPE_MTE1, PIPE_MTE2>(currMte2Idx);
    }

    SetPipeFlag<PIPE_MTE1, PIPE_M>(mte1DBFlag);
    WaitPipeFlag<PIPE_MTE1, PIPE_M>(mte1DBFlag);
    MxMatmulAcc(cTile, aTile[mte1DBFlag], bTile[mte1DBFlag], aScaleTile[mte1DBFlag], bScaleTile[mte1DBFlag], kIter);
    SetPipeFlag<PIPE_M, PIPE_MTE1>(mte1DBFlag);
}

} // namespace pto::manual::common

#endif
