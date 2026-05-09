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

#ifndef GEMM_PIPELINE_COMMON_H
#define GEMM_PIPELINE_COMMON_H

#include <pto/pto-inst.hpp>

namespace pto::manual::common {

template <typename OutTile, typename LeftTile, typename RightTile>
AICORE inline void MatmulAcc(OutTile cTile, LeftTile aTile, RightTile bTile, uint32_t k)
{
    if (k != 0) {
        TMATMUL_ACC(cTile, cTile, aTile, bTile);
        return;
    }
    TMATMUL(cTile, aTile, bTile);
}

template <pipe_t srcPipe, pipe_t dstPipe>
AICORE inline void SetFlag(uint32_t id)
{
    const auto eventId = static_cast<event_t>(id);
    set_flag(srcPipe, dstPipe, eventId);
}

template <pipe_t srcPipe, pipe_t dstPipe>
AICORE inline void WaitFlag(uint32_t id)
{
    const auto eventId = static_cast<event_t>(id);
    wait_flag(srcPipe, dstPipe, eventId);
}

template <typename TileMatA, typename TileMatB, typename GlobalDataSrcA, typename GlobalDataSrcB, typename SrcA,
          typename SrcB, uint32_t baseM, uint32_t baseK, uint32_t baseN, uint32_t stepKa, uint32_t stepKb,
          uint32_t singleCoreK, uint32_t BufferNum>
AICORE inline void LoadPanelsIfNeeded(uint32_t kIter, uint32_t i, uint32_t j, __gm__ SrcA *currentSrc0,
                                      __gm__ SrcB *currentSrc1, TileMatA (&aMatTile)[BufferNum],
                                      TileMatB (&bMatTile)[BufferNum], uint8_t &mte2DBFlag)
{
    const uint32_t kModstepKa = kIter % stepKa;
    if (kModstepKa != 0) {
        return;
    }

    GlobalDataSrcA gmA(currentSrc0 + i * singleCoreK * baseM + kIter * baseK);
    GlobalDataSrcB gmB(currentSrc1 + j * singleCoreK * baseN + kIter * baseK);

    WaitFlag<PIPE_MTE1, PIPE_MTE2>(mte2DBFlag);
    TLOAD(aMatTile[mte2DBFlag], gmA);
    SetFlag<PIPE_MTE2, PIPE_MTE1>(0);
    TLOAD(bMatTile[mte2DBFlag], gmB);
    SetFlag<PIPE_MTE2, PIPE_MTE1>(1);
    mte2DBFlag = (mte2DBFlag == 0) ? 1 : 0;
}

template <typename LeftTile, typename RightTile, typename ResTile, typename TileMatA, typename TileMatB, uint32_t baseK,
          uint32_t stepKa, uint32_t stepKb, uint32_t BufferNum>
AICORE inline void ExtractAndMatmul(uint32_t kIter, uint8_t currMte2Idx, uint8_t &mte1DBFlag,
                                    TileMatA (&aMatTile)[BufferNum], TileMatB (&bMatTile)[BufferNum],
                                    LeftTile (&aTile)[BufferNum], RightTile (&bTile)[BufferNum], ResTile &cTile)
{
    const uint32_t kModstepKa = kIter % stepKa;
    WaitFlag<PIPE_M, PIPE_MTE1>(mte1DBFlag);

    if (kModstepKa == 0) {
        WaitFlag<PIPE_MTE2, PIPE_MTE1>(0);
    }
    TEXTRACT(aTile[mte1DBFlag], aMatTile[currMte2Idx], 0, kModstepKa * baseK);

    if (kModstepKa == 0) {
        WaitFlag<PIPE_MTE2, PIPE_MTE1>(1);
    }
    TEXTRACT(bTile[mte1DBFlag], bMatTile[currMte2Idx], (kIter % stepKb) * baseK, 0);

    if ((kIter + 1) % stepKa == 0) {
        SetFlag<PIPE_MTE1, PIPE_MTE2>(currMte2Idx);
    }

    SetFlag<PIPE_MTE1, PIPE_M>(mte1DBFlag);
    WaitFlag<PIPE_MTE1, PIPE_M>(mte1DBFlag);

    auto &currATile = aTile[mte1DBFlag];
    auto &currBTile = bTile[mte1DBFlag];
    MatmulAcc(cTile, currATile, currBTile, kIter);

    SetFlag<PIPE_M, PIPE_MTE1>(mte1DBFlag);
    mte1DBFlag = (mte1DBFlag == 0) ? 1 : 0;
}

} // namespace pto::manual::common

#endif
