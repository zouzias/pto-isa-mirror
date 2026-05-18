/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under
the terms and conditions of CANN Open Software License Agreement Version 2.0
(the "License"). Please refer to the License for details. You may not use this
file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON AN "AS
IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
PARTICULAR PURPOSE. See LICENSE in the root of the software repository for the
full text of the License.
*/

#include <pto/common/constants.hpp>
#include <pto/pto-inst.hpp>

#include "kernels/manual/common/gemm_pipeline_common.h"

using namespace pto;
using namespace pto::manual::common;

namespace {
constexpr uint32_t kBufferNum = 2;
constexpr uint32_t kL0PingPongBytes = 32 * 1024; // L0A/L0B ping-pong split (32 KiB per buffer)

template <typename T, typename U, typename S>
struct GemmCoreWindow {
    __gm__ T *dst = nullptr;
    __gm__ U *src0 = nullptr;
    __gm__ S *src1 = nullptr;
};

template <typename T>
struct GemmCoreOffsets {
    T src0;
    T src1;
    T dst;
};
} // namespace

template <typename T, typename U, typename S, int m, int k, int n, uint32_t singleCoreM, uint32_t singleCoreK,
          uint32_t singleCoreN>
AICORE inline GemmCoreOffsets<uint64_t> GetCoreOffsets()
{
    constexpr uint32_t mIter = m / singleCoreM;
    const uint32_t coreIdx = get_block_idx();
    const uint32_t mIterIdx = coreIdx % mIter;
    const uint32_t nIterIdx = coreIdx / mIter;
    return {mIterIdx * singleCoreM * k, nIterIdx * k * singleCoreN,
            mIterIdx * singleCoreM * n + nIterIdx * singleCoreN};
}

template <typename T, typename U, typename S, int m, int k, int n, uint32_t singleCoreM, uint32_t singleCoreK,
          uint32_t singleCoreN>
AICORE inline void InitGMOffsets(__gm__ U *&currentSrc0, __gm__ S *&currentSrc1, __gm__ T *&currentDst, __gm__ T *out,
                                 __gm__ U *src0, __gm__ S *src1)
{
    const auto offsets = GetCoreOffsets<T, U, S, m, k, n, singleCoreM, singleCoreK, singleCoreN>();
    currentSrc0 = src0 + offsets.src0;
    currentSrc1 = src1 + offsets.src1;
    currentDst = out + offsets.dst;
}

template <typename T, typename U, typename S, int m, int k, int n, uint32_t singleCoreM, uint32_t singleCoreK,
          uint32_t singleCoreN>
AICORE inline GemmCoreWindow<T, U, S> MakeCoreWindow(__gm__ T *out, __gm__ U *src0, __gm__ S *src1)
{
    const auto offsets = GetCoreOffsets<T, U, S, m, k, n, singleCoreM, singleCoreK, singleCoreN>();
    return {.dst = out + offsets.dst, .src0 = src0 + offsets.src0, .src1 = src1 + offsets.src1};
}

template <typename T, typename U, typename S, int m, int k, int n, uint32_t baseM, uint32_t baseK, uint32_t baseN,
          uint32_t stepKa, uint32_t stepKb, uint32_t singleCoreK, typename TileMatA, typename TileMatB,
          typename LeftTile, typename RightTile, typename ResTile>
AICORE inline void ProcessKIteration(uint32_t kIter, uint32_t i, uint32_t j, __gm__ U *currentSrc0,
                                     __gm__ S *currentSrc1, TileMatA aMatTile[kBufferNum],
                                     TileMatB bMatTile[kBufferNum], LeftTile aTile[kBufferNum],
                                     RightTile bTile[kBufferNum], ResTile &cTile, uint8_t &mte2DBFlag,
                                     uint8_t &mte1DBFlag)
{
    using NDValidShapeA = TileShape2D<U, baseM, baseK * stepKa, Layout::ND>;
    using NDsingleCoreShapeA = BaseShape2D<U, m, k, Layout::ND>;
    using GlobalDataSrcA = GlobalTensor<U, NDValidShapeA, NDsingleCoreShapeA, Layout::ND>;
    using NDValidShapeB = TileShape2D<U, baseK * stepKb, baseN, Layout::DN>;
    using NDsingleCoreShapeB = BaseShape2D<U, k, n, Layout::DN>;
    using GlobalDataSrcB = GlobalTensor<U, NDValidShapeB, NDsingleCoreShapeB, Layout::DN>;

    LoadPanelsIfNeeded<TileMatA, TileMatB, GlobalDataSrcA, GlobalDataSrcB, U, S, baseM, baseK, baseN, stepKa, stepKb,
                       singleCoreK, kBufferNum>(kIter, i, j, currentSrc0, currentSrc1, aMatTile, bMatTile, mte2DBFlag);

    const uint32_t currMte2Idx = (mte2DBFlag == 0) ? 1 : 0;
    ExtractAndMatmul<LeftTile, RightTile, ResTile, TileMatA, TileMatB, baseK, stepKa, stepKb, kBufferNum>(
        kIter, currMte2Idx, mte1DBFlag, aMatTile, bMatTile, aTile, bTile, cTile);
}

template <typename T, typename U, typename S, int m, int n, uint32_t baseM, uint32_t baseN, uint32_t singleCoreK,
          typename ResTile>
AICORE inline void StoreResult(ResTile &cTile, __gm__ T *currentDst, uint32_t i, uint32_t j)
{
    SetFlag<PIPE_M, PIPE_FIX>(0);
    WaitFlag<PIPE_M, PIPE_FIX>(0);

    using NDValidShapeC = TileShape2D<T, baseM, baseN, Layout::ND>;
    using NDWholeShapeC = BaseShape2D<T, m, n, Layout::ND>;
    using GlobalDataOut = GlobalTensor<T, NDValidShapeC, NDWholeShapeC, Layout::ND>;

    GlobalDataOut dstGlobal(currentDst + i * baseM * n + j * baseN);
    TSTORE(dstGlobal, cTile);

    SetFlag<PIPE_FIX, PIPE_M>(0);
    WaitFlag<PIPE_FIX, PIPE_M>(0);
}

template <typename TileMatA, typename TileMatB, typename LeftTile, typename RightTile, typename ResTile, typename U,
          typename S, uint32_t baseM, uint32_t baseK, uint32_t baseN, uint32_t stepKa, uint32_t stepKb>
AICORE inline void InitCoreTiles(TileMatA aMatTile[kBufferNum], TileMatB bMatTile[kBufferNum],
                                 LeftTile aTile[kBufferNum], RightTile bTile[kBufferNum], ResTile &cTile)
{
    TASSIGN(aMatTile[0], 0x0);
    TASSIGN(aMatTile[1], 0x0 + baseM * baseK * stepKa * sizeof(U));
    TASSIGN(bMatTile[0], 0x0 + baseM * baseK * stepKa * kBufferNum * sizeof(U));
    TASSIGN(bMatTile[1], 0x0 + baseM * baseK * stepKa * kBufferNum * sizeof(U) + baseK * baseN * stepKb * sizeof(U));
    TASSIGN(aTile[0], 0x0);
    TASSIGN(aTile[1], 0x0 + kL0PingPongBytes);
    TASSIGN(bTile[0], 0x0);
    TASSIGN(bTile[1], 0x0 + kL0PingPongBytes);
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

template <typename T, typename U, typename S, typename B, uint32_t blockDim, int m, int k, int n, int validM,
          int validK, int validN, uint32_t singleCoreM, uint32_t singleCoreK, uint32_t singleCoreN, uint32_t baseM,
          uint32_t baseK, uint32_t baseN, uint32_t stepM, uint32_t stepKa, uint32_t stepKb, uint32_t stepN>
AICORE inline void RunGemmLoops(__gm__ T *currentDst, __gm__ U *currentSrc0, __gm__ S *currentSrc1)
{
    using TileMatA =
        Tile<TileType::Mat, U, baseM, baseK * stepKa, BLayout::ColMajor, baseM, baseK * stepKa, SLayout::RowMajor>;
    using TileMatB =
        Tile<TileType::Mat, S, baseK * stepKb, baseN, BLayout::RowMajor, baseK * stepKb, baseN, SLayout::ColMajor>;
    using LeftTile = TileLeft<U, baseM, baseK, baseM, baseK>;
    using RightTile = TileRight<S, baseK, baseN, baseK, baseN>;
    using ResTile = TileAcc<T, baseM, baseN, baseM, baseN>;

    TileMatA aMatTile[kBufferNum];
    TileMatB bMatTile[kBufferNum];
    LeftTile aTile[kBufferNum];
    RightTile bTile[kBufferNum];
    ResTile cTile;

    InitCoreTiles<TileMatA, TileMatB, LeftTile, RightTile, ResTile, U, S, baseM, baseK, baseN, stepKa, stepKb>(
        aMatTile, bMatTile, aTile, bTile, cTile);

    constexpr uint32_t mLoop = singleCoreM / baseM;
    constexpr uint32_t nLoop = singleCoreN / baseN;
    constexpr uint32_t kLoop = singleCoreK / baseK;
    uint8_t mte2DBFlag = 0;
    uint8_t mte1DBFlag = 0;

    InitSyncFlags();
    for (uint32_t i = 0; i < mLoop; ++i) {
        for (uint32_t j = 0; j < nLoop; ++j) {
            for (uint32_t kIter = 0; kIter < kLoop; ++kIter) {
                ProcessKIteration<T, U, S, m, k, n, baseM, baseK, baseN, stepKa, stepKb, singleCoreK, TileMatA,
                                  TileMatB, LeftTile, RightTile, ResTile>(kIter, i, j, currentSrc0, currentSrc1,
                                                                          aMatTile, bMatTile, aTile, bTile, cTile,
                                                                          mte2DBFlag, mte1DBFlag);
            }
            StoreResult<T, U, S, m, n, baseM, baseN, singleCoreK, ResTile>(cTile, currentDst, i, j);
        }
    }
    WaitSyncFlags();
}

template <typename T, typename U, typename S, typename B, uint32_t blockDim, int m, int k, int n, int validM,
          int validK, int validN, uint32_t singleCoreM, uint32_t singleCoreK, uint32_t singleCoreN, uint32_t baseM,
          uint32_t baseK, uint32_t baseN, uint32_t stepM, uint32_t stepKa, uint32_t stepKb, uint32_t stepN>
AICORE inline void RunGemmE2E(__gm__ T *out, __gm__ U *src0, __gm__ S *src1)
{
#if (__CHECK_FEATURE_AT_PRECOMPILE) || (__CCE_AICORE__ == 220 && defined(__DAV_C220_CUBE__))
    const auto coreWindow = MakeCoreWindow<T, U, S, m, k, n, singleCoreM, singleCoreK, singleCoreN>(out, src0, src1);
    RunGemmLoops<T, U, S, B, blockDim, m, k, n, validM, validK, validN, singleCoreM, singleCoreK, singleCoreN, baseM,
                 baseK, baseN, stepM, stepKa, stepKb, stepN>(coreWindow.dst, coreWindow.src0, coreWindow.src1);
#else
    (void)out;
    (void)src0;
    (void)src1;
#endif
}
