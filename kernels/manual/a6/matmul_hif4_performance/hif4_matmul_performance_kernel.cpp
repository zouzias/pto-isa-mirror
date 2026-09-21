/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// --------------------------------------------------------------------------------
// hif4_matmul_performance_kernel.cpp
//
// A6 (dav-9201) HiF4 GEMM demo. Mirrors kernels/manual/a5/matmul_mxfp4_performance
// but uses the HiFloat4 data path:
//
//   GM(BF16) --TLOAD--> L1 --TEXTRACT--> L0A/L0B + L0AMX/L0BMX --TMATMUL_MX--> L0C --TSTORE--> GM(BF16)
//
// Differences from the A5 MXFP4 example:
//   * data dtype  : float4_e2m1x2_t -> hifloat4x2_t  (both 4-bit packed, 0.5 B/elem)
//   * scale dtype : float8_e8m0_t   -> uint8_t       (three-level Ea/Eb/Ec, 4 B / 64-group)
//   * scale group : 32 -> 64
//   * scale GM    : MX_A_ND/MX_B_DN (plain 2D) -> HIF4_A_ZZ/HIF4_B_NN (pre-fractalized [16,4] cell)
//
// Because the HiF4 scale is stored pre-fractalized in GM and TLoad consumes it via
// contiguous ZZ2ZZ/NN2NN bursts, the scale panel for a base block is loaded FULL-K
// once (it cannot be K-sliced at TLOAD time). The per-K slice is produced by TEXTRACT
// (L1 -> L0AMX/L0BMX), which does support column slicing.
// --------------------------------------------------------------------------------

#include <pto/common/constants.hpp>
#include <pto/common/debug.h>
#include <pto/pto-inst.hpp>

using namespace pto;

constexpr uint32_t BUFFER_NUM = 2;              // double-buffered data tiles
constexpr uint32_t HIF4_SCALE_GROUP = 64;       // HiF4 groups over 64 elements
constexpr uint32_t HIF4_COL_BYTES = 4;          // 4 bytes per 64-group (Ea/Eb/Ec packed cell width)
constexpr uint32_t L0_PINGPONG_BYTES = 32 * 1024;

// Event ids. 0/1 are the per-K data double-buffer flags; 2/3 are the per-base-block
// full-K scale flags.
constexpr uint32_t EV_DATA_A = 0;       // MTE2 -> MTE1: A data L1 buffer loaded
constexpr uint32_t EV_DATA_B = 1;       // MTE2 -> MTE1: B data L1 buffer loaded
constexpr uint32_t EV_SCALE_LOADED = 2; // MTE2 -> MTE1: full-K scale loaded into L1
constexpr uint32_t EV_SCALE_FREE = 3;   // MTE1 -> MTE2: scale L1 buffer consumed

template <typename OutTile, typename LeftTile, typename RightTile, typename LeftScaleTile, typename RightScaleTile>
AICORE inline void MatmulAcc(
    OutTile cTile, LeftTile aTile, RightTile bTile, LeftScaleTile aScaleTile, RightScaleTile bScaleTile, uint32_t k)
{
    if (k == 0) {
        TMATMUL_MX(cTile, aTile, aScaleTile, bTile, bScaleTile);
    } else {
        TMATMUL_MX(cTile, cTile, aTile, aScaleTile, bTile, bScaleTile);
    }
}

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

// Work partition (SPMD-style): core (mIterIdx, nIterIdx) owns C tile
// [rowStart, rowStart+singleCoreM) x [colStart, colStart+singleCoreN).
template <
    typename T, typename U, typename X, int m, int k, int n, uint32_t singleCoreM, uint32_t singleCoreK,
    uint32_t singleCoreN>
AICORE inline void InitGMOffsets(
    __gm__ U*& currentSrc0, __gm__ U*& currentSrc1, __gm__ X*& currentSrc2, __gm__ X*& currentSrc3,
    __gm__ T*& currentDst, __gm__ T* out, __gm__ U* src0, __gm__ U* src1, __gm__ X* src2, __gm__ X* src3)
{
    uint32_t mIter = (m + singleCoreM - 1) / singleCoreM;
    uint32_t nIter = (n + singleCoreN - 1) / singleCoreN;
    uint32_t mIterIdx = get_block_idx() % mIter;
    uint32_t nIterIdx = get_block_idx() / mIter;
    uint32_t rowStart = mIterIdx * singleCoreM;
    uint32_t colStart = nIterIdx * singleCoreN;

    // Data is hifloat4x2_t: 2 elements per byte (0.5 B/elem).
    uint64_t gmOffsetA = static_cast<uint64_t>(rowStart) * k / 2;
    uint64_t gmOffsetB = static_cast<uint64_t>(colStart) / 2;

    // Scale GM is fractalized:
    //   HIF4_A_ZZ: [m/16][k/64][16,4] -> m-fractal stride = k bytes
    //   HIF4_B_NN: [n/16][k/64][16,4] -> n-fractal stride = k bytes
    // rowStart/colStart are 16-aligned, so the base offset is (start/16)*k bytes.
    uint64_t gmOffsetScaleA = static_cast<uint64_t>(rowStart >> 4) * k;
    uint64_t gmOffsetScaleB = static_cast<uint64_t>(colStart >> 4) * k;

    uint64_t gmOffsetC = static_cast<uint64_t>(rowStart) * n + colStart;

    currentSrc0 = src0 + gmOffsetA;
    currentSrc1 = src1 + gmOffsetB;
    currentSrc2 = src2 + gmOffsetScaleA;
    currentSrc3 = src3 + gmOffsetScaleB;
    currentDst = out + gmOffsetC;
}

// Load the FULL-K HiF4 scale panels for a base block into L1, then signal MTE1.
// The scale L1 buffers are single-buffered and reused every base block, hence:
//   * wait EV_SCALE_FREE (MTE1) before overwriting them;
//   * set  EV_SCALE_LOADED (MTE2) after both TLOADs complete.
template <
    typename X, int m, int k, int n, uint32_t baseM, uint32_t baseN, typename TileScaleA, typename TileScaleB>
AICORE inline void LoadScaleFullK(
    TileScaleA& aScaleMatTile, TileScaleB& bScaleMatTile, __gm__ X* currentSrc2, __gm__ X* currentSrc3, uint32_t i,
    uint32_t j, uint32_t currentM, uint32_t currentN)
{
    constexpr uint32_t fullScaleK = k / HIF4_SCALE_GROUP;

    using DynShapeScaleA = TileShape2D<uint8_t, -1, -1, Layout::HIF4_A_ZZ>;
    using FullStrideScaleA = BaseShape2D<uint8_t, m, fullScaleK, Layout::HIF4_A_ZZ>;
    using GlobalScaleA = GlobalTensor<uint8_t, DynShapeScaleA, FullStrideScaleA, Layout::HIF4_A_ZZ>;

    using DynShapeScaleB = TileShape2D<uint8_t, -1, -1, Layout::HIF4_B_NN>;
    using FullStrideScaleB = BaseShape2D<uint8_t, n, fullScaleK, Layout::HIF4_B_NN>;
    using GlobalScaleB = GlobalTensor<uint8_t, DynShapeScaleB, FullStrideScaleB, Layout::HIF4_B_NN>;

    // i-th M base block, full-K A scale: base pointer advances i*(baseM/16)*k bytes.
    GlobalScaleA gmA(currentSrc2 + static_cast<uint64_t>(i) * (baseM >> 4) * k, DynShapeScaleA(currentM, fullScaleK));
    // j-th N base block, full-K B scale.
    GlobalScaleB gmB(currentSrc3 + static_cast<uint64_t>(j) * (baseN >> 4) * k, DynShapeScaleB(fullScaleK, currentN));

    WaitFlag<PIPE_MTE1, PIPE_MTE2>(EV_SCALE_FREE);
    TLOAD<TileScaleA, GlobalScaleA>(aScaleMatTile, gmA);
    TLOAD<TileScaleB, GlobalScaleB>(bScaleMatTile, gmB);
    SetFlag<PIPE_MTE2, PIPE_MTE1>(EV_SCALE_LOADED);
}

template <
    typename U, typename X, int m, int k, int n, uint32_t baseM, uint32_t baseK, uint32_t baseN, uint32_t baseScaleK,
    uint32_t stepKa, uint32_t stepKb, typename TileMatA, typename TileMatB, typename TileScaleA, typename TileScaleB,
    typename LeftTile, typename RightTile, typename LeftScaleTile, typename RightScaleTile, typename ResTile>
AICORE inline void ProcessKIteration(
    uint32_t kIter, uint32_t loopsK, uint32_t i, uint32_t j, __gm__ U* currentSrc0, __gm__ U* currentSrc1,
    TileMatA aMatTile[BUFFER_NUM], TileMatB bMatTile[BUFFER_NUM], TileScaleA& aScaleMatTile, TileScaleB& bScaleMatTile,
    LeftTile aTile[BUFFER_NUM], RightTile bTile[BUFFER_NUM], LeftScaleTile aScaleTile[BUFFER_NUM],
    RightScaleTile bScaleTile[BUFFER_NUM], ResTile& cTile, uint8_t& mte2DBFlag, uint8_t& mte1DBFlag,
    uint32_t currentM, uint32_t currentN)
{
    using DynShapeDim5 = Shape<1, 1, 1, -1, -1>;
    using GlobalDataSrc0 = GlobalTensor<U, DynShapeDim5, BaseShape2D<U, m, k, Layout::ND>, Layout::ND>;
    using GlobalDataSrc1 = GlobalTensor<U, DynShapeDim5, BaseShape2D<U, k, n, Layout::ND>, Layout::ND>;

    const uint32_t kModstepKa = kIter % stepKa;
    const bool isFirstK = (kIter == 0);
    const bool isLastK = (kIter == loopsK - 1);

    // TLOAD stage (double-buffered data; scale was loaded once in LoadScaleFullK).
    if (kModstepKa == 0) {
        GlobalDataSrc0 gmA(
            currentSrc0 + static_cast<uint64_t>(i) * baseM * k / 2 + static_cast<uint64_t>(kIter) * baseK / 2,
            DynShapeDim5(currentM, baseK * stepKa));
        GlobalDataSrc1 gmB(
            currentSrc1 + static_cast<uint64_t>(kIter) * baseK * n / 2 + static_cast<uint64_t>(j) * baseN / 2,
            DynShapeDim5(baseK * stepKb, currentN));

        WaitFlag<PIPE_MTE1, PIPE_MTE2>(mte2DBFlag);
        TLOAD(aMatTile[mte2DBFlag], gmA);
        SetFlag<PIPE_MTE2, PIPE_MTE1>(EV_DATA_A);
        TLOAD(bMatTile[mte2DBFlag], gmB);
        SetFlag<PIPE_MTE2, PIPE_MTE1>(EV_DATA_B);
        mte2DBFlag = (mte2DBFlag == 0) ? 1 : 0; // flip now: currMte2Idx below is the just-loaded buffer
    }
    const uint32_t currMte2Idx = (mte2DBFlag == 0) ? 1 : 0;

    // TEXTRACT stage: slice data and scale into the ping-pong L0 slot.
    WaitFlag<PIPE_M, PIPE_MTE1>(mte1DBFlag);
    if (kModstepKa == 0) {
        WaitFlag<PIPE_MTE2, PIPE_MTE1>(EV_DATA_A);
    }
    TEXTRACT(aTile[mte1DBFlag], aMatTile[currMte2Idx], 0, kModstepKa * baseK);
    if (isFirstK) {
        WaitFlag<PIPE_MTE2, PIPE_MTE1>(EV_SCALE_LOADED);
    }
    TEXTRACT(aScaleTile[mte1DBFlag], aScaleMatTile, 0, kIter * baseScaleK * HIF4_COL_BYTES);

    if (kModstepKa == 0) {
        WaitFlag<PIPE_MTE2, PIPE_MTE1>(EV_DATA_B);
    }
    TEXTRACT(bTile[mte1DBFlag], bMatTile[currMte2Idx], (kIter % stepKb) * baseK, 0);
    TEXTRACT(bScaleTile[mte1DBFlag], bScaleMatTile, kIter * baseScaleK * HIF4_COL_BYTES, 0);
    if (isLastK) {
        SetFlag<PIPE_MTE1, PIPE_MTE2>(EV_SCALE_FREE);
    }

    if ((kIter + 1) % stepKa == 0) {
        SetFlag<PIPE_MTE1, PIPE_MTE2>(currMte2Idx);
    }

    // TMATMUL stage.
    SetFlag<PIPE_MTE1, PIPE_M>(mte1DBFlag);
    WaitFlag<PIPE_MTE1, PIPE_M>(mte1DBFlag);
    MatmulAcc(cTile, aTile[mte1DBFlag], bTile[mte1DBFlag], aScaleTile[mte1DBFlag], bScaleTile[mte1DBFlag], kIter);
    SetFlag<PIPE_M, PIPE_MTE1>(mte1DBFlag);
    mte1DBFlag = (mte1DBFlag == 0) ? 1 : 0;
}

template <typename T, int m, int n, uint32_t baseM, uint32_t baseN, typename ResTile>
AICORE inline void StoreResult(ResTile& cTile, __gm__ T* currentDst, uint32_t i, uint32_t j, uint32_t currentM,
    uint32_t currentN)
{
    SetFlag<PIPE_M, PIPE_FIX>(0);
    WaitFlag<PIPE_M, PIPE_FIX>(0);

    using DynShapeDim5 = Shape<1, 1, 1, -1, -1>;
    using DynStrideDim5 = pto::Stride<m * n, m * n, m * n, n, 1>;
    using GlobalDataOut = GlobalTensor<T, DynShapeDim5, DynStrideDim5, Layout::ND>;

    uint32_t gmOffset = i * baseM * n + j * baseN;
    GlobalDataOut dstGlobal(currentDst + gmOffset, DynShapeDim5(currentM, currentN));
    TSTORE(dstGlobal, cTile);

    SetFlag<PIPE_FIX, PIPE_M>(0);
    WaitFlag<PIPE_FIX, PIPE_M>(0);
}

AICORE inline void InitSyncFlags()
{
    SetFlag<PIPE_MTE1, PIPE_MTE2>(0);
    SetFlag<PIPE_MTE1, PIPE_MTE2>(1);
    SetFlag<PIPE_M, PIPE_MTE1>(0);
    SetFlag<PIPE_M, PIPE_MTE1>(1);
    SetFlag<PIPE_MTE1, PIPE_MTE2>(EV_SCALE_FREE);
}

AICORE inline void WaitSyncFlags()
{
    WaitFlag<PIPE_M, PIPE_MTE1>(0);
    WaitFlag<PIPE_M, PIPE_MTE1>(1);
    WaitFlag<PIPE_MTE1, PIPE_MTE2>(0);
    WaitFlag<PIPE_MTE1, PIPE_MTE2>(1);
    WaitFlag<PIPE_MTE1, PIPE_MTE2>(EV_SCALE_FREE);
}

template <
    typename T, typename U, typename X, int m, int k, int n, uint32_t singleCoreM, uint32_t singleCoreK,
    uint32_t singleCoreN, uint32_t baseM, uint32_t baseK, uint32_t baseN, uint32_t baseScaleK, uint32_t fullScaleK,
    uint32_t scaleCols, uint32_t stepKa, uint32_t stepKb, typename TileMatA, typename TileMatB, typename TileScaleA,
    typename TileScaleB, typename LeftTile, typename RightTile, typename LeftScaleTile, typename RightScaleTile,
    typename ResTile>
AICORE inline void Compute(
    __gm__ U* currentSrc0, __gm__ U* currentSrc1, __gm__ X* currentSrc2, __gm__ X* currentSrc3, __gm__ T*& currentDst,
    TileMatA aMatTile[BUFFER_NUM], TileMatB bMatTile[BUFFER_NUM], TileScaleA& aScaleMatTile,
    TileScaleB& bScaleMatTile, LeftTile aTile[BUFFER_NUM], RightTile bTile[BUFFER_NUM],
    LeftScaleTile aScaleTile[BUFFER_NUM], RightScaleTile bScaleTile[BUFFER_NUM], ResTile& cTile,
    uint32_t currentSingleCoreM, uint32_t currentSingleCoreN)
{
    const uint32_t loopsM = (singleCoreM + baseM - 1) / baseM;
    const uint32_t loopsN = (singleCoreN + baseN - 1) / baseN;
    const uint32_t loopsK = (singleCoreK + baseK - 1) / baseK;

    uint32_t remM = currentSingleCoreM % baseM;
    uint32_t remN = currentSingleCoreN % baseN;

    for (uint32_t i = 0; i < loopsM; ++i) {
        for (uint32_t j = 0; j < loopsN; ++j) {
            uint32_t currentM = (i == loopsM - 1 && remM > 0) ? remM : baseM;
            uint32_t currentN = (j == loopsN - 1 && remN > 0) ? remN : baseN;

            uint8_t mte2DBFlag = 0;
            uint8_t mte1DBFlag = 0;

            for (int buf = 0; buf < BUFFER_NUM; ++buf) {
                aMatTile[buf] = TileMatA(currentM, baseK * stepKa);
                bMatTile[buf] = TileMatB(baseK * stepKb, currentN);
                aTile[buf] = LeftTile(currentM, baseK);
                bTile[buf] = RightTile(baseK, currentN);
                aScaleTile[buf] = LeftScaleTile(currentM, baseScaleK * HIF4_COL_BYTES);
                bScaleTile[buf] = RightScaleTile(baseScaleK * HIF4_COL_BYTES, currentN);
            }

            ResTile outTile(currentM, currentN);
            TASSIGN(outTile, 0x0);

            // Full-K scale for this base block (once per (i,j)).
            LoadScaleFullK<X, m, k, n, baseM, baseN, TileScaleA, TileScaleB>(
                aScaleMatTile, bScaleMatTile, currentSrc2, currentSrc3, i, j, currentM, currentN);

            for (uint32_t kIter = 0; kIter < loopsK; ++kIter) {
                ProcessKIteration<
                    U, X, m, k, n, baseM, baseK, baseN, baseScaleK, stepKa, stepKb, TileMatA, TileMatB, TileScaleA,
                    TileScaleB, LeftTile, RightTile, LeftScaleTile, RightScaleTile, ResTile>(
                    kIter, loopsK, i, j, currentSrc0, currentSrc1, aMatTile, bMatTile, aScaleMatTile, bScaleMatTile,
                    aTile, bTile, aScaleTile, bScaleTile, outTile, mte2DBFlag, mte1DBFlag, currentM, currentN);
            }

            StoreResult<T, m, n, baseM, baseN, ResTile>(outTile, currentDst, i, j, currentM, currentN);
        }
    }
}

template <
    typename T, typename U, typename X, uint32_t blockDim, int m, int k, int n, uint32_t baseM, uint32_t baseK,
    uint32_t baseN, uint32_t stepKa, uint32_t stepKb, uint32_t singleCoreM, uint32_t singleCoreK, uint32_t singleCoreN>
AICORE inline void RunHif4MatmulDispatch(__gm__ T* out, __gm__ U* src0, __gm__ U* src1, __gm__ X* src2, __gm__ X* src3)
{
    constexpr uint32_t mIter = (m + singleCoreM - 1) / singleCoreM;
    constexpr uint32_t nIter = (n + singleCoreN - 1) / singleCoreN;

    uint32_t coreIdx = get_block_idx();
    if (coreIdx >= mIter * nIter) {
        return;
    }

    __gm__ U* currentSrc0 = nullptr;
    __gm__ U* currentSrc1 = nullptr;
    __gm__ X* currentSrc2 = nullptr;
    __gm__ X* currentSrc3 = nullptr;
    __gm__ T* currentDst = nullptr;

    InitGMOffsets<T, U, X, m, k, n, singleCoreM, singleCoreK, singleCoreN>(
        currentSrc0, currentSrc1, currentSrc2, currentSrc3, currentDst, out, src0, src1, src2, src3);

    constexpr uint32_t baseScaleK = baseK / HIF4_SCALE_GROUP;
    constexpr uint32_t fullScaleK = k / HIF4_SCALE_GROUP;
    constexpr uint32_t scaleCols = fullScaleK * HIF4_COL_BYTES;

    // L1 data tiles (dynamic shape, double-buffered).
    using TileMatA = Tile<TileType::Mat, U, baseM, baseK * stepKa, BLayout::ColMajor, -1, -1, SLayout::RowMajor,
        TileConfig::fractalABSize>;
    using TileMatB = Tile<TileType::Mat, U, baseK * stepKb, baseN, BLayout::ColMajor, -1, -1, SLayout::RowMajor,
        TileConfig::fractalABSize>;
    // L1 scale tiles (full-K, single-buffered).
    using TileScaleA = Tile<TileType::Mat, X, baseM, scaleCols, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 32>;
    using TileScaleB = Tile<TileType::Mat, X, scaleCols, baseN, BLayout::ColMajor, -1, -1, SLayout::ColMajor, 32>;

    TileMatA aMatTile[BUFFER_NUM];
    TileMatB bMatTile[BUFFER_NUM];
    TileScaleA aScaleMatTile;
    TileScaleB bScaleMatTile;

    using LeftTile = TileLeft<U, baseM, baseK, -1, -1>;
    using RightTile = TileRight<U, baseK, baseN, -1, -1>;
    using LeftScaleTile = TileLeftScale<X, baseM, baseScaleK * HIF4_COL_BYTES, -1, -1>;
    using RightScaleTile = TileRightScale<X, baseScaleK * HIF4_COL_BYTES, baseN, -1, -1>;
    using ResTile = TileAcc<float, baseM, baseN, -1, -1>;

    LeftTile aTile[BUFFER_NUM];
    RightTile bTile[BUFFER_NUM];
    LeftScaleTile aScaleTile[BUFFER_NUM];
    RightScaleTile bScaleTile[BUFFER_NUM];
    ResTile cTile;

    // L1 data buffers (ping-pong), then scale buffers after them.
    TASSIGN(aMatTile[0], 0x0);
    TASSIGN(aMatTile[1], 0x0 + baseM * baseK * stepKa * sizeof(U) / 2);
    TASSIGN(bMatTile[0], 0x0 + baseM * baseK * stepKa * sizeof(U) / 2 * BUFFER_NUM);
    TASSIGN(
        bMatTile[1],
        0x0 + baseM * baseK * stepKa * sizeof(U) / 2 * BUFFER_NUM + baseK * baseN * stepKb * sizeof(U) / 2);

    const uint32_t scaleBaseAddr =
        baseM * baseK * stepKa * sizeof(U) / 2 * BUFFER_NUM + baseK * baseN * stepKb * sizeof(U) / 2 * BUFFER_NUM;
    TASSIGN(aScaleMatTile, scaleBaseAddr);
    TASSIGN(bScaleMatTile, scaleBaseAddr + baseM * scaleCols * sizeof(X));

    // L0A/L0B ping-pong buffers.
    TASSIGN(aTile[0], 0x0);
    TASSIGN(aTile[1], 0x0 + L0_PINGPONG_BYTES);
    TASSIGN(bTile[0], 0x0);
    TASSIGN(bTile[1], 0x0 + L0_PINGPONG_BYTES);
    TASSIGN(cTile, 0x0);

    TASSIGN(aScaleTile[0], GetScaleAddr(aTile[0].data()));
    TASSIGN(aScaleTile[1], GetScaleAddr(aTile[1].data()));
    TASSIGN(bScaleTile[0], GetScaleAddr(bTile[0].data()));
    TASSIGN(bScaleTile[1], GetScaleAddr(bTile[1].data()));

    InitSyncFlags();
    Compute<
        T, U, X, m, k, n, singleCoreM, singleCoreK, singleCoreN, baseM, baseK, baseN, baseScaleK, fullScaleK,
        scaleCols, stepKa, stepKb, TileMatA, TileMatB, TileScaleA, TileScaleB, LeftTile, RightTile, LeftScaleTile,
        RightScaleTile, ResTile>(
        currentSrc0, currentSrc1, currentSrc2, currentSrc3, currentDst, aMatTile, bMatTile, aScaleMatTile,
        bScaleMatTile, aTile, bTile, aScaleTile, bScaleTile, cTile, singleCoreM, singleCoreN);

    WaitSyncFlags();
}

template <
    uint32_t blockDim, uint32_t m, uint32_t k, uint32_t n, uint32_t singleCoreM, uint32_t singleCoreK,
    uint32_t singleCoreN, uint32_t baseM, uint32_t baseK, uint32_t baseN, uint32_t stepKa, uint32_t stepKb>
__global__ AICORE void Hif4MatmulPerformance(
    __gm__ uint8_t* out, __gm__ uint8_t* src0, __gm__ uint8_t* src1, __gm__ uint8_t* src2, __gm__ uint8_t* src3)
{
    RunHif4MatmulDispatch<
        bfloat16_t, hifloat4x2_t, uint8_t, blockDim, m, k, n, baseM, baseK, baseN, stepKa, stepKb, singleCoreM,
        singleCoreK, singleCoreN>(
        reinterpret_cast<__gm__ bfloat16_t*>(out), reinterpret_cast<__gm__ hifloat4x2_t*>(src0),
        reinterpret_cast<__gm__ hifloat4x2_t*>(src1), reinterpret_cast<__gm__ uint8_t*>(src2),
        reinterpret_cast<__gm__ uint8_t*>(src3));
}

void LaunchHif4Matmul(uint8_t* out, uint8_t* src0, uint8_t* src1, uint8_t* src2, uint8_t* src3, void* stream)
{
    constexpr uint32_t blockDim = 16;
    constexpr uint32_t m = 2048;
    constexpr uint32_t k = 2048;
    constexpr uint32_t n = 2048;
    constexpr uint32_t singleCoreM = 512;
    constexpr uint32_t singleCoreK = 2048;
    constexpr uint32_t singleCoreN = 512;
    constexpr uint32_t baseM = 256;
    constexpr uint32_t baseK = 256;
    constexpr uint32_t baseN = 256;
    constexpr uint32_t stepKa = 2;
    constexpr uint32_t stepKb = 2;

    Hif4MatmulPerformance<
        blockDim, m, k, n, singleCoreM, singleCoreK, singleCoreN, baseM, baseK, baseN, stepKa, stepKb>
        <<<blockDim, nullptr, stream>>>(out, src0, src1, src2, src3);
}

void LaunchHif4Matmul(uint8_t* out, uint8_t* src0, uint8_t* src1, uint8_t* src2, uint8_t* src3, void* stream);