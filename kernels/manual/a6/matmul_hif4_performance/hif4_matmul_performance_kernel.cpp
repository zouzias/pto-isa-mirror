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
// A6 (dav-9201) HiF4 GEMM demo, single-buffered sequential tiling (correctness-first
// reference). Pipeline:
//
//   GM(BF16) --TLOAD--> L1 --TEXTRACT--> L0A/L0B + L0AMX/L0BMX --TMATMUL_MX--> L0C --TSTORE--> GM(BF16)
//
// HiF4 scale is pre-fractalized in GM ([16,4] cells, HIF4_A_ZZ / HIF4_B_NN) and is
// consumed by contiguous ZZ2ZZ/NN2NN bursts, so it cannot be K-sliced at TLOAD time.
// The full-K scale for each (M,N) base block is therefore loaded once, and the per-K
// slice is produced at TEXTRACT (L1 -> L0AMX/L0BMX).
//
// This version is fully sequential (no ping-pong): each TLOAD/TEXTRACT/TMATMUL stage
// is ordered by a cross-pipe set_flag/wait_flag chain with strict 1:1 pairing. It is
// the validated-correct baseline; a double-buffered variant is a follow-up.
// --------------------------------------------------------------------------------

#include <pto/common/constants.hpp>
#include <pto/common/debug.h>
#include <pto/pto-inst.hpp>

using namespace pto;

constexpr uint32_t HIF4_SCALE_GROUP = 64;  // HiF4 groups over 64 elements
constexpr uint32_t HIF4_COL_BYTES = 4;     // 4 bytes per 64-group (Ea/Eb/Ec packed cell width)

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

// Work partition: core (mIterIdx, nIterIdx) owns C tile
// [rowStart, rowStart+singleCoreM) x [colStart, colStart+singleCoreN).
template <
    typename T, typename U, typename X, int m, int k, int n, uint32_t singleCoreM, uint32_t singleCoreN>
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
    // Scale GM fractal: [*/16][k/64][16,4] -> fractal stride = k bytes.
    uint64_t gmOffsetScaleA = static_cast<uint64_t>(rowStart >> 4) * k;
    uint64_t gmOffsetScaleB = static_cast<uint64_t>(colStart >> 4) * k;
    uint64_t gmOffsetC = static_cast<uint64_t>(rowStart) * n + colStart;

    currentSrc0 = src0 + gmOffsetA;
    currentSrc1 = src1 + gmOffsetB;
    currentSrc2 = src2 + gmOffsetScaleA;
    currentSrc3 = src3 + gmOffsetScaleB;
    currentDst = out + gmOffsetC;
}

constexpr uint32_t EV_SCALE_LOADED = 0;  // MTE2 -> MTE1
constexpr uint32_t EV_DATA_LOADED = 1;   // MTE2 -> MTE1
constexpr uint32_t EV_L1_FREE = 2;       // MTE1 -> MTE2
constexpr uint32_t EV_SCALE_FREE = 2;    // MTE1 -> MTE2
constexpr uint32_t EV_EXTRACT_DONE = 3;  // MTE1 -> M
constexpr uint32_t EV_MATMUL_DONE = 4;   // M -> MTE1
constexpr uint32_t EV_STORE = 5;         // M -> FIX
constexpr uint32_t EV_STORE_DONE = 6;    // FIX -> MTE2

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
    using FullStrideScaleB = BaseShape2D<uint8_t, fullScaleK, n, Layout::HIF4_B_NN>;
    using GlobalScaleB = GlobalTensor<uint8_t, DynShapeScaleB, FullStrideScaleB, Layout::HIF4_B_NN>;

    GlobalScaleA gmA(currentSrc2 + static_cast<uint64_t>(i) * (baseM >> 4) * k, DynShapeScaleA(currentM, fullScaleK));
    GlobalScaleB gmB(currentSrc3 + static_cast<uint64_t>(j) * (baseN >> 4) * k, DynShapeScaleB(fullScaleK, currentN));

    WaitFlag<PIPE_MTE1, PIPE_MTE2>(EV_SCALE_FREE);
    TLOAD<TileScaleA, GlobalScaleA>(aScaleMatTile, gmA);
    TLOAD<TileScaleB, GlobalScaleB>(bScaleMatTile, gmB);
    SetFlag<PIPE_MTE2, PIPE_MTE1>(EV_SCALE_LOADED);
}

template <
    typename T, typename U, typename X, int m, int k, int n, uint32_t singleCoreM, uint32_t singleCoreN,
    uint32_t baseM, uint32_t baseK, uint32_t baseN>
AICORE inline void RunHif4MatmulCore(
    __gm__ T* out, __gm__ U* src0, __gm__ U* src1, __gm__ X* src2, __gm__ X* src3)
{
    __gm__ U* currentSrc0 = nullptr;
    __gm__ U* currentSrc1 = nullptr;
    __gm__ X* currentSrc2 = nullptr;
    __gm__ X* currentSrc3 = nullptr;
    __gm__ T* currentDst = nullptr;
    InitGMOffsets<T, U, X, m, k, n, singleCoreM, singleCoreN>(
        currentSrc0, currentSrc1, currentSrc2, currentSrc3, currentDst, out, src0, src1, src2, src3);

    constexpr uint32_t baseScaleK = baseK / HIF4_SCALE_GROUP;
    constexpr uint32_t fullScaleK = k / HIF4_SCALE_GROUP;
    constexpr uint32_t scaleCols = fullScaleK * HIF4_COL_BYTES;

    using TileMatA = Tile<TileType::Mat, U, baseM, baseK, BLayout::ColMajor, -1, -1, SLayout::RowMajor,
        TileConfig::fractalABSize>;
    using TileMatB = Tile<TileType::Mat, U, baseK, baseN, BLayout::ColMajor, -1, -1, SLayout::RowMajor,
        TileConfig::fractalABSize>;
    using TileScaleA = Tile<TileType::Mat, X, baseM, scaleCols, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 32>;
    using TileScaleB = Tile<TileType::Mat, X, scaleCols, baseN, BLayout::ColMajor, -1, -1, SLayout::ColMajor, 32>;

    using LeftTile = TileLeft<U, baseM, baseK, -1, -1>;
    using RightTile = TileRight<U, baseK, baseN, -1, -1>;
    using LeftScaleTile = TileLeftScale<X, baseM, baseScaleK * HIF4_COL_BYTES, -1, -1>;
    using RightScaleTile = TileRightScale<X, baseScaleK * HIF4_COL_BYTES, baseN, -1, -1>;
    using ResTile = TileAcc<float, baseM, baseN, -1, -1>;

    using DynShapeDim5 = Shape<1, 1, 1, -1, -1>;
    using GlobalDataSrc0 = GlobalTensor<U, DynShapeDim5, BaseShape2D<U, m, k, Layout::ND>, Layout::ND>;
    using GlobalDataSrc1 = GlobalTensor<U, DynShapeDim5, BaseShape2D<U, k, n, Layout::ND>, Layout::ND>;
    using DynStrideDim5 = pto::Stride<m * n, m * n, m * n, n, 1>;
    using GlobalDataOut = GlobalTensor<T, DynShapeDim5, DynStrideDim5, Layout::ND>;

    const uint32_t loopsM = (singleCoreM + baseM - 1) / baseM;
    const uint32_t loopsN = (singleCoreN + baseN - 1) / baseN;
    const uint32_t loopsK = (k + baseK - 1) / baseK;
    const uint32_t remM = singleCoreM % baseM;
    const uint32_t remN = singleCoreN % baseN;

    // Prime the full-K scale buffer as free for the first base block.
    SetFlag<PIPE_MTE1, PIPE_MTE2>(EV_SCALE_FREE);

    for (uint32_t i = 0; i < loopsM; ++i) {
        uint32_t currentM = (i == loopsM - 1 && remM > 0) ? remM : baseM;
        for (uint32_t j = 0; j < loopsN; ++j) {
            uint32_t currentN = (j == loopsN - 1 && remN > 0) ? remN : baseN;

            // Construct with runtime valid dims, then TASSIGN sets the buffer address.
            TileMatA aMatTile(currentM, baseK);
            TileMatB bMatTile(baseK, currentN);
            TileScaleA aScaleMatTile(currentM, scaleCols);
            TileScaleB bScaleMatTile(scaleCols, currentN);
            LeftTile aTile(currentM, baseK);
            RightTile bTile(baseK, currentN);
            LeftScaleTile aScaleTile(currentM, baseScaleK * HIF4_COL_BYTES);
            RightScaleTile bScaleTile(baseScaleK * HIF4_COL_BYTES, currentN);

            // L1 buffer assignment (tightly packed, single-buffered). fp4 data = 0.5 B/elem.
            TASSIGN(aMatTile, 0x0);
            TASSIGN(bMatTile, 0x0 + baseM * baseK / 2);
            TASSIGN(aScaleMatTile, 0x0 + baseM * baseK / 2 + baseK * baseN / 2);
            TASSIGN(bScaleMatTile, 0x0 + baseM * baseK / 2 + baseK * baseN / 2 + baseM * scaleCols);
            // L0 buffer assignment.
            TASSIGN(aTile, 0x0);
            TASSIGN(bTile, 0x0);
            TASSIGN(aScaleTile, GetScaleAddr(aTile.data()));
            TASSIGN(bScaleTile, GetScaleAddr(bTile.data()));

            ResTile outTile(currentM, currentN);
            TASSIGN(outTile, 0x0);

            // Load full-K scale for this base block once.
            LoadScaleFullK<X, m, k, n, baseM, baseN, TileScaleA, TileScaleB>(
                aScaleMatTile, bScaleMatTile, currentSrc2, currentSrc3, i, j, currentM, currentN);

            for (uint32_t kIter = 0; kIter < loopsK; ++kIter) {
                if (kIter > 0) {
                    WaitFlag<PIPE_MTE1, PIPE_MTE2>(EV_L1_FREE);
                }

                // Load data panels (MTE2).
                GlobalDataSrc0 gmA(
                    currentSrc0 + static_cast<uint64_t>(i) * baseM * k / 2 +
                        static_cast<uint64_t>(kIter) * baseK / 2,
                    DynShapeDim5(currentM, baseK));
                GlobalDataSrc1 gmB(
                    currentSrc1 + static_cast<uint64_t>(kIter) * baseK * n / 2 +
                        static_cast<uint64_t>(j) * baseN / 2,
                    DynShapeDim5(baseK, currentN));
                TLOAD(aMatTile, gmA);
                TLOAD(bMatTile, gmB);
                SetFlag<PIPE_MTE2, PIPE_MTE1>(EV_DATA_LOADED);
                WaitFlag<PIPE_MTE2, PIPE_MTE1>(EV_DATA_LOADED);

                if (kIter > 0) {
                    WaitFlag<PIPE_M, PIPE_MTE1>(EV_MATMUL_DONE);
                }

                // Extract data + scale slices (MTE1).
                TEXTRACT(aTile, aMatTile, 0, 0);
                if (kIter == 0) {
                    WaitFlag<PIPE_MTE2, PIPE_MTE1>(EV_SCALE_LOADED);
                }
                TEXTRACT(aScaleTile, aScaleMatTile, 0, kIter * baseScaleK * HIF4_COL_BYTES);
                TEXTRACT(bTile, bMatTile, 0, 0);
                TEXTRACT(bScaleTile, bScaleMatTile, kIter * baseScaleK * HIF4_COL_BYTES, 0);
                if (kIter == loopsK - 1) {
                    SetFlag<PIPE_MTE1, PIPE_MTE2>(EV_SCALE_FREE);
                }

                if (kIter < loopsK - 1) {
                    SetFlag<PIPE_MTE1, PIPE_MTE2>(EV_L1_FREE);
                }
                SetFlag<PIPE_MTE1, PIPE_M>(EV_EXTRACT_DONE);
                WaitFlag<PIPE_MTE1, PIPE_M>(EV_EXTRACT_DONE);

                MatmulAcc(outTile, aTile, bTile, aScaleTile, bScaleTile, kIter);

                if (kIter < loopsK - 1) {
                    SetFlag<PIPE_M, PIPE_MTE1>(EV_MATMUL_DONE);
                }
            }

            // Store (FIX) after the last matmul, then release L1 for the next tile.
            SetFlag<PIPE_M, PIPE_FIX>(EV_STORE);
            WaitFlag<PIPE_M, PIPE_FIX>(EV_STORE);
            GlobalDataOut dstGlobal(
                currentDst + static_cast<uint64_t>(i) * baseM * n + j * baseN, DynShapeDim5(currentM, currentN));
            TSTORE(dstGlobal, outTile);
            SetFlag<PIPE_FIX, PIPE_MTE2>(EV_STORE_DONE);
            WaitFlag<PIPE_FIX, PIPE_MTE2>(EV_STORE_DONE);
        }
    }

    // Drain the final block's dangling scale-free token.
    WaitFlag<PIPE_MTE1, PIPE_MTE2>(EV_SCALE_FREE);
}

template <
    uint32_t blockDim, uint32_t m, uint32_t k, uint32_t n, uint32_t singleCoreM, uint32_t singleCoreN,
    uint32_t baseM, uint32_t baseK, uint32_t baseN>
__global__ AICORE void Hif4MatmulPerformance(
    __gm__ uint8_t* out, __gm__ uint8_t* src0, __gm__ uint8_t* src1, __gm__ uint8_t* src2, __gm__ uint8_t* src3)
{
    RunHif4MatmulCore<
        bfloat16_t, hifloat4x2_t, uint8_t, m, k, n, singleCoreM, singleCoreN, baseM, baseK, baseN>(
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
    constexpr uint32_t singleCoreN = 512;
    constexpr uint32_t baseM = 256;
    constexpr uint32_t baseK = 256;
    constexpr uint32_t baseN = 256;

    Hif4MatmulPerformance<blockDim, m, k, n, singleCoreM, singleCoreN, baseM, baseK, baseN>
        <<<blockDim, nullptr, stream>>>(out, src0, src1, src2, src3);
}

void LaunchHif4Matmul(uint8_t* out, uint8_t* src0, uint8_t* src1, uint8_t* src2, uint8_t* src3, void* stream);