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
// A6 (dav-9201) HiF4 GEMM demo, double-buffered tiling. Pipeline:
//
//   GM(BF16) --TLOAD--> L1 --TEXTRACT--> L0A/L0B + L0AMX/L0BMX --TMATMUL_MX--> L0C --TSTORE--> GM(BF16)
//
// Double-buffering:
//   * L1 data tiles (A/B) ping-pong (2 buffers);
//   * L0A/L0B data tiles ping-pong (2 slots);
//   * L0AMX/L0BMX scale tiles SINGLE-buffered -- a HiF4 scale tile is [256,16]=4 KB,
//     which already fills PTO_SCALELEFT/SCALERIGHT_SIZE_BYTES (4 KB), so two would
//     overflow. Scale reuse is serialized by a dedicated matmul-done event.
//   * full-K HiF4 scale (HIF4_A_ZZ/HIF4_B_NN) is pre-fractalized in GM and cannot be
//     K-sliced at TLOAD; it is loaded once per (M,N) base block and K-sliced at TEXTRACT.
// --------------------------------------------------------------------------------

#include <pto/common/constants.hpp>
#include <pto/pto-inst.hpp>

using namespace pto;

constexpr uint32_t BUFFER_NUM = 2;        // double-buffered data tiles/slots
constexpr uint32_t HIF4_SCALE_GROUP = 64; // HiF4 groups over 64 elements
constexpr uint32_t HIF4_COL_BYTES = 4;    // packed scale cell width per 64-group
constexpr uint32_t L0_PINGPONG_BYTES = 32 * 1024;

// Event ids (namespaced per (src,dst) pipe pair).
constexpr uint32_t EV_DATA_A = 0;            // MTE2 -> MTE1
constexpr uint32_t EV_DATA_B = 1;            // MTE2 -> MTE1
constexpr uint32_t EV_SCALE_LOADED = 2;      // MTE2 -> MTE1
constexpr uint32_t EV_DATA_FREE = 0;         // MTE1 -> MTE2 (ping-pong 0/1)
constexpr uint32_t EV_SCALE_FREE = 2;        // MTE1 -> MTE2
constexpr uint32_t EV_EXTRACT_DONE = 0;      // MTE1 -> M (ping-pong 0/1)
constexpr uint32_t EV_MATMUL_DONE = 0;       // M -> MTE1 (ping-pong 0/1)
constexpr uint32_t EV_SCALE_MATMUL_DONE = 2; // M -> MTE1 (scale L0 single-buffer)
constexpr uint32_t EV_STORE = 0;             // M -> FIX
constexpr uint32_t EV_STORE_DONE = 0;        // FIX -> MTE2

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

template <typename T, typename U, typename X, int m, int k, int n, uint32_t singleCoreM, uint32_t singleCoreN>
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

    uint64_t gmOffsetA = static_cast<uint64_t>(rowStart) * k / 2;
    uint64_t gmOffsetB = static_cast<uint64_t>(colStart) / 2;
    uint64_t gmOffsetScaleA = static_cast<uint64_t>(rowStart >> 4) * k;
    uint64_t gmOffsetScaleB = static_cast<uint64_t>(colStart >> 4) * k;
    uint64_t gmOffsetC = static_cast<uint64_t>(rowStart) * n + colStart;

    currentSrc0 = src0 + gmOffsetA;
    currentSrc1 = src1 + gmOffsetB;
    currentSrc2 = src2 + gmOffsetScaleA;
    currentSrc3 = src3 + gmOffsetScaleB;
    currentDst = out + gmOffsetC;
}

template <typename X, int m, int k, int n, uint32_t baseM, uint32_t baseN, typename TileScaleA, typename TileScaleB>
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
    typename U, typename X, int m, int k, int n, uint32_t baseM, uint32_t baseK, uint32_t baseN, uint32_t baseScaleK,
    uint32_t stepKa, uint32_t stepKb, typename TileMatA, typename TileMatB, typename TileScaleA, typename TileScaleB,
    typename LeftTile, typename RightTile, typename LeftScaleTile, typename RightScaleTile, typename ResTile>
AICORE inline void ProcessKIteration(
    uint32_t kIter, uint32_t loopsK, uint32_t i, uint32_t j, __gm__ U* currentSrc0, __gm__ U* currentSrc1,
    TileMatA aMatTile[BUFFER_NUM], TileMatB bMatTile[BUFFER_NUM], TileScaleA& aScaleMatTile, TileScaleB& bScaleMatTile,
    LeftTile aTile[BUFFER_NUM], RightTile bTile[BUFFER_NUM], LeftScaleTile& aScaleTile, RightScaleTile& bScaleTile,
    ResTile& cTile, uint8_t& mte2DBFlag, uint8_t& mte1DBFlag, uint32_t currentM, uint32_t currentN)
{
    using DynShapeDim5 = Shape<1, 1, 1, -1, -1>;
    using GlobalDataSrc0 = GlobalTensor<U, DynShapeDim5, BaseShape2D<U, m, k, Layout::ND>, Layout::ND>;
    using GlobalDataSrc1 = GlobalTensor<U, DynShapeDim5, BaseShape2D<U, k, n, Layout::ND>, Layout::ND>;

    const uint32_t kModstepKa = kIter % stepKa;
    const bool isFirstK = (kIter == 0);
    const bool isLastK = (kIter == loopsK - 1);

    // TLOAD stage (double-buffered L1 data).
    if (kModstepKa == 0) {
        GlobalDataSrc0 gmA(
            currentSrc0 + static_cast<uint64_t>(i) * baseM * k / 2 + static_cast<uint64_t>(kIter) * baseK / 2,
            DynShapeDim5(currentM, baseK * stepKa));
        GlobalDataSrc1 gmB(
            currentSrc1 + static_cast<uint64_t>(kIter) * baseK * n / 2 + static_cast<uint64_t>(j) * baseN / 2,
            DynShapeDim5(baseK * stepKb, currentN));

        WaitFlag<PIPE_MTE1, PIPE_MTE2>(EV_DATA_FREE + mte2DBFlag);
        TLOAD(aMatTile[mte2DBFlag], gmA);
        SetFlag<PIPE_MTE2, PIPE_MTE1>(EV_DATA_A);
        TLOAD(bMatTile[mte2DBFlag], gmB);
        SetFlag<PIPE_MTE2, PIPE_MTE1>(EV_DATA_B);
        mte2DBFlag = (mte2DBFlag == 0) ? 1 : 0;
    }
    const uint32_t currMte2Idx = (mte2DBFlag == 0) ? 1 : 0;

    // TEXTRACT stage. Data L0 is ping-ponged; scale L0 is single-buffered and must
    // wait for the previous TMATMUL to have consumed it.
    WaitFlag<PIPE_M, PIPE_MTE1>(EV_MATMUL_DONE + mte1DBFlag);
    if (kIter > 0) {
        WaitFlag<PIPE_M, PIPE_MTE1>(EV_SCALE_MATMUL_DONE);
    }
    if (kModstepKa == 0) {
        WaitFlag<PIPE_MTE2, PIPE_MTE1>(EV_DATA_A);
    }
    TEXTRACT(aTile[mte1DBFlag], aMatTile[currMte2Idx], 0, kModstepKa * baseK);
    if (isFirstK) {
        WaitFlag<PIPE_MTE2, PIPE_MTE1>(EV_SCALE_LOADED);
    }
    TEXTRACT(aScaleTile, aScaleMatTile, 0, kIter * baseScaleK * HIF4_COL_BYTES);

    if (kModstepKa == 0) {
        WaitFlag<PIPE_MTE2, PIPE_MTE1>(EV_DATA_B);
    }
    TEXTRACT(bTile[mte1DBFlag], bMatTile[currMte2Idx], (kIter % stepKb) * baseK, 0);
    TEXTRACT(bScaleTile, bScaleMatTile, kIter * baseScaleK * HIF4_COL_BYTES, 0);

    if ((kIter + 1) % stepKa == 0) {
        SetFlag<PIPE_MTE1, PIPE_MTE2>(EV_DATA_FREE + currMte2Idx);
    }
    if (isLastK) {
        SetFlag<PIPE_MTE1, PIPE_MTE2>(EV_SCALE_FREE);
    }

    SetFlag<PIPE_MTE1, PIPE_M>(EV_EXTRACT_DONE + mte1DBFlag);
    WaitFlag<PIPE_MTE1, PIPE_M>(EV_EXTRACT_DONE + mte1DBFlag);
    MatmulAcc(cTile, aTile[mte1DBFlag], bTile[mte1DBFlag], aScaleTile, bScaleTile, kIter);
    SetFlag<PIPE_M, PIPE_MTE1>(EV_MATMUL_DONE + mte1DBFlag);
    if (kIter < loopsK - 1) {
        SetFlag<PIPE_M, PIPE_MTE1>(EV_SCALE_MATMUL_DONE);
    }
    mte1DBFlag = (mte1DBFlag == 0) ? 1 : 0;
}

template <typename T, int m, int n, uint32_t baseM, uint32_t baseN, typename ResTile>
AICORE inline void StoreResult(
    ResTile& cTile, __gm__ T* currentDst, uint32_t i, uint32_t j, uint32_t currentM, uint32_t currentN)
{
    SetFlag<PIPE_M, PIPE_FIX>(EV_STORE);
    WaitFlag<PIPE_M, PIPE_FIX>(EV_STORE);

    using DynShapeDim5 = Shape<1, 1, 1, -1, -1>;
    using DynStrideDim5 = pto::Stride<m * n, m * n, m * n, n, 1>;
    using GlobalDataOut = GlobalTensor<T, DynShapeDim5, DynStrideDim5, Layout::ND>;

    uint32_t gmOffset = i * baseM * n + j * baseN;
    GlobalDataOut dstGlobal(currentDst + gmOffset, DynShapeDim5(currentM, currentN));
    TSTORE(dstGlobal, cTile);

    SetFlag<PIPE_FIX, PIPE_MTE2>(EV_STORE_DONE);
    WaitFlag<PIPE_FIX, PIPE_MTE2>(EV_STORE_DONE);
}

AICORE inline void InitSyncFlags()
{
    SetFlag<PIPE_MTE1, PIPE_MTE2>(EV_DATA_FREE + 0);
    SetFlag<PIPE_MTE1, PIPE_MTE2>(EV_DATA_FREE + 1);
    SetFlag<PIPE_MTE1, PIPE_MTE2>(EV_SCALE_FREE);
    SetFlag<PIPE_M, PIPE_MTE1>(EV_MATMUL_DONE + 0);
    SetFlag<PIPE_M, PIPE_MTE1>(EV_MATMUL_DONE + 1);
}

AICORE inline void WaitSyncFlags()
{
    WaitFlag<PIPE_M, PIPE_MTE1>(EV_MATMUL_DONE + 0);
    WaitFlag<PIPE_M, PIPE_MTE1>(EV_MATMUL_DONE + 1);
    WaitFlag<PIPE_MTE1, PIPE_MTE2>(EV_DATA_FREE + 0);
    WaitFlag<PIPE_MTE1, PIPE_MTE2>(EV_DATA_FREE + 1);
    WaitFlag<PIPE_MTE1, PIPE_MTE2>(EV_SCALE_FREE);
}

template <
    typename T, typename U, typename X, int m, int k, int n, uint32_t singleCoreM, uint32_t singleCoreN, uint32_t baseM,
    uint32_t baseK, uint32_t baseN, uint32_t stepKa, uint32_t stepKb, typename TileMatA, typename TileMatB,
    typename TileScaleA, typename TileScaleB, typename LeftTile, typename RightTile, typename LeftScaleTile,
    typename RightScaleTile, typename ResTile>
AICORE inline void Compute(
    __gm__ U* currentSrc0, __gm__ U* currentSrc1, __gm__ X* currentSrc2, __gm__ X* currentSrc3, __gm__ T*& currentDst,
    TileMatA aMatTile[BUFFER_NUM], TileMatB bMatTile[BUFFER_NUM], TileScaleA& aScaleMatTile, TileScaleB& bScaleMatTile,
    LeftTile aTile[BUFFER_NUM], RightTile bTile[BUFFER_NUM], LeftScaleTile& aScaleTile, RightScaleTile& bScaleTile,
    ResTile& cTile)
{
    constexpr uint32_t baseScaleK = baseK / HIF4_SCALE_GROUP;
    constexpr uint32_t scaleCols = k / HIF4_SCALE_GROUP * HIF4_COL_BYTES;
    const uint32_t loopsM = (singleCoreM + baseM - 1) / baseM;
    const uint32_t loopsN = (singleCoreN + baseN - 1) / baseN;
    const uint32_t loopsK = (k + baseK - 1) / baseK;
    const uint32_t remM = singleCoreM % baseM;
    const uint32_t remN = singleCoreN % baseN;

    for (uint32_t i = 0; i < loopsM; ++i) {
        uint32_t currentM = (i == loopsM - 1 && remM > 0) ? remM : baseM;
        for (uint32_t j = 0; j < loopsN; ++j) {
            uint32_t currentN = (j == loopsN - 1 && remN > 0) ? remN : baseN;

            uint8_t mte2DBFlag = 0;
            uint8_t mte1DBFlag = 0;

            for (int buf = 0; buf < BUFFER_NUM; ++buf) {
                aMatTile[buf] = TileMatA(currentM, baseK * stepKa);
                bMatTile[buf] = TileMatB(baseK * stepKb, currentN);
                aTile[buf] = LeftTile(currentM, baseK);
                bTile[buf] = RightTile(baseK, currentN);
            }
            aScaleMatTile = TileScaleA(currentM, scaleCols);
            bScaleMatTile = TileScaleB(scaleCols, currentN);
            aScaleTile = LeftScaleTile(currentM, baseScaleK * HIF4_COL_BYTES);
            bScaleTile = RightScaleTile(baseScaleK * HIF4_COL_BYTES, currentN);

            // TASSIGN buffer addresses AFTER the dims re-assignment (the tile
            // assignment operator overwrites data_).
            TASSIGN(aMatTile[0], 0x0);
            TASSIGN(aMatTile[1], 0x0 + baseM * baseK * stepKa / 2);
            TASSIGN(bMatTile[0], 0x0 + baseM * baseK * stepKa / 2 * BUFFER_NUM);
            TASSIGN(bMatTile[1], 0x0 + baseM * baseK * stepKa / 2 * BUFFER_NUM + baseK * baseN * stepKb / 2);
            const uint32_t scaleBaseAddr =
                baseM * baseK * stepKa / 2 * BUFFER_NUM + baseK * baseN * stepKb / 2 * BUFFER_NUM;
            TASSIGN(aScaleMatTile, scaleBaseAddr);
            TASSIGN(bScaleMatTile, scaleBaseAddr + baseM * scaleCols);
            TASSIGN(aTile[0], 0x0);
            TASSIGN(aTile[1], 0x0 + L0_PINGPONG_BYTES);
            TASSIGN(bTile[0], 0x0);
            TASSIGN(bTile[1], 0x0 + L0_PINGPONG_BYTES);
            TASSIGN(aScaleTile, GetScaleAddr(aTile[0].data()));
            TASSIGN(bScaleTile, GetScaleAddr(bTile[0].data()));

            ResTile outTile(currentM, currentN);
            TASSIGN(outTile, 0x0);

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
    uint32_t baseN, uint32_t stepKa, uint32_t stepKb, uint32_t singleCoreM, uint32_t singleCoreN>
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
    InitGMOffsets<T, U, X, m, k, n, singleCoreM, singleCoreN>(
        currentSrc0, currentSrc1, currentSrc2, currentSrc3, currentDst, out, src0, src1, src2, src3);

    constexpr uint32_t baseScaleK = baseK / HIF4_SCALE_GROUP;
    constexpr uint32_t fullScaleK = k / HIF4_SCALE_GROUP;
    constexpr uint32_t scaleCols = fullScaleK * HIF4_COL_BYTES;

    using TileMatA = Tile<
        TileType::Mat, U, baseM, baseK * stepKa, BLayout::ColMajor, -1, -1, SLayout::RowMajor,
        TileConfig::fractalABSize>;
    using TileMatB = Tile<
        TileType::Mat, U, baseK * stepKb, baseN, BLayout::ColMajor, -1, -1, SLayout::RowMajor,
        TileConfig::fractalABSize>;
    using TileScaleA = Tile<TileType::Mat, X, baseM, scaleCols, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 32>;
    using TileScaleB = Tile<TileType::Mat, X, scaleCols, baseN, BLayout::ColMajor, -1, -1, SLayout::ColMajor, 32>;

    using LeftTile = TileLeft<U, baseM, baseK, -1, -1>;
    using RightTile = TileRight<U, baseK, baseN, -1, -1>;
    using LeftScaleTile = TileLeftScale<X, baseM, baseScaleK * HIF4_COL_BYTES, -1, -1>;
    using RightScaleTile = TileRightScale<X, baseScaleK * HIF4_COL_BYTES, baseN, -1, -1>;
    using ResTile = TileAcc<float, baseM, baseN, -1, -1>;

    TileMatA aMatTile[BUFFER_NUM];
    TileMatB bMatTile[BUFFER_NUM];
    TileScaleA aScaleMatTile;
    TileScaleB bScaleMatTile;
    LeftTile aTile[BUFFER_NUM];
    RightTile bTile[BUFFER_NUM];
    LeftScaleTile aScaleTile;
    RightScaleTile bScaleTile;
    ResTile cTile;

    InitSyncFlags();
    Compute<
        T, U, X, m, k, n, singleCoreM, singleCoreN, baseM, baseK, baseN, stepKa, stepKb, TileMatA, TileMatB, TileScaleA,
        TileScaleB, LeftTile, RightTile, LeftScaleTile, RightScaleTile, ResTile>(
        currentSrc0, currentSrc1, currentSrc2, currentSrc3, currentDst, aMatTile, bMatTile, aScaleMatTile,
        bScaleMatTile, aTile, bTile, aScaleTile, bScaleTile, cTile);
    WaitSyncFlags();
}

template <
    uint32_t blockDim, uint32_t m, uint32_t k, uint32_t n, uint32_t singleCoreM, uint32_t singleCoreN, uint32_t baseM,
    uint32_t baseK, uint32_t baseN, uint32_t stepKa, uint32_t stepKb>
__global__ AICORE void Hif4MatmulPerformance(
    __gm__ uint8_t* out, __gm__ uint8_t* src0, __gm__ uint8_t* src1, __gm__ uint8_t* src2, __gm__ uint8_t* src3)
{
    RunHif4MatmulDispatch<
        bfloat16_t, hifloat4x2_t, uint8_t, blockDim, m, k, n, baseM, baseK, baseN, stepKa, stepKb, singleCoreM,
        singleCoreN>(
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
    constexpr uint32_t stepKa = 1;
    constexpr uint32_t stepKb = 1;

    Hif4MatmulPerformance<blockDim, m, k, n, singleCoreM, singleCoreN, baseM, baseK, baseN, stepKa, stepKb>
        <<<blockDim, nullptr, stream>>>(out, src0, src1, src2, src3);
}

void LaunchHif4Matmul(uint8_t* out, uint8_t* src0, uint8_t* src1, uint8_t* src2, uint8_t* src3, void* stream);
