/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <pto/common/constants.hpp>
#include <pto/common/debug.h>
#include <pto/common/fifo.hpp>
#include <pto/npu/a5/TCmps.hpp>
#include <pto/npu/a5/TConcat.hpp>
#ifdef __DAV_VEC__
#include <pto/npu/a5/TCvt.hpp>
#endif
#include <pto/npu/a5/TGather.hpp>
#include <pto/npu/a5/THistogram.hpp>
#include <pto/npu/a5/Tci.hpp>
#include <pto/npu/a5/TSels.hpp>
#include <pto/pto-inst.hpp>

using namespace pto;

#ifndef INDEXER_TEST_N
#define INDEXER_TEST_N 2048
#endif
#ifndef INDEXER_TOPK
#define INDEXER_TOPK 512
#endif
#ifndef INDEXER_MERGED_SKIP_SCORE_STORE
#define INDEXER_MERGED_SKIP_SCORE_STORE 1
#endif
#ifndef INDEXER_MERGED_SKIP_MATMUL_STORE
#define INDEXER_MERGED_SKIP_MATMUL_STORE 1
#endif

// FA-style AIC / AIV split (see kernels/manual/a5/flash_atten/fa_performance_kernel.cpp); enabled when built with
// dav-c310 + REGISTER_BASE as in tests/npu/a5 pto_mix_st and this directory's CMakeLists.
#ifdef __DAV_CUBE__
constexpr bool DAV_CUBE = true;
#else
constexpr bool DAV_CUBE = false;
#endif
#ifdef __DAV_VEC__
constexpr bool DAV_VEC = true;
#else
constexpr bool DAV_VEC = false;
#endif
constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t SCALE_FACTOR = 32;
constexpr uint32_t L0_PINGPONG_BYTES = 32 * 1024; // L0A/L0B ping-pong split (32 KiB per buffer)
constexpr uint32_t mxScalePara = 8;
constexpr bool INDEXER_SIMPLE_EXPERIMENT = false; // debug mode: one 128x128 tile, one batch, no topk
constexpr uint32_t INDEXER_SIMPLE_TILE_N = 8;     // debug mode: number of N-tiles to execute
constexpr bool INDEXER_SIMPLE_FULL_BATCH = true;  // debug mode: when true, compute/store both batches
// Same split as fa_performance_kernel: constexpr DAV_CUBE / DAV_VEC from __DAV_*__, then
//   if constexpr (DAV_CUBE) { /* AIC: MTE, TEXTRACT, TMATMUL_MX, TPUSH */ }
//   if constexpr (DAV_VEC)  { /* AIV: TPOP, vector TMULS, TSTORE, TFREE */ }
// Build: dav-c310 + REGISTER_BASE (this dir CMakeLists), see tests/npu pto_mix_st.
//
// ---------------------------------------------------------------------------
// AIC (Cube / dav-c310 cube half): TLOAD, TEXTRACT, TMATMUL_MX, L0C, TMPipe producer TPUSH
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// AIV (Vector): TMPipe consumer TPOP, TMULS, Vec TSTORE, TFREE
// ---------------------------------------------------------------------------

template <typename OutTile, typename LeftTile, typename RightTile, typename LeftScaleTile, typename RightScaleTile>
AICORE inline void MatmulAcc(OutTile cTile, LeftTile aTile, RightTile bTile, LeftScaleTile aScaleTile,
                             RightScaleTile bScaleTile, uint32_t k)
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

template <typename T, typename U, typename X, int m, int k, int n, uint32_t singleCoreM, uint32_t singleCoreK,
          uint32_t singleCoreN>
AICORE inline void InitGMOffsets(__gm__ U *&currentSrc0, __gm__ U *&currentSrc1, __gm__ X *&currentSrc2,
                                 __gm__ X *&currentSrc3, __gm__ T *&currentDst, __gm__ T *out, __gm__ U *src0,
                                 __gm__ U *src1, __gm__ X *src2, __gm__ X *src3)
{
    // Work partition (SPMD-style):
    // - Each core owns a contiguous C tile of shape [singleCoreM, singleCoreN].
    // - It reads the corresponding A panel [singleCoreM, K] and B panel [K, singleCoreN].
    constexpr uint32_t mIter = m / singleCoreM;
    uint32_t mIterIdx = get_block_idx() % mIter; // get current launch core idx
    uint32_t nIterIdx = get_block_idx() / mIter;
    uint64_t gmOffsetA = mIterIdx * singleCoreM * k;
    uint64_t gmOffsetB = nIterIdx * k * singleCoreN;
    uint64_t gmOffsetScaleA = gmOffsetA / SCALE_FACTOR;
    uint64_t gmOffsetScaleB = gmOffsetB / SCALE_FACTOR;
    uint64_t gmOffsetC = mIterIdx * singleCoreM * n + nIterIdx * singleCoreN;
    currentSrc0 = src0 + gmOffsetA;
    currentSrc1 = src1 + gmOffsetB;
    currentSrc2 = src2 + gmOffsetScaleA;
    currentSrc3 = src3 + gmOffsetScaleB;
    currentDst = out + gmOffsetC;
}

template <typename T, typename U, typename X, uint32_t baseM, uint32_t baseK, uint32_t baseN, uint32_t baseScaleK,
          uint32_t stepKa, uint32_t stepKb, uint32_t stepKscaleA, uint32_t stepKscaleB, typename TileMatA,
          typename TileMatB, typename TileScaleA, typename TileScaleB, typename LeftTile, typename RightTile,
          typename LeftScaleTile, typename RightScaleTile, typename ResTile>
AICORE inline void InitBuffers(TileMatA aMatTile[BUFFER_NUM], TileMatB bMatTile[BUFFER_NUM],
                               TileScaleA aScaleMatTile[BUFFER_NUM], TileScaleB bScaleMatTile[BUFFER_NUM],
                               LeftTile aTile[BUFFER_NUM], RightTile bTile[BUFFER_NUM],
                               LeftScaleTile aScaleTile[BUFFER_NUM], RightScaleTile bScaleTile[BUFFER_NUM],
                               ResTile &cTile)
{
    // L1 staging buffers (aMatTile/bMatTile) are double-buffered for TLOAD overlap.
    TASSIGN(aMatTile[0], 0x0);
    TASSIGN(aMatTile[1], 0x0 + baseM * baseK * stepKa * sizeof(U));
    TASSIGN(bMatTile[0], 0x0 + baseM * baseK * stepKa * BUFFER_NUM * sizeof(U));
    TASSIGN(bMatTile[1], 0x0 + baseM * baseK * stepKa * BUFFER_NUM * sizeof(U) + baseK * baseN * stepKb * sizeof(U));

    constexpr uint32_t baseAddr =
        baseM * baseK * stepKa * BUFFER_NUM * sizeof(U) + baseK * baseN * stepKb * sizeof(U) * BUFFER_NUM;
    TASSIGN(aScaleMatTile[0], baseAddr);
    TASSIGN(aScaleMatTile[1], baseAddr + baseM * baseScaleK * stepKscaleA * sizeof(X));
    TASSIGN(bScaleMatTile[0], baseAddr + baseM * baseScaleK * stepKscaleA * BUFFER_NUM * sizeof(X));
    TASSIGN(bScaleMatTile[1], baseAddr + baseM * baseScaleK * stepKscaleA * BUFFER_NUM * sizeof(X) +
                                  baseScaleK * baseN * stepKscaleB * sizeof(X));

    // L0A/L0B ping-pong buffers (TEXTRACT destination).
    // Keep each per-buffer footprint <= 32 KiB to fit in a ping/pang slot.
    TASSIGN(aTile[0], 0x0);                     // L0A ping
    TASSIGN(aTile[1], 0x0 + L0_PINGPONG_BYTES); // L0A pong
    TASSIGN(bTile[0], 0x0);                     // L0B ping
    TASSIGN(bTile[1], 0x0 + L0_PINGPONG_BYTES); // L0B pong
    TASSIGN(cTile, 0x0);

    TASSIGN(aScaleTile[0], GetScaleAddr(aTile[0].data()));
    TASSIGN(aScaleTile[1], GetScaleAddr(aTile[1].data()));
    TASSIGN(bScaleTile[0], GetScaleAddr(bTile[0].data()));
    TASSIGN(bScaleTile[1], GetScaleAddr(bTile[1].data()));
}

template <uint32_t baseK, uint32_t baseScaleK, uint32_t stepKa, uint32_t stepKb, uint32_t stepKscaleA,
          uint32_t stepKscaleB, typename TileMatA, typename TileMatB, typename TileScaleA, typename TileScaleB,
          typename LeftTile, typename RightTile, typename LeftScaleTile, typename RightScaleTile, typename ResTile>
AICORE inline void MacroMatmul(uint32_t kIter, uint8_t currMte2Idx, uint8_t currMte2mxIdx, uint8_t mte1DBFlag,
                               TileMatA aMatTile[BUFFER_NUM], TileMatB bMatTile[BUFFER_NUM],
                               TileScaleA aScaleMatTile[BUFFER_NUM], TileScaleB bScaleMatTile[BUFFER_NUM],
                               LeftTile aTile[BUFFER_NUM], RightTile bTile[BUFFER_NUM],
                               LeftScaleTile aScaleTile[BUFFER_NUM], RightScaleTile bScaleTile[BUFFER_NUM],
                               ResTile &cTile)
{
    const uint32_t kModstepKa = kIter % stepKa;
    // Wait until TMATMUL is done with the current L0A/L0B buffer before overwriting it via TEXTRACT.
    WaitFlag<PIPE_M, PIPE_MTE1>(mte1DBFlag);

    // TEXTRACT stage: slice the loaded L1 panel into the baseK chunk we need this iteration.
    if (kModstepKa == 0)
        WaitFlag<PIPE_MTE2, PIPE_MTE1>(0);
    TEXTRACT(aTile[mte1DBFlag], aMatTile[currMte2Idx], 0, kModstepKa * baseK);
    TEXTRACT(aScaleTile[mte1DBFlag], aScaleMatTile[currMte2mxIdx], 0, (kIter % stepKscaleA) * baseScaleK);

    if (kModstepKa == 0)
        WaitFlag<PIPE_MTE2, PIPE_MTE1>(1);
    TEXTRACT(bTile[mte1DBFlag], bMatTile[currMte2Idx], (kIter % stepKb) * baseK, 0);
    TEXTRACT(bScaleTile[mte1DBFlag], bScaleMatTile[currMte2mxIdx], (kIter % stepKscaleB) * baseScaleK, 0);

    if ((kIter + 1) % stepKa == 0) {
        // Allow the next TLOAD to reuse this L1 slot.
        SetFlag<PIPE_MTE1, PIPE_MTE2>(currMte2Idx);
    }

    // TMATMUL stage: compute (or accumulate) into cTile.
    SetFlag<PIPE_MTE1, PIPE_M>(mte1DBFlag);
    WaitFlag<PIPE_MTE1, PIPE_M>(mte1DBFlag);
    MatmulAcc(cTile, aTile[mte1DBFlag], bTile[mte1DBFlag], aScaleTile[mte1DBFlag], bScaleTile[mte1DBFlag], kIter);
    // Signal that TMATMUL is done, so the next iteration may TEXTRACT into the other ping-pong slot.
    SetFlag<PIPE_M, PIPE_MTE1>(mte1DBFlag);
}

// AIC: one K-slice of GM->L1 TLOAD, TEXTRACT to L0, TMATMUL_MX (see FA ProcessKIteration / compute_qk).
template <typename T, typename U, typename X, int m, int k, int n, uint32_t singleCoreK, uint32_t baseM, uint32_t baseK,
          uint32_t baseN, uint32_t baseScaleK, uint32_t stepKa, uint32_t stepKb, uint32_t stepKscaleA,
          uint32_t stepKscaleB, typename TileMatA, typename TileMatB, typename TileScaleA, typename TileScaleB,
          typename LeftTile, typename RightTile, typename LeftScaleTile, typename RightScaleTile, typename ResTile>
AICORE inline void ProcessKIteration(uint32_t kIter, uint32_t i, uint32_t j, __gm__ U *currentSrc0,
                                     __gm__ U *currentSrc1, __gm__ X *currentSrc2, __gm__ X *currentSrc3,
                                     TileMatA aMatTile[BUFFER_NUM], TileMatB bMatTile[BUFFER_NUM],
                                     TileScaleA aScaleMatTile[BUFFER_NUM], TileScaleB bScaleMatTile[BUFFER_NUM],
                                     LeftTile aTile[BUFFER_NUM], RightTile bTile[BUFFER_NUM],
                                     LeftScaleTile aScaleTile[BUFFER_NUM], RightScaleTile bScaleTile[BUFFER_NUM],
                                     ResTile &cTile, uint8_t &mte2DBFlag, uint8_t &mte2mxDBFlag, uint8_t &mte1DBFlag)
{
    using GlobalDataSrc0 = GlobalTensor<U, TileShape2D<U, baseM, baseK * stepKa, Layout::ND>,
                                        BaseShape2D<U, m, k, Layout::ND>, Layout::ND>;
    using GlobalDataSrc1 = GlobalTensor<U, TileShape2D<U, baseK * stepKb, baseN, Layout::DN>,
                                        BaseShape2D<U, k, n, Layout::DN>, Layout::DN>;
    using GlobalDataSrc2 = GlobalTensor<X, TileShape2D<X, baseM, baseScaleK * stepKscaleA, Layout::MX_A_ND>,
                                        BaseShape2D<X, m, k / SCALE_FACTOR, Layout::MX_A_ND>, Layout::MX_A_ND>;
    using GlobalDataSrc3 = GlobalTensor<X, TileShape2D<X, baseScaleK * stepKscaleB, baseN, Layout::MX_B_DN>,
                                        BaseShape2D<X, k / SCALE_FACTOR, n, Layout::MX_B_DN>, Layout::MX_B_DN>;

    const uint32_t kModstepKa = kIter % stepKa;
    const uint32_t kModstepKscaleA = kIter % stepKscaleA;

    // TLOAD stage:
    // - Every stepKa iterations, load a larger [baseM, baseK * stepKa] panel into L1 and then slice it with TEXTRACT.
    // - Double buffering is driven by mte2DBFlag.

    if (kModstepKa == 0) {
        GlobalDataSrc0 gmA(currentSrc0 + i * singleCoreK * baseM + kIter * baseK);
        GlobalDataSrc1 gmB(currentSrc1 + j * singleCoreK * baseN + kIter * baseK);

        if (kModstepKscaleA == 0) {
            GlobalDataSrc2 gmScaleA(currentSrc2 + i * singleCoreK / SCALE_FACTOR * baseM + kIter * baseScaleK);
            GlobalDataSrc3 gmScaleB(currentSrc3 + j * singleCoreK / SCALE_FACTOR * baseN + kIter * baseScaleK);
            // Wait until TEXTRACT is done with this L1 buffer before reusing it.
            WaitFlag<PIPE_MTE1, PIPE_MTE2>(mte2DBFlag);
            TLOAD(aMatTile[mte2DBFlag], gmA);
            TLOAD<TileScaleA, GlobalDataSrc2>(aScaleMatTile[mte2mxDBFlag], gmScaleA);
            SetFlag<PIPE_MTE2, PIPE_MTE1>(0);
            TLOAD(bMatTile[mte2DBFlag], gmB);
            TLOAD<TileScaleB, GlobalDataSrc3>(bScaleMatTile[mte2mxDBFlag], gmScaleB);
            mte2mxDBFlag = (mte2mxDBFlag == 0) ? 1 : 0;
        } else {
            WaitFlag<PIPE_MTE1, PIPE_MTE2>(mte2DBFlag);
            TLOAD(aMatTile[mte2DBFlag], gmA);
            SetFlag<PIPE_MTE2, PIPE_MTE1>(0);
            TLOAD(bMatTile[mte2DBFlag], gmB);
        }
        SetFlag<PIPE_MTE2, PIPE_MTE1>(1);
        mte2DBFlag = (mte2DBFlag == 0) ? 1 : 0;
    }

    const uint32_t currMte2Idx = (mte2DBFlag == 0) ? 1 : 0;     // mte2DBFlag reversed
    const uint32_t currMte2mxIdx = (mte2mxDBFlag == 0) ? 1 : 0; // mte2mxDBFlag reversed

    MacroMatmul<baseK, baseScaleK, stepKa, stepKb, stepKscaleA, stepKscaleB, TileMatA, TileMatB, TileScaleA, TileScaleB,
                LeftTile, RightTile, LeftScaleTile, RightScaleTile, ResTile>(
        kIter, currMte2Idx, currMte2mxIdx, mte1DBFlag, aMatTile, bMatTile, aScaleMatTile, bScaleMatTile, aTile, bTile,
        aScaleTile, bScaleTile, cTile);
    mte1DBFlag = (mte1DBFlag == 0) ? 1 : 0;
}

// AIC: L0C -> TMPipe producer (Acc tile into VEC_FIFO backing UB). Matches FA cube-side TPUSH to qkPipe / pvPipe.
template <typename ResTile, typename CvOutPipeT>
AICORE inline void IndexerAic_StoreTpushToCvFifo(ResTile &cTile, CvOutPipeT &cvPipe)
{
    if constexpr (DAV_CUBE) {
        cvPipe.prod.setTileId(0, 0);
        cvPipe.prod.setAllocateStatus(true);
        cvPipe.prod.setRecordStatus(true);
        cvPipe.prod.setEntryOffset(0);
        TPUSH(cTile, cvPipe);
    }
}

// AIV: wait / TPOP to Vec UB, GM store, release FIFO. Matches FA vector-side TPOP from qkPipe / softmax output.
template <int m, int n, uint32_t baseM, uint32_t baseN, bool kFusePostProcess, typename VecF, typename CvOutPipeT>
AICORE inline void IndexerAiv_StoreTpopTstore(__gm__ float *currentDst, uint32_t i, uint32_t j, VecF &vecForStore,
                                               CvOutPipeT &cvPipe, __gm__ float *postScale, __gm__ float *scoreOut,
                                               __gm__ uint16_t *scoreOutBf16)
{
#ifdef __DAV_VEC__
        // V1C1_VEC0 mode: vec1 should only participate in FIFO handshake.
        // Keep single UB address, but avoid vec1 touching data path (TPOP/TMULS/TSTORE).
        if (get_subblockid() != 0) {
            cvPipe.cons.setWaitStatus(true);
            cvPipe.cons.setFreeStatus(true);
            cvPipe.cons.wait();
            cvPipe.cons.free();
            return;
        }

        cvPipe.cons.setTileId(0, 0);
        cvPipe.cons.setWaitStatus(true);
        cvPipe.cons.setFreeStatus(false);
        cvPipe.cons.setEntryOffset(0);
        TPOP(vecForStore, cvPipe);
        constexpr bool kNeedMatmulStore = !kFusePostProcess || (INDEXER_MERGED_SKIP_MATMUL_STORE == 0);
        if constexpr (kNeedMatmulStore) {
            using NDValidShapeC = TileShape2D<float, baseM, baseN, Layout::ND>;
            using NDWholeShapeC = BaseShape2D<float, m, n, Layout::ND>;
            using GlobalDataOut = GlobalTensor<float, NDValidShapeC, NDWholeShapeC, Layout::ND>;
            GlobalDataOut dstGlobal(currentDst + i * static_cast<uint64_t>(baseM) * n + j * baseN);
            // Ensure MTE3 GM store consumes vec buffer before FIFO free/next TPUSH overwrite.
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            TSTORE(dstGlobal, vecForStore);
            set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        }
        if constexpr (kFusePostProcess) {
            static_assert(baseM % 2 == 0, "Fused postprocess requires even baseM.");
            if (postScale != nullptr && scoreOutBf16 != nullptr) {
                constexpr uint32_t kHeads = baseM / 2;
                constexpr uint64_t kUbVecBase = 0x30000;
                constexpr uint64_t kUbMat1 = kUbVecBase + static_cast<uint64_t>(kHeads) * baseN * sizeof(float);
                constexpr uint64_t kUbReduce0 = 0x20000;
                constexpr uint64_t kUbReduce1 = kUbReduce0 + static_cast<uint64_t>(32) * baseN * sizeof(float);
                constexpr uint64_t kUbScore0 = kUbReduce1 + static_cast<uint64_t>(32) * baseN * sizeof(float);
                constexpr uint64_t kUbScore1 = kUbScore0 + static_cast<uint64_t>(baseN) * sizeof(float);
                constexpr uint64_t kUbScale0 = kUbScore1 + static_cast<uint64_t>(baseN) * sizeof(float);
                constexpr uint64_t kUbScale1 = kUbScale0 + static_cast<uint64_t>(kHeads) * sizeof(float);
                constexpr uint64_t kUbScoreBf160 = kUbScale1 + static_cast<uint64_t>(kHeads) * sizeof(float);
                constexpr uint64_t kUbScoreBf161 = kUbScoreBf160 + static_cast<uint64_t>(baseN) * sizeof(uint16_t);

                using HalfMatTile = Tile<TileType::Vec, float, kHeads, baseN, BLayout::RowMajor, -1, -1>;
                using ScaleTile = Tile<TileType::Vec, float, kHeads, 1, BLayout::ColMajor, -1, -1>;
                using ReduceTmpTile = Tile<TileType::Vec, float, 32, baseN, BLayout::RowMajor, -1, -1>;
                using ScoreTile = Tile<TileType::Vec, float, 1, baseN, BLayout::RowMajor, -1, -1>;
                using ScoreBf16Tile = Tile<TileType::Vec, bfloat16_t, 1, baseN, BLayout::RowMajor, -1, -1>;
                using ScaleGlobal =
                    GlobalTensor<float, pto::Shape<1, 1, 1, kHeads, 1>, pto::Stride<1, 1, 1, 1, 1>, pto::Layout::DN>;
                using ScoreBf16Global =
                    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, baseN>, pto::Stride<n, n, n, n, 1>>;

                HalfMatTile mat0(kHeads, baseN);
                HalfMatTile mat1(kHeads, baseN);
                ScaleTile scale0(kHeads, 1);
                ScaleTile scale1(kHeads, 1);
                ReduceTmpTile reduce0(32, baseN);
                ReduceTmpTile reduce1(32, baseN);
                ScoreTile score0(1, baseN);
                ScoreTile score1(1, baseN);
                ScoreBf16Tile scoreBf160(1, baseN);
                ScoreBf16Tile scoreBf161(1, baseN);

                TASSIGN(mat0, kUbVecBase);
                TASSIGN(mat1, kUbMat1);
                TASSIGN(scale0, kUbScale0);
                TASSIGN(scale1, kUbScale1);
                TASSIGN(reduce0, kUbReduce0);
                TASSIGN(reduce1, kUbReduce1);
                TASSIGN(score0, kUbScore0);
                TASSIGN(score1, kUbScore1);
                TASSIGN(scoreBf160, kUbScoreBf160);
                TASSIGN(scoreBf161, kUbScoreBf161);

                mat0.SetValidRow(kHeads);
                mat0.SetValidCol(baseN);
                mat1.SetValidRow(kHeads);
                mat1.SetValidCol(baseN);
                scale0.SetValidRow(kHeads);
                scale0.SetValidCol(1);
                scale1.SetValidRow(kHeads);
                scale1.SetValidCol(1);
                reduce0.SetValidRow(32);
                reduce0.SetValidCol(baseN);
                reduce1.SetValidRow(32);
                reduce1.SetValidCol(baseN);
                score0.SetValidRow(1);
                score0.SetValidCol(baseN);
                score1.SetValidRow(1);
                score1.SetValidCol(baseN);
                scoreBf160.SetValidRow(1);
                scoreBf160.SetValidCol(baseN);
                scoreBf161.SetValidRow(1);
                scoreBf161.SetValidCol(baseN);

                ScaleGlobal scaleGlobal0(postScale);
                ScaleGlobal scaleGlobal1(postScale + kHeads);
                TLOAD(scale0, scaleGlobal0);
                TLOAD(scale1, scaleGlobal1);

                TRELU(mat0, mat0);
                TRELU(mat1, mat1);
                TROWEXPANDMUL(mat0, mat0, scale0);
                TROWEXPANDMUL(mat1, mat1, scale1);
                TCOLSUM(score0, mat0, reduce0, true);
                if constexpr (!INDEXER_SIMPLE_EXPERIMENT || INDEXER_SIMPLE_FULL_BATCH) {
                    TCOLSUM(score1, mat1, reduce1, true);
                }

                TCVT(scoreBf160, score0, RoundMode::CAST_TRUNC);
                if constexpr (!INDEXER_SIMPLE_EXPERIMENT || INDEXER_SIMPLE_FULL_BATCH) {
                    TCVT(scoreBf161, score1, RoundMode::CAST_TRUNC);
                }

                const uint32_t colBase = j * baseN;
                ScoreBf16Global scoreBf16Global0(reinterpret_cast<__gm__ bfloat16_t *>(scoreOutBf16 + colBase));
                ScoreBf16Global scoreBf16Global1(reinterpret_cast<__gm__ bfloat16_t *>(scoreOutBf16 + n + colBase));
                if (scoreOut != nullptr) {
                    using ScoreGlobal =
                        GlobalTensor<float, pto::Shape<1, 1, 1, 1, baseN>, pto::Stride<n, n, n, n, 1>>;
                    ScoreGlobal scoreOutGlobal0(scoreOut + colBase);
                    ScoreGlobal scoreOutGlobal1(scoreOut + n + colBase);
                    TSTORE(scoreOutGlobal0, score0);
                    if constexpr (!INDEXER_SIMPLE_EXPERIMENT || INDEXER_SIMPLE_FULL_BATCH) {
                        TSTORE(scoreOutGlobal1, score1);
                    }
                }
#ifndef __PTO_AUTO__
                set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
                wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
#endif
                TSTORE(scoreBf16Global0, scoreBf160);
                if constexpr (!INDEXER_SIMPLE_EXPERIMENT || INDEXER_SIMPLE_FULL_BATCH) {
                    TSTORE(scoreBf16Global1, scoreBf161);
                }
#ifndef __PTO_AUTO__
                set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
                wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
#endif
            }
        }
        cvPipe.cons.setFreeStatus(true);
        TFREE(cvPipe);
#else
    (void)currentDst;
    (void)i;
    (void)j;
    (void)vecForStore;
    (void)cvPipe;
    (void)postScale;
    (void)scoreOut;
    (void)scoreOutBf16;
#endif
}

template <int m, int n, uint32_t baseM, uint32_t baseN, bool kFusePostProcess, typename ResTile, typename VecF,
          typename CvOutPipeT>
AICORE inline void StoreResultWithCvPipe(ResTile &cTile, __gm__ float *currentDst, uint32_t i, uint32_t j,
                                         VecF &vecForStore, CvOutPipeT &cvPipe, __gm__ float *postScale,
                                         __gm__ float *scoreOut, __gm__ uint16_t *scoreOutBf16)
{
    if constexpr (DAV_CUBE) {
        SetFlag<PIPE_M, PIPE_FIX>(0);
        WaitFlag<PIPE_M, PIPE_FIX>(0);
    }

    if constexpr (DAV_CUBE) {
        IndexerAic_StoreTpushToCvFifo(cTile, cvPipe);
    }
    if constexpr (DAV_VEC) {
        IndexerAiv_StoreTpopTstore<m, n, baseM, baseN, kFusePostProcess>(
            currentDst, i, j, vecForStore, cvPipe, postScale, scoreOut, scoreOutBf16);
    }

    if constexpr (DAV_CUBE) {
        SetFlag<PIPE_FIX, PIPE_M>(0);
        WaitFlag<PIPE_FIX, PIPE_M>(0);
    }
}

template <int m, int n, uint32_t baseM, uint32_t baseN, typename ResTile>
AICORE inline void StoreResultFallback(ResTile &cTile, __gm__ float *currentDst, uint32_t i, uint32_t j)
{
    if constexpr (DAV_CUBE) {
        SetFlag<PIPE_M, PIPE_FIX>(0);
        WaitFlag<PIPE_M, PIPE_FIX>(0);
    }

    // Non-DAV fallback: keep functional output path.
    using NDValidShapeC = TileShape2D<float, baseM, baseN, Layout::ND>;
    using NDWholeShapeC = BaseShape2D<float, m, n, Layout::ND>;
    using GlobalDataOut = GlobalTensor<float, NDValidShapeC, NDWholeShapeC, Layout::ND>;
    GlobalDataOut dstGlobal(currentDst + i * static_cast<uint64_t>(baseM) * n + j * baseN);
    constexpr uint64_t kHalfF32Bits = 0x3f000000ULL;
    TSTORE(dstGlobal, cTile, kHalfF32Bits);

    if constexpr (DAV_CUBE) {
        SetFlag<PIPE_FIX, PIPE_M>(0);
        WaitFlag<PIPE_FIX, PIPE_M>(0);
    }
}

template <typename T, typename U, typename X, int m, int k, int n, uint32_t singleCoreM, uint32_t singleCoreK,
          uint32_t singleCoreN, uint32_t baseM, uint32_t baseK, uint32_t baseN, uint32_t baseScaleK, uint32_t stepKa,
          uint32_t stepKb, uint32_t stepKscaleA, uint32_t stepKscaleB, typename TileMatA, typename TileMatB,
          typename TileScaleA, typename TileScaleB, typename LeftTile, typename RightTile, typename LeftScaleTile,
          typename RightScaleTile, typename ResTile, bool kFusePostProcess>
AICORE inline void Compute(__gm__ U *currentSrc0, __gm__ U *currentSrc1, __gm__ X *currentSrc2, __gm__ X *currentSrc3,
                           __gm__ T *&currentDst, TileMatA aMatTile[BUFFER_NUM], TileMatB bMatTile[BUFFER_NUM],
                           TileScaleA aScaleMatTile[BUFFER_NUM], TileScaleB bScaleMatTile[BUFFER_NUM],
                           LeftTile aTile[BUFFER_NUM], RightTile bTile[BUFFER_NUM],
                           LeftScaleTile aScaleTile[BUFFER_NUM], RightScaleTile bScaleTile[BUFFER_NUM], ResTile &cTile,
                           __gm__ float *postScale, __gm__ float *scoreOut, __gm__ uint16_t *scoreOutBf16)
{
    uint8_t mte2DBFlag = 0, mte2mxDBFlag = 0, mte1DBFlag = 0;
    if constexpr (DAV_CUBE || DAV_VEC) {
        using VecF = Tile<TileType::Vec, float, baseM, baseN, BLayout::RowMajor, baseM, baseN>;
        constexpr uint8_t kCvStoreFlag = 8;
        constexpr uint8_t kCvFifoDepth = 1;
        constexpr uint8_t kCvFifoSyncT = 1;
        constexpr uint32_t kUbVecForTp = 0x30000;
        using CvOutPipe = TMPipe<kCvStoreFlag, FIFOType::VEC_FIFO, kCvFifoDepth, kCvFifoSyncT, ResTile, VecF, false, 0,
                                 VecCubeRatio::V1C1_VEC0>;
        VecF vecForStore;
        TASSIGN(vecForStore, kUbVecForTp);
        CvOutPipe cvPipe(static_cast<uint32_t>(kUbVecForTp));

        constexpr uint32_t kIterM = singleCoreM / baseM;
        constexpr uint32_t kTotalIterN = singleCoreN / baseN;
        constexpr uint32_t kIterN = INDEXER_SIMPLE_EXPERIMENT
                                        ? ((INDEXER_SIMPLE_TILE_N < kTotalIterN) ? INDEXER_SIMPLE_TILE_N : kTotalIterN)
                                        : kTotalIterN;
        for (uint32_t i = 0; i < kIterM; i++) {
            for (uint32_t j = 0; j < kIterN; j++) {
                for (uint32_t kIter = 0; kIter < singleCoreK / baseK; kIter++) {
                    if constexpr (DAV_CUBE) {
                        ProcessKIteration<T, U, X, m, k, n, singleCoreK, baseM, baseK, baseN, baseScaleK, stepKa, stepKb,
                                          stepKscaleA, stepKscaleB, TileMatA, TileMatB, TileScaleA, TileScaleB, LeftTile,
                                          RightTile, LeftScaleTile, RightScaleTile, ResTile>(
                            kIter, i, j, currentSrc0, currentSrc1, currentSrc2, currentSrc3, aMatTile, bMatTile,
                            aScaleMatTile, bScaleMatTile, aTile, bTile, aScaleTile, bScaleTile, cTile, mte2DBFlag,
                            mte2mxDBFlag, mte1DBFlag);
                    }
                }
                StoreResultWithCvPipe<m, n, baseM, baseN, kFusePostProcess, ResTile, VecF, CvOutPipe>(
                    cTile, currentDst, i, j, vecForStore, cvPipe, postScale, scoreOut, scoreOutBf16);
            }
        }
    } else {
        for (uint32_t i = 0; i < singleCoreM / baseM; i++) {
            for (uint32_t j = 0; j < singleCoreN / baseN; j++) {
                for (uint32_t kIter = 0; kIter < singleCoreK / baseK; kIter++) {
                    if constexpr (DAV_CUBE) {
                        ProcessKIteration<T, U, X, m, k, n, singleCoreK, baseM, baseK, baseN, baseScaleK, stepKa, stepKb,
                                          stepKscaleA, stepKscaleB, TileMatA, TileMatB, TileScaleA, TileScaleB, LeftTile,
                                          RightTile, LeftScaleTile, RightScaleTile, ResTile>(
                            kIter, i, j, currentSrc0, currentSrc1, currentSrc2, currentSrc3, aMatTile, bMatTile,
                            aScaleMatTile, bScaleMatTile, aTile, bTile, aScaleTile, bScaleTile, cTile, mte2DBFlag,
                            mte2mxDBFlag, mte1DBFlag);
                    }
                }
                StoreResultFallback<m, n, baseM, baseN, ResTile>(cTile, currentDst, i, j);
            }
        }
    }
}

AICORE inline void InitSyncFlags()
{
    if constexpr (DAV_CUBE) {
        // supplement first sync instr for reverse sync in ProcessKIteration
        SetFlag<PIPE_MTE1, PIPE_MTE2>(0);
        SetFlag<PIPE_MTE1, PIPE_MTE2>(1);
        SetFlag<PIPE_M, PIPE_MTE1>(0);
        SetFlag<PIPE_M, PIPE_MTE1>(1);
    }
}

AICORE inline void WaitSyncFlags()
{
    if constexpr (DAV_CUBE) {
        // supplement last sync instr for reverse sync in ProcessKIteration
        WaitFlag<PIPE_M, PIPE_MTE1>(0);
        WaitFlag<PIPE_M, PIPE_MTE1>(1);
        WaitFlag<PIPE_MTE1, PIPE_MTE2>(0);
        WaitFlag<PIPE_MTE1, PIPE_MTE2>(1);
    }
}

template <typename T, typename U, typename X, uint32_t blockDim, int m, int k, int n, uint32_t singleCoreM,
          uint32_t singleCoreK, uint32_t singleCoreN, uint32_t baseM, uint32_t baseK, uint32_t baseN, uint32_t stepM,
          uint32_t stepKa, uint32_t stepKb, uint32_t stepN, bool kFusePostProcess = false>
AICORE inline void RunMxMatmul(__gm__ T *out, __gm__ U *src0, __gm__ U *src1, __gm__ X *src2, __gm__ X *src3,
                               __gm__ float *postScale = nullptr, __gm__ float *scoreOut = nullptr,
                               __gm__ uint16_t *scoreOutBf16 = nullptr)
{
    __gm__ U *currentSrc0 = nullptr;
    __gm__ U *currentSrc1 = nullptr;
    __gm__ X *currentSrc2 = nullptr;
    __gm__ X *currentSrc3 = nullptr;
    __gm__ T *currentDst = nullptr;
    InitGMOffsets<T, U, X, m, k, n, singleCoreM, singleCoreK, singleCoreN>(
        currentSrc0, currentSrc1, currentSrc2, currentSrc3, currentDst, out, src0, src1, src2, src3);

    constexpr uint32_t baseScaleK = baseK / SCALE_FACTOR;
    constexpr uint32_t stepKscaleA = stepKa * mxScalePara;
    constexpr uint32_t stepKscaleB = stepKb * mxScalePara;

    using TileMatA =
        Tile<TileType::Mat, U, baseM, baseK * stepKa, BLayout::ColMajor, baseM, baseK * stepKa, SLayout::RowMajor>;
    using TileMatB =
        Tile<TileType::Mat, U, baseK * stepKb, baseN, BLayout::RowMajor, baseK * stepKb, baseN, SLayout::ColMajor>;
    using TileScaleA = Tile<TileType::Mat, X, baseM, baseScaleK * stepKscaleA, BLayout::RowMajor, baseM,
                            baseScaleK * stepKscaleA, SLayout::RowMajor, 32>;
    using TileScaleB = Tile<TileType::Mat, X, baseScaleK * stepKscaleB, baseN, BLayout::ColMajor,
                            baseScaleK * stepKscaleB, baseN, SLayout::ColMajor, 32>;
    TileMatA aMatTile[BUFFER_NUM];
    TileMatB bMatTile[BUFFER_NUM];
    TileScaleA aScaleMatTile[BUFFER_NUM];
    TileScaleB bScaleMatTile[BUFFER_NUM];

    using LeftTile = TileLeft<U, baseM, baseK, baseM, baseK>;
    using RightTile = TileRight<U, baseK, baseN, baseK, baseN>;
    using LeftScaleTile = TileLeftScale<X, baseM, baseScaleK, baseM, baseScaleK>;
    using RightScaleTile = TileRightScale<X, baseScaleK, baseN, baseScaleK, baseN>;
    using ResTile = TileAcc<float, baseM, baseN, baseM, baseN>;

    LeftTile aTile[BUFFER_NUM];
    RightTile bTile[BUFFER_NUM];
    LeftScaleTile aScaleTile[BUFFER_NUM];
    RightScaleTile bScaleTile[BUFFER_NUM];
    ResTile cTile;

    InitBuffers<T, U, X, baseM, baseK, baseN, baseScaleK, stepKa, stepKb, stepKscaleA, stepKscaleB, TileMatA, TileMatB,
                TileScaleA, TileScaleB, LeftTile, RightTile, LeftScaleTile, RightScaleTile, ResTile>(
        aMatTile, bMatTile, aScaleMatTile, bScaleMatTile, aTile, bTile, aScaleTile, bScaleTile, cTile);

    if constexpr (DAV_CUBE) {
        InitSyncFlags();
    }

    Compute<T, U, X, m, k, n, singleCoreM, singleCoreK, singleCoreN, baseM, baseK, baseN, baseScaleK, stepKa, stepKb,
            stepKscaleA, stepKscaleB, TileMatA, TileMatB, TileScaleA, TileScaleB, LeftTile, RightTile, LeftScaleTile,
            RightScaleTile, ResTile, kFusePostProcess>(currentSrc0, currentSrc1, currentSrc2, currentSrc3, currentDst,
                                                       aMatTile, bMatTile, aScaleMatTile, bScaleMatTile, aTile, bTile,
                                                       aScaleTile, bScaleTile, cTile, postScale, scoreOut, scoreOutBf16);

    if constexpr (DAV_CUBE) {
        WaitSyncFlags();
    }
}

template <uint32_t blockDim, uint32_t m, uint32_t k, uint32_t n, uint32_t singleCoreM, uint32_t singleCoreK,
          uint32_t singleCoreN, uint32_t baseM, uint32_t baseK, uint32_t baseN, uint32_t stepM, uint32_t stepKa,
          uint32_t stepKb, uint32_t stepN>
__global__ AICORE void MxMatmulPerformance(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1,
                                           __gm__ uint8_t *src2, __gm__ uint8_t *src3, __gm__ float *postScale,
                                           __gm__ float *scoreOut, __gm__ uint32_t *outIdx)
{
    (void)postScale;
    (void)scoreOut;
    (void)outIdx;
    RunMxMatmul<float, float8_e5m2_t, float8_e8m0_t, blockDim, m, k, n, singleCoreM, singleCoreK, singleCoreN, baseM,
                baseK, baseN, stepM, stepKa, stepKb, stepN>(reinterpret_cast<__gm__ float *>(out),
                                                            reinterpret_cast<__gm__ float8_e5m2_t *>(src0),
                                                            reinterpret_cast<__gm__ float8_e5m2_t *>(src1),
                                                            reinterpret_cast<__gm__ float8_e8m0_t *>(src2),
                                                            reinterpret_cast<__gm__ float8_e8m0_t *>(src3));
}

template <uint32_t kBatch, uint32_t kHeads, uint32_t kLength>
AICORE inline void PostProcessScoreImpl(__gm__ float *matmulOut, __gm__ float *postScale, __gm__ float *scoreOut,
                                        __gm__ uint16_t *scoreOutBf16)
{
#ifdef __DAV_VEC__
    if (get_subblockid() != 0 || get_block_idx() != 0) {
        return;
    }

    constexpr uint32_t kChunkCols = 128;
    constexpr uint32_t kReduceTmpRows = (kHeads + 1) / 2;
    constexpr uint64_t kUbMat = 0x20000;
    constexpr uint64_t kUbScale = kUbMat + static_cast<uint64_t>(kHeads) * kChunkCols * sizeof(float);
    constexpr uint64_t kUbWeighted = kUbScale + static_cast<uint64_t>(kHeads) * sizeof(float);
    constexpr uint64_t kUbReduceTmp = kUbWeighted + static_cast<uint64_t>(kHeads) * kChunkCols * sizeof(float);
    constexpr uint64_t kUbScore = kUbReduceTmp + static_cast<uint64_t>(kReduceTmpRows) * kChunkCols * sizeof(float);
    constexpr uint64_t kUbScoreBf16 = kUbScore + static_cast<uint64_t>(kChunkCols) * sizeof(float);

    using MatTile = Tile<TileType::Vec, float, kHeads, kChunkCols, BLayout::RowMajor, -1, -1>;
    using ScaleTile = Tile<TileType::Vec, float, kHeads, 1, BLayout::ColMajor, -1, -1>;
    using WeightedTile = Tile<TileType::Vec, float, kHeads, kChunkCols, BLayout::RowMajor, -1, -1>;
    using ReduceTmpTile = Tile<TileType::Vec, float, 32, kChunkCols, BLayout::RowMajor, -1, -1>;
    using ScoreTile = Tile<TileType::Vec, float, 1, kChunkCols, BLayout::RowMajor, -1, -1>;
    using ScoreBf16Tile = Tile<TileType::Vec, bfloat16_t, 1, kChunkCols, BLayout::RowMajor, -1, -1>;
    using MatGlobal =
        GlobalTensor<float, pto::Shape<1, 1, 1, kHeads, kChunkCols>, pto::Stride<kLength, kLength, kLength, kLength, 1>>;
    using ScaleGlobal =
        GlobalTensor<float, pto::Shape<1, 1, 1, kHeads, 1>, pto::Stride<1, 1, 1, 1, 1>, pto::Layout::DN>;
    using ScoreGlobal = GlobalTensor<float, pto::Shape<1, 1, 1, 1, kChunkCols>, pto::Stride<kLength, kLength, kLength, kLength, 1>>;
    using ScoreBf16Global =
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, kChunkCols>, pto::Stride<kLength, kLength, kLength, kLength, 1>>;

    MatTile matTile(kHeads, kChunkCols);
    ScaleTile scaleTile(kHeads, 1);
    WeightedTile weightedTile(kHeads, kChunkCols);
    ReduceTmpTile reduceTmpTile(32, kChunkCols);
    ScoreTile scoreTile(1, kChunkCols);
    ScoreBf16Tile scoreBf16Tile(1, kChunkCols);
    TASSIGN(matTile, kUbMat);
    TASSIGN(scaleTile, kUbScale);
    TASSIGN(weightedTile, kUbWeighted);
    TASSIGN(reduceTmpTile, kUbReduceTmp);
    TASSIGN(scoreTile, kUbScore);
    TASSIGN(scoreBf16Tile, kUbScoreBf16);
#ifndef __PTO_AUTO__
    // Reverse dependency: MTE2 must wait until vector finishes consuming mat/scale buffers.
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
#endif

    for (uint32_t b = 0; b < kBatch; ++b) {
        scaleTile.SetValidRow(kHeads);
        scaleTile.SetValidCol(1);
        ScaleGlobal scaleGlobal(postScale + static_cast<uint64_t>(b) * kHeads);
#ifndef __PTO_AUTO__
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
#endif
        TLOAD(scaleTile, scaleGlobal);
#ifndef __PTO_AUTO__
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
#endif
        for (uint32_t n0 = 0; n0 < kLength; n0 += kChunkCols) {
            uint32_t validCols = (n0 + kChunkCols <= kLength) ? kChunkCols : (kLength - n0);
            matTile.SetValidRow(kHeads);
            matTile.SetValidCol(validCols);
            weightedTile.SetValidRow(kHeads);
            weightedTile.SetValidCol(validCols);
            reduceTmpTile.SetValidRow(kReduceTmpRows);
            reduceTmpTile.SetValidCol(validCols);
            scoreTile.SetValidRow(1);
            scoreTile.SetValidCol(validCols);
            scoreBf16Tile.SetValidRow(1);
            scoreBf16Tile.SetValidCol(validCols);

            MatGlobal matGlobal(matmulOut + static_cast<uint64_t>(b) * kHeads * kLength + n0);
#ifndef __PTO_AUTO__
            wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
#endif
            TLOAD(matTile, matGlobal);
#ifndef __PTO_AUTO__
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif
            TRELU(matTile, matTile);
            TROWEXPANDMUL(weightedTile, matTile, scaleTile);
            TCOLSUM(scoreTile, weightedTile, reduceTmpTile, true);
            TCVT(scoreBf16Tile, scoreTile, RoundMode::CAST_TRUNC);
#ifndef __PTO_AUTO__
            set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
#endif

            ScoreGlobal dstGlobal(scoreOut + static_cast<uint64_t>(b) * kLength + n0);
            ScoreBf16Global dstBf16Global(
                reinterpret_cast<__gm__ bfloat16_t *>(scoreOutBf16 + static_cast<uint64_t>(b) * kLength + n0));
#ifndef __PTO_AUTO__
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
#endif
            TSTORE(dstGlobal, scoreTile);
            TSTORE(dstBf16Global, scoreBf16Tile);
        }
    }
#else
    (void)matmulOut;
    (void)postScale;
    (void)scoreOut;
    (void)scoreOutBf16;
#endif
}

template <uint32_t kLength, uint32_t K>
AICORE inline void TopKFromScoreImpl(__gm__ uint16_t *scoreOutBf16, __gm__ uint32_t *outIdx)
{
    #ifdef __DAV_VEC__
    if constexpr (!DAV_VEC) {
        return;
    }
    if (get_subblockid() != 0 || get_block_idx() != 0) {
        return;
    }
    static_assert(K <= kLength, "TopK must be <= score length.");
    constexpr uint32_t kBatch = 2;
    constexpr uint32_t kBinNum = 256;
    constexpr uint64_t kUbFullKeys = 0x00000;
    constexpr uint64_t kUbTileHist = 0x20000;
    constexpr uint64_t kUbChistMSB = 0x21000;
    constexpr uint64_t kUbChistLSB = 0x22000;
    constexpr uint64_t kUbIdxFilter = 0x23000;
    constexpr uint64_t kWinnerUbMask = 0x24000;
    constexpr uint64_t kWinnerUbIdx = 0x24400;
    constexpr uint64_t kWinnerUbGather = 0x24800;
    constexpr uint64_t kWinnerUbTmp = 0x24C00;
    constexpr uint64_t kWinnerUbRowMinDst = 0x25000;
    constexpr uint64_t kWinnerUbRowMinTmp = 0x25400;
    constexpr uint64_t kWinnerUbSelMask = 0x25800;
    constexpr uint64_t kWinnerUbSelZero = 0x25840;
    constexpr uint64_t kWinnerUbU32One = 0x25880;
    constexpr uint64_t kWinnerUbSelOut = 0x25900;
    constexpr uint64_t kWinnerUbTselTmp = 0x25A00;
    constexpr uint64_t kRemainUbTopk = 0x25E00;
    constexpr uint64_t kRemainUbCw = 0x25F00;
    constexpr uint64_t kRemainUbOut = 0x26000;
    constexpr uint64_t kMsbWinnerSavedUb = 0x25D80;
    constexpr uint64_t kFullGatherGtDst = 0x30000;
    constexpr uint64_t kFullGatherEqDst = 0x38000;
    constexpr uint64_t kChunkConcatGt = 0x28000;
    constexpr uint64_t kChunkConcatEq = 0x28040;
    constexpr uint64_t kGatherUbTmp = 0x29000;
    // Keep ordered-key temporary buffers in high UB region to avoid aliasing with
    // phase2/phase5 buffers when kLength expands to 2048.
    constexpr uint64_t kUbNegKeys = 0x3A000;
    constexpr uint64_t kUbPosKeys = kUbNegKeys + static_cast<uint64_t>(kLength) * sizeof(uint16_t);
    constexpr uint64_t kUbSignMask = kUbPosKeys + static_cast<uint64_t>(kLength) * sizeof(uint16_t);
    constexpr uint64_t kUbSignTmp = kUbSignMask + static_cast<uint64_t>(kLength) * sizeof(uint8_t);
    constexpr uint64_t kUbVecLimit = 0x40000;
    static_assert(kUbSignTmp + 32 <= kUbVecLimit, "TopK UB layout exceeds vector UB limit.");
    constexpr uint64_t kUbMerged = 0x00000;

    constexpr uint16_t kIdxAlignedRows = ((sizeof(uint8_t) + BLOCK_BYTE_SIZE - 1) / BLOCK_BYTE_SIZE) * BLOCK_BYTE_SIZE;
    constexpr int kGatherConcatRows = (1 * static_cast<int>(sizeof(uint32_t)) < 32) ? (32 / static_cast<int>(sizeof(uint32_t)))
                                                                                     : 1;
    constexpr int cmpVCol = (kLength + 7) / 8;
    constexpr int cmpCol = (cmpVCol + 31) / 32 * 32;

    using InTileU16 = Tile<TileType::Vec, uint16_t, 1, kLength, BLayout::RowMajor, -1, -1>;
    using HistTile = Tile<TileType::Vec, uint32_t, 1, kBinNum, BLayout::RowMajor, -1, -1>;
    using WinnerLaneTile = Tile<TileType::Vec, uint32_t, 1, kBinNum, BLayout::RowMajor, -1, -1>;
    using WinnerBinTile = Tile<TileType::Vec, uint32_t, 1, 32, BLayout::RowMajor, -1, -1>;
    using MaskCmpTile = Tile<TileType::Vec, uint8_t, 1, kBinNum, BLayout::RowMajor, -1, -1>;
    using IdxU32Tile = Tile<TileType::Vec, uint32_t, 1, kBinNum, BLayout::RowMajor, -1, -1>;
    using TmpSelsTile = Tile<TileType::Vec, uint8_t, 1, 32, BLayout::RowMajor, -1, -1>;
    using RowMinDstTile = Tile<TileType::Vec, uint32_t, 1, 16, BLayout::RowMajor, -1, -1>;
    using RowMinTmpTile = Tile<TileType::Vec, uint32_t, 1, kBinNum, BLayout::RowMajor, -1, -1>;
    using SelMaskRowTile = Tile<TileType::Vec, uint8_t, 1, 32, BLayout::RowMajor, -1, -1>;
    using GatherIdxU32Tile = Tile<TileType::Vec, uint32_t, 1, 32, BLayout::RowMajor, -1, -1>;
    using RemainKTile = Tile<TileType::Vec, uint32_t, 1, 32, BLayout::RowMajor, -1, -1>;
    using PackedU16Tile = Tile<TileType::Vec, uint16_t, 1, 32, BLayout::RowMajor, -1, -1>;
    using IdxFilterTile = Tile<TileType::Vec, uint8_t, kIdxAlignedRows, 1, BLayout::ColMajor, -1, -1>;
    using KeySignMaskTile = Tile<TileType::Vec, uint8_t, 1, kLength, BLayout::RowMajor, -1, -1>;
    using KeySelTmpTile = Tile<TileType::Vec, uint8_t, 1, 32, BLayout::RowMajor, -1, -1>;
    using GatherSrcI16 = Tile<TileType::Vec, int16_t, 1, kLength, BLayout::RowMajor, -1, -1>;
    using GatherFullU32 = Tile<TileType::Vec, uint32_t, 1, kLength, BLayout::RowMajor, -1, -1>;
    using GatherConcatCountTile = Tile<TileType::Vec, uint32_t, kGatherConcatRows, 1, BLayout::ColMajor, -1, -1>;
    using TmpGatherTile = Tile<TileType::Vec, uint8_t, 1, cmpCol, BLayout::RowMajor, -1, -1>;
    using MergedIdxTile = Tile<TileType::Vec, uint32_t, 1, 2 * K, BLayout::RowMajor, -1, -1>;

#ifndef __PTO_AUTO__
    // Reverse dependency seed: allow the first TLOAD to proceed.
    // MTE2 side waits on MTE3 completion of previous-batch TSTORE.
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
#endif
    for (uint32_t b = 0; b < kBatch; ++b) {
        InTileU16 fullInTile(1, kLength);
        HistTile tileHist(1, kBinNum);
        HistTile chistMSB(1, kBinNum);
        HistTile chistLSB(1, kBinNum);
        IdxFilterTile idxFilter(1, 1);
        WinnerBinTile msbWinnerBin(1, 32);
        WinnerBinTile msbWinnerSaved(1, 32);
        WinnerBinTile lsbWinnerBin(1, 32);
        RemainKTile remainKTile(1, 32);
        PackedU16Tile packedThrU(1, 32);

        // Phase1: TLOAD + Histogram(MSB)
        TASSIGN(fullInTile, kUbFullKeys);
        TASSIGN(tileHist, kUbTileHist);
        TASSIGN(chistMSB, kUbChistMSB);
        TASSIGN(chistLSB, kUbChistLSB);
        TASSIGN(idxFilter, kUbIdxFilter);
        using SrcGlobal = GlobalTensor<uint16_t, pto::Shape<1, 1, 1, 1, kLength>,
                                       pto::Stride<kLength, kLength, kLength, kLength, 1>>;
        SrcGlobal srcGlobal(scoreOutBf16 + static_cast<uint64_t>(b) * kLength);
        fullInTile.SetValidCol(kLength);
#ifndef __PTO_AUTO__
        // Reverse dependency: MTE2 waits until previous batch TSTORE(MTE3) finishes.
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
#endif
        TLOAD(fullInTile, srcGlobal);
#ifndef __PTO_AUTO__
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif
        // Convert bf16 bits to ordered uint16 keys:
        // sign=1 -> ~bits, sign=0 -> bits ^ 0x8000
        InTileU16 negKeyTile(1, kLength);
        InTileU16 posKeyTile(1, kLength);
        KeySignMaskTile signMaskTile(1, kLength);
        KeySelTmpTile signTmpTile(1, 32);
        TASSIGN(negKeyTile, kUbNegKeys);
        TASSIGN(posKeyTile, kUbPosKeys);
        TASSIGN(signMaskTile, kUbSignMask);
        TASSIGN(signTmpTile, kUbSignTmp);
        negKeyTile.SetValidRow(1);
        negKeyTile.SetValidCol(kLength);
        posKeyTile.SetValidRow(1);
        posKeyTile.SetValidCol(kLength);
        signMaskTile.SetValidRow(1);
        signMaskTile.SetValidCol(kLength);
        signTmpTile.SetValidRow(1);
        signTmpTile.SetValidCol(32);
        TNOT(negKeyTile, fullInTile);
        TXORS(posKeyTile, fullInTile, static_cast<uint16_t>(0x8000u), posKeyTile);
        TCMPS(signMaskTile, fullInTile, static_cast<uint16_t>(0x8000u), CmpMode::GE);
        TSEL(fullInTile, signMaskTile, negKeyTile, posKeyTile, signTmpTile);

        chistMSB.SetValidRow(1);
        chistMSB.SetValidCol(kBinNum);
        TEXPANDS(chistMSB, 0u);
        THISTOGRAM<pto::HistByte::BYTE_1>(tileHist, fullInTile, idxFilter);
        TMOV(chistMSB, tileHist);

        // Phase2: Winner(MSB) + remainK
        {
            constexpr uint32_t kThrMsb = static_cast<uint32_t>(kLength - K);
            constexpr uint32_t kSelsFalse = 0xffffffffu;
            WinnerLaneTile msbWinnerLanes(1, kBinNum);
            MaskCmpTile maskTile(1, kBinNum);
            IdxU32Tile indexTile(1, kBinNum);
            TmpSelsTile tmpSelsTile(1, 32);
            TASSIGN(maskTile, kWinnerUbMask);
            TASSIGN(indexTile, kWinnerUbIdx);
            TASSIGN(msbWinnerLanes, kWinnerUbGather);
            TASSIGN(tmpSelsTile, kWinnerUbTmp);
            maskTile.SetValidCol(kBinNum);
            indexTile.SetValidCol(kBinNum);
            msbWinnerLanes.SetValidCol(kBinNum);
            tmpSelsTile.SetValidCol(32);
            chistMSB.SetValidCol(kBinNum);
            TCMPS(maskTile, chistMSB, kThrMsb, CmpMode::GE);
            TCI<IdxU32Tile, IdxU32Tile, uint32_t, 0>(indexTile, static_cast<uint32_t>(0), indexTile);
            TSELS(msbWinnerLanes, maskTile, indexTile, tmpSelsTile, kSelsFalse);

            TASSIGN(msbWinnerSaved, kMsbWinnerSavedUb);
            msbWinnerSaved.SetValidRow(1);
            msbWinnerSaved.SetValidCol(32);
            RowMinDstTile rowMinDst(1, 16);
            RowMinTmpTile rowMinTmp(1, kBinNum);
            TASSIGN(rowMinDst, kWinnerUbRowMinDst);
            TASSIGN(rowMinTmp, kWinnerUbRowMinTmp);
            rowMinDst.SetValidRow(1);
            rowMinDst.SetValidCol(1);
            rowMinTmp.SetValidRow(1);
            rowMinTmp.SetValidCol(kBinNum);
            TROWMIN(rowMinDst, msbWinnerLanes, rowMinTmp);
            GatherIdxU32Tile gatherIdx(1, 32);
            TmpSelsTile gatherTmp(1, 32);
            TASSIGN(gatherIdx, kWinnerUbRowMinTmp);
            TASSIGN(gatherTmp, kWinnerUbTselTmp);
            gatherIdx.SetValidRow(1);
            gatherIdx.SetValidCol(32);
            gatherTmp.SetValidCol(32);
            TEXPANDS(gatherIdx, 0u);
            TGATHER(msbWinnerSaved, rowMinDst, gatherIdx, gatherTmp);

            RowMinDstTile zeroTile(1, 16);
            RowMinDstTile selOut(1, 16);
            SelMaskRowTile selMask(1, 32);
            TmpSelsTile tselTmp(1, 32);
            TASSIGN(zeroTile, kWinnerUbSelZero);
            TASSIGN(selOut, kWinnerUbSelOut);
            TASSIGN(selMask, kWinnerUbSelMask);
            TASSIGN(tselTmp, kWinnerUbTselTmp);
            zeroTile.SetValidRow(1);
            zeroTile.SetValidCol(1);
            selOut.SetValidRow(1);
            selOut.SetValidCol(1);
            selMask.SetValidRow(1);
            selMask.SetValidCol(1);
            tselTmp.SetValidCol(32);
            RowMinDstTile uOne(1, 16);
            TASSIGN(uOne, kWinnerUbU32One);
            uOne.SetValidRow(1);
            uOne.SetValidCol(1);
            TEXPANDS(uOne, 1u);
            TSUB(rowMinDst, rowMinDst, uOne);
            constexpr uint32_t kCmp256 = 256u;
            TCMPS(selMask, rowMinDst, kCmp256, CmpMode::GT);
            TSEL(selOut, selMask, zeroTile, rowMinDst, tselTmp);
            TASSIGN(msbWinnerBin, kWinnerUbTmp);
            msbWinnerBin.SetValidRow(1);
            msbWinnerBin.SetValidCol(32);
            TGATHER(msbWinnerBin, selOut, gatherIdx, gatherTmp);

            using U32x32 = Tile<TileType::Vec, uint32_t, 1, 32, BLayout::RowMajor, -1, -1>;
            U32x32 thrMsbT(1, 32);
            U32x32 cwT(1, 32);
            TASSIGN(thrMsbT, kRemainUbTopk);
            TASSIGN(cwT, kRemainUbCw);
            TASSIGN(remainKTile, kRemainUbOut);
            TASSIGN(gatherTmp, kWinnerUbRowMinTmp);
            thrMsbT.SetValidRow(1);
            thrMsbT.SetValidCol(32);
            cwT.SetValidRow(1);
            cwT.SetValidCol(32);
            remainKTile.SetValidRow(1);
            remainKTile.SetValidCol(32);
            gatherTmp.SetValidCol(32);
            constexpr uint32_t kThrMsbU = static_cast<uint32_t>(kLength - K);
            TEXPANDS(thrMsbT, kThrMsbU);
            TGATHER(cwT, chistMSB, msbWinnerBin, gatherTmp);
            TSUB(remainKTile, thrMsbT, cwT);
        }

        // Phase3: Histogram(LSB) with winner-MSB filter (topk_ub style)
        idxFilter.SetValidRow(1);
        idxFilter.SetValidCol(1);
        msbWinnerSaved.SetValidRow(1);
        msbWinnerSaved.SetValidCol(1);
        TCVT(idxFilter, msbWinnerSaved, RoundMode::CAST_TRUNC);
        chistLSB.SetValidRow(1);
        chistLSB.SetValidCol(kBinNum);
        TEXPANDS(chistLSB, 0u);
        THISTOGRAM<pto::HistByte::BYTE_0>(tileHist, fullInTile, idxFilter);
        TMOV(chistLSB, tileHist);

        // Phase4: Winner(LSB) + packed threshold
        {
            constexpr uint32_t kSelsFalse = 0xffffffffu;
            WinnerLaneTile lsbWinnerLanes(1, kBinNum);
            MaskCmpTile maskTile(1, kBinNum);
            IdxU32Tile indexTile(1, kBinNum);
            TmpSelsTile tmpSelsTile(1, 32);
            TASSIGN(maskTile, kWinnerUbMask);
            TASSIGN(indexTile, kWinnerUbIdx);
            TASSIGN(lsbWinnerLanes, kWinnerUbGather);
            TASSIGN(tmpSelsTile, kWinnerUbTmp);
            maskTile.SetValidCol(kBinNum);
            indexTile.SetValidCol(kBinNum);
            lsbWinnerLanes.SetValidCol(kBinNum);
            tmpSelsTile.SetValidCol(32);
            chistLSB.SetValidCol(kBinNum);
            remainKTile.SetValidRow(1);
            remainKTile.SetValidCol(32);
            TCMPS(maskTile, chistLSB, remainKTile, CmpMode::GT);
            TCI<IdxU32Tile, IdxU32Tile, uint32_t, 0>(indexTile, static_cast<uint32_t>(0), indexTile);
            TSELS(lsbWinnerLanes, maskTile, indexTile, tmpSelsTile, kSelsFalse);

            TASSIGN(lsbWinnerBin, kWinnerUbTmp);
            lsbWinnerBin.SetValidRow(1);
            lsbWinnerBin.SetValidCol(32);
            RowMinDstTile rowMinDst(1, 16);
            RowMinTmpTile rowMinTmp(1, kBinNum);
            TASSIGN(rowMinDst, kWinnerUbRowMinDst);
            TASSIGN(rowMinTmp, kWinnerUbRowMinTmp);
            rowMinDst.SetValidRow(1);
            rowMinDst.SetValidCol(1);
            rowMinTmp.SetValidRow(1);
            rowMinTmp.SetValidCol(kBinNum);
            TROWMIN(rowMinDst, lsbWinnerLanes, rowMinTmp);
            GatherIdxU32Tile gatherIdx(1, 32);
            TmpSelsTile gatherTmp(1, 32);
            TASSIGN(gatherIdx, kWinnerUbRowMinTmp);
            TASSIGN(gatherTmp, kWinnerUbTselTmp);
            gatherIdx.SetValidRow(1);
            gatherIdx.SetValidCol(32);
            gatherTmp.SetValidCol(32);
            TEXPANDS(gatherIdx, 0u);
            TGATHER(lsbWinnerBin, rowMinDst, gatherIdx, gatherTmp);
        }
        {
            PackedU16Tile msbU(1, 32);
            PackedU16Tile hiU(1, 32);
            PackedU16Tile lsbU(1, 32);
            TASSIGN(packedThrU, kRemainUbOut);
            TASSIGN(msbU, kRemainUbTopk);
            TASSIGN(hiU, kRemainUbCw);
            TASSIGN(lsbU, kRemainUbOut);
            packedThrU.SetValidRow(1);
            packedThrU.SetValidCol(1);
            msbU.SetValidRow(1);
            msbU.SetValidCol(1);
            hiU.SetValidRow(1);
            hiU.SetValidCol(1);
            lsbU.SetValidRow(1);
            lsbU.SetValidCol(1);
            msbWinnerSaved.SetValidRow(1);
            msbWinnerSaved.SetValidCol(1);
            lsbWinnerBin.SetValidRow(1);
            lsbWinnerBin.SetValidCol(1);
            TCVT(msbU, msbWinnerSaved, RoundMode::CAST_TRUNC);
            TCVT(lsbU, lsbWinnerBin, RoundMode::CAST_TRUNC);
            constexpr uint16_t kShift8 = 8u;
            TSHLS(hiU, msbU, kShift8);
            TOR(packedThrU, hiU, lsbU);
        }

        // Phase5: TGATHER(GT/EQ) + TCONCAT + TSTORE
        GatherFullU32 gtChunk(1, kLength);
        GatherFullU32 eqChunk(1, kLength);
        GatherConcatCountTile idxGtCnt(1, 1);
        GatherConcatCountTile idxEqCnt(1, 1);
        TmpGatherTile tmpGt(1, cmpVCol);
        TmpGatherTile tmpEq(1, cmpVCol);
        TASSIGN(gtChunk, kFullGatherGtDst);
        TASSIGN(eqChunk, kFullGatherEqDst);
        TASSIGN(idxGtCnt, kChunkConcatGt);
        TASSIGN(idxEqCnt, kChunkConcatEq);
        TASSIGN(tmpGt, kGatherUbTmp);
        TASSIGN(tmpEq, kGatherUbTmp);
        GatherSrcI16 srcGt(1, kLength);
        GatherSrcI16 srcEq(1, kLength);
        TASSIGN(srcGt, kUbFullKeys);
        TASSIGN(srcEq, kUbFullKeys);
        srcGt.SetValidRow(1);
        srcGt.SetValidCol(kLength);
        srcEq.SetValidRow(1);
        srcEq.SetValidCol(kLength);
        gtChunk.SetValidRow(1);
        gtChunk.SetValidCol(kLength);
        eqChunk.SetValidRow(1);
        eqChunk.SetValidCol(kLength);
        idxGtCnt.SetValidRow(1);
        idxGtCnt.SetValidCol(1);
        idxEqCnt.SetValidRow(1);
        idxEqCnt.SetValidCol(1);
        tmpGt.SetValidRow(1);
        tmpGt.SetValidCol(cmpVCol);
        tmpEq.SetValidRow(1);
        tmpEq.SetValidCol(cmpVCol);
        TGATHER<GatherFullU32, GatherSrcI16, PackedU16Tile, GatherConcatCountTile, TmpGatherTile, CmpMode::GT>(
            gtChunk, srcGt, packedThrU, idxGtCnt, tmpGt, 0);
        TGATHER<GatherFullU32, GatherSrcI16, PackedU16Tile, GatherConcatCountTile, TmpGatherTile, CmpMode::EQ>(
            eqChunk, srcEq, packedThrU, idxEqCnt, tmpEq, 0);

        MergedIdxTile mergedIdx(1, 2 * K);
        TASSIGN(mergedIdx, kUbMerged);
        mergedIdx.SetValidRow(1);
        mergedIdx.SetValidCol(2 * K);
        TCONCAT_IMPL(mergedIdx, gtChunk, eqChunk, idxGtCnt, idxEqCnt);
        mergedIdx.SetValidCol(K);
        using OutShape = pto::Shape<1, 1, 1, 1, K>;
        using OutStride = pto::Stride<K, K, K, K, 1>;
        GlobalTensor<uint32_t, OutShape, OutStride> outGlobal(outIdx + b * K);
#ifndef __PTO_AUTO__
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
#endif
        TSTORE(outGlobal, mergedIdx);
#ifndef __PTO_AUTO__
        // Release reverse dependency for next batch TLOAD.
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
#endif
    }
#endif
}

template <uint32_t kBatch, uint32_t kHeads, uint32_t kLength>
__global__ AICORE void PostProcessScoreKernel(__gm__ float *matmulOut, __gm__ float *postScale, __gm__ float *scoreOut,
                                              __gm__ uint16_t *scoreOutBf16)
{
    PostProcessScoreImpl<kBatch, kHeads, kLength>(matmulOut, postScale, scoreOut, scoreOutBf16);
}

template <uint32_t kLength, uint32_t K>
__global__ AICORE void TopKFromScoreKernel(__gm__ uint16_t *scoreOutBf16, __gm__ uint32_t *outIdx)
{
    TopKFromScoreImpl<kLength, K>(scoreOutBf16, outIdx);
}

template <uint32_t blockDim, uint32_t m, uint32_t k, uint32_t n, uint32_t singleCoreM, uint32_t singleCoreK,
          uint32_t singleCoreN, uint32_t baseM, uint32_t baseK, uint32_t baseN, uint32_t stepM, uint32_t stepKa,
          uint32_t stepKb, uint32_t stepN, uint32_t kBatch, uint32_t kHeads, uint32_t kTopK>
__global__ AICORE void IndexerMergedPipelineKernel(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1,
                                                   __gm__ uint8_t *src2, __gm__ uint8_t *src3, __gm__ float *postScale,
                                                   __gm__ float *scoreOut, __gm__ uint16_t *scoreOutBf16,
                                                   __gm__ uint32_t *outIdx)
{
    RunMxMatmul<float, float8_e5m2_t, float8_e8m0_t, blockDim, m, k, n, singleCoreM, singleCoreK, singleCoreN, baseM,
                baseK, baseN, stepM, stepKa, stepKb, stepN, true>(
        reinterpret_cast<__gm__ float *>(out), reinterpret_cast<__gm__ float8_e5m2_t *>(src0),
        reinterpret_cast<__gm__ float8_e5m2_t *>(src1), reinterpret_cast<__gm__ float8_e8m0_t *>(src2),
        reinterpret_cast<__gm__ float8_e8m0_t *>(src3), postScale, scoreOut, scoreOutBf16);
    if constexpr (DAV_VEC) {
        if (get_subblockid() == 0 && get_block_idx() == 0) {
            TopKFromScoreImpl<n, kTopK>(scoreOutBf16, outIdx);
        }
    }
}


void LaunchIndexerMatmul(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, uint8_t *src3, float *postScale,
                         float *scoreOut, uint16_t *scoreOutBf16, uint32_t *outIdx, void *stream)
{
    (void)scoreOutBf16;
    constexpr uint32_t blockDim = 1;
    constexpr uint32_t m = 128;
    constexpr uint32_t k = 1024;
    constexpr uint32_t n = INDEXER_TEST_N;
    constexpr uint32_t singleCoreM = 128;
    constexpr uint32_t singleCoreK = 1024;
    constexpr uint32_t singleCoreN = n;
    constexpr uint32_t baseM = 128;
    constexpr uint32_t baseK = 128;
    constexpr uint32_t baseN = 128;
    constexpr uint32_t stepM = 1;
    constexpr uint32_t stepKa = 1;
    constexpr uint32_t stepKb = 1;
    constexpr uint32_t stepN = 1;
    MxMatmulPerformance<blockDim, m, k, n, singleCoreM, singleCoreK, singleCoreN, baseM, baseK, baseN, stepM, stepKa,
                        stepKb, stepN><<<blockDim, nullptr, stream>>>(out, src0, src1, src2, src3, postScale, scoreOut,
                                                                      outIdx);
}

void LaunchIndexerPostProcess(float *matmulOut, float *postScale, float *scoreOut, uint16_t *scoreOutBf16, void *stream)
{
    constexpr uint32_t m = 128;
    constexpr uint32_t n = INDEXER_TEST_N;
    constexpr uint32_t kBatch = 2;
    constexpr uint32_t kHeads = m / 2;
    constexpr uint32_t kLength = n;
    PostProcessScoreKernel<kBatch, kHeads, kLength><<<1, nullptr, stream>>>(matmulOut, postScale, scoreOut, scoreOutBf16);
}

void LaunchIndexerTopK(uint16_t *scoreOutBf16, uint32_t *outIdx, void *stream)
{
    constexpr uint32_t n = INDEXER_TEST_N;
    constexpr uint32_t kLength = n;
    constexpr uint32_t kTopK = INDEXER_TOPK;
    TopKFromScoreKernel<kLength, kTopK><<<1, nullptr, stream>>>(scoreOutBf16, outIdx);
}

void LaunchIndexerMxfp8(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, uint8_t *src3, float *postScale,
                        float *scoreOut, uint16_t *scoreOutBf16, uint32_t *outIdx, void *stream)
{
    constexpr uint32_t blockDim = 1;
    constexpr uint32_t m = 128;
    constexpr uint32_t k = 1024;
    constexpr uint32_t n = INDEXER_TEST_N;
    constexpr uint32_t singleCoreM = 128;
    constexpr uint32_t singleCoreK = 1024;
    constexpr uint32_t singleCoreN = n;
    constexpr uint32_t baseM = 128;
    constexpr uint32_t baseK = 128;
    constexpr uint32_t baseN = 128;
    constexpr uint32_t stepM = 1;
    constexpr uint32_t stepKa = 1;
    constexpr uint32_t stepKb = 1;
    constexpr uint32_t stepN = 1;
    constexpr uint32_t kBatch = 2;
    constexpr uint32_t kHeads = 64;
    constexpr uint32_t kTopK = INDEXER_TOPK;
    float *mergedScoreOut = scoreOut;
#if INDEXER_MERGED_SKIP_SCORE_STORE
    mergedScoreOut = nullptr;
#endif
    IndexerMergedPipelineKernel<blockDim, m, k, n, singleCoreM, singleCoreK, singleCoreN, baseM, baseK, baseN, stepM,
                                stepKa, stepKb, stepN, kBatch, kHeads, kTopK>
        <<<blockDim, nullptr, stream>>>(out, src0, src1, src2, src3, postScale, mergedScoreOut, scoreOutBf16, outIdx);
}
