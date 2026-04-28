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
#include <pto/common/fifo.hpp>
#include <pto/pto-inst.hpp>

using namespace pto;

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

// AIV: wait / TPOP to Vec UB, scale, GM store, release FIFO. Matches FA vector-side TPOP from qkPipe / softmax output.
template <int m, int n, uint32_t baseM, uint32_t baseN, typename VecF, typename CvOutPipeT>
AICORE inline void IndexerAiv_StoreTpopMulsTstore(__gm__ float *currentDst, uint32_t i, uint32_t j, VecF &vecForStore,
                                                   CvOutPipeT &cvPipe)
{
    if constexpr (DAV_VEC) {
        // V1C1_VEC0 mode: vec1 should only participate in FIFO handshake.
        // Keep single UB address, but avoid vec1 touching data path (TPOP/TMULS/TSTORE).
        if (get_subblockid() != 0) {
            cvPipe.cons.setWaitStatus(true);
            cvPipe.cons.setFreeStatus(true);
            cvPipe.cons.wait();
            cvPipe.cons.free();
            return;
        }

        using NDValidShapeC = TileShape2D<float, baseM, baseN, Layout::ND>;
        using NDWholeShapeC = BaseShape2D<float, m, n, Layout::ND>;
        using GlobalDataOut = GlobalTensor<float, NDValidShapeC, NDWholeShapeC, Layout::ND>;
        GlobalDataOut dstGlobal(currentDst + i * static_cast<uint64_t>(baseM) * n + j * baseN);
        cvPipe.cons.setTileId(0, 0);
        cvPipe.cons.setWaitStatus(true);
        cvPipe.cons.setFreeStatus(false);
        cvPipe.cons.setEntryOffset(0);
        TPOP(vecForStore, cvPipe);
        constexpr float kHalf = 0.5f;
        TMULS(vecForStore, vecForStore, kHalf);
        // Ensure MTE3 GM store consumes vec buffer before FIFO free/next TPUSH overwrite.
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        TSTORE(dstGlobal, vecForStore);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        cvPipe.cons.setFreeStatus(true);
        TFREE(cvPipe);
    }
}

template <int m, int n, uint32_t baseM, uint32_t baseN, typename ResTile, typename VecF, typename CvOutPipeT>
AICORE inline void StoreResultWithCvPipe(ResTile &cTile, __gm__ float *currentDst, uint32_t i, uint32_t j,
                                         VecF &vecForStore, CvOutPipeT &cvPipe)
{
    if constexpr (DAV_CUBE) {
        SetFlag<PIPE_M, PIPE_FIX>(0);
        WaitFlag<PIPE_M, PIPE_FIX>(0);
    }

    if constexpr (DAV_CUBE) {
        IndexerAic_StoreTpushToCvFifo(cTile, cvPipe);
    }
    if constexpr (DAV_VEC) {
        IndexerAiv_StoreTpopMulsTstore<m, n, baseM, baseN>(currentDst, i, j, vecForStore, cvPipe);
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
          typename RightScaleTile, typename ResTile>
AICORE inline void Compute(__gm__ U *currentSrc0, __gm__ U *currentSrc1, __gm__ X *currentSrc2, __gm__ X *currentSrc3,
                           __gm__ T *&currentDst, TileMatA aMatTile[BUFFER_NUM], TileMatB bMatTile[BUFFER_NUM],
                           TileScaleA aScaleMatTile[BUFFER_NUM], TileScaleB bScaleMatTile[BUFFER_NUM],
                           LeftTile aTile[BUFFER_NUM], RightTile bTile[BUFFER_NUM],
                           LeftScaleTile aScaleTile[BUFFER_NUM], RightScaleTile bScaleTile[BUFFER_NUM], ResTile &cTile)
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
                StoreResultWithCvPipe<m, n, baseM, baseN, ResTile, VecF, CvOutPipe>(cTile, currentDst, i, j, vecForStore,
                                                                                      cvPipe);
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
          uint32_t stepKa, uint32_t stepKb, uint32_t stepN>
AICORE inline void RunMxMatmul(__gm__ T *out, __gm__ U *src0, __gm__ U *src1, __gm__ X *src2, __gm__ X *src3)
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
            RightScaleTile, ResTile>(currentSrc0, currentSrc1, currentSrc2, currentSrc3, currentDst, aMatTile, bMatTile,
                                     aScaleMatTile, bScaleMatTile, aTile, bTile, aScaleTile, bScaleTile, cTile);

    if constexpr (DAV_CUBE) {
        WaitSyncFlags();
    }
}

template <uint32_t blockDim, uint32_t m, uint32_t k, uint32_t n, uint32_t singleCoreM, uint32_t singleCoreK,
          uint32_t singleCoreN, uint32_t baseM, uint32_t baseK, uint32_t baseN, uint32_t stepM, uint32_t stepKa,
          uint32_t stepKb, uint32_t stepN>
__global__ AICORE void MxMatmulPerformance(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1,
                                           __gm__ uint8_t *src2, __gm__ uint8_t *src3)
{
    RunMxMatmul<float, float8_e5m2_t, float8_e8m0_t, blockDim, m, k, n, singleCoreM, singleCoreK, singleCoreN, baseM,
                baseK, baseN, stepM, stepKa, stepKb, stepN>(reinterpret_cast<__gm__ float *>(out),
                                                            reinterpret_cast<__gm__ float8_e5m2_t *>(src0),
                                                            reinterpret_cast<__gm__ float8_e5m2_t *>(src1),
                                                            reinterpret_cast<__gm__ float8_e8m0_t *>(src2),
                                                            reinterpret_cast<__gm__ float8_e8m0_t *>(src3));
}

void LaunchIndexerMxfp8(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, uint8_t *src3, void *stream)
{
    // A: [512,1024], B (layout as demo DN k×n): [1024,512] → C: [512,512]; single core, f32 out = 0.5 * MX matmul.
    constexpr uint32_t blockDim = 1;
    constexpr uint32_t m = 512;
    constexpr uint32_t k = 1024;
    constexpr uint32_t n = 512;
    constexpr uint32_t singleCoreM = 512;
    constexpr uint32_t singleCoreK = 1024;
    constexpr uint32_t singleCoreN = 512;
    constexpr uint32_t baseM = 128;
    constexpr uint32_t baseK = 128;
    constexpr uint32_t baseN = 128;
    constexpr uint32_t stepM = 1;
    constexpr uint32_t stepKa = 1;
    constexpr uint32_t stepKb = 1;
    constexpr uint32_t stepN = 1;
    MxMatmulPerformance<blockDim, m, k, n, singleCoreM, singleCoreK, singleCoreN, baseM, baseK, baseN, stepM, stepKa,
                        stepKb, stepN><<<blockDim, nullptr, stream>>>(out, src0, src1, src2, src3);
}
