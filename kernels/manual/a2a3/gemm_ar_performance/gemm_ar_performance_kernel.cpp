/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include "gemm_config.h"
#include "pto/common/pto_tile.hpp"
#include "pto/comm/pto_comm_inst.hpp"
#include "common.hpp"
#include <pto/pto-inst.hpp>
#include <cmath>
#include <random>
#include <vector>

using namespace pto;
constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t L0_PINGPONG_BYTES = 32 * 1024; // L0A/L0B ping-pong split (32 KiB per buffer)

#ifdef __DAV_CUBE__

// Pipeline mental model (instruction-level):
// - TLOAD     (GM -> L1):   fill aMatTile/bMatTile
// - TEXTRACT  (L1 -> L0):   slice aMatTile/bMatTile into aTile/bTile for the current baseK
// - TMATMUL   (Cube):       cTile = A*B (accumulated over K)
// - TSTORE    (L0C -> GM):  write cTile back to GM
//
// The code still uses PIPE_MTE* events for synchronization because those are the underlying hardware pipes;
// comments refer to the high-level PTO instructions to make tuning easier.

template <typename OutTile, typename LeftTile, typename RightTile>
AICORE inline void MatmulAcc(OutTile cTile, LeftTile aTile, RightTile bTile, uint32_t k)
{
    if (k == 0) {
        TMATMUL(cTile, aTile, bTile);
    } else {
        TMATMUL_ACC(cTile, cTile, aTile, bTile);
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

template <typename T, typename U, typename S, int m, int k, int n, uint32_t singleCoreM, uint32_t singleCoreK,
    uint32_t singleCoreN>
AICORE inline void InitGMOffsets(__gm__ U *&currentSrc0, __gm__ S *&currentSrc1, __gm__ T *&currentDst, __gm__ T *out,
    __gm__ U *src0, __gm__ S *src1)
{
    constexpr uint32_t mIter = m / singleCoreM;
    uint32_t mIterIdx = get_block_idx() % mIter;
    uint32_t nIterIdx = get_block_idx() / mIter;
    uint64_t gmOffsetA = mIterIdx * singleCoreM * k;
    uint64_t gmOffsetB = nIterIdx * k * singleCoreN;
    uint64_t gmOffsetC = mIterIdx * singleCoreM * n + nIterIdx * singleCoreN;
    currentSrc0 = src0 + gmOffsetA;
    currentSrc1 = src1 + gmOffsetB;
    currentDst = out + gmOffsetC;
}

template <typename T, typename U, typename S, int m, int k, int n, uint32_t baseM, uint32_t baseK, uint32_t baseN,
    uint32_t stepKa, uint32_t stepKb, uint32_t singleCoreK>
AICORE inline void ProcessKIteration(uint32_t kIter, uint32_t i, uint32_t j, __gm__ U *currentSrc0,
    __gm__ S *currentSrc1,
    Tile<TileType::Mat, U, baseM, baseK * stepKa, BLayout::ColMajor, baseM, baseK * stepKa, SLayout::RowMajor>
        aMatTile[BUFFER_NUM],
    Tile<TileType::Mat, S, baseK * stepKb, baseN, BLayout::RowMajor, baseK * stepKb, baseN, SLayout::ColMajor>
        bMatTile[BUFFER_NUM],
    TileLeft<U, baseM, baseK, baseM, baseK> aTile[BUFFER_NUM],
    TileRight<S, baseK, baseN, baseK, baseN> bTile[BUFFER_NUM], TileAcc<T, baseM, baseN, baseM, baseN> &cTile,
    uint8_t &mte2DBFlag, uint8_t &mte1DBFlag)
{
    using NDValidShapeA = TileShape2D<U, baseM, baseK * stepKa, Layout::ND>;
    using NDsingleCoreShapeA = BaseShape2D<U, m, k, Layout::ND>;
    using GlobalDataSrcA = pto::GlobalTensor<U, NDValidShapeA, NDsingleCoreShapeA, Layout::ND>;

    using NDValidShapeB = TileShape2D<U, baseK * stepKb, baseN, Layout::DN>;
    using NDsingleCoreShapeB = BaseShape2D<U, k, n, Layout::DN>;
    using GlobalDataSrcB = pto::GlobalTensor<U, NDValidShapeB, NDsingleCoreShapeB, Layout::DN>;

    const uint32_t kModstepKa = kIter % stepKa;

    if (kModstepKa == 0) {
        GlobalDataSrcA gmA(currentSrc0 + i * singleCoreK * baseM + kIter * baseK);
        GlobalDataSrcB gmB(currentSrc1 + j * singleCoreK * baseN + kIter * baseK);

        WaitFlag<PIPE_MTE1, PIPE_MTE2>(mte2DBFlag);
        TLOAD(aMatTile[mte2DBFlag], gmA);
        SetFlag<PIPE_MTE2, PIPE_MTE1>(0);
        TLOAD(bMatTile[mte2DBFlag], gmB);
        SetFlag<PIPE_MTE2, PIPE_MTE1>(1);
        mte2DBFlag = (mte2DBFlag == 0) ? 1 : 0;
    }

    const uint32_t currMte2Idx = (mte2DBFlag == 0) ? 1 : 0;
    WaitFlag<PIPE_M, PIPE_MTE1>(mte1DBFlag);

    if (kModstepKa == 0)
        WaitFlag<PIPE_MTE2, PIPE_MTE1>(0);
    TEXTRACT(aTile[mte1DBFlag], aMatTile[currMte2Idx], 0, kModstepKa * baseK);

    if (kModstepKa == 0)
        WaitFlag<PIPE_MTE2, PIPE_MTE1>(1);
    TEXTRACT(bTile[mte1DBFlag], bMatTile[currMte2Idx], (kIter % stepKb) * baseK, 0);

    if ((kIter + 1) % stepKa == 0) {
        SetFlag<PIPE_MTE1, PIPE_MTE2>(currMte2Idx);
    }

    SetFlag<PIPE_MTE1, PIPE_M>(mte1DBFlag);
    WaitFlag<PIPE_MTE1, PIPE_M>(mte1DBFlag);
    MatmulAcc(cTile, aTile[mte1DBFlag], bTile[mte1DBFlag], kIter);
    SetFlag<PIPE_M, PIPE_MTE1>(mte1DBFlag);
    mte1DBFlag = (mte1DBFlag == 0) ? 1 : 0;
}

template <typename T, typename U, typename S, int m, int n, uint32_t baseM, uint32_t baseN, uint32_t singleCoreK>
AICORE inline void StoreResult(
    TileAcc<T, baseM, baseN, baseM, baseN> &cTile, __gm__ T *currentDst, uint32_t i, uint32_t j)
{
    SetFlag<PIPE_M, PIPE_FIX>(0);
    WaitFlag<PIPE_M, PIPE_FIX>(0);

    using NDValidShapeC = TileShape2D<T, baseM, baseN, Layout::ND>;
    using NDWholeShapeC = BaseShape2D<T, m, n, Layout::ND>;
    using GlobalDataOut = pto::GlobalTensor<T, NDValidShapeC, NDWholeShapeC, Layout::ND>;

    GlobalDataOut dstGlobal(currentDst + i * baseM * n + j * baseN);
    TSTORE(dstGlobal, cTile);

    SetFlag<PIPE_FIX, PIPE_M>(0);
    WaitFlag<PIPE_FIX, PIPE_M>(0);
}

template <typename T, typename U, typename S, typename B, uint32_t blockDim, int m, int k, int n, int validM,
    int validK, int validN, uint32_t singleCoreM, uint32_t singleCoreK, uint32_t singleCoreN, uint32_t baseM,
    uint32_t baseK, uint32_t baseN, uint32_t stepM, uint32_t stepKa, uint32_t stepKb, uint32_t stepN>
AICORE inline void RunGemmE2E(__gm__ T *out, __gm__ U *src0, __gm__ S *src1, __gm__ uint8_t* shmem, bool is_overlap)
{
    __gm__ U *currentSrc0 = nullptr;
    __gm__ S *currentSrc1 = nullptr;
    __gm__ T *currentDst = nullptr;

    __gm__ uint32_t *shmemFlag = reinterpret_cast<__gm__ uint32_t*>(shmem + (1UL << 28));

    InitGMOffsets<T, U, S, m, k, n, singleCoreM, singleCoreK, singleCoreN>(currentSrc0, currentSrc1, currentDst, reinterpret_cast<__gm__ T*>(shmem), src0, src1);

    using TileMatA = Tile<TileType::Mat, U, baseM, baseK * stepKa, BLayout::ColMajor, baseM, baseK * stepKa, SLayout::RowMajor>;
    using TileMatB = Tile<TileType::Mat, S, baseK * stepKb, baseN, BLayout::RowMajor, baseK * stepKb, baseN, SLayout::ColMajor>;

    TileMatA aMatTile[BUFFER_NUM];
    TileMatB bMatTile[BUFFER_NUM];

    using LeftTile = TileLeft<U, baseM, baseK, baseM, baseK>;
    using RightTile = TileRight<S, baseK, baseN, baseK, baseN>;
    using ResTile = TileAcc<T, baseM, baseN, baseM, baseN>;

    LeftTile aTile[BUFFER_NUM];
    RightTile bTile[BUFFER_NUM];
    ResTile cTile;

    TASSIGN(aMatTile[0], 0x0);
    TASSIGN(aMatTile[1], 0x0 + baseM * baseK * stepKa * sizeof(U));
    TASSIGN(bMatTile[0], 0x0 + baseM * baseK * stepKa * BUFFER_NUM * sizeof(U));
    TASSIGN(bMatTile[1], 0x0 + baseM * baseK * stepKa * BUFFER_NUM * sizeof(U) + baseK * baseN * stepKb * sizeof(U));

    TASSIGN(aTile[0], 0x0);
    TASSIGN(aTile[1], 0x0 + L0_PINGPONG_BYTES);
    TASSIGN(bTile[0], 0x0);
    TASSIGN(bTile[1], 0x0 + L0_PINGPONG_BYTES);
    TASSIGN(cTile, 0x0);

    constexpr uint32_t mLoop = singleCoreM / baseM;
    constexpr uint32_t nLoop = singleCoreN / baseN;
    constexpr uint32_t kLoop = singleCoreK / baseK;
    uint8_t mte2DBFlag = 0, mte1DBFlag = 0;

    SetFlag<PIPE_MTE1, PIPE_MTE2>(0); 
    SetFlag<PIPE_MTE1, PIPE_MTE2>(1);
    SetFlag<PIPE_M, PIPE_MTE1>(0);
    SetFlag<PIPE_M, PIPE_MTE1>(1);

    for (uint32_t i = 0; i < mLoop; i++) {
        for (uint32_t j = 0; j < nLoop; j++) {
            for (uint32_t kIter = 0; kIter < kLoop; kIter++) {
                ProcessKIteration<T, U, S, m, k, n, baseM, baseK, baseN, stepKa, stepKb, singleCoreK>(kIter, i, j,
                    currentSrc0, currentSrc1, aMatTile, bMatTile, aTile, bTile, cTile, mte2DBFlag, mte1DBFlag);
            }
            StoreResult<T, U, S, m, n, baseM, baseN, singleCoreK>(cTile, currentDst, i, j);

            AscendC::PipeBarrier<PIPE_ALL>();

            // Notify completion using GlobalTensor wrapper (per-tile)
            {    
                using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
                using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
                using GSignal = pto::GlobalTensor<int32_t, ShapeDyn, StrideDyn, pto::Layout::ND>;
                
                ShapeDyn shape(1, 1, 1, 1, 1);
                StrideDyn stride(1, 1, 1, 1, 1);
                
                __gm__ int32_t *flagPtr = reinterpret_cast<__gm__ int32_t*>(shmemFlag) + 
                                        get_block_idx() * mLoop * nLoop + j * mLoop + i;
                GSignal flagSignal(flagPtr, shape, stride);
                comm::TNOTIFY(flagSignal, 1, comm::NotifyOp::Set);
            }
        }
    }

    WaitFlag<PIPE_M, PIPE_MTE1>(0);
    WaitFlag<PIPE_M, PIPE_MTE1>(1);
    WaitFlag<PIPE_MTE1, PIPE_MTE2>(0);
    WaitFlag<PIPE_MTE1, PIPE_MTE2>(1);
}
#endif

#ifdef __DAV_VEC__
template <typename T, uint32_t m, uint32_t n, uint32_t singleCoreM, uint32_t singleCoreN, uint32_t baseM, uint32_t baseN>
AICORE inline void RunAllReduceE2E(__gm__ uint8_t* shmem, __gm__ uint8_t* out, bool is_overlap)
{
    __gm__ int32_t *shmemFlag = reinterpret_cast<__gm__ int32_t*>(shmem + (1UL << 28));

    constexpr uint32_t mLoop = singleCoreM / baseM;
    constexpr uint32_t nLoop = singleCoreN / baseN;

    int my_rank = shmem_my_pe();
    int block_idx = get_block_idx();
    int n_ranks = shmem_n_pes();

    using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using GSignal = pto::GlobalTensor<int32_t, ShapeDyn, StrideDyn, pto::Layout::ND>;

    using NDValidShapeC = TileShape2D<T, baseM / 2, baseN, Layout::ND>;
    using NDWholeShapeC = BaseShape2D<T, m, n, Layout::ND>;
    using GlobalDataAR  = pto::GlobalTensor<T, NDValidShapeC, NDWholeShapeC, Layout::ND>;
    using TileDataAR  = Tile<TileType::Vec, T, baseM / 2, baseN, BLayout::RowMajor, -1, -1>;

    TileDataAR accTile(baseM / 2, baseN);
    TileDataAR recvTile(baseM / 2, baseN);
    TileDataAR dstTile(baseM / 2, baseN);

    TASSIGN(accTile,  0x0);
    TASSIGN(recvTile, 0x0 + baseM / 2 * baseN * sizeof(T));
    TASSIGN(dstTile,  0x0 + baseM / 2 * baseN * sizeof(T) * 2);

    constexpr uint32_t mIter = m / singleCoreM;
    uint32_t mIterIdx = get_block_idx() % mIter;
    uint32_t nIterIdx = get_block_idx() / mIter;
    uint64_t gmOffsetC = mIterIdx * singleCoreM * n + nIterIdx * singleCoreN;
    __gm__ T* out_addr = reinterpret_cast<__gm__ T*>(out) + gmOffsetC;

    ShapeDyn shape(1, 1, 1, 1, 1);
    StrideDyn stride(1, 1, 1, 1, 1);

    for(uint32_t i = 0; i < mLoop; i++){
        for(uint32_t j = 0; j < nLoop; j++){
            if(is_overlap){
                // wait all of the tiles to be finished
                for(uint32_t r = 0; r < n_ranks; r++){
                    __gm__ int32_t *flagPtr = reinterpret_cast<__gm__ int32_t*>(shmem_ptr(shmemFlag, r)) + get_block_idx() * mLoop * nLoop + j * mLoop + i;
                    GSignal flagSignal(flagPtr, shape, stride);
                    comm::TWAIT(flagSignal, 1, comm::WaitCmp::EQ);
                }
            }
            auto tile_offset = i * baseM * n + get_subblockid() * baseM / 2 * n + j * baseN;
            auto total_offset = gmOffsetC + tile_offset;
            
            GlobalDataAR tensors[16];

            for (int r = 0; r < n_ranks && r < 16; ++r) {
                __gm__ T *rank_r_shmem = reinterpret_cast<__gm__ T*>(shmem_ptr(shmem, r));
                __gm__ T *dataPtr = rank_r_shmem + total_offset;
                tensors[r] = GlobalDataAR(dataPtr);
            }
            pto::comm::ParallelGroup<GlobalDataAR> pg(tensors, n_ranks, my_rank);

            GlobalDataAR dstGlobal(out_addr + tile_offset);
            pto::comm::TREDUCE_PINGPONG(pg, dstGlobal, accTile, recvTile, dstTile, comm::ReduceOp::Sum);
        }
    }
    AscendC::PipeBarrier<PIPE_ALL>();
        
}
#endif


// ============================================================================
// Kernel 1: Fused GEMM + AllReduce (overlap mode, tile-level pipeline)
// ============================================================================
template <typename T, uint32_t blockDim, uint32_t m, uint32_t k, uint32_t n, uint32_t singleCoreM, uint32_t singleCoreK,
    uint32_t singleCoreN, uint32_t baseM, uint32_t baseK, uint32_t baseN, uint32_t stepM, uint32_t stepKa,
    uint32_t stepKb, uint32_t stepN>
__global__ AICORE void GemmAllReduce(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1, __gm__ uint8_t* shmem, bool is_overlap)
{
#ifdef __DAV_CUBE__
    RunGemmE2E<float, half, half, float, blockDim, m, k, n, m, k, n, singleCoreM, singleCoreK, singleCoreN, baseM,
        baseK, baseN, stepM, stepKa, stepKb, stepN>(reinterpret_cast<__gm__ float *>(out),
        reinterpret_cast<__gm__ half *>(src0), reinterpret_cast<__gm__ half *>(src1), shmem, is_overlap);
#endif

#ifdef __DAV_VEC__
    RunAllReduceE2E<float, m, n, singleCoreM, singleCoreN, baseM, baseN>(shmem, out, is_overlap);
#endif
}

// ============================================================================
// Kernel 2: GEMM Only (non-overlap mode, step 1)
// Only Cube path is active. Vec path is idle.
// ============================================================================
template <typename T, uint32_t blockDim, uint32_t m, uint32_t k, uint32_t n, uint32_t singleCoreM, uint32_t singleCoreK,
    uint32_t singleCoreN, uint32_t baseM, uint32_t baseK, uint32_t baseN, uint32_t stepM, uint32_t stepKa,
    uint32_t stepKb, uint32_t stepN>
__global__ AICORE void GemmOnlyKernel(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1, __gm__ uint8_t* shmem)
{
#ifdef __DAV_CUBE__
    // is_overlap=false: TNOTIFY still fires but Vec won't TWAIT
    RunGemmE2E<float, half, half, float, blockDim, m, k, n, m, k, n, singleCoreM, singleCoreK, singleCoreN, baseM,
        baseK, baseN, stepM, stepKa, stepKb, stepN>(reinterpret_cast<__gm__ float *>(out),
        reinterpret_cast<__gm__ half *>(src0), reinterpret_cast<__gm__ half *>(src1), shmem, false);
#endif
    // __DAV_VEC__: idle
}

// ============================================================================
// Kernel 3: AllReduce Only (non-overlap mode, step 2)
// Only Vec path is active. Cube path is idle.
// GEMM is already complete on all ranks before this kernel launches.
// ============================================================================
template <typename T, uint32_t blockDim, uint32_t m, uint32_t k, uint32_t n, uint32_t singleCoreM, uint32_t singleCoreK,
    uint32_t singleCoreN, uint32_t baseM, uint32_t baseK, uint32_t baseN, uint32_t stepM, uint32_t stepKa,
    uint32_t stepKb, uint32_t stepN>
__global__ AICORE void AllReduceOnlyKernel(__gm__ uint8_t *out, __gm__ uint8_t* shmem)
{
    // __DAV_CUBE__: idle
#ifdef __DAV_VEC__
    // is_overlap=false: skip TWAIT since GEMM is already complete on all ranks
    RunAllReduceE2E<float, m, n, singleCoreM, singleCoreN, baseM, baseN>(shmem, out, false);
#endif
}

// ============================================================================
// Host-callable kernel launchers (use shared config from gemm_config.h)
// ============================================================================

// Fused GEMM + AllReduce
template <typename T>
void LaunchGEMME2E(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *shmem, void *stream, bool is_overlap)
{
    GemmAllReduce<T, BLOCK_DIM, GEMM_M, GEMM_K, GEMM_N, 
                    SINGLE_CORE_M, SINGLE_CORE_K, SINGLE_CORE_N, 
                    BASE_M, BASE_K, BASE_N, 
                    STEP_M, STEP_KA, STEP_KB, STEP_N>
        <<<BLOCK_DIM, nullptr, stream>>>(out, src0, src1, shmem, is_overlap);
}

// GEMM only (non-overlap step 1)
template <typename T>
void LaunchGemmOnly(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *shmem, void *stream)
{
    GemmOnlyKernel<T, BLOCK_DIM, GEMM_M, GEMM_K, GEMM_N,
                    SINGLE_CORE_M, SINGLE_CORE_K, SINGLE_CORE_N,
                    BASE_M, BASE_K, BASE_N,
                    STEP_M, STEP_KA, STEP_KB, STEP_N>
        <<<BLOCK_DIM, nullptr, stream>>>(out, src0, src1, shmem);
}

// AllReduce only (non-overlap step 2)
template <typename T>
void LaunchAllReduceOnly(uint8_t *out, uint8_t *shmem, void *stream)
{
    AllReduceOnlyKernel<T, BLOCK_DIM, GEMM_M, GEMM_K, GEMM_N,
                    SINGLE_CORE_M, SINGLE_CORE_K, SINGLE_CORE_N,
                    BASE_M, BASE_K, BASE_N,
                    STEP_M, STEP_KA, STEP_KB, STEP_N>
        <<<BLOCK_DIM, nullptr, stream>>>(out, shmem);
}

template void LaunchGEMME2E<uint16_t>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *shmem, void *stream, bool is_overlap);
template void LaunchGemmOnly<uint16_t>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *shmem, void *stream);
template void LaunchAllReduceOnly<uint16_t>(uint8_t *out, uint8_t *shmem, void *stream);
