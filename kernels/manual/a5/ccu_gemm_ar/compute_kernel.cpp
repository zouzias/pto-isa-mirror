/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// GEMM Compute Kernel for CCU GEMM+AllReduce Fusion.
// On tile completion: AtomicAdd into HCCL-window groupDone[flat]
// (peer AIV TNOTIFY owner groupReady; owner TriggerProgressCke).
//
// Addressing: gemm_output is a packed communication buffer. In grouped mode the
// packed order is owner-major so adjacent owner-local tiles can be sent as one
// larger CCU WQE. The final user-visible output is unpacked back to row-major
// after communication.

#ifndef PIPE_FIX
#define PIPE_FIX static_cast<pipe_t>(10)
#endif

#include <pto/common/constants.hpp>
#include <pto/pto-inst.hpp>
#include <pto/common/pto_tile.hpp>
#include "config.h"
#include "kernel_launchers.h"

using namespace pto;

constexpr uint32_t G_K_LOOP = G_K / G_BASE_K;

constexpr uint32_t G_STEP_KA = 4;
constexpr uint32_t G_STEP_KB = 4;
static_assert(G_K_LOOP % G_STEP_KA == 0, "G_K_LOOP must be divisible by G_STEP_KA");
static_assert(G_K_LOOP % G_STEP_KB == 0, "G_K_LOOP must be divisible by G_STEP_KB");
static_assert(G_STEP_KA == G_STEP_KB, "Current implementation assumes stepKa == stepKb");
static_assert(G_K_LOOP >= G_STEP_KA, "K_LOOP must be >= stepKa for L1 caching");

using TileMatAData = Tile<
    TileType::Mat, half, G_BASE_M, G_BASE_K * G_STEP_KA, BLayout::ColMajor, G_BASE_M, G_BASE_K * G_STEP_KA,
    SLayout::RowMajor>;
using TileMatBData = Tile<
    TileType::Mat, half, G_BASE_K * G_STEP_KB, G_BASE_N, BLayout::RowMajor, G_BASE_K * G_STEP_KB, G_BASE_N,
    SLayout::ColMajor>;
using LeftTileT = TileLeft<half, G_BASE_M, G_BASE_K, G_BASE_M, G_BASE_K>;
using RightTileT = TileRight<half, G_BASE_K, G_BASE_N, G_BASE_K, G_BASE_N>;
using ResTileT = TileAcc<float, G_BASE_M, G_BASE_N, G_BASE_M, G_BASE_N>;

AICORE inline void SwizzleTileIndex(int linear_idx, uint32_t& mi, uint32_t& ni)
{
    constexpr uint32_t SWIZZLE_GROUP = G_N_TILES;
    uint32_t group = linear_idx / SWIZZLE_GROUP;
    uint32_t local = linear_idx % SWIZZLE_GROUP;
    mi = group;
    ni = (group & 1) ? (SWIZZLE_GROUP - 1 - local) : local;
    if (mi >= G_M_TILES) {
        mi = linear_idx / G_N_TILES;
        ni = linear_idx % G_N_TILES;
    }
}

template <
    typename T, typename U, typename S, int M, int K, int N, uint32_t baseM, uint32_t baseK, uint32_t baseN,
    uint32_t stepKa, uint32_t stepKb>
AICORE inline void ProcessKIteration(
    uint32_t kIter, __gm__ U* currentSrc0, __gm__ S* currentSrc1,
    Tile<TileType::Mat, U, baseM, baseK * stepKa, BLayout::ColMajor, baseM, baseK * stepKa, SLayout::RowMajor>
        aMatTile[2],
    Tile<TileType::Mat, S, baseK * stepKb, baseN, BLayout::RowMajor, baseK * stepKb, baseN, SLayout::ColMajor>
        bMatTile[2],
    TileLeft<U, baseM, baseK, baseM, baseK> aTile[2], TileRight<S, baseK, baseN, baseK, baseN> bTile[2],
    TileAcc<T, baseM, baseN, baseM, baseN>& cTile, uint8_t& mte2DBFlag, uint8_t& mte1DBFlag, uint32_t k_stride)
{
    using NDValidShapeA = TileShape2D<U, baseM, baseK * stepKa, Layout::ND>;
    using NDsingleCoreShapeA = BaseShape2D<U, M, DYNAMIC, Layout::ND>;
    using GlobalDataSrcA = GlobalTensor<U, NDValidShapeA, NDsingleCoreShapeA, Layout::ND>;

    using NDValidShapeB = TileShape2D<U, baseK * stepKb, baseN, Layout::DN>;
    using NDsingleCoreShapeB = BaseShape2D<U, DYNAMIC, N, Layout::DN>;
    using GlobalDataSrcB = GlobalTensor<U, NDValidShapeB, NDsingleCoreShapeB, Layout::DN>;

    const uint32_t kModStepKa = kIter % stepKa;

    if (kModStepKa == 0) {
        NDsingleCoreShapeA aStride(M, k_stride);
        NDsingleCoreShapeB bStride(k_stride, N);
        NDValidShapeA aShape;
        NDValidShapeB bShape;
        GlobalDataSrcA gmA(currentSrc0 + kIter * baseK, aShape, aStride);
        GlobalDataSrcB gmB(currentSrc1 + kIter * baseK, bShape, bStride);

        wait_flag(PIPE_MTE1, PIPE_MTE2, (event_t)mte2DBFlag);
        TLOAD(aMatTile[mte2DBFlag], gmA);
        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
        TLOAD(bMatTile[mte2DBFlag], gmB);
        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
        mte2DBFlag = (mte2DBFlag == 0) ? 1 : 0;
    }

    const uint32_t currMte2Idx = (mte2DBFlag == 0) ? 1 : 0;

    wait_flag(PIPE_M, PIPE_MTE1, (event_t)mte1DBFlag);

    if (kModStepKa == 0)
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    TEXTRACT(aTile[mte1DBFlag], aMatTile[currMte2Idx], 0, kModStepKa * baseK);

    if (kModStepKa == 0)
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
    TEXTRACT(bTile[mte1DBFlag], bMatTile[currMte2Idx], (kIter % stepKb) * baseK, 0);

    if ((kIter + 1) % stepKa == 0) {
        set_flag(PIPE_MTE1, PIPE_MTE2, (event_t)currMte2Idx);
    }

    set_flag(PIPE_MTE1, PIPE_M, (event_t)mte1DBFlag);
    wait_flag(PIPE_MTE1, PIPE_M, (event_t)mte1DBFlag);
    if (kIter == 0) {
        TMATMUL(cTile, aTile[mte1DBFlag], bTile[mte1DBFlag]);
    } else {
        TMATMUL_ACC(cTile, cTile, aTile[mte1DBFlag], bTile[mte1DBFlag]);
    }
    set_flag(PIPE_M, PIPE_MTE1, (event_t)mte1DBFlag);
    mte1DBFlag = (mte1DBFlag == 0) ? 1 : 0;
}

AICORE inline void ComputeAndStoreTile(
    __gm__ half* gemm_output, __gm__ half* src0, __gm__ half* src1, TileMatAData aMatTile[2], TileMatBData bMatTile[2],
    LeftTileT aTile[2], RightTileT bTile[2], ResTileT& cTile, int linear_idx, uint32_t mi, uint32_t ni,
    uint32_t k_per_rank, uint32_t rank_size, volatile __gm__ int32_t* groupDoneBase)
{
    using NDValidShapeC = TileShape2D<half, G_BASE_M, G_BASE_N>;
    using NDWholeShapeC = BaseShape2D<half, G_BASE_M, G_BASE_N>;
    using GlobalDataOut = GlobalTensor<half, NDValidShapeC, NDWholeShapeC>;

    __gm__ half* currentSrc0 = src0 + mi * G_BASE_M * k_per_rank;
    __gm__ half* currentSrc1 = src1 + ni * G_BASE_N * k_per_rank;

    uint8_t mte2DBFlag = 0, mte1DBFlag = 0;
    uint32_t k_loop_per_rank = k_per_rank / G_BASE_K;
    static_assert(G_BASE_K == 64, "G_BASE_K must be 64 for this implementation");

    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);

    for (uint32_t kIter = 0; kIter < k_loop_per_rank; kIter++) {
        ProcessKIteration<float, half, half, G_M, G_K, G_N, G_BASE_M, G_BASE_K, G_BASE_N, G_STEP_KA, G_STEP_KB>(
            kIter, currentSrc0, currentSrc1, aMatTile, bMatTile, aTile, bTile, cTile, mte2DBFlag, mte1DBFlag,
            k_per_rank);
    }

    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);

    const int32_t tile_idx = static_cast<int32_t>(mi * G_N_TILES + ni);
    const uint32_t packedTile = CcuPackedTileIndex(static_cast<uint32_t>(tile_idx), G_NUM_TILES, rank_size);
    uint64_t outOffset = static_cast<uint64_t>(packedTile) * G_BASE_M * G_BASE_N;
    GlobalDataOut dstGlobal(gemm_output + outOffset);

    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    TSTORE(dstGlobal, cTile);

    // Fence Cube TSTORE (PIPE_FIX) before signaling AIV.
    // MTE1/MTE2/M are already drained by the wait_flags above; PIPE_ALL
    // would stall the whole core unnecessarily on every tile.
    pipe_barrier(PIPE_FIX);
    dsb(DSB_DDR);

    // AtomicAdd into owner-scoped group counter.
    const uint32_t safeRanks = (rank_size > 0) ? rank_size : 1;
    const uint32_t owner = static_cast<uint32_t>(tile_idx) % safeRanks;
    const uint32_t ownerLocalIdx = static_cast<uint32_t>(tile_idx) / safeRanks;
    const uint32_t ownerLocalGroup = ownerLocalIdx / G_COMM_GROUP_TILES;
    const uint32_t flat = CcuOwnerGroupFlatIndex(owner, ownerLocalGroup, G_NUM_TILES, safeRanks);
    __gm__ int32_t* counterPtr = const_cast<__gm__ int32_t*>(groupDoneBase) + flat * kGroupDoneStride;
    dcci(counterPtr, SINGLE_CACHE_LINE);
    st_atomic<int32_t>(1, counterPtr);
    dcci(counterPtr, SINGLE_CACHE_LINE);
    dsb(DSB_DDR);
}

AICORE inline void GemmComputeImpl(
    __gm__ half* gemm_output, __gm__ half* src0, __gm__ half* src1, int compute_block_idx, int compute_block_count,
    uint32_t k_per_rank, uint32_t rank_size, volatile __gm__ int32_t* groupDoneBase)
{
    TileMatAData aMatTile[2];
    TileMatBData bMatTile[2];
    constexpr size_t l1ASize = G_BASE_M * G_BASE_K * G_STEP_KA * sizeof(half);
    constexpr size_t l1BSize = G_BASE_K * G_STEP_KB * G_BASE_N * sizeof(half);
    TASSIGN(aMatTile[0], 0x0);
    TASSIGN(aMatTile[1], 0x0 + l1ASize);
    TASSIGN(bMatTile[0], 0x0 + 2 * l1ASize);
    TASSIGN(bMatTile[1], 0x0 + 2 * l1ASize + l1BSize);

    LeftTileT aTile[2];
    RightTileT bTile[2];
    ResTileT cTile;
    TASSIGN(aTile[0], 0x0);
    TASSIGN(aTile[1], 0x0 + G_BASE_M * G_BASE_K * sizeof(half));
    TASSIGN(bTile[0], 0x0);
    TASSIGN(bTile[1], 0x0 + G_BASE_K * G_BASE_N * sizeof(half));
    TASSIGN(cTile, 0x0);

    const int total_tiles = G_NUM_TILES;

    set_st_atomic_cfg(ATOMIC_S32, ATOMIC_SUM);

    // Round-robin tile assignment: block i produces tiles i, i+B, i+2B, ...
    for (int linear_idx = compute_block_idx; linear_idx < total_tiles; linear_idx += compute_block_count) {
        uint32_t mi, ni;
        SwizzleTileIndex(linear_idx, mi, ni);
        ComputeAndStoreTile(
            gemm_output, src0, src1, aMatTile, bMatTile, aTile, bTile, cTile, linear_idx, mi, ni, k_per_rank, rank_size,
            groupDoneBase);
    }
}

__global__ AICORE void CcuGemmArComputeKernel(
    __gm__ uint8_t* gemm_output, __gm__ uint8_t* src0, __gm__ uint8_t* src1, __gm__ int32_t* groupDone, int rank,
    int launch_block_count, uint32_t k_per_rank, uint32_t rank_size)
{
    (void)rank;
    const int core_idx = get_block_idx();
    if (core_idx >= launch_block_count)
        return;

    GemmComputeImpl(
        reinterpret_cast<__gm__ half*>(gemm_output), reinterpret_cast<__gm__ half*>(src0),
        reinterpret_cast<__gm__ half*>(src1), core_idx, launch_block_count, k_per_rank, rank_size, groupDone);
}

void launchCcuGemmArCompute(
    uint8_t* gemm_output, uint8_t* src0, uint8_t* src1, int32_t* groupDone, int rank, void* stream,
    int compute_block_count, uint32_t k_per_rank, uint32_t rank_size)
{
    CcuGemmArComputeKernel<<<compute_block_count, nullptr, stream>>>(
        gemm_output, src0, src1, groupDone, rank, compute_block_count, k_per_rank, rank_size);
}
