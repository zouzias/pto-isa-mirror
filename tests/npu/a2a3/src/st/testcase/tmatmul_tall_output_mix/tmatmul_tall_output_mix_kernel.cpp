/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

/**
 * Two differently-sized tiles through one cross-core ring, then a matmul.
 *
 * The vector core loads both matmul operands and pushes them through the pipe; the cube pops
 * both, moves them to Left/Right and multiplies. Both tiles are popped before either is
 * consumed, which uses 2 of the ring's 8 slots -- what a FIFO of depth > 1 is for.
 *
 * That is wrong on a2a3 whenever the two tiles differ in size, because the consumer's local
 * slot is strided by the popped tile's own size rather than by the ring's SLOT_SIZE, so slot
 * 1 of the smaller tile lands INSIDE slot 0 of the larger one. For [M,K] @ [K,N] the operands
 * occupy M*K and K*N bytes at local offsets 0 and K*N, which overlap iff K*N < M*K, i.e.
 * N < M -- K cancels. case1/case3 (and case4/case6 over a DIR_BOTH pipe) are exactly that;
 * case2/case5 use equal-sized tiles, where every slot is SLOT_SIZE anyway, and pass either
 * way.
 *
 * The companion `tmatmul_tall_output` runs the same shapes cube-only, with no pipe, and
 * passes -- so this is the ring, not TMATMUL or TSTORE.
 */

#include <pto/pto-inst.hpp>
#include <pto/common/fifo.hpp>

using namespace pto;

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

template <typename T>
AICORE constexpr inline T CeilAlign(T num_1, T num_2)
{
    if (num_2 == 0) {
        return 0;
    }
    return (num_1 + num_2 - 1) / num_2 * num_2;
}

template <typename T, int validM, int validK, int validN, uint8_t DIR = Direction::DIR_V2C>
__global__ AICORE void runMixedTallMatmul(
    __gm__ uint64_t* ffts_addr, __gm__ T* out, __gm__ T* srcA, __gm__ T* srcB, __gm__ T* fifoMem)
{
    set_ffts_base_addr((uint64_t)ffts_addr);

    constexpr uint16_t FLAG_ID = 0;
    // Slot must hold the larger of the two operands; both cross the same ring.
    constexpr uint32_t SLOT_ELEMS = (validM * validK > validK * validN) ? validM * validK : validK * validN;
    constexpr uint32_t SLOT_SIZE = SLOT_ELEMS * sizeof(T);
    constexpr uint32_t SLOT_NUM = 8;
    constexpr uint32_t LOCAL_SLOT_NUM = 8;
    constexpr uint32_t localFiFoBase = 0x0;

    // DIR is the ONLY difference between the V2C and DIR_BOTH cases: same data flow (vector
    // produces both operands, cube consumes them), same slot geometry, same tiles. A
    // DIR_BOTH pipe driven in one direction only is well-defined here -- the destructor
    // drains the producer side alone, and the unused C2V half is simply never signalled --
    // but it does need TWO rings of GM, which the host allocates (see main.cpp).
    using MatPipe = TPipe<FLAG_ID, DIR, SLOT_SIZE, SLOT_NUM, LOCAL_SLOT_NUM, true>;
    MatPipe mPipe((__gm__ void*)fifoMem, 0x0, localFiFoBase);

    constexpr uint32_t blockAlign = C0_SIZE_BYTE / sizeof(T);
    constexpr uint32_t ALIGNED_M = CeilAlign<uint32_t>(validM, 16);
    constexpr uint32_t ALIGNED_K = CeilAlign<uint32_t>(validK, blockAlign);
    constexpr uint32_t ALIGNED_N = CeilAlign<uint32_t>(validN, blockAlign);

    using GlobalA = GlobalTensor<
        T, pto::Shape<1, 1, 1, validM, validK>,
        pto::Stride<validM * validK, validM * validK, validM * validK, validK, 1>>;
    using GlobalB = GlobalTensor<
        T, pto::Shape<1, 1, 1, validK, validN>,
        pto::Stride<validK * validN, validK * validN, validK * validN, validN, 1>>;
    using GlobalOut = GlobalTensor<
        T, pto::Shape<1, 1, 1, validM, validN>,
        pto::Stride<validM * validN, validM * validN, validM * validN, validN, 1>>;

    using VecTileA = Tile<TileType::Vec, T, validM, validK, BLayout::RowMajor, validM, validK>;
    using VecTileB = Tile<TileType::Vec, T, validK, validN, BLayout::RowMajor, validK, validN>;

    using PopTileA =
        Tile<TileType::Mat, T, ALIGNED_M, ALIGNED_K, BLayout::ColMajor, validM, validK, SLayout::RowMajor, 512>;
    using PopTileB =
        Tile<TileType::Mat, T, ALIGNED_K, ALIGNED_N, BLayout::ColMajor, validK, validN, SLayout::RowMajor, 512>;

    using LeftTile = TileLeft<T, ALIGNED_M, ALIGNED_K, validM, validK>;
    using RightTile = TileRight<T, ALIGNED_K, ALIGNED_N, validK, validN>;
    using AccTile = TileAcc<T, validM, validN, validM, validN>;

    if constexpr (DAV_VEC) {
        VecTileA vecA;
        VecTileB vecB;
        TASSIGN(vecA, 0x0);
        TASSIGN(vecB, 0x20000);

        GlobalA globalA(srcA);
        GlobalB globalB(srcB);

        // Flag ledger, per (pipe, pipe, event) triple. Each of the two pushes consumes the
        // two primed flags and re-arms them; the trailing pair of waits drains what the
        // second push re-armed. Net zero on every triple.
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);

        // ---- operand A ----
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
        TLOAD(vecA, globalA);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        TPUSH<MatPipe, VecTileA, TileSplitAxis::TILE_NO_SPLIT>(mPipe, vecA);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);

        // ---- operand B (the smaller one when validN < validM) ----
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
        TLOAD(vecB, globalB);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        TPUSH<MatPipe, VecTileB, TileSplitAxis::TILE_NO_SPLIT>(mPipe, vecB);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);

        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);

        pipe_barrier(PIPE_ALL);
    }

    if constexpr (DAV_CUBE) {
        PopTileA aMatTile;
        PopTileB bMatTile;

        LeftTile aTile;
        RightTile bTile;
        AccTile accTile;
        TASSIGN(aTile, 0x0);
        TASSIGN(bTile, 0x0);
        TASSIGN(accTile, 0x0);

        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);

        // Two tiles are popped before either is consumed. That uses 2 of the ring's
        // LOCAL_SLOT_NUM (8) slots and is exactly what a FIFO of depth > 1 is for: distinct
        // slots must be distinct storage. No MTE1 -> MTE2 flag is needed between them for
        // the same reason -- with disjoint slots the second TPOP (MTE2) and the first TMOV
        // (MTE1) touch different memory.
        //
        // It fails when the two tiles differ in size, because the consumer's local slot is
        // strided by the popped tile's OWN size rather than by the ring's SLOT_SIZE:
        //   V2C_CONSUMER_BUF + (tileIndex % LOCAL_SLOT_NUM) * ConsM * ConsN * sizeof(T)
        // so slot 1 of B[64,32] lands at base+8192, INSIDE slot 0 of A[64,64] at
        // base+[0,16384), and B overwrites the upper half of A. case2/case5 use equal-sized
        // tiles, where every slot is SLOT_SIZE anyway, and pass either way.
        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
        TPOP<MatPipe, PopTileA, TileSplitAxis::TILE_NO_SPLIT>(mPipe, aMatTile);
        TPOP<MatPipe, PopTileB, TileSplitAxis::TILE_NO_SPLIT>(mPipe, bMatTile);

        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);

        TMOV(aTile, aMatTile);
        TMOV(bTile, bMatTile);
        TFREE<MatPipe, TileSplitAxis::TILE_NO_SPLIT>(mPipe);
        TFREE<MatPipe, TileSplitAxis::TILE_NO_SPLIT>(mPipe);

        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);

        TMATMUL(accTile, aTile, bTile);

        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);

        set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
        wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);

        GlobalOut globalOut(out);
        TSTORE<AccTile, GlobalOut>(globalOut, accTile);

        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);

        pipe_barrier(PIPE_ALL);
    }
}

template <int32_t tilingKey>
void LaunchMixedTallMatmul(uint8_t* ffts, uint8_t* out, uint8_t* srcA, uint8_t* srcB, uint8_t* fifoMem, void* stream)
{
    if constexpr (tilingKey == 1) {
        // TALL: 64x64 @ 64x32 -> 64x32, operands via the V2C pipe.
        runMixedTallMatmul<float, 64, 64, 32><<<1, nullptr, stream>>>(
            reinterpret_cast<uint64_t*>(ffts), reinterpret_cast<float*>(out), reinterpret_cast<float*>(srcA),
            reinterpret_cast<float*>(srcB), reinterpret_cast<float*>(fifoMem));
    } else if constexpr (tilingKey == 2) {
        // SQUARE control on the identical path.
        runMixedTallMatmul<float, 64, 64, 64><<<1, nullptr, stream>>>(
            reinterpret_cast<uint64_t*>(ffts), reinterpret_cast<float*>(out), reinterpret_cast<float*>(srcA),
            reinterpret_cast<float*>(srcB), reinterpret_cast<float*>(fifoMem));
    } else if constexpr (tilingKey == 3) {
        // TALL at a second shape.
        runMixedTallMatmul<float, 32, 32, 16><<<1, nullptr, stream>>>(
            reinterpret_cast<uint64_t*>(ffts), reinterpret_cast<float*>(out), reinterpret_cast<float*>(srcA),
            reinterpret_cast<float*>(srcB), reinterpret_cast<float*>(fifoMem));
    } else if constexpr (tilingKey == 4) {
        // Same as case1 but over a DIR_BOTH pipe instead of DIR_V2C.
        runMixedTallMatmul<float, 64, 64, 32, Direction::DIR_BOTH><<<1, nullptr, stream>>>(
            reinterpret_cast<uint64_t*>(ffts), reinterpret_cast<float*>(out), reinterpret_cast<float*>(srcA),
            reinterpret_cast<float*>(srcB), reinterpret_cast<float*>(fifoMem));
    } else if constexpr (tilingKey == 5) {
        // DIR_BOTH square control.
        runMixedTallMatmul<float, 64, 64, 64, Direction::DIR_BOTH><<<1, nullptr, stream>>>(
            reinterpret_cast<uint64_t*>(ffts), reinterpret_cast<float*>(out), reinterpret_cast<float*>(srcA),
            reinterpret_cast<float*>(srcB), reinterpret_cast<float*>(fifoMem));
    } else if constexpr (tilingKey == 6) {
        // Same as case3 but over a DIR_BOTH pipe.
        runMixedTallMatmul<float, 32, 32, 16, Direction::DIR_BOTH><<<1, nullptr, stream>>>(
            reinterpret_cast<uint64_t*>(ffts), reinterpret_cast<float*>(out), reinterpret_cast<float*>(srcA),
            reinterpret_cast<float*>(srcB), reinterpret_cast<float*>(fifoMem));
    }
}

template void LaunchMixedTallMatmul<1>(
    uint8_t* ffts, uint8_t* out, uint8_t* srcA, uint8_t* srcB, uint8_t* fifoMem, void* stream);
template void LaunchMixedTallMatmul<2>(
    uint8_t* ffts, uint8_t* out, uint8_t* srcA, uint8_t* srcB, uint8_t* fifoMem, void* stream);
template void LaunchMixedTallMatmul<3>(
    uint8_t* ffts, uint8_t* out, uint8_t* srcA, uint8_t* srcB, uint8_t* fifoMem, void* stream);
template void LaunchMixedTallMatmul<4>(
    uint8_t* ffts, uint8_t* out, uint8_t* srcA, uint8_t* srcB, uint8_t* fifoMem, void* stream);
template void LaunchMixedTallMatmul<5>(
    uint8_t* ffts, uint8_t* out, uint8_t* srcA, uint8_t* srcB, uint8_t* fifoMem, void* stream);
template void LaunchMixedTallMatmul<6>(
    uint8_t* ffts, uint8_t* out, uint8_t* srcA, uint8_t* srcB, uint8_t* fifoMem, void* stream);
