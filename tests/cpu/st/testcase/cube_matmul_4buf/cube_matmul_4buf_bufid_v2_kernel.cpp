/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

/**
 * @file cube_matmul_4buf_bufid_v2_kernel.cpp
 * @brief 4-Buffer Matmul Kernel with corrected Buffer-ID Sync
 *
 * Fix vs v1: Separate buffer IDs for different memory levels.
 * L1 tiles (aMatTile/bMatTile) and L0 tiles (aTile/bTile) are physically
 * different buffers and must use different buffer IDs.
 *
 * Buffer ID allocation:
 *   IDs 0-3  : L1 A+B tiles (aMatTile/bMatTile) — MTE2 <-> MTE1
 *   IDs 4-7  : L0 A+B tiles (aTile/bTile)       — MTE1 <-> PIPE_M
 *   ID  8    : L0C accumulator (cTile)            — PIPE_M <-> MTE3/FIX
 *
 * Pipeline flow:
 *   MTE2: get_buf<MTE2>(id_l1) -> TLOAD A,B to L1 -> rls_buf<MTE2>(id_l1)
 *   MTE1: get_buf<MTE1>(id_l1) -> TMOV A,B to L0  -> rls_buf<MTE1>(id_l1)
 *         get_buf<MTE1>(id_l0) [release L0 slot]   -> rls_buf<MTE1>(id_l0)
 *   M:    get_buf<M>(id_l0)    -> TMATMUL/ACC      -> rls_buf<M>(id_l0)
 *         get_buf<M>(C_BUF_ID) [accumulator slot]  -> rls_buf<M>(C_BUF_ID)
 */

#include <pto/pto-inst.hpp>
#include <pto/common/constants.hpp>

using namespace pto;

constexpr int M = 32;
constexpr int K = 16;
constexpr int N = 256;
constexpr int K_ITERS = 64;
constexpr int NUM_BUFS = 4;
// L1 tile IDs: 0-3, L0 tile IDs: 4-7, C tile ID: 8
constexpr int L0_ID_OFFSET = 4;
constexpr int C_BUF_ID = 8;

// CPU stub for get_buf/rls_buf (no-op in CPU sim)
template <pipe_t pipe>
inline void get_buf(int buf_id)
{
    (void)buf_id;
}

template <pipe_t pipe>
inline void rls_buf(int buf_id)
{
    (void)buf_id;
}

template <typename outType, typename inType>
__global__ AICORE void RunCubeMatmul4BufBufIdV2(__gm__ outType *out, __gm__ inType *src0, __gm__ inType *src1)
{
    using GlobalDataSrc0 = GlobalTensor<inType, Shape<1, 1, 1, M, K>, Stride<M * K, M * K, M * K, K, 1>>;
    using GlobalDataSrc1 = GlobalTensor<inType, Shape<1, 1, 1, K, N>, Stride<K * N, K * N, K * N, N, 1>>;
    using GlobalDataOut = GlobalTensor<outType, Shape<1, 1, 1, M, N>, Stride<M * N, M * N, M * N, N, 1>>;

    GlobalDataOut dstGlobal(out);

    // L1 buffer types (mat address space)
    using TileMatAData = Tile<TileType::Mat, inType, M, K, BLayout::ColMajor, M, K, SLayout::RowMajor, 512>;
    using TileMatBData = Tile<TileType::Mat, inType, K, N, BLayout::ColMajor, K, N, SLayout::RowMajor, 512>;

    // L0 buffer types
    using LeftTile = TileLeft<inType, M, K, M, K>;
    using RightTile = TileRight<inType, K, N, K, N>;
    using AccTile = TileAcc<outType, M, N, M, N>;

    // 4 L1 buffers for A tiles
    TileMatAData aMatTile0, aMatTile1, aMatTile2, aMatTile3;
    TASSIGN(aMatTile0, 0x0);
    TASSIGN(aMatTile1, 0x800);
    TASSIGN(aMatTile2, 0x1000);
    TASSIGN(aMatTile3, 0x1800);

    // 4 L1 buffers for B tiles
    TileMatBData bMatTile0, bMatTile1, bMatTile2, bMatTile3;
    TASSIGN(bMatTile0, 0x10000);
    TASSIGN(bMatTile1, 0x12000);
    TASSIGN(bMatTile2, 0x14000);
    TASSIGN(bMatTile3, 0x16000);

    // 4 L0A buffers for A tiles
    LeftTile aTile0, aTile1, aTile2, aTile3;
    TASSIGN(aTile0, 0x0);
    TASSIGN(aTile1, 0x400);
    TASSIGN(aTile2, 0x800);
    TASSIGN(aTile3, 0xC00);

    // 4 L0B buffers for B tiles
    RightTile bTile0, bTile1, bTile2, bTile3;
    TASSIGN(bTile0, 0x0);
    TASSIGN(bTile1, 0x2000);
    TASSIGN(bTile2, 0x4000);
    TASSIGN(bTile3, 0x6000);

    // L0C accumulator
    AccTile cTile;
    TASSIGN(cTile, 0x0);

    for (uint32_t k = 0; k < K_ITERS; k++) {
        int id_l1 = k % NUM_BUFS;           // IDs 0-3: L1 tiles
        int id_l0 = id_l1 + L0_ID_OFFSET;   // IDs 4-7: L0 tiles

        GlobalDataSrc0 src0Global(src0 + k * M * K);
        GlobalDataSrc1 src1Global(src1 + k * K * N);

        // ===== MTE2 Stage: GM -> L1 =====
#ifndef __PTO_AUTO__
        get_buf<PIPE_MTE2>(id_l1);
#endif
        if (id_l1 == 0) { TLOAD(aMatTile0, src0Global); TLOAD(bMatTile0, src1Global); }
        else if (id_l1 == 1) { TLOAD(aMatTile1, src0Global); TLOAD(bMatTile1, src1Global); }
        else if (id_l1 == 2) { TLOAD(aMatTile2, src0Global); TLOAD(bMatTile2, src1Global); }
        else { TLOAD(aMatTile3, src0Global); TLOAD(bMatTile3, src1Global); }
#ifndef __PTO_AUTO__
        rls_buf<PIPE_MTE2>(id_l1);
#endif

        // ===== MTE1 Stage: L1 -> L0 =====
#ifndef __PTO_AUTO__
        get_buf<PIPE_MTE1>(id_l1);   // Wait for L1 data ready (from MTE2)
        get_buf<PIPE_MTE1>(id_l0);   // Wait for L0 slot free (from CUBE)
#endif
        if (id_l1 == 0) { TMOV(aTile0, aMatTile0); TMOV(bTile0, bMatTile0); }
        else if (id_l1 == 1) { TMOV(aTile1, aMatTile1); TMOV(bTile1, bMatTile1); }
        else if (id_l1 == 2) { TMOV(aTile2, aMatTile2); TMOV(bTile2, bMatTile2); }
        else { TMOV(aTile3, aMatTile3); TMOV(bTile3, bMatTile3); }
#ifndef __PTO_AUTO__
        rls_buf<PIPE_MTE1>(id_l1);   // Release L1 slot back to MTE2
        rls_buf<PIPE_MTE1>(id_l0);   // Signal L0 data ready to CUBE
#endif

        // ===== CUBE Stage: TMATMUL =====
#ifndef __PTO_AUTO__
        get_buf<PIPE_M>(id_l0);      // Wait for L0 data ready (from MTE1)
        get_buf<PIPE_M>(C_BUF_ID);   // Wait for accumulator slot free
#endif
        if (k == 0) {
            if (id_l1 == 0) TMATMUL(cTile, aTile0, bTile0);
            else if (id_l1 == 1) TMATMUL(cTile, aTile1, bTile1);
            else if (id_l1 == 2) TMATMUL(cTile, aTile2, bTile2);
            else TMATMUL(cTile, aTile3, bTile3);
        } else {
            if (id_l1 == 0) TMATMUL_ACC(cTile, cTile, aTile0, bTile0);
            else if (id_l1 == 1) TMATMUL_ACC(cTile, cTile, aTile1, bTile1);
            else if (id_l1 == 2) TMATMUL_ACC(cTile, cTile, aTile2, bTile2);
            else TMATMUL_ACC(cTile, cTile, aTile3, bTile3);
        }
#ifndef __PTO_AUTO__
        rls_buf<PIPE_M>(id_l0);      // Release L0 slot back to MTE1
        rls_buf<PIPE_M>(C_BUF_ID);   // Signal accumulator updated
#endif
    }

    // ===== Store result: L0C -> GM =====
#ifndef __PTO_AUTO__
    get_buf<PIPE_MTE3>(C_BUF_ID);
#endif
    TSTORE(dstGlobal, cTile);
#ifndef __PTO_AUTO__
    rls_buf<PIPE_MTE3>(C_BUF_ID);
#endif

    out = dstGlobal.data();
}

template <typename T_A, typename T_B, typename T_C>
void LaunchCubeMatmul4BufBufIdV2(T_A *a, T_B *b, T_C *c, void *stream)
{
    RunCubeMatmul4BufBufIdV2<T_C, T_A>(c, a, b);
}

template void LaunchCubeMatmul4BufBufIdV2<half, half, float>(half *, half *, float *, void *);
