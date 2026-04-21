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
 * @file cube_matmul_4buf_kernel.cpp  (v3: correct pipe pairing for L0A/B and L0C)
 *
 * Buffer-ID sync with correct pipeline pairings:
 *
 *   L1 tiles (aMatTile/bMatTile)  IDs 0-3:
 *       get_buf<PIPE_MTE2>(id_l1) ... rls_buf<PIPE_MTE2>(id_l1)  — MTE2 owns L1 write
 *       get_buf<PIPE_MTE1>(id_l1) ... rls_buf<PIPE_MTE1>(id_l1)  — MTE1 owns L1 read
 *
 *   L0A/B tiles (aTile/bTile)     IDs 4-7:
 *       get_buf<PIPE_MTE1>(id_l0) ... rls_buf<PIPE_MTE1>(id_l0)  — MTE1 owns L0A/B write
 *       get_buf<PIPE_M>   (id_l0) ... rls_buf<PIPE_M>   (id_l0)  — CUBE owns L0A/B read
 *
 *   L0C accumulator (cTile)       ID  8:
 *       get_buf<PIPE_M>  (C_BUF_ID) ... rls_buf<PIPE_M>  (C_BUF_ID) — CUBE owns L0C write
 *       get_buf<PIPE_FIX>(C_BUF_ID) ... rls_buf<PIPE_FIX>(C_BUF_ID) — FIXPIPE owns L0C read
 */

#include <pto/pto-inst.hpp>
#include <pto/common/constants.hpp>

using namespace pto;

constexpr int M = 32;
constexpr int K = 16;
constexpr int N = 256;
constexpr int K_ITERS = 64;
constexpr int NUM_BUFS = 4;
constexpr int L0_ID_OFFSET = 4;
constexpr int C_BUF_ID = 8;

template <int pipe>
AICORE void get_buffer(int id) { get_buf(pipe, id, 0); return; }
template <int pipe>
AICORE void rls_buffer(int id) { rls_buf(pipe, id, 0); return; }

template <typename outType, typename inType>
__global__ AICORE void RunCubeMatmul4Buf(__gm__ outType *out, __gm__ inType *src0, __gm__ inType *src1)
{
    using GlobalDataSrc0 = GlobalTensor<inType, pto::Shape<1, 1, 1, M, K>, pto::Stride<M * K, M * K, M * K, K, 1>>;
    using GlobalDataSrc1 = GlobalTensor<inType, pto::Shape<1, 1, 1, K, N>, pto::Stride<K * N, K * N, K * N, N, 1>>;
    using GlobalDataOut = GlobalTensor<outType, pto::Shape<1, 1, 1, M, N>, pto::Stride<M * N, M * N, M * N, N, 1>>;

    GlobalDataOut dstGlobal(out);

    using TileMatAData = Tile<TileType::Mat, inType, M, K, BLayout::ColMajor, M, K, SLayout::RowMajor, 512>;
    using TileMatBData = Tile<TileType::Mat, inType, K, N, BLayout::ColMajor, K, N, SLayout::RowMajor, 512>;
    using LeftTile = TileLeft<inType, M, K, M, K>;
    using RightTile = TileRight<inType, K, N, K, N>;
    using AccTile = TileAcc<outType, M, N, M, N>;

    TileMatAData aMatTile0, aMatTile1, aMatTile2, aMatTile3;
    TASSIGN(aMatTile0, 0x0);
    TASSIGN(aMatTile1, 0x800);
    TASSIGN(aMatTile2, 0x1000);
    TASSIGN(aMatTile3, 0x1800);

    TileMatBData bMatTile0, bMatTile1, bMatTile2, bMatTile3;
    TASSIGN(bMatTile0, 0x10000);
    TASSIGN(bMatTile1, 0x12000);
    TASSIGN(bMatTile2, 0x14000);
    TASSIGN(bMatTile3, 0x16000);

    LeftTile aTile0, aTile1, aTile2, aTile3;
    TASSIGN(aTile0, 0x0);
    TASSIGN(aTile1, 0x400);
    TASSIGN(aTile2, 0x800);
    TASSIGN(aTile3, 0xC00);

    RightTile bTile0, bTile1, bTile2, bTile3;
    TASSIGN(bTile0, 0x0);
    TASSIGN(bTile1, 0x2000);
    TASSIGN(bTile2, 0x4000);
    TASSIGN(bTile3, 0x6000);

    AccTile cTile;
    TASSIGN(cTile, 0x0);

    for (uint32_t k = 0; k < K_ITERS; k++) {
        int id_l1 = k % NUM_BUFS;
        int id_l0 = id_l1 + L0_ID_OFFSET;

        GlobalDataSrc0 src0Global(src0 + k * M * K);
        GlobalDataSrc1 src1Global(src1 + k * K * N);

        // ===== MTE2: GM -> L1 =====
        // Protect L1 slot: wait until MTE1 has consumed it (WAR), then write
#ifndef __PTO_AUTO__
        get_buffer<PIPE_MTE2>(id_l1);
#endif
        if (id_l1 == 0) { TLOAD(aMatTile0, src0Global); TLOAD(bMatTile0, src1Global); }
        else if (id_l1 == 1) { TLOAD(aMatTile1, src0Global); TLOAD(bMatTile1, src1Global); }
        else if (id_l1 == 2) { TLOAD(aMatTile2, src0Global); TLOAD(bMatTile2, src1Global); }
        else { TLOAD(aMatTile3, src0Global); TLOAD(bMatTile3, src1Global); }
        // Signal: L1 slot written, MTE1 can now read
#ifndef __PTO_AUTO__
        rls_buffer<PIPE_MTE2>(id_l1);
#endif

        // ===== MTE1: L1 -> L0A/B =====
        // Wait: L1 data ready (from MTE2), and L0 slot free (from CUBE)
#ifndef __PTO_AUTO__
        get_buffer<PIPE_MTE1>(id_l1);   // RAW: L1 written by MTE2
        get_buffer<PIPE_MTE1>(id_l0);   // WAR: L0A/B consumed by CUBE
#endif
        if (id_l1 == 0) { TMOV(aTile0, aMatTile0); TMOV(bTile0, bMatTile0); }
        else if (id_l1 == 1) { TMOV(aTile1, aMatTile1); TMOV(bTile1, bMatTile1); }
        else if (id_l1 == 2) { TMOV(aTile2, aMatTile2); TMOV(bTile2, bMatTile2); }
        else { TMOV(aTile3, aMatTile3); TMOV(bTile3, bMatTile3); }
        // Signal: L1 slot consumed (MTE2 can reuse), L0A/B written (CUBE can read)
#ifndef __PTO_AUTO__
        rls_buffer<PIPE_MTE1>(id_l1);   // WAR release: L1 slot free for MTE2
        rls_buffer<PIPE_MTE1>(id_l0);   // RAW release: L0A/B ready for CUBE
#endif

        // ===== CUBE: TMATMUL (L0A/B -> L0C) =====
        // Wait: L0A/B data ready (from MTE1), L0C slot free (from FIXPIPE)
#ifndef __PTO_AUTO__
        get_buffer<PIPE_M>(id_l0);      // RAW: L0A/B written by MTE1
        get_buffer<PIPE_M>(C_BUF_ID);   // WAR: L0C consumed by FIXPIPE (or initial free)
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
        // Signal: L0A/B consumed (MTE1 can reuse), L0C updated (FIXPIPE can read for partial store or next acc)
#ifndef __PTO_AUTO__
        rls_buffer<PIPE_M>(id_l0);      // WAR release: L0A/B slot free for MTE1
        rls_buffer<PIPE_M>(C_BUF_ID);   // RAW release: L0C updated, FIXPIPE can read
#endif
    }

    // ===== FIXPIPE: L0C -> GM (TSTORE of AccTile uses FIXPIPE) =====
    // Wait: final L0C result ready from CUBE
#ifndef __PTO_AUTO__
    get_buffer<PIPE_FIX>(C_BUF_ID);
#endif
    TSTORE(dstGlobal, cTile);
    // Signal: L0C consumed (not strictly needed at end, but good practice)
#ifndef __PTO_AUTO__
    rls_buffer<PIPE_FIX>(C_BUF_ID);
#endif

    out = dstGlobal.data();
}

void LaunchCubeMatmul4Buf(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream)
{
    RunCubeMatmul4Buf<float, half><<<1, nullptr, stream>>>(
        reinterpret_cast<float *>(out),
        reinterpret_cast<half *>(src0),
        reinterpret_cast<half *>(src1));
}
