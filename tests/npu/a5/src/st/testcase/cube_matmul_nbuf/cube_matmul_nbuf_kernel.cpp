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
 * @file cube_matmul_nbuf_kernel.cpp
 *
 * N-buffer cube matmul benchmark — 5 configs, all M=32 K_total=1024 N=256 fp16->fp32.
 * Verified against cube_matmul_4buf_kernel.cpp (same buffer/address scheme).
 *
 * GM MEMORY LAYOUT (SPLIT_K tile-major)
 * ─────────────────────────────────────
 * A and B are stored as K_ITERS contiguous tiles so the kernel can index each
 * tile with a simple base-pointer arithmetic:
 *
 *   A_gm.bin : [K_ITERS, M, K_TILE] fp16 row-major
 *              tile k starts at element offset  k * M * K_TILE
 *              i.e.  byte offset  k * M * K_TILE * 2
 *              Kernel pointer:  src0 + k * K
 *
 *   B_gm.bin : [K_ITERS, K_TILE, N] fp16 row-major
 *              tile k starts at element offset  k * K_TILE * N
 *              Kernel pointer:  src1 + k * K * GM_N
 *
 *   golden.bin : [M, N] fp32 = sum_k( A_tile[k] @ B_tile[k] )
 *
 * This is identical to the layout used by cube_matmul_4buf.
 *
 * GlobalTensor Shape/Stride per iteration
 * ─────────────────────────────────────────
 *   src0: Shape<1,1,1, M, K>     Stride<M*K, M*K, M*K, K, 1>   (A tile, row-major)
 *   src1: Shape<1,1,1, K, N>     Stride<K*N, K*N, K*N, N, 1>   (B tile, row-major)
 *
 * TASSIGN (L1/L0 buffer addresses)
 * ─────────────────────────────────────────
 *   L1 A slot stride = 0x800  (2 KB, NZ padded)   K16; 0x1000 for K32
 *   L1 B slot stride = 0x2000 (8 KB)              K16; 0x4000 for K32
 *   L0A slot stride  = 0x400  (1 KB)              K16; 0x800  for K32
 *   L0B slot stride  = 0x2000 (8 KB)              K16; 0x4000 for K32
 *
 * Buffer ID allocation per config
 * ─────────────────────────────────────────
 *   2-buf k16 : L1 0-1,  L0 2-3,   C=4   (5 IDs)
 *   4-buf k16 : L1 0-3,  L0 4-7,   C=8   (9 IDs)
 *   8-buf k16 : L1 0-7,  L0 8-15,  C=16  (17 IDs)
 *   2-buf k32 : L1 0-1,  L0 2-3,   C=4   (5 IDs)
 *   4-buf k32 : L1 0-3,  L0 4-7,   C=8   (9 IDs)
 *   Max IDs used: 17  (within the 32-ID hardware limit)
 */

#include <pto/pto-inst.hpp>
#include <pto/common/constants.hpp>

using namespace pto;

static constexpr int GM_M = 32;
static constexpr int GM_N = 256;
static constexpr int GM_K = 1024;  // total K dimension

template <int pipe>
AICORE void get_buffer(int id) { get_buf(pipe, id, 0); return; }
template <int pipe>
AICORE void rls_buffer(int id) { rls_buf(pipe, id, 0); return; }

// ═══════════════════════════════════════════════════════════════════════════
// Config 1: 2-buffer ping-pong, K_tile=16, 8KB B tile
//   L1 IDs: 0-1 | L0 IDs: 2-3 | C_BUF_ID: 4
// ═══════════════════════════════════════════════════════════════════════════
template <typename outType, typename inType>
__global__ AICORE void RunCubeMatmul2Buf8K(__gm__ outType *out, __gm__ inType *src0, __gm__ inType *src1)
{
    constexpr int K = 16, K_ITERS = 64, NUM_BUFS = 2, L0_OFF = 2, C_BUF_ID = 4;

    using GlobalDataSrc0 = GlobalTensor<inType,  pto::Shape<1,1,1,GM_M,K>,    pto::Stride<GM_M*GM_K,GM_M*GM_K,GM_M*GM_K,GM_K,1>>;
    using GlobalDataSrc1 = GlobalTensor<inType,  pto::Shape<1,1,1,K,GM_N>,    pto::Stride<GM_K*GM_N,GM_K*GM_N,GM_K*GM_N,GM_N,1>>;
    using GlobalDataOut  = GlobalTensor<outType, pto::Shape<1,1,1,GM_M,GM_N>, pto::Stride<GM_M*GM_N,GM_M*GM_N,GM_M*GM_N,GM_N,1>>;
    using TileMatA = Tile<TileType::Mat, inType,  GM_M,K,    BLayout::ColMajor, GM_M,K,    SLayout::RowMajor, 512>;
    using TileMatB = Tile<TileType::Mat, inType,  K,   GM_N, BLayout::ColMajor, K,   GM_N, SLayout::RowMajor, 512>;
    using LeftTile = TileLeft< inType,  GM_M,K,    GM_M,K>;
    using RightTile= TileRight<inType,  K,   GM_N, K,   GM_N>;
    using AccTile  = TileAcc<  outType, GM_M,GM_N, GM_M,GM_N>;

    GlobalDataOut dstGlobal(out);

    TileMatA aM0, aM1;
    TASSIGN(aM0, 0x0);     TASSIGN(aM1, 0x800);
    TileMatB bM0, bM1;
    TASSIGN(bM0, 0x10000); TASSIGN(bM1, 0x12000);
    LeftTile aL0, aL1;
    TASSIGN(aL0, 0x0);     TASSIGN(aL1, 0x400);
    RightTile bL0, bL1;
    TASSIGN(bL0, 0x0);     TASSIGN(bL1, 0x2000);
    AccTile cTile;
    TASSIGN(cTile, 0x0);

    for (uint32_t k = 0; k < K_ITERS; k++) {
        int s = k % NUM_BUFS;
        int id_l0 = s + L0_OFF;
        GlobalDataSrc0 src0Global(src0 + k * K);
        GlobalDataSrc1 src1Global(src1 + k * K * GM_N);
#ifndef __PTO_AUTO__
        get_buffer<PIPE_MTE2>(s);
#endif
        if (s == 0) { TLOAD(aM0, src0Global); TLOAD(bM0, src1Global); }
        else        { TLOAD(aM1, src0Global); TLOAD(bM1, src1Global); }
#ifndef __PTO_AUTO__
        rls_buffer<PIPE_MTE2>(s);
        get_buffer<PIPE_MTE1>(s);
        get_buffer<PIPE_MTE1>(id_l0);
#endif
        if (s == 0) { TMOV(aL0, aM0); TMOV(bL0, bM0); }
        else        { TMOV(aL1, aM1); TMOV(bL1, bM1); }
#ifndef __PTO_AUTO__
        rls_buffer<PIPE_MTE1>(s);
        rls_buffer<PIPE_MTE1>(id_l0);
        get_buffer<PIPE_M>(id_l0);
        get_buffer<PIPE_M>(C_BUF_ID);
#endif
        if (k == 0) {
            if (s == 0) TMATMUL(cTile, aL0, bL0); else TMATMUL(cTile, aL1, bL1);
        } else {
            if (s == 0) TMATMUL_ACC(cTile, cTile, aL0, bL0); else TMATMUL_ACC(cTile, cTile, aL1, bL1);
        }
#ifndef __PTO_AUTO__
        rls_buffer<PIPE_M>(id_l0);
        rls_buffer<PIPE_M>(C_BUF_ID);
#endif
    }
#ifndef __PTO_AUTO__
    get_buffer<PIPE_FIX>(C_BUF_ID);
#endif
    TSTORE(dstGlobal, cTile);
#ifndef __PTO_AUTO__
    rls_buffer<PIPE_FIX>(C_BUF_ID);
#endif
    out = dstGlobal.data();
}

// ═══════════════════════════════════════════════════════════════════════════
// Config 2: 4-buffer, K_tile=16, 8KB B tile  (identical to reference)
//   L1 IDs: 0-3 | L0 IDs: 4-7 | C_BUF_ID: 8
// ═══════════════════════════════════════════════════════════════════════════
template <typename outType, typename inType>
__global__ AICORE void RunCubeMatmul4Buf8K(__gm__ outType *out, __gm__ inType *src0, __gm__ inType *src1)
{
    constexpr int K = 16, K_ITERS = 64, NUM_BUFS = 4, L0_OFF = 4, C_BUF_ID = 8;

    using GlobalDataSrc0 = GlobalTensor<inType,  pto::Shape<1,1,1,GM_M,K>,    pto::Stride<GM_M*GM_K,GM_M*GM_K,GM_M*GM_K,GM_K,1>>;
    using GlobalDataSrc1 = GlobalTensor<inType,  pto::Shape<1,1,1,K,GM_N>,    pto::Stride<GM_K*GM_N,GM_K*GM_N,GM_K*GM_N,GM_N,1>>;
    using GlobalDataOut  = GlobalTensor<outType, pto::Shape<1,1,1,GM_M,GM_N>, pto::Stride<GM_M*GM_N,GM_M*GM_N,GM_M*GM_N,GM_N,1>>;
    using TileMatA = Tile<TileType::Mat, inType,  GM_M,K,    BLayout::ColMajor, GM_M,K,    SLayout::RowMajor, 512>;
    using TileMatB = Tile<TileType::Mat, inType,  K,   GM_N, BLayout::ColMajor, K,   GM_N, SLayout::RowMajor, 512>;
    using LeftTile = TileLeft< inType,  GM_M,K,    GM_M,K>;
    using RightTile= TileRight<inType,  K,   GM_N, K,   GM_N>;
    using AccTile  = TileAcc<  outType, GM_M,GM_N, GM_M,GM_N>;

    GlobalDataOut dstGlobal(out);

    TileMatA aM0, aM1, aM2, aM3;
    TASSIGN(aM0, 0x0); TASSIGN(aM1, 0x800); TASSIGN(aM2, 0x1000); TASSIGN(aM3, 0x1800);
    TileMatB bM0, bM1, bM2, bM3;
    TASSIGN(bM0, 0x10000); TASSIGN(bM1, 0x12000); TASSIGN(bM2, 0x14000); TASSIGN(bM3, 0x16000);
    LeftTile aL0, aL1, aL2, aL3;
    TASSIGN(aL0, 0x0); TASSIGN(aL1, 0x400); TASSIGN(aL2, 0x800); TASSIGN(aL3, 0xC00);
    RightTile bL0, bL1, bL2, bL3;
    TASSIGN(bL0, 0x0); TASSIGN(bL1, 0x2000); TASSIGN(bL2, 0x4000); TASSIGN(bL3, 0x6000);
    AccTile cTile;
    TASSIGN(cTile, 0x0);

    for (uint32_t k = 0; k < K_ITERS; k++) {
        int s = k % NUM_BUFS;
        int id_l0 = s + L0_OFF;
        GlobalDataSrc0 src0Global(src0 + k * K);
        GlobalDataSrc1 src1Global(src1 + k * K * GM_N);
#ifndef __PTO_AUTO__
        get_buffer<PIPE_MTE2>(s);
#endif
        if      (s==0){TLOAD(aM0,src0Global);TLOAD(bM0,src1Global);}
        else if (s==1){TLOAD(aM1,src0Global);TLOAD(bM1,src1Global);}
        else if (s==2){TLOAD(aM2,src0Global);TLOAD(bM2,src1Global);}
        else          {TLOAD(aM3,src0Global);TLOAD(bM3,src1Global);}
#ifndef __PTO_AUTO__
        rls_buffer<PIPE_MTE2>(s);
        get_buffer<PIPE_MTE1>(s);
        get_buffer<PIPE_MTE1>(id_l0);
#endif
        if      (s==0){TMOV(aL0,aM0);TMOV(bL0,bM0);}
        else if (s==1){TMOV(aL1,aM1);TMOV(bL1,bM1);}
        else if (s==2){TMOV(aL2,aM2);TMOV(bL2,bM2);}
        else          {TMOV(aL3,aM3);TMOV(bL3,bM3);}
#ifndef __PTO_AUTO__
        rls_buffer<PIPE_MTE1>(s);
        rls_buffer<PIPE_MTE1>(id_l0);
        get_buffer<PIPE_M>(id_l0);
        get_buffer<PIPE_M>(C_BUF_ID);
#endif
        if (k == 0) {
            if      (s==0) TMATMUL(cTile,aL0,bL0);
            else if (s==1) TMATMUL(cTile,aL1,bL1);
            else if (s==2) TMATMUL(cTile,aL2,bL2);
            else           TMATMUL(cTile,aL3,bL3);
        } else {
            if      (s==0) TMATMUL_ACC(cTile,cTile,aL0,bL0);
            else if (s==1) TMATMUL_ACC(cTile,cTile,aL1,bL1);
            else if (s==2) TMATMUL_ACC(cTile,cTile,aL2,bL2);
            else           TMATMUL_ACC(cTile,cTile,aL3,bL3);
        }
#ifndef __PTO_AUTO__
        rls_buffer<PIPE_M>(id_l0);
        rls_buffer<PIPE_M>(C_BUF_ID);
#endif
    }
#ifndef __PTO_AUTO__
    get_buffer<PIPE_FIX>(C_BUF_ID);
#endif
    TSTORE(dstGlobal, cTile);
#ifndef __PTO_AUTO__
    rls_buffer<PIPE_FIX>(C_BUF_ID);
#endif
    out = dstGlobal.data();
}

// ═══════════════════════════════════════════════════════════════════════════
// Config 3: 8-buffer, K_tile=16, 8KB B tile
//   L1 IDs: 0-7 | L0 IDs: 8-15 | C_BUF_ID: 16
//   L1 A: 8×0x800=16KB, L1 B: 8×0x2000=64KB, L0A: 8×0x400=8KB, L0B: 8×0x2000=64KB
// ═══════════════════════════════════════════════════════════════════════════
template <typename outType, typename inType>
__global__ AICORE void RunCubeMatmul8Buf8K(__gm__ outType *out, __gm__ inType *src0, __gm__ inType *src1)
{
    constexpr int K = 16, K_ITERS = 64, NUM_BUFS = 8, L0_OFF = 8, C_BUF_ID = 16;

    using GlobalDataSrc0 = GlobalTensor<inType,  pto::Shape<1,1,1,GM_M,K>,    pto::Stride<GM_M*GM_K,GM_M*GM_K,GM_M*GM_K,GM_K,1>>;
    using GlobalDataSrc1 = GlobalTensor<inType,  pto::Shape<1,1,1,K,GM_N>,    pto::Stride<GM_K*GM_N,GM_K*GM_N,GM_K*GM_N,GM_N,1>>;
    using GlobalDataOut  = GlobalTensor<outType, pto::Shape<1,1,1,GM_M,GM_N>, pto::Stride<GM_M*GM_N,GM_M*GM_N,GM_M*GM_N,GM_N,1>>;
    using TileMatA = Tile<TileType::Mat, inType,  GM_M,K,    BLayout::ColMajor, GM_M,K,    SLayout::RowMajor, 512>;
    using TileMatB = Tile<TileType::Mat, inType,  K,   GM_N, BLayout::ColMajor, K,   GM_N, SLayout::RowMajor, 512>;
    using LeftTile = TileLeft< inType,  GM_M,K,    GM_M,K>;
    using RightTile= TileRight<inType,  K,   GM_N, K,   GM_N>;
    using AccTile  = TileAcc<  outType, GM_M,GM_N, GM_M,GM_N>;

    GlobalDataOut dstGlobal(out);

    TileMatA aM0,aM1,aM2,aM3,aM4,aM5,aM6,aM7;
    TASSIGN(aM0,0x0);    TASSIGN(aM1,0x800);  TASSIGN(aM2,0x1000); TASSIGN(aM3,0x1800);
    TASSIGN(aM4,0x2000); TASSIGN(aM5,0x2800); TASSIGN(aM6,0x3000); TASSIGN(aM7,0x3800);
    TileMatB bM0,bM1,bM2,bM3,bM4,bM5,bM6,bM7;
    TASSIGN(bM0,0x10000); TASSIGN(bM1,0x12000); TASSIGN(bM2,0x14000); TASSIGN(bM3,0x16000);
    TASSIGN(bM4,0x18000); TASSIGN(bM5,0x1A000); TASSIGN(bM6,0x1C000); TASSIGN(bM7,0x1E000);
    LeftTile aL0,aL1,aL2,aL3,aL4,aL5,aL6,aL7;
    TASSIGN(aL0,0x0);    TASSIGN(aL1,0x400);  TASSIGN(aL2,0x800);  TASSIGN(aL3,0xC00);
    TASSIGN(aL4,0x1000); TASSIGN(aL5,0x1400); TASSIGN(aL6,0x1800); TASSIGN(aL7,0x1C00);
    RightTile bL0,bL1,bL2,bL3,bL4,bL5,bL6,bL7;
    TASSIGN(bL0,0x0);    TASSIGN(bL1,0x2000); TASSIGN(bL2,0x4000); TASSIGN(bL3,0x6000);
    TASSIGN(bL4,0x8000); TASSIGN(bL5,0xA000); TASSIGN(bL6,0xC000); TASSIGN(bL7,0xE000);
    AccTile cTile;
    TASSIGN(cTile, 0x0);

    for (uint32_t k = 0; k < K_ITERS; k++) {
        int s = k % NUM_BUFS;
        int id_l0 = s + L0_OFF;
        GlobalDataSrc0 src0Global(src0 + k * K);
        GlobalDataSrc1 src1Global(src1 + k * K * GM_N);
#ifndef __PTO_AUTO__
        get_buffer<PIPE_MTE2>(s);
#endif
        switch(s) {
            case 0: TLOAD(aM0,src0Global); TLOAD(bM0,src1Global); break;
            case 1: TLOAD(aM1,src0Global); TLOAD(bM1,src1Global); break;
            case 2: TLOAD(aM2,src0Global); TLOAD(bM2,src1Global); break;
            case 3: TLOAD(aM3,src0Global); TLOAD(bM3,src1Global); break;
            case 4: TLOAD(aM4,src0Global); TLOAD(bM4,src1Global); break;
            case 5: TLOAD(aM5,src0Global); TLOAD(bM5,src1Global); break;
            case 6: TLOAD(aM6,src0Global); TLOAD(bM6,src1Global); break;
            default:TLOAD(aM7,src0Global); TLOAD(bM7,src1Global); break;
        }
#ifndef __PTO_AUTO__
        rls_buffer<PIPE_MTE2>(s);
        get_buffer<PIPE_MTE1>(s);
        get_buffer<PIPE_MTE1>(id_l0);
#endif
        switch(s) {
            case 0: TMOV(aL0,aM0); TMOV(bL0,bM0); break;
            case 1: TMOV(aL1,aM1); TMOV(bL1,bM1); break;
            case 2: TMOV(aL2,aM2); TMOV(bL2,bM2); break;
            case 3: TMOV(aL3,aM3); TMOV(bL3,bM3); break;
            case 4: TMOV(aL4,aM4); TMOV(bL4,bM4); break;
            case 5: TMOV(aL5,aM5); TMOV(bL5,bM5); break;
            case 6: TMOV(aL6,aM6); TMOV(bL6,bM6); break;
            default:TMOV(aL7,aM7); TMOV(bL7,bM7); break;
        }
#ifndef __PTO_AUTO__
        rls_buffer<PIPE_MTE1>(s);
        rls_buffer<PIPE_MTE1>(id_l0);
        get_buffer<PIPE_M>(id_l0);
        get_buffer<PIPE_M>(C_BUF_ID);
#endif
        if (k == 0) {
            switch(s) {
                case 0: TMATMUL(cTile,aL0,bL0); break; case 1: TMATMUL(cTile,aL1,bL1); break;
                case 2: TMATMUL(cTile,aL2,bL2); break; case 3: TMATMUL(cTile,aL3,bL3); break;
                case 4: TMATMUL(cTile,aL4,bL4); break; case 5: TMATMUL(cTile,aL5,bL5); break;
                case 6: TMATMUL(cTile,aL6,bL6); break; default:TMATMUL(cTile,aL7,bL7); break;
            }
        } else {
            switch(s) {
                case 0: TMATMUL_ACC(cTile,cTile,aL0,bL0); break; case 1: TMATMUL_ACC(cTile,cTile,aL1,bL1); break;
                case 2: TMATMUL_ACC(cTile,cTile,aL2,bL2); break; case 3: TMATMUL_ACC(cTile,cTile,aL3,bL3); break;
                case 4: TMATMUL_ACC(cTile,cTile,aL4,bL4); break; case 5: TMATMUL_ACC(cTile,cTile,aL5,bL5); break;
                case 6: TMATMUL_ACC(cTile,cTile,aL6,bL6); break; default:TMATMUL_ACC(cTile,cTile,aL7,bL7); break;
            }
        }
#ifndef __PTO_AUTO__
        rls_buffer<PIPE_M>(id_l0);
        rls_buffer<PIPE_M>(C_BUF_ID);
#endif
    }
#ifndef __PTO_AUTO__
    get_buffer<PIPE_FIX>(C_BUF_ID);
#endif
    TSTORE(dstGlobal, cTile);
#ifndef __PTO_AUTO__
    rls_buffer<PIPE_FIX>(C_BUF_ID);
#endif
    out = dstGlobal.data();
}

// ═══════════════════════════════════════════════════════════════════════════
// Config 4: 2-buffer ping-pong, K_tile=32, 16KB B tile
//   L1 IDs: 0-1 | L0 IDs: 2-3 | C_BUF_ID: 4
//   L1 A: 2×0x1000=8KB (32*32*2=2KB per slot, 0x1000 spaced)
//   L1 B: 2×0x4000=32KB (32*256*2=16KB per slot)
//   L0A:  2×0x800=4KB, L0B: 2×0x4000=32KB
// ═══════════════════════════════════════════════════════════════════════════
template <typename outType, typename inType>
__global__ AICORE void RunCubeMatmul2Buf16K(__gm__ outType *out, __gm__ inType *src0, __gm__ inType *src1)
{
    constexpr int K = 32, K_ITERS = 32, NUM_BUFS = 2, L0_OFF = 2, C_BUF_ID = 4;

    using GlobalDataSrc0 = GlobalTensor<inType,  pto::Shape<1,1,1,GM_M,K>,    pto::Stride<GM_M*GM_K,GM_M*GM_K,GM_M*GM_K,GM_K,1>>;
    using GlobalDataSrc1 = GlobalTensor<inType,  pto::Shape<1,1,1,K,GM_N>,    pto::Stride<GM_K*GM_N,GM_K*GM_N,GM_K*GM_N,GM_N,1>>;
    using GlobalDataOut  = GlobalTensor<outType, pto::Shape<1,1,1,GM_M,GM_N>, pto::Stride<GM_M*GM_N,GM_M*GM_N,GM_M*GM_N,GM_N,1>>;
    using TileMatA = Tile<TileType::Mat, inType,  GM_M,K,    BLayout::ColMajor, GM_M,K,    SLayout::RowMajor, 512>;
    using TileMatB = Tile<TileType::Mat, inType,  K,   GM_N, BLayout::ColMajor, K,   GM_N, SLayout::RowMajor, 512>;
    using LeftTile = TileLeft< inType,  GM_M,K,    GM_M,K>;
    using RightTile= TileRight<inType,  K,   GM_N, K,   GM_N>;
    using AccTile  = TileAcc<  outType, GM_M,GM_N, GM_M,GM_N>;

    GlobalDataOut dstGlobal(out);

    TileMatA aM0, aM1;
    TASSIGN(aM0, 0x0);     TASSIGN(aM1, 0x1000);
    TileMatB bM0, bM1;
    TASSIGN(bM0, 0x10000); TASSIGN(bM1, 0x14000);
    LeftTile aL0, aL1;
    TASSIGN(aL0, 0x0);     TASSIGN(aL1, 0x800);
    RightTile bL0, bL1;
    TASSIGN(bL0, 0x0);     TASSIGN(bL1, 0x4000);
    AccTile cTile;
    TASSIGN(cTile, 0x0);

    for (uint32_t k = 0; k < K_ITERS; k++) {
        int s = k % NUM_BUFS;
        int id_l0 = s + L0_OFF;
        GlobalDataSrc0 src0Global(src0 + k * K);
        GlobalDataSrc1 src1Global(src1 + k * K * GM_N);
#ifndef __PTO_AUTO__
        get_buffer<PIPE_MTE2>(s);
#endif
        if (s == 0) { TLOAD(aM0, src0Global); TLOAD(bM0, src1Global); }
        else        { TLOAD(aM1, src0Global); TLOAD(bM1, src1Global); }
#ifndef __PTO_AUTO__
        rls_buffer<PIPE_MTE2>(s);
        get_buffer<PIPE_MTE1>(s);
        get_buffer<PIPE_MTE1>(id_l0);
#endif
        if (s == 0) { TMOV(aL0, aM0); TMOV(bL0, bM0); }
        else        { TMOV(aL1, aM1); TMOV(bL1, bM1); }
#ifndef __PTO_AUTO__
        rls_buffer<PIPE_MTE1>(s);
        rls_buffer<PIPE_MTE1>(id_l0);
        get_buffer<PIPE_M>(id_l0);
        get_buffer<PIPE_M>(C_BUF_ID);
#endif
        if (k == 0) {
            if (s == 0) TMATMUL(cTile, aL0, bL0); else TMATMUL(cTile, aL1, bL1);
        } else {
            if (s == 0) TMATMUL_ACC(cTile, cTile, aL0, bL0); else TMATMUL_ACC(cTile, cTile, aL1, bL1);
        }
#ifndef __PTO_AUTO__
        rls_buffer<PIPE_M>(id_l0);
        rls_buffer<PIPE_M>(C_BUF_ID);
#endif
    }
#ifndef __PTO_AUTO__
    get_buffer<PIPE_FIX>(C_BUF_ID);
#endif
    TSTORE(dstGlobal, cTile);
#ifndef __PTO_AUTO__
    rls_buffer<PIPE_FIX>(C_BUF_ID);
#endif
    out = dstGlobal.data();
}

// ═══════════════════════════════════════════════════════════════════════════
// Config 5: 4-buffer, K_tile=32, 16KB B tile
//   L1 IDs: 0-3 | L0 IDs: 4-7 | C_BUF_ID: 8
//   L1 A: 4×0x1000, L1 B: 4×0x4000, L0A: 4×0x800, L0B: 4×0x4000
// ═══════════════════════════════════════════════════════════════════════════
template <typename outType, typename inType>
__global__ AICORE void RunCubeMatmul4Buf16K(__gm__ outType *out, __gm__ inType *src0, __gm__ inType *src1)
{
    constexpr int K = 32, K_ITERS = 32, NUM_BUFS = 4, L0_OFF = 4, C_BUF_ID = 8;

    using GlobalDataSrc0 = GlobalTensor<inType,  pto::Shape<1,1,1,GM_M,K>,    pto::Stride<GM_M*GM_K,GM_M*GM_K,GM_M*GM_K,GM_K,1>>;
    using GlobalDataSrc1 = GlobalTensor<inType,  pto::Shape<1,1,1,K,GM_N>,    pto::Stride<GM_K*GM_N,GM_K*GM_N,GM_K*GM_N,GM_N,1>>;
    using GlobalDataOut  = GlobalTensor<outType, pto::Shape<1,1,1,GM_M,GM_N>, pto::Stride<GM_M*GM_N,GM_M*GM_N,GM_M*GM_N,GM_N,1>>;
    using TileMatA = Tile<TileType::Mat, inType,  GM_M,K,    BLayout::ColMajor, GM_M,K,    SLayout::RowMajor, 512>;
    using TileMatB = Tile<TileType::Mat, inType,  K,   GM_N, BLayout::ColMajor, K,   GM_N, SLayout::RowMajor, 512>;
    using LeftTile = TileLeft< inType,  GM_M,K,    GM_M,K>;
    using RightTile= TileRight<inType,  K,   GM_N, K,   GM_N>;
    using AccTile  = TileAcc<  outType, GM_M,GM_N, GM_M,GM_N>;

    GlobalDataOut dstGlobal(out);

    TileMatA aM0,aM1,aM2,aM3;
    TASSIGN(aM0,0x0); TASSIGN(aM1,0x1000); TASSIGN(aM2,0x2000); TASSIGN(aM3,0x3000);
    TileMatB bM0,bM1,bM2,bM3;
    TASSIGN(bM0,0x10000); TASSIGN(bM1,0x14000); TASSIGN(bM2,0x18000); TASSIGN(bM3,0x1C000);
    LeftTile aL0,aL1,aL2,aL3;
    TASSIGN(aL0,0x0); TASSIGN(aL1,0x800); TASSIGN(aL2,0x1000); TASSIGN(aL3,0x1800);
    RightTile bL0,bL1,bL2,bL3;
    TASSIGN(bL0,0x0); TASSIGN(bL1,0x4000); TASSIGN(bL2,0x8000); TASSIGN(bL3,0xC000);
    AccTile cTile;
    TASSIGN(cTile, 0x0);

    for (uint32_t k = 0; k < K_ITERS; k++) {
        int s = k % NUM_BUFS;
        int id_l0 = s + L0_OFF;
        GlobalDataSrc0 src0Global(src0 + k * K);
        GlobalDataSrc1 src1Global(src1 + k * K * GM_N);
#ifndef __PTO_AUTO__
        get_buffer<PIPE_MTE2>(s);
#endif
        if      (s==0){TLOAD(aM0,src0Global);TLOAD(bM0,src1Global);}
        else if (s==1){TLOAD(aM1,src0Global);TLOAD(bM1,src1Global);}
        else if (s==2){TLOAD(aM2,src0Global);TLOAD(bM2,src1Global);}
        else          {TLOAD(aM3,src0Global);TLOAD(bM3,src1Global);}
#ifndef __PTO_AUTO__
        rls_buffer<PIPE_MTE2>(s);
        get_buffer<PIPE_MTE1>(s);
        get_buffer<PIPE_MTE1>(id_l0);
#endif
        if      (s==0){TMOV(aL0,aM0);TMOV(bL0,bM0);}
        else if (s==1){TMOV(aL1,aM1);TMOV(bL1,bM1);}
        else if (s==2){TMOV(aL2,aM2);TMOV(bL2,bM2);}
        else          {TMOV(aL3,aM3);TMOV(bL3,bM3);}
#ifndef __PTO_AUTO__
        rls_buffer<PIPE_MTE1>(s);
        rls_buffer<PIPE_MTE1>(id_l0);
        get_buffer<PIPE_M>(id_l0);
        get_buffer<PIPE_M>(C_BUF_ID);
#endif
        if (k == 0) {
            if      (s==0) TMATMUL(cTile,aL0,bL0);
            else if (s==1) TMATMUL(cTile,aL1,bL1);
            else if (s==2) TMATMUL(cTile,aL2,bL2);
            else           TMATMUL(cTile,aL3,bL3);
        } else {
            if      (s==0) TMATMUL_ACC(cTile,cTile,aL0,bL0);
            else if (s==1) TMATMUL_ACC(cTile,cTile,aL1,bL1);
            else if (s==2) TMATMUL_ACC(cTile,cTile,aL2,bL2);
            else           TMATMUL_ACC(cTile,cTile,aL3,bL3);
        }
#ifndef __PTO_AUTO__
        rls_buffer<PIPE_M>(id_l0);
        rls_buffer<PIPE_M>(C_BUF_ID);
#endif
    }
#ifndef __PTO_AUTO__
    get_buffer<PIPE_FIX>(C_BUF_ID);
#endif
    TSTORE(dstGlobal, cTile);
#ifndef __PTO_AUTO__
    rls_buffer<PIPE_FIX>(C_BUF_ID);
#endif
    out = dstGlobal.data();
}

// ─────────────────────────────────────────────────────────────────────────────
// Launchers
// ─────────────────────────────────────────────────────────────────────────────
void LaunchCubeMatmul2Buf8K (uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream) {
    RunCubeMatmul2Buf8K<float,half><<<1,nullptr,stream>>>(
        reinterpret_cast<float*>(out), reinterpret_cast<half*>(src0), reinterpret_cast<half*>(src1));
}
void LaunchCubeMatmul4Buf8K (uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream) {
    RunCubeMatmul4Buf8K<float,half><<<1,nullptr,stream>>>(
        reinterpret_cast<float*>(out), reinterpret_cast<half*>(src0), reinterpret_cast<half*>(src1));
}
void LaunchCubeMatmul8Buf8K (uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream) {
    RunCubeMatmul8Buf8K<float,half><<<1,nullptr,stream>>>(
        reinterpret_cast<float*>(out), reinterpret_cast<half*>(src0), reinterpret_cast<half*>(src1));
}
void LaunchCubeMatmul2Buf16K(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream) {
    RunCubeMatmul2Buf16K<float,half><<<1,nullptr,stream>>>(
        reinterpret_cast<float*>(out), reinterpret_cast<half*>(src0), reinterpret_cast<half*>(src1));
}
void LaunchCubeMatmul4Buf16K(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream) {
    RunCubeMatmul4Buf16K<float,half><<<1,nullptr,stream>>>(
        reinterpret_cast<float*>(out), reinterpret_cast<half*>(src0), reinterpret_cast<half*>(src1));
}
