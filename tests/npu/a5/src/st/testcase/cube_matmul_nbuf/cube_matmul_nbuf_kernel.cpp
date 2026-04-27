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

#define BUFFER_ID_OFFSET 32
// Inline-asm get_buf/rls_buf wrappers, one branch per pipe.
// These bypass the intrinsic's compile-time buf-id checks so we can probe
// id values >= 32 (intrinsic enforces a 5-bit id range). The asm form
// `GET_BUF.<pipe> reg, #mode` / `RLS_BUF.<pipe> reg, #mode` accepts a
// 64-bit register operand for buf_ID, giving us the full HW id space.
// Pipe suffix is the lowercase pipe_t name without the PIPE_ prefix:
//   PIPE_MTE2 -> mte2,  PIPE_MTE1 -> mte1,  PIPE_M -> m,  PIPE_FIX -> fix.
template <pipe_t pipe>
AICORE void get_buffer(int id) {
    if constexpr (pipe == PIPE_MTE2) {
        asm volatile("GET_BUF.mte2 %0, #0\n\t" :: "l"(id + BUFFER_ID_OFFSET) :);
    } else if constexpr (pipe == PIPE_MTE1) {
        asm volatile("GET_BUF.mte1 %0, #0\n\t" :: "l"(id + BUFFER_ID_OFFSET) :);
    } else if constexpr (pipe == PIPE_M) {
        asm volatile("GET_BUF.m %0, #0\n\t"    :: "l"(id + BUFFER_ID_OFFSET) :);
    } else if constexpr (pipe == PIPE_FIX) {
        asm volatile("GET_BUF.f %0, #0\n\t" :: "l"(id + BUFFER_ID_OFFSET) :);
    } else {
        get_buf(pipe, id, 0);
    }
    return;
}
template <pipe_t pipe>
AICORE void rls_buffer(int id) {
    if constexpr (pipe == PIPE_MTE2) {
        asm volatile("RLS_BUF.mte2 %0, #0\n\t" :: "l"(id + BUFFER_ID_OFFSET) :);
    } else if constexpr (pipe == PIPE_MTE1) {
        asm volatile("RLS_BUF.mte1 %0, #0\n\t" :: "l"(id + BUFFER_ID_OFFSET) :);
    } else if constexpr (pipe == PIPE_M) {
        asm volatile("RLS_BUF.m %0, #0\n\t"    :: "l"(id + BUFFER_ID_OFFSET) :);
    } else if constexpr (pipe == PIPE_FIX) {
        asm volatile("RLS_BUF.f %0, #0\n\t" :: "l"(id + BUFFER_ID_OFFSET) :);
    } else {
        rls_buf(pipe, id, 0);
    }
    return;
}

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


// ═══════════════════════════════════════════════════════════════════════════
// Config 7: B-tile N-buffering with FIXED LARGE A tile.
//
//   Goal: study how N-buffering of the *B* tile in L1 hides MTE2 latency,
//   while A is held as a single large [M_TILE, A_K_TILE] tile in L1
//   (ping-pong, 2 slots only) and consumed by L1-view sub-tiles of shape
//   [M_TILE, B_K_TILE] at compile-time-fixed K-offsets.
//
//   Outer loop (K_GROUPS = K_TOTAL / A_K_TILE):
//     - 1 big TLOAD of A[outer] into A ping/pong slot s_a = outer % 2
//   Inner loop (INNER = A_K_TILE / B_K_TILE):
//     - TLOAD B[outer*INNER + i] into B slot s_b = i % N_BUFS_B
//     - TMOV  A_view[s_a][i]  -> L0A[s_a]
//     - TMOV  B_slot[s_b]     -> L0B[s_b]
//     - TMATMUL_ACC  cTile += L0A[s_a] * L0B[s_b]
//
//   Compile-time plan (all constexpr):
//     K_GROUPS       = GM_K / A_K_TILE
//     INNER          = A_K_TILE / B_K_TILE
//     A_BIG_BYTES    = M_TILE * A_K_TILE * 2     (NZ-packed)
//     A_VIEW_STRIDE  = M_TILE * B_K_TILE * 2     (per K-step within a slot)
//     B_SLOT_BYTES   = B_K_TILE * N_TILE * 2
//     L0A_STRIDE     = M_TILE * B_K_TILE * 2
//     L0B_STRIDE     = B_K_TILE * N_TILE * 2
//
//   Buffer-id allocation (flat ID space):
//     A_L1 ping-pong : 0, 1                          (PIPE_MTE2)
//     B_L1 N-buf     : 2 .. 2 + N_BUFS_B - 1          (PIPE_MTE2)
//     L0A ping-pong  : 2 + N_BUFS_B, 3 + N_BUFS_B     (PIPE_MTE1 / PIPE_M)
//     L0B N-buf      : 4 + N_BUFS_B .. 4 + 2*N_BUFS_B - 1
//     C              : 4 + 2*N_BUFS_B
//
//   Memory budget checks (static_assert):
//     A L1 region  : 2 * A_BIG_BYTES    must fit before B_L1_BASE = 0x10000
//     L0B region   : N_BUFS_B * L0B_STRIDE  <= 64 KiB
//     L0A region   : 2 * L0A_STRIDE         <= 64 KiB
// ═══════════════════════════════════════════════════════════════════════════
template <typename outType, typename inType,
          int N_BUFS_B_, int A_K_TILE_, int B_K_TILE_, int M_TILE_, int N_TILE_>
__global__ AICORE void RunCubeMatmulBNBuf(__gm__ outType *out, __gm__ inType *src0, __gm__ inType *src1)
{
    // ── Compile-time problem shape ───────────────────────────────────────
    constexpr int N_BUFS_B  = N_BUFS_B_;
    constexpr int A_K_TILE  = A_K_TILE_;
    constexpr int B_K_TILE  = B_K_TILE_;
    constexpr int M_TILE    = M_TILE_;
    constexpr int N_TILE    = N_TILE_;
    constexpr int K_GROUPS  = GM_K / A_K_TILE;
    constexpr int INNER     = A_K_TILE / B_K_TILE;
    static_assert(K_GROUPS * A_K_TILE == GM_K, "GM_K must be a multiple of A_K_TILE");
    static_assert(INNER * B_K_TILE == A_K_TILE, "A_K_TILE must be a multiple of B_K_TILE");
    static_assert(B_K_TILE % 16 == 0, "B_K_TILE must be a multiple of 16 (NZ K0 group)");

    // ── Address plan (bytes) ─────────────────────────────────────────────
    constexpr int A_BIG_BYTES   = M_TILE * A_K_TILE * 2;       // [M, A_K] fp16 NZ
    constexpr int A_VIEW_STRIDE = M_TILE * B_K_TILE * 2;       // K1-stride within a slot
    constexpr int A_L1_BASE     = 0x00000;
    constexpr int B_L1_BASE     = 0x10000;
    constexpr int B_SLOT_BYTES  = B_K_TILE * N_TILE * 2;
    constexpr int L0A_STRIDE    = M_TILE * B_K_TILE * 2;
    constexpr int L0B_STRIDE    = B_K_TILE * N_TILE * 2;
    static_assert(A_L1_BASE + 2 * A_BIG_BYTES <= B_L1_BASE,    "A L1 region overlaps B L1 region");
    static_assert(N_BUFS_B * B_SLOT_BYTES <= 0x40000,          "B L1 region exceeds 256 KiB budget");
    static_assert(2 * L0A_STRIDE  <= 0x10000, "L0A overflow (>64 KiB)");
    static_assert(N_BUFS_B * L0B_STRIDE <= 0x10000, "L0B overflow (>64 KiB)");

    // ── Buffer-id allocation ─────────────────────────────────────────────
    constexpr int A_ID_BASE   = 0;
    constexpr int B_ID_BASE   = 2;
    constexpr int L0A_ID_BASE = 2 + N_BUFS_B;
    constexpr int L0B_ID_BASE = 4 + N_BUFS_B;
    constexpr int C_BUF_ID    = 4 + 2 * N_BUFS_B;

    using GlobalDataSrc0 = GlobalTensor<inType,  pto::Shape<1,1,1,M_TILE,A_K_TILE>, pto::Stride<GM_M*GM_K,GM_M*GM_K,GM_M*GM_K,GM_K,1>>;
    using GlobalDataSrc1 = GlobalTensor<inType,  pto::Shape<1,1,1,B_K_TILE,N_TILE>, pto::Stride<GM_K*GM_N,GM_K*GM_N,GM_K*GM_N,GM_N,1>>;
    using GlobalDataOut  = GlobalTensor<outType, pto::Shape<1,1,1,GM_M,GM_N>,       pto::Stride<GM_M*GM_N,GM_M*GM_N,GM_M*GM_N,GM_N,1>>;
    // Big A tile in L1 (one full [M, A_K_TILE]) — used only as TLOAD destination.
    using TileMatABig  = Tile<TileType::Mat, inType, M_TILE, A_K_TILE, BLayout::ColMajor, M_TILE, A_K_TILE, SLayout::RowMajor, 512>;
    // L1 sub-view of A, [M, B_K_TILE] — same NZ format, just a smaller K window.
    using TileMatAView = Tile<TileType::Mat, inType, M_TILE, B_K_TILE, BLayout::ColMajor, M_TILE, B_K_TILE, SLayout::RowMajor, 512>;
    using TileMatB     = Tile<TileType::Mat, inType, B_K_TILE, N_TILE, BLayout::ColMajor, B_K_TILE, N_TILE, SLayout::RowMajor, 512>;
    using LeftTile     = TileLeft< inType,  M_TILE, B_K_TILE, M_TILE, B_K_TILE>;
    using RightTile    = TileRight<inType,  B_K_TILE, N_TILE, B_K_TILE, N_TILE>;
    using AccTile      = TileAcc<  outType, GM_M, GM_N, GM_M, GM_N>;

    GlobalDataOut dstGlobal(out);

    TileMatABig  aBig[2];                  // 2 ping-pong slots, each [M, A_K_TILE]
    TileMatAView aView[2][INNER];          // pre-positioned L1 views for each (slot, inner)
    TileMatB     bM[N_BUFS_B];             // N B slots in L1
    LeftTile     aL[2];                    // L0A ping-pong
    RightTile    bL[N_BUFS_B];             // L0B N-buf
    AccTile      cTile;

    // L1 A: 2 big slots, one TASSIGN per slot
    TASSIGN(aBig[0], A_L1_BASE);
    TASSIGN(aBig[1], A_L1_BASE + A_BIG_BYTES);
    // L1 A views: 2 * INNER static views at compile-time-fixed offsets
    for (int p = 0; p < 2; p++) {
        for (int i = 0; i < INNER; i++) {
            TASSIGN(aView[p][i], A_L1_BASE + p * A_BIG_BYTES + i * A_VIEW_STRIDE);
        }
    }
    // L1 B: N_BUFS_B slots
    for (int i = 0; i < N_BUFS_B; i++) {
        TASSIGN(bM[i], B_L1_BASE + i * B_SLOT_BYTES);
    }
    // L0A ping-pong
    TASSIGN(aL[0], 0x0);
    TASSIGN(aL[1], L0A_STRIDE);
    // L0B N-buf
    for (int i = 0; i < N_BUFS_B; i++) {
        TASSIGN(bL[i], i * L0B_STRIDE);
    }
    TASSIGN(cTile, 0x0);

    // ── Outer loop: ping-pong A in L1 ────────────────────────────────────
    for (uint32_t outer = 0; outer < K_GROUPS; outer++) {
        int s_a    = outer % 2;
        int a_id   = A_ID_BASE + s_a;
        int la_id  = L0A_ID_BASE + s_a;

        // Big A TLOAD into ping-pong slot
#ifndef __PTO_AUTO__
        get_buffer<PIPE_MTE2>(a_id);
#endif
        {
            GlobalDataSrc0 g(src0 + outer * A_K_TILE);
            TLOAD(aBig[s_a], g);
        }
#ifndef __PTO_AUTO__
        rls_buffer<PIPE_MTE2>(a_id);
#endif

        // ── Inner loop: B N-buf + per-K-step view TMOV + matmul ──────────
        for (int i = 0; i < INNER; i++) {
            int s_b   = i % N_BUFS_B;
            int b_id  = B_ID_BASE + s_b;
            int lb_id = L0B_ID_BASE + s_b;
            uint32_t k_global = outer * INNER + i;

            // B TLOAD into N-buf slot
            GlobalDataSrc1 src1Global(src1 + k_global * B_K_TILE * GM_N);
#ifndef __PTO_AUTO__
            get_buffer<PIPE_MTE2>(b_id);
#endif
            TLOAD(bM[s_b], src1Global);
#ifndef __PTO_AUTO__
            rls_buffer<PIPE_MTE2>(b_id);

            // L1->L0: A view (must wait for big-A MTE2 a_id) + B slot
            get_buffer<PIPE_MTE1>(a_id);
            get_buffer<PIPE_MTE1>(b_id);
            get_buffer<PIPE_MTE1>(la_id);
            get_buffer<PIPE_MTE1>(lb_id);
#endif
            TMOV(aL[s_a], aView[s_a][i]);
            TMOV(bL[s_b], bM[s_b]);
#ifndef __PTO_AUTO__
            rls_buffer<PIPE_MTE1>(a_id);
            rls_buffer<PIPE_MTE1>(b_id);
            rls_buffer<PIPE_MTE1>(la_id);
            rls_buffer<PIPE_MTE1>(lb_id);

            // Cube
            get_buffer<PIPE_M>(la_id);
            get_buffer<PIPE_M>(lb_id);
            get_buffer<PIPE_M>(C_BUF_ID);
#endif
            if (k_global == 0) {
                TMATMUL(cTile, aL[s_a], bL[s_b]);
            } else {
                TMATMUL_ACC(cTile, cTile, aL[s_a], bL[s_b]);
            }
#ifndef __PTO_AUTO__
            rls_buffer<PIPE_M>(la_id);
            rls_buffer<PIPE_M>(lb_id);
            rls_buffer<PIPE_M>(C_BUF_ID);
#endif
        }
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
// Config 6: A-burst preload, templated.
//
//   Outer loop runs K_GROUPS = K_ITERS / N_BUFS_A iterations. Each outer iter
//   pre-loads N_BUFS_A A tiles (one [M_TILE, K_TILE] per K-block) into
//   N_BUFS_A dedicated L1 A slots before issuing the inner sequence of
//   B-load / TMOV / TMATMUL_ACC steps. Expresses "single large A TLOAD"
//   as a burst of N_BUFS_A separately-synchronised TLOADs.
//
//   ── Compile-time plan (all values below are constexpr) ────────────────
//   Loop counts:
//     K_ITERS  = GM_K / K_TILE
//     K_GROUPS = K_ITERS / N_BUFS_A   (outer)
//     INNER    = N_BUFS_A             (inner)
//   Buffer-id allocation:
//     A_ID_BASE  = 0                   (MTE2, one id per A slot 0..N_BUFS_A-1)
//     B_ID_BASE  = 0                   (reuses A ids — A burst rls'd first)
//     L0_ID_BASE = N_BUFS_A            (8..2*N_BUFS_A-1)
//     C_BUF_ID   = 2 * N_BUFS_A        (16 for N_BUFS_A=8)
//   L1/L0 address plan (NZ-aligned, K_TILE=16/fp16 baseline):
//     A_L1_BASE/STRIDE = 0x00000 / 0x0800
//     B_L1_BASE/STRIDE = 0x10000 / 0x2000
//     A_L0_BASE/STRIDE = 0x00000 / 0x0400
//     B_L0_BASE/STRIDE = 0x00000 / 0x2000
// ═══════════════════════════════════════════════════════════════════════════

template <typename outType, typename inType,
          int N_BUFS_A_, int M_TILE_, int K_TILE_, int N_TILE_>
__global__ AICORE void RunCubeMatmulBurstA(__gm__ outType *out, __gm__ inType *src0, __gm__ inType *src1)
{
    // ── Compile-time problem shape ───────────────────────────────────────
    constexpr int N_BUFS_A = N_BUFS_A_;
    constexpr int M_TILE   = M_TILE_;
    constexpr int K_TILE   = K_TILE_;
    constexpr int N_TILE   = N_TILE_;
    constexpr int K_ITERS  = GM_K / K_TILE;
    constexpr int K_GROUPS = K_ITERS / N_BUFS_A;
    constexpr int INNER    = N_BUFS_A;
    static_assert(K_GROUPS * N_BUFS_A == K_ITERS, "K_ITERS must be multiple of N_BUFS_A");

    // ── Buffer-id allocation plan ────────────────────────────────────────
    constexpr int A_ID_BASE  = 0;
    constexpr int B_ID_BASE  = 0;                // reuses A ids after A burst
    constexpr int L0_ID_BASE = N_BUFS_A;
    constexpr int C_BUF_ID   = 2 * N_BUFS_A;

    // ── L1/L0 address plan (NZ-aligned, fp16; strides scale with K_TILE) ─
    constexpr int K_SCALE     = K_TILE / 16;     // baseline = K_TILE 16
    constexpr int A_L1_BASE   = 0x00000;
    constexpr int A_L1_STRIDE = 0x00800 * K_SCALE;
    constexpr int B_L1_BASE   = 0x10000;
    constexpr int B_L1_STRIDE = 0x02000 * K_SCALE;
    constexpr int A_L0_BASE   = 0x00000;
    constexpr int A_L0_STRIDE = 0x00400 * K_SCALE;
    constexpr int B_L0_BASE   = 0x00000;
    constexpr int B_L0_STRIDE = 0x02000 * K_SCALE;
    static_assert(K_TILE % 16 == 0, "K_TILE must be a multiple of 16");
    static_assert(A_L1_BASE + N_BUFS_A * A_L1_STRIDE <= B_L1_BASE,
                  "A L1 region overlaps B L1 region");

    using GlobalDataSrc0 = GlobalTensor<inType,  pto::Shape<1,1,1,M_TILE,K_TILE>, pto::Stride<GM_M*GM_K,GM_M*GM_K,GM_M*GM_K,GM_K,1>>;
    using GlobalDataSrc1 = GlobalTensor<inType,  pto::Shape<1,1,1,K_TILE,N_TILE>, pto::Stride<GM_K*GM_N,GM_K*GM_N,GM_K*GM_N,GM_N,1>>;
    using GlobalDataOut  = GlobalTensor<outType, pto::Shape<1,1,1,GM_M,GM_N>,     pto::Stride<GM_M*GM_N,GM_M*GM_N,GM_M*GM_N,GM_N,1>>;
    using TileMatA = Tile<TileType::Mat, inType, M_TILE,K_TILE, BLayout::ColMajor, M_TILE,K_TILE, SLayout::RowMajor, 512>;
    using TileMatB = Tile<TileType::Mat, inType, K_TILE,N_TILE, BLayout::ColMajor, K_TILE,N_TILE, SLayout::RowMajor, 512>;
    using LeftTile = TileLeft< inType,  M_TILE,K_TILE, M_TILE,K_TILE>;
    using RightTile= TileRight<inType,  K_TILE,N_TILE, K_TILE,N_TILE>;
    using AccTile  = TileAcc<  outType, GM_M,GM_N,     GM_M,GM_N>;

    GlobalDataOut dstGlobal(out);

    TileMatA  aM[N_BUFS_A];
    TileMatB  bM[N_BUFS_A];
    LeftTile  aL[N_BUFS_A];
    RightTile bL[N_BUFS_A];
    AccTile   cTile;

    // L1/L0 tile addresses derived purely from the constexpr address plan;
    // i is a runtime loop var but the compiler can hoist all arithmetic.
    for (int i = 0; i < N_BUFS_A; i++) {
        TASSIGN(aM[i], A_L1_BASE + i * A_L1_STRIDE);
        TASSIGN(bM[i], B_L1_BASE + i * B_L1_STRIDE);
        TASSIGN(aL[i], A_L0_BASE + i * A_L0_STRIDE);
        TASSIGN(bL[i], B_L0_BASE + i * B_L0_STRIDE);
    }
    TASSIGN(cTile, 0x0);

    for (uint32_t outer = 0; outer < K_GROUPS; outer++) {

        // ── Burst-preload N_BUFS_A A tiles, one MTE2 buf id per slot ─────
        for (int i = 0; i < N_BUFS_A; i++) {
            int id = A_ID_BASE + i;
#ifndef __PTO_AUTO__
            get_buffer<PIPE_MTE2>(id);
#endif
            GlobalDataSrc0 g(src0 + (outer * N_BUFS_A + i) * K_TILE);
            TLOAD(aM[i], g);
#ifndef __PTO_AUTO__
            rls_buffer<PIPE_MTE2>(id);
#endif
        }

        // ── Inner loop: per-slot B load + TMOV + TMATMUL_ACC ─────────────
        for (int s = 0; s < INNER; s++) {
            int id    = B_ID_BASE + s;
            int id_l0 = L0_ID_BASE + s;
            uint32_t k = outer * INNER + s;
            GlobalDataSrc1 src1Global(src1 + k * K_TILE * GM_N);
#ifndef __PTO_AUTO__
            get_buffer<PIPE_MTE2>(id);
#endif
            TLOAD(bM[s], src1Global);
#ifndef __PTO_AUTO__
            rls_buffer<PIPE_MTE2>(id);
            get_buffer<PIPE_MTE1>(id);
            get_buffer<PIPE_MTE1>(id_l0);
#endif
            TMOV(aL[s], aM[s]);
            TMOV(bL[s], bM[s]);
#ifndef __PTO_AUTO__
            rls_buffer<PIPE_MTE1>(id);
            rls_buffer<PIPE_MTE1>(id_l0);
            get_buffer<PIPE_M>(id_l0);
            get_buffer<PIPE_M>(C_BUF_ID);
#endif
            if (k == 0) {
                TMATMUL(cTile, aL[s], bL[s]);
            } else {
                TMATMUL_ACC(cTile, cTile, aL[s], bL[s]);
            }
#ifndef __PTO_AUTO__
            rls_buffer<PIPE_M>(id_l0);
            rls_buffer<PIPE_M>(C_BUF_ID);
#endif
        }
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


void LaunchCubeMatmul4BufALarge(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream) {
    // 4 A buffers × K_TILE=32 → K_A_GROUP=128, 8 outer × 4 inner over K=1024.
    // L0B: 4×0x4000 = 0x10000 (= 64KiB, exact fit).
    RunCubeMatmulBurstA<float, half, /*N_BUFS_A=*/4, /*M_TILE=*/GM_M, /*K_TILE=*/32, /*N_TILE=*/GM_N>
        <<<1, nullptr, stream>>>(
            reinterpret_cast<float*>(out),
            reinterpret_cast<half*>(src0),
            reinterpret_cast<half*>(src1));
}

void LaunchCubeMatmul8BufALarge(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream) {
    // 8 A buffers × K_TILE=16 → K_A_GROUP=128, 8 outer × 8 inner over K=1024.
    // L0B: 8×0x2000 = 0x10000 (= 64KiB, exact fit).
    RunCubeMatmulBurstA<float, half, /*N_BUFS_A=*/8, /*M_TILE=*/GM_M, /*K_TILE=*/16, /*N_TILE=*/GM_N>
        <<<1, nullptr, stream>>>(
            reinterpret_cast<float*>(out),
            reinterpret_cast<half*>(src0),
            reinterpret_cast<half*>(src1));
}

void LaunchCubeMatmul4BufALargeK64(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream) {
    // 4 A buffers × K_TILE=16 → K_A_GROUP=64, 16 outer × 4 inner over K=1024.
    // Same B-tile [16,256] as LaunchCubeMatmul8BufALarge — fair N_BUFS_A comparison.
    RunCubeMatmulBurstA<float, half, /*N_BUFS_A=*/4, /*M_TILE=*/GM_M, /*K_TILE=*/16, /*N_TILE=*/GM_N>
        <<<1, nullptr, stream>>>(
            reinterpret_cast<float*>(out),
            reinterpret_cast<half*>(src0),
            reinterpret_cast<half*>(src1));
}


// ── B-tile N-buffering launchers (Config 7: RunCubeMatmulBNBuf) ─────────────
// All variants: A fixed at [32, 128] big tile (ping-pong, 2 L1 slots);
// B varies in N-buffer count and K_TILE size. INNER = 128 / B_K_TILE.

void LaunchCubeMatmulBNBuf2_K16(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream) {
    // B: 2 buf × [16,256] = 16 KiB L1, 16 KiB L0B. INNER=8.
    RunCubeMatmulBNBuf<float, half, /*N_BUFS_B=*/2, /*A_K=*/128, /*B_K=*/16, /*M=*/GM_M, /*N=*/GM_N>
        <<<1, nullptr, stream>>>(
            reinterpret_cast<float*>(out),
            reinterpret_cast<half*>(src0),
            reinterpret_cast<half*>(src1));
}

void LaunchCubeMatmulBNBuf4_K16(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream) {
    // B: 4 buf × [16,256] = 32 KiB L1, 32 KiB L0B. INNER=8.
    RunCubeMatmulBNBuf<float, half, /*N_BUFS_B=*/4, /*A_K=*/128, /*B_K=*/16, /*M=*/GM_M, /*N=*/GM_N>
        <<<1, nullptr, stream>>>(
            reinterpret_cast<float*>(out),
            reinterpret_cast<half*>(src0),
            reinterpret_cast<half*>(src1));
}

void LaunchCubeMatmulBNBuf8_K16(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream) {
    // B: 8 buf × [16,256] = 64 KiB L1, 64 KiB L0B (exact fit). INNER=8.
    RunCubeMatmulBNBuf<float, half, /*N_BUFS_B=*/8, /*A_K=*/128, /*B_K=*/16, /*M=*/GM_M, /*N=*/GM_N>
        <<<1, nullptr, stream>>>(
            reinterpret_cast<float*>(out),
            reinterpret_cast<half*>(src0),
            reinterpret_cast<half*>(src1));
}

void LaunchCubeMatmulBNBuf2_K32(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream) {
    // B: 2 buf × [32,256] = 32 KiB L1, 32 KiB L0B. INNER=4.
    RunCubeMatmulBNBuf<float, half, /*N_BUFS_B=*/2, /*A_K=*/128, /*B_K=*/32, /*M=*/GM_M, /*N=*/GM_N>
        <<<1, nullptr, stream>>>(
            reinterpret_cast<float*>(out),
            reinterpret_cast<half*>(src0),
            reinterpret_cast<half*>(src1));
}

void LaunchCubeMatmulBNBuf4_K32(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream) {
    // B: 4 buf × [32,256] = 64 KiB L1, 64 KiB L0B (exact fit). INNER=4.
    RunCubeMatmulBNBuf<float, half, /*N_BUFS_B=*/4, /*A_K=*/128, /*B_K=*/32, /*M=*/GM_M, /*N=*/GM_N>
        <<<1, nullptr, stream>>>(
            reinterpret_cast<float*>(out),
            reinterpret_cast<half*>(src0),
            reinterpret_cast<half*>(src1));
}


// ═══════════════════════════════════════════════════════════════════════════
// Config 8: B-tile 4KB exploration.  Half-N split with 2 L0C accumulators.
//
//   Same MTE2 bandwidth from GM as Config 7 (total bytes loaded unchanged):
//   we just split each B [B_K, 256] tile into 2 halves [B_K, 128] = 4 KiB
//   (at B_K=16, fp16) and accumulate into TWO independent L0C AccTiles
//   cTile[0]=out[:,0:128], cTile[1]=out[:,128:256].
//
//   Why: 4 KiB B-tile gives finer N-buffer slots; two independent L0C
//   accumulators decouple the matmul WAW chain across N halves so the
//   cube can interleave them via 2 distinct C buf-ids.
//
//   Outer loop (K_GROUPS = GM_K / A_K_TILE):
//     - 1 big TLOAD A[outer]   -> L1 ping-pong slot s_a = outer % 2
//   Inner K-step loop (INNER_K = A_K_TILE / B_K_TILE):
//     - TMOV A_view[s_a][i]    -> L0A[i % 2]   (loaded once, used by both halves)
//     For h = 0..1:
//       - TLOAD B_half[s_b]    -> L1 N-buf slot
//       - TMOV  bM[s_b]        -> L0B[s_b]
//       - TMATMUL(_ACC) cTile[h] += L0A[i%2] * L0B[s_b]
//
//   Compile-time plan (constexpr):
//     N_HALVES       = 2
//     N_HALF         = N_TILE / N_HALVES                (= 128 for N_TILE=256)
//     INNER_K        = A_K_TILE / B_K_TILE              (= 8 at A_K=128, B_K=16)
//     INNER_FLAT     = INNER_K * N_HALVES               (= 16)
//     A_BIG_BYTES    = M_TILE * A_K_TILE * 2            (NZ-packed fp16)
//     A_VIEW_STRIDE  = M_TILE * B_K_TILE * 2
//     B_SLOT_BYTES   = B_K_TILE * N_HALF * 2            (= 4096 for B_K=16)
//     L0A_STRIDE     = M_TILE * B_K_TILE * 2
//     L0B_STRIDE     = B_K_TILE * N_HALF * 2
//     L0C_STRIDE     = M_TILE * N_HALF * sizeof(outType)(= 16384 for fp32)
//
//   Buffer-id allocation (flat ID space):
//     A L1 ping-pong : 0, 1                             (PIPE_MTE2)
//     B L1 N-buf     : 2 .. 2 + N_BUFS_B - 1            (PIPE_MTE2)
//     L0A ping-pong  : 2 + N_BUFS_B,  3 + N_BUFS_B
//     L0B N-buf      : 4 + N_BUFS_B .. 4 + 2*N_BUFS_B - 1
//     L0C            : 4 + 2*N_BUFS_B,  5 + 2*N_BUFS_B  (one id per N half)
// ═══════════════════════════════════════════════════════════════════════════
template <typename outType, typename inType,
          int N_BUFS_B_, int A_K_TILE_, int B_K_TILE_, int M_TILE_, int N_TILE_>
__global__ AICORE void RunCubeMatmulBNBufNSplit(__gm__ outType *out, __gm__ inType *src0, __gm__ inType *src1)
{
    // ── Compile-time problem shape ───────────────────────────────────────
    constexpr int N_BUFS_B   = N_BUFS_B_;
    constexpr int A_K_TILE   = A_K_TILE_;
    constexpr int B_K_TILE   = B_K_TILE_;
    constexpr int M_TILE     = M_TILE_;
    constexpr int N_TILE     = N_TILE_;
    constexpr int N_HALVES   = 2;
    constexpr int N_HALF     = N_TILE / N_HALVES;
    constexpr int K_GROUPS   = GM_K / A_K_TILE;
    constexpr int INNER_K    = A_K_TILE / B_K_TILE;
    static_assert(K_GROUPS * A_K_TILE == GM_K, "GM_K must be a multiple of A_K_TILE");
    static_assert(INNER_K * B_K_TILE == A_K_TILE, "A_K_TILE must be a multiple of B_K_TILE");
    static_assert(N_HALVES * N_HALF == N_TILE, "N_TILE must split evenly into 2 halves");
    static_assert(B_K_TILE % 16 == 0, "B_K_TILE must be a multiple of 16 (NZ K0 group)");
    static_assert(N_HALF % 16 == 0,  "N_HALF must be a multiple of 16 (NZ N0 group)");

    // ── Address plan (bytes) ─────────────────────────────────────────────
    constexpr int A_BIG_BYTES   = M_TILE * A_K_TILE * 2;
    constexpr int A_VIEW_STRIDE = M_TILE * B_K_TILE * 2;
    constexpr int A_L1_BASE     = 0x00000;
    constexpr int B_L1_BASE     = 0x10000;
    constexpr int B_SLOT_BYTES  = B_K_TILE * N_HALF * 2;
    constexpr int L0A_STRIDE    = M_TILE * B_K_TILE * 2;
    constexpr int L0B_STRIDE    = B_K_TILE * N_HALF * 2;
    constexpr int L0C_STRIDE    = M_TILE * N_HALF * (int)sizeof(outType);
    static_assert(A_L1_BASE + 2 * A_BIG_BYTES <= B_L1_BASE,
                  "A L1 region overlaps B L1 region");
    static_assert(N_BUFS_B * B_SLOT_BYTES <= 0x40000,
                  "B L1 region exceeds 256 KiB budget");
    static_assert(2 * L0A_STRIDE  <= 0x10000, "L0A overflow (>64 KiB)");
    static_assert(N_BUFS_B * L0B_STRIDE <= 0x10000, "L0B overflow (>64 KiB)");
    static_assert(N_HALVES * L0C_STRIDE <= 0x40000, "L0C overflow (>256 KiB)");

    // ── Buffer-id allocation ─────────────────────────────────────────────
    constexpr int A_ID_BASE   = 0;
    constexpr int B_ID_BASE   = 2;
    constexpr int L0A_ID_BASE = 2 + N_BUFS_B;
    constexpr int L0B_ID_BASE = 4 + N_BUFS_B;
    constexpr int C_ID_BASE   = 4 + 2 * N_BUFS_B;

    using GlobalDataSrc0   = GlobalTensor<inType,  pto::Shape<1,1,1,M_TILE,A_K_TILE>, pto::Stride<GM_M*GM_K,GM_M*GM_K,GM_M*GM_K,GM_K,1>>;
    using GlobalDataSrc1   = GlobalTensor<inType,  pto::Shape<1,1,1,B_K_TILE,N_HALF>, pto::Stride<GM_K*GM_N,GM_K*GM_N,GM_K*GM_N,GM_N,1>>;
    using GlobalDataOutHalf= GlobalTensor<outType, pto::Shape<1,1,1,GM_M,N_HALF>,     pto::Stride<GM_M*GM_N,GM_M*GM_N,GM_M*GM_N,GM_N,1>>;
    // Big A tile in L1 (one full [M, A_K_TILE]) — TLOAD destination.
    using TileMatABig  = Tile<TileType::Mat, inType, M_TILE, A_K_TILE, BLayout::ColMajor, M_TILE, A_K_TILE, SLayout::RowMajor, 512>;
    // L1 sub-view of A, [M, B_K_TILE].
    using TileMatAView = Tile<TileType::Mat, inType, M_TILE, B_K_TILE, BLayout::ColMajor, M_TILE, B_K_TILE, SLayout::RowMajor, 512>;
    // Half-N B tile [B_K_TILE, N_HALF].
    using TileMatB     = Tile<TileType::Mat, inType, B_K_TILE, N_HALF, BLayout::ColMajor, B_K_TILE, N_HALF, SLayout::RowMajor, 512>;
    using LeftTile     = TileLeft< inType,  M_TILE, B_K_TILE, M_TILE, B_K_TILE>;
    using RightTile    = TileRight<inType,  B_K_TILE, N_HALF,  B_K_TILE, N_HALF>;
    using AccTileHalf  = TileAcc<  outType, M_TILE,  N_HALF,   M_TILE,  N_HALF>;

    GlobalDataOutHalf dstGlobal0(out);
    GlobalDataOutHalf dstGlobal1(out + N_HALF);

    TileMatABig  aBig[2];
    TileMatAView aView[2][INNER_K];
    TileMatB     bM[N_BUFS_B];
    LeftTile     aL[2];
    RightTile    bL[N_BUFS_B];
    AccTileHalf  cTile[N_HALVES];

    // L1 A: 2 big slots
    TASSIGN(aBig[0], A_L1_BASE);
    TASSIGN(aBig[1], A_L1_BASE + A_BIG_BYTES);
    // L1 A views.
    for (int p = 0; p < 2; p++) {
        for (int i = 0; i < INNER_K; i++) {
            TASSIGN(aView[p][i], A_L1_BASE + p * A_BIG_BYTES + i * A_VIEW_STRIDE);
        }
    }
    // L1 B: N_BUFS_B half-N slots
    for (int i = 0; i < N_BUFS_B; i++) {
        TASSIGN(bM[i], B_L1_BASE + i * B_SLOT_BYTES);
    }
    // L0A ping-pong
    TASSIGN(aL[0], 0x0);
    TASSIGN(aL[1], L0A_STRIDE);
    // L0B N-buf
    for (int i = 0; i < N_BUFS_B; i++) {
        TASSIGN(bL[i], i * L0B_STRIDE);
    }
    // L0C: 2 half-N accumulators
    TASSIGN(cTile[0], 0x0);
    TASSIGN(cTile[1], L0C_STRIDE);

    // ── Outer loop: ping-pong A in L1 ────────────────────────────────────
    for (uint32_t outer = 0; outer < K_GROUPS; outer++) {
        int s_a    = outer % 2;
        int a_id   = A_ID_BASE + s_a;

        // Big A TLOAD into ping-pong slot
#ifndef __PTO_AUTO__
        get_buffer<PIPE_MTE2>(a_id);
#endif
        {
            GlobalDataSrc0 g(src0 + outer * A_K_TILE);
            TLOAD(aBig[s_a], g);
        }
#ifndef __PTO_AUTO__
        rls_buffer<PIPE_MTE2>(a_id);
#endif

        // ── Inner K-step loop: load A view once per i, fan out to 2 N halves ──
        for (int i = 0; i < INNER_K; i++) {
            int la_id = L0A_ID_BASE + (i % 2);

            // A view -> L0A (used by both N halves below)
#ifndef __PTO_AUTO__
            get_buffer<PIPE_MTE1>(a_id);     // RAW: wait big-A in L1
            get_buffer<PIPE_MTE1>(la_id);    // WAR: wait cube done with prev L0A slot
#endif
            TMOV(aL[i % 2], aView[s_a][i]);
#ifndef __PTO_AUTO__
            rls_buffer<PIPE_MTE1>(a_id);
            rls_buffer<PIPE_MTE1>(la_id);
#endif

            for (int h = 0; h < N_HALVES; h++) {
                int flat  = i * N_HALVES + h;
                int s_b   = flat % N_BUFS_B;
                int b_id  = B_ID_BASE  + s_b;
                int lb_id = L0B_ID_BASE + s_b;
                int c_id  = C_ID_BASE  + h;
                uint32_t k_global = outer * INNER_K + i;

                // B half-tile TLOAD into N-buf slot
                GlobalDataSrc1 src1Global(src1 + k_global * B_K_TILE * GM_N + h * N_HALF);
#ifndef __PTO_AUTO__
                get_buffer<PIPE_MTE2>(b_id);
#endif
                TLOAD(bM[s_b], src1Global);
#ifndef __PTO_AUTO__
                rls_buffer<PIPE_MTE2>(b_id);

                // L1->L0: B slot
                get_buffer<PIPE_MTE1>(b_id);
                get_buffer<PIPE_MTE1>(lb_id);
#endif
                TMOV(bL[s_b], bM[s_b]);
#ifndef __PTO_AUTO__
                rls_buffer<PIPE_MTE1>(b_id);
                rls_buffer<PIPE_MTE1>(lb_id);

                // Cube: matmul into per-half L0C accumulator
                get_buffer<PIPE_M>(la_id);
                get_buffer<PIPE_M>(lb_id);
                get_buffer<PIPE_M>(c_id);
#endif
                if (k_global == 0) {
                    TMATMUL(cTile[h], aL[i % 2], bL[s_b]);
                } else {
                    TMATMUL_ACC(cTile[h], cTile[h], aL[i % 2], bL[s_b]);
                }
#ifndef __PTO_AUTO__
                rls_buffer<PIPE_M>(la_id);
                rls_buffer<PIPE_M>(lb_id);
                rls_buffer<PIPE_M>(c_id);
#endif
            }
        }
    }

    // ── TSTORE both halves ──────────────────────────────────────────────
#ifndef __PTO_AUTO__
    get_buffer<PIPE_FIX>(C_ID_BASE + 0);
#endif
    TSTORE(dstGlobal0, cTile[0]);
#ifndef __PTO_AUTO__
    rls_buffer<PIPE_FIX>(C_ID_BASE + 0);
    get_buffer<PIPE_FIX>(C_ID_BASE + 1);
#endif
    TSTORE(dstGlobal1, cTile[1]);
#ifndef __PTO_AUTO__
    rls_buffer<PIPE_FIX>(C_ID_BASE + 1);
#endif
    out = dstGlobal0.data();
}


// ── 4 KiB B-tile launchers (Config 8: RunCubeMatmulBNBufNSplit) ─────────────
// All variants: A fixed at [32,128] big tile (ping-pong, 2 L1 slots);
// B is half-N: [B_K, 128] = 4 KiB at B_K=16 fp16. 2 L0C accumulators.

void LaunchCubeMatmulBNBufNSplit2_K16(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream) {
    // B: 2 buf × [16,128] = 8 KiB L1, 8 KiB L0B. INNER_K=8, INNER_FLAT=16.
    RunCubeMatmulBNBufNSplit<float, half, /*N_BUFS_B=*/2, /*A_K=*/128, /*B_K=*/16, /*M=*/GM_M, /*N=*/GM_N>
        <<<1, nullptr, stream>>>(
            reinterpret_cast<float*>(out),
            reinterpret_cast<half*>(src0),
            reinterpret_cast<half*>(src1));
}

void LaunchCubeMatmulBNBufNSplit4_K16(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream) {
    // B: 4 buf × [16,128] = 16 KiB L1, 16 KiB L0B. INNER_K=8, INNER_FLAT=16.
    RunCubeMatmulBNBufNSplit<float, half, /*N_BUFS_B=*/4, /*A_K=*/128, /*B_K=*/16, /*M=*/GM_M, /*N=*/GM_N>
        <<<1, nullptr, stream>>>(
            reinterpret_cast<float*>(out),
            reinterpret_cast<half*>(src0),
            reinterpret_cast<half*>(src1));
}

void LaunchCubeMatmulBNBufNSplit8_K16(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream) {
    // B: 8 buf × [16,128] = 32 KiB L1, 32 KiB L0B. INNER_K=8, INNER_FLAT=16.
    RunCubeMatmulBNBufNSplit<float, half, /*N_BUFS_B=*/8, /*A_K=*/128, /*B_K=*/16, /*M=*/GM_M, /*N=*/GM_N>
        <<<1, nullptr, stream>>>(
            reinterpret_cast<float*>(out),
            reinterpret_cast<half*>(src0),
            reinterpret_cast<half*>(src1));
}

void LaunchCubeMatmulBNBufNSplit16_K16(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream) {
    // B: 16 buf × [16,128] = 64 KiB L1, 64 KiB L0B (exact L0B fit). INNER_K=8, INNER_FLAT=16 (matches buf count).
    RunCubeMatmulBNBufNSplit<float, half, /*N_BUFS_B=*/16, /*A_K=*/128, /*B_K=*/16, /*M=*/GM_M, /*N=*/GM_N>
        <<<1, nullptr, stream>>>(
            reinterpret_cast<float*>(out),
            reinterpret_cast<half*>(src0),
            reinterpret_cast<half*>(src1));
}
