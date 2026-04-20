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
 * @file cube_matmul_4buf_kernel.cpp
 * @brief Cube matmul kernel with 4-buffer B-tile preload 
 * 
 * Based on tmatmul SPLIT_K reference pattern.
 * Input layout: K_ITERS tiles stored contiguously
 *   A: K_ITERS tiles of (M, K) = (32, 16), each at offset i * M * K
 *   B: K_ITERS tiles of (K, N) = (16, 256), each at offset i * K * N
 * Output: C = sum_{i=0..K_ITERS-1}(A[i] @ B[i])
 */

#include <pto/pto-inst.hpp>
#include <pto/common/constants.hpp>

using namespace pto;

// Matrix dimensions
constexpr int M = 32;
constexpr int K = 16;      // K tile size
constexpr int N = 256;
constexpr int K_ITERS = 64;  // Total iterations

/**
 * K-split matmul with 4-buffer software pipelining
 * Input: A as K_ITERS tiles of [M, K]
 *        B as K_ITERS tiles of [K, N]
 * Output: C[M, N]
 */
template <typename outType, typename inType>
__global__ AICORE void RunCubeMatmul4Buf(__gm__ outType *out, __gm__ inType *src0, __gm__ inType *src1)
{
    // Global tensor types for K-tiles (contiguous storage)
    using GlobalDataSrc0 = GlobalTensor<inType, Shape<1, 1, 1, M, K>, Stride<M * K, M * K, M * K, K, 1>>;
    using GlobalDataSrc1 = GlobalTensor<inType, Shape<1, 1, 1, K, N>, Stride<K * N, K * N, K * N, N, 1>>;
    using GlobalDataOut = GlobalTensor<outType, Shape<1, 1, 1, M, N>, Stride<M * N, M * N, M * N, N, 1>>;

    GlobalDataOut dstGlobal(out);

    // L1 tile types (BLayout::ColMajor for NZ fractal format)
    using TileMatAData = Tile<TileType::Mat, inType, M, K, BLayout::ColMajor, M, K, SLayout::RowMajor, 512>;
    using TileMatBData = Tile<TileType::Mat, inType, K, N, BLayout::ColMajor, K, N, SLayout::RowMajor, 512>;

    // L0 tile types
    using LeftTile = TileLeft<inType, M, K, M, K>;
    using RightTile = TileRight<inType, K, N, K, N>;
    using AccTile = TileAcc<outType, M, N, M, N>;

    // L1 buffers - 4 for B tiles (software pipelining)
    TileMatAData aMatTile;
    TileMatBData bMatTile0, bMatTile1, bMatTile2, bMatTile3;
    TASSIGN(aMatTile, 0x0);
    TASSIGN(bMatTile0, 0x10000);
    TASSIGN(bMatTile1, 0x12000);  // +8KB
    TASSIGN(bMatTile2, 0x14000);  // +8KB
    TASSIGN(bMatTile3, 0x16000);  // +8KB

    // L0 tiles
    LeftTile aTile;
    RightTile bTile0, bTile1, bTile2, bTile3;
    AccTile cTile;
    TASSIGN(aTile, 0x0);
    TASSIGN(bTile0, 0x0);
    TASSIGN(bTile1, 0x2000);
    TASSIGN(bTile2, 0x4000);
    TASSIGN(bTile3, 0x6000);
    TASSIGN(cTile, 0x0);

    // K-loop with 4-buffer software pipelining
    for (uint32_t i = 0; i < K_ITERS; i++) {
        // Create global tensor views for this K tile (SPLIT_K layout)
        GlobalDataSrc0 src0Global(src0 + i * M * K);  // Tile i at offset i*M*K
        GlobalDataSrc1 src1Global(src1 + i * K * N);  // Tile i at offset i*K*N

        int buf_idx = i % 4;

        // Load B tile to L1 (4-buffer rotation)
        if (buf_idx == 0) {
            TLOAD(bMatTile0, src1Global);
        } else if (buf_idx == 1) {
            TLOAD(bMatTile1, src1Global);
        } else if (buf_idx == 2) {
            TLOAD(bMatTile2, src1Global);
        } else {
            TLOAD(bMatTile3, src1Global);
        }

        // Load A tile
        TLOAD(aMatTile, src0Global);

        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

        // Move A to L0A
        TMOV(aTile, aMatTile);

        // Move B to L0B (4-buffer rotation)
        if (buf_idx == 0) {
            TMOV(bTile0, bMatTile0);
        } else if (buf_idx == 1) {
            TMOV(bTile1, bMatTile1);
        } else if (buf_idx == 2) {
            TMOV(bTile2, bMatTile2);
        } else {
            TMOV(bTile3, bMatTile3);
        }

        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);

        // Matmul
        if (i == 0) {
            // First iteration: initialize accumulator
            if (buf_idx == 0) {
                TMATMUL(cTile, aTile, bTile0);
            } else if (buf_idx == 1) {
                TMATMUL(cTile, aTile, bTile1);
            } else if (buf_idx == 2) {
                TMATMUL(cTile, aTile, bTile2);
            } else {
                TMATMUL(cTile, aTile, bTile3);
            }
        } else {
            // Subsequent iterations: accumulate
            if (buf_idx == 0) {
                TMATMUL_ACC(cTile, cTile, aTile, bTile0);
            } else if (buf_idx == 1) {
                TMATMUL_ACC(cTile, cTile, aTile, bTile1);
            } else if (buf_idx == 2) {
                TMATMUL_ACC(cTile, cTile, aTile, bTile2);
            } else {
                TMATMUL_ACC(cTile, cTile, aTile, bTile3);
            }
        }
    }

    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);

    // Store result
    TSTORE(dstGlobal, cTile);
    out = dstGlobal.data();
}

// Launch function
void LaunchCubeMatmul4Buf(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream)
{
    RunCubeMatmul4Buf<float, half>(reinterpret_cast<float *>(out),
                                    reinterpret_cast<half *>(src0),
                                    reinterpret_cast<half *>(src1));
}
