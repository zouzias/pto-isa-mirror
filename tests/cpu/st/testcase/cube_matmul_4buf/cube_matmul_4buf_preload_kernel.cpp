/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
// cube_matmul_4buf_preload_kernel.cpp
// 4-Buffer PRELOAD matmul: overlaps load with compute
// Uses all 8 event IDs for MTE1→MTE2 pipeline pair
//
// Key differences from non-preload:
// 1. Pre-loads first 4 B tiles before loop
// 2. Each iteration: COMPUTE on buf[k%4], PRELOAD into buf[(k+4)%4]
// 3. Requires 8 event IDs: 4 forward (data ready) + 4 reverse (buffer reusable)

#include <pto/pto-inst.hpp>
#include <pto/common/constants.hpp>

using namespace pto;

// Constants
constexpr int M = 32;
constexpr int K_TILE = 16;
constexpr int N = 256;
constexpr int K_ITER = 64;       // Total K = 1024 / 16 = 64 iterations
constexpr int NUM_BUFS = 4;

// Tile types
using TileA_L1 = Tile<TileType::Mat, half, M, K_TILE, BLayout::ColMajor, M, K_TILE, SLayout::RowMajor, 512, PadValue::Null>;
using TileB_L1 = Tile<TileType::Mat, half, K_TILE, N, BLayout::ColMajor, K_TILE, N, SLayout::RowMajor, 512, PadValue::Null>;
using TileA_L0 = Tile<TileType::Left, half, M, K_TILE, BLayout::ColMajor, M, K_TILE, SLayout::RowMajor, 512, PadValue::Null>;
using TileB_L0 = Tile<TileType::Right, half, K_TILE, N, BLayout::RowMajor, K_TILE, N, SLayout::ColMajor, 512, PadValue::Null>;
using TileC_L0 = Tile<TileType::Acc, float, M, N, BLayout::ColMajor, M, N, SLayout::RowMajor, 1024, PadValue::Null>;

// Global tensor types for split-K layout
using ShapeA = pto::Shape<1, 1, 1, M, K_TILE>;
using StrideA = pto::Stride<M * K_TILE, M * K_TILE, M * K_TILE, K_TILE, 1>;
using GlobalA = GlobalTensor<half, ShapeA, StrideA, pto::Layout::ND>;

using ShapeB = pto::Shape<1, 1, 1, K_TILE, N>;
using StrideB = pto::Stride<K_TILE * N, K_TILE * N, K_TILE * N, N, 1>;
using GlobalB = GlobalTensor<half, ShapeB, StrideB, pto::Layout::ND>;

using ShapeC = pto::Shape<1, 1, 1, M, N>;
using StrideC = pto::Stride<M * N, M * N, M * N, N, 1>;
using GlobalC = GlobalTensor<float, ShapeC, StrideC, pto::Layout::ND>;

template <typename T_A, typename T_B, typename T_C>
AICORE void runCubeMatmul4BufPreload(
    __gm__ T_A __in__ *gmA,   // A tiles: [K_ITER][M][K_TILE]
    __gm__ T_B __in__ *gmB,   // B tiles: [K_ITER][K_TILE][N]
    __gm__ T_C __out__ *gmC)  // C: [M][N]
{
#if defined(__DAV_CUBE__)
    // L1 buffers
    TileA_L1 a_l1;
    
    // 4 L1 buffers for B (for preload pipelining)
    TileB_L1 b_l1[NUM_BUFS];
    
    // L0 buffers
    TileA_L0 a_l0;
    TileB_L0 b_l0[NUM_BUFS];
    TileC_L0 c_l0;
    
    // Global tensor views
    GlobalC gC(gmC);
    
    // ========================================
    // PRIMING: Pre-load first 4 B tiles
    // ========================================
    
    // Pre-signal: M→MTE1 (reverse dep - buffers are "free" initially)
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID4);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID5);
    
    // Load B[0..3] into L1 buffers
    for (int i = 0; i < NUM_BUFS; i++) {
        GlobalB gB_i(gmB + i * K_TILE * N);
        TLOAD(b_l1[i], gB_i);
    }
    
    // Pre-signal: MTE1→MTE2 (forward dep - data ready for MTE2)
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID4);
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID5);
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID6);
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID7);
    
    // ========================================
    // MAIN LOOP: Compute on k, preload k+4
    // ========================================
    for (int k = 0; k < K_ITER; k++) {
        int buf_idx = k % NUM_BUFS;
        int preload_k = k + NUM_BUFS;
        bool do_preload = (preload_k < K_ITER);
        
        // --- Load A[k] ---
        GlobalA gA_k(gmA + k * M * K_TILE);
        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
        TLOAD(a_l1, gA_k);
        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
        
        // Wait for buffer to be free (reverse dep from compute)
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID4);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID5);
        
        // Move A to L0A
        pipe_barrier(PIPE_MTE1);
        TMOV(a_l0, a_l1);
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
        
        // --- PRELOAD B[k+4] while computing ---
        // This is the key optimization: overlaps MTE2 load with M compute
        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID4);
        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID6);
        
        if (do_preload) {
            int preload_buf = preload_k % NUM_BUFS;
            GlobalB gB_preload(gmB + preload_k * K_TILE * N);
            
            // The preload goes into the same buffer slot (k+4)%4 == k%4
            // because by now, compute on k is about to start, freeing buf[k%4]
            pipe_barrier(PIPE_MTE2);
            TLOAD(b_l1[preload_buf], gB_preload);
        }
        
        // Signal preload done
        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID2);
        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID3);
        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID4);
        
        // --- Move B[k] to L0B ---
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID2);
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID3);
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID4);
        pipe_barrier(PIPE_MTE1);
        TMOV(b_l0[buf_idx], b_l1[buf_idx]);
        
        // Signal data ready for compute
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID4);
        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID6);
        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID2);
        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID3);
        
        // --- COMPUTE ---
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID2);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID3);
        
        if (k == 0) {
            // First iteration: no accumulate
            TMATMUL(c_l0, a_l0, b_l0[buf_idx]);
        } else {
            // Subsequent: accumulate
            TMATMUL_ACC(c_l0, c_l0, a_l0, b_l0[buf_idx]);
        }
        
        // Signal buffer can be reused
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID4);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID5);
    }
    
    // ========================================
    // DRAINING: Wait for all outstanding flags
    // ========================================
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    
    // Drain MTE1→MTE2 flags
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID4);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID5);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID6);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID7);
    
    // Drain M→MTE1 flags
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID4);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID5);
    
    // Store result
    TSTORE(gC, c_l0);
    
    pipe_barrier(PIPE_ALL);
#endif
}

// Launcher function
template <typename T_A, typename T_B, typename T_C>
void LaunchCubeMatmul4BufPreload(T_A *a, T_B *b, T_C *c, void *stream)
{
    runCubeMatmul4BufPreload<T_A, T_B, T_C>(a, b, c);
}

// Explicit instantiation
template void LaunchCubeMatmul4BufPreload<half, half, float>(half *, half *, float *, void *);
