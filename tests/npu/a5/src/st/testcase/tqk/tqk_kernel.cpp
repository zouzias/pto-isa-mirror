/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <pto/pto-inst.hpp>
#include "pto_macro_matmul.hpp"
#define ND_LAYOUT 0
using namespace pto;

enum class QkAccMode { Init, Acc };

template <int S0, int H, int S1, int CUBE_S0, int CUBE_S1, int TILE_S1>
__global__ AICORE void RunTQK(__gm__ half *q, __gm__ half *k, __gm__ float *qk)
{
    static_assert(TILE_S1 == CUBE_S1, "TILE_S1 must equal CUBE_S1");
    constexpr int blocks = S0 / CUBE_S0;


#if ND_LAYOUT
    // ND Layout: Q(S0,H) row-major, K(H,S1) stored as (S1,H) transposed (DN), QK(S0,S1) row-major
    using GlobalQBlk = GlobalTensor<half, Shape<1, 1, 1, CUBE_S0, H>, pto::Stride<1, 1, 1, H, 1>>;
    using GlobalKAll = GlobalTensor<half, Shape<1, 1, 1, H, CUBE_S1>, pto::Stride<1, 1, 1, 1, H>, pto::Layout::DN>;
   
    using QTile = Tile<TileType::Mat, half, CUBE_S0, H, BLayout::ColMajor, CUBE_S0, H, SLayout::RowMajor, 512>;
    using KTile = Tile<TileType::Mat, half, H, CUBE_S1, BLayout::RowMajor, H, CUBE_S1, SLayout::ColMajor, 512>;
    using AccTile = TileAcc<float, CUBE_S0, CUBE_S1, CUBE_S0, CUBE_S1>;
#else
    // DN Layout: Want K^T @ Q^T = (CUBE_S1,H) @ (H,CUBE_S0) = (CUBE_S1,CUBE_S0)
    // K stored as (S1,H) row-major → read as (CUBE_S1,H) directly
    // Q stored as (S0,H) row-major → use DN to get (H,CUBE_S0) 
    using GlobalKAll = GlobalTensor<half, Shape<1, 1, 1, CUBE_S1, H>, pto::Stride<1, 1, 1, H, 1>>;
    using GlobalQBlk = GlobalTensor<half, Shape<1, 1, 1, H, CUBE_S0>, pto::Stride<1, 1, 1, 1, H>, pto::Layout::DN>;
    using KTile = Tile<TileType::Mat, half, CUBE_S1, H, BLayout::ColMajor, CUBE_S1, H, SLayout::RowMajor, 512>;
    using QTile = Tile<TileType::Mat, half, H, CUBE_S0, BLayout::RowMajor, H, CUBE_S0, SLayout::ColMajor, 512>;
    using AccTile = TileAcc<float, CUBE_S1, CUBE_S0, CUBE_S1, CUBE_S0>;
#endif


    GlobalQBlk qGlobalBlk(q);
    

    QTile qMat;
    KTile kMat;
    AccTile acc;

    TASSIGN(qMat, 0x0);
    TASSIGN(kMat, 0x10000);
    TASSIGN(acc, 0x0);

    set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);

    for (int tile_id = 0; tile_id < S1; tile_id += CUBE_S1) {
        const int s1_off = tile_id;
        
#if ND_LAYOUT
        // K is stored as (S1,H) transposed, access with DN layout
        GlobalKAll kSub(k + static_cast<size_t>(s1_off) * H);
#else
        // DN: K is stored as (S1,H), read as (CUBE_S1,H) row-major
        GlobalKAll kSub(k + static_cast<size_t>(s1_off) * H);
#endif
        if (tile_id == 0 ){
                // Load Q once (reused for all K tiles)
                TLOAD(qMat, qGlobalBlk);
        }
        
        TLOAD(kMat, kSub);

        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);

        wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
        

        // Each tile computes a separate portion - always Init since Q@K_i produces different output columns
#if ND_LAYOUT
        pto_macro_matmul<CUBE_S0, H, CUBE_S1>(qMat, kMat, acc, AccMode::Init);
#else
        // DN: K(CUBE_S1,H) @ Q(H,CUBE_S0) = (CUBE_S1,CUBE_S0)
        pto_macro_matmul<CUBE_S1, H, CUBE_S0>(kMat, qMat, acc, AccMode::Init);
#endif

        // Cleanup: wait for the last iteration to complete
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);

        // Store result
#if ND_LAYOUT
        using GlobalQKBlk = GlobalTensor<float, Shape<1, 1, 1, CUBE_S0, CUBE_S1>, pto::Stride<1, 1, 1, S1, 1>>;

        GlobalQKBlk qkGlobal(qk + static_cast<size_t>(s1_off));
#else
        using GlobalQKBlk = GlobalTensor<float, Shape<1, 1, 1, CUBE_S1, S0>, pto::Stride<1, 1, 1, S0, 1>>;

        GlobalQKBlk qkGlobal(qk + static_cast<size_t>(s1_off * S0));

#endif 
        set_flag(PIPE_M, PIPE_FIX, EVENT_ID1);
        wait_flag(PIPE_M, PIPE_FIX, EVENT_ID1);
        TSTORE(qkGlobal, acc);
        set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    }

    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);


}

template <int S0, int H, int S1, int CUBE_S0, int CUBE_S1, int TILE_S1>
void LaunchTQK(uint16_t *q, uint16_t *k, float *qk, void *stream)
{
    constexpr int blocks = S0 / CUBE_S0;
    RunTQK<S0, H, S1, CUBE_S0, CUBE_S1, TILE_S1>
        <<<blocks, nullptr, stream>>>((__gm__ half *)q, (__gm__ half *)k, (__gm__ float *)qk);
}

template void LaunchTQK<128, 128, 256, 128, 128, 128>(uint16_t *, uint16_t *, float *, void *);
template void LaunchTQK<128, 128, 512, 128, 128, 128>(uint16_t *, uint16_t *, float *, void *);
template void LaunchTQK<128, 128, 128, 128, 128, 128>(uint16_t *, uint16_t *, float *, void *);
template void LaunchTQK<256, 128, 64, 256, 64, 64>(uint16_t *, uint16_t *, float *, void *);