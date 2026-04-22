/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <pto/pto-inst.hpp>

using namespace pto;

// ptoas_auto_sync_tail helper
enum class PTOAutoSyncTailMode : int {
    kBarrierAll = 0,
    kSetWaitMte3ToSEvent0 = 1,
};

static AICORE inline void ptoas_auto_sync_tail(PTOAutoSyncTailMode mode = PTOAutoSyncTailMode::kBarrierAll)
{
    switch (mode) {
        case PTOAutoSyncTailMode::kSetWaitMte3ToSEvent0:
            set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
            wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
            break;
        case PTOAutoSyncTailMode::kBarrierAll:
        default:
            pipe_barrier(PIPE_ALL);
            break;
    }
}

// -------------------------------------------------------------------------
// RunQKAttnIncore5
//   all_raw_scores : [>=1024, 64] float (output, only [16*ctx_blocks, 64] written)
//   k_cache        : [B*32, KVH*16, 64, 128] bfloat16 stored in DN layout
//   q_padded       : [16, 128] bfloat16
//   sb_idx         : super-block index (0-based, for ctx_blocks=1 always 0)
//   ctx_blocks     : number of 64-row context blocks (v5 = ctx_blocks * 64 limit)
//   b_idx          : batch index
//   kvh            : KV head index
// -------------------------------------------------------------------------
__global__ AICORE void RunQKAttnIncore5(__gm__ float *all_raw_scores, __gm__ bfloat16_t *k_cache,
                                        __gm__ bfloat16_t *q_padded, int32_t sb_idx, int32_t ctx_blocks,
                                        int32_t b_idx, int32_t kvh)
{
    unsigned v8 = 0;
    __gm__ void *v9 = nullptr;

    // -----------------------------------------------------------------------
    // AIC (cube) body – verbatim from qwen3_decode_split_incore_5_aic
    // -----------------------------------------------------------------------
#if defined(__DAV_CUBE__)
    {
        const int32_t v10_4096 = 4096;
        const int32_t v11_32768 = 32768;
        const int32_t v12_128 = 128;
        const int32_t v13_1 = 1;
        const int32_t v14_64 = 64;
        const int64_t v15_16384 = 16384;
        const int64_t v16_0 = 0;
        const int32_t v17_0 = 0;

        // v4=sb_idx, v5=ctx_blocks*64 (total rows bound), v6=b_idx, v7=kvh
        int32_t v4 = sb_idx;
        int32_t v5 = ctx_blocks * v14_64;
        int32_t v6 = b_idx;
        int32_t v7 = kvh;

        auto v18 = TPipe<0, Direction::DIR_C2V, 4096, 8>(v9, v17_0, v17_0);
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
        set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);

        for (size_t v19 = (size_t)v17_0; v19 < (size_t)v14_64; v19 += (size_t)v13_1) {
            int32_t v20 = (int32_t)((uint32_t)((int32_t)(uint32_t)v4 * (uint32_t)v14_64) +
                                    (uint32_t)((int32_t)v19));
            if (v20 < v5) {
                // K tile: Mat[128,64] RowMajor/ColMajor, TASSIGN=0x0
                Tile<TileType::Mat, bfloat16_t, 128, 64, BLayout::RowMajor, 128, 64, SLayout::ColMajor, 512,
                     PadValue::Null, CompactMode::Null>
                    v22;
                TASSIGN(v22, v16_0);

                // DN layout GlobalTensor for K
                pto::Shape<1, 1, 1, 128, 64> v23 = pto::Shape<1, 1, 1, 128, 64>();
                pto::Stride<128, 128, 128, 1, 128> v24 = pto::Stride<128, 128, 128, 1, 128>();
                GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 128, 64>, pto::Stride<128, 128, 128, 1, 128>,
                             pto::Layout::DN>
                    v25 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 128, 64>,
                                       pto::Stride<128, 128, 128, 1, 128>, pto::Layout::DN>(
                        k_cache + (v8 + v8 * (unsigned)v13_1 +
                                   (unsigned)((int32_t)(uint32_t)((int32_t)(uint32_t)((int32_t)(uint32_t)v6 *
                                                                                      (uint32_t)v11_32768) +
                                                                  (uint32_t)((int32_t)(uint32_t)v7 *
                                                                             (uint32_t)v10_4096)) +
                                              (uint32_t)((int32_t)(uint32_t)v20 * (uint32_t)v14_64)) *
                                       (unsigned)v12_128),
                        v23, v24);
                wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
                TLOAD(v22, v25);

                // Q tile: Mat[16,128] ColMajor/RowMajor, TASSIGN=0x4000
                Tile<TileType::Mat, bfloat16_t, 16, 128, BLayout::ColMajor, 16, 128, SLayout::RowMajor, 512,
                     PadValue::Null, CompactMode::Null>
                    v26;
                TASSIGN(v26, v15_16384);

                pto::Shape<1, 1, 1, 16, 128> v27 = pto::Shape<1, 1, 1, 16, 128>();
                pto::Stride<2048, 2048, 2048, 128, 1> v28 = pto::Stride<2048, 2048, 2048, 128, 1>();
                GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 16, 128>, pto::Stride<2048, 2048, 2048, 128, 1>,
                             pto::Layout::ND>
                    v29 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 16, 128>,
                                       pto::Stride<2048, 2048, 2048, 128, 1>, pto::Layout::ND>(
                        q_padded + (v8 + v8 * (unsigned)v12_128 + v8 * (unsigned)v13_1), v27, v28);
                TLOAD(v26, v29);
                set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

                // L0A <- Q
                Tile<TileType::Left, bfloat16_t, 16, 128, BLayout::ColMajor, 16, 128, SLayout::RowMajor, 512,
                     PadValue::Null, CompactMode::Null>
                    v30;
                TASSIGN(v30, v16_0);
                wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
                wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
                TMOV(v30, v26);

                // L0B <- K
                Tile<TileType::Right, bfloat16_t, 128, 64, BLayout::RowMajor, 128, 64, SLayout::ColMajor, 512,
                     PadValue::Null, CompactMode::Null>
                    v31;
                TASSIGN(v31, v16_0);
                TMOV(v31, v22);
                set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
                set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);

                // Acc tile, TASSIGN=0x0
                Tile<TileType::Acc, float, 16, 64, BLayout::ColMajor, 16, 64, SLayout::RowMajor, 1024,
                     PadValue::Null, CompactMode::Null>
                    v32;
                TASSIGN(v32, v16_0);
                wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
                wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
                TMATMUL(v32, v30, v31);
                set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
                set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
                wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);

                TPUSH<TPipe<0, Direction::DIR_C2V, 4096, 8>,
                      Tile<TileType::Acc, float, 16, 64, BLayout::ColMajor, 16, 64, SLayout::RowMajor, 1024,
                           PadValue::Null, CompactMode::Null>,
                      TileSplitAxis::TILE_UP_DOWN>(v18, v32);
                set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
            }
        }

        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
        wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    }
#endif  // __DAV_CUBE__

    // -----------------------------------------------------------------------
    // AIV (vector) body – verbatim from qwen3_decode_split_incore_5_aiv
    // -----------------------------------------------------------------------
#if defined(__DAV_VEC__)
    {
        const int32_t v10_8 = 8;
        const float v11_scale = 0.0883883461f;
        const int32_t v12_16 = 16;
        const int32_t v13_1 = 1;
        const int32_t v14_64 = 64;
        const int64_t v15_34816 = 34816;
        const int32_t v16_0 = 0;

        int32_t v4 = sb_idx;
        int32_t v5 = ctx_blocks * v14_64;

        set_mask_norm();
        set_vector_mask(-1, -1);
        int64_t v17 = get_subblockid();
        auto v18 = TPipe<0, Direction::DIR_C2V, 4096, 8>(v9, v16_0, v16_0);
        set_flag(PIPE_V, PIPE_S, EVENT_ID0);
        set_flag(PIPE_V, PIPE_S, EVENT_ID1);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);

        for (size_t v19 = (size_t)v16_0; v19 < (size_t)v14_64; v19 += (size_t)v13_1) {
            int32_t v20 = (int32_t)((uint32_t)((int32_t)(uint32_t)v4 * (uint32_t)v14_64) +
                                    (uint32_t)((int32_t)v19));
            if (v20 < v5) {
                // TPOP: Vec[8,64] NoneBox, TASSIGN implicit from pipe
                Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, 8, 64, SLayout::NoneBox, 512,
                     PadValue::Null, CompactMode::Null>
                    v22;
                wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
                TPOP<TPipe<0, Direction::DIR_C2V, 4096, 8>,
                     Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, 8, 64, SLayout::NoneBox, 512,
                          PadValue::Null, CompactMode::Null>,
                     TileSplitAxis::TILE_UP_DOWN>(v18, v22);
                set_flag(PIPE_S, PIPE_V, EVENT_ID0);

                // Output tile, TASSIGN=0x8800 (34816)
                Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, 8, 64, SLayout::NoneBox, 512,
                     PadValue::Null, CompactMode::Null>
                    v23;
                TASSIGN(v23, v15_34816);
                wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
                wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
                TMULS(v23, v22, v11_scale);
                set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                set_flag(PIPE_V, PIPE_S, EVENT_ID0);

                TFREE<TPipe<0, Direction::DIR_C2V, 4096, 8>, TileSplitAxis::TILE_UP_DOWN>(v18);

                // Output GM: row = v20*16 + subblk*8
                pto::Shape<1, 1, 1, 8, 64> v24 = pto::Shape<1, 1, 1, 8, 64>();
                pto::Stride<512, 512, 512, 64, 1> v25 = pto::Stride<512, 512, 512, 64, 1>();
                GlobalTensor<float, pto::Shape<1, 1, 1, 8, 64>, pto::Stride<512, 512, 512, 64, 1>,
                             pto::Layout::ND>
                    v26 = GlobalTensor<float, pto::Shape<1, 1, 1, 8, 64>,
                                       pto::Stride<512, 512, 512, 64, 1>, pto::Layout::ND>(
                        all_raw_scores +
                            (v8 + (unsigned)((int32_t)(uint32_t)((int32_t)(uint32_t)v20 * (uint32_t)v12_16) +
                                             (uint32_t)((int32_t)(uint32_t)((int32_t)(int64_t)v17) *
                                                        (uint32_t)v10_8)) *
                                      (unsigned)v14_64 +
                             v8 * (unsigned)v13_1),
                        v24, v25);

                wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                pipe_barrier(PIPE_MTE3);
                TSTORE(v26, v23);
                set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
            }
        }

        wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID1);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
    }
#endif  // __DAV_VEC__

    ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
}

// -------------------------------------------------------------------------
// Launch wrapper
// -------------------------------------------------------------------------
template <int32_t tilingKey>
void LaunchQKAttnIncore5(uint8_t *all_raw_scores, uint8_t *k_cache, uint8_t *q_padded, int32_t sb_idx,
                         int32_t ctx_blocks, int32_t b_idx, int32_t kvh, void *stream)
{
    if constexpr (tilingKey == 1) {
        RunQKAttnIncore5<<<1, nullptr, stream>>>(reinterpret_cast<float *>(all_raw_scores),
                                                 reinterpret_cast<bfloat16_t *>(k_cache),
                                                 reinterpret_cast<bfloat16_t *>(q_padded), sb_idx, ctx_blocks,
                                                 b_idx, kvh);
    }
}

template void LaunchQKAttnIncore5<1>(uint8_t *all_raw_scores, uint8_t *k_cache, uint8_t *q_padded,
                                     int32_t sb_idx, int32_t ctx_blocks, int32_t b_idx, int32_t kvh,
                                     void *stream);
