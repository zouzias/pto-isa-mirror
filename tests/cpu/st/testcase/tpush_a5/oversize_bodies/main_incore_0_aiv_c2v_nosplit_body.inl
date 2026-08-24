/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

    unsigned v5 = 0;
    __gm__ void *v6 = nullptr;
    const int32_t v7 = 8;
    const int32_t v8 = 64;
    const int32_t v9 = 1;
    const int32_t v10 = 512;
    const int32_t v11 = 32;
    const int64_t v12 = 65536;
    const int32_t v13 = 0;
    using T = float;

    set_mask_norm();
    set_vector_mask(-1, -1);
    if (get_subblockid() == 0) {
        auto v14 = TPipe<3, Direction::DIR_C2V, 8192, 8, 2, true>(v6, v13, v13);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
        set_flag(PIPE_V, PIPE_S, EVENT_ID0);
        set_flag(PIPE_V, PIPE_S, EVENT_ID1);
        for (size_t v15 = (size_t)v13; v15 < ((size_t)v7); v15 += (size_t)v9) {
            Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                 CompactMode::Null>
                v16 = Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512,
                           PadValue::Null, CompactMode::Null>(v11, v8);
            TASSIGN(v16, v12);
            Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                 CompactMode::Null>
                v17 = Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512,
                           PadValue::Null, CompactMode::Null>(v11, v8);
            __ubuf__ float *v18 = v16.data();
            uint64_t v19 = reinterpret_cast<uint64_t>(v18);
            TASSIGN(v17, v19);
            pto::Shape<1, 1, 1, 32, 64> v20 = pto::Shape<1, 1, 1, 32, 64>();
            pto::Stride<16384, 16384, 16384, 512, 1> v21 = pto::Stride<16384, 16384, 16384, 512, 1>();
            GlobalTensor<float, pto::Shape<1, 1, 1, 32, 64>, pto::Stride<16384, 16384, 16384, 512, 1>, pto::Layout::ND>
                v22 = GlobalTensor<float, pto::Shape<1, 1, 1, 32, 64>, pto::Stride<16384, 16384, 16384, 512, 1>,
                                   pto::Layout::ND>(
                    v1 + (v5 + v5 * (unsigned)v10 +
                          (unsigned)((int32_t)(uint32_t)((int32_t)(uint32_t)((int32_t)(uint32_t)v4 * (uint32_t)v7) +
                                                         (uint32_t)((int32_t)v15)) *
                                     (uint32_t)v8) *
                              (unsigned)v9),
                    v20, v21);
            wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
            TLOAD(v17, v22);
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                 CompactMode::Null>
                v23 = Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512,
                           PadValue::Null, CompactMode::Null>(v11, v8);
            wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
            TPOP<TPipe<3, Direction::DIR_C2V, 8192, 8, 2, true>,
                 Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                      CompactMode::Null>,
                 TileSplitAxis::TILE_NO_SPLIT>(v14, v23);
            set_flag(PIPE_S, PIPE_V, EVENT_ID0);
            Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                 CompactMode::Null>
                v24 = Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512,
                           PadValue::Null, CompactMode::Null>(v11, v8);
            TASSIGN(v24, v12);
            Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                 CompactMode::Null>
                v25 = Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512,
                           PadValue::Null, CompactMode::Null>(v11, v8);
            __ubuf__ float *v26 = v24.data();
            uint64_t v27 = reinterpret_cast<uint64_t>(v26);
            TASSIGN(v25, v27);
            wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            TADD(v25, v17, v23);
            set_flag(PIPE_V, PIPE_S, EVENT_ID0);
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            TFREE<TPipe<3, Direction::DIR_C2V, 8192, 8, 2, true>, TileSplitAxis::TILE_NO_SPLIT>(v14);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            pipe_barrier(PIPE_MTE3);
            TSTORE(v22, v25);
            set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        }
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID1);
    }

    return;
