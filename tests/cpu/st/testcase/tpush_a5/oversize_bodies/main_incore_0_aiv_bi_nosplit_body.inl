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
    const float v7 = 1.0f;
    const int32_t v8 = 8;
    const int32_t v9 = 64;
    const int32_t v10 = 1;
    const int32_t v11 = 512;
    const int32_t v12 = 32;
    const int64_t v13 = 57344;
    const int64_t v14 = 49152;
    const int64_t v15 = 40960;
    const int64_t v16 = 32768;
    const int32_t v17 = 0;
    using T = float;

    set_mask_norm();
    set_vector_mask(-1, -1);
    if (get_subblockid() == 0) {
        auto v18 = TPipe<1, Direction::DIR_BOTH, 8192, 4, 2, true>(v6, v17, v17);
        Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
             CompactMode::Null>
            v19 = Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                       CompactMode::Null>(v12, v9);
        TASSIGN(v19, v16);
        Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
             CompactMode::Null>
            v20 = Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                       CompactMode::Null>(v12, v9);
        __ubuf__ float *v21 = v19.data();
        uint64_t v22 = reinterpret_cast<uint64_t>(v21);
        TASSIGN(v20, v22);
        pto::Shape<1, 1, 1, 32, 64> v23 = pto::Shape<1, 1, 1, 32, 64>();
        pto::Stride<2048, 2048, 2048, 64, 1> v24 = pto::Stride<2048, 2048, 2048, 64, 1>();
        GlobalTensor<float, pto::Shape<1, 1, 1, 32, 64>, pto::Stride<2048, 2048, 2048, 64, 1>, pto::Layout::ND> v25 =
            GlobalTensor<float, pto::Shape<1, 1, 1, 32, 64>, pto::Stride<2048, 2048, 2048, 64, 1>, pto::Layout::ND>(
                v2 + (v5 + v5 * (unsigned)v9 + v5 * (unsigned)v10), v23, v24);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
        set_flag(PIPE_V, PIPE_S, EVENT_ID0);
        set_flag(PIPE_V, PIPE_S, EVENT_ID1);
        TLOAD(v20, v25);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        for (size_t v26 = (size_t)v17; v26 < ((size_t)v8); v26 += (size_t)v10) {
            Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                 CompactMode::Null>
                v27 = Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512,
                           PadValue::Null, CompactMode::Null>(v12, v9);
            TASSIGN(v27, v15);
            Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                 CompactMode::Null>
                v28 = Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512,
                           PadValue::Null, CompactMode::Null>(v12, v9);
            __ubuf__ float *v29 = v27.data();
            uint64_t v30 = reinterpret_cast<uint64_t>(v29);
            TASSIGN(v28, v30);
            pto::Shape<1, 1, 1, 32, 64> v31 = pto::Shape<1, 1, 1, 32, 64>();
            pto::Stride<16384, 16384, 16384, 512, 1> v32 = pto::Stride<16384, 16384, 16384, 512, 1>();
            GlobalTensor<float, pto::Shape<1, 1, 1, 32, 64>, pto::Stride<16384, 16384, 16384, 512, 1>, pto::Layout::ND>
                v33 = GlobalTensor<float, pto::Shape<1, 1, 1, 32, 64>, pto::Stride<16384, 16384, 16384, 512, 1>,
                                   pto::Layout::ND>(
                    v1 + (v5 + v5 * (unsigned)v11 +
                          (unsigned)((int32_t)(uint32_t)((int32_t)(uint32_t)((int32_t)(uint32_t)v4 * (uint32_t)v8) +
                                                         (uint32_t)((int32_t)v26)) *
                                     (uint32_t)v9) *
                              (unsigned)v10),
                    v31, v32);
            wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
            TLOAD(v28, v33);
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
            Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                 CompactMode::Null>
                v34 = Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512,
                           PadValue::Null, CompactMode::Null>(v12, v9);
            TASSIGN(v34, v14);
            Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                 CompactMode::Null>
                v35 = Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512,
                           PadValue::Null, CompactMode::Null>(v12, v9);
            __ubuf__ float *v36 = v34.data();
            uint64_t v37 = reinterpret_cast<uint64_t>(v36);
            TASSIGN(v35, v37);
            TADDS(v35, v20, v7);
            Tile<TileType::Vec, float, 32, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                 CompactMode::Null>
                v38 = Tile<TileType::Vec, float, 32, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512,
                           PadValue::Null, CompactMode::Null>(v12, v9);
            TASSIGN(v38, v13);
            Tile<TileType::Vec, float, 32, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                 CompactMode::Null>
                v39 = Tile<TileType::Vec, float, 32, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512,
                           PadValue::Null, CompactMode::Null>(v12, v9);
            __ubuf__ float *v40 = v38.data();
            uint64_t v41 = reinterpret_cast<uint64_t>(v40);
            TASSIGN(v39, v41);
            wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
            TMOV(v39, v35);
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            TPUSH<TPipe<1, Direction::DIR_BOTH, 8192, 4, 2, true>,
                  Tile<TileType::Vec, float, 32, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                       CompactMode::Null>,
                  TileSplitAxis::TILE_NO_SPLIT>(v18, v39);
            set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
            Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                 CompactMode::Null>
                v42 = Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512,
                           PadValue::Null, CompactMode::Null>(v12, v9);
            wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
            TPOP<TPipe<1, Direction::DIR_BOTH, 8192, 4, 2, true>,
                 Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                      CompactMode::Null>,
                 TileSplitAxis::TILE_NO_SPLIT>(v18, v42);
            set_flag(PIPE_S, PIPE_V, EVENT_ID0);
            Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                 CompactMode::Null>
                v43 = Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512,
                           PadValue::Null, CompactMode::Null>(v12, v9);
            TASSIGN(v43, v15);
            Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                 CompactMode::Null>
                v44 = Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512,
                           PadValue::Null, CompactMode::Null>(v12, v9);
            __ubuf__ float *v45 = v43.data();
            uint64_t v46 = reinterpret_cast<uint64_t>(v45);
            TASSIGN(v44, v46);
            wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
            TADD(v44, v28, v42);
            set_flag(PIPE_V, PIPE_S, EVENT_ID0);
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
            TFREE<TPipe<1, Direction::DIR_BOTH, 8192, 4, 2, true>, TileSplitAxis::TILE_NO_SPLIT>(v18);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
            pipe_barrier(PIPE_MTE3);
            TSTORE(v33, v44);
            set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        }
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID1);
    }

    return;
