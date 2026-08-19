    unsigned v5 = 0;
    __gm__ void *v6 = nullptr;
    const int32_t v7 = 8;
    const int32_t v8 = 64;
    const int32_t v9 = 1;
    const int32_t v10 = 512;
    const int32_t v11 = 32;
    const int64_t v12 = 0;
    const int64_t v13 = 32768;
    const int32_t v14 = 0;
    using T = float;

    auto v15 = TPipe<0, Direction::DIR_BOTH, 8192, 4, 2, false>(v6, v14, v14);
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
    set_flag(PIPE_MTE1, PIPE_S, EVENT_ID0);
    set_flag(PIPE_MTE1, PIPE_S, EVENT_ID1);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    for (size_t v16 = (size_t)v14; v16 < ((size_t)v7); v16 += (size_t)v9) {
        Tile<TileType::Mat, float, 64, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
             CompactMode::Null>
            v17 = Tile<TileType::Mat, float, 64, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                       CompactMode::Null>(v8, v8);
        TASSIGN(v17, v13);
        Tile<TileType::Mat, float, 64, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
             CompactMode::Null>
            v18 = Tile<TileType::Mat, float, 64, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                       CompactMode::Null>(v8, v8);
        __cbuf__ float *v19 = v17.data();
        uint64_t v20 = reinterpret_cast<uint64_t>(v19);
        TASSIGN(v18, v20);
        pto::Shape<1, 1, 1, 64, 64> v21 = pto::Shape<1, 1, 1, 64, 64>();
        pto::Stride<32768, 32768, 32768, 512, 1> v22 = pto::Stride<32768, 32768, 32768, 512, 1>();
        GlobalTensor<float, pto::Shape<1, 1, 1, 64, 64>, pto::Stride<32768, 32768, 32768, 512, 1>, pto::Layout::ND>
            v23 = GlobalTensor<float, pto::Shape<1, 1, 1, 64, 64>, pto::Stride<32768, 32768, 32768, 512, 1>,
                               pto::Layout::ND>(
                v3 + (v5 + v5 * (unsigned)v10 +
                      (unsigned)((int32_t)(uint32_t)((int32_t)(uint32_t)((int32_t)(uint32_t)v4 * (uint32_t)v7) +
                                                     (uint32_t)((int32_t)v16)) *
                                 (uint32_t)v8) *
                          (unsigned)v9),
                v21, v22);
        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
        TLOAD(v18, v23);
        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
        Tile<TileType::Mat, float, 32, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
             CompactMode::Null>
            v24 = Tile<TileType::Mat, float, 32, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                       CompactMode::Null>(v11, v8);
        wait_flag(PIPE_MTE1, PIPE_S, EVENT_ID0);
        TPOP<TPipe<0, Direction::DIR_BOTH, 8192, 4, 2, false>,
             Tile<TileType::Mat, float, 32, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                  CompactMode::Null>,
             TileSplitAxis::TILE_LEFT_RIGHT>(v15, v24);
        set_flag(PIPE_S, PIPE_MTE1, EVENT_ID0);
        Tile<TileType::Left, float, 32, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
             CompactMode::Null>
            v25 = Tile<TileType::Left, float, 32, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                       CompactMode::Null>(v11, v8);
        TASSIGN(v25, v12);
        Tile<TileType::Left, float, 32, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
             CompactMode::Null>
            v26 = Tile<TileType::Left, float, 32, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                       CompactMode::Null>(v11, v8);
        __ca__ float *v27 = v25.data();
        uint64_t v28 = reinterpret_cast<uint64_t>(v27);
        TASSIGN(v26, v28);
        wait_flag(PIPE_S, PIPE_MTE1, EVENT_ID0);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
        TMOV(v26, v24);
        set_flag(PIPE_MTE1, PIPE_S, EVENT_ID0);
        TFREE<TPipe<0, Direction::DIR_BOTH, 8192, 4, 2, false>, TileSplitAxis::TILE_LEFT_RIGHT>(v15);
        Tile<TileType::Right, float, 64, 64, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
             CompactMode::Null>
            v29 = Tile<TileType::Right, float, 64, 64, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512,
                       PadValue::Null, CompactMode::Null>(v8, v8);
        TASSIGN(v29, v12);
        Tile<TileType::Right, float, 64, 64, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
             CompactMode::Null>
            v30 = Tile<TileType::Right, float, 64, 64, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512,
                       PadValue::Null, CompactMode::Null>(v8, v8);
        __cb__ float *v31 = v29.data();
        uint64_t v32 = reinterpret_cast<uint64_t>(v31);
        TASSIGN(v30, v32);
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
        TMOV(v30, v18);
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        Tile<TileType::Acc, float, 32, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
             CompactMode::Null>
            v33 = Tile<TileType::Acc, float, 32, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                       CompactMode::Null>(v11, v8);
        TASSIGN(v33, v12);
        Tile<TileType::Acc, float, 32, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
             CompactMode::Null>
            v34 = Tile<TileType::Acc, float, 32, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                       CompactMode::Null>(v11, v8);
        __cc__ float *v35 = v33.data();
        uint64_t v36 = reinterpret_cast<uint64_t>(v35);
        TASSIGN(v34, v36);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
        TMATMUL(v34, v26, v30);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
        set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
        wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
        TPUSH<TPipe<0, Direction::DIR_BOTH, 8192, 4, 2, false>,
              Tile<TileType::Acc, float, 32, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                   CompactMode::Null>,
              TileSplitAxis::TILE_LEFT_RIGHT>(v15, v34);
        set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    }
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
    wait_flag(PIPE_MTE1, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_S, EVENT_ID1);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);

    return;
