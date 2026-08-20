    unsigned v5 = 0;
    __gm__ void *v6 = nullptr;
    const float v7 = 1.0f;
    const int32_t v8 = 8;
    const int32_t v9 = 16;
    const int32_t v10 = 64;
    const int32_t v11 = 1;
    const int32_t v12 = 512;
    const int64_t v13 = 45056;
    const int64_t v14 = 40960;
    const int64_t v15 = 36864;
    const int64_t v16 = 32768;
    const int32_t v17 = 0;
    using T = float;

    set_mask_norm();
    set_vector_mask(-1, -1);
    int64_t v18 = get_subblockid();
    auto v19 = TPipe<2, Direction::DIR_BOTH, 8192, 4, 2, false>(v6, v17, v17);
    Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
         CompactMode::Null>
        v20 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                   CompactMode::Null>(v9, v10);
    TASSIGN(v20, v16);
    Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
         CompactMode::Null>
        v21 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                   CompactMode::Null>(v9, v10);
    __ubuf__ float *v22 = v20.data();
    uint64_t v23 = reinterpret_cast<uint64_t>(v22);
    TASSIGN(v21, v23);
    int32_t v24 = (int32_t)((uint32_t)((int32_t)(int64_t)v18) * (uint32_t)v9);
    pto::Shape<1, 1, 1, 16, 64> v25 = pto::Shape<1, 1, 1, 16, 64>();
    pto::Stride<1024, 1024, 1024, 64, 1> v26 = pto::Stride<1024, 1024, 1024, 64, 1>();
    GlobalTensor<float, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<1024, 1024, 1024, 64, 1>, pto::Layout::ND> v27 =
        GlobalTensor<float, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<1024, 1024, 1024, 64, 1>, pto::Layout::ND>(
            v2 + (v5 + (unsigned)v24 * (unsigned)v10 + v5 * (unsigned)v11), v25, v26);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
    set_flag(PIPE_V, PIPE_S, EVENT_ID0);
    set_flag(PIPE_V, PIPE_S, EVENT_ID1);
    TLOAD(v21, v27);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    for (size_t v28 = (size_t)v17; v28 < ((size_t)v8); v28 += (size_t)v11) {
        Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
             CompactMode::Null>
            v29 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                       CompactMode::Null>(v9, v10);
        TASSIGN(v29, v15);
        Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
             CompactMode::Null>
            v30 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                       CompactMode::Null>(v9, v10);
        __ubuf__ float *v31 = v29.data();
        uint64_t v32 = reinterpret_cast<uint64_t>(v31);
        TASSIGN(v30, v32);
        pto::Shape<1, 1, 1, 16, 64> v33 = pto::Shape<1, 1, 1, 16, 64>();
        pto::Stride<8192, 8192, 8192, 512, 1> v34 = pto::Stride<8192, 8192, 8192, 512, 1>();
        GlobalTensor<float, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<8192, 8192, 8192, 512, 1>, pto::Layout::ND> v35 =
            GlobalTensor<float, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<8192, 8192, 8192, 512, 1>, pto::Layout::ND>(
                v1 + (v5 + (unsigned)v24 * (unsigned)v12 +
                      (unsigned)((int32_t)(uint32_t)((int32_t)(uint32_t)((int32_t)(uint32_t)v4 * (uint32_t)v8) +
                                                     (uint32_t)((int32_t)v28)) *
                                 (uint32_t)v10) *
                          (unsigned)v11),
                v33, v34);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        TLOAD(v30, v35);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
        Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
             CompactMode::Null>
            v36 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                       CompactMode::Null>(v9, v10);
        TASSIGN(v36, v14);
        Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
             CompactMode::Null>
            v37 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                       CompactMode::Null>(v9, v10);
        __ubuf__ float *v38 = v36.data();
        uint64_t v39 = reinterpret_cast<uint64_t>(v38);
        TASSIGN(v37, v39);
        TADDS(v37, v21, v7);
        Tile<TileType::Vec, float, 16, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
             CompactMode::Null>
            v40 = Tile<TileType::Vec, float, 16, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                       CompactMode::Null>(v9, v10);
        TASSIGN(v40, v13);
        Tile<TileType::Vec, float, 16, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
             CompactMode::Null>
            v41 = Tile<TileType::Vec, float, 16, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                       CompactMode::Null>(v9, v10);
        __ubuf__ float *v42 = v40.data();
        uint64_t v43 = reinterpret_cast<uint64_t>(v42);
        TASSIGN(v41, v43);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        TMOV(v41, v37);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        TPUSH<TPipe<2, Direction::DIR_BOTH, 8192, 4, 2, false>,
              Tile<TileType::Vec, float, 16, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                   CompactMode::Null>,
              TileSplitAxis::TILE_UP_DOWN>(v19, v41);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
             CompactMode::Null>
            v44 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                       CompactMode::Null>(v9, v10);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
        TPOP<TPipe<2, Direction::DIR_BOTH, 8192, 4, 2, false>,
             Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                  CompactMode::Null>,
             TileSplitAxis::TILE_UP_DOWN>(v19, v44);
        set_flag(PIPE_S, PIPE_V, EVENT_ID0);
        Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
             CompactMode::Null>
            v45 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                       CompactMode::Null>(v9, v10);
        TASSIGN(v45, v15);
        Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
             CompactMode::Null>
            v46 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                       CompactMode::Null>(v9, v10);
        __ubuf__ float *v47 = v45.data();
        uint64_t v48 = reinterpret_cast<uint64_t>(v47);
        TASSIGN(v46, v48);
        wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
        TADD(v46, v30, v44);
        set_flag(PIPE_V, PIPE_S, EVENT_ID0);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
        TFREE<TPipe<2, Direction::DIR_BOTH, 8192, 4, 2, false>, TileSplitAxis::TILE_UP_DOWN>(v19);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
        pipe_barrier(PIPE_MTE3);
        TSTORE(v35, v46);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    }
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID1);

    return;
