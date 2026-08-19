    using Src1Shape = Shape<1, 1, 1, 1, 1>;
    using Src1Stride = pto::Stride<32 / sizeof(T), 32 / sizeof(T), 32 / sizeof(T), 32 / sizeof(T), 1>;
    using Src0Shape = Shape<1, 1, 1, ValidRows, ValidCols>;
    using Src0Stride = pto::Stride<Rows * Cols, Rows * Cols, Rows * Cols, Cols, 1>;
    using Src0Global = GlobalTensor<T, Src0Shape, Src0Stride>;
    using Src1Global = GlobalTensor<T, Src1Shape, Src1Stride>;

    constexpr int dstCols = (Cols + 7) / 8;
    constexpr int dstValidCols = (ValidCols + 7) / 8;
    constexpr int dstTileCols = ((Cols / 8) + 31) / 32 * 32;
    using DstShape = Shape<1, 1, 1, ValidRows, dstValidCols>;
    using DstStride = pto::Stride<Rows * dstCols, Rows * dstCols, Rows * dstCols, dstCols, 1>;
    using DstGlobal = GlobalTensor<uint8_t, DstShape, DstStride>;

    Src0Global src0Global(src0);
    Src1Global src1Global(src1);
    DstGlobal dstGlobal(out);

    using Src0Tile = Tile<TileType::Vec, T, Rows, Cols, BLayout::RowMajor, ValidRows, ValidCols>;
    using Src1Tile = Tile<TileType::Vec, T, 1, 32 / sizeof(T), BLayout::RowMajor, 1, 1>;
    using DstTile = Tile<TileType::Vec, uint8_t, Rows, dstTileCols, BLayout::RowMajor, ValidRows, dstValidCols>;

    Src0Tile src0Tile;
    Src1Tile src1Tile;
    DstTile dstTile;
    TASSIGN<0x0>(src0Tile);
    TASSIGN<Src0Tile::Numel * sizeof(T)>(src1Tile);
    TASSIGN<Src0Tile::Numel * sizeof(T) + 32>(dstTile);
    T scalar = *src1;

    TLOAD(src0Tile, src0Global);
    if constexpr (isSrc1Tile) {
        TLOAD(src1Tile, src1Global);
    }
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif

    if constexpr (isSrc1Tile) {
        TCMPS(dstTile, src0Tile, src1Tile, cmpMode);
    } else {
        TCMPS(dstTile, src0Tile, scalar, cmpMode);
    }

#ifndef __PTO_AUTO__
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
    TSTORE(dstGlobal, dstTile);
