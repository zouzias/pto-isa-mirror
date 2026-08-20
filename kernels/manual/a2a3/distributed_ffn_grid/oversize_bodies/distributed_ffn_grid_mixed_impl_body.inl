    int blockIdx = 0;
    if (!InitFfnGridKernelBlock(fftsAddr, gridRows, gridCols, blockIdx)) {
        return;
    }

    constexpr int validM = FFN_TOKEN_TILE; // 16
    constexpr int validK = FFN_MODEL_TILE; // 64
    constexpr int validN = FFN_FFN_TILE;   // 64
    constexpr int blockAlign = C0_SIZE_BYTE / static_cast<int>(sizeof(half));
    constexpr int M = ((validM + 15) / 16) * 16;
    constexpr int K = ((validK + blockAlign - 1) / blockAlign) * blockAlign;
    constexpr int N = ((validN + blockAlign - 1) / blockAlign) * blockAlign;

    using GX = GlobalTensor<half, Shape<1, 1, 1, validM, validK>,
                            Stride<validM * validK, validM * validK, validM * validK, validK, 1>>;
    using GW = GlobalTensor<half, Shape<1, 1, 1, validK, validN>,
                            Stride<validK * validN, validK * validN, validK * validN, validN, 1>>;
    using TileA = Tile<TileType::Mat, half, M, K, BLayout::ColMajor, validM, validK, SLayout::RowMajor, 512>;
    using TileB = Tile<TileType::Mat, half, K, N, BLayout::ColMajor, validK, validN, SLayout::RowMajor, 512>;
    using TL = TileLeft<half, M, K, validM, validK>;
    using TR = TileRight<half, K, N, validK, validN>;
    using TC = TileAcc<float, M, N, validM, validN>;

    using GatePipe = TPipe<0, Direction::DIR_C2V, FFN_GATE_PARTIAL_BYTES, 1>;
    using UpPipe = TPipe<2, Direction::DIR_C2V, FFN_UP_PARTIAL_BYTES, 1>;
    using HiddenPipe = TPipe<4, Direction::DIR_V2C, FFN_HIDDEN_BYTES, 1>;
    using DownPipe = TPipe<6, Direction::DIR_C2V, FFN_DOWN_PARTIAL_BYTES, 1>;

    TileA xMat;
    TileA hiddenMat;
    TileB wGateMat;
    TileB wUpMat;
    TileB wDownMat;
    TASSIGN(xMat, kL1X);
    TASSIGN(hiddenMat, kL1Hidden);
    TASSIGN(wGateMat, kL1WGate);
    TASSIGN(wUpMat, kL1WUp);
    TASSIGN(wDownMat, kL1WDown);

    TL aT;
    TR bT;
    TC cT;
    TASSIGN(aT, 0x0);
    TASSIGN(bT, 0x0);
    TASSIGN(cT, 0x0);

    constexpr int xTileBytes = FFN_X_BYTES;
    constexpr int wGateTileBytes = FFN_W_GATE_BYTES;
    constexpr int wUpTileBytes = FFN_W_UP_BYTES;
    constexpr int wDownTileBytes = FFN_W_DOWN_BYTES;
    constexpr int partialTileBytes = FFN_GATE_PARTIAL_BYTES;
    constexpr int hiddenTileBytes = FFN_HIDDEN_BYTES;
    constexpr int downTileBytes = FFN_DOWN_PARTIAL_BYTES;
    __gm__ uint8_t *xBlock = x + blockIdx * xTileBytes;
    __gm__ uint8_t *wGateBlock = wGate + blockIdx * wGateTileBytes;
    __gm__ uint8_t *wUpBlock = wUp + blockIdx * wUpTileBytes;
    __gm__ uint8_t *wDownBlock = wDown + blockIdx * wDownTileBytes;
    __gm__ uint8_t *gateBlock = gatePartial + blockIdx * partialTileBytes;
    __gm__ uint8_t *upBlock = upPartial + blockIdx * partialTileBytes;
    __gm__ uint8_t *hiddenBlock = hiddenIn + blockIdx * hiddenTileBytes;
    __gm__ uint8_t *downBlock = downPartial + blockIdx * downTileBytes;
    __gm__ uint8_t *yBlock = yOutput + (blockIdx / gridCols) * FFN_Y_OUTPUT_BYTES;

    GatePipe gatePipe(reinterpret_cast<__gm__ void *>(gateBlock), kUbGateF32, 0);
    UpPipe upPipe(reinterpret_cast<__gm__ void *>(upBlock), kUbUpF32, 0);
    HiddenPipe hiddenPipe(reinterpret_cast<__gm__ void *>(hiddenBlock), 0, kL1Hidden);
    DownPipe downPipe(reinterpret_cast<__gm__ void *>(downBlock), kUbDownF32, 0);

    if constexpr (DAV_CUBE) {
        RunFfnCubePrelude<GX, GW, GW, GW>(
            xBlock, wGateBlock, wUpBlock, wDownBlock, xMat, wGateMat, wUpMat, wDownMat, aT, bT, cT, gatePipe, upPipe,
            hiddenMat, hiddenPipe);

        // -------- down: downPartial = hidden @ W_down --------
        TMOV(aT, hiddenMat);
        TMOV(bT, wDownMat);

#ifndef __PTO_AUTO__
        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
#endif

        TMATMUL(cT, aT, bT);

#ifndef __PTO_AUTO__
        set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
        wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
#endif

        TMOV(downPipe, cT); // C2V push -> downPipe

#ifndef __PTO_AUTO__
        pipe_barrier(PIPE_ALL);
#endif
        dsb(DSB_DDR);
    }

    if constexpr (DAV_VEC) {
        GateF32Tile gateF32;
        UpF32Tile upF32;
        HiddenF32Tile hiddenF32;
        HiddenF16Tile hiddenF16;
        DownF32Tile downF32;
        DownF32Tile addF32;
        TASSIGN(gateF32, kUbGateF32);
        TASSIGN(upF32, kUbUpF32);
        TASSIGN(hiddenF32, kUbHiddenF32);
        TASSIGN(hiddenF16, kUbHiddenF16);
        TASSIGN(downF32, kUbDownF32);
        TASSIGN(addF32, kUbAddF32);

        RunFfnVecActivation(gateF32, upF32, hiddenF32, hiddenF16, gatePipe, upPipe);

#ifndef __PTO_AUTO__
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif

        TMOV(hiddenPipe, hiddenF16); // V2C push -> hiddenPipe

        TMOV(downF32, downPipe);     // C2V pop <- downPipe

#ifndef __PTO_AUTO__
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif

        FfnGridPipe reducePipe;
        GridShape shape{gridRows, gridCols};
        GridCoord coord{blockIdx / gridCols, blockIdx - (blockIdx / gridCols) * gridCols};
        __gm__ uint8_t *window = reducePipeWindow + blockIdx * FFN_GRID_WINDOW_BYTES;
        a2a3_grid::InitGridPipeFromWindow(reducePipe, shape, coord, window, reinterpret_cast<__gm__ void *>(hcclCtxRaw),
                                          /*pipeId=*/0);

        using pto::GridDirection;
        int col = coord.col;
        if (col > 0) {
            TPOP<GridDirection::EAST>(reducePipe, addF32);
#ifndef __PTO_AUTO__
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            pipe_barrier(PIPE_ALL);
#endif
            dsb(DSB_DDR);
            TADD(downF32, downF32, addF32);
#ifndef __PTO_AUTO__
            pipe_barrier(PIPE_V);
#endif
        }

        if (col + 1 < gridCols) {
#ifndef __PTO_AUTO__
            if (col > 0) {
                set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            } else {
                set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
                wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            }
            pipe_barrier(PIPE_ALL);
#endif
            dsb(DSB_DDR);
            TPUSH<GridDirection::EAST>(reducePipe, downF32);
        } else {
#ifndef __PTO_AUTO__
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
            GTHF32 yG(reinterpret_cast<__gm__ float *>(yBlock));
            TSTORE(yG, downF32);
        }

#ifndef __PTO_AUTO__
        pipe_barrier(PIPE_ALL);
#endif
        dsb(DSB_DDR);
    }
