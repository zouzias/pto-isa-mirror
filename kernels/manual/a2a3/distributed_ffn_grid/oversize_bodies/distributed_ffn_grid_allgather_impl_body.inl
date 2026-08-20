    static_assert((FFN_MODEL_TILE % FFN_GRID_COLS) == 0, "AllGather split requires H divisible by gridCols.");

    int blockIdx = 0;
    if (!InitFfnGridKernelBlock(fftsAddr, gridRows, gridCols, blockIdx)) {
        return;
    }

    constexpr int validM = FFN_TOKEN_TILE;     // T
    constexpr int validK = FFN_MODEL_TILE;     // H
    constexpr int validN = FFN_FFN_TILE;       // Fi
    constexpr int validF = FFN_FFN_TOTAL_TILE; // F = Fi * cols
    constexpr int validHShard = FFN_MODEL_SHARD_TILE;
    constexpr int blockAlign = C0_SIZE_BYTE / static_cast<int>(sizeof(half));
    constexpr int M = ((validM + 15) / 16) * 16;
    constexpr int K = ((validK + blockAlign - 1) / blockAlign) * blockAlign;
    constexpr int N = ((validN + blockAlign - 1) / blockAlign) * blockAlign;
    constexpr int KDown = ((validF + blockAlign - 1) / blockAlign) * blockAlign;
    constexpr int NDown = ((validHShard + blockAlign - 1) / blockAlign) * blockAlign;

    using GX = GlobalTensor<half, Shape<1, 1, 1, validM, validK>,
                            Stride<validM * validK, validM * validK, validM * validK, validK, 1>>;
    using GW = GlobalTensor<half, Shape<1, 1, 1, validK, validN>,
                            Stride<validK * validN, validK * validN, validK * validN, validN, 1>>;
    using GWDown =
        GlobalTensor<half, Shape<1, 1, 1, validF, validHShard>,
                     Stride<validF * validHShard, validF * validHShard, validF * validHShard, validHShard, 1>>;
    using TileA = Tile<TileType::Mat, half, M, K, BLayout::ColMajor, validM, validK, SLayout::RowMajor, 512>;
    using TileB = Tile<TileType::Mat, half, K, N, BLayout::ColMajor, validK, validN, SLayout::RowMajor, 512>;
    using HiddenFullMat =
        Tile<TileType::Mat, half, M, KDown, BLayout::ColMajor, validM, validF, SLayout::RowMajor, 512>;
    using WDownMat =
        Tile<TileType::Mat, half, KDown, NDown, BLayout::ColMajor, validF, validHShard, SLayout::RowMajor, 512>;
    using TL = TileLeft<half, M, K, validM, validK>;
    using TR = TileRight<half, K, N, validK, validN>;
    using TC = TileAcc<float, M, N, validM, validN>;
    using TLDown = TileLeft<half, M, KDown, validM, validF>;
    using TRDown = TileRight<half, KDown, NDown, validF, validHShard>;
    using TCDown = TileAcc<float, M, NDown, validM, validHShard>;

    using GatePipe = TPipe<0, Direction::DIR_C2V, FFN_GATE_PARTIAL_BYTES, 1>;
    using UpPipe = TPipe<2, Direction::DIR_C2V, FFN_UP_PARTIAL_BYTES, 1>;
    using HiddenPipe = TPipe<4, Direction::DIR_V2C, FFN_HIDDEN_FULL_BYTES, 1>;
    using DownPipe = TPipe<6, Direction::DIR_C2V, FFN_DOWN_PARTIAL_BYTES, 1>;

    TileA xMat;
    HiddenFullMat hiddenMat;
    TileB wGateMat;
    TileB wUpMat;
    WDownMat wDownMat;
    TASSIGN(xMat, kL1X);
    TASSIGN(hiddenMat, kL1Hidden);
    TASSIGN(wGateMat, kL1WGate);
    TASSIGN(wUpMat, kL1WUp);
    TASSIGN(wDownMat, kL1WDown);

    TL aT;
    TR bT;
    TC cT;
    TLDown aDownT;
    TRDown bDownT;
    TCDown cDownT;
    TASSIGN(aT, 0x0);
    TASSIGN(bT, 0x0);
    TASSIGN(cT, 0x0);
    TASSIGN(aDownT, 0x0);
    TASSIGN(bDownT, 0x0);
    TASSIGN(cDownT, 0x0);

    constexpr int xTileBytes = FFN_X_BYTES;
    constexpr int wGateTileBytes = FFN_W_GATE_BYTES;
    constexpr int wUpTileBytes = FFN_W_UP_BYTES;
    constexpr int wDownTileBytes = FFN_W_DOWN_BYTES;
    constexpr int partialTileBytes = FFN_GATE_PARTIAL_BYTES;
    constexpr int hiddenTileBytes = FFN_HIDDEN_FULL_BYTES;
    constexpr int downTileBytes = FFN_DOWN_PARTIAL_BYTES;
    __gm__ uint8_t *xBlock = x + blockIdx * xTileBytes;
    __gm__ uint8_t *wGateBlock = wGate + blockIdx * wGateTileBytes;
    __gm__ uint8_t *wUpBlock = wUp + blockIdx * wUpTileBytes;
    __gm__ uint8_t *wDownBlock = wDown + blockIdx * wDownTileBytes;
    __gm__ uint8_t *gateBlock = gatePartial + blockIdx * partialTileBytes;
    __gm__ uint8_t *upBlock = upPartial + blockIdx * partialTileBytes;
    __gm__ uint8_t *hiddenBlock = hiddenIn + blockIdx * hiddenTileBytes;
    __gm__ uint8_t *downBlock = downPartial + blockIdx * downTileBytes;
    int row = blockIdx / gridCols;
    int col = blockIdx - row * gridCols;
    __gm__ uint8_t *yBlock =
        yOutput + row * FFN_Y_OUTPUT_BYTES + col * FFN_MODEL_SHARD_TILE * static_cast<int>(sizeof(float));

    GatePipe gatePipe(reinterpret_cast<__gm__ void *>(gateBlock), kUbGateF32, 0);
    UpPipe upPipe(reinterpret_cast<__gm__ void *>(upBlock), kUbUpF32, 0);
    HiddenPipe hiddenPipe(reinterpret_cast<__gm__ void *>(hiddenBlock), 0, kL1Hidden);
    DownPipe downPipe(reinterpret_cast<__gm__ void *>(downBlock), kUbDownF32, 0);

    if constexpr (DAV_CUBE) {
        RunFfnCubePrelude<GX, GW, GW, GWDown>(
            xBlock, wGateBlock, wUpBlock, wDownBlock, xMat, wGateMat, wUpMat, wDownMat, aT, bT, cT, gatePipe, upPipe,
            hiddenMat, hiddenPipe);

        // -------- down: yShard = hidden_full @ W_down[:, Hc] --------
        TMOV(aDownT, hiddenMat);
        TMOV(bDownT, wDownMat);

#ifndef __PTO_AUTO__
        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
#endif

        TMATMUL(cDownT, aDownT, bDownT);

#ifndef __PTO_AUTO__
        set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
        wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
#endif

        TMOV(downPipe, cDownT); // C2V push -> downPipe

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
        HiddenF16Tile eastCarryF16;
        HiddenF16Tile westCarryF16;
        HiddenFullF16Tile hiddenFullF16;
        DownF32Tile downF32;
        TASSIGN(gateF32, kUbGateF32);
        TASSIGN(upF32, kUbUpF32);
        TASSIGN(hiddenF32, kUbHiddenF32);
        TASSIGN(hiddenF16, kUbHiddenF16);
        TASSIGN(eastCarryF16, kUbEastCarryF16);
        TASSIGN(westCarryF16, kUbWestCarryF16);
        TASSIGN(hiddenFullF16, kUbHiddenFullF16);
        TASSIGN(downF32, kUbDownF32);

        RunFfnVecActivation(gateF32, upF32, hiddenF32, hiddenF16, gatePipe, upPipe);

#ifndef __PTO_AUTO__
        pipe_barrier(PIPE_V);
#endif

        FfnGatherPipe gatherPipe;
        GridShape shape{gridRows, gridCols};
        GridCoord coord{row, col};
        __gm__ uint8_t *window = gatherPipeWindow + blockIdx * FFN_GRID_WINDOW_BYTES;
        a2a3_grid::InitGridPipeFromWindow(gatherPipe, shape, coord, window, reinterpret_cast<__gm__ void *>(hcclCtxRaw),
                                          /*pipeId=*/0);

        using pto::GridDirection;
        TINSERT(hiddenFullF16, hiddenF16, 0, static_cast<uint16_t>(col * FFN_FFN_TILE));
#ifndef __PTO_AUTO__
        pipe_barrier(PIPE_V);
#endif

        bool haveEastCarry = true;
        bool haveWestCarry = true;
        for (int step = 0; step < gridCols - 1; ++step) {
            if (col + 1 < gridCols && haveEastCarry) {
                if (step == 0) {
#ifndef __PTO_AUTO__
                    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
                    TPUSH<GridDirection::EAST>(gatherPipe, hiddenF16);
                } else {
#ifndef __PTO_AUTO__
                    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
                    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
#endif
                    TPUSH<GridDirection::EAST>(gatherPipe, eastCarryF16);
                }
            }

            if (col > 0 && haveWestCarry) {
                if (step == 0) {
#ifndef __PTO_AUTO__
                    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
                    TPUSH<GridDirection::WEST>(gatherPipe, hiddenF16);
                } else {
#ifndef __PTO_AUTO__
                    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
                    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
#endif
                    TPUSH<GridDirection::WEST>(gatherPipe, westCarryF16);
                }
            }

            if (col > step) {
                TPOP<GridDirection::EAST>(gatherPipe, eastCarryF16);
#ifndef __PTO_AUTO__
                set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif
                int srcCol = col - step - 1;
                TINSERT(hiddenFullF16, eastCarryF16, 0, static_cast<uint16_t>(srcCol * FFN_FFN_TILE));
#ifndef __PTO_AUTO__
                pipe_barrier(PIPE_V);
#endif
                haveEastCarry = true;
            } else {
                haveEastCarry = false;
            }

            if (col + step + 1 < gridCols) {
                TPOP<GridDirection::WEST>(gatherPipe, westCarryF16);
#ifndef __PTO_AUTO__
                set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif
                int srcCol = col + step + 1;
                TINSERT(hiddenFullF16, westCarryF16, 0, static_cast<uint16_t>(srcCol * FFN_FFN_TILE));
#ifndef __PTO_AUTO__
                pipe_barrier(PIPE_V);
#endif
                haveWestCarry = true;
            } else {
                haveWestCarry = false;
            }
        }

#ifndef __PTO_AUTO__
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
        TMOV(hiddenPipe, hiddenFullF16); // V2C push -> hiddenPipe

        TMOV(downF32, downPipe);         // C2V pop <- downPipe

#ifndef __PTO_AUTO__
        set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
#endif
        GTHShardF32 yG(reinterpret_cast<__gm__ float *>(yBlock));
        TSTORE(yG, downF32);

#ifndef __PTO_AUTO__
        pipe_barrier(PIPE_ALL);
#endif
        dsb(DSB_DDR);
    }
