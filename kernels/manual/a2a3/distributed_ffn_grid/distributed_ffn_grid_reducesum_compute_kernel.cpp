/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under
the terms and conditions of CANN Open Software License Agreement Version 2.0
(the "License"). Please refer to the License for details. You may not use this
file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON AN "AS
IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
PARTICULAR PURPOSE. See LICENSE in the root of the software repository for the
full text of the License.
*/

// Single-device multi-block FFN mixed Cube/Vec kernel.
//
// Each block owns one logical grid cell (row, col).  Cube computes the three
// GEMMs, Vec computes the activation and row-local EAST reduce.  Cube/Vec
// exchange intermediate tiles through regular A2/A3 TPipe FIFO handshakes,
// issued as directional TMOV (see tpipe_tmov_inl.hpp): TMOV(pipe, tile) pushes
// and TMOV(tile, pipe) pops, so the C2V/V2C TPUSH/TPOP stays implicit.

#include "ffn_grid_kernel_prelude_inl.hpp"

#ifdef __CCE_AICORE__
#include "ffn_grid_tile_aliases_inl.hpp"
using DownF32Tile = Tile<TileType::Vec, float, FFN_TOKEN_TILE, FFN_MODEL_TILE, BLayout::RowMajor>;
using FfnGridPipe = GridPipe<DownF32Tile, FFN_SLOT_BYTES, FFN_SLOT_COUNT>;

using ShapeTH = Shape<1, 1, 1, FFN_TOKEN_TILE, FFN_MODEL_TILE>;
using StrideTH = Stride<FFN_TOKEN_TILE * FFN_MODEL_TILE, FFN_TOKEN_TILE * FFN_MODEL_TILE,
                        FFN_TOKEN_TILE * FFN_MODEL_TILE, FFN_MODEL_TILE, 1>;
using GTHF32 = GlobalTensor<float, ShapeTH, StrideTH, Layout::ND>;

constexpr int kUbGateF32 = 0x0000;
constexpr int kUbUpF32 = 0x1000;
constexpr int kUbHiddenF32 = 0x2000;
constexpr int kUbHiddenF16 = 0x3000;
constexpr int kUbDownF32 = 0x4000;
constexpr int kUbAddF32 = 0x5000;

constexpr int kL1X = 0x00000;
constexpr int kL1Hidden = 0x10000;
constexpr int kL1WGate = 0x20000;
constexpr int kL1WUp = 0x24000;
constexpr int kL1WDown = 0x28000;

struct MixedVecContext {
    __gm__ uint8_t *reducePipeWindow;
    __gm__ uint8_t *yBlock;
    __gm__ uint8_t *hcclCtxRaw;
    int blockIdx;
    int gridRows;
    int gridCols;
    GatePipe *gatePipe;
    UpPipe *upPipe;
    HiddenPipe *hiddenPipe;
    DownPipe *downPipe;
};

AICORE void PushFfnHiddenToCube(HiddenPipe &hiddenPipe, HiddenF16Tile &hiddenF16)
{
#ifndef __PTO_AUTO__
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
    TMOV(hiddenPipe, hiddenF16);
}

AICORE void PushFfnEastPartial(FfnGridPipe &reducePipe, DownF32Tile &downF32, int col)
{
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
}

AICORE void StoreFfnReducedOutput(__gm__ uint8_t *yBlock, DownF32Tile &downF32)
{
#ifndef __PTO_AUTO__
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
    GTHF32 yG(reinterpret_cast<__gm__ float *>(yBlock));
    TSTORE(yG, downF32);
}

AICORE void RunMixedReduceWrite(MixedVecContext &ctx, DownF32Tile &downF32, DownF32Tile &addF32)
{
    FfnGridPipe reducePipe;
    GridShape shape{ctx.gridRows, ctx.gridCols};
    GridCoord coord{ctx.blockIdx / ctx.gridCols, ctx.blockIdx - (ctx.blockIdx / ctx.gridCols) * ctx.gridCols};
    __gm__ uint8_t *window = ctx.reducePipeWindow + ctx.blockIdx * FFN_GRID_WINDOW_BYTES;
    a2a3_grid::InitGridPipeFromWindow(reducePipe, shape, coord, window, reinterpret_cast<__gm__ void *>(ctx.hcclCtxRaw),
                                      /*pipeId=*/0);

    using pto::GridDirection;
    if (coord.col > 0) {
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

    if (coord.col + 1 < ctx.gridCols) {
        PushFfnEastPartial(reducePipe, downF32, coord.col);
    } else {
        StoreFfnReducedOutput(ctx.yBlock, downF32);
    }
}

AICORE void RunMixedVecStage(MixedVecContext &ctx)
{
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

    RunFfnVecActivation(gateF32, upF32, hiddenF32, hiddenF16, *ctx.gatePipe, *ctx.upPipe);
    PushFfnHiddenToCube(*ctx.hiddenPipe, hiddenF16);
    TMOV(downF32, *ctx.downPipe);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif
    RunMixedReduceWrite(ctx, downF32, addF32);
#ifndef __PTO_AUTO__
    pipe_barrier(PIPE_ALL);
#endif
    dsb(DSB_DDR);
}

#endif

#ifdef __CCE_AICORE__
AICORE void DistributedFfnGridMixedKernelImpl(__gm__ uint8_t *fftsAddr, __gm__ uint8_t *reducePipeWindow,
                                              __gm__ uint8_t *x, __gm__ uint8_t *wGate, __gm__ uint8_t *wUp,
                                              __gm__ uint8_t *wDown, __gm__ uint8_t *gatePartial,
                                              __gm__ uint8_t *upPartial, __gm__ uint8_t *hiddenIn,
                                              __gm__ uint8_t *downPartial, __gm__ uint8_t *yOutput,
                                              __gm__ uint8_t *hcclCtxRaw, int gridRows, int gridCols)
{
#include "oversize_bodies/distributed_ffn_grid_mixed_impl_body.inl"
}
#endif

__global__ AICORE void DistributedFfnGridMixedKernel(__gm__ uint8_t *fftsAddr, __gm__ uint8_t *reducePipeWindow,
                                                     __gm__ uint8_t *x, __gm__ uint8_t *wGate, __gm__ uint8_t *wUp,
                                                     __gm__ uint8_t *wDown, __gm__ uint8_t *gatePartial,
                                                     __gm__ uint8_t *upPartial, __gm__ uint8_t *hiddenIn,
                                                     __gm__ uint8_t *downPartial, __gm__ uint8_t *yOutput,
                                                     __gm__ uint8_t *hcclCtxRaw, int gridRows, int gridCols)
{
#ifdef __CCE_AICORE__
    DistributedFfnGridMixedKernelImpl(
        fftsAddr, reducePipeWindow, x, wGate, wUp, wDown, gatePartial, upPartial, hiddenIn, downPartial, yOutput,
        hcclCtxRaw, gridRows, gridCols);
#else
    FFN_GRID_UNUSED_KERNEL_ARGS(
        fftsAddr, reducePipeWindow, x, wGate, wUp, wDown, gatePartial, upPartial, hiddenIn, downPartial, yOutput,
        hcclCtxRaw, gridRows, gridCols);
#endif
}

void launchDistributedFfnGridMixedKernel(uint8_t *ffts, uint8_t *reducePipeWindow, uint8_t *x, uint8_t *wGate,
                                         uint8_t *wUp, uint8_t *wDown, uint8_t *gatePartial, uint8_t *upPartial,
                                         uint8_t *hiddenIn, uint8_t *downPartial, uint8_t *yOutput, uint8_t *hcclCtx,
                                         int gridRows, int gridCols, void *stream)
{
    int totalBlocks = gridRows * gridCols;
    if (totalBlocks <= 0) {
        return;
    }
    DistributedFfnGridMixedKernel<<<totalBlocks, nullptr, stream>>>(ffts, reducePipeWindow, x, wGate, wUp, wDown,
                                                                    gatePartial, upPartial, hiddenIn, downPartial,
                                                                    yOutput, hcclCtx, gridRows, gridCols);
}
