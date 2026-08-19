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

// Single-device multi-block FFN mixed Cube/Vec kernel, AllGather split variant.
//
// Each block owns one logical grid cell (row, col).  Cube computes the three
// GEMMs, Vec computes the activation and row-local hidden AllGather.  Down
// projection is sharded by output H columns, so there is no post-down reduce.
// Cube<->Vec C2V/V2C traffic is issued as directional TMOV (see
// tpipe_tmov_inl.hpp): TMOV(pipe, tile) pushes and TMOV(tile, pipe) pops, so the
// TPUSH/TPOP stays implicit.  The GridPipe EAST/WEST gather keeps its own
// neighbor-cell TPUSH/TPOP.

#include "ffn_grid_kernel_prelude_inl.hpp"

#ifdef __CCE_AICORE__
using pto::BLayout;
using pto::Direction;
using pto::GlobalTensor;
using pto::GridCoord;
using pto::GridDirection;
using pto::GridPipe;
using pto::GridShape;
using pto::Layout;
using pto::SLayout;
using pto::Shape;
using pto::Stride;
using pto::Tile;
using pto::TileAcc;
using pto::TileLeft;
using pto::TileRight;
using pto::TileType;
using pto::TPipe;

using GateF32Tile = Tile<TileType::Vec, float, FFN_TOKEN_TILE, FFN_FFN_TILE, BLayout::RowMajor>;
using UpF32Tile = GateF32Tile;
using HiddenF32Tile = GateF32Tile;
using HiddenF16Tile = Tile<TileType::Vec, half, FFN_TOKEN_TILE, FFN_FFN_TILE, BLayout::RowMajor>;
using HiddenFullF16Tile = Tile<TileType::Vec, half, FFN_TOKEN_TILE, FFN_FFN_TOTAL_TILE, BLayout::RowMajor>;
using DownF32Tile = Tile<TileType::Vec, float, FFN_TOKEN_TILE, FFN_MODEL_SHARD_TILE, BLayout::RowMajor>;
using FfnGatherPipe = GridPipe<HiddenF16Tile, FFN_SLOT_BYTES, FFN_SLOT_COUNT>;

using ShapeTHShard = Shape<1, 1, 1, FFN_TOKEN_TILE, FFN_MODEL_SHARD_TILE>;
using StrideTHShard = Stride<FFN_TOKEN_TILE * FFN_MODEL_TILE, FFN_TOKEN_TILE * FFN_MODEL_TILE,
                             FFN_TOKEN_TILE * FFN_MODEL_TILE, FFN_MODEL_TILE, 1>;
using GTHShardF32 = GlobalTensor<float, ShapeTHShard, StrideTHShard, Layout::ND>;

constexpr int AlignUp(int value, int align)
{
    return ((value + align - 1) / align) * align;
}

constexpr int kUbAlignBytes = 0x1000;
constexpr int kUbGateF32 = 0x0000;
constexpr int kUbUpF32 = AlignUp(kUbGateF32 + FFN_GATE_PARTIAL_BYTES, kUbAlignBytes);
constexpr int kUbHiddenF32 = AlignUp(kUbUpF32 + FFN_UP_PARTIAL_BYTES, kUbAlignBytes);
constexpr int kUbHiddenF16 = AlignUp(kUbHiddenF32 + FFN_GATE_PARTIAL_BYTES, kUbAlignBytes);
constexpr int kUbHiddenFullF16 = AlignUp(kUbHiddenF16 + FFN_HIDDEN_BYTES, kUbAlignBytes);
constexpr int kUbEastCarryF16 = AlignUp(kUbHiddenFullF16 + FFN_HIDDEN_FULL_BYTES, kUbAlignBytes);
constexpr int kUbWestCarryF16 = AlignUp(kUbEastCarryF16 + FFN_HIDDEN_BYTES, kUbAlignBytes);
// Reuse the first carry scratch for the later down shard.  Both carry tiles are
// dead before DownPipe writes downF32.
constexpr int kUbDownF32 = kUbEastCarryF16;

constexpr int kL1X = 0x00000;
constexpr int kL1Hidden = 0x10000;
constexpr int kL1WGate = 0x20000;
constexpr int kL1WUp = 0x24000;
constexpr int kL1WDown = 0x28000;

struct AllGatherVecContext {
    __gm__ uint8_t *gatherPipeWindow;
    __gm__ uint8_t *yBlock;
    __gm__ uint8_t *hcclCtxRaw;
    int blockIdx;
    int row;
    int col;
    int gridRows;
    int gridCols;
    GatePipe *gatePipe;
    UpPipe *upPipe;
    HiddenPipe *hiddenPipe;
    DownPipe *downPipe;
};

struct GatherExchangeState {
    FfnGatherPipe *pipe;
    HiddenF16Tile *hiddenF16;
    HiddenF16Tile *eastCarryF16;
    HiddenF16Tile *westCarryF16;
    HiddenFullF16Tile *hiddenFullF16;
    int col;
    int gridCols;
    bool haveEastCarry;
    bool haveWestCarry;
};

AICORE void PushGatherCarry(GatherExchangeState &state, int step)
{
    if (state.col + 1 < state.gridCols && state.haveEastCarry) {
#ifndef __PTO_AUTO__
        set_flag(step == 0 ? PIPE_V : PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        wait_flag(step == 0 ? PIPE_V : PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
#endif
        TPUSH<GridDirection::EAST>(*state.pipe, step == 0 ? *state.hiddenF16 : *state.eastCarryF16);
    }
    if (state.col > 0 && state.haveWestCarry) {
#ifndef __PTO_AUTO__
        set_flag(step == 0 ? PIPE_V : PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        wait_flag(step == 0 ? PIPE_V : PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
#endif
        TPUSH<GridDirection::WEST>(*state.pipe, step == 0 ? *state.hiddenF16 : *state.westCarryF16);
    }
}

AICORE void PopGatherEast(GatherExchangeState &state, int step)
{
    if (state.col <= step) {
        state.haveEastCarry = false;
        return;
    }
    TPOP<GridDirection::EAST>(*state.pipe, *state.eastCarryF16);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif
    int srcCol = state.col - step - 1;
    TINSERT(*state.hiddenFullF16, *state.eastCarryF16, 0, static_cast<uint16_t>(srcCol * FFN_FFN_TILE));
#ifndef __PTO_AUTO__
    pipe_barrier(PIPE_V);
#endif
    state.haveEastCarry = true;
}

AICORE void PopGatherWest(GatherExchangeState &state, int step)
{
    if (state.col + step + 1 >= state.gridCols) {
        state.haveWestCarry = false;
        return;
    }
    TPOP<GridDirection::WEST>(*state.pipe, *state.westCarryF16);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif
    int srcCol = state.col + step + 1;
    TINSERT(*state.hiddenFullF16, *state.westCarryF16, 0, static_cast<uint16_t>(srcCol * FFN_FFN_TILE));
#ifndef __PTO_AUTO__
    pipe_barrier(PIPE_V);
#endif
    state.haveWestCarry = true;
}

AICORE void RunAllGatherExchange(GatherExchangeState &state)
{
    for (int step = 0; step < state.gridCols - 1; ++step) {
        PushGatherCarry(state, step);
        PopGatherEast(state, step);
        PopGatherWest(state, step);
    }
}

AICORE void StoreAllGatherOutput(
    HiddenPipe &hiddenPipe, DownPipe &downPipe, __gm__ uint8_t *yBlock, HiddenFullF16Tile &hiddenFullF16,
    DownF32Tile &downF32)
{
#ifndef __PTO_AUTO__
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
    TMOV(hiddenPipe, hiddenFullF16);
    TMOV(downF32, downPipe);

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

AICORE void RunAllGatherVecStage(AllGatherVecContext &ctx)
{
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

    RunFfnVecActivation(gateF32, upF32, hiddenF32, hiddenF16, *ctx.gatePipe, *ctx.upPipe);
#ifndef __PTO_AUTO__
    pipe_barrier(PIPE_V);
#endif
    FfnGatherPipe gatherPipe;
    GridShape shape{ctx.gridRows, ctx.gridCols};
    GridCoord coord{ctx.row, ctx.col};
    __gm__ uint8_t *window = ctx.gatherPipeWindow + ctx.blockIdx * FFN_GRID_WINDOW_BYTES;
    a2a3_grid::InitGridPipeFromWindow(gatherPipe, shape, coord, window, reinterpret_cast<__gm__ void *>(ctx.hcclCtxRaw),
                                      /*pipeId=*/0);
    TINSERT(hiddenFullF16, hiddenF16, 0, static_cast<uint16_t>(ctx.col * FFN_FFN_TILE));
#ifndef __PTO_AUTO__
    pipe_barrier(PIPE_V);
#endif
    GatherExchangeState exchange{&gatherPipe, &hiddenF16, &eastCarryF16, &westCarryF16, &hiddenFullF16,
                                 ctx.col,     ctx.gridCols, true,          true};
    RunAllGatherExchange(exchange);
    StoreAllGatherOutput(*ctx.hiddenPipe, *ctx.downPipe, ctx.yBlock, hiddenFullF16, downF32);
}

#endif

#ifdef __CCE_AICORE__
AICORE void DistributedFfnGridAllGatherMixedKernelImpl(__gm__ uint8_t *fftsAddr,
                                                       __gm__ uint8_t *gatherPipeWindow, __gm__ uint8_t *x,
                                                       __gm__ uint8_t *wGate, __gm__ uint8_t *wUp,
                                                       __gm__ uint8_t *wDown, __gm__ uint8_t *gatePartial,
                                                       __gm__ uint8_t *upPartial, __gm__ uint8_t *hiddenIn,
                                                       __gm__ uint8_t *downPartial, __gm__ uint8_t *yOutput,
                                                       __gm__ uint8_t *hcclCtxRaw, int gridRows, int gridCols)
{
#include "oversize_bodies/distributed_ffn_grid_allgather_impl_body.inl"
}
#endif

__global__ AICORE void DistributedFfnGridAllGatherMixedKernel(__gm__ uint8_t *fftsAddr,
                                                              __gm__ uint8_t *gatherPipeWindow, __gm__ uint8_t *x,
                                                              __gm__ uint8_t *wGate, __gm__ uint8_t *wUp,
                                                              __gm__ uint8_t *wDown, __gm__ uint8_t *gatePartial,
                                                              __gm__ uint8_t *upPartial, __gm__ uint8_t *hiddenIn,
                                                              __gm__ uint8_t *downPartial, __gm__ uint8_t *yOutput,
                                                              __gm__ uint8_t *hcclCtxRaw, int gridRows, int gridCols)
{
#ifdef __CCE_AICORE__
    DistributedFfnGridAllGatherMixedKernelImpl(
        fftsAddr, gatherPipeWindow, x, wGate, wUp, wDown, gatePartial, upPartial, hiddenIn, downPartial, yOutput,
        hcclCtxRaw, gridRows, gridCols);
#else
    FFN_GRID_UNUSED_KERNEL_ARGS(
        fftsAddr, gatherPipeWindow, x, wGate, wUp, wDown, gatePartial, upPartial, hiddenIn, downPartial, yOutput,
        hcclCtxRaw, gridRows, gridCols);
#endif
}

void launchDistributedFfnGridAllGatherMixedKernel(uint8_t *ffts, uint8_t *gatherPipeWindow, uint8_t *x, uint8_t *wGate,
                                                  uint8_t *wUp, uint8_t *wDown, uint8_t *gatePartial,
                                                  uint8_t *upPartial, uint8_t *hiddenIn, uint8_t *downPartial,
                                                  uint8_t *yOutput, uint8_t *hcclCtx, int gridRows, int gridCols,
                                                  void *stream)
{
    int totalBlocks = gridRows * gridCols;
    if (totalBlocks <= 0) {
        return;
    }
    DistributedFfnGridAllGatherMixedKernel<<<totalBlocks, nullptr, stream>>>(
        ffts, gatherPipeWindow, x, wGate, wUp, wDown, gatePartial, upPartial, hiddenIn, downPartial, yOutput, hcclCtx,
        gridRows, gridCols);
}
