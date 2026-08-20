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

#include <cstddef>
#include <cstdint>
#include <pto/pto-inst.hpp>

#include <pto/common/fifo.hpp>
#include <pto/npu/a2a3/grid_intrinsic.hpp>
#include <pto/npu/a2a3/grid_pipe_runtime.hpp>

#include "common.hpp"
#include "ffn_config.hpp"
#include "ffn_grid_compute_common_inl.hpp"
#include "gridpipe_payload_inl.hpp"
#include "tpipe_tmov_inl.hpp"

#ifdef __CCE_AICORE__
using namespace pto;

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
