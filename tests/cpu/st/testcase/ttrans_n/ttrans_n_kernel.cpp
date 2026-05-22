/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <cstdlib>
#include <pto/pto-inst.hpp>
#include <pto/common/constants.hpp>
#include <pto/common/debug.h>
#include <pto/common/pto_tile.hpp>

using namespace pto;

// =========================================================================
// MODE 1: NCHW -> NC1HWC0 (Format ID: 1)
// =========================================================================
template <typename T, int N, int C, int H, int W>
__global__ AICORE void runTTRANSConv_NCHW2NC1HWC0(__gm__ T __out__ *out, __gm__ T __in__ *src)
{
    constexpr int elemNum = N * C * H * W;
    constexpr size_t C0 = 32 / sizeof(T);
    constexpr size_t C1 = (C + C0 - 1) / C0;
    constexpr size_t dstElemNum = N * C1 * H * W * C0;

    using SrcShapeDim5 = Shape<1, 1, 1, 1, elemNum>;
    using SrcStrideDim5 = pto::Stride<elemNum, elemNum, elemNum, elemNum, 1>;
    using SrcGlobalData = GlobalTensor<T, SrcShapeDim5, SrcStrideDim5>;

    using DstShapeDim5 = Shape<1, 1, 1, 1, dstElemNum>;
    using DstStrideDim5 = pto::Stride<dstElemNum, dstElemNum, dstElemNum, dstElemNum, 1>;
    using DstGlobalData = GlobalTensor<T, DstShapeDim5, DstStrideDim5>;

    using SrcTileData = Tile<TileType::Vec, T, 1, elemNum, BLayout::RowMajor, 1, elemNum>;
    SrcTileData src0Tile;
    using SrcConvTile =
        ConvTile<TileType::Vec, T, elemNum * sizeof(T), Layout::NCHW, ConvTileShape<N, C, H, W>>;
    SrcConvTile srcTile;
    static_assert(srcTile.totalDimCount == 4);
    TASSIGN(src0Tile, 0x0);
    srcTile.data() = src0Tile.data();

    using DstConvTile =
        ConvTile<TileType::Vec, T, dstElemNum * sizeof(T), Layout::NC1HWC0, ConvTileShape<N, C1, H, W, C0>>;
    DstConvTile dstTile;
    static_assert(dstTile.totalDimCount == 5);
    using DstTileData = Tile<TileType::Vec, T, 1, dstElemNum, BLayout::RowMajor, 1, dstElemNum>;
    DstTileData dst0Tile;
    TASSIGN(dst0Tile, 0x0 + elemNum * sizeof(T));
    dstTile.data() = dst0Tile.data();

    constexpr int tmpTileH = H * W;
    constexpr unsigned yTileSizeElem = (sizeof(T) == 1) ? 32 : 16;
    constexpr int tmpTileW = (C0 + yTileSizeElem - 1) / yTileSizeElem * yTileSizeElem;
    using TmpTileData = Tile<TileType::Vec, T, tmpTileH, tmpTileW, BLayout::RowMajor, tmpTileH, tmpTileW>;
    TmpTileData tmpTile;
    TASSIGN(tmpTile, 0x0 + (elemNum + dstElemNum) *sizeof(T));

    SrcGlobalData srcGlobal(src);
    DstGlobalData dstGlobal(out);
    TLOAD(src0Tile, srcGlobal);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
    TTRANS(dstTile, srcTile, tmpTile);
    set_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(dstGlobal, dst0Tile);
}


// =========================================================================
// Launch dispatcher for non-grouped transforms (format 1 or 2)
// =========================================================================
template <typename T, int gShape0, int gShape1, int gShape2, int gShape3>
void LaunchTTRANSConv_NCHW2NC1HWC0(T *out, T *src, void *stream)
{
    runTTRANSConv_NCHW2NC1HWC0<T, gShape0, gShape1, gShape2, gShape3>(out, src);
}


// =========================================================================
// Explicit template instantiations (Updated to match exact linker rules)
// =========================================================================

// Non-grouped configurations (LaunchTTRANSConv)
template void LaunchTTRANSConv_NCHW2NC1HWC0<float, 5, 4, 3, 8>(float *out, float *src, void *stream);
template void LaunchTTRANSConv_NCHW2NC1HWC0<int32_t, 5, 14, 13, 16>(int32_t *out, int32_t *src, void *stream);
template void LaunchTTRANSConv_NCHW2NC1HWC0<uint16_t, 1, 11, 13, 16>(uint16_t *out, uint16_t *src, void *stream);
template void LaunchTTRANSConv_NCHW2NC1HWC0<int32_t, 4, 32, 3, 7>(int32_t *out, int32_t *src, void *stream);
template void LaunchTTRANSConv_NCHW2NC1HWC0<int8_t, 4, 32, 3, 7>(int8_t *out, int8_t *src, void *stream);
