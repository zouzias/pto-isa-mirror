/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <pto/pto-inst.hpp>
#include <pto/common/constants.hpp>
#include <pto/common/debug.h>
#include "acl/acl.h"

using namespace pto;

// NCHW -> NC1HWC0
template <typename T, int dstN, int dstC1, int dstH, int dstW, int dstC0,
          int gWholeShape0, int gWholeShape1, int gWholeShape2, int gWholeShape3, int gWholeShape4>
__global__ AICORE void runTTRANSConv1(__gm__ T __out__ *out, __gm__ T __in__ *src)
{
    constexpr int bufferSize = dstN * dstC1 * dstH * dstW * dstC0 * sizeof(T);
    constexpr int validRow = dstN * dstC1 * dstH * dstW;
    constexpr int validCol = dstC0;
    static_assert(gWholeShape0 == 1, "");
    static_assert(gWholeShape1 == dstN, "");
    static_assert(gWholeShape2 == dstC0 * dstC1, "");
    static_assert(gWholeShape3 == dstH, "");
    static_assert(gWholeShape4 == dstW, "");

    constexpr int gmRows = gWholeShape0 * gWholeShape1 * gWholeShape2 * gWholeShape3;
    constexpr int gmCols = gWholeShape4;

    using ShapeDim5 = Shape<1, 1, 1, gmRows, gmCols>;
    using StrideDim5 = Stride<gmRows * gmCols, gmRows * gmCols, gmRows * gmCols, gmCols, 1>;
    using GlobalDataIn = GlobalTensor<T, ShapeDim5, StrideDim5>;

    using SrcTileData = Tile<TileType::Vec, T, gmRows, gmCols, BLayout::RowMajor, gmRows, gmCols>;
    SrcTileData src0Tile;
    TASSIGN(src0Tile, 0x0);
    using TileData = ConvTile<TileType::Vec, T, bufferSize, Layout::NCHW, ConvTileShape<dstN, dstC0*dstC1, dstH, dstW>>;
    TileData srcTile;
    static_assert(srcTile.totalDimCount == 4);
    TASSIGN(srcTile, 0x0);

    using DstTileData = ConvTile<TileType::Vec, T, bufferSize, Layout::NC1HWC0, ConvTileShape<dstN, dstC1, dstH, dstW, dstC0>>;
    DstTileData dstTile;
    static_assert(dstTile.totalDimCount == 5);
    TASSIGN(dstTile, 0x0 + dstN * dstC1 * dstH * dstW * dstC0 * sizeof(T));
    SrcTileData dst0Tile;
    TASSIGN(dst0Tile, 0x0 + dstN * dstC1 * dstH * dstW * dstC0 * sizeof(T));

    DstTileData tmpTile;
    TASSIGN(tmpTile, 0x0 + dstN * dstC1 * dstH * dstW * dstC0 * sizeof(T) * 2);

    GlobalDataIn srcGlobal(src);
    GlobalDataIn dstGlobal(out);
    TLOAD(src0Tile, srcGlobal);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TTRANS(dstTile, srcTile, tmpTile);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(dstGlobal, dst0Tile);
}

template <typename T, int dstC1, int dstH, int dstW, int dstN, int dstC0,
          int srcN, int srcC1, int srcH, int srcW, int srcC0>
__global__ AICORE void runTTRANSConv2(__gm__ T __out__ *out, __gm__ T __in__ *src)
{
    static_assert(srcC0 == dstC0);
    static_assert(srcW == dstW);
    static_assert(srcH == dstH);
    static_assert(srcN == dstN);

    constexpr int bufferSize = dstN * dstC1 * dstH * dstW * dstC0 * sizeof(T);
    constexpr int validRow = dstN * dstC1 * dstH * dstW;
    constexpr int validCol = dstC0;

    constexpr int gmRows = dstN * dstC1 * dstH * dstW;
    constexpr int gmCols = dstC0;

    using ShapeDim5 = Shape<1, 1, 1, gmRows, gmCols>;
    using StrideDim5 = Stride<gmRows * gmCols, gmRows * gmCols, gmRows * gmCols, gmCols, 1>;
    using GlobalDataIn = GlobalTensor<T, ShapeDim5, StrideDim5>;

    using SrcTileData = Tile<TileType::Vec, T, gmRows, gmCols, BLayout::RowMajor, gmRows, gmCols>;
    SrcTileData src0Tile;
    TASSIGN(src0Tile, 0x0);
    using TileData = ConvTile<TileType::Vec, T, bufferSize, Layout::NC1HWC0, ConvTileShape<dstN, dstC1, dstH, dstW, dstC0>>;
    TileData srcTile;
    static_assert(srcTile.totalDimCount == 5);
    TASSIGN(srcTile, 0x0);

    using DstTileData = ConvTile<TileType::Vec, T, bufferSize, Layout::C1HWNC0, ConvTileShape<dstC1, dstH, dstW, dstN, dstC0>>;
    DstTileData dstTile;
    static_assert(dstTile.totalDimCount == 5);
    SrcTileData dst0Tile;
    TASSIGN(dstTile, 0x0 + dstN * dstC1 * dstH * dstW * dstC0 * sizeof(T));
    TASSIGN(dst0Tile, 0x0 + dstN * dstC1 * dstH * dstW * dstC0 * sizeof(T));

    DstTileData tmpTile;
    TASSIGN(tmpTile, 0x0 + dstN * dstC1 * dstH * dstW * dstC0 * sizeof(T) * 2);

    GlobalDataIn srcGlobal(src);
    GlobalDataIn dstGlobal(out);
    TLOAD(src0Tile, srcGlobal);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TTRANS(dstTile, srcTile, tmpTile);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(dstGlobal, dst0Tile);
}

template <typename T, int format, int gShape0, int gShape1, int gShape2, int gShape3, int gShape4,
          int gWholeShape0, int gWholeShape1, int gWholeShape2, int gWholeShape3, int gWholeShape4>
void LaunchTTRANSConv(T *out, T *src, void *stream)
{
    if constexpr (std::is_same_v<T, aclFloat16>) {
        if constexpr (format == 0) {
            runTTRANSConv1<half, gShape0, gShape1, gShape2, gShape3, gShape4,
                 gWholeShape0, gWholeShape1, gWholeShape2,
                 gWholeShape3, gWholeShape4><<<1, nullptr, stream>>>((half *)(out), (half *)(src));
        } else {
            runTTRANSConv2<half, gShape0, gShape1, gShape2, gShape3, gShape4,
                 gWholeShape0, gWholeShape1, gWholeShape2,
                 gWholeShape3, gWholeShape4><<<1, nullptr, stream>>>((half *)(out),
                 (half *)(src));
        }
    } else {
        if constexpr (format == 0) {
            runTTRANSConv1<T, gShape0, gShape1, gShape2, gShape3, gShape4,
                 gWholeShape0, gWholeShape1, gWholeShape2,
                 gWholeShape3, gWholeShape4><<<1, nullptr, stream>>>(out, src);
        } else {
            runTTRANSConv2<T, gShape0, gShape1, gShape2, gShape3, gShape4,
                 gWholeShape0, gWholeShape1, gWholeShape2,
                 gWholeShape3, gWholeShape4><<<1, nullptr, stream>>>(out, src);
        }
    }
}

template void LaunchTTRANSConv<float, 0, 1, 4, 6, 56, 8, 1, 1, 32, 6, 56>(float *out, float *src, void *stream);
template void LaunchTTRANSConv<float, 0, 1, 2, 2, 16, 16, 1, 1, 32, 2, 16>(float *out, float *src, void *stream);
template void LaunchTTRANSConv<float, 1, 2, 2, 16, 2, 16, 2, 2, 2, 16, 16>(float *out, float *src, void *stream);