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
    constexpr size_t C0 = 32 / sizeof(T);
    constexpr size_t C1 = (C + C0 - 1) / C0;
    constexpr size_t dstElemNum = N * C1 * H * W * C0;
    constexpr size_t srcElemNum = N * C * H * W;

    using SrcShapeDim5 = Shape<1, 1, 1, 1, srcElemNum>;
    using SrcStrideDim5 = pto::Stride<srcElemNum, srcElemNum, srcElemNum, srcElemNum, 1>;
    using SrcGlobalData = GlobalTensor<T, SrcShapeDim5, SrcStrideDim5>;

    using DstShapeDim5 = Shape<1, 1, 1, 1, dstElemNum>;
    using DstStrideDim5 = pto::Stride<dstElemNum, dstElemNum, dstElemNum, dstElemNum, 1>;
    using DstGlobalData = GlobalTensor<T, DstShapeDim5, DstStrideDim5>;

    using SrcTileData = Tile<TileType::Vec, T, 1, srcElemNum, BLayout::RowMajor, 1, srcElemNum>;
    using DstTileData = Tile<TileType::Vec, T, 1, dstElemNum, BLayout::RowMajor, 1, dstElemNum>;

    SrcTileData src0Tile;
    DstTileData dst0Tile;

    TASSIGN(src0Tile, 0x0);
    TASSIGN(dst0Tile, 0x0 + srcElemNum * sizeof(T));

    using SrcConvTile = ConvTile<TileType::Vec, T, srcElemNum * sizeof(T), Layout::NCHW, ConvTileShape<N, C, H, W>>;
    using DstConvTile =
        ConvTile<TileType::Vec, T, dstElemNum * sizeof(T), Layout::NC1HWC0, ConvTileShape<N, C1, H, W, C0>>;

    SrcConvTile srcTile;
    DstConvTile dstTile;
    static_assert(srcTile.totalDimCount == 4);
    static_assert(dstTile.totalDimCount == 5);

    srcTile.data() = src0Tile.data();
    dstTile.data() = dst0Tile.data();

    // Not used internally, just for placeholder
    using TmpTileData = Tile<TileType::Vec, T, 1, 32, BLayout::RowMajor, 1, 32>;
    TmpTileData tmpTile;
    TASSIGN(tmpTile, 0x0 + (srcElemNum + dstElemNum) * sizeof(T));

    SrcGlobalData srcGlobal(src);
    DstGlobalData dstGlobal(out);

    TLOAD(src0Tile, srcGlobal);
    TTRANS(dstTile, srcTile, tmpTile);
    TSTORE(dstGlobal, dst0Tile);
}

template <typename T, int srcN, int srcC1, int srcH, int srcW, int srcC0, int dstC1, int dstH, int dstW, int dstN1,
          int dstN0, int dstC0>
__global__ AICORE void runTTRANSConv_NC1HWC02C1HWN1N0C0(__gm__ T __out__ *out, __gm__ T __in__ *src)
{
    static_assert(srcC0 == dstC0);
    static_assert(srcW == dstW);
    static_assert(srcH == dstH);
    static_assert(dstN1 == (srcN + dstN0 - 1) / dstN0);

    constexpr int validRow = dstN1 * dstN0 * dstC1 * dstH * dstW;
    constexpr int validCol = dstC0;

    constexpr int srcElemNum = srcN * srcC1 * srcH * srcW * srcC0;
    constexpr int srcBufferSize = srcElemNum * sizeof(T);

    constexpr int dstElemNum = dstN1 * dstN0 * dstC1 * dstH * dstW * dstC0;
    constexpr int dstBufferSize = dstElemNum * sizeof(T);

    using SrcShapeDim5 = Shape<1, 1, 1, 1, srcElemNum>;
    using SrcStrideDim5 = pto::Stride<srcElemNum, srcElemNum, srcElemNum, srcElemNum, 1>;
    using SrcGlobalData = GlobalTensor<T, SrcShapeDim5, SrcStrideDim5>;

    using DstShapeDim5 = Shape<1, 1, 1, 1, dstElemNum>;
    using DstStrideDim5 = pto::Stride<dstElemNum, dstElemNum, dstElemNum, dstElemNum, 1>;
    using DstGlobalData = GlobalTensor<T, DstShapeDim5, DstShapeDim5>;

    using SrcTileData = Tile<TileType::Vec, T, 1, srcElemNum, BLayout::RowMajor, 1, srcElemNum>;
    using DstTileData = Tile<TileType::Vec, T, 1, dstElemNum, BLayout::RowMajor, 1, dstElemNum>;

    SrcTileData src0Tile;
    DstTileData dst0Tile;

    TASSIGN(src0Tile, 0x0);
    TASSIGN(dst0Tile, 0x0 + srcBufferSize);

    using SrcConvTile =
        ConvTile<TileType::Vec, T, srcBufferSize, Layout::NC1HWC0, ConvTileShape<srcN, srcC1, srcH, srcW, srcC0>>;
    using DstConvTile = ConvTile<TileType::Vec, T, dstBufferSize, Layout::FRACTAL_Z,
                                 ConvTileShape<dstC1 * dstH * dstW, dstN1, dstN0, dstC0>>;
    SrcConvTile srcTile;
    DstConvTile dstTile;
    static_assert(srcTile.totalDimCount == 5);
    static_assert(dstTile.totalDimCount == 4);

    srcTile.data() = src0Tile.data();
    dstTile.data() = dst0Tile.data();

    // Not used internally, just for placeholder
    using TmpTileData = Tile<TileType::Vec, T, 1, 32, BLayout::RowMajor, 1, 32>;
    TmpTileData tmpTile;
    TASSIGN(tmpTile, 0x0 + srcBufferSize + dstBufferSize);

    SrcGlobalData srcGlobal(src);
    DstGlobalData dstGlobal(out);

    TLOAD(src0Tile, srcGlobal);
    TTRANS(dstTile, srcTile, tmpTile);
    TSTORE(dstGlobal, dst0Tile);
}

template <typename T, int format, int srcShape0, int srcShape1, int srcShape2, int srcShape3, int srcShape4,
          int dstShape0, int dstShape1, int dstShape2, int dstShape3, int dstShape4, int dstShape5>
void LaunchTTRANSConv(T *out, T *src, void *stream)
{
    if constexpr (format == 0) {
        runTTRANSConv_NCHW2NC1HWC0<T, srcShape0, srcShape1, srcShape2, srcShape3>(out, src);
    } else if constexpr (format == 0) {
        runTTRANSConv_NC1HWC02C1HWN1N0C0<T, srcShape0, srcShape1, srcShape2, srcShape3, srcShape4, dstShape0, dstShape1,
                                         dstShape2, dstShape3, dstShape4, dstShape5>(out, src);
    }
}

template void LaunchTTRANSConv<float, 0, 5, 4, 3, 8, 1, 5, 1, 3, 8, 8, 1>(float *out, float *src, void *stream);
template void LaunchTTRANSConv<int32_t, 0, 5, 14, 13, 16, 1, 5, 2, 13, 16, 8, 1>(int32_t *out, int32_t *src,
                                                                                 void *stream);
template void LaunchTTRANSConv<uint16_t, 0, 1, 11, 13, 16, 1, 1, 1, 13, 16, 16, 1>(uint16_t *out, uint16_t *src,
                                                                                   void *stream);
template void LaunchTTRANSConv<int32_t, 0, 4, 32, 3, 7, 1, 4, 4, 3, 7, 8, 1>(int32_t *out, int32_t *src, void *stream);
template void LaunchTTRANSConv<int8_t, 0, 4, 32, 3, 7, 1, 4, 1, 3, 7, 32, 1>(int8_t *out, int8_t *src, void *stream);


template void LaunchTTRANSConv<float, 1, 25, 4, 3, 8, 8, 4, 3, 8, 2, 16, 8>(float *out, float *src, void *stream);
template void LaunchTTRANSConv<int32_t, 1, 15, 14, 13, 16, 8, 14, 13, 16, 2, 8, 8>(int32_t *out, int32_t *src,
                                                                                   void *stream);
template void LaunchTTRANSConv<uint16_t, 1, 11, 11, 13, 16, 16, 11, 13, 16, 2, 8, 16>(uint16_t *out, uint16_t *src,
                                                                                      void *stream);
template void LaunchTTRANSConv<int32_t, 1, 4, 32, 3, 7, 8, 32, 3, 7, 1, 4, 8>(int32_t *out, int32_t *src, void *stream);
template void LaunchTTRANSConv<int8_t, 1, 4, 32, 3, 7, 32, 32, 3, 7, 1, 8, 32>(int8_t *out, int8_t *src, void *stream);
