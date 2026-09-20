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

template <typename T, int N, int C, int H, int W, int VN>
__global__ AICORE void runTTRANSConv_NCHW2NC1HWC0_Valid(__gm__ T __out__* out, __gm__ T __in__* src)
{
    constexpr size_t DTypeSize = GetTypeSize<T>();
    constexpr size_t C0 = 32 / DTypeSize;
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
    TASSIGN(dst0Tile, 0x0 + srcElemNum * DTypeSize);

    using SrcPhysShape = ConvTileShape<N, C, H, W>;
    using SrcValidShape = ConvTileValidShape<VN, C, H, W>;
    using SrcConvTile =
        ConvTile<TileType::Vec, T, srcElemNum * DTypeSize, Layout::NCHW, SrcPhysShape, SrcValidShape>;

    using DstPhysShape = ConvTileShape<N, C1, H, W, C0>;
    using DstConvTile = ConvTile<TileType::Vec, T, dstElemNum * DTypeSize, Layout::NC1HWC0, DstPhysShape>;

    SrcConvTile srcTile;
    DstConvTile dstTile;
    static_assert(srcTile.totalDimCount == 4);
    static_assert(dstTile.totalDimCount == 5);

    srcTile.data() = src0Tile.data();
    dstTile.data() = dst0Tile.data();

    std::fill((uint8_t*)dstTile.data(), (uint8_t*)dstTile.data() + dstElemNum * DTypeSize, 0);

    using TmpTileData = Tile<TileType::Vec, T, 1, 32, BLayout::RowMajor, 1, 32>;
    TmpTileData tmpTile;
    TASSIGN(tmpTile, 0x0 + (srcElemNum + dstElemNum) * DTypeSize);

    SrcGlobalData srcGlobal(src);
    DstGlobalData dstGlobal(out);

    TLOAD(src0Tile, srcGlobal);
    TTRANS(dstTile, srcTile, tmpTile);
    TSTORE(dstGlobal, dst0Tile);
}

template <typename T, int N, int C, int H, int W>
__global__ AICORE void runTTRANSConv_NCHW2NC1HWC0_Default(__gm__ T __out__* out, __gm__ T __in__* src)
{
    constexpr size_t DTypeSize = GetTypeSize<T>();
    constexpr size_t C0 = 32 / DTypeSize;
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
    TASSIGN(dst0Tile, 0x0 + srcElemNum * DTypeSize);

    using SrcConvTile = ConvTile<TileType::Vec, T, srcElemNum * DTypeSize, Layout::NCHW, ConvTileShape<N, C, H, W>>;
    using DstConvTile =
        ConvTile<TileType::Vec, T, dstElemNum * DTypeSize, Layout::NC1HWC0, ConvTileShape<N, C1, H, W, C0>>;

    SrcConvTile srcTile;
    DstConvTile dstTile;
    static_assert(srcTile.totalDimCount == 4);
    static_assert(dstTile.totalDimCount == 5);

    srcTile.data() = src0Tile.data();
    dstTile.data() = dst0Tile.data();

    std::fill((uint8_t*)dstTile.data(), (uint8_t*)dstTile.data() + dstElemNum * DTypeSize, 0);

    using TmpTileData = Tile<TileType::Vec, T, 1, 32, BLayout::RowMajor, 1, 32>;
    TmpTileData tmpTile;
    TASSIGN(tmpTile, 0x0 + (srcElemNum + dstElemNum) * DTypeSize);

    SrcGlobalData srcGlobal(src);
    DstGlobalData dstGlobal(out);

    TLOAD(src0Tile, srcGlobal);
    TTRANS(dstTile, srcTile, tmpTile);
    TSTORE(dstGlobal, dst0Tile);
}

template <typename T, int N, int C, int H, int W, int VN>
void LaunchTTRANSConvValid(T* out, T* src, void* stream)
{
    runTTRANSConv_NCHW2NC1HWC0_Valid<T, N, C, H, W, VN>(out, src);
}

template <typename T, int N, int C, int H, int W>
void LaunchTTRANSConvDefault(T* out, T* src, void* stream)
{
    runTTRANSConv_NCHW2NC1HWC0_Default<T, N, C, H, W>(out, src);
}

template void LaunchTTRANSConvValid<half, 4, 4, 4, 4, 2>(half* out, half* src, void* stream);
template void LaunchTTRANSConvValid<float, 4, 4, 4, 4, 2>(float* out, float* src, void* stream);

template void LaunchTTRANSConvDefault<half, 4, 4, 4, 4>(half* out, half* src, void* stream);
template void LaunchTTRANSConvDefault<float, 4, 4, 4, 4>(float* out, float* src, void* stream);
