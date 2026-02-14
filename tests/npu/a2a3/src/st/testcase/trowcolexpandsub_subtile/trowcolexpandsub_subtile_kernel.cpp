/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <pto/pto-inst.hpp>
#include <pto/common/constants.hpp>
#include <pto/npu/a2a3/subtile/subtile_tile.hpp>
#include <pto/npu/a2a3/subtile/subtile_brcb.hpp>
#include <pto/npu/a2a3/subtile/TSub.hpp>
#include "acl/acl.h"

using namespace pto;

// __tf__ helpers (tile ptr access must be in __tf__)

template <typename TileData, typename TileDataSrc1, typename T, int vRows, int vCols>
__tf__ AICORE void DoRowExpandSubSubtile(TileData &dstTile, TileData &src0Tile, TileData &tmpTile, TileDataSrc1 &src1Tile)
{
    __ubuf__ T *src0Ptr = (__ubuf__ T *)__cce_get_tile_ptr(src0Tile.data());
    __ubuf__ T *dstPtr = (__ubuf__ T *)__cce_get_tile_ptr(dstTile.data());
    __ubuf__ T *tmpPtr = (__ubuf__ T *)__cce_get_tile_ptr(tmpTile.data());
    __ubuf__ T *src1Ptr = (__ubuf__ T *)__cce_get_tile_ptr(src1Tile.data());

    // rowStride=0 -> infer contiguous (cols)
    Subtile2D<T> dstS(dstPtr, vRows, vCols, 0);
    Subtile2D<T> src0S(src0Ptr, vRows, vCols, 0);
    Subtile2D<T> tmpS(tmpPtr, vRows, vCols, 0);
    Subtile1D<T> src1S(src1Ptr, vRows);

    SubtileBrcb(tmpS, src1S);
    TSUB_SUBTILE_IMPL_2D(dstS, src0S, tmpS);
}

template <typename TileData, typename TileDataSrc1, typename T, int vRows, int vCols>
__tf__ AICORE void DoColExpandSubSubtile(TileData &dstTile, TileData &src0Tile, TileDataSrc1 &src1Tile)
{
    __ubuf__ T *src0Ptr = (__ubuf__ T *)__cce_get_tile_ptr(src0Tile.data());
    __ubuf__ T *dstPtr = (__ubuf__ T *)__cce_get_tile_ptr(dstTile.data());
    __ubuf__ T *src1Ptr = (__ubuf__ T *)__cce_get_tile_ptr(src1Tile.data());

    for (int r = 0; r < vRows; ++r) {
        Subtile1D<T> dstRow(dstPtr + r * TileData::RowStride, vCols);
        Subtile1D<T> src0Row(src0Ptr + r * TileData::RowStride, vCols);
        Subtile1D<T> src1Row(src1Ptr, vCols);
        TSUB_SUBTILE_IMPL_1D(dstRow, src0Row, src1Row);
    }
}

// Row-expand subtract using SubtileBrcb + Subtile2D

template <typename T, int kTRows_, int kTCols_, int vRows, int vCols>
__global__ AICORE void runTRowExpandSubSubtile(__gm__ T __out__ *out, __gm__ T __in__ *src0, __gm__ T __in__ *src1)
{
    using DynShapeDim5 = Shape<1, 1, 1, vRows, vCols>;
    using DynStrideDim5 = Stride<1, 1, 1, kTCols_, 1>;
    using GlobalData = GlobalTensor<T, DynShapeDim5, DynStrideDim5>;
    using TileData = Tile<TileType::Vec, T, kTRows_, kTCols_, BLayout::RowMajor, -1, -1>;
    using GlobalDataSrc1 = GlobalTensor<T, Shape<1, 1, 1, vRows, 1>, Stride<1, 1, 1, 1, 1>, Layout::DN>;
    using TileDataSrc1 = Tile<TileType::Vec, T, kTRows_, 1, BLayout::ColMajor, -1, -1>;

    TileData src0Tile(vRows, vCols);
    TileData dstTile(vRows, vCols);
    TileData tmpTile(vRows, vCols);
    TileDataSrc1 src1Tile(vRows, 1);

    TASSIGN(src0Tile, 0x0);
    TASSIGN(dstTile, 0x10000);
    TASSIGN(tmpTile, 0x20000);
    TASSIGN(src1Tile, 0x30000);

    GlobalData src0Global(src0);
    GlobalData dstGlobal(out);
    GlobalDataSrc1 src1Global(src1);

    TLOAD(src0Tile, src0Global);
    TLOAD(src1Tile, src1Global);

    pipe_barrier(PIPE_ALL);
    DoRowExpandSubSubtile<TileData, TileDataSrc1, T, vRows, vCols>(dstTile, src0Tile, tmpTile, src1Tile);
    pipe_barrier(PIPE_ALL);

    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

// Col-expand subtract using per-row Subtile1D loop

template <typename T, int kTRows_, int kTCols_, int vRows, int vCols>
__global__ AICORE void runTColExpandSubSubtile(__gm__ T __out__ *out, __gm__ T __in__ *src0, __gm__ T __in__ *src1)
{
    using DynShapeDim5 = Shape<1, 1, 1, vRows, vCols>;
    using DynStrideDim5 = Stride<1, 1, 1, kTCols_, 1>;
    using GlobalData = GlobalTensor<T, DynShapeDim5, DynStrideDim5>;
    using TileData = Tile<TileType::Vec, T, kTRows_, kTCols_, BLayout::RowMajor, -1, -1>;
    using GlobalDataSrc1 = GlobalTensor<T, Shape<1, 1, 1, 1, vCols>, Stride<1, 1, 1, vCols, 1>>;
    using TileDataSrc1 = Tile<TileType::Vec, T, 1, kTCols_, BLayout::RowMajor, -1, -1>;

    TileData src0Tile(vRows, vCols);
    TileData dstTile(vRows, vCols);
    TileDataSrc1 src1Tile(1, vCols);

    TASSIGN(src0Tile, 0x0);
    TASSIGN(dstTile, 0x10000);
    TASSIGN(src1Tile, 0x20000);

    GlobalData src0Global(src0);
    GlobalData dstGlobal(out);
    GlobalDataSrc1 src1Global(src1);

    TLOAD(src0Tile, src0Global);
    TLOAD(src1Tile, src1Global);

    pipe_barrier(PIPE_ALL);
    DoColExpandSubSubtile<TileData, TileDataSrc1, T, vRows, vCols>(dstTile, src0Tile, src1Tile);
    pipe_barrier(PIPE_ALL);

    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

template <typename T, int kTRows_, int kTCols_, int vRows, int vCols>
void LaunchTRowExpandSubSubtile(T *out, T *src0, T *src1, void *stream)
{
    runTRowExpandSubSubtile<T, kTRows_, kTCols_, vRows, vCols><<<1, nullptr, stream>>>(out, src0, src1);
}

template <typename T, int kTRows_, int kTCols_, int vRows, int vCols>
void LaunchTColExpandSubSubtile(T *out, T *src0, T *src1, void *stream)
{
    runTColExpandSubSubtile<T, kTRows_, kTCols_, vRows, vCols><<<1, nullptr, stream>>>(out, src0, src1);
}

// Row-expand: rows multiple of 8, cols == 32B/sizeof(float) == 8
template void LaunchTRowExpandSubSubtile<float, 128, 8, 128, 8>(float *out, float *src0, float *src1, void *stream);
// Col-expand: broadcast row vector across rows
template void LaunchTColExpandSubSubtile<float, 64, 64, 64, 64>(float *out, float *src0, float *src1, void *stream);

