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
#include "acl/acl.h"

using namespace pto;

template <typename T, int kTRows_, int kTCols_, int kGRows_, int kGCols_>
__global__ AICORE void runTAndS( __gm__ T __out__ *out, __gm__ T __in__ *src0, T scalar) {
    using DynShapeDim5 = Shape<1, 1, 1, kGRows_, kGCols_>;
    using DynStridDim5 = pto::Stride<1, 1, 1, kGCols_, 1>;
    using GlobalData = GlobalTensor<T, DynShapeDim5, DynStridDim5>;
    using TileData = Tile<TileType::Vec, T, kTRows_, kTCols_, BLayout::RowMajor, -1, -1>;
    TileData src0Tile(kTRows_, kTCols_);
    TileData dstTile(kTRows_, kTCols_);
    TASSIGN(src0Tile, 0x0 + 0x400 * block_idx);

    int offset = (block_idx / 4) * (64 * 16) + (block_idx % 4) * 16;
    GlobalData src0Global(src0 + offset);
    GlobalData dstGlobal(out + offset);

    TLOAD(src0Tile, src0Global);
    TLOAD(src1Tile, src1Global);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TANDS(dstTile, src0Tile, scalar);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

template <typename T, int dstTileH, int dstTileW, int srcTileH, int srcTileW, int vRows, int vCols>
__global__ AICORE void runTAndS( __gm__ T __out__ *out, __gm__ T __in__ *src0, T scalar) {
    using DynShape = pto::Shape<-1, -1, -1, -1, -1>;
    using DynStride = pto::Stride<-1, -1, -1, -1, -1>;
    using GlobalData = GlobalTensor<T, DynShape, DynStride>;
    GlobalData dstGlobal(out, pto::Shape(1, 1, 1, vRows, vCols), pto::Stride(1, 1, 1, dstTileW, 1));
    GlobalData src0Global(src0, pto::Shape(1, 1, 1, vRows, vCols), pto::Stride(1, 1, 1, srcTileW, 1));

    using TileDataDst = Tile<TileType::Vec, T, dstTileH, dstTileW, BLayout::RowMajor, -1, -1>;
    using TileDataSrc = Tile<TileType::Vec, T, srcTileH, srcTileW, BLayout::RowMajor, -1, -1>;
    TileDataDst dstTile(vRows, vCols);
    TileDataSrc src0Tile(vRows, vCols);
    TASSIGN(src0Tile, 0x0);
    TASSIGN(dstTile, 0x20000);

    TLOAD(src0Tile, src0Global);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TANDS<TileDataDst, TileDataSrc>(dstTile, src0Tile, scalar);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

template <typename T, int dstTileH, int dstTileW, int srcTileH, int srcTileW, int vRows, int vCols, bool sameTile>
void LaunchTAndS(T *out, T *src, T scalar, void *stream)
{
    if constexpr (sameTile) {
        runTAndS<T, dstTileH, dstTileW, vRows, vCols><<<1, nullptr, stream>>>(out, src, scalar);
    } else {
        runTAndS<T, dstTileH, dstTileW, srcTileH, srcTileW, vRows, vCols><<<1, nullptr, stream>>>(out, src, scalar);
    }
}

template void LaunchTAndS<int32_t, 64, 64, 64, 64, 64, 64 true>
    (int32_t *out, int32_t *src, int32_t scalar, void *stream);
template void LaunchTAndS<int16_t, 64, 64, 64, 64, 64, 64 true>
    (int16_t *out, int16_t *src, int32_t scalar, void *stream);
template void LaunchTAndS<int16_t, 32, 128, 32, 128, 32, 256, false>
    (int16_t *out, int16_t *src, int32_t scalar, void *stream);
template void LaunchTAndS<int32_t, 16, 32, 16, 64, 16, 32, false>
    (int32_t *out, int32_t *src, int32_t scalar, void *stream);
template void LaunchTAndS<int16_t, 32, 128, 32, 128, 32, 256, false>
    (int16_t *out, int16_t *src, int32_t scalar, void *stream);
template void LaunchTAndS<int32_t, 16, 32, 16, 64, 16, 32, false>
    (int32_t *out, int32_t *src, int32_t scalar, void *stream);
