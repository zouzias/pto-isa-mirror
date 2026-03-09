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
#include "acl/acl.h"

using namespace pto;

template <typename T, int dstTileH, int dstTileW, int srcTileH, int srcTileW, int vRows, int vCols>
__global__ AICORE void runTRowCMax(__gm__ T __out__ *out, __gm__ T __in__ *src)
{
    using DynShape = pto::Shape<-1, -1, -1, -1, -1>;
    using DynStride = pto::Stride<-1, -1, -1, -1, -1>;
    using GlobalData = GlobalTensor<T, DynShape, DynStride>;
    GlobalData dstGlobal(out, pto::Shape(1, 1, 1, 1, vCols),
                         pto::Stride(dstTileH * dstTileW, dstTileH * dstTileW, dstTileH * dstTileW, dstTileW, 1));
    GlobalData srcGlobal(src, pto::Shape(1, 1, 1, vRows, vCols),
                         pto::Stride(srcTileH * srcTileW, srcTileH * srcTileW, srcTileH * srcTileW, srcTileW, 1));
    using TileDataDst = Tile<TileType::Vec, T, dstTileH, dstTileW, BLayout::RowMajor, -1, -1>;
    using TileDataSrc = Tile<TileType::Vec, T, srcTileH, srcTileW, BLayout::RowMajor, -1, -1>;
    TileDataDst dstTile(vRows, vCols);
    TileDataSrc srcTile(vRows, vCols);
    TileDataDst tmpTile(vRows, vCols);

    size_t dstSize = sizeof(T) * dstTileH * dstTileW;
    size_t srcSize = sizeof(T) * srcTileH * srcTileW;
    size_t srcOffset = dstSize;
    size_t tmpOffset = srcOffset + srcSize;
    TASSIGN(dstTile, 0x0);
    TASSIGN(srcTile, srcOffset);
    TASSIGN(tmpTile, tmpOffset);

    TLOAD(srcTile, srcGlobal);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TXORS(dstTile, srcTile, scalar, tmpTile);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

template <typename T, int dstTileH, int dstTileW, int srcTileH, int srcTileW, int vRows, int vCols>
void LaunchTRowCMax(T *out, T *src, void *stream)
{
    runTRowCMax<T, dstTileH, dstTileW, srcTileH, srcTileW, vRows, vCols><<<1, nullptr, stream>>>(out, src);
}

template <int dstTileH, int dstTileW, int srcTileH, int srcTileW, int vRows, int vCols>
void LaunchTRowCMaxHalf(aclFloat16 *out, aclFloat16 *src, void *stream)
{
    runTRowCMax<half, dstTileH, dstTileW, srcTileH, srcTileW, vRows, vCols><<<1, nullptr, stream>>>((half *)out, (half *)src);
}

template void LaunchTRowCMax<float, 1, 8, 8, 8, 8, 8>(float *out, float *src, void *stream);
