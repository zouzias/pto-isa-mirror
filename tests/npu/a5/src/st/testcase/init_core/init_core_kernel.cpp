/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include "pto/pto-inst.hpp"
#include "acl/acl.h"

using namespace pto;

template <
    typename T, int dstTileH, int dstTileW, int src0TileH, int src0TileW, int src1TileH, int src1TileW, int vRows,
    int vCols>
__global__ AICORE void runInitCore(__gm__ T __out__* out, __gm__ T __in__* src0, __gm__ T __in__* src1)
{
    INIT_CORE();

    using DynShape = pto::Shape<-1, -1, -1, -1, -1>;
    using DynStride = pto::Stride<-1, -1, -1, -1, -1>;
    using GlobalData = GlobalTensor<T, DynShape, DynStride>;
    GlobalData dstGlobal(
        out, pto::Shape(1, 1, 1, vRows, vCols),
        pto::Stride(dstTileH * dstTileW, dstTileH * dstTileW, dstTileH * dstTileW, dstTileW, 1));
    GlobalData src0Global(
        src0, pto::Shape(1, 1, 1, vRows, vCols),
        pto::Stride(src0TileH * src0TileW, src0TileH * src0TileW, src0TileH * src0TileW, src0TileW, 1));
    GlobalData src1Global(
        src1, pto::Shape(1, 1, 1, vRows, vCols),
        pto::Stride(src1TileH * src1TileW, src1TileH * src1TileW, src1TileH * src1TileW, src1TileW, 1));

    using TileDataDst = Tile<TileType::Vec, T, dstTileH, dstTileW, BLayout::RowMajor, -1, -1>;
    using TileDataSrc0 = Tile<TileType::Vec, T, src0TileH, src0TileW, BLayout::RowMajor, -1, -1>;
    using TileDataSrc1 = Tile<TileType::Vec, T, src1TileH, src1TileW, BLayout::RowMajor, -1, -1>;
    TileDataDst dstTile(vRows, vCols);
    TileDataSrc0 src0Tile(vRows, vCols);
    TileDataSrc1 src1Tile(vRows, vCols);
    TASSIGN(src0Tile, 0x0);
    TASSIGN(src1Tile, 0x10000);
    TASSIGN(dstTile, 0x20000);

    Event<Op::TLOAD, Op::TADD> evt0 = TLOAD(src0Tile, src0Global);
    Event<Op::TLOAD, Op::TADD> evt1 = TLOAD(src1Tile, src1Global);
    Event<Op::TADD, Op::TSTORE_VEC> evt2 = TADD(dstTile, src0Tile, src1Tile, evt0, evt1);
    TSTORE(dstGlobal, dstTile, evt2);
}

template <
    typename T, int dstTileH, int dstTileW, int src0TileH, int src0TileW, int src1TileH, int src1TileW, int vRows,
    int vCols>
void LaunchInitCore(T* out, T* src0, T* src1, void* stream)
{
    runInitCore<T, dstTileH, dstTileW, src0TileH, src0TileW, src1TileH, src1TileW, vRows, vCols>
        <<<1, nullptr, stream>>>(out, src0, src1);
}

template void LaunchInitCore<float, 64, 64, 64, 64, 64, 64, 64, 64>(float* out, float* src0, float* src1, void* stream);
