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
#include <acl/acl.h>

using namespace pto;

constexpr int ITERATIONS = 10;

// TADDS 1D kernel - demonstrates RV_VLDS + RVECSU issue
template <typename T, int tileH, int tileW, int vRows, int vCols>
__global__ AICORE void runTADDS(__gm__ T *out, __gm__ T *src, T scalar)
{
    using DynShape = Shape<1, 1, 1, -1, -1>;
    using DynStride = pto::Stride<1, 1, -1, -1, 1>;
    using GlobalData = GlobalTensor<T, DynShape, DynStride>;
    using TileData = Tile<TileType::Vec, T, tileH, tileW, BLayout::RowMajor, -1, -1>;

    GlobalData srcGlobal(src, DynShape(vRows, vCols), DynStride(tileH, tileW));
    GlobalData dstGlobal(out, DynShape(vRows, vCols), DynStride(tileH, tileW));
    TileData srcTile(vRows, vCols);
    TileData dstTile(vRows, vCols);

    TASSIGN(srcTile, 0x0);
    TASSIGN(dstTile, 0x10000);

    for (int i = 0; i < ITERATIONS; i++) {
        TLOAD(srcTile, srcGlobal);
        TADDS(dstTile, srcTile, scalar);
        TSTORE(dstGlobal, dstTile);
        pipe_barrier(PIPE_ALL);
    }
}

// TADD 1D kernel - baseline with RV_VLDI + RV_VLOOP
template <typename T, int tileH, int tileW, int vRows, int vCols>
__global__ AICORE void runTADD(__gm__ T *out, __gm__ T *src0, __gm__ T *src1)
{
    using DynShape = Shape<1, 1, 1, -1, -1>;
    using DynStride = pto::Stride<1, 1, -1, -1, 1>;
    using GlobalData = GlobalTensor<T, DynShape, DynStride>;
    using TileData = Tile<TileType::Vec, T, tileH, tileW, BLayout::RowMajor, -1, -1>;

    GlobalData src0Global(src0, DynShape(vRows, vCols), DynStride(tileH, tileW));
    GlobalData src1Global(src1, DynShape(vRows, vCols), DynStride(tileH, tileW));
    GlobalData dstGlobal(out, DynShape(vRows, vCols), DynStride(tileH, tileW));
    TileData src0Tile(vRows, vCols);
    TileData src1Tile(vRows, vCols);
    TileData dstTile(vRows, vCols);

    TASSIGN(src0Tile, 0x0);
    TASSIGN(src1Tile, 0x4000);
    TASSIGN(dstTile, 0x10000);

    for (int i = 0; i < ITERATIONS; i++) {
        TLOAD(src0Tile, src0Global);
        TLOAD(src1Tile, src1Global);
        TADD(dstTile, src0Tile, src1Tile);
        TSTORE(dstGlobal, dstTile);
        pipe_barrier(PIPE_ALL);
    }
}

// Launchers
template <typename T, int tileH, int tileW, int vRows, int vCols>
void launchTADDS(void *out, void *src, float scalar, aclrtStream stream)
{
    runTADDS<T, tileH, tileW, vRows, vCols><<<1, nullptr, stream>>>((T*)out, (T*)src, (T)scalar);
}

template <typename T, int tileH, int tileW, int vRows, int vCols>
void launchTADD(void *out, void *src0, void *src1, aclrtStream stream)
{
    runTADD<T, tileH, tileW, vRows, vCols><<<1, nullptr, stream>>>((T*)out, (T*)src0, (T*)src1);
}

// Explicit instantiations - 1x4096 float (1D case)
template void launchTADDS<float, 1, 4096, 1, 4096>(void*, void*, float, aclrtStream);
template void launchTADD<float, 1, 4096, 1, 4096>(void*, void*, void*, aclrtStream);
