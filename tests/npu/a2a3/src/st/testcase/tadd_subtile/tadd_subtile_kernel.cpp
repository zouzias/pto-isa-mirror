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
#include "pto/npu/a2a3/subtile/TAdd.hpp"
#include "acl/acl.h"

using namespace pto;

template <typename TileData, typename T, int vRows, int vCols, bool Is1D>
__tf__ AICORE void DoTAddSubtile(TileData &dstTile, TileData &src0Tile, TileData &src1Tile)
{
    __ubuf__ T *dstPtr = (__ubuf__ T *)__cce_get_tile_ptr(dstTile.data());
    __ubuf__ T *src0Ptr = (__ubuf__ T *)__cce_get_tile_ptr(src0Tile.data());
    __ubuf__ T *src1Ptr = (__ubuf__ T *)__cce_get_tile_ptr(src1Tile.data());

    if constexpr (Is1D) {
        Subtile1D<T> dstS(dstPtr, vRows * vCols);
        Subtile1D<T> src0S(src0Ptr, vRows * vCols);
        Subtile1D<T> src1S(src1Ptr, vRows * vCols);
        TADD_SUBTILE_IMPL_1D(dstS, src0S, src1S);
    } else {
        Subtile2D<T> dstS(dstPtr, vRows, vCols, TileData::RowStride);
        Subtile2D<T> src0S(src0Ptr, vRows, vCols, TileData::RowStride);
        Subtile2D<T> src1S(src1Ptr, vRows, vCols, TileData::RowStride);
        TADD_SUBTILE_IMPL_2D(dstS, src0S, src1S);
    }
}

template <typename T, int kTRows_, int kTCols_, int vRows, int vCols, bool Is1D>
__global__ AICORE void runTAddSubtile(__gm__ T __out__ *out, __gm__ T __in__ *src0, __gm__ T __in__ *src1)
{
    using DynShapeDim5 = Shape<1, 1, 1, vRows, vCols>;
    using DynStridDim5 = Stride<1, 1, 1, kTCols_, 1>;
    using GlobalData = GlobalTensor<T, DynShapeDim5, DynStridDim5>;
    using TileData = Tile<TileType::Vec, T, kTRows_, kTCols_, BLayout::RowMajor, -1, -1>;

    TileData src0Tile(vRows, vCols);
    TileData src1Tile(vRows, vCols);
    TileData dstTile(vRows, vCols);
    TASSIGN(src0Tile, 0x0);
    TASSIGN(src1Tile, 0x10000);
    TASSIGN(dstTile, 0x20000);

    GlobalData src0Global(src0);
    GlobalData src1Global(src1);
    GlobalData dstGlobal(out);

    TLOAD(src0Tile, src0Global);
    TLOAD(src1Tile, src1Global);

    DoTAddSubtile<TileData, T, vRows, vCols, Is1D>(dstTile, src0Tile, src1Tile);

    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

template <typename T, int kTRows_, int kTCols_, int vRows, int vCols, bool Is1D>
void LaunchTAddSubtile(T *out, T *src0, T *src1, void *stream)
{
    if constexpr (std::is_same_v<T, aclFloat16>)
        runTAddSubtile<half, kTRows_, kTCols_, vRows, vCols, Is1D>
            <<<1, nullptr, stream>>>((half *)(out), (half *)(src0), (half *)(src1));
    else
        runTAddSubtile<T, kTRows_, kTCols_, vRows, vCols, Is1D><<<1, nullptr, stream>>>(out, src0, src1);
}

// 1D counter mode (len = vRows * vCols)
template void LaunchTAddSubtile<float, 1, 1024, 1, 1024, true>(float *out, float *src0, float *src1, void *stream);
// 2D repeat+mask
template void LaunchTAddSubtile<float, 128, 64, 128, 64, false>(float *out, float *src0, float *src1, void *stream);

