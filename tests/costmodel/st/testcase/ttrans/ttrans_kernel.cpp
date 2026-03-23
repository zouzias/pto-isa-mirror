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
#include <gtest/gtest.h>
#include <cmath>

using namespace pto;

// TTRANS cycle formula (non-conv RowMajor, B16/B32 path):
//   blockSizeElem = 32 / sizeof(T)     (16 for half, 8 for float)
//   yTileSizeElem = 16
//   numSubTileX   = ceil(validCol / blockSizeElem)
//   numSubTileY   = validRow / 16
//   remainY       = validRow % 16
//   Stats: [scatter_vnchwconv(numSubTileY)] x numSubTileX  (full tiles)
//          [scatter_vnchwconv(numSubTileX)] x 1            (Y-tail, if remainY > 0)
//          PIPE_V + copy_ubuf_to_ubuf (0 cycles)
//   Total = startup(14) + (numSubTileX * ceil_Y) * per_repeat(2)
//
// Test cases:
//   half,  16 x 16: numSubTileX=1, ceil_Y=1 → 14 + 1*1*2 = 16
//   half,  16 x 32: numSubTileX=2, ceil_Y=1 → 14 + 2*1*2 = 18
//   float, 16 x 16: numSubTileX=2, ceil_Y=1 → 14 + 2*1*2 = 18

template <typename T, int kSrcRows_, int kSrcCols_, float profiling, float accuracy>
AICORE void runTTrans()
{
    // For TTRANS: src is [kSrcRows_ x kSrcCols_], dst is [kSrcCols_ x kSrcRows_] (transposed)
    using SrcTile = Tile<TileType::Vec, T, kSrcRows_, kSrcCols_, BLayout::RowMajor, -1, -1>;
    using DstTile = Tile<TileType::Vec, T, kSrcCols_, kSrcRows_, BLayout::RowMajor, -1, -1>;
    // Tmp tile must be large enough to hold transposed data; use same shape as dst for simplicity
    using TmpTile = Tile<TileType::Vec, T, kSrcCols_, kSrcRows_, BLayout::RowMajor, -1, -1>;

    SrcTile srcTile(kSrcRows_, kSrcCols_);
    DstTile dstTile(kSrcCols_, kSrcRows_);
    TmpTile tmpTile(kSrcCols_, kSrcRows_);

    TASSIGN(srcTile, 0x0);
    TASSIGN(dstTile, 0x4000);
    TASSIGN(tmpTile, 0x8000);

    TTRANS(dstTile, srcTile, tmpTile);

    float costResult = dstTile.GetCycle();
    float precision = 1 - fabs(profiling - costResult) / profiling;
    bool ret = precision >= accuracy;
    EXPECT_TRUE(ret);
}

template <typename T, int kSrcRows_, int kSrcCols_, float profiling, float accuracy>
void LaunchTTrans(void *stream)
{
    if constexpr (std::is_same_v<T, aclFloat16>)
        runTTrans<half, kSrcRows_, kSrcCols_, profiling, accuracy>();
    else
        runTTrans<T, kSrcRows_, kSrcCols_, profiling, accuracy>();
}

// half, 16x16: numSubTileX=1, ceil_Y=1 → 14 + 1*2 = 16
template void LaunchTTrans<aclFloat16, 16, 16, 16.0f, 1.0f>(void *stream);
// half, 16x32: numSubTileX=2, ceil_Y=1 → 14 + 2*2 = 18
template void LaunchTTrans<aclFloat16, 16, 32, 18.0f, 1.0f>(void *stream);
// float, 16x16: blockSizeElem=8, numSubTileX=2, ceil_Y=1 → 14 + 2*2 = 18
template void LaunchTTrans<float, 16, 16, 18.0f, 1.0f>(void *stream);
