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

// TSORT32 cycle formula (N rows, R=validCol/32 repeats per row):
//   Row 0:   startup(14) + R*per_repeat(2)
//   Row k>0: interval(18) + R*per_repeat(2)
//   Total:   14 + R*2 + (N-1)*(18 + R*2)
//
// 4x32 tile: R=1, Total = (14+2) + 3*(18+2) = 16 + 60 = 76
// 1x32 tile: R=1, Total = 16

template <typename T, int kTRows_, int kTCols_, float profiling, float accuracy>
AICORE void runTSort32()
{
    // dst stride for half is 4x src (values + indices interleaved)
    constexpr int dstCols = kTCols_ * 4;
    using SrcTile = Tile<TileType::Vec, T, kTRows_, kTCols_, BLayout::RowMajor, -1, -1>;
    using DstTile = Tile<TileType::Vec, T, kTRows_, dstCols, BLayout::RowMajor, -1, -1>;
    using IdxTile = Tile<TileType::Vec, uint32_t, kTRows_, kTCols_, BLayout::RowMajor, -1, -1>;

    DstTile dstTile(kTRows_, dstCols);
    SrcTile srcTile(kTRows_, kTCols_);
    IdxTile idxTile(kTRows_, kTCols_);

    TASSIGN(srcTile, 0x0);
    TASSIGN(dstTile, 0x4000);
    TASSIGN(idxTile, 0x8000);

    TSORT32(dstTile, srcTile, idxTile);

    float costResult = dstTile.GetCycle();
    float precision = 1 - fabs(profiling - costResult) / profiling;
    bool ret = precision >= accuracy;
    EXPECT_TRUE(ret);
}

template <typename T, int kTRows_, int kTCols_, float profiling, float accuracy>
void LaunchTSort32(void *stream)
{
    if constexpr (std::is_same_v<T, aclFloat16>)
        runTSort32<half, kTRows_, kTCols_, profiling, accuracy>();
    else
        runTSort32<T, kTRows_, kTCols_, profiling, accuracy>();
}

// 4x32, R=1: 16 + 3*20 = 76
template void LaunchTSort32<aclFloat16, 4, 32, 76.0f, 1.0f>(void *stream);
// 1x32, R=1: 16
template void LaunchTSort32<aclFloat16, 1, 32, 16.0f, 1.0f>(void *stream);
// 4x32, FP32, R=1: 76
template void LaunchTSort32<float, 4, 32, 76.0f, 1.0f>(void *stream);
