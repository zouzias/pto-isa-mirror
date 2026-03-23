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

// TMRGSORT (single-src) cycle formula:
//   repeatTimes = srcCol / (blockLen * 4)
//   Stats: [vmrgsort4(repeatTimes)]
//   Total: startup(14) + repeatTimes * per_repeat(2)
//
// srcCol=256, blockLen=64: repeatTimes=1, Total = 14 + 2 = 16

template <typename T, int kTCols_, uint32_t blockLen, float profiling, float accuracy>
AICORE void runTMrgSort()
{
    using DstTile = Tile<TileType::Vec, T, 1, kTCols_, BLayout::RowMajor, -1, -1>;
    using SrcTile = Tile<TileType::Vec, T, 1, kTCols_, BLayout::RowMajor, -1, -1>;

    DstTile dstTile(1, kTCols_);
    SrcTile srcTile(1, kTCols_);

    TASSIGN(srcTile, 0x0);
    TASSIGN(dstTile, 0x4000);

    TMRGSORT(dstTile, srcTile, blockLen);

    float costResult = dstTile.GetCycle();
    float precision = 1 - fabs(profiling - costResult) / profiling;
    bool ret = precision >= accuracy;
    EXPECT_TRUE(ret);
}

template <typename T, int kTCols_, uint32_t blockLen, float profiling, float accuracy>
void LaunchTMrgSort(void *stream)
{
    if constexpr (std::is_same_v<T, aclFloat16>)
        runTMrgSort<half, kTCols_, blockLen, profiling, accuracy>();
    else
        runTMrgSort<T, kTCols_, blockLen, profiling, accuracy>();
}

// srcCol=256, blockLen=64, repeatTimes=1: startup(14) + 1*2 = 16
template void LaunchTMrgSort<aclFloat16, 256, 64, 16.0f, 1.0f>(void *stream);
// srcCol=512, blockLen=64, repeatTimes=2: startup(14) + 2*2 = 18
template void LaunchTMrgSort<aclFloat16, 512, 64, 18.0f, 1.0f>(void *stream);
