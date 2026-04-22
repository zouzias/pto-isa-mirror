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
#include <gtest/gtest.h>
#include <pto/common/constants.hpp>
#include <cstdint>
#include <vector>

#include "cost_check.hpp"

using namespace pto;

namespace {

template <typename T, int dstRow, int dstCol, int srcRow, int validRow, int srcCol, int validCol, float profiling,
          float accuracy>
void runTMaxS(T scalar)
{
    using DstTile = Tile<TileType::Vec, T, dstRow, dstCol, BLayout::RowMajor, -1, -1>;
    using SrcTile = Tile<TileType::Vec, T, srcRow, srcCol, BLayout::RowMajor, -1, -1>;
    DstTile dstTile(validRow, validCol);
    SrcTile srcTile(validRow, validCol);

    std::vector<T> srcBuf(srcRow * srcCol, T{1});
    std::vector<T> dstBuf(dstRow * dstCol, T{0});
    TASSIGN(srcTile, reinterpret_cast<std::uintptr_t>(srcBuf.data()));
    TASSIGN(dstTile, reinterpret_cast<std::uintptr_t>(dstBuf.data()));

    TMAXS(dstTile, srcTile, scalar);

    EXPECT_CYCLE_NEAR(profiling, accuracy);
}

} // namespace

TEST(TMaxS, case1)
{
    runTMaxS<float, 32, 64, 32, 32, 64, 64, NO_PROFILING, 0.0f>(1.0f);
}
TEST(TMaxS, case2)
{
    runTMaxS<half, 63, 64, 63, 63, 64, 64, NO_PROFILING, 0.0f>((half)1.0f);
}
TEST(TMaxS, case3)
{
    runTMaxS<int32_t, 31, 128, 31, 31, 128, 128, NO_PROFILING, 0.0f>(1);
}
TEST(TMaxS, case4)
{
    runTMaxS<int16_t, 15, 192, 15, 15, 192, 192, NO_PROFILING, 0.0f>(1);
}
TEST(TMaxS, case5)
{
    runTMaxS<float, 7, 448, 7, 7, 448, 448, NO_PROFILING, 0.0f>(1.0f);
}
TEST(TMaxS, case6)
{
    runTMaxS<float, 256, 16, 256, 256, 16, 16, NO_PROFILING, 0.0f>(1.0f);
}
TEST(TMaxS, case7)
{
    runTMaxS<float, 32, 128, 32, 32, 64, 64, NO_PROFILING, 0.0f>(1.0f);
}
TEST(TMaxS, case8)
{
    runTMaxS<half, 63, 128, 63, 63, 64, 64, NO_PROFILING, 0.0f>((half)1.0f);
}
TEST(TMaxS, case9)
{
    runTMaxS<int32_t, 31, 256, 31, 31, 128, 128, NO_PROFILING, 0.0f>(1);
}
TEST(TMaxS, case10)
{
    runTMaxS<int16_t, 15, 192, 15, 15, 192, 192, NO_PROFILING, 0.0f>(1);
}
TEST(TMaxS, case11)
{
    runTMaxS<float, 7, 512, 7, 7, 448, 448, NO_PROFILING, 0.0f>(1.0f);
}
TEST(TMaxS, case12)
{
    runTMaxS<float, 256, 32, 256, 256, 16, 16, NO_PROFILING, 0.0f>(1.0f);
}
