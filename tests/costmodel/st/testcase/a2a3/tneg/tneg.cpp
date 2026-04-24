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
#include <cstdint>
#include <vector>

#include "cost_check.hpp"

using namespace pto;

namespace {

template <typename T, int row, int validRow, int col, int validCol, float profiling, float accuracy>
void runTNeg()
{
    using TileData = Tile<TileType::Vec, T, row, col, BLayout::RowMajor, -1, -1>;
    TileData srcTile(validRow, validCol);
    TileData dstTile(validRow, validCol);

    std::vector<T> srcBuf(row * col, T{1});
    std::vector<T> dstBuf(row * col, T{0});
    TASSIGN(srcTile, reinterpret_cast<std::uintptr_t>(srcBuf.data()));
    TASSIGN(dstTile, reinterpret_cast<std::uintptr_t>(dstBuf.data()));

    TNEG(dstTile, srcTile);

    EXPECT_CYCLE_NEAR(profiling, accuracy);
}

} // namespace

TEST(TNeg, case_float_64x64_64x64)
{
    runTNeg<float, 64, 64, 64, 64, 90.0f, 0.0f>();
}

TEST(TNeg, case_int32_32x32_32x32)
{
    runTNeg<int32_t, 32, 32, 32, 32, 40.0f, 0.0f>();
}

TEST(TNeg, case_half_32x64_32x64)
{
    runTNeg<half, 32, 32, 64, 64, 42.0f, 0.0f>();
}

TEST(TNeg, case_int16_64x16_64x16)
{
    runTNeg<int16_t, 64, 64, 16, 16, 32.0f, 0.0f>();
}
