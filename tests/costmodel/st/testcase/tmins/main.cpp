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

#include "cost_check.hpp"

using namespace pto;

namespace {

template <typename T, int rows, int cols, float profiling, float accuracy>
void runTMins(T scalar)
{
    using TileData = Tile<TileType::Vec, T, rows, cols, BLayout::RowMajor, -1, -1>;
    TileData srcTile(rows, cols);
    TileData dstTile(rows, cols);
    TASSIGN(srcTile, 0x0 + 0x400);
    TASSIGN(dstTile, 0x8000 + 0x400);

    TMINS(dstTile, srcTile, scalar);

    EXPECT_CYCLE_NEAR(profiling, accuracy);
}

} // namespace

TEST(TMins, float_64x64) { runTMins<float, 64, 64, 68.0f, 0.94f>(0.0f); }
TEST(TMins, int32_64x64) { runTMins<int32_t, 64, 64, 69.0f, 0.94f>(1); }
TEST(TMins, int16_64x64) { runTMins<int16_t, 64, 64, 76.0f, 0.94f>(1); }
TEST(TMins, half_64x64) { runTMins<half, 64, 64, 74.0f, 0.94f>(half{0.0f}); }
TEST(TMins, half_16x256) { runTMins<half, 16, 256, 65.0f, 0.94f>(half{1.0f}); }
