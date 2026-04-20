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

// simplest test case
template <typename T, int shape3, int shape4, float profiling, float accuracy>
void runTLoad()
{
    using TileData = Tile<TileType::Vec, T, shape3, shape4>;
    TileData vecTile;
    TASSIGN(vecTile, 0x0);

    constexpr int stride0 = 1 * 1 * shape3 * shape4;
    constexpr int stride1 = 1 * shape3 * shape4;
    constexpr int stride2 = shape3 * shape4;
    using StaticShapeDim5 = Shape<1, 1, 1, shape3, shape4>;
    using StaticStrideDim5 = pto::Stride<stride0, stride1, stride2, shape4, 1>;
    using GlobalData = GlobalTensor<T, StaticShapeDim5, StaticStrideDim5, Layout::ND>;
    GlobalData srcGlobal(0);
    TLOAD(vecTile, srcGlobal);

    EXPECT_CYCLE_NEAR(profiling, accuracy);
}

} // namespace

TEST(TLoad, c01_nd_float_128x128)
{
    runTLoad<float, 128, 128, NO_PROFILING, 0.0f>();
}

TEST(TLoad, c01_nd_float_64x128)
{
    runTLoad<float, 64, 128, NO_PROFILING, 0.0f>();
}

TEST(TLoad, c01_nd_float_128x256)
{
    runTLoad<float, 128, 256, NO_PROFILING, 0.0f>();
}


