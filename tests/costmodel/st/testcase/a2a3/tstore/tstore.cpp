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
void runTStore()
{
    using TileData = Tile<TileType::Vec, T, shape3, shape4>;
    TileData srcTile;
    TASSIGN(srcTile, 0x1000);

    constexpr int stride0 = 1 * 1 * shape3 * shape4;
    constexpr int stride1 = 1 * shape3 * shape4;
    constexpr int stride2 = shape3 * shape4;
    using StaticShapeDim5 = Shape<1, 1, 1, shape3, shape4>;
    using StaticStrideDim5 = pto::Stride<stride0, stride1, stride2, shape4, 1>;
    using GlobalData = GlobalTensor<T, StaticShapeDim5, StaticStrideDim5, Layout::ND>;
    GlobalData dstGlobal(0);
    TSTORE(dstGlobal, srcTile);

    EXPECT_CYCLE_NEAR(profiling, accuracy);
}

} // namespace

TEST(TStore, c01_nd_float_128x128)
{
    runTStore<float, 128, 128, 785.0f, 0.76f>();
}

TEST(TStore, c01_nd_float_64x128)
{
    runTStore<float, 64, 128, 490.0f, 0.61f>();
}

TEST(TStore, c01_nd_float_128x256)
{
    runTStore<float, 128, 256, 1390.0f, 0.86f>();
}

TEST(TStore, c01_nd_int16_t_128x128)
{
    runTStore<int16_t, 128, 128, 492.0f, 0.6f>();
}

TEST(TStore, c01_nd_int16_t_64x128)
{
    runTStore<int16_t, 64, 128, 345.0f, 0.43f>();
}

TEST(TStore, c01_nd_int16_t_128x256)
{
    runTStore<int16_t, 128, 256, 787.0f, 0.76f>();
}


