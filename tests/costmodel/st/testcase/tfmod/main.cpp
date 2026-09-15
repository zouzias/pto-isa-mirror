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

template <int rows, int cols, float expected>
void RunFmod()
{
    using TileData = Tile<TileType::Vec, float, rows, cols, BLayout::RowMajor, -1, -1>;
    TileData dst(rows, cols);
    TileData src0(rows, cols);
    TileData src1(rows, cols);
    TASSIGN(dst, 0x0);
    TASSIGN(src0, 0x4000);
    TASSIGN(src1, 0x8000);
    TFMOD(dst, src0, src1);
    EXPECT_CYCLE_NEAR(expected, 0.999999f);
}

template <int rows, int cols, float expected>
void RunFmods()
{
    using TileData = Tile<TileType::Vec, float, rows, cols, BLayout::RowMajor, -1, -1>;
    TileData dst(rows, cols);
    TileData src(rows, cols);
    TASSIGN(dst, 0x0);
    TASSIGN(src, 0x4000);
    TFMODS(dst, src, 1.25f);
    EXPECT_CYCLE_NEAR(expected, 0.999999f);
}

} // namespace

TEST(TFmod, BinaryTail) { RunFmod<16, 16, 1348.0f>(); }
TEST(TFmod, BinaryFourRepeats) { RunFmod<16, 256, 1764.0f>(); }
TEST(TFmod, ScalarTail) { RunFmods<32, 32, 3076.0f>(); }
TEST(TFmod, ScalarTwoRepeats) { RunFmods<64, 128, 6596.0f>(); }
