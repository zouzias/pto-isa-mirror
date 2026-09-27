/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software; you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <array>
#include <bit>
#include <cstdint>
#include <gtest/gtest.h>
#include <pto/pto-inst.hpp>

using namespace pto;

namespace {
uint32_t FloatBits(float value) { return std::bit_cast<uint32_t>(value); }

struct IntToFloatExpected {
    RoundMode mode;
    uint32_t positive;
    uint32_t negative;
};

constexpr std::array<IntToFloatExpected, 5> INT_TO_FLOAT_EXPECTED{{
    {RoundMode::CAST_CEIL, 0x4b800001, 0xcb800000},
    {RoundMode::CAST_ROUND, 0x4b800001, 0xcb800001},
    {RoundMode::CAST_FLOOR, 0x4b800000, 0xcb800001},
    {RoundMode::CAST_TRUNC, 0x4b800000, 0xcb800000},
    {RoundMode::CAST_RINT, 0x4b800000, 0xcb800000},
}};
} // namespace

TEST(TCVTRoundModeTest, int32_to_fp32_honors_round_mode)
{
    using Src = Tile<TileType::Vec, int32_t, 1, 16, BLayout::RowMajor, 1, 4>;
    using Dst = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, 1, 4>;
    Src src;
    Dst dst;
    TASSIGN(src, 0);
    TASSIGN(dst, 256);
    src.SetElement(0, 0, 16777217);
    src.SetElement(0, 1, -16777217);
    src.SetElement(0, 2, 16777216);
    src.SetElement(0, 3, -16777216);

    for (const auto& test : INT_TO_FLOAT_EXPECTED) {
        TCVT(dst, src, test.mode);
        EXPECT_EQ(FloatBits(dst.GetElement(0, 0)), test.positive);
        EXPECT_EQ(FloatBits(dst.GetElement(0, 1)), test.negative);
        EXPECT_EQ(FloatBits(dst.GetElement(0, 2)), 0x4b800000);
        EXPECT_EQ(FloatBits(dst.GetElement(0, 3)), 0xcb800000);
    }
}

TEST(TCVTRoundModeTest, int64_to_fp32_handles_the_signed_limit)
{
    using Src = Tile<TileType::Vec, int64_t, 1, 16, BLayout::RowMajor, 1, 1>;
    using Dst = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, 1, 1>;
    Src src;
    Dst dst;
    TASSIGN(src, 0);
    TASSIGN(dst, 256);
    src.SetElement(0, 0, INT64_MAX);

    constexpr std::array<RoundMode, 5> modes{
        {RoundMode::CAST_CEIL, RoundMode::CAST_ROUND, RoundMode::CAST_FLOOR, RoundMode::CAST_TRUNC,
         RoundMode::CAST_RINT}};
    constexpr std::array<uint32_t, 5> expected{{0x5f000000, 0x5f000000, 0x5effffff, 0x5effffff, 0x5f000000}};
    for (size_t i = 0; i < modes.size(); ++i) {
        TCVT(dst, src, modes[i]);
        EXPECT_EQ(FloatBits(dst.GetElement(0, 0)), expected[i]);
    }
}

TEST(TCVTRoundModeTest, other_conversion_directions_keep_their_rounding)
{
    constexpr std::array<RoundMode, 5> modes{
        {RoundMode::CAST_CEIL, RoundMode::CAST_ROUND, RoundMode::CAST_FLOOR, RoundMode::CAST_TRUNC,
         RoundMode::CAST_RINT}};
    constexpr std::array<int32_t, 5> positiveExpected{{2, 2, 1, 1, 2}};
    constexpr std::array<int32_t, 5> negativeExpected{{-1, -2, -2, -1, -2}};
    using FloatSrc = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, 1, 2>;
    using IntDst = Tile<TileType::Vec, int32_t, 1, 16, BLayout::RowMajor, 1, 2>;
    FloatSrc floatSrc;
    IntDst intDst;
    TASSIGN(floatSrc, 0);
    TASSIGN(intDst, 256);
    floatSrc.SetElement(0, 0, 1.5f);
    floatSrc.SetElement(0, 1, -1.5f);
    for (size_t i = 0; i < modes.size(); ++i) {
        TCVT(intDst, floatSrc, modes[i]);
        EXPECT_EQ(intDst.GetElement(0, 0), positiveExpected[i]);
        EXPECT_EQ(intDst.GetElement(0, 1), negativeExpected[i]);
    }

    using IntSrc = Tile<TileType::Vec, int32_t, 1, 16, BLayout::RowMajor, 1, 2>;
    using ShortDst = Tile<TileType::Vec, int16_t, 1, 16, BLayout::RowMajor, 1, 2>;
    IntSrc intSrc;
    ShortDst shortDst;
    TASSIGN(intSrc, 512);
    TASSIGN(shortDst, 768);
    intSrc.SetElement(0, 0, 123);
    intSrc.SetElement(0, 1, -123);
    for (RoundMode mode : modes) {
        TCVT(shortDst, intSrc, mode);
        EXPECT_EQ(shortDst.GetElement(0, 0), 123);
        EXPECT_EQ(shortDst.GetElement(0, 1), -123);
    }
}
