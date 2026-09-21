/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <array>
#include <cfenv>
#include <cmath>
#include <limits>
#include <gtest/gtest.h>
#include <pto/pto-inst.hpp>

using namespace pto;

namespace {
constexpr std::array<RoundMode, 7> FP8_MODES{RoundMode::CAST_NONE,  RoundMode::CAST_RINT, RoundMode::CAST_ROUND,
                                             RoundMode::CAST_FLOOR, RoundMode::CAST_CEIL, RoundMode::CAST_TRUNC,
                                             RoundMode::CAST_ODD};

template <typename T>
uint8_t encode(float value, RoundMode mode, SaturationMode sat = SaturationMode::OFF)
{
    return convert_value<T>(value, mode, sat).RawData();
}

// Decode finite positive codes independently of MXType for the reference grid.
float referenceValue(unsigned code, int mantissaBits, int bias)
{
    const unsigned exponent = code >> mantissaBits;
    const unsigned fraction = code & ((1u << mantissaBits) - 1);
    return exponent == 0 ? std::ldexp(float(fraction), 1 - bias - mantissaBits) :
                           std::ldexp(float((1u << mantissaBits) + fraction), int(exponent) - bias - mantissaBits);
}

template <typename T>
void checkFiniteGrid(int mantissaBits, int bias, unsigned maxCode)
{
    for (unsigned code = 0; code <= maxCode; ++code) {
        const float lower = referenceValue(code, mantissaBits, bias);
        for (unsigned sign : {0u, 0x80u}) {
            const float value = sign ? -lower : lower;
            SCOPED_TRACE(::testing::Message() << "code=" << code << " sign=" << sign);
            for (RoundMode mode : FP8_MODES) {
                for (SaturationMode sat : {SaturationMode::OFF, SaturationMode::ON})
                    EXPECT_EQ(encode<T>(value, mode, sat), sign | code);
            }
            EXPECT_EQ(double(T::FromRaw(sign | code)), double(value));
            if (code == maxCode)
                continue;
            const float upper = referenceValue(code + 1, mantissaBits, bias);
            const float midpoint = (lower + upper) / 2;
            for (float magnitude : {std::nextafter(midpoint, lower), midpoint, std::nextafter(midpoint, upper)}) {
                const float input = sign ? -magnitude : magnitude;
                const unsigned nearest = magnitude < midpoint ? code :
                                         magnitude > midpoint ? code + 1 :
                                                                code + (code & 1);
                EXPECT_EQ(encode<T>(input, RoundMode::CAST_RINT), sign | nearest);
                EXPECT_EQ(encode<T>(input, RoundMode::CAST_TRUNC), sign | code);
                EXPECT_EQ(encode<T>(input, RoundMode::CAST_FLOOR), sign | (code + (sign != 0)));
                EXPECT_EQ(encode<T>(input, RoundMode::CAST_CEIL), sign | (code + (sign == 0)));
                EXPECT_EQ(encode<T>(input, RoundMode::CAST_ROUND), sign | (code + (magnitude >= midpoint)));
                EXPECT_EQ(encode<T>(input, RoundMode::CAST_ODD), sign | nearest);
                EXPECT_EQ(encode<T>(input, RoundMode::CAST_NONE), sign | nearest);
            }
        }
    }
}
} // namespace

TEST(TCVTFP8Test, e4m3_all_finite_codes_and_midpoints) { checkFiniteGrid<float8_e4m3_t>(3, 7, 0x7e); }

TEST(TCVTFP8Test, e5m2_all_finite_codes_and_midpoints) { checkFiniteGrid<float8_e5m2_t>(2, 15, 0x7b); }

TEST(TCVTFP8Test, conversion_is_independent_of_host_rounding)
{
    struct RoundingGuard {
        int saved = std::fegetround();
        ~RoundingGuard() { std::fesetround(saved); }
    } guard;
    ASSERT_NE(guard.saved, -1);
    for (int mode : {FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO, FE_TONEAREST}) {
        ASSERT_EQ(std::fesetround(mode), 0);
        checkFiniteGrid<float8_e4m3_t>(3, 7, 0x7e);
        checkFiniteGrid<float8_e5m2_t>(2, 15, 0x7b);
    }
}

TEST(TCVTFP8Test, overflow_and_special_values)
{
    const float inf = std::numeric_limits<float>::infinity();
    for (unsigned sign : {0u, 0x80u}) {
        const float polarity = sign ? -1.0f : 1.0f;
        EXPECT_EQ(encode<float8_e4m3_t>(polarity * 464, RoundMode::CAST_RINT), sign | 0x7e);
        EXPECT_EQ(encode<float8_e4m3_t>(polarity * std::nextafter(464.0f, inf), RoundMode::CAST_RINT), sign | 0x7f);
        EXPECT_EQ(encode<float8_e5m2_t>(polarity * std::nextafter(61440.0f, 0.0f), RoundMode::CAST_RINT), sign | 0x7b);
        EXPECT_EQ(encode<float8_e5m2_t>(polarity * 61440, RoundMode::CAST_RINT), sign | 0x7c);
        EXPECT_EQ(encode<float8_e5m2_t>(polarity * 1e30f, RoundMode::CAST_TRUNC), sign | 0x7b);
        EXPECT_EQ(encode<float8_e4m3_t>(polarity * inf, RoundMode::CAST_RINT), sign | 0x7f);
        EXPECT_EQ(encode<float8_e5m2_t>(polarity * inf, RoundMode::CAST_RINT), sign | 0x7c);
        for (float magnitude : {1e30f, inf}) {
            EXPECT_EQ(
                encode<float8_e4m3_t>(polarity * magnitude, RoundMode::CAST_RINT, SaturationMode::ON), sign | 0x7e);
            EXPECT_EQ(
                encode<float8_e5m2_t>(polarity * magnitude, RoundMode::CAST_RINT, SaturationMode::ON), sign | 0x7b);
        }
        for (SaturationMode sat : {SaturationMode::OFF, SaturationMode::ON}) {
            const float nan = std::copysign(std::numeric_limits<float>::quiet_NaN(), polarity);
            EXPECT_TRUE(std::isnan(double(convert_value<float8_e4m3_t>(nan, RoundMode::CAST_RINT, sat))));
            EXPECT_TRUE(std::isnan(double(convert_value<float8_e5m2_t>(nan, RoundMode::CAST_RINT, sat))));
        }
        EXPECT_EQ(std::signbit(double(float8_e4m3_t::FromRaw(sign))), sign != 0);
        EXPECT_EQ(std::signbit(double(float8_e5m2_t::FromRaw(sign))), sign != 0);
        EXPECT_TRUE(std::isnan(double(float8_e4m3_t::FromRaw(sign | 0x7f))));
        EXPECT_TRUE(std::isinf(double(float8_e5m2_t::FromRaw(sign | 0x7c))));
        for (unsigned code = 0x7d; code <= 0x7f; ++code)
            EXPECT_TRUE(std::isnan(double(float8_e5m2_t::FromRaw(sign | code))));
        const float tiny = polarity * std::numeric_limits<float>::denorm_min();
        EXPECT_EQ(encode<float8_e4m3_t>(tiny, RoundMode::CAST_RINT), sign);
        EXPECT_EQ(encode<float8_e5m2_t>(tiny, RoundMode::CAST_RINT), sign);
        EXPECT_EQ(encode<float8_e4m3_t>(tiny, sign ? RoundMode::CAST_FLOOR : RoundMode::CAST_CEIL), sign | 1);
    }
}

TEST(TCVTFP8Test, legacy_constructors_and_e8m0)
{
    EXPECT_EQ(float8_e4m3_t(1.2).RawData(), 0x39);
    EXPECT_EQ(float8_e5m2_t(1.4).RawData(), 0x3d);
    EXPECT_EQ(encode<float8_e4m3_t>(1.2f, RoundMode::CAST_NONE), 0x3a);
    EXPECT_EQ(encode<float8_e8m0_t>(1.9f, RoundMode::CAST_RINT), float8_e8m0_t(1.9).RawData());
}

template <typename T>
void checkTileOverloads(const std::array<uint8_t, 6>& expected)
{
    using Src = Tile<TileType::Vec, float, 2, 32, BLayout::RowMajor, 1, 6>;
    using Dst = Tile<TileType::Vec, T, 2, 32, BLayout::RowMajor, 1, 6>;
    Src src;
    Dst dst;
    Tile<TileType::Vec, float, 1, 32> tmp;
    TASSIGN(src, 0);
    TASSIGN(dst, 256);
    const std::array<float, 6> inputs{1.1875f, -1.2f,     0.0146484375f,
                                      -0.0f,   100000.0f, std::numeric_limits<float>::quiet_NaN()};
    for (unsigned i = 0; i < inputs.size(); ++i)
        src.SetElement(0, i, inputs[i]);
    dst.SetElement(0, 6, T::FromRaw(0x55));
    dst.SetElement(1, 0, T::FromRaw(0x55));
    for (bool withTmp : {false, true}) {
        if (withTmp)
            TCVT(dst, src, tmp, RoundMode::CAST_RINT);
        else
            TCVT(dst, src, RoundMode::CAST_RINT);
        for (unsigned i = 0; i < inputs.size(); ++i)
            EXPECT_EQ(dst.GetElement(0, i).RawData(), expected[i]);
        if (withTmp)
            TCVT(dst, src, tmp, RoundMode::CAST_RINT, SaturationMode::ON);
        else
            TCVT(dst, src, RoundMode::CAST_RINT, SaturationMode::ON);
        EXPECT_EQ(dst.GetElement(0, 4).RawData(), (std::is_same_v<T, float8_e4m3_t> ? 0x7e : 0x7b));
        EXPECT_EQ(dst.GetElement(0, 5).RawData(), 0x7f);
        for (RoundMode mode : FP8_MODES) {
            for (SaturationMode sat : {SaturationMode::OFF, SaturationMode::ON}) {
                if (withTmp)
                    TCVT(dst, src, tmp, mode, sat);
                else
                    TCVT(dst, src, mode, sat);
                for (unsigned i = 0; i < inputs.size(); ++i)
                    EXPECT_EQ(dst.GetElement(0, i).RawData(), encode<T>(inputs[i], mode, sat));
            }
        }
        EXPECT_EQ(dst.GetElement(0, 6).RawData(), 0x55);
        EXPECT_EQ(dst.GetElement(1, 0).RawData(), 0x55);
    }
}

TEST(TCVTFP8Test, default_modes_preserve_saturation_and_special_values)
{
    const float inf = std::numeric_limits<float>::infinity();
    for (RoundMode mode : {RoundMode::CAST_NONE, RoundMode::CAST_ODD}) {
        EXPECT_EQ(encode<float8_e4m3_t>(1.1875f, mode), 0x3a);
        EXPECT_EQ(encode<float8_e5m2_t>(1.375f, mode), 0x3e);
        for (unsigned sign : {0u, 0x80u}) {
            const float polarity = sign ? -1.0f : 1.0f;
            for (float value : {100000.0f, inf}) {
                EXPECT_EQ(encode<float8_e4m3_t>(polarity * value, mode, SaturationMode::ON), sign | 0x7e);
                EXPECT_EQ(encode<float8_e5m2_t>(polarity * value, mode, SaturationMode::ON), sign | 0x7b);
            }
            for (SaturationMode sat : {SaturationMode::OFF, SaturationMode::ON}) {
                EXPECT_EQ(encode<float8_e4m3_t>(polarity * 0.0f, mode, sat), sign);
                EXPECT_EQ(encode<float8_e5m2_t>(polarity * 0.0f, mode, sat), sign);
                const float nan = std::copysign(std::numeric_limits<float>::quiet_NaN(), polarity);
                EXPECT_EQ(encode<float8_e4m3_t>(nan, mode, sat), sign | 0x7f);
                EXPECT_EQ(encode<float8_e5m2_t>(nan, mode, sat), sign | 0x7f);
            }
        }
    }
}

TEST(TCVTFP8Test, e4m3_tile_overloads_saturation_and_valid_region)
{
    checkTileOverloads<float8_e4m3_t>({0x3a, 0xba, 0x08, 0x80, 0x7f, 0x7f});
}

TEST(TCVTFP8Test, e5m2_tile_overloads_saturation_and_valid_region)
{
    checkTileOverloads<float8_e5m2_t>({0x3d, 0xbd, 0x24, 0x80, 0x7c, 0x7f});
}

namespace {
struct OverflowCase {
    float value;
    uint8_t rint;
    uint8_t round;
    uint8_t trunc;
    uint8_t away;
};

template <typename T>
void checkOverflow(const std::initializer_list<OverflowCase>& cases, unsigned maxCode)
{
    for (const auto& test : cases) {
        for (unsigned sign : {0u, 0x80u}) {
            const float value = sign ? -test.value : test.value;
            SCOPED_TRACE(::testing::Message() << "input=" << value);
            EXPECT_EQ(encode<T>(value, RoundMode::CAST_RINT), sign | test.rint);
            EXPECT_EQ(encode<T>(value, RoundMode::CAST_NONE), sign | test.rint);
            EXPECT_EQ(encode<T>(value, RoundMode::CAST_ODD), sign | test.rint);
            EXPECT_EQ(encode<T>(value, RoundMode::CAST_ROUND), sign | test.round);
            EXPECT_EQ(encode<T>(value, RoundMode::CAST_TRUNC), sign | test.trunc);
            EXPECT_EQ(encode<T>(value, RoundMode::CAST_FLOOR), sign | (sign ? test.away : test.trunc));
            EXPECT_EQ(encode<T>(value, RoundMode::CAST_CEIL), sign | (sign ? test.trunc : test.away));
            for (RoundMode mode : FP8_MODES)
                EXPECT_EQ(encode<T>(value, mode, SaturationMode::ON), sign | maxCode);
        }
    }
}
} // namespace

TEST(TCVTFP8Test, e4m3_overflow_boundaries_all_modes)
{
    const float inf = std::numeric_limits<float>::infinity();
    checkOverflow<float8_e4m3_t>(
        {{std::nextafter(448.0f, inf), 0x7e, 0x7e, 0x7e, 0x7f},
         {464.0f, 0x7e, 0x7f, 0x7e, 0x7f},
         {std::nextafter(464.0f, inf), 0x7f, 0x7f, 0x7e, 0x7f},
         {std::nextafter(480.0f, 0.0f), 0x7f, 0x7f, 0x7e, 0x7f},
         {480.0f, 0x7f, 0x7f, 0x7f, 0x7f},
         {512.0f, 0x7f, 0x7f, 0x7f, 0x7f},
         {std::numeric_limits<float>::max(), 0x7f, 0x7f, 0x7f, 0x7f},
         {inf, 0x7f, 0x7f, 0x7f, 0x7f}},
        0x7e);
}

TEST(TCVTFP8Test, e5m2_overflow_boundaries_all_modes)
{
    const float inf = std::numeric_limits<float>::infinity();
    checkOverflow<float8_e5m2_t>(
        {{std::nextafter(57344.0f, inf), 0x7b, 0x7b, 0x7b, 0x7c},
         {std::nextafter(61440.0f, 0.0f), 0x7b, 0x7b, 0x7b, 0x7c},
         {61440.0f, 0x7c, 0x7c, 0x7b, 0x7c},
         {65536.0f, 0x7c, 0x7c, 0x7b, 0x7c},
         {std::numeric_limits<float>::max(), 0x7c, 0x7c, 0x7b, 0x7c},
         {inf, 0x7c, 0x7c, 0x7c, 0x7c}},
        0x7b);
}

TEST(TCVTFP8Test, double_extremes)
{
    for (double sign : {-1.0, 1.0}) {
        const unsigned signBit = sign < 0 ? 0x80 : 0;
        const double tiny = sign * std::numeric_limits<double>::denorm_min();
        EXPECT_EQ(float8_e4m3_t(tiny, RoundMode::CAST_RINT).RawData(), signBit);
        EXPECT_EQ(float8_e5m2_t(tiny, RoundMode::CAST_RINT).RawData(), signBit);
        const RoundMode away = sign < 0 ? RoundMode::CAST_FLOOR : RoundMode::CAST_CEIL;
        EXPECT_EQ(float8_e4m3_t(tiny, away).RawData(), signBit | 1);
        EXPECT_EQ(float8_e5m2_t(tiny, away).RawData(), signBit | 1);
        const double huge = sign * std::numeric_limits<double>::max();
        EXPECT_EQ(float8_e4m3_t(huge, RoundMode::CAST_RINT).RawData(), signBit | 0x7f);
        EXPECT_EQ(float8_e5m2_t(huge, RoundMode::CAST_RINT).RawData(), signBit | 0x7c);
    }
}
