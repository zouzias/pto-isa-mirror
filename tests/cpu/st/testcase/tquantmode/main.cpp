/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <bit>
#include <cstdint>
#include <gtest/gtest.h>
#include <pto/pto-inst.hpp>

using namespace pto;

namespace {

// GetCastPreQuantMode mirrors the A5 table: only float -> {half, bfloat16} is a plain cast.
static_assert(GetCastPreQuantMode<float, half>() == QuantMode_t::F322F16);
static_assert(GetCastPreQuantMode<float, float>() == QuantMode_t::NoQuant);

// GetScalarPreQuantMode: scalar scale applied to a single value.
static_assert(GetScalarPreQuantMode<float, int8_t>() == QuantMode_t::QF322B8_PRE);
static_assert(GetScalarPreQuantMode<float, uint8_t>() == QuantMode_t::QF322B8_PRE);
static_assert(GetScalarPreQuantMode<float, half>() == QuantMode_t::QF322F16_PRE);
static_assert(GetScalarPreQuantMode<int32_t, int8_t>() == QuantMode_t::REQ8);
static_assert(GetScalarPreQuantMode<int32_t, uint8_t>() == QuantMode_t::REQ8);
static_assert(GetScalarPreQuantMode<int32_t, half>() == QuantMode_t::DEQF16);
static_assert(GetScalarPreQuantMode<int32_t, int16_t>() == QuantMode_t::SHIFTS322S16);
static_assert(GetScalarPreQuantMode<float, float>() == QuantMode_t::NoQuant);
static_assert(GetScalarPreQuantMode<int32_t, float>() == QuantMode_t::NoQuant);

// GetVectorPreQuantMode: per-column (tensor) scale.
static_assert(GetVectorPreQuantMode<float, int8_t>() == QuantMode_t::VQF322B8_PRE);
static_assert(GetVectorPreQuantMode<float, uint8_t>() == QuantMode_t::VQF322B8_PRE);
static_assert(GetVectorPreQuantMode<float, half>() == QuantMode_t::VQF322F16_PRE);
static_assert(GetVectorPreQuantMode<int32_t, int8_t>() == QuantMode_t::VREQ8);
static_assert(GetVectorPreQuantMode<int32_t, uint8_t>() == QuantMode_t::VREQ8);
static_assert(GetVectorPreQuantMode<int32_t, half>() == QuantMode_t::VDEQF16);
static_assert(GetVectorPreQuantMode<int32_t, int16_t>() == QuantMode_t::VSHIFTS322S16);
static_assert(GetVectorPreQuantMode<float, float>() == QuantMode_t::NoQuant);
static_assert(GetVectorPreQuantMode<int32_t, float>() == QuantMode_t::NoQuant);

#ifdef CPU_SIM_BFLOAT_ENABLED
// The A5 mappings only differ from the A2A3 ones once bfloat16 is a distinct type.
static_assert(GetCastPreQuantMode<float, bfloat16_t>() == QuantMode_t::F322BF16);
static_assert(GetScalarPreQuantMode<float, bfloat16_t>() == QuantMode_t::QF322BF16_PRE);
static_assert(GetScalarPreQuantMode<int32_t, bfloat16_t>() == QuantMode_t::QS322BF16_PRE);
static_assert(GetVectorPreQuantMode<float, bfloat16_t>() == QuantMode_t::VQF322BF16_PRE);
static_assert(GetVectorPreQuantMode<int32_t, bfloat16_t>() == QuantMode_t::VQS322BF16_PRE);
#endif

// Every V* mode is a tensor (vector) quant; the scalar/ABI variants are not.
static_assert(is_vector_quant_v<QuantMode_t::VQF322B8_PRE>);
static_assert(is_vector_quant_v<QuantMode_t::VDEQF16>);
static_assert(is_vector_quant_v<QuantMode_t::VREQ8>);
static_assert(is_vector_quant_v<QuantMode_t::VQF322F16_PRE>);
static_assert(is_vector_quant_v<QuantMode_t::VQF322BF16_PRE>);
static_assert(is_vector_quant_v<QuantMode_t::VQS322BF16_PRE>);
static_assert(!is_vector_quant_v<QuantMode_t::QF322F16_PRE>);
static_assert(!is_vector_quant_v<QuantMode_t::QS322BF16_PRE>);
static_assert(!is_vector_quant_v<QuantMode_t::SHIFTS322S16>);

constexpr uint32_t FloatBits(float value) { return std::bit_cast<uint32_t>(value); }

// Encodes a scalar float scale in the low 32 bits of the quant word.
uint64_t ScalarQuant(float scale) { return static_cast<uint64_t>(FloatBits(scale)); }

// Encodes an M1 of exactly 1.0 in the packed (1,8,10) tensor-scale format.
uint64_t UnitTensorQuant()
{
    constexpr uint64_t kM1One = static_cast<uint64_t>(127) << 10;
    return kM1One << 13;
}

} // namespace

// The new float -> half tensor path introduced for A5 must multiply by the per-column scale.
TEST(QuantModeMappingTest, vector_float_to_half_applies_tensor_scale)
{
    const uint64_t quant = UnitTensorQuant();
    const half result = quantize_element<half, float, QuantMode_t::VQF322F16_PRE, false>(3.0f, quant);
    EXPECT_FLOAT_EQ(static_cast<float>(result), 3.0f);
}

// The scalar float -> half path keeps its old behavior and still scales correctly.
TEST(QuantModeMappingTest, scalar_float_to_half_applies_scalar_scale)
{
    const uint64_t quant = ScalarQuant(0.5f);
    const half result = quantize_element<half, float, QuantMode_t::QF322F16_PRE, false>(2.0f, quant);
    EXPECT_FLOAT_EQ(static_cast<float>(result), 1.0f);
}

// Saturation (task cookie bit 48) must clamp an overflowing value to the half maximum instead
// of producing an infinity, matching the fixpipe behavior.
TEST(QuantModeMappingTest, vector_float_to_half_saturates_to_f16_max)
{
    const uint64_t quant = UnitTensorQuant();
    cpu_sim::set_task_cookie(static_cast<uint64_t>(1) << 48);
    const half result = quantize_element<half, float, QuantMode_t::VQF322F16_PRE, false>(1.0e9f, quant);
    cpu_sim::set_task_cookie(0);
    EXPECT_FLOAT_EQ(static_cast<float>(result), F16_MAX);
}

// int32 -> half tensor path clamps into the half range.
TEST(QuantModeMappingTest, vector_int32_to_half_clamps_to_f16)
{
    const uint64_t quant = UnitTensorQuant();
    const half result = quantize_element<half, int32_t, QuantMode_t::VDEQF16, false>(1 << 20, quant);
    EXPECT_FLOAT_EQ(static_cast<float>(result), F16_MAX);
}

#ifdef CPU_SIM_BFLOAT_ENABLED
// int32 -> bfloat16 scalar path is new for A5; the bfloat16 range is wide enough that scaling
// alone determines the result.
TEST(QuantModeMappingTest, scalar_int32_to_bfloat16_applies_scalar_scale)
{
    const uint64_t quant = ScalarQuant(2.0f);
    const bfloat16_t result = quantize_element<bfloat16_t, int32_t, QuantMode_t::QS322BF16_PRE, false>(3, quant);
    EXPECT_FLOAT_EQ(static_cast<float>(result), 6.0f);
}

// float -> bfloat16 tensor path multiplies by the per-column scale.
TEST(QuantModeMappingTest, vector_float_to_bfloat16_applies_tensor_scale)
{
    const uint64_t quant = UnitTensorQuant();
    const bfloat16_t result = quantize_element<bfloat16_t, float, QuantMode_t::VQF322BF16_PRE, false>(5.0f, quant);
    EXPECT_FLOAT_EQ(static_cast<float>(result), 5.0f);
}
#endif
