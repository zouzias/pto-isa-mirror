/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef COMMON_HPP
#define COMMON_HPP

#include <pto/common/type.hpp>

namespace pto {

template <typename SrcType, typename DstType>
PTO_INTERNAL constexpr QuantMode_t GetCastPreQuantMode()
{
    QuantMode_t quantPre = QuantMode_t::NoQuant;
    if constexpr (std::is_same<SrcType, float>::value) {
        if constexpr (std::is_same<DstType, half>::value) {
            quantPre = QuantMode_t::F322F16;
        } else if constexpr (std::is_same<DstType, bfloat16_t>::value) {
            quantPre = QuantMode_t::F322BF16;
        }
    }
    return quantPre;
}

template <typename SrcType, typename DstType>
PTO_INTERNAL constexpr QuantMode_t GetScalarPreQuantMode()
{
    QuantMode_t quantPre = QuantMode_t::NoQuant;
    if constexpr (std::is_same<SrcType, float>::value) {
        if constexpr ((std::is_same<DstType, int8_t>::value) || (std::is_same<DstType, uint8_t>::value)) {
            quantPre = QuantMode_t::QF322B8_PRE;
        } else if constexpr (std::is_same<DstType, half>::value) {
            quantPre = QuantMode_t::QF322F16_PRE;
        } else if constexpr (std::is_same<DstType, bfloat16_t>::value) {
            quantPre = QuantMode_t::QF322BF16_PRE;
        }
    } else if constexpr (std::is_same<SrcType, int32_t>::value) {
        if constexpr ((std::is_same<DstType, int8_t>::value) || (std::is_same<DstType, uint8_t>::value)) {
            quantPre = QuantMode_t::REQ8;
        } else if constexpr (std::is_same<DstType, half>::value) {
            quantPre = QuantMode_t::DEQF16;
        } else if constexpr (std::is_same<DstType, int16_t>::value) {
            quantPre = QuantMode_t::SHIFTS322S16;
        }
    }
    return quantPre;
}

template <typename SrcType, typename DstType>
PTO_INTERNAL constexpr QuantMode_t GetVectorPreQuantMode()
{
    QuantMode_t quantPre = QuantMode_t::NoQuant;
    if constexpr (std::is_same<SrcType, float>::value) {
        if constexpr ((std::is_same<DstType, int8_t>::value) || (std::is_same<DstType, uint8_t>::value)) {
            quantPre = QuantMode_t::VQF322B8_PRE;
        }
    } else if constexpr (std::is_same<SrcType, int32_t>::value) {
        if constexpr ((std::is_same<DstType, int8_t>::value) || (std::is_same<DstType, uint8_t>::value)) {
            quantPre = QuantMode_t::VREQ8;
        } else if constexpr (std::is_same<DstType, half>::value) {
            quantPre = QuantMode_t::VDEQF16;
        } else if constexpr (std::is_same<DstType, int16_t>::value) {
            quantPre = QuantMode_t::VSHIFTS322S16;
        }
    }
    return quantPre;
}

template <typename T>
inline T ReLU(T val)
{
    if (val < 0)
        return 0;
    return val;
}

inline float extract_m1_from_quant(uint64_t quant)
{
    uint32_t m1_bits = static_cast<uint32_t>((quant >> 13) & 0x7FFFF);
    uint32_t sign_bit = (m1_bits >> 18) & 0x1;
    uint32_t exponent = (m1_bits >> 10) & 0xFF;
    uint32_t mantissa = m1_bits & 0x3FF;

    if (exponent == 0 && mantissa == 0)
        return 0.0f;

    float sign_val = (sign_bit == 1) ? -1.0f : 1.0f;
    float mantissa_val = 1.0f + (static_cast<float>(mantissa) / 1024.0f);
    float exponent_val = std::pow(2.0f, static_cast<float>(exponent) - 127.0f);

    return sign_val * mantissa_val * exponent_val;
}

template <typename DstType, typename SrcType, QuantMode_t mode, bool use_relu>
DstType quantize_element(SrcType src_val, uint64_t scalar)
{
    float f_scale = extract_m1_from_quant(scalar);
    uint32_t offset = static_cast<uint32_t>((scalar >> 37) & 0x1FF);
    uint32_t sign = static_cast<uint32_t>((scalar >> 46) & 0x1);
    uint32_t saturate_inf = static_cast<uint32_t>((scalar >> 48) & 0x1);

    float result_f = static_cast<float>(src_val) * f_scale;

    if constexpr (mode == QuantMode_t::QF322B8_PRE || mode == QuantMode_t::VQF322B8_PRE || mode == QuantMode_t::REQ8 ||
                  mode == QuantMode_t::VREQ8) {
        float rounded = std::round(result_f + offset);
        float min = sign == 1 ? -128.0f : 0.0f;
        float max = sign == 1 ? 127.0f : 255.0f;
        result_f = std::clamp(rounded, min, max);
    } else if constexpr (mode == QuantMode_t::DEQF16 || mode == QuantMode_t::VDEQF16) {
        result_f = std::clamp(result_f, -F16_MAX, F16_MAX);
    } else if (mode == QuantMode_t::QF322F16_PRE) {
        if (std::isnan(result_f) && saturate_inf == 1) {
            result_f = 0.0f;
        } else if (std::isfinite(result_f) || saturate_inf == 1) {
            result_f = std::clamp(result_f, -F16_MAX, F16_MAX);
        }
    }
    if constexpr (use_relu)
        result_f = ReLU(result_f);
    return static_cast<DstType>(result_f);
}

} // namespace pto

#endif