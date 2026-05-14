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

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>

#include <pto/common/constants.hpp>
#include <pto/common/cpu_stub.hpp>
#include <pto/common/type.hpp>

namespace pto {

enum QuantModeCPU_t
{
    NoQuant = 0,
    F322F16 = 1,
    F322BF16 = 16,
    DEQF16 = 5,
    VDEQF16 = 4,
    QF322B8_PRE = 24,
    QF322F16_PRE = 32,
    QF322BF16_PRE = 34,
    VQF322B8_PRE = 23,
    REQ8 = 3,
    VREQ8 = 2,
    VSHIFTS322S16 = 12,
    SHIFTS322S16 = 13,
};

template <typename SrcType, typename DstType>
PTO_INTERNAL constexpr QuantModeCPU_t GetCastPreQuantMode()
{
    QuantModeCPU_t quantPre = QuantModeCPU_t::NoQuant;
    if constexpr (std::is_same<SrcType, float>::value) {
        if constexpr (std::is_same<DstType, half>::value) {
            quantPre = QuantModeCPU_t::F322F16;
        } else if constexpr (std::is_same<DstType, bfloat16_t>::value) {
            quantPre = QuantModeCPU_t::F322BF16;
        }
    }
    return quantPre;
}

template <typename SrcType, typename DstType>
PTO_INTERNAL constexpr QuantModeCPU_t GetScalarPreQuantMode()
{
    QuantModeCPU_t quantPre = QuantModeCPU_t::NoQuant;
    if constexpr (std::is_same<SrcType, float>::value) {
        if constexpr ((std::is_same<DstType, int8_t>::value) || (std::is_same<DstType, uint8_t>::value)) {
            quantPre = QuantModeCPU_t::QF322B8_PRE;
        } else if constexpr (std::is_same<DstType, half>::value) {
            quantPre = QuantModeCPU_t::QF322F16_PRE;
        } else if constexpr (std::is_same<DstType, bfloat16_t>::value) {
            quantPre = QuantModeCPU_t::QF322BF16_PRE;
        }
    } else if constexpr (std::is_same<SrcType, int32_t>::value) {
        if constexpr ((std::is_same<DstType, int8_t>::value) || (std::is_same<DstType, uint8_t>::value)) {
            quantPre = QuantModeCPU_t::REQ8;
        } else if constexpr (std::is_same<DstType, half>::value) {
            quantPre = QuantModeCPU_t::DEQF16;
        } else if constexpr (std::is_same<DstType, int16_t>::value) {
            quantPre = QuantModeCPU_t::SHIFTS322S16;
        }
    }
    return quantPre;
}

template <typename SrcType, typename DstType>
PTO_INTERNAL constexpr QuantModeCPU_t GetVectorPreQuantMode()
{
    QuantModeCPU_t quantPre = QuantModeCPU_t::NoQuant;
    if constexpr (std::is_same<SrcType, float>::value) {
        if constexpr ((std::is_same<DstType, int8_t>::value) || (std::is_same<DstType, uint8_t>::value)) {
            quantPre = QuantModeCPU_t::VQF322B8_PRE;
        }
    } else if constexpr (std::is_same<SrcType, int32_t>::value) {
        if constexpr ((std::is_same<DstType, int8_t>::value) || (std::is_same<DstType, uint8_t>::value)) {
            quantPre = QuantModeCPU_t::VREQ8;
        } else if constexpr (std::is_same<DstType, half>::value) {
            quantPre = QuantModeCPU_t::VDEQF16;
        } else if constexpr (std::is_same<DstType, int16_t>::value) {
            quantPre = QuantModeCPU_t::VSHIFTS322S16;
        }
    }
    return quantPre;
}

template <typename T>
inline T ReLU(T val)
{
    if (val < 0) {
        return 0;
    }
    return val;
}

inline float extract_m1_from_quant(uint64_t quant)
{
    uint32_t m1Bits = static_cast<uint32_t>((quant >> 13) & 0x7FFFF);
    uint32_t signBit = (m1Bits >> 18) & 0x1;
    uint32_t exponent = (m1Bits >> 10) & 0xFF;
    uint32_t mantissa = m1Bits & 0x3FF;

    if (exponent == 0 && mantissa == 0) {
        return 0.0f;
    }

    float signVal = (signBit == 1) ? -1.0f : 1.0f;
    float mantissaVal = 1.0f + (static_cast<float>(mantissa) / 1024.0f);
    float exponentVal = std::pow(2.0f, static_cast<float>(exponent) - 127.0f);

    return signVal * mantissaVal * exponentVal;
}

template <typename DstType, typename SrcType, QuantModeCPU_t mode, bool useRelu>
DstType quantize_element(SrcType srcVal, uint64_t scalar)
{
    uint64_t ctrlBits = get_task_cookie();
    float fScale = extract_m1_from_quant(scalar);
    uint32_t offset = static_cast<uint32_t>((scalar >> 37) & 0x1FF);
    uint32_t sign = static_cast<uint32_t>((scalar >> 46) & 0x1);
    uint32_t saturateInf = static_cast<uint32_t>((ctrlBits >> 48) & 0x1);

    float resultF = static_cast<float>(srcVal) * fScale;

    if constexpr (mode == QuantModeCPU_t::QF322B8_PRE || mode == QuantModeCPU_t::VQF322B8_PRE ||
                  mode == QuantModeCPU_t::REQ8 || mode == QuantModeCPU_t::VREQ8) {
        float rounded = std::nearbyint(resultF + offset);
        float minValue = sign == 1 ? -128.0f : 0.0f;
        float maxValue = sign == 1 ? 127.0f : 255.0f;
        resultF = std::clamp(rounded, minValue, maxValue);
    } else if constexpr (mode == QuantModeCPU_t::DEQF16 || mode == QuantModeCPU_t::VDEQF16) {
        resultF = std::clamp(resultF, -F16_MAX, F16_MAX);
    } else if constexpr (mode == QuantModeCPU_t::QF322F16_PRE) {
        if (std::isnan(resultF) && saturateInf == 1) {
            resultF = 0.0f;
        } else if (std::isfinite(resultF) || saturateInf == 1) {
            resultF = std::clamp(resultF, -F16_MAX, F16_MAX);
        }
    } else if constexpr (mode == QuantModeCPU_t::QF322BF16_PRE) {
        if (std::isnan(resultF) && saturateInf == 1) {
            resultF = 0.0f;
        } else if (std::isfinite(resultF) || saturateInf == 1) {
            float f32Max = std::numeric_limits<float>::max();
            resultF = std::clamp(resultF, -f32Max, f32Max);
        }
    } else if constexpr (mode == QuantModeCPU_t::SHIFTS322S16 || mode == QuantModeCPU_t::VSHIFTS322S16) {
        int32_t shiftBit = ((scalar >> 32) & 0xF) + 1;
        int32_t shiftedVal = srcVal >> shiftBit;
        int16_t i16Max = std::numeric_limits<int16_t>::max();
        int16_t i16Min = std::numeric_limits<int16_t>::min();
        resultF = std::clamp(static_cast<int16_t>(shiftedVal), i16Min, i16Max);
    }

    if constexpr (useRelu) {
        resultF = ReLU(resultF);
    }
    return static_cast<DstType>(resultF);
}

} // namespace pto

#endif
