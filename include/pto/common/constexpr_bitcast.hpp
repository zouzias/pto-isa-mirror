/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMMON_CONSTEXPR_BITCAST_HPP
#define PTO_COMMON_CONSTEXPR_BITCAST_HPP

#include <cstdint>

namespace pto {

/**
 * Compile-time IEEE 754 float-to-bits conversion for C++17.
 * Works on NPU compilers that lack std::bit_cast (C++20).
 *
 * Algorithm: Decompose float into sign/exponent/mantissa mathematically,
 * then reconstruct the IEEE 754 bit pattern.
 *
 * Handles: normal numbers, zero, negative zero, infinity, negative infinity.
 * Note: NaN values may not preserve exact payload bits.
 */
constexpr uint32_t constexpr_float_to_bits(float value)
{
    // Handle zero (positive and negative)
    if (value == 0.0f) {
        // Check for negative zero by comparing 1/value
        // Note: In constexpr context, we cannot distinguish -0.0f reliably
        // without C++20 bit_cast, so we return positive zero for both
        return 0u;
    }

    // Handle infinity
    if (value > 3.4028235e+38f) {
        return 0x7F800000u;  // +inf
    }
    if (value < -3.4028235e+38f) {
        return 0xFF800000u;  // -inf
    }

    // Extract sign
    const bool sign = value < 0;
    float abs_val = sign ? -value : value;

    // Handle denormals (very small numbers)
    if (abs_val < 1.17549435e-38f) {
        // Denormal: exponent is 0, mantissa encodes the value
        // mantissa = abs_val / 2^(-126) * 2^23 = abs_val * 2^149
        float scaled = abs_val;
        for (int i = 0; i < 149; ++i) {
            scaled *= 2.0f;
        }
        uint32_t mant_bits = static_cast<uint32_t>(scaled);
        return (sign ? 0x80000000u : 0u) | mant_bits;
    }

    // Normal numbers: find exponent by repeated division/multiplication
    int exp = 0;
    float mantissa = abs_val;

    // Scale to [1.0, 2.0) range
    while (mantissa >= 2.0f) {
        mantissa /= 2.0f;
        ++exp;
    }
    while (mantissa < 1.0f) {
        mantissa *= 2.0f;
        --exp;
    }

    // Bias exponent (IEEE 754 bias is 127)
    int biased_exp = exp + 127;

    // Clamp to valid range
    if (biased_exp >= 255) {
        return sign ? 0xFF800000u : 0x7F800000u;  // infinity
    }
    if (biased_exp <= 0) {
        return sign ? 0x80000000u : 0u;  // zero (denormals handled above)
    }

    // Extract mantissa bits (23 bits, implicit 1 removed)
    // mantissa is in [1.0, 2.0), so (mantissa - 1.0) is in [0.0, 1.0)
    // Multiply by 2^23 to get the 23-bit fraction
    uint32_t mant_bits = static_cast<uint32_t>((mantissa - 1.0f) * 8388608.0f + 0.5f);
    mant_bits &= 0x7FFFFF;  // Ensure only 23 bits

    // Combine: sign (1 bit) | exponent (8 bits) | mantissa (23 bits)
    return (sign ? 0x80000000u : 0u) |
           (static_cast<uint32_t>(biased_exp) << 23) |
           mant_bits;
}

}  // namespace pto

#endif  // PTO_COMMON_CONSTEXPR_BITCAST_HPP
