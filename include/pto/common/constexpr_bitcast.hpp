/**
 * @file constexpr_bitcast.hpp
 * @brief Compile-time bit_cast for C++17 (NPU compiler compatible)
 * 
 * Provides constexpr float-to-bits conversion without std::bit_cast (C++20)
 * or union type-punning (not constexpr in C++17).
 * 
 * Approach: Compute IEEE 754 representation mathematically at compile time.
 * 
 * Supported conversions:
 *   - float -> uint32_t
 *   - half (as float) -> uint16_t  
 *   - bfloat16 (as float) -> uint16_t
 */

#ifndef PTO_COMMON_CONSTEXPR_BITCAST_HPP
#define PTO_COMMON_CONSTEXPR_BITCAST_HPP

#include <cstdint>

namespace pto {
namespace detail {

// ============================================================================
// IEEE 754 Single Precision (float32) Layout:
//   [31]    Sign bit (1 = negative)
//   [30:23] Exponent (biased by 127)
//   [22:0]  Mantissa (implicit leading 1 for normal numbers)
// ============================================================================

constexpr int FLOAT_EXP_BIAS      = 127;
constexpr int FLOAT_MANTISSA_BITS = 23;

// Compile-time absolute value
constexpr float ct_fabs(float x) {
    return x < 0.0f ? -x : x;
}

// Compile-time floor(log2(x)) for x >= 1
constexpr int ct_ilog2_ge1(float x) {
    int exp = 0;
    while (x >= 2.0f) { x /= 2.0f; ++exp; }
    return exp;
}

// Compile-time floor(log2(x)) for 0 < x < 1
constexpr int ct_ilog2_lt1(float x) {
    int exp = 0;
    while (x < 1.0f) { x *= 2.0f; --exp; }
    return exp;
}

// Compile-time ldexp: x * 2^exp
constexpr float ct_ldexp(float x, int exp) {
    if (exp > 0) { while (exp-- > 0) x *= 2.0f; }
    else { while (exp++ < 0) x /= 2.0f; }
    return x;
}

// Main compile-time float-to-bits conversion
constexpr uint32_t ct_float_to_bits(float f) {
    // Handle zero
    if (f == 0.0f) return 0x00000000u;
    
    // Extract sign
    uint32_t sign = (f < 0.0f) ? 1u : 0u;
    float abs_f = ct_fabs(f);
    
    // Find exponent
    int unbiased_exp = (abs_f >= 1.0f) ? ct_ilog2_ge1(abs_f) : ct_ilog2_lt1(abs_f);
    
    // Normalize to [1.0, 2.0)
    float normalized = ct_ldexp(abs_f, -unbiased_exp);
    
    // Extract mantissa (fractional part after the implicit 1)
    float mantissa_f = normalized - 1.0f;
    uint32_t mantissa = static_cast<uint32_t>(mantissa_f * (1u << FLOAT_MANTISSA_BITS) + 0.5f);
    
    // Clamp mantissa (rounding might cause overflow)
    if (mantissa > 0x7FFFFFu) mantissa = 0x7FFFFFu;
    
    // Biased exponent
    uint32_t biased_exp = static_cast<uint32_t>(unbiased_exp + FLOAT_EXP_BIAS);
    
    return (sign << 31) | (biased_exp << FLOAT_MANTISSA_BITS) | mantissa;
}

// Half precision (fp16) conversion
constexpr uint16_t ct_half_to_bits(float f) {
    if (f == 0.0f) return 0x0000u;
    
    uint16_t sign = (f < 0.0f) ? 1u : 0u;
    float abs_f = ct_fabs(f);
    
    // Half precision range check
    constexpr float half_max = 65504.0f;
    if (abs_f > half_max) {
        return static_cast<uint16_t>((sign << 15) | 0x7C00u); // infinity
    }
    
    int unbiased_exp = (abs_f >= 1.0f) ? ct_ilog2_ge1(abs_f) : ct_ilog2_lt1(abs_f);
    float normalized = ct_ldexp(abs_f, -unbiased_exp);
    float mantissa_f = normalized - 1.0f;
    uint16_t mantissa = static_cast<uint16_t>(mantissa_f * 1024.0f + 0.5f); // 2^10
    
    if (mantissa > 0x3FFu) mantissa = 0x3FFu;
    
    uint16_t biased_exp = static_cast<uint16_t>(unbiased_exp + 15);
    
    return static_cast<uint16_t>((sign << 15) | (biased_exp << 10) | mantissa);
}

// BFloat16 conversion (truncated float32)
constexpr uint16_t ct_bf16_to_bits(float f) {
    return static_cast<uint16_t>(ct_float_to_bits(f) >> 16);
}

} // namespace detail

// ============================================================================
// Public API
// ============================================================================

/**
 * @brief Compile-time float to uint32_t bit cast
 */
constexpr uint32_t constexpr_float_to_bits(float f) {
    return detail::ct_float_to_bits(f);
}

/**
 * @brief Compile-time float to half (uint16_t) conversion
 */
constexpr uint16_t constexpr_float_to_half_bits(float f) {
    return detail::ct_half_to_bits(f);
}

/**
 * @brief Compile-time float to bfloat16 (uint16_t) conversion
 */
constexpr uint16_t constexpr_float_to_bf16_bits(float f) {
    return detail::ct_bf16_to_bits(f);
}

/**
 * @brief Create custom PadValue encoding from float (constexpr)
 * 
 * Encoding: (floatBits << 32) | 0x00000001
 * Example: -1.0f -> 0xBF80000000000001
 */
constexpr uint64_t MakePadValueCustom(float value) {
    return (static_cast<uint64_t>(detail::ct_float_to_bits(value)) << 32) | 0x00000001ULL;
}

} // namespace pto

#endif // PTO_COMMON_CONSTEXPR_BITCAST_HPP
