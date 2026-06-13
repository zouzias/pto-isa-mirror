/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
*/

/**
 * @file TCvt.hpp
 * @brief Type Conversion (TCVT) Implementation for NPU KirinX90 Architecture
 */

#ifndef TCVT_HPP
#define TCVT_HPP

#include <pto/common/constants.hpp>
#include <pto/common/utils.hpp>
#include <array>

namespace pto {

// Import rounding type definitions
using __cce_simd::RoundAType;
using __cce_simd::RoundCType;
using __cce_simd::RoundFType;
using __cce_simd::RoundOType;
using __cce_simd::RoundRType;
using __cce_simd::RoundZType;

// Define architecture (uses RS_ENABLE workaround)
#define ARCH_KIRINX90

// Include common implementation
#include <pto/common/arch/register/tcvt_common.hpp>

// Undefine architecture macros
#undef ARCH_KIRINX90
#undef ARCH_RS_SAT
#undef ARCH_HAS_FP8
#undef ARCH_HAS_BFLOAT16

} // namespace pto

#endif // TCVT_HPP
