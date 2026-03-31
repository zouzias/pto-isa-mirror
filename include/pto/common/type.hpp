/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef _PTO_INCLUDE_NPU_TYPE_H_
#define _PTO_INCLUDE_NPU_TYPE_H_

#include <bit>
#include <cstdint>
#include <type_traits>

#if defined(__CPU_SIM) || defined(__COSTMODEL)
#define PTO_HOST_RUNTIME
#endif

#ifndef PTO_HOST_RUNTIME
#define AICORE [aicore]
#else
#define AICORE
#endif
#define PTO_INLINE inline __attribute__((always_inline))

// for pto instruction declaration
#define PTO_INST AICORE PTO_INLINE __attribute__((visibility("default")))
// for pto internal implementation
#define PTO_INTERNAL AICORE PTO_INLINE

#define OP_NAME(Name) __attribute__((vf_name(#Name)))
#define OP_TYPE(TypeName) __attribute__((vf_kind(#TypeName)))

// -----------------------------------------------------------------------------
// PTO assertion helpers
//
// Goals:
// - Provide a consistent diagnostic prefix across compile-time and runtime checks.
// - Always print/encode the violated condition when possible.
// - Provide a stable “next step” for users: see docs/coding/debug.md.
//
// Note:
// - `static_assert` diagnostics are compile-time only; we use a macro so we can
//   include the condition string and a docs hint consistently.
// - CPU simulator uses `PTO_CPU_ASSERT(...)` for runtime checks; it prints and
//   aborts (always enabled).
// -----------------------------------------------------------------------------

#define PTO_DETAIL_STR_(x) #x
#define PTO_DETAIL_STR(x) PTO_DETAIL_STR_(x)

#define PTO_STATIC_ASSERT_1(cond)                   \
    static_assert((cond),                           \
                  "[PTO][SA] Constraint violated. " \
                  "Condition: " #cond               \
                  ". "                              \
                  "Hint: see docs/coding/debug.md and search for " __FILE__ ":" PTO_DETAIL_STR(__LINE__))

#define PTO_STATIC_ASSERT_2(cond, msg)        \
    static_assert((cond), "[PTO][SA] " msg    \
                          " "                 \
                          "Condition: " #cond \
                          ". "                \
                          "Hint: see docs/coding/debug.md and search for " __FILE__ ":" PTO_DETAIL_STR(__LINE__))

#define PTO_DETAIL_GET_MACRO(_1, _2, NAME, ...) NAME
#define PTO_STATIC_ASSERT(...) PTO_DETAIL_GET_MACRO(__VA_ARGS__, PTO_STATIC_ASSERT_2, PTO_STATIC_ASSERT_1)(__VA_ARGS__)

#ifdef PTO_HOST_RUNTIME
#include <cstdio>
#include <cstdlib>

#define PTO_CPU_ASSERT_1(cond)                                                                               \
    do {                                                                                                     \
        if (!(cond)) {                                                                                       \
            std::fprintf(stderr,                                                                             \
                         "[PTO][CA] Constraint violated. Condition: %s. Hint: see docs/coding/debug.md and " \
                         "search for %s:%d\n",                                                               \
                         #cond, __FILE__, __LINE__);                                                         \
            std::abort();                                                                                    \
        }                                                                                                    \
    } while (0)

#define PTO_CPU_ASSERT_2(cond, msg)                                                                                   \
    do {                                                                                                              \
        if (!(cond)) {                                                                                                \
            std::fprintf(stderr, "[PTO][CA] %s Condition: %s. Hint: see docs/coding/debug.md and search for %s:%d\n", \
                         (msg), #cond, __FILE__, __LINE__);                                                           \
            std::abort();                                                                                             \
        }                                                                                                             \
    } while (0)

#define PTO_CPU_ASSERT(...) PTO_DETAIL_GET_MACRO(__VA_ARGS__, PTO_CPU_ASSERT_2, PTO_CPU_ASSERT_1)(__VA_ARGS__)
#else
// Non-CPU builds should not depend on CPU-only assertion behavior.
#define PTO_CPU_ASSERT(...) ((void)0)
#endif

namespace pto {
// 01-bits patterns are read from right to left.
// Right bits are low bits, corresponding to low index positions of data.
enum class MaskPattern : uint8_t
{
    // 以下1~7与指令VREDUCEv2的pattern mode保持一致
    P0101 = 1, // 1: 01010101...0101 # 每个repeat内每两个元素取第一个元素
    P1010 = 2, // 2: 10101010...1010 # 每个repeat内每两个元素取第二个元素
    P0001 = 3, // 3: 00010001...0001 # 每个repeat内每四个元素取第一个元素
    P0010 = 4, // 4: 00100010...0010 # 每个repeat内每四个元素取第二个元素
    P0100 = 5, // 5: 01000100...0100 # 每个repeat内每四个元素取第三个元素
    P1000 = 6, // 6: 10001000...1000 # 每个repeat内每四个元素取第四个元素
    P1111 = 7, // 7: 11111111...1111 # 每个repeat内取全部元素
};

enum class Layout
{
    ND, // ND RowMajor
    DN, // DN ColMajor
    NZ, // NZ for cube
    SCALE,
    MX_A_ND,
    MX_A_DN,
    MX_A_ZZ,
    MX_B_ND,
    MX_B_DN,
    MX_B_NN,
    NC1HWC0,
    NCHW,
    NHWC,
    NDC1HWC0,
    NCDHW,
    FRACTAL_Z,
    FRACTAL_Z_S16S8,
    FRACTAL_Z_3D,
    MAX,
};

enum class CmpMode : uint8_t
{
    EQ = 0,
    NE = 1,
    LT = 2,
    LE = 3,
    GT = 4,
    GE = 5,
};

// UF store phase encodes unit flag behavior for accumulator stores.
enum class STPhase : uint8_t
{
    Unspecified = 0x0,
    Partial = 0x2,
    Final = 0x3,
};

// Accumulate phase for unit-flag aware TMATMUL paths; Unknown is kept as an alias for compatibility.
enum class AccPhase : uint8_t
{
    Unspecified = 0x0,
    Unknown = Unspecified,
    Partial = 0x2,
    Final = 0x3,
};

enum VFImplKind : unsigned
{
    VFIMPL_DEFAULT = 0, // 默认版本
    VFIMPL_1D_NO_POST_UPDATE = 1,
    VFIMPL_2D_NO_POST_UPDATE = 2,
    VFIMPL_1D_POST_UPDATE = 3,
    VFIMPL_2D_POST_UPDATE = 4,
};

enum class RoundMode : uint8_t
{
    CAST_NONE = 0,
    CAST_RINT = 1,  // round to nearest, tie to even
    CAST_ROUND = 2, // round to nearest, tie away from zero
    CAST_FLOOR = 3, // round to minus infinity
    CAST_CEIL = 4,  // round to positive infinity
    CAST_TRUNC = 5, // round to zero
    CAST_ODD = 6,   // round to odd (Von Neumann rounding)
};

enum class TCopyMode : uint8_t
{
    SHALLOW_COPY = 0,
    DEEP_COPY = 1,
};

enum class AccToVecMode : uint8_t
{
    SingleModeVec0 = 0,
    SingleModeVec1 = 1,
    DualModeSplitM = 2,
    DualModeSplitN = 3,
};

enum class ReluPreMode : uint8_t
{
    NoRelu = 0,
    NormalRelu = 1,
};

enum class AtomicType : uint8_t
{
    AtomicNone = 0,
    AtomicAdd = 1,
};

// PadValue enum with uint64_t underlying type to support custom pad values.
// - Standard values (Null, Zero, Max, Min) use values 0-3
// - Custom values use bits [32:63] for the float bit pattern
// - Use PadCustom<-1.0f> helper from constants.hpp for custom values
enum class PadValue : uint64_t
{
    Null = 0,
    Zero = 1,
    Max = 2,
    Min = 3,
    // CustomBase marks the start of custom values (bit 32 set)
    CustomBase = 0x100000000ULL,
};

enum class SaturationMode : uint8_t
{
    // Saturation enabled (default) - CTRL bit 59 = 0
    ON = 0,

    // Saturation disabled - CTRL bit 59 = 1
    OFF = 1,
};

enum class CompactMode
{
    Null,
    Normal,
};

enum class SetFmatrixMode
{
    FMATRIX_A_AUTO,
    FMATRIX_B_AUTO,
    FMATRIX_A_MANUAL,
    FMATRIX_B_MANUAL,
};

enum class TileLayoutCustom : uint8_t
{
    ND,
    DN,
    NZ,
    ZN,
    ZZ,
    NONE,
};

namespace GlobalTensorDim {
constexpr int DIM_0 = 0;
constexpr int DIM_1 = 1;
constexpr int DIM_2 = 2;
constexpr int DIM_3 = 3;
constexpr int DIM_4 = 4;
constexpr int TOTAL_DIM = 5;
} // namespace GlobalTensorDim

} // namespace pto

#ifdef PTO_HOST_RUNTIME
#if defined(__clang__) && !defined(__FLT16_MANT_DIG__)
typedef __fp16 half;
typedef __fp16 aclFloat16;
#elif defined(__FLT16_MANT_DIG__)
typedef _Float16 half;
typedef _Float16 aclFloat16;
#else
inline uint16_t PtoHostFloatToHalfBits(float value)
{
    const uint32_t bits = std::bit_cast<uint32_t>(value);
    const uint32_t sign = (bits >> 16) & 0x8000u;
    uint32_t mantissa = bits & 0x007fffffu;
    int32_t exponent = static_cast<int32_t>((bits >> 23) & 0xffu);

    if (exponent == 0xff) {
        return static_cast<uint16_t>(sign | (mantissa == 0 ? 0x7c00u : 0x7e00u));
    }

    exponent = exponent - 127 + 15;
    if (exponent >= 0x1f) {
        return static_cast<uint16_t>(sign | 0x7c00u);
    }

    if (exponent <= 0) {
        if (exponent < -10) {
            return static_cast<uint16_t>(sign);
        }
        mantissa |= 0x00800000u;
        const uint32_t shift = static_cast<uint32_t>(14 - exponent);
        uint32_t halfMantissa = mantissa >> shift;
        const uint32_t roundBit = (mantissa >> (shift - 1u)) & 1u;
        halfMantissa += roundBit;
        return static_cast<uint16_t>(sign | halfMantissa);
    }

    uint32_t halfMantissa = mantissa >> 13;
    if ((mantissa & 0x00001000u) != 0) {
        ++halfMantissa;
        if (halfMantissa == 0x0400u) {
            halfMantissa = 0;
            ++exponent;
            if (exponent >= 0x1f) {
                return static_cast<uint16_t>(sign | 0x7c00u);
            }
        }
    }

    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exponent) << 10) | (halfMantissa & 0x03ffu));
}

inline float PtoHostHalfBitsToFloat(uint16_t value)
{
    const uint32_t sign = static_cast<uint32_t>(value & 0x8000u) << 16;
    uint32_t exponent = (value >> 10) & 0x1fu;
    uint32_t mantissa = value & 0x03ffu;
    uint32_t bits = 0;

    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {
            int32_t adjustedExponent = -14;
            while ((mantissa & 0x0400u) == 0) {
                mantissa <<= 1;
                --adjustedExponent;
            }
            mantissa &= 0x03ffu;
            bits = sign | (static_cast<uint32_t>(adjustedExponent + 127) << 23) | (mantissa << 13);
        }
    } else if (exponent == 0x1f) {
        bits = sign | 0x7f800000u | (mantissa << 13);
    } else {
        bits = sign | ((exponent + 112u) << 23) | (mantissa << 13);
    }

    return std::bit_cast<float>(bits);
}

struct alignas(2) half {
    uint16_t storage;

    constexpr half() : storage(0) {}
    constexpr half(const half &) = default;
    constexpr half &operator=(const half &) = default;

    template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T> && !std::is_same_v<T, half>>>
    half(T value) : storage(PtoHostFloatToHalfBits(static_cast<float>(value)))
    {}

    operator float() const
    {
        return PtoHostHalfBitsToFloat(storage);
    }
};

using aclFloat16 = half;
#endif
typedef half float16_t;
typedef float float32_t;
// Note: clang version should be >=15 and gcc version should be >=14
#if defined(__has_include) && __has_include(<stdfloat>) && __cplusplus >= 202302L
#include <stdfloat>
typedef std::bfloat16_t bfloat16_t;
#define CPU_SIM_BFLOAT_ENABLED
#else
// macOS libc++ (and some other toolchains) may not ship <stdfloat> yet.
// For CPU simulation, a best-effort 16-bit float type is sufficient.
typedef half bfloat16_t;
#define PTO_HOST_BFLOAT_IS_HALF
#endif
#endif

#endif
