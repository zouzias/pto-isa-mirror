/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

/**
 * @file TCvt.hpp
 * @brief Type Conversion (TCVT) Implementation for NPU A5 Architecture
 *
 * FILE ORGANIZATION (for easy navigation):
 * =======================================
 *
 * 1. CastMode enum and helper macros (lines ~77-100)
 *
 * 2. 1D Helper Templates (lines ~103-466)
 *    - Optimized for contiguous data without padding
 *    - castS64to32_1D_NoPostUpdate, cast32to16_1D_NoPostUpdate, cast32to32_1D_NoPostUpdate, cast32toS64_1D_NoPostUpdate
 *    - cast16to16_1D_NoPostUpdate, cast16to32_1D_NoPostUpdate, cast16to8_1D_NoPostUpdate
 *    - cast8to16_1D_NoPostUpdate, cast8to32_1D_NoPostUpdate, cast32to8_1D_NoPostUpdate, cast32toH8_1D_NoPostUpdate,
 * cast16toH8_1D_NoPostUpdate
 *
 * 3. 2D Helper Templates (lines ~467-855)
 *    - For data with row/column layout and potential padding
 *    - Same function set as 1D but with row iteration
 *
 * 4. castData Overloads - 2D versions (lines ~856-1503)
 *    Organized by SOURCE type for easy lookup:
 *    - FP32 (float)        → fp16, bf16, int16, int32, int64, fp8 variants
 *    - FP16 (half)         → fp32, int32, int16, int8, uint8, h8 (hifloat8 only)
 *    - BFloat16            → fp32, int32, half
 *    - U8, I8 (8-bit int)  → half, uint16, int16, int32
 *    - I16 (16-bit int)    → uint8, half, float, uint32, int32
 *    - I32 (32-bit int)    → float, int16, uint16, int64, uint8
 *    - U32 (32-bit uint)   → uint8, uint16, int16
 *    - I64 (64-bit int)    → float, int32
 *    - FP8 variants        → float
 *
 * 5. castData_1D_NoPostUpdate Overloads (lines ~1504-1710)
 *    - Same organization as 2D versions, optimized for contiguous data
 *
 * 6. Main TCVT Implementation (lines ~1711-end)
 *    - implTCVT: Main template function
 *    - TCVT_IMPL: Rounding mode dispatcher
 *
 * QUICK FIND: To find a specific conversion, search for the source type section header,
 * e.g., "Source: FP32" or "Source: I16", then look for the destination type.
 */

#ifndef TCVT_HPP
#define TCVT_HPP

#include <pto/common/constants.hpp>
#include <pto/common/utils.hpp>
#include <array>
#include "common.hpp"
#include "utils.hpp"

namespace pto {

enum class CastMode;

template <typename R, typename DST, typename SRC>
inline AICORE void cast32to16_1D_NoPostUpdate(__ubuf__ DST *dst, __ubuf__ SRC *src, uint32_t validRows,
                                              uint32_t validCols, uint32_t dstCols, uint32_t srcCols,
                                              SaturationMode satMode);

template <typename R, CastMode MODE, typename DST_VEC, typename DST, typename SRC>
inline AICORE void cast16to8_1D_NoPostUpdate(__ubuf__ DST *dst, __ubuf__ SRC *src, uint32_t validRows,
                                             uint32_t validCols, uint32_t dstCols, uint32_t srcCols,
                                             SaturationMode satMode);

template <typename R, CastMode MODE, typename DST_VEC, typename DST, typename SRC>
inline AICORE void cast32to8_1D_NoPostUpdate(__ubuf__ DST *dst, __ubuf__ SRC *src, uint32_t validRows,
                                             uint32_t validCols, uint32_t dstCols, uint32_t srcCols,
                                             SaturationMode satMode);

template <typename R, typename DST, typename SRC>
inline AICORE void cast32to16(__ubuf__ DST *dst, __ubuf__ SRC *src, uint32_t validRows, uint32_t validCols,
                              uint32_t dstCols, uint32_t srcCols, SaturationMode satMode);

template <typename R, typename DST, typename SRC>
inline AICORE void cast32to16_2D_NoPostUpdate(__ubuf__ DST *dst, __ubuf__ SRC *src, uint32_t validRows,
                                              uint32_t validCols, uint32_t dstCols, uint32_t srcCols,
                                              SaturationMode satMode);

template <typename R, CastMode MODE, typename DST_VEC, typename DST, typename SRC>
inline AICORE void cast16to8(__ubuf__ DST *dst, __ubuf__ SRC *src, uint32_t validRows, uint32_t validCols,
                             uint32_t dstCols, uint32_t srcCols, SaturationMode satMode);

template <typename R, CastMode MODE, typename DST_VEC, typename DST, typename SRC>
inline AICORE void cast16to8_2D_NoPostUpdate(__ubuf__ DST *dst, __ubuf__ SRC *src, uint32_t validRows,
                                             uint32_t validCols, uint32_t dstCols, uint32_t srcCols,
                                             SaturationMode satMode);

template <typename R, CastMode MODE, typename DST_VEC, typename DST, typename SRC>
inline AICORE void cast32to8(__ubuf__ DST *dst, __ubuf__ SRC *src, uint32_t validRows, uint32_t validCols,
                             uint32_t dstCols, uint32_t srcCols, SaturationMode satMode);

} // namespace pto
#include <pto/common/npu_dedup/tcvt_common.hpp>

namespace pto {

// Import rounding type definitions from __cce_simd namespace
using __cce_simd::RoundAType;
using __cce_simd::RoundCType;
using __cce_simd::RoundFType;
using __cce_simd::RoundOType;
using __cce_simd::RoundRType;
using __cce_simd::RoundZType;

//=============================================================================================
// 1D Helper Templates - For contiguous data (optimized fast path)
//=============================================================================================
// These templates handle conversions when data is laid out contiguously in memory without
// padding. They process data in a single pass without row/column iteration overhead.
//
// PERFORMANCE NOTE: 1D versions are significantly faster than 2D versions when applicable,
// as they avoid the FOR_ROWS/FOR_ELEMENTS loop overhead and process data in bulk.

/**
 * Cast 64-bit integer to 32-bit (signed/float) - 1D version
 * Handles: s64 -> s32 #sat #part, s64 -> f32 #rnd #part
 */
template <typename R, typename DST, typename SRC>
inline AICORE void castS64to32_1D_NoPostUpdate(__ubuf__ DST *dst, __ubuf__ SRC *src, uint32_t validRows,
                                               uint32_t validCols, uint32_t dstCols, uint32_t srcCols,
                                               SaturationMode satMode)
{
    vector_s64 v_input_0;
    const uint32_t ELE_CNT_B64 = ELE_CNT_B32 / 2;
    uint32_t totalElements = validRows * validCols;
    uint16_t repeatTimes = CeilDivision(totalElements, ELE_CNT_B64);
    uint32_t sReg = totalElements;
    uint32_t len64 = sReg * 2;
    uint32_t len_even = sReg * 2;

    for (uint16_t i = 0; i < repeatTimes; ++i) {
        RegTensor<DST> v_output;
        MaskReg preg_b64 = CreatePredicate<float>(len64);
        MaskReg preg_b32 = CreatePredicate<float>(len_even);

        vlds(v_input_0, src, i * ELE_CNT_B64, NORM);
        if constexpr (std::is_same<R, void>::value) {
            vcvt(v_output, v_input_0, preg_b64, RS_DISABLE, PART_EVEN);
        } else {
            vcvt(v_output, v_input_0, preg_b64, R(), PART_EVEN);
        }
        vsts(v_output, dst, i * ELE_CNT_B64, PK_B64, preg_b32);
        // sReg is decremented by CreatePredicate with POST_UPDATE
    }
}

// Cast 32-bit -> 16-bit (1D)
template <typename R, typename DST, typename SRC>
inline AICORE void cast32to16_1D_NoPostUpdate(__ubuf__ DST *dst, __ubuf__ SRC *src, uint32_t validRows,
                                              uint32_t validCols, uint32_t dstCols, uint32_t srcCols,
                                              SaturationMode satMode)
{
    uint32_t totalElements = validRows * validCols;
    uint16_t repeatTimes = CeilDivision(totalElements, ELE_CNT_B32);
    uint32_t sReg = totalElements;
    uint32_t len32 = ELE_CNT_B32;
    MaskReg preg_b32 = CreatePredicate<float>(len32);

    for (uint16_t i = 0; i < repeatTimes; ++i) {
        RegTensor<SRC> v_input_0;
        RegTensor<DST> v_output_even;
        MaskReg preg_b32_st = CreatePredicate<float>(sReg);

        vlds(v_input_0, src, i * ELE_CNT_B32, NORM);
        if constexpr (std::is_same<R, void>::value) {
            vcvt(v_output_even, v_input_0, preg_b32, RS_DISABLE, PART_EVEN);
        } else {
            vcvt(v_output_even, v_input_0, preg_b32, R(), RS_DISABLE, PART_EVEN);
        }
        vsts(v_output_even, dst, i * ELE_CNT_B32, PK_B32, preg_b32_st);
        // sReg is decremented by CreatePredicate with POST_UPDATE
    }
}

// Cast 32-bit -> s64 (1D)
template <typename R, typename SRC>
inline AICORE void cast32toS64_1D_NoPostUpdate(__ubuf__ int64_t *dst, __ubuf__ SRC *src, uint32_t validRows,
                                               uint32_t validCols, uint32_t dstCols, uint32_t srcCols,
                                               SaturationMode satMode)
{
    const uint32_t ELE_CNT_B64 = ELE_CNT_B32 / 2;
    uint32_t totalElements = validRows * validCols;
    uint16_t repeatTimes = CeilDivision(totalElements, ELE_CNT_B64);
    uint32_t sReg = totalElements;
    uint32_t len32 = ELE_CNT_B32;
    MaskReg preg_b32 = CreatePredicate<float>(len32);
    uint32_t len64 = sReg * 2;

    for (uint16_t i = 0; i < repeatTimes; ++i) {
        RegTensor<SRC> v_input_0;
        vector_s64 v_output;
        MaskReg preg_b64 = CreatePredicate<float>(len64);

        vlds(v_input_0, src, i * ELE_CNT_B64, UNPK_B32);
        if constexpr (std::is_same<R, void>::value) {
            // For type expansion s32->s64 without rounding, no saturation control
            vcvt(v_output, v_input_0, preg_b32, PART_EVEN);
        } else {
            // For conversions with rounding (e.g., f32->s64), saturation mode is controllable
            vcvt(v_output, v_input_0, preg_b32, R(), RS_DISABLE, PART_EVEN);
        }
        vsts(v_output, dst, i * ELE_CNT_B64, NORM_B32, preg_b64);
        // sReg is decremented by CreatePredicate with POST_UPDATE
    }
}

/**
 * Cast 16-bit to 8-bit types - 1D version
 */
template <typename R, CastMode MODE, typename DST_VEC, typename DST, typename SRC>
inline AICORE void cast16to8_1D_NoPostUpdate(__ubuf__ DST *dst, __ubuf__ SRC *src, uint32_t validRows,
                                             uint32_t validCols, uint32_t dstCols, uint32_t srcCols,
                                             SaturationMode satMode)
{
    uint32_t totalElements = validRows * validCols;
    uint16_t repeatTimes = CeilDivision(totalElements, ELE_CNT_B16);
    uint32_t sReg = totalElements;
    uint32_t len16 = ELE_CNT_B16;
    MaskReg preg_b16 = CreatePredicate<half>(len16);

    for (uint16_t i = 0; i < repeatTimes; ++i) {
        RegTensor<SRC> v_input_0;
        DST_VEC v_output_even;
        MaskReg preg_b16_st = CreatePredicate<half>(sReg);

        vlds(v_input_0, src, i * ELE_CNT_B16, NORM);
        if constexpr (MODE == CastMode::ROUND_SAT_PART) {
            // Saturation controlled by CTRL register - always use RS_DISABLE
            vcvt(v_output_even, v_input_0, preg_b16, R(), RS_DISABLE, PART_EVEN);
        } else {
            // SAT_PART mode for int-to-int
            vcvt(v_output_even, v_input_0, preg_b16, RS_DISABLE, PART_EVEN);
        }
        vsts(v_output_even, dst, i * ELE_CNT_B16, PK_B16, preg_b16_st);
        // sReg is decremented by CreatePredicate with POST_UPDATE
    }
}

/**
 * Cast 32-bit to 8-bit types - 1D version
 *
 * IMPLEMENTATION NOTE: Uses vselr with index vector to extract bytes from 32-bit words.
 * The conversion happens in two steps:
 *   1. vcvt: Convert 32-bit source to target type (PART_P0 extracts low byte)
 *   2. vselr: Gather bytes using index vector for proper byte packing
 */
template <typename R, CastMode MODE, typename DST_VEC, typename DST, typename SRC>
inline AICORE void cast32to8_1D_NoPostUpdate(__ubuf__ DST *dst, __ubuf__ SRC *src, uint32_t validRows,
                                             uint32_t validCols, uint32_t dstCols, uint32_t srcCols,
                                             SaturationMode satMode)
{
    uint32_t totalElements = validRows * validCols;
    uint16_t repeatTimes = CeilDivision(totalElements, ELE_CNT_B32);
    uint32_t sReg = totalElements;
    MaskReg preg_idx = pset_b8(PAT_ALL);

    DST_VEC v_idx;
    vci((RegTensor<int8_t> &)v_idx, (int8_t)0, INC_ORDER);
    vmuls((RegTensor<int16_t> &)v_idx, (RegTensor<int16_t> &)v_idx, (int16_t)4, preg_idx);

    for (uint16_t i = 0; i < repeatTimes; ++i) {
        RegTensor<SRC> v_input;
        DST_VEC v_output_p0;
        uint32_t cur_len = sReg;
        MaskReg preg_b32 = CreatePredicate<float>(sReg);
        MaskReg preg_b8 = CreatePredicate<uint8_t>(cur_len);

        vlds(v_input, src, i * ELE_CNT_B32, NORM);

        if constexpr (MODE == CastMode::ROUND_SAT_PART) {
            vcvt(v_output_p0, v_input, preg_b32, ROUND_R, RS_DISABLE, PART_P0);
        } else {
            vcvt(v_output_p0, v_input, preg_b32, RS_DISABLE, PART_P0);
        }

        // Reuse v_input's preg for vselr output — guaranteed non-overlapping with v_output_p0
        vselr((RegTensor<uint8_t> &)v_input, (RegTensor<uint8_t> &)v_output_p0, (RegTensor<uint8_t> &)v_idx);
        mem_bar(VST_VST);
        vsts((RegTensor<uint8_t> &)v_input, (__ubuf__ uint8_t *)dst, i * ELE_CNT_B32, NORM_B8, preg_b8);
        // sReg is decremented by the first CreatePredicate with POST_UPDATE
    }
}

/**
 * Cast 32-bit to hifloat8 - 1D version
 *
 * SPECIAL HANDLING: H8 (hifloat8) requires ROUND_A (round away from zero) instead of
 * ROUND_R (round to nearest even) for correct IEEE-like behavior with 8-bit precision.
 * This is a hardware requirement specific to the hifloat8 format.
 */
template <typename R>
inline AICORE void cast32toH8_1D_NoPostUpdate(__ubuf__ hifloat8_t *dst, __ubuf__ float *src, uint32_t validRows,
                                              uint32_t validCols, uint32_t dstCols, uint32_t srcCols,
                                              SaturationMode satMode)
{
    uint32_t totalElements = validRows * validCols;
    uint16_t repeatTimes = CeilDivision(totalElements, ELE_CNT_B32);
    uint32_t sReg = totalElements;
    MaskReg preg_idx = pset_b8(PAT_ALL);

    vector_hif8 v_idx;
    vci((RegTensor<int8_t> &)v_idx, (int8_t)0, INC_ORDER);
    vmuls((RegTensor<int16_t> &)v_idx, (RegTensor<int16_t> &)v_idx, (int16_t)4, preg_idx);

    for (uint16_t i = 0; i < repeatTimes; ++i) {
        vector_f32 v_input;
        vector_hif8 v_output_p0;
        uint32_t cur_len = sReg;
        MaskReg preg_b32 = CreatePredicate<float>(sReg);
        MaskReg preg_b8 = CreatePredicate<uint8_t>(cur_len);

        vlds(v_input, src, i * ELE_CNT_B32, NORM);
        vcvt(v_output_p0, v_input, preg_b32, ROUND_A, RS_DISABLE, PART_P0);
        // Reuse v_input's preg for vselr output — guaranteed non-overlapping with v_output_p0
        // since vcvt requires them as separate source/dest pregs
        vselr((RegTensor<uint8_t> &)v_input, (RegTensor<uint8_t> &)v_output_p0, (RegTensor<uint8_t> &)v_idx);
        mem_bar(VST_VST);
        vsts((RegTensor<uint8_t> &)v_input, (__ubuf__ uint8_t *)dst, i * ELE_CNT_B32, NORM_B8, preg_b8);
        // sReg is decremented by CreatePredicate with POST_UPDATE
    }
}

/**
 * Cast 16-bit to hifloat8 - 1D version
 * Special version for H8 that uses ROUND_A instead of template parameter
 */
template <typename R>
inline AICORE void cast16toH8_1D_NoPostUpdate(__ubuf__ hifloat8_t *dst, __ubuf__ half *src, uint32_t validRows,
                                              uint32_t validCols, uint32_t dstCols, uint32_t srcCols,
                                              SaturationMode satMode)
{
    uint32_t totalElements = validRows * validCols;
    uint16_t repeatTimes = CeilDivision(totalElements, ELE_CNT_B16);
    uint32_t sReg = totalElements;
    uint32_t len16 = ELE_CNT_B16;
    MaskReg preg_b16 = CreatePredicate<half>(len16);

    for (uint16_t i = 0; i < repeatTimes; ++i) {
        vector_f16 v_input_0;
        vector_hif8 v_output_even;
        MaskReg preg_b16_st = CreatePredicate<half>(sReg);

        vlds(v_input_0, src, i * ELE_CNT_B16, NORM);
        vcvt(v_output_even, v_input_0, preg_b16, ROUND_A, RS_DISABLE, PART_EVEN);
        vsts((RegTensor<uint8_t> &)v_output_even, (__ubuf__ uint8_t *)dst, i * ELE_CNT_B16, PK_B16, preg_b16_st);
        // sReg is decremented by CreatePredicate with POST_UPDATE
    }
}

//=============================================================================================
// 2D Helper Templates - For non-contiguous data with padding
//=============================================================================================

/**
 * Cast 64-bit integer to 32-bit (signed/float) - 2D version
 * Handles: s64 -> s32 #sat #part, s64 -> f32 #rnd #part
 * Intrinsics:
 *   vcvt(output, input, preg, RS_DISABLE, PART_EVEN)  // s64 -> s32 with saturation
 *   vcvt(output, input, preg, R(), PART_EVEN)        // s64 -> f32 with rounding
 */
template <typename R, typename DST, typename SRC>
inline AICORE void castS64to32(__ubuf__ DST *dst, __ubuf__ SRC *src, uint32_t validRows, uint32_t validCols,
                               uint32_t dstCols, uint32_t srcCols, SaturationMode satMode)
{
    vector_s64 v_input_0;

    const uint32_t ELE_CNT_B64 = ELE_CNT_B32 / 2;

    FOR_ROWS
    uint32_t len64 = sreg * 2; // As we operate with 64bit blocks using 32bit operations
    MaskReg preg_b64 = CreatePredicate<float>(len64);
    uint32_t len_even = sreg * 2; // As only the even part is taken

    FOR_ELEMENTS(ELE_CNT_B64)
    RegTensor<DST> v_output;
    MaskReg preg_b32 = CreatePredicate<float>(len_even);

    vlds(v_input_0, src, srcOffset, NORM);
    if constexpr (std::is_same<R, void>::value) {
        // For type expansion (s64->s32/f32), saturation mode is controllable
        vcvt(v_output, v_input_0, preg_b64, RS_DISABLE, PART_EVEN);
    } else {
        vcvt(v_output, v_input_0, preg_b64, R(), PART_EVEN);
    }
    vsts(v_output, dst, dstOffset, PK_B64, preg_b32);
    END_FOR_ELEMENTS
    END_FOR_ROWS
}

/**
 * Cast 32-bit to 16-bit types
 * Handles: f32 -> f16 #rnd #sat #part, f32 -> bf16 #rnd #sat #part, f32 -> s16 #rnd #sat #part
 * Intrinsics:
 *   vcvt(out_odd, in_1, preg, RS_DISABLE, PART_ODD/EVEN)       // No rounding mode (saturation only)
 *   vcvt(out_odd, in_1, preg, R(), RS_DISABLE, PART_ODD/EVEN)  // With rounding mode
 */
template <typename R, typename DST, typename SRC>
inline AICORE void cast32to16(__ubuf__ DST *dst, __ubuf__ SRC *src, uint32_t validRows, uint32_t validCols,
                              uint32_t dstCols, uint32_t srcCols, SaturationMode satMode)
{
    uint32_t len32 = ELE_CNT_B32;
    MaskReg preg_b32 = CreatePredicate<float>(len32);

    FOR_ROWS
    FOR_ELEMENTS(ELE_CNT_B16)
    RegTensor<SRC> v_input_0, v_input_1;
    RegTensor<DST> v_output_odd, v_output_even, v_output;
    MaskReg preg_b16 = CreatePredicate<half>(sreg);

    vlds(v_input_0, v_input_1, src, srcOffset, DINTLV_B32);
    if constexpr (std::is_same<R, void>::value) {
        vcvt(v_output_odd, v_input_1, preg_b32, RS_DISABLE, PART_ODD);
        vcvt(v_output_even, v_input_0, preg_b32, RS_DISABLE, PART_EVEN);
    } else {
        vcvt(v_output_odd, v_input_1, preg_b32, R(), RS_DISABLE, PART_ODD);
        vcvt(v_output_even, v_input_0, preg_b32, R(), RS_DISABLE, PART_EVEN);
    }
    vor(v_output, v_output_even, v_output_odd, preg_b16);
    vsts(v_output, dst, dstOffset, NORM_B16, preg_b16);
    END_FOR_ELEMENTS
    END_FOR_ROWS
}

/**
 * Cast 32-bit to 16-bit types 2D without interleave version for better fusion
 * Handles: f32 -> f16 #rnd #sat #part, f32 -> bf16 #rnd #sat #part, f32 -> s16 #rnd #sat #part
 * Intrinsics:
 *   vcvt(out_odd, in_1, preg, RS_DISABLE, PART_ODD/EVEN)       // No rounding mode (saturation only)
 *   vcvt(out_odd, in_1, preg, R(), RS_DISABLE, PART_ODD/EVEN)  // With rounding mode
 */
template <typename R, typename DST, typename SRC>
inline AICORE void cast32to16_2D_NoPostUpdate(__ubuf__ DST *dst, __ubuf__ SRC *src, uint32_t validRows,
                                              uint32_t validCols, uint32_t dstCols, uint32_t srcCols,
                                              SaturationMode satMode)
{
    uint32_t len32 = ELE_CNT_B32;
    MaskReg preg_b32 = CreatePredicate<float>(len32);

    FOR_ROWS
    FOR_ELEMENTS(ELE_CNT_B32)
    RegTensor<SRC> v_input_0, v_input_1;
    RegTensor<DST> v_output_odd, v_output_even, v_output;
    MaskReg preg_b32_st = CreatePredicate<float>(sreg);

    vlds(v_input_0, src, srcOffset, NORM);
    if constexpr (std::is_same<R, void>::value) {
        vcvt(v_output_even, v_input_0, preg_b32, RS_DISABLE, PART_EVEN);
    } else {
        vcvt(v_output_even, v_input_0, preg_b32, R(), RS_DISABLE, PART_EVEN);
    }
    vsts(v_output_even, dst, dstOffset, PK_B32, preg_b32_st);
    END_FOR_ELEMENTS
    END_FOR_ROWS
}

/**
 * Cast 32-bit to 64-bit signed integer
 * Handles: s32 -> s64 #part, f32 -> s64 #rnd #sat #part
 * Intrinsics:
 *   vcvt(output, input, preg, PART_EVEN)                    // s32 -> s64 (type expansion)
 *   vcvt(output, input, preg, R(), RS_DISABLE, PART_EVEN)    // f32 -> s64 (with rounding and saturation)
 */
template <typename R, typename SRC>
inline AICORE void cast32toS64(__ubuf__ int64_t *dst, __ubuf__ SRC *src, uint32_t validRows, uint32_t validCols,
                               uint32_t dstCols, uint32_t srcCols, SaturationMode satMode)
{
    const uint32_t ELE_CNT_B64 = ELE_CNT_B32 / 2;
    uint32_t len32 = ELE_CNT_B32;
    MaskReg preg_b32 = CreatePredicate<float>(len32);

    FOR_ROWS
    uint32_t len64 = sreg * 2; // As we operate with 64bit blocks using 32bit operations
    FOR_ELEMENTS(ELE_CNT_B64)
    RegTensor<SRC> v_input_0;
    vector_s64 v_output;

    MaskReg preg_b64 = CreatePredicate<float>(len64);

    vlds(v_input_0, src, srcOffset, UNPK_B32);
    if constexpr (std::is_same<R, void>::value) {
        vcvt(v_output, v_input_0, preg_b32, PART_EVEN);
    } else {
        // For conversions with rounding (e.g., f32->s64), saturation mode controlled by CTRL register
        vcvt(v_output, v_input_0, preg_b32, R(), RS_DISABLE, PART_EVEN);
    }
    vsts(v_output, dst, dstOffset, NORM_B32, preg_b64);
    END_FOR_ELEMENTS
    END_FOR_ROWS
}

/**
 * Cast 16-bit to 8-bit types
 * Modes:
 *   ROUND_SAT_PART: f16 -> s8/u8 #rnd #sat #part → vcvt(..., R(), RS_DISABLE, PART_*)
 *   SAT_PART:       s16 -> u8 #sat #part         → vcvt(..., RS_DISABLE, PART_*)
 */
template <typename R, CastMode MODE, typename DST_VEC, typename DST, typename SRC>
inline AICORE void cast16to8(__ubuf__ DST *dst, __ubuf__ SRC *src, uint32_t validRows, uint32_t validCols,
                             uint32_t dstCols, uint32_t srcCols, SaturationMode satMode)
{
    uint32_t len16 = ELE_CNT_B16;
    MaskReg preg_b16 = CreatePredicate<half>(len16);

    FOR_ROWS
    FOR_ELEMENTS(ELE_CNT_B8)
    RegTensor<SRC> v_input_0, v_input_1;
    DST_VEC v_output_odd, v_output_even, v_output;
    MaskReg preg_b8 = CreatePredicate<uint8_t>(sreg);

    vlds(v_input_0, v_input_1, src, srcOffset, DINTLV_B16);
    if constexpr (MODE == CastMode::ROUND_SAT_PART) {
        // Use rounding with saturation controlled by CTRL register
        vcvt(v_output_odd, v_input_1, preg_b16, R(), RS_DISABLE, PART_ODD);
        vcvt(v_output_even, v_input_0, preg_b16, R(), RS_DISABLE, PART_EVEN);
    } else {
        // SAT_PART mode: saturation without rounding (integer->integer)
        vcvt(v_output_odd, v_input_1, preg_b16, RS_DISABLE, PART_ODD);
        vcvt(v_output_even, v_input_0, preg_b16, RS_DISABLE, PART_EVEN);
    }
    vor(v_output, v_output_even, v_output_odd, preg_b8);
    vsts(v_output, dst, dstOffset, NORM_B8, preg_b8);
    END_FOR_ELEMENTS
    END_FOR_ROWS
}

/**
 * Cast 16-bit to 8-bit types 2D without interleave version for better fusion
 * Modes:
 *   ROUND_SAT_PART: f16 -> s8/u8 #rnd #sat #part → vcvt(..., R(), RS_DISABLE, PART_EVEN)
 *   SAT_PART:       s16 -> u8 #sat #part         → vcvt(..., RS_DISABLE, PART_EVEN)
 */
template <typename R, CastMode MODE, typename DST_VEC, typename DST, typename SRC>
inline AICORE void cast16to8_2D_NoPostUpdate(__ubuf__ DST *dst, __ubuf__ SRC *src, uint32_t validRows,
                                             uint32_t validCols, uint32_t dstCols, uint32_t srcCols,
                                             SaturationMode satMode)
{
    uint32_t len16 = ELE_CNT_B16;
    MaskReg preg_b16 = CreatePredicate<half>(len16);

    FOR_ROWS
    FOR_ELEMENTS(ELE_CNT_B16)
    RegTensor<SRC> v_input_0;
    DST_VEC v_output_even;
    MaskReg preg_b16_st = CreatePredicate<half>(sreg);

    vlds(v_input_0, src, srcOffset, NORM);
    if constexpr (MODE == CastMode::ROUND_SAT_PART) {
        vcvt(v_output_even, v_input_0, preg_b16, R(), RS_DISABLE, PART_EVEN);
    } else {
        // SAT_PART mode: s16 -> u8
        vcvt(v_output_even, v_input_0, preg_b16, RS_DISABLE, PART_EVEN);
    }
    vsts(v_output_even, dst, dstOffset, PK_B16, preg_b16_st);
    END_FOR_ELEMENTS
    END_FOR_ROWS
}

/**
 * Cast 32-bit to 8-bit types (both floating point and integer)
 * Handles:
 *   - f32 -> e4m3/e5m2/h8 #rnd #sat #part (ROUND_SAT_PART mode)
 *   - u32/s32 -> u8/s8 #sat #part (SAT_PART mode)
 * Intrinsics:
 *   vcvt(..., R(), RS_DISABLE, PART_P0) for floating point with rounding
 *   vcvt(..., RS_DISABLE, PART_P0) for integer without rounding
 */
template <typename R, CastMode MODE, typename DST_VEC, typename DST, typename SRC>
inline AICORE void cast32to8(__ubuf__ DST *dst, __ubuf__ SRC *src, uint32_t validRows, uint32_t validCols,
                             uint32_t dstCols, uint32_t srcCols, SaturationMode satMode)
{
    uint32_t len32 = ELE_CNT_B32;
    MaskReg preg_b32 = CreatePredicate<float>(len32);
    MaskReg preg_idx = pset_b8(PAT_ALL);

    // Create index vector for vselr (selecting every 4th byte)
    DST_VEC v_idx;
    vci((RegTensor<int8_t> &)v_idx, (int8_t)0, INC_ORDER);
    vmuls((RegTensor<int16_t> &)v_idx, (RegTensor<int16_t> &)v_idx, (int16_t)4, preg_idx);

    FOR_ROWS
    uint32_t preg_len_tail = (sreg % ELE_CNT_B32 == 0) ? ELE_CNT_B32 : (sreg % ELE_CNT_B32);

    FOR_ELEMENTS(ELE_CNT_B32)
    RegTensor<SRC> v_input;
    DST_VEC v_output_p0;
    uint32_t preg_len = (idx == repeatTimes - 1) ? preg_len_tail : ELE_CNT_B32;
    MaskReg preg_b8 = CreatePredicate<uint8_t>(preg_len);

    vlds(v_input, src, srcOffset, NORM);

    // Convert with or without rounding based on mode - saturation controlled by CTRL register
    if constexpr (MODE == CastMode::ROUND_SAT_PART) {
        vcvt(v_output_p0, v_input, preg_b32, ROUND_R, RS_DISABLE, PART_P0);
    } else {
        vcvt(v_output_p0, v_input, preg_b32, RS_DISABLE, PART_P0);
    }

    // Reuse v_input's preg for vselr output — guaranteed non-overlapping with v_output_p0
    vselr((RegTensor<uint8_t> &)v_input, (RegTensor<uint8_t> &)v_output_p0, (RegTensor<uint8_t> &)v_idx);
    mem_bar(VST_VST);
    vsts((RegTensor<uint8_t> &)v_input, (__ubuf__ uint8_t *)dst, dstOffset, NORM_B8, preg_b8);
    END_FOR_ELEMENTS
    END_FOR_ROWS
}

//=============================================================================================
// castData Overloads (2D - with row/column iteration for non-contiguous data)
//=============================================================================================
// These are the main conversion functions organized by source type for easy navigation.
// Each source type section contains conversions to all supported destination types.
//
// ORGANIZATION: Grouped by source type in ascending bit-width order (8→16→32→64-bit)
// WHY: This ordering provides quick lookup - if you know the source type, you can
// jump directly to its section and find all target conversions in one place.

//---------------------------------------------------------------------------------------------
// Source: FP32 (float) - 2D versions
//---------------------------------------------------------------------------------------------

/**
 * FP32 to BF16
 * Conversion: f32 -> bf16 #rnd #sat #part
 * Uses cast32to16 helper
 */
template <typename R>
inline AICORE void castData(__ubuf__ bfloat16_t *dst, __ubuf__ float *src, uint32_t validRows, uint32_t validCols,
                            uint32_t dstCols, uint32_t srcCols, SaturationMode satMode)
{
    cast32to16<R>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

template <typename R>
inline AICORE void castData_2D_NoPostUpdate(__ubuf__ bfloat16_t *dst, __ubuf__ float *src, uint32_t validRows,
                                            uint32_t validCols, uint32_t dstCols, uint32_t srcCols,
                                            SaturationMode satMode)
{
    cast32to16_2D_NoPostUpdate<R>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

/**
 * FP32 to I64
 * Conversion: f32 -> s64 #rnd #sat #part
 * Uses cast32toS64 helper
 */
template <typename R>
inline AICORE void castData(__ubuf__ int64_t *dst, __ubuf__ float *src, uint32_t validRows, uint32_t validCols,
                            uint32_t dstCols, uint32_t srcCols, SaturationMode satMode)
{
    cast32toS64<R>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

template <typename R>
inline AICORE void castData_2D_NoPostUpdate(__ubuf__ int64_t *dst, __ubuf__ float *src, uint32_t validRows,
                                            uint32_t validCols, uint32_t dstCols, uint32_t srcCols,
                                            SaturationMode satMode)
{
    cast32toS64<R>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

/**
 * FP32 to FP8_E4M3
 * Conversion: f32 -> e4m3 #rnd #sat #pp
 * Uses cast32to8 helper
 */
template <typename R>
inline AICORE void castData(__ubuf__ float8_e4m3_t *dst, __ubuf__ float *src, uint32_t validRows, uint32_t validCols,
                            uint32_t dstCols, uint32_t srcCols, SaturationMode satMode)
{
    cast32to8<R, CastMode::ROUND_SAT_PART, vector_f8e4m3>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

template <typename R>
inline AICORE void castData_2D_NoPostUpdate(__ubuf__ float8_e4m3_t *dst, __ubuf__ float *src, uint32_t validRows,
                                            uint32_t validCols, uint32_t dstCols, uint32_t srcCols,
                                            SaturationMode satMode)
{
    cast32to8<R, CastMode::ROUND_SAT_PART, vector_f8e4m3>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

/**
 * FP32 to FP8_E5M2
 * Conversion: f32 -> e5m2 #rnd #sat #pp
 * Uses cast32to8 helper
 */
template <typename R>
inline AICORE void castData(__ubuf__ float8_e5m2_t *dst, __ubuf__ float *src, uint32_t validRows, uint32_t validCols,
                            uint32_t dstCols, uint32_t srcCols, SaturationMode satMode)
{
    cast32to8<R, CastMode::ROUND_SAT_PART, vector_f8e5m2>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

template <typename R>
inline AICORE void castData_2D_NoPostUpdate(__ubuf__ float8_e5m2_t *dst, __ubuf__ float *src, uint32_t validRows,
                                            uint32_t validCols, uint32_t dstCols, uint32_t srcCols,
                                            SaturationMode satMode)
{
    cast32to8<R, CastMode::ROUND_SAT_PART, vector_f8e5m2>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

/**
 * FP32 to H8
 * Conversion: f32 -> h8 #rnd #sat #part
 * Note: H8 conversion requires ROUND_A mode
 */
template <typename R>
inline AICORE void castData(__ubuf__ hifloat8_t *dst, __ubuf__ float *src, uint32_t validRows, uint32_t validCols,
                            uint32_t dstCols, uint32_t srcCols, SaturationMode satMode)
{
    uint32_t len32 = ELE_CNT_B32;
    MaskReg preg_b32 = CreatePredicate<float>(len32);
    MaskReg preg_idx = pset_b8(PAT_ALL);

    // Create index vector for vselr (selecting every 4th byte)
    vector_u8 v_idx;
    vci((RegTensor<int8_t> &)v_idx, (int8_t)0, INC_ORDER);
    vmuls((RegTensor<int16_t> &)v_idx, (RegTensor<int16_t> &)v_idx, (int16_t)4, preg_idx);

    FOR_ROWS
    uint32_t preg_len_tail = (sreg % ELE_CNT_B32 == 0) ? ELE_CNT_B32 : (sreg % ELE_CNT_B32);

    FOR_ELEMENTS(ELE_CNT_B32)
    vector_f32 v_input;
    vector_hif8 v_output_p0;
    uint32_t preg_len = (idx == repeatTimes - 1) ? preg_len_tail : ELE_CNT_B32;
    MaskReg preg_b8 = CreatePredicate<uint8_t>(preg_len);

    vlds(v_input, src, srcOffset, NORM);
    vcvt(v_output_p0, v_input, preg_b32, ROUND_A, RS_DISABLE, PART_P0);

    // Reuse v_input's preg for vselr output — guaranteed non-overlapping with v_output_p0
    vselr((RegTensor<uint8_t> &)v_input, (RegTensor<uint8_t> &)v_output_p0, (RegTensor<uint8_t> &)v_idx);
    mem_bar(VST_VST);
    vsts((RegTensor<uint8_t> &)v_input, (__ubuf__ uint8_t *)dst, dstOffset, NORM_B8, preg_b8);
    END_FOR_ELEMENTS
    END_FOR_ROWS
}

template <typename R>
inline AICORE void castData_2D_NoPostUpdate(__ubuf__ hifloat8_t *dst, __ubuf__ float *src, uint32_t validRows,
                                            uint32_t validCols, uint32_t dstCols, uint32_t srcCols,
                                            SaturationMode satMode)
{
    // Same complex logic as castData - just reuse it
    castData<R>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

//---------------------------------------------------------------------------------------------
// Source: FP16 (half) - 2D versions
//---------------------------------------------------------------------------------------------

/** FP16 -> H8 #rnd #sat #part */
template <typename R>
inline AICORE void castData(__ubuf__ hifloat8_t *dst, __ubuf__ half *src, uint32_t validRows, uint32_t validCols,
                            uint32_t dstCols, uint32_t srcCols, SaturationMode satMode)
{
    // FP16->H8 conversion only supports ROUND_A or ROUND_H modes
    uint32_t len16 = ELE_CNT_B16;
    MaskReg preg_b16 = CreatePredicate<half>(len16);

    FOR_ROWS
    FOR_ELEMENTS(ELE_CNT_B8)
    vector_f16 v_input_0, v_input_1;
    vector_hif8 v_output_odd, v_output_even, v_output;
    MaskReg preg_b8 = CreatePredicate<uint8_t>(sreg);

    vlds(v_input_0, v_input_1, src, srcOffset, DINTLV_B16);
    vcvt(v_output_odd, v_input_1, preg_b16, ROUND_A, RS_DISABLE, PART_ODD);
    vcvt(v_output_even, v_input_0, preg_b16, ROUND_A, RS_DISABLE, PART_EVEN);
    vor((RegTensor<uint8_t> &)v_output, (RegTensor<uint8_t> &)v_output_even, (RegTensor<uint8_t> &)v_output_odd,
        preg_b8);
    vsts((RegTensor<uint8_t> &)v_output, (__ubuf__ uint8_t *)dst, dstOffset, NORM_B8, preg_b8);
    END_FOR_ELEMENTS
    END_FOR_ROWS
}

template <typename R>
inline AICORE void castData_2D_NoPostUpdate(__ubuf__ hifloat8_t *dst, __ubuf__ half *src, uint32_t validRows,
                                            uint32_t validCols, uint32_t dstCols, uint32_t srcCols,
                                            SaturationMode satMode)
{
    // Same complex logic as castData - just reuse it
    castData<R>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

//---------------------------------------------------------------------------------------------
// Source: BFloat16 - 2D versions
//---------------------------------------------------------------------------------------------

/** BF16 -> FP32 #part (type expansion) → vcvt(output, input, preg, PART_EVEN) */
template <typename R>
inline AICORE void castData(__ubuf__ float *dst, __ubuf__ bfloat16_t *src, uint32_t validRows, uint32_t validCols,
                            uint32_t dstCols, uint32_t srcCols, SaturationMode satMode)
{
    cast16to32<void, CastMode::EXPAND>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

template <typename R>
inline AICORE void castData_2D_NoPostUpdate(__ubuf__ float *dst, __ubuf__ bfloat16_t *src, uint32_t validRows,
                                            uint32_t validCols, uint32_t dstCols, uint32_t srcCols,
                                            SaturationMode satMode)
{
    cast16to32<void, CastMode::EXPAND>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

/** BF16 -> I32 #rnd #sat #part → vcvt(output, input, preg, R(), RS_DISABLE, PART_EVEN) */
template <typename R>
inline AICORE void castData(__ubuf__ int32_t *dst, __ubuf__ bfloat16_t *src, uint32_t validRows, uint32_t validCols,
                            uint32_t dstCols, uint32_t srcCols, SaturationMode satMode)
{
    cast16to32<R, CastMode::ROUND_SAT_PART>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

template <typename R>
inline AICORE void castData_2D_NoPostUpdate(__ubuf__ int32_t *dst, __ubuf__ bfloat16_t *src, uint32_t validRows,
                                            uint32_t validCols, uint32_t dstCols, uint32_t srcCols,
                                            SaturationMode satMode)
{
    cast16to32<R, CastMode::ROUND_SAT_PART>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

/** BF16 -> F16 #sat #rnd → vcvt(output, input, preg, RS_DISABLE, R()) [reversed order] */
template <typename R>
inline AICORE void castData(__ubuf__ half *dst, __ubuf__ bfloat16_t *src, uint32_t validRows, uint32_t validCols,
                            uint32_t dstCols, uint32_t srcCols, SaturationMode satMode)
{
    cast16to16<R, CastMode::SAT_ROUND>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

template <typename R>
inline AICORE void castData_2D_NoPostUpdate(__ubuf__ half *dst, __ubuf__ bfloat16_t *src, uint32_t validRows,
                                            uint32_t validCols, uint32_t dstCols, uint32_t srcCols,
                                            SaturationMode satMode)
{
    cast16to16<R, CastMode::SAT_ROUND>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

/**
 * BF16 to FP4 (packed x2) conversion helpers
 *
 * FP4 is 4-bit: columns are counted as nibbles, stored 2-per-byte in float4_e*x2_t.
 * Intrinsic: vcvt(dst, src, preg, R(), PART_P0) — same 4:1 bit-ratio as FP32→FP8.
 * Output requires vselr (select every 4th byte) to compact scattered result, and
 * all dst byte offsets are >> 1 (nibble count → byte count).
 */
template <typename R, typename DST_VEC, typename DST>
inline AICORE void castBf16toFp4(__ubuf__ DST *dst, __ubuf__ bfloat16_t *src, uint32_t validRows, uint32_t validCols,
                                 uint32_t dstCols, uint32_t srcCols)
{
    uint32_t len16 = ELE_CNT_B16;
    MaskReg preg_b16 = CreatePredicate<half>(len16);
    MaskReg preg_idx = pset_b8(PAT_ALL);

    DST_VEC v_idx;
    vci((RegTensor<int8_t> &)v_idx, (int8_t)0, INC_ORDER);
    vmuls((RegTensor<int16_t> &)v_idx, (RegTensor<int16_t> &)v_idx, (int16_t)4, preg_idx);

    // Zero-fill destination to clear padding bytes (UB is uninitialized on hardware).
    // BF16→FP4 writes only validCols/2 complete packed bytes per row; bytes beyond
    // that (including the boundary byte for odd validCols) must be zero.
    {
        RegTensor<uint8_t> v_zeros;
        MaskReg pg_fill = pset_b8(PAT_ALL);
        vdup(v_zeros, (uint8_t)0, pg_fill, MODE_ZEROING);
        uint32_t totalDstBytes = validRows * (dstCols >> 1);
        uint32_t fillLen = totalDstBytes;
        uint16_t fillRepeats = CeilDivision(totalDstBytes, static_cast<uint32_t>(ELE_CNT_B8));
        for (uint16_t fi = 0; fi < fillRepeats; ++fi) {
            MaskReg preg_fill = CreatePredicate<uint8_t>(fillLen);
            vsts(v_zeros, (__ubuf__ uint8_t *)dst, fi * ELE_CNT_B8, NORM_B8, preg_fill);
        }
    }

    FOR_ROWS
    uint32_t preg_len_tail = (sreg % ELE_CNT_B16 == 0) ? ELE_CNT_B16 : (sreg % ELE_CNT_B16);
    FOR_ELEMENTS(ELE_CNT_B16)
    RegTensor<bfloat16_t> v_input;
    DST_VEC v_output_p0, v_output;
    uint32_t preg_len = (idx == repeatTimes - 1) ? preg_len_tail : ELE_CNT_B16;
    uint32_t byte_len = preg_len >> 1; // nibbles → FP4x2 bytes
    MaskReg preg_b8 = CreatePredicate<uint8_t>(byte_len);

    vlds(v_input, src, srcOffset, NORM);
    vcvt(v_output_p0, v_input, preg_b16, R(), PART_P0);
    vselr((RegTensor<uint8_t> &)v_output, (RegTensor<uint8_t> &)v_output_p0, (RegTensor<uint8_t> &)v_idx);
    mem_bar(VST_VST);
    vsts((RegTensor<uint8_t> &)v_output, (__ubuf__ uint8_t *)dst, dstOffset >> 1, NORM_B8, preg_b8);
    END_FOR_ELEMENTS
    END_FOR_ROWS
}

template <typename R, typename DST_VEC, typename DST>
inline AICORE void castBf16toFp4_2D_NoPostUpdate(__ubuf__ DST *dst, __ubuf__ bfloat16_t *src, uint32_t validRows,
                                                 uint32_t validCols, uint32_t dstCols, uint32_t srcCols)
{
    // Same pass-count as castBf16toFp4 (no interleaved dual-load variant for FP4)
    castBf16toFp4<R, DST_VEC>(dst, src, validRows, validCols, dstCols, srcCols);
}

/**
 * FP4 (packed x2) to BF16 conversion helpers
 *
 * FP4 is 4-bit: columns are counted as nibbles, stored 2-per-byte in float4_e*x2_t.
 * Uses UNPK_B8 + vintlv pattern (same as cast8to32 for FP8→FP32) since the byte
 * expansion ratio is the same (1:4): 1 packed FP4x2 byte → 2 BF16 values = 4 bytes.
 * Intrinsic: vcvt(dst_bf16, src_fp4, preg, PART_P0).
 */
template <typename SRC_VEC, typename DST, typename SRC>
inline AICORE void castFp4toBf16(__ubuf__ DST *dst, __ubuf__ SRC *src, uint32_t validRows, uint32_t validCols,
                                 uint32_t dstCols, uint32_t srcCols)
{
    uint32_t len8 = ELE_CNT_B8;
    MaskReg preg_b8 = CreatePredicate<uint8_t>(len8);
    MaskReg pg = pset_b8(PAT_ALL);
    SRC_VEC v_zero;
    vdup((RegTensor<uint8_t> &)v_zero, 0, pg, MODE_ZEROING);

    for (uint16_t row = 0; row < validRows; row++) {
        int32_t rowSrcByteOffset = (row * srcCols) >> 1;
        int32_t rowDstOffset = row * dstCols;
        uint32_t sreg = validCols;
        uint16_t repeatTimes = CeilDivision(sreg, static_cast<uint32_t>(ELE_CNT_B16 * 2));
        uint32_t next_len = (sreg > ELE_CNT_B16) ? sreg - ELE_CNT_B16 : 0;

        for (uint16_t idx = 0; idx < repeatTimes; idx++) {
            SRC_VEC v_input_0, v_input_1, v_input_2;
            RegTensor<DST> v_output_0, v_output_1;
            MaskReg preg_b16_cur = CreatePredicate<half>(sreg);
            MaskReg preg_b16_next = CreatePredicate<half>(next_len);

            vlds((RegTensor<uint8_t> &)v_input_0, (__ubuf__ uint8_t *)src, rowSrcByteOffset + idx * ELE_CNT_B16,
                 UNPK_B8);
            vintlv((RegTensor<uint8_t> &)v_input_1, (RegTensor<uint8_t> &)v_input_2, (RegTensor<uint8_t> &)v_input_0,
                   (RegTensor<uint8_t> &)v_zero);
            vcvt(v_output_0, v_input_1, preg_b8, PART_P0);
            vcvt(v_output_1, v_input_2, preg_b8, PART_P0);
            vsts(v_output_0, dst, rowDstOffset + ELE_CNT_B16 * (idx * 2), NORM_B16, preg_b16_cur);
            vsts(v_output_1, dst, rowDstOffset + ELE_CNT_B16 * (idx * 2 + 1), NORM_B16, preg_b16_next);
        }
    }
}

template <typename SRC_VEC, typename DST, typename SRC>
inline AICORE void castFp4toBf16_2D_NoPostUpdate(__ubuf__ DST *dst, __ubuf__ SRC *src, uint32_t validRows,
                                                 uint32_t validCols, uint32_t dstCols, uint32_t srcCols)
{
    castFp4toBf16<SRC_VEC>(dst, src, validRows, validCols, dstCols, srcCols);
}

template <typename SRC_VEC, typename DST, typename SRC>
inline AICORE void castFp4toBf16_1D_NoPostUpdate(__ubuf__ DST *dst, __ubuf__ SRC *src, uint32_t validRows,
                                                 uint32_t validCols, uint32_t dstCols, uint32_t srcCols)
{
    uint32_t totalElements = validRows * validCols;
    uint16_t repeatTimes = CeilDivision(totalElements, static_cast<uint32_t>(ELE_CNT_B16 * 2));
    uint32_t sReg = totalElements;
    uint32_t next_len = (sReg > ELE_CNT_B16) ? sReg - ELE_CNT_B16 : 0;
    uint32_t len8 = ELE_CNT_B8;
    MaskReg preg_b8 = CreatePredicate<uint8_t>(len8);
    MaskReg pg = pset_b8(PAT_ALL);
    SRC_VEC v_zero;
    vdup((RegTensor<uint8_t> &)v_zero, 0, pg, MODE_ZEROING);

    for (uint16_t i = 0; i < repeatTimes; ++i) {
        SRC_VEC v_input_0, v_input_1, v_input_2;
        RegTensor<DST> v_output_0, v_output_1;
        MaskReg preg_b16_cur = CreatePredicate<half>(sReg);
        MaskReg preg_b16_next = CreatePredicate<half>(next_len);

        vlds((RegTensor<uint8_t> &)v_input_0, (__ubuf__ uint8_t *)src, i * ELE_CNT_B16, UNPK_B8);
        vintlv((RegTensor<uint8_t> &)v_input_1, (RegTensor<uint8_t> &)v_input_2, (RegTensor<uint8_t> &)v_input_0,
               (RegTensor<uint8_t> &)v_zero);
        vcvt(v_output_0, v_input_1, preg_b8, PART_P0);
        vcvt(v_output_1, v_input_2, preg_b8, PART_P0);
        vsts(v_output_0, dst, ELE_CNT_B16 * (i * 2), NORM_B16, preg_b16_cur);
        vsts(v_output_1, dst, ELE_CNT_B16 * (i * 2 + 1), NORM_B16, preg_b16_next);
    }
}

template <typename R, typename DST_VEC, typename DST>
inline AICORE void castBf16toFp4_1D_NoPostUpdate(__ubuf__ DST *dst, __ubuf__ bfloat16_t *src, uint32_t validRows,
                                                 uint32_t validCols, uint32_t dstCols, uint32_t srcCols)
{
    uint32_t totalElements = validRows * validCols; // counted in nibbles
    uint16_t repeatTimes = CeilDivision(totalElements, ELE_CNT_B16);
    uint32_t sReg = totalElements;
    MaskReg preg_idx = pset_b8(PAT_ALL);

    DST_VEC v_idx;
    vci((RegTensor<int8_t> &)v_idx, (int8_t)0, INC_ORDER);
    vmuls((RegTensor<int16_t> &)v_idx, (RegTensor<int16_t> &)v_idx, (int16_t)4, preg_idx);

    // Zero-fill destination to clear padding bytes (UB is uninitialized on hardware).
    // BF16→FP4 writes only complete packed bytes; the boundary byte for odd element
    // counts and all trailing bytes must be zero.
    {
        MaskReg pg_fill = pset_b8(PAT_ALL);
        RegTensor<uint8_t> v_zeros;
        vdup(v_zeros, (uint8_t)0, pg_fill, MODE_ZEROING);
        uint32_t totalDstBytes = validRows * (dstCols >> 1);
        uint16_t fillRepeats = CeilDivision(totalDstBytes, static_cast<uint32_t>(ELE_CNT_B8));
        uint32_t fillLen = totalDstBytes;
        for (uint16_t fi = 0; fi < fillRepeats; ++fi) {
            MaskReg preg_fill = CreatePredicate<uint8_t>(fillLen);
            vsts(v_zeros, (__ubuf__ uint8_t *)dst, fi * ELE_CNT_B8, NORM_B8, preg_fill);
        }
    }

    for (uint16_t i = 0; i < repeatTimes; ++i) {
        RegTensor<bfloat16_t> v_input;
        DST_VEC v_output_p0, v_output;
        uint32_t cur_len = sReg;
        uint32_t byte_len = cur_len >> 1;               // nibbles → FP4x2 bytes
        MaskReg preg_b16 = CreatePredicate<half>(sReg); // post-updates sReg
        MaskReg preg_b8 = CreatePredicate<uint8_t>(byte_len);

        vlds(v_input, src, i * ELE_CNT_B16, NORM);
        vcvt(v_output_p0, v_input, preg_b16, R(), PART_P0);
        vselr((RegTensor<uint8_t> &)v_output, (RegTensor<uint8_t> &)v_output_p0, (RegTensor<uint8_t> &)v_idx);
        mem_bar(VST_VST);
        vsts((RegTensor<uint8_t> &)v_output, (__ubuf__ uint8_t *)dst, (i * ELE_CNT_B16) >> 1, NORM_B8, preg_b8);
        // sReg is decremented by CreatePredicate<half> with POST_UPDATE
    }
}

/** BF16 -> FP4_E1M2X2 #rnd #sat #part */
template <typename R>
inline AICORE void castData(__ubuf__ float4_e1m2x2_t *dst, __ubuf__ bfloat16_t *src, uint32_t validRows,
                            uint32_t validCols, uint32_t dstCols, uint32_t srcCols,
                            SaturationMode satMode = SaturationMode::ON)
{
    castBf16toFp4<R, vector_f4e1m2x2>(dst, src, validRows, validCols, dstCols, srcCols);
}

template <typename R>
inline AICORE void castData_2D_NoPostUpdate(__ubuf__ float4_e1m2x2_t *dst, __ubuf__ bfloat16_t *src, uint32_t validRows,
                                            uint32_t validCols, uint32_t dstCols, uint32_t srcCols,
                                            SaturationMode satMode = SaturationMode::ON)
{
    castBf16toFp4_2D_NoPostUpdate<R, vector_f4e1m2x2>(dst, src, validRows, validCols, dstCols, srcCols);
}

/** BF16 -> FP4_E2M1X2 #rnd #sat #part */
template <typename R>
inline AICORE void castData(__ubuf__ float4_e2m1x2_t *dst, __ubuf__ bfloat16_t *src, uint32_t validRows,
                            uint32_t validCols, uint32_t dstCols, uint32_t srcCols,
                            SaturationMode satMode = SaturationMode::ON)
{
    castBf16toFp4<R, vector_f4e2m1x2>(dst, src, validRows, validCols, dstCols, srcCols);
}

template <typename R>
inline AICORE void castData_2D_NoPostUpdate(__ubuf__ float4_e2m1x2_t *dst, __ubuf__ bfloat16_t *src, uint32_t validRows,
                                            uint32_t validCols, uint32_t dstCols, uint32_t srcCols,
                                            SaturationMode satMode = SaturationMode::ON)
{
    castBf16toFp4_2D_NoPostUpdate<R, vector_f4e2m1x2>(dst, src, validRows, validCols, dstCols, srcCols);
}

//---------------------------------------------------------------------------------------------
// Source: FP4 variants (float4_e1m2x2_t, float4_e2m1x2_t) - 2D versions
//---------------------------------------------------------------------------------------------

/** FP4_E1M2X2 -> BF16 #part (type expansion) → vcvt(output, input, preg, PART_P0) */
template <typename R>
inline AICORE void castData(__ubuf__ bfloat16_t *dst, __ubuf__ float4_e1m2x2_t *src, uint32_t validRows,
                            uint32_t validCols, uint32_t dstCols, uint32_t srcCols, SaturationMode satMode)
{
    castFp4toBf16<vector_f4e1m2x2>(dst, src, validRows, validCols, dstCols, srcCols);
}

template <typename R>
inline AICORE void castData_2D_NoPostUpdate(__ubuf__ bfloat16_t *dst, __ubuf__ float4_e1m2x2_t *src, uint32_t validRows,
                                            uint32_t validCols, uint32_t dstCols, uint32_t srcCols,
                                            SaturationMode satMode)
{
    castFp4toBf16_2D_NoPostUpdate<vector_f4e1m2x2>(dst, src, validRows, validCols, dstCols, srcCols);
}

/** FP4_E2M1X2 -> BF16 #part (type expansion) → vcvt(output, input, preg, PART_P0) */
template <typename R>
inline AICORE void castData(__ubuf__ bfloat16_t *dst, __ubuf__ float4_e2m1x2_t *src, uint32_t validRows,
                            uint32_t validCols, uint32_t dstCols, uint32_t srcCols, SaturationMode satMode)
{
    castFp4toBf16<vector_f4e2m1x2>(dst, src, validRows, validCols, dstCols, srcCols);
}

template <typename R>
inline AICORE void castData_2D_NoPostUpdate(__ubuf__ bfloat16_t *dst, __ubuf__ float4_e2m1x2_t *src, uint32_t validRows,
                                            uint32_t validCols, uint32_t dstCols, uint32_t srcCols,
                                            SaturationMode satMode)
{
    castFp4toBf16_2D_NoPostUpdate<vector_f4e2m1x2>(dst, src, validRows, validCols, dstCols, srcCols);
}

//---------------------------------------------------------------------------------------------
// Source: U8, I8 (8-bit integers) - 2D versions
//---------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------
// Source: I16 (signed 16-bit integer) - 2D versions
//---------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------
// Source: I32 (signed 32-bit integer) - 2D versions
//---------------------------------------------------------------------------------------------

/** I32 -> I64 #part (type expansion) */
template <typename R>
inline AICORE void castData(__ubuf__ int64_t *dst, __ubuf__ int32_t *src, uint32_t validRows, uint32_t validCols,
                            uint32_t dstCols, uint32_t srcCols, SaturationMode satMode)
{
    cast32toS64<void>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

template <typename R>
inline AICORE void castData_2D_NoPostUpdate(__ubuf__ int64_t *dst, __ubuf__ int32_t *src, uint32_t validRows,
                                            uint32_t validCols, uint32_t dstCols, uint32_t srcCols,
                                            SaturationMode satMode)
{
    cast32toS64<void>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

//---------------------------------------------------------------------------------------------
// Source: U32 (unsigned 32-bit integer) - 2D versions
//---------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------
// Source: I64 (signed 64-bit integer) - 2D versions
//---------------------------------------------------------------------------------------------

/** I64 -> FP32 #rnd #part → vcvt(output, input, preg, R(), PART_EVEN) */
template <typename R>
inline AICORE void castData(__ubuf__ float *dst, __ubuf__ int64_t *src, uint32_t validRows, uint32_t validCols,
                            uint32_t dstCols, uint32_t srcCols, SaturationMode satMode)
{
    castS64to32<R>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

template <typename R>
inline AICORE void castData_2D_NoPostUpdate(__ubuf__ float *dst, __ubuf__ int64_t *src, uint32_t validRows,
                                            uint32_t validCols, uint32_t dstCols, uint32_t srcCols,
                                            SaturationMode satMode)
{
    castS64to32<R>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

/** I64 -> I32 #sat #part → vcvt(output, input, preg, RS_DISABLE, PART_EVEN) */
template <typename R>
inline AICORE void castData(__ubuf__ int32_t *dst, __ubuf__ int64_t *src, uint32_t validRows, uint32_t validCols,
                            uint32_t dstCols, uint32_t srcCols, SaturationMode satMode)
{
    castS64to32<void>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

template <typename R>
inline AICORE void castData_2D_NoPostUpdate(__ubuf__ int32_t *dst, __ubuf__ int64_t *src, uint32_t validRows,
                                            uint32_t validCols, uint32_t dstCols, uint32_t srcCols,
                                            SaturationMode satMode)
{
    castS64to32<void>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

//---------------------------------------------------------------------------------------------
// Source: FP8 variants (float8_e4m3_t, float8_e5m2_t, hifloat8_t) - 2D versions
//---------------------------------------------------------------------------------------------
// FP8 formats are specialized 8-bit floating-point types with different exponent/mantissa splits:
//   - E4M3: 4 exponent bits, 3 mantissa bits (higher precision, smaller range)
//   - E5M2: 5 exponent bits, 2 mantissa bits (lower precision, larger range)
//   - HIF8: Hardware-specific 8-bit float format

/** E4M3 -> FP32 #part (type expansion) → vcvt(output, input, preg, PART_P0) */
template <typename R>
inline AICORE void castData(__ubuf__ float *dst, __ubuf__ float8_e4m3_t *src, uint32_t validRows, uint32_t validCols,
                            uint32_t dstCols, uint32_t srcCols, SaturationMode satMode)
{
    cast8to32<vector_f8e4m3>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

template <typename R>
inline AICORE void castData_2D_NoPostUpdate(__ubuf__ float *dst, __ubuf__ float8_e4m3_t *src, uint32_t validRows,
                                            uint32_t validCols, uint32_t dstCols, uint32_t srcCols,
                                            SaturationMode satMode)
{
    cast8to32<vector_f8e4m3>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

/** E5M2 -> FP32 #part (type expansion) → vcvt(output, input, preg, PART_P0) */
template <typename R>
inline AICORE void castData(__ubuf__ float *dst, __ubuf__ float8_e5m2_t *src, uint32_t validRows, uint32_t validCols,
                            uint32_t dstCols, uint32_t srcCols, SaturationMode satMode)
{
    cast8to32<vector_f8e5m2>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

template <typename R>
inline AICORE void castData_2D_NoPostUpdate(__ubuf__ float *dst, __ubuf__ float8_e5m2_t *src, uint32_t validRows,
                                            uint32_t validCols, uint32_t dstCols, uint32_t srcCols,
                                            SaturationMode satMode)
{
    cast8to32<vector_f8e5m2>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

/** H8 -> FP32 #part (type expansion) → vcvt(output, input, preg, PART_P0) */
template <typename R>
inline AICORE void castData(__ubuf__ float *dst, __ubuf__ hifloat8_t *src, uint32_t validRows, uint32_t validCols,
                            uint32_t dstCols, uint32_t srcCols, SaturationMode satMode)
{
    cast8to32<vector_hif8>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

template <typename R>
inline AICORE void castData_2D_NoPostUpdate(__ubuf__ float *dst, __ubuf__ hifloat8_t *src, uint32_t validRows,
                                            uint32_t validCols, uint32_t dstCols, uint32_t srcCols,
                                            SaturationMode satMode)
{
    cast8to32<vector_hif8>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

//=============================================================================================
// castData_1D_NoPostUpdate Overloads - Organized by Source Type (8→16→32→64-bit sources)
//=============================================================================================
// Optimized 1D versions for contiguous data without padding
// Each section contains conversions FROM a specific source type TO all supported destination types

//---------------------------------------------------------------------------------------------
// Source: 8-bit types (uint8_t, int8_t, float8_e4m3_t, float8_e5m2_t, hifloat8_t) - 1D versions
//---------------------------------------------------------------------------------------------

// Source: FP8_E4M3
template <typename R>
inline AICORE void castData_1D_NoPostUpdate(__ubuf__ float *dst, __ubuf__ float8_e4m3_t *src, uint32_t validRows,
                                            uint32_t validCols, uint32_t dstCols, uint32_t srcCols,
                                            SaturationMode satMode)
{
    cast8to32_1D_NoPostUpdate<vector_f8e4m3>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

// Source: FP8_E5M2
template <typename R>
inline AICORE void castData_1D_NoPostUpdate(__ubuf__ float *dst, __ubuf__ float8_e5m2_t *src, uint32_t validRows,
                                            uint32_t validCols, uint32_t dstCols, uint32_t srcCols,
                                            SaturationMode satMode)
{
    cast8to32_1D_NoPostUpdate<vector_f8e5m2>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

// Source: Hifloat8
template <typename R>
inline AICORE void castData_1D_NoPostUpdate(__ubuf__ float *dst, __ubuf__ hifloat8_t *src, uint32_t validRows,
                                            uint32_t validCols, uint32_t dstCols, uint32_t srcCols,
                                            SaturationMode satMode)
{
    cast8to32_1D_NoPostUpdate<vector_hif8>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

//---------------------------------------------------------------------------------------------
// Source: 16-bit types (half/fp16, bfloat16, int16_t) - 1D versions
//---------------------------------------------------------------------------------------------
// 16-bit conversions are commonly used for mixed-precision training and inference:
//   - FP16 (half): Standard IEEE 754 half-precision (1 sign, 5 exp, 10 mantissa)
//   - BF16 (bfloat16): Brain Float16 (1 sign, 8 exp, 7 mantissa) - FP32-compatible exponent
//   - I16: Signed 16-bit integer

// Note: FP16 -> FP8_E5M2 and FP16 -> FP8_E4M3 conversions are NOT supported
// Only FP16 -> Hifloat8 (H8) conversion is supported

template <typename R>
inline AICORE void castData_1D_NoPostUpdate(__ubuf__ hifloat8_t *dst, __ubuf__ half *src, uint32_t validRows,
                                            uint32_t validCols, uint32_t dstCols, uint32_t srcCols,
                                            SaturationMode satMode)
{
    cast16toH8_1D_NoPostUpdate<R>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

// Source: BFloat16
template <typename R>
inline AICORE void castData_1D_NoPostUpdate(__ubuf__ float *dst, __ubuf__ bfloat16_t *src, uint32_t validRows,
                                            uint32_t validCols, uint32_t dstCols, uint32_t srcCols,
                                            SaturationMode satMode)
{
    cast16to32_1D_NoPostUpdate<void, CastMode::EXPAND>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

template <typename R>
inline AICORE void castData_1D_NoPostUpdate(__ubuf__ int32_t *dst, __ubuf__ bfloat16_t *src, uint32_t validRows,
                                            uint32_t validCols, uint32_t dstCols, uint32_t srcCols,
                                            SaturationMode satMode)
{
    cast16to32_1D_NoPostUpdate<R, CastMode::ROUND_SAT_PART>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

template <typename R>
inline AICORE void castData_1D_NoPostUpdate(__ubuf__ half *dst, __ubuf__ bfloat16_t *src, uint32_t validRows,
                                            uint32_t validCols, uint32_t dstCols, uint32_t srcCols,
                                            SaturationMode satMode)
{
    cast16to16_1D_NoPostUpdate<R, CastMode::SAT_ROUND>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

template <typename R>
inline AICORE void castData_1D_NoPostUpdate(__ubuf__ float4_e1m2x2_t *dst, __ubuf__ bfloat16_t *src, uint32_t validRows,
                                            uint32_t validCols, uint32_t dstCols, uint32_t srcCols,
                                            SaturationMode satMode)
{
    castBf16toFp4_1D_NoPostUpdate<R, vector_f4e1m2x2>(dst, src, validRows, validCols, dstCols, srcCols);
}

template <typename R>
inline AICORE void castData_1D_NoPostUpdate(__ubuf__ float4_e2m1x2_t *dst, __ubuf__ bfloat16_t *src, uint32_t validRows,
                                            uint32_t validCols, uint32_t dstCols, uint32_t srcCols,
                                            SaturationMode satMode)
{
    castBf16toFp4_1D_NoPostUpdate<R, vector_f4e2m1x2>(dst, src, validRows, validCols, dstCols, srcCols);
}

// Source: FP4 variants (float4_e1m2x2_t, float4_e2m1x2_t) - 1D versions
template <typename R>
inline AICORE void castData_1D_NoPostUpdate(__ubuf__ bfloat16_t *dst, __ubuf__ float4_e1m2x2_t *src, uint32_t validRows,
                                            uint32_t validCols, uint32_t dstCols, uint32_t srcCols,
                                            SaturationMode satMode)
{
    castFp4toBf16_1D_NoPostUpdate<vector_f4e1m2x2>(dst, src, validRows, validCols, dstCols, srcCols);
}

template <typename R>
inline AICORE void castData_1D_NoPostUpdate(__ubuf__ bfloat16_t *dst, __ubuf__ float4_e2m1x2_t *src, uint32_t validRows,
                                            uint32_t validCols, uint32_t dstCols, uint32_t srcCols,
                                            SaturationMode satMode)
{
    castFp4toBf16_1D_NoPostUpdate<vector_f4e2m1x2>(dst, src, validRows, validCols, dstCols, srcCols);
}

//---------------------------------------------------------------------------------------------
// Source: 32-bit types (float, int32_t, uint32_t) - 1D versions
//---------------------------------------------------------------------------------------------
// Note: Keep FP32/I32/U32 together for quick lookup of all 32-bit source conversions.

template <typename R>
inline AICORE void castData_1D_NoPostUpdate(__ubuf__ bfloat16_t *dst, __ubuf__ float *src, uint32_t validRows,
                                            uint32_t validCols, uint32_t dstCols, uint32_t srcCols,
                                            SaturationMode satMode)
{
    cast32to16_1D_NoPostUpdate<R>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

template <typename R>
inline AICORE void castData_1D_NoPostUpdate(__ubuf__ int64_t *dst, __ubuf__ float *src, uint32_t validRows,
                                            uint32_t validCols, uint32_t dstCols, uint32_t srcCols,
                                            SaturationMode satMode)
{
    cast32toS64_1D_NoPostUpdate<R>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

template <typename R>
inline AICORE void castData_1D_NoPostUpdate(__ubuf__ float8_e4m3_t *dst, __ubuf__ float *src, uint32_t validRows,
                                            uint32_t validCols, uint32_t dstCols, uint32_t srcCols,
                                            SaturationMode satMode)
{
    cast32to8_1D_NoPostUpdate<R, CastMode::ROUND_SAT_PART, vector_f8e4m3>(dst, src, validRows, validCols, dstCols,
                                                                          srcCols, satMode);
}

template <typename R>
inline AICORE void castData_1D_NoPostUpdate(__ubuf__ float8_e5m2_t *dst, __ubuf__ float *src, uint32_t validRows,
                                            uint32_t validCols, uint32_t dstCols, uint32_t srcCols,
                                            SaturationMode satMode)
{
    cast32to8_1D_NoPostUpdate<R, CastMode::ROUND_SAT_PART, vector_f8e5m2>(dst, src, validRows, validCols, dstCols,
                                                                          srcCols, satMode);
}

template <typename R>
inline AICORE void castData_1D_NoPostUpdate(__ubuf__ hifloat8_t *dst, __ubuf__ float *src, uint32_t validRows,
                                            uint32_t validCols, uint32_t dstCols, uint32_t srcCols,
                                            SaturationMode satMode)
{
    cast32toH8_1D_NoPostUpdate<R>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

template <typename R>
inline AICORE void castData_1D_NoPostUpdate(__ubuf__ int64_t *dst, __ubuf__ int32_t *src, uint32_t validRows,
                                            uint32_t validCols, uint32_t dstCols, uint32_t srcCols,
                                            SaturationMode satMode)
{
    cast32toS64_1D_NoPostUpdate<void>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

//---------------------------------------------------------------------------------------------
// Source: 64-bit types (int64_t)
//---------------------------------------------------------------------------------------------

// Source: I64 (signed 64-bit integer)
template <typename R>
inline AICORE void castData_1D_NoPostUpdate(__ubuf__ float *dst, __ubuf__ int64_t *src, uint32_t validRows,
                                            uint32_t validCols, uint32_t dstCols, uint32_t srcCols,
                                            SaturationMode satMode)
{
    castS64to32_1D_NoPostUpdate<R>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

template <typename R>
inline AICORE void castData_1D_NoPostUpdate(__ubuf__ int32_t *dst, __ubuf__ int64_t *src, uint32_t validRows,
                                            uint32_t validCols, uint32_t dstCols, uint32_t srcCols,
                                            SaturationMode satMode)
{
    castS64to32_1D_NoPostUpdate<void>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

//=============================================================================================
// Main TCVT Implementation
//=============================================================================================

// ============================================================================
// Saturation Control Helper
// ============================================================================
// NOTE: bit selection for a given (Src, Dst) pair is fully compile-time via
// `SaturationCtrlTraits` below — the runtime `SaturationCtrlConfig` struct
// used by other targets is intentionally absent here so all branching folds
// away under `if constexpr`.

// Type trait helpers for cleaner type checking
template <typename T>
struct is_fp16_or_bf16 {
    static constexpr bool value = std::is_same<T, half>::value || std::is_same<T, bfloat16_t>::value;
};

template <typename T>
struct is_any_float {
    static constexpr bool value = std::is_floating_point<T>::value || is_fp16_or_bf16<T>::value;
};

/**
 * Compile-time traits describing which CTRL bits a given (SrcType, DstType)
 * conversion needs to touch. By exposing this as `static constexpr bool`
 * members, every `useCtrl*` check at the use-site collapses via `if constexpr`
 * so dead code is eliminated per template instantiation — eliminating the
 * scalar branches that would otherwise be emitted for every TCVT_IMPL call.
 *
 * The bit *values* (whether to set or clear each bit) still depend on the
 * runtime `satMode`, but the *selection* of bits is fully type-driven.
 *
 * Mapping summary (matches the prior runtime logic exactly):
 *   - dst == float                          : no CTRL touched
 *   - float -> integer                      : CTRL[60]=1, CTRL[59]=(satMode==OFF)
 *   - int -> int (wider -> narrower)        : CTRL[60]=1, CTRL[59]=(satMode==OFF)
 *   - int -> int (narrower -> wider)        : no CTRL touched
 *   - fp32  -> fp16/bf16                    : CTRL[60]=1, CTRL[59]=(satMode==OFF)
 *   - fp16/bf16 <-> fp16/bf16               : CTRL[48]=(satMode==OFF)
 *   - integer -> float (sizeof src >= dst)  : CTRL[60]=1, CTRL[59]=(satMode==OFF)
 */
template <typename SrcType, typename DstType>
struct SaturationCtrlTraits {
private:
    static constexpr bool dstIsFp32 = std::is_same<DstType, float>::value;
    static constexpr bool srcIsFloat = is_any_float<SrcType>::value;
    static constexpr bool dstIsFloat = is_any_float<DstType>::value;
    static constexpr bool srcIsInt = std::is_integral<SrcType>::value;
    static constexpr bool dstIsInt = std::is_integral<DstType>::value;
    static constexpr bool srcIsFp16Bf16 = is_fp16_or_bf16<SrcType>::value;
    static constexpr bool dstIsFp16Bf16 = is_fp16_or_bf16<DstType>::value;

    static constexpr bool floatToInt = srcIsFloat && dstIsInt;
    static constexpr bool intToIntNarrow = srcIsInt && dstIsInt && (sizeof(SrcType) >= sizeof(DstType));
    static constexpr bool fp32ToFp16Bf16 = std::is_same<SrcType, float>::value && dstIsFp16Bf16;
    static constexpr bool fp16Bf16Pair = srcIsFp16Bf16 && dstIsFp16Bf16;
    static constexpr bool intToFloat = srcIsInt && dstIsFloat && (sizeof(SrcType) >= sizeof(DstType));

public:
    // CTRL[60] and CTRL[59] always go together in the original logic.
    static constexpr bool useCtrl60 = !dstIsFp32 && (floatToInt || intToIntNarrow || fp32ToFp16Bf16 || intToFloat);
    static constexpr bool useCtrl59 = useCtrl60;
    static constexpr bool useCtrl48 = !dstIsFp32 && fp16Bf16Pair;
    // True iff this conversion needs to inspect/restore any CTRL bit at all.
    static constexpr bool any = useCtrl60 || useCtrl59 || useCtrl48;
};

/**
 * Apply saturation control bits for a (SrcType, DstType) conversion.
 *
 * All `useCtrl*` checks are compile-time, so for instantiations where the
 * conversion does not touch a given bit the corresponding code is dropped
 * entirely — no scalar test, no register access.
 */
template <typename SrcType, typename DstType>
PTO_INTERNAL void applySaturationCtrlBits(SaturationMode satMode)
{
    using Traits = SaturationCtrlTraits<SrcType, DstType>;
    if constexpr (Traits::useCtrl60) {
        // CTRL[60] is always set to 1 whenever this conversion uses it.
        set_ctrl(sbitset1(get_ctrl(), SAT_MODE_BIT_60));
    }
    if constexpr (Traits::useCtrl59) {
        // CTRL[59] = 1 for OFF, 0 for ON (inverted).
        if (satMode == SaturationMode::OFF) {
            set_ctrl(sbitset1(get_ctrl(), SAT_MODE_BIT_59));
        } else {
            set_ctrl(sbitset0(get_ctrl(), SAT_MODE_BIT_59));
        }
    }
    if constexpr (Traits::useCtrl48) {
        // CTRL[48] = 1 for non-saturation, 0 for saturation.
        if (satMode == SaturationMode::OFF) {
            set_ctrl(sbitset1(get_ctrl(), SAT_MODE_BIT_48));
        } else {
            set_ctrl(sbitset0(get_ctrl(), SAT_MODE_BIT_48));
        }
    }
}

/**
 * Restore the original CTRL bit states that this conversion touched. Bits
 * that were never written are not restored (compile-time eliminated).
 */
template <typename SrcType, typename DstType>
PTO_INTERNAL void restoreSaturationCtrlBits(bool originalCtrl60, bool originalCtrl59, bool originalCtrl48)
{
    using Traits = SaturationCtrlTraits<SrcType, DstType>;
    if constexpr (Traits::useCtrl60) {
        if (originalCtrl60) {
            set_ctrl(sbitset1(get_ctrl(), SAT_MODE_BIT_60));
        } else {
            set_ctrl(sbitset0(get_ctrl(), SAT_MODE_BIT_60));
        }
    }
    if constexpr (Traits::useCtrl59) {
        if (originalCtrl59) {
            set_ctrl(sbitset1(get_ctrl(), SAT_MODE_BIT_59));
        } else {
            set_ctrl(sbitset0(get_ctrl(), SAT_MODE_BIT_59));
        }
    }
    if constexpr (Traits::useCtrl48) {
        if (originalCtrl48) {
            set_ctrl(sbitset1(get_ctrl(), SAT_MODE_BIT_48));
        } else {
            set_ctrl(sbitset0(get_ctrl(), SAT_MODE_BIT_48));
        }
    }
}

// ============================================================================
// High-Level Tile Conversion Interface with explicit SaturationMode
// ============================================================================
// Small dispatch helper to keep TCVT_IMPL compact (avoids repeating the
// implTCVT<...>(dst.data(), src.data(), satMode, rows, cols) boilerplate).
template <typename RoundT, typename TileDataD, typename TileDataS>
PTO_INTERNAL void tcvtDispatch(TileDataD &dst, TileDataS &src, SaturationMode satMode)
{
    implTCVT<TileDataD, TileDataS, RoundT>(dst.data(), src.data(), satMode, dst.GetValidRow(), dst.GetValidCol());
}

// Dispatch on RoundMode -> concrete RoundType. Kept as a separate helper to
// keep TCVT_IMPL under the NBNC line-count limit.
template <typename TileDataD, typename TileDataS>
PTO_INTERNAL void tcvtDispatchByRound(TileDataD &dst, TileDataS &src, RoundMode mode, SaturationMode satMode)
{
    using SrcType = typename TileDataS::DType;
    using DstType = typename TileDataD::DType;
    switch (mode) {
        case RoundMode::CAST_RINT:
            tcvtDispatch<RoundRType>(dst, src, satMode);
            return;
        case RoundMode::CAST_ROUND:
            tcvtDispatch<RoundAType>(dst, src, satMode);
            return;
        case RoundMode::CAST_FLOOR:
            tcvtDispatch<RoundFType>(dst, src, satMode);
            return;
        case RoundMode::CAST_CEIL:
            tcvtDispatch<RoundCType>(dst, src, satMode);
            return;
        case RoundMode::CAST_TRUNC:
            tcvtDispatch<RoundZType>(dst, src, satMode);
            return;
        case RoundMode::CAST_ODD:
            if constexpr (std::is_same<DstType, half>::value && std::is_same<SrcType, float>::value) {
                tcvtDispatch<RoundOType>(dst, src, satMode);
                return;
            }
            break; // fall through to default
        default:
            break;
    }
    // PyTorch-compatible default rounding (also matches a2a3 per-(src,dst) defaults):
    //   float -> integer : truncate toward zero (RoundZType)
    //   everything else  : round-to-nearest-even (RoundRType)
    if constexpr (is_any_float<SrcType>::value && std::is_integral<DstType>::value) {
        tcvtDispatch<RoundZType>(dst, src, satMode);
    } else {
        tcvtDispatch<RoundRType>(dst, src, satMode);
    }
}
/**
 * SATURATION MODE RULES:
 * ======================
 *
 * Hardware Setup:
 * - CTRL[60] and CTRL[59] work together to control saturation mode
 * - CTRL[60]=1, CTRL[59]=0: SaturationMode::ON (saturation enabled)
 * - CTRL[60]=1, CTRL[59]=1: SaturationMode::OFF (saturation disabled)
 * - CTRL[48] value determines saturation for narrower→wider float conversions
 *
 * 1. FLOAT → INTEGER or INTEGER → INTEGER conversions:
 *    - Set CTRL[60]=1 with CTRL[59]=0 for saturation mode
 *    - Set CTRL[60]=1 with CTRL[59]=1 for non-saturation mode
 *    - RS_DISABLE used consistently in vcvt intrinsics
 *
 * 2. NARROWER → WIDER dynamic range conversions (integer):
 *    - No overflow possible, saturation not applicable
 *    - CTRL bits are neglected
 *
 * 3. FLOAT → FLOAT conversions:
 *    a) WIDER → NARROWER range (dst ≠ fp32):
 *       - Set CTRL[60]=1 with CTRL[59]=0 for saturation mode
 *       - Set CTRL[60]=1 with CTRL[59]=1 for non-saturation mode
 *       - RS_DISABLE used consistently in vcvt intrinsics
 *
 *    b) NARROWER → WIDER range (dst ≠ fp32):
 *       - Set CTRL[48]=1 for non-saturation mode
 *       - Set CTRL[48]=0 for saturation mode
 *
 *    c) Where dst = fp32:
 *       - Only non-saturation supported (RS_DISABLE)
 *       - CTRL[48]/[60]/[59] are neglected
 *       - Note: vtrc (fp32→fp32) falls into this category
 */
template <typename TileDataD, typename TileDataS>
PTO_INTERNAL void TCVT_IMPL(TileDataD &dst, TileDataS &src, RoundMode mode, SaturationMode satMode)
{
    using SrcType = typename TileDataS::DType;
    using DstType = typename TileDataD::DType;
    using Traits = SaturationCtrlTraits<SrcType, DstType>;

    // Save original states of only the CTRL bits this conversion will touch.
    // For conversions that touch nothing (e.g. dst==fp32), the entire CTRL
    // save/restore epilogue compiles to nothing.
    bool originalSatMode60 = false;
    bool originalSatMode59 = false;
    bool originalSatMode48 = false;
    if constexpr (Traits::any) {
        uint64_t originalCtrl = get_ctrl();
        if constexpr (Traits::useCtrl60) {
            originalSatMode60 = (originalCtrl & (1ULL << SAT_MODE_BIT_60)) != 0;
        }
        if constexpr (Traits::useCtrl59) {
            originalSatMode59 = (originalCtrl & (1ULL << SAT_MODE_BIT_59)) != 0;
        }
        if constexpr (Traits::useCtrl48) {
            originalSatMode48 = (originalCtrl & (1ULL << SAT_MODE_BIT_48)) != 0;
        }
        applySaturationCtrlBits<SrcType, DstType>(satMode);
    }

    // Execute the conversion with appropriate rounding mode
    tcvtDispatchByRound(dst, src, mode, satMode);

    // Restore original CTRL bit states (compile-time elided when no bits used)
    if constexpr (Traits::any) {
        restoreSaturationCtrlBits<SrcType, DstType>(originalSatMode60, originalSatMode59, originalSatMode48);
    }
}

// ============================================================================
// TCVT_IMPL Overload with Type-Specific Defaults
// ============================================================================
// This overload provides conversion-specific default saturation modes:
// - FP16→UINT8, FP16→INT8: defaults to OFF (PyTorch-compatible truncation)
// - FP32/FP16→INT16: defaults to OFF (truncation behavior)
// - INT64→INT32, INT32→INT16: defaults to OFF (truncation behavior)
// - All others: defaults to ON (native TCVT saturation)
template <typename TileDataD, typename TileDataS>
PTO_INTERNAL void TCVT_IMPL(TileDataD &dst, TileDataS &src, RoundMode mode)
{
    // Conversions that default to OFF for PyTorch compatibility or truncation behavior
    if constexpr (
        // FP16→UINT8 (float→int: CTRL[60] controls saturation)
        (std::is_same<typename TileDataD::DType, uint8_t>::value &&
         std::is_same<typename TileDataS::DType, half>::value) ||
        // FP16→INT8 (float→int: CTRL[60] controls saturation)
        (std::is_same<typename TileDataD::DType, int8_t>::value &&
         std::is_same<typename TileDataS::DType, half>::value) ||
        // FP32→INT16 (float→int: CTRL[60] controls saturation)
        (std::is_same<typename TileDataD::DType, int16_t>::value &&
         std::is_same<typename TileDataS::DType, float>::value) ||
        // FP16→INT16 (float→int: CTRL[60] controls saturation)
        (std::is_same<typename TileDataD::DType, int16_t>::value &&
         std::is_same<typename TileDataS::DType, half>::value) ||
        // INT64→INT32 (int→int: CTRL[60] controls saturation)
        (std::is_same<typename TileDataD::DType, int32_t>::value &&
         std::is_same<typename TileDataS::DType, int64_t>::value) ||
        // INT32→INT16 (int→int: CTRL[60] controls saturation)
        (std::is_same<typename TileDataD::DType, int16_t>::value &&
         std::is_same<typename TileDataS::DType, int32_t>::value)) {
        TCVT_IMPL(dst, src, mode, SaturationMode::OFF);
    } else {
        // All other conversions: default to ON (native TCVT saturation)
        TCVT_IMPL(dst, src, mode, SaturationMode::ON);
    }
}

// ============================================================================
// TCVT_IMPL Overloads with tmp buffer (unused in A5, for API compatibility)
// ============================================================================
template <typename TileDataD, typename TileDataS, typename TmpTileData>
PTO_INTERNAL void TCVT_IMPL(TileDataD &dst, TileDataS &src, TmpTileData &tmp, RoundMode mode, SaturationMode satMode)
{
    TCVT_IMPL(dst, src, mode, satMode);
}

template <typename TileDataD, typename TileDataS, typename TmpTileData>
PTO_INTERNAL void TCVT_IMPL(TileDataD &dst, TileDataS &src, TmpTileData &tmp, RoundMode mode)
{
    TCVT_IMPL(dst, src, mode);
}

} // namespace pto
#endif
