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
 * @file TCvtCommon.hpp
 * @brief Shared utilities for NPU Type Conversion (TCVT) operations
 *
 * This header provides common constants, enums, macros, and helper functions
 * shared across multiple NPU architectures (kirin9030, a5, kirinX90).
 *
 * USAGE: Architecture-specific TCvt.hpp files should include this header:
 *   #include <pto/common/internal/TCvtCommon.hpp>
 *
 * PLATFORM DIFFERENCES:
 *   - Saturation control differs between platforms:
 *     - Kirin9030: CTRL register setting fails, must use RS_ENABLE workaround
 *     - A5: CTRL register works correctly, use RS_DISABLE + CTRL bits
 *   - A5 has additional types (BF16, FP8, H8) not in kirin9030
 *   - A5 requires mem_bar(VST_VST) in certain functions
 *
 * FUNCTIONS EXTRACTED:
 *   Only completely identical functions are extracted here.
 *   Functions with platform differences (RS_ENABLE vs RS_DISABLE) remain in
 *   architecture-specific files.
 */

#ifndef TCVT_COMMON_HPP
#define TCVT_COMMON_HPP

#include <pto/common/constants.hpp>
#include <pto/common/utils.hpp>

namespace pto {

// ============================================================================
// CTRL Register Bit Definitions for Saturation Mode Control
// ============================================================================
/**
 * CTRL[60]: Primary hardware control bit for saturation operations
 * - Used in combination with CTRL[59] to control saturation mode
 * - CTRL[60]=1, CTRL[59]=1: SaturationMode::ON
 * - CTRL[60]=1, CTRL[59]=0: SaturationMode::OFF
 * - Used for: float->integer, integer->integer, float->float (wider->narrower, dst!=fp32)
 */
constexpr const int SAT_MODE_BIT_60 = 60;

/**
 * CTRL[59]: Secondary hardware control bit for saturation operations
 * - Used in combination with CTRL[60] to control saturation mode
 * - CTRL[60]=1, CTRL[59]=0: SaturationMode::ON
 * - CTRL[60]=1, CTRL[59]=1: SaturationMode::OFF
 * - Used for: float->integer, integer->integer, float->float (wider->narrower, dst!=fp32)
 */
constexpr const int SAT_MODE_BIT_59 = 59;

/**
 * CTRL[48]: Saturation control bit for narrower->wider float conversions
 * - Used for: float->float (narrower->wider, dst!=fp32)
 * - CTRL[48]=1: Non-saturation mode
 * - CTRL[48]=0: Saturation mode
 */
constexpr const int SAT_MODE_BIT_48 = 48;

// ============================================================================
// CastMode Enum - Unified enum for all type conversion modes
// ============================================================================
/**
 * Unified enum for all type conversion modes
 * Describes the vcvt intrinsic parameter pattern used for conversion
 */
enum class CastMode
{
    EXPAND,         // vcvt(..., PART_EVEN) - Type expansion only, no conversion
    ROUND,          // vcvt(..., R()) - Conversion with rounding only
    ROUND_SAT,      // vcvt(..., R(), RS_DISABLE) - Conversion with rounding and saturation
    ROUND_PART,     // vcvt(..., R(), PART_EVEN) - Conversion with rounding and part operation
    ROUND_SAT_PART, // vcvt(..., R(), RS_DISABLE, PART_EVEN) - Rounding, saturation, and part
    SAT_PART,       // vcvt(..., RS_DISABLE, PART_EVEN) - Saturation and part (no rounding)
    SAT_ROUND       // vcvt(..., RS_DISABLE, R()) - Saturation then rounding (reversed order)
};

// ============================================================================
// Edge Case Alignment Control
// ============================================================================
// PyTorch alignment for edge cases (inf, -inf, nan, overflow)
// 1 = PyTorch-compatible (uses NonSatTorch), 0 = standard (faster)
#define EDGE_CASE_ALIGN_ENABLE 1

// ============================================================================
// Loop Macros for 2D Processing
// ============================================================================
#define FOR_ROWS                                     \
    for (uint16_t row = 0; row < validRows; row++) { \
        int32_t dstOffset = row * dstCols;           \
        int32_t srcOffset = row * srcCols;           \
        uint32_t sreg = validCols;

#define FOR_ELEMENTS(elNum)                                 \
    constexpr uint16_t elementsNum = (elNum);               \
    uint16_t repeatTimes = CeilDivision(sreg, elementsNum); \
    for (uint16_t idx = 0; idx < repeatTimes; idx++) {

#define END_FOR_ELEMENTS      \
    srcOffset += elementsNum; \
    dstOffset += elementsNum; \
    }

#define END_FOR_ROWS }

// ============================================================================
// Shared NonSatTorch Helper Functions (1D) - Completely Identical
// ============================================================================
// These functions use RS_DISABLE for non-saturation mode, which is identical
// across all platforms. They perform PyTorch-compatible edge case handling.

// FP32 -> INT16 (PyTorch-compatible for inf/-inf)
// Two-step: fp32 -> int32 -> int16 (uses registers, no UB temp)
template <typename R>
inline AICORE void cast32to16_NonSatTorch_1D(__ubuf__ int16_t *dst, __ubuf__ float *src, uint32_t validRows,
                                             uint32_t validCols, uint32_t dstCols, uint32_t srcCols)
{
    uint32_t totalElements = validRows * validCols;
    uint16_t repeatTimes = CeilDivision(totalElements, ELE_CNT_B32);
    uint32_t sReg = totalElements;
    uint32_t len32 = ELE_CNT_B32;
    MaskReg preg_b32 = CreatePredicate<float>(len32);

    for (uint16_t i = 0; i < repeatTimes; ++i) {
        RegTensor<float> v_input_fp32;
        RegTensor<int32_t> v_temp_int32;
        RegTensor<int16_t> v_output_int16;
        MaskReg preg_b32_st = CreatePredicate<float>(sReg);

        vlds(v_input_fp32, src, i * ELE_CNT_B32, NORM);
        vcvt(v_temp_int32, v_input_fp32, preg_b32, R(), RS_DISABLE);
        vcvt(v_output_int16, v_temp_int32, preg_b32, RS_DISABLE, PART_EVEN);
        vsts(v_output_int16, dst, i * ELE_CNT_B32, PK_B32, preg_b32_st);
    }
}

// Float16 (half) to signed 16-bit integer conversion for non-saturation mode (PyTorch-aligned)
// This version matches PyTorch behavior for inf/-inf and performs a two-step conversion:
// 1. fp16 -> int32
// 2. int32 -> int16
template <typename R>
inline AICORE void cast16to16_NonSatTorch_1D(__ubuf__ int16_t *dst, __ubuf__ half *src, uint32_t validRows,
                                             uint32_t validCols, uint32_t dstCols, uint32_t srcCols)
{
    uint32_t totalElements = validRows * validCols;
    uint16_t repeatTimes = CeilDivision(totalElements, ELE_CNT_B32);
    uint32_t sReg = totalElements;
    uint32_t len16 = ELE_CNT_B16;
    uint32_t len32 = ELE_CNT_B32;
    MaskReg preg_b16 = CreatePredicate<half>(len16);
    MaskReg preg_b32 = CreatePredicate<float>(len32);

    // Perform two-step conversion using registers (fp16 -> int32 -> int16)
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        RegTensor<half> v_input_fp16;
        RegTensor<int32_t> v_temp_int32;
        RegTensor<int16_t> v_output_int16;
        MaskReg preg_b32_st = CreatePredicate<float>(sReg);

        // Step 1: Load fp16 and convert to int32 (stays in register)
        vlds(v_input_fp16, src, i * ELE_CNT_B32, UNPK_B16);
        vcvt(v_temp_int32, v_input_fp16, preg_b16, R(), PART_EVEN);

        // Step 2: Convert int32 to int16 with non-saturation and store
        vcvt(v_output_int16, v_temp_int32, preg_b32, RS_DISABLE, PART_EVEN);
        vsts(v_output_int16, dst, i * ELE_CNT_B32, PK_B32, preg_b32_st);
    }
}

// Float16 (half) to signed 8-bit integer conversion for non-saturation mode (PyTorch-aligned)
// This version matches PyTorch behavior for inf/-inf and performs a multi-step conversion:
// 1. fp16 -> int16 (direct conversion)
// 2. bitwise AND with 255 using int16
// 3. int16 -> fp16
// 4. fp16 -> int8
template <typename R>
inline AICORE void cast16to8_NonSatTorch_1D(__ubuf__ int8_t *dst, __ubuf__ half *src, uint32_t validRows,
                                            uint32_t validCols, uint32_t dstCols, uint32_t srcCols)
{
    uint32_t totalElements = validRows * validCols;
    uint16_t repeatTimes = CeilDivision(totalElements, ELE_CNT_B16);
    uint32_t sReg = totalElements;
    uint32_t len16 = ELE_CNT_B16;
    MaskReg preg_b16 = CreatePredicate<half>(len16);
    MaskReg pg = pset_b16(PAT_ALL);

    // Perform four-step conversion using registers (fp16 -> int16 -> AND -> fp16 -> int8)
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        RegTensor<half> v_input_fp16, v_temp_fp16;
        RegTensor<int16_t> v_temp_int16, v_temp_and, v_mask;
        vector_s8 v_output_int8;
        MaskReg preg_b16_st = CreatePredicate<half>(sReg);

        // Step 1: Load fp16 and convert to int16 (stays in register)
        vlds(v_input_fp16, src, i * ELE_CNT_B16, NORM);
        vcvt(v_temp_int16, v_input_fp16, preg_b16, R(), RS_DISABLE);

        // Step 2: Bitwise AND with 255 (stays in register)
        vdup(v_mask, static_cast<int16_t>(255), pg, MODE_ZEROING);
        vand(v_temp_and, v_temp_int16, v_mask, preg_b16_st);

        // Step 3: Convert int16 back to fp16 (stays in register)
        vcvt(v_temp_fp16, v_temp_and, preg_b16, R());

        // Step 4: Convert fp16 to int8 (no saturation) and store
        vcvt(v_output_int8, v_temp_fp16, preg_b16, R(), RS_DISABLE, PART_EVEN);
        vsts(v_output_int8, dst, i * ELE_CNT_B16, PK_B16, preg_b16_st);
    }
}

// ============================================================================
// Shared Platform-Neutral Helper Functions (1D) - Completely Identical
// ============================================================================

/**
 * Cast between 32-bit types - 1D version
 * Handles: f32 -> s32 #rnd #sat, s32 -> f32 #rnd, f32 -> f32 #rnd (same-type rounding)
 */
template <typename R, CastMode MODE, typename DST, typename SRC>
inline AICORE void cast32to32_1D_NoPostUpdate(__ubuf__ DST *dst, __ubuf__ SRC *src, uint32_t validRows,
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
        RegTensor<DST> v_output;
        MaskReg preg_b32_st = CreatePredicate<float>(sReg);

        vlds(v_input_0, src, i * ELE_CNT_B32, NORM);
        if constexpr (std::is_same<DST, SRC>::value) {
            vtrc(v_output, v_input_0, R(), preg_b32_st);
        } else if constexpr (MODE == CastMode::ROUND_SAT) {
            vcvt(v_output, v_input_0, preg_b32, R(), RS_DISABLE);
        } else {
            vcvt(v_output, v_input_0, preg_b32, R());
        }
        vsts(v_output, dst, i * ELE_CNT_B32, NORM_B32, preg_b32_st);
        // sReg is decremented by CreatePredicate with POST_UPDATE
    }
}

/**
 * Cast 8-bit to 16-bit types - 1D version
 */
template <typename SRC_VEC, typename DST, typename SRC>
inline AICORE void cast8to16_1D_NoPostUpdate(__ubuf__ DST *dst, __ubuf__ SRC *src, uint32_t validRows,
                                             uint32_t validCols, uint32_t dstCols, uint32_t srcCols,
                                             SaturationMode satMode)
{
    uint32_t totalElements = validRows * validCols;
    uint16_t repeatTimes = CeilDivision(totalElements, ELE_CNT_B16);
    uint32_t sReg = totalElements;
    uint32_t len8 = ELE_CNT_B8;
    MaskReg preg_b8 = CreatePredicate<uint8_t>(len8);

    for (uint16_t i = 0; i < repeatTimes; ++i) {
        SRC_VEC v_input_0;
        RegTensor<DST> v_output;
        MaskReg preg_b16 = CreatePredicate<half>(sReg);

        vlds(v_input_0, src, i * ELE_CNT_B16, UNPK_B8);
        vcvt(v_output, v_input_0, preg_b8, PART_EVEN);
        vsts(v_output, dst, i * ELE_CNT_B16, NORM_B16, preg_b16);
        // sReg is decremented by CreatePredicate with POST_UPDATE
    }
}

/**
 * Cast 8-bit to 32-bit types - 1D version
 */
template <typename SRC_VEC, typename DST, typename SRC>
inline AICORE void cast8to32_1D_NoPostUpdate(__ubuf__ DST *dst, __ubuf__ SRC *src, uint32_t validRows,
                                             uint32_t validCols, uint32_t dstCols, uint32_t srcCols,
                                             SaturationMode satMode)
{
    uint32_t len8 = ELE_CNT_B8;
    uint32_t totalElements = validRows * validCols;
    uint16_t repeatTimes = CeilDivision(totalElements, ELE_CNT_B16);
    uint32_t sReg = totalElements;
    uint32_t next_len = (sReg > ELE_CNT_B32) ? sReg - ELE_CNT_B32 : 0;
    MaskReg pg = pset_b8(PAT_ALL);
    MaskReg preg_b8 = CreatePredicate<uint8_t>(len8);
    SRC_VEC v_zero;
    vdup((RegTensor<uint8_t> &)v_zero, 0, pg, MODE_ZEROING);

    for (uint16_t i = 0; i < repeatTimes; ++i) {
        SRC_VEC v_input_0, v_input_1, v_input_2;
        RegTensor<DST> v_output_0, v_output_1;
        MaskReg preg_b16_cur = CreatePredicate<half>(sReg);
        MaskReg preg_b16_next = CreatePredicate<half>(next_len);
        MaskReg preg_b32, preg_b32_next;
        punpack(preg_b32, preg_b16_cur, LOWER);
        punpack(preg_b32_next, preg_b16_next, LOWER);

        vlds((RegTensor<uint8_t> &)v_input_0, (__ubuf__ uint8_t *)src, i * ELE_CNT_B16, UNPK_B8);
        vintlv((RegTensor<uint8_t> &)v_input_1, (RegTensor<uint8_t> &)v_input_2, (RegTensor<uint8_t> &)v_input_0,
               (RegTensor<uint8_t> &)v_zero);
        vcvt(v_output_0, v_input_1, preg_b8, PART_P0);
        vcvt(v_output_1, v_input_2, preg_b8, PART_P0);
        vsts(v_output_0, dst, ELE_CNT_B32 * (i * 2), NORM_B32, preg_b32);
        vsts(v_output_1, dst, ELE_CNT_B32 * (i * 2 + 1), NORM_B32, preg_b32_next);
    }
}

} // namespace pto

#endif // TCVT_COMMON_HPP