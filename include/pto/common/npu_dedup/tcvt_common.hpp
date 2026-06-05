/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMMON_NPU_DEDUP_TCVT_COMMON_HPP
#define PTO_COMMON_NPU_DEDUP_TCVT_COMMON_HPP

#include <pto/common/constants.hpp>
#include <pto/common/utils.hpp>

namespace pto {

// This header requires common.hpp to be included before it for MaskReg, RegTensor, CreatePredicate, and related
// vector intrinsic types.
constexpr const int SAT_MODE_BIT_60 = 60;
constexpr const int SAT_MODE_BIT_59 = 59;
constexpr const int SAT_MODE_BIT_48 = 48;

enum class CastMode
{
    EXPAND,
    ROUND,
    ROUND_SAT,
    ROUND_PART,
    ROUND_SAT_PART,
    SAT_PART,
    SAT_ROUND
};

#define EDGE_CASE_ALIGN_ENABLE 1

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

/**
 * Cast between 16-bit types - 1D version
 */
template <typename R, CastMode MODE, typename DST, typename SRC>
inline AICORE void cast16to16_1D_NoPostUpdate(__ubuf__ DST *dst, __ubuf__ SRC *src, uint32_t validRows,
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
        RegTensor<DST> v_output;
        MaskReg preg_b16_st = CreatePredicate<half>(sReg);

        vlds(v_input_0, src, i * ELE_CNT_B16, NORM);
        if constexpr (MODE == CastMode::ROUND_SAT) {
            vcvt(v_output, v_input_0, preg_b16, R(), RS_DISABLE);
        } else if constexpr (MODE == CastMode::SAT_ROUND) {
            vcvt(v_output, v_input_0, preg_b16, RS_DISABLE, R());
        } else {
            vcvt(v_output, v_input_0, preg_b16, R());
        }
        vsts(v_output, dst, i * ELE_CNT_B16, NORM_B16, preg_b16_st);
        // sReg is decremented by CreatePredicate with POST_UPDATE
    }
}

/**
 * Cast 16-bit to 32-bit types - 1D version
 */
template <typename R, CastMode MODE, typename DST, typename SRC>
inline AICORE void cast16to32_1D_NoPostUpdate(__ubuf__ DST *dst, __ubuf__ SRC *src, uint32_t validRows,
                                              uint32_t validCols, uint32_t dstCols, uint32_t srcCols,
                                              SaturationMode satMode)
{
    uint32_t totalElements = validRows * validCols;
    uint16_t repeatTimes = CeilDivision(totalElements, ELE_CNT_B32);
    uint32_t sReg = totalElements;
    uint32_t len16 = ELE_CNT_B16;
    MaskReg preg_b16 = CreatePredicate<half>(len16);

    for (uint16_t i = 0; i < repeatTimes; ++i) {
        RegTensor<SRC> v_input_0;
        RegTensor<DST> v_output;
        MaskReg preg_b32_st = CreatePredicate<float>(sReg);

        vlds(v_input_0, src, i * ELE_CNT_B32, UNPK_B16);
        if constexpr (MODE == CastMode::EXPAND) {
            vcvt(v_output, v_input_0, preg_b16, PART_EVEN);
        } else if constexpr (MODE == CastMode::ROUND_SAT_PART) {
            vcvt(v_output, v_input_0, preg_b16, R(), RS_DISABLE, PART_EVEN);
        } else {
            vcvt(v_output, v_input_0, preg_b16, R(), PART_EVEN);
        }
        vsts(v_output, dst, i * ELE_CNT_B32, NORM_B32, preg_b32_st);
        // sReg is decremented by CreatePredicate with POST_UPDATE
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

/**
 * Cast between 16-bit types
 * Modes:
 *   ROUND_SAT:  f16 -> s16 #rnd #sat → vcvt(output, input, preg, R(), RS_DISABLE)
 *   SAT_ROUND:  bf16 -> f16 #sat #rnd → vcvt(output, input, preg, RS_DISABLE, R()) [reversed order]
 *   ROUND:      s16 -> f16 #rnd      → vcvt(output, input, preg, R())
 */
template <typename R, CastMode MODE, typename DST, typename SRC>
inline AICORE void cast16to16(__ubuf__ DST *dst, __ubuf__ SRC *src, uint32_t validRows, uint32_t validCols,
                              uint32_t dstCols, uint32_t srcCols, SaturationMode satMode)
{
    FOR_ROWS
    FOR_ELEMENTS(ELE_CNT_B16)
    RegTensor<SRC> v_input_0;
    RegTensor<DST> v_output;
    MaskReg preg_b16 = CreatePredicate<half>(sreg);

    vlds(v_input_0, src, srcOffset, NORM);
    if constexpr (MODE == CastMode::ROUND_SAT) {
        vcvt(v_output, v_input_0, preg_b16, R(), RS_DISABLE);
    } else if constexpr (MODE == CastMode::SAT_ROUND) {
        vcvt(v_output, v_input_0, preg_b16, RS_DISABLE, R());
    } else {
        vcvt(v_output, v_input_0, preg_b16, R());
    }
    vsts(v_output, dst, dstOffset, NORM_B16, preg_b16);
    END_FOR_ELEMENTS
    END_FOR_ROWS
}

// This 2D section requires the 1D helpers above and the architecture common.hpp to be included first.

// Float32 to signed 16-bit integer conversion for non-saturation mode (PyTorch-aligned) - 2D version
// This version matches PyTorch behavior for inf/-inf and performs a two-step conversion:
// 1. fp32 -> int32
// 2. int32 -> int16
template <typename R>
inline AICORE void cast32to16_NonSatTorch_2D(__ubuf__ int16_t *dst, __ubuf__ float *src, uint32_t validRows,
                                             uint32_t validCols, uint32_t dstCols, uint32_t srcCols)
{
    uint32_t len32 = ELE_CNT_B32;
    MaskReg preg_b32 = CreatePredicate<float>(len32);

    // Perform two-step conversion using registers (fp32 -> int32 -> int16)
    FOR_ROWS
    FOR_ELEMENTS(ELE_CNT_B32)
    RegTensor<float> v_input_fp32;
    RegTensor<int32_t> v_temp_int32;
    RegTensor<int16_t> v_output_int16;
    MaskReg preg_b32_st = CreatePredicate<float>(sreg);

    // Step 1: Load fp32 and convert to int32 (stays in register)
    vlds(v_input_fp32, src, srcOffset, NORM);
    vcvt(v_temp_int32, v_input_fp32, preg_b32, R(), RS_DISABLE);

    // Step 2: Convert int32 to int16 with non-saturation and store
    vcvt(v_output_int16, v_temp_int32, preg_b32, RS_DISABLE, PART_EVEN);
    vsts(v_output_int16, dst, dstOffset, PK_B32, preg_b32_st);
    END_FOR_ELEMENTS
    END_FOR_ROWS
}

/**
 * Cast between 32-bit types (float <-> int)
 * Modes:
 *   ROUND_SAT: f32 -> s32 #rnd #sat → vcvt(output, input, preg, R(), RS_DISABLE)
 *   ROUND:     s32 -> f32 #rnd     → vcvt(output, input, preg, R())
 */
template <typename R, CastMode MODE, typename DST, typename SRC>
inline AICORE void cast32to32(__ubuf__ DST *dst, __ubuf__ SRC *src, uint32_t validRows, uint32_t validCols,
                              uint32_t dstCols, uint32_t srcCols, SaturationMode satMode)
{
    FOR_ROWS
    FOR_ELEMENTS(ELE_CNT_B32)
    RegTensor<SRC> v_input_0;
    RegTensor<DST> v_output;
    MaskReg preg_b32 = CreatePredicate<float>(sreg);

    vlds(v_input_0, src, srcOffset, NORM);
    if constexpr (MODE == CastMode::ROUND_SAT) {
        vcvt(v_output, v_input_0, preg_b32, R(), RS_DISABLE);
    } else {
        vcvt(v_output, v_input_0, preg_b32, R());
    }
    vsts(v_output, dst, dstOffset, NORM_B32, preg_b32);
    END_FOR_ELEMENTS
    END_FOR_ROWS
}

// Float16 (half) to signed 16-bit integer conversion for non-saturation mode (PyTorch-aligned) - 2D version
// This version matches PyTorch behavior for inf/-inf and performs a two-step conversion:
// 1. fp16 -> int32
// 2. int32 -> int16
// Uses register-based conversion (no UB temp buffers needed for A5 architecture)
template <typename R>
inline AICORE void cast16to16_NonSatTorch_2D(__ubuf__ int16_t *dst, __ubuf__ half *src, uint32_t validRows,
                                             uint32_t validCols, uint32_t dstCols, uint32_t srcCols)
{
    uint32_t len16 = ELE_CNT_B16;
    uint32_t len32 = ELE_CNT_B32;
    MaskReg preg_b16 = CreatePredicate<half>(len16);
    MaskReg preg_b32 = CreatePredicate<float>(len32);

    // Perform two-step conversion using registers (fp16 -> int32 -> int16)
    FOR_ROWS
    FOR_ELEMENTS(ELE_CNT_B32)
    RegTensor<half> v_input_fp16;
    RegTensor<int32_t> v_temp_int32;
    RegTensor<int16_t> v_output_int16;
    MaskReg preg_b32_st = CreatePredicate<float>(sreg);

    // Step 1: Load fp16 and convert to int32 (stays in register)
    vlds(v_input_fp16, src, srcOffset, UNPK_B16);
    vcvt(v_temp_int32, v_input_fp16, preg_b16, R(), PART_EVEN);

    // Step 2: Convert int32 to int16 with non-saturation and store
    vcvt(v_output_int16, v_temp_int32, preg_b32, RS_DISABLE, PART_EVEN);
    vsts(v_output_int16, dst, dstOffset, PK_B32, preg_b32_st);
    END_FOR_ELEMENTS
    END_FOR_ROWS
}

/**
 * Cast 16-bit to 32-bit types
 * Modes:
 *   EXPAND:          Type expansion (f16/bf16/s16 -> f32/u32/s32 #part) → vcvt(output, input, preg, PART_EVEN)
 *   ROUND_PART:      f16 -> s32 #rnd #part                             → vcvt(output, input, preg, R(), PART_EVEN)
 *   ROUND_SAT_PART:  bf16 -> s32 #rnd #sat #part                       → vcvt(output, input, preg, R(), RS_DISABLE,
 * PART_EVEN)
 */
template <typename R, CastMode MODE, typename DST, typename SRC>
inline AICORE void cast16to32(__ubuf__ DST *dst, __ubuf__ SRC *src, uint32_t validRows, uint32_t validCols,
                              uint32_t dstCols, uint32_t srcCols, SaturationMode satMode)
{
    uint32_t len16 = ELE_CNT_B16;
    MaskReg preg_b16 = CreatePredicate<half>(len16);

    FOR_ROWS
    FOR_ELEMENTS(ELE_CNT_B32)
    RegTensor<SRC> v_input_0;
    RegTensor<DST> v_output;
    MaskReg preg_b32 = CreatePredicate<float>(sreg);

    vlds(v_input_0, src, srcOffset, UNPK_B16);
    if constexpr (MODE == CastMode::EXPAND) {
        vcvt(v_output, v_input_0, preg_b16, PART_EVEN);
    } else if constexpr (MODE == CastMode::ROUND_SAT_PART) {
        vcvt(v_output, v_input_0, preg_b16, R(), RS_DISABLE, PART_EVEN);
    } else {
        vcvt(v_output, v_input_0, preg_b16, R(), PART_EVEN);
    }
    vsts(v_output, dst, dstOffset, NORM_B32, preg_b32);
    END_FOR_ELEMENTS
    END_FOR_ROWS
}

// Float16 (half) to signed 8-bit integer conversion for non-saturation mode (PyTorch-aligned) - 2D version
// This version matches PyTorch behavior for inf/-inf and performs a multi-step conversion:
// 1. fp16 -> int16 (direct conversion)
// 2. bitwise AND with 255 using int16
// 3. int16 -> fp16
// 4. fp16 -> int8
template <typename R>
inline AICORE void cast16to8_NonSatTorch_2D(__ubuf__ int8_t *dst, __ubuf__ half *src, uint32_t validRows,
                                            uint32_t validCols, uint32_t dstCols, uint32_t srcCols)
{
    uint32_t len16 = ELE_CNT_B16;
    MaskReg preg_b16 = CreatePredicate<half>(len16);
    MaskReg pg = pset_b16(PAT_ALL);

    // Perform four-step conversion using registers (fp16 -> int16 -> AND -> fp16 -> int8)
    FOR_ROWS
    FOR_ELEMENTS(ELE_CNT_B16)
    RegTensor<half> v_input_fp16, v_temp_fp16;
    RegTensor<int16_t> v_temp_int16, v_temp_and, v_mask;
    vector_s8 v_output_int8;
    MaskReg preg_b16_st = CreatePredicate<half>(sreg);

    // Step 1: Load fp16 and convert to int16 (stays in register)
    vlds(v_input_fp16, src, srcOffset, NORM);
    vcvt(v_temp_int16, v_input_fp16, preg_b16, R(), RS_DISABLE);

    // Step 2: Bitwise AND with 255 (stays in register)
    vdup(v_mask, static_cast<int16_t>(255), pg, MODE_ZEROING);
    vand(v_temp_and, v_temp_int16, v_mask, preg_b16_st);

    // Step 3: Convert int16 back to fp16 (stays in register)
    vcvt(v_temp_fp16, v_temp_and, preg_b16, R());

    // Step 4: Convert fp16 to int8 (no saturation) and store
    vcvt(v_output_int8, v_temp_fp16, preg_b16, R(), RS_DISABLE, PART_EVEN);
    vsts(v_output_int8, dst, dstOffset, PK_B16, preg_b16_st);
    END_FOR_ELEMENTS
    END_FOR_ROWS
}

/**
 * Cast 8-bit to 16-bit types
 * Handles: u8/s8 -> f16/u16/s16 #part (type expansion)
 * Intrinsic: vcvt(output, input, preg, PART_EVEN)
 */
template <typename SRC_VEC, typename DST, typename SRC>
inline AICORE void cast8to16(__ubuf__ DST *dst, __ubuf__ SRC *src, uint32_t validRows, uint32_t validCols,
                             uint32_t dstCols, uint32_t srcCols, SaturationMode satMode)
{
    uint32_t len8 = ELE_CNT_B8;
    MaskReg preg_b8 = CreatePredicate<uint8_t>(len8);

    FOR_ROWS
    FOR_ELEMENTS(ELE_CNT_B16)
    SRC_VEC v_input_0;
    RegTensor<DST> v_output;
    MaskReg preg_b16 = CreatePredicate<half>(sreg);

    vlds(v_input_0, src, srcOffset, UNPK_B8);
    vcvt(v_output, v_input_0, preg_b8, PART_EVEN);
    vsts(v_output, dst, dstOffset, NORM_B16, preg_b16);
    END_FOR_ELEMENTS
    END_FOR_ROWS
}

/**
 * Cast 8-bit to 32-bit types
 * Handles: I8 -> f32 #part
 * Intrinsic: vcvt(output, input, preg, PART_*)
 */
template <typename SRC_VEC, typename DST, typename SRC>
inline AICORE void cast8to32(__ubuf__ DST *dst, __ubuf__ SRC *src, uint32_t validRows, uint32_t validCols,
                             uint32_t dstCols, uint32_t srcCols, SaturationMode satMode)
{
    uint32_t len8 = ELE_CNT_B8;
    MaskReg preg_b8 = CreatePredicate<uint8_t>(len8);
    MaskReg pg = pset_b8(PAT_ALL);
    SRC_VEC v_zero;
    vdup((RegTensor<uint8_t> &)v_zero, 0, pg, MODE_ZEROING);

    FOR_ROWS
    int32_t rowDstOffset = row * dstCols;
    uint32_t next_len = (sreg > ELE_CNT_B32) ? sreg - ELE_CNT_B32 : 0;
    FOR_ELEMENTS(ELE_CNT_B16)
    SRC_VEC v_input_0, v_input_1, v_input_2;
    RegTensor<DST> v_output_0, v_output_1;
    MaskReg preg_b16_cur = CreatePredicate<half>(sreg);
    MaskReg preg_b16_next = CreatePredicate<half>(next_len);
    MaskReg preg_b32;
    MaskReg preg_b32_next;
    punpack(preg_b32, preg_b16_cur, LOWER);
    punpack(preg_b32_next, preg_b16_next, LOWER);

    vlds((RegTensor<uint8_t> &)v_input_0, (__ubuf__ uint8_t *)src, srcOffset, UNPK_B8);
    vintlv((RegTensor<uint8_t> &)v_input_1, (RegTensor<uint8_t> &)v_input_2, (RegTensor<uint8_t> &)v_input_0,
           (RegTensor<uint8_t> &)v_zero); // interleave with zero
    vcvt(v_output_0, v_input_1, preg_b8, PART_P0);
    vcvt(v_output_1, v_input_2, preg_b8, PART_P0);
    vsts(v_output_0, dst, rowDstOffset + ELE_CNT_B32 * (idx * 2), NORM_B32, preg_b32);
    vsts(v_output_1, dst, rowDstOffset + ELE_CNT_B32 * (idx * 2 + 1), NORM_B32, preg_b32_next);
    END_FOR_ELEMENTS
    END_FOR_ROWS
}

//=============================================================================================
// castData Overloads (2D - with row/column iteration for non-contiguous data)
//=============================================================================================
// These are the main conversion functions organized by source type for easy navigation.
// Each source type section contains conversions to all supported destination types.
//
// ORGANIZATION: Grouped by source type in ascending bit-width order (8→16→32-bit)
// WHY: This ordering provides quick lookup - if you know the source type, you can
// jump directly to its section and find all target conversions in one place.

//---------------------------------------------------------------------------------------------
// Source: FP32 (float) - 2D versions
//---------------------------------------------------------------------------------------------

/**
 * FP32 to FP32 - Applies rounding mode without type conversion
 * Intrinsic: vtrc(output, input, R(), preg)
 *
 * NOTE: Same-type conversions like FP32→FP32 are useful for applying rounding modes
 * to existing data without changing the underlying type (e.g., rounding to nearest even).
 */
template <typename R>
inline AICORE void castData(__ubuf__ float *dst, __ubuf__ float *src, uint32_t validRows, uint32_t validCols,
                            uint32_t dstCols, uint32_t srcCols, SaturationMode satMode)
{
    FOR_ROWS
    FOR_ELEMENTS(ELE_CNT_B32)
    vector_f32 v_input_0, v_output;
    MaskReg preg_b32 = CreatePredicate<float>(sreg);

    vlds(v_input_0, src, srcOffset, NORM);
    vtrc(v_output, v_input_0, R(), preg_b32);
    vsts(v_output, dst, dstOffset, NORM_B32, preg_b32);
    END_FOR_ELEMENTS
    END_FOR_ROWS
}

template <typename R>
inline AICORE void castData_2D_NoPostUpdate(__ubuf__ float *dst, __ubuf__ float *src, uint32_t validRows,
                                            uint32_t validCols, uint32_t dstCols, uint32_t srcCols,
                                            SaturationMode satMode)
{
    FOR_ROWS
    FOR_ELEMENTS(ELE_CNT_B32)
    vector_f32 v_input_0, v_output;
    MaskReg preg_b32 = CreatePredicate<float>(sreg);

    vlds(v_input_0, src, srcOffset, NORM);
    vtrc(v_output, v_input_0, R(), preg_b32);
    vsts(v_output, dst, dstOffset, NORM_B32, preg_b32);
    END_FOR_ELEMENTS
    END_FOR_ROWS
}

/**
 * FP32 to FP16
 * Conversion: f32 -> f16 #rnd #sat #part
 * Uses cast32to16 helper
 */
template <typename R>
inline AICORE void castData(__ubuf__ float16_t *dst, __ubuf__ float *src, uint32_t validRows, uint32_t validCols,
                            uint32_t dstCols, uint32_t srcCols, SaturationMode satMode)
{
    cast32to16<R>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

template <typename R>
inline AICORE void castData_2D_NoPostUpdate(__ubuf__ float16_t *dst, __ubuf__ float *src, uint32_t validRows,
                                            uint32_t validCols, uint32_t dstCols, uint32_t srcCols,
                                            SaturationMode satMode)
{
    cast32to16_2D_NoPostUpdate<R>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

/**
 * FP32 to I16
 * Conversion: f32 -> s16 #rnd #sat #part
 * Uses cast32to16 helper
 */
template <typename R>
inline AICORE void castData(__ubuf__ int16_t *dst, __ubuf__ float *src, uint32_t validRows, uint32_t validCols,
                            uint32_t dstCols, uint32_t srcCols, SaturationMode satMode)
{
    cast32to16<R>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

template <typename R>
inline AICORE void castData_2D_NoPostUpdate(__ubuf__ int16_t *dst, __ubuf__ float *src, uint32_t validRows,
                                            uint32_t validCols, uint32_t dstCols, uint32_t srcCols,
                                            SaturationMode satMode)
{
#if EDGE_CASE_ALIGN_ENABLE
    if (satMode == SaturationMode::OFF) {
        // Use PyTorch-aligned implementation when saturation is OFF and edge case alignment is enabled
        cast32to16_NonSatTorch_2D<R>(dst, src, validRows, validCols, dstCols, srcCols);
    } else {
        cast32to16_2D_NoPostUpdate<R>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
    }
#else
    // Use default implementation when edge case alignment is disabled - saturation controlled by CTRL register
    cast32to16_2D_NoPostUpdate<R>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
#endif
}

/**
 * FP32 to I32
 * Conversion: f32 -> s32 #rnd #sat
 * Intrinsic: vcvt(output, input, preg, R(), RS_DISABLE)
 */
template <typename R>
inline AICORE void castData(__ubuf__ int32_t *dst, __ubuf__ float *src, uint32_t validRows, uint32_t validCols,
                            uint32_t dstCols, uint32_t srcCols, SaturationMode satMode)
{
    cast32to32<R, CastMode::ROUND_SAT>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

template <typename R>
inline AICORE void castData_2D_NoPostUpdate(__ubuf__ int32_t *dst, __ubuf__ float *src, uint32_t validRows,
                                            uint32_t validCols, uint32_t dstCols, uint32_t srcCols,
                                            SaturationMode satMode)
{
    cast32to32<R, CastMode::ROUND_SAT>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

//---------------------------------------------------------------------------------------------
// Source: FP16 (half) - 2D versions
//---------------------------------------------------------------------------------------------

/** FP16 -> FP32 #part (type expansion) → vcvt(output, input, preg, PART_EVEN) */
template <typename R>
inline AICORE void castData(__ubuf__ float *dst, __ubuf__ half *src, uint32_t validRows, uint32_t validCols,
                            uint32_t dstCols, uint32_t srcCols, SaturationMode satMode)
{
    cast16to32<void, CastMode::EXPAND>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

template <typename R>
inline AICORE void castData_2D_NoPostUpdate(__ubuf__ float *dst, __ubuf__ half *src, uint32_t validRows,
                                            uint32_t validCols, uint32_t dstCols, uint32_t srcCols,
                                            SaturationMode satMode)
{
    cast16to32<void, CastMode::EXPAND>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

/** FP16 -> I32 #rnd #part → vcvt(output, input, preg, R(), PART_EVEN) */
template <typename R>
inline AICORE void castData(__ubuf__ int32_t *dst, __ubuf__ half *src, uint32_t validRows, uint32_t validCols,
                            uint32_t dstCols, uint32_t srcCols, SaturationMode satMode)
{
    cast16to32<R, CastMode::ROUND_PART>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

template <typename R>
inline AICORE void castData_2D_NoPostUpdate(__ubuf__ int32_t *dst, __ubuf__ half *src, uint32_t validRows,
                                            uint32_t validCols, uint32_t dstCols, uint32_t srcCols,
                                            SaturationMode satMode)
{
    cast16to32<R, CastMode::ROUND_PART>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

/** FP16 -> I16 #rnd #sat → vcvt(output, input, preg, R(), RS_DISABLE) */
template <typename R>
inline AICORE void castData(__ubuf__ int16_t *dst, __ubuf__ half *src, uint32_t validRows, uint32_t validCols,
                            uint32_t dstCols, uint32_t srcCols, SaturationMode satMode)
{
    cast16to16<R, CastMode::ROUND_SAT>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

template <typename R>
inline AICORE void castData_2D_NoPostUpdate(__ubuf__ int16_t *dst, __ubuf__ half *src, uint32_t validRows,
                                            uint32_t validCols, uint32_t dstCols, uint32_t srcCols,
                                            SaturationMode satMode)
{
#if EDGE_CASE_ALIGN_ENABLE
    if (satMode == SaturationMode::OFF) {
        // Use PyTorch-aligned implementation when saturation is OFF and edge case alignment is enabled
        cast16to16_NonSatTorch_2D<R>(dst, src, validRows, validCols, dstCols, srcCols);
    } else {
        cast16to16<R, CastMode::ROUND_SAT>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
    }
#else
    // Use default implementation when edge case alignment is disabled
    cast16to16<R, CastMode::ROUND_SAT>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
#endif
}

/** FP16 -> I8 #rnd #sat #part */
template <typename R>
inline AICORE void castData(__ubuf__ int8_t *dst, __ubuf__ half *src, uint32_t validRows, uint32_t validCols,
                            uint32_t dstCols, uint32_t srcCols, SaturationMode satMode)
{
    cast16to8<R, CastMode::ROUND_SAT_PART, vector_s8>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

template <typename R>
inline AICORE void castData_2D_NoPostUpdate(__ubuf__ int8_t *dst, __ubuf__ half *src, uint32_t validRows,
                                            uint32_t validCols, uint32_t dstCols, uint32_t srcCols,
                                            SaturationMode satMode)
{
#if EDGE_CASE_ALIGN_ENABLE
    if (satMode == SaturationMode::OFF) {
        // Use PyTorch-aligned implementation when saturation is OFF and edge case alignment is enabled
        cast16to8_NonSatTorch_2D<R>(dst, src, validRows, validCols, dstCols, srcCols);
    } else {
        cast16to8_2D_NoPostUpdate<R, CastMode::ROUND_SAT_PART, vector_s8>(dst, src, validRows, validCols, dstCols,
                                                                          srcCols, satMode);
    }
#else
    // Use default implementation when edge case alignment is disabled
    cast16to8_2D_NoPostUpdate<R, CastMode::ROUND_SAT_PART, vector_s8>(dst, src, validRows, validCols, dstCols, srcCols,
                                                                      satMode);
#endif
}

/** FP16 -> U8 #rnd #sat #part */
template <typename R>
inline AICORE void castData(__ubuf__ uint8_t *dst, __ubuf__ half *src, uint32_t validRows, uint32_t validCols,
                            uint32_t dstCols, uint32_t srcCols, SaturationMode satMode)
{
    cast16to8<R, CastMode::ROUND_SAT_PART, vector_u8>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

template <typename R>
inline AICORE void castData_2D_NoPostUpdate(__ubuf__ uint8_t *dst, __ubuf__ half *src, uint32_t validRows,
                                            uint32_t validCols, uint32_t dstCols, uint32_t srcCols,
                                            SaturationMode satMode)
{
    cast16to8_2D_NoPostUpdate<R, CastMode::ROUND_SAT_PART, vector_u8>(dst, src, validRows, validCols, dstCols, srcCols,
                                                                      satMode);
}

//---------------------------------------------------------------------------------------------
// Source: U8, I8 (8-bit integers) - 2D versions
//---------------------------------------------------------------------------------------------

/** U8 -> FP16 #part (type expansion) */
template <typename R>
inline AICORE void castData(__ubuf__ half *dst, __ubuf__ uint8_t *src, uint32_t validRows, uint32_t validCols,
                            uint32_t dstCols, uint32_t srcCols, SaturationMode satMode)
{
    cast8to16<vector_u8>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

template <typename R>
inline AICORE void castData_2D_NoPostUpdate(__ubuf__ half *dst, __ubuf__ uint8_t *src, uint32_t validRows,
                                            uint32_t validCols, uint32_t dstCols, uint32_t srcCols,
                                            SaturationMode satMode)
{
    cast8to16<vector_u8>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

/** U8 -> U16 #part (type expansion) */
template <typename R>
inline AICORE void castData(__ubuf__ uint16_t *dst, __ubuf__ uint8_t *src, uint32_t validRows, uint32_t validCols,
                            uint32_t dstCols, uint32_t srcCols, SaturationMode satMode)
{
    cast8to16<vector_u8>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

template <typename R>
inline AICORE void castData_2D_NoPostUpdate(__ubuf__ uint16_t *dst, __ubuf__ uint8_t *src, uint32_t validRows,
                                            uint32_t validCols, uint32_t dstCols, uint32_t srcCols,
                                            SaturationMode satMode)
{
    cast8to16<vector_u8>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

/** I8 -> FP16 #part (type expansion) */
template <typename R>
inline AICORE void castData(__ubuf__ half *dst, __ubuf__ int8_t *src, uint32_t validRows, uint32_t validCols,
                            uint32_t dstCols, uint32_t srcCols, SaturationMode satMode)
{
    cast8to16<vector_s8>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

template <typename R>
inline AICORE void castData_2D_NoPostUpdate(__ubuf__ half *dst, __ubuf__ int8_t *src, uint32_t validRows,
                                            uint32_t validCols, uint32_t dstCols, uint32_t srcCols,
                                            SaturationMode satMode)
{
    cast8to16<vector_s8>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

/** I8 -> I16 #part (type expansion) */
template <typename R>
inline AICORE void castData(__ubuf__ int16_t *dst, __ubuf__ int8_t *src, uint32_t validRows, uint32_t validCols,
                            uint32_t dstCols, uint32_t srcCols, SaturationMode satMode)
{
    cast8to16<vector_s8>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

template <typename R>
inline AICORE void castData_2D_NoPostUpdate(__ubuf__ int16_t *dst, __ubuf__ int8_t *src, uint32_t validRows,
                                            uint32_t validCols, uint32_t dstCols, uint32_t srcCols,
                                            SaturationMode satMode)
{
    cast8to16<vector_s8>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

/** I8 -> I32 #part (type expansion) */
template <typename R>
inline AICORE void castData(__ubuf__ int32_t *dst, __ubuf__ int8_t *src, uint32_t validRows, uint32_t validCols,
                            uint32_t dstCols, uint32_t srcCols, SaturationMode satMode)
{
    cast8to32<vector_s8>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

template <typename R>
inline AICORE void castData_2D_NoPostUpdate(__ubuf__ int32_t *dst, __ubuf__ int8_t *src, uint32_t validRows,
                                            uint32_t validCols, uint32_t dstCols, uint32_t srcCols,
                                            SaturationMode satMode)
{
    cast8to32<vector_s8>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

//---------------------------------------------------------------------------------------------
// Source: I16 (signed 16-bit integer) - 2D versions
//---------------------------------------------------------------------------------------------

/** I16 -> U8 #sat #part */
template <typename R>
inline AICORE void castData(__ubuf__ uint8_t *dst, __ubuf__ int16_t *src, uint32_t validRows, uint32_t validCols,
                            uint32_t dstCols, uint32_t srcCols, SaturationMode satMode)
{
    cast16to8<void, CastMode::SAT_PART, vector_u8>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

template <typename R>
inline AICORE void castData_2D_NoPostUpdate(__ubuf__ uint8_t *dst, __ubuf__ int16_t *src, uint32_t validRows,
                                            uint32_t validCols, uint32_t dstCols, uint32_t srcCols,
                                            SaturationMode satMode)
{
    cast16to8<void, CastMode::SAT_PART, vector_u8>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

/** I16 -> FP16 #rnd → vcvt(output, input, preg, R()) */
template <typename R>
inline AICORE void castData(__ubuf__ half *dst, __ubuf__ int16_t *src, uint32_t validRows, uint32_t validCols,
                            uint32_t dstCols, uint32_t srcCols, SaturationMode satMode)
{
    cast16to16<R, CastMode::ROUND>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

template <typename R>
inline AICORE void castData_2D_NoPostUpdate(__ubuf__ half *dst, __ubuf__ int16_t *src, uint32_t validRows,
                                            uint32_t validCols, uint32_t dstCols, uint32_t srcCols,
                                            SaturationMode satMode)
{
    cast16to16<R, CastMode::ROUND>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

/** I16 -> FP32 #part (type expansion) */
template <typename R>
inline AICORE void castData(__ubuf__ float *dst, __ubuf__ int16_t *src, uint32_t validRows, uint32_t validCols,
                            uint32_t dstCols, uint32_t srcCols, SaturationMode satMode)
{
    cast16to32<void, CastMode::EXPAND>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

template <typename R>
inline AICORE void castData_2D_NoPostUpdate(__ubuf__ float *dst, __ubuf__ int16_t *src, uint32_t validRows,
                                            uint32_t validCols, uint32_t dstCols, uint32_t srcCols,
                                            SaturationMode satMode)
{
    cast16to32<void, CastMode::EXPAND>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

/** I16 -> U32 #part (type expansion) */
template <typename R>
inline AICORE void castData(__ubuf__ uint32_t *dst, __ubuf__ int16_t *src, uint32_t validRows, uint32_t validCols,
                            uint32_t dstCols, uint32_t srcCols, SaturationMode satMode)
{
    cast16to32<void, CastMode::EXPAND>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

template <typename R>
inline AICORE void castData_2D_NoPostUpdate(__ubuf__ uint32_t *dst, __ubuf__ int16_t *src, uint32_t validRows,
                                            uint32_t validCols, uint32_t dstCols, uint32_t srcCols,
                                            SaturationMode satMode)
{
    cast16to32<void, CastMode::EXPAND>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

/** I16 -> I32 #part (type expansion) */
template <typename R>
inline AICORE void castData(__ubuf__ int32_t *dst, __ubuf__ int16_t *src, uint32_t validRows, uint32_t validCols,
                            uint32_t dstCols, uint32_t srcCols, SaturationMode satMode)
{
    cast16to32<void, CastMode::EXPAND>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

template <typename R>
inline AICORE void castData_2D_NoPostUpdate(__ubuf__ int32_t *dst, __ubuf__ int16_t *src, uint32_t validRows,
                                            uint32_t validCols, uint32_t dstCols, uint32_t srcCols,
                                            SaturationMode satMode)
{
    cast16to32<void, CastMode::EXPAND>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

//---------------------------------------------------------------------------------------------
// Source: I32 (signed 32-bit integer) - 2D versions
//---------------------------------------------------------------------------------------------

/** I32 -> FP32 #rnd → vcvt(output, input, preg, R()) */
template <typename R>
inline AICORE void castData(__ubuf__ float *dst, __ubuf__ int32_t *src, uint32_t validRows, uint32_t validCols,
                            uint32_t dstCols, uint32_t srcCols, SaturationMode satMode)
{
    cast32to32<R, CastMode::ROUND>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

template <typename R>
inline AICORE void castData_2D_NoPostUpdate(__ubuf__ float *dst, __ubuf__ int32_t *src, uint32_t validRows,
                                            uint32_t validCols, uint32_t dstCols, uint32_t srcCols,
                                            SaturationMode satMode)
{
    cast32to32<R, CastMode::ROUND>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

/** I32 -> I16 #sat #part */
template <typename R>
inline AICORE void castData(__ubuf__ int16_t *dst, __ubuf__ int32_t *src, uint32_t validRows, uint32_t validCols,
                            uint32_t dstCols, uint32_t srcCols, SaturationMode satMode)
{
    cast32to16<void>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

template <typename R>
inline AICORE void castData_2D_NoPostUpdate(__ubuf__ int16_t *dst, __ubuf__ int32_t *src, uint32_t validRows,
                                            uint32_t validCols, uint32_t dstCols, uint32_t srcCols,
                                            SaturationMode satMode)
{
    cast32to16_2D_NoPostUpdate<void>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

/** I32 -> U16 #sat #part */
template <typename R>
inline AICORE void castData(__ubuf__ uint16_t *dst, __ubuf__ int32_t *src, uint32_t validRows, uint32_t validCols,
                            uint32_t dstCols, uint32_t srcCols, SaturationMode satMode)
{
    cast32to16<void>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

template <typename R>
inline AICORE void castData_2D_NoPostUpdate(__ubuf__ uint16_t *dst, __ubuf__ int32_t *src, uint32_t validRows,
                                            uint32_t validCols, uint32_t dstCols, uint32_t srcCols,
                                            SaturationMode satMode)
{
    cast32to16_2D_NoPostUpdate<void>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

/** I32 -> U8 #sat #part */
template <typename R>
inline AICORE void castData(__ubuf__ uint8_t *dst, __ubuf__ int32_t *src, uint32_t validRows, uint32_t validCols,
                            uint32_t dstCols, uint32_t srcCols, SaturationMode satMode)
{
    cast32to8<void, CastMode::SAT_PART, vector_u8>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

template <typename R>
inline AICORE void castData_2D_NoPostUpdate(__ubuf__ uint8_t *dst, __ubuf__ int32_t *src, uint32_t validRows,
                                            uint32_t validCols, uint32_t dstCols, uint32_t srcCols,
                                            SaturationMode satMode)
{
    cast32to8<void, CastMode::SAT_PART, vector_u8>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

//---------------------------------------------------------------------------------------------
// Source: U32 (unsigned 32-bit integer) - 2D versions
//---------------------------------------------------------------------------------------------

/** U32 -> U8 #sat #part */
template <typename R>
inline AICORE void castData(__ubuf__ uint8_t *dst, __ubuf__ uint32_t *src, uint32_t validRows, uint32_t validCols,
                            uint32_t dstCols, uint32_t srcCols, SaturationMode satMode)
{
    cast32to8<void, CastMode::SAT_PART, vector_u8>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

template <typename R>
inline AICORE void castData_2D_NoPostUpdate(__ubuf__ uint8_t *dst, __ubuf__ uint32_t *src, uint32_t validRows,
                                            uint32_t validCols, uint32_t dstCols, uint32_t srcCols,
                                            SaturationMode satMode)
{
    cast32to8<void, CastMode::SAT_PART, vector_u8>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

/** U32 -> U16 #sat #part */
template <typename R>
inline AICORE void castData(__ubuf__ uint16_t *dst, __ubuf__ uint32_t *src, uint32_t validRows, uint32_t validCols,
                            uint32_t dstCols, uint32_t srcCols, SaturationMode satMode)
{
    cast32to16<void>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

template <typename R>
inline AICORE void castData_2D_NoPostUpdate(__ubuf__ uint16_t *dst, __ubuf__ uint32_t *src, uint32_t validRows,
                                            uint32_t validCols, uint32_t dstCols, uint32_t srcCols,
                                            SaturationMode satMode)
{
    cast32to16_2D_NoPostUpdate<void>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

/** U32 -> I16 #sat #part */
template <typename R>
inline AICORE void castData(__ubuf__ int16_t *dst, __ubuf__ uint32_t *src, uint32_t validRows, uint32_t validCols,
                            uint32_t dstCols, uint32_t srcCols, SaturationMode satMode)
{
    cast32to16<void>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

template <typename R>
inline AICORE void castData_2D_NoPostUpdate(__ubuf__ int16_t *dst, __ubuf__ uint32_t *src, uint32_t validRows,
                                            uint32_t validCols, uint32_t dstCols, uint32_t srcCols,
                                            SaturationMode satMode)
{
    cast32to16_2D_NoPostUpdate<void>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

//=============================================================================================
// castData_1D_NoPostUpdate Overloads - Organized by Source Type (8→16→32→64-bit sources)
//=============================================================================================
// Optimized 1D versions for contiguous data without padding
// Each section contains conversions FROM a specific source type TO all supported destination types

//---------------------------------------------------------------------------------------------
// Source: 8-bit types (uint8_t, int8_t) - 1D versions
//---------------------------------------------------------------------------------------------

// Source: U8 (unsigned 8-bit integer)
template <typename R>
inline AICORE void castData_1D_NoPostUpdate(__ubuf__ half *dst, __ubuf__ uint8_t *src, uint32_t validRows,
                                            uint32_t validCols, uint32_t dstCols, uint32_t srcCols,
                                            SaturationMode satMode)
{
    cast8to16_1D_NoPostUpdate<vector_u8>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

template <typename R>
inline AICORE void castData_1D_NoPostUpdate(__ubuf__ uint16_t *dst, __ubuf__ uint8_t *src, uint32_t validRows,
                                            uint32_t validCols, uint32_t dstCols, uint32_t srcCols,
                                            SaturationMode satMode)
{
    cast8to16_1D_NoPostUpdate<vector_u8>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

// Source: I8 (signed 8-bit integer)
template <typename R>
inline AICORE void castData_1D_NoPostUpdate(__ubuf__ half *dst, __ubuf__ int8_t *src, uint32_t validRows,
                                            uint32_t validCols, uint32_t dstCols, uint32_t srcCols,
                                            SaturationMode satMode)
{
    cast8to16_1D_NoPostUpdate<vector_s8>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

template <typename R>
inline AICORE void castData_1D_NoPostUpdate(__ubuf__ int16_t *dst, __ubuf__ int8_t *src, uint32_t validRows,
                                            uint32_t validCols, uint32_t dstCols, uint32_t srcCols,
                                            SaturationMode satMode)
{
    cast8to16_1D_NoPostUpdate<vector_s8>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

template <typename R>
inline AICORE void castData_1D_NoPostUpdate(__ubuf__ int32_t *dst, __ubuf__ int8_t *src, uint32_t validRows,
                                            uint32_t validCols, uint32_t dstCols, uint32_t srcCols,
                                            SaturationMode satMode)
{
    cast8to32_1D_NoPostUpdate<vector_s8>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

//---------------------------------------------------------------------------------------------
// Source: 16-bit types (half/fp16, int16_t) - 1D versions
//---------------------------------------------------------------------------------------------
// 16-bit conversions are commonly used for mixed-precision training and inference:
//   - FP16 (half): Standard IEEE 754 half-precision (1 sign, 5 exp, 10 mantissa)
//   - I16: Signed 16-bit integer

// Source: FP16 (half)
template <typename R>
inline AICORE void castData_1D_NoPostUpdate(__ubuf__ float *dst, __ubuf__ half *src, uint32_t validRows,
                                            uint32_t validCols, uint32_t dstCols, uint32_t srcCols,
                                            SaturationMode satMode)
{
    cast16to32_1D_NoPostUpdate<void, CastMode::EXPAND>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

template <typename R>
inline AICORE void castData_1D_NoPostUpdate(__ubuf__ int32_t *dst, __ubuf__ half *src, uint32_t validRows,
                                            uint32_t validCols, uint32_t dstCols, uint32_t srcCols,
                                            SaturationMode satMode)
{
    cast16to32_1D_NoPostUpdate<R, CastMode::ROUND_PART>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

template <typename R>
inline AICORE void castData_1D_NoPostUpdate(__ubuf__ int16_t *dst, __ubuf__ half *src, uint32_t validRows,
                                            uint32_t validCols, uint32_t dstCols, uint32_t srcCols,
                                            SaturationMode satMode)
{
#if EDGE_CASE_ALIGN_ENABLE
    if (satMode == SaturationMode::OFF) {
        // Use PyTorch-aligned implementation when saturation is OFF and edge case alignment is enabled
        cast16to16_NonSatTorch_1D<R>(dst, src, validRows, validCols, dstCols, srcCols);
    } else {
        cast16to16_1D_NoPostUpdate<R, CastMode::ROUND_SAT>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
    }
#else
    // Use default implementation when edge case alignment is disabled
    cast16to16_1D_NoPostUpdate<R, CastMode::ROUND_SAT>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
#endif
}

template <typename R>
inline AICORE void castData_1D_NoPostUpdate(__ubuf__ int8_t *dst, __ubuf__ half *src, uint32_t validRows,
                                            uint32_t validCols, uint32_t dstCols, uint32_t srcCols,
                                            SaturationMode satMode)
{
#if EDGE_CASE_ALIGN_ENABLE
    if (satMode == SaturationMode::OFF) {
        // Use PyTorch-aligned implementation when saturation is OFF and edge case alignment is enabled
        cast16to8_NonSatTorch_1D<R>(dst, src, validRows, validCols, dstCols, srcCols);
    } else {
        cast16to8_1D_NoPostUpdate<R, CastMode::ROUND_SAT_PART, vector_s8>(dst, src, validRows, validCols, dstCols,
                                                                          srcCols, satMode);
    }
#else
    // Use default implementation when edge case alignment is disabled
    cast16to8_1D_NoPostUpdate<R, CastMode::ROUND_SAT_PART, vector_s8>(dst, src, validRows, validCols, dstCols, srcCols,
                                                                      satMode);
#endif
}

template <typename R>
inline AICORE void castData_1D_NoPostUpdate(__ubuf__ uint8_t *dst, __ubuf__ half *src, uint32_t validRows,
                                            uint32_t validCols, uint32_t dstCols, uint32_t srcCols,
                                            SaturationMode satMode)
{
    cast16to8_1D_NoPostUpdate<R, CastMode::ROUND_SAT_PART, vector_u8>(dst, src, validRows, validCols, dstCols, srcCols,
                                                                      satMode);
}

// Source: I16 (signed 16-bit integer)
template <typename R>
inline AICORE void castData_1D_NoPostUpdate(__ubuf__ uint8_t *dst, __ubuf__ int16_t *src, uint32_t validRows,
                                            uint32_t validCols, uint32_t dstCols, uint32_t srcCols,
                                            SaturationMode satMode)
{
    cast16to8_1D_NoPostUpdate<void, CastMode::SAT_PART, vector_u8>(dst, src, validRows, validCols, dstCols, srcCols,
                                                                   satMode);
}

template <typename R>
inline AICORE void castData_1D_NoPostUpdate(__ubuf__ half *dst, __ubuf__ int16_t *src, uint32_t validRows,
                                            uint32_t validCols, uint32_t dstCols, uint32_t srcCols,
                                            SaturationMode satMode)
{
    cast16to16_1D_NoPostUpdate<R, CastMode::ROUND>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

template <typename R>
inline AICORE void castData_1D_NoPostUpdate(__ubuf__ float *dst, __ubuf__ int16_t *src, uint32_t validRows,
                                            uint32_t validCols, uint32_t dstCols, uint32_t srcCols,
                                            SaturationMode satMode)
{
    cast16to32_1D_NoPostUpdate<void, CastMode::EXPAND>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

template <typename R>
inline AICORE void castData_1D_NoPostUpdate(__ubuf__ uint32_t *dst, __ubuf__ int16_t *src, uint32_t validRows,
                                            uint32_t validCols, uint32_t dstCols, uint32_t srcCols,
                                            SaturationMode satMode)
{
    cast16to32_1D_NoPostUpdate<void, CastMode::EXPAND>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

template <typename R>
inline AICORE void castData_1D_NoPostUpdate(__ubuf__ int32_t *dst, __ubuf__ int16_t *src, uint32_t validRows,
                                            uint32_t validCols, uint32_t dstCols, uint32_t srcCols,
                                            SaturationMode satMode)
{
    cast16to32_1D_NoPostUpdate<void, CastMode::EXPAND>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

//---------------------------------------------------------------------------------------------
// Source: 32-bit types (float, int32_t, uint32_t) - 1D versions
//---------------------------------------------------------------------------------------------
// Note: Keep FP32/I32/U32 together for quick lookup of all 32-bit source conversions.

// Source: FP32 (float)
template <typename R>
inline AICORE void castData_1D_NoPostUpdate(__ubuf__ float *dst, __ubuf__ float *src, uint32_t validRows,
                                            uint32_t validCols, uint32_t dstCols, uint32_t srcCols,
                                            SaturationMode satMode)
{
    cast32to32_1D_NoPostUpdate<R, CastMode::ROUND>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

template <typename R>
inline AICORE void castData_1D_NoPostUpdate(__ubuf__ float16_t *dst, __ubuf__ float *src, uint32_t validRows,
                                            uint32_t validCols, uint32_t dstCols, uint32_t srcCols,
                                            SaturationMode satMode)
{
    cast32to16_1D_NoPostUpdate<R>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

template <typename R>
inline AICORE void castData_1D_NoPostUpdate(__ubuf__ int16_t *dst, __ubuf__ float *src, uint32_t validRows,
                                            uint32_t validCols, uint32_t dstCols, uint32_t srcCols,
                                            SaturationMode satMode)
{
#if EDGE_CASE_ALIGN_ENABLE
    if (satMode == SaturationMode::OFF) {
        // Use PyTorch-aligned implementation when saturation is OFF and edge case alignment is enabled
        cast32to16_NonSatTorch_1D<R>(dst, src, validRows, validCols, dstCols, srcCols);
    } else {
        cast32to16_1D_NoPostUpdate<R>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
    }
#else
    // Use default implementation when edge case alignment is disabled
    cast32to16_1D_NoPostUpdate<R>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
#endif
}

template <typename R>
inline AICORE void castData_1D_NoPostUpdate(__ubuf__ int32_t *dst, __ubuf__ float *src, uint32_t validRows,
                                            uint32_t validCols, uint32_t dstCols, uint32_t srcCols,
                                            SaturationMode satMode)
{
    cast32to32_1D_NoPostUpdate<R, CastMode::ROUND_SAT>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

// Source: I32 (signed 32-bit integer)
template <typename R>
inline AICORE void castData_1D_NoPostUpdate(__ubuf__ float *dst, __ubuf__ int32_t *src, uint32_t validRows,
                                            uint32_t validCols, uint32_t dstCols, uint32_t srcCols,
                                            SaturationMode satMode)
{
    cast32to32_1D_NoPostUpdate<R, CastMode::ROUND>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

template <typename R>
inline AICORE void castData_1D_NoPostUpdate(__ubuf__ int16_t *dst, __ubuf__ int32_t *src, uint32_t validRows,
                                            uint32_t validCols, uint32_t dstCols, uint32_t srcCols,
                                            SaturationMode satMode)
{
    cast32to16_1D_NoPostUpdate<void>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

template <typename R>
inline AICORE void castData_1D_NoPostUpdate(__ubuf__ uint16_t *dst, __ubuf__ int32_t *src, uint32_t validRows,
                                            uint32_t validCols, uint32_t dstCols, uint32_t srcCols,
                                            SaturationMode satMode)
{
    cast32to16_1D_NoPostUpdate<void>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

template <typename R>
inline AICORE void castData_1D_NoPostUpdate(__ubuf__ uint8_t *dst, __ubuf__ int32_t *src, uint32_t validRows,
                                            uint32_t validCols, uint32_t dstCols, uint32_t srcCols,
                                            SaturationMode satMode)
{
    cast32to8_1D_NoPostUpdate<void, CastMode::SAT_PART, vector_u8>(dst, src, validRows, validCols, dstCols, srcCols,
                                                                   satMode);
}

// Source: U32 (unsigned 32-bit integer)
template <typename R>
inline AICORE void castData_1D_NoPostUpdate(__ubuf__ uint8_t *dst, __ubuf__ uint32_t *src, uint32_t validRows,
                                            uint32_t validCols, uint32_t dstCols, uint32_t srcCols,
                                            SaturationMode satMode)
{
    cast32to8_1D_NoPostUpdate<void, CastMode::SAT_PART, vector_u8>(dst, src, validRows, validCols, dstCols, srcCols,
                                                                   satMode);
}

template <typename R>
inline AICORE void castData_1D_NoPostUpdate(__ubuf__ uint16_t *dst, __ubuf__ uint32_t *src, uint32_t validRows,
                                            uint32_t validCols, uint32_t dstCols, uint32_t srcCols,
                                            SaturationMode satMode)
{
    cast32to16_1D_NoPostUpdate<void>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

template <typename R>
inline AICORE void castData_1D_NoPostUpdate(__ubuf__ int16_t *dst, __ubuf__ uint32_t *src, uint32_t validRows,
                                            uint32_t validCols, uint32_t dstCols, uint32_t srcCols,
                                            SaturationMode satMode)
{
    cast32to16_1D_NoPostUpdate<void>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

//=============================================================================================
// Main TCVT Implementation
//=============================================================================================

/**
 * Main TCVT implementation function
 * Converts tile data from source type to destination type using specified rounding mode
 * Iterates over rows and calls appropriate castData specialization
 *
 * @param satMode: Saturation mode control (Kirin9030-specific):
 *                 In Kirin9030, saturation is controlled by both:
 *                 1. CTRL register bits [60] and [48] - set by TCVT_IMPL based on conversion type
 *                 2. RS_DISABLE/RS_DISABLE parameters in vcvt intrinsics
 *
 *                 The satMode parameter works in conjunction with CTRL bits:
 *                 - CTRL[60]: Used for float→int, int→int, float→float (wider→narrower, dst≠fp32)
 *                 - CTRL[48]: Used for float→float (narrower→wider, dst≠fp32), VTRC.fp16
 *
 *                 The actual saturation behavior is determined by both the CTRL bit setting
 *                 and the CastMode used in castData template instantiations.
 */
template <typename TileDataD, typename TileDataS, typename R>
__tf__ PTO_INTERNAL OP_NAME(TCVT)
    OP_TYPE(element_wise) void implTCVT(typename TileDataD::TileDType __out__ dst,
                                        typename TileDataS::TileDType __in__ src, SaturationMode satMode,
                                        unsigned validRows, unsigned validCols,
                                        VFImplKind version = VFImplKind::VFIMPL_DEFAULT)
{
    // Saturation is controlled by:
    // 1. CTRL[60]/CTRL[48] register bits (set by caller TCVT_IMPL based on conversion type)
    // 2. RS_DISABLE/RS_DISABLE in vcvt intrinsics (determined by CastMode in castData templates)
    // The satMode parameter is passed through to castData functions which use it to select
    // between RS_DISABLE and RS_DISABLE in the vcvt intrinsic calls.

    using T1 = typename TileDataD::DType;
    using T2 = typename TileDataS::DType;
    __ubuf__ T1 *dstPtr = (__ubuf__ T1 *)__cce_get_tile_ptr(dst);
    __ubuf__ T2 *srcPtr = (__ubuf__ T2 *)__cce_get_tile_ptr(src);
    __VEC_SCOPE__
    {
        // Compile-time check: Use 1D optimization if:
        // 1. ValidCol == Cols (no column padding) for both src and dst, OR
        // 2. Both tiles have Rows == 1 (single row case)
        if constexpr (((TileDataD::ValidCol == TileDataD::Cols) && (TileDataS::ValidCol == TileDataS::Cols)) ||
                      ((TileDataD::Rows == 1) && (TileDataS::Rows == 1))) {
            // Use 1D path: faster bulk processing without row iteration overhead
            switch (version) {
                case VFImplKind::VFIMPL_2D_NO_POST_UPDATE:
                    castData_2D_NoPostUpdate<R>(dstPtr, srcPtr, validRows, validCols, TileDataD::Cols, TileDataS::Cols,
                                                satMode);
                    break;
                case VFImplKind::VFIMPL_DEFAULT:
                case VFImplKind::VFIMPL_1D_NO_POST_UPDATE:
                case VFImplKind::VFIMPL_1D_POST_UPDATE:
                case VFImplKind::VFIMPL_2D_POST_UPDATE:
                default:
                    castData_1D_NoPostUpdate<R>(dstPtr, srcPtr, validRows, validCols, TileDataD::Cols, TileDataS::Cols,
                                                satMode);
                    break;
            }

        } else {
            // Use 2D path: handles strided/padded data with row-by-row iteration
            // version parameter controls predicate update strategy:
            // VFIMPL_2D_NO_POST_UPDATE: manual predicate handling
            // default: auto predicate update
            switch (version) {
                case VFImplKind::VFIMPL_1D_NO_POST_UPDATE:
                case VFImplKind::VFIMPL_2D_NO_POST_UPDATE:
                    castData_2D_NoPostUpdate<R>(dstPtr, srcPtr, validRows, validCols, TileDataD::Cols, TileDataS::Cols,
                                                satMode);
                    break;
                default:
                    castData<R>(dstPtr, srcPtr, validRows, validCols, TileDataD::Cols, TileDataS::Cols, satMode);
                    break;
            }
        }
    }
}

} // namespace pto

#endif
