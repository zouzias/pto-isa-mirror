/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef INT64_REARRANGE_HPP
#define INT64_REARRANGE_HPP

#include <pto/npu/a5/Int64Binary.hpp>

namespace pto {

#if defined(PTO_NPU_ARCH_A5) || defined(PTO_NPU_ARCH_A6)
template <typename T, unsigned DstCols>
PTO_INTERNAL void Int64Fill(__ubuf__ T* dst, T scalar, unsigned validRows, unsigned validCols)
{
    constexpr unsigned elementsPerRepeat = CCE_VL / sizeof(T);
    uint16_t repeatTimes = CeilDivision(validCols, elementsPerRepeat);
    __VEC_SCOPE__
    {
        vector_s32 lowReg, highReg, half0, half1;
        MaskReg lowMask, highMask;
        uint64_t bits = static_cast<uint64_t>(scalar);
        vbr(lowReg, static_cast<int32_t>(bits));
        vbr(highReg, static_cast<int32_t>(bits >> 32));
        uint16_t rows = validRows;
        for (uint16_t row = 0; row < rows; ++row) {
            uint32_t remainingCols = validCols;
            for (uint16_t colRepeat = 0; colRepeat < repeatTimes; ++colRepeat) {
                uint32_t cols = remainingCols > elementsPerRepeat ? elementsPerRepeat : remainingCols;
                uint32_t maskCols = cols;
                MaskReg mask = plt_b32(maskCols, POST_UPDATE);
                uint32_t colOffset = colRepeat * elementsPerRepeat;
                pintlv_b32(lowMask, highMask, mask, mask);
                vintlv(half0, half1, lowReg, highReg);
                vsts(half0, (__ubuf__ int32_t*)dst + (row * DstCols + colOffset) * 2, 0, NORM_B32, lowMask);
                vsts(
                    half1, (__ubuf__ int32_t*)dst + (row * DstCols + colOffset) * 2 + CCE_VL / sizeof(int32_t), 0,
                    NORM_B32, highMask);
                remainingCols -= cols;
            }
        }
    }
}

template <typename T, unsigned DstCols>
PTO_INTERNAL void Int64Tri(__ubuf__ T* dst, unsigned validRows, unsigned validCols, int diagonal, bool upper)
{
    constexpr unsigned elementsPerRepeat = CCE_VL / sizeof(T);
    uint16_t repeatTimes = CeilDivision(validCols, elementsPerRepeat);
    __VEC_SCOPE__
    {
        vector_s32 zero, one;
        vbr(zero, 0);
        vbr(one, 1);
        uint16_t rows = validRows;
        for (uint16_t row = 0; row < rows; ++row) {
            int boundary = static_cast<int>(row) + diagonal;
            uint32_t ones;
            uint32_t zeros;
            if (upper) {
                zeros = boundary <= 0 ? 0 : boundary >= static_cast<int>(validCols) ? validCols : boundary;
                ones = validCols - zeros;
            } else {
                ones = boundary < 0 ? 0 : boundary + 1 >= static_cast<int>(validCols) ? validCols : boundary + 1;
                zeros = validCols - ones;
            }
            uint32_t prefix = upper ? zeros : ones;
            uint32_t remainingCols = validCols;
            for (uint16_t colRepeat = 0; colRepeat < repeatTimes; ++colRepeat) {
                uint32_t cols = remainingCols > elementsPerRepeat ? elementsPerRepeat : remainingCols;
                uint32_t colOffset = colRepeat * elementsPerRepeat;
                uint32_t prefixCols = prefix > colOffset ? prefix - colOffset : 0;
                prefixCols = prefixCols > cols ? cols : prefixCols;
                uint32_t storeCols = cols;
                uint32_t prefixMaskCols = prefixCols;
                MaskReg storeMask = plt_b32(storeCols, POST_UPDATE);
                MaskReg prefixMask = plt_b32(prefixMaskCols, POST_UPDATE);
                vector_s32 low;
                if (upper)
                    vsel(low, zero, one, prefixMask);
                else
                    vsel(low, one, zero, prefixMask);
                vsts(low, zero, (__ubuf__ int32_t*)dst + (row * DstCols + colOffset) * 2, 0, INTLV_B32, storeMask);
                remainingCols -= cols;
            }
        }
    }
}

template <typename T, typename I, unsigned DstCols, unsigned IdxCols>
PTO_INTERNAL void Int64Gather(
    __ubuf__ T* dst, __ubuf__ T* src, __ubuf__ I* index, unsigned validRows, unsigned validCols)
{
    static_assert(sizeof(I) == sizeof(uint32_t), "Int64Gather requires b32 indices");
    constexpr unsigned elementsPerRepeat = CCE_VL / sizeof(T);
    uint16_t repeatTimes = CeilDivision(validCols, elementsPerRepeat);
    __VEC_SCOPE__
    {
        vector_u32 idx, wordIdx, highIdx, low, high;
        uint16_t rows = validRows;
        for (uint16_t row = 0; row < rows; ++row) {
            uint32_t remainingCols = validCols;
            for (uint16_t colRepeat = 0; colRepeat < repeatTimes; ++colRepeat) {
                uint32_t cols = remainingCols > elementsPerRepeat ? elementsPerRepeat : remainingCols;
                uint32_t maskCols = cols;
                MaskReg mask = plt_b32(maskCols, POST_UPDATE);
                uint32_t colOffset = colRepeat * elementsPerRepeat;
                vlds(idx, (__ubuf__ uint32_t*)index + row * IdxCols + colOffset, 0, NORM);
                vadd(wordIdx, idx, idx, mask, MODE_ZEROING);
                vadds(highIdx, wordIdx, 1u, mask, MODE_ZEROING);
                vgather2(low, (__ubuf__ uint32_t*)src, wordIdx, mask);
                vgather2(high, (__ubuf__ uint32_t*)src, highIdx, mask);
                vsts(
                    (vector_s32&)low, (vector_s32&)high, (__ubuf__ int32_t*)dst + (row * DstCols + colOffset) * 2, 0,
                    INTLV_B32, mask);
                remainingCols -= cols;
            }
        }
    }
}

template <typename T, typename I, unsigned DstNumel, unsigned SrcCols, unsigned IdxCols>
PTO_INTERNAL void Int64Scatter(
    __ubuf__ T* dst, __ubuf__ T* src, __ubuf__ I* index, unsigned validRows, unsigned validCols)
{
    static_assert(sizeof(I) == sizeof(uint32_t), "Int64Scatter requires b32 indices");
    constexpr unsigned elementsPerRepeat = CCE_VL / sizeof(T);
    uint16_t repeatTimes = CeilDivision(validCols, elementsPerRepeat);
    __VEC_SCOPE__
    {
        vector_u32 zero, idx, wordIdx, highIdx, low, high;
        vbr(zero, 0u);
        uint32_t remaining = DstNumel * 2;
        constexpr uint16_t wordsPerRepeat = CCE_VL / sizeof(uint32_t);
        constexpr uint16_t initRepeats = (DstNumel * 2 + wordsPerRepeat - 1) / wordsPerRepeat;
        for (uint16_t repeat = 0; repeat < initRepeats; ++repeat) {
            uint32_t words = remaining > wordsPerRepeat ? wordsPerRepeat : remaining;
            uint32_t maskWords = words;
            MaskReg initMask = plt_b32(maskWords, POST_UPDATE);
            vsts(zero, (__ubuf__ uint32_t*)dst + repeat * wordsPerRepeat, 0, NORM_B32, initMask);
            remaining -= words;
        }
        uint16_t rows = validRows;
        for (uint16_t row = 0; row < rows; ++row) {
            uint32_t remainingCols = validCols;
            for (uint16_t colRepeat = 0; colRepeat < repeatTimes; ++colRepeat) {
                uint32_t cols = remainingCols > elementsPerRepeat ? elementsPerRepeat : remainingCols;
                uint32_t maskCols = cols;
                MaskReg mask = plt_b32(maskCols, POST_UPDATE);
                uint32_t colOffset = colRepeat * elementsPerRepeat;
                vlds(low, high, (__ubuf__ uint32_t*)src + (row * SrcCols + colOffset) * 2, 0, DINTLV_B32);
                vlds(idx, (__ubuf__ uint32_t*)index + row * IdxCols + colOffset, 0, NORM);
                vadd(wordIdx, idx, idx, mask, MODE_ZEROING);
                vadds(highIdx, wordIdx, 1u, mask, MODE_ZEROING);
                vscatter(low, (__ubuf__ uint32_t*)dst, wordIdx, mask);
                vscatter(high, (__ubuf__ uint32_t*)dst, highIdx, mask);
                remainingCols -= cols;
            }
        }
    }
}

template <MaskPattern Pattern, ScatterAxis Axis, typename T, unsigned DstNumel, unsigned DstCols, unsigned SrcCols>
PTO_INTERNAL void Int64ScatterPattern(__ubuf__ T* dst, __ubuf__ T* src, unsigned validRows, unsigned validCols)
{
    constexpr unsigned times = GetTimesByMask<Pattern>();
    constexpr unsigned offset = Int64MaskPatternOffset<Pattern>();
    constexpr unsigned elementsPerRepeat = CCE_VL / sizeof(T);
    uint16_t repeatTimes = CeilDivision(validCols, elementsPerRepeat);
    __VEC_SCOPE__
    {
        vector_s32 z, l0, h0;
        vector_u32 lane, elemIndex, lowIndex, highIndex;
        vbr(z, 0);
        uint32_t total = DstNumel * 2;
        uint32_t rem = total;
        constexpr uint16_t vl = CCE_VL / sizeof(uint32_t);
        constexpr uint16_t repeats = (DstNumel * 2 + vl - 1) / vl;
        for (uint16_t r = 0; r < repeats; ++r) {
            uint32_t words = rem > vl ? vl : rem;
            uint32_t maskWords = words;
            MaskReg m = plt_b32(maskWords, POST_UPDATE);
            vsts(z, (__ubuf__ int32_t*)dst + r * vl, 0, NORM_B32, m);
            rem -= words;
        }
        vci((vector_s32&)lane, 0, INC_ORDER);
        uint16_t rows = validRows;
        for (uint16_t i = 0; i < rows; ++i) {
            uint32_t remainingCols = validCols;
            for (uint16_t colRepeat = 0; colRepeat < repeatTimes; ++colRepeat) {
                uint32_t cols = remainingCols > elementsPerRepeat ? elementsPerRepeat : remainingCols;
                uint32_t maskCols = cols;
                MaskReg m = plt_b32(maskCols, POST_UPDATE);
                uint32_t colOffset = colRepeat * elementsPerRepeat;
                vlds(l0, h0, (__ubuf__ int32_t*)src + (i * SrcCols + colOffset) * 2, 0, DINTLV_B32);
                if constexpr (Axis == ScatterAxis::SCATTER_COL) {
                    vsts(
                        l0, h0, (__ubuf__ int32_t*)dst + ((i * times + offset) * DstCols + colOffset) * 2, 0, INTLV_B32,
                        m);
                } else {
                    vadds(elemIndex, lane, static_cast<uint32_t>(colOffset), m, MODE_ZEROING);
                    vmuls(elemIndex, elemIndex, static_cast<uint32_t>(times), m, MODE_ZEROING);
                    vadds(elemIndex, elemIndex, static_cast<uint32_t>(offset), m, MODE_ZEROING);
                    vadd(lowIndex, elemIndex, elemIndex, m, MODE_ZEROING);
                    vadds(highIndex, lowIndex, 1u, m, MODE_ZEROING);
                    __ubuf__ uint32_t* rowDst = (__ubuf__ uint32_t*)dst + i * DstCols * 2;
                    vscatter((vector_u32&)l0, rowDst, lowIndex, m);
                    vscatter((vector_u32&)h0, rowDst, highIndex, m);
                }
                remainingCols -= cols;
            }
        }
    }
}

template <typename T, unsigned DstCols, unsigned SrcRowStride>
PTO_INTERNAL void Int64RowExpand(__ubuf__ T* dst, __ubuf__ T* src, unsigned validRows, unsigned validCols)
{
    constexpr unsigned elementsPerRepeat = CCE_VL / sizeof(T);
    uint16_t repeatTimes = CeilDivision(validCols, elementsPerRepeat);
    __VEC_SCOPE__
    {
        vector_s32 lowReg, highReg, srcLow, srcHigh, wordReg, dummy;
        MaskReg allMask = pset_b32(PAT_ALL);
        uint16_t rows = validRows;
        for (uint16_t row = 0; row < rows; ++row) {
            vlds(srcLow, srcHigh, (__ubuf__ int32_t*)src + row * SrcRowStride * 2, 0, DINTLV_B32);
            vdup(lowReg, srcLow, allMask, POS_LOWEST, MODE_ZEROING);
            vdup(highReg, srcHigh, allMask, POS_LOWEST, MODE_ZEROING);
            vintlv(wordReg, dummy, lowReg, highReg);
            uint32_t remainingCols = validCols;
            for (uint16_t colRepeat = 0; colRepeat < repeatTimes; ++colRepeat) {
                uint32_t cols = remainingCols > elementsPerRepeat ? elementsPerRepeat : remainingCols;
                uint32_t maskWords = cols * 2;
                MaskReg mask = plt_b32(maskWords, POST_UPDATE);
                uint32_t colOffset = colRepeat * elementsPerRepeat;
                vsts(wordReg, (__ubuf__ int32_t*)dst + (row * DstCols + colOffset) * 2, 0, NORM_B32, mask);
                remainingCols -= cols;
            }
        }
    }
}

template <typename T, unsigned DstCols>
PTO_INTERNAL void Int64ColExpand(__ubuf__ T* dst, __ubuf__ T* src, unsigned validRows, unsigned validCols)
{
    constexpr unsigned elementsPerRepeat = CCE_VL / sizeof(T);
    uint16_t repeatTimes = CeilDivision(validCols, elementsPerRepeat);
    __VEC_SCOPE__
    {
        vector_s32 wordReg;
        uint16_t rows = validRows;
        uint32_t remainingCols = validCols;
        for (uint16_t colRepeat = 0; colRepeat < repeatTimes; ++colRepeat) {
            uint32_t cols = remainingCols > elementsPerRepeat ? elementsPerRepeat : remainingCols;
            uint32_t colOffset = colRepeat * elementsPerRepeat;
            vlds(wordReg, (__ubuf__ int32_t*)src + colOffset * 2, 0, NORM);
            uint32_t maskWords = cols * 2;
            MaskReg mask = plt_b32(maskWords, POST_UPDATE);
            for (uint16_t row = 0; row < rows; ++row) {
                vsts(wordReg, (__ubuf__ int32_t*)dst + (row * DstCols + colOffset) * 2, 0, NORM_B32, mask);
            }
            remainingCols -= cols;
        }
    }
}
#else
// Declaration-only stubs for kirin9030/kirinX90 (no 64-bit intrinsics).
// See Int64Binary.hpp for details.
template <typename T, unsigned DstCols>
PTO_INTERNAL void Int64Fill(__ubuf__ T* dst, T scalar, unsigned validRows, unsigned validCols);

template <typename T, unsigned DstCols>
PTO_INTERNAL void Int64Tri(__ubuf__ T* dst, unsigned validRows, unsigned validCols, int diagonal, bool upper);

template <typename T, unsigned DstCols>
PTO_INTERNAL void Int64ColExpand(__ubuf__ T* dst, __ubuf__ T* src, unsigned validRows, unsigned validCols);

template <typename T, unsigned DstCols, unsigned SrcRowStride>
PTO_INTERNAL void Int64RowExpand(__ubuf__ T* dst, __ubuf__ T* src, unsigned validRows, unsigned validCols);

template <typename T, typename I, unsigned DstNumel, unsigned SrcCols, unsigned IdxCols>
PTO_INTERNAL void Int64Scatter(
    __ubuf__ T* dst, __ubuf__ T* src, __ubuf__ I* index, unsigned validRows, unsigned validCols);

template <MaskPattern Pattern, ScatterAxis Axis, typename T, unsigned DstNumel, unsigned DstCols, unsigned SrcCols>
PTO_INTERNAL void Int64ScatterPattern(__ubuf__ T* dst, __ubuf__ T* src, unsigned validRows, unsigned validCols);
#endif

} // namespace pto

#endif
