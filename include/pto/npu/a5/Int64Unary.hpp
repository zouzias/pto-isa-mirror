/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef INT64_UNARY_HPP
#define INT64_UNARY_HPP

#include <pto/npu/a5/Int64Common.hpp>

namespace pto {

#if defined(PTO_NPU_ARCH_A5) || defined(PTO_NPU_ARCH_A6)
template <Int64Op Op, typename T, unsigned DstCols, unsigned SrcCols>
PTO_INTERNAL void Int64Unary(__ubuf__ T* dst, __ubuf__ T* src, unsigned validRows, unsigned validCols)
{
    constexpr unsigned elementsPerRepeat = CCE_VL / sizeof(uint32_t);
    uint16_t repeatTimes = CeilDivision(validCols, elementsPerRepeat);

    __VEC_SCOPE__
    {
        vector_s32 dstLow, dstHigh, srcLow, srcHigh, half0, half1;
        MaskReg lowMask, highMask;
        uint16_t rowCount = validRows;

        for (uint16_t row = 0; row < rowCount; ++row) {
            uint32_t remainingCols = validCols;
            for (uint16_t colRepeat = 0; colRepeat < repeatTimes; ++colRepeat) {
                uint32_t cols = remainingCols > elementsPerRepeat ? elementsPerRepeat : remainingCols;
                MaskReg mask = plt_b32(cols, POST_UPDATE);
                uint32_t colOffset = colRepeat * elementsPerRepeat;
                uint32_t srcOffset = (row * SrcCols + colOffset) * 2;
                uint32_t dstOffset = (row * DstCols + colOffset) * 2;
                vlds(srcLow, srcHigh, (__ubuf__ int32_t*)src, srcOffset, DINTLV_B32);
                if constexpr (Op == Int64Op::Not) {
                    vnot((vector_u32&)dstLow, (vector_u32&)srcLow, mask, MODE_ZEROING);
                    vnot((vector_u32&)dstHigh, (vector_u32&)srcHigh, mask, MODE_ZEROING);
                } else {
                    vector_s32 zero, negLow, negHigh;
                    MaskReg negative, carry, carryOut;
                    vbr(zero, 0);
                    vcmp_lt(negative, srcHigh, zero, mask);
                    vsubc(carry, negLow, zero, srcLow, negative);
                    vsubcs(carryOut, negHigh, zero, srcHigh, carry, negative);
                    vsel(dstLow, negLow, srcLow, negative);
                    vsel(dstHigh, negHigh, srcHigh, negative);
                }
                pintlv_b32(lowMask, highMask, mask, mask);
                vintlv(half0, half1, dstLow, dstHigh);
                vsts(half0, (__ubuf__ int32_t*)dst, dstOffset, NORM_B32, lowMask);
                vsts(half1, (__ubuf__ int32_t*)dst, dstOffset + CCE_VL / sizeof(int32_t), NORM_B32, highMask);
                remainingCols -= cols;
            }
        }
    }
}
#else
// Declaration-only stubs for kirin9030/kirinX90 (no 64-bit intrinsics).
// The A5 instruction headers call these templates from discarded if-constexpr
// branches; they are never instantiated on architectures without 64-bit
// vector support, so a declaration is sufficient for phase-1 name lookup.
template <Int64Op Op, typename T, unsigned DstCols, unsigned SrcCols>
PTO_INTERNAL void Int64Unary(
    __ubuf__ T* dst, __ubuf__ T* src, unsigned validRows, unsigned validCols);
#endif

} // namespace pto

#endif
