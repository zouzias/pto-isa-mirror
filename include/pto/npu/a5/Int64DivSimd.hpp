/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef INT64_DIV_SIMD_HPP
#define INT64_DIV_SIMD_HPP

#include <pto/npu/a5/Int64SimdCommon.hpp>

namespace pto {

PTO_INTERNAL void Int64B128CalcSimd(
    vector_s32& lhsLow, vector_s32& lhsHigh, vector_s32& rhsLow, vector_s32& rhsHigh, vector_u32& zero, MaskReg& mask)
{
    vector_s32 mul0Low, mul0High, mul1Low, mul1High, mul2Low, mul2High, mul3Low, mul3High;
    vector_s32 tmp0, tmp1;
    vmull((vector_u32&)mul0Low, (vector_u32&)mul0High, (vector_u32&)lhsLow, (vector_u32&)rhsLow, mask);
    vmull((vector_u32&)mul1Low, (vector_u32&)mul1High, (vector_u32&)lhsLow, (vector_u32&)rhsHigh, mask);
    vmull((vector_u32&)mul2Low, (vector_u32&)mul2High, (vector_u32&)lhsHigh, (vector_u32&)rhsLow, mask);
    vmull((vector_u32&)mul3Low, (vector_u32&)mul3High, (vector_u32&)lhsHigh, (vector_u32&)rhsHigh, mask);

    MaskReg carry0, carry1;
    vaddc(carry0, tmp0, mul0High, mul1Low, mask);
    vaddc(carry1, tmp1, tmp0, mul2Low, mask);
    vaddcs(carry0, tmp0, mul3Low, mul1High, carry0, mask);
    vaddcs(carry1, rhsLow, tmp0, mul2High, carry1, mask);
    vaddcs(carry0, tmp0, (vector_s32&)zero, mul3High, carry0, mask);
    vaddcs(carry0, rhsHigh, (vector_s32&)zero, tmp0, carry1, mask);
}

PTO_INTERNAL void Int64DivSignedRestoreSignSimd(
    vector_s32& dstLow, vector_s32& dstHigh, vector_s32& qLow, vector_s32& qHigh, vector_s32& lhsHigh,
    vector_s32& rhsHigh, vector_s32& zeroLow, vector_s32& zeroHigh, MaskReg& zeroMask, MaskReg& nonZeroMask)
{
    MaskReg sameSign;
    Int64DivSignSimd(sameSign, lhsHigh, rhsHigh, zeroHigh, nonZeroMask);
    MaskReg allMask = pset_b32(PAT_ALL);
    vector_s32 negLow, negHigh;
    Int64SubRegsSimd(negLow, negHigh, zeroLow, zeroHigh, qLow, qHigh, allMask);
    Int64SelectRegsSimd(qLow, qHigh, qLow, qHigh, negLow, negHigh, sameSign);
    Int64SelectRegsSimd(dstLow, dstHigh, zeroLow, zeroHigh, qLow, qHigh, zeroMask);
}

PTO_INTERNAL void Int64DivReciprocalSimd(
    vector_s32& reciprocalLow, vector_s32& reciprocalHigh, vector_s32& divisorLow, vector_s32& divisorHigh,
    MaskReg& workMask)
{
    vector_f32 divisorFloat;
    vector_u32 reciprocalBits;
    Int64ToFloatSimd(divisorFloat, divisorLow, divisorHigh);
    Int64DivFloatPreprocessSimd(reciprocalBits, divisorFloat, workMask);
    FloatToInt64Simd(reciprocalLow, reciprocalHigh, (vector_f32&)reciprocalBits);
}

PTO_INTERNAL void Int64DivSignedRegsSimd(
    vector_s32& dstLow, vector_s32& dstHigh, vector_s32& lhsLow, vector_s32& lhsHigh, vector_s32& rhsLow,
    vector_s32& rhsHigh, MaskReg& inputMask)
{
    vector_s32 absLhsLow, absLhsHigh, absRhsLow, absRhsHigh;
    vector_s32 zeroLow, zeroHigh, minusOneLow, minusOneHigh, oneLow, oneHigh;
    Int64AbsRegsSimd(absLhsLow, absLhsHigh, lhsLow, lhsHigh, inputMask);
    Int64AbsRegsSimd(absRhsLow, absRhsHigh, rhsLow, rhsHigh, inputMask);
    Int64DuplicateRegsSimd(zeroLow, zeroHigh, 0, 0);
    Int64DuplicateRegsSimd(minusOneLow, minusOneHigh, 0xffffffffU, 0xffffffffU);
    Int64DuplicateRegsSimd(oneLow, oneHigh, 1, 0);

    MaskReg zeroMask, nonZeroMask, oneMask, nonOneMask, workMask;
    Int64CompareEqRegsSimd(zeroMask, rhsLow, rhsHigh, zeroLow, zeroHigh, inputMask);
    pnot(nonZeroMask, zeroMask, inputMask);
    Int64CompareEqRegsSimd(oneMask, absRhsLow, absRhsHigh, oneLow, oneHigh, nonZeroMask);
    pnot(nonOneMask, oneMask, nonZeroMask);
    pand(workMask, nonOneMask, nonZeroMask, nonZeroMask);

    vector_u32 zeroWord;
    vector_s32 reciprocalLow, reciprocalHigh;
    Int64DivReciprocalSimd(reciprocalLow, reciprocalHigh, absRhsLow, absRhsHigh, workMask);

    vector_s32 qLow, qHigh, tLow, tHigh, remLow, remHigh, adjustedLow, adjustedHigh;
    Int64MulRegsSimd(qLow, qHigh, absRhsLow, absRhsHigh, reciprocalLow, reciprocalHigh, workMask);
    Int64NotRegsSimd(qLow, qHigh, workMask);
    Int64AddRegsSimd(qLow, qHigh, qLow, qHigh, oneLow, oneHigh, workMask);
    vbr(zeroWord, 0);
    Int64B128CalcSimd(reciprocalLow, reciprocalHigh, qLow, qHigh, zeroWord, workMask);
    Int64AddRegsSimd(tLow, tHigh, reciprocalLow, reciprocalHigh, qLow, qHigh, workMask);
    Int64MulRegsSimd(qLow, qHigh, absRhsLow, absRhsHigh, tLow, tHigh, workMask);
    Int64NotRegsSimd(qLow, qHigh, workMask);
    Int64AddRegsSimd(qLow, qHigh, qLow, qHigh, oneLow, oneHigh, workMask);
    Int64B128CalcSimd(tLow, tHigh, qLow, qHigh, zeroWord, workMask);
    Int64AddRegsSimd(qLow, qHigh, tLow, tHigh, qLow, qHigh, workMask);
    Int64B128CalcSimd(absLhsLow, absLhsHigh, qLow, qHigh, zeroWord, workMask);

    Int64MulRegsSimd(tLow, tHigh, qLow, qHigh, absRhsLow, absRhsHigh, workMask);
    Int64SubRegsSimd(remLow, remHigh, absLhsLow, absLhsHigh, tLow, tHigh, workMask);
    MaskReg geMask;
    Int64CompareGeURegsSimd(geMask, remLow, remHigh, absRhsLow, absRhsHigh, workMask);
    Int64SubRegsSimd(adjustedLow, adjustedHigh, remLow, remHigh, absRhsLow, absRhsHigh, geMask);
    Int64AddRegsSimd(tLow, tHigh, qLow, qHigh, oneLow, oneHigh, geMask);
    Int64SelectRegsSimd(remLow, remHigh, adjustedLow, adjustedHigh, remLow, remHigh, geMask);
    Int64SelectRegsSimd(qLow, qHigh, tLow, tHigh, qLow, qHigh, geMask);
    Int64CompareGeURegsSimd(geMask, remLow, remHigh, absRhsLow, absRhsHigh, workMask);
    Int64AddRegsSimd(tLow, tHigh, qLow, qHigh, oneLow, oneHigh, geMask);
    Int64SelectRegsSimd(qLow, qHigh, tLow, tHigh, qLow, qHigh, geMask);
    Int64SelectRegsSimd(qLow, qHigh, absLhsLow, absLhsHigh, qLow, qHigh, oneMask);

    Int64DivSignedRestoreSignSimd(
        dstLow, dstHigh, qLow, qHigh, lhsHigh, rhsHigh, zeroLow, zeroHigh, zeroMask, nonZeroMask);
}

PTO_INTERNAL void Int64DivUnsignedClassifySimd(
    MaskReg& zeroMask, MaskReg& oneMask, MaskReg& smallWorkMask, MaskReg& largeResultOne, MaskReg& largeResultZero,
    vector_s32& lhsLow, vector_s32& lhsHigh, vector_s32& rhsLow, vector_s32& rhsHigh, vector_s32& zeroLow,
    vector_s32& zeroHigh, vector_s32& oneLow, vector_s32& oneHigh, MaskReg& inputMask)
{
    MaskReg nonZeroMask, nonOneMask, largeDivisor, lhsGeRhs, smallDivisor;
    Int64CompareEqRegsSimd(zeroMask, rhsLow, rhsHigh, zeroLow, zeroHigh, inputMask);
    pnot(nonZeroMask, zeroMask, inputMask);
    Int64CompareEqRegsSimd(oneMask, rhsLow, rhsHigh, oneLow, oneHigh, nonZeroMask);
    pnot(nonOneMask, oneMask, nonZeroMask);

    vector_u32 signBit;
    vbr(signBit, 0x80000000U);
    vcmp_ge(largeDivisor, (vector_u32&)rhsHigh, signBit, nonZeroMask);
    Int64CompareGeURegsSimd(lhsGeRhs, lhsLow, lhsHigh, rhsLow, rhsHigh, nonZeroMask);
    pand(largeResultOne, lhsGeRhs, largeDivisor, nonZeroMask);
    pnot(largeResultZero, lhsGeRhs, largeDivisor);
    pnot(smallDivisor, largeDivisor, nonZeroMask);
    pand(smallWorkMask, nonOneMask, smallDivisor, smallDivisor);
}

PTO_INTERNAL void Int64DivUnsignedRegsSimd(
    vector_s32& dstLow, vector_s32& dstHigh, vector_s32& lhsLow, vector_s32& lhsHigh, vector_s32& rhsLow,
    vector_s32& rhsHigh, MaskReg& inputMask)
{
    vector_s32 zeroLow, zeroHigh, oneLow, oneHigh, minusOneLow, minusOneHigh;
    Int64DuplicateRegsSimd(zeroLow, zeroHigh, 0, 0);
    Int64DuplicateRegsSimd(oneLow, oneHigh, 1, 0);
    Int64DuplicateRegsSimd(minusOneLow, minusOneHigh, 0xffffffffU, 0xffffffffU);
    MaskReg zeroMask, oneMask, workMask, largeOne, largeZero;
    Int64DivUnsignedClassifySimd(
        zeroMask, oneMask, workMask, largeOne, largeZero, lhsLow, lhsHigh, rhsLow, rhsHigh, zeroLow, zeroHigh, oneLow,
        oneHigh, inputMask);
    vector_f32 divisorFloat;
    vector_u32 reciprocalBits, zeroWord;
    vector_s32 reciprocalLow, reciprocalHigh, qLow, qHigh, tLow, tHigh, remLow, remHigh, adjustedLow, adjustedHigh;
    Int64ToFloatSimd(divisorFloat, rhsLow, rhsHigh);
    Int64DivFloatPreprocessSimd(reciprocalBits, divisorFloat, workMask);
    FloatToInt64Simd(reciprocalLow, reciprocalHigh, (vector_f32&)reciprocalBits);
    Int64MulRegsSimd(qLow, qHigh, rhsLow, rhsHigh, reciprocalLow, reciprocalHigh, workMask);
    Int64NotRegsSimd(qLow, qHigh, workMask);
    Int64AddRegsSimd(qLow, qHigh, qLow, qHigh, oneLow, oneHigh, workMask);
    vbr(zeroWord, 0);
    Int64B128CalcSimd(reciprocalLow, reciprocalHigh, qLow, qHigh, zeroWord, workMask);
    Int64AddRegsSimd(tLow, tHigh, reciprocalLow, reciprocalHigh, qLow, qHigh, workMask);
    Int64MulRegsSimd(qLow, qHigh, rhsLow, rhsHigh, tLow, tHigh, workMask);
    Int64NotRegsSimd(qLow, qHigh, workMask);
    Int64AddRegsSimd(qLow, qHigh, qLow, qHigh, oneLow, oneHigh, workMask);
    Int64B128CalcSimd(tLow, tHigh, qLow, qHigh, zeroWord, workMask);
    Int64AddRegsSimd(qLow, qHigh, tLow, tHigh, qLow, qHigh, workMask);
    Int64B128CalcSimd(lhsLow, lhsHigh, qLow, qHigh, zeroWord, workMask);
    Int64MulRegsSimd(tLow, tHigh, qLow, qHigh, rhsLow, rhsHigh, workMask);
    Int64SubRegsSimd(remLow, remHigh, lhsLow, lhsHigh, tLow, tHigh, workMask);
    MaskReg geMask;
    Int64CompareGeURegsSimd(geMask, remLow, remHigh, rhsLow, rhsHigh, workMask);
    Int64SubRegsSimd(adjustedLow, adjustedHigh, remLow, remHigh, rhsLow, rhsHigh, geMask);
    Int64AddRegsSimd(tLow, tHigh, qLow, qHigh, oneLow, oneHigh, geMask);
    Int64SelectRegsSimd(remLow, remHigh, adjustedLow, adjustedHigh, remLow, remHigh, geMask);
    Int64SelectRegsSimd(qLow, qHigh, tLow, tHigh, qLow, qHigh, geMask);
    Int64CompareGeURegsSimd(geMask, remLow, remHigh, rhsLow, rhsHigh, workMask);
    Int64AddRegsSimd(tLow, tHigh, qLow, qHigh, oneLow, oneHigh, geMask);
    Int64SelectRegsSimd(qLow, qHigh, tLow, tHigh, qLow, qHigh, geMask);
    Int64SelectRegsSimd(qLow, qHigh, oneLow, oneHigh, qLow, qHigh, largeOne);
    Int64SelectRegsSimd(qLow, qHigh, zeroLow, zeroHigh, qLow, qHigh, largeZero);
    Int64SelectRegsSimd(qLow, qHigh, lhsLow, lhsHigh, qLow, qHigh, oneMask);
    Int64SelectRegsSimd(dstLow, dstHigh, zeroLow, zeroHigh, qLow, qHigh, zeroMask);
}

template <typename T, unsigned DstCols, unsigned Src0Cols, unsigned Src1Cols>
PTO_INTERNAL void Int64DivSimd(
    __ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, unsigned validRows, unsigned validCols)
{
    __VEC_SCOPE__
    {
        vector_s32 dstLow, dstHigh, lhsLow, lhsHigh, rhsLow, rhsHigh;
        uint16_t rows = validRows;
        for (uint16_t row = 0; row < rows; ++row) {
            uint32_t cols = validCols;
            MaskReg mask = plt_b32(cols, POST_UPDATE);
            vlds(lhsLow, lhsHigh, (__ubuf__ int32_t*)src0, row * Src0Cols * 2, DINTLV_B32);
            vlds(rhsLow, rhsHigh, (__ubuf__ int32_t*)src1, row * Src1Cols * 2, DINTLV_B32);
            if constexpr (std::is_same_v<T, int64_t>)
                Int64DivSignedRegsSimd(dstLow, dstHigh, lhsLow, lhsHigh, rhsLow, rhsHigh, mask);
            else
                Int64DivUnsignedRegsSimd(dstLow, dstHigh, lhsLow, lhsHigh, rhsLow, rhsHigh, mask);
            vsts(dstLow, dstHigh, (__ubuf__ int32_t*)dst + row * DstCols * 2, 0, INTLV_B32, mask);
        }
    }
}

template <bool ScalarFirst, typename T, unsigned DstCols, unsigned SrcCols>
PTO_INTERNAL void Int64DivScalarSimd(__ubuf__ T* dst, __ubuf__ T* src, T scalar, unsigned validRows, unsigned validCols)
{
    __VEC_SCOPE__
    {
        vector_s32 dstLow, dstHigh, srcLow, srcHigh, scalarLow, scalarHigh;
        uint64_t scalarBits = static_cast<uint64_t>(scalar);
        Int64DuplicateRegsSimd(
            scalarLow, scalarHigh, static_cast<uint32_t>(scalarBits), static_cast<uint32_t>(scalarBits >> 32));
        uint16_t rows = validRows;
        for (uint16_t row = 0; row < rows; ++row) {
            uint32_t cols = validCols;
            MaskReg mask = plt_b32(cols, POST_UPDATE);
            vlds(srcLow, srcHigh, (__ubuf__ int32_t*)src + row * SrcCols * 2, 0, DINTLV_B32);
            if constexpr (ScalarFirst) {
                if constexpr (std::is_same_v<T, int64_t>)
                    Int64DivSignedRegsSimd(dstLow, dstHigh, scalarLow, scalarHigh, srcLow, srcHigh, mask);
                else
                    Int64DivUnsignedRegsSimd(dstLow, dstHigh, scalarLow, scalarHigh, srcLow, srcHigh, mask);
            } else {
                if constexpr (std::is_same_v<T, int64_t>)
                    Int64DivSignedRegsSimd(dstLow, dstHigh, srcLow, srcHigh, scalarLow, scalarHigh, mask);
                else
                    Int64DivUnsignedRegsSimd(dstLow, dstHigh, srcLow, srcHigh, scalarLow, scalarHigh, mask);
            }
            vsts(dstLow, dstHigh, (__ubuf__ int32_t*)dst + row * DstCols * 2, 0, INTLV_B32, mask);
        }
    }
}

template <typename T>
PTO_INTERNAL void Int64RemRegsSimd(
    vector_s32& dstLow, vector_s32& dstHigh, vector_s32& lhsLow, vector_s32& lhsHigh, vector_s32& rhsLow,
    vector_s32& rhsHigh, MaskReg& mask)
{
    vector_s32 qLow, qHigh, productLow, productHigh;
    if constexpr (std::is_same_v<T, int64_t>)
        Int64DivSignedRegsSimd(qLow, qHigh, lhsLow, lhsHigh, rhsLow, rhsHigh, mask);
    else
        Int64DivUnsignedRegsSimd(qLow, qHigh, lhsLow, lhsHigh, rhsLow, rhsHigh, mask);
    Int64MulRegsSimd(productLow, productHigh, qLow, qHigh, rhsLow, rhsHigh, mask);
    Int64SubRegsSimd(dstLow, dstHigh, lhsLow, lhsHigh, productLow, productHigh, mask);
    vector_s32 zeroLow, zeroHigh;
    Int64DuplicateRegsSimd(zeroLow, zeroHigh, 0, 0);
    MaskReg zeroMask;
    Int64CompareEqRegsSimd(zeroMask, rhsLow, rhsHigh, zeroLow, zeroHigh, mask);
    Int64SelectRegsSimd(dstLow, dstHigh, zeroLow, zeroHigh, dstLow, dstHigh, zeroMask);
}

template <typename T, unsigned DstCols, unsigned Src0Cols, unsigned Src1Cols>
PTO_INTERNAL void Int64RemSimd(
    __ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, unsigned validRows, unsigned validCols)
{
    __VEC_SCOPE__
    {
        vector_s32 dl, dh, al, ah, bl, bh;
        uint16_t rows = validRows;
        for (uint16_t row = 0; row < rows; ++row) {
            uint32_t cols = validCols;
            MaskReg mask = plt_b32(cols, POST_UPDATE);
            vlds(al, ah, (__ubuf__ int32_t*)src0 + row * Src0Cols * 2, 0, DINTLV_B32);
            vlds(bl, bh, (__ubuf__ int32_t*)src1 + row * Src1Cols * 2, 0, DINTLV_B32);
            Int64RemRegsSimd<T>(dl, dh, al, ah, bl, bh, mask);
            vsts(dl, dh, (__ubuf__ int32_t*)dst + row * DstCols * 2, 0, INTLV_B32, mask);
        }
    }
}

template <typename T, unsigned DstCols>
PTO_INTERNAL void Int64ZeroSimd(__ubuf__ T* dst, unsigned validRows, unsigned validCols)
{
    __VEC_SCOPE__
    {
        vector_s32 zero;
        vbr(zero, 0);
        uint16_t rows = validRows;
        for (uint16_t row = 0; row < rows; ++row) {
            uint32_t cols = validCols;
            MaskReg mask = plt_b32(cols, POST_UPDATE);
            vsts(zero, zero, (__ubuf__ int32_t*)dst + row * DstCols * 2, 0, INTLV_B32, mask);
        }
    }
}

template <typename T, unsigned DstCols, unsigned SrcCols>
PTO_INTERNAL void Int64RemScalarSimd(__ubuf__ T* dst, __ubuf__ T* src, T scalar, unsigned validRows, unsigned validCols)
{
    if constexpr (std::is_same_v<T, uint64_t>) {
        if (scalar == 0) {
            Int64ZeroSimd<T, DstCols>(dst, validRows, validCols);
            return;
        }
        Int64DivScalarSimd<false, T, DstCols, SrcCols>(dst, src, scalar, validRows, validCols);
        __VEC_SCOPE__
        {
            vector_s32 quotientLow, quotientHigh, srcLow, srcHigh, scalarLow, scalarHigh, productLow, productHigh;
            vector_s32 dstLow, dstHigh;
            uint64_t bits = static_cast<uint64_t>(scalar);
            Int64DuplicateRegsSimd(
                scalarLow, scalarHigh, static_cast<uint32_t>(bits), static_cast<uint32_t>(bits >> 32));
            uint16_t rows = validRows;
            for (uint16_t row = 0; row < rows; ++row) {
                uint32_t cols = validCols;
                MaskReg mask = plt_b32(cols, POST_UPDATE);
                vlds(srcLow, srcHigh, (__ubuf__ int32_t*)src + row * SrcCols * 2, 0, DINTLV_B32);
                vlds(quotientLow, quotientHigh, (__ubuf__ int32_t*)dst + row * DstCols * 2, 0, DINTLV_B32);
                Int64MulRegsSimd(productLow, productHigh, quotientLow, quotientHigh, scalarLow, scalarHigh, mask);
                Int64SubRegsSimd(dstLow, dstHigh, srcLow, srcHigh, productLow, productHigh, mask);
                vsts(dstLow, dstHigh, (__ubuf__ int32_t*)dst + row * DstCols * 2, 0, INTLV_B32, mask);
            }
        }
        return;
    }
    __VEC_SCOPE__
    {
        vector_s32 dl, dh, al, ah, bl, bh;
        uint64_t bits = static_cast<uint64_t>(scalar);
        Int64DuplicateRegsSimd(bl, bh, static_cast<uint32_t>(bits), static_cast<uint32_t>(bits >> 32));
        uint16_t rows = validRows;
        for (uint16_t row = 0; row < rows; ++row) {
            uint32_t cols = validCols;
            MaskReg mask = plt_b32(cols, POST_UPDATE);
            vlds(al, ah, (__ubuf__ int32_t*)src + row * SrcCols * 2, 0, DINTLV_B32);
            Int64RemRegsSimd<T>(dl, dh, al, ah, bl, bh, mask);
            vsts(dl, dh, (__ubuf__ int32_t*)dst + row * DstCols * 2, 0, INTLV_B32, mask);
        }
    }
}

} // namespace pto

#endif
