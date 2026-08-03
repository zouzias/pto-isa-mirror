#ifndef INT64_SIMD_HPP
#define INT64_SIMD_HPP

#include <pto/npu/a5/common.hpp>

namespace pto {

enum class Int64SimdOp { Add, Sub, Mul, Shl, Shr, Max, Min };

PTO_INTERNAL void Int64AddRegsSimd(
    vector_s32& dstLow, vector_s32& dstHigh, vector_s32& lhsLow, vector_s32& lhsHigh, vector_s32& rhsLow,
    vector_s32& rhsHigh, MaskReg& mask)
{
    MaskReg carry, carryOut;
    vaddc(carry, dstLow, lhsLow, rhsLow, mask);
    vaddcs(carryOut, dstHigh, lhsHigh, rhsHigh, carry, mask);
}

PTO_INTERNAL void Int64SubRegsSimd(
    vector_s32& dstLow, vector_s32& dstHigh, vector_s32& lhsLow, vector_s32& lhsHigh, vector_s32& rhsLow,
    vector_s32& rhsHigh, MaskReg& mask)
{
    MaskReg carry, carryOut;
    vsubc(carry, dstLow, lhsLow, rhsLow, mask);
    vsubcs(carryOut, dstHigh, lhsHigh, rhsHigh, carry, mask);
}

PTO_INTERNAL void Int64MulRegsSimd(
    vector_s32& dstLow, vector_s32& dstHigh, vector_s32& lhsLow, vector_s32& lhsHigh, vector_s32& rhsLow,
    vector_s32& rhsHigh, MaskReg& mask)
{
    vmull((vector_u32&)dstLow, (vector_u32&)dstHigh, (vector_u32&)lhsLow, (vector_u32&)rhsLow, mask);
    vmula(dstHigh, lhsLow, rhsHigh, mask, MODE_ZEROING);
    vmula(dstHigh, lhsHigh, rhsLow, mask, MODE_ZEROING);
}

PTO_INTERNAL void Int64SelectRegsSimd(
    vector_s32& dstLow, vector_s32& dstHigh, vector_s32& lhsLow, vector_s32& lhsHigh, vector_s32& rhsLow,
    vector_s32& rhsHigh, MaskReg& mask)
{
    vsel(dstLow, lhsLow, rhsLow, mask);
    vsel(dstHigh, lhsHigh, rhsHigh, mask);
}

PTO_INTERNAL void Int64CompareEqRegsSimd(
    MaskReg& dst, vector_s32& lhsLow, vector_s32& lhsHigh, vector_s32& rhsLow, vector_s32& rhsHigh, MaskReg& mask)
{
    MaskReg lowEq, highEq;
    vcmp_eq(lowEq, (vector_u32&)lhsLow, (vector_u32&)rhsLow, mask);
    vcmp_eq(highEq, (vector_u32&)lhsHigh, (vector_u32&)rhsHigh, mask);
    pand(dst, lowEq, highEq, mask);
}

PTO_INTERNAL void Int64CompareGeURegsSimd(
    MaskReg& dst, vector_s32& lhsLow, vector_s32& lhsHigh, vector_s32& rhsLow, vector_s32& rhsHigh, MaskReg& mask)
{
    MaskReg highEq, lowGe, highGe;
    vcmp_eq(highEq, (vector_u32&)lhsHigh, (vector_u32&)rhsHigh, mask);
    vcmp_ge(lowGe, (vector_u32&)lhsLow, (vector_u32&)rhsLow, mask);
    vcmp_ge(highGe, (vector_u32&)lhsHigh, (vector_u32&)rhsHigh, mask);
    psel(dst, lowGe, highGe, highEq);
}

PTO_INTERNAL void Int64AbsRegsSimd(
    vector_s32& dstLow, vector_s32& dstHigh, vector_s32& srcLow, vector_s32& srcHigh, MaskReg& mask)
{
    vector_s32 zero, negLow, negHigh;
    vbr(zero, 0);
    MaskReg negative, carry, carryOut;
    vcmp_lt(negative, srcHigh, zero, mask);
    vsubc(carry, negLow, zero, srcLow, negative);
    vsubcs(carryOut, negHigh, zero, srcHigh, carry, negative);
    vsel(dstLow, negLow, srcLow, negative);
    vsel(dstHigh, negHigh, srcHigh, negative);
}

PTO_INTERNAL void Int64NotRegsSimd(vector_s32& low, vector_s32& high, MaskReg& mask)
{
    vnot((vector_u32&)low, (vector_u32&)low, mask, MODE_ZEROING);
    vnot((vector_u32&)high, (vector_u32&)high, mask, MODE_ZEROING);
}

PTO_INTERNAL void Int64ToFloatSimd(vector_f32& dst, vector_s32& srcLow, vector_s32& srcHigh)
{
    vector_s64 even, odd;
    vector_f32 evenFloat, oddFloat;
    vintlv((vector_s32&)even, (vector_s32&)odd, srcLow, srcHigh);
    MaskReg allMask = pset_b32(PAT_ALL);
    vcvt(evenFloat, even, allMask, ROUND_R, PART_EVEN);
    vcvt(oddFloat, odd, allMask, ROUND_R, PART_EVEN);
    vdintlv(dst, oddFloat, evenFloat, oddFloat);
}

PTO_INTERNAL void FloatToInt64Simd(vector_s32& dstLow, vector_s32& dstHigh, vector_f32& src)
{
    vector_f32 evenFloat, oddFloat;
    vector_s64 even, odd;
    vintlv(evenFloat, oddFloat, src, src);
    MaskReg allMask = pset_b32(PAT_ALL);
    vcvt(even, evenFloat, allMask, ROUND_Z, RS_DISABLE, PART_EVEN);
    vcvt(odd, oddFloat, allMask, ROUND_Z, RS_DISABLE, PART_EVEN);
    vdintlv(dstLow, dstHigh, (vector_s32&)even, (vector_s32&)odd);
}

PTO_INTERNAL void Int64DivFloatPreprocessSimd(vector_u32& reciprocalBits, vector_f32& divisorFloat, MaskReg& mask)
{
    vector_f32 one;
    vbr(one, 1.0f);
    vdiv(one, one, divisorFloat, mask, MODE_ZEROING);
    vadds(reciprocalBits, (vector_u32&)one, 0x1ffffffeU, mask);
}

PTO_INTERNAL void Int64DuplicateRegsSimd(vector_s32& low, vector_s32& high, uint32_t lowScalar, uint32_t highScalar)
{
    vbr((vector_u32&)low, lowScalar);
    vbr((vector_u32&)high, highScalar);
}

PTO_INTERNAL void Int64DivSignSimd(
    MaskReg& sameSign, vector_s32& lhsHigh, vector_s32& rhsHigh, vector_s32& zero, MaskReg& mask)
{
    MaskReg lhsNonNegative, rhsNonNegative;
    vcmp_ge(lhsNonNegative, lhsHigh, zero, mask);
    vcmp_ge(rhsNonNegative, rhsHigh, zero, mask);
    pxor(sameSign, lhsNonNegative, rhsNonNegative, mask);
    pnot(sameSign, sameSign, mask);
}

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

    vector_f32 divisorFloat;
    vector_u32 reciprocalBits, zeroWord;
    vector_s32 reciprocalLow, reciprocalHigh;
    Int64ToFloatSimd(divisorFloat, absRhsLow, absRhsHigh);
    Int64DivFloatPreprocessSimd(reciprocalBits, divisorFloat, workMask);
    FloatToInt64Simd(reciprocalLow, reciprocalHigh, (vector_f32&)reciprocalBits);

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

    MaskReg sameSign;
    Int64DivSignSimd(sameSign, lhsHigh, rhsHigh, zeroHigh, nonZeroMask);
    MaskReg allMask = pset_b32(PAT_ALL);
    Int64SubRegsSimd(tLow, tHigh, zeroLow, zeroHigh, qLow, qHigh, allMask);
    Int64SelectRegsSimd(qLow, qHigh, qLow, qHigh, tLow, tHigh, sameSign);
    Int64SelectRegsSimd(dstLow, dstHigh, zeroLow, zeroHigh, qLow, qHigh, zeroMask);
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

template <typename T, unsigned DstCols, unsigned SrcCols>
PTO_INTERNAL void Int64RemScalarSimd(__ubuf__ T* dst, __ubuf__ T* src, T scalar, unsigned validRows, unsigned validCols)
{
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

template <typename T>
PTO_INTERNAL void Int64CompareRegsSimd(
    MaskReg& dst, vector_s32& lhsLow, vector_s32& lhsHigh, vector_s32& rhsLow, vector_s32& rhsHigh, CmpMode mode,
    MaskReg& mask)
{
    MaskReg lowEq, highCmp, lowCmp;
    switch (mode) {
        case CmpMode::EQ:
            vcmp_eq(lowEq, (vector_u32&)lhsLow, (vector_u32&)rhsLow, mask);
            vcmp_eq(dst, lhsHigh, rhsHigh, lowEq);
            break;
        case CmpMode::NE: {
            MaskReg highNe;
            vcmp_ne(lowEq, (vector_u32&)lhsLow, (vector_u32&)rhsLow, mask);
            vcmp_ne(highNe, lhsHigh, rhsHigh, mask);
            por(dst, lowEq, highNe, mask);
            break;
        }
        default: {
            vcmp_eq(lowEq, lhsHigh, rhsHigh, mask);
            if (mode == CmpMode::LT || mode == CmpMode::LE) {
                if (mode == CmpMode::LT)
                    vcmp_lt(lowCmp, (vector_u32&)lhsLow, (vector_u32&)rhsLow, mask);
                else
                    vcmp_le(lowCmp, (vector_u32&)lhsLow, (vector_u32&)rhsLow, mask);
                if constexpr (std::is_same_v<T, int64_t>) {
                    if (mode == CmpMode::LT)
                        vcmp_lt(highCmp, lhsHigh, rhsHigh, mask);
                    else
                        vcmp_le(highCmp, lhsHigh, rhsHigh, mask);
                } else {
                    if (mode == CmpMode::LT)
                        vcmp_lt(highCmp, (vector_u32&)lhsHigh, (vector_u32&)rhsHigh, mask);
                    else
                        vcmp_le(highCmp, (vector_u32&)lhsHigh, (vector_u32&)rhsHigh, mask);
                }
            } else {
                if (mode == CmpMode::GT)
                    vcmp_gt(lowCmp, (vector_u32&)lhsLow, (vector_u32&)rhsLow, mask);
                else
                    vcmp_ge(lowCmp, (vector_u32&)lhsLow, (vector_u32&)rhsLow, mask);
                if constexpr (std::is_same_v<T, int64_t>) {
                    if (mode == CmpMode::GT)
                        vcmp_gt(highCmp, lhsHigh, rhsHigh, mask);
                    else
                        vcmp_ge(highCmp, lhsHigh, rhsHigh, mask);
                } else {
                    if (mode == CmpMode::GT)
                        vcmp_gt(highCmp, (vector_u32&)lhsHigh, (vector_u32&)rhsHigh, mask);
                    else
                        vcmp_ge(highCmp, (vector_u32&)lhsHigh, (vector_u32&)rhsHigh, mask);
                }
            }
            psel(dst, lowCmp, highCmp, lowEq);
            break;
        }
    }
}

template <typename T, unsigned DstRowBytes, unsigned Src0Cols, unsigned Src1Cols>
PTO_INTERNAL void Int64CompareSimd(
    __ubuf__ uint8_t* dst, __ubuf__ T* src0, __ubuf__ T* src1, CmpMode mode, unsigned validRows, unsigned validCols)
{
    __VEC_SCOPE__
    {
        vector_s32 lhsLow, lhsHigh, rhsLow, rhsHigh;
        uint16_t rows = validRows;
        for (uint16_t row = 0; row < rows; ++row) {
            uint32_t cols = validCols * 4;
            MaskReg mask = plt_b32(cols, POST_UPDATE);
            uint32_t packedCols = validCols;
            MaskReg packedMask = plt_b8(packedCols, POST_UPDATE);
            MaskReg result;
            vlds(lhsLow, lhsHigh, (__ubuf__ int32_t*)src0 + row * Src0Cols * 2, 0, DINTLV_B32);
            vlds(rhsLow, rhsHigh, (__ubuf__ int32_t*)src1 + row * Src1Cols * 2, 0, DINTLV_B32);
            Int64CompareRegsSimd<T>(result, lhsLow, lhsHigh, rhsLow, rhsHigh, mode, mask);
            ppack(result, result, LOWER);
            ppack(result, result, LOWER);
            pand(result, result, packedMask, packedMask);
            psts(result, (__ubuf__ uint32_t*)(dst + row * DstRowBytes), 0, NORM);
        }
    }
}

template <typename T, unsigned DstRowBytes, unsigned SrcCols>
PTO_INTERNAL void Int64CompareScalarSimd(
    __ubuf__ uint8_t* dst, __ubuf__ T* src, T scalar, CmpMode mode, unsigned validRows, unsigned validCols)
{
    __VEC_SCOPE__
    {
        vector_s32 lhsLow, lhsHigh, rhsLow, rhsHigh;
        vbr(rhsLow, static_cast<uint32_t>(scalar));
        vbr(rhsHigh, static_cast<uint32_t>(static_cast<uint64_t>(scalar) >> 32));
        uint16_t rows = validRows;
        for (uint16_t row = 0; row < rows; ++row) {
            uint32_t cols = validCols * 4;
            MaskReg mask = plt_b32(cols, POST_UPDATE);
            uint32_t packedCols = validCols;
            MaskReg packedMask = plt_b8(packedCols, POST_UPDATE);
            MaskReg result;
            vlds(lhsLow, lhsHigh, (__ubuf__ int32_t*)src, row * SrcCols * 2, DINTLV_B32);
            Int64CompareRegsSimd<T>(result, lhsLow, lhsHigh, rhsLow, rhsHigh, mode, mask);
            ppack(result, result, LOWER);
            ppack(result, result, LOWER);
            pand(result, result, packedMask, packedMask);
            psts(result, (__ubuf__ uint32_t*)(dst + row * DstRowBytes), 0, NORM);
        }
    }
}

template <MaskPattern Pattern>
PTO_INTERNAL constexpr unsigned Int64MaskPatternOffset()
{
    if constexpr (Pattern == MaskPattern::P1010 || Pattern == MaskPattern::P0010)
        return 1;
    if constexpr (Pattern == MaskPattern::P0100)
        return 2;
    if constexpr (Pattern == MaskPattern::P1000)
        return 3;
    return 0;
}

template <bool Right, typename T>
PTO_INTERNAL void Int64ShiftSimdRegs(
    vector_s32& dl, vector_s32& dh, vector_s32& sl, vector_s32& sh, vector_s32& cnt, MaskReg& mask)
{
    vector_s32 c32, cm, bias, norm, sixtyThree;
    vbr(bias, 32);
    vbr(sixtyThree, 63);
    vand((vector_u32&)norm, (vector_u32&)cnt, (vector_u32&)sixtyThree, mask, MODE_ZEROING);
    vadds(c32, norm, 32, mask);
    vsub(cm, bias, norm, mask);
    MaskReg lt32, ge32;
    vcmp_lt(lt32, norm, bias, mask);
    vcmp_ge(ge32, norm, bias, mask);
    if constexpr (!Right) {
        vector_s32 lo0, hi0, lo1, hi1, t;
        vshl(lo0, sl, norm, mask, MODE_ZEROING);
        vshl(hi0, sh, norm, mask, MODE_ZEROING);
        vshr(t, sl, cm, mask, MODE_ZEROING);
        vor(hi0, hi0, t, mask);
        vsub(c32, norm, bias, mask);
        vshl(hi1, sl, c32, mask, MODE_ZEROING);
        vbr(lo1, 0);
        vsel(dl, lo0, lo1, lt32);
        vsel(dh, hi0, hi1, lt32);
    } else {
        vector_s32 lo0, hi0, lo1, hi1, t;
        if constexpr (std::is_same_v<T, int64_t>)
            vshr(hi0, sh, norm, mask, MODE_ZEROING);
        else
            vshr((vector_u32&)hi0, (vector_u32&)sh, norm, mask, MODE_ZEROING);
        vshr((vector_u32&)lo0, (vector_u32&)sl, norm, mask, MODE_ZEROING);
        vshl(t, sh, cm, mask, MODE_ZEROING);
        vor(lo0, lo0, t, mask);
        vsub(c32, norm, bias, mask);
        if constexpr (std::is_same_v<T, int64_t>) {
            vshr(lo1, sh, c32, mask, MODE_ZEROING);
            vshrs(hi1, sh, 31, mask, MODE_ZEROING);
        } else {
            vshr((vector_u32&)lo1, (vector_u32&)sh, c32, mask, MODE_ZEROING);
            vbr(hi1, 0);
        }
        vsel(dl, lo0, lo1, lt32);
        vsel(dh, hi0, hi1, lt32);
    }
}

template <Int64SimdOp Op, typename T>
PTO_INTERNAL void Int64MinMaxSimd(
    vector_s32& dstLow, vector_s32& dstHigh, vector_s32& src0Low, vector_s32& src0High, vector_s32& src1Low,
    vector_s32& src1High, MaskReg& mask)
{
    MaskReg highEq, lowCmp, highCmp, selectMask;
    vcmp_eq(highEq, src0High, src1High, mask);
    if constexpr (Op == Int64SimdOp::Max) {
        vcmp_gt(lowCmp, (vector_u32&)src0Low, (vector_u32&)src1Low, mask);
        if constexpr (std::is_same_v<T, int64_t>) {
            vcmp_gt(highCmp, src0High, src1High, mask);
        } else {
            vcmp_gt(highCmp, (vector_u32&)src0High, (vector_u32&)src1High, mask);
        }
    } else {
        vcmp_lt(lowCmp, (vector_u32&)src0Low, (vector_u32&)src1Low, mask);
        if constexpr (std::is_same_v<T, int64_t>) {
            vcmp_lt(highCmp, src0High, src1High, mask);
        } else {
            vcmp_lt(highCmp, (vector_u32&)src0High, (vector_u32&)src1High, mask);
        }
    }
    psel(selectMask, lowCmp, highCmp, highEq);
    vsel(dstLow, src0Low, src1Low, selectMask);
    vsel(dstHigh, src0High, src1High, selectMask);
}

template <Int64SimdOp Op, typename T, unsigned DstCols, unsigned Src0Cols, unsigned Src1Cols>
PTO_INTERNAL void Int64BinarySimd(
    __ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, unsigned validRows, unsigned validCols)
{
    __VEC_SCOPE__
    {
        vector_s32 dstLow, dstHigh, src0Low, src0High, src1Low, src1High;
        uint32_t maskCount = validCols;
        MaskReg mask = plt_b32(maskCount, POST_UPDATE);
        uint16_t rowCount = validRows;
        for (uint16_t row = 0; row < rowCount; ++row) {
            vlds(src0Low, src0High, (__ubuf__ int32_t*)src0, row * Src0Cols * 2, DINTLV_B32);
            vlds(src1Low, src1High, (__ubuf__ int32_t*)src1, row * Src1Cols * 2, DINTLV_B32);
            MaskReg carry;
            MaskReg carryOut;
            if constexpr (Op == Int64SimdOp::Add) {
                vaddc(carry, dstLow, src0Low, src1Low, mask);
                vaddcs(carryOut, dstHigh, src0High, src1High, carry, mask);
            } else if constexpr (Op == Int64SimdOp::Sub) {
                vsubc(carry, dstLow, src0Low, src1Low, mask);
                vsubcs(carryOut, dstHigh, src0High, src1High, carry, mask);
            } else if constexpr (Op == Int64SimdOp::Mul) {
                vmull((vector_u32&)dstLow, (vector_u32&)dstHigh, (vector_u32&)src0Low, (vector_u32&)src1Low, mask);
                vmula(dstHigh, src0Low, src1High, mask, MODE_ZEROING);
                vmula(dstHigh, src0High, src1Low, mask, MODE_ZEROING);
            } else if constexpr (Op == Int64SimdOp::Shl || Op == Int64SimdOp::Shr) {
                Int64ShiftSimdRegs<Op == Int64SimdOp::Shr, T>(dstLow, dstHigh, src0Low, src0High, src1Low, mask);
            } else {
                Int64MinMaxSimd<Op, T>(dstLow, dstHigh, src0Low, src0High, src1Low, src1High, mask);
            }
            vsts(dstLow, dstHigh, (__ubuf__ int32_t*)dst + row * DstCols * 2, 0, INTLV_B32, mask);
        }
    }
}

template <Int64SimdOp Op, typename T, unsigned DstCols, unsigned Src0Cols, unsigned Src1Cols>
PTO_INTERNAL void Int64PartSimd(
    __ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, unsigned src0Rows, unsigned src0Cols, unsigned src1Rows,
    unsigned src1Cols, unsigned dstRows, unsigned dstCols)
{
    __VEC_SCOPE__
    {
        vector_s32 dl, dh, al, ah, bl, bh;
        uint16_t rows = dstRows;
        for (uint16_t row = 0; row < rows; ++row) {
            uint32_t cols = dstCols;
            MaskReg storeMask = plt_b32(cols, POST_UPDATE);
            if (row < src0Rows)
                vlds(al, ah, (__ubuf__ int32_t*)src0 + row * Src0Cols * 2, 0, DINTLV_B32);
            if (row < src1Rows)
                vlds(bl, bh, (__ubuf__ int32_t*)src1 + row * Src1Cols * 2, 0, DINTLV_B32);
            if (row < src0Rows && row < src1Rows) {
                uint32_t overlap = src0Cols < src1Cols ? src0Cols : src1Cols;
                MaskReg opMask = plt_b32(overlap, POST_UPDATE);
                vector_s32 ol, oh;
                if constexpr (Op == Int64SimdOp::Add) {
                    MaskReg carry, carryOut;
                    vaddc(carry, ol, al, bl, opMask);
                    vaddcs(carryOut, oh, ah, bh, carry, opMask);
                } else {
                    Int64MinMaxSimd<Op, T>(ol, oh, al, ah, bl, bh, opMask);
                }
                if (src0Cols >= src1Cols) {
                    vsel(dl, ol, al, opMask);
                    vsel(dh, oh, ah, opMask);
                } else {
                    vsel(dl, ol, bl, opMask);
                    vsel(dh, oh, bh, opMask);
                }
            } else if (row < src0Rows) {
                dl = al;
                dh = ah;
            } else {
                dl = bl;
                dh = bh;
            }
            vsts(dl, dh, (__ubuf__ int32_t*)dst + row * DstCols * 2, 0, INTLV_B32, storeMask);
        }
    }
}

template <Int64SimdOp Op, typename T, unsigned DstCols, unsigned SrcCols>
PTO_INTERNAL void Int64ColReduceSimd(__ubuf__ T* dst, __ubuf__ T* src, unsigned validRows, unsigned validCols)
{
    __VEC_SCOPE__
    {
        vector_s32 dl, dh, sl, sh, nl, nh;
        uint32_t cols = validCols;
        MaskReg mask = plt_b32(cols, POST_UPDATE);
        vlds(dl, dh, (__ubuf__ int32_t*)src, 0, DINTLV_B32);
        uint16_t rows = validRows;
        for (uint16_t row = 1; row < rows; ++row) {
            vlds(sl, sh, (__ubuf__ int32_t*)src + row * SrcCols * 2, 0, DINTLV_B32);
            if constexpr (Op == Int64SimdOp::Add) {
                MaskReg carry, carryOut;
                vaddc(carry, nl, dl, sl, mask);
                vaddcs(carryOut, nh, dh, sh, carry, mask);
            } else {
                Int64MinMaxSimd<Op, T>(nl, nh, dl, dh, sl, sh, mask);
            }
            dl = nl;
            dh = nh;
        }
        vsts(dl, dh, (__ubuf__ int32_t*)dst, 0, INTLV_B32, mask);
    }
}

template <typename T, unsigned DstCols, unsigned SrcCols>
PTO_INTERNAL void Int64RowSumSimd(__ubuf__ T* dst, __ubuf__ T* src, unsigned validRows, unsigned validCols)
{
    __VEC_SCOPE__
    {
        vector_u32 low, high, low16, mid16, tmp, mask16, outLow, outHigh;
        vbr(mask16, 0xffffu);
        uint32_t cols = validCols;
        MaskReg mask = plt_b32(cols, POST_UPDATE);
        MaskReg oneMask = pset_b32(PAT_VL1);
        uint16_t rows = validRows;
        for (uint16_t row = 0; row < rows; ++row) {
            vlds((vector_s32&)low, (vector_s32&)high, (__ubuf__ int32_t*)src + row * SrcCols * 2, 0, DINTLV_B32);
            vand(low16, low, mask16, mask, MODE_ZEROING);
            vcadd(low16, low16, mask, MODE_ZEROING);
            vshrs(mid16, low, 16, mask, MODE_ZEROING);
            vcadd(mid16, mid16, mask, MODE_ZEROING);
            vcadd(outHigh, high, mask, MODE_ZEROING);
            vshrs(tmp, low16, 16, mask, MODE_ZEROING);
            vadd(mid16, mid16, tmp, mask, MODE_ZEROING);
            vshrs(tmp, mid16, 16, mask, MODE_ZEROING);
            vadd(outHigh, outHigh, tmp, mask, MODE_ZEROING);
            vand(low16, low16, mask16, mask, MODE_ZEROING);
            vand(mid16, mid16, mask16, mask, MODE_ZEROING);
            vshls(mid16, mid16, 16, mask, MODE_ZEROING);
            vor(outLow, low16, mid16, mask);
            vsts(
                (vector_s32&)outLow, (vector_s32&)outHigh, (__ubuf__ int32_t*)dst + row * DstCols * 2, 0, INTLV_B32,
                oneMask);
        }
    }
}

template <Int64SimdOp Op, typename T, unsigned DstCols, unsigned SrcCols>
PTO_INTERNAL void Int64RowMinMaxSimd(__ubuf__ T* dst, __ubuf__ T* src, unsigned validRows, unsigned validCols)
{
    __VEC_SCOPE__
    {
        vector_s32 low, high, reducedHigh, highDup, outputHigh;
        vector_u32 reducedLow;
        uint32_t cols = validCols;
        MaskReg mask = plt_b32(cols, POST_UPDATE);
        MaskReg allMask = pset_b32(PAT_ALL);
        uint16_t rows = validRows;
        for (uint16_t row = 0; row < rows; ++row) {
            vlds(low, high, (__ubuf__ int32_t*)src + row * SrcCols * 2, 0, DINTLV_B32);
            if constexpr (Op == Int64SimdOp::Max) {
                if constexpr (std::is_same_v<T, int64_t>)
                    vcmax(reducedHigh, high, mask, MODE_ZEROING);
                else
                    vcmax((vector_u32&)reducedHigh, (vector_u32&)high, mask, MODE_ZEROING);
            } else {
                if constexpr (std::is_same_v<T, int64_t>)
                    vcmin(reducedHigh, high, mask, MODE_ZEROING);
                else
                    vcmin((vector_u32&)reducedHigh, (vector_u32&)high, mask, MODE_ZEROING);
            }
            vdup(highDup, reducedHigh, allMask, POS_LOWEST, MODE_ZEROING);
            MaskReg equalHigh;
            vcmp_eq(equalHigh, highDup, high, mask);
            if constexpr (Op == Int64SimdOp::Max)
                vcmax(reducedLow, (vector_u32&)low, equalHigh, MODE_ZEROING);
            else
                vcmin(reducedLow, (vector_u32&)low, equalHigh, MODE_ZEROING);
            __ubuf__ int32_t* out = (__ubuf__ int32_t*)dst + row * DstCols * 2;
            MaskReg lowStoreMask = pset_b32(PAT_VL1);
            MaskReg highStoreMask = pset_b32(PAT_VL1);
            vand(
                (vector_u32&)outputHigh, (vector_u32&)reducedHigh, (vector_u32&)reducedHigh, highStoreMask,
                MODE_ZEROING);
            vsts((vector_s32&)reducedLow, out, 0, NORM_B32, lowStoreMask);
            vsts(outputHigh, out + 1, 0, NORM_B32, highStoreMask);
        }
    }
}

template <Int64SimdOp Op, typename T, unsigned DstCols, unsigned SrcCols>
PTO_INTERNAL void Int64ScalarSimd(__ubuf__ T* dst, __ubuf__ T* src, T scalar, unsigned validRows, unsigned validCols)
{
    __VEC_SCOPE__
    {
        vector_s32 dstLow, dstHigh, srcLow, srcHigh, scalarLow, scalarHigh;
        uint64_t scalarBits = static_cast<uint64_t>(scalar);
        int32_t low = static_cast<int32_t>(scalarBits);
        int32_t high = static_cast<int32_t>(scalarBits >> 32);
        vbr(scalarLow, low);
        vbr(scalarHigh, high);
        uint32_t maskCount = validCols;
        MaskReg mask = plt_b32(maskCount, POST_UPDATE);
        uint16_t rowCount = validRows;
        for (uint16_t row = 0; row < rowCount; ++row) {
            vlds(srcLow, srcHigh, (__ubuf__ int32_t*)src + row * SrcCols * 2, 0, DINTLV_B32);
            MaskReg carry;
            MaskReg carryOut;
            if constexpr (Op == Int64SimdOp::Add) {
                vaddc(carry, dstLow, srcLow, scalarLow, mask);
                vaddcs(carryOut, dstHigh, srcHigh, scalarHigh, carry, mask);
            } else if constexpr (Op == Int64SimdOp::Sub) {
                vsubc(carry, dstLow, srcLow, scalarLow, mask);
                vsubcs(carryOut, dstHigh, srcHigh, scalarHigh, carry, mask);
            } else if constexpr (Op == Int64SimdOp::Mul) {
                vmull((vector_u32&)dstLow, (vector_u32&)dstHigh, (vector_u32&)srcLow, (vector_u32&)scalarLow, mask);
                vmula(dstHigh, srcLow, scalarHigh, mask, MODE_ZEROING);
                vmula(dstHigh, srcHigh, scalarLow, mask, MODE_ZEROING);
            } else if constexpr (Op == Int64SimdOp::Shl) {
                vbr(scalarLow, static_cast<int32_t>(scalarBits));
                Int64ShiftSimdRegs<false, T>(dstLow, dstHigh, srcLow, srcHigh, scalarLow, mask);
            } else if constexpr (Op == Int64SimdOp::Shr) {
                vbr(scalarLow, static_cast<int32_t>(scalarBits));
                Int64ShiftSimdRegs<true, T>(dstLow, dstHigh, srcLow, srcHigh, scalarLow, mask);
            } else {
                Int64MinMaxSimd<Op, T>(dstLow, dstHigh, srcLow, srcHigh, scalarLow, scalarHigh, mask);
            }
            vsts(dstLow, dstHigh, (__ubuf__ int32_t*)dst + row * DstCols * 2, 0, INTLV_B32, mask);
        }
    }
}

template <typename T, unsigned DstCols, unsigned MaskRowBytes, unsigned Src0Cols, unsigned Src1Cols>
PTO_INTERNAL void Int64SelectSimd(
    __ubuf__ T* dst, __ubuf__ uint8_t* packedMask, __ubuf__ T* src0, __ubuf__ T* src1, unsigned validRows,
    unsigned validCols)
{
    __VEC_SCOPE__
    {
        vector_s32 dstLow, dstHigh, src0Low, src0High, src1Low, src1High;
        MaskReg validMask = plt_b32(validCols, POST_UPDATE);
        for (uint16_t row = 0; row < (uint16_t)validRows; ++row) {
            MaskReg packed, selectMask;
            plds(packed, (__ubuf__ uint32_t*)packedMask + row * (MaskRowBytes / 4), 0, US);
            punpack(selectMask, packed, LOWER);
            vlds(src0Low, src0High, (__ubuf__ int32_t*)src0, row * Src0Cols * 2, DINTLV_B32);
            vlds(src1Low, src1High, (__ubuf__ int32_t*)src1, row * Src1Cols * 2, DINTLV_B32);
            vsel(dstLow, src0Low, src1Low, selectMask);
            vsel(dstHigh, src0High, src1High, selectMask);
            vsts(dstLow, dstHigh, (__ubuf__ int32_t*)dst, row * DstCols * 2, INTLV_B32, validMask);
        }
    }
}

template <typename T, unsigned DstCols, unsigned MaskRowBytes, unsigned SrcCols>
PTO_INTERNAL void Int64SelectScalarSimd(
    __ubuf__ T* dst, __ubuf__ uint8_t* packedMask, __ubuf__ T* src, T scalar, unsigned validRows, unsigned validCols)
{
    __VEC_SCOPE__
    {
        vector_s32 dstLow, dstHigh, srcLow, srcHigh, scalarLow, scalarHigh;
        uint64_t scalarBits = static_cast<uint64_t>(scalar);
        vbr(scalarLow, static_cast<int32_t>(scalarBits));
        vbr(scalarHigh, static_cast<int32_t>(scalarBits >> 32));
        uint32_t maskCount = validCols;
        MaskReg validMask = plt_b32(maskCount, POST_UPDATE);
        uint16_t rowCount = validRows;
        for (uint16_t row = 0; row < rowCount; ++row) {
            MaskReg packed, selectMask;
            plds(packed, (__ubuf__ uint32_t*)packedMask + row * (MaskRowBytes / 4), 0, US);
            punpack(selectMask, packed, LOWER);
            vlds(srcLow, srcHigh, (__ubuf__ int32_t*)src, row * SrcCols * 2, DINTLV_B32);
            vsel(dstLow, srcLow, scalarLow, selectMask);
            vsel(dstHigh, srcHigh, scalarHigh, selectMask);
            vsts(dstLow, dstHigh, (__ubuf__ int32_t*)dst, row * DstCols * 2, INTLV_B32, validMask);
        }
    }
}

template <typename T, unsigned DstCols>
PTO_INTERNAL void Int64FillSimd(__ubuf__ T* dst, T scalar, unsigned validRows, unsigned validCols)
{
    __VEC_SCOPE__
    {
        vector_s32 lowReg, highReg;
        uint64_t bits = static_cast<uint64_t>(scalar);
        vbr(lowReg, static_cast<int32_t>(bits));
        vbr(highReg, static_cast<int32_t>(bits >> 32));
        uint32_t count = validCols;
        MaskReg mask = plt_b32(count, POST_UPDATE);
        uint16_t rows = validRows;
        for (uint16_t row = 0; row < rows; ++row) {
            vsts(lowReg, highReg, (__ubuf__ int32_t*)dst + row * DstCols * 2, 0, INTLV_B32, mask);
        }
    }
}

template <typename T, unsigned DstCols>
PTO_INTERNAL void Int64TriSimd(__ubuf__ T* dst, unsigned validRows, unsigned validCols, int diagonal, bool upper)
{
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
            uint32_t cols = validCols;
            MaskReg storeMask = plt_b32(cols, POST_UPDATE);
            uint32_t prefix = upper ? zeros : ones;
            MaskReg prefixMask = plt_b32(prefix, POST_UPDATE);
            vector_s32 low;
            if (upper)
                vsel(low, zero, one, prefixMask);
            else
                vsel(low, one, zero, prefixMask);
            vsts(low, zero, (__ubuf__ int32_t*)dst + row * DstCols * 2, 0, INTLV_B32, storeMask);
        }
    }
}

template <typename T, typename I, unsigned DstCols, unsigned IdxCols>
PTO_INTERNAL void Int64GatherSimd(
    __ubuf__ T* dst, __ubuf__ T* src, __ubuf__ I* index, unsigned validRows, unsigned validCols)
{
    __VEC_SCOPE__
    {
        vector_u32 idx, wordIdx, highIdx, low, high;
        uint32_t count = validCols;
        MaskReg mask = plt_b32(count, POST_UPDATE);
        uint16_t rows = validRows;
        for (uint16_t row = 0; row < rows; ++row) {
            vlds(idx, (__ubuf__ uint32_t*)index + row * IdxCols, 0, NORM);
            vadd(wordIdx, idx, idx, mask, MODE_ZEROING);
            vadds(highIdx, wordIdx, 1u, mask, MODE_ZEROING);
            vgather2(low, (__ubuf__ uint32_t*)src, wordIdx, mask);
            vgather2(high, (__ubuf__ uint32_t*)src, highIdx, mask);
            vsts((vector_s32&)low, (vector_s32&)high, (__ubuf__ int32_t*)dst + row * DstCols * 2, 0, INTLV_B32, mask);
        }
    }
}

template <typename T, typename I, unsigned DstNumel, unsigned SrcCols, unsigned IdxCols>
PTO_INTERNAL void Int64ScatterSimd(
    __ubuf__ T* dst, __ubuf__ T* src, __ubuf__ I* index, unsigned validRows, unsigned validCols)
{
    __VEC_SCOPE__
    {
        vector_u32 zero, idx, wordIdx, highIdx, low, high;
        vbr(zero, 0u);
        uint32_t remaining = DstNumel * 2;
        constexpr uint16_t wordsPerRepeat = CCE_VL / sizeof(uint32_t);
        constexpr uint16_t initRepeats = (DstNumel * 2 + wordsPerRepeat - 1) / wordsPerRepeat;
        for (uint16_t repeat = 0; repeat < initRepeats; ++repeat) {
            MaskReg initMask = plt_b32(remaining, POST_UPDATE);
            vsts(zero, (__ubuf__ uint32_t*)dst + repeat * wordsPerRepeat, 0, NORM_B32, initMask);
        }
        uint32_t count = validCols;
        MaskReg mask = plt_b32(count, POST_UPDATE);
        uint16_t rows = validRows;
        for (uint16_t row = 0; row < rows; ++row) {
            vlds(low, high, (__ubuf__ uint32_t*)src + row * SrcCols * 2, 0, DINTLV_B32);
            vlds(idx, (__ubuf__ uint32_t*)index + row * IdxCols, 0, NORM);
            vadd(wordIdx, idx, idx, mask, MODE_ZEROING);
            vadds(highIdx, wordIdx, 1u, mask, MODE_ZEROING);
            vscatter(low, (__ubuf__ uint32_t*)dst, wordIdx, mask);
            vscatter(high, (__ubuf__ uint32_t*)dst, highIdx, mask);
        }
    }
}

template <MaskPattern Pattern, ScatterAxis Axis, typename T, unsigned DstNumel, unsigned DstCols, unsigned SrcCols>
PTO_INTERNAL void Int64ScatterPatternSimd(__ubuf__ T* dst, __ubuf__ T* src, unsigned validRows, unsigned validCols)
{
    constexpr unsigned times = GetTimesByMask<Pattern>();
    constexpr unsigned offset = Int64MaskPatternOffset<Pattern>();
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
            MaskReg m = plt_b32(rem, POST_UPDATE);
            vsts(z, (__ubuf__ int32_t*)dst + r * vl, 0, NORM_B32, m);
        }
        uint32_t count = validCols;
        MaskReg m = plt_b32(count, POST_UPDATE);
        vci((vector_s32&)lane, 0, INC_ORDER);
        vmuls(elemIndex, lane, static_cast<uint32_t>(times), m, MODE_ZEROING);
        vadds(elemIndex, elemIndex, static_cast<uint32_t>(offset), m, MODE_ZEROING);
        vadd(lowIndex, elemIndex, elemIndex, m, MODE_ZEROING);
        vadds(highIndex, lowIndex, 1u, m, MODE_ZEROING);
        uint16_t rows = validRows;
        for (uint16_t i = 0; i < rows; ++i) {
            vlds(l0, h0, (__ubuf__ int32_t*)src + i * SrcCols * 2, 0, DINTLV_B32);
            if constexpr (Axis == ScatterAxis::SCATTER_COL) {
                vsts(l0, h0, (__ubuf__ int32_t*)dst + (i * times + offset) * DstCols * 2, 0, INTLV_B32, m);
            } else {
                __ubuf__ uint32_t* rowDst = (__ubuf__ uint32_t*)dst + i * DstCols * 2;
                vscatter((vector_u32&)l0, rowDst, lowIndex, m);
                vscatter((vector_u32&)h0, rowDst, highIndex, m);
            }
        }
    }
}

template <typename T, unsigned DstCols, unsigned SrcRowStride>
PTO_INTERNAL void Int64RowExpandSimd(__ubuf__ T* dst, __ubuf__ T* src, unsigned validRows, unsigned validCols)
{
    __VEC_SCOPE__
    {
        vector_s32 lowReg, highReg, srcLow, srcHigh;
        uint32_t count = validCols;
        MaskReg mask = plt_b32(count, POST_UPDATE);
        MaskReg allMask = pset_b32(PAT_ALL);
        uint16_t rows = validRows;
        for (uint16_t row = 0; row < rows; ++row) {
            vlds(srcLow, srcHigh, (__ubuf__ int32_t*)src + row * SrcRowStride * 2, 0, DINTLV_B32);
            vdup(lowReg, srcLow, allMask, POS_LOWEST, MODE_ZEROING);
            vdup(highReg, srcHigh, allMask, POS_LOWEST, MODE_ZEROING);
            vsts(lowReg, highReg, (__ubuf__ int32_t*)dst + row * DstCols * 2, 0, INTLV_B32, mask);
        }
    }
}

template <typename T, unsigned DstCols>
PTO_INTERNAL void Int64ColExpandSimd(__ubuf__ T* dst, __ubuf__ T* src, unsigned validRows, unsigned validCols)
{
    __VEC_SCOPE__
    {
        vector_s32 lowReg, highReg;
        vlds(lowReg, highReg, (__ubuf__ int32_t*)src, 0, DINTLV_B32);
        uint32_t count = validCols;
        MaskReg mask = plt_b32(count, POST_UPDATE);
        uint16_t rows = validRows;
        for (uint16_t row = 0; row < rows; ++row) {
            vsts(lowReg, highReg, (__ubuf__ int32_t*)dst + row * DstCols * 2, 0, INTLV_B32, mask);
        }
    }
}
} // namespace pto

#endif
