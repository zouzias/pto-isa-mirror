/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TDIV754_HPP
#define TDIV754_HPP

#include "kernel_reg_compute_utils.h"\


using namespace AscendC::Reg;

namespace AscendC {
namespace Internal {
__aicore__ inline constexpr DivSpecificMode GetDivSpecificMode(MaskMergeMode mrgMode)
{
    return {
        .mrgMode = mrgMode,
        .precisionMode = false,
        //.algo = DivAlgo::INTRINSIC
    };
}

__aicore__ inline constexpr DivSpecificMode GetDivSpecificMode(const DivSpecificMode* sprMode)
{
    return {
        .mrgMode = sprMode->mrgMode,
        .precisionMode = sprMode->precisionMode,
        //.algo = sprMode->algo
    };
}

} // namespace Internal

template <typename T = DefaultType, auto mode = MaskMergeMode::ZEROING, typename U>
PTO_INTERNAL inline void DivPrecisionImpl(U& dstReg, U& srcReg0, U& srcReg1, MaskReg& mask)
{
    constexpr DivSpecificMode sprMode = Internal::GetDivSpecificMode(mode);
    constexpr auto modeValue = GetMaskMergeMode<sprMode.mrgMode>();
    constexpr uint32_t infNanBound = 0xff800000u;
    constexpr uint32_t signBitNum = 0x80000000u;
    RegTensor<T> regNegZero;
    RegTensor<T> tmpDst;
    RegTensor<T> r, z, y;
    RegTensor<uint32_t> infNan;

    MaskReg cmpMaskReg;
    MaskReg infNanCmp;
    MaskReg zeroCmp;
    MaskReg preg;
    preg = pset_b8(PAT_ALL);
    vdup((vector_u32&)regNegZero, signBitNum, preg, modeValue);
    vdiv(z, srcReg0, srcReg1, mask, modeValue);
    vor(infNan, (vector_u32&)z, (vector_u32&)regNegZero, mask, modeValue);
    tmpDst = z;
    vcmps_eq(zeroCmp, z, 0.0f, mask);
    vcmps_ge(infNanCmp, infNan, infNanBound, mask);
    por(infNanCmp, infNanCmp, zeroCmp, mask);

    vmuls(y, srcReg1, -1.0f, mask, modeValue);
    r = srcReg0;
    vmula(r, z, y, mask, modeValue);
    RegTensor<T> rPre, rNext, zPre, zNext;

    vadds((vector_s32&)zPre, (vector_s32&)z, -1, mask, modeValue);
    vadds((vector_s32&)zNext, (vector_s32&)z, 1, mask, modeValue);

    rPre = srcReg0;
    rNext = srcReg0;

    vmula(rPre, zPre, y, mask, modeValue);
    vmula(rNext, zNext, y, mask, modeValue);

    vabs(r, r, mask, modeValue);
    vabs(rPre, rPre, mask, modeValue);
    vabs(rNext, rNext, mask, modeValue);
    vcmp_lt(cmpMaskReg, r, rPre, mask);
    vsel(r, r, rPre, cmpMaskReg);
    vsel(z, z, zPre, cmpMaskReg);

    vcmp_lt(cmpMaskReg, rNext, r, mask);
    vsel(z, zNext, z, cmpMaskReg);
    vsel(dstReg, tmpDst, z, infNanCmp);
}

template <typename T = DefaultType, auto mode = MaskMergeMode::ZEROING, typename U, bool is0ULP>
PTO_INTERNAL inline void DivIEEE754FloatImpl(RegTensor<float>& dst, RegTensor<float>& src0,
                                                RegTensor<float>& src1, MaskReg& mask)
{
    constexpr uint32_t exponentExtractor = 0x807FFFFF;
    constexpr uint32_t signExtractor = 0x80000000;
    constexpr uint32_t exponentNormalizer = 0x3F800000;

    NotNumUnion subnormalThreshold;
    subnormalThreshold.i = 0x007FFFFF;

    NotNumUnion nan;
    nan.i = F32_NAN;
    NotNumUnion min_denormal;
    min_denormal.i = 0x1;

    NotNumUnion normalizeScaleEnlarge;
    normalizeScaleEnlarge.i = 0x4B000000; // 2^23
    NotNumUnion normalizeScaleReduce;
    normalizeScaleReduce.i = 0x34000000; // 2^-23

    RegTensor<float> maxSubnormal;
    RegTensor<uint32_t> tmp0;
    RegTensor<int32_t> tmp1;
    RegTensor<uint32_t> tmp2;

    RegTensor<float> src0Abs;
    RegTensor<float> src0Subnormal;
    RegTensor<float> src0Norm;
    RegTensor<float> src0All;
    RegTensor<float> src0AbsNorm;

    RegTensor<float> src1Abs;
    RegTensor<float> src1Subnormal;
    RegTensor<float> src1Norm;
    RegTensor<float> src1All;
    RegTensor<float> src1AbsNorm;

    MaskReg mask0;
    MaskReg maskSrc0Normal;
    MaskReg maskSrc0Subnormal;
    MaskReg maskSrc1Normal;
    MaskReg maskSrc1Subnormal;
    MaskReg maskTmp;
    MaskReg maskNan; // divisor or dividend 0
    MaskReg maskInf; // divisor or dividend inf
    MaskReg maskSrc0Zero; // dividend 0
    MaskReg maskSrc1Zero; // divisor 0
    MaskReg maskValid;
    MaskReg maskNorm;

    RegTensor<uint32_t> src0Exponent;
    RegTensor<uint32_t> src1Exponent;

    RegTensor<float> z1;
    RegTensor<float> z2;
    RegTensor<int32_t> scale;
    RegTensor<uint32_t> dstExponent;
    RegTensor<uint32_t> dstSign;

    // subnormal threshold
    Duplicate(maxSubnormal, subnormalThreshold.f, mask);

    // ===========================================================
    // acquiring valid numbers (no inf, no 0)
    // ===========================================================
    
    constexpr auto modeValue = GetMaskMergeMode<sprMode.mrgMode>();
    vabs(src0Abs, src0, mask, modeValue);
    vabs(src1Abs, src1, mask, modeValue);

    // get positions of inf values
    Duplicate(tmp0, F32_INF, mask);
    Compare<uint32_t, CMPMODE::EQ>(maskInf, (RegTensor<uint32_t>&)src0Abs, tmp0, mask);
    Compare<uint32_t, CMPMODE::EQ>(maskTmp, (RegTensor<uint32_t>&)src1Abs, tmp0, mask);
    MaskOr(maskValid, maskInf, maskTmp, mask);
    // get positions of 0 divisor or dividend
    Duplicate(tmp0, 0, mask);
    Compare<uint32_t, CMPMODE::EQ>(maskSrc0Zero, (RegTensor<uint32_t>&)src0Abs, tmp0, mask);
    // merge for positions of invalid numbers
    MaskOr(maskValid, maskValid, maskSrc0Zero, mask);
    Compare<uint32_t, CMPMODE::EQ>(maskSrc1Zero, (RegTensor<uint32_t>&)src1Abs, tmp0, mask);
    // negating for positions of valid numbers
    MaskOr(maskValid, maskValid, maskSrc1Zero, mask);
    MaskNot(maskValid, maskValid, mask);

    // normalize subnormal elements of src0
    // get positions of subnormal numbers in dividend
    Compare<float, CMPMODE::LT>(maskSrc0Subnormal, src0Abs, maxSubnormal, mask);
    // negating for normal positions
    MaskNot(maskSrc0Normal, maskSrc0Subnormal, mask);
    // normalizatoin
    Muls(src0Subnormal, src0, normalizeScaleEnlarge.f, maskSrc0Subnormal);

    // normalize subnormal elements of src1
    Compare<float, CMPMODE::LT>(maskSrc1Subnormal, src1Abs, maxSubnormal, mask);
    MaskNot(maskSrc1Normal, maskSrc1Subnormal, mask);
    Muls(src1Subnormal, src1, normalizeScaleEnlarge.f, maskSrc1Subnormal);

    // merge the normalized subnormal elements with normal elements
    Select(src0All, src0, src0Subnormal, maskSrc0Normal);
    Select(src1All, src1, src1Subnormal, maskSrc1Normal);

    // standardized the exponent bits of src0 and src1
    // zero out the exponent bits 00000000
    Duplicate(tmp0, exponentExtractor, mask);
    And((RegTensor<uint32_t>&)src0Norm, (RegTensor<uint32_t>&)src0All, tmp0, maskValid);
    And((RegTensor<uint32_t>&)src1Norm, (RegTensor<uint32_t>&)src1All, tmp0, maskValid);
    // set the exponent bits to 01111111
    Duplicate(tmp0, exponentNormalizer, mask);
    Add((RegTensor<uint32_t>&)src0Norm, (RegTensor<uint32_t>&)src0Norm, tmp0, maskValid);
    Add((RegTensor<uint32_t>&)src1Norm, (RegTensor<uint32_t>&)src1Norm, tmp0, maskValid);
    Select(src0Norm, src0Norm, src0All, maskValid);
    Select(src1Norm, src1Norm, src1All, maskValid);
    Abs(src0AbsNorm, src0Norm, maskValid);
    Abs(src1AbsNorm, src1Norm, maskValid);
    Compare<float, CMPMODE::LE>(maskNorm, src0AbsNorm, src1AbsNorm, maskValid);

    if constexpr (is0ULP) {
        DivPrecisionImpl<float, mode, U>(dst, src0Norm, src1Norm, mask);
    } else {
        constexpr DivSpecificMode sprMode = Internal::GetDivSpecificMode(MaskMergeMode::ZEROING);
        constexpr auto modeValue = GetMaskMergeMode<sprMode.mrgMode>();
        vdiv(dst, src0Norm, src1Norm, mask, modeValue);
    }

    // subnormal dividend, normal divisor
    MaskAnd(mask0, maskSrc0Subnormal, maskSrc1Normal, mask);
    // normalization compensation
    Muls(z1, dst, normalizeScaleReduce.f, mask0);
    Select(dst, z1, dst, mask0);

    // normal dividend, subnormal divisor
    MaskAnd(mask0, maskSrc0Normal, maskSrc1Subnormal, mask);
    // normalization compensation
    Muls(z1, dst, normalizeScaleEnlarge.f, mask0);
    // merge the compensated result
    Select(dst, z1, dst, mask0);

    // preserve sign for error handling section below
    Duplicate(tmp0, signExtractor, mask);
    And((RegTensor<uint32_t>&)dstSign, (RegTensor<uint32_t>&)dst, tmp0, mask);

    // ===========================================================
    // exponent operation
    // ===========================================================
    // extract the exponent section 0 11..11 00..00
    Duplicate(tmp0, F32_INF, mask);
    And(src0Exponent, (RegTensor<uint32_t>&)src0All, tmp0, mask);
    And(src1Exponent, (RegTensor<uint32_t>&)src1All, tmp0, mask);
    And(dstExponent, (RegTensor<uint32_t>&)dst, tmp0, mask);

    // exponent subtraction (effectively fp number division)
    ShiftRights(src0Exponent, src0Exponent, (int16_t)23, mask);
    ShiftRights(src1Exponent, src1Exponent, (int16_t)23, mask);
    ShiftRights(dstExponent, dstExponent, (int16_t)23, mask);
    Sub(scale, (RegTensor<int32_t>&)src0Exponent, (RegTensor<int32_t>&)src1Exponent, mask);
    Adds(scale, scale, 127, mask);
    // ===========================================================
    // exception handling
    // ===========================================================
    // overflow (exponent over 255) underflow (exponent under 0) detection // FP32:1S + 8E + 23M
    Duplicate(tmp1, -23, mask);
    // True if underflow/overflow
    Compare<int32_t, CMPMODE::EQ>(mask0, scale, (RegTensor<int32_t>&)tmp1, mask);
    MaskAnd(mask0, mask0, maskValid, mask);
    Duplicate(tmp0, min_denormal.i, mask0);
    Add((RegTensor<uint32_t>&)z1, (RegTensor<uint32_t>&)dstSign, tmp0, mask0);
    Duplicate(tmp2, static_cast<uint32_t>(0), mask0);
    Add((RegTensor<uint32_t>&)z2, (RegTensor<uint32_t>&)dstSign, tmp2, mask0);
    Select(z1, z2, z1, maskNorm);
    Select(dst, z1, dst, mask0);
    MaskNot(mask0, mask0, mask);
    MaskAnd(maskValid, mask0, maskValid, mask);
    Compare<int32_t, CMPMODE::LT>(mask0, scale, (RegTensor<int32_t>&)tmp1, mask);
    // set overflown/underflown result to infinity
    MaskAnd(mask0, mask0, maskValid, mask);
    Duplicate(tmp0, 0, mask); // set to 0
    Add((RegTensor<uint32_t>&)z1, (RegTensor<uint32_t>&)dstSign, tmp0, mask0);
    Select(dst, z1, dst, mask0);
    MaskNot(mask0, mask0, mask);
    MaskAnd(maskValid, mask0, maskValid, mask);

    Duplicate(tmp0, 255, mask);
    Compare<int32_t, CMPMODE::EQ>(mask0, scale, (RegTensor<int32_t>&)tmp0, mask);
    MaskAnd(mask0, mask0, maskValid, mask);
    Reg::Duplicate(tmp1, 1, mask0);
    Reg::Sub(tmp1, scale, tmp1, mask0);
    Select(scale, tmp1, scale, mask0);
    Muls(z1, dst, 2, mask0);
    Select(dst, z1, dst, mask0);

    Compare<int32_t, CMPMODE::GT>(mask0, scale, (RegTensor<int32_t>&)tmp0, mask);
    MaskAnd(mask0, mask0, maskValid, mask);
    Duplicate(tmp0, F32_INF, mask); // set to infinity
    Add((RegTensor<uint32_t>&)z1, (RegTensor<uint32_t>&)dstSign, tmp0, mask0);
    Select(dst, z1, dst, mask0);
    MaskNot(mask0, mask0, mask);
    MaskAnd(maskValid, mask0, maskValid, mask);

    Duplicate(tmp0, 0, maskValid);
    Compare<int32_t, CMPMODE::GT>(mask0, scale, (RegTensor<int32_t>&)tmp0, maskValid);
    ShiftLefts(tmp1, scale, (int16_t)23, mask0);
    Mul(z1, dst, (RegTensor<float>&)tmp1, mask0);
    Select(dst, z1, dst, mask0);

    MaskNot(mask0, mask0, maskValid);
    Duplicate(tmp0, 4194304, mask0); // set 0x0040 0000
    Abs(scale, scale, mask0);
    ShiftRight(scale, (RegTensor<int32_t>&)tmp0, scale, mask0);
    Mul(z1, dst, (RegTensor<float>&)scale, mask0);
    Select(dst, z1, dst, mask0);

    // get the position of nan
    Duplicate(tmp0, nan.i, mask);
    Compare<float, CMPMODE::NE>(maskNan, src0Abs, src0Abs, mask);
    Compare<float, CMPMODE::NE>(maskTmp, src1Abs, src1Abs, mask);
    MaskOr(maskNan, maskNan, maskTmp, mask);
    // set output with nan input to nan
    Select(dst, (RegTensor<float>&)tmp0, dst, maskNan);
}



template <typename T = DefaultType, auto mode = MaskMergeMode::ZEROING, typename U>
PTO_INTERNAL inline void DivIEEE754HalfImpl(RegTensor<half>& dst, RegTensor<half>& src0,
                                               RegTensor<half>& src1, MaskReg& mask)
{
    constexpr uint16_t exponentExtractor = 0x83FF;
    constexpr uint16_t signExtractor = 0x8000;
    constexpr uint16_t exponentNormalizer = 0x3C00;
    constexpr uint16_t F16_INF = 0x7C00;

    HalfUnion subnormalThreshold;
    subnormalThreshold.i = 0x03FF;

    HalfUnion nan;
    nan.i = 0x7E00;
    HalfUnion min_denormal;
    min_denormal.i = 0x1;

    HalfUnion normalizeScaleEnlarge;
    normalizeScaleEnlarge.i = 0x6400; // 2^10
    HalfUnion normalizeScaleReduce;
    normalizeScaleReduce.i = 0x1400; // 2^-10

    RegTensor<half> maxSubnormal;
    RegTensor<uint16_t> tmp0;
    RegTensor<int16_t> tmp1;
    RegTensor<uint16_t> tmp2;

    RegTensor<half> src0Abs;
    RegTensor<half> src0Subnormal;
    RegTensor<half> src0Norm;
    RegTensor<half> src0All;
    RegTensor<half> src0AbsNorm;

    RegTensor<half> src1Abs;
    RegTensor<half> src1Subnormal;
    RegTensor<half> src1Norm;
    RegTensor<half> src1All;
    RegTensor<half> src1AbsNorm;

    MaskReg mask0;
    MaskReg maskSrc0Normal;
    MaskReg maskSrc0Subnormal;
    MaskReg maskSrc1Normal;
    MaskReg maskSrc1Subnormal;
    MaskReg maskTmp;
    MaskReg maskNan; // divisor or dividend 0
    MaskReg maskInf; // divisor or dividend inf
    MaskReg maskSrc0Zero; // dividend 0
    MaskReg maskSrc1Zero; // divisor 0
    MaskReg maskValid;
    MaskReg maskNorm;

    RegTensor<uint16_t> src0Exponent;
    RegTensor<uint16_t> src1Exponent;

    RegTensor<half> z1;
    RegTensor<half> z2;
    RegTensor<int16_t> scale;
    RegTensor<uint16_t> dstExponent;
    RegTensor<uint16_t> dstSign;

    // subnormal threshold
    Duplicate(maxSubnormal, subnormalThreshold.f, mask);

    // ===========================================================
    // acquiring valid numbers (no inf, no 0)
    // ===========================================================
    Abs(src0Abs, src0, mask);
    Abs(src1Abs, src1, mask);

    // get positions of inf values
    Duplicate(tmp0, F16_INF, mask);
    Compare<uint16_t, CMPMODE::EQ>(maskInf, (RegTensor<uint16_t>&)src0Abs, tmp0, mask);
    Compare<uint16_t, CMPMODE::EQ>(maskTmp, (RegTensor<uint16_t>&)src1Abs, tmp0, mask);
    MaskOr(maskValid, maskInf, maskTmp, mask);
    // get positions of 0 divisor or dividend
    Duplicate(tmp0, 0, mask);
    Compare<uint16_t, CMPMODE::EQ>(maskSrc0Zero, (RegTensor<uint16_t>&)src0Abs, tmp0, mask);
    // merge for positions of invalid numbers
    MaskOr(maskValid, maskValid, maskSrc0Zero, mask);
    Compare<uint16_t, CMPMODE::EQ>(maskSrc1Zero, (RegTensor<uint16_t>&)src1Abs, tmp0, mask);
    // negating for positions of valid numbers
    MaskOr(maskValid, maskValid, maskSrc1Zero, mask);
    MaskNot(maskValid, maskValid, mask);

    // normalize subnormal elements of src0
    // get positions of subnormal numbers in dividend
    Compare<half, CMPMODE::LT>(maskSrc0Subnormal, src0Abs, maxSubnormal, mask);
    // negating for normal positions
    MaskNot(maskSrc0Normal, maskSrc0Subnormal, mask);
    // normalizatoin
    Muls(src0Subnormal, src0, normalizeScaleEnlarge.f, maskSrc0Subnormal);

    // normalize subnormal elements of src1
    Compare<half, CMPMODE::LT>(maskSrc1Subnormal, src1Abs, maxSubnormal, mask);
    MaskNot(maskSrc1Normal, maskSrc1Subnormal, mask);
    Muls(src1Subnormal, src1, normalizeScaleEnlarge.f, maskSrc1Subnormal);

    // merge the normalized subnormal elements with normal elements
    Select(src0All, src0, src0Subnormal, maskSrc0Normal);
    Select(src1All, src1, src1Subnormal, maskSrc1Normal);

    // standardized the exponent bits of src0 and src1
    // zero out the exponent bits 00000000
    Duplicate(tmp0, exponentExtractor, mask);
    And((RegTensor<uint16_t>&)src0Norm, (RegTensor<uint16_t>&)src0All, tmp0, maskValid);
    And((RegTensor<uint16_t>&)src1Norm, (RegTensor<uint16_t>&)src1All, tmp0, maskValid);
    // set the exponent bits to 01111111
    Duplicate(tmp0, exponentNormalizer, mask);
    Add((RegTensor<uint16_t>&)src0Norm, (RegTensor<uint16_t>&)src0Norm, tmp0, maskValid);
    Add((RegTensor<uint16_t>&)src1Norm, (RegTensor<uint16_t>&)src1Norm, tmp0, maskValid);
    Select(src0Norm, src0Norm, src0All, maskValid);
    Select(src1Norm, src1Norm, src1All, maskValid);
    Abs(src0AbsNorm, src0Norm, maskValid);
    Abs(src1AbsNorm, src1Norm, maskValid);
    Compare<half, CMPMODE::LE>(maskNorm, src0AbsNorm, src1AbsNorm, maskValid);

    constexpr DivSpecificMode sprMode = Internal::GetDivSpecificMode(MaskMergeMode::ZEROING);
    constexpr auto modeValue = GetMaskMergeMode<sprMode.mrgMode>();
    vdiv(dst, src0Norm, src1Norm, mask, modeValue);

    // subnormal dividend, normal divisor
    MaskAnd(mask0, maskSrc0Subnormal, maskSrc1Normal, mask);
    // normalization compensation
    Muls(z1, dst, normalizeScaleReduce.f, mask0);
    Select(dst, z1, dst, mask0);

    // normal dividend, subnormal divisor
    MaskAnd(mask0, maskSrc0Normal, maskSrc1Subnormal, mask);
    // normalization compensation
    Muls(z1, dst, normalizeScaleEnlarge.f, mask0);
    // merge the compensated result
    Select(dst, z1, dst, mask0);

    // preserve sign for error handling section below
    Duplicate(tmp0, signExtractor, mask);
    And((RegTensor<uint16_t>&)dstSign, (RegTensor<uint16_t>&)dst, tmp0, mask);

    // ===========================================================
    // exponent operation
    // ===========================================================
    // extract the exponent section 0 11..11 00..00
    Duplicate(tmp0, F16_INF, mask);
    And(src0Exponent, (RegTensor<uint16_t>&)src0All, tmp0, mask);
    And(src1Exponent, (RegTensor<uint16_t>&)src1All, tmp0, mask);
    And(dstExponent, (RegTensor<uint16_t>&)dst, tmp0, mask);

    // exponent subtraction (effectively fp number division)
    ShiftRights(src0Exponent, src0Exponent, (int16_t)10, mask);
    ShiftRights(src1Exponent, src1Exponent, (int16_t)10, mask);
    ShiftRights(dstExponent, dstExponent, (int16_t)10, mask);
    Sub(scale, (RegTensor<int16_t>&)src0Exponent, (RegTensor<int16_t>&)src1Exponent, mask);
    Adds(scale, scale, 15, mask);
    // ===========================================================
    // exception handling
    // ===========================================================
    // overflow (exponent over 31) underflow (exponent under -9) detection // FP16:1S + 5E + 9M
    Duplicate(tmp1, -9, mask);
    Compare<int16_t, CMPMODE::EQ>(mask0, scale, (RegTensor<int16_t>&)tmp1, mask);
    MaskAnd(mask0, mask0, maskValid, mask);
    Duplicate(tmp0, min_denormal.i, mask0);
    Add((RegTensor<uint16_t>&)z1, (RegTensor<uint16_t>&)dstSign, tmp0, mask0);
    Duplicate(tmp2, static_cast<uint16_t>(0), mask0);
    Add((RegTensor<uint16_t>&)z2, (RegTensor<uint16_t>&)dstSign, tmp2, mask0);
    Select(z1, z2, z1, maskNorm);
    Select(dst, z1, dst, mask0);
    MaskNot(mask0, mask0, mask);
    MaskAnd(maskValid, mask0, maskValid, mask);
    // True if underflow/overflow
    Compare<int16_t, CMPMODE::LT>(mask0, scale, (RegTensor<int16_t>&)tmp1, mask);
    // set overflown/underflown result to infinity
    MaskAnd(mask0, mask0, maskValid, mask);
    Duplicate(tmp0, 0, mask); // set to 0
    Add((RegTensor<uint16_t>&)z1, (RegTensor<uint16_t>&)dstSign, tmp0, mask0);
    Select(dst, z1, dst, mask0);
    MaskNot(mask0, mask0, mask);
    MaskAnd(maskValid, mask0, maskValid, mask);

    Duplicate(tmp0, 31, mask);
    Compare<int16_t, CMPMODE::EQ>(mask0, scale, (RegTensor<int16_t>&)tmp0, mask);
    MaskAnd(mask0, mask0, maskValid, mask);
    Reg::Duplicate(tmp1, 1, mask0);
    Reg::Sub(tmp1, scale, tmp1, mask0);
    Select(scale, tmp1, scale, mask0);
    Muls(z1, dst, 2, mask0);
    Select(dst, z1, dst, mask0);

    Compare<int16_t, CMPMODE::GT>(mask0, scale, (RegTensor<int16_t>&)tmp0, mask);
    MaskAnd(mask0, mask0, maskValid, mask);
    Duplicate(tmp0, F16_INF, mask); // set to infinity
    Add((RegTensor<uint16_t>&)z1, (RegTensor<uint16_t>&)dstSign, tmp0, mask0);
    Select(dst, z1, dst, mask0);
    MaskNot(mask0, mask0, mask);
    MaskAnd(maskValid, mask0, maskValid, mask);

    Duplicate(tmp0, 0, maskValid);
    Compare<int16_t, CMPMODE::GT>(mask0, scale, (RegTensor<int16_t>&)tmp0, maskValid);
    ShiftLefts(tmp1, scale, (int16_t)10, mask0);
    Mul(z1, dst, (RegTensor<half>&)tmp1, mask0);
    Select(dst, z1, dst, mask0);

    MaskNot(mask0, mask0, maskValid);
    Duplicate(tmp0, 512, mask0); // set 0x0200
    Abs(scale, scale, mask0);
    ShiftRight(scale, (RegTensor<int16_t>&)tmp0, scale, mask0);
    Mul(z1, dst, (RegTensor<half>&)scale, mask0);
    Select(dst, z1, dst, mask0);

    // get the position of nan
    Duplicate(tmp0, nan.i, mask);
    Compare<half, CMPMODE::NE>(maskNan, src0Abs, src0Abs, mask);
    Compare<half, CMPMODE::NE>(maskTmp, src1Abs, src1Abs, mask);
    MaskOr(maskNan, maskNan, maskTmp, mask);
    // set output with nan input to nan
    Select(dst, (RegTensor<half>&)tmp0, dst, maskNan);
}
}
#endif // TINSERT_CUSTOM_HPP
