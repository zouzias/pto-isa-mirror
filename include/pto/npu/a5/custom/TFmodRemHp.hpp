/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TFMODHP_HPP
#define TFMODHP_HPP

#include <pto/common/constants.hpp>
#include <pto/common/utils.hpp>
#include <pto/npu/a5/common.hpp>
#include <pto/npu/a5/utils.hpp>

namespace pto {
constexpr uint32_t FMOD_ITERATION_NUM_MAX = 11;
constexpr FloatUnion inf(0x7f800000);
constexpr FloatUnion negInf(0xff800000);
constexpr FloatUnion nan(0x7fc00000);

constexpr FloatUnion scaleList1[FMOD_ITERATION_NUM_MAX] = {
    FloatUnion(0x4b800000), FloatUnion(0x4b800000), FloatUnion(0x57800000), FloatUnion(0x63800000),
    FloatUnion(0x6f800000), FloatUnion(0x7b800000), FloatUnion(0x7b800000), FloatUnion(0x7b800000),
    FloatUnion(0x7b800000), FloatUnion(0x7b800000), FloatUnion(0x7b800000)};
constexpr FloatUnion scaleList2[FMOD_ITERATION_NUM_MAX] = {
    FloatUnion(0x3f800000), FloatUnion(0x3f800000), FloatUnion(0x3f800000), FloatUnion(0x3f800000),
    FloatUnion(0x3f800000), FloatUnion(0x3f800000), FloatUnion(0x4b800000), FloatUnion(0x57800000),
    FloatUnion(0x63800000), FloatUnion(0x6f800000), FloatUnion(0x7b800000)};

PTO_INTERNAL void GetSignBit(RegTensor<float> &dstReg, RegTensor<float> &srcReg, MaskReg &mask)
{
    constexpr int16_t signRightNum = BLOCK_BYTE_SIZE - 1;
    RegTensor<uint32_t> oneReg, tmpReg;
    uint32_t len32 = static_cast<uint16_t>(VECTOR_REG_WIDTH / sizeof(float));
    MaskReg preg_b32 = CreatePredicate<float>(len32);
    vdup(oneReg, static_cast<float>(1), mask, MODE_ZEROING);
    vshrs(tmpReg, (RegTensor<uint32_t> &)srcReg, signRightNum, mask, MODE_ZEROING);
    vand(tmpReg, tmpReg, oneReg, mask, MODE_ZEROING);
    vcvt(dstReg, (RegTensor<int32_t> &)tmpReg, preg_b32, __cce_simd::RoundRType());
}

template <int32_t iterationNum>
PTO_INTERNAL void SolveScale(RegTensor<float> &dstReg, RegTensor<float> &srcReg, const float scale1, const float scale2,
                             MaskReg &mask, bool isFmod)
{
    constexpr float maxValue = 3.4028235e38;
    constexpr float subNormal = 1.1754944e-38;

    MaskReg subNormalMask;
    RegTensor<float> bTmpReg, tmpReg, kReg, signReg;

    if constexpr (iterationNum == 1) { // iter 1 (last iteration) handles subnormal case
        vcmps_le(subNormalMask, srcReg, subNormal, mask);
        vmuls(tmpReg, srcReg, scale1, subNormalMask, MODE_ZEROING);
        vsel(bTmpReg, tmpReg, srcReg, subNormalMask);
        vmuls(tmpReg, dstReg, scale1, subNormalMask, MODE_ZEROING);
        vsel(dstReg, tmpReg, dstReg, subNormalMask);
        vdiv(kReg, dstReg, bTmpReg, mask, MODE_ZEROING);
        if (isFmod) {
            vtrc(kReg, kReg, ROUND_Z, mask);
        } else {
            vtrc(kReg, kReg, ROUND_F, mask);
        }
    } else {
        vmuls(bTmpReg, srcReg, scale1, mask, MODE_ZEROING);
        if constexpr (iterationNum > 5) { // last 5 iterations do not need extra scaling
            vmuls(bTmpReg, bTmpReg, scale2, mask, MODE_ZEROING);
        }
        vdiv(kReg, dstReg, bTmpReg, mask, MODE_ZEROING);
        if (isFmod) {
            vtrc(kReg, kReg, ROUND_Z, mask);
        } else {
            vtrc(kReg, kReg, ROUND_F, mask);
        }
    }

    // not necessary to check for inf in the final iteration
    if constexpr (iterationNum != 1) {
        vmins(bTmpReg, bTmpReg, maxValue, mask, MODE_ZEROING);
    }
    vneg(kReg, kReg, mask, MODE_ZEROING);
    // res = -k * bTmp + y
    vmula(dstReg, kReg, bTmpReg, mask, MODE_ZEROING);

    if constexpr (iterationNum == 1) { // iter 1 handles subnormal case
        // r = r + np.float32(np.signbit(r)) * bTmp
        GetSignBit(signReg, dstReg, mask);
        vmul(signReg, signReg, bTmpReg, mask, MODE_ZEROING);
        vadd(dstReg, dstReg, signReg, mask, MODE_ZEROING);
    }
}

// recurse from itermax to 1
template <int32_t iterationNum>
PTO_INTERNAL void SolveScaleIter(RegTensor<float> &dstReg, RegTensor<float> &srcReg, MaskReg &mask, bool isFmod)
{
    SolveScale<iterationNum>(dstReg, srcReg, scaleList1[iterationNum - 1].f, scaleList2[iterationNum - 1].f, mask,
                             isFmod);

    if constexpr (iterationNum > 1) {
        SolveScaleIter<iterationNum - 1>(dstReg, srcReg, mask, isFmod);
    }
}

template <int32_t iterationNum, typename TSrc>
PTO_INTERNAL void SolveScale(__ubuf__ float *dst, TSrc src, const uint16_t unitRepTimes, const float scale1,
                             const float scale2, MaskReg &mask, uint32_t elementsPerRepeat, bool isFmod)
{
    RegTensor<float> dstReg, srcReg, srcOriginReg;
    for (uint16_t i = 0; i < unitRepTimes; i++) {
        if constexpr (std::is_same_v<__ubuf__ float *, TSrc>) {
            vlds(srcOriginReg, src, i * elementsPerRepeat, NORM);
        } else {
            vdup(srcOriginReg, src, mask, MODE_ZEROING);
        }

        vlds(dstReg, dst, i * elementsPerRepeat, NORM);
        vabs(srcReg, srcOriginReg, mask, MODE_ZEROING);
        SolveScale<iterationNum>(dstReg, srcReg, scale1, scale2, mask, isFmod);
        constexpr auto distValue =
            std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<float, DistVST::DIST_NORM>())>();
        vsts(dstReg, dst, i * elementsPerRepeat, distValue, mask);
    }
}

template <int32_t iterationNum, typename TSrc0, typename TSrc1>
PTO_INTERNAL void SolveScaleInit(__ubuf__ float *dst, TSrc0 src0, TSrc1 src1, const uint16_t unitRepTimes,
                                 const float scale1, const float scale2, MaskReg &mask, uint32_t elementsPerRepeat,
                                 bool isFmod)
{
    RegTensor<float> dstReg, srcReg, src0OriginReg, src1OriginReg;
    for (uint16_t i = 0; i < unitRepTimes; i++) {
        if constexpr (std::is_same_v<__ubuf__ float *, TSrc0>) {
            vlds(src0OriginReg, src0, i * elementsPerRepeat, NORM);
        } else {
            vdup(src0OriginReg, src0, mask, MODE_ZEROING);
        }
        if constexpr (std::is_same_v<__ubuf__ float *, TSrc1>) {
            vlds(src1OriginReg, src1, i * elementsPerRepeat, NORM);
        } else {
            vdup(src1OriginReg, src1, mask, MODE_ZEROING);
        }
        vabs(dstReg, src0OriginReg, mask, MODE_ZEROING);
        vabs(srcReg, src1OriginReg, mask, MODE_ZEROING);
        SolveScale<iterationNum>(dstReg, srcReg, scale1, scale2, mask, isFmod);
        constexpr auto distValue =
            std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<float, DistVST::DIST_NORM>())>();
        vsts(dstReg, dst, i * elementsPerRepeat, distValue, mask);
    }
}

template <int32_t iterationNum, int32_t totalIterationNum, typename TSrc0, typename TSrc1>
PTO_INTERNAL void SolveScaleIterImpl(__ubuf__ float *dst, TSrc0 src0, TSrc1 src1, const uint16_t unitRepTimes,
                                     MaskReg &mask, uint32_t elementsPerRepeat, bool isFmod)
{
    if (iterationNum == totalIterationNum) { // first iteration, initialization
        SolveScaleInit<iterationNum>(dst, src0, src1, unitRepTimes, scaleList1[iterationNum - 1].f,
                                     scaleList2[iterationNum - 1].f, mask, elementsPerRepeat, isFmod);
    } else {
        SolveScale<iterationNum>(dst, src1, unitRepTimes, scaleList1[iterationNum - 1].f,
                                 scaleList2[iterationNum - 1].f, mask, elementsPerRepeat, isFmod);
    }

    if constexpr (iterationNum > 1) {
        mem_bar(VST_VLD);
        SolveScaleIterImpl<iterationNum - 1, totalIterationNum>(dst, src0, src1, unitRepTimes, mask, elementsPerRepeat,
                                                                isFmod);
    }
}

template <int32_t iterationNum, typename TSrc0, typename TSrc1>
PTO_INTERNAL void SolveScaleIter(__ubuf__ float *dst, TSrc0 src0, TSrc1 src1, const uint16_t unitRepTimes,
                                 MaskReg &mask, uint32_t elementsPerRepeat, bool isFmod)
{
    SolveScaleIterImpl<iterationNum, iterationNum>(dst, src0, src1, unitRepTimes, mask, elementsPerRepeat, isFmod);
}

PTO_INTERNAL void SolveExceptionScenarios(RegTensor<float> &dstReg, RegTensor<float> &src0Reg,
                                          RegTensor<float> &src1Reg, RegTensor<float> &nanReg, MaskReg &mask)
{
    MaskReg src0Is0CmpReg, src0IsNeg0CmpReg, src0InfCmpReg, src0NegInfCmpReg;
    MaskReg src1Not0CmpReg, src1NotNeg0CmpReg, src1NotNanCmpReg, src1InfCmpReg, src1NegInfCmpReg;
    MaskReg srcBothInfCmpReg;

    vcmps_eq(src1InfCmpReg, src1Reg, inf.f, mask);
    vsel(dstReg, src0Reg, dstReg, src1InfCmpReg);
    vcmps_eq(src1NegInfCmpReg, src1Reg, negInf.f, mask);
    vsel(dstReg, src0Reg, dstReg, src1NegInfCmpReg);
    vcmps_eq(src0InfCmpReg, src0Reg, inf.f, mask);
    vcmps_eq(src0NegInfCmpReg, src0Reg, negInf.f, mask);
    por(src0InfCmpReg, src0InfCmpReg, src0NegInfCmpReg, mask);
    por(src1InfCmpReg, src1InfCmpReg, src1NegInfCmpReg, mask);
    pand(srcBothInfCmpReg, src0InfCmpReg, src1InfCmpReg, mask);
    vsel(dstReg, nanReg, dstReg, srcBothInfCmpReg);
    vcmps_eq(src0Is0CmpReg, src0Reg, static_cast<float>(0), mask);
    vcmps_eq(src0IsNeg0CmpReg, src0Reg, static_cast<float>(-0), mask);
    por(src0Is0CmpReg, src0Is0CmpReg, src0IsNeg0CmpReg, mask);

    vcmps_ne(src1Not0CmpReg, src1Reg, static_cast<float>(0), mask);
    vcmps_ne(src1NotNeg0CmpReg, src1Reg, static_cast<float>(-0), mask);
    vcmp_ne(src1NotNanCmpReg, src1Reg, src1Reg, mask);
    pnot(src1NotNanCmpReg, src1NotNanCmpReg, mask);

    por(src1Not0CmpReg, src1Not0CmpReg, src1NotNeg0CmpReg, mask);
    pand(src1Not0CmpReg, src1Not0CmpReg, src1NotNanCmpReg, mask);

    pand(src0Is0CmpReg, src0Is0CmpReg, src1Not0CmpReg, mask);
    vsel(dstReg, src0Reg, dstReg, src0Is0CmpReg);
}

template <bool isTail, int32_t iterationNum, typename TSrc0, typename TSrc1>
PTO_INTERNAL void ComputeIterationF32Proc(__ubuf__ float *dstTensor, TSrc0 src0Tensor, TSrc1 src1Tensor,
                                          uint32_t offset, uint32_t elementsPerRepeat, RegTensor<float> &nanReg,
                                          RegTensor<float> &n2Reg, RegTensor<float> &oneReg, MaskReg &mask, bool isFmod)
{
    constexpr FloatUnion scale1(0x4b800000); // 2**24
    constexpr FloatUnion scale2(0x33800000); // 2**-24
    constexpr float subNormal = 1.1754944e-38;
    RegTensor<float> src0OriginReg, src1OriginReg, srcReg, dstReg;
    RegTensor<float> src0SignBitReg, src0SignBitTmpReg;
    RegTensor<float> bTmpReg, tmpReg;
    MaskReg subNormalMask, signDiffMask;
    if constexpr (std::is_same_v<__ubuf__ float *, TSrc0>) {
        vlds(src0OriginReg, src0Tensor, offset * elementsPerRepeat, NORM);
    } else {
        vdup(src0OriginReg, src0Tensor, mask, MODE_ZEROING);
    }
    if constexpr (std::is_same_v<__ubuf__ float *, TSrc1>) {
        vlds(src1OriginReg, src1Tensor, offset * elementsPerRepeat, NORM);
    } else {
        vdup(src1OriginReg, src1Tensor, mask, MODE_ZEROING);
    }
    vabs(srcReg, src1OriginReg, mask, MODE_ZEROING);
    if constexpr (isTail) {
        vabs(dstReg, src0OriginReg, mask, MODE_ZEROING);
        SolveScaleIter<iterationNum>(dstReg, srcReg, mask, isFmod);
    } else {
        vlds(dstReg, dstTensor, offset * elementsPerRepeat, NORM);
    }

    GetSignBit(src0SignBitReg, src0OriginReg, mask);
    vmul(src0SignBitTmpReg, src0SignBitReg, n2Reg, mask, MODE_ZEROING);
    vadd(src0SignBitTmpReg, src0SignBitTmpReg, oneReg, mask, MODE_ZEROING);
    vmul(dstReg, dstReg, src0SignBitTmpReg, mask, MODE_ZEROING);

    vcmps_le(subNormalMask, srcReg, subNormal, mask);
    vmuls(tmpReg, srcReg, scale1.f, subNormalMask, MODE_ZEROING);
    vsel(bTmpReg, tmpReg, srcReg, subNormalMask);

    vmuls(tmpReg, dstReg, scale2.f, subNormalMask, MODE_ZEROING);
    vsel(dstReg, tmpReg, dstReg, subNormalMask);

    SolveExceptionScenarios(dstReg, src0OriginReg, src1OriginReg, nanReg, mask);

    if (!isFmod) {
        vmul(tmpReg, src1OriginReg, dstReg, mask, MODE_ZEROING);
        vcmps_lt(signDiffMask, tmpReg, 0.0f, mask);
        vadd(tmpReg, dstReg, src1OriginReg, signDiffMask, MODE_ZEROING);
        vsel(dstReg, tmpReg, dstReg, signDiffMask);
    }

    constexpr auto distValue =
        std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<float, DistVST::DIST_NORM>())>();
    vsts(dstReg, dstTensor, offset * elementsPerRepeat, distValue, mask);
}

template <int32_t iterationNum, typename TSrc0, typename TSrc1>
PTO_INTERNAL void ComputeIterationF32(__ubuf__ float *dstTensor, TSrc0 src0Tensor, TSrc1 src1Tensor,
                                      const uint16_t mainRepeatTimes, uint32_t elementsPerRepeat, uint32_t tailCount,
                                      bool isFmod = true)
{
    __VEC_SCOPE__
    {
        RegTensor<float> nanReg, n2Reg, oneReg;
        MaskReg maskReg, maskFull = pset_b32(PAT_ALL);

        vdup(nanReg, nan.f, maskFull, MODE_ZEROING);
        vdup(n2Reg, static_cast<float>(-2.0), maskFull, MODE_ZEROING);
        vdup(oneReg, static_cast<float>(1), maskFull, MODE_ZEROING);

        SolveScaleIter<iterationNum>(dstTensor, src0Tensor, src1Tensor, mainRepeatTimes, maskFull, elementsPerRepeat,
                                     isFmod);

        mem_bar(VST_VLD);

        for (uint16_t i = 0; i < mainRepeatTimes; i++) {
            ComputeIterationF32Proc<false, iterationNum>(dstTensor, src0Tensor, src1Tensor, i, elementsPerRepeat,
                                                         nanReg, n2Reg, oneReg, maskFull, isFmod);
        }

        if (tailCount > 0) {
            maskReg = CreatePredicate<float>(tailCount);
            ComputeIterationF32Proc<true, iterationNum>(dstTensor, src0Tensor, src1Tensor, mainRepeatTimes,
                                                        elementsPerRepeat, nanReg, n2Reg, oneReg, maskReg, isFmod);
        }
    }
}
} // namespace pto
#endif // TINSERT_CUSTOM_HPP