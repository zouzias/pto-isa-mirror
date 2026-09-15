/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#pragma once

#include <pto/costmodel/a2a3/cce_costmodel/cce_costmodel_core.hpp>

inline void scatter_vnchwconv_b16(auto dst, auto src, auto repeat, auto dstStride, auto srcStride)
{
    const uint64_t cycles =
        pto_costmodel_detail::Estimate910B1LinearCycles(repeat, static_cast<half*>(nullptr), 9.025814, 24.001344,
                                                        9.025814, 24.001344);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::VECTOR, "scatter_vnchwconv_b16", cycles, dst, src, repeat, dstStride,
        srcStride);
}
inline void scatter_vnchwconv_b32(auto dst, auto src, auto repeat, auto dstStride, auto srcStride)
{
    const uint64_t cycles =
        pto_costmodel_detail::Estimate910B1LinearCycles(repeat, static_cast<float*>(nullptr), 9.025814, 24.001344,
                                                        9.025814, 24.001344);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::VECTOR, "scatter_vnchwconv_b32", cycles, dst, src, repeat, dstStride,
        srcStride);
}
inline void scatter_vnchwconv_b8(
    auto dst, auto src, auto repeat, auto dstStride, auto srcStride, auto dstHighHalf, auto srcHighHalf)
{
    const bool reuse = dstStride == 0 && srcStride == 0;
    const uint64_t cycles = reuse ?
                                pto_costmodel_detail::Estimate910B1LinearCycles(
                                    repeat, static_cast<uint8_t*>(nullptr), 11.034115, 46.000395, 11.034115, 46.000395) :
                                pto_costmodel_detail::Estimate910B1LinearCycles(
                                    repeat, static_cast<uint8_t*>(nullptr), 18.388306, 40.003055, 18.388306, 40.003055);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::VECTOR, "scatter_vnchwconv_b8", cycles, dst, src, repeat, dstStride,
        srcStride, dstHighHalf, srcHighHalf);
}
inline void vabs(
    auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride,
    auto srcRepeatStride)
{
    const uint64_t cycles =
        pto_costmodel_detail::Estimate910B1LinearCycles(repeat, dst, 4.0, 1.0, 4.0, 1.0);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::VECTOR, "vabs", cycles, dst, src, repeat, dstBlockStride, srcBlockStride,
        dstRepeatStride, srcRepeatStride);
}
inline void vadd(
    auto dst, auto src0, auto src1, auto repeat, auto dstBlockStride, auto src0BlockStride, auto src1BlockStride,
    auto dstRepeatStride, auto src0RepeatStride, auto src1RepeatStride)
{
    const uint64_t cycles = pto_costmodel_detail::Estimate910B1LinearCyclesWithInteger(
        repeat, dst, 5.934212, 2.065788, 5.933726, 2.066274, 4.985765, 2.000853);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::VECTOR, "vadd", cycles, dst, src0, src1, repeat, dstBlockStride,
        src0BlockStride, src1BlockStride, dstRepeatStride, src0RepeatStride, src1RepeatStride);
}
inline void vadds(
    auto dst, auto src0, auto src1, auto repeat, auto dstBlockStride, auto src0BlockStride, auto dstRepeatStride,
    auto src0RepeatStride)
{
    const uint64_t cycles = pto_costmodel_detail::Estimate910B1LinearCyclesWithInteger(
        repeat, dst, 6.0, 1.0, 6.0, 1.0, 5.010874, 1.000947);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::VECTOR, "vadds", cycles, dst, src0, src1, repeat, dstBlockStride,
        src0BlockStride, dstRepeatStride, src0RepeatStride);
}
inline void vand(
    auto dst, auto src0, auto src1, auto repeat, auto dstBlockStride, auto src0BlockStride, auto src1BlockStride,
    auto dstRepeatStride, auto src0RepeatStride, auto src1RepeatStride)
{
    // Native UINT16 completion profile. The API count-mask dispatch floor is
    // measured separately and remains a PTO-level charge.
    // Keep the CCE leaf identical to tilesim's native UINT16 completion
    // profile.  Count-mask/API dispatch belongs to the PTO projection layer;
    // charging a fixed 16 cycles here overprices every VAND in a composite.
    const uint64_t cycles = pto_costmodel_detail::Estimate910B1LinearCycles(
        repeat, dst, 2.059771, 1.999717, 2.059771, 1.999717);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::VECTOR, "vand", cycles, dst, src0, src1, repeat, dstBlockStride,
        src0BlockStride, src1BlockStride, dstRepeatStride, src0RepeatStride, src1RepeatStride);
}
inline void vaxpy(
    auto dst, auto src0, auto src1, auto repeat, auto dstBlockStride, auto src0BlockStride, auto dstRepeatStride,
    auto src0RepeatStride)
{
    const uint64_t cycles = pto_costmodel_detail::Estimate910B1LinearCycles(repeat, dst, 18.0, 2.0, 18.0, 2.0);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::VECTOR, "vaxpy", cycles, dst, src0, src1, repeat, dstBlockStride,
        src0BlockStride, dstRepeatStride, src0RepeatStride);
}
inline void vbitsort(auto dst, auto src, auto idx, auto repeat)
{
    const uint64_t cycles = pto_costmodel_detail::Estimate910B1LinearCycles(repeat, src, 8.0, 4.0, 8.0, 4.0);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vbitsort", cycles, dst, src, idx, repeat);
}
inline void vbrcb(auto dst, auto src, auto dstBlockStride, auto dstRepeatStride, auto repeat)
{
    const uint64_t cycles =
        pto_costmodel_detail::Estimate910B1LinearCycles(repeat, dst, 1.0, 1.0, 1.0, 1.0);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::VECTOR, "vbrcb", cycles, dst, src, dstBlockStride, dstRepeatStride, repeat);
}
inline void vcadd(
    auto dst, auto src, auto repeat, auto dstRepeatStride, auto srcBlockStride, auto srcRepeatStride, auto mode)
{
    const uint64_t cycles =
        pto_costmodel_detail::Estimate910B1LinearCycles(repeat, dst, 23.0, 7.0, 20.0, 7.0);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::VECTOR, "vcadd", cycles, dst, src, repeat, dstRepeatStride, srcBlockStride,
        srcRepeatStride, mode);
}
inline void vcgadd(auto dst, auto src, auto repeat, auto dstRepeatStride, auto src0RepeatStride, auto src1RepeatStride)
{
    const uint64_t cycles =
        pto_costmodel_detail::Estimate910B1LinearCycles(repeat, dst, 15.0, 7.0, 12.0, 1.0);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::VECTOR, "vcgadd", cycles, dst, src, repeat, dstRepeatStride,
        src0RepeatStride, src1RepeatStride);
}
inline void vcgmax(auto dst, auto src, auto repeat, auto dstRepeatStride, auto src0RepeatStride, auto src1RepeatStride)
{
    const uint64_t cycles =
        pto_costmodel_detail::Estimate910B1LinearCycles(repeat, dst, 6.0, 7.0, 5.0, 1.0);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::VECTOR, "vcgmax", cycles, dst, src, repeat, dstRepeatStride,
        src0RepeatStride, src1RepeatStride);
}
inline void vcgmin(auto dst, auto src, auto repeat, auto dstRepeatStride, auto src0RepeatStride, auto src1RepeatStride)
{
    const uint64_t cycles =
        pto_costmodel_detail::Estimate910B1LinearCycles(repeat, dst, 6.0, 7.0, 5.0, 1.0);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::VECTOR, "vcgmin", cycles, dst, src, repeat, dstRepeatStride,
        src0RepeatStride, src1RepeatStride);
}
inline void vcmax(
    auto dst, auto src, auto repeat, auto dstRepeatStride, auto srcBlockStride, auto srcRepeatStride, auto mode)
{
    const uint64_t cycles =
        pto_costmodel_detail::Estimate910B1LinearCycles(repeat, dst, 9.0, 7.0, 8.0, 7.0);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::VECTOR, "vcmax", cycles, dst, src, repeat, dstRepeatStride, srcBlockStride,
        srcRepeatStride, mode);
}
inline void vcmin(
    auto dst, auto src, auto repeat, auto dstRepeatStride, auto srcBlockStride, auto srcRepeatStride, auto mode)
{
    const uint64_t cycles =
        pto_costmodel_detail::Estimate910B1LinearCycles(repeat, dst, 9.0, 7.0, 8.0, 7.0);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::VECTOR, "vcmin", cycles, dst, src, repeat, dstRepeatStride, srcBlockStride,
        srcRepeatStride, mode);
}
inline void vcopy(
    auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride,
    auto srcRepeatStride)
{
    const uint64_t cycles =
        pto_costmodel_detail::Estimate910B1LinearCycles(repeat, dst, 13.0, 1.0, 0.936906, 1.063094);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::VECTOR, "vcopy", cycles, dst, src, repeat, dstBlockStride, srcBlockStride,
        dstRepeatStride, srcRepeatStride);
}
// vtranspose: single VECTOR instruction transposing a 16x16 b16 block (dst, src are __ubuf__ uint16_t*).
// Cycles are a coarse uncalibrated estimate; kept consistent with the per-repeat cost of
// scatter_vnchwconv_b16 (one instruction per 16-row block).
inline void vtranspose(auto dst, auto src)
{
    const uint64_t cycles = EstimateLinearCycles(1);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vtranspose", cycles, dst, src);
}
inline void vdiv(
    auto dst, auto src0, auto src1, auto repeat, auto dstBlockStride, auto src0BlockStride, auto src1BlockStride,
    auto dstRepeatStride, auto src0RepeatStride, auto src1RepeatStride)
{
    const uint64_t cycles =
        pto_costmodel_detail::Estimate910B1LinearCycles(repeat, dst, 13.0, 8.0, 13.0, 4.0);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::VECTOR, "vdiv", cycles, dst, src0, src1, repeat, dstBlockStride,
        src0BlockStride, src1BlockStride, dstRepeatStride, src0RepeatStride, src1RepeatStride);
}
inline void vector_dup(
    auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride,
    auto srcRepeatStride)
{
    const uint64_t cycles =
        pto_costmodel_detail::Estimate910B1LinearCycles(repeat, dst, 1.0, 1.0, 1.0, 1.0);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::VECTOR, "vector_dup", cycles, dst, src, repeat, dstBlockStride,
        srcBlockStride, dstRepeatStride, srcRepeatStride);
}
inline void vexp(
    auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride,
    auto srcRepeatStride)
{
    const uint64_t cycles =
        pto_costmodel_detail::Estimate910B1LinearCycles(repeat, dst, 12.011812, 3.988188, 12.0, 2.0);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::VECTOR, "vexp", cycles, dst, src, repeat, dstBlockStride, srcBlockStride,
        dstRepeatStride, srcRepeatStride);
}
inline void vgather(auto dst, auto offset, auto srcBaseAddr, auto dstRepeatStride, auto repeat)
{
    // Native contiguous-offset completion profile. Other offset distributions
    // require information that the current PTO stub does not record.
    const uint64_t cycles = pto_costmodel_detail::Estimate910B1LinearCycles(
        repeat, dst, 6.635214, 24.008236, 4.965026, 12.000331);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::VECTOR, "vgather", cycles, dst, offset, srcBaseAddr, dstRepeatStride,
        repeat);
}
inline void vgatherb(auto dst, auto offset, auto srcBaseAddr, auto dstRepeatStride, auto dstBlockStride, auto repeat)
{
    const uint64_t cycles = EstimateLinearCycles(repeat);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::VECTOR, "vgatherb", cycles, dst, offset, srcBaseAddr, dstRepeatStride,
        dstBlockStride, repeat);
}
inline void vln(
    auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride,
    auto srcRepeatStride)
{
    const uint64_t cycles =
        pto_costmodel_detail::Estimate910B1LinearCycles(repeat, dst, 13.985971, 4.014029, 14.0, 2.0);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::VECTOR, "vln", cycles, dst, src, repeat, dstBlockStride, srcBlockStride,
        dstRepeatStride, srcRepeatStride);
}
inline void vlrelu(
    auto dst, auto src0, auto src1, auto repeat, auto dstBlockStride, auto src0BlockStride, auto dstRepeatStride,
    auto src0RepeatStride)
{
    // TileSim maps VLRELU to the RELU timing class on A2/A3.
    const uint64_t cycles =
        pto_costmodel_detail::Estimate910B1LinearCycles(repeat, dst, 7.0, 1.0, 7.0, 1.0);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::VECTOR, "vlrelu", cycles, dst, src0, src1, repeat, dstBlockStride,
        src0BlockStride, dstRepeatStride, src0RepeatStride);
}
inline void vmax(
    auto dst, auto src0, auto src1, auto repeat, auto dstBlockStride, auto src0BlockStride, auto src1BlockStride,
    auto dstRepeatStride, auto src0RepeatStride, auto src1RepeatStride)
{
    const uint64_t cycles = pto_costmodel_detail::Estimate910B1LinearCyclesWithInteger(
        repeat, dst, 4.0, 2.0, 4.0, 2.0, 4.985765, 2.000853);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::VECTOR, "vmax", cycles, dst, src0, src1, repeat, dstBlockStride,
        src0BlockStride, src1BlockStride, dstRepeatStride, src0RepeatStride, src1RepeatStride);
}
inline void vmaxs(
    auto dst, auto src0, auto src1, auto repeat, auto dstBlockStride, auto src0BlockStride, auto dstRepeatStride,
    auto src0RepeatStride)
{
    const uint64_t cycles = pto_costmodel_detail::Estimate910B1LinearCyclesWithInteger(
        repeat, dst, 4.0, 1.0, 4.0, 1.0, 5.010874, 1.000947);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::VECTOR, "vmaxs", cycles, dst, src0, src1, repeat, dstBlockStride,
        src0BlockStride, dstRepeatStride, src0RepeatStride);
}
inline void vmin(
    auto dst, auto src0, auto src1, auto repeat, auto dstBlockStride, auto src0BlockStride, auto src1BlockStride,
    auto dstRepeatStride, auto src0RepeatStride, auto src1RepeatStride)
{
    const uint64_t cycles = pto_costmodel_detail::Estimate910B1LinearCyclesWithInteger(
        repeat, dst, 4.0, 2.0, 4.0, 2.0, 4.985765, 2.000853);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::VECTOR, "vmin", cycles, dst, src0, src1, repeat, dstBlockStride,
        src0BlockStride, src1BlockStride, dstRepeatStride, src0RepeatStride, src1RepeatStride);
}
inline void vmins(
    auto dst, auto src0, auto src1, auto repeat, auto dstBlockStride, auto src0BlockStride, auto dstRepeatStride,
    auto src0RepeatStride)
{
    const uint64_t cycles = pto_costmodel_detail::Estimate910B1LinearCyclesWithInteger(
        repeat, dst, 4.0, 1.0, 4.0, 1.0, 5.010874, 1.000947);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::VECTOR, "vmins", cycles, dst, src0, src1, repeat, dstBlockStride,
        src0BlockStride, dstRepeatStride, src0RepeatStride);
}
inline void vmrgsort4(auto dst, auto addrArray, auto count, auto config)
{
    // VMS4V2 faults the 910B1 vector core during remeasurement, so tilesim
    // intentionally retains the legacy head=14, computing=6 entry.
    const uint64_t repeat = ExtractBits(config, 0, 0xffULL);
    const uint64_t cycles =
        pto_costmodel_detail::Estimate910B1LinearCycles(repeat, dst, 8.0, 6.0, 8.0, 6.0);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::VECTOR, "vmrgsort4", cycles, dst, addrArray, count, config);
}
inline void vmul(
    auto dst, auto src0, auto src1, auto repeat, auto dstBlockStride, auto src0BlockStride, auto src1BlockStride,
    auto dstRepeatStride, auto src0RepeatStride, auto src1RepeatStride)
{
    const uint64_t cycles = pto_costmodel_detail::Estimate910B1LinearCyclesWithInteger(
        repeat, dst, 6.865515, 2.134485, 6.868098, 2.131902, 6.106646, 1.999717);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::VECTOR, "vmul", cycles, dst, src0, src1, repeat, dstBlockStride,
        src0BlockStride, src1BlockStride, dstRepeatStride, src0RepeatStride, src1RepeatStride);
}
inline void vmadd(
    auto dst, auto src0, auto src1, auto repeat, auto dstBlockStride, auto src0BlockStride, auto src1BlockStride,
    auto dstRepeatStride, auto src0RepeatStride, auto src1RepeatStride)
{
    const uint64_t cycles =
        pto_costmodel_detail::Estimate910B1LinearCycles(repeat, dst, 11.021485, 6.000879, 11.021485, 6.000879);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::VECTOR, "vmadd", cycles, dst, src0, src1, repeat, dstBlockStride,
        src0BlockStride, src1BlockStride, dstRepeatStride, src0RepeatStride, src1RepeatStride);
}
inline void vmla(
    auto dst, auto src0, auto src1, auto repeat, auto dstBlockStride, auto src0BlockStride, auto src1BlockStride,
    auto dstRepeatStride, auto src0RepeatStride, auto src1RepeatStride)
{
    // tilesim cce_910b1_20260911.json contains both separate-operand VMLA
    // (11.0174 + 6.00024R) and accumulator-alias VMLA
    // (11.1004 + 3.99978R).  TMULA's full-width path aliases dst with the
    // accumulator and uses repeat stride 8; keep that distinction here.
    const bool accumulatorAlias = dstRepeatStride == 8 && src0RepeatStride == 8 && src1RepeatStride == 8;
    const uint64_t cycles = accumulatorAlias ?
                                pto_costmodel_detail::Estimate910B1LinearCycles(
                                    repeat, dst, 11.100439, 3.999784, 11.100439, 3.999784) :
                                pto_costmodel_detail::Estimate910B1LinearCycles(
                                    repeat, dst, 11.017372, 6.000241, 11.017372, 6.000241);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::VECTOR, "vmla", cycles, dst, src0, src1, repeat, dstBlockStride,
        src0BlockStride, src1BlockStride, dstRepeatStride, src0RepeatStride, src1RepeatStride);
}
inline void vmuls(
    auto dst, auto src0, auto src1, auto repeat, auto dstBlockStride, auto src0BlockStride, auto dstRepeatStride,
    auto src0RepeatStride)
{
    const uint64_t cycles = pto_costmodel_detail::Estimate910B1LinearCyclesWithInteger(
        repeat, dst, 7.0, 1.0, 7.0, 1.0, 6.045203, 0.999906);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::VECTOR, "vmuls", cycles, dst, src0, src1, repeat, dstBlockStride,
        src0BlockStride, dstRepeatStride, src0RepeatStride);
}
inline void vnot(
    auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride,
    auto srcRepeatStride)
{
    const uint64_t cycles = pto_costmodel_detail::Estimate910B1LinearCycles(
        repeat, dst, 2.088006, 0.999735, 2.088006, 0.999735);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::VECTOR, "vnot", cycles, dst, src, repeat, dstBlockStride, srcBlockStride,
        dstRepeatStride, srcRepeatStride);
}
inline void vor(
    auto dst, auto src0, auto src1, auto repeat, auto dstBlockStride, auto src0BlockStride, auto src1BlockStride,
    auto dstRepeatStride, auto src0RepeatStride, auto src1RepeatStride)
{
    // See VAND: preserve tilesim's leaf timing and let PTO-level topology
    // account for mask/observer overhead once per public instruction.
    const uint64_t cycles = pto_costmodel_detail::Estimate910B1LinearCycles(
        repeat, dst, 2.059771, 1.999717, 2.059771, 1.999717);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::VECTOR, "vor", cycles, dst, src0, src1, repeat, dstBlockStride,
        src0BlockStride, src1BlockStride, dstRepeatStride, src0RepeatStride, src1RepeatStride);
}
inline void vreducev2(
    auto dst, auto src0, auto src1, auto repeat, auto src0BlockStride, auto modeOrMaskPattern, auto src0RepeatStride,
    auto src1RepeatStride)
{
    const uint64_t cycles = pto_costmodel_detail::Estimate910B1LinearCycles(repeat, src0, -2.0, 14.0, -2.0, 14.0);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::VECTOR, "vreducev2", cycles, dst, src0, src1, repeat, src0BlockStride,
        modeOrMaskPattern, src0RepeatStride, src1RepeatStride);
}
inline void vrelu(
    auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride,
    auto srcRepeatStride)
{
    const uint64_t cycles =
        pto_costmodel_detail::Estimate910B1LinearCycles(repeat, dst, 4.0, 1.0, 4.0, 1.0);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::VECTOR, "vrelu", cycles, dst, src, repeat, dstBlockStride, srcBlockStride,
        dstRepeatStride, srcRepeatStride);
}
inline void vrsqrt(
    auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride,
    auto srcRepeatStride)
{
    // Native 910B1 VRSQRT completion fit from tilesim's dedicated hardware
    // calibration. FP16 and FP32 share the same contiguous-repeat timing;
    // VRSQRT must not reuse the substantially slower VSQRT profile.
    const uint64_t cycles = pto_costmodel_detail::Estimate910B1LinearCycles(
        repeat, dst, 4.791896, 1.064811, 4.791896, 1.064811);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::VECTOR, "vrsqrt", cycles, dst, src, repeat, dstBlockStride, srcBlockStride,
        dstRepeatStride, srcRepeatStride);
}
// Raw A2/A3 compare-mask intrinsics used by TCOLARG{MAX,MIN} and
// TPARTARG{MAX,MIN}.  They write CMPMASK rather than an explicit destination,
// so the first source pointer is used only to select the FP16/FP32 tilesim
// profile.  The 910B1 configuration gives repeat-1 completion cycles of 12
// for GE (startup 10 + slope 2) and 13 for LE (startup 11 + slope 2).
inline void vcmp_ge(
    auto src0, auto src1, auto repeat, auto src0BlockStride, auto src1BlockStride, auto src0RepeatStride,
    auto src1RepeatStride, auto reserved0, auto reserved1)
{
    const uint64_t cycles =
        pto_costmodel_detail::Estimate910B1LinearCycles(repeat, src0, 10.0, 2.0, 10.0, 2.0);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::VECTOR, "vcmp_ge", cycles, src0, src1, repeat, src0BlockStride,
        src1BlockStride, src0RepeatStride, src1RepeatStride, reserved0, reserved1);
}
inline void vcmp_le(
    auto src0, auto src1, auto repeat, auto src0BlockStride, auto src1BlockStride, auto src0RepeatStride,
    auto src1RepeatStride, auto reserved0, auto reserved1)
{
    const uint64_t cycles =
        pto_costmodel_detail::Estimate910B1LinearCycles(repeat, src0, 11.0, 2.0, 11.0, 2.0);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::VECTOR, "vcmp_le", cycles, src0, src1, repeat, src0BlockStride,
        src1BlockStride, src0RepeatStride, src1RepeatStride, reserved0, reserved1);
}
inline void vsel(
    auto dst, auto src0, auto src1, auto repeat, auto dstBlockStride, auto src0BlockStride, auto src1BlockStride,
    auto dstRepeatStride, auto src0RepeatStride, auto src1RepeatStride, auto mode)
{
    const bool tensorScalarMode = static_cast<int>(mode) == 1;
    const uint64_t cycles = tensorScalarMode ?
                                pto_costmodel_detail::Estimate910B1LinearCycles(
                                    repeat, dst, 1.905969, 1.063258, 1.903226, 1.029961) :
                                pto_costmodel_detail::Estimate910B1LinearCycles(
                                    repeat, dst, 5.0, 2.0, 5.0, 2.0);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::VECTOR, "vsel", cycles, dst, src0, src1, repeat, dstBlockStride,
        src0BlockStride, src1BlockStride, dstRepeatStride, src0RepeatStride, src1RepeatStride, mode);
}
// Some low-level PTO implementations use the legacy CMPMASK overload without
// the explicit mode argument.  It is semantically VSEL_CMPMASK_SPR (mode 0).
inline void vsel(
    auto dst, auto src0, auto src1, auto repeat, auto dstBlockStride, auto src0BlockStride, auto src1BlockStride,
    auto dstRepeatStride, auto src0RepeatStride, auto src1RepeatStride)
{
    vsel(
        dst, src0, src1, repeat, dstBlockStride, src0BlockStride, src1BlockStride, dstRepeatStride, src0RepeatStride,
        src1RepeatStride, 0);
}
inline void vshl(
    auto dst, auto src0, auto src1, auto repeat, auto dstBlockStride, auto src0BlockStride, auto dstRepeatStride,
    auto src0RepeatStride)
{
    const uint64_t cycles = pto_costmodel_detail::Estimate910B1LinearCycles(
        repeat, dst, 6.045203, 0.999906, 6.045203, 0.999906);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::VECTOR, "vshl", cycles, dst, src0, src1, repeat, dstBlockStride,
        src0BlockStride, dstRepeatStride, src0RepeatStride);
}
inline void vshr(
    auto dst, auto src0, auto src1, auto repeat, auto dstBlockStride, auto src0BlockStride, auto dstRepeatStride,
    auto src0RepeatStride, auto isArithmetic = false)
{
    const uint64_t cycles = pto_costmodel_detail::Estimate910B1LinearCycles(
        repeat, dst, 6.045203, 0.999906, 6.045203, 0.999906);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::VECTOR, "vshr", cycles, dst, src0, src1, repeat, dstBlockStride,
        src0BlockStride, dstRepeatStride, src0RepeatStride, isArithmetic);
}
inline void vsqrt(
    auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride,
    auto srcRepeatStride)
{
    const uint64_t cycles =
        pto_costmodel_detail::Estimate910B1LinearCycles(repeat, dst, 13.0, 4.0, 13.0, 2.0);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::VECTOR, "vsqrt", cycles, dst, src, repeat, dstBlockStride, srcBlockStride,
        dstRepeatStride, srcRepeatStride);
}
inline void vsub(
    auto dst, auto src0, auto src1, auto repeat, auto dstBlockStride, auto src0BlockStride, auto src1BlockStride,
    auto dstRepeatStride, auto src0RepeatStride, auto src1RepeatStride)
{
    const uint64_t cycles = pto_costmodel_detail::Estimate910B1LinearCyclesWithInteger(
        repeat, dst, 5.934157, 2.065843, 5.935091, 2.064909, 4.985765, 2.000853);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::VECTOR, "vsub", cycles, dst, src0, src1, repeat, dstBlockStride,
        src0BlockStride, src1BlockStride, dstRepeatStride, src0RepeatStride, src1RepeatStride);
}
