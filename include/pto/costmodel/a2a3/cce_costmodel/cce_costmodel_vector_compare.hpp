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

#include <pto/costmodel/a2a3/cce_costmodel/cce_cycle_profiles_generated.hpp>

inline void vcmpv_eq(
    auto dst, auto src0, auto src1, auto repeat, auto dstBlockStride, auto src0BlockStride, auto src1BlockStride,
    auto dstRepeatStride, auto src0RepeatStride, auto src1RepeatStride)
{
    const uint64_t cycles = cce_costmodel_detail::EstimatePredicateCompletion(repeat, src0, false);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::VECTOR, "vcmpv_eq", cycles, dst, src0, src1, repeat, dstBlockStride,
        src0BlockStride, src1BlockStride, dstRepeatStride, src0RepeatStride, src1RepeatStride);
}
inline void vcmpv_ge(
    auto dst, auto src0, auto src1, auto repeat, auto dstBlockStride, auto src0BlockStride, auto src1BlockStride,
    auto dstRepeatStride, auto src0RepeatStride, auto src1RepeatStride)
{
    const uint64_t cycles =
        cce_costmodel_detail::EstimateLinearCycles(repeat, src0, cce_cycle_profiles::kVcmpvGeProfile);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::VECTOR, "vcmpv_ge", cycles, dst, src0, src1, repeat, dstBlockStride,
        src0BlockStride, src1BlockStride, dstRepeatStride, src0RepeatStride, src1RepeatStride);
}
inline void vcmpv_gt(
    auto dst, auto src0, auto src1, auto repeat, auto dstBlockStride, auto src0BlockStride, auto src1BlockStride,
    auto dstRepeatStride, auto src0RepeatStride, auto src1RepeatStride)
{
    const uint64_t cycles =
        cce_costmodel_detail::EstimateLinearCycles(repeat, src0, cce_cycle_profiles::kVcmpvGtProfile);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::VECTOR, "vcmpv_gt", cycles, dst, src0, src1, repeat, dstBlockStride,
        src0BlockStride, src1BlockStride, dstRepeatStride, src0RepeatStride, src1RepeatStride);
}
inline void vcmpv_le(
    auto dst, auto src0, auto src1, auto repeat, auto dstBlockStride, auto src0BlockStride, auto src1BlockStride,
    auto dstRepeatStride, auto src0RepeatStride, auto src1RepeatStride)
{
    const uint64_t cycles = [](auto r, auto ptr) {
        if constexpr (cce_costmodel_detail::IsIntegralPointer<decltype(ptr)>) {
            return cce_costmodel_detail::EstimateLinearCycles(r, ptr, cce_cycle_profiles::kVcmpvLeProfile);
        }
        return cce_costmodel_detail::EstimatePredicateCompletion(r, ptr, false);
    }(repeat, src0);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::VECTOR, "vcmpv_le", cycles, dst, src0, src1, repeat, dstBlockStride,
        src0BlockStride, src1BlockStride, dstRepeatStride, src0RepeatStride, src1RepeatStride);
}
inline void vcmpv_lt(
    auto dst, auto src0, auto src1, auto repeat, auto dstBlockStride, auto src0BlockStride, auto src1BlockStride,
    auto dstRepeatStride, auto src0RepeatStride, auto src1RepeatStride)
{
    const uint64_t cycles = [](auto r, auto ptr) {
        if constexpr (cce_costmodel_detail::IsIntegralPointer<decltype(ptr)>) {
            return cce_costmodel_detail::EstimateLinearCycles(r, ptr, cce_cycle_profiles::kVcmpvLtIntegerProfile);
        }
        return cce_costmodel_detail::EstimateLinearCycles(r, ptr, cce_cycle_profiles::kVcmpvLtFloatingProfile);
    }(repeat, src0);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::VECTOR, "vcmpv_lt", cycles, dst, src0, src1, repeat, dstBlockStride,
        src0BlockStride, src1BlockStride, dstRepeatStride, src0RepeatStride, src1RepeatStride);
}
inline void vcmpv_ne(
    auto dst, auto src0, auto src1, auto repeat, auto dstBlockStride, auto src0BlockStride, auto src1BlockStride,
    auto dstRepeatStride, auto src0RepeatStride, auto src1RepeatStride)
{
    const uint64_t cycles =
        cce_costmodel_detail::EstimateLinearCycles(repeat, src0, cce_cycle_profiles::kVcmpvNeProfile);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::VECTOR, "vcmpv_ne", cycles, dst, src0, src1, repeat, dstBlockStride,
        src0BlockStride, src1BlockStride, dstRepeatStride, src0RepeatStride, src1RepeatStride);
}
inline void vcmpvs_eq(
    auto dst, auto src0, auto src1, auto repeat, auto dstBlockStride, auto src0BlockStride, auto dstRepeatStride,
    auto srcRepeatStride)
{
    const uint64_t cycles = [](auto r, auto ptr) {
        if constexpr (cce_costmodel_detail::IsIntegralPointer<decltype(ptr)>) {
            return cce_costmodel_detail::EstimateLinearCycles(r, ptr, cce_cycle_profiles::kVcmpvsEqProfile);
        }
        return cce_costmodel_detail::EstimatePredicateCompletion(r, ptr, false);
    }(repeat, src0);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::VECTOR, "vcmpvs_eq", cycles, dst, src0, src1, repeat, dstBlockStride,
        src0BlockStride, dstRepeatStride, srcRepeatStride);
}
inline void vcmpvs_ge(
    auto dst, auto src0, auto src1, auto repeat, auto dstBlockStride, auto src0BlockStride, auto dstRepeatStride,
    auto srcRepeatStride)
{
    const uint64_t cycles = [](auto r, auto ptr) {
        if constexpr (cce_costmodel_detail::IsIntegralPointer<decltype(ptr)>) {
            return cce_costmodel_detail::EstimateLinearCycles(r, ptr, cce_cycle_profiles::kVcmpvsGeProfile);
        }
        return cce_costmodel_detail::EstimatePredicateCompletion(r, ptr, true);
    }(repeat, src0);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::VECTOR, "vcmpvs_ge", cycles, dst, src0, src1, repeat, dstBlockStride,
        src0BlockStride, dstRepeatStride, srcRepeatStride);
}
inline void vcmpvs_gt(
    auto dst, auto src0, auto src1, auto repeat, auto dstBlockStride, auto src0BlockStride, auto dstRepeatStride,
    auto srcRepeatStride)
{
    const uint64_t cycles = [](auto r, auto ptr) {
        if constexpr (cce_costmodel_detail::IsIntegralPointer<decltype(ptr)>) {
            return cce_costmodel_detail::EstimateLinearCycles(r, ptr, cce_cycle_profiles::kVcmpvsGtProfile);
        }
        return cce_costmodel_detail::EstimatePredicateCompletion(r, ptr, true);
    }(repeat, src0);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::VECTOR, "vcmpvs_gt", cycles, dst, src0, src1, repeat, dstBlockStride,
        src0BlockStride, dstRepeatStride, srcRepeatStride);
}
inline void vcmpvs_le(
    auto dst, auto src0, auto src1, auto repeat, auto dstBlockStride, auto src0BlockStride, auto dstRepeatStride,
    auto srcRepeatStride)
{
    const uint64_t cycles = [](auto r, auto ptr) {
        if constexpr (cce_costmodel_detail::IsIntegralPointer<decltype(ptr)>) {
            return cce_costmodel_detail::EstimateLinearCycles(r, ptr, cce_cycle_profiles::kVcmpvsLeProfile);
        }
        return cce_costmodel_detail::EstimatePredicateCompletion(r, ptr, true);
    }(repeat, src0);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::VECTOR, "vcmpvs_le", cycles, dst, src0, src1, repeat, dstBlockStride,
        src0BlockStride, dstRepeatStride, srcRepeatStride);
}
inline void vcmpvs_lt(
    auto dst, auto src0, auto src1, auto repeat, auto dstBlockStride, auto src0BlockStride, auto dstRepeatStride,
    auto srcRepeatStride)
{
    const uint64_t cycles = [](auto r, auto ptr) {
        if constexpr (cce_costmodel_detail::IsIntegralPointer<decltype(ptr)>) {
            return cce_costmodel_detail::EstimateLinearCycles(r, ptr, cce_cycle_profiles::kVcmpvsLtProfile);
        }
        return cce_costmodel_detail::EstimatePredicateCompletion(r, ptr, true);
    }(repeat, src0);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::VECTOR, "vcmpvs_lt", cycles, dst, src0, src1, repeat, dstBlockStride,
        src0BlockStride, dstRepeatStride, srcRepeatStride);
}
inline void vcmpvs_ne(
    auto dst, auto src0, auto src1, auto repeat, auto dstBlockStride, auto src0BlockStride, auto dstRepeatStride,
    auto srcRepeatStride)
{
    const uint64_t cycles = [](auto r, auto ptr) {
        if constexpr (cce_costmodel_detail::IsIntegralPointer<decltype(ptr)>) {
            return cce_costmodel_detail::EstimateLinearCycles(r, ptr, cce_cycle_profiles::kVcmpvsNeProfile);
        }
        return cce_costmodel_detail::EstimatePredicateCompletion(r, ptr, true);
    }(repeat, src0);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::VECTOR, "vcmpvs_ne", cycles, dst, src0, src1, repeat, dstBlockStride,
        src0BlockStride, dstRepeatStride, srcRepeatStride);
}
