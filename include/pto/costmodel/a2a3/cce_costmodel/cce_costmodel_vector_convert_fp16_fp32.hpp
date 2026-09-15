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

namespace cce_cycle_profiles {
inline constexpr cce_costmodel_detail::LinearCycleProfile kVconvBf162f32Profile{6.0, 2.0, 6.0, 2.0};
inline constexpr cce_costmodel_detail::LinearCycleProfile kVconvDeqProfile{7.0, 1.0, 7.0, 1.0};
inline constexpr cce_costmodel_detail::LinearCycleProfile kVconvF162f32Profile{5.0, 2.0, 6.0, 1.0};
inline constexpr cce_costmodel_detail::LinearCycleProfile kVconvF162s32aProfile{5.010874, 1.000947, 5.010874, 1.000947};
inline constexpr cce_costmodel_detail::LinearCycleProfile kVconvF162s32cProfile{5.010874, 1.000947, 5.010874, 1.000947};
inline constexpr cce_costmodel_detail::LinearCycleProfile kVconvF162s32fProfile{5.010874, 1.000947, 5.010874, 1.000947};
inline constexpr cce_costmodel_detail::LinearCycleProfile kVconvF162s32rProfile{5.010874, 1.000947, 5.010874, 1.000947};
inline constexpr cce_costmodel_detail::LinearCycleProfile kVconvF162s32zProfile{5.010874, 1.000947, 5.010874, 1.000947};
inline constexpr cce_costmodel_detail::ExpFillCycleProfile kVconvF162s8rProfile{26.75, 1.5139, 6.55, 4.76, 128, 0};
inline constexpr cce_costmodel_detail::ExpFillCycleProfile kVconvF322bf16aProfile{
    3.18969629639, 1.5323996238, 7.29569981045, 7.84302159992, 0, false};
inline constexpr cce_costmodel_detail::ExpFillCycleProfile kVconvF322bf16cProfile{
    3.03019185592, 1.53319991498, 7.34550694743, 8.27164169992, 0, false};
inline constexpr cce_costmodel_detail::ExpFillCycleProfile kVconvF322bf16fProfile{
    3.03019185592, 1.53319991498, 7.34550694743, 8.27164169992, 0, false};
inline constexpr cce_costmodel_detail::ExpFillCycleProfile kVconvF322bf16rProfile{
    3.03019185592, 1.53319991498, 7.34550694743, 8.27164169992, 0, false};
inline constexpr cce_costmodel_detail::ExpFillCycleProfile kVconvF322bf16zProfile{
    3.03019185592, 1.53319991498, 7.34550694743, 8.27164169992, 0, false};
inline constexpr cce_costmodel_detail::ExpFillCycleProfile kVconvF322f16Profile{-9.83278980605, 1.0, 14.3148137545,
                                                                                20.7763498993,  0,   true};
inline constexpr cce_costmodel_detail::ExpFillCycleProfile kVconvF322f16aProfile{-10.04731634734, 1.0, 14.5320682728,
                                                                                 20.9559502406,   0,   true};
inline constexpr cce_costmodel_detail::ExpFillCycleProfile kVconvF322f16cProfile{-9.83278980605, 1.0, 14.3148137545,
                                                                                 20.7763498993,  0,   true};
inline constexpr cce_costmodel_detail::ExpFillCycleProfile kVconvF322f16fProfile{-9.83278980605, 1.0, 14.3148137545,
                                                                                 20.7763498993,  0,   true};
inline constexpr cce_costmodel_detail::LinearCycleProfile kVconvF322f16oProfile{4.0, 1.0, 4.0, 1.0};
inline constexpr cce_costmodel_detail::ExpFillCycleProfile kVconvF322f16rProfile{-9.83278980605, 1.0, 14.3148137545,
                                                                                 20.7763498993,  0,   true};
inline constexpr cce_costmodel_detail::ExpFillCycleProfile kVconvF322f16zProfile{-9.83278980605, 1.0, 14.3148137545,
                                                                                 20.7763498993,  0,   true};
} // namespace cce_cycle_profiles

inline void vconv_bf162f32(
    auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride,
    auto srcRepeatStride)
{
    const uint64_t cycles =
        cce_costmodel_detail::EstimateLinearCycles(repeat, src, cce_cycle_profiles::kVconvBf162f32Profile);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_bf162f32", cycles, dst, src, repeat, dstBlockStride,
        srcBlockStride, dstRepeatStride, srcRepeatStride);
}
inline void vconv_bf162s32a(
    auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride,
    auto srcRepeatStride)
{
    const uint64_t cycles = EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_bf162s32a", cycles, dst, src, repeat, dstBlockStride,
        srcBlockStride, dstRepeatStride, srcRepeatStride);
}
inline void vconv_bf162s32c(
    auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride,
    auto srcRepeatStride)
{
    const uint64_t cycles = EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_bf162s32c", cycles, dst, src, repeat, dstBlockStride,
        srcBlockStride, dstRepeatStride, srcRepeatStride);
}
inline void vconv_bf162s32f(
    auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride,
    auto srcRepeatStride)
{
    const uint64_t cycles = EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_bf162s32f", cycles, dst, src, repeat, dstBlockStride,
        srcBlockStride, dstRepeatStride, srcRepeatStride);
}
inline void vconv_bf162s32r(
    auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride,
    auto srcRepeatStride)
{
    const uint64_t cycles = EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_bf162s32r", cycles, dst, src, repeat, dstBlockStride,
        srcBlockStride, dstRepeatStride, srcRepeatStride);
}
inline void vconv_bf162s32z(
    auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride,
    auto srcRepeatStride)
{
    const uint64_t cycles = EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_bf162s32z", cycles, dst, src, repeat, dstBlockStride,
        srcBlockStride, dstRepeatStride, srcRepeatStride);
}
inline void vconv_deq(
    auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride,
    auto srcRepeatStride)
{
    const uint64_t cycles =
        cce_costmodel_detail::EstimateLinearCycles(repeat, src, cce_cycle_profiles::kVconvDeqProfile);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_deq", cycles, dst, src, repeat, dstBlockStride,
        srcBlockStride, dstRepeatStride, srcRepeatStride);
}
inline void vconv_f162f32(
    auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride,
    auto srcRepeatStride)
{
    const uint64_t cycles =
        cce_costmodel_detail::EstimateLinearCycles(repeat, src, cce_cycle_profiles::kVconvF162f32Profile);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_f162f32", cycles, dst, src, repeat, dstBlockStride,
        srcBlockStride, dstRepeatStride, srcRepeatStride);
}
inline void vconv_f162s16a(
    auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride,
    auto srcRepeatStride)
{
    const uint64_t cycles = EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_f162s16a", cycles, dst, src, repeat, dstBlockStride,
        srcBlockStride, dstRepeatStride, srcRepeatStride);
}
inline void vconv_f162s16c(
    auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride,
    auto srcRepeatStride)
{
    const uint64_t cycles = EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_f162s16c", cycles, dst, src, repeat, dstBlockStride,
        srcBlockStride, dstRepeatStride, srcRepeatStride);
}
inline void vconv_f162s16f(
    auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride,
    auto srcRepeatStride)
{
    const uint64_t cycles = EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_f162s16f", cycles, dst, src, repeat, dstBlockStride,
        srcBlockStride, dstRepeatStride, srcRepeatStride);
}
inline void vconv_f162s16r(
    auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride,
    auto srcRepeatStride)
{
    const uint64_t cycles = EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_f162s16r", cycles, dst, src, repeat, dstBlockStride,
        srcBlockStride, dstRepeatStride, srcRepeatStride);
}
inline void vconv_f162s16z(
    auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride,
    auto srcRepeatStride)
{
    const uint64_t cycles = EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_f162s16z", cycles, dst, src, repeat, dstBlockStride,
        srcBlockStride, dstRepeatStride, srcRepeatStride);
}
inline void vconv_f162s32a(
    auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride,
    auto srcRepeatStride)
{
    const uint64_t cycles =
        cce_costmodel_detail::EstimateLinearCycles(repeat, src, cce_cycle_profiles::kVconvF162s32aProfile);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_f162s32a", cycles, dst, src, repeat, dstBlockStride,
        srcBlockStride, dstRepeatStride, srcRepeatStride);
}
inline void vconv_f162s32c(
    auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride,
    auto srcRepeatStride)
{
    const uint64_t cycles =
        cce_costmodel_detail::EstimateLinearCycles(repeat, src, cce_cycle_profiles::kVconvF162s32cProfile);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_f162s32c", cycles, dst, src, repeat, dstBlockStride,
        srcBlockStride, dstRepeatStride, srcRepeatStride);
}
inline void vconv_f162s32f(
    auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride,
    auto srcRepeatStride)
{
    const uint64_t cycles =
        cce_costmodel_detail::EstimateLinearCycles(repeat, src, cce_cycle_profiles::kVconvF162s32fProfile);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_f162s32f", cycles, dst, src, repeat, dstBlockStride,
        srcBlockStride, dstRepeatStride, srcRepeatStride);
}
inline void vconv_f162s32r(
    auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride,
    auto srcRepeatStride)
{
    const uint64_t cycles =
        cce_costmodel_detail::EstimateLinearCycles(repeat, src, cce_cycle_profiles::kVconvF162s32rProfile);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_f162s32r", cycles, dst, src, repeat, dstBlockStride,
        srcBlockStride, dstRepeatStride, srcRepeatStride);
}
inline void vconv_f162s32z(
    auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride,
    auto srcRepeatStride)
{
    const uint64_t cycles =
        cce_costmodel_detail::EstimateLinearCycles(repeat, src, cce_cycle_profiles::kVconvF162s32zProfile);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_f162s32z", cycles, dst, src, repeat, dstBlockStride,
        srcBlockStride, dstRepeatStride, srcRepeatStride);
}
inline void vconv_f162s8a(
    auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride,
    auto srcRepeatStride)
{
    const uint64_t cycles = EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_f162s8a", cycles, dst, src, repeat, dstBlockStride,
        srcBlockStride, dstRepeatStride, srcRepeatStride);
}
inline void vconv_f162s8c(
    auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride,
    auto srcRepeatStride)
{
    const uint64_t cycles = EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_f162s8c", cycles, dst, src, repeat, dstBlockStride,
        srcBlockStride, dstRepeatStride, srcRepeatStride);
}
inline void vconv_f162s8f(
    auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride,
    auto srcRepeatStride)
{
    const uint64_t cycles = EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_f162s8f", cycles, dst, src, repeat, dstBlockStride,
        srcBlockStride, dstRepeatStride, srcRepeatStride);
}
inline void vconv_f162s8r(
    auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride,
    auto srcRepeatStride)
{
    const uint64_t cycles =
        cce_costmodel_detail::EstimateExpFillCycles(repeat, src, cce_cycle_profiles::kVconvF162s8rProfile);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_f162s8r", cycles, dst, src, repeat, dstBlockStride,
        srcBlockStride, dstRepeatStride, srcRepeatStride);
}
inline void vconv_f162s8z(
    auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride,
    auto srcRepeatStride)
{
    const uint64_t cycles = EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_f162s8z", cycles, dst, src, repeat, dstBlockStride,
        srcBlockStride, dstRepeatStride, srcRepeatStride);
}
inline void vconv_f162s4(
    auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride,
    auto srcRepeatStride)
{
    const uint64_t cycles = EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_f162s4", cycles, dst, src, repeat, dstBlockStride,
        srcBlockStride, dstRepeatStride, srcRepeatStride);
}
inline void vconv_f162s4a(
    auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride,
    auto srcRepeatStride)
{
    const uint64_t cycles = EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_f162s4a", cycles, dst, src, repeat, dstBlockStride,
        srcBlockStride, dstRepeatStride, srcRepeatStride);
}
inline void vconv_f162s4c(
    auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride,
    auto srcRepeatStride)
{
    const uint64_t cycles = EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_f162s4c", cycles, dst, src, repeat, dstBlockStride,
        srcBlockStride, dstRepeatStride, srcRepeatStride);
}
inline void vconv_f162s4f(
    auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride,
    auto srcRepeatStride)
{
    const uint64_t cycles = EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_f162s4f", cycles, dst, src, repeat, dstBlockStride,
        srcBlockStride, dstRepeatStride, srcRepeatStride);
}
inline void vconv_f162s4r(
    auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride,
    auto srcRepeatStride)
{
    const uint64_t cycles = EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_f162s4r", cycles, dst, src, repeat, dstBlockStride,
        srcBlockStride, dstRepeatStride, srcRepeatStride);
}
inline void vconv_f162s4z(
    auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride,
    auto srcRepeatStride)
{
    const uint64_t cycles = EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_f162s4z", cycles, dst, src, repeat, dstBlockStride,
        srcBlockStride, dstRepeatStride, srcRepeatStride);
}
inline void vconv_f162u8a(
    auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride,
    auto srcRepeatStride)
{
    const uint64_t cycles = EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_f162u8a", cycles, dst, src, repeat, dstBlockStride,
        srcBlockStride, dstRepeatStride, srcRepeatStride);
}
inline void vconv_f162u8c(
    auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride,
    auto srcRepeatStride)
{
    const uint64_t cycles = EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_f162u8c", cycles, dst, src, repeat, dstBlockStride,
        srcBlockStride, dstRepeatStride, srcRepeatStride);
}
inline void vconv_f162u8f(
    auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride,
    auto srcRepeatStride)
{
    const uint64_t cycles = EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_f162u8f", cycles, dst, src, repeat, dstBlockStride,
        srcBlockStride, dstRepeatStride, srcRepeatStride);
}
inline void vconv_f162u8r(
    auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride,
    auto srcRepeatStride)
{
    const uint64_t cycles = EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_f162u8r", cycles, dst, src, repeat, dstBlockStride,
        srcBlockStride, dstRepeatStride, srcRepeatStride);
}
inline void vconv_f162u8z(
    auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride,
    auto srcRepeatStride)
{
    const uint64_t cycles = EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_f162u8z", cycles, dst, src, repeat, dstBlockStride,
        srcBlockStride, dstRepeatStride, srcRepeatStride);
}
inline void vconv_f322bf16a(
    auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride,
    auto srcRepeatStride)
{
    const uint64_t cycles =
        cce_costmodel_detail::EstimateExpFillCycles(repeat, src, cce_cycle_profiles::kVconvF322bf16aProfile);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_f322bf16a", cycles, dst, src, repeat, dstBlockStride,
        srcBlockStride, dstRepeatStride, srcRepeatStride);
}
inline void vconv_f322bf16c(
    auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride,
    auto srcRepeatStride)
{
    const uint64_t cycles =
        cce_costmodel_detail::EstimateExpFillCycles(repeat, src, cce_cycle_profiles::kVconvF322bf16cProfile);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_f322bf16c", cycles, dst, src, repeat, dstBlockStride,
        srcBlockStride, dstRepeatStride, srcRepeatStride);
}
inline void vconv_f322bf16f(
    auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride,
    auto srcRepeatStride)
{
    const uint64_t cycles =
        cce_costmodel_detail::EstimateExpFillCycles(repeat, src, cce_cycle_profiles::kVconvF322bf16fProfile);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_f322bf16f", cycles, dst, src, repeat, dstBlockStride,
        srcBlockStride, dstRepeatStride, srcRepeatStride);
}
inline void vconv_f322bf16r(
    auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride,
    auto srcRepeatStride)
{
    const uint64_t cycles =
        cce_costmodel_detail::EstimateExpFillCycles(repeat, src, cce_cycle_profiles::kVconvF322bf16rProfile);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_f322bf16r", cycles, dst, src, repeat, dstBlockStride,
        srcBlockStride, dstRepeatStride, srcRepeatStride);
}
inline void vconv_f322bf16z(
    auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride,
    auto srcRepeatStride)
{
    const uint64_t cycles =
        cce_costmodel_detail::EstimateExpFillCycles(repeat, src, cce_cycle_profiles::kVconvF322bf16zProfile);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_f322bf16z", cycles, dst, src, repeat, dstBlockStride,
        srcBlockStride, dstRepeatStride, srcRepeatStride);
}
inline void vconv_f322f16(
    auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride,
    auto srcRepeatStride)
{
    const uint64_t cycles =
        cce_costmodel_detail::EstimateExpFillCycles(repeat, src, cce_cycle_profiles::kVconvF322f16Profile);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_f322f16", cycles, dst, src, repeat, dstBlockStride,
        srcBlockStride, dstRepeatStride, srcRepeatStride);
}
inline void vconv_f322f16a(
    auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride,
    auto srcRepeatStride)
{
    const uint64_t cycles =
        cce_costmodel_detail::EstimateExpFillCycles(repeat, src, cce_cycle_profiles::kVconvF322f16aProfile);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_f322f16a", cycles, dst, src, repeat, dstBlockStride,
        srcBlockStride, dstRepeatStride, srcRepeatStride);
}
inline void vconv_f322f16c(
    auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride,
    auto srcRepeatStride)
{
    const uint64_t cycles =
        cce_costmodel_detail::EstimateExpFillCycles(repeat, src, cce_cycle_profiles::kVconvF322f16cProfile);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_f322f16c", cycles, dst, src, repeat, dstBlockStride,
        srcBlockStride, dstRepeatStride, srcRepeatStride);
}
inline void vconv_f322f16f(
    auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride,
    auto srcRepeatStride)
{
    const uint64_t cycles =
        cce_costmodel_detail::EstimateExpFillCycles(repeat, src, cce_cycle_profiles::kVconvF322f16fProfile);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_f322f16f", cycles, dst, src, repeat, dstBlockStride,
        srcBlockStride, dstRepeatStride, srcRepeatStride);
}
inline void vconv_f322f16o(
    auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride,
    auto srcRepeatStride)
{
    const uint64_t cycles =
        cce_costmodel_detail::EstimateLinearCycles(repeat, src, cce_cycle_profiles::kVconvF322f16oProfile);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_f322f16o", cycles, dst, src, repeat, dstBlockStride,
        srcBlockStride, dstRepeatStride, srcRepeatStride);
}
inline void vconv_f322f16r(
    auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride,
    auto srcRepeatStride)
{
    const uint64_t cycles =
        cce_costmodel_detail::EstimateExpFillCycles(repeat, src, cce_cycle_profiles::kVconvF322f16rProfile);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_f322f16r", cycles, dst, src, repeat, dstBlockStride,
        srcBlockStride, dstRepeatStride, srcRepeatStride);
}
inline void vconv_f322f16z(
    auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride,
    auto srcRepeatStride)
{
    const uint64_t cycles =
        cce_costmodel_detail::EstimateExpFillCycles(repeat, src, cce_cycle_profiles::kVconvF322f16zProfile);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_f322f16z", cycles, dst, src, repeat, dstBlockStride,
        srcBlockStride, dstRepeatStride, srcRepeatStride);
}
