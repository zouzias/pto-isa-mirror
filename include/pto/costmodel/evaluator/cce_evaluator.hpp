/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_MOCKER_EVALUATOR_CCE_EVALUATOR_HPP
#define PTO_MOCKER_EVALUATOR_CCE_EVALUATOR_HPP

#include <string>
#include <string_view>
#include <utility>

#include <pto/costmodel/arch_config.hpp>
#include <pto/costmodel/evaluator/arg_reader.hpp>
#include <pto/costmodel/evaluator/types.hpp>

namespace pto::mocker::evaluator {

inline CycleEstimate MakeUnsupportedEstimate(std::string note)
{
    return {0, false, std::move(note)};
}

inline CycleEstimate MakeMissingArgEstimate(const CceLatencyRule &rule, std::string_view missing_args)
{
    std::string note = "missing ";
    note += missing_args;
    note += " for ";
    note += GetCoreTypeName(rule.core_type);
    note += " rule";
    return MakeUnsupportedEstimate(std::move(note));
}

inline CycleEstimate EvaluateFixedRule(const CceLatencyRule &rule)
{
    return {rule.fixed_cycles, true, std::string(rule.note)};
}

inline CycleEstimate EvaluateRepeatRule(const CceCallRecord &call, const CceLatencyRule &rule)
{
    uint64_t repeats = 0;
    if (!TryGetRepeatArg(call, repeats)) {
        return MakeMissingArgEstimate(rule, "repeats/repeat");
    }

    return {rule.startup_cycles + repeats * rule.per_repeat_cycles, true, std::string(rule.note)};
}

inline CycleEstimate EvaluateBurstLenRule(const CceCallRecord &call, const CceLatencyRule &rule)
{
    uint64_t nBurst = 0;
    uint64_t lenBurst = 0;
    if (!TryGetArg(call, "nBurst", nBurst) || !TryGetArg(call, "lenBurst", lenBurst)) {
        return MakeMissingArgEstimate(rule, "nBurst/lenBurst");
    }

    return {rule.startup_cycles + nBurst * rule.per_burst_cycles + lenBurst * rule.per_len_burst_cycles,
            true,
            std::string(rule.note)};
}

inline CycleEstimate EvaluateNd2NzRule(const CceCallRecord &call, const CceLatencyRule &rule)
{
    uint64_t ndNum = 0;
    uint64_t nValue = 0;
    uint64_t dValue = 0;
    if (!TryGetArg(call, "ndNum", ndNum) || !TryGetArg(call, "nValue", nValue) || !TryGetArg(call, "dValue", dValue)) {
        return MakeMissingArgEstimate(rule, "ndNum/nValue/dValue");
    }

    const uint64_t cycles =
        rule.startup_cycles + ndNum * (rule.per_burst_cycles + nValue * rule.per_repeat_cycles +
                                       dValue * rule.per_len_burst_cycles);
    return {cycles, true, std::string(rule.note)};
}

inline uint64_t CeilDivU64(uint64_t value, uint64_t divisor)
{
    return (value + divisor - 1) / divisor;
}

inline CycleEstimate EvaluateMadRule(const CceCallRecord &call, const CceLatencyRule &rule)
{
    uint64_t m = 0;
    uint64_t k = 0;
    uint64_t n = 0;
    if (!TryGetArg(call, "m", m) || !TryGetArg(call, "k", k) || !TryGetArg(call, "n", n)) {
        return MakeMissingArgEstimate(rule, "m/k/n");
    }

    const uint64_t m_tiles = CeilDivU64(m, 16);
    const uint64_t k_tiles = CeilDivU64(k, 16);
    const uint64_t n_tiles = CeilDivU64(n, 16);
    const uint64_t tile_ops = m_tiles * k_tiles * n_tiles;

    return {rule.startup_cycles + tile_ops * rule.per_repeat_cycles, true, std::string(rule.note)};
}

inline CycleEstimate EvaluateCceCall(const CceCallRecord &call, const ArchConfig &arch)
{
    const CceLatencyRule *rule = FindRule(arch, call.name);
    if (rule == nullptr) {
        return MakeUnsupportedEstimate(
            std::string("unsupported on arch ") + std::string(arch.arch_name) + ": " + call.name);
    }

    switch (rule->kind) {
        case LatencyRuleKind::Fixed:
            return EvaluateFixedRule(*rule);
        case LatencyRuleKind::Repeat:
            return EvaluateRepeatRule(call, *rule);
        case LatencyRuleKind::BurstLen:
            return EvaluateBurstLenRule(call, *rule);
        case LatencyRuleKind::Nd2Nz:
            return EvaluateNd2NzRule(call, *rule);
        case LatencyRuleKind::Mad:
            return EvaluateMadRule(call, *rule);
    }

    return MakeUnsupportedEstimate("invalid latency rule");
}

inline CycleEstimate EvaluateCceCall(const CceCallRecord &call)
{
    return EvaluateCceCall(call, GetDefaultArchConfig());
}

inline CceLatencyRecord EvaluateCceCallRecord(const CceCallRecord &call, const ArchConfig &arch)
{
    const CycleEstimate estimate = EvaluateCceCall(call, arch);
    return {call.name, estimate.cycles, estimate.supported, estimate.note};
}

inline CceLatencyRecord EvaluateCceCallRecord(const CceCallRecord &call)
{
    return EvaluateCceCallRecord(call, GetDefaultArchConfig());
}

} // namespace pto::mocker::evaluator

#endif
