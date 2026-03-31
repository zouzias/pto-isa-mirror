/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_MOCKER_EVALUATOR_TRACE_EVALUATOR_HPP
#define PTO_MOCKER_EVALUATOR_TRACE_EVALUATOR_HPP

#include <string>
#include <utility>

#include <pto/costmodel/arch_config.hpp>
#include <pto/costmodel/evaluator/cce_evaluator.hpp>
#include <pto/costmodel/evaluator/types.hpp>
#include <pto/costmodel/trace.hpp>

namespace pto::mocker::evaluator {

inline PtoLatencyRecord EvaluatePtoInstr(const PtoInstrRecord &instr, const ArchConfig &arch)
{
    PtoLatencyRecord record;
    record.name = instr.name;

    for (const auto &call : instr.cce_calls) {
        CceLatencyRecord call_record = EvaluateCceCallRecord(call, arch);
        record.total_cycles += call_record.cycles;
        record.supported = record.supported && call_record.supported;
        record.cce_calls.push_back(std::move(call_record));
    }

    return record;
}

inline TraceLatencyReport EvaluateTrace(const TraceState &trace, const ArchConfig &arch)
{
    TraceLatencyReport report;
    report.arch_name = std::string(arch.arch_name);

    for (const auto &instr : trace.executed_pto) {
        PtoLatencyRecord pto_record = EvaluatePtoInstr(instr, arch);
        report.total_cycles += pto_record.total_cycles;
        report.supported = report.supported && pto_record.supported;
        report.executed_pto.push_back(std::move(pto_record));
    }

    for (const auto &call : trace.raw_cce_calls) {
        CceLatencyRecord call_record = EvaluateCceCallRecord(call, arch);
        report.total_cycles += call_record.cycles;
        report.supported = report.supported && call_record.supported;
        report.raw_cce_calls.push_back(std::move(call_record));
    }

    return report;
}

inline TraceLatencyReport EvaluateTrace(const TraceState &trace)
{
    return EvaluateTrace(trace, GetDefaultArchConfig());
}

} // namespace pto::mocker::evaluator

#endif
