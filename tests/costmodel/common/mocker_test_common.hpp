/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_MOCKER_TEST_COMMON_HPP
#define PTO_MOCKER_TEST_COMMON_HPP

#include <array>
#include <cstddef>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

#include <pto/costmodel/evaluator/trace_evaluator.hpp>
#include <pto/costmodel/trace.hpp>

namespace pto::mocker::test {

constexpr int kRows = 16;
constexpr int kCols = 16;
constexpr std::size_t kElemCount = static_cast<std::size_t>(kRows) * static_cast<std::size_t>(kCols);

template <typename T>
std::array<T, kElemCount> MakeRamp(T start = T{})
{
    std::array<T, kElemCount> values{};
    for (std::size_t i = 0; i < values.size(); ++i) {
        values[i] = static_cast<T>(start + static_cast<T>(i));
    }
    return values;
}

inline void PrintTraceArgs(const std::vector<TraceArgRecord> &args)
{
    if (args.empty()) {
        return;
    }

    std::cout << " (";
    for (std::size_t i = 0; i < args.size(); ++i) {
        if (i != 0) {
            std::cout << ", ";
        }
        std::cout << args[i].name << "=0x" << std::hex << args[i].value << std::dec;
    }
    std::cout << ')';
}

inline void PrintCalls(std::string_view label, const std::vector<CceCallRecord> &calls, std::string_view indent = "  ")
{
    std::cout << label << ':';
    if (calls.empty()) {
        std::cout << " <empty>\n";
        return;
    }

    std::cout << '\n';
    for (std::size_t i = 0; i < calls.size(); ++i) {
        std::cout << indent << '[' << i << "] " << calls[i].name;
        PrintTraceArgs(calls[i].params);
        std::cout << '\n';
    }
}

inline void PrintLatencyCalls(std::string_view label, const std::vector<evaluator::CceLatencyRecord> &calls,
                              std::string_view indent = "  ")
{
    std::cout << label << ':';
    if (calls.empty()) {
        std::cout << " <empty>\n";
        return;
    }

    std::cout << '\n';
    for (std::size_t i = 0; i < calls.size(); ++i) {
        const auto &call = calls[i];
        std::cout << indent << '[' << i << "] " << call.name << " cycles=" << call.cycles;
        if (!call.supported) {
            std::cout << " [unsupported]";
        }
        if (!call.note.empty()) {
            std::cout << " (" << call.note << ')';
        }
        std::cout << '\n';
    }
}

inline void PrintLatencyReport(const evaluator::TraceLatencyReport &report)
{
    std::cout << "latency[" << report.arch_name << "] total_cycles=" << report.total_cycles;
    if (!report.supported) {
        std::cout << " [partial]";
    }
    std::cout << '\n';

    std::cout << "executed_pto_latency:";
    if (report.executed_pto.empty()) {
        std::cout << " <empty>\n";
    } else {
        std::cout << '\n';
        for (std::size_t i = 0; i < report.executed_pto.size(); ++i) {
            const auto &instr = report.executed_pto[i];
            std::cout << "  [" << i << "] " << instr.name << " total_cycles=" << instr.total_cycles;
            if (!instr.supported) {
                std::cout << " [partial]";
            }
            std::cout << '\n';
            PrintLatencyCalls("    cce_latency", instr.cce_calls, "      ");
        }
    }

    PrintLatencyCalls("raw_cce_latency", report.raw_cce_calls);
}

inline void PrintTrace(std::string_view label, const TraceState &trace)
{
    std::cout << "== " << label << " ==\n";
    std::cout << "executed_pto:";
    if (trace.executed_pto.empty()) {
        std::cout << " <empty>\n";
    } else {
        std::cout << '\n';
        for (std::size_t i = 0; i < trace.executed_pto.size(); ++i) {
            const auto &instr = trace.executed_pto[i];
            std::cout << "  [" << i << "] " << instr.name << '\n';
            PrintCalls("    cce_calls", instr.cce_calls, "      ");
        }
    }

    PrintCalls("raw_cce_calls", trace.raw_cce_calls);
    PrintLatencyReport(evaluator::EvaluateTrace(trace));
}

} // namespace pto::mocker::test

#endif
