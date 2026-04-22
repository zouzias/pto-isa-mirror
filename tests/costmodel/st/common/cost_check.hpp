/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COSTMODEL_ST_COST_CHECK_HPP
#define PTO_COSTMODEL_ST_COST_CHECK_HPP

#define NO_PROFILING 0.114514f

#include <cstdlib>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <gtest/gtest.h>
#include "pto/costmodel/trace.hpp"

inline int GetCostmodelLogLevel()
{
    const char *raw = std::getenv("PTO_COSTMODEL_LOG_LEVEL");
    if (raw == nullptr || *raw == '\0') {
        return 0;
    }

    char *end = nullptr;
    long parsed = std::strtol(raw, &end, 10);
    if (end == raw) {
        return 0;
    }
    if (parsed < 0) {
        return 0;
    }
    if (parsed > 2) {
        return 2;
    }
    return static_cast<int>(parsed);
}

inline std::string GetCurrentCostmodelTestName()
{
    const auto *testInfo = ::testing::UnitTest::GetInstance()->current_test_info();
    if (testInfo == nullptr) {
        return "<unknown>";
    }
    return std::string(testInfo->test_suite_name()) + "." + testInfo->name();
}

inline void EmitCostmodelCycleLine(float actual, float expected, float precision, float accuracy)
{
    std::ostringstream oss;
    oss << "[COSTMODEL] " << GetCurrentCostmodelTestName() << " actual=" << actual << " expected=" << expected
        << " precision=" << precision;
    std::cout << oss.str() << std::endl;
}

inline void EmitCostmodelTraceLine(const std::string &line)
{
    std::cout << "[TRACE] " << GetCurrentCostmodelTestName() << " " << line << std::endl;
}

inline void EmitCostmodelTraceBlockLine(const std::string &line)
{
    std::cout << line << std::endl;
}

inline std::string FormatCostmodelTraceArgs(const std::vector<::pto::mocker::Arg> &args)
{
    if (args.empty()) {
        return "[]";
    }

    std::ostringstream oss;
    oss << "[";
    for (std::size_t i = 0; i < args.size(); ++i) {
        if (i != 0) {
            oss << ", ";
        }
        oss << "0x" << std::hex << std::uppercase << args[i] << std::dec;
    }
    oss << "]";
    return oss.str();
}

inline void EmitLatestCostmodelTrace()
{
    const auto &trace = ::pto::mocker::GetTrace();
    EmitCostmodelTraceBlockLine("[TRACE] " + GetCurrentCostmodelTestName());
    if (trace.executed_pto.empty()) {
        EmitCostmodelTraceBlockLine("  trace: <empty>");
        return;
    }

    const auto &ptoRecord = trace.executed_pto.back();
    {
        std::ostringstream oss;
        oss << "  pto: " << ptoRecord.name;
        EmitCostmodelTraceBlockLine(oss.str());
    }
    EmitCostmodelTraceBlockLine("  total_cycles: " + std::to_string(ptoRecord.total_cycles));
    EmitCostmodelTraceBlockLine("  cce_calls: " + std::to_string(ptoRecord.cce_calls.size()));

    for (std::size_t i = 0; i < ptoRecord.cce_calls.size(); ++i) {
        const auto &call = ptoRecord.cce_calls[i];
        std::ostringstream oss;
        oss << "    [" << i << "] name=" << call.name << " cycles=" << call.cycles
            << " args=" << FormatCostmodelTraceArgs(call.args);
        EmitCostmodelTraceBlockLine(oss.str());
    }
}

// Compare the cycle count stored in the most recently executed PTO trace record
// against an expected `profiling` value. The check passes when relative
// precision `1 - |profiling - actual| / profiling` is at least `accuracy`.
#define EXPECT_CYCLE_NEAR(profiling, accuracy)                                                                      \
    do {                                                                                                            \
        float _pto_actual = static_cast<float>(::pto::mocker::GetLastPtoInstrCycles());                             \
        float _pto_expected = static_cast<float>(profiling);                                                        \
        float _pto_precision = (_pto_expected == 0.0f) ?                                                            \
                                   ((_pto_actual == 0.0f) ? 1.0f : 0.0f) :                                          \
                                   std::max(0.0f, (1.0f - std::fabs(_pto_expected - _pto_actual) / _pto_expected)); \
        int _pto_log_level = GetCostmodelLogLevel();                                                                \
        if (_pto_log_level >= 1) {                                                                                  \
            EmitCostmodelCycleLine(_pto_actual, _pto_expected, _pto_precision, static_cast<float>(accuracy));       \
        }                                                                                                           \
        if (_pto_log_level >= 2) {                                                                                  \
            EmitLatestCostmodelTrace();                                                                             \
        }                                                                                                           \
        EXPECT_GE(_pto_precision, static_cast<float>(accuracy));                                                    \
    } while (0)

#endif // PTO_COSTMODEL_ST_COST_CHECK_HPP
