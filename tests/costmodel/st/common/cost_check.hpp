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

#include <cmath>
#include <iostream>
#include <gtest/gtest.h>

#include "pto/costmodel/trace.hpp"
#include "pto/costmodel/evaluator/trace_evaluator.hpp"

namespace pto::mocker {

// Definition of the function forward-declared in pto_tile.hpp.
// Tile::GetCycle() calls this.
inline float GetLastPtoInstrCycles()
{
    const auto &trace = GetTrace();
    if (trace.executed_pto.empty()) {
        return 0.0f;
    }
    auto report = evaluator::EvaluatePtoInstr(
        trace.executed_pto.back(), evaluator::GetDefaultArchConfig());
    return static_cast<float>(report.total_cycles);
}

} // namespace pto::mocker

// Compare the cycle count of the last executed PTO instruction against an
// expected `profiling` value. The check passes when relative precision
// `1 - |profiling - actual| / profiling` is at least `accuracy`.
#define EXPECT_CYCLE_NEAR(tile, profiling, accuracy)                                                                   \
    do {                                                                                                               \
        float _pto_actual = ::pto::mocker::GetLastPtoInstrCycles();                                                    \
        float _pto_expected = static_cast<float>(profiling);                                                           \
        float _pto_precision =                                                                                         \
            (_pto_expected == 0.0f) ? ((_pto_actual == 0.0f) ? 1.0f : 0.0f)                                            \
                                    : (1.0f - std::fabs(_pto_expected - _pto_actual) / _pto_expected);                 \
        std::cout << "[CYCLE] " << ::testing::UnitTest::GetInstance()->current_test_info()->test_suite_name() << "."   \
                  << ::testing::UnitTest::GetInstance()->current_test_info()->name()                                   \
                  << " actual=" << _pto_actual << " expected=" << _pto_expected                                        \
                  << " precision=" << _pto_precision << " accuracy=" << static_cast<float>(accuracy) << std::endl;     \
        EXPECT_GE(_pto_precision, static_cast<float>(accuracy));                                                       \
    } while (0)

#endif // PTO_COSTMODEL_ST_COST_CHECK_HPP
