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

#include "pto/costmodel/perf_sim/recorder.hpp"
#include "pto/costmodel/perf_sim/reporter.hpp"
#include "pto/costmodel/trace.hpp"

#include <gtest/gtest.h>

#include <cstdint>

namespace pto::test::a5 {

inline void ExpectSupportedVfTileOp(const char* opcode, uint64_t expectedCycles)
{
    const auto& trace = ::pto::mocker::GetTrace();
    ASSERT_FALSE(trace.executed_pto.empty());
    const auto& tileOp = trace.executed_pto.back();
    EXPECT_EQ(tileOp.name, opcode);
    EXPECT_EQ(tileOp.total_cycles, expectedCycles);
    EXPECT_TRUE(tileOp.cce_calls.empty());

    const auto& records = ::pto::perf_sim::PtoRecorder::Get();
    ASSERT_FALSE(records.empty());
    const auto& record = records.back();
    EXPECT_EQ(record.opcode, opcode);
    EXPECT_EQ(record.stage, ::pto::perf_sim::PipeStage::Vector);
    EXPECT_EQ(record.estimated_cycles, expectedCycles);

    const auto report = ::pto::perf_sim::PerfSimReporter().Run("a5_formula_vf_test");
    EXPECT_FALSE(report.timeline.events.empty());
}

} // namespace pto::test::a5
