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
#include <sstream>
#include <string>
#include <vector>

namespace pto::test::a5 {

inline void ExpectUnsupportedVfTileOp(const std::vector<std::string>& expectedBody, uint64_t expectedRepeat)
{
    (void)expectedBody;
    (void)expectedRepeat;

    const auto& trace = ::pto::mocker::GetTrace();
    ASSERT_FALSE(trace.executed_pto.empty());
    EXPECT_EQ(trace.executed_pto.back().total_cycles, 0U);

    const auto& records = ::pto::perf_sim::PtoRecorder::Get();
    ASSERT_FALSE(records.empty());
    const auto& record = records.back();
    EXPECT_EQ(record.stage, ::pto::perf_sim::PipeStage::Vector);
    EXPECT_EQ(record.costmodel_status, ::pto::perf_sim::CostModelStatus::Unsupported);
    EXPECT_EQ(record.estimated_cycles, 0U);
    EXPECT_FALSE(record.costmodel_diagnostic.empty());

    const auto report = ::pto::perf_sim::PerfSimReporter().Run("a5_unsupported_vf_test");
    EXPECT_EQ(report.costmodel_status, ::pto::perf_sim::CostModelStatus::Unsupported);
    EXPECT_FALSE(report.costmodel_diagnostics.empty());
    EXPECT_TRUE(report.timeline.events.empty());

    std::ostringstream output;
    ::pto::perf_sim::PerfSimReporter::PrintText(report, output);
    EXPECT_NE(output.str().find("Total cycles : N/A (unsupported)"), std::string::npos);
}

inline void ExpectLastVecTileOp(const std::vector<std::string>& expectedBody, uint64_t expectedRepeat)
{
    ExpectUnsupportedVfTileOp(expectedBody, expectedRepeat);
}

inline void ExpectLastBinaryVecTileOp(const std::vector<std::string>& expectedBody, uint64_t expectedRepeat)
{
    ExpectUnsupportedVfTileOp(expectedBody, expectedRepeat);
}

} // namespace pto::test::a5
