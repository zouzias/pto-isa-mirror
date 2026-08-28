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
#include "pto/costmodel/trace.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

namespace pto::test::a5 {

namespace vf = ::pto::mocker::vf;

inline std::string JoinDiagnostics(const std::vector<std::string>& diagnostics)
{
    std::ostringstream output;
    for (const auto& diagnostic : diagnostics) {
        if (output.tellp() > 0) {
            output << "; ";
        }
        output << diagnostic;
    }
    return output.str();
}

inline bool IsAdapterIgnoredPredicateSetup(const vf::VfInst& inst)
{
    return inst.opName == "plt_b8" || inst.opName == "plt_b16" || inst.opName == "plt_b32" ||
           inst.opName == "pset_b8" || inst.opName == "pset_b16" || inst.opName == "pset_b32";
}

inline void CollectAdapterModeledInstructions(
    const std::vector<vf::VfNode>& nodes, std::vector<const vf::VfInst*>& instructions)
{
    for (const vf::VfNode& node : nodes) {
        if (vf::IsLoop(node)) {
            CollectAdapterModeledInstructions(vf::AsLoop(node).body, instructions);
        } else if (vf::IsInst(node) && !IsAdapterIgnoredPredicateSetup(vf::AsInst(node))) {
            instructions.push_back(&vf::AsInst(node));
        }
    }
}

inline std::vector<const vf::VfInst*> AdapterModeledInstructions(const std::vector<vf::VfNode>& nodes)
{
    std::vector<const vf::VfInst*> instructions;
    CollectAdapterModeledInstructions(nodes, instructions);
    return instructions;
}

inline uint32_t CountAdapterIgnoredPredicateSetup(const std::vector<vf::VfNode>& nodes)
{
    uint32_t count = 0;
    for (const vf::VfNode& node : nodes) {
        if (vf::IsLoop(node)) {
            count += CountAdapterIgnoredPredicateSetup(vf::AsLoop(node).body);
        } else if (vf::IsInst(node) && IsAdapterIgnoredPredicateSetup(vf::AsInst(node))) {
            ++count;
        }
    }
    return count;
}

inline void ExpectLastVecTileOp(const std::vector<std::string>& expectedBody, uint64_t expectedRepeat)
{
    const uint64_t cycles = ::pto::mocker::GetLastPtoInstrCycles();
    const auto& trace = ::pto::mocker::GetTrace();
    ASSERT_FALSE(trace.executed_pto.empty());

    const auto& record = trace.executed_pto.back();
    EXPECT_GT(cycles, 0U);
    ASSERT_EQ(record.vf_infos.size(), 1U);

    const vf::VfInfo& info = record.vf_infos[0];
    ASSERT_EQ(info.tree.size(), 1U);
    ASSERT_TRUE(vf::IsLoop(info.tree[0]));
    EXPECT_EQ(vf::LoopDepth(info.tree), 1U);

    const vf::VfLoop& loop = vf::AsLoop(info.tree[0]);
    EXPECT_EQ(loop.count, expectedRepeat);
    const auto modeledInstructions = AdapterModeledInstructions(loop.body);
    std::vector<std::string> modeledNames;
    modeledNames.reserve(modeledInstructions.size());
    for (const vf::VfInst* inst : modeledInstructions)
        modeledNames.push_back(inst->opName);
    EXPECT_EQ(modeledNames, expectedBody);
    EXPECT_EQ(modeledInstructions.size(), expectedBody.size());
}

inline void ExpectLastBinaryVecTileOp(const std::vector<std::string>& expectedBody, uint64_t expectedRepeat)
{
    ExpectLastVecTileOp(expectedBody, expectedRepeat);

    const auto& trace = ::pto::mocker::GetTrace();
    ASSERT_FALSE(trace.executed_pto.empty());
    const auto& record = trace.executed_pto.back();
    ASSERT_EQ(record.vf_infos.size(), 1U);
    const vf::VfInfo& info = record.vf_infos[0];
    ASSERT_EQ(info.tree.size(), 1U);
    ASSERT_TRUE(vf::IsLoop(info.tree[0]));

    const vf::VfLoop& loop = vf::AsLoop(info.tree[0]);
    const auto modeledInstructions = AdapterModeledInstructions(loop.body);
    ASSERT_GE(modeledInstructions.size(), 4U);
    const vf::VfInst& load0 = *modeledInstructions[0];
    const vf::VfInst& load1 = *modeledInstructions[1];
    const vf::VfInst& op = *modeledInstructions[2];
    const vf::VfInst& store = *modeledInstructions[3];

    EXPECT_EQ(load0.dst.size(), 1U);
    EXPECT_EQ(load0.src.size(), 1U);
    EXPECT_EQ(load0.dst[0].location, vf::MemLocation::PhyRegister);
    EXPECT_EQ(load0.src[0].location, vf::MemLocation::UB);
    EXPECT_EQ(load1.dst.size(), 1U);
    EXPECT_EQ(load1.src.size(), 1U);
    EXPECT_EQ(load1.dst[0].location, vf::MemLocation::PhyRegister);
    EXPECT_EQ(load1.src[0].location, vf::MemLocation::UB);
    EXPECT_EQ(op.dst.size(), 1U);
    ASSERT_GE(op.src.size(), 2U);
    EXPECT_EQ(op.src[0].name, load0.dst[0].name);
    EXPECT_EQ(op.src[1].name, load1.dst[0].name);
    EXPECT_EQ(store.dst.size(), 1U);
    EXPECT_EQ(store.src.size(), 1U);
    EXPECT_EQ(store.dst[0].location, vf::MemLocation::UB);
    EXPECT_EQ(store.src[0].name, op.dst[0].name);
}

inline void ExpectLastVfSimHit()
{
    const auto& trace = ::pto::mocker::GetTrace();
    ASSERT_FALSE(trace.executed_pto.empty());
    const auto& prediction = trace.executed_pto.back().vfPrediction;
    EXPECT_EQ(prediction.status, vf::VfPredictionStatus::VF_SIM_HIT) << JoinDiagnostics(prediction.diagnostics);
    EXPECT_GT(prediction.vfSimHitCount, 0U);
    EXPECT_EQ(prediction.fallbackCount, 0U);
    uint32_t expectedIgnoredInstructionCount = 0;
    for (const vf::VfInfo& info : trace.executed_pto.back().vf_infos)
        expectedIgnoredInstructionCount += CountAdapterIgnoredPredicateSetup(info.tree);
    EXPECT_EQ(prediction.ignoredInstructionCount, expectedIgnoredInstructionCount);

    const auto& perfRecords = ::pto::perf_sim::PtoRecorder::Get();
    ASSERT_FALSE(perfRecords.empty());
    const auto& perfPrediction = perfRecords.back();
    EXPECT_EQ(perfPrediction.predictionStatus, "VfSimHit");
    EXPECT_GT(perfPrediction.vfSimHitCount, 0U);
    EXPECT_EQ(perfPrediction.fallbackCount, 0U);
    EXPECT_EQ(perfPrediction.ignoredInstructionCount, prediction.ignoredInstructionCount);
}

} // namespace pto::test::a5
