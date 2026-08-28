/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the LICENSE.
*/
#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include <gtest/gtest.h>

#include "pto/cpu/Hifloat8.hpp"
#include "pto/costmodel/a5/cce_costmodel/a5_vf_stub.hpp"
#include "pto/costmodel/vfsim/pto_adapter/pto_canonical_lowering.hpp"
#include "pto/costmodel/vfsim/pto_adapter/vfsim_cost_model.hpp"

#include "native/ParamDB.h"
#include "native/OOO.h"
#include "native/SimulatorRunner.h"

namespace {

namespace ptoVf = ::pto::mocker::vf;

constexpr uint64_t LOOP_COUNT = 8;

ptoVf::MemInfo makePtoValue(std::string name, ptoVf::MemLocation location)
{
    return ptoVf::MemInfo{std::move(name), location, "fp32"};
}

ptoVf::VfNode makePtoInst(std::string op, std::vector<ptoVf::MemInfo> dst, std::vector<ptoVf::MemInfo> src)
{
    return ptoVf::MakeInst(ptoVf::VfInst{std::move(op), std::move(dst), std::move(src)});
}

ptoVf::VfInfo buildPtoVfInfo(bool includePredicateSetup = false)
{
    const auto v0 = makePtoValue("v0", ptoVf::MemLocation::PhyRegister);
    const auto v1 = makePtoValue("v1", ptoVf::MemLocation::PhyRegister);
    const auto v2 = makePtoValue("v2", ptoVf::MemLocation::PhyRegister);
    const auto mem0 = makePtoValue("mem0", ptoVf::MemLocation::UB);
    const auto mem1 = makePtoValue("mem1", ptoVf::MemLocation::UB);
    const auto mem2 = makePtoValue("mem2", ptoVf::MemLocation::UB);

    std::vector<ptoVf::VfNode> body;
    body.push_back(makePtoInst("vlds", {v0}, {mem0}));
    body.push_back(makePtoInst("vlds", {v1}, {mem1}));
    if (includePredicateSetup)
        body.push_back(makePtoInst("plt_b32", {}, {}));
    body.push_back(makePtoInst("vadd", {v2}, {v0, v1}));
    body.push_back(makePtoInst("vsts", {mem2}, {v2}));
    return ptoVf::VfInfo{"TADD", "1x512", {ptoVf::MakeLoop(LOOP_COUNT, std::move(body))}};
}

std::string readText(const std::filesystem::path& path)
{
    std::ifstream input(path);
    std::ostringstream buffer;
    buffer << input.rdbuf();
    return buffer.str();
}

int64_t cycleForInstId(const std::string& jsonLines, int64_t instId)
{
    const std::string instructionNeedle = "\"inst_id\":" + std::to_string(instId);
    std::istringstream lines(jsonLines);
    std::string line;
    while (std::getline(lines, line)) {
        if (line.find(instructionNeedle) == std::string::npos)
            continue;
        constexpr const char* CYCLE_NEEDLE = "\"cy\":";
        const auto position = line.find(CYCLE_NEEDLE);
        if (position == std::string::npos)
            break;
        const auto start = position + std::char_traits<char>::length(CYCLE_NEEDLE);
        const auto end = line.find_first_of(",}", start);
        return std::stoll(line.substr(start, end - start));
    }
    throw std::runtime_error("instruction not found in native start log: " + std::to_string(instId));
}

template <typename Invoke>
ptoVf::VfInfo capturePtoVfInfo(Invoke&& invoke)
{
    ptoVf::trace::Reset();
    ptoVf::capture::ResetOperands();
    ptoVf::trace::Arm(true);
    std::forward<Invoke>(invoke)();
    ptoVf::trace::Arm(false);
    auto result = ptoVf::trace::BuildVfInfo("CAPTURE", "");
    if (!result.ok)
        throw std::runtime_error("failed to build capture fixture: " + result.err);
    return std::move(result.info);
}

template <typename Invoke>
ptoVf::VfInst capturePtoInstruction(Invoke&& invoke)
{
    ptoVf::VfInfo info = capturePtoVfInfo(std::forward<Invoke>(invoke));
    if (info.tree.size() != 1 || !ptoVf::IsInst(info.tree.front()))
        throw std::runtime_error("capture fixture did not produce exactly one instruction");
    return ptoVf::AsInst(info.tree.front());
}

TEST(VfSimAdapterGolden, MatchesDirectNativePrediction)
{
    const auto adapterResult = ptoVf::predictVfCyclesWithVfSim({buildPtoVfInfo()});

    const vfsim::ParamDb db(std::filesystem::path(PTO_VFSIM_TEST_SOURCE_ROOT));
    const ptoVf::PtoCanonicalLoweringResult lowering = ptoVf::lowerPtoVfToCanonical(buildPtoVfInfo());
    const auto nativeResult = vfsim::runCanonicalVfInfo(lowering.program, db);
    const uint64_t nativeCycles = static_cast<uint64_t>(std::max<int64_t>(0, nativeResult.vfEndCycle));

    EXPECT_EQ(adapterResult.status, ptoVf::VfPredictionStatus::VF_SIM_HIT);
    EXPECT_EQ(adapterResult.vfSimHitCount, 1U);
    EXPECT_EQ(adapterResult.fallbackCount, 0U);
    EXPECT_EQ(adapterResult.ignoredInstructionCount, 0U);
    EXPECT_TRUE(adapterResult.diagnostics.empty());
    EXPECT_EQ(adapterResult.cycles, nativeCycles);
}

TEST(VfSimAdapterGolden, SharesUbIssueSlotsAcrossLoadsAndStores)
{
    const vfsim::ParamDb db(std::filesystem::path(PTO_VFSIM_TEST_SOURCE_ROOT));
    EXPECT_EQ(db.uarch().ubSlots, 2);
    EXPECT_EQ(db.uarch().lsuStorePriorityPregThreshold, 1);

    const auto runCase = [&](int64_t vregNum, const std::string& suffix) {
        vfsim::UarchConfig uarch = db.uarch();
        uarch.vregNum = vregNum;
        uarch.loadPorts = 2;
        uarch.storePorts = 1;
        uarch.ubSlots = 2;
        uarch.lsuStorePriorityPregThreshold = 1;
        vfsim::OooCoreMainline core(uarch, db, "fp32");

        vfsim::DynamicInst producer;
        producer.type = "inst";
        producer.instId = 9200;
        producer.streamSeq = 0;
        producer.op = "VADD";
        producer.form = "fp32";
        producer.dst = {"v0"};
        core.accept(producer);
        for (int cycle = 0; cycle < 40; ++cycle)
            core.step();

        vfsim::DynamicInst store;
        store.type = "inst";
        store.instId = 9201;
        store.streamSeq = 1;
        store.op = "VSTS";
        store.form = "fp32";
        store.src = {"v0"};
        store.dst = {"mem0"};

        vfsim::DynamicInst firstLoad;
        firstLoad.type = "inst";
        firstLoad.instId = 9202;
        firstLoad.streamSeq = 2;
        firstLoad.op = "VLDS";
        firstLoad.form = "fp32";
        firstLoad.src = {"mem1"};
        firstLoad.dst = {"v1"};

        vfsim::DynamicInst secondLoad = firstLoad;
        secondLoad.instId = 9203;
        secondLoad.streamSeq = 3;
        secondLoad.dst = {"v2"};

        core.accept(store);
        core.accept(firstLoad);
        core.accept(secondLoad);
        for (int cycle = 0; cycle < 20; ++cycle)
            core.step();

        const auto outputRoot = std::filesystem::temp_directory_path() / ("pto_vfsim_ub_slot_test_" + suffix);
        std::filesystem::create_directories(outputRoot);
        core.dumpSimpleLogs((outputRoot / "starts.jsonl").string(), (outputRoot / "done.jsonl").string());
        const std::string starts = readText(outputRoot / "starts.jsonl");
        return std::make_tuple(
            cycleForInstId(starts, 9201), cycleForInstId(starts, 9202), cycleForInstId(starts, 9203));
    };

    const auto relaxed = runCase(16, "relaxed");
    EXPECT_EQ(std::get<1>(relaxed), std::get<2>(relaxed));
    EXPECT_GT(std::get<0>(relaxed), std::get<1>(relaxed));

    const auto pressured = runCase(1, "pressured");
    EXPECT_EQ(std::get<0>(pressured), std::get<1>(pressured));
    EXPECT_GT(std::get<2>(pressured), std::get<1>(pressured));
}

TEST(VfSimAdapterGolden, PtoTraceLowersDirectlyToCanonicalContract)
{
    const ptoVf::PtoCanonicalLoweringResult lowering = ptoVf::lowerPtoVfToCanonical(buildPtoVfInfo());
    const auto adapter = lowering.program.source.find("adapter");

    ASSERT_NE(adapter, lowering.program.source.end());
    ASSERT_TRUE(std::holds_alternative<std::string>(adapter->second));
    EXPECT_EQ(std::get<std::string>(adapter->second), "pto_vf_info");
    EXPECT_EQ(lowering.ignoredInstructionCount, 0U);
    EXPECT_TRUE(vfsim::validateCanonicalVfInfo(lowering.program).ok());
    ASSERT_EQ(lowering.program.context.size(), 1U);
    EXPECT_TRUE(
        std::holds_alternative<std::shared_ptr<const vfsim::CanonicalLoop>>(lowering.program.context.front().payload));
}

TEST(VfSimAdapterGolden, FiltersPredicateSetupAsAnObservableApproximation)
{
    const auto exactResult = ptoVf::predictVfCyclesWithVfSim({buildPtoVfInfo()});
    const auto approximateResult = ptoVf::predictVfCyclesWithVfSim({buildPtoVfInfo(true)});

    EXPECT_EQ(approximateResult.status, ptoVf::VfPredictionStatus::VF_SIM_HIT);
    EXPECT_EQ(approximateResult.vfSimHitCount, 1U);
    EXPECT_EQ(approximateResult.fallbackCount, 0U);
    EXPECT_EQ(approximateResult.ignoredInstructionCount, 1U);
    EXPECT_FALSE(approximateResult.diagnostics.empty());
    EXPECT_EQ(approximateResult.cycles, exactResult.cycles);
}

TEST(VfSimAdapterGolden, ReportsInvalidTraceInsteadOfSilentFallback)
{
    ptoVf::VfInfo invalid;
    invalid.op = "TADD";
    invalid.tree.push_back(ptoVf::MakeInst("vadd"));

    const auto result = ptoVf::predictVfCyclesWithVfSim({invalid});

    EXPECT_EQ(result.status, ptoVf::VfPredictionStatus::INVALID_TRACE);
    EXPECT_EQ(result.vfSimHitCount, 0U);
    EXPECT_EQ(result.fallbackCount, 1U);
    EXPECT_EQ(result.ignoredInstructionCount, 0U);
    EXPECT_FALSE(result.diagnostics.empty());
}

TEST(VfSimAdapterGolden, ScopeSentinelPreservesTraceParserErrors)
{
    ptoVf::trace::Reset();
    ::pto::mocker::ResetTrace();
    ::pto::mocker::BeginPtoInstr("TADD");
    {
        ptoVf::ScopeSentinel scope;
        __pto_trace_loop_iter(123);
        (void)scope;
    }
    ::pto::mocker::EndPtoInstr();

    const auto& trace = ::pto::mocker::GetTrace();
    ASSERT_FALSE(trace.executed_pto.empty());
    const auto& prediction = trace.executed_pto.back().vfPrediction;
    EXPECT_EQ(prediction.status, ptoVf::VfPredictionStatus::INVALID_TRACE);
    EXPECT_EQ(prediction.vfSimHitCount, 0U);
    EXPECT_EQ(prediction.fallbackCount, 1U);
    EXPECT_EQ(prediction.cycles, ptoVf::kUnknownInstructionFallbackCycles);
    EXPECT_EQ(trace.executed_pto.back().total_cycles, ptoVf::kUnknownInstructionFallbackCycles);
    ASSERT_FALSE(prediction.diagnostics.empty());
    EXPECT_NE(prediction.diagnostics.front().find("stray loop_iter/exit"), std::string::npos);
}

TEST(VfSimAdapterGolden, MixedScopeTraceFailureKeepsFallbackCycles)
{
    ptoVf::trace::Reset();
    ::pto::mocker::ResetTrace();
    ::pto::mocker::BeginPtoInstr("TADD");
    {
        ptoVf::ScopeSentinel scope;
        __pto_trace_loop_enter(1, __FILE__, __LINE__, 0);
        __pto_trace_loop_iter(1);
        ptoVf::trace::RecordOp(ptoVf::AsInst(makePtoInst(
            "vlds", {makePtoValue("v0", ptoVf::MemLocation::PhyRegister)}, {makePtoValue("mem0", ptoVf::MemLocation::UB)})));
        ptoVf::trace::RecordOp(ptoVf::AsInst(makePtoInst(
            "vlds", {makePtoValue("v1", ptoVf::MemLocation::PhyRegister)}, {makePtoValue("mem1", ptoVf::MemLocation::UB)})));
        ptoVf::trace::RecordOp(ptoVf::AsInst(makePtoInst(
            "vadd",
            {makePtoValue("v2", ptoVf::MemLocation::PhyRegister)},
            {makePtoValue("v0", ptoVf::MemLocation::PhyRegister),
             makePtoValue("v1", ptoVf::MemLocation::PhyRegister)})));
        ptoVf::trace::RecordOp(ptoVf::AsInst(makePtoInst(
            "vsts", {makePtoValue("mem2", ptoVf::MemLocation::UB)}, {makePtoValue("v2", ptoVf::MemLocation::PhyRegister)})));
        __pto_trace_loop_exit(1);
        (void)scope;
    }
    {
        ptoVf::ScopeSentinel scope;
        ptoVf::trace::RecordOp(ptoVf::VfInst{"vmull", {}, {}, {}});
        __pto_trace_loop_exit(456);
        (void)scope;
    }
    ::pto::mocker::EndPtoInstr();

    const auto& trace = ::pto::mocker::GetTrace();
    ASSERT_FALSE(trace.executed_pto.empty());
    const auto& prediction = trace.executed_pto.back().vfPrediction;
    EXPECT_EQ(prediction.status, ptoVf::VfPredictionStatus::PARTIAL_FALLBACK);
    EXPECT_EQ(prediction.vfSimHitCount, 1U);
    EXPECT_EQ(prediction.fallbackCount, 1U);
    EXPECT_GT(prediction.cycles, ptoVf::FallbackVecCycle("vmull"));
    EXPECT_EQ(trace.executed_pto.back().total_cycles, prediction.cycles);
    ASSERT_FALSE(prediction.diagnostics.empty());
    EXPECT_NE(prediction.diagnostics.back().find("stray loop_iter/exit"), std::string::npos);
}

TEST(VfSimAdapterGolden, CapturesPredicateMemoryAndConfigOperands)
{
    vector_bool predicate0;
    vector_bool predicate1;
    vector_bool predicate2;
    vector_bool mask;
    uint32_t ub[4]{};

    const auto xorInst = capturePtoInstruction([&] { pxor(predicate0, predicate1, predicate2, mask); });
    EXPECT_EQ(xorInst.opName, "pxor");
    ASSERT_EQ(xorInst.dst.size(), 1U);
    EXPECT_EQ(xorInst.dst[0].location, ptoVf::MemLocation::PredicateRegister);
    ASSERT_EQ(xorInst.src.size(), 3U);
    EXPECT_EQ(xorInst.src[0].location, ptoVf::MemLocation::PredicateRegister);

    const auto selectInst = capturePtoInstruction([&] { psel(predicate0, predicate1, predicate2, mask); });
    EXPECT_EQ(selectInst.opName, "psel");
    EXPECT_EQ(selectInst.dst.size(), 1U);
    EXPECT_EQ(selectInst.src.size(), 3U);

    const auto loadInst = capturePtoInstruction([&] { plds(predicate0, static_cast<uint32_t*>(ub), 16, US); });
    EXPECT_EQ(loadInst.opName, "plds");
    ASSERT_EQ(loadInst.dst.size(), 1U);
    EXPECT_EQ(loadInst.dst[0].location, ptoVf::MemLocation::PredicateRegister);
    ASSERT_EQ(loadInst.src.size(), 1U);
    EXPECT_EQ(loadInst.src[0].location, ptoVf::MemLocation::UB);
    ASSERT_EQ(loadInst.arguments.size(), 2U);
    EXPECT_EQ(loadInst.arguments[0].kind, ptoVf::VfArgKind::Immediate);
    EXPECT_EQ(loadInst.arguments[0].value, "16");
    EXPECT_EQ(loadInst.arguments[1].kind, ptoVf::VfArgKind::Config);

    const auto storeInst = capturePtoInstruction([&] { psts(predicate0, static_cast<uint32_t*>(ub), 16, NORM); });
    EXPECT_EQ(storeInst.opName, "psts");
    ASSERT_EQ(storeInst.dst.size(), 1U);
    EXPECT_EQ(storeInst.dst[0].location, ptoVf::MemLocation::UB);
    ASSERT_EQ(storeInst.src.size(), 1U);
    EXPECT_EQ(storeInst.src[0].location, ptoVf::MemLocation::PredicateRegister);

    const auto packInst = capturePtoInstruction([&] { ppack(predicate0, predicate1, LOWER); });
    EXPECT_EQ(packInst.opName, "ppack");
    EXPECT_EQ(packInst.dst.size(), 1U);
    EXPECT_EQ(packInst.src.size(), 1U);
    EXPECT_EQ(packInst.arguments.size(), 1U);

    const auto unpackInst = capturePtoInstruction([&] { punpack(predicate0, predicate1, LOWER); });
    EXPECT_EQ(unpackInst.opName, "punpack");
    EXPECT_EQ(unpackInst.dst.size(), 1U);
    EXPECT_EQ(unpackInst.src.size(), 1U);
    EXPECT_EQ(unpackInst.arguments.size(), 1U);
}

TEST(VfSimAdapterGolden, PredicateFactoryPreservesTraceIdentity)
{
    vector_bool combined;
    vector_bool all;
    const auto info = capturePtoVfInfo([&] {
        vector_bool predicate0 = plt_b32(16, POST_UPDATE);
        vector_bool predicate1 = plt_b32(32, POST_UPDATE);
        pxor(combined, predicate0, predicate1, all);
    });

    ASSERT_EQ(info.tree.size(), 3U);
    const auto& setup0 = ptoVf::AsInst(info.tree[0]);
    const auto& setup1 = ptoVf::AsInst(info.tree[1]);
    const auto& xorInst = ptoVf::AsInst(info.tree[2]);
    ASSERT_EQ(setup0.dst.size(), 1U);
    ASSERT_EQ(setup1.dst.size(), 1U);
    ASSERT_EQ(xorInst.src.size(), 3U);
    EXPECT_NE(setup0.dst[0].name, setup1.dst[0].name);
    EXPECT_EQ(xorInst.src[0].name, setup0.dst[0].name);
    EXPECT_EQ(xorInst.src[1].name, setup1.dst[0].name);
}

TEST(VfSimAdapterGolden, CapturesCarryAndMultiOutputOperands)
{
    vector_bool carryOut;
    vector_bool carryIn;
    vector_bool mask;
    vector_u32 dst0;
    vector_u32 dst1;
    vector_u32 src0;
    vector_u32 src1;

    const auto addCarry = capturePtoInstruction([&] { vaddc(carryOut, dst0, src0, src1, mask); });
    EXPECT_EQ(addCarry.opName, "vaddc");
    EXPECT_EQ(addCarry.dst.size(), 2U);
    EXPECT_EQ(addCarry.src.size(), 3U);

    const auto addCarryIn = capturePtoInstruction([&] { vaddcs(carryOut, dst0, src0, src1, carryIn, mask); });
    EXPECT_EQ(addCarryIn.opName, "vaddcs");
    EXPECT_EQ(addCarryIn.dst.size(), 2U);
    EXPECT_EQ(addCarryIn.src.size(), 4U);

    const auto subCarry = capturePtoInstruction([&] { vsubc(carryOut, dst0, src0, src1, mask); });
    EXPECT_EQ(subCarry.opName, "vsubc");
    EXPECT_EQ(subCarry.dst.size(), 2U);
    EXPECT_EQ(subCarry.src.size(), 3U);

    const auto subCarryIn = capturePtoInstruction([&] { vsubcs(carryOut, dst0, src0, src1, carryIn, mask); });
    EXPECT_EQ(subCarryIn.opName, "vsubcs");
    EXPECT_EQ(subCarryIn.dst.size(), 2U);
    EXPECT_EQ(subCarryIn.src.size(), 4U);

    const auto multiply = capturePtoInstruction([&] { vmull(dst0, dst1, src0, src1, mask); });
    EXPECT_EQ(multiply.opName, "vmull");
    EXPECT_EQ(multiply.dst.size(), 2U);
    EXPECT_EQ(multiply.src.size(), 3U);

    const auto compare = capturePtoInstruction([&] { vcmp_ge(carryOut, src0, src1, mask); });
    EXPECT_EQ(compare.opName, "vcmp_ge");
    EXPECT_EQ(compare.dst.size(), 1U);
    EXPECT_EQ(compare.src.size(), 3U);

    const auto interleave = capturePtoInstruction([&] { vintlv(dst0, dst1, src0, src1); });
    EXPECT_EQ(interleave.opName, "vintlv");
    EXPECT_EQ(interleave.dst.size(), 2U);
    EXPECT_EQ(interleave.src.size(), 2U);

    const auto deinterleave = capturePtoInstruction([&] { vdintlv(dst0, dst1, src0, src1); });
    EXPECT_EQ(deinterleave.opName, "vdintlv");
    EXPECT_EQ(deinterleave.dst.size(), 2U);
    EXPECT_EQ(deinterleave.src.size(), 2U);
}

TEST(VfSimAdapterGolden, CapturedUnsupportedInstructionUsesFormulaFallback)
{
    vector_bool mask;
    vector_u32 dst0;
    vector_u32 dst1;
    vector_u32 src0;
    vector_u32 src1;
    const auto info = capturePtoVfInfo([&] { vmull(dst0, dst1, src0, src1, mask); });

    const auto result = ptoVf::predictVfCyclesWithVfSim({info});

    EXPECT_EQ(result.status, ptoVf::VfPredictionStatus::UNSUPPORTED_FORM);
    EXPECT_EQ(result.vfSimHitCount, 0U);
    EXPECT_EQ(result.fallbackCount, 1U);
    EXPECT_EQ(result.cycles, ptoVf::FallbackVecCycle("vmull"));
    EXPECT_FALSE(result.diagnostics.empty());
}

TEST(VfSimAdapterGolden, CapturedVbrUsesExistingNativeModel)
{
    vector_u32 dst;
    const auto info = capturePtoVfInfo([&] { vbr(dst, 7U); });
    ASSERT_EQ(info.tree.size(), 1U);
    const auto& inst = ptoVf::AsInst(info.tree.front());
    EXPECT_EQ(inst.dst.size(), 1U);
    EXPECT_TRUE(inst.src.empty());
    ASSERT_EQ(inst.arguments.size(), 1U);
    EXPECT_EQ(inst.arguments[0].kind, ptoVf::VfArgKind::Immediate);
    EXPECT_EQ(inst.arguments[0].value, "7");

    const auto result = ptoVf::predictVfCyclesWithVfSim({info});
    EXPECT_EQ(result.status, ptoVf::VfPredictionStatus::VF_SIM_HIT);
    EXPECT_EQ(result.vfSimHitCount, 1U);
    EXPECT_EQ(result.fallbackCount, 0U);
}

} // namespace
