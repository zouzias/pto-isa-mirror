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
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <gtest/gtest.h>

#include "pto/cpu/Hifloat8.hpp"
#include "pto/costmodel/a5/cce_costmodel/a5_vf_stub.hpp"
#include "pto/costmodel/vfsim/pto_adapter/pto_canonical_lowering.hpp"
#include "pto/costmodel/vfsim/pto_adapter/vfsim_cost_model.hpp"

#include "native/ParamDB.h"
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
