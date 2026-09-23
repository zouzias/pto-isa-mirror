/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <system_error>
#include <vector>

#include <gtest/gtest.h>
#include <pto/pto-inst.hpp>

#include "cpu_tile_test_utils.h"

using CpuTileTestUtils::AssignTileStorage;
using CpuTileTestUtils::FillLinear;

void LaunchTraceKernel(float* out, float* src0, float* src1, void* stream);

namespace {

class ScopedEnvironmentVariable {
public:
    explicit ScopedEnvironmentVariable(const char* name) : name_(name)
    {
        if (const auto* value = std::getenv(name)) {
            saved_ = value;
        }
    }

    ~ScopedEnvironmentVariable()
    {
        if (saved_) {
            setenv(name_, saved_->c_str(), 1);
        } else {
            unsetenv(name_);
        }
    }

    ScopedEnvironmentVariable(const ScopedEnvironmentVariable&) = delete;
    ScopedEnvironmentVariable& operator=(const ScopedEnvironmentVariable&) = delete;

private:
    const char* name_;
    std::optional<std::string> saved_;
};

class ScopedRuntimeConfig {
public:
    explicit ScopedRuntimeConfig(pto::cpu_sim::RuntimeConfig& config) : config_(config) {}

    ~ScopedRuntimeConfig()
    {
        config_.initialized = initialized_;
        config_.device_id = deviceId_;
        config_.num_cores = numCores_;
        config_.trace_root = traceRoot_;
        config_.next_stream_id = nextStreamId_;
        config_.next_launch_id = nextLaunchId_;
    }

    ScopedRuntimeConfig(const ScopedRuntimeConfig&) = delete;
    ScopedRuntimeConfig& operator=(const ScopedRuntimeConfig&) = delete;

private:
    pto::cpu_sim::RuntimeConfig& config_;
    bool initialized_ = config_.initialized;
    uint32_t deviceId_ = config_.device_id;
    uint32_t numCores_ = config_.num_cores;
    std::filesystem::path traceRoot_ = config_.trace_root;
    uint64_t nextStreamId_ = config_.next_stream_id;
    uint64_t nextLaunchId_ = config_.next_launch_id;
};

template <typename TileData>
std::uintptr_t TraceAddress(TileData& tile)
{
    return reinterpret_cast<std::uintptr_t>(tile.data());
}

class TTraceTest : public testing::Test {
protected:
    void SetUp() override
    {
        savedTraceSetting = pto::cpu_sim::g_instruction_trace_setting.load();
        std::string pathTemplate = (std::filesystem::temp_directory_path() / "pto_ttrace_XXXXXX").string();
        const char* directory = mkdtemp(pathTemplate.data());
        ASSERT_NE(directory, nullptr);
        testRoot = directory;
        const auto traceRoot = testRoot / "traces";
        ASSERT_EQ(setenv("PTO_CPU_SIM_TRACE_DIR", traceRoot.c_str(), 1), 0);
        pto::cpu_sim::runtime_config().trace_root = traceRoot;
        pto::cpu_sim::SetInstructionTraceEnabled(true);
        pto::cpu_sim::ResetInstructionTrace();
        pto::cpu_sim::reset_execution_context();
        pto::NPU_MEMORY_CLEAR();
        pto::cpu_sim::set_execution_context(7, 0, 1);
    }

    void TearDown() override
    {
        pto::cpu_sim::ResetInstructionTrace();
        pto::cpu_sim::reset_execution_context();
        pto::NPU_MEMORY_CLEAR();
        pto::cpu_sim::g_instruction_trace_setting.store(savedTraceSetting);
        if (!testRoot.empty()) {
            std::error_code error;
            std::filesystem::remove_all(testRoot, error);
            EXPECT_FALSE(error) << error.message();
        }
    }

    ScopedRuntimeConfig savedRuntime{pto::cpu_sim::runtime_config()};
    ScopedEnvironmentVariable savedEnable{"PTO_CPU_SIM_TRACE_ENABLE"};
    ScopedEnvironmentVariable savedDirectory{"PTO_CPU_SIM_TRACE_DIR"};
    pto::cpu_sim::InstructionTraceSetting savedTraceSetting{};
    std::filesystem::path testRoot;
};

TEST_F(TTraceTest, CapturesBasicTileInstructions)
{
    if (!pto::cpu_sim::kInstructionTraceEnabled) {
        GTEST_SKIP() << "PTO_CPU_SIM_TRACE_MODE is disabled for this build.";
    }

    using TileData = pto::Tile<pto::TileType::Vec, float, 2, 32>;

    TileData dst;
    TileData src0;
    TileData src1;
    std::size_t addr = 0;
    AssignTileStorage(addr, dst, src0, src1);
    FillLinear(src0, 1.0f);
    FillLinear(src1, 2.0f);
    pto::cpu_sim::ResetInstructionTrace();

    pto::TADD(dst, src0, src1);
    pto::TDIV(dst, src0, src1);

    const auto trace = pto::cpu_sim::CopyInstructionTraceRecords();
    ASSERT_EQ(trace.size(), 2u);

    EXPECT_EQ(trace[0].opcode, "TADD");
    EXPECT_EQ(trace[0].block_idx, 7u);
    EXPECT_EQ(trace[0].sequence_id, 0u);
    ASSERT_EQ(trace[0].input_tiles.size(), 2u);
    ASSERT_EQ(trace[0].output_tiles.size(), 1u);
    EXPECT_EQ(trace[0].input_tiles[0].address, TraceAddress(src0));
    EXPECT_EQ(trace[0].input_tiles[1].address, TraceAddress(src1));
    EXPECT_EQ(trace[0].output_tiles[0].address, TraceAddress(dst));
    EXPECT_EQ(trace[0].output_tiles[0].shape, (std::vector<int64_t>{2, 32}));

    EXPECT_EQ(trace[1].opcode, "TDIV");
    EXPECT_EQ(trace[1].sequence_id, 1u);
    ASSERT_EQ(trace[1].input_tiles.size(), 2u);
    ASSERT_EQ(trace[1].output_tiles.size(), 1u);

    std::ostringstream json;
    pto::cpu_sim::DumpInstructionTraceJson(json);
    EXPECT_NE(json.str().find("\"opcode\":\"TADD\""), std::string::npos);
    EXPECT_NE(json.str().find("\"opcode\":\"TDIV\""), std::string::npos);
}

TEST_F(TTraceTest, CapturesInterleavedAndMultiOutputOperands)
{
    if (!pto::cpu_sim::kInstructionTraceEnabled) {
        GTEST_SKIP() << "PTO_CPU_SIM_TRACE_MODE is disabled for this build.";
    }

    using SrcTile = pto::Tile<pto::TileType::Vec, float, 4, 32>;
    using ValTile = pto::Tile<pto::TileType::Vec, float, 8, 1, pto::BLayout::ColMajor, 4, 1>;
    using IdxTile = pto::Tile<pto::TileType::Vec, uint32_t, 8, 1, pto::BLayout::ColMajor, 4, 1>;
    using TmpTile = pto::Tile<pto::TileType::Vec, uint32_t, 4, 32>;

    SrcTile src;
    ValTile dstVal;
    IdxTile dstIdx;
    TmpTile tmp;
    std::size_t addr = 0;
    AssignTileStorage(addr, src, dstVal, dstIdx, tmp);
    FillLinear(src, 1.0f);
    pto::cpu_sim::ResetInstructionTrace();

    pto::TROWARGMAX(dstVal, dstIdx, src, tmp);

    const auto trace = pto::cpu_sim::CopyInstructionTraceRecords();
    ASSERT_EQ(trace.size(), 1u);
    EXPECT_EQ(trace[0].opcode, "TROWARGMAX");
    ASSERT_EQ(trace[0].output_tiles.size(), 2u);
    ASSERT_EQ(trace[0].input_tiles.size(), 2u);
    EXPECT_EQ(trace[0].output_tiles[0].address, TraceAddress(dstVal));
    EXPECT_EQ(trace[0].output_tiles[1].address, TraceAddress(dstIdx));
    EXPECT_EQ(trace[0].input_tiles[0].address, TraceAddress(src));
    EXPECT_EQ(trace[0].input_tiles[1].address, TraceAddress(tmp));
}

TEST_F(TTraceTest, CapturesPointerOutputsForMxQuant)
{
    if (!pto::cpu_sim::kInstructionTraceEnabled) {
        GTEST_SKIP() << "PTO_CPU_SIM_TRACE_MODE is disabled for this build.";
    }

    using SrcTile = pto::Tile<pto::TileType::Vec, float, 16, 64>;
    using Fp8Tile = pto::Tile<pto::TileType::Vec, int8_t, 16, 64>;
    using ExpTile = pto::Tile<pto::TileType::Vec, uint8_t, 1, 32, pto::BLayout::RowMajor, 1, 32>;
    using MaxTile = pto::Tile<pto::TileType::Vec, float, 1, 32, pto::BLayout::RowMajor, 1, 32>;

    SrcTile src;
    Fp8Tile dst;
    ExpTile exp;
    ExpTile expZz;
    MaxTile max;
    SrcTile scaling;
    std::size_t addr = 0;
    AssignTileStorage(addr, src, dst, exp, expZz, max, scaling);

    for (int r = 0; r < src.GetValidRow(); ++r) {
        for (int c = 0; c < src.GetValidCol(); ++c) {
            src.data()[pto::GetTileElementOffset<SrcTile>(r, c)] = static_cast<float>((r + c) % 11) * 0.25f - 1.0f;
        }
    }
    pto::cpu_sim::ResetInstructionTrace();

    pto::TQUANT<pto::QuantType::MXFP8>(dst, src, &exp, &max, &scaling);
    pto::TQUANT<pto::QuantType::MXFP8, pto::VecStoreMode::NZ>(dst, src, &exp, &max, &scaling, &expZz);

    const auto trace = pto::cpu_sim::CopyInstructionTraceRecords();
    ASSERT_EQ(trace.size(), 2u);

    EXPECT_EQ(trace[0].opcode, "TQUANT");
    ASSERT_EQ(trace[0].input_tiles.size(), 1u);
    ASSERT_EQ(trace[0].output_tiles.size(), 4u);
    EXPECT_EQ(trace[0].input_tiles[0].address, TraceAddress(src));
    EXPECT_EQ(trace[0].output_tiles[0].address, TraceAddress(dst));
    EXPECT_EQ(trace[0].output_tiles[1].address, TraceAddress(exp));
    EXPECT_EQ(trace[0].output_tiles[2].address, TraceAddress(max));
    EXPECT_EQ(trace[0].output_tiles[3].address, TraceAddress(scaling));

    EXPECT_EQ(trace[1].opcode, "TQUANT");
    ASSERT_EQ(trace[1].input_tiles.size(), 1u);
    ASSERT_EQ(trace[1].output_tiles.size(), 5u);
    EXPECT_EQ(trace[1].output_tiles[4].address, TraceAddress(expZz));
}

TEST_F(TTraceTest, WritesKernelTraceFile)
{
    if (!pto::cpu_sim::kInstructionTraceEnabled) {
        GTEST_SKIP() << "PTO_CPU_SIM_TRACE_MODE is disabled for this build.";
    }

    constexpr std::size_t kNumel = 4 * 32;
    std::vector<float> out(kNumel, 0.0f);
    std::vector<float> src0(kNumel, 0.0f);
    std::vector<float> src1(kNumel, 0.0f);
    for (std::size_t i = 0; i < kNumel; ++i) {
        src0[i] = static_cast<float>(i);
        src1[i] = static_cast<float>(2 * i);
    }

    pto::cpu_sim::ResetInstructionTrace();
    LaunchTraceKernel(out.data(), src0.data(), src1.data(), nullptr);

    const auto tracePath = testRoot / "trace.jsonl";
    std::ofstream outFile(tracePath, std::ios::trunc);
    ASSERT_TRUE(outFile.is_open());
    pto::cpu_sim::DumpInstructionTraceJson(outFile);
    outFile.close();

    std::ifstream inFile(tracePath);
    ASSERT_TRUE(inFile.is_open());
    std::stringstream contents;
    contents << inFile.rdbuf();
    const std::string text = contents.str();
    EXPECT_NE(text.find("\"opcode\":\"TLOAD\""), std::string::npos);
    EXPECT_NE(text.find("\"opcode\":\"TADD\""), std::string::npos);
    EXPECT_NE(text.find("\"opcode\":\"TSTORE\""), std::string::npos);
}

TEST_F(TTraceTest, RuntimeDisablePreservesComputation)
{
    using TileData = pto::Tile<pto::TileType::Vec, float, 2, 32>;
    TileData dst, src;
    std::size_t addr = 0;
    AssignTileStorage(addr, dst, src);
    FillLinear(src, 1.0f);
    pto::cpu_sim::ResetInstructionTrace();
    pto::cpu_sim::SetInstructionTraceEnabled(false);
    pto::TADDS(dst, src, 2.0f);
    EXPECT_TRUE(pto::cpu_sim::CopyInstructionTraceRecords().empty());
    for (int i = 0; i < 64; ++i) {
        EXPECT_FLOAT_EQ(dst.data()[i], src.data()[i] + 2.0f);
    }

    // Exercise the role-based constructor while collection is disabled, too.
    {
        pto::cpu_sim::PtoInstrTraceScope scope("TADDS", "OII", dst, src, 2.0f);
    }
    EXPECT_TRUE(pto::cpu_sim::CopyInstructionTraceRecords().empty());

    pto::cpu_sim::SetInstructionTraceEnabled(true);
    pto::TDIVS(dst, src, 2.0f);
    for (int i = 0; i < 64; ++i) {
        EXPECT_FLOAT_EQ(dst.data()[i], src.data()[i] / 2.0f);
    }
    const auto records = pto::cpu_sim::CopyInstructionTraceRecords();
    ASSERT_EQ(records.size(), pto::cpu_sim::kInstructionTraceEnabled ? 1u : 0u);
    if (!records.empty()) {
        EXPECT_EQ(records[0].sequence_id, 0u);
        EXPECT_EQ(records[0].opcode, "TDIVS");
    }
}

TEST_F(TTraceTest, CapturesTemplatedVectorPipeline)
{
    using TileData = pto::Tile<pto::TileType::Vec, float, 2, 32>;
    TileData src, quotient, logarithm, result;
    std::size_t addr = 0;
    AssignTileStorage(addr, src, quotient, logarithm, result);
    FillLinear(src, 2.0f);
    pto::cpu_sim::ResetInstructionTrace();
    pto::TDIVS(quotient, src, 2.0f);
    pto::TLOG(logarithm, quotient);
    pto::TEXP(result, logarithm);
    for (int i = 0; i < 64; ++i) {
        EXPECT_NEAR(result.data()[i], src.data()[i] / 2.0f, 0.0001f);
    }
    const auto records = pto::cpu_sim::CopyInstructionTraceRecords();
    ASSERT_EQ(records.size(), pto::cpu_sim::kInstructionTraceEnabled ? 3u : 0u);
    if (!records.empty()) {
        EXPECT_EQ(records[0].opcode, "TDIVS");
        ASSERT_EQ(records[0].scalar_inputs.size(), 1u);
        EXPECT_EQ(records[0].scalar_inputs[0].value, "2");
        EXPECT_EQ(records[1].opcode, "TLOG");
        EXPECT_EQ(records[2].opcode, "TEXP");
    }
}

TEST_F(TTraceTest, RuntimeInitializationPreservesExplicitSetting)
{
    using namespace pto::cpu_sim;
    auto& config = runtime_config();
    const auto root = testRoot / "switch";

    constexpr std::size_t kNumel = 4 * 32;
    std::vector<float> src0(kNumel, 1.0f), src1(kNumel, 2.0f), out(kNumel);
    const KernelLaunchOptions options{.kernel_name = "switch", .explicit_block_count = 1};
    for (bool environmentEnabled : {false, true}) {
        ASSERT_EQ(setenv("PTO_CPU_SIM_TRACE_ENABLE", environmentEnabled ? "1" : "0", 1), 0);
        // -1 leaves the choice to the environment; 0 and 1 explicitly override it before initialization.
        for (int explicitSetting : {-1, 0, 1}) {
            SCOPED_TRACE(::testing::Message() << "environment=" << environmentEnabled << ", API=" << explicitSetting);
            g_instruction_trace_setting.store(InstructionTraceSetting::DefaultEnabled);
            ShutdownRuntime();
            if (explicitSetting >= 0) {
                SetInstructionTraceEnabled(explicitSetting != 0);
            }
            const bool expected =
                kInstructionTraceEnabled && (explicitSetting < 0 ? environmentEnabled : explicitSetting != 0);
            const auto caseRoot = root / (std::to_string(environmentEnabled) + "_" + std::to_string(explicitSetting));
            ASSERT_EQ(setenv("PTO_CPU_SIM_TRACE_DIR", caseRoot.c_str(), 1), 0);
            for (int invocation = 0; invocation < 3; ++invocation) {
                if (invocation == 1) {
                    InitializeRuntime();
                } else if (invocation == 2) {
                    SetInstructionTraceEnabled(!expected);
                }
                const bool enabled = invocation == 2 ? kInstructionTraceEnabled && !expected : expected;
                std::fill(src0.begin(), src0.end(), static_cast<float>(invocation + 1));
                std::fill(out.begin(), out.end(), std::numeric_limits<float>::quiet_NaN());
                const auto launchId = config.next_launch_id;
                std::vector<InstructionTraceRecord> records;
                LaunchKernelMultiCore(options, nullptr, [&]() {
                    LaunchTraceKernel(out.data(), src0.data(), src1.data(), nullptr);
                    records = CopyInstructionTraceRecords();
                });
                EXPECT_EQ(IsInstructionTraceEnabled(), enabled);
                EXPECT_EQ(IsTraceEnabled(), enabled);
                EXPECT_EQ(records.size(), enabled ? 7u : 0u);
                for (float value : out) {
                    EXPECT_FLOAT_EQ(value, static_cast<float>(invocation + 3));
                }
                const auto path = caseRoot / "switch" / ("launch_" + std::to_string(launchId)) / "trace.jsonl";
                if (enabled) {
                    ASSERT_TRUE(std::filesystem::exists(path));
                    EXPECT_GT(std::filesystem::file_size(path), 0u);
                } else {
                    EXPECT_FALSE(std::filesystem::exists(path));
                    if (invocation == 0) {
                        EXPECT_FALSE(std::filesystem::exists(caseRoot));
                    }
                }
            }
        }
    }
}

TEST_F(TTraceTest, MultiCoreLaunchExportsEachInvocation)
{
    constexpr std::size_t kBlockCount = 2;
    constexpr std::size_t kSubblockCount = 2;
    constexpr std::size_t kWorkerCount = kBlockCount * kSubblockCount;
    constexpr std::size_t kNumel = 4 * 32;
    auto& config = pto::cpu_sim::runtime_config();
    pto::cpu_sim::EnsureRuntimeInitialized();
    const auto root = config.trace_root;
    const auto firstLaunch = config.next_launch_id;
    const pto::cpu_sim::KernelLaunchOptions options{
        .kernel_name = "ttrace", .subblocks_per_block = kSubblockCount, .explicit_block_count = kBlockCount};
    std::vector<float> src0(kWorkerCount * kNumel, 1.0f), src1(kWorkerCount * kNumel, 2.0f);
    std::vector<float> out(kWorkerCount * kNumel, 0.0f);
    const std::array<std::string, 7> opcodes{"TASSIGN", "TASSIGN", "TASSIGN", "TLOAD", "TLOAD", "TADD", "TSTORE"};
    for (uint64_t invocation = 0; invocation < 2; ++invocation) {
        for (std::size_t worker = 0; worker < kWorkerCount; ++worker) {
            std::fill_n(
                src0.data() + worker * kNumel, kNumel, static_cast<float>(1 + worker + invocation * kWorkerCount));
        }
        std::fill(out.begin(), out.end(), std::numeric_limits<float>::quiet_NaN());
        std::array<std::vector<pto::cpu_sim::InstructionTraceRecord>, kWorkerCount> records;
        pto::cpu_sim::LaunchKernelMultiCore(options, nullptr, [&]() {
            const auto worker = get_block_idx() * kSubblockCount + get_subblockid();
            ASSERT_LT(worker, kWorkerCount);
            const auto offset = worker * kNumel;
            LaunchTraceKernel(out.data() + offset, src0.data() + offset, src1.data() + offset, nullptr);
            records[worker] = pto::cpu_sim::CopyInstructionTraceRecords();
        });
        for (std::size_t i = 0; i < out.size(); ++i) {
            EXPECT_FLOAT_EQ(out[i], src0[i] + src1[i]);
        }
        for (std::size_t worker = 0; worker < kWorkerCount; ++worker) {
            ASSERT_EQ(records[worker].size(), pto::cpu_sim::kInstructionTraceEnabled ? opcodes.size() : 0u);
            for (std::size_t i = 0; i < records[worker].size(); ++i) {
                EXPECT_EQ(records[worker][i].block_idx, worker / kSubblockCount);
                EXPECT_EQ(records[worker][i].subblock_id, worker % kSubblockCount);
                EXPECT_EQ(records[worker][i].sequence_id, i);
                EXPECT_EQ(records[worker][i].opcode, opcodes[i]);
            }
        }
        const auto path = root / "ttrace" / ("launch_" + std::to_string(firstLaunch + invocation)) / "trace.jsonl";
        if constexpr (pto::cpu_sim::kInstructionTraceEnabled) {
            std::ifstream input(path);
            ASSERT_TRUE(input.is_open());
            std::string line;
            for (std::size_t worker = 0; worker < kWorkerCount; ++worker) {
                for (std::size_t i = 0; i < opcodes.size(); ++i) {
                    ASSERT_TRUE(static_cast<bool>(std::getline(input, line)));
                    EXPECT_NE(
                        line.find("\"block_idx\":" + std::to_string(worker / kSubblockCount) + ","), std::string::npos);
                    EXPECT_NE(
                        line.find("\"subblock_id\":" + std::to_string(worker % kSubblockCount) + ","),
                        std::string::npos);
                    EXPECT_NE(line.find("\"sequence_id\":" + std::to_string(i) + ","), std::string::npos);
                    EXPECT_NE(line.find("\"opcode\":\"" + opcodes[i] + "\""), std::string::npos);
                }
            }
            EXPECT_FALSE(static_cast<bool>(std::getline(input, line)));
        } else {
            EXPECT_FALSE(std::filesystem::exists(path));
        }
    }
}

} // namespace
