/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>

#include <gtest/gtest.h>

#include <pto/pto-inst.hpp>
#include <pto/common/constants.hpp>

#include "a5_host_tileop_check.hpp"

using namespace pto;

namespace {

template <typename T, int rows, int cols>
void runTAdd()
{
    using TileData = Tile<TileType::Vec, T, rows, cols, BLayout::RowMajor, -1, -1>;
    TileData src0Tile(rows, cols);
    TileData src1Tile(rows, cols);
    TileData dstTile(rows, cols);
    TASSIGN(src0Tile, 0x0);
    TASSIGN(src1Tile, 0x4000);
    TASSIGN(dstTile, 0x8000);

    ::pto::mocker::ResetTrace();
    TADD(dstTile, src0Tile, src1Tile);

    constexpr uint64_t repeat = (static_cast<uint64_t>(rows) * cols + 63) / 64;
    constexpr uint64_t expectedCycles = [] {
        if constexpr (rows == 1) {
            return static_cast<uint64_t>(1.2469671 * repeat + 55.073657 + 0.5);
        }
        constexpr uint64_t innerRepeat = (static_cast<uint64_t>(cols) + 63) / 64;
        return static_cast<uint64_t>((1.8749443 * innerRepeat + 2.8762808) * rows + 44.344619 + 0.5);
    }();
    pto::test::a5::ExpectSupportedVfTileOp("TADD", expectedCycles);
}

} // namespace

TEST(TAdd, float_1x64) { runTAdd<float, 1, 64>(); }

TEST(TAdd, float_1x512) { runTAdd<float, 1, 512>(); }

TEST(TAdd, float_1x1024) { runTAdd<float, 1, 1024>(); }

TEST(TAdd, float_1x2048) { runTAdd<float, 1, 2048>(); }

TEST(TAdd, float_1x4096) { runTAdd<float, 1, 4096>(); }

TEST(TAdd, float_1x6144) { runTAdd<float, 1, 6144>(); }

// Dynamic ValidCol is not a compile-time contiguous tile, even when its
// runtime valid width happens to equal Cols. The NPU BinaryInstr therefore
// selects its 2D implementation path.
TEST(TAdd, float_dynamic_10x672_uses_2d_path) { runTAdd<float, 10, 672>(); }

TEST(TAdd, MultiCoreUnsupportedReportIsSafeAndExplicit)
{
    ::pto::perf_sim::SimReport report;
    report.op_name = "a5_unsupported_multicore";
    report.num_cores = 2;
    report.costmodel_status = ::pto::perf_sim::CostModelStatus::Unsupported;
    report.costmodel_diagnostics = {"logical core 0, TADD: formula is unavailable"};

    const auto outputDir = std::filesystem::temp_directory_path() / "pto_a5_unsupported_report_test";
    const auto jsonPath = outputDir / "report.json";
    const auto csvPath = outputDir / "report.csv";

    ::pto::perf_sim::PerfSimReporter::WriteSwimlaneJson(jsonPath.string(), report);
    ::pto::perf_sim::PerfSimReporter::WritePipelineSummaryCSV(csvPath.string(), report);

    std::ifstream jsonInput(jsonPath);
    const std::string json((std::istreambuf_iterator<char>(jsonInput)), std::istreambuf_iterator<char>());
    EXPECT_NE(json.find("\"status\":\"Unsupported\""), std::string::npos);
    EXPECT_NE(json.find("formula is unavailable"), std::string::npos);
    jsonInput.close();

    std::ifstream csvInput(csvPath);
    const std::string csv((std::istreambuf_iterator<char>(csvInput)), std::istreambuf_iterator<char>());
    EXPECT_NE(csv.find("costmodel_status,diagnostic"), std::string::npos);
    EXPECT_NE(csv.find("Unsupported"), std::string::npos);
    EXPECT_NE(csv.find("formula is unavailable"), std::string::npos);
    csvInput.close();

    std::error_code ec;
    std::filesystem::remove_all(outputDir, ec);
}
