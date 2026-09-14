/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#include <pto/pto-inst.hpp>
#include <pto/common/constants.hpp>
#include <gtest/gtest.h>

#include "a5_host_tileop_check.hpp"

using namespace pto;

namespace {

template <typename T, int rows, int cols>
void runTDiv()
{
    using TileData = Tile<TileType::Vec, T, rows, cols, BLayout::RowMajor, -1, -1>;
    TileData src0Tile(rows, cols);
    TileData src1Tile(rows, cols);
    TileData dstTile(rows, cols);
    TASSIGN(src0Tile, 0x0);
    TASSIGN(src1Tile, 0x4000);
    TASSIGN(dstTile, 0x8000);

    ::pto::mocker::ResetTrace();
    ::pto::perf_sim::PtoRecorder::Clear();
    TDIV(dstTile, src0Tile, src1Tile);

    constexpr uint64_t repeat = (static_cast<uint64_t>(rows) * cols + 63) / 64;
    pto::test::a5::ExpectLastBinaryVecTileOp({"vlds", "vlds", "vdiv", "vsts"}, repeat);
}

} // namespace

TEST(TDiv, float_1x512) { runTDiv<float, 1, 512>(); }

TEST(TRecip, float_default_1x512)
{
    using TileData = Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1>;
    TileData src(1, 512);
    TileData dst(1, 512);

    ::pto::mocker::ResetTrace();
    ::pto::perf_sim::PtoRecorder::Clear();
    TRECIP(dst, src);

    constexpr uint64_t expectedCycles = static_cast<uint64_t>(1.9081456 * 8 + 64.516464 + 0.5);
    pto::test::a5::ExpectSupportedVfTileOp("TRECIP", expectedCycles);
}

TEST(TRecip, float_high_precision_1x512)
{
    using TileData = Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1>;
    TileData src(1, 512);
    TileData dst(1, 512);

    ::pto::mocker::ResetTrace();
    ::pto::perf_sim::PtoRecorder::Clear();
    TRECIP<RecipAlgorithm::HIGH_PRECISION>(dst, src);

    constexpr uint64_t expectedCycles = static_cast<uint64_t>(345.21788 * 8 + 48.715084 + 0.5);
    pto::test::a5::ExpectSupportedVfTileOp("TRECIP", expectedCycles);
}

TEST(TDivS, float_high_precision_is_unsupported_without_fit)
{
    using TileData = Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1>;
    TileData src(1, 512);
    TileData dst(1, 512);

    ::pto::mocker::ResetTrace();
    ::pto::perf_sim::PtoRecorder::Clear();
    TDIVS<DivAlgorithm::HIGH_PRECISION>(dst, src, 2.0F);

    pto::test::a5::ExpectUnsupportedVfTileOp({}, 0);
}

TEST(TCvt, fp16_to_fp32_rint_sat_on_1x512)
{
    using SrcTile = Tile<TileType::Vec, half, 1, 512, BLayout::RowMajor, -1, -1>;
    using DstTile = Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1>;
    SrcTile src(1, 512);
    DstTile dst(1, 512);

    ::pto::mocker::ResetTrace();
    ::pto::perf_sim::PtoRecorder::Clear();
    TCVT(dst, src, RoundMode::CAST_RINT, SaturationMode::ON);

    constexpr uint64_t expectedCycles = static_cast<uint64_t>(-1.7704918 * 8 + 82.442623 + 0.5);
    pto::test::a5::ExpectSupportedVfTileOp("TCVT", expectedCycles);
}
