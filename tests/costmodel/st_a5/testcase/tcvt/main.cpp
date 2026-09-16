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
