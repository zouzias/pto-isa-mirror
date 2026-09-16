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

TEST(TRecip, float_default_1x512)
{
    using TileData = Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1>;
    TileData src(1, 512);
    TileData dst(1, 512);

    ::pto::mocker::ResetTrace();
    ::pto::perf_sim::PtoRecorder::Clear();
    TRECIP(dst, src);

    constexpr uint64_t expectedCycles = static_cast<uint64_t>(1.9081456 * 8 + 64.516464 + 0.5);
    // Match the NPU wrapper: reciprocal forwards to scalar-first TDIVS.
    pto::test::a5::ExpectSupportedVfTileOp("TDIVS", expectedCycles);
}
