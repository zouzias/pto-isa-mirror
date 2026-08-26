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
#include "cpu_tile_test_utils.h"

#include <gtest/gtest.h>

using namespace pto;
using namespace CpuTileTestUtils;

namespace {

TEST(TGetScaleAddrTest, AliasesSourceStorage)
{
    using TileData = Tile<TileType::Vec, float, 2, 8>;

    TileData src;
    TileData dst;
    std::size_t addr = 0;
    AssignTileStorage(addr, src, dst);

    FillLinear(src, 1.0f);
    TGET_SCALE_ADDR(dst, src);

    ASSERT_EQ(dst.data(), src.data());
    src.data()[3] = 42.0f;
    ExpectValueEquals(dst.data()[3], 42.0f);
}

} // namespace
