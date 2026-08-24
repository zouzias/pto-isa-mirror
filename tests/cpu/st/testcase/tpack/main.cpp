/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <pto/pto-inst.hpp>
#include "cpu_tile_test_utils.h"

#include <gtest/gtest.h>

using namespace pto;
using namespace CpuTileTestUtils;

namespace {

TEST(TPackTest, CopiesValidValuesWithTypeConversion)
{
    using SrcTile = Tile<TileType::Vec, int16_t, 3, 16, BLayout::RowMajor, 2, 3>;
    using DstTile = Tile<TileType::Vec, int32_t, 3, 8, BLayout::RowMajor, 2, 3>;

    SrcTile src;
    DstTile dst;
    std::size_t addr = 0;
    AssignTileStorage(addr, src, dst);

    FillAll(dst, 0);
    const int16_t values[2][3] = {{1, -2, 3}, {4, -5, 6}};
    for (int r = 0; r < src.GetValidRow(); ++r) {
        for (int c = 0; c < src.GetValidCol(); ++c) {
            SetValue(src, r, c, values[r][c]);
        }
    }

    TPACK(dst, src);

    for (int r = 0; r < src.GetValidRow(); ++r) {
        for (int c = 0; c < src.GetValidCol(); ++c) {
            ExpectValueEquals(GetValue(dst, r, c), static_cast<int32_t>(values[r][c]));
        }
    }
}

} // namespace
