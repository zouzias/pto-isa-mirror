/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <cstddef>
#include <cstdint>

#include <gtest/gtest.h>
#include <pto/pto-inst.hpp>
#include <pto/common/constants.hpp>

using namespace pto;

namespace {

template <
    typename T, size_t Rows, size_t Cols, size_t ValidRows, size_t ValidCols, uint16_t IndexRow, uint16_t IndexCol,
    BLayout Layout = BLayout::RowMajor, SLayout Fractal = SLayout::NoneBox>
void runTExtract(int expectedDmaCopies = 1, int expectedVectorCopies = 0)
{
    constexpr int DST_VALID_ROWS = ValidRows - IndexRow;
    constexpr int DST_VALID_COLS = ValidCols - IndexCol;

    Tile<TileType::Vec, T, Rows, Cols, Layout, ValidRows, ValidCols, Fractal> srcTile;
    Tile<TileType::Vec, T, Rows, Cols, Layout, DST_VALID_ROWS, DST_VALID_COLS, Fractal> dstTile;

    TASSIGN(srcTile, 0x0);
    TASSIGN(dstTile, 0x10000);

    mocker::ResetTrace();
    TEXTRACT(dstTile, srcTile, IndexRow, IndexCol);

    const auto& instructions = mocker::GetTrace().executed_pto;
    ASSERT_EQ(instructions.size(), 1);
    EXPECT_EQ(instructions[0].name, "TEXTRACT");
    EXPECT_GT(instructions[0].total_cycles, 0);
    int dmaCopies = 0;
    int vectorCopies = 0;
    for (const auto& call : instructions[0].cce_calls) {
        if (call.name == "copy_ubuf_to_ubuf") {
            ++dmaCopies;
        } else if (call.name == "vcopy") {
            ++vectorCopies;
        }
    }
    EXPECT_EQ(dmaCopies, expectedDmaCopies);
    EXPECT_EQ(vectorCopies, expectedVectorCopies);
}

} // namespace

TEST(TExtract, half_32x32_idx_0_0) { runTExtract<half, 32, 32, 32, 32, 0, 0>(); }

TEST(TExtract, float_128x96_idx_0_0) { runTExtract<float, 128, 96, 128, 96, 0, 0>(); }

TEST(TExtract, half_32x32_idx_8_16) { runTExtract<half, 32, 32, 32, 32, 8, 16>(); }

TEST(TExtract, float_128x96_idx_8_16) { runTExtract<float, 128, 96, 128, 96, 8, 16>(); }

TEST(TExtract, half_32x32_valid31_idx_8_16) { runTExtract<half, 32, 32, 31, 31, 8, 16>(0, 1); }

TEST(TExtract, float_128x96_valid125_idx_8_16) { runTExtract<float, 128, 96, 125, 93, 8, 16>(1, 1); }

TEST(TExtract, half_nz_32x32_idx_0_0)
{
    runTExtract<half, 32, 32, 32, 32, 0, 0, BLayout::ColMajor, SLayout::RowMajor>();
}

TEST(TExtract, float_nz_128x96_idx_16_16)
{
    runTExtract<float, 128, 96, 128, 96, 16, 16, BLayout::ColMajor, SLayout::RowMajor>();
}
