/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <array>
#include <algorithm>
#include <cstdint>
#include <iostream>

#include <gtest/gtest.h>
#include <pto/pto-inst.hpp>

namespace {
using namespace pto;

template <int Rows, int Cols>
bool CheckRowPackedMask(const char* name, const std::array<uint8_t, Rows * 32>& mask)
{
    using DataTile = Tile<TileType::Vec, uint64_t, Rows, Cols>;
    using MaskTile = Tile<TileType::Vec, uint8_t, Rows, 32>;
    using TmpTile = Tile<TileType::Vec, uint8_t, 1, 32>;

    NPU_MEMORY_CLEAR();
    NPU_MEMORY_INIT();

    DataTile dstTile;
    DataTile src0Tile;
    DataTile src1Tile;
    MaskTile maskTile;
    TmpTile tmpTile;
    TASSIGN(src0Tile, 0);
    TASSIGN(src1Tile, 64);
    TASSIGN(dstTile, 128);
    TASSIGN(maskTile, 192);
    TASSIGN(tmpTile, 256);

    for (int i = 0; i < Rows * Cols; ++i) {
        src0Tile.data()[i] = static_cast<uint64_t>(100 + i);
        src1Tile.data()[i] = static_cast<uint64_t>(200 + i);
    }
    std::copy(mask.begin(), mask.end(), maskTile.data());

    TSEL(dstTile, maskTile, src0Tile, src1Tile, tmpTile);

    bool passed = true;
    for (int row = 0; row < Rows; ++row) {
        for (int col = 0; col < Cols; ++col) {
            const uint8_t bit = (maskTile.data()[row * 32 + col / 8] >> (col % 8)) & 1;
            const int index = row * Cols + col;
            const uint64_t expected = bit ? src0Tile.data()[index] : src1Tile.data()[index];
            if (dstTile.data()[index] != expected) {
                std::cerr << name << ": row=" << row << " col=" << col << " expected=" << expected
                          << " actual=" << dstTile.data()[index] << '\n';
                passed = false;
            }
        }
    }
    return passed;
}
} // namespace

TEST(TSelCpuSimRowMaskTest, partial_rows_start_at_their_own_mask_row)
{
    std::array<uint8_t, 2 * 32> mask{};
    mask[32] = 0x0f;
    EXPECT_TRUE((CheckRowPackedMask<2, 4>("partial-row mask", mask)));
}

TEST(TSelCpuSimRowMaskTest, single_row_mask_keeps_element_bit_order)
{
    std::array<uint8_t, 1 * 32> mask{};
    mask[0] = 0xa5;
    EXPECT_TRUE((CheckRowPackedMask<1, 8>("single-row mask", mask)));
}
