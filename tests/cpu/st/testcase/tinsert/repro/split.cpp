/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#include <cstdint>
#include <iostream>

#include <pto/pto-inst.hpp>

using namespace pto;

#ifndef SPLIT_COUNT
#define SPLIT_COUNT 2
#endif
#ifndef SPLIT_TYPE
#define SPLIT_TYPE float
#endif

int main()
{
    NPU_MEMORY_INIT(NPUArch::A5);
    using Src = Tile<
        TileType::Vec, SPLIT_TYPE, 17, 64, BLayout::ColMajor, 16, 64, SLayout::RowMajor, 512, PadValue::Null,
        CompactMode::RowPlusOne>;
    using Dst = Tile<TileType::Mat, SPLIT_TYPE, 16, 64, BLayout::ColMajor, 16, 64, SLayout::RowMajor>;
    Src src;
    Dst dst;
    TASSIGN(src, 0);
    TASSIGN(dst, 0);
    for (int row = 0; row < 16; ++row) {
        for (int col = 0; col < 64; ++col) {
            src.SetElement(row, col, row * 64 + col);
        }
    }
    constexpr auto MODE = SPLIT_COUNT == 2 ? TInsertMode::SPLIT2 : TInsertMode::SPLIT4;
    TINSERT<MODE>(dst, src);
    for (int row = 0; row < 16; ++row) {
        for (int col = 0; col < 64; ++col) {
            if (dst.GetElement(row, col) != row * 64 + col) {
                std::cerr << "SPLIT mismatch at row=" << row << " col=" << col << '\n';
                return 1;
            }
        }
    }
    std::cout << "SPLIT" << SPLIT_COUNT << " copied all 1024 elements correctly\n";
    return 0;
}
