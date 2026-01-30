/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_CPU_TPRINT_HPP
#define PTO_CPU_TPRINT_HPP

#include <cstdio>
#include <cstddef>

#include "pto/cpu/tile_offsets.hpp"

namespace pto {

template <typename TileData>
PTO_INTERNAL void TPRINT_IMPL(TileData &src)
{
    const std::size_t rows = static_cast<std::size_t>(src.GetValidRow());
    const std::size_t cols = static_cast<std::size_t>(src.GetValidCol());

    std::fprintf(stderr, "[PTO][CPU] TPRINT valid=%zux%zu\n", rows, cols);
    const std::size_t maxR = rows < 4 ? rows : 4;
    const std::size_t maxC = cols < 8 ? cols : 8;
    for (std::size_t r = 0; r < maxR; ++r) {
        std::fprintf(stderr, "  r=%zu: ", r);
        for (std::size_t c = 0; c < maxC; ++c) {
            const auto v = src.data()[GetTileElementOffset<TileData>(r, c)];
            // Print as double-ish; integral prints fine too.
            std::fprintf(stderr, "%g ", static_cast<double>(v));
        }
        if (maxC < cols) {
            std::fprintf(stderr, "... ");
        }
        std::fprintf(stderr, "\n");
    }
    if (maxR < rows) {
        std::fprintf(stderr, "  ...\n");
    }
}

} // namespace pto

#endif
