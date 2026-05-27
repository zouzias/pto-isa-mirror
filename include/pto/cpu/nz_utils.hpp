/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef NZ_UTILS_HPP
#define NZ_UTILS_HPP

#include <cstddef>

#include "pto/cpu/parallel.hpp"
#include "common.hpp"

namespace pto {

template <typename Dummy = void>
PTO_INLINE size_t GetNZGlobalOffset(
    size_t r,
    size_t c,
    int gShape4)
{
    constexpr size_t innerRows = 16;
    constexpr size_t innerCols = 8;

    size_t blockRow = r / innerRows;
    size_t innerRow = r % innerRows;

    size_t blockCol = c / innerCols;
    size_t innerCol = c % innerCols;

    size_t numBlockCols =
        (gShape4 + innerCols - 1) / innerCols;

    return
        blockRow * numBlockCols * innerRows * innerCols +
        blockCol * innerRows * innerCols +
        innerRow * innerCols +
        innerCol;
}

template <typename TileData, typename Func>
__tf__ PTO_INLINE void ForEachNZElement(
    int gShape3,
    int gShape4,
    Func&& fn)
{
    cpu::parallel_for_1d(
        0,
        static_cast<std::size_t>(gShape4),
        static_cast<std::size_t>(gShape3) * gShape4,
        [&](std::size_t c) {

            size_t subTileC =
                c / TileData::InnerCols;

            size_t innerC =
                c % TileData::InnerCols;

            for (size_t r = 0;
                 r < static_cast<std::size_t>(gShape3);
                 r++) {

                size_t subTileR =
                    r / TileData::InnerRows;

                size_t innerR =
                    r % TileData::InnerRows;

                size_t tile_idx =
                    GetTileElementOffsetSubfractals<TileData>(
                        subTileR,
                        innerR,
                        subTileC,
                        innerC);

                size_t nz_idx =
                    GetNZGlobalOffset<>(
                        r,
                        c,
                        gShape4);

                fn(r, c, tile_idx, nz_idx);
            }
        });
}

} // namespace pto

#endif