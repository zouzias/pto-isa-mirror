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
    size_t linearRow,
    size_t linearCol,
    size_t logicalCols)
{
    constexpr size_t innerRows = FRACTAL_NZ_ROW; // 16
    constexpr size_t innerCols = FRACTAL_NZ_COL; // 8

    const size_t blockRow = linearRow / innerRows;
    const size_t innerRow = linearRow % innerRows;

    const size_t blockCol = linearCol / innerCols;
    const size_t innerCol = linearCol % innerCols;

    const size_t numBlockCols =
        (logicalCols + innerCols - 1) / innerCols;

    return
        blockRow * numBlockCols * innerRows * innerCols +
        blockCol * innerRows * innerCols +
        innerRow * innerCols +
        innerCol;
}

/*
 * Universal NZ traversal helper.
 *
 * Traverses logical tensor space:
 *
 * [gShape0][gShape1][gShape2][gShape3][gShape4]
 *
 * and maps it into:
 *
 * - Tile subfractal indices
 * - NZ global tensor offset
 *
 * Callback signature:
 *
 * fn(
 *     logicalRow,
 *     logicalCol,
 *     tile_idx,
 *     nz_idx)
 */
template <typename TileData, typename Func>
__tf__ PTO_INLINE void ForEachNZTensorElement(
    int gShape0,
    int gShape1,
    int gShape2,
    int gShape3,
    int gShape4,
    Func&& fn)
{
    constexpr size_t innerRows = FRACTAL_NZ_ROW;
    constexpr size_t innerCols = FRACTAL_NZ_COL;

    const size_t logicalRows =
        static_cast<size_t>(gShape0) *
        gShape1 *
        gShape2 *
        gShape3;

    const size_t logicalCols =
        static_cast<size_t>(gShape4);

    cpu::parallel_for_1d(
        0,
        logicalCols,
        logicalRows * logicalCols,
        [&](std::size_t c) {

            const size_t subTileC =
                c / innerCols;

            const size_t innerC =
                c % innerCols;

            for (size_t r = 0;
                 r < logicalRows;
                 r++) {

                const size_t subTileR =
                    r / innerRows;

                const size_t innerR =
                    r % innerRows;

                const size_t tile_idx =
                    GetTileElementOffsetSubfractals<
                        TileData>(
                            subTileR,
                            innerR,
                            subTileC,
                            innerC);

                const size_t nz_idx =
                    GetNZGlobalOffset<>(
                        r,
                        c,
                        logicalCols);

                fn(
                    r,
                    c,
                    tile_idx,
                    nz_idx);
            }
        });
}

template <typename TileData>
PTO_INLINE size_t GetNZLogicalRows(
    int gShape0,
    int gShape1,
    int gShape2,
    int gShape3)
{
    return
        static_cast<size_t>(gShape0) *
        gShape1 *
        gShape2 *
        gShape3;
}

template <typename TileData>
PTO_INLINE size_t GetNZLogicalCols(
    int gShape4)
{
    return static_cast<size_t>(gShape4);
}

} // namespace pto

#endif