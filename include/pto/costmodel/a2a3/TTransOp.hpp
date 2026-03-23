/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef TTRANS_OP_HPP
#define TTRANS_OP_HPP

#include <vector>
#include <pto/common/constants.hpp>
#include "pto/costmodel/costmodel_types.hpp"

namespace pto {

// TTRANS cycle model — scatter_vnchwconv_b8/b16/b32 (PIPE_V vector pipeline)
// followed by pipe_barrier(PIPE_V) + copy_ubuf_to_ubuf (MTE1).
//
// Layout parameters (element size determines B8/B16/B32 path):
//   blockSizeElem  = 32 / sizeof(T)   (32→B8, 16→B16, 8→B32)
//   yTileSizeElem  = 32 for B8, 16 for B16/B32
//
// Instruction counts:
//   numSubTileX = ceil(validCol / blockSizeElem)
//   numSubTileY = validRow / yTileSizeElem        (full Y-tiles)
//   remainY     = validRow % yTileSizeElem         (tail rows)
//
//   B16/B32: numSubTileX calls of scatter_vnchwconv(numSubTileY)  [full tiles]
//            1 call of scatter_vnchwconv(numSubTileX)              [Y-tail, if remainY > 0]
//   B8:      numSubTileX × 4 calls of scatter_vnchwconv(numSubTileY)
//            4 calls of scatter_vnchwconv(numSubTileX)             [Y-tail, if remainY > 0]
//   Then PIPE_V + PIPE_V (copy_ubuf_to_ubuf, MTE1).

template <typename DstTile, typename SrcTile, typename TmpTile>
PTO_INTERNAL std::vector<CostModelStats> runTransOp(DstTile & /*dst*/, SrcTile &src, TmpTile & /*tmp*/)
{
    using T = typename SrcTile::DType;
    constexpr unsigned BLOCK_BYTE = 32u;
    constexpr unsigned blockSizeElem = BLOCK_BYTE / sizeof(T);
    constexpr unsigned yTileSizeElem = (sizeof(T) == 1u) ? 32u : 16u;

    unsigned validRow = src.GetValidRow();
    unsigned validCol = src.GetValidCol();

    unsigned numSubTileX = (validCol + blockSizeElem - 1u) / blockSizeElem;
    unsigned numSubTileY = validRow / yTileSizeElem;
    unsigned remainY = validRow % yTileSizeElem;

    std::vector<CostModelStats> stats;

    // Full Y-tiles
    if (numSubTileY > 0) {
        if constexpr (sizeof(T) == 1u) {
            for (unsigned i = 0; i < numSubTileX; i++) {
                for (int k = 0; k < 4; k++) {
                    stats.emplace_back("scatter_vnchwconv", static_cast<int>(numSubTileY));
                }
            }
        } else {
            for (unsigned i = 0; i < numSubTileX; i++) {
                stats.emplace_back("scatter_vnchwconv", static_cast<int>(numSubTileY));
            }
        }
    }

    // Y-tail (partial last block of rows)
    if (remainY > 0) {
        if constexpr (sizeof(T) == 1u) {
            for (int k = 0; k < 4; k++) {
                stats.emplace_back("scatter_vnchwconv", static_cast<int>(numSubTileX));
            }
        } else {
            stats.emplace_back("scatter_vnchwconv", static_cast<int>(numSubTileX));
        }
    }

    // pipe_barrier(PIPE_V) then copy_ubuf_to_ubuf (MTE1) to copy the transposed result
    stats.emplace_back("PIPE_V");
    {
        unsigned blockLen = (validRow * validCol * sizeof(T) + BLOCK_BYTE_SIZE - 1) / BLOCK_BYTE_SIZE;
        stats.emplace_back("copy_ubuf_to_ubuf", 1, static_cast<int>(blockLen), 1, 1);
    }

    return stats;
}

} // namespace pto
#endif // TTRANS_OP_HPP
