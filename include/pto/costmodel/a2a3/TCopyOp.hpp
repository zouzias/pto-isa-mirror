/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef TCOPY_OP_HPP
#define TCOPY_OP_HPP

#include <vector>
#include <pto/common/constants.hpp>
#include "pto/costmodel/costmodel_types.hpp"

namespace pto {

// TCOPY: copy_ubuf_to_ubuf (MTE1 pipeline).
// Records the actual CCE instruction name and nBurst/lenBurst/srcGap/dstGap parameters,
// derived from tile shapes the same way the NPU TCopy template does.
//
// Contiguous path (DstTile::Cols == SrcTile::Cols or single row):
//   One burst: nBurst=1, lenBurst = ceil(Cols*validRow*sizeof(T)/32), srcGap=1, dstGap=1
//
// Row-by-row path (columns differ):
//   validRow bursts: nBurst=1, lenBurst = ceil(validCol*sizeof(T)/32), per row
template <typename TileDataD, typename TileDataS>
PTO_INTERNAL std::vector<CostModelStats> runCopyOp(TileDataD &dst, TileDataS &src)
{
    using T = typename TileDataD::DType;
    unsigned validRow = static_cast<unsigned>(dst.GetValidRow());
    unsigned validCol = static_cast<unsigned>(dst.GetValidCol());

    std::vector<CostModelStats> stats;

    if constexpr (TileDataD::Cols == TileDataS::Cols || TileDataD::Rows == 1) {
        // Contiguous copy: one copy_ubuf_to_ubuf burst covering all rows
        unsigned blockLen =
            (static_cast<unsigned>(TileDataD::Cols) * validRow * sizeof(T) + BLOCK_BYTE_SIZE - 1) / BLOCK_BYTE_SIZE;
        stats.emplace_back("copy_ubuf_to_ubuf", 1, static_cast<int>(blockLen), 1, 1);
    } else {
        // Row-by-row copy: one copy_ubuf_to_ubuf call per row
        unsigned blockLen = (validCol * sizeof(T) + BLOCK_BYTE_SIZE - 1) / BLOCK_BYTE_SIZE;
        unsigned srcGap =
            (static_cast<unsigned>(TileDataS::Cols) * sizeof(T) + BLOCK_BYTE_SIZE - 1) / BLOCK_BYTE_SIZE - blockLen;
        unsigned dstGap =
            (static_cast<unsigned>(TileDataD::Cols) * sizeof(T) + BLOCK_BYTE_SIZE - 1) / BLOCK_BYTE_SIZE - blockLen;
        for (unsigned i = 0; i < validRow; i++) {
            stats.emplace_back("copy_ubuf_to_ubuf", 1, static_cast<int>(blockLen), static_cast<int>(srcGap),
                               static_cast<int>(dstGap));
        }
    }
    return stats;
}

} // namespace pto
#endif // TCOPY_OP_HPP
