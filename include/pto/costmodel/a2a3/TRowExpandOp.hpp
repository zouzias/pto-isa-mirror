/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TROW_EXPAND_OP_HPP
#define TROW_EXPAND_OP_HPP

#include <pto/common/utils.hpp>
#include "pto/costmodel/costmodel_types.hpp"

namespace pto {

template <typename TileDataOut, typename TileDataIn>
PTO_INTERNAL std::vector<CostModelStats> TRowExpand(int validCol, int validRow)
{
    std::vector<CostModelStats> stats;

    using TRANS = B82B16Trait<typename TileDataOut::DType>;
    int transValidCol = TRANS::TransSize(validCol);
    // int transValidCol = validCol;
    stats.emplace_back("mask", 0, transValidCol);
    for (int i = 0; i < validRow; i++) {
        stats.emplace_back("PIPE_V");
        // vector_dup(dstPtr + i * dstStride, tempValue, 0, 1, 1, BLOCK_MAX_PER_REPEAT, 0);
        stats.emplace_back("vector_dup", 1, 1, 0, 1, 1, BLOCK_MAX_PER_REPEAT, 0);
    }
    return stats;
}

template <typename TileDataDst, typename TileDataSrc>
PTO_INTERNAL std::vector<CostModelStats> runRowExpandOp(TileDataDst &dst, TileDataSrc &src)
{
    int validCol = src.GetValidCol();
    int validRow = src.GetValidRow();
    std::vector<CostModelStats> stats;
    if (validCol == 0 || validRow == 0) {
        return stats;
    }

    return TRowExpand<TileDataDst, TileDataSrc>(validCol, validRow);
}

} // namespace pto
#endif
