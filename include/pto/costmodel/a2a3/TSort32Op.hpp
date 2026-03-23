/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef TSORT32_OP_HPP
#define TSORT32_OP_HPP

#include <vector>
#include <pto/common/constants.hpp>
#include "pto/costmodel/costmodel_types.hpp"

namespace pto {

// TSORT32: vbitsort(repeatNumPerRow) + PIPE_V per row.
// repeatNumPerRow = validCol / 32
template <typename DstTileData, typename SrcTileData>
PTO_INTERNAL std::vector<CostModelStats> runSort32Op(DstTileData &dst, SrcTileData &src)
{
    constexpr unsigned SORT_BLOCK = 32;
    unsigned validRow = dst.GetValidRow();
    unsigned repeatNumPerRow = src.GetValidCol() / SORT_BLOCK;

    std::vector<CostModelStats> stats;
    for (unsigned i = 0; i < validRow; i++) {
        unsigned numLoop = repeatNumPerRow / REPEAT_MAX;
        unsigned remain = repeatNumPerRow % REPEAT_MAX;
        for (unsigned j = 0; j < numLoop; j++) {
            stats.emplace_back("vbitsort", static_cast<int>(REPEAT_MAX));
            stats.emplace_back("PIPE_V");
        }
        if (remain > 0) {
            stats.emplace_back("vbitsort", static_cast<int>(remain));
            stats.emplace_back("PIPE_V");
        }
    }
    return stats;
}

} // namespace pto
#endif // TSORT32_OP_HPP
