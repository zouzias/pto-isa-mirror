/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef TSEL_OP_HPP
#define TSEL_OP_HPP

#include <vector>
#include "pto/costmodel/costmodel_types.hpp"

namespace pto {

// TSEL per-row pattern: vector_dup(1) + PIPE_V + vsel(1)
// Per row: vector_dup loads the compare-mask, then vsel selects elements.
template <typename DstTile>
PTO_INTERNAL std::vector<CostModelStats> runSelOp(DstTile &dst)
{
    unsigned validRow = dst.GetValidRow();
    std::vector<CostModelStats> stats;
    for (unsigned i = 0; i < validRow; i++) {
        stats.emplace_back("vector_dup", 1);
        stats.emplace_back("PIPE_V");
        stats.emplace_back("vsel", 1);
    }
    return stats;
}

} // namespace pto
#endif // TSEL_OP_HPP
