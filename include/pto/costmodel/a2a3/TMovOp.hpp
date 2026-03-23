/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef TMOV_OP_HPP
#define TMOV_OP_HPP

#include <vector>
#include "pto/costmodel/costmodel_types.hpp"
#include "pto/costmodel/a2a3/TCopyOp.hpp"

namespace pto {

// TMOV basic path (Vec→Vec): internally calls TCopy → copy_ubuf_to_ubuf (MTE1).
// Delegates to runCopyOp to record the actual copy_ubuf_to_ubuf CCE instruction.
template <typename DstTileData, typename SrcTileData>
PTO_INTERNAL std::vector<CostModelStats> runMovVecOp(DstTileData &dst, SrcTileData &src)
{
    return runCopyOp(dst, src);
}

// TMOV non-Vec paths (Acc→Mat, Mat→Left/Right/Bias/Scaling, etc.):
// Uses cube-pipeline (PIPE_M) or MTE1 instructions (copy_cbuf_to_bt, copy_matrix_cc_to_cbuf, etc.).
// Recorded as PIPE_V placeholder (cycle model TBD for PIPE_M).
PTO_INTERNAL std::vector<CostModelStats> runMovCubeOp()
{
    std::vector<CostModelStats> stats;
    stats.emplace_back("PIPE_V"); // PIPE_M / MTE1 cube-related operations
    return stats;
}

} // namespace pto
#endif // TMOV_OP_HPP
