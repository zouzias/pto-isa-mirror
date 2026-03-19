/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TCOL_REDUCE_OP_HPP
#define TCOL_REDUCE_OP_HPP

#include <pto/common/constants.hpp>
#include "pto/costmodel/costmodel_types.hpp"

namespace pto {

// Models ColReduceInstr from TColReduceOps.hpp:
// - copy first row: no repeat cost
// - for each subsequent row: one count-mode instruction covering numRepeatPerLine repeats,
//   plus one masked repeat if there is a remainder (numRemainPerLine > 0)
// Each count-mode instruction is modeled as numRepeatPerLine regular repeats.
template <typename TileDataDst, typename TileDataSrc>
PTO_INTERNAL CostModelStats runColReduceOp(TileDataDst &dst, TileDataSrc &src)
{
    using T = typename TileDataSrc::DType;
    CostModelStats stats;
    constexpr unsigned elemPerRpt = REPEAT_BYTE / sizeof(T);
    unsigned validRow = src.GetValidRow();
    unsigned validCol = src.GetValidCol();
    if (validRow <= 1) {
        return stats;
    }
    unsigned numRepeatPerLine = validCol / elemPerRpt;
    unsigned numRemainPerLine = validCol % elemPerRpt;
    if (numRepeatPerLine > 0) {
        RecordRepeat(stats, (validRow - 1) * numRepeatPerLine);
    }
    if (numRemainPerLine > 0) {
        RecordRepeat(stats, validRow - 1, true);
    }
    return stats;
}

} // namespace pto
#endif
