/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TROW_REDUCE_OP_HPP
#define TROW_REDUCE_OP_HPP

#include <pto/common/constants.hpp>
#include "pto/costmodel/costmodel_types.hpp"

namespace pto {

// Models TRowReduceInstr from TRowReduceOps.hpp (supports FP16 and FP32 only).
// Three cases:
//  1. validCol <= elemPerRpt (OneRepeatProc):
//     - validCol == elemPerRpt: one count-mode ReduceInstr → RecordRepeat(1)
//     - validCol <  elemPerRpt: one masked ReduceInstr with ceil(validRow/REPEAT_MAX) repeats
//  2. validCol > elemPerRpt (general path):
//     - srcRptPerRow instruction calls (FillTmp + TmpProc + ReduceInstr)
//     - if remainder: ceil(validRow/REPEAT_MAX) additional masked repeats
template <typename TileDataDst, typename TileDataSrc>
PTO_INTERNAL CostModelStats runRowReduceOp(TileDataDst &dst, TileDataSrc &src)
{
    using T = typename TileDataSrc::DType;
    CostModelStats stats;
    constexpr unsigned elemPerRpt = REPEAT_BYTE / sizeof(T);
    unsigned validRow = src.GetValidRow();
    unsigned validCol = src.GetValidCol();
    if (validCol == 0 || validRow == 0) {
        return stats;
    }
    unsigned rowLoops = (validRow + REPEAT_MAX - 1) / REPEAT_MAX;
    if (validCol <= elemPerRpt) {
        if (validCol == elemPerRpt) {
            RecordRepeat(stats, 1);
        } else {
            RecordRepeat(stats, rowLoops, true);
        }
    } else {
        unsigned srcRptPerRow = validCol / elemPerRpt;
        unsigned remain = validCol % elemPerRpt;
        RecordRepeat(stats, srcRptPerRow);
        if (remain > 0) {
            RecordRepeat(stats, rowLoops, true);
        }
    }
    return stats;
}

} // namespace pto
#endif
