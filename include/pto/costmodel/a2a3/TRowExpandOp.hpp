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

#include <pto/common/constants.hpp>
#include "pto/costmodel/costmodel_types.hpp"

namespace pto {

// Models TROWEXPAND by treating the operation as filling all dst elements.
// Total elements = validRow * validCol; divided into full repeats plus an optional masked tail.
template <typename TileDataDst, typename TileDataSrc>
PTO_INTERNAL CostModelStats runRowExpandOp(TileDataDst &dst, TileDataSrc &src)
{
    using T = typename TileDataDst::DType;
    CostModelStats stats;
    constexpr unsigned elemPerRpt = REPEAT_BYTE / sizeof(T);
    unsigned validRow = dst.GetValidRow();
    unsigned validCol = dst.GetValidCol();
    if (validRow == 0 || validCol == 0) {
        return stats;
    }
    unsigned nElem = validRow * validCol;
    unsigned headRepeats = nElem / elemPerRpt;
    unsigned tailElements = nElem % elemPerRpt;
    RecordRepeat(stats, headRepeats);
    if (tailElements > 0) {
        RecordRepeat(stats, 1, true);
    }
    return stats;
}

} // namespace pto
#endif
