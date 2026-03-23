/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef TCVT_OP_HPP
#define TCVT_OP_HPP

#include <vector>
#include <pto/common/constants.hpp>
#include "pto/costmodel/costmodel_types.hpp"

namespace pto {

// TCVT: vconv_* (type conversion, PIPE_V vector pipeline).
// Per row: ceil(validCol / elemPerRepeat) vconv calls.
// repeatWidth = max(sizeof(DstT), sizeof(SrcT))
// elemPerRepeat = REPEAT_BYTE / repeatWidth
template <typename TileDataD, typename TileDataS>
PTO_INTERNAL std::vector<CostModelStats> runCvtOp(TileDataD &dst, TileDataS &src)
{
    using DstT = typename TileDataD::DType;
    using SrcT = typename TileDataS::DType;
    constexpr unsigned repeatWidth = (sizeof(DstT) > sizeof(SrcT)) ? sizeof(DstT) : sizeof(SrcT);
    constexpr unsigned elemPerRepeat = REPEAT_BYTE / repeatWidth;

    unsigned validRow = dst.GetValidRow();
    unsigned validCol = dst.GetValidCol();
    unsigned numRepeatPerLine = validCol / elemPerRepeat;
    unsigned numRemainPerLine = validCol % elemPerRepeat;

    std::vector<CostModelStats> stats;
    for (unsigned i = 0; i < validRow; i++) {
        unsigned numLoop = numRepeatPerLine / REPEAT_MAX;
        unsigned remainAfterLoop = numRepeatPerLine % REPEAT_MAX;
        for (unsigned j = 0; j < numLoop; j++) {
            stats.emplace_back("vconv", static_cast<int>(REPEAT_MAX));
        }
        if (remainAfterLoop > 0) {
            stats.emplace_back("vconv", static_cast<int>(remainAfterLoop));
        }
        if (numRemainPerLine > 0) {
            stats.emplace_back("vconv", 1);
        }
    }
    return stats;
}

} // namespace pto
#endif // TCVT_OP_HPP
