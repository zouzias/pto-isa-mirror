/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef TSEL_COSTMODEL_HPP
#define TSEL_COSTMODEL_HPP

#include "pto/costmodel/pto_isa_costmodel.hpp"

namespace pto {

// TSEL: per-row pattern: vector_dup(1) + PIPE_V + vsel(1).
// See TSelOp.hpp for cycle formula details.
template <typename DstTile, typename MaskTile, typename Src0Tile, typename Src1Tile, typename TmpTile>
PTO_INTERNAL void TSEL_IMPL(DstTile &dst, MaskTile &selMask, Src0Tile &src0, Src1Tile &src1, TmpTile &tmp)
{
    using T = typename DstTile::DType;
    auto stats = runSelOp(dst);
    dst.SetCycle(CostModel::GetInstance().PredictCycle<T>(stats));
}

} // namespace pto

#endif // TSEL_COSTMODEL_HPP
