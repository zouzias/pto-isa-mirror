/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef TCVT_COSTMODEL_HPP
#define TCVT_COSTMODEL_HPP

#include <pto/common/type.hpp>
#include "pto/costmodel/pto_isa_costmodel.hpp"

namespace pto {

// TCVT: vconv_* (type conversion, PIPE_V vector pipeline).
// See TCvtOp.hpp for cycle formula details.
template <typename TileDataD, typename TileDataS>
PTO_INTERNAL void TCVT_IMPL(TileDataD &dst, TileDataS &src,
                             RoundMode mode = RoundMode::CAST_NONE,
                             SaturationMode satMode = SaturationMode::ON)
{
    using DstT = typename TileDataD::DType;
    auto stats = runCvtOp(dst, src);
    dst.SetCycle(CostModel::GetInstance().PredictCycle<DstT>(stats));
}

} // namespace pto

#endif // TCVT_COSTMODEL_HPP
