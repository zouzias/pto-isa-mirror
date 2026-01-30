/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_CPU_TMATMUL_MX_HPP
#define PTO_CPU_TMATMUL_MX_HPP

#include "pto/cpu/TMatmul.hpp"

namespace pto {

// CPU-SIM: TMATMUL_MX ignores scaling tiles and falls back to the regular matmul path.
template <typename TileRes, typename TileLeft, typename TileLeftScale, typename TileRight, typename TileRightScale>
PTO_INTERNAL void TMATMUL_MX_IMPL(TileRes &cMatrix,
                                 TileLeft &aMatrix,
                                 TileLeftScale &aScaleMatrix,
                                 TileRight &bMatrix,
                                 TileRightScale &bScaleMatrix)
{
    (void)aScaleMatrix;
    (void)bScaleMatrix;
    TMATMUL_IMPL(cMatrix, aMatrix, bMatrix);
}

template <typename TileRes, typename TileLeft, typename TileLeftScale, typename TileRight, typename TileRightScale>
PTO_INTERNAL void TMATMUL_MX_IMPL(TileRes &cOutMatrix,
                                 TileRes &cInMatrix,
                                 TileLeft &aMatrix,
                                 TileLeftScale &aScaleMatrix,
                                 TileRight &bMatrix,
                                 TileRightScale &bScaleMatrix)
{
    (void)aScaleMatrix;
    (void)bScaleMatrix;
    TMATMUL_ACC_IMPL(cOutMatrix, cInMatrix, aMatrix, bMatrix);
}

} // namespace pto

#endif
