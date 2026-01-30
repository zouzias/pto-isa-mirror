/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_CPU_TGEMV_HPP
#define PTO_CPU_TGEMV_HPP

#include "pto/cpu/TMatmul.hpp"

namespace pto {

// CPU-SIM: implement GEMV family via matmul implementations.
template <typename TileRes, typename TileLeft, typename TileRight>
PTO_INTERNAL void TGEMV_IMPL(TileRes &cMatrix, TileLeft &aMatrix, TileRight &bMatrix)
{
    TMATMUL_IMPL(cMatrix, aMatrix, bMatrix);
}

template <typename TileRes, typename TileLeft, typename TileRight>
PTO_INTERNAL void TGEMV_ACC_IMPL(TileRes &cOutMatrix, TileRes &cInMatrix, TileLeft &aMatrix, TileRight &bMatrix)
{
    TMATMUL_ACC_IMPL(cOutMatrix, cInMatrix, aMatrix, bMatrix);
}

template <typename TileRes, typename TileLeft, typename TileRight, typename TileBias>
PTO_INTERNAL void TGEMV_BIAS_IMPL(TileRes &cMatrix, TileLeft &aMatrix, TileRight &bMatrix, TileBias &biasData)
{
    TMATMUL_BIAS_IMPL(cMatrix, aMatrix, bMatrix, biasData);
}

} // namespace pto

#endif
