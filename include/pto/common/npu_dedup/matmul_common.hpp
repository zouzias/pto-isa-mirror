/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMMON_NPU_DEDUP_MATMUL_COMMON_HPP
#define PTO_COMMON_NPU_DEDUP_MATMUL_COMMON_HPP

#include <cstdint>

namespace pto {

inline namespace TMatmulInternel {
constexpr const int MMAD_MAX_SUPPORT_LENGTH = 4095;
} // namespace TMatmulInternel

// Requires <pto/common/utils.hpp> or an equivalent PTO_ASSERT definition to be included before this header.
PTO_INTERNAL void CheckDynamicMmad(uint16_t aMatrixRow, uint16_t aMatrixCol, uint16_t bMatrixCol)
{
    PTO_ASSERT(aMatrixRow >= 1 && aMatrixRow <= MMAD_MAX_SUPPORT_LENGTH,
               "ERROR: The range of valid aMatrixRow is [1, 4095].");
    PTO_ASSERT(aMatrixCol >= 1 && aMatrixCol <= MMAD_MAX_SUPPORT_LENGTH,
               "ERROR: The range of valid aMatrixCol is [1, 4095].");
    PTO_ASSERT(bMatrixCol >= 1 && bMatrixCol <= MMAD_MAX_SUPPORT_LENGTH,
               "ERROR: The range of valid bMatrixCol is [1, 4095].");
}

} // namespace pto

#endif
