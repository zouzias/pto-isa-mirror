/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TAND_SUBTILE_HPP
#define TAND_SUBTILE_HPP

#include <pto/common/constants.hpp>
#include <pto/common/utils.hpp>
#include "pto/npu/a2a3/subtile/subtile_tile.hpp"

namespace pto {

// Subtile TAND: 1D uses counter mode; 2D uses mask + hw repeat

template <typename T>
PTO_INTERNAL void TAND_SUBTILE_IMPL_1D(Subtile1D<T> &dst, Subtile1D<T> &src0, Subtile1D<T> &src1)
{
    PTO_ASSERT(dst.length() == src0.length() && dst.length() == src1.length(),
               "Subtile 1D: length mismatch.");
    set_mask_count();
    SetVectorCount(dst.length());
    if constexpr (sizeof(T) == 2) {
        vand((__ubuf__ uint16_t *)dst.data(), (__ubuf__ uint16_t *)src0.data(), (__ubuf__ uint16_t *)src1.data(), 0);
    } else {
        vand(dst.data(), src0.data(), src1.data(), 0);
    }
    set_mask_norm();
    SetFullVecMaskByDType<T>();
}

template <typename T>
PTO_INTERNAL void TAND_SUBTILE_IMPL_2D(Subtile2D<T> &dst, Subtile2D<T> &src0, Subtile2D<T> &src1)
{
    PTO_ASSERT(dst.getRows() == src0.getRows() && dst.getRows() == src1.getRows(),
               "Subtile 2D: rows mismatch.");
    PTO_ASSERT(dst.getCols() == src0.getCols() && dst.getCols() == src1.getCols(),
               "Subtile 2D: cols mismatch.");
    constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(T);
    constexpr unsigned blockSizeElem = BLOCK_BYTE_SIZE / sizeof(T);
    PTO_ASSERT(dst.getCols() <= elementsPerRepeat, "Subtile 2D: cols must be <= VL.");
    PTO_ASSERT(dst.getRows() <= REPEAT_MAX, "Subtile 2D: rows must be <= REPEAT_MAX.");
    uint8_t repeatStride = dst.getRowStride() / blockSizeElem;
    PTO_ASSERT(repeatStride <= REPEAT_STRIDE_MAX, "Subtile 2D: repeat stride too large.");

    SetContMaskByDType<T>(dst.getCols());
    if constexpr (sizeof(T) == 2) {
        vand((__ubuf__ uint16_t *)dst.data(), (__ubuf__ uint16_t *)src0.data(), (__ubuf__ uint16_t *)src1.data(),
             (uint8_t)dst.getRows(), 1, 1, 1, repeatStride, repeatStride, repeatStride);
    } else {
        vand(dst.data(), src0.data(), src1.data(), (uint8_t)dst.getRows(), 1, 1, 1, repeatStride, repeatStride,
             repeatStride);
    }
    SetFullVecMaskByDType<T>();
}

} // namespace pto

#endif // TAND_SUBTILE_HPP
