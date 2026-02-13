/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TXOR_SUBTILE_HPP
#define TXOR_SUBTILE_HPP

#include <pto/common/constants.hpp>
#include <pto/common/utils.hpp>
#include "pto/npu/a2a3/subtile/subtile_tile.hpp"

namespace pto {

// Subtile TXOR: implemented via (a|b) & ~(a&b)
// 1D uses counter mode; 2D uses mask + hw repeat

template <typename T>
PTO_INTERNAL void TXOR_SUBTILE_IMPL_1D(Subtile1D<T> &dst, Subtile1D<T> &src0, Subtile1D<T> &src1, Subtile1D<T> &tmp)
{
    PTO_ASSERT(dst.length() == src0.length() && dst.length() == src1.length() && dst.length() == tmp.length(),
               "Subtile 1D: length mismatch.");
    set_mask_count();
    SetVectorCount(dst.length());
    if constexpr (sizeof(T) == 2) {
        auto d = (__ubuf__ uint16_t *)dst.data();
        auto s0 = (__ubuf__ uint16_t *)src0.data();
        auto s1 = (__ubuf__ uint16_t *)src1.data();
        auto t = (__ubuf__ uint16_t *)tmp.data();
        vor(t, s0, s1, 0);
        vand(d, s0, s1, 0);
        vnot(d, d, 0, 1, 1, 8, 8);
        vand(d, d, t, 0);
    } else {
        vor(dst.data(), src0.data(), src1.data(), 0); // reuse dst as tmp
        vand(tmp.data(), src0.data(), src1.data(), 0);
        vnot(tmp.data(), tmp.data(), 0, 1, 1, 8, 8);
        vand(dst.data(), dst.data(), tmp.data(), 0);
    }
    set_mask_norm();
    SetFullVecMaskByDType<T>();
}

template <typename T>
PTO_INTERNAL void TXOR_SUBTILE_IMPL_2D(Subtile2D<T> &dst, Subtile2D<T> &src0, Subtile2D<T> &src1, Subtile2D<T> &tmp)
{
    PTO_ASSERT(dst.getRows() == src0.getRows() && dst.getRows() == src1.getRows() && dst.getRows() == tmp.getRows(),
               "Subtile 2D: rows mismatch.");
    PTO_ASSERT(dst.getCols() == src0.getCols() && dst.getCols() == src1.getCols() && dst.getCols() == tmp.getCols(),
               "Subtile 2D: cols mismatch.");
    constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(T);
    constexpr unsigned blockSizeElem = BLOCK_BYTE_SIZE / sizeof(T);
    PTO_ASSERT(dst.getCols() <= elementsPerRepeat, "Subtile 2D: cols must be <= VL.");
    PTO_ASSERT(dst.getRows() <= REPEAT_MAX, "Subtile 2D: rows must be <= REPEAT_MAX.");
    uint8_t repeatStride = dst.getRowStride() / blockSizeElem;
    PTO_ASSERT(repeatStride <= REPEAT_STRIDE_MAX, "Subtile 2D: repeat stride too large.");

    SetContMaskByDType<T>(dst.getCols());
    if constexpr (sizeof(T) == 2) {
        auto d = (__ubuf__ uint16_t *)dst.data();
        auto s0 = (__ubuf__ uint16_t *)src0.data();
        auto s1 = (__ubuf__ uint16_t *)src1.data();
        auto t = (__ubuf__ uint16_t *)tmp.data();
        vor(t, s0, s1, (uint8_t)dst.getRows(), 1, 1, 1, repeatStride, repeatStride, repeatStride);
        vand(d, s0, s1, (uint8_t)dst.getRows(), 1, 1, 1, repeatStride, repeatStride, repeatStride);
        vnot(d, d, (uint8_t)dst.getRows(), 1, 1, repeatStride, repeatStride);
        vand(d, d, t, (uint8_t)dst.getRows(), 1, 1, 1, repeatStride, repeatStride, repeatStride);
    } else {
        vor(tmp.data(), src0.data(), src1.data(), (uint8_t)dst.getRows(), 1, 1, 1, repeatStride, repeatStride,
            repeatStride);
        vand(dst.data(), src0.data(), src1.data(), (uint8_t)dst.getRows(), 1, 1, 1, repeatStride, repeatStride,
            repeatStride);
        vnot(dst.data(), dst.data(), (uint8_t)dst.getRows(), 1, 1, repeatStride, repeatStride);
        vand(dst.data(), dst.data(), tmp.data(), (uint8_t)dst.getRows(), 1, 1, 1, repeatStride, repeatStride,
            repeatStride);
    }
    SetFullVecMaskByDType<T>();
}

} // namespace pto

#endif // TXOR_SUBTILE_HPP
