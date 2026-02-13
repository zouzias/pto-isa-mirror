/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_SUBTILE_BRCB_HPP
#define PTO_SUBTILE_BRCB_HPP

#include <pto/common/constants.hpp>
#include <pto/common/utils.hpp>
#include <pto/npu/a2a3/subtile/subtile_tile.hpp>

namespace pto {

// SubtileBrcb: broadcast 1 element per row to a 32B block (row-major)
// - src: Subtile1D, length == dst.rows, length must be multiple of 8
// - dst: Subtile2D, cols == elemPerBlock (32B/sizeof(T)), rowStride == cols
// - src/dst must be 32B aligned to satisfy vbrcb requirement

template <typename T>
PTO_INTERNAL void SubtileBrcb(Subtile2D<T> dst, Subtile1D<T> src)
{
    using BrcbType = std::conditional_t<sizeof(T) == sizeof(uint16_t), uint16_t,
                                        std::conditional_t<sizeof(T) == sizeof(uint32_t), uint32_t, T>>;
    constexpr unsigned elemPerBlock = BLOCK_BYTE_SIZE / sizeof(T);
    constexpr unsigned elemPerRepeat = REPEAT_BYTE / sizeof(T);
    constexpr unsigned vbrcbElem = 8;

    PTO_ASSERT(dst.getCols() == elemPerBlock, "SubtileBrcb: dst.cols must be 32B/sizeof(T).");
    PTO_ASSERT(dst.getRowStride() == elemPerBlock, "SubtileBrcb: dst.rowStride must be contiguous.");
    PTO_ASSERT(src.length() == dst.getRows(), "SubtileBrcb: src length must equal dst rows.");
    PTO_ASSERT((src.length() % vbrcbElem) == 0, "SubtileBrcb: src length must be multiple of 8.");

    __ubuf__ BrcbType *dstPtr = (__ubuf__ BrcbType *)dst.data();
    __ubuf__ BrcbType *srcPtr = (__ubuf__ BrcbType *)src.data();

    unsigned repeat = src.length() / vbrcbElem;
    unsigned loop = repeat / (REPEAT_MAX - 1);
    unsigned remain = repeat % (REPEAT_MAX - 1);

    for (unsigned i = 0; i < loop; ++i) {
        vbrcb(dstPtr + i * (REPEAT_MAX - 1) * elemPerRepeat,
              srcPtr + i * (REPEAT_MAX - 1) * vbrcbElem,
              1, BLOCK_MAX_PER_REPEAT, (REPEAT_MAX - 1));
    }
    if (remain > 0) {
        vbrcb(dstPtr + loop * (REPEAT_MAX - 1) * elemPerRepeat,
              srcPtr + loop * (REPEAT_MAX - 1) * vbrcbElem,
              1, BLOCK_MAX_PER_REPEAT, remain);
    }
}

} // namespace pto

#endif // PTO_SUBTILE_BRCB_HPP
