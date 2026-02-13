/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TSUB_SUBTILE_HPP
#define TSUB_SUBTILE_HPP

#include <pto/common/constants.hpp>
#include <pto/common/utils.hpp>

namespace pto {

// Subtile 1:1 version of TSub: one subtile -> one vsub
// Assumes full VL (no tail) and row-major contiguous layout.

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1>
PTO_INTERNAL void TSUB_SUBTILE_IMPL(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1)
{
    using T = typename TileDataDst::DType;
    static_assert(std::is_same_v<T, typename TileDataSrc0::DType> && std::is_same_v<T, typename TileDataSrc1::DType>,
                  "Subtile TSub: dst/src0/src1 dtype must match.");
    static_assert(TileDataDst::isRowMajor && TileDataSrc0::isRowMajor && TileDataSrc1::isRowMajor,
                  "Subtile TSub: only supports row-major layout.");

    constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(T);
    static_assert(TileDataDst::Rows * TileDataDst::Cols == elementsPerRepeat,
                  "Subtile TSub: tile must match exactly one VL (Rows*Cols == VL).");

    unsigned validRows = dst.GetValidRow();
    unsigned validCols = dst.GetValidCol();
    PTO_ASSERT(validRows == TileDataDst::Rows && validCols == TileDataDst::Cols,
               "Subtile TSub: valid shape must equal full tile shape.");

    __ubuf__ T *dstPtr = (__ubuf__ T *)__cce_get_tile_ptr(dst.data());
    __ubuf__ T *src0Ptr = (__ubuf__ T *)__cce_get_tile_ptr(src0.data());
    __ubuf__ T *src1Ptr = (__ubuf__ T *)__cce_get_tile_ptr(src1.data());

    // One intrinsic, no loops, no tail handling
    vsub(dstPtr, src0Ptr, src1Ptr, /*repeats=*/1, 1, 1, 1, 8, 8, 8);
}

} // namespace pto

#endif // TSUB_SUBTILE_HPP
