/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TPARTIALADD_HPP
#define TPARTIALADD_HPP

#include <pto/npu/a1/TPartBinOps.hpp>

namespace pto {
template <typename T>
struct PartAddOp {
    PTO_INTERNAL static void PartInstr(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1, uint8_t repeats)
    {
        vadd(dst, src0, src1, repeats, 1, 1, 1, 8, 8, 8);
    }
    PTO_INTERNAL static void PartInstr(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1, uint8_t repeats,
                                       uint8_t dstRepeatStride, uint8_t src0RepeatStride, uint8_t src1RepeatStride)
    {
        vadd(dst, src0, src1, repeats, 1, 1, 1, dstRepeatStride, src0RepeatStride, src1RepeatStride);
    }
};

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, unsigned elementsPerRepeat,
          unsigned blockSizeElem, unsigned dstRowStride, unsigned src0RowStride, unsigned src1RowStride>
__tf__ PTO_INTERNAL void TPartAdd(typename TileDataDst::TileDType __out__ dst,
                                  typename TileDataSrc0::TileDType __in__ src0,
                                  typename TileDataSrc1::TileDType __in__ src1, unsigned src0ValidRow,
                                  unsigned src0ValidCol, unsigned src1ValidRow, unsigned src1ValidCol,
                                  unsigned dstValidRow, unsigned dstValidCol)
{
    using T = typename TileDataDst::DType;
    TPartBinOp<PartAddOp<T>, TileDataDst, TileDataSrc0, TileDataSrc1, elementsPerRepeat, blockSizeElem, dstRowStride,
               src0RowStride, src1RowStride>(dst, src0, src1, src0ValidRow, src0ValidCol, src1ValidRow, src1ValidCol,
                                             dstValidRow, dstValidCol);
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1>
PTO_INTERNAL void TPARTADD_IMPL(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1)
{
    static_assert(
        std::is_same_v<typename TileDataDst::DType, int32_t> || std::is_same_v<typename TileDataDst::DType, int16_t> ||
            std::is_same_v<typename TileDataDst::DType, half> || std::is_same_v<typename TileDataDst::DType, float>,
        "Fix: TPARTADD Invalid data type.");
    TPartBinOpImpl<PartAddOp<typename TileDataDst::DType>, TileDataDst, TileDataSrc0, TileDataSrc1>(dst, src0, src1);
}
} // namespace pto
#endif