/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TCONCAT_HPP
#define TCONCAT_HPP

#include <pto/common/constants.hpp>
#include <pto/common/utils.hpp>

namespace pto {

template <typename T, typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1>
PTO_INTERNAL void TConcatCheck(const TileDataDst &dst, const TileDataSrc0 &src0, const TileDataSrc1 &src1)
{
    static_assert(
        std::is_same<T, typename TileDataSrc0::DType>::value && std::is_same<T, typename TileDataSrc1::DType>::value,
        "Fix: TCONCAT the data type of dst must be consistent with src0 and src1.");
    static_assert(std::is_same<T, int32_t>::value || std::is_same<T, uint32_t>::value ||
                      std::is_same<T, float>::value || std::is_same<T, int16_t>::value ||
                      std::is_same<T, uint16_t>::value || std::is_same<T, half>::value ||
                      std::is_same<T, bfloat16_t>::value || std::is_same<T, uint8_t>::value ||
                      std::is_same<T, int8_t>::value,
                  "Fix: TCONCAT has invalid data type.");
    static_assert(TileDataDst::isRowMajor && TileDataSrc0::isRowMajor && TileDataSrc1::isRowMajor,
                  "Fix: TCONCAT only support row major layout.");
    unsigned dstValidRows = dst.GetValidRow();
    unsigned src0ValidRows = src0.GetValidRow();
    unsigned src1ValidRows = src1.GetValidRow();
    PTO_ASSERT(src0ValidRows == dstValidRows && src1ValidRows == dstValidRows,
               "Fix: TCONCAT input tile rows must match dst rows.");
    unsigned dstValidCols = dst.GetValidCol();
    unsigned src0ValidCols = src0.GetValidCol();
    unsigned src1ValidCols = src1.GetValidCol();
    PTO_ASSERT(dstValidCols == src0ValidCols + src1ValidCols,
               "Fix: TCONCAT dst cols must equal src0 cols plus src1 cols.");
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, unsigned blockSizeElem,
          unsigned dstRowStride, unsigned src0RowStride, unsigned src1RowStride>
__tf__ PTO_INTERNAL void TConcat(typename TileDataDst::TileDType __out__ dst,
                                 typename TileDataSrc0::TileDType __in__ src0,
                                 typename TileDataSrc1::TileDType __in__ src1, unsigned validRows,
                                 unsigned src0ValidCols, unsigned src1ValidCols)
{
    using T = typename TileDataDst::DType;
    __ubuf__ T *dstPtr = (__ubuf__ T *)__cce_get_tile_ptr(dst);
    __ubuf__ T *src0Ptr = (__ubuf__ T *)__cce_get_tile_ptr(src0);
    __ubuf__ T *src1Ptr = (__ubuf__ T *)__cce_get_tile_ptr(src1);

    unsigned src0BlockLen = (src0ValidCols * sizeof(T) + BLOCK_BYTE_SIZE - 1) / BLOCK_BYTE_SIZE;
    unsigned src1BlockLen = (src1ValidCols * sizeof(T) + BLOCK_BYTE_SIZE - 1) / BLOCK_BYTE_SIZE;
    unsigned dstCols = src0ValidCols + src1ValidCols;
    unsigned dstBlockLen = (dstCols * sizeof(T) + BLOCK_BYTE_SIZE - 1) / BLOCK_BYTE_SIZE;
    unsigned src0Gap = (TileDataSrc0::Cols * sizeof(T) + BLOCK_BYTE_SIZE - 1) / BLOCK_BYTE_SIZE - src0BlockLen;
    unsigned src1Gap = (TileDataSrc1::Cols * sizeof(T) + BLOCK_BYTE_SIZE - 1) / BLOCK_BYTE_SIZE - src1BlockLen;
    unsigned dstGap = (TileDataDst::Cols * sizeof(T) + BLOCK_BYTE_SIZE - 1) / BLOCK_BYTE_SIZE - dstBlockLen;

    for (unsigned i = 0; i < validRows; i++) {
        copy_ubuf_to_ubuf(dstPtr + i * dstRowStride, src0Ptr + i * src0RowStride, 0, 1, src0BlockLen, src0Gap, dstGap);
        copy_ubuf_to_ubuf(dstPtr + i * dstRowStride + src0ValidCols, src1Ptr + i * src1RowStride, 0, 1, src1BlockLen,
                          src1Gap, dstGap);
    }
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1>
PTO_INTERNAL void TCONCAT_IMPL(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1)
{
    using T = typename TileDataDst::DType;
    TConcatCheck<T, TileDataDst, TileDataSrc0, TileDataSrc1>(dst, src0, src1);
    constexpr unsigned blockSizeElem = BLOCK_BYTE_SIZE / sizeof(T);
    constexpr unsigned dstRowStride = TileDataDst::RowStride;
    constexpr unsigned src0RowStride = TileDataSrc0::RowStride;
    constexpr unsigned src1RowStride = TileDataSrc1::RowStride;
    TConcat<TileDataDst, TileDataSrc0, TileDataSrc1, blockSizeElem, dstRowStride, src0RowStride, src1RowStride>(
        dst.data(), src0.data(), src1.data(), dst.GetValidRow(), src0.GetValidCol(), src1.GetValidCol());
}

} // namespace pto
#endif
