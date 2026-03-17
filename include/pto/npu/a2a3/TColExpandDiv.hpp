/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TCOLEXPANDDIV_HPP
#define TCOLEXPANDDIV_HPP

#include <pto/common/constants.hpp>
#include <pto/common/utils.hpp>
#include <pto/npu/a2a3/TColExpandBinOp.hpp>

namespace pto {

template <typename T>
struct ColExpandDivOp {
    PTO_INTERNAL static void ColExpandBinInstr(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1, uint8_t repeats)
    {
        vdiv(dst, src0, src1, repeats, 1, 1, 1, 8, 8, 8);
    }
    PTO_INTERNAL static void ColExpandBinInstr(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1, uint8_t repeats,
                                               uint8_t dstRepeatStride, uint8_t src0RepeatStride,
                                               uint8_t src1RepeatStride)
    {
        vdiv(dst, src0, src1, repeats, 1, 1, 1, dstRepeatStride, 0, src0RepeatStride);
    }
};

template <typename T>
struct ColExpandDivOp2 {
    PTO_INTERNAL static void ColExpandBinInstr(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1, uint8_t repeats)
    {
        vdiv(dst, src1, src0, repeats, 1, 1, 1, 8, 8, 8);
    }
    PTO_INTERNAL static void ColExpandBinInstr(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1, uint8_t repeats,
                                               uint8_t dstRepeatStride, uint8_t src0RepeatStride,
                                               uint8_t src1RepeatStride)
    {
        vdiv(dst, src1, src0, repeats, 1, 1, 1, dstRepeatStride, 0, src0RepeatStride);
    }
};

template <typename TileData, typename TileDataSrc0, typename TileDataSrc1>
PTO_INTERNAL void TCOLEXPANDDIV_IMPL(TileData &dst, TileDataSrc0 &src0, TileDataSrc1 &src1)
{
    using T = typename TileData::DType;
    static_assert(
        std::is_same<typename TileData::DType, float>::value || std::is_same<typename TileData::DType, half>::value,
        "Fix: TCOLEXPANDDIV Invalid data type.");
    static_assert(TileData::isRowMajor, "Fix: TCOLEXPANDDIV not supported Layout type");
    constexpr unsigned blockSizeElem = BLOCK_BYTE_SIZE / sizeof(typename TileData::DType);
    constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(typename TileData::DType);
    constexpr unsigned rowStride = TileData::RowStride;
    unsigned validRow = dst.GetValidRow();
    unsigned validCol = dst.GetValidCol();
    unsigned src0ValidRow = src0.GetValidRow();
    unsigned src0ValidCol = src0.GetValidCol();
    unsigned src1ValidRow = src1.GetValidRow();
    unsigned src1ValidCol = src1.GetValidCol();
    bool src0eqdst = (validRow == src0ValidRow) && (validCol == src0ValidCol);
    bool src1eqdst = (validRow == src1ValidRow) && (validCol == src1ValidCol);

    if (src0eqdst) {
        ColExpandBinaryInstr<ColExpandDivOp<T>, TileData, TileDataSrc1, elementsPerRepeat, blockSizeElem, rowStride>(
            dst.data(), src0.data(), src1.data(), validRow, validCol);
    } else {
        ColExpandBinaryInstr<ColExpandDivOp2<T>, TileData, TileDataSrc0, elementsPerRepeat, blockSizeElem, rowStride>(
            dst.data(), src1.data(), src0.data(), validRow, validCol);
    }
}
} // namespace pto
#endif