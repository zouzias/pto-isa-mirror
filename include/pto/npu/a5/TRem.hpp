/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TREM_HPP
#define TREM_HPP

#include <pto/common/constants.hpp>
#include <pto/common/utils.hpp>
#include <pto/npu/a5/common.hpp>
#include <pto/npu/a5/utils.hpp>
#include <pto/npu/a5/TBinOp.hpp>
#include <pto/common/debug.h>

namespace pto {

template <typename T>
struct RemOp {
    PTO_INTERNAL static void BinInstr(
        RegTensor<T> &reg_dst, RegTensor<T> &reg_src0, RegTensor<T> &reg_src1, MaskReg &preg) {
        // if constexpr (std::is_same<T, float>::value) {
        //     RegTensor<int32_t> reg_tmp;
        //     vdiv(reg_dst, reg_src0, reg_src1, preg, MODE_ZEROING);
        //     vcvt(reg_tmp, reg_dst, preg, ROUND_Z, RS_ENABLE);
        //     vcvt(reg_dst, reg_tmp, preg, ROUND_R);
        //     vmul(reg_dst, reg_dst, reg_src1, preg, MODE_ZEROING);
        //     vsub(reg_dst, reg_src0, reg_dst, preg, MODE_ZEROING);
        // } else if constexpr (std::is_same<T, half>::value) {
        //     RegTensor<int32_t> reg_tmp;
        //     RegTensor<float> reg_tmp0;
        //     RegTensor<float> reg_tmp1;
        //     RegTensor<float> reg_tmp2;
        //     vcvt(reg_tmp0, reg_src0, preg, PART_EVEN);
        //     vcvt(reg_tmp1, reg_tmp1, preg, PART_EVEN);
        //     vdiv(reg_tmp2, reg_tmp0, reg_tmp1, preg, MODE_ZEROING);
        //     vcvt(reg_tmp, reg_tmp2, preg, ROUND_Z, RS_ENABLE);
        //     vcvt(reg_tmp2, reg_tmp, preg, ROUND_R);
        //     vmul(reg_tmp2, reg_tmp2, reg_tmp1, preg, MODE_ZEROING);
        //     vsub(reg_tmp2, reg_tmp0, reg_tmp2, preg, MODE_ZEROING);
        //     vcvt(reg_dst, reg_tmp2, preg, PART_EVEN);
        // } else {
            vmod(reg_dst, reg_src0, reg_src1, preg, MODE_ZEROING);
        // }
    }
};

template <typename TileData, unsigned elementsPerRepeat, unsigned blockSizeElem, unsigned dstRowStride,
    unsigned src0RowStride = dstRowStride, unsigned src1RowStride = dstRowStride>
__tf__ PTO_INTERNAL OP_NAME(TREM) OP_TYPE(element_wise) void TRem(typename TileData::TileDType __out__ dst,
    typename TileData::TileDType __in__ src0, typename TileData::TileDType __in__ src1, unsigned validRows,
    unsigned validCols, VFImplKind version = VFImplKind::VFIMPL_DEFAULT) {
    using T = typename TileData::DType;
    __ubuf__ T *dstPtr = (__ubuf__ T *)__cce_get_tile_ptr(dst);
    __ubuf__ T *src0Ptr = (__ubuf__ T *)__cce_get_tile_ptr(src0);
    __ubuf__ T *src1Ptr = (__ubuf__ T *)__cce_get_tile_ptr(src1);
    if constexpr (dstRowStride == src0RowStride && dstRowStride == src1RowStride) {
        BinaryInstr<RemOp<T>, TileData, elementsPerRepeat, blockSizeElem, dstRowStride>(
            dstPtr, src0Ptr, src1Ptr, validRows, validCols, version);
    } else {
        BinaryInstr<RemOp<T>, TileData, elementsPerRepeat, blockSizeElem, dstRowStride, src0RowStride, src1RowStride>(
            dstPtr, src0Ptr, src1Ptr, validRows, validCols, version);
    }
    return;
}

template <typename T, typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1>
PTO_INTERNAL void TRemCheck(const TileDataDst &dst, const TileDataSrc0 &src0, const TileDataSrc1 &src1) {
    static_assert(std::is_same<T, uint16_t>::value || std::is_same<T, int16_t>::value ||
        std::is_same<T, uint32_t>::value || std::is_same<T, int32_t>::value, "Fix: TRem has invalid data type.");
    static_assert(TileDataDst::isRowMajor && TileDataSrc0::isRowMajor && TileDataSrc1::isRowMajor,
        "Fix: TRem only support row major layout.");
    unsigned validRows = dst.GetValidRow();
    unsigned validCols = dst.GetValidCol();
    PTO_ASSERT(src0.GetValidRow() == validRows && src0.GetValidCol() == validCols,
        "Fix: TREM input tile src0 valid shape mismatch with output tile dst shape.");
    PTO_ASSERT(src1.GetValidRow() == validRows && src1.GetValidCol() == validCols,
        "Fix: TREM input tile src1 valid shape mismatch with output tile dst shape.");
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1>
PTO_INTERNAL void TREM_IMPL(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, TileDataDst &tmp) {
    using T = typename TileDataDst::DType;
    TRemCheck<T, TileDataDst, TileDataSrc0, TileDataSrc1>(dst, src0, src1);
    constexpr unsigned blockSizeElem = BLOCK_BYTE_SIZE / sizeof(T);
    constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(T);
    // when tileshape of src0, src1 and dst are the same, validRows and validCols are also the same
    if constexpr (std::is_same_v<TileDataDst, TileDataSrc0> && std::is_same_v<TileDataDst, TileDataSrc1>) {
        constexpr unsigned dstRowStride = TileDataDst::RowStride;
        TRem<TileDataDst, elementsPerRepeat, blockSizeElem, dstRowStride>(
            dst.data(), src0.data(), src1.data(), dst.GetValidRow(), dst.GetValidCol());
    } else {
        // when tileshape of src0, src1 and dst are different, validRows and validCols are the same
        constexpr unsigned dstRowStride = TileDataDst::RowStride;
        constexpr unsigned src0RowStride = TileDataSrc0::RowStride;
        constexpr unsigned src1RowStride = TileDataSrc1::RowStride;
        TRem<TileDataDst, elementsPerRepeat, blockSizeElem, dstRowStride, src0RowStride, src1RowStride>(
            dst.data(), src0.data(), src1.data(), dst.GetValidRow(), dst.GetValidCol());
    }
}
} // namespace pto
#endif
