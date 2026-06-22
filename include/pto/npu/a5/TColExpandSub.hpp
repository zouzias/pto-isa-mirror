/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TCOLEXPANDSUB_HPP
#define TCOLEXPANDSUB_HPP

#include <pto/common/constants.hpp>
#include <pto/common/utils.hpp>
#include "common.hpp"
#include "utils.hpp"
#include "TColExpandBinOp.hpp"

namespace pto {

template <typename T>
struct ColExpandSubOp {
    PTO_INTERNAL static void ColExpandBinaryInstr(RegTensor<T> &reg_dst, RegTensor<T> &reg_src0, RegTensor<T> &reg_src1,
                                                  MaskReg &preg)
    {
        vsub(reg_dst, reg_src0, reg_src1, preg, MODE_ZEROING);
    }
};

template <typename T>
struct ColExpandSubOp2 {
    PTO_INTERNAL static void ColExpandBinaryInstr(RegTensor<T> &reg_dst, RegTensor<T> &reg_src0, RegTensor<T> &reg_src1,
                                                  MaskReg &preg)
    {
        vsub(reg_dst, reg_src1, reg_src0, preg, MODE_ZEROING);
    }
};

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, unsigned elementsPerRepeat,
          unsigned blockSizeElem, unsigned rowStride>
__tf__ PTO_INTERNAL OP_NAME(TCOLEXPANDSUB)
    OP_TYPE(broadcast) void TColExpandSub(typename TileDataDst::TileDType __out__ dst,
                                          typename TileDataSrc0::TileDType __in__ src0,
                                          typename TileDataSrc1::TileDType __in__ src1, bool src0eqdst,
                                          unsigned validRow, unsigned validCol,
                                          unsigned version = VFImplKind::VFIMPL_DEFAULT)
{
    using T = typename TileDataDst::DType;
    __ubuf__ T *dstPtr = (__ubuf__ T *)__cce_get_tile_ptr(dst);
    __ubuf__ T *src0Ptr = (__ubuf__ T *)__cce_get_tile_ptr(src0);
    __ubuf__ T *src1Ptr = (__ubuf__ T *)__cce_get_tile_ptr(src1);

    if (src0eqdst) {
        ColExpandBinaryInstr<ColExpandSubOp<T>, TileDataDst, TileDataSrc1, elementsPerRepeat, blockSizeElem, rowStride>(
            dstPtr, src0Ptr, src1Ptr, validRow, validCol);
    } else {
        ColExpandBinaryInstr<ColExpandSubOp2<T>, TileDataDst, TileDataSrc0, elementsPerRepeat, blockSizeElem,
                             rowStride>(dstPtr, src1Ptr, src0Ptr, validRow, validCol);
    }
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1>
PTO_INTERNAL void TCOLEXPANDSUB_IMPL(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1)
{
    using T = typename TileDataDst::DType;
    static_assert(std::is_same_v<T, typename TileDataSrc0::DType> && std::is_same_v<T, typename TileDataSrc1::DType>,
                  "Fix: TCOLEXPANDSUB src and dst data type is different!");
    static_assert(std::is_same_v<T, int32_t> || std::is_same_v<T, uint32_t> || std::is_same_v<T, float> ||
                      std::is_same_v<T, int16_t> || std::is_same_v<T, uint16_t> || std::is_same_v<T, half> ||
                      std::is_same_v<T, bfloat16_t> || std::is_same_v<T, uint8_t> || std::is_same_v<T, int8_t>,
                  "Fix: TCOLEXPANDSUB Invalid data type.");
    static_assert(TileDataDst::isRowMajor, "Fix: TCOLEXPANDSUB Invalid tile shape.");

    unsigned validRow = dst.GetValidRow();
    unsigned validCol = dst.GetValidCol();
    unsigned src0ValidRow = src0.GetValidRow();
    unsigned src0ValidCol = src0.GetValidCol();
    unsigned src1ValidRow = src1.GetValidRow();
    unsigned src1ValidCol = src1.GetValidCol();
    bool src0eqdst = (validRow == src0ValidRow) && (validCol == src0ValidCol);
    bool src1eqdst = (validRow == src1ValidRow) && (validCol == src1ValidCol);
    PTO_ASSERT((src0eqdst && TileDataSrc0::isRowMajor) || (src1eqdst && TileDataSrc1::isRowMajor),
               "TCOLEXPANDSUB: the validShape of src0 or src1 should be equal to dst");

    constexpr unsigned blockSizeElem = BLOCK_BYTE_SIZE / sizeof(T);
    constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(T);
    constexpr unsigned rowStride = TileDataDst::RowStride;

    if (src0eqdst) {
        TColExpandSub<TileDataDst, TileDataSrc0, TileDataSrc1, elementsPerRepeat, blockSizeElem, rowStride>(
            dst.data(), src0.data(), src1.data(), src0eqdst, validRow, validCol);
    } else {
        TColExpandSub<TileDataDst, TileDataSrc1, TileDataSrc0, elementsPerRepeat, blockSizeElem, rowStride>(
            dst.data(), src1.data(), src0.data(), src0eqdst, validRow, validCol);
    }
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename TileDataTmp>
PTO_INTERNAL void TCOLEXPANDSUB_IMPL(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1,
                                     [[maybe_unused]] TileDataTmp &tmp)
{
    TCOLEXPANDSUB_IMPL(dst, src0, src1);
}
} // namespace pto
#endif
