/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TREMS_HPP_KIRIN9030
#define TREMS_HPP_KIRIN9030

#include <pto/common/constants.hpp>
#include <pto/common/utils.hpp>
#include <pto/npu/a5/TBinSOp.hpp>
#include <pto/npu/kirin9030/TRem.hpp>
#include <pto/npu/a5/TRemS.hpp>

namespace pto {
namespace kirin9030 {

template <typename T>
struct RemSOp {
    static constexpr bool isDynFunc = false;
    PTO_INTERNAL static void BinSInstr(RegTensor<T> &dst, RegTensor<T> &src0, T scalar, MaskReg &preg)
    {
        RegTensor<T> src1;
        vdup(src1, scalar, preg, MODE_ZEROING);
        RemOp<T>::BinInstr(dst, src0, src1, preg);
    }
};

template <typename DstTile, typename SrcTile>
__tf__ PTO_INTERNAL OP_NAME(TREMS)
    OP_TYPE(element_wise) void TRemS(typename DstTile::TileDType __out__ dst, typename SrcTile::TileDType __in__ src,
                                     typename SrcTile::DType scalar, unsigned kValidRows, unsigned kValidCols,
                                     VFImplKind version = VFImplKind::VFIMPL_DEFAULT)
{
    using T = typename DstTile::DType;
    __ubuf__ T *dstPtr = (__ubuf__ T *)__cce_get_tile_ptr(dst);
    __ubuf__ T *srcPtr = (__ubuf__ T *)__cce_get_tile_ptr(src);

    constexpr unsigned blockSizeElem = CCE_VL / sizeof(T);
    constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(T);
    constexpr unsigned dstRowStride = DstTile::RowStride;
    constexpr unsigned srcRowStride = SrcTile::RowStride;

    a5::BinaryInstr<RemSOp<T>, DstTile, SrcTile, T, elementsPerRepeat, blockSizeElem, dstRowStride, srcRowStride>(
        dstPtr, srcPtr, scalar, kValidRows, kValidCols, version);
}

template <auto PrecisionType = RemSAlgorithm::DEFAULT, typename DstTile, typename SrcTile, typename TileDataTmp>
PTO_INTERNAL void TREMS_IMPL(DstTile &dst, SrcTile &src, typename SrcTile::DType scalar, TileDataTmp &tmp)
{
    using T = typename DstTile::DType;
    unsigned validRow = dst.GetValidRow();
    unsigned validCol = dst.GetValidCol();

    PTO_ASSERT((src.GetValidCol() == validCol) && (src.GetValidRow() == validRow),
               "Number of validColumns and validRows of src and dst must be the same.");

    a5::TRemSCheck<DstTile, SrcTile>();
    TRemS<DstTile, SrcTile>(dst.data(), src.data(), scalar, validRow, validCol);
}
} // namespace kirin9030
} // namespace pto
#endif // TREMS_HPP_KIRIN9030
