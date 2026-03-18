/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under
the terms and conditions of CANN Open Software License Agreement Version 2.0
(the "License"). Please refer to the License for details. You may not use this
file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON AN "AS
IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
PARTICULAR PURPOSE. See LICENSE in the root of the software repository for the
full text of the License.
*/

#ifndef __ROW_REDUCE_IDX__
#define __ROW_REDUCE_IDX__

#include "common.hpp"
#include "pto/common/pto_tile.hpp"
#include <math.h>
#include <type_traits>

namespace pto {

template <typename TileDataOut, typename TileDataIn, typename TileDataTmp>
__tf__ PTO_INTERNAL OP_NAME(TROWCMAX)
    OP_TYPE(reduce) void TRowCMax(typename TileDataOut::TileDType __out__ dst, typename TileDataIn::TileDType __in__ src,
                                  uint32_t rows, uint32_t cols, unsigned version = VFImplKind::VFIMPL_DEFAULT)
{
    using TDst = typename TileDataOut::DType;
    using TSrc = typename TileDataIn::DType;
    constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(TSrc);
    uint16_t repeatTimes = CeilDivision(cols, elementsPerRepeat);
    __ubuf__ TDst *dstPtr = (__ubuf__ TDst *)__cce_get_tile_ptr(dst);
    __ubuf__ TSrc *srcPtr = (__ubuf__ TSrc *)__cce_get_tile_ptr(src);
    __VEC_SCOPE__
    {
        constexpr auto distValue =
            std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<TDst, DistVST::DIST_ONEPT>())>();
        RegTensor<TSrc> vregZero;
        vbr(vregZero, 0);
        RegTensor<TSrc> vregSrc;
        RegTensor<TSrc> vregValOrig;
        RegTensor<TSrc> vregIdxOrig;
        RegTensor<TSrc> vregVal;
        RegTensor<TSrc> vregIdx;

        uint32_t dstDup = 1;
        MaskReg pRegOneElem = CreatePredicate<TSrc>(dstDup);
        MaskReg pregCmp;

        for (uint16_t i = 0; i < (uint16_t)rows; ++i) {
            vbr(vregValOrig, -__FLT_MAX__);
            vbr(vregIdxOrig, 0);
            uint32_t sregDup = elementsPerRepeat;
            for (uint16_t j = 0; j < (uint16_t)repeatTimes; j++) {
                MaskReg pRegSrc = CreatePredicate<TSrc>(sregDup);
                vlds(vregSrc, srcPtr, i * TileDataIn::RowStride + j * elementsPerRepeat, NORM);
                vcmax(vregVal, vregSrc, pRegSrc, MODE_ZEROING);
                vdintlv(vregVal, vregIdx, vregVal, vregZero);
                // vcmp_gt for MIN
                vcmp_lt(pregCmp, vregValOrig, vregVal, pRegOneElem);
                vsel(vregValOrig, vregVal, vregValOrig, pregCmp);
                vsel(vregIdxOrig, vregIdx, vregIdxOrig, pregCmp);
                vadds((RegTensor<TDst>&)vregIdxOrig, (RegTensor<TDst>&)vregIdxOrig, j * elementsPerRepeat, pregCmp, MODE_ZEROING);
            }
            vsts((RegTensor<TDst>&)vregIdxOrig, dstPtr, i * TileDataOut::RowStride, distValue, pRegOneElem);
        }
    }
}

template <typename TileDataOut, typename TileDataIn, typename TileDataTmp>
PTO_INTERNAL void TROWCMAX_IMPL(TileDataOut &dst, TileDataIn &src, TileDataTmp &tmp)
{
    unsigned rows = src.GetValidRow();
    unsigned cols = src.GetValidCol();
    TRowCMax<TileDataOut, TileDataIn, TileDataTmp>(dst.data(), src.data(), rows, cols);
}

template <typename TileDataOut, typename TileDataIn, typename TileDataTmp>
PTO_INTERNAL void TROWCMIN_IMPL(TileDataOut &dst, TileDataIn &src, TileDataTmp &tmp)
{
}
} // namespace pto

#endif