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

template <typename TileDataOut, typename TileDataIn, typename TileDataTmp, unsigned elementsPerRepeat>
__tf__ PTO_INTERNAL OP_NAME(TROWCMAX)
    OP_TYPE(reduce) void TRowCMax(typename TileDataOut::TileDType __out__ dst, typename TileDataIn::TileDType __in__ src,
                                 typename TileDataTmp::TileDType tmp, 
                                 uint32_t srcValidRows, uint32_t srcValidCols, uint32_t dstValidRow,
                                 unsigned version = VFImplKind::VFIMPL_DEFAULT)
{
    __ubuf__ TileDataOut::TileDType *dstPtr = (__ubuf__ TileDataOut::TileDType *)__cce_get_tile_ptr(dst);
    __ubuf__ TileDataIn::TileDType *srcPtr = (__ubuf__ TileDataIn::TileDType *)__cce_get_tile_ptr(src);
    __ubuf__ TileDataTmp::TileDType *tmpPtr = (__ubuf__ TileDataTmp::TileDType *)__cce_get_tile_ptr(tmp);
    __VEC_SCOPE__
    {
        uint32_t sregDup = elementsPerRepeat;
        pReg = CreatePredicate<T>(sregDup);
        RegTensor<TileDataIn::TileDType> vreg0, vreg1, vregDst;
        vlds(vreg0, srcPtr, 0, NORM);
        vslide(vregDst, vreg0, vreg1, 2);
        vsts(vregDst, dstPtr, 0, pReg);
    }
}

template <typename TileDataOut, typename TileDataIn, typename TileDataTmp>
PTO_INTERNAL void TROWMAX_IMPL(TileDataOut &dst, TileDataIn &src, TileDataTmp &tmp)
{
    using T = typename TileDataIn::DType;
    constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(T);
    unsigned rows = src.GetValidRow();
    unsigned cols = src.GetValidCol();

    TRowMax<TileDataOut, TileDataIn, TileDataTmp, elementsPerRepeat>(dst.data(), src.data(), tmp.data(), rows, cols, dst.GetValidRow());
}

template <typename TileDataOut, typename TileDataIn, typename TileDataTmp>
PTO_INTERNAL void TROWMIN_IMPL(TileDataOut &dst, TileDataIn &src, TileDataTmp &tmp)
{
}
} // namespace pto

#endif