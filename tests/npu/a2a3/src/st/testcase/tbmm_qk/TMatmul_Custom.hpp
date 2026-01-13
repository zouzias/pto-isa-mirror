/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TMATMUL_CT_HPP
#define TMATMUL_CT_HPP

namespace pto {


template <typename TileRes, typename TileLeft, typename TileRight, bool cmatrixSource, bool cmatrixInitVal>
__tf__ AICORE void TMatmul_uf(typename TileRes::TileDType __out__ cMatrix, typename TileLeft::TileDType __in__ aMatrix,
    typename TileRight::TileDType __in__ bMatrix, uint16_t m, uint16_t k, uint16_t n, uint16_t unitFlag)
{
    constexpr bool kDirectionAlign = GetKDirectionAlign<TileLeft>();

    __cc__ typename TileRes::DType *c = (__cc__ typename TileRes::DType *)__cce_get_tile_ptr(cMatrix);
    __ca__ typename TileLeft::DType *a = (__ca__ typename TileLeft::DType *)__cce_get_tile_ptr(aMatrix);
    __cb__ typename TileRight::DType *b = (__cb__ typename TileRight::DType *)__cce_get_tile_ptr(bMatrix);

    mad(c, a, b, m, k, n, unitFlag, kDirectionAlign, cmatrixSource, cmatrixInitVal);
}

template <typename TileRes, typename TileLeft, typename TileRight>
PTO_INTERNAL void TMATMUL_UF(TileRes &cMatrix, TileLeft &aMatrix, TileRight &bMatrix, uint32_t unitFlag)
{
    CheckStaticMad<TileRes, TileLeft, TileRight>();
    uint16_t m = aMatrix.GetValidRow();
    uint16_t k = aMatrix.GetValidCol();
    uint16_t n = bMatrix.GetValidCol();
    CheckDynamicMad(m, k, n);
    TMatmul_uf<TileRes, TileLeft, TileRight, false, true>(cMatrix.data(), aMatrix.data(), bMatrix.data(), m, k, n, unitFlag);
}

template <typename TileRes, typename TileLeft, typename TileRight>
PTO_INTERNAL void TMATMUL_ACC_UF(
    TileRes &cOutMatrix, TileRes &cInMatrix, TileLeft &aMatrix, TileRight &bMatrix, uint32_t unitFlag)
{
    CheckStaticMad<TileRes, TileLeft, TileRight>();
    uint16_t m = aMatrix.GetValidRow();
    uint16_t k = aMatrix.GetValidCol();
    uint16_t n = bMatrix.GetValidCol();
    CheckDynamicMad(m, k, n);
    TMatmul_uf<TileRes, TileLeft, TileRight, false, false>(cOutMatrix.data(), aMatrix.data(), bMatrix.data(), m, k, n, unitFlag);
}

} // namespace pto
#endif