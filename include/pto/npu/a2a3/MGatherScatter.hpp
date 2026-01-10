/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef MGATHER_SCATTER_A2A3_HPP
#define MGATHER_SCATTER_A2A3_HPP

#include <type_traits>
#include <pto/common/pto_tile.hpp>
#include <pto/npu/a2a3/TExtraOps.hpp>

namespace pto {

template <typename TileDst, typename GlobalData, typename TileInd>
__tf__ PTO_INTERNAL void MGATHER_TF(typename TileDst::TileDType __out__ dstData, typename GlobalData::DType *base,
    typename TileInd::TileDType __in__ idxData, uint32_t validRow, uint32_t validCol)
{
    using IndexT = typename TileInd::DType;
    using DstT = typename TileDst::DType;
    __ubuf__ DstT *dstPtr = (__ubuf__ DstT *)__cce_get_tile_ptr(dstData);
    __ubuf__ IndexT *idxPtr = (__ubuf__ IndexT *)__cce_get_tile_ptr(idxData);

    for (uint32_t r = 0; r < validRow; ++r) {
        for (uint32_t c = 0; c < validCol; ++c) {
            const uint32_t dstOff = GetTileElementOffset<TileDst>(r, c);
            const uint32_t idxOff = GetTileElementOffset<TileInd>(r, c);
            const uint32_t idx = static_cast<uint32_t>(idxPtr[idxOff]);
            dstPtr[dstOff] = base[idx];
        }
    }
}

template <typename TileDst, typename GlobalData, typename TileInd>
PTO_INTERNAL void MGATHER_IMPL(TileDst &dst, GlobalData &src, TileInd &indexes)
{
    using IndexT = typename TileInd::DType;
    static_assert(std::is_integral_v<IndexT>, "MGATHER: indexes must be an integral type");
    static_assert(sizeof(typename TileDst::DType) == sizeof(typename GlobalData::DType),
        "MGATHER: element sizes must match");
    static_assert(TileDst::Loc == TileType::Vec, "MGATHER: dst must be a Vec tile");
    static_assert(TileInd::Loc == TileType::Vec, "MGATHER: indexes must be a Vec tile");

    const uint32_t validRow = dst.GetValidRow();
    const uint32_t validCol = dst.GetValidCol();
    if (validRow == 0 || validCol == 0) {
        return;
    }

    typename GlobalData::DType *base = src.data();
    MGATHER_TF<TileDst, GlobalData, TileInd>(dst.data(), base, indexes.data(), validRow, validCol);
}

template <typename GlobalData, typename TileSrc, typename TileInd>
__tf__ PTO_INTERNAL void MSCATTER_TF(typename GlobalData::DType *base, typename TileSrc::TileDType __in__ srcData,
    typename TileInd::TileDType __in__ idxData, uint32_t validRow, uint32_t validCol)
{
    using SrcT = typename TileSrc::DType;
    using IndexT = typename TileInd::DType;
    __ubuf__ SrcT *srcPtr = (__ubuf__ SrcT *)__cce_get_tile_ptr(srcData);
    __ubuf__ IndexT *idxPtr = (__ubuf__ IndexT *)__cce_get_tile_ptr(idxData);
    for (uint32_t r = 0; r < validRow; ++r) {
        for (uint32_t c = 0; c < validCol; ++c) {
            const uint32_t srcOff = GetTileElementOffset<TileSrc>(r, c);
            const uint32_t idxOff = GetTileElementOffset<TileInd>(r, c);
            const uint32_t idx = static_cast<uint32_t>(idxPtr[idxOff]);
            base[idx] = srcPtr[srcOff];
        }
    }
}

template <typename GlobalData, typename TileSrc, typename TileInd>
PTO_INTERNAL void MSCATTER_IMPL(GlobalData &dst, TileSrc &src, TileInd &indexes)
{
    using IndexT = typename TileInd::DType;
    static_assert(std::is_integral_v<IndexT>, "MSCATTER: indexes must be an integral type");
    static_assert(sizeof(typename TileSrc::DType) == sizeof(typename GlobalData::DType),
        "MSCATTER: element sizes must match");
    static_assert(TileSrc::Loc == TileType::Vec, "MSCATTER: src must be a Vec tile");
    static_assert(TileInd::Loc == TileType::Vec, "MSCATTER: indexes must be a Vec tile");

    const uint32_t validRow = src.GetValidRow();
    const uint32_t validCol = src.GetValidCol();
    if (validRow == 0 || validCol == 0) {
        return;
    }

    typename GlobalData::DType *base = dst.data();
    MSCATTER_TF<GlobalData, TileSrc, TileInd>(base, src.data(), indexes.data(), validRow, validCol);
}

} // namespace pto

#endif
