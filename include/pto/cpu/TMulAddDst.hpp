/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TMULADDDST_HPP
#define TMULADDDST_HPP

#include <pto/common/pto_tile.hpp>
#include "pto/cpu/tile_offsets.hpp"
#include "pto/cpu/parallel.hpp"

namespace pto {
template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1>
void TMuladddst_Impl(typename TileDataDst::TileDType dst, typename TileDataSrc0::TileDType src0,
                     typename TileDataSrc1::TileDType src1, unsigned validRow, unsigned validCol)
{
    using DstT = typename TileDataDst::DType;
    cpu::parallel_for_rows(validRow, validCol, [&](std::size_t r) {
        PTO_CPU_VECTORIZE_LOOP
        for (std::size_t c = 0; c < validCol; ++c) {
            std::size_t indexDst = GetTileElementOffset<TileDataDst>(r, c);
            std::size_t indexSrc0 = GetTileElementOffset<TileDataSrc0>(r, c);
            std::size_t indexSrc1 = GetTileElementOffset<TileDataSrc1>(r, c);
            dst[indexDst] =
                static_cast<DstT>(((float)src0[indexSrc0]) * ((float)src1[indexSrc1]) + (float)dst[indexDst]);
        }
    });
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1>
PTO_INTERNAL void TMULADDDST_IMPL(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1)
{
    unsigned row = dst.GetValidRow();
    unsigned col = dst.GetValidCol();
    TMuladddst_Impl<TileDataDst, TileDataSrc0, TileDataSrc1>(dst.data(), src0.data(), src1.data(), row, col);
}
} // namespace pto
#endif
