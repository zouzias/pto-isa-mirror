/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef TTRANS_HPP
#define TTRANS_HPP

#include <pto/common/pto_tile.hpp>
#include "pto/cpu/tile_offsets.hpp"
#include <type_traits>
#include <cstdint>

namespace pto {

// Target layout formats defined in your system architecture
template<bool ignored = true> enum class Layout {
    NCHW,
    NC1HWC0,
    FRACTAL_Z
};

// ============================================================================
// Independent Layout Implementation Workers
// ============================================================================

template <typename DstTileData, typename SrcTileData>
inline void TTRANS_NCHW_TO_NC1HWC0_CORE(DstTileData &dst, SrcTileData &src)
{
    using SrcDType = typename SrcTileData::DType;
    using DstDType = typename DstTileData::DType;

    const auto* src_ptr = reinterpret_cast<const SrcDType*>(src.data());
    auto* dst_ptr = reinterpret_cast<DstDType*>(dst.data());

    constexpr int64_t C0 = 32 / sizeof(SrcDType);

    int64_t N = src.GetShape(0);
    int64_t C = src.GetShape(1);
    int64_t H = src.GetShape(2);
    int64_t W = src.GetShape(3);
    int64_t C1 = (C + C0 - 1) / C0;

    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            size_t r = c / C0;
            size_t cl = c % C0;
            for (int64_t h = 0; h < H; ++h) {
                for (int64_t w = 0; w < W; ++w) {
                    size_t srcIndex = w + W*h + W*H*c + W*H*C*n;
                    size_t dstIndex = W*H*C1*C0*n + C0*H*W*r + C0*W*h + C0*w + cl;
                    dst_ptr[dstIndex] = src_ptr[srcIndex];
                }
            }
        }
    }
}

template <typename DstTileData, typename SrcTileData>
void TTrans_Impl(typename DstTileData::TileDType dst, typename SrcTileData::TileDType src, unsigned validRow,
                 unsigned validCol)
{
    for (size_t c = 0; c < validCol; c++) {
        size_t subTileSrcC = c / SrcTileData::InnerCols;
        size_t innerSrcC = c % SrcTileData::InnerCols;
        size_t subTileDstC = c / DstTileData::InnerCols;
        size_t innerDstC = c % DstTileData::InnerCols;

        for (size_t r = 0; r < validRow; r++) {
            size_t srcTileIdx, dstTileIdx;
            if constexpr (SrcTileData::SFractal == SLayout::NoneBox)
                srcTileIdx = GetTileElementOffsetPlain<SrcTileData>(r, c);
            else {
                size_t subTileR = r / SrcTileData::InnerRows;
                size_t innerR = r % SrcTileData::InnerRows;
                srcTileIdx = GetElementOffsetSubfractals<SrcTileData>(subTileSrcC, innerSrcC, subTileR, innerR);
            }

            if constexpr (DstTileData::SFractal == SLayout::NoneBox)
                dstTileIdx = GetTileElementOffsetPlain<DstTileData>(c, r);
            else {
                size_t subTileR = r / DstTileData::InnerRows;
                size_t innerR = r % DstTileData::InnerRows;
                dstTileIdx = GetElementOffsetSubfractals<DstTileData>(subTileR, innerR, subTileDstC, innerDstC);
            }
            dst[dstTileIdx] = src[srcTileIdx];
        }
    }
}

template <typename DstTileData, typename SrcTileData, typename TmpTileData>
PTO_INTERNAL void TTRANS_IMPL(DstTileData &dst, SrcTileData &src, TmpTileData &tmp)
{
    // Validate matching element widths at compilation
    static_assert(sizeof(typename SrcTileData::DType) == sizeof(typename DstTileData::DType), 
                  "Data type sizes between source and destination tiles must match.");

    constexpr Layout src_layout = SrcTileData::layout;
    constexpr Layout dst_layout = DstTileData::layout;

    // Mode 1: NCHW -> NC1HWC0
    if constexpr (src_layout == Layout::NCHW && dst_layout == Layout::NC1HWC0) {
        TTRANS_NCHW_TO_NC1HWC0_CORE(dst, src);
    }
    
    else if constexpr (is_tile_data_v<SrcTileData>) {
        static_assert(SrcTileData::ValidRow == DstTileData::ValidCol && SrcTileData::ValidCol == DstTileData::ValidRow,
                      "Hardware matrix tiles transpose dimension sizes must mirror match.");
        unsigned validRow = src.GetValidRow();
        unsigned validCol = src.GetValidCol();
        TTrans_Impl<DstTileData, SrcTileData>(dst.data(), src.data(), validRow, validCol);
    }   
}

} // namespace pto

#endif // TTRANS_HPP