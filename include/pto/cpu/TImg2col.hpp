/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef TIMG2COL_CPU_HPP
#define TIMG2COL_CPU_HPP

#include <algorithm>
#include <cstdint>
#include "pto/cpu/tile_offsets.hpp"

namespace pto {
namespace cpu_img2col {
template <typename ConvTileData>
inline size_t GetInputOffset(const ConvTileData &src, int64_t n, int64_t d, int64_t c1, int64_t h, int64_t w,
                             int64_t c0)
{
    const int64_t c0Size = src.GetShape(ConvTileData::totalDimCount - 1);
    if constexpr (ConvTileData::layout == Layout::NC1HWC0) {
        const int64_t c1Size = src.GetShape(1);
        const int64_t hSize = src.GetShape(2);
        const int64_t wSize = src.GetShape(3);
        return static_cast<size_t>(((((n * c1Size + c1) * hSize + h) * wSize + w) * c0Size) + c0);
    } else {
        const int64_t dSize = src.GetShape(1);
        const int64_t c1Size = src.GetShape(2);
        const int64_t hSize = src.GetShape(3);
        const int64_t wSize = src.GetShape(4);
        return static_cast<size_t>((((((n * dSize + d) * c1Size + c1) * hSize + h) * wSize + w) * c0Size) + c0);
    }
}

template <typename TileData, typename ConvTileData>
inline void Timg2colCheck()
{
    static_assert(ConvTileData::Loc == TileType::Mat, "TImg2col: Source TileType only support Mat.");
    static_assert(TileData::Loc == TileType::Left, "TImg2col: Destination TileType only support Left.");
    static_assert((ConvTileData::layout == Layout::NC1HWC0) || (ConvTileData::layout == Layout::NDC1HWC0),
                  "TImg2col: Source layout only support NC1HWC0/NDC1HWC0.");
    static_assert(TileData::SFractal == SLayout::RowMajor && !TileData::isRowMajor,
                  "TImg2col: Destination layout only support SLayout RowMajor + BLayout ColMajor.");
    static_assert(std::is_same_v<typename ConvTileData::DType, typename TileData::DType>,
                  "TImg2col: Destination and source tile data types must match.");
}
} // namespace cpu_img2col

template <typename TileData, typename ConvTileData, SetFmatrixMode FmatrixMode = SetFmatrixMode::FMATRIX_A_MANUAL>
PTO_INTERNAL void TIMG2COL_IMPL(TileData &dst, ConvTileData &src, uint16_t posM, uint16_t posK)
{
    (void)FmatrixMode;
    cpu_img2col::Timg2colCheck<TileData, ConvTileData>();

    const int64_t fmapN = src.GetShape(0);
    const int64_t fmapD = (ConvTileData::layout == Layout::NDC1HWC0) ? src.GetShape(1) : 1;
    const int64_t fmapC1 = (ConvTileData::layout == Layout::NDC1HWC0) ? src.GetShape(2) : src.GetShape(1);
    const int64_t fmapH = (ConvTileData::layout == Layout::NDC1HWC0) ? src.GetShape(3) : src.GetShape(2);
    const int64_t fmapW = (ConvTileData::layout == Layout::NDC1HWC0) ? src.GetShape(4) : src.GetShape(3);
    const int64_t fmapC0 = src.GetShape(ConvTileData::totalDimCount - 1);
    const int64_t strideH = src.GetStrideH();
    const int64_t strideW = src.GetStrideW();
    const int64_t dilationH = src.GetDilationH();
    const int64_t dilationW = src.GetDilationW();
    const int64_t filterH = src.GetFilterH();
    const int64_t filterW = src.GetFilterW();
    const int64_t padLeft = src.GetPadList(0);
    const int64_t padRight = src.GetPadList(1);
    const int64_t padTop = src.GetPadList(2);
    const int64_t padBottom = src.GetPadList(3);
    const int64_t channelSize = src.GetChannelSize() > 0 ? src.GetChannelSize() : fmapC1 * fmapC0;

    const int64_t outH = (fmapH + padTop + padBottom - dilationH * (filterH - 1) - 1) / strideH + 1;
    const int64_t outW = (fmapW + padLeft + padRight - dilationW * (filterW - 1) - 1) / strideW + 1;
    const int64_t mPerBatch = fmapD * outH * outW;
    const auto padValue = src.GetPadValue();

    for (int r = 0; r < dst.GetValidRow(); ++r) {
        const int64_t mIndex = static_cast<int64_t>(posM) + r;
        const int64_t nIndex = mIndex / mPerBatch;
        const int64_t remAfterN = mIndex % mPerBatch;
        const int64_t dIndex = remAfterN / (outH * outW);
        const int64_t remAfterD = remAfterN % (outH * outW);
        const int64_t outRow = remAfterD / outW;
        const int64_t outCol = remAfterD % outW;

        for (int c = 0; c < dst.GetValidCol(); ++c) {
            const int64_t kIndex = static_cast<int64_t>(posK) + c;
            const int64_t channelIndex = kIndex / (filterH * filterW);
            const int64_t kernelOffset = kIndex % (filterH * filterW);
            const int64_t kernelH = kernelOffset / filterW;
            const int64_t kernelW = kernelOffset % filterW;

            auto value = padValue;
            if (nIndex < fmapN && channelIndex < channelSize) {
                const int64_t c1Index = channelIndex / fmapC0;
                const int64_t c0Index = channelIndex % fmapC0;
                const int64_t inputH = outRow * strideH + kernelH * dilationH - padTop;
                const int64_t inputW = outCol * strideW + kernelW * dilationW - padLeft;
                if (inputH >= 0 && inputH < fmapH && inputW >= 0 && inputW < fmapW) {
                    const size_t srcOffset =
                        cpu_img2col::GetInputOffset(src, nIndex, dIndex, c1Index, inputH, inputW, c0Index);
                    value = src.data()[srcOffset];
                }
            }
            dst.data()[GetTileElementOffset<TileData>(r, c)] = value;
        }
    }
}
} // namespace pto

#endif
