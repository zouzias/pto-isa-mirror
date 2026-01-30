/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_CPU_TINSERT_HPP
#define PTO_CPU_TINSERT_HPP

#include <cstdint>
#include <type_traits>
#include <algorithm>

#include <pto/common/type.hpp>
#include "pto/cpu/tile_offsets.hpp"

namespace pto {

// CPU-SIM: generic insert using tile element offsets.
// Conceptually: dst[indexRow + r, indexCol + c] = src[r, c] for r/c in src valid region.
template <typename DstTileData, typename SrcTileData>
PTO_INTERNAL void TINSERT_IMPL(DstTileData &dst, SrcTileData &src, uint16_t indexRow, uint16_t indexCol)
{
    static_assert(std::is_same_v<typename DstTileData::DType, typename SrcTileData::DType>,
        "TINSERT: src/dst dtype must match in CPU-SIM");

    const std::size_t srcRows = static_cast<std::size_t>(src.GetValidRow());
    const std::size_t srcCols = static_cast<std::size_t>(src.GetValidCol());
    const std::size_t dstRows = static_cast<std::size_t>(dst.GetValidRow());
    const std::size_t dstCols = static_cast<std::size_t>(dst.GetValidCol());

    const std::size_t i0 = static_cast<std::size_t>(indexRow);
    const std::size_t j0 = static_cast<std::size_t>(indexCol);

    // Bounds: only write into the dst valid region.
    if (i0 >= dstRows || j0 >= dstCols) {
        return;
    }

    const std::size_t maxR = std::min(srcRows, dstRows - i0);
    const std::size_t maxC = std::min(srcCols, dstCols - j0);

    for (std::size_t r = 0; r < maxR; ++r) {
        for (std::size_t c = 0; c < maxC; ++c) {
            dst.data()[GetTileElementOffset<DstTileData>(i0 + r, j0 + c)] =
                static_cast<typename DstTileData::DType>(src.data()[GetTileElementOffset<SrcTileData>(r, c)]);
        }
    }
}

// Overloads used by the C++ wrapper (ignore quant/relu knobs in CPU-SIM and just reuse the base path).
template <typename DstTileData, typename SrcTileData, ReluPreMode reluMode>
PTO_INTERNAL void TINSERT_IMPL(DstTileData &dst, SrcTileData &src, uint16_t indexRow = 0, uint16_t indexCol = 0)
{
    (void)reluMode;
    TINSERT_IMPL(dst, src, indexRow, indexCol);
}

template <typename DstTileData, typename SrcTileData, ReluPreMode reluMode = ReluPreMode::NoRelu>
PTO_INTERNAL void TINSERT_IMPL(DstTileData &dst, SrcTileData &src, uint64_t preQuantScalar,
    uint16_t indexRow = 0, uint16_t indexCol = 0)
{
    (void)reluMode;
    (void)preQuantScalar;
    TINSERT_IMPL(dst, src, indexRow, indexCol);
}

template <typename DstTileData, typename SrcTileData, typename FpTileData, ReluPreMode reluMode = ReluPreMode::NoRelu>
PTO_INTERNAL void TINSERT_IMPL(DstTileData &dst, SrcTileData &src, FpTileData &fp,
    uint16_t indexRow = 0, uint16_t indexCol = 0)
{
    (void)reluMode;
    (void)fp;
    TINSERT_IMPL(dst, src, indexRow, indexCol);
}

} // namespace pto

#endif
