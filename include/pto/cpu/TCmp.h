/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TCMP_H
#define TCMP_H

#include <pto/common/pto_tile.hpp>
#include "pto/cpu/tile_offsets.hpp"
#include "pto/cpu/parallel.hpp"

namespace pto {

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1>
PTO_INTERNAL void TCmp(
    typename TileDataDst::TileDType dst, typename TileDataSrc0::TileDType src0, typename TileDataSrc1::TileDType src1,
    CmpMode mode, unsigned srcValidRow, unsigned srcValidCol, unsigned dstValidCol)
{
    using SrcT = typename TileDataSrc0::DType;

    cpu::parallel_for_rows(srcValidRow, srcValidCol, [&](std::size_t r) {
        unsigned validColBytes = (srcValidCol + 7) / 8;
        unsigned bytesToWrite = std::min<unsigned>(validColBytes, dstValidCol);
        for (size_t b = 0; b < bytesToWrite; b++) {
            uint8_t packedByte = 0;
            size_t bitBase = b * 8;
            for (size_t bit = 0; bit < 8 && (bitBase + bit) < srcValidCol; bit++) {
                size_t c = bitBase + bit;
                size_t srcIdx0 = GetTileElementOffset<TileDataSrc0>(r, c);
                size_t srcIdx1 = GetTileElementOffset<TileDataSrc1>(r, c);
                SrcT a = src0[srcIdx0];
                SrcT b = src1[srcIdx1];
                uint8_t cmp = 0;
                {
                    const double diff = static_cast<double>(a) - static_cast<double>(b);
                    switch (mode) {
                        case CmpMode::EQ:
                            cmp = (std::fabs(diff) < 1e-9);
                            break;
                        case CmpMode::NE:
                            cmp = (std::fabs(diff) > 1e-9);
                            break;
                        case CmpMode::LT:
                            cmp = (a < b);
                            break;
                        case CmpMode::GT:
                            cmp = (a > b);
                            break;
                        case CmpMode::GE:
                            cmp = (a >= b);
                            break;
                        case CmpMode::LE:
                            cmp = (a <= b);
                            break;
                        default:
                            cmp = (std::fabs(diff) < 1e-9);
                            break;
                    }
                }
                packedByte |= (cmp << bit);
            }
            size_t dstIdx = GetTileElementOffset<TileDataDst>(r, b);
            dst[dstIdx] = packedByte;
        }
    });
}

template <typename TileDataDst>
PTO_INTERNAL void ZeroTileData(typename TileDataDst::TileDType data)
{
    using T = typename TileDataDst::DType;
    constexpr int32_t dstRowStride_ = TileDataDst::RowStride;
    constexpr size_t dstRowStride =
        (dstRowStride_ == -1) ? static_cast<size_t>(TileDataDst::Cols) : static_cast<size_t>(dstRowStride_);
    constexpr size_t count = TileDataDst::Rows * dstRowStride;
    for (size_t i = 0; i < count; i++) {
        data[i] = 0;
    }
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1>
PTO_INTERNAL void TCMP_IMPL(TileDataDst& dst, TileDataSrc0& src0, TileDataSrc1& src1, CmpMode cmpMode)
{
    using T = typename TileDataSrc0::DType;
    static_assert(std::is_same_v<T, typename TileDataSrc1::DType>, "TCMP: src0 and src1 must have same type");
    static_assert(TileDataDst::Loc == TileType::Vec, "TCMP: dst tile must be TileType::Vec");
    static_assert(TileDataSrc0::Loc == TileType::Vec, "TCMP: src0 tile must be TileType::Vec");
    static_assert(TileDataSrc1::Loc == TileType::Vec, "TCMP: src1 tile must be TileType::Vec");

    PTO_ASSERT(src0.GetValidRow() == src1.GetValidRow(), "TCMP: src0 and src1 must have same valid rows");
    PTO_ASSERT(src0.GetValidCol() == src1.GetValidCol(), "TCMP: src0 and src1 must have same valid cols");
    PTO_ASSERT(src0.GetValidRow() == dst.GetValidRow(), "TCMP: src0 and dst must have same valid rows");

    unsigned srcValidRow = src0.GetValidRow();
    unsigned srcValidCol = src0.GetValidCol();
    unsigned dstValidCol = dst.GetValidCol();

    ZeroTileData<TileDataDst>(dst.data());

    TCmp<TileDataDst, TileDataSrc0, TileDataSrc1>(
        dst.data(), src0.data(), src1.data(), cmpMode, srcValidRow, srcValidCol, dstValidCol);
}

} // namespace pto
#endif
