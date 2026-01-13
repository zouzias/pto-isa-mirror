/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TCOLEXPAND_HPP
#define TCOLEXPAND_HPP

#include <cstddef>
#include <type_traits>

#include "pto/cpu/tile_offsets.hpp"
#include "pto/cpu/parallel.hpp"
#include <pto/common/pto_tile.hpp>

namespace pto {

template <typename TileOut, typename TileIn>
PTO_INTERNAL void TColExpand_Impl(typename TileOut::TileDType dst, typename TileIn::TileDType src, unsigned validRow,
                                 unsigned validCol)
{
    if (validRow == 0 || validCol == 0) {
        return;
    }
    cpu::parallel_for_1d(0, validCol, static_cast<std::size_t>(validRow) * validCol, [&](std::size_t j) {
        const auto srcVal = src[GetTileElementOffset<TileIn>(0, j)];
        for (std::size_t i = 0; i < validRow; ++i) {
            dst[GetTileElementOffset<TileOut>(i, j)] = srcVal;
        }
    });
}

template <typename TileDataOut, typename TileDataIn>
PTO_INTERNAL void TCOLEXPAND_IMPL(TileDataOut &dst, TileDataIn &src)
{
    TColExpand_Impl<TileDataOut, TileDataIn>(dst.data(), src.data(), dst.GetValidRow(), dst.GetValidCol());
}

namespace {
template <typename TileVec>
PTO_INTERNAL typename TileVec::DType load_col_scalar(TileVec &src1, std::size_t colIndex)
{
    const std::size_t vr = static_cast<std::size_t>(src1.GetValidRow());
    const std::size_t vc = static_cast<std::size_t>(src1.GetValidCol());
    if (vr == 1 && colIndex < vc) {
        return static_cast<typename TileVec::DType>(src1.data()[GetTileElementOffset<TileVec>(0, colIndex)]);
    }
    if (vc == 1 && colIndex < vr) {
        return static_cast<typename TileVec::DType>(src1.data()[GetTileElementOffset<TileVec>(colIndex, 0)]);
    }
    return static_cast<typename TileVec::DType>(src1.data()[colIndex % static_cast<std::size_t>(TileVec::Numel)]);
}
} // namespace

template <typename TileDst, typename TileSrc1>
PTO_INTERNAL void TCOLEXPANDDIV_IMPL(TileDst &dst, TileDst &src0, TileSrc1 &src1)
{
    const std::size_t rows = static_cast<std::size_t>(dst.GetValidRow());
    const std::size_t cols = static_cast<std::size_t>(dst.GetValidCol());
    if (rows == 0 || cols == 0) {
        return;
    }

    cpu::parallel_for_1d(0, cols, rows * cols, [&](std::size_t c) {
        const auto s = static_cast<typename TileDst::DType>(load_col_scalar(src1, c));
        for (std::size_t r = 0; r < rows; ++r) {
            const auto v0 = static_cast<typename TileDst::DType>(src0.data()[GetTileElementOffset<TileDst>(r, c)]);
            dst.data()[GetTileElementOffset<TileDst>(r, c)] = static_cast<typename TileDst::DType>(v0 / s);
        }
    });
}

template <typename TileDst, typename TileSrc1>
PTO_INTERNAL void TCOLEXPANDMUL_IMPL(TileDst &dst, TileDst &src0, TileSrc1 &src1)
{
    const std::size_t rows = static_cast<std::size_t>(dst.GetValidRow());
    const std::size_t cols = static_cast<std::size_t>(dst.GetValidCol());
    if (rows == 0 || cols == 0) {
        return;
    }

    cpu::parallel_for_1d(0, cols, rows * cols, [&](std::size_t c) {
        const auto s = static_cast<typename TileDst::DType>(load_col_scalar(src1, c));
        for (std::size_t r = 0; r < rows; ++r) {
            const auto v0 = static_cast<typename TileDst::DType>(src0.data()[GetTileElementOffset<TileDst>(r, c)]);
            dst.data()[GetTileElementOffset<TileDst>(r, c)] = static_cast<typename TileDst::DType>(v0 * s);
        }
    });
}

template <typename TileDst, typename TileSrc1>
PTO_INTERNAL void TCOLEXPANDSUB_IMPL(TileDst &dst, TileDst &src0, TileSrc1 &src1)
{
    const std::size_t rows = static_cast<std::size_t>(dst.GetValidRow());
    const std::size_t cols = static_cast<std::size_t>(dst.GetValidCol());
    if (rows == 0 || cols == 0) {
        return;
    }

    cpu::parallel_for_1d(0, cols, rows * cols, [&](std::size_t c) {
        const auto s = static_cast<typename TileDst::DType>(load_col_scalar(src1, c));
        for (std::size_t r = 0; r < rows; ++r) {
            const auto v0 = static_cast<typename TileDst::DType>(src0.data()[GetTileElementOffset<TileDst>(r, c)]);
            dst.data()[GetTileElementOffset<TileDst>(r, c)] = static_cast<typename TileDst::DType>(v0 - s);
        }
    });
}

} // namespace pto

#endif
