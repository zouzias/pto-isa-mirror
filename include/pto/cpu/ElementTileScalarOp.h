/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef ELEMENT_TILE_SCLAR_OP_HPP
#define ELEMENT_TILE_SCLAR_OP_HPP

#include "pto/cpu/ElementOp.h"
#include "pto/cpu/parallel.hpp"

namespace pto {
template <typename tile_shape, ElementOp op>
void ZeroTileScalarOp_Impl(typename tile_shape::TileDType dst, typename tile_shape::DType scalar, unsigned validRow,
                           unsigned validCol)
{
    using DType = typename tile_shape::DType;
    if constexpr (tile_shape::SFractal == SLayout::NoneBox) {
        if constexpr (tile_shape::isRowMajor) {
            cpu::parallel_for_rows(validRow, validCol, [&](std::size_t r) {
                const std::size_t base = r * tile_shape::Cols;
                PTO_CPU_VECTORIZE_LOOP
                for (std::size_t c = 0; c < validCol; ++c) {
                    ElementOpCal<DType, op>::apply(dst[base + c], scalar);
                }
            });
        } else {
            cpu::parallel_for_rows(validCol, validRow, [&](std::size_t c) {
                const std::size_t base = c * tile_shape::Rows;
                PTO_CPU_VECTORIZE_LOOP
                for (std::size_t r = 0; r < validRow; ++r) {
                    ElementOpCal<DType, op>::apply(dst[base + r], scalar);
                }
            });
        }
    } else {
        if constexpr (tile_shape::isRowMajor) {
            cpu::parallel_for_rows(validRow, validCol, [&](std::size_t r) {
                for (std::size_t c = 0; c < validCol; ++c) {
                    const std::size_t idx = GetTileElementOffset<tile_shape>(r, c);
                    ElementOpCal<DType, op>::apply(dst[idx], scalar);
                }
            });
        } else {
            cpu::parallel_for_rows(validCol, validRow, [&](std::size_t c) {
                for (std::size_t r = 0; r < validRow; ++r) {
                    const std::size_t idx = GetTileElementOffset<tile_shape>(r, c);
                    ElementOpCal<DType, op>::apply(dst[idx], scalar);
                }
            });
        }
    }
}

template <typename TileDst, typename TileSrc, ElementOp op>
void UnaryTileScalarOpImpl(TileDst &dst, TileSrc &src, typename TileSrc::DType scalar, size_t extra = 0)
{
    using T = typename TileDst::DType;
    static_assert(std::is_same_v<T, typename TileSrc::DType>,
                  "UnaryTileScalarOpImpl: The data type of dst must be consistent with src.");

    static_assert(TileSrc::Loc == TileType::Vec, "TileType of src and dst tiles must be TileType::Vec.");

    PTO_ASSERT(src.GetValidCol() == dst.GetValidCol(), "Number of cols of src and dst must be the same.");
    PTO_ASSERT(src.GetValidRow() == dst.GetValidRow(), "Number of rows of src and dst must be the same.");

    unsigned rows = dst.GetValidRow();
    unsigned cols = dst.GetValidCol();

    if constexpr (TileDst::SFractal == SLayout::NoneBox) {
        if constexpr (TileDst::isRowMajor) {
            cpu::parallel_for_rows(rows, cols, [&](std::size_t r) {
                const std::size_t base = r * TileDst::Cols;
                PTO_CPU_VECTORIZE_LOOP
                for (std::size_t c = 0; c < cols; ++c) {
                    const std::size_t idx = base + c;
                    ElementOpCal<T, op>::apply(dst.data()[idx], src.data()[idx], scalar, extra);
                }
            });
        } else {
            cpu::parallel_for_rows(cols, rows, [&](std::size_t c) {
                const std::size_t base = c * TileDst::Rows;
                PTO_CPU_VECTORIZE_LOOP
                for (std::size_t r = 0; r < rows; ++r) {
                    const std::size_t idx = base + r;
                    ElementOpCal<T, op>::apply(dst.data()[idx], src.data()[idx], scalar, extra);
                }
            });
        }
    } else {
        if constexpr (TileDst::isRowMajor) {
            cpu::parallel_for_rows(rows, cols, [&](std::size_t r) {
                for (std::size_t c = 0; c < cols; ++c) {
                    const std::size_t idx = GetTileElementOffset<TileDst>(r, c);
                    ElementOpCal<T, op>::apply(dst.data()[idx], src.data()[idx], scalar, extra);
                }
            });
        } else {
            cpu::parallel_for_rows(cols, rows, [&](std::size_t c) {
                for (std::size_t r = 0; r < rows; ++r) {
                    const std::size_t idx = GetTileElementOffset<TileDst>(r, c);
                    ElementOpCal<T, op>::apply(dst.data()[idx], src.data()[idx], scalar, extra);
                }
            });
        }
    }
}

template <typename TileDst, typename TileSrc>
PTO_INTERNAL void TADDS_IMPL(TileDst &dst, TileSrc &src, typename TileSrc::DType scalar)
{
    UnaryTileScalarOpImpl<TileDst, TileSrc, ElementOp::OP_ADDS>(dst, src, scalar);
}

template <typename TileDst, typename TileSrc>
PTO_INTERNAL void TSUBS_IMPL(TileDst &dst, TileSrc &src, typename TileSrc::DType scalar)
{
    UnaryTileScalarOpImpl<TileDst, TileSrc, ElementOp::OP_SUBS>(dst, src, scalar);
}

template <typename TileDst, typename TileSrc>
PTO_INTERNAL void TMULS_IMPL(TileDst &dst, TileSrc &src, typename TileSrc::DType scalar)
{
    UnaryTileScalarOpImpl<TileDst, TileSrc, ElementOp::OP_MULS>(dst, src, scalar);
}

template <typename TileDst, typename TileSrc>
PTO_INTERNAL void TDIVS_IMPL(TileDst &dst, TileSrc &src, typename TileSrc::DType scalar)
{
    UnaryTileScalarOpImpl<TileDst, TileSrc, ElementOp::OP_DIVS>(dst, src, scalar);
}

template <typename TileDst, typename TileSrc>
PTO_INTERNAL void TDIVS_IMPL(TileDst &dst, typename TileSrc::DType scalar, TileSrc &src)
{
    UnaryTileScalarOpImpl<TileDst, TileSrc, ElementOp::OP_RDIVS>(dst, src, scalar);
}

template <typename TileDst, typename TileSrc>
PTO_INTERNAL void TMINS_IMPL(TileDst &dst, TileSrc &src, typename TileSrc::DType scalar)
{
    UnaryTileScalarOpImpl<TileDst, TileSrc, ElementOp::OP_MINS>(dst, src, scalar);
}

template <typename TileDst, typename TileSrc>
PTO_INTERNAL void TREMS_IMPL(TileDst &dst, TileSrc &src, typename TileSrc::DType scalar)
{
    UnaryTileScalarOpImpl<TileDst, TileSrc, ElementOp::OP_REMS>(dst, src, scalar);
}

template <typename TileDst, typename TileSrc>
PTO_INTERNAL void TREMS_IMPL(TileDst &dst, TileSrc &src, typename TileSrc::DType scalar, TileDst &tmp)
{
    (void)tmp;
    TREMS_IMPL(dst, src, scalar);
}

template <typename TileDst, typename TileSrc>
PTO_INTERNAL void TMAXS_IMPL(TileDst &dst, TileSrc &src, typename TileSrc::DType scalar)
{
    UnaryTileScalarOpImpl<TileDst, TileSrc, ElementOp::OP_MAXS>(dst, src, scalar);
}

template <typename TileDst, typename TileSrc>
PTO_INTERNAL void TANDS_IMPL(TileDst &dst, TileSrc &src, typename TileSrc::DType scalar)
{
    UnaryTileScalarOpImpl<TileDst, TileSrc, ElementOp::OP_ANDS>(dst, src, scalar);
}

template <typename TileDst, typename TileSrc>
PTO_INTERNAL void TORS_IMPL(TileDst &dst, TileSrc &src, typename TileSrc::DType scalar)
{
    UnaryTileScalarOpImpl<TileDst, TileSrc, ElementOp::OP_ORS>(dst, src, scalar);
}

template <typename TileDst, typename TileSrc>
PTO_INTERNAL void TXORS_IMPL(TileDst &dst, TileSrc &src, typename TileSrc::DType scalar)
{
    UnaryTileScalarOpImpl<TileDst, TileSrc, ElementOp::OP_XORS>(dst, src, scalar);
}

template <typename TileDst, typename TileSrc>
PTO_INTERNAL void TXORS_IMPL(TileDst &dst, TileSrc &src, typename TileSrc::DType scalar, TileDst &tmp)
{
    (void)tmp;
    TXORS_IMPL(dst, src, scalar);
}

template <typename TileDst, typename TileSrc>
PTO_INTERNAL void TLRELU_IMPL(TileDst &dst, TileSrc &src, typename TileSrc::DType scalar)
{
    UnaryTileScalarOpImpl<TileDst, TileSrc, ElementOp::OP_LRELU>(dst, src, scalar);
}

template <typename tile_shape, ElementOp op>
void ElementTileScalarOpWithCarry_Impl(typename tile_shape::TileDType dst, typename tile_shape::TileDType src0,
                                       typename tile_shape::DType scalar, typename tile_shape::TileDType src1,
                                       unsigned validRow, unsigned validCol)
{
    using DType = typename tile_shape::DType;
    if constexpr (tile_shape::SFractal == SLayout::NoneBox) {
        if constexpr (tile_shape::isRowMajor) {
            cpu::parallel_for_rows(validRow, validCol, [&](std::size_t r) {
                const std::size_t base = r * tile_shape::Cols;
                PTO_CPU_VECTORIZE_LOOP
                for (std::size_t c = 0; c < validCol; ++c) {
                    const std::size_t idx = base + c;
                    ElementOpCal<DType, op>::apply(dst[idx], src0[idx], scalar, src1[idx]);
                }
            });
        } else {
            cpu::parallel_for_rows(validCol, validRow, [&](std::size_t c) {
                const std::size_t base = c * tile_shape::Rows;
                PTO_CPU_VECTORIZE_LOOP
                for (std::size_t r = 0; r < validRow; ++r) {
                    const std::size_t idx = base + r;
                    ElementOpCal<DType, op>::apply(dst[idx], src0[idx], scalar, src1[idx]);
                }
            });
        }
    } else {
        if constexpr (tile_shape::isRowMajor) {
            cpu::parallel_for_rows(validRow, validCol, [&](std::size_t r) {
                for (std::size_t c = 0; c < validCol; ++c) {
                    const std::size_t idx = GetTileElementOffset<tile_shape>(r, c);
                    ElementOpCal<DType, op>::apply(dst[idx], src0[idx], scalar, src1[idx]);
                }
            });
        } else {
            cpu::parallel_for_rows(validCol, validRow, [&](std::size_t c) {
                for (std::size_t r = 0; r < validRow; ++r) {
                    const std::size_t idx = GetTileElementOffset<tile_shape>(r, c);
                    ElementOpCal<DType, op>::apply(dst[idx], src0[idx], scalar, src1[idx]);
                }
            });
        }
    }
}

template <typename tile_shape>
PTO_INTERNAL void TADDSC_IMPL(tile_shape &dst, tile_shape &src0, typename tile_shape::DType scalar, tile_shape &src1)
{
    unsigned row = dst.GetValidRow();
    unsigned col = dst.GetValidCol();
    ElementTileScalarOpWithCarry_Impl<tile_shape, ElementOp::OP_ADDCS>(dst.data(), src0.data(), scalar, src1.data(),
                                                                       row, col);
}

template <typename tile_shape>
PTO_INTERNAL void TSUBSC_IMPL(tile_shape &dst, tile_shape &src0, typename tile_shape::DType scalar, tile_shape &src1)
{
    unsigned row = dst.GetValidRow();
    unsigned col = dst.GetValidCol();
    ElementTileScalarOpWithCarry_Impl<tile_shape, ElementOp::OP_SUBCS>(dst.data(), src0.data(), scalar, src1.data(),
                                                                       row, col);
}
} // namespace pto
#endif
