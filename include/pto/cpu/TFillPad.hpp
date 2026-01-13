/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_CPU_TFILLPAD_HPP
#define PTO_CPU_TFILLPAD_HPP

#include <algorithm>
#include <limits>
#include <type_traits>

#include <pto/common/pto_tile.hpp>

#include "pto/cpu/parallel.hpp"
#include "pto/cpu/tile_offsets.hpp"

namespace pto {

template <typename TileDataDst>
PTO_INTERNAL typename TileDataDst::DType GetFillPadValue()
{
    using T = typename TileDataDst::DType;
    static_assert(TileDataDst::PadVal != PadValue::Null, "TFILLPAD: dst tile pad value can't be Null.");

    if constexpr (std::numeric_limits<T>::has_infinity) {
        if constexpr (TileDataDst::PadVal == PadValue::Max) {
            return std::numeric_limits<T>::infinity();
        }
        if constexpr (TileDataDst::PadVal == PadValue::Min) {
            return -std::numeric_limits<T>::infinity();
        }
        return static_cast<T>(0);
    } else {
        if constexpr (TileDataDst::PadVal == PadValue::Max) {
            return std::numeric_limits<T>::max();
        }
        if constexpr (TileDataDst::PadVal == PadValue::Min) {
            return std::numeric_limits<T>::min();
        }
        return static_cast<T>(0);
    }
}

template <typename TileDataDst, typename TileDataSrc>
PTO_INTERNAL void TFillPad_Impl(TileDataDst &dst, TileDataSrc &src)
{
    using DstT = typename TileDataDst::DType;
    using SrcT = typename TileDataSrc::DType;
    static_assert(sizeof(DstT) == sizeof(SrcT), "TFILLPAD: src/dst dtype sizes must match.");
    static_assert(sizeof(DstT) == 4 || sizeof(DstT) == 2 || sizeof(DstT) == 1, "TFILLPAD: invalid data type.");

    const unsigned validSrcRow = src.GetValidRow();
    const unsigned validSrcCol = src.GetValidCol();
    const auto padVal = GetFillPadValue<TileDataDst>();

    cpu::parallel_for_1d(
        0,
        TileDataDst::Rows,
        static_cast<std::size_t>(TileDataDst::Rows) * TileDataDst::Cols,
        [&](std::size_t r) {
            PTO_CPU_VECTORIZE_LOOP
            for (std::size_t c = 0; c < TileDataDst::Cols; ++c) {
                const auto dstOff = GetTileElementOffset<TileDataDst>(r, c);
                if (r < static_cast<std::size_t>(validSrcRow) && c < static_cast<std::size_t>(validSrcCol)) {
                    dst.data()[dstOff] =
                        static_cast<DstT>(src.data()[GetTileElementOffset<TileDataSrc>(r, c)]);
                } else {
                    dst.data()[dstOff] = padVal;
                }
            }
        });
}

template <typename TileDataDst, typename TileDataSrc>
PTO_INTERNAL void TFILLPAD_IMPL(TileDataDst &dst, TileDataSrc &src)
{
    static_assert(TileDataDst::Cols == TileDataSrc::Cols && TileDataDst::Rows == TileDataSrc::Rows,
        "TFILLPAD: dst/src must have the same rows/cols.");
    TFillPad_Impl(dst, src);
}

template <typename TileDataDst, typename TileDataSrc>
PTO_INTERNAL void TFILLPAD_INPLACE_IMPL(TileDataDst &dst, TileDataSrc &src)
{
    static_assert(TileDataDst::Cols == TileDataSrc::Cols && TileDataDst::Rows == TileDataSrc::Rows,
        "TFILLPAD_INPLACE: dst/src must have the same rows/cols.");
    TFillPad_Impl(dst, src);
}

template <typename TileDataDst, typename TileDataSrc>
PTO_INTERNAL void TFILLPAD_EXPAND_IMPL(TileDataDst &dst, TileDataSrc &src)
{
    static_assert(TileDataDst::Cols >= TileDataSrc::Cols && TileDataDst::Rows >= TileDataSrc::Rows,
        "TFILLPAD_EXPAND: dst rows/cols must be >= src rows/cols.");
    TFillPad_Impl(dst, src);
}

} // namespace pto

#endif
