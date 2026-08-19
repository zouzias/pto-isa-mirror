/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef TFMOD_CPU_HPP
#define TFMOD_CPU_HPP

#include <cmath>

#include "pto/cpu/ElementTileOp.h"
#include "pto/cpu/TBinSOps.hpp"

namespace pto {
template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1>
PTO_INTERNAL void TFMOD_IMPL(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1)
{
    const unsigned rows = dst.GetValidRow();
    const unsigned cols = dst.GetValidCol();
    auto compute = [&](std::size_t r, std::size_t c) {
        const auto lhs = static_cast<double>(src0.data()[GetTileElementOffset<TileDataSrc0>(r, c)]);
        const auto rhs = static_cast<double>(src1.data()[GetTileElementOffset<TileDataSrc1>(r, c)]);
        dst.data()[GetTileElementOffset<TileDataDst>(r, c)] =
            static_cast<typename TileDataDst::DType>(std::fmod(lhs, rhs));
    };
    if constexpr (TileDataDst::isRowMajor) {
        cpu::parallel_for_rows(rows, cols, [&](std::size_t r) {
            for (std::size_t c = 0; c < cols; ++c) {
                compute(r, c);
            }
        });
    } else {
        cpu::parallel_for_rows(cols, rows, [&](std::size_t c) {
            for (std::size_t r = 0; r < rows; ++r) {
                compute(r, c);
            }
        });
    }
}
} // namespace pto

#endif
