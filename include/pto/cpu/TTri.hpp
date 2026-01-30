/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_CPU_TTRI_HPP
#define PTO_CPU_TTRI_HPP

#include <pto/common/pto_tile.hpp>
#include "pto/cpu/tile_offsets.hpp"

namespace pto {

// CPU-SIM implementation of TTRI: generate lower/upper triangular mask in-place.
// isUpperOrLower: 0 = lower-triangular, 1 = upper-triangular.
template <typename TileData, int isUpperOrLower>
PTO_INTERNAL void TTRI_IMPL(TileData &dst, int diagonal)
{
    using T = typename TileData::DType;
    const std::size_t rows = static_cast<std::size_t>(dst.GetValidRow());
    const std::size_t cols = static_cast<std::size_t>(dst.GetValidCol());

    // Default to zeros.
    for (std::size_t r = 0; r < rows; ++r) {
        for (std::size_t c = 0; c < cols; ++c) {
            dst.data()[GetTileElementOffset<TileData>(r, c)] = static_cast<T>(0);
        }
    }

    for (std::size_t r = 0; r < rows; ++r) {
        for (std::size_t c = 0; c < cols; ++c) {
            const int rr = static_cast<int>(r);
            const int cc = static_cast<int>(c);
            bool one = false;
            if constexpr (isUpperOrLower == 0) {
                // lower: 1 if c <= r + diagonal
                one = (cc <= rr + diagonal);
            } else {
                // upper: 0 if c < r + diagonal, else 1
                one = (cc >= rr + diagonal);
            }
            dst.data()[GetTileElementOffset<TileData>(r, c)] = static_cast<T>(one ? 1 : 0);
        }
    }
}

} // namespace pto

#endif
