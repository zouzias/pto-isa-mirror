/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TBROADCAST_HPP
#define TBROADCAST_HPP

#include "pto/cpu/tile_offsets.hpp"
#include <pto/common/pto_tile.hpp>
#include <type_traits>

namespace pto {

template <typename TileDataSrc, typename TileDataDst>
PTO_INTERNAL void TBROADCAST_IMPL(TileDataDst &dst, TileDataSrc &src, size_t numProc)
{
    const unsigned validRow = src.GetValidRow();
    const unsigned validCol = src.GetValidCol();
    if (validRow == 0 || validCol == 0 || numProc == 0) {
        return;
    }
    for (unsigned n = 0; n < numProc; ++n) {
        for (unsigned i = 0; i < validRow; ++i) {
            for (unsigned j = 0; j < validCol; ++j) {
                const size_t srcOff = GetTileElementOffset<TileDataSrc>(i, j);
                const auto dstRow = static_cast<unsigned>(n * validRow + i);
                const size_t dstOff = GetTileElementOffset<TileDataDst>(dstRow, j);
                dst.data()[dstOff] = src.data()[srcOff];
            }
        }
    }
}

} // namespace pto

#endif
