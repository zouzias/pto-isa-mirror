/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TBINS_HPP
#define TBINS_HPP

#include <algorithm>
#include <cmath>
#include <pto/common/pto_tile.hpp>
#include "pto/cpu/tile_offsets.hpp"

namespace pto {
template <typename TileDataDst, typename TileDataSrc>
PTO_INTERNAL void TAXPY_IMPL(TileDataDst &dst, TileDataSrc &src, typename TileDataSrc::DType scalar)
{
    unsigned row = dst.GetValidRow();
    unsigned col = dst.GetValidCol();
    for (size_t c = 0; c < col; ++c) {
        for (size_t r = 0; r < row; ++r) {
            const size_t dstIdx = GetTileElementOffset<TileDataDst>(r, c);
            const size_t srcIdx = GetTileElementOffset<TileDataSrc>(r, c);
            dst.data()[dstIdx] = static_cast<typename TileDataDst::DType>(
                dst.data()[dstIdx] + static_cast<typename TileDataDst::DType>(src.data()[srcIdx]) *
                                         static_cast<typename TileDataDst::DType>(scalar));
        }
    }
}
} // namespace pto

#endif
