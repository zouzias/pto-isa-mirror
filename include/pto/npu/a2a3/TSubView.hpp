/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under
the terms and conditions of CANN Open Software License Agreement Version 2.0
(the "License"). Please refer to the License for details. You may not use this
file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON AN "AS
IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
PARTICULAR PURPOSE. See LICENSE in the root of the software repository for the
full text of the License.
*/

#ifndef TILE_TSUBVIEW_HPP
#define TILE_TSUBVIEW_HPP
#include <pto/common/type.hpp>
#include <cstdint>

template <typename TileDataDst, typename TileDataSrc>
PTO_INTERNAL void TSUBVIEW_IMPL(TileDataDst &dst, TileDataSrc &src, uint16_t rowIdx, uint16_t colIdx)
{
    constexpr int kRowStride = TileDataSrc::RowStride;
    constexpr int kColStride = TileDataSrc::ColStride;
    const uint64_t totalOffset = rowIdx * kRowStride + colIdx * kColStride;
    const uint64_t byteOffset = totalOffset * sizeof(typename TileDataSrc::DType);

    static_assert(TileDataDst::Loc == TileDataSrc::Loc,
                  "The destination and source tiles must have the same TileType!");
    static_assert(TileDataDst::BFractal == TileDataSrc::BFractal,
                  "The destination and source tiles must have the same BFractal");

    constexpr size_t kSrcBufferSize = TileDataSrc::Numel * sizeof(typename TileDataSrc::DType);
    constexpr size_t kDstBufferSize = TileDataDst::Numel * sizeof(typename TileDataDst::DType);
    PTO_ASSERT(kDstBufferSize + byteOffset <= kSrcBufferSize,
               "dst tile's memory range must fully overlap with src tile!");

#ifndef __PTO_AUTO__
    TASSIGN_IMPL(dst, (uint64_t)(src.data() + totalOffset));
#else
    __cce_alias(dst.data(), src.data(), byteOffset);
#endif
}

#endif
