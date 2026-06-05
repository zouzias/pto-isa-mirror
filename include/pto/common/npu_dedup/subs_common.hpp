/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMMON_NPU_DEDUP_SUBS_COMMON_HPP
#define PTO_COMMON_NPU_DEDUP_SUBS_COMMON_HPP

namespace pto {

template <typename DstTile, typename SrcTile>
PTO_INTERNAL void TSubSCheckBase()
{
    using T = typename DstTile::DType;
    static_assert(std::is_same_v<T, int32_t> || std::is_same_v<T, int16_t> || std::is_same_v<T, int8_t> ||
                      std::is_same_v<T, uint32_t> || std::is_same_v<T, uint16_t> || std::is_same_v<T, uint8_t> ||
                      std::is_same_v<T, half> || std::is_same_v<T, float32_t>,
                  "TSUBS: Invalid data type");
    static_assert((DstTile::Loc == TileType::Vec) && (SrcTile::Loc == TileType::Vec),
                  "TileType of dst and src tiles must be TileType::Vec.");
    static_assert((DstTile::ValidCol <= DstTile::Cols) && (DstTile::ValidRow <= DstTile::Rows) &&
                      (SrcTile::ValidCol <= SrcTile::Cols) && (SrcTile::ValidRow <= SrcTile::Rows),
                  "Number of valid columns and rows must not be greater than number of tile columns and rows.");
}

} // namespace pto

#endif
