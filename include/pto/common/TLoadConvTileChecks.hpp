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

#ifndef PTO_TLOAD_CONVTILE_CHECKS_HPP
#define PTO_TLOAD_CONVTILE_CHECKS_HPP

#include <type_traits>

namespace pto {

template <typename TileData, typename GlobalData>
PTO_INTERNAL constexpr void CheckConvTileDataCommon()
{
    static_assert(
        std::is_same_v<typename TileData::DType, int8_t> || std::is_same_v<typename TileData::DType, uint8_t> ||
            std::is_same_v<typename TileData::DType, int16_t> || std::is_same_v<typename TileData::DType, uint16_t> ||
            std::is_same_v<typename TileData::DType, int32_t> || std::is_same_v<typename TileData::DType, uint32_t> ||
            std::is_same_v<typename TileData::DType, half> || std::is_same_v<typename TileData::DType, bfloat16_t> ||
            std::is_same_v<typename TileData::DType, float>,
        "Fix: Data type must be "
        "int8_t/uint8_t/int16_t/uint16_t/int32_t/uint32_t/half/"
        "bfloat16_t/float!");
    static_assert(TileData::Loc == pto::TileType::Mat, "Fix: Dst TileType must be Mat!");
    static_assert(sizeof(typename TileData::DType) == sizeof(typename GlobalData::DType),
                  "Fix: Source dtype must be same with dst dtype!");
}

} // namespace pto

#endif // PTO_TLOAD_CONVTILE_CHECKS_HPP
