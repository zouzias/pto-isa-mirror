/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_TMOV_CHECKS_HPP
#define PTO_TMOV_CHECKS_HPP

#include <type_traits>

namespace pto {

template <typename DstTileData, typename SrcTileData>
struct TMovToBtChecks {
    using SrcType = typename SrcTileData::DType;
    using DstType = typename DstTileData::DType;
    static constexpr int BURST_LEN_UNIT = 64;

    static_assert((std::is_same_v<SrcType, int32_t> || std::is_same_v<SrcType, float>) ?
                      std::is_same_v<DstType, SrcType> :
                      true,
                  "TMov: Destination and Source tile data types must be the same.");
    static_assert(std::is_same_v<SrcType, half> ? std::is_same_v<DstType, float> : true,
                  "TMov: When Source tile data types is half, dst tile data types must be float");
    static_assert(SrcTileData::Rows == 1, "TMov: When TileType is Bias, row must be 1");
    static_assert(SrcTileData::Cols * sizeof(SrcType) % BURST_LEN_UNIT == 0,
                  "TMov: When TileType is Bias, col * sizeof(srcDType) must be aligned to 64");
};

} // namespace pto

#endif // PTO_TMOV_CHECKS_HPP
