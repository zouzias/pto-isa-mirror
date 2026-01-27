/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMM_TGET_HPP
#define PTO_COMM_TGET_HPP

#include "pto/common/debug.h"
#include "pto/common/type.hpp"
#include "pto/common/constants.hpp"
#include "pto/comm/comm_types.hpp"

namespace pto {
namespace comm {

// ============================================================================
// TGET: Remote read operation - read remote PE's data to local memory
// 
// Native implementation using Ascend intrinsics.
// Data flow: srcGlobal (remote GM) -> ubTile (UB) -> dstGlobal (local GM)
//
// Parameters:
//   - dstGlobal: Destination GlobalTensor on local PE
//   - srcGlobal: Source GlobalTensor on remote PE
//   - ubTile: UB tile for data staging (must be pre-allocated by compiler)
//
// Note: UB tile must be passed as parameter. The compiler is responsible for
// UB allocation and scheduling.
// ============================================================================

template <typename GlobalDstData, typename GlobalSrcData, typename TileData>
PTO_INTERNAL void TGET_IMPL(GlobalDstData &dstGlobal, GlobalSrcData &srcGlobal, TileData &ubTile)
{
    using T = typename GlobalSrcData::RawDType;
    
    static_assert(std::is_same_v<T, typename GlobalDstData::RawDType>,
        "TGET: src/dst element type mismatch");
    static_assert(GlobalSrcData::layout == GlobalDstData::layout,
        "TGET: src/dst layout mismatch");
    static_assert(std::is_same_v<T, typename TileData::DType>,
        "TGET: TileData element type must match GlobalData element type");

    // Load from remote GM to UB
    TLOAD(ubTile, srcGlobal);
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    
    // Store from UB to local GM
    TSTORE(dstGlobal, ubTile);
}

} // namespace comm
} // namespace pto

#endif // PTO_COMM_TGET_HPP
