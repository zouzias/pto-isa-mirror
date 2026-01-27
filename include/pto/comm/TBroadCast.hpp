/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMM_TBROADCAST_HPP
#define PTO_COMM_TBROADCAST_HPP

#include "pto/common/debug.h"
#include "pto/common/type.hpp"
#include "pto/common/constants.hpp"
#include "pto/comm/comm_types.hpp"

namespace pto {
namespace comm {

// ============================================================================
// TBROADCAST: Broadcast data from root rank to all ranks
// 
// Native implementation using Ascend intrinsics.
//
// Parameters:
//   - pg: ParallelGroup containing GlobalTensors from all participating ranks
//   - srcGlobal: Source GlobalTensor (data from root rank)
//   - root: Root rank index that provides the data
//   - ubTile: UB tile for data staging (must be pre-allocated by compiler)
//
// Note: UB tile must be passed as parameter. The compiler is responsible for
// UB allocation and scheduling.
// ============================================================================

template <typename ParallelGroupType, typename GlobalSrcData, typename TileData>
PTO_INTERNAL void TBROADCAST_IMPL(ParallelGroupType &pg, GlobalSrcData &srcGlobal, int root, TileData &ubTile)
{
    using GlobalDstData = typename ParallelGroupTraits<ParallelGroupType>::GlobalDataType;
    using T = typename GlobalSrcData::RawDType;

    static_assert(std::is_same_v<T, typename TileData::DType>,
        "TBROADCAST: TileData element type must match GlobalData element type");

    const int my_rank = pg.GetRank();
    const int nranks = pg.GetSize();

    PTO_ASSERT(nranks > 0, "ParallelGroup size must be greater than 0!");
    PTO_ASSERT(root >= 0 && root < nranks, "Root rank must be valid!");

    if (nranks == 1) {
        return; // Nothing to broadcast
    }

    if (my_rank == root) {
        // Root loads data to UB once
        TLOAD(ubTile, srcGlobal);
        set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        
        // Broadcast to all ranks
        for (int r = 0; r < nranks; ++r) {
            TSTORE(pg[r], ubTile);
            if (r < nranks - 1) {
                set_flag(PIPE_MTE3, PIPE_MTE3, EVENT_ID0);
                wait_flag(PIPE_MTE3, PIPE_MTE3, EVENT_ID0);
            }
        }
    }
    
    // Synchronization point (barrier)
    // Note: In a real multi-device scenario, this would use hardware sync
}

} // namespace comm
} // namespace pto

#endif // PTO_COMM_TBROADCAST_HPP
