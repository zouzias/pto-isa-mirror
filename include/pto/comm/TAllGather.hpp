/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMM_TALLGATHER_HPP
#define PTO_COMM_TALLGATHER_HPP

#include <type_traits>

#include "pto/common/debug.h"
#include "pto/common/type.hpp"
#include "pto/common/constants.hpp"
#include "pto/comm/comm_types.hpp"

namespace pto {
namespace comm {

// ============================================================================
// TALLGATHER: All-gather operation across parallel group
// 
// Native implementation: Gathers data from all ranks and distributes to all ranks.
// Each rank contributes its local data, and all ranks receive the combined data.
//
// Parameters:
//   - pg: ParallelGroup containing GlobalTensors from all participating ranks
//   - dstGlobal: Destination GlobalTensor for gathered result (size = nranks * per_rank_size)
//   - ubTile: UB tile for data staging (must be pre-allocated by compiler)
//
// Note: UB tile must be passed as parameter. The compiler is responsible for
// UB allocation and scheduling.
// ============================================================================

template <typename ParallelGroupType, typename GlobalDstData, typename TileData>
PTO_INTERNAL void TALLGATHER_IMPL(ParallelGroupType &pg, GlobalDstData &dstGlobal, TileData &ubTile)
{
    using GlobalSrcData = typename ParallelGroupTraits<ParallelGroupType>::GlobalDataType;
    using T = typename GlobalSrcData::RawDType;
    
    static_assert(std::is_same_v<T, typename GlobalDstData::RawDType>, 
        "TALLGATHER: GlobalData type mismatch!");
    static_assert(std::is_same_v<T, typename TileData::DType>,
        "TALLGATHER: TileData element type must match GlobalData element type");

    const int my_rank = pg.GetRank();
    const int nranks = pg.GetSize();

    // Check PG size 
    PTO_ASSERT(nranks > 0, "ParallelGroup size must be greater than 0!");

    if (nranks == 1) {
        // Single rank: just copy local data to output
        TLOAD(ubTile, pg[my_rank]);
        set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        TSTORE(dstGlobal, ubTile);
        return;
    }

    // Calculate per-rank data size
    auto &srcGlobal = pg[my_rank];
    const int elemPerRank = srcGlobal.GetShape(GlobalTensorDim::DIM_0) *
                            srcGlobal.GetShape(GlobalTensorDim::DIM_1) *
                            srcGlobal.GetShape(GlobalTensorDim::DIM_2) *
                            srcGlobal.GetShape(GlobalTensorDim::DIM_3) *
                            srcGlobal.GetShape(GlobalTensorDim::DIM_4);

    // Each rank copies all data to output using staged transfer
    for (int teamRank = 0; teamRank < nranks; ++teamRank) {
        // Load remote/local data to UB
        TLOAD(ubTile, pg[teamRank]);
        set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        
        // Store to appropriate position in output
        // Note: dstGlobal should be set up with proper offset by caller
        // or use view mechanism for different output positions
        TSTORE(dstGlobal, ubTile);
        
        if (teamRank < nranks - 1) {
            set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
            wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        }
    }
}

} // namespace comm
} // namespace pto

#endif // PTO_COMM_TALLGATHER_HPP
