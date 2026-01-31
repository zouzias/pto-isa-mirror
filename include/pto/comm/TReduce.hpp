/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMM_TREDUCE_HPP
#define PTO_COMM_TREDUCE_HPP

#include <type_traits>

#include "pto/common/debug.h"
#include "pto/common/type.hpp"
#include "pto/common/constants.hpp"
#include "pto/comm/comm_types.hpp"

namespace pto {
namespace comm {

// ============================================================================
// TREDUCE_IMPL: Reduce operation - root gathers and reduces data from all ranks
// 
// The calling NPU is the root and gathers data from all ranks, performing
// element-wise reduction locally.
// ============================================================================

namespace detail {

// Element-wise reduction helper
template <typename TileData>
PTO_INTERNAL void ReduceTiles(TileData &acc, TileData &recv, ReduceOp op)
{
    // Perform element-wise reduction based on op
    switch (op) {
        case ReduceOp::Sum:
            TADD(acc, acc, recv);
            break;
        case ReduceOp::Max:
            TMAX(acc, acc, recv);
            break;
        case ReduceOp::Min:
            TMIN(acc, acc, recv);
            break;
    }
}

} // namespace detail

template <typename ParallelGroupType, typename GlobalDstData, typename TileData>
PTO_INTERNAL void TREDUCE_IMPL(ParallelGroupType &parallelGroup, GlobalDstData &dstGlobalData, 
                               TileData &accTileData, TileData &recvTileData, ReduceOp op)
{
    using GlobalSrcData = typename ParallelGroupTraits<ParallelGroupType>::GlobalDataType;
    using T = typename GlobalSrcData::RawDType;
    
    // Type checks
    static_assert(std::is_same_v<T, typename GlobalDstData::RawDType>, 
        "TREDUCE: GlobalData type mismatch!");
    static_assert(std::is_same_v<T, typename TileData::DType>, 
        "TREDUCE: TileData element type must match GlobalData element type");
    
    const int my_rank = parallelGroup.GetRank();
    const int nranks = parallelGroup.GetSize();

    // Check PG size 
    PTO_ASSERT(nranks > 0, "ParallelGroup size must be greater than 0!");

    // Single rank case: just copy local data to output
    if (nranks == 1) {
        TLOAD(accTileData, parallelGroup[my_rank]);
        set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        TSTORE(dstGlobalData, accTileData);
        return;
    }

    // Step 1: Load local data into accumulator
    TLOAD(accTileData, parallelGroup[my_rank]);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    // Step 2: Reduce data from all other ranks
    for (int r = 0; r < nranks; ++r) {
        if (r == my_rank) {
            continue;  // Skip self, already loaded
        }

        // Load remote data into receive buffer
        TLOAD(recvTileData, parallelGroup[r]);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);

        // Perform reduction
        detail::ReduceTiles(accTileData, recvTileData, op);
        
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    }

    // Step 3: Store final result
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(dstGlobalData, accTileData);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
}

} // namespace comm
} // namespace pto

#endif // PTO_COMM_TREDUCE_HPP
