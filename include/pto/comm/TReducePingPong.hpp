/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMM_TREDUCE_PINGPONG_HPP
#define PTO_COMM_TREDUCE_PINGPONG_HPP

#include <type_traits>

#include "pto/common/debug.h"
#include "pto/common/type.hpp"
#include "pto/common/constants.hpp"
#include "pto/comm/comm_types.hpp"

namespace pto {
namespace comm {

// ============================================================================
// TREDUCE_PINGPONG_IMPL: Reduce operation with ping-pong double buffering
// 
// The calling NPU is the root and gathers data from all ranks, performing
// element-wise reduction locally. Uses ping-pong double buffering for better
// performance by overlapping data transfer with computation.
//
// Parameters:
//   - parallelGroup: ParallelGroup containing GlobalTensors from all participating ranks
//   - dstGlobalData: Destination GlobalTensor for reduced result
//   - accTileData: UB tile for accumulator (must be pre-allocated by compiler)
//   - pingTile: UB tile for ping buffer (must be pre-allocated by compiler)
//   - pongTile: UB tile for pong buffer (must be pre-allocated by compiler)
//   - op: Reduction operation (Sum, Max, Min)
//
// Note: All UB tiles must be passed as parameters. The compiler is responsible
// for UB allocation and scheduling.
// ============================================================================

namespace detail {

// Element-wise reduction helper
template <typename TileData>
PTO_INTERNAL void ReduceTilesPingPong(TileData &acc, TileData &recv, ReduceOp op)
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
PTO_INTERNAL void TREDUCE_PINGPONG_IMPL(ParallelGroupType &parallelGroup, GlobalDstData &dstGlobalData, 
                                         TileData &accTileData, TileData &pingTile, TileData &pongTile, 
                                         ReduceOp op)
{
    using GlobalSrcData = typename ParallelGroupTraits<ParallelGroupType>::GlobalDataType;
    using T = typename GlobalSrcData::RawDType;
    
    // Type checks
    static_assert(std::is_same_v<T, typename GlobalDstData::RawDType>, 
        "TREDUCE_PINGPONG: GlobalData type mismatch!");
    static_assert(std::is_same_v<T, typename TileData::DType>, 
        "TREDUCE_PINGPONG: TileData element type must match GlobalData element type");
    
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

    // Build list of remote rank indices (skip self)
    int remoteIdx[16];
    int numRemote = 0;
    for (int i = 0; i < nranks && numRemote < 16; ++i) {
        if (i != my_rank) {
            remoteIdx[numRemote++] = i;
        }
    }

    // Step 1: Load local data into accumulator
    TLOAD(accTileData, parallelGroup[my_rank]);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    // Step 2: Start prefetching first remote data into pingTile
    TLOAD(pingTile, parallelGroup[remoteIdx[0]]);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);

    // Wait for local data ready
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    // Ping-pong processing: overlap data transfer with computation
    for (int i = 0; i < numRemote; ++i) {
        const bool hasNext = (i + 1 < numRemote);
        const bool usePing = (i % 2 == 0);
        
        TileData &currentTile = usePing ? pingTile : pongTile;
        TileData &nextTile = usePing ? pongTile : pingTile;
        const auto currentEvent = usePing ? EVENT_ID1 : EVENT_ID2;
        const auto nextEvent = usePing ? EVENT_ID2 : EVENT_ID1;

        // Start prefetch of next remote data (overlapped with current reduction)
        if (hasNext) {
            TLOAD(nextTile, parallelGroup[remoteIdx[i + 1]]);
            set_flag(PIPE_MTE2, PIPE_V, nextEvent);
        }

        // Wait for current remote data ready
        wait_flag(PIPE_MTE2, PIPE_V, currentEvent);

        // Perform reduction with current remote data
        detail::ReduceTilesPingPong(accTileData, currentTile, op);

        // Sync based on next operation
        if (hasNext) {
            set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        } else {
            // Last iteration: prepare for TSTORE
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        }
    }

    // Step 3: Store final result
    TSTORE(dstGlobalData, accTileData);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
}

} // namespace comm
} // namespace pto

#endif // PTO_COMM_TREDUCE_PINGPONG_HPP
