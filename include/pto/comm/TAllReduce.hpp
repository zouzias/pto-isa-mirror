/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMM_TALLREDUCE_HPP
#define PTO_COMM_TALLREDUCE_HPP

#include <type_traits>

#include "pto/common/debug.h"
#include "pto/common/type.hpp"
#include "pto/common/constants.hpp"
#include "pto/comm/comm_types.hpp"

namespace pto {
namespace comm {

// ============================================================================
// TALLREDUCE: All-reduce operation across parallel group
// 
// Native implementation using Ascend intrinsics with ping-pong double buffering.
// 
// Parameters:
//   - pg: ParallelGroup containing GlobalTensors from all participating ranks
//   - dstGlobal: Destination GlobalTensor for reduced result
//   - accTile: UB tile for accumulator (must be pre-allocated by compiler)
//   - pingTile: UB tile for ping buffer (must be pre-allocated by compiler)
//   - pongTile: UB tile for pong buffer (must be pre-allocated by compiler)
//
// Note: All UB tiles must be passed as parameters. The compiler is responsible
// for UB allocation and scheduling.
// ============================================================================

template <typename ParallelGroupType, typename GlobalDstData, typename TileData>
PTO_INTERNAL void TALLREDUCE_IMPL(ParallelGroupType &pg, GlobalDstData &dstGlobal, 
                                   TileData &accTile, TileData &pingTile, TileData &pongTile)
{
    using GlobalData = typename ParallelGroupTraits<ParallelGroupType>::GlobalDataType;
    using T = typename GlobalData::RawDType;
    
    // Type checks
    static_assert(std::is_same_v<T, typename TileData::DType>, 
        "TALLREDUCE: TileData element type must match GlobalData element type");
    
    const int my_rank = pg.GetRank();
    const int nranks = pg.GetSize();

    // Check PG size 
    PTO_ASSERT(nranks > 0, "ParallelGroup size must be greater than 0!");

    // Single rank case: just copy local data to output
    if (nranks == 1) {
        TLOAD(accTile, pg[0]);
        set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        TSTORE(dstGlobal, accTile);
        return;
    }

    GlobalData srcGlobal;
    // Build list of remote rank indices (skip self)
    int remoteIdx[16];
    int numRemote = 0;
    for (int i = 0; i < nranks && numRemote < 16; ++i) {
        if (pg[i].GetRank() != my_rank) {
            remoteIdx[numRemote++] = i;
        }else{
            srcGlobal = pg[i];
        }
    }

    // Step 1: Load local data into accumulator
    TLOAD(accTile, srcGlobal);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    // Step 2: Start prefetching first remote data into pingTile
    TLOAD(pingTile, pg[remoteIdx[0]]);
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

        // Start prefetch of next remote data (overlapped with current TADD)
        if (hasNext) {
            TLOAD(nextTile, pg[remoteIdx[i + 1]]);
            set_flag(PIPE_MTE2, PIPE_V, nextEvent);
        }

        // Wait for current remote data ready
        wait_flag(PIPE_MTE2, PIPE_V, currentEvent);

        // Add current remote data to accumulator
        TADD(accTile, accTile, currentTile);

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
    TSTORE(dstGlobal, accTile);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
}

} // namespace comm
} // namespace pto

#endif // PTO_COMM_TALLREDUCE_HPP
