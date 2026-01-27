/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMM_TBARRIER_HPP
#define PTO_COMM_TBARRIER_HPP

#include "pto/common/type.hpp"
#include "pto/comm/comm_types.hpp"

namespace pto {
namespace comm {

// ============================================================================
// TBARRIER: Global barrier synchronization
// 
// Native implementation using Ascend intrinsics.
// Synchronizes all ranks in the communication group. All ranks must call
// this instruction, and no rank will proceed until all ranks have arrived.
//
// Parameters:
//   - barrierSignals: Array of signal GlobalTensors for barrier coordination
//   - nranks: Number of ranks participating in the barrier
//   - my_rank: Current rank ID
//
// Note: barrierSignals must be pre-allocated by compiler.
// ============================================================================

template <typename GlobalSignalData>
PTO_INTERNAL void TBARRIER_IMPL(GlobalSignalData *barrierSignals, int nranks, int my_rank)
{
    static_assert(sizeof(typename GlobalSignalData::DType) == sizeof(int32_t),
        "TBARRIER: signal type must be 32-bit (int32_t)");

    if (nranks <= 1) {
        return;
    }

    volatile int32_t *mySignal = reinterpret_cast<volatile int32_t*>(barrierSignals[my_rank].data());

    // Phase 1: Signal arrival
    *mySignal = 1;
    
    // Memory fence
    pipe_barrier(PIPE_ALL);
    
    // Phase 2: Wait for all ranks to arrive
    for (int i = 0; i < nranks; ++i) {
        volatile int32_t *sigPtr = reinterpret_cast<volatile int32_t*>(barrierSignals[i].data());
        while (*sigPtr == 0) {
            // Spin wait with memory fence
            pipe_barrier(PIPE_ALL);
        }
    }
    
    // Phase 3: Reset flags (rank 0 resets all)
    if (my_rank == 0) {
        for (int i = 0; i < nranks; ++i) {
            volatile int32_t *sigPtr = reinterpret_cast<volatile int32_t*>(barrierSignals[i].data());
            *sigPtr = 0;
        }
    }
    
    // Final memory fence
    pipe_barrier(PIPE_ALL);
}

// Simple barrier without explicit signal array (uses internal synchronization)
PTO_INTERNAL void TBARRIER_IMPL()
{
    // Full pipeline barrier
    pipe_barrier(PIPE_ALL);
}

} // namespace comm
} // namespace pto

#endif // PTO_COMM_TBARRIER_HPP
