/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMM_TNOTIFY_HPP
#define PTO_COMM_TNOTIFY_HPP

#include "pto/common/type.hpp"
#include "pto/comm/comm_types.hpp"

namespace pto {
namespace comm {

// ============================================================================
// TNOTIFY: Send flag notification to remote PE
// 
// Native implementation using Ascend intrinsics.
// dstSignal's data() pointer should be a remote address obtained via ShmemPtr.
//
// Parameters:
//   - dstSignal: Remote signal GlobalTensor (contains target address)
//   - value: Value to set/accumulate
//
// Signal type must be int32_t.
// Note: This instruction does not require UB allocation.
// ============================================================================

// Compile-time specified NotifyOp (recommended, zero overhead)
template <NotifyOp op = NotifyOp::Set, typename GlobalSignalData>
PTO_INTERNAL void TNOTIFY_IMPL(GlobalSignalData &dstSignal, int32_t value = 1)
{
    static_assert(sizeof(typename GlobalSignalData::DType) == sizeof(int32_t),
        "TNOTIFY: signal type must be 32-bit (int32_t)");

    volatile __gm__ int32_t *sigPtr = (volatile __gm__ int32_t *)dstSignal.data();

    if constexpr (op == NotifyOp::AtomicAdd) {
        // Atomic add using hardware atomic instruction
        set_st_atomic_cfg(ATOMIC_S32, ATOMIC_SUM);
        st_atomic<int32_t>(value, (__gm__ int32_t *)sigPtr);
    } else {
        // Set operation - direct store to remote memory
        *sigPtr = value;
    }
    
    pipe_barrier(PIPE_ALL);
}

// Runtime specified NotifyOp version
template <typename GlobalSignalData>
PTO_INTERNAL void TNOTIFY_IMPL(GlobalSignalData &dstSignal, int32_t value, NotifyOp op)
{
    static_assert(sizeof(typename GlobalSignalData::DType) == sizeof(int32_t),
        "TNOTIFY: signal type must be 32-bit (int32_t)");

    volatile __gm__ int32_t *sigPtr = (volatile __gm__ int32_t *)dstSignal.data();

    if (op == NotifyOp::AtomicAdd) {
        set_st_atomic_cfg(ATOMIC_S32, ATOMIC_SUM);
        st_atomic<int32_t>(value, (__gm__ int32_t *)sigPtr);
    } else {
        *sigPtr = value;
    }
    
    pipe_barrier(PIPE_ALL);
}

} // namespace comm
} // namespace pto

#endif // PTO_COMM_TNOTIFY_HPP
