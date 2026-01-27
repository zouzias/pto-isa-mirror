/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMM_TTEST_HPP
#define PTO_COMM_TTEST_HPP

#include "pto/common/type.hpp"
#include "pto/comm/comm_types.hpp"

namespace pto {
namespace comm {

// ============================================================================
// TTEST: Non-blocking test if signal(s) meet comparison condition
// 
// Native implementation using Ascend intrinsics.
// Returns true if condition is satisfied, false otherwise.
// Used for polling-based synchronization with timeout or interleaved work.
//
// Parameters:
//   - signal: GlobalTensor containing the signal to test (local memory)
//   - cmpValue: Value to compare against
//
// Signal type must be int32_t.
// Note: This instruction does not require UB allocation.
// ============================================================================

// Compile-time specified comparison (recommended, zero overhead)
template <WaitCmp cmp = WaitCmp::EQ, typename GlobalSignalData>
PTO_INTERNAL bool TTEST_IMPL(GlobalSignalData &signal, int32_t cmpValue)
{
    static_assert(sizeof(typename GlobalSignalData::DType) == sizeof(int32_t),
        "TTEST: signal type must be 32-bit (int32_t)");

    volatile __gm__ int32_t *sigPtr = (volatile __gm__ int32_t *)signal.data();
    pipe_barrier(PIPE_ALL);
    
    int32_t sigVal = *sigPtr;
    
    if constexpr (cmp == WaitCmp::EQ) {
        return sigVal == cmpValue;
    } else if constexpr (cmp == WaitCmp::NE) {
        return sigVal != cmpValue;
    } else if constexpr (cmp == WaitCmp::GT) {
        return sigVal > cmpValue;
    } else if constexpr (cmp == WaitCmp::GE) {
        return sigVal >= cmpValue;
    } else if constexpr (cmp == WaitCmp::LT) {
        return sigVal < cmpValue;
    } else if constexpr (cmp == WaitCmp::LE) {
        return sigVal <= cmpValue;
    } else {
        return false;
    }
}

// Runtime specified comparison
template <typename GlobalSignalData>
PTO_INTERNAL bool TTEST_IMPL(GlobalSignalData &signal, WaitCmp cmp, int32_t cmpValue)
{
    static_assert(sizeof(typename GlobalSignalData::DType) == sizeof(int32_t),
        "TTEST: signal type must be 32-bit (int32_t)");

    volatile __gm__ int32_t *sigPtr = (volatile __gm__ int32_t *)signal.data();
    pipe_barrier(PIPE_ALL);
    
    int32_t sigVal = *sigPtr;
    
    switch (cmp) {
        case WaitCmp::EQ: return sigVal == cmpValue;
        case WaitCmp::NE: return sigVal != cmpValue;
        case WaitCmp::GT: return sigVal > cmpValue;
        case WaitCmp::GE: return sigVal >= cmpValue;
        case WaitCmp::LT: return sigVal < cmpValue;
        case WaitCmp::LE: return sigVal <= cmpValue;
        default: return false;
    }
}

// TTEST_ALL_IMPL: Test if all signals in array meet condition (non-blocking)
template <WaitCmp cmp = WaitCmp::EQ, typename GlobalSignalData>
PTO_INTERNAL bool TTEST_ALL_IMPL(GlobalSignalData *signals, int count, int32_t cmpValue)
{
    static_assert(sizeof(typename GlobalSignalData::DType) == sizeof(int32_t),
        "TTEST_ALL: signal type must be 32-bit (int32_t)");

    for (int i = 0; i < count; ++i) {
        if (!TTEST_IMPL<cmp>(signals[i], cmpValue)) {
            return false;
        }
    }
    return true;
}

// Runtime specified comparison
template <typename GlobalSignalData>
PTO_INTERNAL bool TTEST_ALL_IMPL(GlobalSignalData *signals, int count, WaitCmp cmp, int32_t cmpValue)
{
    static_assert(sizeof(typename GlobalSignalData::DType) == sizeof(int32_t),
        "TTEST_ALL: signal type must be 32-bit (int32_t)");

    for (int i = 0; i < count; ++i) {
        if (!TTEST_IMPL(signals[i], cmp, cmpValue)) {
            return false;
        }
    }
    return true;
}

} // namespace comm
} // namespace pto

#endif // PTO_COMM_TTEST_HPP
