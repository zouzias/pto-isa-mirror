/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMM_TWAIT_HPP
#define PTO_COMM_TWAIT_HPP

#include "pto/common/type.hpp"
#include "pto/comm/comm_types.hpp"

namespace pto {
namespace comm {

// ============================================================================
// TWAIT: Wait until signal(s) meet comparison condition
// 
// Native implementation using Ascend intrinsics.
// Used in conjunction with TNOTIFY for synchronization.
// Signal is stored in GlobalTensor data structure.
//
// Parameters:
//   - signal: GlobalTensor containing the signal to wait on (local memory)
//   - cmpValue: Value to compare against
//
// Signal type must be int32_t.
// Note: This instruction does not require UB allocation.
// ============================================================================

namespace detail {

// Helper: Compare signal value with compile-time comparison operator
template <WaitCmp cmp>
PTO_INTERNAL bool CompareSignal(int32_t sigVal, int32_t cmpVal)
{
    if constexpr (cmp == WaitCmp::EQ) {
        return sigVal == cmpVal;
    } else if constexpr (cmp == WaitCmp::NE) {
        return sigVal != cmpVal;
    } else if constexpr (cmp == WaitCmp::GT) {
        return sigVal > cmpVal;
    } else if constexpr (cmp == WaitCmp::GE) {
        return sigVal >= cmpVal;
    } else if constexpr (cmp == WaitCmp::LT) {
        return sigVal < cmpVal;
    } else if constexpr (cmp == WaitCmp::LE) {
        return sigVal <= cmpVal;
    } else {
        return false;
    }
}

// Helper: Compare signal value with runtime comparison operator
PTO_INTERNAL bool CompareSignalRuntime(int32_t sigVal, WaitCmp cmp, int32_t cmpVal)
{
    switch (cmp) {
        case WaitCmp::EQ: return sigVal == cmpVal;
        case WaitCmp::NE: return sigVal != cmpVal;
        case WaitCmp::GT: return sigVal > cmpVal;
        case WaitCmp::GE: return sigVal >= cmpVal;
        case WaitCmp::LT: return sigVal < cmpVal;
        case WaitCmp::LE: return sigVal <= cmpVal;
        default: return false;
    }
}

} // namespace detail

// Compile-time specified comparison (recommended, zero overhead)
template <WaitCmp cmp = WaitCmp::EQ, typename GlobalSignalData>
PTO_INTERNAL void TWAIT_IMPL(GlobalSignalData &signal, int32_t cmpValue)
{
    static_assert(sizeof(typename GlobalSignalData::DType) == sizeof(int32_t),
        "TWAIT: signal type must be 32-bit (int32_t)");

    volatile int32_t *sigPtr = reinterpret_cast<volatile int32_t*>(signal.data());

    while (!detail::CompareSignal<cmp>(*sigPtr, cmpValue)) {
        // Spin wait with memory fence
        pipe_barrier(PIPE_ALL);
    }
}

// Runtime specified comparison
template <typename GlobalSignalData>
PTO_INTERNAL void TWAIT_IMPL(GlobalSignalData &signal, WaitCmp cmp, int32_t cmpValue)
{
    static_assert(sizeof(typename GlobalSignalData::DType) == sizeof(int32_t),
        "TWAIT: signal type must be 32-bit (int32_t)");

    volatile int32_t *sigPtr = reinterpret_cast<volatile int32_t*>(signal.data());

    while (!detail::CompareSignalRuntime(*sigPtr, cmp, cmpValue)) {
        pipe_barrier(PIPE_ALL);
    }
}

// TWAIT_ALL_IMPL: Wait for all signals in array to meet condition
template <WaitCmp cmp = WaitCmp::EQ, typename GlobalSignalData>
PTO_INTERNAL void TWAIT_ALL_IMPL(GlobalSignalData *signals, int count, int32_t cmpValue)
{
    static_assert(sizeof(typename GlobalSignalData::DType) == sizeof(int32_t),
        "TWAIT_ALL: signal type must be 32-bit (int32_t)");

    for (int i = 0; i < count; ++i) {
        TWAIT_IMPL<cmp>(signals[i], cmpValue);
    }
}

// Runtime specified comparison
template <typename GlobalSignalData>
PTO_INTERNAL void TWAIT_ALL_IMPL(GlobalSignalData *signals, int count, WaitCmp cmp, int32_t cmpValue)
{
    static_assert(sizeof(typename GlobalSignalData::DType) == sizeof(int32_t),
        "TWAIT_ALL: signal type must be 32-bit (int32_t)");

    for (int i = 0; i < count; ++i) {
        TWAIT_IMPL(signals[i], cmp, cmpValue);
    }
}

} // namespace comm
} // namespace pto

#endif // PTO_COMM_TWAIT_HPP
