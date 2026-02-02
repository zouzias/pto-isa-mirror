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
#include "pto/common/utils.hpp"
#include "pto/comm/comm_types.hpp"

namespace pto {
namespace comm {

// ============================================================================
// TWAIT_IMPL: Blocking wait until signal(s) meet comparison condition
// 
// Signal type must be int32_t.
// For signal matrix: Shape determines the 2D region to wait on. All signals must satisfy.
// ============================================================================

namespace detail {

// Helper: Compare signal value with runtime comparison operator
PTO_INTERNAL bool CompareSignalRuntime(int32_t sigVal, int32_t cmpVal, WaitCmp cmp)
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

template <typename GlobalSignalData>
PTO_INTERNAL void TWAIT_IMPL(GlobalSignalData &signalData, int32_t cmpValue, WaitCmp cmp)
{
    static_assert(sizeof(typename GlobalSignalData::DType) == sizeof(int32_t),
        "TWAIT: signal type must be 32-bit (int32_t)");

    // Get signal matrix dimensions from GlobalTensor shape
    const int rows = signalData.GetShape(GlobalTensorDim::DIM_3);
    const int cols = signalData.GetShape(GlobalTensorDim::DIM_4);
    const int totalSignals = rows * cols;

    volatile __gm__ int32_t *basePtr = reinterpret_cast<volatile __gm__ int32_t*>(signalData.data());

    // Wait until all signals in the matrix satisfy the condition
    bool allSatisfied = false;
    uint32_t spin = 0;
    constexpr uint32_t kFenceInterval = 64;
    while (!allSatisfied) {
        allSatisfied = true;
        for (int i = 0; i < totalSignals; ++i) {
            __asm__ __volatile__("");
            dcci((__gm__ void *)(basePtr + i), SINGLE_CACHE_LINE);
            __asm__ __volatile__("");
            if (!detail::CompareSignalRuntime(basePtr[i], cmpValue, cmp)) {
                allSatisfied = false;
                break;
            }
        }
        if (!allSatisfied) {
            // Spin wait with periodic memory fence to reduce contention
            if ((++spin % kFenceInterval) == 0) {
                pipe_barrier(PIPE_ALL);
            }
        }
    }
}

} // namespace comm
} // namespace pto

#endif // PTO_COMM_TWAIT_HPP
