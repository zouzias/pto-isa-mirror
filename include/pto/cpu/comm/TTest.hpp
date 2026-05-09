/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_TTEST_HPP
#define PTO_TTEST_HPP

#include "pto/comm/comm_types.hpp"

namespace pto {
namespace comm {

PTO_INTERNAL bool TestCompareSignal(int32_t sigVal, int32_t cmpVal, WaitCmp cmp)
{
    switch (cmp) {
        case WaitCmp::EQ:
            return sigVal == cmpVal;
        case WaitCmp::NE:
            return sigVal != cmpVal;
        case WaitCmp::GT:
            return sigVal > cmpVal;
        case WaitCmp::GE:
            return sigVal >= cmpVal;
        case WaitCmp::LT:
            return sigVal < cmpVal;
        case WaitCmp::LE:
            return sigVal <= cmpVal;
        default:
            return false;
    }
}

PTO_INTERNAL bool TestPartSignal(volatile int32_t *basePtr, int32_t cmpValue, WaitCmp cmp, int d0, int st0, int d1,
                                 int st1, int d2, int st2, int s3, int st3, int s4, int st4)
{
    for (int d3 = 0; d3 < s3; ++d3) {
        for (int d4 = 0; d4 < s4; ++d4) {
            const int idx = d0 * st0 + d1 * st1 + d2 * st2 + d3 * st3 + d4 * st4;
            if (!TestCompareSignal(basePtr[idx], cmpValue, cmp)) {
                return false;
            }
        }
    }
    return true;
}

template <typename GlobalSignalData>
PTO_INTERNAL bool TTEST_IMPL(GlobalSignalData &signalData, int32_t cmpValue, WaitCmp cmp)
{
    const auto dims = GetTTestDims(signalData);
    const int s0 = dims.s0, s1 = dims.s1, s2 = dims.s2, s3 = dims.s3, s4 = dims.s4;
    const int st0 = dims.st0, st1 = dims.st1, st2 = dims.st2, st3 = dims.st3, st4 = dims.st4;

    volatile int32_t *basePtr = (volatile int32_t *)signalData.data();

    // Test if all signals satisfy the condition (full 5-D traversal)
    for (int d0 = 0; d0 < s0; ++d0) {
        for (int d1 = 0; d1 < s1; ++d1) {
            for (int d2 = 0; d2 < s2; ++d2) {
                bool valid = TestPartSignal(basePtr, cmpValue, cmp, d0, st0, d1, st1, d2, st2, s3, st3, s4, st4);
                if (!valid) {
                    return false;
                }
            }
        }
    }
    return true;
}

} // namespace comm
} // namespace pto

#endif // PTO_COMM_TTEST_HPP
