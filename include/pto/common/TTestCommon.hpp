/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_TTEST_COMMON_HPP
#define PTO_TTEST_COMMON_HPP

#include "pto/common/type.hpp"

namespace pto {

struct TTestTensorDims {
    int s0, s1, s2, s3, s4;
    int st0, st1, st2, st3, st4;
};

template <typename GlobalSignalData>
PTO_INTERNAL TTestTensorDims GetTTestDims(GlobalSignalData &signalData)
{
    static_assert(sizeof(typename GlobalSignalData::DType) == sizeof(int32_t),
                  "TTEST: signal type must be 32-bit (int32_t)");
    TTestTensorDims d;
    d.s0 = signalData.GetShape(GlobalTensorDim::DIM_0);
    d.s1 = signalData.GetShape(GlobalTensorDim::DIM_1);
    d.s2 = signalData.GetShape(GlobalTensorDim::DIM_2);
    d.s3 = signalData.GetShape(GlobalTensorDim::DIM_3);
    d.s4 = signalData.GetShape(GlobalTensorDim::DIM_4);
    d.st0 = signalData.GetStride(GlobalTensorDim::DIM_0);
    d.st1 = signalData.GetStride(GlobalTensorDim::DIM_1);
    d.st2 = signalData.GetStride(GlobalTensorDim::DIM_2);
    d.st3 = signalData.GetStride(GlobalTensorDim::DIM_3);
    d.st4 = signalData.GetStride(GlobalTensorDim::DIM_4);
    return d;
}

} // namespace pto

#endif // PTO_TTEST_COMMON_HPP
