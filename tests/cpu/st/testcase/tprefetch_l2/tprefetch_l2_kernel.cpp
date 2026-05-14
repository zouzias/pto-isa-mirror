/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <pto/pto-inst.hpp>
#include <pto/common/constants.hpp>

using namespace pto;

template <typename T, int kGRows_, int kGCols_>
AICORE void runTPrefetchL2GlobalTensor(__gm__ T __in__ *input)
{
    using GShape = Shape<1, 1, 1, kGRows_, kGCols_>;
    using GStride = Stride<1, 1, 1, kGCols_, 1>;
    using GlobalData = GlobalTensor<T, GShape, GStride>;

    GlobalData inputGlobal(input);
    // Pass nullptr workspace so the IMPL early-returns an empty AsyncEvent;
    // CPU sim has no L2 cache concept and the IMPL itself is a no-op anyway.
    auto evt = pto::TPREFETCH_L2(inputGlobal, static_cast<__gm__ uint8_t *>(nullptr));
    (void)evt;
}

template <typename T, int kGRows_, int kGCols_>
AICORE void runTPrefetchL2RawPtr(__gm__ T __in__ *input)
{
    uint64_t bytes = static_cast<uint64_t>(kGRows_) * kGCols_ * sizeof(T);
    auto evt = pto::TPREFETCH_L2((__gm__ void *)input, bytes, static_cast<__gm__ uint8_t *>(nullptr));
    (void)evt;
}

template <typename T, int kGRows_, int kGCols_>
void LaunchTPrefetchL2(T *src, void *stream)
{
    if constexpr (std::is_same_v<T, aclFloat16>) {
        runTPrefetchL2GlobalTensor<half, kGRows_, kGCols_>((half *)(src));
        runTPrefetchL2RawPtr<half, kGRows_, kGCols_>((half *)(src));
    } else {
        runTPrefetchL2GlobalTensor<T, kGRows_, kGCols_>(src);
        runTPrefetchL2RawPtr<T, kGRows_, kGCols_>(src);
    }
}

template void LaunchTPrefetchL2<float, 64, 64>(float *src, void *stream);
template void LaunchTPrefetchL2<int32_t, 64, 64>(int32_t *src, void *stream);
template void LaunchTPrefetchL2<aclFloat16, 16, 256>(aclFloat16 *src, void *stream);
