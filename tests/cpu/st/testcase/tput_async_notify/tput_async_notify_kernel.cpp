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

constexpr int32_t SIGNAL_VALUE = 7;

template <typename T, int kGRows_, int kGCols_, comm::NotifyOp Op>
AICORE void runTPutAsyncNotify(__gm__ T __out__* out, __gm__ T __in__* input, __gm__ int32_t __out__* signal)
{
    using DynShapeDim5 = Shape<1, 1, 1, kGRows_, kGCols_>;
    using DynStridDim5 = Stride<1, 1, 1, kGCols_, 1>;
    using GlobalData = GlobalTensor<T, DynShapeDim5, DynStridDim5>;
    using SignalData = GlobalTensor<int32_t, Shape<1, 1, 1, 1, 1>, Stride<1, 1, 1, 1, 1>>;

    pto::comm::AsyncSession session;

    GlobalData inputGlobal(input);
    GlobalData outputGlobal(out);
    SignalData signalGlobal(signal);

    comm::TPUT_ASYNC_NOTIFY(outputGlobal, inputGlobal, signalGlobal, SIGNAL_VALUE, Op, session, 0u);
    out = outputGlobal.data();
}

template <typename T, int kGRows_, int kGCols_, comm::NotifyOp Op>
void LaunchTPutAsyncNotify(T* out, T* src, int32_t* signal, void* stream)
{
    if constexpr (std::is_same_v<T, aclFloat16>) {
        runTPutAsyncNotify<half, kGRows_, kGCols_, Op>((half*)(out), (half*)(src), signal);
    } else {
        runTPutAsyncNotify<T, kGRows_, kGCols_, Op>(out, src, signal);
    }
}

template void LaunchTPutAsyncNotify<float, 64, 64, comm::NotifyOp::Set>(
    float* out, float* src, int32_t* signal, void* stream);
template void LaunchTPutAsyncNotify<int32_t, 64, 64, comm::NotifyOp::Set>(
    int32_t* out, int32_t* src, int32_t* signal, void* stream);
template void LaunchTPutAsyncNotify<int16_t, 64, 64, comm::NotifyOp::Set>(
    int16_t* out, int16_t* src, int32_t* signal, void* stream);
template void LaunchTPutAsyncNotify<aclFloat16, 16, 256, comm::NotifyOp::Set>(
    aclFloat16* out, aclFloat16* src, int32_t* signal, void* stream);

template void LaunchTPutAsyncNotify<float, 64, 64, comm::NotifyOp::AtomicAdd>(
    float* out, float* src, int32_t* signal, void* stream);
template void LaunchTPutAsyncNotify<int32_t, 64, 64, comm::NotifyOp::AtomicAdd>(
    int32_t* out, int32_t* src, int32_t* signal, void* stream);
template void LaunchTPutAsyncNotify<int16_t, 64, 64, comm::NotifyOp::AtomicAdd>(
    int16_t* out, int16_t* src, int32_t* signal, void* stream);
template void LaunchTPutAsyncNotify<aclFloat16, 16, 256, comm::NotifyOp::AtomicAdd>(
    aclFloat16* out, aclFloat16* src, int32_t* signal, void* stream);
