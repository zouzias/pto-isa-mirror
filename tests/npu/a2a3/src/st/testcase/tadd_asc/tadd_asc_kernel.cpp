/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <pto/pto-inst.hpp>
#include <pto/common/constants.hpp>
#include "acl/acl.h"
#include "kernel_operator.h"

using namespace pto;

// Step-2: AscendC TPipe/TQue (CopyIn/Compute/CopyOut)

template <typename T, int kTRows_, int kTCols_, int vRows, int vCols>
class KernelTAdd {
public:
    __aicore__ inline KernelTAdd() {}

    __aicore__ inline void Init(__gm__ T *src0, __gm__ T *src1, __gm__ T *dst)
    {
        src0Global.SetGlobalBuffer(src0);
        src1Global.SetGlobalBuffer(src1);
        dstGlobal.SetGlobalBuffer(dst);

        totalRows = vRows;
        totalCols = vCols;
        stride = kTCols_;

        int dataSize = totalRows * stride;
        pipe.InitBuffer(inQueue0, 1, dataSize * sizeof(T));
        pipe.InitBuffer(inQueue1, 1, dataSize * sizeof(T));
        pipe.InitBuffer(outQueue, 1, dataSize * sizeof(T));
    }

    __aicore__ inline void Process()
    {
        CopyIn();
        Compute();
        CopyOut();
    }

private:
    __aicore__ inline void CopyIn()
    {
        AscendC::LocalTensor<T> src0Local = inQueue0.AllocTensor<T>();
        AscendC::LocalTensor<T> src1Local = inQueue1.AllocTensor<T>();

        AscendC::DataCopy(src0Local, src0Global, totalRows * stride);
        AscendC::DataCopy(src1Local, src1Global, totalRows * stride);

        inQueue0.EnQue(src0Local);
        inQueue1.EnQue(src1Local);
    }

    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<T> src0Local = inQueue0.DeQue<T>();
        AscendC::LocalTensor<T> src1Local = inQueue1.DeQue<T>();
        AscendC::LocalTensor<T> dstLocal = outQueue.AllocTensor<T>();

        AscendC::Add(dstLocal, src0Local, src1Local, totalRows * stride);

        outQueue.EnQue(dstLocal);
        inQueue0.FreeTensor(src0Local);
        inQueue1.FreeTensor(src1Local);
    }

    __aicore__ inline void CopyOut()
    {
        AscendC::LocalTensor<T> dstLocal = outQueue.DeQue<T>();
        AscendC::DataCopy(dstGlobal, dstLocal, totalRows * stride);
        outQueue.FreeTensor(dstLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueue0;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueue1;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueue;
    AscendC::GlobalTensor<T> src0Global;
    AscendC::GlobalTensor<T> src1Global;
    AscendC::GlobalTensor<T> dstGlobal;

    int totalRows = 0;
    int totalCols = 0;
    int stride = 0;
};

// Kernel entry

template <typename T, int kTRows_, int kTCols_, int vRows, int vCols>
__global__ AICORE void runTAdd(__gm__ T __out__ *out, __gm__ T __in__ *src0, __gm__ T __in__ *src1)
{
    KernelTAdd<T, kTRows_, kTCols_, vRows, vCols> op;
    op.Init(src0, src1, out);
    op.Process();
}

template <typename T, int kTRows_, int kTCols_, int vRows, int vCols>
void LaunchTAdd(T *out, T *src0, T *src1, void *stream)
{
    if constexpr (std::is_same_v<T, aclFloat16>)
        runTAdd<half, kTRows_, kTCols_, vRows, vCols>
            <<<1, nullptr, stream>>>((half *)(out), (half *)(src0), (half *)(src1));
    else
        runTAdd<T, kTRows_, kTCols_, vRows, vCols><<<1, nullptr, stream>>>(out, src0, src1);
}

template void LaunchTAdd<float, 64, 64, 64, 64>(float *out, float *src0, float *src1, void *stream);
template void LaunchTAdd<int32_t, 64, 64, 64, 64>(int32_t *out, int32_t *src0, int32_t *src1, void *stream);
template void LaunchTAdd<int16_t, 64, 64, 64, 64>(int16_t *out, int16_t *src0, int16_t *src1, void *stream);
template void LaunchTAdd<aclFloat16, 16, 256, 16, 256>(aclFloat16 *out, aclFloat16 *src0, aclFloat16 *src1,
                                                       void *stream);
