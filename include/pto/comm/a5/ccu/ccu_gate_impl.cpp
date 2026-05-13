/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// Separate translation unit compiled with kernel_operator.h (AscendC).
// Provides the proven TPipe + DataCopy CKE trigger as a C-linkage function
// callable from PTO code without header conflicts.

#include <cstdint>
#include "kernel_operator.h"

using namespace AscendC;

extern "C" __aicore__ void pto_ccu_cke_trigger(uint64_t ckeSlotVA, uint32_t mask)
{
    TPipe pipe;
    TQue<QuePosition::VECOUT, 1> outQue;
    pipe.InitBuffer(outQue, 1, 32);

    LocalTensor<uint64_t> ubData = outQue.AllocTensor<uint64_t>();
    ubData.SetValue(0, static_cast<uint64_t>(mask & 0xFFFF));
    ubData.SetValue(1, static_cast<uint64_t>(0));
    ubData.SetValue(2, static_cast<uint64_t>(0));
    ubData.SetValue(3, static_cast<uint64_t>(0));
    pipe_barrier(PIPE_ALL);

    GlobalTensor<uint64_t> gmCke;
    gmCke.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t *>(ckeSlotVA), 4);

    DataCopyParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen   = 32;
    copyParams.srcStride  = 0;
    copyParams.dstStride  = 0;
    DataCopy(gmCke, ubData, copyParams);
    pipe_barrier(PIPE_MTE3);

    dcci(reinterpret_cast<__gm__ void *>(ckeSlotVA), ENTIRE_DATA_CACHE);
    __asm__ __volatile__("" ::: "memory");
    pipe_barrier(PIPE_ALL);
    dsb(DSB_DDR);

    outQue.FreeTensor(ubData);
}
