/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMM_ASYNC_CCU_CCU_TRIGGER_INTRIN_HPP
#define PTO_COMM_ASYNC_CCU_CCU_TRIGGER_INTRIN_HPP

// AIV-side CCU gate trigger intrinsic.
//
// Writes a 16-bit CKE mask into a CCU CKE register via MTE3 DataCopy,
// then flushes with dcci + dsb so the write reaches hardware. This releases
// the CCU stream parked on WaitEvent(gateEvent_).
//
// Usage (inside an AIV kernel):
//   pto::comm::ccu::__ccu_trigger_gate(ckeSlotVA, mask);
//
// Analogous to sdma's __sdma_put_async / urma's __urma_put_async.

#include <cstdint>
#include "kernel_operator.h"

namespace pto {
namespace comm {
namespace ccu {

using namespace AscendC;

// Trigger a CCU gate by writing `mask` into the CKE register at `ckeSlotVA`.
//
// Hardware contract (confirmed 2026-05-12):
//   - ckeSlotVA is a per-CKE 8-byte slot VA from rtGetDevResAddress(dieId, ckeId)
//   - Low 2 bytes (byte 0-1) carry the 16-bit CKE mask
//   - MTE3 DataCopy minimum transfer is 32 bytes; extra bytes are zero-padded
//   - dcci + dsb required to push the write through cache to hardware
//   - CKE registers are auto-consume/self-clearing
__aicore__ inline void __ccu_trigger_gate(__gm__ uint64_t *ckeSlotVA, uint32_t mask)
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
    gmCke.SetGlobalBuffer(ckeSlotVA, 4);

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

} // namespace ccu
} // namespace comm
} // namespace pto

#endif // PTO_COMM_ASYNC_CCU_CCU_TRIGGER_INTRIN_HPP
