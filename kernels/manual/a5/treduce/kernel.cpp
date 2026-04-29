/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// pto_aiv_treduce_kernel — AIV-side trigger for the PTO gated-reduce CKE.
//
// Step 3 of the pto-gated-reduce-kernel plan: replace host-side RaCustomChannel
// (SET_CKE) with an AIV raw store directly into the CCU CKE register file.
// Runs on Vec arch (`dav-c310-vec`); 1 block, 1 thread.
//
// Args (passed by the host launcher in this exact order):
//   mmioAddr — CCU resource base VA on this die (from QueryCcuBaseInfo +0x28)
//   dieId    — informational; AIV resolves its own die so we don't actually need it
//              for the store, but keep it in the signature for symmetry with host
//              trigger and for future per-die handling
//   ckeId    — target CKE entry index
//   mask     — 16-bit mask to set (low 16 bits used; rest must be 0)
//   stride   — bytes between adjacent CKE entries (initial empirical guess: 0x40)
//   byte_off — byte position of the mask u16 within the CKE slot
//              (initial empirical guess: 6, matching host PTO_CKE_LAYOUT=B)
//
// Target address: `mmioAddr + ckeId * stride + byte_off`, written as u16 = mask.
//
// Exit semantics:
//   - On success, returns immediately. The gated CCU kernel (parked at WaitEvent)
//     observes the mask within at most a CCU sync-loop latency.
//   - On a bad mmioAddr, AIV will trap with a device exception. The driver kills
//     the process; recovery requires NPU reset (see Step 2.4 cleanup notes).

#include "kernel_operator.h"
using namespace AscendC;

extern "C" __global__ __aicore__ void pto_aiv_treduce_kernel(
    GM_ADDR ctx)
{
    // ctx is a small GM staging buffer the host launcher fills before launch.
    // Layout (must match host launcher):
    //   +0x00 (8B) mmioAddr
    //   +0x08 (4B) dieId
    //   +0x0c (4B) ckeId
    //   +0x10 (4B) mask
    //   +0x14 (4B) padding
    //   +0x18 (8B) stride
    //   +0x20 (8B) byte_off
    __gm__ uint64_t *ctx64 = reinterpret_cast<__gm__ uint64_t *>(ctx);
    __gm__ uint32_t *ctx32 = reinterpret_cast<__gm__ uint32_t *>(ctx);

    uint64_t mmioAddr = ctx64[0];
    uint32_t ckeId    = ctx32[3];   // [+0x0c]
    uint32_t mask     = ctx32[4];   // [+0x10]
    uint64_t stride   = ctx64[3];   // [+0x18]
    uint64_t byteOff  = ctx64[4];   // [+0x20]

    // Compute target byte address inside the CKE register file.
    uint64_t target = mmioAddr + (uint64_t)ckeId * stride + byteOff;

    // Raw 16-bit store into device MMIO.
    // We don't know the mapping attribute of the resource VA, but on Ascend 950
    // the CCU register window is mapped device-side as `Device-nGnRE`-equivalent
    // (non-cacheable, ordered). A regular store + barrier should be enough.
    *reinterpret_cast<__gm__ uint16_t *>(target) = static_cast<uint16_t>(mask & 0xffff);

    // Memory barrier: ensure the store is visible to the CCU before the kernel
    // returns. Without this, the store may sit in the AIV write buffer when the
    // host stream sync returns, and the gated CCU kernel keeps waiting.
    pipe_barrier(PIPE_ALL);
}
