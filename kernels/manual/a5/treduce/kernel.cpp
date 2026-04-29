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
// File layout (mirrors `kernels/manual/a5/gemm_ar/comm_kernel.cpp`):
//   - The `__global__ __aicore__` kernel below is the AIV side.
//   - The `extern "C"` launcher at the bottom is the host side; both compile
//     together in the same bisheng translation unit, which is the only way
//     the `KernelName<<<...>>>(args)` triple-bracket dispatch resolves
//     correctly on this CANN/bisheng combo. Earlier attempt with a separate
//     plain-C++ launcher.cpp could not link because the launch-symbol thunk
//     is bisheng-internal.
//
// Args (passed by value, marshalled implicitly by `<<<>>>`):
//   mmioAddr — CCU resource base VA on this die (from QueryCcuBaseInfo +0x28)
//   ckeId    — target CKE entry index
//   mask     — 16-bit mask to set (low 16 bits used; rest must be 0)
//   stride   — bytes between adjacent CKE entries (sweep candidate)
//   byte_off — byte position of the mask u16 within the CKE slot (sweep candidate)
//
// Target address: `mmioAddr + ckeId * stride + byte_off`, written as u16 = mask.
//
// Risk notes (see README):
//   - Wrong mmioAddr → AIV traps → driver kills process → NPU reset required.
//   - Wrong (stride, byte_off) but writable → store goes to a benign register;
//     gated kernel keeps WaitEvent-ing → host stream sync timeout (recoverable).

#include "kernel_operator.h"
using namespace AscendC;

// AIV kernel — runs on AIV (vec arch). 1 block, 1 thread.
__global__ __aicore__ void pto_aiv_treduce_kernel(
    uint64_t mmioAddr, uint32_t ckeId, uint32_t mask,
    uint64_t stride, uint64_t byte_off)
{
    // Defensive: ensure only block 0 stores, even if launcher is ever changed
    // to launch >1 blocks. With `<<<1, ...>>>` this is always true.
    if (get_block_idx() != 0) return;

    // Target byte address inside the CCU CKE register file.
    const uint64_t target = mmioAddr + static_cast<uint64_t>(ckeId) * stride + byte_off;

    // Raw 16-bit store into device MMIO.
    // The CCU CKE register window is mapped device-side as Device-nGnRE-equivalent
    // (non-cacheable, ordered) on Ascend 950, so a regular store + barrier is sufficient.
    *reinterpret_cast<__gm__ uint16_t *>(target) = static_cast<uint16_t>(mask & 0xffff);

    // Memory barrier: make the store visible to the CCU before the kernel
    // returns, so when the host stream-sync unblocks, the gated CCU kernel
    // has already observed the mask. Without this, the store may sit in the
    // AIV write buffer and the gated kernel keeps waiting.
    pipe_barrier(PIPE_ALL);
}

// ============================================================================
// Host-side launcher — exported via C ABI for dlopen
// ============================================================================
//
// `dieId` is unused on the AIV side (each AIV resolves its own die), but the
// signature includes it for symmetry with the host trigger path and for any
// future per-die handling.
extern "C" __attribute__((visibility("default"))) int
pto_aiv_treduce_launch(void *stream,
                       uint64_t mmioAddr, uint32_t /*dieId*/,
                       uint32_t ckeId, uint32_t mask,
                       uint64_t stride, uint64_t byte_off)
{
    pto_aiv_treduce_kernel<<<1, nullptr, stream>>>(
        mmioAddr, ckeId, mask, stride, byte_off);
    return 0;
}
