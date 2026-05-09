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
//   mmioAddr — CCU resource base VA on this die (from QueryCcuBaseInfo +0x28
//              or rtGetDevResAddress per-CKE VA)
//   ckeId    — target CKE entry index
//   mask     — 16-bit mask to set (low 16 bits used; rest must be 0)
//   stride   — bytes between adjacent CKE entries (sweep candidate)
//   byte_off — byte position of the mask u16 within the CKE slot (sweep candidate)
//   marker   — host-allocated 64B device buffer (8×u64, cacheline-aligned)
//              for diagnostic readout. May be nullptr (no diag). Layout
//              written by this kernel (8 slots = one host hex-dump line):
//                marker[0] = 0xC0DECAFEDEADBEEF  (entry sentinel — proves
//                            kernel ran past the first instr; H10 detector)
//                marker[1] = 8B value at `target` BEFORE store + before
//                            pipe_barrier (raw observation, no fence)
//                marker[2] = mask the kernel actually received (echo —
//                            verifies host→kernel ABI marshalling correct)
//                marker[3] = 8B value at `target` AFTER store + AFTER
//                            pipe_barrier (the authoritative readback)
//                marker[4] = target address kernel computed
//                            (mmioAddr + ckeId*stride + byte_off) — host
//                            cross-checks against what it intended
//                marker[5] = get_block_idx() — confirms which block ran
//                marker[6] = 8B value at `target` AFTER store but BEFORE
//                            the second pipe_barrier (does barrier affect
//                            readback at all?)
//                marker[7] = 0xFEEDFACECAFEBABE  (tail sentinel — proves
//                            kernel ran ALL the way through, didn't trap
//                            mid-flight)
//              Host triage (after aclrtSynchronizeStream(aivStream)):
//                marker[0] != head sentinel  → kernel never ran (H10)
//                marker[7] != tail sentinel  → kernel trapped mid-store
//                marker[5] != 0              → wrong block ran (shouldn't happen)
//                marker[4] != host_expected  → ckeId/stride/byte_off ABI bug
//                marker[1] == marker[3]      → store had no effect on register
//                marker[2] != desc.mask      → mask got mangled in marshalling
//                else                        → store landed; check marker[3] bits
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
//
// `marker` is declared as `__gm__ uint8_t *` (not `__gm__ uint64_t *`) because
// bisheng forbids the `void* → __gm__ T*` reinterpret_cast in the host-side
// launcher (different address spaces — host doesn't know about __gm__). The
// gemm_ar kernel uses the same convention: launcher takes `uint8_t*`, kernel
// receives `__gm__ uint8_t*`, then re-casts to typed `__gm__ T*` inside the
// kernel body (same address space → OK).
__global__ __aicore__ void pto_aiv_treduce_kernel(
    uint64_t mmioAddr, uint32_t ckeId, uint32_t mask,
    uint64_t stride, uint64_t byte_off,
    __gm__ uint8_t *marker)
{
    if (get_block_idx() != 0) return;

    __gm__ uint64_t *m64 = reinterpret_cast<__gm__ uint64_t *>(marker);

    const uint64_t alignedTarget = mmioAddr + static_cast<uint64_t>(ckeId) * stride;

    if (m64 != nullptr) {
        m64[0] = 0xC0DECAFEDEADBEEFULL;
        m64[1] = static_cast<uint64_t>(
            *reinterpret_cast<__gm__ uint16_t *>(alignedTarget + byte_off));
        m64[2] = static_cast<uint64_t>(mask) & 0xFFFFULL;
        m64[4] = alignedTarget + byte_off;
        m64[5] = static_cast<uint64_t>(get_block_idx());
        pipe_barrier(PIPE_ALL);
    }

    // ── MTE STORE PATH (2026-05-09, per HW team guidance) ────────────────
    //
    // HW team confirmed: AIV must use MTE (DataCopy UB→GM, i.e. PIPE_MTE3)
    // to write CKE VA, NOT scalar store (*ptr = val). Scalar stores go
    // through a different bus path that doesn't trigger the CKE io-map.
    //
    // Strategy:
    //   1. Allocate 32B UB buffer (minimum 1 DataBlock for MTE3)
    //   2. Fill UB with mask pattern (16 × uint16_t = 32B)
    //   3. DataCopy(gmCke, ubBuf, 16) → MTE3 store to CKE VA
    //   4. pipe_barrier(PIPE_MTE3)
    //
    // 32B covers 16 uint16_t elements = bytes [0..31] starting at target.
    // For a single CKE (16-bit), the hardware should pick up the relevant
    // 2 bytes from the MTE3 transaction.

    TPipe pipe;
    TBuf<QuePosition::VECCALC> calcBuf;
    pipe.InitBuffer(calcBuf, 1, 32);
    LocalTensor<uint16_t> ubData = calcBuf.Get<uint16_t>();

    uint16_t maskVal = static_cast<uint16_t>(mask & 0xFFFF);

    // Pre-clear: fill UB with zeros, MTE store to CKE VA
    Duplicate(ubData, static_cast<uint16_t>(0), 16);
    pipe_barrier(PIPE_V);

    GlobalTensor<uint16_t> gmCke;
    gmCke.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t *>(alignedTarget), 16);
    DataCopy(gmCke, ubData, 16);
    pipe_barrier(PIPE_MTE3);

    // Trigger: fill UB with mask, MTE store to CKE VA
    Duplicate(ubData, maskVal, 16);
    pipe_barrier(PIPE_V);

    DataCopy(gmCke, ubData, 16);
    pipe_barrier(PIPE_MTE3);

    // Also try scalar store as fallback (in case MTE write triggers
    // but readback needs scalar path)
    __gm__ uint16_t *target16 = reinterpret_cast<__gm__ uint16_t *>(alignedTarget + byte_off);

    if (m64 != nullptr) {
        m64[6] = static_cast<uint64_t>(*target16);
    }

    pipe_barrier(PIPE_ALL);

    if (m64 != nullptr) {
        m64[3] = static_cast<uint64_t>(*target16);
        m64[7] = 0xFEEDFACECAFEBABEULL;
        pipe_barrier(PIPE_ALL);
    }

    pipe_barrier(PIPE_ALL);
}

// ============================================================================
// Host-side launcher — exported via C ABI for dlopen
// ============================================================================
//
// `dieId` is unused on the AIV side (each AIV resolves its own die), but the
// signature includes it for symmetry with the host trigger path and for any
// future per-die handling.
//
// `marker` is a host-supplied device pointer to a 64B (8×u64) zero-initialized
// buffer. May be nullptr to disable the diagnostic readout. See kernel header
// for the marker[] layout.
//
// Type is `uint8_t *` (not `void *`) because bisheng's `<<<>>>` dispatch
// marshals untyped device pointers via `uint8_t*` → `__gm__ uint8_t*` (the
// cast is implicit in the launch thunk). `void* → __gm__ T*` is rejected.
// Callers passing `void*` (e.g. from `aclrtMalloc`) must `static_cast` it
// before this entry point — handled by the inline `pto::aiv::launch_treduce`
// wrapper in `<pto/aiv/detail/treduce_launcher.hpp>`.
extern "C" __attribute__((visibility("default"))) int
pto_aiv_treduce_launch(void *stream,
                       uint64_t mmioAddr, uint32_t /*dieId*/,
                       uint32_t ckeId, uint32_t mask,
                       uint64_t stride, uint64_t byte_off,
                       uint8_t *marker)
{
    pto_aiv_treduce_kernel<<<1, nullptr, stream>>>(
        mmioAddr, ckeId, mask, stride, byte_off, marker);
    return 0;
}
