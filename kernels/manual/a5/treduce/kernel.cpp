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
__global__ __aicore__ void pto_aiv_treduce_kernel(
    uint64_t mmioAddr, uint32_t ckeId, uint32_t mask,
    uint64_t stride, uint64_t byte_off,
    __gm__ uint64_t *marker)
{
    // Defensive: ensure only block 0 stores, even if launcher is ever changed
    // to launch >1 blocks. With `<<<1, ...>>>` this is always true.
    if (get_block_idx() != 0) return;

    // Target byte address inside the CCU CKE register file.
    const uint64_t target = mmioAddr + static_cast<uint64_t>(ckeId) * stride + byte_off;

    // ── Diagnostic marker (8 × u64 = 64B, cacheline-aligned) ──────────────
    // Pure GM store/load — independent of AscendC::printf / DebugTunnel.
    // See file header for full layout & host-side triage table.
    if (marker != nullptr) {
        // [0] entry sentinel — first thing kernel does, proves we ran
        marker[0] = 0xC0DECAFEDEADBEEFULL;
        // [1] pre-store readback (no fence yet — raw observation)
        marker[1] = *reinterpret_cast<__gm__ uint64_t *>(target);
        // [2] mask echo — confirms host→kernel ABI marshalling correct
        marker[2] = static_cast<uint64_t>(mask) & 0xFFFFULL;
        // [4] target address self-check — host computes same and compares
        marker[4] = target;
        // [5] which block ran — should be 0
        marker[5] = static_cast<uint64_t>(get_block_idx());
        pipe_barrier(PIPE_ALL);
    }

    // ── HYPOTHESIS H5 TEST (printf disabled) — 2026-04-29 ─────────────────
    // Kept commented to preserve the printf-disabled build mode that we
    // currently ship; marker[] gives the same evidence without dragging in
    // DebugTunnel.
    // AscendC::printf("[AIV/treduce] kernel ran: target=0x%lx mask=0x%x\n",
    //                 (unsigned long)target, (unsigned int)(mask & 0xffff));

    // Raw 16-bit store into device MMIO.
    // The CCU CKE register window is mapped device-side as Device-nGnRE-equivalent
    // (non-cacheable, ordered) on Ascend 950, so a regular store + barrier is sufficient.
    *reinterpret_cast<__gm__ uint16_t *>(target) = static_cast<uint16_t>(mask & 0xffff);

    if (marker != nullptr) {
        // [6] post-store, PRE-barrier readback — does barrier matter at all?
        marker[6] = *reinterpret_cast<__gm__ uint64_t *>(target);
    }

    pipe_barrier(PIPE_ALL);

    if (marker != nullptr) {
        // [3] post-store, POST-barrier readback — authoritative observation
        marker[3] = *reinterpret_cast<__gm__ uint64_t *>(target);
        // [7] tail sentinel — proves kernel ran ALL the way through
        marker[7] = 0xFEEDFACECAFEBABEULL;
        pipe_barrier(PIPE_ALL);
    }

    // Memory barrier: make the store visible to the CCU before the kernel
    // returns, so when the host stream-sync unblocks, the gated CCU kernel
    // has already observed the mask.
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
extern "C" __attribute__((visibility("default"))) int
pto_aiv_treduce_launch(void *stream,
                       uint64_t mmioAddr, uint32_t /*dieId*/,
                       uint32_t ckeId, uint32_t mask,
                       uint64_t stride, uint64_t byte_off,
                       void *marker)
{
    pto_aiv_treduce_kernel<<<1, nullptr, stream>>>(
        mmioAddr, ckeId, mask, stride, byte_off,
        reinterpret_cast<__gm__ uint64_t *>(marker));
    return 0;
}
