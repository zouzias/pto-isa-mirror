/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_AIV_DETAIL_TREDUCE_LAUNCHER_HPP
#define PTO_AIV_DETAIL_TREDUCE_LAUNCHER_HPP

// Inline host-side launcher for the AIV trigger kernel.
//
// IMPORTANT: this used to dlopen `libpto_aiv_treduce.so` at runtime. It no
// longer does. Bisheng's AscendC runtime registers `__global__ __aicore__`
// kernels by SCANNING already-loaded shared libraries for fat-object metadata
// at `aclInit()` time. .so files dlopen'd AFTER aclInit miss that scan, so
// the kernel never gets registered, and `KernelName<<<...>>>(args)` queues a
// no-op (returns 0, but nothing actually runs on AIV). This was empirically
// confirmed in Step 3 of the pto-gated-reduce-kernel plan: with dlopen, the
// kernel-side `AscendC::printf` never fired, while host saw "AIV released".
//
// Fix: link-time link `libpto_aiv_treduce.so` into the host executable so
// ld-linux loads it before main() (and therefore before aclInit). The
// reference pattern is in pto-isa's `tests/npu/a2a3/comm/st/testcase/CMakeLists.txt`
// (`pto_comm_st()` function), specifically:
//   - kernel .so built with `--cce-fatobj-link`           (this side)
//   - host executable: `target_link_libraries(... ${NAME}_kernel)` (caller side)
//
// Build expectation:
//   - libpto_aiv_treduce.so must be on the host executable's link line, e.g.
//       g++ ... -L/path/to/dist -lpto_aiv_treduce -Wl,-rpath,/path/to/dist ...
//     (see hccl/examples/02_collectives/04_reduce_scatter/Makefile, which honors
//     `PTO_AIV_TREDUCE_LIB_DIR` env var)
//
// Stride/byte-off come from env (PTO_AIV_TRIGGER_STRIDE, PTO_AIV_TRIGGER_BYTE_OFF)
// with defaults matching host PTO_CKE_LAYOUT=B (stride=0x40, byte_off=6).

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "pto/aiv/treduce.hpp"
#include "pto/host/pto_gate_descriptor.hpp"

// C ABI symbol exported from libpto_aiv_treduce.so (kernels/manual/a5/treduce/kernel.cpp).
// MUST be link-time linked, NOT dlopen'd — see comment block above.
//
// `marker` is an optional 32B device buffer (4×u64). If non-null, the kernel
// stores diagnostic readouts there (see treduce.hpp for layout). Pass nullptr
// to disable.
extern "C" int pto_aiv_treduce_launch(void *stream,
                                       uint64_t mmioAddr, uint32_t dieId,
                                       uint32_t ckeId, uint32_t mask,
                                       uint64_t stride, uint64_t byte_off,
                                       void *marker);

namespace pto {
namespace aiv {
namespace detail {

// Resolve stride / byte-off from env, with defaults matching host PTO_CKE_LAYOUT=B.
//
// Empirical sweep candidates (Step 3.5 — see kernels/manual/a5/treduce/README.md):
//   STRIDE  BYTE_OFF  rationale
//   ────────────────────────────────────────────────────────────
//   0x04    0         32-bit register array (most likely; CKE entries u32)
//   0x40    6         host layout B (default; matches host_trigger_cke_impl.hpp)
//   0x40    0         host layout A position
//   0x10    0         32-byte slot
//   0x08    0         8-byte slot
//   0x02    0         16-bit packed
//
// Override via `PTO_AIV_TRIGGER_STRIDE=0x40` etc.
inline uint64_t ResolveStride() {
    const char *env = std::getenv("PTO_AIV_TRIGGER_STRIDE");
    if (env == nullptr || env[0] == '\0') return 0x40ULL;
    return std::strtoull(env, nullptr, 0);  // base=0 → auto-detect 0x prefix
}
inline uint64_t ResolveByteOff() {
    const char *env = std::getenv("PTO_AIV_TRIGGER_BYTE_OFF");
    if (env == nullptr || env[0] == '\0') return 6ULL;
    return std::strtoull(env, nullptr, 0);
}

} // namespace detail

inline int32_t launch_treduce(void *stream, const host::PtoGateDescriptor &desc,
                              void *marker) {
    if (stream == nullptr || desc.mask == 0 || desc.ckeId > 4096) {
        std::fprintf(stderr,
            "[PTO_AIV_TREDUCE] BadArg stream=%p mask=0x%x ckeId=%u\n",
            stream, desc.mask, desc.ckeId);
        return kLaunchTReduceBadArg;
    }
    if (desc.mmioAddr == 0) {
        std::fprintf(stderr,
            "[PTO_AIV_TREDUCE] mmioAddr is 0 — caller must call "
            "pto::host::QueryCcuBaseInfo and fill desc.mmioAddr first.\n");
        return kLaunchTReduceMmioAddrZero;
    }

    const uint64_t stride   = detail::ResolveStride();
    const uint64_t byte_off = detail::ResolveByteOff();

    std::fprintf(stderr,
        "[PTO_AIV_TREDUCE] launching: mmioAddr=0x%lx dieId=%u ckeId=%u "
        "mask=0x%x stride=0x%lx byte_off=%lu (target=0x%lx) marker=%p\n",
        static_cast<unsigned long>(desc.mmioAddr),
        desc.dieId, desc.ckeId, desc.mask,
        static_cast<unsigned long>(stride),
        static_cast<unsigned long>(byte_off),
        static_cast<unsigned long>(desc.mmioAddr + desc.ckeId * stride + byte_off),
        marker);

    int rc = pto_aiv_treduce_launch(stream, desc.mmioAddr, desc.dieId, desc.ckeId,
                                     desc.mask, stride, byte_off, marker);
    if (rc != 0) {
        std::fprintf(stderr,
            "[PTO_AIV_TREDUCE] pto_aiv_treduce_launch returned %d\n", rc);
        return kLaunchTReduceLaunchFailed;
    }
    return kLaunchTReduceOk;
}

} // namespace aiv
} // namespace pto

#endif // PTO_AIV_DETAIL_TREDUCE_LAUNCHER_HPP
