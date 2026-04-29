/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// pto_aiv_treduce_launch — host-side stub launcher (Phase 1 of Step 3).
//
// THIS IS A STUB. It does NOT actually launch the AIV kernel. It dumps the
// arguments and returns a sentinel error code so the caller (main.cc) falls
// back to the host trigger path.
//
// Why a stub: launching an AscendC `__global__ __aicore__` kernel from plain
// C++ requires the CCE toolchain's `ACLRT_LAUNCH_KERNEL` / `<<<>>>` syntax,
// whose ABI varies across CANN versions and isn't portable across plain g++
// and bisheng. The stub keeps the host link chain (main.cc → dlopen → call
// → target-address compute) testable while we figure out the right launch
// syntax for this CANN/driver combo.
//
// Once the kernel-side launch is wired up, replace this implementation with
// the real launch sequence (alloc GM ctx → memcpy → ACLRT_LAUNCH_KERNEL).

#include <cstdint>
#include <cstdio>

extern "C" __attribute__((visibility("default"))) int
pto_aiv_treduce_launch(void *stream,
                       uint64_t mmioAddr, uint32_t dieId,
                       uint32_t ckeId, uint32_t mask,
                       uint64_t stride, uint64_t byte_off)
{
    (void)stream;
    (void)dieId;

    // Compute the target address that a real AIV store would hit.
    const uint64_t target = mmioAddr + (uint64_t)ckeId * stride + byte_off;

    std::fprintf(stderr,
        "[STUB pto_aiv_treduce_launch] would AIV-store mask=0x%x to target=0x%lx\n"
        "    (mmioAddr=0x%lx + ckeId=%u * stride=0x%lx + byte_off=%lu)\n"
        "    NOT actually launching AIV kernel — caller will fall back to host trigger\n",
        mask, (unsigned long)target,
        (unsigned long)mmioAddr, ckeId,
        (unsigned long)stride, (unsigned long)byte_off);

    // Return non-zero so the caller knows we didn't actually trigger anything.
    // -99 is arbitrary; pto::aiv::launch_treduce maps it to kLaunchTReduceLaunchFailed.
    return -99;
}
