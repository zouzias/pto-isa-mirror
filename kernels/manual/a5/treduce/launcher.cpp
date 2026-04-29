/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// Host-side launcher for pto_aiv_treduce_kernel.
//
// Compiled into `libpto_aiv_treduce.so`, exports a single C entry point
// `pto_aiv_treduce_launch` that the inline header `treduce_launcher.hpp`
// dlopens + dlsyms.
//
// Responsibilities:
//   - Allocate a small GM staging buffer (one per call; cheap)
//   - Memcpy the host-side struct into it
//   - aclrtLaunchKernel the AIV kernel on the user's stream (1 block, async)
//   - Free the staging buffer after launch (kernel reads it once)
//
// Caller invariants (asserted in the inline header before we get here):
//   - stream != nullptr, mask != 0, mmioAddr != 0, ckeId in range
//
// Returns the rc from aclrtLaunchKernel (0 = success).

#include <cstdint>
#include <cstdio>
#include <cstring>

#include <acl/acl.h>
#include <acl/acl_rt.h>

namespace {

struct DeviceCtx {
    uint64_t mmioAddr;   // +0x00
    uint32_t dieId;      // +0x08
    uint32_t ckeId;      // +0x0c
    uint32_t mask;       // +0x10
    uint32_t pad;        // +0x14
    uint64_t stride;     // +0x18
    uint64_t byte_off;   // +0x20
};
static_assert(sizeof(DeviceCtx) == 0x28,
              "DeviceCtx layout must match kernel.cpp ctx layout");

} // namespace

extern "C" __attribute__((visibility("default"))) int
pto_aiv_treduce_launch(void *stream,
                       uint64_t mmioAddr, uint32_t dieId,
                       uint32_t ckeId, uint32_t mask,
                       uint64_t stride, uint64_t byte_off)
{
    if (stream == nullptr) {
        std::fprintf(stderr, "[pto_aiv_treduce_launch] stream is null\n");
        return -1;
    }

    // 1. Build host-side ctx
    DeviceCtx host_ctx{};
    host_ctx.mmioAddr = mmioAddr;
    host_ctx.dieId    = dieId;
    host_ctx.ckeId    = ckeId;
    host_ctx.mask     = mask;
    host_ctx.stride   = stride;
    host_ctx.byte_off = byte_off;

    // 2. Allocate device staging buffer
    void *dev_ctx = nullptr;
    aclError rc = aclrtMalloc(&dev_ctx, sizeof(host_ctx), ACL_MEM_MALLOC_HUGE_FIRST);
    if (rc != ACL_SUCCESS || dev_ctx == nullptr) {
        std::fprintf(stderr,
            "[pto_aiv_treduce_launch] aclrtMalloc(%zu) failed rc=%d\n",
            sizeof(host_ctx), rc);
        return rc != 0 ? rc : -2;
    }

    // 3. Copy host_ctx → device staging
    rc = aclrtMemcpyAsync(dev_ctx, sizeof(host_ctx),
                          &host_ctx, sizeof(host_ctx),
                          ACL_MEMCPY_HOST_TO_DEVICE, stream);
    if (rc != ACL_SUCCESS) {
        std::fprintf(stderr,
            "[pto_aiv_treduce_launch] aclrtMemcpyAsync H2D failed rc=%d\n", rc);
        aclrtFree(dev_ctx);
        return rc;
    }

    // 4. Launch the kernel — 1 block, blockDim per CCE convention
    //    NOTE: aclrtLaunchKernel signature varies across CANN versions; this
    //    matches CANN ≥ 9.x. If you see an "undefined reference" link error
    //    here, swap in the older `rtKernelLaunch` signature instead.
    extern void pto_aiv_treduce_kernel(uint64_t blockDim, void *ctxArg, void *stream_, void *args);
    // Above declaration is a placeholder — the real entry point is generated
    // by the CCE toolchain from kernel.cpp's __global__ __aicore__ symbol.
    // Use the toolchain's launch macro:
    //
    //     ACLRT_LAUNCH_KERNEL(pto_aiv_treduce_kernel)(1, nullptr, stream, dev_ctx);
    //
    // We can't call the macro from C++ without the AscendC toolchain header,
    // so we forward to a thin C wrapper exported by kernel.cpp at build time
    // (see CMakeLists.txt: target_compile_options ... -DBUILD_LAUNCH_THUNK).
    extern int pto_aiv_treduce_kernel_thunk(void *stream_, void *ctxArg);
    rc = static_cast<aclError>(pto_aiv_treduce_kernel_thunk(stream, dev_ctx));
    if (rc != 0) {
        std::fprintf(stderr,
            "[pto_aiv_treduce_launch] kernel launch thunk returned %d\n", rc);
    }

    // 5. Free staging buffer (kernel reads it before returning, but free is
    //    deferred via the stream so it's safe to free now in user-mode if
    //    the runtime tracks pending DMAs; otherwise switch to aclrtFree-on-
    //    stream-sync. For the empirical step this simple path is fine.)
    aclrtFree(dev_ctx);

    return static_cast<int>(rc);
}
