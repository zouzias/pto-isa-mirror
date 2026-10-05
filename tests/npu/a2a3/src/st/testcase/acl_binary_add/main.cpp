/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// Host side of the ACL-binary add ST. Nothing here is linked against the kernel:
// the device object is loaded at runtime and the entry point is resolved by name.

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <string>
#include <vector>

#include "acl/acl.h"

#define CHECK(expr)                                                          \
    do {                                                                     \
        aclError _err = (expr);                                              \
        if (_err != ACL_SUCCESS) {                                           \
            std::printf("[FAIL] %s -> %d\n", #expr, static_cast<int>(_err)); \
            return 1;                                                        \
        }                                                                    \
    } while (0)

namespace {
constexpr uint32_t kCount = 256;
constexpr uint32_t kBlockDim = 1;
constexpr const char* kKernelName = "add_custom";
} // namespace

int main(int argc, char** argv)
{
    const std::string binPath = (argc > 1) ? argv[1] : std::string("./add_custom_device.o");
    const size_t bytes = static_cast<size_t>(kCount) * sizeof(aclFloat16);

    CHECK(aclInit(nullptr));
    CHECK(aclrtSetDevice(0));
    aclrtStream stream = nullptr;
    CHECK(aclrtCreateStream(&stream));

    aclFloat16* xHost = nullptr;
    aclFloat16* yHost = nullptr;
    aclFloat16* zHost = nullptr;
    CHECK(aclrtMallocHost(reinterpret_cast<void**>(&xHost), bytes));
    CHECK(aclrtMallocHost(reinterpret_cast<void**>(&yHost), bytes));
    CHECK(aclrtMallocHost(reinterpret_cast<void**>(&zHost), bytes));

    std::vector<float> golden(kCount);
    for (uint32_t i = 0; i < kCount; ++i) {
        const float a = static_cast<float>(i % 32);
        const float b = 2.0f;
        xHost[i] = aclFloatToFloat16(a);
        yHost[i] = aclFloatToFloat16(b);
        zHost[i] = aclFloatToFloat16(0.0f);
        golden[i] = a + b;
    }

    void* xDevice = nullptr;
    void* yDevice = nullptr;
    void* zDevice = nullptr;
    CHECK(aclrtMalloc(&xDevice, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK(aclrtMalloc(&yDevice, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK(aclrtMalloc(&zDevice, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK(aclrtMemcpy(xDevice, bytes, xHost, bytes, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK(aclrtMemcpy(yDevice, bytes, yHost, bytes, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK(aclrtMemcpy(zDevice, bytes, zHost, bytes, ACL_MEMCPY_HOST_TO_DEVICE));

    aclrtBinHandle binHandle = nullptr;
    aclrtFuncHandle funcHandle = nullptr;
    aclrtArgsHandle argsHandle = nullptr;
    aclrtParamHandle pX = nullptr;
    aclrtParamHandle pY = nullptr;
    aclrtParamHandle pZ = nullptr;

    std::printf("[INFO] loading device binary %s\n", binPath.c_str());
    CHECK(aclrtBinaryLoadFromFile(binPath.c_str(), nullptr, &binHandle));
    std::printf("[INFO] resolving kernel by name: %s\n", kKernelName);
    CHECK(aclrtBinaryGetFunction(binHandle, kKernelName, &funcHandle));

    CHECK(aclrtKernelArgsInit(funcHandle, &argsHandle));
    CHECK(aclrtKernelArgsAppend(argsHandle, &xDevice, sizeof(uintptr_t), &pX));
    CHECK(aclrtKernelArgsAppend(argsHandle, &yDevice, sizeof(uintptr_t), &pY));
    CHECK(aclrtKernelArgsAppend(argsHandle, &zDevice, sizeof(uintptr_t), &pZ));
    CHECK(aclrtKernelArgsFinalize(argsHandle));

    CHECK(aclrtLaunchKernelWithConfig(funcHandle, kBlockDim, stream, nullptr, argsHandle, nullptr));
    CHECK(aclrtSynchronizeStream(stream));
    CHECK(aclrtMemcpy(zHost, bytes, zDevice, bytes, ACL_MEMCPY_DEVICE_TO_HOST));

    uint32_t mismatches = 0;
    for (uint32_t i = 0; i < kCount; ++i) {
        const float got = aclFloat16ToFloat(zHost[i]);
        if (std::fabs(got - golden[i]) > 1e-3f) {
            if (mismatches < 8) {
                std::printf("[DIFF] i=%u got=%f want=%f\n", i, got, golden[i]);
            }
            ++mismatches;
        }
    }
    std::printf("[INFO] first 8 outputs:");
    for (uint32_t i = 0; i < 8; ++i) {
        std::printf(" %.1f", aclFloat16ToFloat(zHost[i]));
    }
    std::printf("\n");

    CHECK(aclrtBinaryUnLoad(binHandle));
    CHECK(aclrtFree(xDevice));
    CHECK(aclrtFree(yDevice));
    CHECK(aclrtFree(zDevice));
    CHECK(aclrtFreeHost(xHost));
    CHECK(aclrtFreeHost(yHost));
    CHECK(aclrtFreeHost(zHost));
    CHECK(aclrtDestroyStream(stream));
    CHECK(aclrtResetDevice(0));
    CHECK(aclFinalize());

    if (mismatches != 0) {
        std::printf("[FAILED] %u/%u elements mismatched\n", mismatches, kCount);
        return 1;
    }
    std::printf(
        "[PASSED] add_custom loaded from %s and launched by name, %u elements correct\n", binPath.c_str(), kCount);
    return 0;
}
