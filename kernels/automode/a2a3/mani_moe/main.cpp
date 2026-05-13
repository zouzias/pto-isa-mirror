/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

/**
 * main.cpp - host driver for mani_moe.
 *
 * Current simple contract, matching scripts/gen_data.py:
 *   input_X.bin          [kT, kH]       fp16
 *   input_W_router.bin   [kH, kE]       fp16
 *   input_W1.bin         [kE, kH, kF]   fp16
 *   input_W2.bin         [kE, kF, kH]   fp16
 *
 *   golden_logits.bin    [kT, kE]       fp32
 *   golden_expert_id.bin [kT]           uint32
 *   golden_Z.bin         [kT, kH]       fp32
 *
 * The kernel should expose launchManiMoeFp16(...) below. The host keeps all
 * fp16 buffers as uint8_t* so main.cpp never has to name the device-only
 * `half` type.
 */

#include "acl/acl.h"
#include "test_common.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>

using namespace PtoTestCommon;

extern "C" void launchManiMoeFp16(uint8_t *z_fp32,
                                  uint8_t *logits_fp32,
                                  uint8_t *expert_id_u32,
                                  uint8_t *x_fp16,
                                  uint8_t *w_router_fp16,
                                  uint8_t *w1_fp16,
                                  uint8_t *w2_fp16,
                                  void *stream);

namespace {

constexpr int kT = 256;
constexpr int kH = 64;
constexpr int kF = 64;
constexpr int kE = 16;

constexpr size_t kHalfBytes = 2;
constexpr size_t kFp32Bytes = 4;
constexpr size_t kU32Bytes = 4;

bool CheckAcl(aclError ret, const char *op)
{
    if (ret != ACL_SUCCESS) {
        std::cerr << "[ERROR] " << op << " failed: " << static_cast<int>(ret) << std::endl;
        return false;
    }
    return true;
}

template <typename T>
bool ValidateBuffer(const char *goldenPath, const char *outputPath, size_t numBytes, const char *label, float eps)
{
    std::vector<T> golden(numBytes / sizeof(T));
    std::vector<T> output(numBytes / sizeof(T));
    ReadFile(goldenPath, numBytes, golden.data(), numBytes);
    ReadFile(outputPath, numBytes, output.data(), numBytes);

    bool ok = ResultCmp(golden, output, eps);
    printf("[validate] %-10s : %s\n", label, ok ? "PASS" : "FAIL");
    return ok;
}

}  // namespace

int main()
{
    const size_t xBytes = static_cast<size_t>(kT) * kH * kHalfBytes;
    const size_t wRouterBytes = static_cast<size_t>(kH) * kE * kHalfBytes;
    const size_t w1Bytes = static_cast<size_t>(kE) * kH * kF * kHalfBytes;
    const size_t w2Bytes = static_cast<size_t>(kE) * kF * kH * kHalfBytes;

    const size_t logitsBytes = static_cast<size_t>(kT) * kE * kFp32Bytes;
    const size_t expertIdBytes = static_cast<size_t>(kT) * kU32Bytes;
    const size_t zBytes = static_cast<size_t>(kT) * kH * kFp32Bytes;

    printf("[main] T=%d H=%d F=%d E=%d\n", kT, kH, kF, kE);
    printf("[main] x=%zu w_router=%zu w1=%zu w2=%zu logits=%zu expert_id=%zu Z=%zu bytes\n",
           xBytes, wRouterBytes, w1Bytes, w2Bytes, logitsBytes, expertIdBytes, zBytes);

    if (!CheckAcl(aclInit(nullptr), "aclInit")) std::exit(3);
    if (!CheckAcl(aclrtSetDevice(0), "aclrtSetDevice")) std::exit(3);

    aclrtStream stream = nullptr;
    if (!CheckAcl(aclrtCreateStream(&stream), "aclrtCreateStream")) std::exit(3);

    uint8_t *xHost = nullptr;
    uint8_t *wRouterHost = nullptr;
    uint8_t *w1Host = nullptr;
    uint8_t *w2Host = nullptr;
    uint8_t *logitsHost = nullptr;
    uint8_t *expertIdHost = nullptr;
    uint8_t *zHost = nullptr;

    uint8_t *xDev = nullptr;
    uint8_t *wRouterDev = nullptr;
    uint8_t *w1Dev = nullptr;
    uint8_t *w2Dev = nullptr;
    uint8_t *logitsDev = nullptr;
    uint8_t *expertIdDev = nullptr;
    uint8_t *zDev = nullptr;

    CheckAcl(aclrtMallocHost(reinterpret_cast<void **>(&xHost), xBytes), "aclrtMallocHost(xHost)");
    CheckAcl(aclrtMallocHost(reinterpret_cast<void **>(&wRouterHost), wRouterBytes),
             "aclrtMallocHost(wRouterHost)");
    CheckAcl(aclrtMallocHost(reinterpret_cast<void **>(&w1Host), w1Bytes), "aclrtMallocHost(w1Host)");
    CheckAcl(aclrtMallocHost(reinterpret_cast<void **>(&w2Host), w2Bytes), "aclrtMallocHost(w2Host)");
    CheckAcl(aclrtMallocHost(reinterpret_cast<void **>(&logitsHost), logitsBytes), "aclrtMallocHost(logitsHost)");
    CheckAcl(aclrtMallocHost(reinterpret_cast<void **>(&expertIdHost), expertIdBytes),
             "aclrtMallocHost(expertIdHost)");
    CheckAcl(aclrtMallocHost(reinterpret_cast<void **>(&zHost), zBytes), "aclrtMallocHost(zHost)");

    CheckAcl(aclrtMalloc(reinterpret_cast<void **>(&xDev), xBytes, ACL_MEM_MALLOC_HUGE_FIRST),
             "aclrtMalloc(xDev)");
    CheckAcl(aclrtMalloc(reinterpret_cast<void **>(&wRouterDev), wRouterBytes, ACL_MEM_MALLOC_HUGE_FIRST),
             "aclrtMalloc(wRouterDev)");
    CheckAcl(aclrtMalloc(reinterpret_cast<void **>(&w1Dev), w1Bytes, ACL_MEM_MALLOC_HUGE_FIRST),
             "aclrtMalloc(w1Dev)");
    CheckAcl(aclrtMalloc(reinterpret_cast<void **>(&w2Dev), w2Bytes, ACL_MEM_MALLOC_HUGE_FIRST),
             "aclrtMalloc(w2Dev)");
    CheckAcl(aclrtMalloc(reinterpret_cast<void **>(&logitsDev), logitsBytes, ACL_MEM_MALLOC_HUGE_FIRST),
             "aclrtMalloc(logitsDev)");
    CheckAcl(aclrtMalloc(reinterpret_cast<void **>(&expertIdDev), expertIdBytes, ACL_MEM_MALLOC_HUGE_FIRST),
             "aclrtMalloc(expertIdDev)");
    CheckAcl(aclrtMalloc(reinterpret_cast<void **>(&zDev), zBytes, ACL_MEM_MALLOC_HUGE_FIRST),
             "aclrtMalloc(zDev)");

    ReadFile("../input/input_X.bin", xBytes, xHost, xBytes);
    ReadFile("../input/input_W_router.bin", wRouterBytes, wRouterHost, wRouterBytes);
    ReadFile("../input/input_W1.bin", w1Bytes, w1Host, w1Bytes);
    ReadFile("../input/input_W2.bin", w2Bytes, w2Host, w2Bytes);

    CheckAcl(aclrtMemcpy(xDev, xBytes, xHost, xBytes, ACL_MEMCPY_HOST_TO_DEVICE), "aclrtMemcpy(xDev)");
    CheckAcl(aclrtMemcpy(wRouterDev, wRouterBytes, wRouterHost, wRouterBytes, ACL_MEMCPY_HOST_TO_DEVICE),
             "aclrtMemcpy(wRouterDev)");
    CheckAcl(aclrtMemcpy(w1Dev, w1Bytes, w1Host, w1Bytes, ACL_MEMCPY_HOST_TO_DEVICE), "aclrtMemcpy(w1Dev)");
    CheckAcl(aclrtMemcpy(w2Dev, w2Bytes, w2Host, w2Bytes, ACL_MEMCPY_HOST_TO_DEVICE), "aclrtMemcpy(w2Dev)");

    CheckAcl(aclrtMemset(logitsDev, logitsBytes, 0x5A, logitsBytes), "aclrtMemset(logitsDev)");
    CheckAcl(aclrtMemset(expertIdDev, expertIdBytes, 0x7B, expertIdBytes), "aclrtMemset(expertIdDev)");
    CheckAcl(aclrtMemset(zDev, zBytes, 0x4C, zBytes), "aclrtMemset(zDev)");

    launchManiMoeFp16(zDev, logitsDev, expertIdDev, xDev, wRouterDev, w1Dev, w2Dev, stream);

    if (!CheckAcl(aclrtSynchronizeStream(stream), "aclrtSynchronizeStream")) {
        std::cerr << "[main] stream sync failed.\n";
    }

    CheckAcl(aclrtMemcpy(logitsHost, logitsBytes, logitsDev, logitsBytes, ACL_MEMCPY_DEVICE_TO_HOST),
             "aclrtMemcpy(logitsHost)");
    CheckAcl(aclrtMemcpy(expertIdHost, expertIdBytes, expertIdDev, expertIdBytes, ACL_MEMCPY_DEVICE_TO_HOST),
             "aclrtMemcpy(expertIdHost)");
    CheckAcl(aclrtMemcpy(zHost, zBytes, zDev, zBytes, ACL_MEMCPY_DEVICE_TO_HOST), "aclrtMemcpy(zHost)");

    WriteFile("../output/output_logits.bin", logitsHost, logitsBytes);
    WriteFile("../output/output_expert_id.bin", expertIdHost, expertIdBytes);
    WriteFile("../output/output_Z.bin", zHost, zBytes);

    printf("[main] poison sentinels after launch: logits[0]=0x%02X expert_id[0]=0x%02X Z[0]=0x%02X\n",
           logitsHost[0], expertIdHost[0], zHost[0]);

    aclrtFree(zDev);
    aclrtFree(expertIdDev);
    aclrtFree(logitsDev);
    aclrtFree(w2Dev);
    aclrtFree(w1Dev);
    aclrtFree(wRouterDev);
    aclrtFree(xDev);
    aclrtFreeHost(zHost);
    aclrtFreeHost(expertIdHost);
    aclrtFreeHost(logitsHost);
    aclrtFreeHost(w2Host);
    aclrtFreeHost(w1Host);
    aclrtFreeHost(wRouterHost);
    aclrtFreeHost(xHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    bool logitsOk = ValidateBuffer<float>("../output/golden_logits.bin", "../output/output_logits.bin", logitsBytes,
                                          "logits", 1e-2f);
    bool expertIdOk = ValidateBuffer<uint32_t>("../output/golden_expert_id.bin", "../output/output_expert_id.bin",
                                               expertIdBytes, "expert_id", 0.0f);
    bool zOk = ValidateBuffer<float>("../output/golden_Z.bin", "../output/output_Z.bin", zBytes, "Z", 1e-1f);

    if (logitsOk && expertIdOk && zOk) {
        printf("test success\n");
        return 0;
    }

    printf("test failed\n");
    return 1;
}
