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

template <int NUM_TOKENS_, int HIDDEN_ROWS_, int FFN_ROWS_, int NUM_EXPERTS_>
int MoE()
{
    size_t xBytes = static_cast<size_t>(NUM_TOKENS_) * HIDDEN_ROWS_ * kHalfBytes;
    size_t wRouterBytes = static_cast<size_t>(HIDDEN_ROWS_) * NUM_EXPERTS_ * kHalfBytes;
    size_t wBytes = static_cast<size_t>(NUM_EXPERTS_) * HIDDEN_ROWS_ * FFN_ROWS_ * kHalfBytes;
    size_t logitsBytes = static_cast<size_t>(NUM_TOKENS_) * NUM_EXPERTS_ * kFp32Bytes;
    size_t expertIdBytes = static_cast<size_t>(NUM_TOKENS_) * kU32Bytes;
    size_t outBytes = static_cast<size_t>(NUM_TOKENS_) * HIDDEN_ROWS_ * kFp32Bytes;

    constexpr uint8_t kPoisonLogits = 0x5A;
    constexpr uint8_t kPoisonExpertId = 0x7B;
    constexpr uint8_t kPoisonOut = 0x6C;

    printf("[main] T=%d  H=%d  F=%d  E=%d\n"
           "       xBytes=%zu  wRouterBytes=%zu  wBytes=%zu  logitsBytes=%zu  expertIdBytes=%zu  outBytes=%zu\n",
           NUM_TOKENS_, HIDDEN_ROWS_, FFN_ROWS_, NUM_EXPERTS_, xBytes, wRouterBytes, wBytes, logitsBytes,
           expertIdBytes, outBytes);

    if (!CheckAcl(aclInit(nullptr), "aclInit")) {
        return 1;
    }
    if (!CheckAcl(aclrtSetDevice(0), "aclrtSetDevice")) {
        aclFinalize();
        return 1;
    }

    aclrtStream stream = nullptr;
    if (!CheckAcl(aclrtCreateStream(&stream), "aclrtCreateStream")) {
        aclrtResetDevice(0);
        aclFinalize();
        return 1;
    }

    uint8_t *xHost = nullptr, *wRouterHost = nullptr, *w1Host = nullptr, *w2Host = nullptr;
    uint8_t *outHost = nullptr, *logitsHost = nullptr, *expertIdHost = nullptr;
    uint8_t *xDev = nullptr, *wRouterDev = nullptr, *w1Dev = nullptr, *w2Dev = nullptr;
    uint8_t *outDev = nullptr, *logitsDev = nullptr, *expertIdDev = nullptr;

    CheckAcl(aclrtMallocHost((void **)(&xHost), xBytes), "aclrtMallocHost(xHost)");
    CheckAcl(aclrtMallocHost((void **)(&wRouterHost), wRouterBytes), "aclrtMallocHost(wRouterHost)");
    CheckAcl(aclrtMallocHost((void **)(&w1Host), wBytes), "aclrtMallocHost(w1Host)");
    CheckAcl(aclrtMallocHost((void **)(&w2Host), wBytes), "aclrtMallocHost(w2Host)");
    CheckAcl(aclrtMallocHost((void **)(&outHost), outBytes), "aclrtMallocHost(outHost)");
    CheckAcl(aclrtMallocHost((void **)(&logitsHost), logitsBytes), "aclrtMallocHost(logitsHost)");
    CheckAcl(aclrtMallocHost((void **)(&expertIdHost), expertIdBytes), "aclrtMallocHost(expertIdHost)");

    CheckAcl(aclrtMalloc((void **)(&xDev), xBytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc(xDev)");
    CheckAcl(aclrtMalloc((void **)(&wRouterDev), wRouterBytes, ACL_MEM_MALLOC_HUGE_FIRST),
             "aclrtMalloc(wRouterDev)");
    CheckAcl(aclrtMalloc((void **)(&w1Dev), wBytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc(w1Dev)");
    CheckAcl(aclrtMalloc((void **)(&w2Dev), wBytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc(w2Dev)");
    CheckAcl(aclrtMalloc((void **)(&outDev), outBytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc(outDev)");
    CheckAcl(aclrtMalloc((void **)(&logitsDev), logitsBytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc(logitsDev)");
    CheckAcl(aclrtMalloc((void **)(&expertIdDev), expertIdBytes, ACL_MEM_MALLOC_HUGE_FIRST),
             "aclrtMalloc(expertIdDev)");

    ReadFile("../input/input_X.bin", xBytes, xHost, xBytes);
    ReadFile("../input/input_W_router.bin", wRouterBytes, wRouterHost, wRouterBytes);
    ReadFile("../input/input_W1.bin", wBytes, w1Host, wBytes);
    ReadFile("../input/input_W2.bin", wBytes, w2Host, wBytes);

    CheckAcl(aclrtMemcpy(xDev, xBytes, xHost, xBytes, ACL_MEMCPY_HOST_TO_DEVICE), "aclrtMemcpy(xDev)");
    CheckAcl(aclrtMemcpy(wRouterDev, wRouterBytes, wRouterHost, wRouterBytes, ACL_MEMCPY_HOST_TO_DEVICE),
             "aclrtMemcpy(wRouterDev)");
    CheckAcl(aclrtMemcpy(w1Dev, wBytes, w1Host, wBytes, ACL_MEMCPY_HOST_TO_DEVICE), "aclrtMemcpy(w1Dev)");
    CheckAcl(aclrtMemcpy(w2Dev, wBytes, w2Host, wBytes, ACL_MEMCPY_HOST_TO_DEVICE), "aclrtMemcpy(w2Dev)");

    CheckAcl(aclrtMemset(logitsDev, logitsBytes, kPoisonLogits, logitsBytes), "aclrtMemset(logitsDev)");
    CheckAcl(aclrtMemset(expertIdDev, expertIdBytes, kPoisonExpertId, expertIdBytes),
             "aclrtMemset(expertIdDev)");
    CheckAcl(aclrtMemset(outDev, outBytes, kPoisonOut, outBytes), "aclrtMemset(outDev)");

    launchManiMoeFp16(outDev, logitsDev, expertIdDev, xDev, wRouterDev, w1Dev, w2Dev, stream);

    if (!CheckAcl(aclrtSynchronizeStream(stream), "aclrtSynchronizeStream")) {
        std::cerr << "[main] stream sync failed.\n";
    }

    CheckAcl(aclrtMemcpy(logitsHost, logitsBytes, logitsDev, logitsBytes, ACL_MEMCPY_DEVICE_TO_HOST),
             "aclrtMemcpy(logitsHost)");
    CheckAcl(aclrtMemcpy(expertIdHost, expertIdBytes, expertIdDev, expertIdBytes, ACL_MEMCPY_DEVICE_TO_HOST),
             "aclrtMemcpy(expertIdHost)");
    CheckAcl(aclrtMemcpy(outHost, outBytes, outDev, outBytes, ACL_MEMCPY_DEVICE_TO_HOST), "aclrtMemcpy(outHost)");

    WriteFile("../output/output_logits.bin", logitsHost, logitsBytes);
    WriteFile("../output/output_expert_id.bin", expertIdHost, expertIdBytes);
    WriteFile("../output/output_Z.bin", outHost, outBytes);

    printf("[main] poison sentinels after launch: logits[0]=0x%02X expert_id[0]=0x%02X Z[0]=0x%02X\n",
           logitsHost[0], expertIdHost[0], outHost[0]);

    aclrtFree(outDev);
    aclrtFree(expertIdDev);
    aclrtFree(logitsDev);
    aclrtFree(w2Dev);
    aclrtFree(w1Dev);
    aclrtFree(wRouterDev);
    aclrtFree(xDev);
    aclrtFreeHost(outHost);
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
    bool outOk = ValidateBuffer<float>("../output/golden_Z.bin", "../output/output_Z.bin", outBytes, "Z", 1e-1f);

    if (logitsOk && expertIdOk && outOk) {
        printf("test success\n");
        return 0;
    }

    printf("test failed\n");
    return 1;
}

int main()
{
    return MoE<kT, kH, kF, kE>();
}
