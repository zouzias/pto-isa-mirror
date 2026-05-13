/**
 * main.cpp - host driver for moe_router_top1.
 *
 * SKELETON. Final file shape. The kernel TU exposes two stage launchers
 * (cube GEMM, vec TROWARGMAX); this driver fires both on one stream.
 *
 * I/O contract (all little-endian, contiguous, no header):
 *   ../output/t.txt                       (single int; T)
 *   ../input/input_X.bin                  (T * H float16)
 *   ../input/input_W_router.bin           (H * E float16)
 *   ../output/golden_logits.bin           (T * E float32; debug)
 *   ../output/golden_expert_id.bin        (T     uint32 ; primary)
 *   ../output/output_logits.bin           (T * E float32; this driver)
 *   ../output/output_expert_id.bin        (T     uint32 ; this driver)
 */

#include "test_common.h"
#include "acl/acl.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>

using namespace std;
using namespace PtoTestCommon;

// Static configuration. Must match kernel.cpp constants and scripts/gen_data.py.
constexpr int kH = 64;
constexpr int kE = 16;

// Non-template host-boundary wrapper exposed by the kernel TU. Hides `half`
// (compile_error_logbook.md §E13) by passing FP16 buffers as uint8_t*.
extern "C" void launchMoeRouterTop1Fp16(uint8_t *logits_fp32,
                                        uint8_t *expert_id_u32,
                                        uint8_t *X_fp16,
                                        uint8_t *W_fp16,
                                        void    *stream);

static bool CheckAcl(aclError ret, const char *op)
{
    if (ret != ACL_SUCCESS) {
        std::cerr << "[ERROR] " << op << " failed: " << static_cast<int>(ret) << std::endl;
        return false;
    }
    return true;
}

static int ReadIntFile(const char *path)
{
    std::ifstream f(path);
    if (!f.is_open()) {
        printf("[main] FATAL: cannot open %s\n", path);
        std::exit(2);
    }
    int n = 0;
    f >> n;
    if (n <= 0) {
        printf("[main] FATAL: invalid value %d in %s\n", n, path);
        std::exit(2);
    }
    return n;
}

int main()
{
    constexpr size_t halfBytes = 2;
    constexpr size_t fp32Bytes = 4;
    constexpr size_t u32Bytes  = 4;

    constexpr uint8_t kPoisonLogits   = 0x5A;
    constexpr uint8_t kPoisonExpertId = 0x7B;

    const int T = ReadIntFile("../output/t.txt");

    size_t xBytes        = static_cast<size_t>(T) * kH * halfBytes;
    size_t wBytes        = static_cast<size_t>(kH) * kE * halfBytes;
    size_t logitsBytes   = static_cast<size_t>(T) * kE * fp32Bytes;
    size_t expertIdBytes = static_cast<size_t>(T)      * u32Bytes;

    printf("[main] T=%d  H=%d  E=%d\n"
           "       xBytes=%zu  wBytes=%zu  logitsBytes=%zu  expertIdBytes=%zu\n",
           T, kH, kE, xBytes, wBytes, logitsBytes, expertIdBytes);

    if (!CheckAcl(aclInit(nullptr), "aclInit")) std::exit(3);
    if (!CheckAcl(aclrtSetDevice(0), "aclrtSetDevice")) std::exit(3);
    aclrtStream stream;
    if (!CheckAcl(aclrtCreateStream(&stream), "aclrtCreateStream")) std::exit(3);

    uint8_t *xHost = nullptr, *wHost = nullptr;
    uint8_t *logitsHost = nullptr, *expertIdHost = nullptr;
    uint8_t *xDev = nullptr, *wDev = nullptr;
    uint8_t *logitsDev = nullptr, *expertIdDev = nullptr;

    CheckAcl(aclrtMallocHost((void **)(&xHost),        xBytes),        "aclrtMallocHost(xHost)");
    CheckAcl(aclrtMallocHost((void **)(&wHost),        wBytes),        "aclrtMallocHost(wHost)");
    CheckAcl(aclrtMallocHost((void **)(&logitsHost),   logitsBytes),   "aclrtMallocHost(logitsHost)");
    CheckAcl(aclrtMallocHost((void **)(&expertIdHost), expertIdBytes), "aclrtMallocHost(expertIdHost)");

    CheckAcl(aclrtMalloc((void **)&xDev,        xBytes,        ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc(xDev)");
    CheckAcl(aclrtMalloc((void **)&wDev,        wBytes,        ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc(wDev)");
    CheckAcl(aclrtMalloc((void **)&logitsDev,   logitsBytes,   ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc(logitsDev)");
    CheckAcl(aclrtMalloc((void **)&expertIdDev, expertIdBytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc(expertIdDev)");

    ReadFile("../input/input_X.bin",        xBytes, xHost, xBytes);
    ReadFile("../input/input_W_router.bin", wBytes, wHost, wBytes);

    CheckAcl(aclrtMemcpy(xDev, xBytes, xHost, xBytes, ACL_MEMCPY_HOST_TO_DEVICE), "aclrtMemcpy(xDev)");
    CheckAcl(aclrtMemcpy(wDev, wBytes, wHost, wBytes, ACL_MEMCPY_HOST_TO_DEVICE), "aclrtMemcpy(wDev)");

    CheckAcl(aclrtMemset(logitsDev,   logitsBytes,   kPoisonLogits,   logitsBytes),
             "aclrtMemset(logitsDev=0x5A)");
    CheckAcl(aclrtMemset(expertIdDev, expertIdBytes, kPoisonExpertId, expertIdBytes),
             "aclrtMemset(expertIdDev=0x7B)");

    launchMoeRouterTop1Fp16(logitsDev, expertIdDev, xDev, wDev, stream);

    if (!CheckAcl(aclrtSynchronizeStream(stream), "aclrtSynchronizeStream")) {
        std::cerr << "[main] stream sync failed.\n";
    }

    CheckAcl(aclrtMemcpy(logitsHost,   logitsBytes,   logitsDev,   logitsBytes,   ACL_MEMCPY_DEVICE_TO_HOST),
             "aclrtMemcpy(logitsHost)");
    CheckAcl(aclrtMemcpy(expertIdHost, expertIdBytes, expertIdDev, expertIdBytes, ACL_MEMCPY_DEVICE_TO_HOST),
             "aclrtMemcpy(expertIdHost)");

    WriteFile("../output/output_logits.bin",    logitsHost,   logitsBytes);
    WriteFile("../output/output_expert_id.bin", expertIdHost, expertIdBytes);

    printf("[main] after launch: logitsHost[0]=0x%02X (poison=0x%02X)  expertIdHost[0]=0x%02X (poison=0x%02X)\n",
           logitsHost[0], kPoisonLogits, expertIdHost[0], kPoisonExpertId);

    aclrtFree(expertIdDev);
    aclrtFree(logitsDev);
    aclrtFree(wDev);
    aclrtFree(xDev);
    aclrtFreeHost(expertIdHost);
    aclrtFreeHost(logitsHost);
    aclrtFreeHost(wHost);
    aclrtFreeHost(xHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    printf("[main] device run complete. Compare against the Python golden with:\n"
           "       python ../scripts/compare_outputs.py\n");
    return 0;
}
