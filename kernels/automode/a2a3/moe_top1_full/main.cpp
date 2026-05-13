/**
 * main.cpp - host driver for moe_top1_full.
 *
 * SKELETON. Final file shape. The kernel TU exposes one extern "C" wrapper
 * that fires five __global__ AICORE kernels on the same stream. The bodies
 * of those kernels are TODO; this driver is complete.
 *
 * I/O contract (all little-endian, contiguous, no header):
 *   ../output/t.txt                       (single int; T)
 *   ../output/t_padded_max.txt            (single int; host scratch upper bound)
 *   ../input/input_X.bin                  (T * H float16)
 *   ../input/input_W_router.bin           (H * E float16)
 *   ../input/input_W1.bin                 (kE * H * F float16)
 *   ../input/input_W2.bin                 (kE * F * H float16)
 *   ../output/golden_logits.bin           (T * E float32; debug)
 *   ../output/golden_expert_id.bin        (T     uint32 ; debug)
 *   ../output/golden_Y.bin                (T * H float32; primary)
 *   ../output/output_logits.bin           (T * E float32; this driver)
 *   ../output/output_expert_id.bin        (T     uint32 ; this driver)
 *   ../output/output_Y.bin                (T * H float32; this driver)
 *
 * Stage-isolation: all three observable output buffers (logits, expert_id, Y)
 * are poisoned with distinct bytes before the launch.
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

// Static configuration. Must match kernel.cpp and scripts/gen_data.py.
constexpr int kH      = 64;
constexpr int kF      = 64;
constexpr int kE      = 16;
constexpr int kTileM  = 128;

// Non-template host-boundary wrapper exposed by the kernel TU. Fires all
// five stage kernels on the same stream; ACL stream-order does the
// cross-kernel sequencing.
extern "C" void launchMoeTop1FullFp16(uint8_t *Y_fp32,
                                       uint8_t *expert_id_u32,
                                       uint8_t *logits_fp32,
                                       uint8_t *X_fp16,
                                       uint8_t *W_router_fp16,
                                       uint8_t *W1_fp16,
                                       uint8_t *W2_fp16,
                                       // host-managed device scratch buffers:
                                       uint8_t *packed_tokens_fp16,
                                       int32_t *expert_count,
                                       int32_t *expert_start,
                                       int32_t *token_to_packed,
                                       uint8_t *ffn_scratch_fp16,
                                       uint8_t *packed_output_fp32,
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

template <typename TVal>
static bool ValidateFloatBuffer(const char *goldenPath, const char *outputPath,
                                size_t numBytes, const char *label, float eps)
{
    std::vector<TVal> golden(numBytes / sizeof(TVal));
    std::vector<TVal> got   (numBytes / sizeof(TVal));
    ReadFile(goldenPath, numBytes, golden.data(), numBytes);
    ReadFile(outputPath, numBytes, got.data(),    numBytes);
    bool ok = ResultCmp(golden, got, eps);
    printf("[validate] %-10s : %s\n", label, ok ? "PASS" : "FAIL");
    return ok;
}

template <typename TIdx>
static bool ValidateIdxBuffer(const char *goldenPath, const char *outputPath,
                              size_t numBytes, const char *label)
{
    std::vector<TIdx> golden(numBytes / sizeof(TIdx));
    std::vector<TIdx> got   (numBytes / sizeof(TIdx));
    ReadFile(goldenPath, numBytes, golden.data(), numBytes);
    ReadFile(outputPath, numBytes, got.data(),    numBytes);
    bool ok = ResultCmp(golden, got, 0.0f);
    printf("[validate] %-10s : %s\n", label, ok ? "PASS" : "FAIL");
    return ok;
}

int main()
{
    constexpr size_t halfBytes = 2;
    constexpr size_t fp32Bytes = 4;
    constexpr size_t i32Bytes  = 4;
    constexpr size_t u32Bytes  = 4;

    constexpr uint8_t kPoisonLogits   = 0x5A;
    constexpr uint8_t kPoisonExpertId = 0x7B;
    constexpr uint8_t kPoisonY        = 0x4C;
    constexpr uint8_t kPoisonScratch  = 0x29;  // packed_tokens, ffn_scratch, packed_output

    const int T            = ReadIntFile("../output/t.txt");
    const int T_padded_max = ReadIntFile("../output/t_padded_max.txt");

    size_t xBytes        = static_cast<size_t>(T) * kH * halfBytes;
    size_t wRouterBytes  = static_cast<size_t>(kH) * kE * halfBytes;
    size_t w1Bytes       = static_cast<size_t>(kE) * kH * kF * halfBytes;
    size_t w2Bytes       = static_cast<size_t>(kE) * kF * kH * halfBytes;
    size_t logitsBytes   = static_cast<size_t>(T) * kE * fp32Bytes;
    size_t expertIdBytes = static_cast<size_t>(T)      * u32Bytes;
    size_t yBytes        = static_cast<size_t>(T) * kH * fp32Bytes;
    size_t metaBytes     = static_cast<size_t>(kE)     * i32Bytes;
    size_t tokToPackBy   = static_cast<size_t>(T)      * i32Bytes;
    size_t packedTokBy   = static_cast<size_t>(T_padded_max) * kH * halfBytes;
    size_t ffnScratchBy  = static_cast<size_t>(T_padded_max) * kF * halfBytes;
    size_t packedOutBy   = static_cast<size_t>(T_padded_max) * kH * fp32Bytes;

    printf("[main] T=%d  H=%d  F=%d  E=%d  kTileM=%d  T_padded_max=%d\n",
           T, kH, kF, kE, kTileM, T_padded_max);

    if (!CheckAcl(aclInit(nullptr), "aclInit")) std::exit(3);
    if (!CheckAcl(aclrtSetDevice(0), "aclrtSetDevice")) std::exit(3);
    aclrtStream stream;
    if (!CheckAcl(aclrtCreateStream(&stream), "aclrtCreateStream")) std::exit(3);

    uint8_t *xHost = nullptr, *wRouterHost = nullptr;
    uint8_t *w1Host = nullptr, *w2Host = nullptr;
    uint8_t *logitsHost = nullptr, *expertIdHost = nullptr, *yHost = nullptr;
    uint8_t *xDev = nullptr, *wRouterDev = nullptr;
    uint8_t *w1Dev = nullptr, *w2Dev = nullptr;
    uint8_t *logitsDev = nullptr, *expertIdDev = nullptr, *yDev = nullptr;
    uint8_t *packedTokDev = nullptr, *ffnScratchDev = nullptr, *packedOutDev = nullptr;
    int32_t *expertCountDev = nullptr, *expertStartDev = nullptr, *tokToPackDev = nullptr;

    CheckAcl(aclrtMallocHost((void **)(&xHost),        xBytes),        "aclrtMallocHost(xHost)");
    CheckAcl(aclrtMallocHost((void **)(&wRouterHost),  wRouterBytes),  "aclrtMallocHost(wRouterHost)");
    CheckAcl(aclrtMallocHost((void **)(&w1Host),       w1Bytes),       "aclrtMallocHost(w1Host)");
    CheckAcl(aclrtMallocHost((void **)(&w2Host),       w2Bytes),       "aclrtMallocHost(w2Host)");
    CheckAcl(aclrtMallocHost((void **)(&logitsHost),   logitsBytes),   "aclrtMallocHost(logitsHost)");
    CheckAcl(aclrtMallocHost((void **)(&expertIdHost), expertIdBytes), "aclrtMallocHost(expertIdHost)");
    CheckAcl(aclrtMallocHost((void **)(&yHost),        yBytes),        "aclrtMallocHost(yHost)");

    CheckAcl(aclrtMalloc((void **)&xDev,           xBytes,        ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc(xDev)");
    CheckAcl(aclrtMalloc((void **)&wRouterDev,     wRouterBytes,  ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc(wRouterDev)");
    CheckAcl(aclrtMalloc((void **)&w1Dev,          w1Bytes,       ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc(w1Dev)");
    CheckAcl(aclrtMalloc((void **)&w2Dev,          w2Bytes,       ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc(w2Dev)");
    CheckAcl(aclrtMalloc((void **)&logitsDev,      logitsBytes,   ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc(logitsDev)");
    CheckAcl(aclrtMalloc((void **)&expertIdDev,    expertIdBytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc(expertIdDev)");
    CheckAcl(aclrtMalloc((void **)&yDev,           yBytes,        ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc(yDev)");
    CheckAcl(aclrtMalloc((void **)&packedTokDev,   packedTokBy,   ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc(packedTokDev)");
    CheckAcl(aclrtMalloc((void **)&ffnScratchDev,  ffnScratchBy,  ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc(ffnScratchDev)");
    CheckAcl(aclrtMalloc((void **)&packedOutDev,   packedOutBy,   ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc(packedOutDev)");
    CheckAcl(aclrtMalloc((void **)&expertCountDev, metaBytes,     ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc(expertCountDev)");
    CheckAcl(aclrtMalloc((void **)&expertStartDev, metaBytes,     ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc(expertStartDev)");
    CheckAcl(aclrtMalloc((void **)&tokToPackDev,   tokToPackBy,   ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc(tokToPackDev)");

    ReadFile("../input/input_X.bin",         xBytes,       xHost,        xBytes);
    ReadFile("../input/input_W_router.bin",  wRouterBytes, wRouterHost,  wRouterBytes);
    ReadFile("../input/input_W1.bin",        w1Bytes,      w1Host,       w1Bytes);
    ReadFile("../input/input_W2.bin",        w2Bytes,      w2Host,       w2Bytes);

    CheckAcl(aclrtMemcpy(xDev,       xBytes,       xHost,       xBytes,       ACL_MEMCPY_HOST_TO_DEVICE), "memcpy(xDev)");
    CheckAcl(aclrtMemcpy(wRouterDev, wRouterBytes, wRouterHost, wRouterBytes, ACL_MEMCPY_HOST_TO_DEVICE), "memcpy(wRouterDev)");
    CheckAcl(aclrtMemcpy(w1Dev,      w1Bytes,      w1Host,      w1Bytes,      ACL_MEMCPY_HOST_TO_DEVICE), "memcpy(w1Dev)");
    CheckAcl(aclrtMemcpy(w2Dev,      w2Bytes,      w2Host,      w2Bytes,      ACL_MEMCPY_HOST_TO_DEVICE), "memcpy(w2Dev)");

    // Poison all observable / scratch buffers so the comparator can tell
    // "kernel never reached this stage" from "wrote wrong values".
    CheckAcl(aclrtMemset(logitsDev,     logitsBytes,    kPoisonLogits,   logitsBytes),    "memset(logitsDev)");
    CheckAcl(aclrtMemset(expertIdDev,   expertIdBytes,  kPoisonExpertId, expertIdBytes),  "memset(expertIdDev)");
    CheckAcl(aclrtMemset(yDev,          yBytes,         kPoisonY,        yBytes),         "memset(yDev)");
    CheckAcl(aclrtMemset(packedTokDev,  packedTokBy,    kPoisonScratch,  packedTokBy),    "memset(packedTokDev)");
    CheckAcl(aclrtMemset(ffnScratchDev, ffnScratchBy,   kPoisonScratch,  ffnScratchBy),   "memset(ffnScratchDev)");
    CheckAcl(aclrtMemset(packedOutDev,  packedOutBy,    kPoisonScratch,  packedOutBy),    "memset(packedOutDev)");
    CheckAcl(aclrtMemset(expertCountDev, metaBytes,     0,               metaBytes),      "memset(expertCountDev)");
    CheckAcl(aclrtMemset(expertStartDev, metaBytes,     0,               metaBytes),      "memset(expertStartDev)");
    CheckAcl(aclrtMemset(tokToPackDev,   tokToPackBy,   0,               tokToPackBy),    "memset(tokToPackDev)");

    launchMoeTop1FullFp16(
        yDev, expertIdDev, logitsDev,
        xDev, wRouterDev, w1Dev, w2Dev,
        packedTokDev, expertCountDev, expertStartDev, tokToPackDev,
        ffnScratchDev, packedOutDev, stream);

    if (!CheckAcl(aclrtSynchronizeStream(stream), "aclrtSynchronizeStream")) {
        std::cerr << "[main] stream sync failed.\n";
    }

    CheckAcl(aclrtMemcpy(logitsHost,   logitsBytes,   logitsDev,   logitsBytes,   ACL_MEMCPY_DEVICE_TO_HOST), "memcpy(logitsHost)");
    CheckAcl(aclrtMemcpy(expertIdHost, expertIdBytes, expertIdDev, expertIdBytes, ACL_MEMCPY_DEVICE_TO_HOST), "memcpy(expertIdHost)");
    CheckAcl(aclrtMemcpy(yHost,        yBytes,        yDev,        yBytes,        ACL_MEMCPY_DEVICE_TO_HOST), "memcpy(yHost)");

    WriteFile("../output/output_logits.bin",    logitsHost,   logitsBytes);
    WriteFile("../output/output_expert_id.bin", expertIdHost, expertIdBytes);
    WriteFile("../output/output_Y.bin",         yHost,        yBytes);

    printf("[main] poison sentinels after launch: "
           "logits[0]=0x%02X (poison=0x%02X)  "
           "expert_id[0]=0x%02X (poison=0x%02X)  "
           "Y[0]=0x%02X (poison=0x%02X)\n",
           logitsHost[0],   kPoisonLogits,
           expertIdHost[0], kPoisonExpertId,
           yHost[0],        kPoisonY);

    aclrtFree(tokToPackDev);
    aclrtFree(expertStartDev);
    aclrtFree(expertCountDev);
    aclrtFree(packedOutDev);
    aclrtFree(ffnScratchDev);
    aclrtFree(packedTokDev);
    aclrtFree(yDev);
    aclrtFree(expertIdDev);
    aclrtFree(logitsDev);
    aclrtFree(w2Dev);
    aclrtFree(w1Dev);
    aclrtFree(wRouterDev);
    aclrtFree(xDev);
    aclrtFreeHost(yHost);
    aclrtFreeHost(expertIdHost);
    aclrtFreeHost(logitsHost);
    aclrtFreeHost(w2Host);
    aclrtFreeHost(w1Host);
    aclrtFreeHost(wRouterHost);
    aclrtFreeHost(xHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    // Stage-isolation diagnostics: validate each observable buffer
    // independently so a downstream FAIL doesn't mask an upstream root cause.
    bool logitsOk   = ValidateFloatBuffer<float>(
        "../output/golden_logits.bin",
        "../output/output_logits.bin",
        logitsBytes,   "logits",     1e-3f);
    bool expertIdOk = ValidateIdxBuffer<uint32_t>(
        "../output/golden_expert_id.bin",
        "../output/output_expert_id.bin",
        expertIdBytes, "expert_id");
    bool yOk        = ValidateFloatBuffer<float>(
        "../output/golden_Y.bin",
        "../output/output_Y.bin",
        yBytes,        "Y",          1e-3f);

    if (logitsOk && expertIdOk && yOk) {
        printf("test success\n");
        return 0;
    }
    printf("test failed\n");
    return 1;
}
