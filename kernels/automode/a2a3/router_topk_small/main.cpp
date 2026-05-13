/**
 * main.cpp - host driver for router_topk_small.
 *
 * SKELETON. Final file shape; the only thing missing is the kernel body
 * behind launchRouterTopkSmall.
 *
 * I/O contract (all little-endian, contiguous, no header):
 *   ../output/t.txt                     (single int; written by gen_data.py)
 *   ../output/k.txt                     (single int; written by gen_data.py)
 *   ../input/input_scores.bin           (T * E float32)
 *   ../output/golden_topk_values.bin    (T * K float32)
 *   ../output/golden_topk_indices.bin   (T * K uint32)
 *   ../output/output_topk_values.bin    (T * K float32; this driver)
 *   ../output/output_topk_indices.bin   (T * K uint32 ; this driver)
 *
 * Stage-isolation: poisons both output buffers BEFORE the launch so the
 * compare script can distinguish "kernel never wrote" from "wrote wrong
 * values" (same pattern as moe_segmented_ffn_top1/main.cpp).
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
constexpr int kE = 16;

// Non-template host-boundary wrapper exposed by the kernel TU. The kernel
// internally uses `float` (FP32) for scores/values and `uint32_t` for indices.
extern "C" void launchRouterTopkSmallFp32(uint8_t *topk_values,
                                          uint8_t *topk_indices,
                                          uint8_t *scores,
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
    constexpr size_t f32Bytes = 4;
    constexpr size_t u32Bytes = 4;

    constexpr uint8_t kPoisonVal = 0x5A;  // FP32 topk_values
    constexpr uint8_t kPoisonIdx = 0x7B;  // uint32 topk_indices

    const int T = ReadIntFile("../output/t.txt");
    const int K = ReadIntFile("../output/k.txt");

    size_t scoresBytes = static_cast<size_t>(T) * kE * f32Bytes;
    size_t valBytes    = static_cast<size_t>(T) * K  * f32Bytes;
    size_t idxBytes    = static_cast<size_t>(T) * K  * u32Bytes;

    printf("[main] T=%d  E=%d  K=%d\n"
           "       scoresBytes=%zu  valBytes=%zu  idxBytes=%zu\n",
           T, kE, K, scoresBytes, valBytes, idxBytes);

    if (!CheckAcl(aclInit(nullptr), "aclInit")) std::exit(3);
    if (!CheckAcl(aclrtSetDevice(0), "aclrtSetDevice")) std::exit(3);
    aclrtStream stream;
    if (!CheckAcl(aclrtCreateStream(&stream), "aclrtCreateStream")) std::exit(3);

    uint8_t *scoresHost = nullptr, *valHost = nullptr, *idxHost = nullptr;
    uint8_t *scoresDev  = nullptr, *valDev  = nullptr, *idxDev  = nullptr;

    CheckAcl(aclrtMallocHost((void **)(&scoresHost), scoresBytes), "aclrtMallocHost(scoresHost)");
    CheckAcl(aclrtMallocHost((void **)(&valHost),    valBytes),    "aclrtMallocHost(valHost)");
    CheckAcl(aclrtMallocHost((void **)(&idxHost),    idxBytes),    "aclrtMallocHost(idxHost)");

    CheckAcl(aclrtMalloc((void **)&scoresDev, scoresBytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc(scoresDev)");
    CheckAcl(aclrtMalloc((void **)&valDev,    valBytes,    ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc(valDev)");
    CheckAcl(aclrtMalloc((void **)&idxDev,    idxBytes,    ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc(idxDev)");

    ReadFile("../input/input_scores.bin", scoresBytes, scoresHost, scoresBytes);
    CheckAcl(aclrtMemcpy(scoresDev, scoresBytes, scoresHost, scoresBytes, ACL_MEMCPY_HOST_TO_DEVICE),
             "aclrtMemcpy(scoresDev)");

    CheckAcl(aclrtMemset(valDev, valBytes, kPoisonVal, valBytes), "aclrtMemset(valDev=0x5A)");
    CheckAcl(aclrtMemset(idxDev, idxBytes, kPoisonIdx, idxBytes), "aclrtMemset(idxDev=0x7B)");

    launchRouterTopkSmallFp32(valDev, idxDev, scoresDev, stream);

    if (!CheckAcl(aclrtSynchronizeStream(stream), "aclrtSynchronizeStream")) {
        std::cerr << "[main] stream sync failed.\n";
    }

    CheckAcl(aclrtMemcpy(valHost, valBytes, valDev, valBytes, ACL_MEMCPY_DEVICE_TO_HOST),
             "aclrtMemcpy(valHost <- valDev)");
    CheckAcl(aclrtMemcpy(idxHost, idxBytes, idxDev, idxBytes, ACL_MEMCPY_DEVICE_TO_HOST),
             "aclrtMemcpy(idxHost <- idxDev)");

    WriteFile("../output/output_topk_values.bin",  valHost, valBytes);
    WriteFile("../output/output_topk_indices.bin", idxHost, idxBytes);

    printf("[main] after launch: valHost[0]=0x%02X (poison=0x%02X), idxHost[0]=0x%02X (poison=0x%02X)\n",
           valHost[0], kPoisonVal, idxHost[0], kPoisonIdx);

    aclrtFree(idxDev);
    aclrtFree(valDev);
    aclrtFree(scoresDev);
    aclrtFreeHost(idxHost);
    aclrtFreeHost(valHost);
    aclrtFreeHost(scoresHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    printf("[main] device run complete. Compare against the Python golden with:\n"
           "       python ../scripts/compare_outputs.py\n");
    return 0;
}
