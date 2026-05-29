/**
 * main.cpp — host driver for DeepSeek-V4 expert_ffn.
 *
 * I/O contract (all little-endian, contiguous, no header):
 *   ./input/input_A.bin              (T*N_ACTIVATED + 16) * DIM      bfloat16
 *   ./input/input_W1.bin             N_ROUTED * INTER_DIM * DIM      bfloat16
 *   ./input/input_W3.bin             N_ROUTED * INTER_DIM * DIM      bfloat16
 *   ./input/input_W2.bin             N_ROUTED * DIM       * INTER_DIM bfloat16
 *   ./input/input_weights.bin        (T*N_ACTIVATED + 16)            float32 (or zeros)
 *   ./input/input_expert_count.bin   N_ROUTED                        int32
 *   ./input/input_expert_start.bin   N_ROUTED                        int32
 *   ./output/golden_B.bin            (T*N_ACTIVATED + 16) * DIM      float32
 *   ./output/output_B.bin            (kernel-emitted)
 */

#include "test_common.h"
#include "acl/acl.h"
#include "../../../kernel_timing.h"
#include "generated_cases.h"

#include <cstdint>
#include <cstdio>
#include <vector>

using namespace std;
using namespace PtoTestCommon;

extern "C" void launch_expert_ffn(uint8_t *B, uint8_t *A,
                                  int32_t *expert_count, int32_t *expert_start,
                                  uint8_t *W1, uint8_t *W3, uint8_t *W2,
                                  float *weights, void *stream);

int main()
{
    constexpr int kT          = kDsmoeT;
    constexpr int kDim        = kDsmoeDim;
    constexpr int kInterDim   = kDsmoeInterDim;
    constexpr int kNRouted    = kDsmoeNRouted;
    constexpr int kNActivated = kDsmoeNActivated;
    constexpr int kAlloc      = kT * kNActivated + 16;

    constexpr size_t bf16Bytes = 2;
    constexpr size_t f32Bytes  = 4;
    constexpr size_t i32Bytes  = 4;

    size_t aBytes = static_cast<size_t>(kAlloc)    * kDim                    * bf16Bytes;
    size_t bBytes = static_cast<size_t>(kAlloc)    * kDim                    * f32Bytes;
    size_t w1Bytes= static_cast<size_t>(kNRouted)  * kInterDim * kDim        * bf16Bytes;
    size_t w3Bytes= w1Bytes;
    size_t w2Bytes= static_cast<size_t>(kNRouted)  * kDim      * kInterDim   * bf16Bytes;
    size_t wBytes = static_cast<size_t>(kAlloc)                              * f32Bytes;
    size_t expBytes = static_cast<size_t>(kNRouted)                          * i32Bytes;

    printf("[expert_ffn] T=%d DIM=%d INTER_DIM=%d N_ROUTED=%d N_ACTIVATED=%d kAlloc=%d\n",
           kT, kDim, kInterDim, kNRouted, kNActivated, kAlloc);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint8_t *aHost = nullptr, *bHost = nullptr;
    uint8_t *w1Host = nullptr, *w3Host = nullptr, *w2Host = nullptr;
    uint8_t *wHost = nullptr;
    int32_t *countHost = nullptr, *startHost = nullptr;

    uint8_t *aDev = nullptr, *bDev = nullptr;
    uint8_t *w1Dev = nullptr, *w3Dev = nullptr, *w2Dev = nullptr;
    uint8_t *wDev = nullptr;
    int32_t *countDev = nullptr, *startDev = nullptr;

    aclrtMallocHost((void **)&aHost,     aBytes);
    aclrtMallocHost((void **)&bHost,     bBytes);
    aclrtMallocHost((void **)&w1Host,    w1Bytes);
    aclrtMallocHost((void **)&w3Host,    w3Bytes);
    aclrtMallocHost((void **)&w2Host,    w2Bytes);
    aclrtMallocHost((void **)&wHost,     wBytes);
    aclrtMallocHost((void **)&countHost, expBytes);
    aclrtMallocHost((void **)&startHost, expBytes);

    aclrtMalloc((void **)&aDev,     aBytes,    ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&bDev,     bBytes,    ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&w1Dev,    w1Bytes,   ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&w3Dev,    w3Bytes,   ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&w2Dev,    w2Bytes,   ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&wDev,     wBytes,    ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&countDev, expBytes,  ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&startDev, expBytes,  ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile("./input/input_A.bin",            aBytes,    aHost,    aBytes);
    ReadFile("./input/input_W1.bin",           w1Bytes,   w1Host,   w1Bytes);
    ReadFile("./input/input_W3.bin",           w3Bytes,   w3Host,   w3Bytes);
    ReadFile("./input/input_W2.bin",           w2Bytes,   w2Host,   w2Bytes);
    ReadFile("./input/input_weights.bin",      wBytes,    wHost,    wBytes);
    ReadFile("./input/input_expert_count.bin", expBytes,  countHost,expBytes);
    ReadFile("./input/input_expert_start.bin", expBytes,  startHost,expBytes);

    aclrtMemcpy(aDev,     aBytes,    aHost,     aBytes,    ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(w1Dev,    w1Bytes,   w1Host,    w1Bytes,   ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(w3Dev,    w3Bytes,   w3Host,    w3Bytes,   ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(w2Dev,    w2Bytes,   w2Host,    w2Bytes,   ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(wDev,     wBytes,    wHost,     wBytes,    ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(countDev, expBytes,  countHost, expBytes,  ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(startDev, expBytes,  startHost, expBytes,  ACL_MEMCPY_HOST_TO_DEVICE);

    (void)PtoTiming::TimeKernelCallUs("expert_ffn", stream, [&]() {
        launch_expert_ffn(bDev, aDev, countDev, startDev,
                          w1Dev, w3Dev, w2Dev,
                          reinterpret_cast<float *>(wDev), stream);
    });

    aclrtMemcpy(bHost, bBytes, bDev, bBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    WriteFile("./output/output_B.bin", bHost, bBytes);

    aclrtFree(startDev);
    aclrtFree(countDev);
    aclrtFree(wDev);
    aclrtFree(w2Dev);
    aclrtFree(w3Dev);
    aclrtFree(w1Dev);
    aclrtFree(bDev);
    aclrtFree(aDev);
    aclrtFreeHost(startHost);
    aclrtFreeHost(countHost);
    aclrtFreeHost(wHost);
    aclrtFreeHost(w2Host);
    aclrtFreeHost(w3Host);
    aclrtFreeHost(w1Host);
    aclrtFreeHost(bHost);
    aclrtFreeHost(aHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<float> golden(bBytes / sizeof(float));
    std::vector<float> dev   (bBytes / sizeof(float));
    ReadFile("./output/golden_B.bin", bBytes, golden.data(), bBytes);
    ReadFile("./output/output_B.bin", bBytes, dev.data(),    bBytes);

    bool ok = ResultCmp(golden, dev, 1e-2f);
    if (ok) {
        printf("test data success\n");
    } else {
        printf("test data failed\n");
    }
    return ok ? 0 : 1;
}
