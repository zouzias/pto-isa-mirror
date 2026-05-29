/**
 * main.cpp — host driver for DeepSeek-V4 gather.
 *
 * I/O contract (all little-endian, contiguous, no header):
 *   ./input/input_B.bin        (T*N_ACTIVATED + 16) * DIM      bfloat16
 *   ./input/input_A_id.bin     (T*N_ACTIVATED + 16)            int32
 *   ./input/input_rank_id.bin  (T*N_ACTIVATED + 16)            int32
 *   ./input/input_weights.bin  T * N_ACTIVATED                 float32
 *   ./output/golden_Y.bin      T * DIM                         bfloat16
 *   ./output/output_Y.bin      (kernel-emitted)
 *
 * Y must be device-zero-initialized before launch (the kernel does +=).
 */

#include "test_common.h"
#include "acl/acl.h"
#include "../../../kernel_timing.h"
#include "generated_cases.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

using namespace std;
using namespace PtoTestCommon;

extern "C" void launch_gather(uint8_t *Y, uint8_t *B,
                              int32_t *A_id, int32_t *rank_id,
                              float *weights, void *stream);

int main()
{
    constexpr int kT          = kDsmoeT;
    constexpr int kDim        = kDsmoeDim;
    constexpr int kNActivated = kDsmoeNActivated;
    constexpr int kAlloc      = kT * kNActivated + 16;

    constexpr size_t bf16Bytes = 2;
    constexpr size_t f32Bytes  = 4;
    constexpr size_t i32Bytes  = 4;

    size_t bBytes    = static_cast<size_t>(kAlloc) * kDim          * bf16Bytes;
    size_t yBytes    = static_cast<size_t>(kT)     * kDim          * bf16Bytes;
    size_t aIdBytes  = static_cast<size_t>(kAlloc)                 * i32Bytes;
    size_t wBytes    = static_cast<size_t>(kT)     * kNActivated   * f32Bytes;

    printf("[gather] T=%d DIM=%d N_ACTIVATED=%d kAlloc=%d\n",
           kT, kDim, kNActivated, kAlloc);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint8_t *bHost = nullptr, *yHost = nullptr;
    int32_t *aIdHost = nullptr, *rankIdHost = nullptr;
    uint8_t *wHost = nullptr;
    uint8_t *bDev = nullptr, *yDev = nullptr;
    int32_t *aIdDev = nullptr, *rankIdDev = nullptr;
    uint8_t *wDev = nullptr;

    aclrtMallocHost((void **)&bHost,      bBytes);
    aclrtMallocHost((void **)&yHost,      yBytes);
    aclrtMallocHost((void **)&aIdHost,    aIdBytes);
    aclrtMallocHost((void **)&rankIdHost, aIdBytes);
    aclrtMallocHost((void **)&wHost,      wBytes);

    aclrtMalloc((void **)&bDev,      bBytes,   ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&yDev,      yBytes,   ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&aIdDev,    aIdBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&rankIdDev, aIdBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&wDev,      wBytes,   ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile("./input/input_B.bin",       bBytes,   bHost,      bBytes);
    ReadFile("./input/input_A_id.bin",    aIdBytes, aIdHost,    aIdBytes);
    ReadFile("./input/input_rank_id.bin", aIdBytes, rankIdHost, aIdBytes);
    ReadFile("./input/input_weights.bin", wBytes,   wHost,      wBytes);

    // Y must be device-zeroed before launch (kernel does +=).
    aclrtMemset(yDev, yBytes, 0x00, yBytes);

    aclrtMemcpy(bDev,      bBytes,   bHost,      bBytes,   ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(aIdDev,    aIdBytes, aIdHost,    aIdBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(rankIdDev, aIdBytes, rankIdHost, aIdBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(wDev,      wBytes,   wHost,      wBytes,   ACL_MEMCPY_HOST_TO_DEVICE);

    (void)PtoTiming::TimeKernelCallUs("gather", stream, [&]() {
        launch_gather(yDev, bDev, aIdDev, rankIdDev,
                      reinterpret_cast<float *>(wDev), stream);
    });

    aclrtMemcpy(yHost, yBytes, yDev, yBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    WriteFile("./output/output_Y.bin", yHost, yBytes);

    aclrtFree(wDev);
    aclrtFree(rankIdDev);
    aclrtFree(aIdDev);
    aclrtFree(yDev);
    aclrtFree(bDev);
    aclrtFreeHost(wHost);
    aclrtFreeHost(rankIdHost);
    aclrtFreeHost(aIdHost);
    aclrtFreeHost(yHost);
    aclrtFreeHost(bHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<uint16_t> goldBits(yBytes / sizeof(uint16_t));
    std::vector<uint16_t> outBits (yBytes / sizeof(uint16_t));
    ReadFile("./output/golden_Y.bin", yBytes, goldBits.data(), yBytes);
    ReadFile("./output/output_Y.bin", yBytes, outBits.data(),  yBytes);

    std::vector<float> gold(goldBits.size());
    std::vector<float> dev (outBits.size());
    for (size_t i = 0; i < gold.size(); ++i) {
        uint32_t g = static_cast<uint32_t>(goldBits[i]) << 16;
        uint32_t o = static_cast<uint32_t>(outBits [i]) << 16;
        std::memcpy(&gold[i], &g, sizeof(float));
        std::memcpy(&dev [i], &o, sizeof(float));
    }

    bool ok = ResultCmp(gold, dev, 1e-2f);
    if (ok) {
        printf("test data success\n");
    } else {
        printf("test data failed\n");
    }
    return ok ? 0 : 1;
}
