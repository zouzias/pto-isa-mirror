/**
 * main.cpp - host driver for moe_top1_permute (Milestone 1b).
 *
 * Pattern source: kernels/automode/a2a3/add_tile_array/main.cpp and the M1a
 * moe_top1_gather_precomp/main.cpp; extended with three extra GM buffers
 * for the device-emitted expert_count / expert_start / token_to_packed.
 *
 * I/O contract (all little-endian, contiguous, no header):
 *   ../input/input_tokens.bin             (kT * kH float32)
 *   ../input/input_expert_id.bin          (kT      int32)
 *   ../output/golden_packed_tokens.bin    (kT * kH float32)
 *   ../output/golden_expert_count.bin     (kE      int32)
 *   ../output/golden_expert_start.bin     (kE      int32)
 *   ../output/golden_token_to_packed.bin  (kT      int32)
 *   ../output/output_*.bin                (same shapes; written by this driver)
 */

#include "test_common.h"
#include "acl/acl.h"
#include "../kernel_timing.h"

#include <cstdint>
#include <cstdio>
#include <vector>

using namespace std;
using namespace PtoTestCommon;

template <typename T>
void launchMoeTop1Permute(T *packed_tokens,
                          int32_t *expert_count, int32_t *expert_start,
                          int32_t *token_to_packed,
                          T *tokens, int32_t *expert_id, void *stream);

template <typename T, int kT_, int kH_, int kE_>
inline bool ValidateAllOutputs(size_t packedBytes, size_t expertBytes, size_t tokIdxBytes)
{
    std::vector<T>       goldPacked(packedBytes / sizeof(T));
    std::vector<T>       devPacked (packedBytes / sizeof(T));
    std::vector<int32_t> goldCount (expertBytes / sizeof(int32_t));
    std::vector<int32_t> devCount  (expertBytes / sizeof(int32_t));
    std::vector<int32_t> goldStart (expertBytes / sizeof(int32_t));
    std::vector<int32_t> devStart  (expertBytes / sizeof(int32_t));
    std::vector<int32_t> goldTtoP  (tokIdxBytes / sizeof(int32_t));
    std::vector<int32_t> devTtoP   (tokIdxBytes / sizeof(int32_t));

    ReadFile("../output/golden_packed_tokens.bin",   packedBytes, goldPacked.data(), packedBytes);
    ReadFile("../output/output_packed_tokens.bin",   packedBytes, devPacked.data(),  packedBytes);
    ReadFile("../output/golden_expert_count.bin",    expertBytes, goldCount.data(),  expertBytes);
    ReadFile("../output/output_expert_count.bin",    expertBytes, devCount.data(),   expertBytes);
    ReadFile("../output/golden_expert_start.bin",    expertBytes, goldStart.data(),  expertBytes);
    ReadFile("../output/output_expert_start.bin",    expertBytes, devStart.data(),   expertBytes);
    ReadFile("../output/golden_token_to_packed.bin", tokIdxBytes, goldTtoP.data(),   tokIdxBytes);
    ReadFile("../output/output_token_to_packed.bin", tokIdxBytes, devTtoP.data(),    tokIdxBytes);

    bool packedOk = ResultCmp(goldPacked, devPacked, 0.001f);
    bool countOk  = (goldCount  == devCount);
    bool startOk  = (goldStart  == devStart);
    bool ttopOk   = (goldTtoP   == devTtoP);

    printf("packed_tokens    : %s\n", packedOk ? "success" : "FAILED");
    printf("expert_count     : %s\n", countOk  ? "success" : "FAILED");
    printf("expert_start     : %s\n", startOk  ? "success" : "FAILED");
    printf("token_to_packed  : %s\n", ttopOk   ? "success" : "FAILED");

    bool ret = packedOk && countOk && startOk && ttopOk;
    if (ret) {
        printf("test data success\n");
    } else {
        printf("test data failed\n");
    }
    return ret;
}

template <typename T, int kT_, int kH_, int kE_>
void MoeTop1Permute()
{
    constexpr size_t totalTokenElems = static_cast<size_t>(kT_) * kH_;
    size_t tokensBytes  = totalTokenElems       * sizeof(T);
    size_t expertIdBytes = static_cast<size_t>(kT_) * sizeof(int32_t);
    size_t expertBytes  = static_cast<size_t>(kE_) * sizeof(int32_t);
    size_t tokIdxBytes  = static_cast<size_t>(kT_) * sizeof(int32_t);
    size_t packedBytes  = tokensBytes;

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    T       *tokensHost = nullptr, *packedHost = nullptr;
    int32_t *expertIdHost = nullptr;
    int32_t *countHost = nullptr, *startHost = nullptr, *ttopHost = nullptr;

    T       *tokensDev = nullptr, *packedDev = nullptr;
    int32_t *expertIdDev = nullptr;
    int32_t *countDev = nullptr, *startDev = nullptr, *ttopDev = nullptr;

    aclrtMallocHost((void **)&tokensHost,   tokensBytes);
    aclrtMallocHost((void **)&packedHost,   packedBytes);
    aclrtMallocHost((void **)&expertIdHost, expertIdBytes);
    aclrtMallocHost((void **)&countHost,    expertBytes);
    aclrtMallocHost((void **)&startHost,    expertBytes);
    aclrtMallocHost((void **)&ttopHost,     tokIdxBytes);

    aclrtMalloc((void **)&tokensDev,   tokensBytes,   ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&packedDev,   packedBytes,   ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&expertIdDev, expertIdBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&countDev,    expertBytes,   ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&startDev,    expertBytes,   ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&ttopDev,     tokIdxBytes,   ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile("../input/input_tokens.bin",    tokensBytes,   tokensHost,   tokensBytes);
    ReadFile("../input/input_expert_id.bin", expertIdBytes, expertIdHost, expertIdBytes);

    aclrtMemcpy(tokensDev,   tokensBytes,   tokensHost,   tokensBytes,   ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(expertIdDev, expertIdBytes, expertIdHost, expertIdBytes, ACL_MEMCPY_HOST_TO_DEVICE);

    (void)PtoTiming::TimeKernelCallUs("moe_top1_permute", stream, [&]() {
        launchMoeTop1Permute<T>(packedDev, countDev, startDev, ttopDev,
                                tokensDev, expertIdDev, stream);
    });
    aclrtMemcpy(packedHost, packedBytes, packedDev, packedBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(countHost,  expertBytes, countDev,  expertBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(startHost,  expertBytes, startDev,  expertBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(ttopHost,   tokIdxBytes, ttopDev,   tokIdxBytes, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile("../output/output_packed_tokens.bin",   packedHost, packedBytes);
    WriteFile("../output/output_expert_count.bin",    countHost,  expertBytes);
    WriteFile("../output/output_expert_start.bin",    startHost,  expertBytes);
    WriteFile("../output/output_token_to_packed.bin", ttopHost,   tokIdxBytes);

    aclrtFree(ttopDev);
    aclrtFree(startDev);
    aclrtFree(countDev);
    aclrtFree(expertIdDev);
    aclrtFree(packedDev);
    aclrtFree(tokensDev);
    aclrtFreeHost(ttopHost);
    aclrtFreeHost(startHost);
    aclrtFreeHost(countHost);
    aclrtFreeHost(expertIdHost);
    aclrtFreeHost(packedHost);
    aclrtFreeHost(tokensHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    bool dataSuccess = ValidateAllOutputs<T, kT_, kH_, kE_>(packedBytes, expertBytes, tokIdxBytes);
    if (dataSuccess) {
        printf("test success\n");
    } else {
        printf("test failed\n");
    }
}

int main()
{
    constexpr int kT = 256;
    constexpr int kH = 64;
    constexpr int kE = 4;
    MoeTop1Permute<float, kT, kH, kE>();
    return 0;
}
