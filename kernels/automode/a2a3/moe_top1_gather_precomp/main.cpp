/**
 * main.cpp - host driver for moe_top1_gather_precomp (Milestone 1a).
 *
 * Pattern source: kernels/automode/a2a3/add_tile_array/main.cpp.
 *
 * I/O contract (all little-endian, contiguous, no header):
 *   ../input/input_tokens.bin            (kT * kH * sizeof(float))
 *   ../input/input_packed_to_token.bin   (kT * sizeof(uint32_t))
 *   ../output/golden_packed_tokens.bin   (kT * kH * sizeof(float)) (gen_data.py)
 *   ../output/output_packed_tokens.bin   (kT * kH * sizeof(float)) (this driver)
 *
 * Inputs are integer-valued floats so element-wise equality is exact in
 * IEEE-754 float32; ResultCmp tolerance 0.001f is comfortably wide.
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
void launchMoeTop1GatherPrecomp(T *packed_tokens, T *tokens,
                                uint32_t *packed_to_token, void *stream);

template <typename T, int kT_, int kH_>
inline bool ValidateDataResults(size_t outFileSize)
{
    std::vector<T> golden(outFileSize / sizeof(T));
    std::vector<T> devFinal(outFileSize / sizeof(T));

    ReadFile("../output/golden_packed_tokens.bin", outFileSize, golden.data(), outFileSize);
    ReadFile("../output/output_packed_tokens.bin", outFileSize, devFinal.data(), outFileSize);

    bool ret = ResultCmp(golden, devFinal, 0.001f);
    if (ret) {
        printf("test data success\n");
    } else {
        printf("test data failed\n");
    }
    return ret;
}

template <typename T, int kT_, int kH_>
void MoeTop1GatherPrecomp()
{
    constexpr size_t totalElements = static_cast<size_t>(kT_) * kH_;
    size_t tokensBytes  = totalElements * sizeof(T);
    size_t indicesBytes = static_cast<size_t>(kT_) * sizeof(uint32_t);
    size_t packedBytes  = tokensBytes;

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    T        *tokensHost = nullptr, *packedHost = nullptr;
    uint32_t *ptHost = nullptr;
    T        *tokensDev = nullptr, *packedDev = nullptr;
    uint32_t *ptDev = nullptr;

    aclrtMallocHost((void **)(&tokensHost), tokensBytes);
    aclrtMallocHost((void **)(&packedHost), packedBytes);
    aclrtMallocHost((void **)(&ptHost),     indicesBytes);

    aclrtMalloc((void **)&tokensDev, tokensBytes,  ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&packedDev, packedBytes,  ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&ptDev,     indicesBytes, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile("../input/input_tokens.bin",          tokensBytes,  tokensHost, tokensBytes);
    ReadFile("../input/input_packed_to_token.bin", indicesBytes, ptHost,     indicesBytes);

    aclrtMemcpy(tokensDev, tokensBytes,  tokensHost, tokensBytes,  ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(ptDev,     indicesBytes, ptHost,     indicesBytes, ACL_MEMCPY_HOST_TO_DEVICE);

    (void)PtoTiming::TimeKernelCallUs("moe_top1_gather_precomp", stream, [&]() {
        launchMoeTop1GatherPrecomp<T>(packedDev, tokensDev, ptDev, stream);
    });
    aclrtMemcpy(packedHost, packedBytes, packedDev, packedBytes, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile("../output/output_packed_tokens.bin", packedHost, packedBytes);

    aclrtFree(ptDev);
    aclrtFree(packedDev);
    aclrtFree(tokensDev);
    aclrtFreeHost(ptHost);
    aclrtFreeHost(packedHost);
    aclrtFreeHost(tokensHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    bool dataSuccess = ValidateDataResults<T, kT_, kH_>(packedBytes);
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
    MoeTop1GatherPrecomp<float, kT, kH>();
    return 0;
}
