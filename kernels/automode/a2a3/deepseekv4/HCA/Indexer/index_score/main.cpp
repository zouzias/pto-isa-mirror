/**
 * main.cpp - host driver for Indexer.index_score
 *
 *   score[B, S, T] = sum_h (relu(einsum("bshd,btd->bsht", q, kv)) * w[..., None])
 *
 * I/O contract (all little-endian, contiguous, no header):
 *   ./input/input_q.bin        B*S*H*D  bfloat16
 *   ./input/input_kv.bin       B*T*D    bfloat16
 *   ./input/input_weights.bin  B*S*H    float32
 *   ./output/golden_score.bin  B*S*T    float32
 *   ./output/output_score.bin  B*S*T    float32
 */

#include "test_common.h"
#include "acl/acl.h"
#include "../../../../kernel_timing.h"
#include "generated_cases.h"

#include <cstdint>
#include <cstdio>
#include <vector>

using namespace std;
using namespace PtoTestCommon;

extern "C" void launch_index_score(uint8_t *score, uint8_t *q, uint8_t *kv_cache,
                                   uint8_t *weights,
                                   uint64_t B, uint64_t S, uint64_t T,
                                   uint64_t H, uint64_t D, void *stream);

int main()
{
    constexpr int kB        = kIdxB;
    constexpr int kS        = kIdxS;
    constexpr int kT        = kIdxT;
    constexpr int kH        = kIdxNHeads;
    constexpr int kD        = kIdxHeadDim;

    constexpr size_t bf16Bytes  = 2;
    constexpr size_t floatBytes = 4;

    size_t qBytes  = static_cast<size_t>(kB) * kS * kH * kD * bf16Bytes;
    size_t kvBytes = static_cast<size_t>(kB) * kT * kD * bf16Bytes;
    size_t wBytes  = static_cast<size_t>(kB) * kS * kH * floatBytes;
    size_t oBytes  = static_cast<size_t>(kB) * kS * kT * floatBytes;

    printf("[index_score] B=%d S=%d T=%d H=%d D=%d\n",
           kB, kS, kT, kH, kD);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint8_t *qHost = nullptr, *kvHost = nullptr, *wHost = nullptr, *oHost = nullptr;
    uint8_t *qDev  = nullptr, *kvDev  = nullptr, *wDev  = nullptr, *oDev  = nullptr;

    aclrtMallocHost((void **)&qHost,  qBytes);
    aclrtMallocHost((void **)&kvHost, kvBytes);
    aclrtMallocHost((void **)&wHost,  wBytes);
    aclrtMallocHost((void **)&oHost,  oBytes);

    aclrtMalloc((void **)&qDev,  qBytes,  ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&kvDev, kvBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&wDev,  wBytes,  ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&oDev,  oBytes,  ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile("./input/input_q.bin",       qBytes,  qHost,  qBytes);
    ReadFile("./input/input_kv.bin",      kvBytes, kvHost, kvBytes);
    ReadFile("./input/input_weights.bin", wBytes,  wHost,  wBytes);

    aclrtMemcpy(qDev,  qBytes,  qHost,  qBytes,  ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(kvDev, kvBytes, kvHost, kvBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(wDev,  wBytes,  wHost,  wBytes,  ACL_MEMCPY_HOST_TO_DEVICE);

    (void)PtoTiming::TimeKernelCallUs("index_score", stream, [&]() {
        launch_index_score(oDev, qDev, kvDev, wDev,
                           static_cast<uint64_t>(kB),
                           static_cast<uint64_t>(kS),
                           static_cast<uint64_t>(kT),
                           static_cast<uint64_t>(kH),
                           static_cast<uint64_t>(kD),
                           stream);
    });

    aclrtMemcpy(oHost, oBytes, oDev, oBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    WriteFile("./output/output_score.bin", oHost, oBytes);

    aclrtFree(oDev);
    aclrtFree(wDev);
    aclrtFree(kvDev);
    aclrtFree(qDev);
    aclrtFreeHost(oHost);
    aclrtFreeHost(wHost);
    aclrtFreeHost(kvHost);
    aclrtFreeHost(qHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<float> golden(oBytes / sizeof(float));
    std::vector<float> devFinal(oBytes / sizeof(float));
    ReadFile("./output/golden_score.bin", oBytes, golden.data(),   oBytes);
    ReadFile("./output/output_score.bin", oBytes, devFinal.data(), oBytes);

    bool ok = ResultCmp(golden, devFinal, 1e-2f);
    if (ok) {
        printf("test data success\n");
    } else {
        printf("test data failed\n");
    }
    return ok ? 0 : 1;
}
