/**
 * main.cpp - host driver for Compressor.c_comp (FP32 weighted-sum reduction).
 *
 * Computes:
 *     kv_comp[b, sb, n] = sum_r kv[b, sb, r, n] * softmax_score[b, sb, r, n]
 *
 * Shapes:
 *   kv, softmax_score : (B, SB, R, N)  FP32
 *   kv_comp           : (B, SB, N)     FP32
 *
 * I/O contract (all little-endian, contiguous, no header):
 *   ./input/input_kv.bin           B*SB*R*N  float32
 *   ./input/input_softmax.bin      B*SB*R*N  float32
 *   ./output/golden_kv_comp.bin    B*SB*N    float32
 *   ./output/output_kv_comp.bin    B*SB*N    float32
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

extern "C" void launch_c_comp(uint8_t *out, uint8_t *kv, uint8_t *softmax_score,
                              uint64_t B, uint64_t SB, uint64_t R, uint64_t N,
                              void *stream);

int main()
{
    constexpr int kB        = kCompB;
    constexpr int kS        = kCompS;
    constexpr int kHeadDim  = kCompHeadDim;
    constexpr int kRatio    = kCompRatio;
    constexpr int kOverlap  = kCompOverlap;
    constexpr int kCoff     = kCompCoff;
    constexpr int kSB       = kS / kRatio;
    constexpr int kR        = kOverlap ? (2 * kRatio) : kRatio;
    constexpr int kN        = kCoff * kHeadDim;

    constexpr size_t floatBytes = 4;

    size_t kvBytes      = static_cast<size_t>(kB) * kSB * kR * kN * floatBytes;
    size_t softBytes    = kvBytes;
    size_t outBytes     = static_cast<size_t>(kB) * kSB * kN * floatBytes;

    printf("[c_comp] B=%d SB=%d R=%d N=%d -> out (B, SB, N)\n", kB, kSB, kR, kN);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint8_t *kvHost = nullptr, *softHost = nullptr, *outHost = nullptr;
    uint8_t *kvDev  = nullptr, *softDev  = nullptr, *outDev  = nullptr;

    aclrtMallocHost((void **)&kvHost,   kvBytes);
    aclrtMallocHost((void **)&softHost, softBytes);
    aclrtMallocHost((void **)&outHost,  outBytes);

    aclrtMalloc((void **)&kvDev,   kvBytes,   ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&softDev, softBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&outDev,  outBytes,  ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile("./input/input_kv.bin",      kvBytes,   kvHost,   kvBytes);
    ReadFile("./input/input_softmax.bin", softBytes, softHost, softBytes);

    aclrtMemcpy(kvDev,   kvBytes,   kvHost,   kvBytes,   ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(softDev, softBytes, softHost, softBytes, ACL_MEMCPY_HOST_TO_DEVICE);

    (void)PtoTiming::TimeKernelCallUs("c_comp", stream, [&]() {
        launch_c_comp(outDev, kvDev, softDev,
                      static_cast<uint64_t>(kB),
                      static_cast<uint64_t>(kSB),
                      static_cast<uint64_t>(kR),
                      static_cast<uint64_t>(kN),
                      stream);
    });

    aclrtMemcpy(outHost, outBytes, outDev, outBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    WriteFile("./output/output_kv_comp.bin", outHost, outBytes);

    aclrtFree(outDev);
    aclrtFree(softDev);
    aclrtFree(kvDev);
    aclrtFreeHost(outHost);
    aclrtFreeHost(softHost);
    aclrtFreeHost(kvHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<float> golden(outBytes / sizeof(float));
    std::vector<float> devFinal(outBytes / sizeof(float));
    ReadFile("./output/golden_kv_comp.bin", outBytes, golden.data(),   outBytes);
    ReadFile("./output/output_kv_comp.bin", outBytes, devFinal.data(), outBytes);

    // FP32 mul + small-R reduction. Tight tolerance.
    bool ok = ResultCmp(golden, devFinal, 1e-3f);
    if (ok) {
        printf("test data success\n");
    } else {
        printf("test data failed\n");
    }
    return ok ? 0 : 1;
}
