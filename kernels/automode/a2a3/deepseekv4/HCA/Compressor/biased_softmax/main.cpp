/**
 * main.cpp - host driver for Compressor.biased_softmax (FP32).
 *
 * Computes:
 *     tmp           = score + broadcast(ape, (B, SB))     (FP32)
 *     softmax_score = softmax(tmp, dim=2)                 (FP32)
 *
 * Shapes:
 *   score, softmax_score : (B, SB, R, N)  FP32
 *   ape                  : (R, N)         FP32
 *
 * I/O contract (all little-endian, contiguous, no header):
 *   ./input/input_score.bin       B*SB*R*N  float32
 *   ./input/input_ape.bin         R*N       float32
 *   ./output/golden_softmax.bin   B*SB*R*N  float32
 *   ./output/output_softmax.bin   B*SB*R*N  float32
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

extern "C" void launch_biased_softmax(uint8_t *out, uint8_t *score, uint8_t *ape,
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

    size_t scoreBytes = static_cast<size_t>(kB) * kSB * kR * kN * floatBytes;
    size_t apeBytes   = static_cast<size_t>(kR) * kN * floatBytes;
    size_t outBytes   = scoreBytes;

    printf("[biased_softmax] B=%d SB=%d R=%d N=%d (ratio=%d overlap=%d coff=%d)\n",
           kB, kSB, kR, kN, kRatio, kOverlap, kCoff);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint8_t *scoreHost = nullptr, *apeHost = nullptr, *outHost = nullptr;
    uint8_t *scoreDev  = nullptr, *apeDev  = nullptr, *outDev  = nullptr;

    aclrtMallocHost((void **)&scoreHost, scoreBytes);
    aclrtMallocHost((void **)&apeHost,   apeBytes);
    aclrtMallocHost((void **)&outHost,   outBytes);

    aclrtMalloc((void **)&scoreDev, scoreBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&apeDev,   apeBytes,   ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&outDev,   outBytes,   ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile("./input/input_score.bin", scoreBytes, scoreHost, scoreBytes);
    ReadFile("./input/input_ape.bin",   apeBytes,   apeHost,   apeBytes);

    aclrtMemcpy(scoreDev, scoreBytes, scoreHost, scoreBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(apeDev,   apeBytes,   apeHost,   apeBytes,   ACL_MEMCPY_HOST_TO_DEVICE);

    (void)PtoTiming::TimeKernelCallUs("biased_softmax", stream, [&]() {
        launch_biased_softmax(outDev, scoreDev, apeDev,
                              static_cast<uint64_t>(kB),
                              static_cast<uint64_t>(kSB),
                              static_cast<uint64_t>(kR),
                              static_cast<uint64_t>(kN),
                              stream);
    });

    aclrtMemcpy(outHost, outBytes, outDev, outBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    WriteFile("./output/output_softmax.bin", outHost, outBytes);

    aclrtFree(outDev);
    aclrtFree(apeDev);
    aclrtFree(scoreDev);
    aclrtFreeHost(outHost);
    aclrtFreeHost(apeHost);
    aclrtFreeHost(scoreHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<float> golden(outBytes / sizeof(float));
    std::vector<float> devFinal(outBytes / sizeof(float));
    ReadFile("./output/golden_softmax.bin", outBytes, golden.data(),   outBytes);
    ReadFile("./output/output_softmax.bin", outBytes, devFinal.data(), outBytes);

    // FP32 throughout — softmax shouldn't accumulate much error beyond the
    // exp ULP. Tight tolerance.
    bool ok = ResultCmp(golden, devFinal, 1e-3f);
    if (ok) {
        printf("test data success\n");
    } else {
        printf("test data failed\n");
    }
    return ok ? 0 : 1;
}
