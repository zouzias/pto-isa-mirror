/**
 * main.cpp - host driver for Compressor.rope (FP32 rotary embedding).
 *
 * Computes apply_rotary_emb on the tail slice kv_comp[..., -rd:]:
 *     re_out = re_in * cs - im_in * sn
 *     im_out = re_in * sn + im_in * cs
 *
 * Shapes:
 *   kv_tail, rotated : (B, SB, RD)      FP32
 *   freqs_cos, freqs_sin : (SB, RD/2)   FP32
 *
 * I/O contract (all little-endian, contiguous, no header):
 *   ./input/input_kv_tail.bin       B*SB*RD       float32
 *   ./input/input_freqs_cos.bin     SB*(RD/2)     float32
 *   ./input/input_freqs_sin.bin     SB*(RD/2)     float32
 *   ./output/golden_rotated.bin     B*SB*RD       float32
 *   ./output/output_rotated.bin     B*SB*RD       float32
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

extern "C" void launch_rope(uint8_t *out, uint8_t *kv_tail,
                            uint8_t *freqs_cos, uint8_t *freqs_sin,
                            uint64_t B, uint64_t SB, uint64_t RD, void *stream);

int main()
{
    constexpr int kB      = kCompB;
    constexpr int kS      = kCompS;
    constexpr int kRatio  = kCompRatio;
    constexpr int kRD     = kCompRopeDim;
    constexpr int kSB     = kS / kRatio;
    constexpr int kHalfRD = kRD / 2;

    constexpr size_t floatBytes = 4;

    size_t tailBytes  = static_cast<size_t>(kB) * kSB * kRD     * floatBytes;
    size_t freqsBytes = static_cast<size_t>(kSB) * kHalfRD       * floatBytes;
    size_t outBytes   = tailBytes;

    printf("[rope] B=%d SB=%d RD=%d (half=%d)\n", kB, kSB, kRD, kHalfRD);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint8_t *tailHost = nullptr, *cosHost = nullptr, *sinHost = nullptr, *outHost = nullptr;
    uint8_t *tailDev  = nullptr, *cosDev  = nullptr, *sinDev  = nullptr, *outDev  = nullptr;

    aclrtMallocHost((void **)&tailHost, tailBytes);
    aclrtMallocHost((void **)&cosHost,  freqsBytes);
    aclrtMallocHost((void **)&sinHost,  freqsBytes);
    aclrtMallocHost((void **)&outHost,  outBytes);

    aclrtMalloc((void **)&tailDev, tailBytes,  ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&cosDev,  freqsBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&sinDev,  freqsBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&outDev,  outBytes,   ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile("./input/input_kv_tail.bin",   tailBytes,  tailHost, tailBytes);
    ReadFile("./input/input_freqs_cos.bin", freqsBytes, cosHost,  freqsBytes);
    ReadFile("./input/input_freqs_sin.bin", freqsBytes, sinHost,  freqsBytes);

    aclrtMemcpy(tailDev, tailBytes,  tailHost, tailBytes,  ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(cosDev,  freqsBytes, cosHost,  freqsBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(sinDev,  freqsBytes, sinHost,  freqsBytes, ACL_MEMCPY_HOST_TO_DEVICE);

    (void)PtoTiming::TimeKernelCallUs("rope", stream, [&]() {
        launch_rope(outDev, tailDev, cosDev, sinDev,
                    static_cast<uint64_t>(kB),
                    static_cast<uint64_t>(kSB),
                    static_cast<uint64_t>(kRD),
                    stream);
    });

    aclrtMemcpy(outHost, outBytes, outDev, outBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    WriteFile("./output/output_rotated.bin", outHost, outBytes);

    aclrtFree(outDev);
    aclrtFree(sinDev);
    aclrtFree(cosDev);
    aclrtFree(tailDev);
    aclrtFreeHost(outHost);
    aclrtFreeHost(sinHost);
    aclrtFreeHost(cosHost);
    aclrtFreeHost(tailHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<float> golden(outBytes / sizeof(float));
    std::vector<float> devFinal(outBytes / sizeof(float));
    ReadFile("./output/golden_rotated.bin", outBytes, golden.data(),   outBytes);
    ReadFile("./output/output_rotated.bin", outBytes, devFinal.data(), outBytes);

    // FP32 mul-add only; tight tolerance.
    bool ok = ResultCmp(golden, devFinal, 1e-3f);
    if (ok) {
        printf("test data success\n");
    } else {
        printf("test data failed\n");
    }
    return ok ? 0 : 1;
}
