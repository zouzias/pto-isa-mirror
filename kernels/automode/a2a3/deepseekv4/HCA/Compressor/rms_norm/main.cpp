/**
 * main.cpp - host driver for Compressor.rms_norm (FP32 RMSNorm along HEAD_DIM).
 *
 * Computes:
 *     var   = mean_h(x^2)
 *     x_hat = x * rsqrt(var + eps)
 *     out   = weight * x_hat
 *
 * Shapes:
 *   x, out : (B, SB, D)   FP32
 *   weight : (D,)         FP32
 *
 * D    = kCompHeadDim
 * SB   = kCompS / kCompRatio
 *
 * I/O contract (all little-endian, contiguous, no header):
 *   ./input/input_x.bin         B*SB*D    float32
 *   ./input/input_weight.bin    D         float32
 *   ./output/golden_out.bin     B*SB*D    float32
 *   ./output/output_out.bin     B*SB*D    float32
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

extern "C" void launch_rms_norm(uint8_t *out, uint8_t *x, uint8_t *weight,
                                uint64_t B, uint64_t SB, uint64_t D, void *stream);

int main()
{
    constexpr int kB     = kCompB;
    constexpr int kS     = kCompS;
    constexpr int kRatio = kCompRatio;
    constexpr int kD     = kCompHeadDim;
    constexpr int kSB    = kS / kRatio;

    constexpr size_t floatBytes = 4;

    size_t xBytes = static_cast<size_t>(kB) * kSB * kD * floatBytes;
    size_t wBytes = static_cast<size_t>(kD) * floatBytes;
    size_t oBytes = xBytes;

    printf("[rms_norm] B=%d SB=%d D=%d (eps=1e-6)\n", kB, kSB, kD);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint8_t *xHost = nullptr, *wHost = nullptr, *oHost = nullptr;
    uint8_t *xDev  = nullptr, *wDev  = nullptr, *oDev  = nullptr;

    aclrtMallocHost((void **)&xHost, xBytes);
    aclrtMallocHost((void **)&wHost, wBytes);
    aclrtMallocHost((void **)&oHost, oBytes);

    aclrtMalloc((void **)&xDev, xBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&wDev, wBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&oDev, oBytes, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile("./input/input_x.bin",      xBytes, xHost, xBytes);
    ReadFile("./input/input_weight.bin", wBytes, wHost, wBytes);

    aclrtMemcpy(xDev, xBytes, xHost, xBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(wDev, wBytes, wHost, wBytes, ACL_MEMCPY_HOST_TO_DEVICE);

    (void)PtoTiming::TimeKernelCallUs("rms_norm", stream, [&]() {
        launch_rms_norm(oDev, xDev, wDev,
                        static_cast<uint64_t>(kB),
                        static_cast<uint64_t>(kSB),
                        static_cast<uint64_t>(kD),
                        stream);
    });

    aclrtMemcpy(oHost, oBytes, oDev, oBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    WriteFile("./output/output_out.bin", oHost, oBytes);

    aclrtFree(oDev);
    aclrtFree(wDev);
    aclrtFree(xDev);
    aclrtFreeHost(oHost);
    aclrtFreeHost(wHost);
    aclrtFreeHost(xHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<float> golden(oBytes / sizeof(float));
    std::vector<float> devFinal(oBytes / sizeof(float));
    ReadFile("./output/golden_out.bin", oBytes, golden.data(),   oBytes);
    ReadFile("./output/output_out.bin", oBytes, devFinal.data(), oBytes);

    // FP32 throughout. Tight tolerance.
    bool ok = ResultCmp(golden, devFinal, 1e-3f);
    if (ok) {
        printf("test data success\n");
    } else {
        printf("test data failed\n");
    }
    return ok ? 0 : 1;
}
