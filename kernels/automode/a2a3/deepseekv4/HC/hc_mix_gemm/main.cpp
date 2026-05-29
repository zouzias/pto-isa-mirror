/**
 * main.cpp - host driver for hc_mix_gemm  (FP32 x FP32 -> FP32 mixes).
 *
 * I/O contract (all little-endian, contiguous, no header):
 *   ./input/input_x.bin        M*K   float32   (x_flat = x.flatten(2).float())
 *   ./input/input_hc_fn.bin    N*K   float32   (HC mixing weight)
 *   ./input/input_rsqrt.bin    M     float32   (precomputed rsqrt of row-mean-square)
 *   ./output/golden_mixes.bin  M*N   float32   (numpy reference)
 *   ./output/output_mixes.bin  M*N   float32   (kernel output)
 *
 *   M = B*S,  K = HC_MULT*DIM,  N = MIX_HC  (see ../README.md).
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

extern "C" void launch_hc_mix_gemm(uint8_t *mixes, uint8_t *x, uint8_t *hc_fn,
                                   uint8_t *rsqrt, void *stream);

int main()
{
    constexpr int kB     = kHcB;
    constexpr int kS     = kHcS;
    constexpr int kDim   = kHcDim;
    constexpr int kHcM   = kHcMult;
    constexpr int kMixHc = kHcMixHc;

    constexpr int kM = kB * kS;
    constexpr int kK = kHcM * kDim;
    constexpr int kN = kMixHc;

    constexpr size_t floatBytes = 4;

    size_t xBytes     = static_cast<size_t>(kM) * kK * floatBytes;
    size_t wBytes     = static_cast<size_t>(kN) * kK * floatBytes;
    size_t rsqrtBytes = static_cast<size_t>(kM) * floatBytes;
    size_t mixBytes   = static_cast<size_t>(kM) * kN * floatBytes;

    printf("[hc_mix_gemm] B=%d S=%d DIM=%d HC_MULT=%d MIX_HC=%d  -> M=%d K=%d N=%d\n",
           kB, kS, kDim, kHcM, kMixHc, kM, kK, kN);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint8_t *xHost = nullptr, *wHost = nullptr, *rHost = nullptr, *mHost = nullptr;
    uint8_t *xDev  = nullptr, *wDev  = nullptr, *rDev  = nullptr, *mDev  = nullptr;

    aclrtMallocHost((void **)&xHost, xBytes);
    aclrtMallocHost((void **)&wHost, wBytes);
    aclrtMallocHost((void **)&rHost, rsqrtBytes);
    aclrtMallocHost((void **)&mHost, mixBytes);

    aclrtMalloc((void **)&xDev, xBytes,     ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&wDev, wBytes,     ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&rDev, rsqrtBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&mDev, mixBytes,   ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile("./input/input_x.bin",     xBytes,     xHost, xBytes);
    ReadFile("./input/input_hc_fn.bin", wBytes,     wHost, wBytes);
    ReadFile("./input/input_rsqrt.bin", rsqrtBytes, rHost, rsqrtBytes);

    aclrtMemcpy(xDev, xBytes,     xHost, xBytes,     ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(wDev, wBytes,     wHost, wBytes,     ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(rDev, rsqrtBytes, rHost, rsqrtBytes, ACL_MEMCPY_HOST_TO_DEVICE);

    (void)PtoTiming::TimeKernelCallUs("hc_mix_gemm", stream, [&]() {
        launch_hc_mix_gemm(mDev, xDev, wDev, rDev, stream);
    });

    aclrtMemcpy(mHost, mixBytes, mDev, mixBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    WriteFile("./output/output_mixes.bin", mHost, mixBytes);

    aclrtFree(mDev);
    aclrtFree(rDev);
    aclrtFree(wDev);
    aclrtFree(xDev);
    aclrtFreeHost(mHost);
    aclrtFreeHost(rHost);
    aclrtFreeHost(wHost);
    aclrtFreeHost(xHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<float> golden(mixBytes / sizeof(float));
    std::vector<float> devFinal(mixBytes / sizeof(float));
    ReadFile("./output/golden_mixes.bin", mixBytes, golden.data(),   mixBytes);
    ReadFile("./output/output_mixes.bin", mixBytes, devFinal.data(), mixBytes);

    // FP32 inputs with small integer ranges in numpy reference. Cumulative
    // drift over K = HC_MULT * DIM is bounded; tolerance reflects FP32 cube
    // reduction-order variance.
    bool ok = ResultCmp(golden, devFinal, 1e-3f);
    if (ok) {
        printf("test data success\n");
    } else {
        printf("test data failed\n");
    }
    return ok ? 0 : 1;
}
