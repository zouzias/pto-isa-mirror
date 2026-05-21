/**
 * main.cpp - host driver for router_matmul.
 *
 * I/O contract (all little-endian, contiguous, no header):
 *   ../input/input_x.bin          kT * kH  float16  (token features)
 *   ../input/input_w_router.bin   kH * kE  float16  (router weight matrix)
 *   ../output/golden_logits.bin   kT * kE  float32  (reference logits)
 *   ../output/output_logits.bin   kT * kE  float32  (kernel output)
 */

#include "test_common.h"
#include "acl/acl.h"
#include "../../kernel_timing.h"
#include "generated_cases.h"

#include <cstdint>
#include <cstdio>
#include <vector>

using namespace std;
using namespace PtoTestCommon;

extern "C" void launchRouterMatmulFp16(uint8_t *logits, uint8_t *x, uint8_t *w_router, void *stream);

int main()
{
    constexpr int kT = kMoeT;
    constexpr int kH = kMoeH;
    constexpr int kE = kMoeE;
    constexpr size_t halfBytes  = 2;
    constexpr size_t floatBytes = 4;

    size_t xBytes      = static_cast<size_t>(kT) * kH * halfBytes;
    size_t wBytes      = static_cast<size_t>(kH) * kE * halfBytes;
    size_t logitsBytes = static_cast<size_t>(kT) * kE * floatBytes;

    printf("[main] kT=%d  kH=%d  kE=%d\n"
           "       xBytes=%zu  wBytes=%zu  logitsBytes=%zu\n",
           kT, kH, kE, xBytes, wBytes, logitsBytes);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint8_t *xHost = nullptr, *wHost = nullptr, *logitsHost = nullptr;
    uint8_t *xDev  = nullptr, *wDev  = nullptr, *logitsDev  = nullptr;

    aclrtMallocHost((void **)&xHost,      xBytes);
    aclrtMallocHost((void **)&wHost,      wBytes);
    aclrtMallocHost((void **)&logitsHost, logitsBytes);

    aclrtMalloc((void **)&xDev,      xBytes,      ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&wDev,      wBytes,      ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&logitsDev, logitsBytes, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile("../input/input_x.bin",        xBytes, xHost, xBytes);
    ReadFile("../input/input_w_router.bin", wBytes, wHost, wBytes);

    aclrtMemcpy(xDev, xBytes, xHost, xBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(wDev, wBytes, wHost, wBytes, ACL_MEMCPY_HOST_TO_DEVICE);

    (void)PtoTiming::TimeKernelCallUs("router_matmul", stream, [&]() {
        launchRouterMatmulFp16(logitsDev, xDev, wDev, stream);
    });
    aclrtMemcpy(logitsHost, logitsBytes, logitsDev, logitsBytes, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile("../output/output_logits.bin", logitsHost, logitsBytes);

    aclrtFree(logitsDev);
    aclrtFree(wDev);
    aclrtFree(xDev);
    aclrtFreeHost(logitsHost);
    aclrtFreeHost(wHost);
    aclrtFreeHost(xHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<float> golden(logitsBytes / sizeof(float));
    std::vector<float> devFinal(logitsBytes / sizeof(float));
    ReadFile("../output/golden_logits.bin", logitsBytes, golden.data(),   logitsBytes);
    ReadFile("../output/output_logits.bin", logitsBytes, devFinal.data(), logitsBytes);

    // Tolerance: FP16 inputs (-4..4 integers, exactly representable) summed
    // over kH=64 products. Values should match to ~0.01f.
    bool ok = ResultCmp(golden, devFinal, 0.01f);
    if (ok) {
        printf("test data success\n");
    } else {
        printf("test data failed\n");
    }
    return ok ? 0 : 1;
}
