/**
 * main.cpp - host driver for hc_pre_reduce.
 *
 *   y[b,s,d] = sum_h pre[b,s,h] * x[b,s,h,d]
 *
 * I/O contract (all little-endian, contiguous, no header):
 *   ./input/input_pre.bin       B*S*HC_MULT          float32
 *   ./input/input_x.bin         B*S*HC_MULT*DIM      float32
 *   ./output/golden_y.bin       B*S*DIM              float32
 *   ./output/output_y.bin       B*S*DIM              float32
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

extern "C" void launch_hc_pre_reduce(uint8_t *y, uint8_t *pre, uint8_t *x,
                                     void *stream);

int main()
{
    constexpr int kB   = kHcB;
    constexpr int kS   = kHcS;
    constexpr int kDim = kHcDim;
    constexpr int kHc  = kHcMult;

    constexpr size_t floatBytes = 4;
    size_t preBytes = static_cast<size_t>(kB) * kS * kHc * floatBytes;
    size_t xBytes   = static_cast<size_t>(kB) * kS * kHc * kDim * floatBytes;
    size_t yBytes   = static_cast<size_t>(kB) * kS * kDim * floatBytes;

    printf("[hc_pre_reduce] B=%d S=%d HC_MULT=%d DIM=%d\n", kB, kS, kHc, kDim);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint8_t *preHost = nullptr, *xHost = nullptr, *yHost = nullptr;
    uint8_t *preDev  = nullptr, *xDev  = nullptr, *yDev  = nullptr;

    aclrtMallocHost((void **)&preHost, preBytes);
    aclrtMallocHost((void **)&xHost,   xBytes);
    aclrtMallocHost((void **)&yHost,   yBytes);

    aclrtMalloc((void **)&preDev, preBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&xDev,   xBytes,   ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&yDev,   yBytes,   ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile("./input/input_pre.bin", preBytes, preHost, preBytes);
    ReadFile("./input/input_x.bin",   xBytes,   xHost,   xBytes);

    aclrtMemcpy(preDev, preBytes, preHost, preBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(xDev,   xBytes,   xHost,   xBytes,   ACL_MEMCPY_HOST_TO_DEVICE);

    (void)PtoTiming::TimeKernelCallUs("hc_pre_reduce", stream, [&]() {
        launch_hc_pre_reduce(yDev, preDev, xDev, stream);
    });

    aclrtMemcpy(yHost, yBytes, yDev, yBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    WriteFile("./output/output_y.bin", yHost, yBytes);

    aclrtFree(yDev);
    aclrtFree(xDev);
    aclrtFree(preDev);
    aclrtFreeHost(yHost);
    aclrtFreeHost(xHost);
    aclrtFreeHost(preHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<float> golden(yBytes / sizeof(float));
    std::vector<float> devFinal(yBytes / sizeof(float));
    ReadFile("./output/golden_y.bin", yBytes, golden.data(),   yBytes);
    ReadFile("./output/output_y.bin", yBytes, devFinal.data(), yBytes);

    // Reduction over HC_MULT=4 FP32 products per element; small drift.
    bool ok = ResultCmp(golden, devFinal, 1e-4f);
    if (ok) {
        printf("test data success\n");
    } else {
        printf("test data failed\n");
    }
    return ok ? 0 : 1;
}
