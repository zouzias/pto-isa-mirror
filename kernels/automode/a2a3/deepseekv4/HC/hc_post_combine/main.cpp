/**
 * main.cpp - host driver for hc_post_combine.
 *
 *   y[b,s,h,d] = post[b,s,h] * x[b,s,d]
 *              + sum_{h2} comb[b,s,h2,h] * residual[b,s,h2,d]
 *
 * I/O contract (all little-endian, contiguous, no header):
 *   ./input/input_x.bin         B*S*DIM                  float32
 *   ./input/input_residual.bin  B*S*HC_MULT*DIM          float32
 *   ./input/input_post.bin      B*S*HC_MULT              float32
 *   ./input/input_comb.bin      B*S*HC_MULT*HC_MULT      float32
 *   ./output/golden_y.bin       B*S*HC_MULT*DIM          float32
 *   ./output/output_y.bin       B*S*HC_MULT*DIM          float32
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

extern "C" void launch_hc_post_combine(uint8_t *y, uint8_t *x, uint8_t *residual,
                                       uint8_t *post, uint8_t *comb, void *stream);

int main()
{
    constexpr int kB   = kHcB;
    constexpr int kS   = kHcS;
    constexpr int kDim = kHcDim;
    constexpr int kHc  = kHcMult;

    constexpr size_t floatBytes = 4;
    size_t xBytes    = static_cast<size_t>(kB) * kS * kDim * floatBytes;
    size_t resBytes  = static_cast<size_t>(kB) * kS * kHc * kDim * floatBytes;
    size_t postBytes = static_cast<size_t>(kB) * kS * kHc * floatBytes;
    size_t combBytes = static_cast<size_t>(kB) * kS * kHc * kHc * floatBytes;
    size_t yBytes    = static_cast<size_t>(kB) * kS * kHc * kDim * floatBytes;

    printf("[hc_post_combine] B=%d S=%d HC_MULT=%d DIM=%d\n",
           kB, kS, kHc, kDim);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint8_t *xHost = nullptr, *resHost = nullptr, *postHost = nullptr,
            *combHost = nullptr, *yHost = nullptr;
    uint8_t *xDev  = nullptr, *resDev  = nullptr, *postDev  = nullptr,
            *combDev  = nullptr, *yDev  = nullptr;

    aclrtMallocHost((void **)&xHost,    xBytes);
    aclrtMallocHost((void **)&resHost,  resBytes);
    aclrtMallocHost((void **)&postHost, postBytes);
    aclrtMallocHost((void **)&combHost, combBytes);
    aclrtMallocHost((void **)&yHost,    yBytes);

    aclrtMalloc((void **)&xDev,    xBytes,    ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&resDev,  resBytes,  ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&postDev, postBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&combDev, combBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&yDev,    yBytes,    ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile("./input/input_x.bin",        xBytes,    xHost,    xBytes);
    ReadFile("./input/input_residual.bin", resBytes,  resHost,  resBytes);
    ReadFile("./input/input_post.bin",     postBytes, postHost, postBytes);
    ReadFile("./input/input_comb.bin",     combBytes, combHost, combBytes);

    aclrtMemcpy(xDev,    xBytes,    xHost,    xBytes,    ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(resDev,  resBytes,  resHost,  resBytes,  ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(postDev, postBytes, postHost, postBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(combDev, combBytes, combHost, combBytes, ACL_MEMCPY_HOST_TO_DEVICE);

    (void)PtoTiming::TimeKernelCallUs("hc_post_combine", stream, [&]() {
        launch_hc_post_combine(yDev, xDev, resDev, postDev, combDev, stream);
    });

    aclrtMemcpy(yHost, yBytes, yDev, yBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    WriteFile("./output/output_y.bin", yHost, yBytes);

    aclrtFree(yDev);
    aclrtFree(combDev);
    aclrtFree(postDev);
    aclrtFree(resDev);
    aclrtFree(xDev);
    aclrtFreeHost(yHost);
    aclrtFreeHost(combHost);
    aclrtFreeHost(postHost);
    aclrtFreeHost(resHost);
    aclrtFreeHost(xHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<float> golden(yBytes / sizeof(float));
    std::vector<float> devFinal(yBytes / sizeof(float));
    ReadFile("./output/golden_y.bin", yBytes, golden.data(),   yBytes);
    ReadFile("./output/output_y.bin", yBytes, devFinal.data(), yBytes);

    // Per-element sum is post*x + sum over HC_MULT=4 comb*residual products;
    // ~5 FP32 products per output element -> tight tolerance ok.
    bool ok = ResultCmp(golden, devFinal, 1e-4f);
    if (ok) {
        printf("test data success\n");
    } else {
        printf("test data failed\n");
    }
    return ok ? 0 : 1;
}
