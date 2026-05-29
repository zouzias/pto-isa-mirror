/**
 * main.cpp - host driver for Indexer.weights_proj
 *   weights = (X @ W_wproj^T) * (softmax_scale * n_heads^-0.5)
 *
 * I/O contract (all little-endian, contiguous, no header):
 *   ./input/input_x.bin       M*K   bfloat16
 *   ./input/input_wproj.bin   N*K   bfloat16
 *   ./output/golden_w.bin     M*N   float32  (post-scale)
 *   ./output/output_w.bin     M*N   float32  (kernel output, post-scale)
 *
 * M = B*S, K = DIM, N = N_HEADS (see ../README.md).
 */

#include "test_common.h"
#include "acl/acl.h"
#include "../../../../kernel_timing.h"
#include "generated_cases.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

using namespace std;
using namespace PtoTestCommon;

extern "C" void launch_weights_proj_gemm(uint8_t *out, uint8_t *X, uint8_t *W_wproj,
                                         float scale,
                                         uint64_t M, uint64_t N, uint64_t K, void *stream);

int main()
{
    constexpr int kB        = kIdxB;
    constexpr int kS        = kIdxS;
    constexpr int kDim      = kIdxDim;
    constexpr int kNHeads   = kIdxNHeads;
    constexpr int kHeadDim  = kIdxHeadDim;
    constexpr int kM = kB * kS;
    constexpr int kK = kDim;
    constexpr int kN = kNHeads;

    // softmax_scale * n_heads^-0.5  (model.py:395, model.py:419).
    // softmax_scale = index_head_dim ** -0.5
    const float kSoftmaxScale = 1.0f / std::sqrt(static_cast<float>(kHeadDim));
    const float kHeadsScale   = 1.0f / std::sqrt(static_cast<float>(kNHeads));
    const float kScale        = kSoftmaxScale * kHeadsScale;

    constexpr size_t bf16Bytes  = 2;
    constexpr size_t floatBytes = 4;

    size_t xBytes = static_cast<size_t>(kM) * kK * bf16Bytes;
    size_t wBytes = static_cast<size_t>(kN) * kK * bf16Bytes;
    size_t oBytes = static_cast<size_t>(kM) * kN * floatBytes;

    printf("[weights_proj_gemm] B=%d S=%d Dim=%d NHeads=%d HeadDim=%d "
           "-> M=%d N=%d K=%d  scale=%.6f\n",
           kB, kS, kDim, kNHeads, kHeadDim, kM, kN, kK, kScale);

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

    ReadFile("./input/input_x.bin",     xBytes, xHost, xBytes);
    ReadFile("./input/input_wproj.bin", wBytes, wHost, wBytes);

    aclrtMemcpy(xDev, xBytes, xHost, xBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(wDev, wBytes, wHost, wBytes, ACL_MEMCPY_HOST_TO_DEVICE);

    (void)PtoTiming::TimeKernelCallUs("weights_proj_gemm", stream, [&]() {
        launch_weights_proj_gemm(oDev, xDev, wDev, kScale,
                                 static_cast<uint64_t>(kM),
                                 static_cast<uint64_t>(kN),
                                 static_cast<uint64_t>(kK),
                                 stream);
    });

    aclrtMemcpy(oHost, oBytes, oDev, oBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    WriteFile("./output/output_w.bin", oHost, oBytes);

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
    ReadFile("./output/golden_w.bin", oBytes, golden.data(),   oBytes);
    ReadFile("./output/output_w.bin", oBytes, devFinal.data(), oBytes);

    bool ok = ResultCmp(golden, devFinal, 1e-2f);
    if (ok) {
        printf("test data success\n");
    } else {
        printf("test data failed\n");
    }
    return ok ? 0 : 1;
}
