/**
 * main.cpp — host driver for DeepSeek-V4 gate_logits (FP32 × FP32 → FP32).
 *
 * I/O contract (all little-endian, contiguous, no header):
 *   ./input/input_x.bin       T * DIM        float32
 *   ./input/input_w_gate.bin  N_ROUTED * DIM float32
 *   ./output/golden_scores.bin T * N_ROUTED  float32 (numpy reference)
 *   ./output/output_scores.bin T * N_ROUTED  float32 (kernel output)
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

extern "C" void launch_gate_logits(uint8_t *scores, uint8_t *X, uint8_t *W_gate,
                                   uint64_t T, uint64_t N, uint64_t K, void *stream);

int main()
{
    constexpr int kT       = kDsmoeT;
    constexpr int kDim     = kDsmoeDim;
    constexpr int kNRouted = kDsmoeNRouted;

    constexpr size_t f32Bytes = 4;

    size_t xBytes     = static_cast<size_t>(kT)       * kDim     * f32Bytes;
    size_t wBytes     = static_cast<size_t>(kNRouted) * kDim     * f32Bytes;
    size_t scoreBytes = static_cast<size_t>(kT)       * kNRouted * f32Bytes;

    printf("[gate_logits] T=%d DIM=%d N_ROUTED=%d\n", kT, kDim, kNRouted);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint8_t *xHost = nullptr, *wHost = nullptr, *scoreHost = nullptr;
    uint8_t *xDev  = nullptr, *wDev  = nullptr, *scoreDev  = nullptr;

    aclrtMallocHost((void **)&xHost,     xBytes);
    aclrtMallocHost((void **)&wHost,     wBytes);
    aclrtMallocHost((void **)&scoreHost, scoreBytes);

    aclrtMalloc((void **)&xDev,     xBytes,     ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&wDev,     wBytes,     ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&scoreDev, scoreBytes, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile("./input/input_x.bin",      xBytes, xHost, xBytes);
    ReadFile("./input/input_w_gate.bin", wBytes, wHost, wBytes);

    aclrtMemcpy(xDev, xBytes, xHost, xBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(wDev, wBytes, wHost, wBytes, ACL_MEMCPY_HOST_TO_DEVICE);

    (void)PtoTiming::TimeKernelCallUs("gate_logits", stream, [&]() {
        launch_gate_logits(scoreDev, xDev, wDev,
                           static_cast<uint64_t>(kT),
                           static_cast<uint64_t>(kNRouted),
                           static_cast<uint64_t>(kDim),
                           stream);
    });

    aclrtMemcpy(scoreHost, scoreBytes, scoreDev, scoreBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    WriteFile("./output/output_scores.bin", scoreHost, scoreBytes);

    aclrtFree(scoreDev);
    aclrtFree(wDev);
    aclrtFree(xDev);
    aclrtFreeHost(scoreHost);
    aclrtFreeHost(wHost);
    aclrtFreeHost(xHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<float> golden(scoreBytes / sizeof(float));
    std::vector<float> dev   (scoreBytes / sizeof(float));
    ReadFile("./output/golden_scores.bin", scoreBytes, golden.data(), scoreBytes);
    ReadFile("./output/output_scores.bin", scoreBytes, dev.data(),    scoreBytes);

    // FP32 × FP32 → FP32: pure rounding error from FMA reordering; tighter
    // tolerance than the BF16/FP16 router_matmul reference.
    bool ok = ResultCmp(golden, dev, 1e-3f);
    if (ok) {
        printf("test data success\n");
    } else {
        printf("test data failed\n");
    }
    return ok ? 0 : 1;
}
