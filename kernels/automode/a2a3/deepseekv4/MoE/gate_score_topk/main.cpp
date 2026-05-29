/**
 * main.cpp — host driver for DeepSeek-V4 gate_score_topk.
 *
 * I/O contract (all little-endian, contiguous, no header):
 *   ./input/input_scores.bin       T * N_ROUTED      float32  (un-biased)
 *   ./input/input_bias.bin         N_ROUTED          float32  (zero if disabled)
 *   ./output/golden_indices.bin    T * N_ACTIVATED   int32
 *   ./output/golden_weights.bin    T * N_ACTIVATED   float32
 *   ./output/output_indices.bin    (kernel-emitted; same shape)
 *   ./output/output_weights.bin    (kernel-emitted; same shape)
 */

#include "test_common.h"
#include "acl/acl.h"
#include "../../../kernel_timing.h"
#include "generated_cases.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

using namespace std;
using namespace PtoTestCommon;

extern "C" void launch_gate_score_topk(uint8_t *indices_out, uint8_t *weights_out,
                                       uint8_t *original_scores, uint8_t *bias,
                                       uint64_t T, uint64_t N, uint64_t K, void *stream);

int main()
{
    constexpr int kT          = kDsmoeT;
    constexpr int kNRouted    = kDsmoeNRouted;
    constexpr int kNActivated = kDsmoeNActivated;

    constexpr size_t f32Bytes = 4;
    constexpr size_t i32Bytes = 4;

    size_t scoresBytes  = static_cast<size_t>(kT) * kNRouted    * f32Bytes;
    size_t biasBytes    = static_cast<size_t>(kNRouted)         * f32Bytes;
    size_t indicesBytes = static_cast<size_t>(kT) * kNActivated * i32Bytes;
    size_t weightsBytes = static_cast<size_t>(kT) * kNActivated * f32Bytes;

    printf("[gate_score_topk] T=%d N_ROUTED=%d N_ACTIVATED=%d\n",
           kT, kNRouted, kNActivated);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint8_t *scoresHost = nullptr, *biasHost = nullptr;
    uint8_t *indicesHost = nullptr, *weightsHost = nullptr;
    uint8_t *scoresDev = nullptr, *biasDev = nullptr;
    uint8_t *indicesDev = nullptr, *weightsDev = nullptr;

    aclrtMallocHost((void **)&scoresHost,  scoresBytes);
    aclrtMallocHost((void **)&biasHost,    biasBytes);
    aclrtMallocHost((void **)&indicesHost, indicesBytes);
    aclrtMallocHost((void **)&weightsHost, weightsBytes);

    aclrtMalloc((void **)&scoresDev,  scoresBytes,  ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&biasDev,    biasBytes,    ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&indicesDev, indicesBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&weightsDev, weightsBytes, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile("./input/input_scores.bin", scoresBytes, scoresHost, scoresBytes);
    ReadFile("./input/input_bias.bin",   biasBytes,   biasHost,   biasBytes);

    aclrtMemcpy(scoresDev, scoresBytes, scoresHost, scoresBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(biasDev,   biasBytes,   biasHost,   biasBytes,   ACL_MEMCPY_HOST_TO_DEVICE);

    (void)PtoTiming::TimeKernelCallUs("gate_score_topk", stream, [&]() {
        launch_gate_score_topk(indicesDev, weightsDev, scoresDev, biasDev,
                               static_cast<uint64_t>(kT),
                               static_cast<uint64_t>(kNRouted),
                               static_cast<uint64_t>(kNActivated),
                               stream);
    });

    aclrtMemcpy(indicesHost, indicesBytes, indicesDev, indicesBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(weightsHost, weightsBytes, weightsDev, weightsBytes, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile("./output/output_indices.bin", indicesHost, indicesBytes);
    WriteFile("./output/output_weights.bin", weightsHost, weightsBytes);

    aclrtFree(weightsDev);
    aclrtFree(indicesDev);
    aclrtFree(biasDev);
    aclrtFree(scoresDev);
    aclrtFreeHost(weightsHost);
    aclrtFreeHost(indicesHost);
    aclrtFreeHost(biasHost);
    aclrtFreeHost(scoresHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    // Indices: exact equality.
    std::vector<int32_t> goldIdx (indicesBytes / sizeof(int32_t));
    std::vector<int32_t> outIdx  (indicesBytes / sizeof(int32_t));
    ReadFile("./output/golden_indices.bin", indicesBytes, goldIdx.data(), indicesBytes);
    ReadFile("./output/output_indices.bin", indicesBytes, outIdx.data(),  indicesBytes);
    bool idxOk = (goldIdx == outIdx);
    printf("indices : %s\n", idxOk ? "success" : "FAILED");

    // Weights: tolerance compare.
    std::vector<float> goldW(weightsBytes / sizeof(float));
    std::vector<float> outW (weightsBytes / sizeof(float));
    ReadFile("./output/golden_weights.bin", weightsBytes, goldW.data(), weightsBytes);
    ReadFile("./output/output_weights.bin", weightsBytes, outW.data(),  weightsBytes);
    bool wOk = ResultCmp(goldW, outW, 1e-3f);
    printf("weights : %s\n", wOk ? "success" : "FAILED");

    bool ok = idxOk && wOk;
    if (ok) {
        printf("test data success\n");
    } else {
        printf("test data failed\n");
    }
    return ok ? 0 : 1;
}
