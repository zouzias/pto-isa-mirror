/**
 * main.cpp — host driver for DeepSeek-V4 gate_hash_routing.
 *
 * I/O contract (all little-endian, contiguous, no header):
 *   ./input/input_ids.bin       T                   int32
 *   ./input/input_tid2eid.bin   VOCAB * N_ACTIVATED int32
 *   ./input/input_scores.bin    T * N_ROUTED        float32  (un-biased original_scores)
 *   ./output/golden_indices.bin T * N_ACTIVATED     int32
 *   ./output/golden_weights.bin T * N_ACTIVATED     float32
 *   ./output/output_indices.bin (kernel-emitted)
 *   ./output/output_weights.bin (kernel-emitted)
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

extern "C" void launch_gate_hash_routing(uint8_t *indices_out, uint8_t *weights_out,
                                         uint8_t *input_ids, uint8_t *tid2eid,
                                         uint8_t *original_scores,
                                         uint64_t T, uint64_t N, uint64_t K, uint64_t V,
                                         void *stream);

int main()
{
    constexpr int kT          = kDsmoeT;
    constexpr int kNRouted    = kDsmoeNRouted;
    constexpr int kNActivated = kDsmoeNActivated;
    constexpr int kVocab      = kDsmoeVocab;

    constexpr size_t f32Bytes = 4;
    constexpr size_t i32Bytes = 4;

    size_t idsBytes     = static_cast<size_t>(kT)                              * i32Bytes;
    size_t tabBytes     = static_cast<size_t>(kVocab) * kNActivated            * i32Bytes;
    size_t scoresBytes  = static_cast<size_t>(kT)     * kNRouted               * f32Bytes;
    size_t indicesBytes = static_cast<size_t>(kT)     * kNActivated            * i32Bytes;
    size_t weightsBytes = static_cast<size_t>(kT)     * kNActivated            * f32Bytes;

    printf("[gate_hash_routing] T=%d N_ROUTED=%d N_ACTIVATED=%d VOCAB=%d\n",
           kT, kNRouted, kNActivated, kVocab);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint8_t *idsHost = nullptr, *tabHost = nullptr, *scoresHost = nullptr;
    uint8_t *indicesHost = nullptr, *weightsHost = nullptr;
    uint8_t *idsDev = nullptr, *tabDev = nullptr, *scoresDev = nullptr;
    uint8_t *indicesDev = nullptr, *weightsDev = nullptr;

    aclrtMallocHost((void **)&idsHost,     idsBytes);
    aclrtMallocHost((void **)&tabHost,     tabBytes);
    aclrtMallocHost((void **)&scoresHost,  scoresBytes);
    aclrtMallocHost((void **)&indicesHost, indicesBytes);
    aclrtMallocHost((void **)&weightsHost, weightsBytes);

    aclrtMalloc((void **)&idsDev,     idsBytes,     ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&tabDev,     tabBytes,     ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&scoresDev,  scoresBytes,  ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&indicesDev, indicesBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&weightsDev, weightsBytes, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile("./input/input_ids.bin",     idsBytes,    idsHost,    idsBytes);
    ReadFile("./input/input_tid2eid.bin", tabBytes,    tabHost,    tabBytes);
    ReadFile("./input/input_scores.bin",  scoresBytes, scoresHost, scoresBytes);

    aclrtMemcpy(idsDev,    idsBytes,    idsHost,    idsBytes,    ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(tabDev,    tabBytes,    tabHost,    tabBytes,    ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(scoresDev, scoresBytes, scoresHost, scoresBytes, ACL_MEMCPY_HOST_TO_DEVICE);

    (void)PtoTiming::TimeKernelCallUs("gate_hash_routing", stream, [&]() {
        launch_gate_hash_routing(indicesDev, weightsDev, idsDev, tabDev, scoresDev,
                                 static_cast<uint64_t>(kT),
                                 static_cast<uint64_t>(kNRouted),
                                 static_cast<uint64_t>(kNActivated),
                                 static_cast<uint64_t>(kVocab),
                                 stream);
    });

    aclrtMemcpy(indicesHost, indicesBytes, indicesDev, indicesBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(weightsHost, weightsBytes, weightsDev, weightsBytes, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile("./output/output_indices.bin", indicesHost, indicesBytes);
    WriteFile("./output/output_weights.bin", weightsHost, weightsBytes);

    aclrtFree(weightsDev);
    aclrtFree(indicesDev);
    aclrtFree(scoresDev);
    aclrtFree(tabDev);
    aclrtFree(idsDev);
    aclrtFreeHost(weightsHost);
    aclrtFreeHost(indicesHost);
    aclrtFreeHost(scoresHost);
    aclrtFreeHost(tabHost);
    aclrtFreeHost(idsHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<int32_t> goldIdx(indicesBytes / sizeof(int32_t));
    std::vector<int32_t> outIdx (indicesBytes / sizeof(int32_t));
    ReadFile("./output/golden_indices.bin", indicesBytes, goldIdx.data(), indicesBytes);
    ReadFile("./output/output_indices.bin", indicesBytes, outIdx.data(),  indicesBytes);
    bool idxOk = (goldIdx == outIdx);
    printf("indices : %s\n", idxOk ? "success" : "FAILED");

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
