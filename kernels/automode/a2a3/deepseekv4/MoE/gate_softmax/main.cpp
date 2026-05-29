/**
 * main.cpp — host driver for DeepSeek-V4 gate_softmax (FP32 elementwise).
 *
 * I/O contract (all little-endian, contiguous, no header):
 *   ./input/input_scores.bin   T * N_ROUTED float32
 *   ./output/golden_scores.bin T * N_ROUTED float32 (numpy reference)
 *   ./output/output_scores.bin T * N_ROUTED float32 (kernel output)
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

extern "C" void launch_gate_softmax(uint8_t *scores_out, uint8_t *scores_in,
                                    uint64_t T, uint64_t N, void *stream);

int main()
{
    constexpr int kT       = kDsmoeT;
    constexpr int kNRouted = kDsmoeNRouted;

    constexpr size_t f32Bytes = 4;
    size_t bytes = static_cast<size_t>(kT) * kNRouted * f32Bytes;

    printf("[gate_softmax] T=%d N_ROUTED=%d\n", kT, kNRouted);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint8_t *inHost = nullptr, *outHost = nullptr;
    uint8_t *inDev  = nullptr, *outDev  = nullptr;

    aclrtMallocHost((void **)&inHost,  bytes);
    aclrtMallocHost((void **)&outHost, bytes);

    aclrtMalloc((void **)&inDev,  bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&outDev, bytes, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile("./input/input_scores.bin", bytes, inHost, bytes);
    aclrtMemcpy(inDev, bytes, inHost, bytes, ACL_MEMCPY_HOST_TO_DEVICE);

    (void)PtoTiming::TimeKernelCallUs("gate_softmax", stream, [&]() {
        launch_gate_softmax(outDev, inDev,
                            static_cast<uint64_t>(kT),
                            static_cast<uint64_t>(kNRouted),
                            stream);
    });

    aclrtMemcpy(outHost, bytes, outDev, bytes, ACL_MEMCPY_DEVICE_TO_HOST);
    WriteFile("./output/output_scores.bin", outHost, bytes);

    aclrtFree(outDev);
    aclrtFree(inDev);
    aclrtFreeHost(outHost);
    aclrtFreeHost(inHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<float> golden(bytes / sizeof(float));
    std::vector<float> dev   (bytes / sizeof(float));
    ReadFile("./output/golden_scores.bin", bytes, golden.data(), bytes);
    ReadFile("./output/output_scores.bin", bytes, dev.data(),    bytes);

    bool ok = ResultCmp(golden, dev, 1e-3f);
    if (ok) {
        printf("test data success\n");
    } else {
        printf("test data failed\n");
    }
    return ok ? 0 : 1;
}
