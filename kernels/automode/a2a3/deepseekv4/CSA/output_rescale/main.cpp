/**
 * main.cpp — host driver for sparse_attn.output_rescale (post-loop tail).
 *
 * I/O contract (all little-endian, contiguous, no header):
 *   ./input/input_acc_o.bin      H*D   float32   (post-loop accumulator)
 *   ./input/input_scores_max.bin H     float32   (final running max)
 *   ./input/input_sum_exp.bin    H     float32   (running denom, pre-sink)
 *   ./input/input_attn_sink.bin  H     float32   (per-head learnable bias)
 *   ./output/golden_o.bin        H*D   bfloat16
 *   ./output/output_o.bin        H*D   bfloat16
 *
 * sum_exp is also written back (sink-adjusted), but we don't compare it —
 * the only consumer is the BF16 `o` tile.
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

extern "C" void launch_output_rescale(uint8_t *o, uint8_t *acc_o,
                                      uint8_t *scores_max, uint8_t *sum_exp,
                                      uint8_t *attn_sink, void *stream);

int main()
{
    constexpr int kH = kCsaH;
    constexpr int kD = kCsaD;

    constexpr size_t bf16Bytes  = 2;
    constexpr size_t floatBytes = 4;

    size_t accBytes  = static_cast<size_t>(kH) * kD * floatBytes;
    size_t vecBytes  = static_cast<size_t>(kH)      * floatBytes;
    size_t oBytes    = static_cast<size_t>(kH) * kD * bf16Bytes;

    printf("[output_rescale] H=%d D=%d\n", kH, kD);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint8_t *accHost = nullptr, *maxHost = nullptr, *sumHost = nullptr;
    uint8_t *sinkHost = nullptr, *oHost = nullptr;
    uint8_t *accDev  = nullptr, *maxDev  = nullptr, *sumDev  = nullptr;
    uint8_t *sinkDev = nullptr, *oDev   = nullptr;

    aclrtMallocHost((void **)&accHost,  accBytes);
    aclrtMallocHost((void **)&maxHost,  vecBytes);
    aclrtMallocHost((void **)&sumHost,  vecBytes);
    aclrtMallocHost((void **)&sinkHost, vecBytes);
    aclrtMallocHost((void **)&oHost,    oBytes);

    aclrtMalloc((void **)&accDev,  accBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&maxDev,  vecBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&sumDev,  vecBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&sinkDev, vecBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&oDev,    oBytes,   ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile("./input/input_acc_o.bin",      accBytes, accHost,  accBytes);
    ReadFile("./input/input_scores_max.bin", vecBytes, maxHost,  vecBytes);
    ReadFile("./input/input_sum_exp.bin",    vecBytes, sumHost,  vecBytes);
    ReadFile("./input/input_attn_sink.bin",  vecBytes, sinkHost, vecBytes);

    aclrtMemcpy(accDev,  accBytes, accHost,  accBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(maxDev,  vecBytes, maxHost,  vecBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(sumDev,  vecBytes, sumHost,  vecBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(sinkDev, vecBytes, sinkHost, vecBytes, ACL_MEMCPY_HOST_TO_DEVICE);

    (void)PtoTiming::TimeKernelCallUs("output_rescale", stream, [&]() {
        launch_output_rescale(oDev, accDev, maxDev, sumDev, sinkDev, stream);
    });

    aclrtMemcpy(oHost, oBytes, oDev, oBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    WriteFile("./output/output_o.bin", oHost, oBytes);

    aclrtFree(oDev); aclrtFree(sinkDev); aclrtFree(sumDev);
    aclrtFree(maxDev); aclrtFree(accDev);
    aclrtFreeHost(oHost); aclrtFreeHost(sinkHost); aclrtFreeHost(sumHost);
    aclrtFreeHost(maxHost); aclrtFreeHost(accHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    // BF16 compare: bit-exact uint16 against the python BF16 rounding.
    std::vector<uint16_t> golden(oBytes / sizeof(uint16_t));
    std::vector<uint16_t> dev(oBytes    / sizeof(uint16_t));
    ReadFile("./output/golden_o.bin", oBytes, golden.data(), oBytes);
    ReadFile("./output/output_o.bin", oBytes, dev.data(),    oBytes);

    bool ok = (golden == dev);
    if (ok) {
        printf("test data success\n");
    } else {
        printf("test data failed\n");
    }
    return ok ? 0 : 1;
}
