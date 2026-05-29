/**
 * main.cpp — host driver for sparse_attn.online_softmax (one pipelined-block iter).
 *
 * I/O contract (all little-endian, contiguous, no header):
 *   ./input/input_acc_s.bin           H*BLOCK   float32  (from qk_matmul)
 *   ./input/input_scores_max.bin      H         float32  (running max before)
 *   ./input/input_sum_exp.bin         H         float32  (running denom before)
 *   ./input/input_topk_idxs.bin       BLOCK     int32    (mask slice for t_block)
 *   ./output/golden_acc_s.bin         H*BLOCK   float32
 *   ./output/golden_acc_s_cast.bin    H*BLOCK   bfloat16
 *   ./output/golden_scores_max.bin    H         float32
 *   ./output/golden_sum_exp.bin       H         float32
 *   ./output/golden_scores_scale.bin  H         float32
 *   ./output/output_acc_s.bin         H*BLOCK   float32
 *   ./output/output_acc_s_cast.bin    H*BLOCK   bfloat16
 *   ./output/output_scores_max.bin    H         float32
 *   ./output/output_sum_exp.bin       H         float32
 *   ./output/output_scores_scale.bin  H         float32
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

extern "C" void launch_online_softmax(uint8_t *acc_s, uint8_t *acc_s_cast,
                                      uint8_t *scores_max, uint8_t *sum_exp,
                                      uint8_t *scores_scale, uint8_t *topk_idxs,
                                      uint32_t topk_len, uint32_t t_block,
                                      void *stream);

int main()
{
    constexpr int kH     = kCsaH;
    constexpr int kBlock = kCsaBlock;
    constexpr int kS     = kCsaS;
    constexpr uint32_t kTblock = 0;

    constexpr size_t bf16Bytes  = 2;
    constexpr size_t floatBytes = 4;
    constexpr size_t i32Bytes   = 4;

    size_t accBytes      = static_cast<size_t>(kH) * kBlock * floatBytes;
    size_t castBytes     = static_cast<size_t>(kH) * kBlock * bf16Bytes;
    size_t maxBytes      = static_cast<size_t>(kH)          * floatBytes;
    size_t sumBytes      = maxBytes;
    size_t scaleBytes    = maxBytes;
    size_t idxBytes      = static_cast<size_t>(kBlock)       * i32Bytes;

    printf("[online_softmax] H=%d BLOCK=%d  t_block=%u\n", kH, kBlock, kTblock);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint8_t *accHost = nullptr, *castHost = nullptr;
    uint8_t *maxHost = nullptr, *sumHost = nullptr, *scaleHost = nullptr;
    uint8_t *idxHost = nullptr;
    uint8_t *accDev = nullptr, *castDev = nullptr;
    uint8_t *maxDev = nullptr, *sumDev = nullptr, *scaleDev = nullptr;
    uint8_t *idxDev = nullptr;

    aclrtMallocHost((void **)&accHost,   accBytes);
    aclrtMallocHost((void **)&castHost,  castBytes);
    aclrtMallocHost((void **)&maxHost,   maxBytes);
    aclrtMallocHost((void **)&sumHost,   sumBytes);
    aclrtMallocHost((void **)&scaleHost, scaleBytes);
    aclrtMallocHost((void **)&idxHost,   idxBytes);

    aclrtMalloc((void **)&accDev,   accBytes,   ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&castDev,  castBytes,  ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&maxDev,   maxBytes,   ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&sumDev,   sumBytes,   ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&scaleDev, scaleBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&idxDev,   idxBytes,   ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile("./input/input_acc_s.bin",      accBytes, accHost, accBytes);
    ReadFile("./input/input_scores_max.bin", maxBytes, maxHost, maxBytes);
    ReadFile("./input/input_sum_exp.bin",    sumBytes, sumHost, sumBytes);
    ReadFile("./input/input_topk_idxs.bin",  idxBytes, idxHost, idxBytes);

    aclrtMemcpy(accDev, accBytes, accHost, accBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(maxDev, maxBytes, maxHost, maxBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(sumDev, sumBytes, sumHost, sumBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(idxDev, idxBytes, idxHost, idxBytes, ACL_MEMCPY_HOST_TO_DEVICE);

    (void)PtoTiming::TimeKernelCallUs("online_softmax", stream, [&]() {
        launch_online_softmax(accDev, castDev, maxDev, sumDev, scaleDev, idxDev,
                              static_cast<uint32_t>(kS), kTblock, stream);
    });

    aclrtMemcpy(accHost,   accBytes,   accDev,   accBytes,   ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(castHost,  castBytes,  castDev,  castBytes,  ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(maxHost,   maxBytes,   maxDev,   maxBytes,   ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(sumHost,   sumBytes,   sumDev,   sumBytes,   ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(scaleHost, scaleBytes, scaleDev, scaleBytes, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile("./output/output_acc_s.bin",        accHost,   accBytes);
    WriteFile("./output/output_acc_s_cast.bin",   castHost,  castBytes);
    WriteFile("./output/output_scores_max.bin",   maxHost,   maxBytes);
    WriteFile("./output/output_sum_exp.bin",      sumHost,   sumBytes);
    WriteFile("./output/output_scores_scale.bin", scaleHost, scaleBytes);

    aclrtFree(idxDev); aclrtFree(scaleDev); aclrtFree(sumDev); aclrtFree(maxDev);
    aclrtFree(castDev); aclrtFree(accDev);
    aclrtFreeHost(idxHost); aclrtFreeHost(scaleHost); aclrtFreeHost(sumHost);
    aclrtFreeHost(maxHost); aclrtFreeHost(castHost); aclrtFreeHost(accHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    // Compare FP32 outputs with tolerance; the BF16 cast we compare bit-exact
    // against the python BF16 rounding.
    auto cmp_fp32 = [&](const char *gold_path, const char *out_path, size_t bytes) {
        std::vector<float> g(bytes / sizeof(float));
        std::vector<float> d(bytes / sizeof(float));
        ReadFile(gold_path, bytes, g.data(), bytes);
        ReadFile(out_path,  bytes, d.data(), bytes);
        return ResultCmp(g, d, 1e-2f);
    };
    auto cmp_bf16 = [&](const char *gold_path, const char *out_path, size_t bytes) {
        std::vector<uint16_t> g(bytes / sizeof(uint16_t));
        std::vector<uint16_t> d(bytes / sizeof(uint16_t));
        ReadFile(gold_path, bytes, g.data(), bytes);
        ReadFile(out_path,  bytes, d.data(), bytes);
        return g == d;
    };

    bool ok =
        cmp_fp32("./output/golden_acc_s.bin",        "./output/output_acc_s.bin",        accBytes) &&
        cmp_bf16("./output/golden_acc_s_cast.bin",   "./output/output_acc_s_cast.bin",   castBytes) &&
        cmp_fp32("./output/golden_scores_max.bin",   "./output/output_scores_max.bin",   maxBytes) &&
        cmp_fp32("./output/golden_sum_exp.bin",      "./output/output_sum_exp.bin",      sumBytes) &&
        cmp_fp32("./output/golden_scores_scale.bin", "./output/output_scores_scale.bin", scaleBytes);

    if (ok) {
        printf("test data success\n");
    } else {
        printf("test data failed\n");
    }
    return ok ? 0 : 1;
}
