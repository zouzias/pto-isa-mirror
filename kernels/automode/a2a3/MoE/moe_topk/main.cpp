/**
 * main.cpp - host driver for moe_topk.
 *
 * I/O contract (all little-endian, contiguous, no header):
 *   ../input/input_logits.bin    kRows * kCols  float32  (router logits per token)
 *   ../input/input_idx.bin       1     * kCols  uint32   ([0..kCols-1] identity)
 *   ../output/golden_val.bin     kRows * kTopK  float32  (top-K values, descending)
 *   ../output/golden_idx.bin     kRows * kTopK  uint32   (expert indices)
 *   ../output/output_val.bin     kRows * kTopK  float32  (kernel values)
 *   ../output/output_idx.bin     kRows * kTopK  uint32   (kernel indices)
 */

#include "test_common.h"
#include "acl/acl.h"
#include "../../kernel_timing.h"
#include "generated_cases.h"

#include <cstdint>
#include <cstdio>
#include <cmath>
#include <vector>

using namespace std;
using namespace PtoTestCommon;

template <typename T>
void launchMoeTopk(uint8_t *outVal, uint8_t *outIdx, uint8_t *src, uint8_t *idx, void *stream);

template <typename T, int kRows, int kTopK>
inline bool ValidateValueResults(size_t outValSize)
{
    std::vector<T> golden(outValSize / sizeof(T));
    std::vector<T> devFinal(outValSize / sizeof(T));

    ReadFile("../output/golden_val.bin", outValSize, golden.data(),   outValSize);
    ReadFile("../output/output_val.bin", outValSize, devFinal.data(), outValSize);

    bool ret = ResultCmp(golden, devFinal, 0.001f);
    if (ret) {
        printf("test value success\n");
    } else {
        printf("test value failed\n");
        int printed = 0;
        for (int r = 0; r < kRows && printed < 5; ++r) {
            for (int c = 0; c < kTopK && printed < 5; ++c) {
                int i = r * kTopK + c;
                float g = static_cast<float>(golden[i]);
                float o = static_cast<float>(devFinal[i]);
                if (std::fabs(g - o) > 0.001f) {
                    printf("  val mismatch row=%d col=%d  gold=%g  out=%g\n", r, c, g, o);
                    ++printed;
                }
            }
        }
    }
    return ret;
}

template <int kRows, int kTopK>
inline bool ValidateIndexResults(size_t outIdxSize)
{
    std::vector<uint32_t> golden(outIdxSize / sizeof(uint32_t));
    std::vector<uint32_t> devFinal(outIdxSize / sizeof(uint32_t));

    ReadFile("../output/golden_idx.bin", outIdxSize, golden.data(),   outIdxSize);
    ReadFile("../output/output_idx.bin", outIdxSize, devFinal.data(), outIdxSize);

    bool ret = ResultCmp(golden, devFinal, 0.001f);
    if (ret) {
        printf("test index success\n");
    } else {
        printf("test index failed\n");
        int printed = 0;
        for (int r = 0; r < kRows && printed < 5; ++r) {
            for (int c = 0; c < kTopK && printed < 5; ++c) {
                int i = r * kTopK + c;
                if (golden[i] != devFinal[i]) {
                    printf("  idx mismatch row=%d col=%d  gold=%u  out=%u\n",
                           r, c, golden[i], devFinal[i]);
                    ++printed;
                }
            }
        }
    }
    return ret;
}

int main()
{
    constexpr int kRows = kMoeT;
    constexpr int kCols = kMoeE;
    constexpr int kTopK = (kMoeTopK >= 2) ? kMoeTopK : 2;
    using indexT = uint32_t;

    size_t srcSize    = static_cast<size_t>(kRows) * kCols * sizeof(float);
    size_t idxSize    = static_cast<size_t>(kCols)         * sizeof(indexT);
    size_t outValSize = static_cast<size_t>(kRows) * kTopK * sizeof(float);
    size_t outIdxSize = static_cast<size_t>(kRows) * kTopK * sizeof(indexT);

    printf("[main] kRows=%d  kCols=%d  kTopK=%d\n"
           "       srcSize=%zu  idxSize=%zu  outValSize=%zu  outIdxSize=%zu\n",
           kRows, kCols, kTopK, srcSize, idxSize, outValSize, outIdxSize);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint8_t *srcHost = nullptr, *idxHost = nullptr;
    uint8_t *outValHost = nullptr, *outIdxHost = nullptr;
    uint8_t *srcDev  = nullptr, *idxDev  = nullptr;
    uint8_t *outValDev  = nullptr, *outIdxDev  = nullptr;

    aclrtMallocHost((void **)&srcHost,    srcSize);
    aclrtMallocHost((void **)&idxHost,    idxSize);
    aclrtMallocHost((void **)&outValHost, outValSize);
    aclrtMallocHost((void **)&outIdxHost, outIdxSize);

    aclrtMalloc((void **)&srcDev,    srcSize,    ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&idxDev,    idxSize,    ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&outValDev, outValSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&outIdxDev, outIdxSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile("../input/input_logits.bin", srcSize, srcHost, srcSize);
    ReadFile("../input/input_idx.bin",    idxSize, idxHost, idxSize);

    aclrtMemcpy(srcDev, srcSize, srcHost, srcSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(idxDev, idxSize, idxHost, idxSize, ACL_MEMCPY_HOST_TO_DEVICE);

    (void)PtoTiming::TimeKernelCallUs("moe_topk", stream, [&]() {
        launchMoeTopk<float>(outValDev, outIdxDev, srcDev, idxDev, stream);
    });
    aclrtMemcpy(outValHost, outValSize, outValDev, outValSize, ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(outIdxHost, outIdxSize, outIdxDev, outIdxSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile("../output/output_val.bin", outValHost, outValSize);
    WriteFile("../output/output_idx.bin", outIdxHost, outIdxSize);

    aclrtFree(outIdxDev);
    aclrtFree(outValDev);
    aclrtFree(idxDev);
    aclrtFree(srcDev);
    aclrtFreeHost(outIdxHost);
    aclrtFreeHost(outValHost);
    aclrtFreeHost(idxHost);
    aclrtFreeHost(srcHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    bool valOk = ValidateValueResults<float, kRows, kTopK>(outValSize);
    bool idxOk = ValidateIndexResults<kRows, kTopK>(outIdxSize);
    if (valOk && idxOk) {
        printf("test success\n");
    } else {
        printf("test failed\n");
    }
    return (valOk && idxOk) ? 0 : 1;
}
