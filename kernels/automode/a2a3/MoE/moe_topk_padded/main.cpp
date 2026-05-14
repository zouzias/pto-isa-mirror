/**
 * main.cpp - host driver for moe_topk_padded.
 *
 * Per-row top-K selection over router logits, generic over
 * kTopK in {1, 2, 4, 8, 16}. To sweep kTopK, edit the constexpr in main()
 * AND the matching constant in moe_topk_padded_kernel.cpp / gen_data.py,
 * then rebuild.
 *
 * I/O contract (all little-endian, contiguous, no header):
 *   ../input/input_logits.bin    kT * kE    float32  (router logits per token)
 *   ../input/input_idx.bin       1  * kE    uint32   ([0..kE-1] identity)
 *   ../output/golden_val.bin     kT * kTopK float32  (top-K values, descending)
 *   ../output/golden_idx.bin     kT * kTopK uint32   (matching expert ids)
 *   ../output/output_val.bin     kT * kTopK float32  (kernel values)
 *   ../output/output_idx.bin     kT * kTopK uint32   (kernel indices)
 */

#include "test_common.h"
#include "acl/acl.h"

#include <cstdint>
#include <cstdio>
#include <cmath>
#include <vector>

using namespace std;
using namespace PtoTestCommon;

template <typename T>
void launchMoeTopkPadded(uint8_t *outVal, uint8_t *outIdx,
                         uint8_t *src,    uint8_t *idx,
                         void *stream);

template <typename T, int kT_, int kTopK_>
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
        for (int r = 0; r < kT_ && printed < 5; ++r) {
            for (int c = 0; c < kTopK_ && printed < 5; ++c) {
                int i = r * kTopK_ + c;
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

template <int kT_, int kTopK_>
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
        for (int r = 0; r < kT_ && printed < 5; ++r) {
            for (int c = 0; c < kTopK_ && printed < 5; ++c) {
                int i = r * kTopK_ + c;
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
    // v1 shape — must match moe_topk_padded_kernel.cpp and gen_data.py.
    constexpr int kT    = 256;
    constexpr int kE    = 32;
    constexpr int kTopK = 1;
    using indexT = uint32_t;

    size_t srcSize    = static_cast<size_t>(kT) * kE    * sizeof(float);
    size_t idxSize    = static_cast<size_t>(kE)         * sizeof(indexT);
    size_t outValSize = static_cast<size_t>(kT) * kTopK * sizeof(float);
    size_t outIdxSize = static_cast<size_t>(kT) * kTopK * sizeof(indexT);

    printf("[main] kT=%d  kE=%d  kTopK=%d\n"
           "       srcSize=%zu  idxSize=%zu  outValSize=%zu  outIdxSize=%zu\n",
           kT, kE, kTopK, srcSize, idxSize, outValSize, outIdxSize);

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

    launchMoeTopkPadded<float>(outValDev, outIdxDev, srcDev, idxDev, stream);

    aclrtSynchronizeStream(stream);
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

    bool valOk = ValidateValueResults<float, kT, kTopK>(outValSize);
    bool idxOk = ValidateIndexResults<kT, kTopK>(outIdxSize);
    if (valOk && idxOk) {
        printf("test data success\n");
        printf("test success\n");
    } else {
        printf("test data failed\n");
        printf("test failed\n");
    }
    return (valOk && idxOk) ? 0 : 1;
}
