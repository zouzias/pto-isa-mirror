/**
 * main.cpp - host driver for topk (auto-mode A3 prototype, full TopK).
 *
 * Pattern source: kernels/manual/a2a3/topk/main.cpp (4 GM tensors:
 * out_val, out_idx, src, idx; ResultCmp on both values and indices).
 *
 * I/O contract (all little-endian, contiguous, no header):
 *   ../input/input_src.bin     1 * kCols  float32   (raw random unsorted)
 *   ../input/input_idx.bin     1 * kCols  uint32_t  ([0..kCols-1])
 *   ../output/golden_val.bin   1 * kTopK  float32   (top-K values, descending)
 *   ../output/golden_idx.bin   1 * kTopK  uint32_t  (matching indices)
 *   ../output/output_val.bin   1 * kTopK  float32   (kernel value output)
 *   ../output/output_idx.bin   1 * kTopK  uint32_t  (kernel index output)
 */

#include "test_common.h"
#include "acl/acl.h"

#include <cstdint>
#include <cstdio>
#include <vector>

using namespace std;
using namespace PtoTestCommon;

template <typename T>
void launchTopk(uint8_t *outVal, uint8_t *outIdx, uint8_t *src, uint8_t *idx, void *stream);

template <typename T>
inline bool ValidateValueResults(size_t outValSize)
{
    std::vector<T> golden(outValSize / sizeof(T));
    std::vector<T> devFinal(outValSize / sizeof(T));

    ReadFile("../output/golden_val.bin", outValSize, golden.data(), outValSize);
    ReadFile("../output/output_val.bin", outValSize, devFinal.data(), outValSize);

    bool ret = ResultCmp(golden, devFinal, 0.001f);
    if (ret) {
        printf("test value success\n");
    } else {
        printf("test value failed\n");
    }
    return ret;
}

inline bool ValidateIndexResults(size_t outIdxSize)
{
    std::vector<uint32_t> golden(outIdxSize / sizeof(uint32_t));
    std::vector<uint32_t> devFinal(outIdxSize / sizeof(uint32_t));

    ReadFile("../output/golden_idx.bin", outIdxSize, golden.data(), outIdxSize);
    ReadFile("../output/output_idx.bin", outIdxSize, devFinal.data(), outIdxSize);

    bool ret = ResultCmp(golden, devFinal, 0.001f);
    if (ret) {
        printf("test index success\n");
    } else {
        printf("test index failed\n");
    }
    return ret;
}

template <typename T, int kCols, int kTopK>
void TopkKernel()
{
    using indexT = uint32_t;
    size_t srcSize    = static_cast<size_t>(kCols) * sizeof(T);
    size_t idxSize    = static_cast<size_t>(kCols) * sizeof(indexT);
    size_t outValSize = static_cast<size_t>(kTopK) * sizeof(T);
    size_t outIdxSize = static_cast<size_t>(kTopK) * sizeof(indexT);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint8_t *srcHost = nullptr, *idxHost = nullptr, *outValHost = nullptr, *outIdxHost = nullptr;
    uint8_t *srcDev  = nullptr, *idxDev  = nullptr, *outValDev  = nullptr, *outIdxDev  = nullptr;

    aclrtMallocHost((void **)(&srcHost),    srcSize);
    aclrtMallocHost((void **)(&idxHost),    idxSize);
    aclrtMallocHost((void **)(&outValHost), outValSize);
    aclrtMallocHost((void **)(&outIdxHost), outIdxSize);

    aclrtMalloc((void **)&srcDev,    srcSize,    ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&idxDev,    idxSize,    ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&outValDev, outValSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&outIdxDev, outIdxSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile("../input/input_src.bin", srcSize, srcHost, srcSize);
    ReadFile("../input/input_idx.bin", idxSize, idxHost, idxSize);

    aclrtMemcpy(srcDev, srcSize, srcHost, srcSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(idxDev, idxSize, idxHost, idxSize, ACL_MEMCPY_HOST_TO_DEVICE);

    launchTopk<T>(outValDev, outIdxDev, srcDev, idxDev, stream);

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

    bool valOk = ValidateValueResults<T>(outValSize);
    bool idxOk = ValidateIndexResults(outIdxSize);
    if (valOk && idxOk) {
        printf("test success\n");
    } else {
        printf("test failed\n");
    }
}

int main()
{
    constexpr int kCols = 1280;
    constexpr int kTopK = 512;
    TopkKernel<float, kCols, kTopK>();
    return 0;
}
