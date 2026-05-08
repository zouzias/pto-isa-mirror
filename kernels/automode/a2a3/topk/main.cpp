/**
 * main.cpp - host driver for topk (auto-mode A3 prototype, v1 values-only).
 *
 * Pattern source: kernels/automode/a2a3/add_tile_array/main.cpp (which
 * itself mirrors kernels/manual/a2a3/topk/main.cpp). Reads ../input/*.bin
 * and writes ../output/*.bin relative to the build/ directory; uses
 * `tests/common/test_common.h` `ReadFile` / `WriteFile` / `ResultCmp`.
 *
 * I/O contract (all little-endian, contiguous, no header):
 *   ../input/input_src.bin    1 row * 1280 float32 (pre-sorted in 64-blocks descending)
 *   ../output/golden_val.bin  1 row * 512  float32 top-K (descending)
 *   ../output/output_val.bin  1 row * 512  float32 top-K from kernel
 *
 * The pre-sort precondition is enforced by scripts/gen_data.py.
 */

#include "test_common.h"
#include "acl/acl.h"

#include <cstdint>
#include <cstdio>
#include <vector>

using namespace std;
using namespace PtoTestCommon;

template <typename T>
void launchTopk(uint8_t *out, uint8_t *src, void *stream);

template <typename T>
inline bool ValidateValueResults(size_t outFileSize)
{
    std::vector<T> golden(outFileSize / sizeof(T));
    std::vector<T> devFinal(outFileSize / sizeof(T));

    ReadFile("../output/golden_val.bin", outFileSize, golden.data(), outFileSize);
    ReadFile("../output/output_val.bin", outFileSize, devFinal.data(), outFileSize);

    bool ret = ResultCmp(golden, devFinal, 0.001f);
    if (ret) {
        printf("test data success\n");
    } else {
        printf("test data failed\n");
    }
    return ret;
}

template <typename T, int kCols, int kTopK>
void TopkKernel()
{
    size_t inFileSize = static_cast<size_t>(kCols) * sizeof(T);
    size_t outFileSize = static_cast<size_t>(kTopK) * sizeof(T);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint8_t *srcHost = nullptr, *dstHost = nullptr;
    uint8_t *srcDevice = nullptr, *dstDevice = nullptr;

    aclrtMallocHost((void **)(&srcHost), inFileSize);
    aclrtMallocHost((void **)(&dstHost), outFileSize);

    aclrtMalloc((void **)&srcDevice, inFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&dstDevice, outFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile("../input/input_src.bin", inFileSize, srcHost, inFileSize);
    aclrtMemcpy(srcDevice, inFileSize, srcHost, inFileSize, ACL_MEMCPY_HOST_TO_DEVICE);

    launchTopk<T>(dstDevice, srcDevice, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, outFileSize, dstDevice, outFileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile("../output/output_val.bin", dstHost, outFileSize);

    aclrtFree(dstDevice);
    aclrtFree(srcDevice);
    aclrtFreeHost(dstHost);
    aclrtFreeHost(srcHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    bool dataSuccess = ValidateValueResults<T>(outFileSize);
    if (dataSuccess) {
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
