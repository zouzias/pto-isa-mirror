/**
 * main.cpp - host driver for expert_ffn.
 *
 * I/O contract (all little-endian, contiguous, no header):
 *   ../input/input_x_packed.bin     kT * kD  float32  (tokens sorted by expert)
 *   ../input/input_expert_count.bin kE       int32    (tokens per expert)
 *   ../input/input_expert_start.bin kE       int32    (start offset per expert)
 *   ../input/input_w1.bin           kE * kD  float32  (gate weights W1)
 *   ../input/input_w2.bin           kE * kD  float32  (gate weights W2)
 *   ../output/golden_output.bin     kT * kD  float32  (reference output)
 *   ../output/output_output.bin     kT * kD  float32  (kernel output)
 */

#include "test_common.h"
#include "acl/acl.h"

#include <cstdint>
#include <cstdio>
#include <vector>

using namespace std;
using namespace PtoTestCommon;

extern "C" void launchExpertFfnFloat(uint8_t *output, uint8_t *x_packed,
                                      int32_t *expert_count, int32_t *expert_start,
                                      uint8_t *w1, uint8_t *w2, void *stream);

int main()
{
    constexpr int kT = 256;
    constexpr int kD = 64;
    constexpr int kE = 32;
    constexpr size_t floatBytes = 4;
    constexpr size_t int32Bytes = 4;

    size_t xBytes    = static_cast<size_t>(kT) * kD * floatBytes;
    size_t wBytes    = static_cast<size_t>(kE) * kD * floatBytes;
    size_t outBytes  = static_cast<size_t>(kT) * kD * floatBytes;
    size_t metaBytes = static_cast<size_t>(kE) * int32Bytes;

    printf("[main] kT=%d  kD=%d  kE=%d\n"
           "       xBytes=%zu  wBytes=%zu  outBytes=%zu\n",
           kT, kD, kE, xBytes, wBytes, outBytes);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint8_t *xHost = nullptr, *w1Host = nullptr, *w2Host = nullptr, *outHost = nullptr;
    int32_t *countHost = nullptr, *startHost = nullptr;
    uint8_t *xDev = nullptr, *w1Dev = nullptr, *w2Dev = nullptr, *outDev = nullptr;
    int32_t *countDev = nullptr, *startDev = nullptr;

    aclrtMallocHost((void **)&xHost,     xBytes);
    aclrtMallocHost((void **)&w1Host,    wBytes);
    aclrtMallocHost((void **)&w2Host,    wBytes);
    aclrtMallocHost((void **)&outHost,   outBytes);
    aclrtMallocHost((void **)&countHost, metaBytes);
    aclrtMallocHost((void **)&startHost, metaBytes);

    aclrtMalloc((void **)&xDev,     xBytes,    ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&w1Dev,    wBytes,    ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&w2Dev,    wBytes,    ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&outDev,   outBytes,  ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&countDev, metaBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&startDev, metaBytes, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile("../input/input_x_packed.bin",     xBytes,    xHost,     xBytes);
    ReadFile("../input/input_w1.bin",           wBytes,    w1Host,    wBytes);
    ReadFile("../input/input_w2.bin",           wBytes,    w2Host,    wBytes);
    ReadFile("../input/input_expert_count.bin", metaBytes, countHost, metaBytes);
    ReadFile("../input/input_expert_start.bin", metaBytes, startHost, metaBytes);

    aclrtMemcpy(xDev,     xBytes,    xHost,     xBytes,    ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(w1Dev,    wBytes,    w1Host,    wBytes,    ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(w2Dev,    wBytes,    w2Host,    wBytes,    ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(countDev, metaBytes, countHost, metaBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(startDev, metaBytes, startHost, metaBytes, ACL_MEMCPY_HOST_TO_DEVICE);

    launchExpertFfnFloat(outDev, xDev, countDev, startDev, w1Dev, w2Dev, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(outHost, outBytes, outDev, outBytes, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile("../output/output_output.bin", outHost, outBytes);

    aclrtFree(startDev);
    aclrtFree(countDev);
    aclrtFree(outDev);
    aclrtFree(w2Dev);
    aclrtFree(w1Dev);
    aclrtFree(xDev);
    aclrtFreeHost(startHost);
    aclrtFreeHost(countHost);
    aclrtFreeHost(outHost);
    aclrtFreeHost(w2Host);
    aclrtFreeHost(w1Host);
    aclrtFreeHost(xHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<float> golden(outBytes / sizeof(float));
    std::vector<float> devFinal(outBytes / sizeof(float));
    ReadFile("../output/golden_output.bin", outBytes, golden.data(),   outBytes);
    ReadFile("../output/output_output.bin", outBytes, devFinal.data(), outBytes);

    // All float32 element-wise ops; results should match closely.
    bool ok = ResultCmp(golden, devFinal, 1e-4f);
    if (ok) {
        printf("test data success\n");
    } else {
        printf("test data failed\n");
    }
    return ok ? 0 : 1;
}
