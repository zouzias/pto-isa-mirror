/**
 * main.cpp - host driver for gather.
 *
 * Unpack-and-accumulate kernel: C[A_id[r]] += B[r] for r in [0, kT*kTopK).
 *
 * I/O contract (all little-endian, contiguous, no header):
 *   ../input/input_B.bin       (kT*kTopK + 16) * kH    float32
 *   ../input/input_A_id.bin    (kT*kTopK + 16)         int32 (trailing 16 = -1, ignored)
 *   ../output/golden_C.bin     kT * kH                 float32
 *   ../output/output_C.bin     kT * kH                 float32 (kernel-emitted)
 */

#include "test_common.h"
#include "acl/acl.h"

#include <cstdint>
#include <cstdio>
#include <vector>

using namespace std;
using namespace PtoTestCommon;

template <typename T>
void launchGather(T *C, T *B, int32_t *A_id, void *stream);

namespace {

constexpr int kT    = 256;
constexpr int kH    = 64;
constexpr int kTopK = 1;

constexpr int kPackedRows   = kT * kTopK;
constexpr int kOverspillPad = 16;
constexpr int kAlloc        = kPackedRows + kOverspillPad;

bool ValidateC(size_t cBytes)
{
    std::vector<float> gold(cBytes / sizeof(float));
    std::vector<float> out (cBytes / sizeof(float));
    ReadFile("../output/golden_C.bin", cBytes, gold.data(), cBytes);
    ReadFile("../output/output_C.bin", cBytes, out.data(),  cBytes);

    bool ok = ResultCmp(gold, out, 1e-4f);
    printf("C                : %s\n", ok ? "success" : "FAILED");
    return ok;
}

}  // namespace

int main()
{
    size_t bBytes   = static_cast<size_t>(kAlloc)      * kH * sizeof(float);
    size_t aIdBytes = static_cast<size_t>(kAlloc)      * sizeof(int32_t);
    size_t cBytes   = static_cast<size_t>(kT)          * kH * sizeof(float);

    printf("[main] kT=%d  kH=%d  kTopK=%d  kAlloc=%d\n", kT, kH, kTopK, kAlloc);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    float   *bHost = nullptr, *cHost = nullptr;
    int32_t *aIdHost = nullptr;

    float   *bDev = nullptr, *cDev = nullptr;
    int32_t *aIdDev = nullptr;

    aclrtMallocHost((void **)&bHost,   bBytes);
    aclrtMallocHost((void **)&cHost,   cBytes);
    aclrtMallocHost((void **)&aIdHost, aIdBytes);

    aclrtMalloc((void **)&bDev,   bBytes,   ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&cDev,   cBytes,   ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&aIdDev, aIdBytes, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile("../input/input_B.bin",    bBytes,   bHost,   bBytes);
    ReadFile("../input/input_A_id.bin", aIdBytes, aIdHost, aIdBytes);

    // Zero-init C: kernel does TLOAD(C[t]) on the very first contribution,
    // so C must be zero on entry. 0x00 bytes in float32 = +0.0f.
    aclrtMemset(cDev, cBytes, 0x00, cBytes);

    aclrtMemcpy(bDev,   bBytes,   bHost,   bBytes,   ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(aIdDev, aIdBytes, aIdHost, aIdBytes, ACL_MEMCPY_HOST_TO_DEVICE);

    launchGather<float>(cDev, bDev, aIdDev, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(cHost, cBytes, cDev, cBytes, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile("../output/output_C.bin", cHost, cBytes);

    aclrtFree(aIdDev);
    aclrtFree(cDev);
    aclrtFree(bDev);
    aclrtFreeHost(aIdHost);
    aclrtFreeHost(cHost);
    aclrtFreeHost(bHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    bool ok = ValidateC(cBytes);
    if (ok) {
        printf("test data success\n");
        printf("test success\n");
    } else {
        printf("test data failed\n");
        printf("test failed\n");
    }
    return ok ? 0 : 1;
}
