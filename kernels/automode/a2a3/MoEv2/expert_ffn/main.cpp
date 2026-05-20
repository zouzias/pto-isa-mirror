/**
 * main.cpp - host driver for expert_ffn.
 *
 * Fused per-expert FFN: for each expert-local A_s tile, compute
 * relu(A_s @ W1_t) @ W2_t and store B_s to GM before moving to the next A_s.
 *
 * I/O contract (all little-endian, contiguous, no header):
 *   ../input/input_A.bin             (kT*kTopK + 16) * kH        half (fp16)
 *   ../input/input_expert_count.bin   kE                          int32
 *   ../input/input_expert_start.bin   kE                          int32
 *   ../input/input_W1.bin             kE * kH * kF                half
 *   ../input/input_W2.bin             kE * kF * kH                half
 *   ../output/golden_B.bin            (kT*kTopK + 16) * kH        float32
 *   ../output/output_B.bin            (kT*kTopK + 16) * kH        float32  (kernel-emitted)
 *
 * Validation compares only the first kT*kTopK rows of B; the trailing 16-row
 * ABI pad is ignored.
 */

#include "test_common.h"
#include "acl/acl.h"
#include "../../kernel_timing.h"

#include <cstdint>
#include <cstdio>
#include <vector>

using namespace std;
using namespace PtoTestCommon;

extern "C" void launchExpertFfnFp16(uint8_t *B,
                                    uint8_t *A,
                                    int32_t *expert_count,
                                    int32_t *expert_start,
                                    uint8_t *W1,
                                    uint8_t *W2,
                                    uint8_t *Y_scratch,
                                    void    *stream);

namespace {

constexpr int kT     = 256;
constexpr int kH     = 64;
constexpr int kF     = 64;
constexpr int kE     = 32;
constexpr int kTopK  = 1;
constexpr int kTileM = 16;

constexpr int kPackedRows   = kT * kTopK;
constexpr int kOverspillPad = kTileM;
constexpr int kAlloc        = kPackedRows + kOverspillPad;

// Cube blockAlign for fp16 = 16.  A/W1/W2 are zero-padded to these dims
// so the GEMM MatTile K/N padding columns read zeros, not garbage UB.
constexpr int kH_aligned = ((kH + 15) / 16) * 16;
constexpr int kF_aligned = ((kF + 15) / 16) * 16;

constexpr size_t kHalfBytes  = 2;
constexpr size_t kFloatBytes = 4;

bool ValidateB(size_t allocBytes)
{
    std::vector<float> gold(allocBytes / sizeof(float));
    std::vector<float> out (allocBytes / sizeof(float));
    ReadFile("../output/golden_B.bin", allocBytes, gold.data(), allocBytes);
    ReadFile("../output/output_B.bin", allocBytes, out.data(),  allocBytes);

    // Compare only the first kPackedRows * kH floats.
    size_t validElems = static_cast<size_t>(kPackedRows) * kH;
    std::vector<float> goldValid(gold.begin(), gold.begin() + validElems);
    std::vector<float> outValid (out.begin(),  out.begin()  + validElems);

    // fp16-input GEMM accumulated in fp32; small distribution so abs tol 1e-2
    // is comfortable (mirrors mani_moe gen_data scale).
    bool ok = ResultCmp(goldValid, outValid, 1e-2f);
    printf("B                : %s\n", ok ? "success" : "FAILED");
    return ok;
}

}  // namespace

int main()
{
    size_t aBytes      = static_cast<size_t>(kAlloc) * kH_aligned          * kHalfBytes;
    size_t yBytes      = static_cast<size_t>(kAlloc) * kF_aligned          * kHalfBytes;
    size_t bBytes      = static_cast<size_t>(kAlloc) * kH                  * kFloatBytes;
    size_t w1Bytes     = static_cast<size_t>(kE)     * kH_aligned * kF_aligned * kHalfBytes;
    size_t w2Bytes     = static_cast<size_t>(kE)     * kF_aligned * kH_aligned * kHalfBytes;
    size_t expBytes    = static_cast<size_t>(kE)                          * sizeof(int32_t);

    printf("[main] kT=%d  kH=%d  kF=%d  kE=%d  kTopK=%d  kTileM=%d  kAlloc=%d\n",
           kT, kH, kF, kE, kTopK, kTileM, kAlloc);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint8_t *aHost = nullptr, *w1Host = nullptr, *w2Host = nullptr, *bHost = nullptr;
    int32_t *countHost = nullptr, *startHost = nullptr;

    uint8_t *aDev = nullptr, *w1Dev = nullptr, *w2Dev = nullptr,
            *bDev = nullptr, *yDev = nullptr;
    int32_t *countDev = nullptr, *startDev = nullptr;

    aclrtMallocHost((void **)&aHost,     aBytes);
    aclrtMallocHost((void **)&w1Host,    w1Bytes);
    aclrtMallocHost((void **)&w2Host,    w2Bytes);
    aclrtMallocHost((void **)&bHost,     bBytes);
    aclrtMallocHost((void **)&countHost, expBytes);
    aclrtMallocHost((void **)&startHost, expBytes);

    aclrtMalloc((void **)&aDev,     aBytes,   ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&w1Dev,    w1Bytes,  ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&w2Dev,    w2Bytes,  ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&bDev,     bBytes,   ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&yDev,     yBytes,   ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&countDev, expBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&startDev, expBytes, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile("../input/input_A.bin",            aBytes,   aHost,     aBytes);
    ReadFile("../input/input_W1.bin",           w1Bytes,  w1Host,    w1Bytes);
    ReadFile("../input/input_W2.bin",           w2Bytes,  w2Host,    w2Bytes);
    ReadFile("../input/input_expert_count.bin", expBytes, countHost, expBytes);
    ReadFile("../input/input_expert_start.bin", expBytes, startHost, expBytes);

    aclrtMemcpy(aDev,     aBytes,   aHost,     aBytes,   ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(w1Dev,    w1Bytes,  w1Host,    w1Bytes,  ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(w2Dev,    w2Bytes,  w2Host,    w2Bytes,  ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(countDev, expBytes, countHost, expBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(startDev, expBytes, startHost, expBytes, ACL_MEMCPY_HOST_TO_DEVICE);

    // Poison B and Y scratch so we can detect unwritten rows in the overspill
    // and confirm the kernel actually overwrites the valid region.
    aclrtMemset(bDev, bBytes, 0x5A, bBytes);
    aclrtMemset(yDev, yBytes, 0x7B, yBytes);

    (void)PtoTiming::TimeKernelCallUs("expert_ffn", stream, [&]() {
        launchExpertFfnFp16(bDev, aDev, countDev, startDev, w1Dev, w2Dev, yDev, stream);
    });
    aclrtMemcpy(bHost, bBytes, bDev, bBytes, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile("../output/output_B.bin", bHost, bBytes);

    aclrtFree(startDev);
    aclrtFree(countDev);
    aclrtFree(yDev);
    aclrtFree(bDev);
    aclrtFree(w2Dev);
    aclrtFree(w1Dev);
    aclrtFree(aDev);
    aclrtFreeHost(startHost);
    aclrtFreeHost(countHost);
    aclrtFreeHost(bHost);
    aclrtFreeHost(w2Host);
    aclrtFreeHost(w1Host);
    aclrtFreeHost(aHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    bool ok = ValidateB(bBytes);
    if (ok) {
        printf("test data success\n");
        printf("test success\n");
    } else {
        printf("test data failed\n");
        printf("test failed\n");
    }
    return ok ? 0 : 1;
}
