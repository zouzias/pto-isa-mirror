/**
 * main.cpp - host driver for gather.
 *
 * Unpack-and-accumulate with softmax routing weights when kTopK > 1.
 * For kTopK == 1 the kernel takes the fast path (no softmax, no TMULS).
 *
 * I/O contract (all little-endian, contiguous, no header):
 *   ../input/input_B.bin       (kT*kTopK + 64) * kH    float32
 *   ../input/input_A_id.bin    (kT*kTopK + 64)         int32 (trailing 64 = -1)
 *   ../input/input_rank_id.bin (kT*kTopK + 64)         int32 (trailing 64 = -1, only consulted when kTopK > 1)
 *   ../input/input_outVal.bin  kT * kPadded            float32 (cols kTopK..kPadded-1 host-padded with -1e30)
 *   ../output/golden_C.bin     kT * kH                 float32
 *   ../output/output_C.bin     kT * kH                 float32 (kernel-emitted)
 *
 *   kPadded = max(8, kTopK) — softmax tile column padding for 32-byte UB alignment.
 *
 * weights_scratch is GM-only (no host file). The host allocates it but never
 * touches the contents; the kernel writes it in pass 1 and reads in pass 2.
 */

#include "test_common.h"
#include "acl/acl.h"
#include "generated_cases.h"
#include "../../kernel_timing.h"

#include <cstdint>
#include <cstdio>
#include <vector>

using namespace std;
using namespace PtoTestCommon;

template <typename T>
void launchGather(T *C, T *B,
                  int32_t *A_id, int32_t *rank_id,
                  T *outVal, T *weights_scratch,
                  void *stream);

namespace {

constexpr int kT    = kMoeT;
constexpr int kH    = kMoeH;
constexpr int kTopK = kMoeTopK;

constexpr int kPackedRows   = kT * kTopK;
constexpr int kOverspillPad = 64;
constexpr int kAlloc        = kPackedRows + kOverspillPad;

constexpr int kPadded = (kTopK < 8) ? 8 : kTopK;

bool ValidateC(size_t cBytes)
{
    std::vector<float> gold(cBytes / sizeof(float));
    std::vector<float> out (cBytes / sizeof(float));
    ReadFile("../output/golden_C.bin", cBytes, gold.data(), cBytes);
    ReadFile("../output/output_C.bin", cBytes, out.data(),  cBytes);

    // 1e-3 abs tolerance — softmax adds TEXP / divide rounding on top of
    // the fp32 accumulation; matches numpy reference within fp32 epsilon.
    bool ok = ResultCmp(gold, out, 1e-3f);
    printf("C                : %s\n", ok ? "success" : "FAILED");
    return ok;
}

}  // namespace

int main()
{
    size_t bBytes        = static_cast<size_t>(kAlloc)  * kH       * sizeof(float);
    size_t aIdBytes      = static_cast<size_t>(kAlloc)             * sizeof(int32_t);
    size_t rankIdBytes   = static_cast<size_t>(kAlloc)             * sizeof(int32_t);
    size_t outValBytes   = static_cast<size_t>(kT)      * kPadded  * sizeof(float);
    size_t weightsBytes  = static_cast<size_t>(kT)      * kPadded  * sizeof(float);
    size_t cBytes        = static_cast<size_t>(kT)      * kH       * sizeof(float);

    printf("[main] kT=%d  kH=%d  kTopK=%d  kPadded=%d  kAlloc=%d\n",
           kT, kH, kTopK, kPadded, kAlloc);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    float   *bHost = nullptr, *cHost = nullptr, *outValHost = nullptr;
    int32_t *aIdHost = nullptr, *rankIdHost = nullptr;

    float   *bDev = nullptr, *cDev = nullptr, *outValDev = nullptr, *weightsDev = nullptr;
    int32_t *aIdDev = nullptr, *rankIdDev = nullptr;

    aclrtMallocHost((void **)&bHost,      bBytes);
    aclrtMallocHost((void **)&cHost,      cBytes);
    aclrtMallocHost((void **)&aIdHost,    aIdBytes);
    aclrtMallocHost((void **)&rankIdHost, rankIdBytes);
    aclrtMallocHost((void **)&outValHost, outValBytes);

    aclrtMalloc((void **)&bDev,       bBytes,       ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&cDev,       cBytes,       ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&aIdDev,     aIdBytes,     ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&rankIdDev,  rankIdBytes,  ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&outValDev,  outValBytes,  ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&weightsDev, weightsBytes, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile("../input/input_B.bin",       bBytes,      bHost,      bBytes);
    ReadFile("../input/input_A_id.bin",    aIdBytes,    aIdHost,    aIdBytes);
    ReadFile("../input/input_rank_id.bin", rankIdBytes, rankIdHost, rankIdBytes);
    ReadFile("../input/input_outVal.bin",  outValBytes, outValHost, outValBytes);

    // Zero-init C: kernel does TLOAD(C[t]) on the very first contribution,
    // so C must be zero on entry. 0x00 bytes in float32 = +0.0f.
    aclrtMemset(cDev, cBytes, 0x00, cBytes);

    // Poison weights_scratch so we can detect if the kernel forgot to write
    // it before reading in pass 2. (Only matters when kTopK > 1.)
    aclrtMemset(weightsDev, weightsBytes, 0x5A, weightsBytes);

    aclrtMemcpy(bDev,      bBytes,      bHost,      bBytes,      ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(aIdDev,    aIdBytes,    aIdHost,    aIdBytes,    ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(rankIdDev, rankIdBytes, rankIdHost, rankIdBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(outValDev, outValBytes, outValHost, outValBytes, ACL_MEMCPY_HOST_TO_DEVICE);

    (void)PtoTiming::TimeKernelCallUs("gather", stream, [&]() {
        launchGather<float>(cDev, bDev, aIdDev, rankIdDev, outValDev, weightsDev, stream);
    });
    aclrtMemcpy(cHost, cBytes, cDev, cBytes, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile("../output/output_C.bin", cHost, cBytes);

    aclrtFree(weightsDev);
    aclrtFree(outValDev);
    aclrtFree(rankIdDev);
    aclrtFree(aIdDev);
    aclrtFree(cDev);
    aclrtFree(bDev);
    aclrtFreeHost(outValHost);
    aclrtFreeHost(rankIdHost);
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
