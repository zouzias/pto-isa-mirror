/**
 * main.cpp - host driver for scatter.
 *
 * Pack tokens by expert assignment. Generic over kTopK in {1, 2, 4, 8, 16}.
 *
 * I/O contract (all little-endian, contiguous, no header):
 *   ../input/input_X.bin             kT * kH                      half  (fp16)
 *   ../input/input_expert_id.bin     kT * kTopK                   int32
 *   ../output/golden_A.bin           (kT*kTopK + 64) * kH         half  (last 64 rows = zeros)
 *   ../output/golden_A_id.bin        (kT*kTopK + 64)              int32 (last 64 = -1)
 *   ../output/golden_expert_count.bin kE                          int32
 *   ../output/golden_expert_start.bin kE                          int32
 *   ../output/output_*.bin            (kernel-emitted; same shapes)
 *
 * Validation compares only the first kT*kTopK rows of A and A_id; the
 * trailing 64-row overspill pad is ignored.
 */

#include "test_common.h"
#include "acl/acl.h"
#include "../../kernel_timing.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

using namespace std;
using namespace PtoTestCommon;

extern "C" void launchScatterFp16(uint8_t *A, int32_t *A_id, int32_t *rank_id,
                                  int32_t *expert_count, int32_t *expert_start,
                                  uint8_t *X, int32_t *expert_id,
                                  void *stream);

namespace {

constexpr int kT     = 256;
constexpr int kH     = 64;
constexpr int kE     = 32;
constexpr int kTopK  = 1;

constexpr int kPackedRows   = kT * kTopK;
constexpr int kOverspillPad = 64;
constexpr int kAlloc        = kPackedRows + kOverspillPad;

constexpr size_t kHalfBytes = 2;

// Bytewise equality on the first kPackedRows*kH fp16 elements. fp16 stays
// bit-exact through a permute, so an exact memcmp is the right check.
bool ValidateA(size_t allocBytes)
{
    std::vector<uint8_t> gold(allocBytes);
    std::vector<uint8_t> out (allocBytes);
    ReadFile("../output/golden_A.bin", allocBytes, gold.data(), allocBytes);
    ReadFile("../output/output_A.bin", allocBytes, out.data(),  allocBytes);
    size_t validBytes = static_cast<size_t>(kPackedRows) * kH * kHalfBytes;
    bool ok = std::memcmp(gold.data(), out.data(), validBytes) == 0;
    printf("A                : %s\n", ok ? "success" : "FAILED");
    if (!ok) {
        int printed = 0;
        for (size_t i = 0; i < validBytes && printed < 16; ++i) {
            if (gold[i] != out[i]) {
                printf("  A byte mismatch off=%zu  gold=0x%02X  out=0x%02X\n",
                       i, gold[i], out[i]);
                ++printed;
            }
        }
    }
    return ok;
}

bool ValidateAid(size_t allocBytes)
{
    std::vector<int32_t> gold(allocBytes / sizeof(int32_t));
    std::vector<int32_t> out (allocBytes / sizeof(int32_t));
    ReadFile("../output/golden_A_id.bin", allocBytes, gold.data(), allocBytes);
    ReadFile("../output/output_A_id.bin", allocBytes, out.data(),  allocBytes);
    bool ok = true;
    int printed = 0;
    for (int i = 0; i < kPackedRows; ++i) {
        if (gold[i] != out[i]) {
            if (printed < 16) {
                printf("  A_id mismatch i=%d  gold=%d  out=%d\n", i, gold[i], out[i]);
                ++printed;
            }
            ok = false;
        }
    }
    printf("A_id             : %s\n", ok ? "success" : "FAILED");
    return ok;
}

bool ValidateRankId(size_t allocBytes)
{
    std::vector<int32_t> gold(allocBytes / sizeof(int32_t));
    std::vector<int32_t> out (allocBytes / sizeof(int32_t));
    ReadFile("../output/golden_rank_id.bin", allocBytes, gold.data(), allocBytes);
    ReadFile("../output/output_rank_id.bin", allocBytes, out.data(),  allocBytes);
    bool ok = true;
    int printed = 0;
    for (int i = 0; i < kPackedRows; ++i) {
        if (gold[i] != out[i]) {
            if (printed < 16) {
                printf("  rank_id mismatch i=%d  gold=%d  out=%d\n", i, gold[i], out[i]);
                ++printed;
            }
            ok = false;
        }
    }
    printf("rank_id          : %s\n", ok ? "success" : "FAILED");
    return ok;
}

bool ValidateExpertArr(const char *goldPath, const char *outPath, const char *label, size_t bytes)
{
    std::vector<int32_t> gold(bytes / sizeof(int32_t));
    std::vector<int32_t> out (bytes / sizeof(int32_t));
    ReadFile(goldPath, bytes, gold.data(), bytes);
    ReadFile(outPath,  bytes, out.data(),  bytes);
    bool ok = (gold == out);
    printf("%-16s : %s\n", label, ok ? "success" : "FAILED");
    if (!ok) {
        for (int i = 0; i < kE; ++i) {
            if (gold[i] != out[i]) {
                printf("  %s mismatch e=%d  gold=%d  out=%d\n", label, i, gold[i], out[i]);
            }
        }
    }
    return ok;
}

}  // namespace

int main()
{
    size_t xBytes        = static_cast<size_t>(kT)        * kH * kHalfBytes;
    size_t expertIdBytes = static_cast<size_t>(kT)        * kTopK * sizeof(int32_t);
    size_t aBytes        = static_cast<size_t>(kAlloc)    * kH * kHalfBytes;
    size_t aIdBytes      = static_cast<size_t>(kAlloc)    * sizeof(int32_t);
    size_t expBytes      = static_cast<size_t>(kE)        * sizeof(int32_t);

    printf("[main] kT=%d  kH=%d  kE=%d  kTopK=%d  kAlloc=%d\n", kT, kH, kE, kTopK, kAlloc);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint8_t *xHost = nullptr, *aHost = nullptr;
    int32_t *expertIdHost = nullptr, *aIdHost = nullptr, *rankIdHost = nullptr;
    int32_t *countHost = nullptr, *startHost = nullptr;

    uint8_t *xDev = nullptr, *aDev = nullptr;
    int32_t *expertIdDev = nullptr, *aIdDev = nullptr, *rankIdDev = nullptr;
    int32_t *countDev = nullptr, *startDev = nullptr;

    aclrtMallocHost((void **)&xHost,        xBytes);
    aclrtMallocHost((void **)&expertIdHost, expertIdBytes);
    aclrtMallocHost((void **)&aHost,        aBytes);
    aclrtMallocHost((void **)&aIdHost,      aIdBytes);
    aclrtMallocHost((void **)&rankIdHost,   aIdBytes);
    aclrtMallocHost((void **)&countHost,    expBytes);
    aclrtMallocHost((void **)&startHost,    expBytes);

    aclrtMalloc((void **)&xDev,        xBytes,        ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&expertIdDev, expertIdBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&aDev,        aBytes,        ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&aIdDev,      aIdBytes,      ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&rankIdDev,   aIdBytes,      ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&countDev,    expBytes,      ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&startDev,    expBytes,      ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile("../input/input_X.bin",         xBytes,        xHost,        xBytes);
    ReadFile("../input/input_expert_id.bin", expertIdBytes, expertIdHost, expertIdBytes);

    // Poison the device-side A / A_id / rank_id allocations so we can confirm
    // the kernel actually wrote them.
    aclrtMemset(aDev,      aBytes,   0x00, aBytes);
    aclrtMemset(aIdDev,    aIdBytes, 0xFF, aIdBytes);    // -1 sentinel for unwritten
    aclrtMemset(rankIdDev, aIdBytes, 0xFF, aIdBytes);

    aclrtMemcpy(xDev,        xBytes,        xHost,        xBytes,        ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(expertIdDev, expertIdBytes, expertIdHost, expertIdBytes, ACL_MEMCPY_HOST_TO_DEVICE);

    (void)PtoTiming::TimeKernelCallUs("scatter", stream, [&]() {
        launchScatterFp16(aDev, aIdDev, rankIdDev, countDev, startDev, xDev, expertIdDev, stream);
    });

    aclrtMemcpy(aHost,      aBytes,        aDev,      aBytes,        ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(aIdHost,    aIdBytes,      aIdDev,    aIdBytes,      ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(rankIdHost, aIdBytes,      rankIdDev, aIdBytes,      ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(countHost,  expBytes,      countDev,  expBytes,      ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(startHost,  expBytes,      startDev,  expBytes,      ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile("../output/output_A.bin",            aHost,      aBytes);
    WriteFile("../output/output_A_id.bin",         aIdHost,    aIdBytes);
    WriteFile("../output/output_rank_id.bin",      rankIdHost, aIdBytes);
    WriteFile("../output/output_expert_count.bin", countHost,  expBytes);
    WriteFile("../output/output_expert_start.bin", startHost,  expBytes);

    aclrtFree(startDev);
    aclrtFree(countDev);
    aclrtFree(rankIdDev);
    aclrtFree(aIdDev);
    aclrtFree(aDev);
    aclrtFree(expertIdDev);
    aclrtFree(xDev);
    aclrtFreeHost(startHost);
    aclrtFreeHost(countHost);
    aclrtFreeHost(rankIdHost);
    aclrtFreeHost(aIdHost);
    aclrtFreeHost(aHost);
    aclrtFreeHost(expertIdHost);
    aclrtFreeHost(xHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    bool aOk      = ValidateA(aBytes);
    bool aIdOk    = ValidateAid(aIdBytes);
    bool rankIdOk = ValidateRankId(aIdBytes);
    bool countOk  = ValidateExpertArr("../output/golden_expert_count.bin",
                                      "../output/output_expert_count.bin",
                                      "expert_count", expBytes);
    bool startOk  = ValidateExpertArr("../output/golden_expert_start.bin",
                                      "../output/output_expert_start.bin",
                                      "expert_start", expBytes);

    bool allOk = aOk && aIdOk && rankIdOk && countOk && startOk;
    if (allOk) {
        printf("test data success\n");
        printf("test success\n");
    } else {
        printf("test data failed\n");
        printf("test failed\n");
    }
    return allOk ? 0 : 1;
}
