/**
 * main.cpp — host driver for DeepSeek-V4 scatter.
 *
 * I/O contract (all little-endian, contiguous, no header):
 *   ./input/input_X.bin              T * DIM                        bfloat16
 *   ./input/input_expert_id.bin      T * N_ACTIVATED                int32
 *   ./output/golden_A.bin            (T*N_ACTIVATED + 16) * DIM     bfloat16
 *   ./output/golden_A_id.bin         (T*N_ACTIVATED + 16)           int32
 *   ./output/golden_rank_id.bin      (T*N_ACTIVATED + 16)           int32
 *   ./output/golden_expert_count.bin N_ROUTED                       int32
 *   ./output/golden_expert_start.bin N_ROUTED                       int32
 *   ./output/output_*.bin            (kernel-emitted; same shapes)
 *
 * Validation compares only the first T*N_ACTIVATED rows of A and A_id;
 * the trailing 16-row overspill pad is ignored.
 */

#include "test_common.h"
#include "acl/acl.h"
#include "../../../kernel_timing.h"
#include "generated_cases.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

using namespace std;
using namespace PtoTestCommon;

extern "C" void launch_scatter(uint8_t *A, int32_t *A_id, int32_t *rank_id,
                               int32_t *expert_count, int32_t *expert_start,
                               uint8_t *X, int32_t *expert_id,
                               void *stream);

namespace {
constexpr int kT          = kDsmoeT;
constexpr int kDim        = kDsmoeDim;
constexpr int kNRouted    = kDsmoeNRouted;
constexpr int kNActivated = kDsmoeNActivated;

constexpr int kPackedRows   = kT * kNActivated;
constexpr int kOverspillPad = 16;
constexpr int kAlloc        = kPackedRows + kOverspillPad;

constexpr size_t kBf16Bytes = 2;
}  // namespace

int main()
{
    size_t xBytes        = static_cast<size_t>(kT)     * kDim         * kBf16Bytes;
    size_t expertIdBytes = static_cast<size_t>(kT)     * kNActivated  * sizeof(int32_t);
    size_t aBytes        = static_cast<size_t>(kAlloc) * kDim         * kBf16Bytes;
    size_t aIdBytes      = static_cast<size_t>(kAlloc)                * sizeof(int32_t);
    size_t expBytes      = static_cast<size_t>(kNRouted)              * sizeof(int32_t);

    printf("[scatter] T=%d DIM=%d N_ROUTED=%d N_ACTIVATED=%d kAlloc=%d\n",
           kT, kDim, kNRouted, kNActivated, kAlloc);

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

    ReadFile("./input/input_X.bin",         xBytes,        xHost,        xBytes);
    ReadFile("./input/input_expert_id.bin", expertIdBytes, expertIdHost, expertIdBytes);

    // Poison the device-side packed outputs to confirm the kernel wrote them.
    aclrtMemset(aDev,      aBytes,   0x00, aBytes);
    aclrtMemset(aIdDev,    aIdBytes, 0xFF, aIdBytes);     // -1 sentinel
    aclrtMemset(rankIdDev, aIdBytes, 0xFF, aIdBytes);

    aclrtMemcpy(xDev,        xBytes,        xHost,        xBytes,        ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(expertIdDev, expertIdBytes, expertIdHost, expertIdBytes, ACL_MEMCPY_HOST_TO_DEVICE);

    (void)PtoTiming::TimeKernelCallUs("scatter", stream, [&]() {
        launch_scatter(aDev, aIdDev, rankIdDev, countDev, startDev,
                       xDev, expertIdDev, stream);
    });

    aclrtMemcpy(aHost,      aBytes,   aDev,      aBytes,   ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(aIdHost,    aIdBytes, aIdDev,    aIdBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(rankIdHost, aIdBytes, rankIdDev, aIdBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(countHost,  expBytes, countDev,  expBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(startHost,  expBytes, startDev,  expBytes, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile("./output/output_A.bin",            aHost,      aBytes);
    WriteFile("./output/output_A_id.bin",         aIdHost,    aIdBytes);
    WriteFile("./output/output_rank_id.bin",      rankIdHost, aIdBytes);
    WriteFile("./output/output_expert_count.bin", countHost,  expBytes);
    WriteFile("./output/output_expert_start.bin", startHost,  expBytes);

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

    // Validate A: byte-exact over the first kPackedRows*kDim BF16 elements.
    std::vector<uint8_t> goldA(aBytes);
    std::vector<uint8_t> outA (aBytes);
    ReadFile("./output/golden_A.bin", aBytes, goldA.data(), aBytes);
    ReadFile("./output/output_A.bin", aBytes, outA.data(),  aBytes);
    size_t validBytes = static_cast<size_t>(kPackedRows) * kDim * kBf16Bytes;
    bool aOk = (std::memcmp(goldA.data(), outA.data(), validBytes) == 0);
    printf("A             : %s\n", aOk ? "success" : "FAILED");

    // Validate A_id / rank_id over the first kPackedRows entries.
    auto validateInt = [&](const char *gp, const char *op,
                           const char *label) {
        std::vector<int32_t> g(aIdBytes / sizeof(int32_t));
        std::vector<int32_t> o(aIdBytes / sizeof(int32_t));
        ReadFile(gp, aIdBytes, g.data(), aIdBytes);
        ReadFile(op, aIdBytes, o.data(), aIdBytes);
        bool ok = true;
        for (int i = 0; i < kPackedRows; ++i) {
            if (g[i] != o[i]) { ok = false; break; }
        }
        printf("%-14s: %s\n", label, ok ? "success" : "FAILED");
        return ok;
    };

    bool aidOk  = validateInt("./output/golden_A_id.bin",
                              "./output/output_A_id.bin", "A_id");
    bool rkOk   = validateInt("./output/golden_rank_id.bin",
                              "./output/output_rank_id.bin", "rank_id");

    // Validate expert_count, expert_start over kNRouted entries.
    auto validateExp = [&](const char *gp, const char *op,
                           const char *label) {
        std::vector<int32_t> g(expBytes / sizeof(int32_t));
        std::vector<int32_t> o(expBytes / sizeof(int32_t));
        ReadFile(gp, expBytes, g.data(), expBytes);
        ReadFile(op, expBytes, o.data(), expBytes);
        bool ok = (g == o);
        printf("%-14s: %s\n", label, ok ? "success" : "FAILED");
        return ok;
    };
    bool cntOk = validateExp("./output/golden_expert_count.bin",
                             "./output/output_expert_count.bin", "expert_count");
    bool stOk  = validateExp("./output/golden_expert_start.bin",
                             "./output/output_expert_start.bin", "expert_start");

    bool ok = aOk && aidOk && rkOk && cntOk && stOk;
    if (ok) {
        printf("test data success\n");
    } else {
        printf("test data failed\n");
    }
    return ok ? 0 : 1;
}
