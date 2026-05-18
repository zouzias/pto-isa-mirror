/**
 * main.cpp - host driver for moe_segmented_identity.
 *
 * Pattern source: kernels/automode/a2a3/moe_top1_permute/main.cpp; extended
 * to (a) read T_PADDED at runtime from ./output/t_padded.txt (the seeded
 * expert distribution determines it, so it is not a compile-time constant),
 * and (b) carry one extra GM input (expert_start).
 *
 * I/O contract (all little-endian, contiguous, no header):
 *   ../output/t_padded.txt                (single int line; written by gen_data.py)
 *   ../input/input_packed_tokens.bin      (T_PADDED * H float32)
 *   ../input/input_expert_count.bin       (kE        int32)  PADDED counts
 *   ../input/input_expert_start.bin       (kE        int32)  PADDED starts
 *   ../output/golden_packed_output.bin    (T_PADDED * H float32)
 *   ../output/output_packed_output.bin    (T_PADDED * H float32)  (this driver)
 *
 * Numerical contract: integer-valued tokens + exact +1.0 → bit-exact in
 * IEEE-754 float32; ResultCmp tolerance 0.001f is comfortably wide.
 */

#include "test_common.h"
#include "acl/acl.h"
#include "../kernel_timing.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

using namespace std;
using namespace PtoTestCommon;

template <typename T>
void launchMoeSegmentedIdentity(T *packed_output, T *packed_tokens,
                                int32_t *expert_count, int32_t *expert_start,
                                void *stream);

static int ReadTPadded()
{
    std::ifstream f("../output/t_padded.txt");
    if (!f.is_open()) {
        printf("[main] FATAL: cannot open ../output/t_padded.txt; did scripts/gen_data.py run?\n");
        std::exit(2);
    }
    int t_padded = 0;
    f >> t_padded;
    if (t_padded <= 0) {
        printf("[main] FATAL: invalid T_PADDED=%d from t_padded.txt\n", t_padded);
        std::exit(2);
    }
    return t_padded;
}

template <typename T, int kH_, int kE_>
inline bool ValidateDataResults(size_t outFileSize)
{
    std::vector<T> golden(outFileSize / sizeof(T));
    std::vector<T> devFinal(outFileSize / sizeof(T));

    ReadFile("../output/golden_packed_output.bin", outFileSize, golden.data(),   outFileSize);
    ReadFile("../output/output_packed_output.bin", outFileSize, devFinal.data(), outFileSize);

    bool ret = ResultCmp(golden, devFinal, 0.001f);
    if (ret) {
        printf("test data success\n");
    } else {
        printf("test data failed\n");
    }
    return ret;
}

template <typename T, int kH_, int kE_>
void MoeSegmentedIdentity()
{
    const int    T_padded         = ReadTPadded();
    const size_t totalElements    = static_cast<size_t>(T_padded) * kH_;
    size_t       packedTokensBytes = totalElements * sizeof(T);
    size_t       packedOutputBytes = totalElements * sizeof(T);
    size_t       expertBytes       = static_cast<size_t>(kE_) * sizeof(int32_t);

    printf("[main] T_padded=%d  H=%d  E=%d  packedBytes=%zu  expertBytes=%zu\n",
           T_padded, kH_, kE_, packedTokensBytes, expertBytes);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    T       *packedTokensHost = nullptr, *packedOutputHost = nullptr;
    int32_t *countHost = nullptr, *startHost = nullptr;
    T       *packedTokensDev  = nullptr, *packedOutputDev  = nullptr;
    int32_t *countDev  = nullptr, *startDev  = nullptr;

    aclrtMallocHost((void **)(&packedTokensHost), packedTokensBytes);
    aclrtMallocHost((void **)(&packedOutputHost), packedOutputBytes);
    aclrtMallocHost((void **)(&countHost),        expertBytes);
    aclrtMallocHost((void **)(&startHost),        expertBytes);

    aclrtMalloc((void **)&packedTokensDev, packedTokensBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&packedOutputDev, packedOutputBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&countDev,        expertBytes,       ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&startDev,        expertBytes,       ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile("../input/input_packed_tokens.bin", packedTokensBytes, packedTokensHost, packedTokensBytes);
    ReadFile("../input/input_expert_count.bin",  expertBytes,       countHost,        expertBytes);
    ReadFile("../input/input_expert_start.bin",  expertBytes,       startHost,        expertBytes);

    aclrtMemcpy(packedTokensDev, packedTokensBytes, packedTokensHost, packedTokensBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(countDev,        expertBytes,       countHost,        expertBytes,       ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(startDev,        expertBytes,       startHost,        expertBytes,       ACL_MEMCPY_HOST_TO_DEVICE);

    (void)PtoTiming::TimeKernelCallUs("moe_segmented_identity", stream, [&]() {
        launchMoeSegmentedIdentity<T>(packedOutputDev, packedTokensDev,
                                      countDev, startDev, stream);
    });
    aclrtMemcpy(packedOutputHost, packedOutputBytes, packedOutputDev, packedOutputBytes, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile("../output/output_packed_output.bin", packedOutputHost, packedOutputBytes);

    aclrtFree(startDev);
    aclrtFree(countDev);
    aclrtFree(packedOutputDev);
    aclrtFree(packedTokensDev);
    aclrtFreeHost(startHost);
    aclrtFreeHost(countHost);
    aclrtFreeHost(packedOutputHost);
    aclrtFreeHost(packedTokensHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    bool dataSuccess = ValidateDataResults<T, kH_, kE_>(packedOutputBytes);
    if (dataSuccess) {
        printf("test success\n");
    } else {
        printf("test failed\n");
    }
}

int main()
{
    constexpr int kH = 64;
    constexpr int kE = 4;
    MoeSegmentedIdentity<float, kH, kE>();
    return 0;
}
