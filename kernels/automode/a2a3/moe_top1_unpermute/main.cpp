/**
 * main.cpp - host driver for moe_top1_unpermute.
 *
 * Pattern source: kernels/automode/a2a3/moe_top1_permute/main.cpp (extended
 * to two GM inputs + one GM output, all bare-pointer); shape mirrors that
 * milestone (T = 256, H = 64, float32 tokens, int32 metadata).
 *
 * I/O contract (all little-endian, contiguous, no header):
 *   ../input/input_packed_output.bin     (kT * kH float32)
 *   ../input/input_token_to_packed.bin   (kT      int32)
 *   ../output/golden_output.bin          (kT * kH float32) (gen_data.py)
 *   ../output/output_output.bin          (kT * kH float32) (this driver)
 *
 * Numerical contract: gen_data.py uses integer-valued tokens + an exact +1.0
 * to build packed_output, and the unpermute is a pure gather (no math), so
 * the result is bit-exact under IEEE-754 float32; ResultCmp tolerance 0.001f
 * is comfortably wide.
 */

#include "test_common.h"
#include "acl/acl.h"

#include <cstdint>
#include <cstdio>
#include <vector>

using namespace std;
using namespace PtoTestCommon;

template <typename T>
void launchMoeTop1Unpermute(T *output, T *packed_output,
                            int32_t *token_to_packed, void *stream);

template <typename T, int kT_, int kH_>
inline bool ValidateDataResults(size_t outFileSize)
{
    std::vector<T> golden(outFileSize / sizeof(T));
    std::vector<T> devFinal(outFileSize / sizeof(T));

    ReadFile("../output/golden_output.bin", outFileSize, golden.data(),   outFileSize);
    ReadFile("../output/output_output.bin", outFileSize, devFinal.data(), outFileSize);

    bool ret = ResultCmp(golden, devFinal, 0.001f);
    if (ret) {
        printf("test data success\n");
    } else {
        printf("test data failed\n");
    }
    return ret;
}

template <typename T, int kT_, int kH_>
void MoeTop1Unpermute()
{
    constexpr size_t totalElements = static_cast<size_t>(kT_) * kH_;
    size_t outputBytes       = totalElements * sizeof(T);
    size_t packedOutputBytes = totalElements * sizeof(T);
    size_t tokIdxBytes       = static_cast<size_t>(kT_) * sizeof(int32_t);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    T       *packedOutputHost = nullptr, *outputHost = nullptr;
    int32_t *ttopHost = nullptr;
    T       *packedOutputDev  = nullptr, *outputDev  = nullptr;
    int32_t *ttopDev  = nullptr;

    aclrtMallocHost((void **)(&packedOutputHost), packedOutputBytes);
    aclrtMallocHost((void **)(&outputHost),       outputBytes);
    aclrtMallocHost((void **)(&ttopHost),         tokIdxBytes);

    aclrtMalloc((void **)&packedOutputDev, packedOutputBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&outputDev,       outputBytes,       ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&ttopDev,         tokIdxBytes,       ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile("../input/input_packed_output.bin",   packedOutputBytes, packedOutputHost, packedOutputBytes);
    ReadFile("../input/input_token_to_packed.bin", tokIdxBytes,       ttopHost,         tokIdxBytes);

    aclrtMemcpy(packedOutputDev, packedOutputBytes, packedOutputHost, packedOutputBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(ttopDev,         tokIdxBytes,       ttopHost,         tokIdxBytes,       ACL_MEMCPY_HOST_TO_DEVICE);

    launchMoeTop1Unpermute<T>(outputDev, packedOutputDev, ttopDev, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(outputHost, outputBytes, outputDev, outputBytes, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile("../output/output_output.bin", outputHost, outputBytes);

    aclrtFree(ttopDev);
    aclrtFree(outputDev);
    aclrtFree(packedOutputDev);
    aclrtFreeHost(ttopHost);
    aclrtFreeHost(outputHost);
    aclrtFreeHost(packedOutputHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    bool dataSuccess = ValidateDataResults<T, kT_, kH_>(outputBytes);
    if (dataSuccess) {
        printf("test success\n");
    } else {
        printf("test failed\n");
    }
}

int main()
{
    constexpr int kT = 256;
    constexpr int kH = 64;
    MoeTop1Unpermute<float, kT, kH>();
    return 0;
}
