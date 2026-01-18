/*
 * gtest driver for TBMM_QK (128x128x128)
 */

#include "test_common.h"
#include "acl/acl.h"
#include <gtest/gtest.h>

using namespace std;
using namespace PtoTestCommon;

class TBMMQKTest : public testing::Test {};

std::string GetGoldenDir() {
    const testing::TestInfo *testInfo = testing::UnitTest::GetInstance()->current_test_info();
    const std::string caseName = testInfo->name();
    std::string suiteName = testInfo->test_suite_name();
    std::string fullPath = "../" + suiteName + "." + caseName;
    return fullPath;
}

extern void LaunchTBMM_QK(float *out, aclFloat16 *q, aclFloat16 *k, void *stream);
// Specialized wrappers declared for explicit kernel instantiations
extern "C" void LaunchTBMM_QK_128_128_128(float *out, aclFloat16 *q, aclFloat16 *k, void *stream);
extern "C" void LaunchTBMM_QK_128_128_128_NT(float *out, aclFloat16 *q, aclFloat16 *k, void *stream);
extern "C" void LaunchTBMM_QK_256_128_64_NT(float *out, aclFloat16 *q, aclFloat16 *k, void *stream);
extern "C" void LaunchTBMM_QK_64_256_64_NT(float *out, aclFloat16 *q, aclFloat16 *k, void *stream);
extern "C" void LaunchTBMM_QK_64_128_128_NT(float *out, aclFloat16 *q, aclFloat16 *k, void *stream);
extern "C" void LaunchTBMM_QK_256_128_128_NT(float *out, aclFloat16 *q, aclFloat16 *k, void *stream);
extern "C" void LaunchTBMM_QK_128_256_64_NT(float *out, aclFloat16 *q, aclFloat16 *k, void *stream);
extern "C" void LaunchTBMM_QK_128_64_128_NT(float *out, aclFloat16 *q, aclFloat16 *k, void *stream);
extern "C" void LaunchTBMM_QK_128_128_64_NT(float *out, aclFloat16 *q, aclFloat16 *k, void *stream);
extern "C" void LaunchTBMM_QK_64_64_128_NT(float *out, aclFloat16 *q, aclFloat16 *k, void *stream);
extern "C" void LaunchTBMM_QK_128_128_128_split(float *out, aclFloat16 *q, aclFloat16 *k, void *stream);
extern "C" void LaunchTBMM_QK_128_64_128(float *out, aclFloat16 *q, aclFloat16 *k, void *stream);

template<typename T, int M, int K, int N, bool IS_NT = false>
void run_tbmm_qk() {
    size_t fullSize = M * N * sizeof(T); // Keep output as float
    size_t qSize = M * K * sizeof(aclFloat16);
    size_t kSize = K * N * sizeof(aclFloat16);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    T *outHost;
    aclFloat16 *qHost, *kHost;
    T *outDevice;
    aclFloat16 *qDevice, *kDevice;

    aclrtMallocHost((void **)(&outHost), fullSize); // Allocate output buffer
    aclrtMallocHost((void **)(&qHost), qSize);
    aclrtMallocHost((void **)(&kHost), kSize);

    aclrtMalloc((void **)&outDevice, fullSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&qDevice, qSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&kDevice, kSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/q.bin", qSize, qHost, qSize); // Read q data
    ReadFile(GetGoldenDir() + "/kt.bin", kSize, kHost, kSize);

    aclrtMemcpy(qDevice, qSize, qHost, qSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(kDevice, kSize, kHost, kSize, ACL_MEMCPY_HOST_TO_DEVICE);

    if constexpr (IS_NT && M == 128 && K == 128 && N == 128) {
        LaunchTBMM_QK_128_128_128_NT(outDevice, qDevice, kDevice, stream);
    } else if constexpr (IS_NT && M == 256 && K == 128 && N == 64) {
        LaunchTBMM_QK_256_128_64_NT(outDevice, qDevice, kDevice, stream);
    } else if constexpr (IS_NT && M == 64 && K == 256 && N == 64) {
        LaunchTBMM_QK_64_256_64_NT(outDevice, qDevice, kDevice, stream);
    } else if constexpr (IS_NT && M == 64 && K == 128 && N == 128) {
        LaunchTBMM_QK_64_128_128_NT(outDevice, qDevice, kDevice, stream);
    } else if constexpr (IS_NT && M == 256 && K == 128 && N == 128) {
        LaunchTBMM_QK_256_128_128_NT(outDevice, qDevice, kDevice, stream);
    } else if constexpr (IS_NT && M == 128 && K == 256 && N == 64) {
        LaunchTBMM_QK_128_256_64_NT(outDevice, qDevice, kDevice, stream);
    } else if constexpr (IS_NT && M == 128 && K == 64 && N == 128) {
        LaunchTBMM_QK_128_64_128_NT(outDevice, qDevice, kDevice, stream);
    } else if constexpr (IS_NT && M == 128 && K == 128 && N == 64) {
        LaunchTBMM_QK_128_128_64_NT(outDevice, qDevice, kDevice, stream);
    } else if constexpr (IS_NT && M == 64 && K == 64 && N == 128) {
        LaunchTBMM_QK_64_64_128_NT(outDevice, qDevice, kDevice, stream);
    } else if constexpr (M == 128 && K == 128 && N == 128) {
        LaunchTBMM_QK_128_128_128(outDevice, qDevice, kDevice, stream);
    }

    aclrtSynchronizeStream(stream);

    aclrtMemcpy(outHost, fullSize, outDevice, fullSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output.bin", outHost, fullSize);

    aclrtFree(outDevice);
    aclrtFree(qDevice);
    aclrtFree(kDevice);

    aclrtFreeHost(outHost); // Free host memory
    aclrtFreeHost(qHost);
    aclrtFreeHost(kHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    // compare
    std::vector<T> golden(fullSize / sizeof(T));
    std::vector<T> devFinal(fullSize / sizeof(T));
    ReadFile(GetGoldenDir() + "/golden.bin", fullSize, golden.data(), fullSize);
    ReadFile(GetGoldenDir() + "/output.bin", fullSize, devFinal.data(), fullSize);

    bool ok = ResultCmp<T>(golden, devFinal, 0.001f);
    std::cout << "ok: " << (ok ? "true" : "false") << std::endl;
    EXPECT_TRUE(ok);
}


TEST_F(TBMMQKTest, case_float_128x128x128_NT) {
    run_tbmm_qk<float, 128, 128, 128, true>();
}

TEST_F(TBMMQKTest, case_float_256x128x64_NT) {
    run_tbmm_qk<float, 256, 128, 64, true>();
}

TEST_F(TBMMQKTest, case_float_64x256x64_NT) {
    run_tbmm_qk<float, 64, 256, 64, true>();
}

TEST_F(TBMMQKTest, case_float_64x128x128_NT) {
    run_tbmm_qk<float, 64, 128, 128, true>();
}

TEST_F(TBMMQKTest, case_float_256x128x128_NT) {
    run_tbmm_qk<float, 256, 128, 128, true>();
}

TEST_F(TBMMQKTest, case_float_128x256x64_NT) {
    run_tbmm_qk<float, 128, 256, 64, true>();
}

TEST_F(TBMMQKTest, case_float_128x64x128_NT) {
    run_tbmm_qk<float, 128, 64, 128, true>();
}

TEST_F(TBMMQKTest, case_float_128x128x64_NT) {
    run_tbmm_qk<float, 128, 128, 64, true>();
}

TEST_F(TBMMQKTest, case_float_64x64x128_NT) {
    run_tbmm_qk<float, 64, 64, 128, true>();
}

