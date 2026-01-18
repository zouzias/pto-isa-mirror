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

// Macro to declare kernel wrappers
#define DECLARE_KERNEL_WRAPPER(M, K, N, VARIANT) \
    extern "C" void LaunchTBMM_QK_##M##_##K##_##N##_##VARIANT(float *out, aclFloat16 *q, aclFloat16 *k, void *stream);

// NT variant wrappers (outer K loop - default)
DECLARE_KERNEL_WRAPPER(128, 128, 128, NT)
DECLARE_KERNEL_WRAPPER(256, 128, 64, NT)
DECLARE_KERNEL_WRAPPER(64, 256, 64, NT)
DECLARE_KERNEL_WRAPPER(64, 128, 128, NT)
DECLARE_KERNEL_WRAPPER(256, 128, 128, NT)
DECLARE_KERNEL_WRAPPER(128, 256, 64, NT)
DECLARE_KERNEL_WRAPPER(128, 128, 64, NT)

// NT_inner variant wrappers (inner K loop)
DECLARE_KERNEL_WRAPPER(128, 128, 128, NT_inner)
DECLARE_KERNEL_WRAPPER(256, 128, 64, NT_inner)
DECLARE_KERNEL_WRAPPER(64, 256, 64, NT_inner)
DECLARE_KERNEL_WRAPPER(64, 128, 128, NT_inner)
DECLARE_KERNEL_WRAPPER(256, 128, 128, NT_inner)
DECLARE_KERNEL_WRAPPER(128, 256, 64, NT_inner)
DECLARE_KERNEL_WRAPPER(128, 128, 64, NT_inner)

// TN variant wrappers
DECLARE_KERNEL_WRAPPER(128, 128, 128, TN)
DECLARE_KERNEL_WRAPPER(256, 128, 64, TN)
DECLARE_KERNEL_WRAPPER(64, 256, 64, TN)
DECLARE_KERNEL_WRAPPER(64, 128, 128, TN)
DECLARE_KERNEL_WRAPPER(256, 128, 128, TN)
DECLARE_KERNEL_WRAPPER(128, 256, 64, TN)
DECLARE_KERNEL_WRAPPER(128, 128, 64, TN)

// NN variant wrappers
DECLARE_KERNEL_WRAPPER(128, 128, 128, NN)
DECLARE_KERNEL_WRAPPER(256, 128, 64, NN)
DECLARE_KERNEL_WRAPPER(64, 256, 64, NN)
DECLARE_KERNEL_WRAPPER(64, 128, 128, NN)
DECLARE_KERNEL_WRAPPER(256, 128, 128, NN)
DECLARE_KERNEL_WRAPPER(128, 256, 64, NN)
DECLARE_KERNEL_WRAPPER(128, 128, 64, NN)

// TT variant wrappers
DECLARE_KERNEL_WRAPPER(128, 128, 128, TT)
DECLARE_KERNEL_WRAPPER(256, 128, 64, TT)
DECLARE_KERNEL_WRAPPER(64, 256, 64, TT)
DECLARE_KERNEL_WRAPPER(64, 128, 128, TT)
DECLARE_KERNEL_WRAPPER(256, 128, 128, TT)
DECLARE_KERNEL_WRAPPER(128, 256, 64, TT)
DECLARE_KERNEL_WRAPPER(128, 128, 64, TT)

#undef DECLARE_KERNEL_WRAPPER

template<typename T, int M, int K, int N, bool IS_NT = false, bool IS_TN = false, bool IS_NN = false, bool IS_NT_INNER = false, bool IS_TT = false>
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

    std::string qFile = (IS_TN || IS_TT) ? "/qt.bin" : "/q.bin";
    std::string kFile = (IS_NT || IS_NT_INNER || IS_TT) ? "/kt.bin" : "/k.bin";
    ReadFile(GetGoldenDir() + qFile, qSize, qHost, qSize); // Read q data
    ReadFile(GetGoldenDir() + kFile, kSize, kHost, kSize);

    aclrtMemcpy(qDevice, qSize, qHost, qSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(kDevice, kSize, kHost, kSize, ACL_MEMCPY_HOST_TO_DEVICE);

    // Macro to generate kernel dispatch logic
    #define DISPATCH_KERNEL(M_VAL, K_VAL, N_VAL, VARIANT, FLAG) \
        if constexpr (FLAG && M == M_VAL && K == K_VAL && N == N_VAL) { \
            LaunchTBMM_QK_##M_VAL##_##K_VAL##_##N_VAL##_##VARIANT(outDevice, qDevice, kDevice, stream); \
        } else

    DISPATCH_KERNEL(128, 128, 128, NT, IS_NT)
    DISPATCH_KERNEL(256, 128, 64, NT, IS_NT)
    DISPATCH_KERNEL(64, 256, 64, NT, IS_NT)
    DISPATCH_KERNEL(64, 128, 128, NT, IS_NT)
    DISPATCH_KERNEL(256, 128, 128, NT, IS_NT)
    DISPATCH_KERNEL(128, 256, 64, NT, IS_NT)
    DISPATCH_KERNEL(128, 128, 64, NT, IS_NT)
    DISPATCH_KERNEL(128, 128, 128, NT_inner, IS_NT_INNER)
    DISPATCH_KERNEL(256, 128, 64, NT_inner, IS_NT_INNER)
    DISPATCH_KERNEL(64, 256, 64, NT_inner, IS_NT_INNER)
    DISPATCH_KERNEL(64, 128, 128, NT_inner, IS_NT_INNER)
    DISPATCH_KERNEL(256, 128, 128, NT_inner, IS_NT_INNER)
    DISPATCH_KERNEL(128, 256, 64, NT_inner, IS_NT_INNER)
    DISPATCH_KERNEL(128, 128, 64, NT_inner, IS_NT_INNER)
    DISPATCH_KERNEL(128, 128, 128, TN, IS_TN)
    DISPATCH_KERNEL(256, 128, 64, TN, IS_TN)
    DISPATCH_KERNEL(64, 256, 64, TN, IS_TN)
    DISPATCH_KERNEL(64, 128, 128, TN, IS_TN)
    DISPATCH_KERNEL(256, 128, 128, TN, IS_TN)
    DISPATCH_KERNEL(128, 256, 64, TN, IS_TN)
    DISPATCH_KERNEL(128, 128, 64, TN, IS_TN)
    DISPATCH_KERNEL(128, 128, 128, NN, IS_NN)
    DISPATCH_KERNEL(256, 128, 64, NN, IS_NN)
    DISPATCH_KERNEL(64, 256, 64, NN, IS_NN)
    DISPATCH_KERNEL(64, 128, 128, NN, IS_NN)
    DISPATCH_KERNEL(256, 128, 128, NN, IS_NN)
    DISPATCH_KERNEL(128, 256, 64, NN, IS_NN)
    DISPATCH_KERNEL(128, 128, 64, NN, IS_NN)
    DISPATCH_KERNEL(128, 128, 128, TT, IS_TT)
    DISPATCH_KERNEL(256, 128, 64, TT, IS_TT)
    DISPATCH_KERNEL(64, 256, 64, TT, IS_TT)
    DISPATCH_KERNEL(64, 128, 128, TT, IS_TT)
    DISPATCH_KERNEL(256, 128, 128, TT, IS_TT)
    DISPATCH_KERNEL(128, 256, 64, TT, IS_TT)
    DISPATCH_KERNEL(128, 128, 64, TT, IS_TT)
    { /* Default: do nothing */ }

    #undef DISPATCH_KERNEL

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

// Macro to generate test cases
#define DEFINE_TEST_CASE(M, K, N, VARIANT, ...) \
    TEST_F(TBMMQKTest, case_float_##M##x##K##x##N##_##VARIANT) { \
        run_tbmm_qk<float, M, K, N, ##__VA_ARGS__>(); \
    }

// NT variant test cases (outer K loop - default)
DEFINE_TEST_CASE(128, 128, 128, NT, true)
DEFINE_TEST_CASE(256, 128, 64, NT, true)
DEFINE_TEST_CASE(64, 256, 64, NT, true)
DEFINE_TEST_CASE(64, 128, 128, NT, true)
DEFINE_TEST_CASE(256, 128, 128, NT, true)
DEFINE_TEST_CASE(128, 256, 64, NT, true)
DEFINE_TEST_CASE(128, 128, 64, NT, true)

// NT_inner variant test cases (inner K loop)
DEFINE_TEST_CASE(128, 128, 128, NT_inner, false, false, false, true)
DEFINE_TEST_CASE(256, 128, 64, NT_inner, false, false, false, true)
DEFINE_TEST_CASE(64, 256, 64, NT_inner, false, false, false, true)
DEFINE_TEST_CASE(64, 128, 128, NT_inner, false, false, false, true)
DEFINE_TEST_CASE(256, 128, 128, NT_inner, false, false, false, true)
DEFINE_TEST_CASE(128, 256, 64, NT_inner, false, false, false, true)
DEFINE_TEST_CASE(128, 128, 64, NT_inner, false, false, false, true)

// TN variant test cases
DEFINE_TEST_CASE(128, 128, 128, TN, false, true)
DEFINE_TEST_CASE(256, 128, 64, TN, false, true)
DEFINE_TEST_CASE(64, 256, 64, TN, false, true)
DEFINE_TEST_CASE(64, 128, 128, TN, false, true)
DEFINE_TEST_CASE(256, 128, 128, TN, false, true)
DEFINE_TEST_CASE(128, 256, 64, TN, false, true)
DEFINE_TEST_CASE(128, 128, 64, TN, false, true)

// NN variant test cases
DEFINE_TEST_CASE(128, 128, 128, NN, false, false, true)
DEFINE_TEST_CASE(256, 128, 64, NN, false, false, true)
DEFINE_TEST_CASE(64, 256, 64, NN, false, false, true)
DEFINE_TEST_CASE(64, 128, 128, NN, false, false, true)
DEFINE_TEST_CASE(256, 128, 128, NN, false, false, true)
DEFINE_TEST_CASE(128, 256, 64, NN, false, false, true)
DEFINE_TEST_CASE(128, 128, 64, NN, false, false, true)

// TT variant test cases
DEFINE_TEST_CASE(128, 128, 128, TT, false, false, false, false, true)
DEFINE_TEST_CASE(256, 128, 64, TT, false, false, false, false, true)
DEFINE_TEST_CASE(64, 256, 64, TT, false, false, false, false, true)
DEFINE_TEST_CASE(64, 128, 128, TT, false, false, false, false, true)
DEFINE_TEST_CASE(256, 128, 128, TT, false, false, false, false, true)
DEFINE_TEST_CASE(128, 256, 64, TT, false, false, false, false, true)
DEFINE_TEST_CASE(128, 128, 64, TT, false, false, false, false, true)

#undef DEFINE_TEST_CASE