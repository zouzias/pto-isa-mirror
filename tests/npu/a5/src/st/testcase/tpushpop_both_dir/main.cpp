#include "test_common.h"
#include "acl/acl.h"
#include <gtest/gtest.h>

using namespace std;
using namespace PtoTestCommon;

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

void LaunchScope3Incore0Incore0(float *attn_out, uint16_t *hidden_states, float *resid_out, uint16_t *wo, void *stream);


class TPushPopBothDirTest : public testing::Test {
protected:
    void SetUp() override
    {}
    void TearDown() override
    {}
};

std::string GetGoldenDir()
{
    const testing::TestInfo *testInfo = testing::UnitTest::GetInstance()->current_test_info();
    const std::string caseName = testInfo->name();
    std::string suiteName = testInfo->test_suite_name();
    std::string fullPath = "../" + suiteName + "." + caseName;
    return fullPath;
}

#define ACL_CHECK(expr)                                                                                          \
    do {                                                                                                         \
        aclError _ret = (expr);                                                                                  \
        if (_ret != ACL_SUCCESS) {                                                                               \
            std::fprintf(stderr, "[ACL ERROR] %s failed: %d (%s:%d)\n", #expr, (int)_ret, __FILE__, __LINE__); \
            return 1;                                                                                            \
        }                                                                                                        \
    } while (0)

static void printMatrix(const char *title, const std::vector<float> &data, int rows, int cols)
{
    std::printf("%s\n", title);
    for (int row = 0; row < rows; ++row) {
        std::printf("row %02d:", row);
        for (int col = 0; col < cols; ++col)
            std::printf(" %7.1f", data[static_cast<size_t>(row) * cols + col]);
        std::printf("\n");
    }
}

int TPushPopBothDirTestFunc()
{
    constexpr int Rows = 16;
    constexpr int KCols = 128;
    constexpr int OutCols = 64;
    constexpr int InitValue = -777;
    constexpr float Atol = 1e-3f;
    constexpr float Rtol = 1e-3f;

    constexpr size_t attnElems = static_cast<size_t>(Rows) * KCols;
    constexpr size_t hiddenElems = static_cast<size_t>(Rows) * OutCols;
    constexpr size_t outElems = static_cast<size_t>(Rows) * OutCols;
    constexpr size_t weightElems = static_cast<size_t>(KCols) * OutCols;

    constexpr size_t attnBytes = attnElems * sizeof(float);
    constexpr size_t hiddenBytes = hiddenElems * sizeof(uint16_t);
    constexpr size_t outBytes = outElems * sizeof(float);
    constexpr size_t weightBytes = weightElems * sizeof(uint16_t);

    const std::string goldenDir = GetGoldenDir();

    std::vector<float> hostAttn(attnElems, 0.0f);
    std::vector<uint16_t> hostHidden(hiddenElems, 0);
    std::vector<uint16_t> hostWeight(weightElems, 0);
    std::vector<float> hostOut(outElems, static_cast<float>(InitValue));
    std::vector<float> hostGolden(outElems, 0.0f);

    size_t attnFileSize = attnBytes;
    size_t hiddenFileSize = hiddenBytes;
    size_t outFileSize = outBytes;
    size_t weightFileSize = weightBytes;
    size_t goldenFileSize = outBytes;

    ReadFile(goldenDir + "/attn_out.bin", attnFileSize, hostAttn.data(), attnBytes);
    ReadFile(goldenDir + "/hidden_states.bin", hiddenFileSize, hostHidden.data(), hiddenBytes);
    ReadFile(goldenDir + "/resid_out_init.bin", outFileSize, hostOut.data(), outBytes);
    ReadFile(goldenDir + "/wo.bin", weightFileSize, hostWeight.data(), weightBytes);
    ReadFile(goldenDir + "/golden.bin", goldenFileSize, hostGolden.data(), outBytes);

    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(0));

    aclrtStream stream = nullptr;
    ACL_CHECK(aclrtCreateStream(&stream));

    float *devAttn = nullptr;
    uint16_t *devHidden = nullptr;
    float *devOut = nullptr;
    uint16_t *devWeight = nullptr;

    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&devAttn), attnBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&devHidden), hiddenBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&devOut), outBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&devWeight), weightBytes, ACL_MEM_MALLOC_HUGE_FIRST));

    ACL_CHECK(aclrtMemcpy(devAttn, attnBytes, hostAttn.data(), attnBytes, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(devHidden, hiddenBytes, hostHidden.data(), hiddenBytes, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(devOut, outBytes, hostOut.data(), outBytes, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(devWeight, weightBytes, hostWeight.data(), weightBytes, ACL_MEMCPY_HOST_TO_DEVICE));

    LaunchScope3Incore0Incore0(devAttn, devHidden, devOut, devWeight, stream);

    ACL_CHECK(aclrtSynchronizeStream(stream));
    ACL_CHECK(aclrtMemcpy(hostOut.data(), outBytes, devOut, outBytes, ACL_MEMCPY_DEVICE_TO_HOST));
    WriteFile(goldenDir + "/output.bin", hostOut.data(), outBytes);

    printMatrix("device output:", hostOut, Rows, OutCols);
    printMatrix("golden output:", hostGolden, Rows, OutCols);

    int mismatchCount = 0;
    for (int row = 0; row < Rows; ++row) {
        for (int col = 0; col < OutCols; ++col) {
            const size_t idx = static_cast<size_t>(row) * OutCols + col;
            const float got = hostOut[idx];
            const float expect = hostGolden[idx];
            const float diff = std::fabs(got - expect);
            const float limit = Atol + Rtol * std::fabs(expect);
            if (diff > limit) {
                if (mismatchCount < 16) {
                    std::fprintf(stderr,
                                 "Mismatch at (%d, %d): got %.6f, expect %.6f, diff %.6f, "
                                 "limit %.6f\n",
                                 row, col, got, expect, diff, limit);
                }
                ++mismatchCount;
            }
        }
    }

    if (mismatchCount == 0) {
        std::puts("scope3_incore_0_incore_0 test7 passed.");
    } else {
        std::fprintf(stderr, "Found %d mismatches.\n", mismatchCount);
    }

    aclrtFree(devWeight);
    aclrtFree(devOut);
    aclrtFree(devHidden);
    aclrtFree(devAttn);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
    return mismatchCount == 0 ? 0 : 1;
}

// TILE_UP_DOWN: cube result split along rows, each vector core gets upper/lower half
TEST_F(TPushPopBothDirTest, case1)
{
    EXPECT_EQ(TPushPopBothDirTestFunc(), 0);
}