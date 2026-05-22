#include "test_common.h"
#include <pto/pto-inst.hpp>
#include <gtest/gtest.h>

using namespace std;
using namespace PtoTestCommon;

// Updated Launch Signatures: gWholeShape tracking parameters completely removed
template <typename T, int gShape0, int gShape1, int gShape2, int gShape3>
void LaunchTTRANSConv_NCHW2NC1HWC0(T *out, T *src, void *stream);

class TTRANSConvTest : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

std::string GetGoldenDir()
{
    const testing::TestInfo *testInfo = testing::UnitTest::GetInstance()->current_test_info();
    const std::string caseName = testInfo->name();
    std::string suiteName = testInfo->test_suite_name();
    std::string fullPath = "../" + suiteName + "." + caseName;
    return fullPath;
}

// Updated Test Wrapper for Conv: Direct execution shape allocation and launching
template <typename T, int gShape0, int gShape1, int gShape2, int gShape3>
void test_ttrans_NCHW2NC1HWC0()
{
    size_t srcFileSize = gShape0 * gShape1 * gShape2 * gShape3 * sizeof(T);
    size_t C0 = 32 / sizeof(T);
    size_t C1 = (gShape1 + C0 - 1 ) / C0;
    size_t dstFileSize = gShape0 * C1 * gShape2 * gShape3 * C0 * sizeof(T);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    T *dstHost, *srcHost;
    T *dstDevice, *srcDevice;

    aclrtMallocHost((void **)(&dstHost), dstFileSize);
    aclrtMallocHost((void **)(&srcHost), srcFileSize);

    aclrtMalloc((void **)&dstDevice, dstFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&srcDevice, srcFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/input.bin", srcFileSize, srcHost, srcFileSize);

    aclrtMemcpy(srcDevice, srcFileSize, srcHost, srcFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    
    LaunchTTRANSConv_NCHW2NC1HWC0<T, gShape0, gShape1, gShape2, gShape3>(dstDevice, srcDevice, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, dstFileSize, dstDevice, dstFileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output.bin", dstHost, dstFileSize);

    aclrtFree(dstDevice);
    aclrtFree(srcDevice);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(srcHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<T> golden(dstFileSize / sizeof(T), 0);
    std::vector<T> result(dstFileSize / sizeof(T), 0);
    ReadFile(GetGoldenDir() + "/golden.bin", dstFileSize, golden.data(), dstFileSize);
    ReadFile(GetGoldenDir() + "/output.bin", dstFileSize, result.data(), dstFileSize);

    bool ret = ResultCmp(golden, result, 0.001f);

    EXPECT_TRUE(ret);
}


TEST_F(TTRANSConvTest, NCHW2NC1HWC0_1)
{
    test_ttrans_NCHW2NC1HWC0<float, 5, 4, 3, 8>();
}

TEST_F(TTRANSConvTest, NCHW2NC1HWC0_2)
{
    test_ttrans_NCHW2NC1HWC0<int32_t, 5, 14, 13, 16>();
}

TEST_F(TTRANSConvTest, NCHW2NC1HWC0_3)
{
    test_ttrans_NCHW2NC1HWC0<uint16_t, 1, 11, 13, 16>();
}

TEST_F(TTRANSConvTest, NCHW2NC1HWC0_4)
{
    test_ttrans_NCHW2NC1HWC0<int32_t, 4, 32, 3, 7>();
}

TEST_F(TTRANSConvTest, NCHW2NC1HWC0_5)
{
    test_ttrans_NCHW2NC1HWC0<int8_t, 4, 32, 3, 7>();
}