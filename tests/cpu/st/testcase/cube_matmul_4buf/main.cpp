/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
CANN Open Software License Agreement Version 2.0
*/

#include "test_common.h"
#include <pto/pto-inst.hpp>
#include <gtest/gtest.h>

using namespace std;
using namespace PtoTestCommon;

void LaunchCubeMatmul4Buf(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template <typename T_A, typename T_B, typename T_C>
void LaunchCubeMatmul4BufPreload(T_A *a, T_B *b, T_C *c, void *stream);

constexpr uint32_t M = 32;
constexpr uint32_t K = 1024;
constexpr uint32_t N = 256;

class CubeMatmul4BufTest : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

class CubeMatmul4BufPreloadTest : public testing::Test {
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

TEST_F(CubeMatmul4BufTest, case_f16_32x1024_1024x256)
{
    size_t aFileSize = M * K * sizeof(uint16_t);
    size_t bFileSize = K * N * sizeof(uint16_t);
    size_t cFileSize = M * N * sizeof(float);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint8_t *dstHost, *src0Host, *src1Host;
    uint8_t *dstDevice, *src0Device, *src1Device;

    aclrtMallocHost((void **)(&dstHost), cFileSize);
    aclrtMallocHost((void **)(&src0Host), aFileSize);
    aclrtMallocHost((void **)(&src1Host), bFileSize);

    aclrtMalloc((void **)&dstDevice, cFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src0Device, aFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src1Device, bFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/A_gm.bin", aFileSize, src0Host, aFileSize));
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/B_gm.bin", bFileSize, src1Host, bFileSize));

    aclrtMemcpy(src0Device, aFileSize, src0Host, aFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src1Device, bFileSize, src1Host, bFileSize, ACL_MEMCPY_HOST_TO_DEVICE);

    LaunchCubeMatmul4Buf(dstDevice, src0Device, src1Device, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, cFileSize, dstDevice, cFileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output_C.bin", dstHost, cFileSize);

    aclrtFree(dstDevice);
    aclrtFree(src0Device);
    aclrtFree(src1Device);
    aclrtFreeHost(dstHost);
    aclrtFreeHost(src0Host);
    aclrtFreeHost(src1Host);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<float> golden(M * N);
    std::vector<float> result(M * N);
    
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/golden.bin", cFileSize, golden.data(), cFileSize));
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/output_C.bin", cFileSize, result.data(), cFileSize));

    bool ret = ResultCmp(golden, result, 0.01f);
    EXPECT_TRUE(ret);
}

TEST_F(CubeMatmul4BufPreloadTest, case_f16_32x1024_1024x256_preload)
{
    size_t aFileSize = M * K * sizeof(uint16_t);
    size_t bFileSize = K * N * sizeof(uint16_t);
    size_t cFileSize = M * N * sizeof(float);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    half *src0Device, *src1Device;
    float *dstDevice;
    uint8_t *dstHost, *src0Host, *src1Host;

    aclrtMallocHost((void **)(&dstHost), cFileSize);
    aclrtMallocHost((void **)(&src0Host), aFileSize);
    aclrtMallocHost((void **)(&src1Host), bFileSize);

    aclrtMalloc((void **)&dstDevice, cFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src0Device, aFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src1Device, bFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    // Reuse the same golden data directory
    CHECK_RESULT_GTEST(ReadFile("../CubeMatmul4BufTest.case_f16_32x1024_1024x256/A_gm.bin", aFileSize, src0Host, aFileSize));
    CHECK_RESULT_GTEST(ReadFile("../CubeMatmul4BufTest.case_f16_32x1024_1024x256/B_gm.bin", bFileSize, src1Host, bFileSize));

    aclrtMemcpy(src0Device, aFileSize, src0Host, aFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src1Device, bFileSize, src1Host, bFileSize, ACL_MEMCPY_HOST_TO_DEVICE);

    LaunchCubeMatmul4BufPreload<half, half, float>(src0Device, src1Device, dstDevice, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, cFileSize, dstDevice, cFileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    system(("mkdir -p " + GetGoldenDir()).c_str()); WriteFile(GetGoldenDir() + "/output_C_preload.bin", dstHost, cFileSize);

    aclrtFree(dstDevice);
    aclrtFree(src0Device);
    aclrtFree(src1Device);
    aclrtFreeHost(dstHost);
    aclrtFreeHost(src0Host);
    aclrtFreeHost(src1Host);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<float> golden(M * N);
    std::vector<float> result(M * N);
    
    CHECK_RESULT_GTEST(ReadFile("../CubeMatmul4BufTest.case_f16_32x1024_1024x256/golden.bin", cFileSize, golden.data(), cFileSize));
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/output_C_preload.bin", cFileSize, result.data(), cFileSize));

    bool ret = ResultCmp(golden, result, 0.01f);
    EXPECT_TRUE(ret);
}
