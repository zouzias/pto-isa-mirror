/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include "test_common.h"
#include "acl/acl.h"
#include <gtest/gtest.h>

using namespace std;
using namespace PtoTestCommon;

class TEXPM1Test : public testing::Test {
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

template <typename T, typename U, int dstRow, int dstCol, int srcRow, int srcCol, int validRow, int validCol>
void LaunchTExpM1(T *out, U *src, void *stream);

template <typename T, typename U, int dstRow, int dstCol, int srcRow, int srcCol, int validRow, int validCol>
void test_texpm1()
{
    size_t dstFileSize = dstRow * dstCol * sizeof(T);
    size_t srcFileSize = srcRow * srcCol * sizeof(U);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    T *dstHost, *dstDevice;
    U *srcHost, *srcDevice;

    aclrtMallocHost((void **)(&dstHost), dstFileSize);
    aclrtMallocHost((void **)(&srcHost), srcFileSize);
    aclrtMalloc((void **)&dstDevice, dstFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&srcDevice, srcFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/input.bin", srcFileSize, srcHost, srcFileSize);
    aclrtMemcpy(srcDevice, srcFileSize, srcHost, srcFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    LaunchTExpM1<T, U, dstRow, dstCol, srcRow, srcCol, validRow, validCol>(dstDevice, srcDevice, stream);

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

    std::vector<T> golden(dstFileSize);
    std::vector<T> devFinal(dstFileSize);
    ReadFile(GetGoldenDir() + "/golden.bin", dstFileSize, golden.data(), dstFileSize);
    ReadFile(GetGoldenDir() + "/output.bin", dstFileSize, devFinal.data(), dstFileSize);

    float eps = 0.0005f;
    if constexpr (std::is_same_v<T, float>) {
        eps = 0.00005f;
    }
    bool ret = ResultCmp(golden, devFinal, eps);

    EXPECT_TRUE(ret);
}

TEST_F(TEXPM1Test, case1)
{
    test_texpm1<float, float, 16, 64, 16, 64, 16, 64>();
}
TEST_F(TEXPM1Test, case2)
{
    test_texpm1<float, float, 1, 1024, 1, 1024, 1, 1024>();
}
TEST_F(TEXPM1Test, case3)
{
    test_texpm1<aclFloat16, aclFloat16, 32, 32, 32, 32, 32, 32>();
}
TEST_F(TEXPM1Test, case4)
{
    test_texpm1<float, aclFloat16, 32, 32, 32, 32, 32, 32>();
}
TEST_F(TEXPM1Test, case5)
{
    test_texpm1<float, int32_t, 32, 32, 32, 32, 32, 32>();
}
TEST_F(TEXPM1Test, case6)
{
    test_texpm1<float, int16_t, 32, 32, 32, 32, 32, 32>();
}
TEST_F(TEXPM1Test, case7)
{
    test_texpm1<aclFloat16, int16_t, 32, 32, 32, 32, 32, 32>();
}