/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
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

class TEXPANDSTest : public testing::Test {
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

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_, int kVRows_, int kVCols_, int padValueType,
          bool isBf16>
void LaunchTExpandS(void *out, void *scalar, void *stream);

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_, int kVRows_, int kVCols_, int padValueType,
          bool isBf16 = false>
void test_texpands()
{
    size_t tSize = sizeof(T);
    size_t fileSize = kGRows_ * kGCols_ * sizeof(T);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    T *dstHost;
    T *dstDevice;
    T *scalarHost;
    T *scalarDevice;

    aclrtMallocHost((void **)(&dstHost), fileSize);
    aclrtMallocHost((void **)(&scalarHost), tSize);
    aclrtMalloc((void **)&dstDevice, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&scalarDevice, tSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/scalar.bin", tSize, scalarHost, tSize);
    aclrtMemcpy(scalarDevice, tSize, scalarHost, tSize, ACL_MEMCPY_HOST_TO_DEVICE);

    LaunchTExpandS<T, kGRows_, kGCols_, kTRows_, kTCols_, kVRows_, kVCols_, padValueType, isBf16>(dstDevice,
                                                                                                  scalarDevice, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, fileSize, dstDevice, fileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output.bin", dstHost, fileSize);

    aclrtFree(dstDevice);

    aclrtFreeHost(dstHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<T> golden(fileSize);
    std::vector<T> devFinal(fileSize);
    ReadFile(GetGoldenDir() + "/golden.bin", fileSize, golden.data(), fileSize);
    ReadFile(GetGoldenDir() + "/output.bin", fileSize, devFinal.data(), fileSize);

    bool ret = ResultCmp<T>(golden, devFinal, 0.001f);

    EXPECT_TRUE(ret);
}

TEST_F(TEXPANDSTest, case_float_32x512_32x512_32x512_PAD_VALUE_NULL)
{
    test_texpands<float, 32, 512, 32, 512, 32, 512, PAD_VALUE_NULL>();
}
TEST_F(TEXPANDSTest, case_float_512x32_512x32_512x32_PAD_VALUE_NULL)
{
    test_texpands<float, 512, 32, 512, 32, 512, 32, PAD_VALUE_NULL>();
}
TEST_F(TEXPANDSTest, case_float_128x128_128x128_128x128_PAD_VALUE_NULL)
{
    test_texpands<float, 128, 128, 128, 128, 128, 128, PAD_VALUE_NULL>();
}