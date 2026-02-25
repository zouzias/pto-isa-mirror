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

template <typename T, int format, int gShape0, int gShape1, int gShape2, int gShape3, int gShape4,
          int gWholeShape0, int gWholeShape1, int gWholeShape2, int gWholeShape3, int gWholeShape4>
void LaunchTTRANSConv(T *out, T *src, void *stream);

class TTRANSConvTest : public testing::Test {
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

template <typename T, int format, int gShape0, int gShape1, int gShape2, int gShape3, int gShape4,
          int gWholeShape0, int gWholeShape1, int gWholeShape2, int gWholeShape3, int gWholeShape4>
void test_ttrans()
{
    size_t srcFileSize = gWholeShape0 * gWholeShape1 * gWholeShape2 * gWholeShape3 * gWholeShape4 * sizeof(T);
    size_t dstFileSize = srcFileSize;

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
    LaunchTTRANSConv<T, format, gShape0, gShape1, gShape2, gShape3, gShape4,
                 gWholeShape0, gWholeShape1, gWholeShape2, gWholeShape3, gWholeShape4>
                 (dstDevice, srcDevice, stream);

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

    std::vector<T> golden(dstFileSize / sizeof(T));
    std::vector<T> result(dstFileSize / sizeof(T));
    ReadFile(GetGoldenDir() + "/golden.bin", dstFileSize, golden.data(), dstFileSize);
    ReadFile(GetGoldenDir() + "/output.bin", dstFileSize, result.data(), dstFileSize);

    bool ret = ResultCmp(golden, result, 0.001f);

    EXPECT_TRUE(ret);
}

TEST_F(TTRANSConvTest, float32_1_32_6_56)
{
    test_ttrans<float, 0, 1, 4, 6, 56, 8, 1, 1, 32, 6, 56>();
}

TEST_F(TTRANSConvTest, float32_1_32_2_16)
{
    test_ttrans<float, 0, 1, 2, 2, 16, 16, 1, 1, 32, 2, 16>();
}

TEST_F(TTRANSConvTest, float32_2_2_2_16_16)
{
    test_ttrans<float, 1, 2, 2, 16, 2, 16, 2, 2, 2, 16, 16>();
}
