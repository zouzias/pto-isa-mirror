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
#include <pto/pto-inst.hpp>
#include <gtest/gtest.h>

using namespace std;
using namespace PtoTestCommon;

class TBROADCASTTest : public testing::Test {
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
    return "../" + suiteName + "." + caseName;
}

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_>
void LaunchTBroadcast(T *dst0, T *dst1, T *src, void *stream);

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_>
void test_tbroadcast()
{
    size_t tileSize = kTRows_ * kTCols_ * sizeof(T);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    T *dst0Host, *dst1Host, *srcHost;
    T *dst0Device, *dst1Device, *srcDevice;

    aclrtMallocHost((void **)(&dst0Host), tileSize);
    aclrtMallocHost((void **)(&dst1Host), tileSize);
    aclrtMallocHost((void **)(&srcHost), tileSize);

    aclrtMalloc((void **)&dst0Device, tileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&dst1Device, tileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&srcDevice, tileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/input.bin", tileSize, srcHost, tileSize));
    aclrtMemcpy(srcDevice, tileSize, srcHost, tileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    LaunchTBroadcast<T, kGRows_, kGCols_, kTRows_, kTCols_>(dst0Device, dst1Device, srcDevice, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dst0Host, tileSize, dst0Device, tileSize, ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(dst1Host, tileSize, dst1Device, tileSize, ACL_MEMCPY_DEVICE_TO_HOST);
    WriteFile(GetGoldenDir() + "/output0.bin", dst0Host, tileSize);
    WriteFile(GetGoldenDir() + "/output1.bin", dst1Host, tileSize);

    aclrtFree(dst0Device);
    aclrtFree(dst1Device);
    aclrtFree(srcDevice);

    aclrtFreeHost(dst0Host);
    aclrtFreeHost(dst1Host);
    aclrtFreeHost(srcHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<T> golden(tileSize / sizeof(T));
    std::vector<T> devFinal0(tileSize / sizeof(T));
    std::vector<T> devFinal1(tileSize / sizeof(T));
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/golden.bin", tileSize, golden.data(), tileSize));
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/output0.bin", tileSize, devFinal0.data(), tileSize));
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/output1.bin", tileSize, devFinal1.data(), tileSize));
    EXPECT_TRUE(ResultCmp<T>(golden, devFinal0, 0.001f));
    EXPECT_TRUE(ResultCmp<T>(golden, devFinal1, 0.001f));
}

TEST_F(TBROADCASTTest, case_float_16x16_16x16_16x16_2proc)
{
    test_tbroadcast<float, 16, 16, 16, 16>();
}