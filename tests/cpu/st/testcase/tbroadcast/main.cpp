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

template <int kTRows_, int kTCols_, int kTNumProc_>
void LaunchTBroadcast(float *out, float *src, void *stream);

template <int kTRows_, int kTCols_, int kTNumProc_>
void test_tbroadcast()
{
    const size_t tileBytesSrc = kTRows_ * kTCols_ * sizeof(float);
    const size_t tileBytesDst = kTNumProc_ * kTRows_ * kTCols_ * sizeof(float);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    float *dstHost, *srcHost;
    float *dstDevice, *srcDevice;

    aclrtMallocHost((void **)(&dstHost), tileBytesDst);
    aclrtMallocHost((void **)(&srcHost), tileBytesSrc);

    aclrtMalloc((void **)&dstDevice, tileBytesDst, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&srcDevice, tileBytesSrc, ACL_MEM_MALLOC_HUGE_FIRST);

    size_t tileSizeSrc = tileBytesSrc;
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/input.bin", tileSizeSrc, srcHost, tileBytesSrc));
    aclrtMemcpy(srcDevice, tileBytesSrc, srcHost, tileBytesSrc, ACL_MEMCPY_HOST_TO_DEVICE);
    LaunchTBroadcast<kTRows_, kTCols_, kTNumProc_>(dstDevice, srcDevice, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, tileBytesDst, dstDevice, tileBytesDst, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output.bin", dstHost, tileBytesDst);

    aclrtFree(dstDevice);
    aclrtFree(srcDevice);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(srcHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<float> golden(tileBytesDst / sizeof(float));
    std::vector<float> devFinal(tileBytesDst / sizeof(float));
    size_t tileSizeDst = tileBytesDst;
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/golden.bin", tileSizeDst, golden.data(), tileBytesDst));
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/output.bin", tileSizeDst, devFinal.data(), tileBytesDst));
    EXPECT_TRUE(ResultCmp<float>(golden, devFinal, 0.001f));
}

TEST_F(TBROADCASTTest, case_float_16x16_16x16_16x16_2proc)
{
    test_tbroadcast<16, 16, 2>();
}
