/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to License for details. You may not use this file except in compliance with License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include "test_common.h"
#include "acl/acl.h"
#include <gtest/gtest.h>

using namespace std;
using namespace PtoTestCommon;

class TAddSubMulDivTest : public testing::Test {
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

template <typename T, int kTRows_, int kTCols_, int vRows, int vCols>
void LaunchTAddSubMulDiv(T *out, T *src0, T *src1, T *src2, T *src3, T *src4, void *stream);

template <typename T, int kTRows_, int kTCols_, int vRows, int vCols>
void test_tTAddSubMulDiv()
{
    size_t fileSize = kTRows_ * kTCols_ * sizeof(T);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    T *dstHost, *src0Host, *src1Host, *src2Host, *src3Host, *src4Host;
    T *dstDevice, *src0Device, *src1Device, *src2Device, *src3Device, *src4Device;

    aclrtMallocHost((void **)(&dstHost), fileSize);
    aclrtMallocHost((void **)(&src0Host), fileSize);
    aclrtMallocHost((void **)(&src1Host), fileSize);
    aclrtMallocHost((void **)(&src2Host), fileSize);
    aclrtMallocHost((void **)(&src3Host), fileSize);
    aclrtMallocHost((void **)(&src4Host), fileSize);

    aclrtMalloc((void **)&dstDevice, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src0Device, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src1Device, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src2Device, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src3Device, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src4Device, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/input1.bin", fileSize, src0Host, fileSize);
    ReadFile(GetGoldenDir() + "/input2.bin", fileSize, src1Host, fileSize);
    ReadFile(GetGoldenDir() + "/input3.bin", fileSize, src2Host, fileSize);
    ReadFile(GetGoldenDir() + "/input4.bin", fileSize, src3Host, fileSize);
    ReadFile(GetGoldenDir() + "/input5.bin", fileSize, src4Host, fileSize);

    aclrtMemcpy(src0Device, fileSize, src0Host, fileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src1Device, fileSize, src1Host, fileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src2Device, fileSize, src2Host, fileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src3Device, fileSize, src3Host, fileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src4Device, fileSize, src4Host, fileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    LaunchTAddSubMulDiv<T, kTRows_, kTCols_, vRows, vCols>(dstDevice, src0Device, src1Device, src2Device, src3Device, src4Device, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, fileSize, dstDevice, fileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output.bin", dstHost, fileSize);

    aclrtFree(dstDevice);
    aclrtFree(src0Device);
    aclrtFree(src1Device);
    aclrtFree(src2Device);
    aclrtFree(src3Device);
    aclrtFree(src4Device);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(src0Host);
    aclrtFreeHost(src1Host);
    aclrtFreeHost(src2Host);
    aclrtFreeHost(src3Host);
    aclrtFreeHost(src4Host);

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

TEST_F(TAddSubMulDivTest, case_float_64x64_64x64)
{
    test_tTAddSubMulDiv<float, 64, 64, 64, 64>();
}

TEST_F(TAddSubMulDivTest, case_int32_64x64_64x64)
{
    test_tTAddSubMulDiv<int32_t, 64, 64, 64, 64>();
}

TEST_F(TAddSubMulDivTest, case_int16_64x64_64x64)
{
    test_tTAddSubMulDiv<int16_t, 64, 64, 64, 64>();
}

TEST_F(TAddSubMulDivTest, case_half_16x256_16x256)
{
    test_tTAddSubMulDiv<aclFloat16, 16, 256, 16, 256>();
}