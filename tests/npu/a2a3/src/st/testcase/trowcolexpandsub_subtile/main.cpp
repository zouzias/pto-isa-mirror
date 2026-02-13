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

class TRowColExpandSubSubtileTest : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

std::string GetGoldenDir()
{
    const testing::TestInfo *testInfo = testing::UnitTest::GetInstance()->current_test_info();
    const std::string caseName = testInfo->name();
    std::string suiteName = testInfo->test_suite_name();
    std::string fullPath = "../trowcolexpandsub_subtile/" + suiteName + "." + caseName;
    return fullPath;
}

template <typename T, int kTRows_, int kTCols_, int vRows, int vCols>
void LaunchTRowExpandSubSubtile(T *out, T *src0, T *src1, void *stream);

template <typename T, int kTRows_, int kTCols_, int vRows, int vCols>
void LaunchTColExpandSubSubtile(T *out, T *src0, T *src1, void *stream);

template <typename T, int kTRows_, int kTCols_, int vRows, int vCols>
void test_row_expand_subtile()
{
    size_t dstFileSize = kTRows_ * kTCols_ * sizeof(T);
    size_t src1FileSize = vRows * sizeof(T);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    T *dstHost, *src0Host, *src1Host;
    T *dstDevice, *src0Device, *src1Device;

    aclrtMallocHost((void **)(&dstHost), dstFileSize);
    aclrtMallocHost((void **)(&src0Host), dstFileSize);
    aclrtMallocHost((void **)(&src1Host), src1FileSize);

    aclrtMalloc((void **)&dstDevice, dstFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src0Device, dstFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src1Device, src1FileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/input1.bin", dstFileSize, src0Host, dstFileSize);
    ReadFile(GetGoldenDir() + "/input2.bin", src1FileSize, src1Host, src1FileSize);

    aclrtMemcpy(src0Device, dstFileSize, src0Host, dstFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src1Device, src1FileSize, src1Host, src1FileSize, ACL_MEMCPY_HOST_TO_DEVICE);

    LaunchTRowExpandSubSubtile<T, kTRows_, kTCols_, vRows, vCols>(dstDevice, src0Device, src1Device, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, dstFileSize, dstDevice, dstFileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output.bin", dstHost, dstFileSize);

    aclrtFree(dstDevice);
    aclrtFree(src0Device);
    aclrtFree(src1Device);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(src0Host);
    aclrtFreeHost(src1Host);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<T> golden(dstFileSize);
    std::vector<T> devFinal(dstFileSize);
    ReadFile(GetGoldenDir() + "/golden.bin", dstFileSize, golden.data(), dstFileSize);
    ReadFile(GetGoldenDir() + "/output.bin", dstFileSize, devFinal.data(), dstFileSize);

    bool ret = ResultCmp<T>(golden, devFinal, 0.001f);
    EXPECT_TRUE(ret);
}

template <typename T, int kTRows_, int kTCols_, int vRows, int vCols>
void test_col_expand_subtile()
{
    size_t dstFileSize = kTRows_ * kTCols_ * sizeof(T);
    size_t src1FileSize = vCols * sizeof(T);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    T *dstHost, *src0Host, *src1Host;
    T *dstDevice, *src0Device, *src1Device;

    aclrtMallocHost((void **)(&dstHost), dstFileSize);
    aclrtMallocHost((void **)(&src0Host), dstFileSize);
    aclrtMallocHost((void **)(&src1Host), src1FileSize);

    aclrtMalloc((void **)&dstDevice, dstFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src0Device, dstFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src1Device, src1FileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/input1.bin", dstFileSize, src0Host, dstFileSize);
    ReadFile(GetGoldenDir() + "/input2.bin", src1FileSize, src1Host, src1FileSize);

    aclrtMemcpy(src0Device, dstFileSize, src0Host, dstFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src1Device, src1FileSize, src1Host, src1FileSize, ACL_MEMCPY_HOST_TO_DEVICE);

    LaunchTColExpandSubSubtile<T, kTRows_, kTCols_, vRows, vCols>(dstDevice, src0Device, src1Device, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, dstFileSize, dstDevice, dstFileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output.bin", dstHost, dstFileSize);

    aclrtFree(dstDevice);
    aclrtFree(src0Device);
    aclrtFree(src1Device);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(src0Host);
    aclrtFreeHost(src1Host);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<T> golden(dstFileSize);
    std::vector<T> devFinal(dstFileSize);
    ReadFile(GetGoldenDir() + "/golden.bin", dstFileSize, golden.data(), dstFileSize);
    ReadFile(GetGoldenDir() + "/output.bin", dstFileSize, devFinal.data(), dstFileSize);

    bool ret = ResultCmp<T>(golden, devFinal, 0.001f);
    EXPECT_TRUE(ret);
}

TEST_F(TRowColExpandSubSubtileTest, case_row_float_128x8)
{
    test_row_expand_subtile<float, 128, 8, 128, 8>();
}

TEST_F(TRowColExpandSubSubtileTest, case_col_float_64x64)
{
    test_col_expand_subtile<float, 64, 64, 64, 64>();
}

