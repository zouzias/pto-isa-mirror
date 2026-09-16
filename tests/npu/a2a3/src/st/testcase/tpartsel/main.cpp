/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <cstring>

#include "test_common.h"
#include "acl/acl.h"
#include <gtest/gtest.h>

using namespace std;
using namespace PtoTestCommon;

template <typename T, int Rows, int Cols, int ValidRows, int ValidCols>
void LaunchTPartSel(T* out, uint8_t* mask, T* src0, T* src1, void* stream);

class TPARTSELTest : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

std::string GetGoldenDir()
{
    const testing::TestInfo* testInfo = testing::UnitTest::GetInstance()->current_test_info();
    const std::string caseName = testInfo->name();
    std::string suiteName = testInfo->test_suite_name();
    std::string fullPath = "../" + suiteName + "." + caseName;
    return fullPath;
}

template <typename T, int Rows, int Cols, int ValidRows, int ValidCols>
void test_tpartsel()
{
    size_t fileSize = ValidRows * ValidCols * sizeof(T);
    size_t maskFileSize = ValidRows * ((ValidCols + 7) / 8) * sizeof(uint8_t);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    T *dstHost, *src0Host, *src1Host;
    uint8_t* maskHost;
    T *dstDevice, *src0Device, *src1Device;
    uint8_t* maskDevice;

    aclrtMallocHost((void**)(&dstHost), fileSize);
    aclrtMallocHost((void**)(&maskHost), maskFileSize);
    aclrtMallocHost((void**)(&src0Host), fileSize);
    aclrtMallocHost((void**)(&src1Host), fileSize);

    aclrtMalloc((void**)&dstDevice, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&maskDevice, maskFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&src0Device, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&src1Device, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/input0.bin", fileSize, src0Host, fileSize));
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/input1.bin", fileSize, src1Host, fileSize));
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/mask.bin", maskFileSize, maskHost, maskFileSize));

    aclrtMemcpy(src0Device, fileSize, src0Host, fileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src1Device, fileSize, src1Host, fileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(maskDevice, maskFileSize, maskHost, maskFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    LaunchTPartSel<T, Rows, Cols, ValidRows, ValidCols>(dstDevice, maskDevice, src0Device, src1Device, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, fileSize, dstDevice, fileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output.bin", dstHost, fileSize);

    aclrtFree(dstDevice);
    aclrtFree(maskDevice);
    aclrtFree(src0Device);
    aclrtFree(src1Device);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(maskHost);
    aclrtFreeHost(src0Host);
    aclrtFreeHost(src1Host);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<T> golden(fileSize / sizeof(T));
    std::vector<T> devFinal(fileSize / sizeof(T));
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/golden.bin", fileSize, golden.data(), fileSize));
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/output.bin", fileSize, devFinal.data(), fileSize));

    bool ret = std::memcmp(golden.data(), devFinal.data(), fileSize) == 0;

    EXPECT_TRUE(ret);
}

TEST_F(TPARTSELTest, float_full) { test_tpartsel<float, 4, 128, 4, 128>(); }
TEST_F(TPARTSELTest, float_tail) { test_tpartsel<float, 4, 192, 3, 131>(); }
TEST_F(TPARTSELTest, half_full) { test_tpartsel<aclFloat16, 4, 128, 4, 128>(); }
TEST_F(TPARTSELTest, half_tail) { test_tpartsel<aclFloat16, 4, 192, 3, 131>(); }
TEST_F(TPARTSELTest, int32_tail) { test_tpartsel<int32_t, 4, 64, 3, 37>(); }
TEST_F(TPARTSELTest, int16_tail) { test_tpartsel<int16_t, 4, 64, 3, 37>(); }
