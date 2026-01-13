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

template <typename T, uint32_t dstRow, uint32_t dstCol, uint32_t src1Row, uint32_t src1Col>
void LaunchTColExpandDiv(T *out, T *src0, T *src1, void *stream);

class TCOLEXPANDDIVTest : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

static std::string GetGoldenDir()
{
    const testing::TestInfo *testInfo = testing::UnitTest::GetInstance()->current_test_info();
    const std::string caseName = testInfo->name();
    const std::string suiteName = testInfo->test_suite_name();
    return "../" + suiteName + "." + caseName;
}

template <typename T, uint32_t dstRow, uint32_t dstCol, uint32_t src1Row, uint32_t src1Col>
static void test_tcolexpanddiv()
{
    size_t input1Bytes = src1Row * src1Col * sizeof(T);
    size_t outputBytes = dstRow * dstCol * sizeof(T);

    aclInit(nullptr);
    aclrtSetDevice(GetDeviceId());
    aclrtStream stream;
    aclrtCreateStream(&stream);

    T *dstHost, *src0Host, *src1Host;
    T *dstDevice, *src0Device, *src1Device;

    aclrtMallocHost((void **)(&dstHost), outputBytes);
    aclrtMallocHost((void **)(&src0Host), outputBytes);
    aclrtMallocHost((void **)(&src1Host), input1Bytes);

    aclrtMalloc((void **)&dstDevice, outputBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src0Device, outputBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src1Device, input1Bytes, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/input0.bin", outputBytes, src0Host, outputBytes);
    ReadFile(GetGoldenDir() + "/input1.bin", input1Bytes, src1Host, input1Bytes);

    aclrtMemcpy(src0Device, outputBytes, src0Host, outputBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src1Device, input1Bytes, src1Host, input1Bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    LaunchTColExpandDiv<T, dstRow, dstCol, src1Row, src1Col>(dstDevice, src0Device, src1Device, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, outputBytes, dstDevice, outputBytes, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output.bin", dstHost, outputBytes);

    aclrtFree(dstDevice);
    aclrtFree(src0Device);
    aclrtFree(src1Device);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(src0Host);
    aclrtFreeHost(src1Host);

    aclrtDestroyStream(stream);
    aclrtResetDevice(GetDeviceId());
    aclFinalize();

    std::vector<T> golden(outputBytes);
    std::vector<T> devFinal(outputBytes);
    ReadFile(GetGoldenDir() + "/golden.bin", outputBytes, golden.data(), outputBytes);
    ReadFile(GetGoldenDir() + "/output.bin", outputBytes, devFinal.data(), outputBytes);
    EXPECT_TRUE(ResultCmp(golden, devFinal, 0.001f));
}

TEST_F(TCOLEXPANDDIVTest, case_fp32_32_64_1_64)
{
    test_tcolexpanddiv<float, 32, 64, 1, 64>();
}

TEST_F(TCOLEXPANDDIVTest, case_fp32_8_32_1_32)
{
    test_tcolexpanddiv<float, 8, 32, 1, 32>();
}

TEST_F(TCOLEXPANDDIVTest, case_fp16_16_64_1_64)
{
    test_tcolexpanddiv<aclFloat16, 16, 64, 1, 64>();
}

TEST_F(TCOLEXPANDDIVTest, case_fp16_4_128_1_128)
{
    test_tcolexpanddiv<aclFloat16, 4, 128, 1, 128>();
}
