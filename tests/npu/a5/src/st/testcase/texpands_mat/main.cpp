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
#include <cstdint>

using namespace std;
using namespace PtoTestCommon;

template <int32_t testKey>
void launchTSETVALUE(uint8_t *out, void *stream);

class TSETVALUETest : public testing::Test {
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

template <int32_t testKey, typename T>
void tsetvalue_test(uint32_t M, uint32_t N)
{
    size_t fileSize = M * N * sizeof(T);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint8_t *dstHost, *src0Host, *src1Host;
    uint8_t *dstDevice, *src0Device, *src1Device;

    aclrtMallocHost((void **)(&dstHost), fileSize);
    aclrtMallocHost((void **)(&src0Host), fileSize);
    aclrtMallocHost((void **)(&src1Host), fileSize);

    aclrtMalloc((void **)&dstDevice, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src0Device, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src1Device, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/input.bin", fileSize, src0Host, fileSize);
    ReadFile(GetGoldenDir() + "/input.bin", fileSize, src1Host, fileSize);

    aclrtMemcpy(src0Device, fileSize, src0Host, fileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src1Device, fileSize, src1Host, fileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    launchTLOADMIX<T, format, N1, N2, N3, N4, N5, WN1, WN2, WN3, WN4, WN5, BASEM, BASEK>(dstDevice, src0Device,
                                                                                         src1Device, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, fileSize, dstDevice, fileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output_z.bin", dstHost, fileSize);

    aclrtFree(dstDevice);
    aclrtFree(src0Device);
    aclrtFree(src1Device);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(src0Host);
    aclrtFreeHost(src1Host);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<T> golden(fileSize);
    std::vector<T> devFinal(fileSize);
    ReadFile(GetGoldenDir() + "/golden.bin", fileSize, golden.data(), fileSize);
    ReadFile(GetGoldenDir() + "/output_z.bin", fileSize, devFinal.data(), fileSize);

    bool ret = ResultCmp(golden, devFinal, 0.001f);

    EXPECT_TRUE(ret);
}

template <int32_t testKey, typename T>
void tsetvalue_test(uint32_t N, uint32_t C1, uint32_t H, uint32_t W, uint32_t C0)
{
    size_t fileSize = N * C1 * H * W * C0 * sizeof(T);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint8_t *dstHost, *src0Host, *src1Host;
    uint8_t *dstDevice, *src0Device, *src1Device;

    aclrtMallocHost((void **)(&dstHost), fileSize);
    aclrtMallocHost((void **)(&src0Host), fileSize);
    aclrtMallocHost((void **)(&src1Host), fileSize);

    aclrtMalloc((void **)&dstDevice, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src0Device, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src1Device, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/input.bin", fileSize, src0Host, fileSize);
    ReadFile(GetGoldenDir() + "/input.bin", fileSize, src1Host, fileSize);

    aclrtMemcpy(src0Device, fileSize, src0Host, fileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src1Device, fileSize, src1Host, fileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    launchTLOADMIX<T, format, N1, N2, N3, N4, N5, WN1, WN2, WN3, WN4, WN5, BASEM, BASEK>(dstDevice, src0Device,
                                                                                         src1Device, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, fileSize, dstDevice, fileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output_z.bin", dstHost, fileSize);

    aclrtFree(dstDevice);
    aclrtFree(src0Device);
    aclrtFree(src1Device);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(src0Host);
    aclrtFreeHost(src1Host);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<T> golden(fileSize);
    std::vector<T> devFinal(fileSize);
    ReadFile(GetGoldenDir() + "/golden.bin", fileSize, golden.data(), fileSize);
    ReadFile(GetGoldenDir() + "/output_z.bin", fileSize, devFinal.data(), fileSize);

    bool ret = ResultCmp(golden, devFinal, 0.001f);

    EXPECT_TRUE(ret);
}

TEST_F(TSETVALUETest, case1)
{
    tsetvalue_test<1, uint16_t>(128, 128);  // uint16_t represent half
}

TEST_F(TSETVALUETest, case2)
{
    tsetvalue_test<2, int16_t>(32, 64);
}

TEST_F(TSETVALUETest, case3)
{
    tsetvalue_test<3, float>(32, 32);
}

TEST_F(TSETVALUETest, case4)
{
    tsetvalue_test<4, int8_t>(32, 32);
}

TEST_F(TSETVALUETest, case5)
{
    tsetvalue_test<5, uint16_t>(256, 256);
}

TEST_F(TSETVALUETest, case6)
{
    tsetvalue_test_convtile<6, uint16_t>(1, 16, 7, 7, 16);
}

TEST_F(TSETVALUETest, case7)
{
    tsetvalue_test_convtile<7, float>(2, 10, 5, 5, 8);
}