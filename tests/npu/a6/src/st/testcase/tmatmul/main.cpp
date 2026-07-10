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

template <int32_t tilingKey>
void LaunchTMATMUL(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);

class TMATMULTest : public testing::Test {
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

template <typename OutType, typename AType, typename BType, int32_t key>
void tmatmul_test(uint32_t m, uint32_t k, uint32_t n)
{
    size_t aFileSize = m * k * sizeof(AType);
    size_t bFileSize = k * n * sizeof(BType);
    size_t cFileSize = m * n * sizeof(OutType);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint8_t *dstHost, *src0Host, *src1Host;
    uint8_t *dstDevice, *src0Device, *src1Device;

    aclrtMallocHost((void **)(&dstHost), cFileSize);
    aclrtMallocHost((void **)(&src0Host), aFileSize);
    aclrtMallocHost((void **)(&src1Host), bFileSize);

    aclrtMalloc((void **)&dstDevice, cFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src0Device, aFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src1Device, bFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/x1_gm.bin", aFileSize, src0Host, aFileSize);
    ReadFile(GetGoldenDir() + "/x2_gm.bin", bFileSize, src1Host, bFileSize);

    aclrtMemcpy(src0Device, aFileSize, src0Host, aFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src1Device, bFileSize, src1Host, bFileSize, ACL_MEMCPY_HOST_TO_DEVICE);

    LaunchTMATMUL<key>(dstDevice, src0Device, src1Device, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, cFileSize, dstDevice, cFileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output.bin", dstHost, cFileSize);

    aclrtFree(dstDevice);
    aclrtFree(src0Device);
    aclrtFree(src1Device);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(src0Host);
    aclrtFreeHost(src1Host);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<OutType> golden(cFileSize / sizeof(OutType));
    std::vector<OutType> devFinal(cFileSize / sizeof(OutType));
    ReadFile(GetGoldenDir() + "/golden.bin", cFileSize, golden.data(), cFileSize);
    ReadFile(GetGoldenDir() + "/output.bin", cFileSize, devFinal.data(), cFileSize);

    bool ret = ResultCmp(golden, devFinal, 0.001f);
    EXPECT_TRUE(ret);
}

TEST_F(TMATMULTest, case_fp16_fp16_to_fp32_31x96x47)
{
    tmatmul_test<float, uint16_t, uint16_t, 1>(31, 96, 47);
}

TEST_F(TMATMULTest, case_int8_int8_to_int32_65x90x89)
{
    tmatmul_test<int32_t, int8_t, int8_t, 2>(65, 90, 89);
}

TEST_F(TMATMULTest, case_fp32_fp32_to_fp32_16x32x64)
{
    tmatmul_test<float, float, float, 3>(16, 32, 64);
}

TEST_F(TMATMULTest, case_fp16_fp16_to_fp32_1x256x64)
{
    tmatmul_test<float, uint16_t, uint16_t, 4>(1, 256, 64);
}
