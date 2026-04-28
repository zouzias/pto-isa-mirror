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

class EngramFusedWeightTest : public testing::Test {
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

template <typename T, bool isBf16, int Rows, int Cols>
void LaunchEngramFusedWeight(void *out, void *src0, void *src1, void *stream);

template <typename T, bool isBf16, int Rows, int Cols>
void test_engram_fused_weight()
{
    using U = std::conditional_t<isBf16, uint16_t, float>;
    size_t fileSizeDst = Rows * Cols * sizeof(T);
    size_t fileSizeSrc = Rows * Cols * sizeof(U);
    
    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);
    
    void *dstHost, *src0Host, *src1Host;
    
    aclrtMallocHost(&dstHost, fileSizeDst);
    aclrtMallocHost(&src0Host, fileSizeSrc);
    aclrtMallocHost(&src1Host, fileSizeSrc);
    
    void *dstDevice, *src0Device, *src1Device;
    
    aclrtMalloc(&dstDevice, fileSizeDst, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&src0Device, fileSizeSrc, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&src1Device, fileSizeSrc, ACL_MEM_MALLOC_HUGE_FIRST);
    
    ReadFile(GetGoldenDir() + "/input1.bin", fileSizeSrc, src0Host, fileSizeSrc);
    ReadFile(GetGoldenDir() + "/input2.bin", fileSizeSrc, src1Host, fileSizeSrc);
    
    aclrtMemcpy(src0Device, fileSizeSrc, src0Host, fileSizeSrc, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src1Device, fileSizeSrc, src1Host, fileSizeSrc, ACL_MEMCPY_HOST_TO_DEVICE);
    
    LaunchEngramFusedWeight<T, isBf16, Rows, Cols>(dstDevice, src0Device, src1Device, stream);
    
    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, fileSizeDst, dstDevice, fileSizeDst, ACL_MEMCPY_DEVICE_TO_HOST);
    
    WriteFile(GetGoldenDir() + "/output.bin", dstHost, fileSizeDst);
    
    aclrtFree(dstDevice);
    aclrtFree(src0Device);
    aclrtFree(src1Device);
    
    aclrtFreeHost(dstHost);
    aclrtFreeHost(src0Host);
    aclrtFreeHost(src1Host);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
    
    std::vector<T> golden(fileSizeDst);
    std::vector<T> devFinal(fileSizeDst);
    ReadFile(GetGoldenDir() + "/golden.bin", fileSizeDst, golden.data(), fileSizeDst);
    ReadFile(GetGoldenDir() + "/output.bin", fileSizeDst, devFinal.data(), fileSizeDst);
    
    bool ret = ResultCmp<T>(golden, devFinal, 0.001f);
    
    EXPECT_TRUE(ret);
}

TEST_F(EngramFusedWeightTest, case_bf16_float_4x4096)
{
    test_engram_fused_weight<float, true, 4, 4096>();
}

// TEST_F(EngramFusedWeightTest, case_bf16_float_4x7168)
// {
//     test_engram_fused_weight<float, true, 4, 7168>();
// }
// 
// TEST_F(EngramFusedWeightTest, case_float_float_4x4096)
// {
//     test_engram_fused_weight<float, false, 4, 4096>();
// }