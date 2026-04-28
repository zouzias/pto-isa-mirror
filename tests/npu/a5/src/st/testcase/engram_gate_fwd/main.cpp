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

class EngramGateFwdTest : public testing::Test {
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

template <typename TFloat, bool isBf16, int kHiddenSize, int kTileRows, int kTileCols, int validRows, int validCols,
           float eps, float clampValue>
void LaunchEngramGateFwd(void *output, void *x, void *k, void *v, void *weight_fused, float scalar, void *stream);

template <typename TFloat, bool isBf16, int kHiddenSize, int kTileRows, int kTileCols, int validRows, int validCols,
           float eps = 1e-20f, float clampValue = 1e-6f>
void test_engram_gate_fwd()
{
    using T = std::conditional_t<isBf16, uint16_t, float>;
    size_t tileSizeBytes = kTileRows * kTileCols * sizeof(T);
    size_t tileSizeFloatBytes = kTileRows * kTileCols * sizeof(TFloat);
    float scalar = 1.0f / sqrt(static_cast<float>(kHiddenSize));

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    void *outputHost, *xHost, *kHost, *vHost;
    void *weightHost;
    void *outputDevice, *xDevice, *kDevice, *vDevice;
    void *weightDevice;

    aclrtMallocHost(&outputHost, tileSizeBytes);
    aclrtMallocHost(&xHost, tileSizeBytes);
    aclrtMallocHost(&kHost, tileSizeBytes);
    aclrtMallocHost(&vHost, tileSizeBytes);
    aclrtMallocHost(&weightHost, tileSizeFloatBytes);

    aclrtMalloc(&outputDevice, tileSizeBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&xDevice, tileSizeBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&kDevice, tileSizeBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&vDevice, tileSizeBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&weightDevice, tileSizeFloatBytes, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/input_x.bin", tileSizeBytes, xHost, tileSizeBytes);
    ReadFile(GetGoldenDir() + "/input_k.bin", tileSizeBytes, kHost, tileSizeBytes);
    ReadFile(GetGoldenDir() + "/input_v.bin", tileSizeBytes, vHost, tileSizeBytes);
    ReadFile(GetGoldenDir() + "/input_weight.bin", tileSizeFloatBytes, weightHost, tileSizeFloatBytes);

    aclrtMemcpy(xDevice, tileSizeBytes, xHost, tileSizeBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(kDevice, tileSizeBytes, kHost, tileSizeBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(vDevice, tileSizeBytes, vHost, tileSizeBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(weightDevice, tileSizeFloatBytes, weightHost, tileSizeFloatBytes, ACL_MEMCPY_HOST_TO_DEVICE);

    LaunchEngramGateFwd<TFloat, isBf16, kHiddenSize, kTileRows, kTileCols, validRows, validCols, eps, clampValue>(
        outputDevice, xDevice, kDevice, vDevice, weightDevice, scalar, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(outputHost, tileSizeBytes, outputDevice, tileSizeBytes, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output.bin", outputHost, tileSizeBytes);

    aclrtFree(outputDevice);
    aclrtFree(xDevice);
    aclrtFree(kDevice);
    aclrtFree(vDevice);
    aclrtFree(weightDevice);

    aclrtFreeHost(outputHost);
    aclrtFreeHost(xHost);
    aclrtFreeHost(kHost);
    aclrtFreeHost(vHost);
    aclrtFreeHost(weightHost);

    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<T> golden(tileSizeBytes);
    std::vector<T> devFinal(tileSizeBytes);
    ReadFile(GetGoldenDir() + "/golden.bin", tileSizeBytes, golden.data(), tileSizeBytes);
    ReadFile(GetGoldenDir() + "/output.bin", tileSizeBytes, devFinal.data(), tileSizeBytes);

    bool ret = ResultCmp<T>(golden, devFinal, 0.001f);

    EXPECT_TRUE(ret);
}

TEST_F(EngramGateFwdTest, case_bf16_64x64_64x64)
{
    test_engram_gate_fwd<float, true, 64, 64, 64, 64, 64>();
}

TEST_F(EngramGateFwdTest, case_bf16_4096_64x64_64x64)
{
    test_engram_gate_fwd<float, true, 4096, 64, 64, 64, 64>();
}