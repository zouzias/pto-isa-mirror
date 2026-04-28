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

class EngramGateBwdTest : public testing::Test {
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

template <bool isBf16, int hidden_size, int kTRows_, int kTCols_>
void launchEngramGateBwd(
    void *grad_x, void *grad_k, void *grad_v, void *grad_w,
    void *grad_out, void *x, void *k, void *v, void *w,
    void *gate, void *rstd_x, void *rstd_k, void *raw_dot,
    float scalar, float clamp_value, void *stream);

template <bool isBf16, int hidden_size>
void test_engram_gate_bwd()
{
    using T = std::conditional_t<isBf16, uint16_t, float>;
    constexpr int kTRows_ = 1;
    constexpr int kTCols_ = hidden_size;
    constexpr float scalar = 1.0f / sqrt(static_cast<float>(hidden_size));
    constexpr float clamp_value = 1e-6f;
    
    size_t dataSize = hidden_size * sizeof(T);
    size_t floatDataSize = hidden_size * sizeof(float);
    size_t scalarSize = sizeof(float);
    
    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);
    
    void *gradXHost, *gradKHost, *gradVHost, *gradOutHost, *xHost, *kHost, *vHost;
    void *gradWHost, *wHost, *gateHost, *rstdXHost, *rstdKHost, *rawDotHost;
    void *gradXDevice, *gradKDevice, *gradVDevice, *gradOutDevice, *xDevice, *kDevice, *vDevice;
    void *gradWDevice, *wDevice, *gateDevice, *rstdXDevice, *rstdKDevice, *rawDotDevice;
    
    aclrtMallocHost(&gradXHost, dataSize);
    aclrtMallocHost(&gradKHost, dataSize);
    aclrtMallocHost(&gradVHost, dataSize);
    aclrtMallocHost(&gradOutHost, dataSize);
    aclrtMallocHost(&xHost, dataSize);
    aclrtMallocHost(&kHost, dataSize);
    aclrtMallocHost(&vHost, dataSize);
    aclrtMallocHost(&gradWHost, floatDataSize);
    aclrtMallocHost(&wHost, floatDataSize);
    aclrtMallocHost(&gateHost, scalarSize);
    aclrtMallocHost(&rstdXHost, scalarSize);
    aclrtMallocHost(&rstdKHost, scalarSize);
    aclrtMallocHost(&rawDotHost, scalarSize);
    
    aclrtMalloc(&gradXDevice, dataSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&gradKDevice, dataSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&gradVDevice, dataSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&gradOutDevice, dataSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&xDevice, dataSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&kDevice, dataSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&vDevice, dataSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&gradWDevice, floatDataSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&wDevice, floatDataSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&gateDevice, scalarSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&rstdXDevice, scalarSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&rstdKDevice, scalarSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&rawDotDevice, scalarSize, ACL_MEM_MALLOC_HUGE_FIRST);
    
    ReadFile(GetGoldenDir() + "/grad_out.bin", dataSize, gradOutHost, dataSize);
    ReadFile(GetGoldenDir() + "/x.bin", dataSize, xHost, dataSize);
    ReadFile(GetGoldenDir() + "/k.bin", dataSize, kHost, dataSize);
    ReadFile(GetGoldenDir() + "/v.bin", dataSize, vHost, dataSize);
    ReadFile(GetGoldenDir() + "/w.bin", floatDataSize, wHost, floatDataSize);
    ReadFile(GetGoldenDir() + "/gate.bin", scalarSize, gateHost, scalarSize);
    ReadFile(GetGoldenDir() + "/rstd_x.bin", scalarSize, rstdXHost, scalarSize);
    ReadFile(GetGoldenDir() + "/rstd_k.bin", scalarSize, rstdKHost, scalarSize);
    ReadFile(GetGoldenDir() + "/raw_dot.bin", scalarSize, rawDotHost, scalarSize);
    
    aclrtMemcpy(gradOutDevice, dataSize, gradOutHost, dataSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(xDevice, dataSize, xHost, dataSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(kDevice, dataSize, kHost, dataSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(vDevice, dataSize, vHost, dataSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(wDevice, floatDataSize, wHost, floatDataSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(gateDevice, scalarSize, gateHost, scalarSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(rstdXDevice, scalarSize, rstdXHost, scalarSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(rstdKDevice, scalarSize, rstdKHost, scalarSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(rawDotDevice, scalarSize, rawDotHost, scalarSize, ACL_MEMCPY_HOST_TO_DEVICE);
    
    launchEngramGateBwd<isBf16, hidden_size, kTRows_, kTCols_>(
        gradXDevice, gradKDevice, gradVDevice, gradWDevice,
        gradOutDevice, xDevice, kDevice, vDevice, wDevice,
        gateDevice, rstdXDevice, rstdKDevice, rawDotDevice,
        scalar, clamp_value, stream);
    
    aclrtSynchronizeStream(stream);
    
    aclrtMemcpy(gradXHost, dataSize, gradXDevice, dataSize, ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(gradKHost, dataSize, gradKDevice, dataSize, ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(gradVHost, dataSize, gradVDevice, dataSize, ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(gradWHost, floatDataSize, gradWDevice, floatDataSize, ACL_MEMCPY_DEVICE_TO_HOST);
    
    WriteFile(GetGoldenDir() + "/output_grad_x.bin", gradXHost, dataSize);
    WriteFile(GetGoldenDir() + "/output_grad_k.bin", gradKHost, dataSize);
    WriteFile(GetGoldenDir() + "/output_grad_v.bin", gradVHost, dataSize);
    WriteFile(GetGoldenDir() + "/output_grad_w.bin", gradWHost, floatDataSize);
    
    aclrtFree(gradXDevice);
    aclrtFree(gradKDevice);
    aclrtFree(gradVDevice);
    aclrtFree(gradOutDevice);
    aclrtFree(xDevice);
    aclrtFree(kDevice);
    aclrtFree(vDevice);
    aclrtFree(gradWDevice);
    aclrtFree(wDevice);
    aclrtFree(gateDevice);
    aclrtFree(rstdXDevice);
    aclrtFree(rstdKDevice);
    aclrtFree(rawDotDevice);
    
    aclrtFreeHost(gradXHost);
    aclrtFreeHost(gradKHost);
    aclrtFreeHost(gradVHost);
    aclrtFreeHost(gradOutHost);
    aclrtFreeHost(xHost);
    aclrtFreeHost(kHost);
    aclrtFreeHost(vHost);
    aclrtFreeHost(gradWHost);
    aclrtFreeHost(wHost);
    aclrtFreeHost(gateHost);
    aclrtFreeHost(rstdXHost);
    aclrtFreeHost(rstdKHost);
    aclrtFreeHost(rawDotHost);
    
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
    
    std::vector<T> goldenGradX(dataSize);
    std::vector<T> goldenGradK(dataSize);
    std::vector<T> goldenGradV(dataSize);
    std::vector<float> goldenGradW(floatDataSize);
    std::vector<T> outputGradX(dataSize);
    std::vector<T> outputGradK(dataSize);
    std::vector<T> outputGradV(dataSize);
    std::vector<float> outputGradW(floatDataSize);
    
    ReadFile(GetGoldenDir() + "/golden_grad_x.bin", dataSize, goldenGradX.data(), dataSize);
    ReadFile(GetGoldenDir() + "/golden_grad_k.bin", dataSize, goldenGradK.data(), dataSize);
    ReadFile(GetGoldenDir() + "/golden_grad_v.bin", dataSize, goldenGradV.data(), dataSize);
    ReadFile(GetGoldenDir() + "/golden_grad_w.bin", floatDataSize, goldenGradW.data(), floatDataSize);
    ReadFile(GetGoldenDir() + "/output_grad_x.bin", dataSize, outputGradX.data(), dataSize);
    ReadFile(GetGoldenDir() + "/output_grad_k.bin", dataSize, outputGradK.data(), dataSize);
    ReadFile(GetGoldenDir() + "/output_grad_v.bin", dataSize, outputGradV.data(), dataSize);
    ReadFile(GetGoldenDir() + "/output_grad_w.bin", floatDataSize, outputGradW.data(), floatDataSize);
    
    bool retGradX = ResultCmp<T>(goldenGradX, outputGradX, 0.001f);
    bool retGradK = ResultCmp<T>(goldenGradK, outputGradK, 0.001f);
    bool retGradV = ResultCmp<T>(goldenGradV, outputGradV, 0.001f);
    bool retGradW = ResultCmp<float>(goldenGradW, outputGradW, 0.001f);
    
    EXPECT_TRUE(retGradX);
    EXPECT_TRUE(retGradK);
    EXPECT_TRUE(retGradV);
    EXPECT_TRUE(retGradW);
}

TEST_F(EngramGateBwdTest, case_float_4096)
{
    test_engram_gate_bwd<false, 4096>();
}

TEST_F(EngramGateBwdTest, case_bfloat16_4096)
{
    test_engram_gate_bwd<true, 4096>();
}