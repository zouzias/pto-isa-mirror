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

template <int32_t testKey>
void LaunchTLoadL2Hint(float* out, float* src, uint64_t* gLog, void* stream);

class TLOADL2HintMatTest : public testing::Test {
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

#define LOGSIZE 128
#define PRINTLOG 5
#define DEBUGLOG
#define MAXBLOCK 64

template <int32_t testKey>
void TestTloadL2HintMat()
{
    constexpr int gShape0 = 1;
    constexpr int gShape1 = 1;
    constexpr int gShape2 = 1;
    constexpr int gShape3 = 128;
    constexpr int gShape4 = 256;
    constexpr int gWholeShape0 = 1;
    constexpr int gWholeShape1 = 1;
    constexpr int gWholeShape2 = 1;
    constexpr int gWholeShape3 = 128;
    constexpr int gWholeShape4 = 256;

    size_t srcDataSize = static_cast<size_t>(gWholeShape0) * static_cast<size_t>(gWholeShape1) *
                         static_cast<size_t>(gWholeShape2) * static_cast<size_t>(gWholeShape3) *
                         static_cast<size_t>(gWholeShape4) * sizeof(float);
    size_t dstDataSize = static_cast<size_t>(gShape0) * static_cast<size_t>(gShape1) * static_cast<size_t>(gShape2) *
                         static_cast<size_t>(gShape3) * static_cast<size_t>(gShape4) * sizeof(float);

    aclInit(nullptr);
    aclrtSetDevice(0);

    aclrtStream stream;
    aclrtCreateStream(&stream);

    float *dstHost, *srcHost;
    float *dstDevice, *srcDevice;
    void* logDevice = nullptr;

    aclrtMallocHost((void**)(&dstHost), dstDataSize);
    aclrtMallocHost((void**)(&srcHost), srcDataSize);

    aclrtMalloc((void**)&dstDevice, dstDataSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&srcDevice, srcDataSize, ACL_MEM_MALLOC_HUGE_FIRST);
    ReadFile(GetGoldenDir() + "/input.bin", srcDataSize, srcHost, srcDataSize);
    aclrtMemset(dstDevice, dstDataSize, 0, dstDataSize);
    aclrtMemcpy(srcDevice, srcDataSize, srcHost, srcDataSize, ACL_MEMCPY_HOST_TO_DEVICE);

#ifdef DEBUGLOG
    uint64_t logHost[MAXBLOCK][LOGSIZE];
    std::fill((uint8_t*)logHost, ((uint8_t*)(logHost)) + sizeof(logHost), 0);
    aclrtMalloc((void**)&logDevice, sizeof(logHost), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemcpy(logDevice, sizeof(logHost), logHost, sizeof(logHost), ACL_MEMCPY_HOST_TO_DEVICE);
#endif

    LaunchTLoadL2Hint<testKey>(dstDevice, srcDevice, (uint64_t*)logDevice, stream);
    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, dstDataSize, dstDevice, dstDataSize, ACL_MEMCPY_DEVICE_TO_HOST);
#ifdef DEBUGLOG
    aclrtMemcpy(logHost, sizeof(logHost), logDevice, sizeof(logHost), ACL_MEMCPY_DEVICE_TO_HOST);
#endif

    WriteFile(GetGoldenDir() + "/output.bin", dstHost, dstDataSize);

    aclrtFree(dstDevice);
    aclrtFree(srcDevice);
#ifdef DEBUGLOG
    aclrtFree(logDevice);
#endif

    aclrtFreeHost(dstHost);
    aclrtFreeHost(srcHost);

    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<float> golden(dstDataSize / sizeof(float));
    std::vector<float> devFinal(dstDataSize / sizeof(float));
    ReadFile(GetGoldenDir() + "/golden.bin", dstDataSize, golden.data(), dstDataSize);
    ReadFile(GetGoldenDir() + "/output.bin", dstDataSize, devFinal.data(), dstDataSize);

    bool ret = ResultCmp<float>(golden, devFinal, 0.001f);

#ifdef DEBUGLOG
    for (int b = 0; b < 1; b++) {
        cout << "Block: " << setw(2) << b << " ";
        for (int l = 0; l < sizeof(logHost[0]) / sizeof(logHost[0][0]) && l < PRINTLOG; l++) {
            cout << hex << setfill('0') << setw(16) << logHost[b][l] << " ";
        }
        cout << dec << endl;
    }
#endif

    EXPECT_TRUE(ret);
}

TEST_F(TLOADL2HintMatTest, case_mat_float_ND_1_1_1_128_256_alloc_x5) { TestTloadL2HintMat<1>(); }

TEST_F(TLOADL2HintMatTest, case_mat_float_ND_1_1_1_128_256_notalloc_x5) { TestTloadL2HintMat<2>(); }
