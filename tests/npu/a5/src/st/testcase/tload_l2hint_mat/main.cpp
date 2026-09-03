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
void LaunchTLoadL2HintMat(uint8_t* out, uint8_t* src, uint64_t* gLog, void* stream);

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
#define PRINTLOG 4
#define DEBUGLOG
#define MAXBLOCK 64

template <int32_t testKey>
void TestTloadL2HintMat()
{
    constexpr int M = 128;
    constexpr int K = 256;
    size_t aFileSize = static_cast<size_t>(M) * static_cast<size_t>(K) * sizeof(float);
    size_t cFileSize = aFileSize;

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint8_t *dstHost, *srcHost;
    uint8_t *dstDevice, *srcDevice;
    void* logDevice = nullptr;

    aclrtMallocHost((void**)(&dstHost), cFileSize);
    aclrtMallocHost((void**)(&srcHost), aFileSize);

    aclrtMalloc((void**)&dstDevice, cFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&srcDevice, aFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/x1_gm.bin", aFileSize, srcHost, aFileSize);
    aclrtMemset(dstDevice, cFileSize, 0, cFileSize);
    aclrtMemcpy(srcDevice, aFileSize, srcHost, aFileSize, ACL_MEMCPY_HOST_TO_DEVICE);

#ifdef DEBUGLOG
    uint64_t logHost[MAXBLOCK][LOGSIZE];
    std::fill((uint8_t*)logHost, ((uint8_t*)(logHost)) + sizeof(logHost), 0);
    aclrtMalloc((void**)&logDevice, sizeof(logHost), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemcpy(logDevice, sizeof(logHost), logHost, sizeof(logHost), ACL_MEMCPY_HOST_TO_DEVICE);
#endif

    LaunchTLoadL2HintMat<testKey>(dstDevice, srcDevice, (uint64_t*)logDevice, stream);
    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, cFileSize, dstDevice, cFileSize, ACL_MEMCPY_DEVICE_TO_HOST);
#ifdef DEBUGLOG
    aclrtMemcpy(logHost, sizeof(logHost), logDevice, sizeof(logHost), ACL_MEMCPY_DEVICE_TO_HOST);
#endif

    WriteFile(GetGoldenDir() + "/output_z.bin", dstHost, cFileSize);

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

    std::vector<float> golden(cFileSize / sizeof(float));
    std::vector<float> devFinal(cFileSize / sizeof(float));
    ReadFile(GetGoldenDir() + "/golden.bin", cFileSize, golden.data(), cFileSize);
    ReadFile(GetGoldenDir() + "/output_z.bin", cFileSize, devFinal.data(), cFileSize);

    bool ret = ResultCmp(golden, devFinal, 0.001f);

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
