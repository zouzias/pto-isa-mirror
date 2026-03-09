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

using namespace PtoTestCommon;
#define ND_LAYOUT 0
// QK computation testcase
// Q: (S0, H) in ND layout (row-major)
// K: (H, S1) in DN layout (K^T stored column-major)  
// QK: (S0, S1) in ND layout (row-major)

template <int S0, int H, int S1, int CUBE_S0, int CUBE_S1, int TILE_S1>
void LaunchTQK(uint16_t *q, uint16_t *k, float *qk, void *stream);

class TQKTest : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

std::string GetGoldenDir()
{
    const testing::TestInfo *testInfo = testing::UnitTest::GetInstance()->current_test_info();
    const std::string caseName = testInfo->name();
    std::string suiteName = testInfo->test_suite_name();
    std::string fullPath = "../" + suiteName + "." + caseName;
    return fullPath;
}

static void tqk_test_case(int s1)
{
    constexpr int S0 = 128;
    constexpr int H = 128;
    constexpr int CUBE_S0 = 128;
    constexpr int CUBE_S1 = 128;
    constexpr int TILE_S1 = 128;

    const int S1 = s1;

    size_t qBytes = static_cast<size_t>(S0) * static_cast<size_t>(H) * sizeof(uint16_t);
    size_t kBytes = static_cast<size_t>(H) * static_cast<size_t>(S1) * sizeof(uint16_t);
    size_t qkBytes = static_cast<size_t>(S0) * static_cast<size_t>(S1) * sizeof(float);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint16_t *qHost = nullptr;
    uint16_t *kHost = nullptr;
    float *qkHost = nullptr;

    uint16_t *qDev = nullptr;
    uint16_t *kDev = nullptr;
    float *qkDev = nullptr;

    aclrtMallocHost((void **)(&qHost), qBytes);
    aclrtMallocHost((void **)(&kHost), kBytes);
    aclrtMallocHost((void **)(&qkHost), qkBytes);

    aclrtMalloc((void **)&qDev, qBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&kDev, kBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&qkDev, qkBytes, ACL_MEM_MALLOC_HUGE_FIRST);

    size_t fileSize = 0;
    ReadFile(GetGoldenDir() + "/q.bin", fileSize, qHost, qBytes);
    ReadFile(GetGoldenDir() + "/k_t.bin", fileSize, kHost, kBytes);

    aclrtMemcpy(qDev, qBytes, qHost, qBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(kDev, kBytes, kHost, kBytes, ACL_MEMCPY_HOST_TO_DEVICE);

    if (S1 == 128) {
        LaunchTQK<S0, H, 128, CUBE_S0, CUBE_S1, TILE_S1>(qDev, kDev, qkDev, stream);
    } else if (S1 == 256) {
        LaunchTQK<S0, H, 256, CUBE_S0, CUBE_S1, TILE_S1>(qDev, kDev, qkDev, stream);
    } else if (S1 == 512) {
        LaunchTQK<S0, H, 512, CUBE_S0, CUBE_S1, TILE_S1>(qDev, kDev, qkDev, stream);
    }

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(qkHost, qkBytes, qkDev, qkBytes, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output_z.bin", qkHost, qkBytes);

    aclrtFree(qDev);
    aclrtFree(kDev);
    aclrtFree(qkDev);

    aclrtFreeHost(qHost);
    aclrtFreeHost(kHost);
    aclrtFreeHost(qkHost);

    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<float> golden(qkBytes / sizeof(float));
    std::vector<float> devFinal(qkBytes / sizeof(float));
#if ND_LAYOUT
    ReadFile(GetGoldenDir() + "/golden_nd.bin", qkBytes, golden.data(), qkBytes);
#else
    ReadFile(GetGoldenDir() + "/golden_dn.bin", qkBytes, golden.data(), qkBytes);
#endif
    ReadFile(GetGoldenDir() + "/output_z.bin", qkBytes, devFinal.data(), qkBytes);

    bool ret = ResultCmp(golden, devFinal, 0.001f);
    EXPECT_TRUE(ret);
}

static void tqk_test_case_s0_256_s1_64()
{
    constexpr int S0 = 256;
    constexpr int H = 128;
    constexpr int S1 = 64;
    constexpr int CUBE_S0 = 256;
    constexpr int CUBE_S1 = 64;
    constexpr int TILE_S1 = 64;

    size_t qBytes = static_cast<size_t>(S0) * static_cast<size_t>(H) * sizeof(uint16_t);
    size_t kBytes = static_cast<size_t>(H) * static_cast<size_t>(S1) * sizeof(uint16_t);
    size_t qkBytes = static_cast<size_t>(S0) * static_cast<size_t>(S1) * sizeof(float);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint16_t *qHost = nullptr;
    uint16_t *kHost = nullptr;
    float *qkHost = nullptr;

    uint16_t *qDev = nullptr;
    uint16_t *kDev = nullptr;
    float *qkDev = nullptr;

    aclrtMallocHost((void **)(&qHost), qBytes);
    aclrtMallocHost((void **)(&kHost), kBytes);
    aclrtMallocHost((void **)(&qkHost), qkBytes);

    aclrtMalloc((void **)&qDev, qBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&kDev, kBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&qkDev, qkBytes, ACL_MEM_MALLOC_HUGE_FIRST);

    size_t fileSize = 0;
    ReadFile(GetGoldenDir() + "/q.bin", fileSize, qHost, qBytes);
    ReadFile(GetGoldenDir() + "/k_t.bin", fileSize, kHost, kBytes);

    aclrtMemcpy(qDev, qBytes, qHost, qBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(kDev, kBytes, kHost, kBytes, ACL_MEMCPY_HOST_TO_DEVICE);

    LaunchTQK<S0, H, S1, CUBE_S0, CUBE_S1, TILE_S1>(qDev, kDev, qkDev, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(qkHost, qkBytes, qkDev, qkBytes, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output_z.bin", qkHost, qkBytes);

    aclrtFree(qDev);
    aclrtFree(kDev);
    aclrtFree(qkDev);

    aclrtFreeHost(qHost);
    aclrtFreeHost(kHost);
    aclrtFreeHost(qkHost);

    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<float> golden(qkBytes / sizeof(float));
    std::vector<float> devFinal(qkBytes / sizeof(float));
#if ND_LAYOUT
    ReadFile(GetGoldenDir() + "/golden_nd.bin", qkBytes, golden.data(), qkBytes);
#else
    ReadFile(GetGoldenDir() + "/golden_dn.bin", qkBytes, golden.data(), qkBytes);
#endif
    ReadFile(GetGoldenDir() + "/output_z.bin", qkBytes, devFinal.data(), qkBytes);

    bool ret = ResultCmp(golden, devFinal, 0.001f);
    EXPECT_TRUE(ret);
}

TEST_F(TQKTest, case512)
{
    tqk_test_case(512);
}

TEST_F(TQKTest, case256)
{
    tqk_test_case_s0_256_s1_64();
}

TEST_F(TQKTest, case128)
{
    tqk_test_case(128);
}
