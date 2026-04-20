/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

/**
 * @file main.cpp
 * @brief Test harness for cube_matmul_4buf kernel (both preload and non-preload)
 * 
 * Tests: A[32,1024] @ B[1024,256] = C[32,256]
 * with 4-buffer software pipelining and K-loop tiling
 * 
 * Two versions tested:
 * 1. Non-preload: Load → Move → Compute sequential
 * 2. Preload: Overlaps Load with Compute (uses all 8 event IDs)
 */

#include "test_common.h"
#include <pto/pto-inst.hpp>
#include <gtest/gtest.h>

using namespace std;
using namespace PtoTestCommon;

// Forward declarations
void LaunchCubeMatmul4Buf(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template <typename T_A, typename T_B, typename T_C> void LaunchCubeMatmul4BufPreload(T_A *a, T_B *b, T_C *c, void *stream);

// Matrix dimensions
constexpr uint32_t M = 32;
constexpr uint32_t K = 1024;
constexpr uint32_t N = 256;

class CubeMatmul4BufTest : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

class CubeMatmul4BufPreloadTest : public testing::Test {
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

// Non-preload version test
TEST_F(CubeMatmul4BufTest, case_f16_32x1024_1024x256)
{
    // A: 32x1024 half, B: 1024x256 half, C: 32x256 float
    size_t aFileSize = M * K * sizeof(uint16_t);  // half
    size_t bFileSize = K * N * sizeof(uint16_t);  // half
    size_t cFileSize = M * N * sizeof(float);

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

    // Read input data
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/A_gm.bin", aFileSize, src0Host, aFileSize));
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/B_gm.bin", bFileSize, src1Host, bFileSize));

    aclrtMemcpy(src0Device, aFileSize, src0Host, aFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src1Device, bFileSize, src1Host, bFileSize, ACL_MEMCPY_HOST_TO_DEVICE);

    // Launch kernel
    LaunchCubeMatmul4Buf(dstDevice, src0Device, src1Device, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, cFileSize, dstDevice, cFileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    // Write output for inspection
    WriteFile(GetGoldenDir() + "/output_C.bin", dstHost, cFileSize);

    aclrtFree(dstDevice);
    aclrtFree(src0Device);
    aclrtFree(src1Device);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(src0Host);
    aclrtFreeHost(src1Host);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    // Compare with golden
    std::vector<float> golden(M * N);
    std::vector<float> result(M * N);
    
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/golden.bin", cFileSize, golden.data(), cFileSize));
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/output_C.bin", cFileSize, result.data(), cFileSize));

    // Use relative tolerance for matmul (accumulated f16 -> f32)
    bool ret = ResultCmp(golden, result, 0.01f);  // 1% tolerance for fp16 accumulation
    EXPECT_TRUE(ret);
}

// Preload version test (uses all 8 event IDs)
TEST_F(CubeMatmul4BufPreloadTest, case_f16_32x1024_1024x256_preload)
{
    // A: 32x1024 half, B: 1024x256 half, C: 32x256 float
    size_t aFileSize = M * K * sizeof(uint16_t);  // half
    size_t bFileSize = K * N * sizeof(uint16_t);  // half
    size_t cFileSize = M * N * sizeof(float);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    half *src0Host, *src1Host;
    float *dstHost;
    half *src0Device, *src1Device;
    float *dstDevice;

    aclrtMallocHost((void **)(&dstHost), cFileSize);
    aclrtMallocHost((void **)(&src0Host), aFileSize);
    aclrtMallocHost((void **)(&src1Host), bFileSize);

    aclrtMalloc((void **)&dstDevice, cFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src0Device, aFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src1Device, bFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    // Read input data (reuse non-preload test's golden data directory)
    std::string goldenDir = "../CubeMatmul4BufTest.case_f16_32x1024_1024x256";
    CHECK_RESULT_GTEST(ReadFile(goldenDir + "/A_gm.bin", aFileSize, src0Host, aFileSize));
    CHECK_RESULT_GTEST(ReadFile(goldenDir + "/B_gm.bin", bFileSize, src1Host, bFileSize));

    aclrtMemcpy(src0Device, aFileSize, src0Host, aFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src1Device, bFileSize, src1Host, bFileSize, ACL_MEMCPY_HOST_TO_DEVICE);

    // Launch PRELOAD kernel
    LaunchCubeMatmul4BufPreload(src0Device, src1Device, dstDevice, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, cFileSize, dstDevice, cFileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    // Write output for inspection
    WriteFile(goldenDir + "/output_C_preload.bin", dstHost, cFileSize);

    aclrtFree(dstDevice);
    aclrtFree(src0Device);
    aclrtFree(src1Device);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(src0Host);
    aclrtFreeHost(src1Host);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    // Compare with same golden (both compute same result, just different pipelining)
    std::vector<float> golden(M * N);
    std::vector<float> result(M * N);
    
    CHECK_RESULT_GTEST(ReadFile(goldenDir + "/golden.bin", cFileSize, golden.data(), cFileSize));
    CHECK_RESULT_GTEST(ReadFile(goldenDir + "/output_C_preload.bin", cFileSize, result.data(), cFileSize));

    // Use relative tolerance for matmul (accumulated f16 -> f32)
    bool ret = ResultCmp(golden, result, 0.01f);  // 1% tolerance for fp16 accumulation
    EXPECT_TRUE(ret);
}
