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
#include <pto/cpu/TMatmul.hpp>
#include <gtest/gtest.h>
#include <string>
#include <sstream>
#include <cstdlib>

using namespace std;
using namespace PtoTestCommon;
using namespace pto;

struct SizeArgs {
    uint32_t M, K, N;
    std::string transmode;
    bool valid;
    bool use_custom_size;
};

SizeArgs parse_size_args(int argc, char** argv) {
    SizeArgs args = {0, 0, 0, "NN", false, false};

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--size" && i + 1 < argc) {
            std::string size_str = argv[++i];
            std::stringstream ss(size_str);
            char comma;
            if (ss >> args.M >> comma >> args.K >> comma >> args.N) {
                args.valid = true;
                args.use_custom_size = true;
            }
        } else if (arg == "--transmode" && i + 1 < argc) {
            args.transmode = argv[++i];
        }
    }
    return args;
}

template <int32_t tilingKey>
void LaunchTMATMUL(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);

template <int32_t tilingKey>
void LaunchTMATMULBIAS(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream);

// External global variable from tmatmul_kernel.cpp
extern TransMode g_transmode;

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

template <typename T, typename U, typename S, int32_t key>
void tmatmul_test(uint32_t M, uint32_t K, uint32_t N)
{
    size_t aFileSize = M * K * sizeof(U);
    size_t bFileSize = K * N * sizeof(S);
    size_t cFileSize = M * N * sizeof(T);

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

    WriteFile(GetGoldenDir() + "/output_z.bin", dstHost, cFileSize);

    aclrtFree(dstDevice);
    aclrtFree(src0Device);
    aclrtFree(src1Device);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(src0Host);
    aclrtFreeHost(src1Host);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<float> golden(cFileSize);
    std::vector<float> devFinal(cFileSize);
    ReadFile(GetGoldenDir() + "/golden.bin", cFileSize, golden.data(), cFileSize);
    ReadFile(GetGoldenDir() + "/output_z.bin", cFileSize, devFinal.data(), cFileSize);

    bool ret = ResultCmp(golden, devFinal, 0.001f);

    EXPECT_TRUE(ret);
}

template <typename T, typename U, typename S, int32_t key>
void tmatmul_test_custom(uint32_t M, uint32_t K, uint32_t N, const std::string& transmode_str = "NN")
{
    // Convert transmode string to enum
    TransMode transmode = TransMode::NN;
    if (transmode_str == "NN") transmode = TransMode::NN;
    else if (transmode_str == "NT") transmode = TransMode::NT;
    else if (transmode_str == "TN") transmode = TransMode::TN;
    else if (transmode_str == "TT") transmode = TransMode::TT;

    // Calculate file sizes based on transmode
    size_t aFileSize, bFileSize, cFileSize;

    if (transmode == TransMode::NN || transmode == TransMode::NT) {
        aFileSize = M * K * sizeof(U);  // A is MxK
    } else {  // TN, TT - A is KxM
        aFileSize = K * M * sizeof(U);
    }

    if (transmode == TransMode::NN || transmode == TransMode::TN) {
        bFileSize = K * N * sizeof(S);  // B is KxN
    } else {  // NT, TT - B is NxK
        bFileSize = N * K * sizeof(S);
    }

    cFileSize = M * N * sizeof(T);

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

    // For custom size, use current directory
    ReadFile("./x1_gm.bin", aFileSize, src0Host, aFileSize);
    ReadFile("./x2_gm.bin", bFileSize, src1Host, bFileSize);

    aclrtMemcpy(src0Device, aFileSize, src0Host, aFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src1Device, bFileSize, src1Host, bFileSize, ACL_MEMCPY_HOST_TO_DEVICE);

    // Note: Need to modify LaunchTMATMUL to accept transmode
    // For now, we'll pass it but the implementation will need to be updated
    LaunchTMATMUL<key>(dstDevice, src0Device, src1Device, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, cFileSize, dstDevice, cFileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile("./output_z.bin", dstHost, cFileSize);

    aclrtFree(dstDevice);
    aclrtFree(src0Device);
    aclrtFree(src1Device);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(src0Host);
    aclrtFreeHost(src1Host);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<float> golden(cFileSize);
    std::vector<float> devFinal(cFileSize);
    ReadFile("./golden.bin", cFileSize, golden.data(), cFileSize);
    ReadFile("./output_z.bin", cFileSize, devFinal.data(), cFileSize);

    bool ret = ResultCmp(golden, devFinal, 0.001f);

    EXPECT_TRUE(ret);
}

TEST_F(TMATMULTest, case1)
{
    tmatmul_test<float, uint16_t, uint16_t, 1>(40, 50, 60);
}

TEST_F(TMATMULTest, case2)
{
    tmatmul_test<int32_t, int8_t, int8_t, 2>(6, 7, 8);
}

TEST_F(TMATMULTest, case3)
{
    uint32_t M = 128;
    uint32_t N = 64;
    uint32_t K = 128;
    uint32_t repeats = 5;

    size_t aFileSize = repeats * M * K * sizeof(uint16_t); // uint16_t represent half
    size_t bFileSize = repeats * K * N * sizeof(uint16_t); // uint16_t represent half
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

    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/x1_gm.bin", aFileSize, src0Host, aFileSize));
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/x2_gm.bin", bFileSize, src1Host, bFileSize));

    aclrtMemcpy(src0Device, aFileSize, src0Host, aFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src1Device, bFileSize, src1Host, bFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    LaunchTMATMUL<3>(dstDevice, src0Device, src1Device, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, cFileSize, dstDevice, cFileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output_z.bin", dstHost, cFileSize);

    aclrtFree(dstDevice);
    aclrtFree(src0Device);
    aclrtFree(src1Device);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(src0Host);
    aclrtFreeHost(src1Host);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<float> golden(cFileSize);
    std::vector<float> devFinal(cFileSize);

    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/golden.bin", cFileSize, golden.data(), cFileSize));
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/output_z.bin", cFileSize, devFinal.data(), cFileSize));

    bool ret = ResultCmp(golden, devFinal, 0.001f);
    EXPECT_TRUE(ret);
}

TEST_F(TMATMULTest, case4)
{
    tmatmul_test<float, float, float, 4>(120, 110, 50);
}

#ifdef CPU_SIM_BFLOAT_ENABLED
TEST_F(TMATMULTest, case_bf16_1)
{
    tmatmul_test<float, bfloat16_t, bfloat16_t, 6>(40, 50, 60);
}
#endif

template <typename T, typename U, typename S, typename B, int32_t key>
void tmatmul_bias_test(uint32_t M, uint32_t K, uint32_t N)
{
    size_t aFileSize = M * K * sizeof(U);
    size_t bFileSize = K * N * sizeof(S);
    size_t cFileSize = M * N * sizeof(T);
    size_t biasFileSize = 1 * N * sizeof(B);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint8_t *dstHost, *src0Host, *src1Host, *src2Host;
    uint8_t *dstDevice, *src0Device, *src1Device, *src2Device;

    aclrtMallocHost((void **)(&dstHost), cFileSize);
    aclrtMallocHost((void **)(&src0Host), aFileSize);
    aclrtMallocHost((void **)(&src1Host), bFileSize);
    aclrtMallocHost((void **)(&src2Host), biasFileSize);

    aclrtMalloc((void **)&dstDevice, cFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src0Device, aFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src1Device, bFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src2Device, biasFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/x1_gm.bin", aFileSize, src0Host, aFileSize);
    ReadFile(GetGoldenDir() + "/x2_gm.bin", bFileSize, src1Host, bFileSize);
    ReadFile(GetGoldenDir() + "/bias_gm.bin", biasFileSize, src2Host, biasFileSize);

    aclrtMemcpy(src0Device, aFileSize, src0Host, aFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src1Device, bFileSize, src1Host, bFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src2Device, biasFileSize, src2Host, biasFileSize, ACL_MEMCPY_HOST_TO_DEVICE);

    LaunchTMATMULBIAS<key>(dstDevice, src0Device, src1Device, src2Device, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, cFileSize, dstDevice, cFileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output_z.bin", dstHost, cFileSize);

    aclrtFree(dstDevice);
    aclrtFree(src0Device);
    aclrtFree(src1Device);
    aclrtFree(src2Device);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(src0Host);
    aclrtFreeHost(src1Host);
    aclrtFreeHost(src2Host);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<float> golden(cFileSize);
    std::vector<float> devFinal(cFileSize);
    ReadFile(GetGoldenDir() + "/golden.bin", cFileSize, golden.data(), cFileSize);
    ReadFile(GetGoldenDir() + "/output_z.bin", cFileSize, devFinal.data(), cFileSize);

    bool ret = ResultCmp(golden, devFinal, 0.001f);

    EXPECT_TRUE(ret);
}

TEST_F(TMATMULTest, case_bias_1)
{
    tmatmul_bias_test<int32_t, int8_t, int8_t, int32_t, 1>(8, 7, 6);
}

TEST_F(TMATMULTest, case_bias_2)
{
    tmatmul_bias_test<float, uint16_t, uint16_t, float, 2>(16, 15, 16);
}

TEST_F(TMATMULTest, case_bias_5)
{
    uint32_t M = 127;
    uint32_t N = 63;
    uint32_t K = 128;

    tmatmul_bias_test<float, float, float, float, 5>(M, K, N);
}

#ifdef CPU_SIM_BFLOAT_ENABLED
TEST_F(TMATMULTest, case_bf16_bias_1)
{
    tmatmul_bias_test<float, bfloat16_t, bfloat16_t, float, 7>(16, 15, 16);
}
#endif

int main(int argc, char** argv) {
    SizeArgs size_args = parse_size_args(argc, argv);

    if (size_args.use_custom_size && size_args.valid) {
        // Convert transmode string to enum and set global variable
        if (size_args.transmode == "NN") g_transmode = TransMode::NN;
        else if (size_args.transmode == "NT") g_transmode = TransMode::NT;
        else if (size_args.transmode == "TN") g_transmode = TransMode::TN;
        else if (size_args.transmode == "TT") g_transmode = TransMode::TT;

        // Custom size mode
        std::cout << "Running custom size test: " << size_args.M << "x" << size_args.K << "x" << size_args.N
                  << " with transmode=" << size_args.transmode << std::endl;
        tmatmul_test_custom<float, uint16_t, uint16_t, 1>(size_args.M, size_args.K, size_args.N, size_args.transmode);
        return 0;
    } else if (size_args.use_custom_size && !size_args.valid) {
        std::cerr << "Error: Invalid --size format. Expected M,K,N" << std::endl;
        return 1;
    }

    // Default: run Google Test
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
