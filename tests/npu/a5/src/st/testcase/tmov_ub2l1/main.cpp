/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <cstdint>
#include <cstdlib>
#include <vector>

#include <gtest/gtest.h>
#include "acl/acl.h"

#include "test_common.h"

using namespace std;
using namespace PtoTestCommon;

template <int32_t testKey>
void launchTmovUb2l1(uint64_t* out, uint64_t* src, void* stream);

class TMovUb2l1Test : public testing::Test {
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

template <int32_t testKey, typename dType>
void testTMovUb2L1(int32_t srcRows, int32_t srcCols, int32_t dstRows, int32_t dstCols)
{
    aclInit(nullptr);
    const char* deviceEnv = std::getenv("PTO_DEVICE_ID");
    const int deviceId = deviceEnv ? std::atoi(deviceEnv) : 0;
    aclrtSetDevice(deviceId);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    size_t srcByteSize = srcRows * srcCols * sizeof(dType);
    size_t dstByteSize = dstRows * dstCols * sizeof(dType);
    uint64_t *dstHost, *srcHost, *dstDevice, *srcDevice;

    aclrtMallocHost((void**)(&dstHost), dstByteSize);
    aclrtMallocHost((void**)(&srcHost), srcByteSize);
    aclrtMalloc((void**)&dstDevice, dstByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&srcDevice, srcByteSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/input_arr.bin", srcByteSize, srcHost, srcByteSize);
    aclrtMemset(dstDevice, dstByteSize, 0, dstByteSize);
    aclrtMemcpy(srcDevice, srcByteSize, srcHost, srcByteSize, ACL_MEMCPY_HOST_TO_DEVICE);

    launchTmovUb2l1<testKey>(dstDevice, srcDevice, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, dstByteSize, dstDevice, dstByteSize, ACL_MEMCPY_DEVICE_TO_HOST);
    WriteFile(GetGoldenDir() + "/output.bin", dstHost, dstByteSize);

    aclrtFree(dstDevice);
    aclrtFree(srcDevice);
    aclrtFreeHost(dstHost);
    aclrtFreeHost(srcHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    std::vector<dType> golden(dstByteSize / sizeof(dType));
    std::vector<dType> devFinal(dstByteSize / sizeof(dType));
    ReadFile(GetGoldenDir() + "/golden_output.bin", dstByteSize, golden.data(), dstByteSize);
    ReadFile(GetGoldenDir() + "/output.bin", dstByteSize, devFinal.data(), dstByteSize);
    bool ret = ResultCmp(golden, devFinal, 0.001f);
    EXPECT_TRUE(ret);
}

TEST_F(TMovUb2l1Test, case1) { testTMovUb2L1<1, uint16_t>(16, 32, 16, 32); }

TEST_F(TMovUb2l1Test, case2) { testTMovUb2L1<2, uint16_t>(64, 256, 64, 256); }

TEST_F(TMovUb2l1Test, case3) { testTMovUb2L1<3, float>(48, 72, 48, 72); }

TEST_F(TMovUb2l1Test, case4) { testTMovUb2L1<4, float>(96, 8, 96, 8); }

TEST_F(TMovUb2l1Test, case5) { testTMovUb2L1<5, int8_t>(32, 512, 32, 512); }

TEST_F(TMovUb2l1Test, case6) { testTMovUb2L1<6, int8_t>(64, 96, 64, 96); }

TEST_F(TMovUb2l1Test, case7) { testTMovUb2L1<7, uint16_t>(64, 64, 48, 48); }

TEST_F(TMovUb2l1Test, case8) { testTMovUb2L1<8, float>(128, 128, 64, 64); }

TEST_F(TMovUb2l1Test, case9) { testTMovUb2L1<9, int8_t>(256, 256, 32, 32); }

#ifndef PTO_SKIP_UB2L1_ND2NZ_ST
static void testTMovUbToL1Nd2Nz(
    void (*launch)(uint64_t*, uint64_t*, void*), int elementBits, int srcRows, int srcCols, int dstRows, int dstCols,
    int validRows, int validCols, bool useDefaultCopy = false)
{
    const size_t srcBytes = srcRows * srcCols * elementBits / 8;
    const size_t dstBytes = dstRows * dstCols * elementBits / 8;
    std::vector<uint8_t> input(srcBytes);
    std::vector<uint8_t> output(dstBytes, 0xa5);
    std::vector<uint8_t> golden = output;
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = static_cast<uint8_t>((i * 17 + i / 7) % 251);
    }
    if (useDefaultCopy) {
        for (size_t byte = 0; byte < static_cast<size_t>(validRows * validCols * elementBits / 8); ++byte) {
            golden[byte] = input[byte];
        }
    } else {
        for (int row = 0; row < validRows; ++row) {
            for (int colByte = 0; colByte < validCols * elementBits / 8; ++colByte) {
                const size_t offset = (colByte / 32 * dstRows + row) * 32 + colByte % 32;
                golden[offset] = input[row * srcCols * elementBits / 8 + colByte];
            }
        }
    }
    ASSERT_EQ(aclInit(nullptr), ACL_SUCCESS);
    const char* deviceEnv = std::getenv("PTO_DEVICE_ID");
    const int deviceId = deviceEnv ? std::atoi(deviceEnv) : 0;
    ASSERT_EQ(aclrtSetDevice(deviceId), ACL_SUCCESS);
    aclrtStream stream;
    ASSERT_EQ(aclrtCreateStream(&stream), ACL_SUCCESS);
    void* srcDevice = nullptr;
    void* dstDevice = nullptr;
    ASSERT_EQ(aclrtMalloc(&srcDevice, srcBytes, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
    ASSERT_EQ(aclrtMalloc(&dstDevice, dstBytes, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
    ASSERT_EQ(aclrtMemcpy(srcDevice, srcBytes, input.data(), srcBytes, ACL_MEMCPY_HOST_TO_DEVICE), ACL_SUCCESS);
    ASSERT_EQ(aclrtMemcpy(dstDevice, dstBytes, output.data(), dstBytes, ACL_MEMCPY_HOST_TO_DEVICE), ACL_SUCCESS);
    launch(static_cast<uint64_t*>(dstDevice), static_cast<uint64_t*>(srcDevice), stream);
    EXPECT_EQ(aclrtSynchronizeStream(stream), ACL_SUCCESS);
    EXPECT_EQ(aclrtMemcpy(output.data(), dstBytes, dstDevice, dstBytes, ACL_MEMCPY_DEVICE_TO_HOST), ACL_SUCCESS);
    EXPECT_EQ(aclrtFree(srcDevice), ACL_SUCCESS);
    EXPECT_EQ(aclrtFree(dstDevice), ACL_SUCCESS);
    EXPECT_EQ(aclrtDestroyStream(stream), ACL_SUCCESS);
    EXPECT_EQ(aclrtResetDevice(deviceId), ACL_SUCCESS);
    EXPECT_EQ(aclFinalize(), ACL_SUCCESS);
    EXPECT_EQ(output, golden);
}

TEST_F(TMovUb2l1Test, nd2nz_half_static)
{
    testTMovUbToL1Nd2Nz(launchTmovUb2l1<10>, 16, 16, 32, 16, 32, 16, 32, false);
}

TEST_F(TMovUb2l1Test, nd2nz_half_wide)
{
    testTMovUbToL1Nd2Nz(launchTmovUb2l1<11>, 16, 16, 512, 16, 512, 3, 512, false);
}

TEST_F(TMovUb2l1Test, nd2nz_float_strides)
{
    testTMovUbToL1Nd2Nz(launchTmovUb2l1<12>, 32, 32, 64, 32, 32, 19, 24, false);
}

TEST_F(TMovUb2l1Test, nd2nz_bfloat16_strides)
{
    testTMovUbToL1Nd2Nz(launchTmovUb2l1<13>, 16, 32, 96, 32, 64, 17, 48, false);
}

TEST_F(TMovUb2l1Test, nd2nz_int8_strides)
{
    testTMovUbToL1Nd2Nz(launchTmovUb2l1<14>, 8, 32, 160, 32, 128, 31, 96, false);
}

TEST_F(TMovUb2l1Test, nd2nz_fp8_e4m3) { testTMovUbToL1Nd2Nz(launchTmovUb2l1<15>, 8, 16, 64, 16, 64, 16, 64, false); }

TEST_F(TMovUb2l1Test, nd2nz_fp8_e5m2) { testTMovUbToL1Nd2Nz(launchTmovUb2l1<16>, 8, 16, 64, 16, 64, 16, 64, false); }

TEST_F(TMovUb2l1Test, nd2nz_hifloat8) { testTMovUbToL1Nd2Nz(launchTmovUb2l1<17>, 8, 16, 64, 16, 64, 16, 64, false); }

TEST_F(TMovUb2l1Test, nd2nz_fp8_e8m0) { testTMovUbToL1Nd2Nz(launchTmovUb2l1<18>, 8, 16, 64, 16, 64, 16, 64, false); }

TEST_F(TMovUb2l1Test, nd2nz_fp4_e2m1) { testTMovUbToL1Nd2Nz(launchTmovUb2l1<19>, 4, 32, 256, 32, 192, 17, 128, false); }

TEST_F(TMovUb2l1Test, nd2nz_fp4_e1m2) { testTMovUbToL1Nd2Nz(launchTmovUb2l1<20>, 4, 16, 128, 16, 128, 16, 128, false); }

TEST_F(TMovUb2l1Test, nd2nz_burst_split)
{
    testTMovUbToL1Nd2Nz(launchTmovUb2l1<24>, 16, 4112, 16, 4112, 16, 4101, 16, false);
}

TEST_F(TMovUb2l1Test, nd2nz_empty) { testTMovUbToL1Nd2Nz(launchTmovUb2l1<25>, 16, 16, 32, 16, 32, 0, 32, false); }

TEST_F(TMovUb2l1Test, nd2nz_empty_cols) { testTMovUbToL1Nd2Nz(launchTmovUb2l1<27>, 32, 16, 32, 16, 32, 7, 0, false); }

TEST_F(TMovUb2l1Test, legacy_null_static) { testTMovUbToL1Nd2Nz(launchTmovUb2l1<70>, 16, 16, 64, 16, 64, 2, 32, true); }

TEST_F(TMovUb2l1Test, legacy_null_dynamic)
{
    testTMovUbToL1Nd2Nz(launchTmovUb2l1<71>, 16, 16, 64, 16, 64, 7, 32, true);
}

#endif
