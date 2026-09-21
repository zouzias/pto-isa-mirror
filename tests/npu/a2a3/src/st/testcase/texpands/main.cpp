/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <cstddef>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "runtime/rt.h"
#include "test_common.h"

using namespace std;
using namespace PtoTestCommon;

class TEXPANDSTest : public testing::Test {
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

template <
    typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_, int kVRows_, int kVCols_, int padValueType,
    bool StaticValid = false, bool InitializeFromGlobal = false>
void LaunchTExpandS(void* out, float scalar, void* stream);

struct RuntimeResources {
    rtStream_t stream = nullptr;
    void* dst = nullptr;
    bool deviceSet = false;

    ~RuntimeResources()
    {
        if (stream != nullptr) {
            EXPECT_EQ(rtStreamDestroy(stream), RT_ERROR_NONE);
        }
        if (dst != nullptr) {
            EXPECT_EQ(rtFree(dst), RT_ERROR_NONE);
        }
        if (deviceSet) {
            EXPECT_EQ(rtDeviceReset(0), RT_ERROR_NONE);
        }
    }
};

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_, int kVRows_, int kVCols_, int padValueType>
void testTExpands()
{
    size_t fileSize = kGRows_ * kGCols_ * sizeof(T);

    RuntimeResources resources;
    ASSERT_EQ(rtSetDevice(0), RT_ERROR_NONE);
    resources.deviceSet = true;
    ASSERT_EQ(rtStreamCreate(&resources.stream, 0), RT_ERROR_NONE);
    ASSERT_EQ(rtMalloc(&resources.dst, fileSize, RT_MEMORY_HBM, 0), RT_ERROR_NONE);
    std::vector<T> dstHost(fileSize / sizeof(T));

    float scalar;
    std::string scalar_file = GetGoldenDir() + "/scalar.bin";
    std::ifstream file(scalar_file, std::ios::binary);
    file.read(reinterpret_cast<char*>(&scalar), 4);
    file.close();

    LaunchTExpandS<T, kGRows_, kGCols_, kTRows_, kTCols_, kVRows_, kVCols_, padValueType>(
        resources.dst, scalar, resources.stream);

    ASSERT_EQ(rtStreamSynchronize(resources.stream), RT_ERROR_NONE);
    ASSERT_EQ(rtMemcpy(dstHost.data(), fileSize, resources.dst, fileSize, RT_MEMCPY_DEVICE_TO_HOST), RT_ERROR_NONE);

    WriteFile(GetGoldenDir() + "/output.bin", dstHost.data(), fileSize);

    std::vector<T> golden(fileSize / sizeof(T));
    std::vector<T> devFinal(fileSize / sizeof(T));
    ReadFile(GetGoldenDir() + "/golden.bin", fileSize, golden.data(), fileSize);
    ReadFile(GetGoldenDir() + "/output.bin", fileSize, devFinal.data(), fileSize);

    bool ret = ResultCmp<T>(golden, devFinal, 0.001f);

    EXPECT_TRUE(ret);
}

TEST_F(TEXPANDSTest, case_float_64x64_64x64_64x64_PAD_VALUE_NULL)
{
    testTExpands<float, 64, 64, 64, 64, 64, 64, PAD_VALUE_NULL>();
}
TEST_F(TEXPANDSTest, case_int32_64x64_64x64_64x64_PAD_VALUE_NULL)
{
    testTExpands<int32_t, 64, 64, 64, 64, 64, 64, PAD_VALUE_NULL>();
}
TEST_F(TEXPANDSTest, case_half_64x64_64x64_64x64_PAD_VALUE_NULL)
{
    testTExpands<aclFloat16, 64, 64, 64, 64, 64, 64, PAD_VALUE_NULL>();
}
TEST_F(TEXPANDSTest, case_int16_64x64_64x64_64x64_PAD_VALUE_NULL)
{
    testTExpands<int16_t, 64, 64, 64, 64, 64, 64, PAD_VALUE_NULL>();
}

TEST_F(TEXPANDSTest, case_float_60x60_64x64_60x60_PAD_VALUE_MAX)
{
    testTExpands<float, 60, 60, 64, 64, 60, 60, PAD_VALUE_MAX>();
}
TEST_F(TEXPANDSTest, case_int32_60x60_64x64_60x60_PAD_VALUE_MAX)
{
    testTExpands<int32_t, 60, 60, 64, 64, 60, 60, PAD_VALUE_MAX>();
}
TEST_F(TEXPANDSTest, case_half_1x3600_2x4096_1x3600_PAD_VALUE_MAX)
{
    testTExpands<aclFloat16, 1, 3600, 2, 4096, 1, 3600, PAD_VALUE_MAX>();
}
TEST_F(TEXPANDSTest, case_int16_16x200_20x512_16x200_PAD_VALUE_MAX)
{
    testTExpands<int16_t, 16, 200, 20, 512, 16, 200, PAD_VALUE_MAX>();
}
TEST_F(TEXPANDSTest, case_int8_16x200_20x512_16x200_PAD_VALUE_MAX)
{
    testTExpands<int8_t, 16, 200, 20, 512, 16, 200, PAD_VALUE_MAX>();
}

template <int TileRows, int TileCols, int ValidRows, int ValidCols, bool StaticValid>
void testTExpandsValidShape()
{
    constexpr float SCALAR = 1.4020596f;
    constexpr size_t ELEMENT_COUNT = ValidRows * ValidCols;
    constexpr size_t OUTPUT_BYTES = ELEMENT_COUNT * sizeof(float);
    RuntimeResources resources;
    ASSERT_EQ(rtSetDevice(0), RT_ERROR_NONE);
    resources.deviceSet = true;
    ASSERT_EQ(rtStreamCreate(&resources.stream, 0), RT_ERROR_NONE);
    ASSERT_EQ(rtMalloc(&resources.dst, OUTPUT_BYTES, RT_MEMORY_HBM, 0), RT_ERROR_NONE);
    ASSERT_EQ(rtMemset(resources.dst, OUTPUT_BYTES, 0, OUTPUT_BYTES), RT_ERROR_NONE);

    LaunchTExpandS<
        float, ValidRows, ValidCols, TileRows, TileCols, ValidRows, ValidCols, PAD_VALUE_NULL, StaticValid, true>(
        resources.dst, SCALAR, resources.stream);
    ASSERT_EQ(rtStreamSynchronize(resources.stream), RT_ERROR_NONE);

    std::vector<float> actual(ELEMENT_COUNT);
    ASSERT_EQ(
        rtMemcpy(actual.data(), OUTPUT_BYTES, resources.dst, OUTPUT_BYTES, RT_MEMCPY_DEVICE_TO_HOST), RT_ERROR_NONE);
    size_t matchedCount = 0;
    size_t zeroCount = 0;
    size_t firstMismatch = ELEMENT_COUNT;
    for (size_t i = 0; i < ELEMENT_COUNT; ++i) {
        matchedCount += actual[i] == SCALAR;
        zeroCount += actual[i] == 0.0f;
        if (actual[i] != SCALAR && firstMismatch == ELEMENT_COUNT) {
            firstMismatch = i;
        }
    }
    std::cout << "matched=" << matchedCount << "/" << ELEMENT_COUNT << " zeros=" << zeroCount << std::endl;
    EXPECT_EQ(matchedCount, ELEMENT_COUNT)
        << "first mismatch=" << firstMismatch << ", expected=" << SCALAR << ", tail=" << actual.back();
}

TEST_F(TEXPANDSTest, case_float_rowmajor_4x16_4x8_static_valid) { testTExpandsValidShape<4, 16, 4, 8, true>(); }

TEST_F(TEXPANDSTest, case_float_rowmajor_4x16_4x8_dynamic_valid) { testTExpandsValidShape<4, 16, 4, 8, false>(); }

TEST_F(TEXPANDSTest, case_float_colmajor_16x4_8x4_static_valid) { testTExpandsValidShape<16, 4, 8, 4, true>(); }

TEST_F(TEXPANDSTest, case_float_colmajor_16x4_8x4_dynamic_valid) { testTExpandsValidShape<16, 4, 8, 4, false>(); }

TEST_F(TEXPANDSTest, case_float_rowmajor_101x400_101x78_static_valid)
{
    testTExpandsValidShape<101, 400, 101, 78, true>();
}

TEST_F(TEXPANDSTest, case_float_rowmajor_101x400_101x78_dynamic_valid)
{
    testTExpandsValidShape<101, 400, 101, 78, false>();
}
