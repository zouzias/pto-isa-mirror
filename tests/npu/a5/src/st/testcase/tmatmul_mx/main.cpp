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
#include "runtime/rt.h"
#include <gtest/gtest.h>

using namespace std;
using namespace PtoTestCommon;

template <int32_t tilingKey>
void LaunchTMATMUL_MX(uint8_t* out, uint8_t* src0, uint8_t* src1, uint8_t* src2, uint8_t* src3, void* stream);

template <int32_t tilingKey>
void LaunchTMATMUL_MX_BIAS(
    uint8_t* out, uint8_t* src0, uint8_t* src1, uint8_t* src2, uint8_t* src3, uint8_t* src4, void* stream);

template <int32_t key>
void LaunchTMATMUL_MX_SPLITK(
    uint8_t* out, uint8_t* src0, uint8_t* src1, uint8_t* src2, uint8_t* src3, uint8_t* src4, void* stream);

template <int32_t key>
void LaunchTGEMV_MX_SPLITK(
    uint8_t* out, uint8_t* src0, uint8_t* src1, uint8_t* src2, uint8_t* src3, uint8_t* src4, void* stream);

class TMATMULMXTest : public testing::Test {
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

template <typename T>
constexpr T CeilDiv(T num_1, T num_2)
{
    if (num_2 == 0) {
        return 0;
    }
    return (num_1 + num_2 - 1) / num_2;
}

template <typename T>
const T CeilAlign(T num_1, T num_2)
{
    if (num_2 == 0) {
        return 0;
    }
    return (num_1 + num_2 - 1) / num_2 * num_2;
}

// Own Runtime resources even when a fatal GTest assertion returns early.
struct RuntimeResources {
    rtStream_t stream = nullptr;
    bool deviceSet = false;
    std::vector<void*> allocations;

    rtError_t allocate(uint8_t*& ptr, size_t bytes)
    {
        void* allocation = nullptr;
        const auto status = rtMalloc(&allocation, bytes, RT_MEMORY_HBM, 0);
        if (status == RT_ERROR_NONE) {
            allocations.push_back(allocation);
            ptr = static_cast<uint8_t*>(allocation);
        }
        return status;
    }

    ~RuntimeResources()
    {
        if (stream != nullptr) {
            EXPECT_EQ(rtStreamDestroy(stream), RT_ERROR_NONE);
        }
        for (void* allocation : allocations) {
            EXPECT_EQ(rtFree(allocation), RT_ERROR_NONE);
        }
        if (deviceSet) {
            EXPECT_EQ(rtDeviceReset(0), RT_ERROR_NONE);
        }
    }
};

template <
    typename T, typename U, typename S, bool isBias, bool isFp4, int32_t key, bool splitK = false, bool gemv = false>
void TmatmulMXTest(uint32_t M, uint32_t K, uint32_t N, uint32_t validM, uint32_t validK, uint32_t validN)
{
    uint32_t kAlign = CeilAlign<uint32_t>(validK, 64);
    size_t aFileSize = isFp4 ? CeilDiv<uint32_t>(validM * validK, 2) : validM * validK * sizeof(U);
    size_t bFileSize = isFp4 ? CeilDiv<uint32_t>(validK * validN, 2) : validK * validN * sizeof(S);
    size_t aScaleFileSize = gemv ? validM * CeilDiv<uint32_t>(validK, 32) : M * CeilDiv<uint32_t>(kAlign, 32);
    size_t bScaleFileSize = N * CeilDiv<uint32_t>(kAlign, 32);
    const size_t cFileSize = validM * validN * sizeof(T);

    RuntimeResources resources;
    ASSERT_EQ(rtSetDevice(0), RT_ERROR_NONE);
    resources.deviceSet = true;
    ASSERT_EQ(rtStreamCreate(&resources.stream, 0), RT_ERROR_NONE);
    const auto stream = resources.stream;
    uint8_t* dstDevice = nullptr;
    uint8_t* sources[5] = {};
    const size_t sizes[] = {aFileSize, bFileSize, aScaleFileSize, bScaleFileSize, validN * sizeof(T)};
    const char* files[] = {"x1_gm.bin", "x2_gm.bin", "x1_mx_gm.bin", "x2_mx_gm.bin", "bias_gm.bin"};
    for (size_t i = 0; i < (isBias ? 5 : 4); ++i) {
        // Poison bytes beyond each GEMV payload so full-tail overreads cannot silently return zeros.
        const size_t guardBytes = gemv && i < 4 ? (i == 1 ? 1024 * validN : i == 3 ? 32 * N : 1024) : 0;
        const uint8_t poison = i < 2 ? (isFp4 ? 0x11 : 0x38) : 127;
        std::vector<uint8_t> data(sizes[i] + guardBytes, guardBytes ? poison : 0);
        size_t bytesRead = sizes[i];
        ASSERT_TRUE(ReadFile(GetGoldenDir() + "/" + files[i], bytesRead, data.data(), sizes[i]));
        if constexpr (splitK) {
            ASSERT_EQ(bytesRead, sizes[i]);
        }
        ASSERT_EQ(resources.allocate(sources[i], data.size()), RT_ERROR_NONE);
        ASSERT_EQ(rtMemcpy(sources[i], data.size(), data.data(), data.size(), RT_MEMCPY_HOST_TO_DEVICE), RT_ERROR_NONE);
    }
    ASSERT_EQ(resources.allocate(dstDevice, cFileSize), RT_ERROR_NONE);
    ASSERT_EQ(rtMemset(dstDevice, cFileSize, 0, cFileSize), RT_ERROR_NONE);
    auto* src0Device = sources[0];
    auto* src1Device = sources[1];
    auto* src2Device = sources[2];
    auto* src3Device = sources[3];
    auto* src4Device = sources[4];
    if constexpr (gemv) {
        LaunchTGEMV_MX_SPLITK<key>(dstDevice, src0Device, src1Device, src2Device, src3Device, src4Device, stream);
    } else if constexpr (splitK) {
        LaunchTMATMUL_MX_SPLITK<key>(dstDevice, src0Device, src1Device, src2Device, src3Device, src4Device, stream);
    } else if constexpr (isBias) {
        LaunchTMATMUL_MX_BIAS<key>(dstDevice, src0Device, src1Device, src2Device, src3Device, src4Device, stream);
    } else {
        LaunchTMATMUL_MX<key>(dstDevice, src0Device, src1Device, src2Device, src3Device, stream);
    }

    ASSERT_EQ(rtStreamSynchronize(stream), RT_ERROR_NONE);
    std::vector<T> golden(cFileSize / sizeof(T));
    std::vector<T> devFinal(cFileSize / sizeof(T));
    ASSERT_EQ(rtMemcpy(devFinal.data(), cFileSize, dstDevice, cFileSize, RT_MEMCPY_DEVICE_TO_HOST), RT_ERROR_NONE);
    size_t goldenBytesRead = cFileSize;
    ASSERT_TRUE(ReadFile(GetGoldenDir() + "/golden.bin", goldenBytesRead, golden.data(), cFileSize));
    ASSERT_EQ(goldenBytesRead, cFileSize);
    ASSERT_TRUE(WriteFile(GetGoldenDir() + "/output_z.bin", devFinal.data(), cFileSize));

    if constexpr (splitK) {
        for (size_t i = 0; i < golden.size(); ++i) {
            ASSERT_FLOAT_EQ(golden[i], devFinal[i]) << "element " << i;
        }
    } else {
        EXPECT_TRUE(ResultCmp(golden, devFinal, 0.001f));
    }
}

TEST_F(TMATMULMXTest, case1)
{
    uint32_t M = 128;
    uint32_t K = 64;
    uint32_t N = 64;

    TmatmulMXTest<float, uint8_t, uint8_t, false, false, 1>(M, K, N, M, K, N);
}

TEST_F(TMATMULMXTest, case2)
{
    uint32_t M = 127;
    uint32_t K = 72;
    uint32_t N = 64;

    TmatmulMXTest<float, uint8_t, uint8_t, false, false, 2>(128, 128, 64, M, K, N);
}

TEST_F(TMATMULMXTest, case3)
{
    uint32_t M = 128;
    uint32_t K = 110;
    uint32_t N = 63;

    TmatmulMXTest<float, uint8_t, uint8_t, false, false, 3>(128, 128, 64, M, K, N);
}

TEST_F(TMATMULMXTest, case4)
{
    uint32_t M = 128;
    uint32_t K = 64;
    uint32_t N = 64;

    TmatmulMXTest<float, uint8_t, uint8_t, false, true, 4>(128, 64, 64, M, K, N);
}

TEST_F(TMATMULMXTest, case5)
{
    uint32_t M = 117;
    uint32_t K = 64;
    uint32_t N = 60;

    TmatmulMXTest<float, uint8_t, uint8_t, false, true, 5>(128, 64, 64, M, K, N);
}

TEST_F(TMATMULMXTest, case6)
{
    uint32_t M = 128;
    uint32_t K = 118;
    uint32_t N = 64;

    TmatmulMXTest<float, uint8_t, uint8_t, false, true, 6>(128, 128, 64, M, K, N);
}

TEST_F(TMATMULMXTest, case7)
{
    uint32_t M = 115;
    uint32_t K = 64;
    uint32_t N = 30;

    TmatmulMXTest<float, uint8_t, uint8_t, false, true, 7>(128, 64, 64, M, K, N);
}

TEST_F(TMATMULMXTest, case8)
{
    uint32_t M = 16;
    uint32_t K = 32;
    uint32_t N = 16;

    TmatmulMXTest<float, uint8_t, uint8_t, false, false, 8>(16, 64, 32, M, K, N);
}

TEST_F(TMATMULMXTest, case9)
{
    uint32_t M = 10;
    uint32_t K = 50;
    uint32_t N = 54;

    TmatmulMXTest<float, uint8_t, uint8_t, false, false, 9>(16, 64, 64, M, K, N);
}

TEST_F(TMATMULMXTest, case10)
{
    uint32_t M = 4;
    uint32_t K = 30;
    uint32_t N = 8;

    TmatmulMXTest<float, uint8_t, uint8_t, false, true, 10>(16, 64, 64, M, K, N);
}

TEST_F(TMATMULMXTest, case11)
{
    uint32_t M = 1;
    uint32_t K = 128;
    uint32_t N = 62;

    TmatmulMXTest<float, uint8_t, uint8_t, false, true, 11>(16, 128, 64, M, K, N);
}

TEST_F(TMATMULMXTest, case12)
{
    uint32_t M = 1;
    uint32_t K = 256;
    uint32_t N = 20;

    TmatmulMXTest<float, uint8_t, uint8_t, false, false, 12>(16, 256, 64, M, K, N);
}

TEST_F(TMATMULMXTest, case13)
{
    uint32_t M = 115;
    uint32_t K = 64;
    uint32_t N = 30;

    TmatmulMXTest<float, uint8_t, uint8_t, true, false, 1>(128, 64, 32, M, K, N);
}

TEST_F(TMATMULMXTest, case14)
{
    uint32_t M = 200;
    uint32_t K = 192;
    uint32_t N = 95;

    TmatmulMXTest<float, uint8_t, uint8_t, true, false, 2>(208, 192, 128, M, K, N);
}

TEST_F(TMATMULMXTest, case15)
{
    uint32_t M = 35;
    uint32_t K = 128;
    uint32_t N = 56;

    TmatmulMXTest<float, uint8_t, uint8_t, true, true, 3>(48, 128, 64, M, K, N);
}

TEST_F(TMATMULMXTest, case16)
{
    uint32_t M = 47;
    uint32_t K = 128;
    uint32_t N = 62;

    TmatmulMXTest<float, uint8_t, uint8_t, true, true, 4>(48, 128, 64, M, K, N);
}

TEST_F(TMATMULMXTest, case17)
{
    uint32_t M = 64;
    uint32_t K = 192;
    uint32_t N = 64;

    TmatmulMXTest<float, uint8_t, uint8_t, true, false, 5>(64, 192, 64, M, K, N);
}

TEST_F(TMATMULMXTest, case18)
{
    uint32_t M = 1;
    uint32_t K = 64;
    uint32_t N = 62;

    TmatmulMXTest<float, uint8_t, uint8_t, true, true, 6>(16, 64, 64, M, K, N);
}

TEST_F(TMATMULMXTest, case19)
{
    uint32_t M = 1;
    uint32_t K = 2048;
    uint32_t N = 64;

    TmatmulMXTest<float, uint8_t, uint8_t, true, true, 7>(16, 2048, 64, M, K, N);
}

TEST_F(TMATMULMXTest, splitk_fp4_e1m2_e1m2_k62_nobias)
{
    TmatmulMXTest<float, uint8_t, uint8_t, false, true, 0, true>(48, 64, 128, 47, 62, 128);
}

TEST_F(TMATMULMXTest, splitk_fp4_e1m2_e1m2_k62_bias)
{
    TmatmulMXTest<float, uint8_t, uint8_t, true, true, 1, true>(48, 64, 128, 47, 62, 128);
}

TEST_F(TMATMULMXTest, splitk_fp4_e1m2_e2m1_k62_nobias)
{
    TmatmulMXTest<float, uint8_t, uint8_t, false, true, 2, true>(48, 64, 128, 47, 62, 128);
}

TEST_F(TMATMULMXTest, splitk_fp4_e1m2_e2m1_k62_bias)
{
    TmatmulMXTest<float, uint8_t, uint8_t, true, true, 3, true>(48, 64, 128, 47, 62, 128);
}

TEST_F(TMATMULMXTest, splitk_fp4_e2m1_e1m2_k62_nobias)
{
    TmatmulMXTest<float, uint8_t, uint8_t, false, true, 4, true>(48, 64, 128, 47, 62, 128);
}

TEST_F(TMATMULMXTest, splitk_fp4_e2m1_e1m2_k62_bias)
{
    TmatmulMXTest<float, uint8_t, uint8_t, true, true, 5, true>(48, 64, 128, 47, 62, 128);
}

TEST_F(TMATMULMXTest, splitk_fp4_e2m1_e2m1_k62_nobias)
{
    TmatmulMXTest<float, uint8_t, uint8_t, false, true, 6, true>(48, 64, 128, 47, 62, 128);
}

TEST_F(TMATMULMXTest, splitk_fp4_e2m1_e2m1_k62_bias)
{
    TmatmulMXTest<float, uint8_t, uint8_t, true, true, 7, true>(48, 64, 128, 47, 62, 128);
}

TEST_F(TMATMULMXTest, splitk_fp8_e4m3_e5m2_k62_nobias)
{
    TmatmulMXTest<float, uint8_t, uint8_t, false, false, 8, true>(48, 64, 128, 47, 62, 128);
}

TEST_F(TMATMULMXTest, splitk_fp8_e4m3_e5m2_k72_bias)
{
    TmatmulMXTest<float, uint8_t, uint8_t, true, false, 9, true>(48, 128, 128, 47, 72, 128);
}

TEST_F(TMATMULMXTest, splitk_fp4_e1m2_e2m1_k72_bias)
{
    TmatmulMXTest<float, uint8_t, uint8_t, true, true, 10, true>(48, 128, 128, 47, 72, 128);
}

TEST_F(TMATMULMXTest, splitk_fp4_e1m2_e2m1_k64_nobias)
{
    TmatmulMXTest<float, uint8_t, uint8_t, false, true, 11, true>(48, 64, 128, 47, 64, 128);
}

TEST_F(TMATMULMXTest, splitk_fp8_e4m3_e5m2_k64_bias)
{
    TmatmulMXTest<float, uint8_t, uint8_t, true, false, 12, true>(48, 64, 128, 47, 64, 128);
}

TEST_F(TMATMULMXTest, splitk_fp8_e4m3_e5m2_k96_nobias)
{
    TmatmulMXTest<float, uint8_t, uint8_t, false, false, 13, true>(48, 128, 128, 47, 96, 128);
}

TEST_F(TMATMULMXTest, splitk_fp4_e2m1_e2m1_k126_nobias)
{
    TmatmulMXTest<float, uint8_t, uint8_t, false, true, 14, true>(48, 128, 128, 47, 126, 128);
}

TEST_F(TMATMULMXTest, splitk_fp4_e1m2_e1m2_k62_nobias_dynamic)
{
    TmatmulMXTest<float, uint8_t, uint8_t, false, true, 15, true>(48, 64, 128, 47, 62, 128);
}

TEST_F(TMATMULMXTest, splitk_fp4_e1m2_e1m2_k62_bias_dynamic)
{
    TmatmulMXTest<float, uint8_t, uint8_t, true, true, 16, true>(48, 64, 128, 47, 62, 128);
}

TEST_F(TMATMULMXTest, splitk_fp4_e1m2_e2m1_k62_nobias_dynamic)
{
    TmatmulMXTest<float, uint8_t, uint8_t, false, true, 17, true>(48, 64, 128, 47, 62, 128);
}

TEST_F(TMATMULMXTest, splitk_fp4_e1m2_e2m1_k62_bias_dynamic)
{
    TmatmulMXTest<float, uint8_t, uint8_t, true, true, 18, true>(48, 64, 128, 47, 62, 128);
}

TEST_F(TMATMULMXTest, splitk_fp4_e2m1_e1m2_k62_nobias_dynamic)
{
    TmatmulMXTest<float, uint8_t, uint8_t, false, true, 19, true>(48, 64, 128, 47, 62, 128);
}

TEST_F(TMATMULMXTest, splitk_fp4_e2m1_e1m2_k62_bias_dynamic)
{
    TmatmulMXTest<float, uint8_t, uint8_t, true, true, 20, true>(48, 64, 128, 47, 62, 128);
}

TEST_F(TMATMULMXTest, splitk_fp4_e2m1_e2m1_k62_nobias_dynamic)
{
    TmatmulMXTest<float, uint8_t, uint8_t, false, true, 21, true>(48, 64, 128, 47, 62, 128);
}

TEST_F(TMATMULMXTest, splitk_fp4_e2m1_e2m1_k62_bias_dynamic)
{
    TmatmulMXTest<float, uint8_t, uint8_t, true, true, 22, true>(48, 64, 128, 47, 62, 128);
}

TEST_F(TMATMULMXTest, splitk_fp8_e4m3_e5m2_k62_nobias_dynamic)
{
    TmatmulMXTest<float, uint8_t, uint8_t, false, false, 23, true>(48, 64, 128, 47, 62, 128);
}

TEST_F(TMATMULMXTest, splitk_fp8_e4m3_e5m2_k72_bias_dynamic)
{
    TmatmulMXTest<float, uint8_t, uint8_t, true, false, 24, true>(48, 128, 128, 47, 72, 128);
}

TEST_F(TMATMULMXTest, splitk_fp4_e1m2_e2m1_k72_bias_dynamic)
{
    TmatmulMXTest<float, uint8_t, uint8_t, true, true, 25, true>(48, 128, 128, 47, 72, 128);
}

TEST_F(TMATMULMXTest, splitk_fp4_e1m2_e2m1_k64_nobias_dynamic)
{
    TmatmulMXTest<float, uint8_t, uint8_t, false, true, 26, true>(48, 64, 128, 47, 64, 128);
}

TEST_F(TMATMULMXTest, splitk_fp8_e4m3_e5m2_k64_bias_dynamic)
{
    TmatmulMXTest<float, uint8_t, uint8_t, true, false, 27, true>(48, 64, 128, 47, 64, 128);
}

TEST_F(TMATMULMXTest, splitk_fp8_e4m3_e5m2_k96_nobias_dynamic)
{
    TmatmulMXTest<float, uint8_t, uint8_t, false, false, 28, true>(48, 128, 128, 47, 96, 128);
}

TEST_F(TMATMULMXTest, splitk_fp4_e2m1_e2m1_k126_nobias_dynamic)
{
    TmatmulMXTest<float, uint8_t, uint8_t, false, true, 29, true>(48, 128, 128, 47, 126, 128);
}

TEST_F(TMATMULMXTest, splitk_gemv_fp4_k62_bias)
{
    TmatmulMXTest<float, uint8_t, uint8_t, true, true, 0, true, true>(16, 62, 64, 1, 62, 64);
}

TEST_F(TMATMULMXTest, splitk_gemv_fp4_k1032_bias)
{
    TmatmulMXTest<float, uint8_t, uint8_t, true, true, 1, true, true>(16, 1032, 64, 1, 1032, 64);
}

TEST_F(TMATMULMXTest, splitk_gemv_fp4_k1056_bias)
{
    TmatmulMXTest<float, uint8_t, uint8_t, true, true, 2, true, true>(16, 1056, 64, 1, 1056, 64);
}

TEST_F(TMATMULMXTest, splitk_gemv_fp4_k1024_bias)
{
    TmatmulMXTest<float, uint8_t, uint8_t, true, true, 3, true, true>(16, 1024, 64, 1, 1024, 64);
}

TEST_F(TMATMULMXTest, splitk_gemv_fp8_k62_bias)
{
    TmatmulMXTest<float, uint8_t, uint8_t, true, false, 4, true, true>(16, 62, 64, 1, 62, 64);
}

TEST_F(TMATMULMXTest, splitk_gemv_fp8_k1032_bias)
{
    TmatmulMXTest<float, uint8_t, uint8_t, true, false, 5, true, true>(16, 1032, 64, 1, 1032, 64);
}
