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
#include <gtest/gtest.h>
#include <pto/pto-inst.hpp>
#include <pto/cpu/MXTypes.hpp>

using namespace std;
using namespace PtoTestCommon;
using namespace pto;

template <typename D, typename S, int kGRows_, int kGCols_, int kTRows_, int kTCols_, pto::SaturationMode saturation>
void launchTCVT(D* dst, S* src, void* stream);

class TCVTTest : public testing::Test {
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
struct IsFP4Type : std::false_type {};

template <>
struct IsFP4Type<float4_e2m1x2_t> : std::true_type {};

template <>
struct IsFP4Type<float4_e1m2x2_t> : std::true_type {};

template <typename T>
constexpr bool is_fp4_type_v = IsFP4Type<T>::value;

template <typename T>
size_t GetElemSizeForFile(uint32_t M, uint32_t N)
{
    if constexpr (is_fp4_type_v<T>) {
        return (M * N + 1) / 2;
    } else {
        return M * N * sizeof(T);
    }
}

template<typename T>
void UnpackFP4(const uint8_t* in, T* out, size_t count)
{
    for(size_t i = 0; i < count; i++)
    {
        uint8_t byte = in[i / 2];
        uint8_t raw;
        if (i & 1) {
            raw = (byte >> 4) & 0xF;
        } else {
            raw = byte & 0xF;
        }
        out[i] = T::FromRaw(raw);
    }
}

template<typename T>
void PackFP4(const T* in, uint8_t* out, size_t count)
{
    memset(out, 0, (count + 1) / 2);
    for(size_t i = 0; i < count; i++)
    {
        uint8_t raw = in[i].RawData() & 0xF;
        if (i & 1) {
            out[i / 2] |= (raw << 4);
        } else {
            out[i / 2] |= raw;
        }
    }
}

template <typename D, typename S, int kGRows_, int kGCols_, int kTRows_, int kTCols_,
          pto::SaturationMode saturation = pto::SaturationMode::OFF>
void test_tcvt()
{
    uint32_t M = kGRows_;
    uint32_t N = kGCols_;

    size_t srcFileSize = GetElemSizeForFile<S>(M, N);
    size_t dstFileSize = GetElemSizeForFile<D>(M, N);
    size_t srcHostSize = M * N * sizeof(S);
    size_t dstHostSize = M * N * sizeof(D);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    D *dstHost, *dstDevice;
    S *srcHost, *srcDevice;

    aclrtMallocHost((void **)(&dstHost), dstHostSize);
    aclrtMallocHost((void **)(&srcHost), srcHostSize);
    aclrtMalloc((void **)&dstDevice, dstHostSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&srcDevice, srcHostSize, ACL_MEM_MALLOC_HUGE_FIRST);

    if constexpr (is_fp4_type_v<S>) {
        std::vector<uint8_t> packed(srcFileSize);
        CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/x1_gm.bin", srcFileSize, packed.data(), srcFileSize));
        UnpackFP4(packed.data(), srcHost, M * N);
    }
    else {
        CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/x1_gm.bin", srcFileSize, srcHost, srcHostSize));
    }

    aclrtMemcpy(srcDevice, srcHostSize, srcHost, srcHostSize, ACL_MEMCPY_HOST_TO_DEVICE);
    launchTCVT<D, S, kGRows_, kGCols_, kTRows_, kTCols_, saturation>(dstDevice, srcDevice, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, dstHostSize, dstDevice, dstHostSize, ACL_MEMCPY_DEVICE_TO_HOST);

    if constexpr (is_fp4_type_v<D>) {
        std::vector<uint8_t> packed(dstFileSize);
        PackFP4(dstHost, packed.data(), M * N);
        WriteFile(GetGoldenDir() + "/output_z.bin", packed.data(), dstFileSize);
    }
    else {
        WriteFile(GetGoldenDir() + "/output_z.bin", dstHost, dstHostSize);
    }

    aclrtFree(dstDevice);
    aclrtFree(srcDevice);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(srcHost);

    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<D> golden(M * N);
    std::vector<D> devFinal(M * N);
    if constexpr (is_fp4_type_v<D>) {
        std::vector<uint8_t> packed(dstFileSize);
        CHECK_RESULT_GTEST(ReadFile(GetGoldenDir()+"/golden.bin", dstFileSize, packed.data(), dstFileSize));
        UnpackFP4(packed.data(), golden.data(), M * N);
    }
    else {
        CHECK_RESULT_GTEST(ReadFile(GetGoldenDir()+"/golden.bin", dstFileSize, golden.data(), dstHostSize));
    }
    
    if constexpr (is_fp4_type_v<D>) {
        std::vector<uint8_t> packed(dstFileSize);
        CHECK_RESULT_GTEST(ReadFile(GetGoldenDir()+"/output_z.bin", dstFileSize, packed.data(), dstFileSize));
        UnpackFP4(packed.data(), devFinal.data(), M * N);
    }
    else {
        CHECK_RESULT_GTEST(ReadFile(GetGoldenDir()+"/output_z.bin", dstFileSize, devFinal.data(), dstHostSize));
    }

    bool ret = ResultCmp<D>(golden, devFinal, 0.001f);

    EXPECT_TRUE(ret);
}

TEST_F(TCVTTest, case1) { test_tcvt<int32_t, float, 128, 128, 128, 128>(); }

TEST_F(TCVTTest, case2) { test_tcvt<float, int32_t, 256, 64, 256, 64>(); }

TEST_F(TCVTTest, case3) { test_tcvt<int16_t, float, 16, 32, 16, 32>(); }

TEST_F(TCVTTest, case4) { test_tcvt<int32_t, float, 32, 512, 32, 512>(); }

TEST_F(TCVTTest, case5) { test_tcvt<int32_t, int16_t, 2, 512, 2, 512>(); }

TEST_F(TCVTTest, case6) { test_tcvt<int32_t, float, 4, 4096, 4, 4096>(); }

TEST_F(TCVTTest, case7) { test_tcvt<float, int16_t, 64, 64, 64, 64>(); }

TEST_F(TCVTTest, case8) { test_tcvt<aclFloat16, float, 64, 64, 64, 64>(); }

TEST_F(TCVTTest, case9) { test_tcvt<uint8_t, aclFloat16, 64, 64, 64, 64>(); }

TEST_F(TCVTTest, case10) { test_tcvt<float, int32_t, 64, 64, 64, 64, pto::SaturationMode::ON>(); }

TEST_F(TCVTTest, case11) { test_tcvt<float, int8_t, 128, 128, 128, 128, pto::SaturationMode::ON>(); }

TEST_F(TCVTTest, case12) { test_tcvt<uint8_t, float, 64, 64, 64, 64, pto::SaturationMode::ON>(); }

TEST_F(TCVTTest, case13) { test_tcvt<int16_t, int32_t, 64, 64, 64, 64, pto::SaturationMode::ON>(); }

TEST_F(TCVTTest, case14) { test_tcvt<int8_t, aclFloat16, 32, 32, 32, 32, pto::SaturationMode::ON>(); }

TEST_F(TCVTTest, case15)
{
    test_tcvt<uint8_t, aclFloat16, 64, 64, 64, 64, pto::SaturationMode::ON>();
}

TEST_F(TCVTTest, case16)
{
    test_tcvt<uint16_t, float, 64, 64, 64, 64>();
}

TEST_F(TCVTTest, case17)
{
    test_tcvt<float, uint16_t, 64, 64, 64, 64>();
}

TEST_F(TCVTTest, case18)
{
    test_tcvt<uint16_t, int32_t, 64, 64, 64, 64, pto::SaturationMode::ON>();
}

TEST_F(TCVTTest, case19)
{
    test_tcvt<int32_t, uint16_t, 64, 64, 64, 64, pto::SaturationMode::ON>();
}

TEST_F(TCVTTest, case20)
{
    test_tcvt<int4b_t, float, 64, 64, 64, 64>();
}

TEST_F(TCVTTest, case21)
{
    test_tcvt<float, int4b_t, 64, 64, 64, 64>();
}

TEST_F(TCVTTest, case22)
{
    test_tcvt<int4b_t, float, 64, 64, 64, 64, pto::SaturationMode::ON>();
}

TEST_F(TCVTTest, case23)
{
    test_tcvt<float, int4b_t, 64, 64, 64, 64, pto::SaturationMode::ON>();
}

TEST_F(TCVTTest, case24)
{
    test_tcvt<float4_e2m1x2_t, float, 64, 64, 64, 64>();
}

TEST_F(TCVTTest, case25)
{
    test_tcvt<float, float4_e2m1x2_t, 64, 64, 64, 64>();
}

TEST_F(TCVTTest, case26)
{
    test_tcvt<float4_e2m1x2_t, float, 64, 64, 64, 64, pto::SaturationMode::ON>();
}

TEST_F(TCVTTest, case27)
{
    test_tcvt<float, float4_e2m1x2_t, 64, 64, 64, 64, pto::SaturationMode::ON>();
}

TEST_F(TCVTTest, case28)
{
    test_tcvt<float4_e1m2x2_t, float, 64, 64, 64, 64>();
}

TEST_F(TCVTTest, case29)
{
    test_tcvt<float, float4_e1m2x2_t, 64, 64, 64, 64>();
}

TEST_F(TCVTTest, case30)
{
    test_tcvt<float4_e1m2x2_t, float, 64, 64, 64, 64, pto::SaturationMode::ON>();
}

TEST_F(TCVTTest, case31)
{
    test_tcvt<float, float4_e1m2x2_t, 64, 64, 64, 64, pto::SaturationMode::ON>();
}
