/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software; you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include "test_common.h"
#include <pto/pto-inst.hpp>
#include <gtest/gtest.h>

using namespace PtoTestCommon;

template <
    typename T, int RowsSrc0, int ColsSrc0, int RowsSrc1, int ColsSrc1, int RowsOut, int ColsOut>
void LaunchTPARTMUL(T* out, T* src0, T* src1, void* stream);

class TPARTMUL_Test : public testing::Test {};

namespace {

constexpr int kDeviceId = 0;
constexpr float kEpsilon = 0.001f;

} // namespace

static std::string GetGoldenDir()
{
    const testing::TestInfo* testInfo = testing::UnitTest::GetInstance()->current_test_info();
    return "../testcases/" + std::string(testInfo->test_suite_name()) + "." + testInfo->name();
}

template <typename T, int RowsSrc0, int ColsSrc0, int RowsSrc1, int ColsSrc1, int RowsOut, int ColsOut>
void test_tpartmul()
{
    const size_t src0Size = static_cast<size_t>(RowsSrc0) * static_cast<size_t>(ColsSrc0) * sizeof(T);
    const size_t src1Size = static_cast<size_t>(RowsSrc1) * static_cast<size_t>(ColsSrc1) * sizeof(T);
    const size_t outSize = static_cast<size_t>(RowsOut) * static_cast<size_t>(ColsOut) * sizeof(T);

    aclInit(nullptr);
    aclrtSetDevice(kDeviceId);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    T *dstHost, *src0Host, *src1Host;
    T *dstDevice, *src0Device, *src1Device;
    aclrtMallocHost((void**)(&dstHost), outSize);
    aclrtMallocHost((void**)(&src0Host), src0Size);
    aclrtMallocHost((void**)(&src1Host), src1Size);
    aclrtMalloc((void**)&dstDevice, outSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&src0Device, src0Size, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&src1Device, src1Size, ACL_MEM_MALLOC_HUGE_FIRST);

    size_t readSize = 0;
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/input1.bin", readSize, src0Host, src0Size));
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/input2.bin", readSize, src1Host, src1Size));
    aclrtMemcpy(src0Device, src0Size, src0Host, src0Size, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src1Device, src1Size, src1Host, src1Size, ACL_MEMCPY_HOST_TO_DEVICE);

    LaunchTPARTMUL<T, RowsSrc0, ColsSrc0, RowsSrc1, ColsSrc1, RowsOut, ColsOut>(
        dstDevice, src0Device, src1Device, stream);
    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, outSize, dstDevice, outSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output.bin", dstHost, outSize);

    aclrtFree(dstDevice);
    aclrtFree(src0Device);
    aclrtFree(src1Device);
    aclrtFreeHost(dstHost);
    aclrtFreeHost(src0Host);
    aclrtFreeHost(src1Host);
    aclrtDestroyStream(stream);
    aclrtResetDevice(kDeviceId);
    aclFinalize();

    std::vector<T> golden(static_cast<size_t>(RowsOut) * static_cast<size_t>(ColsOut));
    std::vector<T> out(static_cast<size_t>(RowsOut) * static_cast<size_t>(ColsOut));
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/golden.bin", readSize, golden.data(), outSize));
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/output.bin", readSize, out.data(), outSize));
    EXPECT_TRUE(ResultCmp<T>(golden, out.data(), kEpsilon));
}

TEST_F(TPARTMUL_Test, case_float_64x64_64x64_64x64)
{
    test_tpartmul<float, 64, 64, 64, 64, 64, 64>();
}

TEST_F(TPARTMUL_Test, case_float_64x64_64x64_32x32)
{
    test_tpartmul<float, 64, 64, 32, 32, 64, 64>();
}

TEST_F(TPARTMUL_Test, case_float_64x64_32x32_64x64)
{
    test_tpartmul<float, 32, 32, 64, 64, 64, 64>();
}

TEST_F(TPARTMUL_Test, case_float_64x64_64x64_32x64)
{
    test_tpartmul<float, 64, 64, 32, 64, 64, 64>();
}

TEST_F(TPARTMUL_Test, case_float_64x64_64x64_64x32)
{
    test_tpartmul<float, 64, 64, 64, 32, 64, 64>();
}

#define DEFINE_TPARTMUL_INTEGER_TESTS(dtype, name)                                             \
    TEST_F(TPARTMUL_Test, case_##name##_64x64_64x64_64x64)                                      \
    {                                                                                           \
        test_tpartmul<dtype, 64, 64, 64, 64, 64, 64>();                                          \
    }                                                                                           \
    TEST_F(TPARTMUL_Test, case_##name##_64x64_64x64_32x32)                                      \
    {                                                                                           \
        test_tpartmul<dtype, 64, 64, 32, 32, 64, 64>();                                         \
    }

DEFINE_TPARTMUL_INTEGER_TESTS(int8_t, int8);
DEFINE_TPARTMUL_INTEGER_TESTS(uint8_t, uint8);
DEFINE_TPARTMUL_INTEGER_TESTS(int16_t, int16);
DEFINE_TPARTMUL_INTEGER_TESTS(uint16_t, uint16);
DEFINE_TPARTMUL_INTEGER_TESTS(int32_t, int32);
DEFINE_TPARTMUL_INTEGER_TESTS(uint32_t, uint32);
DEFINE_TPARTMUL_INTEGER_TESTS(int64_t, int64);
DEFINE_TPARTMUL_INTEGER_TESTS(uint64_t, uint64);

#undef DEFINE_TPARTMUL_INTEGER_TESTS

TEST_F(TPARTMUL_Test, case_half_16x256_16x256_16x256)
{
    test_tpartmul<aclFloat16, 16, 256, 16, 256, 16, 256>();
}

#ifdef CPU_SIM_BFLOAT_ENABLED
TEST_F(TPARTMUL_Test, case_bf16_16x256_16x256_16x256)
{
    test_tpartmul<bfloat16_t, 16, 256, 16, 256, 16, 256>();
}
#endif
