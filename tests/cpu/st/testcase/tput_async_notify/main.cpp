/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include "test_common.h"
#include <pto/pto-inst.hpp>
#include <gtest/gtest.h>

using namespace std;
using namespace PtoTestCommon;

class TPUT_ASYNC_NOTIFY_Test : public testing::Test {
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

template <typename T, int kGRows_, int kGCols_, pto::comm::NotifyOp Op>
void LaunchTPutAsyncNotify(T* out, T* src, int32_t* signal, void* stream);

template <typename T, int kGRows_, int kGCols_, pto::comm::NotifyOp Op>
void test_tput_async_notify()
{
    constexpr int32_t INIT_SIGNAL = 5;
    constexpr int32_t SIGNAL_VALUE = 7;
    constexpr int32_t expectedSignal =
        (Op == pto::comm::NotifyOp::AtomicAdd) ? INIT_SIGNAL + SIGNAL_VALUE : SIGNAL_VALUE;
    size_t fileSize = kGRows_ * kGCols_ * sizeof(T);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    T *dstHost, *srcHost;
    T *dstDevice, *srcDevice;
    int32_t *signalHost, *signalBack;
    int32_t* signalDevice;

    aclrtMallocHost((void**)(&dstHost), fileSize);
    aclrtMallocHost((void**)(&srcHost), fileSize);
    aclrtMallocHost((void**)(&signalHost), sizeof(int32_t));
    aclrtMallocHost((void**)(&signalBack), sizeof(int32_t));

    aclrtMalloc((void**)&dstDevice, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&srcDevice, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&signalDevice, sizeof(int32_t), ACL_MEM_MALLOC_HUGE_FIRST);

    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/input.bin", fileSize, srcHost, fileSize));

    signalHost[0] = INIT_SIGNAL;
    aclrtMemcpy(srcDevice, fileSize, srcHost, fileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(signalDevice, sizeof(int32_t), signalHost, sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE);
    LaunchTPutAsyncNotify<T, kGRows_, kGCols_, Op>(dstDevice, srcDevice, signalDevice, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, fileSize, dstDevice, fileSize, ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(signalBack, sizeof(int32_t), signalDevice, sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST);
    int32_t actualSignal = signalBack[0];

    WriteFile(GetGoldenDir() + "/output.bin", dstHost, fileSize);

    aclrtFree(dstDevice);
    aclrtFree(srcDevice);
    aclrtFree(signalDevice);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(srcHost);
    aclrtFreeHost(signalHost);
    aclrtFreeHost(signalBack);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<T> golden(fileSize / sizeof(T));
    std::vector<T> devFinal(fileSize / sizeof(T));
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/golden.bin", fileSize, golden.data(), fileSize));
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/output.bin", fileSize, devFinal.data(), fileSize));

    bool ret = ResultCmp<T>(golden, devFinal, 0.001f);
    EXPECT_TRUE(ret);
    EXPECT_EQ(expectedSignal, actualSignal);
}

TEST_F(TPUT_ASYNC_NOTIFY_Test, case_float_64x64_set)
{
    test_tput_async_notify<float, 64, 64, pto::comm::NotifyOp::Set>();
}
TEST_F(TPUT_ASYNC_NOTIFY_Test, case_int32_64x64_set)
{
    test_tput_async_notify<int32_t, 64, 64, pto::comm::NotifyOp::Set>();
}
TEST_F(TPUT_ASYNC_NOTIFY_Test, case_int16_64x64_set)
{
    test_tput_async_notify<int16_t, 64, 64, pto::comm::NotifyOp::Set>();
}
TEST_F(TPUT_ASYNC_NOTIFY_Test, case_half_16x256_set)
{
    test_tput_async_notify<aclFloat16, 16, 256, pto::comm::NotifyOp::Set>();
}
TEST_F(TPUT_ASYNC_NOTIFY_Test, case_float_64x64_atomicadd)
{
    test_tput_async_notify<float, 64, 64, pto::comm::NotifyOp::AtomicAdd>();
}
TEST_F(TPUT_ASYNC_NOTIFY_Test, case_int32_64x64_atomicadd)
{
    test_tput_async_notify<int32_t, 64, 64, pto::comm::NotifyOp::AtomicAdd>();
}
TEST_F(TPUT_ASYNC_NOTIFY_Test, case_int16_64x64_atomicadd)
{
    test_tput_async_notify<int16_t, 64, 64, pto::comm::NotifyOp::AtomicAdd>();
}
TEST_F(TPUT_ASYNC_NOTIFY_Test, case_half_16x256_atomicadd)
{
    test_tput_async_notify<aclFloat16, 16, 256, pto::comm::NotifyOp::AtomicAdd>();
}
