/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>

#include "../comm_mpi.h"
#include "../tput_async_hns1825/tput_async_hns1825_kernel.h"

template <typename T, size_t count>
void ExpectGetPlan(
    int elemOffset, int elemCount, int operationCount, RoceHns1825CompletionMode completionMode, int nRanks = 2,
    int nDevices = 2)
{
    RoceHns1825TestResult result = RunGetAsyncHns1825RootGetPlan<T, count>(
        nRanks, nDevices, 0, 0, elemOffset, elemCount, operationCount, completionMode);
    if (result == RoceHns1825TestResult::SKIPPED) {
        GTEST_SKIP() << "HNS1825 runtime prerequisites are unavailable on at least one rank";
    }
    ASSERT_EQ(result, RoceHns1825TestResult::PASSED);
}

template <typename T, size_t count>
void ExpectGetResult(int nRanks = 2, int nDevices = 2)
{
    ExpectGetPlan<T, count>(
        0, static_cast<int>(count), 1, RoceHns1825CompletionMode::STATUS_WAIT_EACH, nRanks, nDevices);
}

// GET is covered by its own target, parallel to the PUT target.
TEST(TGetAsyncHns1825, Vec_FloatSmall)
{
    SKIP_IF_RANKS_LT(2);
    ExpectGetResult<float, 256>();
}

TEST(TGetAsyncHns1825, Vec_Int32Large)
{
    SKIP_IF_RANKS_LT(2);
    ExpectGetResult<int32_t, 4096>();
}

TEST(TGetAsyncHns1825, Vec_Uint8Small)
{
    SKIP_IF_RANKS_LT(2);
    ExpectGetResult<uint8_t, 512>();
}

TEST(TGetAsyncHns1825, Vec_Uint8_64B)
{
    SKIP_IF_RANKS_LT(2);
    ExpectGetResult<uint8_t, 64>();
}

TEST(TGetAsyncHns1825, Vec_Float_256B)
{
    SKIP_IF_RANKS_LT(2);
    ExpectGetResult<float, 64>();
}

TEST(TGetAsyncHns1825, Vec_Uint8_Offset_63B)
{
    SKIP_IF_RANKS_LT(2);
    ExpectGetPlan<uint8_t, 512>(17, 63, 1, RoceHns1825CompletionMode::STATUS_WAIT_EACH);
}

TEST(TGetAsyncHns1825, Vec_Int32_MultiWqe_WaitEach)
{
    SKIP_IF_RANKS_LT(2);
    ExpectGetPlan<int32_t, 4096>(0, 256, 16, RoceHns1825CompletionMode::STATUS_WAIT_EACH);
}

TEST(TGetAsyncHns1825, Vec_Int32_MultiWqe_WaitLast)
{
    SKIP_IF_RANKS_LT(2);
    ExpectGetPlan<int32_t, 4096>(0, 256, 16, RoceHns1825CompletionMode::STATUS_WAIT_LAST);
}

TEST(TGetAsyncHns1825, Vec_Float_PublicEventWaitTest)
{
    SKIP_IF_RANKS_LT(2);
    ExpectGetPlan<float, 256>(0, 256, 1, RoceHns1825CompletionMode::PUBLIC_EVENT_WAIT_TEST);
}

TEST(TGetAsyncHns1825, Vec_Float_MR_6MB)
{
    SKIP_IF_RANKS_LT(2);
    ExpectGetResult<float, 524288>();
}

TEST(TGetAsyncHns1825, Vec_FloatSmall_4Ranks)
{
    SKIP_IF_RANKS_LT(4);
    ExpectGetResult<float, 256>(4, 4);
}

int main(int argc, char** argv)
{
    if (!CommMpiInit(&argc, &argv)) {
        return 1;
    }
    ::testing::InitGoogleTest(&argc, argv);
    int ret = RUN_ALL_TESTS();
    CommMpiFinalize();
    return ret;
}
