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

#include "tput_async_hns1825_kernel.h"
#include "../comm_mpi.h"

template <typename T, size_t count>
void ExpectPutResult(int nRanks = 2, int nDevices = 2)
{
    RoceHns1825TestResult result = RunPutAsyncHns1825RootPut<T, count>(nRanks, nDevices, 0, 0);
    if (result == RoceHns1825TestResult::SKIPPED) {
        GTEST_SKIP() << "HNS1825 runtime prerequisites are unavailable on at least one rank";
    }
    ASSERT_EQ(result, RoceHns1825TestResult::PASSED);
}

template <typename T, size_t count>
void ExpectPutPlan(
    int elemOffset, int elemCount, int operationCount, RoceHns1825CompletionMode completionMode, int nRanks = 2,
    int nDevices = 2)
{
    RoceHns1825TestResult result = RunPutAsyncHns1825RootPutPlan<T, count>(
        nRanks, nDevices, 0, 0, elemOffset, elemCount, operationCount, completionMode);
    if (result == RoceHns1825TestResult::SKIPPED) {
        GTEST_SKIP() << "HNS1825 runtime prerequisites are unavailable on at least one rank";
    }
    ASSERT_EQ(result, RoceHns1825TestResult::PASSED);
}

TEST(TPutAsyncHns1825, Vec_FloatSmall)
{
    SKIP_IF_RANKS_LT(2);
    ExpectPutResult<float, 256>();
}
TEST(TPutAsyncHns1825, Vec_Int32Large)
{
    SKIP_IF_RANKS_LT(2);
    ExpectPutResult<int32_t, 4096>();
}
TEST(TPutAsyncHns1825, Vec_Uint8Small)
{
    SKIP_IF_RANKS_LT(2);
    ExpectPutResult<uint8_t, 512>();
}

TEST(TPutAsyncHns1825, Vec_Uint8_SingleChunk)
{
    SKIP_IF_RANKS_LT(2);
    ExpectPutResult<uint8_t, 64>();
}
TEST(TPutAsyncHns1825, Vec_Float_ExactChunk)
{
    SKIP_IF_RANKS_LT(2);
    ExpectPutResult<float, 64>();
}

// Non-zero source/destination offset with canaries on both sides.
TEST(TPutAsyncHns1825, Vec_Uint8_Offset_63B)
{
    SKIP_IF_RANKS_LT(2);
    ExpectPutPlan<uint8_t, 512>(17, 63, 1, RoceHns1825CompletionMode::STATUS_WAIT_EACH);
}

// Exercise repeated SQ/CQ advancement on one QP instead of rebuilding after
// every WQE. The two completion modes cover per-WQE drain and last-event drain.
TEST(TPutAsyncHns1825, Vec_Int32_MultiWqe_WaitEach)
{
    SKIP_IF_RANKS_LT(2);
    ExpectPutPlan<int32_t, 4096>(0, 256, 16, RoceHns1825CompletionMode::STATUS_WAIT_EACH);
}

TEST(TPutAsyncHns1825, Vec_Int32_MultiWqe_WaitLast)
{
    SKIP_IF_RANKS_LT(2);
    ExpectPutPlan<int32_t, 4096>(0, 256, 16, RoceHns1825CompletionMode::STATUS_WAIT_LAST);
}

// Cover the public AsyncEvent dispatch, including Test before/after Wait.
TEST(TPutAsyncHns1825, Vec_Float_PublicEventWaitTest)
{
    SKIP_IF_RANKS_LT(2);
    ExpectPutPlan<float, 256>(0, 256, 1, RoceHns1825CompletionMode::PUBLIC_EVENT_WAIT_TEST);
}

// Match the first URMA large-MR tier without making the default PUT suite
// allocate hundreds of MiB: 2 MiB payload, approximately 4 MiB registered MR.
TEST(TPutAsyncHns1825, Vec_Float_MR_4MB)
{
    SKIP_IF_RANKS_LT(2);
    ExpectPutResult<float, 524288>();
}

// Run this case with mpirun -n 4 and a matching gtest filter. The 2-rank
// cases require a 2-rank MPI world and are intentionally not mixed into it.
TEST(TPutAsyncHns1825, Vec_FloatSmall_4Ranks)
{
    SKIP_IF_RANKS_LT(4);
    ExpectPutResult<float, 256>(4, 4);
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
