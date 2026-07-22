/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <gtest/gtest.h>

#include "tput_async_kernel.h"
#include "../async_post_stability_kernel.h"
#include "../comm_mpi.h"

namespace {

int FirstDeviceId()
{
    const char* value = std::getenv("PTO_COMM_ST_FIRST_DEVICE_ID");
    if (value == nullptr || *value == '\0') {
        return 0;
    }
    errno = 0;
    char* end = nullptr;
    long parsed = std::strtol(value, &end, 10);
    return errno == 0 && end != value && *end == '\0' && parsed >= 0 ? static_cast<int>(parsed) : 0;
}

bool RunPostStabilityCase(int nRanks, AsyncCheckMode checkMode, uint32_t postCount, uint32_t rounds, uint32_t queueNum)
{
    return RunAsyncPostStability(
        nRanks, nRanks, 0, FirstDeviceId(), AsyncTransferKind::TPut, checkMode, postCount, rounds, queueNum);
}

class TPutAsyncPostStability2Ranks : public ::testing::Test {
protected:
    void SetUp() override
    {
        constexpr int kRanks = 2;
        if (CommMpiSize() != kRanks) {
            GTEST_SKIP() << "Requires exactly 2 MPI ranks";
        }
        if (!IsAsyncPostStabilityDeviceRangeAvailable(kRanks, FirstDeviceId())) {
            GTEST_SKIP() << "Requested device range is unavailable";
        }
    }
};

class TPutAsyncPostStability3Ranks : public ::testing::Test {
protected:
    void SetUp() override
    {
        constexpr int kRanks = 3;
        if (CommMpiSize() != kRanks) {
            GTEST_SKIP() << "Requires exactly 3 MPI ranks";
        }
        if (!IsAsyncPostStabilityDeviceRangeAvailable(kRanks, FirstDeviceId())) {
            GTEST_SKIP() << "Requested device range is unavailable";
        }
    }
};

} // namespace

// ============================================================================
// 1D Vector Tile Tests
// ============================================================================
TEST(TPutAsync, Vec_FloatSmall_4Ranks) { ASSERT_TRUE((RunPutAsyncRootPut<float, 256>(4, 4, 0, 0))); }
TEST(TPutAsync, Vec_Int32Large) { ASSERT_TRUE((RunPutAsyncRootPut<int32_t, 4096>(2, 2, 0, 0))); }
TEST(TPutAsync, Vec_Uint8Small_8Ranks) { ASSERT_TRUE((RunPutAsyncRootPut<uint8_t, 512>(8, 8, 0, 0))); }

// ============================================================================
// Configurable SdmaBaseConfig Tests
// ============================================================================
TEST(TPutAsync, Vec_Int32_QueueNum2) { ASSERT_TRUE((RunPutAsyncWithConfig<int32_t, 4096>(2, 2, 0, 0, 4096, 0, 2))); }
TEST(TPutAsync, Vec_Float_SmallBlockBytes)
{
    ASSERT_TRUE((RunPutAsyncWithConfig<float, 4096>(2, 2, 0, 0, 4096, 0, 1)));
}
TEST(TPutAsync, Vec_Float_LargeBlockBytes)
{
    ASSERT_TRUE((RunPutAsyncWithConfig<float, 4096>(2, 2, 0, 0, 2 * 1024 * 1024, 0, 1)));
}
TEST(TPutAsync, Vec_Float_CommOffset)
{
    ASSERT_TRUE((RunPutAsyncWithConfig<float, 2048>(2, 2, 0, 0, 1024 * 1024, 1024 * sizeof(float), 1)));
}

// ============================================================================
// Multi-Core Tests (blockDim > 1)
// ============================================================================
TEST(TPutAsync, Vec_Float_MultiCoreSplit) { ASSERT_TRUE((RunPutAsyncMultiCore<float, 2048>(2, 2, 0, 0, 2, 0))); }
TEST(TPutAsync, Vec_Float_MultiCoreIndep) { ASSERT_TRUE((RunPutAsyncMultiCore<float, 256>(2, 2, 0, 0, 2, 1))); }

// ============================================================================
// Concurrent Per-Rank Scatter Tests (every rank: nranks cores, distinct channels)
// Mirrors the tget_async ConcurrentRank pattern with TPUT_ASYNC.
// ============================================================================
// iters=1: single TPUT+Wait per core session (baseline concurrency).
TEST(TPutAsync, ConcurrentRank_Float_8Ranks)
{
    ASSERT_TRUE((RunPutAsyncConcurrentRank<float, 8192>(8, 8, 0, 0, 1, 0)));
}
TEST(TPutAsync, ConcurrentRank_Int32_8Ranks)
{
    ASSERT_TRUE((RunPutAsyncConcurrentRank<int32_t, 8192>(8, 8, 0, 0, 1, 0)));
}
// iters=16 reusing one session per core.
TEST(TPutAsync, ConcurrentRank_FloatIter16Reuse_8Ranks)
{
    ASSERT_TRUE((RunPutAsyncConcurrentRank<float, 8192>(8, 8, 0, 0, 16, 0)));
}
// iters=16 rebuilding a fresh session each round.
TEST(TPutAsync, ConcurrentRank_FloatIter16Fresh_8Ranks)
{
    ASSERT_TRUE((RunPutAsyncConcurrentRank<float, 8192>(8, 8, 0, 0, 16, 1)));
}

TEST_F(TPutAsyncPostStability2Ranks, Immediate_P4_Q1)
{
    ASSERT_TRUE(RunPostStabilityCase(2, AsyncCheckMode::Immediate, 4, 1, 1));
}

TEST_F(TPutAsyncPostStability2Ranks, Deferred_P16_Q4)
{
    ASSERT_TRUE(RunPostStabilityCase(2, AsyncCheckMode::Deferred, 16, 1, 4));
}

TEST_F(TPutAsyncPostStability2Ranks, Immediate_512Posts_Q4)
{
    ASSERT_TRUE(RunPostStabilityCase(2, AsyncCheckMode::Immediate, 8, 64, 4));
}

TEST_F(TPutAsyncPostStability2Ranks, UsedQueueCount_Q4ThenQ1_LastWaitOnly)
{
    ASSERT_TRUE(RunPostStabilityCase(2, AsyncCheckMode::LastWaitOnly, 1, 1, 4));
}

TEST_F(TPutAsyncPostStability3Ranks, Deferred_P8_Q4_3Ranks)
{
    ASSERT_TRUE(RunPostStabilityCase(3, AsyncCheckMode::Deferred, 8, 1, 4));
}

int main(int argc, char** argv)
{
    CommMpiInit(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    int ret = RUN_ALL_TESTS();
    CommMpiFinalize();
    return ret;
}
