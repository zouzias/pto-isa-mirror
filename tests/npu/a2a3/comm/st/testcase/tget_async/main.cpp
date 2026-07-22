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

#include "tget_async_kernel.h"
#include "../async_generation_stability/async_generation_stability_kernel.h"
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

bool RunGenerationCase(int nRanks, AsyncCheckMode checkMode, uint32_t postCount, uint32_t rounds, uint32_t queueNum)
{
    return RunAsyncGenerationStability(
        nRanks, nRanks, 0, FirstDeviceId(), AsyncTransferKind::TGet, checkMode, postCount, rounds, queueNum);
}

class TGetAsyncGeneration2Ranks : public ::testing::Test {
protected:
    void SetUp() override
    {
        constexpr int kRanks = 2;
        if (CommMpiSize() != kRanks) {
            GTEST_SKIP() << "Requires exactly 2 MPI ranks";
        }
        if (!IsAsyncGenerationStabilityDeviceRangeAvailable(kRanks, FirstDeviceId())) {
            GTEST_SKIP() << "Requested device range is unavailable";
        }
    }
};

class TGetAsyncGeneration3Ranks : public ::testing::Test {
protected:
    void SetUp() override
    {
        constexpr int kRanks = 3;
        if (CommMpiSize() != kRanks) {
            GTEST_SKIP() << "Requires exactly 3 MPI ranks";
        }
        if (!IsAsyncGenerationStabilityDeviceRangeAvailable(kRanks, FirstDeviceId())) {
            GTEST_SKIP() << "Requested device range is unavailable";
        }
    }
};

} // namespace

// ============================================================================
// 1D Vector Tile Tests
// ============================================================================
TEST(TGetAsync, Vec_FloatSmall_4Ranks) { ASSERT_TRUE((RunGetAsyncRootGet<float, 256>(4, 4, 0, 0))); }
TEST(TGetAsync, Vec_Int32Large) { ASSERT_TRUE((RunGetAsyncRootGet<int32_t, 4096>(2, 2, 0, 0))); }
TEST(TGetAsync, Vec_Uint8Small_8Ranks) { ASSERT_TRUE((RunGetAsyncRootGet<uint8_t, 512>(8, 8, 0, 0))); }

// ============================================================================
// Configurable SdmaBaseConfig Tests
// ============================================================================
TEST(TGetAsync, Vec_Int32_QueueNum2) { ASSERT_TRUE((RunGetAsyncWithConfig<int32_t, 4096>(2, 2, 0, 0, 4096, 0, 2))); }
TEST(TGetAsync, Vec_Float_SmallBlockBytes)
{
    ASSERT_TRUE((RunGetAsyncWithConfig<float, 4096>(2, 2, 0, 0, 4096, 0, 1)));
}
TEST(TGetAsync, Vec_Float_LargeBlockBytes)
{
    ASSERT_TRUE((RunGetAsyncWithConfig<float, 4096>(2, 2, 0, 0, 2 * 1024 * 1024, 0, 1)));
}
TEST(TGetAsync, Vec_Float_CommOffset)
{
    ASSERT_TRUE((RunGetAsyncWithConfig<float, 2048>(2, 2, 0, 0, 1024 * 1024, 1024 * sizeof(float), 1)));
}

// ============================================================================
// Multi-Core Tests (blockDim > 1)
// ============================================================================
TEST(TGetAsync, Vec_Float_MultiCoreSplit) { ASSERT_TRUE((RunGetAsyncMultiCore<float, 2048>(2, 2, 0, 0, 2, 0))); }
TEST(TGetAsync, Vec_Float_MultiCoreIndep) { ASSERT_TRUE((RunGetAsyncMultiCore<float, 256>(2, 2, 0, 0, 2, 1))); }

// ============================================================================
// Concurrent Per-Rank Gather Tests (every rank: nranks cores, distinct channels)
// Reproduces the DispatchGather SDMA pattern in isolation.
// ============================================================================
// iters=1: single TGET+Wait per core session (baseline concurrency).
TEST(TGetAsync, ConcurrentRank_Float_8Ranks)
{
    ASSERT_TRUE((RunGetAsyncConcurrentRank<float, 8192>(8, 8, 0, 0, 1, 0)));
}
TEST(TGetAsync, ConcurrentRank_Int32_8Ranks)
{
    ASSERT_TRUE((RunGetAsyncConcurrentRank<int32_t, 8192>(8, 8, 0, 0, 1, 0)));
}
// iters=16 reusing one session per core (mirrors DispatchGather's per-group loop).
TEST(TGetAsync, ConcurrentRank_FloatIter16Reuse_8Ranks)
{
    ASSERT_TRUE((RunGetAsyncConcurrentRank<float, 8192>(8, 8, 0, 0, 16, 0)));
}
// iters=16 rebuilding a fresh session each round: isolates session-reuse as the cause.
TEST(TGetAsync, ConcurrentRank_FloatIter16Fresh_8Ranks)
{
    ASSERT_TRUE((RunGetAsyncConcurrentRank<float, 8192>(8, 8, 0, 0, 16, 1)));
}

TEST_F(TGetAsyncGeneration2Ranks, Immediate_P4_Q1)
{
    ASSERT_TRUE(RunGenerationCase(2, AsyncCheckMode::Immediate, 4, 1, 1));
}

TEST_F(TGetAsyncGeneration2Ranks, Deferred_P16_Q4)
{
    ASSERT_TRUE(RunGenerationCase(2, AsyncCheckMode::Deferred, 16, 1, 4));
}

TEST_F(TGetAsyncGeneration2Ranks, Immediate_512Posts_Q4)
{
    ASSERT_TRUE(RunGenerationCase(2, AsyncCheckMode::Immediate, 8, 64, 4));
}

TEST_F(TGetAsyncGeneration2Ranks, UsedQueueCount_Q4ThenQ1_LastWaitOnly)
{
    ASSERT_TRUE(RunGenerationCase(2, AsyncCheckMode::LastWaitOnly, 1, 1, 4));
}

TEST_F(TGetAsyncGeneration3Ranks, Deferred_P8_Q4_3Ranks)
{
    ASSERT_TRUE(RunGenerationCase(3, AsyncCheckMode::Deferred, 8, 1, 4));
}

int main(int argc, char** argv)
{
    CommMpiInit(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    int ret = RUN_ALL_TESTS();
    CommMpiFinalize();
    return ret;
}
