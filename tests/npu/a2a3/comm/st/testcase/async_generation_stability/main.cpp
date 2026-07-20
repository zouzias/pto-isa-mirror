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
#include <cstdlib>

#include <gtest/gtest.h>

#include "async_generation_stability_kernel.h"
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
    if (errno != 0 || end == value || *end != '\0' || parsed < 0) {
        return 0;
    }
    return static_cast<int>(parsed);
}

bool RunCase(
    int nRanks, AsyncTransferKind transferKind, AsyncCheckMode checkMode, uint32_t postCount, uint32_t rounds,
    uint32_t queueNum)
{
    return RunAsyncGenerationStability(
        nRanks, nRanks, 0, FirstDeviceId(), transferKind, checkMode, postCount, rounds, queueNum);
}

class AsyncGenerationStability : public ::testing::Test {
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

class AsyncGenerationStabilityOneToTwo : public ::testing::Test {
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

TEST_F(AsyncGenerationStability, TGet_Immediate_P4_Q1)
{
    ASSERT_TRUE(RunCase(2, AsyncTransferKind::TGet, AsyncCheckMode::Immediate, 4, 1, 1));
}

TEST_F(AsyncGenerationStability, TGet_Deferred_P8_Q4)
{
    ASSERT_TRUE(RunCase(2, AsyncTransferKind::TGet, AsyncCheckMode::Deferred, 8, 1, 4));
}

TEST_F(AsyncGenerationStability, TGet_Deferred_P16_Q4)
{
    ASSERT_TRUE(RunCase(2, AsyncTransferKind::TGet, AsyncCheckMode::Deferred, 16, 1, 4));
}

TEST_F(AsyncGenerationStability, TGet_Immediate_512Posts_Q4)
{
    ASSERT_TRUE(RunCase(2, AsyncTransferKind::TGet, AsyncCheckMode::Immediate, 8, 64, 4));
}

TEST_F(AsyncGenerationStability, TPut_Immediate_P4_Q1)
{
    ASSERT_TRUE(RunCase(2, AsyncTransferKind::TPut, AsyncCheckMode::Immediate, 4, 1, 1));
}

TEST_F(AsyncGenerationStability, TPut_Deferred_P8_Q4)
{
    ASSERT_TRUE(RunCase(2, AsyncTransferKind::TPut, AsyncCheckMode::Deferred, 8, 1, 4));
}

TEST_F(AsyncGenerationStability, TPut_Deferred_P16_Q4)
{
    ASSERT_TRUE(RunCase(2, AsyncTransferKind::TPut, AsyncCheckMode::Deferred, 16, 1, 4));
}

TEST_F(AsyncGenerationStability, TPut_Immediate_512Posts_Q4)
{
    ASSERT_TRUE(RunCase(2, AsyncTransferKind::TPut, AsyncCheckMode::Immediate, 8, 64, 4));
}

TEST_F(AsyncGenerationStabilityOneToTwo, TGet_Deferred_P8_Q4_3Ranks)
{
    ASSERT_TRUE(RunCase(3, AsyncTransferKind::TGet, AsyncCheckMode::Deferred, 8, 1, 4));
}

TEST_F(AsyncGenerationStabilityOneToTwo, TPut_Deferred_P8_Q4_3Ranks)
{
    ASSERT_TRUE(RunCase(3, AsyncTransferKind::TPut, AsyncCheckMode::Deferred, 8, 1, 4));
}

int main(int argc, char** argv)
{
    CommMpiInit(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    int ret = RUN_ALL_TESTS();
    CommMpiFinalize();
    return ret;
}
