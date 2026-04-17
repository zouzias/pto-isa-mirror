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
#include <iostream>
#include <gtest/gtest.h>

#include "tprefetch_l2_kernel.h"
#include "../comm_mpi.h"

// ============================================================================
// Single-card Tests (rank 0 only — no HCCL needed)
// ============================================================================
TEST(TPrefetchL2, Baseline_Float_4096)
{
    if (CommMpiRank() != 0) return;
    ASSERT_TRUE((RunBaseline<float, 4096>(0)));
}

TEST(TPrefetchL2, Baseline_Int32_4096)
{
    if (CommMpiRank() != 0) return;
    ASSERT_TRUE((RunBaseline<int32_t, 4096>(0)));
}

TEST(TPrefetchL2, Correctness_Float_4096)
{
    if (CommMpiRank() != 0) return;
    ASSERT_TRUE((RunPrefetchL2Correctness<float, 4096>(0)));
}

TEST(TPrefetchL2, Correctness_Int32_4096)
{
    if (CommMpiRank() != 0) return;
    ASSERT_TRUE((RunPrefetchL2Correctness<int32_t, 4096>(0)));
}

TEST(TPrefetchL2, RawPtr_Float_4096)
{
    if (CommMpiRank() != 0) return;
    ASSERT_TRUE((RunPrefetchL2RawPtr<float, 4096>(0)));
}

TEST(TPrefetchL2, RawPtr_Int32_4096)
{
    if (CommMpiRank() != 0) return;
    ASSERT_TRUE((RunPrefetchL2RawPtr<int32_t, 4096>(0)));
}

// ============================================================================
// Multi-card Tests: TPUT_ASYNC baseline (no prefetch) — sanity check
// ============================================================================
TEST(TPrefetchL2, TputAsyncOnly_Float_4096)
{
    ASSERT_TRUE((RunPrefetchL2TputAsync<float, 4096>(2, 2, 0, 0, false)));
}

TEST(TPrefetchL2, TputAsyncOnly_Int32_4096)
{
    ASSERT_TRUE((RunPrefetchL2TputAsync<int32_t, 4096>(2, 2, 0, 0, false)));
}

// ============================================================================
// Multi-card Tests: TPREFETCH_L2 + TPUT_ASYNC (all ranks participate)
// ============================================================================
TEST(TPrefetchL2, TputAsync_Float_4096)
{
    ASSERT_TRUE((RunPrefetchL2TputAsync<float, 4096>(2, 2, 0, 0, true)));
}

TEST(TPrefetchL2, TputAsync_Int32_4096)
{
    ASSERT_TRUE((RunPrefetchL2TputAsync<int32_t, 4096>(2, 2, 0, 0, true)));
}

// ============================================================================
// Performance Tests: TPUT_ASYNC with vs without TPREFETCH_L2 (multi-card)
// ============================================================================
TEST(TPrefetchL2, Perf_TputAsync_Float_16KB)
{
    ASSERT_TRUE((RunPrefetchL2Perf<float, 4096>(2, 2, 0, 0)));
}

TEST(TPrefetchL2, Perf_TputAsync_Float_256KB)
{
    ASSERT_TRUE((RunPrefetchL2Perf<float, 65536>(2, 2, 0, 0)));
}

TEST(TPrefetchL2, Perf_TputAsync_Float_1MB)
{
    ASSERT_TRUE((RunPrefetchL2Perf<float, 262144>(2, 2, 0, 0)));
}

// ============================================================================
// Performance Tests: TLOAD latency — L2-cold vs L2-prefetched (single-card)
// ============================================================================
TEST(TPrefetchL2, Perf_Tload_Float_16KB)
{
    if (CommMpiRank() != 0) return;
    ASSERT_TRUE((RunTloadPerf<float, 4096>(0)));
}

TEST(TPrefetchL2, Perf_Tload_Float_256KB)
{
    if (CommMpiRank() != 0) return;
    ASSERT_TRUE((RunTloadPerf<float, 65536>(0)));
}

TEST(TPrefetchL2, Perf_Tload_Float_1MB)
{
    if (CommMpiRank() != 0) return;
    ASSERT_TRUE((RunTloadPerf<float, 262144>(0)));
}

TEST(TPrefetchL2, Perf_Tload_Float_4MB)
{
    if (CommMpiRank() != 0) return;
    ASSERT_TRUE((RunTloadPerf<float, 1048576>(0)));
}

// ============================================================================
// Performance Tests: Remote TLOAD — Rank 0 → TPUT_ASYNC → Rank 1 → TLOAD
// ============================================================================
TEST(TPrefetchL2, Perf_RemoteTload_Float_16KB)
{
    ASSERT_TRUE((RunTloadRemotePerf<float, 4096>(2, 2, 0, 0)));
}

TEST(TPrefetchL2, Perf_RemoteTload_Float_256KB)
{
    ASSERT_TRUE((RunTloadRemotePerf<float, 65536>(2, 2, 0, 0)));
}

TEST(TPrefetchL2, Perf_RemoteTload_Float_1MB)
{
    ASSERT_TRUE((RunTloadRemotePerf<float, 262144>(2, 2, 0, 0)));
}

// ============================================================================
// Performance Tests: prefetch before TPUT
// Practical: prefetch local src to L2 → TPUT to remote → measure TPUT time
// ============================================================================
TEST(TPrefetchL2, Perf_TputSync_Float_16KB)
{
    ASSERT_TRUE((RunTputSyncPerf<float, 4096>(2, 2, 0, 0)));
}

TEST(TPrefetchL2, Perf_TputSync_Float_256KB)
{
    ASSERT_TRUE((RunTputSyncPerf<float, 65536>(2, 2, 0, 0)));
}

TEST(TPrefetchL2, Perf_TputSync_Float_1MB)
{
    ASSERT_TRUE((RunTputSyncPerf<float, 262144>(2, 2, 0, 0)));
}

// ============================================================================
// Performance Tests: remote prefetch before TGET
// Practical: remote rank prefetches its data → local rank TGETs → measure time
// ============================================================================
TEST(TPrefetchL2, Perf_Tget_Float_16KB)
{
    ASSERT_TRUE((RunTgetPerf<float, 4096>(2, 2, 0, 0)));
}

TEST(TPrefetchL2, Perf_Tget_Float_256KB)
{
    ASSERT_TRUE((RunTgetPerf<float, 65536>(2, 2, 0, 0)));
}

TEST(TPrefetchL2, Perf_Tget_Float_1MB)
{
    ASSERT_TRUE((RunTgetPerf<float, 262144>(2, 2, 0, 0)));
}

int main(int argc, char **argv)
{
    CommMpiInit(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    int ret = RUN_ALL_TESTS();
    CommMpiFinalize();
    return ret;
}
