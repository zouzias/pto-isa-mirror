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

#include "tprefetch_compare_kernel.h"

// ============================================================================
// Single-card prefetch comparison: Scenarios A / B / C / F.
// Cross-rank Scenario D lives under comm/st/testcase/tprefetch_compare/.
//
// Statistics: each TEST runs 100 iterations (override via env
// TPREFETCH_COMPARE_ITER) with 1 warmup. Reports p5/p50/p95 to console and
// appends per-(size, config) rows to CSV files in CWD (override location via
// env TPREFETCH_COMPARE_CSV_DIR):
//   tprefetch_compare_scenarioA.csv
//   tprefetch_compare_scenarioB.csv
//   tprefetch_compare_scenarioC.csv
//   tprefetch_compare_scenarioF.csv
//   tprefetch_compare_scenarioE1.csv
//
// Scenario A scans 5 sizes (64KB, 1MB, 16MB, 64MB, 128MB) for a "real-world
// prefetch + warm TLOAD" comparison. Scenario B is a payload scan of issue
// overhead. Scenario C measures overlap with compute at 3 representative
// sizes. Scenario F sweeps N (number of consecutive prefetch calls over
// disjoint regions) at fixed chunk=64KB to expose the structural difference
// between "N host calls" and "1 kernel launch + N in-kernel issues".
//
// Scenario E1 is a pure-overhead microbenchmark (empty AICore kernel) used
// to isolate the device-path mandatory "launch + dispatch + sync" tax so
// any other scenario's device wall can be normalised against it. See
// PROFILING.md in this directory for how to combine E1 with msprof to fully
// localise where the host-vs-device gap lives.
// ============================================================================

// ---- Scenario A: end-to-end wall (5 sizes) -------------------------------
TEST(TPrefetchCompare, A_EndToEnd_64KB)
{
    ASSERT_TRUE((RunScenarioAEndToEnd<float, 16384>(0)));
}

TEST(TPrefetchCompare, A_EndToEnd_1MB)
{
    ASSERT_TRUE((RunScenarioAEndToEnd<float, 262144>(0)));
}

TEST(TPrefetchCompare, A_EndToEnd_16MB)
{
    ASSERT_TRUE((RunScenarioAEndToEnd<float, 4194304>(0)));
}

TEST(TPrefetchCompare, A_EndToEnd_64MB)
{
    ASSERT_TRUE((RunScenarioAEndToEnd<float, 16777216>(0)));
}

TEST(TPrefetchCompare, A_EndToEnd_128MB)
{
    ASSERT_TRUE((RunScenarioAEndToEnd<float, 33554432>(0)));
}

// ---- Scenario B: prefetch-issue overhead (payload scan) -----------------
// All B tests share the same 16 MB allocation so any payload up to 16 MB
// fits without re-instantiating the kernel for each size. Scanning payloads
// from 4 KB through 4 MB lets us see two things:
//   1. Below ~64 KB the wall is dominated by the fixed software stack
//      (issue + completion roundtrip). host wall and device wall should
//      be roughly flat across these sizes — that's what proves the
//      overhead is a constant rather than a payload-dependent cost.
//   2. Above ~256 KB the SDMA transfer time starts to dominate; the gap
//      between B and Scenario A at the same payload should shrink.
TEST(TPrefetchCompare, B_IssueOverhead_4KB)
{
    ASSERT_TRUE((RunScenarioBIssueOverhead<float, 4194304>(0, 4ULL * 1024)));
}

TEST(TPrefetchCompare, B_IssueOverhead_16KB)
{
    ASSERT_TRUE((RunScenarioBIssueOverhead<float, 4194304>(0, 16ULL * 1024)));
}

TEST(TPrefetchCompare, B_IssueOverhead_64KB)
{
    ASSERT_TRUE((RunScenarioBIssueOverhead<float, 4194304>(0, 64ULL * 1024)));
}

TEST(TPrefetchCompare, B_IssueOverhead_256KB)
{
    ASSERT_TRUE((RunScenarioBIssueOverhead<float, 4194304>(0, 256ULL * 1024)));
}

TEST(TPrefetchCompare, B_IssueOverhead_1MB)
{
    ASSERT_TRUE((RunScenarioBIssueOverhead<float, 4194304>(0, 1024ULL * 1024)));
}

TEST(TPrefetchCompare, B_IssueOverhead_4MB)
{
    ASSERT_TRUE((RunScenarioBIssueOverhead<float, 4194304>(0, 4096ULL * 1024)));
}

// ---- Scenario C: overlap with compute (3 representative sizes) ----------
TEST(TPrefetchCompare, C_Overlap_1MB)
{
    ASSERT_TRUE((RunScenarioCOverlap<float, 262144>(0)));
}

TEST(TPrefetchCompare, C_Overlap_16MB)
{
    ASSERT_TRUE((RunScenarioCOverlap<float, 4194304>(0)));
}

TEST(TPrefetchCompare, C_Overlap_128MB)
{
    ASSERT_TRUE((RunScenarioCOverlap<float, 33554432>(0)));
}

// ---- Scenario F: chunked prefetch (one launch vs N host calls) ----------
// Fixed chunkBytes = 64 KB; N is varied so we can observe the per-call cost
// of the host path accumulating linearly while the device path stays roughly
// constant per chunk (1 launch shell + N x in-kernel issue+Wait). Each TEST
// appends two rows (host_async, device_kernel) to scenarioF CSV; combine
// across N to plot wall(N) for both paths and read off the crossover N.
//
// Total data per N (chunk fixed at 64 KB):
//   N=1    -> 64 KB    (degenerate: identical to Scenario A 64KB single call)
//   N=4    -> 256 KB
//   N=16   -> 1 MB
//   N=64   -> 4 MB
//   N=256  -> 16 MB
//   N=1024 -> 64 MB
TEST(TPrefetchCompare, F_MultiPrefetch_N1_64KB)
{
    ASSERT_TRUE(RunScenarioFMultiChunkPrefetch(0, 64ULL * 1024, 1));
}

TEST(TPrefetchCompare, F_MultiPrefetch_N4_64KB)
{
    ASSERT_TRUE(RunScenarioFMultiChunkPrefetch(0, 64ULL * 1024, 4));
}

TEST(TPrefetchCompare, F_MultiPrefetch_N16_64KB)
{
    ASSERT_TRUE(RunScenarioFMultiChunkPrefetch(0, 64ULL * 1024, 16));
}

TEST(TPrefetchCompare, F_MultiPrefetch_N64_64KB)
{
    ASSERT_TRUE(RunScenarioFMultiChunkPrefetch(0, 64ULL * 1024, 64));
}

TEST(TPrefetchCompare, F_MultiPrefetch_N256_64KB)
{
    ASSERT_TRUE(RunScenarioFMultiChunkPrefetch(0, 64ULL * 1024, 256));
}

TEST(TPrefetchCompare, F_MultiPrefetch_N1024_64KB)
{
    ASSERT_TRUE(RunScenarioFMultiChunkPrefetch(0, 64ULL * 1024, 1024));
}

// ---- Scenario E1: kernel launch + dispatch + sync overhead --------------
// Localises the "device path mandatory launch tax" so we can subtract it
// from any other device-path measurement to get a fair instruction-vs-
// instruction comparison. Run alongside Scenario A; the result is a single
// number (p50, microseconds) that you subtract from
// `tprefetch_compare_scenarioA.csv` device-path wall_p50.
TEST(TPrefetchCompare, E1_NoopKernel)
{
    ASSERT_TRUE(RunScenarioE1NoopKernel(0));
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
