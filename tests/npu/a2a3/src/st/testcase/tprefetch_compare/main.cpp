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
// Single-card prefetch comparison: Scenarios A / B / C.
// Cross-rank Scenario D lives under comm/st/testcase/tprefetch_compare/.
//
// Statistics: each TEST runs 100 iterations (override via env
// TPREFETCH_COMPARE_ITER) with 1 warmup. Reports p5/p50/p95 to console and
// appends per-(size, config) rows to CSV files in CWD (override location via
// env TPREFETCH_COMPARE_CSV_DIR):
//   tprefetch_compare_scenarioA.csv
//   tprefetch_compare_scenarioB.csv
//   tprefetch_compare_scenarioC.csv
//
// Scenario A scans 5 sizes (64KB, 1MB, 16MB, 64MB, 128MB) for a "real-world
// prefetch + warm TLOAD" comparison. Scenario B is a single 4KB-payload
// micro-benchmark of issue overhead. Scenario C measures overlap with
// compute at 3 representative sizes.
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

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
