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
// Single-card prefetch comparison: Scenarios A / B / C / F / G.
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
//   tprefetch_compare_scenarioG.csv
//   tprefetch_compare_scenarioE1.csv
//
// Scenario A scans 5 sizes (64KB, 1MB, 16MB, 64MB, 128MB) for a "real-world
// prefetch + warm TLOAD" comparison. Scenario B is a payload scan of issue
// overhead. Scenario C measures overlap with compute at 3 representative
// sizes. Scenario F holds total bytes = 16 MB and varies N to compare
// "1 big prefetch + TLOAD" against "N small prefetches + same TLOAD".
// Scenario G models a "compute-comm fusion" workload: M consecutive stages
// of (prefetch + compute + TLOAD); host_serial pays M kernel launches with
// no prefetch-compute overlap on a single stream, while device_pipelined
// runs everything inside one fused kernel that overlaps prefetch_{i+1}
// with compute_i — this is the deployment pattern where device should
// structurally win, scaled across M and across compute/prefetch ratios.
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

// ---- Scenario F: chunked prefetch + warm TLOAD (mirror of Scenario A) ---
// Holds total bytes constant (= 16 MB, = Scenario A's A_EndToEnd_16MB) and
// varies N (number of chunks the prefetch is split into). Same TLOAD body
// as Scenario A, same single trailing sync. Each TEST appends two rows
// (host_async, device_kernel) to scenarioF CSV; combine across N to see
// how "1 big prefetch vs N small prefetches" affects end-to-end wall.
//
// Chunk size derived from N (totalBytes / N):
//   N=1    -> chunk=16 MB    (sanity check: should match Scenario A 16MB rows)
//   N=4    -> chunk=4 MB
//   N=16   -> chunk=1 MB
//   N=64   -> chunk=256 KB
//   N=256  -> chunk=64 KB
//   N=1024 -> chunk=16 KB
TEST(TPrefetchCompare, F_ChunkedPrefetchTload_16MB_N1)
{
    ASSERT_TRUE((RunScenarioFChunkedPrefetchAndTload<float, 4194304>(0, 1)));
}

TEST(TPrefetchCompare, F_ChunkedPrefetchTload_16MB_N4)
{
    ASSERT_TRUE((RunScenarioFChunkedPrefetchAndTload<float, 4194304>(0, 4)));
}

TEST(TPrefetchCompare, F_ChunkedPrefetchTload_16MB_N16)
{
    ASSERT_TRUE((RunScenarioFChunkedPrefetchAndTload<float, 4194304>(0, 16)));
}

TEST(TPrefetchCompare, F_ChunkedPrefetchTload_16MB_N64)
{
    ASSERT_TRUE((RunScenarioFChunkedPrefetchAndTload<float, 4194304>(0, 64)));
}

TEST(TPrefetchCompare, F_ChunkedPrefetchTload_16MB_N256)
{
    ASSERT_TRUE((RunScenarioFChunkedPrefetchAndTload<float, 4194304>(0, 256)));
}

TEST(TPrefetchCompare, F_ChunkedPrefetchTload_16MB_N1024)
{
    ASSERT_TRUE((RunScenarioFChunkedPrefetchAndTload<float, 4194304>(0, 1024)));
}

// ---- Scenario G: fused multi-stage prefetch + compute -------------------
// Workload: M consecutive stages, each = (prefetch 1 MB) + (spin compute) +
// (warm TLOAD 1 MB). host_serial is the natural single-stream pattern (M
// kernel launches interleaved with M PTO_PREFETCHs); device_pipelined is
// one fused kernel that overlaps prefetch_{i+1} with compute_i.
//
// Two axes scanned:
//   * M     in {2, 4, 8, 16}  — to see launch-shell savings accumulate
//   * spin  in {5000, 20000, 80000} cycles  ~= {50, 200, 800} us per stage
//                                              (compute < / ~= / >> prefetch)
//
// Predicted device advantage:
//   - For any M >= 2: device saves (M-1) launch shells (~17 us each).
//   - For spin >= prefetch (~23 us @ 1 MB): pipeline fully hides the
//     prefetch under compute, so device wall ~= M*spin + 1 prefetch + 1 launch.
TEST(TPrefetchCompare, G_Fused_M2_1MB_compute50us)
{
    ASSERT_TRUE((RunScenarioGFusedComputePrefetch<float, 262144>(0, 2, 5000)));
}

TEST(TPrefetchCompare, G_Fused_M2_1MB_compute200us)
{
    ASSERT_TRUE((RunScenarioGFusedComputePrefetch<float, 262144>(0, 2, 20000)));
}

TEST(TPrefetchCompare, G_Fused_M2_1MB_compute800us)
{
    ASSERT_TRUE((RunScenarioGFusedComputePrefetch<float, 262144>(0, 2, 80000)));
}

TEST(TPrefetchCompare, G_Fused_M4_1MB_compute50us)
{
    ASSERT_TRUE((RunScenarioGFusedComputePrefetch<float, 262144>(0, 4, 5000)));
}

TEST(TPrefetchCompare, G_Fused_M4_1MB_compute200us)
{
    ASSERT_TRUE((RunScenarioGFusedComputePrefetch<float, 262144>(0, 4, 20000)));
}

TEST(TPrefetchCompare, G_Fused_M4_1MB_compute800us)
{
    ASSERT_TRUE((RunScenarioGFusedComputePrefetch<float, 262144>(0, 4, 80000)));
}

TEST(TPrefetchCompare, G_Fused_M8_1MB_compute50us)
{
    ASSERT_TRUE((RunScenarioGFusedComputePrefetch<float, 262144>(0, 8, 5000)));
}

TEST(TPrefetchCompare, G_Fused_M8_1MB_compute200us)
{
    ASSERT_TRUE((RunScenarioGFusedComputePrefetch<float, 262144>(0, 8, 20000)));
}

TEST(TPrefetchCompare, G_Fused_M8_1MB_compute800us)
{
    ASSERT_TRUE((RunScenarioGFusedComputePrefetch<float, 262144>(0, 8, 80000)));
}

TEST(TPrefetchCompare, G_Fused_M16_1MB_compute50us)
{
    ASSERT_TRUE((RunScenarioGFusedComputePrefetch<float, 262144>(0, 16, 5000)));
}

TEST(TPrefetchCompare, G_Fused_M16_1MB_compute200us)
{
    ASSERT_TRUE((RunScenarioGFusedComputePrefetch<float, 262144>(0, 16, 20000)));
}

TEST(TPrefetchCompare, G_Fused_M16_1MB_compute800us)
{
    ASSERT_TRUE((RunScenarioGFusedComputePrefetch<float, 262144>(0, 16, 80000)));
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
