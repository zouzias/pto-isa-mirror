/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#pragma once

#include <cstddef>
#include <cstdint>

// ============================================================================
// tprefetch_compare (single-card) — performance comparison between
// host-initiated pto::PTO_PREFETCH (SDMA path, driven by aclrtCmoAsync) and
// device-initiated pto::TPREFETCH_L2 (SDMA CMO SQE built by the AI Core).
//
// Methodology (mirrors shmem/examples/cmo for consistency):
//   * 100 measured iterations per configuration (override via env
//     TPREFETCH_COMPARE_ITER), with 1 warmup.
//   * L2 is evicted with a 512 MB trash buffer before every measured iteration
//     to ensure a true cold start.
//   * Statistics: p5 / p50 / p95 are reported. Median (p50) is the headline
//     number; p5/p95 reveal tail behaviour.
//   * Per-(size, config) rows are appended to CSV files in CWD (override
//     location with env TPREFETCH_COMPARE_CSV_DIR):
//       tprefetch_compare_scenarioA.csv  (Scenario A end-to-end)
//       tprefetch_compare_scenarioB.csv  (Scenario B issue overhead)
//       tprefetch_compare_scenarioC.csv  (Scenario C overlap)
//   * Each runner returns false on ACL/runtime failure; a "false" returned
//     from a prefetch-comparison test does NOT mean PREFETCH lost the race,
//     only that the hardware refused to cooperate.
//
// Cross-rank coverage (Scenario D, receiver-side prefetch over TPUT_ASYNC)
// lives under tests/npu/a2a3/comm/st/testcase/tprefetch_compare/.
// ============================================================================

// ---- Scenario A: end-to-end wall-clock latency ---------------------------
// For a given data size, runs 3 configurations:
//   * baseline           — trash L2 + kernel TLOAD (no prefetch)
//   * host PTO_PREFETCH  — trash L2 + host pto::PTO_PREFETCH + kernel TLOAD
//   * device TPREFETCH_L2— trash L2 + kernel (TPREFETCH_L2 -> Wait -> TLOAD)
// All three measurements time the entire host-visible window (from the first
// op issued after L2 trash → aclrtSynchronizeStream returns). This is the
// ONLY apples-to-apples instruction-vs-instruction wall-clock comparison.
// In-kernel syscnt cycles are also reported as a supplementary diagnostic
// for the "embedded in an existing kernel" deployment model — but those
// numbers are NOT directly comparable across host vs device because the two
// kernels' bodies differ.
template <typename T, size_t count>
bool RunScenarioAEndToEnd(int deviceId);

// ---- Scenario B: prefetch-issue overhead ---------------------------------
// Micro-benchmark: prefetch a small region and measure the issue+completion
// roundtrip. payloadBytes is varied at the call site so we can verify the
// fixed-software-overhead claim and observe where transfer time starts to
// dominate the wall. Three metrics per call:
//   * host wall   — aclrtCmoAsync + aclrtSynchronizeStream wall-clock
//   * device wall — kernel launch + in-kernel TPREFETCH_L2 + Wait + sync wall
//                   (the apples-to-apples counterpart to host wall above)
//   * device in-kernel — only the TPREFETCH_L2 + Wait segment inside the
//                       kernel; models the marginal cost of embedding into
//                       a kernel that's already being launched anyway.
// The buffer size (template param `count`) just has to be >= payloadBytes;
// it does not affect the measurement.
template <typename T, size_t count>
bool RunScenarioBIssueOverhead(int deviceId, size_t payloadBytes = 4096);

// ---- Scenario C: overlap with AI-Core compute ----------------------------
// Kernel structure:    [optional device prefetch] -> compute-A -> [optional wait] -> TLOAD
// host side may optionally issue a PTO_PREFETCH before kernel launch.
// Three configurations:
//   C0 no-prefetch     : trash L2 -> launch kernel(compute-A -> cold TLOAD)
//   C1 host prefetch   : trash L2 -> host PTO_PREFETCH -> kernel(compute-A -> warm TLOAD)
//   C2 device prefetch : trash L2 -> kernel(TPREFETCH_L2 -> compute-A -> Wait -> warm TLOAD)
// Two metrics reported per config: kernel-self cycles (AI-Core view, via
// syscnt), and host wall-clock (end-to-end including prefetch issue).
// "cold-reduc" = 1 - kern(C*).p50 / kern(C0).p50 — fraction of cold-TLOAD
// latency hidden by the prefetch. Higher = better overlap.
template <typename T, size_t count>
bool RunScenarioCOverlap(int deviceId);

// ---- Scenario E1: kernel launch + dispatch + sync overhead --------------
// Pure-overhead micro-benchmark used to localise where Scenario A's "device
// path is 4 us slower at small payloads" gap actually lives.
//
// Launches an empty AI-Core kernel (no SDMA, no TLOAD, no UB use) and times
// the host-visible window:
//
//     t0 = HrClock::now()
//     NoopKernel<<<1, nullptr, stream>>>()
//     aclrtSynchronizeStream(stream)
//     t1 = HrClock::now()
//
// The reported `wall_p50` is therefore the cost of "kernel launch + AICORE
// dispatch + sync return" with NO useful work done in the kernel.
//
// How to use the result:
//     fair_device_in_kernel_overhead
//         = ScenarioA<size>.device_wall_p50 - ScenarioE1.wall_p50
//
// In other words, subtract this number from the device-path wall in any
// other scenario to isolate the in-kernel cost (transient session build,
// SQE submission, busy poll, SDMA itself). This is the fair "instruction
// vs instruction" view that the user asked for in the experiment plan.
//
// Also reports the in-kernel syscnt delta around an empty `pipe_barrier`,
// which doubles as a sanity check on the syscnt-to-microsecond conversion.
//
// L2 is intentionally NOT trashed between iterations — a no-op kernel does
// not touch L2, so trashing would only add measurement noise.
//
// 100 iterations, p5/p50/p95 reported, single CSV row appended to:
//     tprefetch_compare_scenarioE1.csv
bool RunScenarioE1NoopKernel(int deviceId);
