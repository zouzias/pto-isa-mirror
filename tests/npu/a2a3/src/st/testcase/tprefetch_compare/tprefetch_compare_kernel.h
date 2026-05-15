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

// ---- Scenario F: chunked prefetch + warm TLOAD (mirror of Scenario A) ---
// Goal: hold the total prefetched bytes constant (= Scenario A's `count *
// sizeof(T)`) and vary N — the number of chunks the prefetch is split into.
// Compare host vs device end-to-end wall under the SAME consumer (a TLOAD
// sweep over the full buffer). This isolates ONE variable: "1 big prefetch
// vs N small prefetches", everything else (data size, TLOAD body, L2
// trashing, single trailing sync) matches Scenario A exactly.
//
// Two paths, identical total work (sum of all chunks = full buffer):
//
//   * host path:
//         for i in [0, N): aclrtCmoAsync(src + i*chunk, chunk, stream)
//         ScenarioA_TloadOnlyKernel<<<1, nullptr, stream>>>(...)   // warm TLOAD
//         aclrtSynchronizeStream(stream)                            // ONE sync
//     N aclrtCmoAsync are async-enqueues onto the same stream; the trailing
//     TLOAD-only kernel is the "consumer" that reads everything; the FIFO
//     order on the stream guarantees all CMO complete before TLOAD starts.
//
//   * device path:
//         ScenarioF_DeviceChunkedPrefetchAndTloadKernel<<<1, ...>>>(
//             src, chunk, N, workspace, cycleOut)
//         aclrtSynchronizeStream(stream)                            // ONE sync
//     ONE kernel launch. Inside: loop N x { TPREFETCH_L2(addr_i) + Wait },
//     then TLOAD the whole buffer — same TLOAD body as Scenario A device.
//
// Synchronisation correctness — both paths confirm "all chunks prefetched
// and consumed by TLOAD" before the wall-clock timer stops:
//   - host: stream FIFO orders the N CMO tasks before the TLOAD kernel; the
//     trailing aclrtSynchronizeStream is a barrier against everything queued.
//   - device: in-kernel evt.Wait() after every TPREFETCH_L2 ensures the
//     event flag is consumed clean before reuse; the kernel cannot exit
//     until the TLOAD sweep finishes; the trailing sync surfaces "kernel
//     done" to the host.
//
// Same-as-Scenario-A invariants: total bytes equal `count * sizeof(T)`,
// L2 trashed once per measured iteration, single trailing sync, 100 iter
// with 1 warmup, p5/p50/p95 reported. **N=1 reduces exactly to Scenario A's
// host_sdma / device_l2 rows for the same `count`** — that's the natural
// sanity check.
//
// totalBytes (= count * sizeof(T)) MUST be divisible by numChunks; the
// runner checks and bails out early otherwise.
//
// CSV: one row per (totalBytes, N, config) — host_async + device_kernel.
// host_kernel_p50 / device_kernel_p50 are the in-kernel cycle counters
// (host: TLOAD only; device: N x prefetch+Wait + TLOAD), reported as
// supplementary diagnostics; the apples-to-apples number is the wall.
template <typename T, size_t count>
bool RunScenarioFChunkedPrefetchAndTload(int deviceId, uint32_t numChunks);

// ---- Scenario G: fused multi-stage prefetch + compute -------------------
// Mirrors a "compute-comm fusion" workload: M consecutive stages, each
// stage = (prefetch chunk_i) + (compute spinCycles on chunk_i) + (warm
// TLOAD chunk_i). Total bytes prefetched = M * chunkBytes; total compute
// = M * spinCycles. This is the deployment pattern where device should
// structurally win — host has to interleave M kernel launches between M
// prefetches on the same stream, while device can express the whole thing
// as a single pipelined kernel.
//
// host_serial:
//     for i in [0, M):
//         pto::PTO_PREFETCH(src + i*chunk, chunk, stream)
//         ScenarioC_ComputeThenTloadKernel<<<...stream>>>(stage_i)
//     aclrtSynchronizeStream(stream)
//   - M kernel launches; each pays the ~17us launch shell (E1).
//   - Single stream FIFO orders prefetch_i before kernel_i; but
//     prefetch_{i+1} cannot start until kernel_i finishes — so there is
//     NO prefetch-compute overlap on a single stream.
//   - Total wall ~ M * (T_prefetch + T_launch + T_compute) + T_sync.
//
// device_pipelined:
//     ScenarioG_DeviceFusedKernel<<<1, nullptr, stream>>>(...)
//         # in-kernel:
//         # prefetch block_0; wait                    (no overlap on first)
//         # for i in [0, M-1):
//         #     issue prefetch block_{i+1}            (async fire)
//         #     spin compute_i                        (AICORE busy)
//         #     TLOAD block_i                         (warm)
//         #     wait block_{i+1}                      (overlap: SDMA was running)
//         # last stage: spin + TLOAD on block_{M-1}
//     aclrtSynchronizeStream(stream)
//   - ONE launch shell (saves (M-1) launches vs host).
//   - prefetch_{i+1} runs concurrently with compute_i. If T_compute >=
//     T_prefetch, the prefetch is fully hidden.
//   - Total wall ~ T_launch + T_prefetch_first + (M-1)*max(T_compute,
//     T_prefetch) + T_compute_last + T_sync.
//
// Same-as-Scenario-A invariants: total bytes = M*chunkBytes, L2 trashed
// once per measured iteration, single trailing sync per path. The compute
// body inside each host stage uses ScenarioC_ComputeThenTloadKernel so the
// per-stage compute primitive is bit-identical to Scenario C.
//
// Reports per-(M, chunkBytes, spinCycles): host wall, device wall, plus
// in-kernel cycles (host: last-stage representative; device: full
// pipelined body sum). One CSV row per (M, chunk, spin, config) appended
// to tprefetch_compare_scenarioG.csv.
template <typename T, size_t chunkElems>
bool RunScenarioGFusedComputePrefetch(int deviceId, uint32_t numStages, uint64_t spinCycles);

// ---- Scenario H: data-dependent prefetch (host forced to sync per stage) ---
// Variant of Scenario G that makes the prefetch address of stage i+1 depend
// on a value computed by stage i. The host CANNOT pre-issue all stages on
// the stream up-front because the address for prefetch_{i+1} is not known
// until kernel_i has run and written its result to device memory; therefore
// host must aclrtSynchronizeStream + aclrtMemcpy(D2H) between every pair
// of stages — exactly the stream pipeline that made Scenario G a tie is
// broken here. Device, on the other hand, keeps the offset in an AICORE
// register and pipelines as in Scenario G.
//
// Algorithm (identical on both paths): offset_{i+1} = (offset_i + 1) % M.
// The trivial computation is irrelevant; what matters is WHERE the value
// lives — the contract is "the address comes from the previous stage's
// device-side output". host has to materialize it on the host side; device
// passes it through a register.
//
// host_dep_sync :
//     offsets[0] = 0
//     for i in [0, M):
//         pto::PTO_PREFETCH(src + offsets[i]*chunk, chunk, stream)
//         ScenarioH_HostStageKernel<<<...stream>>>(stage_i, ..., offsets[i],
//                                                   &deviceOffsetBuf[i])
//         if i < M-1:
//             aclrtSynchronizeStream(stream)                  # FORCED SYNC
//             aclrtMemcpy(&offsets[i+1], &deviceOffsetBuf[i],
//                         sizeof(uint32_t), ACL_MEMCPY_DEVICE_TO_HOST)
//     aclrtSynchronizeStream(stream)
//   - M launches AND (M-1) forced sync+D2H roundtrips.
//   - Each forced sync drains the stream — kills the SDMA-launch overlap
//     that made Scenario G's host_serial competitive.
//
// device_in_reg :
//     ScenarioH_DeviceFusedKernel<<<1, ...>>>(src, chunk, M, M, spin, ws, ...)
//         # in-kernel: offset starts at 0, pipelined loop
//         #            (prefetch first; for i in M-1: issue next, spin,
//         #             TLOAD current, wait next; last: spin + TLOAD)
//     aclrtSynchronizeStream(stream)
//   - ONE launch + ONE sync. Offset propagation is in-register, instant.
//
// Predicted device advantage on top of Scenario G:
//   ~ (M-1) * (T_sync_round + T_d2h_memcpy_overhead) ~ (M-1) * 15-20 us.
template <typename T, size_t chunkElems>
bool RunScenarioHDependentPrefetch(int deviceId, uint32_t numStages, uint64_t spinCycles);

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
