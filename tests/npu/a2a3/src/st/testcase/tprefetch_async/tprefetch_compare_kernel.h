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
// tprefetch_async comparison runners — focused comparison between
// host-initiated pto::PTO_PREFETCH and device-initiated pto::TPREFETCH_L2.
//
// Kept scenarios:
//   A  End-to-end latency for "prefetch then TLOAD" at small/large sizes.
//   G  Static-address multi-stage workload; host can enqueue all stages up front.
//   H  Data-dependent multi-stage workload; host must sync+D2H between stages.
//
// Each runner uses 100 measured iterations (override via TPREFETCH_COMPARE_ITER),
// evicts L2 before each measured iteration, reports p5/p50/p95, and appends CSV
// rows under TPREFETCH_COMPARE_CSV_DIR (or the current working directory).
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

// ---- Scenario G: static-address multi-stage prefetch + compute ----------
// Host knows every stage address and can enqueue the whole stream up front.
// Device expresses the same work as one fused in-kernel pipeline.
//
// host_serial:
//     for i in [0, M):
//         pto::PTO_PREFETCH(src + i*chunk, chunk, stream)
//         ScenarioC_ComputeThenTloadKernel<<<...stream>>>(stage_i)
//     aclrtSynchronizeStream(stream)
//
// device_pipelined:
//     ScenarioG_DeviceFusedKernel<<<1, nullptr, stream>>>(...)
//     aclrtSynchronizeStream(stream)
template <typename T, size_t chunkElems>
bool RunScenarioGFusedComputePrefetch(int deviceId, uint32_t numStages, uint64_t spinCycles);

// ---- Scenario H: data-dependent prefetch (host forced to sync per stage) ---
// Variant of Scenario G that makes the prefetch address of stage i+1 depend
// on a value computed by stage i. Host cannot pre-issue all stages on
// the stream up-front because the address for prefetch_{i+1} is not known
// until kernel_i has run and written its result to device memory. Host must
// sync + D2H that value before issuing the next prefetch; device keeps the
// value in an AICORE register inside one fused kernel.
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
//   - Each forced sync drains the stream and prevents the host from enqueueing
//     later stages until the previous device-side output is copied back.
//
// device_in_reg :
//     ScenarioH_DeviceFusedKernel<<<1, ...>>>(src, chunk, M, M, spin, ws, ...)
//         # in-kernel: offset starts at 0, pipelined loop
//         #            (prefetch first; for i in M-1: issue next, spin,
//         #             TLOAD current, wait next; last: spin + TLOAD)
//     aclrtSynchronizeStream(stream)
//   - ONE launch + ONE sync. Offset propagation is in-register, instant.
template <typename T, size_t chunkElems>
bool RunScenarioHDependentPrefetch(int deviceId, uint32_t numStages, uint64_t spinCycles);
