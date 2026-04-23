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
// tprefetch_compare — performance comparison between host-initiated
// pto::PTO_PREFETCH (SDMA path, driven by aclrtCmoAsync) and device-initiated
// pto::comm::TPREFETCH_L2 (SDMA CMO SQE built by the AI Core).
//
// Every runner below is single-card (no HCCL): the focus is on the prefetch
// primitive itself, not on inter-rank communication.
//
// Each runner returns false on ACL/runtime failure; a "false" returned from a
// prefetch-comparison test does NOT mean PREFETCH lost the race, only that the
// hardware refused to cooperate. The perf verdict is printed to stdout.
// ============================================================================

// ---- Scenario A: end-to-end wall-clock latency ---------------------------
// For a given data size, runs 3 configurations:
//   * baseline           — trash L2 + kernel TLOAD (no prefetch)
//   * host PTO_PREFETCH  — trash L2 + host pto::PTO_PREFETCH + kernel TLOAD
//   * device TPREFETCH_L2— trash L2 + kernel (TPREFETCH_L2 -> Wait -> TLOAD)
// Wall-clock is measured host-side around "start of first op after trash" to
// "aclrtSynchronizeStream returns".
template <typename T, size_t count>
bool RunScenarioAEndToEnd(int deviceId);

// ---- Scenario B: prefetch-issue overhead ---------------------------------
// Micro-benchmark: prefetch a tiny region (so transfer cost is negligible)
// and measure the *issue+wait* time.
//   * host side : host wall-clock over PTO_PREFETCH + aclrtSynchronizeStream.
//   * device side: in-kernel syscnt cycles around TPREFETCH_L2 + evt.Wait.
template <typename T, size_t count>
bool RunScenarioBIssueOverhead(int deviceId);

// ---- Scenario C: overlap with AI-Core compute ----------------------------
// Kernel structure:    [optional device prefetch] -> compute-A -> [optional wait] -> TLOAD
// host side may optionally issue a PTO_PREFETCH before kernel launch.
// Three configurations:
//   C0 no-prefetch     : trash L2 -> launch kernel(compute-A -> cold TLOAD)
//   C1 host prefetch   : trash L2 -> host PTO_PREFETCH -> kernel(compute-A -> warm TLOAD)
//   C2 device prefetch : trash L2 -> kernel(TPREFETCH_L2 -> compute-A -> Wait -> warm TLOAD)
// Two metrics reported per config: kernel-self cycles (AI-Core view, via
// syscnt), and host wall-clock (end-to-end including prefetch issue).
template <typename T, size_t count>
bool RunScenarioCOverlap(int deviceId);

// ---- Scenario D: cross-rank receiver-side prefetch -----------------------
// Producer-consumer pipeline over a TPUT_ASYNC link between two ranks:
//   Rank 0 (sender)   → pto::comm::TPUT_ASYNC(remote_recvBuf, sendBuf)
//   Rank 1 (receiver) → 3 prefetch modes on its own recvBuf, then TLOAD:
//     D0 no-prefetch
//     D1 host   pto::PTO_PREFETCH (host-issued aclrtCmoAsync + stream sync)
//     D2 device pto::comm::TPREFETCH_L2 (in-kernel SQE + Wait)
// All three issue exactly one SDMA CMO prefetch (or none) against the
// freshly received recvBuf, then measure the TLOAD sweep. The receiver
// returns kernel cycles (TLOAD-only in D0/D1, prefetch+wait+TLOAD in D2)
// and host wall-clock around the "prefetch + TLOAD kernel" phase.
// Requires mpirun -n 2; sender rank = 0, first device id = 0.
template <typename T, size_t count>
bool RunScenarioDCrossRank(int nRanks, int firstDeviceId);
