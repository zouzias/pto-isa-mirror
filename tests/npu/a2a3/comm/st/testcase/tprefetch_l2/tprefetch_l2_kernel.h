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
// Cross-rank runners for pto::TPREFETCH_L2.
//
// Single-card coverage (Baseline / Correctness / RawPtr / single-card TLOAD
// perf) lives under tests/npu/a2a3/src/st/testcase/tprefetch_l2/ since none
// of those need the HCCL test scaffold pulled in by `comm/st`.
// ============================================================================

// Multi-card test: TPUT_ASYNC with optional TPREFETCH_L2 (HCCL)
template <typename T, size_t count>
bool RunPrefetchL2TputAsync(int n_ranks, int n_devices, int first_rank_id, int first_device_id, bool prefetch);

// Performance comparison: TPUT_ASYNC with vs without TPREFETCH_L2
template <typename T, size_t count>
bool RunPrefetchL2Perf(int n_ranks, int n_devices, int first_rank_id, int first_device_id);

// Remote TLOAD perf: Rank 0 → TPUT_ASYNC → Rank 1 → (optional prefetch) → TLOAD
template <typename T, size_t count>
bool RunTloadRemotePerf(int n_ranks, int n_devices, int first_rank_id, int first_device_id);

// Prefetch before TPUT: prefetch local src → TPUT to remote → measure TPUT perf
template <typename T, size_t count>
bool RunTputSyncPerf(int n_ranks, int n_devices, int first_rank_id, int first_device_id);

// Remote prefetch before TGET: remote prefetches → local TGETs → measure TGET perf
template <typename T, size_t count>
bool RunTgetPerf(int n_ranks, int n_devices, int first_rank_id, int first_device_id);
