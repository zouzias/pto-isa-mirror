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

// Single-card tests (no HCCL)
template <typename T, size_t count>
bool RunBaseline(int deviceId);

template <typename T, size_t count>
bool RunPrefetchL2Correctness(int deviceId);

template <typename T, size_t count>
bool RunPrefetchL2RawPtr(int deviceId);

// Multi-card test: TPUT_ASYNC with optional TPREFETCH_L2 (HCCL)
template <typename T, size_t count>
bool RunPrefetchL2TputAsync(int n_ranks, int n_devices, int first_rank_id, int first_device_id, bool prefetch);

// Performance comparison: TPUT_ASYNC with vs without TPREFETCH_L2
template <typename T, size_t count>
bool RunPrefetchL2Perf(int n_ranks, int n_devices, int first_rank_id, int first_device_id);

// TLOAD latency comparison: L2-cold vs L2-prefetched (single-card)
template <typename T, size_t count>
bool RunTloadPerf(int deviceId);

// Remote TLOAD perf: Rank 0 → TPUT_ASYNC → Rank 1 → (optional prefetch) → TLOAD
template <typename T, size_t count>
bool RunTloadRemotePerf(int n_ranks, int n_devices, int first_rank_id, int first_device_id);

// Prefetch before TPUT: prefetch local src → TPUT to remote → measure TPUT perf
template <typename T, size_t count>
bool RunTputSyncPerf(int n_ranks, int n_devices, int first_rank_id, int first_device_id);

// Remote prefetch before TGET: remote prefetches → local TGETs → measure TGET perf
template <typename T, size_t count>
bool RunTgetPerf(int n_ranks, int n_devices, int first_rank_id, int first_device_id);
