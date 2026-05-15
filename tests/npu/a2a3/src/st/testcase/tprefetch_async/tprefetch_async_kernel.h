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
// Single-card runners for device-side async L2 prefetch.
// ============================================================================

template <typename T, size_t count>
bool RunPrefetchAsyncCorrectness(int deviceId);

template <typename T, size_t count>
bool RunPrefetchAsyncRawPtr(int deviceId);

// Effectiveness check: TLOAD after async prefetch must be faster than cold TLOAD.
template <typename T, size_t count>
bool RunTloadPerf(int deviceId);

// ---- Host/device end-to-end wall-clock latency ---------------------------
// For a given data size, runs 3 configurations:
//   * baseline           - trash L2 + kernel TLOAD (no prefetch)
//   * host prefetch      - trash L2 + host pto::PTO_PREFETCH + kernel TLOAD
//   * device async       - trash L2 + kernel async prefetch + Wait + TLOAD
template <typename T, size_t count>
bool RunHostDevicePrefetch(int deviceId);

// ---- Static-address multi-stage prefetch + compute -----------------------
// Host knows every stage address and can enqueue the whole stream up front.
// Device expresses the same work as one fused in-kernel pipeline.
template <typename T, size_t chunkElems>
bool RunStaticAddressPrefetch(int deviceId, uint32_t numStages, uint64_t spinCycles);

// ---- Data-dependent multi-stage prefetch ---------------------------------
// The next prefetch address depends on device-side output from the previous
// stage. Host must sync+D2H between stages; device keeps the value in-register.
template <typename T, size_t chunkElems>
bool RunDataDependentPrefetch(int deviceId, uint32_t numStages, uint64_t spinCycles);
