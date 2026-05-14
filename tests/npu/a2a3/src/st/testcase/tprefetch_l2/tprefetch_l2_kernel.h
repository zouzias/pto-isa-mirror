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
// Single-card runners for pto::TPREFETCH_L2 (workspace API).
//
// Cross-rank coverage (TPUT_ASYNC + prefetch, remote TLOAD perf, TGET perf,
// etc.) lives under tests/npu/a2a3/comm/st/testcase/tprefetch_l2/ since
// those drivers depend on the HCCL test scaffold.
// ============================================================================

template <typename T, size_t count>
bool RunBaseline(int deviceId);

template <typename T, size_t count>
bool RunPrefetchL2Correctness(int deviceId);

template <typename T, size_t count>
bool RunPrefetchL2RawPtr(int deviceId);

// L2-cold vs L2-prefetched TLOAD latency (single-card)
template <typename T, size_t count>
bool RunTloadPerf(int deviceId);
