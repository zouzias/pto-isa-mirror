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

// ============================================================================
// Single-card runner for device-side async L2 prefetch.
//
// Only the functional-correctness path is exposed here. The host-vs-device,
// static-address fused multi-stage and data-dependent fused multi-stage perf
// runners that used to live in this file were removed together with their
// kernels because each extra prefetch-using AICORE kernel instantiation
// embeds the SDMA session-init cast chain into the function body and trips
// Bisheng's optimizer (Segmentation fault in CastInst::CreateBitOrPointerCast)
// once a single TU accumulates too many of those expansions. They will return
// in a dedicated perf TU once the optimizer budget is increased.
// ============================================================================

template <typename T, size_t count>
bool RunPrefetchAsyncCorrectness(int deviceId);
