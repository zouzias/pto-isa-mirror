/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef GEMM_CONFIG_H
#define GEMM_CONFIG_H

#include <cstdint>

// ============================================================================
// GEMM Parameters - Shared between host and kernel
// ============================================================================

// Global matrix dimensions
constexpr uint32_t GEMM_M = 6144;
constexpr uint32_t GEMM_K = 6144;
constexpr uint32_t GEMM_N = 6144;

// Single core workload dimensions
constexpr uint32_t SINGLE_CORE_M = 1536;
constexpr uint32_t SINGLE_CORE_K = 6144;
constexpr uint32_t SINGLE_CORE_N = 1024;

// Block dimension (number of AI cores)
constexpr uint32_t BLOCK_DIM = 24;

// Base tile dimensions
constexpr uint32_t BASE_M = 128;
constexpr uint32_t BASE_K = 64;
constexpr uint32_t BASE_N = 256;

// Step factors for L1 cache staging
constexpr uint32_t STEP_M = 1;
constexpr uint32_t STEP_KA = 4;
constexpr uint32_t STEP_KB = 4;
constexpr uint32_t STEP_N = 1;

#endif // GEMM_CONFIG_H
