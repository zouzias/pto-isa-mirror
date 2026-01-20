/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// NOTE: ascendc.cmake (dynamic mode) may compile sources for both AIC(CUBE) and AIV(VEC).
// - CUBE implementation: compiled under __DAV_C220_CUBE__
// - Precompile stage: uses __CHECK_FEATURE_AT_PRECOMPILE (often with VEC arch)
// - VEC placeholder: needed so the preprocess AIV step doesn't fail on an empty TU
#if defined(__CHECK_FEATURE_AT_PRECOMPILE) || (__CCE_AICORE__ == 220 && defined(__DAV_C220_CUBE__))

#define MEMORY_BASE

#include "kernel_operator.h"

#include "../../../gemm_basic_impl.hpp"

using namespace pto_gemm_basic;

extern "C" __global__ AICORE void gemm_basic_custom(GM_ADDR a, GM_ADDR b_dn, GM_ADDR out)
{
    constexpr uint32_t M = 512;
    constexpr uint32_t K = 2048;
    constexpr uint32_t N = 1536;
    constexpr uint32_t singleCoreM = 128;
    constexpr uint32_t singleCoreK = 2048;
    constexpr uint32_t singleCoreN = 256;
    constexpr uint32_t baseM = 128;
    constexpr uint32_t baseK = 64;
    constexpr uint32_t baseN = 256;
    runGEMMBASIC<float, half, half, M, K, N, singleCoreM, singleCoreK, singleCoreN, baseM, baseK, baseN>(
        reinterpret_cast<__gm__ float *>(out), reinterpret_cast<__gm__ half *>(a),
        reinterpret_cast<__gm__ half *>(b_dn));
}

#elif __CCE_AICORE__ == 220 && defined(__DAV_C220_VEC__)

// Placeholder for VEC compilation (the real kernel is CUBE-only).
#define MEMORY_BASE
#include "kernel_operator.h"
#include <pto/common/type.hpp>
extern "C" __global__ AICORE void gemm_basic_custom(GM_ADDR a, GM_ADDR b_dn, GM_ADDR out) {}

#endif
