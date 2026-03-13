/**
* Copyright (c) 2026 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

/*!
 * \file sys_macros.h
 * \brief
 */
#ifndef IMPL_UTILS_SYS_MACROS_H
#define IMPL_UTILS_SYS_MACROS_H

#include <cstdint>
#include "impl/utils/sys_constants.h"

#if (defined(ASCENDC_CPU_DEBUG) && (ASCENDC_CPU_DEBUG == 1))
#include "stub_def.h"
using float4_e1m2x2_t = fp4x2_e1m2_t;
using float4_e2m1x2_t = fp4x2_e2m1_t;
using float8_e4m3_t = fp8_e4m3fn_t;
using float8_e5m2_t = fp8_e5m2_t;
using float8_e8m0_t = fp8_e8m0_t;
#endif

#if !defined(ASCENDC_CPU_DEBUG) || ASCENDC_CPU_DEBUG != 1
// For ascc preprocess: __global__ should not be replaced
#ifdef __ASCC_PRE__
#ifdef __global__
#undef __global__
#endif
#else

#ifndef __aicore__
#define __aicore__ [aicore]
#endif // __aicore__

#ifndef __host_aicore__
#define __host_aicore__ [host, aicore]
#endif // __host_aicore__

#ifndef __disable_kernel_type_autoinfer__
#define __disable_kernel_type_autoinfer__
#endif // __disable_kernel_type_autoinfer__

#endif // __ASCC_PRE__
#endif

#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 8
#endif

#if (defined(__DAV_CUBE__) && (__NPU_ARCH__ == 2201 || __NPU_ARCH__ == 3510))
#define SPLIT_CORE_CUBE
#endif

#if (defined(__DAV_VEC__) && (__NPU_ARCH__ == 2201 || __NPU_ARCH__ == 3510))
#define SPLIT_CORE_VEC
#endif

#if defined(ASCENDC_CPU_DEBUG)
extern int32_t g_coreType;
#define ASCEND_IS_AIV (g_coreType == AscendC::AIV)
#define ASCEND_IS_AIC (g_coreType == AscendC::AIC)
#define ASCEND_IS_NOT_AIV (g_coreType != AscendC::AIV)
#define ASCEND_IS_NOT_AIC (g_coreType != AscendC::AIC)
#else
#if defined(SPLIT_CORE_CUBE)
constexpr int32_t g_coreType = AscendC::AIC;
#elif defined(SPLIT_CORE_VEC)
constexpr int32_t g_coreType = AscendC::AIV;
#else
constexpr int32_t g_coreType = AscendC::MIX;
#endif
#define ASCEND_IS_AIV constexpr(g_coreType == AscendC::AIV)
#define ASCEND_IS_AIC constexpr(g_coreType == AscendC::AIC)
#define ASCEND_IS_NOT_AIV constexpr(g_coreType != AscendC::AIV)
#define ASCEND_IS_NOT_AIC constexpr(g_coreType != AscendC::AIC)
#endif

#endif