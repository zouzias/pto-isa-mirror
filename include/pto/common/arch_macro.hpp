/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
// Implementation of interface adaptation layer for device-side and cloud-side compatibility
#ifndef ARCH_MACRO_HPP
#define ARCH_MACRO_HPP

#if PTO_DEVICE_SIDE_ARCH
#define float8_e4m3_t char
#define float8_e5m2_t char  
#define hifloat8_t char
// #define bfloat16_t half
#define float4_e2m1x2_t char
#define float4_e1m2x2_t char
#define float8_e8m0_t char
#define __tf__
#define __in__
#define __out__
#define __cce_get_tile_ptr
#endif
#endif