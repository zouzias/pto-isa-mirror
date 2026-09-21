/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TMOV_UB2L1_ND2NZ_CASES_H_
#define TMOV_UB2L1_ND2NZ_CASES_H_

#include "ub2l1_nd2nz_types.h"

#ifdef PTO_SKIP_UB2L1_ND2NZ_ST
#define TMOV_UB2L1_ND2NZ_CASES(X)
#else
#define TMOV_UB2L1_ND2NZ_CASES(X)                                                                                   \
    X(10, nd2nz_half_static, half, 16, 16, 32, 16, 32, 16, 32, Nd2NzOperation::MOV, 0, 0, false, 16, 32)            \
    X(11, nd2nz_half_wide, half, 16, 16, 512, 16, 512, 3, 512, Nd2NzOperation::MOV, 0, 0, true, 3, 512)             \
    X(12, nd2nz_float_strides, float, 32, 32, 64, 32, 32, 19, 24, Nd2NzOperation::MOV, 0, 0, true, 19, 24)          \
    X(13, nd2nz_bfloat16_strides, bfloat16_t, 16, 32, 96, 32, 64, 17, 48, Nd2NzOperation::MOV, 0, 0, true, 17, 48)  \
    X(14, nd2nz_int8_strides, int8_t, 8, 32, 160, 32, 128, 31, 96, Nd2NzOperation::MOV, 0, 0, true, 31, 96)         \
    X(15, nd2nz_fp8_e4m3, float8_e4m3_t, 8, 16, 64, 16, 64, 16, 64, Nd2NzOperation::MOV, 0, 0, false, 16, 64)       \
    X(16, nd2nz_fp8_e5m2, float8_e5m2_t, 8, 16, 64, 16, 64, 16, 64, Nd2NzOperation::MOV, 0, 0, false, 16, 64)       \
    X(17, nd2nz_hifloat8, hifloat8_t, 8, 16, 64, 16, 64, 16, 64, Nd2NzOperation::MOV, 0, 0, false, 16, 64)          \
    X(18, nd2nz_fp8_e8m0, float8_e8m0_t, 8, 16, 64, 16, 64, 16, 64, Nd2NzOperation::MOV, 0, 0, false, 16, 64)       \
    X(19, nd2nz_fp4_e2m1, float4_e2m1x2_t, 4, 32, 256, 32, 192, 17, 128, Nd2NzOperation::MOV, 0, 0, true, 17, 128)  \
    X(20, nd2nz_fp4_e1m2, float4_e1m2x2_t, 4, 16, 128, 16, 128, 16, 128, Nd2NzOperation::MOV, 0, 0, false, 16, 128) \
    X(24, nd2nz_burst_split, half, 16, 4112, 16, 4112, 16, 4101, 16, Nd2NzOperation::MOV, 0, 0, true, 4101, 16)     \
    X(25, nd2nz_empty, half, 16, 16, 32, 16, 32, 0, 32, Nd2NzOperation::MOV, 0, 0, true, 0, 32)                     \
    X(27, nd2nz_empty_cols, float, 32, 16, 32, 16, 32, 7, 0, Nd2NzOperation::MOV, 0, 0, true, 7, 0)
#endif

#endif
