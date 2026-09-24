/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TINSERT_UB2L1_ND2NZ_CASES_H_
#define TINSERT_UB2L1_ND2NZ_CASES_H_

#include "../tmov_ub2l1/ub2l1_nd2nz_types.h"

#define TINSERT_UB2L1_ND2NZ_CASES(X)                                                                                   \
    X(22, nd2nz_insert_offset, half, 16, 16, 64, 48, 96, 7, 48, Nd2NzOperation::INSERT, 5, 32, true, 7, 48)            \
    X(23, nd2nz_insert_wide, float, 32, 16, 128, 32, 160, 3, 128, Nd2NzOperation::INSERT, 17, 8, true, 3, 128)         \
    X(26, nd2nz_dual_insert, half, 16, 16, 64, 32, 64, 16, 64, Nd2NzOperation::DUAL_INSERT, 0, 0, true, 16, 64)        \
    X(30, nd2nz_insert_bfloat16, bfloat16_t, 16, 16, 64, 48, 96, 7, 48, Nd2NzOperation::INSERT, 5, 32, true, 7, 48)    \
    X(31, nd2nz_insert_int8, int8_t, 8, 16, 128, 48, 192, 7, 96, Nd2NzOperation::INSERT, 5, 64, true, 7, 96)           \
    X(32, nd2nz_insert_hifloat8, hifloat8_t, 8, 16, 128, 48, 192, 7, 96, Nd2NzOperation::INSERT, 5, 64, true, 7, 96)   \
    X(33, nd2nz_insert_fp8_e4m3, float8_e4m3_t, 8, 16, 128, 48, 192, 7, 96, Nd2NzOperation::INSERT, 5, 64, true, 7,    \
      96)                                                                                                              \
    X(34, nd2nz_insert_fp8_e5m2, float8_e5m2_t, 8, 16, 128, 48, 192, 7, 96, Nd2NzOperation::INSERT, 5, 64, true, 7,    \
      96)                                                                                                              \
    X(35, nd2nz_insert_fp8_e8m0, float8_e8m0_t, 8, 16, 128, 48, 192, 7, 96, Nd2NzOperation::INSERT, 5, 64, true, 7,    \
      96)                                                                                                              \
    X(36, nd2nz_insert_fp4_e2m1, float4_e2m1x2_t, 4, 16, 256, 48, 384, 7, 192, Nd2NzOperation::INSERT, 5, 128, true,   \
      7, 192)                                                                                                          \
    X(37, nd2nz_insert_fp4_e1m2, float4_e1m2x2_t, 4, 16, 256, 48, 384, 7, 192, Nd2NzOperation::INSERT, 5, 128, true,   \
      7, 192)                                                                                                          \
    X(38, nd2nz_insert_int32, int32_t, 32, 16, 32, 48, 48, 7, 24, Nd2NzOperation::INSERT, 5, 16, true, 7, 24)          \
    X(40, nd2nz_insert_static, half, 16, 16, 64, 48, 96, 7, 48, Nd2NzOperation::INSERT, 5, 32, false, 7, 48)           \
    X(41, nd2nz_insert_empty_rows, half, 16, 16, 64, 32, 64, 0, 32, Nd2NzOperation::INSERT, 5, 16, true, 0, 32)        \
    X(42, nd2nz_insert_empty_cols, float, 32, 16, 32, 32, 32, 7, 0, Nd2NzOperation::INSERT, 3, 8, true, 7, 0)          \
    X(43, nd2nz_dual_insert_fp4, float4_e2m1x2_t, 4, 16, 128, 32, 128, 16, 128, Nd2NzOperation::DUAL_INSERT, 0, 0,     \
      true, 16, 128)                                                                                                   \
    X(44, nd2nz_insert_fp4_65536_static, float4_e2m1x2_t, 4, 1, 65536, 16, 65536, 1, 65536, Nd2NzOperation::INSERT, 5, \
      0, false, 1, 65536)                                                                                              \
    X(45, nd2nz_insert_fp4_65536_dynamic, float4_e1m2x2_t, 4, 1, 65600, 16, 65536, 1, 65536, Nd2NzOperation::INSERT,   \
      5, 0, true, 1, 65536)

#endif
