/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TEXTRACT_UB2L1_ND2NZ_CASES_H_
#define TEXTRACT_UB2L1_ND2NZ_CASES_H_

#include "../tmov_ub2l1/ub2l1_nd2nz_types.h"

#define TEXTRACT_UB2L1_ND2NZ_CASES(X)                                                                                  \
    X(21, nd2nz_extract_offset, half, 16, 48, 96, 32, 64, 17, 48, Nd2NzOperation::EXTRACT, 3, 16, true, 48, 96)        \
    X(30, nd2nz_extract_bfloat16, bfloat16_t, 16, 48, 96, 32, 64, 17, 48, Nd2NzOperation::EXTRACT, 3, 16, true, 24,    \
      80)                                                                                                              \
    X(31, nd2nz_extract_float, float, 32, 48, 48, 32, 32, 17, 24, Nd2NzOperation::EXTRACT, 3, 8, true, 24, 40)         \
    X(32, nd2nz_extract_int8, int8_t, 8, 48, 192, 32, 128, 17, 96, Nd2NzOperation::EXTRACT, 3, 32, true, 24, 160)      \
    X(33, nd2nz_extract_hifloat8, hifloat8_t, 8, 48, 192, 32, 128, 17, 96, Nd2NzOperation::EXTRACT, 3, 32, true, 24,   \
      160)                                                                                                             \
    X(34, nd2nz_extract_fp8_e4m3, float8_e4m3_t, 8, 48, 192, 32, 128, 17, 96, Nd2NzOperation::EXTRACT, 3, 32, true,    \
      24, 160)                                                                                                         \
    X(35, nd2nz_extract_fp8_e5m2, float8_e5m2_t, 8, 48, 192, 32, 128, 17, 96, Nd2NzOperation::EXTRACT, 3, 32, true,    \
      24, 160)                                                                                                         \
    X(36, nd2nz_extract_fp8_e8m0, float8_e8m0_t, 8, 48, 192, 32, 128, 17, 96, Nd2NzOperation::EXTRACT, 3, 32, true,    \
      24, 160)                                                                                                         \
    X(37, nd2nz_extract_fp4_e2m1, float4_e2m1x2_t, 4, 48, 384, 32, 256, 17, 192, Nd2NzOperation::EXTRACT, 3, 64, true, \
      24, 320)                                                                                                         \
    X(38, nd2nz_extract_fp4_e1m2, float4_e1m2x2_t, 4, 48, 384, 32, 256, 17, 192, Nd2NzOperation::EXTRACT, 3, 64, true, \
      24, 320)                                                                                                         \
    X(40, nd2nz_extract_static, half, 16, 48, 96, 32, 64, 17, 48, Nd2NzOperation::EXTRACT, 3, 16, false, 20, 64)       \
    X(41, nd2nz_extract_empty_rows, half, 16, 16, 64, 16, 32, 0, 16, Nd2NzOperation::EXTRACT, 3, 16, true, 8, 48)      \
    X(42, nd2nz_extract_empty_cols, half, 16, 16, 64, 16, 32, 7, 0, Nd2NzOperation::EXTRACT, 3, 16, true, 10, 48)      \
    X(43, nd2nz_extract_valid_edge, float, 32, 48, 64, 32, 32, 17, 24, Nd2NzOperation::EXTRACT, 3, 8, true, 20, 32)

#endif
