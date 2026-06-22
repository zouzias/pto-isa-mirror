/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef GQA_PERFORMANCE_KERNEL_H
#define GQA_PERFORMANCE_KERNEL_H

#include <acl/acl.h>
#include <cstddef>
#include <cstdint>

constexpr int kGqaCvFifoSize = 8;
constexpr int kGqaCvFifoConsSyncPeriod = kGqaCvFifoSize / 2;
constexpr int kGqaCubeS1 = 128;
constexpr int kGqaTileS1 = 256;
constexpr int kGqaQkPreload = 4;
constexpr std::size_t kGqaProfileBytesPerBlock = 1024 * 3;
constexpr std::size_t kGqaCvCommSlotBytes = 512U;
constexpr int VEC_CORES = 2;

template <int S0, int HEAD_SIZE, int S1, int CUBE_S0, int CUBE_S1 = kGqaCubeS1, int TILE_S1 = kGqaTileS1,
          int QK_PRELOAD = kGqaQkPreload, int CV_FIFO_SIZE = kGqaCvFifoSize, bool INTERMEDIATE_CHECK = false,
          bool CAUSAL_MASK = false, int CV_FIFO_CONS_SYNC_PERIOD = kGqaCvFifoConsSyncPeriod>
void LaunchTGQA(uint16_t *ffts, aclFloat16 *q, aclFloat16 *k, aclFloat16 *v, aclFloat16 *p_tile_fifo,
                float *exp_max_ififo, float *global_sum_out, float *exp_max_out, float *o_out, float *o_parts_out,
                float *qk_tile_fifo, float *pv_tile_fifo, uint8_t *profile_data, aclrtStream stream,
                uint8_t *cv_comm_buf = nullptr);

template <int S0, int HEAD_SIZE, int S1, int CUBE_S0, int CUBE_S1, int TILE_S1, int QK_PRELOAD, int CV_FIFO_SIZE,
          bool INTERMEDIATE_CHECK, bool CAUSAL_MASK, int CV_FIFO_CONS_SYNC_PERIOD>
void LaunchTGQA(uint16_t *ffts, aclFloat16 *q, aclFloat16 *k, aclFloat16 *v, aclFloat16 *p_tile_fifo,
                float *exp_max_ififo, float *global_sum_out, float *exp_max_out, float *o_out, float *o_parts_out,
                float *qk_tile_fifo, float *pv_tile_fifo, aclrtStream stream, uint8_t *cv_comm_buf = nullptr);

#endif // GQA_PERFORMANCE_KERNEL_H
