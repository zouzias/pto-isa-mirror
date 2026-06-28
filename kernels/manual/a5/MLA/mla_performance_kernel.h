/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef MLA_PERFORMANCE_KERNEL_H
#define MLA_PERFORMANCE_KERNEL_H

#include <acl/acl.h>
#include <cstddef>
#include <cstdint>

constexpr int kMlaCvFifoSize = 8;
constexpr int kMlaCvFifoConsSyncPeriod = kMlaCvFifoSize / 2;
constexpr int kMlaCubeS1 = 128;
constexpr int kMlaTileS1 = 256;
constexpr int kMlaQkPreload = 4;
constexpr std::size_t kMlaProfileBytesPerBlock = 1024 * 3;
constexpr std::size_t kMlaCvCommSlotBytes = 512U;
constexpr int VEC_CORES = 2;

template <int S0, int HEAD_SIZE, int KV_LATENT_DIM, int S1, int CUBE_S0, int CUBE_S1 = kMlaCubeS1,
          int TILE_S1 = kMlaTileS1, int QK_PRELOAD = kMlaQkPreload, int CV_FIFO_SIZE = kMlaCvFifoSize,
          bool INTERMEDIATE_CHECK = false, bool CAUSAL_MASK = false,
          int CV_FIFO_CONS_SYNC_PERIOD = kMlaCvFifoConsSyncPeriod>
void LaunchTMLA(uint16_t *ffts, aclFloat16 *q, aclFloat16 *c_kv, aclFloat16 *w_uv, aclFloat16 *v_recons_fifo,
                aclFloat16 *p_tile_fifo, float *exp_max_ififo, float *global_sum_out, float *exp_max_out, float *o_out,
                float *o_parts_out, float *qk_tile_fifo, float *pv_tile_fifo, uint8_t *profile_data, aclrtStream stream,
                uint8_t *cv_comm_buf = nullptr);

template <int S0, int HEAD_SIZE, int KV_LATENT_DIM, int S1, int CUBE_S0, int CUBE_S1, int TILE_S1, int QK_PRELOAD,
          int CV_FIFO_SIZE, bool INTERMEDIATE_CHECK, bool CAUSAL_MASK, int CV_FIFO_CONS_SYNC_PERIOD>
void LaunchTMLA(uint16_t *ffts, aclFloat16 *q, aclFloat16 *c_kv, aclFloat16 *w_uv, aclFloat16 *v_recons_fifo,
                aclFloat16 *p_tile_fifo, float *exp_max_ififo, float *global_sum_out, float *exp_max_out, float *o_out,
                float *o_parts_out, float *qk_tile_fifo, float *pv_tile_fifo, aclrtStream stream,
                uint8_t *cv_comm_buf = nullptr);

#endif // MLA_PERFORMANCE_KERNEL_H
