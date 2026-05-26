/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef FA_PERFORMANCE_KERNEL_H
#define FA_PERFORMANCE_KERNEL_H

#include <acl/acl.h>
#include <cstddef>
#include <cstdint>

// Shared defaults for FA performance kernels and host driver
constexpr int kFaCvFifoSize = 8;
constexpr int kFaCvFifoConsSyncPeriod = kFaCvFifoSize / 2;
constexpr int kFaCubeS1 = 128;
constexpr int kFaTileS1 = 256;
constexpr int kFaQkPreload = 4;
constexpr int kFaLaunchCoreCount = 28;
constexpr std::size_t kFaProfileBytesPerBlock = 1024 * 3; // cube + two vec subblocks
constexpr std::size_t kFaCvCommSlotBytes = 512U;
constexpr int VEC_CORES = 2;

constexpr int SKIP_STATUS_FIFO_SIZE = kFaCvFifoSize;
constexpr int SKIP_STATUS_SLOT_SIZE = 2;
constexpr uint64_t SKIP_STATUS_SSBUF_BASE = 0x100;
constexpr uint64_t SKIP_STATUS_SLOT_BYTES = SKIP_STATUS_SLOT_SIZE * sizeof(uint64_t);

// -----------------------------------------------------------------------------
// Buffer flag values for FFTS pipeline coordination
// Each TSync object uses TWO consecutive flags (flag_id and flag_id+1)
// for forward (record/wait) and backward (allocate/free) dependencies.
// Therefore, flag IDs must be spaced by 2 to avoid collisions.
// -----------------------------------------------------------------------------
#ifndef FFTS_BUFFER_FLAG_ENUM
#define FFTS_BUFFER_FLAG_ENUM
enum FftsBufferFlag : uint32_t
{
    BUF0_QK_READY = 0,   // qk2smSync: uses flags 0, 1 (+ 16, 17 for dual core)
    BUF1_SM_READY = 2,   // sm2pvSync: uses flags 2, 3 (+ 18, 19 for dual core)
    UPDATE_READY = 4,    // pv2guSync: uses flags 4, 5 (+ 20, 21 for dual core)
    UB_BUF_READY = 6,    // ubBufSync: uses flags 6, 7 (+ 22, 23 for dual core)
    PV_UB_BUF_READY = 8, // pvUbBufSync: uses flags 8, 9 (+ 24, 25 for dual core)
    CV_BLOCK_END = 10,   // CV comm slot block end (CV_COMM_CTRL reserved in TSyncCVID)
    SS_BUF_READY = 12,
    RUNNING_O_AVALIABLE = 14,
};
#endif

#ifndef MARK_STAMP
// #define MARK_STAMP
// #define MARK_STAMP_DATA_PIPE

enum StageStamp : uint16_t
{
    QK_DONE,
    P_DONE,
    PV_DONE,
    GU_DONE,
};

#endif

template <int S0, int HEAD_SIZE, int S1, int CUBE_S0, int CUBE_S1 = kFaCubeS1, int TILE_S1 = kFaTileS1,
          int QK_PRELOAD = kFaQkPreload, int CV_FIFO_SIZE = kFaCvFifoSize, bool INTERMEDIATE_CHECK = false,
          bool CAUSAL_MASK = false, int CV_FIFO_CONS_SYNC_PERIOD = kFaCvFifoConsSyncPeriod>
void LaunchTFA(uint16_t *ffts, aclFloat16 *q, aclFloat16 *k, aclFloat16 *v, aclFloat16 *p_tile_fifo,
               float *exp_max_ififo, float *o_out, float *o_parts_out, float *qk_tile_fifo, float *pv_tile_fifo,
               float *pv_pend_tile_fifo, uint8_t *profile_data, aclrtStream stream, uint8_t *cv_comm_buf = nullptr);

// Overload without profiling buffer.
template <int S0, int HEAD_SIZE, int S1, int CUBE_S0, int CUBE_S1, int TILE_S1, int QK_PRELOAD, int CV_FIFO_SIZE,
          bool INTERMEDIATE_CHECK, bool CAUSAL_MASK, int CV_FIFO_CONS_SYNC_PERIOD>
void LaunchTFA(uint16_t *ffts, aclFloat16 *q, aclFloat16 *k, aclFloat16 *v, aclFloat16 *p_tile_fifo,
               float *exp_max_ififo, float *o_out, float *o_parts_out, float *qk_tile_fifo, float *pv_tile_fifo,
               float *pv_pend_tile_fifo, aclrtStream stream, uint8_t *cv_comm_buf = nullptr);

#endif // FA_PERFORMANCE_KERNEL_H
