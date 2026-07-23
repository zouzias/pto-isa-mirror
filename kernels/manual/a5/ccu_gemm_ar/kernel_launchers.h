/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#pragma once

#include <cstdint>

#include "config.h"
#include "comm_context.h"


// Max parallel CCU missions (host arrays + ProgressDeviceCtx::missions).
static constexpr uint32_t kMaxCcuMissions = 16;

// Progress CKE context passed by value to AIV kernels.
// GM addresses are uint64_t for host portability; device casts to __gm__.
struct ProgressCkeCtx {
    uint64_t ckeSlotVA[2];
    uint32_t ckeMask[2];
    uint64_t itemsDoneAddr;
    uint64_t kernelReadyAddr; // CCU writes 1 after gate WaitEvent; AIV polls !=0
};

// Per owner-group counters in HCCL window (see config.h G_SIGNAL_GROUP_*).
static constexpr uint32_t kGroupDoneStride = G_SIGNAL_GROUP_DONE_STRIDE;
static constexpr uint32_t kGroupDoneMaxGroups = G_SIGNAL_GROUP_DONE_MAX_GROUPS;
static constexpr uint32_t kGroupReadyStride = G_SIGNAL_GROUP_READY_STRIDE;
static constexpr uint32_t kGroupReadyMaxGroups = G_SIGNAL_GROUP_READY_MAX_GROUPS;

// Shared host/device progress header (SCHED_TIMING for both progress paths).
struct ProgressCtx {
    volatile uint32_t numTiles;
    volatile uint32_t schedTimingEnable;
    volatile uint32_t _pad0[2];

    struct Timing {
        volatile uint64_t totalCycles;
        volatile uint64_t notifyCycles;       // USE_PROGRESS=0: TNOTIFY
        volatile uint64_t backpressureCycles; // WaitItemsDoneGE (depth-1 + final drain)
        volatile uint64_t idleSpinCycles;     // USE_PROGRESS=0: idle; progress: WaitCkeCanPublish
        volatile uint64_t triggerCount;
        volatile uint64_t groupDoneCycles;    // progress: WaitAllRanksGroupDone
        volatile uint64_t kernelReadyCycles;  // progress: poll rsKernelReady
        volatile uint64_t finalWaitCycles;    // progress: final WaitItemsDoneGE(issued)
    };

    Timing rsSchedTiming;
};

// AIC compute: groupDone AtomicAdd (default) or readyQueue enqueue (USE_PROGRESS=0).
void launchCcuGemmArCompute(uint8_t *gemm_output, uint8_t *src0, uint8_t *src1, uint8_t *readyQueue, int32_t *groupDone,
                            int rank, void *stream, int compute_block_count, uint32_t k_per_rank, uint32_t rank_size);

// readyQueue escape path (USE_PROGRESS=0).
void launchCcuGemmArScheduler(void *stream, ProgressCkeCtx rsCke, uint8_t *readyQueue, uint8_t *progressCtx,
                                uint8_t *signal_matrix, uint8_t *commCtx, uint32_t rankSize, uint32_t myRank);

// Default progress: local groupDone + peer TNOTIFY(groupReady) + owner TriggerProgressCke.
void launchCcuGemmArProgress(void *stream, ProgressCkeCtx rsCke, int32_t *groupDone, uint8_t *progressCtx,
                               uint8_t *signal_matrix, uint8_t *commCtx, uint32_t numTiles, uint32_t rankSize,
                               uint32_t myRank);

// Convert packed reduced output back to row-major [M, G_N].
void launchCcuGemmArUnpack(uint8_t *packed_output, uint8_t *row_output, void *stream, uint32_t subtilesPerTile,
                             uint32_t rankSize);

// Trigger the CCU gate CKE once.
int launchCcuGemmArGateTrigger(void *stream, uint64_t gateCkeVA, uint32_t gateMask);
