/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <cstdint>

#include <pto/common/kernel_meta.hpp>
#include <pto/pto-inst.hpp>

#include "moe_dispatch_combine_a8w8_types.hpp"
#include "protocol_core.hpp"

#ifndef GM_ADDR
#define GM_ADDR __gm__ uint8_t *
#endif

#if defined(M2_MIXED_SPIKE_BUILD_AIC) || defined(M2_MIXED_SPIKE_BUILD_AIV)
#if defined(M2_MIXED_SPIKE_BUILD_AIC)
#define M2_FUSED_INCLUDE_DEVICE_BODY_ONLY
#define RunInt8GmmTile M2FusedRunInt8GmmTile
#define MakeGlobal2D M2FusedGmmMakeGlobal2D
#define GlobalNd M2FusedGmmGlobalNd
#define Shape2DDyn M2FusedGmmShape2DDyn
#define StrideDyn M2FusedGmmStrideDyn
#define Align64Device M2FusedGmmAlign64Device
#define M2CeilDivDevice M2FusedGmmCeilDivDevice
#define M2DTypeBytes M2FusedGmmDTypeBytes
#define M2GlobalExpertNum M2FusedGmmGlobalExpertNum
#define M2LocalRows M2FusedGmmLocalRows
#define M2TokenPerExpertMatrixRowStride M2FusedGmmTokenPerExpertMatrixRowStride
#define M2DispatchPayloadRowBytes M2FusedGmmDispatchPayloadRowBytes
#define M2ReturnPayloadRowBytes M2FusedGmmReturnPayloadRowBytes
#define M2ReturnHiddenChunkCols M2FusedGmmReturnHiddenChunkCols
#define M2GmmTileTaskCapacity M2FusedGmmTileTaskCapacity
#define M2ReturnSegmentCapacity M2FusedGmmReturnSegmentCapacity
#define M2AppendFieldDevice M2FusedGmmAppendFieldDevice
#define MakeM2WorkspaceLayoutDevice M2FusedGmmMakeWorkspaceLayoutDevice
#define LoadScalarI32 M2FusedGmmLoadScalarI32
#define StoreScalarI32 M2FusedGmmStoreScalarI32
#define BuildGmmTileTaskPlan M2FusedGmmBuildTileTaskPlan
#include "moe_dispatch_combine_a8w8_gmm_kernel.cpp"
#undef BuildGmmTileTaskPlan
#undef StoreScalarI32
#undef LoadScalarI32
#undef MakeM2WorkspaceLayoutDevice
#undef M2AppendFieldDevice
#undef M2ReturnSegmentCapacity
#undef M2GmmTileTaskCapacity
#undef M2ReturnHiddenChunkCols
#undef M2ReturnPayloadRowBytes
#undef M2DispatchPayloadRowBytes
#undef M2TokenPerExpertMatrixRowStride
#undef M2LocalRows
#undef M2GlobalExpertNum
#undef M2DTypeBytes
#undef M2CeilDivDevice
#undef Align64Device
#undef StrideDyn
#undef Shape2DDyn
#undef GlobalNd
#undef MakeGlobal2D
#undef RunInt8GmmTile
#undef M2_FUSED_INCLUDE_DEVICE_BODY_ONLY
#endif

#if defined(M2_MIXED_SPIKE_BUILD_AIV)
#define M2_FUSED_INCLUDE_DEVICE_BODY_ONLY
#define DispatchCombineTileDispatch M2FusedAivDispatchCombineTileDispatch
#define DispatchCombineTileCombine M2FusedAivDispatchCombineTileCombine
#include "moe_dispatch_combine_a8w8_kernel.cpp"
#undef DispatchCombineTileCombine
#undef DispatchCombineTileDispatch
#undef M2_FUSED_INCLUDE_DEVICE_BODY_ONLY
#endif
#endif

#if defined(M2_MIXED_SPIKE_BUILD_AIC) && !defined(M2_MIXED_SPIKE_REGISTER_BUILD)
#include "kernel_launchers.hpp"
#include "runtime/rt.h"
#include "runtime/rt_ffts.h"

#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <fstream>
#include <vector>
#endif

namespace {

constexpr uint64_t kM2MixedSpikeTilingKey = 2801;
constexpr uint64_t kM2FusedSkeletonTilingKey = 2802;
constexpr uint64_t kM2FusedFullTilingKey = 2803;
constexpr int32_t kM2MixedSpikeMagic = 0x4D328A8;
constexpr int32_t kM2FusedSkeletonMagic = 0x4D328B9;
constexpr int32_t kM2FusedFullMagic = 0x4D328CA;
constexpr uint32_t kM2MixedSpikeAicBase = 16;
constexpr uint32_t kM2MixedSpikeAicHeader = 112;
constexpr uint32_t kM2MixedSpikeAivHeader = 120;
constexpr uint32_t kM2MixedSpikeAivBase = 128;
constexpr uint32_t kM2MixedSpikeMaxAicSlots = 96;
constexpr uint32_t kM2MixedSpikeMaxAivSlots = 192;
constexpr uint32_t kM2MixedSpikeParamRank = 8;
constexpr uint32_t kM2MixedSpikeParamAicBlocks = 9;
constexpr uint32_t kM2MixedSpikeParamAivRatio = 10;
constexpr uint32_t kM2FusedSkeletonAicHeader = 256;
constexpr uint32_t kM2FusedSkeletonAivHeader = 264;
constexpr uint32_t kM2FusedSkeletonAicStageBase = 288;
constexpr uint32_t kM2FusedSkeletonStageCount = 4;
constexpr uint32_t kM2FusedSkeletonParticipantCount = 48;
constexpr uint64_t kM2FusedSkeletonStoreUbAddr = 0x3000;
constexpr uint64_t kM2FusedSkeletonStoreL1Addr = 0x3000;
constexpr uint64_t kM2FusedVecProbeUbAddr = 0x8000;
constexpr uint32_t kM2FusedSkeletonStageSlotWords = 8;
constexpr uint32_t kM2FusedSkeletonAivStageBase = kM2FusedSkeletonAicStageBase + kM2FusedSkeletonStageCount *
                                                                                     kM2FusedSkeletonParticipantCount *
                                                                                     kM2FusedSkeletonStageSlotWords;
constexpr uint32_t kM2FusedFullAicHeaderSlot = 8U * 16U;
constexpr uint32_t kM2FusedFullAivHeaderSlot = 9U * 16U;
constexpr uint32_t kM2FusedFullStageBaseSlot = 10U * 16U;
constexpr uint32_t kM2FusedFullAicStageBaseSlot = 11U * 16U;
constexpr uint32_t kM2FusedFullAivStageBaseSlot = 12U * 16U;
constexpr uint32_t kM2FusedFullM3N5AicEvidenceSlot = 14U * 16U;
constexpr uint32_t kM2FusedFullM3N6EvidenceSlot = kM2FusedFullM3N5AicEvidenceSlot + 2U;
constexpr uint32_t kM2FusedFullM3N7EvidenceSlot = kM2FusedFullM3N6EvidenceSlot + 3U;
constexpr uint32_t kM2FusedFullM3N8EvidenceSlot = kM2FusedFullM3N7EvidenceSlot + 3U;
constexpr uint32_t kM2FusedFullDebugStopSlot = 15U * 16U;
constexpr uint32_t kM2FusedFullM3N7Gmm2CounterBase = 24U * 16U + 64U;
constexpr uint32_t kM2FusedFullStageCount = 7U;
constexpr uint32_t kM3NFusedDispatchScratchBase = 32U * 16U;
constexpr uint32_t kM3NFusedDispatchScratchLimit = 40U * 16U;
constexpr uint32_t kM3NFusedDispatchLaneDebugBase = 40U * 16U;
constexpr uint32_t kM3NFusedDispatchMaxLaneSlots = 48U;

AICORE inline bool IsM2FusedMainAic()
{
#if defined(__DAV_CUBE__)
    return true;
#else
    return false;
#endif
}

AICORE inline bool IsM2FusedMainAiv()
{
#if defined(__DAV_VEC__)
    return get_block_idx() == 0 && get_subblockid() == 0;
#else
    return false;
#endif
}

AICORE inline uint32_t M2FusedLogicalAivId()
{
#if defined(__DAV_VEC__)
    return static_cast<uint32_t>(get_block_idx() * get_subblockdim() + get_subblockid());
#else
    return 0;
#endif
}

AICORE inline uint32_t M2FusedRawAivSlot()
{
#if defined(__DAV_VEC__)
    return static_cast<uint32_t>(get_block_idx() * get_subblockdim() + get_subblockid());
#else
    return 0;
#endif
}

AICORE inline uint32_t M2FusedLogicalAivCount()
{
#if defined(__DAV_VEC__)
    return static_cast<uint32_t>(get_block_num() * get_subblockdim());
#else
    return 1U;
#endif
}

AICORE inline void M3NDispatchHardPhaseSync()
{
    pipe_barrier(PIPE_ALL);
    dsb(DSB_DDR);
    pto::SYNCALL<pto::SyncCoreType::Mix>();
}

AICORE inline void M3NDispatchAivOnlyPhaseSync()
{
    moe_dispatch_combine_a8w8::M3N4AivAllDoneCoarseSync();
}

AICORE inline bool M3NFusedDispatchScratchFits(moe_dispatch_combine_a8w8::ShapeConfig shape)
{
    uint32_t globalExpertNum = shape.rankNum * shape.expertPerRank;
    return globalExpertNum > 0U && globalExpertNum <= kM3NFusedDispatchScratchLimit - kM3NFusedDispatchScratchBase;
}

AICORE inline bool M3N5DispatchGmm1OverlapEnabled(uint32_t debugStopStage, uint32_t overlapMode,
                                                  moe_dispatch_combine_a8w8::ShapeConfig shape)
{
    return overlapMode != 0U && debugStopStage < 100U && M3NFusedDispatchScratchFits(shape) &&
           moe_dispatch_combine_a8w8::M3N5DispatchExpertFlagsSupported(shape);
}

AICORE inline bool M3N6Gmm1ActivationOverlapEnabled(uint32_t debugStopStage, uint32_t overlapMode,
                                                    moe_dispatch_combine_a8w8::ShapeConfig shape)
{
    return overlapMode != 0U && debugStopStage < 100U && M3NFusedDispatchScratchFits(shape) &&
           moe_dispatch_combine_a8w8::M3N6Gmm1SyncGroupFlagsSupported(shape);
}

AICORE inline bool M3N7ActivationGmm2OverlapEnabled(uint32_t debugStopStage, uint32_t overlapMode,
                                                    moe_dispatch_combine_a8w8::ShapeConfig shape)
{
    return M3N6Gmm1ActivationOverlapEnabled(debugStopStage, overlapMode, shape);
}

AICORE inline bool M3N8Gmm2CombineOverlapEnabled(uint32_t debugStopStage, uint32_t overlapMode,
                                                 moe_dispatch_combine_a8w8::ShapeConfig shape)
{
    return M3N7ActivationGmm2OverlapEnabled(debugStopStage, overlapMode, shape);
}

AICORE inline void M3NDispatchAicWaitForAivSubphases(uint32_t debugStopStage, bool m3n5DispatchGmm1Overlap,
                                                     moe_dispatch_combine_a8w8::ShapeConfig shape)
{
    if (debugStopStage >= 100U || !M3NFusedDispatchScratchFits(shape)) {
        return;
    }
    uint32_t dispatchSubphaseSyncs = m3n5DispatchGmm1Overlap ? 10U : 11U;
    for (uint32_t sync = 0; sync < dispatchSubphaseSyncs; ++sync) {
        M3NDispatchHardPhaseSync();
    }
}

AICORE inline void M3NActivationAicWaitForAivSubphases(uint32_t debugStopStage,
                                                       moe_dispatch_combine_a8w8::ShapeConfig shape)
{
    constexpr uint32_t kActivationSubphaseSyncs = 3U;
    if (debugStopStage >= 100U || !M3NFusedDispatchScratchFits(shape)) {
        return;
    }
    for (uint32_t sync = 0; sync < kActivationSubphaseSyncs; ++sync) {
        M3NDispatchHardPhaseSync();
    }
}

AICORE inline bool M3NCombineAicWaitForAivSubphases(uint32_t debugStopStage,
                                                    moe_dispatch_combine_a8w8::ShapeConfig shape)
{
    if (debugStopStage >= 100U || !M3NFusedDispatchScratchFits(shape)) {
        return false;
    }
    M3NDispatchHardPhaseSync();
    if (debugStopStage == 51U) {
        return true;
    }
    M3NDispatchHardPhaseSync();
    return debugStopStage == 52U || debugStopStage == 53U || debugStopStage == 54U || debugStopStage == 56U;
}

AICORE inline void M3N5WaitAllDispatchExpertsReady(moe_dispatch_combine_a8w8::ShapeConfig shape)
{
    for (uint32_t localExpert = 0; localExpert < shape.expertPerRank; ++localExpert) {
        moe_dispatch_combine_a8w8::M3N5WaitDispatchExpertReady(localExpert);
    }
}

AICORE inline void M3N6SyncGroupExpertRange(moe_dispatch_combine_a8w8::ShapeConfig shape, uint32_t syncIdx,
                                            uint32_t *expertBegin, uint32_t *expertEnd)
{
    uint32_t begin = 0;
    for (uint32_t group = 0; group < syncIdx && begin < shape.expertPerRank; ++group) {
        begin += moe_dispatch_combine_a8w8::M3N6NextSwigluGroupSize(shape.expertPerRank - begin);
    }
    uint32_t end = begin;
    if (begin < shape.expertPerRank) {
        end = begin + moe_dispatch_combine_a8w8::M3N6NextSwigluGroupSize(shape.expertPerRank - begin);
    }
    if (end > shape.expertPerRank) {
        end = shape.expertPerRank;
    }
    *expertBegin = begin;
    *expertEnd = end;
}

#if defined(M2_MIXED_SPIKE_BUILD_AIC)
AICORE inline void M3N5RecordGmm1StartBeforeLastDispatchReady(moe_dispatch_combine_a8w8::ShapeConfig shape,
                                                              __gm__ int32_t *stageStatus, uint32_t localExpert)
{
    if (localExpert != 0U || shape.expertPerRank <= 1U) {
        return;
    }
    M2FusedGmmStoreScalarI32(stageStatus + kM2FusedFullM3N5AicEvidenceSlot, 1);
    M2FusedGmmStoreScalarI32(stageStatus + kM2FusedFullM3N5AicEvidenceSlot + 1U, 1);
    pipe_barrier(PIPE_ALL);
    dsb(DSB_DDR);
}

AICORE inline void M3N6RecordGmm1SyncGroupReady(__gm__ int32_t *stageStatus,
                                                moe_dispatch_combine_a8w8::WorkspaceLayout layout, GM_ADDR workspace,
                                                uint32_t syncIdx, uint32_t syncGroupCount)
{
    __gm__ int32_t *gmm1SyncGroupReady =
        reinterpret_cast<__gm__ int32_t *>(workspace + layout.gmm1SyncGroupReady.offset);
    M2FusedGmmStoreScalarI32(gmm1SyncGroupReady + syncIdx * 16U, 1);
    if (syncIdx == 0U && syncGroupCount > 1U) {
        M2FusedGmmStoreScalarI32(stageStatus + kM2FusedFullM3N6EvidenceSlot + 0U, 1);
    }
    if (syncIdx + 1U == syncGroupCount) {
        M2FusedGmmStoreScalarI32(stageStatus + kM2FusedFullM3N6EvidenceSlot + 1U, 1);
    }
    pipe_barrier(PIPE_ALL);
    dsb(DSB_DDR);
}

AICORE inline uint64_t M3NFusedPeerAppendOffset(uint64_t *offset, uint64_t bytes)
{
    *offset = M2FusedGmmAlign64Device(*offset);
    uint64_t field = *offset;
    *offset += bytes;
    return field;
}

AICORE inline __gm__ int32_t *M3NFusedPeerDebugCounters(GM_ADDR peerWindow,
                                                        moe_dispatch_combine_a8w8::ShapeConfig shape)
{
    uint64_t offset = 0;
    uint64_t expandedRows = static_cast<uint64_t>(shape.m) * shape.topK;
    uint64_t tokenMatrixCount = static_cast<uint64_t>(shape.rankNum) * M2FusedGmmTokenPerExpertMatrixRowStride(shape);
    (void)M3NFusedPeerAppendOffset(&offset, moe_dispatch_combine_a8w8::kPeerWindowHeaderBytes);
    (void)M3NFusedPeerAppendOffset(&offset, tokenMatrixCount * sizeof(int32_t));
    (void)M3NFusedPeerAppendOffset(&offset, static_cast<uint64_t>(shape.rankNum) * 64U);
    (void)M3NFusedPeerAppendOffset(&offset, expandedRows * M2FusedGmmDispatchPayloadRowBytes(shape));
    (void)M3NFusedPeerAppendOffset(&offset, expandedRows * sizeof(float));
    (void)M3NFusedPeerAppendOffset(&offset, expandedRows * M2FusedGmmReturnPayloadRowBytes(shape));
    (void)M3NFusedPeerAppendOffset(&offset, static_cast<uint64_t>(shape.rankNum) * 64U);
    (void)M3NFusedPeerAppendOffset(&offset, static_cast<uint64_t>(shape.rankNum) * shape.expertPerRank * 64U);
    uint64_t debugOffset = M3NFusedPeerAppendOffset(&offset, 80U * 64U);
    return reinterpret_cast<__gm__ int32_t *>(peerWindow + debugOffset);
}

AICORE inline bool M3N7Gmm2TaskIntersectsSyncGroup(__gm__ int32_t *task, __gm__ int32_t *group)
{
    int32_t expertBegin = M2FusedGmmLoadScalarI32(group + 1U);
    int32_t expertEnd = M2FusedGmmLoadScalarI32(group + 2U);
    int32_t rowBegin = M2FusedGmmLoadScalarI32(group + 3U);
    int32_t rowEnd = M2FusedGmmLoadScalarI32(group + 4U);
    int32_t localExpert = M2FusedGmmLoadScalarI32(task + 2U);
    int32_t taskRowBegin = M2FusedGmmLoadScalarI32(task + 3U);
    int32_t taskRows = M2FusedGmmLoadScalarI32(task + 4U);
    int32_t taskRowEnd = taskRowBegin + taskRows;
    return localExpert >= expertBegin && localExpert < expertEnd && taskRows > 0 && taskRowBegin < rowEnd &&
           taskRowEnd > rowBegin;
}

AICORE inline uint32_t M3N7CountGmm2SyncGroupTasks(__gm__ int32_t *taskPlan, uint32_t taskCount, __gm__ int32_t *group,
                                                   int32_t *firstTaskId, int32_t *firstTaskExpert,
                                                   int32_t *firstTaskRowBegin)
{
    uint32_t intersectTaskCount = 0;
    *firstTaskId = -1;
    *firstTaskExpert = -1;
    *firstTaskRowBegin = -1;
    for (uint32_t taskId = 0; taskId < taskCount; ++taskId) {
        __gm__ int32_t *task = taskPlan + taskId * kM2GmmTileTaskFields;
        if (!M3N7Gmm2TaskIntersectsSyncGroup(task, group)) {
            continue;
        }
        if (intersectTaskCount == 0U) {
            *firstTaskId = static_cast<int32_t>(taskId);
            *firstTaskExpert = M2FusedGmmLoadScalarI32(task + 2U);
            *firstTaskRowBegin = M2FusedGmmLoadScalarI32(task + 3U);
        }
        ++intersectTaskCount;
    }
    return intersectTaskCount;
}

AICORE inline void M3N7InitGmm2Counters(__gm__ int32_t *debugCounters, uint32_t syncGroupCount)
{
    for (uint32_t idx = 0; idx < 16U; ++idx) {
        M2FusedGmmStoreScalarI32(debugCounters + kM2FusedFullM3N7Gmm2CounterBase + idx, 0);
    }
    M2FusedGmmStoreScalarI32(debugCounters + kM2FusedFullM3N7Gmm2CounterBase + 0U, 1);
    M2FusedGmmStoreScalarI32(debugCounters + kM2FusedFullM3N7Gmm2CounterBase + 1U,
                             static_cast<int32_t>(syncGroupCount));
    pipe_barrier(PIPE_ALL);
    dsb(DSB_DDR);
}

AICORE inline void M3N7WaitActivationSyncGroupReadyGm(__gm__ int32_t *activationSyncGroupReady, uint32_t syncIdx)
{
    __gm__ int32_t *ready = activationSyncGroupReady + syncIdx * 16U;
    while (true) {
        pipe_barrier(PIPE_ALL);
        dcci(static_cast<__gm__ void *>(ready), SINGLE_CACHE_LINE);
        dsb(DSB_DDR);
        if (M2FusedGmmLoadScalarI32(ready) != 0) {
            return;
        }
    }
}

AICORE inline void M3N7RecordGmm2SyncGroupConsumed(__gm__ int32_t *stageStatus, __gm__ int32_t *debugCounters,
                                                   uint32_t syncIdx, uint32_t syncGroupCount, __gm__ int32_t *group,
                                                   uint32_t intersectTaskCount, int32_t firstTaskId,
                                                   int32_t firstTaskExpert, int32_t firstTaskRowBegin)
{
    int32_t previousTasks = M2FusedGmmLoadScalarI32(debugCounters + kM2FusedFullM3N7Gmm2CounterBase + 3U);
    M2FusedGmmStoreScalarI32(debugCounters + kM2FusedFullM3N7Gmm2CounterBase + 3U,
                             previousTasks + static_cast<int32_t>(intersectTaskCount));
    int32_t consumed = M2FusedGmmLoadScalarI32(debugCounters + kM2FusedFullM3N7Gmm2CounterBase + 4U) + 1;
    M2FusedGmmStoreScalarI32(debugCounters + kM2FusedFullM3N7Gmm2CounterBase + 4U, consumed);
    M2FusedGmmStoreScalarI32(debugCounters + kM2FusedFullM3N7Gmm2CounterBase + 5U, static_cast<int32_t>(syncIdx));
    M2FusedGmmStoreScalarI32(debugCounters + kM2FusedFullM3N7Gmm2CounterBase + 6U, M2FusedGmmLoadScalarI32(group + 1U));
    M2FusedGmmStoreScalarI32(debugCounters + kM2FusedFullM3N7Gmm2CounterBase + 7U, M2FusedGmmLoadScalarI32(group + 2U));
    M2FusedGmmStoreScalarI32(debugCounters + kM2FusedFullM3N7Gmm2CounterBase + 8U, M2FusedGmmLoadScalarI32(group + 3U));
    M2FusedGmmStoreScalarI32(debugCounters + kM2FusedFullM3N7Gmm2CounterBase + 9U, M2FusedGmmLoadScalarI32(group + 4U));
    M2FusedGmmStoreScalarI32(debugCounters + kM2FusedFullM3N7Gmm2CounterBase + 10U,
                             M2FusedGmmLoadScalarI32(group + 5U));
    M2FusedGmmStoreScalarI32(debugCounters + kM2FusedFullM3N7Gmm2CounterBase + 11U,
                             M2FusedGmmLoadScalarI32(group + 6U));
    if (previousTasks == 0 && intersectTaskCount > 0U) {
        M2FusedGmmStoreScalarI32(debugCounters + kM2FusedFullM3N7Gmm2CounterBase + 12U, firstTaskId);
        M2FusedGmmStoreScalarI32(debugCounters + kM2FusedFullM3N7Gmm2CounterBase + 13U, firstTaskExpert);
        M2FusedGmmStoreScalarI32(debugCounters + kM2FusedFullM3N7Gmm2CounterBase + 15U, firstTaskRowBegin);
    }
    if (syncIdx == 0U && syncGroupCount > 1U) {
        M2FusedGmmStoreScalarI32(debugCounters + kM2FusedFullM3N7Gmm2CounterBase + 14U, 1);
        M2FusedGmmStoreScalarI32(stageStatus + kM2FusedFullM3N7EvidenceSlot, 1);
    }
    if (syncIdx + 1U == syncGroupCount) {
        M2FusedGmmStoreScalarI32(stageStatus + kM2FusedFullM3N7EvidenceSlot + 1U, 1);
    }
    M2FusedGmmStoreScalarI32(stageStatus + kM2FusedFullM3N7EvidenceSlot + 2U, static_cast<int32_t>(intersectTaskCount));
    pipe_barrier(PIPE_ALL);
    dsb(DSB_DDR);
}
#endif

#if defined(M2_MIXED_SPIKE_BUILD_AIV)
AICORE inline void M3N5WaitDispatchGroupReadyGm(__gm__ int32_t *ready)
{
    while (true) {
        InvalidateGmCacheLines(ready, sizeof(int32_t));
        if (LoadScalarI32(ready) != 0) {
            return;
        }
    }
}

AICORE inline void M3N5WaitAicFirstExpertStarted(__gm__ int32_t *stageStatus)
{
    __gm__ int32_t *started = stageStatus + kM2FusedFullM3N5AicEvidenceSlot + 1U;
    while (true) {
        InvalidateGmCacheLines(started, sizeof(int32_t));
        if (LoadScalarI32(started) != 0) {
            return;
        }
    }
}

AICORE inline void M3N5RecordAivObservedAicFirstExpertStarted(__gm__ int32_t *stageStatus,
                                                              M2PeerWindowViewDevice localPeer)
{
    StoreScalarI32(stageStatus + kM2FusedFullM3N5AicEvidenceSlot, 1);
    StoreScalarI32(localPeer.debugCounters + kM3NDispatchCounterBase + 15U, 1);
    pipe_barrier(PIPE_ALL);
    dsb(DSB_DDR);
}

AICORE inline void M3N6RecordActivationStartedBeforeLastGmm1Ready(__gm__ int32_t *stageStatus,
                                                                  M2PeerWindowViewDevice localPeer, uint32_t syncIdx,
                                                                  uint32_t syncGroupCount)
{
    (void)localPeer;
    if (syncIdx != 0U || syncGroupCount <= 1U) {
        return;
    }
    __gm__ int32_t *firstReady = stageStatus + kM2FusedFullM3N6EvidenceSlot + 0U;
    InvalidateGmCacheLines(firstReady, sizeof(int32_t) * 2U);
    if (LoadScalarI32(firstReady) == 0) {
        return;
    }
    if (LoadScalarI32(firstReady + 1U) != 0) {
        return;
    }
    StoreScalarI32(stageStatus + kM2FusedFullM3N6EvidenceSlot + 2U, 1);
    pipe_barrier(PIPE_ALL);
    dsb(DSB_DDR);
}

AICORE inline void M3N5GatherOneDispatchExpert(moe_dispatch_combine_a8w8::ShapeConfig shape,
                                               M2WorkspaceViewDevice workspaceView, M2PeerWindowViewDevice localPeer,
                                               __gm__ HcclDeviceContext *ctx, GM_ADDR peerWindow, uint32_t myRank,
                                               uint32_t rowBytes,
                                               const moe_dispatch_combine_a8w8::PeerWindowLayout &peerWindowLayout,
                                               uint32_t localExpert)
{
    uint32_t globalExpert = M2GlobalExpert(myRank, localExpert, shape);
    for (uint32_t tokenOwner = 0; tokenOwner < shape.rankNum; ++tokenOwner) {
        int32_t current = LoadScalarI32(workspaceView.cumsumMM + tokenOwner * shape.expertPerRank + localExpert);
        int32_t previous =
            tokenOwner == 0U ?
                0 :
                LoadScalarI32(workspaceView.cumsumMM + (tokenOwner - 1U) * shape.expertPerRank + localExpert);
        int32_t rows = current - previous;
        int32_t dstStart =
            LoadScalarI32(workspaceView.dispatchOffset + localExpert) +
            LoadScalarI32(workspaceView.preSumBeforeRank + tokenOwner * shape.expertPerRank + localExpert);
        M2RecordDispatchLedger(shape, workspaceView, tokenOwner, localExpert, dstStart, rows);
        if (rows <= 0) {
            continue;
        }
        int32_t srcStart = M2SourceGlobalExpertRowBase(shape, localPeer, tokenOwner, globalExpert);
        int32_t localRowCap = static_cast<int32_t>(M2LocalRows(shape));
        if (srcStart >= localRowCap) {
            M2PublishDispatchLedgerCopyDone(shape, workspaceView, tokenOwner, localExpert);
            continue;
        }
        if (srcStart + rows > localRowCap) {
            rows = localRowCap - srcStart;
        }
        if (rows <= 0) {
            M2PublishDispatchLedgerCopyDone(shape, workspaceView, tokenOwner, localExpert);
            continue;
        }
        M2PeerWindowViewDevice remotePeer =
            MakeM2RemotePeerWindowViewDevice(ctx, peerWindow, tokenOwner, peerWindowLayout);
        M2TGetRowsInt8(workspaceView.gmm1InputInt8, static_cast<int32_t>(rowBytes), dstStart,
                       remotePeer.dispatchPayload, static_cast<int32_t>(rowBytes), srcStart, rows,
                       static_cast<int32_t>(rowBytes));
        M2TGetRowsFloat(workspaceView.routingPerTokenScale, dstStart, remotePeer.dispatchScale, srcStart, rows);
        M2PublishDispatchLedgerCopyDone(shape, workspaceView, tokenOwner, localExpert);
    }
    M2UpdateDispatchScoreboardDomain(shape, workspaceView, localExpert);
    pipe_barrier(PIPE_ALL);
    dsb(DSB_DDR);
    StoreScalarI32(workspaceView.dispatchGroupReady + localExpert * 16U, 1);
}

AICORE inline void M3N5GatherDispatchToGmm1InputShardByExpert(
    moe_dispatch_combine_a8w8::ShapeConfig shape, M2WorkspaceViewDevice workspaceView, M2PeerWindowViewDevice localPeer,
    __gm__ HcclDeviceContext *ctx, GM_ADDR peerWindow, uint32_t myRank, uint32_t rowBytes,
    const moe_dispatch_combine_a8w8::PeerWindowLayout &peerWindowLayout, uint32_t workerId, uint32_t workerCount,
    bool activeWorker, __gm__ int32_t *stageStatus, bool requireAicStartAck)
{
    if (IsM2FusedMainAiv()) {
        StoreScalarI32(localPeer.debugCounters + kM3NDispatchCounterBase + 12U, 1);
        StoreScalarI32(localPeer.debugCounters + kM3NDispatchCounterBase + 13U, 0);
        StoreScalarI32(localPeer.debugCounters + kM3NDispatchCounterBase + 14U, 0);
        StoreScalarI32(localPeer.debugCounters + kM3NDispatchCounterBase + 15U, 0);
    }
    for (uint32_t localExpert = 0; localExpert < shape.expertPerRank; ++localExpert) {
        if (activeWorker && workerCount != 0U && (localExpert % workerCount) == workerId) {
            M3N5GatherOneDispatchExpert(shape, workspaceView, localPeer, ctx, peerWindow, myRank, rowBytes,
                                        peerWindowLayout, localExpert);
        }
        M3N5WaitDispatchGroupReadyGm(workspaceView.dispatchGroupReady + localExpert * 16U);
        moe_dispatch_combine_a8w8::M3N5SignalDispatchExpertReady(localExpert);
        if (requireAicStartAck && localExpert == 0U && shape.expertPerRank > 1U) {
            M3N5WaitAicFirstExpertStarted(stageStatus);
            M3N5RecordAivObservedAicFirstExpertStarted(stageStatus, localPeer);
        }
    }
    if (IsM2FusedMainAiv()) {
        int32_t readyExperts = 0;
        int32_t zeroTokenExperts = 0;
        for (uint32_t localExpert = 0; localExpert < shape.expertPerRank; ++localExpert) {
            if (LoadScalarI32(workspaceView.dispatchGroupReady + localExpert * 16U) != 0) {
                ++readyExperts;
            }
            if (LoadScalarI32(workspaceView.expertTokenNums + localExpert) <= 0) {
                ++zeroTokenExperts;
            }
        }
        StoreScalarI32(localPeer.debugCounters + kM3NDispatchCounterBase + 13U, readyExperts);
        StoreScalarI32(localPeer.debugCounters + kM3NDispatchCounterBase + 14U, zeroTokenExperts);
    }
    InvalidateGmCacheLines(workspaceView.gmm1InputInt8, static_cast<uint32_t>(M2LocalRows(shape) * rowBytes));
    InvalidateGmCacheLines(workspaceView.routingPerTokenScale,
                           static_cast<uint32_t>(M2LocalRows(shape) * sizeof(float)));
}
#endif

AICORE inline void StoreI32(__gm__ int32_t *ptr, int32_t value)
{
    *ptr = value;
}

AICORE inline void StoreI32Line(__gm__ int32_t *dst, int32_t value, uint64_t ubAddr, uint64_t l1Addr)
{
#if defined(__DAV_VEC__)
    (void)l1Addr;
    __ubuf__ int32_t *ub = reinterpret_cast<__ubuf__ int32_t *>(ubAddr);
    ub[0] = value;
    pipe_barrier(PIPE_ALL);
    copy_ubuf_to_gm(static_cast<__gm__ void *>(dst), static_cast<__ubuf__ void *>(ub), 0, 1, 1, 0, 0);
    pipe_barrier(PIPE_ALL);
    dcci(static_cast<__gm__ void *>(dst), SINGLE_CACHE_LINE);
    dsb(DSB_DDR);
#elif defined(__DAV_CUBE__)
    (void)ubAddr;
    __cbuf__ int32_t *l1 = reinterpret_cast<__cbuf__ int32_t *>(l1Addr);
    constexpr int64_t repeatConfig = (static_cast<int64_t>(1) << 16) | 1;
    create_cbuf_matrix(l1, repeatConfig, static_cast<uint32_t>(value));
    pipe_barrier(PIPE_ALL);
    copy_cbuf_to_gm(static_cast<__gm__ void *>(dst), static_cast<__cbuf__ void *>(l1), 0, 1, 1, 0, 0);
    pipe_barrier(PIPE_ALL);
    dcci(static_cast<__gm__ void *>(dst), SINGLE_CACHE_LINE);
    dsb(DSB_DDR);
#else
    (void)dst;
    (void)value;
    (void)ubAddr;
    (void)l1Addr;
#endif
}

AICORE inline void M2FusedRecordStage(__gm__ int32_t *stageStatus, uint32_t slot, int32_t value)
{
    StoreI32Line(stageStatus + slot, value, kM2FusedSkeletonStoreUbAddr, kM2FusedSkeletonStoreL1Addr);
}

#if defined(M2_MIXED_SPIKE_BUILD_AIV)
AICORE inline __gm__ int32_t *M3NDispatchLaneSlot(M2PeerWindowViewDevice localPeer, uint32_t rawAivSlot)
{
    return localPeer.debugCounters + kM3NFusedDispatchLaneDebugBase + rawAivSlot * 8U;
}

AICORE inline void M3NRecordAivLaneDebug(M2PeerWindowViewDevice localPeer, uint32_t rawAivSlot)
{
#if defined(__DAV_VEC__)
    if (rawAivSlot >= kM3NFusedDispatchMaxLaneSlots) {
        return;
    }
    __gm__ int32_t *slot = M3NDispatchLaneSlot(localPeer, rawAivSlot);
    StoreScalarI32(slot + 0U, 1);
    StoreScalarI32(slot + 1U, static_cast<int32_t>(get_block_idx()));
    StoreScalarI32(slot + 2U, static_cast<int32_t>(get_subblockid()));
    StoreScalarI32(slot + 3U, static_cast<int32_t>(get_subblockdim()));
    dcci(static_cast<__gm__ void *>(slot), SINGLE_CACHE_LINE);
    dsb(DSB_DDR);
#else
    (void)localPeer;
    (void)rawAivSlot;
#endif
}

AICORE inline void M3N9RecordTimeoutDump(M2WorkspaceViewDevice workspaceView, M2PeerWindowViewDevice localPeer,
                                         moe_dispatch_combine_a8w8::RankConfig rank,
                                         moe_dispatch_combine_a8w8::ShapeConfig shape, uint32_t localExpert,
                                         uint32_t tokenOwnerRank, uint32_t stage, uint32_t signalId,
                                         uint32_t debugStopStage)
{
    if (shape.expertPerRank == 0U) {
        return;
    }
    if (localExpert >= shape.expertPerRank) {
        localExpert = 0U;
    }
    if (tokenOwnerRank >= shape.rankNum) {
        tokenOwnerRank = 0U;
    }
    __gm__ int32_t *dump =
        workspaceView.scoreboardTimeoutCounters + localExpert * moe_dispatch_combine_a8w8::kM3N9TimeoutDumpStride;
    StoreScalarI32(dump + moe_dispatch_combine_a8w8::kM3N9TimeoutDumpPresentSlot, 1);
    StoreScalarI32(dump + moe_dispatch_combine_a8w8::kM3N9TimeoutDumpRankSlot, static_cast<int32_t>(rank.rankId));
    StoreScalarI32(dump + moe_dispatch_combine_a8w8::kM3N9TimeoutDumpExpertSlot, static_cast<int32_t>(localExpert));
    StoreScalarI32(dump + moe_dispatch_combine_a8w8::kM3N9TimeoutDumpTokenOwnerRankSlot,
                   static_cast<int32_t>(tokenOwnerRank));
    StoreScalarI32(dump + moe_dispatch_combine_a8w8::kM3N9TimeoutDumpExpertOwnerRankSlot,
                   static_cast<int32_t>(rank.rankId));
    StoreScalarI32(dump + moe_dispatch_combine_a8w8::kM3N9TimeoutDumpStageSlot, static_cast<int32_t>(stage));
    StoreScalarI32(dump + moe_dispatch_combine_a8w8::kM3N9TimeoutDumpSignalIdSlot, static_cast<int32_t>(signalId));
    StoreScalarI32(dump + moe_dispatch_combine_a8w8::kM3N9TimeoutDumpDebugStopStageSlot,
                   static_cast<int32_t>(debugStopStage));
    StoreScalarI32(localPeer.debugCounters + kM3CounterBase + 15U,
                   LoadScalarI32(localPeer.debugCounters + kM3CounterBase + 15U) + 1);
    pipe_barrier(PIPE_ALL);
    dsb(DSB_DDR);
}

AICORE inline uint32_t M3NAssignDispatchWorkers(moe_dispatch_combine_a8w8::ShapeConfig shape,
                                                M2PeerWindowViewDevice localPeer, uint32_t logicalAivCount)
{
    uint32_t targetWorkers = M3NDispatchWorkerCount(shape, logicalAivCount);
    uint32_t scanSlots = logicalAivCount;
    if (scanSlots > kM3NFusedDispatchMaxLaneSlots) {
        scanSlots = kM3NFusedDispatchMaxLaneSlots;
    }
    uint32_t assigned = 0;
    for (uint32_t raw = 0; raw < scanSlots; ++raw) {
        __gm__ int32_t *slot = M3NDispatchLaneSlot(localPeer, raw);
        InvalidateGmCacheLines(slot, 64U);
        if (LoadScalarI32(slot) == 0) {
            continue;
        }
        if (LoadScalarI32(slot + 2U) == 0) {
            StoreScalarI32(slot + 4U, 0);
            dcci(static_cast<__gm__ void *>(slot), SINGLE_CACHE_LINE);
            continue;
        }
        if (assigned < targetWorkers) {
            StoreScalarI32(slot + 4U, static_cast<int32_t>(assigned + 1U));
            ++assigned;
        } else {
            StoreScalarI32(slot + 4U, 0);
        }
        dcci(static_cast<__gm__ void *>(slot), SINGLE_CACHE_LINE);
    }
    if (assigned == 0U) {
        assigned = 1U;
    }
    StoreScalarI32(localPeer.debugCounters + kM3NDispatchCounterBase + 3U, static_cast<int32_t>(assigned));
    StoreScalarI32(localPeer.debugCounters + kM3NDispatchCounterBase + 11U, static_cast<int32_t>(assigned));
    return assigned;
}

struct M3NDispatchWorkerAssignment {
    uint32_t workerId;
    uint32_t workerCount;
    bool active;
};

struct M3NActivationLaneSummary {
    uint32_t rowsProcessed;
    uint32_t activeWorkers;
    uint32_t activeWorkerMask;
};

AICORE inline M3NDispatchWorkerAssignment M3NLoadDispatchWorkerAssignment(M2PeerWindowViewDevice localPeer,
                                                                          uint32_t rawAivSlot)
{
    M3NDispatchWorkerAssignment assignment{0U, 1U, false};
    int32_t workerCount = LoadScalarI32(localPeer.debugCounters + kM3NDispatchCounterBase + 3U);
    if (workerCount > 0) {
        assignment.workerCount = static_cast<uint32_t>(workerCount);
    }
    if (rawAivSlot >= kM3NFusedDispatchMaxLaneSlots) {
        return assignment;
    }
    __gm__ int32_t *slot = M3NDispatchLaneSlot(localPeer, rawAivSlot);
    InvalidateGmCacheLines(slot, 64U);
    int32_t workerPlusOne = LoadScalarI32(slot + 4U);
    if (workerPlusOne <= 0) {
        return assignment;
    }
    assignment.workerId = static_cast<uint32_t>(workerPlusOne - 1);
    assignment.active = assignment.workerId < assignment.workerCount;
    return assignment;
}

AICORE inline void M3NRecordActivationLaneDebug(M2PeerWindowViewDevice localPeer, uint32_t rawAivSlot,
                                                uint32_t rowsProcessed, uint32_t groupCount, bool active)
{
    if (rawAivSlot >= kM3NFusedDispatchMaxLaneSlots) {
        return;
    }
    __gm__ int32_t *slot = M3NDispatchLaneSlot(localPeer, rawAivSlot);
    StoreScalarI32(slot + 5U, static_cast<int32_t>(rowsProcessed));
    StoreScalarI32(slot + 6U, static_cast<int32_t>(groupCount));
    StoreScalarI32(slot + 7U, active ? 1 : 0);
    dcci(static_cast<__gm__ void *>(slot), SINGLE_CACHE_LINE);
}

AICORE inline M3NActivationLaneSummary M3NAggregateActivationLaneDebug(M2PeerWindowViewDevice localPeer,
                                                                       uint32_t logicalAivCount)
{
    M3NActivationLaneSummary summary{0U, 0U, 0U};
    uint32_t scanSlots = logicalAivCount;
    if (scanSlots > kM3NFusedDispatchMaxLaneSlots) {
        scanSlots = kM3NFusedDispatchMaxLaneSlots;
    }
    for (uint32_t raw = 0; raw < scanSlots; ++raw) {
        __gm__ int32_t *slot = M3NDispatchLaneSlot(localPeer, raw);
        InvalidateGmCacheLines(slot, 64U);
        int32_t workerPlusOne = LoadScalarI32(slot + 4U);
        if (workerPlusOne <= 0 || LoadScalarI32(slot + 7U) == 0) {
            continue;
        }
        summary.rowsProcessed += static_cast<uint32_t>(LoadScalarI32(slot + 5U));
        ++summary.activeWorkers;
        uint32_t workerId = static_cast<uint32_t>(workerPlusOne - 1);
        if (workerId < 31U) {
            summary.activeWorkerMask |= (1U << workerId);
        }
    }
    return summary;
}
#endif

AICORE inline int32_t M2FusedLoadConfigI32(__gm__ int32_t *config, uint32_t slot)
{
    return *(config + slot);
}

AICORE inline uint64_t M2FusedLoadConfigU64(__gm__ int32_t *config, uint32_t slot)
{
    uint64_t lo = static_cast<uint32_t>(M2FusedLoadConfigI32(config, slot));
    uint64_t hi = static_cast<uint32_t>(M2FusedLoadConfigI32(config, slot + 1U));
    return lo | (hi << 32U);
}

AICORE inline moe_dispatch_combine_a8w8::ShapeConfig M2FusedLoadShapeConfig(__gm__ int32_t *config)
{
    return moe_dispatch_combine_a8w8::ShapeConfig{
        static_cast<uint32_t>(M2FusedLoadConfigI32(config, moe_dispatch_combine_a8w8::kM2FusedFullShapeRankNumSlot)),
        static_cast<uint32_t>(
            M2FusedLoadConfigI32(config, moe_dispatch_combine_a8w8::kM2FusedFullShapeExpertPerRankSlot)),
        static_cast<uint32_t>(M2FusedLoadConfigI32(config, moe_dispatch_combine_a8w8::kM2FusedFullShapeTopKSlot)),
        static_cast<uint32_t>(M2FusedLoadConfigI32(config, moe_dispatch_combine_a8w8::kM2FusedFullShapeMSlot)),
        static_cast<uint32_t>(M2FusedLoadConfigI32(config, moe_dispatch_combine_a8w8::kM2FusedFullShapeHiddenSizeSlot)),
        static_cast<uint32_t>(
            M2FusedLoadConfigI32(config, moe_dispatch_combine_a8w8::kM2FusedFullShapeIntermediateSizeSlot)),
        static_cast<uint32_t>(
            M2FusedLoadConfigI32(config, moe_dispatch_combine_a8w8::kM2FusedFullShapeMaxTokensPerExpertSlot)),
        static_cast<uint32_t>(
            M2FusedLoadConfigI32(config, moe_dispatch_combine_a8w8::kM2FusedFullShapePayloadTileColsSlot)),
        static_cast<uint32_t>(M2FusedLoadConfigI32(config, moe_dispatch_combine_a8w8::kM2FusedFullShapeGmmBlockMSlot)),
        static_cast<uint32_t>(M2FusedLoadConfigI32(config, moe_dispatch_combine_a8w8::kM2FusedFullShapeGmmBlockNSlot)),
        static_cast<uint32_t>(M2FusedLoadConfigI32(config, moe_dispatch_combine_a8w8::kM2FusedFullShapeGmmBlockKSlot)),
        static_cast<uint32_t>(M2FusedLoadConfigI32(config, moe_dispatch_combine_a8w8::kM2FusedFullShapeDtypeInSlot)),
        static_cast<uint32_t>(M2FusedLoadConfigI32(config, moe_dispatch_combine_a8w8::kM2FusedFullShapeDtypeOutSlot)),
    };
}

AICORE inline moe_dispatch_combine_a8w8::RankConfig M2FusedLoadRankConfig(__gm__ int32_t *config)
{
    return moe_dispatch_combine_a8w8::RankConfig{
        static_cast<uint32_t>(M2FusedLoadConfigI32(config, moe_dispatch_combine_a8w8::kM2FusedFullRankRankNumSlot)),
        static_cast<uint32_t>(M2FusedLoadConfigI32(config, moe_dispatch_combine_a8w8::kM2FusedFullRankRankIdSlot)),
        static_cast<uint32_t>(M2FusedLoadConfigI32(config, moe_dispatch_combine_a8w8::kM2FusedFullRankFromMpiSlot)),
        static_cast<uint32_t>(M2FusedLoadConfigI32(config, moe_dispatch_combine_a8w8::kM2FusedFullRankDeviceBaseSlot)),
        static_cast<uint32_t>(M2FusedLoadConfigI32(config, moe_dispatch_combine_a8w8::kM2FusedFullRankNdevicesSlot)),
    };
}

AICORE inline uint32_t M2FusedLoadDebugStopStage(__gm__ moe_dispatch_combine_a8w8::M2FusedFullLaunchArgs *launchArgs)
{
    return launchArgs->debugStopStage;
}

#if defined(M2_MIXED_SPIKE_BUILD_AIV)
AICORE inline void M2FusedBasicVecProbe(__gm__ int32_t *stageStatus, uint32_t slot, int32_t value)
{
    using ProbeTile = pto::Tile<pto::TileType::Vec, int32_t, 1, 16, pto::BLayout::RowMajor, 1, 16>;
    ProbeTile probeTile;
    TASSIGN(probeTile, kM2FusedVecProbeUbAddr);
    TEXPANDS(probeTile, value);
    GlobalNd<int32_t> probeGlobal = MakeGlobal2D(stageStatus + slot, 1, 16, 16);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(probeGlobal, probeTile);
    WaitStoreTileReusable();
}

AICORE inline void M2FusedRouteQuantProbe(moe_dispatch_combine_a8w8::ShapeConfig shape,
                                          M2PeerWindowViewDevice localPeer, GM_ADDR inputA, uint32_t packedRow,
                                          uint32_t rowBytes, __gm__ int32_t *stageStatus, uint32_t probeStage)
{
    if (probeStage == 102301U) {
        M2FusedRecordStage(stageStatus + kM2FusedFullStageBaseSlot, 5U, 102301);
        return;
    }
    __gm__ half *input = reinterpret_cast<__gm__ half *>(inputA);
    M2RouteRowStatTile rowMaxTile;
    M2RouteRowStatTile chunkMaxTile;
    TASSIGN(rowMaxTile, kM2RouteRowMaxTileOffset);
    TASSIGN(chunkMaxTile, kM2RouteChunkMaxTileOffset);
    TEXPANDS(rowMaxTile, 0.0f);
    pipe_barrier(PIPE_V);
    if (probeStage == 102302U) {
        return;
    }
    if (probeStage == 10231U) {
        M2FusedRecordStage(stageStatus + kM2FusedFullStageBaseSlot, 5U, 10231);
        return;
    }

    uint32_t cols = shape.hiddenSize;
    if (cols > kM2RouteQuantTileCols) {
        cols = kM2RouteQuantTileCols;
    }
    M2RouteHalfTile halfTile(cols);
    M2RouteFloatTile fpTile(cols);
    M2RouteFloatTile absTile(cols);
    TASSIGN(halfTile, kM2RouteHalfTileOffset);
    TASSIGN(fpTile, kM2RouteFloatTileOffset);
    TASSIGN(absTile, kM2RouteAbsTileOffset);
    GlobalNd<half> src = MakeGlobal2D(input, 1, static_cast<int32_t>(cols), static_cast<int32_t>(shape.hiddenSize));
    TLOAD(halfTile, src);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    if (probeStage == 10232U) {
        M2FusedRecordStage(stageStatus + kM2FusedFullStageBaseSlot, 5U, 10232);
        return;
    }

    TCVT(fpTile, halfTile, pto::RoundMode::CAST_RINT);
    pipe_barrier(PIPE_V);
    TABS(absTile, fpTile);
    pipe_barrier(PIPE_V);
    TROWMAX(chunkMaxTile, absTile, fpTile);
    pipe_barrier(PIPE_V);
    TMAX(rowMaxTile, rowMaxTile, chunkMaxTile);
    pipe_barrier(PIPE_V);
    if (probeStage == 10233U) {
        M2FusedRecordStage(stageStatus + kM2FusedFullStageBaseSlot, 5U, 10233);
        return;
    }

    pto::PtoSetWaitFlag<PIPE_V, PIPE_S>();
    float maxAbs = rowMaxTile.GetValue(0);
    pto::PtoSetWaitFlag<PIPE_S, PIPE_V>();
    float scale = maxAbs == 0.0f ? 1.0f : maxAbs / 127.0f;
    float invScale = maxAbs == 0.0f ? 1.0f : 1.0f / scale;
    if (probeStage == 10234U) {
        M2FusedRecordStage(stageStatus + kM2FusedFullStageBaseSlot, 5U, 10234);
        return;
    }

    M2RouteScaleStoreTile scaleStoreTile;
    M2RouteParamTile invScaleTile;
    TASSIGN(scaleStoreTile, kM2RouteScaleStoreTileOffset);
    TASSIGN(invScaleTile, kM2RouteInvScaleParamTileOffset);
    TEXPANDS(scaleStoreTile, scale);
    pipe_barrier(PIPE_V);
    TEXPANDS(invScaleTile, invScale);
    pipe_barrier(PIPE_V);
    GlobalNd<float> scaleGlobal = MakeGlobal2D(localPeer.dispatchScale + packedRow, 1, 1, 1);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(scaleGlobal, scaleStoreTile);
    WaitStoreTileReusable();
    if (probeStage == 10235U) {
        M2FusedRecordStage(stageStatus + kM2FusedFullStageBaseSlot, 5U, 10235);
        return;
    }

    M2RouteQuantTile quantTile(cols);
    TASSIGN(quantTile, kM2RouteQuantTileOffset);
    __gm__ int8_t *dst = localPeer.dispatchPayload + static_cast<uint64_t>(packedRow) * rowBytes;
    GlobalNd<int8_t> quantDst = MakeGlobal2D(dst, 1, static_cast<int32_t>(cols), static_cast<int32_t>(rowBytes));
    pto::TQUANT<pto::QuantType::INT8_SYM>(quantTile, fpTile, invScaleTile);
    pipe_barrier(PIPE_V);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(quantDst, quantTile);
    WaitStoreTileReusable();
    M2FusedRecordStage(stageStatus + kM2FusedFullStageBaseSlot, 5U, 10236);
}
#endif

AICORE inline void RunM2FusedSkeletonBody(__gm__ uint64_t *fftsAddr, __gm__ int32_t *ledger)
{
    set_ffts_base_addr(reinterpret_cast<uint64_t>(fftsAddr));
    const int32_t participantIdx = pto::SYNCALL_GET_MIX_PARTICIPANT_IDX();
    const int32_t aicBlocks = *(ledger + kM2MixedSpikeParamAicBlocks);
    const bool isAic = participantIdx < aicBlocks;
    const bool inRange = participantIdx >= 0 && participantIdx < static_cast<int32_t>(kM2FusedSkeletonParticipantCount);
    for (uint32_t stage = 0; stage < kM2FusedSkeletonStageCount; ++stage) {
        if (inRange) {
            uint32_t base = isAic ? kM2FusedSkeletonAicStageBase : kM2FusedSkeletonAivStageBase;
            uint32_t slot = (stage * kM2FusedSkeletonParticipantCount + static_cast<uint32_t>(participantIdx)) *
                            kM2FusedSkeletonStageSlotWords;
            StoreI32Line(ledger + base + slot,
                         static_cast<int32_t>((stage + 1U) * 100U + static_cast<uint32_t>(participantIdx)),
                         kM2FusedSkeletonStoreUbAddr, kM2FusedSkeletonStoreL1Addr);
        }
        pto::SYNCALL<pto::SyncCoreType::Mix>();
    }
}

} // namespace

#if defined(M2_MIXED_SPIKE_BUILD_AIC)
PTO_SYNCALL_MIX_AIC_KERNEL_META(M2MixedSpike_2801_mix_aic, 1, 2);
PTO_SYNCALL_MIX_AIC_KERNEL_META(M2FusedSkeleton_2802_mix_aic, 1, 2);
PTO_SYNCALL_MIX_AIC_KERNEL_META(M2FusedFull_2803_mix_aic, 1, 2);

extern "C" __global__ AICORE void M2MixedSpike_2801_mix_aic(__gm__ int32_t *heartbeat)
{
    uint32_t blockId = static_cast<uint32_t>(get_block_idx());
    StoreI32(heartbeat + kM2MixedSpikeAicHeader + 0U, kM2MixedSpikeMagic);
    StoreI32(heartbeat + kM2MixedSpikeAicHeader + 1U, *(heartbeat + kM2MixedSpikeParamRank));
    StoreI32(heartbeat + kM2MixedSpikeAicHeader + 2U, *(heartbeat + kM2MixedSpikeParamAicBlocks));
    StoreI32(heartbeat + kM2MixedSpikeAicHeader + 3U, *(heartbeat + kM2MixedSpikeParamAivRatio));
    StoreI32(heartbeat + kM2MixedSpikeAicHeader + 4U, 1);
    StoreI32(heartbeat + kM2MixedSpikeAicHeader + 5U, static_cast<int32_t>(get_block_num()));
    if (blockId < kM2MixedSpikeMaxAicSlots) {
        StoreI32(heartbeat + kM2MixedSpikeAicBase + blockId, 1000 + static_cast<int32_t>(blockId));
    }
    pipe_barrier(PIPE_ALL);
    dsb(DSB_DDR);
}

extern "C" __global__ AICORE void M2FusedSkeleton_2802_mix_aic(__gm__ uint64_t *fftsAddr, __gm__ int32_t *ledger)
{
    StoreI32(ledger + kM2FusedSkeletonAicHeader + 0U, kM2FusedSkeletonMagic);
    StoreI32(ledger + kM2FusedSkeletonAicHeader + 1U, *(ledger + kM2MixedSpikeParamRank));
    StoreI32(ledger + kM2FusedSkeletonAicHeader + 2U, *(ledger + kM2MixedSpikeParamAicBlocks));
    StoreI32(ledger + kM2FusedSkeletonAicHeader + 3U, *(ledger + kM2MixedSpikeParamAivRatio));
    StoreI32(ledger + kM2FusedSkeletonAicHeader + 4U, static_cast<int32_t>(kM2FusedSkeletonStageCount));
    StoreI32(ledger + kM2FusedSkeletonAicHeader + 5U, static_cast<int32_t>(get_block_num()));
    RunM2FusedSkeletonBody(fftsAddr, ledger);
    pipe_barrier(PIPE_ALL);
    dsb(DSB_DDR);
}

extern "C" __global__ AICORE void M2FusedFull_2803_mix_aic(
    __gm__ uint64_t *fftsAddr, __gm__ moe_dispatch_combine_a8w8::M2FusedFullLaunchArgs *launchArgs)
{
    uint32_t earlyDebugStopStage = launchArgs->debugStopStage;
    __gm__ int32_t *earlyStageStatus = reinterpret_cast<__gm__ int32_t *>(launchArgs->stageStatusAddr);
    if (earlyDebugStopStage == 99U) {
        set_ffts_base_addr(reinterpret_cast<uint64_t>(fftsAddr));
        if (get_block_idx() == 0) {
            M2FusedRecordStage(earlyStageStatus + kM2FusedFullAicHeaderSlot, 0U, kM2FusedFullMagic);
            M2FusedRecordStage(earlyStageStatus + kM2FusedFullAicHeaderSlot, 1U, static_cast<int32_t>(get_block_num()));
        }
        pto::SYNCALL<pto::SyncCoreType::Mix>();
        return;
    }
    if (earlyDebugStopStage == 96U) {
        set_ffts_base_addr(reinterpret_cast<uint64_t>(fftsAddr));
        if (get_block_idx() == 0) {
            M2FusedRecordStage(earlyStageStatus + kM2FusedFullAicHeaderSlot, 0U, kM2FusedFullMagic);
            M2FusedRecordStage(earlyStageStatus + kM2FusedFullAicHeaderSlot, 1U, static_cast<int32_t>(get_block_num()));
            M2FusedRecordStage(earlyStageStatus + kM2FusedFullAicHeaderSlot, 2U, 96);
            M2FusedRecordStage(earlyStageStatus + kM2FusedFullAicHeaderSlot, 3U, 96);
        }
        pto::SYNCALL<pto::SyncCoreType::Mix>();
        return;
    }
    if (earlyDebugStopStage == 97U) {
        set_ffts_base_addr(reinterpret_cast<uint64_t>(fftsAddr));
        uint32_t shapeRankNum = launchArgs->shapeRankNum;
        if (get_block_idx() == 0) {
            M2FusedRecordStage(earlyStageStatus + kM2FusedFullAicHeaderSlot, 0U, kM2FusedFullMagic);
            M2FusedRecordStage(earlyStageStatus + kM2FusedFullAicHeaderSlot, 1U, static_cast<int32_t>(get_block_num()));
            M2FusedRecordStage(earlyStageStatus + kM2FusedFullAicHeaderSlot, 2U, 97);
            M2FusedRecordStage(earlyStageStatus + kM2FusedFullAicHeaderSlot, 3U, static_cast<int32_t>(shapeRankNum));
        }
        pto::SYNCALL<pto::SyncCoreType::Mix>();
        return;
    }
    if (earlyDebugStopStage == 92U) {
        set_ffts_base_addr(reinterpret_cast<uint64_t>(fftsAddr));
        int32_t rankNum =
            M2FusedLoadConfigI32(earlyStageStatus, moe_dispatch_combine_a8w8::kM2FusedFullShapeRankNumSlot);
        int32_t hidden =
            M2FusedLoadConfigI32(earlyStageStatus, moe_dispatch_combine_a8w8::kM2FusedFullShapeHiddenSizeSlot);
        int32_t intermediate =
            M2FusedLoadConfigI32(earlyStageStatus, moe_dispatch_combine_a8w8::kM2FusedFullShapeIntermediateSizeSlot);
        int32_t blockK =
            M2FusedLoadConfigI32(earlyStageStatus, moe_dispatch_combine_a8w8::kM2FusedFullShapeGmmBlockKSlot);
        if (get_block_idx() == 0) {
            M2FusedRecordStage(earlyStageStatus + kM2FusedFullAicHeaderSlot, 0U, kM2FusedFullMagic);
            M2FusedRecordStage(earlyStageStatus + kM2FusedFullAicHeaderSlot, 1U, static_cast<int32_t>(get_block_num()));
            M2FusedRecordStage(earlyStageStatus + kM2FusedFullAicHeaderSlot, 2U, rankNum);
            M2FusedRecordStage(earlyStageStatus + kM2FusedFullAicHeaderSlot, 3U, hidden + intermediate + blockK);
        }
        pto::SYNCALL<pto::SyncCoreType::Mix>();
        return;
    }
    if (earlyDebugStopStage == 91U) {
        set_ffts_base_addr(reinterpret_cast<uint64_t>(fftsAddr));
        int32_t sum = 0;
        for (uint32_t slot = moe_dispatch_combine_a8w8::kM2FusedFullShapeRankNumSlot;
             slot <= moe_dispatch_combine_a8w8::kM2FusedFullShapeDtypeOutSlot; ++slot) {
            sum += M2FusedLoadConfigI32(earlyStageStatus, slot);
        }
        if (get_block_idx() == 0) {
            M2FusedRecordStage(earlyStageStatus + kM2FusedFullAicHeaderSlot, 0U, kM2FusedFullMagic);
            M2FusedRecordStage(earlyStageStatus + kM2FusedFullAicHeaderSlot, 1U, static_cast<int32_t>(get_block_num()));
            M2FusedRecordStage(earlyStageStatus + kM2FusedFullAicHeaderSlot, 2U, 91);
            M2FusedRecordStage(earlyStageStatus + kM2FusedFullAicHeaderSlot, 3U, sum);
        }
        pto::SYNCALL<pto::SyncCoreType::Mix>();
        return;
    }
    if (earlyDebugStopStage == 93U) {
        set_ffts_base_addr(reinterpret_cast<uint64_t>(fftsAddr));
        if (get_block_idx() == 0) {
            moe_dispatch_combine_a8w8::ShapeConfig shape = M2FusedLoadShapeConfig(earlyStageStatus);
            M2FusedRecordStage(earlyStageStatus + kM2FusedFullAicHeaderSlot, 0U, kM2FusedFullMagic);
            M2FusedRecordStage(earlyStageStatus + kM2FusedFullAicHeaderSlot, 1U, static_cast<int32_t>(get_block_num()));
            M2FusedRecordStage(earlyStageStatus + kM2FusedFullAicHeaderSlot, 2U, static_cast<int32_t>(shape.rankNum));
            M2FusedRecordStage(earlyStageStatus + kM2FusedFullAicHeaderSlot, 3U,
                               static_cast<int32_t>(shape.hiddenSize + shape.intermediateSize));
        }
        pto::SYNCALL<pto::SyncCoreType::Mix>();
        return;
    }
    if (earlyDebugStopStage == 95U) {
        set_ffts_base_addr(reinterpret_cast<uint64_t>(fftsAddr));
        moe_dispatch_combine_a8w8::ShapeConfig shape = M2FusedLoadShapeConfig(earlyStageStatus);
        if (get_block_idx() == 0) {
            M2FusedRecordStage(earlyStageStatus + kM2FusedFullAicHeaderSlot, 0U, kM2FusedFullMagic);
            M2FusedRecordStage(earlyStageStatus + kM2FusedFullAicHeaderSlot, 1U, static_cast<int32_t>(get_block_num()));
            M2FusedRecordStage(earlyStageStatus + kM2FusedFullAicHeaderSlot, 2U, static_cast<int32_t>(shape.rankNum));
            M2FusedRecordStage(earlyStageStatus + kM2FusedFullAicHeaderSlot, 3U,
                               static_cast<int32_t>(shape.hiddenSize + shape.intermediateSize));
        }
        pto::SYNCALL<pto::SyncCoreType::Mix>();
        return;
    }
    if (earlyDebugStopStage == 94U) {
        set_ffts_base_addr(reinterpret_cast<uint64_t>(fftsAddr));
        moe_dispatch_combine_a8w8::RankConfig rank = M2FusedLoadRankConfig(earlyStageStatus);
        if (get_block_idx() == 0) {
            M2FusedRecordStage(earlyStageStatus + kM2FusedFullAicHeaderSlot, 0U, kM2FusedFullMagic);
            M2FusedRecordStage(earlyStageStatus + kM2FusedFullAicHeaderSlot, 1U, static_cast<int32_t>(get_block_num()));
            M2FusedRecordStage(earlyStageStatus + kM2FusedFullAicHeaderSlot, 2U, static_cast<int32_t>(rank.rankId));
            M2FusedRecordStage(earlyStageStatus + kM2FusedFullAicHeaderSlot, 3U, static_cast<int32_t>(rank.rankNum));
        }
        pto::SYNCALL<pto::SyncCoreType::Mix>();
        return;
    }
    if (earlyDebugStopStage == 98U) {
        set_ffts_base_addr(reinterpret_cast<uint64_t>(fftsAddr));
        __gm__ int32_t *stageStatus = reinterpret_cast<__gm__ int32_t *>(launchArgs->stageStatusAddr);
        moe_dispatch_combine_a8w8::ShapeConfig shape = M2FusedLoadShapeConfig(stageStatus);
        moe_dispatch_combine_a8w8::RankConfig rank = M2FusedLoadRankConfig(stageStatus);
        if (get_block_idx() == 0) {
            M2FusedRecordStage(stageStatus + kM2FusedFullAicHeaderSlot, 0U, kM2FusedFullMagic);
            M2FusedRecordStage(stageStatus + kM2FusedFullAicHeaderSlot, 1U, static_cast<int32_t>(get_block_num()));
            M2FusedRecordStage(stageStatus + kM2FusedFullAicHeaderSlot, 2U, static_cast<int32_t>(rank.rankId));
            M2FusedRecordStage(stageStatus + kM2FusedFullAicHeaderSlot, 3U, static_cast<int32_t>(shape.rankNum));
        }
        pto::SYNCALL<pto::SyncCoreType::Mix>();
        return;
    }
    if (earlyDebugStopStage == 89U) {
        set_ffts_base_addr(reinterpret_cast<uint64_t>(fftsAddr));
        pto::SYNCALL<pto::SyncCoreType::Mix>();
        return;
    }
    set_ffts_base_addr(reinterpret_cast<uint64_t>(fftsAddr));
    __gm__ int32_t *stageStatus = reinterpret_cast<__gm__ int32_t *>(launchArgs->stageStatusAddr);
    moe_dispatch_combine_a8w8::ShapeConfig shape = M2FusedLoadShapeConfig(stageStatus);
    moe_dispatch_combine_a8w8::RankConfig rank = M2FusedLoadRankConfig(stageStatus);
    GM_ADDR workspace = reinterpret_cast<GM_ADDR>(
        M2FusedLoadConfigU64(stageStatus, moe_dispatch_combine_a8w8::kM2FusedFullPtrWorkspaceSlot));
    GM_ADDR peerWindow = reinterpret_cast<GM_ADDR>(
        M2FusedLoadConfigU64(stageStatus, moe_dispatch_combine_a8w8::kM2FusedFullPtrPeerWindowSlot));
    auto layout = M2FusedGmmMakeWorkspaceLayoutDevice(shape);
    __gm__ int32_t *debugCounters = M3NFusedPeerDebugCounters(peerWindow, shape);
    uint32_t debugStopStage = static_cast<uint32_t>(M2FusedGmmLoadScalarI32(stageStatus + kM2FusedFullDebugStopSlot));
    if (debugStopStage == 0U) {
        debugStopStage = M2FusedLoadDebugStopStage(launchArgs);
    }
    uint32_t overlapMode = static_cast<uint32_t>(
        M2FusedLoadConfigI32(stageStatus, moe_dispatch_combine_a8w8::kM2FusedFullOverlapModeSlot));
    if (get_block_idx() == 0) {
        M2FusedRecordStage(stageStatus + kM2FusedFullAicHeaderSlot, 0U, kM2FusedFullMagic);
        M2FusedRecordStage(stageStatus + kM2FusedFullAicHeaderSlot, 1U, static_cast<int32_t>(get_block_num()));
        M2FusedRecordStage(stageStatus + kM2FusedFullAicHeaderSlot, 2U, static_cast<int32_t>(rank.rankId));
        M2FusedRecordStage(stageStatus + kM2FusedFullAicHeaderSlot, 3U, static_cast<int32_t>(shape.rankNum));
        M2FusedRecordStage(stageStatus + kM2FusedFullM3N5AicEvidenceSlot, 0U, 0);
        M2FusedRecordStage(stageStatus + kM2FusedFullM3N5AicEvidenceSlot + 1U, 0U, 0);
        M2FusedRecordStage(stageStatus + kM2FusedFullM3N6EvidenceSlot + 0U, 0U, 0);
        M2FusedRecordStage(stageStatus + kM2FusedFullM3N6EvidenceSlot + 1U, 0U, 0);
        M2FusedRecordStage(stageStatus + kM2FusedFullM3N6EvidenceSlot + 2U, 0U, 0);
        M2FusedRecordStage(stageStatus + kM2FusedFullM3N7EvidenceSlot + 0U, 0U, 0);
        M2FusedRecordStage(stageStatus + kM2FusedFullM3N7EvidenceSlot + 1U, 0U, 0);
        M2FusedRecordStage(stageStatus + kM2FusedFullM3N7EvidenceSlot + 2U, 0U, 0);
        M2FusedRecordStage(stageStatus + kM2FusedFullM3N8EvidenceSlot + 0U, 0U, 0);
        M2FusedRecordStage(stageStatus + kM2FusedFullM3N8EvidenceSlot + 1U, 0U, 0);
        M2FusedRecordStage(stageStatus + kM2FusedFullM3N8EvidenceSlot + 2U, 0U, 0);
        pipe_barrier(PIPE_ALL);
        dsb(DSB_DDR);
    }
    bool m3n5DispatchGmm1Overlap = M3N5DispatchGmm1OverlapEnabled(debugStopStage, overlapMode, shape);
    bool m3n6Gmm1ActivationOverlap = M3N6Gmm1ActivationOverlapEnabled(debugStopStage, overlapMode, shape);
    bool m3n7ActivationGmm2Overlap = M3N7ActivationGmm2OverlapEnabled(debugStopStage, overlapMode, shape);
    bool m3n8Gmm2CombineOverlap = M3N8Gmm2CombineOverlapEnabled(debugStopStage, overlapMode, shape);
    M3NDispatchAicWaitForAivSubphases(debugStopStage, m3n5DispatchGmm1Overlap, shape);
    if (debugStopStage >= 100U) {
        pto::SYNCALL<pto::SyncCoreType::Mix>();
        return;
    }
    if (!m3n5DispatchGmm1Overlap) {
        moe_dispatch_combine_a8w8::M3N4WaitV2C<moe_dispatch_combine_a8w8::kM3N4DispatchToGmm1Flag>();
    }
    if (debugStopStage == 1U) {
        if (m3n5DispatchGmm1Overlap) {
            M3N5WaitAllDispatchExpertsReady(shape);
        }
        return;
    }
    M2FusedRecordStage(stageStatus + kM2FusedFullAicStageBaseSlot, static_cast<uint32_t>(get_block_idx()), 100);
    if (shape.rankNum != 0 && shape.expertPerRank != 0 && shape.hiddenSize != 0U && shape.intermediateSize != 0U &&
        shape.gmmBlockM == moe_dispatch_combine_a8w8::kGmmBaseM &&
        shape.gmmBlockN == moe_dispatch_combine_a8w8::kGmmBaseN &&
        shape.gmmBlockK == moe_dispatch_combine_a8w8::kGmmBaseK &&
        shape.hiddenSize % moe_dispatch_combine_a8w8::kGmmBaseK == 0U &&
        shape.intermediateSize % moe_dispatch_combine_a8w8::kGmmBaseK == 0U) {
        uint32_t rowBytes = static_cast<uint32_t>(M2FusedGmmDispatchPayloadRowBytes(shape));
        uint32_t w1Cols = shape.intermediateSize * 2U;
        __gm__ int8_t *gmm1Input = reinterpret_cast<__gm__ int8_t *>(workspace + layout.gmm1InputInt8.offset);
        __gm__ int8_t *weight1 = reinterpret_cast<__gm__ int8_t *>(workspace + layout.gmm1WeightInt8.offset);
        __gm__ int32_t *gmm1Acc = reinterpret_cast<__gm__ int32_t *>(workspace + layout.gmm1AccInt32.offset);
        __gm__ int32_t *dispatchOffset = reinterpret_cast<__gm__ int32_t *>(workspace + layout.dispatchOffset.offset);
        __gm__ int32_t *expertTokenNums = reinterpret_cast<__gm__ int32_t *>(workspace + layout.expertTokenNums.offset);
        __gm__ int32_t *gmm1TileTaskPlan =
            reinterpret_cast<__gm__ int32_t *>(workspace + layout.gmm1TileTaskPlan.offset);
        uint32_t taskCount = M2FusedGmmBuildTileTaskPlan(shape, dispatchOffset, expertTokenNums, gmm1TileTaskPlan,
                                                         w1Cols, shape.hiddenSize, 1U);
        if (get_block_idx() == 0) {
            M2FusedGmmStoreScalarI32(stageStatus + 4U * 16U, static_cast<int32_t>(taskCount));
            M2FusedGmmStoreScalarI32(stageStatus + 5U * 16U, static_cast<int32_t>(get_block_num()));
        }
        if (m3n6Gmm1ActivationOverlap) {
            uint32_t syncGroupCount = moe_dispatch_combine_a8w8::M3N6SwigluSyncGroupCount(shape);
            for (uint32_t syncIdx = 0; syncIdx < syncGroupCount; ++syncIdx) {
                uint32_t expertBegin = 0;
                uint32_t expertEnd = 0;
                M3N6SyncGroupExpertRange(shape, syncIdx, &expertBegin, &expertEnd);
                for (uint32_t currentExpert = expertBegin; currentExpert < expertEnd; ++currentExpert) {
                    if (m3n5DispatchGmm1Overlap) {
                        moe_dispatch_combine_a8w8::M3N5WaitDispatchExpertReady(currentExpert);
                    }
                    M3N5RecordGmm1StartBeforeLastDispatchReady(shape, stageStatus, currentExpert);
                    for (uint32_t taskId = static_cast<uint32_t>(get_block_idx()); taskId < taskCount;
                         taskId += static_cast<uint32_t>(get_block_num())) {
                        __gm__ int32_t *task = gmm1TileTaskPlan + taskId * kM2GmmTileTaskFields;
                        uint32_t localExpert = static_cast<uint32_t>(M2FusedGmmLoadScalarI32(task + 2U));
                        if (localExpert != currentExpert) {
                            continue;
                        }
                        uint32_t globalExpert = rank.rankId * shape.expertPerRank + localExpert;
                        uint32_t rowBegin = static_cast<uint32_t>(M2FusedGmmLoadScalarI32(task + 3U));
                        uint32_t mValid = static_cast<uint32_t>(M2FusedGmmLoadScalarI32(task + 4U));
                        uint32_t nBase = static_cast<uint32_t>(M2FusedGmmLoadScalarI32(task + 5U));
                        uint32_t nValid = static_cast<uint32_t>(M2FusedGmmLoadScalarI32(task + 6U));
                        __gm__ int8_t *expertWeight =
                            weight1 + static_cast<uint64_t>(globalExpert) * shape.hiddenSize * w1Cols;
                        __gm__ int8_t *tileInput = gmm1Input + static_cast<uint64_t>(rowBegin) * rowBytes;
                        __gm__ int32_t *tileOutput = gmm1Acc + static_cast<uint64_t>(rowBegin) * w1Cols;
                        M2FusedRunInt8GmmTile(tileInput, expertWeight + nBase, tileOutput + nBase, mValid,
                                              shape.hiddenSize, nValid, rowBytes, w1Cols, w1Cols);
                    }
                }
                moe_dispatch_combine_a8w8::M3N4AicAllDoneCoarseSync();
                if (get_block_idx() == 0) {
                    M3N6RecordGmm1SyncGroupReady(stageStatus, layout, workspace, syncIdx, syncGroupCount);
                }
                moe_dispatch_combine_a8w8::M3N6SignalGmm1SyncGroupReady(syncIdx);
            }
        } else if (m3n5DispatchGmm1Overlap) {
            for (uint32_t currentExpert = 0; currentExpert < shape.expertPerRank; ++currentExpert) {
                moe_dispatch_combine_a8w8::M3N5WaitDispatchExpertReady(currentExpert);
                M3N5RecordGmm1StartBeforeLastDispatchReady(shape, stageStatus, currentExpert);
                for (uint32_t taskId = static_cast<uint32_t>(get_block_idx()); taskId < taskCount;
                     taskId += static_cast<uint32_t>(get_block_num())) {
                    __gm__ int32_t *task = gmm1TileTaskPlan + taskId * kM2GmmTileTaskFields;
                    uint32_t localExpert = static_cast<uint32_t>(M2FusedGmmLoadScalarI32(task + 2U));
                    if (localExpert != currentExpert) {
                        continue;
                    }
                    uint32_t globalExpert = rank.rankId * shape.expertPerRank + localExpert;
                    uint32_t rowBegin = static_cast<uint32_t>(M2FusedGmmLoadScalarI32(task + 3U));
                    uint32_t mValid = static_cast<uint32_t>(M2FusedGmmLoadScalarI32(task + 4U));
                    uint32_t nBase = static_cast<uint32_t>(M2FusedGmmLoadScalarI32(task + 5U));
                    uint32_t nValid = static_cast<uint32_t>(M2FusedGmmLoadScalarI32(task + 6U));
                    __gm__ int8_t *expertWeight =
                        weight1 + static_cast<uint64_t>(globalExpert) * shape.hiddenSize * w1Cols;
                    __gm__ int8_t *tileInput = gmm1Input + static_cast<uint64_t>(rowBegin) * rowBytes;
                    __gm__ int32_t *tileOutput = gmm1Acc + static_cast<uint64_t>(rowBegin) * w1Cols;
                    M2FusedRunInt8GmmTile(tileInput, expertWeight + nBase, tileOutput + nBase, mValid, shape.hiddenSize,
                                          nValid, rowBytes, w1Cols, w1Cols);
                }
            }
        } else {
            for (uint32_t taskId = static_cast<uint32_t>(get_block_idx()); taskId < taskCount;
                 taskId += static_cast<uint32_t>(get_block_num())) {
                __gm__ int32_t *task = gmm1TileTaskPlan + taskId * kM2GmmTileTaskFields;
                uint32_t localExpert = static_cast<uint32_t>(M2FusedGmmLoadScalarI32(task + 2U));
                uint32_t globalExpert = rank.rankId * shape.expertPerRank + localExpert;
                uint32_t rowBegin = static_cast<uint32_t>(M2FusedGmmLoadScalarI32(task + 3U));
                uint32_t mValid = static_cast<uint32_t>(M2FusedGmmLoadScalarI32(task + 4U));
                uint32_t nBase = static_cast<uint32_t>(M2FusedGmmLoadScalarI32(task + 5U));
                uint32_t nValid = static_cast<uint32_t>(M2FusedGmmLoadScalarI32(task + 6U));
                __gm__ int8_t *expertWeight = weight1 + static_cast<uint64_t>(globalExpert) * shape.hiddenSize * w1Cols;
                __gm__ int8_t *tileInput = gmm1Input + static_cast<uint64_t>(rowBegin) * rowBytes;
                __gm__ int32_t *tileOutput = gmm1Acc + static_cast<uint64_t>(rowBegin) * w1Cols;
                M2FusedRunInt8GmmTile(tileInput, expertWeight + nBase, tileOutput + nBase, mValid, shape.hiddenSize,
                                      nValid, rowBytes, w1Cols, w1Cols);
            }
        }
    }
    pipe_barrier(PIPE_ALL);
    dsb(DSB_DDR);
    if (!m3n6Gmm1ActivationOverlap) {
        moe_dispatch_combine_a8w8::M3N4AicAllDoneCoarseSync();
        moe_dispatch_combine_a8w8::M3N4SignalC2V<moe_dispatch_combine_a8w8::kM3N4Gmm1ToEpilogueFlag>();
    }
    if (debugStopStage == 2U) {
        return;
    }
    if (!m3n6Gmm1ActivationOverlap) {
        moe_dispatch_combine_a8w8::M3N4WaitV2C<moe_dispatch_combine_a8w8::kM3N4Gmm1EpilogueToActivationFlag>();
    }
    if (debugStopStage == 3U) {
        return;
    }
    if (!m3n6Gmm1ActivationOverlap) {
        M3NActivationAicWaitForAivSubphases(debugStopStage, shape);
    }
    if (!m3n7ActivationGmm2Overlap) {
        moe_dispatch_combine_a8w8::M3N4WaitV2C<moe_dispatch_combine_a8w8::kM3N4ActivationToGmm2Flag>();
    }
    if (debugStopStage == 4U) {
        if (m3n7ActivationGmm2Overlap) {
            __gm__ int32_t *activationSyncGroupReady =
                reinterpret_cast<__gm__ int32_t *>(workspace + layout.activationSyncGroupReady.offset);
            uint32_t syncGroupCount = moe_dispatch_combine_a8w8::M3N6SwigluSyncGroupCount(shape);
            for (uint32_t syncIdx = 0; syncIdx < syncGroupCount; ++syncIdx) {
                M3N7WaitActivationSyncGroupReadyGm(activationSyncGroupReady, syncIdx);
            }
        }
        return;
    }
    M2FusedRecordStage(stageStatus + kM2FusedFullAicStageBaseSlot, static_cast<uint32_t>(get_block_idx()), 200);
    if (shape.rankNum != 0 && shape.expertPerRank != 0 && shape.hiddenSize != 0U && shape.intermediateSize != 0U &&
        shape.gmmBlockM == moe_dispatch_combine_a8w8::kGmmBaseM &&
        shape.gmmBlockN == moe_dispatch_combine_a8w8::kGmmBaseN &&
        shape.gmmBlockK == moe_dispatch_combine_a8w8::kGmmBaseK &&
        shape.hiddenSize % moe_dispatch_combine_a8w8::kGmmBaseK == 0U &&
        shape.intermediateSize % moe_dispatch_combine_a8w8::kGmmBaseK == 0U) {
        uint32_t rowBytes = static_cast<uint32_t>(M2FusedGmmAlign64Device(shape.intermediateSize));
        __gm__ int8_t *gmm2Input = reinterpret_cast<__gm__ int8_t *>(workspace + layout.gmm2InputInt8.offset);
        __gm__ int8_t *weight2 = reinterpret_cast<__gm__ int8_t *>(workspace + layout.gmm2WeightInt8.offset);
        __gm__ int32_t *gmm2Acc = reinterpret_cast<__gm__ int32_t *>(workspace + layout.gmm2AccInt32.offset);
        __gm__ int32_t *dispatchOffset = reinterpret_cast<__gm__ int32_t *>(workspace + layout.dispatchOffset.offset);
        __gm__ int32_t *expertTokenNums = reinterpret_cast<__gm__ int32_t *>(workspace + layout.expertTokenNums.offset);
        __gm__ int32_t *gmm2GroupReady = reinterpret_cast<__gm__ int32_t *>(workspace + layout.gmm2GroupReady.offset);
        __gm__ int32_t *activationSyncGroupReady =
            reinterpret_cast<__gm__ int32_t *>(workspace + layout.activationSyncGroupReady.offset);
        __gm__ int32_t *swigluGroupDesc = reinterpret_cast<__gm__ int32_t *>(workspace + layout.swigluGroupDesc.offset);
        __gm__ int32_t *gmm2TileTaskPlan =
            reinterpret_cast<__gm__ int32_t *>(workspace + layout.gmm2TileTaskPlan.offset);
        uint32_t taskCount = M2FusedGmmBuildTileTaskPlan(shape, dispatchOffset, expertTokenNums, gmm2TileTaskPlan,
                                                         shape.hiddenSize, shape.intermediateSize, 2U);
        if (get_block_idx() == 0) {
            M2FusedGmmStoreScalarI32(stageStatus + 6U * 16U, static_cast<int32_t>(taskCount));
            M2FusedGmmStoreScalarI32(stageStatus + 7U * 16U, static_cast<int32_t>(get_block_num()));
            if (m3n7ActivationGmm2Overlap) {
                M3N7InitGmm2Counters(debugCounters, moe_dispatch_combine_a8w8::M3N6SwigluSyncGroupCount(shape));
            }
        }
        for (uint32_t localExpert = 0; localExpert < shape.expertPerRank; ++localExpert) {
            int32_t rowCount = M2FusedGmmLoadScalarI32(expertTokenNums + localExpert);
            if (rowCount <= 0) {
                M2FusedGmmStoreScalarI32(gmm2GroupReady + localExpert * 16U, 1);
            }
        }
        if (m3n7ActivationGmm2Overlap) {
            uint32_t syncGroupCount = moe_dispatch_combine_a8w8::M3N6SwigluSyncGroupCount(shape);
            for (uint32_t syncIdx = 0; syncIdx < syncGroupCount; ++syncIdx) {
                M3N7WaitActivationSyncGroupReadyGm(activationSyncGroupReady, syncIdx);
                __gm__ int32_t *group = swigluGroupDesc + syncIdx * kM2SwigluGroupFields;
                if (get_block_idx() == 0) {
                    int32_t firstTaskId = -1;
                    int32_t firstTaskExpert = -1;
                    int32_t firstTaskRowBegin = -1;
                    uint32_t intersectTaskCount = M3N7CountGmm2SyncGroupTasks(
                        gmm2TileTaskPlan, taskCount, group, &firstTaskId, &firstTaskExpert, &firstTaskRowBegin);
                    M3N7RecordGmm2SyncGroupConsumed(stageStatus, debugCounters, syncIdx, syncGroupCount, group,
                                                    intersectTaskCount, firstTaskId, firstTaskExpert,
                                                    firstTaskRowBegin);
                }
                for (uint32_t taskId = static_cast<uint32_t>(get_block_idx()); taskId < taskCount;
                     taskId += static_cast<uint32_t>(get_block_num())) {
                    __gm__ int32_t *task = gmm2TileTaskPlan + taskId * kM2GmmTileTaskFields;
                    if (!M3N7Gmm2TaskIntersectsSyncGroup(task, group)) {
                        continue;
                    }
                    uint32_t localExpert = static_cast<uint32_t>(M2FusedGmmLoadScalarI32(task + 2U));
                    uint32_t globalExpert = rank.rankId * shape.expertPerRank + localExpert;
                    uint32_t rowBegin = static_cast<uint32_t>(M2FusedGmmLoadScalarI32(task + 3U));
                    uint32_t mValid = static_cast<uint32_t>(M2FusedGmmLoadScalarI32(task + 4U));
                    uint32_t nBase = static_cast<uint32_t>(M2FusedGmmLoadScalarI32(task + 5U));
                    uint32_t nValid = static_cast<uint32_t>(M2FusedGmmLoadScalarI32(task + 6U));
                    __gm__ int8_t *expertWeight =
                        weight2 + static_cast<uint64_t>(globalExpert) * shape.intermediateSize * shape.hiddenSize;
                    __gm__ int8_t *tileInput = gmm2Input + static_cast<uint64_t>(rowBegin) * rowBytes;
                    __gm__ int32_t *tileOutput = gmm2Acc + static_cast<uint64_t>(rowBegin) * shape.hiddenSize;
                    M2FusedRunInt8GmmTile(tileInput, expertWeight + nBase, tileOutput + nBase, mValid,
                                          shape.intermediateSize, nValid, rowBytes, shape.hiddenSize, shape.hiddenSize);
                }
                moe_dispatch_combine_a8w8::M3N4AicAllDoneCoarseSync();
                if (get_block_idx() == 0) {
                    int32_t expertBeginI32 = M2FusedGmmLoadScalarI32(group + 1U);
                    int32_t expertEndI32 = M2FusedGmmLoadScalarI32(group + 2U);
                    uint32_t expertBegin = expertBeginI32 > 0 ? static_cast<uint32_t>(expertBeginI32) : 0U;
                    uint32_t expertEnd = expertEndI32 > 0 ? static_cast<uint32_t>(expertEndI32) : 0U;
                    if (expertEnd > shape.expertPerRank) {
                        expertEnd = shape.expertPerRank;
                    }
                    for (uint32_t localExpert = expertBegin; localExpert < expertEnd; ++localExpert) {
                        M2FusedGmmStoreScalarI32(gmm2GroupReady + localExpert * 16U, 1);
                    }
                    pipe_barrier(PIPE_ALL);
                    dsb(DSB_DDR);
                }
            }
        } else {
            for (uint32_t taskId = static_cast<uint32_t>(get_block_idx()); taskId < taskCount;
                 taskId += static_cast<uint32_t>(get_block_num())) {
                __gm__ int32_t *task = gmm2TileTaskPlan + taskId * kM2GmmTileTaskFields;
                uint32_t localExpert = static_cast<uint32_t>(M2FusedGmmLoadScalarI32(task + 2U));
                uint32_t globalExpert = rank.rankId * shape.expertPerRank + localExpert;
                uint32_t rowBegin = static_cast<uint32_t>(M2FusedGmmLoadScalarI32(task + 3U));
                uint32_t mValid = static_cast<uint32_t>(M2FusedGmmLoadScalarI32(task + 4U));
                uint32_t nBase = static_cast<uint32_t>(M2FusedGmmLoadScalarI32(task + 5U));
                uint32_t nValid = static_cast<uint32_t>(M2FusedGmmLoadScalarI32(task + 6U));
                __gm__ int8_t *expertWeight =
                    weight2 + static_cast<uint64_t>(globalExpert) * shape.intermediateSize * shape.hiddenSize;
                __gm__ int8_t *tileInput = gmm2Input + static_cast<uint64_t>(rowBegin) * rowBytes;
                __gm__ int32_t *tileOutput = gmm2Acc + static_cast<uint64_t>(rowBegin) * shape.hiddenSize;
                M2FusedRunInt8GmmTile(tileInput, expertWeight + nBase, tileOutput + nBase, mValid,
                                      shape.intermediateSize, nValid, rowBytes, shape.hiddenSize, shape.hiddenSize);
            }
        }
    }
    pipe_barrier(PIPE_ALL);
    dsb(DSB_DDR);
    moe_dispatch_combine_a8w8::M3N4AicAllDoneCoarseSync();
    if (get_block_idx() == 0 && !m3n7ActivationGmm2Overlap && shape.rankNum != 0 && shape.expertPerRank != 0 &&
        shape.hiddenSize != 0U && shape.intermediateSize != 0U &&
        shape.gmmBlockM == moe_dispatch_combine_a8w8::kGmmBaseM &&
        shape.gmmBlockN == moe_dispatch_combine_a8w8::kGmmBaseN &&
        shape.gmmBlockK == moe_dispatch_combine_a8w8::kGmmBaseK &&
        shape.hiddenSize % moe_dispatch_combine_a8w8::kGmmBaseK == 0U &&
        shape.intermediateSize % moe_dispatch_combine_a8w8::kGmmBaseK == 0U) {
        __gm__ int32_t *gmm2GroupReady = reinterpret_cast<__gm__ int32_t *>(workspace + layout.gmm2GroupReady.offset);
        for (uint32_t localExpert = 0; localExpert < shape.expertPerRank; ++localExpert) {
            M2FusedGmmStoreScalarI32(gmm2GroupReady + localExpert * 16U, 1);
        }
    }
    pipe_barrier(PIPE_ALL);
    dsb(DSB_DDR);
    if (!m3n8Gmm2CombineOverlap) {
        moe_dispatch_combine_a8w8::M3N4SignalC2V<moe_dispatch_combine_a8w8::kM3N4Gmm2ToCombineFlag>();
    }
    if (debugStopStage == 5U) {
        return;
    }
    if (!m3n8Gmm2CombineOverlap && M3NCombineAicWaitForAivSubphases(debugStopStage, shape)) {
        return;
    }
    if (m3n8Gmm2CombineOverlap && (debugStopStage == 51U || debugStopStage == 52U || debugStopStage == 53U ||
                                   debugStopStage == 54U || debugStopStage == 56U)) {
        return;
    }
    moe_dispatch_combine_a8w8::M3N4WaitV2C<moe_dispatch_combine_a8w8::kM3N4CombineToRestoreFlag>();
    if (debugStopStage == 6U) {
        return;
    }
}
#endif

#if defined(M2_MIXED_SPIKE_BUILD_AIV)
PTO_SYNCALL_MIX_AIC_KERNEL_META(M2MixedSpike_2801_mix_aiv, 1, 2);
PTO_SYNCALL_MIX_AIC_KERNEL_META(M2FusedSkeleton_2802_mix_aiv, 1, 2);
PTO_SYNCALL_MIX_AIC_KERNEL_META(M2FusedFull_2803_mix_aiv, 1, 2);

extern "C" __global__ AICORE void M2MixedSpike_2801_mix_aiv(__gm__ int32_t *heartbeat)
{
    uint32_t blockId = static_cast<uint32_t>(get_block_idx());
    uint32_t subblockId = static_cast<uint32_t>(get_subblockid());
    uint32_t subblockDim = static_cast<uint32_t>(get_subblockdim());
    uint32_t logicalId = blockId * subblockDim + subblockId;
    StoreI32(heartbeat + kM2MixedSpikeAivHeader + 0U, kM2MixedSpikeMagic);
    StoreI32(heartbeat + kM2MixedSpikeAivHeader + 1U, *(heartbeat + kM2MixedSpikeParamRank));
    StoreI32(heartbeat + kM2MixedSpikeAivHeader + 2U, *(heartbeat + kM2MixedSpikeParamAicBlocks));
    StoreI32(heartbeat + kM2MixedSpikeAivHeader + 3U, *(heartbeat + kM2MixedSpikeParamAivRatio));
    StoreI32(heartbeat + kM2MixedSpikeAivHeader + 4U, 1);
    StoreI32(heartbeat + kM2MixedSpikeAivHeader + 5U, static_cast<int32_t>(get_block_num() * get_subblockdim()));
    if (logicalId < kM2MixedSpikeMaxAivSlots) {
        StoreI32(heartbeat + kM2MixedSpikeAivBase + logicalId, 2000 + static_cast<int32_t>(logicalId));
    }
    pipe_barrier(PIPE_ALL);
    dsb(DSB_DDR);
}

extern "C" __global__ AICORE void M2FusedSkeleton_2802_mix_aiv(__gm__ uint64_t *fftsAddr, __gm__ int32_t *ledger)
{
    StoreI32(ledger + kM2FusedSkeletonAivHeader + 0U, kM2FusedSkeletonMagic);
    StoreI32(ledger + kM2FusedSkeletonAivHeader + 1U, *(ledger + kM2MixedSpikeParamRank));
    StoreI32(ledger + kM2FusedSkeletonAivHeader + 2U, *(ledger + kM2MixedSpikeParamAicBlocks));
    StoreI32(ledger + kM2FusedSkeletonAivHeader + 3U, *(ledger + kM2MixedSpikeParamAivRatio));
    StoreI32(ledger + kM2FusedSkeletonAivHeader + 4U, static_cast<int32_t>(kM2FusedSkeletonStageCount));
    StoreI32(ledger + kM2FusedSkeletonAivHeader + 5U, static_cast<int32_t>(get_block_num() * get_subblockdim()));
    RunM2FusedSkeletonBody(fftsAddr, ledger);
    pipe_barrier(PIPE_ALL);
    dsb(DSB_DDR);
}

extern "C" __global__ AICORE void M2FusedFull_2803_mix_aiv(
    __gm__ uint64_t *fftsAddr, __gm__ moe_dispatch_combine_a8w8::M2FusedFullLaunchArgs *launchArgs)
{
    uint32_t earlyDebugStopStage = launchArgs->debugStopStage;
    __gm__ int32_t *earlyStageStatus = reinterpret_cast<__gm__ int32_t *>(launchArgs->stageStatusAddr);
    if (earlyDebugStopStage == 99U) {
        set_ffts_base_addr(reinterpret_cast<uint64_t>(fftsAddr));
        if (IsM2FusedMainAiv()) {
            M2FusedRecordStage(earlyStageStatus + kM2FusedFullAivHeaderSlot, 0U, kM2FusedFullMagic);
            M2FusedRecordStage(earlyStageStatus + kM2FusedFullAivHeaderSlot, 1U,
                               static_cast<int32_t>(get_block_num() * get_subblockdim()));
            M2FusedRecordStage(earlyStageStatus + kM2FusedFullStageBaseSlot, 5U, 99);
        }
        pto::SYNCALL<pto::SyncCoreType::Mix>();
        return;
    }
    if (earlyDebugStopStage == 96U) {
        set_ffts_base_addr(reinterpret_cast<uint64_t>(fftsAddr));
        if (IsM2FusedMainAiv()) {
            M2FusedRecordStage(earlyStageStatus + kM2FusedFullAivHeaderSlot, 0U, kM2FusedFullMagic);
            M2FusedRecordStage(earlyStageStatus + kM2FusedFullAivHeaderSlot, 1U,
                               static_cast<int32_t>(get_block_num() * get_subblockdim()));
            M2FusedRecordStage(earlyStageStatus + kM2FusedFullAivHeaderSlot, 2U, 96);
            M2FusedRecordStage(earlyStageStatus + kM2FusedFullAivHeaderSlot, 3U, 96);
            M2FusedRecordStage(earlyStageStatus + kM2FusedFullStageBaseSlot, 5U, 96);
        }
        pto::SYNCALL<pto::SyncCoreType::Mix>();
        return;
    }
    if (earlyDebugStopStage == 97U) {
        set_ffts_base_addr(reinterpret_cast<uint64_t>(fftsAddr));
        uint32_t shapeRankNum = launchArgs->shapeRankNum;
        if (IsM2FusedMainAiv()) {
            M2FusedRecordStage(earlyStageStatus + kM2FusedFullAivHeaderSlot, 0U, kM2FusedFullMagic);
            M2FusedRecordStage(earlyStageStatus + kM2FusedFullAivHeaderSlot, 1U,
                               static_cast<int32_t>(get_block_num() * get_subblockdim()));
            M2FusedRecordStage(earlyStageStatus + kM2FusedFullAivHeaderSlot, 2U, 97);
            M2FusedRecordStage(earlyStageStatus + kM2FusedFullAivHeaderSlot, 3U, static_cast<int32_t>(shapeRankNum));
            M2FusedRecordStage(earlyStageStatus + kM2FusedFullStageBaseSlot, 5U, 97);
        }
        pto::SYNCALL<pto::SyncCoreType::Mix>();
        return;
    }
    if (earlyDebugStopStage == 92U) {
        set_ffts_base_addr(reinterpret_cast<uint64_t>(fftsAddr));
        int32_t rankNum =
            M2FusedLoadConfigI32(earlyStageStatus, moe_dispatch_combine_a8w8::kM2FusedFullShapeRankNumSlot);
        int32_t hidden =
            M2FusedLoadConfigI32(earlyStageStatus, moe_dispatch_combine_a8w8::kM2FusedFullShapeHiddenSizeSlot);
        int32_t intermediate =
            M2FusedLoadConfigI32(earlyStageStatus, moe_dispatch_combine_a8w8::kM2FusedFullShapeIntermediateSizeSlot);
        int32_t blockK =
            M2FusedLoadConfigI32(earlyStageStatus, moe_dispatch_combine_a8w8::kM2FusedFullShapeGmmBlockKSlot);
        if (IsM2FusedMainAiv()) {
            M2FusedRecordStage(earlyStageStatus + kM2FusedFullAivHeaderSlot, 0U, kM2FusedFullMagic);
            M2FusedRecordStage(earlyStageStatus + kM2FusedFullAivHeaderSlot, 1U,
                               static_cast<int32_t>(get_block_num() * get_subblockdim()));
            M2FusedRecordStage(earlyStageStatus + kM2FusedFullAivHeaderSlot, 2U, rankNum);
            M2FusedRecordStage(earlyStageStatus + kM2FusedFullAivHeaderSlot, 3U, hidden + intermediate + blockK);
            M2FusedRecordStage(earlyStageStatus + kM2FusedFullStageBaseSlot, 5U, 92);
        }
        pto::SYNCALL<pto::SyncCoreType::Mix>();
        return;
    }
    if (earlyDebugStopStage == 91U) {
        set_ffts_base_addr(reinterpret_cast<uint64_t>(fftsAddr));
        int32_t sum = 0;
        for (uint32_t slot = moe_dispatch_combine_a8w8::kM2FusedFullShapeRankNumSlot;
             slot <= moe_dispatch_combine_a8w8::kM2FusedFullShapeDtypeOutSlot; ++slot) {
            sum += M2FusedLoadConfigI32(earlyStageStatus, slot);
        }
        if (IsM2FusedMainAiv()) {
            M2FusedRecordStage(earlyStageStatus + kM2FusedFullAivHeaderSlot, 0U, kM2FusedFullMagic);
            M2FusedRecordStage(earlyStageStatus + kM2FusedFullAivHeaderSlot, 1U,
                               static_cast<int32_t>(get_block_num() * get_subblockdim()));
            M2FusedRecordStage(earlyStageStatus + kM2FusedFullAivHeaderSlot, 2U, 91);
            M2FusedRecordStage(earlyStageStatus + kM2FusedFullAivHeaderSlot, 3U, sum);
            M2FusedRecordStage(earlyStageStatus + kM2FusedFullStageBaseSlot, 5U, 91);
        }
        pto::SYNCALL<pto::SyncCoreType::Mix>();
        return;
    }
    if (earlyDebugStopStage == 93U) {
        set_ffts_base_addr(reinterpret_cast<uint64_t>(fftsAddr));
        if (IsM2FusedMainAiv()) {
            moe_dispatch_combine_a8w8::ShapeConfig shape = M2FusedLoadShapeConfig(earlyStageStatus);
            M2FusedRecordStage(earlyStageStatus + kM2FusedFullAivHeaderSlot, 0U, kM2FusedFullMagic);
            M2FusedRecordStage(earlyStageStatus + kM2FusedFullAivHeaderSlot, 1U,
                               static_cast<int32_t>(get_block_num() * get_subblockdim()));
            M2FusedRecordStage(earlyStageStatus + kM2FusedFullAivHeaderSlot, 2U, static_cast<int32_t>(shape.rankNum));
            M2FusedRecordStage(earlyStageStatus + kM2FusedFullAivHeaderSlot, 3U,
                               static_cast<int32_t>(shape.hiddenSize + shape.intermediateSize));
            M2FusedRecordStage(earlyStageStatus + kM2FusedFullStageBaseSlot, 5U, 93);
        }
        pto::SYNCALL<pto::SyncCoreType::Mix>();
        return;
    }
    if (earlyDebugStopStage == 95U) {
        set_ffts_base_addr(reinterpret_cast<uint64_t>(fftsAddr));
        moe_dispatch_combine_a8w8::ShapeConfig shape = M2FusedLoadShapeConfig(earlyStageStatus);
        if (IsM2FusedMainAiv()) {
            M2FusedRecordStage(earlyStageStatus + kM2FusedFullAivHeaderSlot, 0U, kM2FusedFullMagic);
            M2FusedRecordStage(earlyStageStatus + kM2FusedFullAivHeaderSlot, 1U,
                               static_cast<int32_t>(get_block_num() * get_subblockdim()));
            M2FusedRecordStage(earlyStageStatus + kM2FusedFullAivHeaderSlot, 2U, static_cast<int32_t>(shape.rankNum));
            M2FusedRecordStage(earlyStageStatus + kM2FusedFullAivHeaderSlot, 3U,
                               static_cast<int32_t>(shape.hiddenSize + shape.intermediateSize));
            M2FusedRecordStage(earlyStageStatus + kM2FusedFullStageBaseSlot, 5U, 95);
        }
        pto::SYNCALL<pto::SyncCoreType::Mix>();
        return;
    }
    if (earlyDebugStopStage == 94U) {
        set_ffts_base_addr(reinterpret_cast<uint64_t>(fftsAddr));
        moe_dispatch_combine_a8w8::RankConfig rank = M2FusedLoadRankConfig(earlyStageStatus);
        if (IsM2FusedMainAiv()) {
            M2FusedRecordStage(earlyStageStatus + kM2FusedFullAivHeaderSlot, 0U, kM2FusedFullMagic);
            M2FusedRecordStage(earlyStageStatus + kM2FusedFullAivHeaderSlot, 1U,
                               static_cast<int32_t>(get_block_num() * get_subblockdim()));
            M2FusedRecordStage(earlyStageStatus + kM2FusedFullAivHeaderSlot, 2U, static_cast<int32_t>(rank.rankId));
            M2FusedRecordStage(earlyStageStatus + kM2FusedFullAivHeaderSlot, 3U, static_cast<int32_t>(rank.rankNum));
            M2FusedRecordStage(earlyStageStatus + kM2FusedFullStageBaseSlot, 5U, 94);
        }
        pto::SYNCALL<pto::SyncCoreType::Mix>();
        return;
    }
    if (earlyDebugStopStage == 98U) {
        set_ffts_base_addr(reinterpret_cast<uint64_t>(fftsAddr));
        __gm__ int32_t *stageStatus = reinterpret_cast<__gm__ int32_t *>(launchArgs->stageStatusAddr);
        moe_dispatch_combine_a8w8::ShapeConfig shape = M2FusedLoadShapeConfig(stageStatus);
        moe_dispatch_combine_a8w8::RankConfig rank = M2FusedLoadRankConfig(stageStatus);
        if (IsM2FusedMainAiv()) {
            M2FusedRecordStage(stageStatus + kM2FusedFullAivHeaderSlot, 0U, kM2FusedFullMagic);
            M2FusedRecordStage(stageStatus + kM2FusedFullAivHeaderSlot, 1U,
                               static_cast<int32_t>(get_block_num() * get_subblockdim()));
            M2FusedRecordStage(stageStatus + kM2FusedFullAivHeaderSlot, 2U, static_cast<int32_t>(rank.rankId));
            M2FusedRecordStage(stageStatus + kM2FusedFullAivHeaderSlot, 3U, static_cast<int32_t>(shape.rankNum));
            M2FusedRecordStage(stageStatus + kM2FusedFullStageBaseSlot, 5U, 98);
        }
        pto::SYNCALL<pto::SyncCoreType::Mix>();
        return;
    }
    if (earlyDebugStopStage == 89U) {
        set_ffts_base_addr(reinterpret_cast<uint64_t>(fftsAddr));
        if (IsM2FusedMainAiv()) {
            M2FusedBasicVecProbe(earlyStageStatus + kM2FusedFullStageBaseSlot, 5U, 89);
        }
        pto::SYNCALL<pto::SyncCoreType::Mix>();
        return;
    }
    set_ffts_base_addr(reinterpret_cast<uint64_t>(fftsAddr));
    __gm__ int32_t *stageStatusBase = reinterpret_cast<__gm__ int32_t *>(launchArgs->stageStatusAddr);
    moe_dispatch_combine_a8w8::ShapeConfig shape = M2FusedLoadShapeConfig(stageStatusBase);
    moe_dispatch_combine_a8w8::RankConfig rank = M2FusedLoadRankConfig(stageStatusBase);
    GM_ADDR inputA = reinterpret_cast<GM_ADDR>(
        M2FusedLoadConfigU64(stageStatusBase, moe_dispatch_combine_a8w8::kM2FusedFullPtrInputASlot));
    GM_ADDR expertIdx = reinterpret_cast<GM_ADDR>(
        M2FusedLoadConfigU64(stageStatusBase, moe_dispatch_combine_a8w8::kM2FusedFullPtrExpertIdxSlot));
    GM_ADDR xActiveMask = reinterpret_cast<GM_ADDR>(
        M2FusedLoadConfigU64(stageStatusBase, moe_dispatch_combine_a8w8::kM2FusedFullPtrXActiveMaskSlot));
    GM_ADDR probs = reinterpret_cast<GM_ADDR>(
        M2FusedLoadConfigU64(stageStatusBase, moe_dispatch_combine_a8w8::kM2FusedFullPtrProbsSlot));
    GM_ADDR outputC = reinterpret_cast<GM_ADDR>(
        M2FusedLoadConfigU64(stageStatusBase, moe_dispatch_combine_a8w8::kM2FusedFullPtrOutputCSlot));
    GM_ADDR peerWindow = reinterpret_cast<GM_ADDR>(
        M2FusedLoadConfigU64(stageStatusBase, moe_dispatch_combine_a8w8::kM2FusedFullPtrPeerWindowSlot));
    GM_ADDR hcclCtx = reinterpret_cast<GM_ADDR>(
        M2FusedLoadConfigU64(stageStatusBase, moe_dispatch_combine_a8w8::kM2FusedFullPtrHcclCtxSlot));
    GM_ADDR workspace = reinterpret_cast<GM_ADDR>(
        M2FusedLoadConfigU64(stageStatusBase, moe_dispatch_combine_a8w8::kM2FusedFullPtrWorkspaceSlot));
    auto workspaceLayout = MakeM2WorkspaceLayoutDevice(shape);
    auto peerWindowLayout = MakeM2PeerWindowLayoutDevice(shape);
    M2WorkspaceViewDevice workspaceView = MakeM2WorkspaceViewDevice(workspace, workspaceLayout);
    M2PeerWindowViewDevice localPeer = MakeM2PeerWindowViewDevice(peerWindow, peerWindowLayout);
    __gm__ HcclDeviceContext *ctx = reinterpret_cast<__gm__ HcclDeviceContext *>(hcclCtx);
    __gm__ int32_t *stageStatus = workspaceView.stageStatus;
    uint32_t debugStopStage = static_cast<uint32_t>(LoadScalarI32(stageStatus + kM2FusedFullDebugStopSlot));
    if (debugStopStage == 0U) {
        debugStopStage = M2FusedLoadDebugStopStage(launchArgs);
    }
    uint32_t timelineEnable = static_cast<uint32_t>(
        M2FusedLoadConfigI32(stageStatusBase, moe_dispatch_combine_a8w8::kM2FusedFullTimelineEnableSlot));
    uint32_t overlapMode = static_cast<uint32_t>(
        M2FusedLoadConfigI32(stageStatusBase, moe_dispatch_combine_a8w8::kM2FusedFullOverlapModeSlot));
    uint32_t logicalAiv = M2FusedLogicalAivId();
    uint32_t rawAivSlot = M2FusedRawAivSlot();
    bool m3n5DispatchGmm1Overlap = M3N5DispatchGmm1OverlapEnabled(debugStopStage, overlapMode, shape);
    bool m3n6Gmm1ActivationOverlap = M3N6Gmm1ActivationOverlapEnabled(debugStopStage, overlapMode, shape);
    bool m3n7ActivationGmm2Overlap = M3N7ActivationGmm2OverlapEnabled(debugStopStage, overlapMode, shape);
    bool m3n8Gmm2CombineOverlap = M3N8Gmm2CombineOverlapEnabled(debugStopStage, overlapMode, shape);
    uint32_t m3n6SyncGroupCount = moe_dispatch_combine_a8w8::M3N6SwigluSyncGroupCount(shape);
    M3NRecordAivLaneDebug(localPeer, rawAivSlot);
    if (IsM2FusedMainAiv()) {
        uint32_t m3nHandshakeSites = moe_dispatch_combine_a8w8::kM3N4StreamHandshakeSites;
        uint32_t m3nCvWaitCount = moe_dispatch_combine_a8w8::kM3N4FullOpenCvWaitCount;
        if (m3n5DispatchGmm1Overlap) {
            m3nHandshakeSites = m3nHandshakeSites - 1U + shape.expertPerRank;
            m3nCvWaitCount = m3nCvWaitCount - 1U + shape.expertPerRank;
        }
        if (m3n6Gmm1ActivationOverlap) {
            m3nHandshakeSites = m3nHandshakeSites - 2U + m3n6SyncGroupCount;
            m3nCvWaitCount = m3nCvWaitCount - 2U + m3n6SyncGroupCount;
        }
        if (m3n7ActivationGmm2Overlap) {
            m3nHandshakeSites = m3nHandshakeSites - 1U + m3n6SyncGroupCount;
            m3nCvWaitCount = m3nCvWaitCount - 1U + m3n6SyncGroupCount;
        }
        if (m3n8Gmm2CombineOverlap) {
            --m3nHandshakeSites;
            --m3nCvWaitCount;
        }
        M2FusedRecordStage(stageStatus + kM2FusedFullAivHeaderSlot, 0U, kM2FusedFullMagic);
        M2FusedRecordStage(stageStatus + kM2FusedFullAivHeaderSlot, 1U,
                           static_cast<int32_t>(get_block_num() * get_subblockdim()));
        M2FusedRecordStage(stageStatus + kM2FusedFullAivHeaderSlot, 2U, static_cast<int32_t>(rank.rankId));
        M2FusedRecordStage(stageStatus + kM2FusedFullAivHeaderSlot, 3U, static_cast<int32_t>(shape.rankNum));
        StoreScalarI32(localPeer.debugCounters + kM3CounterBase + 0U, static_cast<int32_t>(overlapMode));
        StoreScalarI32(localPeer.debugCounters + kM3CounterBase + 1U, static_cast<int32_t>(timelineEnable));
        StoreScalarI32(localPeer.debugCounters + kM3CounterBase + 2U, 1);
        StoreScalarI32(localPeer.debugCounters + kM3CounterBase + 3U, 1);
        StoreScalarI32(localPeer.debugCounters + kM3CounterBase + 4U, 1);
        StoreScalarI32(localPeer.debugCounters + kM3CounterBase + 5U, 6);
        StoreScalarI32(localPeer.debugCounters + kM3CounterBase + 6U, 1);
        StoreScalarI32(localPeer.debugCounters + kM3CounterBase + 11U,
                       (m3n5DispatchGmm1Overlap || m3n6Gmm1ActivationOverlap) ? 0 : 1);
        StoreScalarI32(localPeer.debugCounters + kM3CounterBase + 12U, static_cast<int32_t>(m3nHandshakeSites));
        StoreScalarI32(localPeer.debugCounters + kM3CounterBase + 13U,
                       static_cast<int32_t>(moe_dispatch_combine_a8w8::kM3N4CoarseMixSyncallCount +
                                            moe_dispatch_combine_a8w8::kM3N4AicOnlySyncCount +
                                            moe_dispatch_combine_a8w8::kM3N4AivOnlySyncCount));
        StoreScalarI32(localPeer.debugCounters + kM3CounterBase + 14U, static_cast<int32_t>(m3nCvWaitCount));
        StoreScalarI32(localPeer.debugCounters + kM3CounterBase + 15U, 0);
        StoreScalarI32(localPeer.debugCounters + kM3NActivationCounterBase + 8U, m3n6Gmm1ActivationOverlap ? 1 : 0);
        for (uint32_t idx = 0; idx < 16U; ++idx) {
            StoreScalarI32(localPeer.debugCounters + kM3NGmm2CounterBase + idx, 0);
        }
        StoreScalarI32(localPeer.debugCounters + kM3NGmm2CounterBase + 0U, m3n7ActivationGmm2Overlap ? 1 : 0);
        StoreScalarI32(localPeer.debugCounters + kM3NGmm2CounterBase + 1U, static_cast<int32_t>(m3n6SyncGroupCount));
        StoreScalarI32(localPeer.debugCounters + kM3NGmm2CounterBase + 2U, 0);
    }
    if (debugStopStage == 102303U) {
        if (IsM2FusedMainAiv()) {
            M2FusedBasicVecProbe(stageStatus + kM2FusedFullStageBaseSlot, 5U, 102303);
        }
        pto::SYNCALL<pto::SyncCoreType::Mix>();
        return;
    }
    if (IsM2FusedMainAiv()) {
        M2FusedRecordStage(stageStatus + kM2FusedFullStageBaseSlot, 0U, 1);
        if (debugStopStage == 102304U) {
            M2FusedBasicVecProbe(stageStatus + kM2FusedFullStageBaseSlot, 5U, 102304);
            M2FusedRecordStage(stageStatus + kM2FusedFullStageBaseSlot, 5U, 102304);
        }
    }
    if (debugStopStage == 102304U) {
        pto::SYNCALL<pto::SyncCoreType::Mix>();
        return;
    }
    uint32_t rowBytes = static_cast<uint32_t>(peerWindowLayout.dispatchPayloadRowBytes);
    uint32_t logicalAivCount = M2FusedLogicalAivCount();
    uint32_t dispatchWorkerCount = M3NDispatchWorkerCount(shape, logicalAivCount);
    bool m3nDispatchEnabled = debugStopStage < 100U && M3NFusedDispatchScratchFits(shape);
    if (m3nDispatchEnabled) {
        M3NDispatchHardPhaseSync();
        if (IsM2FusedMainAiv()) {
            dispatchWorkerCount = M3NAssignDispatchWorkers(shape, localPeer, logicalAivCount);
        }
        M3NDispatchHardPhaseSync();
        M3NDispatchWorkerAssignment dispatchAssignment = M3NLoadDispatchWorkerAssignment(localPeer, rawAivSlot);
        uint32_t dispatchWorkerId = dispatchAssignment.workerId;
        dispatchWorkerCount = dispatchAssignment.workerCount;
        bool activeDispatchWorker = dispatchAssignment.active;
        if (IsM2FusedMainAiv()) {
            M2FusedRecordStage(stageStatus + kM2FusedFullStageBaseSlot, 5U, 100);
            M2ClearDispatchState(shape, workspaceView, localPeer);
            M2FusedRecordStage(stageStatus + kM2FusedFullStageBaseSlot, 5U, 101);
        }
        M3NDispatchHardPhaseSync();
        if (activeDispatchWorker) {
            M3NClearDispatchWorkerScratch(shape, localPeer, dispatchWorkerId, dispatchWorkerCount);
        }
        M3NDispatchHardPhaseSync();
        if (activeDispatchWorker) {
            M3NCountLocalRoutesShard(shape, localPeer, expertIdx, xActiveMask, dispatchWorkerId, dispatchWorkerCount);
        }
        M3NDispatchHardPhaseSync();
        if (IsM2FusedMainAiv()) {
            M3NMergeLocalRouteCounts(shape, workspaceView, localPeer, rank.rankId, dispatchWorkerCount);
            M2FusedRecordStage(stageStatus + kM2FusedFullStageBaseSlot, 5U, 102);
        }
        M3NDispatchHardPhaseSync();
        if (activeDispatchWorker) {
            M3NRoutePackQuantLocalShard(shape, workspaceView, localPeer, inputA, expertIdx, xActiveMask, rowBytes,
                                        dispatchWorkerId, dispatchWorkerCount);
        }
        M3NDispatchHardPhaseSync();
        if (activeDispatchWorker) {
            M3NPublishCountRowsShard(shape, workspaceView, ctx, peerWindow, rank.rankId, peerWindowLayout,
                                     dispatchWorkerId, dispatchWorkerCount);
        }
        if (IsM2FusedMainAiv()) {
            StoreScalarI32(localPeer.debugCounters + kM3NDispatchCounterBase + 1U,
                           static_cast<int32_t>(dispatchWorkerCount));
            M2FusedRecordStage(stageStatus + kM2FusedFullStageBaseSlot, 5U, 104);
        }
        M3NDispatchHardPhaseSync();
        if (activeDispatchWorker) {
            M3NWaitCountRowsShard(shape, localPeer, dispatchWorkerId, dispatchWorkerCount);
        }
        M3NDispatchHardPhaseSync();
        if (IsM2FusedMainAiv()) {
            M2BuildPrefixMetadata(shape, workspaceView, localPeer, rank.rankId);
            M2FusedRecordStage(stageStatus + kM2FusedFullStageBaseSlot, 5U, 106);
        }
        M3NDispatchHardPhaseSync();
        if (m3n5DispatchGmm1Overlap) {
            bool requireAicStartAck = debugStopStage != 1U;
            M3N5GatherDispatchToGmm1InputShardByExpert(
                shape, workspaceView, localPeer, ctx, peerWindow, rank.rankId, rowBytes, peerWindowLayout,
                dispatchWorkerId, dispatchWorkerCount, activeDispatchWorker, stageStatus, requireAicStartAck);
        } else if (activeDispatchWorker) {
            M3NGatherDispatchToGmm1InputShard(shape, workspaceView, localPeer, ctx, peerWindow, rank.rankId, rowBytes,
                                              peerWindowLayout, dispatchWorkerId, dispatchWorkerCount);
        }
        if (m3n5DispatchGmm1Overlap) {
            M3NDispatchAivOnlyPhaseSync();
        } else {
            M3NDispatchHardPhaseSync();
        }
        if (IsM2FusedMainAiv()) {
            M3NFinalizeDispatchLedgerAfterGather(shape, workspaceView);
            M2BuildSwigluSyncMetadata(shape, workspaceView);
            uint32_t gatherWorkers =
                dispatchWorkerCount < shape.expertPerRank ? dispatchWorkerCount : shape.expertPerRank;
            StoreScalarI32(localPeer.debugCounters, 1);
            StoreScalarI32(localPeer.debugCounters + kM3CounterBase + 7U, static_cast<int32_t>(shape.expertPerRank));
            StoreScalarI32(localPeer.debugCounters + kM3NDispatchCounterBase + 2U, static_cast<int32_t>(gatherWorkers));
            StoreScalarI32(localPeer.debugCounters + kM3NDispatchCounterBase + 5U,
                           static_cast<int32_t>(shape.expertPerRank * shape.rankNum));
            StoreScalarI32(localPeer.debugCounters + kM3NDispatchCounterBase + 6U, static_cast<int32_t>(shape.m));
            StoreScalarI32(localPeer.debugCounters + kM3NDispatchCounterBase + 9U, 1);
            M2FusedRecordStage(stageStatus + kM2FusedFullStageBaseSlot, 5U, 107);
        }
        if (m3n6Gmm1ActivationOverlap) {
            moe_dispatch_combine_a8w8::M3N4AivAllDoneCoarseSync();
        }
    } else if (IsM2FusedMainAiv()) {
        M2FusedRecordStage(stageStatus + kM2FusedFullStageBaseSlot, 5U, 100);
        if (debugStopStage != 100U) {
            M2ClearDispatchState(shape, workspaceView, localPeer);
            M2FusedRecordStage(stageStatus + kM2FusedFullStageBaseSlot, 5U, 101);
        }
        if (debugStopStage != 100U && debugStopStage != 101U) {
            M2CountLocalRoutes(shape, workspaceView, localPeer, expertIdx, xActiveMask, rank.rankId);
            M2FusedRecordStage(stageStatus + kM2FusedFullStageBaseSlot, 5U, 102);
        }
        if (debugStopStage != 100U && debugStopStage != 101U && debugStopStage != 102U) {
            if (debugStopStage == 1021U) {
                uint32_t globalExpertNum = shape.rankNum * shape.expertPerRank;
                __gm__ int32_t *expertIds = reinterpret_cast<__gm__ int32_t *>(expertIdx);
                for (uint32_t globalExpert = 0; globalExpert < globalExpertNum; ++globalExpert) {
                    StoreScalarI32(workspaceView.tokenOwnerRankOffsets + globalExpert,
                                   LoadScalarI32(workspaceView.blockPrefixPerExpert + globalExpert));
                }
                for (uint32_t token = 0; token < shape.m; ++token) {
                    for (uint32_t slot = 0; slot < shape.topK; ++slot) {
                        uint32_t routeIndex = token * shape.topK + slot;
                        int32_t expert = LoadScalarI32(expertIds + routeIndex);
                        if (expert < 0 || static_cast<uint32_t>(expert) >= globalExpertNum) {
                            StoreScalarI32(workspaceView.expandedRowIdx + routeIndex, -1);
                            continue;
                        }
                        uint32_t globalExpert = static_cast<uint32_t>(expert);
                        int32_t packedRow = LoadScalarI32(workspaceView.tokenOwnerRankOffsets + globalExpert);
                        StoreScalarI32(workspaceView.tokenOwnerRankOffsets + globalExpert, packedRow + 1);
                        StoreScalarI32(workspaceView.expandedRowIdx + routeIndex, packedRow);
                    }
                }
                InvalidateGmCacheLines(workspaceView.expandedRowIdx,
                                       static_cast<uint32_t>(shape.m * shape.topK * sizeof(int32_t)));
                M2FusedRecordStage(stageStatus + kM2FusedFullStageBaseSlot, 5U, 1021);
            } else if (debugStopStage == 1022U) {
                uint32_t globalExpertNum = shape.rankNum * shape.expertPerRank;
                __gm__ int32_t *expertIds = reinterpret_cast<__gm__ int32_t *>(expertIdx);
                __gm__ half *input = reinterpret_cast<__gm__ half *>(inputA);
                int32_t expert = LoadScalarI32(expertIds);
                int32_t firstValue = static_cast<int32_t>(*reinterpret_cast<__gm__ uint16_t *>(input));
                if (expert >= 0 && static_cast<uint32_t>(expert) < globalExpertNum) {
                    StoreScalarI32(workspaceView.expandedRowIdx, 0);
                    StoreScalarI32(workspaceView.tokenOwnerRankOffsets + static_cast<uint32_t>(expert), 1);
                    localPeer.dispatchScale[0] = 1.0f;
                    localPeer.dispatchPayload[0] = static_cast<int8_t>(firstValue & 0x7f);
                    InvalidateGmCacheLines(workspaceView.expandedRowIdx, sizeof(int32_t));
                    InvalidateGmCacheLines(localPeer.dispatchPayload, 64U);
                    InvalidateGmCacheLines(localPeer.dispatchScale, sizeof(float));
                }
                M2FusedRecordStage(stageStatus + kM2FusedFullStageBaseSlot, 5U, 1022);
            } else if ((debugStopStage >= 10230U && debugStopStage <= 10236U) || debugStopStage == 102301U ||
                       debugStopStage == 102302U) {
                uint32_t globalExpertNum = shape.rankNum * shape.expertPerRank;
                __gm__ int32_t *expertIds = reinterpret_cast<__gm__ int32_t *>(expertIdx);
                for (uint32_t globalExpert = 0; globalExpert < globalExpertNum; ++globalExpert) {
                    StoreScalarI32(workspaceView.tokenOwnerRankOffsets + globalExpert,
                                   LoadScalarI32(workspaceView.blockPrefixPerExpert + globalExpert));
                }
                int32_t expert = LoadScalarI32(expertIds);
                if (expert >= 0 && static_cast<uint32_t>(expert) < globalExpertNum) {
                    uint32_t globalExpert = static_cast<uint32_t>(expert);
                    int32_t packedRow = LoadScalarI32(workspaceView.tokenOwnerRankOffsets + globalExpert);
                    StoreScalarI32(workspaceView.tokenOwnerRankOffsets + globalExpert, packedRow + 1);
                    StoreScalarI32(workspaceView.expandedRowIdx, packedRow);
                    if (debugStopStage == 10230U) {
                        M2FusedRecordStage(stageStatus + kM2FusedFullStageBaseSlot, 5U, 10230);
                    } else {
                        M2FusedRouteQuantProbe(shape, localPeer, inputA, static_cast<uint32_t>(packedRow), rowBytes,
                                               stageStatus, debugStopStage);
                    }
                    InvalidateGmCacheLines(workspaceView.expandedRowIdx, sizeof(int32_t));
                    InvalidateGmCacheLines(localPeer.dispatchPayload + static_cast<uint64_t>(packedRow) * rowBytes,
                                           rowBytes);
                    InvalidateGmCacheLines(localPeer.dispatchScale + packedRow, sizeof(float));
                }
            } else if (debugStopStage == 1023U) {
                uint32_t globalExpertNum = shape.rankNum * shape.expertPerRank;
                __gm__ int32_t *expertIds = reinterpret_cast<__gm__ int32_t *>(expertIdx);
                for (uint32_t globalExpert = 0; globalExpert < globalExpertNum; ++globalExpert) {
                    StoreScalarI32(workspaceView.tokenOwnerRankOffsets + globalExpert,
                                   LoadScalarI32(workspaceView.blockPrefixPerExpert + globalExpert));
                }
                int32_t expert = LoadScalarI32(expertIds);
                if (expert >= 0 && static_cast<uint32_t>(expert) < globalExpertNum) {
                    uint32_t globalExpert = static_cast<uint32_t>(expert);
                    int32_t packedRow = LoadScalarI32(workspaceView.tokenOwnerRankOffsets + globalExpert);
                    StoreScalarI32(workspaceView.tokenOwnerRankOffsets + globalExpert, packedRow + 1);
                    StoreScalarI32(workspaceView.expandedRowIdx, packedRow);
                    M2QuantizeRowToPeerPayload(shape, localPeer, inputA, 0U, static_cast<uint32_t>(packedRow),
                                               rowBytes);
                    InvalidateGmCacheLines(workspaceView.expandedRowIdx, sizeof(int32_t));
                    InvalidateGmCacheLines(localPeer.dispatchPayload + static_cast<uint64_t>(packedRow) * rowBytes,
                                           rowBytes);
                    InvalidateGmCacheLines(localPeer.dispatchScale + packedRow, sizeof(float));
                }
                M2FusedRecordStage(stageStatus + kM2FusedFullStageBaseSlot, 5U, 1023);
            } else {
                M2RoutePackQuantLocal(shape, workspaceView, localPeer, inputA, expertIdx, xActiveMask, rowBytes);
                M2FusedRecordStage(stageStatus + kM2FusedFullStageBaseSlot, 5U, 103);
            }
        }
        if (debugStopStage < 1021U && debugStopStage != 100U && debugStopStage != 101U && debugStopStage != 102U &&
            debugStopStage != 103U) {
            M2PublishCountRows(shape, workspaceView, localPeer, ctx, peerWindow, rank.rankId, peerWindowLayout);
            M2FusedRecordStage(stageStatus + kM2FusedFullStageBaseSlot, 5U, 104);
        }
        if (debugStopStage < 1021U && debugStopStage != 100U && debugStopStage != 101U && debugStopStage != 102U &&
            debugStopStage != 103U && debugStopStage != 104U) {
            M2WaitCountRows(shape, localPeer);
            M2FusedRecordStage(stageStatus + kM2FusedFullStageBaseSlot, 5U, 105);
        }
        if (debugStopStage < 1021U && debugStopStage != 100U && debugStopStage != 101U && debugStopStage != 102U &&
            debugStopStage != 103U && debugStopStage != 104U && debugStopStage != 105U) {
            M2BuildPrefixMetadata(shape, workspaceView, localPeer, rank.rankId);
            M2FusedRecordStage(stageStatus + kM2FusedFullStageBaseSlot, 5U, 106);
        }
        if (debugStopStage < 1021U && debugStopStage != 100U && debugStopStage != 101U && debugStopStage != 102U &&
            debugStopStage != 103U && debugStopStage != 104U && debugStopStage != 105U && debugStopStage != 106U) {
            M2GatherDispatchToGmm1Input(shape, workspaceView, localPeer, ctx, peerWindow, rank.rankId, rowBytes,
                                        peerWindowLayout);
            M2BuildSwigluSyncMetadata(shape, workspaceView);
            StoreScalarI32(localPeer.debugCounters, 1);
            StoreScalarI32(localPeer.debugCounters + kM3CounterBase + 7U, static_cast<int32_t>(shape.expertPerRank));
            StoreScalarI32(localPeer.debugCounters + kM3NDispatchCounterBase + 0U, 1);
            StoreScalarI32(localPeer.debugCounters + kM3NDispatchCounterBase + 1U, 1);
            StoreScalarI32(localPeer.debugCounters + kM3NDispatchCounterBase + 2U, 1);
            M2FusedRecordStage(stageStatus + kM2FusedFullStageBaseSlot, 5U, 107);
        }
    }
    if (rawAivSlot < 64U) {
        M2FusedRecordStage(stageStatus + kM2FusedFullAivStageBaseSlot, rawAivSlot, 100);
    }
    pipe_barrier(PIPE_ALL);
    dsb(DSB_DDR);
    if (debugStopStage == 109U) {
        if (IsM2FusedMainAiv()) {
            M3N9RecordTimeoutDump(workspaceView, localPeer, rank, shape, 0U, 0U,
                                  moe_dispatch_combine_a8w8::kM3N9TimeoutStageDispatchToGmm1,
                                  moe_dispatch_combine_a8w8::kM3N4DispatchToGmm1Flag, debugStopStage);
            M2FusedRecordStage(stageStatus + kM2FusedFullStageBaseSlot, 5U, 109);
        }
        pto::SYNCALL<pto::SyncCoreType::Mix>();
        return;
    }
    if (debugStopStage >= 100U) {
        pto::SYNCALL<pto::SyncCoreType::Mix>();
        return;
    }
    if (!m3n5DispatchGmm1Overlap) {
        moe_dispatch_combine_a8w8::M3N4SignalV2C<moe_dispatch_combine_a8w8::kM3N4DispatchToGmm1Flag>();
    }
    if (debugStopStage == 1U) {
        return;
    }
    bool m3nActivationEnabled = debugStopStage < 100U && M3NFusedDispatchScratchFits(shape);
    if (m3n6Gmm1ActivationOverlap) {
        if (debugStopStage == 2U) {
            return;
        }
        M3NDispatchWorkerAssignment activationAssignment = M3NLoadDispatchWorkerAssignment(localPeer, rawAivSlot);
        uint32_t activationWorkerCount = activationAssignment.workerCount;
        bool activeActivationWorker = activationAssignment.active;
        uint32_t syncGroupCount = static_cast<uint32_t>(LoadScalarI32(workspaceView.swigluSyncGroups));
        if (syncGroupCount == 0U || syncGroupCount > shape.expertPerRank) {
            syncGroupCount = shape.expertPerRank;
        }
        if (IsM2FusedMainAiv()) {
            M2FusedRecordStage(stageStatus + kM2FusedFullStageBaseSlot, 2U, 1);
            StoreScalarI32(localPeer.debugCounters + kM3CounterBase + 8U, 0);
            StoreScalarI32(localPeer.debugCounters + kM3CounterBase + 9U, 0);
            for (uint32_t idx = 0; idx < 16U; ++idx) {
                StoreScalarI32(localPeer.debugCounters + kM3NActivationCounterBase + idx, 0);
            }
            StoreScalarI32(localPeer.debugCounters + kM3NActivationCounterBase + 8U, 1);
        }
        if (activeActivationWorker) {
            M3NRecordActivationLaneDebug(localPeer, rawAivSlot, 0U, syncGroupCount, false);
            M3NInitActivationScaleShard(shape, workspaceView, activationAssignment.workerId, activationWorkerCount);
        }
        moe_dispatch_combine_a8w8::M3N4AivAllDoneCoarseSync();
        for (uint32_t syncIdx = 0; syncIdx < syncGroupCount; ++syncIdx) {
            moe_dispatch_combine_a8w8::M3N6WaitGmm1SyncGroupReady(syncIdx);
            if (IsM2FusedMainAiv()) {
                M2FusedRecordStage(stageStatus + kM2FusedFullStageBaseSlot, 1U, 1);
                M3N6RecordActivationStartedBeforeLastGmm1Ready(stageStatus, localPeer, syncIdx, syncGroupCount);
                M3N6RunGmm1EpilogueSyncGroup(shape, workspaceView, syncIdx);
                StoreScalarI32(localPeer.debugCounters + kM3CounterBase + 8U,
                               LoadScalarI32(localPeer.debugCounters + kM3CounterBase + 8U) + 1);
            }
            moe_dispatch_combine_a8w8::M3N4AivAllDoneCoarseSync();
            if (activeActivationWorker) {
                uint32_t rowsProcessed = M3NRunActivationQuantSyncGroupShard(
                    shape, workspaceView, syncIdx, activationAssignment.workerId, activationWorkerCount);
                M3NRecordActivationLaneDebug(localPeer, rawAivSlot, rowsProcessed, syncGroupCount, true);
            }
            moe_dispatch_combine_a8w8::M3N4AivAllDoneCoarseSync();
            if (IsM2FusedMainAiv()) {
                M3NActivationLaneSummary activationSummary =
                    M3NAggregateActivationLaneDebug(localPeer, logicalAivCount);
                M3N6FinalizeActivationQuantSyncGroup(
                    shape, workspaceView, localPeer, syncIdx, activationWorkerCount, activationSummary.activeWorkerMask,
                    activationSummary.rowsProcessed, syncIdx == 0U, syncIdx + 1U == syncGroupCount);
                StoreScalarI32(localPeer.debugCounters + kM3CounterBase + 9U,
                               LoadScalarI32(localPeer.debugCounters + kM3CounterBase + 9U) + 1);
            }
            moe_dispatch_combine_a8w8::M3N4AivAllDoneCoarseSync();
        }
        if (debugStopStage == 3U) {
            return;
        }
    } else {
        moe_dispatch_combine_a8w8::M3N4WaitC2V<moe_dispatch_combine_a8w8::kM3N4Gmm1ToEpilogueFlag>();
        if (debugStopStage == 2U) {
            return;
        }
        if (IsM2FusedMainAiv()) {
            M2FusedRecordStage(stageStatus + kM2FusedFullStageBaseSlot, 1U, 1);
            M2RunGmm1Epilogue(shape, workspaceView, rank.rankId);
            StoreScalarI32(localPeer.debugCounters + kM3CounterBase + 8U,
                           LoadScalarI32(workspaceView.swigluSyncGroups));
        }
        pipe_barrier(PIPE_ALL);
        dsb(DSB_DDR);
        moe_dispatch_combine_a8w8::M3N4AivAllDoneCoarseSync();
        moe_dispatch_combine_a8w8::M3N4SignalV2C<moe_dispatch_combine_a8w8::kM3N4Gmm1EpilogueToActivationFlag>();
        if (debugStopStage == 3U) {
            return;
        }
        if (m3nActivationEnabled) {
            M3NDispatchWorkerAssignment activationAssignment = M3NLoadDispatchWorkerAssignment(localPeer, rawAivSlot);
            uint32_t activationWorkerCount = activationAssignment.workerCount;
            bool activeActivationWorker = activationAssignment.active;
            uint32_t syncGroupCount = static_cast<uint32_t>(LoadScalarI32(workspaceView.swigluSyncGroups));
            if (syncGroupCount == 0U || syncGroupCount > shape.expertPerRank) {
                syncGroupCount = shape.expertPerRank;
            }
            if (IsM2FusedMainAiv()) {
                M2FusedRecordStage(stageStatus + kM2FusedFullStageBaseSlot, 2U, 1);
                for (uint32_t idx = 0; idx < 16U; ++idx) {
                    StoreScalarI32(localPeer.debugCounters + kM3NActivationCounterBase + idx, 0);
                }
            }
            if (activeActivationWorker) {
                M3NRecordActivationLaneDebug(localPeer, rawAivSlot, 0U, syncGroupCount, false);
            }
            M3NDispatchHardPhaseSync();
            if (activeActivationWorker) {
                M3NInitActivationScaleShard(shape, workspaceView, activationAssignment.workerId, activationWorkerCount);
            }
            M3NDispatchHardPhaseSync();
            if (activeActivationWorker) {
                uint32_t rowsProcessed = M3NRunActivationQuantShard(shape, workspaceView, activationAssignment.workerId,
                                                                    activationWorkerCount);
                M3NRecordActivationLaneDebug(localPeer, rawAivSlot, rowsProcessed, syncGroupCount, true);
            }
            M3NDispatchHardPhaseSync();
            if (IsM2FusedMainAiv()) {
                M3NActivationLaneSummary activationSummary =
                    M3NAggregateActivationLaneDebug(localPeer, logicalAivCount);
                if (activationSummary.activeWorkers == 0U) {
                    M2RunActivationQuant(shape, workspaceView, rank.rankId);
                    StoreScalarI32(localPeer.debugCounters + kM3NActivationCounterBase + 0U, 1);
                    StoreScalarI32(localPeer.debugCounters + kM3NActivationCounterBase + 4U, 1);
                    StoreScalarI32(localPeer.debugCounters + kM3NActivationCounterBase + 7U, 1);
                } else {
                    M3NFinalizeActivationQuant(shape, workspaceView, localPeer, activationSummary.activeWorkers,
                                               activationSummary.activeWorkerMask, activationSummary.rowsProcessed);
                }
                StoreScalarI32(localPeer.debugCounters + kM3CounterBase + 9U,
                               LoadScalarI32(workspaceView.swigluSyncGroups));
            }
        } else if (IsM2FusedMainAiv()) {
            M2FusedRecordStage(stageStatus + kM2FusedFullStageBaseSlot, 2U, 1);
            M2RunActivationQuant(shape, workspaceView, rank.rankId);
            StoreScalarI32(localPeer.debugCounters + kM3CounterBase + 9U,
                           LoadScalarI32(workspaceView.swigluSyncGroups));
        }
    }
    pipe_barrier(PIPE_ALL);
    dsb(DSB_DDR);
    if (!m3n7ActivationGmm2Overlap) {
        moe_dispatch_combine_a8w8::M3N4SignalV2C<moe_dispatch_combine_a8w8::kM3N4ActivationToGmm2Flag>();
    }
    if (debugStopStage == 4U) {
        return;
    }
    if (!m3n8Gmm2CombineOverlap) {
        moe_dispatch_combine_a8w8::M3N4WaitC2V<moe_dispatch_combine_a8w8::kM3N4Gmm2ToCombineFlag>();
    }
    if (debugStopStage == 5U) {
        if (m3n8Gmm2CombineOverlap && IsM2FusedMainAiv()) {
            for (uint32_t localExpert = 0; localExpert < shape.expertPerRank; ++localExpert) {
                M3N8WaitGmm2ExpertReadyGm(workspaceView, localExpert);
            }
        }
        return;
    }
    bool m3nCombineEnabled = debugStopStage < 100U && M3NFusedDispatchScratchFits(shape);
    if (m3nCombineEnabled) {
        M3NDispatchWorkerAssignment combineAssignment = M3NLoadDispatchWorkerAssignment(localPeer, rawAivSlot);
        uint32_t combineWorkerCount = combineAssignment.workerCount;
        bool activeCombineWorker = combineAssignment.active;
        if (IsM2FusedMainAiv()) {
            M2FusedRecordStage(stageStatus + kM2FusedFullStageBaseSlot, 3U, 1);
            if (m3n8Gmm2CombineOverlap) {
                M3N8InitCombineCounters(localPeer, combineWorkerCount, shape.expertPerRank);
            } else {
                M3NInitCombineCounters(localPeer, combineWorkerCount);
                M2BuildReturnSegmentMap(shape, workspaceView, localPeer, rank.rankId);
            }
        }
        if (!m3n8Gmm2CombineOverlap) {
            M3NDispatchHardPhaseSync();
        } else {
            M3N8AivOnlyPhaseSync();
        }
        if (debugStopStage == 51U) {
            return;
        }
        bool materializeEnabled = debugStopStage != 56U;
        bool transferEnabled = debugStopStage != 53U;
        bool notifyEnabled = debugStopStage != 53U && debugStopStage != 54U;
        if (m3n8Gmm2CombineOverlap) {
            M3N8RunGmm2EpilogueAndReturnByExpert(shape, workspaceView, localPeer, ctx, peerWindow, rank.rankId,
                                                 peerWindowLayout, combineAssignment.workerId, combineWorkerCount,
                                                 activeCombineWorker, IsM2FusedMainAiv(), materializeEnabled,
                                                 transferEnabled, notifyEnabled);
            if (activeCombineWorker) {
                StoreScalarI32(localPeer.debugCounters + kM3N8CombineCounterBase + 11U,
                               LoadScalarI32(localPeer.debugCounters + kM3N8CombineCounterBase + 11U) + 1);
            }
            M3N8AivOnlyPhaseSync();
        } else if (activeCombineWorker) {
            M3NCombineShardStats combineStats = M3NRunGmm2EpilogueAndReturnShard(
                shape, workspaceView, localPeer, ctx, peerWindow, rank.rankId, peerWindowLayout,
                combineAssignment.workerId, combineWorkerCount, materializeEnabled, transferEnabled, notifyEnabled);
            (void)combineStats;
        }
        if (!m3n8Gmm2CombineOverlap) {
            M3NDispatchHardPhaseSync();
        }
        if (debugStopStage == 52U || debugStopStage == 53U || debugStopStage == 54U || debugStopStage == 56U) {
            return;
        }
        if (IsM2FusedMainAiv()) {
            M3NFinalizeGmm2EpilogueAndReturn(shape, localPeer, ctx, peerWindow, rank.rankId, peerWindowLayout,
                                             combineWorkerCount);
            StoreScalarI32(localPeer.debugCounters + kM3CounterBase + 10U, static_cast<int32_t>(shape.expertPerRank));
            if (m3n8Gmm2CombineOverlap) {
                M2FusedRecordStage(stageStatus + kM2FusedFullM3N8EvidenceSlot, 0U,
                                   LoadScalarI32(localPeer.debugCounters + kM3N8CombineCounterBase + 5U));
                M2FusedRecordStage(stageStatus + kM2FusedFullM3N8EvidenceSlot, 1U,
                                   LoadScalarI32(localPeer.debugCounters + kM3N8CombineCounterBase + 2U));
                M2FusedRecordStage(stageStatus + kM2FusedFullM3N8EvidenceSlot, 2U,
                                   LoadScalarI32(localPeer.debugCounters + kM3N8CombineCounterBase + 6U));
            }
        }
    } else if (IsM2FusedMainAiv()) {
        M2FusedRecordStage(stageStatus + kM2FusedFullStageBaseSlot, 3U, 1);
        M2RunGmm2EpilogueAndReturn(shape, workspaceView, localPeer, ctx, peerWindow, rank.rankId, peerWindowLayout);
        StoreScalarI32(localPeer.debugCounters + kM3CounterBase + 10U, static_cast<int32_t>(shape.expertPerRank));
    }
    pipe_barrier(PIPE_ALL);
    dsb(DSB_DDR);
    moe_dispatch_combine_a8w8::M3N4AivAllDoneCoarseSync();
    moe_dispatch_combine_a8w8::M3N4SignalV2C<moe_dispatch_combine_a8w8::kM3N4CombineToRestoreFlag>();
    if (debugStopStage == 6U) {
        return;
    }
    if (logicalAiv < 8U) {
        uint32_t tokenBegin = TokenShardBegin(shape.m, logicalAiv, 8U);
        uint32_t tokenEnd = TokenShardEnd(shape.m, logicalAiv, 8U);
        __gm__ float *probValues = reinterpret_cast<__gm__ float *>(probs);
        __gm__ half *output = reinterpret_cast<__gm__ half *>(outputC);
        int32_t outputRowStride = static_cast<int32_t>(shape.hiddenSize);
        int32_t returnRowStride = static_cast<int32_t>(peerWindowLayout.returnPayloadRowBytes / sizeof(half));
        for (uint32_t token = tokenBegin; token < tokenEnd; ++token) {
            StoreZeroRowHalf(output, outputRowStride, static_cast<int32_t>(token),
                             static_cast<int32_t>(shape.hiddenSize));
            for (uint32_t slot = 0; slot < shape.topK; ++slot) {
                uint32_t routeIndex = token * shape.topK + slot;
                int32_t ptrDRow = LoadScalarI32(workspaceView.expandedRowIdx + routeIndex);
                if (ptrDRow < 0 || static_cast<uint32_t>(ptrDRow) >= static_cast<uint32_t>(M2LocalRows(shape))) {
                    continue;
                }
                float prob = probValues[routeIndex];
                for (int32_t col = 0; col < static_cast<int32_t>(shape.hiddenSize); col += kDefaultTileCols) {
                    int32_t cols = static_cast<int32_t>(shape.hiddenSize) - col;
                    if (cols > kDefaultTileCols) {
                        cols = kDefaultTileCols;
                    }
                    VecTile<half, kDefaultTileCols> outTile(1, cols);
                    VecTile<half, kDefaultTileCols> ptrTile(1, cols);
                    TASSIGN(outTile, kPingUbAddr);
                    TASSIGN(ptrTile, kPongUbAddr);
                    GlobalNd<half> outGlobal = MakeGlobal2D(
                        output + static_cast<int64_t>(token) * outputRowStride + col, 1, cols, outputRowStride);
                    __gm__ half *returnChunk =
                        localPeer.returnPayload + static_cast<int64_t>(ptrDRow) * returnRowStride + col;
                    InvalidateGmCacheLines(returnChunk, static_cast<uint32_t>(cols) * sizeof(half));
                    GlobalNd<half> returnGlobal = MakeGlobal2D(returnChunk, 1, cols, returnRowStride);
                    TLOAD(ptrTile, returnGlobal);
                    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                    TLOAD(outTile, outGlobal);
                    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                    TAXPY(outTile, ptrTile, static_cast<half>(prob));
                    pipe_barrier(PIPE_V);
                    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                    TSTORE(outGlobal, outTile);
                    WaitStoreTileReusable();
                }
            }
        }
    }
    if (IsM2FusedMainAiv()) {
        M2FusedRecordStage(stageStatus + kM2FusedFullStageBaseSlot, 4U, 1);
        StoreScalarI32(localPeer.debugCounters + 6U * 16U, 1);
    }
    pipe_barrier(PIPE_ALL);
    dsb(DSB_DDR);
}
#endif

#if defined(M2_MIXED_SPIKE_BUILD_AIC) && !defined(M2_MIXED_SPIKE_REGISTER_BUILD)
namespace {

const char *GetRegisterElfPath(const void *anchor)
{
#if defined(M2_MIXED_SPIKE_REGISTER_OBJECT_PATH)
    (void)anchor;
    return M2_MIXED_SPIKE_REGISTER_OBJECT_PATH;
#else
    Dl_info info{};
    if (dladdr(anchor, &info) == 0 || info.dli_fname == nullptr) {
        std::fprintf(stderr, "dladdr failed for M2 mixed spike kernel\n");
        std::abort();
    }
    return info.dli_fname;
#endif
}

std::vector<char> ReadRegisterElf(const char *path)
{
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::fprintf(stderr, "failed to open M2 mixed spike register ELF: %s\n", path);
        std::abort();
    }
    const std::streamsize size = file.tellg();
    if (size <= 0) {
        std::fprintf(stderr, "invalid M2 mixed spike register ELF size: %s\n", path);
        std::abort();
    }
    std::vector<char> data(static_cast<size_t>(size));
    file.seekg(0, std::ios::beg);
    if (!file.read(data.data(), size)) {
        std::fprintf(stderr, "failed to read M2 mixed spike register ELF: %s\n", path);
        std::abort();
    }
    return data;
}

void LaunchM2MixedSpikeWithHandle(const void *anchor, uint8_t *heartbeat, uint32_t rank, uint32_t aicBlocks,
                                  uint32_t aivRatio, void *stream)
{
    const char *path = GetRegisterElfPath(anchor);
    static const std::vector<char> kernelBinary = ReadRegisterElf(path);
    rtDevBinary_t binary{RT_DEV_BINARY_MAGIC_ELF, 0, kernelBinary.data(), kernelBinary.size()};
    void *handle = nullptr;
    rtError_t ret = rtRegisterAllKernel(&binary, &handle);
    if (ret != RT_ERROR_NONE || handle == nullptr) {
        ret = rtBinaryLoadWithoutTilingKey(kernelBinary.data(), kernelBinary.size(), &handle);
        if (ret != RT_ERROR_NONE || handle == nullptr) {
            std::fprintf(stderr, "register M2 mixed spike failed, path=%s, size=%zu, ret=%d\n", path,
                         kernelBinary.size(), ret);
            std::abort();
        }
    }

    (void)rank;
    (void)aivRatio;
    void *args[] = {heartbeat};
    rtArgsEx_t argsInfo{};
    argsInfo.args = args;
    argsInfo.argsSize = sizeof(args);
    rtTaskCfgInfo_t cfgInfo{};
    ret = rtKernelLaunchWithHandleV2(handle, kM2MixedSpikeTilingKey, aicBlocks, &argsInfo, nullptr, stream, &cfgInfo);
    if (ret != RT_ERROR_NONE) {
        std::fprintf(stderr, "rtKernelLaunchWithHandleV2 failed for M2 mixed spike, ret=%d\n", ret);
        std::abort();
    }
}

void LaunchM2FusedSkeletonWithHandle(const void *anchor, uint8_t *ledger, uint32_t rank, uint32_t aicBlocks,
                                     uint32_t aivRatio, void *stream)
{
    const char *path = GetRegisterElfPath(anchor);
    static const std::vector<char> kernelBinary = ReadRegisterElf(path);
    rtDevBinary_t binary{RT_DEV_BINARY_MAGIC_ELF, 0, kernelBinary.data(), kernelBinary.size()};
    void *handle = nullptr;
    rtError_t ret = rtRegisterAllKernel(&binary, &handle);
    if (ret != RT_ERROR_NONE || handle == nullptr) {
        ret = rtBinaryLoadWithoutTilingKey(kernelBinary.data(), kernelBinary.size(), &handle);
        if (ret != RT_ERROR_NONE || handle == nullptr) {
            std::fprintf(stderr, "register M2 fused skeleton failed, path=%s, size=%zu, ret=%d\n", path,
                         kernelBinary.size(), ret);
            std::abort();
        }
    }

    (void)rank;
    (void)aivRatio;
    uint64_t ffts = 0;
    uint32_t fftsLen = 0;
    ret = rtGetC2cCtrlAddr(&ffts, &fftsLen);
    if (ret != RT_ERROR_NONE || ffts == 0) {
        std::fprintf(stderr, "rtGetC2cCtrlAddr failed for M2 fused skeleton, ret=%d, ffts=%lu, len=%u\n", ret, ffts,
                     fftsLen);
        std::abort();
    }
    void *args[] = {reinterpret_cast<uint64_t *>(ffts), ledger};
    rtArgsEx_t argsInfo{};
    argsInfo.args = args;
    argsInfo.argsSize = sizeof(args);
    rtTaskCfgInfo_t cfgInfo{};
    ret =
        rtKernelLaunchWithHandleV2(handle, kM2FusedSkeletonTilingKey, aicBlocks, &argsInfo, nullptr, stream, &cfgInfo);
    if (ret != RT_ERROR_NONE) {
        std::fprintf(stderr, "rtKernelLaunchWithHandleV2 failed for M2 fused skeleton, ret=%d\n", ret);
        std::abort();
    }
}

void LaunchM2FusedFullWithHandle(const void *anchor, moe_dispatch_combine_a8w8::M2FusedFullLaunchArgs *launchArgsDevice,
                                 uint32_t aicBlocks, uint32_t aivRatio, void *stream)
{
    const char *path = GetRegisterElfPath(anchor);
    static const std::vector<char> kernelBinary = ReadRegisterElf(path);
    rtDevBinary_t binary{RT_DEV_BINARY_MAGIC_ELF, 0, kernelBinary.data(), kernelBinary.size()};
    void *handle = nullptr;
    rtError_t ret = rtRegisterAllKernel(&binary, &handle);
    if (ret != RT_ERROR_NONE || handle == nullptr) {
        ret = rtBinaryLoadWithoutTilingKey(kernelBinary.data(), kernelBinary.size(), &handle);
        if (ret != RT_ERROR_NONE || handle == nullptr) {
            std::fprintf(stderr, "register M2 fused full failed, path=%s, size=%zu, ret=%d\n", path,
                         kernelBinary.size(), ret);
            std::abort();
        }
    }

    (void)aivRatio;
    uint64_t ffts = 0;
    uint32_t fftsLen = 0;
    ret = rtGetC2cCtrlAddr(&ffts, &fftsLen);
    if (ret != RT_ERROR_NONE || ffts == 0) {
        std::fprintf(stderr, "rtGetC2cCtrlAddr failed for M2 fused full, ret=%d, ffts=%lu, len=%u\n", ret, ffts,
                     fftsLen);
        std::abort();
    }
    void *args[] = {reinterpret_cast<uint64_t *>(ffts), launchArgsDevice};
    rtArgsEx_t argsInfo{};
    argsInfo.args = args;
    argsInfo.argsSize = sizeof(args);
    rtTaskCfgInfo_t cfgInfo{};
    ret = rtKernelLaunchWithHandleV2(handle, kM2FusedFullTilingKey, aicBlocks, &argsInfo, nullptr, stream, &cfgInfo);
    if (ret != RT_ERROR_NONE) {
        std::fprintf(stderr, "rtKernelLaunchWithHandleV2 failed for M2 fused full, ret=%d\n", ret);
        std::abort();
    }
}

} // namespace

namespace dispatch_combine_tile {

void LaunchM2MixedSpike(uint8_t *workspace, uint32_t rank, uint32_t aicBlocks, uint32_t aivRatio, void *stream)
{
    LaunchM2MixedSpikeWithHandle(reinterpret_cast<const void *>(&LaunchM2MixedSpike), workspace, rank, aicBlocks,
                                 aivRatio, stream);
}

void LaunchM2FusedSkeleton(uint8_t *workspace, uint32_t rank, uint32_t aicBlocks, uint32_t aivRatio, void *stream)
{
    LaunchM2FusedSkeletonWithHandle(reinterpret_cast<const void *>(&LaunchM2FusedSkeleton), workspace, rank, aicBlocks,
                                    aivRatio, stream);
}

void LaunchM2FusedFull(moe_dispatch_combine_a8w8::M2FusedFullLaunchArgs *launchArgsDevice, uint32_t aicBlocks,
                       uint32_t aivRatio, void *stream)
{
    LaunchM2FusedFullWithHandle(reinterpret_cast<const void *>(&LaunchM2FusedFull), launchArgsDevice, aicBlocks,
                                aivRatio, stream);
}

} // namespace dispatch_combine_tile
#endif
