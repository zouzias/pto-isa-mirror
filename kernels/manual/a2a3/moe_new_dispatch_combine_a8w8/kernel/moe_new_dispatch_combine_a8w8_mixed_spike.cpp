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

#include "moe_new_dispatch_combine_a8w8_types.hpp"
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
#include "moe_new_dispatch_combine_a8w8_gmm_kernel.cpp"
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
#include "moe_new_dispatch_combine_a8w8_kernel.cpp"
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
constexpr uint64_t kM2FusedVecProbeLowUbAddr = 0x0;
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
constexpr uint32_t kM2FusedFullM3CounterBase = 24U * 16U;
constexpr uint32_t kM2FusedFullM3NActivationCounterBase = kM2FusedFullM3CounterBase + 32U;
constexpr uint32_t kM2FusedFullM3N7Gmm2CounterBase = 24U * 16U + 64U;
constexpr uint32_t kM2FusedFullM3OGmm1CounterBase = 24U * 16U + 160U;
constexpr uint32_t kM2FusedFullM3OGmm2CounterBase = 24U * 16U + 208U;
constexpr uint32_t kM2FusedFullM3OGmmCounterWords = 48U;
constexpr uint32_t kM2FusedFullM3OGmmBlockCountBase = 16U;
constexpr uint32_t kM2FusedFullM3ORestoreCounterBase = moe_new_dispatch_combine_a8w8::kM3ORestoreCounterBase;
constexpr uint32_t kM2FusedFullM3ORestoreCounterWords = moe_new_dispatch_combine_a8w8::kM3ORestoreCounterWords;
constexpr uint32_t kM2FusedFullM3ORestoreWorkerBase = moe_new_dispatch_combine_a8w8::kM3ORestoreWorkerBase;
constexpr uint32_t kM2FusedFullM3ORestoreWorkerWords = moe_new_dispatch_combine_a8w8::kM3ORestoreWorkerWords;

AICORE inline uint64_t M3N12GetSysCnt()
{
    uint64_t syscnt = 0;
    asm volatile("MOV %0, SYS_CNT\n" : "+l"(syscnt));
    return syscnt;
}

AICORE inline uint64_t M3N12PackMeta0(moe_new_dispatch_combine_a8w8::M3N12TimelineKind kind,
                                      moe_new_dispatch_combine_a8w8::M3N12TimelineCoreType coreType, uint32_t coreId,
                                      moe_new_dispatch_combine_a8w8::M3N12TimelineStatus status, uint32_t aux0,
                                      uint32_t aux1)
{
    return static_cast<uint64_t>(static_cast<uint32_t>(kind) & 0xffU) |
           (static_cast<uint64_t>(static_cast<uint32_t>(coreType) & 0xffU) << 8U) |
           (static_cast<uint64_t>(coreId & 0xffU) << 16U) |
           (static_cast<uint64_t>(static_cast<uint32_t>(status) & 0xffU) << 24U) |
           (static_cast<uint64_t>(aux0 & 0xffffU) << 32U) | (static_cast<uint64_t>(aux1 & 0xffffU) << 48U);
}

AICORE inline uint64_t M3N12PackMeta1(uint32_t value0, uint32_t value1, uint32_t value2, uint32_t value3)
{
    return static_cast<uint64_t>(value0 & 0xffffU) | (static_cast<uint64_t>(value1 & 0xffffU) << 16U) |
           (static_cast<uint64_t>(value2 & 0xffffU) << 32U) | (static_cast<uint64_t>(value3 & 0xffffU) << 48U);
}

AICORE inline void M3N12RecordTimeline(__gm__ uint64_t *timeline, uint32_t slot, uint64_t begin, uint64_t end,
                                       uint64_t meta0, uint64_t meta1)
{
    if (slot >= moe_new_dispatch_combine_a8w8::kM3N12TimelineRecordCount) {
        return;
    }
    uint32_t base = slot * moe_new_dispatch_combine_a8w8::kM3N12TimelineRecordWords;
    timeline[base + 0U] = begin;
    timeline[base + 1U] = end;
    timeline[base + 2U] = meta0;
    timeline[base + 3U] = meta1;
    pipe_barrier(PIPE_ALL);
    dsb(DSB_DDR);
}

AICORE inline void M3N12RecordAivRange(__gm__ uint64_t *timeline, uint32_t slot,
                                       moe_new_dispatch_combine_a8w8::M3N12TimelineKind kind, uint32_t logicalAiv,
                                       uint32_t workerId, uint32_t rangeBegin, uint32_t rangeEnd, uint32_t processed,
                                       uint32_t skipped, uint64_t begin, uint64_t end)
{
    M3N12RecordTimeline(
        timeline, slot, begin, end,
        M3N12PackMeta0(kind, moe_new_dispatch_combine_a8w8::M3N12TimelineCoreType::kAiv, logicalAiv,
                       moe_new_dispatch_combine_a8w8::M3N12TimelineStatus::kProcessed, workerId, logicalAiv),
        M3N12PackMeta1(rangeBegin, rangeEnd, processed, skipped));
}

AICORE inline void M3N12RecordSkipped(__gm__ uint64_t *timeline, uint32_t slot,
                                      moe_new_dispatch_combine_a8w8::M3N12TimelineKind kind,
                                      moe_new_dispatch_combine_a8w8::M3N12TimelineCoreType coreType, uint32_t coreId)
{
    uint64_t stamp = M3N12GetSysCnt();
    M3N12RecordTimeline(
        timeline, slot, stamp, stamp,
        M3N12PackMeta0(kind, coreType, coreId, moe_new_dispatch_combine_a8w8::M3N12TimelineStatus::kSkipped, 0U, 0U),
        0U);
}
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
    moe_new_dispatch_combine_a8w8::M3N4AivAllDoneCoarseSync();
}

AICORE inline bool M3NDispatchUseInitQuantAivOnlySync(uint32_t debugStopStage)
{
    return debugStopStage == 0U || debugStopStage == 17U;
}

AICORE inline void M3NDispatchInitQuantPhaseSync(uint32_t debugStopStage)
{
    if (!M3NDispatchUseInitQuantAivOnlySync(debugStopStage)) {
        M3NDispatchHardPhaseSync();
        return;
    }
    M3NDispatchAivOnlyPhaseSync();
}

AICORE inline uint32_t M3NDispatchMixedSyncsWithInitQuantAivOnly(uint32_t debugStopStage, uint32_t mixedSyncs)
{
    if (!M3NDispatchUseInitQuantAivOnlySync(debugStopStage)) {
        return mixedSyncs;
    }
    if (mixedSyncs <= 4U) {
        return mixedSyncs;
    }
    uint32_t initQuantInternalSyncs = mixedSyncs - 4U;
    if (initQuantInternalSyncs > 3U) {
        initQuantInternalSyncs = 3U;
    }
    return mixedSyncs - initQuantInternalSyncs;
}

AICORE inline bool M3NFusedDispatchScratchFits(moe_new_dispatch_combine_a8w8::ShapeConfig shape)
{
    uint32_t globalExpertNum = shape.rankNum * shape.expertPerRank;
    return globalExpertNum > 0U;
}

AICORE inline bool M3N5DispatchGmm1OverlapEnabled(uint32_t debugStopStage, uint32_t overlapMode,
                                                  moe_new_dispatch_combine_a8w8::ShapeConfig shape)
{
    return overlapMode != 0U && debugStopStage < 100U && M3NFusedDispatchScratchFits(shape) &&
           moe_new_dispatch_combine_a8w8::M3N5DispatchExpertFlagsSupported(shape);
}

AICORE inline bool M3N6Gmm1ActivationOverlapEnabled(uint32_t debugStopStage, uint32_t overlapMode,
                                                    moe_new_dispatch_combine_a8w8::ShapeConfig shape)
{
    (void)debugStopStage;
    (void)overlapMode;
    (void)shape;
    return false;
}

AICORE inline bool M3N7ActivationGmm2OverlapEnabled(uint32_t debugStopStage, uint32_t overlapMode,
                                                    moe_new_dispatch_combine_a8w8::ShapeConfig shape)
{
    return M3N6Gmm1ActivationOverlapEnabled(debugStopStage, overlapMode, shape);
}

AICORE inline bool M3N8Gmm2CombineOverlapEnabled(uint32_t debugStopStage, uint32_t overlapMode,
                                                 moe_new_dispatch_combine_a8w8::ShapeConfig shape)
{
    return M3N7ActivationGmm2OverlapEnabled(debugStopStage, overlapMode, shape);
}

AICORE inline void M3NDispatchAicWaitForAivSubphases(uint32_t debugStopStage, bool m3n5DispatchGmm1Overlap,
                                                     moe_new_dispatch_combine_a8w8::ShapeConfig shape)
{
    if (debugStopStage >= 100U || !M3NFusedDispatchScratchFits(shape)) {
        return;
    }
    uint32_t dispatchSubphaseSyncs = m3n5DispatchGmm1Overlap ? 10U : 11U;
    if (debugStopStage >= 11U && debugStopStage <= 17U) {
        dispatchSubphaseSyncs = debugStopStage < 16U ? debugStopStage - 5U : 11U;
    } else if (debugStopStage >= 18U && debugStopStage <= 21U) {
        dispatchSubphaseSyncs = 8U;
    }
    dispatchSubphaseSyncs = M3NDispatchMixedSyncsWithInitQuantAivOnly(debugStopStage, dispatchSubphaseSyncs);
    for (uint32_t sync = 0; sync < dispatchSubphaseSyncs; ++sync) {
        M3NDispatchHardPhaseSync();
    }
}

AICORE inline bool M3NDispatchDebugProbeEnabled(uint32_t debugStopStage)
{
    return (debugStopStage >= 11U && debugStopStage <= 17U) || (debugStopStage >= 18U && debugStopStage <= 21U);
}

AICORE inline bool M3OActivationShardDebugProbeEnabled(uint32_t debugStopStage)
{
    return debugStopStage >= 41U && debugStopStage <= 56U;
}

AICORE inline void M3NActivationAicWaitForAivSubphases(uint32_t debugStopStage,
                                                       moe_new_dispatch_combine_a8w8::ShapeConfig shape)
{
    if (debugStopStage >= 100U || !M3NFusedDispatchScratchFits(shape)) {
        return;
    }
    uint32_t activationSubphaseSyncs = 3U;
    if (debugStopStage >= 41U && debugStopStage <= 43U) {
        activationSubphaseSyncs = debugStopStage - 40U;
    } else if (debugStopStage >= 44U && debugStopStage <= 48U) {
        activationSubphaseSyncs = 3U;
    }
    for (uint32_t sync = 0; sync < activationSubphaseSyncs; ++sync) {
        M3NDispatchHardPhaseSync();
    }
}

AICORE inline bool M3NCombineAicWaitForAivSubphases(uint32_t debugStopStage,
                                                    moe_new_dispatch_combine_a8w8::ShapeConfig shape)
{
    if (debugStopStage >= 100U || !M3NFusedDispatchScratchFits(shape)) {
        return false;
    }
    M3NDispatchHardPhaseSync();
    if (debugStopStage == 51U || debugStopStage == 57U || debugStopStage == 58U ||
        (debugStopStage >= 59U && debugStopStage <= 86U)) {
        return true;
    }
    M3NDispatchHardPhaseSync();
    return debugStopStage == 52U || debugStopStage == 53U || debugStopStage == 54U || debugStopStage == 56U;
}

AICORE inline void M3N5WaitAllDispatchExpertsReady(moe_new_dispatch_combine_a8w8::ShapeConfig shape)
{
    for (uint32_t localExpert = 0; localExpert < shape.expertPerRank; ++localExpert) {
        moe_new_dispatch_combine_a8w8::M3N5WaitDispatchExpertReady(localExpert);
    }
}

AICORE inline void M3N6SyncGroupExpertRange(moe_new_dispatch_combine_a8w8::ShapeConfig shape, uint32_t syncIdx,
                                            uint32_t *expertBegin, uint32_t *expertEnd)
{
    uint32_t begin = 0;
    for (uint32_t group = 0; group < syncIdx && begin < shape.expertPerRank; ++group) {
        begin += moe_new_dispatch_combine_a8w8::M3N6NextSwigluGroupSize(shape.expertPerRank - begin);
    }
    uint32_t end = begin;
    if (begin < shape.expertPerRank) {
        end = begin + moe_new_dispatch_combine_a8w8::M3N6NextSwigluGroupSize(shape.expertPerRank - begin);
    }
    if (end > shape.expertPerRank) {
        end = shape.expertPerRank;
    }
    *expertBegin = begin;
    *expertEnd = end;
}

#if defined(M2_MIXED_SPIKE_BUILD_AIC)
AICORE inline void M3N5RecordGmm1StartBeforeLastDispatchReady(moe_new_dispatch_combine_a8w8::ShapeConfig shape,
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
                                                moe_new_dispatch_combine_a8w8::WorkspaceLayout layout,
                                                GM_ADDR workspace, uint32_t syncIdx, uint32_t syncGroupCount)
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

AICORE inline void M3N6WaitGmm1SyncGroupReadyGm(__gm__ int32_t *gmm1SyncGroupReady, uint32_t syncIdx)
{
    __gm__ int32_t *ready = gmm1SyncGroupReady + syncIdx * 16U;
    while (true) {
        pipe_barrier(PIPE_ALL);
        dcci(static_cast<__gm__ void *>(ready), SINGLE_CACHE_LINE);
        dsb(DSB_DDR);
        if (M2FusedGmmLoadScalarI32(ready) != 0) {
            return;
        }
    }
}

AICORE inline void M3N6WaitAllGmm1SyncGroupsReadyGm(moe_new_dispatch_combine_a8w8::ShapeConfig shape, GM_ADDR workspace,
                                                    const moe_new_dispatch_combine_a8w8::WorkspaceLayout &layout)
{
    __gm__ int32_t *gmm1SyncGroupReady =
        reinterpret_cast<__gm__ int32_t *>(workspace + layout.gmm1SyncGroupReady.offset);
    uint32_t syncGroupCount = moe_new_dispatch_combine_a8w8::M3N6SwigluSyncGroupCount(shape);
    for (uint32_t syncIdx = 0; syncIdx < syncGroupCount; ++syncIdx) {
        M3N6WaitGmm1SyncGroupReadyGm(gmm1SyncGroupReady, syncIdx);
    }
}

AICORE inline uint64_t M3NFusedPeerAppendOffset(uint64_t *offset, uint64_t bytes)
{
    *offset = M2FusedGmmAlign64Device(*offset);
    uint64_t field = *offset;
    *offset += bytes;
    return field;
}

AICORE inline __gm__ int32_t *M3NFusedPeerDebugCounters(GM_ADDR peerWindow,
                                                        moe_new_dispatch_combine_a8w8::ShapeConfig shape)
{
    uint64_t offset = 0;
    uint64_t expandedRows = static_cast<uint64_t>(shape.m) * shape.topK;
    uint64_t tokenMatrixCount = static_cast<uint64_t>(shape.rankNum) * M2FusedGmmTokenPerExpertMatrixRowStride(shape);
    (void)M3NFusedPeerAppendOffset(&offset, moe_new_dispatch_combine_a8w8::kPeerWindowHeaderBytes);
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

#if defined(M2_MIXED_SPIKE_BUILD_AIC)
AICORE inline void M2FusedAicInvalidateGmCacheLines(__gm__ void *ptr, uint32_t bytes)
{
    pipe_barrier(PIPE_ALL);
    uint64_t start = reinterpret_cast<uint64_t>(ptr) & ~static_cast<uint64_t>(63);
    uint64_t end = (reinterpret_cast<uint64_t>(ptr) + bytes + 63) & ~static_cast<uint64_t>(63);
    for (uint64_t addr = start; addr < end; addr += 64) {
        dcci(reinterpret_cast<__gm__ void *>(addr), SINGLE_CACHE_LINE);
    }
    dsb(DSB_DDR);
}

AICORE inline void M2FusedRunInt8GmmTileScalar(__gm__ int8_t *input, __gm__ int8_t *weight, __gm__ int32_t *output,
                                               uint32_t mValid, uint32_t kSize, uint32_t nValid, uint32_t inputStride,
                                               uint32_t weightStride, uint32_t outputStride)
{
    if (mValid == 0U || kSize == 0U || nValid == 0U) {
        return;
    }
    M2FusedAicInvalidateGmCacheLines(
        input, static_cast<uint32_t>((static_cast<uint64_t>(mValid - 1U) * inputStride + kSize) * sizeof(int8_t)));
    M2FusedAicInvalidateGmCacheLines(
        weight, static_cast<uint32_t>((static_cast<uint64_t>(kSize - 1U) * weightStride + nValid) * sizeof(int8_t)));
    uint64_t total = static_cast<uint64_t>(mValid) * nValid;
    for (uint64_t linear = 0; linear < total; ++linear) {
        uint32_t row = static_cast<uint32_t>(linear / nValid);
        uint32_t col = static_cast<uint32_t>(linear - static_cast<uint64_t>(row) * nValid);
        int32_t acc = 0;
        __gm__ int8_t *inputRow = input + static_cast<uint64_t>(row) * inputStride;
        for (uint32_t k = 0; k < kSize; ++k) {
            int32_t lhs = static_cast<int32_t>(inputRow[k]);
            int32_t rhs = static_cast<int32_t>(weight[static_cast<uint64_t>(k) * weightStride + col]);
            acc += lhs * rhs;
        }
        M2FusedGmmStoreScalarI32(output + static_cast<uint64_t>(row) * outputStride + col, acc);
    }
}

AICORE inline void M2FusedClearGmmTaskEvidence(__gm__ int32_t *debugCounters, uint32_t base, uint32_t taskCount)
{
    uint32_t blockIdx = static_cast<uint32_t>(get_block_idx());
    uint32_t blockNum = static_cast<uint32_t>(get_block_num());
    if (blockIdx == 0U) {
        for (uint32_t idx = 0; idx < 8U; ++idx) {
            M2FusedGmmStoreScalarI32(debugCounters + base + idx, 0);
        }
        M2FusedGmmStoreScalarI32(debugCounters + base + 2U, -1);
        M2FusedGmmStoreScalarI32(debugCounters + base + 3U, -1);
        M2FusedGmmStoreScalarI32(debugCounters + base + 4U, -1);
        M2FusedGmmStoreScalarI32(debugCounters + base + 5U, -1);
        if (taskCount == 0U) {
            M2FusedGmmStoreScalarI32(debugCounters + base + 8U, -1);
            M2FusedGmmStoreScalarI32(debugCounters + base + 9U, -1);
            M2FusedGmmStoreScalarI32(debugCounters + base + 10U, -1);
            M2FusedGmmStoreScalarI32(debugCounters + base + 11U, -1);
            M2FusedGmmStoreScalarI32(debugCounters + base + 12U, -1);
        }
    }
    uint32_t lastTaskBlock = blockNum == 0U || taskCount == 0U ? 0U : (taskCount - 1U) % blockNum;
    if (taskCount != 0U && blockIdx == lastTaskBlock) {
        M2FusedGmmStoreScalarI32(debugCounters + base + 8U, -1);
        M2FusedGmmStoreScalarI32(debugCounters + base + 9U, -1);
        M2FusedGmmStoreScalarI32(debugCounters + base + 10U, -1);
        M2FusedGmmStoreScalarI32(debugCounters + base + 11U, -1);
        M2FusedGmmStoreScalarI32(debugCounters + base + 12U, -1);
    }
    if (blockIdx + kM2FusedFullM3OGmmBlockCountBase < kM2FusedFullM3OGmmCounterWords) {
        M2FusedGmmStoreScalarI32(debugCounters + base + kM2FusedFullM3OGmmBlockCountBase + blockIdx, 0);
    }
    pipe_barrier(PIPE_ALL);
    dsb(DSB_DDR);
}

AICORE inline void M2FusedRecordGmmTaskEvidence(__gm__ int32_t *debugCounters, uint32_t base, uint32_t taskCount,
                                                __gm__ int32_t *taskPlan)
{
    uint32_t blockIdx = static_cast<uint32_t>(get_block_idx());
    uint32_t blockNum = static_cast<uint32_t>(get_block_num());
    uint32_t localTaskCount = 0;
    uint32_t firstTaskId = 0;
    uint32_t lastTaskId = 0;
    bool haveTask = false;
    for (uint32_t taskId = blockIdx; taskId < taskCount; taskId += blockNum) {
        if (!haveTask) {
            firstTaskId = taskId;
            haveTask = true;
        }
        lastTaskId = taskId;
        ++localTaskCount;
    }
    if (!haveTask) {
        return;
    }
    __gm__ int32_t *firstTask = taskPlan + firstTaskId * kM2GmmTileTaskFields;
    __gm__ int32_t *lastTask = taskPlan + lastTaskId * kM2GmmTileTaskFields;
    if (blockIdx + kM2FusedFullM3OGmmBlockCountBase < kM2FusedFullM3OGmmCounterWords) {
        M2FusedGmmStoreScalarI32(debugCounters + base + kM2FusedFullM3OGmmBlockCountBase + blockIdx,
                                 static_cast<int32_t>(localTaskCount));
    }
    if (blockIdx == 0U) {
        M2FusedGmmStoreScalarI32(debugCounters + base + 2U, static_cast<int32_t>(blockIdx));
        M2FusedGmmStoreScalarI32(debugCounters + base + 3U, static_cast<int32_t>(firstTaskId));
        M2FusedGmmStoreScalarI32(debugCounters + base + 4U, M2FusedGmmLoadScalarI32(firstTask + 2U));
        M2FusedGmmStoreScalarI32(debugCounters + base + 5U, M2FusedGmmLoadScalarI32(firstTask + 3U));
        M2FusedGmmStoreScalarI32(debugCounters + base + 6U, M2FusedGmmLoadScalarI32(firstTask + 5U));
    }
    uint32_t lastTaskBlock = blockNum == 0U || taskCount == 0U ? 0U : (taskCount - 1U) % blockNum;
    if (blockIdx == lastTaskBlock) {
        M2FusedGmmStoreScalarI32(debugCounters + base + 8U, static_cast<int32_t>(blockIdx));
        M2FusedGmmStoreScalarI32(debugCounters + base + 9U, static_cast<int32_t>(lastTaskId));
        M2FusedGmmStoreScalarI32(debugCounters + base + 10U, M2FusedGmmLoadScalarI32(lastTask + 2U));
        M2FusedGmmStoreScalarI32(debugCounters + base + 11U, M2FusedGmmLoadScalarI32(lastTask + 3U));
        M2FusedGmmStoreScalarI32(debugCounters + base + 12U, M2FusedGmmLoadScalarI32(lastTask + 5U));
    }
    pipe_barrier(PIPE_ALL);
    dsb(DSB_DDR);
}

AICORE inline uint32_t M2FusedGmmActiveBlocks(uint32_t taskCount, uint32_t blockNum)
{
    return taskCount < blockNum ? taskCount : blockNum;
}

AICORE inline void M2FusedFinalizeGmmTaskEvidence(__gm__ int32_t *debugCounters, uint32_t base, uint32_t taskCount)
{
    uint32_t blockNum = static_cast<uint32_t>(get_block_num());
    if (static_cast<uint32_t>(get_block_idx()) != 0U) {
        return;
    }
    M2FusedGmmStoreScalarI32(debugCounters + base + 0U,
                             static_cast<int32_t>(M2FusedGmmActiveBlocks(taskCount, blockNum)));
    M2FusedGmmStoreScalarI32(debugCounters + base + 1U, static_cast<int32_t>(taskCount));
    M2FusedGmmStoreScalarI32(debugCounters + base + 7U, static_cast<int32_t>(blockNum));
    pipe_barrier(PIPE_ALL);
    dsb(DSB_DDR);
}

AICORE inline float M2FusedAicFloatFromBits(int32_t value)
{
    union {
        uint32_t u;
        float f;
    } bits{static_cast<uint32_t>(value)};
    return bits.f;
}

AICORE inline int32_t M2FusedAicFloatToBits(float value)
{
    union {
        float f;
        uint32_t u;
    } bits{value};
    return static_cast<int32_t>(bits.u);
}

AICORE inline float M2FusedAicLoadFloat(__gm__ float *ptr)
{
    return M2FusedAicFloatFromBits(M2FusedGmmLoadScalarI32(reinterpret_cast<__gm__ int32_t *>(ptr)));
}

AICORE inline void M2FusedAicStoreFloat(__gm__ float *ptr, float value)
{
    M2FusedGmmStoreScalarI32(reinterpret_cast<__gm__ int32_t *>(ptr), M2FusedAicFloatToBits(value));
}

AICORE inline float M2FusedAicAbsFloat(float value)
{
    return value < 0.0f ? -value : value;
}

AICORE inline float M2FusedAicExpApprox(float value)
{
    if (!(value == value)) {
        return 1.0f;
    }
    if (value <= -16.0f) {
        return 0.0f;
    }
    if (value >= 16.0f) {
        value = 16.0f;
    }
    constexpr float kInvLn2 = 1.4426950408889634f;
    constexpr float kLn2 = 0.6931471805599453f;
    int32_t exponent = static_cast<int32_t>(value * kInvLn2);
    if (value < 0.0f && static_cast<float>(exponent) * kLn2 > value) {
        --exponent;
    }
    if (exponent > 24) {
        exponent = 24;
    }
    if (exponent < -24) {
        exponent = -24;
    }
    float reduced = value - static_cast<float>(exponent) * kLn2;
    float reduced2 = reduced * reduced;
    float reduced4 = reduced2 * reduced2;
    float poly = 1.0f + reduced + reduced2 * 0.5f + reduced2 * reduced * 0.1666666667f + reduced4 * 0.0416666667f +
                 reduced4 * reduced * 0.0083333333f + reduced4 * reduced2 * 0.0013888889f +
                 reduced4 * reduced2 * reduced * 0.0001984127f + reduced4 * reduced4 * 0.0000248016f;
    if (exponent >= 0) {
        for (int32_t idx = 0; idx < exponent; ++idx) {
            poly *= 2.0f;
        }
    } else {
        for (int32_t idx = 0; idx < -exponent; ++idx) {
            poly *= 0.5f;
        }
    }
    return poly;
}

AICORE inline float M2FusedAicSigmoidScalar(float value)
{
    if (!(value == value)) {
        return 0.5f;
    }
    if (value >= 16.0f) {
        return 1.0f;
    }
    if (value <= -16.0f) {
        return 0.0f;
    }
    if (value >= 0.0f) {
        float expNeg = M2FusedAicExpApprox(-value);
        return 1.0f / (1.0f + expNeg);
    }
    float expPos = M2FusedAicExpApprox(value);
    return expPos / (1.0f + expPos);
}

AICORE inline int8_t M2FusedAicQuantizeToInt8Scalar(float value, float scale)
{
    if (scale <= 0.0f || !(scale == scale) || !(value == value)) {
        return 0;
    }
    float scaled = value / scale;
    int32_t quant = scaled >= 0.0f ? static_cast<int32_t>(scaled + 0.5f) : static_cast<int32_t>(scaled - 0.5f);
    if (quant > 127) {
        quant = 127;
    }
    if (quant < -127) {
        quant = -127;
    }
    return static_cast<int8_t>(quant);
}

AICORE inline float M2FusedAicActivationValue(moe_new_dispatch_combine_a8w8::ShapeConfig shape, __gm__ float *gmm1Out,
                                              uint32_t globalRow, uint32_t col, float routingScale)
{
    uint32_t w1Cols = shape.intermediateSize * 2U;
    float gate = M2FusedAicLoadFloat(gmm1Out + static_cast<uint64_t>(globalRow) * w1Cols + col) * routingScale;
    float up = M2FusedAicLoadFloat(gmm1Out + static_cast<uint64_t>(globalRow) * w1Cols + shape.intermediateSize + col) *
               routingScale;
    return M2FusedAicSigmoidScalar(gate) * up;
}

AICORE inline uint32_t M2FusedRunActivationQuantAicScalar(moe_new_dispatch_combine_a8w8::ShapeConfig shape,
                                                          GM_ADDR workspace,
                                                          const moe_new_dispatch_combine_a8w8::WorkspaceLayout &layout)
{
    uint32_t blockIdx = static_cast<uint32_t>(get_block_idx());
    uint32_t localRows = static_cast<uint32_t>(M2FusedGmmLocalRows(shape));
    uint32_t gmm2RowStride = static_cast<uint32_t>(M2FusedGmmAlign64Device(shape.intermediateSize));
    if (blockIdx != 0U) {
        return 0U;
    }
    __gm__ int32_t *dispatchOffset = reinterpret_cast<__gm__ int32_t *>(workspace + layout.dispatchOffset.offset);
    __gm__ int32_t *expertTokenNums = reinterpret_cast<__gm__ int32_t *>(workspace + layout.expertTokenNums.offset);
    __gm__ float *routingScale = reinterpret_cast<__gm__ float *>(workspace + layout.routingPerTokenScale.offset);
    __gm__ float *gmm1Out = reinterpret_cast<__gm__ float *>(workspace + layout.gmm1Out.offset);
    __gm__ float *swigluOut = reinterpret_cast<__gm__ float *>(workspace + layout.swigluOut.offset);
    __gm__ int8_t *gmm2Input = reinterpret_cast<__gm__ int8_t *>(workspace + layout.gmm2InputInt8.offset);
    __gm__ float *gmm2Scale = reinterpret_cast<__gm__ float *>(workspace + layout.gmm2PerTokenScale.offset);
    uint32_t w1Cols = shape.intermediateSize * 2U;
    M2FusedAicInvalidateGmCacheLines(routingScale, static_cast<uint32_t>(localRows * sizeof(float)));
    M2FusedAicInvalidateGmCacheLines(gmm1Out,
                                     static_cast<uint32_t>(static_cast<uint64_t>(localRows) * w1Cols * sizeof(float)));
    for (uint32_t row = 0; row < localRows; ++row) {
        M2FusedAicStoreFloat(gmm2Scale + row, 1.0f);
    }
    uint32_t rowsProcessed = 0;
    for (uint32_t localExpert = 0; localExpert < shape.expertPerRank; ++localExpert) {
        int32_t rowBeginI32 = M2FusedGmmLoadScalarI32(dispatchOffset + localExpert);
        int32_t rowCountI32 = M2FusedGmmLoadScalarI32(expertTokenNums + localExpert);
        if (rowBeginI32 < 0 || rowCountI32 <= 0) {
            continue;
        }
        uint32_t rowBegin = static_cast<uint32_t>(rowBeginI32);
        uint32_t rowCount = static_cast<uint32_t>(rowCountI32);
        for (uint32_t row = 0; row < rowCount; ++row) {
            uint32_t globalRow = rowBegin + row;
            if (globalRow >= localRows) {
                continue;
            }
            float rowRoutingScale = M2FusedAicLoadFloat(routingScale + globalRow);
            float maxAbs = 0.0f;
            for (uint32_t col = 0; col < shape.intermediateSize; ++col) {
                float value = M2FusedAicActivationValue(shape, gmm1Out, globalRow, col, rowRoutingScale);
                M2FusedAicStoreFloat(swigluOut + static_cast<uint64_t>(globalRow) * shape.intermediateSize + col,
                                     value);
                float absValue = M2FusedAicAbsFloat(value);
                if (absValue > maxAbs) {
                    maxAbs = absValue;
                }
            }
            float tokenScale = maxAbs == 0.0f ? 1.0f : maxAbs / 127.0f;
            M2FusedAicStoreFloat(gmm2Scale + globalRow, tokenScale);
            __gm__ int8_t *dst = gmm2Input + static_cast<uint64_t>(globalRow) * gmm2RowStride;
            for (uint32_t col = 0; col < shape.intermediateSize; ++col) {
                float value =
                    M2FusedAicLoadFloat(swigluOut + static_cast<uint64_t>(globalRow) * shape.intermediateSize + col);
                dst[col] = M2FusedAicQuantizeToInt8Scalar(value, tokenScale);
            }
            for (uint32_t col = shape.intermediateSize; col < gmm2RowStride; ++col) {
                dst[col] = 0;
            }
            ++rowsProcessed;
        }
    }
    pipe_barrier(PIPE_ALL);
    dsb(DSB_DDR);
    return rowsProcessed;
}

AICORE inline void M2FusedFinalizeActivationQuantAicScalar(moe_new_dispatch_combine_a8w8::ShapeConfig shape,
                                                           GM_ADDR workspace,
                                                           const moe_new_dispatch_combine_a8w8::WorkspaceLayout &layout,
                                                           __gm__ int32_t *debugCounters, uint32_t rowsProcessed)
{
    __gm__ int32_t *activationReady =
        reinterpret_cast<__gm__ int32_t *>(workspace + layout.activationSyncGroupReady.offset);
    __gm__ int32_t *swigluSyncGroups = reinterpret_cast<__gm__ int32_t *>(workspace + layout.swigluSyncGroups.offset);
    __gm__ int32_t *swigluGroupDesc = reinterpret_cast<__gm__ int32_t *>(workspace + layout.swigluGroupDesc.offset);
    __gm__ float *swigluOut = reinterpret_cast<__gm__ float *>(workspace + layout.swigluOut.offset);
    __gm__ int8_t *gmm2Input = reinterpret_cast<__gm__ int8_t *>(workspace + layout.gmm2InputInt8.offset);
    __gm__ float *gmm2Scale = reinterpret_cast<__gm__ float *>(workspace + layout.gmm2PerTokenScale.offset);
    uint32_t syncGroupCount = static_cast<uint32_t>(M2FusedGmmLoadScalarI32(swigluSyncGroups));
    if (syncGroupCount == 0U || syncGroupCount > shape.expertPerRank) {
        syncGroupCount = shape.expertPerRank;
    }
    uint32_t tileCount = 0;
    uint32_t skippedGroups = 0;
    uint32_t lastRowEnd = 0;
    uint32_t firstTileBegin = 0;
    uint32_t lastTileEnd = 0;
    uint32_t firstRowBegin = 0;
    bool haveGroup = false;
    for (uint32_t syncIdx = 0; syncIdx < syncGroupCount; ++syncIdx) {
        __gm__ int32_t *group = swigluGroupDesc + syncIdx * kM2SwigluGroupFields;
        int32_t rowBegin = M2FusedGmmLoadScalarI32(group + 3U);
        int32_t rowEnd = M2FusedGmmLoadScalarI32(group + 4U);
        int32_t tileBegin = M2FusedGmmLoadScalarI32(group + 5U);
        int32_t tileEnd = M2FusedGmmLoadScalarI32(group + 6U);
        if (rowEnd <= rowBegin) {
            ++skippedGroups;
        }
        if (tileEnd > tileBegin) {
            tileCount += static_cast<uint32_t>(tileEnd - tileBegin);
        }
        if (!haveGroup) {
            firstTileBegin = tileBegin > 0 ? static_cast<uint32_t>(tileBegin) : 0U;
            firstRowBegin = rowBegin > 0 ? static_cast<uint32_t>(rowBegin) : 0U;
            haveGroup = true;
        }
        lastRowEnd = rowEnd > 0 ? static_cast<uint32_t>(rowEnd) : 0U;
        lastTileEnd = tileEnd > 0 ? static_cast<uint32_t>(tileEnd) : 0U;
        M2FusedGmmStoreScalarI32(activationReady + syncIdx * 16U, 1);
    }
    M2FusedGmmStoreScalarI32(debugCounters + kM2FusedFullM3CounterBase + 9U, static_cast<int32_t>(syncGroupCount));
    M2FusedGmmStoreScalarI32(debugCounters + kM2FusedFullM3NActivationCounterBase + 0U, 1);
    M2FusedGmmStoreScalarI32(debugCounters + kM2FusedFullM3NActivationCounterBase + 1U,
                             static_cast<int32_t>(rowsProcessed));
    M2FusedGmmStoreScalarI32(debugCounters + kM2FusedFullM3NActivationCounterBase + 2U,
                             static_cast<int32_t>(tileCount));
    M2FusedGmmStoreScalarI32(debugCounters + kM2FusedFullM3NActivationCounterBase + 3U,
                             static_cast<int32_t>(skippedGroups));
    M2FusedGmmStoreScalarI32(debugCounters + kM2FusedFullM3NActivationCounterBase + 4U, 1);
    M2FusedGmmStoreScalarI32(debugCounters + kM2FusedFullM3NActivationCounterBase + 5U, 0);
    M2FusedGmmStoreScalarI32(debugCounters + kM2FusedFullM3NActivationCounterBase + 6U, 0);
    M2FusedGmmStoreScalarI32(debugCounters + kM2FusedFullM3NActivationCounterBase + 7U, 1);
    M2FusedGmmStoreScalarI32(debugCounters + kM2FusedFullM3NActivationCounterBase + 8U, 0);
    M2FusedGmmStoreScalarI32(debugCounters + kM2FusedFullM3NActivationCounterBase + 9U,
                             static_cast<int32_t>(syncGroupCount));
    M2FusedGmmStoreScalarI32(debugCounters + kM2FusedFullM3NActivationCounterBase + 10U, 1);
    M2FusedGmmStoreScalarI32(debugCounters + kM2FusedFullM3NActivationCounterBase + 11U,
                             static_cast<int32_t>(lastRowEnd));
    M2FusedGmmStoreScalarI32(debugCounters + kM2FusedFullM3NActivationCounterBase + 12U,
                             static_cast<int32_t>(firstTileBegin));
    M2FusedGmmStoreScalarI32(debugCounters + kM2FusedFullM3NActivationCounterBase + 13U,
                             static_cast<int32_t>(lastTileEnd));
    M2FusedGmmStoreScalarI32(debugCounters + kM2FusedFullM3NActivationCounterBase + 14U,
                             static_cast<int32_t>(firstRowBegin));
    M2FusedGmmStoreScalarI32(debugCounters + kM2FusedFullM3NActivationCounterBase + 15U, static_cast<int32_t>(1));
    uint32_t localRows = static_cast<uint32_t>(M2FusedGmmLocalRows(shape));
    uint32_t gmm2RowStride = static_cast<uint32_t>(M2FusedGmmAlign64Device(shape.intermediateSize));
    M2FusedAicInvalidateGmCacheLines(activationReady, static_cast<uint32_t>(syncGroupCount * 16U * sizeof(int32_t)));
    M2FusedAicInvalidateGmCacheLines(swigluOut,
                                     static_cast<uint32_t>(localRows * shape.intermediateSize * sizeof(float)));
    M2FusedAicInvalidateGmCacheLines(gmm2Input, static_cast<uint32_t>(localRows * gmm2RowStride));
    M2FusedAicInvalidateGmCacheLines(gmm2Scale, static_cast<uint32_t>(localRows * sizeof(float)));
    pipe_barrier(PIPE_ALL);
    dsb(DSB_DDR);
}
#endif

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

AICORE inline void M3N5GatherOneDispatchExpert(moe_new_dispatch_combine_a8w8::ShapeConfig shape,
                                               M2WorkspaceViewDevice workspaceView, M2PeerWindowViewDevice localPeer,
                                               __gm__ HcclDeviceContext *ctx, GM_ADDR peerWindow, uint32_t myRank,
                                               uint32_t rowBytes,
                                               const moe_new_dispatch_combine_a8w8::PeerWindowLayout &peerWindowLayout,
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
        if (rows <= 0) {
            continue;
        }
        int32_t srcStart = M2SourceGlobalExpertRowBase(shape, localPeer, tokenOwner, globalExpert);
        int32_t localRowCap = static_cast<int32_t>(M2LocalRows(shape));
        if (srcStart >= localRowCap) {
            continue;
        }
        if (srcStart + rows > localRowCap) {
            rows = localRowCap - srcStart;
        }
        if (rows <= 0) {
            continue;
        }
        if (tokenOwner == myRank) {
            M2CopyLocalDispatchRowsToGmm1(workspaceView.gmm1InputInt8, rowBytes, workspaceView.routingPerTokenScale,
                                          localPeer, rowBytes, dstStart, srcStart, rows);
        } else {
            M2PeerWindowViewDevice remotePeer =
                MakeM2RemotePeerWindowViewDevice(ctx, peerWindow, tokenOwner, peerWindowLayout);
            M2CopyRemoteDispatchRowsToGmm1Scalar(workspaceView.gmm1InputInt8, rowBytes,
                                                 workspaceView.routingPerTokenScale, remotePeer, rowBytes, dstStart,
                                                 srcStart, rows);
        }
    }
    pipe_barrier(PIPE_ALL);
    dsb(DSB_DDR);
    StoreScalarI32(workspaceView.dispatchGroupReady + localExpert * 16U, 1);
}

AICORE inline void M3N5GatherDispatchToGmm1InputShardByExpert(
    moe_new_dispatch_combine_a8w8::ShapeConfig shape, M2WorkspaceViewDevice workspaceView,
    M2PeerWindowViewDevice localPeer, __gm__ HcclDeviceContext *ctx, GM_ADDR peerWindow, uint32_t myRank,
    uint32_t rowBytes, const moe_new_dispatch_combine_a8w8::PeerWindowLayout &peerWindowLayout, uint32_t workerId,
    uint32_t workerCount, bool activeWorker, __gm__ int32_t *stageStatus, bool requireAicStartAck)
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
        moe_new_dispatch_combine_a8w8::M3N5SignalDispatchExpertReady(localExpert);
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
                                         moe_new_dispatch_combine_a8w8::RankConfig rank,
                                         moe_new_dispatch_combine_a8w8::ShapeConfig shape, uint32_t localExpert,
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
        workspaceView.timeoutDump + localExpert * moe_new_dispatch_combine_a8w8::kM3N9TimeoutDumpStride;
    StoreScalarI32(dump + moe_new_dispatch_combine_a8w8::kM3N9TimeoutDumpPresentSlot, 1);
    StoreScalarI32(dump + moe_new_dispatch_combine_a8w8::kM3N9TimeoutDumpRankSlot, static_cast<int32_t>(rank.rankId));
    StoreScalarI32(dump + moe_new_dispatch_combine_a8w8::kM3N9TimeoutDumpExpertSlot, static_cast<int32_t>(localExpert));
    StoreScalarI32(dump + moe_new_dispatch_combine_a8w8::kM3N9TimeoutDumpTokenOwnerRankSlot,
                   static_cast<int32_t>(tokenOwnerRank));
    StoreScalarI32(dump + moe_new_dispatch_combine_a8w8::kM3N9TimeoutDumpExpertOwnerRankSlot,
                   static_cast<int32_t>(rank.rankId));
    StoreScalarI32(dump + moe_new_dispatch_combine_a8w8::kM3N9TimeoutDumpStageSlot, static_cast<int32_t>(stage));
    StoreScalarI32(dump + moe_new_dispatch_combine_a8w8::kM3N9TimeoutDumpSignalIdSlot, static_cast<int32_t>(signalId));
    StoreScalarI32(dump + moe_new_dispatch_combine_a8w8::kM3N9TimeoutDumpDebugStopStageSlot,
                   static_cast<int32_t>(debugStopStage));
    StoreScalarI32(dump + moe_new_dispatch_combine_a8w8::kM3N9TimeoutDumpDispatchReadySlot,
                   LoadScalarI32(workspaceView.dispatchGroupReady + localExpert * 16U));
    StoreScalarI32(dump + moe_new_dispatch_combine_a8w8::kM3N9TimeoutDumpGmm1ReadySlot,
                   LoadScalarI32(workspaceView.gmm1SyncGroupReady));
    StoreScalarI32(dump + moe_new_dispatch_combine_a8w8::kM3N9TimeoutDumpActivationReadySlot,
                   LoadScalarI32(workspaceView.activationSyncGroupReady));
    StoreScalarI32(dump + moe_new_dispatch_combine_a8w8::kM3N9TimeoutDumpGmm2ReadySlot,
                   LoadScalarI32(workspaceView.gmm2GroupReady + localExpert * 16U));
    StoreScalarI32(dump + moe_new_dispatch_combine_a8w8::kM3N9TimeoutDumpReadyExpertSlot,
                   static_cast<int32_t>(localExpert));
    StoreScalarI32(localPeer.debugCounters + kM3CounterBase + 15U,
                   LoadScalarI32(localPeer.debugCounters + kM3CounterBase + 15U) + 1);
    pipe_barrier(PIPE_ALL);
    dsb(DSB_DDR);
}

AICORE inline uint32_t M3NAssignDispatchWorkers(moe_new_dispatch_combine_a8w8::ShapeConfig shape,
                                                M2PeerWindowViewDevice localPeer, uint32_t logicalAivCount)
{
    uint32_t targetWorkers = logicalAivCount;
    if (targetWorkers > kM3NFusedDispatchMaxLaneSlots) {
        targetWorkers = kM3NFusedDispatchMaxLaneSlots;
    }
    if (targetWorkers > kM3NDispatchMaxWorkers) {
        targetWorkers = kM3NDispatchMaxWorkers;
    }
    uint32_t globalExpertNum = shape.rankNum * shape.expertPerRank;
    if (logicalAivCount == 0U || globalExpertNum == 0U || !M3NDispatchScratchFits(shape)) {
        targetWorkers = 1U;
    } else {
        constexpr uint32_t kMinRoutesPerDispatchWorker = 64U;
        uint32_t routeCount = shape.m * shape.topK;
        uint32_t maxUsefulWorkers = (routeCount + kMinRoutesPerDispatchWorker - 1U) / kMinRoutesPerDispatchWorker;
        if (maxUsefulWorkers == 0U) {
            maxUsefulWorkers = 1U;
        }
        if (targetWorkers > maxUsefulWorkers) {
            targetWorkers = maxUsefulWorkers;
        }
    }
    if (targetWorkers == 0U) {
        targetWorkers = 1U;
    }
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

struct M3ORestoreShardStats {
    uint32_t tokenCount;
    uint32_t routeCount;
    uint32_t skippedRouteCount;
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

AICORE inline M3NDispatchWorkerAssignment M3OLoadDenseWorkerAssignment(uint32_t rawAivSlot, uint32_t logicalAivCount,
                                                                       uint32_t targetWorkerCount)
{
    M3NDispatchWorkerAssignment assignment{0U, targetWorkerCount == 0U ? 1U : targetWorkerCount, false};
    uint32_t availableWorkers = logicalAivCount;
    if (availableWorkers > kM3NFusedDispatchMaxLaneSlots) {
        availableWorkers = kM3NFusedDispatchMaxLaneSlots;
    }
    if (availableWorkers > assignment.workerCount) {
        availableWorkers = assignment.workerCount;
    }
    if (availableWorkers == 0U) {
        availableWorkers = 1U;
    }
    assignment.workerCount = availableWorkers;
    if (rawAivSlot < availableWorkers) {
        assignment.workerId = rawAivSlot;
        assignment.active = true;
    }
    return assignment;
}

AICORE inline void M3NRecordActivationLaneDebug(M2PeerWindowViewDevice localPeer, uint32_t rawAivSlot,
                                                uint32_t workerId, uint32_t rowsProcessed, uint32_t groupCount,
                                                bool active)
{
    (void)workerId;
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
        if (LoadScalarI32(slot + 7U) == 0) {
            continue;
        }
        summary.rowsProcessed += static_cast<uint32_t>(LoadScalarI32(slot + 5U));
        ++summary.activeWorkers;
        uint32_t workerId = raw;
        if (workerId < 31U) {
            summary.activeWorkerMask |= (1U << workerId);
        }
    }
    return summary;
}

AICORE inline void M3OClearRestoreCounters(M2PeerWindowViewDevice localPeer)
{
    for (uint32_t idx = 0; idx < kM2FusedFullM3ORestoreCounterWords; ++idx) {
        StoreScalarI32(localPeer.debugCounters + kM2FusedFullM3ORestoreCounterBase + idx, 0);
    }
    constexpr uint32_t kWordsPerCacheLine = 16U;
    for (uint32_t line = 0; line < kM2FusedFullM3ORestoreCounterWords / kWordsPerCacheLine; ++line) {
        dcci(static_cast<__gm__ void *>(localPeer.debugCounters + kM2FusedFullM3ORestoreCounterBase + line * 16U),
             SINGLE_CACHE_LINE);
    }
    dsb(DSB_DDR);
}

AICORE inline void M3ORecordRestoreWorker(M2PeerWindowViewDevice localPeer, uint32_t workerId, uint32_t tokenBegin,
                                          uint32_t tokenEnd, M3ORestoreShardStats stats)
{
    if (workerId >= moe_new_dispatch_combine_a8w8::kM3ORestoreWorkerCount) {
        return;
    }
    uint32_t base = kM2FusedFullM3ORestoreCounterBase + kM2FusedFullM3ORestoreWorkerBase +
                    workerId * kM2FusedFullM3ORestoreWorkerWords;
    StoreScalarI32(localPeer.debugCounters + base + 0U, static_cast<int32_t>(tokenBegin));
    StoreScalarI32(localPeer.debugCounters + base + 1U, static_cast<int32_t>(tokenEnd));
    StoreScalarI32(localPeer.debugCounters + base + 2U, static_cast<int32_t>(stats.tokenCount));
    StoreScalarI32(localPeer.debugCounters + base + 3U, static_cast<int32_t>(stats.routeCount));
    StoreScalarI32(localPeer.debugCounters + base + 4U, static_cast<int32_t>(stats.skippedRouteCount));
    StoreScalarI32(localPeer.debugCounters + base + 5U, 1);
    dcci(static_cast<__gm__ void *>(localPeer.debugCounters + base), SINGLE_CACHE_LINE);
    dsb(DSB_DDR);
}

AICORE inline void M3OFinalizeRestoreCounters(M2PeerWindowViewDevice localPeer, uint32_t activeWorkerCount,
                                              uint32_t totalTokens, uint32_t totalRoutes, uint32_t skippedRoutes)
{
    StoreScalarI32(localPeer.debugCounters + kM2FusedFullM3ORestoreCounterBase + 0U,
                   static_cast<int32_t>(activeWorkerCount));
    StoreScalarI32(localPeer.debugCounters + kM2FusedFullM3ORestoreCounterBase + 1U, static_cast<int32_t>(totalTokens));
    StoreScalarI32(localPeer.debugCounters + kM2FusedFullM3ORestoreCounterBase + 2U, static_cast<int32_t>(totalRoutes));
    StoreScalarI32(localPeer.debugCounters + kM2FusedFullM3ORestoreCounterBase + 3U,
                   static_cast<int32_t>(skippedRoutes));
    StoreScalarI32(localPeer.debugCounters + kM2FusedFullM3ORestoreCounterBase + 4U, 1);
    StoreScalarI32(localPeer.debugCounters + kM2FusedFullM3ORestoreCounterBase + 5U,
                   static_cast<int32_t>(moe_new_dispatch_combine_a8w8::kM3ORestoreWorkerCount));
    dcci(static_cast<__gm__ void *>(localPeer.debugCounters + kM2FusedFullM3ORestoreCounterBase), SINGLE_CACHE_LINE);
    dsb(DSB_DDR);
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

AICORE inline moe_new_dispatch_combine_a8w8::ShapeConfig M2FusedLoadShapeConfig(__gm__ int32_t *config)
{
    return moe_new_dispatch_combine_a8w8::ShapeConfig{
        static_cast<uint32_t>(
            M2FusedLoadConfigI32(config, moe_new_dispatch_combine_a8w8::kM2FusedFullShapeRankNumSlot)),
        static_cast<uint32_t>(
            M2FusedLoadConfigI32(config, moe_new_dispatch_combine_a8w8::kM2FusedFullShapeExpertPerRankSlot)),
        static_cast<uint32_t>(M2FusedLoadConfigI32(config, moe_new_dispatch_combine_a8w8::kM2FusedFullShapeTopKSlot)),
        static_cast<uint32_t>(M2FusedLoadConfigI32(config, moe_new_dispatch_combine_a8w8::kM2FusedFullShapeMSlot)),
        static_cast<uint32_t>(
            M2FusedLoadConfigI32(config, moe_new_dispatch_combine_a8w8::kM2FusedFullShapeHiddenSizeSlot)),
        static_cast<uint32_t>(
            M2FusedLoadConfigI32(config, moe_new_dispatch_combine_a8w8::kM2FusedFullShapeIntermediateSizeSlot)),
        static_cast<uint32_t>(
            M2FusedLoadConfigI32(config, moe_new_dispatch_combine_a8w8::kM2FusedFullShapeMaxTokensPerExpertSlot)),
        static_cast<uint32_t>(
            M2FusedLoadConfigI32(config, moe_new_dispatch_combine_a8w8::kM2FusedFullShapePayloadTileColsSlot)),
        static_cast<uint32_t>(
            M2FusedLoadConfigI32(config, moe_new_dispatch_combine_a8w8::kM2FusedFullShapeGmmBlockMSlot)),
        static_cast<uint32_t>(
            M2FusedLoadConfigI32(config, moe_new_dispatch_combine_a8w8::kM2FusedFullShapeGmmBlockNSlot)),
        static_cast<uint32_t>(
            M2FusedLoadConfigI32(config, moe_new_dispatch_combine_a8w8::kM2FusedFullShapeGmmBlockKSlot)),
        static_cast<uint32_t>(
            M2FusedLoadConfigI32(config, moe_new_dispatch_combine_a8w8::kM2FusedFullShapeDtypeInSlot)),
        static_cast<uint32_t>(
            M2FusedLoadConfigI32(config, moe_new_dispatch_combine_a8w8::kM2FusedFullShapeDtypeOutSlot)),
    };
}

AICORE inline moe_new_dispatch_combine_a8w8::RankConfig M2FusedLoadRankConfig(__gm__ int32_t *config)
{
    return moe_new_dispatch_combine_a8w8::RankConfig{
        static_cast<uint32_t>(M2FusedLoadConfigI32(config, moe_new_dispatch_combine_a8w8::kM2FusedFullRankRankNumSlot)),
        static_cast<uint32_t>(M2FusedLoadConfigI32(config, moe_new_dispatch_combine_a8w8::kM2FusedFullRankRankIdSlot)),
        static_cast<uint32_t>(M2FusedLoadConfigI32(config, moe_new_dispatch_combine_a8w8::kM2FusedFullRankFromMpiSlot)),
        static_cast<uint32_t>(
            M2FusedLoadConfigI32(config, moe_new_dispatch_combine_a8w8::kM2FusedFullRankDeviceBaseSlot)),
        static_cast<uint32_t>(
            M2FusedLoadConfigI32(config, moe_new_dispatch_combine_a8w8::kM2FusedFullRankNdevicesSlot)),
    };
}

AICORE inline uint32_t M2FusedLoadDebugStopStage(
    __gm__ moe_new_dispatch_combine_a8w8::M2FusedFullLaunchArgs *launchArgs)
{
    return launchArgs->debugStopStage;
}

#if defined(M2_MIXED_SPIKE_BUILD_AIV)
AICORE inline int32_t M2FusedFloatToBits(float value)
{
    union {
        float f;
        uint32_t u;
    } bits{value};
    return static_cast<int32_t>(bits.u);
}

AICORE inline void M2FusedStoreFloatScalar(__gm__ float *ptr, float value)
{
    StoreScalarI32(reinterpret_cast<__gm__ int32_t *>(ptr), M2FusedFloatToBits(value));
}

AICORE inline void M2RoutePackQuantLocalPtoVec(moe_new_dispatch_combine_a8w8::ShapeConfig shape,
                                               M2WorkspaceViewDevice workspaceView, M2PeerWindowViewDevice localPeer,
                                               GM_ADDR inputA, GM_ADDR expertIdx, GM_ADDR xActiveMask,
                                               uint32_t rowBytes)
{
    uint32_t globalExpertNum = shape.rankNum * shape.expertPerRank;
    uint32_t localRows = static_cast<uint32_t>(M2LocalRows(shape));
    __gm__ int32_t *expertIds = reinterpret_cast<__gm__ int32_t *>(expertIdx);
    for (uint32_t globalExpert = 0; globalExpert < globalExpertNum; ++globalExpert) {
        StoreScalarI32(workspaceView.tokenOwnerRankOffsets + globalExpert,
                       LoadScalarI32(workspaceView.blockPrefixPerExpert + globalExpert));
    }
    for (uint32_t token = 0; token < shape.m; ++token) {
        for (uint32_t slot = 0; slot < shape.topK; ++slot) {
            uint32_t routeIndex = token * shape.topK + slot;
            if (!M2TokenIsActive(xActiveMask, token)) {
                StoreScalarI32(workspaceView.expandedRowIdx + routeIndex, static_cast<int32_t>(localRows));
                continue;
            }
            int32_t expert = LoadScalarI32(expertIds + routeIndex);
            if (expert < 0 || static_cast<uint32_t>(expert) >= globalExpertNum) {
                StoreScalarI32(workspaceView.expandedRowIdx + routeIndex, -1);
                continue;
            }
            uint32_t globalExpert = static_cast<uint32_t>(expert);
            int32_t packedRow = LoadScalarI32(workspaceView.tokenOwnerRankOffsets + globalExpert);
            StoreScalarI32(workspaceView.tokenOwnerRankOffsets + globalExpert, packedRow + 1);
            if (packedRow < 0 || static_cast<uint32_t>(packedRow) >= localRows) {
                StoreScalarI32(workspaceView.expandedRowIdx + routeIndex, static_cast<int32_t>(localRows));
                continue;
            }
            StoreScalarI32(workspaceView.expandedRowIdx + routeIndex, packedRow);
            StoreScalarI32(workspaceView.packedRowToRouteIndex + static_cast<uint32_t>(packedRow),
                           static_cast<int32_t>(routeIndex));
            M2QuantizeRowToPeerPayload(shape, localPeer, inputA, token, static_cast<uint32_t>(packedRow), rowBytes);
        }
    }
    InvalidateGmCacheLines(workspaceView.expandedRowIdx, static_cast<uint32_t>(shape.m * shape.topK * sizeof(int32_t)));
    InvalidateGmCacheLines(workspaceView.packedRowToRouteIndex,
                           static_cast<uint32_t>(shape.m * shape.topK * sizeof(int32_t)));
}

AICORE inline void M3NRoutePackQuantLocalShardPtoVec(moe_new_dispatch_combine_a8w8::ShapeConfig shape,
                                                     M2WorkspaceViewDevice workspaceView,
                                                     M2PeerWindowViewDevice localPeer, GM_ADDR inputA,
                                                     GM_ADDR expertIdx, GM_ADDR xActiveMask, uint32_t rowBytes,
                                                     uint32_t workerId, uint32_t workerCount)
{
    uint32_t globalExpertNum = shape.rankNum * shape.expertPerRank;
    uint32_t workerStride = M3NDispatchWorkerScratchStride(globalExpertNum);
    uint32_t localRows = static_cast<uint32_t>(M2LocalRows(shape));
    __gm__ int32_t *expertIds = reinterpret_cast<__gm__ int32_t *>(expertIdx);
    InvalidateGmCacheLines(M3NDispatchWorkerCountScratch(workspaceView),
                           static_cast<uint32_t>(workerCount * workerStride * sizeof(int32_t)));
    InvalidateGmCacheLines(M3NDispatchWorkerPrefixScratch(workspaceView),
                           static_cast<uint32_t>(workerCount * workerStride * sizeof(int32_t)));
    __gm__ int32_t *localOrdinalCursor =
        M3NDispatchWorkerCountScratch(workspaceView) + static_cast<uint64_t>(workerId) * workerStride;
    for (uint32_t idx = 0; idx < workerStride; ++idx) {
        StoreScalarI32(localOrdinalCursor + idx, 0);
    }
    uint32_t tokenBegin = TokenShardBegin(shape.m, workerId, workerCount);
    uint32_t tokenEnd = TokenShardEnd(shape.m, workerId, workerCount);
    for (uint32_t token = tokenBegin; token < tokenEnd; ++token) {
        for (uint32_t slot = 0; slot < shape.topK; ++slot) {
            uint32_t routeIndex = token * shape.topK + slot;
            if (!M2TokenIsActive(xActiveMask, token)) {
                StoreScalarI32(workspaceView.expandedRowIdx + routeIndex, static_cast<int32_t>(localRows));
                continue;
            }
            int32_t expert = LoadScalarI32(expertIds + routeIndex);
            if (expert < 0 || static_cast<uint32_t>(expert) >= globalExpertNum) {
                StoreScalarI32(workspaceView.expandedRowIdx + routeIndex, -1);
                continue;
            }
            uint32_t globalExpert = static_cast<uint32_t>(expert);
            int32_t packedRow = LoadScalarI32(workspaceView.blockPrefixPerExpert + globalExpert) +
                                M3NDispatchLoadWorkerExpertPrefix(shape, workspaceView, workerId, globalExpert) +
                                LoadScalarI32(localOrdinalCursor + globalExpert);
            StoreScalarI32(localOrdinalCursor + globalExpert, LoadScalarI32(localOrdinalCursor + globalExpert) + 1);
            if (packedRow < 0 || static_cast<uint32_t>(packedRow) >= localRows) {
                StoreScalarI32(workspaceView.expandedRowIdx + routeIndex, static_cast<int32_t>(localRows));
                continue;
            }
            StoreScalarI32(workspaceView.expandedRowIdx + routeIndex, packedRow);
            StoreScalarI32(workspaceView.packedRowToRouteIndex + static_cast<uint32_t>(packedRow),
                           static_cast<int32_t>(routeIndex));
        }
        M2QuantizeTokenToPackedRows(shape, workspaceView, localPeer, inputA, token, rowBytes);
    }
    InvalidateGmCacheLines(workspaceView.expandedRowIdx + tokenBegin * shape.topK,
                           static_cast<uint32_t>((tokenEnd - tokenBegin) * shape.topK * sizeof(int32_t)));
    InvalidateGmCacheLines(workspaceView.packedRowToRouteIndex,
                           static_cast<uint32_t>(shape.m * shape.topK * sizeof(int32_t)));
}

PTO_INTERNAL void M2FusedBasicVecProbe(__gm__ int32_t *stageStatus, uint32_t slot, int32_t value)
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

PTO_INTERNAL void M2FusedSetValueProbe(__gm__ int32_t *stageStatus, uint32_t slot, int32_t value, uint64_t ubAddr)
{
    M2RouteRowStatTile probeTile;
    TASSIGN(probeTile, ubAddr);
    probeTile.SetValue(0, static_cast<float>(value));
    pipe_barrier(PIPE_ALL);
    M2FusedRecordStage(stageStatus + kM2FusedFullStageBaseSlot, slot, value);
}

PTO_INTERNAL void M2FusedExpandOnlyProbe(__gm__ int32_t *stageStatus, uint32_t slot, int32_t value, uint64_t ubAddr)
{
    using ProbeTile = pto::Tile<pto::TileType::Vec, int32_t, 1, 16, pto::BLayout::RowMajor, 1, 16>;
    ProbeTile probeTile;
    TASSIGN(probeTile, ubAddr);
    TEXPANDS(probeTile, value);
    pipe_barrier(PIPE_V);
    M2FusedRecordStage(stageStatus + kM2FusedFullStageBaseSlot, slot, value);
}

PTO_INTERNAL void M2FusedAssignOnlyProbe(__gm__ int32_t *stageStatus, uint32_t slot, int32_t value, uint64_t ubAddr)
{
    M2RouteRowStatTile probeTile;
    TASSIGN(probeTile, ubAddr);
    M2FusedRecordStage(stageStatus + kM2FusedFullStageBaseSlot, slot, value);
}

PTO_INTERNAL void M2FusedTileConstructOnlyProbe(__gm__ int32_t *stageStatus, uint32_t slot, int32_t value)
{
    M2RouteRowStatTile probeTile;
    probeTile.SetKAligned(true);
    int32_t recorded = probeTile.GetKAligned() ? value : -value;
    M2FusedRecordStage(stageStatus + kM2FusedFullStageBaseSlot, slot, recorded);
}

struct M2FusedPlainVecTileLike {
    __ubuf__ float *data;
    bool isKAligned;
};

PTO_INTERNAL void M2FusedPlainVecTileLikeProbe(__gm__ int32_t *stageStatus, uint32_t slot, int32_t value)
{
    M2FusedPlainVecTileLike probe;
    probe.data = reinterpret_cast<__ubuf__ float *>(kM2FusedVecProbeUbAddr);
    probe.isKAligned = true;
    int32_t recorded = (probe.data != nullptr && probe.isKAligned) ? value : -value;
    M2FusedRecordStage(stageStatus + kM2FusedFullStageBaseSlot, slot, recorded);
}

AICORE inline void M2FusedRawVectorDupProbe(__gm__ int32_t *stageStatus, uint32_t slot, int32_t value, uint64_t ubAddr)
{
    __ubuf__ int32_t *ub = reinterpret_cast<__ubuf__ int32_t *>(ubAddr);
    set_mask_count();
    pto::SetVectorCount(16);
    vector_dup(ub, value, 0, 1, 1, 8, 8);
    set_mask_norm();
    pto::SetFullVecMaskByDType<int32_t>();
    pipe_barrier(PIPE_V);
    M2FusedRecordStage(stageStatus + kM2FusedFullStageBaseSlot, slot, value);
}

template <typename ProbeTile>
__tf__ PTO_INTERNAL void M2FusedTileRawVectorDup(typename ProbeTile::TileDType __out__ data, int32_t value)
{
    __ubuf__ int32_t *ub = reinterpret_cast<__ubuf__ int32_t *>(__cce_get_tile_ptr(data));
    set_mask_count();
    pto::SetVectorCount(16);
    vector_dup(ub, value, 0, 1, 1, 8, 8);
    set_mask_norm();
    pto::SetFullVecMaskByDType<int32_t>();
}

template <typename ProbeTile>
__tf__ PTO_INTERNAL void M2FusedTileRawVectorDupByRef(ProbeTile &tile, int32_t value)
{
    __ubuf__ int32_t *ub = reinterpret_cast<__ubuf__ int32_t *>(__cce_get_tile_ptr(tile.data()));
    set_mask_count();
    pto::SetVectorCount(16);
    vector_dup(ub, value, 0, 1, 1, 8, 8);
    set_mask_norm();
    pto::SetFullVecMaskByDType<int32_t>();
}

PTO_INTERNAL void M2FusedTileRawVectorDupProbe(__gm__ int32_t *stageStatus, uint32_t slot, int32_t value,
                                               uint64_t ubAddr)
{
    using ProbeTile = pto::Tile<pto::TileType::Vec, int32_t, 1, 16, pto::BLayout::RowMajor, 1, 16>;
    ProbeTile probeTile;
    TASSIGN(probeTile, ubAddr);
    M2FusedTileRawVectorDup<ProbeTile>(probeTile.data(), value);
    pipe_barrier(PIPE_V);
    M2FusedRecordStage(stageStatus + kM2FusedFullStageBaseSlot, slot, value);
}

PTO_INTERNAL void M2FusedTileRawVectorDupByRefProbe(__gm__ int32_t *stageStatus, uint32_t slot, int32_t value,
                                                    uint64_t ubAddr)
{
    using ProbeTile = pto::Tile<pto::TileType::Vec, int32_t, 1, 16, pto::BLayout::RowMajor, 1, 16>;
    ProbeTile probeTile;
    TASSIGN(probeTile, ubAddr);
    M2FusedTileRawVectorDupByRef<ProbeTile>(probeTile, value);
    pipe_barrier(PIPE_V);
    M2FusedRecordStage(stageStatus + kM2FusedFullStageBaseSlot, slot, value);
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
    __gm__ uint64_t *fftsAddr, __gm__ moe_new_dispatch_combine_a8w8::M2FusedFullLaunchArgs *launchArgs)
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
            M2FusedLoadConfigI32(earlyStageStatus, moe_new_dispatch_combine_a8w8::kM2FusedFullShapeRankNumSlot);
        int32_t hidden =
            M2FusedLoadConfigI32(earlyStageStatus, moe_new_dispatch_combine_a8w8::kM2FusedFullShapeHiddenSizeSlot);
        int32_t intermediate = M2FusedLoadConfigI32(
            earlyStageStatus, moe_new_dispatch_combine_a8w8::kM2FusedFullShapeIntermediateSizeSlot);
        int32_t blockK =
            M2FusedLoadConfigI32(earlyStageStatus, moe_new_dispatch_combine_a8w8::kM2FusedFullShapeGmmBlockKSlot);
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
        for (uint32_t slot = moe_new_dispatch_combine_a8w8::kM2FusedFullShapeRankNumSlot;
             slot <= moe_new_dispatch_combine_a8w8::kM2FusedFullShapeDtypeOutSlot; ++slot) {
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
            moe_new_dispatch_combine_a8w8::ShapeConfig shape = M2FusedLoadShapeConfig(earlyStageStatus);
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
        moe_new_dispatch_combine_a8w8::ShapeConfig shape = M2FusedLoadShapeConfig(earlyStageStatus);
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
        moe_new_dispatch_combine_a8w8::RankConfig rank = M2FusedLoadRankConfig(earlyStageStatus);
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
        moe_new_dispatch_combine_a8w8::ShapeConfig shape = M2FusedLoadShapeConfig(stageStatus);
        moe_new_dispatch_combine_a8w8::RankConfig rank = M2FusedLoadRankConfig(stageStatus);
        if (get_block_idx() == 0) {
            M2FusedRecordStage(stageStatus + kM2FusedFullAicHeaderSlot, 0U, kM2FusedFullMagic);
            M2FusedRecordStage(stageStatus + kM2FusedFullAicHeaderSlot, 1U, static_cast<int32_t>(get_block_num()));
            M2FusedRecordStage(stageStatus + kM2FusedFullAicHeaderSlot, 2U, static_cast<int32_t>(rank.rankId));
            M2FusedRecordStage(stageStatus + kM2FusedFullAicHeaderSlot, 3U, static_cast<int32_t>(shape.rankNum));
        }
        pto::SYNCALL<pto::SyncCoreType::Mix>();
        return;
    }
    set_ffts_base_addr(reinterpret_cast<uint64_t>(fftsAddr));
    __gm__ int32_t *stageStatus = reinterpret_cast<__gm__ int32_t *>(launchArgs->stageStatusAddr);
    moe_new_dispatch_combine_a8w8::ShapeConfig shape = M2FusedLoadShapeConfig(stageStatus);
    moe_new_dispatch_combine_a8w8::RankConfig rank = M2FusedLoadRankConfig(stageStatus);
    GM_ADDR workspace = reinterpret_cast<GM_ADDR>(
        M2FusedLoadConfigU64(stageStatus, moe_new_dispatch_combine_a8w8::kM2FusedFullPtrWorkspaceSlot));
    GM_ADDR peerWindow = reinterpret_cast<GM_ADDR>(
        M2FusedLoadConfigU64(stageStatus, moe_new_dispatch_combine_a8w8::kM2FusedFullPtrPeerWindowSlot));
    auto layout = M2FusedGmmMakeWorkspaceLayoutDevice(shape);
    __gm__ int32_t *debugCounters = M3NFusedPeerDebugCounters(peerWindow, shape);
    uint32_t debugStopStage = static_cast<uint32_t>(M2FusedGmmLoadScalarI32(stageStatus + kM2FusedFullDebugStopSlot));
    if (debugStopStage == 0U) {
        debugStopStage = M2FusedLoadDebugStopStage(launchArgs);
    }
    uint32_t overlapMode = static_cast<uint32_t>(
        M2FusedLoadConfigI32(stageStatus, moe_new_dispatch_combine_a8w8::kM2FusedFullOverlapModeSlot));
    uint32_t timelineEnable = static_cast<uint32_t>(
        M2FusedLoadConfigI32(stageStatus, moe_new_dispatch_combine_a8w8::kM2FusedFullTimelineEnableSlot));
    __gm__ uint64_t *timelineScratch = reinterpret_cast<__gm__ uint64_t *>(workspace + layout.timelineScratch.offset);
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
    bool m3oActivationShardDebugProbe = M3OActivationShardDebugProbeEnabled(debugStopStage);
    M3NDispatchAicWaitForAivSubphases(debugStopStage, m3n5DispatchGmm1Overlap, shape);
    if (debugStopStage == 17U) {
        pto::SYNCALL<pto::SyncCoreType::Mix>();
        return;
    }
    if (!m3n5DispatchGmm1Overlap) {
        moe_new_dispatch_combine_a8w8::M3N4WaitV2C<moe_new_dispatch_combine_a8w8::kM3N4DispatchToGmm1Flag>();
    }
    if (debugStopStage == 1U) {
        if (m3n5DispatchGmm1Overlap) {
            M3N5WaitAllDispatchExpertsReady(shape);
        }
        return;
    }
    M2FusedRecordStage(stageStatus + kM2FusedFullAicStageBaseSlot, static_cast<uint32_t>(get_block_idx()), 100);
    if (shape.rankNum != 0 && shape.expertPerRank != 0 && shape.hiddenSize != 0U && shape.intermediateSize != 0U &&
        shape.gmmBlockM == moe_new_dispatch_combine_a8w8::kGmmBaseM &&
        shape.gmmBlockN == moe_new_dispatch_combine_a8w8::kGmmBaseN &&
        shape.gmmBlockK == moe_new_dispatch_combine_a8w8::kGmmBaseK &&
        shape.hiddenSize % moe_new_dispatch_combine_a8w8::kGmmBaseK == 0U &&
        shape.intermediateSize % moe_new_dispatch_combine_a8w8::kGmmBaseK == 0U) {
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
        M2FusedClearGmmTaskEvidence(debugCounters, kM2FusedFullM3OGmm1CounterBase, taskCount);
        bool m3n12Gmm1TileRecorded = false;
        if (timelineEnable != 0U && get_block_idx() == 0 && taskCount == 0U) {
            M3N12RecordSkipped(timelineScratch, moe_new_dispatch_combine_a8w8::kM3N12TimelineSlotGmm1,
                               moe_new_dispatch_combine_a8w8::M3N12TimelineKind::kGmm1Tile,
                               moe_new_dispatch_combine_a8w8::M3N12TimelineCoreType::kAic,
                               static_cast<uint32_t>(get_block_idx()));
            m3n12Gmm1TileRecorded = true;
        }
        if (m3n6Gmm1ActivationOverlap) {
            uint32_t syncGroupCount = moe_new_dispatch_combine_a8w8::M3N6SwigluSyncGroupCount(shape);
            for (uint32_t syncIdx = 0; syncIdx < syncGroupCount; ++syncIdx) {
                uint32_t expertBegin = 0;
                uint32_t expertEnd = 0;
                M3N6SyncGroupExpertRange(shape, syncIdx, &expertBegin, &expertEnd);
                for (uint32_t currentExpert = expertBegin; currentExpert < expertEnd; ++currentExpert) {
                    if (m3n5DispatchGmm1Overlap) {
                        moe_new_dispatch_combine_a8w8::M3N5WaitDispatchExpertReady(currentExpert);
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
                        bool recordTile = timelineEnable != 0U && get_block_idx() == 0 && !m3n12Gmm1TileRecorded;
                        uint64_t tileBegin = recordTile ? M3N12GetSysCnt() : 0U;
                        M2FusedRunInt8GmmTileScalar(tileInput, expertWeight + nBase, tileOutput + nBase, mValid,
                                                    shape.hiddenSize, nValid, rowBytes, w1Cols, w1Cols);
                        if (recordTile) {
                            uint32_t waitSource = static_cast<uint32_t>(
                                moe_new_dispatch_combine_a8w8::M3N12TimelineWaitSource::kPtoEvent);
                            M3N12RecordTimeline(
                                timelineScratch, moe_new_dispatch_combine_a8w8::kM3N12TimelineSlotGmm1, tileBegin,
                                M3N12GetSysCnt(),
                                M3N12PackMeta0(moe_new_dispatch_combine_a8w8::M3N12TimelineKind::kGmm1Tile,
                                               moe_new_dispatch_combine_a8w8::M3N12TimelineCoreType::kAic,
                                               static_cast<uint32_t>(get_block_idx()),
                                               moe_new_dispatch_combine_a8w8::M3N12TimelineStatus::kProcessed, syncIdx,
                                               taskId),
                                M3N12PackMeta1(rowBegin / moe_new_dispatch_combine_a8w8::kGmmBaseM,
                                               nBase / moe_new_dispatch_combine_a8w8::kGmmBaseN,
                                               shape.hiddenSize / moe_new_dispatch_combine_a8w8::kGmmBaseK,
                                               waitSource));
                            m3n12Gmm1TileRecorded = true;
                        }
                    }
                }
                moe_new_dispatch_combine_a8w8::M3N4AicAllDoneCoarseSync();
                if (get_block_idx() == 0) {
                    M3N6RecordGmm1SyncGroupReady(stageStatus, layout, workspace, syncIdx, syncGroupCount);
                }
                moe_new_dispatch_combine_a8w8::M3N6SignalGmm1SyncGroupReady(syncIdx);
            }
        } else if (m3n5DispatchGmm1Overlap) {
            for (uint32_t currentExpert = 0; currentExpert < shape.expertPerRank; ++currentExpert) {
                moe_new_dispatch_combine_a8w8::M3N5WaitDispatchExpertReady(currentExpert);
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
                    bool recordTile = timelineEnable != 0U && get_block_idx() == 0 && !m3n12Gmm1TileRecorded;
                    uint64_t tileBegin = recordTile ? M3N12GetSysCnt() : 0U;
                    M2FusedRunInt8GmmTileScalar(tileInput, expertWeight + nBase, tileOutput + nBase, mValid,
                                                shape.hiddenSize, nValid, rowBytes, w1Cols, w1Cols);
                    if (recordTile) {
                        uint32_t waitSource =
                            static_cast<uint32_t>(moe_new_dispatch_combine_a8w8::M3N12TimelineWaitSource::kPtoEvent);
                        M3N12RecordTimeline(
                            timelineScratch, moe_new_dispatch_combine_a8w8::kM3N12TimelineSlotGmm1, tileBegin,
                            M3N12GetSysCnt(),
                            M3N12PackMeta0(moe_new_dispatch_combine_a8w8::M3N12TimelineKind::kGmm1Tile,
                                           moe_new_dispatch_combine_a8w8::M3N12TimelineCoreType::kAic,
                                           static_cast<uint32_t>(get_block_idx()),
                                           moe_new_dispatch_combine_a8w8::M3N12TimelineStatus::kProcessed,
                                           currentExpert, taskId),
                            M3N12PackMeta1(rowBegin / moe_new_dispatch_combine_a8w8::kGmmBaseM,
                                           nBase / moe_new_dispatch_combine_a8w8::kGmmBaseN,
                                           shape.hiddenSize / moe_new_dispatch_combine_a8w8::kGmmBaseK, waitSource));
                        m3n12Gmm1TileRecorded = true;
                    }
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
                bool recordTile = timelineEnable != 0U && get_block_idx() == 0 && !m3n12Gmm1TileRecorded;
                uint64_t tileBegin = recordTile ? M3N12GetSysCnt() : 0U;
                M2FusedRunInt8GmmTileScalar(tileInput, expertWeight + nBase, tileOutput + nBase, mValid,
                                            shape.hiddenSize, nValid, rowBytes, w1Cols, w1Cols);
                if (recordTile) {
                    M3N12RecordTimeline(
                        timelineScratch, moe_new_dispatch_combine_a8w8::kM3N12TimelineSlotGmm1, tileBegin,
                        M3N12GetSysCnt(),
                        M3N12PackMeta0(moe_new_dispatch_combine_a8w8::M3N12TimelineKind::kGmm1Tile,
                                       moe_new_dispatch_combine_a8w8::M3N12TimelineCoreType::kAic,
                                       static_cast<uint32_t>(get_block_idx()),
                                       moe_new_dispatch_combine_a8w8::M3N12TimelineStatus::kProcessed, 0U, taskId),
                        M3N12PackMeta1(
                            rowBegin / moe_new_dispatch_combine_a8w8::kGmmBaseM,
                            nBase / moe_new_dispatch_combine_a8w8::kGmmBaseN,
                            shape.hiddenSize / moe_new_dispatch_combine_a8w8::kGmmBaseK,
                            static_cast<uint32_t>(moe_new_dispatch_combine_a8w8::M3N12TimelineWaitSource::kSyncAll)));
                    m3n12Gmm1TileRecorded = true;
                }
            }
        }
        M2FusedRecordGmmTaskEvidence(debugCounters, kM2FusedFullM3OGmm1CounterBase, taskCount, gmm1TileTaskPlan);
        moe_new_dispatch_combine_a8w8::M3N4AicAllDoneCoarseSync();
        if (get_block_idx() == 0) {
            M2FusedFinalizeGmmTaskEvidence(debugCounters, kM2FusedFullM3OGmm1CounterBase, taskCount);
        }
        if (debugStopStage == 22U) {
            if (get_block_idx() == 0) {
                M2FusedRecordStage(stageStatus + kM2FusedFullStageBaseSlot, 4U, 22);
            }
            return;
        }
    }
    pipe_barrier(PIPE_ALL);
    dsb(DSB_DDR);
    if (!m3n6Gmm1ActivationOverlap) {
        moe_new_dispatch_combine_a8w8::M3N4AicAllDoneCoarseSync();
        if (debugStopStage == 23U) {
            if (get_block_idx() == 0) {
                M2FusedRecordStage(stageStatus + kM2FusedFullStageBaseSlot, 4U, 23);
            }
            return;
        }
        moe_new_dispatch_combine_a8w8::M3N4SignalC2V<moe_new_dispatch_combine_a8w8::kM3N4Gmm1ToEpilogueFlag>();
        if (debugStopStage == 24U) {
            if (get_block_idx() == 0) {
                M2FusedRecordStage(stageStatus + kM2FusedFullStageBaseSlot, 4U, 24);
            }
            return;
        }
    }
    if (debugStopStage == 2U) {
        return;
    }
    if (!m3n6Gmm1ActivationOverlap) {
        moe_new_dispatch_combine_a8w8::M3N4WaitV2C<moe_new_dispatch_combine_a8w8::kM3N4Gmm1EpilogueToActivationFlag>();
        M3N6WaitAllGmm1SyncGroupsReadyGm(shape, workspace, layout);
        if (m3oActivationShardDebugProbe) {
            if (get_block_idx() == 0) {
                M2FusedRecordStage(stageStatus + kM2FusedFullStageBaseSlot, 4U, static_cast<int32_t>(debugStopStage));
            }
            return;
        }
    }
    if (debugStopStage == 3U) {
        return;
    }
    if (!m3n6Gmm1ActivationOverlap) {
        uint32_t rowsProcessed = M2FusedRunActivationQuantAicScalar(shape, workspace, layout);
        moe_new_dispatch_combine_a8w8::M3N4AicAllDoneCoarseSync();
        if (get_block_idx() == 0) {
            M2FusedFinalizeActivationQuantAicScalar(shape, workspace, layout, debugCounters, rowsProcessed);
        }
    }
    if (debugStopStage == 4U) {
        if (m3n7ActivationGmm2Overlap) {
            __gm__ int32_t *activationSyncGroupReady =
                reinterpret_cast<__gm__ int32_t *>(workspace + layout.activationSyncGroupReady.offset);
            uint32_t syncGroupCount = moe_new_dispatch_combine_a8w8::M3N6SwigluSyncGroupCount(shape);
            for (uint32_t syncIdx = 0; syncIdx < syncGroupCount; ++syncIdx) {
                M3N7WaitActivationSyncGroupReadyGm(activationSyncGroupReady, syncIdx);
            }
        }
        return;
    }
    M2FusedRecordStage(stageStatus + kM2FusedFullAicStageBaseSlot, static_cast<uint32_t>(get_block_idx()), 200);
    if (shape.rankNum != 0 && shape.expertPerRank != 0 && shape.hiddenSize != 0U && shape.intermediateSize != 0U &&
        shape.gmmBlockM == moe_new_dispatch_combine_a8w8::kGmmBaseM &&
        shape.gmmBlockN == moe_new_dispatch_combine_a8w8::kGmmBaseN &&
        shape.gmmBlockK == moe_new_dispatch_combine_a8w8::kGmmBaseK &&
        shape.hiddenSize % moe_new_dispatch_combine_a8w8::kGmmBaseK == 0U &&
        shape.intermediateSize % moe_new_dispatch_combine_a8w8::kGmmBaseK == 0U) {
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
                M3N7InitGmm2Counters(debugCounters, moe_new_dispatch_combine_a8w8::M3N6SwigluSyncGroupCount(shape));
            }
        }
        M2FusedClearGmmTaskEvidence(debugCounters, kM2FusedFullM3OGmm2CounterBase, taskCount);
        bool m3n12Gmm2TileRecorded = false;
        if (timelineEnable != 0U && get_block_idx() == 0 && taskCount == 0U) {
            M3N12RecordSkipped(timelineScratch, moe_new_dispatch_combine_a8w8::kM3N12TimelineSlotGmm2,
                               moe_new_dispatch_combine_a8w8::M3N12TimelineKind::kGmm2Tile,
                               moe_new_dispatch_combine_a8w8::M3N12TimelineCoreType::kAic,
                               static_cast<uint32_t>(get_block_idx()));
            m3n12Gmm2TileRecorded = true;
        }
        for (uint32_t localExpert = 0; localExpert < shape.expertPerRank; ++localExpert) {
            int32_t rowCount = M2FusedGmmLoadScalarI32(expertTokenNums + localExpert);
            if (rowCount <= 0) {
                M2FusedGmmStoreScalarI32(gmm2GroupReady + localExpert * 16U, 1);
            }
        }
        if (m3n7ActivationGmm2Overlap) {
            uint32_t syncGroupCount = moe_new_dispatch_combine_a8w8::M3N6SwigluSyncGroupCount(shape);
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
                    bool recordTile = timelineEnable != 0U && get_block_idx() == 0 && !m3n12Gmm2TileRecorded;
                    uint64_t tileBegin = recordTile ? M3N12GetSysCnt() : 0U;
                    M2FusedRunInt8GmmTileScalar(tileInput, expertWeight + nBase, tileOutput + nBase, mValid,
                                                shape.intermediateSize, nValid, rowBytes, shape.hiddenSize,
                                                shape.hiddenSize);
                    if (recordTile) {
                        M3N12RecordTimeline(
                            timelineScratch, moe_new_dispatch_combine_a8w8::kM3N12TimelineSlotGmm2, tileBegin,
                            M3N12GetSysCnt(),
                            M3N12PackMeta0(moe_new_dispatch_combine_a8w8::M3N12TimelineKind::kGmm2Tile,
                                           moe_new_dispatch_combine_a8w8::M3N12TimelineCoreType::kAic,
                                           static_cast<uint32_t>(get_block_idx()),
                                           moe_new_dispatch_combine_a8w8::M3N12TimelineStatus::kProcessed, syncIdx,
                                           taskId),
                            M3N12PackMeta1(rowBegin / moe_new_dispatch_combine_a8w8::kGmmBaseM,
                                           nBase / moe_new_dispatch_combine_a8w8::kGmmBaseN,
                                           shape.intermediateSize / moe_new_dispatch_combine_a8w8::kGmmBaseK,
                                           static_cast<uint32_t>(
                                               moe_new_dispatch_combine_a8w8::M3N12TimelineWaitSource::kGmPoll)));
                        m3n12Gmm2TileRecorded = true;
                    }
                }
                moe_new_dispatch_combine_a8w8::M3N4AicAllDoneCoarseSync();
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
                bool recordTile = timelineEnable != 0U && get_block_idx() == 0 && !m3n12Gmm2TileRecorded;
                uint64_t tileBegin = recordTile ? M3N12GetSysCnt() : 0U;
                M2FusedRunInt8GmmTileScalar(tileInput, expertWeight + nBase, tileOutput + nBase, mValid,
                                            shape.intermediateSize, nValid, rowBytes, shape.hiddenSize,
                                            shape.hiddenSize);
                if (recordTile) {
                    M3N12RecordTimeline(
                        timelineScratch, moe_new_dispatch_combine_a8w8::kM3N12TimelineSlotGmm2, tileBegin,
                        M3N12GetSysCnt(),
                        M3N12PackMeta0(moe_new_dispatch_combine_a8w8::M3N12TimelineKind::kGmm2Tile,
                                       moe_new_dispatch_combine_a8w8::M3N12TimelineCoreType::kAic,
                                       static_cast<uint32_t>(get_block_idx()),
                                       moe_new_dispatch_combine_a8w8::M3N12TimelineStatus::kProcessed, 0U, taskId),
                        M3N12PackMeta1(
                            rowBegin / moe_new_dispatch_combine_a8w8::kGmmBaseM,
                            nBase / moe_new_dispatch_combine_a8w8::kGmmBaseN,
                            shape.intermediateSize / moe_new_dispatch_combine_a8w8::kGmmBaseK,
                            static_cast<uint32_t>(moe_new_dispatch_combine_a8w8::M3N12TimelineWaitSource::kSyncAll)));
                    m3n12Gmm2TileRecorded = true;
                }
            }
        }
        M2FusedRecordGmmTaskEvidence(debugCounters, kM2FusedFullM3OGmm2CounterBase, taskCount, gmm2TileTaskPlan);
        moe_new_dispatch_combine_a8w8::M3N4AicAllDoneCoarseSync();
        if (get_block_idx() == 0) {
            M2FusedFinalizeGmmTaskEvidence(debugCounters, kM2FusedFullM3OGmm2CounterBase, taskCount);
        }
    }
    pipe_barrier(PIPE_ALL);
    dsb(DSB_DDR);
    moe_new_dispatch_combine_a8w8::M3N4AicAllDoneCoarseSync();
    if (get_block_idx() == 0 && !m3n7ActivationGmm2Overlap && shape.rankNum != 0 && shape.expertPerRank != 0 &&
        shape.hiddenSize != 0U && shape.intermediateSize != 0U &&
        shape.gmmBlockM == moe_new_dispatch_combine_a8w8::kGmmBaseM &&
        shape.gmmBlockN == moe_new_dispatch_combine_a8w8::kGmmBaseN &&
        shape.gmmBlockK == moe_new_dispatch_combine_a8w8::kGmmBaseK &&
        shape.hiddenSize % moe_new_dispatch_combine_a8w8::kGmmBaseK == 0U &&
        shape.intermediateSize % moe_new_dispatch_combine_a8w8::kGmmBaseK == 0U) {
        __gm__ int32_t *gmm2GroupReady = reinterpret_cast<__gm__ int32_t *>(workspace + layout.gmm2GroupReady.offset);
        for (uint32_t localExpert = 0; localExpert < shape.expertPerRank; ++localExpert) {
            M2FusedGmmStoreScalarI32(gmm2GroupReady + localExpert * 16U, 1);
        }
    }
    pipe_barrier(PIPE_ALL);
    dsb(DSB_DDR);
    if (!m3n8Gmm2CombineOverlap) {
        moe_new_dispatch_combine_a8w8::M3N4SignalC2V<moe_new_dispatch_combine_a8w8::kM3N4Gmm2ToCombineFlag>();
    }
    if (debugStopStage == 5U) {
        return;
    }
    if (!m3n8Gmm2CombineOverlap && (debugStopStage == 51U || debugStopStage == 52U || debugStopStage == 53U ||
                                    debugStopStage == 54U || debugStopStage == 56U || debugStopStage == 57U ||
                                    debugStopStage == 58U || (debugStopStage >= 59U && debugStopStage <= 86U))) {
        return;
    }
    if (m3n8Gmm2CombineOverlap && (debugStopStage == 51U || debugStopStage == 52U || debugStopStage == 53U ||
                                   debugStopStage == 54U || debugStopStage == 56U)) {
        return;
    }
    moe_new_dispatch_combine_a8w8::M3N4WaitV2C<moe_new_dispatch_combine_a8w8::kM3N4CombineToRestoreFlag>();
    if (debugStopStage == 6U) {
        return;
    }
}
#endif

#if defined(M2_MIXED_SPIKE_BUILD_AIV)
PTO_SYNCALL_MIX_AIC_KERNEL_META(M2MixedSpike_2801_mix_aiv, 1, 2);
PTO_SYNCALL_MIX_AIC_KERNEL_META(M2FusedSkeleton_2802_mix_aiv, 1, 2);
PTO_SYNCALL_MIX_AIC_KERNEL_META(M2FusedFull_2803_mix_aiv, 1, 2);

AICORE inline void M2FusedRunGmm1EpilogueScalarRange(moe_new_dispatch_combine_a8w8::ShapeConfig shape,
                                                     M2WorkspaceViewDevice workspaceView, uint32_t expertBegin,
                                                     uint32_t expertEnd)
{
    uint32_t w1Cols = shape.intermediateSize * 2U;
    if (expertEnd > shape.expertPerRank) {
        expertEnd = shape.expertPerRank;
    }
    for (uint32_t localExpert = expertBegin; localExpert < expertEnd; ++localExpert) {
        int32_t rowBegin = LoadScalarI32(workspaceView.dispatchOffset + localExpert);
        int32_t rowCount = LoadScalarI32(workspaceView.expertTokenNums + localExpert);
        if (rowCount <= 0) {
            continue;
        }
        for (uint32_t row = 0; row < static_cast<uint32_t>(rowCount); ++row) {
            uint32_t globalRow = static_cast<uint32_t>(rowBegin) + row;
            for (uint32_t col = 0; col < w1Cols; ++col) {
                int32_t acc =
                    LoadScalarI32(workspaceView.gmm1AccInt32 + static_cast<uint64_t>(globalRow) * w1Cols + col);
                float scale = M2DecodeUint64Scale(workspaceView.scale1Uint64[col]);
                workspaceView.gmm1Out[static_cast<uint64_t>(globalRow) * w1Cols + col] =
                    static_cast<float>(acc) * scale;
            }
        }
    }
}

AICORE inline void M2FusedRunGmm1EpilogueScalar(moe_new_dispatch_combine_a8w8::ShapeConfig shape,
                                                M2WorkspaceViewDevice workspaceView)
{
    uint32_t w1Cols = shape.intermediateSize * 2U;
    M2FusedRunGmm1EpilogueScalarRange(shape, workspaceView, 0U, shape.expertPerRank);
    uint32_t syncGroupCount = static_cast<uint32_t>(LoadScalarI32(workspaceView.swigluSyncGroups));
    if (syncGroupCount == 0U || syncGroupCount > shape.expertPerRank) {
        syncGroupCount = shape.expertPerRank;
    }
    for (uint32_t syncIdx = 0; syncIdx < syncGroupCount; ++syncIdx) {
        StoreScalarI32(workspaceView.gmm1SyncGroupReady + syncIdx * 16U, 1);
    }
    InvalidateGmCacheLines(workspaceView.gmm1Out, static_cast<uint32_t>(M2LocalRows(shape) * w1Cols * sizeof(float)));
}

AICORE inline void M3N6RunGmm1EpilogueSyncGroupScalar(moe_new_dispatch_combine_a8w8::ShapeConfig shape,
                                                      M2WorkspaceViewDevice workspaceView, uint32_t syncIdx)
{
    uint32_t syncGroupCount = static_cast<uint32_t>(LoadScalarI32(workspaceView.swigluSyncGroups));
    if (syncGroupCount == 0U || syncGroupCount > shape.expertPerRank || syncIdx >= syncGroupCount) {
        return;
    }
    __gm__ int32_t *group = workspaceView.swigluGroupDesc + syncIdx * kM2SwigluGroupFields;
    int32_t expertBeginI32 = LoadScalarI32(group + 1U);
    int32_t expertEndI32 = LoadScalarI32(group + 2U);
    if (expertBeginI32 < 0 || expertEndI32 <= expertBeginI32) {
        return;
    }
    uint32_t w1Cols = shape.intermediateSize * 2U;
    M2FusedRunGmm1EpilogueScalarRange(shape, workspaceView, static_cast<uint32_t>(expertBeginI32),
                                      static_cast<uint32_t>(expertEndI32));
    InvalidateGmCacheLines(workspaceView.gmm1Out, static_cast<uint32_t>(M2LocalRows(shape) * w1Cols * sizeof(float)));
}

AICORE inline float M2FusedAbsFloat(float value)
{
    return value < 0.0f ? -value : value;
}

AICORE inline float M2FusedExpApprox(float value)
{
    if (!(value == value)) {
        return 1.0f;
    }
    if (value <= -16.0f) {
        return 0.0f;
    }
    if (value >= 16.0f) {
        value = 16.0f;
    }
    constexpr float kInvLn2 = 1.4426950408889634f;
    constexpr float kLn2 = 0.6931471805599453f;
    int32_t exponent = static_cast<int32_t>(value * kInvLn2);
    if (value < 0.0f && static_cast<float>(exponent) * kLn2 > value) {
        --exponent;
    }
    if (exponent > 24) {
        exponent = 24;
    }
    if (exponent < -24) {
        exponent = -24;
    }
    float reduced = value - static_cast<float>(exponent) * kLn2;
    float reduced2 = reduced * reduced;
    float reduced4 = reduced2 * reduced2;
    float poly = 1.0f + reduced + reduced2 * 0.5f + reduced2 * reduced * 0.1666666667f + reduced4 * 0.0416666667f +
                 reduced4 * reduced * 0.0083333333f + reduced4 * reduced2 * 0.0013888889f +
                 reduced4 * reduced2 * reduced * 0.0001984127f + reduced4 * reduced4 * 0.0000248016f;
    if (exponent >= 0) {
        for (int32_t idx = 0; idx < exponent; ++idx) {
            poly *= 2.0f;
        }
    } else {
        for (int32_t idx = 0; idx < -exponent; ++idx) {
            poly *= 0.5f;
        }
    }
    return poly;
}

AICORE inline float M2FusedSigmoidScalar(float value)
{
    if (!(value == value)) {
        return 0.5f;
    }
    if (value >= 16.0f) {
        return 1.0f;
    }
    if (value <= -16.0f) {
        return 0.0f;
    }
    if (value >= 0.0f) {
        float expNeg = M2FusedExpApprox(-value);
        return 1.0f / (1.0f + expNeg);
    }
    float expPos = M2FusedExpApprox(value);
    return expPos / (1.0f + expPos);
}

AICORE inline int8_t M2FusedQuantizeToInt8Scalar(float value, float scale)
{
    if (scale <= 0.0f || !(scale == scale) || !(value == value)) {
        return 0;
    }
    float scaled = value / scale;
    int32_t quant = scaled >= 0.0f ? static_cast<int32_t>(scaled + 0.5f) : static_cast<int32_t>(scaled - 0.5f);
    if (quant > 127) {
        quant = 127;
    }
    if (quant < -127) {
        quant = -127;
    }
    return static_cast<int8_t>(quant);
}

AICORE inline void M2FusedComputeSwigluRowScalar(moe_new_dispatch_combine_a8w8::ShapeConfig shape,
                                                 M2WorkspaceViewDevice workspaceView, uint32_t globalRow,
                                                 float routingScale)
{
    uint32_t w1Cols = shape.intermediateSize * 2U;
    for (uint32_t col = 0; col < shape.intermediateSize; ++col) {
        float gate = workspaceView.gmm1Out[static_cast<uint64_t>(globalRow) * w1Cols + col] * routingScale;
        float up = workspaceView.gmm1Out[static_cast<uint64_t>(globalRow) * w1Cols + shape.intermediateSize + col] *
                   routingScale;
        float value = M2FusedSigmoidScalar(gate) * up;
        workspaceView.swigluOut[static_cast<uint64_t>(globalRow) * shape.intermediateSize + col] = value;
    }
}

AICORE inline void M2FusedRunActivationQuantRowScalar(moe_new_dispatch_combine_a8w8::ShapeConfig shape,
                                                      M2WorkspaceViewDevice workspaceView, uint32_t globalRow,
                                                      float routingScale, uint32_t gmm2RowStride)
{
    M2FusedComputeSwigluRowScalar(shape, workspaceView, globalRow, routingScale);
    InvalidateGmCacheLines(workspaceView.swigluOut + static_cast<uint64_t>(globalRow) * shape.intermediateSize,
                           static_cast<uint32_t>(shape.intermediateSize * sizeof(float)));
    M2RequantizeSwigluRowPto(shape, workspaceView, globalRow, gmm2RowStride);
}

AICORE inline void M2FusedRunActivationQuantRowScalarProbe(moe_new_dispatch_combine_a8w8::ShapeConfig shape,
                                                           M2WorkspaceViewDevice workspaceView, uint32_t globalRow,
                                                           float routingScale, uint32_t gmm2RowStride,
                                                           uint32_t probeMode)
{
    uint32_t w1Cols = shape.intermediateSize * 2U;
    __gm__ int8_t *dst = workspaceView.gmm2InputInt8 + static_cast<uint64_t>(globalRow) * gmm2RowStride;
    if (probeMode == 1U) {
        workspaceView.gmm2PerTokenScale[globalRow] = 1.0f;
        for (uint32_t col = 0; col < shape.intermediateSize; ++col) {
            workspaceView.swigluOut[static_cast<uint64_t>(globalRow) * shape.intermediateSize + col] = 0.0f;
            dst[col] = 0;
        }
        for (uint32_t col = shape.intermediateSize; col < gmm2RowStride; ++col) {
            dst[col] = 0;
        }
        return;
    }
    if (probeMode >= 4U && probeMode <= 8U) {
        for (uint32_t col = 0; col < shape.intermediateSize; ++col) {
            float gate = workspaceView.gmm1Out[static_cast<uint64_t>(globalRow) * w1Cols + col];
            float up = workspaceView.gmm1Out[static_cast<uint64_t>(globalRow) * w1Cols + shape.intermediateSize + col];
            float value = routingScale;
            if (probeMode == 5U) {
                value = gate;
            } else if (probeMode == 6U) {
                value = gate * routingScale;
            } else if (probeMode == 7U) {
                value = up;
            } else if (probeMode == 8U) {
                value = gate + up;
            }
            workspaceView.swigluOut[static_cast<uint64_t>(globalRow) * shape.intermediateSize + col] = value;
        }
        return;
    }
    if (probeMode == 9U) {
        M2FusedComputeSwigluRowScalar(shape, workspaceView, globalRow, routingScale);
        return;
    }
    if (probeMode == 10U || probeMode == 11U) {
        for (uint32_t col = 0; col < shape.intermediateSize; ++col) {
            float value = probeMode == 10U ? 0.0f : routingScale;
            workspaceView.swigluOut[static_cast<uint64_t>(globalRow) * shape.intermediateSize + col] = value;
        }
        InvalidateGmCacheLines(workspaceView.swigluOut + static_cast<uint64_t>(globalRow) * shape.intermediateSize,
                               static_cast<uint32_t>(shape.intermediateSize * sizeof(float)));
        M2RequantizeSwigluRowPto(shape, workspaceView, globalRow, gmm2RowStride);
        return;
    }
    float maxAbs = 0.0f;
    for (uint32_t col = 0; col < shape.intermediateSize; ++col) {
        float gate = workspaceView.gmm1Out[static_cast<uint64_t>(globalRow) * w1Cols + col] * routingScale;
        float up = workspaceView.gmm1Out[static_cast<uint64_t>(globalRow) * w1Cols + shape.intermediateSize + col] *
                   routingScale;
        float value = probeMode == 2U ? gate + up : M2FusedSigmoidScalar(gate) * up;
        workspaceView.swigluOut[static_cast<uint64_t>(globalRow) * shape.intermediateSize + col] = value;
        float absValue = M2FusedAbsFloat(value);
        if (absValue > maxAbs) {
            maxAbs = absValue;
        }
    }
    if (probeMode == 2U || probeMode == 3U) {
        return;
    }
    float scale = maxAbs == 0.0f ? 1.0f : maxAbs / 127.0f;
    workspaceView.gmm2PerTokenScale[globalRow] = scale;
    for (uint32_t col = 0; col < shape.intermediateSize; ++col) {
        float value = workspaceView.swigluOut[static_cast<uint64_t>(globalRow) * shape.intermediateSize + col];
        dst[col] = M2FusedQuantizeToInt8Scalar(value, scale);
    }
    for (uint32_t col = shape.intermediateSize; col < gmm2RowStride; ++col) {
        dst[col] = 0;
    }
}

AICORE inline uint32_t M3NRunActivationQuantShardScalar(moe_new_dispatch_combine_a8w8::ShapeConfig shape,
                                                        M2WorkspaceViewDevice workspaceView, uint32_t workerId,
                                                        uint32_t workerCount, uint32_t maxRowsPerWorker,
                                                        uint32_t probeMode)
{
    if (workerCount == 0U) {
        return 0U;
    }
    uint32_t gmm2RowStride = static_cast<uint32_t>(Align64Device(shape.intermediateSize));
    uint32_t syncGroupCount = static_cast<uint32_t>(LoadScalarI32(workspaceView.swigluSyncGroups));
    if (syncGroupCount == 0U || syncGroupCount > shape.expertPerRank) {
        syncGroupCount = shape.expertPerRank;
    }
    uint32_t rowsProcessed = 0;
    for (uint32_t syncIdx = 0; syncIdx < syncGroupCount; ++syncIdx) {
        __gm__ int32_t *group = workspaceView.swigluGroupDesc + syncIdx * kM2SwigluGroupFields;
        int32_t rowBeginI32 = LoadScalarI32(group + 3U);
        int32_t rowEndI32 = LoadScalarI32(group + 4U);
        if (rowBeginI32 < 0 || rowEndI32 <= rowBeginI32) {
            continue;
        }
        for (uint32_t globalRow = static_cast<uint32_t>(rowBeginI32) + workerId;
             globalRow < static_cast<uint32_t>(rowEndI32); globalRow += workerCount) {
            if (rowsProcessed >= maxRowsPerWorker) {
                return rowsProcessed;
            }
            float routingScale = workspaceView.routingPerTokenScale[globalRow];
            if (probeMode == 0U) {
                M2FusedRunActivationQuantRowScalar(shape, workspaceView, globalRow, routingScale, gmm2RowStride);
            } else {
                M2FusedRunActivationQuantRowScalarProbe(shape, workspaceView, globalRow, routingScale, gmm2RowStride,
                                                        probeMode);
            }
            ++rowsProcessed;
        }
    }
    return rowsProcessed;
}

AICORE inline uint32_t M3NRunActivationQuantSyncGroupShardScalar(moe_new_dispatch_combine_a8w8::ShapeConfig shape,
                                                                 M2WorkspaceViewDevice workspaceView, uint32_t syncIdx,
                                                                 uint32_t workerId, uint32_t workerCount)
{
    if (workerCount == 0U) {
        return 0U;
    }
    uint32_t gmm2RowStride = static_cast<uint32_t>(Align64Device(shape.intermediateSize));
    uint32_t syncGroupCount = static_cast<uint32_t>(LoadScalarI32(workspaceView.swigluSyncGroups));
    if (syncGroupCount == 0U || syncGroupCount > shape.expertPerRank || syncIdx >= syncGroupCount) {
        return 0U;
    }
    __gm__ int32_t *group = workspaceView.swigluGroupDesc + syncIdx * kM2SwigluGroupFields;
    int32_t rowBeginI32 = LoadScalarI32(group + 3U);
    int32_t rowEndI32 = LoadScalarI32(group + 4U);
    if (rowBeginI32 < 0 || rowEndI32 <= rowBeginI32) {
        return 0U;
    }
    uint32_t rowsProcessed = 0;
    for (uint32_t globalRow = static_cast<uint32_t>(rowBeginI32) + workerId;
         globalRow < static_cast<uint32_t>(rowEndI32); globalRow += workerCount) {
        float routingScale = workspaceView.routingPerTokenScale[globalRow];
        M2FusedRunActivationQuantRowScalar(shape, workspaceView, globalRow, routingScale, gmm2RowStride);
        ++rowsProcessed;
    }
    return rowsProcessed;
}

AICORE inline void M2RunActivationQuantScalar(moe_new_dispatch_combine_a8w8::ShapeConfig shape,
                                              M2WorkspaceViewDevice workspaceView)
{
    uint32_t gmm2RowStride = static_cast<uint32_t>(Align64Device(shape.intermediateSize));
    uint32_t localRows = static_cast<uint32_t>(M2LocalRows(shape));
    for (uint32_t row = 0; row < localRows; ++row) {
        workspaceView.gmm2PerTokenScale[row] = 1.0f;
    }
    for (uint32_t localExpert = 0; localExpert < shape.expertPerRank; ++localExpert) {
        int32_t rowBegin = LoadScalarI32(workspaceView.dispatchOffset + localExpert);
        int32_t rowCount = LoadScalarI32(workspaceView.expertTokenNums + localExpert);
        if (rowCount <= 0) {
            continue;
        }
        for (uint32_t row = 0; row < static_cast<uint32_t>(rowCount); ++row) {
            uint32_t globalRow = static_cast<uint32_t>(rowBegin) + row;
            float routingScale = workspaceView.routingPerTokenScale[globalRow];
            M2FusedRunActivationQuantRowScalar(shape, workspaceView, globalRow, routingScale, gmm2RowStride);
        }
    }
    M2BuildSwigluSyncMetadata(shape, workspaceView);
    uint32_t syncGroupCount = static_cast<uint32_t>(LoadScalarI32(workspaceView.swigluSyncGroups));
    if (syncGroupCount == 0U || syncGroupCount > shape.expertPerRank) {
        syncGroupCount = shape.expertPerRank;
    }
    for (uint32_t syncIdx = 0; syncIdx < syncGroupCount; ++syncIdx) {
        StoreScalarI32(workspaceView.activationSyncGroupReady + syncIdx * 16U, 1);
    }
    InvalidateGmCacheLines(workspaceView.swigluOut,
                           static_cast<uint32_t>(localRows * shape.intermediateSize * sizeof(float)));
    InvalidateGmCacheLines(workspaceView.gmm2InputInt8, static_cast<uint32_t>(localRows * gmm2RowStride));
    InvalidateGmCacheLines(workspaceView.gmm2PerTokenScale, static_cast<uint32_t>(localRows * sizeof(float)));
}

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
    __gm__ uint64_t *fftsAddr, __gm__ moe_new_dispatch_combine_a8w8::M2FusedFullLaunchArgs *launchArgs)
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
            M2FusedLoadConfigI32(earlyStageStatus, moe_new_dispatch_combine_a8w8::kM2FusedFullShapeRankNumSlot);
        int32_t hidden =
            M2FusedLoadConfigI32(earlyStageStatus, moe_new_dispatch_combine_a8w8::kM2FusedFullShapeHiddenSizeSlot);
        int32_t intermediate = M2FusedLoadConfigI32(
            earlyStageStatus, moe_new_dispatch_combine_a8w8::kM2FusedFullShapeIntermediateSizeSlot);
        int32_t blockK =
            M2FusedLoadConfigI32(earlyStageStatus, moe_new_dispatch_combine_a8w8::kM2FusedFullShapeGmmBlockKSlot);
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
        for (uint32_t slot = moe_new_dispatch_combine_a8w8::kM2FusedFullShapeRankNumSlot;
             slot <= moe_new_dispatch_combine_a8w8::kM2FusedFullShapeDtypeOutSlot; ++slot) {
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
            moe_new_dispatch_combine_a8w8::ShapeConfig shape = M2FusedLoadShapeConfig(earlyStageStatus);
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
        moe_new_dispatch_combine_a8w8::ShapeConfig shape = M2FusedLoadShapeConfig(earlyStageStatus);
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
        moe_new_dispatch_combine_a8w8::RankConfig rank = M2FusedLoadRankConfig(earlyStageStatus);
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
        moe_new_dispatch_combine_a8w8::ShapeConfig shape = M2FusedLoadShapeConfig(stageStatus);
        moe_new_dispatch_combine_a8w8::RankConfig rank = M2FusedLoadRankConfig(stageStatus);
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
    set_ffts_base_addr(reinterpret_cast<uint64_t>(fftsAddr));
    __gm__ int32_t *stageStatusBase = reinterpret_cast<__gm__ int32_t *>(launchArgs->stageStatusAddr);
    moe_new_dispatch_combine_a8w8::ShapeConfig shape = M2FusedLoadShapeConfig(stageStatusBase);
    moe_new_dispatch_combine_a8w8::RankConfig rank = M2FusedLoadRankConfig(stageStatusBase);
    GM_ADDR inputA = reinterpret_cast<GM_ADDR>(
        M2FusedLoadConfigU64(stageStatusBase, moe_new_dispatch_combine_a8w8::kM2FusedFullPtrInputASlot));
    GM_ADDR expertIdx = reinterpret_cast<GM_ADDR>(
        M2FusedLoadConfigU64(stageStatusBase, moe_new_dispatch_combine_a8w8::kM2FusedFullPtrExpertIdxSlot));
    GM_ADDR xActiveMask = reinterpret_cast<GM_ADDR>(
        M2FusedLoadConfigU64(stageStatusBase, moe_new_dispatch_combine_a8w8::kM2FusedFullPtrXActiveMaskSlot));
    GM_ADDR probs = reinterpret_cast<GM_ADDR>(
        M2FusedLoadConfigU64(stageStatusBase, moe_new_dispatch_combine_a8w8::kM2FusedFullPtrProbsSlot));
    GM_ADDR outputC = reinterpret_cast<GM_ADDR>(
        M2FusedLoadConfigU64(stageStatusBase, moe_new_dispatch_combine_a8w8::kM2FusedFullPtrOutputCSlot));
    GM_ADDR peerWindow = reinterpret_cast<GM_ADDR>(
        M2FusedLoadConfigU64(stageStatusBase, moe_new_dispatch_combine_a8w8::kM2FusedFullPtrPeerWindowSlot));
    GM_ADDR hcclCtx = reinterpret_cast<GM_ADDR>(
        M2FusedLoadConfigU64(stageStatusBase, moe_new_dispatch_combine_a8w8::kM2FusedFullPtrHcclCtxSlot));
    GM_ADDR workspace = reinterpret_cast<GM_ADDR>(
        M2FusedLoadConfigU64(stageStatusBase, moe_new_dispatch_combine_a8w8::kM2FusedFullPtrWorkspaceSlot));
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
        M2FusedLoadConfigI32(stageStatusBase, moe_new_dispatch_combine_a8w8::kM2FusedFullTimelineEnableSlot));
    uint32_t overlapMode = static_cast<uint32_t>(
        M2FusedLoadConfigI32(stageStatusBase, moe_new_dispatch_combine_a8w8::kM2FusedFullOverlapModeSlot));
    uint32_t logicalAiv = M2FusedLogicalAivId();
    uint32_t rawAivSlot = M2FusedRawAivSlot();
    bool m3n5DispatchGmm1Overlap = M3N5DispatchGmm1OverlapEnabled(debugStopStage, overlapMode, shape);
    bool m3n6Gmm1ActivationOverlap = M3N6Gmm1ActivationOverlapEnabled(debugStopStage, overlapMode, shape);
    bool m3n7ActivationGmm2Overlap = M3N7ActivationGmm2OverlapEnabled(debugStopStage, overlapMode, shape);
    bool m3n8Gmm2CombineOverlap = M3N8Gmm2CombineOverlapEnabled(debugStopStage, overlapMode, shape);
    uint32_t m3n6SyncGroupCount = moe_new_dispatch_combine_a8w8::M3N6SwigluSyncGroupCount(shape);
    M3NRecordAivLaneDebug(localPeer, rawAivSlot);
    if (IsM2FusedMainAiv()) {
        uint32_t m3nHandshakeSites = moe_new_dispatch_combine_a8w8::kM3N4StreamHandshakeSites;
        uint32_t m3nCvWaitCount = moe_new_dispatch_combine_a8w8::kM3N4FullOpenCvWaitCount;
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
                       static_cast<int32_t>(moe_new_dispatch_combine_a8w8::kM3N4CoarseMixSyncallCount +
                                            moe_new_dispatch_combine_a8w8::kM3N4AicOnlySyncCount +
                                            moe_new_dispatch_combine_a8w8::kM3N4AivOnlySyncCount));
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
    bool initQuantFullLoad = M3NInitQuantFullLoadFits(shape);
    bool m3nDispatchEnabled = debugStopStage < 100U && M3NFusedDispatchScratchFits(shape);
    if (m3nDispatchEnabled) {
        pipe_barrier(PIPE_ALL);
        uint64_t initQuantE2eBegin = IsM2FusedMainAiv() ? M3N12GetSysCnt() : 0U;
        M3NDispatchHardPhaseSync();
        if (IsM2FusedMainAiv()) {
            dispatchWorkerCount = initQuantFullLoad ? 1U : M3NAssignDispatchWorkers(shape, localPeer, logicalAivCount);
            if (initQuantFullLoad) {
                StoreScalarI32(localPeer.debugCounters + kM3NDispatchCounterBase + 3U, 1);
                StoreScalarI32(localPeer.debugCounters + kM3NDispatchCounterBase + 11U, 1);
            }
        }
        M3NDispatchHardPhaseSync();
        M3NDispatchWorkerAssignment dispatchAssignment =
            initQuantFullLoad ? M3NDispatchWorkerAssignment{0U, 1U, IsM2FusedMainAiv()} :
                                M3NLoadDispatchWorkerAssignment(localPeer, rawAivSlot);
        uint32_t dispatchWorkerId = dispatchAssignment.workerId;
        dispatchWorkerCount = dispatchAssignment.workerCount;
        bool activeDispatchWorker = dispatchAssignment.active;
        uint64_t countTimelineBegin = 0U;
        if (IsM2FusedMainAiv()) {
            M2FusedRecordStage(stageStatus + kM2FusedFullStageBaseSlot, 5U, 100);
            M2ClearDispatchState(shape, workspaceView, localPeer);
            M2FusedRecordStage(stageStatus + kM2FusedFullStageBaseSlot, 5U, 101);
        }
        M3NDispatchHardPhaseSync();
        if (activeDispatchWorker && !initQuantFullLoad) {
            M3NClearDispatchWorkerScratch(shape, workspaceView, localPeer, dispatchWorkerId, dispatchWorkerCount);
        }
        M3NDispatchHardPhaseSync();
        if (initQuantFullLoad) {
            if (IsM2FusedMainAiv()) {
                uint64_t countBegin = timelineEnable != 0U ? M3N12GetSysCnt() : 0U;
                M2RunInitQuantFullLoadPtoVec(shape, workspaceView, localPeer, inputA, expertIdx, xActiveMask,
                                             rank.rankId, rowBytes);
                if (timelineEnable != 0U) {
                    M3N12RecordAivRange(
                        workspaceView.timelineScratch, moe_new_dispatch_combine_a8w8::kM3N12TimelineSlotRouteCount,
                        moe_new_dispatch_combine_a8w8::M3N12TimelineKind::kRouteCount, logicalAiv, 0U, 0U, shape.m,
                        shape.m, 0U, countBegin, M3N12GetSysCnt());
                }
                M2FusedRecordStage(stageStatus + kM2FusedFullStageBaseSlot, 5U, 102);
                if (timelineEnable != 0U) {
                    M3N12RecordAivRange(
                        workspaceView.timelineScratch, moe_new_dispatch_combine_a8w8::kM3N12TimelineSlotRoute,
                        moe_new_dispatch_combine_a8w8::M3N12TimelineKind::kRoute, logicalAiv, 0U, 0U, shape.m,
                        shape.m, 0U, countBegin, M3N12GetSysCnt());
                }
                StoreScalarI32(localPeer.debugCounters + kM3NDispatchCounterBase + 0U, 1);
                StoreScalarI32(localPeer.debugCounters + kM3NDispatchCounterBase + 1U, 1);
                StoreScalarI32(localPeer.debugCounters + kM3NDispatchCounterBase + 3U, 1);
                StoreScalarI32(localPeer.debugCounters + kM3NDispatchCounterBase + 4U,
                               static_cast<int32_t>(shape.m * shape.topK));
                StoreScalarI32(localPeer.debugCounters + kM3NDispatchCounterBase + 7U, 1);
                StoreScalarI32(localPeer.debugCounters + kM3NDispatchCounterBase + 8U, 1);
                StoreScalarI32(localPeer.debugCounters + kM3NDispatchCounterBase + 10U, 1);
                StoreScalarI32(localPeer.debugCounters + kM3NDispatchCounterBase + 11U, 1);
            }
        } else if (activeDispatchWorker) {
            uint32_t tokenBegin = TokenShardBegin(shape.m, dispatchWorkerId, dispatchWorkerCount);
            uint32_t tokenEnd = TokenShardEnd(shape.m, dispatchWorkerId, dispatchWorkerCount);
            bool recordCount = timelineEnable != 0U && dispatchWorkerId == 0U;
            uint64_t countBegin = recordCount ? M3N12GetSysCnt() : 0U;
            M3NCountLocalRoutesShard(shape, workspaceView, expertIdx, xActiveMask, dispatchWorkerId,
                                     dispatchWorkerCount);
            if (recordCount) {
                M3N12RecordAivRange(
                    workspaceView.timelineScratch, moe_new_dispatch_combine_a8w8::kM3N12TimelineSlotRouteCount,
                    moe_new_dispatch_combine_a8w8::M3N12TimelineKind::kRouteCount, logicalAiv, dispatchWorkerId,
                    tokenBegin, tokenEnd, tokenEnd - tokenBegin, 0U, countBegin, M3N12GetSysCnt());
            }
        }
        if (!initQuantFullLoad) {
            M3NDispatchInitQuantPhaseSync(debugStopStage);
            if (IsM2FusedMainAiv()) {
                M3NMergeLocalRouteCounts(shape, workspaceView, localPeer, rank.rankId, dispatchWorkerCount);
                M2FusedRecordStage(stageStatus + kM2FusedFullStageBaseSlot, 5U, 102);
            }
            M3NDispatchInitQuantPhaseSync(debugStopStage);
        }
        if (activeDispatchWorker && !initQuantFullLoad) {
            uint32_t tokenBegin = TokenShardBegin(shape.m, dispatchWorkerId, dispatchWorkerCount);
            uint32_t tokenEnd = TokenShardEnd(shape.m, dispatchWorkerId, dispatchWorkerCount);
            bool recordRoute = timelineEnable != 0U && dispatchWorkerId == 0U;
            uint64_t routeBegin = recordRoute ? M3N12GetSysCnt() : 0U;
            M3NRoutePackQuantLocalShardPtoVec(shape, workspaceView, localPeer, inputA, expertIdx, xActiveMask, rowBytes,
                                              dispatchWorkerId, dispatchWorkerCount);
            if (recordRoute) {
                M3N12RecordAivRange(
                    workspaceView.timelineScratch, moe_new_dispatch_combine_a8w8::kM3N12TimelineSlotRoute,
                    moe_new_dispatch_combine_a8w8::M3N12TimelineKind::kRoute, logicalAiv, dispatchWorkerId, tokenBegin,
                    tokenEnd, tokenEnd - tokenBegin, 0U, routeBegin, M3N12GetSysCnt());
            }
        }
        if (!initQuantFullLoad) {
            M3NDispatchInitQuantPhaseSync(debugStopStage);
        }
        if (activeDispatchWorker && debugStopStage != 18U) {
            if (timelineEnable != 0U && dispatchWorkerId == 0U) {
                countTimelineBegin = M3N12GetSysCnt();
            }
            if (debugStopStage == 19U || debugStopStage == 20U || debugStopStage == 21U) {
                uint32_t debugPublishMode = debugStopStage == 19U ? 1U : (debugStopStage == 20U ? 2U : 3U);
                M3NPublishCountRowsShard(shape, workspaceView, ctx, peerWindow, rank.rankId, peerWindowLayout,
                                         dispatchWorkerId, dispatchWorkerCount, debugPublishMode);
            } else {
                M3NPublishCountRowsShard(shape, workspaceView, ctx, peerWindow, rank.rankId, peerWindowLayout,
                                         dispatchWorkerId, dispatchWorkerCount, 3U);
            }
        }
        if (IsM2FusedMainAiv()) {
            StoreScalarI32(localPeer.debugCounters + kM3NDispatchCounterBase + 1U,
                           static_cast<int32_t>(dispatchWorkerCount));
            M2FusedRecordStage(
                stageStatus + kM2FusedFullStageBaseSlot, 5U,
                debugStopStage >= 18U && debugStopStage <= 21U ? static_cast<int32_t>(debugStopStage) : 104);
        }
        M3NDispatchHardPhaseSync();
        if (debugStopStage == 13U || (debugStopStage >= 18U && debugStopStage <= 21U)) {
            if (IsM2FusedMainAiv()) {
                M2FusedRecordStage(stageStatus + kM2FusedFullStageBaseSlot, 5U, static_cast<int32_t>(debugStopStage));
            }
            pto::SYNCALL<pto::SyncCoreType::Mix>();
            return;
        }
        if (activeDispatchWorker) {
            M3NWaitCountRowsShardScalar(shape, localPeer, rank.rankId, dispatchWorkerId, dispatchWorkerCount);
            if (timelineEnable != 0U && dispatchWorkerId == 0U) {
                M3N12RecordAivRange(
                    workspaceView.timelineScratch, moe_new_dispatch_combine_a8w8::kM3N12TimelineSlotCountSync,
                    moe_new_dispatch_combine_a8w8::M3N12TimelineKind::kCountSync, logicalAiv, dispatchWorkerId, 0U,
                    shape.rankNum, shape.rankNum, 0U, countTimelineBegin, M3N12GetSysCnt());
            }
        }
        M3NDispatchHardPhaseSync();
        if (IsM2FusedMainAiv()) {
            M3NApplyDispatchCapacityClip(shape, workspaceView, localPeer, rank.rankId);
        }
        if (IsM2FusedMainAiv()) {
            uint64_t prefixBegin = timelineEnable != 0U ? M3N12GetSysCnt() : 0U;
            M2BuildPrefixMetadata(shape, workspaceView, localPeer, rank.rankId);
            if (timelineEnable != 0U) {
                M3N12RecordAivRange(workspaceView.timelineScratch,
                                    moe_new_dispatch_combine_a8w8::kM3N12TimelineSlotPrefix,
                                    moe_new_dispatch_combine_a8w8::M3N12TimelineKind::kPrefix, logicalAiv, 0U, 0U,
                                    shape.expertPerRank, shape.expertPerRank, 0U, prefixBegin, M3N12GetSysCnt());
            }
            M2FusedRecordStage(stageStatus + kM2FusedFullStageBaseSlot, 5U, 106);
        }
        M3NDispatchHardPhaseSync();
        if (m3n5DispatchGmm1Overlap) {
            bool requireAicStartAck = debugStopStage != 1U;
            bool recordGather = timelineEnable != 0U && activeDispatchWorker && dispatchWorkerId == 0U;
            uint64_t gatherBegin = recordGather ? M3N12GetSysCnt() : 0U;
            M3N5GatherDispatchToGmm1InputShardByExpert(
                shape, workspaceView, localPeer, ctx, peerWindow, rank.rankId, rowBytes, peerWindowLayout,
                dispatchWorkerId, dispatchWorkerCount, activeDispatchWorker, stageStatus, requireAicStartAck);
            if (recordGather) {
                uint32_t expertEnd = dispatchWorkerCount == 0U ?
                                         0U :
                                         (shape.expertPerRank + dispatchWorkerCount - 1U) / dispatchWorkerCount;
                M3N12RecordAivRange(workspaceView.timelineScratch,
                                    moe_new_dispatch_combine_a8w8::kM3N12TimelineSlotDispatchGather,
                                    moe_new_dispatch_combine_a8w8::M3N12TimelineKind::kDispatchGather, logicalAiv,
                                    dispatchWorkerId, 0U, expertEnd, expertEnd, 0U, gatherBegin, M3N12GetSysCnt());
            }
        } else if (activeDispatchWorker) {
            bool recordGather = timelineEnable != 0U && dispatchWorkerId == 0U;
            uint64_t gatherBegin = recordGather ? M3N12GetSysCnt() : 0U;
            M3NGatherDispatchToGmm1InputShard(shape, workspaceView, localPeer, ctx, peerWindow, rank.rankId, rowBytes,
                                              peerWindowLayout, dispatchWorkerId, dispatchWorkerCount);
            if (recordGather) {
                uint32_t expertEnd = dispatchWorkerCount == 0U ?
                                         0U :
                                         (shape.expertPerRank + dispatchWorkerCount - 1U) / dispatchWorkerCount;
                M3N12RecordAivRange(workspaceView.timelineScratch,
                                    moe_new_dispatch_combine_a8w8::kM3N12TimelineSlotDispatchGather,
                                    moe_new_dispatch_combine_a8w8::M3N12TimelineKind::kDispatchGather, logicalAiv,
                                    dispatchWorkerId, 0U, expertEnd, expertEnd, 0U, gatherBegin, M3N12GetSysCnt());
            }
        }
        if (m3n5DispatchGmm1Overlap) {
            M3NDispatchAivOnlyPhaseSync();
        } else {
            M3NDispatchHardPhaseSync();
        }
        if (IsM2FusedMainAiv()) {
            M3NPublishAllDispatchExpertsReadyAfterGather(shape, workspaceView);
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
        if (debugStopStage == 17U) {
            if (IsM2FusedMainAiv()) {
                M2FusedRecordStage(stageStatus + kM2FusedFullStageBaseSlot, 5U, 17);
            }
            pto::SYNCALL<pto::SyncCoreType::Mix>();
            if (IsM2FusedMainAiv()) {
                M3N12RecordTimeline(
                    workspaceView.timelineScratch, moe_new_dispatch_combine_a8w8::kM3N12TimelineSlotInitQuantE2E,
                    initQuantE2eBegin, M3N12GetSysCnt(),
                    M3N12PackMeta0(moe_new_dispatch_combine_a8w8::M3N12TimelineKind::kInitQuantE2E,
                                   moe_new_dispatch_combine_a8w8::M3N12TimelineCoreType::kAiv, logicalAiv,
                                   moe_new_dispatch_combine_a8w8::M3N12TimelineStatus::kProcessed, 0U, logicalAiv),
                    M3N12PackMeta1(0U, shape.m, shape.m * shape.topK, 0U));
            }
            return;
        }
        if (m3n6Gmm1ActivationOverlap) {
            moe_new_dispatch_combine_a8w8::M3N4AivAllDoneCoarseSync();
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
            } else if (debugStopStage == 10230U) {
                M2FusedRecordStage(stageStatus + kM2FusedFullStageBaseSlot, 5U, 10230);
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
                M2RoutePackQuantLocalPtoVec(shape, workspaceView, localPeer, inputA, expertIdx, xActiveMask, rowBytes);
                M2FusedRecordStage(stageStatus + kM2FusedFullStageBaseSlot, 5U, 103);
            }
        }
        if (debugStopStage < 1021U && debugStopStage != 100U && debugStopStage != 101U && debugStopStage != 102U &&
            debugStopStage != 103U) {
            M3NPublishCountRowsShard(shape, workspaceView, ctx, peerWindow, rank.rankId, peerWindowLayout, 0U, 1U, 3U);
            M2FusedRecordStage(stageStatus + kM2FusedFullStageBaseSlot, 5U, 104);
        }
        if (debugStopStage < 1021U && debugStopStage != 100U && debugStopStage != 101U && debugStopStage != 102U &&
            debugStopStage != 103U && debugStopStage != 104U) {
            M3NWaitCountRowsShardScalar(shape, localPeer, rank.rankId, 0U, 1U);
            M2FusedRecordStage(stageStatus + kM2FusedFullStageBaseSlot, 5U, 105);
        }
        if (debugStopStage < 1021U && debugStopStage != 100U && debugStopStage != 101U && debugStopStage != 102U &&
            debugStopStage != 103U && debugStopStage != 104U && debugStopStage != 105U) {
            M2BuildPrefixMetadata(shape, workspaceView, localPeer, rank.rankId);
            M2FusedRecordStage(stageStatus + kM2FusedFullStageBaseSlot, 5U, 106);
        }
        if (debugStopStage < 1021U && debugStopStage != 100U && debugStopStage != 101U && debugStopStage != 102U &&
            debugStopStage != 103U && debugStopStage != 104U && debugStopStage != 105U && debugStopStage != 106U) {
            M3NGatherDispatchToGmm1InputShard(shape, workspaceView, localPeer, ctx, peerWindow, rank.rankId, rowBytes,
                                              peerWindowLayout, 0U, 1U);
            M3NPublishAllDispatchExpertsReadyAfterGather(shape, workspaceView);
            M2BuildSwigluSyncMetadata(shape, workspaceView);
            StoreScalarI32(localPeer.debugCounters, 1);
            StoreScalarI32(localPeer.debugCounters + kM3CounterBase + 7U, static_cast<int32_t>(shape.expertPerRank));
            StoreScalarI32(localPeer.debugCounters + kM3NDispatchCounterBase + 0U, 1);
            StoreScalarI32(localPeer.debugCounters + kM3NDispatchCounterBase + 1U, 1);
            StoreScalarI32(localPeer.debugCounters + kM3NDispatchCounterBase + 2U, 1);
            StoreScalarI32(localPeer.debugCounters + kM3NDispatchCounterBase + 3U, 1);
            StoreScalarI32(localPeer.debugCounters + kM3NDispatchCounterBase + 10U, 1);
            StoreScalarI32(localPeer.debugCounters + kM3NDispatchCounterBase + 11U, 1);
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
                                  moe_new_dispatch_combine_a8w8::kM3N9TimeoutStageDispatchToGmm1,
                                  moe_new_dispatch_combine_a8w8::kM3N4DispatchToGmm1Flag, debugStopStage);
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
        moe_new_dispatch_combine_a8w8::M3N4SignalV2C<moe_new_dispatch_combine_a8w8::kM3N4DispatchToGmm1Flag>();
    }
    if (debugStopStage >= 21U && debugStopStage <= 23U) {
        return;
    }
    if (debugStopStage == 1U) {
        return;
    }
    bool m3nActivationEnabled = debugStopStage < 100U && M3NFusedDispatchScratchFits(shape);
    bool m3oActivationShardDebugProbe = M3OActivationShardDebugProbeEnabled(debugStopStage);
    bool m3nAicActivationScalar = m3nActivationEnabled && !m3n6Gmm1ActivationOverlap && !m3oActivationShardDebugProbe;
    if (m3n6Gmm1ActivationOverlap) {
        if (debugStopStage == 2U) {
            return;
        }
        M3NDispatchWorkerAssignment activationAssignment =
            M3OLoadDenseWorkerAssignment(rawAivSlot, logicalAivCount, kM3NFusedDispatchMaxLaneSlots);
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
            M3NRecordActivationLaneDebug(localPeer, rawAivSlot, activationAssignment.workerId, 0U, syncGroupCount,
                                         false);
            M3NInitActivationScaleShard(shape, workspaceView, activationAssignment.workerId, activationWorkerCount);
        }
        moe_new_dispatch_combine_a8w8::M3N4AivAllDoneCoarseSync();
        for (uint32_t syncIdx = 0; syncIdx < syncGroupCount; ++syncIdx) {
            bool recordSwiglu = timelineEnable != 0U && IsM2FusedMainAiv() &&
                                syncIdx < moe_new_dispatch_combine_a8w8::kM3N12TimelineSwigluSlotCount;
            uint64_t swigluBegin = recordSwiglu ? M3N12GetSysCnt() : 0U;
            moe_new_dispatch_combine_a8w8::M3N6WaitGmm1SyncGroupReady(syncIdx);
            if (IsM2FusedMainAiv()) {
                M2FusedRecordStage(stageStatus + kM2FusedFullStageBaseSlot, 1U, 1);
                M3N6RecordActivationStartedBeforeLastGmm1Ready(stageStatus, localPeer, syncIdx, syncGroupCount);
                M3N6RunGmm1EpilogueSyncGroupScalar(shape, workspaceView, syncIdx);
                StoreScalarI32(localPeer.debugCounters + kM3CounterBase + 8U,
                               LoadScalarI32(localPeer.debugCounters + kM3CounterBase + 8U) + 1);
            }
            moe_new_dispatch_combine_a8w8::M3N4AivAllDoneCoarseSync();
            if (activeActivationWorker) {
                uint32_t rowsProcessed = M3NRunActivationQuantSyncGroupShardScalar(
                    shape, workspaceView, syncIdx, activationAssignment.workerId, activationWorkerCount);
                M3NRecordActivationLaneDebug(localPeer, rawAivSlot, activationAssignment.workerId, rowsProcessed,
                                             syncGroupCount, true);
            }
            moe_new_dispatch_combine_a8w8::M3N4AivAllDoneCoarseSync();
            if (IsM2FusedMainAiv()) {
                M3NActivationLaneSummary activationSummary =
                    M3NAggregateActivationLaneDebug(localPeer, logicalAivCount);
                M3N6FinalizeActivationQuantSyncGroup(
                    shape, workspaceView, localPeer, syncIdx, activationWorkerCount, activationSummary.activeWorkerMask,
                    activationSummary.rowsProcessed, syncIdx == 0U, syncIdx + 1U == syncGroupCount);
                StoreScalarI32(localPeer.debugCounters + kM3CounterBase + 9U,
                               LoadScalarI32(localPeer.debugCounters + kM3CounterBase + 9U) + 1);
                if (recordSwiglu) {
                    __gm__ int32_t *group = workspaceView.swigluGroupDesc + syncIdx * kM2SwigluGroupFields;
                    int32_t groupId = LoadScalarI32(group + 0U);
                    int32_t expertBegin = LoadScalarI32(group + 1U);
                    int32_t expertEnd = LoadScalarI32(group + 2U);
                    int32_t rowBegin = LoadScalarI32(group + 3U);
                    int32_t rowEnd = LoadScalarI32(group + 4U);
                    M3N12RecordTimeline(
                        workspaceView.timelineScratch,
                        moe_new_dispatch_combine_a8w8::kM3N12TimelineSlotSwigluBase + syncIdx, swigluBegin,
                        M3N12GetSysCnt(),
                        M3N12PackMeta0(moe_new_dispatch_combine_a8w8::M3N12TimelineKind::kSwigluGroup,
                                       moe_new_dispatch_combine_a8w8::M3N12TimelineCoreType::kAiv, logicalAiv,
                                       moe_new_dispatch_combine_a8w8::M3N12TimelineStatus::kProcessed, syncIdx,
                                       static_cast<uint32_t>(groupId < 0 ? 0 : groupId)),
                        M3N12PackMeta1(static_cast<uint32_t>(rowBegin < 0 ? 0 : rowBegin),
                                       static_cast<uint32_t>(rowEnd < 0 ? 0 : rowEnd),
                                       static_cast<uint32_t>(expertBegin < 0 ? 0 : expertBegin),
                                       static_cast<uint32_t>(expertEnd < 0 ? 0 : expertEnd)));
                }
            }
            moe_new_dispatch_combine_a8w8::M3N4AivAllDoneCoarseSync();
        }
        if (debugStopStage == 3U) {
            return;
        }
    } else {
        moe_new_dispatch_combine_a8w8::M3N4WaitC2V<moe_new_dispatch_combine_a8w8::kM3N4Gmm1ToEpilogueFlag>();
        if (debugStopStage == 24U) {
            return;
        }
        if (debugStopStage == 2U) {
            return;
        }
        if (IsM2FusedMainAiv()) {
            M2FusedRecordStage(stageStatus + kM2FusedFullStageBaseSlot, 1U, 1);
            M2FusedRunGmm1EpilogueScalar(shape, workspaceView);
            StoreScalarI32(localPeer.debugCounters + kM3CounterBase + 8U,
                           LoadScalarI32(workspaceView.swigluSyncGroups));
        }
        pipe_barrier(PIPE_ALL);
        dsb(DSB_DDR);
        moe_new_dispatch_combine_a8w8::M3N4AivAllDoneCoarseSync();
        moe_new_dispatch_combine_a8w8::M3N4SignalV2C<
            moe_new_dispatch_combine_a8w8::kM3N4Gmm1EpilogueToActivationFlag>();
        if (debugStopStage == 3U) {
            return;
        }
        if (m3nActivationEnabled && !m3nAicActivationScalar) {
            M3NDispatchWorkerAssignment activationAssignment =
                M3OLoadDenseWorkerAssignment(rawAivSlot, logicalAivCount, kM3NFusedDispatchMaxLaneSlots);
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
                M3NRecordActivationLaneDebug(localPeer, rawAivSlot, activationAssignment.workerId, 0U, syncGroupCount,
                                             false);
            }
            M3NDispatchAivOnlyPhaseSync();
            if (debugStopStage == 41U) {
                return;
            }
            if (activeActivationWorker) {
                M3NInitActivationScaleShard(shape, workspaceView, activationAssignment.workerId, activationWorkerCount);
            }
            M3NDispatchAivOnlyPhaseSync();
            if (debugStopStage == 42U) {
                return;
            }
            if (activeActivationWorker && debugStopStage != 44U) {
                uint32_t activationMaxRows = debugStopStage == 45U ? 1U : 0xFFFFFFFFU;
                uint32_t activationProbeMode = 0U;
                if (debugStopStage >= 46U && debugStopStage <= 56U) {
                    activationMaxRows = 1U;
                    activationProbeMode = debugStopStage - 45U;
                }
                uint32_t rowsProcessed =
                    M3NRunActivationQuantShardScalar(shape, workspaceView, activationAssignment.workerId,
                                                     activationWorkerCount, activationMaxRows, activationProbeMode);
                M3NRecordActivationLaneDebug(localPeer, rawAivSlot, activationAssignment.workerId, rowsProcessed,
                                             syncGroupCount, true);
            }
            M3NDispatchAivOnlyPhaseSync();
            if (debugStopStage >= 43U && debugStopStage <= 56U) {
                if (IsM2FusedMainAiv()) {
                    M3NActivationLaneSummary activationSummary =
                        M3NAggregateActivationLaneDebug(localPeer, logicalAivCount);
                    StoreScalarI32(localPeer.debugCounters + kM3NActivationCounterBase + 0U,
                                   static_cast<int32_t>(activationSummary.activeWorkers));
                    StoreScalarI32(localPeer.debugCounters + kM3NActivationCounterBase + 1U,
                                   static_cast<int32_t>(activationSummary.rowsProcessed));
                    StoreScalarI32(localPeer.debugCounters + kM3NActivationCounterBase + 7U,
                                   activationSummary.activeWorkers > 0U ? 1 : 0);
                    StoreScalarI32(localPeer.debugCounters + kM3NActivationCounterBase + 15U,
                                   static_cast<int32_t>(activationSummary.activeWorkerMask));
                }
                return;
            }
            if (IsM2FusedMainAiv()) {
                M3NActivationLaneSummary activationSummary =
                    M3NAggregateActivationLaneDebug(localPeer, logicalAivCount);
                if (activationSummary.activeWorkers == 0U) {
                    M2RunActivationQuantScalar(shape, workspaceView);
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
        } else if (m3nAicActivationScalar) {
            if (IsM2FusedMainAiv()) {
                M2FusedRecordStage(stageStatus + kM2FusedFullStageBaseSlot, 2U, 1);
            }
        } else if (IsM2FusedMainAiv()) {
            M2FusedRecordStage(stageStatus + kM2FusedFullStageBaseSlot, 2U, 1);
            M2RunActivationQuantScalar(shape, workspaceView);
            StoreScalarI32(localPeer.debugCounters + kM3CounterBase + 9U,
                           LoadScalarI32(workspaceView.swigluSyncGroups));
        }
    }
    pipe_barrier(PIPE_ALL);
    dsb(DSB_DDR);
    if (debugStopStage == 4U) {
        return;
    }
    if (!m3n8Gmm2CombineOverlap) {
        moe_new_dispatch_combine_a8w8::M3N4WaitC2V<moe_new_dispatch_combine_a8w8::kM3N4Gmm2ToCombineFlag>();
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
        M3NDispatchWorkerAssignment combineAssignment =
            M3OLoadDenseWorkerAssignment(rawAivSlot, logicalAivCount, kM3NDispatchMaxWorkers);
        uint32_t combineWorkerCount = combineAssignment.workerCount;
        bool activeCombineWorker = combineAssignment.active;
        if (IsM2FusedMainAiv()) {
            M2FusedRecordStage(stageStatus + kM2FusedFullStageBaseSlot, 3U, 1);
            if (m3n8Gmm2CombineOverlap) {
                M3N8InitCombineCounters(localPeer, combineWorkerCount, shape.expertPerRank);
            } else {
                if (debugStopStage == 58U) {
                    M3NDispatchAivOnlyPhaseSync();
                    return;
                }
                M3NInitCombineCounters(localPeer, combineWorkerCount);
                if (debugStopStage == 57U) {
                    M3NDispatchAivOnlyPhaseSync();
                    return;
                }
                if (debugStopStage >= 59U && debugStopStage <= 67U) {
                    M2DebugProbeReturnSegmentMap(shape, workspaceView, localPeer, rank.rankId, debugStopStage);
                    M3NDispatchAivOnlyPhaseSync();
                    return;
                }
                M2BuildReturnSegmentMap(shape, workspaceView, localPeer, rank.rankId);
            }
        }
        if (!m3n8Gmm2CombineOverlap) {
            M3NDispatchAivOnlyPhaseSync();
        } else {
            M3N8AivOnlyPhaseSync();
        }
        if (debugStopStage == 51U || debugStopStage == 57U || debugStopStage == 58U ||
            (debugStopStage >= 59U && debugStopStage <= 67U)) {
            return;
        }
        bool materializeEnabled = debugStopStage != 56U;
        bool transferEnabled = debugStopStage != 53U;
        bool notifyEnabled = debugStopStage != 53U && debugStopStage != 54U;
        bool recordCombine = timelineEnable != 0U && activeCombineWorker && combineAssignment.workerId == 0U;
        uint64_t combineBegin = recordCombine ? M3N12GetSysCnt() : 0U;
        if (m3n8Gmm2CombineOverlap) {
            M3N8RunGmm2EpilogueAndReturnByExpert(shape, workspaceView, localPeer, ctx, peerWindow, rank.rankId,
                                                 peerWindowLayout, combineAssignment.workerId, combineWorkerCount,
                                                 activeCombineWorker, IsM2FusedMainAiv(), materializeEnabled,
                                                 transferEnabled, notifyEnabled);
            if (activeCombineWorker) {
                StoreScalarI32(localPeer.debugCounters + kM3N8CombineCounterBase + 11U,
                               LoadScalarI32(localPeer.debugCounters + kM3N8CombineCounterBase + 11U) + 1);
            }
            if (recordCombine) {
                uint32_t segmentCap = static_cast<uint32_t>(M2ReturnSegmentCapacity(shape));
                M3N12RecordAivRange(workspaceView.timelineScratch,
                                    moe_new_dispatch_combine_a8w8::kM3N12TimelineSlotCombine,
                                    moe_new_dispatch_combine_a8w8::M3N12TimelineKind::kCombineOwnerSegment, logicalAiv,
                                    combineAssignment.workerId, 0U, segmentCap, 1U, 0U, combineBegin, M3N12GetSysCnt());
            }
            M3N8AivOnlyPhaseSync();
        } else if (activeCombineWorker) {
            M3NCombineShardStats combineStats =
                M3NRunGmm2EpilogueAndReturnShard(shape, workspaceView, localPeer, ctx, peerWindow, rank.rankId,
                                                 peerWindowLayout, combineAssignment.workerId, combineWorkerCount,
                                                 materializeEnabled, transferEnabled, notifyEnabled, debugStopStage);
            if (recordCombine) {
                uint32_t segmentCap = static_cast<uint32_t>(M2ReturnSegmentCapacity(shape));
                M3N12RecordAivRange(
                    workspaceView.timelineScratch, moe_new_dispatch_combine_a8w8::kM3N12TimelineSlotCombine,
                    moe_new_dispatch_combine_a8w8::M3N12TimelineKind::kCombineOwnerSegment, logicalAiv,
                    combineAssignment.workerId, 0U, segmentCap, static_cast<uint32_t>(combineStats.segmentCount),
                    static_cast<uint32_t>(combineStats.skippedSegmentCount), combineBegin, M3N12GetSysCnt());
            }
            (void)combineStats;
        }
        if (!m3n8Gmm2CombineOverlap) {
            M3NDispatchAivOnlyPhaseSync();
        }
        if (debugStopStage == 52U || debugStopStage == 53U || debugStopStage == 54U || debugStopStage == 56U ||
            (debugStopStage >= 68U && debugStopStage <= 86U)) {
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
    moe_new_dispatch_combine_a8w8::M3N4AivAllDoneCoarseSync();
    moe_new_dispatch_combine_a8w8::M3N4SignalV2C<moe_new_dispatch_combine_a8w8::kM3N4CombineToRestoreFlag>();
    if (debugStopStage == 6U) {
        return;
    }
    if (IsM2FusedMainAiv()) {
        M3OClearRestoreCounters(localPeer);
    }
    moe_new_dispatch_combine_a8w8::M3N4AivAllDoneCoarseSync();
    M3ORestoreShardStats restoreStats{0U, 0U, 0U};
    if (logicalAiv < 8U) {
        uint32_t tokenBegin = TokenShardBegin(shape.m, logicalAiv, 8U);
        uint32_t tokenEnd = TokenShardEnd(shape.m, logicalAiv, 8U);
        bool recordRestore = timelineEnable != 0U && logicalAiv == 0U;
        uint64_t restoreBegin = recordRestore ? M3N12GetSysCnt() : 0U;
        __gm__ float *probValues = reinterpret_cast<__gm__ float *>(probs);
        __gm__ half *output = reinterpret_cast<__gm__ half *>(outputC);
        int32_t outputRowStride = static_cast<int32_t>(shape.hiddenSize);
        int32_t returnRowStride = static_cast<int32_t>(peerWindowLayout.returnPayloadRowBytes / sizeof(half));
        for (uint32_t token = tokenBegin; token < tokenEnd; ++token) {
            ++restoreStats.tokenCount;
            StoreZeroRowHalfScalar(output, outputRowStride, static_cast<int32_t>(token),
                                   static_cast<int32_t>(shape.hiddenSize));
            for (uint32_t slot = 0; slot < shape.topK; ++slot) {
                uint32_t routeIndex = token * shape.topK + slot;
                int32_t ptrDRow = LoadScalarI32(workspaceView.expandedRowIdx + routeIndex);
                if (ptrDRow < 0 || static_cast<uint32_t>(ptrDRow) >= static_cast<uint32_t>(M2LocalRows(shape))) {
                    ++restoreStats.skippedRouteCount;
                    continue;
                }
                ++restoreStats.routeCount;
                float prob = probValues[routeIndex];
                AddWeightedRowHalfStrideScalar(output, outputRowStride, static_cast<int32_t>(token),
                                               localPeer.returnPayload, returnRowStride, ptrDRow,
                                               static_cast<int32_t>(shape.hiddenSize), prob);
            }
        }
        M3ORecordRestoreWorker(localPeer, logicalAiv, tokenBegin, tokenEnd, restoreStats);
        if (recordRestore) {
            M3N12RecordAivRange(workspaceView.timelineScratch, moe_new_dispatch_combine_a8w8::kM3N12TimelineSlotRestore,
                                moe_new_dispatch_combine_a8w8::M3N12TimelineKind::kRestore, logicalAiv, logicalAiv,
                                tokenBegin, tokenEnd, restoreStats.routeCount, restoreStats.skippedRouteCount,
                                restoreBegin, M3N12GetSysCnt());
        }
    }
    moe_new_dispatch_combine_a8w8::M3N4AivAllDoneCoarseSync();
    if (IsM2FusedMainAiv()) {
        uint32_t totalTokens = 0U;
        uint32_t totalRoutes = 0U;
        uint32_t totalSkippedRoutes = 0U;
        uint32_t activeRestoreWorkers = 0U;
        for (uint32_t worker = 0; worker < moe_new_dispatch_combine_a8w8::kM3ORestoreWorkerCount; ++worker) {
            uint32_t base = kM2FusedFullM3ORestoreCounterBase + kM2FusedFullM3ORestoreWorkerBase +
                            worker * kM2FusedFullM3ORestoreWorkerWords;
            InvalidateGmCacheLines(localPeer.debugCounters + base, 64U);
            int32_t active = LoadScalarI32(localPeer.debugCounters + base + 5U);
            if (active == 0) {
                continue;
            }
            ++activeRestoreWorkers;
            totalTokens += static_cast<uint32_t>(LoadScalarI32(localPeer.debugCounters + base + 2U));
            totalRoutes += static_cast<uint32_t>(LoadScalarI32(localPeer.debugCounters + base + 3U));
            totalSkippedRoutes += static_cast<uint32_t>(LoadScalarI32(localPeer.debugCounters + base + 4U));
        }
        M3OFinalizeRestoreCounters(localPeer, activeRestoreWorkers, totalTokens, totalRoutes, totalSkippedRoutes);
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

void LaunchM2FusedFullWithHandle(const void *anchor,
                                 moe_new_dispatch_combine_a8w8::M2FusedFullLaunchArgs *launchArgsDevice,
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

void LaunchM2FusedFull(moe_new_dispatch_combine_a8w8::M2FusedFullLaunchArgs *launchArgsDevice, uint32_t aicBlocks,
                       uint32_t aivRatio, void *stream)
{
    LaunchM2FusedFullWithHandle(reinterpret_cast<const void *>(&LaunchM2FusedFull), launchArgsDevice, aicBlocks,
                                aivRatio, stream);
}

} // namespace dispatch_combine_tile
#endif
