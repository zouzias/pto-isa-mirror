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
constexpr uint32_t kM2FusedSkeletonAivStageBase =
    kM2FusedSkeletonAicStageBase +
    kM2FusedSkeletonStageCount * kM2FusedSkeletonParticipantCount * kM2FusedSkeletonStageSlotWords;
constexpr uint32_t kM2FusedFullAicHeaderSlot = 8U * 16U;
constexpr uint32_t kM2FusedFullAivHeaderSlot = 9U * 16U;
constexpr uint32_t kM2FusedFullStageBaseSlot = 10U * 16U;
constexpr uint32_t kM2FusedFullAicStageBaseSlot = 11U * 16U;
constexpr uint32_t kM2FusedFullAivStageBaseSlot = 12U * 16U;
constexpr uint32_t kM2FusedFullDebugStopSlot = 15U * 16U;
constexpr uint32_t kM2FusedFullStageCount = 7U;

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
        static_cast<uint32_t>(
            M2FusedLoadConfigI32(config, moe_dispatch_combine_a8w8::kM2FusedFullShapeRankNumSlot)),
        static_cast<uint32_t>(
            M2FusedLoadConfigI32(config, moe_dispatch_combine_a8w8::kM2FusedFullShapeExpertPerRankSlot)),
        static_cast<uint32_t>(
            M2FusedLoadConfigI32(config, moe_dispatch_combine_a8w8::kM2FusedFullShapeTopKSlot)),
        static_cast<uint32_t>(M2FusedLoadConfigI32(config, moe_dispatch_combine_a8w8::kM2FusedFullShapeMSlot)),
        static_cast<uint32_t>(
            M2FusedLoadConfigI32(config, moe_dispatch_combine_a8w8::kM2FusedFullShapeHiddenSizeSlot)),
        static_cast<uint32_t>(
            M2FusedLoadConfigI32(config, moe_dispatch_combine_a8w8::kM2FusedFullShapeIntermediateSizeSlot)),
        static_cast<uint32_t>(
            M2FusedLoadConfigI32(config, moe_dispatch_combine_a8w8::kM2FusedFullShapeMaxTokensPerExpertSlot)),
        static_cast<uint32_t>(
            M2FusedLoadConfigI32(config, moe_dispatch_combine_a8w8::kM2FusedFullShapePayloadTileColsSlot)),
        static_cast<uint32_t>(
            M2FusedLoadConfigI32(config, moe_dispatch_combine_a8w8::kM2FusedFullShapeGmmBlockMSlot)),
        static_cast<uint32_t>(
            M2FusedLoadConfigI32(config, moe_dispatch_combine_a8w8::kM2FusedFullShapeGmmBlockNSlot)),
        static_cast<uint32_t>(
            M2FusedLoadConfigI32(config, moe_dispatch_combine_a8w8::kM2FusedFullShapeGmmBlockKSlot)),
        static_cast<uint32_t>(
            M2FusedLoadConfigI32(config, moe_dispatch_combine_a8w8::kM2FusedFullShapeDtypeInSlot)),
        static_cast<uint32_t>(
            M2FusedLoadConfigI32(config, moe_dispatch_combine_a8w8::kM2FusedFullShapeDtypeOutSlot)),
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

AICORE inline uint32_t M2FusedLoadDebugStopStage(
    __gm__ moe_dispatch_combine_a8w8::M2FusedFullLaunchArgs *launchArgs)
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
    const bool inRange = participantIdx >= 0 &&
                         participantIdx < static_cast<int32_t>(kM2FusedSkeletonParticipantCount);
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
PTO_SYNCALL_MIX_AIC_KERNEL_META(M2MixedSpike_2801_mix_aic, 1, 1);
PTO_SYNCALL_MIX_AIC_KERNEL_META(M2FusedSkeleton_2802_mix_aic, 1, 1);
PTO_SYNCALL_MIX_AIC_KERNEL_META(M2FusedFull_2803_mix_aic, 1, 1);

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
        int32_t rankNum = M2FusedLoadConfigI32(earlyStageStatus, moe_dispatch_combine_a8w8::kM2FusedFullShapeRankNumSlot);
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
    GM_ADDR workspace =
        reinterpret_cast<GM_ADDR>(M2FusedLoadConfigU64(stageStatus, moe_dispatch_combine_a8w8::kM2FusedFullPtrWorkspaceSlot));
    auto layout = M2FusedGmmMakeWorkspaceLayoutDevice(shape);
    (void)layout;
    uint32_t debugStopStage = static_cast<uint32_t>(M2FusedGmmLoadScalarI32(stageStatus + kM2FusedFullDebugStopSlot));
    if (debugStopStage == 0U) {
        debugStopStage = M2FusedLoadDebugStopStage(launchArgs);
    }
    if (get_block_idx() == 0) {
        M2FusedRecordStage(stageStatus + kM2FusedFullAicHeaderSlot, 0U, kM2FusedFullMagic);
        M2FusedRecordStage(stageStatus + kM2FusedFullAicHeaderSlot, 1U, static_cast<int32_t>(get_block_num()));
        M2FusedRecordStage(stageStatus + kM2FusedFullAicHeaderSlot, 2U, static_cast<int32_t>(rank.rankId));
        M2FusedRecordStage(stageStatus + kM2FusedFullAicHeaderSlot, 3U, static_cast<int32_t>(shape.rankNum));
    }
    pto::SYNCALL<pto::SyncCoreType::Mix>();
    if (debugStopStage == 1U || debugStopStage >= 100U) {
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
    pipe_barrier(PIPE_ALL);
    dsb(DSB_DDR);
    pto::SYNCALL<pto::SyncCoreType::Mix>();
    if (debugStopStage == 2U) {
        return;
    }
    pto::SYNCALL<pto::SyncCoreType::Mix>();
    if (debugStopStage == 3U) {
        return;
    }
    pto::SYNCALL<pto::SyncCoreType::Mix>();
    if (debugStopStage == 4U) {
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
        __gm__ int32_t *gmm2TileTaskPlan =
            reinterpret_cast<__gm__ int32_t *>(workspace + layout.gmm2TileTaskPlan.offset);
        uint32_t taskCount = M2FusedGmmBuildTileTaskPlan(shape, dispatchOffset, expertTokenNums, gmm2TileTaskPlan,
                                                         shape.hiddenSize, shape.intermediateSize, 2U);
        if (get_block_idx() == 0) {
            M2FusedGmmStoreScalarI32(stageStatus + 6U * 16U, static_cast<int32_t>(taskCount));
            M2FusedGmmStoreScalarI32(stageStatus + 7U * 16U, static_cast<int32_t>(get_block_num()));
        }
        for (uint32_t localExpert = 0; localExpert < shape.expertPerRank; ++localExpert) {
            int32_t rowCount = M2FusedGmmLoadScalarI32(expertTokenNums + localExpert);
            if (rowCount <= 0) {
                M2FusedGmmStoreScalarI32(gmm2GroupReady + localExpert * 16U, 1);
            }
        }
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
            M2FusedRunInt8GmmTile(tileInput, expertWeight + nBase, tileOutput + nBase, mValid, shape.intermediateSize,
                                  nValid, rowBytes, shape.hiddenSize, shape.hiddenSize);
        }
        pipe_barrier(PIPE_ALL);
        dsb(DSB_DDR);
        if (get_block_idx() == 0) {
            for (uint32_t localExpert = 0; localExpert < shape.expertPerRank; ++localExpert) {
                M2FusedGmmStoreScalarI32(gmm2GroupReady + localExpert * 16U, 1);
            }
        }
    }
    pipe_barrier(PIPE_ALL);
    dsb(DSB_DDR);
    pto::SYNCALL<pto::SyncCoreType::Mix>();
    if (debugStopStage == 5U) {
        return;
    }
    pto::SYNCALL<pto::SyncCoreType::Mix>();
    if (debugStopStage == 6U) {
        return;
    }
}
#endif

#if defined(M2_MIXED_SPIKE_BUILD_AIV)
PTO_SYNCALL_MIX_AIC_KERNEL_META(M2MixedSpike_2801_mix_aiv, 1, 1);
PTO_SYNCALL_MIX_AIC_KERNEL_META(M2FusedSkeleton_2802_mix_aiv, 1, 1);
PTO_SYNCALL_MIX_AIC_KERNEL_META(M2FusedFull_2803_mix_aiv, 1, 1);

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
        int32_t rankNum = M2FusedLoadConfigI32(earlyStageStatus, moe_dispatch_combine_a8w8::kM2FusedFullShapeRankNumSlot);
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
    GM_ADDR inputA =
        reinterpret_cast<GM_ADDR>(M2FusedLoadConfigU64(stageStatusBase, moe_dispatch_combine_a8w8::kM2FusedFullPtrInputASlot));
    GM_ADDR expertIdx = reinterpret_cast<GM_ADDR>(
        M2FusedLoadConfigU64(stageStatusBase, moe_dispatch_combine_a8w8::kM2FusedFullPtrExpertIdxSlot));
    GM_ADDR probs =
        reinterpret_cast<GM_ADDR>(M2FusedLoadConfigU64(stageStatusBase, moe_dispatch_combine_a8w8::kM2FusedFullPtrProbsSlot));
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
    uint32_t logicalAiv = M2FusedLogicalAivId();
    if (IsM2FusedMainAiv()) {
        M2FusedRecordStage(stageStatus + kM2FusedFullAivHeaderSlot, 0U, kM2FusedFullMagic);
        M2FusedRecordStage(stageStatus + kM2FusedFullAivHeaderSlot, 1U,
                           static_cast<int32_t>(get_block_num() * get_subblockdim()));
        M2FusedRecordStage(stageStatus + kM2FusedFullAivHeaderSlot, 2U, static_cast<int32_t>(rank.rankId));
        M2FusedRecordStage(stageStatus + kM2FusedFullAivHeaderSlot, 3U, static_cast<int32_t>(shape.rankNum));
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
    if (IsM2FusedMainAiv()) {
        uint32_t rowBytes = static_cast<uint32_t>(peerWindowLayout.dispatchPayloadRowBytes);
        M2FusedRecordStage(stageStatus + kM2FusedFullStageBaseSlot, 5U, 100);
        if (debugStopStage != 100U) {
            M2ClearDispatchState(shape, workspaceView, localPeer);
            M2FusedRecordStage(stageStatus + kM2FusedFullStageBaseSlot, 5U, 101);
        }
        if (debugStopStage != 100U && debugStopStage != 101U) {
            M2CountLocalRoutes(shape, workspaceView, localPeer, expertIdx, rank.rankId);
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
                M2RoutePackQuantLocal(shape, workspaceView, localPeer, inputA, expertIdx, rowBytes);
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
            debugStopStage != 103U && debugStopStage != 104U && debugStopStage != 105U &&
            debugStopStage != 106U) {
            M2GatherDispatchToGmm1Input(shape, workspaceView, localPeer, ctx, peerWindow, rank.rankId, rowBytes,
                                        peerWindowLayout);
            StoreScalarI32(workspaceView.swigluSyncGroups, static_cast<int32_t>(shape.expertPerRank));
            StoreScalarI32(workspaceView.dequantSum, 0);
            StoreScalarI32(workspaceView.dequantSum + 1, LoadScalarI32(workspaceView.expertTokenNums));
            StoreScalarI32(localPeer.debugCounters, 1);
            M2FusedRecordStage(stageStatus + kM2FusedFullStageBaseSlot, 5U, 107);
        }
    }
    M2FusedRecordStage(stageStatus + kM2FusedFullAivStageBaseSlot, logicalAiv, 100);
    pipe_barrier(PIPE_ALL);
    dsb(DSB_DDR);
    pto::SYNCALL<pto::SyncCoreType::Mix>();
    if (debugStopStage == 1U || debugStopStage >= 100U) {
        return;
    }
    pto::SYNCALL<pto::SyncCoreType::Mix>();
    if (debugStopStage == 2U) {
        return;
    }
    if (IsM2FusedMainAiv()) {
        M2FusedRecordStage(stageStatus + kM2FusedFullStageBaseSlot, 1U, 1);
        M2RunGmm1Epilogue(shape, workspaceView, rank.rankId);
    }
    pipe_barrier(PIPE_ALL);
    dsb(DSB_DDR);
    pto::SYNCALL<pto::SyncCoreType::Mix>();
    if (debugStopStage == 3U) {
        return;
    }
    if (IsM2FusedMainAiv()) {
        M2FusedRecordStage(stageStatus + kM2FusedFullStageBaseSlot, 2U, 1);
        M2RunActivationQuant(shape, workspaceView, rank.rankId);
    }
    pipe_barrier(PIPE_ALL);
    dsb(DSB_DDR);
    pto::SYNCALL<pto::SyncCoreType::Mix>();
    if (debugStopStage == 4U) {
        return;
    }
    pto::SYNCALL<pto::SyncCoreType::Mix>();
    if (debugStopStage == 5U) {
        return;
    }
    if (IsM2FusedMainAiv()) {
        M2FusedRecordStage(stageStatus + kM2FusedFullStageBaseSlot, 3U, 1);
        M2RunGmm2EpilogueAndReturn(shape, workspaceView, localPeer, ctx, peerWindow, rank.rankId, peerWindowLayout);
    }
    pipe_barrier(PIPE_ALL);
    dsb(DSB_DDR);
    pto::SYNCALL<pto::SyncCoreType::Mix>();
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
                if (ptrDRow < 0) {
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
                    GlobalNd<half> outGlobal =
                        MakeGlobal2D(output + static_cast<int64_t>(token) * outputRowStride + col, 1, cols,
                                     outputRowStride);
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
    ret = rtKernelLaunchWithHandleV2(handle, kM2FusedSkeletonTilingKey, aicBlocks, &argsInfo, nullptr, stream,
                                     &cfgInfo);
    if (ret != RT_ERROR_NONE) {
        std::fprintf(stderr, "rtKernelLaunchWithHandleV2 failed for M2 fused skeleton, ret=%d\n", ret);
        std::abort();
    }
}

void LaunchM2FusedFullWithHandle(const void *anchor,
                                 moe_dispatch_combine_a8w8::M2FusedFullLaunchArgs *launchArgsDevice,
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
