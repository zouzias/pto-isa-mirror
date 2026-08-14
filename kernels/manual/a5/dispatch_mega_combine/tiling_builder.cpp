/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include "tiling_builder.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>

#include "op_kernel/utils/const_args.hpp"
#include "op_kernel/utils/mega_wave_schedule.hpp"

namespace {
constexpr uint32_t kDispatchGatherPackedTileCols = 8192U;
constexpr uint32_t kDispatchMinBufferCount = 2U;
constexpr uint32_t kDispatchMaxBufferCount = 6U;
constexpr uint32_t kDispatchBaseRouteItemsPerBatch = 12288U;
constexpr uint32_t kDispatchRouteItemAlignment = 256U;
constexpr uint32_t kDispatchMetaSlotBytes = 32U;
constexpr uint32_t kDispatchRouteCountBytes = 32U;
constexpr uint32_t kDispatchMaskLoadGuardBytes = 256U;
constexpr uint32_t kMaxCombineVecTileElems = 8192U;
constexpr uint32_t kMaxUnpermuteVecTileElems = 8192U;
constexpr uint32_t kHalfDataBlockElems = 16U;
constexpr uint32_t kFrontMetadataSortRunMaxElems = 6144U;
constexpr uint32_t kFrontMetadataSortAlignElems = 32U;
constexpr uint32_t kFrontMetadataSortOutLoopElems = 2040U;

uint64_t AlignUp(uint64_t value, uint64_t align);

uint64_t CheckedMul(uint64_t lhs, uint64_t rhs, const char *name);

uint64_t CheckedAdd(uint64_t lhs, uint64_t rhs, const char *name);

uint32_t Pow4Ceil(uint32_t value)
{
    uint32_t result = 1U;
    while (result < value) {
        if (result > UINT32_MAX / 4U) {
            throw std::runtime_error("front metadata sort run count overflows uint32");
        }
        result *= 4U;
    }
    return result;
}

uint32_t CeilDivU32(uint32_t value, uint32_t divisor)
{
    return value / divisor + static_cast<uint32_t>(value % divisor != 0U);
}

// Rows without shape keys are the defaults for an AIC count. Exact shape rows
// override them. AIV roles are derived because mixed launch has two AIV subblocks.
constexpr A5FixedScheduleConfig kCanonicalShapeSchedules[] = {
    {.physicalAicNum = 28U, .dispatchGroupSize = 20U, .gmm1GroupSize = 20U, .gmm2GroupSize = 8U},
    {.physicalAicNum = 32U, .dispatchGroupSize = 21U, .gmm1GroupSize = 21U, .gmm2GroupSize = 11U},
    {.physicalAicNum = 36U, .dispatchGroupSize = 24U, .gmm1GroupSize = 24U, .gmm2GroupSize = 12U},
    {.epSize = 8U,
     .shapeConfigM = 2048U,
     .expertPerRank = 16U,
     .physicalAicNum = 36U,
     .dispatchGroupSize = 24U,
     .gmm1GroupSize = 22U,
     .gmm2GroupSize = 14U},
};

constexpr bool IsDefaultSchedule(const A5FixedScheduleConfig &schedule)
{
    return schedule.epSize == 0U && schedule.shapeConfigM == 0U && schedule.expertPerRank == 0U;
}

bool IsCanonicalShapeFamily(const CaseConfig &cfg)
{
    const bool canonicalExpertShape =
        cfg.expert_per_rank == 16U || (cfg.expert_per_rank == 4U && cfg.world_size == 8U && cfg.m == 512U);
    return cfg.k == 7168U && cfg.n == 4096U && cfg.topk == 8U && canonicalExpertShape &&
           (cfg.world_size == 8U || cfg.world_size == 16U);
}

bool MatchesShape(const A5FixedScheduleConfig &schedule, const CaseConfig &cfg)
{
    return !IsDefaultSchedule(schedule) && schedule.epSize == cfg.world_size && schedule.shapeConfigM == cfg.m &&
           schedule.expertPerRank == cfg.expert_per_rank && schedule.physicalAicNum == cfg.aic_num;
}

void RequireMxShape(uint32_t k, uint32_t n)
{
    if (n == 0U || (n & 1U) != 0U) {
        throw std::runtime_error("N must be positive and even for SwiGLU");
    }
    if (k % 128U != 0U || (n / 2U) % 128U != 0U) {
        throw std::runtime_error("MXFP8 GMM reduction dimensions must be multiples of 128");
    }
}

uint64_t MxQuantDataStorageBytes(uint32_t cols)
{
    return AlignUp(cols, kMegaMoeMxDataAlignmentBytes);
}

uint64_t MxQuantScaleCols(uint32_t cols)
{
    return cols / kMegaMoeMxGroupSize;
}

uint64_t MxQuantScaleStorageBytes(uint32_t cols)
{
    return AlignUp(MxQuantScaleCols(cols), kMegaMoeMxScaleAlignmentBytes);
}

uint64_t MxPackedRowStride(uint32_t cols)
{
    return MxQuantDataStorageBytes(cols) + MxQuantScaleStorageBytes(cols);
}

void RequireDispatchPackedRowCapacity(uint64_t packedRowStride)
{
    if (packedRowStride > kDispatchGatherPackedTileCols) {
        throw std::runtime_error("dispatch packed row exceeds the A5 vector tile width");
    }
}

void RequireFrontQuantUbCapacity(uint32_t k)
{
    auto alignUpBytes = [](uint64_t value) { return (value + UB_ALIGN - 1U) / UB_ALIGN * UB_ALIGN; };
    const uint64_t scaleCols = MxQuantScaleCols(k);
    const uint64_t oneBufferBytes = alignUpBytes(static_cast<uint64_t>(k) * sizeof(uint16_t)) +
                                    alignUpBytes(static_cast<uint64_t>(k)) + alignUpBytes(scaleCols) +
                                    alignUpBytes(scaleCols * sizeof(uint16_t)) * 2U;
    if (oneBufferBytes * 2U > A5_MAIN_UB_SIZE) {
        throw std::runtime_error("front quant ping-pong buffers exceed A5 main UB budget");
    }
}

uint64_t SwigluUbBytes(uint32_t n)
{
    auto alignUpBytes = [](uint64_t value) { return (value + UB_ALIGN - 1U) / UB_ALIGN * UB_ALIGN; };
    const uint64_t outputN = n / 2U;
    const uint64_t cBf16Bytes = alignUpBytes(CheckedMul(n, sizeof(uint16_t), "SwiGLU BF16 row"));
    const uint64_t cFp32Bytes = alignUpBytes(CheckedMul(n, sizeof(float), "SwiGLU FP32 row"));
    const uint64_t workFp32Bytes = alignUpBytes(CheckedMul(outputN, sizeof(float), "SwiGLU work row"));
    const uint64_t stageBytes =
        CheckedAdd(CheckedAdd(cBf16Bytes, cFp32Bytes, "SwiGLU UB stage"), workFp32Bytes, "SwiGLU UB stage");
    return CheckedMul(stageBytes, 2U, "SwiGLU ping-pong UB");
}

void RequireSwigluUbCapacity(uint32_t n)
{
    if (SwigluUbBytes(n) > A5_MAIN_UB_SIZE) {
        throw std::runtime_error("SwiGLU row ping-pong buffers exceed A5 main UB budget");
    }
}

uint64_t CombineUbBytes(uint32_t tileCols)
{
    (void)tileCols;
    return 2U * static_cast<uint64_t>(kMegaMoeCombineHbmBatchRows) * kMaxCombineVecTileElems * sizeof(uint16_t);
}

void RequireCombineUbCapacity(uint32_t tileCols)
{
    if (tileCols > kMaxCombineVecTileElems) {
        throw std::runtime_error("combine K exceeds the A5 vector tile width");
    }
    if (CombineUbBytes(tileCols) > A5_MAIN_UB_SIZE) {
        throw std::runtime_error("combine UB buffers exceed A5 main UB budget");
    }
}

uint64_t UnpermuteUbBytes(uint32_t metadataTokenRows, uint32_t topk, uint32_t tileCols,
                          uint32_t accumulatorBufferCount)
{
    auto alignUpBytes = [](uint64_t value) { return (value + UB_ALIGN - 1U) / UB_ALIGN * UB_ALIGN; };
    uint64_t ubBytes = 0;
    for (uint32_t i = 0; i < 2U; ++i) {
        ubBytes += alignUpBytes(static_cast<uint64_t>(metadataTokenRows) * topk * sizeof(int32_t));
        ubBytes += alignUpBytes(static_cast<uint64_t>(metadataTokenRows) * topk * sizeof(float));
    }
    for (uint32_t i = 0; i < accumulatorBufferCount; ++i) {
        ubBytes += alignUpBytes(static_cast<uint64_t>(tileCols) * sizeof(float));
    }
    for (uint32_t i = 0; i < 2U; ++i) {
        ubBytes += alignUpBytes(static_cast<uint64_t>(tileCols) * sizeof(uint16_t));
        ubBytes += alignUpBytes(static_cast<uint64_t>(tileCols) * sizeof(float));
    }
    ubBytes += alignUpBytes(static_cast<uint64_t>(tileCols) * sizeof(uint16_t));
    return ubBytes;
}

void RequireUnpermuteUbCapacity(uint32_t metadataTokenRows, uint32_t topk, uint32_t tileCols,
                                uint32_t accumulatorBufferCount)
{
    if (UnpermuteUbBytes(metadataTokenRows, topk, tileCols, accumulatorBufferCount) > A5_MAIN_UB_SIZE) {
        throw std::runtime_error("unpermute UB buffers exceed A5 main UB budget");
    }
}

uint32_t ChooseUnpermuteTileCols(uint32_t k, uint32_t metadataTokenRows, uint32_t topk,
                                 uint32_t accumulatorBufferCount)
{
    uint32_t tileCols = std::min(k, kMaxUnpermuteVecTileElems);
    tileCols = tileCols / kHalfDataBlockElems * kHalfDataBlockElems;
    while (tileCols > kHalfDataBlockElems &&
           UnpermuteUbBytes(metadataTokenRows, topk, tileCols, accumulatorBufferCount) > A5_MAIN_UB_SIZE) {
        tileCols -= kHalfDataBlockElems;
    }
    if (tileCols == 0U ||
        UnpermuteUbBytes(metadataTokenRows, topk, tileCols, accumulatorBufferCount) > A5_MAIN_UB_SIZE) {
        throw std::runtime_error("unable to choose unpermute tile cols within A5 main UB budget");
    }
    return tileCols;
}

uint64_t AlignUp(uint64_t value, uint64_t align)
{
    if (align == 0U) {
        throw std::runtime_error("AlignUp requires nonzero align");
    }
    if (value > UINT64_MAX - (align - 1U)) {
        throw std::runtime_error("AlignUp overflows uint64");
    }
    return (value + align - 1) / align * align;
}

void PopulateDispatchBufferTiling(MegaMoeDispatchTiling &dispatch, const MegaMoeFrontReorderTiling &front)
{
    if (front.routeElems == 0U || front.packedRowStride == 0U) {
        throw std::runtime_error("dispatch requires nonzero route items and packed row stride");
    }

    const uint64_t alignedTotalRouteItems = AlignUp(front.routeElems, kDispatchRouteItemAlignment);
    uint64_t routeItemsPerBatch = std::min<uint64_t>(alignedTotalRouteItems, kDispatchBaseRouteItemsPerBatch);
    const uint64_t copyBufferBytes = front.packedRowStride;
    const uint64_t dispatchSlotBytes = copyBufferBytes + kDispatchMetaSlotBytes;

    auto routeBufferBytes = [](uint64_t routeItems) {
        const uint64_t routeIndexBytes = AlignUp(routeItems * sizeof(uint32_t), UB_ALIGN);
        const uint64_t maskBytes = AlignUp((routeItems + 7U) / 8U, UB_ALIGN);
        return routeIndexBytes + maskBytes + kDispatchRouteCountBytes + kDispatchMaskLoadGuardBytes;
    };

    const uint64_t nonSlotBytes = routeBufferBytes(routeItemsPerBatch);
    const uint64_t slotBudget = nonSlotBytes < A5_MAIN_UB_SIZE ? A5_MAIN_UB_SIZE - nonSlotBytes : 0U;
    uint64_t bufferCount = std::min<uint64_t>(slotBudget / dispatchSlotBytes, kDispatchMaxBufferCount);
    if (bufferCount < kDispatchMinBufferCount) {
        throw std::runtime_error("dispatch cannot fit the minimum two token ring slots in A5 main UB");
    }

    if (routeItemsPerBatch < alignedTotalRouteItems) {
        const uint64_t fixedBytes =
            bufferCount * dispatchSlotBytes + kDispatchRouteCountBytes + kDispatchMaskLoadGuardBytes;
        const uint64_t routeBudget = fixedBytes < A5_MAIN_UB_SIZE ? A5_MAIN_UB_SIZE - fixedBytes : 0U;
        uint64_t expandedRouteItems = routeBudget * 8U / 33U;
        expandedRouteItems = expandedRouteItems / kDispatchRouteItemAlignment * kDispatchRouteItemAlignment;
        expandedRouteItems = std::min(expandedRouteItems, alignedTotalRouteItems);
        routeItemsPerBatch = std::max(routeItemsPerBatch, expandedRouteItems);
    }

    dispatch.routeItemsPerBatch = static_cast<uint32_t>(routeItemsPerBatch);
    dispatch.routeBatchCount = static_cast<uint32_t>((front.routeElems + routeItemsPerBatch - 1U) / routeItemsPerBatch);
    dispatch.bufferCount = static_cast<uint32_t>(bufferCount);
    dispatch.copyBufferBytes = static_cast<uint32_t>(copyBufferBytes);

    uint64_t ubOffset = 0U;
    dispatch.copyBufferUbOffset = static_cast<uint32_t>(ubOffset);
    ubOffset += bufferCount * copyBufferBytes;
    dispatch.metaBufferUbOffset = static_cast<uint32_t>(ubOffset);
    ubOffset += bufferCount * kDispatchMetaSlotBytes;
    dispatch.routeIndexUbOffset = static_cast<uint32_t>(ubOffset);
    ubOffset = AlignUp(ubOffset + routeItemsPerBatch * sizeof(uint32_t), UB_ALIGN);
    dispatch.routeCountUbOffset = static_cast<uint32_t>(ubOffset);
    ubOffset += kDispatchRouteCountBytes;
    dispatch.maskBufferUbOffset = static_cast<uint32_t>(ubOffset);
    const uint64_t maskBufferBytes = AlignUp((routeItemsPerBatch + 7U) / 8U, UB_ALIGN);
    ubOffset += maskBufferBytes + kDispatchMaskLoadGuardBytes;
    if (ubOffset > A5_MAIN_UB_SIZE) {
        throw std::runtime_error("dispatch token ring and route buffers exceed A5 main UB budget");
    }
}

uint64_t CeilDivU64(uint64_t value, uint64_t divisor)
{
    if (divisor == 0U) {
        throw std::runtime_error("CeilDivU64 requires a nonzero divisor");
    }
    return value / divisor + (value % divisor != 0U ? 1U : 0U);
}

void PopulateWaveSchedule(MegaMoeTilingData &tiling, const A5FixedScheduleConfig &schedule, const CaseConfig &cfg)
{
    MegaMoeFixedGroupTiling &fixed = tiling.fixedGroupTiling;
    MegaMoeWavePlannerInput plannerInput = {
        .inputRows = cfg.m,
        .topK = cfg.topk,
        .expertCount = cfg.expert_per_rank,
        .activeAicNum = schedule.physicalAicNum,
        .gmm1TileM = kMegaMoeGmmTileM,
        .gmm1TileN = kMegaMoeGmmTileN,
        .gmm1OutputN = cfg.n,
        .gmm2TileM = kMegaMoeGmmTileM,
        .gmm2TileN = kMegaMoeGmmTileN,
        .gmm2OutputN = cfg.k,
    };
    fixed.fullAicExpertsPerWave = CalcExpertsPerWave(plannerInput);
    plannerInput.activeAicNum = fixed.gmm1GroupSize;
    fixed.expertsPerWave = CalcExpertsPerWave(plannerInput);
    fixed.totalWaveCount = GetTotalWaveCount(
        cfg.expert_per_rank, fixed.fullAicExpertsPerWave, fixed.expertsPerWave, kMegaMoeFullAicGmm1WaveCount);
}

void ValidateFixedSchedule(const MegaMoeFixedGroupTiling &fixed, const CaseConfig &cfg)
{
    if (fixed.gmm1GroupSize + fixed.gmm2GroupSize != fixed.physicalAicNum) {
        throw std::runtime_error("fixed AIC group sizes must cover all physical AICs");
    }
    if (fixed.dispatchGroupSize == 0U || fixed.dispatchGroupSize > kMegaMoeFixedDispatchGroupSize ||
        fixed.dispatchGroupSize > fixed.physicalAicNum || fixed.gmm1GroupSize == 0U ||
        fixed.gmm1GroupSize > fixed.dispatchGroupSize || fixed.gmm1GroupSize > kMegaMoeFixedGmm1GroupSize ||
        fixed.gmm2GroupSize == 0U || fixed.gmm2GroupSize > kMegaMoeFixedGmm2GroupSize) {
        throw std::runtime_error("fixed Dispatch/GMM/SwiGLU groups exceed the A5 mixed-core capacities");
    }
    if (fixed.fullAicExpertsPerWave == 0U || fixed.fullAicExpertsPerWave > cfg.expert_per_rank ||
        fixed.expertsPerWave == 0U || fixed.expertsPerWave > cfg.expert_per_rank || fixed.totalWaveCount == 0U ||
        fixed.totalWaveCount !=
            GetTotalWaveCount(cfg.expert_per_rank, fixed.fullAicExpertsPerWave,
                                                      fixed.expertsPerWave, kMegaMoeFullAicGmm1WaveCount)) {
        throw std::runtime_error("invalid expert wave partition");
    }
}

uint64_t CheckedMul(uint64_t lhs, uint64_t rhs, const char *name)
{
    if (lhs != 0U && rhs > UINT64_MAX / lhs) {
        throw std::runtime_error(std::string(name) + " byte size overflows uint64");
    }
    return lhs * rhs;
}

uint64_t CheckedAdd(uint64_t lhs, uint64_t rhs, const char *name)
{
    if (rhs > UINT64_MAX - lhs) {
        throw std::runtime_error(std::string(name) + " end offset overflows uint64");
    }
    return lhs + rhs;
}

void RequireAlignedRange(const char *name, uint64_t offset, uint64_t bytes)
{
    if (offset % 512U != 0U || bytes % 512U != 0U) {
        throw std::runtime_error(std::string(name) + " must be 512-byte aligned");
    }
}

void PopulateMegaMoeInfo(MegaMoeInfo &info, const CaseConfig &cfg)
{
    info.M = cfg.m;
    info.K = cfg.k;
    info.N = cfg.n;
    info.topK = cfg.topk;
    info.expertPerRank = cfg.expert_per_rank;
}

void PopulateRuntimeInfo(MegaMoeRuntimeInfo &runtimeInfo, const StandaloneRankRuntime &runtime)
{
    runtimeInfo.remoteWindowContext = reinterpret_cast<uint64_t>(runtime.hccl.RemoteWindowContextPtr());
    runtimeInfo.rank = static_cast<uint32_t>(runtime.hccl.rank_id);
    runtimeInfo.rankSize = static_cast<uint32_t>(runtime.hccl.world_size);
}

uint64_t PopulateFrontTiling(MegaMoeFrontReorderTiling &front, const CaseConfig &cfg,
                             const StandaloneRankRuntime &runtime)
{
    constexpr uint64_t kPeerAlignBytes = 512U;
    constexpr uint64_t kPeerSignalBytes = MB_SIZE;
    constexpr uint32_t kMaskRouteItemsPerBatch = 2048U;

    const uint64_t expertNum = CheckedMul(cfg.world_size, cfg.expert_per_rank, "expert count");
    const uint64_t routeElems = CheckedMul(cfg.m, cfg.topk, "route count");
    if (expertNum > INT32_MAX || routeElems > INT32_MAX) {
        throw std::runtime_error("expert or route count exceeds int32 metadata capacity");
    }

    const uint64_t quantDataStorageBytes = MxQuantDataStorageBytes(cfg.k);
    const uint64_t quantScaleCols = MxQuantScaleCols(cfg.k);
    const uint64_t quantScaleStorageBytes = MxQuantScaleStorageBytes(cfg.k);
    const uint64_t packedRowStride = CheckedAdd(quantDataStorageBytes, quantScaleStorageBytes, "MX token record");
    if (quantDataStorageBytes > UINT32_MAX || quantScaleCols > UINT32_MAX || quantScaleStorageBytes > UINT32_MAX ||
        packedRowStride > UINT32_MAX) {
        throw std::runtime_error("MX token record fields exceed uint32");
    }

    front.expertNum = static_cast<uint32_t>(expertNum);
    front.routeElems = static_cast<uint32_t>(routeElems);
    front.quantDataStorageBytes = static_cast<uint32_t>(quantDataStorageBytes);
    front.quantScaleCols = static_cast<uint32_t>(quantScaleCols);
    front.packedRowStride = static_cast<uint32_t>(packedRowStride);
    front.maskBytes = static_cast<uint32_t>(AlignUp((routeElems + 7U) / 8U, 32U));
    const uint64_t maskBlockCount = front.maskBytes / kMegaMoeFrontMaskCountRecordBytes;
    const uint64_t maxAllocatedLanes = cfg.aiv_num >= expertNum ? (cfg.aiv_num + expertNum - 1U) / expertNum : 1U;
    const uint64_t maskLaneCapacity = std::min(maskBlockCount, maxAllocatedLanes);
    if (maskLaneCapacity == 0U || maskLaneCapacity > kMegaMoeFixedPhysicalAivNum) {
        throw std::runtime_error("front mask lane capacity exceeds the compiled A5 topology");
    }
    const uint64_t maskCountBytes =
        CheckedMul(maskLaneCapacity, kMegaMoeFrontMaskCountRecordBytes, "route mask lane counts");
    const uint64_t maskSlotBytes = CheckedAdd(front.maskBytes, maskCountBytes, "route mask slot");
    if (maskSlotBytes > UINT32_MAX) {
        throw std::runtime_error("route mask slot exceeds uint32");
    }
    front.maskSlotBytes = static_cast<uint32_t>(maskSlotBytes);
    front.maskRouteItemsPerBatch = kMaskRouteItemsPerBatch;

    const uint64_t sourceTokenRecordBytes = CheckedMul(cfg.m, front.packedRowStride, "source token records");
    front.routeMaskOffset = AlignUp(sourceTokenRecordBytes, kPeerAlignBytes);
    const uint64_t maskSlotCount = CheckedMul(cfg.expert_per_rank, cfg.world_size, "route mask slots");
    const uint64_t routeMaskBytes = CheckedMul(maskSlotCount, front.maskSlotBytes, "route mask slots");
    front.combineOutputOffset =
        AlignUp(CheckedAdd(front.routeMaskOffset, routeMaskBytes, "route mask slots"), kPeerAlignBytes);
    const uint64_t combineRowsAndCols = CheckedMul(routeElems, cfg.k, "combine route output");
    const uint64_t combineOutputBytes = CheckedMul(combineRowsAndCols, sizeof(uint16_t), "combine route output");
    front.preSumBeforeRankPeerOffset = AlignUp(
        CheckedAdd(front.combineOutputOffset, combineOutputBytes, "combine route output"), kPeerAlignBytes);
    const uint64_t preSumBeforeRankPeerBytes =
        AlignUp(CheckedMul(expertNum, sizeof(int32_t), "preSumBeforeRank peer rows"), kPeerAlignBytes);
    const uint64_t peerDataBytes =
        AlignUp(CheckedAdd(front.preSumBeforeRankPeerOffset, preSumBeforeRankPeerBytes, "preSumBeforeRank peer rows"),
                kPeerAlignBytes);

    const uint64_t windowBytes = runtime.hccl.WindowBytes();
    if (windowBytes < kPeerSignalBytes) {
        throw std::runtime_error("HCCL window is smaller than the reserved signal section");
    }
    const uint64_t peerSignalOffset = windowBytes - kPeerSignalBytes;
    if (peerDataBytes > peerSignalOffset) {
        throw std::runtime_error("HCCL window is too small for Mask Pull peer layout: windowBytes=" +
                                 std::to_string(windowBytes) + " dataBytes=" + std::to_string(peerDataBytes) +
                                 " signalOffset=" + std::to_string(peerSignalOffset));
    }
    RequireAlignedRange("Mask Pull peer data", 0U, peerDataBytes);

    uint64_t frontWorkspaceOffset = 0U;
    front.cumsumMMOffset = frontWorkspaceOffset;
    const uint64_t cumsumBytes = CheckedMul(maskSlotCount, sizeof(int32_t), "cumsumMM");
    frontWorkspaceOffset = AlignUp(CheckedAdd(frontWorkspaceOffset, cumsumBytes, "cumsumMM"), kPeerAlignBytes);

    front.expandedRowIdxOffset = frontWorkspaceOffset;
    const uint64_t expandedRowIdxBytes =
        AlignUp(CheckedMul(routeElems, sizeof(int32_t), "expandedRowIdx"), kPeerAlignBytes);
    frontWorkspaceOffset =
        AlignUp(CheckedAdd(frontWorkspaceOffset, expandedRowIdxBytes, "expandedRowIdx"), kPeerAlignBytes);

    const uint64_t alignedRouteElems = AlignUp(routeElems, 128U);
    const uint32_t minimumRunCount = CeilDivU32(front.routeElems, kFrontMetadataSortRunMaxElems);
    front.sortRunCount = Pow4Ceil(minimumRunCount);
    front.sortRunElems =
        static_cast<uint32_t>(AlignUp(CeilDivU32(front.routeElems, front.sortRunCount), kFrontMetadataSortAlignElems));
    front.sortOutLoopElems = kFrontMetadataSortOutLoopElems;
    if (front.sortRunElems == 0U || front.sortRunElems > kFrontMetadataSortRunMaxElems ||
        static_cast<uint64_t>(front.sortRunCount - 1U) * front.sortRunElems >= front.routeElems) {
        throw std::runtime_error("invalid front metadata VBS run split");
    }

    const uint64_t sortedArrayBytes =
        AlignUp(CheckedMul(alignedRouteElems, sizeof(int32_t), "front sorted metadata"), kPeerAlignBytes);
    front.sortedRouteSlotOffset = frontWorkspaceOffset;
    frontWorkspaceOffset =
        AlignUp(CheckedAdd(frontWorkspaceOffset, sortedArrayBytes, "front sorted route slots"), kPeerAlignBytes);

    front.sortWorkspaceBytes = AlignUp(
        CheckedMul(alignedRouteElems, 2U * sizeof(float), "front packed sort workspace"), kPeerAlignBytes);
    front.sortWorkspace0Offset = frontWorkspaceOffset;
    frontWorkspaceOffset = AlignUp(
        CheckedAdd(frontWorkspaceOffset, front.sortWorkspaceBytes, "front packed sort workspace 0"), kPeerAlignBytes);
    front.sortWorkspace1Offset = frontWorkspaceOffset;
    frontWorkspaceOffset = AlignUp(
        CheckedAdd(frontWorkspaceOffset, front.sortWorkspaceBytes, "front packed sort workspace 1"), kPeerAlignBytes);
    RequireAlignedRange("expandedRowIdx", front.expandedRowIdxOffset, expandedRowIdxBytes);
    RequireAlignedRange("sortedRouteSlot", front.sortedRouteSlotOffset, sortedArrayBytes);
    RequireAlignedRange("sortWorkspace0", front.sortWorkspace0Offset, front.sortWorkspaceBytes);
    RequireAlignedRange("sortWorkspace1", front.sortWorkspace1Offset, front.sortWorkspaceBytes);
    return frontWorkspaceOffset;
}

uint64_t AllocatePipelineWorkspace(MegaMoeTilingData &tiling, const CaseConfig &cfg, uint64_t frontWorkspaceBytes)
{
    auto &dispatch = tiling.dispatchTiling;
    auto &swiglu = tiling.swigluTiling;
    auto allocateSection = [](uint64_t &offset, uint64_t rawBytes, const char *name) {
        const uint64_t begin = AlignUp(offset, 512U);
        offset = AlignUp(CheckedAdd(begin, rawBytes, name), 512U);
        return begin;
    };

    const uint64_t rowCount = cfg.max_output_size;
    const uint64_t gmAElems = CheckedMul(rowCount, cfg.k, "gmA elements");
    const uint64_t gmAScaleElems = CheckedMul(rowCount, cfg.k / kMegaMoeMxGroupSize, "gmAScale elements");
    const uint64_t swigluCols = cfg.n / 2U;
    const uint64_t gmSwigluAElems = CheckedMul(rowCount, swigluCols, "gmSwigluA elements");
    const uint64_t gmSwigluScaleElems =
        CheckedMul(rowCount, swigluCols / kMegaMoeMxGroupSize, "gmSwigluScale elements");

    uint64_t workspaceOffset = frontWorkspaceBytes;
    dispatch.gmAOffset = allocateSection(workspaceOffset, gmAElems, "gmA");
    dispatch.gmAScaleOffset = allocateSection(workspaceOffset, gmAScaleElems, "gmAScale");
    swiglu.gmSwigluAOffset = allocateSection(workspaceOffset, gmSwigluAElems, "gmSwigluA");
    swiglu.gmSwigluScaleOffset = allocateSection(workspaceOffset, gmSwigluScaleElems, "gmSwigluScale");

    const uint64_t routeMetaRawBytes =
        CheckedMul(CheckedMul(rowCount, kMegaMoeRouteMetaFields, "routeMeta fields"), sizeof(int32_t), "routeMeta");
    dispatch.routeMetaOffset = allocateSection(workspaceOffset, routeMetaRawBytes, "routeMeta");
    const uint64_t routeMetaBytes = workspaceOffset - dispatch.routeMetaOffset;
    RequireAlignedRange("routeMeta", dispatch.routeMetaOffset, routeMetaBytes);

    const uint64_t maxTilesPerExpert = CeilDivU64(rowCount, kMegaMoeGmmTileM);
    if (maxTilesPerExpert == 0U || maxTilesPerExpert > UINT32_MAX) {
        throw std::runtime_error("Dispatch ready-count tile capacity exceeds uint32");
    }
    dispatch.readyCountExpertStrideBytes =
        CheckedMul(maxTilesPerExpert, kMegaMoeReadyCountSlotBytes, "Dispatch ready-count expert stride");
    const uint64_t readyCountRawBytes =
        CheckedMul(cfg.expert_per_rank, dispatch.readyCountExpertStrideBytes, "Dispatch ready-count workspace");
    dispatch.readyCountOffset = allocateSection(workspaceOffset, readyCountRawBytes, "Dispatch ready-count workspace");
    const uint64_t readyCountBytes = workspaceOffset - dispatch.readyCountOffset;
    RequireAlignedRange("Dispatch ready-count workspace", dispatch.readyCountOffset, readyCountBytes);
    return workspaceOffset;
}

void AllocateFixedGroupWorkspace(MegaMoeTilingData &tiling, const A5FixedScheduleConfig &schedule,
                                 const CaseConfig &cfg, uint64_t &workspaceOffset)
{
    MegaMoeFixedGroupTiling &fixed = tiling.fixedGroupTiling;
    fixed.physicalAicNum = schedule.physicalAicNum;
    fixed.dispatchGroupSize = schedule.dispatchGroupSize;
    fixed.gmm1GroupSize = schedule.gmm1GroupSize;
    fixed.gmm2GroupSize = schedule.gmm2GroupSize;
    PopulateWaveSchedule(tiling, schedule, cfg);
    ValidateFixedSchedule(fixed, cfg);

    fixed.syncOffset = AlignUp(workspaceOffset, 512U);
    workspaceOffset = fixed.syncOffset + kMegaMoeFixedSyncBytes;
    fixed.completionOffset = AlignUp(workspaceOffset, 512U);
    const uint64_t completionBytes = MegaMoeFixedCompletionBytes(schedule.physicalAicNum);
    workspaceOffset = CheckedAdd(fixed.completionOffset, completionBytes, "completion doorbell workspace");
    RequireAlignedRange("fixedGroupSync", fixed.syncOffset, kMegaMoeFixedSyncBytes);
    RequireAlignedRange("fixedGroupCompletion", fixed.completionOffset, completionBytes);
}

void PopulateUnpermuteTiling(MegaMoeTilingData &tiling, const CaseConfig &cfg)
{
    MegaMoeUnpermuteTiling &unpermute = tiling.unpermuteTiling;
    const uint32_t physicalAicCount = tiling.fixedGroupTiling.physicalAicNum;
    const uint32_t physicalAivCount = physicalAicCount * kMegaMoeFixedAivSubblocksPerPhysicalBlock;
    const uint32_t initialWorkerCount =
        std::min(physicalAicCount, kMegaMoeFixedInitialUnpermuteAiv0WorkerCount);
    const uint64_t tokensPerWorker = CeilDivU64(cfg.m, physicalAivCount);
    const uint64_t phase1TokensPerWorker = CeilDivU64(cfg.m, initialWorkerCount);
    if (cfg.m == 0U || cfg.topk == 0U || cfg.topk > 32U || cfg.expert_per_rank == 0U || cfg.world_size == 0U ||
        cfg.world_size > kMegaMoeExpertProgressMaxRanks ||
        tokensPerWorker > kMegaMoeRankStreamingMaxTokensPerWorker ||
        phase1TokensPerWorker > kMegaMoeRankStreamingMaxTokensPerWorker) {
        throw std::runtime_error("case exceeds the production rank-streaming unpermute limits");
    }
    unpermute.unpermuteTokenBatch = static_cast<uint32_t>(phase1TokensPerWorker);
    constexpr uint32_t accumulatorBufferCount = 2U;
    unpermute.unpermuteTileCols =
        ChooseUnpermuteTileCols(cfg.k, unpermute.unpermuteTokenBatch, cfg.topk, accumulatorBufferCount);
    RequireUnpermuteUbCapacity(unpermute.unpermuteTokenBatch, cfg.topk, unpermute.unpermuteTileCols,
                               accumulatorBufferCount);
}

} // namespace

const A5FixedScheduleConfig *FindA5DefaultSchedule(uint32_t effectiveAicNum)
{
    for (const A5FixedScheduleConfig &schedule : kCanonicalShapeSchedules) {
        if (IsDefaultSchedule(schedule) && schedule.physicalAicNum == effectiveAicNum) {
            return &schedule;
        }
    }
    return nullptr;
}

const A5FixedScheduleConfig *SelectA5FixedSchedule(const CaseConfig &cfg)
{
    const A5FixedScheduleConfig *defaultSchedule = FindA5DefaultSchedule(cfg.aic_num);
    if (defaultSchedule == nullptr || !IsCanonicalShapeFamily(cfg)) {
        return defaultSchedule;
    }
    for (const A5FixedScheduleConfig &schedule : kCanonicalShapeSchedules) {
        if (MatchesShape(schedule, cfg)) {
            return &schedule;
        }
    }
    return defaultSchedule;
}

MegaMoeBuildResult BuildMegaMoeTiling(const CaseConfig &cfg, const StandaloneRankRuntime &runtime)
{
    const A5FixedScheduleConfig *schedule = SelectA5FixedSchedule(cfg);
    if (schedule == nullptr) {
        throw std::runtime_error("unsupported A5 AICore count " + std::to_string(cfg.aic_num) +
                                 "; supported counts are 28, 32, and 36");
    }
    if (cfg.aiv_num != schedule->PhysicalAivNum()) {
        throw std::runtime_error("AIV count does not match the selected A5 AICore topology");
    }
    if (runtime.hccl.world_size <= 0 ||
        static_cast<uint32_t>(runtime.hccl.world_size) > schedule->dispatchGroupSize) {
        throw std::runtime_error("fixed-group dispatch requires a positive rank size no larger than its fixed group");
    }
    RequireMxShape(cfg.k, cfg.n);
    RequireDispatchPackedRowCapacity(MxPackedRowStride(cfg.k));
    RequireFrontQuantUbCapacity(cfg.k);
    RequireSwigluUbCapacity(cfg.n);
    RequireCombineUbCapacity(cfg.k);

    MegaMoeBuildResult result;
    result.block_dim = cfg.aic_num;
    PopulateMegaMoeInfo(result.tiling.megaMoeInfo, cfg);
    PopulateRuntimeInfo(result.tiling.runtimeInfo, runtime);
    const uint64_t frontWorkspaceBytes = PopulateFrontTiling(result.tiling.frontReorderTiling, cfg, runtime);
    PopulateDispatchBufferTiling(result.tiling.dispatchTiling, result.tiling.frontReorderTiling);
    result.workspace_bytes = AllocatePipelineWorkspace(result.tiling, cfg, frontWorkspaceBytes);
    AllocateFixedGroupWorkspace(result.tiling, *schedule, cfg, result.workspace_bytes);
    PopulateUnpermuteTiling(result.tiling, cfg);
    return result;
}
