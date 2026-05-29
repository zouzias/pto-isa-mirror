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

#ifndef PIPE_FIX
#define PIPE_FIX static_cast<pipe_t>(10)
#endif

#include <pto/comm/pto_comm_inst.hpp>
#include <pto/pto-inst.hpp>

#include "control_metadata.hpp"
#include "kernel_launchers.hpp"
#include "moe_dispatch_combine_a8w8_runtime_types.hpp"

using dispatch_combine_tile::DispatchCombineTileShape;
using dispatch_combine_tile::HcclDeviceContext;
using dispatch_combine_tile::PeerWindowLayout;
using dispatch_combine_tile::WorkspaceLayout;

#ifndef GM_ADDR
#define GM_ADDR __gm__ uint8_t *
#endif

// Kernel source contract for later tasks:
//   - Only PTO and C/C++ headers are allowed in this file.
//   - Device payload movement will use pto::GlobalTensor and pto::Tile views.
//   - Cross-rank movement/readiness will use PTO comm primitives.
//   - There are exactly two public device kernel names in this project.

namespace {

constexpr int kDefaultTileCols = 1024;
constexpr int kMetaTileCols = 16;
constexpr uint64_t kPingUbAddr = 0x0;
constexpr uint64_t kPongUbAddr = 0x1000;
constexpr uint64_t kSoftSyncUbAddr = 0x5000;

using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;

template <typename T>
using GlobalNd = pto::GlobalTensor<T, ShapeDyn, StrideDyn, pto::Layout::ND>;

template <typename T, int kCols = kDefaultTileCols>
using VecTile = pto::Tile<pto::TileType::Vec, T, 1, kCols, pto::BLayout::RowMajor, -1, -1>;

AICORE inline uint64_t Align64Device(uint64_t value)
{
    return ((value + 63) / 64) * 64;
}

AICORE inline uint64_t AppendFieldDevice(uint64_t &offset, uint64_t bytes)
{
    offset = Align64Device(offset);
    uint64_t fieldOffset = offset;
    offset += bytes;
    return fieldOffset;
}

struct LocalWorkspaceView {
    GM_ADDR base;
    __gm__ int32_t *localTokenPerExpert;
    __gm__ int32_t *blockTokenPerExpert;
    __gm__ int32_t *blockPrefixPerExpert;
    __gm__ int32_t *cumsumPerExpert;
    __gm__ int32_t *dispatchOffset;
    __gm__ int32_t *prevSumBeforeRank;
    __gm__ int32_t *localSync;
    __gm__ float *floatScratch;
    __gm__ half *dispatchedA;
    __gm__ half *ptrDLocal;
};

struct LocalPeerWindowView {
    GM_ADDR base;
    __gm__ int32_t *peerTokenPerExpert;
    __gm__ int32_t *expandedRowIdx;
    __gm__ half *packedA;
    __gm__ half *ptrD;
    __gm__ int32_t *countReadySignal;
    __gm__ int32_t *combineDoneSignal;
};

AICORE inline WorkspaceLayout MakeWorkspaceLayout(DispatchCombineTileShape shape)
{
    const uint64_t i32 = 4;
    const uint64_t f32 = 4;
    const uint64_t f16 = 2;
    uint64_t expertNumPadded =
        ((static_cast<uint64_t>(shape.expertNum) + shape.metadataPad - 1) / shape.metadataPad) * shape.metadataPad;
    uint64_t aivBlocks = shape.aivBlocks == 0 ? 1 : shape.aivBlocks;
    uint64_t expandedRows = static_cast<uint64_t>(shape.m) * shape.topK;
    uint64_t offset = 0;

    WorkspaceLayout layout{};
    layout.localTokenPerExpert = AppendFieldDevice(offset, expertNumPadded * i32);
    layout.blockTokenPerExpert = AppendFieldDevice(offset, aivBlocks * expertNumPadded * i32);
    layout.blockPrefixPerExpert = AppendFieldDevice(offset, aivBlocks * expertNumPadded * i32);
    layout.cumsumPerExpert = AppendFieldDevice(offset, static_cast<uint64_t>(shape.ep) * expertNumPadded * i32);
    layout.dispatchOffset = AppendFieldDevice(offset, static_cast<uint64_t>(shape.expertPerRank) * i32);
    layout.prevSumBeforeRank = AppendFieldDevice(offset, static_cast<uint64_t>(shape.ep) * shape.expertPerRank * i32);
    uint64_t syncSlots = aivBlocks * (8 + expertNumPadded);
    syncSlots = syncSlots < 64 ? 64 : syncSlots;
    layout.localSync = AppendFieldDevice(offset, syncSlots * i32);
    layout.floatScratch = AppendFieldDevice(offset, aivBlocks * shape.tileCols * f32);
    layout.dispatchedA = AppendFieldDevice(offset, static_cast<uint64_t>(shape.maxOutputSize) * shape.k * f16);
    layout.ptrDLocal = AppendFieldDevice(offset, expandedRows * shape.k * f16);
    layout.totalBytes = Align64Device(offset);
    return layout;
}

AICORE inline PeerWindowLayout MakePeerWindowLayout(DispatchCombineTileShape shape)
{
    const uint64_t i32 = 4;
    const uint64_t f16 = 2;
    uint64_t expertNumPadded =
        ((static_cast<uint64_t>(shape.expertNum) + shape.metadataPad - 1) / shape.metadataPad) * shape.metadataPad;
    uint64_t expandedRows = static_cast<uint64_t>(shape.m) * shape.topK;
    uint64_t offset = 0;

    PeerWindowLayout layout{};
    layout.peerTokenPerExpert = AppendFieldDevice(offset, static_cast<uint64_t>(shape.ep) * expertNumPadded * i32);
    layout.expandedRowIdx = AppendFieldDevice(offset, expandedRows * i32);
    layout.packedA = AppendFieldDevice(offset, expandedRows * shape.k * f16);
    layout.ptrD = AppendFieldDevice(offset, expandedRows * shape.k * f16);
    layout.countReadySignal = AppendFieldDevice(offset, static_cast<uint64_t>(shape.ep) * i32);
    layout.combineDoneSignal = AppendFieldDevice(offset, static_cast<uint64_t>(shape.ep) * i32);
    layout.totalBytes = Align64Device(offset);
    return layout;
}

AICORE inline LocalWorkspaceView MakeLocalWorkspaceView(GM_ADDR workspaceBase, const WorkspaceLayout &layout)
{
    LocalWorkspaceView view{};
    view.base = workspaceBase;
    view.localTokenPerExpert = reinterpret_cast<__gm__ int32_t *>(workspaceBase + layout.localTokenPerExpert);
    view.blockTokenPerExpert = reinterpret_cast<__gm__ int32_t *>(workspaceBase + layout.blockTokenPerExpert);
    view.blockPrefixPerExpert = reinterpret_cast<__gm__ int32_t *>(workspaceBase + layout.blockPrefixPerExpert);
    view.cumsumPerExpert = reinterpret_cast<__gm__ int32_t *>(workspaceBase + layout.cumsumPerExpert);
    view.dispatchOffset = reinterpret_cast<__gm__ int32_t *>(workspaceBase + layout.dispatchOffset);
    view.prevSumBeforeRank = reinterpret_cast<__gm__ int32_t *>(workspaceBase + layout.prevSumBeforeRank);
    view.localSync = reinterpret_cast<__gm__ int32_t *>(workspaceBase + layout.localSync);
    view.floatScratch = reinterpret_cast<__gm__ float *>(workspaceBase + layout.floatScratch);
    view.dispatchedA = reinterpret_cast<__gm__ half *>(workspaceBase + layout.dispatchedA);
    view.ptrDLocal = reinterpret_cast<__gm__ half *>(workspaceBase + layout.ptrDLocal);
    return view;
}

AICORE inline LocalPeerWindowView MakeLocalPeerWindowView(GM_ADDR peerWindowBase, const PeerWindowLayout &layout)
{
    LocalPeerWindowView view{};
    view.base = peerWindowBase;
    view.peerTokenPerExpert = reinterpret_cast<__gm__ int32_t *>(peerWindowBase + layout.peerTokenPerExpert);
    view.expandedRowIdx = reinterpret_cast<__gm__ int32_t *>(peerWindowBase + layout.expandedRowIdx);
    view.packedA = reinterpret_cast<__gm__ half *>(peerWindowBase + layout.packedA);
    view.ptrD = reinterpret_cast<__gm__ half *>(peerWindowBase + layout.ptrD);
    view.countReadySignal = reinterpret_cast<__gm__ int32_t *>(peerWindowBase + layout.countReadySignal);
    view.combineDoneSignal = reinterpret_cast<__gm__ int32_t *>(peerWindowBase + layout.combineDoneSignal);
    return view;
}

template <typename T>
AICORE inline __gm__ T *RemotePtr(__gm__ HcclDeviceContext *ctx, __gm__ T *localPtr, uint32_t peerRank)
{
    uint64_t localBase = ctx->windowsIn[ctx->rankId];
    uint64_t offset = reinterpret_cast<uint64_t>(localPtr) - localBase;
    return reinterpret_cast<__gm__ T *>(ctx->windowsIn[peerRank] + offset);
}

AICORE inline LocalPeerWindowView MakeRemotePeerWindowView(__gm__ HcclDeviceContext *ctx, GM_ADDR localPeerWindowBase,
                                                           uint32_t peerRank, const PeerWindowLayout &layout)
{
    GM_ADDR remoteBase = RemotePtr<uint8_t>(ctx, localPeerWindowBase, peerRank);
    return MakeLocalPeerWindowView(remoteBase, layout);
}

AICORE inline moe_dispatch_combine_a8w8::FieldLayout M2AppendFieldDevice(uint64_t &offset, uint64_t bytes)
{
    offset = Align64Device(offset);
    moe_dispatch_combine_a8w8::FieldLayout field{offset, bytes, 64};
    offset += bytes;
    return field;
}

AICORE inline uint64_t M2DTypeBytes(uint32_t dtype)
{
    if (dtype == 3U) {
        return 1;
    }
    if (dtype == 4U) {
        return 4;
    }
    return 2;
}

AICORE inline uint64_t M2ExpandedRows(moe_dispatch_combine_a8w8::ShapeConfig shape)
{
    return static_cast<uint64_t>(shape.m) * shape.topK;
}

AICORE inline uint64_t M2LocalRows(moe_dispatch_combine_a8w8::ShapeConfig shape)
{
    return static_cast<uint64_t>(shape.maxTokensPerExpert) * shape.expertPerRank;
}

AICORE inline uint64_t M2GlobalExpertNum(moe_dispatch_combine_a8w8::ShapeConfig shape)
{
    return static_cast<uint64_t>(shape.rankNum) * shape.expertPerRank;
}

AICORE inline uint64_t M2TokenPerExpertMatrixRowStride(moe_dispatch_combine_a8w8::ShapeConfig shape)
{
    constexpr uint64_t kI32PerCacheLine = 16;
    return ((M2GlobalExpertNum(shape) + kI32PerCacheLine - 1) / kI32PerCacheLine) * kI32PerCacheLine;
}

AICORE inline uint64_t M2DispatchPayloadRowBytes(moe_dispatch_combine_a8w8::ShapeConfig shape)
{
    return Align64Device(static_cast<uint64_t>(shape.hiddenSize) * M2DTypeBytes(3U));
}

AICORE inline uint64_t M2ReturnPayloadRowBytes(moe_dispatch_combine_a8w8::ShapeConfig shape)
{
    return Align64Device(static_cast<uint64_t>(shape.hiddenSize) * M2DTypeBytes(shape.dtypeOut));
}

AICORE inline moe_dispatch_combine_a8w8::WorkspaceLayout MakeM2WorkspaceLayoutDevice(
    moe_dispatch_combine_a8w8::ShapeConfig shape)
{
    moe_dispatch_combine_a8w8::WorkspaceLayout layout{};
    uint64_t offset = 0;
    uint64_t expandedRows = M2ExpandedRows(shape);
    uint64_t localRows = M2LocalRows(shape);
    uint64_t globalExpertNum = M2GlobalExpertNum(shape);
    uint64_t tokenMatrixRowStride = M2TokenPerExpertMatrixRowStride(shape);
    uint64_t matrixCount = static_cast<uint64_t>(shape.rankNum) * tokenMatrixRowStride;
    uint64_t rankExpertCount = static_cast<uint64_t>(shape.rankNum) * shape.expertPerRank;
    uint64_t dispatchRowBytes = M2DispatchPayloadRowBytes(shape);
    uint64_t returnRowBytes = M2ReturnPayloadRowBytes(shape);
    uint64_t w1Cols = static_cast<uint64_t>(shape.intermediateSize) * 2U;
    uint64_t syncGroupCap = static_cast<uint64_t>(shape.expertPerRank) + 1U;
    uint64_t scoreboardTasks = rankExpertCount;
    uint64_t subTileCap = static_cast<uint64_t>(shape.expertPerRank) * 8U;

    layout.tokenPerExpertMatrix = M2AppendFieldDevice(offset, matrixCount * sizeof(int32_t));
    layout.blockTokenPerExpert = M2AppendFieldDevice(offset, globalExpertNum * sizeof(int32_t));
    layout.blockPrefixPerExpert = M2AppendFieldDevice(offset, globalExpertNum * sizeof(int32_t));
    layout.expandedRowIdx = M2AppendFieldDevice(offset, expandedRows * sizeof(int32_t));
    layout.dispatchOffset = M2AppendFieldDevice(offset, expandedRows * sizeof(int32_t));
    layout.cumsumMM = M2AppendFieldDevice(offset, rankExpertCount * sizeof(int32_t));
    layout.preSumBeforeRank = M2AppendFieldDevice(offset, rankExpertCount * sizeof(int32_t));
    layout.expertTokenNums = M2AppendFieldDevice(offset, shape.expertPerRank * sizeof(int32_t));
    layout.tokenOwnerRankOffsets = M2AppendFieldDevice(offset, rankExpertCount * sizeof(int32_t));
    layout.dispatchedA = M2AppendFieldDevice(offset, expandedRows * returnRowBytes);
    layout.dispatchedScale = M2AppendFieldDevice(offset, expandedRows * sizeof(float));
    layout.gmm1InputInt8 = M2AppendFieldDevice(offset, localRows * dispatchRowBytes);
    layout.routingPerTokenScale = M2AppendFieldDevice(offset, localRows * sizeof(float));
    layout.gmm1WeightInt8 = M2AppendFieldDevice(offset, globalExpertNum * shape.hiddenSize * w1Cols);
    layout.scale1Uint64 = M2AppendFieldDevice(offset, w1Cols * sizeof(uint64_t));
    layout.gmm1AccInt32 = M2AppendFieldDevice(offset, localRows * w1Cols * sizeof(int32_t));
    layout.gmm1Out = M2AppendFieldDevice(offset, localRows * w1Cols * sizeof(float));
    layout.swigluOut = M2AppendFieldDevice(offset, localRows * shape.intermediateSize * sizeof(float));
    layout.gmm2InputInt8 = M2AppendFieldDevice(offset, localRows * Align64Device(shape.intermediateSize));
    layout.gmm2PerTokenScale = M2AppendFieldDevice(offset, localRows * sizeof(float));
    layout.gmm2WeightInt8 = M2AppendFieldDevice(offset, globalExpertNum * shape.intermediateSize * shape.hiddenSize);
    layout.scale2Uint64 = M2AppendFieldDevice(offset, shape.hiddenSize * sizeof(uint64_t));
    layout.gmm2AccInt32 = M2AppendFieldDevice(offset, localRows * shape.hiddenSize * sizeof(int32_t));
    layout.gmm2Out = M2AppendFieldDevice(offset, localRows * returnRowBytes);
    layout.readyCounters = M2AppendFieldDevice(offset, 16U * 64U);
    layout.dispatchGroupReady = M2AppendFieldDevice(offset, shape.expertPerRank * 64U);
    layout.gmm1SyncGroupReady = M2AppendFieldDevice(offset, syncGroupCap * 64U);
    layout.activationSyncGroupReady = M2AppendFieldDevice(offset, syncGroupCap * 64U);
    layout.gmm2GroupReady = M2AppendFieldDevice(offset, shape.expertPerRank * 64U);
    layout.stageStatus = M2AppendFieldDevice(offset, 16U * 64U);
    layout.swigluSyncGroups = M2AppendFieldDevice(offset, syncGroupCap * sizeof(int32_t));
    layout.dequantSum = M2AppendFieldDevice(offset, (syncGroupCap + 1U) * sizeof(int32_t));
    layout.scoreboardTaskMap = M2AppendFieldDevice(offset, scoreboardTasks * 4U * sizeof(int32_t));
    layout.producerStatus = M2AppendFieldDevice(offset, scoreboardTasks * 64U);
    layout.scoreboardMinStatus = M2AppendFieldDevice(offset, scoreboardTasks * 64U);
    layout.workerWaitCounters = M2AppendFieldDevice(offset, scoreboardTasks * 64U);
    layout.scoreboardTimeoutCounters = M2AppendFieldDevice(offset, scoreboardTasks * 64U);
    layout.subTileReturnPlan = M2AppendFieldDevice(offset, subTileCap * 6U * sizeof(int32_t));
    layout.subTileOwnerSegments = M2AppendFieldDevice(offset, subTileCap * 6U * sizeof(int32_t));
    layout.subTileReady = M2AppendFieldDevice(offset, subTileCap * 64U);
    layout.timelineScratch = M2AppendFieldDevice(offset, 64U * 4U * sizeof(uint64_t));
    layout.totalBytes = Align64Device(offset);
    return layout;
}

AICORE inline moe_dispatch_combine_a8w8::PeerWindowLayout MakeM2PeerWindowLayoutDevice(
    moe_dispatch_combine_a8w8::ShapeConfig shape)
{
    moe_dispatch_combine_a8w8::PeerWindowLayout layout{};
    uint64_t offset = 0;
    uint64_t expandedRows = M2ExpandedRows(shape);
    uint64_t tokenMatrixRowStride = M2TokenPerExpertMatrixRowStride(shape);
    uint64_t matrixCount = static_cast<uint64_t>(shape.rankNum) * tokenMatrixRowStride;
    layout.dispatchPayloadRowBytes = M2DispatchPayloadRowBytes(shape);
    layout.returnPayloadRowBytes = M2ReturnPayloadRowBytes(shape);
    layout.header = M2AppendFieldDevice(offset, moe_dispatch_combine_a8w8::kPeerWindowHeaderBytes);
    layout.tokenPerExpertMatrix = M2AppendFieldDevice(offset, matrixCount * sizeof(int32_t));
    layout.countReadySignal = M2AppendFieldDevice(offset, shape.rankNum * 64U);
    layout.dispatchPayload = M2AppendFieldDevice(offset, expandedRows * layout.dispatchPayloadRowBytes);
    layout.dispatchScale = M2AppendFieldDevice(offset, expandedRows * sizeof(float));
    layout.returnPayload = M2AppendFieldDevice(offset, expandedRows * layout.returnPayloadRowBytes);
    layout.combineDoneSignal = M2AppendFieldDevice(offset, shape.rankNum * 64U);
    layout.returnSegmentCounters =
        M2AppendFieldDevice(offset, static_cast<uint64_t>(shape.rankNum) * shape.expertPerRank * 64U);
    layout.debugCounters = M2AppendFieldDevice(offset, 64U * 64U);
    layout.timeline = M2AppendFieldDevice(offset, 64U * 4U * sizeof(uint64_t));
    layout.totalBytes = Align64Device(offset);
    return layout;
}

struct M2WorkspaceViewDevice {
    GM_ADDR base;
    __gm__ int32_t *tokenPerExpertMatrix;
    __gm__ int32_t *blockTokenPerExpert;
    __gm__ int32_t *blockPrefixPerExpert;
    __gm__ int32_t *expandedRowIdx;
    __gm__ int32_t *dispatchOffset;
    __gm__ int32_t *cumsumMM;
    __gm__ int32_t *preSumBeforeRank;
    __gm__ int32_t *expertTokenNums;
    __gm__ int32_t *tokenOwnerRankOffsets;
    __gm__ int8_t *gmm1InputInt8;
    __gm__ float *routingPerTokenScale;
    __gm__ uint64_t *scale1Uint64;
    __gm__ int32_t *gmm1AccInt32;
    __gm__ float *gmm1Out;
    __gm__ float *swigluOut;
    __gm__ int8_t *gmm2InputInt8;
    __gm__ float *gmm2PerTokenScale;
    __gm__ uint64_t *scale2Uint64;
    __gm__ int32_t *gmm2AccInt32;
    __gm__ half *gmm2Out;
    __gm__ int32_t *dispatchGroupReady;
    __gm__ int32_t *gmm1SyncGroupReady;
    __gm__ int32_t *activationSyncGroupReady;
    __gm__ int32_t *gmm2GroupReady;
    __gm__ int32_t *swigluSyncGroups;
    __gm__ int32_t *dequantSum;
    __gm__ int32_t *scoreboardTaskMap;
    __gm__ int32_t *producerStatus;
    __gm__ int32_t *scoreboardMinStatus;
    __gm__ int32_t *workerWaitCounters;
    __gm__ int32_t *scoreboardTimeoutCounters;
    __gm__ int32_t *subTileReturnPlan;
    __gm__ int32_t *subTileOwnerSegments;
    __gm__ int32_t *subTileReady;
};

struct M2PeerWindowViewDevice {
    GM_ADDR base;
    __gm__ int32_t *tokenPerExpertMatrix;
    __gm__ int32_t *countReadySignal;
    __gm__ int8_t *dispatchPayload;
    __gm__ float *dispatchScale;
    __gm__ half *returnPayload;
    __gm__ int32_t *combineDoneSignal;
    __gm__ int32_t *returnSegmentCounters;
    __gm__ int32_t *debugCounters;
};

AICORE inline M2WorkspaceViewDevice MakeM2WorkspaceViewDevice(GM_ADDR workspaceBase,
                                                              const moe_dispatch_combine_a8w8::WorkspaceLayout &layout)
{
    M2WorkspaceViewDevice view{};
    view.base = workspaceBase;
    view.tokenPerExpertMatrix = reinterpret_cast<__gm__ int32_t *>(workspaceBase + layout.tokenPerExpertMatrix.offset);
    view.blockTokenPerExpert = reinterpret_cast<__gm__ int32_t *>(workspaceBase + layout.blockTokenPerExpert.offset);
    view.blockPrefixPerExpert = reinterpret_cast<__gm__ int32_t *>(workspaceBase + layout.blockPrefixPerExpert.offset);
    view.expandedRowIdx = reinterpret_cast<__gm__ int32_t *>(workspaceBase + layout.expandedRowIdx.offset);
    view.dispatchOffset = reinterpret_cast<__gm__ int32_t *>(workspaceBase + layout.dispatchOffset.offset);
    view.cumsumMM = reinterpret_cast<__gm__ int32_t *>(workspaceBase + layout.cumsumMM.offset);
    view.preSumBeforeRank = reinterpret_cast<__gm__ int32_t *>(workspaceBase + layout.preSumBeforeRank.offset);
    view.expertTokenNums = reinterpret_cast<__gm__ int32_t *>(workspaceBase + layout.expertTokenNums.offset);
    view.tokenOwnerRankOffsets =
        reinterpret_cast<__gm__ int32_t *>(workspaceBase + layout.tokenOwnerRankOffsets.offset);
    view.gmm1InputInt8 = reinterpret_cast<__gm__ int8_t *>(workspaceBase + layout.gmm1InputInt8.offset);
    view.routingPerTokenScale = reinterpret_cast<__gm__ float *>(workspaceBase + layout.routingPerTokenScale.offset);
    view.scale1Uint64 = reinterpret_cast<__gm__ uint64_t *>(workspaceBase + layout.scale1Uint64.offset);
    view.gmm1AccInt32 = reinterpret_cast<__gm__ int32_t *>(workspaceBase + layout.gmm1AccInt32.offset);
    view.gmm1Out = reinterpret_cast<__gm__ float *>(workspaceBase + layout.gmm1Out.offset);
    view.swigluOut = reinterpret_cast<__gm__ float *>(workspaceBase + layout.swigluOut.offset);
    view.gmm2InputInt8 = reinterpret_cast<__gm__ int8_t *>(workspaceBase + layout.gmm2InputInt8.offset);
    view.gmm2PerTokenScale = reinterpret_cast<__gm__ float *>(workspaceBase + layout.gmm2PerTokenScale.offset);
    view.scale2Uint64 = reinterpret_cast<__gm__ uint64_t *>(workspaceBase + layout.scale2Uint64.offset);
    view.gmm2AccInt32 = reinterpret_cast<__gm__ int32_t *>(workspaceBase + layout.gmm2AccInt32.offset);
    view.gmm2Out = reinterpret_cast<__gm__ half *>(workspaceBase + layout.gmm2Out.offset);
    view.dispatchGroupReady = reinterpret_cast<__gm__ int32_t *>(workspaceBase + layout.dispatchGroupReady.offset);
    view.gmm1SyncGroupReady = reinterpret_cast<__gm__ int32_t *>(workspaceBase + layout.gmm1SyncGroupReady.offset);
    view.activationSyncGroupReady =
        reinterpret_cast<__gm__ int32_t *>(workspaceBase + layout.activationSyncGroupReady.offset);
    view.gmm2GroupReady = reinterpret_cast<__gm__ int32_t *>(workspaceBase + layout.gmm2GroupReady.offset);
    view.swigluSyncGroups = reinterpret_cast<__gm__ int32_t *>(workspaceBase + layout.swigluSyncGroups.offset);
    view.dequantSum = reinterpret_cast<__gm__ int32_t *>(workspaceBase + layout.dequantSum.offset);
    view.scoreboardTaskMap = reinterpret_cast<__gm__ int32_t *>(workspaceBase + layout.scoreboardTaskMap.offset);
    view.producerStatus = reinterpret_cast<__gm__ int32_t *>(workspaceBase + layout.producerStatus.offset);
    view.scoreboardMinStatus = reinterpret_cast<__gm__ int32_t *>(workspaceBase + layout.scoreboardMinStatus.offset);
    view.workerWaitCounters = reinterpret_cast<__gm__ int32_t *>(workspaceBase + layout.workerWaitCounters.offset);
    view.scoreboardTimeoutCounters =
        reinterpret_cast<__gm__ int32_t *>(workspaceBase + layout.scoreboardTimeoutCounters.offset);
    view.subTileReturnPlan = reinterpret_cast<__gm__ int32_t *>(workspaceBase + layout.subTileReturnPlan.offset);
    view.subTileOwnerSegments = reinterpret_cast<__gm__ int32_t *>(workspaceBase + layout.subTileOwnerSegments.offset);
    view.subTileReady = reinterpret_cast<__gm__ int32_t *>(workspaceBase + layout.subTileReady.offset);
    return view;
}

AICORE inline M2PeerWindowViewDevice MakeM2PeerWindowViewDevice(
    GM_ADDR peerWindowBase, const moe_dispatch_combine_a8w8::PeerWindowLayout &layout)
{
    M2PeerWindowViewDevice view{};
    view.base = peerWindowBase;
    view.tokenPerExpertMatrix = reinterpret_cast<__gm__ int32_t *>(peerWindowBase + layout.tokenPerExpertMatrix.offset);
    view.countReadySignal = reinterpret_cast<__gm__ int32_t *>(peerWindowBase + layout.countReadySignal.offset);
    view.dispatchPayload = reinterpret_cast<__gm__ int8_t *>(peerWindowBase + layout.dispatchPayload.offset);
    view.dispatchScale = reinterpret_cast<__gm__ float *>(peerWindowBase + layout.dispatchScale.offset);
    view.returnPayload = reinterpret_cast<__gm__ half *>(peerWindowBase + layout.returnPayload.offset);
    view.combineDoneSignal = reinterpret_cast<__gm__ int32_t *>(peerWindowBase + layout.combineDoneSignal.offset);
    view.returnSegmentCounters =
        reinterpret_cast<__gm__ int32_t *>(peerWindowBase + layout.returnSegmentCounters.offset);
    view.debugCounters = reinterpret_cast<__gm__ int32_t *>(peerWindowBase + layout.debugCounters.offset);
    return view;
}

AICORE inline M2PeerWindowViewDevice MakeM2RemotePeerWindowViewDevice(
    __gm__ HcclDeviceContext *ctx, GM_ADDR localPeerWindowBase, uint32_t peerRank,
    const moe_dispatch_combine_a8w8::PeerWindowLayout &layout)
{
    GM_ADDR remoteBase = RemotePtr<uint8_t>(ctx, localPeerWindowBase, peerRank);
    return MakeM2PeerWindowViewDevice(remoteBase, layout);
}

AICORE inline uint64_t M2TokenPerExpertIndex(moe_dispatch_combine_a8w8::ShapeConfig shape, uint32_t tokenOwnerRank,
                                             uint32_t expertOwnerRank, uint32_t localExpert)
{
    return static_cast<uint64_t>(tokenOwnerRank) * M2TokenPerExpertMatrixRowStride(shape) +
           static_cast<uint64_t>(expertOwnerRank) * shape.expertPerRank + localExpert;
}

AICORE inline int32_t M2LoadTokenPerExpert(moe_dispatch_combine_a8w8::ShapeConfig shape,
                                           M2PeerWindowViewDevice localPeer, uint32_t tokenOwnerRank,
                                           uint32_t expertOwnerRank, uint32_t localExpert)
{
    return *(localPeer.tokenPerExpertMatrix +
             M2TokenPerExpertIndex(shape, tokenOwnerRank, expertOwnerRank, localExpert));
}

AICORE inline uint32_t M2GlobalExpert(uint32_t expertOwnerRank, uint32_t localExpert,
                                      moe_dispatch_combine_a8w8::ShapeConfig shape)
{
    return expertOwnerRank * shape.expertPerRank + localExpert;
}

AICORE inline uint32_t M2ExpertOwnerRank(uint32_t globalExpert, moe_dispatch_combine_a8w8::ShapeConfig shape)
{
    return globalExpert / shape.expertPerRank;
}

AICORE inline uint32_t M2LocalExpert(uint32_t globalExpert, moe_dispatch_combine_a8w8::ShapeConfig shape)
{
    return globalExpert % shape.expertPerRank;
}

AICORE inline float M2Abs(float value)
{
    return value < 0.0f ? -value : value;
}

AICORE inline float M2Sigmoid(float value)
{
    float clipped = value;
    if (clipped > 16.0f) {
        clipped = 16.0f;
    }
    if (clipped < -16.0f) {
        clipped = -16.0f;
    }
    return 0.5f + clipped / (2.0f * (1.0f + M2Abs(clipped)));
}

AICORE inline int8_t M2QuantizeToInt8(float value, float scale)
{
    if (scale <= 0.0f) {
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

AICORE inline float M2DecodeUint64Scale(uint64_t value)
{
    union {
        uint32_t u;
        float f;
    } bits{static_cast<uint32_t>(value & 0xFFFFFFFFULL)};
    return bits.f;
}

template <typename T>
AICORE inline GlobalNd<T> MakeGlobal1D(__gm__ T *ptr, int32_t elems)
{
    ShapeDyn shape(1, 1, 1, 1, elems);
    StrideDyn stride(elems, elems, elems, elems, 1);
    return GlobalNd<T>(ptr, shape, stride);
}

template <typename T>
AICORE inline GlobalNd<T> MakeGlobal2D(__gm__ T *ptr, int32_t rows, int32_t cols, int32_t rowStride)
{
    ShapeDyn shape(1, 1, 1, rows, cols);
    StrideDyn stride(rowStride * rows, rowStride * rows, rowStride * rows, rowStride, 1);
    return GlobalNd<T>(ptr, shape, stride);
}

AICORE inline pto::comm::Signal MakeSignal(__gm__ int32_t *ptr)
{
    return pto::comm::Signal(ptr);
}

AICORE inline void WaitStoreTileReusable()
{
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
}

template <int kCols = kDefaultTileCols>
AICORE inline void CopyRowHalf(__gm__ half *dstBase, int32_t dstRowStride, int32_t dstRow, __gm__ half *srcBase,
                               int32_t srcRowStride, int32_t srcRow, int32_t rowLen)
{
    for (int32_t col = 0; col < rowLen; col += kCols) {
        int32_t cols = rowLen - col < kCols ? rowLen - col : kCols;
        VecTile<half, kCols> ping(1, cols);
        VecTile<half, kCols> pong(1, cols);
        TASSIGN(ping, kPingUbAddr);
        TASSIGN(pong, kPongUbAddr);
        VecTile<half, kCols> &tile = ((col / kCols) & 1) == 0 ? ping : pong;
        event_t event = ((col / kCols) & 1) == 0 ? EVENT_ID0 : EVENT_ID1;
        GlobalNd<half> src =
            MakeGlobal2D(srcBase + static_cast<int64_t>(srcRow) * srcRowStride + col, 1, cols, srcRowStride);
        GlobalNd<half> dst =
            MakeGlobal2D(dstBase + static_cast<int64_t>(dstRow) * dstRowStride + col, 1, cols, dstRowStride);
        TLOAD(tile, src);
        set_flag(PIPE_MTE2, PIPE_MTE3, event);
        wait_flag(PIPE_MTE2, PIPE_MTE3, event);
        TSTORE(dst, tile);
        WaitStoreTileReusable();
    }
}

template <typename T, int kCols = kDefaultTileCols>
AICORE inline void TGetRows(GlobalNd<T> &dst, GlobalNd<T> &remoteSrc)
{
    VecTile<T, kCols> ping(1, kCols);
    VecTile<T, kCols> pong(1, kCols);
    TASSIGN(ping, kPingUbAddr);
    TASSIGN(pong, kPongUbAddr);
    pto::comm::TGET(dst, remoteSrc, ping, pong);
}

template <typename T, int kCols = kDefaultTileCols>
AICORE inline void TPutRows(GlobalNd<T> &remoteDst, GlobalNd<T> &src)
{
    VecTile<T, kCols> ping(1, kCols);
    VecTile<T, kCols> pong(1, kCols);
    TASSIGN(ping, kPingUbAddr);
    TASSIGN(pong, kPongUbAddr);
    pto::comm::TPUT(remoteDst, src, ping, pong);
}

AICORE inline void NotifySignal(__gm__ int32_t *signal, int32_t value)
{
    pipe_barrier(PIPE_ALL);
    dsb(DSB_DDR);
    pto::comm::Signal sig = MakeSignal(signal);
    (void)value;
    pto::comm::TNOTIFY(sig, 1, pto::comm::NotifyOp::AtomicAdd);
}

AICORE inline void WaitSignal(__gm__ int32_t *signal, int32_t value)
{
    pto::comm::Signal sig = MakeSignal(signal);
    if (!pto::comm::TTEST(sig, value, pto::comm::WaitCmp::GE)) {
        pto::comm::TWAIT(sig, value, pto::comm::WaitCmp::GE);
    }
}

AICORE inline int32_t LoadScalarI32(__gm__ int32_t *ptr)
{
    return *ptr;
}

AICORE inline void StoreScalarI32(__gm__ int32_t *ptr, int32_t value)
{
    *ptr = value;
}

AICORE inline void InvalidateGmCacheLines(__gm__ void *ptr, uint32_t bytes)
{
    pipe_barrier(PIPE_ALL);
    uint64_t start = reinterpret_cast<uint64_t>(ptr) & ~static_cast<uint64_t>(63);
    uint64_t end = (reinterpret_cast<uint64_t>(ptr) + bytes + 63) & ~static_cast<uint64_t>(63);
    for (uint64_t addr = start; addr < end; addr += 64) {
        dcci(reinterpret_cast<__gm__ void *>(addr), SINGLE_CACHE_LINE);
    }
    dsb(DSB_DDR);
}

AICORE inline uint32_t ExpertNumPaddedDevice(DispatchCombineTileShape shape)
{
    return ((shape.expertNum + shape.metadataPad - 1) / shape.metadataPad) * shape.metadataPad;
}

AICORE inline uint32_t TokenShardBegin(uint32_t totalTokens, uint32_t blockId, uint32_t blockNum)
{
    uint32_t base = totalTokens / blockNum;
    uint32_t rem = totalTokens % blockNum;
    return blockId * base + (blockId < rem ? blockId : rem);
}

AICORE inline uint32_t TokenShardEnd(uint32_t totalTokens, uint32_t blockId, uint32_t blockNum)
{
    return TokenShardBegin(totalTokens, blockId + 1, blockNum);
}

AICORE inline uint32_t TokenShardBlockForToken(uint32_t totalTokens, uint32_t token, uint32_t blockNum)
{
    uint32_t base = totalTokens / blockNum;
    uint32_t rem = totalTokens % blockNum;
    uint32_t largeShardTokens = (base + 1) * rem;
    if (token < largeShardTokens) {
        return token / (base + 1);
    }
    if (base == 0) {
        return token;
    }
    return rem + (token - largeShardTokens) / base;
}

AICORE inline void SoftSyncAiv(__gm__ int32_t *gmWorkspace, uint32_t blockNum)
{
    pto::Tile<pto::TileType::Vec, int32_t, 1, pto::SYNCALL_SOFT_SLOT_INT32, pto::BLayout::RowMajor, -1, -1> syncTile(
        1, pto::SYNCALL_SOFT_SLOT_INT32);
#ifndef __PTO_AUTO__
    syncTile.data() = reinterpret_cast<__ubuf__ int32_t *>(kSoftSyncUbAddr);
#endif
    GlobalNd<int32_t> syncGlobal =
        MakeGlobal1D(gmWorkspace, static_cast<int32_t>(blockNum * pto::SYNCALL_SOFT_SLOT_INT32));
    pto::SYNCALL<pto::SyncAllMode::Soft>(syncGlobal, syncTile, static_cast<int32_t>(blockNum));
}

AICORE inline __gm__ int32_t *PackCursorBase(LocalWorkspaceView workspaceView, uint32_t blockNum)
{
    return workspaceView.localSync + blockNum * pto::SYNCALL_SOFT_SLOT_INT32;
}

AICORE inline void ClearDispatchState(DispatchCombineTileShape shape, LocalWorkspaceView workspaceView,
                                      LocalPeerWindowView localPeer, uint32_t blockId, uint32_t blockNum)
{
    uint32_t expertNumPadded = ExpertNumPaddedDevice(shape);
    uint32_t expandedRows = shape.m * shape.topK;
    for (uint32_t idx = blockId; idx < expertNumPadded; idx += blockNum) {
        StoreScalarI32(workspaceView.localTokenPerExpert + idx, 0);
    }
    for (uint32_t idx = blockId; idx < blockNum * expertNumPadded; idx += blockNum) {
        StoreScalarI32(workspaceView.blockTokenPerExpert + idx, 0);
        StoreScalarI32(workspaceView.blockPrefixPerExpert + idx, 0);
    }
    for (uint32_t idx = blockId; idx < shape.ep * expertNumPadded; idx += blockNum) {
        StoreScalarI32(workspaceView.cumsumPerExpert + idx, 0);
    }
    for (uint32_t idx = blockId; idx < shape.expertPerRank; idx += blockNum) {
        StoreScalarI32(workspaceView.dispatchOffset + idx, 0);
    }
    for (uint32_t idx = blockId; idx < shape.ep * shape.expertPerRank; idx += blockNum) {
        StoreScalarI32(workspaceView.prevSumBeforeRank + idx, 0);
    }
    (void)expandedRows;
    (void)localPeer;
}

AICORE inline void InitPackCursors(DispatchCombineTileShape shape, LocalWorkspaceView workspaceView, uint32_t blockId,
                                   uint32_t blockNum)
{
    uint32_t expertNumPadded = ExpertNumPaddedDevice(shape);
    InvalidateGmCacheLines(workspaceView.blockPrefixPerExpert,
                           static_cast<uint32_t>(blockNum * expertNumPadded * sizeof(int32_t)));
    __gm__ int32_t *cursorBase = PackCursorBase(workspaceView, blockNum) + blockId * expertNumPadded;
    for (uint32_t expert = 0; expert < expertNumPadded; ++expert) {
        int32_t prefix = LoadScalarI32(workspaceView.blockPrefixPerExpert +
                                       static_cast<uint64_t>(blockId) * expertNumPadded + expert);
        StoreScalarI32(cursorBase + expert, prefix);
    }
}

AICORE inline int32_t PackedExpertOffset(DispatchCombineTileShape shape, LocalWorkspaceView workspaceView,
                                         uint32_t myRank, uint32_t expert)
{
    uint32_t expertNumPadded = ExpertNumPaddedDevice(shape);
    return LoadScalarI32(workspaceView.cumsumPerExpert + static_cast<uint64_t>(myRank) * expertNumPadded + expert);
}

AICORE inline void PackLocalRowsToWindow(DispatchCombineTileShape shape, LocalWorkspaceView workspaceView,
                                         LocalPeerWindowView localPeer, GM_ADDR inputA, GM_ADDR expertIdx,
                                         uint32_t myRank, uint32_t blockId, uint32_t blockNum)
{
    uint32_t expertNumPadded = ExpertNumPaddedDevice(shape);
    __gm__ int32_t *cursorBase = PackCursorBase(workspaceView, blockNum) + blockId * expertNumPadded;
    __gm__ int32_t *expertIds = reinterpret_cast<__gm__ int32_t *>(expertIdx);
    __gm__ half *input = reinterpret_cast<__gm__ half *>(inputA);
    InvalidateGmCacheLines(workspaceView.cumsumPerExpert + static_cast<uint64_t>(myRank) * expertNumPadded,
                           static_cast<uint32_t>(expertNumPadded * sizeof(int32_t)));
    uint32_t tokenBegin = TokenShardBegin(shape.m, blockId, blockNum);
    uint32_t tokenEnd = TokenShardEnd(shape.m, blockId, blockNum);
    for (uint32_t token = tokenBegin; token < tokenEnd; ++token) {
        for (uint32_t slot = 0; slot < shape.topK; ++slot) {
            uint32_t routeIndex = token * shape.topK + slot;
            int32_t expert = LoadScalarI32(expertIds + routeIndex);
            if (expert < 0 || static_cast<uint32_t>(expert) >= shape.expertNum) {
                continue;
            }
            uint32_t expertId = static_cast<uint32_t>(expert);
            int32_t cursor = LoadScalarI32(cursorBase + expertId);
            StoreScalarI32(cursorBase + expertId, cursor + 1);
            int32_t packedRow = PackedExpertOffset(shape, workspaceView, myRank, expertId) + cursor;
            CopyRowHalf(localPeer.packedA, static_cast<int32_t>(shape.k), packedRow, input,
                        static_cast<int32_t>(shape.k), static_cast<int32_t>(token), static_cast<int32_t>(shape.k));
        }
    }
}

AICORE inline void CountLocalRoutes(DispatchCombineTileShape shape, LocalWorkspaceView workspaceView,
                                    LocalPeerWindowView localPeer, GM_ADDR expertIdx, uint32_t blockId,
                                    uint32_t blockNum)
{
    uint32_t expertNumPadded = ExpertNumPaddedDevice(shape);
    __gm__ int32_t *expertIds = reinterpret_cast<__gm__ int32_t *>(expertIdx);
    uint32_t tokenBegin = TokenShardBegin(shape.m, blockId, blockNum);
    uint32_t tokenEnd = TokenShardEnd(shape.m, blockId, blockNum);
    __gm__ int32_t *blockCounts = workspaceView.blockTokenPerExpert + blockId * expertNumPadded;
    for (uint32_t token = tokenBegin; token < tokenEnd; ++token) {
        for (uint32_t slot = 0; slot < shape.topK; ++slot) {
            uint32_t routeIndex = token * shape.topK + slot;
            int32_t expert = LoadScalarI32(expertIds + routeIndex);
            if (expert < 0 || static_cast<uint32_t>(expert) >= shape.expertNum) {
                continue;
            }
            __gm__ int32_t *count = blockCounts + static_cast<uint32_t>(expert);
            StoreScalarI32(count, LoadScalarI32(count) + 1);
        }
    }
    InvalidateGmCacheLines(blockCounts, static_cast<uint32_t>(expertNumPadded * sizeof(int32_t)));
}

AICORE inline void RebuildExpandedRowIdx(DispatchCombineTileShape shape, LocalWorkspaceView workspaceView,
                                         LocalPeerWindowView localPeer, GM_ADDR expertIdx, uint32_t myRank,
                                         uint32_t blockId, uint32_t blockNum)
{
    constexpr uint32_t kI32PerCacheLine = 16;
    uint32_t expertNumPadded = ExpertNumPaddedDevice(shape);
    uint32_t routeCount = shape.m * shape.topK;
    uint32_t lineCount = (routeCount + kI32PerCacheLine - 1) / kI32PerCacheLine;
    __gm__ int32_t *expertIds = reinterpret_cast<__gm__ int32_t *>(expertIdx);
    InvalidateGmCacheLines(workspaceView.blockPrefixPerExpert,
                           static_cast<uint32_t>(blockNum * expertNumPadded * sizeof(int32_t)));
    InvalidateGmCacheLines(workspaceView.cumsumPerExpert + static_cast<uint64_t>(myRank) * expertNumPadded,
                           static_cast<uint32_t>(expertNumPadded * sizeof(int32_t)));
    for (uint32_t line = blockId; line < lineCount; line += blockNum) {
        uint32_t begin = line * kI32PerCacheLine;
        uint32_t end = begin + kI32PerCacheLine;
        if (end > routeCount) {
            end = routeCount;
        }
        for (uint32_t routeIndex = begin; routeIndex < end; ++routeIndex) {
            int32_t expert = LoadScalarI32(expertIds + routeIndex);
            if (expert < 0 || static_cast<uint32_t>(expert) >= shape.expertNum) {
                StoreScalarI32(localPeer.expandedRowIdx + routeIndex, -1);
                continue;
            }
            uint32_t expertId = static_cast<uint32_t>(expert);
            uint32_t token = routeIndex / shape.topK;
            uint32_t sourceBlock = TokenShardBlockForToken(shape.m, token, blockNum);
            uint32_t localBegin = TokenShardBegin(shape.m, sourceBlock, blockNum) * shape.topK;
            int32_t localOrdinal = 0;
            for (uint32_t prev = localBegin; prev < routeIndex; ++prev) {
                int32_t prevExpert = LoadScalarI32(expertIds + prev);
                if (prevExpert == expert) {
                    ++localOrdinal;
                }
            }
            int32_t blockPrefix = LoadScalarI32(workspaceView.blockPrefixPerExpert +
                                                static_cast<uint64_t>(sourceBlock) * expertNumPadded + expertId);
            int32_t packedRow = PackedExpertOffset(shape, workspaceView, myRank, expertId) + blockPrefix + localOrdinal;
            StoreScalarI32(localPeer.expandedRowIdx + routeIndex, packedRow);
        }
        InvalidateGmCacheLines(localPeer.expandedRowIdx + begin,
                               static_cast<uint32_t>((end - begin) * sizeof(int32_t)));
    }
}

AICORE inline void BuildBlockPrefixAndLocalCounts(DispatchCombineTileShape shape, LocalWorkspaceView workspaceView,
                                                  uint32_t blockId, uint32_t blockNum)
{
    if (blockId != 0) {
        return;
    }
    uint32_t expertNumPadded = ExpertNumPaddedDevice(shape);
    InvalidateGmCacheLines(workspaceView.blockTokenPerExpert,
                           static_cast<uint32_t>(blockNum * expertNumPadded * sizeof(int32_t)));
    for (uint32_t expert = 0; expert < expertNumPadded; ++expert) {
        int32_t sum = 0;
        for (uint32_t block = 0; block < blockNum; ++block) {
            uint32_t idx = block * expertNumPadded + expert;
            StoreScalarI32(workspaceView.blockPrefixPerExpert + idx, sum);
            if (expert < shape.expertNum) {
                sum += LoadScalarI32(workspaceView.blockTokenPerExpert + idx);
            }
        }
        StoreScalarI32(workspaceView.localTokenPerExpert + expert, expert < shape.expertNum ? sum : 0);
    }
    InvalidateGmCacheLines(workspaceView.blockPrefixPerExpert,
                           static_cast<uint32_t>(blockNum * expertNumPadded * sizeof(int32_t)));
    InvalidateGmCacheLines(workspaceView.localTokenPerExpert, static_cast<uint32_t>(expertNumPadded * sizeof(int32_t)));
}

AICORE inline void BuildPackedExpertOffset(DispatchCombineTileShape shape, LocalWorkspaceView workspaceView,
                                           uint32_t myRank, uint32_t blockId)
{
    if (blockId != 0) {
        return;
    }
    uint32_t expertNumPadded = ExpertNumPaddedDevice(shape);
    InvalidateGmCacheLines(workspaceView.localTokenPerExpert, static_cast<uint32_t>(expertNumPadded * sizeof(int32_t)));
    int32_t sum = 0;
    for (uint32_t expert = 0; expert < expertNumPadded; ++expert) {
        int32_t count = LoadScalarI32(workspaceView.localTokenPerExpert + expert);
        StoreScalarI32(workspaceView.cumsumPerExpert + static_cast<uint64_t>(myRank) * expertNumPadded + expert, sum);
        if (expert < shape.expertNum) {
            sum += count;
        }
    }
    InvalidateGmCacheLines(workspaceView.cumsumPerExpert + static_cast<uint64_t>(myRank) * expertNumPadded,
                           static_cast<uint32_t>(expertNumPadded * sizeof(int32_t)));
}

AICORE inline void PublishCountRows(DispatchCombineTileShape shape, LocalWorkspaceView workspaceView,
                                    LocalPeerWindowView localPeer, __gm__ HcclDeviceContext *ctx, GM_ADDR peerWindow,
                                    uint32_t myRank, uint32_t blockId, uint32_t blockNum,
                                    const PeerWindowLayout &peerWindowLayout)
{
    uint32_t expertNumPadded = ExpertNumPaddedDevice(shape);
    for (uint32_t dst = blockId; dst < shape.ep; dst += blockNum) {
        LocalPeerWindowView remotePeer = MakeRemotePeerWindowView(ctx, peerWindow, dst, peerWindowLayout);
        InvalidateGmCacheLines(workspaceView.localTokenPerExpert,
                               static_cast<uint32_t>(expertNumPadded * sizeof(int32_t)));
        GlobalNd<int32_t> localCount =
            MakeGlobal2D(workspaceView.localTokenPerExpert, 1, static_cast<int32_t>(expertNumPadded),
                         static_cast<int32_t>(expertNumPadded));
        GlobalNd<int32_t> remoteCount =
            MakeGlobal2D(remotePeer.peerTokenPerExpert + static_cast<uint64_t>(myRank) * expertNumPadded, 1,
                         static_cast<int32_t>(expertNumPadded), static_cast<int32_t>(expertNumPadded));
        TPutRows<int32_t, kMetaTileCols>(remoteCount, localCount);
        NotifySignal(remotePeer.countReadySignal + myRank, static_cast<int32_t>(shape.signalValue));
    }
}

AICORE inline void WaitCountRows(DispatchCombineTileShape shape, LocalPeerWindowView localPeer, uint32_t blockId,
                                 uint32_t blockNum)
{
    for (uint32_t src = blockId; src < shape.ep; src += blockNum) {
        WaitSignal(localPeer.countReadySignal + src, static_cast<int32_t>(shape.signalValue));
    }
}

AICORE inline void BuildPrefixMetadata(DispatchCombineTileShape shape, LocalWorkspaceView workspaceView,
                                       LocalPeerWindowView localPeer, uint32_t myRank, uint32_t blockId,
                                       uint32_t blockNum)
{
    uint32_t expertNumPadded = ExpertNumPaddedDevice(shape);
    InvalidateGmCacheLines(localPeer.peerTokenPerExpert,
                           static_cast<uint32_t>(shape.ep * expertNumPadded * sizeof(int32_t)));
    for (uint32_t src = blockId; src < shape.ep; src += blockNum) {
        int32_t sum = 0;
        for (uint32_t expert = 0; expert < expertNumPadded; ++expert) {
            sum += LoadScalarI32(localPeer.peerTokenPerExpert + static_cast<uint64_t>(src) * expertNumPadded + expert);
            StoreScalarI32(workspaceView.cumsumPerExpert + static_cast<uint64_t>(src) * expertNumPadded + expert, sum);
        }
        InvalidateGmCacheLines(workspaceView.cumsumPerExpert + static_cast<uint64_t>(src) * expertNumPadded,
                               static_cast<uint32_t>(expertNumPadded * sizeof(int32_t)));
    }

    if (blockId == 0) {
        int32_t dispatchCursor = 0;
        for (uint32_t localExpert = 0; localExpert < shape.expertPerRank; ++localExpert) {
            uint32_t globalExpert = myRank * shape.expertPerRank + localExpert;
            StoreScalarI32(workspaceView.dispatchOffset + localExpert, dispatchCursor);
            int32_t beforeRank = 0;
            for (uint32_t src = 0; src < shape.ep; ++src) {
                StoreScalarI32(
                    workspaceView.prevSumBeforeRank + static_cast<uint64_t>(src) * shape.expertPerRank + localExpert,
                    beforeRank);
                int32_t rows = LoadScalarI32(localPeer.peerTokenPerExpert +
                                             static_cast<uint64_t>(src) * expertNumPadded + globalExpert);
                beforeRank += rows;
                dispatchCursor += rows;
            }
        }
        InvalidateGmCacheLines(workspaceView.dispatchOffset,
                               static_cast<uint32_t>(shape.expertPerRank * sizeof(int32_t)));
        InvalidateGmCacheLines(workspaceView.prevSumBeforeRank,
                               static_cast<uint32_t>(shape.ep * shape.expertPerRank * sizeof(int32_t)));
    }
}

AICORE inline void TGetRowsHalf(__gm__ half *dstBase, int32_t dstRowStride, int32_t dstRow, __gm__ half *remoteSrcBase,
                                int32_t srcRowStride, int32_t srcRow, int32_t rows, int32_t cols)
{
    if (rows <= 0 || cols <= 0) {
        return;
    }
    GlobalNd<half> dst = MakeGlobal2D(dstBase + static_cast<int64_t>(dstRow) * dstRowStride, rows, cols, dstRowStride);
    GlobalNd<half> src =
        MakeGlobal2D(remoteSrcBase + static_cast<int64_t>(srcRow) * srcRowStride, rows, cols, srcRowStride);
    TGetRows<half, kDefaultTileCols>(dst, src);
}

AICORE inline void TPutRowsHalf(__gm__ half *remoteDstBase, int32_t dstRowStride, int32_t dstRow, __gm__ half *srcBase,
                                int32_t srcRowStride, int32_t srcRow, int32_t rows, int32_t cols)
{
    if (rows <= 0 || cols <= 0) {
        return;
    }
    GlobalNd<half> dst =
        MakeGlobal2D(remoteDstBase + static_cast<int64_t>(dstRow) * dstRowStride, rows, cols, dstRowStride);
    GlobalNd<half> src = MakeGlobal2D(srcBase + static_cast<int64_t>(srcRow) * srcRowStride, rows, cols, srcRowStride);
    TPutRows<half, kDefaultTileCols>(dst, src);
}

AICORE inline void GatherLocalExpertPayload(DispatchCombineTileShape shape, LocalWorkspaceView workspaceView,
                                            LocalPeerWindowView localPeer, __gm__ HcclDeviceContext *ctx,
                                            GM_ADDR peerWindow, uint32_t myRank, uint32_t blockId, uint32_t blockNum,
                                            const PeerWindowLayout &peerWindowLayout)
{
    uint32_t expertNumPadded = ExpertNumPaddedDevice(shape);
    InvalidateGmCacheLines(localPeer.peerTokenPerExpert,
                           static_cast<uint32_t>(shape.ep * expertNumPadded * sizeof(int32_t)));
    InvalidateGmCacheLines(workspaceView.cumsumPerExpert,
                           static_cast<uint32_t>(shape.ep * expertNumPadded * sizeof(int32_t)));
    InvalidateGmCacheLines(workspaceView.dispatchOffset, static_cast<uint32_t>(shape.expertPerRank * sizeof(int32_t)));
    InvalidateGmCacheLines(workspaceView.prevSumBeforeRank,
                           static_cast<uint32_t>(shape.ep * shape.expertPerRank * sizeof(int32_t)));
    for (uint32_t localExpert = 0; localExpert < shape.expertPerRank; ++localExpert) {
        uint32_t globalExpert = myRank * shape.expertPerRank + localExpert;
        for (uint32_t src = blockId; src < shape.ep; src += blockNum) {
            int32_t rows = LoadScalarI32(localPeer.peerTokenPerExpert + static_cast<uint64_t>(src) * expertNumPadded +
                                         globalExpert);
            if (rows <= 0) {
                continue;
            }
            int32_t srcStart = globalExpert == 0 ?
                                   0 :
                                   LoadScalarI32(workspaceView.cumsumPerExpert +
                                                 static_cast<uint64_t>(src) * expertNumPadded + globalExpert - 1);
            int32_t dstStart = LoadScalarI32(workspaceView.dispatchOffset + localExpert) +
                               LoadScalarI32(workspaceView.prevSumBeforeRank +
                                             static_cast<uint64_t>(src) * shape.expertPerRank + localExpert);
            LocalPeerWindowView remotePeer = MakeRemotePeerWindowView(ctx, peerWindow, src, peerWindowLayout);
            TGetRowsHalf(workspaceView.dispatchedA, static_cast<int32_t>(shape.k), dstStart, remotePeer.packedA,
                         static_cast<int32_t>(shape.k), srcStart, rows, static_cast<int32_t>(shape.k));
        }
    }
}

AICORE inline void M2ClearDispatchState(moe_dispatch_combine_a8w8::ShapeConfig shape,
                                        M2WorkspaceViewDevice workspaceView, M2PeerWindowViewDevice localPeer)
{
    uint32_t globalExpertNum = shape.rankNum * shape.expertPerRank;
    uint32_t expandedRows = shape.m * shape.topK;
    uint32_t rankExpertCount = shape.rankNum * shape.expertPerRank;
    uint32_t tokenMatrixStorageCount = static_cast<uint32_t>(shape.rankNum * M2TokenPerExpertMatrixRowStride(shape));
    uint32_t rowBytes = static_cast<uint32_t>(M2DispatchPayloadRowBytes(shape));
    uint32_t localRows = static_cast<uint32_t>(M2LocalRows(shape));
    uint32_t subTileCap = shape.expertPerRank * 8U;
    for (uint32_t idx = 0; idx < globalExpertNum; ++idx) {
        StoreScalarI32(workspaceView.blockTokenPerExpert + idx, 0);
        StoreScalarI32(workspaceView.blockPrefixPerExpert + idx, 0);
        StoreScalarI32(workspaceView.tokenOwnerRankOffsets + idx, 0);
    }
    for (uint32_t idx = 0; idx < tokenMatrixStorageCount; ++idx) {
        StoreScalarI32(workspaceView.tokenPerExpertMatrix + idx, 0);
    }
    for (uint32_t idx = 0; idx < expandedRows; ++idx) {
        StoreScalarI32(workspaceView.expandedRowIdx + idx, -1);
    }
    for (uint32_t idx = 0; idx < rankExpertCount; ++idx) {
        StoreScalarI32(workspaceView.cumsumMM + idx, 0);
        StoreScalarI32(workspaceView.preSumBeforeRank + idx, 0);
        StoreScalarI32(workspaceView.scoreboardTaskMap + idx * 4U + 0U, 0);
        StoreScalarI32(workspaceView.scoreboardTaskMap + idx * 4U + 1U, 0);
        StoreScalarI32(workspaceView.scoreboardTaskMap + idx * 4U + 2U, 0);
        StoreScalarI32(workspaceView.scoreboardTaskMap + idx * 4U + 3U, 0);
        StoreScalarI32(workspaceView.producerStatus + idx * 16U, 0);
        StoreScalarI32(workspaceView.scoreboardMinStatus + idx * 16U, 0);
        StoreScalarI32(workspaceView.workerWaitCounters + idx * 16U, 0);
        StoreScalarI32(workspaceView.scoreboardTimeoutCounters + idx * 16U, 0);
    }
    for (uint32_t localExpert = 0; localExpert < shape.expertPerRank; ++localExpert) {
        StoreScalarI32(workspaceView.expertTokenNums + localExpert, 0);
        StoreScalarI32(workspaceView.dispatchGroupReady + localExpert * 16U, 0);
    }
    for (uint32_t row = 0; row < localRows; ++row) {
        workspaceView.routingPerTokenScale[row] = 1.0f;
        for (uint32_t col = 0; col < rowBytes; ++col) {
            workspaceView.gmm1InputInt8[static_cast<uint64_t>(row) * rowBytes + col] = 0;
        }
    }
    // Peer-visible count matrix and ready signals are cleared by the host before
    // the cross-rank launch. Clearing them inside the kernel races with peer
    // ranks that may already have published their count rows.
    for (uint32_t idx = 0; idx < subTileCap; ++idx) {
        StoreScalarI32(workspaceView.subTileReturnPlan + idx * 6U + 0U, 0);
        StoreScalarI32(workspaceView.subTileOwnerSegments + idx * 6U + 0U, 0);
        StoreScalarI32(workspaceView.subTileReady + idx * 16U, 0);
    }
}

AICORE inline void M2CountLocalRoutes(moe_dispatch_combine_a8w8::ShapeConfig shape, M2WorkspaceViewDevice workspaceView,
                                      M2PeerWindowViewDevice localPeer, GM_ADDR expertIdx, uint32_t myRank)
{
    __gm__ int32_t *expertIds = reinterpret_cast<__gm__ int32_t *>(expertIdx);
    uint32_t globalExpertNum = shape.rankNum * shape.expertPerRank;
    for (uint32_t routeIndex = 0; routeIndex < shape.m * shape.topK; ++routeIndex) {
        int32_t expert = LoadScalarI32(expertIds + routeIndex);
        if (expert < 0 || static_cast<uint32_t>(expert) >= globalExpertNum) {
            continue;
        }
        uint32_t globalExpert = static_cast<uint32_t>(expert);
        StoreScalarI32(workspaceView.blockTokenPerExpert + globalExpert,
                       LoadScalarI32(workspaceView.blockTokenPerExpert + globalExpert) + 1);
    }
    int32_t running = 0;
    for (uint32_t globalExpert = 0; globalExpert < globalExpertNum; ++globalExpert) {
        StoreScalarI32(workspaceView.blockPrefixPerExpert + globalExpert, running);
        uint32_t expertOwner = M2ExpertOwnerRank(globalExpert, shape);
        uint32_t localExpert = M2LocalExpert(globalExpert, shape);
        int32_t rows = LoadScalarI32(workspaceView.blockTokenPerExpert + globalExpert);
        StoreScalarI32(localPeer.tokenPerExpertMatrix + M2TokenPerExpertIndex(shape, myRank, expertOwner, localExpert),
                       rows);
        StoreScalarI32(
            workspaceView.tokenPerExpertMatrix + M2TokenPerExpertIndex(shape, myRank, expertOwner, localExpert), rows);
        running += rows;
    }
    InvalidateGmCacheLines(workspaceView.blockTokenPerExpert, static_cast<uint32_t>(globalExpertNum * sizeof(int32_t)));
    InvalidateGmCacheLines(workspaceView.blockPrefixPerExpert,
                           static_cast<uint32_t>(globalExpertNum * sizeof(int32_t)));
    InvalidateGmCacheLines(
        workspaceView.tokenPerExpertMatrix,
        static_cast<uint32_t>(shape.rankNum * M2TokenPerExpertMatrixRowStride(shape) * sizeof(int32_t)));
    InvalidateGmCacheLines(
        localPeer.tokenPerExpertMatrix,
        static_cast<uint32_t>(shape.rankNum * M2TokenPerExpertMatrixRowStride(shape) * sizeof(int32_t)));
}

AICORE inline void M2QuantizeRowToPeerPayload(moe_dispatch_combine_a8w8::ShapeConfig shape,
                                              M2PeerWindowViewDevice localPeer, GM_ADDR inputA, uint32_t token,
                                              uint32_t packedRow, uint32_t rowBytes)
{
    __gm__ half *input = reinterpret_cast<__gm__ half *>(inputA);
    float maxAbs = 0.0f;
    for (uint32_t col = 0; col < shape.hiddenSize; ++col) {
        float value = static_cast<float>(input[static_cast<uint64_t>(token) * shape.hiddenSize + col]);
        float absValue = M2Abs(value);
        if (absValue > maxAbs) {
            maxAbs = absValue;
        }
    }
    float scale = maxAbs == 0.0f ? 1.0f : maxAbs / 127.0f;
    localPeer.dispatchScale[packedRow] = scale;
    __gm__ int8_t *dst = localPeer.dispatchPayload + static_cast<uint64_t>(packedRow) * rowBytes;
    for (uint32_t col = 0; col < shape.hiddenSize; ++col) {
        float value = static_cast<float>(input[static_cast<uint64_t>(token) * shape.hiddenSize + col]);
        dst[col] = M2QuantizeToInt8(value, scale);
    }
    for (uint32_t col = shape.hiddenSize; col < rowBytes; ++col) {
        dst[col] = 0;
    }
}

AICORE inline void M2RoutePackQuantLocal(moe_dispatch_combine_a8w8::ShapeConfig shape,
                                         M2WorkspaceViewDevice workspaceView, M2PeerWindowViewDevice localPeer,
                                         GM_ADDR inputA, GM_ADDR expertIdx, uint32_t rowBytes)
{
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
            M2QuantizeRowToPeerPayload(shape, localPeer, inputA, token, static_cast<uint32_t>(packedRow), rowBytes);
        }
    }
    InvalidateGmCacheLines(workspaceView.expandedRowIdx, static_cast<uint32_t>(shape.m * shape.topK * sizeof(int32_t)));
    InvalidateGmCacheLines(localPeer.dispatchPayload, static_cast<uint32_t>(M2ExpandedRows(shape) * rowBytes));
    InvalidateGmCacheLines(localPeer.dispatchScale, static_cast<uint32_t>(M2ExpandedRows(shape) * sizeof(float)));
}

AICORE inline void M2PublishCountRows(moe_dispatch_combine_a8w8::ShapeConfig shape, M2WorkspaceViewDevice workspaceView,
                                      M2PeerWindowViewDevice localPeer, __gm__ HcclDeviceContext *ctx,
                                      GM_ADDR peerWindow, uint32_t myRank,
                                      const moe_dispatch_combine_a8w8::PeerWindowLayout &peerWindowLayout)
{
    uint32_t rowElems = static_cast<uint32_t>(M2TokenPerExpertMatrixRowStride(shape));
    __gm__ int32_t *localRow =
        workspaceView.tokenPerExpertMatrix + static_cast<uint64_t>(myRank) * M2TokenPerExpertMatrixRowStride(shape);
    InvalidateGmCacheLines(localRow, rowElems * sizeof(int32_t));
    for (uint32_t dst = 0; dst < shape.rankNum; ++dst) {
        M2PeerWindowViewDevice remotePeer = MakeM2RemotePeerWindowViewDevice(ctx, peerWindow, dst, peerWindowLayout);
        __gm__ int32_t *remoteRow =
            remotePeer.tokenPerExpertMatrix + static_cast<uint64_t>(myRank) * M2TokenPerExpertMatrixRowStride(shape);
        GlobalNd<int32_t> localCount =
            MakeGlobal2D(localRow, 1, static_cast<int32_t>(rowElems), static_cast<int32_t>(rowElems));
        GlobalNd<int32_t> remoteCount =
            MakeGlobal2D(remoteRow, 1, static_cast<int32_t>(rowElems), static_cast<int32_t>(rowElems));
        TPutRows<int32_t, kMetaTileCols>(remoteCount, localCount);
        NotifySignal(remotePeer.countReadySignal + myRank * 16U, 1);
    }
}

AICORE inline void M2WaitCountRows(moe_dispatch_combine_a8w8::ShapeConfig shape, M2PeerWindowViewDevice localPeer)
{
    for (uint32_t src = 0; src < shape.rankNum; ++src) {
        WaitSignal(localPeer.countReadySignal + src * 16U, 1);
    }
}

AICORE inline void M2BuildPrefixMetadata(moe_dispatch_combine_a8w8::ShapeConfig shape,
                                         M2WorkspaceViewDevice workspaceView, M2PeerWindowViewDevice localPeer,
                                         uint32_t myRank)
{
    int32_t dispatchCursor = 0;
    for (uint32_t localExpert = 0; localExpert < shape.expertPerRank; ++localExpert) {
        StoreScalarI32(workspaceView.dispatchOffset + localExpert, dispatchCursor);
        int32_t before = 0;
        for (uint32_t tokenOwner = 0; tokenOwner < shape.rankNum; ++tokenOwner) {
            int32_t rows = M2LoadTokenPerExpert(shape, localPeer, tokenOwner, myRank, localExpert);
            StoreScalarI32(workspaceView.preSumBeforeRank + tokenOwner * shape.expertPerRank + localExpert, before);
            before += rows;
            StoreScalarI32(workspaceView.cumsumMM + tokenOwner * shape.expertPerRank + localExpert, before);
        }
        StoreScalarI32(workspaceView.expertTokenNums + localExpert, before);
        dispatchCursor += before;
    }
    InvalidateGmCacheLines(workspaceView.dispatchOffset, static_cast<uint32_t>(shape.expertPerRank * sizeof(int32_t)));
    InvalidateGmCacheLines(workspaceView.preSumBeforeRank,
                           static_cast<uint32_t>(shape.rankNum * shape.expertPerRank * sizeof(int32_t)));
    InvalidateGmCacheLines(workspaceView.cumsumMM,
                           static_cast<uint32_t>(shape.rankNum * shape.expertPerRank * sizeof(int32_t)));
    InvalidateGmCacheLines(workspaceView.expertTokenNums, static_cast<uint32_t>(shape.expertPerRank * sizeof(int32_t)));
}

AICORE inline int32_t M2SourceGlobalExpertRowBase(moe_dispatch_combine_a8w8::ShapeConfig shape,
                                                  M2PeerWindowViewDevice localPeer, uint32_t tokenOwnerRank,
                                                  uint32_t globalExpert)
{
    int32_t base = 0;
    for (uint32_t expert = 0; expert < globalExpert; ++expert) {
        base += M2LoadTokenPerExpert(shape, localPeer, tokenOwnerRank, M2ExpertOwnerRank(expert, shape),
                                     M2LocalExpert(expert, shape));
    }
    return base;
}

AICORE inline void M2TGetRowsInt8(__gm__ int8_t *dstBase, int32_t dstRowStride, int32_t dstRow,
                                  __gm__ int8_t *remoteSrcBase, int32_t srcRowStride, int32_t srcRow, int32_t rows,
                                  int32_t cols)
{
    if (rows <= 0 || cols <= 0) {
        return;
    }
    GlobalNd<int8_t> dst =
        MakeGlobal2D(dstBase + static_cast<int64_t>(dstRow) * dstRowStride, rows, cols, dstRowStride);
    GlobalNd<int8_t> src =
        MakeGlobal2D(remoteSrcBase + static_cast<int64_t>(srcRow) * srcRowStride, rows, cols, srcRowStride);
    TGetRows<int8_t, kDefaultTileCols>(dst, src);
}

AICORE inline void M2TGetRowsFloat(__gm__ float *dstBase, int32_t dstRow, __gm__ float *remoteSrcBase, int32_t srcRow,
                                   int32_t rows)
{
    if (rows <= 0) {
        return;
    }
    GlobalNd<float> dst = MakeGlobal2D(dstBase + dstRow, rows, 1, 1);
    GlobalNd<float> src = MakeGlobal2D(remoteSrcBase + srcRow, rows, 1, 1);
    TGetRows<float, kMetaTileCols>(dst, src);
}

AICORE inline void M2RecordDispatchLedger(moe_dispatch_combine_a8w8::ShapeConfig shape,
                                          M2WorkspaceViewDevice workspaceView, uint32_t tokenOwner,
                                          uint32_t localExpert, int32_t dstStart, int32_t rows)
{
    uint32_t taskId = tokenOwner * shape.expertPerRank + localExpert;
    StoreScalarI32(workspaceView.scoreboardTaskMap + taskId * 4U + 0U, static_cast<int32_t>(tokenOwner));
    StoreScalarI32(workspaceView.scoreboardTaskMap + taskId * 4U + 1U, static_cast<int32_t>(localExpert));
    StoreScalarI32(workspaceView.scoreboardTaskMap + taskId * 4U + 2U, dstStart);
    StoreScalarI32(workspaceView.scoreboardTaskMap + taskId * 4U + 3U, rows);
    StoreScalarI32(workspaceView.producerStatus + taskId * 16U, rows > 0 ? 1 : 0);
    StoreScalarI32(workspaceView.scoreboardMinStatus + taskId * 16U, rows > 0 ? 1 : 0);
}

AICORE inline void M2GatherDispatchToGmm1Input(moe_dispatch_combine_a8w8::ShapeConfig shape,
                                               M2WorkspaceViewDevice workspaceView, M2PeerWindowViewDevice localPeer,
                                               __gm__ HcclDeviceContext *ctx, GM_ADDR peerWindow, uint32_t myRank,
                                               uint32_t rowBytes,
                                               const moe_dispatch_combine_a8w8::PeerWindowLayout &peerWindowLayout)
{
    for (uint32_t localExpert = 0; localExpert < shape.expertPerRank; ++localExpert) {
        uint32_t globalExpert = M2GlobalExpert(myRank, localExpert, shape);
        for (uint32_t tokenOwner = 0; tokenOwner < shape.rankNum; ++tokenOwner) {
            int32_t rows = M2LoadTokenPerExpert(shape, localPeer, tokenOwner, myRank, localExpert);
            int32_t dstStart =
                LoadScalarI32(workspaceView.dispatchOffset + localExpert) +
                LoadScalarI32(workspaceView.preSumBeforeRank + tokenOwner * shape.expertPerRank + localExpert);
            M2RecordDispatchLedger(shape, workspaceView, tokenOwner, localExpert, dstStart, rows);
            if (rows <= 0) {
                continue;
            }
            int32_t srcStart = M2SourceGlobalExpertRowBase(shape, localPeer, tokenOwner, globalExpert);
            M2PeerWindowViewDevice remotePeer =
                MakeM2RemotePeerWindowViewDevice(ctx, peerWindow, tokenOwner, peerWindowLayout);
            M2TGetRowsInt8(workspaceView.gmm1InputInt8, static_cast<int32_t>(rowBytes), dstStart,
                           remotePeer.dispatchPayload, static_cast<int32_t>(rowBytes), srcStart, rows,
                           static_cast<int32_t>(rowBytes));
            M2TGetRowsFloat(workspaceView.routingPerTokenScale, dstStart, remotePeer.dispatchScale, srcStart, rows);
        }
        StoreScalarI32(workspaceView.dispatchGroupReady + localExpert * 16U, 1);
    }
    InvalidateGmCacheLines(workspaceView.gmm1InputInt8, static_cast<uint32_t>(M2LocalRows(shape) * rowBytes));
    InvalidateGmCacheLines(workspaceView.routingPerTokenScale,
                           static_cast<uint32_t>(M2LocalRows(shape) * sizeof(float)));
}

AICORE inline void M2RunGmm1Epilogue(moe_dispatch_combine_a8w8::ShapeConfig shape, M2WorkspaceViewDevice workspaceView,
                                     uint32_t myRank)
{
    uint32_t w1Cols = shape.intermediateSize * 2U;
    for (uint32_t localExpert = 0; localExpert < shape.expertPerRank; ++localExpert) {
        int32_t rowBegin = LoadScalarI32(workspaceView.dispatchOffset + localExpert);
        int32_t rowCount = LoadScalarI32(workspaceView.expertTokenNums + localExpert);
        if (rowCount <= 0) {
            continue;
        }
        for (uint32_t row = 0; row < static_cast<uint32_t>(rowCount); ++row) {
            uint32_t globalRow = static_cast<uint32_t>(rowBegin) + row;
            for (uint32_t col = 0; col < w1Cols; ++col) {
                uint64_t index = static_cast<uint64_t>(globalRow) * w1Cols + col;
                __gm__ uint32_t *scaleBits = reinterpret_cast<__gm__ uint32_t *>(workspaceView.scale1Uint64 + col);
                union {
                    uint32_t u;
                    float f;
                } bits{*scaleBits};
                workspaceView.gmm1Out[index] = static_cast<float>(workspaceView.gmm1AccInt32[index]) * bits.f;
            }
        }
    }
    StoreScalarI32(workspaceView.gmm1SyncGroupReady, 1);
    InvalidateGmCacheLines(workspaceView.gmm1Out, static_cast<uint32_t>(M2LocalRows(shape) * w1Cols * sizeof(float)));
    (void)myRank;
}

AICORE inline uint32_t M2NextSwigluGroupSize(uint32_t remainingExperts)
{
    if (remainingExperts <= 1U) {
        return 1U;
    }
    return remainingExperts / 2U;
}

AICORE inline void M2BuildSwigluSyncMetadata(moe_dispatch_combine_a8w8::ShapeConfig shape,
                                             M2WorkspaceViewDevice workspaceView)
{
    uint32_t groupCount = 0;
    uint32_t expert = 0;
    int32_t rowPrefix = 0;
    StoreScalarI32(workspaceView.dequantSum, 0);
    while (expert < shape.expertPerRank) {
        uint32_t remaining = shape.expertPerRank - expert;
        uint32_t groupSize = M2NextSwigluGroupSize(remaining);
        StoreScalarI32(workspaceView.swigluSyncGroups + groupCount + 1U, static_cast<int32_t>(groupSize));
        for (uint32_t idx = 0; idx < groupSize; ++idx) {
            rowPrefix += LoadScalarI32(workspaceView.expertTokenNums + expert + idx);
        }
        StoreScalarI32(workspaceView.dequantSum + groupCount + 1U, rowPrefix);
        ++groupCount;
        expert += groupSize;
    }
    StoreScalarI32(workspaceView.swigluSyncGroups, static_cast<int32_t>(groupCount));
}

AICORE inline void M2ComputeSwigluRowPto(moe_dispatch_combine_a8w8::ShapeConfig shape,
                                         M2WorkspaceViewDevice workspaceView, uint32_t globalRow, float routingScale)
{
    constexpr int kSwiGluCols = 32;
    uint32_t w1Cols = shape.intermediateSize * 2U;
    uint32_t cols = shape.intermediateSize;
    VecTile<float, kSwiGluCols> gateTile(1, cols);
    VecTile<float, kSwiGluCols> upTile(1, cols);
    VecTile<float, kSwiGluCols> negGateTile(1, cols);
    VecTile<float, kSwiGluCols> expTile(1, cols);
    VecTile<float, kSwiGluCols> denomTile(1, cols);
    VecTile<float, kSwiGluCols> sigmoidTile(1, cols);
    VecTile<float, kSwiGluCols> scaledTile(1, cols);
    TASSIGN(gateTile, 0x0);
    TASSIGN(upTile, 0x400);
    TASSIGN(negGateTile, 0x800);
    TASSIGN(expTile, 0xC00);
    TASSIGN(denomTile, 0x1000);
    TASSIGN(sigmoidTile, 0x1400);
    TASSIGN(scaledTile, 0x1800);

    GlobalNd<float> gateGlobal = MakeGlobal2D(workspaceView.gmm1Out + static_cast<uint64_t>(globalRow) * w1Cols, 1,
                                              static_cast<int32_t>(cols), static_cast<int32_t>(w1Cols));
    GlobalNd<float> upGlobal =
        MakeGlobal2D(workspaceView.gmm1Out + static_cast<uint64_t>(globalRow) * w1Cols + shape.intermediateSize, 1,
                     static_cast<int32_t>(cols), static_cast<int32_t>(w1Cols));
    GlobalNd<float> dstGlobal =
        MakeGlobal2D(workspaceView.swigluOut + static_cast<uint64_t>(globalRow) * shape.intermediateSize, 1,
                     static_cast<int32_t>(cols), static_cast<int32_t>(shape.intermediateSize));
    TLOAD(gateTile, gateGlobal);
    TLOAD(upTile, upGlobal);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TMULS(gateTile, gateTile, routingScale);
    pipe_barrier(PIPE_V);
    TMULS(upTile, upTile, routingScale);
    pipe_barrier(PIPE_V);
    TMULS(negGateTile, gateTile, -1.0f);
    pipe_barrier(PIPE_V);
    TEXP(expTile, negGateTile);
    pipe_barrier(PIPE_V);
    TADDS(denomTile, expTile, 1.0f);
    pipe_barrier(PIPE_V);
    TDIVS(sigmoidTile, 1.0f, denomTile);
    pipe_barrier(PIPE_V);
    TMUL(scaledTile, sigmoidTile, upTile);
    pipe_barrier(PIPE_V);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(dstGlobal, scaledTile);
    WaitStoreTileReusable();
}

AICORE inline void M2RunActivationQuant(moe_dispatch_combine_a8w8::ShapeConfig shape,
                                        M2WorkspaceViewDevice workspaceView, uint32_t myRank)
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
            StoreScalarI32(workspaceView.activationSyncGroupReady + localExpert * 16U, 1);
            continue;
        }
        for (uint32_t row = 0; row < static_cast<uint32_t>(rowCount); ++row) {
            uint32_t globalRow = static_cast<uint32_t>(rowBegin) + row;
            float routingScale = workspaceView.routingPerTokenScale[globalRow];
            M2ComputeSwigluRowPto(shape, workspaceView, globalRow, routingScale);
            InvalidateGmCacheLines(workspaceView.swigluOut + static_cast<uint64_t>(globalRow) * shape.intermediateSize,
                                   static_cast<uint32_t>(shape.intermediateSize * sizeof(float)));
            float maxAbs = 0.0f;
            for (uint32_t col = 0; col < shape.intermediateSize; ++col) {
                float value = workspaceView.swigluOut[static_cast<uint64_t>(globalRow) * shape.intermediateSize + col];
                float absValue = M2Abs(value);
                if (absValue > maxAbs) {
                    maxAbs = absValue;
                }
            }
            float gmm2Scale = maxAbs == 0.0f ? 1.0f : maxAbs / 127.0f;
            workspaceView.gmm2PerTokenScale[globalRow] = gmm2Scale;
            __gm__ int8_t *dst = workspaceView.gmm2InputInt8 + static_cast<uint64_t>(globalRow) * gmm2RowStride;
            for (uint32_t col = 0; col < shape.intermediateSize; ++col) {
                float value = workspaceView.swigluOut[static_cast<uint64_t>(globalRow) * shape.intermediateSize + col];
                dst[col] = M2QuantizeToInt8(value, gmm2Scale);
            }
            for (uint32_t col = shape.intermediateSize; col < gmm2RowStride; ++col) {
                dst[col] = 0;
            }
        }
        StoreScalarI32(workspaceView.activationSyncGroupReady + localExpert * 16U, 1);
    }
    M2BuildSwigluSyncMetadata(shape, workspaceView);
    InvalidateGmCacheLines(workspaceView.swigluOut,
                           static_cast<uint32_t>(localRows * shape.intermediateSize * sizeof(float)));
    InvalidateGmCacheLines(workspaceView.gmm2InputInt8, static_cast<uint32_t>(localRows * gmm2RowStride));
    InvalidateGmCacheLines(workspaceView.gmm2PerTokenScale, static_cast<uint32_t>(localRows * sizeof(float)));
    (void)myRank;
}

__global__ AICORE void M2Int8Dispatch(moe_dispatch_combine_a8w8::ShapeConfig shape,
                                      moe_dispatch_combine_a8w8::RankConfig rank, GM_ADDR inputA, GM_ADDR expertIdx,
                                      GM_ADDR peerWindow, GM_ADDR hcclCtx, GM_ADDR workspace)
{
    uint32_t blockId = static_cast<uint32_t>(get_block_idx());
    if (blockId != 0 || shape.rankNum == 0 || shape.m == 0 || shape.hiddenSize == 0 || shape.topK == 0 ||
        shape.expertPerRank == 0) {
        return;
    }
    auto workspaceLayout = MakeM2WorkspaceLayoutDevice(shape);
    auto peerWindowLayout = MakeM2PeerWindowLayoutDevice(shape);
    M2WorkspaceViewDevice workspaceView = MakeM2WorkspaceViewDevice(workspace, workspaceLayout);
    M2PeerWindowViewDevice localPeer = MakeM2PeerWindowViewDevice(peerWindow, peerWindowLayout);
    __gm__ HcclDeviceContext *ctx = reinterpret_cast<__gm__ HcclDeviceContext *>(hcclCtx);
    uint32_t rowBytes = static_cast<uint32_t>(peerWindowLayout.dispatchPayloadRowBytes);
    uint32_t myRank = rank.rankId;

    M2ClearDispatchState(shape, workspaceView, localPeer);
    M2CountLocalRoutes(shape, workspaceView, localPeer, expertIdx, myRank);
    M2RoutePackQuantLocal(shape, workspaceView, localPeer, inputA, expertIdx, rowBytes);
    M2PublishCountRows(shape, workspaceView, localPeer, ctx, peerWindow, myRank, peerWindowLayout);
    M2WaitCountRows(shape, localPeer);
    M2BuildPrefixMetadata(shape, workspaceView, localPeer, myRank);
    M2GatherDispatchToGmm1Input(shape, workspaceView, localPeer, ctx, peerWindow, myRank, rowBytes, peerWindowLayout);
    StoreScalarI32(workspaceView.swigluSyncGroups, static_cast<int32_t>(shape.expertPerRank));
    StoreScalarI32(workspaceView.dequantSum, 0);
    StoreScalarI32(workspaceView.dequantSum + 1, LoadScalarI32(workspaceView.expertTokenNums));
    StoreScalarI32(localPeer.debugCounters, 1);
}

__global__ AICORE void M2Gmm1Epilogue(moe_dispatch_combine_a8w8::ShapeConfig shape,
                                      moe_dispatch_combine_a8w8::RankConfig rank, GM_ADDR workspace)
{
    uint32_t blockId = static_cast<uint32_t>(get_block_idx());
    if (blockId != 0 || shape.rankNum == 0 || shape.intermediateSize == 0 || shape.expertPerRank == 0) {
        return;
    }
    auto workspaceLayout = MakeM2WorkspaceLayoutDevice(shape);
    M2WorkspaceViewDevice workspaceView = MakeM2WorkspaceViewDevice(workspace, workspaceLayout);
    M2RunGmm1Epilogue(shape, workspaceView, rank.rankId);
}

__global__ AICORE void M2ActivationQuant(moe_dispatch_combine_a8w8::ShapeConfig shape,
                                         moe_dispatch_combine_a8w8::RankConfig rank, GM_ADDR workspace)
{
    uint32_t blockId = static_cast<uint32_t>(get_block_idx());
    if (blockId != 0 || shape.rankNum == 0 || shape.intermediateSize == 0 || shape.expertPerRank == 0) {
        return;
    }
    auto workspaceLayout = MakeM2WorkspaceLayoutDevice(shape);
    M2WorkspaceViewDevice workspaceView = MakeM2WorkspaceViewDevice(workspace, workspaceLayout);
    M2RunActivationQuant(shape, workspaceView, rank.rankId);
}

AICORE inline void M2MaterializeGmm2EpilogueRows(moe_dispatch_combine_a8w8::ShapeConfig shape,
                                                 M2WorkspaceViewDevice workspaceView, int32_t srcStart, int32_t rows,
                                                 int32_t returnRowStride)
{
    for (int32_t row = 0; row < rows; ++row) {
        int32_t srcRow = srcStart + row;
        float tokenScale = workspaceView.gmm2PerTokenScale[srcRow];
        __gm__ half *dst = workspaceView.gmm2Out + static_cast<int64_t>(srcRow) * returnRowStride;
        for (uint32_t col = 0; col < shape.hiddenSize; ++col) {
            int32_t acc = workspaceView.gmm2AccInt32[static_cast<uint64_t>(srcRow) * shape.hiddenSize + col];
            float channelScale = M2DecodeUint64Scale(workspaceView.scale2Uint64[col]);
            dst[col] = static_cast<half>(static_cast<float>(acc) * channelScale * tokenScale);
        }
        for (int32_t col = static_cast<int32_t>(shape.hiddenSize); col < returnRowStride; ++col) {
            dst[col] = static_cast<half>(0.0);
        }
    }
    InvalidateGmCacheLines(workspaceView.gmm2Out + static_cast<int64_t>(srcStart) * returnRowStride,
                           static_cast<uint32_t>(rows * returnRowStride * sizeof(half)));
}

AICORE inline uint32_t M2BuildReturnSegmentMap(moe_dispatch_combine_a8w8::ShapeConfig shape,
                                               M2WorkspaceViewDevice workspaceView, M2PeerWindowViewDevice localPeer,
                                               uint32_t myRank)
{
    uint32_t tileRows = shape.gmmBlockM == 0 ? 16U : shape.gmmBlockM;
    uint32_t capacity = shape.expertPerRank * 8U;
    uint32_t tileId = 0;
    uint32_t segmentId = 0;
    uint32_t maxSegmentsPerTile = 0;
    uint32_t overflow = 0;
    for (uint32_t localExpert = 0; localExpert < shape.expertPerRank; ++localExpert) {
        WaitSignal(workspaceView.gmm2GroupReady + localExpert * 16U, 1);
        uint32_t globalExpert = M2GlobalExpert(myRank, localExpert, shape);
        int32_t rowBegin = LoadScalarI32(workspaceView.dispatchOffset + localExpert);
        int32_t rowCount = LoadScalarI32(workspaceView.expertTokenNums + localExpert);
        if (rowCount <= 0) {
            continue;
        }
        for (int32_t tileOffset = 0; tileOffset < rowCount; tileOffset += static_cast<int32_t>(tileRows)) {
            int32_t tileStart = rowBegin + tileOffset;
            int32_t tileCount = rowCount - tileOffset;
            if (tileCount > static_cast<int32_t>(tileRows)) {
                tileCount = static_cast<int32_t>(tileRows);
            }
            uint32_t firstSegment = segmentId;
            uint32_t tileSegmentCount = 0;
            for (uint32_t tokenOwner = 0; tokenOwner < shape.rankNum; ++tokenOwner) {
                int32_t ownerRows = M2LoadTokenPerExpert(shape, localPeer, tokenOwner, myRank, localExpert);
                if (ownerRows <= 0) {
                    continue;
                }
                int32_t ownerStart =
                    LoadScalarI32(workspaceView.dispatchOffset + localExpert) +
                    LoadScalarI32(workspaceView.preSumBeforeRank + tokenOwner * shape.expertPerRank + localExpert);
                int32_t ownerEnd = ownerStart + ownerRows;
                int32_t tileEnd = tileStart + tileCount;
                int32_t segmentStart = tileStart > ownerStart ? tileStart : ownerStart;
                int32_t segmentEnd = tileEnd < ownerEnd ? tileEnd : ownerEnd;
                if (segmentEnd <= segmentStart) {
                    continue;
                }
                int32_t dstStart =
                    M2SourceGlobalExpertRowBase(shape, localPeer, tokenOwner, globalExpert) + segmentStart - ownerStart;
                if (segmentId < capacity) {
                    __gm__ int32_t *segment = workspaceView.subTileOwnerSegments + segmentId * 6U;
                    StoreScalarI32(segment + 0U, static_cast<int32_t>(tokenOwner));
                    StoreScalarI32(segment + 1U, segmentStart);
                    StoreScalarI32(segment + 2U, segmentEnd - segmentStart);
                    StoreScalarI32(segment + 3U, dstStart);
                    StoreScalarI32(segment + 4U, static_cast<int32_t>(shape.hiddenSize));
                    StoreScalarI32(segment + 5U, static_cast<int32_t>(localExpert));
                } else {
                    ++overflow;
                }
                ++segmentId;
                ++tileSegmentCount;
            }
            if (tileId < capacity) {
                __gm__ int32_t *plan = workspaceView.subTileReturnPlan + tileId * 6U;
                StoreScalarI32(plan + 0U, static_cast<int32_t>(tileId));
                StoreScalarI32(plan + 1U, static_cast<int32_t>(localExpert));
                StoreScalarI32(plan + 2U, tileStart);
                StoreScalarI32(plan + 3U, tileCount);
                StoreScalarI32(plan + 4U, static_cast<int32_t>(firstSegment));
                StoreScalarI32(plan + 5U, static_cast<int32_t>(tileSegmentCount));
                StoreScalarI32(workspaceView.subTileReady + tileId * 16U, 1);
            } else {
                ++overflow;
            }
            if (tileSegmentCount > maxSegmentsPerTile) {
                maxSegmentsPerTile = tileSegmentCount;
            }
            ++tileId;
        }
    }
    StoreScalarI32(localPeer.debugCounters + 2U * 16U, static_cast<int32_t>(tileId));
    StoreScalarI32(localPeer.debugCounters + 3U * 16U, static_cast<int32_t>(segmentId));
    StoreScalarI32(localPeer.debugCounters + 4U * 16U, static_cast<int32_t>(maxSegmentsPerTile));
    StoreScalarI32(localPeer.debugCounters + 5U * 16U, static_cast<int32_t>(overflow));
    return segmentId;
}

AICORE inline void M2RunGmm2EpilogueAndReturn(moe_dispatch_combine_a8w8::ShapeConfig shape,
                                              M2WorkspaceViewDevice workspaceView, M2PeerWindowViewDevice localPeer,
                                              __gm__ HcclDeviceContext *ctx, GM_ADDR peerWindow, uint32_t myRank,
                                              const moe_dispatch_combine_a8w8::PeerWindowLayout &peerWindowLayout)
{
    int32_t returnRowStride = static_cast<int32_t>(peerWindowLayout.returnPayloadRowBytes / sizeof(half));
    (void)M2BuildReturnSegmentMap(shape, workspaceView, localPeer, myRank);
    int32_t segmentCount = 0;
    for (uint32_t tokenOwner = 0; tokenOwner < shape.rankNum; ++tokenOwner) {
        M2PeerWindowViewDevice remotePeer =
            MakeM2RemotePeerWindowViewDevice(ctx, peerWindow, tokenOwner, peerWindowLayout);
        for (uint32_t localExpert = 0; localExpert < shape.expertPerRank; ++localExpert) {
            WaitSignal(workspaceView.gmm2GroupReady + localExpert * 16U, 1);
            uint32_t globalExpert = M2GlobalExpert(myRank, localExpert, shape);
            int32_t rows = M2LoadTokenPerExpert(shape, localPeer, tokenOwner, myRank, localExpert);
            if (rows <= 0) {
                continue;
            }
            int32_t srcStart =
                LoadScalarI32(workspaceView.dispatchOffset + localExpert) +
                LoadScalarI32(workspaceView.preSumBeforeRank + tokenOwner * shape.expertPerRank + localExpert);
            int32_t dstStart = M2SourceGlobalExpertRowBase(shape, localPeer, tokenOwner, globalExpert);
            M2MaterializeGmm2EpilogueRows(shape, workspaceView, srcStart, rows, returnRowStride);
            if (tokenOwner == myRank) {
                for (int32_t row = 0; row < rows; ++row) {
                    CopyRowHalf(localPeer.returnPayload, returnRowStride, dstStart + row, workspaceView.gmm2Out,
                                returnRowStride, srcStart + row, static_cast<int32_t>(shape.hiddenSize));
                }
            } else {
                TPutRowsHalf(remotePeer.returnPayload, returnRowStride, dstStart, workspaceView.gmm2Out,
                             returnRowStride, srcStart, rows, static_cast<int32_t>(shape.hiddenSize));
            }
            NotifySignal(remotePeer.returnSegmentCounters +
                             (static_cast<uint64_t>(myRank) * shape.expertPerRank + localExpert) * 16U,
                         1);
            ++segmentCount;
        }
        NotifySignal(remotePeer.combineDoneSignal + myRank * 16U, 1);
    }
    StoreScalarI32(localPeer.debugCounters, segmentCount);
    StoreScalarI32(localPeer.debugCounters + 16U, 1);
}

__global__ AICORE void M2Gmm2EpilogueAndReturn(moe_dispatch_combine_a8w8::ShapeConfig shape,
                                               moe_dispatch_combine_a8w8::RankConfig rank, GM_ADDR peerWindow,
                                               GM_ADDR hcclCtx, GM_ADDR workspace)
{
    uint32_t blockId = static_cast<uint32_t>(get_block_idx());
    if (blockId != 0 || shape.rankNum == 0 || shape.hiddenSize == 0 || shape.expertPerRank == 0) {
        return;
    }
    auto workspaceLayout = MakeM2WorkspaceLayoutDevice(shape);
    auto peerWindowLayout = MakeM2PeerWindowLayoutDevice(shape);
    M2WorkspaceViewDevice workspaceView = MakeM2WorkspaceViewDevice(workspace, workspaceLayout);
    M2PeerWindowViewDevice localPeer = MakeM2PeerWindowViewDevice(peerWindow, peerWindowLayout);
    __gm__ HcclDeviceContext *ctx = reinterpret_cast<__gm__ HcclDeviceContext *>(hcclCtx);
    M2RunGmm2EpilogueAndReturn(shape, workspaceView, localPeer, ctx, peerWindow, rank.rankId, peerWindowLayout);
}

AICORE inline void WaitCombinePhase(DispatchCombineTileShape shape, LocalPeerWindowView localPeer, uint32_t blockId,
                                    uint32_t blockNum, int32_t value)
{
    for (uint32_t peer = blockId; peer < shape.ep; peer += blockNum) {
        WaitSignal(localPeer.combineDoneSignal + peer, value);
    }
}

AICORE inline void ReturnExpertRowsToOwners(DispatchCombineTileShape shape, LocalWorkspaceView workspaceView,
                                            LocalPeerWindowView localPeer, __gm__ HcclDeviceContext *ctx,
                                            GM_ADDR peerWindow, GM_ADDR expertOutput, uint32_t myRank, uint32_t blockId,
                                            uint32_t blockNum, const PeerWindowLayout &peerWindowLayout)
{
    uint32_t expertNumPadded = ExpertNumPaddedDevice(shape);
    __gm__ half *localExpertOutput = reinterpret_cast<__gm__ half *>(expertOutput);
    InvalidateGmCacheLines(localPeer.peerTokenPerExpert,
                           static_cast<uint32_t>(shape.ep * expertNumPadded * sizeof(int32_t)));
    InvalidateGmCacheLines(workspaceView.cumsumPerExpert,
                           static_cast<uint32_t>(shape.ep * expertNumPadded * sizeof(int32_t)));
    InvalidateGmCacheLines(workspaceView.dispatchOffset, static_cast<uint32_t>(shape.expertPerRank * sizeof(int32_t)));
    InvalidateGmCacheLines(workspaceView.prevSumBeforeRank,
                           static_cast<uint32_t>(shape.ep * shape.expertPerRank * sizeof(int32_t)));
    for (uint32_t src = blockId; src < shape.ep; src += blockNum) {
        LocalPeerWindowView remotePeer = MakeRemotePeerWindowView(ctx, peerWindow, src, peerWindowLayout);
        for (uint32_t localExpert = 0; localExpert < shape.expertPerRank; ++localExpert) {
            uint32_t globalExpert = myRank * shape.expertPerRank + localExpert;
            int32_t rows = LoadScalarI32(localPeer.peerTokenPerExpert + static_cast<uint64_t>(src) * expertNumPadded +
                                         globalExpert);
            if (rows <= 0) {
                continue;
            }
            int32_t srcStart = LoadScalarI32(workspaceView.dispatchOffset + localExpert) +
                               LoadScalarI32(workspaceView.prevSumBeforeRank +
                                             static_cast<uint64_t>(src) * shape.expertPerRank + localExpert);
            int32_t dstStart = globalExpert == 0 ?
                                   0 :
                                   LoadScalarI32(workspaceView.cumsumPerExpert +
                                                 static_cast<uint64_t>(src) * expertNumPadded + globalExpert - 1);
            if (src == myRank) {
                for (int32_t row = 0; row < rows; ++row) {
                    CopyRowHalf(localPeer.ptrD, static_cast<int32_t>(shape.k), dstStart + row, localExpertOutput,
                                static_cast<int32_t>(shape.k), srcStart + row, static_cast<int32_t>(shape.k));
                }
            } else {
                TPutRowsHalf(remotePeer.ptrD, static_cast<int32_t>(shape.k), dstStart, localExpertOutput,
                             static_cast<int32_t>(shape.k), srcStart, rows, static_cast<int32_t>(shape.k));
            }
        }
        NotifySignal(remotePeer.combineDoneSignal + myRank, static_cast<int32_t>(shape.signalValue));
    }
}

AICORE inline void StoreZeroRowHalf(__gm__ half *dstBase, int32_t dstRowStride, int32_t dstRow, int32_t rowLen)
{
    for (int32_t col = 0; col < rowLen; col += kDefaultTileCols) {
        int32_t cols = rowLen - col < kDefaultTileCols ? rowLen - col : kDefaultTileCols;
        VecTile<half, kDefaultTileCols> zeroTile(1, cols);
        TASSIGN(zeroTile, kPingUbAddr);
        GlobalNd<half> dst =
            MakeGlobal2D(dstBase + static_cast<int64_t>(dstRow) * dstRowStride + col, 1, cols, dstRowStride);
        TEXPANDS(zeroTile, static_cast<half>(0.0));
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        TSTORE(dst, zeroTile);
        WaitStoreTileReusable();
    }
}

AICORE inline void AddWeightedRowHalf(__gm__ half *outputBase, __gm__ half *ptrDBase, int32_t rowStride,
                                      int32_t outputRow, int32_t ptrDRow, int32_t rowLen, half prob)
{
    for (int32_t col = 0; col < rowLen; col += kDefaultTileCols) {
        int32_t cols = rowLen - col < kDefaultTileCols ? rowLen - col : kDefaultTileCols;
        VecTile<half, kDefaultTileCols> outTile(1, cols);
        VecTile<half, kDefaultTileCols> ptrTile(1, cols);
        TASSIGN(outTile, kPingUbAddr);
        TASSIGN(ptrTile, kPongUbAddr);
        GlobalNd<half> outGlobal =
            MakeGlobal2D(outputBase + static_cast<int64_t>(outputRow) * rowStride + col, 1, cols, rowStride);
        __gm__ half *ptrChunk = ptrDBase + static_cast<int64_t>(ptrDRow) * rowStride + col;
        InvalidateGmCacheLines(ptrChunk, static_cast<uint32_t>(cols) * sizeof(half));
        GlobalNd<half> ptrGlobal = MakeGlobal2D(ptrChunk, 1, cols, rowStride);
        pto::Event<pto::Op::TLOAD, pto::Op::TAXPY> loadToAxpy;
        pto::Event<pto::Op::TAXPY, pto::Op::TSTORE_VEC> axpyToStore;
        TLOAD(ptrTile, ptrGlobal);
        loadToAxpy = TLOAD(outTile, outGlobal);
        axpyToStore = TAXPY(outTile, ptrTile, prob, loadToAxpy);
        TSTORE(outGlobal, outTile, axpyToStore);
        WaitStoreTileReusable();
    }
}

AICORE inline void RestoreOutputRows(DispatchCombineTileShape shape, LocalPeerWindowView localPeer, GM_ADDR probs,
                                     GM_ADDR outputC, uint32_t blockId, uint32_t blockNum)
{
    __gm__ float *probValues = reinterpret_cast<__gm__ float *>(probs);
    __gm__ half *output = reinterpret_cast<__gm__ half *>(outputC);
    uint32_t tokenBegin = TokenShardBegin(shape.m, blockId, blockNum);
    uint32_t tokenEnd = TokenShardEnd(shape.m, blockId, blockNum);
    for (uint32_t token = tokenBegin; token < tokenEnd; ++token) {
        StoreZeroRowHalf(output, static_cast<int32_t>(shape.k), static_cast<int32_t>(token),
                         static_cast<int32_t>(shape.k));
        for (uint32_t slot = 0; slot < shape.topK; ++slot) {
            uint32_t routeIndex = token * shape.topK + slot;
            int32_t ptrDRow = LoadScalarI32(localPeer.expandedRowIdx + routeIndex);
            if (ptrDRow < 0) {
                continue;
            }
            float prob = probValues[routeIndex];
            AddWeightedRowHalf(output, localPeer.ptrD, static_cast<int32_t>(shape.k), static_cast<int32_t>(token),
                               ptrDRow, static_cast<int32_t>(shape.k), static_cast<half>(prob));
        }
    }
}

__global__ AICORE void M2RestoreOutput(moe_dispatch_combine_a8w8::ShapeConfig shape,
                                       moe_dispatch_combine_a8w8::RankConfig rank, GM_ADDR probs, GM_ADDR outputC,
                                       GM_ADDR peerWindow, GM_ADDR workspace)
{
    uint32_t blockId = static_cast<uint32_t>(get_block_idx());
    uint32_t blockNum = static_cast<uint32_t>(get_block_num());
    if (shape.rankNum == 0 || shape.m == 0 || shape.hiddenSize == 0 || shape.topK == 0 || blockNum == 0 ||
        blockId >= blockNum) {
        return;
    }
    auto workspaceLayout = MakeM2WorkspaceLayoutDevice(shape);
    auto peerWindowLayout = MakeM2PeerWindowLayoutDevice(shape);
    M2WorkspaceViewDevice workspaceView = MakeM2WorkspaceViewDevice(workspace, workspaceLayout);
    M2PeerWindowViewDevice localPeer = MakeM2PeerWindowViewDevice(peerWindow, peerWindowLayout);
    __gm__ float *probValues = reinterpret_cast<__gm__ float *>(probs);
    __gm__ half *output = reinterpret_cast<__gm__ half *>(outputC);
    int32_t outputRowStride = static_cast<int32_t>(shape.hiddenSize);
    int32_t returnRowStride = static_cast<int32_t>(peerWindowLayout.returnPayloadRowBytes / sizeof(half));

    for (uint32_t peer = 0; peer < shape.rankNum; ++peer) {
        WaitSignal(localPeer.combineDoneSignal + peer * 16U, 1);
    }

    uint32_t tokenBegin = TokenShardBegin(shape.m, blockId, blockNum);
    uint32_t tokenEnd = TokenShardEnd(shape.m, blockId, blockNum);
    for (uint32_t token = tokenBegin; token < tokenEnd; ++token) {
        StoreZeroRowHalf(output, outputRowStride, static_cast<int32_t>(token), static_cast<int32_t>(shape.hiddenSize));
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
                GlobalNd<half> outGlobal = MakeGlobal2D(output + static_cast<int64_t>(token) * outputRowStride + col, 1,
                                                        cols, outputRowStride);
                __gm__ half *returnChunk =
                    localPeer.returnPayload + static_cast<int64_t>(ptrDRow) * returnRowStride + col;
                InvalidateGmCacheLines(returnChunk, static_cast<uint32_t>(cols) * sizeof(half));
                GlobalNd<half> returnGlobal = MakeGlobal2D(returnChunk, 1, cols, returnRowStride);
                pto::Event<pto::Op::TLOAD, pto::Op::TAXPY> loadToAxpy;
                pto::Event<pto::Op::TAXPY, pto::Op::TSTORE_VEC> axpyToStore;
                TLOAD(ptrTile, returnGlobal);
                loadToAxpy = TLOAD(outTile, outGlobal);
                axpyToStore = TAXPY(outTile, ptrTile, static_cast<half>(prob), loadToAxpy);
                TSTORE(outGlobal, outTile, axpyToStore);
                WaitStoreTileReusable();
            }
        }
    }
    if (blockId == 0) {
        StoreScalarI32(localPeer.debugCounters + 6U * 16U, 1);
    }
    (void)rank;
}

} // namespace

__global__ AICORE void DispatchCombineTileDispatch(DispatchCombineTileShape shape, uint32_t myRank, GM_ADDR inputA,
                                                   GM_ADDR expertIdx, GM_ADDR peerWindow, GM_ADDR hcclCtx,
                                                   GM_ADDR workspace)
{
    WorkspaceLayout workspaceLayout = MakeWorkspaceLayout(shape);
    PeerWindowLayout peerWindowLayout = MakePeerWindowLayout(shape);
    LocalWorkspaceView workspaceView = MakeLocalWorkspaceView(workspace, workspaceLayout);
    LocalPeerWindowView localPeer = MakeLocalPeerWindowView(peerWindow, peerWindowLayout);
    __gm__ HcclDeviceContext *ctx = reinterpret_cast<__gm__ HcclDeviceContext *>(hcclCtx);
    uint32_t blockId = static_cast<uint32_t>(get_block_idx());
    uint32_t blockNum = shape.aivBlocks == 0 ? 1 : shape.aivBlocks;
    if (shape.ep == 0 || shape.m == 0 || shape.topK == 0 || shape.expertPerRank == 0 || shape.expertNum == 0 ||
        shape.metadataPad == 0 || blockId >= blockNum) {
        return;
    }

    ClearDispatchState(shape, workspaceView, localPeer, blockId, blockNum);
    SoftSyncAiv(workspaceView.localSync, blockNum);
    CountLocalRoutes(shape, workspaceView, localPeer, expertIdx, blockId, blockNum);
    SoftSyncAiv(workspaceView.localSync, blockNum);
    BuildBlockPrefixAndLocalCounts(shape, workspaceView, blockId, blockNum);
    SoftSyncAiv(workspaceView.localSync, blockNum);
    BuildPackedExpertOffset(shape, workspaceView, myRank, blockId);
    SoftSyncAiv(workspaceView.localSync, blockNum);
    InitPackCursors(shape, workspaceView, blockId, blockNum);
    SoftSyncAiv(workspaceView.localSync, blockNum);
    PackLocalRowsToWindow(shape, workspaceView, localPeer, inputA, expertIdx, myRank, blockId, blockNum);
    SoftSyncAiv(workspaceView.localSync, blockNum);
    RebuildExpandedRowIdx(shape, workspaceView, localPeer, expertIdx, myRank, blockId, blockNum);
    SoftSyncAiv(workspaceView.localSync, blockNum);
    PublishCountRows(shape, workspaceView, localPeer, ctx, peerWindow, myRank, blockId, blockNum, peerWindowLayout);
    SoftSyncAiv(workspaceView.localSync, blockNum);
    WaitCountRows(shape, localPeer, blockId, blockNum);
    SoftSyncAiv(workspaceView.localSync, blockNum);
    BuildPrefixMetadata(shape, workspaceView, localPeer, myRank, blockId, blockNum);
    SoftSyncAiv(workspaceView.localSync, blockNum);
    GatherLocalExpertPayload(shape, workspaceView, localPeer, ctx, peerWindow, myRank, blockId, blockNum,
                             peerWindowLayout);
    SoftSyncAiv(workspaceView.localSync, blockNum);
}

__global__ AICORE void DispatchCombineTileCombine(DispatchCombineTileShape shape, uint32_t myRank, GM_ADDR expertOutput,
                                                  GM_ADDR probs, GM_ADDR outputC, GM_ADDR peerWindow, GM_ADDR hcclCtx,
                                                  GM_ADDR workspace)
{
    WorkspaceLayout workspaceLayout = MakeWorkspaceLayout(shape);
    PeerWindowLayout peerWindowLayout = MakePeerWindowLayout(shape);
    LocalWorkspaceView workspaceView = MakeLocalWorkspaceView(workspace, workspaceLayout);
    LocalPeerWindowView localPeer = MakeLocalPeerWindowView(peerWindow, peerWindowLayout);
    __gm__ HcclDeviceContext *ctx = reinterpret_cast<__gm__ HcclDeviceContext *>(hcclCtx);
    uint32_t blockId = static_cast<uint32_t>(get_block_idx());
    uint32_t blockNum = shape.aivBlocks == 0 ? 1 : shape.aivBlocks;
    if (shape.ep == 0 || shape.m == 0 || shape.k == 0 || shape.topK == 0 || shape.expertPerRank == 0 ||
        shape.expertNum == 0 || shape.metadataPad == 0 || blockId >= blockNum) {
        return;
    }

    ReturnExpertRowsToOwners(shape, workspaceView, localPeer, ctx, peerWindow, expertOutput, myRank, blockId, blockNum,
                             peerWindowLayout);
    SoftSyncAiv(workspaceView.localSync, blockNum);
    WaitCombinePhase(shape, localPeer, blockId, blockNum, static_cast<int32_t>(shape.signalValue));
    SoftSyncAiv(workspaceView.localSync, blockNum);
    RestoreOutputRows(shape, localPeer, probs, outputC, blockId, blockNum);
    SoftSyncAiv(workspaceView.localSync, blockNum);
}

namespace dispatch_combine_tile {

void LaunchDispatchCombineTileDispatch(DispatchCombineTileShape shape, uint32_t myRank, uint8_t *inputA,
                                       uint8_t *expertIdx, uint8_t *peerWindow, uint8_t *hcclCtx, uint8_t *workspace,
                                       void *stream, uint32_t launchBlockCount)
{
    DispatchCombineTileDispatch<<<launchBlockCount, nullptr, stream>>>(shape, myRank, inputA, expertIdx, peerWindow,
                                                                       hcclCtx, workspace);
}

void LaunchDispatchCombineTileCombine(DispatchCombineTileShape shape, uint32_t myRank, uint8_t *expertOutput,
                                      uint8_t *probs, uint8_t *outputC, uint8_t *peerWindow, uint8_t *hcclCtx,
                                      uint8_t *workspace, void *stream, uint32_t launchBlockCount)
{
    DispatchCombineTileCombine<<<launchBlockCount, nullptr, stream>>>(shape, myRank, expertOutput, probs, outputC,
                                                                      peerWindow, hcclCtx, workspace);
}

void LaunchM2Int8Dispatch(moe_dispatch_combine_a8w8::ShapeConfig shape, moe_dispatch_combine_a8w8::RankConfig rank,
                          uint8_t *inputA, uint8_t *expertIdx, uint8_t *peerWindow, uint8_t *hcclCtx,
                          uint8_t *workspace, void *stream, uint32_t launchBlockCount)
{
    M2Int8Dispatch<<<launchBlockCount, nullptr, stream>>>(shape, rank, inputA, expertIdx, peerWindow, hcclCtx,
                                                          workspace);
}

void LaunchM2Gmm1Epilogue(moe_dispatch_combine_a8w8::ShapeConfig shape, moe_dispatch_combine_a8w8::RankConfig rank,
                          uint8_t *workspace, void *stream, uint32_t launchBlockCount)
{
    M2Gmm1Epilogue<<<launchBlockCount, nullptr, stream>>>(shape, rank, workspace);
}

void LaunchM2ActivationQuant(moe_dispatch_combine_a8w8::ShapeConfig shape, moe_dispatch_combine_a8w8::RankConfig rank,
                             uint8_t *workspace, void *stream, uint32_t launchBlockCount)
{
    M2ActivationQuant<<<launchBlockCount, nullptr, stream>>>(shape, rank, workspace);
}

void LaunchM2Gmm2EpilogueAndReturn(moe_dispatch_combine_a8w8::ShapeConfig shape,
                                   moe_dispatch_combine_a8w8::RankConfig rank, uint8_t *peerWindow, uint8_t *hcclCtx,
                                   uint8_t *workspace, void *stream, uint32_t launchBlockCount)
{
    M2Gmm2EpilogueAndReturn<<<launchBlockCount, nullptr, stream>>>(shape, rank, peerWindow, hcclCtx, workspace);
}

void LaunchM2RestoreOutput(moe_dispatch_combine_a8w8::ShapeConfig shape, moe_dispatch_combine_a8w8::RankConfig rank,
                           uint8_t *probs, uint8_t *outputC, uint8_t *peerWindow, uint8_t *workspace, void *stream,
                           uint32_t launchBlockCount)
{
    M2RestoreOutput<<<launchBlockCount, nullptr, stream>>>(shape, rank, probs, outputC, peerWindow, workspace);
}

} // namespace dispatch_combine_tile
