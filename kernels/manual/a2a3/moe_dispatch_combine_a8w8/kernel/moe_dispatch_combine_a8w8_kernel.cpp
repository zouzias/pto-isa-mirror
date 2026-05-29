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
constexpr int kM2RouteQuantTileCols = 1024;
constexpr uint64_t kM2RouteHalfTileOffset = 0x0;
constexpr uint64_t kM2RouteFloatTileOffset = 0x1000;
constexpr uint64_t kM2RouteAbsTileOffset = 0x2400;
constexpr uint64_t kM2RouteQuantTileOffset = 0x3800;
constexpr uint64_t kM2RouteRowMaxTileOffset = 0x3C00;
constexpr uint64_t kM2RouteChunkMaxTileOffset = 0x3D00;
constexpr uint64_t kM2RouteScaleParamTileOffset = 0x3E00;
constexpr uint64_t kM2RouteInvScaleParamTileOffset = 0x3F00;
constexpr uint64_t kM2RouteScaleStoreTileOffset = 0x4000;
constexpr int kM2EpilogueTileCols = 1024;
constexpr uint64_t kM2EpilogueAccTileOffset = 0x0;
constexpr uint64_t kM2EpilogueFloatTileOffset = 0x1000;
constexpr uint64_t kM2EpilogueScaleTileOffset = 0x2400;
constexpr uint64_t kM2EpilogueScaledTileOffset = 0x3800;
constexpr uint64_t kM2EpilogueHalfTileOffset = 0x4C00;
constexpr int32_t kM2ScoreboardStatusInit = 0;
constexpr int32_t kM2ScoreboardStatusRowsPresent = 1;
constexpr int32_t kM2ScoreboardStatusCopyDone = 2;
constexpr int32_t kM2ScoreboardStatusSkipDone = 3;
constexpr uint32_t kM2GmmBaseM = moe_dispatch_combine_a8w8::kGmmBaseM;
constexpr uint32_t kM2GmmBaseN = moe_dispatch_combine_a8w8::kGmmBaseN;
constexpr uint32_t kM2ReturnTileRows = moe_dispatch_combine_a8w8::kReturnTileRows;
constexpr uint32_t kM2SwigluGroupFields = 8;
constexpr uint32_t kM2GmmTileTaskFields = 8;
constexpr uint32_t kM2ReturnPlanFields = 8;
constexpr uint32_t kM2OwnerSegmentFields = 8;
constexpr uint32_t kM3CounterBase = 24U * 16U;

using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;

template <typename T>
using GlobalNd = pto::GlobalTensor<T, ShapeDyn, StrideDyn, pto::Layout::ND>;

template <typename T, int kCols = kDefaultTileCols>
using VecTile = pto::Tile<pto::TileType::Vec, T, 1, kCols, pto::BLayout::RowMajor, -1, -1>;

using M2RouteHalfTile =
    pto::Tile<pto::TileType::Vec, half, 1, kM2RouteQuantTileCols, pto::BLayout::RowMajor, 1, pto::DYNAMIC>;
using M2RouteFloatTile =
    pto::Tile<pto::TileType::Vec, float, 1, kM2RouteQuantTileCols, pto::BLayout::RowMajor, 1, pto::DYNAMIC>;
using M2RouteQuantTile =
    pto::Tile<pto::TileType::Vec, int8_t, 1, kM2RouteQuantTileCols, pto::BLayout::RowMajor, 1, pto::DYNAMIC>;
using M2RouteRowStatTile = pto::Tile<pto::TileType::Vec, float, 1, 16, pto::BLayout::RowMajor, 1, 1>;
using M2RouteParamTile = pto::Tile<pto::TileType::Vec, float, 1, 8, pto::BLayout::RowMajor, 1, 8>;
using M2RouteScaleStoreTile = pto::Tile<pto::TileType::Vec, float, 1, 8, pto::BLayout::RowMajor, 1, 1>;
using M2EpilogueAccTile =
    pto::Tile<pto::TileType::Vec, int32_t, 1, kM2EpilogueTileCols, pto::BLayout::RowMajor, 1, pto::DYNAMIC>;
using M2EpilogueFloatTile =
    pto::Tile<pto::TileType::Vec, float, 1, kM2EpilogueTileCols, pto::BLayout::RowMajor, 1, pto::DYNAMIC>;
using M2EpilogueHalfTile =
    pto::Tile<pto::TileType::Vec, half, 1, kM2EpilogueTileCols, pto::BLayout::RowMajor, 1, pto::DYNAMIC>;

AICORE inline uint64_t Align64Device(uint64_t value)
{
    return ((value + 63) / 64) * 64;
}

AICORE inline uint64_t M2CeilDivDevice(uint64_t value, uint64_t divisor)
{
    if (divisor == 0) {
        return 0;
    }
    return (value + divisor - 1U) / divisor;
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

AICORE inline uint64_t M2ReturnHiddenChunkCols(moe_dispatch_combine_a8w8::ShapeConfig shape)
{
    return shape.gmmBlockN == 0 ? kM2GmmBaseN : shape.gmmBlockN;
}

AICORE inline uint64_t M2GmmTileTaskCapacity(moe_dispatch_combine_a8w8::ShapeConfig shape)
{
    uint64_t rowTiles = M2CeilDivDevice(M2LocalRows(shape), kM2GmmBaseM);
    uint64_t w1Cols = static_cast<uint64_t>(shape.intermediateSize) * 2U;
    uint64_t gmm1NTiles = M2CeilDivDevice(w1Cols, kM2GmmBaseN);
    uint64_t gmm2NTiles = M2CeilDivDevice(shape.hiddenSize, kM2GmmBaseN);
    uint64_t nTiles = gmm1NTiles > gmm2NTiles ? gmm1NTiles : gmm2NTiles;
    return static_cast<uint64_t>(shape.expertPerRank) * rowTiles * nTiles;
}

AICORE inline uint64_t M2ReturnSegmentCapacity(moe_dispatch_combine_a8w8::ShapeConfig shape)
{
    uint64_t rowTiles = M2CeilDivDevice(M2LocalRows(shape), kM2ReturnTileRows);
    uint64_t hiddenChunks = M2CeilDivDevice(shape.hiddenSize, M2ReturnHiddenChunkCols(shape));
    return static_cast<uint64_t>(shape.expertPerRank) * rowTiles * hiddenChunks * shape.rankNum;
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
    uint64_t gmmTaskCap = M2GmmTileTaskCapacity(shape);
    uint64_t subTileCap = M2ReturnSegmentCapacity(shape);

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
    layout.returnSegmentStaging =
        M2AppendFieldDevice(offset, kM2ReturnTileRows * M2ReturnHiddenChunkCols(shape) * M2DTypeBytes(shape.dtypeOut));
    layout.readyCounters = M2AppendFieldDevice(offset, 16U * 64U);
    layout.dispatchGroupReady = M2AppendFieldDevice(offset, shape.expertPerRank * 64U);
    layout.gmm1SyncGroupReady = M2AppendFieldDevice(offset, syncGroupCap * 64U);
    layout.activationSyncGroupReady = M2AppendFieldDevice(offset, syncGroupCap * 64U);
    layout.gmm2GroupReady = M2AppendFieldDevice(offset, shape.expertPerRank * 64U);
    layout.stageStatus = M2AppendFieldDevice(offset, 16U * 64U);
    layout.swigluSyncGroups = M2AppendFieldDevice(offset, syncGroupCap * sizeof(int32_t));
    layout.dequantSum = M2AppendFieldDevice(offset, (syncGroupCap + 1U) * sizeof(int32_t));
    layout.swigluGroupDesc = M2AppendFieldDevice(offset, syncGroupCap * kM2SwigluGroupFields * sizeof(int32_t));
    layout.gmm1TileTaskPlan = M2AppendFieldDevice(offset, gmmTaskCap * kM2GmmTileTaskFields * sizeof(int32_t));
    layout.gmm2TileTaskPlan = M2AppendFieldDevice(offset, gmmTaskCap * kM2GmmTileTaskFields * sizeof(int32_t));
    layout.scoreboardTaskMap = M2AppendFieldDevice(offset, scoreboardTasks * 4U * sizeof(int32_t));
    layout.producerStatus = M2AppendFieldDevice(offset, scoreboardTasks * 64U);
    layout.scoreboardMinStatus = M2AppendFieldDevice(offset, scoreboardTasks * 64U);
    layout.workerWaitCounters = M2AppendFieldDevice(offset, scoreboardTasks * 64U);
    layout.scoreboardTimeoutCounters = M2AppendFieldDevice(offset, scoreboardTasks * 64U);
    layout.subTileReturnPlan = M2AppendFieldDevice(offset, subTileCap * kM2ReturnPlanFields * sizeof(int32_t));
    layout.subTileOwnerSegments = M2AppendFieldDevice(offset, subTileCap * kM2OwnerSegmentFields * sizeof(int32_t));
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
    __gm__ half *returnSegmentStaging;
    __gm__ int32_t *dispatchGroupReady;
    __gm__ int32_t *gmm1SyncGroupReady;
    __gm__ int32_t *activationSyncGroupReady;
    __gm__ int32_t *gmm2GroupReady;
    __gm__ int32_t *stageStatus;
    __gm__ int32_t *swigluSyncGroups;
    __gm__ int32_t *dequantSum;
    __gm__ int32_t *swigluGroupDesc;
    __gm__ int32_t *gmm1TileTaskPlan;
    __gm__ int32_t *gmm2TileTaskPlan;
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
    view.returnSegmentStaging = reinterpret_cast<__gm__ half *>(workspaceBase + layout.returnSegmentStaging.offset);
    view.dispatchGroupReady = reinterpret_cast<__gm__ int32_t *>(workspaceBase + layout.dispatchGroupReady.offset);
    view.gmm1SyncGroupReady = reinterpret_cast<__gm__ int32_t *>(workspaceBase + layout.gmm1SyncGroupReady.offset);
    view.activationSyncGroupReady =
        reinterpret_cast<__gm__ int32_t *>(workspaceBase + layout.activationSyncGroupReady.offset);
    view.gmm2GroupReady = reinterpret_cast<__gm__ int32_t *>(workspaceBase + layout.gmm2GroupReady.offset);
    view.stageStatus = reinterpret_cast<__gm__ int32_t *>(workspaceBase + layout.stageStatus.offset);
    view.swigluSyncGroups = reinterpret_cast<__gm__ int32_t *>(workspaceBase + layout.swigluSyncGroups.offset);
    view.dequantSum = reinterpret_cast<__gm__ int32_t *>(workspaceBase + layout.dequantSum.offset);
    view.swigluGroupDesc = reinterpret_cast<__gm__ int32_t *>(workspaceBase + layout.swigluGroupDesc.offset);
    view.gmm1TileTaskPlan = reinterpret_cast<__gm__ int32_t *>(workspaceBase + layout.gmm1TileTaskPlan.offset);
    view.gmm2TileTaskPlan = reinterpret_cast<__gm__ int32_t *>(workspaceBase + layout.gmm2TileTaskPlan.offset);
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

AICORE inline bool M2TokenIsActive(GM_ADDR xActiveMask, uint32_t token)
{
    if (xActiveMask == nullptr) {
        return true;
    }
    __gm__ uint8_t *mask = reinterpret_cast<__gm__ uint8_t *>(xActiveMask);
    return mask[token] != 0U;
}

AICORE inline int32_t M2EffectiveTokenOwnerRows(moe_dispatch_combine_a8w8::ShapeConfig shape,
                                                M2PeerWindowViewDevice localPeer, uint32_t tokenOwnerRank,
                                                uint32_t expertOwnerRank, uint32_t localExpert)
{
    int32_t cursor = 0;
    int32_t cap = static_cast<int32_t>(M2LocalRows(shape));
    for (uint32_t prevLocalExpert = 0; prevLocalExpert < shape.expertPerRank; ++prevLocalExpert) {
        for (uint32_t src = 0; src < shape.rankNum; ++src) {
            int32_t rows = M2LoadTokenPerExpert(shape, localPeer, src, expertOwnerRank, prevLocalExpert);
            int32_t effective = 0;
            if (cursor < cap && rows > 0) {
                int32_t available = cap - cursor;
                effective = rows < available ? rows : available;
            }
            if (prevLocalExpert == localExpert && src == tokenOwnerRank) {
                return effective;
            }
            cursor += effective;
        }
    }
    return 0;
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

AICORE inline void TPutContiguousHalf(__gm__ half *remoteDst, __gm__ half *src, int32_t elems)
{
    if (elems <= 0) {
        return;
    }
    for (int32_t offset = 0; offset < elems; offset += kDefaultTileCols) {
        int32_t cols = elems - offset;
        if (cols > kDefaultTileCols) {
            cols = kDefaultTileCols;
        }
        ShapeDyn shape(1, 1, 1, 1, cols);
        StrideDyn stride(cols, cols, cols, cols, 1);
        GlobalNd<half> dst(remoteDst + offset, shape, stride);
        GlobalNd<half> srcGlobal(src + offset, shape, stride);
        VecTile<half, kDefaultTileCols> tile(1, cols);
        TASSIGN(tile, kPingUbAddr);
        pto::comm::TPUT(dst, srcGlobal, tile);
        pipe_barrier(PIPE_ALL);
    }
    dsb(DSB_DDR);
}

AICORE inline void TPutRowsHalfContiguous(__gm__ half *remoteDstBase, int32_t dstRowStride, int32_t dstRow,
                                          __gm__ half *srcBase, int32_t srcRowStride, int32_t srcRow, int32_t rows,
                                          int32_t cols)
{
    if (rows <= 0 || cols <= 0) {
        return;
    }
    for (int32_t row = 0; row < rows; ++row) {
        __gm__ half *dst = remoteDstBase + static_cast<int64_t>(dstRow + row) * dstRowStride;
        __gm__ half *src = srcBase + static_cast<int64_t>(srcRow + row) * srcRowStride;
        TPutContiguousHalf(dst, src, cols);
    }
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
    uint32_t syncGroupCap = shape.expertPerRank + 1U;
    uint32_t gmmTaskCap = static_cast<uint32_t>(M2GmmTileTaskCapacity(shape));
    uint32_t subTileCap = static_cast<uint32_t>(M2ReturnSegmentCapacity(shape));
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
        StoreScalarI32(workspaceView.gmm2GroupReady + localExpert * 16U, 0);
    }
    for (uint32_t idx = 0; idx < syncGroupCap; ++idx) {
        StoreScalarI32(workspaceView.swigluSyncGroups + idx, 0);
        StoreScalarI32(workspaceView.dequantSum + idx, 0);
        StoreScalarI32(workspaceView.gmm1SyncGroupReady + idx * 16U, 0);
        StoreScalarI32(workspaceView.activationSyncGroupReady + idx * 16U, 0);
        for (uint32_t field = 0; field < kM2SwigluGroupFields; ++field) {
            StoreScalarI32(workspaceView.swigluGroupDesc + idx * kM2SwigluGroupFields + field, 0);
        }
    }
    StoreScalarI32(workspaceView.dequantSum + syncGroupCap, 0);
    for (uint32_t idx = 0; idx < gmmTaskCap; ++idx) {
        for (uint32_t field = 0; field < kM2GmmTileTaskFields; ++field) {
            StoreScalarI32(workspaceView.gmm1TileTaskPlan + idx * kM2GmmTileTaskFields + field, 0);
            StoreScalarI32(workspaceView.gmm2TileTaskPlan + idx * kM2GmmTileTaskFields + field, 0);
        }
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
        for (uint32_t field = 0; field < kM2ReturnPlanFields; ++field) {
            StoreScalarI32(workspaceView.subTileReturnPlan + idx * kM2ReturnPlanFields + field, 0);
        }
        for (uint32_t field = 0; field < kM2OwnerSegmentFields; ++field) {
            StoreScalarI32(workspaceView.subTileOwnerSegments + idx * kM2OwnerSegmentFields + field, 0);
        }
        StoreScalarI32(workspaceView.subTileReady + idx * 16U, 0);
    }
}

AICORE inline void M2CountLocalRoutes(moe_dispatch_combine_a8w8::ShapeConfig shape, M2WorkspaceViewDevice workspaceView,
                                      M2PeerWindowViewDevice localPeer, GM_ADDR expertIdx, GM_ADDR xActiveMask,
                                      uint32_t myRank)
{
    __gm__ int32_t *expertIds = reinterpret_cast<__gm__ int32_t *>(expertIdx);
    uint32_t globalExpertNum = shape.rankNum * shape.expertPerRank;
    for (uint32_t routeIndex = 0; routeIndex < shape.m * shape.topK; ++routeIndex) {
        uint32_t token = routeIndex / shape.topK;
        if (!M2TokenIsActive(xActiveMask, token)) {
            continue;
        }
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
    M2RouteRowStatTile rowMaxTile;
    M2RouteRowStatTile chunkMaxTile;
    TASSIGN(rowMaxTile, kM2RouteRowMaxTileOffset);
    TASSIGN(chunkMaxTile, kM2RouteChunkMaxTileOffset);
    TEXPANDS(rowMaxTile, 0.0f);
    pipe_barrier(PIPE_V);

    for (uint32_t colBegin = 0; colBegin < shape.hiddenSize; colBegin += kM2RouteQuantTileCols) {
        uint32_t cols = shape.hiddenSize - colBegin;
        if (cols > kM2RouteQuantTileCols) {
            cols = kM2RouteQuantTileCols;
        }
        M2RouteHalfTile halfTile(cols);
        M2RouteFloatTile fpTile(cols);
        M2RouteFloatTile absTile(cols);
        TASSIGN(halfTile, kM2RouteHalfTileOffset);
        TASSIGN(fpTile, kM2RouteFloatTileOffset);
        TASSIGN(absTile, kM2RouteAbsTileOffset);

        GlobalNd<half> src = MakeGlobal2D(input + static_cast<uint64_t>(token) * shape.hiddenSize + colBegin, 1,
                                          static_cast<int32_t>(cols), static_cast<int32_t>(shape.hiddenSize));
        TLOAD(halfTile, src);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        TCVT(fpTile, halfTile, pto::RoundMode::CAST_RINT);
        pipe_barrier(PIPE_V);
        TABS(absTile, fpTile);
        pipe_barrier(PIPE_V);
        TROWMAX(chunkMaxTile, absTile, fpTile);
        pipe_barrier(PIPE_V);
        TMAX(rowMaxTile, rowMaxTile, chunkMaxTile);
        pipe_barrier(PIPE_V);
    }

    pto::PtoSetWaitFlag<PIPE_V, PIPE_S>();
    float maxAbs = rowMaxTile.GetValue(0);
    pto::PtoSetWaitFlag<PIPE_S, PIPE_V>();
    float scale = maxAbs == 0.0f ? 1.0f : maxAbs / 127.0f;
    float invScale = maxAbs == 0.0f ? 1.0f : 1.0f / scale;

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

    __gm__ int8_t *dst = localPeer.dispatchPayload + static_cast<uint64_t>(packedRow) * rowBytes;
    for (uint32_t colBegin = 0; colBegin < shape.hiddenSize; colBegin += kM2RouteQuantTileCols) {
        uint32_t cols = shape.hiddenSize - colBegin;
        if (cols > kM2RouteQuantTileCols) {
            cols = kM2RouteQuantTileCols;
        }
        M2RouteHalfTile halfTile(cols);
        M2RouteFloatTile fpTile(cols);
        M2RouteQuantTile quantTile(cols);
        TASSIGN(halfTile, kM2RouteHalfTileOffset);
        TASSIGN(fpTile, kM2RouteFloatTileOffset);
        TASSIGN(quantTile, kM2RouteQuantTileOffset);

        GlobalNd<half> src = MakeGlobal2D(input + static_cast<uint64_t>(token) * shape.hiddenSize + colBegin, 1,
                                          static_cast<int32_t>(cols), static_cast<int32_t>(shape.hiddenSize));
        GlobalNd<int8_t> quantDst =
            MakeGlobal2D(dst + colBegin, 1, static_cast<int32_t>(cols), static_cast<int32_t>(rowBytes));
        TLOAD(halfTile, src);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        TCVT(fpTile, halfTile, pto::RoundMode::CAST_RINT);
        pipe_barrier(PIPE_V);
        pto::TQUANT<pto::QuantType::INT8_SYM>(quantTile, fpTile, invScaleTile);
        pipe_barrier(PIPE_V);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        TSTORE(quantDst, quantTile);
        WaitStoreTileReusable();
    }

    for (uint32_t colBegin = shape.hiddenSize; colBegin < rowBytes; colBegin += kM2RouteQuantTileCols) {
        uint32_t cols = rowBytes - colBegin;
        if (cols > kM2RouteQuantTileCols) {
            cols = kM2RouteQuantTileCols;
        }
        M2RouteQuantTile zeroTile(cols);
        TASSIGN(zeroTile, kM2RouteQuantTileOffset);
        TEXPANDS(zeroTile, static_cast<int8_t>(0));
        pipe_barrier(PIPE_V);
        GlobalNd<int8_t> padDst =
            MakeGlobal2D(dst + colBegin, 1, static_cast<int32_t>(cols), static_cast<int32_t>(rowBytes));
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        TSTORE(padDst, zeroTile);
        WaitStoreTileReusable();
    }
}

AICORE inline void M2RoutePackQuantLocal(moe_dispatch_combine_a8w8::ShapeConfig shape,
                                         M2WorkspaceViewDevice workspaceView, M2PeerWindowViewDevice localPeer,
                                         GM_ADDR inputA, GM_ADDR expertIdx, GM_ADDR xActiveMask, uint32_t rowBytes)
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
    int32_t localRowCap = static_cast<int32_t>(M2LocalRows(shape));
    for (uint32_t localExpert = 0; localExpert < shape.expertPerRank; ++localExpert) {
        StoreScalarI32(workspaceView.dispatchOffset + localExpert, dispatchCursor);
        int32_t before = 0;
        for (uint32_t tokenOwner = 0; tokenOwner < shape.rankNum; ++tokenOwner) {
            int32_t rows = M2LoadTokenPerExpert(shape, localPeer, tokenOwner, myRank, localExpert);
            if (dispatchCursor >= localRowCap || rows <= 0) {
                rows = 0;
            } else {
                int32_t available = localRowCap - dispatchCursor;
                if (rows > available) {
                    rows = available;
                }
            }
            StoreScalarI32(workspaceView.preSumBeforeRank + tokenOwner * shape.expertPerRank + localExpert, before);
            before += rows;
            StoreScalarI32(workspaceView.cumsumMM + tokenOwner * shape.expertPerRank + localExpert, before);
            dispatchCursor += rows;
        }
        StoreScalarI32(workspaceView.expertTokenNums + localExpert, before);
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
    int32_t producerStatus = rows > 0 ? kM2ScoreboardStatusRowsPresent : kM2ScoreboardStatusSkipDone;
    StoreScalarI32(workspaceView.producerStatus + taskId * 16U, producerStatus);
}

AICORE inline void M2PublishDispatchLedgerCopyDone(moe_dispatch_combine_a8w8::ShapeConfig shape,
                                                   M2WorkspaceViewDevice workspaceView, uint32_t tokenOwner,
                                                   uint32_t localExpert)
{
    uint32_t taskId = tokenOwner * shape.expertPerRank + localExpert;
    pipe_barrier(PIPE_ALL);
    dsb(DSB_DDR);
    StoreScalarI32(workspaceView.producerStatus + taskId * 16U, kM2ScoreboardStatusCopyDone);
}

AICORE inline void M2UpdateDispatchScoreboardDomain(moe_dispatch_combine_a8w8::ShapeConfig shape,
                                                    M2WorkspaceViewDevice workspaceView, uint32_t localExpert)
{
    int32_t rowsTotal = 0;
    int32_t activeSegments = 0;
    int32_t completedSegments = 0;
    int32_t skippedSegments = 0;
    int32_t minProducerStatus = kM2ScoreboardStatusSkipDone;
    for (uint32_t tokenOwner = 0; tokenOwner < shape.rankNum; ++tokenOwner) {
        uint32_t taskId = tokenOwner * shape.expertPerRank + localExpert;
        int32_t rows = LoadScalarI32(workspaceView.scoreboardTaskMap + taskId * 4U + 3U);
        int32_t producerStatus = LoadScalarI32(workspaceView.producerStatus + taskId * 16U);
        rowsTotal += rows > 0 ? rows : 0;
        if (rows > 0) {
            ++activeSegments;
            if (producerStatus == kM2ScoreboardStatusCopyDone) {
                ++completedSegments;
            }
            if (minProducerStatus == kM2ScoreboardStatusSkipDone || producerStatus < minProducerStatus) {
                minProducerStatus = producerStatus;
            }
        } else {
            ++skippedSegments;
        }
    }
    int32_t domainStatus = activeSegments == 0 ?
                               kM2ScoreboardStatusSkipDone :
                               (completedSegments == activeSegments ? kM2ScoreboardStatusCopyDone : minProducerStatus);
    uint32_t domainId = localExpert;
    __gm__ int32_t *minStatus = workspaceView.scoreboardMinStatus + domainId * 16U;
    __gm__ int32_t *waitPlan = workspaceView.workerWaitCounters + domainId * 16U;
    __gm__ int32_t *timeout = workspaceView.scoreboardTimeoutCounters + domainId * 16U;
    StoreScalarI32(minStatus + 0U, domainStatus);
    StoreScalarI32(minStatus + 1U, 1);
    StoreScalarI32(minStatus + 2U, static_cast<int32_t>(localExpert));
    StoreScalarI32(minStatus + 3U, static_cast<int32_t>(localExpert));
    StoreScalarI32(minStatus + 4U, static_cast<int32_t>(shape.expertPerRank));
    StoreScalarI32(minStatus + 5U, static_cast<int32_t>(shape.rankNum));
    StoreScalarI32(minStatus + 6U, rowsTotal);
    StoreScalarI32(minStatus + 7U, activeSegments);
    StoreScalarI32(minStatus + 8U, skippedSegments);
    StoreScalarI32(minStatus + 9U, completedSegments);
    StoreScalarI32(minStatus + 10U, activeSegments + skippedSegments);
    StoreScalarI32(minStatus + 11U, static_cast<int32_t>(localExpert));
    StoreScalarI32(minStatus + 12U, static_cast<int32_t>((shape.rankNum - 1U) * shape.expertPerRank + localExpert));
    StoreScalarI32(minStatus + 13U, skippedSegments);
    StoreScalarI32(minStatus + 14U, 0);
    StoreScalarI32(minStatus + 15U, 1);

    StoreScalarI32(waitPlan + 0U, 1);
    StoreScalarI32(waitPlan + 1U, static_cast<int32_t>(localExpert));
    StoreScalarI32(waitPlan + 2U, static_cast<int32_t>(shape.rankNum));
    StoreScalarI32(waitPlan + 3U, activeSegments);
    StoreScalarI32(waitPlan + 4U, completedSegments);
    StoreScalarI32(waitPlan + 5U, skippedSegments);
    StoreScalarI32(waitPlan + 6U, rowsTotal);
    StoreScalarI32(waitPlan + 7U, static_cast<int32_t>(localExpert));
    StoreScalarI32(waitPlan + 8U, static_cast<int32_t>(shape.expertPerRank));
    StoreScalarI32(waitPlan + 9U, domainStatus);
    StoreScalarI32(waitPlan + 10U, 0);
    StoreScalarI32(waitPlan + 11U, 0);
    StoreScalarI32(waitPlan + 12U, 0);
    StoreScalarI32(waitPlan + 13U, 1);
    StoreScalarI32(waitPlan + 14U, 1);
    StoreScalarI32(waitPlan + 15U, 0);

    StoreScalarI32(timeout + 0U, 0);
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
            for (uint32_t colBegin = 0; colBegin < w1Cols; colBegin += kM2EpilogueTileCols) {
                uint32_t cols = w1Cols - colBegin;
                if (cols > kM2EpilogueTileCols) {
                    cols = kM2EpilogueTileCols;
                }
                M2EpilogueAccTile accTile(cols);
                M2EpilogueFloatTile fpTile(cols);
                M2EpilogueFloatTile scaleTile(cols);
                M2EpilogueFloatTile outTile(cols);
                TASSIGN(accTile, kM2EpilogueAccTileOffset);
                TASSIGN(fpTile, kM2EpilogueFloatTileOffset);
                TASSIGN(scaleTile, kM2EpilogueScaleTileOffset);
                TASSIGN(outTile, kM2EpilogueScaledTileOffset);

                GlobalNd<int32_t> accGlobal =
                    MakeGlobal2D(workspaceView.gmm1AccInt32 + static_cast<uint64_t>(globalRow) * w1Cols + colBegin, 1,
                                 static_cast<int32_t>(cols), static_cast<int32_t>(w1Cols));
                GlobalNd<float> outGlobal =
                    MakeGlobal2D(workspaceView.gmm1Out + static_cast<uint64_t>(globalRow) * w1Cols + colBegin, 1,
                                 static_cast<int32_t>(cols), static_cast<int32_t>(w1Cols));
                for (uint32_t col = 0; col < cols; ++col) {
                    scaleTile.SetValue(col, M2DecodeUint64Scale(workspaceView.scale1Uint64[colBegin + col]));
                }
                pipe_barrier(PIPE_ALL);
                TLOAD(accTile, accGlobal);
                set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                TCVT(fpTile, accTile, pto::RoundMode::CAST_RINT);
                pipe_barrier(PIPE_V);
                TMUL(outTile, fpTile, scaleTile);
                pipe_barrier(PIPE_V);
                set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                TSTORE(outGlobal, outTile);
                WaitStoreTileReusable();
            }
        }
    }
    uint32_t syncGroupCount = static_cast<uint32_t>(LoadScalarI32(workspaceView.swigluSyncGroups));
    if (syncGroupCount == 0U || syncGroupCount > shape.expertPerRank) {
        syncGroupCount = shape.expertPerRank;
    }
    for (uint32_t syncIdx = 0; syncIdx < syncGroupCount; ++syncIdx) {
        StoreScalarI32(workspaceView.gmm1SyncGroupReady + syncIdx * 16U, 1);
    }
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

AICORE inline uint32_t M2BuildGmmTileTaskPlan(moe_dispatch_combine_a8w8::ShapeConfig shape,
                                              M2WorkspaceViewDevice workspaceView, __gm__ int32_t *taskPlan,
                                              uint32_t nCols, uint32_t kSize, uint32_t stageId)
{
    uint32_t taskId = 0;
    uint32_t maxTasks = static_cast<uint32_t>(M2GmmTileTaskCapacity(shape));
    for (uint32_t localExpert = 0; localExpert < shape.expertPerRank; ++localExpert) {
        int32_t rowBegin = LoadScalarI32(workspaceView.dispatchOffset + localExpert);
        int32_t rowCount = LoadScalarI32(workspaceView.expertTokenNums + localExpert);
        if (rowCount <= 0) {
            continue;
        }
        for (uint32_t rowOffset = 0; rowOffset < static_cast<uint32_t>(rowCount); rowOffset += kM2GmmBaseM) {
            uint32_t rows = static_cast<uint32_t>(rowCount) - rowOffset;
            if (rows > kM2GmmBaseM) {
                rows = kM2GmmBaseM;
            }
            for (uint32_t nBase = 0; nBase < nCols; nBase += kM2GmmBaseN) {
                uint32_t cols = nCols - nBase;
                if (cols > kM2GmmBaseN) {
                    cols = kM2GmmBaseN;
                }
                if (taskId < maxTasks) {
                    __gm__ int32_t *task = taskPlan + taskId * kM2GmmTileTaskFields;
                    StoreScalarI32(task + 0U, static_cast<int32_t>(taskId));
                    StoreScalarI32(task + 1U, static_cast<int32_t>(stageId));
                    StoreScalarI32(task + 2U, static_cast<int32_t>(localExpert));
                    StoreScalarI32(task + 3U, rowBegin + static_cast<int32_t>(rowOffset));
                    StoreScalarI32(task + 4U, static_cast<int32_t>(rows));
                    StoreScalarI32(task + 5U, static_cast<int32_t>(nBase));
                    StoreScalarI32(task + 6U, static_cast<int32_t>(cols));
                    StoreScalarI32(task + 7U, static_cast<int32_t>(kSize));
                }
                ++taskId;
            }
        }
    }
    return taskId;
}

AICORE inline void M2BuildSwigluSyncMetadata(moe_dispatch_combine_a8w8::ShapeConfig shape,
                                             M2WorkspaceViewDevice workspaceView)
{
    uint32_t groupCount = 0;
    uint32_t expert = 0;
    int32_t rowPrefix = 0;
    uint32_t tilePrefix = 0;
    StoreScalarI32(workspaceView.dequantSum, 0);
    while (expert < shape.expertPerRank) {
        uint32_t remaining = shape.expertPerRank - expert;
        uint32_t groupSize = M2NextSwigluGroupSize(remaining);
        int32_t rowBegin = rowPrefix;
        StoreScalarI32(workspaceView.swigluSyncGroups + groupCount + 1U, static_cast<int32_t>(groupSize));
        for (uint32_t idx = 0; idx < groupSize; ++idx) {
            rowPrefix += LoadScalarI32(workspaceView.expertTokenNums + expert + idx);
        }
        uint32_t tileBegin = tilePrefix;
        uint32_t tileEnd = tileBegin + static_cast<uint32_t>(
                                           M2CeilDivDevice(static_cast<uint64_t>(rowPrefix - rowBegin), kM2GmmBaseM));
        __gm__ int32_t *group = workspaceView.swigluGroupDesc + groupCount * kM2SwigluGroupFields;
        StoreScalarI32(group + 0U, static_cast<int32_t>(groupCount));
        StoreScalarI32(group + 1U, static_cast<int32_t>(expert));
        StoreScalarI32(group + 2U, static_cast<int32_t>(expert + groupSize));
        StoreScalarI32(group + 3U, rowBegin);
        StoreScalarI32(group + 4U, rowPrefix);
        StoreScalarI32(group + 5U, static_cast<int32_t>(tileBegin));
        StoreScalarI32(group + 6U, static_cast<int32_t>(tileEnd));
        StoreScalarI32(group + 7U, rowPrefix == rowBegin ? 1 : 0);
        StoreScalarI32(workspaceView.dequantSum + groupCount + 1U, rowPrefix);
        ++groupCount;
        tilePrefix = tileEnd;
        expert += groupSize;
    }
    StoreScalarI32(workspaceView.swigluSyncGroups, static_cast<int32_t>(groupCount));
    uint32_t gmm2Tasks = M2BuildGmmTileTaskPlan(shape, workspaceView, workspaceView.gmm2TileTaskPlan, shape.hiddenSize,
                                                shape.intermediateSize, 2U);
    StoreScalarI32(workspaceView.stageStatus + 2U * 16U, static_cast<int32_t>(groupCount));
    StoreScalarI32(workspaceView.stageStatus + 3U * 16U, static_cast<int32_t>(gmm2Tasks));
}

AICORE inline void M2ComputeSwigluRowPto(moe_dispatch_combine_a8w8::ShapeConfig shape,
                                         M2WorkspaceViewDevice workspaceView, uint32_t globalRow, float routingScale)
{
    constexpr int kSwiGluCols = 64;
    uint32_t w1Cols = shape.intermediateSize * 2U;
    for (uint32_t colBegin = 0; colBegin < shape.intermediateSize; colBegin += kSwiGluCols) {
        uint32_t cols = shape.intermediateSize - colBegin;
        if (cols > kSwiGluCols) {
            cols = kSwiGluCols;
        }
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

        GlobalNd<float> gateGlobal =
            MakeGlobal2D(workspaceView.gmm1Out + static_cast<uint64_t>(globalRow) * w1Cols + colBegin, 1,
                         static_cast<int32_t>(cols), static_cast<int32_t>(w1Cols));
        GlobalNd<float> upGlobal = MakeGlobal2D(
            workspaceView.gmm1Out + static_cast<uint64_t>(globalRow) * w1Cols + shape.intermediateSize + colBegin, 1,
            static_cast<int32_t>(cols), static_cast<int32_t>(w1Cols));
        GlobalNd<float> dstGlobal =
            MakeGlobal2D(workspaceView.swigluOut + static_cast<uint64_t>(globalRow) * shape.intermediateSize + colBegin,
                         1, static_cast<int32_t>(cols), static_cast<int32_t>(shape.intermediateSize));
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
}

AICORE inline void M2RequantizeSwigluRowPto(moe_dispatch_combine_a8w8::ShapeConfig shape,
                                            M2WorkspaceViewDevice workspaceView, uint32_t globalRow,
                                            uint32_t gmm2RowStride)
{
    M2RouteRowStatTile rowMaxTile;
    M2RouteRowStatTile chunkMaxTile;
    TASSIGN(rowMaxTile, kM2RouteRowMaxTileOffset);
    TASSIGN(chunkMaxTile, kM2RouteChunkMaxTileOffset);
    TEXPANDS(rowMaxTile, 0.0f);
    pipe_barrier(PIPE_V);

    for (uint32_t colBegin = 0; colBegin < shape.intermediateSize; colBegin += kM2RouteQuantTileCols) {
        uint32_t cols = shape.intermediateSize - colBegin;
        if (cols > kM2RouteQuantTileCols) {
            cols = kM2RouteQuantTileCols;
        }
        M2RouteFloatTile fpTile(cols);
        M2RouteFloatTile absTile(cols);
        TASSIGN(fpTile, kM2RouteFloatTileOffset);
        TASSIGN(absTile, kM2RouteAbsTileOffset);

        GlobalNd<float> src =
            MakeGlobal2D(workspaceView.swigluOut + static_cast<uint64_t>(globalRow) * shape.intermediateSize + colBegin,
                         1, static_cast<int32_t>(cols), static_cast<int32_t>(shape.intermediateSize));
        TLOAD(fpTile, src);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        TABS(absTile, fpTile);
        pipe_barrier(PIPE_V);
        TROWMAX(chunkMaxTile, absTile, fpTile);
        pipe_barrier(PIPE_V);
        TMAX(rowMaxTile, rowMaxTile, chunkMaxTile);
        pipe_barrier(PIPE_V);
    }

    pto::PtoSetWaitFlag<PIPE_V, PIPE_S>();
    float maxAbs = rowMaxTile.GetValue(0);
    pto::PtoSetWaitFlag<PIPE_S, PIPE_V>();
    float scale = maxAbs == 0.0f ? 1.0f : maxAbs / 127.0f;
    float invScale = maxAbs == 0.0f ? 1.0f : 1.0f / scale;

    M2RouteScaleStoreTile scaleStoreTile;
    M2RouteParamTile invScaleTile;
    TASSIGN(scaleStoreTile, kM2RouteScaleStoreTileOffset);
    TASSIGN(invScaleTile, kM2RouteInvScaleParamTileOffset);
    TEXPANDS(scaleStoreTile, scale);
    pipe_barrier(PIPE_V);
    TEXPANDS(invScaleTile, invScale);
    pipe_barrier(PIPE_V);

    GlobalNd<float> scaleGlobal = MakeGlobal2D(workspaceView.gmm2PerTokenScale + globalRow, 1, 1, 1);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(scaleGlobal, scaleStoreTile);
    WaitStoreTileReusable();

    __gm__ int8_t *dst = workspaceView.gmm2InputInt8 + static_cast<uint64_t>(globalRow) * gmm2RowStride;
    for (uint32_t colBegin = 0; colBegin < shape.intermediateSize; colBegin += kM2RouteQuantTileCols) {
        uint32_t cols = shape.intermediateSize - colBegin;
        if (cols > kM2RouteQuantTileCols) {
            cols = kM2RouteQuantTileCols;
        }
        M2RouteFloatTile fpTile(cols);
        M2RouteQuantTile quantTile(cols);
        TASSIGN(fpTile, kM2RouteFloatTileOffset);
        TASSIGN(quantTile, kM2RouteQuantTileOffset);

        GlobalNd<float> src =
            MakeGlobal2D(workspaceView.swigluOut + static_cast<uint64_t>(globalRow) * shape.intermediateSize + colBegin,
                         1, static_cast<int32_t>(cols), static_cast<int32_t>(shape.intermediateSize));
        GlobalNd<int8_t> quantDst =
            MakeGlobal2D(dst + colBegin, 1, static_cast<int32_t>(cols), static_cast<int32_t>(gmm2RowStride));
        TLOAD(fpTile, src);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        pto::TQUANT<pto::QuantType::INT8_SYM>(quantTile, fpTile, invScaleTile);
        pipe_barrier(PIPE_V);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        TSTORE(quantDst, quantTile);
        WaitStoreTileReusable();
    }

    for (uint32_t colBegin = shape.intermediateSize; colBegin < gmm2RowStride; colBegin += kM2RouteQuantTileCols) {
        uint32_t cols = gmm2RowStride - colBegin;
        if (cols > kM2RouteQuantTileCols) {
            cols = kM2RouteQuantTileCols;
        }
        M2RouteQuantTile zeroTile(cols);
        TASSIGN(zeroTile, kM2RouteQuantTileOffset);
        TEXPANDS(zeroTile, static_cast<int8_t>(0));
        pipe_barrier(PIPE_V);
        GlobalNd<int8_t> padDst =
            MakeGlobal2D(dst + colBegin, 1, static_cast<int32_t>(cols), static_cast<int32_t>(gmm2RowStride));
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        TSTORE(padDst, zeroTile);
        WaitStoreTileReusable();
    }
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
            continue;
        }
        for (uint32_t row = 0; row < static_cast<uint32_t>(rowCount); ++row) {
            uint32_t globalRow = static_cast<uint32_t>(rowBegin) + row;
            float routingScale = workspaceView.routingPerTokenScale[globalRow];
            M2ComputeSwigluRowPto(shape, workspaceView, globalRow, routingScale);
            InvalidateGmCacheLines(workspaceView.swigluOut + static_cast<uint64_t>(globalRow) * shape.intermediateSize,
                                   static_cast<uint32_t>(shape.intermediateSize * sizeof(float)));
            M2RequantizeSwigluRowPto(shape, workspaceView, globalRow, gmm2RowStride);
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
    (void)myRank;
}

__global__ AICORE void M2Int8Dispatch(moe_dispatch_combine_a8w8::ShapeConfig shape,
                                      moe_dispatch_combine_a8w8::RankConfig rank, GM_ADDR inputA, GM_ADDR expertIdx,
                                      GM_ADDR xActiveMask, GM_ADDR peerWindow, GM_ADDR hcclCtx, GM_ADDR workspace)
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
    M2CountLocalRoutes(shape, workspaceView, localPeer, expertIdx, xActiveMask, myRank);
    M2RoutePackQuantLocal(shape, workspaceView, localPeer, inputA, expertIdx, xActiveMask, rowBytes);
    M2PublishCountRows(shape, workspaceView, localPeer, ctx, peerWindow, myRank, peerWindowLayout);
    M2WaitCountRows(shape, localPeer);
    M2BuildPrefixMetadata(shape, workspaceView, localPeer, myRank);
    M2GatherDispatchToGmm1Input(shape, workspaceView, localPeer, ctx, peerWindow, myRank, rowBytes, peerWindowLayout);
    M2BuildSwigluSyncMetadata(shape, workspaceView);
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
        for (uint32_t colBegin = 0; colBegin < shape.hiddenSize; colBegin += kM2EpilogueTileCols) {
            uint32_t cols = shape.hiddenSize - colBegin;
            if (cols > kM2EpilogueTileCols) {
                cols = kM2EpilogueTileCols;
            }
            M2EpilogueAccTile accTile(cols);
            M2EpilogueFloatTile fpTile(cols);
            M2EpilogueFloatTile scaleTile(cols);
            M2EpilogueFloatTile scaledTile(cols);
            M2EpilogueHalfTile halfTile(cols);
            TASSIGN(accTile, kM2EpilogueAccTileOffset);
            TASSIGN(fpTile, kM2EpilogueFloatTileOffset);
            TASSIGN(scaleTile, kM2EpilogueScaleTileOffset);
            TASSIGN(scaledTile, kM2EpilogueScaledTileOffset);
            TASSIGN(halfTile, kM2EpilogueHalfTileOffset);

            GlobalNd<int32_t> accGlobal =
                MakeGlobal2D(workspaceView.gmm2AccInt32 + static_cast<uint64_t>(srcRow) * shape.hiddenSize + colBegin,
                             1, static_cast<int32_t>(cols), static_cast<int32_t>(shape.hiddenSize));
            GlobalNd<half> dstGlobal = MakeGlobal2D(dst + colBegin, 1, static_cast<int32_t>(cols), returnRowStride);
            for (uint32_t col = 0; col < cols; ++col) {
                float scale = M2DecodeUint64Scale(workspaceView.scale2Uint64[colBegin + col]) * tokenScale;
                scaleTile.SetValue(col, scale);
            }
            pipe_barrier(PIPE_ALL);
            TLOAD(accTile, accGlobal);
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            TCVT(fpTile, accTile, pto::RoundMode::CAST_RINT);
            pipe_barrier(PIPE_V);
            TMUL(scaledTile, fpTile, scaleTile);
            pipe_barrier(PIPE_V);
            TCVT(halfTile, scaledTile, pto::RoundMode::CAST_RINT);
            pipe_barrier(PIPE_V);
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            TSTORE(dstGlobal, halfTile);
            WaitStoreTileReusable();
        }
        for (int32_t col = static_cast<int32_t>(shape.hiddenSize); col < returnRowStride; ++col) {
            dst[col] = static_cast<half>(0.0);
        }
    }
    InvalidateGmCacheLines(workspaceView.gmm2Out + static_cast<int64_t>(srcStart) * returnRowStride,
                           static_cast<uint32_t>(rows * returnRowStride * sizeof(half)));
}

AICORE inline void M2MaterializeGmm2EpilogueSegment(moe_dispatch_combine_a8w8::ShapeConfig shape,
                                                    M2WorkspaceViewDevice workspaceView, int32_t srcStart, int32_t rows,
                                                    int32_t hiddenBegin, int32_t hiddenCount, int32_t returnRowStride)
{
    for (int32_t row = 0; row < rows; ++row) {
        int32_t srcRow = srcStart + row;
        float tokenScale = workspaceView.gmm2PerTokenScale[srcRow];
        M2EpilogueAccTile accTile(static_cast<uint32_t>(hiddenCount));
        M2EpilogueFloatTile fpTile(static_cast<uint32_t>(hiddenCount));
        M2EpilogueFloatTile scaleTile(static_cast<uint32_t>(hiddenCount));
        M2EpilogueFloatTile scaledTile(static_cast<uint32_t>(hiddenCount));
        M2EpilogueHalfTile halfTile(static_cast<uint32_t>(hiddenCount));
        TASSIGN(accTile, kM2EpilogueAccTileOffset);
        TASSIGN(fpTile, kM2EpilogueFloatTileOffset);
        TASSIGN(scaleTile, kM2EpilogueScaleTileOffset);
        TASSIGN(scaledTile, kM2EpilogueScaledTileOffset);
        TASSIGN(halfTile, kM2EpilogueHalfTileOffset);

        GlobalNd<int32_t> accGlobal =
            MakeGlobal2D(workspaceView.gmm2AccInt32 + static_cast<uint64_t>(srcRow) * shape.hiddenSize + hiddenBegin, 1,
                         hiddenCount, static_cast<int32_t>(shape.hiddenSize));
        GlobalNd<half> stagingGlobal = MakeGlobal2D(
            workspaceView.returnSegmentStaging + static_cast<int64_t>(row) * hiddenCount, 1, hiddenCount, hiddenCount);
        GlobalNd<half> gmm2OutGlobal =
            MakeGlobal2D(workspaceView.gmm2Out + static_cast<int64_t>(srcRow) * returnRowStride + hiddenBegin, 1,
                         hiddenCount, returnRowStride);
        pipe_barrier(PIPE_ALL);
        for (int32_t col = 0; col < hiddenCount; ++col) {
            int32_t hiddenCol = hiddenBegin + col;
            float scale = M2DecodeUint64Scale(workspaceView.scale2Uint64[hiddenCol]) * tokenScale;
            scaleTile.SetValue(static_cast<uint32_t>(col), scale);
        }
        pipe_barrier(PIPE_ALL);
        TLOAD(accTile, accGlobal);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        TCVT(fpTile, accTile, pto::RoundMode::CAST_RINT);
        pipe_barrier(PIPE_V);
        TMUL(scaledTile, fpTile, scaleTile);
        pipe_barrier(PIPE_V);
        TCVT(halfTile, scaledTile, pto::RoundMode::CAST_RINT);
        pipe_barrier(PIPE_V);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        TSTORE(stagingGlobal, halfTile);
        WaitStoreTileReusable();
        pipe_barrier(PIPE_ALL);
        TSTORE(gmm2OutGlobal, halfTile);
        WaitStoreTileReusable();
    }
    pipe_barrier(PIPE_ALL);
    dsb(DSB_DDR);
    InvalidateGmCacheLines(workspaceView.returnSegmentStaging,
                           static_cast<uint32_t>(rows * hiddenCount * sizeof(half)));
}

AICORE inline uint32_t M2BuildReturnSegmentMap(moe_dispatch_combine_a8w8::ShapeConfig shape,
                                               M2WorkspaceViewDevice workspaceView, M2PeerWindowViewDevice localPeer,
                                               uint32_t myRank)
{
    uint32_t tileRows = shape.gmmBlockM == 0 ? kM2ReturnTileRows : shape.gmmBlockM;
    uint32_t hiddenChunk = static_cast<uint32_t>(M2ReturnHiddenChunkCols(shape));
    uint32_t capacity = static_cast<uint32_t>(M2ReturnSegmentCapacity(shape));
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
            for (uint32_t hiddenBegin = 0; hiddenBegin < shape.hiddenSize; hiddenBegin += hiddenChunk) {
                uint32_t hiddenCount = shape.hiddenSize - hiddenBegin;
                if (hiddenCount > hiddenChunk) {
                    hiddenCount = hiddenChunk;
                }
                uint32_t firstSegment = segmentId;
                uint32_t tileSegmentCount = 0;
                for (uint32_t tokenOwner = 0; tokenOwner < shape.rankNum; ++tokenOwner) {
                    int32_t current =
                        LoadScalarI32(workspaceView.cumsumMM + tokenOwner * shape.expertPerRank + localExpert);
                    int32_t previous = tokenOwner == 0U ?
                                           0 :
                                           LoadScalarI32(workspaceView.cumsumMM +
                                                         (tokenOwner - 1U) * shape.expertPerRank + localExpert);
                    int32_t ownerRows = current - previous;
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
                    int32_t dstStart = M2SourceGlobalExpertRowBase(shape, localPeer, tokenOwner, globalExpert) +
                                       segmentStart - ownerStart;
                    if (segmentId < capacity) {
                        __gm__ int32_t *segment =
                            workspaceView.subTileOwnerSegments + segmentId * kM2OwnerSegmentFields;
                        StoreScalarI32(segment + 0U, static_cast<int32_t>(tokenOwner));
                        StoreScalarI32(segment + 1U, segmentStart);
                        StoreScalarI32(segment + 2U, segmentEnd - segmentStart);
                        StoreScalarI32(segment + 3U, dstStart);
                        StoreScalarI32(segment + 4U, static_cast<int32_t>(hiddenBegin));
                        StoreScalarI32(segment + 5U, static_cast<int32_t>(hiddenCount));
                        StoreScalarI32(segment + 6U, static_cast<int32_t>(localExpert));
                        StoreScalarI32(segment + 7U, static_cast<int32_t>(tileId));
                    } else {
                        ++overflow;
                    }
                    ++segmentId;
                    ++tileSegmentCount;
                }
                if (tileId < capacity) {
                    __gm__ int32_t *plan = workspaceView.subTileReturnPlan + tileId * kM2ReturnPlanFields;
                    StoreScalarI32(plan + 0U, static_cast<int32_t>(tileId));
                    StoreScalarI32(plan + 1U, static_cast<int32_t>(localExpert));
                    StoreScalarI32(plan + 2U, tileStart);
                    StoreScalarI32(plan + 3U, tileCount);
                    StoreScalarI32(plan + 4U, static_cast<int32_t>(hiddenBegin));
                    StoreScalarI32(plan + 5U, static_cast<int32_t>(hiddenCount));
                    StoreScalarI32(plan + 6U, static_cast<int32_t>(firstSegment));
                    StoreScalarI32(plan + 7U, static_cast<int32_t>(tileSegmentCount));
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
    }
    StoreScalarI32(localPeer.debugCounters + 2U * 16U, static_cast<int32_t>(tileId));
    StoreScalarI32(localPeer.debugCounters + 3U * 16U, static_cast<int32_t>(segmentId));
    StoreScalarI32(localPeer.debugCounters + 4U * 16U, static_cast<int32_t>(maxSegmentsPerTile));
    StoreScalarI32(localPeer.debugCounters + 5U * 16U, static_cast<int32_t>(overflow));
    return segmentId;
}

AICORE inline int32_t M2ExpectedReturnSegmentCount(moe_dispatch_combine_a8w8::ShapeConfig shape,
                                                   M2PeerWindowViewDevice localPeer, uint32_t myRank,
                                                   uint32_t expertOwner, uint32_t localExpert)
{
    uint32_t tileRows = shape.gmmBlockM == 0 ? kM2ReturnTileRows : shape.gmmBlockM;
    uint32_t hiddenChunks = static_cast<uint32_t>(M2CeilDivDevice(shape.hiddenSize, M2ReturnHiddenChunkCols(shape)));
    int32_t ownerRows = M2EffectiveTokenOwnerRows(shape, localPeer, myRank, expertOwner, localExpert);
    if (ownerRows <= 0) {
        return 0;
    }
    int32_t rowBegin = 0;
    for (uint32_t prevLocalExpert = 0; prevLocalExpert < localExpert; ++prevLocalExpert) {
        for (uint32_t tokenOwner = 0; tokenOwner < shape.rankNum; ++tokenOwner) {
            rowBegin += M2EffectiveTokenOwnerRows(shape, localPeer, tokenOwner, expertOwner, prevLocalExpert);
        }
    }
    int32_t ownerPrefix = 0;
    for (uint32_t tokenOwner = 0; tokenOwner < myRank; ++tokenOwner) {
        ownerPrefix += M2EffectiveTokenOwnerRows(shape, localPeer, tokenOwner, expertOwner, localExpert);
    }
    int32_t rowCount = 0;
    for (uint32_t tokenOwner = 0; tokenOwner < shape.rankNum; ++tokenOwner) {
        rowCount += M2EffectiveTokenOwnerRows(shape, localPeer, tokenOwner, expertOwner, localExpert);
    }
    int32_t ownerStart = rowBegin + ownerPrefix;
    int32_t ownerEnd = ownerStart + ownerRows;
    int32_t expected = 0;
    for (int32_t tileOffset = 0; tileOffset < rowCount; tileOffset += static_cast<int32_t>(tileRows)) {
        int32_t tileStart = rowBegin + tileOffset;
        int32_t tileCount = rowCount - tileOffset;
        if (tileCount > static_cast<int32_t>(tileRows)) {
            tileCount = static_cast<int32_t>(tileRows);
        }
        int32_t tileEnd = tileStart + tileCount;
        int32_t segmentStart = tileStart > ownerStart ? tileStart : ownerStart;
        int32_t segmentEnd = tileEnd < ownerEnd ? tileEnd : ownerEnd;
        if (segmentEnd > segmentStart) {
            expected += static_cast<int32_t>(hiddenChunks);
        }
    }
    return expected;
}

AICORE inline int32_t M2WaitReturnSegmentCounters(moe_dispatch_combine_a8w8::ShapeConfig shape,
                                                  M2PeerWindowViewDevice localPeer, uint32_t myRank)
{
    int32_t expectedTotal = 0;
    for (uint32_t expertOwner = 0; expertOwner < shape.rankNum; ++expertOwner) {
        for (uint32_t localExpert = 0; localExpert < shape.expertPerRank; ++localExpert) {
            int32_t expected = M2ExpectedReturnSegmentCount(shape, localPeer, myRank, expertOwner, localExpert);
            if (expected <= 0) {
                continue;
            }
            WaitSignal(localPeer.returnSegmentCounters +
                           (static_cast<uint64_t>(expertOwner) * shape.expertPerRank + localExpert) * 16U,
                       expected);
            expectedTotal += expected;
        }
    }
    return expectedTotal;
}

AICORE inline int32_t LoadHalfBitsI32(__gm__ half *ptr)
{
    return static_cast<int32_t>(*reinterpret_cast<__gm__ uint16_t *>(ptr));
}

AICORE inline void M2RecordReturnSendTrace(moe_dispatch_combine_a8w8::ShapeConfig shape,
                                           M2WorkspaceViewDevice workspaceView, M2PeerWindowViewDevice localPeer,
                                           uint32_t tokenOwner, uint32_t localExpert, uint32_t segmentId,
                                           uint32_t tileId, int32_t srcStart, int32_t rows, int32_t dstStart,
                                           int32_t hiddenBegin, int32_t hiddenCount)
{
    constexpr uint32_t kTraceBase = 8U * 16U;
    uint32_t traceSlot = kTraceBase + (tokenOwner * shape.expertPerRank + localExpert) * 16U;
    __gm__ int32_t *trace = localPeer.debugCounters + traceSlot;
    int32_t count = LoadScalarI32(trace);
    int32_t firstBits = (rows > 0 && hiddenCount > 0) ? LoadHalfBitsI32(workspaceView.returnSegmentStaging) : 0;
    if (count == 0) {
        StoreScalarI32(trace + 1U, static_cast<int32_t>(segmentId));
        StoreScalarI32(trace + 2U, static_cast<int32_t>(tileId));
        StoreScalarI32(trace + 3U, srcStart);
        StoreScalarI32(trace + 4U, rows);
        StoreScalarI32(trace + 5U, dstStart);
        StoreScalarI32(trace + 6U, hiddenBegin);
        StoreScalarI32(trace + 7U, hiddenCount);
        StoreScalarI32(trace + 8U, firstBits);
    }
    StoreScalarI32(trace + 9U, static_cast<int32_t>(segmentId));
    StoreScalarI32(trace + 10U, srcStart);
    StoreScalarI32(trace + 11U, rows);
    StoreScalarI32(trace + 12U, dstStart);
    StoreScalarI32(trace + 13U, hiddenBegin);
    StoreScalarI32(trace + 14U, hiddenCount);
    StoreScalarI32(trace + 15U, firstBits);
    StoreScalarI32(trace, count + 1);
}

AICORE inline void M2RunGmm2EpilogueAndReturn(moe_dispatch_combine_a8w8::ShapeConfig shape,
                                              M2WorkspaceViewDevice workspaceView, M2PeerWindowViewDevice localPeer,
                                              __gm__ HcclDeviceContext *ctx, GM_ADDR peerWindow, uint32_t myRank,
                                              const moe_dispatch_combine_a8w8::PeerWindowLayout &peerWindowLayout)
{
    int32_t returnRowStride = static_cast<int32_t>(peerWindowLayout.returnPayloadRowBytes / sizeof(half));
    uint32_t mappedSegments = M2BuildReturnSegmentMap(shape, workspaceView, localPeer, myRank);
    uint32_t tileCount = static_cast<uint32_t>(LoadScalarI32(localPeer.debugCounters + 2U * 16U));
    int32_t segmentCount = 0;
    for (uint32_t tileId = 0; tileId < tileCount; ++tileId) {
        __gm__ int32_t *plan = workspaceView.subTileReturnPlan + tileId * kM2ReturnPlanFields;
        uint32_t firstSegment = static_cast<uint32_t>(LoadScalarI32(plan + 6U));
        uint32_t tileSegmentCount = static_cast<uint32_t>(LoadScalarI32(plan + 7U));
        for (uint32_t localSegment = 0; localSegment < tileSegmentCount; ++localSegment) {
            uint32_t segmentId = firstSegment + localSegment;
            if (segmentId >= mappedSegments) {
                continue;
            }
            __gm__ int32_t *segment = workspaceView.subTileOwnerSegments + segmentId * kM2OwnerSegmentFields;
            uint32_t tokenOwner = static_cast<uint32_t>(LoadScalarI32(segment + 0U));
            int32_t srcStart = LoadScalarI32(segment + 1U);
            int32_t rows = LoadScalarI32(segment + 2U);
            int32_t dstStart = LoadScalarI32(segment + 3U);
            int32_t hiddenBegin = LoadScalarI32(segment + 4U);
            int32_t hiddenCount = LoadScalarI32(segment + 5U);
            uint32_t localExpert = static_cast<uint32_t>(LoadScalarI32(segment + 6U));
            if (rows <= 0 || hiddenCount <= 0 || tokenOwner >= shape.rankNum) {
                continue;
            }
            M2MaterializeGmm2EpilogueSegment(shape, workspaceView, srcStart, rows, hiddenBegin, hiddenCount,
                                             returnRowStride);
            M2RecordReturnSendTrace(shape, workspaceView, localPeer, tokenOwner, localExpert, segmentId, tileId,
                                    srcStart, rows, dstStart, hiddenBegin, hiddenCount);
            M2PeerWindowViewDevice remotePeer =
                MakeM2RemotePeerWindowViewDevice(ctx, peerWindow, tokenOwner, peerWindowLayout);
            if (tokenOwner == myRank) {
                for (int32_t row = 0; row < rows; ++row) {
                    CopyRowHalf(localPeer.returnPayload + hiddenBegin, returnRowStride, dstStart + row,
                                workspaceView.returnSegmentStaging, hiddenCount, row, hiddenCount);
                }
            } else {
                TPutRowsHalfContiguous(remotePeer.returnPayload + hiddenBegin, returnRowStride, dstStart,
                                       workspaceView.returnSegmentStaging, hiddenCount, 0, rows, hiddenCount);
            }
            NotifySignal(remotePeer.returnSegmentCounters +
                             (static_cast<uint64_t>(myRank) * shape.expertPerRank + localExpert) * 16U,
                         1);
            ++segmentCount;
        }
    }
    for (uint32_t tokenOwner = 0; tokenOwner < shape.rankNum; ++tokenOwner) {
        M2PeerWindowViewDevice remotePeer =
            MakeM2RemotePeerWindowViewDevice(ctx, peerWindow, tokenOwner, peerWindowLayout);
        NotifySignal(remotePeer.combineDoneSignal + myRank * 16U, 1);
    }
    for (uint32_t peer = 0; peer < shape.rankNum; ++peer) {
        WaitSignal(localPeer.combineDoneSignal + peer * 16U, 1);
    }
    int32_t expectedReturnSegments = M2WaitReturnSegmentCounters(shape, localPeer, myRank);
    InvalidateGmCacheLines(localPeer.returnPayload,
                           static_cast<uint32_t>(shape.m * shape.topK * returnRowStride * sizeof(half)));
    StoreScalarI32(localPeer.debugCounters, segmentCount);
    StoreScalarI32(localPeer.debugCounters + 16U, 1);
    StoreScalarI32(localPeer.debugCounters + 6U * 16U, expectedReturnSegments);
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
        TLOAD(ptrTile, ptrGlobal);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        TLOAD(outTile, outGlobal);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        TAXPY(outTile, ptrTile, prob);
        pipe_barrier(PIPE_V);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        TSTORE(outGlobal, outTile);
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
            if (ptrDRow < 0 || static_cast<uint32_t>(ptrDRow) >= shape.maxOutputSize) {
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
                GlobalNd<half> outGlobal = MakeGlobal2D(output + static_cast<int64_t>(token) * outputRowStride + col, 1,
                                                        cols, outputRowStride);
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
    if (blockId == 0) {
        StoreScalarI32(localPeer.debugCounters + 6U * 16U, 1);
    }
    (void)rank;
}

} // namespace

#ifndef M2_FUSED_INCLUDE_DEVICE_BODY_ONLY
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
                          uint8_t *inputA, uint8_t *expertIdx, uint8_t *xActiveMask, uint8_t *peerWindow,
                          uint8_t *hcclCtx, uint8_t *workspace, void *stream, uint32_t launchBlockCount)
{
    M2Int8Dispatch<<<launchBlockCount, nullptr, stream>>>(shape, rank, inputA, expertIdx, xActiveMask, peerWindow,
                                                          hcclCtx, workspace);
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
#endif // M2_FUSED_INCLUDE_DEVICE_BODY_ONLY
