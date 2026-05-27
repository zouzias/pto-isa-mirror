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

#include "common.h"
#include "kernel_launchers.h"

using moe_dispatch::HcclDeviceContext;
using moe_dispatch::MoeDispatchShape;
using moe_dispatch::PeerWindowLayout;
using moe_dispatch::WorkspaceLayout;

#ifndef GM_ADDR
#define GM_ADDR __gm__ uint8_t *
#endif

// Kernel source contract for later tasks:
//   - Only PTO and C/C++ headers are allowed in this file.
//   - Device payload movement will use pto::GlobalTensor and pto::Tile views.
//   - Cross-rank movement/readiness will use PTO comm primitives.
//   - There are exactly two public device kernel names in this project.

namespace {

constexpr int kDefaultTileCols = 7168;
constexpr int kMetaTileCols = 16;
constexpr uint32_t kI32PerCacheLine = 16;
constexpr uint64_t kPingUbAddr = 0x0;
constexpr uint64_t kPongUbAddr = 0x4000;
constexpr uint64_t kSoftSyncUbAddr = 0x8000;

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
    __gm__ half *dispatchedA;
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

AICORE inline WorkspaceLayout MakeWorkspaceLayout(MoeDispatchShape shape)
{
    const uint64_t i32 = 4;
    const uint64_t f16 = 2;
    uint64_t expertNumPadded =
        ((static_cast<uint64_t>(shape.expertNum) + shape.metadataPad - 1) / shape.metadataPad) * shape.metadataPad;
    uint64_t aivBlocks = shape.aivBlocks == 0 ? 1 : shape.aivBlocks;
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
    layout.dispatchedA = AppendFieldDevice(offset, static_cast<uint64_t>(shape.maxOutputSize) * shape.k * f16);
    layout.totalBytes = Align64Device(offset);
    return layout;
}

AICORE inline PeerWindowLayout MakePeerWindowLayout(MoeDispatchShape shape)
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
    view.dispatchedA = reinterpret_cast<__gm__ half *>(workspaceBase + layout.dispatchedA);
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

AICORE inline uint32_t ExpertNumPaddedDevice(MoeDispatchShape shape)
{
    return ((shape.expertNum + shape.metadataPad - 1) / shape.metadataPad) * shape.metadataPad;
}

AICORE inline uint32_t ShardBegin(uint32_t totalItems, uint32_t blockId, uint32_t blockNum)
{
    uint32_t base = totalItems / blockNum;
    uint32_t rem = totalItems % blockNum;
    return blockId * base + (blockId < rem ? blockId : rem);
}

AICORE inline uint32_t ShardEnd(uint32_t totalItems, uint32_t blockId, uint32_t blockNum)
{
    return ShardBegin(totalItems, blockId + 1, blockNum);
}

AICORE inline uint32_t RouteCount(MoeDispatchShape shape)
{
    return shape.m * shape.topK;
}

AICORE inline uint32_t RouteLineCount(MoeDispatchShape shape)
{
    return (RouteCount(shape) + kI32PerCacheLine - 1) / kI32PerCacheLine;
}

AICORE inline uint32_t RouteShardBegin(MoeDispatchShape shape, uint32_t blockId, uint32_t blockNum)
{
    return ShardBegin(RouteLineCount(shape), blockId, blockNum) * kI32PerCacheLine;
}

AICORE inline uint32_t RouteShardEnd(MoeDispatchShape shape, uint32_t blockId, uint32_t blockNum)
{
    uint32_t routeEnd = ShardEnd(RouteLineCount(shape), blockId, blockNum) * kI32PerCacheLine;
    uint32_t routeTotal = RouteCount(shape);
    return routeEnd < routeTotal ? routeEnd : routeTotal;
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

AICORE inline void ClearDispatchState(MoeDispatchShape shape, LocalWorkspaceView workspaceView,
                                      LocalPeerWindowView localPeer, uint32_t blockId, uint32_t blockNum)
{
    uint32_t expertNumPadded = ExpertNumPaddedDevice(shape);
    uint32_t routeBegin = RouteShardBegin(shape, blockId, blockNum);
    uint32_t routeEnd = RouteShardEnd(shape, blockId, blockNum);
    for (uint32_t idx = blockId; idx < expertNumPadded; idx += blockNum) {
        *(workspaceView.localTokenPerExpert + idx) = 0;
    }
    for (uint32_t idx = blockId; idx < blockNum * expertNumPadded; idx += blockNum) {
        *(workspaceView.blockTokenPerExpert + idx) = 0;
        *(workspaceView.blockPrefixPerExpert + idx) = 0;
    }
    for (uint32_t idx = blockId; idx < shape.ep * expertNumPadded; idx += blockNum) {
        *(workspaceView.cumsumPerExpert + idx) = 0;
    }
    for (uint32_t idx = blockId; idx < shape.expertPerRank; idx += blockNum) {
        *(workspaceView.dispatchOffset + idx) = 0;
    }
    for (uint32_t idx = blockId; idx < shape.ep * shape.expertPerRank; idx += blockNum) {
        *(workspaceView.prevSumBeforeRank + idx) = 0;
    }
    for (uint32_t idx = routeBegin; idx < routeEnd; ++idx) {
        *(localPeer.expandedRowIdx + idx) = -1;
    }
    if (routeEnd > routeBegin) {
        InvalidateGmCacheLines(localPeer.expandedRowIdx + routeBegin,
                               static_cast<uint32_t>((routeEnd - routeBegin) * sizeof(int32_t)));
    }
}

AICORE inline int32_t PackedExpertOffset(MoeDispatchShape shape, LocalWorkspaceView workspaceView, uint32_t myRank,
                                         uint32_t expert)
{
    uint32_t expertNumPadded = ExpertNumPaddedDevice(shape);
    return *(workspaceView.cumsumPerExpert + static_cast<uint64_t>(myRank) * expertNumPadded + expert);
}

AICORE inline void PackLocalRowsToWindow(MoeDispatchShape shape, LocalWorkspaceView workspaceView,
                                         LocalPeerWindowView localPeer, GM_ADDR inputA, GM_ADDR expertIdx,
                                         uint32_t myRank, uint32_t blockId, uint32_t blockNum)
{
    uint32_t expertNumPadded = ExpertNumPaddedDevice(shape);
    __gm__ int32_t *cursorBase = PackCursorBase(workspaceView, blockNum) + blockId * expertNumPadded;
    __gm__ int32_t *expertIds = reinterpret_cast<__gm__ int32_t *>(expertIdx);
    __gm__ half *input = reinterpret_cast<__gm__ half *>(inputA);
    __gm__ int32_t *blockPrefix = workspaceView.blockPrefixPerExpert + static_cast<uint64_t>(blockId) * expertNumPadded;
    InvalidateGmCacheLines(blockPrefix, static_cast<uint32_t>(expertNumPadded * sizeof(int32_t)));
    InvalidateGmCacheLines(workspaceView.cumsumPerExpert + static_cast<uint64_t>(myRank) * expertNumPadded,
                           static_cast<uint32_t>(expertNumPadded * sizeof(int32_t)));
    for (uint32_t expert = 0; expert < expertNumPadded; ++expert) {
        *(cursorBase + expert) = *(blockPrefix + expert);
    }
    uint32_t routeBegin = RouteShardBegin(shape, blockId, blockNum);
    uint32_t routeEnd = RouteShardEnd(shape, blockId, blockNum);
    for (uint32_t routeIndex = routeBegin; routeIndex < routeEnd; ++routeIndex) {
        int32_t expert = *(expertIds + routeIndex);
        if (expert < 0 || static_cast<uint32_t>(expert) >= shape.expertNum) {
            continue;
        }
        uint32_t expertId = static_cast<uint32_t>(expert);
        uint32_t token = routeIndex / shape.topK;
        int32_t cursor = *(cursorBase + expertId);
        *(cursorBase + expertId) = cursor + 1;
        int32_t packedRow = PackedExpertOffset(shape, workspaceView, myRank, expertId) + cursor;
        *(localPeer.expandedRowIdx + routeIndex) = packedRow;
        GlobalNd<half> src = MakeGlobal2D(input + static_cast<int64_t>(token) * shape.k, 1,
                                          static_cast<int32_t>(shape.k), static_cast<int32_t>(shape.k));
        GlobalNd<half> dst = MakeGlobal2D(localPeer.packedA + static_cast<int64_t>(packedRow) * shape.k, 1,
                                          static_cast<int32_t>(shape.k), static_cast<int32_t>(shape.k));
        int32_t payloadTileCols =
            static_cast<int32_t>(shape.k) < kDefaultTileCols ? static_cast<int32_t>(shape.k) : kDefaultTileCols;
        VecTile<half> ping(1, payloadTileCols);
        VecTile<half> pong(1, payloadTileCols);
        TASSIGN(ping, kPingUbAddr);
        TASSIGN(pong, kPongUbAddr);
        pto::comm::TPUT(dst, src, ping, pong);
    }
    if (routeEnd > routeBegin) {
        InvalidateGmCacheLines(localPeer.expandedRowIdx + routeBegin,
                               static_cast<uint32_t>((routeEnd - routeBegin) * sizeof(int32_t)));
    }
}

AICORE inline void CountLocalRoutes(MoeDispatchShape shape, LocalWorkspaceView workspaceView, GM_ADDR expertIdx,
                                    uint32_t blockId, uint32_t blockNum)
{
    uint32_t expertNumPadded = ExpertNumPaddedDevice(shape);
    __gm__ int32_t *expertIds = reinterpret_cast<__gm__ int32_t *>(expertIdx);
    __gm__ int32_t *blockCounts = workspaceView.blockTokenPerExpert + blockId * expertNumPadded;
    uint32_t routeBegin = RouteShardBegin(shape, blockId, blockNum);
    uint32_t routeEnd = RouteShardEnd(shape, blockId, blockNum);
    for (uint32_t routeIndex = routeBegin; routeIndex < routeEnd; ++routeIndex) {
        int32_t expert = *(expertIds + routeIndex);
        if (expert < 0 || static_cast<uint32_t>(expert) >= shape.expertNum) {
            continue;
        }
        __gm__ int32_t *count = blockCounts + static_cast<uint32_t>(expert);
        *count = *count + 1;
    }
    InvalidateGmCacheLines(blockCounts, static_cast<uint32_t>(expertNumPadded * sizeof(int32_t)));
}

AICORE inline void BuildLocalPackMetadata(MoeDispatchShape shape, LocalWorkspaceView workspaceView, uint32_t myRank,
                                          uint32_t blockId, uint32_t blockNum)
{
    if (blockId != 0) {
        return;
    }
    uint32_t expertNumPadded = ExpertNumPaddedDevice(shape);
    InvalidateGmCacheLines(workspaceView.blockTokenPerExpert,
                           static_cast<uint32_t>(blockNum * expertNumPadded * sizeof(int32_t)));
    int32_t packedOffset = 0;
    __gm__ int32_t *localPackedOffset = workspaceView.cumsumPerExpert + static_cast<uint64_t>(myRank) * expertNumPadded;
    for (uint32_t expert = 0; expert < expertNumPadded; ++expert) {
        int32_t localCount = 0;
        for (uint32_t block = 0; block < blockNum; ++block) {
            uint32_t idx = block * expertNumPadded + expert;
            *(workspaceView.blockPrefixPerExpert + idx) = localCount;
            if (expert < shape.expertNum) {
                localCount += *(workspaceView.blockTokenPerExpert + idx);
            }
        }
        *(workspaceView.localTokenPerExpert + expert) = expert < shape.expertNum ? localCount : 0;
        *(localPackedOffset + expert) = packedOffset;
        if (expert < shape.expertNum) {
            packedOffset += localCount;
        }
    }
    InvalidateGmCacheLines(workspaceView.blockPrefixPerExpert,
                           static_cast<uint32_t>(blockNum * expertNumPadded * sizeof(int32_t)));
    InvalidateGmCacheLines(workspaceView.localTokenPerExpert, static_cast<uint32_t>(expertNumPadded * sizeof(int32_t)));
    InvalidateGmCacheLines(workspaceView.cumsumPerExpert + static_cast<uint64_t>(myRank) * expertNumPadded,
                           static_cast<uint32_t>(expertNumPadded * sizeof(int32_t)));
}

AICORE inline void PublishCountRows(MoeDispatchShape shape, LocalWorkspaceView workspaceView,
                                    __gm__ HcclDeviceContext *ctx, GM_ADDR peerWindow, uint32_t myRank,
                                    uint32_t blockId, uint32_t blockNum, const PeerWindowLayout &peerWindowLayout)
{
    uint32_t expertNumPadded = ExpertNumPaddedDevice(shape);
    InvalidateGmCacheLines(workspaceView.localTokenPerExpert, static_cast<uint32_t>(expertNumPadded * sizeof(int32_t)));
    for (uint32_t dst = blockId; dst < shape.ep; dst += blockNum) {
        LocalPeerWindowView remotePeer = MakeRemotePeerWindowView(ctx, peerWindow, dst, peerWindowLayout);
        GlobalNd<int32_t> localCount =
            MakeGlobal2D(workspaceView.localTokenPerExpert, 1, static_cast<int32_t>(expertNumPadded),
                         static_cast<int32_t>(expertNumPadded));
        GlobalNd<int32_t> remoteCount =
            MakeGlobal2D(remotePeer.peerTokenPerExpert + static_cast<uint64_t>(myRank) * expertNumPadded, 1,
                         static_cast<int32_t>(expertNumPadded), static_cast<int32_t>(expertNumPadded));
        VecTile<int32_t, kMetaTileCols> ping(1, kMetaTileCols);
        VecTile<int32_t, kMetaTileCols> pong(1, kMetaTileCols);
        TASSIGN(ping, kPingUbAddr);
        TASSIGN(pong, kPongUbAddr);
        pto::comm::TPUT(remoteCount, localCount, ping, pong);
        pipe_barrier(PIPE_ALL);
        dsb(DSB_DDR);
        pto::comm::Signal signal = MakeSignal(remotePeer.countReadySignal + myRank);
        pto::comm::TNOTIFY(signal, 1, pto::comm::NotifyOp::AtomicAdd);
    }
}

AICORE inline void WaitCountRows(MoeDispatchShape shape, LocalPeerWindowView localPeer, uint32_t blockId,
                                 uint32_t blockNum)
{
    for (uint32_t src = blockId; src < shape.ep; src += blockNum) {
        pto::comm::Signal signal = MakeSignal(localPeer.countReadySignal + src);
        pto::comm::TWAIT(signal, static_cast<int32_t>(shape.signalValue), pto::comm::WaitCmp::GE);
    }
}

AICORE inline void BuildPrefixMetadata(MoeDispatchShape shape, LocalWorkspaceView workspaceView,
                                       LocalPeerWindowView localPeer, uint32_t myRank, uint32_t blockId,
                                       uint32_t blockNum)
{
    uint32_t expertNumPadded = ExpertNumPaddedDevice(shape);
    InvalidateGmCacheLines(localPeer.peerTokenPerExpert,
                           static_cast<uint32_t>(shape.ep * expertNumPadded * sizeof(int32_t)));
    for (uint32_t src = blockId; src < shape.ep; src += blockNum) {
        int32_t sum = 0;
        for (uint32_t expert = 0; expert < expertNumPadded; ++expert) {
            sum += *(localPeer.peerTokenPerExpert + static_cast<uint64_t>(src) * expertNumPadded + expert);
            *(workspaceView.cumsumPerExpert + static_cast<uint64_t>(src) * expertNumPadded + expert) = sum;
        }
        InvalidateGmCacheLines(workspaceView.cumsumPerExpert + static_cast<uint64_t>(src) * expertNumPadded,
                               static_cast<uint32_t>(expertNumPadded * sizeof(int32_t)));
    }

    if (blockId == 0) {
        int32_t dispatchCursor = 0;
        for (uint32_t localExpert = 0; localExpert < shape.expertPerRank; ++localExpert) {
            uint32_t globalExpert = myRank * shape.expertPerRank + localExpert;
            *(workspaceView.dispatchOffset + localExpert) = dispatchCursor;
            int32_t beforeRank = 0;
            for (uint32_t src = 0; src < shape.ep; ++src) {
                *(workspaceView.prevSumBeforeRank + static_cast<uint64_t>(src) * shape.expertPerRank + localExpert) =
                    beforeRank;
                int32_t rows =
                    *(localPeer.peerTokenPerExpert + static_cast<uint64_t>(src) * expertNumPadded + globalExpert);
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

AICORE inline void GatherLocalExpertPayload(MoeDispatchShape shape, LocalWorkspaceView workspaceView,
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
            int32_t rows =
                *(localPeer.peerTokenPerExpert + static_cast<uint64_t>(src) * expertNumPadded + globalExpert);
            if (rows <= 0) {
                continue;
            }
            int32_t srcStart =
                globalExpert == 0 ?
                    0 :
                    *(workspaceView.cumsumPerExpert + static_cast<uint64_t>(src) * expertNumPadded + globalExpert - 1);
            int32_t dstStart =
                *(workspaceView.dispatchOffset + localExpert) +
                *(workspaceView.prevSumBeforeRank + static_cast<uint64_t>(src) * shape.expertPerRank + localExpert);
            LocalPeerWindowView remotePeer = MakeRemotePeerWindowView(ctx, peerWindow, src, peerWindowLayout);
            GlobalNd<half> dstGlobal =
                MakeGlobal2D(workspaceView.dispatchedA + static_cast<int64_t>(dstStart) * shape.k, rows,
                             static_cast<int32_t>(shape.k), static_cast<int32_t>(shape.k));
            GlobalNd<half> remoteSrcGlobal =
                MakeGlobal2D(remotePeer.packedA + static_cast<int64_t>(srcStart) * shape.k, rows,
                             static_cast<int32_t>(shape.k), static_cast<int32_t>(shape.k));
            int32_t payloadTileCols =
                static_cast<int32_t>(shape.k) < kDefaultTileCols ? static_cast<int32_t>(shape.k) : kDefaultTileCols;
            VecTile<half> ping(1, payloadTileCols);
            VecTile<half> pong(1, payloadTileCols);
            TASSIGN(ping, kPingUbAddr);
            TASSIGN(pong, kPongUbAddr);
            pto::comm::TGET(dstGlobal, remoteSrcGlobal, ping, pong);
        }
    }
}

} // namespace

__global__ AICORE void MoeDispatchKernel(MoeDispatchShape shape, uint32_t myRank, GM_ADDR inputA, GM_ADDR expertIdx,
                                         GM_ADDR peerWindow, GM_ADDR hcclCtx, GM_ADDR workspace)
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
    CountLocalRoutes(shape, workspaceView, expertIdx, blockId, blockNum);
    SoftSyncAiv(workspaceView.localSync, blockNum);
    BuildLocalPackMetadata(shape, workspaceView, myRank, blockId, blockNum);
    SoftSyncAiv(workspaceView.localSync, blockNum);
    PackLocalRowsToWindow(shape, workspaceView, localPeer, inputA, expertIdx, myRank, blockId, blockNum);
    SoftSyncAiv(workspaceView.localSync, blockNum);
    PublishCountRows(shape, workspaceView, ctx, peerWindow, myRank, blockId, blockNum, peerWindowLayout);
    WaitCountRows(shape, localPeer, blockId, blockNum);
    SoftSyncAiv(workspaceView.localSync, blockNum);
    BuildPrefixMetadata(shape, workspaceView, localPeer, myRank, blockId, blockNum);
    SoftSyncAiv(workspaceView.localSync, blockNum);
    GatherLocalExpertPayload(shape, workspaceView, localPeer, ctx, peerWindow, myRank, blockId, blockNum,
                             peerWindowLayout);
    SoftSyncAiv(workspaceView.localSync, blockNum);
}

namespace moe_dispatch {

void LaunchMoeDispatchKernel(MoeDispatchShape shape, uint32_t myRank, uint8_t *inputA, uint8_t *expertIdx,
                             uint8_t *peerWindow, uint8_t *hcclCtx, uint8_t *workspace, void *stream,
                             uint32_t launchBlockCount)
{
    MoeDispatchKernel<<<launchBlockCount, nullptr, stream>>>(shape, myRank, inputA, expertIdx, peerWindow, hcclCtx,
                                                             workspace);
}

} // namespace moe_dispatch
