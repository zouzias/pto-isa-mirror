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
    pto::comm::Signal sig = MakeSignal(signal);
    pto::comm::TNOTIFY(sig, value, pto::comm::NotifyOp::Set);
}

AICORE inline void WaitSignal(__gm__ int32_t *signal, int32_t value)
{
    pto::comm::Signal sig = MakeSignal(signal);
    pto::comm::TWAIT(sig, value, pto::comm::WaitCmp::GE);
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
    for (uint32_t idx = blockId; idx < shape.ep; idx += blockNum) {
        StoreScalarI32(localPeer.combineDoneSignal + idx, 0);
    }
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

} // namespace dispatch_combine_tile
