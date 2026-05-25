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
constexpr uint64_t kSoftSyncUbAddr = 0x3000;

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
        set_flag(PIPE_MTE3, PIPE_MTE2, event);
        wait_flag(PIPE_MTE3, PIPE_MTE2, event);
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

AICORE inline void SoftSyncAiv(__gm__ int32_t *gmWorkspace, uint32_t blockNum)
{
    pto::Tile<pto::TileType::Vec, int32_t, 1, pto::SYNCALL_SOFT_SLOT_INT32, pto::BLayout::RowMajor, -1, -1>
        syncTile(1, pto::SYNCALL_SOFT_SLOT_INT32);
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
    for (uint32_t idx = blockId; idx < expandedRows; idx += blockNum) {
        StoreScalarI32(localPeer.expandedRowIdx + idx, -1);
    }
    for (uint32_t idx = blockId; idx < shape.ep; idx += blockNum) {
        StoreScalarI32(localPeer.combineDoneSignal + idx, 0);
    }
}

AICORE inline void InitPackCursors(DispatchCombineTileShape shape, LocalWorkspaceView workspaceView, uint32_t blockId,
                                   uint32_t blockNum)
{
    uint32_t expertNumPadded = ExpertNumPaddedDevice(shape);
    __gm__ int32_t *cursorBase = PackCursorBase(workspaceView, blockNum) + blockId * expertNumPadded;
    for (uint32_t expert = 0; expert < expertNumPadded; ++expert) {
        int32_t prefix =
            LoadScalarI32(workspaceView.blockPrefixPerExpert + static_cast<uint64_t>(blockId) * expertNumPadded +
                          expert);
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
    uint32_t tokenBegin = TokenShardBegin(shape.m, blockId, blockNum);
    uint32_t tokenEnd = TokenShardEnd(shape.m, blockId, blockNum);
    for (uint32_t token = tokenBegin; token < tokenEnd; ++token) {
        for (uint32_t slot = 0; slot < shape.topK; ++slot) {
            uint32_t routeIndex = token * shape.topK + slot;
            int32_t expert = LoadScalarI32(expertIds + routeIndex);
            if (expert < 0 || static_cast<uint32_t>(expert) >= shape.expertNum) {
                StoreScalarI32(localPeer.expandedRowIdx + routeIndex, -1);
                continue;
            }
            uint32_t expertId = static_cast<uint32_t>(expert);
            int32_t cursor = LoadScalarI32(cursorBase + expertId);
            StoreScalarI32(cursorBase + expertId, cursor + 1);
            int32_t packedRow = PackedExpertOffset(shape, workspaceView, myRank, expertId) + cursor;
            StoreScalarI32(localPeer.expandedRowIdx + routeIndex, packedRow);
            CopyRowHalf(localPeer.packedA, static_cast<int32_t>(shape.k), packedRow, input, static_cast<int32_t>(shape.k),
                        static_cast<int32_t>(token), static_cast<int32_t>(shape.k));
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
                StoreScalarI32(localPeer.expandedRowIdx + routeIndex, -1);
                continue;
            }
            __gm__ int32_t *count = blockCounts + static_cast<uint32_t>(expert);
            StoreScalarI32(count, LoadScalarI32(count) + 1);
        }
    }
}

AICORE inline void BuildBlockPrefixAndLocalCounts(DispatchCombineTileShape shape, LocalWorkspaceView workspaceView,
                                                  uint32_t blockId, uint32_t blockNum)
{
    uint32_t expertNumPadded = ExpertNumPaddedDevice(shape);
    for (uint32_t expert = blockId; expert < expertNumPadded; expert += blockNum) {
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
}

AICORE inline void BuildPackedExpertOffset(DispatchCombineTileShape shape, LocalWorkspaceView workspaceView,
                                           uint32_t myRank, uint32_t blockId)
{
    if (blockId != 0) {
        return;
    }
    uint32_t expertNumPadded = ExpertNumPaddedDevice(shape);
    int32_t sum = 0;
    for (uint32_t expert = 0; expert < expertNumPadded; ++expert) {
        int32_t count = LoadScalarI32(workspaceView.localTokenPerExpert + expert);
        StoreScalarI32(workspaceView.cumsumPerExpert + static_cast<uint64_t>(myRank) * expertNumPadded + expert, sum);
        if (expert < shape.expertNum) {
            sum += count;
        }
    }
}

AICORE inline void PublishCountRows(DispatchCombineTileShape shape, LocalWorkspaceView workspaceView,
                                    LocalPeerWindowView localPeer, __gm__ HcclDeviceContext *ctx, GM_ADDR peerWindow,
                                    uint32_t myRank, uint32_t blockId, uint32_t blockNum,
                                    const PeerWindowLayout &peerWindowLayout)
{
    uint32_t expertNumPadded = ExpertNumPaddedDevice(shape);
    for (uint32_t dst = blockId; dst < shape.ep; dst += blockNum) {
        LocalPeerWindowView remotePeer = MakeRemotePeerWindowView(ctx, peerWindow, dst, peerWindowLayout);
        GlobalNd<int32_t> localCount = MakeGlobal2D(workspaceView.localTokenPerExpert, 1,
                                                    static_cast<int32_t>(expertNumPadded),
                                                    static_cast<int32_t>(expertNumPadded));
        GlobalNd<int32_t> remoteCount =
            MakeGlobal2D(remotePeer.peerTokenPerExpert + static_cast<uint64_t>(myRank) * expertNumPadded, 1,
                         static_cast<int32_t>(expertNumPadded), static_cast<int32_t>(expertNumPadded));
        TPutRows<int32_t, kMetaTileCols>(remoteCount, localCount);
        NotifySignal(remotePeer.countReadySignal + myRank, 1);
    }
}

AICORE inline void WaitCountRows(DispatchCombineTileShape shape, LocalPeerWindowView localPeer, uint32_t blockId,
                                 uint32_t blockNum)
{
    for (uint32_t src = blockId; src < shape.ep; src += blockNum) {
        WaitSignal(localPeer.countReadySignal + src, 1);
    }
}

AICORE inline void BuildPrefixMetadata(DispatchCombineTileShape shape, LocalWorkspaceView workspaceView,
                                       LocalPeerWindowView localPeer, uint32_t myRank, uint32_t blockId,
                                       uint32_t blockNum)
{
    uint32_t expertNumPadded = ExpertNumPaddedDevice(shape);
    for (uint32_t src = blockId; src < shape.ep; src += blockNum) {
        int32_t sum = 0;
        for (uint32_t expert = 0; expert < expertNumPadded; ++expert) {
            sum += LoadScalarI32(localPeer.peerTokenPerExpert + static_cast<uint64_t>(src) * expertNumPadded + expert);
            StoreScalarI32(workspaceView.cumsumPerExpert + static_cast<uint64_t>(src) * expertNumPadded + expert, sum);
        }
    }

    if (blockId == 0) {
        int32_t dispatchCursor = 0;
        for (uint32_t localExpert = 0; localExpert < shape.expertPerRank; ++localExpert) {
            uint32_t globalExpert = myRank * shape.expertPerRank + localExpert;
            StoreScalarI32(workspaceView.dispatchOffset + localExpert, dispatchCursor);
            int32_t beforeRank = 0;
            for (uint32_t src = 0; src < shape.ep; ++src) {
                StoreScalarI32(workspaceView.prevSumBeforeRank + static_cast<uint64_t>(src) * shape.expertPerRank +
                                   localExpert,
                               beforeRank);
                int32_t rows =
                    LoadScalarI32(localPeer.peerTokenPerExpert + static_cast<uint64_t>(src) * expertNumPadded +
                                  globalExpert);
                beforeRank += rows;
                dispatchCursor += rows;
            }
        }
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
    for (uint32_t localExpert = 0; localExpert < shape.expertPerRank; ++localExpert) {
        uint32_t globalExpert = myRank * shape.expertPerRank + localExpert;
        for (uint32_t src = blockId; src < shape.ep; src += blockNum) {
            int32_t rows =
                LoadScalarI32(localPeer.peerTokenPerExpert + static_cast<uint64_t>(src) * expertNumPadded +
                              globalExpert);
            if (rows <= 0) {
                continue;
            }
            int32_t srcStart = globalExpert == 0 ?
                                   0 :
                                   LoadScalarI32(workspaceView.cumsumPerExpert + static_cast<uint64_t>(src) *
                                                                              expertNumPadded + globalExpert - 1);
            int32_t dstStart =
                LoadScalarI32(workspaceView.dispatchOffset + localExpert) +
                LoadScalarI32(workspaceView.prevSumBeforeRank + static_cast<uint64_t>(src) * shape.expertPerRank +
                              localExpert);
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
                                            GM_ADDR peerWindow, GM_ADDR expertOutput, uint32_t myRank,
                                            uint32_t blockId, uint32_t blockNum,
                                            const PeerWindowLayout &peerWindowLayout)
{
    uint32_t expertNumPadded = ExpertNumPaddedDevice(shape);
    __gm__ half *localExpertOutput = reinterpret_cast<__gm__ half *>(expertOutput);
    for (uint32_t src = blockId; src < shape.ep; src += blockNum) {
        LocalPeerWindowView remotePeer = MakeRemotePeerWindowView(ctx, peerWindow, src, peerWindowLayout);
        for (uint32_t localExpert = 0; localExpert < shape.expertPerRank; ++localExpert) {
            uint32_t globalExpert = myRank * shape.expertPerRank + localExpert;
            int32_t rows =
                LoadScalarI32(localPeer.peerTokenPerExpert + static_cast<uint64_t>(src) * expertNumPadded +
                              globalExpert);
            if (rows <= 0) {
                continue;
            }
            int32_t srcStart =
                LoadScalarI32(workspaceView.dispatchOffset + localExpert) +
                LoadScalarI32(workspaceView.prevSumBeforeRank + static_cast<uint64_t>(src) * shape.expertPerRank +
                              localExpert);
            int32_t dstStart = globalExpert == 0 ?
                                   0 :
                                   LoadScalarI32(workspaceView.cumsumPerExpert + static_cast<uint64_t>(src) *
                                                                              expertNumPadded + globalExpert - 1);
            TPutRowsHalf(remotePeer.ptrD, static_cast<int32_t>(shape.k), dstStart, localExpertOutput,
                         static_cast<int32_t>(shape.k), srcStart, rows, static_cast<int32_t>(shape.k));
        }
        NotifySignal(remotePeer.combineDoneSignal + myRank, 2);
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
    if (shape.ep == 0 || shape.k == 0 || shape.expertPerRank == 0 || shape.expertNum == 0 || shape.metadataPad == 0 ||
        blockId >= blockNum) {
        return;
    }

    (void)probs;
    (void)outputC;

    ReturnExpertRowsToOwners(shape, workspaceView, localPeer, ctx, peerWindow, expertOutput, myRank, blockId, blockNum,
                             peerWindowLayout);
    SoftSyncAiv(workspaceView.localSync, blockNum);
    WaitCombinePhase(shape, localPeer, blockId, blockNum, 2);
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
