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

using moe_combine::MoeCombineShape;
using moe_combine::HcclDeviceContext;
using moe_combine::PeerWindowLayout;
using moe_combine::WorkspaceLayout;

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

AICORE inline WorkspaceLayout MakeWorkspaceLayout(MoeCombineShape shape)
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

AICORE inline PeerWindowLayout MakePeerWindowLayout(MoeCombineShape shape)
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

AICORE inline void DcciGmRange(__gm__ void *ptr, uint64_t bytes)
{
    if (bytes == 0) {
        return;
    }
    constexpr uint64_t cacheLineBytes = 64;
    uint64_t start = reinterpret_cast<uint64_t>(ptr) & ~(cacheLineBytes - 1);
    uint64_t end = (reinterpret_cast<uint64_t>(ptr) + bytes + cacheLineBytes - 1) & ~(cacheLineBytes - 1);
    for (uint64_t addr = start; addr < end; addr += cacheLineBytes) {
        dcci(reinterpret_cast<__gm__ void *>(addr), SINGLE_CACHE_LINE);
    }
    dsb(DSB_DDR);
}

AICORE inline void AcquireGmRangeBeforeRead(__gm__ void *ptr, uint64_t bytes)
{
    pipe_barrier(PIPE_ALL);
    DcciGmRange(ptr, bytes);
}

AICORE inline uint32_t ExpertNumPaddedDevice(MoeCombineShape shape)
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
    pto::Tile<pto::TileType::Vec, int32_t, 1, pto::SYNCALL_SOFT_SLOT_INT32, pto::BLayout::RowMajor, -1, -1> syncTile(
        1, pto::SYNCALL_SOFT_SLOT_INT32);
#ifndef __PTO_AUTO__
    syncTile.data() = reinterpret_cast<__ubuf__ int32_t *>(kSoftSyncUbAddr);
#endif
    GlobalNd<int32_t> syncGlobal =
        MakeGlobal1D(gmWorkspace, static_cast<int32_t>(blockNum * pto::SYNCALL_SOFT_SLOT_INT32));
    pto::SYNCALL<pto::SyncAllMode::Soft>(syncGlobal, syncTile, static_cast<int32_t>(blockNum));
}

AICORE inline uint32_t EffectiveRowChunk(MoeCombineShape shape)
{
    return shape.rowChunk == 0 ? 8 : shape.rowChunk;
}

AICORE inline void WaitCombinePhase(MoeCombineShape shape, LocalPeerWindowView localPeer, uint32_t blockId,
                                    uint32_t blockNum, int32_t value)
{
    for (uint32_t peer = blockId; peer < shape.ep; peer += blockNum) {
        pto::comm::Signal sig = MakeSignal(localPeer.combineDoneSignal + peer);
        pto::comm::TWAIT(sig, value, pto::comm::WaitCmp::GE);
    }
}

AICORE inline void ReturnExpertRowsToOwners(MoeCombineShape shape, LocalWorkspaceView workspaceView,
                                            LocalPeerWindowView localPeer, __gm__ HcclDeviceContext *ctx,
                                            GM_ADDR peerWindow, GM_ADDR expertOutput, uint32_t myRank, uint32_t blockId,
                                            uint32_t blockNum, const PeerWindowLayout &peerWindowLayout)
{
    uint32_t expertNumPadded = ExpertNumPaddedDevice(shape);
    __gm__ half *localExpertOutput = reinterpret_cast<__gm__ half *>(expertOutput);
    uint32_t rowChunk = EffectiveRowChunk(shape);
    uint32_t segmentCount = shape.ep * shape.expertPerRank;
    AcquireGmRangeBeforeRead(localPeer.peerTokenPerExpert,
                             static_cast<uint32_t>(shape.ep * expertNumPadded * sizeof(int32_t)));
    AcquireGmRangeBeforeRead(workspaceView.cumsumPerExpert,
                             static_cast<uint32_t>(shape.ep * expertNumPadded * sizeof(int32_t)));
    AcquireGmRangeBeforeRead(workspaceView.dispatchOffset,
                             static_cast<uint32_t>(shape.expertPerRank * sizeof(int32_t)));
    AcquireGmRangeBeforeRead(workspaceView.prevSumBeforeRank,
                             static_cast<uint32_t>(shape.ep * shape.expertPerRank * sizeof(int32_t)));

    uint32_t chunkBase = 0;
    for (uint32_t segment = 0; segment < segmentCount; ++segment) {
        uint32_t src = segment / shape.expertPerRank;
        uint32_t localExpert = segment % shape.expertPerRank;
        uint32_t globalExpert = myRank * shape.expertPerRank + localExpert;
        int32_t rows = *(localPeer.peerTokenPerExpert + static_cast<uint64_t>(src) * expertNumPadded + globalExpert);
        if (rows <= 0) {
            continue;
        }
        uint32_t chunkCount = (static_cast<uint32_t>(rows) + rowChunk - 1) / rowChunk;
        LocalPeerWindowView remotePeer = MakeRemotePeerWindowView(ctx, peerWindow, src, peerWindowLayout);
        int32_t srcStart =
            *(workspaceView.dispatchOffset + localExpert) +
            *(workspaceView.prevSumBeforeRank + static_cast<uint64_t>(src) * shape.expertPerRank + localExpert);
        int32_t dstStart =
            globalExpert == 0 ?
                0 :
                *(workspaceView.cumsumPerExpert + static_cast<uint64_t>(src) * expertNumPadded + globalExpert - 1);
        for (uint32_t chunk = 0; chunk < chunkCount; ++chunk) {
            if (((chunkBase + chunk) % blockNum) != blockId) {
                continue;
            }
            uint32_t rowBegin = chunk * rowChunk;
            uint32_t rowsThisChunk = static_cast<uint32_t>(rows) - rowBegin;
            rowsThisChunk = rowsThisChunk < rowChunk ? rowsThisChunk : rowChunk;
            if (src == myRank) {
                for (uint32_t row = 0; row < rowsThisChunk; ++row) {
                    int32_t dstRow = dstStart + static_cast<int32_t>(rowBegin + row);
                    int32_t srcRow = srcStart + static_cast<int32_t>(rowBegin + row);
                    for (int32_t col = 0; col < static_cast<int32_t>(shape.k); col += kDefaultTileCols) {
                        int32_t cols = static_cast<int32_t>(shape.k) - col < kDefaultTileCols ?
                                           static_cast<int32_t>(shape.k) - col :
                                           kDefaultTileCols;
                        VecTile<half, kDefaultTileCols> ping(1, cols);
                        VecTile<half, kDefaultTileCols> pong(1, cols);
                        TASSIGN(ping, kPingUbAddr);
                        TASSIGN(pong, kPongUbAddr);
                        VecTile<half, kDefaultTileCols> &tile = ((col / kDefaultTileCols) & 1) == 0 ? ping : pong;
                        event_t event = ((col / kDefaultTileCols) & 1) == 0 ? EVENT_ID0 : EVENT_ID1;
                        GlobalNd<half> srcGlobal = MakeGlobal2D(
                            localExpertOutput + static_cast<int64_t>(srcRow) * static_cast<int32_t>(shape.k) + col, 1,
                            cols, static_cast<int32_t>(shape.k));
                        GlobalNd<half> dstGlobal = MakeGlobal2D(
                            localPeer.ptrD + static_cast<int64_t>(dstRow) * static_cast<int32_t>(shape.k) + col, 1,
                            cols, static_cast<int32_t>(shape.k));
                        TLOAD(tile, srcGlobal);
                        set_flag(PIPE_MTE2, PIPE_MTE3, event);
                        wait_flag(PIPE_MTE2, PIPE_MTE3, event);
                        TSTORE(dstGlobal, tile);
                        WaitStoreTileReusable();
                    }
                }
            } else {
                GlobalNd<half> remoteDst = MakeGlobal2D(
                    remotePeer.ptrD +
                        static_cast<int64_t>(dstStart + static_cast<int32_t>(rowBegin)) * static_cast<int32_t>(shape.k),
                    static_cast<int32_t>(rowsThisChunk), static_cast<int32_t>(shape.k), static_cast<int32_t>(shape.k));
                GlobalNd<half> localSrc = MakeGlobal2D(
                    localExpertOutput +
                        static_cast<int64_t>(srcStart + static_cast<int32_t>(rowBegin)) * static_cast<int32_t>(shape.k),
                    static_cast<int32_t>(rowsThisChunk), static_cast<int32_t>(shape.k), static_cast<int32_t>(shape.k));
                VecTile<half, kDefaultTileCols> ping(1, kDefaultTileCols);
                VecTile<half, kDefaultTileCols> pong(1, kDefaultTileCols);
                TASSIGN(ping, kPingUbAddr);
                TASSIGN(pong, kPongUbAddr);
                pto::comm::TPUT(remoteDst, localSrc, ping, pong);
            }
        }
        chunkBase += chunkCount;
    }
    SoftSyncAiv(workspaceView.localSync, blockNum);
    for (uint32_t src = blockId; src < shape.ep; src += blockNum) {
        LocalPeerWindowView remotePeer = MakeRemotePeerWindowView(ctx, peerWindow, src, peerWindowLayout);
        pipe_barrier(PIPE_ALL);
        pto::comm::Signal sig = MakeSignal(remotePeer.combineDoneSignal + myRank);
        (void)shape.signalValue;
        pto::comm::TNOTIFY(sig, 1, pto::comm::NotifyOp::AtomicAdd);
    }
}

AICORE inline void RestoreOutputRows(MoeCombineShape shape, LocalPeerWindowView localPeer, GM_ADDR probs,
                                     GM_ADDR outputC, uint32_t blockId, uint32_t blockNum)
{
    __gm__ float *probValues = reinterpret_cast<__gm__ float *>(probs);
    __gm__ half *output = reinterpret_cast<__gm__ half *>(outputC);
    uint32_t tokenBegin = TokenShardBegin(shape.m, blockId, blockNum);
    uint32_t tokenEnd = TokenShardEnd(shape.m, blockId, blockNum);
    for (uint32_t token = tokenBegin; token < tokenEnd; ++token) {
        for (uint32_t slot = 0; slot < shape.topK; ++slot) {
            uint32_t routeIndex = token * shape.topK + slot;
            int32_t ptrDRow = *(localPeer.expandedRowIdx + routeIndex);
            if (ptrDRow < 0) {
                continue;
            }
            __gm__ half *ptrRow = localPeer.ptrD + static_cast<int64_t>(ptrDRow) * static_cast<int32_t>(shape.k);
            AcquireGmRangeBeforeRead(ptrRow, static_cast<uint64_t>(shape.k) * sizeof(half));
        }
        for (int32_t col = 0; col < static_cast<int32_t>(shape.k); col += kDefaultTileCols) {
            int32_t cols = static_cast<int32_t>(shape.k) - col < kDefaultTileCols ?
                               static_cast<int32_t>(shape.k) - col :
                               kDefaultTileCols;
            VecTile<half, kDefaultTileCols> outTile(1, cols);
            VecTile<half, kDefaultTileCols> ptrTile(1, cols);
            TASSIGN(outTile, kPingUbAddr);
            TASSIGN(ptrTile, kPongUbAddr);
            GlobalNd<half> outGlobal =
                MakeGlobal2D(output + static_cast<int64_t>(token) * static_cast<int32_t>(shape.k) + col, 1, cols,
                             static_cast<int32_t>(shape.k));
            TEXPANDS(outTile, static_cast<half>(0.0));
            pipe_barrier(PIPE_ALL);
            for (uint32_t slot = 0; slot < shape.topK; ++slot) {
                uint32_t routeIndex = token * shape.topK + slot;
                int32_t ptrDRow = *(localPeer.expandedRowIdx + routeIndex);
                if (ptrDRow < 0) {
                    continue;
                }
                __gm__ half *ptrRow = localPeer.ptrD + static_cast<int64_t>(ptrDRow) * static_cast<int32_t>(shape.k);
                __gm__ half *ptrChunk = ptrRow + col;
                GlobalNd<half> ptrGlobal = MakeGlobal2D(ptrChunk, 1, cols, static_cast<int32_t>(shape.k));
                pto::Event<pto::Op::TLOAD, pto::Op::TAXPY> loadToAxpy;
                float prob = probValues[routeIndex];
                loadToAxpy = TLOAD(ptrTile, ptrGlobal);
                TAXPY(outTile, ptrTile, static_cast<half>(prob), loadToAxpy);
                pipe_barrier(PIPE_ALL);
            }
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            TSTORE(outGlobal, outTile);
            WaitStoreTileReusable();
        }
    }
}

} // namespace

__global__ AICORE void MoeCombineKernel(MoeCombineShape shape, uint32_t myRank, GM_ADDR expertOutput,
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
    WaitCombinePhase(shape, localPeer, blockId, blockNum, static_cast<int32_t>(shape.signalValue));
    SoftSyncAiv(workspaceView.localSync, blockNum);
    RestoreOutputRows(shape, localPeer, probs, outputC, blockId, blockNum);
    SoftSyncAiv(workspaceView.localSync, blockNum);
}

namespace moe_combine {

void LaunchMoeCombineKernel(MoeCombineShape shape, uint32_t myRank, uint8_t *expertOutput,
                                      uint8_t *probs, uint8_t *outputC, uint8_t *peerWindow, uint8_t *hcclCtx,
                                      uint8_t *workspace, void *stream, uint32_t launchBlockCount)
{
    MoeCombineKernel<<<launchBlockCount, nullptr, stream>>>(shape, myRank, expertOutput, probs, outputC,
                                                                      peerWindow, hcclCtx, workspace);
}

} // namespace moe_combine
