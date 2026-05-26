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

namespace {

// ============================================================================
// UB Storage Plan
// ============================================================================
namespace ub_plan {
constexpr uint64_t kPingOffset = 0x0000;
constexpr uint64_t kPongOffset = 0x1000;
constexpr uint64_t kMetaOffset = 0x2000;
constexpr uint64_t kSdmaScratchOffset = 0x3000;
constexpr uint64_t kSyncOffset = 0x5000;

constexpr uint64_t kTotalUsed = kSyncOffset + 0x100;
static_assert(kTotalUsed <= 192 * 1024, "UB budget exceeded for A2/A3");
} // namespace ub_plan

constexpr int kDefaultTileCols = 1024;
constexpr int kMetaTileElems = 256;

using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;

template <typename T>
using GlobalNd = pto::GlobalTensor<T, ShapeDyn, StrideDyn, pto::Layout::ND>;

template <typename T, int kCols = kDefaultTileCols>
using VecTile = pto::Tile<pto::TileType::Vec, T, 1, kCols, pto::BLayout::RowMajor, -1, -1>;

using MetaTile = pto::Tile<pto::TileType::Vec, int32_t, 1, kMetaTileElems, pto::BLayout::RowMajor, -1, -1>;
using SdmaScratchTile =
    pto::Tile<pto::TileType::Vec, uint8_t, 1, pto::comm::sdma::UB_ALIGN_SIZE, pto::BLayout::RowMajor, -1, -1>;

AICORE inline uint64_t Align64Device(uint64_t value)
{
    return ((value + 63) / 64) * 64;
}

AICORE inline uint64_t AllocAligned64(uint64_t &offset, uint64_t bytes)
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
    layout.localTokenPerExpert = AllocAligned64(offset, expertNumPadded * i32);
    layout.blockTokenPerExpert = AllocAligned64(offset, aivBlocks * expertNumPadded * i32);
    layout.blockPrefixPerExpert = AllocAligned64(offset, aivBlocks * expertNumPadded * i32);
    layout.cumsumPerExpert = AllocAligned64(offset, static_cast<uint64_t>(shape.ep) * expertNumPadded * i32);
    layout.dispatchOffset = AllocAligned64(offset, static_cast<uint64_t>(shape.expertPerRank) * i32);
    layout.prevSumBeforeRank = AllocAligned64(offset, static_cast<uint64_t>(shape.ep) * shape.expertPerRank * i32);
    uint64_t syncSlots = aivBlocks * (8 + expertNumPadded);
    syncSlots = syncSlots < 64 ? 64 : syncSlots;
    layout.localSync = AllocAligned64(offset, syncSlots * i32);
    layout.floatScratch = AllocAligned64(offset, aivBlocks * shape.tileCols * f32);
    layout.dispatchedA = AllocAligned64(offset, static_cast<uint64_t>(shape.maxOutputSize) * shape.k * f16);
    layout.ptrDLocal = AllocAligned64(offset, expandedRows * shape.k * f16);
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
    layout.peerTokenPerExpert = AllocAligned64(offset, static_cast<uint64_t>(shape.ep) * expertNumPadded * i32);
    layout.expandedRowIdx = AllocAligned64(offset, expandedRows * i32);
    layout.packedA = AllocAligned64(offset, expandedRows * shape.k * f16);
    layout.ptrD = AllocAligned64(offset, expandedRows * shape.k * f16);
    layout.countReadySignal = AllocAligned64(offset, static_cast<uint64_t>(shape.ep) * i32);
    layout.combineDoneSignal = AllocAligned64(offset, static_cast<uint64_t>(shape.ep) * i32);
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

// Load metadata from GM into UB tile via MTE2 (bypasses scalar D-cache),
// then read scalar from UB. Replaces AcquireGmRangeBeforeRead + scalar GM read.
AICORE inline int32_t LoadMetadataScalar(__gm__ int32_t *gmBase, uint32_t index, MetaTile &metaTile)
{
    uint32_t tileBase = (index / kMetaTileElems) * kMetaTileElems;
    uint32_t localIdx = index - tileBase;
    int32_t elemsToLoad = static_cast<int32_t>(kMetaTileElems);
    metaTile.SetValidCol(elemsToLoad);
    GlobalNd<int32_t> metaGlobal = MakeGlobal1D(gmBase + tileBase, elemsToLoad);
    TLOAD(metaTile, metaGlobal);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID7);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID7);
    return metaTile.GetValue(localIdx);
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
    pto::Tile<pto::TileType::Vec, int32_t, 1, pto::SYNCALL_SOFT_SLOT_INT32, pto::BLayout::RowMajor, -1, -1> syncTile(
        1, pto::SYNCALL_SOFT_SLOT_INT32);
#ifndef __PTO_AUTO__
    syncTile.data() = reinterpret_cast<__ubuf__ int32_t *>(ub_plan::kSyncOffset);
#endif
    GlobalNd<int32_t> syncGlobal =
        MakeGlobal1D(gmWorkspace, static_cast<int32_t>(blockNum * pto::SYNCALL_SOFT_SLOT_INT32));
    pto::SYNCALL<pto::SyncAllMode::Soft>(syncGlobal, syncTile, static_cast<int32_t>(blockNum));
}

AICORE inline uint32_t EffectiveRowChunk(DispatchCombineTileShape shape)
{
    return shape.rowChunk == 0 ? 8 : shape.rowChunk;
}

AICORE inline void WaitCombinePhase(DispatchCombineTileShape shape, LocalPeerWindowView localPeer, uint32_t blockId,
                                    uint32_t blockNum, int32_t value)
{
    for (uint32_t peer = blockId; peer < shape.ep; peer += blockNum) {
        pto::comm::Signal sig = MakeSignal(localPeer.combineDoneSignal + peer);
        pto::comm::TWAIT(sig, value, pto::comm::WaitCmp::GE);
    }
}

AICORE inline void ReturnExpertRowsToOwners(DispatchCombineTileShape shape, LocalWorkspaceView workspaceView,
                                            LocalPeerWindowView localPeer, __gm__ HcclDeviceContext *ctx,
                                            GM_ADDR peerWindow, GM_ADDR expertOutput, uint32_t myRank, uint32_t blockId,
                                            uint32_t blockNum, const PeerWindowLayout &peerWindowLayout,
                                            pto::comm::AsyncSession &sdmaSession)
{
    uint32_t expertNumPadded = ExpertNumPaddedDevice(shape);
    __gm__ half *localExpertOutput = reinterpret_cast<__gm__ half *>(expertOutput);
    uint32_t rowChunk = EffectiveRowChunk(shape);
    uint32_t segmentCount = shape.ep * shape.expertPerRank;
    int32_t kCols = static_cast<int32_t>(shape.k);

    MetaTile metaTile(1, kMetaTileElems);
#ifndef __PTO_AUTO__
    metaTile.data() = reinterpret_cast<__ubuf__ int32_t *>(ub_plan::kMetaOffset);
#endif

    pto::comm::AsyncEvent lastAsyncEvent;
    bool hasAsyncOps = false;

    uint32_t chunkBase = 0;
    for (uint32_t segment = 0; segment < segmentCount; ++segment) {
        uint32_t src = segment / shape.expertPerRank;
        uint32_t localExpert = segment % shape.expertPerRank;
        uint32_t globalExpert = myRank * shape.expertPerRank + localExpert;
        int32_t rows = LoadMetadataScalar(localPeer.peerTokenPerExpert,
                                          static_cast<uint32_t>(src) * expertNumPadded + globalExpert, metaTile);
        if (rows <= 0) {
            continue;
        }
        uint32_t chunkCount = (static_cast<uint32_t>(rows) + rowChunk - 1) / rowChunk;
        LocalPeerWindowView remotePeer = MakeRemotePeerWindowView(ctx, peerWindow, src, peerWindowLayout);
        int32_t srcStart =
            LoadMetadataScalar(workspaceView.dispatchOffset, localExpert, metaTile) +
            LoadMetadataScalar(workspaceView.prevSumBeforeRank,
                               static_cast<uint32_t>(src) * shape.expertPerRank + localExpert, metaTile);
        int32_t dstStart =
            globalExpert == 0
                ? 0
                : LoadMetadataScalar(workspaceView.cumsumPerExpert,
                                     static_cast<uint32_t>(src) * expertNumPadded + globalExpert - 1, metaTile);
        for (uint32_t chunk = 0; chunk < chunkCount; ++chunk) {
            if (((chunkBase + chunk) % blockNum) != blockId) {
                continue;
            }
            uint32_t rowBegin = chunk * rowChunk;
            uint32_t rowsThisChunk = static_cast<uint32_t>(rows) - rowBegin;
            rowsThisChunk = rowsThisChunk < rowChunk ? rowsThisChunk : rowChunk;
            int64_t chunkOffset = static_cast<int64_t>(dstStart + static_cast<int32_t>(rowBegin)) * kCols;
            int32_t totalElems = static_cast<int32_t>(rowsThisChunk) * kCols;
            if (src == myRank) {
                VecTile<half, kDefaultTileCols> ping(1, kDefaultTileCols);
                VecTile<half, kDefaultTileCols> pong(1, kDefaultTileCols);
                TASSIGN(ping, ub_plan::kPingOffset);
                TASSIGN(pong, ub_plan::kPongOffset);
                int64_t srcOffset = static_cast<int64_t>(srcStart + static_cast<int32_t>(rowBegin)) * kCols;
                GlobalNd<half> srcGlobal = MakeGlobal1D(localExpertOutput + srcOffset, totalElems);
                GlobalNd<half> dstGlobal = MakeGlobal1D(localPeer.ptrD + chunkOffset, totalElems);
                pto::comm::TPUT(dstGlobal, srcGlobal, ping, pong);
            } else {
                GlobalNd<half> remoteDst = MakeGlobal1D(remotePeer.ptrD + chunkOffset, totalElems);
                int64_t srcOffset = static_cast<int64_t>(srcStart + static_cast<int32_t>(rowBegin)) * kCols;
                GlobalNd<half> localSrc = MakeGlobal1D(localExpertOutput + srcOffset, totalElems);
                lastAsyncEvent = pto::comm::TPUT_ASYNC<pto::comm::DmaEngine::SDMA>(remoteDst, localSrc, sdmaSession);
                hasAsyncOps = true;
            }
        }
        chunkBase += chunkCount;
    }

    if (hasAsyncOps) {
        (void)lastAsyncEvent.Wait(sdmaSession);
    }

    SoftSyncAiv(workspaceView.localSync, blockNum);
    for (uint32_t src = blockId; src < shape.ep; src += blockNum) {
        LocalPeerWindowView remotePeer = MakeRemotePeerWindowView(ctx, peerWindow, src, peerWindowLayout);
        pto::comm::Signal sig = MakeSignal(remotePeer.combineDoneSignal + myRank);
        pto::comm::TNOTIFY(sig, 1, pto::comm::NotifyOp::AtomicAdd);
    }
}

AICORE inline void RestoreOutputRows(DispatchCombineTileShape shape, LocalPeerWindowView localPeer, GM_ADDR probs,
                                     GM_ADDR outputC, uint32_t blockId, uint32_t blockNum)
{
    __gm__ float *probValues = reinterpret_cast<__gm__ float *>(probs);
    __gm__ half *output = reinterpret_cast<__gm__ half *>(outputC);
    int32_t kCols = static_cast<int32_t>(shape.k);
    uint32_t tokenBegin = TokenShardBegin(shape.m, blockId, blockNum);
    uint32_t tokenEnd = TokenShardEnd(shape.m, blockId, blockNum);
    for (uint32_t token = tokenBegin; token < tokenEnd; ++token) {
        int64_t tokenOffset = static_cast<int64_t>(token) * kCols;
        for (int32_t col = 0; col < kCols; col += kDefaultTileCols) {
            int32_t cols = kCols - col < kDefaultTileCols ? kCols - col : kDefaultTileCols;
            VecTile<half, kDefaultTileCols> zeroTile(1, cols);
            TASSIGN(zeroTile, ub_plan::kPingOffset);
            GlobalNd<half> outputGlobal = MakeGlobal2D(output + tokenOffset + col, 1, cols, kCols);
            TEXPANDS(zeroTile, static_cast<half>(0.0));
            TSTORE(outputGlobal, zeroTile);
            set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
            wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        }
        for (uint32_t slot = 0; slot < shape.topK; ++slot) {
            uint32_t routeIndex = token * shape.topK + slot;
            int32_t ptrDRow = *(localPeer.expandedRowIdx + routeIndex);
            if (ptrDRow < 0) {
                continue;
            }
            float prob = probValues[routeIndex];
            int64_t ptrDOffset = static_cast<int64_t>(ptrDRow) * kCols;
            for (int32_t col = 0; col < kCols; col += kDefaultTileCols) {
                int32_t cols = kCols - col < kDefaultTileCols ? kCols - col : kDefaultTileCols;
                VecTile<half, kDefaultTileCols> outTile(1, cols);
                VecTile<half, kDefaultTileCols> ptrTile(1, cols);
                TASSIGN(outTile, ub_plan::kPingOffset);
                TASSIGN(ptrTile, ub_plan::kPongOffset);
                GlobalNd<half> outGlobal = MakeGlobal2D(output + tokenOffset + col, 1, cols, kCols);
                GlobalNd<half> ptrGlobal = MakeGlobal2D(localPeer.ptrD + ptrDOffset + col, 1, cols, kCols);
                pto::Event<pto::Op::TLOAD, pto::Op::TAXPY> loadToAxpy;
                pto::Event<pto::Op::TAXPY, pto::Op::TSTORE_VEC> axpyToStore;
                TLOAD(ptrTile, ptrGlobal);
                loadToAxpy = TLOAD(outTile, outGlobal);
                axpyToStore = TAXPY(outTile, ptrTile, static_cast<half>(prob), loadToAxpy);
                TSTORE(outGlobal, outTile, axpyToStore);
                set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
                wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
            }
        }
    }
}

} // namespace

__global__ AICORE void DispatchCombineTileCombine(DispatchCombineTileShape shape, uint32_t myRank, GM_ADDR expertOutput,
                                                  GM_ADDR probs, GM_ADDR outputC, GM_ADDR peerWindow, GM_ADDR hcclCtx,
                                                  GM_ADDR workspace, GM_ADDR sdmaWorkspace)
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

    SdmaScratchTile scratchTile(1, pto::comm::sdma::UB_ALIGN_SIZE);
#ifndef __PTO_AUTO__
    scratchTile.data() = reinterpret_cast<__ubuf__ uint8_t *>(ub_plan::kSdmaScratchOffset);
#endif
    TASSIGN(scratchTile, static_cast<uint8_t>(0));

    pto::comm::AsyncSession sdmaSession;
    pto::comm::BuildAsyncSession<pto::comm::DmaEngine::SDMA>(scratchTile, sdmaWorkspace, sdmaSession);

    ReturnExpertRowsToOwners(shape, workspaceView, localPeer, ctx, peerWindow, expertOutput, myRank, blockId, blockNum,
                             peerWindowLayout, sdmaSession);
    WaitCombinePhase(shape, localPeer, blockId, blockNum, static_cast<int32_t>(shape.signalValue));
    SoftSyncAiv(workspaceView.localSync, blockNum);
    RestoreOutputRows(shape, localPeer, probs, outputC, blockId, blockNum);
    SoftSyncAiv(workspaceView.localSync, blockNum);
}

namespace dispatch_combine_tile {

void LaunchDispatchCombineTileCombine(DispatchCombineTileShape shape, uint32_t myRank, uint8_t *expertOutput,
                                      uint8_t *probs, uint8_t *outputC, uint8_t *peerWindow, uint8_t *hcclCtx,
                                      uint8_t *workspace, uint8_t *sdmaWorkspace, void *stream,
                                      uint32_t launchBlockCount)
{
    DispatchCombineTileCombine<<<launchBlockCount, nullptr, stream>>>(shape, myRank, expertOutput, probs, outputC,
                                                                      peerWindow, hcclCtx, workspace, sdmaWorkspace);
}

} // namespace dispatch_combine_tile
