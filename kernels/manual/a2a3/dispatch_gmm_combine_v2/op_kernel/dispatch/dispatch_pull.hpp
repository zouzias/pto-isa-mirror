#pragma once

#if defined(__CCE_AICORE__)
#include "kernel_operator.h"
#include "pto/comm/pto_comm_inst.hpp"
#include <pto/pto-inst.hpp>

#include "../dispatch_ffn_combine_tiling.h"
#include "../protocol/remote_window.hpp"
#include "../protocol/signal_protocol.hpp"
#include "../protocol/task_plan.hpp"
#include "../protocol/pto_sync_bridge.hpp"


#define V4_FORCE_INLINE_AICORE inline __attribute__((always_inline)) __aicore__

namespace v2_dispatch {

using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;

template <typename T>
using PtoGlobal = pto::GlobalTensor<T, ShapeDyn, StrideDyn, pto::Layout::ND>;

template <typename T, int TileElems = 64>
V4_FORCE_INLINE_AICORE void DispatchTGetContiguous(__gm__ T* dstPtr, __gm__ T* srcPtr, int32_t elemNum)
{
    using TileData = pto::Tile<pto::TileType::Vec, T, 1, TileElems, pto::BLayout::RowMajor, -1, -1>;
    using Global = PtoGlobal<T>;
    for (int32_t offset = 0; offset < elemNum; offset += TileElems) {
        int32_t cur = elemNum - offset;
        if (cur > TileElems) {
            cur = TileElems;
        }
        ShapeDyn shape(1, 1, 1, 1, cur);
        StrideDyn stride(cur, cur, cur, cur, 1);
        Global dstG(dstPtr + offset, shape, stride);
        Global srcG(srcPtr + offset, shape, stride);
        TileData tile(1, cur);
        TASSIGN(tile, 0x0);
        pto::comm::TGET(dstG, srcG, tile);
        pipe_barrier(PIPE_ALL);
    }
    dsb(DSB_DDR);
}

template <typename T, int TileElems = 64>
V4_FORCE_INLINE_AICORE void DispatchSelfCopyContiguous(__gm__ T* dstPtr, __gm__ T* srcPtr, int32_t elemNum)
{
    using TileData = pto::Tile<pto::TileType::Vec, T, 1, TileElems, pto::BLayout::RowMajor, -1, -1>;
    using Global = PtoGlobal<T>;
    constexpr uint64_t kPingOffset = 0x0;
    constexpr uint64_t kPongOffset = 0x400;
    const int32_t fullElems = (elemNum / TileElems) * TileElems;
    TileData pingTile(1, TileElems);
    TileData pongTile(1, TileElems);
    TASSIGN(pingTile, kPingOffset);
    TASSIGN(pongTile, kPongOffset);
    for (int32_t offset = 0; offset < fullElems; offset += TileElems) {
        const uint32_t chunk = static_cast<uint32_t>(offset / TileElems);
        TileData& tile = (chunk & 1) == 0 ? pingTile : pongTile;
        const uint32_t event = (chunk & 1) == 0 ? EVENT_ID0 : EVENT_ID1;
        ShapeDyn shape(1, 1, 1, 1, TileElems);
        StrideDyn stride(TileElems, TileElems, TileElems, TileElems, 1);
        Global dstG(dstPtr + offset, shape, stride);
        Global srcG(srcPtr + offset, shape, stride);
        pto::TLOAD(tile, srcG);
        set_flag(PIPE_MTE2, PIPE_MTE3, event);
        wait_flag(PIPE_MTE2, PIPE_MTE3, event);
        pto::TSTORE(dstG, tile);
        set_flag(PIPE_MTE3, PIPE_MTE2, event);
        wait_flag(PIPE_MTE3, PIPE_MTE2, event);
        pipe_barrier(PIPE_ALL);
    }
    if (fullElems < elemNum) {
        const int32_t cur = elemNum - fullElems;
        ShapeDyn shape(1, 1, 1, 1, cur);
        StrideDyn stride(cur, cur, cur, cur, 1);
        Global dstG(dstPtr + fullElems, shape, stride);
        Global srcG(srcPtr + fullElems, shape, stride);
        TileData tile(1, cur);
        TASSIGN(tile, kPingOffset);
        pto::TLOAD(tile, srcG);
        set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        pto::TSTORE(dstG, tile);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        pipe_barrier(PIPE_ALL);
    }
    dsb(DSB_DDR);
}

V4_FORCE_INLINE_AICORE __gm__ int32_t* DispatchSignalBase(__gm__ RemoteWindowContext* ctx)
{
    return reinterpret_cast<__gm__ int32_t*>(ctx->workspaceBase + ctx->signalRegionOffset);
}

V4_FORCE_INLINE_AICORE __gm__ uint8_t* DispatchRemoteBytes(__gm__ RemoteWindowContext* ctx,
                                                           uint32_t rank,
                                                           uint64_t offsetBytes)
{
    return reinterpret_cast<__gm__ uint8_t*>(ctx->windowIn[rank] + ctx->dispatchRegionOffset + offsetBytes);
}

V4_FORCE_INLINE_AICORE __gm__ uint8_t* DispatchLocalComputeBytes(__gm__ RemoteWindowContext* ctx,
                                                                 uint64_t offsetBytes)
{
    return reinterpret_cast<__gm__ uint8_t*>(ctx->workspaceBase + ctx->computeRegionOffset + offsetBytes);
}

V4_FORCE_INLINE_AICORE pto::comm::Signal DispatchReadySignal(__gm__ int32_t* signalBase, uint32_t index)
{
    return pto::comm::Signal(signalBase + index);
}

V4_FORCE_INLINE_AICORE pto::comm::Signal DispatchDoneSignal(__gm__ int32_t* signalBase, uint32_t index)
{
    return pto::comm::Signal(signalBase + index);
}

V4_FORCE_INLINE_AICORE void WaitDispatchReady(pto::comm::Signal ready, int32_t epoch)
{
    v2_pto_sync::WaitCommSignalReady(ready, epoch);
}

V4_FORCE_INLINE_AICORE void CopyDispatchBytes(__gm__ RemoteWindowContext* ctx,
                                              uint32_t srcRank,
                                              __gm__ uint8_t* dst,
                                              __gm__ uint8_t* src,
                                              uint32_t bytes)
{
    if ((bytes & 0x3U) == 0) {
        auto* srcWords = reinterpret_cast<__gm__ uint32_t*>(src);
        auto* dstWords = reinterpret_cast<__gm__ uint32_t*>(dst);
        const int32_t wordCount = static_cast<int32_t>(bytes / sizeof(uint32_t));
        if (srcRank == ctx->rank) {
            DispatchSelfCopyContiguous<uint32_t>(dstWords, srcWords, wordCount);
        } else {
            DispatchTGetContiguous<uint32_t>(dstWords, srcWords, wordCount);
        }
    } else if (srcRank == ctx->rank) {
        DispatchSelfCopyContiguous<uint8_t>(dst, src, static_cast<int32_t>(bytes));
    } else {
        DispatchTGetContiguous<uint8_t>(dst, src, static_cast<int32_t>(bytes));
    }
}

V4_FORCE_INLINE_AICORE void PublishDispatchRangeCompletion(__gm__ int32_t* signalBase,
                                                           const __gm__ DispatchRangeTask* task)
{
    v2_pto_sync::StoreWindowReadyNoFence(
        v2_pto_sync::WindowSignal(signalBase, task->rangeStatusIndex), static_cast<int32_t>(task->readyRows));
    v2_pto_sync::StoreWindowReadyNoFence(
        v2_pto_sync::WindowSignal(signalBase, task->dispatchRangeCompletionIndex), 1);
    v2_pto_sync::StoreWindowReadyNoFence(
        v2_pto_sync::WindowSignal(signalBase, task->expertSafeRowsIndex), static_cast<int32_t>(task->localExpertRowEnd));
    v2_pto_sync::PublishGmWrites();
}

V4_FORCE_INLINE_AICORE void RunDispatchPullTask(__gm__ RemoteWindowContext* ctx,
                                                const __gm__ DispatchPullTask* task)
{
    __gm__ int32_t* signalBase = DispatchSignalBase(ctx);
    auto ready = DispatchReadySignal(signalBase, DispatchReadyIndex(task->srcRank, task->srcRowBegin));
    auto done = DispatchDoneSignal(signalBase, DispatchDoneIndex(task->srcRank, task->srcRowBegin));

    WaitDispatchReady(ready, task->readyEpoch);
    CopyDispatchBytes(ctx,
                      task->srcRank,
                      DispatchLocalComputeBytes(ctx, task->dstPayloadOffsetBytes),
                      DispatchRemoteBytes(ctx, task->srcRank, task->srcPayloadOffsetBytes),
                      task->hiddenBytes);
    CopyDispatchBytes(ctx,
                      task->srcRank,
                      DispatchLocalComputeBytes(ctx, task->dstScaleOffsetBytes),
                      DispatchRemoteBytes(ctx, task->srcRank, task->srcScaleOffsetBytes),
                      kDispatchScaleSlotBytes);
    v2_pto_sync::NotifyCommSignalReady(done, task->readyEpoch);
}

V4_FORCE_INLINE_AICORE void RunDispatchRangeTask(__gm__ RemoteWindowContext* ctx,
                                                 const __gm__ DispatchRangeTask* task)
{
    __gm__ int32_t* signalBase = DispatchSignalBase(ctx);
    auto ready = DispatchReadySignal(signalBase, task->readySignalIndex);
    auto done = DispatchDoneSignal(signalBase, task->doneSignalIndex);

    WaitDispatchReady(ready, 1);
    CopyDispatchBytes(ctx,
                      task->srcRank,
                      DispatchLocalComputeBytes(ctx, task->dstExpertMajorOffsetBytes),
                      DispatchRemoteBytes(ctx, task->srcRank, task->srcPayloadOffsetBytes),
                      task->rowCount * task->hiddenBytes);
    CopyDispatchBytes(ctx,
                      task->srcRank,
                      DispatchLocalComputeBytes(ctx, task->dstScaleOffsetBytes),
                      DispatchRemoteBytes(ctx, task->srcRank, task->srcScaleOffsetBytes),
                      task->rowCount * task->scaleBytes);
    PublishDispatchRangeCompletion(signalBase, task);
    v2_pto_sync::NotifyCommSignalReady(done, 1);
}

V4_FORCE_INLINE_AICORE void RunDispatchRangeOnlyKernel(__gm__ RemoteWindowContext* ctx,
                                                       __gm__ DispatchRangeTask* tasks,
                                                       const __gm__ StandaloneKernelTilingData* tiling)
{
    if (get_block_idx() != 0) {
        return;
    }
    for (uint32_t taskIdx = 0; taskIdx < tiling->taskCount; ++taskIdx) {
        RunDispatchRangeTask(ctx, tasks + taskIdx);
    }
}

}  // namespace v2_dispatch

#endif
