#pragma once

#if defined(__CCE_AICORE__)
#include "kernel_operator.h"
#include "pto/comm/pto_comm_inst.hpp"
#include <pto/pto-inst.hpp>

#include "combine_progress.hpp"
#include "../dispatch_ffn_combine_tiling.h"
#include "../output/unpermute_reduce.hpp"
#include "../protocol/remote_window.hpp"
#include "../protocol/task_plan.hpp"
#include "../protocol/pto_sync_bridge.hpp"


#ifndef V4_FORCE_INLINE_AICORE
#define V4_FORCE_INLINE_AICORE inline __attribute__((always_inline)) __aicore__
#endif

namespace v2_combine {

using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
using CombineGlobal = pto::GlobalTensor<half, ShapeDyn, StrideDyn, pto::Layout::ND>;
using CombineTile = pto::Tile<pto::TileType::Vec, half, 1, kCombineTileAlignBytes / sizeof(half), pto::BLayout::RowMajor, -1, -1>;
using DoneGlobal = pto::GlobalTensor<int32_t, ShapeDyn, StrideDyn, pto::Layout::ND>;
using DoneTile = pto::Tile<pto::TileType::Vec, int32_t, 1, kCombineTileAlignBytes / sizeof(int32_t), pto::BLayout::RowMajor, -1, -1>;

V4_FORCE_INLINE_AICORE void PublishGmWrites()
{
    v2_pto_sync::PublishGmWrites();
}

V4_FORCE_INLINE_AICORE void MakeVisibleBeforeCombineRead(__gm__ void* ptr)
{
    v2_pto_sync::MakeCacheLineVisible(ptr);
}

V4_FORCE_INLINE_AICORE void SynchronizeCombinePipe()
{
    v2_pto_sync::SynchronizePipe();
}

V4_FORCE_INLINE_AICORE __gm__ half* LocalPartialHalf(__gm__ CombineOnlyParams* params, uint64_t offsetBytes)
{
    return reinterpret_cast<__gm__ half*>(params->localPartial + offsetBytes);
}

V4_FORCE_INLINE_AICORE __gm__ half* LocalCombineHalf(__gm__ RemoteWindowContext* ctx, uint64_t offsetBytes)
{
    return reinterpret_cast<__gm__ half*>(ctx->workspaceBase + ctx->combineRegionOffset + offsetBytes);
}

V4_FORCE_INLINE_AICORE __gm__ half* RemoteCombineHalf(__gm__ RemoteWindowContext* ctx,
                                                      uint32_t rank,
                                                      uint64_t offsetBytes)
{
    return reinterpret_cast<__gm__ half*>(ctx->windowIn[rank] + ctx->combineRegionOffset + offsetBytes);
}

V4_FORCE_INLINE_AICORE __gm__ int32_t* LocalSignalBase(__gm__ RemoteWindowContext* ctx)
{
    return reinterpret_cast<__gm__ int32_t*>(ctx->workspaceBase + ctx->signalRegionOffset);
}

V4_FORCE_INLINE_AICORE volatile __gm__ int32_t* LocalSignalBaseVolatile(__gm__ RemoteWindowContext* ctx)
{
    return reinterpret_cast<volatile __gm__ int32_t*>(ctx->workspaceBase + ctx->signalRegionOffset);
}

V4_FORCE_INLINE_AICORE __gm__ int32_t* RemoteSignalInt32(__gm__ RemoteWindowContext* ctx,
                                                         uint32_t rank,
                                                         uint32_t index)
{
    return reinterpret_cast<__gm__ int32_t*>(ctx->windowIn[rank] + ctx->signalRegionOffset) + index;
}

V4_FORCE_INLINE_AICORE __gm__ int32_t* DoneValue(__gm__ CombineOnlyParams* params)
{
    return reinterpret_cast<__gm__ int32_t*>(params->doneValue);
}

V4_FORCE_INLINE_AICORE void WaitVisibleSignal(volatile __gm__ int32_t* signal, int32_t expected)
{
    v2_pto_sync::WaitVisibleGmValue(signal, expected);
}

V4_FORCE_INLINE_AICORE void MakeLocalPartialVisible(__gm__ half* src, uint32_t elems)
{
    constexpr uint32_t tileElems = kCombineTileAlignBytes / sizeof(half);
    for (uint32_t offset = 0; offset < elems; offset += tileElems) {
        MakeVisibleBeforeCombineRead((__gm__ void*)(src + offset));
    }
    SynchronizeCombinePipe();
}

using ScaleFloatTile = pto::Tile<pto::TileType::Vec, float, 1, kCombineTileAlignBytes / sizeof(float),
                                 pto::BLayout::RowMajor, -1, -1>;
using ScaleHalfTile = pto::Tile<pto::TileType::Vec, half, 1, kCombineTileAlignBytes / sizeof(half),
                                pto::BLayout::RowMajor, -1, -1>;

V4_FORCE_INLINE_AICORE void ScaleRowsInPlace(__gm__ half* rowBase,
                                             __gm__ float* perTokenScale2,
                                             uint32_t rowCount,
                                             uint32_t outputElems)
{
    if (perTokenScale2 == nullptr || rowCount == 0 || outputElems == 0) {
        return;
    }
    constexpr uint32_t floatTileElems = kCombineTileAlignBytes / sizeof(float);
    constexpr uint32_t halfTileElems = kCombineTileAlignBytes / sizeof(half);
    constexpr uint64_t kHalfLoadOff = 0x0;
    constexpr uint64_t kFloatCvtOff = kCombineTileAlignBytes;
    constexpr uint64_t kHalfStoreOff = kFloatCvtOff + kCombineTileAlignBytes * 2;

    for (uint32_t row = 0; row < rowCount; ++row) {
        const float scale = *reinterpret_cast<volatile __gm__ float*>(perTokenScale2 + row);
        auto* rowPtr = rowBase + static_cast<uint64_t>(row) * outputElems;

        for (uint32_t offset = 0; offset < outputElems; offset += floatTileElems) {
            uint32_t cur = outputElems - offset;
            if (cur > floatTileElems) {
                cur = floatTileElems;
            }
            ScaleHalfTile halfLoadTile(1, cur);
            ScaleFloatTile floatTile(1, cur);
            ScaleHalfTile halfStoreTile(1, cur);
            pto::TASSIGN(halfLoadTile, kHalfLoadOff);
            pto::TASSIGN(floatTile, kFloatCvtOff);
            pto::TASSIGN(halfStoreTile, kHalfStoreOff);

            ShapeDyn shape(1, 1, 1, 1, cur);
            StrideDyn stride(cur, cur, cur, cur, 1);
            CombineGlobal srcG(rowPtr + offset, shape, stride);
            CombineGlobal dstG(rowPtr + offset, shape, stride);

            pto::TLOAD(halfLoadTile, srcG);
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

            pto::TCVT(floatTile, halfLoadTile, pto::RoundMode::CAST_RINT);
            pipe_barrier(PIPE_V);
            pto::TMULS(floatTile, floatTile, scale);
            pipe_barrier(PIPE_V);
            pto::TCVT(halfStoreTile, floatTile, pto::RoundMode::CAST_RINT);
            pipe_barrier(PIPE_V);

            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            pto::TSTORE(dstG, halfStoreTile);
            set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
            wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        }
    }
    PublishGmWrites();
}

V4_FORCE_INLINE_AICORE void RunCombinePutContiguous(__gm__ half* dstPtr, __gm__ half* srcPtr, uint32_t bytes)
{
    const uint32_t elems = bytes / sizeof(half);
    constexpr uint32_t tileElems = kCombineTileAlignBytes / sizeof(half);
    for (uint32_t offset = 0; offset < elems; offset += tileElems) {
        uint32_t cur = elems - offset;
        if (cur > tileElems) {
            cur = tileElems;
        }
        ShapeDyn shape(1, 1, 1, 1, cur);
        StrideDyn stride(cur, cur, cur, cur, 1);
        CombineGlobal dstG(dstPtr + offset, shape, stride);
        CombineGlobal srcG(srcPtr + offset, shape, stride);
        CombineTile tile(1, cur);
        TASSIGN(tile, 0x0);
        pto::comm::TPUT(dstG, srcG, tile);
        SynchronizeCombinePipe();
    }
    dsb(DSB_DDR);
}

V4_FORCE_INLINE_AICORE void RunCombinePutRow(__gm__ half* dstPtr, __gm__ half* srcPtr, uint32_t outputBytes)
{
    RunCombinePutContiguous(dstPtr, srcPtr, outputBytes);
}

V4_FORCE_INLINE_AICORE void RunCombineCopyContiguous(__gm__ half* dstPtr, __gm__ half* srcPtr, uint32_t bytes)
{
    const uint32_t elems = bytes / sizeof(half);
    constexpr uint32_t tileElems = kCombineTileAlignBytes / sizeof(half);
    const uint32_t fullElems = (elems / tileElems) * tileElems;
    CombineTile pingTile(1, tileElems);
    CombineTile pongTile(1, tileElems);
    TASSIGN(pingTile, 0x0);
    TASSIGN(pongTile, kCombineTileAlignBytes);
    for (uint32_t offset = 0; offset < fullElems; offset += tileElems) {
        const uint32_t chunk = offset / tileElems;
        CombineTile& tile = (chunk & 1) == 0 ? pingTile : pongTile;
        const uint32_t event = (chunk & 1) == 0 ? EVENT_ID0 : EVENT_ID1;
        ShapeDyn shape(1, 1, 1, 1, tileElems);
        StrideDyn stride(tileElems, tileElems, tileElems, tileElems, 1);
        CombineGlobal dstG(dstPtr + offset, shape, stride);
        CombineGlobal srcG(srcPtr + offset, shape, stride);
        pto::TLOAD(tile, srcG);
        set_flag(PIPE_MTE2, PIPE_MTE3, event);
        wait_flag(PIPE_MTE2, PIPE_MTE3, event);
        pto::TSTORE(dstG, tile);
        set_flag(PIPE_MTE3, PIPE_MTE2, event);
        wait_flag(PIPE_MTE3, PIPE_MTE2, event);
        SynchronizeCombinePipe();
    }
    if (fullElems < elems) {
        const uint32_t cur = elems - fullElems;
        ShapeDyn shape(1, 1, 1, 1, cur);
        StrideDyn stride(cur, cur, cur, cur, 1);
        CombineGlobal dstG(dstPtr + fullElems, shape, stride);
        CombineGlobal srcG(srcPtr + fullElems, shape, stride);
        CombineTile tile(1, cur);
        TASSIGN(tile, 0x0);
        pto::TLOAD(tile, srcG);
        set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        pto::TSTORE(dstG, tile);
        SynchronizeCombinePipe();
    }
    dsb(DSB_DDR);
}

V4_FORCE_INLINE_AICORE void RunCombineCopyRow(__gm__ half* dstPtr, __gm__ half* srcPtr, uint32_t outputBytes)
{
    RunCombineCopyContiguous(dstPtr, srcPtr, outputBytes);
}

V4_FORCE_INLINE_AICORE uint32_t CombineDoneSlot(uint32_t srcRank, uint32_t dstRowBegin)
{
    constexpr uint32_t doneSlotElems = kCombineTileAlignBytes / sizeof(int32_t);
    constexpr uint32_t combineDoneBase = 16384U;
    return combineDoneBase + (srcRank * 64U + dstRowBegin) * doneSlotElems;
}

V4_FORCE_INLINE_AICORE volatile __gm__ int32_t* IncomingCompletionSignal(volatile __gm__ int32_t* signalBase,
                                                                         const __gm__ CombinePushTask* task)
{
    const bool useOwnerSummary = (task->taskFlags & kCombineTaskPublishOwnerCompletion) != 0;
    return useOwnerSummary
        ? signalBase + task->ownerCompletionIndex
        : signalBase + CombineDoneSlot(task->srcRank, task->dstRowBegin);
}

V4_FORCE_INLINE_AICORE void RunCombinePutDone(__gm__ int32_t* dstPtr, __gm__ int32_t* srcPtr)
{
    constexpr uint32_t doneSlotElems = kCombineTileAlignBytes / sizeof(int32_t);
    ShapeDyn shape(1, 1, 1, 1, doneSlotElems);
    StrideDyn stride(doneSlotElems, doneSlotElems, doneSlotElems, doneSlotElems, 1);
    DoneGlobal dstG(dstPtr, shape, stride);
    DoneGlobal srcG(srcPtr, shape, stride);
    DoneTile tile(1, doneSlotElems);
    TASSIGN(tile, 0x0);
    pto::comm::TPUT(dstG, srcG, tile);
    PublishGmWrites();
}

V4_FORCE_INLINE_AICORE void WaitGmm2Ready(__gm__ RemoteWindowContext* ctx,
                                           uint32_t taskFlags,
                                           uint32_t gmm2TileReadyIndex,
                                           uint32_t readyRows,
                                           uint32_t rowCount)
{
    if ((taskFlags & kCombineTaskWaitGmm2Ready) == 0) {
        return;
    }
    auto* signalBase = LocalSignalBaseVolatile(ctx);
    volatile __gm__ int32_t* ready = signalBase + gmm2TileReadyIndex;
    const uint32_t expectedRows = readyRows == 0 ? rowCount : readyRows;
    WaitVisibleSignal(ready, static_cast<int32_t>(expectedRows));
}

V4_FORCE_INLINE_AICORE void RunCombinePushTaskBody(__gm__ RemoteWindowContext* ctx,
                                                   __gm__ CombineOnlyParams* params,
                                                   const __gm__ CombinePushTask* task)
{
    auto* src = LocalPartialHalf(params, task->srcPayloadOffsetBytes);
    const uint32_t elems = task->outputBytes / sizeof(half);
    const uint32_t outputElemsPerRow = params->outputElems;
    MakeLocalPartialVisible(src, elems);
    if (params->perTokenScale2 != 0 && outputElemsPerRow != 0) {
        auto* scale2Base = reinterpret_cast<__gm__ float*>(params->perTokenScale2);
        ScaleRowsInPlace(src, scale2Base + task->srcRowBegin, task->rowCount, outputElemsPerRow);
    }
    if (task->srcRank == task->dstRank) {
        RunCombineCopyRow(LocalCombineHalf(ctx, task->dstPayloadOffsetBytes), src, task->outputBytes);
    } else {
        RunCombinePutRow(RemoteCombineHalf(ctx, task->dstRank, task->dstPayloadOffsetBytes), src, task->outputBytes);
    }
    auto* doneSrc = DoneValue(params);
    auto* doneDst = RemoteSignalInt32(ctx, task->dstRank, CombineDoneSlot(task->srcRank, task->dstRowBegin));
    RunCombinePutDone(doneDst, doneSrc);
    if ((task->taskFlags & kCombineTaskPublishOwnerCompletion) != 0) {
        RunCombinePutDone(RemoteSignalInt32(ctx, task->dstRank, task->ownerCompletionIndex), doneSrc);
    }
}

V4_FORCE_INLINE_AICORE void RunCombinePushTask(__gm__ RemoteWindowContext* ctx,
                                               __gm__ CombineOnlyParams* params,
                                               const __gm__ CombinePushTask* task)
{
    WaitGmm2Ready(ctx, task->taskFlags, task->gmm2TileReadyIndex, task->gmm2ReadyRows, task->rowCount);
    RunCombinePushTaskBody(ctx, params, task);
}

V4_FORCE_INLINE_AICORE void RunCombineRangeTaskBody(__gm__ RemoteWindowContext* ctx,
                                                    __gm__ CombineOnlyParams* params,
                                                    const __gm__ CombineRangeTask* task)
{
    auto* src = LocalPartialHalf(params, task->srcGmm2OffsetBytes);
    const uint32_t rangeBytes = task->rowCount * task->outputBytes;
    const uint32_t rangeElems = rangeBytes / sizeof(half);
    const uint32_t outputElemsPerRow = params->outputElems;
    MakeLocalPartialVisible(src, rangeElems);
    if (params->perTokenScale2 != 0 && outputElemsPerRow != 0) {
        auto* scale2Base = reinterpret_cast<__gm__ float*>(params->perTokenScale2);
        const uint32_t rowBegin = static_cast<uint32_t>(task->srcGmm2OffsetBytes / task->outputBytes);
        ScaleRowsInPlace(src, scale2Base + rowBegin, task->rowCount, outputElemsPerRow);
    }
    if (task->srcRank == task->ownerRank) {
        RunCombineCopyContiguous(LocalCombineHalf(ctx, task->dstCombineOffsetBytes), src, rangeBytes);
    } else {
        RunCombinePutContiguous(RemoteCombineHalf(ctx, task->ownerRank, task->dstCombineOffsetBytes), src, rangeBytes);
    }
    auto* doneSrc = DoneValue(params);
    if ((task->taskFlags & kCombineTaskPublishOwnerCompletion) != 0) {
        const uint32_t firstOwnerOrdinal = static_cast<uint32_t>(task->dstCombineOffsetBytes / task->outputBytes);
        for (uint32_t row = 0; row < task->rowCount; ++row) {
            constexpr uint32_t doneSlotElems = kCombineTileAlignBytes / sizeof(int32_t);
            constexpr uint32_t ownerSummaryBase = 20480U + 32U * 64U;
            const uint32_t ownerIndex = ownerSummaryBase + task->ownerRank * 64U * doneSlotElems + (firstOwnerOrdinal + row) * doneSlotElems;
            RunCombinePutDone(RemoteSignalInt32(ctx, task->ownerRank, ownerIndex), doneSrc);
        }
    }
}

V4_FORCE_INLINE_AICORE void RunCombineRangeTask(__gm__ RemoteWindowContext* ctx,
                                                __gm__ CombineOnlyParams* params,
                                                const __gm__ CombineRangeTask* task)
{
    WaitGmm2Ready(ctx, task->taskFlags, task->gmm2TileReadyIndex, task->gmm2ReadyRows, task->rowCount);
    RunCombineRangeTaskBody(ctx, params, task);
}

V4_FORCE_INLINE_AICORE uint32_t CombineWorkBlockCount(__gm__ CombineOnlyParams* params)
{
    if (params->phase != 1 && params->phase != 3) {
        return 0;
    }
    return params->rangeTaskCount != 0 ? params->rangeTaskCount : params->taskCount;
}

V4_FORCE_INLINE_AICORE void RunCombinePipelineWorkBlock(__gm__ RemoteWindowContext* ctx,
                                                         __gm__ CombineOnlyParams* params,
                                                         uint32_t workIndex)
{
    if (params->phase != 1 && params->phase != 3) {
        return;
    }
    auto* tasks = reinterpret_cast<__gm__ CombinePushTask*>(params->combineTasks);
    auto* rangeTasks = reinterpret_cast<__gm__ CombineRangeTask*>(params->combineRangeTasks);
    if (params->rangeTaskCount != 0) {
        if (workIndex < params->rangeTaskCount) {
            RunCombineRangeTaskBody(ctx, params, rangeTasks + workIndex);
        }
    } else if (workIndex < params->taskCount) {
        RunCombinePushTaskBody(ctx, params, tasks + workIndex);
    }
}

V4_FORCE_INLINE_AICORE void RunCombineRestoreBlock(__gm__ RemoteWindowContext* ctx,
                                                    __gm__ CombineOnlyParams* params,
                                                    const __gm__ StandaloneKernelTilingData* tiling)
{
    if (params->phase != 2 && params->phase != 3) {
        return;
    }
    auto* incomingTasks = reinterpret_cast<__gm__ CombinePushTask*>(params->incomingCombineTasks);
    auto* signalBase = LocalSignalBaseVolatile(ctx);
    for (uint32_t i = 0; i < params->incomingTaskCount; ++i) {
        const auto* incoming = incomingTasks + i;
        WaitVisibleSignal(IncomingCompletionSignal(signalBase, incoming), static_cast<int32_t>(incoming->completionEpoch));
    }
    v2_restore::RunCombineRestorePhase(ctx, params, tiling->outputBytes);
}

V4_FORCE_INLINE_AICORE void RunCombineOnlyKernel(__gm__ RemoteWindowContext* ctx,
                                                 __gm__ CombineOnlyParams* params,
                                                 const __gm__ StandaloneKernelTilingData* tiling)
{
    if (tiling->mode != static_cast<uint32_t>(KernelMode::CombineOnly)) {
        return;
    }
    const uint32_t blockIdx = get_block_idx();
    auto* tasks = reinterpret_cast<__gm__ CombinePushTask*>(params->combineTasks);
    auto* rangeTasks = reinterpret_cast<__gm__ CombineRangeTask*>(params->combineRangeTasks);
    if (params->phase == 1 || params->phase == 3) {
        if (params->rangeTaskCount != 0) {
            if (blockIdx < params->rangeTaskCount) {
                const auto* task = rangeTasks + blockIdx;
                if (task->srcRank != task->ownerRank) {
                    RunCombineRangeTask(ctx, params, task);
                }
            }
            if (blockIdx == 0) {
                for (uint32_t i = 0; i < params->rangeTaskCount; ++i) {
                    const auto* task = rangeTasks + i;
                    if (task->srcRank == task->ownerRank) {
                        RunCombineRangeTask(ctx, params, task);
                    }
                }
            }
        } else if (blockIdx < params->taskCount) {
            RunCombinePushTask(ctx, params, tasks + blockIdx);
        }
        if (params->phase == 1) {
            return;
        }
    }
    if (blockIdx == 0) {
        RunCombineRestoreBlock(ctx, params, tiling);
    }
}

}  // namespace v2_combine

#endif
