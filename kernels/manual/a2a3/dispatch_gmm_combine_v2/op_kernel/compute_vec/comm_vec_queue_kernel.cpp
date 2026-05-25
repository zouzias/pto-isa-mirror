#include "kernel_operator.h"
#include "../../kernel_launch.hpp"
#include "../dispatch_ffn_combine_tiling.h"
#include "../protocol/remote_window.hpp"
#include "../protocol/task_plan.hpp"
#include "../protocol/pipeline_queue.hpp"
#include "../protocol/pto_sync_bridge.hpp"
#include "../combine/combine_push.hpp"
#include "../dispatch/dispatch_pull.hpp"
#include "swiglu_fp16_vec.hpp"

namespace {

using v2_compute_vec::kComputeTileRows;
constexpr uint32_t kGmm1OutputElems = v2_compute_vec::kGmm1OutputElems;

template <typename T>
__aicore__ inline __attribute__((always_inline)) void MakeRowsVisible(__gm__ T* ptr,
                                                                      uint32_t rows,
                                                                      uint32_t strideElems,
                                                                      uint32_t visibleElems)
{
    const uint32_t visibleBytes = visibleElems * sizeof(T);
    for (uint32_t localRow = 0; localRow < rows; ++localRow) {
        v2_pto_sync::MakeCacheRangeVisible(
            (__gm__ void*)(ptr + static_cast<uint64_t>(localRow) * strideElems), visibleBytes);
    }
    v2_pto_sync::SynchronizePipe();
}

__aicore__ inline __attribute__((always_inline)) bool IsDispatchTileReady(__gm__ int32_t* signalBase,
                                                                            const __gm__ ComputeTileMeta* computeTiles,
                                                                            const __gm__ ExpertSourceSegmentMeta* sourceSegments,
                                                                            uint32_t tileIndex,
                                                                            uint32_t computeTileCount)
{
    if (signalBase == nullptr || computeTiles == nullptr || sourceSegments == nullptr || tileIndex >= computeTileCount) {
        return true;
    }
    const __gm__ ComputeTileMeta* tile = computeTiles + tileIndex;
    for (uint32_t segmentIndex = tile->sourceSegmentBegin; segmentIndex < tile->sourceSegmentEnd; ++segmentIndex) {
        const __gm__ ExpertSourceSegmentMeta* segment = sourceSegments + segmentIndex;
        if (!v2_pto_sync::TestWindowReady(
                v2_pto_sync::WindowSignal(signalBase, segment->doneSignalIndex), 1)) {
            return false;
        }
    }
    return true;
}

__aicore__ inline __attribute__((always_inline)) bool TryProduceDispatchTile(
    __gm__ int8_t* quantInput,
    __gm__ int32_t* signalBase,
    const __gm__ ComputeTileMeta* computeTiles,
    const __gm__ ExpertSourceSegmentMeta* expertSourceSegments,
    __gm__ PipelineQueueSet* dispatchToGmm1Queue,
    PipelineProducerCursor& dispatchCursor,
    uint32_t tile,
    uint32_t computeTileCount,
    uint32_t rowCount,
    uint32_t hiddenK)
{
    if (tile >= computeTileCount ||
        !IsDispatchTileReady(signalBase, computeTiles, expertSourceSegments, tile, computeTileCount)) {
        return false;
    }
    const uint32_t rowBegin = tile * kComputeTileRows;
    if (rowBegin >= rowCount) {
        return false;
    }
    const uint32_t rows = rowCount - rowBegin < kComputeTileRows ? rowCount - rowBegin : kComputeTileRows;

    PipelineQueueEntry entry = v2_pipeline_queue::MakeTileEntry(tile,
                                                                rowBegin,
                                                                rowBegin + rows,
                                                                computeTiles,
                                                                computeTileCount);
    if (!v2_pipeline_queue::PublishNext(dispatchToGmm1Queue, dispatchCursor, entry)) {
        return false;
    }
    return true;
}

__aicore__ inline __attribute__((always_inline)) bool TryConsumeGmm1AndProduceSwiGlu(
    __gm__ half* gmm1Out,
    __gm__ float* perTokenScale1,
    __gm__ int8_t* swigluQ,
    __gm__ float* scale2,
    __gm__ PipelineQueueSet* gmm1ToSwiGluQueue,
    __gm__ PipelineQueueSet* swiGluToGmm2Queue,
    PipelineConsumerHeads& gmm1Heads,
    PipelineProducerCursor& swigluCursor,
    uint32_t gmm1N,
    uint32_t scaleStrideBytes)
{
    PipelineQueueEntry entry{};
    uint32_t producerId = 0;
    const uint32_t swigluProducerCount = v2_pipeline_queue::ProducerCount(swiGluToGmm2Queue);
    if (!v2_pipeline_queue::TryPopRoundRobinOwnedByTileModulo(gmm1ToSwiGluQueue,
                                                              gmm1Heads,
                                                              entry,
                                                              producerId,
                                                              swigluCursor.producerId,
                                                              swigluProducerCount)) {
        return false;
    }
    const uint32_t tile = entry.windowId;
    const uint32_t rowBegin = entry.rowBegin;
    const uint32_t rows = entry.rowEnd - entry.rowBegin;
    const uint32_t k2Total = gmm1N / 2;
    auto* tileGmm1 = gmm1Out + static_cast<uint64_t>(tile) * kComputeTileRows * gmm1N;
    auto* tileSwiGluQ = swigluQ + static_cast<uint64_t>(tile) * kComputeTileRows * k2Total;
    auto* tileScale2 = scale2 + static_cast<uint64_t>(tile) * kComputeTileRows;
    MakeRowsVisible(tileGmm1, rows, gmm1N, gmm1N);

    auto* tilePerTokenScale1 = reinterpret_cast<__gm__ float*>(
        reinterpret_cast<__gm__ uint8_t*>(perTokenScale1) + static_cast<uint64_t>(rowBegin) * scaleStrideBytes);
    v2_compute_vec::RunSwiGluFp16TilePto(tileGmm1, tilePerTokenScale1, tileSwiGluQ, tileScale2, rows, gmm1N);
    v2_pto_sync::PublishGmWrites();
    v2_pipeline_queue::PublishNext(swiGluToGmm2Queue, swigluCursor, entry);
    return true;
}

__aicore__ inline __attribute__((always_inline)) bool TryConsumeGmm2Ready(
    __gm__ PipelineQueueSet* gmm2ToCombineQueue,
    PipelineConsumerHeads& gmm2Heads)
{
    PipelineQueueEntry entry{};
    uint32_t producerId = 0;
    return v2_pipeline_queue::TryPopRoundRobin(gmm2ToCombineQueue, gmm2Heads, entry, producerId);
}

__aicore__ inline __attribute__((always_inline)) bool GetCombineWorkRows(__gm__ CombineOnlyParams* params,
                                                                          uint32_t workIndex,
                                                                          uint32_t& rowBegin,
                                                                          uint32_t& rowEnd)
{
    if (params == nullptr || (params->phase != 1 && params->phase != 3)) {
        return false;
    }
    if (params->rangeTaskCount != 0) {
        if (workIndex >= params->rangeTaskCount) {
            return false;
        }
        auto* rangeTasks = reinterpret_cast<__gm__ CombineRangeTask*>(params->combineRangeTasks);
        const __gm__ CombineRangeTask* task = rangeTasks + workIndex;
        rowBegin = static_cast<uint32_t>(task->srcGmm2OffsetBytes / task->outputBytes);
        rowEnd = rowBegin + task->rowCount;
        return true;
    }
    if (workIndex >= params->taskCount) {
        return false;
    }
    auto* tasks = reinterpret_cast<__gm__ CombinePushTask*>(params->combineTasks);
    const __gm__ CombinePushTask* task = tasks + workIndex;
    rowBegin = static_cast<uint32_t>(task->srcPayloadOffsetBytes / task->outputBytes);
    rowEnd = rowBegin + task->rowCount;
    return true;
}

__aicore__ inline __attribute__((always_inline)) bool QueueEntryCoversRows(const PipelineQueueEntry& entry,
                                                                           uint32_t rowBegin,
                                                                           uint32_t rowEnd)
{
    return entry.rowBegin <= rowBegin && entry.rowEnd >= rowEnd;
}

__aicore__ inline __attribute__((always_inline)) bool WaitGmm2QueueEntryForCombineWork(
    __gm__ PipelineQueueSet* gmm2ToCombineQueue,
    __gm__ CombineOnlyParams* combineParams,
    uint32_t workIndex,
    PipelineQueueEntry& readyEntry)
{
    uint32_t rowBegin = 0;
    uint32_t rowEnd = 0;
    if (!GetCombineWorkRows(combineParams, workIndex, rowBegin, rowEnd)) {
        return false;
    }
    PipelineConsumerHeads localHeads{};
    v2_pipeline_queue::InitConsumerHeads(localHeads);
    uint32_t producerId = 0;
    while (true) {
        if (v2_pipeline_queue::TryPopRoundRobin(gmm2ToCombineQueue, localHeads, readyEntry, producerId)) {
            if (QueueEntryCoversRows(readyEntry, rowBegin, rowEnd)) {
                return true;
            }
        } else {
            v2_pto_sync::SynchronizePipe();
        }
    }
}

struct GroupScheduleCursor {
    uint32_t nextDispatchRange[kPipelineQueueMaxProducers]{};
    uint32_t nextComputeTile[kPipelineQueueMaxProducers]{};
    uint32_t nextGroup = 0;
    uint32_t producedTiles = 0;
};

__aicore__ inline __attribute__((always_inline)) uint32_t EffectiveGroupCount(uint32_t expertGroupCount)
{
    if (expertGroupCount == 0) {
        return 0;
    }
    return expertGroupCount > kPipelineQueueMaxProducers ? kPipelineQueueMaxProducers : expertGroupCount;
}

__aicore__ inline __attribute__((always_inline)) uint32_t FirstOwnedIndex(uint32_t begin,
                                                                                uint32_t end,
                                                                                uint32_t ownerId,
                                                                                uint32_t ownerCount)
{
    if (ownerCount == 0) {
        return end;
    }
    uint32_t index = begin;
    while (index < end && (index % ownerCount) != ownerId) {
        ++index;
    }
    return index;
}

__aicore__ inline __attribute__((always_inline)) uint32_t NextOwnedIndex(uint32_t index,
                                                                         uint32_t end,
                                                                         uint32_t ownerCount)
{
    if (ownerCount == 0) {
        return end;
    }
    index += ownerCount;
    return index > end ? end : index;
}

__aicore__ inline __attribute__((always_inline)) void InitGroupScheduleCursor(
    GroupScheduleCursor& cursor,
    const __gm__ ExpertGroupTaskRange* expertGroupRanges,
    uint32_t expertGroupCount,
    uint32_t ownerId,
    uint32_t ownerCount)
{
    const uint32_t groupCount = EffectiveGroupCount(expertGroupCount);
    for (uint32_t group = 0; group < kPipelineQueueMaxProducers; ++group) {
        if (group < groupCount && expertGroupRanges != nullptr) {
            cursor.nextDispatchRange[group] = FirstOwnedIndex(expertGroupRanges[group].dispatchRangeBegin,
                                                              expertGroupRanges[group].dispatchRangeEnd,
                                                              ownerId,
                                                              ownerCount);
            cursor.nextComputeTile[group] = FirstOwnedIndex(expertGroupRanges[group].computeTileBegin,
                                                            expertGroupRanges[group].computeTileEnd,
                                                            ownerId,
                                                            ownerCount);
        } else {
            cursor.nextDispatchRange[group] = 0;
            cursor.nextComputeTile[group] = 0;
        }
    }
    cursor.nextGroup = 0;
    cursor.producedTiles = 0;
}

__aicore__ inline __attribute__((always_inline)) bool HasPendingOwnedDispatchRange(
    const GroupScheduleCursor& cursor,
    const __gm__ ExpertGroupTaskRange* expertGroupRanges,
    uint32_t expertGroupCount)
{
    const uint32_t groupCount = EffectiveGroupCount(expertGroupCount);
    if (groupCount == 0 || expertGroupRanges == nullptr) {
        return false;
    }
    for (uint32_t group = 0; group < groupCount; ++group) {
        if (cursor.nextDispatchRange[group] < expertGroupRanges[group].dispatchRangeEnd) {
            return true;
        }
    }
    return false;
}

__aicore__ inline __attribute__((always_inline)) bool TryAdvanceOneGroup(
    GroupScheduleCursor& cursor,
    const __gm__ ExpertGroupTaskRange* expertGroupRanges,
    uint32_t expertGroupCount,
    __gm__ RemoteWindowContext* remoteCtx,
    __gm__ DispatchRangeTask* dispatchRanges,
    __gm__ int8_t* quantInput,
    __gm__ int32_t* signalBase,
    const __gm__ ComputeTileMeta* computeTiles,
    const __gm__ ExpertSourceSegmentMeta* expertSourceSegments,
    __gm__ PipelineQueueSet* dispatchToGmm1Queue,
    PipelineProducerCursor& dispatchCursor,
    uint32_t computeTileCount,
    uint32_t rowCount,
    uint32_t hiddenK,
    uint32_t ownerId,
    uint32_t ownerCount,
    bool allowTileProduce)
{
    const uint32_t groupCount = EffectiveGroupCount(expertGroupCount);
    if (groupCount == 0 || expertGroupRanges == nullptr) {
        return false;
    }
    const uint32_t startGroup = cursor.nextGroup >= groupCount ? 0 : cursor.nextGroup;
    for (uint32_t offset = 0; offset < groupCount; ++offset) {
        const uint32_t group = (startGroup + offset) % groupCount;
        const __gm__ ExpertGroupTaskRange* range = expertGroupRanges + group;
        bool progressed = false;
        if (cursor.nextDispatchRange[group] < range->dispatchRangeEnd) {
            v2_dispatch::RunDispatchRangeTask(remoteCtx, dispatchRanges + cursor.nextDispatchRange[group]);
            cursor.nextDispatchRange[group] = NextOwnedIndex(cursor.nextDispatchRange[group],
                                                             range->dispatchRangeEnd,
                                                             ownerCount);
            progressed = true;
        }
        while (allowTileProduce && cursor.nextComputeTile[group] < range->computeTileEnd) {
            const uint32_t tile = cursor.nextComputeTile[group];
            if (!TryProduceDispatchTile(quantInput,
                                        signalBase,
                                        computeTiles,
                                        expertSourceSegments,
                                        dispatchToGmm1Queue,
                                        dispatchCursor,
                                        tile,
                                        computeTileCount,
                                        rowCount,
                                        hiddenK)) {
                break;
            }
            cursor.nextComputeTile[group] = NextOwnedIndex(cursor.nextComputeTile[group],
                                                            range->computeTileEnd,
                                                            ownerCount);
            ++cursor.producedTiles;
            progressed = true;
        }
        if (progressed) {
            cursor.nextGroup = (group + 1) % groupCount;
            return true;
        }
    }
    cursor.nextGroup = (startGroup + 1) % groupCount;
    return false;
}

__aicore__ inline __attribute__((always_inline)) void RunCommVecQueueLoop(
    __gm__ RemoteWindowContext* remoteCtx,
    __gm__ DispatchRangeTask* dispatchRanges,
    __gm__ StandaloneKernelTilingData* dispatchTiling,
    __gm__ int8_t* quantInput,
    __gm__ float* scale1,
    __gm__ half* gmm1Out,
    __gm__ int8_t* swigluQ,
    __gm__ float* scale2,
    __gm__ half* gmm2Out,
    __gm__ int32_t* signalBase,
    const __gm__ ComputeTileMeta* computeTiles,
    const __gm__ ExpertSourceSegmentMeta* expertSourceSegments,
    const __gm__ ExpertGroupTaskRange* expertGroupRanges,
    __gm__ CombineOnlyParams* combineParams,
    __gm__ StandaloneKernelTilingData* combineTiling,
    __gm__ PipelineQueueSet* dispatchToGmm1Queue,
    __gm__ PipelineQueueSet* gmm1ToSwiGluQueue,
    __gm__ PipelineQueueSet* swiGluToGmm2Queue,
    __gm__ PipelineQueueSet* gmm2ToCombineQueue,
    uint32_t expertSafeRowsIndex,
    uint32_t requiredSafeRows,
    uint32_t expertGroupCount,
    uint32_t vecDispatchProducerCount,
    uint32_t vecSwiGluProducerCount,
    uint32_t computeTileCount,
    uint32_t rowCount,
    uint32_t inputStrideBytes,
    uint32_t scaleStrideBytes,
    uint32_t hiddenK,
    uint32_t gmm1N,
    uint32_t outputElems,
    uint32_t outputStrideElems)
{
    const uint32_t blockIdx = get_block_idx();
    const uint32_t tileCount = (rowCount + kComputeTileRows - 1) / kComputeTileRows;
    const uint32_t combineWorkBlockCount = v2_combine::CombineWorkBlockCount(combineParams);
    const uint32_t dispatchProducerCount = vecDispatchProducerCount == 0 ? 1U : vecDispatchProducerCount;
    const uint32_t swigluProducerCount = vecSwiGluProducerCount == 0 ? 1U : vecSwiGluProducerCount;
    const uint32_t dispatchBlockBegin = 0;
    const uint32_t dispatchBlockEnd = dispatchProducerCount;
    const uint32_t swigluBlockBegin = dispatchBlockEnd;
    const uint32_t swigluBlockEnd = swigluBlockBegin + swigluProducerCount;
    const uint32_t combineBlockBegin = swigluBlockEnd;
    const uint32_t combineBlockEnd = combineBlockBegin + combineWorkBlockCount;
    const uint32_t restoreBlock = combineBlockEnd;

    if (blockIdx >= dispatchBlockBegin && blockIdx < dispatchBlockEnd) {
        const uint32_t producerId = blockIdx - dispatchBlockBegin;
        PipelineProducerCursor dispatchCursor =
            v2_pipeline_queue::MakeProducerCursor(dispatchToGmm1Queue, producerId);
        const bool groupFirstEnabled = expertGroupRanges != nullptr && expertGroupCount != 0;
        GroupScheduleCursor groupCursor{};
        if (groupFirstEnabled) {
            InitGroupScheduleCursor(groupCursor, expertGroupRanges, expertGroupCount, producerId, dispatchProducerCount);
        }
        uint32_t dispatchRangeHead = producerId;
        uint32_t dispatchProduced = producerId;
        const uint32_t assignedTileCount = (tileCount + dispatchProducerCount - 1U - producerId) / dispatchProducerCount;
        while (dispatchCursor.tail < assignedTileCount ||
               (groupFirstEnabled && HasPendingOwnedDispatchRange(groupCursor, expertGroupRanges, expertGroupCount))) {
            bool progressed = false;
            if (groupFirstEnabled) {
                progressed = TryAdvanceOneGroup(groupCursor,
                                                expertGroupRanges,
                                                expertGroupCount,
                                                remoteCtx,
                                                dispatchRanges,
                                                quantInput,
                                                signalBase,
                                                computeTiles,
                                                expertSourceSegments,
                                                dispatchToGmm1Queue,
                                                dispatchCursor,
                                                computeTileCount,
                                                rowCount,
                                                hiddenK,
                                                producerId,
                                                dispatchProducerCount,
                                                dispatchCursor.tail < assignedTileCount);
            } else {
                if (dispatchRangeHead < dispatchTiling->taskCount) {
                    v2_dispatch::RunDispatchRangeTask(remoteCtx, dispatchRanges + dispatchRangeHead);
                    dispatchRangeHead += dispatchProducerCount;
                    progressed = true;
                }
                if (dispatchProduced < tileCount &&
                    TryProduceDispatchTile(quantInput,
                                           signalBase,
                                           computeTiles,
                                           expertSourceSegments,
                                           dispatchToGmm1Queue,
                                           dispatchCursor,
                                           dispatchProduced,
                                           computeTileCount,
                                           rowCount,
                                           hiddenK)) {
                    dispatchProduced += dispatchProducerCount;
                    progressed = true;
                }
            }
            if (!progressed) {
                v2_pto_sync::SynchronizePipe();
            }
        }
        return;
    }

    if (blockIdx >= swigluBlockBegin && blockIdx < swigluBlockEnd) {
        const uint32_t producerId = blockIdx - swigluBlockBegin;
        PipelineConsumerHeads gmm1Heads{};
        v2_pipeline_queue::InitConsumerHeads(gmm1Heads);
        PipelineProducerCursor swigluCursor =
            v2_pipeline_queue::MakeProducerCursor(swiGluToGmm2Queue, producerId);
        const uint32_t assignedTileCount = (tileCount + swigluProducerCount - 1U - producerId) / swigluProducerCount;
        while (swigluCursor.tail < assignedTileCount) {
            if (!TryConsumeGmm1AndProduceSwiGlu(gmm1Out,
                                               scale1,
                                               swigluQ,
                                               scale2,
                                               gmm1ToSwiGluQueue,
                                               swiGluToGmm2Queue,
                                               gmm1Heads,
                                               swigluCursor,
                                               gmm1N,
                                               scaleStrideBytes)) {
                v2_pto_sync::SynchronizePipe();
            }
        }
        return;
    }

    if (blockIdx >= combineBlockBegin && blockIdx < combineBlockEnd) {
        const uint32_t workIndex = blockIdx - combineBlockBegin;
        PipelineQueueEntry readyEntry{};
        if (WaitGmm2QueueEntryForCombineWork(gmm2ToCombineQueue, combineParams, workIndex, readyEntry)) {
            v2_pto_sync::PublishGmWrites();
        }
        v2_combine::RunCombinePipelineWorkBlock(remoteCtx, combineParams, workIndex);
        return;
    }

    if (blockIdx == restoreBlock) {
        v2_combine::RunCombineRestoreBlock(remoteCtx, combineParams, combineTiling);
    }
}

}  // namespace

extern "C" __global__ __aicore__ void dispatch_combine_moe_v2_comm_vec_queue_pipeline(
    __gm__ RemoteWindowContext* remoteCtx,
    __gm__ int8_t* quantInput,
    __gm__ float* scale1,
    __gm__ half* gmm1Out,
    __gm__ int8_t* swigluQ,
    __gm__ float* scale2,
    __gm__ half* gmm2Out,
    __gm__ int32_t* signalBase,
    const __gm__ ComputeTileMeta* computeTiles,
    const __gm__ ExpertSourceSegmentMeta* expertSourceSegments,
    const __gm__ ExpertGroupTaskRange* expertGroupRanges,
    __gm__ CombineOnlyParams* combineParams,
    __gm__ StandaloneKernelTilingData* combineTiling,
    __gm__ DispatchRangeTask* dispatchRanges,
    __gm__ StandaloneKernelTilingData* dispatchTiling,
    __gm__ PipelineQueueSet* dispatchToGmm1Queue,
    __gm__ PipelineQueueSet* gmm1ToSwiGluQueue,
    __gm__ PipelineQueueSet* swiGluToGmm2Queue,
    __gm__ PipelineQueueSet* gmm2ToCombineQueue,
    uint32_t expertSafeRowsIndex,
    uint32_t requiredSafeRows,
    uint32_t expertGroupCount,
    uint32_t vecDispatchProducerCount,
    uint32_t vecSwiGluProducerCount,
    uint32_t computeTileCount,
    uint32_t rowCount,
    uint32_t inputStrideBytes,
    uint32_t scaleStrideBytes,
    uint32_t hiddenK,
    uint32_t gmm1N,
    uint32_t outputElems,
    uint32_t outputStrideElems)
{
    RunCommVecQueueLoop(remoteCtx,
                        dispatchRanges,
                        dispatchTiling,
                        quantInput,
                        scale1,
                        gmm1Out,
                        swigluQ,
                        scale2,
                        gmm2Out,
                        signalBase,
                        computeTiles,
                        expertSourceSegments,
                        expertGroupRanges,
                        combineParams,
                        combineTiling,
                        dispatchToGmm1Queue,
                        gmm1ToSwiGluQueue,
                        swiGluToGmm2Queue,
                        gmm2ToCombineQueue,
                        expertSafeRowsIndex,
                        requiredSafeRows,
                        expertGroupCount,
                        vecDispatchProducerCount,
                        vecSwiGluProducerCount,
                        computeTileCount,
                        rowCount,
                        inputStrideBytes,
                        scaleStrideBytes,
                        hiddenK,
                        gmm1N,
                        outputElems,
                        outputStrideElems);
}

void launchCommVecQueuePipeline(const CommVecQueueLaunchArgs& args, void* stream)
{
    dispatch_combine_moe_v2_comm_vec_queue_pipeline<<<args.blockDim, nullptr, stream>>>(
        static_cast<RemoteWindowContext*>(args.remoteWindow),
        static_cast<int8_t*>(args.quantInput),
        static_cast<float*>(args.scale1),
        static_cast<half*>(args.gmm1Out),
        static_cast<int8_t*>(args.swigluQ),
        static_cast<float*>(args.scale2),
        static_cast<half*>(args.gmm2Out),
        static_cast<int32_t*>(args.signalBase),
        static_cast<ComputeTileMeta*>(args.computeTiles),
        static_cast<ExpertSourceSegmentMeta*>(args.expertSourceSegments),
        static_cast<ExpertGroupTaskRange*>(args.expertGroupRanges),
        static_cast<CombineOnlyParams*>(args.combineParams),
        static_cast<StandaloneKernelTilingData*>(args.combineTiling),
        static_cast<DispatchRangeTask*>(args.dispatchRanges),
        static_cast<StandaloneKernelTilingData*>(args.dispatchTiling),
        static_cast<PipelineQueueSet*>(args.dispatchToGmm1Queue),
        static_cast<PipelineQueueSet*>(args.gmm1ToSwiGluQueue),
        static_cast<PipelineQueueSet*>(args.swiGluToGmm2Queue),
        static_cast<PipelineQueueSet*>(args.gmm2ToCombineQueue),
        args.expertSafeRowsIndex,
        args.requiredSafeRows,
        args.expertGroupCount,
        args.vecDispatchProducerCount,
        args.vecSwiGluProducerCount,
        args.computeTileCount,
        args.rowCount,
        args.inputStrideBytes,
        args.scaleStrideBytes,
        args.hiddenK,
        args.gmm1N,
        args.outputElems,
        args.outputStrideElems);
}
