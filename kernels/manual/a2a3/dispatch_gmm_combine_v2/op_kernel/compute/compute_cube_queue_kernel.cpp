#include "kernel_operator.h"
#include "pto_gmm_block_int8.hpp"
#include "../../kernel_launch.hpp"
#include "../protocol/pipeline_queue.hpp"
#include "../protocol/pto_sync_bridge.hpp"

namespace {

constexpr uint32_t kComputeTileRows = 8;

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

__aicore__ inline __attribute__((always_inline)) void PublishGmm2WindowReady(__gm__ int32_t* signalBase,
                                                                              uint32_t readyIndex,
                                                                              uint32_t readyRows)
{
    if (signalBase == nullptr) {
        return;
    }
    v2_pto_sync::PublishWindowReadyStore(
        v2_pto_sync::WindowSignal(signalBase, readyIndex), static_cast<int32_t>(readyRows));
}

__aicore__ inline __attribute__((always_inline)) uint32_t AssignedTileCount(uint32_t tileCount,
                                                                            uint32_t producerId,
                                                                            uint32_t producerCount)
{
    if (producerCount == 0 || producerId >= producerCount || tileCount == 0) {
        return 0;
    }
    const uint32_t base = tileCount / producerCount;
    const uint32_t extra = tileCount % producerCount;
    return base + (producerId < extra ? 1U : 0U);
}

__aicore__ inline __attribute__((always_inline)) bool TryConsumeDispatchAndProduceGmm1(
    __gm__ int8_t* quantInput,
    __gm__ int8_t* weight1,
    __gm__ half* gmm1Out,
    __gm__ uint64_t* scale1Channel,
    __gm__ PipelineQueueSet* dispatchToGmm1Queue,
    __gm__ PipelineQueueSet* gmm1ToSwiGluQueue,
    PipelineConsumerHeads& dispatchHeads,
    PipelineProducerCursor& gmm1Cursor,
    uint32_t ownerCount,
    uint32_t hiddenK,
    uint32_t gmm1N)
{
    PipelineQueueEntry entry{};
    uint32_t producerId = 0;
    if (!v2_pipeline_queue::TryPopRoundRobinOwnedByTileModulo(dispatchToGmm1Queue,
                                                              dispatchHeads,
                                                              entry,
                                                              producerId,
                                                              gmm1Cursor.producerId,
                                                              ownerCount)) {
        return false;
    }
    const uint32_t tile = entry.windowId;
    const uint32_t rows = entry.rowEnd - entry.rowBegin;
    auto* tileInput = quantInput + static_cast<uint64_t>(tile) * kComputeTileRows * hiddenK;
    auto* tileGmm1 = gmm1Out + static_cast<uint64_t>(tile) * kComputeTileRows * gmm1N;
    MakeRowsVisible(tileInput, rows, hiddenK, hiddenK);

    const GmmTileRuntimeMeta tileMeta{
        .rowBegin = entry.rowBegin,
        .rowEnd = entry.rowEnd,
        .kBegin = entry.reserved1,
        .kEnd = entry.reserved2,
        .nBegin = entry.nBase,
        .nEnd = entry.nBase + entry.validN,
        .localExpert = entry.localExpert,
        .expertRangeIndex = entry.reserved0,
    };

    auto* expertScale = scale1Channel + static_cast<uint64_t>(entry.localExpert) * gmm1N;
    RunGmm1Int8Tile(tileInput, weight1, tileGmm1, expertScale, tileMeta, hiddenK, gmm1N, gmm1N);
    v2_pipeline_queue::PublishNext(gmm1ToSwiGluQueue, gmm1Cursor, entry);
    return true;
}

__aicore__ inline __attribute__((always_inline)) bool TryConsumeSwiGluAndProduceGmm2(
    __gm__ int8_t* swigluQ,
    __gm__ int8_t* weight2,
    __gm__ half* gmm2Out,
    __gm__ uint64_t* scale2Channel,
    __gm__ int32_t* signalBase,
    __gm__ PipelineQueueSet* swiGluToGmm2Queue,
    __gm__ PipelineQueueSet* gmm2ToCombineQueue,
    uint32_t gmm2TileReadyIndex,
    uint32_t gmm2TileReadyCount,
    uint32_t outputElems,
    uint32_t outputStrideElems,
    uint32_t weight2StrideElems,
    uint32_t gmm1N,
    PipelineConsumerHeads& swigluHeads,
    PipelineProducerCursor& gmm2Cursor,
    uint32_t ownerCount)
{
    PipelineQueueEntry entry{};
    uint32_t producerId = 0;
    if (!v2_pipeline_queue::TryPopRoundRobinOwnedByTileModulo(swiGluToGmm2Queue,
                                                              swigluHeads,
                                                              entry,
                                                              producerId,
                                                              gmm2Cursor.producerId,
                                                              ownerCount)) {
        return false;
    }
    const uint32_t tile = entry.windowId;
    const uint32_t rowBegin = entry.rowBegin;
    const uint32_t rows = entry.rowEnd - entry.rowBegin;
    const uint32_t k2Total = gmm1N / 2;
    auto* tileSwiGluQ = swigluQ + static_cast<uint64_t>(tile) * kComputeTileRows * k2Total;
    auto* tileOutput = gmm2Out + static_cast<uint64_t>(rowBegin) * outputStrideElems;
    MakeRowsVisible(tileSwiGluQ, rows, k2Total, k2Total);

    auto* expertScale2 = scale2Channel + static_cast<uint64_t>(entry.localExpert) * outputElems;
    const Gmm2TileRuntimeMeta tileMeta{
        .rowBegin = entry.rowBegin,
        .rowEnd = entry.rowEnd,
        .k2Begin = 0,
        .k2End = k2Total,
        .n2Begin = 0,
        .n2End = outputStrideElems,
        .localExpert = entry.localExpert,
        .expertRangeIndex = entry.reserved0,
    };
    RunGmm2Int8Tile(tileSwiGluQ,
                     weight2,
                     tileOutput,
                     expertScale2,
                     tileMeta,
                     k2Total,
                     outputElems,
                     weight2StrideElems,
                     outputStrideElems);
    v2_pto_sync::PublishGmWrites();
    if (tile < gmm2TileReadyCount) {
        PublishGmm2WindowReady(signalBase, gmm2TileReadyIndex + tile, rowBegin + rows);
    }
    v2_pipeline_queue::PublishNext(gmm2ToCombineQueue, gmm2Cursor, entry);
    return true;
}

__aicore__ inline __attribute__((always_inline)) void RunComputeCubeQueueLoop(
    __gm__ int8_t* quantInput,
    __gm__ int8_t* weight1,
    __gm__ half* gmm1Out,
    __gm__ uint64_t* scale1Channel,
    __gm__ int8_t* swigluQ,
    __gm__ int8_t* weight2,
    __gm__ half* gmm2Out,
    __gm__ uint64_t* scale2Channel,
    __gm__ int32_t* signalBase,
    __gm__ PipelineQueueSet* dispatchToGmm1Queue,
    __gm__ PipelineQueueSet* gmm1ToSwiGluQueue,
    __gm__ PipelineQueueSet* swiGluToGmm2Queue,
    __gm__ PipelineQueueSet* gmm2ToCombineQueue,
    uint32_t gmm2TileReadyIndex,
    uint32_t gmm2TileReadyCount,
    uint32_t rowCount,
    uint32_t outputElems,
    uint32_t outputStrideElems,
    uint32_t weight2StrideElems,
    uint32_t hiddenK,
    uint32_t gmm1N)
{
    const uint32_t tileCount = (rowCount + kComputeTileRows - 1) / kComputeTileRows;
    const uint32_t blockIdx = get_block_idx();
    const uint32_t producerCount = v2_pipeline_queue::ProducerCount(gmm1ToSwiGluQueue);
    if (producerCount == 0 || blockIdx >= producerCount) {
        return;
    }
    PipelineConsumerHeads dispatchHeads{};
    PipelineConsumerHeads swigluHeads{};
    v2_pipeline_queue::InitConsumerHeads(dispatchHeads);
    v2_pipeline_queue::InitConsumerHeads(swigluHeads);
    PipelineProducerCursor gmm1Cursor =
        v2_pipeline_queue::MakeProducerCursor(gmm1ToSwiGluQueue, blockIdx);
    PipelineProducerCursor gmm2Cursor =
        v2_pipeline_queue::MakeProducerCursor(gmm2ToCombineQueue, blockIdx);
    const uint32_t assignedTileCount = AssignedTileCount(tileCount, gmm1Cursor.producerId, producerCount);
    while (gmm2Cursor.tail < assignedTileCount) {
        if (gmm1Cursor.tail < assignedTileCount) {
            TryConsumeDispatchAndProduceGmm1(quantInput,
                                             weight1,
                                             gmm1Out,
                                             scale1Channel,
                                             dispatchToGmm1Queue,
                                             gmm1ToSwiGluQueue,
                                             dispatchHeads,
                                             gmm1Cursor,
                                             producerCount,
                                             hiddenK,
                                             gmm1N);
        }
        if (swigluHeads.totalConsumed < tileCount) {
            TryConsumeSwiGluAndProduceGmm2(swigluQ,
                                           weight2,
                                           gmm2Out,
                                           scale2Channel,
                                           signalBase,
                                           swiGluToGmm2Queue,
                                           gmm2ToCombineQueue,
                                           gmm2TileReadyIndex,
                                           gmm2TileReadyCount,
                                           outputElems,
                                           outputStrideElems,
                                           weight2StrideElems,
                                           gmm1N,
                                           swigluHeads,
                                           gmm2Cursor,
                                           producerCount);
        }
        v2_pto_sync::SynchronizePipe();
    }
    v2_pto_sync::PublishGmWrites();
}

}  // namespace

extern "C" __global__ __aicore__ void dispatch_combine_moe_v2_compute_cube_queue_pipeline(
    __gm__ int8_t* quantInput,
    __gm__ int8_t* weight1,
    __gm__ half* gmm1Out,
    __gm__ uint64_t* scale1Channel,
    __gm__ int8_t* swigluQ,
    __gm__ int8_t* weight2,
    __gm__ half* gmm2Out,
    __gm__ uint64_t* scale2Channel,
    __gm__ int32_t* signalBase,
    __gm__ PipelineQueueSet* dispatchToGmm1Queue,
    __gm__ PipelineQueueSet* gmm1ToSwiGluQueue,
    __gm__ PipelineQueueSet* swiGluToGmm2Queue,
    __gm__ PipelineQueueSet* gmm2ToCombineQueue,
    uint32_t gmm2TileReadyIndex,
    uint32_t gmm2TileReadyCount,
    uint32_t rowCount,
    uint32_t outputElems,
    uint32_t outputStrideElems,
    uint32_t weight2StrideElems,
    uint32_t hiddenK,
    uint32_t gmm1N)
{
    RunComputeCubeQueueLoop(quantInput,
                            weight1,
                            gmm1Out,
                            scale1Channel,
                            swigluQ,
                            weight2,
                            gmm2Out,
                            scale2Channel,
                            signalBase,
                            dispatchToGmm1Queue,
                            gmm1ToSwiGluQueue,
                            swiGluToGmm2Queue,
                            gmm2ToCombineQueue,
                            gmm2TileReadyIndex,
                            gmm2TileReadyCount,
                            rowCount,
                            outputElems,
                            outputStrideElems,
                            weight2StrideElems,
                            hiddenK,
                            gmm1N);
}

void launchComputeCubeQueuePipeline(const ComputeCubeQueueLaunchArgs& args, void* stream)
{
    dispatch_combine_moe_v2_compute_cube_queue_pipeline<<<args.blockDim, nullptr, stream>>>(
        static_cast<int8_t*>(args.quantInput),
        static_cast<int8_t*>(args.weight1),
        static_cast<half*>(args.gmm1Out),
        static_cast<uint64_t*>(args.scale1Channel),
        static_cast<int8_t*>(args.swigluQ),
        static_cast<int8_t*>(args.weight2),
        static_cast<half*>(args.gmm2Out),
        static_cast<uint64_t*>(args.scale2Channel),
        static_cast<int32_t*>(args.signalBase),
        static_cast<PipelineQueueSet*>(args.dispatchToGmm1Queue),
        static_cast<PipelineQueueSet*>(args.gmm1ToSwiGluQueue),
        static_cast<PipelineQueueSet*>(args.swiGluToGmm2Queue),
        static_cast<PipelineQueueSet*>(args.gmm2ToCombineQueue),
        args.gmm2TileReadyIndex,
        args.gmm2TileReadyCount,
        args.rowCount,
        args.outputElems,
        args.outputStrideElems,
        args.weight2StrideElems,
        args.hiddenK,
        args.gmm1N);
}
