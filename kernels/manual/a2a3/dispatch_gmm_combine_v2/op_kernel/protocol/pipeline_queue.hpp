#pragma once

#include <cstdint>
#if !defined(__CCE_AICORE__)
#include <cstring>
#include <stdexcept>
#include <vector>
#endif

#include "task_plan.hpp"
#include "pto_sync_bridge.hpp"

static constexpr uint32_t kPipelineQueueMaxProducers = 8;
static constexpr uint32_t kPipelineQueueCapacity = 4096;
static constexpr uint32_t kPipelineQueueProducerId0 = 0;
static constexpr uint32_t kPipelineQueueFlagNone = 0;

struct alignas(kCombineTileAlignBytes) PipelineQueueEntry {
    uint32_t windowId = 0;
    uint32_t rowBegin = 0;
    uint32_t rowEnd = 0;
    uint32_t sourceSegmentBegin = 0;
    uint32_t sourceSegmentEnd = 0;
    uint32_t localExpert = 0;
    uint32_t nBase = 0;
    uint32_t validN = 0;
    uint32_t flags = 0;
    uint32_t reserved0 = 0;
    uint32_t reserved1 = 0;
    uint32_t reserved2 = 0;
    uint32_t reserved3 = 0;
    uint32_t reserved4 = 0;
    uint32_t reserved5 = 0;
    uint32_t reserved6 = 0;
};

struct alignas(kCombineTileAlignBytes) PipelineProducerQueue {
    int32_t count = 0;
    uint32_t expectedCount = 0;
    uint32_t capacity = kPipelineQueueCapacity;
    uint32_t reserved0 = 0;
    uint32_t reserved1 = 0;
    uint32_t reserved2 = 0;
    uint32_t reserved3 = 0;
    uint32_t reserved4 = 0;
    PipelineQueueEntry entries[kPipelineQueueCapacity];
};

struct PipelineQueueSet {
    uint32_t producerCount = 1;
    uint32_t expectedTotal = 0;
    uint32_t capacity = kPipelineQueueCapacity;
    uint32_t reserved0 = 0;
    uint32_t reserved1 = 0;
    uint32_t reserved2 = 0;
    uint32_t reserved3 = 0;
    uint32_t reserved4 = 0;
    PipelineProducerQueue producers[kPipelineQueueMaxProducers];
};

struct PipelineQueuePlan {
    uint32_t producerCount = 1;
    uint32_t capacity = kPipelineQueueCapacity;
    uint32_t dispatchToGmm1Expected = 0;
    uint32_t gmm1ToSwiGluExpected = 0;
    uint32_t swiGluToGmm2Expected = 0;
    uint32_t gmm2ToCombineExpected = 0;
};

struct PipelineProducerCursor {
    uint32_t producerId = 0;
    uint32_t tail = 0;
};

struct PipelineConsumerHeads {
    uint32_t heads[kPipelineQueueMaxProducers]{};
    uint32_t nextProducer = 0;
    uint32_t totalConsumed = 0;
};

#if !defined(__CCE_AICORE__)
inline uint32_t ClampPipelineQueueCapacity(uint32_t expectedTotal)
{
    if (expectedTotal == 0) {
        return 1;
    }
    return expectedTotal > kPipelineQueueCapacity ? kPipelineQueueCapacity : expectedTotal;
}

inline PipelineQueuePlan BuildPipelineQueuePlanHost(uint32_t windowCount,
                                                    uint32_t producerCount = 1)
{
    PipelineQueuePlan plan{};
    plan.producerCount = producerCount == 0 ? 1 :
        (producerCount > kPipelineQueueMaxProducers ? kPipelineQueueMaxProducers : producerCount);
    plan.capacity = ClampPipelineQueueCapacity(windowCount);
    plan.dispatchToGmm1Expected = windowCount;
    plan.gmm1ToSwiGluExpected = windowCount;
    plan.swiGluToGmm2Expected = windowCount;
    plan.gmm2ToCombineExpected = windowCount;
    return plan;
}

inline void InitPipelineQueueSetHost(PipelineQueueSet& queueSet,
                                     uint32_t producerCount,
                                     uint32_t expectedTotal,
                                     uint32_t capacity = kPipelineQueueCapacity)
{
    std::memset(&queueSet, 0, sizeof(queueSet));
    const uint32_t clampedProducerCount = producerCount == 0 ? 1 :
        (producerCount > kPipelineQueueMaxProducers ? kPipelineQueueMaxProducers : producerCount);
    const uint32_t clampedCapacity = capacity > kPipelineQueueCapacity ? kPipelineQueueCapacity : capacity;
    queueSet.producerCount = clampedProducerCount;
    queueSet.expectedTotal = expectedTotal;
    queueSet.capacity = clampedCapacity;
    const uint32_t baseExpected = expectedTotal / clampedProducerCount;
    uint32_t extraExpected = expectedTotal % clampedProducerCount;
    uint32_t expectedSum = 0;
    for (uint32_t producer = 0; producer < clampedProducerCount; ++producer) {
        const uint32_t producerExpected = baseExpected + (extraExpected != 0 ? 1U : 0U);
        if (producerExpected > clampedCapacity) {
            throw std::runtime_error("pipeline queue expected count exceeds producer capacity");
        }
        queueSet.producers[producer].expectedCount = producerExpected;
        queueSet.producers[producer].capacity = clampedCapacity;
        expectedSum += producerExpected;
        if (extraExpected != 0) {
            --extraExpected;
        }
    }
    if (expectedSum != expectedTotal) {
        throw std::runtime_error("pipeline queue expected count split mismatch");
    }
}

inline std::vector<unsigned char> MakePipelineQueueSetHost(uint32_t producerCount,
                                                           uint32_t expectedTotal,
                                                           uint32_t capacity = kPipelineQueueCapacity)
{
    PipelineQueueSet queueSet{};
    InitPipelineQueueSetHost(queueSet, producerCount, expectedTotal, capacity);
    std::vector<unsigned char> bytes(sizeof(PipelineQueueSet), 0);
    std::memcpy(bytes.data(), &queueSet, sizeof(queueSet));
    return bytes;
}
#endif

#if defined(__CCE_AICORE__)
#include "kernel_operator.h"
#include "pto/comm/pto_comm_inst.hpp"

#ifndef V4_FORCE_INLINE_AICORE
#define V4_FORCE_INLINE_AICORE inline __attribute__((always_inline)) __aicore__
#endif

namespace v2_pipeline_queue {

V4_FORCE_INLINE_AICORE PipelineQueueEntry MakeTileEntry(uint32_t tileIndex,
                                                        uint32_t rowBegin,
                                                        uint32_t rowEnd,
                                                        const __gm__ ComputeTileMeta* computeTiles,
                                                        uint32_t computeTileCount)
{
    PipelineQueueEntry entry{};
    entry.windowId = tileIndex;
    entry.rowBegin = rowBegin;
    entry.rowEnd = rowEnd;
    entry.flags = kPipelineQueueFlagNone;
    if (computeTiles != nullptr && tileIndex < computeTileCount) {
        const __gm__ ComputeTileMeta* tile = computeTiles + tileIndex;
        entry.sourceSegmentBegin = tile->sourceSegmentBegin;
        entry.sourceSegmentEnd = tile->sourceSegmentEnd;
        entry.localExpert = tile->localExpert;
        entry.nBase = tile->nBegin;
        entry.validN = tile->nEnd > tile->nBegin ? tile->nEnd - tile->nBegin : 0;
        entry.reserved0 = tile->expertRangeIndex;
        entry.reserved1 = tile->kBegin;
        entry.reserved2 = tile->kEnd;
        entry.reserved3 = tile->expandedRowBegin;
        entry.reserved4 = tile->scaleOffsetBegin;
    }
    return entry;
}

V4_FORCE_INLINE_AICORE __gm__ PipelineProducerQueue* ProducerQueue(__gm__ PipelineQueueSet* queueSet,
                                                                   uint32_t producerId)
{
    return queueSet->producers + producerId;
}

V4_FORCE_INLINE_AICORE uint32_t ProducerCount(__gm__ PipelineQueueSet* queueSet)
{
    if (queueSet == nullptr) {
        return 0;
    }
    const uint32_t producerCount = queueSet->producerCount;
    if (producerCount == 0) {
        return 1;
    }
    return producerCount > kPipelineQueueMaxProducers ? kPipelineQueueMaxProducers : producerCount;
}

V4_FORCE_INLINE_AICORE PipelineProducerCursor MakeProducerCursor(__gm__ PipelineQueueSet* queueSet,
                                                                 uint32_t producerId)
{
    PipelineProducerCursor cursor{};
    const uint32_t producerCount = ProducerCount(queueSet);
    cursor.producerId = producerCount == 0 ? 0 : (producerId % producerCount);
    cursor.tail = 0;
    return cursor;
}

V4_FORCE_INLINE_AICORE void InitConsumerHeads(PipelineConsumerHeads& heads)
{
    for (uint32_t producer = 0; producer < kPipelineQueueMaxProducers; ++producer) {
        heads.heads[producer] = 0;
    }
    heads.nextProducer = 0;
    heads.totalConsumed = 0;
}

V4_FORCE_INLINE_AICORE bool Publish(__gm__ PipelineQueueSet* queueSet,
                                    uint32_t producerId,
                                    uint32_t tail,
                                    const PipelineQueueEntry& entry)
{
    if (queueSet == nullptr || producerId >= ProducerCount(queueSet) || tail >= kPipelineQueueCapacity) {
        return false;
    }
    __gm__ PipelineProducerQueue* queue = ProducerQueue(queueSet, producerId);
    if (tail >= queue->capacity) {
        return false;
    }
    __gm__ PipelineQueueEntry* dst = queue->entries + tail;
    dst->windowId = entry.windowId;
    dst->rowBegin = entry.rowBegin;
    dst->rowEnd = entry.rowEnd;
    dst->sourceSegmentBegin = entry.sourceSegmentBegin;
    dst->sourceSegmentEnd = entry.sourceSegmentEnd;
    dst->localExpert = entry.localExpert;
    dst->nBase = entry.nBase;
    dst->validN = entry.validN;
    dst->flags = entry.flags;
    dst->reserved0 = entry.reserved0;
    dst->reserved1 = entry.reserved1;
    dst->reserved2 = entry.reserved2;
    dst->reserved3 = entry.reserved3;
    dst->reserved4 = entry.reserved4;
    dst->reserved5 = entry.reserved5;
    dst->reserved6 = entry.reserved6;
    v2_pto_sync::PublishEntryBeforeCount(reinterpret_cast<__gm__ void*>(dst),
                                         reinterpret_cast<__gm__ int32_t*>(&queue->count),
                                         static_cast<int32_t>(tail + 1));
    return true;
}

V4_FORCE_INLINE_AICORE void WaitCount(__gm__ PipelineProducerQueue* queue, int32_t expected)
{
    v2_pto_sync::WaitWindowReady(
        v2_pto_sync::WindowSignal(reinterpret_cast<__gm__ int32_t*>(&queue->count), 0), expected);
}

V4_FORCE_INLINE_AICORE bool PublishNext(__gm__ PipelineQueueSet* queueSet,
                                        PipelineProducerCursor& cursor,
                                        const PipelineQueueEntry& entry)
{
    if (!Publish(queueSet, cursor.producerId, cursor.tail, entry)) {
        return false;
    }
    ++cursor.tail;
    return true;
}

V4_FORCE_INLINE_AICORE bool TryPop(__gm__ PipelineQueueSet* queueSet,
                                   uint32_t producerId,
                                   uint32_t& head,
                                   PipelineQueueEntry& entry)
{
    if (queueSet == nullptr || producerId >= ProducerCount(queueSet) || head >= kPipelineQueueCapacity) {
        return false;
    }
    __gm__ PipelineProducerQueue* queue = ProducerQueue(queueSet, producerId);
    if (!v2_pto_sync::TestWindowReady(
            v2_pto_sync::WindowSignal(reinterpret_cast<__gm__ int32_t*>(&queue->count), 0),
            static_cast<int32_t>(head + 1))) {
        return false;
    }
    __gm__ PipelineQueueEntry* src = queue->entries + head;
    v2_pto_sync::MakeCacheLineVisibleAndSync(reinterpret_cast<__gm__ void*>(src));
    entry.windowId = src->windowId;
    entry.rowBegin = src->rowBegin;
    entry.rowEnd = src->rowEnd;
    entry.sourceSegmentBegin = src->sourceSegmentBegin;
    entry.sourceSegmentEnd = src->sourceSegmentEnd;
    entry.localExpert = src->localExpert;
    entry.nBase = src->nBase;
    entry.validN = src->validN;
    entry.flags = src->flags;
    entry.reserved0 = src->reserved0;
    entry.reserved1 = src->reserved1;
    entry.reserved2 = src->reserved2;
    entry.reserved3 = src->reserved3;
    entry.reserved4 = src->reserved4;
    entry.reserved5 = src->reserved5;
    entry.reserved6 = src->reserved6;
    ++head;
    return true;
}

V4_FORCE_INLINE_AICORE bool TryPopRoundRobin(__gm__ PipelineQueueSet* queueSet,
                                             PipelineConsumerHeads& heads,
                                             PipelineQueueEntry& entry,
                                             uint32_t& producerId)
{
    const uint32_t producerCount = ProducerCount(queueSet);
    if (producerCount == 0) {
        return false;
    }
    uint32_t startProducer = heads.nextProducer;
    if (startProducer >= producerCount) {
        startProducer = 0;
    }
    for (uint32_t offset = 0; offset < producerCount; ++offset) {
        const uint32_t candidate = (startProducer + offset) % producerCount;
        if (TryPop(queueSet, candidate, heads.heads[candidate], entry)) {
            producerId = candidate;
            heads.nextProducer = (candidate + 1) % producerCount;
            ++heads.totalConsumed;
            return true;
        }
    }
    return false;
}

V4_FORCE_INLINE_AICORE bool TryPopRoundRobinOwnedByTileModulo(__gm__ PipelineQueueSet* queueSet,
                                                              PipelineConsumerHeads& heads,
                                                              PipelineQueueEntry& entry,
                                                              uint32_t& producerId,
                                                              uint32_t ownerId,
                                                              uint32_t ownerCount)
{
    const uint32_t producerCount = ProducerCount(queueSet);
    if (producerCount == 0 || ownerCount == 0 || ownerId >= ownerCount) {
        return false;
    }
    uint32_t startProducer = heads.nextProducer;
    if (startProducer >= producerCount) {
        startProducer = 0;
    }
    for (uint32_t offset = 0; offset < producerCount; ++offset) {
        const uint32_t candidate = (startProducer + offset) % producerCount;
        uint32_t head = heads.heads[candidate];
        PipelineQueueEntry candidateEntry{};
        while (TryPop(queueSet, candidate, head, candidateEntry)) {
            if ((candidateEntry.windowId % ownerCount) == ownerId) {
                heads.heads[candidate] = head;
                entry = candidateEntry;
                producerId = candidate;
                heads.nextProducer = (candidate + 1) % producerCount;
                ++heads.totalConsumed;
                return true;
            }
        }
        heads.heads[candidate] = head;
    }
    return false;
}

V4_FORCE_INLINE_AICORE bool WaitPop(__gm__ PipelineQueueSet* queueSet,
                                    uint32_t producerId,
                                    uint32_t& head,
                                    PipelineQueueEntry& entry)
{
    if (queueSet == nullptr || producerId >= ProducerCount(queueSet) || head >= kPipelineQueueCapacity) {
        return false;
    }
    __gm__ PipelineProducerQueue* queue = ProducerQueue(queueSet, producerId);
    WaitCount(queue, static_cast<int32_t>(head + 1));
    __gm__ PipelineQueueEntry* src = queue->entries + head;
    v2_pto_sync::MakeCacheLineVisibleAndSync(reinterpret_cast<__gm__ void*>(src));
    entry.windowId = src->windowId;
    entry.rowBegin = src->rowBegin;
    entry.rowEnd = src->rowEnd;
    entry.sourceSegmentBegin = src->sourceSegmentBegin;
    entry.sourceSegmentEnd = src->sourceSegmentEnd;
    entry.localExpert = src->localExpert;
    entry.nBase = src->nBase;
    entry.validN = src->validN;
    entry.flags = src->flags;
    entry.reserved0 = src->reserved0;
    entry.reserved1 = src->reserved1;
    entry.reserved2 = src->reserved2;
    entry.reserved3 = src->reserved3;
    entry.reserved4 = src->reserved4;
    entry.reserved5 = src->reserved5;
    entry.reserved6 = src->reserved6;
    ++head;
    return true;
}

}  // namespace v2_pipeline_queue

#endif
