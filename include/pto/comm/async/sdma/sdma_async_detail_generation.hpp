/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMM_ASYNC_SDMA_SDMA_ASYNC_DETAIL_GENERATION_HPP
#define PTO_COMM_ASYNC_SDMA_SDMA_ASYNC_DETAIL_GENERATION_HPP

#include "pto/comm/async/sdma/sdma_async_detail_basic.hpp"

namespace pto {
namespace comm {
namespace sdma {
namespace detail {

PTO_INTERNAL uint64_t LoadGenerationValue(__gm__ uint8_t* address, UbTmpBuf& tmpBuf)
{
    // Completion is written asynchronously by SDMA. Invalidate the cached line before every
    // poll so that Wait/Test observes the latest value instead of a stale AI Core cache entry.
    __asm__ __volatile__("");
    dcci((__gm__ void*)address, SINGLE_CACHE_LINE);
    __asm__ __volatile__("");
    return GetValue<uint64_t>(address, tmpBuf);
}

PTO_INTERNAL void MarkGenerationCompleted(uint64_t generation, uint64_t queueMask, SdmaRuntimeContext& runtimeCtx)
{
    uint32_t queue = 0U;
    while (queueMask != 0ULL) {
        if ((queueMask & 1ULL) != 0ULL && runtimeCtx.completedGeneration[queue] < generation) {
            runtimeCtx.completedGeneration[queue] = generation;
        }
        queueMask >>= 1U;
        ++queue;
    }
}

PTO_INTERNAL bool PollGeneration(
    __gm__ uint8_t* completionBase, uint64_t generation, uint64_t queueMask, UbTmpBuf& tmpBuf, bool blocking)
{
    uint32_t queue = 0U;
    while (queueMask != 0ULL) {
        if ((queueMask & 1ULL) == 0ULL) {
            queueMask >>= 1U;
            ++queue;
            continue;
        }
        uint32_t attempts = 0U;
        while (LoadGenerationValue(GenerationCompletion(completionBase, queue), tmpBuf) < generation) {
            if (!blocking || ++attempts >= kGenerationPollLimit) {
                return false;
            }
        }
        queueMask >>= 1U;
        ++queue;
    }
    return true;
}

PTO_INTERNAL bool EnsureRuntimeContext(const SdmaSession& session)
{
    SdmaRuntimeContext& runtimeCtx = session.runtimeCtx;
    if (runtimeCtx.initialized) {
        return true;
    }
    const SdmaExecContext& execCtx = session.execCtx;
    if (execCtx.contextGm == nullptr || !IsValidTmpBuffer(execCtx.tmpBuf)) {
        return false;
    }

    UbTmpBuf tmpBuf = execCtx.tmpBuf;
    runtimeCtx.completionBase = ResolveGenerationCompletionBase(execCtx);
    if (runtimeCtx.completionBase == nullptr) {
        return false;
    }
    for (uint32_t queue = 0U; queue < execCtx.baseConfig.queue_num; ++queue) {
        SetValue<uint64_t>(GenerationCompletion(runtimeCtx.completionBase, queue), tmpBuf, execCtx.syncId, 0ULL);
    }
    pipe_barrier(PIPE_ALL);
    runtimeCtx.nextGeneration = 0ULL;
    for (uint32_t queue = 0U; queue < kGenerationMaxQueues; ++queue) {
        runtimeCtx.completedGeneration[queue] = 0ULL;
        runtimeCtx.sqTail[queue] = 0U;
        runtimeCtx.sqHead[queue] = 0U;
    }
    for (uint32_t slot = 0U; slot < kGenerationPayloadDepth; ++slot) {
        runtimeCtx.payloadActiveQueues[slot] = 0U;
    }
    runtimeCtx.tailInitialized = false;
    runtimeCtx.initialized = true;
    return true;
}

PTO_INTERNAL void InitializeQueueRuntimeContext(
    __gm__ BatchWriteChannelInfo* channels, uint32_t queueNum, UbTmpBuf& tmpBuf, SdmaRuntimeContext& runtimeCtx)
{
    __asm__ __volatile__("");
    dcci((__gm__ void*)channels, ENTIRE_DATA_CACHE);
    __asm__ __volatile__("");
    dsb(DSB_DDR);
    for (uint32_t queue = 0U; queue < queueNum; ++queue) {
        __gm__ BatchWriteChannelInfo* channel = channels + queue;
        const uint64_t packedHeadTail = GetValue<uint64_t>((__gm__ uint8_t*)channel, tmpBuf);
        runtimeCtx.sqHead[queue] = static_cast<uint32_t>(packedHeadTail);
        runtimeCtx.sqTail[queue] = static_cast<uint32_t>(packedHeadTail >> 32U);
    }
    runtimeCtx.tailInitialized = true;
}

PTO_INTERNAL bool StoreGenerationPayload(
    uint64_t generation, uint32_t activeQueues, __gm__ uint8_t* payload, const SdmaSession& session, UbTmpBuf& tmpBuf)
{
    SdmaRuntimeContext& runtimeCtx = session.runtimeCtx;
    if (generation > kGenerationPayloadDepth) {
        // Generation numbers are contiguous, so this slot was last used by generation - depth.
        // Wait for that flag source to be consumed before overwriting the slot.
        const uint64_t oldGeneration = generation - kGenerationPayloadDepth;
        const uint32_t slot = static_cast<uint32_t>(generation % kGenerationPayloadDepth);
        const uint64_t oldQueueMask = ActiveQueuesToMask(runtimeCtx.payloadActiveQueues[slot]);
        bool knownComplete = true;
        uint64_t remaining = oldQueueMask;
        uint32_t queue = 0U;
        while (remaining != 0ULL) {
            if ((remaining & 1ULL) != 0ULL && runtimeCtx.completedGeneration[queue] < oldGeneration) {
                knownComplete = false;
                break;
            }
            remaining >>= 1U;
            ++queue;
        }
        if (!knownComplete) {
            if (!PollGeneration(runtimeCtx.completionBase, oldGeneration, oldQueueMask, tmpBuf, true)) {
                return false;
            }
            MarkGenerationCompleted(oldGeneration, oldQueueMask, runtimeCtx);
        }
    }

    *reinterpret_cast<volatile __gm__ uint64_t*>(payload) = generation;
    __asm__ __volatile__("" ::: "memory");
    runtimeCtx.payloadActiveQueues[generation % kGenerationPayloadDepth] = static_cast<uint8_t>(activeQueues);
    return true;
}

PTO_INTERNAL bool ValidateSinglePostSqCapacity(
    __gm__ BatchWriteChannelInfo* channels, const SdmaConfig& config, uint32_t activeQueues)
{
    for (uint32_t queue = 0U; queue < activeQueues; ++queue) {
        const uint32_t dataSqes = (config.iter_num - 1U - queue) / config.queue_num + 1U;
        const uint32_t sqesPerPost = dataSqes + 1U;
        const uint32_t sqDepth = channels[queue].sq_depth;
        if (sqDepth == 0U || sqesPerPost > sqDepth) {
            return false;
        }
    }
    return true;
}

PTO_INTERNAL void SubmitDataTransferSqes(
    __gm__ BatchWriteChannelInfo* channels, __gm__ uint8_t* recvBuffer, __gm__ uint8_t* sendBuffer, uint64_t opcode,
    const SdmaConfig& config, SdmaRuntimeContext& runtimeCtx)
{
    for (uint32_t index = 0U; index < config.iter_num; ++index) {
        const uint32_t queue = index % config.queue_num;
        uint32_t transferBytes = static_cast<uint32_t>(config.block_bytes);
        if (index + 1U == config.iter_num) {
            transferBytes = static_cast<uint32_t>(config.per_core_bytes - index * config.block_bytes);
        }
        __gm__ uint8_t* src = sendBuffer + config.comm_block_offset + index * config.block_bytes;
        __gm__ uint8_t* dst = recvBuffer + config.comm_block_offset + index * config.block_bytes;
        AddOneMemcpySqe(
            channels + queue, src, dst, opcode, transferBytes, runtimeCtx.sqTail[queue],
            runtimeCtx.sqTail[queue] - runtimeCtx.sqHead[queue]);
        runtimeCtx.sqTail[queue] = (runtimeCtx.sqTail[queue] + 1U) % channels[queue].sq_depth;
    }
    pipe_barrier(PIPE_ALL);
}

PTO_INTERNAL void RingDoorbell(
    __gm__ BatchWriteChannelInfo* channel, uint32_t sqTail, UbTmpBuf& tmpBuf, uint32_t syncId)
{
#ifdef PTO_NPU_ARCH_A5
    SetValue<uint32_t>((__gm__ uint8_t*)channel->sq_reg_base, tmpBuf, syncId, sqTail);
#else
    SetValue<uint32_t>((__gm__ uint8_t*)channel->sq_reg_base + 8U, tmpBuf, syncId, sqTail);
#endif
}

PTO_INTERNAL void PublishDataTransferSqes(
    __gm__ BatchWriteChannelInfo* channels, uint32_t activeQueues, const uint32_t* sqTail, UbTmpBuf& tmpBuf,
    uint32_t syncId)
{
    __asm__ __volatile__("");
    dcci((__gm__ void*)channels->sq_base, ENTIRE_DATA_CACHE);
    __asm__ __volatile__("");
    dsb(DSB_DDR);
    for (uint32_t queue = 0U; queue < activeQueues; ++queue) {
        RingDoorbell(channels + queue, sqTail[queue], tmpBuf, syncId);
    }
}

PTO_INTERNAL void SubmitFlagTransferSqes(
    __gm__ BatchWriteChannelInfo* channels, __gm__ uint8_t* payload, uint32_t activeQueues,
    SdmaRuntimeContext& runtimeCtx)
{
    for (uint32_t queue = 0U; queue < activeQueues; ++queue) {
        AddOneMemcpySqe(
            channels + queue, payload, GenerationCompletion(runtimeCtx.completionBase, queue), 0U, kGenerationFlagBytes,
            runtimeCtx.sqTail[queue], runtimeCtx.sqTail[queue] - runtimeCtx.sqHead[queue]);
        runtimeCtx.sqTail[queue] = (runtimeCtx.sqTail[queue] + 1U) % channels[queue].sq_depth;
    }
    pipe_barrier(PIPE_ALL);
}

PTO_INTERNAL void PersistSqTails(
    __gm__ BatchWriteChannelInfo* channels, uint32_t queueNum, const SdmaRuntimeContext& runtimeCtx, UbTmpBuf& tmpBuf,
    uint32_t syncId)
{
    for (uint32_t queue = 0U; queue < queueNum; ++queue) {
        const uint64_t packed = (static_cast<uint64_t>(runtimeCtx.sqTail[queue]) << 32U) | runtimeCtx.sqHead[queue];
        SetValue<uint64_t>((__gm__ uint8_t*)(channels + queue), tmpBuf, syncId, packed);
    }
}

PTO_INTERNAL void PublishFlagTransferSqes(
    __gm__ BatchWriteChannelInfo* channels, uint32_t activeQueues, const uint32_t* sqTail, UbTmpBuf& tmpBuf,
    uint32_t syncId)
{
    for (uint32_t queue = 0U; queue < activeQueues; ++queue) {
        const uint32_t flagIndex = (sqTail[queue] + channels[queue].sq_depth - 1U) % channels[queue].sq_depth;
        __gm__ BatchWriteItem* ring = reinterpret_cast<__gm__ BatchWriteItem*>(channels[queue].sq_base);
        __asm__ __volatile__("");
        dcci((__gm__ void*)(ring + flagIndex), SINGLE_CACHE_LINE);
        __asm__ __volatile__("");
    }
    dsb(DSB_DDR);
    for (uint32_t queue = 0U; queue < activeQueues; ++queue) {
        RingDoorbell(channels + queue, sqTail[queue], tmpBuf, syncId);
    }
}

PTO_INTERNAL AsyncEvent SdmaPostAsync(
    __gm__ uint8_t* recvBuffer, __gm__ uint8_t* sendBuffer, uint64_t opcode, uint64_t messageLen,
    const SdmaSession& session)
{
    const SdmaExecContext& execCtx = session.execCtx;
    if (!session.valid || recvBuffer == nullptr || sendBuffer == nullptr || !EnsureRuntimeContext(session)) {
        return {};
    }

    SdmaConfig config{};
    if (!BuildTransferConfig(execCtx.baseConfig, messageLen, config) || config.iter_num == 0U ||
        config.queue_num == 0U || config.queue_num > kGenerationMaxQueues ||
        execCtx.channelGroupIdx >= kSdmaMaxChannel / config.queue_num) {
        return {};
    }
    const uint32_t activeQueues = config.iter_num < config.queue_num ? config.iter_num : config.queue_num;

    __gm__ BatchWriteChannelInfo* channelBase =
        reinterpret_cast<__gm__ BatchWriteChannelInfo*>(execCtx.contextGm + sizeof(BatchWriteFlagInfo));
    __gm__ BatchWriteChannelInfo* channels = channelBase + execCtx.channelGroupIdx * config.queue_num;
    UbTmpBuf tmpBuf = execCtx.tmpBuf;
    SdmaRuntimeContext& runtimeCtx = session.runtimeCtx;
    if (!runtimeCtx.tailInitialized) {
        InitializeQueueRuntimeContext(channels, config.queue_num, tmpBuf, runtimeCtx);
    }
    if (!ValidateSinglePostSqCapacity(channels, config, activeQueues) ||
        runtimeCtx.nextGeneration >= kSdmaHandleGenerationMask) {
        return {};
    }

    const uint64_t generation = runtimeCtx.nextGeneration + 1ULL;
    __gm__ uint8_t* payload = GenerationPayload(ResolveGenerationPayloadBase(execCtx), generation);
    if (!StoreGenerationPayload(generation, activeQueues, payload, session, tmpBuf)) {
        return {};
    }
    uint64_t eventHandle = 0ULL;
    if (!EncodeSdmaEventHandle(generation, activeQueues, eventHandle)) {
        return {};
    }
    runtimeCtx.nextGeneration = generation;
    SubmitDataTransferSqes(channels, recvBuffer, sendBuffer, opcode, config, runtimeCtx);
    PublishDataTransferSqes(channels, activeQueues, runtimeCtx.sqTail, tmpBuf, execCtx.syncId);
    SubmitFlagTransferSqes(channels, payload, activeQueues, runtimeCtx);
    PersistSqTails(channels, config.queue_num, runtimeCtx, tmpBuf, execCtx.syncId);
    PublishFlagTransferSqes(channels, activeQueues, runtimeCtx.sqTail, tmpBuf, execCtx.syncId);
    return AsyncEvent(eventHandle, DmaEngine::SDMA);
}

PTO_INTERNAL bool SdmaEventCheck(uint64_t generation, uint64_t queueMask, const SdmaSession& session, bool blocking)
{
    if (generation == 0ULL || queueMask == 0ULL || !session.runtimeCtx.initialized) {
        return false;
    }
    UbTmpBuf tmpBuf = session.eventCtx.tmpBuf;
    if (!PollGeneration(session.runtimeCtx.completionBase, generation, queueMask, tmpBuf, blocking)) {
        return false;
    }
    MarkGenerationCompleted(generation, queueMask, session.runtimeCtx);
    return true;
}

PTO_INTERNAL bool SdmaTestEvent(uint64_t handle, const SdmaSession& session)
{
    uint64_t generation = 0ULL;
    uint32_t activeQueueNum = 0U;
    if (!DecodeSdmaEventHandle(handle, generation, activeQueueNum)) {
        return false;
    }
    const uint64_t queueMask = ActiveQueuesToMask(activeQueueNum);
    return SdmaEventCheck(generation, queueMask, session, false);
}

PTO_INTERNAL bool SdmaWaitEvent(uint64_t handle, const SdmaSession& session)
{
    uint64_t generation = 0ULL;
    uint32_t activeQueueNum = 0U;
    if (!DecodeSdmaEventHandle(handle, generation, activeQueueNum)) {
        return false;
    }
    const uint64_t queueMask = ActiveQueuesToMask(activeQueueNum);
    return SdmaEventCheck(generation, queueMask, session, true);
}

} // namespace detail
} // namespace sdma
} // namespace comm
} // namespace pto

#endif // PTO_COMM_ASYNC_SDMA_SDMA_ASYNC_DETAIL_GENERATION_HPP
