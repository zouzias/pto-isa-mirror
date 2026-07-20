/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMM_ASYNC_SDMA_SDMA_ASYNC_INTRIN_HPP
#define PTO_COMM_ASYNC_SDMA_SDMA_ASYNC_INTRIN_HPP

#include "pto/comm/async/sdma/sdma_types.hpp"
#include "pto/comm/comm_types.hpp"
#include "pto/comm/async_common/async_types.hpp"
#include "pto/pto-inst.hpp"
#include <cstddef>
#include <cstdint>

namespace pto {
namespace comm {
namespace sdma {

namespace detail {

static_assert(kSdmaEventSlotCount > 0, "SDMA_EVENT_SLOT_COUNT must be >= 1");

using UbTmpBuf = TmpBuffer;
constexpr uint32_t kGenerationCacheLineBytes = 64U;
constexpr uint32_t kGenerationFlagBytes = sizeof(uint64_t);
constexpr uint32_t kGenerationMaxQueues = kGenerationStateMaxQueues;
constexpr uint32_t kGenerationPayloadDepth = kGenerationStatePayloadDepth;
constexpr uint32_t kGenerationPayloadBytes = kGenerationPayloadDepth * kGenerationCacheLineBytes;
constexpr uint32_t kGenerationPollLimit = 100000U;
constexpr uint32_t kSdmaHandleQueueBits = 6U;
constexpr uint32_t kSdmaHandleGenerationBits = 64U - kSdmaHandleQueueBits;
constexpr uint64_t kSdmaHandleGenerationMask = (1ULL << kSdmaHandleGenerationBits) - 1ULL;
static_assert(kGenerationPayloadBytes == kSdmaPayloadBytesPerGroup);
static_assert(kGenerationMaxQueues == kSdmaMaxChannelGroups);

PTO_INTERNAL uint64_t ActiveQueuesToMask(uint32_t activeQueues)
{
    return activeQueues == 0U ? 0ULL : (1ULL << activeQueues) - 1ULL;
}

PTO_INTERNAL bool EncodeSdmaEventHandle(uint64_t generation, uint32_t activeQueueNum, uint64_t& handle)
{
    if (generation == 0ULL || generation > kSdmaHandleGenerationMask || activeQueueNum == 0U ||
        activeQueueNum > kGenerationMaxQueues) {
        handle = 0ULL;
        return false;
    }
    handle = (static_cast<uint64_t>(activeQueueNum) << kSdmaHandleGenerationBits) | generation;
    return true;
}

PTO_INTERNAL bool DecodeSdmaEventHandle(uint64_t handle, uint64_t& generation, uint32_t& activeQueueNum)
{
    generation = handle & kSdmaHandleGenerationMask;
    activeQueueNum = static_cast<uint32_t>(handle >> kSdmaHandleGenerationBits);
    if (generation == 0ULL || activeQueueNum == 0U || activeQueueNum > kGenerationMaxQueues) {
        generation = 0ULL;
        activeQueueNum = 0U;
        return false;
    }
    return true;
}

PTO_INTERNAL bool MakeSdmaTmpLocal(__ubuf__ uint8_t* addr, uint32_t size, UbTmpBuf& tmpBuf)
{
    if (addr == nullptr || size < sizeof(uint64_t)) {
        return false;
    }
    tmpBuf.addr = addr;
    tmpBuf.size = size;
    return true;
}

PTO_INTERNAL bool IsValidTmpBuffer(const UbTmpBuf& tmpBuf)
{
    return tmpBuf.addr != nullptr && tmpBuf.size >= sizeof(uint64_t);
}

template <typename ScratchTile>
PTO_INTERNAL bool MakeTmpBufferFromTile(ScratchTile& scratchTile, UbTmpBuf& tmpBuf)
{
    static_assert(is_tile_data_v<ScratchTile>, "scratchTile must be a pto::Tile type");
    static_assert(ScratchTile::Loc == TileType::Vec, "scratchTile must be in Vec(UB) memory");
    tmpBuf.addr = reinterpret_cast<__ubuf__ uint8_t*>(scratchTile.data());
    tmpBuf.size = static_cast<uint32_t>(ScratchTile::Numel * sizeof(typename ScratchTile::DType));
    return IsValidTmpBuffer(tmpBuf);
}

template <typename T>
PTO_INTERNAL void SetValue(__gm__ uint8_t* addr, UbTmpBuf& tmpBuf, uint32_t syncId, T x)
{
    __ubuf__ T* ubPtr = reinterpret_cast<__ubuf__ T*>(tmpBuf.addr);
    *ubPtr = x;
    pipe_barrier(PIPE_ALL);

#ifdef PTO_NPU_ARCH_A5
    copy_ubuf_to_gm_align_v2(
        reinterpret_cast<__gm__ uint32_t*>(addr), reinterpret_cast<__ubuf__ uint32_t*>(ubPtr), 0, 1,
        static_cast<uint32_t>(sizeof(T)), 0, 0, 0);
#else
    copy_ubuf_to_gm_align_b32(
        (__gm__ void*)addr, (__ubuf__ void*)ubPtr, 0, 1, static_cast<uint32_t>(sizeof(T)), 0, 0, 0, 0);
#endif
    set_flag(PIPE_MTE3, PIPE_MTE2, syncId);
    wait_flag(PIPE_MTE3, PIPE_MTE2, syncId);
    pipe_barrier(PIPE_ALL);
}

template <typename T>
PTO_INTERNAL T GetValue(__gm__ uint8_t* addr, UbTmpBuf& tmpBuf)
{
    __ubuf__ T* ubPtr = reinterpret_cast<__ubuf__ T*>(tmpBuf.addr);

#ifdef PTO_NPU_ARCH_A5
    copy_gm_to_ubuf_align_v2(
        reinterpret_cast<__ubuf__ uint32_t*>(ubPtr), reinterpret_cast<__gm__ uint32_t*>(addr), 0, 1,
        static_cast<uint32_t>(sizeof(T)), 0, 0, 0, 0, 0, 0);
#else
    copy_gm_to_ubuf_align_b32(
        (__ubuf__ void*)ubPtr, (__gm__ void*)addr, 0, 1, static_cast<uint32_t>(sizeof(T)), 0, 0, 0, 0);
#endif
    pipe_barrier(PIPE_ALL);

    return *ubPtr;
}

PTO_INTERNAL void AddOneMemcpySqe(
    __gm__ BatchWriteChannelInfo* channelInfo, __gm__ uint8_t* src, __gm__ uint8_t* dst, uint64_t opcode,
    uint32_t length, uint32_t sqTail, uint32_t taskId)
{
    __gm__ BatchWriteItem* sqe = (__gm__ BatchWriteItem*)(channelInfo->sq_base);
    sqe += (sqTail % channelInfo->sq_depth);

#ifdef PTO_NPU_ARCH_A5
    sqe->type = RT_STARS_SQE_TYPE_SDMA;
    sqe->wrCqe = 1;
    sqe->numBlocks = 0;
    sqe->rtStreamId = channelInfo->stream_id;
    sqe->taskId = taskId;
    sqe->kernelCredit = K_CREDIT_TIME_DEFAULT;
    sqe->opcode = static_cast<uint32_t>(opcode);
    sqe->sssv = 1U;
    sqe->dssv = 1U;
    sqe->sns = 1U;
    sqe->dns = 1U;
    sqe->lengthMove = length;

    uint64_t src_addr = reinterpret_cast<uint64_t>(src);
    uint64_t dst_addr = reinterpret_cast<uint64_t>(dst);

    *reinterpret_cast<__gm__ uint64_t*>(&sqe->srcAddrLow) = src_addr;
    *reinterpret_cast<__gm__ uint64_t*>(&sqe->dstAddrLow) = dst_addr;
#else
    sqe->type = RT_STARS_SQE_TYPE_SDMA;
    sqe->blockDim = 0;
    sqe->rtStreamId = channelInfo->stream_id;
    sqe->taskId = taskId;
    sqe->kernel_credit = K_CREDIT_TIME_DEFAULT;
    sqe->ptr_mode = 0;
    sqe->opcode = static_cast<uint32_t>(opcode);
    sqe->ie2 = 0;
    sqe->sssv = 1U;
    sqe->dssv = 1U;
    sqe->sns = 1U;
    sqe->dns = 1U;
    sqe->qos = 6;
    sqe->partid = 0U;
    sqe->mpam = 0;
    sqe->length = length;

    uint64_t src_addr = reinterpret_cast<uint64_t>(src);
    uint64_t dst_addr = reinterpret_cast<uint64_t>(dst);

    *reinterpret_cast<__gm__ uint64_t*>(&sqe->srcAddrLow) = src_addr;
    *reinterpret_cast<__gm__ uint64_t*>(&sqe->dstAddrLow) = dst_addr;
    sqe->linkType = static_cast<uint8_t>(255U);
#endif
}

PTO_INTERNAL bool BuildTransferConfig(const SdmaBaseConfig& baseConfig, uint64_t messageLen, SdmaConfig& config)
{
    if (baseConfig.queue_num == 0 || baseConfig.block_bytes == 0) {
        return false;
    }
    config.queue_num = baseConfig.queue_num;
    config.block_bytes = baseConfig.block_bytes;
    config.comm_block_offset = baseConfig.comm_block_offset;
    config.per_core_bytes = messageLen;
    config.iter_num = (config.per_core_bytes + config.block_bytes - 1) / config.block_bytes;
    return true;
}

PTO_INTERNAL void PrepareWorkspace(
    __gm__ uint8_t* workspace, const SdmaConfig& config, WorkspaceLayout& layout, uint32_t channelGroupIdx)
{
    const uint64_t perCoreRecvSize = static_cast<uint64_t>(config.queue_num) * kSdmaFlagLength;
    const uint64_t perGroupSendSize = static_cast<uint64_t>(config.queue_num) * kSdmaMinTransferBytes;

    // Event workspace layout:
    // [send staging: kSdmaSendWorkspaceBytes — one 64B slot per queue per channel group]
    // [recv records: channelGroupIdx * (queue_num * kSdmaFlagLength)]
    __gm__ uint8_t* recvBase = workspace + kSdmaSendWorkspaceBytes;
    __gm__ uint8_t* myRecv = recvBase + static_cast<uint64_t>(channelGroupIdx) * perCoreRecvSize;

    layout.send_workspace = workspace + static_cast<uint64_t>(channelGroupIdx) * perGroupSendSize;
    layout.recv_workspace = myRecv;
}

PTO_INTERNAL void InitSqTailArray(
    __gm__ BatchWriteChannelInfo* batchWriteChannelInfo, uint32_t queueNum, uint32_t* sqTail, uint32_t sqTailLen,
    UbTmpBuf& tmpBuf)
{
    for (uint32_t queueId = 0U; queueId < queueNum; ++queueId) {
        __gm__ BatchWriteChannelInfo* channelInfo = batchWriteChannelInfo + queueId;
        // Invalidate the cache line holding sq_head/sq_tail before reading, so the Wait path
        // observes the sq_tail just persisted by the post path (matches shmem's dcci before
        // reading channel_info+4). Without this the flag SQE can be built on a stale tail and
        // land on an already-consumed slot, so it is never processed and Wait spins to timeout.
        __asm__ __volatile__("");
        dcci((__gm__ void*)channelInfo, SINGLE_CACHE_LINE);
        __asm__ __volatile__("");
        sqTail[queueId] = GetValue<uint32_t>(((__gm__ uint8_t*)channelInfo) + 4, tmpBuf);
    }
}

PTO_INTERNAL void FlushCacheAndRingDoorbell(
    __gm__ BatchWriteChannelInfo* batchWriteChannelInfo, const SdmaConfig& config, uint32_t* sqTail, UbTmpBuf& tmpBuf,
    uint32_t syncId)
{
    for (uint8_t queueId = 0; queueId < config.queue_num; queueId++) {
        __gm__ BatchWriteChannelInfo* channelInfo = batchWriteChannelInfo + queueId;

        __asm__ __volatile__("");
        dcci((__gm__ void*)(channelInfo->sq_base), ENTIRE_DATA_CACHE);
        __asm__ __volatile__("");
        pipe_barrier(PIPE_ALL);
        dsb(DSB_DDR);

#ifdef PTO_NPU_ARCH_A5
        SetValue<uint32_t>((__gm__ uint8_t*)(channelInfo->sq_reg_base), tmpBuf, syncId, sqTail[queueId]);
#else
        SetValue<uint32_t>((__gm__ uint8_t*)(channelInfo->sq_reg_base) + 8, tmpBuf, syncId, sqTail[queueId]);
#endif
    }
}

PTO_INTERNAL void UpdateSqTailState(
    __gm__ BatchWriteChannelInfo* batchWriteChannelInfo, const SdmaConfig& config, uint32_t* sqTail, UbTmpBuf& tmpBuf,
    uint32_t syncId)
{
    for (uint8_t queueId = 0; queueId < config.queue_num; queueId++) {
        __gm__ BatchWriteChannelInfo* channelInfo = batchWriteChannelInfo + queueId;
        // Read current sq_head so the 8-byte write preserves it.
        uint32_t currentHead = GetValue<uint32_t>((__gm__ uint8_t*)channelInfo, tmpBuf);
        uint64_t packed = (static_cast<uint64_t>(sqTail[queueId]) << 32) | static_cast<uint64_t>(currentHead);
        SetValue<uint64_t>((__gm__ uint8_t*)channelInfo, tmpBuf, syncId, packed);
    }
}

PTO_INTERNAL __gm__ uint8_t* GenerationCompletion(__gm__ uint8_t* completionBase, uint32_t queue)
{
    return completionBase + static_cast<uint64_t>(queue) * kGenerationCacheLineBytes;
}

PTO_INTERNAL __gm__ uint8_t* GenerationPayload(__gm__ uint8_t* control, uint64_t generation)
{
    return control + (generation % kGenerationPayloadDepth) * kGenerationCacheLineBytes;
}

PTO_INTERNAL __gm__ uint8_t* ResolveGenerationCompletionBase(const SdmaExecContext& execCtx)
{
    __gm__ uint8_t* workspace =
        execCtx.contextGm + sizeof(BatchWriteFlagInfo) + kSdmaMaxChannel * sizeof(BatchWriteChannelInfo);
    SdmaConfig config{};
    config.queue_num = execCtx.baseConfig.queue_num;
    WorkspaceLayout layout{};
    PrepareWorkspace(workspace, config, layout, execCtx.channelGroupIdx);
    return layout.recv_workspace;
}

PTO_INTERNAL uint64_t LoadGenerationValue(__gm__ uint8_t* address, UbTmpBuf& tmpBuf)
{
    return GetValue<uint64_t>(address, tmpBuf);
}

PTO_INTERNAL void MarkGenerationCompleted(uint64_t generation, uint64_t queueMask, SdmaGenerationSessionState& state)
{
    uint32_t queue = 0U;
    while (queueMask != 0ULL) {
        if ((queueMask & 1ULL) != 0ULL && state.completedGeneration[queue] < generation) {
            state.completedGeneration[queue] = generation;
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

PTO_INTERNAL bool EnsureGenerationSessionState(const SdmaSession& session)
{
    SdmaGenerationSessionState& state = session.generationState;
    if (state.initialized) {
        return true;
    }
    const SdmaExecContext& execCtx = session.execCtx;
    if (session.generationControl == nullptr || execCtx.contextGm == nullptr || !IsValidTmpBuffer(execCtx.tmpBuf)) {
        return false;
    }

    UbTmpBuf tmpBuf = execCtx.tmpBuf;
    state.completionBase = ResolveGenerationCompletionBase(execCtx);
    if (state.completionBase == nullptr) {
        return false;
    }
    for (uint32_t queue = 0U; queue < execCtx.baseConfig.queue_num; ++queue) {
        SetValue<uint64_t>(GenerationCompletion(state.completionBase, queue), tmpBuf, execCtx.syncId, 0ULL);
    }
    pipe_barrier(PIPE_ALL);
    state.nextGeneration = 0ULL;
    for (uint32_t queue = 0U; queue < kGenerationMaxQueues; ++queue) {
        state.completedGeneration[queue] = 0ULL;
        state.sqTail[queue] = 0U;
        state.sqHead[queue] = 0U;
    }
    for (uint32_t slot = 0U; slot < kGenerationPayloadDepth; ++slot) {
        state.payloadActiveQueues[slot] = 0U;
    }
    state.tailInitialized = false;
    state.initialized = true;
    return true;
}

PTO_INTERNAL void InitializeGenerationQueueState(
    __gm__ BatchWriteChannelInfo* channels, uint32_t queueNum, UbTmpBuf& tmpBuf, SdmaGenerationSessionState& state)
{
    __asm__ __volatile__("");
    dcci((__gm__ void*)channels, ENTIRE_DATA_CACHE);
    __asm__ __volatile__("");
    dsb(DSB_DDR);
    for (uint32_t queue = 0U; queue < queueNum; ++queue) {
        __gm__ BatchWriteChannelInfo* channel = channels + queue;
        const uint64_t packedHeadTail = GetValue<uint64_t>((__gm__ uint8_t*)channel, tmpBuf);
        state.sqHead[queue] = static_cast<uint32_t>(packedHeadTail);
        state.sqTail[queue] = static_cast<uint32_t>(packedHeadTail >> 32U);
    }
    state.tailInitialized = true;
}

PTO_INTERNAL bool StoreGenerationPayload(
    uint64_t generation, uint32_t activeQueues, const SdmaSession& session, UbTmpBuf& tmpBuf)
{
    SdmaGenerationSessionState& state = session.generationState;
    if (generation > kGenerationPayloadDepth) {
        // Generation numbers are contiguous, so this slot was last used by generation - depth.
        // Wait for that flag source to be consumed before overwriting the slot.
        const uint64_t oldGeneration = generation - kGenerationPayloadDepth;
        const uint32_t slot = static_cast<uint32_t>(generation % kGenerationPayloadDepth);
        const uint64_t oldQueueMask = ActiveQueuesToMask(state.payloadActiveQueues[slot]);
        bool knownComplete = true;
        uint64_t remaining = oldQueueMask;
        uint32_t queue = 0U;
        while (remaining != 0ULL) {
            if ((remaining & 1ULL) != 0ULL && state.completedGeneration[queue] < oldGeneration) {
                knownComplete = false;
                break;
            }
            remaining >>= 1U;
            ++queue;
        }
        if (!knownComplete) {
            if (!PollGeneration(state.completionBase, oldGeneration, oldQueueMask, tmpBuf, true)) {
                return false;
            }
            MarkGenerationCompleted(oldGeneration, oldQueueMask, state);
        }
    }

    __gm__ uint8_t* payload = GenerationPayload(session.generationControl, generation);
    *reinterpret_cast<volatile __gm__ uint64_t*>(payload) = generation;
    __asm__ __volatile__("" ::: "memory");
    state.payloadActiveQueues[generation % kGenerationPayloadDepth] = static_cast<uint8_t>(activeQueues);
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

PTO_INTERNAL void SubmitGenerationDataSqes(
    __gm__ BatchWriteChannelInfo* channels, __gm__ uint8_t* recvBuffer, __gm__ uint8_t* sendBuffer, uint64_t opcode,
    const SdmaConfig& config, SdmaGenerationSessionState& state)
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
            channels + queue, src, dst, opcode, transferBytes, state.sqTail[queue],
            state.sqTail[queue] - state.sqHead[queue]);
        state.sqTail[queue] = (state.sqTail[queue] + 1U) % channels[queue].sq_depth;
    }
    pipe_barrier(PIPE_ALL);
}

PTO_INTERNAL void RingGenerationDoorbell(
    __gm__ BatchWriteChannelInfo* channel, uint32_t sqTail, UbTmpBuf& tmpBuf, uint32_t syncId)
{
#ifdef PTO_NPU_ARCH_A5
    SetValue<uint32_t>((__gm__ uint8_t*)channel->sq_reg_base, tmpBuf, syncId, sqTail);
#else
    SetValue<uint32_t>((__gm__ uint8_t*)channel->sq_reg_base + 8U, tmpBuf, syncId, sqTail);
#endif
}

PTO_INTERNAL void PublishGenerationDataSqes(
    __gm__ BatchWriteChannelInfo* channels, uint32_t activeQueues, const uint32_t* sqTail, UbTmpBuf& tmpBuf,
    uint32_t syncId)
{
    __asm__ __volatile__("");
    dcci((__gm__ void*)channels->sq_base, ENTIRE_DATA_CACHE);
    __asm__ __volatile__("");
    dsb(DSB_DDR);
    for (uint32_t queue = 0U; queue < activeQueues; ++queue) {
        RingGenerationDoorbell(channels + queue, sqTail[queue], tmpBuf, syncId);
    }
}

PTO_INTERNAL void SubmitGenerationFlagSqes(
    __gm__ BatchWriteChannelInfo* channels, __gm__ uint8_t* payload, uint32_t activeQueues,
    SdmaGenerationSessionState& state)
{
    for (uint32_t queue = 0U; queue < activeQueues; ++queue) {
        AddOneMemcpySqe(
            channels + queue, payload, GenerationCompletion(state.completionBase, queue), 0U, kGenerationFlagBytes,
            state.sqTail[queue], state.sqTail[queue] - state.sqHead[queue]);
        state.sqTail[queue] = (state.sqTail[queue] + 1U) % channels[queue].sq_depth;
    }
    pipe_barrier(PIPE_ALL);
}

PTO_INTERNAL void PersistGenerationTails(
    __gm__ BatchWriteChannelInfo* channels, uint32_t queueNum, const SdmaGenerationSessionState& state,
    UbTmpBuf& tmpBuf, uint32_t syncId)
{
    for (uint32_t queue = 0U; queue < queueNum; ++queue) {
        const uint64_t packed = (static_cast<uint64_t>(state.sqTail[queue]) << 32U) | state.sqHead[queue];
        SetValue<uint64_t>((__gm__ uint8_t*)(channels + queue), tmpBuf, syncId, packed);
    }
}

PTO_INTERNAL void PublishGenerationFlagSqes(
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
        RingGenerationDoorbell(channels + queue, sqTail[queue], tmpBuf, syncId);
    }
}

PTO_INTERNAL AsyncEvent SdmaPostGeneration(
    __gm__ uint8_t* recvBuffer, __gm__ uint8_t* sendBuffer, uint64_t opcode, uint64_t messageLen,
    const SdmaSession& session)
{
    const SdmaExecContext& execCtx = session.execCtx;
    if (!session.valid || recvBuffer == nullptr || sendBuffer == nullptr || !EnsureGenerationSessionState(session)) {
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
    SdmaGenerationSessionState& state = session.generationState;
    if (!state.tailInitialized) {
        InitializeGenerationQueueState(channels, config.queue_num, tmpBuf, state);
    }
    if (!ValidateSinglePostSqCapacity(channels, config, activeQueues) ||
        state.nextGeneration >= kSdmaHandleGenerationMask) {
        return {};
    }

    const uint64_t generation = state.nextGeneration + 1ULL;
    if (!StoreGenerationPayload(generation, activeQueues, session, tmpBuf)) {
        return {};
    }
    uint64_t eventHandle = 0ULL;
    if (!EncodeSdmaEventHandle(generation, activeQueues, eventHandle)) {
        return {};
    }
    state.nextGeneration = generation;
    SubmitGenerationDataSqes(channels, recvBuffer, sendBuffer, opcode, config, state);
    PublishGenerationDataSqes(channels, activeQueues, state.sqTail, tmpBuf, execCtx.syncId);
    SubmitGenerationFlagSqes(channels, GenerationPayload(session.generationControl, generation), activeQueues, state);
    PersistGenerationTails(channels, config.queue_num, state, tmpBuf, execCtx.syncId);
    PublishGenerationFlagSqes(channels, activeQueues, state.sqTail, tmpBuf, execCtx.syncId);
    return AsyncEvent(eventHandle, DmaEngine::SDMA);
}

PTO_INTERNAL bool SdmaGenerationEventCheck(
    uint64_t generation, uint64_t queueMask, const SdmaSession& session, bool blocking)
{
    if (generation == 0ULL || queueMask == 0ULL || !session.generationState.initialized) {
        return false;
    }
    UbTmpBuf tmpBuf = session.eventCtx.tmpBuf;
    if (!PollGeneration(session.generationState.completionBase, generation, queueMask, tmpBuf, blocking)) {
        return false;
    }
    MarkGenerationCompleted(generation, queueMask, session.generationState);
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
    return SdmaGenerationEventCheck(generation, queueMask, session, false);
}

PTO_INTERNAL bool SdmaWaitEvent(uint64_t handle, const SdmaSession& session)
{
    uint64_t generation = 0ULL;
    uint32_t activeQueueNum = 0U;
    if (!DecodeSdmaEventHandle(handle, generation, activeQueueNum)) {
        return false;
    }
    const uint64_t queueMask = ActiveQueuesToMask(activeQueueNum);
    return SdmaGenerationEventCheck(generation, queueMask, session, true);
}

} // namespace detail

// ============================================================================
// Explicit SDMA context builders (explicit contextGm / syncId parameters)
// ============================================================================
template <typename ScratchTile>
PTO_INTERNAL bool BuildSdmaExecContext(
    ScratchTile& scratchTile, uint32_t channelGroupIdx, const SdmaBaseConfig& baseConfig, __gm__ uint8_t* contextGm,
    uint32_t syncId, SdmaExecContext& execCtx)
{
    if (contextGm == nullptr) {
        return false;
    }
    TmpBuffer tmpBuf;
    if (!detail::MakeTmpBufferFromTile(scratchTile, tmpBuf)) {
        return false;
    }
    execCtx.contextGm = contextGm;
    execCtx.tmpBuf = tmpBuf;
    execCtx.syncId = syncId;
    execCtx.channelGroupIdx = channelGroupIdx;
    execCtx.baseConfig = baseConfig;
    return true;
}

template <typename ScratchTile>
PTO_INTERNAL bool BuildSdmaEventContext(ScratchTile& scratchTile, uint32_t syncId, SdmaEventContext& eventCtx)
{
    TmpBuffer tmpBuf;
    if (!detail::MakeTmpBufferFromTile(scratchTile, tmpBuf)) {
        return false;
    }
    eventCtx.tmpBuf = tmpBuf;
    eventCtx.syncId = syncId;
    return true;
}

template <typename ScratchTile>
PTO_INTERNAL bool BuildSdmaSession(
    ScratchTile& scratchTile, __gm__ uint8_t* workspace, SdmaSession& session, uint32_t syncId = 0,
    const SdmaBaseConfig& baseConfig = {kDefaultSdmaBlockBytes, 0, 1}, uint32_t channelGroupIdx = kAutoChannelGroupIdx)
{
    session.generationState = {};
    if (channelGroupIdx == kAutoChannelGroupIdx) {
        channelGroupIdx = static_cast<uint32_t>(get_block_idx());
    }
    if (workspace == nullptr || syncId > 7 || baseConfig.queue_num == 0 || baseConfig.queue_num > kSdmaMaxChannel ||
        channelGroupIdx >= (kSdmaMaxChannel / baseConfig.queue_num)) {
        session.valid = false;
        return false;
    }
    session.generationControl =
        workspace + kSdmaContextWorkspaceBytes + static_cast<uint64_t>(channelGroupIdx) * kSdmaPayloadBytesPerGroup;
    session.valid =
        BuildSdmaExecContext(scratchTile, channelGroupIdx, baseConfig, workspace, syncId, session.execCtx) &&
        BuildSdmaEventContext(scratchTile, syncId, session.eventCtx);
    return session.valid;
}

// ============================================================================
// Async SDMA intrinsics (standalone re-implementation)
// ============================================================================
template <typename T>
PTO_INTERNAL AsyncEvent
__sdma_put_async(__gm__ T* dst, __gm__ T* src, uint64_t transferSize, const SdmaSession& session)
{
    if (transferSize == 0) {
        return {};
    }
    return detail::SdmaPostGeneration((__gm__ uint8_t*)dst, (__gm__ uint8_t*)src, 0U, transferSize, session);
}

template <typename T>
PTO_INTERNAL AsyncEvent
__sdma_get_async(__gm__ T* dst, __gm__ T* src, uint64_t transferSize, const SdmaSession& session)
{
    if (transferSize == 0) {
        return {};
    }
    return detail::SdmaPostGeneration((__gm__ uint8_t*)dst, (__gm__ uint8_t*)src, 0U, transferSize, session);
}

} // namespace sdma
} // namespace comm
} // namespace pto

#endif // PTO_COMM_ASYNC_SDMA_SDMA_ASYNC_INTRIN_HPP
