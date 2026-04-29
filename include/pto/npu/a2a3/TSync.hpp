/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TSYNC_HPP
#define TSYNC_HPP

#include <pto/common/type.hpp>
#include <pto/common/event.hpp>

#define FFTS_BASE_COUNT_WIDTH 0xf
#define FFTS_MODE_VAL 0x2
#define FFTS_MODE_WIDTH 0x3
#define FFTS_MODE_OFFSET 4
#define FFTS_EVENT_ID_WIDTH 0xf
#define FFTS_EVENT_ID_OFFSET 8
namespace pto {
constexpr uint16_t SYNC_AIC_FLAG = 11;
constexpr uint16_t SYNC_AIV_FLAG = 12;
constexpr uint16_t SYNC_AIC_AIV_FLAG = 13;
constexpr uint16_t SYNC_AIV_ONLY_ALL = 14;
constexpr uint16_t SYNC_FLAG_ID_MAX = 16;
constexpr int32_t SYNCALL_SOFT_SLOT_INT32 = 8;

PTO_INTERNAL void TSYNCALL_SOFT_DCCI(__gm__ void *ptr)
{
    __asm__ __volatile__("");
    dcci(ptr, SINGLE_CACHE_LINE);
    __asm__ __volatile__("");
}

PTO_INTERNAL void TSYNCALL_SOFT_DCCI_RANGE(__gm__ int32_t *ptr, int32_t lines)
{
    for (int32_t i = 0; i < lines; ++i) {
        TSYNCALL_SOFT_DCCI(static_cast<__gm__ void *>(ptr + i * SYNCALL_SOFT_SLOT_INT32));
    }
    dsb(DSB_DDR);
}

template <Op OpCode>
PTO_INTERNAL static constexpr pipe_t GetPipeByOp()
{
    if constexpr ((OpCode >= static_cast<Op>(0)) && (OpCode < Op::OP_COUNT)) {
        return opPipeList[static_cast<int>(OpCode)];
    }
    return PIPE_ALL;
}

// single pipeline wait, only support Vector pipeline
template <Op OpCode>
PTO_INTERNAL void TSYNC_IMPL()
{
#ifndef __PTO_AUTO__
    constexpr pipe_t pipe = GetPipeByOp<OpCode>();
    PTO_STATIC_ASSERT(pipe == PIPE_V, "Single Op TSYNC only supports Vector PTO Instruction.");
    pipe_barrier((pipe_t)pipe);
#endif
}

PTO_INTERNAL uint16_t getFFTSMsg(uint16_t mode, uint16_t eventId, uint16_t baseConst = 0x1)
{
    return ((baseConst & FFTS_BASE_COUNT_WIDTH) + ((mode & FFTS_MODE_WIDTH) << FFTS_MODE_OFFSET) +
            ((eventId & FFTS_EVENT_ID_WIDTH) << FFTS_EVENT_ID_OFFSET));
}

template <bool IsAIVOnly = true>
PTO_INTERNAL void TSYNCALL_IMPL()
{
#ifndef __PTO_AUTO__
    pipe_barrier(PIPE_ALL);
    if constexpr (IsAIVOnly) {
#if defined(__DAV_VEC__)
        ffts_cross_core_sync(PIPE_MTE3, getFFTSMsg(0x0, SYNC_AIV_ONLY_ALL));
        wait_flag_dev(SYNC_AIV_ONLY_ALL);
#endif
        return;
    }

#if defined(__DAV_CUBE__)
    wait_flag_dev(SYNC_AIV_FLAG);
    ffts_cross_core_sync(PIPE_FIX, getFFTSMsg(0x0, SYNC_AIC_FLAG));
    wait_flag_dev(SYNC_AIC_FLAG);
    ffts_cross_core_sync(PIPE_MTE3, getFFTSMsg(0x2, SYNC_AIC_AIV_FLAG));
#elif defined(__DAV_VEC__)
    ffts_cross_core_sync(PIPE_MTE3, getFFTSMsg(0x2, SYNC_AIV_FLAG));
    wait_flag_dev(SYNC_AIC_AIV_FLAG);
#endif
#endif
}

template <bool IsAIVOnly = true>
PTO_INTERNAL void TSYNCALL_SOFT_IMPL(__gm__ int32_t *gmWorkspace, __ubuf__ int32_t *ubWorkspace,
                                     int32_t usedCores = 0)
{
#ifndef __PTO_AUTO__
    PTO_STATIC_ASSERT(IsAIVOnly, "Software TSYNCALL currently only supports AIV-only kernels.");
    pipe_barrier(PIPE_ALL);

#if defined(__DAV_VEC__)
    const int32_t totalBlocks = (usedCores != 0) ? usedCores : static_cast<int32_t>(get_block_num());
    const int32_t blockIdx = static_cast<int32_t>(get_block_idx());
    __gm__ int32_t *localSyncGM = gmWorkspace + blockIdx * SYNCALL_SOFT_SLOT_INT32;

    TSYNCALL_SOFT_DCCI(static_cast<__gm__ void *>(localSyncGM));
    copy_gm_to_ubuf(static_cast<__ubuf__ void *>(ubWorkspace), static_cast<__gm__ void *>(localSyncGM), 0, 1, 1, 0,
                    0);
    set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);

    const int32_t curValue = ubWorkspace[0] + 1;
    ubWorkspace[0] = curValue;

    set_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
    copy_ubuf_to_gm(static_cast<__gm__ void *>(localSyncGM), static_cast<__ubuf__ void *>(ubWorkspace), 0, 1, 1, 0,
                    0);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    TSYNCALL_SOFT_DCCI(static_cast<__gm__ void *>(localSyncGM));
    dsb(DSB_DDR);

    while (true) {
        TSYNCALL_SOFT_DCCI_RANGE(gmWorkspace, totalBlocks);
        copy_gm_to_ubuf(static_cast<__ubuf__ void *>(ubWorkspace), static_cast<__gm__ void *>(gmWorkspace), 0, 1,
                        totalBlocks, 0, 0);
        set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);

        int32_t readyCount = 0;
        for (int32_t i = 0; i < totalBlocks; ++i) {
            if (ubWorkspace[i * SYNCALL_SOFT_SLOT_INT32] >= curValue) {
                ++readyCount;
            }
        }
        pipe_barrier(PIPE_ALL);
        if (readyCount >= totalBlocks) {
            break;
        }
    }
#endif
    pipe_barrier(PIPE_ALL);
#endif
}

template <Op SrcOp, Op DstOp, bool AutoToken = true, event_t EventID = EVENT_ID0>
struct Event {
#ifndef __PTO_AUTO__
    static constexpr Op srcOp = SrcOp;
    static constexpr Op dstOp = DstOp;
    static constexpr pipe_t srcPipe = GetPipeByOp<srcOp>();
    static constexpr pipe_t dstPipe = GetPipeByOp<dstOp>();
    PTO_STATIC_ASSERT(SrcOp != DstOp, "SrcOp is not allowed to be equal to DstOp.");
    PTO_STATIC_ASSERT(dstPipe != srcPipe, "SrcPipe is not allowed to be equal to dstPipe.");

    PTO_INTERNAL static constexpr bool IsCrossCoreEvent()
    {
        return ((srcOp == Op::TMOV_A2V) && (GetPipeByOp<dstOp>() == PIPE_V)) || // dstOp为搬运到GM的MTE3是否需要考虑
               ((srcOp == Op::TMOV_V2M || srcOp == Op::TEXTRACT_V2M) && (GetPipeByOp<dstOp>() == PIPE_MTE1));
    }

    static constexpr bool IsCrossCore = IsCrossCoreEvent();
    PTO_STATIC_ASSERT(IsCrossCore || (srcPipe != PIPE_ALL), "SrcOp are invalid.");
    PTO_STATIC_ASSERT(IsCrossCore || (dstPipe != PIPE_ALL), "DstOp are invalid.");
    PTO_STATIC_ASSERT((!IsCrossCore) || (!AutoToken), "Cross-core events must manually specify EventID.");

#ifdef PTO_FLAG_TEST
    CceEventIdType token = {};
#else
    const event_t token = AutoToken ? EventIdCounter<srcPipe, dstPipe>::GetNextId() : EventID;
#endif
#endif

    PTO_INTERNAL Event &InitAddr(uint64_t fftsAddr)
    {
#ifndef __PTO_AUTO__
        PTO_STATIC_ASSERT(IsCrossCore, "Only cross-core events require setting the initial addr.");
        set_ffts_base_addr(fftsAddr);
#endif
        return *this;
    }

    template <uint8_t CrossCoreId = 0xff>
    PTO_INTERNAL Event &Wait()
    {
#ifndef __PTO_AUTO__
        if constexpr (IsCrossCore) {
            PTO_STATIC_ASSERT(CrossCoreId != 0xff,
                              "Fix: The cross-core id must be assigned by user when the event is a cross-core event.");
            wait_flag_dev(CrossCoreId);
        } else {
#ifdef PTO_FLAG_TEST
            __pto_wait_flag((pipe_t)srcPipe, (pipe_t)dstPipe, token);
#else
            wait_flag((pipe_t)srcPipe, (pipe_t)dstPipe, token);
#endif
        }
#endif
        return *this;
    }

    template <uint8_t CrossCoreId = 0xff>
    PTO_INTERNAL Event &Init()
    {
#ifndef __PTO_AUTO__
        if constexpr (IsCrossCore) {
            PTO_STATIC_ASSERT(CrossCoreId != 0xff,
                              "Fix: The cross-core id must be assigned by user when the event is a cross-core event.");
            ffts_cross_core_sync(srcPipe, getFFTSMsg(FFTS_MODE_VAL, CrossCoreId));
        } else {
#ifdef PTO_FLAG_TEST
            token = __pto_set_flag((pipe_t)srcPipe, (pipe_t)dstPipe);
#else
            set_flag((pipe_t)srcPipe, (pipe_t)dstPipe, token);
#endif
        }
#endif
        return *this;
    }

    template <uint8_t CrossCoreId = 0xff>
    PTO_INTERNAL Event &Record()
    {
        return Init<CrossCoreId>();
    }

    PTO_INTERNAL Event &operator=(RecordEvent)
    {
        return Init();
    }

    PTO_INTERNAL Event() = default;
    PTO_INTERNAL Event(RecordEvent)
    {
        Init();
    }
};

template <typename T>
struct is_event : std::false_type {
};

template <Op SrcOp, Op DstOp, bool AutoToken, event_t EventID>
struct is_event<Event<SrcOp, DstOp, AutoToken, EventID>> : std::true_type {
};

template <typename... Ts>
inline constexpr bool all_events_v = (is_event<Ts>::value && ...);
} // namespace pto
#endif
