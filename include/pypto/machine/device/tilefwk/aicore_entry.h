/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file aicore_entry.h
 * \brief
 */

#ifndef AICORE_ENTRY_H
#define AICORE_ENTRY_H

#include <stdint.h>
#include <cstdint>
#include "tilefwk/aicpu_common.h"
#include "tilefwk/aicore_runtime.h"
#include "tilefwk/aicore_print.h"
#include "tilefwk/core_func_data.h"

// device switch head file begin
namespace npu::tile_fwk {
#define PERF_PMU_TEST_SWITCH 0

#define DEBUG_SWITCH 0

#define ENABLE_AICORE_PRINT 0

#define ENABLE_AICORE_PERF_TRACE  0

/* The DFX swimlane performance statistics use host pre-allocated memory mode, which avoids data collection during
   AICPU scheduling to minimize scheduling interference. However, each AICore only supports tracking up to
   MAX_DFX_TASK_NUM_PER_CORE tasks, with excess tasks being discarded.
*/
#define PROF_DFX_HOST_PREPARE_MEMORY_MODE 1
}
// device switch head file end

#define TO_STRING_IMPL(str) #str
#define TO_STRING(str) TO_STRING_IMPL(str)

#ifdef __HAS_SUB_FUNC__
#if defined(__MIX__) && defined(__AIV__)
#include TO_STRING(__HEAD_FILE__)
#else
#include TO_STRING(__HEAD_FILE__)
#endif
#endif

#define AICORE_DEVICE_TASK_RUN_TIMEOUT 3000000000
#define AICORE_DEVICE_TASK_WAIT_TIME_OUT 500000000
#define AICORE_LEAF_TASK_RUN_TIMEOUT 3000000000
#define AICORE_LEAF_TASK_WAIT_TIMEOUT 500000000

using npu::tile_fwk::DynFuncHeader;
using npu::tile_fwk::DynFuncData;
using npu::tile_fwk::DynFuncBin;
using npu::tile_fwk::DevRawTensorDesc;
using npu::tile_fwk::CoreFunctionData;

enum DFX_STAGE_STATUS {
    STAGE_HANDSHAKE_START = 1,
    STAGE_HANDSHAKE_END = 2,
    STAGE_CORE_EXIT = 3,
    STAGE_GET_NEXT_TASK_STOP = 4,
    STAGE_PRE_EXEC_COREFUNC_KERNEL = 5,
    STAGE_FINISH_EXEC_COREFUNC_KERNEL = 6,
    STAGE_FINISH_PIPE_SYNC = 7,
    STAGE_FINISH_CUR_TASK = 8,
    STAGE_GET_COREFUNC_DATA_TIMEOUT = 9,
    STAGE_GET_NEXT_TASK_TIMEOUT = 10
};

struct ExecuteContext {
    __gm__ KernelArgs *args;
    int32_t blockIdx;
    uint32_t seqNo{0};
    __gm__ DynFuncData *funcDataList{nullptr};
    uint64_t lastTaskFinishCycle{0};
#if ENABLE_AICORE_PRINT
    AicoreLogger logger;
#endif
};

#if IS_AICORE
INLINE uint64_t GetDataMainBase() {
    uint64_t coreStatus = 0;
    __asm__ volatile("MOV %0, DATA_MAIN_BASE\n" : "+l"(coreStatus));
    return coreStatus;
}
#endif

INLINE uint32_t GetNextLeafTask(uint32_t lastTaskIdx, uint32_t curDevTaskId) {
    uint32_t nextLowIdx = 0;
    uint64_t coreStatus = 0;
    uint64_t t0 = get_sys_cnt();
    uint64_t loop_count = 0;
    bool isForceContinue = false;
    do {
        isForceContinue = false;
        coreStatus = GetDataMainBase();
        nextLowIdx = coreStatus & 0xFFFFFFFF;
        nextLowIdx -= 1;

        if (nextLowIdx == AICORE_FUNC_STOP) {
            if (curDevTaskId == (uint32_t)(coreStatus >> REG_HIGH_DTASKID_SHIFT)) {
                return nextLowIdx;
            } else {
                isForceContinue = true;
            }
        }
        ++loop_count;
        if ((loop_count % 1000 == 0) && (get_sys_cnt() - t0 > AICORE_LEAF_TASK_WAIT_TIMEOUT)) {
            return AICORE_TASK_STOP;
        }
    } while (nextLowIdx == lastTaskIdx || isForceContinue);

    return nextLowIdx;
}

INLINE void PipeSync() {
#if defined(__AIV__)
    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID7);
    wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID7);
#else
    set_flag(PIPE_FIX, PIPE_S, EVENT_ID7);
    wait_flag(PIPE_FIX, PIPE_S, EVENT_ID7);
#endif
}

INLINE void Barrier()
{
#if defined(__CCE_KT_TEST__) && __CCE_KT_TEST__ == 1
    __asm__ __volatile__("" ::: "memory");
#else
    __asm__ __volatile__("");
#endif
}

INLINE void HandshakeClient(volatile __gm__ int64_t *shakeBuf) {
    set_cond(AICORE_TASK_INIT);
    volatile __gm__ int64_t *hello = shakeBuf;
    *hello = (int64_t)get_coreid() << 32 | AICORE_SAY_HELLO;
    Barrier();
    dcci(hello, SINGLE_CACHE_LINE, CACHELINE_OUT);
    Barrier();
}

INLINE void SetStatus(__gm__ KernelArgs *args, int64_t val) {
    if (!IS_AICORE || DEBUG_SWITCH) {
        Barrier();
        args->shakeBuffer[2] = val;
        dcci(args->shakeBuffer, SINGLE_CACHE_LINE, CACHELINE_OUT);
    }
}

INLINE void SendRegFinish(uint32_t curTaskIdx) {
    set_cond(curTaskIdx | AICORE_FIN_MASK);
}

INLINE void SendRegDevTaskStop(uint32_t dTaskId) {
    set_cond(((uint64_t)dTaskId << REG_HIGH_DTASKID_SHIFT) | (AICORE_FUNC_STOP | AICORE_FIN_MASK));
}

INLINE void SendRegAck(uint32_t taskIdx) {
    set_cond(taskIdx);
}

INLINE void PerfTraceRecord(uint32_t devTaskId, __gm__ Metrics* metric, AicorePerfTrace type, __gm__ KernelArgs *args,
                            uint64_t cycle = 0) {
    if (unlikely(args->taskEntry.reserved[0] == PRO_LEVEL2 || args->taskEntry.reserved[0] == PRO_LEVEL1 ||
        ENABLE_AICORE_PERF_TRACE == 1)) {
        uint32_t cnt = metric->perfTraceCnt[type];
        if (cnt < PERF_TRACE_INST_MAX_NUM_EVERY_TYPE) {
            metric->perfTrace[type][cnt] = cycle == 0 ? get_sys_cnt() : cycle;
            metric->perfTraceDevTaskId[type][cnt] = devTaskId;
            metric->perfTraceCnt[type]++;
        }
    }
}

INLINE void SetTaskStatistic(__gm__ KernelArgs *args, int32_t& dfxPose,
                             int32_t taskId, int32_t subGraphId, int64_t tStart, uint16_t seqNo = 0)
{
    __gm__ volatile TaskStat *stat = &args->taskStat[dfxPose];
    stat->subGraphId = subGraphId;
    stat->taskId = taskId;
    stat->execStart = tStart;
    stat->execEnd = get_sys_cnt();
    stat->seqNo = seqNo;
    dcci(stat, SINGLE_CACHE_LINE, CACHELINE_OUT);
}

INLINE void AddMetricStatistic(ExecuteContext *ctx, uint32_t seqNo, uint32_t taskId, int32_t subGraphId, int64_t t1) {
    UNUSED(ctx); UNUSED(seqNo); UNUSED(taskId); UNUSED(subGraphId); UNUSED(t1);
#if PROF_DFX_HOST_PREPARE_MEMORY_MODE
    auto m = (__gm__ Metrics*)(ctx->args->shakeBuffer[SHAK_BUF_DFX_DATA_INDEX]);
    if (m && m->taskCount < MAX_DFX_TASK_NUM_PER_CORE) {
        m->tasks[m->taskCount].subGraphId = subGraphId;
        m->tasks[m->taskCount].seqNo = seqNo;
        m->tasks[m->taskCount].taskId = taskId;
        m->tasks[m->taskCount].execStart = t1;
        ctx->lastTaskFinishCycle = get_sys_cnt();
        m->tasks[m->taskCount].execEnd = ctx->lastTaskFinishCycle;
        m->taskCount++;
    }
#endif
}

INLINE void FlushMetricStatistic(__gm__ volatile KernelArgs* args) {
    __gm__ volatile Metrics* m = (__gm__ volatile Metrics*)(args->shakeBuffer[SHAK_BUF_DFX_DATA_INDEX]);
    if (m == nullptr) {
        return;
    }

    for (uint32_t i = 0; i < m->taskCount; i++) {
        dcci(&m->tasks[i], SINGLE_CACHE_LINE, CACHELINE_OUT);
    }

    m->isMetricStop = 1;
    dcci(m, SINGLE_CACHE_LINE, CACHELINE_OUT);
    dcci((__gm__ void *)0, ENTIRE_DATA_CACHE, CACHELINE_OUT);
}

INLINE void DfxProcWhenCoreExit(ExecuteContext *ctx, __gm__ KernelArgs *args, __gm__ Metrics* metric) {
    PerfTraceRecord(INVALID_DEV_TASK_ID, metric, PERF_TRACE_CORE_WAIT_EXIT_NOTIFY, args);
    if (ctx->lastTaskFinishCycle > 0) {
        PerfTraceRecord(INVALID_DEV_TASK_ID, metric,
            PERF_TRACE_CORE_WAIT_ALL_DEV_TASK_CALLOP_EXEC_FINISH, args, ctx->lastTaskFinishCycle);
    }
    if (unlikely(args->taskEntry.reserved[0] == PRO_LEVEL2 || args->taskEntry.reserved[0] == PRO_LEVEL1 ||
        ENABLE_AICORE_PERF_TRACE == 1)) {
        FlushMetricStatistic(args);
    }
}

INLINE void DfxProcWhenDevTaskStop(ExecuteContext *ctx, __gm__ KernelArgs *args, __gm__ Metrics* metric) {
    PerfTraceRecord(ctx->seqNo, metric, PERF_TRACE_CORE_DEV_TASK_WAIT_SYNC_STOP_NOTIFY, args);
    if (ctx->lastTaskFinishCycle > 0) {
        PerfTraceRecord(ctx->seqNo, metric, PERF_TRACE_CORE_DEV_TASK_CALLOP_TASK_EXEC, args, ctx->lastTaskFinishCycle);
    }
    SetStatus(args, STAGE_GET_NEXT_TASK_STOP);
}

INLINE uint64_t GetCoreFuncionData(__gm__ KernelArgs *args, int64_t lastFunc) {
    uint64_t t0 = get_sys_cnt();
    uint64_t loop_count = 0;
    while (true) {
        if (lastFunc) {
            volatile __gm__ int64_t *waveBuffer = args->waveBufferCpuToCore;
            dcci(waveBuffer, SINGLE_CACHE_LINE, CACHELINE_OUT);
            if (*waveBuffer == AICORE_SAY_GOODBYE) {
                return 0;
            }
        }

        ++loop_count;
        if ((loop_count % 1000 == 0) && (get_sys_cnt() - t0 > AICORE_DEVICE_TASK_WAIT_TIME_OUT)) {
            SetStatus(args, STAGE_GET_COREFUNC_DATA_TIMEOUT);
            break;
        }
        volatile __gm__ int64_t *shakebufferCpuToCore = args->shakeBufferCpuToCore;
        dcci(shakebufferCpuToCore, SINGLE_CACHE_LINE, CACHELINE_OUT);
        auto newFunc = shakebufferCpuToCore[CPU_TO_CORE_SHAK_BUF_COREFUNC_DATA_INDEX];
        if (newFunc != lastFunc && newFunc != 0) {
            dcci((__gm__ void *)newFunc, SINGLE_CACHE_LINE, CACHELINE_OUT);
            return newFunc;
        }
    }
    return 0;
}

INLINE void PmuTestBegin(__gm__ KernelArgs *args) {
    UNUSED(args);
#if PERF_PMU_TEST_SWITCH
    if (args->taskEntry.reserved[0] == PRO_LEVEL2) {
        set_ctrl((uint64_t) get_ctrl() | 0x1);
    }
#endif
}

INLINE void PmuTestEnd(__gm__ KernelArgs *args) {
    UNUSED(args);
#if PERF_PMU_TEST_SWITCH
        if (args->taskEntry.reserved[0] == PRO_LEVEL2) {
            set_ctrl((uint64_t) get_ctrl() - 1);
        }
#endif
}

#define FuncNum(id)      TaskID(id)

#ifdef __HAS_SUB_FUNC__
INLINE void ExecDynCoreFunctionKernel(ExecuteContext *ctx, uint32_t taskId) {
    uint64_t t1 = get_sys_cnt();
    SetStatus(ctx->args, ((uint64_t)taskId << 32) | STAGE_PRE_EXEC_COREFUNC_KERNEL); // high 32 bits used for taskId
    auto funcData = &ctx->funcDataList[npu::tile_fwk::FuncID(taskId)];
    auto opAttrs = &funcData->opAttrs[funcData->opAtrrOffsets[npu::tile_fwk::TaskID(taskId)]];
#if ENABLE_AICORE_PRINT
    CoreFuncParam param = {funcData, opAttrs, funcData->exprTbl, taskId, ctx->logger.context()};
#else
    CoreFuncParam param = {funcData, opAttrs, funcData->exprTbl, taskId, nullptr};
#endif
    CallSubFuncTask(opAttrs[0] + funcData->exprTbl[0], &param, funcData->stackWorkSpaceAddr + ctx->blockIdx * funcData->stackWorkSpaceSize,
                    (__gm__ int64_t *)funcData->startArgs->commContexts);
    SetStatus(ctx->args, STAGE_FINISH_EXEC_COREFUNC_KERNEL);
    PipeSync();
    SetStatus(ctx->args, STAGE_FINISH_PIPE_SYNC);
    if (unlikely(ctx->args->taskEntry.reserved[0] == PRO_LEVEL2 || ctx->args->taskEntry.reserved[0] == PRO_LEVEL1)) {
        AddMetricStatistic(ctx, ctx->seqNo, taskId, opAttrs[0], t1);
    }
#if ENABLE_AICORE_PERF_TRACE
    ctx->lastTaskFinishCycle = get_sys_cnt();
#endif

#if PROF_DFX_HOST_PREPARE_MEMORY_MODE != 1
    static int32_t taskDfxPos = REG_LOW_TASK_PING;
    SetTaskStatistic(ctx->args, taskDfxPos, taskId, opAttrs[0], t1, ctx->seqNo);
#endif
}
#endif

INLINE void InitCtx(ExecuteContext *ctx, __gm__ Metrics* metric, uint64_t coreFuncData) {
    __gm__ DynFuncHeader *header = (__gm__ DynFuncHeader *)coreFuncData;
    ctx->seqNo = header->seqNo;
    PerfTraceRecord(ctx->seqNo, metric, PERF_TRACE_CORE_DEV_TASK_RCV_MODEL, ctx->args);
    ctx->funcDataList = (__gm__ npu::tile_fwk::DynFuncData *)(header + 1);
    ctx->lastTaskFinishCycle = 0;
#if ENABLE_AICORE_PRINT
    auto buffer = reinterpret_cast<__gm__ uint8_t *>(ctx->args->shakeBuffer[SHAK_BUF_PRINT_BUFFER_INDEX]);
    if (ctx->logger.GetBuffer() != buffer) {
        ctx->logger.Init(buffer, PRINT_BUFFER_SIZE);
    }
#endif
    dcci((__gm__ void *)0, ENTIRE_DATA_CACHE, CACHELINE_OUT);
    return;
}

INLINE void ExecCoreFunctionKernel(ExecuteContext *ctx, uint32_t curTaskIdx) {
    UNUSED(ctx); UNUSED(curTaskIdx);
#ifdef __HAS_SUB_FUNC__
    ExecDynCoreFunctionKernel(ctx, curTaskIdx);
    return;
#endif
}

INLINE void WaitWaveSignal(__gm__ KernelArgs *args) {
    uint64_t t2 = get_sys_cnt();
    volatile __gm__ int64_t *waveBuffer = args->waveBufferCpuToCore;
    while (true) {
        dcci(waveBuffer, SINGLE_CACHE_LINE, CACHELINE_OUT);
        if (*waveBuffer == AICORE_SAY_GOODBYE) {
            return;
        }
        if ((get_sys_cnt() - t2 > 50000000)) {
            return;
        }
    }
}

INLINE void KernelEntry(int64_t ffts_addr, int64_t inputs,
        int64_t outputs, int64_t workspace, int64_t tilingdata, int64_t cfgdata) {
    UNUSED(ffts_addr); UNUSED(inputs); UNUSED(outputs); UNUSED(workspace); UNUSED(tilingdata);
#if defined(__AIV__) and defined(__MIX__)
    int32_t blockIdx = get_block_idx() * get_subblockdim() + get_subblockid() + get_block_num();
#else
    int32_t blockIdx = get_block_idx();
#endif
    auto devArgs = (DeviceArgs*)cfgdata;
    __gm__ KernelArgs *args = (__gm__ KernelArgs *)(devArgs->sharedBuffer + blockIdx * SHARED_BUFFER_SIZE);
    __gm__ Metrics* metric = (__gm__ Metrics*)(args->shakeBuffer[SHAK_BUF_DFX_DATA_INDEX]);
    args->taskEntry.reserved[0] = ((__gm__ DevDfxArgs*)devArgs->devDfxArgAddr)->isOpenSwim | args->taskEntry.reserved[0];
    PerfTraceRecord(INVALID_DEV_TASK_ID, metric, PERF_TRACE_CORE_BEGIN, args);
    bool isFirstTask = true;
    SetStatus(args, STAGE_HANDSHAKE_START);
    HandshakeClient(args->shakeBuffer);
    SetStatus(args, STAGE_HANDSHAKE_END);
    set_mask_norm();
    uint32_t curTaskIdx;
    uint32_t lastTaskIdx;
    int64_t coreFuncData = 0;
    ExecuteContext ctx = {.args = args, .blockIdx = blockIdx };
    //get core task data
    uint64_t t0 = get_sys_cnt();
    uint64_t loop_count = 0;
    bool bIsExit = false;
    PerfTraceRecord(INVALID_DEV_TASK_ID, metric, PERF_TRACE_CORE_INIT, args);
    while (true) {
        ++loop_count;
        if ((loop_count % 1000 == 0) && (get_sys_cnt() - t0 > AICORE_DEVICE_TASK_RUN_TIMEOUT)) {
            break;
        }
        lastTaskIdx = AICORE_TASK_INIT;
        if (bIsExit) {
            DfxProcWhenCoreExit(&ctx, args, metric);
            return WaitWaveSignal(args); // no data exit
        }
        coreFuncData = GetCoreFuncionData(args, coreFuncData);
        if (coreFuncData == 0) {
            DfxProcWhenCoreExit(&ctx, args, metric);
            return; // no data exit
        }
        InitCtx(&ctx, metric, coreFuncData);
        uint64_t t1 = get_sys_cnt();
        uint64_t inner_loop_count = 0;
        isFirstTask = true;
        while (true) {
            ++inner_loop_count;
            if ((inner_loop_count % 1000 == 0) && (get_sys_cnt() - t1 > AICORE_LEAF_TASK_RUN_TIMEOUT)) {
                break;
            }
            curTaskIdx = GetNextLeafTask(lastTaskIdx, ctx.seqNo);
            if (curTaskIdx == AICORE_TASK_STOP) {
                DfxProcWhenDevTaskStop(&ctx, args, metric);
                SetStatus(args, STAGE_CORE_EXIT);
                bIsExit = true;
                break;
            } else if (curTaskIdx == AICORE_FUNC_STOP) {
                DfxProcWhenDevTaskStop(&ctx, args, metric);
                SendRegDevTaskStop(ctx.seqNo);
                break;
            }

            if (isFirstTask) {
                PerfTraceRecord(ctx.seqNo, metric, PERF_TRACE_CORE_DEV_TASK_WAIT_RCV_FIRST_CALLOP_TASK, args);
                isFirstTask = false;
            }

            SendRegAck(curTaskIdx);
            PmuTestBegin(args);
            ExecCoreFunctionKernel(&ctx, curTaskIdx);
            PmuTestEnd(args);
            SendRegFinish(curTaskIdx);
            lastTaskIdx = curTaskIdx;
            SetStatus(args, STAGE_FINISH_CUR_TASK);
        }
    }
}

#endif