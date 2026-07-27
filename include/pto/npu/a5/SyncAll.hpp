/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef SYNCALL_HPP
#define SYNCALL_HPP

#include <pto/npu/a5/TSync.hpp>

namespace pto {

#define FFTS_BASE_COUNT_WIDTH 0xf
#define FFTS_MODE_WIDTH 0x3
#define FFTS_MODE_OFFSET 4
#define FFTS_EVENT_ID_WIDTH 0xf
#define FFTS_EVENT_ID_OFFSET 8

PTO_INTERNAL uint16_t getFFTSMsg(uint16_t mode, uint16_t eventId, uint16_t baseConst = 0x1)
{
    return (
        (baseConst & FFTS_BASE_COUNT_WIDTH) + ((mode & FFTS_MODE_WIDTH) << FFTS_MODE_OFFSET) +
        ((eventId & FFTS_EVENT_ID_WIDTH) << FFTS_EVENT_ID_OFFSET));
}

PTO_INTERNAL void SYNCALL_SOFT_DCCI(__gm__ void* ptr)
{
    __asm__ __volatile__("");
    dcci(ptr, SINGLE_CACHE_LINE);
    __asm__ __volatile__("");
}

PTO_INTERNAL int32_t SYNCALL_GET_MIX_AIC_BLOCKS()
{
#if defined(__MIX_CORE_AIC_BLOCKS__)
    return static_cast<int32_t>(__MIX_CORE_AIC_BLOCKS__);
#else
    return static_cast<int32_t>(get_block_num());
#endif
}

// __MIX_CORE_AIV_RATIO__ wins when the build declares it: it states the ratio the
// launch actually uses, which get_subblockdim() cannot report on the cube side.
PTO_INTERNAL int32_t SYNCALL_GET_MIX_AIV_RATIO()
{
#if defined(__MIX_CORE_AIV_RATIO__)
    return static_cast<int32_t>(__MIX_CORE_AIV_RATIO__);
#elif defined(__DAV_VEC__)
    return static_cast<int32_t>(get_subblockdim());
#else
    return 1;
#endif
}

PTO_INTERNAL int32_t SYNCALL_GET_MIX_PARTICIPANT_COUNT()
{
    return static_cast<int32_t>(SYNCALL_GET_MIX_AIC_BLOCKS() * (1 + SYNCALL_GET_MIX_AIV_RATIO()));
}

template <SyncCoreType CoreType = SyncCoreType::AIVOnly>
PTO_INTERNAL void SYNCALL_IMPL()
{
#ifndef __PTO_AUTO__
    pipe_barrier(PIPE_ALL);
    if constexpr (CoreType == SyncCoreType::AIVOnly) {
#if defined(__DAV_VEC__)
        ffts_cross_core_sync(PIPE_MTE3, getFFTSMsg(0x0, SYNC_AIV_ONLY_ALL));
        wait_flag_dev(PIPE_S, SYNC_AIV_ONLY_ALL);
#endif
        return;
    } else if constexpr (CoreType == SyncCoreType::AICOnly) {
#if defined(__DAV_CUBE__)
        ffts_cross_core_sync(PIPE_FIX, getFFTSMsg(0x0, SYNC_AIC_FLAG));
        wait_flag_dev(PIPE_S, SYNC_AIC_FLAG);
#endif
        return;
    }

#if defined(__DAV_CUBE__)
    wait_intra_block(PIPE_S, SYNC_AIV_FLAG);
    wait_intra_block(PIPE_S, SYNC_AIV_FLAG + SYNC_FLAG_ID_MAX);
    ffts_cross_core_sync(PIPE_FIX, getFFTSMsg(0x0, SYNC_AIC_FLAG));
    wait_flag_dev(PIPE_S, SYNC_AIC_FLAG);
    set_intra_block(PIPE_S, SYNC_AIC_AIV_FLAG);
    set_intra_block(PIPE_S, SYNC_AIC_AIV_FLAG + SYNC_FLAG_ID_MAX);
#elif defined(__DAV_VEC__)
    set_intra_block(PIPE_MTE3, SYNC_AIV_FLAG);
    wait_intra_block(PIPE_S, SYNC_AIC_AIV_FLAG);
#endif
#endif
}

// Non-cacheable scalar read of the shared counter via ld_dev. Valid on both A5
// AIC (cube) and AIV cores.
PTO_INTERNAL int32_t SYNCALL_SOFT_ATOMIC_LOAD(__gm__ int32_t* counter)
{
    return static_cast<int32_t>(ld_dev(reinterpret_cast<__gm__ uint32_t*>(counter), 0));
}

// Spin until the shared counter reaches target. Reached by every participant
// core (AIC and AIV); ld_dev is legal on both on A5.
PTO_INTERNAL void SYNCALL_SOFT_POLL(__gm__ int32_t* counter, int32_t target)
{
    int32_t pollCount = 0;
    while (SYNCALL_SOFT_ATOMIC_LOAD(counter) < target) {
        if ((++pollCount % SYNCALL_SOFT_BACKOFF_THRESHOLD) == 0) {
            pipe_barrier(PIPE_ALL);
        }
        if (pollCount >= SYNCALL_SOFT_MAX_POLL_ITERATIONS) {
            PTO_CPU_ASSERT(false, "SYNCALL soft barrier timeout - possible deadlock");
            break;
        }
    }
}

// Hardware scalar atomic-add of 1 to the shared counter; the dcci write-back is
// the atomic publication point (same pattern as comm TNOTIFY AtomicAdd).
// set_st_atomic_cfg enables a STICKY DMA atomic-accumulate mode that also affects
// subsequent non-atomic GM writes (copy_ubuf_to_gm, fixpipe/create_cbuf on AIC),
// so it must be reset with set_atomic_none() once the atomic store has landed;
// otherwise the next GM/Tile store corrupts data or faults (507015) on AIC.
PTO_INTERNAL void SYNCALL_SOFT_ATOMIC_ADD(__gm__ int32_t* counter)
{
    set_st_atomic_cfg(ATOMIC_S32, ATOMIC_SUM);
    SYNCALL_SOFT_DCCI(static_cast<__gm__ void*>(counter));
    st_atomic<int32_t>(1, counter);
    SYNCALL_SOFT_DCCI(static_cast<__gm__ void*>(counter));
    dsb(DSB_DDR);
    // set_atomic_none();
}

// Shared atomic-counter barrier for AIV-only / AIC-only / MIX soft SYNCALL.
// Counter is monotonic (never reset); epoch is derived from the pre-arrival value.
PTO_INTERNAL void SYNCALL_SOFT_ATOMIC_BARRIER(__gm__ int32_t* gmWorkspace, int32_t totalBlocks)
{
    dsb(DSB_DDR);
    const int32_t before = SYNCALL_SOFT_ATOMIC_LOAD(gmWorkspace);
    const int32_t target = (before / totalBlocks + 1) * totalBlocks;
    SYNCALL_SOFT_ATOMIC_ADD(gmWorkspace);
    SYNCALL_SOFT_POLL(gmWorkspace, target);
    dsb(DSB_DDR);
}

// MIX software SYNCALL: every AIC/AIV participant arrives on one shared counter,
// the same barrier the AIV-only and AIC-only paths use. Only element 0 of the
// workspace is touched.
template <SyncCoreType CoreType = SyncCoreType::Mix>
PTO_INTERNAL void SYNCALL_SOFT_MIX_IMPL(__gm__ int32_t* gmWorkspace, int32_t usedCores = 0)
{
#ifndef __PTO_AUTO__
    PTO_STATIC_ASSERT(CoreType == SyncCoreType::Mix, "Software SYNCALL mix overload is for AIC/AIV kernels.");
    pipe_barrier(PIPE_ALL);

#if defined(__DAV_CUBE__) || defined(__DAV_VEC__)
    const int32_t totalBlocks = (usedCores != 0) ? usedCores : SYNCALL_GET_MIX_PARTICIPANT_COUNT();
    SYNCALL_SOFT_ATOMIC_BARRIER(gmWorkspace, totalBlocks);
#else
    (void)gmWorkspace;
    (void)usedCores;
#endif
    pipe_barrier(PIPE_ALL);
#endif
}

// AIC-only software SYNCALL: shared atomic counter over cube cores only. A5 AIC
// supports scalar st_atomic/ld_dev, so it runs the same barrier as AIV-only.
PTO_INTERNAL void SYNCALL_SOFT_AIC_IMPL(__gm__ int32_t* gmWorkspace, int32_t usedCores = 0)
{
#ifndef __PTO_AUTO__
    pipe_barrier(PIPE_ALL);
#if defined(__DAV_CUBE__)
    const int32_t totalBlocks = (usedCores != 0) ? usedCores : static_cast<int32_t>(get_block_num());
    SYNCALL_SOFT_ATOMIC_BARRIER(gmWorkspace, totalBlocks);
#else
    (void)gmWorkspace;
    (void)usedCores;
#endif
    pipe_barrier(PIPE_ALL);
#endif
}

// AIV-only software SYNCALL: atomic shared counter.
template <SyncCoreType CoreType = SyncCoreType::AIVOnly>
PTO_INTERNAL void SYNCALL_SOFT_IMPL(__gm__ int32_t* gmWorkspace, int32_t usedCores = 0)
{
#ifndef __PTO_AUTO__
    PTO_STATIC_ASSERT(
        CoreType == SyncCoreType::AIVOnly, "Software SYNCALL soft GM overload only supports AIV-only kernels on A5.");
    pipe_barrier(PIPE_ALL);

#if defined(__DAV_VEC__)
    const int32_t totalBlocks = (usedCores != 0) ? usedCores : static_cast<int32_t>(get_block_num());
    SYNCALL_SOFT_ATOMIC_BARRIER(gmWorkspace, totalBlocks);
#else
    (void)gmWorkspace;
    (void)usedCores;
#endif
    pipe_barrier(PIPE_ALL);
#endif
}
} // namespace pto
#endif
