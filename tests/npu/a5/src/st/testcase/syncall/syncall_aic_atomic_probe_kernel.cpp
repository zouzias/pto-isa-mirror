/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// AIC-only concurrent-atomic probes, dav-c310-cube single-chevron launch, 18 cube
// blocks all hitting ONE shared GM counter. A single cube st_atomic/ld_dev already
// passes (aic_probe), and the AIV-only atomic barrier passes; the unknown is
// multiple cube cores concurrently updating one counter. These three probes peel
// the SYNCALL_SOFT_ATOMIC_BARRIER apart layer by layer so the layer that faults
// (507015) or loses updates can be pinpointed:
//   add      : pure concurrent st_atomic, no dcci, no poll.
//   add_dcci : + the scalar dcci write-back around st_atomic (SYNCALL_SOFT_ATOMIC_ADD)
//              -- prime suspect: one core's dcci writes back a stale cache line and
//              clobbers/faults against another core's DMA atomic on the same address.
//   barrier  : the full public AICOnly soft barrier once (add + ld_dev poll).
// Counter is preset 0; every probe should leave it == block count (18) with ret 0.

#include <pto/pto-inst.hpp>
#include "acl/acl.h"

using namespace pto;

constexpr int32_t kAicAtomicBlockCount = 18;

// Probe 1: pure concurrent atomic add, no dcci write-back, no poll.
extern "C" __global__ AICORE void RunAicAtomicProbeAdd_4101(__gm__ int32_t __out__* counter)
{
#if defined(__DAV_CUBE__)
    set_st_atomic_cfg(ATOMIC_S32, ATOMIC_SUM);
    st_atomic<int32_t>(1, counter);
    set_atomic_none();
    dsb(DSB_DDR);
#else
    (void)counter;
#endif
}

// Probe 2: atomic add wrapped by dcci exactly as SYNCALL_SOFT_ATOMIC_ADD, no poll.
extern "C" __global__ AICORE void RunAicAtomicProbeAddDcci_4102(__gm__ int32_t __out__* counter)
{
#if defined(__DAV_CUBE__)
    set_st_atomic_cfg(ATOMIC_S32, ATOMIC_SUM);
    dcci(static_cast<__gm__ void*>(counter), SINGLE_CACHE_LINE);
    st_atomic<int32_t>(1, counter);
    dcci(static_cast<__gm__ void*>(counter), SINGLE_CACHE_LINE);
    dsb(DSB_DDR);
    set_atomic_none();
#else
    (void)counter;
#endif
}

// Probe 3: one full AICOnly soft barrier (public path == SYNCALL_SOFT_ATOMIC_BARRIER:
// add + ld_dev poll to target). Counter ends at block count once all cores arrive.
extern "C" __global__ AICORE void RunAicAtomicProbeBarrier_4103(__gm__ int32_t __out__* counter)
{
#if defined(__DAV_CUBE__)
    const int32_t total = static_cast<int32_t>(get_block_num());
    GlobalTensor<int32_t, pto::Shape<>, pto::Stride<>> gmWs(counter);
    SYNCALL<SyncAllMode::Soft, SyncCoreType::AICOnly>(gmWs, total);
#else
    (void)counter;
#endif
}

void LaunchAicAtomicProbeAdd(int32_t* counter, void* stream)
{
    RunAicAtomicProbeAdd_4101<<<kAicAtomicBlockCount, nullptr, stream>>>(counter);
}

void LaunchAicAtomicProbeAddDcci(int32_t* counter, void* stream)
{
    RunAicAtomicProbeAddDcci_4102<<<kAicAtomicBlockCount, nullptr, stream>>>(counter);
}

void LaunchAicAtomicProbeBarrier(int32_t* counter, void* stream)
{
    RunAicAtomicProbeBarrier_4103<<<kAicAtomicBlockCount, nullptr, stream>>>(counter);
}
