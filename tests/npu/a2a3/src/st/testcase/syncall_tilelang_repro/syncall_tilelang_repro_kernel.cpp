/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include "syncall_tilelang_repro_common.hpp"
#include <pto/pto-inst.hpp>
#include "acl/acl.h"
#include "runtime/rt.h"

using namespace pto;

PTO_SYNCALL_MIX_AIC_KERNEL_META(syncall_test_kernel, 1, 1);
PTO_SYNCALL_AIV_KERNEL_META(syncall_test_kernel);

PTO_INTERNAL int32_t GetReproWorkspaceElemsDevice(ReproVariant variant)
{
    switch (variant) {
        case ReproVariant::FixCacheLine:
        case ReproVariant::FixCacheLineSyncMix:
            return kReproCacheLineWsSize;
        default:
            return kReproBaselineWsSize;
    }
}

PTO_INTERNAL int32_t GetSlotStride(ReproVariant variant)
{
    switch (variant) {
        case ReproVariant::FixCacheLine:
        case ReproVariant::FixCacheLineSyncMix:
            return SYNCALL_SOFT_SLOT_INT32;
        default:
            return kReproNumExperts;
    }
}

PTO_INTERNAL int32_t GetCopyBurstCount(ReproVariant variant)
{
    switch (variant) {
        case ReproVariant::FixCacheLine:
        case ReproVariant::FixCacheLineSyncMix:
            return kReproNumCores;
        default:
            return kReproNumCores;
    }
}

PTO_INTERNAL void RunAicSync(ReproVariant variant)
{
#if defined(__DAV_CUBE__)
    switch (variant) {
        case ReproVariant::Baseline:
        case ReproVariant::FixCacheLine:
            SYNCALL<SyncCoreType::AICOnly>();
            break;
        case ReproVariant::FixSyncMix:
        case ReproVariant::FixCacheLineSyncMix:
            SYNCALL<SyncCoreType::Mix>();
            break;
        case ReproVariant::FixAivOnly:
        default:
            break;
    }
#else
    (void)variant;
#endif
}

PTO_INTERNAL void RunAivSync(ReproVariant variant)
{
#if defined(__DAV_VEC__)
    switch (variant) {
        case ReproVariant::FixSyncMix:
        case ReproVariant::FixCacheLineSyncMix:
            SYNCALL<SyncCoreType::Mix>();
            break;
        case ReproVariant::FixAivOnly:
            SYNCALL();
            break;
        case ReproVariant::Baseline:
        case ReproVariant::FixCacheLine:
        default:
            SYNCALL<SyncCoreType::AIVOnly>();
            break;
    }
#else
    (void)variant;
#endif
}

PTO_INTERNAL void InvalidateWorkspace(__gm__ int32_t *workspaceGm, ReproVariant variant)
{
#if defined(__DAV_VEC__)
    const int32_t stride = GetSlotStride(variant);
    const int32_t slots = kReproNumCores;
    for (int32_t i = 0; i < slots; ++i) {
        __asm__ __volatile__("");
        dcci(static_cast<__gm__ void *>(workspaceGm + i * stride), SINGLE_CACHE_LINE);
        __asm__ __volatile__("");
    }
    dsb(DSB_DDR);
#else
    (void)workspaceGm;
    (void)variant;
#endif
}

AICORE void syncall_test_kernel(__gm__ int32_t *workspaceGm, __gm__ int32_t *resultGm, uint64_t fftsAddr,
                                int32_t variantRaw)
{
    const auto variant = static_cast<ReproVariant>(variantRaw);
    const auto cid = get_block_idx();
    const int32_t slotStride = GetSlotStride(variant);
    const int32_t wsElems = GetReproWorkspaceElemsDevice(variant);

    set_ffts_base_addr(fftsAddr);

#if defined(__DAV_CUBE__)
    RunAicSync(variant);
#endif

#if defined(__DAV_VEC__)
    __ubuf__ int32_t *histUb = reinterpret_cast<__ubuf__ int32_t *>(0x0);
    for (int32_t e = 0; e < kReproNumExperts; ++e) {
        histUb[e] = cid + 1;
    }
    for (int32_t e = kReproNumExperts; e < 8; ++e) {
        histUb[e] = 0;
    }
    pipe_barrier(PIPE_ALL);
    copy_ubuf_to_gm(static_cast<__gm__ void *>(workspaceGm + cid * slotStride), static_cast<__ubuf__ void *>(histUb), 0,
                    1, 1, 0, 0);
    pipe_barrier(PIPE_ALL);
    dcci(static_cast<__gm__ void *>(workspaceGm + cid * slotStride), SINGLE_CACHE_LINE);
    dsb(DSB_DDR);
    RunAivSync(variant);
    InvalidateWorkspace(workspaceGm, variant);
    __ubuf__ int32_t *wsUb = reinterpret_cast<__ubuf__ int32_t *>(0x1000);
    copy_gm_to_ubuf(static_cast<__ubuf__ void *>(wsUb), static_cast<__gm__ void *>(workspaceGm), 0, 1,
                    GetCopyBurstCount(variant), 0, 0);
    pipe_barrier(PIPE_ALL);
    copy_ubuf_to_gm(static_cast<__gm__ void *>(resultGm + cid * wsElems), static_cast<__ubuf__ void *>(wsUb), 0, 1,
                    GetCopyBurstCount(variant), 0, 0);
    pipe_barrier(PIPE_ALL);
#endif
}

extern "C" __global__ AICORE void launch_kernel(__gm__ uint8_t *workspaceGm, __gm__ uint8_t *resultGm,
                                                uint64_t fftsAddr, int32_t variantRaw)
{
    syncall_test_kernel(reinterpret_cast<__gm__ int32_t *>(workspaceGm), reinterpret_cast<__gm__ int32_t *>(resultGm),
                        fftsAddr, variantRaw);
}

void LaunchSyncallTilelangRepro(uint8_t *workspaceGm, uint8_t *resultGm, void *stream, int32_t variantRaw)
{
    uint32_t fftsLen{0};
    uint64_t fftsAddr{0};
    rtGetC2cCtrlAddr(&fftsAddr, &fftsLen);
    launch_kernel<<<kReproNumCores, nullptr, stream>>>(workspaceGm, resultGm, fftsAddr, variantRaw);
}
