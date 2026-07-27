/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// AIC-only software SYNCALL ST, dav-c310-cube single-chevron launch. Every cube
// block publishes a flag to GM via a scalar store (AIC has no copy_ubuf_to_gm,
// but A5 AIC does support scalar GM store/ld_dev), runs the
// AIC-only soft barrier, then scalar-reads every flag to confirm all cube cores
// synchronized. out[idx] == 1 iff this core saw all peers' round-1 and round-2
// writes, proving the barrier ordered them.

#include <pto/pto-inst.hpp>
#include "acl/acl.h"

using namespace pto;

constexpr int32_t kAicSoftBlockCount = 18;
// 64-byte (16 int32) per-core slot: A5 cache line is 64 B, and cube publishes via
// scalar store + dcci (whole-cache-line write-back). At the old 32-byte stride two
// cores shared one line, so one core's dcci clobbered the neighbor's flag (false
// sharing) and the cross-core check failed. One full cache line per slot fixes it.
// Must match int32PerCacheLine in case_soft_aic_only_all_blocks.
constexpr int32_t kAicSoftCacheLine = 16;

PTO_INTERNAL void AicScalarStore(__gm__ int32_t* dst, int32_t value)
{
    dst[0] = value;
    dcci(static_cast<__gm__ void*>(dst), SINGLE_CACHE_LINE);
    dsb(DSB_DDR);
}

// Read every peer flag with ld_dev (non-cacheable, straight from DDR). A batched
// dcci + cached scalar load returns stale values on the A5 cube core; ld_dev is
// the read idiom used by SYNCALL_SOFT_ATOMIC_LOAD.
PTO_INTERNAL int32_t AicCheckFlags(__gm__ int32_t* flags, int32_t total, int32_t multiplier)
{
    int32_t allVisible = 1;
    for (int32_t i = 0; i < total; ++i) {
        __gm__ int32_t* slot = flags + i * kAicSoftCacheLine;
        const int32_t value = static_cast<int32_t>(ld_dev(reinterpret_cast<__gm__ uint32_t*>(slot), 0));
        if (value != (i + 1) * multiplier) {
            allVisible = 0;
        }
    }
    return allVisible;
}

extern "C" __global__ AICORE void RunSoftSyncAllAIC(
    __gm__ int32_t __out__* out, __gm__ int32_t __out__* flags, __gm__ int32_t __out__* syncWorkspace)
{
#if defined(__DAV_CUBE__)
    const int32_t total = static_cast<int32_t>(get_block_num());
    const int32_t idx = static_cast<int32_t>(get_block_idx());
    GlobalTensor<int32_t, pto::Shape<>, pto::Stride<>> gmWs(syncWorkspace);

    AicScalarStore(flags + idx * kAicSoftCacheLine, idx + 1);
    SYNCALL<SyncAllMode::Soft, SyncCoreType::AICOnly>(gmWs, total);
    const int32_t allFirstVisible = AicCheckFlags(flags, total, 1);

    SYNCALL<SyncAllMode::Soft, SyncCoreType::AICOnly>(gmWs, total);
    AicScalarStore(flags + idx * kAicSoftCacheLine, (idx + 1) * 2);
    SYNCALL<SyncAllMode::Soft, SyncCoreType::AICOnly>(gmWs, total);

    const int32_t allSecondVisible = AicCheckFlags(flags, total, 2);
    AicScalarStore(out + idx * kAicSoftCacheLine, allFirstVisible & allSecondVisible);
#else
    (void)out;
    (void)flags;
    (void)syncWorkspace;
#endif
}

void LaunchSoftSyncAllAIC(int32_t* out, int32_t* flags, int32_t* syncWorkspace, void* stream)
{
    RunSoftSyncAllAIC<<<kAicSoftBlockCount, nullptr, stream>>>(out, flags, syncWorkspace);
}
