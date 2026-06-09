/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// Soft MIX 1:1 SYNCALL test kernel.
//
// A 1:1 ratio needs 20 AIC + 20 AIV. dav-c220 auto-split is fixed at the
// hardware ratio 1:2 (20 AIC + 40 AIV), so 1:1 cannot be expressed by a single
// auto-split chevron kernel. Instead the AIC and AIV halves are compiled for
// their own arch (cube / vec) into one shared object and launched on two streams
// (chevron <<<20>>> each). Soft sync uses GM-workspace polling and does not need
// a unified MIX FFTS context, so two independent streams are fine.
//
// This mirrors the original pto_syncall_mix_kernel soft 1:1 path; it does NOT
// benefit from auto-split simplification (that only applies to 1:2).

#include "syncall_mix_chevron_common.hpp"
#include "acl/acl.h"

constexpr int32_t kMix11Participants = 40; // 20 AIC + 20 AIV (1:1)
constexpr int32_t kMix11CoreCount = 20;

#if defined(SYNCALL_MIX_BUILD_AIC)
// Defined in the AIV (dav-c220-vec) object; declared here so the host launcher
// in this AIC object can issue its chevron launch.
extern "C" __global__ AICORE void launch_soft_mix11_aiv(__gm__ uint64_t __in__ *fftsAddr, __gm__ int32_t __out__ *out,
                                                        __gm__ int32_t __out__ *flags,
                                                        __gm__ int32_t __out__ *syncWorkspace);

extern "C" __global__ AICORE void launch_soft_mix11_aic(__gm__ uint64_t __in__ *fftsAddr, __gm__ int32_t __out__ *out,
                                                        __gm__ int32_t __out__ *flags,
                                                        __gm__ int32_t __out__ *syncWorkspace)
{
    RunMixSyncAllBody<kMix11Participants, true>(fftsAddr, out, flags, syncWorkspace);
}

void LaunchSoftMix11(uint8_t *ffts, int32_t *out, int32_t *flags, int32_t *syncWs, void *stream)
{
    aclrtStream aivStream;
    (void)aclrtCreateStream(&aivStream);
    launch_soft_mix11_aic<<<kMix11CoreCount, nullptr, stream>>>(reinterpret_cast<uint64_t *>(ffts), out, flags, syncWs);
    launch_soft_mix11_aiv<<<kMix11CoreCount, nullptr, aivStream>>>(reinterpret_cast<uint64_t *>(ffts), out, flags,
                                                                   syncWs);
    (void)aclrtSynchronizeStream(aivStream);
    (void)aclrtDestroyStream(aivStream);
}
#endif

#if defined(SYNCALL_MIX_BUILD_AIV)
extern "C" __global__ AICORE void launch_soft_mix11_aiv(__gm__ uint64_t __in__ *fftsAddr, __gm__ int32_t __out__ *out,
                                                        __gm__ int32_t __out__ *flags,
                                                        __gm__ int32_t __out__ *syncWorkspace)
{
    RunMixSyncAllBody<kMix11Participants, true>(fftsAddr, out, flags, syncWorkspace);
}
#endif
