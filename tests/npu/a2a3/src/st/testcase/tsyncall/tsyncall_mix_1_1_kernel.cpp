/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include "tsyncall_mix_common.hpp"

constexpr int32_t kMix11HardParticipants = 48;
constexpr int32_t kMix11SoftParticipants = 48;

#if defined(TSYNCALL_MIX_BUILD_AIC)
extern "C" __global__ AICORE void RunTSyncAllMix11_mix_aiv(__gm__ uint64_t __in__ *fftsAddr,
                                                           __gm__ int32_t __out__ *out,
                                                           __gm__ int32_t __out__ *flags);
extern "C" __global__ AICORE void RunSoftTSyncAllMix11_mix_aiv(__gm__ uint64_t __in__ *fftsAddr,
                                                               __gm__ int32_t __out__ *out,
                                                               __gm__ int32_t __out__ *flags,
                                                               __gm__ int32_t __out__ *syncWorkspace);
#endif

#if defined(TSYNCALL_MIX_BUILD_AIC)
PTO_A2A3_TSYNCALL_MIX_AIC_KERNEL_META(RunTSyncAllMix11_mix_aic, 1, 1);
PTO_A2A3_TSYNCALL_MIX_AIC_KERNEL_META(RunSoftTSyncAllMix11_mix_aic, 1, 1);

extern "C" __global__ AICORE void RunTSyncAllMix11_mix_aic(__gm__ uint64_t __in__ *fftsAddr,
                                                           __gm__ int32_t __out__ *out,
                                                           __gm__ int32_t __out__ *flags)
{
    RunMixSyncAllBody<kMix11HardParticipants, false>(fftsAddr, out, flags, nullptr);
}

extern "C" __global__ AICORE void RunSoftTSyncAllMix11_mix_aic(__gm__ uint64_t __in__ *fftsAddr,
                                                               __gm__ int32_t __out__ *out,
                                                               __gm__ int32_t __out__ *flags,
                                                               __gm__ int32_t __out__ *syncWorkspace)
{
    RunMixSyncAllBody<kMix11SoftParticipants, true>(fftsAddr, out, flags, syncWorkspace);
}

#endif

#if defined(TSYNCALL_MIX_BUILD_AIV)
PTO_A2A3_TSYNCALL_MIX_AIC_KERNEL_META(RunTSyncAllMix11_mix_aiv, 1, 1);
PTO_A2A3_TSYNCALL_MIX_AIC_KERNEL_META(RunSoftTSyncAllMix11_mix_aiv, 1, 1);

extern "C" __global__ AICORE void RunTSyncAllMix11_mix_aiv(__gm__ uint64_t __in__ *fftsAddr,
                                                           __gm__ int32_t __out__ *out,
                                                           __gm__ int32_t __out__ *flags)
{
    RunMixSyncAllBody<kMix11HardParticipants, false>(fftsAddr, out, flags, nullptr);
}

extern "C" __global__ AICORE void RunSoftTSyncAllMix11_mix_aiv(__gm__ uint64_t __in__ *fftsAddr,
                                                               __gm__ int32_t __out__ *out,
                                                               __gm__ int32_t __out__ *flags,
                                                               __gm__ int32_t __out__ *syncWorkspace)
{
    RunMixSyncAllBody<kMix11SoftParticipants, true>(fftsAddr, out, flags, syncWorkspace);
}

#endif

#if defined(TSYNCALL_MIX_BUILD_AIC)
void LaunchTSyncAllMix11(uint8_t *ffts, int32_t *out, int32_t *flags, void *stream)
{
    RunTSyncAllMix11_mix_aic<<<24, nullptr, stream>>>(reinterpret_cast<uint64_t *>(ffts), out, flags);
}

void LaunchSoftTSyncAllMix11(uint8_t *ffts, int32_t *out, int32_t *flags, int32_t *syncWorkspace, void *stream)
{
    aclrtStream aivStream;
    (void)aclrtCreateStream(&aivStream);
    RunSoftTSyncAllMix11_mix_aic<<<24, nullptr, stream>>>(reinterpret_cast<uint64_t *>(ffts), out, flags,
                                                          syncWorkspace);
    RunSoftTSyncAllMix11_mix_aiv<<<24, nullptr, aivStream>>>(reinterpret_cast<uint64_t *>(ffts), out, flags,
                                                             syncWorkspace);
    (void)aclrtSynchronizeStream(aivStream);
    (void)aclrtDestroyStream(aivStream);
}
#endif
