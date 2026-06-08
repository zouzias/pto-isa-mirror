/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include "syncall_mix_chevron_common.hpp"
#include "acl/acl.h"
#include "runtime/rt.h"

constexpr int32_t kChevronMixParticipants = 60; // 20 AIC + 40 AIV (1:2 ratio)
constexpr int32_t kChevronCoreCount = 20;

extern "C" __global__ AICORE void launch_hard_mix_12(
    __gm__ uint64_t __in__ *fftsAddr,
    __gm__ int32_t __out__ *out,
    __gm__ int32_t __out__ *flags)
{
    RunMixSyncAllBody<kChevronMixParticipants, false>(fftsAddr, out, flags, nullptr);
}

extern "C" __global__ AICORE void launch_soft_mix_12(
    __gm__ uint64_t __in__ *fftsAddr,
    __gm__ int32_t __out__ *out,
    __gm__ int32_t __out__ *flags,
    __gm__ int32_t __out__ *syncWorkspace)
{
    RunMixSyncAllBody<kChevronMixParticipants, true>(fftsAddr, out, flags, syncWorkspace);
}

void LaunchHardMix12(uint8_t *ffts, int32_t *out, int32_t *flags, void *stream)
{
    launch_hard_mix_12<<<kChevronCoreCount, nullptr, stream>>>(
        reinterpret_cast<uint64_t *>(ffts), out, flags);
}

void LaunchSoftMix12(uint8_t *ffts, int32_t *out, int32_t *flags, int32_t *syncWs, void *stream)
{
    launch_soft_mix_12<<<kChevronCoreCount, nullptr, stream>>>(
        reinterpret_cast<uint64_t *>(ffts), out, flags, syncWs);
}
