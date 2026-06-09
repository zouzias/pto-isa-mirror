/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// Hard MIX 1:1 SYNCALL test kernel (register-ELF launch).
//
// A true 1:1 ratio (20 AIC + 20 AIV) cannot be produced by a dav-c220 auto-split
// chevron launch: on ccec/bisheng GetTaskRation() is always 2, so auto-split is
// physically fixed at 1:2 (20 AIC + 40 AIV). CANN's own 1:1 mix operators
// (conv3d_transpose_v2, incre_flash_attention, ...) get a real 1:1 only on the
// opc compile path (GetTaskRation()==1); see the comment in
// incre_flash_attention_split_Bbn2s2_Us2.h. Hard sync needs a unified MIX FFTS
// context, so the soft dual-stream trick does not apply here.
//
// To get a real 1:1 hard sync we therefore reproduce the original
// pto_syncall_mix_kernel register-ELF path: compile the kernel for cube and vec
// arch, embed an explicit 1:1 ratio via PTO_SYNCALL_MIX_AIC_KERNEL_META, merge
// the nested device ELFs into a compact registration ELF
// (make_mix_register_elf.py), then launch with rtKernelLaunchWithHandleV2 so the
// runtime schedules 20 AIC + 20 AIV under one MIX context.

#include "syncall_mix_chevron_common.hpp"

#if defined(SYNCALL_MIX_BUILD_AIC) && !defined(SYNCALL_MIX_REGISTER_BUILD)
#include "runtime/rt.h"

#include <cstdlib>
#include <cstdio>
#include <dlfcn.h>
#include <fstream>
#include <vector>
#endif

// 910B4: 20 AIC + 20 AIV = 40 participants (910B1: 24+24=48).
constexpr int32_t kMix11HardParticipants = 40;
constexpr uint64_t kMix11HardTilingKey = 1101;

#if defined(SYNCALL_MIX_BUILD_AIC)
PTO_SYNCALL_MIX_AIC_KERNEL_META(RunSyncAllMix11_1101_mix_aic, 1, 1);

extern "C" __global__ AICORE void RunSyncAllMix11_1101_mix_aic(__gm__ uint64_t __in__ *fftsAddr,
                                                               __gm__ int32_t __out__ *out,
                                                               __gm__ int32_t __out__ *flags)
{
    RunMixSyncAllBody<kMix11HardParticipants, false>(fftsAddr, out, flags, nullptr);
}
#endif

#if defined(SYNCALL_MIX_BUILD_AIV)
PTO_SYNCALL_MIX_AIC_KERNEL_META(RunSyncAllMix11_1101_mix_aiv, 1, 1);

extern "C" __global__ AICORE void RunSyncAllMix11_1101_mix_aiv(__gm__ uint64_t __in__ *fftsAddr,
                                                               __gm__ int32_t __out__ *out,
                                                               __gm__ int32_t __out__ *flags)
{
    RunMixSyncAllBody<kMix11HardParticipants, false>(fftsAddr, out, flags, nullptr);
}
#endif

#if defined(SYNCALL_MIX_BUILD_AIC) && !defined(SYNCALL_MIX_REGISTER_BUILD)
namespace {
const char *GetCurrentSharedObjectPath(const void *anchor)
{
#if defined(SYNCALL_MIX_REGISTER_OBJECT_PATH)
    (void)anchor;
    return SYNCALL_MIX_REGISTER_OBJECT_PATH;
#else
    Dl_info info{};
    if (dladdr(anchor, &info) == 0 || info.dli_fname == nullptr) {
        std::fprintf(stderr, "dladdr failed for SYNCALL mix 1:1 hard kernel\n");
        std::abort();
    }
    return info.dli_fname;
#endif
}

std::vector<char> ReadCurrentSharedObject(const char *path)
{
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::fprintf(stderr, "failed to open SYNCALL mix 1:1 hard kernel binary: %s\n", path);
        std::abort();
    }

    const std::streamsize size = file.tellg();
    if (size <= 0) {
        std::fprintf(stderr, "invalid SYNCALL mix 1:1 hard kernel binary size: %s\n", path);
        std::abort();
    }

    std::vector<char> data(static_cast<size_t>(size));
    file.seekg(0, std::ios::beg);
    if (!file.read(data.data(), size)) {
        std::fprintf(stderr, "failed to read SYNCALL mix 1:1 hard kernel binary: %s\n", path);
        std::abort();
    }
    return data;
}

void LaunchHardMixKernel(const void *anchor, uint64_t tilingKey, uint8_t *ffts, int32_t *out, int32_t *flags,
                         void *stream)
{
    const char *path = GetCurrentSharedObjectPath(anchor);
    static const std::vector<char> kernelBinary = ReadCurrentSharedObject(path);
    rtDevBinary_t binary{RT_DEV_BINARY_MAGIC_ELF, 0, kernelBinary.data(), kernelBinary.size()};
    void *handle = nullptr;
    rtError_t ret = rtRegisterAllKernel(&binary, &handle);
    if (ret != RT_ERROR_NONE || handle == nullptr) {
        ret = rtBinaryLoadWithoutTilingKey(kernelBinary.data(), kernelBinary.size(), &handle);
        if (ret != RT_ERROR_NONE || handle == nullptr) {
            std::fprintf(stderr, "register SYNCALL mix 1:1 hard kernel failed, path=%s, size=%zu, ret=%d\n", path,
                         kernelBinary.size(), ret);
            std::abort();
        }
    }

    void *args[] = {reinterpret_cast<uint64_t *>(ffts), out, flags};
    rtArgsEx_t argsInfo{};
    argsInfo.args = args;
    argsInfo.argsSize = sizeof(args);
    rtTaskCfgInfo_t cfgInfo{};
    ret = rtKernelLaunchWithHandleV2(handle, tilingKey, 20, &argsInfo, nullptr, stream, &cfgInfo);
    if (ret != RT_ERROR_NONE) {
        std::fprintf(stderr, "rtKernelLaunchWithHandleV2 failed for SYNCALL mix 1:1 hard, ret=%d\n", ret);
        std::abort();
    }
}
} // namespace

void LaunchHardMix11(uint8_t *ffts, int32_t *out, int32_t *flags, void *stream)
{
    LaunchHardMixKernel(reinterpret_cast<const void *>(&LaunchHardMix11), kMix11HardTilingKey, ffts, out, flags, stream);
}
#endif
