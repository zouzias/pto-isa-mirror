/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// Decisive MIX-mode probe: launch AIC + AIV together (ratio 1:2). AIC does ONE
// scalar GM store (marker[aicIdx]=42); AIV does a control scalar GM store
// (marker[idx]=100). Nothing else runs, so the stream sync return code and the
// marker read-back tell us unambiguously whether an A5 AIC (cube) core can write
// GM at all while in MIX mode (aic_atomic_probe only covered AIC-only mode).

#include "syncall_mix_common.hpp"

#if defined(SYNCALL_MIX_BUILD_AIC) && !defined(SYNCALL_MIX_REGISTER_BUILD)
#include "runtime/rt.h"

#include <cstdlib>
#include <cstdio>
#include <dlfcn.h>
#include <fstream>
#include <vector>
#endif

constexpr uint64_t kMixProbeTilingKey = 1301;
constexpr int32_t kMixProbeAicBlocks = 18;

#if defined(SYNCALL_MIX_BUILD_AIC)
PTO_SYNCALL_MIX_AIC_KERNEL_META(RunMixProbe_1301_mix_aic, 1, 2);
extern "C" __global__ AICORE void RunMixProbe_1301_mix_aic(__gm__ int32_t __out__* marker)
{
#if defined(__DAV_CUBE__)
    const int32_t aicIdx = static_cast<int32_t>(get_block_idx());
    __gm__ int32_t* slot = marker + aicIdx * kInt32PerCacheLine;
    slot[0] = 42;
    SoftDcci(static_cast<__gm__ void*>(slot));
    dsb(DSB_DDR);
#else
    (void)marker;
#endif
}
#endif

#if defined(SYNCALL_MIX_BUILD_AIV)
PTO_SYNCALL_MIX_AIC_KERNEL_META(RunMixProbe_1301_mix_aiv, 1, 2);
extern "C" __global__ AICORE void RunMixProbe_1301_mix_aiv(__gm__ int32_t __out__* marker)
{
#if defined(__DAV_VEC__)
    const int32_t idx = GetMixLogicalIdx();
    __gm__ int32_t* slot = marker + idx * kInt32PerCacheLine;
    slot[0] = 100;
    SoftDcci(static_cast<__gm__ void*>(slot));
    dsb(DSB_DDR);
#else
    (void)marker;
#endif
}
#endif

#if defined(SYNCALL_MIX_BUILD_AIC) && !defined(SYNCALL_MIX_REGISTER_BUILD)
namespace {
const char* GetMixProbeElfPath(const void* anchor)
{
#if defined(SYNCALL_MIX_REGISTER_OBJECT_PATH)
    (void)anchor;
    return SYNCALL_MIX_REGISTER_OBJECT_PATH;
#else
    Dl_info info{};
    if (dladdr(anchor, &info) == 0 || info.dli_fname == nullptr) {
        std::fprintf(stderr, "dladdr failed for A5 MIX probe kernel\n");
        std::abort();
    }
    return info.dli_fname;
#endif
}

std::vector<char> ReadMixProbeBinary(const char* path)
{
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::fprintf(stderr, "failed to open A5 MIX probe register ELF: %s\n", path);
        std::abort();
    }
    const std::streamsize size = file.tellg();
    if (size <= 0) {
        std::fprintf(stderr, "invalid A5 MIX probe register ELF size: %s\n", path);
        std::abort();
    }
    std::vector<char> data(static_cast<size_t>(size));
    file.seekg(0, std::ios::beg);
    if (!file.read(data.data(), size)) {
        std::fprintf(stderr, "failed to read A5 MIX probe register ELF: %s\n", path);
        std::abort();
    }
    return data;
}
} // namespace

void LaunchMixProbe(int32_t* marker, void* stream)
{
    const char* path = GetMixProbeElfPath(reinterpret_cast<const void*>(&LaunchMixProbe));
    static const std::vector<char> kernelBin = ReadMixProbeBinary(path);

    rtDevBinary_t binary{RT_DEV_BINARY_MAGIC_ELF, 0, kernelBin.data(), kernelBin.size()};
    void* handle = nullptr;
    rtError_t ret = rtRegisterAllKernel(&binary, &handle);
    if (ret != RT_ERROR_NONE || handle == nullptr) {
        ret = rtBinaryLoadWithoutTilingKey(kernelBin.data(), kernelBin.size(), &handle);
        if (ret != RT_ERROR_NONE || handle == nullptr) {
            std::fprintf(
                stderr, "register A5 MIX probe kernel failed, path=%s, size=%zu, ret=%d\n", path, kernelBin.size(),
                ret);
            std::abort();
        }
    }

    void* args[] = {marker};
    rtArgsEx_t argsInfo{};
    argsInfo.args = args;
    argsInfo.argsSize = sizeof(args);
    rtTaskCfgInfo_t cfgInfo{};
    ret = rtKernelLaunchWithHandleV2(
        handle, kMixProbeTilingKey, kMixProbeAicBlocks, &argsInfo, nullptr, stream, &cfgInfo);
    if (ret != RT_ERROR_NONE) {
        std::fprintf(stderr, "rtKernelLaunchWithHandleV2 failed for A5 MIX probe, ret=%d\n", ret);
        std::abort();
    }
}
#endif
