/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// AIC-only software SYNCALL ST. Every cube block publishes a flag to GM via a
// scalar store (AIC has no copy_ubuf_to_gm; scalar GM store/ld_dev on A5 AIC is
// verified by the aic_probe ST), runs the AIC-only soft barrier, then scalar-reads
// every flag to confirm all cube cores synchronized. out[idx] == 1 iff this core
// saw all peers' round-1 and round-2 writes, proving the barrier ordered them.

#include <pto/pto-inst.hpp>
#include "acl/acl.h"

#if defined(SYNCALL_MIX_BUILD_AIC) && !defined(SYNCALL_MIX_REGISTER_BUILD)
#include "runtime/rt.h"

#include <cstdlib>
#include <cstdio>
#include <dlfcn.h>
#include <fstream>
#include <vector>
#endif

using namespace pto;

constexpr int32_t kAicSoftBlockCount = 18;
constexpr int32_t kAicSoftCacheLine = 8;
constexpr uint64_t kAicSoftTilingKey = 3101;

#if defined(SYNCALL_MIX_BUILD_AIC)
PTO_INTERNAL void AicScalarStore(__gm__ int32_t* dst, int32_t value)
{
    dst[0] = value;
    dcci(static_cast<__gm__ void*>(dst), SINGLE_CACHE_LINE);
    dsb(DSB_DDR);
}

PTO_INTERNAL int32_t AicCheckFlags(__gm__ int32_t* flags, int32_t total, int32_t multiplier)
{
    for (int32_t i = 0; i < total; ++i) {
        dcci(static_cast<__gm__ void*>(flags + i * kAicSoftCacheLine), SINGLE_CACHE_LINE);
    }
    dsb(DSB_DDR);
    int32_t allVisible = 1;
    for (int32_t i = 0; i < total; ++i) {
        if ((flags + i * kAicSoftCacheLine)[0] != (i + 1) * multiplier) {
            allVisible = 0;
        }
    }
    return allVisible;
}

PTO_SYNCALL_MIX_AIC_KERNEL_META(RunSoftSyncAllAIC_3101_mix_aic, 1, 0);
extern "C" __global__ AICORE void RunSoftSyncAllAIC_3101_mix_aic(
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
#endif

#if defined(SYNCALL_MIX_BUILD_AIV)
PTO_SYNCALL_MIX_AIC_KERNEL_META(RunSoftSyncAllAIC_3101_mix_aiv, 1, 0);
extern "C" __global__ AICORE void RunSoftSyncAllAIC_3101_mix_aiv(
    __gm__ int32_t __out__* out, __gm__ int32_t __out__* flags, __gm__ int32_t __out__* syncWorkspace)
{
    (void)out;
    (void)flags;
    (void)syncWorkspace;
}
#endif

#if defined(SYNCALL_MIX_BUILD_AIC) && !defined(SYNCALL_MIX_REGISTER_BUILD)
namespace {
const char* GetAicSoftElfPath(const void* anchor)
{
#if defined(SYNCALL_MIX_REGISTER_OBJECT_PATH)
    (void)anchor;
    return SYNCALL_MIX_REGISTER_OBJECT_PATH;
#else
    Dl_info info{};
    if (dladdr(anchor, &info) == 0 || info.dli_fname == nullptr) {
        std::fprintf(stderr, "dladdr failed for A5 SYNCALL AIC-only soft kernel\n");
        std::abort();
    }
    return info.dli_fname;
#endif
}

std::vector<char> ReadAicSoftBinary(const char* path)
{
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::fprintf(stderr, "failed to open A5 AIC-only soft register ELF: %s\n", path);
        std::abort();
    }
    const std::streamsize size = file.tellg();
    if (size <= 0) {
        std::fprintf(stderr, "invalid A5 AIC-only soft register ELF size: %s\n", path);
        std::abort();
    }
    std::vector<char> data(static_cast<size_t>(size));
    file.seekg(0, std::ios::beg);
    if (!file.read(data.data(), size)) {
        std::fprintf(stderr, "failed to read A5 AIC-only soft register ELF: %s\n", path);
        std::abort();
    }
    return data;
}
} // namespace

void LaunchSoftSyncAllAIC(int32_t* out, int32_t* flags, int32_t* syncWorkspace, void* stream)
{
    const char* path = GetAicSoftElfPath(reinterpret_cast<const void*>(&LaunchSoftSyncAllAIC));
    static const std::vector<char> kernelBin = ReadAicSoftBinary(path);

    rtDevBinary_t binary{RT_DEV_BINARY_MAGIC_ELF, 0, kernelBin.data(), kernelBin.size()};
    void* handle = nullptr;
    rtError_t ret = rtRegisterAllKernel(&binary, &handle);
    if (ret != RT_ERROR_NONE || handle == nullptr) {
        ret = rtBinaryLoadWithoutTilingKey(kernelBin.data(), kernelBin.size(), &handle);
        if (ret != RT_ERROR_NONE || handle == nullptr) {
            std::fprintf(
                stderr, "register A5 AIC-only soft kernel failed, path=%s, size=%zu, ret=%d\n", path, kernelBin.size(),
                ret);
            std::abort();
        }
    }

    void* args[] = {out, flags, syncWorkspace};
    rtArgsEx_t argsInfo{};
    argsInfo.args = args;
    argsInfo.argsSize = sizeof(args);
    rtTaskCfgInfo_t cfgInfo{};
    ret = rtKernelLaunchWithHandleV2(
        handle, kAicSoftTilingKey, kAicSoftBlockCount, &argsInfo, nullptr, stream, &cfgInfo);
    if (ret != RT_ERROR_NONE) {
        std::fprintf(stderr, "rtKernelLaunchWithHandleV2 failed for A5 AIC-only soft, ret=%d\n", ret);
        std::abort();
    }
}
#endif
