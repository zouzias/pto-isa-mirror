/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// MIX-mode probes launched with AIC + AIV together (ratio 1:2):
//   key 1301 RunMixProbe        : each core does ONE scalar GM store (can AIC
//                                 write GM in MIX mode at all?).
//   key 1302 RunMixBarrierProbe : each core marks stage 0, runs ONE real
//                                 SYNCALL<Soft,Mix> barrier, marks stage 1 --
//                                 isolates whether the bare mix atomic barrier
//                                 (no proxy/business) faults.
// Markers use a 128-byte (32 int32) stride so adjacent cores never share a cache
// line (the 32-byte kInt32PerCacheLine spacing caused dcci false-sharing).

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
constexpr uint64_t kMixBarrierProbeTilingKey = 1302;
constexpr int32_t kMixProbeAicBlocks = 18;
constexpr int32_t kMixProbeParticipants = 54;
constexpr int32_t kMarkStride = 32; // 128 bytes: no cache-line false sharing.

PTO_INTERNAL void BigMark(__gm__ int32_t* marker, int32_t slotIdx, int32_t stage)
{
#if defined(__DAV_CUBE__) || defined(__DAV_VEC__)
    if (marker != nullptr) {
        __gm__ int32_t* slot = marker + slotIdx * kMarkStride;
        slot[0] = stage;
        SoftDcci(static_cast<__gm__ void*>(slot));
        dsb(DSB_DDR);
    }
#else
    (void)marker;
    (void)slotIdx;
    (void)stage;
#endif
}

#if defined(SYNCALL_MIX_BUILD_AIC)
PTO_SYNCALL_MIX_AIC_KERNEL_META(RunMixProbe_1301_mix_aic, 1, 2);
extern "C" __global__ AICORE void RunMixProbe_1301_mix_aic(__gm__ int32_t __out__* marker)
{
#if defined(__DAV_CUBE__)
    BigMark(marker, GetMixLogicalIdx(), 42);
#else
    (void)marker;
#endif
}

PTO_SYNCALL_MIX_AIC_KERNEL_META(RunMixBarrierProbe_1302_mix_aic, 1, 2);
extern "C" __global__ AICORE void RunMixBarrierProbe_1302_mix_aic(
    __gm__ int32_t __out__* syncWorkspace, __gm__ int32_t __out__* marker)
{
#if defined(__DAV_CUBE__)
    const int32_t idx = GetMixLogicalIdx();
    BigMark(marker, idx, 0);
    GlobalTensor<int32_t, pto::Shape<>, pto::Stride<>> gmWs(syncWorkspace);
    SYNCALL<SyncAllMode::Soft, SyncCoreType::Mix>(gmWs, kMixProbeParticipants);
    BigMark(marker, idx, 1);
#else
    (void)syncWorkspace;
    (void)marker;
#endif
}
#endif

#if defined(SYNCALL_MIX_BUILD_AIV)
PTO_SYNCALL_MIX_AIC_KERNEL_META(RunMixProbe_1301_mix_aiv, 1, 2);
extern "C" __global__ AICORE void RunMixProbe_1301_mix_aiv(__gm__ int32_t __out__* marker)
{
#if defined(__DAV_VEC__)
    BigMark(marker, GetMixLogicalIdx(), 100);
#else
    (void)marker;
#endif
}

PTO_SYNCALL_MIX_AIC_KERNEL_META(RunMixBarrierProbe_1302_mix_aiv, 1, 2);
extern "C" __global__ AICORE void RunMixBarrierProbe_1302_mix_aiv(
    __gm__ int32_t __out__* syncWorkspace, __gm__ int32_t __out__* marker)
{
#if defined(__DAV_VEC__)
    const int32_t idx = GetMixLogicalIdx();
    BigMark(marker, idx, 0);
    GlobalTensor<int32_t, pto::Shape<>, pto::Stride<>> gmWs(syncWorkspace);
    SYNCALL<SyncAllMode::Soft, SyncCoreType::Mix>(gmWs, kMixProbeParticipants);
    BigMark(marker, idx, 1);
#else
    (void)syncWorkspace;
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

void* RegisterMixProbe(const void* anchor)
{
    const char* path = GetMixProbeElfPath(anchor);
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
    return handle;
}

void LaunchMixProbeHandle(void* handle, uint64_t tilingKey, void** args, size_t argsSize, void* stream)
{
    rtArgsEx_t argsInfo{};
    argsInfo.args = args;
    argsInfo.argsSize = argsSize;
    rtTaskCfgInfo_t cfgInfo{};
    rtError_t ret =
        rtKernelLaunchWithHandleV2(handle, tilingKey, kMixProbeAicBlocks, &argsInfo, nullptr, stream, &cfgInfo);
    if (ret != RT_ERROR_NONE) {
        std::fprintf(stderr, "rtKernelLaunchWithHandleV2 failed for A5 MIX probe key=%lu, ret=%d\n", tilingKey, ret);
        std::abort();
    }
}
} // namespace

void LaunchMixProbe(int32_t* marker, void* stream)
{
    void* handle = RegisterMixProbe(reinterpret_cast<const void*>(&LaunchMixProbe));
    void* args[] = {marker};
    LaunchMixProbeHandle(handle, kMixProbeTilingKey, args, sizeof(args), stream);
}

void LaunchMixBarrierProbe(int32_t* syncWorkspace, int32_t* marker, void* stream)
{
    void* handle = RegisterMixProbe(reinterpret_cast<const void*>(&LaunchMixProbe));
    void* args[] = {syncWorkspace, marker};
    LaunchMixProbeHandle(handle, kMixBarrierProbeTilingKey, args, sizeof(args), stream);
}
#endif
