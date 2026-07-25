/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// Minimal AIC-only probes that isolate whether the A5 AIC (dav-c310 cube) core
// can execute plain GM scalar store, st_atomic and ld_dev at runtime. Each probe
// is launched on a single AIC block; the stream sync return code (0 vs 507015)
// and the read-back GM contents form the truth table interpreted by main.cpp.

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

constexpr int32_t kProbeBlockCount = 1;
constexpr int32_t kProbeCacheLine = 8;
constexpr uint64_t kProbeStoreKey = 4001;
constexpr uint64_t kProbeStAtomicKey = 4002;
constexpr uint64_t kProbeLdDevKey = 4003;

#if defined(SYNCALL_MIX_BUILD_AIC)
// Probe A: plain AIC scalar GM store baseline. out[0] preset 0, expect 42.
PTO_SYNCALL_MIX_AIC_KERNEL_META(RunAicProbeStore_4001_mix_aic, 1, 0);
extern "C" __global__ AICORE void RunAicProbeStore_4001_mix_aic(__gm__ int32_t __out__* out)
{
#if defined(__DAV_CUBE__)
    out[0] = 42;
    dcci(static_cast<__gm__ void*>(out), SINGLE_CACHE_LINE);
    dsb(DSB_DDR);
#else
    (void)out;
#endif
}

// Probe B: AIC st_atomic add. out[0] preset 10, expect 11 if supported.
PTO_SYNCALL_MIX_AIC_KERNEL_META(RunAicProbeStAtomic_4002_mix_aic, 1, 0);
extern "C" __global__ AICORE void RunAicProbeStAtomic_4002_mix_aic(__gm__ int32_t __out__* out)
{
#if defined(__DAV_CUBE__)
    set_st_atomic_cfg(ATOMIC_S32, ATOMIC_SUM);
    dcci(static_cast<__gm__ void*>(out), SINGLE_CACHE_LINE);
    st_atomic<int32_t>(1, out);
    dcci(static_cast<__gm__ void*>(out), SINGLE_CACHE_LINE);
    dsb(DSB_DDR);
#else
    (void)out;
#endif
}

// Probe C: AIC ld_dev read. out[8] preset 1234, out[0] preset 0; read out[8]
// via ld_dev then publish it to out[0] (plain store, interpret with Probe A).
PTO_SYNCALL_MIX_AIC_KERNEL_META(RunAicProbeLdDev_4003_mix_aic, 1, 0);
extern "C" __global__ AICORE void RunAicProbeLdDev_4003_mix_aic(__gm__ int32_t __out__* out)
{
#if defined(__DAV_CUBE__)
    __gm__ int32_t* src = out + kProbeCacheLine;
    dcci(static_cast<__gm__ void*>(src), SINGLE_CACHE_LINE);
    const int32_t value = static_cast<int32_t>(ld_dev(reinterpret_cast<__gm__ uint32_t*>(src), 0));
    out[0] = value;
    dcci(static_cast<__gm__ void*>(out), SINGLE_CACHE_LINE);
    dsb(DSB_DDR);
#else
    (void)out;
#endif
}
#endif

#if defined(SYNCALL_MIX_BUILD_AIV)
PTO_SYNCALL_MIX_AIC_KERNEL_META(RunAicProbeStore_4001_mix_aiv, 1, 0);
extern "C" __global__ AICORE void RunAicProbeStore_4001_mix_aiv(__gm__ int32_t __out__* out) { (void)out; }

PTO_SYNCALL_MIX_AIC_KERNEL_META(RunAicProbeStAtomic_4002_mix_aiv, 1, 0);
extern "C" __global__ AICORE void RunAicProbeStAtomic_4002_mix_aiv(__gm__ int32_t __out__* out) { (void)out; }

PTO_SYNCALL_MIX_AIC_KERNEL_META(RunAicProbeLdDev_4003_mix_aiv, 1, 0);
extern "C" __global__ AICORE void RunAicProbeLdDev_4003_mix_aiv(__gm__ int32_t __out__* out) { (void)out; }
#endif

#if defined(SYNCALL_MIX_BUILD_AIC) && !defined(SYNCALL_MIX_REGISTER_BUILD)
namespace {
const char* GetProbeElfPath(const void* anchor)
{
#if defined(SYNCALL_MIX_REGISTER_OBJECT_PATH)
    (void)anchor;
    return SYNCALL_MIX_REGISTER_OBJECT_PATH;
#else
    Dl_info info{};
    if (dladdr(anchor, &info) == 0 || info.dli_fname == nullptr) {
        std::fprintf(stderr, "dladdr failed for A5 AIC atomic probe kernel\n");
        std::abort();
    }
    return info.dli_fname;
#endif
}

std::vector<char> ReadProbeBinary(const char* path)
{
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::fprintf(stderr, "failed to open A5 AIC atomic probe register ELF: %s\n", path);
        std::abort();
    }
    const std::streamsize size = file.tellg();
    if (size <= 0) {
        std::fprintf(stderr, "invalid A5 AIC atomic probe register ELF size: %s\n", path);
        std::abort();
    }
    std::vector<char> data(static_cast<size_t>(size));
    file.seekg(0, std::ios::beg);
    if (!file.read(data.data(), size)) {
        std::fprintf(stderr, "failed to read A5 AIC atomic probe register ELF: %s\n", path);
        std::abort();
    }
    return data;
}

void LaunchAicProbe(const void* anchor, uint64_t tilingKey, int32_t* out, void* stream)
{
    const char* path = GetProbeElfPath(anchor);
    static const std::vector<char> kernelBin = ReadProbeBinary(path);

    rtDevBinary_t binary{RT_DEV_BINARY_MAGIC_ELF, 0, kernelBin.data(), kernelBin.size()};
    void* handle = nullptr;
    rtError_t ret = rtRegisterAllKernel(&binary, &handle);
    if (ret != RT_ERROR_NONE || handle == nullptr) {
        ret = rtBinaryLoadWithoutTilingKey(kernelBin.data(), kernelBin.size(), &handle);
        if (ret != RT_ERROR_NONE || handle == nullptr) {
            std::fprintf(
                stderr, "register A5 AIC atomic probe kernel failed, path=%s, size=%zu, ret=%d\n", path,
                kernelBin.size(), ret);
            std::abort();
        }
    }

    void* args[] = {out};
    rtArgsEx_t argsInfo{};
    argsInfo.args = args;
    argsInfo.argsSize = sizeof(args);
    rtTaskCfgInfo_t cfgInfo{};
    ret = rtKernelLaunchWithHandleV2(handle, tilingKey, kProbeBlockCount, &argsInfo, nullptr, stream, &cfgInfo);
    if (ret != RT_ERROR_NONE) {
        std::fprintf(stderr, "rtKernelLaunchWithHandleV2 failed for A5 AIC atomic probe, ret=%d\n", ret);
        std::abort();
    }
}
} // namespace

void LaunchAicProbeStore(int32_t* out, void* stream)
{
    LaunchAicProbe(reinterpret_cast<const void*>(&LaunchAicProbeStore), kProbeStoreKey, out, stream);
}

void LaunchAicProbeStAtomic(int32_t* out, void* stream)
{
    LaunchAicProbe(reinterpret_cast<const void*>(&LaunchAicProbeStAtomic), kProbeStAtomicKey, out, stream);
}

void LaunchAicProbeLdDev(int32_t* out, void* stream)
{
    LaunchAicProbe(reinterpret_cast<const void*>(&LaunchAicProbeLdDev), kProbeLdDevKey, out, stream);
}
#endif
