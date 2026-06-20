/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <pto/pto-inst.hpp>
#include "acl/acl.h"

#if defined(SYNCALL_MIX_BUILD_AIC) && !defined(SYNCALL_MIX_REGISTER_BUILD)
#include "runtime/rt.h"

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <dlfcn.h>
#include <fstream>
#include <vector>
#endif

using namespace pto;

constexpr int32_t kAicHardBlockCount = 18;
constexpr uint64_t kAicHardTilingKey = 3001;
constexpr int32_t kInt32PerCacheLine = 8;
constexpr int32_t kVcAtomicVecParticipants = 14;
constexpr uint64_t kVcAtomicTilingKey = 3002;
constexpr uint64_t kVcAtomicProxyUbAddr = 0x3000;
constexpr uint64_t kVcAtomicProxyL1Addr = 0x0;
constexpr uint16_t kVcAtomicProxyReqId = 7;
constexpr uint16_t kVcAtomicProxyDoneId = 8;

#if defined(SYNCALL_MIX_BUILD_AIC)
PTO_SYNCALL_MIX_AIC_KERNEL_META(RunHardSyncAllAIC_3001_mix_aic, 1, 0);
PTO_SYNCALL_MIX_AIC_KERNEL_META(RunVcSoftAtomicCubeWaitConsumer_3002_mix_aic, 1, 1);

extern "C" __global__ AICORE void RunHardSyncAllAIC_3001_mix_aic(__gm__ int32_t __out__ *out)
{
    (void)out;
    SYNCALL<SyncCoreType::AICOnly>();
}

PTO_INTERNAL int32_t LoadInt32Line(__gm__ int32_t *src)
{
    dcci(static_cast<__gm__ void *>(src), SINGLE_CACHE_LINE);
    dsb(DSB_DDR);
    return src[0];
}

PTO_INTERNAL void AicRequestProxyWrite(int32_t value)
{
    __cbuf__ int32_t *l1 = reinterpret_cast<__cbuf__ int32_t *>(kVcAtomicProxyL1Addr);
    __ubuf__ int32_t *ub = reinterpret_cast<__ubuf__ int32_t *>(kVcAtomicProxyUbAddr);
    constexpr int64_t repeatConfig = (static_cast<int64_t>(1) << 16) | 1;
    create_cbuf_matrix(l1, repeatConfig, static_cast<uint32_t>(value));
    pipe_barrier(PIPE_ALL);
    copy_cbuf_to_ubuf(static_cast<__ubuf__ void *>(ub), static_cast<__cbuf__ void *>(l1), 0, 1, 1, 0, 0);
    pipe_barrier(PIPE_ALL);
    set_intra_block(PIPE_S, kVcAtomicProxyReqId);
    wait_intra_block(PIPE_S, kVcAtomicProxyDoneId);
}

PTO_INTERNAL int32_t WaitInt32LineAtLeast(__gm__ int32_t *src, int32_t expect)
{
    int32_t value = LoadInt32Line(src);
    int32_t poll = 0;
    while (value < expect) {
        ++poll;
        if (poll > SYNCALL_SOFT_BACKOFF_THRESHOLD) {
            __asm__ __volatile__("");
        }
        if (poll > SYNCALL_SOFT_MAX_POLL_ITERATIONS) {
            PTO_CPU_ASSERT(false, "A5 VC atomic soft sync cube wait timeout");
            break;
        }
        value = LoadInt32Line(src);
    }
    return value;
}

PTO_INTERNAL int32_t CheckVcSoftAtomicPayload(__gm__ int32_t *payload, __gm__ int32_t *syncState, int32_t seq)
{
    __gm__ int32_t *counter = syncState;
    __gm__ int32_t *doorbell = syncState + kInt32PerCacheLine;
    const int32_t targetCount = seq * kVcAtomicVecParticipants;

    int32_t ok = WaitInt32LineAtLeast(doorbell, seq) >= seq;
    ok &= LoadInt32Line(counter) >= targetCount;
    for (int32_t i = 0; i < kVcAtomicVecParticipants; ++i) {
        ok &= LoadInt32Line(payload + i * kInt32PerCacheLine) == seq * 100 + i;
    }
    return ok;
}

extern "C" __global__ AICORE void RunVcSoftAtomicCubeWaitConsumer_3002_mix_aic(__gm__ int32_t __out__ *out,
                                                                               __gm__ int32_t __in__ *payload,
                                                                               __gm__ int32_t __in__ *syncState,
                                                                               int32_t seq)
{
    const int32_t ok = CheckVcSoftAtomicPayload(payload, syncState, seq);
    (void)out;
    AicRequestProxyWrite(ok);
}
#endif

#if defined(SYNCALL_MIX_BUILD_AIV)
PTO_SYNCALL_MIX_AIC_KERNEL_META(RunHardSyncAllAIC_3001_mix_aiv, 1, 0);
PTO_SYNCALL_MIX_AIC_KERNEL_META(RunVcSoftAtomicCubeWaitConsumer_3002_mix_aiv, 1, 1);

extern "C" __global__ AICORE void RunHardSyncAllAIC_3001_mix_aiv(__gm__ int32_t __out__ *out)
{
    (void)out;
}

PTO_INTERNAL void AivServeProxyWrite(__gm__ int32_t *dst)
{
    __ubuf__ int32_t *ub = reinterpret_cast<__ubuf__ int32_t *>(kVcAtomicProxyUbAddr);
    wait_intra_block(PIPE_S, kVcAtomicProxyReqId);
    pipe_barrier(PIPE_ALL);
    copy_ubuf_to_gm_align_v2(static_cast<__gm__ void *>(dst), static_cast<__ubuf__ void *>(ub), 0, 1, 1, 0, 0, 0);
    pipe_barrier(PIPE_ALL);
    dcci(static_cast<__gm__ void *>(dst), SINGLE_CACHE_LINE);
    dsb(DSB_DDR);
    set_intra_block(PIPE_MTE3, kVcAtomicProxyDoneId);
}

extern "C" __global__ AICORE void RunVcSoftAtomicCubeWaitConsumer_3002_mix_aiv(__gm__ int32_t __out__ *out,
                                                                               __gm__ int32_t __in__ *payload,
                                                                               __gm__ int32_t __in__ *syncState,
                                                                               int32_t seq)
{
    (void)payload;
    (void)syncState;
    (void)seq;
    if (get_subblockid() == 0) {
        AivServeProxyWrite(out + static_cast<int32_t>(get_block_idx()) * kInt32PerCacheLine);
    }
}
#endif

#if defined(SYNCALL_MIX_BUILD_AIC) && !defined(SYNCALL_MIX_REGISTER_BUILD)
namespace {
const char *GetRegisterElfPath(const void *anchor)
{
#if defined(SYNCALL_MIX_REGISTER_OBJECT_PATH)
    (void)anchor;
    return SYNCALL_MIX_REGISTER_OBJECT_PATH;
#else
    Dl_info info{};
    if (dladdr(anchor, &info) == 0 || info.dli_fname == nullptr) {
        std::fprintf(stderr, "dladdr failed for A5 SYNCALL AIC-only hard kernel\n");
        std::abort();
    }
    return info.dli_fname;
#endif
}

std::vector<char> ReadBinaryFile(const char *path)
{
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::fprintf(stderr, "failed to open A5 AIC-only hard register ELF: %s\n", path);
        std::abort();
    }
    const std::streamsize size = file.tellg();
    if (size <= 0) {
        std::fprintf(stderr, "invalid A5 AIC-only hard register ELF size: %s\n", path);
        std::abort();
    }
    std::vector<char> data(static_cast<size_t>(size));
    file.seekg(0, std::ios::beg);
    if (!file.read(data.data(), size)) {
        std::fprintf(stderr, "failed to read A5 AIC-only hard register ELF: %s\n", path);
        std::abort();
    }
    return data;
}
} // namespace

void LaunchHardSyncAllAIC(int32_t *out, void *stream)
{
    const char *path = GetRegisterElfPath(reinterpret_cast<const void *>(&LaunchHardSyncAllAIC));
    static const std::vector<char> kernelBin = ReadBinaryFile(path);

    rtDevBinary_t binary{RT_DEV_BINARY_MAGIC_ELF, 0, kernelBin.data(), kernelBin.size()};
    void *handle = nullptr;
    rtError_t ret = rtRegisterAllKernel(&binary, &handle);
    if (ret != RT_ERROR_NONE || handle == nullptr) {
        ret = rtBinaryLoadWithoutTilingKey(kernelBin.data(), kernelBin.size(), &handle);
        if (ret != RT_ERROR_NONE || handle == nullptr) {
            std::fprintf(stderr, "register A5 AIC-only hard kernel failed, path=%s, size=%zu, ret=%d\n", path,
                         kernelBin.size(), ret);
            std::abort();
        }
    }

    void *args[] = {out};
    rtArgsEx_t argsInfo{};
    argsInfo.args = args;
    argsInfo.argsSize = sizeof(args);
    rtTaskCfgInfo_t cfgInfo{};
    ret =
        rtKernelLaunchWithHandleV2(handle, kAicHardTilingKey, kAicHardBlockCount, &argsInfo, nullptr, stream, &cfgInfo);
    if (ret != RT_ERROR_NONE) {
        std::fprintf(stderr, "rtKernelLaunchWithHandleV2 failed for A5 AIC-only hard, ret=%d\n", ret);
        std::abort();
    }
}

void LaunchVcSoftAtomicCubeWaitConsumer(int32_t *out, int32_t *payload, int32_t *syncState, int32_t aicBlocks,
                                        int32_t vecParticipants, int32_t seq, void *stream)
{
    if (vecParticipants != kVcAtomicVecParticipants || seq <= 0 || aicBlocks <= 0) {
        std::fprintf(stderr,
                     "A5 VC atomic soft sync spike expects vecParticipants=%d and seq > 0, got %d/%d, aicBlocks=%d\n",
                     kVcAtomicVecParticipants, vecParticipants, seq, aicBlocks);
        std::abort();
    }

    const char *path = GetRegisterElfPath(reinterpret_cast<const void *>(&LaunchVcSoftAtomicCubeWaitConsumer));
    static const std::vector<char> kernelBin = ReadBinaryFile(path);

    rtDevBinary_t binary{RT_DEV_BINARY_MAGIC_ELF, 0, kernelBin.data(), kernelBin.size()};
    void *handle = nullptr;
    rtError_t ret = rtRegisterAllKernel(&binary, &handle);
    if (ret != RT_ERROR_NONE || handle == nullptr) {
        ret = rtBinaryLoadWithoutTilingKey(kernelBin.data(), kernelBin.size(), &handle);
        if (ret != RT_ERROR_NONE || handle == nullptr) {
            std::fprintf(stderr, "register A5 VC atomic soft sync cube kernel failed, path=%s, size=%zu, ret=%d\n",
                         path, kernelBin.size(), ret);
            std::abort();
        }
    }

    void *args[] = {out, payload, syncState, reinterpret_cast<void *>(static_cast<uintptr_t>(seq))};
    rtArgsEx_t argsInfo{};
    argsInfo.args = args;
    argsInfo.argsSize = sizeof(args);
    rtTaskCfgInfo_t cfgInfo{};
    ret = rtKernelLaunchWithHandleV2(handle, kVcAtomicTilingKey, aicBlocks, &argsInfo, nullptr, stream, &cfgInfo);
    if (ret != RT_ERROR_NONE) {
        std::fprintf(stderr, "rtKernelLaunchWithHandleV2 failed for A5 VC atomic soft sync cube, ret=%d\n", ret);
        std::abort();
    }
}
#endif
