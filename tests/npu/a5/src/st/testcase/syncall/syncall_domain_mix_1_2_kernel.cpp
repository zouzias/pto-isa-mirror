/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include "syncall_mix_common.hpp"

#if defined(SYNCALL_MIX_BUILD_AIC) && !defined(SYNCALL_MIX_REGISTER_BUILD)
#include "runtime/rt.h"

#include <cstdlib>
#include <cstdio>
#include <dlfcn.h>
#include <fstream>
#include <vector>
#endif

constexpr int32_t kDomainAicBlocks = 18;
constexpr int32_t kDomainVecParticipants = 36;
constexpr int32_t kDomainTotalParticipants = kDomainAicBlocks + kDomainVecParticipants;
constexpr int32_t kA5VecPhyStart = 18;
constexpr int32_t kA5WorldCores = 54;
constexpr int32_t kDomainOffset = 512;
constexpr int32_t kDomainScratchInt32 = 512;
constexpr uint64_t kDomainTilingKey = 1301;

PTO_INTERNAL bool IsCurrentCoreInRange(const SoftSyncCoreRange &range)
{
    const int32_t phyCore = static_cast<int32_t>(get_coreid());
    const int32_t local = (phyCore - range.startPhyCore + range.worldCoreCount) % range.worldCoreCount;
    return local >= 0 && local < range.coreCount;
}

PTO_INTERNAL void StoreDomainStatus(__gm__ int32_t *out, int32_t idx, int32_t status)
{
#if defined(__DAV_CUBE__)
    AicRequestProxyWrite(status);
#elif defined(__DAV_VEC__)
    if (get_subblockid() == 0) {
        AivServeProxyWrite(out + static_cast<int32_t>(get_block_idx()) * kInt32PerCacheLine);
    }
    AivWriteGm(out + idx * kInt32PerCacheLine, status, kMixOutUbAddr);
#else
    (void)out;
    (void)idx;
    (void)status;
#endif
}

#if defined(SYNCALL_MIX_BUILD_AIC)
PTO_SYNCALL_MIX_AIC_KERNEL_META(RunSoftSyncDomainMix12_1301_mix_aic, 1, 2);

extern "C" __global__ AICORE void RunSoftSyncDomainMix12_1301_mix_aic(__gm__ int32_t __out__ *out,
                                                                      __gm__ int32_t __out__ *syncWorkspace)
{
    const int32_t idx = GetMixLogicalIdx();
    int32_t status = 0;

    GlobalTensor<int32_t, pto::Shape<>, pto::Stride<>> gmWs(syncWorkspace);
    Tile<TileType::Vec, int32_t, 1, kDomainScratchInt32> syncUbTile;
    Tile<TileType::Mat, int32_t, 1, SYNCALL_SOFT_SLOT_INT32> syncL1Tile;
#ifndef __PTO_AUTO__
    syncUbTile.data() = reinterpret_cast<__ubuf__ int32_t *>(kProxyUbAddr);
    syncL1Tile.data() = reinterpret_cast<__cbuf__ int32_t *>(kProxyL1Addr);
#endif

    const SoftSyncDomainDesc fullDomain{
        SoftSyncCoreRange{kA5VecPhyStart, kDomainVecParticipants, kA5WorldCores},
        SoftSyncCoreRange{0, kDomainAicBlocks, kA5WorldCores},
        0,
    };
    SoftSyncLocalState fullState = SOFT_SYNC_INIT(gmWs, fullDomain);

    SOFT_SYNC_WAIT<SoftSyncDirection::VToC>(gmWs, fullDomain, fullState, syncL1Tile);
    SOFT_SYNC_WAIT<SoftSyncDirection::VToC>(gmWs, fullDomain, fullState, syncL1Tile);
    status |= 1;

    SOFT_SYNC_NOTIFY<SoftSyncDirection::CToV>(gmWs, fullDomain, fullState, syncL1Tile);
    SOFT_SYNC_NOTIFY<SoftSyncDirection::CToV>(gmWs, fullDomain, fullState, syncL1Tile);

    const SoftSyncDomainDesc splitDomain{
        SoftSyncCoreRange{kA5VecPhyStart, 3, kA5WorldCores},
        SoftSyncCoreRange{0, 3, kA5WorldCores},
        kDomainOffset,
    };
    SoftSyncLocalState splitState = SOFT_SYNC_INIT(gmWs, splitDomain);
    if (IsCurrentCoreInRange(splitDomain.cubeGroup)) {
        SOFT_SYNC_WAIT<SoftSyncDirection::VToC>(gmWs, splitDomain, splitState, syncL1Tile);
        status |= 4;
    }

    const SoftSyncDomainDesc wrapDomain{
        SoftSyncCoreRange{kA5VecPhyStart, kDomainVecParticipants, kA5WorldCores},
        SoftSyncCoreRange{kDomainAicBlocks - 1, 3, kDomainAicBlocks},
        kDomainOffset + 256,
    };
    SoftSyncLocalState wrapState = SOFT_SYNC_INIT(gmWs, wrapDomain);
    if (IsCurrentCoreInRange(wrapDomain.cubeGroup)) {
        SOFT_SYNC_NOTIFY<SoftSyncDirection::CToV>(gmWs, wrapDomain, wrapState, syncL1Tile);
    }

    StoreDomainStatus(out, idx, status);
}
#endif

#if defined(SYNCALL_MIX_BUILD_AIV)
PTO_SYNCALL_MIX_AIC_KERNEL_META(RunSoftSyncDomainMix12_1301_mix_aiv, 1, 2);

extern "C" __global__ AICORE void RunSoftSyncDomainMix12_1301_mix_aiv(__gm__ int32_t __out__ *out,
                                                                      __gm__ int32_t __out__ *syncWorkspace)
{
    const int32_t idx = GetMixLogicalIdx();
    int32_t status = 0;

    GlobalTensor<int32_t, pto::Shape<>, pto::Stride<>> gmWs(syncWorkspace);
    Tile<TileType::Vec, int32_t, 1, kDomainScratchInt32> syncUbTile;
    Tile<TileType::Mat, int32_t, 1, SYNCALL_SOFT_SLOT_INT32> syncL1Tile;
#ifndef __PTO_AUTO__
    syncUbTile.data() = reinterpret_cast<__ubuf__ int32_t *>(kProxyUbAddr);
    syncL1Tile.data() = reinterpret_cast<__cbuf__ int32_t *>(kProxyL1Addr);
#endif

    const SoftSyncDomainDesc fullDomain{
        SoftSyncCoreRange{kA5VecPhyStart, kDomainVecParticipants, kA5WorldCores},
        SoftSyncCoreRange{0, kDomainAicBlocks, kA5WorldCores},
        0,
    };
    SoftSyncLocalState fullState = SOFT_SYNC_INIT(gmWs, fullDomain);

    SOFT_SYNC_NOTIFY<SoftSyncDirection::VToC>(gmWs, fullDomain, fullState, syncUbTile);
    SOFT_SYNC_NOTIFY<SoftSyncDirection::VToC>(gmWs, fullDomain, fullState, syncUbTile);

    SOFT_SYNC_WAIT<SoftSyncDirection::CToV>(gmWs, fullDomain, fullState, syncUbTile);
    SOFT_SYNC_WAIT<SoftSyncDirection::CToV>(gmWs, fullDomain, fullState, syncUbTile);
    status |= 2;

    const SoftSyncDomainDesc splitDomain{
        SoftSyncCoreRange{kA5VecPhyStart, 3, kA5WorldCores},
        SoftSyncCoreRange{0, 3, kA5WorldCores},
        kDomainOffset,
    };
    SoftSyncLocalState splitState = SOFT_SYNC_INIT(gmWs, splitDomain);
    if (IsCurrentCoreInRange(splitDomain.vecGroup)) {
        SOFT_SYNC_NOTIFY<SoftSyncDirection::VToC>(gmWs, splitDomain, splitState, syncUbTile);
    }

    const SoftSyncDomainDesc wrapDomain{
        SoftSyncCoreRange{kA5VecPhyStart, kDomainVecParticipants, kA5WorldCores},
        SoftSyncCoreRange{kDomainAicBlocks - 1, 3, kDomainAicBlocks},
        kDomainOffset + 256,
    };
    SoftSyncLocalState wrapState = SOFT_SYNC_INIT(gmWs, wrapDomain);
    SOFT_SYNC_WAIT<SoftSyncDirection::CToV>(gmWs, wrapDomain, wrapState, syncUbTile);
    status |= 8;

    StoreDomainStatus(out, idx, status);
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
        std::fprintf(stderr, "dladdr failed for soft sync domain mix kernel\n");
        std::abort();
    }
    return info.dli_fname;
#endif
}

std::vector<char> ReadCurrentSharedObject(const char *path)
{
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::fprintf(stderr, "failed to open soft sync domain kernel binary: %s\n", path);
        std::abort();
    }
    const std::streamsize size = file.tellg();
    if (size <= 0) {
        std::fprintf(stderr, "invalid soft sync domain kernel binary size: %s\n", path);
        std::abort();
    }
    std::vector<char> data(static_cast<size_t>(size));
    file.seekg(0, std::ios::beg);
    if (!file.read(data.data(), size)) {
        std::fprintf(stderr, "failed to read soft sync domain kernel binary: %s\n", path);
        std::abort();
    }
    return data;
}

void LaunchSoftDomainKernel(const void *anchor, int32_t *out, int32_t *syncWorkspace, void *stream)
{
    const char *path = GetCurrentSharedObjectPath(anchor);
    static const std::vector<char> kernelBinary = ReadCurrentSharedObject(path);
    rtDevBinary_t binary{RT_DEV_BINARY_MAGIC_ELF, 0, kernelBinary.data(), kernelBinary.size()};
    void *handle = nullptr;
    rtError_t ret = rtRegisterAllKernel(&binary, &handle);
    if (ret != RT_ERROR_NONE || handle == nullptr) {
        ret = rtBinaryLoadWithoutTilingKey(kernelBinary.data(), kernelBinary.size(), &handle);
        if (ret != RT_ERROR_NONE || handle == nullptr) {
            std::fprintf(stderr, "register soft sync domain kernel failed, path=%s, size=%zu, ret=%d\n", path,
                         kernelBinary.size(), ret);
            std::abort();
        }
    }

    void *args[] = {out, syncWorkspace};
    rtArgsEx_t argsInfo{};
    argsInfo.args = args;
    argsInfo.argsSize = sizeof(args);
    rtTaskCfgInfo_t cfgInfo{};
    ret = rtKernelLaunchWithHandleV2(handle, kDomainTilingKey, kDomainAicBlocks, &argsInfo, nullptr, stream, &cfgInfo);
    if (ret != RT_ERROR_NONE) {
        std::fprintf(stderr, "rtKernelLaunchWithHandleV2 failed for soft sync domain mix, ret=%d\n", ret);
        std::abort();
    }
}
} // namespace

void LaunchSoftSyncDomainMix12(int32_t *out, int32_t *syncWorkspace, void *stream)
{
    LaunchSoftDomainKernel(reinterpret_cast<const void *>(&LaunchSoftSyncDomainMix12), out, syncWorkspace, stream);
}
#endif
