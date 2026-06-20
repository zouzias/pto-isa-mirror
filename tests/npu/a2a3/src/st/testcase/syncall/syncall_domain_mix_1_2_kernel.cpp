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

constexpr int32_t kA2A3VecPhyStart = 48;
constexpr int32_t kA2A3WorldCores = 128;
constexpr int32_t kDomainOffset = 512;
constexpr int32_t kDomainScratchInt32 = 512;

PTO_INTERNAL bool IsCurrentCoreInRange(const SoftSyncCoreRange &range)
{
    const int32_t phyCore = static_cast<int32_t>(get_coreid());
    const int32_t local = (phyCore - range.startPhyCore + range.worldCoreCount) % range.worldCoreCount;
    return local >= 0 && local < range.coreCount;
}

extern "C" __global__ AICORE void RunSoftSyncDomainMix12(__gm__ int32_t __out__ *out,
                                                         __gm__ int32_t __out__ *syncWorkspace, int32_t aicBlocks,
                                                         int32_t totalParticipants)
{
    const int32_t idx = GetMixLogicalIdx(aicBlocks);
    int32_t status = 0;

    GlobalTensor<int32_t, pto::Shape<>, pto::Stride<>> gmWs(syncWorkspace);
    Tile<TileType::Vec, int32_t, 1, kDomainScratchInt32> syncUbTile;
    Tile<TileType::Mat, int32_t, 1, SYNCALL_SOFT_SLOT_INT32> syncL1Tile;
#ifndef __PTO_AUTO__
    syncUbTile.data() = reinterpret_cast<__ubuf__ int32_t *>(kMixSoftUbAddr);
    syncL1Tile.data() = reinterpret_cast<__cbuf__ int32_t *>(kMixSoftL1Addr);
#endif

    const SoftSyncDomainDesc fullDomain{
        SoftSyncCoreRange{kA2A3VecPhyStart, aicBlocks * 2, kA2A3WorldCores},
        SoftSyncCoreRange{0, aicBlocks, kA2A3WorldCores},
        0,
    };
    SoftSyncLocalState fullState = SOFT_SYNC_INIT(gmWs, fullDomain);

#if defined(__DAV_VEC__)
    SOFT_SYNC_NOTIFY<SoftSyncDirection::VToC>(gmWs, fullDomain, fullState, syncUbTile);
    SOFT_SYNC_NOTIFY<SoftSyncDirection::VToC>(gmWs, fullDomain, fullState, syncUbTile);
#elif defined(__DAV_CUBE__)
    SOFT_SYNC_WAIT<SoftSyncDirection::VToC>(gmWs, fullDomain, fullState, syncL1Tile);
    SOFT_SYNC_WAIT<SoftSyncDirection::VToC>(gmWs, fullDomain, fullState, syncL1Tile);
    status |= 1;
#endif

#if defined(__DAV_CUBE__)
    SOFT_SYNC_NOTIFY<SoftSyncDirection::CToV>(gmWs, fullDomain, fullState, syncL1Tile);
    SOFT_SYNC_NOTIFY<SoftSyncDirection::CToV>(gmWs, fullDomain, fullState, syncL1Tile);
#elif defined(__DAV_VEC__)
    SOFT_SYNC_WAIT<SoftSyncDirection::CToV>(gmWs, fullDomain, fullState, syncUbTile);
    SOFT_SYNC_WAIT<SoftSyncDirection::CToV>(gmWs, fullDomain, fullState, syncUbTile);
    status |= 2;
#endif

    const SoftSyncDomainDesc splitDomain{
        SoftSyncCoreRange{kA2A3VecPhyStart, 3, kA2A3WorldCores},
        SoftSyncCoreRange{0, 3, kA2A3WorldCores},
        kDomainOffset,
    };
    SoftSyncLocalState splitState = SOFT_SYNC_INIT(gmWs, splitDomain);
#if defined(__DAV_VEC__)
    if (IsCurrentCoreInRange(splitDomain.vecGroup)) {
        SOFT_SYNC_NOTIFY<SoftSyncDirection::VToC>(gmWs, splitDomain, splitState, syncUbTile);
    }
#elif defined(__DAV_CUBE__)
    if (IsCurrentCoreInRange(splitDomain.cubeGroup)) {
        SOFT_SYNC_WAIT<SoftSyncDirection::VToC>(gmWs, splitDomain, splitState, syncL1Tile);
        status |= 4;
    }
#endif

    const SoftSyncDomainDesc wrapDomain{
        SoftSyncCoreRange{kA2A3VecPhyStart, aicBlocks * 2, kA2A3WorldCores},
        SoftSyncCoreRange{aicBlocks - 1, 3, aicBlocks},
        kDomainOffset + 256,
    };
    SoftSyncLocalState wrapState = SOFT_SYNC_INIT(gmWs, wrapDomain);
#if defined(__DAV_CUBE__)
    if (IsCurrentCoreInRange(wrapDomain.cubeGroup)) {
        SOFT_SYNC_NOTIFY<SoftSyncDirection::CToV>(gmWs, wrapDomain, wrapState, syncL1Tile);
    }
#elif defined(__DAV_VEC__)
    SOFT_SYNC_WAIT<SoftSyncDirection::CToV>(gmWs, wrapDomain, wrapState, syncUbTile);
    status |= 8;
#endif

    StoreMixInt32Line(out + idx * kInt32PerCacheLine, status, kMixOutUbAddr, kMixOutL1Addr);
}

void LaunchSoftSyncDomainMix12(int32_t *out, int32_t *syncWorkspace, int32_t aicBlocks, int32_t totalParticipants,
                               void *stream)
{
    RunSoftSyncDomainMix12<<<aicBlocks, nullptr, stream>>>(out, syncWorkspace, aicBlocks, totalParticipants);
}
