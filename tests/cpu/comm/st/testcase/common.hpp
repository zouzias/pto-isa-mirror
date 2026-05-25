/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#pragma once

#include <cstdint>
#include <cstdlib>
#include <cstring>

#include "pto/common/cpu_stub.hpp"

using namespace pto::comm;

template <typename T>
inline T *HcclRemotePtr(HcclDeviceContext *ctx, T *localPtr, int pe)
{
    if (ctx->rankId == pe) {
        return localPtr;
    }
    int memberSize = ctx->winSize / sizeof(T);
    T *buffer = new T[memberSize];
    return buffer;
}

inline void *WindowAlloc(uint64_t windowBase, size_t &offset, size_t bytes)
{
    void *ptr = std::malloc(bytes);
    return ptr;
}

template <typename T, size_t count>
struct TestContext {
    int32_t deviceId{-1};
    aclrtStream stream{nullptr};
    int aclStatus{0};
    HcclDeviceContext *deviceCtx{nullptr};
    HcclDeviceContext hostCtx{};

    bool Init(int rankId, int nRanks, int nDevices, int firstDeviceId, const HcclRootInfo *rootInfo)
    {
        if (nDevices <= 0 || nRanks <= 0) {
            std::cerr << "[ERROR] n_devices and n_ranks must be > 0\n";
            return false;
        }

        size_t bytesPerRank = count * sizeof(T);
        hostCtx.rankId = rankId;
        hostCtx.rankNum = nRanks;
        hostCtx.winSize = bytesPerRank;
        void *base = std::malloc(nRanks * bytesPerRank);

        for (uint32_t i = 0; i < HCCL_MAX_RANK_NUM; ++i) {
            if (i < static_cast<uint32_t>(nRanks)) {
                uint64_t baseAddress = reinterpret_cast<uintptr_t>(base);
                hostCtx.windowsIn[i] = baseAddress + i * bytesPerRank;
            } else {
                hostCtx.windowsIn[i] = 0;
            }
        }

        deviceCtx = &hostCtx;

        this->deviceId = rankId % nDevices + firstDeviceId;

        return true;
    }

    bool Finalize()
    {
        return true;
    }
};

template <typename Func>
inline bool ForkAndRunWithHcclRootInfo(int nRanks, int firstRankId, int firstDeviceId, Func &&perRankFn)
{
    if (nRanks <= 0) {
        return false;
    }

    HcclRootInfo rootInfo{};
    bool res = false;
    for (size_t i = 0; i < nRanks; i++) {
        res = res || perRankFn(i, &rootInfo);
    }

    return res;
}

/*-------------------URMA-------------------*/
struct UrmaInfo {
    uint32_t qpNum;
    uint32_t localTokenId;
    uint32_t rankCount;
    uint64_t sqPtr;
    uint64_t rqPtr;
    uint64_t scqPtr;
    uint64_t rcqPtr;
    uint64_t memPtr;
};

struct UrmaMemInfo {
    bool tokenValueValid;
    uint32_t rmtJettyType : 2;
    uint8_t targetHint;
    uint32_t tpn;
    uint32_t tid;
    uint32_t rmtTokenValue;
    uint32_t len;
    uint64_t addr;
    uint64_t eidAddr;
};

AICORE inline uint64_t UrmaPeerMrBaseAddr(__gm__ uint8_t *urmaWorkspace, uint32_t peerRank)
{
    __gm__ UrmaInfo *info = (__gm__ UrmaInfo *)urmaWorkspace;
    PTO_ASSERT(peerRank < info->rankCount, "UrmaPeerMrBaseAddr: peerRank out of range");
    __gm__ UrmaMemInfo *memRow = reinterpret_cast<__gm__ UrmaMemInfo *>(info->memPtr) + peerRank;
    return memRow->addr;
}

AICORE inline bool BuildUrmaSession(__gm__ uint8_t *contextGm, uint32_t destRankId, urma::UrmaSession &session)
{
    session.execCtx.contextGm = contextGm;
    session.execCtx.destRankId = destRankId;
    session.execCtx.qpIdx = 0;
    session.eventCtx.contextGm = contextGm;
    session.valid = (contextGm != nullptr);
    return session.valid;
}

template <DmaEngine engine>
PTO_INTERNAL bool BuildAsyncSession(__gm__ uint8_t *workspace, uint32_t destRankId, AsyncSession &session)
{
    static_assert(engine == DmaEngine::URMA, "This overload is for URMA only");
    session.engine = engine;
    session.valid = BuildUrmaSession(workspace, destRankId, session.urmaSession);
    return session.valid;
}