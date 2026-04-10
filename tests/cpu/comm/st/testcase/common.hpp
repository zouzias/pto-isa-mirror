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
#include <iostream>
#include <vector>
#include <dlfcn.h>

#include "hccl_context.h"
#include "comm_mpi.h"
#include "pto/common/cpu_stub.hpp"

// ============================================================================
// Debug logging helpers.  Enabled by cmake -DDEBUG_MODE=ON  (defines COMM_DEBUG).
// Uses COMM_DEBUG instead of _DEBUG to avoid activating PTO's PTO_ASSERT which
// calls cce::printf (unsupported on A5).
// ============================================================================
#ifdef COMM_DEBUG
#include <chrono>
#include <iomanip>
static inline double DbgNowMs()
{
    using clk = std::chrono::steady_clock;
    static const auto g_start = clk::now();
    return std::chrono::duration<double, std::milli>(clk::now() - g_start).count();
}
#define COMM_DBG(fmt, ...)                                                                                      \
    do {                                                                                                        \
        std::cerr << "[DBG " << std::fixed << std::setprecision(1) << DbgNowMs() << "ms] " << fmt << std::endl; \
    } while (0)
#define COMM_LOG(x)                  \
    do {                             \
        std::cerr << x << std::endl; \
    } while (0)
#else
#define COMM_DBG(fmt, ...) ((void)0)
#define COMM_LOG(x) ((void)0)
#endif

typedef void *HcclComm;

void HcclHostBarrier(HcclComm, aclrtStream){

};

// Runtime APIs — lower-level device/stream management (from libruntime.so).
// PyPTO uses rtSetDevice on rank 0 and rtStreamCreate for streams.
using rtError_t = int32_t;
using rtStream_t = void *;
static constexpr int32_t RT_STREAM_PRIORITY_DEFAULT = 0;
extern "C" rtError_t rtSetDevice(int32_t device);
extern "C" rtError_t rtStreamCreate(rtStream_t *stream, int32_t priority);
extern "C" rtError_t rtStreamDestroy(rtStream_t stream);

// Internal HCCL APIs — declared here instead of including hcom.h because
// hcom.h uses internal types (s32 etc.) unavailable under bisheng -xcce.

using CommTopo = uint32_t;
static constexpr uint32_t COMM_IS_NOT_SET_DEVICE = 0;
static constexpr uint32_t COMM_TOPO_MESH = 0b1u;

// ============================================================================
// V2 tiling structures (same as A5).
// init.version=100 routes through HCCL's V2 code path.
// On MESH: returns HcclCombinOpParamA5 (windowsIn[64] directly usable).
// On RING:  returns HcclOpResParam (requires remoteRes extraction).
// ============================================================================

static constexpr uint32_t MAX_CC_TILING_NUM = 8U;
static constexpr uint32_t GROUP_NAME_SIZE = 128U;
static constexpr uint32_t ALG_CONFIG_SIZE = 128U;

struct Mc2InitTilingInner {
    uint32_t version;
    uint32_t mc2HcommCnt;
    uint32_t offset[MAX_CC_TILING_NUM];
    uint8_t debugMode;
    uint8_t preparePosition;
    uint16_t queueNum;
    uint16_t commBlockNum;
    uint8_t devType;
    char reserved[17];
};

struct Mc2cCTilingInner {
    uint8_t skipLocalRankCopy;
    uint8_t skipBufferWindowCopy;
    uint8_t stepSize;
    uint8_t version;
    char reserved[9];
    uint8_t commEngine;
    uint8_t srcDataType;
    uint8_t dstDataType;
    char groupName[GROUP_NAME_SIZE];
    char algConfig[ALG_CONFIG_SIZE];
    uint32_t opType;
    uint32_t reduceType;
};

// ============================================================================
// HcclOpResParam compat structs — binary-compatible copies of HCCL internal
// types (from PyPTO hccl_context.h).  Used only on host side to compute
// offsetof(HcclOpResParam, remoteRes) for RING topology.
// ============================================================================
struct HcclRootInfo {};

namespace hccl_compat {

static constexpr uint32_t COMPAT_LOCAL_NOTIFY_MAX_NUM = 64;
static constexpr uint32_t COMPAT_LOCAL_STREAM_MAX_NUM = 19;
static constexpr uint32_t COMPAT_AICPU_OP_NOTIFY_MAX_NUM = 2;

} // namespace hccl_compat

// ============================================================================
// Device-side helper: convert a local window pointer to the equivalent address
// on a remote rank.
// ============================================================================
template <typename T>
AICORE inline __gm__ T *HcclRemotePtr(__gm__ HcclDeviceContext *ctx, __gm__ T *localPtr, int pe)
{
    int memberSize = ctx->winSize / sizeof(T);
    if (ctx->rankId == pe) {
        return localPtr;
    }
    T *buffer = new T[memberSize];
    return buffer;
}

inline void *WindowAlloc(uint64_t windowBase, size_t &offset, size_t bytes)
{
    void *ptr = std::malloc(bytes);
    return ptr;
}

// ============================================================================
// TestContext: ACL + HCCL initialization / teardown helper.
// ============================================================================
template <typename T, size_t count>
struct TestContext {
    int32_t deviceId{-1};
    aclrtStream stream{nullptr};
    int aclStatus{0};
    HcclComm comm{nullptr};
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

// ============================================================================
// ForkAndRunWithHcclRootInfo: MPI-based multi-rank test execution.
//
// Requires the binary to be launched via: mpirun -n <nRanks> ./test_binary
// Each MPI process runs the perRankFn for its assigned rank.
// Rank 0 generates HcclRootInfo and broadcasts it to all ranks via MPI_Bcast.
// MPI_Barrier ensures all ranks are synchronized before HCCL operations.
// ============================================================================
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
