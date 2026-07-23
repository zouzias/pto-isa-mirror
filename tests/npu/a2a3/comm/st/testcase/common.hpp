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
#include <limits>
#include <numeric>
#include <vector>
#include <dlfcn.h>
#include "acl/acl.h"
#include "hccl/hccl_comm.h"
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>

#if __has_include("hccl/hccl.h")
#include "hccl/hccl.h"
#endif
#include "hccl/hccl_types.h"
#include "hccl_context.h"
#include "comm_mpi.h"
#include "pto/comm/domain/host/comm_context.hpp"

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

// Runtime APIs — lower-level device/stream management (from libruntime.so).
// PyPTO uses rtSetDevice on rank 0 and rtStreamCreate for streams.
using rtError_t = int32_t;
using rtStream_t = void*;
static constexpr int32_t RT_STREAM_PRIORITY_DEFAULT = 0;
extern "C" rtError_t rtSetDevice(int32_t device);
extern "C" rtError_t rtStreamCreate(rtStream_t* stream, int32_t priority);
extern "C" rtError_t rtStreamDestroy(rtStream_t stream);

// aclnn tensor API (from aclnn/acl_meta.h, linked via libnnopbase).
// Forward-declared here to avoid pulling in aclnn headers that may
// conflict with the bisheng -xcce compilation mode.
struct aclTensor;
struct aclOpExecutor;
extern "C" aclTensor* aclCreateTensor(
    const int64_t* viewDims, uint64_t viewDimsNum, aclDataType dataType, const int64_t* stride, int64_t offset,
    aclFormat format, const int64_t* storageDims, uint64_t storageDimsNum, void* tensorData);
extern "C" int32_t aclDestroyTensor(const aclTensor* tensor);

// Mc2 tiling structures passed to HcclAllocComResourceByTiling.
// Binary layout must match the HCCL internal expectation.
#pragma pack(push, 8)
struct Mc2ServerCfg {
    uint32_t version = 0;
    uint8_t debugMode = 0;
    uint8_t sendArgIndex = 0;
    uint8_t recvArgIndex = 0;
    uint8_t commOutArgIndex = 0;
    uint8_t reserved[8] = {};
};
#pragma pack(pop)

// ============================================================================
// V2 tiling structures (same as A5).
// init.version=100 routes through HCCL's V2 code path.
// On MESH: returns HcclCombinOpParamA5 (windowsIn[64] directly usable).
// On RING:  returns CommOpResParam (requires remoteRes extraction).
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

struct Mc2CommConfigV2 {
    Mc2InitTilingInner init;
    Mc2cCTilingInner inner;
};

// ============================================================================
// CommOpResParam compat structs — binary-compatible copies of HCCL internal
// types (from PyPTO hccl_context.h).  Used only on host side to compute
// offsetof(CommOpResParam, remoteRes) for RING topology.
// ============================================================================
namespace hccl_compat {

struct CommSignalInfo {
    uint64_t resId;
    uint64_t addr;
    uint32_t devId;
    uint32_t tsId;
    uint32_t rankId;
    uint32_t flag;
};

struct CommStreamInfo {
    int32_t streamIds;
    uint32_t sqIds;
    uint32_t cqIds;
    uint32_t logicCqids;
};

struct ListCommon {
    uint64_t nextHost;
    uint64_t preHost;
    uint64_t nextDevice;
    uint64_t preDevice;
};

static constexpr uint32_t COMPAT_LOCAL_NOTIFY_MAX_NUM = 64;
static constexpr uint32_t COMPAT_LOCAL_STREAM_MAX_NUM = 19;
static constexpr uint32_t COMPAT_AICPU_OP_NOTIFY_MAX_NUM = 2;

struct LocalResInfoV2 {
    uint32_t streamNum;
    uint32_t signalNum;
    CommSignalInfo localSignals[COMPAT_LOCAL_NOTIFY_MAX_NUM];
    CommStreamInfo streamInfo[COMPAT_LOCAL_STREAM_MAX_NUM];
    CommStreamInfo mainStreamInfo;
    CommSignalInfo aicpuOpNotify[COMPAT_AICPU_OP_NOTIFY_MAX_NUM];
    ListCommon nextTagRes;
};

struct AlgoTopoInfo {
    uint32_t userRank;
    uint32_t userRankSize;
    int32_t deviceLogicId;
    bool isSingleMeshAggregation;
    uint32_t deviceNumPerAggregation;
    uint32_t superPodNum;
    uint32_t devicePhyId;
    uint32_t topoType;
    uint32_t deviceType;
    uint32_t serverNum;
    uint32_t meshAggregationRankSize;
    uint32_t multiModuleDiffDeviceNumMode;
    uint32_t multiSuperPodDiffServerNumMode;
    uint32_t realUserRank;
    bool isDiffDeviceModule;
    bool isDiffDeviceType;
    uint32_t gcdDeviceNumPerAggregation;
    uint32_t moduleNum;
    uint32_t isUsedRdmaRankPairNum;
    uint64_t isUsedRdmaRankPair;
    uint32_t pairLinkCounterNum;
    uint64_t pairLinkCounter;
    uint32_t nicNum;
    uint64_t nicList;
    uint64_t complanRankLength;
    uint64_t complanRank;
    uint64_t bridgeRankNum;
    uint64_t bridgeRank;
    uint64_t serverAndsuperPodRankLength;
    uint64_t serverAndsuperPodRank;
};

struct CommOpConfig {
    uint8_t deterministic;
    uint8_t retryEnable;
    uint8_t highPerfEnable;
    uint8_t padding[5];
    uint8_t linkTimeOut[8];
    uint64_t notifyWaitTime;
    uint32_t retryHoldTime;
    uint32_t retryIntervalTime;
    bool interXLinkDisable;
    uint32_t floatOverflowMode;
    uint32_t multiQpThreshold;
};

struct HDCommunicateParams {
    uint64_t hostAddr;
    uint64_t deviceAddr;
    uint64_t readCacheAddr;
    uint32_t devMemSize;
    uint32_t buffLen;
    uint32_t flag;
};

struct RemoteResPtr {
    uint64_t nextHostPtr;
    uint64_t nextDevicePtr;
};

struct HcclMC2WorkSpace {
    uint64_t workspace;
    uint64_t workspaceSize;
};

struct CommRankRelationResV2 {
    uint32_t remoteUsrRankId;
    uint32_t remoteWorldRank;
    uint64_t windowsIn;
    uint64_t windowsOut;
    uint64_t windowsExp;
    ListCommon nextTagRes;
};

struct CommOpResParamHead {
    uint32_t localUsrRankId;
    uint32_t rankSize;
    uint64_t winSize;
    uint64_t localWindowsIn;
    uint64_t localWindowsOut;
    char hcomId[128];
    uint64_t winExpSize;
    uint64_t localWindowsExp;
};

// Full struct layout for offsetof(remoteRes) computation.
// Array size of remoteRes does not affect the offset calculation.
struct CommOpResParam {
    HcclMC2WorkSpace mc2WorkSpace;
    uint32_t localUsrRankId;
    uint32_t rankSize;
    uint64_t winSize;
    uint64_t localWindowsIn;
    uint64_t localWindowsOut;
    char hcomId[128];
    uint64_t winExpSize;
    uint64_t localWindowsExp;
    uint32_t rWinStart;
    uint32_t rWinOffset;
    uint64_t version;
    LocalResInfoV2 localRes;
    AlgoTopoInfo topoInfo;
    CommOpConfig config;
    uint64_t hostStateInfo;
    uint64_t aicpuStateInfo;
    uint64_t lockAddr;
    uint32_t rsv[16];
    uint32_t notifysize;
    uint32_t remoteResNum;
    RemoteResPtr remoteRes[1];
};

} // namespace hccl_compat

// ============================================================================
// Device-side helper: convert a local window pointer to the equivalent address
// on a remote rank.
// ============================================================================
template <typename T>
AICORE inline __gm__ T* CommRemotePtr(__gm__ CommDeviceContext* ctx, __gm__ T* localPtr, int pe)
{
    uint64_t localBase = ctx->windowsIn[ctx->rankId];
    uint64_t offset = (uint64_t)localPtr - localBase;
    return (__gm__ T*)(ctx->windowsIn[pe] + offset);
}

// ============================================================================
// Host-side helpers
// ============================================================================
inline void HcclHostBarrier(HcclComm comm, aclrtStream stream)
{
    COMM_DBG("  HcclHostBarrier: calling HcclBarrier ...");
    HcclResult hret = HcclBarrier(comm, stream);
    COMM_DBG("  HcclHostBarrier: HcclBarrier returned " << (int)hret << ", syncing stream ...");
    aclError aret = aclrtSynchronizeStream(stream);
    COMM_DBG("  HcclHostBarrier: stream sync done (acl=" << (int)aret << ")");
}

inline void* WindowAlloc(uint64_t windowBase, size_t& offset, size_t bytes)
{
    void* ptr = reinterpret_cast<void*>(windowBase + offset);
    offset += bytes;
    return ptr;
}

inline bool BuildTestComm(
    pto::comm::domain::CommContext& out, int rankId, int nRanks, int nDevices, int firstDeviceId,
    const HcclRootInfo* rootInfo)
{
    pto::comm::domain::CommConfig cfg{};
    cfg.rankId = rankId;
    cfg.rankNum = nRanks;
    cfg.deviceId = rankId % nDevices + firstDeviceId;
    cfg.backends = pto::comm::domain::CommBackend::MTE;
    cfg.bootstrap = pto::comm::domain::Bootstrap::Mpi;
    cfg.rootInfo = rootInfo;
    return pto::comm::domain::BuildComm(cfg, out);
}

// ============================================================================
// TestContext: ACL + HCCL initialization / teardown helper.
// ============================================================================
struct TestContext {
    int32_t deviceId{-1};
    rtStream_t stream{nullptr};
    int aclStatus{0};
    HcclComm comm{nullptr};

    CommDeviceContext* deviceCtx{nullptr};
    CommDeviceContext hostCtx{};
    pto::comm::domain::CommContext domainCtx{};

    bool Init(int rankId, int nRanks, int nDevices, int firstDeviceId, const HcclRootInfo* rootInfo)
    {
        if (nDevices <= 0 || nRanks <= 0) {
            std::cerr << "[ERROR] n_devices and n_ranks must be > 0\n";
            return false;
        }
        deviceId = rankId % nDevices + firstDeviceId;

        pto::comm::domain::CommConfig cfg{};
        cfg.rankId = rankId;
        cfg.rankNum = nRanks;
        cfg.deviceId = deviceId;
        cfg.backends = pto::comm::domain::CommBackend::MTE;
        cfg.bootstrap = pto::comm::domain::Bootstrap::Mpi;
        cfg.rootInfo = rootInfo;
        if (!pto::comm::domain::BuildComm(cfg, domainCtx)) {
            std::cerr << "[ERROR] BuildComm failed for rank " << rankId << std::endl;
            return false;
        }

        stream = domainCtx.stream;
        comm = domainCtx.comm;
        std::memcpy(&hostCtx, &domainCtx.winHostCtx, sizeof(hostCtx));
        deviceCtx =
            (CommDeviceContext*)pto::comm::domain::GetDeviceContext(domainCtx, pto::comm::domain::AddrFamily::Window);
        if (deviceCtx == nullptr ||
            pto::comm::domain::GetSymmetricBase(domainCtx, pto::comm::domain::AddrFamily::Window) == nullptr) {
            std::cerr << "[ERROR] BuildComm did not provide window device context/base for rank " << rankId
                      << std::endl;
            return false;
        }
        return true;
    }

    bool Finalize()
    {
        pto::comm::domain::DestroyComm(domainCtx);
        stream = nullptr;
        comm = nullptr;
        deviceCtx = nullptr;
        hostCtx = {};
        return (aclStatus == 0);
    }

    void* WindowAlloc(size_t& offset, size_t bytes)
    {
        void* base = pto::comm::domain::GetSymmetricBase(domainCtx, pto::comm::domain::AddrFamily::Window);
        return ::WindowAlloc(reinterpret_cast<uint64_t>(base), offset, bytes);
    }

    void HostBarrier() { pto::comm::domain::HostBarrier(domainCtx); }
};

// ============================================================================
// ForkAndRunWithHcclRootInfo: MPI-based multi-rank test execution.
//
// Requires the binary to be launched via: mpirun -n <nRanks> ./test_binary
// Each MPI process runs the perRankFn for its assigned rank.
// Rank 0 generates HcclRootInfo and broadcasts it to all ranks via MPI_Bcast.
// MPI_Barrier ensures all ranks are synchronized before HCCL operations.
// ============================================================================
// Query the number of physical NPUs available on this machine.
// Caches the result after the first successful call.
inline int GetAvailableDeviceCount()
{
    static int cachedCount = -1;
    if (cachedCount >= 0)
        return cachedCount;
    constexpr int kAclRepeatInit = 100002;
    aclError aRet = aclInit(nullptr);
    if (aRet != ACL_SUCCESS && static_cast<int>(aRet) != kAclRepeatInit) {
        return 0;
    }
    uint32_t count = 0;
    aRet = aclrtGetDeviceCount(&count);
    if (aRet != ACL_SUCCESS) {
        return 0;
    }
    cachedCount = static_cast<int>(count);
    return cachedCount;
}

// One-time ACL/device initialization guard.
// Ensures aclInit + aclrtSetDevice run only once per process, avoiding
// repeated init/finalize cycles that exhaust driver Notify resources.
// Cleanup (aclrtResetDevice / aclFinalize) is intentionally omitted:
// the OS and driver reclaim all resources when the process exits.
inline bool EnsureAclDeviceInit(int mpiRank, int deviceId)
{
    static int cachedDeviceId = -1;
    static bool initialized = false;
    if (initialized && cachedDeviceId == deviceId)
        return true;

    constexpr int kAclRepeatInit = 100002;
    aclError aRet = aclInit(nullptr);
    if (aRet != ACL_SUCCESS && static_cast<int>(aRet) != kAclRepeatInit) {
        std::cerr << "[ERROR] Rank " << mpiRank << ": aclInit failed: " << static_cast<int>(aRet) << std::endl;
        return false;
    }

    if (mpiRank == 0) {
        int32_t rtRet = rtSetDevice(deviceId);
        COMM_LOG("[INIT] Rank 0: rtSetDevice(" << deviceId << ") -> " << rtRet);
    }

    aRet = aclrtSetDevice(deviceId);
    if (aRet != ACL_SUCCESS) {
        std::cerr << "[ERROR] Rank " << mpiRank << ": aclrtSetDevice(" << deviceId
                  << ") failed: " << static_cast<int>(aRet) << std::endl;
        return false;
    }

    cachedDeviceId = deviceId;
    initialized = true;
    return true;
}

template <typename Func>
inline bool ForkAndRunWithHcclRootInfo(int nRanks, int firstRankId, int firstDeviceId, Func&& perRankFn)
{
    int mpiSize = CommMpiSize();
    int mpiRank = CommMpiRank();

    if (mpiSize != nRanks) {
        if (mpiRank == 0) {
            std::cerr << "[ERROR] MPI world size (" << mpiSize << ") != expected nRanks (" << nRanks
                      << "). Launch with: mpirun -n " << nRanks << " ./test_binary" << std::endl;
        }
        return false;
    }

    int rankId = firstRankId + mpiRank;
    if (nRanks <= 0) {
        return false;
    }

    int availableDevices = GetAvailableDeviceCount();
    int requiredDevices = nRanks + firstDeviceId;
    if (availableDevices < requiredDevices) {
        if (mpiRank == 0) {
            std::cerr << "[SKIP] Test requires " << requiredDevices << " NPU(s) (nRanks=" << nRanks
                      << ", firstDeviceId=" << firstDeviceId << ") but only " << availableDevices
                      << " available. Skipping." << std::endl;
        }
        return true;
    }

    int deviceId = rankId % nRanks + firstDeviceId;

    if (!EnsureAclDeviceInit(mpiRank, deviceId))
        return false;

    HcclRootInfo rootInfo{};
    if (mpiRank == 0) {
        COMM_LOG("[INIT] Rank 0: calling HcclGetRootInfo ...");
        HcclResult hret = HcclGetRootInfo(&rootInfo);
        COMM_LOG("[INIT] Rank 0: HcclGetRootInfo -> " << (int)hret);
        if (hret != HCCL_SUCCESS) {
            std::cerr << "[ERROR] HcclGetRootInfo failed: " << hret << std::endl;
            return false;
        }
    }

    CommMpiBcast(&rootInfo, HCCL_ROOT_INFO_BYTES, COMM_MPI_CHAR, 0);
    CommMpiBarrier();

    COMM_LOG("[INIT] Rank " << mpiRank << ": rootInfo broadcast complete, proceeding to test");

    return perRankFn(rankId, &rootInfo);
}

// SdmaWorkspaceManager moved to pto/comm/async/sdma/sdma_workspace_manager.hpp
#include "pto/comm/async/sdma/sdma_workspace_manager.hpp"
using SdmaWorkspaceManager = pto::comm::sdma::SdmaWorkspaceManager;
