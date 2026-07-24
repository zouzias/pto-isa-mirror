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

// Host-side communication stack for ccu_gemm_ar:
// buffers / HCCL window / CCU channel+CKE+Register+Launch.

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

#include "securec.h"
#include "acl/acl.h"
#include "hccl/hccl.h"
#include "hccl/hccl_types.h"
#include "hccl/hccl_comm.h"
#include "hccl/hccl_rank_graph.h"
#include "hccl/hccl_tiling.h"
#include "hccl/hccl_res.h"
#include "hccl/hccl_ccu_res.h"
#include "hcomm/ccu/ccu_launch.h"
#include "kernel_tiling/kernel_tiling.h"
#include "pto/comm/async/ccu/ccu_gate_registry.hpp"

#include "comm_mpi.h"
#include "comm_context.h"
#include "config.h"
#include "kernel_launchers.h"
#include "ccu_reduce_broadcast_kernel.hpp"

inline constexpr size_t WINDOW_GUARD_BYTES = 4096;

inline bool EnvFlagEnabled(const char* name)
{
    const char* env = std::getenv(name);
    if (env == nullptr || env[0] == '\0')
        return false;
    if (std::strcmp(env, "0") == 0 || std::strcmp(env, "false") == 0 || std::strcmp(env, "off") == 0 ||
        std::strcmp(env, "no") == 0)
        return false;
    return true;
}

inline bool VerboseLog()
{
    static const bool enabled = EnvFlagEnabled("PTO_CCU_GEMM_AR_VERBOSE");
    return enabled;
}

inline bool CommDiagEnabled()
{
    static const bool enabled = EnvFlagEnabled("PTO_CCU_GEMM_AR_COMM_DIAG");
    return enabled;
}

inline bool SchedTimingEnabled()
{
    static const bool enabled = EnvFlagEnabled("PTO_CCU_GEMM_AR_SCHED_TIMING");
    return enabled;
}

inline double SchedTickNs()
{
    static const double tick_ns = []() {
        const char* env = std::getenv("PTO_CCU_GEMM_AR_SCHED_TICK_NS");
        if (env == nullptr || env[0] == '\0')
            return 1.0;
        return std::atof(env);
    }();
    return tick_ns;
}

inline double SchedCyclesToUs(uint64_t cycles) { return static_cast<double>(cycles) * SchedTickNs() / 1000.0; }

inline double ChronoMicros(
    const std::chrono::high_resolution_clock::time_point& t0, const std::chrono::high_resolution_clock::time_point& t1)
{
    return std::chrono::duration<double, std::micro>(t1 - t0).count();
}

// Comm phase breakdown: wall-clock split at stream syncs + aclrtEvent elapsed.
struct CommPhaseTiming {
    double comm_wall_us{0};
    double aiv_wall_us{0};
    double ccu_wall_us{0};
    double comm_event_us{-1.0};
    double aiv_event_us{-1.0};
    double ccu_event_us{-1.0};
};

inline bool TryEventElapsedUs(aclrtEvent start, aclrtEvent end, double& out_us)
{
    out_us = -1.0;
    if (start == nullptr || end == nullptr)
        return false;
    float ms = 0.0f;
    if (aclrtEventElapsedTime(&ms, start, end) != ACL_SUCCESS)
        return false;
    const double us = static_cast<double>(ms) * 1000.0;
    // Reject garbage/uninitialized elapsed times from cross-stream queries.
    if (us < 0.0 || us > 60000000.0)
        return false;
    out_us = us;
    return true;
}

inline void DestroyAclEvents(aclrtEvent ev0, aclrtEvent ev1, aclrtEvent ev2)
{
    if (ev0 != nullptr)
        aclrtDestroyEvent(ev0);
    if (ev1 != nullptr)
        aclrtDestroyEvent(ev1);
    if (ev2 != nullptr)
        aclrtDestroyEvent(ev2);
}

inline void FillCommEventElapsed(CommPhaseTiming& timing, aclrtEvent evStart, aclrtEvent evAiv, aclrtEvent /*evCcu*/)
{
    timing.comm_event_us = -1.0;
    timing.aiv_event_us = -1.0;
    timing.ccu_event_us = -1.0;
    // Same-stream (aivStream) elapsed is reliable; cross-stream aiv→ccu is not.
    if (TryEventElapsedUs(evStart, evAiv, timing.aiv_event_us))
        timing.comm_event_us = timing.aiv_event_us + timing.ccu_wall_us;
}

inline std::vector<double> ExtractCommField(
    const std::vector<CommPhaseTiming>& records, double CommPhaseTiming::* field)
{
    std::vector<double> vals;
    vals.reserve(records.size());
    for (const CommPhaseTiming& rec : records)
        vals.push_back(rec.*field);
    return vals;
}

inline std::vector<double> ExtractValidEventField(
    const std::vector<CommPhaseTiming>& records, double CommPhaseTiming::* field)
{
    std::vector<double> vals;
    vals.reserve(records.size());
    for (const CommPhaseTiming& rec : records) {
        const double us = rec.*field;
        if (us >= 0.0)
            vals.push_back(us);
    }
    return vals;
}

inline std::string FormatEventUs(double us)
{
    if (us < 0.0)
        return "n/a";
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(1) << us;
    return oss.str();
}

inline uint32_t CountCommOutliers(const std::vector<double>& comm_us, double median_us, double std_us)
{
    uint32_t count = 0;
    const double abs_thresh = median_us + 2.0 * std_us;
    const double rel_thresh = median_us * 1.25;
    for (double us : comm_us) {
        if (us > abs_thresh || us > rel_thresh)
            ++count;
    }
    return count;
}

// Per-rank device buffer management
struct DeviceBuffers {
    void* ccuBlock{nullptr};
    // Packed communication buffers. Grouped mode stores owner-local tiles
    // contiguously so CCU can transfer several full tiles as one larger WQE.
    void* gemm_output{nullptr};
    void* reduced_output{nullptr};
    // Final row-major [M, G_N] output unpacked from reduced_output for verify.
    void* row_output{nullptr};
    // HCCL window signal_matrix: groupDone + groupReady.
    // groupReady must be in-window for peer AIV TNOTIFY via CommRemotePtr.
    void* signal_matrix{nullptr};
    uint64_t ccuToken{0};
    void* src0_dev{nullptr};
    void* src1_dev{nullptr};
    void* progressCtx_dev{nullptr};
    // Non-owning alias into signal_matrix[G_SIGNAL_GROUP_DONE_OFFSET] (do not aclrtFree).
    void* groupDone_dev{nullptr};
    size_t groupDoneSize{0};
    size_t packedSize{0};    // owner-padded packed gemm/reduced buffers
    size_t rowOutputSize{0}; // row-major [M,G_N] verify buffer
    size_t signalBytes{0};
    // HCCL window context for AIV CommRemotePtr cross-rank signal access.
    CommDeviceContext* windowDeviceCtx{nullptr};
};

inline constexpr uint32_t kMaxGemmArRanks = 16;
using SchedTiming = ProgressCtx::Timing;
inline constexpr uint32_t kCcuMissionParallel = CCU_MISSION_PARALLEL;
inline constexpr uint32_t kCcuSubtilesPerTile = G_COMM_SUBTILES_PER_TILE;

static_assert(G_NUM_TILES <= 2048, "G_NUM_TILES exceeds supported tile capacity");

struct MissionState {
    volatile uint64_t rsItemsDone;
    volatile uint64_t rsKernelReady;
    uint64_t _pad[6];
};

struct ProgressDeviceCtx {
    ProgressCtx header;
    MissionState missions[kMaxCcuMissions];
};

inline uint64_t MissionRsItemsDoneAddr(void* devCtx, uint32_t mission)
{
    return reinterpret_cast<uint64_t>(devCtx) + offsetof(ProgressDeviceCtx, missions) +
           static_cast<uint64_t>(mission) * sizeof(MissionState) + offsetof(MissionState, rsItemsDone);
}

inline uint64_t MissionRsKernelReadyAddr(void* devCtx, uint32_t mission)
{
    return reinterpret_cast<uint64_t>(devCtx) + offsetof(ProgressDeviceCtx, missions) +
           static_cast<uint64_t>(mission) * sizeof(MissionState) + offsetof(MissionState, rsKernelReady);
}

inline uint32_t MissionWorkItemCount(int nRanks, int rankId)
{
    const uint32_t safeRanks = (nRanks > 0) ? static_cast<uint32_t>(nRanks) : 1;
    return CcuOwnerGroupCount(static_cast<uint32_t>(rankId), G_NUM_TILES, safeRanks);
}

struct CcuState {
    ThreadHandle threadHandle{0};
    std::vector<ChannelHandle> channels;
    aclrtStream ccuStream{nullptr};
    aclrtStream aivStream{nullptr};

    CcuInsHandle ccuIns{0};
    CcuKernelHandle rsHandles[kMaxCcuMissions]{};
    uint32_t missionCount{0};
    bool kernelRegistered{false};

    // One-shot sequential baseline (same CCU instance, different handle; gate only, no progress).
    CcuKernelHandle seqOneShotHandle{0};
    bool seqKernelRegistered{false};
    // One-shot WaitEvent uses a dedicated seqGate CKE (PublishSeqGate); not the pipelined gate.
    uint64_t seqGateVA{0};
    uint32_t seqGateMask{0};
    uint32_t seqGateDieId{0};
    uint32_t seqGateCkeId{0};

    // Cached register-time args for HcommCcuKernelLaunch packing.
    CcuFusedReduceBroadcastKernelArg rsArgs[kMaxCcuMissions]{};
    CcuFusedReduceBroadcastKernelArg seqArg{};

    uint64_t rsProgressVA[kMaxCcuMissions][2]{};
    uint32_t rsProgressMask[kMaxCcuMissions][2]{};
    uint64_t rsGateVA[kMaxCcuMissions]{};
    uint32_t rsGateMask[kMaxCcuMissions]{};
    uint32_t rsGateDieId{0};
    uint32_t rsGateCkeId{0};
    bool ckeResolved{false}; // pipelined gate/progress VA cached after first resolve
};

// Launch args captured once so Sequential can re-launch the seq one-shot handle.
struct CcuLaunchContext {
    HcclComm comm{nullptr};
    int rankId{0};
    int nRanks{0};
    uint64_t allInputVA[kMaxGemmArRanks]{};
    uint64_t allOutputVA[kMaxGemmArRanks]{};
    uint64_t allTokens[kMaxGemmArRanks]{};
    bool valid{false};
};

inline CcuLaunchContext g_ccuLaunchCtx{};

extern "C" HcclResult HcclAllocComResourceByTiling(HcclComm comm, void* stream, void* mc2Tiling, void** commContext);
extern "C" HcclResult HcomGetCommHandleByGroup(const char* group, HcclComm* commHandle);
extern "C" HcclResult HcomGetL0TopoTypeEx(const char* group, CommTopo* topoType, uint32_t isSetDevice);

inline constexpr uint32_t COMM_IS_NOT_SET_DEVICE = 0;
inline constexpr uint32_t COMM_TOPO_MESH = 0b1u;

namespace gemm_ar_tiling {
struct CommTilingData {
    Mc2InitTiling mc2InitTiling;
    Mc2CcTiling mc2CcTiling;
};
} // namespace gemm_ar_tiling

// ============================================================================
// HcclOpResParam compat structs for RING topology
// ============================================================================
namespace hccl_compat {

struct HcclSignalInfo {
    uint64_t resId;
    uint64_t addr;
    uint32_t devId;
    uint32_t tsId;
    uint32_t rankId;
    uint32_t flag;
};

struct HcclStreamInfo {
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
static constexpr uint32_t COMPAT_USER_MEM_RES_NUM = 768;

struct ReservedStruct {
    uint8_t reserve[520];
};

struct OpCounterInfo {
    int64_t count;
    uint64_t offset;
};

struct LocalResInfoV2 {
    uint32_t streamNum;
    uint32_t signalNum;
    HcclSignalInfo localSignals[COMPAT_LOCAL_NOTIFY_MAX_NUM];
    HcclStreamInfo streamInfo[COMPAT_LOCAL_STREAM_MAX_NUM];
    HcclStreamInfo mainStreamInfo;
    HcclSignalInfo aicpuOpNotify[COMPAT_AICPU_OP_NOTIFY_MAX_NUM];
    ListCommon nextTagRes;
};

struct HierarchicalAlgInfo {
    bool isUsedHierarchicalAlg;
    uint32_t rankOrderType;
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

struct HcclOpConfig {
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

struct HcclRankRelationResV2 {
    uint32_t remoteUsrRankId;
    uint32_t remoteWorldRank;
    uint64_t windowsIn;
    uint64_t windowsOut;
    uint64_t windowsExp;
    ListCommon nextTagRes;
};

struct MemDetails1 {
    uint64_t size;
    uint64_t addr;
    uint32_t key;
};

struct HcclOpResParamHead {
    uint32_t localUsrRankId;
    uint32_t rankSize;
    uint64_t winSize;
    uint64_t localWindowsIn;
    uint64_t localWindowsOut;
    char hcomId[128];
    uint64_t winExpSize;
    uint64_t localWindowsExp;
};

struct HcclOpResParam {
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
    ReservedStruct reservedStruct;
    AlgoTopoInfo topoInfo;
    HcclOpConfig config;
    uint64_t hostStateInfo;
    uint64_t aicpuStateInfo;
    uint64_t lockAddr;
    uint32_t rsv[16];
    uint32_t notifysize;
    uint32_t remoteResNum;
    RemoteResPtr remoteRes[HCCL_MAX_RANK_NUM];
    HDCommunicateParams kfcControlTransferH2DParams;
    HDCommunicateParams kfcStatusTransferD2HParams;
    uint64_t tinyMem;
    uint64_t tinyMemSize;
    uint64_t zeroCopyHeadPtr;
    uint64_t zeroCopyTailPtr;
    uint64_t zeroCopyRingBuffer;
    uint64_t zeroCopyIpcPtrs[HCCL_MAX_RANK_NUM];
    uint32_t zeroCopyDevicePhyId[HCCL_MAX_RANK_NUM];
    bool utraceStatusFlag;
    OpCounterInfo opCounterInfo;
    HierarchicalAlgInfo hierarchicalAlgInfo;
    LocalResInfoV2 localRes;
    uint64_t debugConfig;
    uint64_t aicpuCustomParamAddr;
    uint64_t aicpuCustomParamSize;
    MemDetails1 userMemRes[COMPAT_USER_MEM_RES_NUM];
    uint32_t userMemType;
};

} // namespace hccl_compat

struct GemmHcclContext {
    HcclComm comm{nullptr};
    CommDeviceContext* deviceCtx{nullptr};
    CommDeviceContext hostCtx{};
    bool ownsDeviceCtx{false};
    bool ownsComm{false};

    // Allocate HCCL windows on an existing comm (after CCU channel setup).
    bool InitWindowOnExistingComm(int rankId, int nRanks, HcclComm existingComm, aclrtStream hcclStream)
    {
        comm = existingComm;
        ownsComm = false;
        return InitWindowResources(rankId, nRanks, hcclStream);
    }

    void Finalize()
    {
        if (ownsDeviceCtx && deviceCtx != nullptr) {
            aclrtFree(deviceCtx);
            deviceCtx = nullptr;
        }
        if (ownsComm && comm != nullptr) {
            HcclCommDestroy(comm);
            comm = nullptr;
        }
    }

private:
    bool InitWindowResources(int rankId, int nRanks, aclrtStream hcclStream)
    {
        char group[128] = {};
        CommTopo topoRet = COMM_TOPO_RESERVED;
        HcclComm commHandle = nullptr;
        if (!QueryCommTopology(rankId, group, topoRet, commHandle))
            return false;

        CommMpiBarrier();

        void* ctxPtr = nullptr;
        if (!AllocCommResource(rankId, commHandle, hcclStream, group, ctxPtr))
            return false;

        if (InitDirectPath(rankId, nRanks, ctxPtr)) {
            return true;
        }

        if (topoRet == COMM_TOPO_MESH) {
            std::cerr << "[WARN] Rank " << rankId
                      << ": direct A5 HCCL context decode failed under mesh topology, falling back to mesh-compatible"
                         " memcpy path"
                      << std::endl;
            return InitMeshPath(rankId, ctxPtr);
        }

        std::cerr << "[WARN] Rank " << rankId << ": direct A5 HCCL context decode failed, falling back to ring bridge"
                  << std::endl;
        return InitRingPath(rankId, nRanks, ctxPtr);
    }

    bool QueryCommTopology(int rankId, char* group, CommTopo& topoRet, HcclComm& commHandle)
    {
        HcclResult hret = HcclGetCommName(comm, group);
        if (hret != HCCL_SUCCESS) {
            std::cerr << "[ERROR] Rank " << rankId << ": HcclGetCommName failed: " << hret << std::endl;
            return false;
        }

        hret = HcomGetL0TopoTypeEx(group, &topoRet, COMM_IS_NOT_SET_DEVICE);
        if (hret != HCCL_SUCCESS) {
            std::cerr << "[ERROR] Rank " << rankId << ": HcomGetL0TopoTypeEx failed: " << hret << std::endl;
            return false;
        }

        hret = HcomGetCommHandleByGroup(group, &commHandle);
        if (hret != HCCL_SUCCESS) {
            std::cerr << "[ERROR] Rank " << rankId << ": HcomGetCommHandleByGroup failed: " << hret << std::endl;
            return false;
        }
        return true;
    }

    bool AllocCommResource(int rankId, HcclComm commHandle, aclrtStream hcclStream, const char* group, void*& ctxPtr)
    {
        gemm_ar_tiling::CommTilingData tiling{};
        AscendC::Mc2CcTilingConfig tilingConfig(group, HCCL_CMD_BATCH_WRITE, "BatchWrite=level0:fullmesh");
        uint32_t tilingRet = tilingConfig.SetCommBlockNum(static_cast<uint32_t>(COMM_BLOCK_NUM));
        tilingRet |= tilingConfig.SetCommEngine(3U);
        tilingRet |= tilingConfig.GetTiling(tiling.mc2InitTiling);
        tilingRet |= tilingConfig.GetTiling(tiling.mc2CcTiling);
        if (tilingRet != HCCL_SUCCESS) {
            std::cerr << "[ERROR] Rank " << rankId << ": Mc2CcTilingConfig build failed: " << tilingRet << std::endl;
            return false;
        }

        HcclResult hret = HcclAllocComResourceByTiling(commHandle, hcclStream, &tiling, &ctxPtr);
        if (hret != HCCL_SUCCESS || ctxPtr == nullptr) {
            std::cerr << "[ERROR] Rank " << rankId << ": HcclAllocComResourceByTiling failed: " << hret << std::endl;
            return false;
        }
        return true;
    }

    bool InitDirectPath(int rankId, int nRanks, void* ctxPtr)
    {
        memset_s(&hostCtx, sizeof(hostCtx), 0, sizeof(hostCtx));

        aclError aRet = aclrtMemcpy(&hostCtx, sizeof(hostCtx), ctxPtr, sizeof(hostCtx), ACL_MEMCPY_DEVICE_TO_HOST);
        if (aRet != ACL_SUCCESS) {
            return false;
        }
        if (hostCtx.rankNum == 0 || hostCtx.rankNum > HCCL_MAX_RANK_NUM) {
            return false;
        }
        if (hostCtx.rankId >= hostCtx.rankNum) {
            return false;
        }
        if (hostCtx.rankNum != static_cast<uint32_t>(nRanks)) {
            return false;
        }
        if (hostCtx.windowsIn[hostCtx.rankId] == 0 && hostCtx.windowsOut[hostCtx.rankId] == 0) {
            return false;
        }

        deviceCtx = reinterpret_cast<CommDeviceContext*>(ctxPtr);
        return true;
    }

    bool InitMeshPath(int rankId, void* ctxPtr)
    {
        deviceCtx = reinterpret_cast<CommDeviceContext*>(ctxPtr);
        aclError aRet = aclrtMemcpy(&hostCtx, sizeof(hostCtx), deviceCtx, sizeof(hostCtx), ACL_MEMCPY_DEVICE_TO_HOST);
        if (aRet != ACL_SUCCESS) {
            std::cerr << "[ERROR] Rank " << rankId << ": aclrtMemcpy(deviceCtx) failed in mesh fallback: " << (int)aRet
                      << std::endl;
            return false;
        }
        if (rankId == 0 && VerboseLog()) {
            std::cout << "[INFO] HCCL mesh-compatible fallback init OK"
                      << " rankId=" << hostCtx.rankId << " rankNum=" << hostCtx.rankNum
                      << " winSize=" << hostCtx.winSize << std::endl;
        }
        return true;
    }

    bool ReadRingParams(
        int rankId, uint8_t* rawCtx, hccl_compat::HcclOpResParamHead& head,
        std::vector<hccl_compat::RemoteResPtr>& remoteResArr)
    {
        using namespace hccl_compat;
        const size_t headOff = offsetof(HcclOpResParam, localUsrRankId);
        aclError aRet = aclrtMemcpy(&head, sizeof(head), rawCtx + headOff, sizeof(head), ACL_MEMCPY_DEVICE_TO_HOST);
        if (aRet != ACL_SUCCESS) {
            std::cerr << "[ERROR] Rank " << rankId << ": read HcclOpResParam head failed\n";
            return false;
        }

        if (head.rankSize == 0 || head.rankSize > HCCL_MAX_RANK_NUM) {
            std::cerr << "[ERROR] Rank " << rankId << ": invalid rankSize=" << head.rankSize << std::endl;
            return false;
        }

        const size_t remoteResOff = offsetof(HcclOpResParam, remoteRes);
        const size_t remoteResBytes = head.rankSize * sizeof(RemoteResPtr);
        remoteResArr.resize(head.rankSize);

        aRet = aclrtMemcpy(
            remoteResArr.data(), remoteResBytes, rawCtx + remoteResOff, remoteResBytes, ACL_MEMCPY_DEVICE_TO_HOST);
        if (aRet != ACL_SUCCESS) {
            std::cerr << "[ERROR] Rank " << rankId << ": read remoteRes failed\n";
            return false;
        }
        return true;
    }

    bool BuildRingHostCtx(
        int rankId, uint8_t* rawCtx, const hccl_compat::HcclOpResParamHead& head,
        const std::vector<hccl_compat::RemoteResPtr>& remoteResArr)
    {
        using namespace hccl_compat;
        memset_s(&hostCtx, sizeof(hostCtx), 0, sizeof(hostCtx));

        uint64_t wsFields[2] = {0, 0};
        aclError aRet = aclrtMemcpy(wsFields, sizeof(wsFields), rawCtx, sizeof(wsFields), ACL_MEMCPY_DEVICE_TO_HOST);
        if (aRet == ACL_SUCCESS) {
            hostCtx.workSpace = wsFields[0];
            hostCtx.workSpaceSize = wsFields[1];
        }

        hostCtx.rankId = head.localUsrRankId;
        hostCtx.rankNum = head.rankSize;
        hostCtx.winSize = head.winSize;

        for (uint32_t i = 0; i < head.rankSize; ++i) {
            if (i == head.localUsrRankId) {
                hostCtx.windowsIn[i] = head.localWindowsIn;
                hostCtx.windowsOut[i] = head.localWindowsOut;
                continue;
            }

            uint64_t devPtr = remoteResArr[i].nextDevicePtr;
            if (devPtr == 0) {
                std::cerr << "[ERROR] Rank " << rankId << ": remoteRes[" << i << "].nextDevicePtr is null\n";
                return false;
            }

            HcclRankRelationResV2 remoteInfo{};
            aRet = aclrtMemcpy(
                &remoteInfo, sizeof(remoteInfo), reinterpret_cast<void*>(devPtr), sizeof(remoteInfo),
                ACL_MEMCPY_DEVICE_TO_HOST);
            if (aRet != ACL_SUCCESS) {
                std::cerr << "[ERROR] Rank " << rankId << ": read remote rank " << i << " info failed\n";
                return false;
            }

            hostCtx.windowsIn[i] = remoteInfo.windowsIn;
            hostCtx.windowsOut[i] = remoteInfo.windowsOut;
        }
        return true;
    }

    bool CopyHostCtxToDevice(int rankId)
    {
        void* newDevMem = nullptr;
        aclError aRet = aclrtMalloc(&newDevMem, sizeof(CommDeviceContext), ACL_MEM_MALLOC_HUGE_FIRST);
        if (aRet != ACL_SUCCESS || newDevMem == nullptr) {
            std::cerr << "[ERROR] Rank " << rankId << ": aclrtMalloc for RING deviceCtx failed\n";
            return false;
        }

        aRet = aclrtMemcpy(
            newDevMem, sizeof(CommDeviceContext), &hostCtx, sizeof(CommDeviceContext), ACL_MEMCPY_HOST_TO_DEVICE);
        if (aRet != ACL_SUCCESS) {
            aclrtFree(newDevMem);
            std::cerr << "[ERROR] Rank " << rankId << ": copy RING deviceCtx to device failed\n";
            return false;
        }

        deviceCtx = reinterpret_cast<CommDeviceContext*>(newDevMem);
        ownsDeviceCtx = true;
        return true;
    }

    bool InitRingPath(int rankId, int nRanks, void* ctxPtr)
    {
        auto* rawCtx = reinterpret_cast<uint8_t*>(ctxPtr);

        hccl_compat::HcclOpResParamHead head{};
        std::vector<hccl_compat::RemoteResPtr> remoteResArr;
        if (!ReadRingParams(rankId, rawCtx, head, remoteResArr))
            return false;
        if (!BuildRingHostCtx(rankId, rawCtx, head, remoteResArr))
            return false;
        if (!CopyHostCtxToDevice(rankId))
            return false;

        if (rankId == 0 && VerboseLog()) {
            std::cout << "[INFO] HCCL RING init OK"
                      << " rankId=" << hostCtx.rankId << " rankNum=" << hostCtx.rankNum
                      << " winSize=" << hostCtx.winSize << std::endl;
        }
        return true;
    }
};

inline bool SyncCcuStreams(const CcuState& ccu, int rankId, const char* phase)
{
    if (ccu.ccuStream == nullptr)
        return true;
    aclError syncRet = aclrtSynchronizeStream(ccu.ccuStream);
    if (syncRet != ACL_SUCCESS) {
        std::cerr << "[CCU-AR] rank=" << rankId << ": " << phase
                  << " ccuStream sync failed: " << static_cast<int>(syncRet) << std::endl;
        return false;
    }
    return true;
}

// Probe netLayers for one peer and append the first UBC_CTP link descriptor.
inline bool TryAppendUbcCtpChannel(HcclComm comm, int rankId, int peer, std::vector<HcclChannelDesc>& requests)
{
    for (uint32_t netLayer = 0; netLayer < 3; ++netLayer) {
        uint32_t linkNum = 0;
        CommLink* linkList = nullptr;
        HcclResult rc = HcclRankGraphGetLinks(
            comm, netLayer, static_cast<uint32_t>(rankId), static_cast<uint32_t>(peer), &linkList, &linkNum);
        if (rc != HCCL_SUCCESS) {
            if (VerboseLog()) {
                std::cerr << "[CCU-AR] rank=" << rankId << ": HcclRankGraphGetLinks(layer=" << netLayer
                          << ", peer=" << peer << ") failed: " << static_cast<int>(rc) << std::endl;
            }
            continue;
        }
        for (uint32_t i = 0; i < linkNum; ++i) {
            auto proto = linkList[i].linkAttr.linkProtocol;
            if (VerboseLog()) {
                std::cerr << "[CCU-AR] rank=" << rankId << ": layer=" << netLayer << " peer=" << peer << " link[" << i
                          << "] proto=" << static_cast<int>(proto) << std::endl;
            }
            if (proto != COMM_PROTOCOL_UBC_CTP) {
                continue;
            }
            HcclChannelDesc desc;
            HcclChannelDescInit(&desc, 1);
            desc.remoteRank = static_cast<uint32_t>(peer);
            // UBC_CTP channel notify capacity used by CCU event/handshake.
            desc.notifyNum = 4;
            desc.channelProtocol = linkList[i].linkAttr.linkProtocol;
            desc.localEndpoint = linkList[i].srcEndpointDesc;
            desc.remoteEndpoint = linkList[i].dstEndpointDesc;
            if (VerboseLog()) {
                std::cerr << "[CCU-AR] rank=" << rankId << ": selected UBC_CTP link to peer=" << peer
                          << " at layer=" << netLayer << " locProto=" << static_cast<int>(desc.localEndpoint.protocol)
                          << " rmtProto=" << static_cast<int>(desc.remoteEndpoint.protocol) << std::endl;
            }
            requests.push_back(desc);
            return true;
        }
    }
    return false;
}

inline bool AcquireRequestedCcuChannels(
    HcclComm comm, int rankId, std::vector<HcclChannelDesc>& requests, std::vector<ChannelHandle>& channels)
{
    if (VerboseLog()) {
        std::cerr << "[CCU-AR] rank=" << rankId << ": submitting " << requests.size()
                  << " channel request(s) to HcclChannelAcquire" << std::endl;
    }
    channels.resize(requests.size());
    if (!requests.empty()) {
        HcclResult rc = HcclChannelAcquire(
            comm, COMM_ENGINE_CCU, requests.data(), static_cast<uint32_t>(requests.size()), channels.data());
        if (rc != HCCL_SUCCESS) {
            std::cerr << "[CCU-AR] rank=" << rankId << ": HcclChannelAcquire failed: " << static_cast<int>(rc)
                      << std::endl;
            return false;
        }
    }
    if (VerboseLog()) {
        std::cerr << "[CCU-AR] rank=" << rankId << ": acquired " << channels.size() << " CCU channel(s)" << std::endl;
    }
    return true;
}

inline bool SetupCcuChannels(HcclComm comm, int rankId, int nRanks, std::vector<ChannelHandle>& channels)
{
    std::vector<HcclChannelDesc> requests;
    for (int peer = 0; peer < nRanks; ++peer) {
        if (peer == rankId) {
            continue;
        }
        if (!TryAppendUbcCtpChannel(comm, rankId, peer, requests)) {
            std::cerr << "[CCU-AR] rank=" << rankId << ": no UBC_CTP link to peer=" << peer << " at any netLayer"
                      << std::endl;
            return false;
        }
    }
    return AcquireRequestedCcuChannels(comm, rankId, requests, channels);
}

// Same as ST/mesh: Published DieId/Id → rtGetDevResAddress → AIV poke VA.
inline uint64_t ResolveOneCkeVA(uint32_t dieId, uint32_t ckeId)
{
    constexpr int kRT_PROCESS_CP1 = 0;
    constexpr int kRT_RES_TYPE_CCU_CKE = 3;
    struct rtDevResInfo_t {
        uint32_t dieId;
        int procType;
        int resType;
        uint32_t resId;
        uint32_t flag;
    };
    struct rtDevResAddrInfo_t {
        uint64_t* resAddress;
        uint32_t* len;
    };
    using rtGetFn = int (*)(rtDevResInfo_t*, rtDevResAddrInfo_t*);

    static void* rt = dlopen("libruntime.so", RTLD_NOW | RTLD_GLOBAL);
    if (!rt)
        return 0;
    static auto rtGet = reinterpret_cast<rtGetFn>(dlsym(rt, "rtGetDevResAddress"));
    if (!rtGet)
        return 0;

    rtDevResInfo_t in{};
    in.dieId = dieId;
    in.procType = kRT_PROCESS_CP1;
    in.resType = kRT_RES_TYPE_CCU_CKE;
    in.resId = ckeId;
    in.flag = 0;
    uint64_t addr = 0;
    uint32_t len = 0;
    rtDevResAddrInfo_t out{&addr, &len};
    int rc = rtGet(&in, &out);
    return (rc == 0) ? addr : 0;
}

// Shared registry poll: TryGet* may race Translate/Publish after RegisterEnd.
template <typename TryFn>
inline bool TryCcuDescriptorWithRetry(int rankId, const char* what, TryFn&& tryFn)
{
    for (int retry = 0; retry < 200; ++retry) {
        if (tryFn()) {
            return true;
        }
        usleep(10000);
    }
    std::cerr << "[CCU-AR] rank=" << rankId << ": " << what << " failed after retries" << std::endl;
    return false;
}

inline CcuFusedReduceBroadcastTaskArg MakeFusedTaskArg(
    int rankId, int nRanks, const uint64_t* allInputVA, const uint64_t* allOutputVA, const uint64_t* allTokens,
    uint64_t length)
{
    CcuFusedReduceBroadcastTaskArg taskArg{};
    taskArg.inputAddr = allInputVA[rankId];
    taskArg.outputAddr = allOutputVA[rankId];
    taskArg.length = length;
    taskArg.SetPeerAddrs(static_cast<uint32_t>(nRanks), allInputVA, allOutputVA, allTokens);
    return taskArg;
}

// Gate CKE via dedicated registry Publish/TryGet (not a progress[] slot).
inline bool ResolveCcuGate(int rankId, CcuState& ccu)
{
    if (ccu.ckeResolved && ccu.rsGateVA[0] != 0)
        return true;

    pto::comm::ccu::CcuGateDescriptor desc{};
    if (!TryCcuDescriptorWithRetry(
            rankId, "gate TryGet", [&]() { return pto::comm::ccu::TryGet(static_cast<uint32_t>(rankId), desc); })) {
        return false;
    }

    // Pipelined gate (registry `gate` slot). Sequential uses `seqGate` separately.
    uint64_t addr = ResolveOneCkeVA(desc.dieId, desc.ckeId);
    if (addr == 0) {
        std::cerr << "[CCU-AR] rank=" << rankId << ": fused gate VA resolution failed (die=" << desc.dieId
                  << " cke=" << desc.ckeId << ")" << std::endl;
        return false;
    }
    ccu.rsGateVA[0] = addr;
    ccu.rsGateMask[0] = desc.mask;
    ccu.rsGateDieId = desc.dieId;
    ccu.rsGateCkeId = desc.ckeId;
    if (VerboseLog()) {
        std::cerr << "[CCU-AR] rank=" << rankId << ": gate die=" << desc.dieId << " cke=" << desc.ckeId << " mask=0x"
                  << std::hex << desc.mask << " VA=0x" << addr << std::dec << std::endl;
    }
    return true;
}

// Edge-triggered gate poke (no host D2H). A couple of poke+sync rounds cover CCU
// arriving at WaitEvent slightly later.
inline bool TriggerCkeGate(CcuState& ccu, int rankId, uint64_t gateVA, uint32_t gateMask, const char* label)
{
    if (gateVA == 0) {
        std::cerr << "[CCU-AR] rank=" << rankId << ": " << label << " called with null gate VA" << std::endl;
        return false;
    }
    constexpr int kPokes = 2;
    for (int i = 0; i < kPokes; ++i) {
        if (launchCcuGemmArGateTrigger(ccu.aivStream, gateVA, gateMask) != 0) {
            std::cerr << "[CCU-AR] rank=" << rankId << ": " << label << " trigger failed" << std::endl;
            return false;
        }
        aclError ar = aclrtSynchronizeStream(ccu.aivStream);
        if (ar != ACL_SUCCESS) {
            std::cerr << "[CCU-AR] rank=" << rankId << ": " << label
                      << " aivStream sync failed: " << static_cast<int>(ar) << std::endl;
            return false;
        }
    }
    return true;
}

inline bool TriggerCcuGate(DeviceBuffers& /*buf*/, CcuState& ccu, int rankId)
{
    return TriggerCkeGate(ccu, rankId, ccu.rsGateVA[0], ccu.rsGateMask[0], "pipelined gate");
}

// Sequential: after full GEMM, peer-sync then a single seqGate poke (inside the AIV kernel).
// Do not add a host follow-up poke: after WaitEvent consumes the edge, an extra poke latches
// the next Launch and lets one-shot RS→AG run during GEMM.
inline bool TriggerSeqOneShotAfterPeerSync(CcuState& ccu, const DeviceBuffers& buf, int rankId, int nRanks)
{
    if (ccu.seqGateVA == 0) {
        std::cerr << "[CCU-AR] rank=" << rankId << ": seq peer-sync gate called with null gate VA" << std::endl;
        return false;
    }
    launchCcuGemmArSeqPeerSyncGate(
        ccu.aivStream, ccu.seqGateVA, ccu.seqGateMask, reinterpret_cast<int32_t*>(buf.groupDone_dev),
        reinterpret_cast<uint8_t*>(buf.signal_matrix), reinterpret_cast<uint8_t*>(buf.windowDeviceCtx), G_NUM_TILES,
        static_cast<uint32_t>(nRanks), static_cast<uint32_t>(rankId));
    aclError ar = aclrtSynchronizeStream(ccu.aivStream);
    if (ar != ACL_SUCCESS) {
        std::cerr << "[CCU-AR] rank=" << rankId << ": seq peer-sync aivStream sync failed: " << static_cast<int>(ar)
                  << std::endl;
        return false;
    }
    return true;
}

// After batch RegisterEnd/Publish: cache Sequential seqGate VA (separate from pipelined gate).
inline bool CacheSeqOneShotGate(int rankId, CcuState& ccu)
{
    pto::comm::ccu::CcuGateDescriptor desc{};
    if (!TryCcuDescriptorWithRetry(rankId, "seq one-shot gate TryGetSeqGate", [&]() {
            return pto::comm::ccu::TryGetSeqGate(static_cast<uint32_t>(rankId), desc);
        })) {
        return false;
    }
    uint64_t addr = ResolveOneCkeVA(desc.dieId, desc.ckeId);
    if (addr == 0) {
        std::cerr << "[CCU-AR] rank=" << rankId << ": seq one-shot gate VA resolve failed (die=" << desc.dieId
                  << " cke=" << desc.ckeId << ")" << std::endl;
        return false;
    }
    ccu.seqGateVA = addr;
    ccu.seqGateMask = desc.mask;
    ccu.seqGateDieId = desc.dieId;
    ccu.seqGateCkeId = desc.ckeId;
    if (VerboseLog()) {
        std::cerr << "[CCU-AR] rank=" << rankId << ": seq gate die=" << desc.dieId << " cke=" << desc.ckeId
                  << " mask=0x" << std::hex << desc.mask << " VA=0x" << addr << std::dec << std::endl;
    }
    return true;
}

// After pipelined gate resolve: prove seq and pipe gates are distinct physical CKEs.
inline bool EnsureSeqGateDistinctFromPipelined(int rankId, const CcuState& ccu)
{
    if (!ccu.seqKernelRegistered || ccu.seqGateVA == 0 || ccu.rsGateVA[0] == 0) {
        return true;
    }
    if (ccu.seqGateVA == ccu.rsGateVA[0] ||
        (ccu.seqGateDieId == ccu.rsGateDieId && ccu.seqGateCkeId == ccu.rsGateCkeId)) {
        std::cerr << "[CCU-AR] rank=" << rankId << ": FATAL seq/pipe gate collide"
                  << " seq(die=" << ccu.seqGateDieId << " cke=" << ccu.seqGateCkeId << " VA=0x" << std::hex
                  << ccu.seqGateVA << ") pipe(die=" << ccu.rsGateDieId << " cke=" << ccu.rsGateCkeId << " VA=0x"
                  << ccu.rsGateVA[0] << ")" << std::dec << " — WaitEvent cannot isolate Sequential from PrepareCcu poke"
                  << std::endl;
        return false;
    }
    if (VerboseLog()) {
        std::cerr << "[CCU-AR] rank=" << rankId << ": seq gate die=" << ccu.seqGateDieId << " cke=" << ccu.seqGateCkeId
                  << " VA=0x" << std::hex << ccu.seqGateVA << " | pipe gate die=" << std::dec << ccu.rsGateDieId
                  << " cke=" << ccu.rsGateCkeId << " VA=0x" << std::hex << ccu.rsGateVA[0] << std::dec
                  << " (distinct OK)" << std::endl;
    }
    return true;
}

// Fused progress CKEs: registry slots A/B (ping-pong).
inline constexpr uint32_t kProgressCkeSlotA = 0;
inline constexpr uint32_t kProgressCkeSlotB = 1;
inline constexpr uint32_t kProgressCkeSlotCount = 2;

inline bool ResolveCcuProgress(int rankId, CcuState& ccu)
{
    if (ccu.ckeResolved && ccu.rsProgressVA[0][0] != 0)
        return true;

    std::vector<pto::comm::ccu::CcuGateDescriptor> progDescs;
    const std::string progressWhat = "progress TryGet (need " + std::to_string(kProgressCkeSlotCount) + " descriptors)";
    if (!TryCcuDescriptorWithRetry(rankId, progressWhat.c_str(), [&]() {
            return pto::comm::ccu::TryGetProgress(static_cast<uint32_t>(rankId), progDescs) &&
                   progDescs.size() >= kProgressCkeSlotCount;
        })) {
        return false;
    }

    const uint32_t slots[kProgressCkeSlotCount] = {kProgressCkeSlotA, kProgressCkeSlotB};
    for (uint32_t p = 0; p < kProgressCkeSlotCount; ++p) {
        const auto& desc = progDescs[slots[p]];
        uint64_t progAddr = ResolveOneCkeVA(desc.dieId, desc.ckeId);
        if (progAddr == 0) {
            std::cerr << "[CCU-AR] rank=" << rankId << ": fused progress[p=" << p << "] VA resolution failed"
                      << std::endl;
            return false;
        }
        ccu.rsProgressVA[0][p] = progAddr;
        ccu.rsProgressMask[0][p] = desc.mask;
    }
    ccu.ckeResolved = true;
    return true;
}

inline bool WriteProgressCtxToDevice(DeviceBuffers& buf)
{
    aclrtMemset(buf.progressCtx_dev, sizeof(ProgressDeviceCtx), 0, sizeof(ProgressDeviceCtx));

    ProgressCtx header{};
    header.numTiles = G_NUM_TILES;
    header.schedTimingEnable = SchedTimingEnabled() ? 1u : 0u;

    aclrtMemcpy(buf.progressCtx_dev, sizeof(ProgressCtx), &header, sizeof(ProgressCtx), ACL_MEMCPY_HOST_TO_DEVICE);
    return true;
}

// Sequential one-shot payload: real owner tile footprint (not group-padded), so
// seq_comm does not depend on COMM_GROUP_TILES — same intent as v3 oneShotPayloadBytes.
inline uint64_t SeqOneShotPayloadBytes(int rankId, int nRanks)
{
    const uint32_t safeRanks = (nRanks > 0) ? static_cast<uint32_t>(nRanks) : 1;
    const uint32_t ownerTiles = CcuOwnerTileCount(static_cast<uint32_t>(rankId), G_NUM_TILES, safeRanks);
    return static_cast<uint64_t>(ownerTiles) * G_TILE_BYTES;
}

inline void FillFusedKernelArg(
    CcuFusedReduceBroadcastKernelArg& kernelArg, CcuState& ccu, int rankId, int nRanks, const DeviceBuffers& buf,
    uint32_t mission, bool forceOneShot = false)
{
    kernelArg.rankId = static_cast<uint32_t>(rankId);
    kernelArg.rankSize = static_cast<uint32_t>(nRanks);
    kernelArg.rootId = static_cast<uint32_t>(rankId);
    kernelArg.dataType = HcclDataType::HCCL_DATA_TYPE_FP16;
    kernelArg.outputDataType = HcclDataType::HCCL_DATA_TYPE_FP16;
    kernelArg.reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    kernelArg.gateMask = pto::comm::ccu::CCU_GATE_MASK;
    kernelArg.doneMask = pto::comm::ccu::CCU_DONE_MASK;
    kernelArg.channels = ccu.channels;
    kernelArg.oneShotBaselineOnly = forceOneShot ? 1 : 0;

    // v3-style sequential group offsets: owner shard starts at packed prefix.
    const uint32_t safeRanks = (nRanks > 0) ? static_cast<uint32_t>(nRanks) : 1;
    kernelArg.baseOffsetBytes =
        static_cast<uint64_t>(CcuOwnerTilePrefix(static_cast<uint32_t>(rankId), G_NUM_TILES, safeRanks)) *
        static_cast<uint64_t>(G_TILE_BYTES);

    if (forceOneShot) {
        kernelArg.payloadBytes = SeqOneShotPayloadBytes(rankId, nRanks);
        kernelArg.loopCount = pto::comm::ccu::CalcLoopCount(kernelArg.payloadBytes);
        kernelArg.missionWorkItems = 1;
        kernelArg.progressMask = 0;
        kernelArg.progressMaskB = 0;
        return;
    }

    // Event-mask bits: gate/done bit0; progress A/B bit1/2 (K=1; no per-mission stride).
    (void)mission;
    kernelArg.payloadBytes = G_GROUP_BYTES;
    kernelArg.progressMask = 1u << 1;
    kernelArg.progressMaskB = 1u << 2;
    kernelArg.missionWorkItems = MissionWorkItemCount(nRanks, rankId);
    if (buf.progressCtx_dev != nullptr) {
        kernelArg.itemsDoneAddr = MissionRsItemsDoneAddr(buf.progressCtx_dev, mission);
        kernelArg.kernelReadyAddr = MissionRsKernelReadyAddr(buf.progressCtx_dev, mission);
    }
}

// Match mesh/ST: pin CCU IO die (env HCCL_PTO_GATE_DIE_ID, default 1 on A5).
inline uint32_t ResolveCcuDieId()
{
    uint32_t dieId = 1;
    if (const char* env = std::getenv("HCCL_PTO_GATE_DIE_ID"); env != nullptr && env[0] != '\0') {
        const int v = std::atoi(env);
        if (v >= 0) {
            dieId = static_cast<uint32_t>(v);
        }
    }
    return dieId;
}

// Register seq one-shot into an open RegisterStart/End batch (stash only; Publish after End).
inline bool RegisterSeqOneShotKernelInBatch(
    CcuState& ccu, int rankId, int nRanks, const DeviceBuffers& buf, uint32_t dieId)
{
    FillFusedKernelArg(ccu.seqArg, ccu, rankId, nRanks, buf, 0, true);
    CcuKernelHandle seqHandle = 0;
    CcuResult cret = RegisterFusedReduceBroadcastKernel(ccu.ccuIns, dieId, ccu.seqArg, &seqHandle);
    if (cret != CcuResult::CCU_SUCCESS) {
        std::cerr << "[CCU-AR] rank=" << rankId << ": seq one-shot register failed: " << static_cast<int>(cret)
                  << std::endl;
        return false;
    }
    ccu.seqOneShotHandle = seqHandle;
    ccu.seqKernelRegistered = true;
    return true;
}

// Register pipelined fused kernel(s) into an open RegisterStart/End batch.
inline bool RegisterPipelinedKernelsInBatch(
    CcuState& ccu, int rankId, int nRanks, const DeviceBuffers& buf, uint32_t dieId)
{
    for (uint32_t m = 0; m < kCcuMissionParallel; ++m) {
        FillFusedKernelArg(ccu.rsArgs[m], ccu, rankId, nRanks, buf, m, false);
        CcuKernelHandle handle = 0;
        CcuResult cret = RegisterFusedReduceBroadcastKernel(ccu.ccuIns, dieId, ccu.rsArgs[m], &handle);
        if (cret != CcuResult::CCU_SUCCESS) {
            std::cerr << "[CCU-AR] rank=" << rankId << ": fused register(mission=" << m
                      << ") failed: " << static_cast<int>(cret) << std::endl;
            return false;
        }
        ccu.rsHandles[m] = handle;
    }
    ccu.missionCount = kCcuMissionParallel;
    return true;
}

inline void LogCcuRegistration(const CcuState& ccu, int rankId, int nRanks)
{
    if (!VerboseLog()) {
        return;
    }
    const uint64_t payloadBytes = G_GROUP_BYTES;
    const uint32_t loopCount = pto::comm::ccu::CalcLoopCount(payloadBytes);
    std::cerr << "[CCU-AR] rank=" << rankId << ": registered " << ccu.missionCount
              << " fused Reduce+Broadcast kernel(s), numTiles=" << G_NUM_TILES << " payloadBytes=" << payloadBytes
              << " groupTiles=" << G_COMM_GROUP_TILES << " items=" << MissionWorkItemCount(nRanks, rankId)
              << " loopCount=" << loopCount << (ccu.seqKernelRegistered ? " + seq one-shot baseline" : "") << std::endl;
    if (ccu.seqKernelRegistered) {
        std::cerr << "[CCU-AR] rank=" << rankId
                  << ": seq one-shot payloadBytes=" << SeqOneShotPayloadBytes(rankId, nRanks)
                  << " (owner tiles, independent of group_tiles)" << std::endl;
    }
}

inline bool RegisterCcuGemmArOnce(HcclComm comm, CcuState& ccu, int rankId, int nRanks, const DeviceBuffers& buf)
{
    if (ccu.kernelRegistered) {
        return true;
    }
    if (QueryPrimaryCcuIns(comm, &ccu.ccuIns) != CcuResult::CCU_SUCCESS) {
        std::cerr << "[CCU-AR] rank=" << rankId << ": QueryPrimaryCcuIns failed" << std::endl;
        return false;
    }
    const uint32_t dieId = ResolveCcuDieId();

    // One RegisterStart/End batch so seq + pipelined gates get distinct physical CKEs.
    // Split sessions previously recycled the same gate CKE → PrepareCcu poke released Sequential.
    ClearFusedCkePublishTls();
    CcuResult cret = HcommCcuKernelRegisterStart(ccu.ccuIns);
    if (cret != CcuResult::CCU_SUCCESS) {
        std::cerr << "[CCU-AR] rank=" << rankId << ": RegisterStart failed: " << static_cast<int>(cret) << std::endl;
        return false;
    }
    if (!RegisterSeqOneShotKernelInBatch(ccu, rankId, nRanks, buf, dieId) ||
        !RegisterPipelinedKernelsInBatch(ccu, rankId, nRanks, buf, dieId)) {
        (void)HcommCcuKernelRegisterEnd(ccu.ccuIns);
        ClearFusedCkePublishTls();
        return false;
    }
    cret = HcommCcuKernelRegisterEnd(ccu.ccuIns);
    if (cret != CcuResult::CCU_SUCCESS) {
        std::cerr << "[CCU-AR] rank=" << rankId << ": RegisterEnd failed: " << static_cast<int>(cret) << std::endl;
        ClearFusedCkePublishTls();
        return false;
    }
    // Translate done — publish seqGate + pipelined gate/progress from the stash list.
    cret = PublishStashedCkeAfterRegisterEnd();
    if (cret != CcuResult::CCU_SUCCESS) {
        std::cerr << "[CCU-AR] rank=" << rankId << ": post-RegisterEnd CKE Publish failed: " << static_cast<int>(cret)
                  << std::endl;
        return false;
    }
    if (!CacheSeqOneShotGate(rankId, ccu)) {
        return false;
    }
    if (!ResolveCcuGate(rankId, ccu) || !EnsureSeqGateDistinctFromPipelined(rankId, ccu)) {
        return false;
    }
    ccu.kernelRegistered = true;
    LogCcuRegistration(ccu, rankId, nRanks);
    return true;
}

inline bool LaunchCcuGemmAr(
    HcclComm /*comm*/, CcuState& ccu, int rankId, int nRanks, const uint64_t* allInputVA, const uint64_t* allOutputVA,
    const uint64_t* allTokens, const DeviceBuffers& buf)
{
    (void)buf;
    if (!ccu.kernelRegistered) {
        std::cerr << "[CCU-AR] rank=" << rankId << ": LaunchCcuGemmAr called before RegisterCcuGemmArOnce" << std::endl;
        return false;
    }

    const uint64_t payloadBytes = G_GROUP_BYTES;
    for (uint32_t m = 0; m < ccu.missionCount; ++m) {
        // input=gemm_output (pull), output=reduced_output (reduce dst + broadcast src/dst)
        const CcuFusedReduceBroadcastTaskArg taskArg =
            MakeFusedTaskArg(rankId, nRanks, allInputVA, allOutputVA, allTokens, payloadBytes);

        CcuResult cret = LaunchFusedReduceBroadcastKernel(ccu.threadHandle, ccu.rsHandles[m], ccu.rsArgs[m], taskArg);
        if (cret != CcuResult::CCU_SUCCESS) {
            std::cerr << "[CCU-AR] rank=" << rankId << ": fused Launch(mission=" << m
                      << ") failed: " << static_cast<int>(cret) << std::endl;
            return false;
        }
    }
    return true;
}

inline void StoreCcuLaunchContext(
    HcclComm comm, int rankId, int nRanks, const uint64_t* allInputVA, const uint64_t* allOutputVA,
    const uint64_t* allTokens)
{
    g_ccuLaunchCtx.comm = comm;
    g_ccuLaunchCtx.rankId = rankId;
    g_ccuLaunchCtx.nRanks = nRanks;
    for (int i = 0; i < nRanks && i < static_cast<int>(kMaxGemmArRanks); ++i) {
        g_ccuLaunchCtx.allInputVA[i] = allInputVA[i];
        g_ccuLaunchCtx.allOutputVA[i] = allOutputVA[i];
        g_ccuLaunchCtx.allTokens[i] = allTokens[i];
    }
    g_ccuLaunchCtx.valid = true;
}

inline bool LaunchSeqOneShotCcuGemmAr(
    CcuState& ccu, int rankId, int nRanks, const uint64_t* allInputVA, const uint64_t* allOutputVA,
    const uint64_t* allTokens, HcclComm /*comm*/)
{
    if (!ccu.seqKernelRegistered) {
        std::cerr << "[CCU-AR] rank=" << rankId << ": seq one-shot kernel not registered" << std::endl;
        return false;
    }
    const CcuFusedReduceBroadcastTaskArg taskArg =
        MakeFusedTaskArg(rankId, nRanks, allInputVA, allOutputVA, allTokens, SeqOneShotPayloadBytes(rankId, nRanks));

    CcuResult cret = LaunchFusedReduceBroadcastKernel(ccu.threadHandle, ccu.seqOneShotHandle, ccu.seqArg, taskArg);
    if (cret != CcuResult::CCU_SUCCESS) {
        std::cerr << "[CCU-AR] rank=" << rankId << ": seq one-shot Launch failed: " << static_cast<int>(cret)
                  << std::endl;
        return false;
    }
    return true;
}

inline bool LaunchStoredSeqOneShotCcuGemmAr(CcuState& ccu)
{
    if (!g_ccuLaunchCtx.valid)
        return false;
    return LaunchSeqOneShotCcuGemmAr(
        ccu, g_ccuLaunchCtx.rankId, g_ccuLaunchCtx.nRanks, g_ccuLaunchCtx.allInputVA, g_ccuLaunchCtx.allOutputVA,
        g_ccuLaunchCtx.allTokens, g_ccuLaunchCtx.comm);
}

// Full CCU preparation: sync prior pass, refresh ctx, launch CCU, resolve gate/progress CKEs.
// CKE descriptors are published once after RegisterEnd (Translate) and remain valid across launches.
inline bool PrepareCcuKernel(
    DeviceBuffers& buf, HcclComm comm, CcuState& ccu, int rankId, int nRanks, const uint64_t* allInputVA,
    const uint64_t* allOutputVA, const uint64_t* allTokens)
{
    // Wait for the previous CCU pass before launching another microcode instance
    // on the same thread; otherwise the new kernel can stall behind a stuck waiter.
    if (!SyncCcuStreams(ccu, rankId, "ccuStream pre-launch"))
        return false;

    if (!WriteProgressCtxToDevice(buf))
        return false;

    if (!LaunchCcuGemmAr(comm, ccu, rankId, nRanks, allInputVA, allOutputVA, allTokens, buf))
        return false;

    if (!ResolveCcuGate(rankId, ccu))
        return false;
    // Light poke; CCU Stores rsKernelReady after WaitEvent returns.
    if (!TriggerCcuGate(buf, ccu, rankId))
        return false;

    if (!ResolveCcuProgress(rankId, ccu))
        return false;

    return true;
}

inline bool InitCcuThreadAndChannels(HcclComm hcclComm, int rankId, int nRanks, CcuState& ccu)
{
    constexpr uint32_t kNotifyNum = 1;
    HcclResult hret =
        HcclThreadAcquireWithStream(hcclComm, COMM_ENGINE_CCU, ccu.ccuStream, kNotifyNum, &ccu.threadHandle);
    if (hret != HCCL_SUCCESS) {
        std::cerr << "[ERROR] Rank " << rankId << ": HcclThreadAcquireWithStream failed: " << static_cast<int>(hret)
                  << std::endl;
        return false;
    }
    if (!SetupCcuChannels(hcclComm, rankId, nRanks, ccu.channels)) {
        std::cerr << "[ERROR] Rank " << rankId << ": CCU channel setup failed!\n";
        return false;
    }
    return true;
}
