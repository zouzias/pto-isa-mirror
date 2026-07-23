/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

/**
 * CCU GEMM AllReduce — CCU RS+AG (Pull Reduce @ owner + Push Broadcast)
 * HCCL backend — launched via mpirun
 *
 * Unlike v3 (full-buffer GroupReduce on every rank), this demo scopes communication
 * to owner shards for RS+AG-equivalent bandwidth ~ 2*(P-1)/P:
 *   1. AIC: PerBlockQueue + swizzle; gemm_output packed by owner-contiguous layout
 *      (each owner shard padded to a multiple of comm-group-tiles for residual groups).
 *   2. AIV progress: local groupDone → peer TNOTIFY(groupReady) → owner TriggerProgressCke.
 *   3. CCU pipelined: HcommCcuKernelRegister fused RS+AG (pull reduce then push broadcast per group).
 *   4. CCU Sequential baseline: one-shot fused RS+AG on the same CCU instance
 *      (one fused RS→AG over the full owner shard; no AIV progress CKE).
 *
 * Usage:
 *   mpirun -n <NRANKS> ./ccu_gemm_allreduce [--first-device ID]
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cstddef>
#include "securec.h"
#include <cmath>
#include <vector>
#include <random>
#include <thread>
#include <chrono>
#include <algorithm>
#include <string>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>

#include <sys/stat.h>
#include <sys/wait.h>
#include <signal.h>
#include <unistd.h>

#include "acl/acl.h"
#include "acl/error_codes/rt_error_codes.h"
#include "hccl/hccl.h"
#include "hccl/hccl_types.h"
#include "hccl/hccl_comm.h"
#include "hccl/hccl_tiling.h"
#include "kernel_tiling/kernel_tiling.h"
#include "comm_mpi.h"

#include "comm_context.h"

#include "config.h"

#include <dlfcn.h>

#include "hccl/hccl_res.h"
#include "hccl/hccl_rank_graph.h"
#include "hccl/hccl_ccu_res.h"
#include "hcomm/ccu/ccu_launch.h"
#include "pto/comm/async/ccu/ccu_gate_registry.hpp"
#include "ccu_reduce_broadcast_kernel.hpp"
#include "kernel_launchers.h"

// ccu_ready_queue.hpp pulls in device-only pto_comm_inst.hpp unless built for
// the kernel test path. main.cpp is host code, so suppress that include and use
// only the host-side queue helpers (MultiBlockQueueSetInit / Size).
#ifndef __CCE_KT_TEST__
#define __CCE_KT_TEST__
#define __CCU_GEMM_AR_KT_TEST_DEFINED_HERE__
#endif
#include "ready_queue.hpp"
#ifdef __CCU_GEMM_AR_KT_TEST_DEFINED_HERE__
#undef __CCE_KT_TEST__
#undef __CCU_GEMM_AR_KT_TEST_DEFINED_HERE__
#endif

extern "C" HcclResult HcclAllocComResourceByTiling(HcclComm comm, void* stream, void* mc2Tiling, void** commContext);
extern "C" HcclResult HcomGetCommHandleByGroup(const char* group, HcclComm* commHandle);

extern "C" HcclResult HcomGetL0TopoTypeEx(const char* group, CommTopo* topoType, uint32_t isSetDevice);

// cann-9.2 libtiling_api.a may pull platform_ascendc.o which needs C CheckLogLevel.
// Stub here so CMake link libs stay peer-style (no -lunified_dlog / extra .cpp).
extern "C" int32_t CheckLogLevel(int32_t moduleId, int32_t logLevel)
{
    (void)moduleId;
    (void)logLevel;
    return 0;
}
static constexpr uint32_t COMM_IS_NOT_SET_DEVICE = 0;
static constexpr uint32_t COMM_TOPO_MESH = 0b1u;
static constexpr size_t WINDOW_GUARD_BYTES = 4096;

static bool EnvFlagEnabled(const char* name)
{
    const char* env = std::getenv(name);
    if (env == nullptr || env[0] == '\0')
        return false;
    if (std::strcmp(env, "0") == 0 || std::strcmp(env, "false") == 0 || std::strcmp(env, "off") == 0 ||
        std::strcmp(env, "no") == 0)
        return false;
    return true;
}

static bool VerboseLog()
{
    static const bool enabled = EnvFlagEnabled("PTO_CCU_GEMM_AR_VERBOSE");
    return enabled;
}

static bool CommDiagEnabled()
{
    static const bool enabled = EnvFlagEnabled("PTO_CCU_GEMM_AR_COMM_DIAG");
    return enabled;
}

static bool SchedTimingEnabled()
{
    static const bool enabled = EnvFlagEnabled("PTO_CCU_GEMM_AR_SCHED_TIMING");
    return enabled;
}

// Default AIV path (run.sh USE_PROGRESS=1 exports PTO_CCU_GEMM_AR_USE_PROGRESS):
// local groupDone + peer TNOTIFY(groupReady) + owner progress CKE.
// USE_PROGRESS=0 unsets / sets false → readyQueue escape path.
static bool UseProgressPath() { return EnvFlagEnabled("PTO_CCU_GEMM_AR_USE_PROGRESS"); }

static double SchedTickNs()
{
    static const double tick_ns = []() {
        const char* env = std::getenv("PTO_CCU_GEMM_AR_SCHED_TICK_NS");
        if (env == nullptr || env[0] == '\0')
            return 1.0;
        return std::atof(env);
    }();
    return tick_ns;
}

static double SchedCyclesToUs(uint64_t cycles) { return static_cast<double>(cycles) * SchedTickNs() / 1000.0; }

static double ChronoMicros(
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

static bool TryEventElapsedUs(aclrtEvent start, aclrtEvent end, double& out_us)
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

static void DestroyAclEvents(aclrtEvent ev0, aclrtEvent ev1, aclrtEvent ev2)
{
    if (ev0 != nullptr)
        aclrtDestroyEvent(ev0);
    if (ev1 != nullptr)
        aclrtDestroyEvent(ev1);
    if (ev2 != nullptr)
        aclrtDestroyEvent(ev2);
}

static void FillCommEventElapsed(CommPhaseTiming& timing, aclrtEvent evStart, aclrtEvent evAiv, aclrtEvent /*evCcu*/)
{
    timing.comm_event_us = -1.0;
    timing.aiv_event_us = -1.0;
    timing.ccu_event_us = -1.0;
    // Same-stream (aivStream) elapsed is reliable; cross-stream aiv→ccu is not.
    if (TryEventElapsedUs(evStart, evAiv, timing.aiv_event_us))
        timing.comm_event_us = timing.aiv_event_us + timing.ccu_wall_us;
}

static std::vector<double> ExtractCommField(
    const std::vector<CommPhaseTiming>& records, double CommPhaseTiming::* field)
{
    std::vector<double> vals;
    vals.reserve(records.size());
    for (const CommPhaseTiming& rec : records)
        vals.push_back(rec.*field);
    return vals;
}

static std::vector<double> ExtractValidEventField(
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

static std::string FormatEventUs(double us)
{
    if (us < 0.0)
        return "n/a";
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(1) << us;
    return oss.str();
}

static uint32_t CountCommOutliers(const std::vector<double>& comm_us, double median_us, double std_us)
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

// Step trace for hang localization. Uses cerr + flush so mpirun sees it promptly.
static void TraceStep(int rank_id, const char* step)
{
    if (!VerboseLog())
        return;
    std::cerr << "[TRACE] rank=" << rank_id << " " << step << std::endl;
    std::cerr.flush();
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
    // HCCL window signal_matrix: subtile_ready + groupDone + groupReady.
    // groupReady must be in-window for peer AIV TNOTIFY via CommRemotePtr.
    void* signal_matrix{nullptr};
    uint64_t ccuToken{0};
    void* src0_dev{nullptr};
    void* src1_dev{nullptr};
    void* progressCtx_dev{nullptr};
    void* readyQueue_dev{nullptr};
    // Non-owning alias into signal_matrix[G_SIGNAL_GROUP_DONE_OFFSET] (do not aclrtFree).
    void* groupDone_dev{nullptr};
    MultiBlockQueueSet* readyQueue_reset_host{nullptr};
    size_t readyQueueSize{0};
    size_t groupDoneSize{0};
    size_t packedSize{0};    // owner-padded packed gemm/reduced buffers
    size_t rowOutputSize{0}; // row-major [M,G_N] verify buffer
    size_t signalBytes{0};
    // HCCL window context for AIV CommRemotePtr cross-rank signal access.
    CommDeviceContext* windowDeviceCtx{nullptr};
};

// ============================================================================
// CCU progress context — host/AIV share ProgressCtx / ProgressCkeCtx (kernel_launchers.h).
// ============================================================================
static constexpr uint32_t kMaxGemmArRanks = 16;

// Sched timing type — matches ProgressCtx::Timing layout.
using SchedTiming = ProgressCtx::Timing;

static SchedTiming g_lastPipeRsTiming{};
static bool g_lastPipeSchedTimingValid = false;

static constexpr uint32_t kCcuMissionParallel = CCU_MISSION_PARALLEL;
static constexpr uint32_t kCcuSubtilesPerTile = G_COMM_SUBTILES_PER_TILE;

static_assert(G_NUM_TILES <= 2048, "G_NUM_TILES exceeds scheduler pending-array capacity (kSchedMaxTiles)");

// Per-mission device state: itemsDone + kernelReady counters.
struct MissionState {
    volatile uint64_t rsItemsDone;   // CCU writes, AIV reads
    volatile uint64_t rsKernelReady; // CCU: 1 after gate WaitEvent returns; AIV waits !=0
    uint64_t _pad[6];
};

// Full device progress context: header + per-mission data.
struct ProgressDeviceCtx {
    ProgressCtx header;
    MissionState missions[kMaxCcuMissions];
};

static inline uint64_t MissionRsItemsDoneAddr(void* devCtx, uint32_t mission)
{
    return reinterpret_cast<uint64_t>(devCtx) + offsetof(ProgressDeviceCtx, missions) +
           static_cast<uint64_t>(mission) * sizeof(MissionState) + offsetof(MissionState, rsItemsDone);
}

static inline uint64_t MissionRsKernelReadyAddr(void* devCtx, uint32_t mission)
{
    return reinterpret_cast<uint64_t>(devCtx) + offsetof(ProgressDeviceCtx, missions) +
           static_cast<uint64_t>(mission) * sizeof(MissionState) + offsetof(MissionState, rsKernelReady);
}

static inline uint32_t MissionWorkItemCount(int nRanks, int rankId)
{
    const uint32_t safeRanks = (nRanks > 0) ? static_cast<uint32_t>(nRanks) : 1;
    return CcuOwnerGroupCount(static_cast<uint32_t>(rankId), G_NUM_TILES, safeRanks);
}

// ============================================================================
// CCU state per rank — channels, thread handles, kernel handles, progress CKE info
// ============================================================================
struct CcuState {
    ThreadHandle threadHandle{0};
    std::vector<ChannelHandle> channels;
    aclrtStream ccuStream{nullptr};
    aclrtStream aivStream{nullptr};

    CcuInsHandle ccuIns{0};
    CcuKernelHandle rsHandles[kMaxCcuMissions]{};
    uint32_t missionCount{0};
    bool kernelRegistered{false};

    // One-shot sequential baseline (same CCU instance, different handle; no progress CKE).
    CcuKernelHandle seqOneShotHandle{0};
    bool seqKernelRegistered{false};

    // Cached register-time args for HcommCcuKernelLaunch packing.
    CcuFusedReduceBroadcastKernelArg rsArgs[kMaxCcuMissions]{};
    CcuFusedReduceBroadcastKernelArg seqArg{};

    uint64_t rsProgressVA[kMaxCcuMissions][2]{};
    uint32_t rsProgressMask[kMaxCcuMissions][2]{};
    uint64_t rsGateVA[kMaxCcuMissions]{};
    uint32_t rsGateMask[kMaxCcuMissions]{};
    bool ckeResolved{false}; // gate/progress VA cached after first resolve
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
static CcuLaunchContext g_ccuLaunchCtx{};

static bool SyncCcuStreams(const CcuState& ccu, int rankId, const char* phase)
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

static bool SetupCcuChannels(HcclComm comm, int rankId, int nRanks, std::vector<ChannelHandle>& channels)
{
    std::vector<HcclChannelDesc> requests;
    for (int peer = 0; peer < nRanks; ++peer) {
        if (peer == rankId)
            continue;

        // Probe multiple netLayers — UBC_CTP may appear at layer 0 or 1 depending on topology
        bool found = false;
        for (uint32_t netLayer = 0; netLayer < 3 && !found; ++netLayer) {
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
                    std::cerr << "[CCU-AR] rank=" << rankId << ": layer=" << netLayer << " peer=" << peer << " link["
                              << i << "] proto=" << static_cast<int>(proto) << std::endl;
                }
                if (proto != COMM_PROTOCOL_UBC_CTP)
                    continue;
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
                              << " at layer=" << netLayer
                              << " locProto=" << static_cast<int>(desc.localEndpoint.protocol)
                              << " rmtProto=" << static_cast<int>(desc.remoteEndpoint.protocol) << std::endl;
                }
                requests.push_back(desc);
                found = true;
                break;
            }
        }
        if (!found) {
            std::cerr << "[CCU-AR] rank=" << rankId << ": no UBC_CTP link to peer=" << peer << " at any netLayer"
                      << std::endl;
            return false;
        }
    }
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

// Same as ST/mesh: Published DieId/Id → rtGetDevResAddress → AIV poke VA.
static uint64_t ResolveOneCkeVA(uint32_t dieId, uint32_t ckeId)
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

// Gate CKE via dedicated registry Publish/TryGet (not a progress[] slot).
static bool ResolveCcuGate(int rankId, CcuState& ccu)
{
    if (ccu.ckeResolved && ccu.rsGateVA[0] != 0)
        return true;

    pto::comm::ccu::CcuGateDescriptor desc{};
    for (int retry = 0; retry < 200; ++retry) {
        if (pto::comm::ccu::TryGet(static_cast<uint32_t>(rankId), desc))
            goto found;
        usleep(10000);
    }
    std::cerr << "[CCU-AR] rank=" << rankId << ": gate TryGet failed after retries" << std::endl;
    return false;

found:
    // K=1: one gate descriptor per rank (registry has a single gate slot).
    uint64_t addr = ResolveOneCkeVA(desc.dieId, desc.ckeId);
    if (addr == 0) {
        std::cerr << "[CCU-AR] rank=" << rankId << ": fused gate VA resolution failed (die=" << desc.dieId
                  << " cke=" << desc.ckeId << ")" << std::endl;
        return false;
    }
    ccu.rsGateVA[0] = addr;
    ccu.rsGateMask[0] = desc.mask;
    if (VerboseLog()) {
        std::cerr << "[CCU-AR] rank=" << rankId << ": gate die=" << desc.dieId << " cke=" << desc.ckeId << " mask=0x"
                  << std::hex << desc.mask << " VA=0x" << addr << std::dec << std::endl;
    }
    return true;
}

// Edge-triggered gate poke (no host D2H). A couple of poke+sync rounds cover CCU
// arriving at WaitEvent slightly later; AIV still polls rsKernelReady.
static bool TriggerCcuGate(DeviceBuffers& /*buf*/, CcuState& ccu, int rankId)
{
    if (ccu.rsGateVA[0] == 0) {
        std::cerr << "[CCU-AR] rank=" << rankId << ": TriggerCcuGate called before ResolveCcuGate" << std::endl;
        return false;
    }
    constexpr int kPokes = 2;
    for (int i = 0; i < kPokes; ++i) {
        if (launchCcuGemmArGateTrigger(ccu.aivStream, ccu.rsGateVA[0], ccu.rsGateMask[0]) != 0) {
            std::cerr << "[CCU-AR] rank=" << rankId << ": fused gate trigger failed" << std::endl;
            return false;
        }
        aclError ar = aclrtSynchronizeStream(ccu.aivStream);
        if (ar != ACL_SUCCESS) {
            std::cerr << "[CCU-AR] rank=" << rankId << ": gate trigger aivStream sync failed: " << static_cast<int>(ar)
                      << std::endl;
            return false;
        }
    }
    return true;
}

// Fused progress CKEs: registry slots A/B (ping-pong).
static constexpr uint32_t kProgressCkeSlotA = 0;
static constexpr uint32_t kProgressCkeSlotB = 1;
static constexpr uint32_t kProgressCkeSlotCount = 2;

static bool ResolveCcuProgress(int rankId, CcuState& ccu)
{
    if (ccu.ckeResolved && ccu.rsProgressVA[0][0] != 0)
        return true;

    std::vector<pto::comm::ccu::CcuGateDescriptor> progDescs;
    for (int retry = 0; retry < 200; ++retry) {
        if (pto::comm::ccu::TryGetProgress(static_cast<uint32_t>(rankId), progDescs) &&
            progDescs.size() >= kProgressCkeSlotCount)
            goto found;
        usleep(10000);
    }
    std::cerr << "[CCU-AR] rank=" << rankId << ": progress TryGet failed after retries (need " << kProgressCkeSlotCount
              << " descriptors)" << std::endl;
    return false;

found:
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

static bool WriteProgressCtxToDevice(DeviceBuffers& buf)
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
static uint64_t SeqOneShotPayloadBytes(int rankId, int nRanks)
{
    const uint32_t safeRanks = (nRanks > 0) ? static_cast<uint32_t>(nRanks) : 1;
    const uint32_t ownerTiles = CcuOwnerTileCount(static_cast<uint32_t>(rankId), G_NUM_TILES, safeRanks);
    return static_cast<uint64_t>(ownerTiles) * G_TILE_BYTES;
}

static void FillFusedKernelArg(
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

static bool RegisterCcuGemmArOnce(HcclComm comm, CcuState& ccu, int rankId, int nRanks, const DeviceBuffers& buf)
{
    if (ccu.kernelRegistered)
        return true;

    // Match mesh/ST: pin CCU IO die (env HCCL_PTO_GATE_DIE_ID, default 1 on A5).
    uint32_t kDieId = 1;
    if (const char* env = std::getenv("HCCL_PTO_GATE_DIE_ID"); env != nullptr && env[0] != '\0') {
        const int v = std::atoi(env);
        if (v >= 0)
            kDieId = static_cast<uint32_t>(v);
    }
    if (QueryPrimaryCcuIns(comm, &ccu.ccuIns) != CcuResult::CCU_SUCCESS) {
        std::cerr << "[CCU-AR] rank=" << rankId << ": QueryPrimaryCcuIns failed" << std::endl;
        return false;
    }

    // Kernel 1: sequential one-shot baseline (full owner shard, no progress CKE)
    {
        FillFusedKernelArg(ccu.seqArg, ccu, rankId, nRanks, buf, 0, true);
        CcuResult cret = HcommCcuKernelRegisterStart(ccu.ccuIns);
        if (cret != CcuResult::CCU_SUCCESS) {
            std::cerr << "[CCU-AR] rank=" << rankId << ": seq RegisterStart failed: " << static_cast<int>(cret)
                      << std::endl;
        } else {
            CcuKernelHandle seqHandle = 0;
            cret = RegisterFusedReduceBroadcastKernel(ccu.ccuIns, kDieId, ccu.seqArg, &seqHandle);
            CcuResult endRet = HcommCcuKernelRegisterEnd(ccu.ccuIns);
            if (cret != CcuResult::CCU_SUCCESS) {
                std::cerr << "[CCU-AR] rank=" << rankId << ": seq one-shot register failed: " << static_cast<int>(cret)
                          << std::endl;
            } else if (endRet != CcuResult::CCU_SUCCESS) {
                std::cerr << "[CCU-AR] rank=" << rankId << ": seq RegisterEnd failed: " << static_cast<int>(endRet)
                          << std::endl;
            } else {
                ccu.seqOneShotHandle = seqHandle;
                ccu.seqKernelRegistered = true;
            }
        }
    }

    // Kernel 2: pipelined fused Reduce+Broadcast (per-group progress CKE)
    {
        CcuResult cret = HcommCcuKernelRegisterStart(ccu.ccuIns);
        if (cret != CcuResult::CCU_SUCCESS) {
            std::cerr << "[CCU-AR] rank=" << rankId << ": fused RegisterStart failed: " << static_cast<int>(cret)
                      << std::endl;
            return false;
        }
        for (uint32_t m = 0; m < kCcuMissionParallel; ++m) {
            FillFusedKernelArg(ccu.rsArgs[m], ccu, rankId, nRanks, buf, m, false);
            CcuKernelHandle handle = 0;
            cret = RegisterFusedReduceBroadcastKernel(ccu.ccuIns, kDieId, ccu.rsArgs[m], &handle);
            if (cret != CcuResult::CCU_SUCCESS) {
                std::cerr << "[CCU-AR] rank=" << rankId << ": fused register(mission=" << m
                          << ") failed: " << static_cast<int>(cret) << std::endl;
                (void)HcommCcuKernelRegisterEnd(ccu.ccuIns);
                return false;
            }
            ccu.rsHandles[m] = handle;
        }
        ccu.missionCount = kCcuMissionParallel;
        cret = HcommCcuKernelRegisterEnd(ccu.ccuIns);
        if (cret != CcuResult::CCU_SUCCESS) {
            std::cerr << "[CCU-AR] rank=" << rankId << ": fused RegisterEnd failed: " << static_cast<int>(cret)
                      << std::endl;
            return false;
        }
        // Translate (physical DieId/Id) runs inside RegisterEnd — Publish only after that.
        cret = PublishStashedCkeAfterRegisterEnd();
        if (cret != CcuResult::CCU_SUCCESS) {
            std::cerr << "[CCU-AR] rank=" << rankId
                      << ": post-RegisterEnd CKE Publish failed: " << static_cast<int>(cret) << std::endl;
            return false;
        }
    }

    ccu.kernelRegistered = true;
    if (VerboseLog()) {
        const uint64_t payloadBytes = G_GROUP_BYTES;
        const uint32_t loopCount = pto::comm::ccu::CalcLoopCount(payloadBytes);
        std::cerr << "[CCU-AR] rank=" << rankId << ": registered " << ccu.missionCount
                  << " fused Reduce+Broadcast kernel(s), numTiles=" << G_NUM_TILES << " payloadBytes=" << payloadBytes
                  << " groupTiles=" << G_COMM_GROUP_TILES << " items=" << MissionWorkItemCount(nRanks, rankId)
                  << " loopCount=" << loopCount << (ccu.seqKernelRegistered ? " + seq one-shot baseline" : "")
                  << std::endl;
        if (ccu.seqKernelRegistered) {
            std::cerr << "[CCU-AR] rank=" << rankId
                      << ": seq one-shot payloadBytes=" << SeqOneShotPayloadBytes(rankId, nRanks)
                      << " (owner tiles, independent of group_tiles)" << std::endl;
        }
    }
    return true;
}

static bool LaunchCcuGemmAr(
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
        CcuFusedReduceBroadcastTaskArg taskArg{};
        taskArg.inputAddr = allInputVA[rankId];
        taskArg.outputAddr = allOutputVA[rankId];
        taskArg.length = payloadBytes;
        taskArg.SetPeerAddrs(static_cast<uint32_t>(nRanks), allInputVA, allOutputVA, allTokens);

        CcuResult cret = LaunchFusedReduceBroadcastKernel(ccu.threadHandle, ccu.rsHandles[m], ccu.rsArgs[m], taskArg);
        if (cret != CcuResult::CCU_SUCCESS) {
            std::cerr << "[CCU-AR] rank=" << rankId << ": fused Launch(mission=" << m
                      << ") failed: " << static_cast<int>(cret) << std::endl;
            return false;
        }
    }
    return true;
}

static void StoreCcuLaunchContext(
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

static bool LaunchSeqOneShotCcuGemmAr(
    CcuState& ccu, int rankId, int nRanks, const uint64_t* allInputVA, const uint64_t* allOutputVA,
    const uint64_t* allTokens, HcclComm /*comm*/)
{
    if (!ccu.seqKernelRegistered) {
        std::cerr << "[CCU-AR] rank=" << rankId << ": seq one-shot kernel not registered" << std::endl;
        return false;
    }
    CcuFusedReduceBroadcastTaskArg taskArg{};
    taskArg.inputAddr = allInputVA[rankId];
    taskArg.outputAddr = allOutputVA[rankId];
    taskArg.length = SeqOneShotPayloadBytes(rankId, nRanks);
    taskArg.SetPeerAddrs(static_cast<uint32_t>(nRanks), allInputVA, allOutputVA, allTokens);

    CcuResult cret = LaunchFusedReduceBroadcastKernel(ccu.threadHandle, ccu.seqOneShotHandle, ccu.seqArg, taskArg);
    if (cret != CcuResult::CCU_SUCCESS) {
        std::cerr << "[CCU-AR] rank=" << rankId << ": seq one-shot Launch failed: " << static_cast<int>(cret)
                  << std::endl;
        return false;
    }
    return true;
}

static bool LaunchStoredSeqOneShotCcuGemmAr(CcuState& ccu)
{
    if (!g_ccuLaunchCtx.valid)
        return false;
    return LaunchSeqOneShotCcuGemmAr(
        ccu, g_ccuLaunchCtx.rankId, g_ccuLaunchCtx.nRanks, g_ccuLaunchCtx.allInputVA, g_ccuLaunchCtx.allOutputVA,
        g_ccuLaunchCtx.allTokens, g_ccuLaunchCtx.comm);
}

// Full CCU preparation: sync prior pass, refresh ctx, launch CCU, resolve gate/progress CKEs.
// CKE descriptors are published once after RegisterEnd (Translate) and remain valid across launches.
static bool PrepareCcuKernel(
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

// ============================================================================
// Official MC2 tiling layout used by CANN HCCL tiling APIs (same as gemm_ar).
// ============================================================================
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

// ============================================================================
// Host-side helpers
// ============================================================================
inline void HcclHostBarrier(HcclComm comm, aclrtStream stream)
{
    HcclBarrier(comm, stream);
    aclrtSynchronizeStream(stream);
}

inline void* WindowAlloc(uint64_t windowBase, size_t& offset, size_t bytes)
{
    void* ptr = reinterpret_cast<void*>(windowBase + offset);
    offset += bytes;
    return ptr;
}

// ============================================================================
// Helpers
// ============================================================================

struct PerfStats {
    double avg;
    double med;
    double min_val;
    double max_val;
    double std_dev;
};

// Headline metric is avg (matches ccu_gemm_ar_v3 / HCCL all_reduce_test);
// median is reported in brackets.
static PerfStats calcStats(const std::vector<double>& times)
{
    if (times.empty())
        return {0.0, 0.0, 0.0, 0.0, 0.0};

    double sum = 0.0;
    double mn = times[0];
    double mx = times[0];
    for (double t : times) {
        sum += t;
        if (t < mn)
            mn = t;
        if (t > mx)
            mx = t;
    }
    double avg = sum / times.size();
    double var = 0.0;
    for (double t : times)
        var += (t - avg) * (t - avg);

    std::vector<double> sorted = times;
    std::sort(sorted.begin(), sorted.end());
    size_t n = sorted.size();
    double med = (n % 2 == 0) ? (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0 : sorted[n / 2];

    return {avg, med, mn, mx, std::sqrt(var / times.size())};
}

// ============================================================================
// HCCL context initialization:
//   - A5/fullmesh path uses the device context returned directly by HCCL.
//   - non-mesh topologies fall back to a host-side compatibility extraction.
// ============================================================================
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

// ============================================================================
// Per-rank execution: sub-functions
// ============================================================================

static float halfToFloat(uint16_t h)
{
    uint32_t sign = ((uint32_t)h & 0x8000) << 16;
    uint32_t exp = ((uint32_t)h >> 10) & 0x1F;
    uint32_t mant = (uint32_t)h & 0x03FF;
    if (exp == 0) {
        if (mant == 0) {
            union {
                uint32_t u;
                float f;
            } r;
            r.u = sign;
            return r.f;
        }
        while (!(mant & 0x0400)) {
            mant <<= 1;
            exp--;
        }
        exp++;
        mant &= ~0x0400;
    } else if (exp == 31) {
        union {
            uint32_t u;
            float f;
        } r;
        r.u = sign | 0x7F800000 | (mant << 13);
        return r.f;
    }
    exp = exp + (127 - 15);
    uint32_t bits = sign | (exp << 23) | (mant << 13);
    union {
        uint32_t u;
        float f;
    } r;
    r.u = bits;
    return r.f;
}

static bool VerifyOutput(const uint16_t* output_fp16, const float* golden, int nRanks)
{
    const float atol = 1.0f;
    const float rtol = 0.01f;
    const size_t valid_elements = (size_t)G_ORIG_M * G_ORIG_N;
    float max_diff = 0.0f;
    float max_diff_ratio = 0.0f;
    size_t err_count = 0;
    const size_t err_threshold = static_cast<size_t>(valid_elements * rtol);
    constexpr size_t MAX_ERR_PRINT = 4;

    for (size_t row = 0; row < G_ORIG_M; ++row) {
        for (size_t col = 0; col < G_ORIG_N; ++col) {
            size_t idx = row * G_N + col;
            float exp_val = golden[idx];
            float act_val = halfToFloat(output_fp16[idx]);
            float diff = std::abs(exp_val - act_val);
            float rel = (std::abs(exp_val) > 1e-3f) ? (diff / std::abs(exp_val)) : 0.0f;
            if (diff > max_diff)
                max_diff = diff;
            if (rel > max_diff_ratio)
                max_diff_ratio = rel;
            if (diff > atol + rtol * std::abs(exp_val)) {
                err_count++;
                if (err_count <= MAX_ERR_PRINT) {
                    uint32_t ti = (uint32_t)(row / G_BASE_M);
                    uint32_t tj = (uint32_t)(col / G_BASE_N);
                    int tile_idx = ti * G_N_TILES + tj;
                    int owner = (nRanks > 0) ? (tile_idx % nRanks) : 0;
                    printf(
                        "  ERR[%zu] row=%zu col=%zu tile=(%u,%u) tile_idx=%d owner=%d "
                        "exp=%.4f act=%.4f diff=%.4f rel=%.4f\n",
                        err_count, row, col, ti, tj, tile_idx, owner, exp_val, act_val, diff, rel);
                }
            }
        }
    }

    bool ok = (err_count <= err_threshold);
    std::cout << "[VERIFY] valid_region=" << G_ORIG_M << "x" << G_ORIG_N << " max_diff=" << max_diff
              << " max_ratio=" << max_diff_ratio << " err=" << err_count << "/" << err_threshold << " -> "
              << (ok ? "PASS" : "FAIL") << std::endl;
    return ok;
}

static void DumpVerifySamples(const DeviceBuffers& buf, int rank_id, const float* golden)
{
    constexpr size_t kSample = 16;
    uint16_t packed[kSample]{};
    uint16_t row[kSample]{};
    aclrtMemcpy(packed, sizeof(packed), buf.reduced_output, sizeof(packed), ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(row, sizeof(row), buf.row_output, sizeof(row), ACL_MEMCPY_DEVICE_TO_HOST);

    std::cerr << "[VERIFY-DIAG] rank=" << rank_id << " tile0 row0 sample";
    for (size_t i = 0; i < kSample; ++i) {
        std::cerr << " c" << i << "{gold=" << golden[i] << ",packed=" << halfToFloat(packed[i])
                  << ",row=" << halfToFloat(row[i]) << "}";
    }
    std::cerr << std::endl;
}

static void DumpReadyCountersAfterCcu(const DeviceBuffers& buf, int rank_id)
{
    constexpr uint32_t kSampleLocalIds[] = {0, 1, 2, 3, 126, 127, 128, 129, 254, 255, 256, 257};
    std::vector<int32_t> signalHost(G_SIGNAL_MATRIX_SLOTS, 0);
    aclrtMemcpy(
        signalHost.data(), signalHost.size() * sizeof(int32_t), buf.signal_matrix, signalHost.size() * sizeof(int32_t),
        ACL_MEMCPY_DEVICE_TO_HOST);
    std::cerr << "[VERIFY-DIAG] rank=" << rank_id << " readyCounters";
    for (uint32_t lid : kSampleLocalIds) {
        if (G_SIGNAL_SUBTILE_READY_OFFSET + lid < signalHost.size())
            std::cerr << " s[" << lid << "]=" << signalHost[G_SIGNAL_SUBTILE_READY_OFFSET + lid];
    }
    std::cerr << std::endl;
}

static void PrintTimingDetails(
    const PerfStats& comp_s, const PerfStats& seq_s, const PerfStats& pipe_s, const PerfStats& seq_comp_s,
    const PerfStats& seq_comm_s, const PerfStats& pipe_comp_s, const PerfStats& pipe_comm_s, double flops_per_rank,
    double flops_total, double rs_bytes, double ag_bytes)
{
    auto gflops = [](double flops, double us) { return (us > 0) ? (flops / (us * 1e-6) / 1e9) : 0.0; };
    auto bw_gbs = [&](double us) {
        return (us > 0) ? ((rs_bytes + ag_bytes) / (us * 1e-6) / (1024.0 * 1024.0 * 1024.0)) : 0.0;
    };

    std::cout << "\n  (avg over iterations; brackets: med=median, std=stddev vs avg)" << std::endl;
    std::cout << "  Compute-only:   " << std::setprecision(1) << comp_s.avg << " us"
              << "  (" << std::setprecision(0) << gflops(flops_per_rank, comp_s.avg) << " GFLOPS)"
              << "  [med=" << std::setprecision(1) << comp_s.med << ", std=" << comp_s.std_dev << "]" << std::endl;
    std::cout << "\n  Sequential:     " << std::setprecision(1) << seq_s.avg << " us"
              << "  [med=" << seq_s.med << ", std=" << seq_s.std_dev << "]  (one-shot fused RS→AG)" << std::endl;
    std::cout << "    compute:      " << seq_comp_s.avg << " us"
              << "  (" << std::setprecision(0) << gflops(flops_per_rank, seq_comp_s.avg) << " GFLOPS)" << std::endl;
    std::cout << "    comm:         " << std::setprecision(1) << seq_comm_s.avg << " us"
              << "  (" << std::setprecision(1) << bw_gbs(seq_comm_s.avg) << " GB/s)"
              << "  [med=" << seq_comm_s.med << ", std=" << seq_comm_s.std_dev << "]" << std::endl;
    std::cout << "\n  Pipelined:      " << std::setprecision(1) << pipe_s.avg << " us (end-to-end)"
              << "  [med=" << pipe_s.med << ", std=" << pipe_s.std_dev << "]" << std::endl;
    std::cout << "    compute done: " << pipe_comp_s.avg << " us (kernel)"
              << "  (" << std::setprecision(0) << gflops(flops_per_rank, pipe_comp_s.avg) << " GFLOPS, "
              << std::setprecision(1)
              << (gflops(flops_per_rank, pipe_comp_s.avg) / gflops(flops_per_rank, comp_s.avg) * 100.0) << "% of pure)"
              << std::endl;
    std::cout << "    comm done:    " << std::setprecision(1) << pipe_comm_s.avg << " us"
              << "  (" << std::setprecision(1) << bw_gbs(pipe_comm_s.avg) << " GB/s)" << std::endl;

    double speedup = (pipe_s.avg > 0) ? (seq_s.avg / pipe_s.avg) : 0.0;
    // Overlap uses medians so a few sequential/pipe outliers do not inflate overlap >100%.
    double overlap_time = (seq_comp_s.med + seq_comm_s.med) - pipe_s.med;
    double overlap_eff = (overlap_time > 0) ? (overlap_time / std::min(seq_comp_s.med, seq_comm_s.med) * 100.0) : 0.0;

    std::cout << "\n  Speedup:        " << std::setprecision(3) << speedup << "x" << std::endl;
    std::cout << "  Time saved:     " << std::setprecision(1) << (seq_s.avg - pipe_s.avg) << " us"
              << " (" << std::setprecision(1)
              << ((seq_s.avg > 0) ? ((seq_s.avg - pipe_s.avg) / seq_s.avg * 100.0) : 0.0) << "%)" << std::endl;
    std::cout << "  Overlap eff:    " << std::setprecision(1) << overlap_eff << "%" << std::endl;
    std::cout << "  Throughput:     " << std::setprecision(0) << gflops(flops_total, pipe_s.avg) << " GFLOPS (total)"
              << std::endl;
    std::cout << "================================================================\n" << std::endl;
}

static bool ReadSchedTimingFromDevice(const DeviceBuffers& buf, SchedTiming& rsTiming)
{
    if (buf.progressCtx_dev == nullptr)
        return false;

    const uint8_t* base = reinterpret_cast<const uint8_t*>(buf.progressCtx_dev);
    aclError rsRet = aclrtMemcpy(
        &rsTiming, sizeof(rsTiming), base + offsetof(ProgressDeviceCtx, header) + offsetof(ProgressCtx, rsSchedTiming),
        sizeof(rsTiming), ACL_MEMCPY_DEVICE_TO_HOST);
    return rsRet == ACL_SUCCESS;
}

static void PrintSchedTimingField(const char* label, uint64_t cycles, uint64_t totalCycles)
{
    const double us = SchedCyclesToUs(cycles);
    const double pct =
        (totalCycles > 0) ? (static_cast<double>(cycles) * 100.0 / static_cast<double>(totalCycles)) : 0.0;
    std::cout << "      " << std::setw(18) << std::left << label << std::right << std::setw(12) << std::setprecision(1)
              << us << " us";
    if (totalCycles > 0)
        std::cout << "  (" << std::setprecision(1) << pct << "%)";
    std::cout << std::endl;
}

static void PrintSchedTimingBlock(const char* label, const SchedTiming& timing)
{
    const uint64_t total = timing.totalCycles;
    std::cout << "    [" << label << "] total=" << SchedCyclesToUs(total) << " us  triggerCount=" << timing.triggerCount
              << std::endl;
    // Default progress_kernel: local GD + peer groupReady TNOTIFY + BP/ring.
    PrintSchedTimingField("kernelReady", timing.kernelReadyCycles, total);
    PrintSchedTimingField("waitLocalGD", timing.groupDoneCycles, total);
    PrintSchedTimingField("waitPeerReady", timing.notifyCycles, total);
    PrintSchedTimingField("waitRing(CKE)", timing.idleSpinCycles, total);
    PrintSchedTimingField("waitBackpressure", timing.backpressureCycles, total);
    PrintSchedTimingField("finalWaitItems", timing.finalWaitCycles, total);

    const uint64_t accounted = timing.kernelReadyCycles + timing.groupDoneCycles + timing.idleSpinCycles +
                               timing.backpressureCycles + timing.finalWaitCycles + timing.notifyCycles;
    PrintSchedTimingField("accountedSum", accounted, total);
}

static void PrintSchedTimingReport(int rank_id, double pipe_aiv_wall_us)
{
    if (!SchedTimingEnabled() || !g_lastPipeSchedTimingValid)
        return;

    // All ranks print (stderr) so 2/4-rank critical-path owner can be compared.
    std::cerr << "\n  --- Progress wait breakdown (PTO_CCU_GEMM_AR_SCHED_TIMING=1, last pipelined iter, rank" << rank_id
              << ") ---" << std::endl;
    if (rank_id == 0) {
        std::cerr << "  SYS_CNT tick: " << SchedTickNs() << " ns (override via PTO_CCU_GEMM_AR_SCHED_TICK_NS)"
                  << std::endl;
        std::cerr << "  Default path: localGD → peer TNOTIFY(groupReady) → owner (peerReady||BP||ring) → fire; "
                     "peerReady samples → waitPeer; BP/ring attribute to first still-pending"
                  << std::endl;
    }
    const uint64_t total = g_lastPipeRsTiming.totalCycles;
    std::cerr << "    [progress_kernel] total=" << SchedCyclesToUs(total)
              << " us  triggerCount=" << g_lastPipeRsTiming.triggerCount
              << "  waitLocalGD=" << SchedCyclesToUs(g_lastPipeRsTiming.groupDoneCycles)
              << " us  waitPeer=" << SchedCyclesToUs(g_lastPipeRsTiming.notifyCycles)
              << " us  waitBP=" << SchedCyclesToUs(g_lastPipeRsTiming.backpressureCycles)
              << " us  finalWait=" << SchedCyclesToUs(g_lastPipeRsTiming.finalWaitCycles)
              << " us  vs aiv_wall=" << std::setprecision(1) << pipe_aiv_wall_us << " us" << std::endl;
    std::cerr.flush();
}

static void PrintCommDiagnosticReport(
    int rank_id, const std::vector<CommPhaseTiming>& seq_comm_diag, const std::vector<CommPhaseTiming>& pipe_comm_diag)
{
    if (!CommDiagEnabled() || rank_id != 0 || seq_comm_diag.empty() || pipe_comm_diag.empty())
        return;

    auto printPhaseStats = [](const char* label, const std::vector<CommPhaseTiming>& records) {
        const PerfStats comm_wall = calcStats(ExtractCommField(records, &CommPhaseTiming::comm_wall_us));
        const PerfStats aiv_wall = calcStats(ExtractCommField(records, &CommPhaseTiming::aiv_wall_us));
        const PerfStats ccu_wall = calcStats(ExtractCommField(records, &CommPhaseTiming::ccu_wall_us));
        const std::vector<double> aiv_event_vals = ExtractValidEventField(records, &CommPhaseTiming::aiv_event_us);
        const std::vector<double> comm_event_vals = ExtractValidEventField(records, &CommPhaseTiming::comm_event_us);
        const uint32_t outliers = CountCommOutliers(
            ExtractCommField(records, &CommPhaseTiming::comm_wall_us), comm_wall.med, comm_wall.std_dev);

        std::cout << "  [" << label << "] comm wall med=" << std::setprecision(1) << comm_wall.med << " us"
                  << " [" << comm_wall.avg << " ± " << comm_wall.std_dev << ", min=" << comm_wall.min_val
                  << ", max=" << comm_wall.max_val << "]" << std::endl;
        std::cout << "    aiv wall:  med=" << aiv_wall.med << " us (" << std::setprecision(1)
                  << (comm_wall.med > 0.0 ? (aiv_wall.med / comm_wall.med * 100.0) : 0.0)
                  << "% of comm)  ccu wall: med=" << ccu_wall.med << " us" << std::endl;
        if (!aiv_event_vals.empty()) {
            const PerfStats aiv_event = calcStats(aiv_event_vals);
            const PerfStats comm_event = comm_event_vals.empty() ? aiv_event : calcStats(comm_event_vals);
            std::cout << "    aiv event: med=" << aiv_event.med << " us  comm~event: med=" << comm_event.med << " us"
                      << "  (aivStream only; ccu tail = wall)" << std::endl;
        }
        std::cout << "    outliers:  " << outliers << "/" << records.size() << "  (>1.25x median or > median+2σ)"
                  << std::endl;
    };

    std::cout << "\n  --- Comm timing breakdown (aiv scheduler vs ccu stream sync) ---" << std::endl;
    std::cout << "  Note: ccu wall is sync(ccuStream) after scheduler done; ~0 us means CCU"
              << " finished before/with scheduler exit." << std::endl;
    printPhaseStats("sequential comm", seq_comm_diag);
    printPhaseStats("pipelined comm", pipe_comm_diag);

    std::cout << "\n  --- Per-iteration comm samples ---" << std::endl;
    std::cout << "  iter  seq_wall  seq_aiv  seq_ccu  seq_aiv_ev  pipe_wall pipe_aiv pipe_ccu pipe_aiv_ev" << std::endl;
    const size_t n = std::max(seq_comm_diag.size(), pipe_comm_diag.size());
    for (size_t i = 0; i < n; ++i) {
        std::cout << "  " << std::setw(4) << i;
        if (i < seq_comm_diag.size()) {
            const CommPhaseTiming& s = seq_comm_diag[i];
            std::cout << "  " << std::setw(8) << std::setprecision(1) << s.comm_wall_us << "  " << std::setw(8)
                      << s.aiv_wall_us << "  " << std::setw(8) << s.ccu_wall_us << "  " << std::setw(9)
                      << FormatEventUs(s.aiv_event_us);
        } else {
            std::cout << "  " << std::setw(8) << "-" << "  " << std::setw(8) << "-" << "  " << std::setw(8) << "-"
                      << "  " << std::setw(9) << "-";
        }
        if (i < pipe_comm_diag.size()) {
            const CommPhaseTiming& p = pipe_comm_diag[i];
            std::cout << "  " << std::setw(9) << p.comm_wall_us << "  " << std::setw(8) << p.aiv_wall_us << "  "
                      << std::setw(8) << p.ccu_wall_us << "  " << std::setw(9) << FormatEventUs(p.aiv_event_us);
        }
        std::cout << std::endl;
    }
}

static void PrintPerfReport(
    bool is_ok, int n_ranks, const std::vector<double>& compute_times_us,
    const std::vector<double>& sequential_times_us, const std::vector<double>& pipelined_times_us,
    const std::vector<double>& seq_compute_us, const std::vector<double>& seq_comm_us,
    const std::vector<double>& pipe_compute_us, const std::vector<double>& pipe_comm_us, int rank_id,
    const std::vector<CommPhaseTiming>& seq_comm_diag, const std::vector<CommPhaseTiming>& pipe_comm_diag)
{
    PerfStats comp_s = calcStats(compute_times_us);
    PerfStats seq_s = calcStats(sequential_times_us);
    PerfStats pipe_s = calcStats(pipelined_times_us);
    PerfStats seq_comp_s = calcStats(seq_compute_us);
    PerfStats seq_comm_s = calcStats(seq_comm_us);
    PerfStats pipe_comp_s = calcStats(pipe_compute_us);
    PerfStats pipe_comm_s = calcStats(pipe_comm_us);

    double flops_per_rank = 2.0 * G_ORIG_M * (double)G_K * G_ORIG_N;
    double flops_total = flops_per_rank * ((n_ranks > 0) ? n_ranks : 1);

    size_t tileBytes = static_cast<size_t>(G_BASE_M) * G_BASE_N * sizeof(uint16_t);
    int tiles_per_owner = (n_ranks > 0) ? ((G_NUM_TILES + n_ranks - 1) / n_ranks) : G_NUM_TILES;
    double rs_bytes = static_cast<double>(G_NUM_TILES - tiles_per_owner) * tileBytes;
    int safe_remotes = (n_ranks > 1) ? (n_ranks - 1) : 0;
    double ag_bytes = static_cast<double>(tiles_per_owner) * safe_remotes * tileBytes;
    double data_gb = (rs_bytes + ag_bytes) / (1024.0 * 1024.0 * 1024.0);

    std::cout << std::fixed << std::setprecision(1);
    std::cout << "\n================================================================" << std::endl;
    std::cout << (is_ok ? "[SUCCESS]" : "[FAILED]") << " CCU GEMM AllReduce A5 FP16 (HCCL)" << std::endl;
    std::cout << "  M=" << G_ORIG_M << " K=" << G_K << " N=" << G_ORIG_N;
    if (G_M != G_ORIG_M || G_N != G_ORIG_N)
        std::cout << "  (padded " << G_M << "x" << G_K << "x" << G_N << ")";
    std::cout << "  ranks=" << n_ranks << "  compute_blocks=" << COMPUTE_BLOCK_NUM
              << "  ccu_group_tiles=" << G_COMM_GROUP_TILES << std::endl;
    std::cout << "  tiles=" << G_NUM_TILES << " (" << G_M_TILES << "x" << G_N_TILES << ")"
              << "  comm_data=" << std::setprecision(3) << data_gb << " GB/rank" << std::endl;

    PrintTimingDetails(
        comp_s, seq_s, pipe_s, seq_comp_s, seq_comm_s, pipe_comp_s, pipe_comm_s, flops_per_rank, flops_total, rs_bytes,
        ag_bytes);
    PrintCommDiagnosticReport(rank_id, seq_comm_diag, pipe_comm_diag);
}

// Host-side snapshot of a PerBlockQueue header (count/tail only).
struct QueueCountSnap {
    int32_t count{0};
    int32_t tail{0};
};

static bool DumpMissionEnabled()
{
    static const bool enabled = EnvFlagEnabled("PTO_CCU_GEMM_AR_DUMP_MISSION");
    return enabled;
}

static void DumpMissionState(int rank_id, uint32_t mission, const MissionState& ms)
{
    if (!DumpMissionEnabled())
        return;
    std::cerr << "[CCU-MISSION] rank=" << rank_id << " mission=" << mission << " rsItemsDone=" << ms.rsItemsDone
              << " rsKernelReady=" << ms.rsKernelReady << std::endl;
}

// Non-blocking stream probe: ACL_SUCCESS => already idle.
static bool StreamLooksIdle(aclrtStream stream)
{
    if (stream == nullptr)
        return true;
    return aclrtSynchronizeStreamWithTimeout(stream, 0) == ACL_SUCCESS;
}

static void ReadReadyQueueCounts(const DeviceBuffers& buf, std::vector<QueueCountSnap>& out)
{
    out.clear();
    if (buf.readyQueue_dev == nullptr || buf.readyQueueSize == 0)
        return;

    MultiBlockQueueSet qsetHost{};
    aclrtMemcpy(&qsetHost, sizeof(qsetHost), buf.readyQueue_dev, sizeof(qsetHost), ACL_MEMCPY_DEVICE_TO_HOST);
    if (qsetHost.num_blocks <= 0)
        return;

    out.resize(static_cast<size_t>(qsetHost.num_blocks));
    for (int b = 0; b < qsetHost.num_blocks; ++b) {
        PerBlockQueue pqHost{};
        aclrtMemcpy(
            &pqHost, sizeof(pqHost), reinterpret_cast<uint8_t*>(buf.readyQueue_dev) + qsetHost.queue_offsets[b],
            sizeof(pqHost), ACL_MEMCPY_DEVICE_TO_HOST);
        out[static_cast<size_t>(b)].count = pqHost.count;
        out[static_cast<size_t>(b)].tail = pqHost.tail;
    }
}

// Periodic hang diagnostics — read-only, does not change execution.
static void DumpHangDiagnostics(
    const DeviceBuffers& buf, int rank_id, int n_ranks, const char* phase, const char* waitStreamLabel,
    aclrtStream waitStream, int waitSec, aclrtStream aivStream, aclrtStream computeStream,
    const uint64_t* prevCcuDonePerMission, uint64_t* curCcuDonePerMission)
{
    ProgressCtx hdr{};
    aclrtMemcpy(&hdr, sizeof(hdr), buf.progressCtx_dev, sizeof(hdr), ACL_MEMCPY_DEVICE_TO_HOST);

    const int safeRanks = (n_ranks > 0) ? n_ranks : 1;

    std::cerr << "[CCU-WATCHDOG] rank=" << rank_id << " phase=" << phase << " wait=" << waitSec
              << "s waitingOn=" << waitStreamLabel << " numTiles=" << hdr.numTiles << " nranks=" << safeRanks
              << std::endl;

    std::cerr << "[CCU-WATCHDOG] rank=" << rank_id
              << " streams: wait=" << (StreamLooksIdle(waitStream) ? "idle" : "pending")
              << " aiv=" << (StreamLooksIdle(aivStream) ? "idle" : "pending")
              << " compute=" << (StreamLooksIdle(computeStream) ? "idle" : "pending") << std::endl;

    for (uint32_t m = 0; m < kCcuMissionParallel && m < kMaxCcuMissions; ++m) {
        MissionState ms{};
        aclrtMemcpy(
            &ms, sizeof(ms),
            reinterpret_cast<uint8_t*>(buf.progressCtx_dev) + offsetof(ProgressDeviceCtx, missions) +
                m * sizeof(MissionState),
            sizeof(ms), ACL_MEMCPY_DEVICE_TO_HOST);

        const uint64_t prevDone = prevCcuDonePerMission != nullptr ? prevCcuDonePerMission[m] : 0;
        const uint64_t rsDelta = ms.rsItemsDone >= prevDone ? ms.rsItemsDone - prevDone : 0;
        if (curCcuDonePerMission != nullptr)
            curCcuDonePerMission[m] = ms.rsItemsDone;

        std::cerr << "[CCU-WATCHDOG] rank=" << rank_id << " mission=" << m << " rsItemsDone=" << ms.rsItemsDone
                  << " (+delta=" << rsDelta << ") rsKernelReady=" << ms.rsKernelReady << std::endl;

        DumpMissionState(rank_id, m, ms);
    }

    std::vector<QueueCountSnap> qcounts;
    ReadReadyQueueCounts(buf, qcounts);
    if (!qcounts.empty()) {
        int32_t qsum = 0;
        std::cerr << "[CCU-WATCHDOG] rank=" << rank_id << " readyQueue counts:";
        for (size_t b = 0; b < qcounts.size(); ++b) {
            std::cerr << " b" << b << "=" << qcounts[b].count;
            qsum += qcounts[b].count;
        }
        std::cerr << " sum=" << qsum << " (numTiles=" << hdr.numTiles << ")" << std::endl;
    }

    std::cerr << "[CCU-WATCHDOG] rank=" << rank_id << " phase=" << phase << " wait=" << waitSec << "s" << std::endl;
}

// Sync with periodic progress-ctx dump every 5 s. Use the runtime timed sync
// directly so stream completion is driven the same way as the validated
// baseline path.
static aclError SyncStreamDumpProgress(
    aclrtStream stream, const DeviceBuffers& buf, int rank_id, const char* phase, int n_ranks,
    const char* waitStreamLabel, aclrtStream aivStream, aclrtStream computeStream)
{
    constexpr int kIntervalMs = 5000;
    constexpr int kMaxRounds = 24;
    uint64_t prevDone[kMaxCcuMissions]{};
    TraceStep(rank_id, phase);
    for (int round = 0; round < kMaxRounds; ++round) {
        aclError ret = aclrtSynchronizeStreamWithTimeout(stream, kIntervalMs);
        if (ret == ACL_SUCCESS)
            return ret;

        uint64_t curDone[kMaxCcuMissions]{};
        DumpHangDiagnostics(
            buf, rank_id, n_ranks, phase, waitStreamLabel, stream, (round + 1) * 5, aivStream, computeStream, prevDone,
            curDone);
        std::cerr.flush();
        for (uint32_t m = 0; m < kMaxCcuMissions; ++m)
            prevDone[m] = curDone[m];
    }
    std::cerr << "[WATCHDOG] rank=" << rank_id << " phase=" << phase << " TIMEOUT after 120s" << std::endl;
    std::cerr.flush();
    return ACL_ERROR_RT_STREAM_SYNC_TIMEOUT;
}

static aclError SyncCcuStreamsDumpProgress(
    const CcuState& ccu, const DeviceBuffers& buf, int rank_id, const char* phase, int n_ranks,
    const char* waitStreamLabel, aclrtStream aivStream, aclrtStream computeStream)
{
    return SyncStreamDumpProgress(
        ccu.ccuStream, buf, rank_id, phase, n_ranks, waitStreamLabel, aivStream, computeStream);
}

template <typename ResetState, typename LaunchCompute, typename SyncAll>
static void RunComputeOnlyBenchmark(
    int rank_id, int n_ranks, const DeviceBuffers& buf, ResetState& resetState, LaunchCompute& launchComp,
    SyncAll& syncAll, aclrtStream computeStream, aclrtStream commStream, HcclComm comm,
    std::vector<double>& compute_times_us)
{
    for (int iter = 0; iter < COMPUTE_ONLY_ITERS; ++iter) {
        resetState();
        aclrtSynchronizeStream(computeStream);
        HcclHostBarrier(comm, commStream);
        auto t0 = std::chrono::high_resolution_clock::now();
        launchComp(computeStream);
        aclrtSynchronizeStream(computeStream);
        auto t1 = std::chrono::high_resolution_clock::now();
        compute_times_us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
        HcclHostBarrier(comm, commStream);
    }
}

// Measured iterations for avg/median under CCU re-launch jitter; warmup/verify
// keep the watchdog, measured loops use plain stream sync (no 5s timeout cost).
static constexpr int CCU_MEASURE_ITERS = 20;

// Warm up the dedicated seq one-shot CCU handle before measuring sequential comm.
// Pipelined warmup only exercises the progress-path kernel; without this, the first
// sequential samples include cold-start on the one-shot handle.
template <typename ResetState, typename LaunchCompute, typename SyncAll, typename HostBarrier>
static void WarmupSeqOneShotBaseline(
    int rank_id, ResetState& resetState, LaunchCompute& launchComp, SyncAll& syncAll, HostBarrier& hostBarrier,
    aclrtStream computeStream, CcuState& ccu)
{
    if (!ccu.seqKernelRegistered || std::getenv("PTO_VERIFY_ONLY") != nullptr)
        return;

    for (int i = 0; i < WARMUP_ITERS; ++i) {
        resetState();
        syncAll();
        launchComp(computeStream);
        aclrtSynchronizeStream(computeStream);
        hostBarrier();
        if (!LaunchStoredSeqOneShotCcuGemmAr(ccu))
            continue;
        if (!SyncCcuStreams(ccu, rank_id, "seq one-shot warmup"))
            continue;
        syncAll();
    }
    hostBarrier();
}

// Sequential benchmark: compute → hostBarrier → one-shot fused RS→AG (no AIV progress).
// Like gemm_ar: hostBarrier is required for one-shot readiness but is NOT counted in seq_comm.
// seq_comm = CCU launch → ccuStream sync only (indep. of group_tiles).
template <
    typename ResetState, typename LaunchCompute, typename PrepareCcu, typename LaunchProgress, typename SyncAll,
    typename HostBarrier>
static void RunSequentialBenchmark(
    int rank_id, int n_ranks, const DeviceBuffers& buf, ResetState& resetState, LaunchCompute& launchComp,
    PrepareCcu& prepareCcu, LaunchProgress& launchProgress, SyncAll& syncAll, HostBarrier& hostBarrier,
    aclrtStream computeStream, CcuState& ccu, std::vector<double>& seq_us, std::vector<double>& seq_comp_us,
    std::vector<double>& seq_comm_us, std::vector<CommPhaseTiming>& seq_comm_diag)
{
    (void)n_ranks;
    (void)buf;

    aclrtEvent evCcuStart = nullptr;
    aclrtEvent evCcuEnd = nullptr;
    aclrtCreateEvent(&evCcuStart);
    aclrtCreateEvent(&evCcuEnd);

    for (int iter = 0; iter < CCU_MEASURE_ITERS; ++iter) {
        resetState();
        syncAll();

        if (!ccu.seqKernelRegistered) {
            // Fallback: old progress path (not a true Sequential baseline).
            if (!prepareCcu())
                continue;
            auto t0 = std::chrono::high_resolution_clock::now();
            launchComp(computeStream);
            aclrtSynchronizeStream(computeStream);
            auto t1 = std::chrono::high_resolution_clock::now();
            auto tComm0 = std::chrono::high_resolution_clock::now();
            launchProgress();
            aclrtSynchronizeStream(ccu.aivStream);
            if (!SyncCcuStreams(ccu, rank_id, "sequential fallback ccu"))
                continue;
            auto tCommDone = std::chrono::high_resolution_clock::now();
            CommPhaseTiming commDiag{};
            commDiag.comm_wall_us = ChronoMicros(tComm0, tCommDone);
            seq_comm_diag.push_back(commDiag);
            seq_comp_us.push_back(ChronoMicros(t0, t1));
            seq_comm_us.push_back(commDiag.comm_wall_us);
            seq_us.push_back(ChronoMicros(t0, tCommDone));
            syncAll();
            continue;
        }

        auto t0 = std::chrono::high_resolution_clock::now();
        launchComp(computeStream);
        aclrtSynchronizeStream(computeStream);
        auto t1 = std::chrono::high_resolution_clock::now();

        // Cross-rank gemm_output readiness (one-shot has no AIV groupDone / TNOTIFY path).
        // Same as gemm_ar: barrier is outside the timed comm window.
        hostBarrier();

        auto tComm0 = std::chrono::high_resolution_clock::now();
        aclrtRecordEvent(evCcuStart, ccu.ccuStream);
        if (!LaunchStoredSeqOneShotCcuGemmAr(ccu))
            continue;
        aclrtRecordEvent(evCcuEnd, ccu.ccuStream);
        if (!SyncCcuStreams(ccu, rank_id, "sequential one-shot"))
            continue;
        auto t2 = std::chrono::high_resolution_clock::now();

        CommPhaseTiming commDiag{};
        commDiag.comm_wall_us = ChronoMicros(tComm0, t2);
        commDiag.ccu_wall_us = commDiag.comm_wall_us;
        if (TryEventElapsedUs(evCcuStart, evCcuEnd, commDiag.ccu_event_us))
            commDiag.comm_event_us = commDiag.ccu_event_us;
        seq_comm_diag.push_back(commDiag);

        const double comp_us = ChronoMicros(t0, t1);
        const double comm_us = commDiag.comm_wall_us;
        seq_comp_us.push_back(comp_us);
        seq_comm_us.push_back(comm_us);
        // Like gemm_ar: Sequential = compute + comm (barrier not counted).
        seq_us.push_back(comp_us + comm_us);

        syncAll();
    }

    DestroyAclEvents(evCcuStart, evCcuEnd, nullptr);
}

// Pipelined benchmark: overlap via window groupDone + owner remote poll + CKE.
// Ordering: prepareCcu → launchProgress(AIV monitor) → launchComp
template <typename ResetState, typename LaunchCompute, typename PrepareCcu, typename LaunchProgress, typename SyncAll>
static void RunPipelinedBenchmark(
    int rank_id, int n_ranks, const DeviceBuffers& buf, ResetState& resetState, LaunchCompute& launchComp,
    PrepareCcu& prepareCcu, LaunchProgress& launchProgress, SyncAll& syncAll, aclrtStream computeStream, CcuState& ccu,
    std::vector<double>& pipe_us, std::vector<double>& pipe_comp_us, std::vector<double>& pipe_comm_us,
    std::vector<CommPhaseTiming>& pipe_comm_diag)
{
    (void)n_ranks;
    (void)buf;
    aclrtEvent evStart = nullptr;
    aclrtEvent evEnd = nullptr;
    aclrtEvent evCommStart = nullptr;
    aclrtEvent evAivDone = nullptr;
    aclrtEvent evCcuDone = nullptr;
    aclrtCreateEvent(&evStart);
    aclrtCreateEvent(&evEnd);
    aclrtCreateEvent(&evCommStart);
    aclrtCreateEvent(&evAivDone);
    aclrtCreateEvent(&evCcuDone);

    for (int iter = 0; iter < CCU_MEASURE_ITERS; ++iter) {
        resetState();
        syncAll();

        if (!prepareCcu())
            continue;

        // t0 aligns with sequential benchmark: after prepareCcu, before overlapped work.
        auto t0 = std::chrono::high_resolution_clock::now();
        aclrtRecordEvent(evCommStart, ccu.aivStream);
        launchProgress();

        aclrtRecordEvent(evStart, computeStream);
        launchComp(computeStream);
        aclrtRecordEvent(evEnd, computeStream);

        aclrtSynchronizeStream(ccu.aivStream);
        aclrtRecordEvent(evAivDone, ccu.aivStream);
        auto tAivDone = std::chrono::high_resolution_clock::now();

        if (!SyncCcuStreams(ccu, rank_id, "pipelined benchmark ccu"))
            continue;
        aclrtRecordEvent(evCcuDone, ccu.ccuStream);
        auto tCommDone = std::chrono::high_resolution_clock::now();

        aclrtSynchronizeStream(computeStream);
        auto t1 = std::chrono::high_resolution_clock::now();

        float compute_ms = 0.0f;
        aclrtEventElapsedTime(&compute_ms, evStart, evEnd);

        CommPhaseTiming commDiag{};
        commDiag.comm_wall_us = ChronoMicros(t0, tCommDone);
        commDiag.aiv_wall_us = ChronoMicros(t0, tAivDone);
        commDiag.ccu_wall_us = ChronoMicros(tAivDone, tCommDone);
        FillCommEventElapsed(commDiag, evCommStart, evAivDone, evCcuDone);
        pipe_comm_diag.push_back(commDiag);

        if (SchedTimingEnabled() && iter == CCU_MEASURE_ITERS - 1) {
            if (ReadSchedTimingFromDevice(buf, g_lastPipeRsTiming))
                g_lastPipeSchedTimingValid = true;
        }

        pipe_comp_us.push_back(static_cast<double>(compute_ms) * 1000.0);
        pipe_comm_us.push_back(commDiag.comm_wall_us);
        pipe_us.push_back(ChronoMicros(t0, t1));

        syncAll();
    }

    DestroyAclEvents(evCommStart, evAivDone, evCcuDone);
    aclrtDestroyEvent(evStart);
    aclrtDestroyEvent(evEnd);
}

// ============================================================================
// Per-rank device buffer management
// ============================================================================

static bool AllocDeviceBuffers(
    DeviceBuffers& buf, int rank_id, int n_ranks, const CommDeviceContext& windowHostCtx, const uint16_t* a_data,
    size_t a_bytes, const uint16_t* b_data, size_t b_bytes)
{
    const uint32_t safeRanks = (n_ranks > 0) ? static_cast<uint32_t>(n_ranks) : 1;
    buf.packedSize =
        static_cast<size_t>(CcuTotalPaddedTiles(G_NUM_TILES, safeRanks)) * static_cast<size_t>(G_TILE_BYTES);
    buf.rowOutputSize = static_cast<size_t>(G_M) * G_N * sizeof(uint16_t);
    // Layout inside the single registered CCU block:
    //   [gemm_output_packed | reduced_output_packed | row_output]
    // signal_matrix lives in the HCCL window (groupDone + readyQueue subtile_ready).
    auto alignUp = [](size_t v, size_t a) { return (v + a - 1) / a * a; };
    buf.signalBytes = alignUp(static_cast<size_t>(G_SIGNAL_MATRIX_SLOTS) * sizeof(int32_t), 512);
    size_t ccuBlockSize = 2 * buf.packedSize + buf.rowOutputSize;

    aclrtMallocAttrValue modVal{};
    modVal.moduleId = 3;
    aclrtMallocAttribute attr{ACL_RT_MEM_ATTR_MODULE_ID, modVal};
    aclrtMallocConfig cfg{&attr, 1};
    aclError aRet = aclrtMallocWithCfg(&buf.ccuBlock, ccuBlockSize, ACL_MEM_TYPE_HIGH_BAND_WIDTH, &cfg);
    if (aRet != ACL_SUCCESS || !buf.ccuBlock) {
        std::cerr << "[ERROR] Rank " << rank_id << ": alloc CCU HBM block failed: " << static_cast<int>(aRet)
                  << std::endl;
        return false;
    }

    buf.gemm_output = buf.ccuBlock;
    buf.reduced_output = static_cast<char*>(buf.ccuBlock) + buf.packedSize;
    buf.row_output = static_cast<char*>(buf.ccuBlock) + 2 * buf.packedSize;

    buf.ccuToken = hcomm::CcuRep::GetTokenInfo(reinterpret_cast<uint64_t>(buf.ccuBlock), ccuBlockSize);

    // Allocate signal_matrix from the HCCL window (groupDone + legacy subtile_ready).
    uint64_t windowBase = windowHostCtx.windowsIn[windowHostCtx.rankId];
    size_t winOffset = 0;
    void* guardPad = WindowAlloc(windowBase, winOffset, WINDOW_GUARD_BYTES);
    buf.signal_matrix = WindowAlloc(windowBase, winOffset, buf.signalBytes);
    if (winOffset > windowHostCtx.winSize) {
        std::cerr << "[ERROR] Rank " << rank_id << ": HCCL window too small for signal_matrix (need " << winOffset
                  << ", have " << windowHostCtx.winSize << ")" << std::endl;
        aclrtFree(buf.ccuBlock);
        return false;
    }
    // Progress-path groupDone: alias into window (AIC write + owner remote poll).
    buf.groupDoneSize = static_cast<size_t>(G_SIGNAL_GROUP_DONE_SLOTS) * sizeof(int32_t);
    buf.groupDone_dev = reinterpret_cast<int32_t*>(buf.signal_matrix) + G_SIGNAL_GROUP_DONE_OFFSET;

    if (VerboseLog()) {
        std::cerr << "[INFO] Rank " << rank_id << ": CCU block VA=0x" << std::hex
                  << reinterpret_cast<uint64_t>(buf.ccuBlock) << " size=" << std::dec << ccuBlockSize << " token=0x"
                  << std::hex << buf.ccuToken << std::dec << " signal_matrix(window)=0x" << std::hex
                  << reinterpret_cast<uint64_t>(buf.signal_matrix) << std::dec << std::endl;
    }

    aclrtMemset(buf.gemm_output, buf.packedSize, 0, buf.packedSize);
    aclrtMemset(buf.reduced_output, buf.packedSize, 0, buf.packedSize);
    aclrtMemset(buf.row_output, buf.rowOutputSize, 0, buf.rowOutputSize);
    aclrtMemset(guardPad, WINDOW_GUARD_BYTES, 0, WINDOW_GUARD_BYTES);
    aclrtMemset(buf.signal_matrix, buf.signalBytes, 0, buf.signalBytes);

    size_t aSize = (size_t)G_M * G_K * sizeof(uint16_t);
    size_t bSize = (size_t)G_K * G_N * sizeof(uint16_t);
    aRet = aclrtMalloc(&buf.src0_dev, aSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (aRet != ACL_SUCCESS || !buf.src0_dev) {
        std::cerr << "[ERROR] Rank " << rank_id << ": alloc src0_dev failed: " << static_cast<int>(aRet) << std::endl;
        aclrtFree(buf.ccuBlock);
        return false;
    }
    aRet = aclrtMalloc(&buf.src1_dev, bSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (aRet != ACL_SUCCESS || !buf.src1_dev) {
        std::cerr << "[ERROR] Rank " << rank_id << ": alloc src1_dev failed: " << static_cast<int>(aRet) << std::endl;
        aclrtFree(buf.ccuBlock);
        aclrtFree(buf.src0_dev);
        return false;
    }

    aRet = aclrtMalloc(&buf.progressCtx_dev, sizeof(ProgressDeviceCtx), ACL_MEM_MALLOC_HUGE_FIRST);
    if (aRet != ACL_SUCCESS || !buf.progressCtx_dev) {
        std::cerr << "[ERROR] Rank " << rank_id << ": alloc progressCtx_dev failed: " << static_cast<int>(aRet)
                  << std::endl;
        aclrtFree(buf.ccuBlock);
        aclrtFree(buf.src0_dev);
        aclrtFree(buf.src1_dev);
        return false;
    }

    WriteProgressCtxToDevice(buf);

    // Ready queue (MultiBlockQueueSet): the AIC compute kernel enqueues each
    // finished tile id into its PerBlockQueue; the AIV scheduler dequeues them.
    buf.readyQueueSize = MultiBlockQueueSetSize(COMPUTE_BLOCK_NUM, G_NUM_TILES);
    aRet = aclrtMalloc(&buf.readyQueue_dev, buf.readyQueueSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (aRet != ACL_SUCCESS || !buf.readyQueue_dev) {
        std::cerr << "[ERROR] Rank " << rank_id << ": alloc readyQueue_dev failed: " << static_cast<int>(aRet)
                  << std::endl;
        aclrtFree(buf.ccuBlock);
        aclrtFree(buf.src0_dev);
        aclrtFree(buf.src1_dev);
        aclrtFree(buf.progressCtx_dev);
        return false;
    }
    buf.readyQueue_reset_host = nullptr;
    aclrtMallocHost(reinterpret_cast<void**>(&buf.readyQueue_reset_host), buf.readyQueueSize);
    if (!buf.readyQueue_reset_host) {
        std::cerr << "[ERROR] Rank " << rank_id << ": alloc readyQueue_reset_host failed" << std::endl;
        aclrtFree(buf.ccuBlock);
        aclrtFree(buf.src0_dev);
        aclrtFree(buf.src1_dev);
        aclrtFree(buf.progressCtx_dev);
        aclrtFree(buf.readyQueue_dev);
        return false;
    }
    MultiBlockQueueSetInit(buf.readyQueue_reset_host, COMPUTE_BLOCK_NUM, G_NUM_TILES);
    aclrtMemcpy(
        buf.readyQueue_dev, buf.readyQueueSize, buf.readyQueue_reset_host, buf.readyQueueSize,
        ACL_MEMCPY_HOST_TO_DEVICE);

    aclrtMemcpy(buf.src0_dev, aSize, a_data, a_bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(buf.src1_dev, bSize, b_data, b_bytes, ACL_MEMCPY_HOST_TO_DEVICE);

    return true;
}

static void FreeDeviceBuffers(DeviceBuffers& buf)
{
    if (buf.ccuBlock)
        aclrtFree(buf.ccuBlock);
    if (buf.src0_dev)
        aclrtFree(buf.src0_dev);
    if (buf.src1_dev)
        aclrtFree(buf.src1_dev);
    if (buf.progressCtx_dev)
        aclrtFree(buf.progressCtx_dev);
    if (buf.readyQueue_dev)
        aclrtFree(buf.readyQueue_dev);
    // groupDone_dev aliases signal_matrix (HCCL window) — not separately freed.
    if (buf.readyQueue_reset_host)
        aclrtFreeHost(buf.readyQueue_reset_host);
    buf = {};
}

// ============================================================================
// Per-rank execution logic
// ============================================================================
// Warmup: pipelined ordering — progress monitor then overlapped compute.
template <typename ResetState, typename LaunchCompute, typename PrepareCcu, typename LaunchProgress, typename SyncAll>
static void RunWarmupPasses(
    int rank_id, int n_ranks, ResetState& resetState, LaunchCompute& launchComp, PrepareCcu& prepareCcu,
    LaunchProgress& launchProgress, SyncAll& syncAll, aclrtStream computeStream, CcuState& ccu,
    const DeviceBuffers& buf)
{
    const int warmupIters = (std::getenv("PTO_VERIFY_ONLY") != nullptr) ? 0 : WARMUP_ITERS;
    for (int i = 0; i < warmupIters; ++i) {
        resetState();
        syncAll();
        if (!prepareCcu())
            continue;
        if (VerboseLog()) {
            std::cerr << "[CCU-AR] rank=" << rank_id << ": prepare done, fire scheduler+compute (warmup " << i << ")"
                      << std::endl;
            std::cerr.flush();
        }
        TraceStep(rank_id, "warmup: launchProgress");
        launchProgress();
        TraceStep(rank_id, "warmup: launchComp");
        launchComp(computeStream);
        TraceStep(rank_id, "warmup: launchComp done, sync progress");
        if (SyncStreamDumpProgress(
                computeStream, buf, rank_id, "warmup-compute", n_ranks, "compute", ccu.aivStream, computeStream) !=
            ACL_SUCCESS)
            return;
        if (SyncCcuStreamsDumpProgress(ccu, buf, rank_id, "warmup-ccu", n_ranks, "ccu", ccu.aivStream, computeStream) !=
            ACL_SUCCESS)
            return;
        aclrtSynchronizeStream(ccu.aivStream);
        syncAll();
    }
}

// Final verification pass.
template <typename ResetState, typename LaunchCompute, typename PrepareCcu, typename LaunchProgress, typename SyncAll>
static bool RunFinalVerificationPass(
    int rank_id, int n_ranks, ResetState& resetState, LaunchCompute& launchComp, PrepareCcu& prepareCcu,
    LaunchProgress& launchProgress, SyncAll& syncAll, aclrtStream computeStream, CcuState& ccu,
    const DeviceBuffers& buf, const float* golden)
{
    resetState();
    syncAll();
    if (!prepareCcu())
        return false;
    launchProgress();
    launchComp(computeStream);
    if (SyncStreamDumpProgress(
            computeStream, buf, rank_id, "verify-compute", n_ranks, "compute", ccu.aivStream, computeStream) !=
        ACL_SUCCESS)
        return false;
    if (SyncCcuStreamsDumpProgress(ccu, buf, rank_id, "verify-ccu", n_ranks, "ccu", ccu.aivStream, computeStream) !=
        ACL_SUCCESS)
        return false;
    if (SyncStreamDumpProgress(
            ccu.aivStream, buf, rank_id, "verify-aiv", n_ranks, "aiv", ccu.ccuStream, computeStream) != ACL_SUCCESS)
        return false;

    // AG writes into this rank's reduced_output are issued by peer CCU streams.
    // Local stream completion only proves this rank has finished its own work,
    // so wait for every rank before unpacking the peer-written packed buffer.
    syncAll();
    launchCcuGemmArUnpack(
        reinterpret_cast<uint8_t*>(buf.reduced_output), reinterpret_cast<uint8_t*>(buf.row_output), ccu.aivStream,
        kCcuSubtilesPerTile, static_cast<uint32_t>(n_ranks));
    aclrtSynchronizeStream(ccu.aivStream);
    syncAll();
    if (VerboseLog()) {
        DumpReadyCountersAfterCcu(buf, rank_id);
        DumpVerifySamples(buf, rank_id, golden);
    }

    uint16_t* output_host_fp16 = nullptr;
    aclrtMallocHost(reinterpret_cast<void**>(&output_host_fp16), buf.rowOutputSize);
    aclrtMemcpy(output_host_fp16, buf.rowOutputSize, buf.row_output, buf.rowOutputSize, ACL_MEMCPY_DEVICE_TO_HOST);

    // row_output is [M, G_N] row-major; CCU communication uses packed buffers.
    bool is_ok = VerifyOutput(output_host_fp16, golden, n_ranks);
    std::cerr << "[VERIFY] rank=" << rank_id << " result=" << (is_ok ? "PASS" : "FAIL") << std::endl;
    aclrtFreeHost(output_host_fp16);
    return is_ok;
}

template <
    typename ResetState, typename LaunchCompute, typename PrepareCcu, typename LaunchProgress, typename SyncAll,
    typename HostBarrier>
static bool RunBenchmarkAndVerify(
    int rank_id, int n_ranks, ResetState& resetState, LaunchCompute& launchComp, PrepareCcu& prepareCcu,
    LaunchProgress& launchProgress, SyncAll& syncAll, HostBarrier& hostBarrier, aclrtStream computeStream,
    aclrtStream commStream, HcclComm comm, CcuState& ccu, const DeviceBuffers& buf, const float* golden)
{
    RunWarmupPasses(
        rank_id, n_ranks, resetState, launchComp, prepareCcu, launchProgress, syncAll, computeStream, ccu, buf);
    WarmupSeqOneShotBaseline(rank_id, resetState, launchComp, syncAll, hostBarrier, computeStream, ccu);

    std::vector<double> compute_us;
    std::vector<double> seq_us;
    std::vector<double> seq_comp_us;
    std::vector<double> seq_comm_us;
    std::vector<double> pipe_us;
    std::vector<double> pipe_comp_us;
    std::vector<double> pipe_comm_us;
    std::vector<CommPhaseTiming> seq_comm_diag;
    std::vector<CommPhaseTiming> pipe_comm_diag;
    // PTO_VERIFY_ONLY=1: skip benchmarks so verify runs against the first and only
    // CCU pass (after warmup) — isolates correctness from repeated-launch state.
    const bool verifyOnly = std::getenv("PTO_VERIFY_ONLY") != nullptr;
    if (!verifyOnly) {
        g_lastPipeSchedTimingValid = false;
        RunComputeOnlyBenchmark(
            rank_id, n_ranks, buf, resetState, launchComp, syncAll, computeStream, commStream, comm, compute_us);
        RunSequentialBenchmark(
            rank_id, n_ranks, buf, resetState, launchComp, prepareCcu, launchProgress, syncAll, hostBarrier,
            computeStream, ccu, seq_us, seq_comp_us, seq_comm_us, seq_comm_diag);
        RunPipelinedBenchmark(
            rank_id, n_ranks, buf, resetState, launchComp, prepareCcu, launchProgress, syncAll, computeStream, ccu,
            pipe_us, pipe_comp_us, pipe_comm_us, pipe_comm_diag);
    }

    const bool is_ok = RunFinalVerificationPass(
        rank_id, n_ranks, resetState, launchComp, prepareCcu, launchProgress, syncAll, computeStream, ccu, buf, golden);

    if (!verifyOnly) {
        if (rank_id == 0) {
            PrintPerfReport(
                is_ok, n_ranks, compute_us, seq_us, pipe_us, seq_comp_us, seq_comm_us, pipe_comp_us, pipe_comm_us,
                rank_id, seq_comm_diag, pipe_comm_diag);
        }
        // Every rank reports its own progress wait mix (critical-path owner may not be rank0).
        syncAll();
        const double last_pipe_aiv_us = pipe_comm_diag.empty() ? 0.0 : pipe_comm_diag.back().aiv_wall_us;
        PrintSchedTimingReport(rank_id, last_pipe_aiv_us);
        syncAll();
    }
    return is_ok;
}

static void ResetDeviceState(int rank_id, int n_ranks, const DeviceBuffers& buf)
{
    if (buf.readyQueue_reset_host != nullptr && buf.readyQueue_dev != nullptr && buf.readyQueueSize > 0) {
        MultiBlockQueueSetInit(buf.readyQueue_reset_host, COMPUTE_BLOCK_NUM, G_NUM_TILES);
        aclrtMemcpy(
            buf.readyQueue_dev, buf.readyQueueSize, buf.readyQueue_reset_host, buf.readyQueueSize,
            ACL_MEMCPY_HOST_TO_DEVICE);
    }
    aclrtMemset(buf.gemm_output, buf.packedSize, 0, buf.packedSize);
    aclrtMemset(buf.reduced_output, buf.packedSize, 0, buf.packedSize);
    aclrtMemset(buf.row_output, buf.rowOutputSize, 0, buf.rowOutputSize);
    // Clears subtile_ready and window groupDone together.
    aclrtMemset(buf.signal_matrix, buf.signalBytes, 0, buf.signalBytes);
}

static void LaunchCompute(const DeviceBuffers& buf, int rank_id, int n_ranks, aclrtStream s)
{
    const bool useProgress = UseProgressPath();
    launchCcuGemmArCompute(
        reinterpret_cast<uint8_t*>(buf.gemm_output), reinterpret_cast<uint8_t*>(buf.src0_dev),
        reinterpret_cast<uint8_t*>(buf.src1_dev),
        useProgress ? nullptr : reinterpret_cast<uint8_t*>(buf.readyQueue_dev),
        useProgress ? reinterpret_cast<int32_t*>(buf.groupDone_dev) : nullptr, rank_id, s, COMPUTE_BLOCK_NUM, G_K,
        static_cast<uint32_t>(n_ranks));
}

// Launch the AIV progress monitor after any prior pass on the same stream has finished.
static void LaunchCcuProgress(CcuState& ccu, const DeviceBuffers& buf, int rankId, int nRanks)
{
    ProgressCkeCtx rsCke{};
    for (uint32_t p = 0; p < 2; ++p) {
        rsCke.ckeSlotVA[p] = ccu.rsProgressVA[0][p];
        rsCke.ckeMask[p] = ccu.rsProgressMask[0][p];
    }
    rsCke.itemsDoneAddr = MissionRsItemsDoneAddr(buf.progressCtx_dev, 0);
    rsCke.kernelReadyAddr = MissionRsKernelReadyAddr(buf.progressCtx_dev, 0);

    const bool useProgress = UseProgressPath();

    if (useProgress && buf.groupDone_dev != nullptr) {
        launchCcuGemmArProgress(
            ccu.aivStream, rsCke, reinterpret_cast<int32_t*>(buf.groupDone_dev),
            reinterpret_cast<uint8_t*>(buf.progressCtx_dev), reinterpret_cast<uint8_t*>(buf.signal_matrix),
            reinterpret_cast<uint8_t*>(buf.windowDeviceCtx), G_NUM_TILES, static_cast<uint32_t>(nRanks),
            static_cast<uint32_t>(rankId));
        TraceStep(rankId, "launchProgress: localGD + groupReady TNOTIFY progress launched");
    } else {
        launchCcuGemmArScheduler(
            ccu.aivStream, rsCke, reinterpret_cast<uint8_t*>(buf.readyQueue_dev),
            reinterpret_cast<uint8_t*>(buf.progressCtx_dev), reinterpret_cast<uint8_t*>(buf.signal_matrix),
            reinterpret_cast<uint8_t*>(buf.windowDeviceCtx), static_cast<uint32_t>(nRanks),
            static_cast<uint32_t>(rankId));
        TraceStep(rankId, "launchProgress: readyQueue scheduler launched");
    }
}

static bool RunGemmAllReducePerRank(
    int rank_id, int n_ranks, const uint16_t* a_data, size_t a_bytes, const uint16_t* b_data, size_t b_bytes,
    const float* golden, const HcclRootInfo* rootInfo)
{
    int status = 0;
    aclrtStream computeStream = nullptr;
    aclrtStream commStream = nullptr;
    status |= aclrtCreateStream(&computeStream);
    status |= aclrtCreateStream(&commStream);

    // Lightweight HCCL comm init — CCU does NOT need the heavy GemmHcclContext
    // (AllocComResourceByTiling for AICPU_TS may conflict with CCU channel setup).
    HcclComm hcclComm = nullptr;
    {
        constexpr int kMaxRetries = 3;
        HcclResult hret = HCCL_SUCCESS;
        for (int attempt = 0; attempt < kMaxRetries; ++attempt) {
            hret = HcclCommInitRootInfo(
                static_cast<uint32_t>(n_ranks), rootInfo, static_cast<uint32_t>(rank_id), &hcclComm);
            if (hret == HCCL_SUCCESS)
                break;
            std::cerr << "[WARN] Rank " << rank_id << ": HcclCommInitRootInfo failed: " << hret << " (attempt "
                      << (attempt + 1) << "/" << kMaxRetries << "), retrying in 5s..." << std::endl;
            sleep(5);
        }
        if (hret != HCCL_SUCCESS) {
            std::cerr << "[ERROR] Rank " << rank_id << ": HcclCommInitRootInfo failed\n";
            status |= aclrtDestroyStream(computeStream);
            status |= aclrtDestroyStream(commStream);
            return false;
        }
    }

    // CCU infrastructure (channels must be acquired before HCCL window alloc).
    CcuState ccu;
    status |= aclrtCreateStream(&ccu.ccuStream);
    status |= aclrtCreateStream(&ccu.aivStream);

    constexpr uint32_t kNotifyNum = 1;
    HcclResult hret =
        HcclThreadAcquireWithStream(hcclComm, COMM_ENGINE_CCU, ccu.ccuStream, kNotifyNum, &ccu.threadHandle);
    if (hret != HCCL_SUCCESS) {
        std::cerr << "[ERROR] Rank " << rank_id << ": HcclThreadAcquireWithStream failed: " << static_cast<int>(hret)
                  << std::endl;
        HcclCommDestroy(hcclComm);
        return false;
    }

    if (!SetupCcuChannels(hcclComm, rank_id, n_ranks, ccu.channels)) {
        std::cerr << "[ERROR] Rank " << rank_id << ": CCU channel setup failed!\n";
        HcclCommDestroy(hcclComm);
        return false;
    }

    // HCCL window for AIV TNOTIFY — alloc after CCU channels on the same comm.
    GemmHcclContext windowCtx;
    if (!windowCtx.InitWindowOnExistingComm(rank_id, n_ranks, hcclComm, commStream)) {
        std::cerr << "[ERROR] Rank " << rank_id << ": HCCL window context init failed\n";
        HcclCommDestroy(hcclComm);
        status |= aclrtDestroyStream(computeStream);
        status |= aclrtDestroyStream(commStream);
        return false;
    }

    // Buffer allocation (CCU block for data, HCCL window for signal)
    DeviceBuffers buf{};
    buf.windowDeviceCtx = windowCtx.deviceCtx;
    if (!AllocDeviceBuffers(buf, rank_id, n_ranks, windowCtx.hostCtx, a_data, a_bytes, b_data, b_bytes)) {
        windowCtx.Finalize();
        HcclCommDestroy(hcclComm);
        status |= aclrtDestroyStream(computeStream);
        status |= aclrtDestroyStream(commStream);
        return false;
    }

    HcclHostBarrier(hcclComm, commStream);

    // Exchange CCU buffer VAs and tokens across all ranks
    struct CcuAddrPack {
        uint64_t inputVA;
        uint64_t outputVA;
        uint64_t token;
    };
    CcuAddrPack myPack{
        reinterpret_cast<uint64_t>(buf.gemm_output), reinterpret_cast<uint64_t>(buf.reduced_output), buf.ccuToken};
    std::vector<CcuAddrPack> allPacks(n_ranks);
    CommMpiAllgather(&myPack, sizeof(CcuAddrPack), allPacks.data(), sizeof(CcuAddrPack));

    static_assert(MAX_RANKS <= kMaxGemmArRanks, "MAX_RANKS exceeds CCU limit");
    uint64_t allInputVA[kMaxGemmArRanks]{};
    uint64_t allOutputVA[kMaxGemmArRanks]{};
    uint64_t allTokens[kMaxGemmArRanks]{};
    for (int i = 0; i < n_ranks && i < static_cast<int>(kMaxGemmArRanks); ++i) {
        allInputVA[i] = allPacks[i].inputVA;
        allOutputVA[i] = allPacks[i].outputVA;
        allTokens[i] = allPacks[i].token;
    }
    if (VerboseLog()) {
        std::cerr << "[INFO] Rank " << rank_id << ": CCU VA exchange done, " << n_ranks << " peers" << std::endl;
    }

    CommMpiBarrier();

    if (!RegisterCcuGemmArOnce(hcclComm, ccu, rank_id, n_ranks, buf)) {
        FreeDeviceBuffers(buf);
        HcclCommDestroy(hcclComm);
        status |= aclrtDestroyStream(computeStream);
        status |= aclrtDestroyStream(commStream);
        return false;
    }
    StoreCcuLaunchContext(hcclComm, rank_id, n_ranks, allInputVA, allOutputVA, allTokens);

    auto resetState = [&]() { ResetDeviceState(rank_id, n_ranks, buf); };
    auto launchComp = [&](aclrtStream s) { LaunchCompute(buf, rank_id, n_ranks, s); };
    auto prepareCcu = [&]() {
        return PrepareCcuKernel(buf, hcclComm, ccu, rank_id, n_ranks, allInputVA, allOutputVA, allTokens);
    };
    auto launchProgress = [&]() { LaunchCcuProgress(ccu, buf, rank_id, n_ranks); };
    auto hostBarrier = [&]() { HcclHostBarrier(hcclComm, commStream); };
    auto syncAll = [&]() {
        aclrtSynchronizeStream(computeStream);
        SyncCcuStreams(ccu, rank_id, "syncAll ccu");
        aclrtSynchronizeStream(ccu.aivStream);
        aclrtSynchronizeStream(commStream);
        HcclHostBarrier(hcclComm, commStream);
    };

    bool is_ok = RunBenchmarkAndVerify(
        rank_id, n_ranks, resetState, launchComp, prepareCcu, launchProgress, syncAll, hostBarrier, computeStream,
        commStream, hcclComm, ccu, buf, golden);

    FreeDeviceBuffers(buf);
    if (ccu.ccuStream)
        aclrtDestroyStream(ccu.ccuStream);
    if (ccu.aivStream)
        aclrtDestroyStream(ccu.aivStream);
    windowCtx.Finalize();
    HcclCommDestroy(hcclComm);
    status |= aclrtDestroyStream(computeStream);
    status |= aclrtDestroyStream(commStream);
    return (status == 0) && is_ok;
}

// ============================================================================
// MPI-based multi-process launcher
// ============================================================================
static void PrintLaunchBanner(int n_ranks, int first_device_id)
{
    std::cout << "\n================================================================" << std::endl;
    std::cout << "  CCU GEMM AllReduce A5 FP16 (ReduceScatter + AllGather) — HCCL backend" << std::endl;
    std::cout << "  M=" << G_ORIG_M << " K=" << G_K << " N=" << G_ORIG_N;
    if (G_M != G_ORIG_M || G_N != G_ORIG_N)
        std::cout << "  (padded " << G_M << "x" << G_N << ")";
    std::cout << "  tile=" << G_BASE_M << "x" << G_BASE_K << "x" << G_BASE_N << "  tiles=" << G_NUM_TILES << std::endl;
    std::cout << "  ranks=" << n_ranks << "  devices=[" << first_device_id << "," << (first_device_id + n_ranks) << ")"
              << "  compute_blocks=" << COMPUTE_BLOCK_NUM << "  ccu_group_tiles=" << G_COMM_GROUP_TILES << std::endl;
    std::cout << "  mode: independent A per rank, shared B (CCU tile-level fusion)" << std::endl;
    std::cout << "================================================================" << std::endl;
}

static bool InitHcclRootInfoWithRetry(HcclRootInfo& rootInfo)
{
    constexpr int kMaxRetries = 3;
    HcclResult hret = HCCL_SUCCESS;
    for (int attempt = 0; attempt < kMaxRetries; ++attempt) {
        hret = HcclGetRootInfo(&rootInfo);
        if (hret == HCCL_SUCCESS)
            return true;
        std::cerr << "[WARN] HcclGetRootInfo failed: " << hret << " (attempt " << (attempt + 1) << "/" << kMaxRetries
                  << "), retrying in 5s..." << std::endl;
        sleep(5);
    }
    std::cerr << "[ERROR] HcclGetRootInfo failed after " << kMaxRetries << " attempts: " << hret << std::endl;
    return false;
}

static bool RunGemmAllReduce(
    int n_ranks, int first_device_id, const uint16_t* a_parts, const uint16_t* b_data, const float* golden)
{
    if (n_ranks <= 0 || n_ranks > 8) {
        std::cerr << "[ERROR] Invalid n_ranks: " << n_ranks << " (must be 1-8)\n";
        return false;
    }

    int mpiRank = CommMpiRank();
    if (mpiRank == 0)
        PrintLaunchBanner(n_ranks, first_device_id);

    int device_id = mpiRank % n_ranks + first_device_id;

    constexpr int kAclRepeatInit = 100002;
    aclError aRet = aclInit(nullptr);
    if (aRet != ACL_SUCCESS && static_cast<int>(aRet) != kAclRepeatInit) {
        std::cerr << "[ERROR] Rank " << mpiRank << ": aclInit failed: " << (int)aRet << std::endl;
        return false;
    }

    aRet = aclrtSetDevice(device_id);
    if (aRet != ACL_SUCCESS) {
        std::cerr << "[ERROR] Rank " << mpiRank << ": aclrtSetDevice(" << device_id << ") failed\n";
        return false;
    }

    HcclRootInfo rootInfo{};
    if (mpiRank == 0 && !InitHcclRootInfoWithRetry(rootInfo))
        return false;
    CommMpiBcast(&rootInfo, HCCL_ROOT_INFO_BYTES, COMM_MPI_CHAR, 0);
    CommMpiBarrier();

    size_t a_rank_elems = (size_t)G_M * G_K;
    bool ok = RunGemmAllReducePerRank(
        mpiRank, n_ranks, a_parts + (size_t)mpiRank * a_rank_elems, (size_t)G_M * G_K * sizeof(uint16_t), b_data,
        (size_t)G_N * G_K * sizeof(uint16_t), golden, &rootInfo);
    CommMpiBarrier();
    aclrtResetDevice(device_id);
    aclFinalize();
    return ok;
}

// ============================================================================
// Data generation helpers
// ============================================================================

static uint16_t floatToHalf(float f)
{
    union {
        float f;
        uint32_t u;
    } bits;
    bits.f = f;
    uint32_t x = bits.u;
    uint32_t sign = (x >> 16) & 0x8000;
    int32_t exp = (int32_t)((x >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = x & 0x007FFFFF;
    if (exp <= 0)
        return (uint16_t)sign;
    if (exp >= 31)
        return (uint16_t)(sign | 0x7C00);
    return (uint16_t)(sign | ((uint32_t)exp << 10) | (mant >> 13));
}

static void gemmBlockedTile(
    const float* B, float* C_row, int K, int N, const float* A_row, int kk, int kEnd, int jj, int jEnd)
{
    for (int k = kk; k < kEnd; k++) {
        float aik = A_row[k];
        for (int j = jj; j < jEnd; j++)
            C_row[(size_t)j] += aik * B[(size_t)k * N + j];
    }
}

static void gemmBlockedRowRange(const float* A, const float* B, float* C, int K, int N, int r0, int r1)
{
    constexpr int BLK = 64;
    for (int i = r0; i < r1; i++) {
        for (int kk = 0; kk < K; kk += BLK) {
            int kEnd = std::min(kk + BLK, K);
            for (int jj = 0; jj < N; jj += BLK)
                gemmBlockedTile(B, &C[(size_t)i * N], K, N, &A[(size_t)i * K], kk, kEnd, jj, std::min(jj + BLK, N));
        }
    }
}

static void computeGolden(const float* A, const float* B, float* C, int M, int K, int N)
{
    memset_s(C, (size_t)M * N * sizeof(float), 0, (size_t)M * N * sizeof(float));

    unsigned hw = std::thread::hardware_concurrency();
    if (hw == 0)
        hw = 1;
    unsigned n_threads = std::min(hw, 64u);

    int rows_per = (M + (int)n_threads - 1) / (int)n_threads;
    std::vector<std::thread> threads;

    for (unsigned t = 0; t < n_threads; t++) {
        int r0 = (int)t * rows_per;
        int r1 = std::min(r0 + rows_per, M);
        if (r0 >= M)
            break;
        threads.emplace_back(gemmBlockedRowRange, A, B, C, K, N, r0, r1);
    }
    for (auto& th : threads)
        th.join();
}

// ============================================================================
// Input file caching: save/load binary matrices to avoid regeneration
// ============================================================================

struct CachedEnvVars {
    std::string gemm_ar_dir;
    std::string first_device;
};

static CachedEnvVars g_cached_env;

static std::string SafeGetEnv(const char* name)
{
    std::string prefix = std::string(name) + "=";
    std::ifstream ifs("/proc/self/environ", std::ios::binary);
    if (!ifs.is_open()) {
        return {};
    }
    std::string entry;
    while (std::getline(ifs, entry, '\0')) {
        if (entry.compare(0, prefix.size(), prefix) == 0) {
            return entry.substr(prefix.size());
        }
    }
    return {};
}

static void InitCachedEnv()
{
    g_cached_env.gemm_ar_dir = SafeGetEnv("GEMM_AR_DIR");
    g_cached_env.first_device = SafeGetEnv("GEMM_ALLREDUCE_FIRST_DEVICE");
}

static std::string getInputDir()
{
    const std::string& envDir = g_cached_env.gemm_ar_dir;
    if (!envDir.empty()) {
        return envDir + "/input";
    }
    return "input";
}

static std::string getInputPrefix(int nranks)
{
    std::string dir = getInputDir();
    return dir + "/M" + std::to_string(G_ORIG_M) + "_K" + std::to_string(G_ORIG_K) + "_N" + std::to_string(G_ORIG_N) +
           "_R" + std::to_string(nranks);
}

static void ensureDirExists(const std::string& dir) { mkdir(dir.c_str(), 0755); }

template <typename T>
static bool saveBinary(const std::string& path, const T* data, size_t count)
{
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs)
        return false;
    ofs.write(reinterpret_cast<const char*>(data), count * sizeof(T));
    return ofs.good();
}

template <typename T>
static bool loadBinary(const std::string& path, T* data, size_t count)
{
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs)
        return false;
    ifs.read(reinterpret_cast<char*>(data), count * sizeof(T));
    return ifs.good();
}

static bool inputFilesExist(int nranks)
{
    std::string prefix = getInputPrefix(nranks);
    std::string aFile = prefix + "_A.bin";
    std::string bFile = prefix + "_B.bin";
    std::string gFile = prefix + "_golden.bin";
    struct stat st;
    return (stat(aFile.c_str(), &st) == 0 && stat(bFile.c_str(), &st) == 0 && stat(gFile.c_str(), &st) == 0);
}

static bool loadInputFiles(
    int nranks, std::vector<uint16_t>& a_parts, std::vector<uint16_t>& b_data, std::vector<float>& golden)
{
    std::string prefix = getInputPrefix(nranks);
    std::string aFile = prefix + "_A.bin";
    std::string bFile = prefix + "_B.bin";
    std::string gFile = prefix + "_golden.bin";

    size_t a_total = (size_t)nranks * G_M * G_K;
    size_t b_total = (size_t)G_N * G_K;
    size_t g_total = (size_t)G_M * G_N;

    a_parts.resize(a_total);
    b_data.resize(b_total);
    golden.resize(g_total);

    if (VerboseLog())
        printf("Loading cached input from: %s_*.bin\n", prefix.c_str());

    if (!loadBinary(aFile, a_parts.data(), a_total)) {
        fprintf(stderr, "[ERROR] Failed to load %s\n", aFile.c_str());
        return false;
    }
    if (!loadBinary(bFile, b_data.data(), b_total)) {
        fprintf(stderr, "[ERROR] Failed to load %s\n", bFile.c_str());
        return false;
    }
    if (!loadBinary(gFile, golden.data(), g_total)) {
        fprintf(stderr, "[ERROR] Failed to load %s\n", gFile.c_str());
        return false;
    }

    if (VerboseLog())
        printf("  Loaded A[%d ranks × %d × %d], B[%d × %d], golden[%d × %d]\n", nranks, G_M, G_K, G_N, G_K, G_M, G_N);
    return true;
}

static bool saveInputFiles(
    int nranks, const std::vector<uint16_t>& a_parts, const std::vector<uint16_t>& b_data,
    const std::vector<float>& golden)
{
    ensureDirExists(getInputDir());
    std::string prefix = getInputPrefix(nranks);
    std::string aFile = prefix + "_A.bin";
    std::string bFile = prefix + "_B.bin";
    std::string gFile = prefix + "_golden.bin";

    size_t a_total = (size_t)nranks * G_M * G_K;
    size_t b_total = (size_t)G_N * G_K;
    size_t g_total = (size_t)G_M * G_N;

    if (VerboseLog())
        printf("Saving input data to: %s_*.bin\n", prefix.c_str());

    if (!saveBinary(aFile, a_parts.data(), a_total)) {
        fprintf(stderr, "[WARN] Failed to save %s\n", aFile.c_str());
        return false;
    }
    if (!saveBinary(bFile, b_data.data(), b_total)) {
        fprintf(stderr, "[WARN] Failed to save %s\n", bFile.c_str());
        return false;
    }
    if (!saveBinary(gFile, golden.data(), g_total)) {
        fprintf(stderr, "[WARN] Failed to save %s\n", gFile.c_str());
        return false;
    }

    double a_mb = a_total * sizeof(uint16_t) / (1024.0 * 1024.0);
    double b_mb = b_total * sizeof(uint16_t) / (1024.0 * 1024.0);
    double g_mb = g_total * sizeof(float) / (1024.0 * 1024.0);
    if (VerboseLog())
        printf("  Saved: A=%.1f MB, B=%.1f MB, golden=%.1f MB\n", a_mb, b_mb, g_mb);
    return true;
}

static void convertFp32ToFp16Padded(
    const std::vector<std::vector<float>>& A_fp32_all, const std::vector<float>& B_fp32, int nranks,
    std::vector<uint16_t>& a_parts, std::vector<uint16_t>& b_data)
{
    size_t a_rank_elems = (size_t)G_M * G_K;
    size_t b_elems = (size_t)G_N * G_K;
    a_parts.assign((size_t)nranks * a_rank_elems, 0);
    b_data.assign(b_elems, 0);

    for (int r = 0; r < nranks; r++) {
        uint16_t* a_dst = a_parts.data() + (size_t)r * a_rank_elems;
        for (int i = 0; i < (int)G_ORIG_M; i++)
            for (int j = 0; j < (int)G_K; j++)
                a_dst[(size_t)i * G_K + j] = floatToHalf(A_fp32_all[r][(size_t)i * G_K + j]);
    }

    uint16_t* b_dst = b_data.data();
    for (int i = 0; i < (int)G_ORIG_N; i++)
        for (int j = 0; j < (int)G_K; j++)
            b_dst[(size_t)i * G_K + j] = floatToHalf(B_fp32[(size_t)j * G_ORIG_N + i]);
}

static void padGoldenToAligned(const std::vector<float>& golden_orig, std::vector<float>& golden)
{
    golden.assign((size_t)G_M * G_N, 0.0f);
    for (int i = 0; i < (int)G_ORIG_M; i++)
        memcpy_s(
            &golden[(size_t)i * G_N], G_N * sizeof(float), &golden_orig[(size_t)i * G_ORIG_N],
            G_ORIG_N * sizeof(float));
}

static bool generateData(
    int nranks, std::vector<uint16_t>& a_parts, std::vector<uint16_t>& b_data, std::vector<float>& golden)
{
    if (VerboseLog()) {
        printf(
            "Data Parallel: each rank has independent A[%d,%d], shared B[%d,%d], %d ranks\n", G_ORIG_M, G_K, G_K,
            G_ORIG_N, nranks);
    }

    std::mt19937 gen(42);
    float scale = std::sqrt(65000.0f / ((float)G_K * nranks * 4.0f));
    std::uniform_real_distribution<float> dist(-scale, scale);

    std::vector<std::vector<float>> A_fp32_all(nranks);
    for (int r = 0; r < nranks; r++) {
        A_fp32_all[r].resize((size_t)G_ORIG_M * G_K);
        for (auto& v : A_fp32_all[r])
            v = dist(gen);
    }
    std::vector<float> B_fp32((size_t)G_K * G_ORIG_N);
    for (auto& v : B_fp32)
        v = dist(gen);

    if (VerboseLog())
        printf("  Computing golden reference (sum of %d CPU GEMMs %d×%d×%d)...\n", nranks, G_ORIG_M, G_K, G_ORIG_N);
    auto t0 = std::chrono::high_resolution_clock::now();

    std::vector<float> golden_orig((size_t)G_ORIG_M * G_ORIG_N, 0.0f);
    std::vector<float> tmp((size_t)G_ORIG_M * G_ORIG_N);
    for (int r = 0; r < nranks; r++) {
        memset_s(tmp.data(), tmp.size() * sizeof(float), 0, tmp.size() * sizeof(float));
        computeGolden(A_fp32_all[r].data(), B_fp32.data(), tmp.data(), G_ORIG_M, G_K, G_ORIG_N);
        for (size_t i = 0; i < golden_orig.size(); i++)
            golden_orig[i] += tmp[i];
    }

    double secs = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();
    if (VerboseLog())
        printf("  Golden computed in %.1f s\n", secs);

    padGoldenToAligned(golden_orig, golden);
    convertFp32ToFp16Padded(A_fp32_all, B_fp32, nranks, a_parts, b_data);

    double gsum = 0.0;
    for (auto v : golden_orig)
        gsum += v;
    if (VerboseLog())
        printf("  Golden = sum(A_i × B): shape=(%d, %d), sum=%.2f\n", G_ORIG_M, G_ORIG_N, gsum);
    return true;
}

// ============================================================================
// Entry point
// ============================================================================

static int parseFirstDevice(int argc, char* argv[])
{
    const std::string& envVal = g_cached_env.first_device;
    if (!envVal.empty()) {
        int val = atoi(envVal.c_str());
        if (val >= 0)
            return val;
    }
    for (int i = 1; i < argc - 1; i++) {
        if (strcmp(argv[i], "--first-device") == 0) {
            int val = atoi(argv[i + 1]);
            if (val >= 0)
                return val;
        }
    }
    return 0;
}

static bool PrepareInputData(
    int n_ranks, std::vector<uint16_t>& a_parts, std::vector<uint16_t>& b_data, std::vector<float>& golden)
{
    size_t a_total = (size_t)n_ranks * G_M * G_K;
    size_t b_total = (size_t)G_N * G_K;
    size_t g_total = (size_t)G_M * G_N;

    int data_ok = 0;
    if (CommMpiRank() == 0) {
        if (inputFilesExist(n_ranks)) {
            if (VerboseLog())
                printf("[INFO] Found cached input files, loading...\n");
            if (!loadInputFiles(n_ranks, a_parts, b_data, golden))
                data_ok = 1;
        } else {
            if (VerboseLog())
                printf("[INFO] No cached input files, generating...\n");
            if (!generateData(n_ranks, a_parts, b_data, golden)) {
                data_ok = 1;
            } else {
                saveInputFiles(n_ranks, a_parts, b_data, golden);
            }
        }
    } else {
        a_parts.resize(a_total);
        b_data.resize(b_total);
        golden.resize(g_total);
    }

    CommMpiBcast(&data_ok, 1, COMM_MPI_INT, 0);
    if (data_ok != 0)
        return false;

    CommMpiBcast(a_parts.data(), (int)(a_total * sizeof(uint16_t)), COMM_MPI_CHAR, 0);
    CommMpiBcast(b_data.data(), (int)(b_total * sizeof(uint16_t)), COMM_MPI_CHAR, 0);
    CommMpiBcast(golden.data(), (int)(g_total * sizeof(float)), COMM_MPI_CHAR, 0);
    return true;
}

int main(int argc, char* argv[])
{
    InitCachedEnv();

    if (!CommMpiInit(&argc, &argv)) {
        fprintf(stderr, "[ERROR] MPI_Init failed. Launch with: mpirun -n <NRANKS> ./ccu_gemm_allreduce\n");
        return 1;
    }

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            printf("Usage: mpirun -n <NRANKS> %s [--first-device ID]\n", argv[0]);
            CommMpiFinalize();
            return 0;
        }
    }

    int n_ranks = CommMpiSize();
    int first_device_id = parseFirstDevice(argc, argv);

    if (CommMpiRank() == 0) {
        printf(
            "CCU GEMM AllReduce A5 FP16 (HCCL): ranks=%d, devices=[%d, %d)\n\n", n_ranks, first_device_id,
            first_device_id + n_ranks);
    }

    std::vector<uint16_t> a_parts;
    std::vector<uint16_t> b_data;
    std::vector<float> golden;

    if (!PrepareInputData(n_ranks, a_parts, b_data, golden)) {
        CommMpiFinalize();
        return 1;
    }

    bool ok = RunGemmAllReduce(n_ranks, first_device_id, a_parts.data(), b_data.data(), golden.data());

    if (CommMpiRank() == 0) {
        printf(ok ? "\nCCU GEMM AllReduce demo completed successfully.\n" : "\nCCU GEMM AllReduce demo FAILED.\n");
    }

    CommMpiFinalize();
    return ok ? 0 : 1;
}
