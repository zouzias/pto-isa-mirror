/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <pto/pto-inst.hpp>

#include "../common.hpp"
#include "tput_async_hns1825_kernel.h"
#include "pto/common/pto_tile.hpp"
#ifdef PTO_RDMA_SUPPORTED
#include "pto/comm/async/rdma/rdma_async_intrin.hpp"
#include "pto/comm/async/rdma/rdma_workspace_manager.hpp"
#include "hns1825_bootstrap.hpp"
#endif

#ifdef PTO_RDMA_SUPPORTED
template <typename... Args>
static void RoceTrace(uint32_t caseId, int rankId, Args&&... args)
{
    const char* verbose = std::getenv("PTO_ROCE_VERBOSE");
    if (verbose == nullptr || verbose[0] != '1') {
        return;
    }
    std::ostringstream os;
    os << "[RoCE][case " << caseId << "][rank " << rankId << "] ";
    (os << ... << std::forward<Args>(args));
    os << '\n';
    const std::string line = os.str();
    (void)std::fwrite(line.data(), 1, line.size(), stderr);
    std::fflush(stderr);
}

static uint64_t RoceElapsedUs(const std::chrono::steady_clock::time_point& start)
{
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count());
}

static std::string RoceHex(uint64_t value)
{
    std::ostringstream os;
    os << "0x" << std::hex << value;
    return os.str();
}

static uint32_t gRoceCaseSequence = 0;
#endif

// ============================================================================
// TPUT_ASYNC via RoCE RDMA (Hi1825 / HNS_1825) — device kernel.
//
// Data plane is a true one-sided remote write posted from AIV: root rank writes
// its send buffer into every peer's recv buffer. The symmetric communication
// buffer layout matches the URMA test:
//   [64 x int32 header][sendBuf: count x T][recvBuf: count x T]
// The remote target VA is computed from the peer's registered MR base VA
// (PeerMrBaseAddr) plus the recv-region offset.
// ============================================================================

#ifdef PTO_RDMA_SUPPORTED
constexpr uint32_t kRocePublicEventWaitError = 0x30000;
constexpr uint32_t kRocePublicEventTestError = 0x30001;

AICORE inline uint32_t CompleteRoceEvent(
    pto::comm::AsyncEvent& event, pto::comm::AsyncSession& session, RoceHns1825CompletionMode completionMode)
{
    if (completionMode != RoceHns1825CompletionMode::PUBLIC_EVENT_WAIT_TEST) {
        return pto::comm::rdma::WaitEventStatus(event.handle, session);
    }

    // The first Test may legitimately be false. Wait must complete the event,
    // and Test must then observe the already-consumed target index as complete.
    (void)event.Test(session);
    if (!event.Wait(session)) {
        return kRocePublicEventWaitError;
    }
    return event.Test(session) ? 0 : kRocePublicEventTestError;
}
#endif

template <typename T, size_t count>
[[bisheng::core_ratio(0, 1)]] __global__ AICORE void TPutAsyncHns1825KernelImpl(
    __gm__ T* localBuf, int nranks, int my_rank, int first_rank_id, int root_rank, int elem_offset, int elem_count,
    int operation_count, RoceHns1825CompletionMode completionMode, __gm__ uint8_t* rdmaWorkspace, uint32_t syncId)
{
    using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using Global = pto::GlobalTensor<T, ShapeDyn, StrideDyn, pto::Layout::ND>;
    // RoCE, like SDMA, needs a UB scratch tile for WQE staging / CQE polling.
    using ScratchTile = pto::Tile<pto::TileType::Vec, uint8_t, 1, pto::comm::sdma::UB_ALIGN_SIZE>;

    // Device→host status word in the 64×int32 header (word0). Host always prints it.
    // A5: cce::printf unsupported; AscendC::printf usually needs ASCENDC_DUMP=1 to flush.
    __gm__ uint32_t* devStatus = reinterpret_cast<__gm__ uint32_t*>(localBuf);
    *devStatus = 0;

    if (elem_count <= 0 || elem_offset < 0 || operation_count <= 0 ||
        static_cast<int64_t>(elem_offset) + static_cast<int64_t>(elem_count) * operation_count >
            static_cast<int64_t>(count)) {
        pipe_barrier(PIPE_ALL);
        return;
    }

    ShapeDyn shape(1, 1, 1, 1, elem_count);
    StrideDyn stride(elem_count, elem_count, elem_count, elem_count, 1);

    constexpr size_t kDataOffset = 64 * sizeof(int32_t);

    __gm__ T* sendBuf = reinterpret_cast<__gm__ T*>(reinterpret_cast<__gm__ uint8_t*>(localBuf) + kDataOffset);

    pipe_barrier(PIPE_ALL);

    if (my_rank == root_rank) {
#ifdef PTO_RDMA_SUPPORTED
        const int my_peer = my_rank - first_rank_id;
        ScratchTile scratchTile;
        TASSIGN(scratchTile, 0x0);
        pto::comm::AsyncSession session;
        if (!pto::comm::BuildAsyncSession<pto::comm::DmaEngine::RDMA>(
                scratchTile, rdmaWorkspace, static_cast<uint32_t>(my_peer), session, syncId)) {
            *devStatus = pto::comm::rdma::kRdmaSessionBuildError;
        } else {
            for (int target_peer = 0; target_peer < nranks; ++target_peer) {
                if (target_peer == my_peer) {
                    continue;
                }
                uint64_t peerBase = pto::comm::rdma::PeerMrBaseAddr(rdmaWorkspace, static_cast<uint32_t>(target_peer));

                pto::comm::AsyncEvent lastEvent;
                uint32_t completionStatus = 0;
                for (int operation = 0; operation < operation_count; ++operation) {
                    const int operationOffset = elem_offset + operation * elem_count;
                    __gm__ T* sendBufCore = sendBuf + operationOffset;
                    __gm__ T* remoteRecvBuf =
                        reinterpret_cast<__gm__ T*>(peerBase + kDataOffset) + count + operationOffset;
                    Global sendG(sendBufCore, shape, stride);
                    Global remoteRecvG(remoteRecvBuf, shape, stride);
                    lastEvent = pto::comm::TPUT_ASYNC<pto::comm::DmaEngine::RDMA>(
                        remoteRecvG, sendG, session, static_cast<uint32_t>(target_peer));
                    if (completionMode == RoceHns1825CompletionMode::STATUS_WAIT_EACH) {
                        completionStatus = CompleteRoceEvent(lastEvent, session, completionMode);
                        if (completionStatus != 0) {
                            break;
                        }
                    }
                }
                if (completionStatus == 0 && completionMode != RoceHns1825CompletionMode::STATUS_WAIT_EACH) {
                    completionStatus = CompleteRoceEvent(lastEvent, session, completionMode);
                }
                *devStatus = completionStatus;
                if (completionStatus != 0) {
                    break; // stop posting further peers; leave QP as-is for Finalize
                }
            }
        }
#else
        (void)first_rank_id;
        (void)rdmaWorkspace;
        (void)syncId;
#endif
    }

    pipe_barrier(PIPE_ALL);
}

#if defined(PTO_RDMA_SUPPORTED) && defined(PTO_RDMA_GET_TEST)
// Build the GET entry points for the tget_async_hns1825 target.
template <typename T, size_t count>
[[bisheng::core_ratio(0, 1)]] __global__ AICORE void TGetAsyncHns1825KernelImpl(
    __gm__ T* localBuf, int nranks, int my_rank, int first_rank_id, int root_rank, int elem_offset, int elem_count,
    int operation_count, RoceHns1825CompletionMode completionMode, __gm__ uint8_t* rdmaWorkspace, uint32_t syncId)
{
    constexpr size_t kDataOffset = 64 * sizeof(int32_t);
    using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using Global = pto::GlobalTensor<T, ShapeDyn, StrideDyn, pto::Layout::ND>;
    using ScratchTile = pto::Tile<pto::TileType::Vec, uint8_t, 1, pto::comm::sdma::UB_ALIGN_SIZE>;

    __gm__ uint32_t* devStatus = reinterpret_cast<__gm__ uint32_t*>(localBuf);
    *devStatus = 0;

    if (elem_count <= 0 || elem_offset < 0 || operation_count <= 0 ||
        static_cast<int64_t>(elem_offset) + static_cast<int64_t>(elem_count) * operation_count >
            static_cast<int64_t>(count)) {
        pipe_barrier(PIPE_ALL);
        return;
    }

    ShapeDyn shape(1, 1, 1, 1, elem_count);
    StrideDyn stride(elem_count, elem_count, elem_count, elem_count, 1);
    __gm__ T* sendBuf = reinterpret_cast<__gm__ T*>(reinterpret_cast<__gm__ uint8_t*>(localBuf) + kDataOffset);
    __gm__ T* recvBuf = sendBuf + count;

    pipe_barrier(PIPE_ALL);
    if (my_rank != root_rank) {
        pipe_barrier(PIPE_ALL);
        return;
    }

    const int myPeer = my_rank - first_rank_id;
    ScratchTile scratchTile;
    TASSIGN(scratchTile, 0x0);
    pto::comm::AsyncSession session;
    if (!pto::comm::BuildAsyncSession<pto::comm::DmaEngine::RDMA>(
            scratchTile, rdmaWorkspace, static_cast<uint32_t>(myPeer), session, syncId)) {
        *devStatus = pto::comm::rdma::kRdmaSessionBuildError;
    } else {
        for (int sourcePeer = 0; sourcePeer < nranks; ++sourcePeer) {
            if (sourcePeer == myPeer) {
                continue;
            }
            const uint64_t peerBase = pto::comm::rdma::PeerMrBaseAddr(rdmaWorkspace, static_cast<uint32_t>(sourcePeer));

            pto::comm::AsyncEvent lastEvent;
            uint32_t completionStatus = 0;
            for (int operation = 0; operation < operation_count; ++operation) {
                const int operationOffset = elem_offset + operation * elem_count;
                __gm__ T* remoteSend = reinterpret_cast<__gm__ T*>(peerBase + kDataOffset) + operationOffset;
                __gm__ T* localRecv = recvBuf + static_cast<size_t>(sourcePeer) * count + operationOffset;
                Global localRecvG(localRecv, shape, stride);
                Global remoteSendG(remoteSend, shape, stride);
                lastEvent = pto::comm::TGET_ASYNC<pto::comm::DmaEngine::RDMA>(
                    localRecvG, remoteSendG, session, static_cast<uint32_t>(sourcePeer));
                if (completionMode == RoceHns1825CompletionMode::STATUS_WAIT_EACH) {
                    completionStatus = CompleteRoceEvent(lastEvent, session, completionMode);
                    if (completionStatus != 0) {
                        break;
                    }
                }
            }
            if (completionStatus == 0 && completionMode != RoceHns1825CompletionMode::STATUS_WAIT_EACH) {
                completionStatus = CompleteRoceEvent(lastEvent, session, completionMode);
            }
            *devStatus = completionStatus;
            if (completionStatus != 0) {
                break;
            }
        }
    }

    pipe_barrier(PIPE_ALL);
}
#endif // PTO_RDMA_SUPPORTED && PTO_RDMA_GET_TEST

static bool AllRanksReady(bool localReady, int nRanks, const char* stage, bool* anyRankNotReady = nullptr)
{
    if (anyRankNotReady != nullptr) {
        *anyRankNotReady = false;
    }
    uint8_t local = localReady ? 1 : 0;
    std::vector<uint8_t> all(static_cast<size_t>(nRanks), 0);
    if (CommMpiAllgather(&local, 1, all.data(), 1) != 0) {
        std::cerr << "[ERROR] MPI_Allgather failed while agreeing on " << stage << std::endl;
        return false;
    }
    for (int rank = 0; rank < nRanks; ++rank) {
        if (all[rank] == 0) {
            if (anyRankNotReady != nullptr) {
                *anyRankNotReady = true;
            }
            std::cerr << "[RoCE] rank " << rank << " is not ready at stage: " << stage << std::endl;
        }
    }
    return anyRankNotReady == nullptr ? std::all_of(all.begin(), all.end(), [](uint8_t ready) { return ready != 0; }) :
                                        !*anyRankNotReady;
}

#ifdef PTO_RDMA_SUPPORTED
static pto::comm::rdma::WorkspaceInitResult AgreeOnRdmaPreflight(int nRanks)
{
    using pto::comm::rdma::WorkspaceInitResult;

    const auto localPreflight = pto::comm::rdma::RdmaWorkspaceManager::Preflight();
    const uint8_t localCode = static_cast<uint8_t>(localPreflight);
    std::vector<uint8_t> allCodes(static_cast<size_t>(nRanks), 0);
    if (CommMpiAllgather(&localCode, 1, allCodes.data(), 1) != 0) {
        std::cerr << "[ERROR] MPI_Allgather failed while agreeing on RDMA backend selection" << std::endl;
        return WorkspaceInitResult::ERROR;
    }

    const bool allDisabled = std::all_of(allCodes.begin(), allCodes.end(), [](uint8_t value) {
        return value == static_cast<uint8_t>(WorkspaceInitResult::DISABLED);
    });
    const bool allReady = std::all_of(allCodes.begin(), allCodes.end(), [](uint8_t value) {
        return value == static_cast<uint8_t>(WorkspaceInitResult::READY);
    });
    if (allDisabled) {
        if (CommMpiRank() == 0) {
            std::cerr << "[SKIP] no RDMA backend was compiled into this binary (compiled backend="
                      << pto::comm::rdma::RdmaWorkspaceManager::ConfiguredBackendName() << ")" << std::endl;
        }
        return WorkspaceInitResult::DISABLED;
    }
    if (!allReady) {
        if (CommMpiRank() == 0) {
            std::cerr << "[ERROR] inconsistent or unsupported RDMA backend selection across ranks" << std::endl;
        }
        return WorkspaceInitResult::ERROR;
    }
    return WorkspaceInitResult::READY;
}

// ============================================================================
// Out-of-band bootstrap for the current HNS_1825 backend.
//
// Per rank: phyId, RDMA NIC IPv4, symmetric-buffer base VA.
//   phyId    : ResolvePhyId(), else PTO_ROCE_PHYIDS[rank], else ACL device id
//   local IP : rootinfo CLOS entry, then HCOMM's fixed virtualTopology.xml, then the manual ST fallbacks
//              PTO_ROCE_LOCAL_IP (per process) / PTO_ROCE_IPS (MPI-rank-ordered list)
//   sym addr : MPI_Allgather of local symmetric buffer VA
// Missing local IP on any rank -> collective skip (not a false pass).
// Optional: PTO_ROCE_ROOTINFO, PTO_ROCE_LOCAL_IP, PTO_ROCE_IPS, PTO_ROCE_BASE_PORT, PTO_ROCE_VERBOSE=1
// ============================================================================
struct RoceBootstrap {
    bool skipped{false};
    std::vector<std::string> peerIps;
    std::vector<uint32_t> peerPhyIds;
    std::vector<uint64_t> peerSymAddrs;
    uint32_t phyId{0};
    uint16_t basePort{60032};

    static std::vector<std::string> SplitCsv(const char* env)
    {
        std::vector<std::string> out;
        if (env == nullptr) {
            return out;
        }
        std::stringstream ss(env);
        std::string item;
        while (std::getline(ss, item, ',')) {
            size_t b = item.find_first_not_of(" \t");
            size_t e = item.find_last_not_of(" \t");
            if (b != std::string::npos) {
                out.push_back(item.substr(b, e - b + 1));
            }
        }
        return out;
    }

    // rankId/nRanks — global rank index space (0-based when first_rank_id == 0).
    // deviceId      — this rank's ACL device id (phyId fallback).
    // symAddr       — this rank's symmetric buffer base VA.
    bool Init(int rankId, int nRanks, int deviceId, void* symAddr)
    {
        skipped = false;
        // ---- phyId: runtime resolution, env override, device-id fallback ----
        std::vector<std::string> phyStrs = SplitCsv(std::getenv("PTO_ROCE_PHYIDS"));
        if (static_cast<int>(phyStrs.size()) == nRanks) {
            phyId = static_cast<uint32_t>(std::strtoul(phyStrs[rankId].c_str(), nullptr, 10));
        } else if (!pto::comm::rdma::hns_1825::bootstrap::ResolvePhyId(phyId)) {
            phyId = static_cast<uint32_t>(deviceId);
            std::cerr << "[RoCE] phyId resolution unavailable, falling back to device id " << deviceId << std::endl;
        }

        // ---- local RDMA NIC IPv4: rootinfo, virtual topology, then IP env fallback ----
        std::string localIp;
        const char* rootPath = std::getenv("PTO_ROCE_ROOTINFO");
        if (rootPath == nullptr || rootPath[0] == '\0') {
            rootPath = pto::comm::rdma::hns_1825::bootstrap::kDefaultRootInfoPath;
        }
        if (!pto::comm::rdma::hns_1825::bootstrap::ResolveLocalRdmaIp(phyId, localIp, rootPath) &&
            !pto::comm::rdma::hns_1825::bootstrap::ResolveLocalRdmaIpFromVirtualTopology(phyId, localIp)) {
            const char* ipEnv = std::getenv("PTO_ROCE_LOCAL_IP");
            if (ipEnv != nullptr && ipEnv[0] != '\0') {
                localIp = ipEnv;
            } else {
                std::vector<std::string> ipList = SplitCsv(std::getenv("PTO_ROCE_IPS"));
                if (static_cast<int>(ipList.size()) == nRanks) {
                    localIp = ipList[rankId];
                }
            }
        }
        const bool localIpReady = !localIp.empty();
        if (!localIpReady) {
            std::cerr << "[SKIP] RoCE test could not resolve local RDMA IP for phyId " << phyId
                      << " (no rootinfo CLOS entry, no usable "
                      << pto::comm::rdma::hns_1825::bootstrap::kDefaultVirtualTopologyPath
                      << ", and no PTO_ROCE_LOCAL_IP / PTO_ROCE_IPS)" << std::endl;
        }
        bool anyIpMissing = false;
        if (!AllRanksReady(localIpReady, nRanks, "local RDMA IP resolution", &anyIpMissing)) {
            skipped = anyIpMissing;
            return false;
        }

        const char* portEnv = std::getenv("PTO_ROCE_BASE_PORT");
        bool portReady = true;
        if (portEnv != nullptr) {
            char* end = nullptr;
            unsigned long parsed = std::strtoul(portEnv, &end, 10);
            if (end == portEnv || *end != '\0' || parsed > UINT16_MAX) {
                std::cerr << "[ERROR] invalid PTO_ROCE_BASE_PORT='" << portEnv << "'" << std::endl;
                portReady = false;
            } else {
                basePort = static_cast<uint16_t>(parsed);
            }
        }
        if (!AllRanksReady(portReady, nRanks, "base-port parsing")) {
            return false;
        }
        std::vector<uint16_t> allPorts(static_cast<size_t>(nRanks), 0);
        if (CommMpiAllgather(
                &basePort, static_cast<int>(sizeof(basePort)), allPorts.data(), static_cast<int>(sizeof(basePort))) !=
            0) {
            std::cerr << "[ERROR] MPI_Allgather(basePort) failed" << std::endl;
            return false;
        }
        for (int rank = 0; rank < nRanks; ++rank) {
            if (allPorts[rank] != basePort) {
                std::cerr << "[ERROR] inconsistent PTO_ROCE_BASE_PORT: rank " << rank << " uses " << allPorts[rank]
                          << ", local rank uses " << basePort << std::endl;
                return false;
            }
        }

        // ---- allgather each rank's local IP into peerIps (fixed-width buffers) ----
        constexpr int kIpBufLen = 64;
        std::vector<char> localBuf(kIpBufLen, 0);
        std::snprintf(localBuf.data(), kIpBufLen, "%s", localIp.c_str());
        std::vector<char> allBuf(static_cast<size_t>(nRanks) * kIpBufLen, 0);
        if (CommMpiAllgather(localBuf.data(), kIpBufLen, allBuf.data(), kIpBufLen) != 0) {
            std::cerr << "[ERROR] MPI_Allgather(localIp) failed" << std::endl;
            return false;
        }
        peerIps.assign(static_cast<size_t>(nRanks), std::string());
        for (int r = 0; r < nRanks; ++r) {
            peerIps[r] = std::string(allBuf.data() + static_cast<size_t>(r) * kIpBufLen);
        }

        // ---- allgather physical device ids used by HCOMM's remote LinkData ----
        peerPhyIds.assign(static_cast<size_t>(nRanks), 0);
        if (CommMpiAllgather(
                &phyId, static_cast<int>(sizeof(phyId)), peerPhyIds.data(), static_cast<int>(sizeof(phyId))) != 0) {
            std::cerr << "[ERROR] MPI_Allgather(phyId) failed" << std::endl;
            return false;
        }

        // ---- allgather symmetric-buffer base VAs ----
        peerSymAddrs.assign(static_cast<size_t>(nRanks), 0);
        uint64_t localAddr = reinterpret_cast<uint64_t>(symAddr);
        if (CommMpiAllgather(
                &localAddr, static_cast<int>(sizeof(uint64_t)), peerSymAddrs.data(),
                static_cast<int>(sizeof(uint64_t))) != 0) {
            std::cerr << "[ERROR] MPI_Allgather(symAddr) failed" << std::endl;
            return false;
        }

        return true;
    }
};

// ============================================================================
// RoceTestContext: device + symmetric buffer + RoCE workspace manager.
//
// Each case owns a complete HCOMM lifecycle. Reusing the same port across the
// suite intentionally validates ChannelDestroy -> ChannelCreate reconnect.
// ============================================================================
struct RoceTestContext {
    int deviceId{-1};
    int rankId{-1};
    int nRanks{0};
    rtStream_t stream{nullptr};
    void* devBuf{nullptr};
    size_t allocSize{0};
    pto::comm::rdma::RdmaWorkspaceManager rdmaMgr;
    RoceBootstrap boot;
    uint32_t traceId{0};

    enum class SetupResult {
        READY,
        FAILED,
        SKIPPED,
    };

    SetupResult Setup(
        uint32_t caseId, int rank_id, int n_ranks, int n_devices, int first_device_id, size_t commBytesNeeded)
    {
        const auto setupStart = std::chrono::steady_clock::now();
        traceId = caseId;
        rankId = rank_id;
        nRanks = n_ranks;
        deviceId = rank_id % n_devices + first_device_id;
        rdmaMgr.SetTraceId(traceId);
        RoceTrace(traceId, rankId, "SETUP begin device=", deviceId, " commBytes=", commBytesNeeded);

        bool localReady = true;
        if (aclrtSetDevice(deviceId) != ACL_SUCCESS) {
            std::cerr << "[ERROR] aclrtSetDevice(" << deviceId << ") failed" << std::endl;
            localReady = false;
        }
        if (localReady && rtStreamCreate(&stream, RT_STREAM_PRIORITY_DEFAULT) != 0) {
            std::cerr << "[ERROR] rtStreamCreate failed" << std::endl;
            localReady = false;
        }

        allocSize = commBytesNeeded;
        if (localReady &&
            (aclrtMalloc(&devBuf, allocSize, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS || devBuf == nullptr)) {
            std::cerr << "[ERROR] aclrtMalloc(" << allocSize << ") failed" << std::endl;
            localReady = false;
        }
        if (localReady && aclrtMemset(devBuf, allocSize, 0, allocSize) != ACL_SUCCESS) {
            std::cerr << "[ERROR] aclrtMemset failed" << std::endl;
            localReady = false;
        }
        if (!AllRanksReady(localReady, n_ranks, "local ACL resource setup")) {
            return SetupResult::FAILED;
        }
        RoceTrace(
            traceId, rankId, "SETUP local resources ready stream=", stream, " devBuf=", devBuf,
            " allocSize=", allocSize, " elapsed_us=", RoceElapsedUs(setupStart));

        if (!boot.Init(rank_id, n_ranks, deviceId, devBuf)) {
            return boot.skipped ? SetupResult::SKIPPED : SetupResult::FAILED;
        }
        std::ostringstream peers;
        for (int peer = 0; peer < n_ranks; ++peer) {
            if (peer != 0) {
                peers << ',';
            }
            peers << peer << ':' << boot.peerIps[peer] << "/phy" << boot.peerPhyIds[peer] << "/sym"
                  << RoceHex(boot.peerSymAddrs[peer]);
        }
        RoceTrace(traceId, rankId, "SETUP bootstrap ready basePort=", boot.basePort, " peers=[", peers.str(), ']');

        CommMpiBarrier();
        const auto connectStart = std::chrono::steady_clock::now();
        pto::comm::rdma::WorkspaceConfig rdmaConfig{};
        rdmaConfig.rankId = static_cast<uint32_t>(rank_id);
        rdmaConfig.rankCount = static_cast<uint32_t>(n_ranks);
        rdmaConfig.phyId = boot.phyId;
        rdmaConfig.localIp = boot.peerIps[rank_id];
        rdmaConfig.basePort = boot.basePort;
        rdmaConfig.peerIps = boot.peerIps;
        rdmaConfig.peerPhyIds = boot.peerPhyIds;
        rdmaConfig.peerSymAddrs = boot.peerSymAddrs;
        rdmaConfig.symmetricAddr = devBuf;
        rdmaConfig.symmetricSize = allocSize;
        const auto initResult = rdmaMgr.Init(rdmaConfig);
        const bool localConnected = initResult == pto::comm::rdma::WorkspaceInitResult::READY;
        if (!localConnected) {
            std::cerr << "[ERROR] RDMA workspace initialization failed" << std::endl;
        }
        if (!AllRanksReady(localConnected, n_ranks, "HCOMM channel initialization")) {
            return SetupResult::FAILED;
        }
        RoceTrace(
            traceId, rankId, "SETUP HCOMM ready connect_us=", RoceElapsedUs(connectStart),
            " total_us=", RoceElapsedUs(setupStart));
        return SetupResult::READY;
    }

    bool Cleanup()
    {
        const auto cleanupStart = std::chrono::steady_clock::now();
        RoceTrace(traceId, rankId, "CLEANUP begin devBuf=", devBuf, " stream=", stream);
        CommMpiBarrier();
        bool localOk = rdmaMgr.Finalize();
        if (devBuf != nullptr) {
            aclError ret = aclrtFree(devBuf);
            if (ret != ACL_SUCCESS) {
                std::cerr << "[ERROR] aclrtFree(symmetric buffer) failed: " << static_cast<int>(ret) << std::endl;
                localOk = false;
            } else {
                devBuf = nullptr;
            }
        }
        if (stream != nullptr) {
            rtError_t ret = rtStreamDestroy(stream);
            if (ret != 0) {
                std::cerr << "[ERROR] rtStreamDestroy failed: " << ret << std::endl;
                localOk = false;
            }
            stream = nullptr;
        }
        const bool allOk = AllRanksReady(localOk, nRanks, "RDMA test resource cleanup");
        CommMpiBarrier();
        RoceTrace(
            traceId, rankId, "CLEANUP end localOk=", localOk, " allOk=", allOk,
            " elapsed_us=", RoceElapsedUs(cleanupStart));
        return allOk;
    }
};
#endif // PTO_RDMA_SUPPORTED

template <typename T>
static T RoceInputValue(size_t index, int rankId)
{
    if constexpr (sizeof(T) == sizeof(uint8_t)) {
        return static_cast<T>(((index * 131U) ^ (index >> 8U) ^ (static_cast<size_t>(rankId) * 17U)) & 0xffU);
    }
    return static_cast<T>(index + static_cast<size_t>(rankId) * 10000U);
}

template <typename T>
static T RoceSentinelValue(size_t index)
{
    if constexpr (sizeof(T) == sizeof(uint8_t)) {
        return static_cast<T>((0xa5U ^ (index * 29U) ^ (index >> 8U)) & 0xffU);
    }
    return static_cast<T>(-1);
}

static const char* RoceCompletionModeName(RoceHns1825CompletionMode mode)
{
    switch (mode) {
        case RoceHns1825CompletionMode::STATUS_WAIT_EACH:
            return "status-wait-each";
        case RoceHns1825CompletionMode::STATUS_WAIT_LAST:
            return "status-wait-last";
        case RoceHns1825CompletionMode::PUBLIC_EVENT_WAIT_TEST:
            return "public-event-wait-test";
        default:
            return "unknown";
    }
}

#ifdef PTO_RDMA_SUPPORTED
static void PrintRoceDeviceStatus(const char* operation, int rankId, int deviceId, int syncRet, uint32_t devStatus)
{
    std::cerr << "[RoCE] " << operation << " Rank " << rankId << " Device " << deviceId << " SyncRet " << syncRet
              << " DevStatus 0x" << std::hex << devStatus << std::dec;
    if (devStatus == pto::comm::rdma::hns_1825::kHns1825PollCqTimeoutError) {
        std::cerr << " (poll_cq_timeout)";
    } else if (devStatus == pto::comm::rdma::kRdmaSessionBuildError) {
        std::cerr << " (session_build_fail)";
    } else if (devStatus == pto::comm::rdma::hns_1825::kHns1825InvalidArgumentError) {
        std::cerr << " (invalid_transfer)";
    } else if (devStatus == pto::comm::rdma::hns_1825::kHns1825InvalidContextError) {
        std::cerr << " (invalid_context)";
    } else if (devStatus == pto::comm::rdma::hns_1825::kHns1825CqeError) {
        std::cerr << " (cqe_error_without_syndrome)";
    } else if (devStatus == kRocePublicEventWaitError) {
        std::cerr << " (public_event_wait_failed)";
    } else if (devStatus == kRocePublicEventTestError) {
        std::cerr << " (public_event_test_after_wait_failed)";
    } else if (devStatus != 0) {
        std::cerr << " (cqe_syndrome_or_error)";
    }
    std::cerr << std::endl;
}
#endif

// ============================================================================
// Host-side runner.
// ============================================================================
template <typename T, size_t count>
RoceHns1825TestResult RunPutAsyncHns1825RootPutKernel(
    int rank_id, int n_ranks, int n_devices, int first_device_id, int first_rank_id, int root_rank, uint32_t caseId,
    int elemOffset, int elemCount, int operationCount, RoceHns1825CompletionMode completionMode)
{
#ifndef PTO_RDMA_SUPPORTED
    (void)rank_id;
    (void)n_ranks;
    (void)n_devices;
    (void)first_device_id;
    (void)first_rank_id;
    (void)root_rank;
    (void)caseId;
    (void)elemOffset;
    (void)elemCount;
    (void)operationCount;
    (void)completionMode;
    std::cerr << "[SKIP] built without PTO_RDMA_SUPPORTED" << std::endl;
    return RoceHns1825TestResult::SKIPPED;
#else
    const auto caseStart = std::chrono::steady_clock::now();
    CommMpiBarrier();
    RoceTrace(
        caseId, rank_id, "CASE begin op=PUT elemSize=", sizeof(T), " count=", count, " bytes=", count * sizeof(T),
        " offset=", elemOffset, " elemsPerOp=", elemCount, " operations=", operationCount,
        " completion=", RoceCompletionModeName(completionMode));
    const bool localPlanValid = elemOffset >= 0 && elemCount > 0 && operationCount > 0 &&
                                static_cast<int64_t>(elemOffset) + static_cast<int64_t>(elemCount) * operationCount <=
                                    static_cast<int64_t>(count);
    if (!AllRanksReady(localPlanValid, n_ranks, "PUT transfer plan validation")) {
        return RoceHns1825TestResult::FAILED;
    }

    const size_t commBytesNeeded = 64 * sizeof(int32_t) + 2 * count * sizeof(T);
    RoceTestContext ctx;
    const auto setupStart = std::chrono::steady_clock::now();
    RoceTestContext::SetupResult setup =
        ctx.Setup(caseId, rank_id - first_rank_id, n_ranks, n_devices, first_device_id, commBytesNeeded);
    RoceTrace(
        caseId, rank_id, "CASE setup result=", static_cast<int>(setup), " elapsed_us=", RoceElapsedUs(setupStart));
    if (setup != RoceTestContext::SetupResult::READY) {
        const bool cleanupOk = ctx.Cleanup();
        if (!cleanupOk || setup == RoceTestContext::SetupResult::FAILED) {
            return RoceHns1825TestResult::FAILED;
        }
        return RoceHns1825TestResult::SKIPPED;
    }

    uint8_t* input_host = nullptr;
    uint8_t* output_host = nullptr;
    bool localOk = aclrtMallocHost(reinterpret_cast<void**>(&input_host), count * sizeof(T)) == ACL_SUCCESS &&
                   input_host != nullptr;
    localOk = (aclrtMallocHost(reinterpret_cast<void**>(&output_host), count * sizeof(T)) == ACL_SUCCESS &&
               output_host != nullptr) &&
              localOk;
    if (!localOk) {
        std::cerr << "[ERROR] aclrtMallocHost failed!" << std::endl;
    }
    if (!AllRanksReady(localOk, n_ranks, "host staging allocation")) {
        if (input_host != nullptr) {
            (void)aclrtFreeHost(input_host);
        }
        if (output_host != nullptr) {
            (void)aclrtFreeHost(output_host);
        }
        (void)ctx.Cleanup();
        return RoceHns1825TestResult::FAILED;
    }

    for (size_t i = 0; i < count; ++i) {
        reinterpret_cast<T*>(input_host)[i] = RoceInputValue<T>(i, rank_id);
        reinterpret_cast<T*>(output_host)[i] = RoceSentinelValue<T>(i);
    }

    constexpr size_t kDataOffset = 64 * sizeof(int32_t);
    uint8_t* commBytes = reinterpret_cast<uint8_t*>(ctx.devBuf);
    T* sendBuf = reinterpret_cast<T*>(commBytes + kDataOffset);
    T* recvBuf = sendBuf + count;

    localOk = aclrtMemcpy(sendBuf, count * sizeof(T), input_host, count * sizeof(T), ACL_MEMCPY_HOST_TO_DEVICE) ==
              ACL_SUCCESS;
    localOk = (aclrtMemcpy(recvBuf, count * sizeof(T), output_host, count * sizeof(T), ACL_MEMCPY_HOST_TO_DEVICE) ==
               ACL_SUCCESS) &&
              localOk;
    uint32_t zero = 0;
    localOk = (aclrtMemcpy(ctx.devBuf, sizeof(zero), &zero, sizeof(zero), ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS) &&
              localOk;
    if (!AllRanksReady(localOk, n_ranks, "PUT input initialization")) {
        (void)aclrtFreeHost(input_host);
        (void)aclrtFreeHost(output_host);
        (void)ctx.Cleanup();
        return RoceHns1825TestResult::FAILED;
    }

    CommMpiBarrier();

    const auto kernelStart = std::chrono::steady_clock::now();
    TPutAsyncHns1825KernelImpl<T, count><<<1, nullptr, ctx.stream> > >(
        reinterpret_cast<T*>(ctx.devBuf), n_ranks, rank_id, first_rank_id, root_rank, elemOffset, elemCount,
        operationCount, completionMode, reinterpret_cast<uint8_t*>(ctx.rdmaMgr.GetWorkspaceAddr()), 0);
    int syncRet = aclrtSynchronizeStream(ctx.stream);

    CommMpiBarrier();
    RoceTrace(
        caseId, rank_id, "CASE kernel synchronized syncRet=", syncRet, " elapsed_us=", RoceElapsedUs(kernelStart));
    uint32_t devStatus = 0;
    localOk = aclrtMemcpy(&devStatus, sizeof(devStatus), ctx.devBuf, sizeof(devStatus), ACL_MEMCPY_DEVICE_TO_HOST) ==
              ACL_SUCCESS;
    PrintRoceDeviceStatus("PUT", rank_id, ctx.deviceId, syncRet, devStatus);

    localOk = (aclrtMemcpy(output_host, count * sizeof(T), recvBuf, count * sizeof(T), ACL_MEMCPY_DEVICE_TO_HOST) ==
               ACL_SUCCESS) &&
              localOk;

    bool is_ok = localOk && (syncRet == 0) && (devStatus == 0);
    const size_t writeBegin = static_cast<size_t>(elemOffset);
    const size_t writeEnd = writeBegin + static_cast<size_t>(elemCount) * static_cast<size_t>(operationCount);
    for (size_t i = 0; i < count && is_ok; ++i) {
        const bool shouldReceive = rank_id != root_rank && i >= writeBegin && i < writeEnd;
        const T expected = shouldReceive ? RoceInputValue<T>(i, root_rank) : RoceSentinelValue<T>(i);
        const T value = reinterpret_cast<T*>(output_host)[i];
        if (value != expected) {
            std::cerr << "PUT Rank " << rank_id << " Device " << ctx.deviceId << " SyncRet " << syncRet << " Index "
                      << i << " Expected: " << static_cast<float>(expected) << " Actual: " << static_cast<float>(value)
                      << std::endl;
            is_ok = false;
        }
    }

    is_ok = AllRanksReady(is_ok, n_ranks, "PUT result verification");
    bool hostFreed = aclrtFreeHost(input_host) == ACL_SUCCESS;
    hostFreed = (aclrtFreeHost(output_host) == ACL_SUCCESS) && hostFreed;
    const bool allHostFreed = AllRanksReady(hostFreed, n_ranks, "host staging release");
    is_ok = is_ok && allHostFreed;
    const auto cleanupStart = std::chrono::steady_clock::now();
    const bool cleanupOk = ctx.Cleanup();
    RoceTrace(
        caseId, rank_id, "CASE end dataOk=", is_ok, " cleanupOk=", cleanupOk,
        " cleanup_us=", RoceElapsedUs(cleanupStart), " total_us=", RoceElapsedUs(caseStart));
    return (is_ok && cleanupOk) ? RoceHns1825TestResult::PASSED : RoceHns1825TestResult::FAILED;
#endif // PTO_RDMA_SUPPORTED
}

#ifdef PTO_RDMA_GET_TEST
template <typename T, size_t count>
RoceHns1825TestResult RunGetAsyncHns1825RootGetKernel(
    int rank_id, int n_ranks, int n_devices, int first_device_id, int first_rank_id, int root_rank, uint32_t caseId,
    int elemOffset, int elemCount, int operationCount, RoceHns1825CompletionMode completionMode)
{
#ifndef PTO_RDMA_SUPPORTED
    (void)rank_id;
    (void)n_ranks;
    (void)n_devices;
    (void)first_device_id;
    (void)first_rank_id;
    (void)root_rank;
    (void)caseId;
    (void)elemOffset;
    (void)elemCount;
    (void)operationCount;
    (void)completionMode;
    std::cerr << "[SKIP] built without PTO_RDMA_SUPPORTED" << std::endl;
    return RoceHns1825TestResult::SKIPPED;
#else
    const auto caseStart = std::chrono::steady_clock::now();
    CommMpiBarrier();
    RoceTrace(
        caseId, rank_id, "CASE begin op=GET elemSize=", sizeof(T), " count=", count, " bytes=", count * sizeof(T),
        " offset=", elemOffset, " elemsPerOp=", elemCount, " operations=", operationCount,
        " completion=", RoceCompletionModeName(completionMode));
    const bool localPlanValid = elemOffset >= 0 && elemCount > 0 && operationCount > 0 &&
                                static_cast<int64_t>(elemOffset) + static_cast<int64_t>(elemCount) * operationCount <=
                                    static_cast<int64_t>(count);
    if (!AllRanksReady(localPlanValid, n_ranks, "GET transfer plan validation")) {
        return RoceHns1825TestResult::FAILED;
    }

    const size_t recvElems = static_cast<size_t>(n_ranks) * count;
    const size_t commBytesNeeded = 64 * sizeof(int32_t) + (static_cast<size_t>(n_ranks) + 1) * count * sizeof(T);
    RoceTestContext ctx;
    const auto setupStart = std::chrono::steady_clock::now();
    RoceTestContext::SetupResult setup =
        ctx.Setup(caseId, rank_id - first_rank_id, n_ranks, n_devices, first_device_id, commBytesNeeded);
    RoceTrace(
        caseId, rank_id, "CASE setup result=", static_cast<int>(setup), " elapsed_us=", RoceElapsedUs(setupStart));
    if (setup != RoceTestContext::SetupResult::READY) {
        const bool cleanupOk = ctx.Cleanup();
        if (!cleanupOk || setup == RoceTestContext::SetupResult::FAILED) {
            return RoceHns1825TestResult::FAILED;
        }
        return RoceHns1825TestResult::SKIPPED;
    }

    T* inputHost = nullptr;
    T* outputHost = nullptr;
    bool localOk =
        aclrtMallocHost(reinterpret_cast<void**>(&inputHost), count * sizeof(T)) == ACL_SUCCESS && inputHost != nullptr;
    localOk = (aclrtMallocHost(reinterpret_cast<void**>(&outputHost), recvElems * sizeof(T)) == ACL_SUCCESS &&
               outputHost != nullptr) &&
              localOk;
    if (!localOk) {
        std::cerr << "[ERROR] aclrtMallocHost failed!" << std::endl;
    }
    if (!AllRanksReady(localOk, n_ranks, "GET host staging allocation")) {
        if (inputHost != nullptr) {
            (void)aclrtFreeHost(inputHost);
        }
        if (outputHost != nullptr) {
            (void)aclrtFreeHost(outputHost);
        }
        (void)ctx.Cleanup();
        return RoceHns1825TestResult::FAILED;
    }

    for (size_t i = 0; i < count; ++i) {
        inputHost[i] = RoceInputValue<T>(i, rank_id);
    }
    for (size_t i = 0; i < recvElems; ++i) {
        outputHost[i] = RoceSentinelValue<T>(i % count);
    }

    constexpr size_t kDataOffset = 64 * sizeof(int32_t);
    uint8_t* commBytes = reinterpret_cast<uint8_t*>(ctx.devBuf);
    T* sendBuf = reinterpret_cast<T*>(commBytes + kDataOffset);
    T* recvBuf = sendBuf + count;
    localOk =
        aclrtMemcpy(sendBuf, count * sizeof(T), inputHost, count * sizeof(T), ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS;
    localOk =
        (aclrtMemcpy(recvBuf, recvElems * sizeof(T), outputHost, recvElems * sizeof(T), ACL_MEMCPY_HOST_TO_DEVICE) ==
         ACL_SUCCESS) &&
        localOk;
    uint32_t zero = 0;
    localOk = (aclrtMemcpy(ctx.devBuf, sizeof(zero), &zero, sizeof(zero), ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS) &&
              localOk;
    if (!AllRanksReady(localOk, n_ranks, "GET input initialization")) {
        (void)aclrtFreeHost(inputHost);
        (void)aclrtFreeHost(outputHost);
        (void)ctx.Cleanup();
        return RoceHns1825TestResult::FAILED;
    }

    CommMpiBarrier();
    const auto kernelStart = std::chrono::steady_clock::now();
    TGetAsyncHns1825KernelImpl<T, count><<<1, nullptr, ctx.stream> > >(
        reinterpret_cast<T*>(ctx.devBuf), n_ranks, rank_id, first_rank_id, root_rank, elemOffset, elemCount,
        operationCount, completionMode, reinterpret_cast<uint8_t*>(ctx.rdmaMgr.GetWorkspaceAddr()), 0);
    const int syncRet = aclrtSynchronizeStream(ctx.stream);

    CommMpiBarrier();
    RoceTrace(
        caseId, rank_id, "CASE GET kernel synchronized syncRet=", syncRet, " elapsed_us=", RoceElapsedUs(kernelStart));
    uint32_t devStatus = 0;
    localOk = aclrtMemcpy(&devStatus, sizeof(devStatus), ctx.devBuf, sizeof(devStatus), ACL_MEMCPY_DEVICE_TO_HOST) ==
              ACL_SUCCESS;
    PrintRoceDeviceStatus("GET", rank_id, ctx.deviceId, syncRet, devStatus);
    localOk =
        (aclrtMemcpy(outputHost, recvElems * sizeof(T), recvBuf, recvElems * sizeof(T), ACL_MEMCPY_DEVICE_TO_HOST) ==
         ACL_SUCCESS) &&
        localOk;

    bool isOk = localOk && syncRet == 0 && devStatus == 0;
    const size_t readBegin = static_cast<size_t>(elemOffset);
    const size_t readEnd = readBegin + static_cast<size_t>(elemCount) * static_cast<size_t>(operationCount);
    const int rootPeer = root_rank - first_rank_id;
    for (int sourcePeer = 0; sourcePeer < n_ranks && isOk; ++sourcePeer) {
        for (size_t i = 0; i < count; ++i) {
            const bool shouldReceive = rank_id == root_rank && sourcePeer != rootPeer && i >= readBegin && i < readEnd;
            const T expected =
                shouldReceive ? RoceInputValue<T>(i, first_rank_id + sourcePeer) : RoceSentinelValue<T>(i);
            const T value = outputHost[static_cast<size_t>(sourcePeer) * count + i];
            if (value != expected) {
                std::cerr << "GET Rank " << rank_id << " Device " << ctx.deviceId << " SourcePeer " << sourcePeer
                          << " Index " << i << " Expected: " << static_cast<float>(expected)
                          << " Actual: " << static_cast<float>(value) << std::endl;
                isOk = false;
                break;
            }
        }
    }

    isOk = AllRanksReady(isOk, n_ranks, "GET result verification");
    bool hostFreed = aclrtFreeHost(inputHost) == ACL_SUCCESS;
    hostFreed = (aclrtFreeHost(outputHost) == ACL_SUCCESS) && hostFreed;
    const bool allHostFreed = AllRanksReady(hostFreed, n_ranks, "GET host staging release");
    isOk = isOk && allHostFreed;
    const auto cleanupStart = std::chrono::steady_clock::now();
    const bool cleanupOk = ctx.Cleanup();
    RoceTrace(
        caseId, rank_id, "CASE end op=GET dataOk=", isOk, " cleanupOk=", cleanupOk,
        " cleanup_us=", RoceElapsedUs(cleanupStart), " total_us=", RoceElapsedUs(caseStart));
    return (isOk && cleanupOk) ? RoceHns1825TestResult::PASSED : RoceHns1825TestResult::FAILED;
#endif // PTO_RDMA_SUPPORTED
}
#endif // PTO_RDMA_GET_TEST

// ============================================================================
// MPI-based multi-rank launch with explicit pass/fail/skip propagation.
// ============================================================================
template <typename T, size_t count>
RoceHns1825TestResult RunPutAsyncHns1825RootPut(int n_ranks, int n_devices, int first_rank_id, int first_device_id)
{
    return RunPutAsyncHns1825RootPutPlan<T, count>(
        n_ranks, n_devices, first_rank_id, first_device_id, 0, static_cast<int>(count), 1,
        RoceHns1825CompletionMode::STATUS_WAIT_EACH);
}

template <typename T, size_t count>
RoceHns1825TestResult RunPutAsyncHns1825RootPutPlan(
    int n_ranks, int n_devices, int first_rank_id, int first_device_id, int elem_offset, int elem_count,
    int operation_count, RoceHns1825CompletionMode completion_mode)
{
    const int mpiRank = CommMpiRank();
    const int mpiSize = CommMpiSize();
    if (n_ranks <= 0 || n_devices <= 0 || mpiSize != n_ranks) {
        if (mpiRank == 0) {
            std::cerr << "[ERROR] invalid launch configuration: mpiSize=" << mpiSize << " nRanks=" << n_ranks
                      << " nDevices=" << n_devices << std::endl;
        }
        return RoceHns1825TestResult::FAILED;
    }

#ifndef PTO_RDMA_SUPPORTED
    (void)first_rank_id;
    (void)first_device_id;
    (void)elem_offset;
    (void)elem_count;
    (void)operation_count;
    (void)completion_mode;
    if (mpiRank == 0) {
        std::cerr << "[SKIP] built without PTO_RDMA_SUPPORTED" << std::endl;
    }
    return RoceHns1825TestResult::SKIPPED;
#else
    const auto rdmaPreflight = AgreeOnRdmaPreflight(n_ranks);
    if (rdmaPreflight == pto::comm::rdma::WorkspaceInitResult::DISABLED) {
        return RoceHns1825TestResult::SKIPPED;
    }
    if (rdmaPreflight != pto::comm::rdma::WorkspaceInitResult::READY) {
        return RoceHns1825TestResult::FAILED;
    }

    const int deviceCount = GetAvailableDeviceCount();
    const bool localDevicesReady = deviceCount >= n_devices + first_device_id;
    bool anyDeviceMissing = false;
    if (!AllRanksReady(localDevicesReady, n_ranks, "device availability", &anyDeviceMissing)) {
        if (mpiRank == 0) {
            std::cerr << "[SKIP] Need " << (n_devices + first_device_id) << " NPU(s), have " << deviceCount
                      << std::endl;
        }
        return anyDeviceMissing ? RoceHns1825TestResult::SKIPPED : RoceHns1825TestResult::FAILED;
    }

    constexpr int kAclRepeatInit = 100002;
    const aclError aclRet = aclInit(nullptr);
    const bool aclReady = aclRet == ACL_SUCCESS || static_cast<int>(aclRet) == kAclRepeatInit;
    if (!aclReady) {
        std::cerr << "[ERROR] aclInit failed: " << static_cast<int>(aclRet) << std::endl;
    }
    if (!AllRanksReady(aclReady, n_ranks, "ACL initialization")) {
        return RoceHns1825TestResult::FAILED;
    }

    const int rankId = first_rank_id + mpiRank;
    const int rootRank = first_rank_id;
    const uint32_t caseId = ++gRoceCaseSequence;
    return RunPutAsyncHns1825RootPutKernel<T, count>(
        rankId, n_ranks, n_devices, first_device_id, first_rank_id, rootRank, caseId, elem_offset, elem_count,
        operation_count, completion_mode);
#endif
}

#ifdef PTO_RDMA_GET_TEST
template <typename T, size_t count>
RoceHns1825TestResult RunGetAsyncHns1825RootGetPlan(
    int n_ranks, int n_devices, int first_rank_id, int first_device_id, int elem_offset, int elem_count,
    int operation_count, RoceHns1825CompletionMode completion_mode)
{
    const int mpiRank = CommMpiRank();
    const int mpiSize = CommMpiSize();
    if (n_ranks <= 0 || n_devices <= 0 || mpiSize != n_ranks) {
        if (mpiRank == 0) {
            std::cerr << "[ERROR] invalid GET launch configuration: mpiSize=" << mpiSize << " nRanks=" << n_ranks
                      << " nDevices=" << n_devices << std::endl;
        }
        return RoceHns1825TestResult::FAILED;
    }

#ifndef PTO_RDMA_SUPPORTED
    (void)first_rank_id;
    (void)first_device_id;
    (void)elem_offset;
    (void)elem_count;
    (void)operation_count;
    (void)completion_mode;
    if (mpiRank == 0) {
        std::cerr << "[SKIP] built without PTO_RDMA_SUPPORTED" << std::endl;
    }
    return RoceHns1825TestResult::SKIPPED;
#else
    const auto rdmaPreflight = AgreeOnRdmaPreflight(n_ranks);
    if (rdmaPreflight == pto::comm::rdma::WorkspaceInitResult::DISABLED) {
        return RoceHns1825TestResult::SKIPPED;
    }
    if (rdmaPreflight != pto::comm::rdma::WorkspaceInitResult::READY) {
        return RoceHns1825TestResult::FAILED;
    }

    const int deviceCount = GetAvailableDeviceCount();
    const bool localDevicesReady = deviceCount >= n_devices + first_device_id;
    bool anyDeviceMissing = false;
    if (!AllRanksReady(localDevicesReady, n_ranks, "GET device availability", &anyDeviceMissing)) {
        if (mpiRank == 0) {
            std::cerr << "[SKIP] GET needs " << (n_devices + first_device_id) << " NPU(s), have " << deviceCount
                      << std::endl;
        }
        return anyDeviceMissing ? RoceHns1825TestResult::SKIPPED : RoceHns1825TestResult::FAILED;
    }

    constexpr int kAclRepeatInit = 100002;
    const aclError aclRet = aclInit(nullptr);
    const bool aclReady = aclRet == ACL_SUCCESS || static_cast<int>(aclRet) == kAclRepeatInit;
    if (!aclReady) {
        std::cerr << "[ERROR] aclInit failed before GET: " << static_cast<int>(aclRet) << std::endl;
    }
    if (!AllRanksReady(aclReady, n_ranks, "GET ACL initialization")) {
        return RoceHns1825TestResult::FAILED;
    }

    const int rankId = first_rank_id + mpiRank;
    const int rootRank = first_rank_id;
    const uint32_t caseId = ++gRoceCaseSequence;
    return RunGetAsyncHns1825RootGetKernel<T, count>(
        rankId, n_ranks, n_devices, first_device_id, first_rank_id, rootRank, caseId, elem_offset, elem_count,
        operation_count, completion_mode);
#endif
}
#endif // PTO_RDMA_GET_TEST

// Explicit instantiations
template RoceHns1825TestResult RunPutAsyncHns1825RootPut<float, 256>(int, int, int, int);
template RoceHns1825TestResult RunPutAsyncHns1825RootPut<int32_t, 4096>(int, int, int, int);
template RoceHns1825TestResult RunPutAsyncHns1825RootPut<uint8_t, 512>(int, int, int, int);
template RoceHns1825TestResult RunPutAsyncHns1825RootPut<uint8_t, 64>(int, int, int, int);
template RoceHns1825TestResult RunPutAsyncHns1825RootPut<float, 64>(int, int, int, int);
template RoceHns1825TestResult RunPutAsyncHns1825RootPut<float, 524288>(int, int, int, int);

template RoceHns1825TestResult RunPutAsyncHns1825RootPutPlan<float, 256>(
    int, int, int, int, int, int, int, RoceHns1825CompletionMode);
template RoceHns1825TestResult RunPutAsyncHns1825RootPutPlan<int32_t, 4096>(
    int, int, int, int, int, int, int, RoceHns1825CompletionMode);
template RoceHns1825TestResult RunPutAsyncHns1825RootPutPlan<uint8_t, 512>(
    int, int, int, int, int, int, int, RoceHns1825CompletionMode);
template RoceHns1825TestResult RunPutAsyncHns1825RootPutPlan<uint8_t, 64>(
    int, int, int, int, int, int, int, RoceHns1825CompletionMode);
template RoceHns1825TestResult RunPutAsyncHns1825RootPutPlan<float, 64>(
    int, int, int, int, int, int, int, RoceHns1825CompletionMode);
template RoceHns1825TestResult RunPutAsyncHns1825RootPutPlan<float, 524288>(
    int, int, int, int, int, int, int, RoceHns1825CompletionMode);

#ifdef PTO_RDMA_GET_TEST
template RoceHns1825TestResult RunGetAsyncHns1825RootGetPlan<float, 256>(
    int, int, int, int, int, int, int, RoceHns1825CompletionMode);
template RoceHns1825TestResult RunGetAsyncHns1825RootGetPlan<int32_t, 4096>(
    int, int, int, int, int, int, int, RoceHns1825CompletionMode);
template RoceHns1825TestResult RunGetAsyncHns1825RootGetPlan<uint8_t, 512>(
    int, int, int, int, int, int, int, RoceHns1825CompletionMode);
template RoceHns1825TestResult RunGetAsyncHns1825RootGetPlan<uint8_t, 64>(
    int, int, int, int, int, int, int, RoceHns1825CompletionMode);
template RoceHns1825TestResult RunGetAsyncHns1825RootGetPlan<float, 64>(
    int, int, int, int, int, int, int, RoceHns1825CompletionMode);
template RoceHns1825TestResult RunGetAsyncHns1825RootGetPlan<float, 524288>(
    int, int, int, int, int, int, int, RoceHns1825CompletionMode);
#endif // PTO_RDMA_GET_TEST
