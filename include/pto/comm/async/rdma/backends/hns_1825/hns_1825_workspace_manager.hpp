/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMM_ASYNC_RDMA_BACKENDS_HNS_1825_WORKSPACE_MANAGER_HPP
#define PTO_COMM_ASYNC_RDMA_BACKENDS_HNS_1825_WORKSPACE_MANAGER_HPP

#if defined(__CCE_KT_TEST__)
#error "hns_1825_workspace_manager.hpp is a host-only header and cannot be included in device code."
#endif

#include <arpa/inet.h>

#include <chrono>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "acl/acl.h"

#include "pto/comm/async/rdma/rdma_types.hpp"
#include "pto/comm/async/rdma/backends/hns_1825/hns_1825_hcomm_defs.hpp"
#include "pto/comm/async/rdma/backends/hns_1825/hns_1825_types.hpp"

namespace pto {
namespace comm {
namespace rdma {
namespace hns_1825 {

// ============================================================================
// WorkspaceManager: HCOMM-based RoCE control plane for the HNS_1825 backend.
//
// Uses hcomm APIs for connection establishment (HcommEndpointCreate,
// HcommMemReg, HcommChannelCreate), then reads ChannelEntity from device and
// fills RdmaInfo for AIV-side hns_1825_backend.hpp.
// Flow:
//   1. HcommEndpointCreate (ROCE, phyId + local IPv4)
//   2. HcommMemReg (register symmetric memory)
//   3. HcommChannelCreate(exchangeAllMems=true)
//   4. Poll HcommChannelGetStatus until every channel is READY (this drives the QP handshake)
//   5. Read back ChannelEntity + SQ/CQ/MR and copy RdmaInfo to device
//
// Teardown releases resources in reverse ownership order and leaves no
// PTO-owned resources:
// DestroyChannels -> UnregisterMemory -> DestroyEndpoint -> free device meta,
// then clear all handles/state. Every release error is reported, while cleanup
// continues through the remaining resources and a later Init starts fresh.
//
// Caller supplies peer RDMA NIC IPs, physical device ids, and symmetric-buffer base VAs via Init()
// (typically gathered over MPI / HCCL). Opt-in progress logs: PTO_ROCE_VERBOSE=1.
// ============================================================================
class WorkspaceManager {
public:
    WorkspaceManager() = default;
    ~WorkspaceManager() { (void)Finalize(); }

    WorkspaceManager(const WorkspaceManager&) = delete;
    WorkspaceManager& operator=(const WorkspaceManager&) = delete;

    void SetTraceId(uint32_t traceId) { traceId_ = traceId; }

    // rankId/rankCount     — this PE and the total number of PEs.
    // phyId                — global physical device id (caller resolves it, e.g. via runtime APIs).
    // localIp              — this rank's RDMA NIC IPv4 string.
    // basePort             — base TCP port for the QP handshake (e.g. 60032).
    // peerIps[rank]        — every rank's RDMA NIC IPv4 string (peerIps[rankId] == localIp).
    // peerPhyIds[rank]     — every rank's physical device id (peerPhyIds[rankId] == phyId).
    // peerSymAddrs[rank]   — every rank's symmetric-buffer base VA (peerSymAddrs[rankId] == symmetricAddr).
    // symmetricAddr/Size   — this rank's symmetric communication buffer (device memory).
    bool Init(
        uint32_t rankId, uint32_t rankCount, uint32_t phyId, const std::string& localIp, uint16_t basePort,
        const std::vector<std::string>& peerIps, const std::vector<uint32_t>& peerPhyIds,
        const std::vector<uint64_t>& peerSymAddrs, void* symmetricAddr, uint64_t symmetricSize)
    {
        if (HasOwnedResources() && !Finalize()) {
            std::cerr << "[RoCE] previous session cleanup reported errors; continuing with a fresh HCOMM session"
                      << std::endl;
        }
        if (!ValidateInitArguments(
                rankId, rankCount, phyId, localIp, basePort, peerIps, peerPhyIds, peerSymAddrs, symmetricAddr,
                symmetricSize)) {
            return false;
        }

        rankId_ = rankId;
        rankCount_ = rankCount;
        phyId_ = phyId;
        localIp_ = localIp;
        basePort_ = basePort;
        peerIps_ = peerIps;
        peerPhyIds_ = peerPhyIds;
        peerSymAddrs_ = peerSymAddrs;
        symmetricAddr_ = symmetricAddr;
        symmetricSize_ = symmetricSize;
        const auto initStart = std::chrono::steady_clock::now();
        Trace(
            "INIT begin manager=", static_cast<const void*>(this), " endpoint=", endpoint_, " memHandle=", memHandle_,
            " channels=", channelHandles_.size(), " rdmaInfo=", rdmaInfoDevice_, " symAddr=", symmetricAddr_,
            " symSize=", symmetricSize_, " basePort=", basePort_);

        if (Verbose()) {
            std::cerr << "[RoCE][rank " << rankId_ << "/" << rankCount_ << "] connect start: phyId=" << phyId_
                      << " localIp=" << localIp_ << " basePort=" << basePort_ << " symAddr=0x" << std::hex
                      << reinterpret_cast<uint64_t>(symmetricAddr_) << std::dec << " symSize=" << symmetricSize_
                      << std::endl;
            for (uint32_t r = 0; r < rankCount_; ++r) {
                std::cerr << "[RoCE][rank " << rankId_ << "]   peer[" << r << "] ip=" << peerIps_[r]
                          << " phyId=" << peerPhyIds_[r] << " symAddr=0x" << std::hex << peerSymAddrs_[r] << std::dec
                          << std::endl;
            }
        }
        auto phaseStart = std::chrono::steady_clock::now();
        if (!CreateEndpoint()) {
            (void)Finalize();
            return false;
        }
        Trace("INIT stage=EndpointCreate ok elapsed_us=", ElapsedUs(phaseStart));
        phaseStart = std::chrono::steady_clock::now();
        if (!RegisterMemory()) {
            (void)Finalize();
            return false;
        }
        Trace("INIT stage=MemReg ok elapsed_us=", ElapsedUs(phaseStart));
        phaseStart = std::chrono::steady_clock::now();
        if (!BuildChannels()) {
            (void)Finalize();
            return false;
        }
        Trace("INIT stage=ChannelCreateReady ok elapsed_us=", ElapsedUs(phaseStart));
        phaseStart = std::chrono::steady_clock::now();
        if (!FillRdmaInfo()) {
            (void)Finalize();
            return false;
        }
        Trace("INIT stage=FillRdmaInfo ok elapsed_us=", ElapsedUs(phaseStart));
        if (Verbose()) {
            std::cerr << "[RoCE][rank " << rankId_ << "] connect DONE: workspace(RdmaInfo)=0x" << std::hex
                      << reinterpret_cast<uint64_t>(rdmaInfoDevice_) << std::dec << std::endl;
        }
        initialized_ = true;
        Trace(
            "INIT success total_us=", ElapsedUs(initStart), " endpoint=", endpoint_, " memHandle=", memHandle_,
            " channels=", channelHandles_.size(), " rdmaInfo=", rdmaInfoDevice_);
        return true;
    }

    // Opt-in host progress logging (enable with PTO_ROCE_VERBOSE=1). Errors always log.
    static bool Verbose()
    {
        const char* v = std::getenv("PTO_ROCE_VERBOSE");
        return v != nullptr && v[0] == '1';
    }

    // Best-effort teardown matching the other PTO transport managers. Every
    // owned resource is attempted exactly once, failures are logged and
    // reflected in the return value, and PTO-side handles are always cleared.
    bool Finalize()
    {
        const bool hadResources = HasOwnedResources();
        const auto finalizeStart = std::chrono::steady_clock::now();
        if (hadResources) {
            Trace(
                "FINALIZE begin initialized=", initialized_, " endpoint=", endpoint_, " memHandle=", memHandle_,
                " channels=", channelHandles_.size(), " rdmaInfo=", rdmaInfoDevice_, " symAddr=", symmetricAddr_);
        }
        // Release channels before their registered memory and endpoint.
        bool ok = true;
        auto phaseStart = std::chrono::steady_clock::now();
        const bool channelsOk = DestroyChannels();
        if (!channelsOk) {
            Trace("FINALIZE failed stage=ChannelDestroy elapsed_us=", ElapsedUs(phaseStart));
            ok = false;
        } else if (hadResources) {
            Trace("FINALIZE stage=ChannelDestroy ok elapsed_us=", ElapsedUs(phaseStart));
        }
        phaseStart = std::chrono::steady_clock::now();
        const bool memoryOk = UnregisterMemory();
        if (!memoryOk) {
            Trace("FINALIZE failed stage=MemUnreg elapsed_us=", ElapsedUs(phaseStart));
            ok = false;
        } else if (hadResources) {
            Trace("FINALIZE stage=MemUnreg ok elapsed_us=", ElapsedUs(phaseStart));
        }
        phaseStart = std::chrono::steady_clock::now();
        const bool endpointOk = DestroyEndpoint();
        if (!endpointOk) {
            Trace("FINALIZE failed stage=EndpointDestroy elapsed_us=", ElapsedUs(phaseStart));
            ok = false;
        } else if (hadResources) {
            Trace("FINALIZE stage=EndpointDestroy ok elapsed_us=", ElapsedUs(phaseStart));
        }
        phaseStart = std::chrono::steady_clock::now();
        const bool deviceInfoOk = FreeDeviceInfo();
        if (!deviceInfoOk) {
            Trace("FINALIZE failed stage=FreeRdmaInfo elapsed_us=", ElapsedUs(phaseStart));
            ok = false;
        } else if (hadResources) {
            Trace("FINALIZE stage=FreeRdmaInfo ok elapsed_us=", ElapsedUs(phaseStart));
        }
        ResetOwnedState();
        if (hadResources) {
            Trace("FINALIZE end ok=", ok, " total_us=", ElapsedUs(finalizeStart));
        }
        return ok;
    }

    // Device address of the RdmaInfo table; pass this to the kernel as the RoCE workspace.
    void* GetWorkspaceAddr() const { return rdmaInfoDevice_; }

private:
    template <typename... Args>
    void Trace(Args&&... args) const
    {
        if (!Verbose()) {
            return;
        }
        std::ostringstream os;
        os << "[RoCE][case " << traceId_ << "][rank " << rankId_ << "] ";
        (os << ... << std::forward<Args>(args));
        os << '\n';
        const std::string line = os.str();
        (void)std::fwrite(line.data(), 1, line.size(), stderr);
        std::fflush(stderr);
    }

    static uint64_t ElapsedUs(const std::chrono::steady_clock::time_point& start)
    {
        return static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count());
    }

    static std::string Hex(uint64_t value)
    {
        std::ostringstream os;
        os << "0x" << std::hex << value;
        return os.str();
    }

    static bool ReadU32Device(uint64_t devAddr, uint32_t& value)
    {
        return devAddr != 0 &&
               ReadDeviceStruct(reinterpret_cast<const void*>(static_cast<uintptr_t>(devAddr)), &value, sizeof(value));
    }

    static bool WriteU32Device(uint64_t devAddr, uint32_t value)
    {
        return devAddr != 0 && aclrtMemcpy(
                                   reinterpret_cast<void*>(static_cast<uintptr_t>(devAddr)), sizeof(value), &value,
                                   sizeof(value), ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS;
    }

    bool InitializeQueueMirrors(const host::SqContext& sq, const host::CqContext& cq, uint32_t peer) const
    {
        const auto& roceSq = sq.contextInfo.roceSq;
        const auto& roceCq = cq.contextInfo.roceCq;
        uint32_t sqHead = UINT32_MAX;
        uint32_t sqTail = UINT32_MAX;
        uint32_t cqHead = UINT32_MAX;
        uint32_t cqTail = UINT32_MAX;
        const bool readOk = ReadU32Device(roceSq.headAddr, sqHead) && ReadU32Device(roceSq.tailAddr, sqTail) &&
                            ReadU32Device(roceCq.headAddr, cqHead) && ReadU32Device(roceCq.tailAddr, cqTail);
        Trace(
            "QUEUE initial peer=", peer, " sq(head/tail)=", sqHead, "/", sqTail, " cq(head/tail)=", cqHead, "/", cqTail,
            " readOk=", readOk, " sqMirror=", Hex(roceSq.headAddr), "/", Hex(roceSq.tailAddr),
            " cqMirror=", Hex(roceCq.headAddr), "/", Hex(roceCq.tailAddr));
        if (!readOk) {
            std::cerr << "[RoCE] peer " << peer << " queue mirror D2H read failed" << std::endl;
            return false;
        }

        // HCOMM exposes these freshly allocated PI/CI mirrors to the AIV data
        // path, but device allocation contents are not an initialization
        // contract. A new QP starts at producer/consumer index zero, so make
        // that state explicit before the first PTO WQE is posted.
        constexpr uint32_t kInitialQueueIndex = 0;
        const bool writeOk = WriteU32Device(roceSq.headAddr, kInitialQueueIndex) &&
                             WriteU32Device(roceSq.tailAddr, kInitialQueueIndex) &&
                             WriteU32Device(roceCq.headAddr, kInitialQueueIndex) &&
                             WriteU32Device(roceCq.tailAddr, kInitialQueueIndex);
        if (!writeOk) {
            std::cerr << "[RoCE] peer " << peer << " queue mirror initialization failed" << std::endl;
            return false;
        }
        Trace("QUEUE reset peer=", peer, " sq(head/tail)=0/0 cq(head/tail)=0/0");
        return true;
    }

    bool DestroyChannels()
    {
        if (channelHandles_.empty()) {
            return true;
        }
        std::vector<ChannelHandle> valid;
        valid.reserve(channelHandles_.size());
        for (ChannelHandle h : channelHandles_) {
            if (h != 0) {
                valid.push_back(h);
            }
        }
        if (valid.empty()) {
            channelHandles_.clear();
            channelPeer_.clear();
            return true;
        }
        if (Verbose()) {
            std::cerr << "[RoCE][rank " << rankId_ << "] HcommChannelDestroy n=" << valid.size() << std::endl;
        }
        for (size_t idx = 0; idx < channelHandles_.size(); ++idx) {
            const ChannelHandle handle = channelHandles_[idx];
            if (handle == 0) {
                continue;
            }
            const uint32_t peer = idx < channelPeer_.size() ? channelPeer_[idx] : UINT32_MAX;
            Trace("DESTROY channel index=", idx, " peer=", peer, " handle=", Hex(static_cast<uint64_t>(handle)));
        }
        const HcommResult hret = HcommChannelDestroy(valid.data(), static_cast<uint32_t>(valid.size()));
        channelHandles_.clear();
        channelPeer_.clear();
        if (hret != 0) {
            std::cerr << "[RoCE] HcommChannelDestroy failed: " << hret << std::endl;
            return false;
        }
        if (Verbose()) {
            std::cerr << "[RoCE][rank " << rankId_ << "] HcommChannelDestroy ok" << std::endl;
        }
        Trace("DESTROY channels API returned success");
        return true;
    }

    bool UnregisterMemory()
    {
        if (memHandle_ == nullptr) {
            return true;
        }
        const HcommMemHandle memHandle = memHandle_;
        memHandle_ = nullptr;
        if (endpoint_ == nullptr) {
            std::cerr << "[RoCE] cannot unregister memory without its endpoint" << std::endl;
            return false;
        }
        HcommResult hret = HcommMemUnreg(endpoint_, memHandle);
        if (hret != 0) {
            std::cerr << "[RoCE] HcommMemUnreg failed: " << hret << std::endl;
            return false;
        }
        Trace("DESTROY mem-unreg API returned success handle=", memHandle);
        return true;
    }

    bool DestroyEndpoint()
    {
        if (endpoint_ == nullptr) {
            return true;
        }
        const EndpointHandle endpoint = endpoint_;
        endpoint_ = nullptr;
        HcommResult hret = HcommEndpointDestroy(endpoint);
        if (hret != 0) {
            std::cerr << "[RoCE] HcommEndpointDestroy failed: " << hret << std::endl;
            return false;
        }
        Trace("DESTROY endpoint API returned success handle=", endpoint);
        return true;
    }

    bool FreeDeviceInfo()
    {
        if (rdmaInfoDevice_ != nullptr) {
            void* rdmaInfo = rdmaInfoDevice_;
            rdmaInfoDevice_ = nullptr;
            aclError ret = aclrtFree(rdmaInfo);
            if (ret != ACL_SUCCESS) {
                std::cerr << "[RoCE] aclrtFree(rdmaInfo) failed: " << static_cast<int>(ret) << std::endl;
                return false;
            }
            Trace("DESTROY rdma-info free returned success addr=", rdmaInfo);
        }
        return true;
    }

    bool HasOwnedResources() const
    {
        return initialized_ || endpoint_ != nullptr || memHandle_ != nullptr || rdmaInfoDevice_ != nullptr ||
               !channelHandles_.empty();
    }

    static bool ValidateInitArguments(
        uint32_t rankId, uint32_t rankCount, uint32_t phyId, const std::string& localIp, uint16_t basePort,
        const std::vector<std::string>& peerIps, const std::vector<uint32_t>& peerPhyIds,
        const std::vector<uint64_t>& peerSymAddrs, void* symmetricAddr, uint64_t symmetricSize)
    {
        if (rankCount == 0 || rankId >= rankCount || peerIps.size() != rankCount || peerPhyIds.size() != rankCount ||
            peerSymAddrs.size() != rankCount || localIp.empty() || symmetricAddr == nullptr || symmetricSize == 0) {
            std::cerr << "[RoCE] invalid rank, peer, IP, or symmetric-memory arguments" << std::endl;
            return false;
        }
        if (peerIps[rankId] != localIp || peerPhyIds[rankId] != phyId ||
            peerSymAddrs[rankId] != reinterpret_cast<uint64_t>(symmetricAddr)) {
            std::cerr << "[RoCE] local IP/phyId/address does not match this rank's gathered peer entry" << std::endl;
            return false;
        }
        if (basePort > UINT16_MAX - (kMaxRanksPerNic * kMaxRanksPerNic - 1)) {
            std::cerr << "[RoCE] basePort leaves no room for the per-rank channel port range" << std::endl;
            return false;
        }
        for (uint32_t rank = 0; rank < rankCount; ++rank) {
            if (peerIps[rank].empty() || peerPhyIds[rank] == UINT32_MAX || peerSymAddrs[rank] == 0) {
                std::cerr << "[RoCE] peer " << rank << " has an empty IP, invalid phyId, or null symmetric address"
                          << std::endl;
                return false;
            }
            uint32_t ranksOnNic = 0;
            bool occupiedSlots[kMaxRanksPerNic] = {};
            for (uint32_t peer = 0; peer < rankCount; ++peer) {
                if (peerIps[peer] != peerIps[rank]) {
                    continue;
                }
                ++ranksOnNic;
                const uint32_t slot = peer % kMaxRanksPerNic;
                if (occupiedSlots[slot]) {
                    std::cerr << "[RoCE] ranks sharing NIC IP " << peerIps[rank]
                              << " collide after modulo port mapping at slot " << slot << std::endl;
                    return false;
                }
                occupiedSlots[slot] = true;
            }
            if (ranksOnNic > kMaxRanksPerNic) {
                std::cerr << "[RoCE] more than " << kMaxRanksPerNic << " ranks share NIC IP " << peerIps[rank]
                          << "; the channel port mapping would collide" << std::endl;
                return false;
            }
        }
        return true;
    }

    // Drop all session fields so Finalize leaves no residual connect state.
    void ResetOwnedState()
    {
        channelHandles_.clear();
        channelPeer_.clear();
        peerIps_.clear();
        peerPhyIds_.clear();
        peerSymAddrs_.clear();
        memHandle_ = nullptr;
        endpoint_ = nullptr;
        rdmaInfoDevice_ = nullptr;
        symmetricAddr_ = nullptr;
        symmetricSize_ = 0;
        localIp_.clear();
        rankId_ = 0;
        rankCount_ = 1;
        phyId_ = 0;
        basePort_ = 60032;
        initialized_ = false;
    }

    bool CreateEndpoint()
    {
        EndpointDesc desc{};
        HcommResult hret = EndpointDescInit(&desc, 1);
        if (hret != 0) {
            std::cerr << "[RoCE] EndpointDescInit failed: " << hret << std::endl;
            return false;
        }
        // EndpointDescInit deliberately fills unspecified bytes with 0xFF. Do
        // not pass those sentinels into HCOMM as real endpoint attributes.
        SetEndpointDescDefaults(desc);
        desc.protocol = COMM_PROTOCOL_ROCE;
        desc.commAddr.type = COMM_ADDR_TYPE_IP_V4;
        if (inet_pton(AF_INET, localIp_.c_str(), &desc.commAddr.addr) != 1) {
            std::cerr << "[RoCE] invalid local ip: " << localIp_ << std::endl;
            return false;
        }
        desc.loc.locType = ENDPOINT_LOC_TYPE_HOST;
        desc.loc.device.devPhyId = phyId_;

        hret = HcommEndpointCreate(&desc, &endpoint_);
        if (hret != 0 || endpoint_ == nullptr) {
            std::cerr << "[RoCE] HcommEndpointCreate failed: " << hret << std::endl;
            return false;
        }
        Trace("CREATE endpoint handle=", endpoint_, " phyId=", phyId_, " ip=", localIp_);
        if (Verbose()) {
            std::cerr << "[RoCE][rank " << rankId_ << "] (1/4) endpoint created (ROCE, phyId=" << phyId_
                      << ", ip=" << localIp_ << ")" << std::endl;
        }
        return true;
    }

    bool RegisterMemory()
    {
        CommMem mem{};
        mem.type = COMM_MEM_TYPE_DEVICE;
        mem.addr = symmetricAddr_;
        mem.size = symmetricSize_;
        HcommResult hret = HcommMemReg(endpoint_, "PtoRoceBuffer", &mem, &memHandle_);
        if (hret != 0 || memHandle_ == nullptr) {
            std::cerr << "[RoCE] HcommMemReg failed: " << hret << std::endl;
            return false;
        }
        Trace("CREATE mem-reg handle=", memHandle_, " addr=", symmetricAddr_, " size=", symmetricSize_);
        if (Verbose()) {
            std::cerr << "[RoCE][rank " << rankId_ << "] (2/4) MR registered: addr=0x" << std::hex
                      << reinterpret_cast<uint64_t>(symmetricAddr_) << std::dec << " size=" << symmetricSize_
                      << std::endl;
        }
        return true;
    }

    bool BuildChannels()
    {
        if (rankCount_ <= 1) {
            return true;
        }
        const uint32_t channelNum = rankCount_ - 1;
        std::vector<HcommChannelDesc> descs(channelNum);
        HcommResult hret = HcommChannelDescInit(descs.data(), channelNum);
        if (hret != 0) {
            std::cerr << "[RoCE] HcommChannelDescInit failed: " << hret << std::endl;
            return false;
        }
        for (HcommChannelDesc& desc : descs) {
            SetChannelDescDefaults(desc);
        }

        const uint8_t tc = GetEnvU8("HCCL_RDMA_TC", 132, 255, /*requireMultipleOf4*/ true);
        const uint8_t sl = GetEnvU8("HCCL_RDMA_SL", 4, 7, /*requireMultipleOf4*/ false);

        channelPeer_.clear();
        uint32_t chIdx = 0;
        for (uint32_t remoteRank = 0; remoteRank < rankCount_; ++remoteRank) {
            if (remoteRank == rankId_) {
                continue;
            }
            descs[chIdx].remoteEndpoint.protocol = COMM_PROTOCOL_ROCE;
            descs[chIdx].remoteEndpoint.commAddr.type = COMM_ADDR_TYPE_IP_V4;
            if (inet_pton(AF_INET, peerIps_[remoteRank].c_str(), &descs[chIdx].remoteEndpoint.commAddr.addr) != 1) {
                std::cerr << "[RoCE] invalid peer ip[" << remoteRank << "]: " << peerIps_[remoteRank] << std::endl;
                return false;
            }
            descs[chIdx].remoteEndpoint.loc.locType = ENDPOINT_LOC_TYPE_HOST;
            descs[chIdx].remoteEndpoint.loc.device.devPhyId = peerPhyIds_[remoteRank];
            descs[chIdx].notifyNum = 3;
            descs[chIdx].exchangeAllMems = true;
            descs[chIdx].memHandles = nullptr;
            descs[chIdx].memHandleNum = 0;
            descs[chIdx].roceAttr.queueNum = 1;
            descs[chIdx].roceAttr.retryCnt = kDefaultRoceRetryCount;
            descs[chIdx].roceAttr.retryInterval = kDefaultRoceRetryIntervalMs;
            descs[chIdx].roceAttr.tc = tc;
            descs[chIdx].roceAttr.sl = sl;
            descs[chIdx].roceAttr.qpThreshold = 0;
            descs[chIdx].socket = nullptr;
            descs[chIdx].qos = 0;
            descs[chIdx].channelName = nullptr;
            const bool isServer = (rankId_ < remoteRank);
            descs[chIdx].role = isServer ? HCOMM_SOCKET_ROLE_SERVER : HCOMM_SOCKET_ROLE_CLIENT;
            const uint32_t serverRank = isServer ? rankId_ : remoteRank;
            const uint32_t clientRank = isServer ? remoteRank : rankId_;
            descs[chIdx].port = static_cast<uint16_t>(
                basePort_ + (serverRank % kMaxRanksPerNic) * kMaxRanksPerNic + (clientRank % kMaxRanksPerNic));
            if (Verbose()) {
                std::cerr << "[RoCE][rank " << rankId_ << "]   channel[" << chIdx << "] -> peer " << remoteRank
                          << " ip=" << peerIps_[remoteRank] << " role=" << (isServer ? "SERVER" : "CLIENT")
                          << " phyId=" << peerPhyIds_[remoteRank] << " port=" << descs[chIdx].port
                          << " tc=" << static_cast<int>(tc) << " sl=" << static_cast<int>(sl)
                          << " retryCnt=" << descs[chIdx].roceAttr.retryCnt
                          << " retryIntervalMs=" << descs[chIdx].roceAttr.retryInterval << std::endl;
            }
            channelPeer_.push_back(remoteRank);
            ++chIdx;
        }

        channelHandles_.assign(channelNum, 0);
        if (Verbose()) {
            std::cerr << "[RoCE][rank " << rankId_ << "] (3/4) HcommChannelCreate: " << channelNum
                      << " channel(s), waiting for QP handshake ..." << std::endl;
        }
        hret = HcommChannelCreate(
            endpoint_, COMM_ENGINE_AIV, descs.data(), channelNum,
            reinterpret_cast<ChannelHandle*>(channelHandles_.data()));
        if (hret != 0) {
            std::cerr << "[RoCE] HcommChannelCreate failed: " << hret << std::endl;
            return false;
        }
        for (uint32_t idx = 0; idx < channelNum; ++idx) {
            if (channelHandles_[idx] == 0) {
                std::cerr << "[RoCE] HcommChannelCreate returned a null handle at index " << idx << std::endl;
                return false;
            }
            Trace(
                "CREATE channel index=", idx, " peer=", channelPeer_[idx],
                " handle=", Hex(static_cast<uint64_t>(channelHandles_[idx])));
        }
        if (!WaitChannelsReady()) {
            return false;
        }
        if (Verbose()) {
            std::cerr << "[RoCE][rank " << rankId_ << "] (3/4) all channels READY" << std::endl;
        }
        return true;
    }

    bool WaitChannelsReady()
    {
        if (channelHandles_.empty()) {
            return true;
        }
        constexpr int32_t kChannelReady = 0;
        constexpr int32_t kChannelConnecting = 1;
        constexpr int32_t kChannelFailed = 2;
        constexpr int32_t kChannelTimeout = 3;
        const uint32_t channelNum = static_cast<uint32_t>(channelHandles_.size());
        const uint32_t pollIntervalMs = channelNum * kChannelStatusPollIntervalPerChannelMs;
        const uint32_t pollTimeoutMs = channelNum * kChannelStatusPollTimeoutPerChannelMs;
        std::vector<int32_t> statuses(channelNum, kChannelConnecting);
        uint32_t elapsedMs = 0;

        while (elapsedMs <= pollTimeoutMs) {
            HcommResult hret = HcommChannelGetStatus(channelHandles_.data(), channelNum, statuses.data());
            if (hret != 0) {
                std::cerr << "[RoCE] HcommChannelGetStatus failed: " << hret << std::endl;
                return false;
            }
            bool allReady = true;
            for (uint32_t idx = 0; idx < channelNum; ++idx) {
                const int32_t status = statuses[idx];
                if (status == kChannelReady) {
                    continue;
                }
                allReady = false;
                if (status == kChannelConnecting) {
                    continue;
                }
                const char* reason = status == kChannelFailed  ? "failed" :
                                     status == kChannelTimeout ? "timed out" :
                                                                 "returned an unknown status";
                std::cerr << "[RoCE] channel " << idx << " (peer " << channelPeer_[idx] << ") " << reason << ": "
                          << status << std::endl;
                return false;
            }
            if (allReady) {
                if (Verbose()) {
                    std::cerr << "[RoCE][rank " << rankId_ << "] channels READY after " << elapsedMs << " ms"
                              << std::endl;
                }
                return true;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(pollIntervalMs));
            elapsedMs += pollIntervalMs;
        }
        std::cerr << "[RoCE] wait channel READY timed out after " << elapsedMs << " ms" << std::endl;
        return false;
    }

    bool FillRdmaInfo()
    {
        constexpr uint32_t kQpNum = 1;
        const size_t sqBytes = sizeof(RoceSqCtx) * rankCount_ * kQpNum;
        const size_t cqBytes = sizeof(RoceCqCtx) * rankCount_ * kQpNum;
        const size_t memBytes = sizeof(RdmaMemInfo) * rankCount_;
        const size_t totalSize = sizeof(RdmaInfo) + 2 * sqBytes + 2 * cqBytes + memBytes;

        if (aclrtMalloc(&rdmaInfoDevice_, totalSize, ACL_MEM_MALLOC_HUGE_FIRST) != 0 || rdmaInfoDevice_ == nullptr) {
            std::cerr << "[RoCE] aclrtMalloc(rdmaInfo) failed" << std::endl;
            return false;
        }

        std::vector<uint8_t> hostBuf(totalSize, 0);
        auto* info = reinterpret_cast<RdmaInfo*>(hostBuf.data());
        info->magic = kRdmaWorkspaceMagic;
        info->version = kRdmaWorkspaceVersion;
        info->backend = RdmaBackend::HNS_1825;
        info->qpNum = kQpNum;
        info->rankCount = rankCount_;

        uint8_t* devBase = static_cast<uint8_t*>(rdmaInfoDevice_) + sizeof(RdmaInfo);
        info->sqPtr = reinterpret_cast<uint64_t>(devBase);
        info->rqPtr = reinterpret_cast<uint64_t>(devBase + sqBytes);
        info->scqPtr = reinterpret_cast<uint64_t>(devBase + 2 * sqBytes);
        info->rcqPtr = reinterpret_cast<uint64_t>(devBase + 2 * sqBytes + cqBytes);
        info->memPtr = reinterpret_cast<uint64_t>(devBase + 2 * sqBytes + 2 * cqBytes);

        uint8_t* hostAddr = hostBuf.data() + sizeof(RdmaInfo);
        auto* sqArr = reinterpret_cast<RoceSqCtx*>(hostAddr);
        auto* rqArr = reinterpret_cast<RoceSqCtx*>(hostAddr + sqBytes);
        auto* scqArr = reinterpret_cast<RoceCqCtx*>(hostAddr + 2 * sqBytes);
        auto* rcqArr = reinterpret_cast<RoceCqCtx*>(hostAddr + 2 * sqBytes + cqBytes);
        auto* memArr = reinterpret_cast<RdmaMemInfo*>(hostAddr + 2 * sqBytes + 2 * cqBytes);

        bool localMrRead = rankCount_ == 1;
        if (rankCount_ == 1) {
            memArr[rankId_].addr = reinterpret_cast<uint64_t>(symmetricAddr_);
            memArr[rankId_].size = symmetricSize_;
        }
        for (size_t c = 0; c < channelHandles_.size() && !localMrRead; ++c) {
            host::ChannelEntity entity{};
            if (!ReadChannelEntity(channelHandles_[c], entity)) {
                std::cerr << "[RoCE] failed to read ChannelEntity while resolving the local MR" << std::endl;
                return false;
            }
            if (!ValidateChannelEntity(entity, channelPeer_[c]) || entity.localBufferNum == 0 ||
                entity.localBufferAddr == nullptr) {
                std::cerr << "[RoCE] local registered buffer is absent for peer " << channelPeer_[c] << std::endl;
                return false;
            }
            host::RegedBufferEntity localBuf{};
            if (!ReadDeviceStruct(entity.localBufferAddr, &localBuf, sizeof(localBuf)) ||
                !ValidateRegisteredBuffer(
                    localBuf, reinterpret_cast<uint64_t>(symmetricAddr_), symmetricSize_, "local", rankId_)) {
                return false;
            }
            memArr[rankId_].addr = localBuf.bufferInfo.rma.addr;
            memArr[rankId_].size = localBuf.bufferInfo.rma.size;
            memArr[rankId_].lkey = localBuf.bufferInfo.rma.protectionInfo.memInfo.roce.lkey;
            memArr[rankId_].rkey = localBuf.bufferInfo.rma.protectionInfo.memInfo.roce.rkey;
            localMrRead = true;
        }
        if (!localMrRead) {
            std::cerr << "[RoCE] failed to read the local registered buffer from any channel" << std::endl;
            return false;
        }

        for (size_t c = 0; c < channelHandles_.size(); ++c) {
            const uint32_t peer = channelPeer_[c];
            host::ChannelEntity entity{};
            if (!ReadChannelEntity(channelHandles_[c], entity)) {
                std::cerr << "[RoCE] read ChannelEntity failed for peer " << peer << std::endl;
                return false;
            }
            if (!ValidateChannelEntity(entity, peer)) {
                return false;
            }
            if (entity.remoteBufferNum == 0 || entity.remoteBufferAddr == nullptr) {
                std::cerr << "[RoCE] peer " << peer << " has no exchanged remote registered buffer" << std::endl;
                return false;
            }
            host::RegedBufferEntity remoteBuf{};
            if (!ReadDeviceStruct(entity.remoteBufferAddr, &remoteBuf, sizeof(remoteBuf)) ||
                !ValidateRegisteredBuffer(remoteBuf, peerSymAddrs_[peer], symmetricSize_, "remote", peer)) {
                return false;
            }
            memArr[peer].addr = remoteBuf.bufferInfo.rma.addr;
            memArr[peer].size = remoteBuf.bufferInfo.rma.size;
            memArr[peer].lkey = remoteBuf.bufferInfo.rma.protectionInfo.memInfo.roce.lkey;
            memArr[peer].rkey = remoteBuf.bufferInfo.rma.protectionInfo.memInfo.roce.rkey;

            host::SqContext sq{};
            if (entity.sqNum == 0 || entity.sqContextAddr == nullptr ||
                !ReadDeviceStruct(entity.sqContextAddr, &sq, sizeof(sq)) || !ValidateSqContext(sq, peer)) {
                return false;
            }
            CopyRoceSq(sqArr[peer], sq);
            rqArr[peer] = sqArr[peer];

            host::CqContext cq{};
            if (entity.cqNum == 0 || entity.cqContextAddr == nullptr ||
                !ReadDeviceStruct(entity.cqContextAddr, &cq, sizeof(cq)) || !ValidateCqContext(cq, peer)) {
                return false;
            }
            if (!InitializeQueueMirrors(sq, cq, peer)) {
                return false;
            }
            CopyRoceCq(scqArr[peer], cq);
            rcqArr[peer] = scqArr[peer];
        }

        if (Verbose()) {
            std::cerr << "[RoCE][rank " << rankId_ << "] (4/4) RdmaInfo filled (qpNum=" << info->qpNum
                      << "):" << std::endl;
            std::cerr << "[RoCE][rank " << rankId_ << "]   localMR[" << rankId_ << "] addr=0x" << std::hex
                      << memArr[rankId_].addr << std::dec << " size=" << memArr[rankId_].size << " lkey=0x" << std::hex
                      << memArr[rankId_].lkey << " rkey=0x" << memArr[rankId_].rkey << std::dec << std::endl;
            for (size_t c = 0; c < channelPeer_.size(); ++c) {
                const uint32_t peer = channelPeer_[c];
                std::cerr << "[RoCE][rank " << rankId_ << "]   peer[" << peer << "] MR addr=0x" << std::hex
                          << memArr[peer].addr << " rkey=0x" << memArr[peer].rkey << std::dec
                          << " | SQ wqn=" << sqArr[peer].wqn << " depth=" << sqArr[peer].depth << " sqVa=0x" << std::hex
                          << sqArr[peer].bufAddr << " dbHw=0x" << sqArr[peer].dbAddr << " dbSw=0x"
                          << sqArr[peer].dbSwAddr << std::dec << " mtuShift=" << static_cast<int>(sqArr[peer].mtuShift)
                          << " | CQ cqn=" << scqArr[peer].cqn << " depth=" << scqArr[peer].depth << " cqVa=0x"
                          << std::hex << scqArr[peer].bufAddr << " dbSw=0x" << scqArr[peer].dbSwAddr << std::dec
                          << std::endl;
            }
        }

        if (aclrtMemcpy(rdmaInfoDevice_, totalSize, hostBuf.data(), totalSize, ACL_MEMCPY_HOST_TO_DEVICE) != 0) {
            std::cerr << "[RoCE] aclrtMemcpy(rdmaInfo H2D) failed" << std::endl;
            return false;
        }
        return true;
    }

    static bool IsPowerOfTwo(uint64_t value) { return value != 0 && (value & (value - 1)) == 0; }

    static bool ValidateChannelEntity(const host::ChannelEntity& entity, uint32_t peer)
    {
        if (entity.engine != COMM_ENGINE_AIV || entity.protocol != COMM_PROTOCOL_ROCE) {
            std::cerr << "[RoCE] peer " << peer
                      << " ChannelEntity has unexpected engine/protocol: " << static_cast<int>(entity.engine) << "/"
                      << static_cast<int>(entity.protocol) << std::endl;
            return false;
        }
        return true;
    }

    static bool ValidateRegisteredBuffer(
        const host::RegedBufferEntity& buffer, uint64_t expectedAddr, uint64_t expectedSize, const char* side,
        uint32_t rank)
    {
        const auto& rma = buffer.bufferInfo.rma;
        if (buffer.type != host::REGED_BUFFER_TYPE_RMA || rma.protectionInfo.type != host::PROTECTION_TYPE_ROCE ||
            rma.addr != expectedAddr || rma.size < expectedSize) {
            std::cerr << "[RoCE] " << side << " MR for rank " << rank
                      << " has invalid type/protection/address/size: type=" << static_cast<int>(buffer.type)
                      << " protection=" << static_cast<int>(rma.protectionInfo.type) << " addr=0x" << std::hex
                      << rma.addr << " expected=0x" << expectedAddr << std::dec << " size=" << rma.size
                      << " expected-at-least=" << expectedSize << std::endl;
            return false;
        }
        return true;
    }

    static bool ValidateSqContext(const host::SqContext& sq, uint32_t peer)
    {
        const auto& r = sq.contextInfo.roceSq;
        if (sq.type != host::SQ_CONTEXT_TYPE_ROCE || !IsPowerOfTwo(r.depth) || r.depth <= kHns1825PollCqThreshold ||
            r.sqVa == 0 || r.headAddr == 0 || r.tailAddr == 0 || r.dbHwVa == 0 || r.dbSwVa == 0) {
            std::cerr << "[RoCE] peer " << peer
                      << " has an unusable RoCE SQ context: type=" << static_cast<int>(sq.type) << " depth=" << r.depth
                      << " wqeSize=" << r.wqeSize << std::endl;
            return false;
        }
        return true;
    }

    static bool ValidateCqContext(const host::CqContext& cq, uint32_t peer)
    {
        const auto& r = cq.contextInfo.roceCq;
        const uint32_t cqeSize = r.cqeSize == 0 ? kHns1825DefaultCqeSize : r.cqeSize;
        const uint64_t cqRing = static_cast<uint64_t>(r.cqDepth) + kHns1825CqeMaxGenNum;
        if (cq.type != host::CQ_CONTEXT_TYPE_ROCE || !IsPowerOfTwo(cqRing) || cqeSize < sizeof(Hns1825Cqe) ||
            cqeSize > kHns1825WriteReadWqeSize || r.cqVa == 0 || r.headAddr == 0 || r.tailAddr == 0 || r.dbSwVa == 0) {
            std::cerr << "[RoCE] peer " << peer
                      << " has an unusable RoCE CQ context: type=" << static_cast<int>(cq.type)
                      << " depth=" << r.cqDepth << " cqeSize=" << r.cqeSize << std::endl;
            return false;
        }
        return true;
    }

    static void CopyRoceSq(RoceSqCtx& dst, const host::SqContext& src)
    {
        const auto& r = src.contextInfo.roceSq;
        dst.wqn = r.qpn;
        dst.bufAddr = r.sqVa;
        dst.wqeSize = r.wqeSize;
        dst.depth = r.depth;
        dst.headAddr = r.headAddr;
        dst.tailAddr = r.tailAddr;
        dst.dbMode = RdmaDbMode::SW_DB;
        dst.dbAddr = r.dbHwVa;
        dst.sl = r.sl;
        dst.amoAddr = 0;
        dst.amoLkey = 0;
        dst.dbSwAddr = r.dbSwVa;
        dst.mtuShift = r.mtuShift;
    }

    static void CopyRoceCq(RoceCqCtx& dst, const host::CqContext& src)
    {
        const auto& r = src.contextInfo.roceCq;
        dst.cqn = r.cqn;
        dst.bufAddr = r.cqVa;
        dst.cqeSize = r.cqeSize;
        dst.depth = r.cqDepth;
        dst.headAddr = r.headAddr;
        dst.tailAddr = r.tailAddr;
        dst.dbMode = RdmaDbMode::SW_DB;
        dst.dbAddr = r.dbHwVa;
        dst.dbSwAddr = r.dbSwVa;
    }

    bool ReadChannelEntity(ChannelHandle handle, host::ChannelEntity& out)
    {
        if (handle == 0) {
            return false;
        }
        return ReadDeviceStruct(reinterpret_cast<void*>(static_cast<uintptr_t>(handle)), &out, sizeof(out));
    }

    static bool ReadDeviceStruct(const void* devPtr, void* host, size_t size)
    {
        if (devPtr == nullptr) {
            return false;
        }
        return aclrtMemcpy(host, size, devPtr, size, ACL_MEMCPY_DEVICE_TO_HOST) == 0;
    }

    static uint8_t GetEnvU8(const char* name, uint8_t defVal, long maxVal, bool requireMultipleOf4)
    {
        const char* v = std::getenv(name);
        if (v == nullptr) {
            return defVal;
        }
        char* end = nullptr;
        long parsed = std::strtol(v, &end, 10);
        if (end == v || *end != '\0' || parsed < 0 || parsed > maxVal) {
            std::cerr << "[RoCE] env " << name << "='" << v << "' invalid (expect 0.." << maxVal << "); using default "
                      << static_cast<int>(defVal) << std::endl;
            return defVal;
        }
        if (requireMultipleOf4 && (parsed % 4 != 0)) {
            std::cerr << "[RoCE] env " << name << "=" << parsed << " must be a multiple of 4; using default "
                      << static_cast<int>(defVal) << std::endl;
            return defVal;
        }
        return static_cast<uint8_t>(parsed);
    }

    static void SetEndpointDescDefaults(EndpointDesc& desc)
    {
        std::memset(&desc, 0, sizeof(desc));
        desc.protocol = COMM_PROTOCOL_RESERVED;
        desc.commAddr.type = COMM_ADDR_TYPE_RESERVED;
        desc.loc.locType = ENDPOINT_LOC_TYPE_RESERVED;
    }

    static void SetChannelDescDefaults(HcommChannelDesc& desc)
    {
        // Keep the ABI header produced by HcommChannelDescInit, but replace
        // every 0xFF-filled payload and padding byte with a PTO-owned
        // deterministic value.
        const CommAbiHeader header = desc.header;
        std::memset(&desc, 0, sizeof(desc));
        desc.header = header;
        SetEndpointDescDefaults(desc.remoteEndpoint);
        desc.role = HCOMM_SOCKET_ROLE_RESERVED;
    }

    static constexpr uint32_t kMaxRanksPerNic = 16;
    // Match the current HCOMM RoCE defaults, but pass them explicitly so a
    // descriptor initialized to 0xFF can never become a huge retry request.
    static constexpr uint32_t kDefaultRoceRetryCount = 7;
    static constexpr uint32_t kDefaultRoceRetryIntervalMs = 20;
    static constexpr uint32_t kChannelStatusPollIntervalPerChannelMs = 10;
    static constexpr uint32_t kChannelStatusPollTimeoutPerChannelMs = 60000;

    uint32_t rankId_{0};
    uint32_t rankCount_{1};
    uint32_t phyId_{0};
    std::string localIp_;
    uint16_t basePort_{60032};
    std::vector<std::string> peerIps_;
    std::vector<uint32_t> peerPhyIds_;
    std::vector<uint64_t> peerSymAddrs_;
    void* symmetricAddr_{nullptr};
    uint64_t symmetricSize_{0};

    EndpointHandle endpoint_{nullptr};
    HcommMemHandle memHandle_{nullptr};
    std::vector<ChannelHandle> channelHandles_;
    std::vector<uint32_t> channelPeer_;
    void* rdmaInfoDevice_{nullptr};
    bool initialized_{false};
    uint32_t traceId_{0};
};

} // namespace hns_1825
} // namespace rdma
} // namespace comm
} // namespace pto

#endif // PTO_COMM_ASYNC_RDMA_BACKENDS_HNS_1825_WORKSPACE_MANAGER_HPP
