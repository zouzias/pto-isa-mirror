/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMM_ASYNC_URMA_WORKSPACE_MANAGER_HPP
#define PTO_COMM_ASYNC_URMA_WORKSPACE_MANAGER_HPP

#if defined(__CCE_KT_TEST__)
#error "urma_workspace_manager.hpp is a host-only header and cannot be included in device code."
#endif

#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "securec.h"

#include "acl/acl.h"
#include "hccl/hccl.h"
#include "hccl/hccl_types.h"
#include "hccl/hccl_res.h"
#include "hccl/hccl_rank_graph.h"

#include "pto/comm/async/urma/urma_types.hpp"
#include "pto/comm/async/urma/urma_hccl_defs.hpp"
#include "pto/comm/async/urma/urma_channel_helper.hpp"

namespace pto {
namespace comm {
namespace urma {

// ============================================================================
// UrmaWorkspaceManager: HCCL-based URMA workspace initialization.
//
// Uses HCCL public APIs for connection establishment (HcclCommMemReg,
// HcclRankGraphGetLinks, HcclChannelAcquire), then reads ChannelEntity from
// device/host and fills UrmaInfo for AIV-side urma_async_intrin.hpp.
// Flow:
//   1. Allocate the notify pool and register symmetric + notify memory
//   2. HcclRankGraphGetLinks (find UBC_TP/CTP links per peer)
//   3. HcclChannelAcquire(COMM_ENGINE_AIV) — returns device ChannelEntity* handles
//      (requires CANN with ConvertAivChannelHandlesToDevicePtrs in HcclChannelAcquire)
//   4. Read back ChannelEntity + sub-structures from device
//   5. Convert to UrmaInfo format and copy to device
//
// Two layouts, chosen by the sharedJettyCount argument of Init():
//
//   PerPeer (sharedJettyCount == 0), the historical layout. One jetty per peer,
//   acquired as one channel each. Jetty i talks to peer i and nothing else, so a
//   post needs no jetty argument — the destination rank picks the queue.
//
//   SharedPool (sharedJettyCount == N >= 1). N jetties, each reaching every peer,
//   built by calling HcclChannelAcquireWithConfig once per jetty with
//   IS_SHARED_QUEUE and a distinct SHARED_QUEUE_TAG; the rankCount-1 channels under
//   one tag land on one jetty. A post then needs both a jettyIdx (which queue) and
//   a peer (which target), and the caller owns jetty assignment: two producers on
//   one jetty at the same time corrupt its producer index.
// ============================================================================
class UrmaWorkspaceManager {
public:
    UrmaWorkspaceManager() = default;
    ~UrmaWorkspaceManager() { Finalize(); }

    UrmaWorkspaceManager(const UrmaWorkspaceManager&) = delete;
    UrmaWorkspaceManager& operator=(const UrmaWorkspaceManager&) = delete;

    static constexpr const char* kDefaultSharedTagPrefix = "pto_urma_jetty";

    // sharedJettyCount: 0 selects PerPeer, N >= 1 builds a SharedPool of N jetties.
    // sqDepth: 0 keeps the hcomm default. On the shared path this is the only way in,
    // see ApplySqDepth().
    bool Init(
        HcclComm comm, uint32_t rankId, uint32_t rankCount, void* symmetricAddr, uint64_t symmetricSize,
        uint32_t sharedJettyCount = 0, uint32_t sqDepth = 0, const char* sharedTagPrefix = kDefaultSharedTagPrefix)
    {
        comm_ = comm;
        rankId_ = rankId;
        rankCount_ = rankCount;
        symmetricAddr_ = symmetricAddr;
        symmetricSize_ = symmetricSize;
        sharedJettyCount_ = sharedJettyCount;
        sqDepth_ = sqDepth;
        sharedTagPrefix_ = (sharedTagPrefix != nullptr) ? sharedTagPrefix : kDefaultSharedTagPrefix;

        if (!AllocateNotifyPool()) {
            return false;
        }
        if (!RegisterMemory()) {
            return false;
        }
        if (!BuildChannels()) {
            return false;
        }
        if (!ExtractAndFillUrmaInfo()) {
            return false;
        }

        initialized_ = true;
        return true;
    }

    // The owning communication context must stop kernels, drain QPs, and
    // destroy HCCL communication resources before calling Finalize().
    void Finalize()
    {
        FreeDeviceAddr(urmaInfoDevice_);
        FreeDeviceAddr(eidDevice_);
        FreeDeviceAddr(notifyPoolDevice_);
        channelHandles_.clear();
        peerBaseAddrs_.clear();
        sharedPeerOrder_.clear();
        channelsPerJetty_ = 0;
        initialized_ = false;
    }

    void* GetWorkspaceAddr() const { return urmaInfoDevice_; }

    // Per-peer symmetric MR base address (self = symmetricAddr_). Valid after Init().
    uint64_t PeerBaseAddr(uint32_t peer) const
    {
        if (peer >= peerBaseAddrs_.size()) {
            return 0;
        }
        return peerBaseAddrs_[peer];
    }

    // Layout the device side will see. Callers building a session need this to know
    // whether a jettyIdx is required.
    UrmaLayout Layout() const { return (sharedJettyCount_ == 0) ? UrmaLayout::PER_PEER : UrmaLayout::SHARED_POOL; }

    // Number of local jetties, i.e. the exclusive upper bound on jettyIdx.
    uint32_t JettyCount() const { return (sharedJettyCount_ == 0) ? rankCount_ : sharedJettyCount_; }

private:
    // SQ/CQ context rows: one per peer in PerPeer, one per jetty in SharedPool.
    uint64_t CtxRowCount() const { return (sharedJettyCount_ == 0) ? rankCount_ : sharedJettyCount_; }

    // Rows of any per-target table (remote mem info, notify regions). SharedPool needs
    // one per (jetty, peer): tpID and remoteEID belong to the jetty-peer connection,
    // not to the peer, so each jetty carries its own copy of the peer table.
    uint64_t TargetRowCount() const
    {
        return (sharedJettyCount_ == 0) ? rankCount_ : static_cast<uint64_t>(sharedJettyCount_) * rankCount_;
    }

    bool AllocateNotifyPool()
    {
        const uint64_t regionCount = TargetRowCount();
        if (regionCount == 0U ||
            regionCount > std::numeric_limits<uint64_t>::max() / sizeof(UrmaNotifyResourceRegion)) {
            std::cerr << "[URMA] invalid notify pool region count=" << regionCount << std::endl;
            return false;
        }
        notifyPoolSize_ = regionCount * sizeof(UrmaNotifyResourceRegion);
        aclError err = aclrtMalloc(&notifyPoolDevice_, notifyPoolSize_, ACL_MEM_MALLOC_HUGE_FIRST);
        if (err != ACL_SUCCESS) {
            std::cerr << "[URMA] aclrtMalloc(notifyPool) failed: " << err << " size=" << notifyPoolSize_ << std::endl;
            return false;
        }
        err = aclrtMemset(notifyPoolDevice_, notifyPoolSize_, 0, notifyPoolSize_);
        if (err != ACL_SUCCESS) {
            std::cerr << "[URMA] aclrtMemset(notifyPool) failed: " << err << std::endl;
            return false;
        }
        return true;
    }

    bool RegisterMemory()
    {
        CommMem mem{};
        mem.type = COMM_MEM_TYPE_DEVICE;
        mem.addr = symmetricAddr_;
        mem.size = symmetricSize_;

        HcclResult ret = HcclCommMemReg(comm_, kUrmaSymMemTag, &mem, &memHandle_);
        if (ret != HCCL_SUCCESS) {
            std::cerr << "[URMA] HcclCommMemReg failed: " << static_cast<int>(ret) << std::endl;
            return false;
        }

        CommMem notifyMem{};
        notifyMem.type = COMM_MEM_TYPE_DEVICE;
        notifyMem.addr = notifyPoolDevice_;
        notifyMem.size = notifyPoolSize_;
        ret = HcclCommMemReg(comm_, kUrmaNotifyMemTag, &notifyMem, &notifyMemHandle_);
        if (ret != HCCL_SUCCESS) {
            std::cerr << "[URMA] HcclCommMemReg(notifyPool) failed: " << static_cast<int>(ret) << std::endl;
            return false;
        }
        memHandles_[0] = memHandle_;
        memHandles_[1] = notifyMemHandle_;
        return true;
    }

    // One usable UB link to a peer, resolved from the rank graph.
    struct PeerLink {
        uint32_t peer{0};
        CommProtocol proto{};
        EndpointDesc localEp{};
        EndpointDesc remoteEp{};
    };

    // First UB link from srcRank to dstRank. SharedPool then replaces both
    // endpoints with per-rank canonicals so the batch shares one local jetty
    // and the acquire handshake still agrees pair-wise.
    bool FindPeerLinkFrom(uint32_t srcRank, uint32_t dstRank, PeerLink& out) const
    {
        uint32_t linkNum = 0;
        CommLink* linkList = nullptr;
        HcclResult rc = HcclRankGraphGetLinks(comm_, 0, srcRank, dstRank, &linkList, &linkNum);
        if (rc != HCCL_SUCCESS) {
            std::cerr << "[URMA] HcclRankGraphGetLinks src=" << srcRank << " dst=" << dstRank
                      << " ret=" << static_cast<int>(rc) << std::endl;
            return false;
        }
        for (uint32_t i = 0; i < linkNum; ++i) {
            CommProtocol proto = linkList[i].linkAttr.linkProtocol;
            if (proto != kCommProtocolUbcCtp && proto != kCommProtocolUbcTp) {
                continue;
            }
            out = PeerLink{dstRank, proto, linkList[i].srcEndpointDesc, linkList[i].dstEndpointDesc};
            return true;
        }
        std::cerr << "[URMA] rank=" << srcRank << " no UBC_TP/CTP link to peer=" << dstRank << std::endl;
        return false;
    }

    bool FindPeerLink(uint32_t peer, PeerLink& out) const { return FindPeerLinkFrom(rankId_, peer, out); }

    // Same peer the existing rank-graph scan would pick first: the lowest
    // rank id other than `rank`. Every rank must use this so my canonical
    // local is exactly the remote the peer writes for me.
    static uint32_t CanonicalPeerOf(uint32_t rank, uint32_t rankCount)
    {
        for (uint32_t peer = 0; peer < rankCount; ++peer) {
            if (peer != rank) {
                return peer;
            }
        }
        return rank;
    }

    bool FindRankCanonicalLocal(uint32_t rank, EndpointDesc& out) const
    {
        const uint32_t peer = CanonicalPeerOf(rank, rankCount_);
        if (peer == rank) {
            return false;
        }
        PeerLink link{};
        if (!FindPeerLinkFrom(rank, peer, link)) {
            return false;
        }
        out = link.localEp;
        return true;
    }

    void FillChannelDesc(HcclChannelDesc& desc, const PeerLink& link)
    {
        HcclChannelDescInit(&desc, 1);
        desc.remoteRank = link.peer;
        desc.notifyNum = 0;
        desc.channelProtocol = link.proto;
        desc.localEndpoint = link.localEp;
        desc.remoteEndpoint = link.remoteEp;
        desc.memHandles = memHandles_;
        desc.memHandleNum = kUrmaRegisteredMemCount;
        ApplySqDepth(desc);
    }

    void ApplySqDepth(HcclChannelDesc& desc) const
    {
        if (sqDepth_ == 0) {
            return;
        }
        // hcomm's ubAttr.sqDepth sits at offset 0 of the desc union and the raw bytes
        // are copied straight into HcommChannelDesc, so writing the first four bytes
        // sets it. The shared path converts with a default CommConfig and never calls
        // ConfigSqDepthByExpansionMode, so this is the only way in. Must run after
        // HcclChannelDescInit, whose 0xFF fill means "not set".
        const uint32_t depth = sqDepth_;
        (void)memcpy_s(desc.raws, sizeof(desc.raws), &depth, sizeof(depth));
    }

    // Same rule hcomm applies in ValidateSharedQueueDescs; checking it here turns a
    // late generic failure into a message that names the endpoint problem.
    static bool IsSameLocalEndpoint(const EndpointDesc& a, const EndpointDesc& b)
    {
        return a.protocol == b.protocol && a.commAddr.type == b.commAddr.type &&
               std::memcmp(a.commAddr.raws, b.commAddr.raws, sizeof(a.commAddr.raws)) == 0 &&
               a.loc.locType == b.loc.locType && std::memcmp(a.loc.raws, b.loc.raws, sizeof(a.loc.raws)) == 0;
    }

    bool BuildChannels() { return (sharedJettyCount_ == 0) ? BuildChannelsPerPeer() : BuildChannelsSharedPool(); }

    bool BuildChannelsPerPeer()
    {
        std::vector<HcclChannelDesc> descs;
        descs.reserve(rankCount_ - 1);
        for (uint32_t peer = 0; peer < rankCount_; ++peer) {
            if (peer == rankId_) {
                continue;
            }
            PeerLink link{};
            if (!FindPeerLink(peer, link)) {
                return false;
            }
            HcclChannelDesc desc{};
            FillChannelDesc(desc, link);
            descs.push_back(desc);
        }
        channelHandles_.resize(descs.size());
        HcclResult rc = HcclChannelAcquire(
            comm_, COMM_ENGINE_AIV, descs.data(), static_cast<uint32_t>(descs.size()), channelHandles_.data());
        if (rc != HCCL_SUCCESS) {
            std::cerr << "[URMA] HcclChannelAcquire failed: " << static_cast<int>(rc) << std::endl;
            return false;
        }
        return true;
    }

    bool BuildChannelsSharedPool()
    {
        std::vector<PeerLink> links;
        if (!SelectUniformLocalEndpoint(links)) {
            return false;
        }
        channelsPerJetty_ = static_cast<uint32_t>(links.size());

        sharedPeerOrder_.clear();
        sharedPeerOrder_.reserve(links.size());
        for (const PeerLink& link : links) {
            sharedPeerOrder_.push_back(link.peer);
        }

        channelHandles_.assign(static_cast<size_t>(sharedJettyCount_) * channelsPerJetty_, 0);
        for (uint32_t jetty = 0; jetty < sharedJettyCount_; ++jetty) {
            if (!AcquireOneJetty(jetty, links)) {
                return false;
            }
        }
        std::cerr << "[URMA] shared pool ready rank=" << rankId_ << " jetties=" << sharedJettyCount_
                  << " channelsPerJetty=" << channelsPerJetty_ << std::endl;
        return true;
    }

    // One shared-jetty acquire: all channels under tag "<prefix>_j<jetty>" reuse a
    // single jetty, indexed inside the tag by endpoint pair, so one call with one desc
    // per peer yields one jetty reaching every peer.
    bool AcquireOneJetty(uint32_t jetty, const std::vector<PeerLink>& links)
    {
        const std::string tag = sharedTagPrefix_ + "_j" + std::to_string(jetty);
        if (tag.size() >= HCCL_CHANNEL_CONFIG_SHARED_QUEUE_TAG_MAX_LEN) {
            std::cerr << "[URMA] shared queue tag too long: " << tag << std::endl;
            return false;
        }

        std::vector<HcclChannelDesc> descs(links.size());
        for (size_t i = 0; i < links.size(); ++i) {
            FillChannelDesc(descs[i], links[i]);
        }

        HcclChannelConfig config = nullptr;
        HcclResult rc = HcclChannelConfigCreate(&config);
        if (rc != HCCL_SUCCESS) {
            std::cerr << "[URMA] HcclChannelConfigCreate failed: " << static_cast<int>(rc) << std::endl;
            return false;
        }
        rc = HcclChannelConfigSetInt(config, HCCL_CHANNEL_CONFIG_TYPE_IS_SHARED_QUEUE, 1U);
        if (rc == HCCL_SUCCESS) {
            rc = HcclChannelConfigSetStr(config, HCCL_CHANNEL_CONFIG_TYPE_SHARED_QUEUE_TAG, tag.c_str());
        }
        if (rc == HCCL_SUCCESS) {
            rc = HcclChannelAcquireWithConfig(
                comm_, COMM_ENGINE_AIV, descs.data(), static_cast<uint32_t>(descs.size()), config,
                &channelHandles_[static_cast<size_t>(jetty) * channelsPerJetty_]);
        }
        (void)HcclChannelConfigDestroy(config);

        if (rc != HCCL_SUCCESS) {
            std::cerr << "[URMA] shared jetty acquire failed, tag=" << tag << " ret=" << static_cast<int>(rc)
                      << (rc == HCCL_E_TIMEOUT ? " (HCCL_E_TIMEOUT: endpoint handshake did not match)" : "")
                      << std::endl;
            return false;
        }
        return true;
    }

    // A shared jetty is one local endpoint talking to many remotes. hcomm rejects
    // the batch unless every desc carries the same localEndpoint bytes, and the
    // acquire itself is a pairwise handshake: my (local, remote) must equal the
    // reverse of the peer's (local, remote). Rank-graph links to different peers
    // often look like the same port (same proto / IPv6 type / device id) but fail
    // a full memcmp: hcomm stores {magic, netLayer, topoInstId} in the unused tail
    // of loc.raws, and that topo instance is per path. Stamping only our local
    // makes the batch valid and then times out, because the peer still has the
    // original path-specific remote for us. Align both sides to a deterministic
    // per-rank canonical (first UB link to the lowest other rank) so:
    //   rank i uses (Canonical(i), Canonical(j)) for peer j
    //   rank j uses (Canonical(j), Canonical(i)) for peer i
    bool SelectUniformLocalEndpoint(std::vector<PeerLink>& links) const
    {
        if (rankCount_ <= 1) {
            std::cerr << "[URMA] shared pool needs at least one peer, rankCount=" << rankCount_ << std::endl;
            return false;
        }

        EndpointDesc myCanonical{};
        if (!FindRankCanonicalLocal(rankId_, myCanonical)) {
            return false;
        }

        links.clear();
        links.reserve(rankCount_ - 1);
        bool localStamped = false;
        bool remoteRewritten = false;
        for (uint32_t peer = 0; peer < rankCount_; ++peer) {
            if (peer == rankId_) {
                continue;
            }
            PeerLink link{};
            if (!FindPeerLinkFrom(rankId_, peer, link)) {
                return false;
            }
            EndpointDesc peerCanonical{};
            if (!FindRankCanonicalLocal(peer, peerCanonical)) {
                return false;
            }
            if (!IsSameLocalEndpoint(myCanonical, link.localEp)) {
                localStamped = true;
            }
            if (!IsSameLocalEndpoint(peerCanonical, link.remoteEp)) {
                remoteRewritten = true;
            }
            link.localEp = myCanonical;
            link.remoteEp = peerCanonical;
            links.push_back(link);
        }
        if (localStamped || remoteRewritten) {
            std::cerr << "[URMA] rank=" << rankId_
                      << " rank-graph endpoints differ across peers; "
                         "aligning shared-jetty descs to per-rank canonicals"
                      << std::endl;
        }
        return !links.empty();
    }

    // Host-side tables filled during extraction, sized to the active layout:
    // context rows (SQ/CQ) are per-peer in PerPeer and per-jetty in SharedPool;
    // target rows (mem/eid) are per-peer in PerPeer and per (jetty, peer) in
    // SharedPool. Per-channel tokens live in memList rows because hcomm issues a
    // fresh tokenId per channel even for one HcclCommMemReg.
    struct UrmaPeerInfoTables {
        UrmaPeerInfoTables(size_t ctxRows, size_t targetRows)
            : wqList(ctxRows), cqList(ctxRows), memList(targetRows), eidTable(targetRows * kUrmaEidBytes, 0)
        {}

        std::vector<UrmaWQCtx> wqList;
        std::vector<UrmaCqCtx> cqList;
        std::vector<UrmaMemInfo> memList;
        std::vector<uint8_t> eidTable;
    };

    bool ExtractAndFillUrmaInfo()
    {
        UrmaPeerInfoTables tables(static_cast<size_t>(CtxRowCount()), static_cast<size_t>(TargetRowCount()));

        const bool extracted = (sharedJettyCount_ == 0) ? ExtractPerPeerInfo(tables) : ExtractSharedPoolInfo(tables);
        if (!extracted) {
            return false;
        }
        if (!InitializeAsyncQueueState(tables.wqList, tables.cqList)) {
            return false;
        }
        // A peer's symmetric MR base is the same whichever local jetty reaches it, so
        // jetty 0's slice is enough. Rows are jetty-major, which puts that slice at
        // [0, rankCount_) in both layouts.
        peerBaseAddrs_.resize(rankCount_);
        for (uint32_t peer = 0; peer < rankCount_; ++peer) {
            peerBaseAddrs_[peer] = tables.memList[peer].addr;
        }
        if (!AllocAndCopyEidTable(tables)) {
            return false;
        }
        if (!BuildAndCopyUrmaInfoTable(tables)) {
            return false;
        }

        std::cerr << "[URMA] UrmaInfo OK rank=" << rankId_ << " notifyPoolBytes=" << notifyPoolSize_ << std::endl;
        return true;
    }

    // Runs over context rows, not peers: a row is a peer in PerPeer and a jetty in
    // SharedPool. The unfilled self row in PerPeer has depth 0 and is skipped.
    bool InitializeAsyncQueueState(std::vector<UrmaWQCtx>& wqList, const std::vector<UrmaCqCtx>& cqList)
    {
        for (size_t row = 0; row < wqList.size(); ++row) {
            if (wqList[row].depth == 0U) {
                continue;
            }
            uint32_t sqHead = 0U;
            uint32_t sqTail = 0U;
            uint32_t cqTail = 0U;
            if (aclrtMemcpy(
                    &sqHead, sizeof(sqHead), reinterpret_cast<void*>(wqList[row].headAddr), sizeof(sqHead),
                    ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS ||
                aclrtMemcpy(
                    &sqTail, sizeof(sqTail), reinterpret_cast<void*>(wqList[row].tailAddr), sizeof(sqTail),
                    ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS ||
                aclrtMemcpy(
                    &cqTail, sizeof(cqTail), reinterpret_cast<void*>(cqList[row].tailAddr), sizeof(cqTail),
                    ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS ||
                sqHead != sqTail) {
                std::cerr << "[URMA] ctx row=" << row << " async queue initialization requires an empty SQ"
                          << std::endl;
                return false;
            }
            wqList[row].submittedWqeCount = cqTail;
        }
        return true;
    }

    bool ExtractPerPeerInfo(UrmaPeerInfoTables& tables)
    {
        uint32_t channelIdx = 0;
        for (uint32_t peer = 0; peer < rankCount_; ++peer) {
            if (peer == rankId_) {
                tables.wqList[peer] = UrmaWQCtx{};
                tables.cqList[peer] = UrmaCqCtx{};
                tables.memList[peer] = UrmaMemInfo{};
                tables.memList[peer].addr = reinterpret_cast<uint64_t>(symmetricAddr_);
                tables.memList[peer].len = static_cast<uint32_t>(symmetricSize_);
                continue;
            }
            if (!ExtractChannel(
                    peer, channelIdx, tables.wqList[peer], tables.cqList[peer], tables.memList[peer],
                    &tables.eidTable[static_cast<size_t>(peer) * kUrmaEidBytes])) {
                return false;
            }
            ++channelIdx;
        }
        return true;
    }

    // Every channel under one tag reports the same jetty, so context rows are indexed
    // by jetty while target rows stay per (jetty, peer).
    bool ExtractSharedPoolInfo(UrmaPeerInfoTables& tables)
    {
        for (uint32_t jetty = 0; jetty < sharedJettyCount_; ++jetty) {
            for (uint32_t slot = 0; slot < channelsPerJetty_; ++slot) {
                const uint32_t peer = sharedPeerOrder_[slot];
                const size_t channelIdx = static_cast<size_t>(jetty) * channelsPerJetty_ + slot;
                const size_t targetRow = static_cast<size_t>(jetty) * rankCount_ + peer;
                UrmaWQCtx wq{};
                UrmaCqCtx cq{};
                if (!ExtractChannel(
                        peer, static_cast<uint32_t>(channelIdx), wq, cq, tables.memList[targetRow],
                        &tables.eidTable[targetRow * kUrmaEidBytes])) {
                    return false;
                }
                if (slot == 0) {
                    tables.wqList[jetty] = wq;
                    tables.cqList[jetty] = cq;
                } else if (!IsSameJetty(tables.wqList[jetty], tables.cqList[jetty], wq, cq, jetty, peer)) {
                    return false;
                }
            }
            const size_t selfRow = static_cast<size_t>(jetty) * rankCount_ + rankId_;
            tables.memList[selfRow] = UrmaMemInfo{};
            tables.memList[selfRow].addr = reinterpret_cast<uint64_t>(symmetricAddr_);
            tables.memList[selfRow].len = static_cast<uint32_t>(symmetricSize_);
        }
        return true;
    }

    // Guards the property the whole layout rests on: sharing is real, not just
    // requested. If hcomm ever handed out distinct queues under one tag, the device
    // side would post to jetty A while polling jetty B's CQ and hang.
    static bool IsSameJetty(
        const UrmaWQCtx& refWq, const UrmaCqCtx& refCq, const UrmaWQCtx& wq, const UrmaCqCtx& cq, uint32_t jetty,
        uint32_t peer)
    {
        const bool same = refWq.wqn == wq.wqn && refWq.bufAddr == wq.bufAddr && refWq.dbAddr == wq.dbAddr &&
                          refWq.headAddr == wq.headAddr && refWq.tailAddr == wq.tailAddr && refCq.cqn == cq.cqn &&
                          refCq.bufAddr == cq.bufAddr && refCq.headAddr == cq.headAddr;
        if (!same) {
            std::cerr << "[URMA] shared jetty " << jetty << " not shared: peer=" << peer << " reports jfsID=" << wq.wqn
                      << "/jfcID=" << cq.cqn << " but jetty uses jfsID=" << refWq.wqn << "/jfcID=" << refCq.cqn
                      << std::endl;
        }
        return same;
    }

    struct PeerChannelState {
        ChannelHandle handle{};
        ChannelEntity entity{};
        SqContext sq{};
        CqContext cq{};
    };

    bool ReadPeerChannel(uint32_t peer, uint32_t channelIdx, PeerChannelState& state)
    {
        state.handle = channelHandles_[channelIdx];
        if (state.handle != 0 && static_cast<uint64_t>(state.handle) < kDeviceVaThreshold) {
            std::cerr << "[URMA] ChannelHandle looks like host pointer (0x" << std::hex << state.handle << std::dec
                      << ") for peer=" << peer
                      << ". Upgrade CANN: HcclChannelAcquire must return device ChannelEntity pointers for AIV URMA."
                      << std::endl;
            return false;
        }
        RegedBufferEntity remoteBuf{};
        RegedBufferEntity localBuf{};
        if (state.handle == 0 || !UrmaChannelHelper::TryReadChannelEntity(
                                     state.handle, peer, state.entity, state.sq, state.cq, remoteBuf, localBuf)) {
            std::cerr << "[URMA] Cannot read ChannelEntity for peer=" << peer << " handle=0x" << std::hex
                      << static_cast<uint64_t>(state.handle) << std::dec << std::endl;
            return false;
        }
        return true;
    }

    bool ExtractPeerRegistrations(
        uint32_t peer, const PeerChannelState& state, RegedBufferEntity& symRemoteBuf, uint64_t& symRmaAddr,
        uint32_t& symRmaSize, UrmaMemInfo& mem)
    {
        if (!UrmaChannelHelper::SelectSymmetricRemoteBuffer(
                comm_, kUrmaSymMemTag, symmetricSize_, state.handle, peer, state.entity, symRemoteBuf, symRmaAddr,
                symRmaSize)) {
            return false;
        }
        RegedBufferEntity symLocalBuf{};
        if (!UrmaChannelHelper::FindLocalRmaRegistration(
                reinterpret_cast<uint64_t>(symmetricAddr_), symmetricSize_, state.entity, peer, symLocalBuf)) {
            std::cerr << "[URMA] peer=" << peer << " no local symmetric registration found" << std::endl;
            return false;
        }
        mem.localTokenId = symLocalBuf.bufferInfo.rma.protectionInfo.memInfo.ub.tokenId;
        RegedBufferEntity notifyLocalBuf{};
        if (!UrmaChannelHelper::FindLocalRmaRegistration(
                reinterpret_cast<uint64_t>(notifyPoolDevice_), notifyPoolSize_, state.entity, peer, notifyLocalBuf)) {
            std::cerr << "[URMA] peer=" << peer << " no local notify pool registration found" << std::endl;
            return false;
        }
        mem.notifyTokenId = notifyLocalBuf.bufferInfo.rma.protectionInfo.memInfo.ub.tokenId;
        return true;
    }

    static void LogPeerInfo(uint32_t peer, const UrmaWQCtx& wq, const UrmaMemInfo& mem)
    {
        std::cerr << "[URMA] peer=" << peer << " tpId=" << mem.tpn << " rmtAddr=0x" << std::hex << mem.addr
                  << " sqVa=0x" << wq.bufAddr << " dbAddr=0x" << wq.dbAddr << std::dec << std::endl;
    }

    bool ExtractChannel(
        uint32_t peer, uint32_t channelIdx, UrmaWQCtx& wq, UrmaCqCtx& cq, UrmaMemInfo& mem, uint8_t* eidDst)
    {
        PeerChannelState state{};
        RegedBufferEntity symRemoteBuf{};
        uint64_t symRmaAddr = 0;
        uint32_t symRmaSize = 0;
        if (!ReadPeerChannel(peer, channelIdx, state) ||
            !ExtractPeerRegistrations(peer, state, symRemoteBuf, symRmaAddr, symRmaSize, mem)) {
            return false;
        }
        FillWqCtx(wq, state.sq);
        FillCqCtx(cq, state.cq);
        FillMemInfo(mem, state.sq, symRemoteBuf, symRmaAddr, symRmaSize);
        (void)memcpy_s(eidDst, kUrmaEidBytes, state.sq.contextInfo.ubJfs.remoteEID, kUrmaEidBytes);
        LogPeerInfo(peer, wq, mem);
        return true;
    }

    static void FillWqCtx(UrmaWQCtx& wq, const SqContext& sq)
    {
        wq.wqn = sq.contextInfo.ubJfs.jfsID;
        wq.bufAddr = sq.contextInfo.ubJfs.sqVa;
        wq.wqeShiftSize = Log2U32(sq.contextInfo.ubJfs.wqeSize);
        wq.depth = sq.contextInfo.ubJfs.sqDepth;
        wq.headAddr = sq.contextInfo.ubJfs.headAddr;
        wq.tailAddr = sq.contextInfo.ubJfs.tailAddr;
        wq.dbMode = UrmaDbMode::SW_DB;
        wq.dbAddr = sq.contextInfo.ubJfs.dbVa;
        wq.sl = 0;
    }

    static void FillCqCtx(UrmaCqCtx& cqCtx, const CqContext& cq)
    {
        cqCtx.cqn = cq.contextInfo.ubJfc.jfcID;
        cqCtx.bufAddr = cq.contextInfo.ubJfc.scqVa;
        cqCtx.cqeShiftSize = Log2U32(cq.contextInfo.ubJfc.cqeSize);
        cqCtx.depth = cq.contextInfo.ubJfc.cqDepth;
        cqCtx.headAddr = cq.contextInfo.ubJfc.headAddr;
        cqCtx.tailAddr = cq.contextInfo.ubJfc.tailAddr;
        cqCtx.dbMode = UrmaDbMode::SW_DB;
        cqCtx.dbAddr = cq.contextInfo.ubJfc.dbVa;
    }

    static void FillMemInfo(
        UrmaMemInfo& mem, const SqContext& sq, const RegedBufferEntity& symRemoteBuf, uint64_t symRmaAddr,
        uint32_t symRmaSize)
    {
        mem.tokenValueValid = true;
        mem.rmtJettyType = 1;
        mem.targetHint = 0;
        mem.tpn = sq.contextInfo.ubJfs.tpID;
        mem.tid = symRemoteBuf.bufferInfo.rma.protectionInfo.memInfo.ub.tokenId;
        mem.rmtTokenValue = symRemoteBuf.bufferInfo.rma.protectionInfo.memInfo.ub.tokenValue;
        mem.len = symRmaSize;
        mem.addr = symRmaAddr;
    }

    bool AllocAndCopyEidTable(UrmaPeerInfoTables& tables)
    {
        const size_t eidDevSize = tables.eidTable.size();
        aclError err = aclrtMalloc(&eidDevice_, eidDevSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (err != ACL_SUCCESS) {
            std::cerr << "[URMA] aclrtMalloc(eidTable) failed: " << err << std::endl;
            return false;
        }
        err = aclrtMemcpy(eidDevice_, eidDevSize, tables.eidTable.data(), eidDevSize, ACL_MEMCPY_HOST_TO_DEVICE);
        if (err != ACL_SUCCESS) {
            std::cerr << "[URMA] aclrtMemcpy(eidTable) failed: " << err << std::endl;
            return false;
        }
        for (size_t row = 0; row < tables.memList.size(); ++row) {
            tables.memList[row].eidAddr =
                reinterpret_cast<uint64_t>(static_cast<uint8_t*>(eidDevice_) + row * kUrmaEidBytes);
        }
        return true;
    }

    bool BuildAndCopyUrmaInfoTable(const UrmaPeerInfoTables& tables)
    {
        // sq + rq share the WQ layout, scq + rcq share the CQ layout.
        const size_t totalSize = static_cast<size_t>(
            sizeof(UrmaInfo) + CtxRowCount() * (2U * sizeof(UrmaWQCtx) + 2U * sizeof(UrmaCqCtx)) +
            TargetRowCount() * sizeof(UrmaMemInfo));

        aclError err = aclrtMalloc(&urmaInfoDevice_, totalSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (err != ACL_SUCCESS) {
            std::cerr << "[URMA] aclrtMalloc(urmaInfo) failed: " << err << std::endl;
            return false;
        }

        std::vector<uint8_t> hostBuf(totalSize, 0);
        FillUrmaInfoLayout(hostBuf, tables);

        err = aclrtMemcpy(urmaInfoDevice_, totalSize, hostBuf.data(), totalSize, ACL_MEMCPY_HOST_TO_DEVICE);
        if (err != ACL_SUCCESS) {
            std::cerr << "[URMA] aclrtMemcpy(urmaInfo) failed: " << err << std::endl;
            return false;
        }
        return true;
    }

    void FillUrmaInfoLayout(std::vector<uint8_t>& hostBuf, const UrmaPeerInfoTables& tables)
    {
        const size_t ctxRows = tables.wqList.size();
        auto* info = reinterpret_cast<UrmaInfo*>(hostBuf.data());
        // One context row per peer today. Kept as a field so a future "k jetties per
        // peer" layout is a value change rather than a format change.
        info->rowsPerPeer = kUrmaQpNum;
        info->rankCount = rankCount_;
        info->notifyPoolPtr = reinterpret_cast<uint64_t>(notifyPoolDevice_);
        info->layout = Layout();
        info->jettyCount = static_cast<uint32_t>(ctxRows);

        uint8_t* devAddr = static_cast<uint8_t*>(urmaInfoDevice_) + sizeof(UrmaInfo);
        info->sqPtr = reinterpret_cast<uint64_t>(devAddr);
        devAddr += sizeof(UrmaWQCtx) * ctxRows;
        info->rqPtr = reinterpret_cast<uint64_t>(devAddr);
        devAddr += sizeof(UrmaWQCtx) * ctxRows;
        info->scqPtr = reinterpret_cast<uint64_t>(devAddr);
        devAddr += sizeof(UrmaCqCtx) * ctxRows;
        info->rcqPtr = reinterpret_cast<uint64_t>(devAddr);
        devAddr += sizeof(UrmaCqCtx) * ctxRows;
        info->memPtr = reinterpret_cast<uint64_t>(devAddr);

        uint8_t* hostAddr = hostBuf.data() + sizeof(UrmaInfo);
        auto* sqArr = reinterpret_cast<UrmaWQCtx*>(hostAddr);
        hostAddr += sizeof(UrmaWQCtx) * ctxRows;
        auto* rqArr = reinterpret_cast<UrmaWQCtx*>(hostAddr);
        hostAddr += sizeof(UrmaWQCtx) * ctxRows;
        auto* scqArr = reinterpret_cast<UrmaCqCtx*>(hostAddr);
        hostAddr += sizeof(UrmaCqCtx) * ctxRows;
        auto* rcqArr = reinterpret_cast<UrmaCqCtx*>(hostAddr);
        hostAddr += sizeof(UrmaCqCtx) * ctxRows;
        auto* memArr = reinterpret_cast<UrmaMemInfo*>(hostAddr);

        for (size_t row = 0; row < ctxRows; ++row) {
            sqArr[row] = tables.wqList[row];
            rqArr[row] = tables.wqList[row];
            scqArr[row] = tables.cqList[row];
            rcqArr[row] = tables.cqList[row];
        }
        for (size_t row = 0; row < tables.memList.size(); ++row) {
            memArr[row] = tables.memList[row];
        }
    }

    static uint32_t Log2U32(uint32_t n) { return (n <= 1) ? 0 : __builtin_ctz(n); }

    static void FreeDeviceAddr(void*& addr)
    {
        if (addr) {
            aclrtFree(addr);
            addr = nullptr;
        }
    }

    static constexpr const char* kUrmaSymMemTag = "pto_urma_sym";
    static constexpr const char* kUrmaNotifyMemTag = "pto_urma_notify";
    static constexpr uint32_t kUrmaQpNum = 1U;
    static constexpr uint32_t kUrmaRegisteredMemCount = 2U;
    static constexpr uint64_t kDeviceVaThreshold = 0x100000000000ULL;
    static constexpr CommProtocol kCommProtocolUbcCtp = static_cast<CommProtocol>(4);
    static constexpr CommProtocol kCommProtocolUbcTp = static_cast<CommProtocol>(5);

    HcclComm comm_{nullptr};
    uint32_t rankId_{0};
    uint32_t rankCount_{0};
    void* symmetricAddr_{nullptr};
    uint64_t symmetricSize_{0};
    HcclMemHandle memHandle_{nullptr};
    HcclMemHandle notifyMemHandle_{nullptr};
    HcclMemHandle memHandles_[kUrmaRegisteredMemCount]{};

    uint32_t sharedJettyCount_{0};
    uint32_t sqDepth_{0};
    std::string sharedTagPrefix_{kDefaultSharedTagPrefix};

    // SharedPool only: channels per jetty (== rankCount_ - 1) and the peer order the
    // descs were submitted in, which is the order hcomm returns handles in.
    uint32_t channelsPerJetty_{0};
    std::vector<uint32_t> sharedPeerOrder_;

    // PerPeer: one handle per peer in ascending peer order, self skipped.
    // SharedPool: jetty-major, jetty * channelsPerJetty_ + slot.
    std::vector<ChannelHandle> channelHandles_;

    void* urmaInfoDevice_{nullptr};
    void* eidDevice_{nullptr};
    void* notifyPoolDevice_{nullptr};
    uint64_t notifyPoolSize_{0};
    std::vector<uint64_t> peerBaseAddrs_;

    bool initialized_{false};
};

} // namespace urma
} // namespace comm
} // namespace pto

#endif // PTO_COMM_ASYNC_URMA_WORKSPACE_MANAGER_HPP
