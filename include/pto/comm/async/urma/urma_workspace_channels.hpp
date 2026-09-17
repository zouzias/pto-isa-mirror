/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMM_ASYNC_URMA_WORKSPACE_CHANNELS_HPP
#define PTO_COMM_ASYNC_URMA_WORKSPACE_CHANNELS_HPP

// Included from UrmaWorkspaceManager's private section.
// Per-peer and SharedPool channel acquire / endpoint alignment.

bool BuildChannels() { return (aivCount_ == 0) ? BuildChannelsPerPeer() : BuildChannelsSharedPool(); }

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

    const uint32_t jettyTotal = JettyCount();
    channelHandles_.assign(static_cast<size_t>(jettyTotal) * channelsPerJetty_, 0);
    for (uint32_t jetty = 0; jetty < jettyTotal; ++jetty) {
        if (!AcquireOneJetty(jetty, links)) {
            return false;
        }
    }
    std::cerr << "[URMA] shared pool ready rank=" << rankId_ << " jetties=" << jettyTotal << " aivCount=" << aivCount_
              << " jettiesPerCore=" << jettiesPerCore_ << " channelsPerJetty=" << channelsPerJetty_ << std::endl;
    return true;
}

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
                  << (rc == HCCL_E_TIMEOUT ? " (HCCL_E_TIMEOUT: endpoint handshake did not match)" : "") << std::endl;
        return false;
    }
    return true;
}

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

#endif // PTO_COMM_ASYNC_URMA_WORKSPACE_CHANNELS_HPP
