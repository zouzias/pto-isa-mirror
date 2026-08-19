/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <vector>

#include <pto/pto-inst.hpp>

#include "../common.hpp"
#include "pto/common/pto_tile.hpp"
#include "tput_async_notify_kernel.h"

namespace {

constexpr uint32_t kPayloadElems = 16U;
constexpr uint32_t kQueueCount = 4U;
constexpr uint64_t kBlockBytes = kPayloadElems * sizeof(int32_t);
constexpr uint32_t kRingReusePosts = 65U;
constexpr uint32_t kMixedOrdinaryPosts = 63U;
constexpr uint32_t kSignalPollLimit = 10000000U;
constexpr int32_t kSetValue = 7;
constexpr int32_t kAddInitial = 5;
constexpr int32_t kAddValue = 3;
constexpr int32_t kCanaryBefore = 0x13572468;
constexpr int32_t kCanaryAfter = 0x24681357;

enum DeviceStatus : uint32_t {
    kDeviceSuccess = 0U,
    kDeviceSessionFailure = 1U,
    kDeviceSubmitFailure = 2U,
    kDeviceWaitFailure = 3U,
    kDeviceSignalTimeout = 4U,
    kDevicePayloadMismatch = 5U,
};

using GlobalPayload = pto::GlobalTensor<
    int32_t, pto::Shape<1, 1, 1, 1, kPayloadElems>,
    pto::Stride<kPayloadElems, kPayloadElems, kPayloadElems, kPayloadElems, 1>, pto::Layout::ND>;
using ScratchTile = pto::Tile<pto::TileType::Vec, uint8_t, 1, pto::comm::sdma::UB_ALIGN_SIZE>;

struct BufferLayout {
    __gm__ int32_t* send;
    __gm__ int32_t* receive;
    __gm__ int32_t* canaryBefore;
    __gm__ int32_t* signal;
    __gm__ int32_t* canaryAfter;
    __gm__ uint32_t* status;
};

AICORE inline BufferLayout GetLayout(__gm__ int32_t* buffer)
{
    BufferLayout layout{};
    layout.send = buffer;
    layout.receive = layout.send + kPayloadElems;
    layout.canaryBefore = layout.receive + kPayloadElems;
    layout.signal = layout.canaryBefore + 1;
    layout.canaryAfter = layout.signal + 1;
    layout.status = reinterpret_cast<__gm__ uint32_t*>(layout.canaryAfter + 1);
    return layout;
}

AICORE inline int32_t ExpectedSignal(uint32_t mode)
{
    if (mode == static_cast<uint32_t>(SdmaNotifyMode::Set)) {
        return kSetValue;
    }
    if (mode == static_cast<uint32_t>(SdmaNotifyMode::MixedAsyncOperations)) {
        return kAddInitial + 2 * kAddValue;
    }
    const uint32_t posts = mode == static_cast<uint32_t>(SdmaNotifyMode::AtomicAddRingReuse) ? kRingReusePosts : 1U;
    return kAddInitial + static_cast<int32_t>(posts) * kAddValue;
}

__global__ AICORE void TPutAsyncNotifySdmaKernel(
    __gm__ int32_t* buffer, __gm__ CommDeviceContext* hcclContext, __gm__ uint8_t* sdmaWorkspace, uint32_t rootRank,
    uint32_t targetRank, uint32_t mode, uint32_t channelGroupIdx)
{
    const BufferLayout local = GetLayout(buffer);
    const uint32_t rank = hcclContext->rankId;
    const int32_t expectedSignal = ExpectedSignal(mode);

    if (rank == rootRank) {
        ScratchTile scratchTile;
        TASSIGN(scratchTile, 0x0);
        pto::comm::AsyncSession session;
        const pto::comm::sdma::SdmaBaseConfig config{kBlockBytes, 0U, kQueueCount};
        if (!pto::comm::BuildAsyncSession(scratchTile, sdmaWorkspace, session, 0U, config, channelGroupIdx)) {
            *local.status = kDeviceSessionFailure;
            pipe_barrier(PIPE_ALL);
            return;
        }

        GlobalPayload source(local.send);
        GlobalPayload destination(CommRemotePtr(hcclContext, local.receive, static_cast<int>(targetRank)));
        pto::comm::Signal signal(CommRemotePtr(hcclContext, local.signal, static_cast<int>(targetRank)));
        const pto::comm::NotifyOp operation = mode == static_cast<uint32_t>(SdmaNotifyMode::Set) ?
                                                  pto::comm::NotifyOp::Set :
                                                  pto::comm::NotifyOp::AtomicAdd;
        const uint32_t posts = mode == static_cast<uint32_t>(SdmaNotifyMode::AtomicAddRingReuse) ? kRingReusePosts : 1U;
        pto::comm::AsyncEvent event;
        if (mode == static_cast<uint32_t>(SdmaNotifyMode::MixedAsyncOperations)) {
            // The notify posts use IDs 2 and 66. Ordinary posts between them
            // force reuse of the same 64-entry signal-value slot.
            event = pto::comm::TPUT_ASYNC<pto::comm::DmaEngine::SDMA>(destination, source, session);
            if (!event.valid()) {
                *local.status = kDeviceSubmitFailure;
                pipe_barrier(PIPE_ALL);
                return;
            }
            event = pto::comm::TPUT_ASYNC_NOTIFY<pto::comm::DmaEngine::SDMA>(
                destination, source, signal, kAddValue, pto::comm::NotifyOp::AtomicAdd, session, targetRank);
            for (uint32_t post = 0U; post < kMixedOrdinaryPosts && event.valid(); ++post) {
                event = pto::comm::TPUT_ASYNC<pto::comm::DmaEngine::SDMA>(destination, source, session);
            }
            if (event.valid()) {
                event = pto::comm::TPUT_ASYNC_NOTIFY<pto::comm::DmaEngine::SDMA>(
                    destination, source, signal, kAddValue, pto::comm::NotifyOp::AtomicAdd, session, targetRank);
            }
        } else {
            for (uint32_t post = 0U; post < posts; ++post) {
                event = pto::comm::TPUT_ASYNC_NOTIFY<pto::comm::DmaEngine::SDMA>(
                    destination, source, signal, operation == pto::comm::NotifyOp::Set ? kSetValue : kAddValue,
                    operation, session, targetRank);
                if (!event.valid()) {
                    break;
                }
            }
        }
        if (!event.valid()) {
            *local.status = kDeviceSubmitFailure;
            pipe_barrier(PIPE_ALL);
            return;
        }
        *local.status = event.Wait(session) ? kDeviceSuccess : kDeviceWaitFailure;
        pipe_barrier(PIPE_ALL);
        return;
    }

    if (rank == targetRank) {
        pto::comm::Signal signal(local.signal);
        bool signaled = false;
        for (uint32_t poll = 0U; poll < kSignalPollLimit; ++poll) {
            if (pto::comm::TTEST(signal, expectedSignal, pto::comm::WaitCmp::EQ)) {
                signaled = true;
                break;
            }
        }
        if (!signaled) {
            *local.status = kDeviceSignalTimeout;
            pipe_barrier(PIPE_ALL);
            return;
        }

        __asm__ __volatile__("");
        dcci(static_cast<__gm__ void*>(0), cache_line_t::ENTIRE_DATA_CACHE);
        __asm__ __volatile__("");
        for (uint32_t index = 0U; index < kPayloadElems; ++index) {
            if (local.receive[index] != static_cast<int32_t>(1000U + index)) {
                *local.status = kDevicePayloadMismatch;
                pipe_barrier(PIPE_ALL);
                return;
            }
        }
        *local.status = kDeviceSuccess;
        pipe_barrier(PIPE_ALL);
    }
}

bool RunPerRank(
    int rankId, int nRanks, int nDevices, int firstDeviceId, const HcclRootInfo* rootInfo, SdmaNotifyMode mode,
    uint32_t channelGroupIdx)
{
    TestContext context;
    if (!context.Init(rankId, nRanks, nDevices, firstDeviceId, rootInfo)) {
        return false;
    }

    uint64_t windowBase = context.hostCtx.windowsIn[rankId];
    size_t windowOffset = 0U;
    constexpr size_t kBufferBytes = (2U * kPayloadElems + 3U) * sizeof(int32_t) + sizeof(uint32_t);
    auto* buffer = reinterpret_cast<int32_t*>(WindowAlloc(windowBase, windowOffset, kBufferBytes));
    const BufferLayout layout = GetLayout(buffer);

    std::vector<int32_t> source(kPayloadElems);
    std::vector<int32_t> destination(kPayloadElems, -1);
    for (uint32_t index = 0U; index < kPayloadElems; ++index) {
        source[index] = static_cast<int32_t>(1000U + index);
    }
    const int32_t initialSignal = mode == SdmaNotifyMode::Set ? 0 : kAddInitial;
    uint32_t status = UINT32_MAX;
    context.aclStatus |= aclrtMemcpy(
        layout.send, kPayloadElems * sizeof(int32_t), source.data(), kPayloadElems * sizeof(int32_t),
        ACL_MEMCPY_HOST_TO_DEVICE);
    context.aclStatus |= aclrtMemcpy(
        layout.receive, kPayloadElems * sizeof(int32_t), destination.data(), kPayloadElems * sizeof(int32_t),
        ACL_MEMCPY_HOST_TO_DEVICE);
    context.aclStatus |= aclrtMemcpy(
        layout.canaryBefore, sizeof(kCanaryBefore), &kCanaryBefore, sizeof(kCanaryBefore), ACL_MEMCPY_HOST_TO_DEVICE);
    context.aclStatus |= aclrtMemcpy(
        layout.signal, sizeof(initialSignal), &initialSignal, sizeof(initialSignal), ACL_MEMCPY_HOST_TO_DEVICE);
    context.aclStatus |= aclrtMemcpy(
        layout.canaryAfter, sizeof(kCanaryAfter), &kCanaryAfter, sizeof(kCanaryAfter), ACL_MEMCPY_HOST_TO_DEVICE);
    context.aclStatus |= aclrtMemcpy(layout.status, sizeof(status), &status, sizeof(status), ACL_MEMCPY_HOST_TO_DEVICE);

    SdmaWorkspaceManager workspaceManager;
    if (context.aclStatus != ACL_SUCCESS || !workspaceManager.Init()) {
        std::cerr << "[ERROR] SDMA notify test setup failed on rank " << rankId << std::endl;
        return false;
    }

    constexpr uint32_t rootRank = 0U;
    const uint32_t targetRank = rootRank + 1U;
    HcclHostBarrier(context.comm, context.stream);
    TPutAsyncNotifySdmaKernel<<<1, nullptr, context.stream>>>(
        buffer, context.deviceCtx, reinterpret_cast<uint8_t*>(workspaceManager.GetWorkspaceAddr()), rootRank,
        targetRank, static_cast<uint32_t>(mode), channelGroupIdx);
    context.aclStatus |= aclrtSynchronizeStream(context.stream);
    HcclHostBarrier(context.comm, context.stream);

    int32_t actualSignal = 0;
    int32_t actualBefore = 0;
    int32_t actualAfter = 0;
    context.aclStatus |= aclrtMemcpy(&status, sizeof(status), layout.status, sizeof(status), ACL_MEMCPY_DEVICE_TO_HOST);
    context.aclStatus |= aclrtMemcpy(
        &actualSignal, sizeof(actualSignal), layout.signal, sizeof(actualSignal), ACL_MEMCPY_DEVICE_TO_HOST);
    context.aclStatus |= aclrtMemcpy(
        &actualBefore, sizeof(actualBefore), layout.canaryBefore, sizeof(actualBefore), ACL_MEMCPY_DEVICE_TO_HOST);
    context.aclStatus |= aclrtMemcpy(
        &actualAfter, sizeof(actualAfter), layout.canaryAfter, sizeof(actualAfter), ACL_MEMCPY_DEVICE_TO_HOST);
    if (rankId == static_cast<int>(targetRank)) {
        context.aclStatus |= aclrtMemcpy(
            destination.data(), kPayloadElems * sizeof(int32_t), layout.receive, kPayloadElems * sizeof(int32_t),
            ACL_MEMCPY_DEVICE_TO_HOST);
    }

    bool passed = context.aclStatus == ACL_SUCCESS && status == kDeviceSuccess && actualBefore == kCanaryBefore &&
                  actualAfter == kCanaryAfter;
    if (rankId == static_cast<int>(targetRank)) {
        const uint32_t posts = mode == SdmaNotifyMode::AtomicAddRingReuse ? kRingReusePosts : 1U;
        const int32_t expectedSignal =
            mode == SdmaNotifyMode::Set                  ? kSetValue :
            mode == SdmaNotifyMode::MixedAsyncOperations ? kAddInitial + 2 * kAddValue :
                                                           kAddInitial + static_cast<int32_t>(posts) * kAddValue;
        passed = passed && actualSignal == expectedSignal;
        for (uint32_t index = 0U; passed && index < kPayloadElems; ++index) {
            passed = destination[index] == source[index];
        }
    }
    if (!passed) {
        std::cerr << "[ERROR] SDMA notify case failed: rank=" << rankId << " status=" << status
                  << " signal=" << actualSignal << std::endl;
    }

    workspaceManager.Finalize();
    return context.Finalize() && passed;
}

SdmaNotifyMode gMode = SdmaNotifyMode::Set;
uint32_t gChannelGroupIdx = 0U;

bool RunEntry(int rankId, int nRanks, int nDevices, int firstDeviceId, const HcclRootInfo* rootInfo)
{
    return RunPerRank(rankId, nRanks, nDevices, firstDeviceId, rootInfo, gMode, gChannelGroupIdx);
}

} // namespace

bool RunTPutAsyncNotifySdma(
    int nRanks, int nDevices, int firstRankId, int firstDeviceId, SdmaNotifyMode mode, uint32_t channelGroupIdx)
{
    if (nRanks != 2 || firstRankId != 0) {
        return false;
    }
    gMode = mode;
    gChannelGroupIdx = channelGroupIdx;
    return ForkAndRunWithHcclRootInfo(nRanks, firstRankId, firstDeviceId, RunEntry);
}
