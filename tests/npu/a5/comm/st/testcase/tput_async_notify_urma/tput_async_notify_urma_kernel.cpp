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
#include "tput_async_notify_urma_kernel.h"

namespace {

constexpr uint32_t kPayloadElems = 4U;
constexpr uint32_t kNotifyPosts = 65U;
constexpr uint32_t kTotalPayloadElems = kPayloadElems * kNotifyPosts;
constexpr uint32_t kSignalPollLimit = 10000000U;
constexpr size_t kCanaryBeforeOffset = 0U;
constexpr size_t kSignalOffset = 4U;
constexpr size_t kCanaryAfterOffset = 8U;
constexpr size_t kStatusOffset = 64U;
constexpr size_t kSourceOffset = 128U;
constexpr size_t kTargetOffset = 1216U;
constexpr size_t kGetSourceOffset = 2304U;
constexpr size_t kGetDestinationOffset = 2336U;
constexpr size_t kCommunicationBytes = 2400U;
constexpr int32_t kCanaryBefore = 0x13572468;
constexpr int32_t kCanaryAfter = 0x24681357;
constexpr int32_t kSetValue = 37;
constexpr int32_t kAddInitial = 11;
constexpr int32_t kAddValue = 5;
constexpr int32_t kMixedSetValue = 100;
constexpr int32_t kMixedAddValue = 7;
constexpr int32_t kPoison = -777777;

enum DeviceStatus : int32_t {
    kDeviceSuccess = 0,
    kDeviceSessionFailure = -1,
    kDeviceSubmitFailure = -2,
    kDeviceWaitFailure = -3,
    kDeviceSignalTimeout = -4,
    kDevicePayloadMismatch = -5,
    kDeviceGetMismatch = -6,
};

using Global4 = pto::GlobalTensor<
    int32_t, pto::Shape<1, 1, 1, 1, kPayloadElems>,
    pto::Stride<kPayloadElems, kPayloadElems, kPayloadElems, kPayloadElems, 1>, pto::Layout::ND>;

AICORE inline int32_t ExpectedSignal(uint32_t mode)
{
    if (mode == static_cast<uint32_t>(UrmaNotifyMode::Set)) {
        return kSetValue;
    }
    if (mode == static_cast<uint32_t>(UrmaNotifyMode::AtomicAdd)) {
        return kAddInitial + kAddValue;
    }
    if (mode == static_cast<uint32_t>(UrmaNotifyMode::AtomicAddRingReuse)) {
        return static_cast<int32_t>(kNotifyPosts);
    }
    return kMixedSetValue + kMixedAddValue;
}

AICORE inline uint32_t PayloadCount(uint32_t mode)
{
    if (mode == static_cast<uint32_t>(UrmaNotifyMode::AtomicAddRingReuse)) {
        return kNotifyPosts;
    }
    if (mode == static_cast<uint32_t>(UrmaNotifyMode::MixedAsyncOperations)) {
        return 3U;
    }
    return 1U;
}

AICORE inline void StoreStatus(__gm__ uint8_t* base, int32_t status)
{
    __gm__ int32_t* statusAddress = reinterpret_cast<__gm__ int32_t*>(base + kStatusOffset);
    *statusAddress = status;
    pipe_barrier(PIPE_ALL);
    dcci(reinterpret_cast<__gm__ void*>(statusAddress), cache_line_t::SINGLE_CACHE_LINE);
}

__global__ AICORE void TPutAsyncNotifyUrmaKernel(
    __gm__ uint8_t* localBase, __gm__ uint8_t* urmaWorkspace, uint32_t rankId, uint32_t rootRank, uint32_t mode)
{
#ifdef PTO_URMA_SUPPORTED
    const uint32_t targetPeer = rootRank + 1U;
    __gm__ int32_t* localSignal = reinterpret_cast<__gm__ int32_t*>(localBase + kSignalOffset);
    __gm__ int32_t* localSource = reinterpret_cast<__gm__ int32_t*>(localBase + kSourceOffset);
    __gm__ int32_t* localTarget = reinterpret_cast<__gm__ int32_t*>(localBase + kTargetOffset);

    if (rankId != rootRank) {
        pto::comm::Signal signal(localSignal);
        bool signaled = false;
        for (uint32_t poll = 0U; poll < kSignalPollLimit; ++poll) {
            if (pto::comm::TTEST(signal, ExpectedSignal(mode), pto::comm::WaitCmp::EQ)) {
                signaled = true;
                break;
            }
        }
        if (!signaled) {
            StoreStatus(localBase, kDeviceSignalTimeout);
            return;
        }

        __asm__ __volatile__("");
        dcci(static_cast<__gm__ void*>(0), cache_line_t::ENTIRE_DATA_CACHE);
        __asm__ __volatile__("");
        const uint32_t elementCount = PayloadCount(mode) * kPayloadElems;
        for (uint32_t index = 0U; index < elementCount; ++index) {
            if (localTarget[index] != static_cast<int32_t>(1000U + index)) {
                StoreStatus(localBase, kDevicePayloadMismatch);
                return;
            }
        }
        StoreStatus(localBase, kDeviceSuccess);
        return;
    }

    pto::comm::AsyncSession session;
    if (!pto::comm::BuildAsyncSession<pto::comm::DmaEngine::URMA>(urmaWorkspace, session)) {
        StoreStatus(localBase, kDeviceSessionFailure);
        return;
    }

    const uint64_t peerBase = pto::comm::urma::UrmaPeerMrBaseAddr(urmaWorkspace, targetPeer);
    __gm__ int32_t* remoteTarget = reinterpret_cast<__gm__ int32_t*>(peerBase + kTargetOffset);
    pto::comm::Signal remoteSignal(reinterpret_cast<__gm__ int32_t*>(peerBase + kSignalOffset));
    pto::comm::AsyncEvent finalEvent;

    if (mode == static_cast<uint32_t>(UrmaNotifyMode::Set) ||
        mode == static_cast<uint32_t>(UrmaNotifyMode::AtomicAdd)) {
        Global4 source(localSource);
        Global4 target(remoteTarget);
        const bool isSet = mode == static_cast<uint32_t>(UrmaNotifyMode::Set);
        finalEvent = pto::comm::TPUT_ASYNC_NOTIFY<pto::comm::DmaEngine::URMA>(
            target, source, remoteSignal, isSet ? kSetValue : kAddValue,
            isSet ? pto::comm::NotifyOp::Set : pto::comm::NotifyOp::AtomicAdd, session, targetPeer);
    } else if (mode == static_cast<uint32_t>(UrmaNotifyMode::AtomicAddRingReuse)) {
        for (uint32_t post = 0U; post < kNotifyPosts; ++post) {
            Global4 source(localSource + post * kPayloadElems);
            Global4 target(remoteTarget + post * kPayloadElems);
            finalEvent = pto::comm::TPUT_ASYNC_NOTIFY<pto::comm::DmaEngine::URMA>(
                target, source, remoteSignal, 1, pto::comm::NotifyOp::AtomicAdd, session, targetPeer);
            if (!finalEvent.valid()) {
                StoreStatus(localBase, kDeviceSubmitFailure);
                return;
            }
        }
    } else {
        Global4 putSource(localSource);
        Global4 putTarget(remoteTarget);
        Global4 setSource(localSource + kPayloadElems);
        Global4 setTarget(remoteTarget + kPayloadElems);
        Global4 addSource(localSource + 2U * kPayloadElems);
        Global4 addTarget(remoteTarget + 2U * kPayloadElems);
        Global4 getSource(reinterpret_cast<__gm__ int32_t*>(peerBase + kGetSourceOffset));
        Global4 getDestination(reinterpret_cast<__gm__ int32_t*>(localBase + kGetDestinationOffset));

        const pto::comm::AsyncEvent putEvent =
            pto::comm::TPUT_ASYNC<pto::comm::DmaEngine::URMA>(putTarget, putSource, session, targetPeer);
        const pto::comm::AsyncEvent setEvent = pto::comm::TPUT_ASYNC_NOTIFY<pto::comm::DmaEngine::URMA>(
            setTarget, setSource, remoteSignal, kMixedSetValue, pto::comm::NotifyOp::Set, session, targetPeer);
        const pto::comm::AsyncEvent getEvent =
            pto::comm::TGET_ASYNC<pto::comm::DmaEngine::URMA>(getDestination, getSource, session, targetPeer);
        finalEvent = pto::comm::TPUT_ASYNC_NOTIFY<pto::comm::DmaEngine::URMA>(
            addTarget, addSource, remoteSignal, kMixedAddValue, pto::comm::NotifyOp::AtomicAdd, session, targetPeer);
        if (!putEvent.valid() || !setEvent.valid() || !getEvent.valid()) {
            StoreStatus(localBase, kDeviceSubmitFailure);
            return;
        }
    }

    if (!finalEvent.valid()) {
        StoreStatus(localBase, kDeviceSubmitFailure);
        return;
    }
    if (!finalEvent.Wait(session) || !finalEvent.Test(session)) {
        StoreStatus(localBase, kDeviceWaitFailure);
        return;
    }
    if (mode == static_cast<uint32_t>(UrmaNotifyMode::MixedAsyncOperations)) {
        __gm__ int32_t* getDestination = reinterpret_cast<__gm__ int32_t*>(localBase + kGetDestinationOffset);
        pto::comm::urma::DcciCachelines(
            reinterpret_cast<__gm__ uint8_t*>(getDestination), kPayloadElems * sizeof(int32_t));
        for (uint32_t index = 0U; index < kPayloadElems; ++index) {
            if (getDestination[index] != static_cast<int32_t>(9000U + index)) {
                StoreStatus(localBase, kDeviceGetMismatch);
                return;
            }
        }
    }
    StoreStatus(localBase, kDeviceSuccess);
#else
    (void)localBase;
    (void)urmaWorkspace;
    (void)rankId;
    (void)rootRank;
    (void)mode;
#endif
}

bool ValidateAllRanks(bool localPassed, int nRanks)
{
    const uint8_t local = localPassed ? 1U : 0U;
    std::vector<uint8_t> states(static_cast<size_t>(nRanks), 0U);
    if (CommMpiAllgather(&local, 1, states.data(), 1) != 0) {
        return false;
    }
    for (const uint8_t state : states) {
        if (state == 0U) {
            return false;
        }
    }
    return true;
}

UrmaNotifyMode gMode = UrmaNotifyMode::Set;

bool RunPerRank(int rankId, int nRanks, int nDevices, int firstDeviceId, int firstRankId, int rootRank)
{
    UrmaTestContext context;
    if (!context.Setup(rankId, nRanks, nDevices, firstDeviceId, rootRank, kCommunicationBytes)) {
        return false;
    }

    auto* base = reinterpret_cast<uint8_t*>(context.devBuf);
    std::vector<int32_t> source(kTotalPayloadElems);
    std::vector<int32_t> target(kTotalPayloadElems, kPoison);
    for (uint32_t index = 0U; index < kTotalPayloadElems; ++index) {
        source[index] = static_cast<int32_t>(1000U + index);
    }
    const int32_t initialSignal = gMode == UrmaNotifyMode::AtomicAdd ? kAddInitial : 0;
    const int32_t statusInitial = 1;
    int32_t getSource[kPayloadElems] = {9000, 9001, 9002, 9003};
    int32_t getDestination[kPayloadElems] = {kPoison, kPoison, kPoison, kPoison};

    bool localPassed = aclrtMemcpy(
                           base + kCanaryBeforeOffset, sizeof(kCanaryBefore), &kCanaryBefore, sizeof(kCanaryBefore),
                           ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS &&
                       aclrtMemcpy(
                           base + kSignalOffset, sizeof(initialSignal), &initialSignal, sizeof(initialSignal),
                           ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS &&
                       aclrtMemcpy(
                           base + kCanaryAfterOffset, sizeof(kCanaryAfter), &kCanaryAfter, sizeof(kCanaryAfter),
                           ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS &&
                       aclrtMemcpy(
                           base + kStatusOffset, sizeof(statusInitial), &statusInitial, sizeof(statusInitial),
                           ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS &&
                       aclrtMemcpy(
                           base + kSourceOffset, source.size() * sizeof(int32_t), source.data(),
                           source.size() * sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS &&
                       aclrtMemcpy(
                           base + kTargetOffset, target.size() * sizeof(int32_t), target.data(),
                           target.size() * sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS &&
                       aclrtMemcpy(
                           base + kGetSourceOffset, sizeof(getSource), getSource, sizeof(getSource),
                           ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS &&
                       aclrtMemcpy(
                           base + kGetDestinationOffset, sizeof(getDestination), getDestination, sizeof(getDestination),
                           ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS;
    if (!ValidateAllRanks(localPassed, nRanks)) {
        context.Cleanup();
        return false;
    }

    CommMpiBarrier();
    TPutAsyncNotifyUrmaKernel<<<1, nullptr, context.stream>>>(
        base, reinterpret_cast<uint8_t*>(context.urmaMgr.GetWorkspaceAddr()), static_cast<uint32_t>(rankId),
        static_cast<uint32_t>(rootRank), static_cast<uint32_t>(gMode));
    const int synchronizeResult = aclrtSynchronizeStream(context.stream);
    CommMpiBarrier();

    int32_t status = 1;
    int32_t signal = 0;
    int32_t canaryBefore = 0;
    int32_t canaryAfter = 0;
    localPassed =
        synchronizeResult == ACL_SUCCESS &&
        aclrtMemcpy(&status, sizeof(status), base + kStatusOffset, sizeof(status), ACL_MEMCPY_DEVICE_TO_HOST) ==
            ACL_SUCCESS &&
        aclrtMemcpy(&signal, sizeof(signal), base + kSignalOffset, sizeof(signal), ACL_MEMCPY_DEVICE_TO_HOST) ==
            ACL_SUCCESS &&
        aclrtMemcpy(
            &canaryBefore, sizeof(canaryBefore), base + kCanaryBeforeOffset, sizeof(canaryBefore),
            ACL_MEMCPY_DEVICE_TO_HOST) == ACL_SUCCESS &&
        aclrtMemcpy(
            &canaryAfter, sizeof(canaryAfter), base + kCanaryAfterOffset, sizeof(canaryAfter),
            ACL_MEMCPY_DEVICE_TO_HOST) == ACL_SUCCESS;
    localPassed =
        localPassed && status == kDeviceSuccess && canaryBefore == kCanaryBefore && canaryAfter == kCanaryAfter;
    if (rankId != rootRank) {
        const uint32_t posts = gMode == UrmaNotifyMode::AtomicAddRingReuse   ? kNotifyPosts :
                               gMode == UrmaNotifyMode::MixedAsyncOperations ? 3U :
                                                                               1U;
        const int32_t expectedSignal = gMode == UrmaNotifyMode::Set       ? kSetValue :
                                       gMode == UrmaNotifyMode::AtomicAdd ? kAddInitial + kAddValue :
                                       gMode == UrmaNotifyMode::AtomicAddRingReuse ?
                                                                            static_cast<int32_t>(kNotifyPosts) :
                                                                            kMixedSetValue + kMixedAddValue;
        localPassed = localPassed && signal == expectedSignal;
        std::vector<int32_t> actual(posts * kPayloadElems);
        localPassed = localPassed && aclrtMemcpy(
                                         actual.data(), actual.size() * sizeof(int32_t), base + kTargetOffset,
                                         actual.size() * sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST) == ACL_SUCCESS;
        for (size_t index = 0U; localPassed && index < actual.size(); ++index) {
            localPassed = actual[index] == source[index];
        }
    }
    if (!localPassed) {
        std::cerr << "[ERROR] URMA notify case failed: rank=" << rankId << " status=" << status << " signal=" << signal
                  << " mode=" << static_cast<uint32_t>(gMode) << std::endl;
    }

    const bool allPassed = ValidateAllRanks(localPassed, nRanks);
    context.Cleanup();
    return allPassed;
}

} // namespace

bool RunTPutAsyncNotifyUrma(int nRanks, int nDevices, int firstRankId, int firstDeviceId, UrmaNotifyMode mode)
{
    if (nRanks != 2 || firstRankId != 0) {
        return false;
    }
    gMode = mode;
    return RunUrmaTestMpiLaunch(nRanks, nDevices, firstRankId, firstDeviceId, RunPerRank);
}
