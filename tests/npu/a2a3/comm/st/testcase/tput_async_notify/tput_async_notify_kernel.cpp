/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
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
#include "pto/common/pto_tile.hpp"
#include "../common.hpp"

namespace {

constexpr uint32_t kElemCount = 256U;
constexpr uint64_t kBlockBytes = 64U;
constexpr uint32_t kSignalPollLimit = 10000000U;
constexpr uint32_t kSignalSlotInt32Count = 2U;
constexpr uint32_t kGuardInt32Count = 16U;
constexpr int32_t kGuardValue = 0x13579BDF;
constexpr int32_t kSetValue = 7;
constexpr int32_t kAddInitial = 5;
constexpr int32_t kAddValue = 3;
constexpr uint32_t kDeviceSuccess = 0U;
constexpr uint32_t kDeviceInvalidEvent = 1U;
constexpr uint32_t kDeviceWaitFailed = 2U;
constexpr uint32_t kDeviceSignalTimeout = 3U;
constexpr uint32_t kDevicePayloadMismatch = 4U;

using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
using Global = pto::GlobalTensor<int32_t, ShapeDyn, StrideDyn, pto::Layout::ND>;
using ScratchTile = pto::Tile<pto::TileType::Vec, uint8_t, 1, pto::comm::sdma::UB_ALIGN_SIZE>;

__global__ AICORE void TPutAsyncNotifyKernel(
    __gm__ int32_t* commBuf, __gm__ CommDeviceContext* hcclCtx, __gm__ uint8_t* sdmaWorkspace, uint32_t queueNum,
    int32_t signalValue, int32_t expectedSignal, uint32_t notifyOp, uint32_t postCount)
{
    __gm__ int32_t* sendBuf = commBuf;
    __gm__ int32_t* recvBuf = sendBuf + kElemCount;
    __gm__ int32_t* priorRecvBuf = recvBuf + kElemCount;
    __gm__ int32_t* localSignal = priorRecvBuf + kElemCount;
    __gm__ int32_t* localGuard = localSignal + kSignalSlotInt32Count;
    __gm__ uint32_t* localStatus = reinterpret_cast<__gm__ uint32_t*>(localGuard + kGuardInt32Count);
    const uint32_t rank = hcclCtx->rankId;

    ShapeDyn shape(1, 1, 1, 1, kElemCount);
    StrideDyn stride(kElemCount, kElemCount, kElemCount, kElemCount, 1);

    if (rank == 0U) {
        ScratchTile scratchTile;
        TASSIGN(scratchTile, 0x0);
        pto::comm::AsyncSession session;
        const pto::comm::sdma::SdmaBaseConfig config{kBlockBytes, 0U, queueNum};
        if (!pto::comm::BuildAsyncSession(scratchTile, sdmaWorkspace, session, 0U, config, 0U)) {
            *localStatus = kDeviceInvalidEvent;
            pipe_barrier(PIPE_ALL);
            return;
        }

        Global src(sendBuf, shape, stride);
        Global priorDst(CommRemotePtr(hcclCtx, priorRecvBuf, 1), shape, stride);
        Global dst(CommRemotePtr(hcclCtx, recvBuf, 1), shape, stride);
        pto::comm::Signal signal(CommRemotePtr(hcclCtx, localSignal, 1));
        // First use all configured queues. Waiting only for the following notify event
        // must fence those queues locally through runtimeCtx.usedQueueCount. It does not
        // make the remote signal order prior transfers from other queues.
        (void)pto::comm::TPUT_ASYNC(priorDst, src, session);
        const pto::comm::NotifyOp op = notifyOp == 0U ? pto::comm::NotifyOp::Set : pto::comm::NotifyOp::AtomicAdd;
        pto::comm::AsyncEvent event;
        for (uint32_t post = 0U; post < postCount; ++post) {
            const int32_t value = signalValue + static_cast<int32_t>(post);
            event = pto::comm::TPUT_ASYNC_NOTIFY(dst, src, signal, value, op, session);
            if (!event.valid()) {
                break;
            }
        }
        if (!event.valid()) {
            *localStatus = kDeviceInvalidEvent;
        } else if (!event.Wait(session)) {
            *localStatus = kDeviceWaitFailed;
        } else {
            *localStatus = kDeviceSuccess;
        }
        pipe_barrier(PIPE_ALL);
        return;
    }

    pto::comm::Signal signal(localSignal);
    bool signaled = false;
    for (uint32_t poll = 0U; poll < kSignalPollLimit; ++poll) {
        if (pto::comm::TTEST(signal, expectedSignal, pto::comm::WaitCmp::GE)) {
            signaled = true;
            break;
        }
    }
    if (!signaled) {
        *localStatus = kDeviceSignalTimeout;
        pipe_barrier(PIPE_ALL);
        return;
    }

    __asm__ __volatile__("");
    dcci(static_cast<__gm__ void*>(0), cache_line_t::ENTIRE_DATA_CACHE);
    __asm__ __volatile__("");
    for (uint32_t i = 0U; i < kElemCount; ++i) {
        // Signal visibility only orders the notify payload on queue 0.
        if (recvBuf[i] != static_cast<int32_t>(1000U + i)) {
            *localStatus = kDevicePayloadMismatch;
            pipe_barrier(PIPE_ALL);
            return;
        }
    }
    *localStatus = kDeviceSuccess;
    pipe_barrier(PIPE_ALL);
}

bool RunNotifyCase(
    int rankId, int nRanks, int nDevices, int firstDeviceId, const HcclRootInfo* rootInfo, uint32_t queueNum,
    int32_t signalInitial, int32_t signalValue, int32_t expectedSignal, pto::comm::NotifyOp notifyOp,
    uint32_t postCount)
{
    TestContext ctx;
    if (!ctx.Init(rankId, nRanks, nDevices, firstDeviceId, rootInfo)) {
        return false;
    }

    uint64_t localWinBase = ctx.hostCtx.windowsIn[rankId];
    size_t winOffset = 0U;
    const size_t commBytes = 3U * kElemCount * sizeof(int32_t) +
                             (kSignalSlotInt32Count + kGuardInt32Count) * sizeof(int32_t) + sizeof(uint32_t);
    auto* commBuf = reinterpret_cast<int32_t*>(WindowAlloc(localWinBase, winOffset, commBytes));
    auto* sendBuf = commBuf;
    auto* recvBuf = sendBuf + kElemCount;
    auto* priorRecvBuf = recvBuf + kElemCount;
    auto* signal = priorRecvBuf + kElemCount;
    auto* guard = signal + kSignalSlotInt32Count;
    auto* status = reinterpret_cast<uint32_t*>(guard + kGuardInt32Count);

    std::vector<int32_t> sendHost(kElemCount);
    std::vector<int32_t> recvHost(kElemCount, -1);
    for (uint32_t i = 0U; i < kElemCount; ++i) {
        sendHost[i] = static_cast<int32_t>(1000U + i);
    }
    uint32_t statusHost = UINT32_MAX;
    std::vector<int32_t> signalSlotHost(kSignalSlotInt32Count, 0);
    std::vector<int32_t> guardHost(kGuardInt32Count, kGuardValue);
    signalSlotHost[0] = signalInitial;
    ctx.aclStatus |= aclrtMemcpy(
        sendBuf, kElemCount * sizeof(int32_t), sendHost.data(), kElemCount * sizeof(int32_t),
        ACL_MEMCPY_HOST_TO_DEVICE);
    ctx.aclStatus |= aclrtMemcpy(
        recvBuf, kElemCount * sizeof(int32_t), recvHost.data(), kElemCount * sizeof(int32_t),
        ACL_MEMCPY_HOST_TO_DEVICE);
    ctx.aclStatus |= aclrtMemcpy(
        priorRecvBuf, kElemCount * sizeof(int32_t), recvHost.data(), kElemCount * sizeof(int32_t),
        ACL_MEMCPY_HOST_TO_DEVICE);
    ctx.aclStatus |= aclrtMemcpy(
        signal, kSignalSlotInt32Count * sizeof(int32_t), signalSlotHost.data(), kSignalSlotInt32Count * sizeof(int32_t),
        ACL_MEMCPY_HOST_TO_DEVICE);
    ctx.aclStatus |= aclrtMemcpy(
        guard, kGuardInt32Count * sizeof(int32_t), guardHost.data(), kGuardInt32Count * sizeof(int32_t),
        ACL_MEMCPY_HOST_TO_DEVICE);
    ctx.aclStatus |= aclrtMemcpy(status, sizeof(uint32_t), &statusHost, sizeof(uint32_t), ACL_MEMCPY_HOST_TO_DEVICE);

    SdmaWorkspaceManager sdmaMgr;
    if (ctx.aclStatus != 0 || !sdmaMgr.Init()) {
        std::cerr << "[ERROR] notify test initialization failed on rank " << rankId << std::endl;
        return false;
    }

    HcclHostBarrier(ctx.comm, ctx.stream);
    TPutAsyncNotifyKernel<<<1, nullptr, ctx.stream>>>(
        commBuf, ctx.deviceCtx, reinterpret_cast<uint8_t*>(sdmaMgr.GetWorkspaceAddr()), queueNum, signalValue,
        expectedSignal, notifyOp == pto::comm::NotifyOp::Set ? 0U : 1U, postCount);
    ctx.aclStatus |= aclrtSynchronizeStream(ctx.stream);
    HcclHostBarrier(ctx.comm, ctx.stream);

    int32_t signalHost = 0;
    ctx.aclStatus |= aclrtMemcpy(&statusHost, sizeof(uint32_t), status, sizeof(uint32_t), ACL_MEMCPY_DEVICE_TO_HOST);
    ctx.aclStatus |= aclrtMemcpy(&signalHost, sizeof(int32_t), signal, sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST);
    ctx.aclStatus |= aclrtMemcpy(
        signalSlotHost.data(), kSignalSlotInt32Count * sizeof(int32_t), signal, kSignalSlotInt32Count * sizeof(int32_t),
        ACL_MEMCPY_DEVICE_TO_HOST);
    ctx.aclStatus |= aclrtMemcpy(
        guardHost.data(), kGuardInt32Count * sizeof(int32_t), guard, kGuardInt32Count * sizeof(int32_t),
        ACL_MEMCPY_DEVICE_TO_HOST);
    std::vector<int32_t> priorRecvHost(kElemCount, -1);
    if (rankId == 1) {
        ctx.aclStatus |= aclrtMemcpy(
            recvHost.data(), kElemCount * sizeof(int32_t), recvBuf, kElemCount * sizeof(int32_t),
            ACL_MEMCPY_DEVICE_TO_HOST);
        ctx.aclStatus |= aclrtMemcpy(
            priorRecvHost.data(), kElemCount * sizeof(int32_t), priorRecvBuf, kElemCount * sizeof(int32_t),
            ACL_MEMCPY_DEVICE_TO_HOST);
    }

    bool ok = ctx.aclStatus == 0 && statusHost == kDeviceSuccess;
    if (rankId == 1) {
        ok = ok && signalHost == expectedSignal;
        for (uint32_t i = 1U; ok && i < kSignalSlotInt32Count; ++i) {
            ok = signalSlotHost[i] == 0;
        }
        for (uint32_t i = 0U; ok && i < kGuardInt32Count; ++i) {
            ok = guardHost[i] == kGuardValue;
        }
        for (uint32_t i = 0U; ok && i < kElemCount; ++i) {
            ok = recvHost[i] == static_cast<int32_t>(1000U + i) && priorRecvHost[i] == static_cast<int32_t>(1000U + i);
        }
    }
    if (!ok) {
        std::cerr << "[ERROR] notify case failed: rank=" << rankId << " status=" << statusHost
                  << " signal=" << signalHost << " expectedSignal=" << expectedSignal << std::endl;
    }

    sdmaMgr.Finalize();
    return ctx.Finalize() && ok;
}

bool RunNotify(
    int nRanks, int nDevices, int firstRankId, int firstDeviceId, uint32_t queueNum, int32_t signalInitial,
    int32_t signalValue, int32_t expectedSignal, pto::comm::NotifyOp notifyOp, uint32_t postCount)
{
    if (nRanks != 2 || queueNum == 0U || postCount == 0U) {
        return false;
    }
    return ForkAndRunWithHcclRootInfo(
        nRanks, firstRankId, firstDeviceId, [&](int rankId, const HcclRootInfo* rootInfo) {
            return RunNotifyCase(
                rankId, nRanks, nDevices, firstDeviceId, rootInfo, queueNum, signalInitial, signalValue, expectedSignal,
                notifyOp, postCount);
        });
}

} // namespace

bool RunTPutAsyncNotifySet(int nRanks, int nDevices, int firstRankId, int firstDeviceId, uint32_t queueNum)
{
    return RunNotify(
        nRanks, nDevices, firstRankId, firstDeviceId, queueNum, 0, kSetValue, kSetValue, pto::comm::NotifyOp::Set, 1U);
}

bool RunTPutAsyncNotifyAdd(int nRanks, int nDevices, int firstRankId, int firstDeviceId, uint32_t queueNum)
{
    return RunNotify(
        nRanks, nDevices, firstRankId, firstDeviceId, queueNum, kAddInitial, kAddValue, kAddInitial + kAddValue,
        pto::comm::NotifyOp::AtomicAdd, 1U);
}

bool RunTPutAsyncNotifyAddRingReuse(int nRanks, int nDevices, int firstRankId, int firstDeviceId, uint32_t queueNum)
{
    constexpr uint32_t kPostCount = 65U;
    return RunNotify(
        nRanks, nDevices, firstRankId, firstDeviceId, queueNum, kAddInitial, kAddValue,
        kAddInitial + static_cast<int32_t>(kPostCount) * kAddValue +
            static_cast<int32_t>((kPostCount - 1U) * kPostCount / 2U),
        pto::comm::NotifyOp::AtomicAdd, kPostCount);
}
