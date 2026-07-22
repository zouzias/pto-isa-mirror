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
#include "pto/comm/async/sdma/sdma_types.hpp"
#include "pto/common/pto_tile.hpp"
#include "async_post_stability_kernel.h"
#include "common.hpp"

namespace {

constexpr int kRootRank = 0;
constexpr uint32_t kElemsPerPost = 1024;
constexpr uint32_t kTileElems = 256;
constexpr uint32_t kMaxDeferredEvents = 16;
constexpr uint32_t kSignalPollLimit = 1000000;
constexpr int32_t kConsumeAdd = 100;
constexpr int32_t kSourceBias = 7;
constexpr int32_t kRecvPoison = -777777;
constexpr int32_t kConsumePoison = -888888;
constexpr size_t kWindowSyncPrefix = 64 * sizeof(int32_t);

constexpr uint32_t kStatusEventValid = 1U << 0;
constexpr uint32_t kStatusWaitPassed = 1U << 1;
constexpr uint32_t kStatusConsumed = 1U << 2;
constexpr uint32_t kExpectedPostStatus = kStatusEventValid | kStatusWaitPassed | kStatusConsumed;

using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
using GlobalData = pto::GlobalTensor<int32_t, ShapeDyn, StrideDyn, pto::Layout::ND>;
using ScratchTile = pto::Tile<pto::TileType::Vec, uint8_t, 1, pto::comm::sdma::UB_ALIGN_SIZE>;
using ConsumeTile = pto::Tile<pto::TileType::Vec, int32_t, 1, kTileElems>;

AICORE inline void ConsumePost(__gm__ int32_t* input, __gm__ int32_t* output)
{
    ShapeDyn shape(1, 1, 1, 1, kTileElems);
    StrideDyn stride(kTileElems, kTileElems, kTileElems, kTileElems, 1);
    ConsumeTile inputTile;
    ConsumeTile outputTile;
    TASSIGN(inputTile, 0x1000);
    TASSIGN(outputTile, 0x2000);

    for (uint32_t offset = 0; offset < kElemsPerPost; offset += kTileElems) {
        GlobalData inputGlobal(input + offset, shape, stride);
        GlobalData outputGlobal(output + offset, shape, stride);
        TLOAD(inputTile, inputGlobal);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        TADDS(outputTile, inputTile, kConsumeAdd);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        TSTORE(outputGlobal, outputTile);
        pipe_barrier(PIPE_ALL);
    }
}

AICORE inline pto::comm::AsyncEvent PostOneSized(
    AsyncTransferKind transferKind, __gm__ int32_t* localSend, __gm__ int32_t* localRecv,
    __gm__ CommDeviceContext* hcclCtx, int targetRank, uint32_t transferIndex, uint32_t elemCount,
    const pto::comm::AsyncSession& session)
{
    ShapeDyn shape(1, 1, 1, 1, elemCount);
    StrideDyn stride(elemCount, elemCount, elemCount, elemCount, 1);
    const size_t offset = static_cast<size_t>(transferIndex) * kElemsPerPost;
    if (transferKind == AsyncTransferKind::TGet) {
        __gm__ int32_t* remoteSend = CommRemotePtr(hcclCtx, localSend, targetRank) + offset;
        GlobalData remoteSendGlobal(remoteSend, shape, stride);
        GlobalData localRecvGlobal(localRecv + offset, shape, stride);
        return pto::comm::TGET_ASYNC(localRecvGlobal, remoteSendGlobal, session);
    }

    __gm__ int32_t* remoteRecv = CommRemotePtr(hcclCtx, localRecv, targetRank) + offset;
    GlobalData localSendGlobal(localSend + offset, shape, stride);
    GlobalData remoteRecvGlobal(remoteRecv, shape, stride);
    return pto::comm::TPUT_ASYNC(remoteRecvGlobal, localSendGlobal, session);
}

AICORE inline pto::comm::AsyncEvent PostOne(
    AsyncTransferKind transferKind, __gm__ int32_t* localSend, __gm__ int32_t* localRecv,
    __gm__ CommDeviceContext* hcclCtx, int targetRank, uint32_t transferIndex, const pto::comm::AsyncSession& session)
{
    return PostOneSized(transferKind, localSend, localRecv, hcclCtx, targetRank, transferIndex, kElemsPerPost, session);
}

AICORE inline void NotifyTPutConsumer(
    __gm__ int32_t* localSignal, __gm__ CommDeviceContext* hcclCtx, int targetRank, uint32_t sequence)
{
    __gm__ int32_t* remoteSignal = CommRemotePtr(hcclCtx, localSignal, targetRank);
    pto::comm::Signal signal(remoteSignal);
    pipe_barrier(PIPE_ALL);
    pto::comm::TNOTIFY(signal, static_cast<int32_t>(sequence + 1), pto::comm::NotifyOp::Set);
    pipe_barrier(PIPE_ALL);
}

AICORE inline void RecordAndConsume(
    AsyncTransferKind transferKind, pto::comm::AsyncEvent& event, const pto::comm::AsyncSession& session,
    __gm__ int32_t* localRecv, __gm__ int32_t* consumeOutput, __gm__ uint32_t* postStatus, __gm__ int32_t* localSignal,
    __gm__ CommDeviceContext* hcclCtx, int targetRank, uint32_t sequence, uint32_t transferIndex)
{
    uint32_t status = 0;
    if (event.valid()) {
        status |= kStatusEventValid;
        if (event.Wait(session)) {
            status |= kStatusWaitPassed;
            if (transferKind == AsyncTransferKind::TGet) {
                dcci(static_cast<__gm__ void*>(0), ENTIRE_DATA_CACHE);
                dsb(DSB_DDR);
                const size_t offset = static_cast<size_t>(transferIndex) * kElemsPerPost;
                ConsumePost(localRecv + offset, consumeOutput + offset);
            }
            status |= kStatusConsumed;
        }
    }
    postStatus[transferIndex] = status;
    if (transferKind == AsyncTransferKind::TPut) {
        // Notify even on failure so the peer records poisoned data instead of hanging in TWAIT.
        NotifyTPutConsumer(localSignal, hcclCtx, targetRank, sequence);
    }
}

AICORE inline void ConsumeTPutPosts(
    __gm__ int32_t* localRecv, __gm__ int32_t* localSignal, __gm__ int32_t* consumeOutput, __gm__ uint32_t* postStatus,
    uint32_t rank, uint32_t postsPerPeer)
{
    pto::comm::Signal signal(localSignal);
    const uint32_t peerIndex = rank - 1;
    for (uint32_t sequence = 0; sequence < postsPerPeer; ++sequence) {
        bool signaled = false;
        for (uint32_t poll = 0; poll < kSignalPollLimit; ++poll) {
            if (pto::comm::TTEST(signal, static_cast<int32_t>(sequence + 1), pto::comm::WaitCmp::GE)) {
                signaled = true;
                break;
            }
        }
        const uint32_t transferIndex = peerIndex * postsPerPeer + sequence;
        if (!signaled) {
            postStatus[transferIndex] = 0;
            return;
        }
        dcci(static_cast<__gm__ void*>(0), ENTIRE_DATA_CACHE);
        dsb(DSB_DDR);
        const size_t offset = static_cast<size_t>(transferIndex) * kElemsPerPost;
        ConsumePost(localRecv + offset, consumeOutput + offset);
        postStatus[transferIndex] = kStatusConsumed;
    }
}

AICORE inline void NotifyTPutFailure(
    AsyncTransferKind transferKind, __gm__ int32_t* localSignal, __gm__ CommDeviceContext* hcclCtx,
    uint32_t postsPerPeer)
{
    if (transferKind != AsyncTransferKind::TPut) {
        return;
    }
    for (int targetRank = 1; targetRank < static_cast<int>(hcclCtx->rankNum); ++targetRank) {
        for (uint32_t sequence = 0; sequence < postsPerPeer; ++sequence) {
            NotifyTPutConsumer(localSignal, hcclCtx, targetRank, sequence);
        }
    }
}

AICORE inline void RunDeferredPosts(
    AsyncTransferKind transferKind, __gm__ int32_t* localSend, __gm__ int32_t* localRecv, __gm__ int32_t* localSignal,
    __gm__ int32_t* consumeOutput, __gm__ uint32_t* postStatus, __gm__ CommDeviceContext* hcclCtx, uint32_t postCount,
    uint32_t postsPerPeer, const pto::comm::AsyncSession& session)
{
    pto::comm::AsyncEvent events[kMaxDeferredEvents];
    for (int targetRank = 1; targetRank < static_cast<int>(hcclCtx->rankNum); ++targetRank) {
        const uint32_t peerIndex = static_cast<uint32_t>(targetRank - 1);
        for (uint32_t sequence = 0; sequence < postCount; ++sequence) {
            const uint32_t transferIndex = peerIndex * postsPerPeer + sequence;
            events[transferIndex] =
                PostOne(transferKind, localSend, localRecv, hcclCtx, targetRank, transferIndex, session);
        }
    }
    for (uint32_t sequence = 0; sequence < postCount; ++sequence) {
        for (int targetRank = 1; targetRank < static_cast<int>(hcclCtx->rankNum); ++targetRank) {
            const uint32_t peerIndex = static_cast<uint32_t>(targetRank - 1);
            const uint32_t transferIndex = peerIndex * postsPerPeer + sequence;
            RecordAndConsume(
                transferKind, events[transferIndex], session, localRecv, consumeOutput, postStatus, localSignal,
                hcclCtx, targetRank, sequence, transferIndex);
        }
    }
}

AICORE inline void RunImmediatePosts(
    AsyncTransferKind transferKind, __gm__ int32_t* localSend, __gm__ int32_t* localRecv, __gm__ int32_t* localSignal,
    __gm__ int32_t* consumeOutput, __gm__ uint32_t* postStatus, __gm__ CommDeviceContext* hcclCtx,
    uint32_t postsPerPeer, const pto::comm::AsyncSession& session)
{
    for (uint32_t sequence = 0; sequence < postsPerPeer; ++sequence) {
        for (int targetRank = 1; targetRank < static_cast<int>(hcclCtx->rankNum); ++targetRank) {
            const uint32_t peerIndex = static_cast<uint32_t>(targetRank - 1);
            const uint32_t transferIndex = peerIndex * postsPerPeer + sequence;
            pto::comm::AsyncEvent event =
                PostOne(transferKind, localSend, localRecv, hcclCtx, targetRank, transferIndex, session);
            RecordAndConsume(
                transferKind, event, session, localRecv, consumeOutput, postStatus, localSignal, hcclCtx, targetRank,
                sequence, transferIndex);
        }
    }
}

AICORE inline void RunLastWaitOnlyPosts(
    AsyncTransferKind transferKind, __gm__ int32_t* localSend, __gm__ int32_t* localRecv, __gm__ int32_t* localSignal,
    __gm__ int32_t* consumeOutput, __gm__ uint32_t* postStatus, __gm__ CommDeviceContext* hcclCtx,
    const pto::comm::AsyncSession& session)
{
    constexpr int kTargetRank = 1;
    constexpr uint32_t kTransferCount = 1U;
    constexpr uint32_t kSmallPostElems = kElemsPerPost / 4U;
    pto::comm::AsyncEvent firstEvent = PostOne(transferKind, localSend, localRecv, hcclCtx, kTargetRank, 0U, session);
    pto::comm::AsyncEvent lastEvent =
        PostOneSized(transferKind, localSend, localRecv, hcclCtx, kTargetRank, 0U, kSmallPostElems, session);

    uint32_t status = 0U;
    const bool eventsValid = firstEvent.valid() && lastEvent.valid();
    if (eventsValid) {
        status |= kStatusEventValid;
    }
    const bool waitPassed = eventsValid && lastEvent.Wait(session);
    if (waitPassed) {
        status |= kStatusWaitPassed;
        if (transferKind == AsyncTransferKind::TGet) {
            dcci(static_cast<__gm__ void*>(0), ENTIRE_DATA_CACHE);
            dsb(DSB_DDR);
            for (uint32_t transferIndex = 0U; transferIndex < kTransferCount; ++transferIndex) {
                const size_t offset = static_cast<size_t>(transferIndex) * kElemsPerPost;
                ConsumePost(localRecv + offset, consumeOutput + offset);
            }
        }
        status |= kStatusConsumed;
    }
    for (uint32_t transferIndex = 0U; transferIndex < kTransferCount; ++transferIndex) {
        postStatus[transferIndex] = status;
        if (transferKind == AsyncTransferKind::TPut) {
            NotifyTPutConsumer(localSignal, hcclCtx, kTargetRank, transferIndex);
        }
    }
}

AICORE inline void RunRootProducer(
    __gm__ int32_t* localSend, __gm__ int32_t* localRecv, __gm__ int32_t* localSignal, __gm__ int32_t* consumeOutput,
    __gm__ uint32_t* postStatus, __gm__ CommDeviceContext* hcclCtx, __gm__ uint8_t* sdmaWorkspace,
    AsyncTransferKind transferKind, AsyncCheckMode checkMode, uint32_t postCount, uint32_t postsPerPeer,
    uint32_t queueNum)
{
    ScratchTile scratchTile;
    TASSIGN(scratchTile, 0x0);
    pto::comm::AsyncSession session;
    const uint64_t bytesPerPost = static_cast<uint64_t>(kElemsPerPost) * sizeof(int32_t);
    pto::comm::sdma::SdmaBaseConfig config{bytesPerPost / queueNum, 0, queueNum};
    if (!pto::comm::BuildAsyncSession(scratchTile, sdmaWorkspace, session, 0, config)) {
        NotifyTPutFailure(transferKind, localSignal, hcclCtx, postsPerPeer);
        return;
    }
    if (checkMode == AsyncCheckMode::LastWaitOnly) {
        RunLastWaitOnlyPosts(
            transferKind, localSend, localRecv, localSignal, consumeOutput, postStatus, hcclCtx, session);
        return;
    }
    if (checkMode == AsyncCheckMode::Deferred) {
        RunDeferredPosts(
            transferKind, localSend, localRecv, localSignal, consumeOutput, postStatus, hcclCtx, postCount,
            postsPerPeer, session);
        return;
    }
    RunImmediatePosts(
        transferKind, localSend, localRecv, localSignal, consumeOutput, postStatus, hcclCtx, postsPerPeer, session);
}

__global__ AICORE void AsyncPostStabilityKernel(
    __gm__ int32_t* localSend, __gm__ int32_t* localRecv, __gm__ int32_t* localSignal, __gm__ int32_t* consumeOutput,
    __gm__ uint32_t* postStatus, __gm__ CommDeviceContext* hcclCtx, __gm__ uint8_t* sdmaWorkspace,
    AsyncTransferKind transferKind, AsyncCheckMode checkMode, uint32_t postCount, uint32_t rounds, uint32_t queueNum)
{
    const int rank = static_cast<int>(hcclCtx->rankId);
    const uint32_t postsPerPeer = postCount * rounds;
    if (transferKind == AsyncTransferKind::TPut && rank != kRootRank) {
        ConsumeTPutPosts(localRecv, localSignal, consumeOutput, postStatus, static_cast<uint32_t>(rank), postsPerPeer);
        pipe_barrier(PIPE_ALL);
        return;
    }
    if (rank != kRootRank) {
        pipe_barrier(PIPE_ALL);
        return;
    }
    RunRootProducer(
        localSend, localRecv, localSignal, consumeOutput, postStatus, hcclCtx, sdmaWorkspace, transferKind, checkMode,
        postCount, postsPerPeer, queueNum);
    pipe_barrier(PIPE_ALL);
}

int32_t SourceValue(int rank, size_t index) { return static_cast<int32_t>(rank * 1000000 + index + kSourceBias); }

const char* TransferName(AsyncTransferKind transferKind)
{
    return transferKind == AsyncTransferKind::TGet ? "TGET_ASYNC" : "TPUT_ASYNC";
}

const char* ModeName(AsyncCheckMode checkMode)
{
    if (checkMode == AsyncCheckMode::Deferred) {
        return "deferred";
    }
    return checkMode == AsyncCheckMode::LastWaitOnly ? "last_wait_only" : "immediate";
}

bool AllRanksReady(bool localReady, int nRanks)
{
    bool allReady = true;
    const int mpiRank = CommMpiRank();
    for (int root = 0; root < nRanks; ++root) {
        uint8_t ready = mpiRank == root && localReady ? 1 : 0;
        CommMpiBcast(&ready, sizeof(ready), COMM_MPI_CHAR, root);
        allReady = allReady && ready != 0;
    }
    CommMpiBarrier();
    return allReady;
}

struct TransferLayout {
    uint32_t postsPerPeer;
    uint32_t peerCount;
    uint32_t totalTransfers;
    size_t totalElems;
    size_t dataBytes;
    size_t statusBytes;
    size_t requiredWindowBytes;
};

struct HostBuffers {
    std::vector<int32_t> source;
    std::vector<int32_t> recv;
    std::vector<int32_t> consumed;
    std::vector<uint32_t> status;
};

struct DeviceResources {
    int32_t* localSend{nullptr};
    int32_t* localRecv{nullptr};
    int32_t* localSignal{nullptr};
    int32_t* consume{nullptr};
    uint32_t* status{nullptr};
    SdmaWorkspaceManager sdmaManager;
    bool sdmaInitialized{false};
};

struct ConsumerRange {
    uint32_t firstTransfer;
    uint32_t transferCount;
};

TransferLayout MakeTransferLayout(int nRanks, uint32_t postCount, uint32_t rounds)
{
    TransferLayout layout{};
    layout.postsPerPeer = postCount * rounds;
    layout.peerCount = static_cast<uint32_t>(nRanks - 1);
    layout.totalTransfers = layout.postsPerPeer * layout.peerCount;
    layout.totalElems = static_cast<size_t>(layout.totalTransfers) * kElemsPerPost;
    layout.dataBytes = layout.totalElems * sizeof(int32_t);
    layout.statusBytes = static_cast<size_t>(layout.totalTransfers) * sizeof(uint32_t);
    layout.requiredWindowBytes = kWindowSyncPrefix + 2 * layout.dataBytes + sizeof(int32_t);
    return layout;
}

HostBuffers MakeHostBuffers(const TransferLayout& layout, int rankId)
{
    HostBuffers host{
        std::vector<int32_t>(layout.totalElems), std::vector<int32_t>(layout.totalElems, kRecvPoison),
        std::vector<int32_t>(layout.totalElems, kConsumePoison), std::vector<uint32_t>(layout.totalTransfers, 0)};
    for (size_t index = 0; index < layout.totalElems; ++index) {
        host.source[index] = SourceValue(rankId, index);
    }
    return host;
}

bool AllocateDeviceMemory(const TransferLayout& layout, DeviceResources& device)
{
    if (aclrtMalloc(reinterpret_cast<void**>(&device.consume), layout.dataBytes, ACL_MEM_MALLOC_HUGE_FIRST) !=
        ACL_SUCCESS) {
        return false;
    }
    return aclrtMalloc(reinterpret_cast<void**>(&device.status), layout.statusBytes, ACL_MEM_MALLOC_HUGE_FIRST) ==
           ACL_SUCCESS;
}

void AllocateWindowBuffers(TestContext& ctx, int rankId, const TransferLayout& layout, DeviceResources& device)
{
    const uint64_t localWinBase = ctx.hostCtx.windowsIn[rankId];
    size_t winOffset = 0;
    WindowAlloc(localWinBase, winOffset, kWindowSyncPrefix);
    device.localSend = static_cast<int32_t*>(WindowAlloc(localWinBase, winOffset, layout.dataBytes));
    device.localRecv = static_cast<int32_t*>(WindowAlloc(localWinBase, winOffset, layout.dataBytes));
    device.localSignal = static_cast<int32_t*>(WindowAlloc(localWinBase, winOffset, sizeof(int32_t)));
}

bool CopyInitialData(
    TestContext& ctx, const TransferLayout& layout, const HostBuffers& host, const DeviceResources& device)
{
    int32_t signalValue = 0;
    ctx.aclStatus |= aclrtMemcpy(
        device.localSend, layout.dataBytes, host.source.data(), layout.dataBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    ctx.aclStatus |=
        aclrtMemcpy(device.localRecv, layout.dataBytes, host.recv.data(), layout.dataBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    ctx.aclStatus |= aclrtMemcpy(
        device.consume, layout.dataBytes, host.consumed.data(), layout.dataBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    ctx.aclStatus |= aclrtMemset(device.status, layout.statusBytes, 0, layout.statusBytes);
    ctx.aclStatus |= aclrtMemcpy(
        device.localSignal, sizeof(signalValue), &signalValue, sizeof(signalValue), ACL_MEMCPY_HOST_TO_DEVICE);
    return ctx.aclStatus == ACL_SUCCESS;
}

bool SetupDeviceResources(
    TestContext& ctx, int rankId, const TransferLayout& layout, const HostBuffers& host, DeviceResources& device)
{
    if (layout.requiredWindowBytes > ctx.hostCtx.winSize) {
        std::cerr << "[ERROR] required HCCL window bytes " << layout.requiredWindowBytes << " exceed available "
                  << ctx.hostCtx.winSize << std::endl;
        return false;
    }
    AllocateWindowBuffers(ctx, rankId, layout, device);
    if (!AllocateDeviceMemory(layout, device) || !CopyInitialData(ctx, layout, host, device)) {
        return false;
    }
    device.sdmaInitialized = device.sdmaManager.Init();
    return device.sdmaInitialized;
}

void CleanupPartialSetup(TestContext& ctx, DeviceResources& device)
{
    if (device.consume != nullptr) {
        (void)aclrtFree(device.consume);
    }
    if (device.status != nullptr) {
        (void)aclrtFree(device.status);
    }
    if (device.sdmaInitialized) {
        device.sdmaManager.Finalize();
    }
    (void)ctx.Finalize();
}

void CleanupDeviceResources(TestContext& ctx, DeviceResources& device)
{
    ctx.aclStatus |= aclrtFree(device.consume);
    ctx.aclStatus |= aclrtFree(device.status);
    device.sdmaManager.Finalize();
}

bool ValidateRootStatus(
    TestContext& ctx, AsyncTransferKind transferKind, const TransferLayout& layout, HostBuffers& host,
    const DeviceResources& device)
{
    ctx.aclStatus |= aclrtMemcpy(
        host.status.data(), layout.statusBytes, device.status, layout.statusBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    for (uint32_t transferIndex = 0; transferIndex < layout.totalTransfers; ++transferIndex) {
        if (host.status[transferIndex] == kExpectedPostStatus) {
            continue;
        }
        std::cerr << "[FAIL] " << TransferName(transferKind) << " transfer=" << transferIndex << " status=0x"
                  << std::hex << host.status[transferIndex] << std::dec << " expected=0x" << std::hex
                  << kExpectedPostStatus << std::dec << std::endl;
        return false;
    }
    return true;
}

ConsumerRange GetConsumerRange(int rankId, AsyncTransferKind transferKind, const TransferLayout& layout)
{
    if (transferKind == AsyncTransferKind::TGet) {
        return {0, layout.totalTransfers};
    }
    return {static_cast<uint32_t>(rankId - 1) * layout.postsPerPeer, layout.postsPerPeer};
}

bool ValidateConsumerStatus(
    TestContext& ctx, const TransferLayout& layout, const ConsumerRange& range, HostBuffers& host,
    const DeviceResources& device)
{
    ctx.aclStatus |= aclrtMemcpy(
        host.status.data(), layout.statusBytes, device.status, layout.statusBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    for (uint32_t sequence = 0; sequence < range.transferCount; ++sequence) {
        const uint32_t transferIndex = range.firstTransfer + sequence;
        if (host.status[transferIndex] == kStatusConsumed) {
            continue;
        }
        std::cerr << "[FAIL] TPUT_ASYNC consumer transfer=" << transferIndex << " status=0x" << std::hex
                  << host.status[transferIndex] << std::dec << " expected=0x" << std::hex << kStatusConsumed << std::dec
                  << std::endl;
        return false;
    }
    return true;
}

void CopyConsumerPayload(
    TestContext& ctx, const TransferLayout& layout, HostBuffers& host, const DeviceResources& device)
{
    ctx.aclStatus |=
        aclrtMemcpy(host.recv.data(), layout.dataBytes, device.localRecv, layout.dataBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    ctx.aclStatus |= aclrtMemcpy(
        host.consumed.data(), layout.dataBytes, device.consume, layout.dataBytes, ACL_MEMCPY_DEVICE_TO_HOST);
}

bool ValidateConsumerPayload(
    AsyncTransferKind transferKind, const TransferLayout& layout, const ConsumerRange& range, const HostBuffers& host)
{
    size_t mismatchCount = 0;
    const size_t firstElement = static_cast<size_t>(range.firstTransfer) * kElemsPerPost;
    const size_t checkedElements = static_cast<size_t>(range.transferCount) * kElemsPerPost;
    for (size_t index = firstElement; index < firstElement + checkedElements; ++index) {
        const uint32_t transferIndex = static_cast<uint32_t>(index / kElemsPerPost);
        const uint32_t peerIndex = transferIndex / layout.postsPerPeer;
        const int sourceRank = transferKind == AsyncTransferKind::TGet ? static_cast<int>(peerIndex + 1) : kRootRank;
        const int32_t expectedRaw = SourceValue(sourceRank, index);
        const int32_t expectedConsumed = expectedRaw + kConsumeAdd;
        if (host.recv[index] == expectedRaw && host.consumed[index] == expectedConsumed) {
            continue;
        }
        if (mismatchCount < 8) {
            std::cerr << "[FAIL] " << TransferName(transferKind) << " index=" << index << " raw=" << host.recv[index]
                      << " expected_raw=" << expectedRaw << " consumed=" << host.consumed[index]
                      << " expected_consumed=" << expectedConsumed << std::endl;
        }
        ++mismatchCount;
    }
    if (mismatchCount == 0) {
        return true;
    }
    std::cerr << "[FAIL] " << TransferName(transferKind) << " mismatches=" << mismatchCount << "/" << checkedElements
              << std::endl;
    return false;
}

bool ValidateResults(
    TestContext& ctx, int rankId, AsyncTransferKind transferKind, const TransferLayout& layout, HostBuffers& host,
    const DeviceResources& device)
{
    const bool isConsumer = transferKind == AsyncTransferKind::TGet ? rankId == kRootRank : rankId != kRootRank;
    bool isOk = ctx.aclStatus == 0;
    if (rankId == kRootRank) {
        isOk = ValidateRootStatus(ctx, transferKind, layout, host, device) && isOk;
    }
    if (!isConsumer) {
        return isOk;
    }
    const ConsumerRange range = GetConsumerRange(rankId, transferKind, layout);
    if (transferKind == AsyncTransferKind::TPut) {
        isOk = ValidateConsumerStatus(ctx, layout, range, host, device) && isOk;
    }
    CopyConsumerPayload(ctx, layout, host, device);
    return ValidateConsumerPayload(transferKind, layout, range, host) && isOk;
}

void PrintPass(
    int rankId, AsyncTransferKind transferKind, AsyncCheckMode checkMode, uint32_t postCount, uint32_t rounds,
    uint32_t queueNum, const TransferLayout& layout)
{
    std::cout << "[PASS] " << TransferName(transferKind) << " mode=" << ModeName(checkMode)
              << " transfers=" << layout.totalTransfers << " posts_per_peer=" << layout.postsPerPeer
              << " peer_count=" << layout.peerCount << " posts_per_round=" << postCount << " rounds=" << rounds
              << " queue_num=" << queueNum << " rank=" << rankId << std::endl;
}

bool RunAsyncPostStabilityKernel(
    int rankId, int nRanks, int nDevices, int firstDeviceId, const HcclRootInfo* rootInfo,
    AsyncTransferKind transferKind, AsyncCheckMode checkMode, uint32_t postCount, uint32_t rounds, uint32_t queueNum)
{
    TestContext ctx;
    if (!ctx.Init(rankId, nRanks, nDevices, firstDeviceId, rootInfo)) {
        return false;
    }
    const TransferLayout layout = MakeTransferLayout(nRanks, postCount, rounds);
    HostBuffers host = MakeHostBuffers(layout, rankId);
    DeviceResources device;
    const bool setupOk = SetupDeviceResources(ctx, rankId, layout, host, device);
    if (!AllRanksReady(setupOk, nRanks)) {
        CleanupPartialSetup(ctx, device);
        return false;
    }
    HcclHostBarrier(ctx.comm, ctx.stream);
    AsyncPostStabilityKernel<<<1, nullptr, ctx.stream>>>(
        device.localSend, device.localRecv, device.localSignal, device.consume, device.status, ctx.deviceCtx,
        static_cast<uint8_t*>(device.sdmaManager.GetWorkspaceAddr()), transferKind, checkMode, postCount, rounds,
        queueNum);
    ctx.aclStatus |= aclrtSynchronizeStream(ctx.stream);
    HcclHostBarrier(ctx.comm, ctx.stream);
    const bool isConsumer = transferKind == AsyncTransferKind::TGet ? rankId == kRootRank : rankId != kRootRank;
    const bool isOk = ValidateResults(ctx, rankId, transferKind, layout, host, device);
    if (isOk && (rankId == kRootRank || isConsumer)) {
        PrintPass(rankId, transferKind, checkMode, postCount, rounds, queueNum, layout);
    }
    CleanupDeviceResources(ctx, device);
    return ctx.Finalize() && isOk && ctx.aclStatus == 0;
}

} // namespace

bool IsAsyncPostStabilityDeviceRangeAvailable(int nRanks, int firstDeviceId)
{
    return nRanks > 0 && firstDeviceId >= 0 && GetAvailableDeviceCount() >= nRanks + firstDeviceId;
}

bool RunAsyncPostStability(
    int nRanks, int nDevices, int firstRankId, int firstDeviceId, AsyncTransferKind transferKind,
    AsyncCheckMode checkMode, uint32_t postCount, uint32_t rounds, uint32_t queueNum)
{
    if (nRanks < 2 || nRanks > 3 || postCount == 0 || rounds == 0 || postCount > UINT32_MAX / rounds || queueNum == 0 ||
        kElemsPerPost % queueNum != 0) {
        return false;
    }
    const uint32_t postsPerPeer = postCount * rounds;
    const uint32_t peerCount = static_cast<uint32_t>(nRanks - 1);
    if ((checkMode == AsyncCheckMode::LastWaitOnly &&
         (nRanks != 2 || postCount != 1U || rounds != 1U || queueNum != 4U)) ||
        postsPerPeer > UINT32_MAX / peerCount ||
        (checkMode == AsyncCheckMode::Deferred && (rounds != 1 || postCount > kMaxDeferredEvents / peerCount))) {
        return false;
    }
    return ForkAndRunWithHcclRootInfo(
        nRanks, firstRankId, firstDeviceId, [&](int rankId, const HcclRootInfo* rootInfo) {
            return RunAsyncPostStabilityKernel(
                rankId, nRanks, nDevices, firstDeviceId, rootInfo, transferKind, checkMode, postCount, rounds,
                queueNum);
        });
}
