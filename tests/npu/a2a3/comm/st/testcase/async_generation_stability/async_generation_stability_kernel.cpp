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
#include "async_generation_stability_kernel.h"
#include "../common.hpp"

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

AICORE inline pto::comm::AsyncEvent PostOne(
    AsyncTransferKind transferKind, __gm__ int32_t* localSend, __gm__ int32_t* localRecv,
    __gm__ CommDeviceContext* hcclCtx, int targetRank, uint32_t transferIndex, const pto::comm::AsyncSession& session)
{
    ShapeDyn shape(1, 1, 1, 1, kElemsPerPost);
    StrideDyn stride(kElemsPerPost, kElemsPerPost, kElemsPerPost, kElemsPerPost, 1);
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

__global__ AICORE void AsyncGenerationStabilityKernel(
    __gm__ int32_t* localSend, __gm__ int32_t* localRecv, __gm__ int32_t* localSignal, __gm__ int32_t* consumeOutput,
    __gm__ uint32_t* postStatus, __gm__ CommDeviceContext* hcclCtx, __gm__ uint8_t* sdmaWorkspace,
    AsyncTransferKind transferKind, AsyncCheckMode checkMode, uint32_t postCount, uint32_t rounds, uint32_t queueNum)
{
    const int rank = static_cast<int>(hcclCtx->rankId);
    const uint32_t postsPerPeer = postCount * rounds;

    if (transferKind == AsyncTransferKind::TPut && rank != kRootRank) {
        pto::comm::Signal signal(localSignal);
        const uint32_t peerIndex = static_cast<uint32_t>(rank - 1);
        for (uint32_t sequence = 0; sequence < postsPerPeer; ++sequence) {
            bool signaled = false;
            for (uint32_t poll = 0; poll < kSignalPollLimit; ++poll) {
                if (pto::comm::TTEST(signal, static_cast<int32_t>(sequence + 1), pto::comm::WaitCmp::GE)) {
                    signaled = true;
                    break;
                }
            }
            if (!signaled) {
                const uint32_t transferIndex = peerIndex * postsPerPeer + sequence;
                postStatus[transferIndex] = 0;
                pipe_barrier(PIPE_ALL);
                return;
            }
            dcci(static_cast<__gm__ void*>(0), ENTIRE_DATA_CACHE);
            dsb(DSB_DDR);
            const uint32_t transferIndex = peerIndex * postsPerPeer + sequence;
            const size_t offset = static_cast<size_t>(transferIndex) * kElemsPerPost;
            ConsumePost(localRecv + offset, consumeOutput + offset);
            postStatus[transferIndex] = kStatusConsumed;
        }
        pipe_barrier(PIPE_ALL);
        return;
    }

    if (rank != kRootRank) {
        pipe_barrier(PIPE_ALL);
        return;
    }

    ScratchTile scratchTile;
    TASSIGN(scratchTile, 0x0);
    pto::comm::AsyncSession session;
    const uint64_t bytesPerPost = static_cast<uint64_t>(kElemsPerPost) * sizeof(int32_t);
    pto::comm::sdma::SdmaBaseConfig config{bytesPerPost / queueNum, 0, queueNum};
    if (!pto::comm::BuildAsyncSession(scratchTile, sdmaWorkspace, session, 0, config)) {
        if (transferKind == AsyncTransferKind::TPut) {
            for (int targetRank = 1; targetRank < static_cast<int>(hcclCtx->rankNum); ++targetRank) {
                for (uint32_t sequence = 0; sequence < postsPerPeer; ++sequence) {
                    NotifyTPutConsumer(localSignal, hcclCtx, targetRank, sequence);
                }
            }
        }
        pipe_barrier(PIPE_ALL);
        return;
    }

    if (checkMode == AsyncCheckMode::Deferred) {
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
    } else {
        for (uint32_t sequence = 0; sequence < postsPerPeer; ++sequence) {
            for (int targetRank = 1; targetRank < static_cast<int>(hcclCtx->rankNum); ++targetRank) {
                const uint32_t peerIndex = static_cast<uint32_t>(targetRank - 1);
                const uint32_t transferIndex = peerIndex * postsPerPeer + sequence;
                pto::comm::AsyncEvent event =
                    PostOne(transferKind, localSend, localRecv, hcclCtx, targetRank, transferIndex, session);
                RecordAndConsume(
                    transferKind, event, session, localRecv, consumeOutput, postStatus, localSignal, hcclCtx,
                    targetRank, sequence, transferIndex);
            }
        }
    }
    pipe_barrier(PIPE_ALL);
}

int32_t SourceValue(int rank, size_t index) { return static_cast<int32_t>(rank * 1000000 + index + kSourceBias); }

const char* TransferName(AsyncTransferKind transferKind)
{
    return transferKind == AsyncTransferKind::TGet ? "TGET_ASYNC" : "TPUT_ASYNC";
}

const char* ModeName(AsyncCheckMode checkMode)
{
    return checkMode == AsyncCheckMode::Deferred ? "deferred" : "immediate";
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

bool RunAsyncGenerationStabilityKernel(
    int rankId, int nRanks, int nDevices, int firstDeviceId, const HcclRootInfo* rootInfo,
    AsyncTransferKind transferKind, AsyncCheckMode checkMode, uint32_t postCount, uint32_t rounds, uint32_t queueNum)
{
    TestContext ctx;
    if (!ctx.Init(rankId, nRanks, nDevices, firstDeviceId, rootInfo)) {
        return false;
    }

    const uint32_t postsPerPeer = postCount * rounds;
    const uint32_t peerCount = static_cast<uint32_t>(nRanks - 1);
    const uint32_t totalTransfers = postsPerPeer * peerCount;
    const size_t totalElems = static_cast<size_t>(totalTransfers) * kElemsPerPost;
    const size_t dataBytes = totalElems * sizeof(int32_t);
    const size_t statusBytes = static_cast<size_t>(totalTransfers) * sizeof(uint32_t);

    std::vector<int32_t> sourceHost(totalElems);
    std::vector<int32_t> recvHost(totalElems, kRecvPoison);
    std::vector<int32_t> consumeHost(totalElems, kConsumePoison);
    std::vector<uint32_t> statusHost(totalTransfers, 0);
    for (size_t index = 0; index < totalElems; ++index) {
        sourceHost[index] = SourceValue(rankId, index);
    }

    const size_t requiredWindowBytes = kWindowSyncPrefix + 2 * dataBytes + sizeof(int32_t);
    int32_t* localSend = nullptr;
    int32_t* localRecv = nullptr;
    int32_t* localSignal = nullptr;
    int32_t* consumeDevice = nullptr;
    uint32_t* statusDevice = nullptr;
    SdmaWorkspaceManager sdmaManager;
    bool sdmaInitialized = false;
    bool setupOk = requiredWindowBytes <= ctx.hostCtx.winSize;
    if (!setupOk) {
        std::cerr << "[ERROR] required HCCL window bytes " << requiredWindowBytes << " exceed available "
                  << ctx.hostCtx.winSize << std::endl;
    } else {
        uint64_t localWinBase = ctx.hostCtx.windowsIn[rankId];
        size_t winOffset = 0;
        WindowAlloc(localWinBase, winOffset, kWindowSyncPrefix);
        localSend = static_cast<int32_t*>(WindowAlloc(localWinBase, winOffset, dataBytes));
        localRecv = static_cast<int32_t*>(WindowAlloc(localWinBase, winOffset, dataBytes));
        localSignal = static_cast<int32_t*>(WindowAlloc(localWinBase, winOffset, sizeof(int32_t)));

        setupOk =
            aclrtMalloc(reinterpret_cast<void**>(&consumeDevice), dataBytes, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS;
        if (setupOk) {
            setupOk = aclrtMalloc(reinterpret_cast<void**>(&statusDevice), statusBytes, ACL_MEM_MALLOC_HUGE_FIRST) ==
                      ACL_SUCCESS;
        }
        if (setupOk) {
            int32_t signalValue = 0;
            ctx.aclStatus |= aclrtMemcpy(localSend, dataBytes, sourceHost.data(), dataBytes, ACL_MEMCPY_HOST_TO_DEVICE);
            ctx.aclStatus |= aclrtMemcpy(localRecv, dataBytes, recvHost.data(), dataBytes, ACL_MEMCPY_HOST_TO_DEVICE);
            ctx.aclStatus |=
                aclrtMemcpy(consumeDevice, dataBytes, consumeHost.data(), dataBytes, ACL_MEMCPY_HOST_TO_DEVICE);
            ctx.aclStatus |= aclrtMemset(statusDevice, statusBytes, 0, statusBytes);
            ctx.aclStatus |=
                aclrtMemcpy(localSignal, sizeof(int32_t), &signalValue, sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE);
            setupOk = ctx.aclStatus == ACL_SUCCESS;
        }
        if (setupOk) {
            sdmaInitialized = sdmaManager.Init();
            setupOk = sdmaInitialized;
        }
    }

    if (!AllRanksReady(setupOk, nRanks)) {
        if (consumeDevice != nullptr) {
            (void)aclrtFree(consumeDevice);
        }
        if (statusDevice != nullptr) {
            (void)aclrtFree(statusDevice);
        }
        if (sdmaInitialized) {
            sdmaManager.Finalize();
        }
        (void)ctx.Finalize();
        return false;
    }

    HcclHostBarrier(ctx.comm, ctx.stream);
    AsyncGenerationStabilityKernel<<<1, nullptr, ctx.stream>>>(
        localSend, localRecv, localSignal, consumeDevice, statusDevice, ctx.deviceCtx,
        static_cast<uint8_t*>(sdmaManager.GetWorkspaceAddr()), transferKind, checkMode, postCount, rounds, queueNum);
    ctx.aclStatus |= aclrtSynchronizeStream(ctx.stream);
    HcclHostBarrier(ctx.comm, ctx.stream);

    const bool isConsumer = transferKind == AsyncTransferKind::TGet ? rankId == kRootRank : rankId != kRootRank;
    bool isOk = ctx.aclStatus == 0;
    if (rankId == kRootRank) {
        ctx.aclStatus |=
            aclrtMemcpy(statusHost.data(), statusBytes, statusDevice, statusBytes, ACL_MEMCPY_DEVICE_TO_HOST);
        for (uint32_t transferIndex = 0; transferIndex < totalTransfers; ++transferIndex) {
            if (statusHost[transferIndex] != kExpectedPostStatus) {
                std::cerr << "[FAIL] " << TransferName(transferKind) << " transfer=" << transferIndex << " status=0x"
                          << std::hex << statusHost[transferIndex] << std::dec << " expected=0x" << std::hex
                          << kExpectedPostStatus << std::dec << std::endl;
                isOk = false;
                break;
            }
        }
    }

    if (isConsumer) {
        const uint32_t firstTransfer =
            transferKind == AsyncTransferKind::TGet ? 0 : static_cast<uint32_t>(rankId - 1) * postsPerPeer;
        const uint32_t checkedTransfers = transferKind == AsyncTransferKind::TGet ? totalTransfers : postsPerPeer;
        if (transferKind == AsyncTransferKind::TPut) {
            ctx.aclStatus |=
                aclrtMemcpy(statusHost.data(), statusBytes, statusDevice, statusBytes, ACL_MEMCPY_DEVICE_TO_HOST);
            for (uint32_t sequence = 0; sequence < postsPerPeer; ++sequence) {
                const uint32_t transferIndex = firstTransfer + sequence;
                if (statusHost[transferIndex] != kStatusConsumed) {
                    std::cerr << "[FAIL] TPUT_ASYNC consumer transfer=" << transferIndex << " status=0x" << std::hex
                              << statusHost[transferIndex] << std::dec << " expected=0x" << std::hex << kStatusConsumed
                              << std::dec << std::endl;
                    isOk = false;
                    break;
                }
            }
        }
        ctx.aclStatus |= aclrtMemcpy(recvHost.data(), dataBytes, localRecv, dataBytes, ACL_MEMCPY_DEVICE_TO_HOST);
        ctx.aclStatus |=
            aclrtMemcpy(consumeHost.data(), dataBytes, consumeDevice, dataBytes, ACL_MEMCPY_DEVICE_TO_HOST);
        size_t mismatchCount = 0;
        const size_t firstElement = static_cast<size_t>(firstTransfer) * kElemsPerPost;
        const size_t checkedElements = static_cast<size_t>(checkedTransfers) * kElemsPerPost;
        for (size_t index = firstElement; index < firstElement + checkedElements; ++index) {
            const uint32_t transferIndex = static_cast<uint32_t>(index / kElemsPerPost);
            const uint32_t peerIndex = transferIndex / postsPerPeer;
            const int sourceRank =
                transferKind == AsyncTransferKind::TGet ? static_cast<int>(peerIndex + 1) : kRootRank;
            const int32_t expectedRaw = SourceValue(sourceRank, index);
            const int32_t expectedConsumed = expectedRaw + kConsumeAdd;
            if (recvHost[index] != expectedRaw || consumeHost[index] != expectedConsumed) {
                if (mismatchCount < 8) {
                    std::cerr << "[FAIL] " << TransferName(transferKind) << " index=" << index
                              << " raw=" << recvHost[index] << " expected_raw=" << expectedRaw
                              << " consumed=" << consumeHost[index] << " expected_consumed=" << expectedConsumed
                              << std::endl;
                }
                ++mismatchCount;
            }
        }
        if (mismatchCount != 0) {
            std::cerr << "[FAIL] " << TransferName(transferKind) << " mismatches=" << mismatchCount << "/"
                      << checkedElements << std::endl;
            isOk = false;
        }
    }

    if (isOk && (rankId == kRootRank || isConsumer)) {
        std::cout << "[PASS] " << TransferName(transferKind) << " mode=" << ModeName(checkMode)
                  << " transfers=" << totalTransfers << " posts_per_peer=" << postsPerPeer
                  << " peer_count=" << peerCount << " posts_per_round=" << postCount << " rounds=" << rounds
                  << " queue_num=" << queueNum << " rank=" << rankId << std::endl;
    }

    ctx.aclStatus |= aclrtFree(consumeDevice);
    ctx.aclStatus |= aclrtFree(statusDevice);
    sdmaManager.Finalize();
    return ctx.Finalize() && isOk && ctx.aclStatus == 0;
}

} // namespace

bool IsAsyncGenerationStabilityDeviceRangeAvailable(int nRanks, int firstDeviceId)
{
    return nRanks > 0 && firstDeviceId >= 0 && GetAvailableDeviceCount() >= nRanks + firstDeviceId;
}

bool RunAsyncGenerationStability(
    int nRanks, int nDevices, int firstRankId, int firstDeviceId, AsyncTransferKind transferKind,
    AsyncCheckMode checkMode, uint32_t postCount, uint32_t rounds, uint32_t queueNum)
{
    if (nRanks < 2 || nRanks > 3 || postCount == 0 || rounds == 0 || postCount > UINT32_MAX / rounds || queueNum == 0 ||
        kElemsPerPost % queueNum != 0) {
        return false;
    }
    const uint32_t postsPerPeer = postCount * rounds;
    const uint32_t peerCount = static_cast<uint32_t>(nRanks - 1);
    if (postsPerPeer > UINT32_MAX / peerCount ||
        (checkMode == AsyncCheckMode::Deferred && (rounds != 1 || postCount > kMaxDeferredEvents / peerCount))) {
        return false;
    }
    return ForkAndRunWithHcclRootInfo(
        nRanks, firstRankId, firstDeviceId, [&](int rankId, const HcclRootInfo* rootInfo) {
            return RunAsyncGenerationStabilityKernel(
                rankId, nRanks, nDevices, firstDeviceId, rootInfo, transferKind, checkMode, postCount, rounds,
                queueNum);
        });
}
