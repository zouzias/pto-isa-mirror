/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <cstdint>
#include <iostream>
#include <vector>

#include <pto/pto-inst.hpp>
#include "pto/common/pto_tile.hpp"
#include "../common.hpp"
#include "tput_async_urma_batch_kernel.h"

namespace {

constexpr uint32_t kBatchOperationCount = 4U;
constexpr uint32_t kStressBatchRounds = 65U;
constexpr uint32_t kElementsPerOperation = 256U;
constexpr uint32_t kElementsPerBatch = kBatchOperationCount * kElementsPerOperation;
constexpr uint32_t kRequestedSqDepth = 64U;
constexpr uint32_t kSelectedJettyIndex = 1U;
constexpr uint32_t kSelectedJettiesPerCore = 2U;
constexpr int32_t kConsumeAdd = 100;
constexpr int32_t kPoison = -777777;
constexpr size_t kDataOffset = 64U * sizeof(int32_t);

using BatchShape = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
using BatchStride = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
using BatchGlobal = pto::GlobalTensor<int32_t, BatchShape, BatchStride, pto::Layout::ND>;
using ConsumeTile = pto::Tile<pto::TileType::Vec, int32_t, 1, kElementsPerOperation>;

bool AllRanksReady(bool localReady, int rankId, int nRanks)
{
    bool allReady = true;
    for (int root = 0; root < nRanks; ++root) {
        char ready = rankId == root && localReady ? 1 : 0;
        CommMpiBcast(&ready, 1, COMM_MPI_CHAR, root);
        allReady = allReady && ready != 0;
    }
    return allReady;
}

__global__ AICORE void BatchPutUrma(
    __gm__ int32_t* remoteDst, __gm__ int32_t* localSrc, __gm__ uint8_t* urmaWorkspace, __gm__ uint32_t* status,
    uint32_t peer)
{
#ifdef PTO_URMA_SUPPORTED
    constexpr uint32_t kElementsPerWrite = 16U;
    using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using GT = pto::GlobalTensor<int32_t, ShapeDyn, StrideDyn, pto::Layout::ND>;

    status[0] = 0U;
    ShapeDyn shape(1, 1, 1, 1, kElementsPerWrite);
    StrideDyn stride(kElementsPerWrite, kElementsPerWrite, kElementsPerWrite, kElementsPerWrite, 1);
    GT dst0(remoteDst, shape, stride);
    GT src0(localSrc, shape, stride);
    GT dst1(remoteDst + kElementsPerWrite, shape, stride);
    GT src1(localSrc + kElementsPerWrite, shape, stride);

    pto::comm::AsyncSession session;
    if (!pto::comm::BuildAsyncSession<pto::comm::DmaEngine::URMA>(urmaWorkspace, session)) {
        return;
    }

    constexpr uint32_t kJettyIndex = 0U;
    pto::comm::TPUT_ASYNC_DEFER<pto::comm::DmaEngine::URMA>(dst0, src0, session, peer, kJettyIndex);
    pto::comm::TPUT_ASYNC_DEFER<pto::comm::DmaEngine::URMA>(dst1, src1, session, peer, kJettyIndex);

    pto::comm::AsyncEvent event = pto::comm::TPUT_ASYNC_SUBMIT<pto::comm::DmaEngine::URMA>(session, peer, kJettyIndex);
    if (!event.valid()) {
        return;
    }
    if (!event.Wait(session)) {
        return;
    }
    status[0] = 1U;
#else
    (void)remoteDst;
    (void)localSrc;
    (void)urmaWorkspace;
    (void)status;
    (void)peer;
#endif
}

AICORE inline void PreloadUrmaBatchPayload(__gm__ int32_t* recv, uint32_t batchRounds)
{
    BatchShape shape(1, 1, 1, 1, kElementsPerOperation);
    BatchStride stride(kElementsPerOperation, kElementsPerOperation, kElementsPerOperation, kElementsPerOperation, 1);
    ConsumeTile inputTile;
    TASSIGN(inputTile, 0x1000);
    for (uint32_t round = 0U; round < batchRounds; ++round) {
        for (uint32_t operation = 0U; operation < kBatchOperationCount; ++operation) {
            const uint32_t offset = (round * kBatchOperationCount + operation) * kElementsPerOperation;
            BatchGlobal recvGlobal(recv + offset, shape, stride);
            TLOAD(inputTile, recvGlobal);
            pipe_barrier(PIPE_ALL);
        }
    }
}

AICORE inline void ConsumeUrmaBatchPayload(__gm__ int32_t* recv, __gm__ int32_t* consumed, uint32_t batchRounds)
{
    BatchShape shape(1, 1, 1, 1, kElementsPerOperation);
    BatchStride stride(kElementsPerOperation, kElementsPerOperation, kElementsPerOperation, kElementsPerOperation, 1);
    ConsumeTile inputTile;
    ConsumeTile outputTile;
    TASSIGN(inputTile, 0x1000);
    TASSIGN(outputTile, 0x2000);
    for (uint32_t round = 0U; round < batchRounds; ++round) {
        for (uint32_t operation = 0U; operation < kBatchOperationCount; ++operation) {
            const uint32_t offset = (round * kBatchOperationCount + operation) * kElementsPerOperation;
            BatchGlobal recvGlobal(recv + offset, shape, stride);
            BatchGlobal consumedGlobal(consumed + offset, shape, stride);
            TLOAD(inputTile, recvGlobal);
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            TADDS(outputTile, inputTile, kConsumeAdd);
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            TSTORE(consumedGlobal, outputTile);
            pipe_barrier(PIPE_ALL);
        }
    }
}

__global__ AICORE void TPutAsyncUrmaBatchConsumeKernel(
    __gm__ uint8_t* localBuf, int myRank, int firstRankId, int rootRank, __gm__ uint8_t* urmaWorkspace,
    __gm__ int32_t* notifySignal, __gm__ CommDeviceContext* notifyCtx, __gm__ int32_t* consumed,
    __gm__ uint32_t* status, uint32_t batchRounds, bool requireWqWrap, bool sharedPool, bool oldNotifyPrefix,
    uint32_t jettyIndex, bool verifyJettySelection)
{
    const uint32_t totalElements = batchRounds * kElementsPerBatch;
    __gm__ int32_t* send = reinterpret_cast<__gm__ int32_t*>(localBuf + kDataOffset);
    __gm__ int32_t* recv = send + totalElements;
    const uint32_t myPeer = static_cast<uint32_t>(myRank - firstRankId);
    if (myRank == rootRank) {
#ifdef PTO_URMA_SUPPORTED
        pto::comm::Signal readySignal(notifySignal);
        pto::comm::TWAIT(readySignal, 1, pto::comm::WaitCmp::GE);

        constexpr uint32_t kTargetPeer = 1U;
        pto::comm::AsyncSession session;
        if (!pto::comm::BuildAsyncSession<pto::comm::DmaEngine::URMA>(urmaWorkspace, session)) {
            status[0] = 0U;
            return;
        }
        if (jettyIndex >= session.qpCount) {
            status[0] = 0U;
            return;
        }
        const uint32_t physicalJetty = session.qpIdxBase + jettyIndex;
        __gm__ pto::comm::urma::UrmaWQCtx* wq =
            pto::comm::urma::detail::GetWqContextAt(session, kTargetPeer, physicalJetty);
        __gm__ pto::comm::urma::UrmaCqCtx* cq =
            pto::comm::urma::detail::GetCqContextAt(session, kTargetPeer, physicalJetty);
        __gm__ pto::comm::urma::UrmaWQCtx* baseWq =
            pto::comm::urma::detail::GetWqContextAt(session, kTargetPeer, session.qpIdxBase);
        const uint32_t initialSelectedHead = ld_dev(reinterpret_cast<__gm__ uint32_t*>(wq->headAddr), 0);
        const uint32_t initialBaseHead = ld_dev(reinterpret_cast<__gm__ uint32_t*>(baseWq->headAddr), 0);
        const bool depthsValid = wq->depth != 0U && cq->depth != 0U && (wq->depth & (wq->depth - 1U)) == 0U &&
                                 (cq->depth & (cq->depth - 1U)) == 0U;
        if (!depthsValid || (requireWqWrap && batchRounds * kBatchOperationCount <= wq->depth)) {
            status[0] = 0xD0000000U | ((wq->depth & 0xFFFFU) << 16U) | (cq->depth & 0xFFFFU);
            return;
        }

        BatchShape shape(1, 1, 1, 1, kElementsPerOperation);
        BatchStride stride(
            kElementsPerOperation, kElementsPerOperation, kElementsPerOperation, kElementsPerOperation, 1);
        const uint64_t peerBase = pto::comm::urma::UrmaPeerMrBaseAddr(urmaWorkspace, kTargetPeer);
        __gm__ int32_t* remoteRecv = reinterpret_cast<__gm__ int32_t*>(peerBase + kDataOffset) + totalElements;
        uint32_t producerStatus = 3U;
        pto::comm::AsyncEvent lastBatchEvent;
        uint32_t prefixTargetBb = 0U;
        uint32_t prefixTargetCqe = 0U;
        if (oldNotifyPrefix) {
            BatchShape prefixShape(1, 1, 1, 1, 1);
            BatchStride prefixStride(1, 1, 1, 1, 1);
            BatchGlobal prefixSrc(send, prefixShape, prefixStride);
            BatchGlobal prefixDst(remoteRecv, prefixShape, prefixStride);
            BatchGlobal prefixSignal(reinterpret_cast<__gm__ int32_t*>(peerBase), prefixShape, prefixStride);
            const uint32_t startBb = ld_dev(reinterpret_cast<__gm__ uint32_t*>(wq->headAddr), 0);
            const uint32_t startCqe = wq->submittedWqeCount;
            pto::comm::AsyncEvent prefixEvent = pto::comm::TPUT_ASYNC_NOTIFY<pto::comm::DmaEngine::URMA>(
                prefixDst, prefixSrc, prefixSignal, 1, pto::comm::NotifyOp::AtomicAdd, session, kTargetPeer);
            prefixTargetBb = startBb + 3U;
            prefixTargetCqe = startCqe + 2U;
            if (!prefixEvent.valid() || static_cast<uint32_t>(prefixEvent.handle) != prefixTargetBb ||
                prefixEvent.urmaTargetCqe != prefixTargetCqe) {
                status[0] = 0U;
                return;
            }
        }
        for (uint32_t round = 0U; round < batchRounds; ++round) {
            for (uint32_t operation = 0U; operation < kBatchOperationCount; ++operation) {
                const uint32_t offset = (round * kBatchOperationCount + operation) * kElementsPerOperation;
                BatchGlobal sendGlobal(send + offset, shape, stride);
                BatchGlobal recvGlobal(remoteRecv + offset, shape, stride);
                pto::comm::TPUT_ASYNC_DEFER<pto::comm::DmaEngine::URMA>(
                    recvGlobal, sendGlobal, session, kTargetPeer, jettyIndex);
                if (oldNotifyPrefix && round == 0U && operation == 0U &&
                    (session.urmaRuntimeCtx.batchStartBbProducer != prefixTargetBb ||
                     session.urmaRuntimeCtx.batchStartCqeExpected != prefixTargetCqe ||
                     session.urmaRuntimeCtx.batchJettyIndex != jettyIndex)) {
                    status[0] = 0U;
                    return;
                }
            }
            lastBatchEvent = pto::comm::TPUT_ASYNC_SUBMIT<pto::comm::DmaEngine::URMA>(session, kTargetPeer, jettyIndex);
            if (!lastBatchEvent.valid() || lastBatchEvent.urmaJettyCount != 1U ||
                lastBatchEvent.urmaJettyBase != physicalJetty) {
                producerStatus = 0U;
                break;
            }
            if (oldNotifyPrefix && round == 0U &&
                (static_cast<uint32_t>(lastBatchEvent.handle) != prefixTargetBb + kBatchOperationCount ||
                 lastBatchEvent.urmaTargetBbPerJetty[0] != prefixTargetBb + kBatchOperationCount ||
                 lastBatchEvent.urmaTargetCqePerJetty[0] != prefixTargetCqe + kBatchOperationCount)) {
                producerStatus = 0U;
                break;
            }
        }
        if (producerStatus != 0U && !lastBatchEvent.Wait(session)) {
            producerStatus = 0U;
        }
        if (producerStatus != 0U && verifyJettySelection &&
            (ld_dev(reinterpret_cast<__gm__ uint32_t*>(wq->headAddr), 0) == initialSelectedHead ||
             ld_dev(reinterpret_cast<__gm__ uint32_t*>(baseWq->headAddr), 0) != initialBaseHead)) {
            producerStatus = 0U;
        }

        __gm__ int32_t* remoteSignal = CommRemotePtr(notifyCtx, notifySignal, static_cast<int>(kTargetPeer));
        pto::comm::Signal doneSignal(remoteSignal);
        pto::comm::TNOTIFY(doneSignal, 2, pto::comm::NotifyOp::Set);
        producerStatus |= 4U;
        status[0] = producerStatus;
#endif
        return;
    }

    if (myPeer != 1U) {
        return;
    }
    pto::comm::Signal localSignal(notifySignal);
    PreloadUrmaBatchPayload(recv, batchRounds);
#ifdef PTO_URMA_SUPPORTED
    constexpr uint32_t kRootPeer = 0U;
    __gm__ int32_t* rootSignal = CommRemotePtr(notifyCtx, notifySignal, static_cast<int>(kRootPeer));
    pto::comm::Signal readySignal(rootSignal);
    pto::comm::TNOTIFY(readySignal, 1, pto::comm::NotifyOp::Set);
#endif
    pto::comm::TWAIT(localSignal, 2, pto::comm::WaitCmp::GE);
    dcci(static_cast<__gm__ void*>(0), cache_line_t::ENTIRE_DATA_CACHE);
    dsb(DSB_DDR);
    ConsumeUrmaBatchPayload(recv, consumed, batchRounds);
    status[0] = 8U;
}

bool RunTPutAsyncUrmaBatchDocExampleRank(
    int rankId, int nRanks, int nDevices, int firstDeviceId, int firstRankId, int rootRank)
{
    (void)firstRankId;
    constexpr uint32_t kExampleElements = 32U;
    const size_t dataBytes = kExampleElements * sizeof(int32_t);
    const size_t commBytesNeeded = kDataOffset + 2U * dataBytes;
    UrmaTestContext ctx;
    const bool contextReady =
        ctx.Setup(rankId, nRanks, nDevices, firstDeviceId, rootRank, commBytesNeeded, UrmaLayout::PER_PEER);
    if (!AllRanksReady(contextReady, rankId, nRanks)) {
        ctx.Cleanup();
        return false;
    }

    std::vector<int32_t> source(kExampleElements);
    std::vector<int32_t> recv(kExampleElements, kPoison);
    for (uint32_t index = 0U; index < kExampleElements; ++index) {
        source[index] = static_cast<int32_t>(index + 17U);
    }
    uint8_t* localBuf = reinterpret_cast<uint8_t*>(ctx.devBuf);
    int32_t* deviceSend = reinterpret_cast<int32_t*>(localBuf + kDataOffset);
    int32_t* deviceRecv = deviceSend + kExampleElements;
    uint32_t* deviceStatus = nullptr;
    bool setupOk = aclrtMalloc(reinterpret_cast<void**>(&deviceStatus), sizeof(uint32_t), ACL_MEM_MALLOC_HUGE_FIRST) ==
                   ACL_SUCCESS;
    if (setupOk) {
        setupOk =
            aclrtMemcpy(deviceSend, dataBytes, source.data(), dataBytes, ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS &&
            aclrtMemcpy(deviceRecv, dataBytes, recv.data(), dataBytes, ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS &&
            aclrtMemset(deviceStatus, sizeof(uint32_t), 0, sizeof(uint32_t)) == ACL_SUCCESS;
    }
    if (!AllRanksReady(setupOk, rankId, nRanks)) {
        if (deviceStatus != nullptr) {
            (void)aclrtFree(deviceStatus);
        }
        ctx.Cleanup();
        return false;
    }

    CommMpiBarrier();
    uint32_t statusValue = 0U;
    int syncRet = ACL_SUCCESS;
    if (rankId == rootRank) {
        constexpr uint32_t kTargetPeer = 1U;
        int32_t* remoteDst =
            reinterpret_cast<int32_t*>(ctx.urmaMgr.PeerBaseAddr(kTargetPeer) + kDataOffset + dataBytes);
        BatchPutUrma<<<1, nullptr, ctx.stream>>>(
            remoteDst, deviceSend, reinterpret_cast<uint8_t*>(ctx.urmaMgr.GetWorkspaceAddr()), deviceStatus,
            kTargetPeer);
        syncRet = aclrtSynchronizeStream(ctx.stream);
        (void)aclrtMemcpy(
            &statusValue, sizeof(statusValue), deviceStatus, sizeof(statusValue), ACL_MEMCPY_DEVICE_TO_HOST);
    }
    CommMpiBarrier();

    bool isOk = syncRet == ACL_SUCCESS;
    if (rankId == rootRank) {
        isOk = isOk && statusValue == 1U;
    } else {
        (void)aclrtMemcpy(recv.data(), dataBytes, deviceRecv, dataBytes, ACL_MEMCPY_DEVICE_TO_HOST);
        isOk = isOk && recv == source;
    }
    if (!isOk) {
        std::cerr << "[FAIL] URMA documentation kernel rank=" << rankId << " status=" << statusValue << std::endl;
    }

    (void)aclrtFree(deviceStatus);
    ctx.Cleanup();
    return isOk;
}

bool RunTPutAsyncUrmaBatchConsumeRank(
    int rankId, int nRanks, int nDevices, int firstDeviceId, int firstRankId, int rootRank, bool sharedPool,
    bool oldNotifyPrefix, uint32_t batchRounds, bool requireWqWrap, uint32_t jettyIndex = 0U,
    uint32_t jettiesPerCore = 1U, bool verifyJettySelection = false)
{
    const uint32_t totalElements = batchRounds * kElementsPerBatch;
    const size_t dataBytes = static_cast<size_t>(totalElements) * sizeof(int32_t);
    const size_t commBytesNeeded = kDataOffset + 2U * dataBytes;
    UrmaTestContext ctx;
    const uint32_t aivCount = jettiesPerCore > 1U ? 1U : pto::comm::urma::kUrmaAutoAivCount;
    if (!ctx.Setup(
            rankId, nRanks, nDevices, firstDeviceId, rootRank, commBytesNeeded,
            sharedPool ? UrmaLayout::SHARED_POOL : UrmaLayout::PER_PEER, aivCount, jettiesPerCore, kRequestedSqDepth,
            true)) {
        return false;
    }

    std::vector<int32_t> source(totalElements);
    std::vector<int32_t> recv(totalElements, kPoison);
    std::vector<int32_t> consumed(totalElements, kPoison);
    for (uint32_t index = 0U; index < totalElements; ++index) {
        source[index] = static_cast<int32_t>(index + 7U);
    }
    uint8_t* localBuf = reinterpret_cast<uint8_t*>(ctx.devBuf);
    int32_t* deviceSend = reinterpret_cast<int32_t*>(localBuf + kDataOffset);
    int32_t* deviceRecv = deviceSend + totalElements;
    int32_t* deviceSignal =
        reinterpret_cast<int32_t*>(ctx.notifyHostCtx.windowsIn[ctx.notifyHostCtx.rankId] + kDirectNotifyWindowOffset);
    int32_t* deviceConsumed = nullptr;
    uint32_t* deviceStatus = nullptr;
    bool setupOk =
        aclrtMalloc(reinterpret_cast<void**>(&deviceConsumed), dataBytes, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS &&
        aclrtMalloc(reinterpret_cast<void**>(&deviceStatus), sizeof(uint32_t), ACL_MEM_MALLOC_HUGE_FIRST) ==
            ACL_SUCCESS;
    int32_t signalValue = 0;
    uint32_t statusValue = 0U;
    if (setupOk) {
        setupOk =
            aclrtMemcpy(deviceSend, dataBytes, source.data(), dataBytes, ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS &&
            aclrtMemcpy(deviceRecv, dataBytes, recv.data(), dataBytes, ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS &&
            aclrtMemcpy(deviceConsumed, dataBytes, consumed.data(), dataBytes, ACL_MEMCPY_HOST_TO_DEVICE) ==
                ACL_SUCCESS &&
            aclrtMemset(deviceSignal, sizeof(signalValue), 0, sizeof(signalValue)) == ACL_SUCCESS &&
            aclrtMemset(deviceStatus, sizeof(uint32_t), 0, sizeof(uint32_t)) == ACL_SUCCESS;
    }
    if (!setupOk) {
        if (deviceConsumed != nullptr) {
            (void)aclrtFree(deviceConsumed);
        }
        if (deviceStatus != nullptr) {
            (void)aclrtFree(deviceStatus);
        }
        ctx.Cleanup();
        return false;
    }

    CommMpiBarrier();
    TPutAsyncUrmaBatchConsumeKernel<<<1, nullptr, ctx.stream>>>(
        localBuf, rankId, firstRankId, rootRank, reinterpret_cast<uint8_t*>(ctx.urmaMgr.GetWorkspaceAddr()),
        deviceSignal, ctx.notifyDeviceCtx, deviceConsumed, deviceStatus, batchRounds, requireWqWrap, sharedPool,
        oldNotifyPrefix, jettyIndex, verifyJettySelection);
    const int syncRet = aclrtSynchronizeStream(ctx.stream);
    CommMpiBarrier();

    aclrtMemcpy(&statusValue, sizeof(statusValue), deviceStatus, sizeof(statusValue), ACL_MEMCPY_DEVICE_TO_HOST);
    bool isOk = syncRet == ACL_SUCCESS;
    if (rankId == rootRank) {
        isOk = isOk && statusValue == 7U;
    } else {
        aclrtMemcpy(recv.data(), dataBytes, deviceRecv, dataBytes, ACL_MEMCPY_DEVICE_TO_HOST);
        aclrtMemcpy(consumed.data(), dataBytes, deviceConsumed, dataBytes, ACL_MEMCPY_DEVICE_TO_HOST);
        isOk = isOk && statusValue == 8U;
        for (uint32_t index = 0U; isOk && index < totalElements; ++index) {
            isOk = recv[index] == source[index] && consumed[index] == source[index] + kConsumeAdd;
        }
    }
    if (!isOk) {
        std::cerr << "[FAIL] URMA batch consume rank=" << rankId << " status=" << statusValue
                  << " sharedPool=" << sharedPool << " oldNotifyPrefix=" << oldNotifyPrefix << std::endl;
    }

    (void)aclrtFree(deviceConsumed);
    (void)aclrtFree(deviceStatus);
    ctx.Cleanup();
    return isOk;
}

bool RunTPutAsyncUrmaBatchPerPeerRank(
    int rankId, int nRanks, int nDevices, int firstDeviceId, int firstRankId, int rootRank)
{
    return RunTPutAsyncUrmaBatchConsumeRank(
        rankId, nRanks, nDevices, firstDeviceId, firstRankId, rootRank, false, false, kStressBatchRounds, true);
}

bool RunTPutAsyncUrmaBatchSharedPoolRank(
    int rankId, int nRanks, int nDevices, int firstDeviceId, int firstRankId, int rootRank)
{
    return RunTPutAsyncUrmaBatchConsumeRank(
        rankId, nRanks, nDevices, firstDeviceId, firstRankId, rootRank, true, false, kStressBatchRounds, true);
}

bool RunTPutAsyncUrmaBatchAfterNotifyRank(
    int rankId, int nRanks, int nDevices, int firstDeviceId, int firstRankId, int rootRank)
{
    return RunTPutAsyncUrmaBatchConsumeRank(
        rankId, nRanks, nDevices, firstDeviceId, firstRankId, rootRank, false, true, kStressBatchRounds, true);
}

bool RunTPutAsyncUrmaBatchPerPeerBasicRank(
    int rankId, int nRanks, int nDevices, int firstDeviceId, int firstRankId, int rootRank)
{
    return RunTPutAsyncUrmaBatchConsumeRank(
        rankId, nRanks, nDevices, firstDeviceId, firstRankId, rootRank, false, false, 1U, false);
}

bool RunTPutAsyncUrmaBatchSharedPoolBasicRank(
    int rankId, int nRanks, int nDevices, int firstDeviceId, int firstRankId, int rootRank)
{
    return RunTPutAsyncUrmaBatchConsumeRank(
        rankId, nRanks, nDevices, firstDeviceId, firstRankId, rootRank, true, false, 1U, false);
}

bool RunTPutAsyncUrmaBatchSelectedJettyRank(
    int rankId, int nRanks, int nDevices, int firstDeviceId, int firstRankId, int rootRank)
{
    return RunTPutAsyncUrmaBatchConsumeRank(
        rankId, nRanks, nDevices, firstDeviceId, firstRankId, rootRank, true, false, 1U, false, kSelectedJettyIndex,
        kSelectedJettiesPerCore, true);
}

} // namespace

bool RunTPutAsyncUrmaBatchConsume(
    int nRanks, int nDevices, int firstRankId, int firstDeviceId, bool sharedPool, bool oldNotifyPrefix)
{
    if (oldNotifyPrefix && sharedPool) {
        return false;
    }
    return RunUrmaTestMpiLaunch(
        nRanks, nDevices, firstRankId, firstDeviceId,
        oldNotifyPrefix ? RunTPutAsyncUrmaBatchAfterNotifyRank :
                          (sharedPool ? RunTPutAsyncUrmaBatchSharedPoolRank : RunTPutAsyncUrmaBatchPerPeerRank));
}

bool RunTPutAsyncUrmaBatchBasic(int nRanks, int nDevices, int firstRankId, int firstDeviceId, bool sharedPool)
{
    return RunUrmaTestMpiLaunch(
        nRanks, nDevices, firstRankId, firstDeviceId,
        sharedPool ? RunTPutAsyncUrmaBatchSharedPoolBasicRank : RunTPutAsyncUrmaBatchPerPeerBasicRank);
}

bool RunTPutAsyncUrmaBatchDocExample(int nRanks, int nDevices, int firstRankId, int firstDeviceId)
{
    return RunUrmaTestMpiLaunch(nRanks, nDevices, firstRankId, firstDeviceId, RunTPutAsyncUrmaBatchDocExampleRank);
}

bool RunTPutAsyncUrmaBatchSelectedJetty(int nRanks, int nDevices, int firstRankId, int firstDeviceId)
{
    return RunUrmaTestMpiLaunch(nRanks, nDevices, firstRankId, firstDeviceId, RunTPutAsyncUrmaBatchSelectedJettyRank);
}
