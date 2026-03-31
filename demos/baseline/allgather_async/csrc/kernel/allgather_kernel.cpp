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

#include <pto/pto-inst.hpp>
#include "pto/npu/comm/async/sdma/sdma_types.hpp"
#include "pto/common/pto_tile.hpp"
#include "common.hpp"

// ============================================================================
// Constants
// ============================================================================
static constexpr size_t ELEM_COUNT = 256;
static constexpr size_t SYNC_BUF_BYTES = 64 * sizeof(int32_t);
static constexpr int32_t RANK_BASE = 1000;

using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
using GlobalI32 = pto::GlobalTensor<int32_t, ShapeDyn, StrideDyn, pto::Layout::ND>;
using ScratchTile = pto::Tile<pto::TileType::Vec, uint8_t, 1, pto::comm::sdma::UB_ALIGN_SIZE>;
using LocalTile = pto::Tile<pto::TileType::Vec, int32_t, 1, ELEM_COUNT, pto::BLayout::RowMajor, -1, -1>;

// ============================================================================
// Allgather via TPUT_ASYNC
//
// Every rank writes its sendBuf to every other rank's recvBuf slot.
// Also copies its own data locally to complete the allgather.
//
// Memory layout per rank (in shared window):
//   sendBuf[ELEM_COUNT]               -- this rank's contribution
//   recvBuf[nranks * ELEM_COUNT]      -- gathered result
// ============================================================================
__global__ AICORE void AllgatherPutAsyncKernel(__gm__ int32_t *dataBuf, int nranks,
                                               __gm__ HcclDeviceContext *hcclCtx,
                                               __gm__ uint8_t *sdmaWorkspace, uint32_t sdmaSyncId)
{
    if (nranks < 2) return;

    ShapeDyn shape(1, 1, 1, 1, ELEM_COUNT);
    StrideDyn stride(ELEM_COUNT, ELEM_COUNT, ELEM_COUNT, ELEM_COUNT, 1);

    int myRank = static_cast<int>(hcclCtx->rankId);
    __gm__ int32_t *sendBuf = dataBuf;
    __gm__ int32_t *recvBuf = dataBuf + ELEM_COUNT;

    // Local copy: recvBuf[myRank * ELEM_COUNT] = sendBuf
    GlobalI32 srcG(sendBuf, shape, stride);
    GlobalI32 localSlotG(recvBuf + myRank * ELEM_COUNT, shape, stride);
    LocalTile tile(1, ELEM_COUNT);
    TASSIGN(tile, 0x10000);
    TLOAD(tile, srcG);
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    TSTORE(localSlotG, tile);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);

    // Async remote writes
    GlobalI32 sendG(sendBuf, shape, stride);
    constexpr int kEventSlots = pto::comm::sdma::SDMA_EVENT_SLOT_COUNT;
    pto::comm::AsyncEvent events[kEventSlots];
    ScratchTile scratchTile;
    TASSIGN(scratchTile, 0x0);
    pto::comm::AsyncSession session;
    if (!pto::comm::BuildAsyncSession(scratchTile, sdmaWorkspace, session, sdmaSyncId)) {
        pipe_barrier(PIPE_ALL);
        return;
    }

    int issued = 0;
    for (int target = 0; target < nranks; ++target) {
        if (target == myRank) continue;
        __gm__ int32_t *remoteSlot = HcclRemotePtr(hcclCtx, recvBuf, target) + myRank * ELEM_COUNT;
        GlobalI32 remoteG(remoteSlot, shape, stride);
        if (issued >= kEventSlots) {
            (void)events[issued % kEventSlots].Wait(session);
        }
        events[issued % kEventSlots] = pto::comm::TPUT_ASYNC(remoteG, sendG, session);
        issued++;
    }
    const int pending = (issued < kEventSlots) ? issued : kEventSlots;
    for (int i = 0; i < pending; ++i) {
        (void)events[i].Wait(session);
    }

    pipe_barrier(PIPE_ALL);
}

// ============================================================================
// Allgather via TGET_ASYNC
//
// Every rank pulls every other rank's sendBuf into its local recvBuf slot.
// Also copies its own data locally to complete the allgather.
// ============================================================================
__global__ AICORE void AllgatherGetAsyncKernel(__gm__ int32_t *dataBuf, int nranks,
                                               __gm__ HcclDeviceContext *hcclCtx,
                                               __gm__ uint8_t *sdmaWorkspace, uint32_t sdmaSyncId)
{
    if (nranks < 2) return;

    ShapeDyn shape(1, 1, 1, 1, ELEM_COUNT);
    StrideDyn stride(ELEM_COUNT, ELEM_COUNT, ELEM_COUNT, ELEM_COUNT, 1);

    int myRank = static_cast<int>(hcclCtx->rankId);
    __gm__ int32_t *sendBuf = dataBuf;
    __gm__ int32_t *recvBuf = dataBuf + ELEM_COUNT;

    // Local copy: recvBuf[myRank * ELEM_COUNT] = sendBuf
    GlobalI32 srcG(sendBuf, shape, stride);
    GlobalI32 localSlotG(recvBuf + myRank * ELEM_COUNT, shape, stride);
    LocalTile tile(1, ELEM_COUNT);
    TASSIGN(tile, 0x10000);
    TLOAD(tile, srcG);
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    TSTORE(localSlotG, tile);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);

    // Async remote reads
    constexpr int kEventSlots = pto::comm::sdma::SDMA_EVENT_SLOT_COUNT;
    pto::comm::AsyncEvent events[kEventSlots];
    ScratchTile scratchTile;
    TASSIGN(scratchTile, 0x0);
    pto::comm::AsyncSession session;
    if (!pto::comm::BuildAsyncSession(scratchTile, sdmaWorkspace, session, sdmaSyncId)) {
        pipe_barrier(PIPE_ALL);
        return;
    }

    int issued = 0;
    for (int src = 0; src < nranks; ++src) {
        if (src == myRank) continue;
        __gm__ int32_t *remoteSend = HcclRemotePtr(hcclCtx, sendBuf, src);
        GlobalI32 remoteG(remoteSend, shape, stride);
        GlobalI32 localG(recvBuf + src * ELEM_COUNT, shape, stride);
        if (issued >= kEventSlots) {
            (void)events[issued % kEventSlots].Wait(session);
        }
        events[issued % kEventSlots] = pto::comm::TGET_ASYNC(localG, remoteG, session);
        issued++;
    }
    const int pending = (issued < kEventSlots) ? issued : kEventSlots;
    for (int i = 0; i < pending; ++i) {
        (void)events[i].Wait(session);
    }

    pipe_barrier(PIPE_ALL);
}

// ============================================================================
// Host-side helpers
// ============================================================================
static bool VerifyAllgather(const int32_t *host, int nRanks, size_t elemCount, int rankId, const char *tag)
{
    for (int r = 0; r < nRanks; ++r) {
        for (size_t i = 0; i < elemCount; ++i) {
            int32_t expected = static_cast<int32_t>(r) * RANK_BASE + static_cast<int32_t>(i);
            int32_t actual = host[r * elemCount + i];
            if (actual != expected) {
                std::cerr << "[" << tag << " FAIL] Rank " << rankId << ": recvBuf[" << r << "][" << i << "] = "
                          << actual << ", expected " << expected << std::endl;
                return false;
            }
        }
    }
    return true;
}

static void PrintSample(const int32_t *host, int nRanks, size_t elemCount, int rankId, const char *tag)
{
    std::cout << "[" << tag << " PASS] Rank " << rankId << ": ";
    for (int r = 0; r < nRanks && r < 3; ++r) {
        std::cout << "slot[" << r << "]=[";
        for (size_t i = 0; i < 3 && i < elemCount; ++i)
            std::cout << (i ? "," : "") << host[r * elemCount + i];
        std::cout << ",...] ";
    }
    if (nRanks > 3) std::cout << "...";
    std::cout << std::endl;
}

// ============================================================================
// RunAllgatherPutAsync
// ============================================================================
static bool RunAllgatherPutAsyncKernel(int rankId, int nRanks, int nDevices, int firstDeviceId,
                                       const HcclRootInfo *rootInfo)
{
    TestContext ctx;
    if (!ctx.Init(rankId, nRanks, nDevices, firstDeviceId, rootInfo))
        return false;

    const size_t recvElems = static_cast<size_t>(nRanks) * ELEM_COUNT;

    int32_t *sendHost = nullptr;
    int32_t *recvHost = nullptr;
    if (aclrtMallocHost(reinterpret_cast<void **>(&sendHost), ELEM_COUNT * sizeof(int32_t)) != 0 ||
        aclrtMallocHost(reinterpret_cast<void **>(&recvHost), recvElems * sizeof(int32_t)) != 0) {
        std::cerr << "[ERROR] aclrtMallocHost failed" << std::endl;
        return false;
    }

    for (size_t i = 0; i < ELEM_COUNT; ++i)
        sendHost[i] = static_cast<int32_t>(rankId) * RANK_BASE + static_cast<int32_t>(i);
    for (size_t i = 0; i < recvElems; ++i)
        recvHost[i] = -1;

    uint64_t winBase = ctx.hostCtx.windowsIn[rankId];
    size_t winOff = 0;
    size_t winBytes = SYNC_BUF_BYTES + (ELEM_COUNT + recvElems) * sizeof(int32_t);
    void *commPtr = WindowAlloc(winBase, winOff, winBytes);

    int32_t *dataBuf = reinterpret_cast<int32_t *>(
        reinterpret_cast<uint8_t *>(commPtr) + SYNC_BUF_BYTES);
    int32_t *sendBuf = dataBuf;
    int32_t *recvBuf = dataBuf + ELEM_COUNT;

    aclrtMemcpy(sendBuf, ELEM_COUNT * sizeof(int32_t), sendHost, ELEM_COUNT * sizeof(int32_t),
                ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(recvBuf, recvElems * sizeof(int32_t), recvHost, recvElems * sizeof(int32_t),
                ACL_MEMCPY_HOST_TO_DEVICE);

    SdmaWorkspaceManager sdmaMgr;
    if (!sdmaMgr.Init()) {
        std::cerr << "[ERROR] SdmaWorkspaceManager Init failed" << std::endl;
        return false;
    }

    HcclHostBarrier(ctx.comm, ctx.stream);

    AllgatherPutAsyncKernel<<<1, nullptr, ctx.stream>>>(
        dataBuf, nRanks, ctx.deviceCtx, (uint8_t *)sdmaMgr.GetWorkspaceAddr(), 0);
    ctx.aclStatus = aclrtSynchronizeStream(ctx.stream);

    HcclHostBarrier(ctx.comm, ctx.stream);

    aclrtMemcpy(recvHost, recvElems * sizeof(int32_t), recvBuf, recvElems * sizeof(int32_t),
                ACL_MEMCPY_DEVICE_TO_HOST);

    bool ok = VerifyAllgather(recvHost, nRanks, ELEM_COUNT, rankId, "TPUT_ASYNC");
    if (ok) PrintSample(recvHost, nRanks, ELEM_COUNT, rankId, "TPUT_ASYNC");

    aclrtFreeHost(sendHost);
    aclrtFreeHost(recvHost);
    sdmaMgr.Finalize();

    return ctx.Finalize() && ok;
}

bool RunAllgatherPutAsync(int nRanks, int firstRankId, int firstDeviceId)
{
    return ForkAndRunWithHcclRootInfo(
        nRanks, firstRankId, firstDeviceId, [&](int rankId, const HcclRootInfo *rootInfo) {
            return RunAllgatherPutAsyncKernel(rankId, nRanks, nRanks, firstDeviceId, rootInfo);
        });
}

// ============================================================================
// RunAllgatherGetAsync
// ============================================================================
static bool RunAllgatherGetAsyncKernel(int rankId, int nRanks, int nDevices, int firstDeviceId,
                                       const HcclRootInfo *rootInfo)
{
    TestContext ctx;
    if (!ctx.Init(rankId, nRanks, nDevices, firstDeviceId, rootInfo))
        return false;

    const size_t recvElems = static_cast<size_t>(nRanks) * ELEM_COUNT;

    int32_t *sendHost = nullptr;
    int32_t *recvHost = nullptr;
    if (aclrtMallocHost(reinterpret_cast<void **>(&sendHost), ELEM_COUNT * sizeof(int32_t)) != 0 ||
        aclrtMallocHost(reinterpret_cast<void **>(&recvHost), recvElems * sizeof(int32_t)) != 0) {
        std::cerr << "[ERROR] aclrtMallocHost failed" << std::endl;
        return false;
    }

    for (size_t i = 0; i < ELEM_COUNT; ++i)
        sendHost[i] = static_cast<int32_t>(rankId) * RANK_BASE + static_cast<int32_t>(i);
    for (size_t i = 0; i < recvElems; ++i)
        recvHost[i] = -1;

    uint64_t winBase = ctx.hostCtx.windowsIn[rankId];
    size_t winOff = 0;
    size_t winBytes = SYNC_BUF_BYTES + (ELEM_COUNT + recvElems) * sizeof(int32_t);
    void *commPtr = WindowAlloc(winBase, winOff, winBytes);

    int32_t *dataBuf = reinterpret_cast<int32_t *>(
        reinterpret_cast<uint8_t *>(commPtr) + SYNC_BUF_BYTES);
    int32_t *sendBuf = dataBuf;
    int32_t *recvBuf = dataBuf + ELEM_COUNT;

    aclrtMemcpy(sendBuf, ELEM_COUNT * sizeof(int32_t), sendHost, ELEM_COUNT * sizeof(int32_t),
                ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(recvBuf, recvElems * sizeof(int32_t), recvHost, recvElems * sizeof(int32_t),
                ACL_MEMCPY_HOST_TO_DEVICE);

    SdmaWorkspaceManager sdmaMgr;
    if (!sdmaMgr.Init()) {
        std::cerr << "[ERROR] SdmaWorkspaceManager Init failed" << std::endl;
        return false;
    }

    HcclHostBarrier(ctx.comm, ctx.stream);

    AllgatherGetAsyncKernel<<<1, nullptr, ctx.stream>>>(
        dataBuf, nRanks, ctx.deviceCtx, (uint8_t *)sdmaMgr.GetWorkspaceAddr(), 0);
    ctx.aclStatus = aclrtSynchronizeStream(ctx.stream);

    HcclHostBarrier(ctx.comm, ctx.stream);

    aclrtMemcpy(recvHost, recvElems * sizeof(int32_t), recvBuf, recvElems * sizeof(int32_t),
                ACL_MEMCPY_DEVICE_TO_HOST);

    bool ok = VerifyAllgather(recvHost, nRanks, ELEM_COUNT, rankId, "TGET_ASYNC");
    if (ok) PrintSample(recvHost, nRanks, ELEM_COUNT, rankId, "TGET_ASYNC");

    aclrtFreeHost(sendHost);
    aclrtFreeHost(recvHost);
    sdmaMgr.Finalize();

    return ctx.Finalize() && ok;
}

bool RunAllgatherGetAsync(int nRanks, int firstRankId, int firstDeviceId)
{
    return ForkAndRunWithHcclRootInfo(
        nRanks, firstRankId, firstDeviceId, [&](int rankId, const HcclRootInfo *rootInfo) {
            return RunAllgatherGetAsyncKernel(rankId, nRanks, nRanks, firstDeviceId, rootInfo);
        });
}
