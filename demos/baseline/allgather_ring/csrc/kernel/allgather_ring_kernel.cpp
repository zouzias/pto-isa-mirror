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
#include <unistd.h>

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
// Ring AllGather Round Kernel (TPUT_ASYNC)
//
// Ring algorithm: N-1 rounds for N ranks.
//   - Round 0: rank i copies sendBuf -> recvBuf[i] (local), then pushes
//     sendBuf -> rank (i+1)'s recvBuf[i] via TPUT_ASYNC.
//   - Round r (r >= 1): rank i pushes recvBuf[chunk] -> rank (i+1)'s
//     recvBuf[chunk] via TPUT_ASYNC, where chunk = (i - r + N) % N.
//
// Each round is launched as a separate kernel. Host-side barrier between
// rounds ensures all SDMA writes complete before the next round begins.
//
// NOTE: For N > 2, rounds >= 1 read data that was written by a remote rank's
// SDMA in the previous round. This may cause cache coherency issues on
// current hardware (L2 cache may serve stale data instead of the fresh
// HBM content written by SDMA). This demo serves as a reference
// implementation; the cache coherency limitation is documented here
// for awareness.
// ============================================================================
__global__ AICORE void RingAllgatherRoundKernel(__gm__ int32_t *dataBuf, int nranks,
                                                __gm__ HcclDeviceContext *hcclCtx,
                                                __gm__ uint8_t *sdmaWorkspace, uint32_t sdmaSyncId,
                                                int elemCount, int round)
{
    if (nranks < 2) return;

    int myRank = static_cast<int>(hcclCtx->rankId);
    int nextRank = (myRank + 1) % nranks;

    __gm__ int32_t *sendBuf = dataBuf;
    __gm__ int32_t *recvBuf = dataBuf + elemCount;

    ShapeDyn shape(1, 1, 1, 1, elemCount);
    StrideDyn stride(elemCount, elemCount, elemCount, elemCount, 1);

    // Round 0: copy own sendBuf into recvBuf[myRank]
    if (round == 0) {
        LocalTile tile(1, ELEM_COUNT);
        TASSIGN(tile, 0x10000);
        int chunkSize = static_cast<int>(ELEM_COUNT);
        int numChunks = (elemCount + chunkSize - 1) / chunkSize;
        ShapeDyn cShape(1, 1, 1, 1, chunkSize);
        StrideDyn cStride(chunkSize, chunkSize, chunkSize, chunkSize, 1);
        for (int c = 0; c < numChunks; ++c) {
            int off = c * chunkSize;
            GlobalI32 srcC(sendBuf + off, cShape, cStride);
            GlobalI32 dstC(recvBuf + myRank * elemCount + off, cShape, cStride);
            TLOAD(tile, srcC);
            set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            TSTORE(dstC, tile);
            set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
            wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        }
        pipe_barrier(PIPE_ALL);
    }

    // Determine which chunk to push to the next rank
    int sendChunkIdx = (myRank - round + nranks) % nranks;

    // Round 0: source is sendBuf (own original data, no cache issue)
    // Round >= 1: source is recvBuf[sendChunkIdx] (received from previous round)
    __gm__ int32_t *srcPtr = (round == 0) ? sendBuf : (recvBuf + sendChunkIdx * elemCount);
    __gm__ int32_t *remoteDst = HcclRemotePtr(hcclCtx, recvBuf, nextRank) + sendChunkIdx * elemCount;

    GlobalI32 srcG(srcPtr, shape, stride);
    GlobalI32 remoteDstG(remoteDst, shape, stride);

    ScratchTile scratchTile;
    TASSIGN(scratchTile, 0x0);
    pto::comm::AsyncSession session;
    if (!pto::comm::BuildAsyncSession(scratchTile, sdmaWorkspace, session, sdmaSyncId)) {
        pipe_barrier(PIPE_ALL);
        return;
    }

    pto::comm::AsyncEvent event = pto::comm::TPUT_ASYNC(remoteDstG, srcG, session);
    (void)event.Wait(session);

    pipe_barrier(PIPE_ALL);
}

// ============================================================================
// Ring AllGather Round Kernel (TPUT_ASYNC + dcci cache invalidation)
//
// Same as the TPUT_ASYNC kernel, but adds explicit L2 cache invalidation
// (dcci) on the source data before SDMA reads it in rounds >= 1.
// This forces L2 misses so that SDMA/AICORE fetches fresh data from HBM.
// ============================================================================
static constexpr uint64_t CACHE_LINE_SIZE = 64;

static AICORE inline void InvalidateRange(__gm__ void *addr, uint64_t bytes)
{
    __gm__ uint8_t *base = reinterpret_cast<__gm__ uint8_t *>(
        reinterpret_cast<uint64_t>(addr) / CACHE_LINE_SIZE * CACHE_LINE_SIZE);
    __gm__ uint8_t *end = reinterpret_cast<__gm__ uint8_t *>(
        (reinterpret_cast<uint64_t>(addr) + bytes) / CACHE_LINE_SIZE * CACHE_LINE_SIZE);
    for (uint64_t off = 0; off <= static_cast<uint64_t>(end - base); off += CACHE_LINE_SIZE) {
        __asm__ __volatile__("");
        dcci((__gm__ void *)(base + off), SINGLE_CACHE_LINE);
        __asm__ __volatile__("");
    }
}

__global__ AICORE void RingAllgatherDcciRoundKernel(__gm__ int32_t *dataBuf, int nranks,
                                                    __gm__ HcclDeviceContext *hcclCtx,
                                                    __gm__ uint8_t *sdmaWorkspace, uint32_t sdmaSyncId,
                                                    int elemCount, int round)
{
    if (nranks < 2) return;

    int myRank = static_cast<int>(hcclCtx->rankId);
    int nextRank = (myRank + 1) % nranks;

    __gm__ int32_t *sendBuf = dataBuf;
    __gm__ int32_t *recvBuf = dataBuf + elemCount;

    ShapeDyn shape(1, 1, 1, 1, elemCount);
    StrideDyn stride(elemCount, elemCount, elemCount, elemCount, 1);

    if (round == 0) {
        LocalTile tile(1, ELEM_COUNT);
        TASSIGN(tile, 0x10000);
        int chunkSize = static_cast<int>(ELEM_COUNT);
        int numChunks = (elemCount + chunkSize - 1) / chunkSize;
        ShapeDyn cShape(1, 1, 1, 1, chunkSize);
        StrideDyn cStride(chunkSize, chunkSize, chunkSize, chunkSize, 1);
        for (int c = 0; c < numChunks; ++c) {
            int off = c * chunkSize;
            GlobalI32 srcC(sendBuf + off, cShape, cStride);
            GlobalI32 dstC(recvBuf + myRank * elemCount + off, cShape, cStride);
            TLOAD(tile, srcC);
            set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            TSTORE(dstC, tile);
            set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
            wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        }
        pipe_barrier(PIPE_ALL);
    }

    int sendChunkIdx = (myRank - round + nranks) % nranks;

    __gm__ int32_t *srcPtr = (round == 0) ? sendBuf : (recvBuf + sendChunkIdx * elemCount);
    __gm__ int32_t *remoteDst = HcclRemotePtr(hcclCtx, recvBuf, nextRank) + sendChunkIdx * elemCount;

    if (round > 0) {
        InvalidateRange(srcPtr, static_cast<uint64_t>(elemCount) * sizeof(int32_t));
        dsb(DSB_DDR);
    }

    GlobalI32 srcG(srcPtr, shape, stride);
    GlobalI32 remoteDstG(remoteDst, shape, stride);

    ScratchTile scratchTile;
    TASSIGN(scratchTile, 0x0);
    pto::comm::AsyncSession session;
    if (!pto::comm::BuildAsyncSession(scratchTile, sdmaWorkspace, session, sdmaSyncId)) {
        pipe_barrier(PIPE_ALL);
        return;
    }

    pto::comm::AsyncEvent event = pto::comm::TPUT_ASYNC(remoteDstG, srcG, session);
    (void)event.Wait(session);

    pipe_barrier(PIPE_ALL);
}

// ============================================================================
// Ring AllGather Round Kernel (TPUT_ASYNC + AICORE stage-through)
//
// For round >= 1, instead of letting SDMA directly read from recvBuf
// (which reads stale data), we:
//   1. dcci — invalidate AICORE's L2 for the source chunk
//   2. TLOAD — AICORE reads from HBM (cache miss) → UB
//   3. TSTORE — AICORE writes from UB → sendBuf (a known-fresh HBM location)
//   4. TPUT_ASYNC — SDMA reads from sendBuf → remote
// This routes the data through AICORE's coherent path before SDMA touches it.
// ============================================================================
// diagBuf layout: [nranks] int32_t values, written by each round's kernel
// for diagnosing what TLOAD actually reads after dcci.
__global__ AICORE void RingAllgatherStageRoundKernel(__gm__ int32_t *dataBuf, int nranks,
                                                     __gm__ HcclDeviceContext *hcclCtx,
                                                     __gm__ uint8_t *sdmaWorkspace, uint32_t sdmaSyncId,
                                                     int elemCount, int round,
                                                     __gm__ int32_t *diagBuf)
{
    if (nranks < 2) return;

    int myRank = static_cast<int>(hcclCtx->rankId);
    int nextRank = (myRank + 1) % nranks;

    __gm__ int32_t *sendBuf = dataBuf;
    __gm__ int32_t *recvBuf = dataBuf + elemCount;

    ShapeDyn shape(1, 1, 1, 1, elemCount);
    StrideDyn stride(elemCount, elemCount, elemCount, elemCount, 1);

    LocalTile tile(1, ELEM_COUNT);
    TASSIGN(tile, 0x10000);

    if (round == 0) {
        int chunkSize = static_cast<int>(ELEM_COUNT);
        int numChunks = (elemCount + chunkSize - 1) / chunkSize;
        ShapeDyn cShape(1, 1, 1, 1, chunkSize);
        StrideDyn cStride(chunkSize, chunkSize, chunkSize, chunkSize, 1);
        for (int c = 0; c < numChunks; ++c) {
            int off = c * chunkSize;
            GlobalI32 srcC(sendBuf + off, cShape, cStride);
            GlobalI32 dstC(recvBuf + myRank * elemCount + off, cShape, cStride);
            TLOAD(tile, srcC);
            set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            TSTORE(dstC, tile);
            set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
            wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        }
        pipe_barrier(PIPE_ALL);
    }

    int sendChunkIdx = (myRank - round + nranks) % nranks;

    __gm__ int32_t *srcPtr;
    if (round == 0) {
        srcPtr = sendBuf;
    } else {
        __gm__ int32_t *recvChunk = recvBuf + sendChunkIdx * elemCount;

        __asm__ __volatile__("");
        dcci((__gm__ void *)recvChunk, ENTIRE_DATA_CACHE);
        __asm__ __volatile__("");
        dsb(DSB_DDR);

        int chunkSize = static_cast<int>(ELEM_COUNT);
        int numChunks = (elemCount + chunkSize - 1) / chunkSize;
        ShapeDyn cShape(1, 1, 1, 1, chunkSize);
        StrideDyn cStride(chunkSize, chunkSize, chunkSize, chunkSize, 1);
        for (int c = 0; c < numChunks; ++c) {
            int off = c * chunkSize;
            GlobalI32 srcC(recvChunk + off, cShape, cStride);
            GlobalI32 dstC(sendBuf + off, cShape, cStride);
            TLOAD(tile, srcC);
            set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            TSTORE(dstC, tile);
            set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
            wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        }
        pipe_barrier(PIPE_ALL);

        // Flush sendBuf from L2 to HBM so SDMA can read the staged data
        InvalidateRange(sendBuf, static_cast<uint64_t>(elemCount) * sizeof(int32_t));
        dsb(DSB_DDR);

        // Write first element of staged data to diagBuf for host inspection
        if (diagBuf != nullptr) {
            diagBuf[round] = *sendBuf;
            dcci((__gm__ void *)&diagBuf[round], SINGLE_CACHE_LINE);
            dsb(DSB_DDR);
        }

        srcPtr = sendBuf;
    }

    __gm__ int32_t *remoteDst = HcclRemotePtr(hcclCtx, recvBuf, nextRank) + sendChunkIdx * elemCount;

    GlobalI32 srcG(srcPtr, shape, stride);
    GlobalI32 remoteDstG(remoteDst, shape, stride);

    ScratchTile scratchTile;
    TASSIGN(scratchTile, 0x0);
    pto::comm::AsyncSession session;
    if (!pto::comm::BuildAsyncSession(scratchTile, sdmaWorkspace, session, sdmaSyncId)) {
        pipe_barrier(PIPE_ALL);
        return;
    }

    pto::comm::AsyncEvent event = pto::comm::TPUT_ASYNC(remoteDstG, srcG, session);
    (void)event.Wait(session);

    pipe_barrier(PIPE_ALL);
}

// ============================================================================
// Ring AllGather Round Kernel (TPUT — synchronous, MTE2 pipeline)
//
// Same ring algorithm as above, but uses synchronous pto::comm::TPUT instead
// of TPUT_ASYNC. TPUT goes through the AICORE's MTE2 pipeline rather than
// the external SDMA engine. This kernel is used to verify whether the
// L2 cache coherency issue also affects the synchronous TPUT path.
// ============================================================================
__global__ AICORE void RingAllgatherSyncRoundKernel(__gm__ int32_t *dataBuf, int nranks,
                                                    __gm__ HcclDeviceContext *hcclCtx, int elemCount, int round)
{
    if (nranks < 2) return;

    int myRank = static_cast<int>(hcclCtx->rankId);
    int nextRank = (myRank + 1) % nranks;

    __gm__ int32_t *sendBuf = dataBuf;
    __gm__ int32_t *recvBuf = dataBuf + elemCount;

    ShapeDyn shape(1, 1, 1, 1, elemCount);
    StrideDyn stride(elemCount, elemCount, elemCount, elemCount, 1);

    LocalTile stagingTile(1, ELEM_COUNT);
    TASSIGN(stagingTile, 0x0);

    if (round == 0) {
        int chunkSize = static_cast<int>(ELEM_COUNT);
        int numChunks = (elemCount + chunkSize - 1) / chunkSize;
        ShapeDyn cShape(1, 1, 1, 1, chunkSize);
        StrideDyn cStride(chunkSize, chunkSize, chunkSize, chunkSize, 1);
        for (int c = 0; c < numChunks; ++c) {
            int off = c * chunkSize;
            GlobalI32 srcC(sendBuf + off, cShape, cStride);
            GlobalI32 dstC(recvBuf + myRank * elemCount + off, cShape, cStride);
            TLOAD(stagingTile, srcC);
            set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            TSTORE(dstC, stagingTile);
            set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
            wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        }
        pipe_barrier(PIPE_ALL);
    }

    int sendChunkIdx = (myRank - round + nranks) % nranks;

    __gm__ int32_t *srcPtr = (round == 0) ? sendBuf : (recvBuf + sendChunkIdx * elemCount);
    __gm__ int32_t *remoteDst = HcclRemotePtr(hcclCtx, recvBuf, nextRank) + sendChunkIdx * elemCount;

    GlobalI32 srcG(srcPtr, shape, stride);
    GlobalI32 remoteDstG(remoteDst, shape, stride);

    pto::comm::TPUT(remoteDstG, srcG, stagingTile);

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
// RunAllgatherRing — host runner
// ============================================================================
static bool RunAllgatherRingKernel(int rankId, int nRanks, int nDevices, int firstDeviceId,
                                   const HcclRootInfo *rootInfo)
{
    TestContext ctx;
    if (!ctx.Init(rankId, nRanks, nDevices, firstDeviceId, rootInfo))
        return false;

    const size_t recvElems = static_cast<size_t>(nRanks) * ELEM_COUNT;

    int32_t *sendHost = nullptr;
    int32_t *recvHost = nullptr;
    aclrtMallocHost(reinterpret_cast<void **>(&sendHost), ELEM_COUNT * sizeof(int32_t));
    aclrtMallocHost(reinterpret_cast<void **>(&recvHost), recvElems * sizeof(int32_t));

    for (size_t i = 0; i < ELEM_COUNT; ++i)
        sendHost[i] = static_cast<int32_t>(rankId) * RANK_BASE + static_cast<int32_t>(i);
    for (size_t i = 0; i < recvElems; ++i)
        recvHost[i] = -1;

    uint64_t winBase = ctx.hostCtx.windowsIn[rankId];
    size_t winOff = 0;
    size_t winBytes = SYNC_BUF_BYTES + (ELEM_COUNT + recvElems) * sizeof(int32_t);
    void *commPtr = WindowAlloc(winBase, winOff, winBytes);

    int32_t *dataBuf = reinterpret_cast<int32_t *>(reinterpret_cast<uint8_t *>(commPtr) + SYNC_BUF_BYTES);
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

    int numRounds = nRanks - 1;
    for (int r = 0; r < numRounds; ++r) {
        RingAllgatherRoundKernel<<<1, nullptr, ctx.stream>>>(dataBuf, nRanks, ctx.deviceCtx,
                                                             (uint8_t *)sdmaMgr.GetWorkspaceAddr(), 0,
                                                             static_cast<int>(ELEM_COUNT), r);
        ctx.aclStatus = aclrtSynchronizeStream(ctx.stream);
        HcclHostBarrier(ctx.comm, ctx.stream);
    }

    aclrtMemcpy(recvHost, recvElems * sizeof(int32_t), recvBuf, recvElems * sizeof(int32_t),
                ACL_MEMCPY_DEVICE_TO_HOST);

    bool ok = VerifyAllgather(recvHost, nRanks, ELEM_COUNT, rankId, "RING_TPUT_ASYNC");
    if (ok) PrintSample(recvHost, nRanks, ELEM_COUNT, rankId, "RING_TPUT_ASYNC");

    aclrtFreeHost(sendHost);
    aclrtFreeHost(recvHost);
    sdmaMgr.Finalize();
    return ctx.Finalize() && ok;
}

bool RunAllgatherRing(int nRanks, int firstRankId, int firstDeviceId)
{
    return ForkAndRunWithHcclRootInfo(
        nRanks, firstRankId, firstDeviceId,
        [&](int rankId, const HcclRootInfo *rootInfo) {
            return RunAllgatherRingKernel(rankId, nRanks, nRanks, firstDeviceId, rootInfo);
        });
}

// ============================================================================
// RunAllgatherRingStage — host runner (TPUT_ASYNC + AICORE stage-through)
// ============================================================================
static bool RunAllgatherRingStageKernel(int rankId, int nRanks, int nDevices, int firstDeviceId,
                                        const HcclRootInfo *rootInfo)
{
    TestContext ctx;
    if (!ctx.Init(rankId, nRanks, nDevices, firstDeviceId, rootInfo))
        return false;

    const size_t recvElems = static_cast<size_t>(nRanks) * ELEM_COUNT;
    const int numRounds = nRanks - 1;

    int32_t *sendHost = nullptr;
    int32_t *recvHost = nullptr;
    int32_t *diagHost = nullptr;
    aclrtMallocHost(reinterpret_cast<void **>(&sendHost), ELEM_COUNT * sizeof(int32_t));
    aclrtMallocHost(reinterpret_cast<void **>(&recvHost), recvElems * sizeof(int32_t));
    aclrtMallocHost(reinterpret_cast<void **>(&diagHost), numRounds * sizeof(int32_t));

    for (size_t i = 0; i < ELEM_COUNT; ++i)
        sendHost[i] = static_cast<int32_t>(rankId) * RANK_BASE + static_cast<int32_t>(i);
    for (size_t i = 0; i < recvElems; ++i)
        recvHost[i] = -1;
    for (int i = 0; i < numRounds; ++i)
        diagHost[i] = -99;

    uint64_t winBase = ctx.hostCtx.windowsIn[rankId];
    size_t winOff = 0;
    size_t diagBytes = static_cast<size_t>(numRounds) * sizeof(int32_t);
    size_t winBytes = SYNC_BUF_BYTES + (ELEM_COUNT + recvElems) * sizeof(int32_t) + diagBytes;
    void *commPtr = WindowAlloc(winBase, winOff, winBytes);

    int32_t *dataBuf = reinterpret_cast<int32_t *>(reinterpret_cast<uint8_t *>(commPtr) + SYNC_BUF_BYTES);
    int32_t *sendBuf = dataBuf;
    int32_t *recvBuf = dataBuf + ELEM_COUNT;
    int32_t *diagBuf = reinterpret_cast<int32_t *>(reinterpret_cast<uint8_t *>(recvBuf) +
                                                    recvElems * sizeof(int32_t));

    aclrtMemcpy(sendBuf, ELEM_COUNT * sizeof(int32_t), sendHost, ELEM_COUNT * sizeof(int32_t),
                ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(recvBuf, recvElems * sizeof(int32_t), recvHost, recvElems * sizeof(int32_t),
                ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(diagBuf, diagBytes, diagHost, diagBytes, ACL_MEMCPY_HOST_TO_DEVICE);

    SdmaWorkspaceManager sdmaMgr;
    if (!sdmaMgr.Init()) {
        std::cerr << "[ERROR] SdmaWorkspaceManager Init failed" << std::endl;
        return false;
    }

    HcclHostBarrier(ctx.comm, ctx.stream);

    for (int r = 0; r < numRounds; ++r) {
        RingAllgatherStageRoundKernel<<<1, nullptr, ctx.stream>>>(dataBuf, nRanks, ctx.deviceCtx,
                                                                   (uint8_t *)sdmaMgr.GetWorkspaceAddr(), 0,
                                                                   static_cast<int>(ELEM_COUNT), r, diagBuf);
        ctx.aclStatus = aclrtSynchronizeStream(ctx.stream);
        HcclHostBarrier(ctx.comm, ctx.stream);
    }

    aclrtMemcpy(recvHost, recvElems * sizeof(int32_t), recvBuf, recvElems * sizeof(int32_t),
                ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(diagHost, diagBytes, diagBuf, diagBytes, ACL_MEMCPY_DEVICE_TO_HOST);

    std::cout << "[STAGE DIAG] Rank " << rankId << " TLOAD-after-dcci read: ";
    for (int r = 0; r < numRounds; ++r)
        std::cout << "r" << r << "=" << diagHost[r] << " ";
    std::cout << std::endl;

    bool ok = VerifyAllgather(recvHost, nRanks, ELEM_COUNT, rankId, "RING_STAGE");
    if (ok) PrintSample(recvHost, nRanks, ELEM_COUNT, rankId, "RING_STAGE");

    aclrtFreeHost(sendHost);
    aclrtFreeHost(recvHost);
    aclrtFreeHost(diagHost);
    sdmaMgr.Finalize();
    return ctx.Finalize() && ok;
}

bool RunAllgatherRingStage(int nRanks, int firstRankId, int firstDeviceId)
{
    return ForkAndRunWithHcclRootInfo(
        nRanks, firstRankId, firstDeviceId,
        [&](int rankId, const HcclRootInfo *rootInfo) {
            return RunAllgatherRingStageKernel(rankId, nRanks, nRanks, firstDeviceId, rootInfo);
        });
}

// ============================================================================
// RunAllgatherRingDcci — host runner (TPUT_ASYNC + dcci)
// ============================================================================
static bool RunAllgatherRingDcciKernel(int rankId, int nRanks, int nDevices, int firstDeviceId,
                                       const HcclRootInfo *rootInfo)
{
    TestContext ctx;
    if (!ctx.Init(rankId, nRanks, nDevices, firstDeviceId, rootInfo))
        return false;

    const size_t recvElems = static_cast<size_t>(nRanks) * ELEM_COUNT;

    int32_t *sendHost = nullptr;
    int32_t *recvHost = nullptr;
    aclrtMallocHost(reinterpret_cast<void **>(&sendHost), ELEM_COUNT * sizeof(int32_t));
    aclrtMallocHost(reinterpret_cast<void **>(&recvHost), recvElems * sizeof(int32_t));

    for (size_t i = 0; i < ELEM_COUNT; ++i)
        sendHost[i] = static_cast<int32_t>(rankId) * RANK_BASE + static_cast<int32_t>(i);
    for (size_t i = 0; i < recvElems; ++i)
        recvHost[i] = -1;

    uint64_t winBase = ctx.hostCtx.windowsIn[rankId];
    size_t winOff = 0;
    size_t winBytes = SYNC_BUF_BYTES + (ELEM_COUNT + recvElems) * sizeof(int32_t);
    void *commPtr = WindowAlloc(winBase, winOff, winBytes);

    int32_t *dataBuf = reinterpret_cast<int32_t *>(reinterpret_cast<uint8_t *>(commPtr) + SYNC_BUF_BYTES);
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

    int numRounds = nRanks - 1;
    for (int r = 0; r < numRounds; ++r) {
        RingAllgatherDcciRoundKernel<<<1, nullptr, ctx.stream>>>(dataBuf, nRanks, ctx.deviceCtx,
                                                                  (uint8_t *)sdmaMgr.GetWorkspaceAddr(), 0,
                                                                  static_cast<int>(ELEM_COUNT), r);
        ctx.aclStatus = aclrtSynchronizeStream(ctx.stream);
        HcclHostBarrier(ctx.comm, ctx.stream);
    }

    aclrtMemcpy(recvHost, recvElems * sizeof(int32_t), recvBuf, recvElems * sizeof(int32_t),
                ACL_MEMCPY_DEVICE_TO_HOST);

    bool ok = VerifyAllgather(recvHost, nRanks, ELEM_COUNT, rankId, "RING_TPUT_ASYNC_DCCI");
    if (ok) PrintSample(recvHost, nRanks, ELEM_COUNT, rankId, "RING_TPUT_ASYNC_DCCI");

    aclrtFreeHost(sendHost);
    aclrtFreeHost(recvHost);
    sdmaMgr.Finalize();
    return ctx.Finalize() && ok;
}

bool RunAllgatherRingDcci(int nRanks, int firstRankId, int firstDeviceId)
{
    return ForkAndRunWithHcclRootInfo(
        nRanks, firstRankId, firstDeviceId,
        [&](int rankId, const HcclRootInfo *rootInfo) {
            return RunAllgatherRingDcciKernel(rankId, nRanks, nRanks, firstDeviceId, rootInfo);
        });
}

// ============================================================================
// RunAllgatherRingDiag — diagnostic: read back recvBuf after each round
// to determine if SDMA data reaches HBM
// ============================================================================
static bool RunAllgatherRingDiagKernel(int rankId, int nRanks, int nDevices, int firstDeviceId,
                                       const HcclRootInfo *rootInfo)
{
    TestContext ctx;
    if (!ctx.Init(rankId, nRanks, nDevices, firstDeviceId, rootInfo))
        return false;

    const size_t recvElems = static_cast<size_t>(nRanks) * ELEM_COUNT;

    int32_t *sendHost = nullptr;
    int32_t *recvHost = nullptr;
    aclrtMallocHost(reinterpret_cast<void **>(&sendHost), ELEM_COUNT * sizeof(int32_t));
    aclrtMallocHost(reinterpret_cast<void **>(&recvHost), recvElems * sizeof(int32_t));

    for (size_t i = 0; i < ELEM_COUNT; ++i)
        sendHost[i] = static_cast<int32_t>(rankId) * RANK_BASE + static_cast<int32_t>(i);
    for (size_t i = 0; i < recvElems; ++i)
        recvHost[i] = -1;

    uint64_t winBase = ctx.hostCtx.windowsIn[rankId];
    size_t winOff = 0;
    size_t winBytes = SYNC_BUF_BYTES + (ELEM_COUNT + recvElems) * sizeof(int32_t);
    void *commPtr = WindowAlloc(winBase, winOff, winBytes);

    int32_t *dataBuf = reinterpret_cast<int32_t *>(reinterpret_cast<uint8_t *>(commPtr) + SYNC_BUF_BYTES);
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

    int numRounds = nRanks - 1;
    for (int r = 0; r < numRounds; ++r) {
        RingAllgatherRoundKernel<<<1, nullptr, ctx.stream>>>(dataBuf, nRanks, ctx.deviceCtx,
                                                             (uint8_t *)sdmaMgr.GetWorkspaceAddr(), 0,
                                                             static_cast<int>(ELEM_COUNT), r);
        ctx.aclStatus = aclrtSynchronizeStream(ctx.stream);
        HcclHostBarrier(ctx.comm, ctx.stream);

        aclrtMemcpy(recvHost, recvElems * sizeof(int32_t), recvBuf, recvElems * sizeof(int32_t),
                     ACL_MEMCPY_DEVICE_TO_HOST);
        std::cout << "[DIAG] Rank " << rankId << " after round " << r << ": ";
        for (int s = 0; s < nRanks; ++s)
            std::cout << "slot[" << s << "][0]=" << recvHost[s * ELEM_COUNT] << " ";
        std::cout << std::endl;
    }

    bool ok = VerifyAllgather(recvHost, nRanks, ELEM_COUNT, rankId, "RING_DIAG");
    if (ok) PrintSample(recvHost, nRanks, ELEM_COUNT, rankId, "RING_DIAG");

    aclrtFreeHost(sendHost);
    aclrtFreeHost(recvHost);
    sdmaMgr.Finalize();
    return ctx.Finalize() && ok;
}

bool RunAllgatherRingDiag(int nRanks, int firstRankId, int firstDeviceId)
{
    return ForkAndRunWithHcclRootInfo(
        nRanks, firstRankId, firstDeviceId,
        [&](int rankId, const HcclRootInfo *rootInfo) {
            return RunAllgatherRingDiagKernel(rankId, nRanks, nRanks, firstDeviceId, rootInfo);
        });
}

// ============================================================================
// RunAllgatherRingSync — host runner (synchronous TPUT)
// ============================================================================
static bool RunAllgatherRingSyncKernel(int rankId, int nRanks, int nDevices, int firstDeviceId,
                                       const HcclRootInfo *rootInfo)
{
    TestContext ctx;
    if (!ctx.Init(rankId, nRanks, nDevices, firstDeviceId, rootInfo))
        return false;

    const size_t recvElems = static_cast<size_t>(nRanks) * ELEM_COUNT;

    int32_t *sendHost = nullptr;
    int32_t *recvHost = nullptr;
    aclrtMallocHost(reinterpret_cast<void **>(&sendHost), ELEM_COUNT * sizeof(int32_t));
    aclrtMallocHost(reinterpret_cast<void **>(&recvHost), recvElems * sizeof(int32_t));

    for (size_t i = 0; i < ELEM_COUNT; ++i)
        sendHost[i] = static_cast<int32_t>(rankId) * RANK_BASE + static_cast<int32_t>(i);
    for (size_t i = 0; i < recvElems; ++i)
        recvHost[i] = -1;

    uint64_t winBase = ctx.hostCtx.windowsIn[rankId];
    size_t winOff = 0;
    size_t winBytes = SYNC_BUF_BYTES + (ELEM_COUNT + recvElems) * sizeof(int32_t);
    void *commPtr = WindowAlloc(winBase, winOff, winBytes);

    int32_t *dataBuf = reinterpret_cast<int32_t *>(reinterpret_cast<uint8_t *>(commPtr) + SYNC_BUF_BYTES);
    int32_t *sendBuf = dataBuf;
    int32_t *recvBuf = dataBuf + ELEM_COUNT;

    aclrtMemcpy(sendBuf, ELEM_COUNT * sizeof(int32_t), sendHost, ELEM_COUNT * sizeof(int32_t),
                ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(recvBuf, recvElems * sizeof(int32_t), recvHost, recvElems * sizeof(int32_t),
                ACL_MEMCPY_HOST_TO_DEVICE);

    HcclHostBarrier(ctx.comm, ctx.stream);

    int numRounds = nRanks - 1;
    for (int r = 0; r < numRounds; ++r) {
        RingAllgatherSyncRoundKernel<<<1, nullptr, ctx.stream>>>(dataBuf, nRanks, ctx.deviceCtx,
                                                                  static_cast<int>(ELEM_COUNT), r);
        ctx.aclStatus = aclrtSynchronizeStream(ctx.stream);
        HcclHostBarrier(ctx.comm, ctx.stream);
    }

    aclrtMemcpy(recvHost, recvElems * sizeof(int32_t), recvBuf, recvElems * sizeof(int32_t),
                ACL_MEMCPY_DEVICE_TO_HOST);

    bool ok = VerifyAllgather(recvHost, nRanks, ELEM_COUNT, rankId, "RING_TPUT_SYNC");
    if (ok) PrintSample(recvHost, nRanks, ELEM_COUNT, rankId, "RING_TPUT_SYNC");

    aclrtFreeHost(sendHost);
    aclrtFreeHost(recvHost);
    return ctx.Finalize() && ok;
}

bool RunAllgatherRingSync(int nRanks, int firstRankId, int firstDeviceId)
{
    return ForkAndRunWithHcclRootInfo(
        nRanks, firstRankId, firstDeviceId,
        [&](int rankId, const HcclRootInfo *rootInfo) {
            return RunAllgatherRingSyncKernel(rankId, nRanks, nRanks, firstDeviceId, rootInfo);
        });
}

// ============================================================================
// RunAllgatherRingDelayed — TPUT_ASYNC with a configurable sleep between rounds
// Used to distinguish "data not yet arrived" from "L2 cache stale".
// If sleep fixes the issue → data arrival timing problem.
// If sleep doesn't help → L2 cache coherency problem.
// ============================================================================
static bool RunAllgatherRingDelayedKernel(int rankId, int nRanks, int nDevices, int firstDeviceId,
                                          const HcclRootInfo *rootInfo, unsigned int delaySec)
{
    TestContext ctx;
    if (!ctx.Init(rankId, nRanks, nDevices, firstDeviceId, rootInfo))
        return false;

    const size_t recvElems = static_cast<size_t>(nRanks) * ELEM_COUNT;

    int32_t *sendHost = nullptr;
    int32_t *recvHost = nullptr;
    aclrtMallocHost(reinterpret_cast<void **>(&sendHost), ELEM_COUNT * sizeof(int32_t));
    aclrtMallocHost(reinterpret_cast<void **>(&recvHost), recvElems * sizeof(int32_t));

    for (size_t i = 0; i < ELEM_COUNT; ++i)
        sendHost[i] = static_cast<int32_t>(rankId) * RANK_BASE + static_cast<int32_t>(i);
    for (size_t i = 0; i < recvElems; ++i)
        recvHost[i] = -1;

    uint64_t winBase = ctx.hostCtx.windowsIn[rankId];
    size_t winOff = 0;
    size_t winBytes = SYNC_BUF_BYTES + (ELEM_COUNT + recvElems) * sizeof(int32_t);
    void *commPtr = WindowAlloc(winBase, winOff, winBytes);

    int32_t *dataBuf = reinterpret_cast<int32_t *>(reinterpret_cast<uint8_t *>(commPtr) + SYNC_BUF_BYTES);
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

    int numRounds = nRanks - 1;
    for (int r = 0; r < numRounds; ++r) {
        RingAllgatherRoundKernel<<<1, nullptr, ctx.stream>>>(dataBuf, nRanks, ctx.deviceCtx,
                                                             (uint8_t *)sdmaMgr.GetWorkspaceAddr(), 0,
                                                             static_cast<int>(ELEM_COUNT), r);
        ctx.aclStatus = aclrtSynchronizeStream(ctx.stream);
        HcclHostBarrier(ctx.comm, ctx.stream);
        sleep(delaySec);
        HcclHostBarrier(ctx.comm, ctx.stream);
    }

    aclrtMemcpy(recvHost, recvElems * sizeof(int32_t), recvBuf, recvElems * sizeof(int32_t),
                ACL_MEMCPY_DEVICE_TO_HOST);

    char tag[64];
    snprintf(tag, sizeof(tag), "RING_TPUT_ASYNC_DELAY_%us", delaySec);
    bool ok = VerifyAllgather(recvHost, nRanks, ELEM_COUNT, rankId, tag);
    if (ok) PrintSample(recvHost, nRanks, ELEM_COUNT, rankId, tag);

    aclrtFreeHost(sendHost);
    aclrtFreeHost(recvHost);
    sdmaMgr.Finalize();
    return ctx.Finalize() && ok;
}

bool RunAllgatherRingDelayed(int nRanks, int firstRankId, int firstDeviceId, unsigned int delaySec)
{
    return ForkAndRunWithHcclRootInfo(
        nRanks, firstRankId, firstDeviceId,
        [&](int rankId, const HcclRootInfo *rootInfo) {
            return RunAllgatherRingDelayedKernel(rankId, nRanks, nRanks, firstDeviceId, rootInfo, delaySec);
        });
}
