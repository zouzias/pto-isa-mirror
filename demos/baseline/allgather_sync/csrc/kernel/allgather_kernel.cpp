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
#include <iomanip>
#include <sys/time.h>

#include <pto/pto-inst.hpp>
#include "pto/common/pto_tile.hpp"
#include "common.hpp"

// ============================================================================
// Constants
// ============================================================================
static constexpr size_t ELEM_COUNT = 256;
static constexpr size_t SYNC_BUF_BYTES = 64 * sizeof(int32_t);
static constexpr int32_t RANK_BASE = 1000;

// Perf profiling constants
static constexpr int WARMUP_ITERS = 20;
static constexpr int TIMED_ITERS = 100;
static constexpr int kLatLocalCopyIdx = 0;
static constexpr int kLatRemoteCommIdx = 1;
static constexpr int kLatTotalIdx = 2;
static constexpr int kLatMetricsCount = 3;
static constexpr double kSysCntPerUs = 50.0;

inline AICORE uint64_t get_syscnt()
{
    uint64_t syscnt;
    asm volatile("MOV %0, SYS_CNT\n" : "+l"(syscnt));
    return syscnt;
}

static double NowUs()
{
    timeval tv{};
    gettimeofday(&tv, nullptr);
    return static_cast<double>(tv.tv_sec) * 1e6 + static_cast<double>(tv.tv_usec);
}

static inline double CyclesToUs(double cycles) { return cycles / kSysCntPerUs; }

using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
using GlobalI32 = pto::GlobalTensor<int32_t, ShapeDyn, StrideDyn, pto::Layout::ND>;
using VecTile = pto::Tile<pto::TileType::Vec, int32_t, 1, ELEM_COUNT, pto::BLayout::RowMajor, -1, -1>;

// ============================================================================
// Allgather via synchronous TPUT
//
// Every rank writes its sendBuf to every other rank's recvBuf slot using
// pto::comm::TPUT (local GM -> UB staging tile -> remote GM).
// Also copies its own data locally to complete the allgather.
//
// Memory layout per rank (in shared window):
//   sendBuf[ELEM_COUNT]               -- this rank's contribution
//   recvBuf[nranks * ELEM_COUNT]      -- gathered result
// ============================================================================
__global__ AICORE void AllgatherPutSyncKernel(__gm__ int32_t *dataBuf, int nranks,
                                              __gm__ HcclDeviceContext *hcclCtx)
{
    if (nranks < 2) return;

    ShapeDyn shape(1, 1, 1, 1, ELEM_COUNT);
    StrideDyn stride(ELEM_COUNT, ELEM_COUNT, ELEM_COUNT, ELEM_COUNT, 1);

    int myRank = static_cast<int>(hcclCtx->rankId);
    __gm__ int32_t *sendBuf = dataBuf;
    __gm__ int32_t *recvBuf = dataBuf + ELEM_COUNT;

    VecTile stagingTile(1, ELEM_COUNT);
    TASSIGN(stagingTile, 0x0);

    // Local copy: recvBuf[myRank * ELEM_COUNT] = sendBuf
    GlobalI32 srcG(sendBuf, shape, stride);
    GlobalI32 localSlotG(recvBuf + myRank * ELEM_COUNT, shape, stride);
    TLOAD(stagingTile, srcG);
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    TSTORE(localSlotG, stagingTile);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);

    // Synchronous remote writes to every other rank
    GlobalI32 sendG(sendBuf, shape, stride);
    for (int target = 0; target < nranks; ++target) {
        if (target == myRank) continue;
        __gm__ int32_t *remoteSlot = HcclRemotePtr(hcclCtx, recvBuf, target) + myRank * ELEM_COUNT;
        GlobalI32 remoteG(remoteSlot, shape, stride);
        pto::comm::TPUT(remoteG, sendG, stagingTile);
    }

    pipe_barrier(PIPE_ALL);
}

// ============================================================================
// Allgather via synchronous TGET
//
// Every rank pulls every other rank's sendBuf into its local recvBuf slot
// using pto::comm::TGET (remote GM -> UB staging tile -> local GM).
// Also copies its own data locally to complete the allgather.
// ============================================================================
__global__ AICORE void AllgatherGetSyncKernel(__gm__ int32_t *dataBuf, int nranks,
                                              __gm__ HcclDeviceContext *hcclCtx)
{
    if (nranks < 2) return;

    ShapeDyn shape(1, 1, 1, 1, ELEM_COUNT);
    StrideDyn stride(ELEM_COUNT, ELEM_COUNT, ELEM_COUNT, ELEM_COUNT, 1);

    int myRank = static_cast<int>(hcclCtx->rankId);
    __gm__ int32_t *sendBuf = dataBuf;
    __gm__ int32_t *recvBuf = dataBuf + ELEM_COUNT;

    VecTile stagingTile(1, ELEM_COUNT);
    TASSIGN(stagingTile, 0x0);

    // Local copy: recvBuf[myRank * ELEM_COUNT] = sendBuf
    GlobalI32 srcG(sendBuf, shape, stride);
    GlobalI32 localSlotG(recvBuf + myRank * ELEM_COUNT, shape, stride);
    TLOAD(stagingTile, srcG);
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    TSTORE(localSlotG, stagingTile);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);

    // Synchronous remote reads from every other rank
    for (int src = 0; src < nranks; ++src) {
        if (src == myRank) continue;
        __gm__ int32_t *remoteSend = HcclRemotePtr(hcclCtx, sendBuf, src);
        GlobalI32 remoteG(remoteSend, shape, stride);
        GlobalI32 localG(recvBuf + src * ELEM_COUNT, shape, stride);
        pto::comm::TGET(localG, remoteG, stagingTile);
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
// RunAllgatherPutSync
// ============================================================================
static bool RunAllgatherPutSyncKernel(int rankId, int nRanks, int nDevices, int firstDeviceId,
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

    HcclHostBarrier(ctx.comm, ctx.stream);

    AllgatherPutSyncKernel<<<1, nullptr, ctx.stream>>>(dataBuf, nRanks, ctx.deviceCtx);
    ctx.aclStatus = aclrtSynchronizeStream(ctx.stream);

    HcclHostBarrier(ctx.comm, ctx.stream);

    aclrtMemcpy(recvHost, recvElems * sizeof(int32_t), recvBuf, recvElems * sizeof(int32_t),
                ACL_MEMCPY_DEVICE_TO_HOST);

    bool ok = VerifyAllgather(recvHost, nRanks, ELEM_COUNT, rankId, "TPUT_SYNC");
    if (ok) PrintSample(recvHost, nRanks, ELEM_COUNT, rankId, "TPUT_SYNC");

    aclrtFreeHost(sendHost);
    aclrtFreeHost(recvHost);

    return ctx.Finalize() && ok;
}

bool RunAllgatherPutSync(int nRanks, int firstRankId, int firstDeviceId)
{
    return ForkAndRunWithHcclRootInfo(
        nRanks, firstRankId, firstDeviceId, [&](int rankId, const HcclRootInfo *rootInfo) {
            return RunAllgatherPutSyncKernel(rankId, nRanks, nRanks, firstDeviceId, rootInfo);
        });
}

// ============================================================================
// RunAllgatherGetSync
// ============================================================================
static bool RunAllgatherGetSyncKernel(int rankId, int nRanks, int nDevices, int firstDeviceId,
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

    HcclHostBarrier(ctx.comm, ctx.stream);

    AllgatherGetSyncKernel<<<1, nullptr, ctx.stream>>>(dataBuf, nRanks, ctx.deviceCtx);
    ctx.aclStatus = aclrtSynchronizeStream(ctx.stream);

    HcclHostBarrier(ctx.comm, ctx.stream);

    aclrtMemcpy(recvHost, recvElems * sizeof(int32_t), recvBuf, recvElems * sizeof(int32_t),
                ACL_MEMCPY_DEVICE_TO_HOST);

    bool ok = VerifyAllgather(recvHost, nRanks, ELEM_COUNT, rankId, "TGET_SYNC");
    if (ok) PrintSample(recvHost, nRanks, ELEM_COUNT, rankId, "TGET_SYNC");

    aclrtFreeHost(sendHost);
    aclrtFreeHost(recvHost);

    return ctx.Finalize() && ok;
}

bool RunAllgatherGetSync(int nRanks, int firstRankId, int firstDeviceId)
{
    return ForkAndRunWithHcclRootInfo(
        nRanks, firstRankId, firstDeviceId, [&](int rankId, const HcclRootInfo *rootInfo) {
            return RunAllgatherGetSyncKernel(rankId, nRanks, nRanks, firstDeviceId, rootInfo);
        });
}

// ============================================================================
// Bandwidth Sweep: variable-size perf kernels with UB chunking
// ============================================================================
static constexpr int CHUNK_ELEMS = 8192;  // 32 KB staging tile for UB
using ChunkTile = pto::Tile<pto::TileType::Vec, int32_t, 1, CHUNK_ELEMS, pto::BLayout::RowMajor, -1, -1>;

__global__ AICORE void AllgatherPutSyncPerfKernel(__gm__ int32_t *dataBuf, int nranks,
                                                  __gm__ HcclDeviceContext *hcclCtx,
                                                  __gm__ uint64_t *latencyMetrics,
                                                  int elemCount, int warmupIters, int timedIters)
{
    if (nranks < 2) return;

    int myRank = static_cast<int>(hcclCtx->rankId);
    __gm__ int32_t *sendBuf = dataBuf;
    __gm__ int32_t *recvBuf = dataBuf + elemCount;

    int chunkSize = (elemCount < CHUNK_ELEMS) ? elemCount : CHUNK_ELEMS;
    int numChunks = (elemCount + chunkSize - 1) / chunkSize;
    ChunkTile pingTile(1, chunkSize);
    ChunkTile pongTile(1, chunkSize);
    TASSIGN(pingTile, 0x0);
    TASSIGN(pongTile, 0x8000);
    ShapeDyn cShape(1, 1, 1, 1, chunkSize);
    StrideDyn cStride(chunkSize, chunkSize, chunkSize, chunkSize, 1);
    ShapeDyn fullShape(1, 1, 1, 1, elemCount);
    StrideDyn fullStride(elemCount, elemCount, elemCount, elemCount, 1);

    uint64_t localCopyCycles = 0, remoteCommCycles = 0, totalCycles = 0;

    for (int iter = 0; iter < warmupIters + timedIters; ++iter) {
        pipe_barrier(PIPE_ALL);
        uint64_t t0 = get_syscnt();

        for (int c = 0; c < numChunks; ++c) {
            int off = c * chunkSize;
            GlobalI32 srcC(sendBuf + off, cShape, cStride);
            GlobalI32 dstC(recvBuf + myRank * elemCount + off, cShape, cStride);
            TLOAD(pingTile, srcC);
            set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            TSTORE(dstC, pingTile);
            set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
            wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        }

        pipe_barrier(PIPE_ALL);
        uint64_t t1 = get_syscnt();

        GlobalI32 sendG(sendBuf, fullShape, fullStride);
        for (int target = 0; target < nranks; ++target) {
            if (target == myRank) continue;
            __gm__ int32_t *remoteRecv = HcclRemotePtr(hcclCtx, recvBuf, target) + myRank * elemCount;
            GlobalI32 remoteG(remoteRecv, fullShape, fullStride);
            pto::comm::TPUT(remoteG, sendG, pingTile, pongTile);
        }

        pipe_barrier(PIPE_ALL);
        uint64_t t2 = get_syscnt();

        if (iter >= warmupIters) {
            localCopyCycles += t1 - t0;
            remoteCommCycles += t2 - t1;
            totalCycles += t2 - t0;
        }
    }

    pipe_barrier(PIPE_ALL);
    if (latencyMetrics) {
        latencyMetrics[kLatLocalCopyIdx] = localCopyCycles;
        latencyMetrics[kLatRemoteCommIdx] = remoteCommCycles;
        latencyMetrics[kLatTotalIdx] = totalCycles;
    }
}

__global__ AICORE void AllgatherGetSyncPerfKernel(__gm__ int32_t *dataBuf, int nranks,
                                                  __gm__ HcclDeviceContext *hcclCtx,
                                                  __gm__ uint64_t *latencyMetrics,
                                                  int elemCount, int warmupIters, int timedIters)
{
    if (nranks < 2) return;

    int myRank = static_cast<int>(hcclCtx->rankId);
    __gm__ int32_t *sendBuf = dataBuf;
    __gm__ int32_t *recvBuf = dataBuf + elemCount;

    int chunkSize = (elemCount < CHUNK_ELEMS) ? elemCount : CHUNK_ELEMS;
    int numChunks = (elemCount + chunkSize - 1) / chunkSize;
    ChunkTile pingTile(1, chunkSize);
    ChunkTile pongTile(1, chunkSize);
    TASSIGN(pingTile, 0x0);
    TASSIGN(pongTile, 0x8000);
    ShapeDyn cShape(1, 1, 1, 1, chunkSize);
    StrideDyn cStride(chunkSize, chunkSize, chunkSize, chunkSize, 1);
    ShapeDyn fullShape(1, 1, 1, 1, elemCount);
    StrideDyn fullStride(elemCount, elemCount, elemCount, elemCount, 1);

    uint64_t localCopyCycles = 0, remoteCommCycles = 0, totalCycles = 0;

    for (int iter = 0; iter < warmupIters + timedIters; ++iter) {
        pipe_barrier(PIPE_ALL);
        uint64_t t0 = get_syscnt();

        for (int c = 0; c < numChunks; ++c) {
            int off = c * chunkSize;
            GlobalI32 srcC(sendBuf + off, cShape, cStride);
            GlobalI32 dstC(recvBuf + myRank * elemCount + off, cShape, cStride);
            TLOAD(pingTile, srcC);
            set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            TSTORE(dstC, pingTile);
            set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
            wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        }

        pipe_barrier(PIPE_ALL);
        uint64_t t1 = get_syscnt();

        for (int src = 0; src < nranks; ++src) {
            if (src == myRank) continue;
            __gm__ int32_t *remoteSend = HcclRemotePtr(hcclCtx, sendBuf, src);
            GlobalI32 remoteG(remoteSend, fullShape, fullStride);
            GlobalI32 localG(recvBuf + src * elemCount, fullShape, fullStride);
            pto::comm::TGET(localG, remoteG, pingTile, pongTile);
        }

        pipe_barrier(PIPE_ALL);
        uint64_t t2 = get_syscnt();

        if (iter >= warmupIters) {
            localCopyCycles += t1 - t0;
            remoteCommCycles += t2 - t1;
            totalCycles += t2 - t0;
        }
    }

    pipe_barrier(PIPE_ALL);
    if (latencyMetrics) {
        latencyMetrics[kLatLocalCopyIdx] = localCopyCycles;
        latencyMetrics[kLatRemoteCommIdx] = remoteCommCycles;
        latencyMetrics[kLatTotalIdx] = totalCycles;
    }
}

// ============================================================================
// RunAllgatherSyncSweep — single TestContext, all sizes, TPUT + TGET
// ============================================================================
static void PrintSweepLine(const char *instr, size_t sizeBytes,
                           double hostE2e, double devTotal, double localCopy, double remoteComm)
{
    std::cout << std::fixed << std::setprecision(4)
              << "[SWEEP] instr=" << instr
              << "  size_bytes=" << sizeBytes
              << "  host_e2e_avg_us=" << hostE2e
              << "  device_total_avg_us=" << devTotal
              << "  local_copy_avg_us=" << localCopy
              << "  remote_comm_avg_us=" << remoteComm
              << std::endl;
}

static bool RunAllgatherSyncSweepKernel(int rankId, int nRanks, int nDevices, int firstDeviceId,
                                        const HcclRootInfo *rootInfo)
{
    TestContext ctx;
    if (!ctx.Init(rankId, nRanks, nDevices, firstDeviceId, rootInfo))
        return false;

    static const size_t kSweepElems[] = {1024, 4096, 16384, 65536, 262144, 1048576};
    static const int kNumSizes = sizeof(kSweepElems) / sizeof(kSweepElems[0]);

    uint64_t *latDev = nullptr;
    uint64_t *latHost = nullptr;
    aclrtMalloc(reinterpret_cast<void **>(&latDev), kLatMetricsCount * sizeof(uint64_t), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMallocHost(reinterpret_cast<void **>(&latHost), kLatMetricsCount * sizeof(uint64_t));

    for (int si = 0; si < kNumSizes; ++si) {
        size_t elemCount = kSweepElems[si];
        size_t recvElems = static_cast<size_t>(nRanks) * elemCount;
        size_t winBytes = SYNC_BUF_BYTES + (elemCount + recvElems) * sizeof(int32_t);

        if (winBytes > ctx.hostCtx.winSize) {
            if (rankId == 0)
                std::cout << "[SWEEP] instr=TPUT_SYNC  size_bytes=" << elemCount * 4 << "  SKIPPED (window=" << ctx.hostCtx.winSize << ")" << std::endl;
            continue;
        }

        uint64_t winBase = ctx.hostCtx.windowsIn[rankId];
        size_t winOff = 0;
        void *commPtr = WindowAlloc(winBase, winOff, winBytes);
        int32_t *dataBuf = reinterpret_cast<int32_t *>(
            reinterpret_cast<uint8_t *>(commPtr) + SYNC_BUF_BYTES);

        aclrtMemset(dataBuf, (elemCount + recvElems) * sizeof(int32_t), 0,
                    (elemCount + recvElems) * sizeof(int32_t));
        aclrtMemset(latDev, kLatMetricsCount * sizeof(uint64_t), 0, kLatMetricsCount * sizeof(uint64_t));

        HcclHostBarrier(ctx.comm, ctx.stream);

        // TPUT sweep
        double t0 = NowUs();
        AllgatherPutSyncPerfKernel<<<1, nullptr, ctx.stream>>>(
            dataBuf, nRanks, ctx.deviceCtx, latDev, static_cast<int>(elemCount), WARMUP_ITERS, TIMED_ITERS);
        ctx.aclStatus = aclrtSynchronizeStream(ctx.stream);
        double t1 = NowUs();

        HcclHostBarrier(ctx.comm, ctx.stream);

        aclrtMemcpy(latHost, kLatMetricsCount * sizeof(uint64_t),
                    latDev, kLatMetricsCount * sizeof(uint64_t), ACL_MEMCPY_DEVICE_TO_HOST);

        if (rankId == 0) {
            PrintSweepLine("TPUT_SYNC", elemCount * 4,
                           (t1 - t0) / TIMED_ITERS,
                           CyclesToUs(static_cast<double>(latHost[kLatTotalIdx]) / TIMED_ITERS),
                           CyclesToUs(static_cast<double>(latHost[kLatLocalCopyIdx]) / TIMED_ITERS),
                           CyclesToUs(static_cast<double>(latHost[kLatRemoteCommIdx]) / TIMED_ITERS));
        }

        // TGET sweep with same size
        aclrtMemset(dataBuf, (elemCount + recvElems) * sizeof(int32_t), 0,
                    (elemCount + recvElems) * sizeof(int32_t));
        aclrtMemset(latDev, kLatMetricsCount * sizeof(uint64_t), 0, kLatMetricsCount * sizeof(uint64_t));

        HcclHostBarrier(ctx.comm, ctx.stream);

        t0 = NowUs();
        AllgatherGetSyncPerfKernel<<<1, nullptr, ctx.stream>>>(
            dataBuf, nRanks, ctx.deviceCtx, latDev, static_cast<int>(elemCount), WARMUP_ITERS, TIMED_ITERS);
        ctx.aclStatus = aclrtSynchronizeStream(ctx.stream);
        t1 = NowUs();

        HcclHostBarrier(ctx.comm, ctx.stream);

        aclrtMemcpy(latHost, kLatMetricsCount * sizeof(uint64_t),
                    latDev, kLatMetricsCount * sizeof(uint64_t), ACL_MEMCPY_DEVICE_TO_HOST);

        if (rankId == 0) {
            PrintSweepLine("TGET_SYNC", elemCount * 4,
                           (t1 - t0) / TIMED_ITERS,
                           CyclesToUs(static_cast<double>(latHost[kLatTotalIdx]) / TIMED_ITERS),
                           CyclesToUs(static_cast<double>(latHost[kLatLocalCopyIdx]) / TIMED_ITERS),
                           CyclesToUs(static_cast<double>(latHost[kLatRemoteCommIdx]) / TIMED_ITERS));
        }
    }

    aclrtFreeHost(latHost);
    aclrtFree(latDev);
    return ctx.Finalize();
}

bool RunAllgatherSyncSweep(int nRanks, int firstRankId, int firstDeviceId)
{
    return ForkAndRunWithHcclRootInfo(
        nRanks, firstRankId, firstDeviceId, [&](int rankId, const HcclRootInfo *rootInfo) {
            return RunAllgatherSyncSweepKernel(rankId, nRanks, nRanks, firstDeviceId, rootInfo);
        });
}
