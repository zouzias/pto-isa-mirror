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
#include "pto/npu/comm/async/sdma/sdma_types.hpp"
#include "pto/common/pto_tile.hpp"
#include "common.hpp"

// ============================================================================
// Constants
// ============================================================================
static constexpr size_t SYNC_BUF_BYTES = 64 * sizeof(int32_t);

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
using ScratchTile = pto::Tile<pto::TileType::Vec, uint8_t, 1, pto::comm::sdma::UB_ALIGN_SIZE>;

// ============================================================================
// Bandwidth Sweep: Ring + Recursive Doubling perf kernels
// ============================================================================
static constexpr int CHUNK_ELEMS = 8192;
using ChunkTile = pto::Tile<pto::TileType::Vec, int32_t, 1, CHUNK_ELEMS, pto::BLayout::RowMajor, -1, -1>;

// ============================================================================
// Strategy: Ring AllGather — one TPUT_ASYNC per round, nRanks-1 rounds
// ============================================================================
__global__ AICORE void RingRoundKernel(__gm__ int32_t *dataBuf, int nranks,
                                       __gm__ HcclDeviceContext *hcclCtx,
                                       __gm__ uint8_t *sdmaWorkspace, uint32_t sdmaSyncId,
                                       __gm__ uint64_t *roundCyclesBuf,
                                       int elemCount, int round)
{
    if (nranks < 2) return;

    int myRank = static_cast<int>(hcclCtx->rankId);
    __gm__ int32_t *sendBuf = dataBuf;
    __gm__ int32_t *recvBuf = dataBuf + elemCount;

    int chunkSize = (elemCount < CHUNK_ELEMS) ? elemCount : CHUNK_ELEMS;
    int numChunks = (elemCount + chunkSize - 1) / chunkSize;
    ChunkTile localTile(1, chunkSize);
    TASSIGN(localTile, 0x10000);
    ShapeDyn cShape(1, 1, 1, 1, chunkSize);
    StrideDyn cStride(chunkSize, chunkSize, chunkSize, chunkSize, 1);

    ShapeDyn fullShape(1, 1, 1, 1, elemCount);
    StrideDyn fullStride(elemCount, elemCount, elemCount, elemCount, 1);

    uint64_t t0 = get_syscnt();
    pipe_barrier(PIPE_ALL);

    if (round == 0) {
        for (int c = 0; c < numChunks; ++c) {
            int off = c * chunkSize;
            GlobalI32 srcC(sendBuf + off, cShape, cStride);
            GlobalI32 dstC(recvBuf + myRank * elemCount + off, cShape, cStride);
            TLOAD(localTile, srcC);
            set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            TSTORE(dstC, localTile);
            set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
            wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        }
        pipe_barrier(PIPE_ALL);
    }

    int sendChunkIdx = (myRank - round + nranks) % nranks;
    int nextRank = (myRank + 1) % nranks;

    __gm__ int32_t *localChunk = recvBuf + sendChunkIdx * elemCount;
    __gm__ int32_t *remoteChunk = HcclRemotePtr(hcclCtx, recvBuf, nextRank) + sendChunkIdx * elemCount;
    GlobalI32 localG(localChunk, fullShape, fullStride);
    GlobalI32 remoteG(remoteChunk, fullShape, fullStride);

    ScratchTile scratchTile;
    TASSIGN(scratchTile, 0x0);
    pto::comm::AsyncSession session;
    if (!pto::comm::BuildAsyncSession(scratchTile, sdmaWorkspace, session, sdmaSyncId)) {
        pipe_barrier(PIPE_ALL);
        return;
    }

    pto::comm::AsyncEvent event = pto::comm::TPUT_ASYNC(remoteG, localG, session);
    (void)event.Wait(session);

    pipe_barrier(PIPE_ALL);
    uint64_t t1 = get_syscnt();

    if (roundCyclesBuf)
        roundCyclesBuf[0] = t1 - t0;
}

// ============================================================================
// Strategy: Recursive Doubling — log2(N) rounds, exchange 2^r chunks/round
// ============================================================================
__global__ AICORE void RecDoublingRoundKernel(__gm__ int32_t *dataBuf, int nranks,
                                              __gm__ HcclDeviceContext *hcclCtx,
                                              __gm__ uint8_t *sdmaWorkspace, uint32_t sdmaSyncId,
                                              __gm__ uint64_t *roundCyclesBuf,
                                              int elemCount, int round)
{
    if (nranks < 2) return;

    int myRank = static_cast<int>(hcclCtx->rankId);
    __gm__ int32_t *sendBuf = dataBuf;
    __gm__ int32_t *recvBuf = dataBuf + elemCount;

    int chunkSize = (elemCount < CHUNK_ELEMS) ? elemCount : CHUNK_ELEMS;
    int numChunks = (elemCount + chunkSize - 1) / chunkSize;
    ChunkTile localTile(1, chunkSize);
    TASSIGN(localTile, 0x10000);
    ShapeDyn cShape(1, 1, 1, 1, chunkSize);
    StrideDyn cStride(chunkSize, chunkSize, chunkSize, chunkSize, 1);

    ShapeDyn fullShape(1, 1, 1, 1, elemCount);
    StrideDyn fullStride(elemCount, elemCount, elemCount, elemCount, 1);

    uint64_t t0 = get_syscnt();
    pipe_barrier(PIPE_ALL);

    if (round == 0) {
        for (int c = 0; c < numChunks; ++c) {
            int off = c * chunkSize;
            GlobalI32 srcC(sendBuf + off, cShape, cStride);
            GlobalI32 dstC(recvBuf + myRank * elemCount + off, cShape, cStride);
            TLOAD(localTile, srcC);
            set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            TSTORE(dstC, localTile);
            set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
            wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        }
        pipe_barrier(PIPE_ALL);
    }

    int partner = myRank ^ (1 << round);
    if (partner >= nranks) {
        pipe_barrier(PIPE_ALL);
        uint64_t t1 = get_syscnt();
        if (roundCyclesBuf) roundCyclesBuf[0] = t1 - t0;
        return;
    }

    ScratchTile scratchTile;
    TASSIGN(scratchTile, 0x0);
    pto::comm::AsyncSession session;
    if (!pto::comm::BuildAsyncSession(scratchTile, sdmaWorkspace, session, sdmaSyncId)) {
        pipe_barrier(PIPE_ALL);
        return;
    }

    constexpr int kEventSlots = pto::comm::sdma::SDMA_EVENT_SLOT_COUNT;
    pto::comm::AsyncEvent events[kEventSlots];
    int issued = 0;

    int myBit = (myRank >> round) & 1;
    for (int chunkIdx = 0; chunkIdx < nranks; ++chunkIdx) {
        if (((chunkIdx >> round) & 1) != myBit) continue;

        bool iHaveIt = true;
        for (int b = 0; b < round; ++b) {
            if (((chunkIdx >> b) & 1) != ((myRank >> b) & 1)) {
                iHaveIt = false;
                break;
            }
        }
        if (!iHaveIt) continue;

        __gm__ int32_t *lc = recvBuf + chunkIdx * elemCount;
        __gm__ int32_t *rc = HcclRemotePtr(hcclCtx, recvBuf, partner) + chunkIdx * elemCount;
        GlobalI32 lG(lc, fullShape, fullStride);
        GlobalI32 rG(rc, fullShape, fullStride);

        if (issued >= kEventSlots)
            (void)events[issued % kEventSlots].Wait(session);
        events[issued % kEventSlots] = pto::comm::TPUT_ASYNC(rG, lG, session);
        issued++;
    }

    int pending = (issued < kEventSlots) ? issued : kEventSlots;
    for (int i = 0; i < pending; ++i)
        (void)events[i].Wait(session);

    pipe_barrier(PIPE_ALL);
    uint64_t t1 = get_syscnt();

    if (roundCyclesBuf)
        roundCyclesBuf[0] = t1 - t0;
}

// ============================================================================
// RunAllgatherAsyncSweep — all sizes, all strategies
// ============================================================================
static double BwGbps(size_t sizeBytes, double latencyUs)
{
    if (latencyUs <= 0.0) return 0.0;
    return static_cast<double>(sizeBytes) / latencyUs / 1000.0;
}

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
              << "  host_bw_gbps=" << BwGbps(sizeBytes, hostE2e)
              << "  device_bw_gbps=" << BwGbps(sizeBytes, devTotal)
              << std::endl;
}

static int Log2Int(int n) { int r = 0; while ((1 << r) < n) ++r; return r; }

static void RunRingSweep(TestContext &ctx, int rankId, int nRanks,
                         SdmaWorkspaceManager &sdmaMgr,
                         uint64_t *latDev, uint64_t * /*latHost*/,
                         size_t elemCount, size_t recvElems, int32_t *dataBuf)
{
    int numRounds = nRanks - 1;
    double hostTotalUs = 0.0;
    uint64_t devTotalCycles = 0;

    for (int iter = 0; iter < WARMUP_ITERS + TIMED_ITERS; ++iter) {
        aclrtMemset(dataBuf, (elemCount + recvElems) * sizeof(int32_t), 0,
                    (elemCount + recvElems) * sizeof(int32_t));
        HcclHostBarrier(ctx.comm, ctx.stream);

        double iterT0 = NowUs();
        uint64_t iterDevCycles = 0;

        for (int r = 0; r < numRounds; ++r) {
            aclrtMemset(latDev, sizeof(uint64_t), 0, sizeof(uint64_t));
            RingRoundKernel<<<1, nullptr, ctx.stream>>>(
                dataBuf, nRanks, ctx.deviceCtx, (uint8_t *)sdmaMgr.GetWorkspaceAddr(), 0,
                latDev, static_cast<int>(elemCount), r);
            ctx.aclStatus = aclrtSynchronizeStream(ctx.stream);
            HcclHostBarrier(ctx.comm, ctx.stream);

            if (rankId == 0) {
                uint64_t rc = 0;
                aclrtMemcpy(&rc, sizeof(uint64_t), latDev, sizeof(uint64_t), ACL_MEMCPY_DEVICE_TO_HOST);
                iterDevCycles += rc;
            }
        }

        double iterT1 = NowUs();
        if (iter >= WARMUP_ITERS) {
            hostTotalUs += (iterT1 - iterT0);
            devTotalCycles += iterDevCycles;
        }
    }

    if (rankId == 0) {
        double hostAvg = hostTotalUs / TIMED_ITERS;
        double devAvg = CyclesToUs(static_cast<double>(devTotalCycles) / TIMED_ITERS);
        PrintSweepLine("RING", elemCount * 4, hostAvg, devAvg, 0.0, devAvg);
    }
}

static void RunRecDblSweep(TestContext &ctx, int rankId, int nRanks,
                           SdmaWorkspaceManager &sdmaMgr,
                           uint64_t *latDev, uint64_t * /*latHost*/,
                           size_t elemCount, size_t recvElems, int32_t *dataBuf)
{
    int numRounds = Log2Int(nRanks);
    double hostTotalUs = 0.0;
    uint64_t devTotalCycles = 0;

    for (int iter = 0; iter < WARMUP_ITERS + TIMED_ITERS; ++iter) {
        aclrtMemset(dataBuf, (elemCount + recvElems) * sizeof(int32_t), 0,
                    (elemCount + recvElems) * sizeof(int32_t));
        HcclHostBarrier(ctx.comm, ctx.stream);

        double iterT0 = NowUs();
        uint64_t iterDevCycles = 0;

        for (int r = 0; r < numRounds; ++r) {
            aclrtMemset(latDev, sizeof(uint64_t), 0, sizeof(uint64_t));
            RecDoublingRoundKernel<<<1, nullptr, ctx.stream>>>(
                dataBuf, nRanks, ctx.deviceCtx, (uint8_t *)sdmaMgr.GetWorkspaceAddr(), 0,
                latDev, static_cast<int>(elemCount), r);
            ctx.aclStatus = aclrtSynchronizeStream(ctx.stream);
            HcclHostBarrier(ctx.comm, ctx.stream);

            if (rankId == 0) {
                uint64_t rc = 0;
                aclrtMemcpy(&rc, sizeof(uint64_t), latDev, sizeof(uint64_t), ACL_MEMCPY_DEVICE_TO_HOST);
                iterDevCycles += rc;
            }
        }

        double iterT1 = NowUs();
        if (iter >= WARMUP_ITERS) {
            hostTotalUs += (iterT1 - iterT0);
            devTotalCycles += iterDevCycles;
        }
    }

    if (rankId == 0) {
        double hostAvg = hostTotalUs / TIMED_ITERS;
        double devAvg = CyclesToUs(static_cast<double>(devTotalCycles) / TIMED_ITERS);
        PrintSweepLine("REC_DBL", elemCount * 4, hostAvg, devAvg, 0.0, devAvg);
    }
}

static bool RunAllgatherAsyncSweepKernel(int rankId, int nRanks, int nDevices, int firstDeviceId,
                                         const HcclRootInfo *rootInfo)
{
    TestContext ctx;
    if (!ctx.Init(rankId, nRanks, nDevices, firstDeviceId, rootInfo))
        return false;

    SdmaWorkspaceManager sdmaMgr;
    if (!sdmaMgr.Init()) {
        std::cerr << "[ERROR] SdmaWorkspaceManager Init failed" << std::endl;
        return false;
    }

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
                std::cout << "[SWEEP] size_bytes=" << elemCount * 4 << "  SKIPPED (window=" << ctx.hostCtx.winSize << ")" << std::endl;
            continue;
        }

        uint64_t winBase = ctx.hostCtx.windowsIn[rankId];
        size_t winOff = 0;
        void *commPtr = WindowAlloc(winBase, winOff, winBytes);
        int32_t *dataBuf = reinterpret_cast<int32_t *>(
            reinterpret_cast<uint8_t *>(commPtr) + SYNC_BUF_BYTES);

        RunRingSweep(ctx, rankId, nRanks, sdmaMgr, latDev, latHost, elemCount, recvElems, dataBuf);
        RunRecDblSweep(ctx, rankId, nRanks, sdmaMgr, latDev, latHost, elemCount, recvElems, dataBuf);
    }

    aclrtFreeHost(latHost);
    aclrtFree(latDev);
    sdmaMgr.Finalize();
    return ctx.Finalize();
}

bool RunAllgatherAsyncSweep(int nRanks, int firstRankId, int firstDeviceId)
{
    return ForkAndRunWithHcclRootInfo(
        nRanks, firstRankId, firstDeviceId, [&](int rankId, const HcclRootInfo *rootInfo) {
            return RunAllgatherAsyncSweepKernel(rankId, nRanks, nRanks, firstDeviceId, rootInfo);
        });
}

// ============================================================================
// Perf: Multi-core TPUT_ASYNC with variable-size sweep
// ============================================================================
__global__ AICORE void AllgatherPutAsyncMulticorePerfKernel(
    __gm__ int32_t *dataBuf, int nranks,
    __gm__ HcclDeviceContext *hcclCtx,
    __gm__ uint8_t *sdmaWorkspace, uint32_t sdmaSyncId,
    __gm__ uint64_t *perBlockCycles,
    int elemCount, int warmupIters, int timedIters)
{
    if (nranks < 2) return;

    int bid = block_idx;
    int myRank = static_cast<int>(hcclCtx->rankId);
    __gm__ int32_t *sendBuf = dataBuf;
    __gm__ int32_t *recvBuf = dataBuf + elemCount;

    int chunkSize = (elemCount < CHUNK_ELEMS) ? elemCount : CHUNK_ELEMS;
    int numChunks = (elemCount + chunkSize - 1) / chunkSize;
    ChunkTile localTile(1, chunkSize);
    TASSIGN(localTile, 0x10000);
    ShapeDyn cShape(1, 1, 1, 1, chunkSize);
    ShapeDyn fullShape(1, 1, 1, 1, elemCount);
    StrideDyn cStride(chunkSize, chunkSize, chunkSize, chunkSize, 1);
    StrideDyn fullStride(elemCount, elemCount, elemCount, elemCount, 1);

    ScratchTile scratchTile;
    TASSIGN(scratchTile, 0x0);
    pto::comm::AsyncSession session;
    bool hasSession = false;
    if (bid != myRank) {
        hasSession = pto::comm::BuildAsyncSession(scratchTile, sdmaWorkspace, session, sdmaSyncId);
        if (!hasSession) {
            pipe_barrier(PIPE_ALL);
            if (perBlockCycles) perBlockCycles[bid] = 0;
            return;
        }
    }

    uint64_t totalCycles = 0;

    for (int iter = 0; iter < warmupIters + timedIters; ++iter) {
        pipe_barrier(PIPE_ALL);
        uint64_t t0 = get_syscnt();

        if (bid == myRank) {
            for (int c = 0; c < numChunks; ++c) {
                int off = c * chunkSize;
                GlobalI32 srcC(sendBuf + off, cShape, cStride);
                GlobalI32 dstC(recvBuf + myRank * elemCount + off, cShape, cStride);
                TLOAD(localTile, srcC);
                set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
                wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
                TSTORE(dstC, localTile);
                set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
                wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
            }
        } else {
            int target = bid;
            GlobalI32 sendG(sendBuf, fullShape, fullStride);
            __gm__ int32_t *remoteSlot = HcclRemotePtr(hcclCtx, recvBuf, target) + myRank * elemCount;
            GlobalI32 remoteG(remoteSlot, fullShape, fullStride);

            pto::comm::AsyncEvent event = pto::comm::TPUT_ASYNC(remoteG, sendG, session);
            (void)event.Wait(session);
        }

        pipe_barrier(PIPE_ALL);
        uint64_t t1 = get_syscnt();

        if (iter >= warmupIters) {
            totalCycles += t1 - t0;
        }
    }

    pipe_barrier(PIPE_ALL);
    if (perBlockCycles) {
        perBlockCycles[bid] = totalCycles;
    }
}

// ============================================================================
// Perf: Multi-core TGET_ASYNC with variable-size sweep
// ============================================================================
__global__ AICORE void AllgatherGetAsyncMulticorePerfKernel(
    __gm__ int32_t *dataBuf, int nranks,
    __gm__ HcclDeviceContext *hcclCtx,
    __gm__ uint8_t *sdmaWorkspace, uint32_t sdmaSyncId,
    __gm__ uint64_t *perBlockCycles,
    int elemCount, int warmupIters, int timedIters)
{
    if (nranks < 2) return;

    int bid = block_idx;
    int myRank = static_cast<int>(hcclCtx->rankId);
    __gm__ int32_t *sendBuf = dataBuf;
    __gm__ int32_t *recvBuf = dataBuf + elemCount;

    int chunkSize = (elemCount < CHUNK_ELEMS) ? elemCount : CHUNK_ELEMS;
    int numChunks = (elemCount + chunkSize - 1) / chunkSize;
    ChunkTile localTile(1, chunkSize);
    TASSIGN(localTile, 0x10000);
    ShapeDyn cShape(1, 1, 1, 1, chunkSize);
    ShapeDyn fullShape(1, 1, 1, 1, elemCount);
    StrideDyn cStride(chunkSize, chunkSize, chunkSize, chunkSize, 1);
    StrideDyn fullStride(elemCount, elemCount, elemCount, elemCount, 1);

    ScratchTile scratchTile;
    TASSIGN(scratchTile, 0x0);
    pto::comm::AsyncSession session;
    bool hasSession = false;
    if (bid != myRank) {
        hasSession = pto::comm::BuildAsyncSession(scratchTile, sdmaWorkspace, session, sdmaSyncId);
        if (!hasSession) {
            pipe_barrier(PIPE_ALL);
            if (perBlockCycles) perBlockCycles[bid] = 0;
            return;
        }
    }

    uint64_t totalCycles = 0;

    for (int iter = 0; iter < warmupIters + timedIters; ++iter) {
        pipe_barrier(PIPE_ALL);
        uint64_t t0 = get_syscnt();

        if (bid == myRank) {
            for (int c = 0; c < numChunks; ++c) {
                int off = c * chunkSize;
                GlobalI32 srcC(sendBuf + off, cShape, cStride);
                GlobalI32 dstC(recvBuf + myRank * elemCount + off, cShape, cStride);
                TLOAD(localTile, srcC);
                set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
                wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
                TSTORE(dstC, localTile);
                set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
                wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
            }
        } else {
            int srcRank = bid;
            __gm__ int32_t *remoteSend = HcclRemotePtr(hcclCtx, sendBuf, srcRank);
            GlobalI32 remoteG(remoteSend, fullShape, fullStride);
            GlobalI32 localG(recvBuf + srcRank * elemCount, fullShape, fullStride);

            pto::comm::AsyncEvent event = pto::comm::TGET_ASYNC(localG, remoteG, session);
            (void)event.Wait(session);
        }

        pipe_barrier(PIPE_ALL);
        uint64_t t1 = get_syscnt();

        if (iter >= warmupIters) {
            totalCycles += t1 - t0;
        }
    }

    pipe_barrier(PIPE_ALL);
    if (perBlockCycles) {
        perBlockCycles[bid] = totalCycles;
    }
}

// ============================================================================
// RunAllgatherMcAsyncSweep — TPUT_ASYNC_MC + TGET_ASYNC_MC across data sizes
// ============================================================================
static bool RunAllgatherMcAsyncSweepKernel(int rankId, int nRanks, int nDevices, int firstDeviceId,
                                           const HcclRootInfo *rootInfo)
{
    TestContext ctx;
    if (!ctx.Init(rankId, nRanks, nDevices, firstDeviceId, rootInfo))
        return false;

    SdmaWorkspaceManager sdmaMgr;
    if (!sdmaMgr.Init()) {
        std::cerr << "[ERROR] SdmaWorkspaceManager Init failed" << std::endl;
        return false;
    }

    static const size_t kSweepElems[] = {1024, 4096, 16384, 65536, 262144, 1048576};
    static const int kNumSizes = sizeof(kSweepElems) / sizeof(kSweepElems[0]);

    size_t mcLatBytes = static_cast<size_t>(nRanks) * sizeof(uint64_t);
    uint64_t *mcLatDev = nullptr;
    uint64_t *mcLatHost = nullptr;
    aclrtMalloc(reinterpret_cast<void **>(&mcLatDev), mcLatBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMallocHost(reinterpret_cast<void **>(&mcLatHost), mcLatBytes);

    for (int si = 0; si < kNumSizes; ++si) {
        size_t elemCount = kSweepElems[si];
        size_t recvElems = static_cast<size_t>(nRanks) * elemCount;
        size_t winBytes = SYNC_BUF_BYTES + (elemCount + recvElems) * sizeof(int32_t);

        if (winBytes > ctx.hostCtx.winSize) {
            if (rankId == 0)
                std::cout << "[SWEEP] size_bytes=" << elemCount * 4 << "  SKIPPED (window=" << ctx.hostCtx.winSize << ")" << std::endl;
            continue;
        }

        uint64_t winBase = ctx.hostCtx.windowsIn[rankId];
        size_t winOff = 0;
        void *commPtr = WindowAlloc(winBase, winOff, winBytes);
        int32_t *dataBuf = reinterpret_cast<int32_t *>(
            reinterpret_cast<uint8_t *>(commPtr) + SYNC_BUF_BYTES);

        // --- TPUT_ASYNC_MC ---
        aclrtMemset(dataBuf, (elemCount + recvElems) * sizeof(int32_t), 0,
                    (elemCount + recvElems) * sizeof(int32_t));
        aclrtMemset(mcLatDev, mcLatBytes, 0, mcLatBytes);
        HcclHostBarrier(ctx.comm, ctx.stream);

        double t0 = NowUs();
        AllgatherPutAsyncMulticorePerfKernel<<<nRanks, nullptr, ctx.stream>>>(
            dataBuf, nRanks, ctx.deviceCtx,
            (uint8_t *)sdmaMgr.GetWorkspaceAddr(), 0,
            mcLatDev, static_cast<int>(elemCount), WARMUP_ITERS, TIMED_ITERS);
        ctx.aclStatus = aclrtSynchronizeStream(ctx.stream);
        double t1 = NowUs();

        HcclHostBarrier(ctx.comm, ctx.stream);
        aclrtMemcpy(mcLatHost, mcLatBytes, mcLatDev, mcLatBytes, ACL_MEMCPY_DEVICE_TO_HOST);

        if (rankId == 0) {
            uint64_t maxCycles = 0;
            for (int b = 0; b < nRanks; ++b) {
                if (mcLatHost[b] > maxCycles) maxCycles = mcLatHost[b];
            }
            double devTotal = CyclesToUs(static_cast<double>(maxCycles) / TIMED_ITERS);
            PrintSweepLine("TPUT_ASYNC_MC", elemCount * 4,
                           (t1 - t0) / TIMED_ITERS, devTotal, 0.0, devTotal);
        }

        // --- TGET_ASYNC_MC ---
        aclrtMemset(dataBuf, (elemCount + recvElems) * sizeof(int32_t), 0,
                    (elemCount + recvElems) * sizeof(int32_t));
        aclrtMemset(mcLatDev, mcLatBytes, 0, mcLatBytes);
        HcclHostBarrier(ctx.comm, ctx.stream);

        t0 = NowUs();
        AllgatherGetAsyncMulticorePerfKernel<<<nRanks, nullptr, ctx.stream>>>(
            dataBuf, nRanks, ctx.deviceCtx,
            (uint8_t *)sdmaMgr.GetWorkspaceAddr(), 0,
            mcLatDev, static_cast<int>(elemCount), WARMUP_ITERS, TIMED_ITERS);
        ctx.aclStatus = aclrtSynchronizeStream(ctx.stream);
        t1 = NowUs();

        HcclHostBarrier(ctx.comm, ctx.stream);
        aclrtMemcpy(mcLatHost, mcLatBytes, mcLatDev, mcLatBytes, ACL_MEMCPY_DEVICE_TO_HOST);

        if (rankId == 0) {
            uint64_t maxCycles = 0;
            for (int b = 0; b < nRanks; ++b) {
                if (mcLatHost[b] > maxCycles) maxCycles = mcLatHost[b];
            }
            double devTotal = CyclesToUs(static_cast<double>(maxCycles) / TIMED_ITERS);
            PrintSweepLine("TGET_ASYNC_MC", elemCount * 4,
                           (t1 - t0) / TIMED_ITERS, devTotal, 0.0, devTotal);
        }
    }

    aclrtFreeHost(mcLatHost);
    aclrtFree(mcLatDev);
    sdmaMgr.Finalize();
    return ctx.Finalize();
}

bool RunAllgatherMcAsyncSweep(int nRanks, int firstRankId, int firstDeviceId)
{
    return ForkAndRunWithHcclRootInfo(
        nRanks, firstRankId, firstDeviceId, [&](int rankId, const HcclRootInfo *rootInfo) {
            return RunAllgatherMcAsyncSweepKernel(rankId, nRanks, nRanks, firstDeviceId, rootInfo);
        });
}
