/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// ============================================================================
// Scenario D — cross-rank receiver-side prefetch over TPUT_ASYNC. Compares
// host-initiated pto::PTO_PREFETCH against device-initiated pto::TPREFETCH_L2
// for the same producer-consumer flow.
//
// Single-card scenarios (A/B/C) live under
// tests/npu/a2a3/src/st/testcase/tprefetch_compare/.
//
// The TPUT_ASYNC sender side still uses an explicit AsyncSession because we
// also issue TPREFETCH_L2 against the same workspace inside the kernel for the
// D2 receiver path. Building one session and reusing it amortises the
// per-call session-build cost.
// ============================================================================

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>

#include <pto/pto-inst.hpp>
#include <pto/npu/kernels/Pto_prefetch.hpp>
#include "pto/npu/comm/async/sdma/sdma_types.hpp"
#include "pto/npu/comm/async/sdma/sdma_workspace_manager.hpp"
#include "pto/common/pto_tile.hpp"
#include "../common.hpp"

using SdmaWorkspaceManager = pto::comm::sdma::SdmaWorkspaceManager;

// ============================================================================
// Shared device-side aliases and helpers
// ============================================================================
using CmpShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
using CmpStrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
template <typename T>
using CmpGlobal = pto::GlobalTensor<T, CmpShapeDyn, CmpStrideDyn, pto::Layout::ND>;
using CmpScratchTile = pto::Tile<pto::TileType::Vec, uint8_t, 1, pto::comm::sdma::UB_ALIGN_SIZE>;

// Cycle counter. Same asm pattern used elsewhere in the prefetch tests.
inline AICORE uint64_t cmp_syscnt()
{
    uint64_t syscnt;
    asm volatile("MOV %0, SYS_CNT\n" : "+l"(syscnt));
    return syscnt;
}

// Standard "chunked TLOAD" loop used by the receiver kernels. Returns the
// number of syscnt cycles spent inside the loop so callers can fold it into a
// larger kernel-cycle measurement.
template <typename T, size_t count>
PTO_INTERNAL uint64_t TloadSweepCycles(__gm__ T *srcBuf, int elem_count)
{
    constexpr int kTileCols = (count <= 256) ? static_cast<int>(count) : 256;
    static_assert(count % kTileCols == 0, "count must be a multiple of kTileCols");
    using TileData = pto::Tile<pto::TileType::Vec, T, 1, kTileCols, pto::BLayout::RowMajor>;
    using ChunkShape = pto::Shape<1, 1, 1, 1, kTileCols>;
    using ChunkStride = pto::Stride<1, 1, 1, 1, 1>;

    TileData tile;
    TASSIGN(tile, 0x0);

    uint64_t t0 = cmp_syscnt();
    for (int offset = 0; offset < elem_count; offset += kTileCols) {
        pto::GlobalTensor<T, ChunkShape, ChunkStride, pto::Layout::ND> srcChunk(srcBuf + offset);
        TLOAD(tile, srcChunk);
        set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    }
    uint64_t t1 = cmp_syscnt();
    return t1 - t0;
}

PTO_INTERNAL bool BuildCmpSession(pto::comm::AsyncSession &session, __gm__ uint8_t *sdmaWorkspace)
{
    CmpScratchTile scratchTile;
    TASSIGN(scratchTile, 0x0);
    return pto::comm::BuildAsyncSession(scratchTile, sdmaWorkspace, session, 0U);
}

// ============================================================================
// Host-side: shared sizing + clock helpers (kept here so we don't leak the
// scenario-A/B/C test fixtures into this cross-rank-only file).
// ============================================================================

// Size of the L2-trashing buffer. Must be larger than the device L2 so a full
// CMO prefetch of it evicts every test-related line. 512 MB is comfortably
// above the L2 on A2/A3 (~192 MB) and leaves enough margin for A5 as well,
// while still being small relative to a single HBM stack. Override with
// -DCMP_L2_TRASH_BYTES=<n> at compile time if a specific platform needs more.
#ifndef CMP_L2_TRASH_BYTES
#define CMP_L2_TRASH_BYTES (512ULL * 1024ULL * 1024ULL)
#endif
static constexpr size_t kL2TrashBytes = CMP_L2_TRASH_BYTES;

// Convert device syscnt cycles to microseconds. On aarch64 hosts (the Ascend
// server case) we read CNTFRQ_EL0 directly, which is exactly the frequency
// used by the AICore `MOV %0, SYS_CNT` intrinsic. On non-aarch64 hosts (e.g.
// x86 cross-build) we fall back to 50 MHz, which matches the default
// Ascend910 system-counter frequency; override with -DCMP_SYSCNT_HZ=<n> to
// match a specific platform.
inline double SyscntHz()
{
#if defined(__aarch64__)
    uint64_t freq = 0;
    asm volatile("mrs %0, cntfrq_el0" : "=r"(freq));
    if (freq > 0) {
        return static_cast<double>(freq);
    }
#endif
#ifdef CMP_SYSCNT_HZ
    return static_cast<double>(CMP_SYSCNT_HZ);
#else
    return 50.0e6;
#endif
}

inline double CyclesToUs(uint64_t cycles)
{
    return static_cast<double>(cycles) * 1.0e6 / SyscntHz();
}

using HrClock = std::chrono::steady_clock;

inline uint64_t ElapsedMicros(HrClock::time_point t0, HrClock::time_point t1)
{
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count());
}

// ============================================================================
// Scenario D — cross-rank receiver-side prefetch
//
//   Rank 0 (sender)   : pto::comm::TPUT_ASYNC(remote_recvBuf, sendBuf)
//   Rank 1 (receiver) : trash L2 -> [mode-specific prefetch] -> TLOAD(recvBuf)
//
// Three modes on the receiver:
//   D0  no prefetch
//   D1  host   pto::PTO_PREFETCH  (aclrtCmoAsync on receiver's stream)
//   D2  device pto::TPREFETCH_L2 (SQE issued inside the TLOAD kernel)
//
// commBuf layout (allocated via HCCL window, so the sender can reach it via
// HcclRemotePtr):   [ 64*int32 header | sendBuf: count*T | recvBuf: count*T ]
// (The 64*int32 header is a convention carried over from other comm tests.)
// ============================================================================

// ---- Device kernels --------------------------------------------------------

// Sender-side kernel: fan out sendBuf to every remote recvBuf via TPUT_ASYNC.
// Receiver ranks early-return; the per-rank runner decides who launches which
// kernel, but a rank guard inside the kernel is kept so that a mis-launch
// on the wrong rank is a no-op instead of a data-race on someone else's buf.
template <typename T, size_t count>
__global__ AICORE void ScenarioD_SenderKernel(__gm__ T *commBuf, int nranks, int sender_rank, int elem_count,
                                              __gm__ HcclDeviceContext *hcclCtx, __gm__ uint8_t *sdmaWorkspace)
{
    if (elem_count <= 0 || elem_count > static_cast<int>(count)) {
        pipe_barrier(PIPE_ALL);
        return;
    }
    int my_rank = static_cast<int>(hcclCtx->rankId);
    if (my_rank != sender_rank) {
        pipe_barrier(PIPE_ALL);
        return;
    }

    pto::comm::AsyncSession session;
    if (!BuildCmpSession(session, sdmaWorkspace)) {
        pipe_barrier(PIPE_ALL);
        return;
    }

    __gm__ T *sendBuf = commBuf;
    __gm__ T *recvBuf = commBuf + count;
    CmpShapeDyn shape(1, 1, 1, 1, elem_count);
    CmpStrideDyn stride(elem_count, elem_count, elem_count, elem_count, 1);
    CmpGlobal<T> sendG(sendBuf, shape, stride);

    pto::comm::AsyncEvent lastEvent;
    for (int target = 0; target < nranks; ++target) {
        if (target == sender_rank) {
            continue;
        }
        __gm__ T *remoteRecv = HcclRemotePtr(hcclCtx, recvBuf, target);
        CmpGlobal<T> remoteRecvG(remoteRecv, shape, stride);
        lastEvent = pto::comm::TPUT_ASYNC(remoteRecvG, sendG, session);
    }
    (void)lastEvent.Wait(session);
    pipe_barrier(PIPE_ALL);
}

// Receiver kernel for D0/D1 — just TLOAD. For D1 the host-side prefetch has
// already been enqueued (and synchronized) on the receiver stream before this
// kernel is launched, so TLOAD sees a warm L2.
template <typename T, size_t count>
__global__ AICORE void ScenarioD_ReceiverTloadOnlyKernel(__gm__ T *commBuf, int sender_rank, int elem_count,
                                                         __gm__ HcclDeviceContext *hcclCtx,
                                                         __gm__ uint64_t *cycleOut)
{
    if (elem_count <= 0 || elem_count > static_cast<int>(count)) {
        pipe_barrier(PIPE_ALL);
        return;
    }
    int my_rank = static_cast<int>(hcclCtx->rankId);
    if (my_rank == sender_rank) {
        pipe_barrier(PIPE_ALL);
        return;
    }
    __gm__ T *recvBuf = commBuf + count;
    *cycleOut = TloadSweepCycles<T, count>(recvBuf, elem_count);
    pipe_barrier(PIPE_ALL);
}

// Receiver kernel for D2 — device TPREFETCH_L2 (workspace API) + Wait + TLOAD,
// reports total kernel cycles (matches what the host wall-clock is measuring
// around the single kernel launch).
template <typename T, size_t count>
__global__ AICORE void ScenarioD_ReceiverDevicePrefetchKernel(__gm__ T *commBuf, int sender_rank, int elem_count,
                                                              __gm__ HcclDeviceContext *hcclCtx,
                                                              __gm__ uint8_t *sdmaWorkspace,
                                                              __gm__ uint64_t *cycleOut)
{
    if (elem_count <= 0 || elem_count > static_cast<int>(count)) {
        pipe_barrier(PIPE_ALL);
        return;
    }
    int my_rank = static_cast<int>(hcclCtx->rankId);
    if (my_rank == sender_rank) {
        pipe_barrier(PIPE_ALL);
        return;
    }
    __gm__ T *recvBuf = commBuf + count;

    uint64_t t0 = cmp_syscnt();

    {
        uint64_t totalBytes = static_cast<uint64_t>(elem_count) * sizeof(T);
        auto evt = pto::TPREFETCH_L2(reinterpret_cast<__gm__ void *>(recvBuf), totalBytes, sdmaWorkspace);
        (void)evt.Wait();
    }

    uint64_t tloadCycles = TloadSweepCycles<T, count>(recvBuf, elem_count);

    uint64_t t1 = cmp_syscnt();
    *cycleOut = t1 - t0;
    (void)tloadCycles;
    pipe_barrier(PIPE_ALL);
}

// ---- Host-side per-rank environment ---------------------------------------

template <typename T, size_t count>
struct CrossRankEnv {
    TestContext ctx;
    SdmaWorkspaceManager sdmaMgr;
    T *sendBuf = nullptr;
    T *recvBuf = nullptr;
    void *cycleDev = nullptr;
    void *trashDev = nullptr;
    uint8_t *input_host = nullptr;

    // HCCL window offset layout (bytes):
    //   [ 0 .. 64*int32 )                 reserved header
    //   [ 64*int32 .. 64*int32 + NT )     sendBuf
    //   [ 64*int32 + NT .. 64*int32 + 2NT ) recvBuf
    //   NT = count * sizeof(T)
    bool Init(int rankId, int nRanks, int firstDeviceId, const HcclRootInfo *rootInfo)
    {
        if (!ctx.Init(rankId, nRanks, /*nDevices=*/nRanks, firstDeviceId, rootInfo)) {
            return false;
        }
        if (aclrtMallocHost(reinterpret_cast<void **>(&input_host), count * sizeof(T)) != 0) {
            std::cerr << "[ERROR] ScenarioD env: aclrtMallocHost failed" << std::endl;
            return false;
        }
        for (size_t i = 0; i < count; ++i) {
            reinterpret_cast<T *>(input_host)[i] = static_cast<T>(i);
        }

        uint64_t localWinBase = ctx.hostCtx.windowsIn[rankId];
        size_t winOffset = 0;
        void *commBufPtr = WindowAlloc(localWinBase, winOffset, 64 * sizeof(int32_t) + 2 * count * sizeof(T));
        uint8_t *commBytes = reinterpret_cast<uint8_t *>(commBufPtr);
        sendBuf = reinterpret_cast<T *>(commBytes + 64 * sizeof(int32_t));
        recvBuf = sendBuf + count;

        if (aclrtMalloc(&trashDev, kL2TrashBytes, ACL_MEM_MALLOC_HUGE_FIRST) != 0) {
            std::cerr << "[ERROR] ScenarioD env: trashDev malloc failed" << std::endl;
            return false;
        }
        if (aclrtMalloc(&cycleDev, sizeof(uint64_t), ACL_MEM_MALLOC_HUGE_FIRST) != 0) {
            std::cerr << "[ERROR] ScenarioD env: cycleDev malloc failed" << std::endl;
            return false;
        }
        uint64_t zero = 0;
        aclrtMemcpy(cycleDev, sizeof(uint64_t), &zero, sizeof(uint64_t), ACL_MEMCPY_HOST_TO_DEVICE);

        if (!sdmaMgr.Init()) {
            std::cerr << "[ERROR] ScenarioD env: SdmaWorkspaceManager Init failed" << std::endl;
            return false;
        }
        return true;
    }

    void UploadInputToSend()
    {
        aclrtMemcpy(sendBuf, count * sizeof(T), input_host, count * sizeof(T), ACL_MEMCPY_HOST_TO_DEVICE);
    }

    void TrashL2OnThisRank()
    {
        pto::PTO_PREFETCH(trashDev, kL2TrashBytes, ctx.stream);
        aclrtSynchronizeStream(ctx.stream);
    }

    uint64_t ReadCycles()
    {
        uint64_t c = 0;
        aclrtMemcpy(&c, sizeof(uint64_t), cycleDev, sizeof(uint64_t), ACL_MEMCPY_DEVICE_TO_HOST);
        return c;
    }

    bool Finalize()
    {
        sdmaMgr.Finalize();
        if (cycleDev)
            aclrtFree(cycleDev);
        if (trashDev)
            aclrtFree(trashDev);
        if (input_host)
            aclrtFreeHost(input_host);
        return ctx.Finalize();
    }
};

// ---- Per-rank runner -------------------------------------------------------

// Runs kWarmup + kMeasured iterations for one prefetch mode on the receiver.
// The collective barrier pattern below is:
//
//   (start of iteration)
//   barrier 1  → both ranks aligned; receiver trashes its own L2 while sender waits
//   barrier 2  → trash done, sender launches TPUT_ASYNC kernel
//                (receiver waits so the producer-side SDMA is fully inflight)
//   barrier 3  → sender done; receiver now does prefetch + TLOAD and times it
//   barrier 4  → receiver done, loop around
//
// Timing ONLY the receiver's prefetch + TLOAD phase is the whole point of
// Scenario D; everything else (setup, TPUT_ASYNC, inter-rank wait) is hidden
// behind host-side barriers so it doesn't pollute the measurement.
template <typename T, size_t count>
bool RunScenarioDPerRankMode(CrossRankEnv<T, count> &env, int rankId, int sender_rank, int nRanks, int mode,
                             uint64_t &avgWallUsOut, uint64_t &avgKernCyclesOut)
{
    constexpr int kWarmup = 2;
    constexpr int kMeasured = 5;

    const bool isSender = (rankId == sender_rank);
    const bool isReceiver = !isSender;
    const uint64_t bytes = static_cast<uint64_t>(count) * sizeof(T);

    uint64_t wallAcc = 0;
    uint64_t kernAcc = 0;

    for (int it = 0; it < kWarmup + kMeasured; ++it) {
        HcclHostBarrier(env.ctx.comm, env.ctx.stream);

        if (isReceiver) {
            env.TrashL2OnThisRank();
        }

        HcclHostBarrier(env.ctx.comm, env.ctx.stream);

        if (isSender) {
            ScenarioD_SenderKernel<T, count><<<1, nullptr, env.ctx.stream>>>(
                env.sendBuf, nRanks, sender_rank, static_cast<int>(count), env.ctx.deviceCtx,
                reinterpret_cast<uint8_t *>(env.sdmaMgr.GetWorkspaceAddr()));
            aclrtSynchronizeStream(env.ctx.stream);
        }

        HcclHostBarrier(env.ctx.comm, env.ctx.stream);

        uint64_t wallUs = 0;
        uint64_t kernCyc = 0;
        if (isReceiver) {
            auto t0 = HrClock::now();
            if (mode == 1) {
                pto::PTO_PREFETCH(env.recvBuf, bytes, env.ctx.stream);
                // no host-side sync here: launching the TLOAD kernel on the
                // same stream immediately after naturally serializes, which
                // is the defining shape of host-side prefetch.
            }
            if (mode == 2) {
                ScenarioD_ReceiverDevicePrefetchKernel<T, count><<<1, nullptr, env.ctx.stream>>>(
                    env.sendBuf, sender_rank, static_cast<int>(count), env.ctx.deviceCtx,
                    reinterpret_cast<uint8_t *>(env.sdmaMgr.GetWorkspaceAddr()),
                    reinterpret_cast<uint64_t *>(env.cycleDev));
            } else {
                ScenarioD_ReceiverTloadOnlyKernel<T, count><<<1, nullptr, env.ctx.stream>>>(
                    env.sendBuf, sender_rank, static_cast<int>(count), env.ctx.deviceCtx,
                    reinterpret_cast<uint64_t *>(env.cycleDev));
            }
            aclrtSynchronizeStream(env.ctx.stream);
            auto t1 = HrClock::now();
            wallUs = ElapsedMicros(t0, t1);
            kernCyc = env.ReadCycles();
        }

        HcclHostBarrier(env.ctx.comm, env.ctx.stream);

        if (isReceiver && it >= kWarmup) {
            wallAcc += wallUs;
            kernAcc += kernCyc;
        }
    }

    avgWallUsOut = isReceiver ? (wallAcc / static_cast<uint64_t>(kMeasured)) : 0ULL;
    avgKernCyclesOut = isReceiver ? (kernAcc / static_cast<uint64_t>(kMeasured)) : 0ULL;
    return true;
}

// ---- Top-level driver ------------------------------------------------------
template <typename T, size_t count>
bool RunScenarioDCrossRank(int nRanks, int firstDeviceId)
{
    const int sender_rank = 0;
    constexpr size_t dataBytes = count * sizeof(T);

    return ForkAndRunWithHcclRootInfo(nRanks, 0, firstDeviceId, [&](int rankId, const HcclRootInfo *rootInfo) {
        CrossRankEnv<T, count> env;
        if (!env.Init(rankId, nRanks, firstDeviceId, rootInfo)) {
            env.Finalize();
            return false;
        }
        if (rankId == sender_rank) {
            env.UploadInputToSend();
        }

        uint64_t wall[3] = {0, 0, 0};
        uint64_t kern[3] = {0, 0, 0};
        for (int mode = 0; mode < 3; ++mode) {
            if (!RunScenarioDPerRankMode<T, count>(env, rankId, sender_rank, nRanks, mode, wall[mode], kern[mode])) {
                env.Finalize();
                return false;
            }
        }

        // Only the receiver has meaningful numbers — print from the receiver
        // rank so the log doesn't show duplicated "0 us" blocks from the sender.
        if (rankId != sender_rank) {
            std::cout << std::fixed << std::setprecision(2);
            std::cout << "\n================================================================" << std::endl;
            std::cout << "[PERF] Scenario D - cross-rank receiver-side prefetch (TPUT_ASYNC -> TLOAD)"
                      << std::endl;
            std::cout << "  Data size:             " << dataBytes << " bytes (" << dataBytes / 1024 / 1024 << " MB)"
                      << std::endl;
            std::cout << "  Ranks:                 " << nRanks << "  (sender=" << sender_rank
                      << ", receiver=" << rankId << ")" << std::endl;
            std::cout << "  Syscnt freq assumed:   " << (SyscntHz() / 1.0e6) << " MHz" << std::endl;
            std::cout << "  D0 no-prefetch           wall=" << static_cast<double>(wall[0])
                      << " us   kernel=" << CyclesToUs(kern[0]) << " us" << std::endl;
            std::cout << "  D1 host   PTO_PREFETCH   wall=" << static_cast<double>(wall[1])
                      << " us   kernel=" << CyclesToUs(kern[1]) << " us" << std::endl;
            std::cout << "  D2 device TPREFETCH_L2   wall=" << static_cast<double>(wall[2])
                      << " us   kernel=" << CyclesToUs(kern[2]) << " us   (incl. prefetch+wait)" << std::endl;
            std::cout << "  (D2 kernel time covers prefetch+Wait+TLOAD; D0/D1 kernel is TLOAD only)" << std::endl;
            std::cout << "================================================================\n" << std::endl;
        }

        return env.Finalize();
    });
}

template bool RunScenarioDCrossRank<float, 262144>(int, int);
template bool RunScenarioDCrossRank<float, 4194304>(int, int);
template bool RunScenarioDCrossRank<float, 33554432>(int, int);
