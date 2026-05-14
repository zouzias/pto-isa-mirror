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
// Cross-rank ST for pto::TPREFETCH_L2 over HCCL.
//
// All kernels in this file are multi-rank by construction (they need
// HcclDeviceContext / TPUT_ASYNC / TGET / HcclRemotePtr). Single-card
// coverage lives under tests/npu/a2a3/src/st/testcase/tprefetch_l2/.
//
// Even though TPREFETCH_L2 itself takes a `__gm__ uint8_t *workspace`
// (workspace overload), this file still uses the explicit AsyncSession
// overload because every kernel below also issues TPUT_ASYNC / TPUT
// against the same workspace. Building one session and reusing it for
// both TPREFETCH_L2 and TPUT_ASYNC saves the per-call session-build cost
// (a transient 256B scratch tile + BuildAsyncSession round trip) which
// would otherwise show up in the perf numbers.
// ============================================================================

#include <cstddef>
#include <cstdint>
#include <iostream>

#include <pto/pto-inst.hpp>
#include "pto/npu/comm/async/sdma/sdma_types.hpp"
#include "pto/npu/comm/async/sdma/sdma_workspace_manager.hpp"
#include "pto/common/pto_tile.hpp"
#include "../common.hpp"

#define ENABLE_DEBUG_PRINT 1

using SdmaWorkspaceManager = pto::comm::sdma::SdmaWorkspaceManager;

// ============================================================================
// Kernel-wide type aliases — every TPREFETCH_L2 / TPUT / TGET kernel below
// uses the same fully-dynamic 5-D Shape/Stride + a uint8 ScratchTile for
// building AsyncSession.
// ============================================================================
using KernelShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
using KernelStrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
template <typename T>
using KernelGlobal = pto::GlobalTensor<T, KernelShapeDyn, KernelStrideDyn, pto::Layout::ND>;
using KernelScratchTile = pto::Tile<pto::TileType::Vec, uint8_t, 1, pto::comm::sdma::UB_ALIGN_SIZE>;

// ============================================================================
// Device-side helpers shared by multiple __global__ AICORE kernels.
//
//   * BuildKernelSession    — allocate ScratchTile + BuildAsyncSession.
//   * PrefetchRealOrTrash   — issue TPREFETCH_L2 against real or trash buffer
//                             (keeps SDMA CMO overhead identical between warm
//                             and cold measurements).
//   * BroadcastViaTputAsync — fan-out a sendG to every peer via TPUT_ASYNC,
//                             returning after Waiting on the last event.
// ============================================================================

PTO_INTERNAL bool BuildKernelSession(pto::comm::AsyncSession &session, __gm__ uint8_t *sdmaWorkspace,
                                     uint32_t sdmaSyncId)
{
    KernelScratchTile scratchTile;
    TASSIGN(scratchTile, 0x0);
    return pto::comm::BuildAsyncSession(scratchTile, sdmaWorkspace, session, sdmaSyncId);
}

template <typename T>
PTO_INTERNAL pto::comm::AsyncEvent PrefetchRealOrTrash(__gm__ T *realBuf, __gm__ uint8_t *trashBuf, int elem_count,
                                                       int enablePrefetch, pto::comm::AsyncSession &session)
{
    uint64_t totalBytes = static_cast<uint64_t>(elem_count) * sizeof(T);
    __gm__ void *target =
        enablePrefetch ? reinterpret_cast<__gm__ void *>(realBuf) : reinterpret_cast<__gm__ void *>(trashBuf);
    return pto::TPREFETCH_L2(target, totalBytes, session);
}

template <typename T>
PTO_INTERNAL void BroadcastViaTputAsync(KernelGlobal<T> &sendG, __gm__ T *recvBuf, const KernelShapeDyn &shape,
                                        const KernelStrideDyn &stride, int nranks, int self_rank,
                                        __gm__ HcclDeviceContext *hcclCtx, pto::comm::AsyncSession &session)
{
    pto::comm::AsyncEvent lastEvent;
    for (int target = 0; target < nranks; ++target) {
        if (target == self_rank) {
            continue;
        }
        __gm__ T *remoteRecv = HcclRemotePtr(hcclCtx, recvBuf, target);
        KernelGlobal<T> remoteRecvG(remoteRecv, shape, stride);
        lastEvent = pto::comm::TPUT_ASYNC(remoteRecvG, sendG, session);
    }
    (void)lastEvent.Wait(session);
}

// BoundsOkOrFinalize — range guard for elem_count against the compile-time
// `count` template parameter. On failure, emits the closing pipe_barrier and
// returns false so the kernel can early-return in one line instead of four.
template <size_t count>
PTO_INTERNAL bool BoundsOkOrFinalize(int elem_count)
{
    if (elem_count <= 0 || elem_count > static_cast<int>(count)) {
        pipe_barrier(PIPE_ALL);
        return false;
    }
    return true;
}

// RootBroadcastSetup — bundled setup state for "root-only TPREFETCH + broadcast"
// kernels (TPrefetchL2TputAsyncKernel / TPrefetchL2PerfKernel). Using one named
// struct keeps the kernel body focused on what's different (prefetch policy /
// timing) rather than re-stating the shared plumbing.
template <typename T, size_t count>
struct RootBroadcastSetup {
    pto::comm::AsyncSession session;
    KernelShapeDyn shape;
    KernelStrideDyn stride;
    __gm__ T *sendBuf = nullptr;
    __gm__ T *recvBuf = nullptr;
};

// EnterRootBroadcastOrReturn — unified gate for "root-only TPREFETCH + broadcast"
// kernels. Performs bounds check, root-rank gate, session build, and
// buffer/shape setup in one call. On any failure emits pipe_barrier(PIPE_ALL)
// and returns false so the caller can `return` immediately. On success, `s`
// is fully populated and the caller is guaranteed to be on the root rank.
template <typename T, size_t count>
PTO_INTERNAL bool EnterRootBroadcastOrReturn(__gm__ T *commBuf, __gm__ HcclDeviceContext *hcclCtx, int root_rank,
                                             int elem_count, __gm__ uint8_t *sdmaWorkspace, uint32_t sdmaSyncId,
                                             RootBroadcastSetup<T, count> &s)
{
    if (!BoundsOkOrFinalize<count>(elem_count)) {
        return false;
    }
    if (static_cast<int>(hcclCtx->rankId) != root_rank) {
        pipe_barrier(PIPE_ALL);
        return false;
    }
    if (!BuildKernelSession(s.session, sdmaWorkspace, sdmaSyncId)) {
        pipe_barrier(PIPE_ALL);
        return false;
    }
    s.shape = KernelShapeDyn(1, 1, 1, 1, elem_count);
    s.stride = KernelStrideDyn(elem_count, elem_count, elem_count, elem_count, 1);
    s.sendBuf = commBuf;
    s.recvBuf = commBuf + count;
    return true;
}

// ----------------------------------------------------------------------------
// Perf-kernel tile tiling — file-scope aliases shared by all the multi-rank
// performance kernels (TloadRemotePerfKernel, TputSyncPerfKernel,
// TgetPerfKernel). These wrap the "chunk `count` elements into tiles of at
// most 256 columns" convention so each kernel body doesn't repeat the
// kTileCols/TileData alias block.
// ----------------------------------------------------------------------------
template <size_t count>
constexpr int kPerfTileCols = (count <= 256) ? static_cast<int>(count) : 256;

template <typename T, size_t count>
using PerfTileData = pto::Tile<pto::TileType::Vec, T, 1, kPerfTileCols<count>, pto::BLayout::RowMajor>;

// PerfKernelSetup — bundled "my_rank + commBuf split (sendBuf/recvBuf)" that
// every perf kernel needs before branching on rank role. Avoids repeating the
// same 3-line triple across TputSync / Tget / TloadRemote kernels.
template <typename T, size_t count>
struct PerfKernelSetup {
    int my_rank = 0;
    __gm__ T *sendBuf = nullptr;
    __gm__ T *recvBuf = nullptr;
};

template <typename T, size_t count>
PTO_INTERNAL bool SetupPerfKernelOrReturn(__gm__ T *commBuf, int elem_count, __gm__ HcclDeviceContext *hcclCtx,
                                          PerfKernelSetup<T, count> &s)
{
    if (!BoundsOkOrFinalize<count>(elem_count)) {
        return false;
    }
    s.my_rank = static_cast<int>(hcclCtx->rankId);
    s.sendBuf = commBuf;
    s.recvBuf = commBuf + count;
    return true;
}

// ----------------------------------------------------------------------------
// MultiRankPerfEnv: test-file-scoped fixture for all multi-rank TPREFETCH_L2
// tests (both the correctness test RunPrefetchL2TputAsyncKernel and the perf
// tests RunPrefetchL2PerfKernel / RunTloadRemotePerfKernel / RunTputSyncPerf /
// RunTgetPerfKernel).
//
// Bundles:
//   * TestContext ctx      -- HCCL communicator + ACL stream + deviceCtx
//   * input_host           -- host-side input buffer, pre-filled with {0..count-1}
//   * sendBuf / recvBuf    -- carved from the local HCCL window
//   * cycleDev / trashDev  -- perf measurement + L2-cold "trash" buffer
//                             (allocated unconditionally; correctness tests
//                              leave them unused — cost is one uint64 + one
//                              count*sizeof(T) allocation on the device)
//   * sdmaMgr              -- SDMA workspace manager
//
// The struct intentionally does NOT call HcclHostBarrier anywhere. Collective
// synchronization is a test-flow concern and belongs at the call site so the
// reader can see the exact barrier count matching the kernel-launch pattern.
// ----------------------------------------------------------------------------
template <typename T, size_t count>
struct MultiRankPerfEnv {
    TestContext ctx;
    uint8_t *input_host = nullptr;
    T *sendBuf = nullptr;
    T *recvBuf = nullptr;
    void *cycleDev = nullptr;
    void *trashDev = nullptr;
    SdmaWorkspaceManager sdmaMgr;

    bool Init(int rank_id, int n_ranks, int n_devices, int first_device_id, const HcclRootInfo *rootInfo)
    {
        if (!ctx.Init(rank_id, n_ranks, n_devices, first_device_id, rootInfo))
            return false;
        if (aclrtMallocHost(reinterpret_cast<void **>(&input_host), count * sizeof(T)) != 0) {
            std::cerr << "[ERROR] PerfEnv: aclrtMallocHost failed!" << std::endl;
            return false;
        }
        for (size_t i = 0; i < count; ++i)
            reinterpret_cast<T *>(input_host)[i] = static_cast<T>(i);

        uint64_t localWinBase = ctx.hostCtx.windowsIn[rank_id];
        size_t winOffset = 0;
        void *commBufPtr = WindowAlloc(localWinBase, winOffset, 64 * sizeof(int32_t) + 2 * count * sizeof(T));
        uint8_t *commBytes = reinterpret_cast<uint8_t *>(commBufPtr);
        sendBuf = reinterpret_cast<T *>(commBytes + 64 * sizeof(int32_t));
        recvBuf = sendBuf + count;

        aclrtMalloc(&trashDev, count * sizeof(T), ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc(&cycleDev, sizeof(uint64_t), ACL_MEM_MALLOC_HUGE_FIRST);
        uint64_t zero = 0;
        aclrtMemcpy(cycleDev, sizeof(uint64_t), &zero, sizeof(uint64_t), ACL_MEMCPY_HOST_TO_DEVICE);

        if (!sdmaMgr.Init()) {
            std::cerr << "[ERROR] SdmaWorkspaceManager Init failed!" << std::endl;
            aclrtFree(cycleDev);
            aclrtFree(trashDev);
            aclrtFreeHost(input_host);
            return false;
        }
        return true;
    }

    void UploadInputToSend()
    {
        aclrtMemcpy(sendBuf, count * sizeof(T), input_host, count * sizeof(T), ACL_MEMCPY_HOST_TO_DEVICE);
    }

    void DownloadCycles(uint64_t &out, bool myRole)
    {
        uint64_t cycles = 0;
        if (myRole) {
            aclrtMemcpy(&cycles, sizeof(uint64_t), cycleDev, sizeof(uint64_t), ACL_MEMCPY_DEVICE_TO_HOST);
        }
        out = cycles;
    }

    bool Finalize()
    {
        aclrtFree(cycleDev);
        aclrtFree(trashDev);
        aclrtFreeHost(input_host);
        sdmaMgr.Finalize();
        return ctx.Finalize();
    }
};

// ----------------------------------------------------------------------------
// LaunchMultiRankKernel: bracket a kernel launch with pre/post collective
// barriers and a host-side stream sync. `launchFn` is a no-arg callable that
// performs the actual <<<...>>> kernel launch; this lets each caller retain
// its specific argument list (kernel-launch syntax cannot take a function
// pointer in CCE) while the barrier + sync boilerplate is centralized here.
// ----------------------------------------------------------------------------
template <typename T, size_t count, typename LaunchFn>
void LaunchMultiRankKernel(MultiRankPerfEnv<T, count> &env, LaunchFn &&launchFn)
{
    HcclHostBarrier(env.ctx.comm, env.ctx.stream);
    launchFn();
    env.ctx.aclStatus = aclrtSynchronizeStream(env.ctx.stream);
    HcclHostBarrier(env.ctx.comm, env.ctx.stream);
}

// Whether every rank uploads its own input (All) or only the root rank does
// (RootOnly). Different perf tests have different semantics: TLOAD-based tests
// only need the root to populate its send buffer, while the TPUT_ASYNC perf
// test uploads from all ranks so each can independently verify its output.
enum class PerfUploadMode
{
    All,
    RootOnly
};

// ----------------------------------------------------------------------------
// RunMultiRankPerfKernelGeneric — shared driver for single-iteration multi-rank
// performance kernels. All of RunPrefetchL2PerfKernel / RunTloadRemotePerfKernel
// / RunTputSyncPerfKernel follow the same shape (Init env → optional Upload →
// LaunchMultiRankKernel → DownloadCycles from the measuring rank → Finalize);
// they only differ in (a) the kernel being launched, (b) whether upload is
// root-only, and (c) which rank holds the measured cycles. Factoring this
// driver out removes the ~10-line boilerplate that otherwise repeats verbatim
// across those three wrappers.
// ----------------------------------------------------------------------------
template <typename T, size_t count, typename DeviceLaunchFn>
bool RunMultiRankPerfKernelGeneric(int rank_id, int n_ranks, int n_devices, int first_device_id,
                                   const HcclRootInfo *rootInfo, int root_rank, bool prefetch, uint64_t &outCycles,
                                   PerfUploadMode uploadMode, bool cyclesFromRoot, DeviceLaunchFn &&deviceLaunchFn)
{
    MultiRankPerfEnv<T, count> env;
    if (!env.Init(rank_id, n_ranks, n_devices, first_device_id, rootInfo)) {
        return false;
    }
    if (uploadMode == PerfUploadMode::All || rank_id == root_rank) {
        env.UploadInputToSend();
    }

    int enablePrefetch = prefetch ? 1 : 0;
    LaunchMultiRankKernel(env, [&] { deviceLaunchFn(env, n_ranks, root_rank, enablePrefetch); });

    const bool cyclesOnMe = cyclesFromRoot ? (rank_id == root_rank) : (rank_id != root_rank);
    env.DownloadCycles(outCycles, cyclesOnMe);
    return env.Finalize();
}

// ----------------------------------------------------------------------------
// PrintL2ColdWarmSummary: shared printer for all "L2-cold vs L2-warm" perf tests.
//
// Remote-TLOAD / TPUT-sync / TGET all report the same block structure
// (title → data size → chunk line → iterations → cold cycles → warm cycles →
// speedup verdict). The only variations are the test title, the measured
// instruction name ("TLOAD"/"TPUT"/"TGET"), and the chunk-line style.
// Callers fill in a small spec struct and we render the common scaffold.
// ----------------------------------------------------------------------------
struct L2PerfSummaryFmt {
    const char *title;
    const char *metric;
    size_t dataBytes;
    int kWarmup;
    int kMeasured;
    // Exactly one of the following two is used (the other left at 0):
    //   tloadChunks    — prints "  TLOAD chunks:   <tloadChunks>"
    //   tileChunkElems — prints "  Tile chunk:     <tileChunkElems> elements"
    int tloadChunks;
    int tileChunkElems;
};

inline void PrintL2ColdWarmSummary(const L2PerfSummaryFmt &fmt, uint64_t avgCold, uint64_t avgWarm)
{
    double speedup = (avgWarm > 0) ? static_cast<double>(avgCold) / static_cast<double>(avgWarm) : 0.0;

    std::cout << "\n================================================================" << std::endl;
    std::cout << "[PERF] " << fmt.title << std::endl;
    std::cout << "  Data size:      " << fmt.dataBytes << " bytes (" << fmt.dataBytes / 1024 << " KB)" << std::endl;
    if (fmt.tloadChunks > 0) {
        std::cout << "  TLOAD chunks:   " << fmt.tloadChunks << std::endl;
    } else if (fmt.tileChunkElems > 0) {
        std::cout << "  Tile chunk:     " << fmt.tileChunkElems << " elements" << std::endl;
    }
    std::cout << "  Iterations:     " << fmt.kMeasured << " (warmup=" << fmt.kWarmup << ")" << std::endl;
    std::cout << "  L2-cold " << fmt.metric << ":  " << avgCold << " cycles (avg)" << std::endl;
    std::cout << "  L2-warm " << fmt.metric << ":  " << avgWarm << " cycles (avg)" << std::endl;
    std::cout << "  Speedup:        " << speedup << "x  ("
              << (speedup > 1.05 ? "PREFETCH HELPS" : (speedup < 0.95 ? "PREFETCH HURTS" : "NO SIGNIFICANT DIFF"))
              << ")" << std::endl;
    std::cout << "================================================================\n" << std::endl;
}

// ----------------------------------------------------------------------------
// RunColdWarmPerfSweep: generic "L2 cold vs L2 warm" benchmark driver.
//
// Runs `kWarmup + kMeasured` iterations of `runOnce(false, cycles)` (cold),
// then another `kWarmup + kMeasured` of `runOnce(true, cycles)` (warm),
// accumulating measurement-phase cycles into coldTotal/warmTotal. The
// caller-provided `printSummary` lambda receives the two averages and is
// responsible for deciding WHICH rank prints (the measuring rank differs
// per test: sender for TPUT-style, receiver for TLOAD-style).
// ----------------------------------------------------------------------------
template <typename RunOnce, typename PrintSummary>
bool RunColdWarmPerfSweep(int kWarmup, int kMeasured, RunOnce &&runOnce, PrintSummary &&printSummary)
{
    if (kMeasured <= 0) {
        return false;
    }

    uint64_t coldTotal = 0;
    uint64_t warmTotal = 0;

    for (int iter = 0; iter < kWarmup + kMeasured; ++iter) {
        uint64_t cycles = 0;
        if (!runOnce(false, cycles)) {
            return false;
        }
        if (iter >= kWarmup) {
            coldTotal += cycles;
        }
    }
    for (int iter = 0; iter < kWarmup + kMeasured; ++iter) {
        uint64_t cycles = 0;
        if (!runOnce(true, cycles)) {
            return false;
        }
        if (iter >= kWarmup) {
            warmTotal += cycles;
        }
    }

    uint64_t avgCold = coldTotal / static_cast<uint64_t>(kMeasured);
    uint64_t avgWarm = warmTotal / static_cast<uint64_t>(kMeasured);
    printSummary(avgCold, avgWarm);
    return true;
}

// ============================================================================
// Multi-card: TPREFETCH_L2 + TPUT_ASYNC Kernel
//
// Rank root_rank:
//   1. Prefetch src data: GM → L2  (SDMA CMO, async, no intermediate wait)
//   2. TPUT_ASYNC: GM(L2-hot) → remote GM (SDMA memcpy, queued after CMO SQEs)
//   3. Single Wait at end (SDMA FIFO guarantees CMO completes before memcpy)
// Other ranks: idle, receive via HCCL window.
//
// NOTE: All SQEs (CMO + memcpy + Flag) share the same SDMA queue.
//       FIFO ordering ensures prefetch completes before the put reads the data.
//       An intermediate Wait() between prefetch and put is unnecessary and can
//       cause stale SQ-tail issues on some hardware revisions.
// ============================================================================
template <typename T, size_t count>
__global__ AICORE void TPrefetchL2TputAsyncKernel(__gm__ T *commBuf, int nranks, int root_rank, int elem_count,
                                                  int enablePrefetch, __gm__ HcclDeviceContext *hcclCtx,
                                                  __gm__ uint8_t *sdmaWorkspace, uint32_t sdmaSyncId)
{
    RootBroadcastSetup<T, count> s;
    if (!EnterRootBroadcastOrReturn<T, count>(commBuf, hcclCtx, root_rank, elem_count, sdmaWorkspace, sdmaSyncId, s)) {
        return;
    }
    KernelGlobal<T> sendG(s.sendBuf, s.shape, s.stride);

    if (enablePrefetch) {
        (void)pto::TPREFETCH_L2(sendG, s.session);
    }

    BroadcastViaTputAsync<T>(sendG, s.recvBuf, s.shape, s.stride, nranks, root_rank, hcclCtx, s.session);

    pipe_barrier(PIPE_ALL);
}

// ============================================================================
// Host-side runner: TPUT_ASYNC with optional TPREFETCH_L2 (multi-rank via HCCL)
// ============================================================================
template <typename T>
inline bool VerifyTputAsyncOutput(int rank_id, int root_rank, const uint8_t *output_host, size_t count, bool prefetch)
{
    if (rank_id == root_rank) {
        return true;
    }
    const T *out = reinterpret_cast<const T *>(output_host);
    for (size_t i = 0; i < count; ++i) {
        T expected = static_cast<T>(i + root_rank * 10000);
        if (out[i] != expected) {
            std::cout << "Rank " << rank_id << " idx " << i << " expected " << (float)expected << " got "
                      << (float)out[i] << std::endl;
            return false;
        }
    }
#if ENABLE_DEBUG_PRINT
    const char *mode = prefetch ? "TPREFETCH_L2 + TPUT_ASYNC" : "TPUT_ASYNC only (no prefetch)";
    std::cout << "\n================================================================" << std::endl;
    std::cout << "[DEBUG] Rank " << rank_id << ": " << mode << " SUCCESSFUL!" << std::endl;
    std::cout << "  count=" << count << ", dtype_size=" << sizeof(T) << std::endl;
    std::cout << "  Sample: [ ";
    for (size_t i = 0; i < (count > 5 ? 5 : count); ++i)
        std::cout << (float)out[i] << " ";
    if (count > 5)
        std::cout << "... ";
    std::cout << "]" << std::endl;
    std::cout << "================================================================\n" << std::endl;
#else
    (void)prefetch;
#endif
    return true;
}

template <typename T, size_t count>
bool RunPrefetchL2TputAsyncKernel(int rank_id, int n_ranks, int n_devices, int first_device_id,
                                  const HcclRootInfo *rootInfo, int root_rank, bool prefetch)
{
    MultiRankPerfEnv<T, count> env;
    if (!env.Init(rank_id, n_ranks, n_devices, first_device_id, rootInfo))
        return false;

    for (size_t i = 0; i < count; ++i)
        reinterpret_cast<T *>(env.input_host)[i] = static_cast<T>(i + rank_id * 10000);
    env.UploadInputToSend();

    uint8_t *output_host = nullptr;
    if (aclrtMallocHost(reinterpret_cast<void **>(&output_host), count * sizeof(T)) != 0) {
        std::cerr << "[ERROR] TputAsync: output host alloc failed!" << std::endl;
        env.Finalize();
        return false;
    }
    for (size_t i = 0; i < count; ++i)
        reinterpret_cast<T *>(output_host)[i] = static_cast<T>(-1);
    aclrtMemcpy(env.recvBuf, count * sizeof(T), output_host, count * sizeof(T), ACL_MEMCPY_HOST_TO_DEVICE);

    int enablePrefetch = prefetch ? 1 : 0;
    LaunchMultiRankKernel(env, [&] {
        TPrefetchL2TputAsyncKernel<T, count>
            <<<1, nullptr, env.ctx.stream>>>(env.sendBuf, n_ranks, root_rank, static_cast<int>(count), enablePrefetch,
                                             env.ctx.deviceCtx, (uint8_t *)env.sdmaMgr.GetWorkspaceAddr(), 0);
    });

    aclrtMemcpy(output_host, count * sizeof(T), env.recvBuf, count * sizeof(T), ACL_MEMCPY_DEVICE_TO_HOST);
    bool is_ok = VerifyTputAsyncOutput<T>(rank_id, root_rank, output_host, count, prefetch);
    aclrtFreeHost(output_host);
    return env.Finalize() && is_ok;
}

template <typename T, size_t count>
bool RunPrefetchL2TputAsync(int n_ranks, int n_devices, int first_rank_id, int first_device_id, bool prefetch)
{
    const int root_rank = first_rank_id;
    return ForkAndRunWithHcclRootInfo(
        n_ranks, first_rank_id, first_device_id, [&](int rankId, const HcclRootInfo *rootInfo) {
            return RunPrefetchL2TputAsyncKernel<T, count>(rankId, n_ranks, n_devices, first_device_id, rootInfo,
                                                          root_rank, prefetch);
        });
}

template bool RunPrefetchL2TputAsync<float, 4096>(int, int, int, int, bool);
template bool RunPrefetchL2TputAsync<int32_t, 4096>(int, int, int, int, bool);

// ============================================================================
// Cycle counter helper (volatile to prevent compiler reordering)
// ============================================================================
inline AICORE uint64_t get_syscnt()
{
    uint64_t syscnt;
    asm volatile("MOV %0, SYS_CNT\n" : "+l"(syscnt));
    return syscnt;
}

// ============================================================================
// Performance Kernel: measure TPUT_ASYNC cycles with/without TPREFETCH_L2
//
// Uses trash-buffer technique for L2 cold control (cf. shmem/CMO example).
// Both paths always execute a SDMA CMO prefetch for fair comparison:
//   enablePrefetch=1: prefetch sendBuf → L2 warm
//   enablePrefetch=0: prefetch trashBuf → L2 cold for sendBuf
// ============================================================================
template <typename T, size_t count>
__global__ AICORE void TPrefetchL2PerfKernel(__gm__ T *commBuf, int nranks, int root_rank, int elem_count,
                                             int enablePrefetch, __gm__ HcclDeviceContext *hcclCtx,
                                             __gm__ uint8_t *sdmaWorkspace, uint32_t sdmaSyncId,
                                             __gm__ uint8_t *trashBuf, __gm__ uint64_t *cycleOut)
{
    RootBroadcastSetup<T, count> s;
    if (!EnterRootBroadcastOrReturn<T, count>(commBuf, hcclCtx, root_rank, elem_count, sdmaWorkspace, sdmaSyncId, s)) {
        return;
    }
    KernelGlobal<T> sendG(s.sendBuf, s.shape, s.stride);

    // Prefetch hint is fire-and-forget here: TPUT_ASYNC below may overlap with it.
    (void)PrefetchRealOrTrash<T>(s.sendBuf, trashBuf, elem_count, enablePrefetch, s.session);

    uint64_t t0 = get_syscnt();
    BroadcastViaTputAsync<T>(sendG, s.recvBuf, s.shape, s.stride, nranks, root_rank, hcclCtx, s.session);
    uint64_t t1 = get_syscnt();
    *cycleOut = t1 - t0;

    pipe_barrier(PIPE_ALL);
}

// ============================================================================
// Host-side perf runner: single iteration, returns cycle count
// ============================================================================
template <typename T, size_t count>
bool RunPrefetchL2PerfKernel(int rank_id, int n_ranks, int n_devices, int first_device_id, const HcclRootInfo *rootInfo,
                             int root_rank, bool prefetch, uint64_t &outCycles)
{
    return RunMultiRankPerfKernelGeneric<T, count>(
        rank_id, n_ranks, n_devices, first_device_id, rootInfo, root_rank, prefetch, outCycles, PerfUploadMode::All,
        /*cyclesFromRoot=*/true, [](MultiRankPerfEnv<T, count> &env, int nRanks, int rootRank, int enablePrefetch) {
            TPrefetchL2PerfKernel<T, count><<<1, nullptr, env.ctx.stream>>>(
                env.sendBuf, nRanks, rootRank, static_cast<int>(count), enablePrefetch, env.ctx.deviceCtx,
                (uint8_t *)env.sdmaMgr.GetWorkspaceAddr(), 0, reinterpret_cast<uint8_t *>(env.trashDev),
                reinterpret_cast<uint64_t *>(env.cycleDev));
        });
}

// ============================================================================
// Top-level perf comparison: runs both modes, prints results (rank 0 only)
// ============================================================================
template <typename T, size_t count>
bool RunPrefetchL2Perf(int n_ranks, int n_devices, int first_rank_id, int first_device_id)
{
    const int root_rank = first_rank_id;
    constexpr int kWarmup = 2;
    constexpr int kMeasured = 5;

    auto runOnce = [&](bool prefetch, uint64_t &cycles) {
        return ForkAndRunWithHcclRootInfo(
            n_ranks, first_rank_id, first_device_id, [&](int rankId, const HcclRootInfo *rootInfo) {
                return RunPrefetchL2PerfKernel<T, count>(rankId, n_ranks, n_devices, first_device_id, rootInfo,
                                                         root_rank, prefetch, cycles);
            });
    };
    auto printSummary = [&](uint64_t avgCold, uint64_t avgWarm) {
        if (CommMpiRank() != 0)
            return;
        double ratio = (avgCold > 0) ? (double)avgWarm / (double)avgCold : 0.0;
        size_t dataBytes = count * sizeof(T);

        std::cout << "\n================================================================" << std::endl;
        std::cout << "[PERF] TPUT_ASYNC Performance Comparison" << std::endl;
        std::cout << "  Data size:     " << dataBytes << " bytes (" << dataBytes / 1024 << " KB)" << std::endl;
        std::cout << "  Iterations:    " << kMeasured << " (warmup=" << kWarmup << ")" << std::endl;
        std::cout << "  No prefetch:   " << avgCold << " cycles (avg)" << std::endl;
        std::cout << "  With prefetch: " << avgWarm << " cycles (avg)" << std::endl;
        std::cout << "  Ratio:         " << ratio << "x  ("
                  << (ratio < 1.0 ? "PREFETCH FASTER" : (ratio > 1.0 ? "NO PREFETCH FASTER" : "EQUAL")) << ")"
                  << std::endl;
        std::cout << "================================================================\n" << std::endl;
    };
    return RunColdWarmPerfSweep(kWarmup, kMeasured, runOnce, printSummary);
}

template bool RunPrefetchL2Perf<float, 4096>(int, int, int, int);
template bool RunPrefetchL2Perf<float, 65536>(int, int, int, int);
template bool RunPrefetchL2Perf<float, 262144>(int, int, int, int);

// ============================================================================
// Multi-rank TLOAD Perf Kernel
//
// Rank 0 (sender):  TPUT_ASYNC → send data to Rank 1
// Rank 1 (receiver): prefetch (real or trash) → TLOAD loop → cycles
//
// Uses trash-buffer technique for L2 cold control (cf. shmem/CMO example).
// Both paths always execute a SDMA CMO prefetch for fair comparison.
// ============================================================================
// Phase 1 helper: root rank pushes sendBuf to all other ranks via TPUT_ASYNC.
template <typename T, size_t count>
PTO_INTERNAL void RemotePerfSendPhase(__gm__ T *sendBuf, __gm__ T *recvBuf, int elem_count, int nranks, int root_rank,
                                      __gm__ HcclDeviceContext *hcclCtx, pto::comm::AsyncSession &session)
{
    KernelShapeDyn shape(1, 1, 1, 1, elem_count);
    KernelStrideDyn stride(elem_count, elem_count, elem_count, elem_count, 1);
    KernelGlobal<T> sendG(sendBuf, shape, stride);

    BroadcastViaTputAsync<T>(sendG, recvBuf, shape, stride, nranks, root_rank, hcclCtx, session);
}

// Phase 2 helper: receiver optionally prefetches, then measures TLOAD loop cycles.
template <typename T, size_t count>
PTO_INTERNAL void RemotePerfTloadPhase(__gm__ T *recvBuf, int elem_count, int enablePrefetch, bool sessionOk,
                                       __gm__ uint8_t *trashBuf, pto::comm::AsyncSession &session,
                                       __gm__ uint64_t *cycleOut)
{
    constexpr int kTileCols = (count <= 256) ? static_cast<int>(count) : 256;
    using TileData = pto::Tile<pto::TileType::Vec, T, 1, kTileCols, pto::BLayout::RowMajor>;
    using ChunkShape = pto::Shape<1, 1, 1, 1, kTileCols>;
    using ChunkStride = pto::Stride<1, 1, 1, 1, 1>;

    if (sessionOk) {
        auto evt = PrefetchRealOrTrash<T>(recvBuf, trashBuf, elem_count, enablePrefetch, session);
        (void)evt.Wait(session);
    }

    TileData tile;
    TASSIGN(tile, 0x0);
    uint64_t t0 = get_syscnt();
    for (int offset = 0; offset < elem_count; offset += kTileCols) {
        pto::GlobalTensor<T, ChunkShape, ChunkStride, pto::Layout::ND> chunk(recvBuf + offset);
        TLOAD(tile, chunk);
        set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    }
    uint64_t t1 = get_syscnt();
    *cycleOut = t1 - t0;
}

template <typename T, size_t count>
__global__ AICORE void TloadRemotePerfKernel(__gm__ T *commBuf, int nranks, int root_rank, int elem_count,
                                             int enablePrefetch, __gm__ HcclDeviceContext *hcclCtx,
                                             __gm__ uint8_t *sdmaWorkspace, uint32_t sdmaSyncId,
                                             __gm__ uint8_t *trashBuf, __gm__ uint64_t *cycleOut)
{
    PerfKernelSetup<T, count> ctx;
    if (!SetupPerfKernelOrReturn<T, count>(commBuf, elem_count, hcclCtx, ctx)) {
        return;
    }

    pto::comm::AsyncSession session;
    bool sessionOk = BuildKernelSession(session, sdmaWorkspace, sdmaSyncId);

    if (ctx.my_rank == root_rank && sessionOk) {
        RemotePerfSendPhase<T, count>(ctx.sendBuf, ctx.recvBuf, elem_count, nranks, root_rank, hcclCtx, session);
    }
    pipe_barrier(PIPE_ALL);

    if (ctx.my_rank != root_rank) {
        RemotePerfTloadPhase<T, count>(ctx.recvBuf, elem_count, enablePrefetch, sessionOk, trashBuf, session, cycleOut);
    }
    pipe_barrier(PIPE_ALL);
}

// ============================================================================
// Host-side runner for multi-rank TLOAD perf (single iteration)
// ============================================================================
template <typename T, size_t count>
bool RunTloadRemotePerfKernel(int rank_id, int n_ranks, int n_devices, int first_device_id,
                              const HcclRootInfo *rootInfo, int root_rank, bool prefetch, uint64_t &outCycles)
{
    return RunMultiRankPerfKernelGeneric<T, count>(
        rank_id, n_ranks, n_devices, first_device_id, rootInfo, root_rank, prefetch, outCycles,
        PerfUploadMode::RootOnly, /*cyclesFromRoot=*/false,
        [](MultiRankPerfEnv<T, count> &env, int nRanks, int rootRank, int enablePrefetch) {
            TloadRemotePerfKernel<T, count><<<1, nullptr, env.ctx.stream>>>(
                env.sendBuf, nRanks, rootRank, static_cast<int>(count), enablePrefetch, env.ctx.deviceCtx,
                (uint8_t *)env.sdmaMgr.GetWorkspaceAddr(), 0, reinterpret_cast<uint8_t *>(env.trashDev),
                reinterpret_cast<uint64_t *>(env.cycleDev));
        });
}

// ============================================================================
// Top-level multi-rank TLOAD perf comparison
// ============================================================================
template <typename T, size_t count>
bool RunTloadRemotePerf(int n_ranks, int n_devices, int first_rank_id, int first_device_id)
{
    const int root_rank = first_rank_id;
    constexpr int kWarmup = 2;
    constexpr int kMeasured = 5;

    auto runOnce = [&](bool prefetch, uint64_t &cycles) {
        return ForkAndRunWithHcclRootInfo(
            n_ranks, first_rank_id, first_device_id, [&](int rankId, const HcclRootInfo *rootInfo) {
                return RunTloadRemotePerfKernel<T, count>(rankId, n_ranks, n_devices, first_device_id, rootInfo,
                                                          root_rank, prefetch, cycles);
            });
    };
    // Measurement lives on the receiver (rank != root), so print from non-rank-0.
    auto printSummary = [&](uint64_t avgCold, uint64_t avgWarm) {
        if (CommMpiRank() == 0)
            return;
        L2PerfSummaryFmt fmt{};
        fmt.title = "Remote TLOAD: Rank 0 → Rank 1 → TLOAD";
        fmt.metric = "TLOAD";
        fmt.dataBytes = count * sizeof(T);
        fmt.kWarmup = kWarmup;
        fmt.kMeasured = kMeasured;
        fmt.tloadChunks = static_cast<int>(count / ((count <= 256) ? count : 256));
        PrintL2ColdWarmSummary(fmt, avgCold, avgWarm);
    };
    return RunColdWarmPerfSweep(kWarmup, kMeasured, runOnce, printSummary);
}

template bool RunTloadRemotePerf<float, 4096>(int, int, int, int);
template bool RunTloadRemotePerf<float, 65536>(int, int, int, int);
template bool RunTloadRemotePerf<float, 262144>(int, int, int, int);

// ============================================================================
// TPUT (sync) Perf Kernel — practical pattern: prefetch before TPUT
//
// Uses trash-buffer technique for L2 cold control (cf. shmem/CMO example).
// Both paths always execute a SDMA CMO prefetch for fair comparison:
//   enablePrefetch=1: prefetch sendBuf → L2 warm (accelerates TPUT internal TLOAD)
//   enablePrefetch=0: prefetch trashBuf → L2 cold for sendBuf
//
// Rank 0 (sender): prefetch(real or trash) → Wait → TPUT → measure cycles
// Rank 1 (receiver): idle
// ============================================================================
template <typename T, size_t count>
__global__ AICORE void TputSyncPerfKernel(__gm__ T *commBuf, int nranks, int root_rank, int elem_count,
                                          int enablePrefetch, __gm__ HcclDeviceContext *hcclCtx,
                                          __gm__ uint8_t *sdmaWorkspace, uint32_t sdmaSyncId, __gm__ uint8_t *trashBuf,
                                          __gm__ uint64_t *cycleOut)
{
    static_assert(count % kPerfTileCols<count> == 0, "count must be a multiple of kPerfTileCols");

    PerfKernelSetup<T, count> ctx;
    if (!SetupPerfKernelOrReturn<T, count>(commBuf, elem_count, hcclCtx, ctx)) {
        return;
    }

    if (ctx.my_rank == root_rank) {
        pto::comm::AsyncSession session;
        bool sessionOk = BuildKernelSession(session, sdmaWorkspace, sdmaSyncId);

        if (sessionOk) {
            auto evt = PrefetchRealOrTrash<T>(ctx.sendBuf, trashBuf, elem_count, enablePrefetch, session);
            (void)evt.Wait(session);
        }

        PerfTileData<T, count> stagingTile;
        TASSIGN(stagingTile, 0x0);

        KernelShapeDyn shape(1, 1, 1, 1, elem_count);
        KernelStrideDyn stride(elem_count, elem_count, elem_count, elem_count, 1);
        KernelGlobal<T> sendG(ctx.sendBuf, shape, stride);

        uint64_t t0 = get_syscnt();
        for (int target = 0; target < nranks; ++target) {
            if (target == root_rank)
                continue;
            __gm__ T *remoteRecvBuf = HcclRemotePtr(hcclCtx, ctx.recvBuf, target);
            KernelGlobal<T> remoteRecvG(remoteRecvBuf, shape, stride);
            pto::comm::TPUT(remoteRecvG, sendG, stagingTile);
        }
        pipe_barrier(PIPE_ALL);
        uint64_t t1 = get_syscnt();
        *cycleOut = t1 - t0;
    }

    pipe_barrier(PIPE_ALL);
}

template <typename T, size_t count>
bool RunTputSyncPerfKernel(int rank_id, int n_ranks, int n_devices, int first_device_id, const HcclRootInfo *rootInfo,
                           int root_rank, bool prefetch, uint64_t &outCycles)
{
    return RunMultiRankPerfKernelGeneric<T, count>(
        rank_id, n_ranks, n_devices, first_device_id, rootInfo, root_rank, prefetch, outCycles,
        PerfUploadMode::RootOnly, /*cyclesFromRoot=*/true,
        [](MultiRankPerfEnv<T, count> &env, int nRanks, int rootRank, int enablePrefetch) {
            TputSyncPerfKernel<T, count><<<1, nullptr, env.ctx.stream>>>(
                env.sendBuf, nRanks, rootRank, static_cast<int>(count), enablePrefetch, env.ctx.deviceCtx,
                (uint8_t *)env.sdmaMgr.GetWorkspaceAddr(), 0, reinterpret_cast<uint8_t *>(env.trashDev),
                reinterpret_cast<uint64_t *>(env.cycleDev));
        });
}

template <typename T, size_t count>
bool RunTputSyncPerf(int n_ranks, int n_devices, int first_rank_id, int first_device_id)
{
    const int root_rank = first_rank_id;
    constexpr int kWarmup = 2;
    constexpr int kMeasured = 5;

    auto runOnce = [&](bool prefetch, uint64_t &cycles) {
        return ForkAndRunWithHcclRootInfo(
            n_ranks, first_rank_id, first_device_id, [&](int rankId, const HcclRootInfo *rootInfo) {
                return RunTputSyncPerfKernel<T, count>(rankId, n_ranks, n_devices, first_device_id, rootInfo, root_rank,
                                                       prefetch, cycles);
            });
    };
    // Measurement lives on the sender (rank == root), which is mpiRank 0.
    auto printSummary = [&](uint64_t avgCold, uint64_t avgWarm) {
        if (CommMpiRank() != 0)
            return;
        L2PerfSummaryFmt fmt{};
        fmt.title = "TPUT (sync): sender-side L2 prefetch";
        fmt.metric = "TPUT";
        fmt.dataBytes = count * sizeof(T);
        fmt.kWarmup = kWarmup;
        fmt.kMeasured = kMeasured;
        fmt.tileChunkElems = static_cast<int>((count <= 256) ? count : 256);
        PrintL2ColdWarmSummary(fmt, avgCold, avgWarm);
    };
    return RunColdWarmPerfSweep(kWarmup, kMeasured, runOnce, printSummary);
}

template bool RunTputSyncPerf<float, 4096>(int, int, int, int);
template bool RunTputSyncPerf<float, 65536>(int, int, int, int);
template bool RunTputSyncPerf<float, 262144>(int, int, int, int);

// ============================================================================
// TGET Perf Kernel — practical pattern: remote prefetch before TGET
//
// Uses trash-buffer technique for L2 cold control (cf. shmem/CMO example).
// Both paths always execute a SDMA CMO prefetch for fair comparison:
//   enablePrefetch=1: source prefetches sendBuf → L2 warm (HCCS may serve from remote L2)
//   enablePrefetch=0: source prefetches trashBuf → L2 cold for sendBuf
//
// Two-phase (separate kernel launches with host barrier in between):
//   Phase 0 — Source rank: prefetch(real or trash) + Wait
//   Phase 1 — Reader rank: TGET(localRecv, remoteSend, tile) → measure cycles
// ============================================================================
template <typename T, size_t count>
__global__ AICORE void TgetPerfKernel(__gm__ T *commBuf, int nranks, int source_rank, int elem_count,
                                      int enablePrefetch, int phase, __gm__ HcclDeviceContext *hcclCtx,
                                      __gm__ uint8_t *sdmaWorkspace, uint32_t sdmaSyncId, __gm__ uint8_t *trashBuf,
                                      __gm__ uint64_t *cycleOut)
{
    static_assert(count % kPerfTileCols<count> == 0, "count must be a multiple of kPerfTileCols");

    PerfKernelSetup<T, count> ctx;
    if (!SetupPerfKernelOrReturn<T, count>(commBuf, elem_count, hcclCtx, ctx)) {
        return;
    }

    if (phase == 0) {
        if (ctx.my_rank == source_rank) {
            pto::comm::AsyncSession session;
            if (BuildKernelSession(session, sdmaWorkspace, sdmaSyncId)) {
                auto evt = PrefetchRealOrTrash<T>(ctx.sendBuf, trashBuf, elem_count, enablePrefetch, session);
                (void)evt.Wait(session);
            }
        }
    } else {
        if (ctx.my_rank != source_rank) {
            PerfTileData<T, count> stagingTile;
            TASSIGN(stagingTile, 0x0);

            KernelShapeDyn shape(1, 1, 1, 1, elem_count);
            KernelStrideDyn stride(elem_count, elem_count, elem_count, elem_count, 1);
            KernelGlobal<T> localRecvG(ctx.recvBuf, shape, stride);

            __gm__ T *remoteSendBuf = HcclRemotePtr(hcclCtx, ctx.sendBuf, source_rank);
            KernelGlobal<T> remoteSendG(remoteSendBuf, shape, stride);

            uint64_t t0 = get_syscnt();
            pto::comm::TGET(localRecvG, remoteSendG, stagingTile);
            pipe_barrier(PIPE_ALL);
            uint64_t t1 = get_syscnt();
            *cycleOut = t1 - t0;
        }
    }

    pipe_barrier(PIPE_ALL);
}

// Launch one phase of TgetPerfKernel and host-sync. Collective barriers are
// inserted by the caller because the exact number of barriers is part of the
// two-phase test protocol (source-rank prefetch → inter-phase barrier → reader
// TGET + measurement).
template <typename T, size_t count>
inline void LaunchAndSyncTgetPhase(MultiRankPerfEnv<T, count> &env, int n_ranks, int source_rank, int enablePrefetch,
                                   int phase)
{
    TgetPerfKernel<T, count><<<1, nullptr, env.ctx.stream>>>(
        env.sendBuf, n_ranks, source_rank, static_cast<int>(count), enablePrefetch, phase, env.ctx.deviceCtx,
        (uint8_t *)env.sdmaMgr.GetWorkspaceAddr(), 0, reinterpret_cast<uint8_t *>(env.trashDev),
        reinterpret_cast<uint64_t *>(env.cycleDev));
    env.ctx.aclStatus = aclrtSynchronizeStream(env.ctx.stream);
}

template <typename T, size_t count>
bool RunTgetPerfKernel(int rank_id, int n_ranks, int n_devices, int first_device_id, const HcclRootInfo *rootInfo,
                       int source_rank, bool prefetch, uint64_t &outCycles)
{
    MultiRankPerfEnv<T, count> env;
    if (!env.Init(rank_id, n_ranks, n_devices, first_device_id, rootInfo))
        return false;
    if (rank_id == source_rank)
        env.UploadInputToSend();

    int enablePrefetch = prefetch ? 1 : 0;

    HcclHostBarrier(env.ctx.comm, env.ctx.stream);
    LaunchAndSyncTgetPhase<T, count>(env, n_ranks, source_rank, enablePrefetch, 0);
    HcclHostBarrier(env.ctx.comm, env.ctx.stream);
    LaunchAndSyncTgetPhase<T, count>(env, n_ranks, source_rank, enablePrefetch, 1);
    HcclHostBarrier(env.ctx.comm, env.ctx.stream);

    env.DownloadCycles(outCycles, rank_id != source_rank);
    return env.Finalize();
}

template <typename T, size_t count>
bool RunTgetPerf(int n_ranks, int n_devices, int first_rank_id, int first_device_id)
{
    const int source_rank = first_rank_id;
    constexpr int kWarmup = 2;
    constexpr int kMeasured = 5;

    auto runOnce = [&](bool prefetch, uint64_t &cycles) {
        return ForkAndRunWithHcclRootInfo(
            n_ranks, first_rank_id, first_device_id, [&](int rankId, const HcclRootInfo *rootInfo) {
                return RunTgetPerfKernel<T, count>(rankId, n_ranks, n_devices, first_device_id, rootInfo, source_rank,
                                                   prefetch, cycles);
            });
    };
    // Measurement lives on the reader rank (rank != source), so print from non-rank-0.
    auto printSummary = [&](uint64_t avgCold, uint64_t avgWarm) {
        if (CommMpiRank() == 0)
            return;
        L2PerfSummaryFmt fmt{};
        fmt.title = "TGET: source-side L2 prefetch (HCCS coherency test)";
        fmt.metric = "TGET";
        fmt.dataBytes = count * sizeof(T);
        fmt.kWarmup = kWarmup;
        fmt.kMeasured = kMeasured;
        fmt.tileChunkElems = static_cast<int>((count <= 256) ? count : 256);
        PrintL2ColdWarmSummary(fmt, avgCold, avgWarm);
    };
    return RunColdWarmPerfSweep(kWarmup, kMeasured, runOnce, printSummary);
}

template bool RunTgetPerf<float, 4096>(int, int, int, int);
template bool RunTgetPerf<float, 65536>(int, int, int, int);
template bool RunTgetPerf<float, 262144>(int, int, int, int);
