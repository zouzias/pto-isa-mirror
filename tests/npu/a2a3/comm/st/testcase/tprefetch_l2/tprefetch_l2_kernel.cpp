/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
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
#include "pto/npu/comm/async/sdma/sdma_workspace_manager.hpp"
#include "pto/common/pto_tile.hpp"
#include "../common.hpp"

#define ENABLE_DEBUG_PRINT 1

using SdmaWorkspaceManager = pto::comm::sdma::SdmaWorkspaceManager;

// ============================================================================
// Common: TLOAD/TSTORE copy loop (shared by all kernels)
// ============================================================================
template <typename T, size_t count>
PTO_INTERNAL void CopyViaTile(__gm__ T *src, __gm__ T *dst, int elem_count)
{
    constexpr int kTileCols = (count <= 256) ? static_cast<int>(count) : 256;
    static_assert(count % kTileCols == 0, "count must be a multiple of kTileCols for fixed-size Tile");
    using TileData = pto::Tile<pto::TileType::Vec, T, 1, kTileCols, pto::BLayout::RowMajor>;
    using ChunkShape = pto::Shape<1, 1, 1, 1, kTileCols>;
    using ChunkStride = pto::Stride<1, 1, 1, 1, 1>;

    TileData tile;
    TASSIGN(tile, 0x0);

    for (int offset = 0; offset < elem_count; offset += kTileCols) {
        pto::GlobalTensor<T, ChunkShape, ChunkStride, pto::Layout::ND> srcChunk(src + offset);
        pto::GlobalTensor<T, ChunkShape, ChunkStride, pto::Layout::ND> dstChunk(dst + offset);

        TLOAD(tile, srcChunk);
        set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        TSTORE(dstChunk, tile);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    }
}

// ============================================================================
// Baseline Kernel: pure TLOAD/TSTORE, no SDMA.  Sanity check.
// ============================================================================
template <typename T, size_t count>
__global__ AICORE void BaselineKernel(__gm__ T *src, __gm__ T *dst, int elem_count)
{
    if (elem_count <= 0 || elem_count > static_cast<int>(count)) {
        pipe_barrier(PIPE_ALL);
        return;
    }

    CopyViaTile<T, count>(src, dst, elem_count);

    pipe_barrier(PIPE_ALL);
}

// ============================================================================
// TPREFETCH_L2 Correctness Test Kernel (GlobalTensor overload)
//
// Prefetch is a performance hint; even if it fails the TLOAD/TSTORE loop
// must still execute to validate correctness independently.
// ============================================================================
template <typename T, size_t count>
__global__ AICORE void TPrefetchL2CorrectnessKernel(__gm__ T *src, __gm__ T *dst,
                                                     int elem_count,
                                                     __gm__ uint8_t *sdmaWorkspace, uint32_t sdmaSyncId)
{
    using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using Global = pto::GlobalTensor<T, ShapeDyn, StrideDyn, pto::Layout::ND>;
    using ScratchTile = pto::Tile<pto::TileType::Vec, uint8_t, 1, pto::comm::sdma::UB_ALIGN_SIZE>;

    if (elem_count <= 0 || elem_count > static_cast<int>(count)) {
        pipe_barrier(PIPE_ALL);
        return;
    }

    ShapeDyn shape(1, 1, 1, 1, elem_count);
    StrideDyn stride(elem_count, elem_count, elem_count, elem_count, 1);
    Global srcGlobal(src, shape, stride);

    ScratchTile scratchTile;
    TASSIGN(scratchTile, 0x0);

    pto::comm::AsyncSession session;
    bool sessionOk = pto::comm::BuildAsyncSession(scratchTile, sdmaWorkspace, session, sdmaSyncId);
    if (sessionOk) {
        auto evt = pto::comm::TPREFETCH_L2(srcGlobal, session);
        (void)evt.Wait(session);
    }

    CopyViaTile<T, count>(src, dst, elem_count);

    pipe_barrier(PIPE_ALL);
}

// ============================================================================
// TPREFETCH_L2 Raw Pointer Test Kernel
// ============================================================================
template <typename T, size_t count>
__global__ AICORE void TPrefetchL2RawPtrKernel(__gm__ T *src, __gm__ T *dst,
                                                int elem_count,
                                                __gm__ uint8_t *sdmaWorkspace, uint32_t sdmaSyncId)
{
    using ScratchTile = pto::Tile<pto::TileType::Vec, uint8_t, 1, pto::comm::sdma::UB_ALIGN_SIZE>;

    if (elem_count <= 0 || elem_count > static_cast<int>(count)) {
        pipe_barrier(PIPE_ALL);
        return;
    }

    uint64_t totalBytes = static_cast<uint64_t>(elem_count) * sizeof(T);

    ScratchTile scratchTile;
    TASSIGN(scratchTile, 0x0);

    pto::comm::AsyncSession session;
    bool sessionOk = pto::comm::BuildAsyncSession(scratchTile, sdmaWorkspace, session, sdmaSyncId);
    if (sessionOk) {
        auto evt = pto::comm::TPREFETCH_L2((__gm__ void *)src, totalBytes, session);
        (void)evt.Wait(session);
    }

    CopyViaTile<T, count>(src, dst, elem_count);

    pipe_barrier(PIPE_ALL);
}

// ============================================================================
// Test helpers — shared setup/teardown/verify to cut duplication
// ============================================================================
struct SingleCardTestEnv {
    aclrtStream stream = nullptr;
    uint8_t *inputHost = nullptr;
    uint8_t *outputHost = nullptr;
    void *srcDevice = nullptr;
    void *dstDevice = nullptr;
    size_t dataBytes = 0;
    int aclStatus = 0;

    bool Init(int deviceId, size_t bytes)
    {
        dataBytes = bytes;
        aclStatus |= aclrtSetDevice(deviceId);
        aclStatus |= aclrtCreateStream(&stream);
        aclStatus |= aclrtMallocHost(reinterpret_cast<void **>(&inputHost), dataBytes);
        aclStatus |= aclrtMallocHost(reinterpret_cast<void **>(&outputHost), dataBytes);
        aclStatus |= aclrtMalloc(&srcDevice, dataBytes, ACL_MEM_MALLOC_HUGE_FIRST);
        aclStatus |= aclrtMalloc(&dstDevice, dataBytes, ACL_MEM_MALLOC_HUGE_FIRST);
        return aclStatus == 0;
    }
    void SyncAndReadBack()
    {
        aclStatus |= aclrtSynchronizeStream(stream);
        aclStatus |= aclrtMemcpy(outputHost, dataBytes, dstDevice, dataBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    }
    void Teardown()
    {
        aclStatus |= aclrtFree(srcDevice);
        aclStatus |= aclrtFree(dstDevice);
        aclStatus |= aclrtFreeHost(inputHost);
        aclStatus |= aclrtFreeHost(outputHost);
        aclStatus |= aclrtDestroyStream(stream);
    }
};

template <typename T>
PTO_INTERNAL void FillAndUpload(SingleCardTestEnv &env, size_t count, int modulus)
{
    T *in = reinterpret_cast<T *>(env.inputHost);
    T *out = reinterpret_cast<T *>(env.outputHost);
    for (size_t i = 0; i < count; ++i) {
        in[i] = static_cast<T>(i % modulus);
        out[i] = static_cast<T>(-1);
    }
    env.aclStatus |= aclrtMemcpy(env.srcDevice, env.dataBytes, env.inputHost, env.dataBytes,
                                 ACL_MEMCPY_HOST_TO_DEVICE);
    env.aclStatus |= aclrtMemcpy(env.dstDevice, env.dataBytes, env.outputHost, env.dataBytes,
                                 ACL_MEMCPY_HOST_TO_DEVICE);
}

template <typename T>
PTO_INTERNAL bool VerifyOutputAndPrint(const SingleCardTestEnv &env, size_t count, int modulus, const char *tag)
{
    const T *out = reinterpret_cast<const T *>(env.outputHost);
    for (size_t i = 0; i < count; ++i) {
        T expected = static_cast<T>(i % modulus);
        if (out[i] != expected) {
            std::cout << tag << ": index " << i << " expected " << (float)expected << " got " << (float)out[i]
                      << std::endl;
            return false;
        }
    }
#if ENABLE_DEBUG_PRINT
    std::cout << "\n================================================================" << std::endl;
    std::cout << "[DEBUG] " << tag << " SUCCESSFUL!" << std::endl;
    std::cout << "  count=" << count << ", dtype_size=" << sizeof(T) << std::endl;
    std::cout << "  Sample: [ ";
    for (size_t i = 0; i < (count > 5 ? 5 : count); ++i)
        std::cout << (float)out[i] << " ";
    if (count > 5)
        std::cout << "... ";
    std::cout << "]" << std::endl;
    std::cout << "================================================================\n" << std::endl;
#endif
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

// ============================================================================
// Host-side test runners
// ============================================================================

template <typename T, size_t count>
bool RunBaseline(int deviceId)
{
    constexpr size_t dataBytes = count * sizeof(T);
    SingleCardTestEnv env;
    if (!env.Init(deviceId, dataBytes)) {
        std::cerr << "[ERROR] Baseline: init failed!" << std::endl;
        return false;
    }
    FillAndUpload<T>(env, count, 1000);

    BaselineKernel<T, count><<<1, nullptr, env.stream>>>(
        reinterpret_cast<T *>(env.srcDevice), reinterpret_cast<T *>(env.dstDevice),
        static_cast<int>(count));
    env.SyncAndReadBack();

    bool is_ok = VerifyOutputAndPrint<T>(env, count, 1000, "Baseline TLOAD/TSTORE");
    env.Teardown();
    return is_ok && (env.aclStatus == 0);
}

template <typename T, size_t count>
bool RunPrefetchL2Correctness(int deviceId)
{
    constexpr size_t dataBytes = count * sizeof(T);
    SingleCardTestEnv env;
    if (!env.Init(deviceId, dataBytes)) {
        std::cerr << "[ERROR] PrefetchL2: init failed!" << std::endl;
        return false;
    }
    FillAndUpload<T>(env, count, 1000);

    SdmaWorkspaceManager sdmaMgr;
    if (!sdmaMgr.Init()) {
        std::cerr << "[WARN] SdmaWorkspaceManager Init failed — prefetch will be skipped inside kernel" << std::endl;
    }

    TPrefetchL2CorrectnessKernel<T, count><<<1, nullptr, env.stream>>>(
        reinterpret_cast<T *>(env.srcDevice), reinterpret_cast<T *>(env.dstDevice),
        static_cast<int>(count),
        reinterpret_cast<uint8_t *>(sdmaMgr.GetWorkspaceAddr()), 0);
    env.SyncAndReadBack();

    bool is_ok = VerifyOutputAndPrint<T>(env, count, 1000, "TPREFETCH_L2 GlobalTensor correctness");
    env.Teardown();
    sdmaMgr.Finalize();
    return is_ok && (env.aclStatus == 0);
}

template <typename T, size_t count>
bool RunPrefetchL2RawPtr(int deviceId)
{
    constexpr size_t dataBytes = count * sizeof(T);
    SingleCardTestEnv env;
    if (!env.Init(deviceId, dataBytes)) {
        std::cerr << "[ERROR] PrefetchL2 RawPtr: init failed!" << std::endl;
        return false;
    }
    FillAndUpload<T>(env, count, 500);

    SdmaWorkspaceManager sdmaMgr;
    if (!sdmaMgr.Init()) {
        std::cerr << "[WARN] SdmaWorkspaceManager Init failed — prefetch will be skipped inside kernel" << std::endl;
    }

    TPrefetchL2RawPtrKernel<T, count><<<1, nullptr, env.stream>>>(
        reinterpret_cast<T *>(env.srcDevice), reinterpret_cast<T *>(env.dstDevice),
        static_cast<int>(count),
        reinterpret_cast<uint8_t *>(sdmaMgr.GetWorkspaceAddr()), 0);
    env.SyncAndReadBack();

    bool is_ok = VerifyOutputAndPrint<T>(env, count, 500, "TPREFETCH_L2 raw pointer correctness");
    env.Teardown();
    sdmaMgr.Finalize();
    return is_ok && (env.aclStatus == 0);
}

template bool RunBaseline<float, 4096>(int deviceId);
template bool RunBaseline<int32_t, 4096>(int deviceId);
template bool RunPrefetchL2Correctness<float, 4096>(int deviceId);
template bool RunPrefetchL2Correctness<int32_t, 4096>(int deviceId);
template bool RunPrefetchL2RawPtr<float, 4096>(int deviceId);
template bool RunPrefetchL2RawPtr<int32_t, 4096>(int deviceId);

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
__global__ AICORE void TPrefetchL2TputAsyncKernel(__gm__ T *commBuf, int nranks, int root_rank,
                                                   int elem_count, int enablePrefetch,
                                                   __gm__ HcclDeviceContext *hcclCtx,
                                                   __gm__ uint8_t *sdmaWorkspace, uint32_t sdmaSyncId)
{
    using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using Global = pto::GlobalTensor<T, ShapeDyn, StrideDyn, pto::Layout::ND>;
    using ScratchTile = pto::Tile<pto::TileType::Vec, uint8_t, 1, pto::comm::sdma::UB_ALIGN_SIZE>;

    if (elem_count <= 0 || elem_count > static_cast<int>(count)) {
        pipe_barrier(PIPE_ALL);
        return;
    }

    ShapeDyn shape(1, 1, 1, 1, elem_count);
    StrideDyn stride(elem_count, elem_count, elem_count, elem_count, 1);

    int my_rank = static_cast<int>(hcclCtx->rankId);

    __gm__ T *commData = reinterpret_cast<__gm__ T *>(commBuf);
    __gm__ T *sendBuf = commData;
    __gm__ T *recvBuf = commData + count;

    Global sendG(sendBuf, shape, stride);

    if (my_rank == root_rank) {
        ScratchTile scratchTile;
        TASSIGN(scratchTile, 0x0);
        pto::comm::AsyncSession session;
        if (!pto::comm::BuildAsyncSession(scratchTile, sdmaWorkspace, session, sdmaSyncId)) {
            pipe_barrier(PIPE_ALL);
            return;
        }

        if (enablePrefetch) {
            (void)pto::comm::TPREFETCH_L2(sendG, session);
        }

        pto::comm::AsyncEvent lastEvent;
        for (int target_rank = 0; target_rank < nranks; ++target_rank) {
            if (target_rank == root_rank) {
                continue;
            }
            __gm__ T *remoteRecvBuf = HcclRemotePtr(hcclCtx, recvBuf, target_rank);
            Global remoteRecvG(remoteRecvBuf, shape, stride);
            lastEvent = pto::comm::TPUT_ASYNC(remoteRecvG, sendG, session);
        }
        (void)lastEvent.Wait(session);
    }

    pipe_barrier(PIPE_ALL);
}

// ============================================================================
// Host-side runner: TPUT_ASYNC with optional TPREFETCH_L2 (multi-rank via HCCL)
// ============================================================================
template <typename T>
PTO_INTERNAL bool VerifyTputAsyncOutput(int rank_id, int root_rank, const uint8_t *output_host, size_t count,
                                        bool prefetch)
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

    HcclHostBarrier(env.ctx.comm, env.ctx.stream);

    int enablePrefetch = prefetch ? 1 : 0;
    TPrefetchL2TputAsyncKernel<T, count><<<1, nullptr, env.ctx.stream>>>(
        env.sendBuf, n_ranks, root_rank, static_cast<int>(count), enablePrefetch,
        env.ctx.deviceCtx, (uint8_t *)env.sdmaMgr.GetWorkspaceAddr(), 0);
    env.ctx.aclStatus = aclrtSynchronizeStream(env.ctx.stream);

    HcclHostBarrier(env.ctx.comm, env.ctx.stream);

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
__global__ AICORE void TPrefetchL2PerfKernel(__gm__ T *commBuf, int nranks, int root_rank,
                                              int elem_count, int enablePrefetch,
                                              __gm__ HcclDeviceContext *hcclCtx,
                                              __gm__ uint8_t *sdmaWorkspace, uint32_t sdmaSyncId,
                                              __gm__ uint8_t *trashBuf,
                                              __gm__ uint64_t *cycleOut)
{
    using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using Global = pto::GlobalTensor<T, ShapeDyn, StrideDyn, pto::Layout::ND>;
    using ScratchTile = pto::Tile<pto::TileType::Vec, uint8_t, 1, pto::comm::sdma::UB_ALIGN_SIZE>;

    if (elem_count <= 0 || elem_count > static_cast<int>(count)) {
        pipe_barrier(PIPE_ALL);
        return;
    }

    ShapeDyn shape(1, 1, 1, 1, elem_count);
    StrideDyn stride(elem_count, elem_count, elem_count, elem_count, 1);

    int my_rank = static_cast<int>(hcclCtx->rankId);

    __gm__ T *commData = reinterpret_cast<__gm__ T *>(commBuf);
    __gm__ T *sendBuf = commData;
    __gm__ T *recvBuf = commData + count;

    Global sendG(sendBuf, shape, stride);

    if (my_rank == root_rank) {
        ScratchTile scratchTile;
        TASSIGN(scratchTile, 0x0);
        pto::comm::AsyncSession session;
        if (!pto::comm::BuildAsyncSession(scratchTile, sdmaWorkspace, session, sdmaSyncId)) {
            pipe_barrier(PIPE_ALL);
            return;
        }

        uint64_t totalBytes = static_cast<uint64_t>(elem_count) * sizeof(T);
        __gm__ void *prefetchTarget = enablePrefetch ?
            (__gm__ void *)sendBuf : (__gm__ void *)trashBuf;
        (void)pto::comm::TPREFETCH_L2(prefetchTarget, totalBytes, session);

        uint64_t t0 = get_syscnt();

        pto::comm::AsyncEvent lastEvent;
        for (int target_rank = 0; target_rank < nranks; ++target_rank) {
            if (target_rank == root_rank) {
                continue;
            }
            __gm__ T *remoteRecvBuf = HcclRemotePtr(hcclCtx, recvBuf, target_rank);
            Global remoteRecvG(remoteRecvBuf, shape, stride);
            lastEvent = pto::comm::TPUT_ASYNC(remoteRecvG, sendG, session);
        }
        (void)lastEvent.Wait(session);

        uint64_t t1 = get_syscnt();
        *cycleOut = t1 - t0;
    }

    pipe_barrier(PIPE_ALL);
}

// ============================================================================
// Host-side perf runner: single iteration, returns cycle count
// ============================================================================
template <typename T, size_t count>
bool RunPrefetchL2PerfKernel(int rank_id, int n_ranks, int n_devices, int first_device_id,
                              const HcclRootInfo *rootInfo, int root_rank, bool prefetch,
                              uint64_t &outCycles)
{
    MultiRankPerfEnv<T, count> env;
    if (!env.Init(rank_id, n_ranks, n_devices, first_device_id, rootInfo))
        return false;
    env.UploadInputToSend();

    HcclHostBarrier(env.ctx.comm, env.ctx.stream);

    int enablePrefetch = prefetch ? 1 : 0;
    TPrefetchL2PerfKernel<T, count><<<1, nullptr, env.ctx.stream>>>(
        env.sendBuf, n_ranks, root_rank, static_cast<int>(count), enablePrefetch,
        env.ctx.deviceCtx, (uint8_t *)env.sdmaMgr.GetWorkspaceAddr(), 0,
        reinterpret_cast<uint8_t *>(env.trashDev),
        reinterpret_cast<uint64_t *>(env.cycleDev));
    env.ctx.aclStatus = aclrtSynchronizeStream(env.ctx.stream);

    HcclHostBarrier(env.ctx.comm, env.ctx.stream);

    env.DownloadCycles(outCycles, rank_id == root_rank);
    return env.Finalize();
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

    uint64_t noPrefetchTotal = 0;
    uint64_t prefetchTotal = 0;

    for (int iter = 0; iter < kWarmup + kMeasured; ++iter) {
        uint64_t cycles = 0;
        bool ok = ForkAndRunWithHcclRootInfo(
            n_ranks, first_rank_id, first_device_id, [&](int rankId, const HcclRootInfo *rootInfo) {
                return RunPrefetchL2PerfKernel<T, count>(rankId, n_ranks, n_devices, first_device_id, rootInfo,
                                                          root_rank, false, cycles);
            });
        if (!ok) return false;
        if (iter >= kWarmup)
            noPrefetchTotal += cycles;
    }

    for (int iter = 0; iter < kWarmup + kMeasured; ++iter) {
        uint64_t cycles = 0;
        bool ok = ForkAndRunWithHcclRootInfo(
            n_ranks, first_rank_id, first_device_id, [&](int rankId, const HcclRootInfo *rootInfo) {
                return RunPrefetchL2PerfKernel<T, count>(rankId, n_ranks, n_devices, first_device_id, rootInfo,
                                                          root_rank, true, cycles);
            });
        if (!ok) return false;
        if (iter >= kWarmup)
            prefetchTotal += cycles;
    }

    int mpiRank = CommMpiRank();
    if (mpiRank == 0) {
        uint64_t avgNoPrefetch = noPrefetchTotal / kMeasured;
        uint64_t avgPrefetch = prefetchTotal / kMeasured;
        double ratio = (avgNoPrefetch > 0) ? (double)avgPrefetch / (double)avgNoPrefetch : 0.0;
        size_t dataBytes = count * sizeof(T);

        std::cout << "\n================================================================" << std::endl;
        std::cout << "[PERF] TPUT_ASYNC Performance Comparison" << std::endl;
        std::cout << "  Data size:     " << dataBytes << " bytes (" << dataBytes / 1024 << " KB)" << std::endl;
        std::cout << "  Iterations:    " << kMeasured << " (warmup=" << kWarmup << ")" << std::endl;
        std::cout << "  No prefetch:   " << avgNoPrefetch << " cycles (avg)" << std::endl;
        std::cout << "  With prefetch: " << avgPrefetch << " cycles (avg)" << std::endl;
        std::cout << "  Ratio:         " << ratio << "x  ("
                  << (ratio < 1.0 ? "PREFETCH FASTER" : (ratio > 1.0 ? "NO PREFETCH FASTER" : "EQUAL"))
                  << ")" << std::endl;
        std::cout << "================================================================\n" << std::endl;
    }

    return true;
}

template bool RunPrefetchL2Perf<float, 4096>(int, int, int, int);
template bool RunPrefetchL2Perf<float, 65536>(int, int, int, int);
template bool RunPrefetchL2Perf<float, 262144>(int, int, int, int);

// ============================================================================
// TLOAD Perf Kernel: measure TLOAD latency with/without L2 prefetch
//
// Uses trash-buffer technique for L2 cold control (cf. shmem/CMO example).
// Both paths always execute a SDMA CMO prefetch for fair comparison:
//   enablePrefetch=1: prefetch srcBuf → L2 warm
//   enablePrefetch=0: prefetch trashBuf → L2 cold for srcBuf
// ============================================================================
template <typename T, size_t count>
__global__ AICORE void TloadPerfKernel(__gm__ T *srcBuf, int elem_count, int enablePrefetch,
                                        __gm__ uint8_t *sdmaWorkspace, uint32_t sdmaSyncId,
                                        __gm__ uint8_t *trashBuf,
                                        __gm__ uint64_t *cycleOut)
{
    using ScratchTile = pto::Tile<pto::TileType::Vec, uint8_t, 1, pto::comm::sdma::UB_ALIGN_SIZE>;

    constexpr int kTileCols = (count <= 256) ? static_cast<int>(count) : 256;
    static_assert(count % kTileCols == 0, "count must be a multiple of kTileCols");
    using TileData = pto::Tile<pto::TileType::Vec, T, 1, kTileCols, pto::BLayout::RowMajor>;
    using ChunkShape = pto::Shape<1, 1, 1, 1, kTileCols>;
    using ChunkStride = pto::Stride<1, 1, 1, 1, 1>;

    if (elem_count <= 0 || elem_count > static_cast<int>(count)) {
        pipe_barrier(PIPE_ALL);
        return;
    }

    {
        uint64_t totalBytes = static_cast<uint64_t>(elem_count) * sizeof(T);
        __gm__ void *prefetchTarget = enablePrefetch ?
            (__gm__ void *)srcBuf : (__gm__ void *)trashBuf;

        ScratchTile scratchTile;
        TASSIGN(scratchTile, 0x0);
        pto::comm::AsyncSession session;
        if (pto::comm::BuildAsyncSession(scratchTile, sdmaWorkspace, session, sdmaSyncId)) {
            auto evt = pto::comm::TPREFETCH_L2(prefetchTarget, totalBytes, session);
            (void)evt.Wait(session);
        }
    }

    TileData tile;
    TASSIGN(tile, 0x0);

    uint64_t t0 = get_syscnt();

    for (int offset = 0; offset < elem_count; offset += kTileCols) {
        pto::GlobalTensor<T, ChunkShape, ChunkStride, pto::Layout::ND> srcChunk(srcBuf + offset);
        TLOAD(tile, srcChunk);
        set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    }

    uint64_t t1 = get_syscnt();
    *cycleOut = t1 - t0;

    pipe_barrier(PIPE_ALL);
}

// ============================================================================
// Host-side TLOAD perf runner
// ============================================================================
template <typename T, size_t count>
bool RunTloadPerfOnce(int deviceId, bool prefetch, uint64_t &outCycles)
{
    constexpr size_t dataBytes = count * sizeof(T);
    int aclStatus = 0;

    aclStatus |= aclrtSetDevice(deviceId);
    aclrtStream stream = nullptr;
    aclStatus |= aclrtCreateStream(&stream);

    uint8_t *inputHost = nullptr;
    aclStatus |= aclrtMallocHost(reinterpret_cast<void **>(&inputHost), dataBytes);
    if (aclStatus != 0) {
        std::cerr << "[ERROR] TloadPerf: host alloc failed!" << std::endl;
        return false;
    }
    for (size_t i = 0; i < count; ++i)
        reinterpret_cast<T *>(inputHost)[i] = static_cast<T>(i);

    void *srcDevice = nullptr;
    void *trashDevice = nullptr;
    aclStatus |= aclrtMalloc(&srcDevice, dataBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclStatus |= aclrtMalloc(&trashDevice, dataBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclStatus |= aclrtMemcpy(srcDevice, dataBytes, inputHost, dataBytes, ACL_MEMCPY_HOST_TO_DEVICE);

    void *cycleDev = nullptr;
    aclStatus |= aclrtMalloc(&cycleDev, sizeof(uint64_t), ACL_MEM_MALLOC_HUGE_FIRST);
    uint64_t zero = 0;
    aclStatus |= aclrtMemcpy(cycleDev, sizeof(uint64_t), &zero, sizeof(uint64_t), ACL_MEMCPY_HOST_TO_DEVICE);

    SdmaWorkspaceManager sdmaMgr;
    if (!sdmaMgr.Init()) {
        std::cerr << "[ERROR] TloadPerf: SdmaWorkspaceManager Init failed!" << std::endl;
        aclrtFree(cycleDev);
        aclrtFree(trashDevice);
        aclrtFree(srcDevice);
        aclrtFreeHost(inputHost);
        aclrtDestroyStream(stream);
        return false;
    }

    int enablePrefetch = prefetch ? 1 : 0;
    TloadPerfKernel<T, count><<<1, nullptr, stream>>>(
        reinterpret_cast<T *>(srcDevice), static_cast<int>(count), enablePrefetch,
        (uint8_t *)sdmaMgr.GetWorkspaceAddr(), 0,
        reinterpret_cast<uint8_t *>(trashDevice),
        reinterpret_cast<uint64_t *>(cycleDev));
    aclStatus |= aclrtSynchronizeStream(stream);

    aclrtMemcpy(&outCycles, sizeof(uint64_t), cycleDev, sizeof(uint64_t), ACL_MEMCPY_DEVICE_TO_HOST);

    sdmaMgr.Finalize();
    aclrtFree(cycleDev);
    aclrtFree(trashDevice);
    aclrtFree(srcDevice);
    aclrtFreeHost(inputHost);
    aclrtDestroyStream(stream);

    return aclStatus == 0;
}

// ============================================================================
// Top-level TLOAD perf comparison (single-card, rank 0 only)
// ============================================================================
template <typename T, size_t count>
bool RunTloadPerf(int deviceId)
{
    constexpr int kWarmup = 3;
    constexpr int kMeasured = 10;

    uint64_t coldTotal = 0;
    uint64_t warmTotal = 0;

    for (int iter = 0; iter < kWarmup + kMeasured; ++iter) {
        uint64_t cycles = 0;
        if (!RunTloadPerfOnce<T, count>(deviceId, false, cycles))
            return false;
        if (iter >= kWarmup)
            coldTotal += cycles;
    }

    for (int iter = 0; iter < kWarmup + kMeasured; ++iter) {
        uint64_t cycles = 0;
        if (!RunTloadPerfOnce<T, count>(deviceId, true, cycles))
            return false;
        if (iter >= kWarmup)
            warmTotal += cycles;
    }

    uint64_t avgCold = coldTotal / kMeasured;
    uint64_t avgWarm = warmTotal / kMeasured;
    double speedup = (avgWarm > 0) ? (double)avgCold / (double)avgWarm : 0.0;
    size_t dataBytes = count * sizeof(T);

    std::cout << "\n================================================================" << std::endl;
    std::cout << "[PERF] TLOAD Latency: L2-cold vs L2-prefetched" << std::endl;
    std::cout << "  Data size:      " << dataBytes << " bytes (" << dataBytes / 1024 << " KB)" << std::endl;
    std::cout << "  TLOAD chunks:   " << count / ((count <= 256) ? count : 256) << std::endl;
    std::cout << "  Iterations:     " << kMeasured << " (warmup=" << kWarmup << ")" << std::endl;
    std::cout << "  L2-cold TLOAD:  " << avgCold << " cycles (avg)" << std::endl;
    std::cout << "  L2-warm TLOAD:  " << avgWarm << " cycles (avg)" << std::endl;
    std::cout << "  Speedup:        " << speedup << "x  ("
              << (speedup > 1.05 ? "PREFETCH HELPS" : (speedup < 0.95 ? "PREFETCH HURTS" : "NO SIGNIFICANT DIFF"))
              << ")" << std::endl;
    std::cout << "================================================================\n" << std::endl;

    return true;
}

template bool RunTloadPerf<float, 4096>(int);
template bool RunTloadPerf<float, 65536>(int);
template bool RunTloadPerf<float, 262144>(int);
template bool RunTloadPerf<float, 1048576>(int);

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
    using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using Global = pto::GlobalTensor<T, ShapeDyn, StrideDyn, pto::Layout::ND>;

    ShapeDyn shape(1, 1, 1, 1, elem_count);
    StrideDyn stride(elem_count, elem_count, elem_count, elem_count, 1);
    Global sendG(sendBuf, shape, stride);

    pto::comm::AsyncEvent lastEvent;
    for (int target = 0; target < nranks; ++target) {
        if (target == root_rank) continue;
        __gm__ T *remoteRecvBuf = HcclRemotePtr(hcclCtx, recvBuf, target);
        Global remoteRecvG(remoteRecvBuf, shape, stride);
        lastEvent = pto::comm::TPUT_ASYNC(remoteRecvG, sendG, session);
    }
    (void)lastEvent.Wait(session);
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
        uint64_t totalBytes = static_cast<uint64_t>(elem_count) * sizeof(T);
        __gm__ void *prefetchTarget = enablePrefetch ? (__gm__ void *)recvBuf : (__gm__ void *)trashBuf;
        auto evt = pto::comm::TPREFETCH_L2(prefetchTarget, totalBytes, session);
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
__global__ AICORE void TloadRemotePerfKernel(__gm__ T *commBuf, int nranks, int root_rank,
                                              int elem_count, int enablePrefetch,
                                              __gm__ HcclDeviceContext *hcclCtx,
                                              __gm__ uint8_t *sdmaWorkspace, uint32_t sdmaSyncId,
                                              __gm__ uint8_t *trashBuf,
                                              __gm__ uint64_t *cycleOut)
{
    using ScratchTile = pto::Tile<pto::TileType::Vec, uint8_t, 1, pto::comm::sdma::UB_ALIGN_SIZE>;

    if (elem_count <= 0 || elem_count > static_cast<int>(count)) {
        pipe_barrier(PIPE_ALL);
        return;
    }

    int my_rank = static_cast<int>(hcclCtx->rankId);
    __gm__ T *sendBuf = commBuf;
    __gm__ T *recvBuf = commBuf + count;

    ScratchTile scratchTile;
    TASSIGN(scratchTile, 0x0);
    pto::comm::AsyncSession session;
    bool sessionOk = pto::comm::BuildAsyncSession(scratchTile, sdmaWorkspace, session, sdmaSyncId);

    if (my_rank == root_rank && sessionOk) {
        RemotePerfSendPhase<T, count>(sendBuf, recvBuf, elem_count, nranks, root_rank, hcclCtx, session);
    }
    pipe_barrier(PIPE_ALL);

    if (my_rank != root_rank) {
        RemotePerfTloadPhase<T, count>(recvBuf, elem_count, enablePrefetch, sessionOk, trashBuf, session, cycleOut);
    }
    pipe_barrier(PIPE_ALL);
}

// ============================================================================
// Host-side runner for multi-rank TLOAD perf (single iteration)
// ============================================================================
template <typename T, size_t count>
bool RunTloadRemotePerfKernel(int rank_id, int n_ranks, int n_devices, int first_device_id,
                               const HcclRootInfo *rootInfo, int root_rank, bool prefetch,
                               uint64_t &outCycles)
{
    MultiRankPerfEnv<T, count> env;
    if (!env.Init(rank_id, n_ranks, n_devices, first_device_id, rootInfo))
        return false;
    if (rank_id == root_rank)
        env.UploadInputToSend();

    HcclHostBarrier(env.ctx.comm, env.ctx.stream);

    int enablePrefetch = prefetch ? 1 : 0;
    TloadRemotePerfKernel<T, count><<<1, nullptr, env.ctx.stream>>>(
        env.sendBuf, n_ranks, root_rank, static_cast<int>(count), enablePrefetch,
        env.ctx.deviceCtx, (uint8_t *)env.sdmaMgr.GetWorkspaceAddr(), 0,
        reinterpret_cast<uint8_t *>(env.trashDev),
        reinterpret_cast<uint64_t *>(env.cycleDev));
    env.ctx.aclStatus = aclrtSynchronizeStream(env.ctx.stream);

    HcclHostBarrier(env.ctx.comm, env.ctx.stream);

    env.DownloadCycles(outCycles, rank_id != root_rank);
    return env.Finalize();
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

    uint64_t coldTotal = 0;
    uint64_t warmTotal = 0;
    int mpiRank = CommMpiRank();

    for (int iter = 0; iter < kWarmup + kMeasured; ++iter) {
        uint64_t cycles = 0;
        bool ok = ForkAndRunWithHcclRootInfo(
            n_ranks, first_rank_id, first_device_id, [&](int rankId, const HcclRootInfo *rootInfo) {
                return RunTloadRemotePerfKernel<T, count>(rankId, n_ranks, n_devices, first_device_id, rootInfo,
                                                           root_rank, false, cycles);
            });
        if (!ok) return false;
        if (iter >= kWarmup)
            coldTotal += cycles;
    }

    for (int iter = 0; iter < kWarmup + kMeasured; ++iter) {
        uint64_t cycles = 0;
        bool ok = ForkAndRunWithHcclRootInfo(
            n_ranks, first_rank_id, first_device_id, [&](int rankId, const HcclRootInfo *rootInfo) {
                return RunTloadRemotePerfKernel<T, count>(rankId, n_ranks, n_devices, first_device_id, rootInfo,
                                                           root_rank, true, cycles);
            });
        if (!ok) return false;
        if (iter >= kWarmup)
            warmTotal += cycles;
    }

    if (mpiRank != 0) {
        uint64_t avgCold = coldTotal / kMeasured;
        uint64_t avgWarm = warmTotal / kMeasured;
        double speedup = (avgWarm > 0) ? (double)avgCold / (double)avgWarm : 0.0;
        size_t dataBytes = count * sizeof(T);

        std::cout << "\n================================================================" << std::endl;
        std::cout << "[PERF] Remote TLOAD: Rank 0 → Rank 1 → TLOAD" << std::endl;
        std::cout << "  Data size:      " << dataBytes << " bytes (" << dataBytes / 1024 << " KB)" << std::endl;
        std::cout << "  TLOAD chunks:   " << count / ((count <= 256) ? count : 256) << std::endl;
        std::cout << "  Iterations:     " << kMeasured << " (warmup=" << kWarmup << ")" << std::endl;
        std::cout << "  L2-cold TLOAD:  " << avgCold << " cycles (avg)" << std::endl;
        std::cout << "  L2-warm TLOAD:  " << avgWarm << " cycles (avg)" << std::endl;
        std::cout << "  Speedup:        " << speedup << "x  ("
                  << (speedup > 1.05 ? "PREFETCH HELPS" : (speedup < 0.95 ? "PREFETCH HURTS" : "NO SIGNIFICANT DIFF"))
                  << ")" << std::endl;
        std::cout << "================================================================\n" << std::endl;
    }

    return true;
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
__global__ AICORE void TputSyncPerfKernel(__gm__ T *commBuf, int nranks, int root_rank,
                                           int elem_count, int enablePrefetch,
                                           __gm__ HcclDeviceContext *hcclCtx,
                                           __gm__ uint8_t *sdmaWorkspace, uint32_t sdmaSyncId,
                                           __gm__ uint8_t *trashBuf,
                                           __gm__ uint64_t *cycleOut)
{
    using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using Global = pto::GlobalTensor<T, ShapeDyn, StrideDyn, pto::Layout::ND>;
    using ScratchTile = pto::Tile<pto::TileType::Vec, uint8_t, 1, pto::comm::sdma::UB_ALIGN_SIZE>;

    constexpr int kTileCols = (count <= 256) ? static_cast<int>(count) : 256;
    static_assert(count % kTileCols == 0, "count must be a multiple of kTileCols");
    using TileData = pto::Tile<pto::TileType::Vec, T, 1, kTileCols, pto::BLayout::RowMajor>;

    if (elem_count <= 0 || elem_count > static_cast<int>(count)) {
        pipe_barrier(PIPE_ALL);
        return;
    }

    int my_rank = static_cast<int>(hcclCtx->rankId);

    __gm__ T *sendBuf = commBuf;
    __gm__ T *recvBuf = commBuf + count;

    if (my_rank == root_rank) {
        ScratchTile scratchTile;
        TASSIGN(scratchTile, 0x0);
        pto::comm::AsyncSession session;
        bool sessionOk = pto::comm::BuildAsyncSession(scratchTile, sdmaWorkspace, session, sdmaSyncId);

        if (sessionOk) {
            uint64_t totalBytes = static_cast<uint64_t>(elem_count) * sizeof(T);
            __gm__ void *prefetchTarget = enablePrefetch ?
                (__gm__ void *)sendBuf : (__gm__ void *)trashBuf;
            auto evt = pto::comm::TPREFETCH_L2(prefetchTarget, totalBytes, session);
            (void)evt.Wait(session);
        }

        TileData stagingTile;
        TASSIGN(stagingTile, 0x0);

        ShapeDyn shape(1, 1, 1, 1, elem_count);
        StrideDyn stride(elem_count, elem_count, elem_count, elem_count, 1);
        Global sendG(sendBuf, shape, stride);

        uint64_t t0 = get_syscnt();
        for (int target = 0; target < nranks; ++target) {
            if (target == root_rank) continue;
            __gm__ T *remoteRecvBuf = HcclRemotePtr(hcclCtx, recvBuf, target);
            Global remoteRecvG(remoteRecvBuf, shape, stride);
            pto::comm::TPUT(remoteRecvG, sendG, stagingTile);
        }
        pipe_barrier(PIPE_ALL);
        uint64_t t1 = get_syscnt();
        *cycleOut = t1 - t0;
    }

    pipe_barrier(PIPE_ALL);
}

template <typename T, size_t count>
bool RunTputSyncPerfKernel(int rank_id, int n_ranks, int n_devices, int first_device_id,
                            const HcclRootInfo *rootInfo, int root_rank, bool prefetch,
                            uint64_t &outCycles)
{
    MultiRankPerfEnv<T, count> env;
    if (!env.Init(rank_id, n_ranks, n_devices, first_device_id, rootInfo))
        return false;
    if (rank_id == root_rank)
        env.UploadInputToSend();

    HcclHostBarrier(env.ctx.comm, env.ctx.stream);

    int enablePrefetch = prefetch ? 1 : 0;
    TputSyncPerfKernel<T, count><<<1, nullptr, env.ctx.stream>>>(
        env.sendBuf, n_ranks, root_rank, static_cast<int>(count), enablePrefetch,
        env.ctx.deviceCtx, (uint8_t *)env.sdmaMgr.GetWorkspaceAddr(), 0,
        reinterpret_cast<uint8_t *>(env.trashDev),
        reinterpret_cast<uint64_t *>(env.cycleDev));
    env.ctx.aclStatus = aclrtSynchronizeStream(env.ctx.stream);

    HcclHostBarrier(env.ctx.comm, env.ctx.stream);

    env.DownloadCycles(outCycles, rank_id == root_rank);
    return env.Finalize();
}

template <typename T, size_t count>
bool RunTputSyncPerf(int n_ranks, int n_devices, int first_rank_id, int first_device_id)
{
    const int root_rank = first_rank_id;
    constexpr int kWarmup = 2;
    constexpr int kMeasured = 5;

    uint64_t coldTotal = 0;
    uint64_t warmTotal = 0;
    int mpiRank = CommMpiRank();

    for (int iter = 0; iter < kWarmup + kMeasured; ++iter) {
        uint64_t cycles = 0;
        bool ok = ForkAndRunWithHcclRootInfo(
            n_ranks, first_rank_id, first_device_id, [&](int rankId, const HcclRootInfo *rootInfo) {
                return RunTputSyncPerfKernel<T, count>(rankId, n_ranks, n_devices, first_device_id, rootInfo,
                                                       root_rank, false, cycles);
            });
        if (!ok) return false;
        if (iter >= kWarmup)
            coldTotal += cycles;
    }

    for (int iter = 0; iter < kWarmup + kMeasured; ++iter) {
        uint64_t cycles = 0;
        bool ok = ForkAndRunWithHcclRootInfo(
            n_ranks, first_rank_id, first_device_id, [&](int rankId, const HcclRootInfo *rootInfo) {
                return RunTputSyncPerfKernel<T, count>(rankId, n_ranks, n_devices, first_device_id, rootInfo,
                                                       root_rank, true, cycles);
            });
        if (!ok) return false;
        if (iter >= kWarmup)
            warmTotal += cycles;
    }

    if (mpiRank == 0) {
        uint64_t avgCold = coldTotal / kMeasured;
        uint64_t avgWarm = warmTotal / kMeasured;
        double speedup = (avgWarm > 0) ? (double)avgCold / (double)avgWarm : 0.0;
        size_t dataBytes = count * sizeof(T);

        std::cout << "\n================================================================" << std::endl;
        std::cout << "[PERF] TPUT (sync): sender-side L2 prefetch" << std::endl;
        std::cout << "  Data size:      " << dataBytes << " bytes (" << dataBytes / 1024 << " KB)" << std::endl;
        std::cout << "  Tile chunk:     " << ((count <= 256) ? count : 256) << " elements" << std::endl;
        std::cout << "  Iterations:     " << kMeasured << " (warmup=" << kWarmup << ")" << std::endl;
        std::cout << "  L2-cold TPUT:   " << avgCold << " cycles (avg)" << std::endl;
        std::cout << "  L2-warm TPUT:   " << avgWarm << " cycles (avg)" << std::endl;
        std::cout << "  Speedup:        " << speedup << "x  ("
                  << (speedup > 1.05 ? "PREFETCH HELPS" : (speedup < 0.95 ? "PREFETCH HURTS" : "NO SIGNIFICANT DIFF"))
                  << ")" << std::endl;
        std::cout << "================================================================\n" << std::endl;
    }

    return true;
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
__global__ AICORE void TgetPerfKernel(__gm__ T *commBuf, int nranks, int source_rank,
                                       int elem_count, int enablePrefetch, int phase,
                                       __gm__ HcclDeviceContext *hcclCtx,
                                       __gm__ uint8_t *sdmaWorkspace, uint32_t sdmaSyncId,
                                       __gm__ uint8_t *trashBuf,
                                       __gm__ uint64_t *cycleOut)
{
    using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using Global = pto::GlobalTensor<T, ShapeDyn, StrideDyn, pto::Layout::ND>;
    using ScratchTile = pto::Tile<pto::TileType::Vec, uint8_t, 1, pto::comm::sdma::UB_ALIGN_SIZE>;

    constexpr int kTileCols = (count <= 256) ? static_cast<int>(count) : 256;
    static_assert(count % kTileCols == 0, "count must be a multiple of kTileCols");
    using TileData = pto::Tile<pto::TileType::Vec, T, 1, kTileCols, pto::BLayout::RowMajor>;

    if (elem_count <= 0 || elem_count > static_cast<int>(count)) {
        pipe_barrier(PIPE_ALL);
        return;
    }

    int my_rank = static_cast<int>(hcclCtx->rankId);
    __gm__ T *sendBuf = commBuf;
    __gm__ T *recvBuf = commBuf + count;

    if (phase == 0) {
        if (my_rank == source_rank) {
            ScratchTile scratchTile;
            TASSIGN(scratchTile, 0x0);
            pto::comm::AsyncSession session;
            bool sessionOk = pto::comm::BuildAsyncSession(scratchTile, sdmaWorkspace, session, sdmaSyncId);
            if (sessionOk) {
                uint64_t totalBytes = static_cast<uint64_t>(elem_count) * sizeof(T);
                __gm__ void *prefetchTarget = enablePrefetch ?
                    (__gm__ void *)sendBuf : (__gm__ void *)trashBuf;
                auto evt = pto::comm::TPREFETCH_L2(prefetchTarget, totalBytes, session);
                (void)evt.Wait(session);
            }
        }
    } else {
        if (my_rank != source_rank) {
            TileData stagingTile;
            TASSIGN(stagingTile, 0x0);

            ShapeDyn shape(1, 1, 1, 1, elem_count);
            StrideDyn stride(elem_count, elem_count, elem_count, elem_count, 1);
            Global localRecvG(recvBuf, shape, stride);

            __gm__ T *remoteSendBuf = HcclRemotePtr(hcclCtx, sendBuf, source_rank);
            Global remoteSendG(remoteSendBuf, shape, stride);

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
PTO_INTERNAL void LaunchAndSyncTgetPhase(MultiRankPerfEnv<T, count> &env, int n_ranks, int source_rank,
                                         int enablePrefetch, int phase)
{
    TgetPerfKernel<T, count><<<1, nullptr, env.ctx.stream>>>(
        env.sendBuf, n_ranks, source_rank, static_cast<int>(count), enablePrefetch, phase,
        env.ctx.deviceCtx, (uint8_t *)env.sdmaMgr.GetWorkspaceAddr(), 0,
        reinterpret_cast<uint8_t *>(env.trashDev),
        reinterpret_cast<uint64_t *>(env.cycleDev));
    env.ctx.aclStatus = aclrtSynchronizeStream(env.ctx.stream);
}

template <typename T, size_t count>
bool RunTgetPerfKernel(int rank_id, int n_ranks, int n_devices, int first_device_id,
                        const HcclRootInfo *rootInfo, int source_rank, bool prefetch,
                        uint64_t &outCycles)
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

    uint64_t coldTotal = 0;
    uint64_t warmTotal = 0;
    int mpiRank = CommMpiRank();

    for (int iter = 0; iter < kWarmup + kMeasured; ++iter) {
        uint64_t cycles = 0;
        bool ok = ForkAndRunWithHcclRootInfo(
            n_ranks, first_rank_id, first_device_id, [&](int rankId, const HcclRootInfo *rootInfo) {
                return RunTgetPerfKernel<T, count>(rankId, n_ranks, n_devices, first_device_id, rootInfo,
                                                    source_rank, false, cycles);
            });
        if (!ok) return false;
        if (iter >= kWarmup)
            coldTotal += cycles;
    }

    for (int iter = 0; iter < kWarmup + kMeasured; ++iter) {
        uint64_t cycles = 0;
        bool ok = ForkAndRunWithHcclRootInfo(
            n_ranks, first_rank_id, first_device_id, [&](int rankId, const HcclRootInfo *rootInfo) {
                return RunTgetPerfKernel<T, count>(rankId, n_ranks, n_devices, first_device_id, rootInfo,
                                                    source_rank, true, cycles);
            });
        if (!ok) return false;
        if (iter >= kWarmup)
            warmTotal += cycles;
    }

    if (mpiRank != 0) {
        uint64_t avgCold = coldTotal / kMeasured;
        uint64_t avgWarm = warmTotal / kMeasured;
        double speedup = (avgWarm > 0) ? (double)avgCold / (double)avgWarm : 0.0;
        size_t dataBytes = count * sizeof(T);

        std::cout << "\n================================================================" << std::endl;
        std::cout << "[PERF] TGET: source-side L2 prefetch (HCCS coherency test)" << std::endl;
        std::cout << "  Data size:      " << dataBytes << " bytes (" << dataBytes / 1024 << " KB)" << std::endl;
        std::cout << "  Tile chunk:     " << ((count <= 256) ? count : 256) << " elements" << std::endl;
        std::cout << "  Iterations:     " << kMeasured << " (warmup=" << kWarmup << ")" << std::endl;
        std::cout << "  L2-cold TGET:   " << avgCold << " cycles (avg)" << std::endl;
        std::cout << "  L2-warm TGET:   " << avgWarm << " cycles (avg)" << std::endl;
        std::cout << "  Speedup:        " << speedup << "x  ("
                  << (speedup > 1.05 ? "PREFETCH HELPS" : (speedup < 0.95 ? "PREFETCH HURTS" : "NO SIGNIFICANT DIFF"))
                  << ")" << std::endl;
        std::cout << "================================================================\n" << std::endl;
    }

    return true;
}

template bool RunTgetPerf<float, 4096>(int, int, int, int);
template bool RunTgetPerf<float, 65536>(int, int, int, int);
template bool RunTgetPerf<float, 262144>(int, int, int, int);
