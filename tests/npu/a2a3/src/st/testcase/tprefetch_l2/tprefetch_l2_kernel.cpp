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
// Single-card ST for pto::TPREFETCH_L2 (workspace API only).
//
// This file purposely does NOT pull in HCCL or any comm/* host infrastructure
// beyond the host-side SdmaWorkspaceManager that allocates the SDMA workspace
// (a runtime-only helper). Cross-rank coverage for TPREFETCH_L2 lives under
// tests/npu/a2a3/comm/st/testcase/tprefetch_l2/ where TPUT_ASYNC / TGET test
// drivers already require the full HCCL test scaffold.
// ============================================================================

#include <cstddef>
#include <cstdint>
#include <iostream>

#include <pto/pto-inst.hpp>
#include "pto/npu/comm/async/sdma/sdma_workspace_manager.hpp"
#include "pto/common/pto_tile.hpp"

#include "tprefetch_l2_kernel.h"

#define ENABLE_DEBUG_PRINT 1

using SdmaWorkspaceManager = pto::comm::sdma::SdmaWorkspaceManager;

// ============================================================================
// Kernel-wide type aliases — fully-dynamic 5-D Shape/Stride is the canonical
// shape for prefetch GlobalTensors so the same kernel handles every elem count.
// ============================================================================
using KernelShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
using KernelStrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
template <typename T>
using KernelGlobal = pto::GlobalTensor<T, KernelShapeDyn, KernelStrideDyn, pto::Layout::ND>;

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

// ============================================================================
// TLOAD/TSTORE copy loop shared by Baseline + correctness kernels.
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
// Baseline kernel: pure TLOAD/TSTORE, no SDMA. Sanity check.
// ============================================================================
template <typename T, size_t count>
__global__ AICORE void BaselineKernel(__gm__ T *src, __gm__ T *dst, int elem_count)
{
    if (!BoundsOkOrFinalize<count>(elem_count)) {
        return;
    }
    CopyViaTile<T, count>(src, dst, elem_count);
    pipe_barrier(PIPE_ALL);
}

// ============================================================================
// Correctness kernel — workspace API, GlobalTensor overload.
//
// pto::TPREFETCH_L2(srcGlobal, workspace) builds a transient AsyncSession on
// the AICORE stack and returns an AsyncEvent whose handle is the workspace
// base. The 0-arg evt.Wait() recovers the equivalent session for the wait,
// so the caller never has to mention sessions explicitly.
// Prefetch is a performance hint; even if it fails the TLOAD/TSTORE loop
// must still execute to validate correctness independently.
// ============================================================================
template <typename T, size_t count>
__global__ AICORE void TPrefetchL2CorrectnessKernel(__gm__ T *src, __gm__ T *dst, int elem_count,
                                                    __gm__ uint8_t *sdmaWorkspace)
{
    if (!BoundsOkOrFinalize<count>(elem_count)) {
        return;
    }

    KernelShapeDyn shape(1, 1, 1, 1, elem_count);
    KernelStrideDyn stride(elem_count, elem_count, elem_count, elem_count, 1);
    KernelGlobal<T> srcGlobal(src, shape, stride);

    auto evt = pto::TPREFETCH_L2(srcGlobal, sdmaWorkspace);
    (void)evt.Wait();

    CopyViaTile<T, count>(src, dst, elem_count);
    pipe_barrier(PIPE_ALL);
}

// ============================================================================
// Correctness kernel — workspace API, raw pointer overload.
// ============================================================================
template <typename T, size_t count>
__global__ AICORE void TPrefetchL2RawPtrKernel(__gm__ T *src, __gm__ T *dst, int elem_count,
                                               __gm__ uint8_t *sdmaWorkspace)
{
    if (!BoundsOkOrFinalize<count>(elem_count)) {
        return;
    }

    uint64_t totalBytes = static_cast<uint64_t>(elem_count) * sizeof(T);
    auto evt = pto::TPREFETCH_L2(reinterpret_cast<__gm__ void *>(src), totalBytes, sdmaWorkspace);
    (void)evt.Wait();

    CopyViaTile<T, count>(src, dst, elem_count);
    pipe_barrier(PIPE_ALL);
}

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
// TLOAD perf kernel: measure TLOAD latency with/without L2 prefetch.
//
// Trash-buffer technique for L2 cold control: BOTH branches always issue an
// SDMA CMO prefetch so the per-call SDMA cost is held constant; only the
// target buffer differs (real srcBuf for "warm", trashBuf for "cold").
// ============================================================================
template <typename T, size_t count>
__global__ AICORE void TloadPerfKernel(__gm__ T *srcBuf, int elem_count, int enablePrefetch,
                                       __gm__ uint8_t *sdmaWorkspace, __gm__ uint8_t *trashBuf,
                                       __gm__ uint64_t *cycleOut)
{
    constexpr int kTileCols = (count <= 256) ? static_cast<int>(count) : 256;
    static_assert(count % kTileCols == 0, "count must be a multiple of kTileCols");
    using TileData = pto::Tile<pto::TileType::Vec, T, 1, kTileCols, pto::BLayout::RowMajor>;
    using ChunkShape = pto::Shape<1, 1, 1, 1, kTileCols>;
    using ChunkStride = pto::Stride<1, 1, 1, 1, 1>;

    if (!BoundsOkOrFinalize<count>(elem_count)) {
        return;
    }

    {
        uint64_t totalBytes = static_cast<uint64_t>(elem_count) * sizeof(T);
        __gm__ void *target =
            enablePrefetch ? reinterpret_cast<__gm__ void *>(srcBuf) : reinterpret_cast<__gm__ void *>(trashBuf);
        auto evt = pto::TPREFETCH_L2(target, totalBytes, sdmaWorkspace);
        (void)evt.Wait();
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
inline void FillAndUpload(SingleCardTestEnv &env, size_t count, int modulus)
{
    T *in = reinterpret_cast<T *>(env.inputHost);
    T *out = reinterpret_cast<T *>(env.outputHost);
    for (size_t i = 0; i < count; ++i) {
        in[i] = static_cast<T>(i % modulus);
        out[i] = static_cast<T>(-1);
    }
    env.aclStatus |= aclrtMemcpy(env.srcDevice, env.dataBytes, env.inputHost, env.dataBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    env.aclStatus |=
        aclrtMemcpy(env.dstDevice, env.dataBytes, env.outputHost, env.dataBytes, ACL_MEMCPY_HOST_TO_DEVICE);
}

template <typename T>
inline bool VerifyOutputAndPrint(const SingleCardTestEnv &env, size_t count, int modulus, const char *tag)
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

// ============================================================================
// L2 cold/warm sweep printer — shared between the TLOAD perf runners below.
// ============================================================================
struct L2PerfSummaryFmt {
    const char *title;
    const char *metric;
    size_t dataBytes;
    int kWarmup;
    int kMeasured;
    int tloadChunks;
};

inline void PrintL2ColdWarmSummary(const L2PerfSummaryFmt &fmt, uint64_t avgCold, uint64_t avgWarm)
{
    double speedup = (avgWarm > 0) ? static_cast<double>(avgCold) / static_cast<double>(avgWarm) : 0.0;

    std::cout << "\n================================================================" << std::endl;
    std::cout << "[PERF] " << fmt.title << std::endl;
    std::cout << "  Data size:      " << fmt.dataBytes << " bytes (" << fmt.dataBytes / 1024 << " KB)" << std::endl;
    std::cout << "  TLOAD chunks:   " << fmt.tloadChunks << std::endl;
    std::cout << "  Iterations:     " << fmt.kMeasured << " (warmup=" << fmt.kWarmup << ")" << std::endl;
    std::cout << "  L2-cold " << fmt.metric << ":  " << avgCold << " cycles (avg)" << std::endl;
    std::cout << "  L2-warm " << fmt.metric << ":  " << avgWarm << " cycles (avg)" << std::endl;
    std::cout << "  Speedup:        " << speedup << "x  ("
              << (speedup > 1.05 ? "PREFETCH HELPS" : (speedup < 0.95 ? "PREFETCH HURTS" : "NO SIGNIFICANT DIFF"))
              << ")" << std::endl;
    std::cout << "================================================================\n" << std::endl;
}

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

    BaselineKernel<T, count><<<1, nullptr, env.stream>>>(reinterpret_cast<T *>(env.srcDevice),
                                                         reinterpret_cast<T *>(env.dstDevice), static_cast<int>(count));
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
        reinterpret_cast<T *>(env.srcDevice), reinterpret_cast<T *>(env.dstDevice), static_cast<int>(count),
        reinterpret_cast<uint8_t *>(sdmaMgr.GetWorkspaceAddr()));
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
        reinterpret_cast<T *>(env.srcDevice), reinterpret_cast<T *>(env.dstDevice), static_cast<int>(count),
        reinterpret_cast<uint8_t *>(sdmaMgr.GetWorkspaceAddr()));
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
// Single-card TLOAD perf runner — one iteration.
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
    TloadPerfKernel<T, count><<<1, nullptr, stream>>>(reinterpret_cast<T *>(srcDevice), static_cast<int>(count),
                                                      enablePrefetch, (uint8_t *)sdmaMgr.GetWorkspaceAddr(),
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

    auto runOnce = [&](bool prefetch, uint64_t &cycles) {
        return RunTloadPerfOnce<T, count>(deviceId, prefetch, cycles);
    };
    auto printSummary = [&](uint64_t avgCold, uint64_t avgWarm) {
        L2PerfSummaryFmt fmt{};
        fmt.title = "TLOAD Latency: L2-cold vs L2-prefetched";
        fmt.metric = "TLOAD";
        fmt.dataBytes = count * sizeof(T);
        fmt.kWarmup = kWarmup;
        fmt.kMeasured = kMeasured;
        fmt.tloadChunks = static_cast<int>(count / ((count <= 256) ? count : 256));
        PrintL2ColdWarmSummary(fmt, avgCold, avgWarm);
    };
    return RunColdWarmPerfSweep(kWarmup, kMeasured, runOnce, printSummary);
}

template bool RunTloadPerf<float, 4096>(int);
template bool RunTloadPerf<float, 65536>(int);
template bool RunTloadPerf<float, 262144>(int);
template bool RunTloadPerf<float, 1048576>(int);
