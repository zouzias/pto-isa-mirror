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
// Single-card ST for device-side async L2 prefetch. This file purposely does
// NOT pull in HCCL or any comm/* host infrastructure beyond the host-side
// SdmaWorkspaceManager that allocates the SDMA workspace.
// ============================================================================

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <pto/pto-inst.hpp>
#include <pto/npu/kernels/Pto_prefetch.hpp>
#include "pto/npu/comm/async/sdma/sdma_workspace_manager.hpp"
#include "pto/common/pto_tile.hpp"

#include "tprefetch_async_kernel.h"

#define ENABLE_DEBUG_PRINT 0

using SdmaWorkspaceManager = pto::comm::sdma::SdmaWorkspaceManager;

// ============================================================================
// Kernel-wide type aliases -fully-dynamic 5-D Shape/Stride is the canonical
// shape for prefetch GlobalTensors so the same kernel handles every elem count.
// ============================================================================
using KernelShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
using KernelStrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
template <typename T>
using KernelGlobal = pto::GlobalTensor<T, KernelShapeDyn, KernelStrideDyn, pto::Layout::ND>;

// BoundsOkOrFinalize -range guard for elem_count against the compile-time
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
// Correctness kernel - public GlobalTensor API.
// ============================================================================
template <typename T, size_t count>
__global__ AICORE void TPrefetchAsyncCorrectnessKernel(__gm__ T *src, __gm__ T *dst, int elem_count,
                                                       __gm__ uint8_t *sdmaWorkspace)
{
    if (!BoundsOkOrFinalize<count>(elem_count)) {
        return;
    }

    KernelShapeDyn shape(1, 1, 1, 1, elem_count);
    KernelStrideDyn stride(elem_count, elem_count, elem_count, elem_count, 1);
    KernelGlobal<T> srcGlobal(src, shape, stride);
    pto::PrefetchAsyncContext ctx(sdmaWorkspace);

    auto evt = pto::TPREFETCH_ASYNC(srcGlobal, ctx);
    (void)evt.Wait(ctx.session);

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

// BISECT: Plan A -temporarily disable perf/multi-stage kernels and runners
// to check if the Bisheng optimizer ICE in tprefetch_async_kernel.cpp.o
// reproduces with only BaselineKernel + TPrefetchAsyncCorrectnessKernel.
// Restore by removing the surrounding `#if 0` once the trigger is known.
#if 0
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
        pto::PrefetchAsyncContext ctx(sdmaWorkspace);
        KernelShapeDyn shape(1, 1, 1, 1, static_cast<int>(totalBytes));
        KernelStrideDyn stride(static_cast<int>(totalBytes), static_cast<int>(totalBytes), static_cast<int>(totalBytes),
                               static_cast<int>(totalBytes), 1);
        __gm__ uint8_t *target = enablePrefetch ? reinterpret_cast<__gm__ uint8_t *>(srcBuf) : trashBuf;
        KernelGlobal<uint8_t> targetGlobal(target, shape, stride);
        auto evt = pto::TPREFETCH_ASYNC(targetGlobal, ctx);
        (void)evt.Wait(ctx.session);
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
#endif // BISECT: end of perf-kernel block 1

// ============================================================================
// Test helpers -shared setup/teardown/verify to cut duplication
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
    if (modulus <= 0) {
        env.aclStatus |= -1;
        return;
    }
    const size_t checkedModulus = static_cast<size_t>(modulus);
    T *in = reinterpret_cast<T *>(env.inputHost);
    T *out = reinterpret_cast<T *>(env.outputHost);
    for (size_t i = 0; i < count; ++i) {
        in[i] = static_cast<T>(i % checkedModulus);
        out[i] = static_cast<T>(-1);
    }
    env.aclStatus |= aclrtMemcpy(env.srcDevice, env.dataBytes, env.inputHost, env.dataBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    env.aclStatus |=
        aclrtMemcpy(env.dstDevice, env.dataBytes, env.outputHost, env.dataBytes, ACL_MEMCPY_HOST_TO_DEVICE);
}

template <typename T>
inline bool VerifyOutputAndPrint(const SingleCardTestEnv &env, size_t count, int modulus, const char *tag)
{
    if (modulus <= 0) {
        std::cout << tag << ": invalid modulus " << modulus << std::endl;
        return false;
    }
    const size_t checkedModulus = static_cast<size_t>(modulus);
    const T *out = reinterpret_cast<const T *>(env.outputHost);
    for (size_t i = 0; i < count; ++i) {
        T expected = static_cast<T>(i % checkedModulus);
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

#if 0 // BISECT: perf summary helpers (only used by perf runners below)
// ============================================================================
// L2 cold/warm sweep printer -shared between the TLOAD perf runners below.
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

    const bool perfOk = avgWarm < avgCold;
    std::cout << "\n[PERF] " << fmt.title << ", size=" << fmt.dataBytes / 1024 << "KB: cold=" << avgCold << " cycles"
              << ", prefetched=" << avgWarm << " cycles" << ", speedup=" << speedup << "x"
              << ", expect prefetched < cold: " << (perfOk ? "PASS" : "FAIL") << std::endl;
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
    if (avgWarm >= avgCold) {
        std::cerr << "[ERROR] L2 prefetch did not reduce TLOAD latency: cold=" << avgCold << " cycles, warm=" << avgWarm
                  << " cycles" << std::endl;
        return false;
    }
    return true;
}
#endif // BISECT: end of perf-summary block

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
bool RunPrefetchAsyncCorrectness(int deviceId)
{
    constexpr size_t dataBytes = count * sizeof(T);
    SingleCardTestEnv env;
    if (!env.Init(deviceId, dataBytes)) {
        std::cerr << "[ERROR] TPREFETCH_ASYNC: init failed!" << std::endl;
        return false;
    }
    FillAndUpload<T>(env, count, 1000);

    SdmaWorkspaceManager sdmaMgr;
    if (!sdmaMgr.Init()) {
        std::cerr << "[WARN] SdmaWorkspaceManager Init failed -prefetch will be skipped inside kernel" << std::endl;
    }

    TPrefetchAsyncCorrectnessKernel<T, count>
        <<<1, nullptr, env.stream>>>(reinterpret_cast<T *>(env.srcDevice), reinterpret_cast<T *>(env.dstDevice),
                                     static_cast<int>(count), reinterpret_cast<uint8_t *>(sdmaMgr.GetWorkspaceAddr()));
    env.SyncAndReadBack();

    bool is_ok = VerifyOutputAndPrint<T>(env, count, 1000, "TPREFETCH_ASYNC GlobalTensor correctness");
    env.Teardown();
    sdmaMgr.Finalize();
    return is_ok && (env.aclStatus == 0);
}

template bool RunBaseline<float, 4096>(int deviceId);
template bool RunBaseline<int32_t, 4096>(int deviceId);
template bool RunPrefetchAsyncCorrectness<float, 4096>(int deviceId);
template bool RunPrefetchAsyncCorrectness<int32_t, 4096>(int deviceId);

#if 0 // BISECT: all perf / multi-stage runners and instantiations
// ============================================================================
// Single-card TLOAD perf runner -one iteration.
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
        fmt.title = "TLOAD Latency: cold vs prefetched";
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
// ============================================================================
// Shared device-side aliases and helpers
// ============================================================================
using CmpShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
using CmpStrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
template <typename T>
using CmpGlobal = pto::GlobalTensor<T, CmpShapeDyn, CmpStrideDyn, pto::Layout::ND>;

// Cycle counter. Same asm pattern used elsewhere in the prefetch tests.
inline AICORE uint64_t cmp_syscnt()
{
    uint64_t syscnt;
    asm volatile("MOV %0, SYS_CNT\n" : "+l"(syscnt));
    return syscnt;
}

// Device-side scalar spin used as the "compute-A" workload in compute overlap path.
// A busy-loop on the scalar pipe keeps the AI Core occupied without using any
// MTE/SDMA pipe, so it faithfully isolates "compute that runs in parallel with
// the SDMA engine". Iteration count is chosen by the host at launch time.
PTO_INTERNAL void SpinCycles(uint64_t minCycles)
{
    uint64_t start = cmp_syscnt();
    // Empty-body loop: the compiler must treat cmp_syscnt() as a volatile asm
    // side-effect, so it cannot optimize this away.
    for (uint64_t now = cmp_syscnt(); (now - start) < minCycles; now = cmp_syscnt()) {
    }
}

// Standard "chunked TLOAD" loop used by every measurement kernel. Returns the
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

// ============================================================================
// Host/device prefetch kernels
// ============================================================================

// A baseline / A "host PTO_PREFETCH" -both use the same kernel. The host-side
// orchestration is what differs (whether aclrtCmoAsync was enqueued before
// launch). The kernel itself just runs the TLOAD sweep and reports cycles.
template <typename T, size_t count>
__global__ AICORE void HostDeviceTloadOnlyKernel(__gm__ T *srcBuf, int elem_count, __gm__ uint64_t *cycleOut)
{
    if (elem_count <= 0 || elem_count > static_cast<int>(count)) {
        pipe_barrier(PIPE_ALL);
        return;
    }
    *cycleOut = TloadSweepCycles<T, count>(srcBuf, elem_count);
    pipe_barrier(PIPE_ALL);
}

// A "device async prefetch" -prefetch inside the kernel via the workspace API,
// wait, then TLOAD. Kernel cycles here cover prefetch+wait+TLOAD, matching
// the end-to-end host timing we do around the kernel launch.
template <typename T, size_t count>
__global__ AICORE void HostDevicePrefetchKernel(__gm__ T *srcBuf, int elem_count, __gm__ uint8_t *sdmaWorkspace,
                                                __gm__ uint64_t *cycleOut)
{
    if (elem_count <= 0 || elem_count > static_cast<int>(count)) {
        pipe_barrier(PIPE_ALL);
        return;
    }

    uint64_t t0 = cmp_syscnt();

    {
        CmpShapeDyn shape(1, 1, 1, 1, elem_count);
        CmpStrideDyn stride(elem_count, elem_count, elem_count, elem_count, 1);
        CmpGlobal<T> srcGlobal(srcBuf, shape, stride);
        pto::PrefetchAsyncContext ctx(sdmaWorkspace);
        auto evt = pto::TPREFETCH_ASYNC(srcGlobal, ctx);
        (void)evt.Wait(ctx.session);
    }

    uint64_t tloadCycles = TloadSweepCycles<T, count>(srcBuf, elem_count);

    uint64_t t1 = cmp_syscnt();
    *cycleOut = t1 - t0;
    (void)tloadCycles;

    pipe_barrier(PIPE_ALL);
}

// Static and data-dependent tests share this compute + TLOAD stage kernel.
// compute-A (scalar spin) -> TLOAD sweep.
template <typename T, size_t count>
__global__ AICORE void ComputeThenTloadKernel(__gm__ T *srcBuf, int elem_count, uint64_t spinCycles,
                                              __gm__ uint64_t *cycleOut)
{
    if (elem_count <= 0 || elem_count > static_cast<int>(count)) {
        pipe_barrier(PIPE_ALL);
        return;
    }

    uint64_t t0 = cmp_syscnt();
    SpinCycles(spinCycles);
    uint64_t tloadCycles = TloadSweepCycles<T, count>(srcBuf, elem_count);
    uint64_t t1 = cmp_syscnt();

    *cycleOut = t1 - t0;
    (void)tloadCycles;
    pipe_barrier(PIPE_ALL);
}

// ============================================================================
// Static-address prefetch kernel -device-side fused multi-stage prefetch + compute,
// pipelined.
//
// Pattern: M stages, each = prefetch chunk + spin compute on chunk + warm
// TLOAD chunk. The pipeline overlaps prefetch_{i+1} with compute_i:
//
//   prefetch block_0 ; wait                    (first block has nothing to
//                                               overlap with)
//   for i in [0, M-1):
//       issue prefetch block_{i+1}             (async fire)
//       spin compute_i                         (AICORE busy here, SDMA
//                                               concurrently moving block_{i+1})
//       TLOAD block_i                          (warm; was prefetched in
//                                               previous iteration)
//       wait block_{i+1}                       (cheap if compute >= prefetch)
//   last stage (i = M-1):
//       spin compute_{M-1}
//       TLOAD block_{M-1}                      (was warmed by previous wait)
//
// Each (issue, Wait) pair is paired before the next issue, so the shared
// done flag in the workspace is consumed clean before reuse.
// ============================================================================
template <typename T, size_t chunkElems>
__global__ AICORE void StaticAddressDeviceFusedKernel(__gm__ T *srcBuf, uint64_t chunkBytes, uint32_t numStages,
                                                      uint64_t spinCycles, __gm__ uint8_t *sdmaWorkspace,
                                                      __gm__ uint64_t *cycleOut)
{
    if (numStages == 0) {
        pipe_barrier(PIPE_ALL);
        return;
    }

    uint64_t t0 = cmp_syscnt();

    __gm__ uint8_t *srcBytes = reinterpret_cast<__gm__ uint8_t *>(srcBuf);
    pto::PrefetchAsyncContext ctx(sdmaWorkspace);
    CmpShapeDyn chunkShape(1, 1, 1, 1, static_cast<int>(chunkElems));
    CmpStrideDyn chunkStride(static_cast<int>(chunkElems), static_cast<int>(chunkElems), static_cast<int>(chunkElems),
                             static_cast<int>(chunkElems), 1);

    // Step 1: prime the pipeline by prefetching block_0 + waiting.
    {
        CmpGlobal<T> firstGlobal(reinterpret_cast<__gm__ T *>(srcBytes), chunkShape, chunkStride);
        auto evt = pto::TPREFETCH_ASYNC(firstGlobal, ctx);
        (void)evt.Wait(ctx.session);
    }

    uint64_t tloadCyclesAcc = 0;

    // Pipelined body: stages 0..M-2 overlap prefetch_{i+1} with compute_i.
    for (uint32_t i = 0; i + 1 < numStages; ++i) {
        __gm__ uint8_t *nextAddr = srcBytes + static_cast<uint64_t>(i + 1) * chunkBytes;
        CmpGlobal<T> nextGlobal(reinterpret_cast<__gm__ T *>(nextAddr), chunkShape, chunkStride);
        auto evtNext = pto::TPREFETCH_ASYNC(nextGlobal, ctx);

        SpinCycles(spinCycles);

        __gm__ T *stageSrc = reinterpret_cast<__gm__ T *>(srcBytes + static_cast<uint64_t>(i) * chunkBytes);
        tloadCyclesAcc += TloadSweepCycles<T, chunkElems>(stageSrc, static_cast<int>(chunkElems));

        (void)evtNext.Wait(ctx.session);
    }

    // Last stage (i = M-1): compute + TLOAD on the already-warm block_{M-1}.
    {
        SpinCycles(spinCycles);
        __gm__ T *lastSrc = reinterpret_cast<__gm__ T *>(srcBytes + static_cast<uint64_t>(numStages - 1) * chunkBytes);
        tloadCyclesAcc += TloadSweepCycles<T, chunkElems>(lastSrc, static_cast<int>(chunkElems));
    }

    uint64_t t1 = cmp_syscnt();
    *cycleOut = t1 - t0;
    (void)tloadCyclesAcc;

    pipe_barrier(PIPE_ALL);
}

// ============================================================================
// Data-dependent prefetch: data-dependent prefetch (host forced to sync per stage)
// ----------------------------------------------------------------------------
// Two kernels to express the same semantic on host vs device:
//   * DataDependentHostStageKernel    - one stage of the host_dep_sync loop:
//                                    compute spin + warm TLOAD + write next
//                                    offset to a device slot. The host then
//                                    aclrtSynchronizeStream + aclrtMemcpy to
//                                    read the slot before issuing the next
//                                    PTO_PREFETCH.
//   * DataDependentDeviceFusedKernel  - one launch covering all M stages,
//                                    pipelined exactly like Static-address prefetch; the
//                                    "next offset" lives in an AICORE
//                                    register and is fed straight into the
//                                    next TPREFETCH_ASYNC with no host hop.
//
// Both use the trivial recurrence offset_{i+1} = (offset_i + 1) % modulus.
// The recurrence itself is cheap on either side; what differs is whether the
// value crosses the host-device boundary.
// ============================================================================
template <typename T, size_t count>
__global__ AICORE void DataDependentHostStageKernel(__gm__ T *srcBuf, int elem_count, uint64_t spinCycles,
                                                    uint32_t prevOffset, uint32_t modulus, __gm__ uint64_t *cycleOut,
                                                    __gm__ uint32_t *nextOffsetOut)
{
    if (elem_count <= 0 || elem_count > static_cast<int>(count) || modulus == 0) {
        pipe_barrier(PIPE_ALL);
        return;
    }

    uint64_t t0 = cmp_syscnt();
    SpinCycles(spinCycles);
    uint64_t tloadCycles = TloadSweepCycles<T, count>(srcBuf, elem_count);
    uint64_t t1 = cmp_syscnt();

    *cycleOut = t1 - t0;

    // Trivial recurrence -what matters is that the host has to read this
    // back from device memory before it can issue the next prefetch.
    *nextOffsetOut = (prevOffset + 1U) % modulus;

    (void)tloadCycles;
    pipe_barrier(PIPE_ALL);
}

template <typename T, size_t chunkElems>
__global__ AICORE void DataDependentDeviceFusedKernel(__gm__ T *srcBuf, uint64_t chunkBytes, uint32_t numStages,
                                                      uint32_t modulus, uint64_t spinCycles,
                                                      __gm__ uint8_t *sdmaWorkspace, __gm__ uint64_t *cycleOut)
{
    if (numStages == 0 || modulus == 0) {
        pipe_barrier(PIPE_ALL);
        return;
    }

    uint64_t t0 = cmp_syscnt();

    __gm__ uint8_t *srcBytes = reinterpret_cast<__gm__ uint8_t *>(srcBuf);
    uint32_t offset = 0U;
    pto::PrefetchAsyncContext ctx(sdmaWorkspace);
    CmpShapeDyn chunkShape(1, 1, 1, 1, static_cast<int>(chunkElems));
    CmpStrideDyn chunkStride(static_cast<int>(chunkElems), static_cast<int>(chunkElems), static_cast<int>(chunkElems),
                             static_cast<int>(chunkElems), 1);

    // Step 1: prime with prefetch of block_offset_0.
    {
        __gm__ uint8_t *firstAddr = srcBytes + static_cast<uint64_t>(offset) * chunkBytes;
        CmpGlobal<T> firstGlobal(reinterpret_cast<__gm__ T *>(firstAddr), chunkShape, chunkStride);
        auto evt = pto::TPREFETCH_ASYNC(firstGlobal, ctx);
        (void)evt.Wait(ctx.session);
    }

    uint64_t tloadCyclesAcc = 0;

    // Pipelined body: stages 0..M-2 overlap prefetch_{i+1} with compute_i.
    for (uint32_t i = 0; i + 1 < numStages; ++i) {
        // Compute the next offset in-register; no host roundtrip needed.
        uint32_t nextOffset = (offset + 1U) % modulus;

        __gm__ uint8_t *nextAddr = srcBytes + static_cast<uint64_t>(nextOffset) * chunkBytes;
        CmpGlobal<T> nextGlobal(reinterpret_cast<__gm__ T *>(nextAddr), chunkShape, chunkStride);
        auto evtNext = pto::TPREFETCH_ASYNC(nextGlobal, ctx);

        SpinCycles(spinCycles);

        __gm__ T *stageSrc = reinterpret_cast<__gm__ T *>(srcBytes + static_cast<uint64_t>(offset) * chunkBytes);
        tloadCyclesAcc += TloadSweepCycles<T, chunkElems>(stageSrc, static_cast<int>(chunkElems));

        (void)evtNext.Wait(ctx.session);
        offset = nextOffset;
    }

    // Last stage.
    {
        SpinCycles(spinCycles);
        __gm__ T *lastSrc = reinterpret_cast<__gm__ T *>(srcBytes + static_cast<uint64_t>(offset) * chunkBytes);
        tloadCyclesAcc += TloadSweepCycles<T, chunkElems>(lastSrc, static_cast<int>(chunkElems));
    }

    uint64_t t1 = cmp_syscnt();
    *cycleOut = t1 - t0;
    (void)tloadCyclesAcc;

    pipe_barrier(PIPE_ALL);
}

// Explicit instantiations for the three sizes listed in main.cpp.
// float/int32 aliasing doesn't matter here since the kernels only TLOAD/write
// cycle counters; we test float throughout.
#define CMP_INSTANTIATE(T, COUNT)                                                                                      \
    template __global__ AICORE void ComputeThenTloadKernel<T, COUNT>(__gm__ T *, int, uint64_t, __gm__ uint64_t *);    \
    template __global__ AICORE void StaticAddressDeviceFusedKernel<T, COUNT>(__gm__ T *, uint64_t, uint32_t, uint64_t, \
                                                                             __gm__ uint8_t *, __gm__ uint64_t *);     \
    template __global__ AICORE void DataDependentHostStageKernel<T, COUNT>(                                            \
        __gm__ T *, int, uint64_t, uint32_t, uint32_t, __gm__ uint64_t *, __gm__ uint32_t *);                          \
    template __global__ AICORE void DataDependentDeviceFusedKernel<T, COUNT>(                                          \
        __gm__ T *, uint64_t, uint32_t, uint32_t, uint64_t, __gm__ uint8_t *, __gm__ uint64_t *)

CMP_INSTANTIATE(float, 16384);    // 64 KB
CMP_INSTANTIATE(float, 262144);   // 1 MB
CMP_INSTANTIATE(float, 4194304);  // 16 MB
CMP_INSTANTIATE(float, 16777216); // 64 MB
CMP_INSTANTIATE(float, 33554432); // 128 MB
#undef CMP_INSTANTIATE

// ============================================================================
// Host-side: shared setup/teardown
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

// ============================================================================
// Statistics helper -collect samples and report p50 for stable pass/fail checks.
// ============================================================================
struct SampleSet {
    std::vector<double> samples;

    void Add(double v)
    {
        samples.push_back(v);
    }

    double Percentile(double p) const
    {
        if (samples.empty()) {
            return 0.0;
        }
        std::vector<double> sorted = samples;
        std::sort(sorted.begin(), sorted.end());
        size_t idx = static_cast<size_t>(p / 100.0 * static_cast<double>(sorted.size()));
        if (idx >= sorted.size()) {
            idx = sorted.size() - 1;
        }
        return sorted[idx];
    }

    double P50() const
    {
        return Percentile(50.0);
    }
};

// Allow CI to dial iteration counts down for smoke runs while keeping the
// "real" (shmem-style) 100-iter default for serious measurements. Callers
// pass their default; env var TPREFETCH_ASYNC_ITER overrides.
inline int IterCount(int defaultIter)
{
    const char *env = std::getenv("TPREFETCH_ASYNC_ITER");
    if (env != nullptr && env[0] != '\0') {
        int v = std::atoi(env);
        if (v > 0) {
            return v;
        }
    }
    return defaultIter;
}

inline std::string BytesLabel(uint64_t bytes)
{
    std::ostringstream oss;
    if (bytes >= 1024ULL * 1024) {
        oss << bytes / 1024 / 1024 << "MB";
    } else if (bytes >= 1024) {
        oss << bytes / 1024 << "KB";
    } else {
        oss << bytes << "B";
    }
    return oss.str();
}

struct StagePerfSamples {
    SampleSet hostWall;
    SampleSet deviceWall;
    SampleSet hostKern;
    SampleSet deviceKern;
};

inline void AddStagePerfSample(StagePerfSamples &samples, int iter, int warmup, uint64_t hostWallUs,
                               uint64_t deviceWallUs, uint64_t hostKernCycles, uint64_t deviceKernCycles)
{
    if (iter < warmup) {
        return;
    }
    samples.hostWall.Add(static_cast<double>(hostWallUs));
    samples.deviceWall.Add(static_cast<double>(deviceWallUs));
    samples.hostKern.Add(CyclesToUs(hostKernCycles));
    samples.deviceKern.Add(CyclesToUs(deviceKernCycles));
}

struct CmpEnv {
    aclrtStream stream = nullptr;
    void *srcDevice = nullptr;   // real source buffer being prefetched / TLOADed
    void *trashDevice = nullptr; // L2-evicting buffer (>= kL2TrashBytes)
    void *cycleDev = nullptr;    // uint64_t slot for in-kernel cycle counter
    SdmaWorkspaceManager sdmaMgr;
    size_t dataBytes = 0;
    int aclStatus = 0;

    bool Init(int deviceId, size_t bytes)
    {
        dataBytes = bytes;
        aclStatus |= aclrtSetDevice(deviceId);
        aclStatus |= aclrtCreateStream(&stream);
        aclStatus |= aclrtMalloc(&srcDevice, dataBytes, ACL_MEM_MALLOC_HUGE_FIRST);
        aclStatus |= aclrtMalloc(&trashDevice, kL2TrashBytes, ACL_MEM_MALLOC_HUGE_FIRST);
        aclStatus |= aclrtMalloc(&cycleDev, sizeof(uint64_t), ACL_MEM_MALLOC_HUGE_FIRST);
        uint64_t zero = 0;
        aclStatus |= aclrtMemcpy(cycleDev, sizeof(uint64_t), &zero, sizeof(uint64_t), ACL_MEMCPY_HOST_TO_DEVICE);
        if (!sdmaMgr.Init()) {
            std::cerr << "[ERROR] CmpEnv: SdmaWorkspaceManager Init failed!" << std::endl;
            aclStatus |= -1;
        }
        return aclStatus == 0;
    }
    void Teardown()
    {
        sdmaMgr.Finalize();
        aclrtFree(cycleDev);
        aclrtFree(trashDevice);
        aclrtFree(srcDevice);
        aclrtDestroyStream(stream);
    }
    // Evict the L2 before each measured run. aclrtCmoAsync + sync is the
    // host-side equivalent of "rewrite every L2 line with unrelated data".
    void TrashL2()
    {
        pto::PTO_PREFETCH(trashDevice, kL2TrashBytes, stream);
        aclStatus |= aclrtSynchronizeStream(stream);
    }
    uint64_t ReadCycles()
    {
        uint64_t c = 0;
        aclrtMemcpy(&c, sizeof(uint64_t), cycleDev, sizeof(uint64_t), ACL_MEMCPY_DEVICE_TO_HOST);
        return c;
    }
};

using HrClock = std::chrono::steady_clock;

inline uint64_t ElapsedMicros(HrClock::time_point t0, HrClock::time_point t1)
{
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count());
}

// ============================================================================
// Host/device prefetch: end-to-end wall-clock latency
//
// The "real-world" question this measurement answers: starting from a cold L2,
// how long does it take from issuing a prefetch to having the warm TLOAD
// complete? Three configurations measured at each size:
//
//   * baseline      - trash L2, then kernel(cold TLOAD only)
//   * host prefetch - trash L2, then host PTO_PREFETCH on stream + kernel(warm TLOAD)
//   * device async  - trash L2, then kernel(async prefetch -> Wait -> warm TLOAD)
//
// All three measurements time the entire host-visible window (start of first
// op after trash to aclrtSynchronizeStream returns), so they are apples-to-
// apples wall-clock comparable.
//
// Collect kIter samples per config and use p50 for the pass/fail check.
// 1 warmup precedes the measured samples.
// ============================================================================

// Single measurement of one (mode) pair. Returns wall-clock us.
template <typename T, size_t count>
inline bool RunAOne(CmpEnv &env, int mode, uint64_t &wallUs, uint64_t &kernCycles)
{
    // mode: 0 = baseline, 1 = host SDMA prefetch, 2 = device async prefetch
    env.TrashL2();

    auto t0 = HrClock::now();
    switch (mode) {
        case 1:
            pto::PTO_PREFETCH(env.srcDevice, env.dataBytes, env.stream);
            // fall through to kernel launch: stream FIFO guarantees prefetch
            // completes before the kernel starts (this is the defining
            // property of host prefetch).
            [[fallthrough]];
        case 0:
            HostDeviceTloadOnlyKernel<T, count><<<1, nullptr, env.stream>>>(reinterpret_cast<T *>(env.srcDevice),
                                                                            static_cast<int>(count),
                                                                            reinterpret_cast<uint64_t *>(env.cycleDev));
            break;
        case 2:
            HostDevicePrefetchKernel<T, count>
                <<<1, nullptr, env.stream>>>(reinterpret_cast<T *>(env.srcDevice), static_cast<int>(count),
                                             reinterpret_cast<uint8_t *>(env.sdmaMgr.GetWorkspaceAddr()),
                                             reinterpret_cast<uint64_t *>(env.cycleDev));
            break;
        default:
            return false;
    }
    env.aclStatus |= aclrtSynchronizeStream(env.stream);
    auto t1 = HrClock::now();

    wallUs = ElapsedMicros(t0, t1);
    kernCycles = env.ReadCycles();
    return env.aclStatus == 0;
}

template <typename T, size_t count>
inline bool CollectHostDeviceSamples(CmpEnv &env, int mode, int warmup, int iter, SampleSet &wallSet,
                                     SampleSet &kernSet)
{
    for (int i = 0; i < warmup + iter; ++i) {
        uint64_t wall = 0;
        uint64_t kern = 0;
        if (!RunAOne<T, count>(env, mode, wall, kern)) {
            return false;
        }
        if (i >= warmup) {
            wallSet.Add(static_cast<double>(wall));
            kernSet.Add(CyclesToUs(kern));
        }
    }
    return true;
}

inline bool PrintHostDeviceSummary(size_t dataBytes, const SampleSet &wallBase, const SampleSet &wallHost,
                                   const SampleSet &wallDev)
{
    const bool perfOk = wallHost.P50() < wallBase.P50() && wallDev.P50() < wallBase.P50();
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "\n[PERF] host/device prefetch, size=" << BytesLabel(dataBytes) << ": no_prefetch=" << wallBase.P50()
              << "us" << ", host_prefetch=" << wallHost.P50() << "us" << ", device_prefetch=" << wallDev.P50() << "us"
              << ", expect host/device < no_prefetch: " << (perfOk ? "PASS" : "FAIL") << std::endl;
    if (!perfOk) {
        std::cerr << "[ERROR] prefetch did not reduce end-to-end latency" << std::endl;
    }
    return perfOk;
}

template <typename T, size_t count>
bool RunHostDevicePrefetch(int deviceId)
{
    constexpr size_t dataBytes = count * sizeof(T);
    constexpr int kWarmup = 1;
    const int kIter = IterCount(100);

    CmpEnv env;
    if (!env.Init(deviceId, dataBytes)) {
        std::cerr << "[ERROR] host_device: env init failed" << std::endl;
        env.Teardown();
        return false;
    }

    SampleSet wallBase;
    SampleSet wallHost;
    SampleSet wallDev;
    SampleSet kernBase;
    SampleSet kernHost;
    SampleSet kernDev;

    bool ok = CollectHostDeviceSamples<T, count>(env, 0, kWarmup, kIter, wallBase, kernBase) &&
              CollectHostDeviceSamples<T, count>(env, 1, kWarmup, kIter, wallHost, kernHost) &&
              CollectHostDeviceSamples<T, count>(env, 2, kWarmup, kIter, wallDev, kernDev);
    env.Teardown();
    if (!ok) {
        return false;
    }

    const bool perfOk = PrintHostDeviceSummary(dataBytes, wallBase, wallHost, wallDev);
    return env.aclStatus == 0 && perfOk;
}

template bool RunHostDevicePrefetch<float, 16384>(int);    // 64 KB
template bool RunHostDevicePrefetch<float, 262144>(int);   // 1 MB
template bool RunHostDevicePrefetch<float, 4194304>(int);  // 16 MB
template bool RunHostDevicePrefetch<float, 16777216>(int); // 64 MB
template bool RunHostDevicePrefetch<float, 33554432>(int); // 128 MB

// ============================================================================
// Static-address prefetch: fused multi-stage prefetch + compute
//
// Runs M consecutive stages, each = (prefetch chunk_i) + (compute spinCycles
// on chunk_i) + (warm TLOAD chunk_i).
//
//   host_serial   : trash L2 -> for i in [0, M):
//                                   pto::PTO_PREFETCH(src + i*chunk, chunk, stream)
//                                   ComputeThenTloadKernel<<<...stream>>>
//                  -> 1 x aclrtSynchronizeStream
//                  M kernel launches; no prefetch-compute overlap on a single
//                  stream.
//   device_pipelined : trash L2 -> StaticAddressDeviceFusedKernel<<<1, ...>>>(...)
//                                  (loops M stages, overlapping prefetch_{i+1}
//                                   with compute_i)
//                      -> 1 x aclrtSynchronizeStream
//                      ONE launch shell; in-kernel pipeline hides prefetch
//                      under compute when spin >= prefetch.
//
// chunkBytes = chunkElems * sizeof(T) is held constant per test (= 1 MB by
// default for the chunkElems=262144 instantiation). Total bytes prefetched
// = M * chunkBytes. The per-stage compute body inside both paths is
// bit-identical (ComputeThenTloadKernel vs the same TloadSweep +
// SpinCycles primitives in StaticAddressDeviceFusedKernel).
// ============================================================================
template <typename T, size_t chunkElems>
inline bool RunStaticAddressOne(CmpEnv &env, uint32_t numStages, uint64_t chunkBytes, uint64_t spinCycles,
                                uint64_t &hostWallUs, uint64_t &deviceWallUs, uint64_t &hostKernCycles,
                                uint64_t &deviceKernCycles)
{
    env.TrashL2();
    auto h0 = HrClock::now();
    for (uint32_t s = 0; s < numStages; ++s) {
        uint8_t *addr = static_cast<uint8_t *>(env.srcDevice) + static_cast<uint64_t>(s) * chunkBytes;
        pto::PTO_PREFETCH(addr, chunkBytes, env.stream);
        T *stageSrc = reinterpret_cast<T *>(addr);
        ComputeThenTloadKernel<T, chunkElems><<<1, nullptr, env.stream>>>(
            stageSrc, static_cast<int>(chunkElems), spinCycles, reinterpret_cast<uint64_t *>(env.cycleDev));
    }
    env.aclStatus |= aclrtSynchronizeStream(env.stream);
    hostWallUs = ElapsedMicros(h0, HrClock::now());
    hostKernCycles = env.ReadCycles();

    env.TrashL2();
    auto d0 = HrClock::now();
    StaticAddressDeviceFusedKernel<T, chunkElems><<<1, nullptr, env.stream>>>(
        reinterpret_cast<T *>(env.srcDevice), chunkBytes, numStages, spinCycles,
        reinterpret_cast<uint8_t *>(env.sdmaMgr.GetWorkspaceAddr()), reinterpret_cast<uint64_t *>(env.cycleDev));
    env.aclStatus |= aclrtSynchronizeStream(env.stream);
    deviceWallUs = ElapsedMicros(d0, HrClock::now());
    deviceKernCycles = env.ReadCycles();
    return env.aclStatus == 0;
}

inline bool PrintStaticAddressSummary(uint32_t numStages, uint64_t chunkBytes, uint64_t totalBytes,
                                      const StagePerfSamples &samples)
{
    const bool perfOk = samples.deviceWall.P50() <= samples.hostWall.P50() * 1.25;
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "\n[PERF] static-address prefetch" << " (M=" << numStages << ", chunk=" << BytesLabel(chunkBytes)
              << ", total=" << BytesLabel(totalBytes) << "): host=" << samples.hostWall.P50() << "us"
              << ", device=" << samples.deviceWall.P50() << "us"
              << ", expect device <= 1.25x host: " << (perfOk ? "PASS" : "FAIL") << std::endl;
    if (!perfOk) {
        std::cerr << "[ERROR] static-address device path is unexpectedly slower than host path" << std::endl;
    }
    return perfOk;
}

template <typename T, size_t chunkElems>
bool RunStaticAddressPrefetch(int deviceId, uint32_t numStages, uint64_t spinCycles)
{
    if (numStages == 0) {
        std::cerr << "[ERROR] static_address: numStages must be > 0" << std::endl;
        return false;
    }
    constexpr uint64_t chunkBytes = static_cast<uint64_t>(chunkElems) * sizeof(T);
    const uint64_t totalBytes = chunkBytes * static_cast<uint64_t>(numStages);
    constexpr int kWarmup = 1;
    const int kIter = IterCount(100);

    CmpEnv env;
    if (!env.Init(deviceId, totalBytes)) {
        std::cerr << "[ERROR] static_address: env init failed (totalBytes=" << totalBytes << ")" << std::endl;
        env.Teardown();
        return false;
    }

    StagePerfSamples samples;

    for (int i = 0; i < kWarmup + kIter; ++i) {
        uint64_t hostWallUs = 0;
        uint64_t deviceWallUs = 0;
        uint64_t hostKernCycles = 0;
        uint64_t deviceKernCycles = 0;
        if (!RunStaticAddressOne<T, chunkElems>(env, numStages, chunkBytes, spinCycles, hostWallUs, deviceWallUs,
                                                hostKernCycles, deviceKernCycles)) {
            break;
        }
        AddStagePerfSample(samples, i, kWarmup, hostWallUs, deviceWallUs, hostKernCycles, deviceKernCycles);
    }

    env.Teardown();

    const bool perfOk = PrintStaticAddressSummary(numStages, chunkBytes, totalBytes, samples);
    return env.aclStatus == 0 && perfOk;
}

template bool RunStaticAddressPrefetch<float, 262144>(int, uint32_t, uint64_t); // chunk = 1 MB

// ============================================================================
// Data-dependent prefetch: data-dependent prefetch (host forced to sync per stage)
// ----------------------------------------------------------------------------
// Same total / per-stage primitives as Static-address prefetch, but the address of the
// next prefetch is the previous stage's device-side output -so the host
// has to aclrtSynchronizeStream + aclrtMemcpy(D2H) one uint32_t between
// every pair of stages before it can enqueue the next prefetch.
//
//   host_dep_sync :
//       offsets[0] = 0
//       trash L2
//       for i in [0, M):
//           addr = src + offsets[i] * chunkBytes
//           pto::PTO_PREFETCH(addr, chunkBytes, stream)
//           DataDependentHostStageKernel<<<...stream>>>(addr, ..., offsets[i],
//                                                     &deviceOffsets[i])
//           if i < M-1:
//               aclrtSynchronizeStream(stream)
//               aclrtMemcpy(&offsets[i+1], &deviceOffsets[i], 4, D2H)
//       aclrtSynchronizeStream(stream)
//
//   device_in_reg :
//       trash L2
//       DataDependentDeviceFusedKernel<<<1, ...>>>(src, chunk, M, M, spin, ws, ...)
//       aclrtSynchronizeStream(stream)
//
// chunkBytes = chunkElems * sizeof(T) is held constant per test (= 1 MB by
// default for chunkElems=262144). Total bytes prefetched = M * chunkBytes,
// matching Static-address prefetch for any (M, spin) pair so cross-result diffing
// isolates exactly the (M-1) forced syncs.
// ============================================================================
template <typename T, size_t chunkElems>
inline bool RunDataDependentOne(CmpEnv &env, void *deviceOffsets, std::vector<uint32_t> &hostOffsets,
                                uint32_t numStages, uint64_t chunkBytes, uint64_t spinCycles, uint64_t &hostWallUs,
                                uint64_t &deviceWallUs, uint64_t &hostKernCycles, uint64_t &deviceKernCycles)
{
    const uint32_t modulus = numStages;
    env.TrashL2();
    std::fill(hostOffsets.begin(), hostOffsets.end(), 0U);
    auto h0 = HrClock::now();
    for (uint32_t s = 0; s < numStages; ++s) {
        uint32_t offset = hostOffsets[s];
        uint8_t *addr = static_cast<uint8_t *>(env.srcDevice) + static_cast<uint64_t>(offset) * chunkBytes;
        pto::PTO_PREFETCH(addr, chunkBytes, env.stream);
        uint32_t *devOffsetSlot = reinterpret_cast<uint32_t *>(deviceOffsets) + s;
        DataDependentHostStageKernel<T, chunkElems>
            <<<1, nullptr, env.stream>>>(reinterpret_cast<T *>(addr), static_cast<int>(chunkElems), spinCycles, offset,
                                         modulus, reinterpret_cast<uint64_t *>(env.cycleDev), devOffsetSlot);
        if (s + 1 < numStages) {
            env.aclStatus |= aclrtSynchronizeStream(env.stream);
            env.aclStatus |= aclrtMemcpy(&hostOffsets[s + 1], sizeof(uint32_t), devOffsetSlot, sizeof(uint32_t),
                                         ACL_MEMCPY_DEVICE_TO_HOST);
        }
    }
    env.aclStatus |= aclrtSynchronizeStream(env.stream);
    hostWallUs = ElapsedMicros(h0, HrClock::now());
    hostKernCycles = env.ReadCycles();

    env.TrashL2();
    auto d0 = HrClock::now();
    DataDependentDeviceFusedKernel<T, chunkElems><<<1, nullptr, env.stream>>>(
        reinterpret_cast<T *>(env.srcDevice), chunkBytes, numStages, modulus, spinCycles,
        reinterpret_cast<uint8_t *>(env.sdmaMgr.GetWorkspaceAddr()), reinterpret_cast<uint64_t *>(env.cycleDev));
    env.aclStatus |= aclrtSynchronizeStream(env.stream);
    deviceWallUs = ElapsedMicros(d0, HrClock::now());
    deviceKernCycles = env.ReadCycles();
    return env.aclStatus == 0;
}

inline bool PrintDataDependentSummary(uint32_t numStages, uint64_t chunkBytes, uint64_t totalBytes,
                                      const StagePerfSamples &samples)
{
    const bool perfOk = samples.deviceWall.P50() < samples.hostWall.P50();
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "\n[PERF] data-dependent prefetch" << " (M=" << numStages << ", chunk=" << BytesLabel(chunkBytes)
              << ", total=" << BytesLabel(totalBytes) << "): host_sync=" << samples.hostWall.P50() << "us"
              << ", device_in_reg=" << samples.deviceWall.P50() << "us"
              << ", expect device < host: " << (perfOk ? "PASS" : "FAIL") << std::endl;
    if (!perfOk) {
        std::cerr << "[ERROR] data-dependent device path did not beat host sync path" << std::endl;
    }
    return perfOk;
}

template <typename T, size_t chunkElems>
bool RunDataDependentPrefetch(int deviceId, uint32_t numStages, uint64_t spinCycles)
{
    if (numStages == 0) {
        std::cerr << "[ERROR] data_dependent: numStages must be > 0" << std::endl;
        return false;
    }
    constexpr uint64_t chunkBytes = static_cast<uint64_t>(chunkElems) * sizeof(T);
    const uint64_t totalBytes = chunkBytes * static_cast<uint64_t>(numStages);
    constexpr int kWarmup = 1;
    const int kIter = IterCount(100);

    CmpEnv env;
    if (!env.Init(deviceId, totalBytes)) {
        std::cerr << "[ERROR] data_dependent: env init failed (totalBytes=" << totalBytes << ")" << std::endl;
        env.Teardown();
        return false;
    }

    // Per-stage device slots holding "next offset" written by each host stage.
    void *deviceOffsets = nullptr;
    int allocStatus = aclrtMalloc(&deviceOffsets, sizeof(uint32_t) * numStages, ACL_MEM_MALLOC_HUGE_FIRST);
    if (allocStatus != 0 || deviceOffsets == nullptr) {
        std::cerr << "[ERROR] data_dependent: aclrtMalloc(deviceOffsets) failed" << std::endl;
        env.Teardown();
        return false;
    }
    {
        // Initialise to a known value so the first read after kernel run is
        // always meaningful.
        std::vector<uint32_t> zeros(numStages, 0U);
        env.aclStatus |= aclrtMemcpy(deviceOffsets, sizeof(uint32_t) * numStages, zeros.data(),
                                     sizeof(uint32_t) * numStages, ACL_MEMCPY_HOST_TO_DEVICE);
    }
    std::vector<uint32_t> hostOffsets(numStages, 0U);

    StagePerfSamples samples;

    for (int i = 0; i < kWarmup + kIter; ++i) {
        uint64_t hostWallUs = 0;
        uint64_t deviceWallUs = 0;
        uint64_t hostKernCycles = 0;
        uint64_t deviceKernCycles = 0;
        if (!RunDataDependentOne<T, chunkElems>(env, deviceOffsets, hostOffsets, numStages, chunkBytes, spinCycles,
                                                hostWallUs, deviceWallUs, hostKernCycles, deviceKernCycles)) {
            break;
        }
        AddStagePerfSample(samples, i, kWarmup, hostWallUs, deviceWallUs, hostKernCycles, deviceKernCycles);
    }

    aclrtFree(deviceOffsets);
    env.Teardown();

    const bool perfOk = PrintDataDependentSummary(numStages, chunkBytes, totalBytes, samples);
    return env.aclStatus == 0 && perfOk;
}

template bool RunDataDependentPrefetch<float, 262144>(int, uint32_t, uint64_t); // chunk = 1 MB
#endif // BISECT: end of perf / multi-stage block
