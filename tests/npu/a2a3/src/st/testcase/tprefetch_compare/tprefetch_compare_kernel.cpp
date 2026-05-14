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
// Single-card performance comparison: host-initiated pto::PTO_PREFETCH (SDMA
// path via aclrtCmoAsync) vs device-initiated pto::TPREFETCH_L2 (SDMA CMO SQE
// issued by the AI Core).
//
// Three scenarios are covered; see the header for full description.
//   A  End-to-end wall-clock latency for "prefetch then TLOAD".
//   B  Issue overhead for a tiny (few KB) prefetch.
//   C  Overlap with AI-Core compute.
//
// Cross-rank coverage (Scenario D, receiver-side prefetch over TPUT_ASYNC)
// lives under tests/npu/a2a3/comm/st/testcase/tprefetch_compare/ where the
// HCCL test scaffold is available.
// ============================================================================

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>

#include <pto/pto-inst.hpp>
#include <pto/npu/kernels/Pto_prefetch.hpp>
#include "pto/npu/comm/async/sdma/sdma_workspace_manager.hpp"
#include "pto/common/pto_tile.hpp"

#include "tprefetch_compare_kernel.h"

using SdmaWorkspaceManager = pto::comm::sdma::SdmaWorkspaceManager;

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

// Device-side scalar spin used as the "compute-A" workload in scenario C.
// A busy-loop on the scalar pipe keeps the AI Core occupied without using any
// MTE/SDMA pipe, so it faithfully isolates "compute that runs in parallel with
// the SDMA engine". Iteration count is chosen by the host at launch time.
PTO_INTERNAL void SpinCycles(uint64_t minCycles)
{
    uint64_t start = cmp_syscnt();
    // Empty-body loop: the compiler must treat cmp_syscnt() as a volatile asm
    // side-effect, so it cannot optimize this away.
    while ((cmp_syscnt() - start) < minCycles) {
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
// Scenario A kernels
// ============================================================================

// A baseline / A "host PTO_PREFETCH" — both use the same kernel. The host-side
// orchestration is what differs (whether aclrtCmoAsync was enqueued before
// launch). The kernel itself just runs the TLOAD sweep and reports cycles.
template <typename T, size_t count>
__global__ AICORE void ScenarioA_TloadOnlyKernel(__gm__ T *srcBuf, int elem_count, __gm__ uint64_t *cycleOut)
{
    if (elem_count <= 0 || elem_count > static_cast<int>(count)) {
        pipe_barrier(PIPE_ALL);
        return;
    }
    *cycleOut = TloadSweepCycles<T, count>(srcBuf, elem_count);
    pipe_barrier(PIPE_ALL);
}

// A "device TPREFETCH_L2" — prefetch inside the kernel via the workspace API,
// wait, then TLOAD. Kernel cycles here cover prefetch+wait+TLOAD, matching
// the end-to-end host timing we do around the kernel launch.
template <typename T, size_t count>
__global__ AICORE void ScenarioA_DevicePrefetchKernel(__gm__ T *srcBuf, int elem_count, __gm__ uint8_t *sdmaWorkspace,
                                                      __gm__ uint64_t *cycleOut)
{
    if (elem_count <= 0 || elem_count > static_cast<int>(count)) {
        pipe_barrier(PIPE_ALL);
        return;
    }

    uint64_t t0 = cmp_syscnt();

    {
        uint64_t totalBytes = static_cast<uint64_t>(elem_count) * sizeof(T);
        auto evt = pto::TPREFETCH_L2(reinterpret_cast<__gm__ void *>(srcBuf), totalBytes, sdmaWorkspace);
        (void)evt.Wait();
    }

    uint64_t tloadCycles = TloadSweepCycles<T, count>(srcBuf, elem_count);

    uint64_t t1 = cmp_syscnt();
    *cycleOut = t1 - t0;
    (void)tloadCycles;

    pipe_barrier(PIPE_ALL);
}

// ============================================================================
// Scenario B kernel — device-side issue+wait cycle measurement for a tiny
// TPREFETCH_L2. The host side measures its own wall-clock around
// PTO_PREFETCH+sync in the runner; we don't need a dedicated host-kernel for
// that path.
// ============================================================================
template <typename T, size_t count>
__global__ AICORE void ScenarioB_DeviceIssueKernel(__gm__ T *srcBuf, int elem_count, __gm__ uint8_t *sdmaWorkspace,
                                                   __gm__ uint64_t *cycleOut)
{
    if (elem_count <= 0 || elem_count > static_cast<int>(count)) {
        pipe_barrier(PIPE_ALL);
        return;
    }

    uint64_t t0 = cmp_syscnt();
    uint64_t totalBytes = static_cast<uint64_t>(elem_count) * sizeof(T);
    auto evt = pto::TPREFETCH_L2(reinterpret_cast<__gm__ void *>(srcBuf), totalBytes, sdmaWorkspace);
    (void)evt.Wait();
    uint64_t t1 = cmp_syscnt();

    *cycleOut = t1 - t0;
    pipe_barrier(PIPE_ALL);
}

// ============================================================================
// Scenario C kernels — overlap with AI-Core compute
// ============================================================================

// C0 / C1 (no-prefetch and host-prefetch share the same kernel body):
// compute-A (scalar spin) -> TLOAD sweep.
// The distinction between C0 and C1 lives entirely in the host runner (whether
// a PTO_PREFETCH was enqueued on the stream before this kernel).
template <typename T, size_t count>
__global__ AICORE void ScenarioC_ComputeThenTloadKernel(__gm__ T *srcBuf, int elem_count, uint64_t spinCycles,
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

// C2: issue device TPREFETCH_L2, then run compute-A in parallel with SDMA,
// then Wait and TLOAD. Observable win: kernel cycles ~= max(compute-A, prefetch)
// + warm TLOAD, i.e. prefetch cost is hidden behind compute-A.
template <typename T, size_t count>
__global__ AICORE void ScenarioC_DeviceOverlapKernel(__gm__ T *srcBuf, int elem_count, uint64_t spinCycles,
                                                     __gm__ uint8_t *sdmaWorkspace, __gm__ uint64_t *cycleOut)
{
    if (elem_count <= 0 || elem_count > static_cast<int>(count)) {
        pipe_barrier(PIPE_ALL);
        return;
    }

    uint64_t t0 = cmp_syscnt();

    uint64_t totalBytes = static_cast<uint64_t>(elem_count) * sizeof(T);
    auto evt = pto::TPREFETCH_L2(reinterpret_cast<__gm__ void *>(srcBuf), totalBytes, sdmaWorkspace);

    SpinCycles(spinCycles);

    (void)evt.Wait();

    uint64_t tloadCycles = TloadSweepCycles<T, count>(srcBuf, elem_count);
    uint64_t t1 = cmp_syscnt();

    *cycleOut = t1 - t0;
    (void)tloadCycles;
    pipe_barrier(PIPE_ALL);
}

// Explicit instantiations for the three sizes listed in main.cpp.
// float/int32 aliasing doesn't matter here since the kernels only TLOAD/write
// cycle counters; we test float throughout.
#define CMP_INSTANTIATE(T, COUNT)                                                                                      \
    template __global__ AICORE void ScenarioA_TloadOnlyKernel<T, COUNT>(__gm__ T *, int, __gm__ uint64_t *);           \
    template __global__ AICORE void ScenarioA_DevicePrefetchKernel<T, COUNT>(__gm__ T *, int, __gm__ uint8_t *,        \
                                                                              __gm__ uint64_t *);                      \
    template __global__ AICORE void ScenarioB_DeviceIssueKernel<T, COUNT>(__gm__ T *, int, __gm__ uint8_t *,           \
                                                                           __gm__ uint64_t *);                         \
    template __global__ AICORE void ScenarioC_ComputeThenTloadKernel<T, COUNT>(__gm__ T *, int, uint64_t,              \
                                                                                __gm__ uint64_t *);                    \
    template __global__ AICORE void ScenarioC_DeviceOverlapKernel<T, COUNT>(__gm__ T *, int, uint64_t,                 \
                                                                             __gm__ uint8_t *, __gm__ uint64_t *)

CMP_INSTANTIATE(float, 262144);    // 1 MB
CMP_INSTANTIATE(float, 4194304);   // 16 MB
CMP_INSTANTIATE(float, 33554432);  // 128 MB
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

struct CmpEnv {
    aclrtStream stream = nullptr;
    void *srcDevice = nullptr;     // real source buffer being prefetched / TLOADed
    void *trashDevice = nullptr;   // L2-evicting buffer (>= kL2TrashBytes)
    void *cycleDev = nullptr;      // uint64_t slot for in-kernel cycle counter
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
// Scenario A: end-to-end wall-clock latency
// ============================================================================
struct ScenarioASample {
    uint64_t wallUs_baseline = 0;
    uint64_t wallUs_hostSdma = 0;
    uint64_t wallUs_deviceL2 = 0;
    uint64_t kernCyc_baseline = 0;
    uint64_t kernCyc_hostSdma = 0;
    uint64_t kernCyc_deviceL2 = 0;
};

// Host runner: single iteration of one (size, config) pair. Returns the
// measured wall-clock microseconds and the kernel-side cycle counter.
template <typename T, size_t count>
inline bool RunAOne(CmpEnv &env, int mode, uint64_t &wallUs, uint64_t &kernCycles)
{
    // mode: 0 = baseline, 1 = host SDMA prefetch, 2 = device TPREFETCH_L2
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
            ScenarioA_TloadOnlyKernel<T, count>
                <<<1, nullptr, env.stream>>>(reinterpret_cast<T *>(env.srcDevice), static_cast<int>(count),
                                             reinterpret_cast<uint64_t *>(env.cycleDev));
            break;
        case 2:
            ScenarioA_DevicePrefetchKernel<T, count>
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
bool RunScenarioAEndToEnd(int deviceId)
{
    constexpr size_t dataBytes = count * sizeof(T);
    constexpr int kWarmup = 2;
    constexpr int kMeasured = 5;

    CmpEnv env;
    if (!env.Init(deviceId, dataBytes)) {
        std::cerr << "[ERROR] ScenarioA: env init failed" << std::endl;
        env.Teardown();
        return false;
    }

    ScenarioASample s;
    auto accumulate = [&](int mode, uint64_t &wallAcc, uint64_t &kernAcc) {
        uint64_t totalWall = 0;
        uint64_t totalKern = 0;
        for (int i = 0; i < kWarmup + kMeasured; ++i) {
            uint64_t wall = 0;
            uint64_t kern = 0;
            if (!RunAOne<T, count>(env, mode, wall, kern)) {
                return false;
            }
            if (i >= kWarmup) {
                totalWall += wall;
                totalKern += kern;
            }
        }
        wallAcc = totalWall / static_cast<uint64_t>(kMeasured);
        kernAcc = totalKern / static_cast<uint64_t>(kMeasured);
        return true;
    };

    bool ok = accumulate(0, s.wallUs_baseline, s.kernCyc_baseline) &&
              accumulate(1, s.wallUs_hostSdma, s.kernCyc_hostSdma) &&
              accumulate(2, s.wallUs_deviceL2, s.kernCyc_deviceL2);
    env.Teardown();
    if (!ok) {
        return false;
    }

    double ratioHost = (s.wallUs_baseline > 0)
                            ? static_cast<double>(s.wallUs_hostSdma) / static_cast<double>(s.wallUs_baseline)
                            : 0.0;
    double ratioDev = (s.wallUs_baseline > 0)
                           ? static_cast<double>(s.wallUs_deviceL2) / static_cast<double>(s.wallUs_baseline)
                           : 0.0;
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "\n================================================================" << std::endl;
    std::cout << "[PERF] Scenario A - end-to-end wall-clock latency" << std::endl;
    std::cout << "  Data size:             " << dataBytes << " bytes (" << dataBytes / 1024 / 1024 << " MB)"
              << std::endl;
    std::cout << "  Iterations:            " << kMeasured << " (warmup=" << kWarmup << ")" << std::endl;
    std::cout << "  Syscnt freq assumed:   " << (SyscntHz() / 1.0e6) << " MHz" << std::endl;
    std::cout << "  baseline    wall:      " << static_cast<double>(s.wallUs_baseline) << " us"
              << "   kernel: " << CyclesToUs(s.kernCyc_baseline) << " us" << std::endl;
    std::cout << "  hostSDMA    wall:      " << static_cast<double>(s.wallUs_hostSdma) << " us"
              << "   kernel: " << CyclesToUs(s.kernCyc_hostSdma) << " us"
              << "   wall-ratio vs baseline: " << ratioHost << std::endl;
    std::cout << "  deviceL2    wall:      " << static_cast<double>(s.wallUs_deviceL2) << " us"
              << "   kernel: " << CyclesToUs(s.kernCyc_deviceL2) << " us   (incl. prefetch+wait)"
              << "   wall-ratio vs baseline: " << ratioDev << std::endl;
    std::cout << "================================================================\n" << std::endl;
    return true;
}

template bool RunScenarioAEndToEnd<float, 262144>(int);
template bool RunScenarioAEndToEnd<float, 4194304>(int);
template bool RunScenarioAEndToEnd<float, 33554432>(int);

// ============================================================================
// Scenario B: prefetch-issue overhead
// ============================================================================
template <typename T, size_t count>
bool RunScenarioBIssueOverhead(int deviceId)
{
    // We intentionally keep the "prefetch payload" very small so transfer
    // time is dominated by issue overhead. 4 KB is below a single L2 line
    // group; any observed cost is the fixed SDMA CMO issue+completion
    // roundtrip, not the actual bytes moved.
    constexpr size_t payloadElems = 1024;  // 4 KB for float
    static_assert(payloadElems <= count, "payload must fit inside allocated buffer");
    constexpr size_t dataBytes = count * sizeof(T);
    constexpr int kWarmup = 5;
    constexpr int kMeasured = 50;

    CmpEnv env;
    if (!env.Init(deviceId, dataBytes)) {
        std::cerr << "[ERROR] ScenarioB: env init failed" << std::endl;
        env.Teardown();
        return false;
    }

    uint64_t hostTotal = 0;
    uint64_t deviceTotal = 0;

    for (int i = 0; i < kWarmup + kMeasured; ++i) {
        env.TrashL2();

        // host path: PTO_PREFETCH + sync, host-timed.
        auto h0 = HrClock::now();
        pto::PTO_PREFETCH(env.srcDevice, payloadElems * sizeof(T), env.stream);
        env.aclStatus |= aclrtSynchronizeStream(env.stream);
        auto h1 = HrClock::now();
        uint64_t hostUs = ElapsedMicros(h0, h1);

        // device path: in-kernel TPREFETCH_L2 + Wait, reports syscnt cycles.
        ScenarioB_DeviceIssueKernel<T, count><<<1, nullptr, env.stream>>>(
            reinterpret_cast<T *>(env.srcDevice), static_cast<int>(payloadElems),
            reinterpret_cast<uint8_t *>(env.sdmaMgr.GetWorkspaceAddr()),
            reinterpret_cast<uint64_t *>(env.cycleDev));
        env.aclStatus |= aclrtSynchronizeStream(env.stream);
        uint64_t devCycles = env.ReadCycles();

        if (i >= kWarmup) {
            hostTotal += hostUs;
            deviceTotal += devCycles;
        }
    }

    uint64_t avgHostUs = hostTotal / static_cast<uint64_t>(kMeasured);
    uint64_t avgDevCycles = deviceTotal / static_cast<uint64_t>(kMeasured);
    env.Teardown();

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "\n================================================================" << std::endl;
    std::cout << "[PERF] Scenario B - prefetch issue overhead (4 KB payload)" << std::endl;
    std::cout << "  Iterations:            " << kMeasured << " (warmup=" << kWarmup << ")" << std::endl;
    std::cout << "  Syscnt freq assumed:   " << (SyscntHz() / 1.0e6) << " MHz" << std::endl;
    std::cout << "  host   PTO_PREFETCH:   " << static_cast<double>(avgHostUs)
              << " us   (wall-clock, incl. kernel launch + sync)" << std::endl;
    std::cout << "  device TPREFETCH_L2:   " << CyclesToUs(avgDevCycles)
              << " us   (in-kernel issue+Wait, excl. launch cost)" << std::endl;
    std::cout << "================================================================\n" << std::endl;
    return true;
}

template bool RunScenarioBIssueOverhead<float, 262144>(int);
template bool RunScenarioBIssueOverhead<float, 4194304>(int);
template bool RunScenarioBIssueOverhead<float, 33554432>(int);

// ============================================================================
// Scenario C: overlap with AI-Core compute
// ============================================================================

// Pick a spin-cycle count roughly equal to the expected SDMA prefetch time for
// the given buffer size. Empirical numbers on A2 (~1 TB/s HBM, 1 GHz scalar):
//   1 MB   -> ~1000 ns ->  ~1_000 cycles x  ~20 safety = ~20_000 cycles
//   16 MB  -> ~16 us   -> ~16_000 cycles x  ~5 safety  = ~80_000 cycles
//   128 MB -> ~128 us  -> ~128_000 cycles x ~2 safety  = ~256_000 cycles
// We intentionally over-shoot slightly so compute-A is long enough to fully
// hide the SDMA prefetch in the C2 case; undersize would make overlap look
// worse than it really is.
constexpr uint64_t SpinCyclesFor(size_t dataBytes)
{
    if (dataBytes <= 1 * 1024 * 1024)
        return 20000ULL;
    if (dataBytes <= 16 * 1024 * 1024)
        return 80000ULL;
    return 256000ULL;
}

template <typename T, size_t count>
bool RunScenarioCOverlap(int deviceId)
{
    constexpr size_t dataBytes = count * sizeof(T);
    constexpr int kWarmup = 2;
    constexpr int kMeasured = 5;
    constexpr uint64_t spinCycles = SpinCyclesFor(dataBytes);

    CmpEnv env;
    if (!env.Init(deviceId, dataBytes)) {
        std::cerr << "[ERROR] ScenarioC: env init failed" << std::endl;
        env.Teardown();
        return false;
    }

    auto runOne = [&](int mode, uint64_t &wallUs, uint64_t &kernCycles) {
        env.TrashL2();

        auto t0 = HrClock::now();
        switch (mode) {
            case 1:  // C1: host prefetch then kernel(compute-A -> warm TLOAD)
                pto::PTO_PREFETCH(env.srcDevice, env.dataBytes, env.stream);
                [[fallthrough]];
            case 0:  // C0: no prefetch, kernel(compute-A -> cold TLOAD)
                ScenarioC_ComputeThenTloadKernel<T, count>
                    <<<1, nullptr, env.stream>>>(reinterpret_cast<T *>(env.srcDevice), static_cast<int>(count),
                                                 spinCycles, reinterpret_cast<uint64_t *>(env.cycleDev));
                break;
            case 2:  // C2: kernel(TPREFETCH_L2 -> compute-A -> Wait -> warm TLOAD)
                ScenarioC_DeviceOverlapKernel<T, count><<<1, nullptr, env.stream>>>(
                    reinterpret_cast<T *>(env.srcDevice), static_cast<int>(count), spinCycles,
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
    };

    uint64_t wallTot[3] = {0, 0, 0};
    uint64_t kernTot[3] = {0, 0, 0};
    for (int mode = 0; mode < 3; ++mode) {
        for (int i = 0; i < kWarmup + kMeasured; ++i) {
            uint64_t wall = 0;
            uint64_t kern = 0;
            if (!runOne(mode, wall, kern)) {
                env.Teardown();
                return false;
            }
            if (i >= kWarmup) {
                wallTot[mode] += wall;
                kernTot[mode] += kern;
            }
        }
    }
    env.Teardown();

    uint64_t wallAvg[3];
    uint64_t kernAvg[3];
    for (int i = 0; i < 3; ++i) {
        wallAvg[i] = wallTot[i] / static_cast<uint64_t>(kMeasured);
        kernAvg[i] = kernTot[i] / static_cast<uint64_t>(kMeasured);
    }

    // cold-reduc = 1 - kern(C*) / kern(C0). Higher means more cold-TLOAD
    // latency was hidden behind the prefetch (positive value = win).
    double reducHost = (kernAvg[0] > 0)
                            ? 1.0 - static_cast<double>(kernAvg[1]) / static_cast<double>(kernAvg[0])
                            : 0.0;
    double reducDev = (kernAvg[0] > 0) ? 1.0 - static_cast<double>(kernAvg[2]) / static_cast<double>(kernAvg[0]) : 0.0;

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "\n================================================================" << std::endl;
    std::cout << "[PERF] Scenario C - prefetch overlapped with compute-A" << std::endl;
    std::cout << "  Data size:             " << dataBytes << " bytes (" << dataBytes / 1024 / 1024 << " MB)"
              << std::endl;
    std::cout << "  compute-A spin:        " << spinCycles << " cycles (" << CyclesToUs(spinCycles) << " us)"
              << std::endl;
    std::cout << "  Iterations:            " << kMeasured << " (warmup=" << kWarmup << ")" << std::endl;
    std::cout << "  Syscnt freq assumed:   " << (SyscntHz() / 1.0e6) << " MHz" << std::endl;
    std::cout << "  C0 no-prefetch           wall=" << static_cast<double>(wallAvg[0])
              << " us   kernel=" << CyclesToUs(kernAvg[0]) << " us" << std::endl;
    std::cout << "  C1 host   PTO_PREFETCH   wall=" << static_cast<double>(wallAvg[1])
              << " us   kernel=" << CyclesToUs(kernAvg[1]) << " us"
              << "   cold-reduc=" << reducHost << std::endl;
    std::cout << "  C2 device TPREFETCH_L2   wall=" << static_cast<double>(wallAvg[2])
              << " us   kernel=" << CyclesToUs(kernAvg[2]) << " us"
              << "   cold-reduc=" << reducDev << std::endl;
    std::cout << "  (higher cold-reduc = more cold-TLOAD latency hidden by prefetch)" << std::endl;
    std::cout << "================================================================\n" << std::endl;
    return true;
}

template bool RunScenarioCOverlap<float, 262144>(int);
template bool RunScenarioCOverlap<float, 4194304>(int);
template bool RunScenarioCOverlap<float, 33554432>(int);
