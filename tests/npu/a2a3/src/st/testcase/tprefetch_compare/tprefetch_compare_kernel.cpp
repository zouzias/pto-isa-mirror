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

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

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

CMP_INSTANTIATE(float, 16384);     // 64 KB
CMP_INSTANTIATE(float, 262144);    // 1 MB
CMP_INSTANTIATE(float, 4194304);   // 16 MB
CMP_INSTANTIATE(float, 16777216);  // 64 MB
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

// ============================================================================
// Statistics helpers — collect samples, report p5/p50/p95/min/max
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

    double P5() const { return Percentile(5.0); }
    double P50() const { return Percentile(50.0); }
    double P95() const { return Percentile(95.0); }
    double Min() const
    {
        if (samples.empty()) {
            return 0.0;
        }
        return *std::min_element(samples.begin(), samples.end());
    }
    double Max() const
    {
        if (samples.empty()) {
            return 0.0;
        }
        return *std::max_element(samples.begin(), samples.end());
    }
};

// ============================================================================
// CSV writer — appends rows to ./tprefetch_compare_<scenario>.csv (or path
// from env var TPREFETCH_COMPARE_CSV_DIR). Header is written automatically
// the first time a file is touched in this process.
// ============================================================================
inline std::string CsvFilePath(const std::string &scenario)
{
    const char *dir = std::getenv("TPREFETCH_COMPARE_CSV_DIR");
    std::string base = (dir != nullptr && dir[0] != '\0') ? std::string(dir) : std::string(".");
    return base + "/tprefetch_compare_" + scenario + ".csv";
}

inline void CsvAppendRow(const std::string &scenario, const std::string &header, const std::string &row)
{
    std::string path = CsvFilePath(scenario);

    // Determine whether the file is empty (need to write header) without
    // racing against concurrent appends. Tests in this binary run serially
    // (gtest default), so a simple stat-then-append is safe enough.
    bool need_header = false;
    {
        std::ifstream check(path);
        if (!check.good() || check.peek() == std::ifstream::traits_type::eof()) {
            need_header = true;
        }
    }

    std::ofstream f(path, std::ios::app);
    if (!f.is_open()) {
        std::cerr << "[WARN] CsvAppendRow: failed to open " << path << std::endl;
        return;
    }
    if (need_header) {
        f << header << "\n";
    }
    f << row << "\n";
}

// Allow CI to dial iteration counts down for smoke runs while keeping the
// "real" (shmem-style) 100-iter default for serious measurements. Callers
// pass their default; env var TPREFETCH_COMPARE_ITER overrides.
inline int IterCount(int defaultIter)
{
    const char *env = std::getenv("TPREFETCH_COMPARE_ITER");
    if (env != nullptr && env[0] != '\0') {
        int v = std::atoi(env);
        if (v > 0) {
            return v;
        }
    }
    return defaultIter;
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
//
// The "real-world" question this scenario answers: starting from a cold L2,
// how long does it take from issuing a prefetch to having the warm TLOAD
// complete? Three configurations measured at each size:
//
//   * baseline   — trash L2, then kernel(cold TLOAD only)
//   * host SDMA  — trash L2, then host PTO_PREFETCH on stream + kernel(warm TLOAD)
//   * device L2  — trash L2, then kernel(TPREFETCH_L2 -> Wait -> warm TLOAD)
//
// All three measurements time the entire host-visible window (start of first
// op after trash → aclrtSynchronizeStream returns), so they are apples-to-
// apples wall-clock comparable.
//
// Statistics (shmem-style): collect kIter raw samples per config, report
// p5 / p50 / p95. Per-iteration values are also persisted to a CSV for
// offline post-processing. 1 warmup precedes the measured samples.
// ============================================================================

// Single measurement of one (mode) pair. Returns wall-clock μs.
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
    constexpr int kWarmup = 1;
    const int kIter = IterCount(100);

    CmpEnv env;
    if (!env.Init(deviceId, dataBytes)) {
        std::cerr << "[ERROR] ScenarioA: env init failed" << std::endl;
        env.Teardown();
        return false;
    }

    SampleSet wallBase;
    SampleSet wallHost;
    SampleSet wallDev;
    SampleSet kernBase;
    SampleSet kernHost;
    SampleSet kernDev;

    auto collect = [&](int mode, SampleSet &wallSet, SampleSet &kernSet) {
        for (int i = 0; i < kWarmup + kIter; ++i) {
            uint64_t wall = 0;
            uint64_t kern = 0;
            if (!RunAOne<T, count>(env, mode, wall, kern)) {
                return false;
            }
            if (i >= kWarmup) {
                wallSet.Add(static_cast<double>(wall));
                kernSet.Add(CyclesToUs(kern));
            }
        }
        return true;
    };

    bool ok = collect(0, wallBase, kernBase) && collect(1, wallHost, kernHost) && collect(2, wallDev, kernDev);
    env.Teardown();
    if (!ok) {
        return false;
    }

    auto bandGBs = [&](double us) {
        return us > 0.0 ? (static_cast<double>(dataBytes) / us / 1000.0) : 0.0;
    };

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "\n================================================================" << std::endl;
    std::cout << "[PERF] Scenario A - end-to-end wall-clock latency" << std::endl;
    std::cout << "  Data size:             " << dataBytes << " bytes (";
    if (dataBytes >= 1024 * 1024) {
        std::cout << dataBytes / 1024 / 1024 << " MB)";
    } else {
        std::cout << dataBytes / 1024 << " KB)";
    }
    std::cout << std::endl;
    std::cout << "  Iterations:            " << kIter << " (warmup=" << kWarmup << ")" << std::endl;
    std::cout << "  Syscnt freq assumed:   " << (SyscntHz() / 1.0e6) << " MHz" << std::endl;
    std::cout << "  --- end-to-end wall (apples-to-apples instruction comparison) ---" << std::endl;
    std::cout << "  baseline               p50=" << wallBase.P50()
              << "us   p5=" << wallBase.P5() << " p95=" << wallBase.P95() << std::endl;
    std::cout << "  hostSDMA  PTO_PREFETCH p50=" << wallHost.P50()
              << "us   p5=" << wallHost.P5() << " p95=" << wallHost.P95()
              << "   band(p50)=" << bandGBs(wallHost.P50()) << " GB/s"
              << "   ratio vs base: " << (wallBase.P50() > 0 ? wallHost.P50() / wallBase.P50() : 0.0)
              << std::endl;
    std::cout << "  deviceL2  TPREFETCH_L2 p50=" << wallDev.P50()
              << "us   p5=" << wallDev.P5() << " p95=" << wallDev.P95()
              << "   band(p50)=" << bandGBs(wallDev.P50()) << " GB/s"
              << "   ratio vs base: " << (wallBase.P50() > 0 ? wallDev.P50() / wallBase.P50() : 0.0)
              << "   ratio vs host: " << (wallHost.P50() > 0 ? wallDev.P50() / wallHost.P50() : 0.0)
              << std::endl;
    std::cout << "  --- supplementary: in-kernel cycles (NOT apples-to-apples) ---" << std::endl;
    std::cout << "  baseline               p50=" << kernBase.P50() << "us" << std::endl;
    std::cout << "  hostSDMA  PTO_PREFETCH p50=" << kernHost.P50() << "us  (warm TLOAD only)" << std::endl;
    std::cout << "  deviceL2  TPREFETCH_L2 p50=" << kernDev.P50() << "us  (incl. in-kernel prefetch+wait)"
              << std::endl;
    std::cout << "  CSV: " << CsvFilePath("scenarioA") << std::endl;
    std::cout << "================================================================\n" << std::endl;

    // CSV: one row per (size, config) — six rows total per RunScenarioAEndToEnd
    // call. Combined with the per-size TESTs in main.cpp, the final CSV has
    // six rows per size.
    const std::string csvHeader =
        "size_bytes,size_label,config,iter,wall_p5_us,wall_p50_us,wall_p95_us,"
        "wall_min_us,wall_max_us,band_p50_gbs,kernel_p50_us";
    auto sizeLabel = [&]() -> std::string {
        std::ostringstream oss;
        if (dataBytes >= 1024 * 1024) {
            oss << dataBytes / 1024 / 1024 << "MB";
        } else {
            oss << dataBytes / 1024 << "KB";
        }
        return oss.str();
    };
    auto rowOf = [&](const std::string &cfg, const SampleSet &wall, const SampleSet &kern) {
        std::ostringstream oss;
        oss << dataBytes << ',' << sizeLabel() << ',' << cfg << ',' << kIter << ','
            << wall.P5() << ',' << wall.P50() << ',' << wall.P95() << ',' << wall.Min() << ',' << wall.Max() << ','
            << bandGBs(wall.P50()) << ',' << kern.P50();
        return oss.str();
    };
    CsvAppendRow("scenarioA", csvHeader, rowOf("baseline", wallBase, kernBase));
    CsvAppendRow("scenarioA", csvHeader, rowOf("host_sdma", wallHost, kernHost));
    CsvAppendRow("scenarioA", csvHeader, rowOf("device_l2", wallDev, kernDev));
    return true;
}

template bool RunScenarioAEndToEnd<float, 16384>(int);     // 64 KB
template bool RunScenarioAEndToEnd<float, 262144>(int);    // 1 MB
template bool RunScenarioAEndToEnd<float, 4194304>(int);   // 16 MB
template bool RunScenarioAEndToEnd<float, 16777216>(int);  // 64 MB
template bool RunScenarioAEndToEnd<float, 33554432>(int);  // 128 MB

// ============================================================================
// Scenario B: prefetch-issue overhead
//
// Real-world question: how much does it cost to "just fire one prefetch and
// wait", independent of how many bytes are moved? We use a 4 KB payload so
// the actual SDMA transfer is well under a microsecond and the measurement
// is dominated by the issue + completion roundtrip.
//
// Two metrics per side (apples-to-apples):
//   * host wall      — aclrtCmoAsync + aclrtSynchronizeStream wall-clock
//   * device wall    — kernel launch + in-kernel TPREFETCH_L2 + Wait + sync wall-clock
//
// Plus one supplementary metric:
//   * device in-kernel — only the TPREFETCH_L2 issue+Wait segment inside the
//     kernel. This models the marginal cost of embedding TPREFETCH_L2 into a
//     kernel that the application is already launching anyway (no extra
//     launch+sync amortized). NOT comparable to host wall directly.
//
// 100 iterations, p5/p50/p95 reported, CSV per-size row.
// ============================================================================
template <typename T, size_t count>
bool RunScenarioBIssueOverhead(int deviceId)
{
    // Tiny prefetch payload — below a single L2 line group, so any observed
    // cost is the fixed SDMA CMO issue+completion roundtrip, not actual bytes.
    constexpr size_t payloadElems = 1024;  // 4 KB for float
    static_assert(payloadElems <= count, "payload must fit inside allocated buffer");
    constexpr size_t dataBytes = count * sizeof(T);
    constexpr int kWarmup = 5;
    const int kIter = IterCount(100);

    CmpEnv env;
    if (!env.Init(deviceId, dataBytes)) {
        std::cerr << "[ERROR] ScenarioB: env init failed" << std::endl;
        env.Teardown();
        return false;
    }

    SampleSet hostWall;
    SampleSet deviceWall;
    SampleSet deviceCycles;

    for (int i = 0; i < kWarmup + kIter; ++i) {
        env.TrashL2();

        // host path wall.
        auto h0 = HrClock::now();
        pto::PTO_PREFETCH(env.srcDevice, payloadElems * sizeof(T), env.stream);
        env.aclStatus |= aclrtSynchronizeStream(env.stream);
        auto h1 = HrClock::now();
        uint64_t hostWallUs = ElapsedMicros(h0, h1);

        env.TrashL2();

        // device path wall.
        auto d0 = HrClock::now();
        ScenarioB_DeviceIssueKernel<T, count><<<1, nullptr, env.stream>>>(
            reinterpret_cast<T *>(env.srcDevice), static_cast<int>(payloadElems),
            reinterpret_cast<uint8_t *>(env.sdmaMgr.GetWorkspaceAddr()),
            reinterpret_cast<uint64_t *>(env.cycleDev));
        env.aclStatus |= aclrtSynchronizeStream(env.stream);
        auto d1 = HrClock::now();
        uint64_t deviceWallUs = ElapsedMicros(d0, d1);
        uint64_t devCyc = env.ReadCycles();

        if (i >= kWarmup) {
            hostWall.Add(static_cast<double>(hostWallUs));
            deviceWall.Add(static_cast<double>(deviceWallUs));
            deviceCycles.Add(CyclesToUs(devCyc));
        }
    }

    env.Teardown();

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "\n================================================================" << std::endl;
    std::cout << "[PERF] Scenario B - prefetch issue overhead (4 KB payload)" << std::endl;
    std::cout << "  Iterations:            " << kIter << " (warmup=" << kWarmup << ")" << std::endl;
    std::cout << "  Syscnt freq assumed:   " << (SyscntHz() / 1.0e6) << " MHz" << std::endl;
    std::cout << "  --- end-to-end wall (apples-to-apples instruction comparison) ---" << std::endl;
    std::cout << "  host   PTO_PREFETCH    p50=" << hostWall.P50()
              << "us   p5=" << hostWall.P5() << " p95=" << hostWall.P95()
              << "   (aclrtCmoAsync + aclrtSynchronizeStream)" << std::endl;
    std::cout << "  device TPREFETCH_L2    p50=" << deviceWall.P50()
              << "us   p5=" << deviceWall.P5() << " p95=" << deviceWall.P95()
              << "   (launch + in-kernel issue+Wait + sync)"
              << "   ratio vs host: " << (hostWall.P50() > 0 ? deviceWall.P50() / hostWall.P50() : 0.0)
              << std::endl;
    std::cout << "  --- supplementary: device marginal cost when embedded ---" << std::endl;
    std::cout << "  device TPREFETCH_L2    in-kernel p50=" << deviceCycles.P50()
              << "us   p5=" << deviceCycles.P5() << " p95=" << deviceCycles.P95() << std::endl;
    std::cout << "                                  (issue+Wait inside kernel, excl. launch+sync;"
              << " what you actually pay" << std::endl;
    std::cout << "                                   when adding TPREFETCH_L2 to a kernel that's"
              << " already running)" << std::endl;
    std::cout << "  CSV: " << CsvFilePath("scenarioB") << std::endl;
    std::cout << "================================================================\n" << std::endl;

    const std::string csvHeader =
        "buffer_bytes,payload_bytes,config,iter,wall_p5_us,wall_p50_us,wall_p95_us,"
        "wall_min_us,wall_max_us";
    auto rowOf = [&](const std::string &cfg, const SampleSet &s) {
        std::ostringstream oss;
        oss << dataBytes << ',' << (payloadElems * sizeof(T)) << ',' << cfg << ',' << kIter << ','
            << s.P5() << ',' << s.P50() << ',' << s.P95() << ',' << s.Min() << ',' << s.Max();
        return oss.str();
    };
    CsvAppendRow("scenarioB", csvHeader, rowOf("host_wall", hostWall));
    CsvAppendRow("scenarioB", csvHeader, rowOf("device_wall", deviceWall));
    CsvAppendRow("scenarioB", csvHeader, rowOf("device_in_kernel", deviceCycles));
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
    constexpr int kWarmup = 1;
    constexpr uint64_t spinCycles = SpinCyclesFor(dataBytes);
    const int kIter = IterCount(100);

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

    SampleSet wall[3];
    SampleSet kern[3];
    for (int mode = 0; mode < 3; ++mode) {
        for (int i = 0; i < kWarmup + kIter; ++i) {
            uint64_t w = 0;
            uint64_t k = 0;
            if (!runOne(mode, w, k)) {
                env.Teardown();
                return false;
            }
            if (i >= kWarmup) {
                wall[mode].Add(static_cast<double>(w));
                kern[mode].Add(CyclesToUs(k));
            }
        }
    }
    env.Teardown();

    // cold-reduc = 1 - kern(C*)/kern(C0). Higher means more cold-TLOAD
    // latency was hidden behind the prefetch (positive value = win).
    auto coldReduc = [&](int mode) {
        return kern[0].P50() > 0 ? 1.0 - kern[mode].P50() / kern[0].P50() : 0.0;
    };

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "\n================================================================" << std::endl;
    std::cout << "[PERF] Scenario C - prefetch overlapped with compute-A" << std::endl;
    std::cout << "  Data size:             " << dataBytes << " bytes (" << dataBytes / 1024 / 1024 << " MB)"
              << std::endl;
    std::cout << "  compute-A spin:        " << spinCycles << " cycles (" << CyclesToUs(spinCycles) << " us)"
              << std::endl;
    std::cout << "  Iterations:            " << kIter << " (warmup=" << kWarmup << ")" << std::endl;
    std::cout << "  Syscnt freq assumed:   " << (SyscntHz() / 1.0e6) << " MHz" << std::endl;
    std::cout << "  C0 no-prefetch         wall p50=" << wall[0].P50()
              << "us   kernel p50=" << kern[0].P50() << "us" << std::endl;
    std::cout << "  C1 host   PTO_PREFETCH wall p50=" << wall[1].P50()
              << "us   kernel p50=" << kern[1].P50() << "us"
              << "   cold-reduc=" << coldReduc(1) << std::endl;
    std::cout << "  C2 device TPREFETCH_L2 wall p50=" << wall[2].P50()
              << "us   kernel p50=" << kern[2].P50() << "us"
              << "   cold-reduc=" << coldReduc(2) << std::endl;
    std::cout << "  (higher cold-reduc = more cold-TLOAD latency hidden by prefetch)" << std::endl;
    std::cout << "  CSV: " << CsvFilePath("scenarioC") << std::endl;
    std::cout << "================================================================\n" << std::endl;

    const std::string csvHeader =
        "size_bytes,size_label,spin_cycles,config,iter,wall_p5_us,wall_p50_us,wall_p95_us,"
        "kernel_p5_us,kernel_p50_us,kernel_p95_us,cold_reduc";
    auto sizeLabel = [&]() -> std::string {
        std::ostringstream oss;
        if (dataBytes >= 1024 * 1024) {
            oss << dataBytes / 1024 / 1024 << "MB";
        } else {
            oss << dataBytes / 1024 << "KB";
        }
        return oss.str();
    };
    auto rowOf = [&](int mode, const std::string &cfg) {
        std::ostringstream oss;
        oss << dataBytes << ',' << sizeLabel() << ',' << spinCycles << ',' << cfg << ',' << kIter << ','
            << wall[mode].P5() << ',' << wall[mode].P50() << ',' << wall[mode].P95() << ','
            << kern[mode].P5() << ',' << kern[mode].P50() << ',' << kern[mode].P95() << ','
            << (mode == 0 ? 0.0 : coldReduc(mode));
        return oss.str();
    };
    CsvAppendRow("scenarioC", csvHeader, rowOf(0, "C0_no_prefetch"));
    CsvAppendRow("scenarioC", csvHeader, rowOf(1, "C1_host_prefetch"));
    CsvAppendRow("scenarioC", csvHeader, rowOf(2, "C2_device_prefetch"));
    return true;
}

template bool RunScenarioCOverlap<float, 262144>(int);
template bool RunScenarioCOverlap<float, 4194304>(int);
template bool RunScenarioCOverlap<float, 33554432>(int);
