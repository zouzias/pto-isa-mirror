/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <numeric>
#include <algorithm>
#include <sys/wait.h>
#include <unistd.h>

#include "acl/acl.h"
#include "common.hpp"
#include "test_common.h"
#include "gemm_config.h"

// ============================================================================
// External Kernel Launch Functions (defined in gemm_ar_performance_kernel.cpp)
// ============================================================================

// Fused GEMM + AllReduce (overlap: tile-level pipeline)
template <typename T>
void LaunchGEMME2E(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *shmem, void *stream, bool is_overlap);

// GEMM only (non-overlap step 1)
template <typename T>
void LaunchGemmOnly(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *shmem, void *stream);

// AllReduce only (non-overlap step 2)
template <typename T>
void LaunchAllReduceOnly(uint8_t *out, uint8_t *shmem, void *stream);


// ============================================================================
// Benchmark mode
// ============================================================================
enum class BenchMode {
    BOTH,           // Test both overlap and non-overlap, then compare
    OVERLAP_ONLY,   // Test overlap mode only
    NO_OVERLAP_ONLY // Test non-overlap mode only
};

// ============================================================================
// Command Line Argument Parser
// ============================================================================
struct TestArgs {
    int n_ranks = 2;
    int n_devices = 2;
    int first_rank_id = 0;
    int first_device_id = 0;
    int warmup_iters = 5;
    int perf_iters = 20;
    BenchMode bench_mode = BenchMode::BOTH;
    bool verify = true;
    bool verbose = true;
    bool help = false;
};

void PrintUsage(const char *prog) {
    std::cout << "Usage: " << prog << " [options]\n"
              << "\n"
              << "GEMM with AllReduce Performance Benchmark (Overlap vs Non-Overlap)\n"
              << "===================================================================\n"
              << "\n"
              << "GEMM Configuration:\n"
              << "  - Global shape:        [" << GEMM_M << " x " << GEMM_K
              << "] * [" << GEMM_K << " x " << GEMM_N << "]\n"
              << "  - Single core shape:   [" << SINGLE_CORE_M << ", " << SINGLE_CORE_K
              << ", " << SINGLE_CORE_N << "]\n"
              << "  - Base tile:           [" << BASE_M << ", " << BASE_K
              << ", " << BASE_N << "]\n"
              << "  - Block dim:           " << BLOCK_DIM << "\n"
              << "\n"
              << "Overlap mode:     Single fused kernel, GEMM and AllReduce run concurrently\n"
              << "                  with tile-level TNOTIFY/TWAIT pipeline.\n"
              << "Non-overlap mode: Two separate kernel launches — GEMM first, then AllReduce.\n"
              << "\n"
              << "Options:\n"
              << "  -r, --ranks N          Number of ranks (default: 4)\n"
              << "  -d, --devices N        Number of devices (default: 4)\n"
              << "  -w, --warmup N         Number of warmup iterations (default: 5)\n"
              << "  -i, --iters N          Number of measurement iterations (default: 20)\n"
              << "  --both                 Benchmark both overlap & non-overlap (default)\n"
              << "  --overlap-only         Benchmark overlap mode only\n"
              << "  --no-overlap-only      Benchmark non-overlap mode only\n"
              << "  --no-verify            Skip correctness verification\n"
              << "  -q, --quiet            Disable per-iteration verbose output\n"
              << "  -h, --help             Show this help message\n"
              << std::endl;
}

TestArgs ParseArgs(int argc, char **argv) {
    TestArgs args;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            args.help = true;
        } else if ((arg == "-r" || arg == "--ranks") && i + 1 < argc) {
            args.n_ranks = std::atoi(argv[++i]);
        } else if ((arg == "-d" || arg == "--devices") && i + 1 < argc) {
            args.n_devices = std::atoi(argv[++i]);
        } else if ((arg == "-w" || arg == "--warmup") && i + 1 < argc) {
            args.warmup_iters = std::atoi(argv[++i]);
        } else if ((arg == "-i" || arg == "--iters") && i + 1 < argc) {
            args.perf_iters = std::atoi(argv[++i]);
        } else if (arg == "--both") {
            args.bench_mode = BenchMode::BOTH;
        } else if (arg == "--overlap-only") {
            args.bench_mode = BenchMode::OVERLAP_ONLY;
        } else if (arg == "--no-overlap-only") {
            args.bench_mode = BenchMode::NO_OVERLAP_ONLY;
        } else if (arg == "--no-verify") {
            args.verify = false;
        } else if (arg == "-q" || arg == "--quiet") {
            args.verbose = false;
        }
    }
    return args;
}

// ============================================================================
// Performance Statistics
// ============================================================================
struct PerfStats {
    double min_ms;
    double max_ms;
    double avg_ms;
    double median_ms;
    double stddev_ms;
    double p90_ms;
    double p99_ms;
    double tflops;
    double bandwidth_gb;
};

PerfStats ComputeStats(std::vector<float> &latencies_ms, uint64_t flops, uint64_t bytes_transferred)
{
    PerfStats stats = {};
    if (latencies_ms.empty()) return stats;

    std::sort(latencies_ms.begin(), latencies_ms.end());
    size_t n = latencies_ms.size();

    stats.min_ms = latencies_ms.front();
    stats.max_ms = latencies_ms.back();

    double sum = 0.0;
    for (float v : latencies_ms) sum += v;
    stats.avg_ms = sum / n;

    if (n % 2 == 0)
        stats.median_ms = (latencies_ms[n / 2 - 1] + latencies_ms[n / 2]) / 2.0;
    else
        stats.median_ms = latencies_ms[n / 2];

    double sum_sq = 0.0;
    for (float v : latencies_ms) {
        double diff = v - stats.avg_ms;
        sum_sq += diff * diff;
    }
    stats.stddev_ms = std::sqrt(sum_sq / n);

    auto pIdx = [n](double pct) -> size_t {
        size_t idx = static_cast<size_t>(n * pct);
        return idx >= n ? n - 1 : idx;
    };
    stats.p90_ms = latencies_ms[pIdx(0.90)];
    stats.p99_ms = latencies_ms[pIdx(0.99)];

    if (stats.avg_ms > 0) {
        stats.tflops = static_cast<double>(flops) / (stats.avg_ms * 1e-3) / 1e12;
        stats.bandwidth_gb = static_cast<double>(bytes_transferred) / (stats.avg_ms * 1e-3) / 1e9;
    }
    return stats;
}

void PrintStats(const char *label, const PerfStats &s)
{
    std::cout << "  [" << label << "]" << std::endl;
    std::cout << "    Latency (ms):  Min=" << std::fixed << std::setprecision(3) << s.min_ms
              << "  Max=" << s.max_ms << "  Avg=" << s.avg_ms
              << "  Median=" << s.median_ms << "  StdDev=" << s.stddev_ms << std::endl;
    std::cout << "                   P90=" << s.p90_ms << "  P99=" << s.p99_ms << std::endl;
    if (s.tflops > 0)
        std::cout << "    TFLOPS:        " << std::setprecision(2) << s.tflops << std::endl;
    if (s.bandwidth_gb > 0)
        std::cout << "    Bandwidth:     " << std::setprecision(2) << s.bandwidth_gb << " GB/s" << std::endl;
    std::cout << std::endl;
}

// ============================================================================
// Non-overlap benchmark result (has GEMM / AR / Total breakdown)
// ============================================================================
struct NonOverlapResult {
    std::vector<float> gemm_ms;
    std::vector<float> ar_ms;
    std::vector<float> total_ms;
};

// ============================================================================
// Overlap benchmark pass: single fused kernel
// ============================================================================
std::vector<float> RunOverlapBenchmark(
    int warmup_iters, int perf_iters,
    uint8_t *dstDevice, uint8_t *src0Device, uint8_t *src1Device,
    uint8_t *shmemDevice, aclrtStream stream,
    aclrtEvent startEvent, aclrtEvent endEvent,
    int rank_id, bool verbose)
{
    // Warmup
    if (verbose && rank_id == 0)
        std::cout << "\n[Perf] Overlap warmup: " << warmup_iters << " iterations..." << std::endl;
    for (int iter = 0; iter < warmup_iters; ++iter) {
        ShmemBarrierAll();
        LaunchGEMME2E<uint16_t>(dstDevice, src0Device, src1Device, shmemDevice, stream, true);
        aclrtSynchronizeStream(stream);
        ShmemBarrierAll();
    }

    // Measurement
    if (verbose && rank_id == 0)
        std::cout << "[Perf] Overlap measurement: " << perf_iters << " iterations..." << std::endl;

    std::vector<float> latencies(perf_iters);
    for (int iter = 0; iter < perf_iters; ++iter) {
        ShmemBarrierAll();
        aclrtRecordEvent(startEvent, stream);
        LaunchGEMME2E<uint16_t>(dstDevice, src0Device, src1Device, shmemDevice, stream, true);
        aclrtRecordEvent(endEvent, stream);
        aclrtSynchronizeStream(stream);
        ShmemBarrierAll();

        float ms = 0.0f;
        aclrtEventElapsedTime(&ms, startEvent, endEvent);
        latencies[iter] = ms;

        if (verbose && rank_id == 0)
            std::cout << "  [Overlap Iter " << std::setw(3) << iter << "] "
                      << std::fixed << std::setprecision(3) << ms << " ms" << std::endl;
    }
    return latencies;
}

// ============================================================================
// Non-overlap benchmark pass: GEMM kernel first, then AllReduce kernel
// Reports GEMM time, AR time, and total (= GEMM + AR) separately.
// ============================================================================
NonOverlapResult RunNonOverlapBenchmark(
    int warmup_iters, int perf_iters,
    uint8_t *dstDevice, uint8_t *src0Device, uint8_t *src1Device,
    uint8_t *shmemDevice, aclrtStream stream,
    aclrtEvent startEvent, aclrtEvent endEvent,
    int rank_id, bool verbose)
{
    // Warmup: run the full GEMM-then-AR sequence
    if (verbose && rank_id == 0)
        std::cout << "\n[Perf] Non-Overlap warmup: " << warmup_iters << " iterations..." << std::endl;
    for (int iter = 0; iter < warmup_iters; ++iter) {
        // Step 1: GEMM
        ShmemBarrierAll();
        LaunchGemmOnly<uint16_t>(dstDevice, src0Device, src1Device, shmemDevice, stream);
        aclrtSynchronizeStream(stream);
        // Step 2: Barrier to ensure all ranks finished GEMM
        ShmemBarrierAll();
        // Step 3: AllReduce
        LaunchAllReduceOnly<uint16_t>(dstDevice, shmemDevice, stream);
        aclrtSynchronizeStream(stream);
        ShmemBarrierAll();
    }

    // Measurement
    if (verbose && rank_id == 0)
        std::cout << "[Perf] Non-Overlap measurement: " << perf_iters << " iterations..." << std::endl;

    NonOverlapResult result;
    result.gemm_ms.resize(perf_iters);
    result.ar_ms.resize(perf_iters);
    result.total_ms.resize(perf_iters);

    for (int iter = 0; iter < perf_iters; ++iter) {
        // ---- Step 1: GEMM Only ----
        float gemm_ms = 0.0f;        
        float ar_ms = 0.0f;
        ShmemBarrierAll();
        aclrtRecordEvent(startEvent, stream);
        LaunchGemmOnly<uint16_t>(dstDevice, src0Device, src1Device, shmemDevice, stream);
        aclrtSynchronizeStream(stream);
        ShmemBarrierAll();

        // ---- Step 2: AllReduce Only ----
        // aclrtRecordEvent(startEvent, stream);
        LaunchAllReduceOnly<uint16_t>(dstDevice, shmemDevice, stream);
        aclrtRecordEvent(endEvent, stream);
        aclrtSynchronizeStream(stream);
        aclrtEventElapsedTime(&ar_ms, startEvent, endEvent);

        ShmemBarrierAll();

        result.gemm_ms[iter] = gemm_ms;
        result.ar_ms[iter] = ar_ms;
        result.total_ms[iter] = gemm_ms + ar_ms;

        if (verbose && rank_id == 0)
            std::cout << "  [Non-Overlap Iter " << std::setw(3) << iter << "] "
                      << std::fixed << std::setprecision(3)
                      << "GEMM=" << gemm_ms << " ms  AR=" << ar_ms
                      << " ms  Total=" << result.total_ms[iter] << " ms" << std::endl;
    }
    return result;
}

// ============================================================================
// Single Rank GEMM+AR Performance Test
// ============================================================================
bool RunGemmKernelPerf(int rank_id, int n_ranks, int n_devices, int first_device_id,
                       int warmup_iters, int perf_iters, BenchMode bench_mode,
                       bool verify, bool verbose)
{
    int32_t ret = shmem_set_conf_store_tls(false, nullptr, 0);
    if (ret != 0) {
        std::cerr << "[Rank " << rank_id << "] Failed to init shmem tls\n";
        return false;
    }

    const int32_t device_id = rank_id % n_devices + first_device_id;
    int status = 0;
    aclrtStream stream = nullptr;

    status |= aclInit(nullptr);
    status |= aclrtSetDevice(device_id);
    status |= aclrtCreateStream(&stream);

    if (status != 0) {
        std::cerr << "[Rank " << rank_id << "] ACL initialization failed\n";
        return false;
    }

    // Initialize shmem
    ShmemEnv env;
    const char *ip = "tcp://127.0.0.1:8778";
    env.rank = rank_id;
    env.size = n_ranks;
    env.ipPort = ip;

    if (!ShmemInitFromEnv(env)) {
        std::cerr << "[Rank " << rank_id << "] Shmem initialization failed\n";
        return false;
    }

    if (verbose)
        std::cout << "[Rank " << rank_id << "] Running on device " << device_id << std::endl;

    // Calculate sizes
    size_t aFileSize = GEMM_M * GEMM_K * sizeof(uint16_t);
    size_t bFileSize = GEMM_K * GEMM_N * sizeof(uint16_t);
    size_t cFileSize = GEMM_M * GEMM_N * sizeof(float);

    // Allocate host memory
    uint8_t *dstHost = nullptr, *src0Host = nullptr, *src1Host = nullptr;
    uint8_t *dstDevice = nullptr, *src0Device = nullptr, *src1Device = nullptr;

    aclrtMallocHost((void **)(&dstHost), cFileSize);
    aclrtMallocHost((void **)(&src0Host), aFileSize);
    aclrtMallocHost((void **)(&src1Host), bFileSize);

    aclrtMalloc((void **)&dstDevice, cFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src0Device, aFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src1Device, bFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    uint8_t *shmemDevice = reinterpret_cast<uint8_t*>(ShmemCalloc((1UL << 28), sizeof(uint32_t)));
    if (shmemDevice == nullptr) {
        std::cerr << "[Rank " << rank_id << "] ShmemMalloc failed\n";
        aclrtFree(dstDevice); aclrtFree(src0Device); aclrtFree(src1Device);
        aclrtFreeHost(dstHost); aclrtFreeHost(src0Host); aclrtFreeHost(src1Host);
        ShmemFinalize();
        return false;
    }

    // Initialize input data
    PtoTestCommon::ReadFile("../input/x1_gm.bin", aFileSize, src0Host, aFileSize);
    PtoTestCommon::ReadFile("../input/x2_gm.bin", bFileSize, src1Host, bFileSize);
    aclrtMemcpy(src0Device, aFileSize, src0Host, aFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src1Device, bFileSize, src1Host, bFileSize, ACL_MEMCPY_HOST_TO_DEVICE);

    ShmemBarrierAll();

    // Create ACL events for timing
    aclrtEvent startEvent = nullptr, endEvent = nullptr;
    aclrtCreateEvent(&startEvent);
    aclrtCreateEvent(&endEvent);

    // ================================================================
    // Determine which modes to benchmark
    // ================================================================
    bool run_overlap    = (bench_mode == BenchMode::BOTH || bench_mode == BenchMode::OVERLAP_ONLY);
    bool run_no_overlap = (bench_mode == BenchMode::BOTH || bench_mode == BenchMode::NO_OVERLAP_ONLY);

    // ================================================================
    // Run benchmarks
    // ================================================================
    std::vector<float> latencies_overlap;
    NonOverlapResult   result_no_overlap;

    if (run_overlap) {
        latencies_overlap = RunOverlapBenchmark(
            warmup_iters, perf_iters,
            dstDevice, src0Device, src1Device, shmemDevice, stream,
            startEvent, endEvent, rank_id, verbose);
    }

    if (run_no_overlap) {
        result_no_overlap = RunNonOverlapBenchmark(
            warmup_iters, perf_iters,
            dstDevice, src0Device, src1Device, shmemDevice, stream,
            startEvent, endEvent, rank_id, verbose);
    }

    // ================================================================
    // Report (rank 0 only)
    // ================================================================
    if (rank_id == 0) {
        uint64_t gemm_flops = 2ULL * GEMM_M * GEMM_N * GEMM_K;
        uint64_t bytes_io = static_cast<uint64_t>(GEMM_M) * GEMM_K * sizeof(uint16_t)
                          + static_cast<uint64_t>(GEMM_K) * GEMM_N * sizeof(uint16_t)
                          + static_cast<uint64_t>(GEMM_M) * GEMM_N * sizeof(float)
                          + static_cast<uint64_t>(GEMM_M) * GEMM_N * sizeof(float) * (n_ranks - 1);
        // AllReduce-only bytes: each rank reads (n_ranks-1) remote tiles + writes 1 local result
        uint64_t ar_bytes = static_cast<uint64_t>(GEMM_M) * GEMM_N * sizeof(float) * n_ranks;

        std::cout << "\n";
        std::cout << "============================================================" << std::endl;
        std::cout << "  GEMM + AllReduce Performance Results" << std::endl;
        std::cout << "============================================================" << std::endl;
        std::cout << "  Configuration:" << std::endl;
        std::cout << "    Matrix shape:     [" << GEMM_M << " x " << GEMM_K
                  << "] * [" << GEMM_K << " x " << GEMM_N << "]" << std::endl;
        std::cout << "    Ranks:            " << n_ranks << std::endl;
        std::cout << "    Warmup iters:     " << warmup_iters << std::endl;
        std::cout << "    Measure iters:    " << perf_iters << std::endl;
        std::cout << "    GEMM FLOPs:       " << gemm_flops << std::endl;
        std::cout << "------------------------------------------------------------" << std::endl;

        PerfStats stats_overlap = {};
        PerfStats stats_gemm_only = {}, stats_ar_only = {}, stats_no_overlap_total = {};

        if (run_overlap) {
            stats_overlap = ComputeStats(latencies_overlap, gemm_flops, bytes_io);
            PrintStats("Overlap (fused GEMM+AR, tile-level pipeline)", stats_overlap);
        }

        if (run_no_overlap) {
            stats_gemm_only       = ComputeStats(result_no_overlap.gemm_ms,  gemm_flops, 0);
            stats_ar_only         = ComputeStats(result_no_overlap.ar_ms,    0,          ar_bytes);
            stats_no_overlap_total = ComputeStats(result_no_overlap.total_ms, gemm_flops, bytes_io);

            PrintStats("Non-Overlap: GEMM Only", stats_gemm_only);
            PrintStats("Non-Overlap: AllReduce Only", stats_ar_only);
            PrintStats("Non-Overlap: Total (GEMM + AR sequential)", stats_no_overlap_total);
        }

        // ============================================================
        // Comparison (when both modes tested)
        // ============================================================
        if (run_overlap && run_no_overlap) {
            double speedup_avg    = stats_no_overlap_total.avg_ms / stats_overlap.avg_ms;
            double speedup_median = stats_no_overlap_total.median_ms / stats_overlap.median_ms;
            double saved_avg_ms   = stats_no_overlap_total.avg_ms - stats_overlap.avg_ms;
            double saved_pct      = saved_avg_ms / stats_no_overlap_total.avg_ms * 100.0;

            // How much of the AR time is hidden by overlap?
            // If overlap_time ≈ gemm_time, then AR is 100% hidden.
            // ar_overhead_in_overlap = overlap_time - gemm_only_time
            // hidden_pct = (1 - ar_overhead / ar_only_time) * 100
            double ar_overhead = stats_overlap.avg_ms - stats_gemm_only.avg_ms;
            double ar_hidden_pct = 0.0;
            if (stats_ar_only.avg_ms > 0)
                ar_hidden_pct = (1.0 - ar_overhead / stats_ar_only.avg_ms) * 100.0;

            std::cout << "============================================================" << std::endl;
            std::cout << "  Overlap vs Non-Overlap Comparison" << std::endl;
            std::cout << "============================================================" << std::endl;
            std::cout << std::endl;

            // Table header
            std::cout << "  " << std::left  << std::setw(20) << "Metric"
                      << std::right
                      << std::setw(12) << "Overlap"
                      << std::setw(12) << "GEMM-Only"
                      << std::setw(12) << "AR-Only"
                      << std::setw(16) << "Non-OL Total"
                      << std::setw(10) << "Speedup"
                      << std::endl;
            std::cout << "  " << std::string(82, '-') << std::endl;

            auto row = [](const char *metric, double ol, double gemm, double ar, double total, double su) {
                std::cout << "  " << std::left  << std::setw(20) << metric
                          << std::right << std::fixed << std::setprecision(3)
                          << std::setw(10) << ol   << "ms"
                          << std::setw(10) << gemm << "ms"
                          << std::setw(10) << ar   << "ms"
                          << std::setw(14) << total << "ms"
                          << std::setprecision(2) << std::setw(8) << su << "x"
                          << std::endl;
            };

            row("Avg Latency",
                stats_overlap.avg_ms, stats_gemm_only.avg_ms, stats_ar_only.avg_ms,
                stats_no_overlap_total.avg_ms, speedup_avg);
            row("Median Latency",
                stats_overlap.median_ms, stats_gemm_only.median_ms, stats_ar_only.median_ms,
                stats_no_overlap_total.median_ms, speedup_median);
            row("Min Latency",
                stats_overlap.min_ms, stats_gemm_only.min_ms, stats_ar_only.min_ms,
                stats_no_overlap_total.min_ms,
                stats_no_overlap_total.min_ms / stats_overlap.min_ms);
            row("Max Latency",
                stats_overlap.max_ms, stats_gemm_only.max_ms, stats_ar_only.max_ms,
                stats_no_overlap_total.max_ms,
                stats_no_overlap_total.max_ms / stats_overlap.max_ms);

            std::cout << std::endl;
            std::cout << "  " << std::left  << std::setw(20) << "TFLOPS"
                      << std::right << std::fixed << std::setprecision(2)
                      << std::setw(12) << stats_overlap.tflops
                      << std::setw(12) << stats_gemm_only.tflops
                      << std::setw(12) << "-"
                      << std::setw(16) << stats_no_overlap_total.tflops
                      << std::endl;

            std::cout << std::endl;
            std::cout << "  Summary:" << std::endl;
            std::cout << "    Overlap speedup (avg):     " << std::fixed << std::setprecision(2)
                      << speedup_avg << "x" << std::endl;
            std::cout << "    Time saved (avg):          " << std::fixed << std::setprecision(3)
                      << saved_avg_ms << " ms (" << std::setprecision(1) << saved_pct << "%)" << std::endl;
            std::cout << "    GEMM-only avg:             " << std::fixed << std::setprecision(3)
                      << stats_gemm_only.avg_ms << " ms" << std::endl;
            std::cout << "    AR-only avg:               " << std::fixed << std::setprecision(3)
                      << stats_ar_only.avg_ms << " ms" << std::endl;
            std::cout << "    AR overhead in overlap:    " << std::fixed << std::setprecision(3)
                      << (ar_overhead > 0 ? ar_overhead : 0) << " ms" << std::endl;
            std::cout << "    AR hidden by overlap:      " << std::fixed << std::setprecision(1)
                      << ar_hidden_pct << "%" << std::endl;
            std::cout << "============================================================" << std::endl;
        }
    }

    // ================================================================
    // Correctness Verification
    // ================================================================
    bool test_passed = true;

    if (verify) {
        // Run one pass with overlap=true for verification
        ShmemBarrierAll();
        LaunchGEMME2E<uint16_t>(dstDevice, src0Device, src1Device, shmemDevice, stream, true);
        aclrtSynchronizeStream(stream);
        ShmemBarrierAll();

        aclrtMemcpy(dstHost, cFileSize, dstDevice, cFileSize, ACL_MEMCPY_DEVICE_TO_HOST);

        if (rank_id == 0)
            PtoTestCommon::WriteFile("../output/output_z.bin", dstHost, cFileSize);

        ShmemBarrierAll();

        if (rank_id == 0) {
            std::vector<float> golden(cFileSize);
            std::vector<float> devFinal(cFileSize);
            PtoTestCommon::ReadFile("../output/golden.bin", cFileSize, golden.data(), cFileSize);
            PtoTestCommon::ReadFile("../output/output_z.bin", cFileSize, devFinal.data(), cFileSize);

            for (int i = 0; i < static_cast<int>(cFileSize); ++i)
                devFinal[i] = devFinal[i] / n_ranks;

            if (PtoTestCommon::ResultCmp(golden, devFinal, 0.001f)) {
                std::cout << "\n[Verify] Correctness check: PASSED" << std::endl;
            } else {
                std::cout << "\n[Verify] Correctness check: FAILED" << std::endl;
                test_passed = false;
            }
        }
    } else {
        ShmemBarrierAll();
    }

    // Cleanup
    aclrtDestroyEvent(startEvent);
    aclrtDestroyEvent(endEvent);
    aclrtFree(dstDevice); aclrtFree(src0Device); aclrtFree(src1Device);
    aclrtFreeHost(dstHost); aclrtFreeHost(src0Host); aclrtFreeHost(src1Host);
    ShmemFinalize();
    aclrtDestroyStream(stream);
    aclrtResetDevice(device_id);
    aclFinalize();
    return test_passed;
}

// ============================================================================
// Multi-process Launcher
// ============================================================================
bool RunGemmMultiProcess(int n_ranks, int n_devices, int first_rank_id, int first_device_id,
                         int warmup_iters, int perf_iters, BenchMode bench_mode,
                         bool verify, bool verbose)
{
    std::vector<pid_t> pids;
    for (int r = 0; r < n_ranks; ++r) {
        pid_t pid = fork();
        if (pid == 0) {
            const bool ok = RunGemmKernelPerf(
                first_rank_id + r, n_ranks, n_devices, first_device_id,
                warmup_iters, perf_iters, bench_mode, verify, verbose);
            _exit(ok ? 0 : 1);
        } else if (pid > 0) {
            pids.push_back(pid);
        } else {
            std::cerr << "Fork failed for rank " << r << std::endl;
            return false;
        }
    }
    bool success = true;
    for (pid_t p : pids) {
        int st = 0;
        waitpid(p, &st, 0);
        if (!(WIFEXITED(st) && WEXITSTATUS(st) == 0)) success = false;
    }
    return success;
}

// ============================================================================
// Main
// ============================================================================
int main(int argc, char **argv)
{
    TestArgs args = ParseArgs(argc, argv);

    if (args.help) {
        PrintUsage(argv[0]);
        return 0;
    }

    const char *mode_str = "both (overlap & non-overlap)";
    if (args.bench_mode == BenchMode::OVERLAP_ONLY)    mode_str = "overlap only";
    if (args.bench_mode == BenchMode::NO_OVERLAP_ONLY) mode_str = "non-overlap only";

    std::cout << "============================================================" << std::endl;
    std::cout << "  GEMM + AllReduce Performance Benchmark" << std::endl;
    std::cout << "============================================================" << std::endl;
    std::cout << "  GEMM Configuration:" << std::endl;
    std::cout << "    Global shape:     [" << GEMM_M << " x " << GEMM_K
              << "] * [" << GEMM_K << " x " << GEMM_N << "]" << std::endl;
    std::cout << "    Block dim:        " << BLOCK_DIM << std::endl;
    std::cout << std::endl;
    std::cout << "  Test Parameters:" << std::endl;
    std::cout << "    Ranks:            " << args.n_ranks << std::endl;
    std::cout << "    Devices:          " << args.n_devices << std::endl;
    std::cout << "    Warmup iters:     " << args.warmup_iters << std::endl;
    std::cout << "    Measure iters:    " << args.perf_iters << std::endl;
    std::cout << "    Bench mode:       " << mode_str << std::endl;
    std::cout << "    Verify:           " << (args.verify ? "yes" : "no") << std::endl;
    std::cout << "    Verbose:          " << (args.verbose ? "yes" : "no") << std::endl;
    std::cout << "============================================================\n" << std::endl;

    bool success = RunGemmMultiProcess(
        args.n_ranks, args.n_devices, args.first_rank_id, args.first_device_id,
        args.warmup_iters, args.perf_iters, args.bench_mode, args.verify, args.verbose);

    std::cout << "\n============================================================" << std::endl;
    std::cout << "  Benchmark " << (success ? "PASSED" : "FAILED") << std::endl;
    std::cout << "============================================================\n" << std::endl;

    return success ? 0 : 1;
}
