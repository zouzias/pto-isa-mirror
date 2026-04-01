/**
 * AllGather GEMM Demo - Main Entry Point (HCCL backend)
 *
 * Multi-card AllGather then GEMM: each card first gathers all portions of A
 * via AllGather, then computes the full C = A * B.
 *
 * Tile-level pipelining: comm kernel and compute kernel run on separate
 * streams; the compute kernel starts processing each row group as soon as
 * the comm kernel signals its availability via RowGroupSignals.
 *
 * This is the producer-consumer mirror of gemm_allgather:
 *   gemm_allgather:  GEMM (producer) → AllGather (consumer)
 *   allgather_gemm:  AllGather (producer) → GEMM (consumer)
 *
 * Launch with: mpirun -n <N_RANKS> ./allgather_gemm
 */

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <string>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <algorithm>
#include <tuple>

#include "ready_queue.hpp"

#ifdef DT_UNDEFINED
#define DT_UNDEFINED_SAVED DT_UNDEFINED
#undef DT_UNDEFINED
#endif
#include "test_common.h"
#ifdef DT_UNDEFINED_SAVED
#define DT_UNDEFINED DT_UNDEFINED_SAVED
#undef DT_UNDEFINED_SAVED
#endif

#include "common.hpp"

// ============================================================================
// Compile-time configuration (injected via CMake -D flags)
// ============================================================================
#ifndef CONFIG_G_M
#define CONFIG_G_M 2048
#endif
#ifndef CONFIG_G_K
#define CONFIG_G_K 2048
#endif
#ifndef CONFIG_G_N
#define CONFIG_G_N 1024
#endif

constexpr uint32_t G_M = CONFIG_G_M;
constexpr uint32_t G_K = CONFIG_G_K;
constexpr uint32_t G_N = CONFIG_G_N;

#ifndef CONFIG_ORIG_M
#define CONFIG_ORIG_M CONFIG_G_M
#endif
#ifndef CONFIG_ORIG_K
#define CONFIG_ORIG_K CONFIG_G_K
#endif
#ifndef CONFIG_ORIG_N
#define CONFIG_ORIG_N CONFIG_G_N
#endif

constexpr uint32_t ORIG_M = CONFIG_ORIG_M;
constexpr uint32_t ORIG_K = CONFIG_ORIG_K;
constexpr uint32_t ORIG_N = CONFIG_ORIG_N;

#ifndef CONFIG_SIZE_NAME
#define CONFIG_SIZE_NAME "large"
#endif
constexpr const char* SIZE_NAME = CONFIG_SIZE_NAME;

constexpr uint32_t G_BASE_M = 128;
constexpr uint32_t G_BASE_K = 64;
constexpr uint32_t G_BASE_N = 256;

#ifndef CONFIG_COMPUTE_BLOCK_NUM
#define CONFIG_COMPUTE_BLOCK_NUM 20
#endif
#ifndef CONFIG_COMM_BLOCK_NUM
#define CONFIG_COMM_BLOCK_NUM 4
#endif
constexpr int COMPUTE_BLOCK_NUM = CONFIG_COMPUTE_BLOCK_NUM;
constexpr int COMM_BLOCK_NUM = CONFIG_COMM_BLOCK_NUM;

constexpr int WARMUP_ITERS = 5;
constexpr int MEASURE_ITERS = 100;

// ============================================================================
// Extern declarations for kernel launch functions
// ============================================================================
extern void launchRingCommStreaming(
    uint8_t* shmem_input,
    uint8_t* tile_flags,
    uint8_t* hccl_ctx,
    int n_ranks,
    void* stream);

extern void launchAllGatherGemmComputeStreaming(
    uint8_t* output,
    uint8_t* shmem_input,
    uint8_t* src1,
    uint8_t* tile_flags,
    void* stream,
    int block_num);

// ============================================================================
// RunAllGatherGemmPerRank: per-rank test harness
// ============================================================================
static bool RunAllGatherGemmPerRank(int rank_id, int n_ranks, int device_id,
                                    const HcclRootInfo* rootInfo)
{
    int status = 0;

    TestContext hcclTestCtx;
    if (!hcclTestCtx.Init(rank_id, n_ranks, n_ranks, 0, rootInfo)) {
        std::cerr << "[ERROR] Rank " << rank_id << ": HCCL TestContext Init failed\n";
        return false;
    }

    aclrtStream computeStream = nullptr;
    aclrtStream commStream = nullptr;
    status |= aclrtCreateStream(&computeStream);
    status |= aclrtCreateStream(&commStream);

    size_t inputShmemBytes = static_cast<size_t>(G_M) * G_K * sizeof(uint16_t);

    int m_tiles = static_cast<int>(G_M / G_BASE_M);
    int m_tiles_local = m_tiles / n_ranks;
    int k_chunks = static_cast<int>(G_K / G_BASE_N);
    int num_blocks_per_src = m_tiles_local * k_chunks;

    int optimal_tile_size = ComputeOptimalTileSize(num_blocks_per_src);
    size_t tileFlagMatrixSize = TileFlagMatrixSize(n_ranks, num_blocks_per_src, optimal_tile_size);
    size_t tileFlagWithSummarySize = TileFlagMatrixWithSummarySize(n_ranks, num_blocks_per_src, optimal_tile_size);

    uint64_t localWinBase = hcclTestCtx.hostCtx.windowsIn[rank_id];
    size_t winOffset = 0;
    void* shmem_input = WindowAlloc(localWinBase, winOffset, inputShmemBytes);
    aclrtMemset(shmem_input, inputShmemBytes, 0, inputShmemBytes);

    void* tile_flag_shmem = WindowAlloc(localWinBase, winOffset, tileFlagWithSummarySize);
    TileFlagMatrix* tile_flag_host = nullptr;
    aclrtMallocHost(reinterpret_cast<void**>(&tile_flag_host), tileFlagWithSummarySize);
    TileFlagMatrixInit(tile_flag_host, n_ranks, num_blocks_per_src, optimal_tile_size);
    tile_flag_host->my_rank = rank_id;
    int32_t* summary_host = reinterpret_cast<int32_t*>(reinterpret_cast<uint8_t*>(tile_flag_host) + tileFlagMatrixSize);
    TileFlagMatrixSummaryInit(summary_host, n_ranks);
    aclrtMemcpy(tile_flag_shmem, tileFlagWithSummarySize, tile_flag_host, tileFlagWithSummarySize, ACL_MEMCPY_HOST_TO_DEVICE);

    size_t bSize = static_cast<size_t>(G_K) * G_N * sizeof(uint16_t);
    void* src1_dev = nullptr;
    aclrtMalloc(&src1_dev, bSize, ACL_MEM_MALLOC_HUGE_FIRST);

    size_t outputSize = static_cast<size_t>(G_M) * G_N * sizeof(float);
    void* output_dev = nullptr;
    aclrtMalloc(&output_dev, outputSize, ACL_MEM_MALLOC_HUGE_FIRST);

    std::string input_dir = std::string("../input/") + SIZE_NAME;
    std::string a_file = input_dir + "/a_rank" + std::to_string(rank_id) + ".bin";
    std::string b_file = input_dir + "/b.bin";

    size_t m_local = static_cast<size_t>(G_M) / n_ranks;
    size_t aLocalSize = m_local * G_K * sizeof(uint16_t);
    uint16_t* a_local_host = nullptr;
    uint16_t* b_host = nullptr;
    aclrtMallocHost(reinterpret_cast<void**>(&a_local_host), aLocalSize);
    aclrtMallocHost(reinterpret_cast<void**>(&b_host), bSize);

    size_t a_file_size = 0;
    size_t b_file_size = 0;
    if (!PtoTestCommon::ReadFile(a_file, a_file_size, a_local_host, aLocalSize) || a_file_size != aLocalSize) {
        std::cerr << "[ERROR] Rank " << rank_id << ": A file mismatch for M-slice: " << a_file << std::endl;
        return false;
    }
    if (!PtoTestCommon::ReadFile(b_file, b_file_size, b_host, bSize) || b_file_size != bSize) {
        std::cerr << "[ERROR] Rank " << rank_id << ": B file mismatch: " << b_file << std::endl;
        return false;
    }

    aclrtMemcpy(reinterpret_cast<uint8_t*>(shmem_input) + rank_id * aLocalSize,
                aLocalSize, a_local_host, aLocalSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src1_dev, bSize, b_host, bSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtFreeHost(a_local_host);
    aclrtFreeHost(b_host);

    HcclHostBarrier(hcclTestCtx.comm, hcclTestCtx.stream);

    // ------------------------------------------------------------------
    // Helper lambdas
    // ------------------------------------------------------------------
    bool is_ok = true;

    auto resetStreamingState = [&]() {
        TileFlagMatrixReset(tile_flag_host);
        TileFlagMatrixSummaryInit(summary_host, n_ranks);
        aclrtMemcpy(tile_flag_shmem, tileFlagWithSummarySize, tile_flag_host, tileFlagWithSummarySize, ACL_MEMCPY_HOST_TO_DEVICE);
        aclrtMemset(output_dev, outputSize, 0, outputSize);
    };

    auto prePopulateTileFlags = [&]() {
        for (int src = 0; src < n_ranks; ++src) {
            TileFlagMatrixSetLocalReady(tile_flag_host, src);
        }
        int num_tiles = tile_flag_host->num_tiles_per_src;
        for (int src = 0; src < n_ranks; ++src) {
            summary_host[src] = num_tiles;
        }
        aclrtMemcpy(tile_flag_shmem, tileFlagWithSummarySize, tile_flag_host, tileFlagWithSummarySize, ACL_MEMCPY_HOST_TO_DEVICE);
    };

    // ------------------------------------------------------------------
    // Warmup Phase 1: Comm kernel only
    // ------------------------------------------------------------------
    if (rank_id == 0) {
        std::cout << "\n[PERF] Starting comm-only warmup (" << WARMUP_ITERS << " iterations)..." << std::endl;
    }

    for (int i = 0; i < WARMUP_ITERS; ++i) {
        resetStreamingState();
        HcclHostBarrier(hcclTestCtx.comm, hcclTestCtx.stream);

        launchRingCommStreaming(
            reinterpret_cast<uint8_t*>(shmem_input),
            reinterpret_cast<uint8_t*>(tile_flag_shmem),
            reinterpret_cast<uint8_t*>(hcclTestCtx.deviceCtx),
            n_ranks,
            commStream);
        aclrtSynchronizeStream(commStream);

        if (rank_id == 0) std::cout << "  Comm warmup " << i << " done" << std::endl;
        HcclHostBarrier(hcclTestCtx.comm, hcclTestCtx.stream);
    }

    // ------------------------------------------------------------------
    // Warmup Phase 2: Compute kernel only (pre-populate tile flags)
    // ------------------------------------------------------------------
    if (rank_id == 0) {
        std::cout << "[PERF] Starting compute-only warmup (" << WARMUP_ITERS << " iterations)..." << std::endl;
    }

    for (int i = 0; i < WARMUP_ITERS; ++i) {
        resetStreamingState();
        prePopulateTileFlags();
        HcclHostBarrier(hcclTestCtx.comm, hcclTestCtx.stream);

        launchAllGatherGemmComputeStreaming(
            reinterpret_cast<uint8_t*>(output_dev),
            reinterpret_cast<uint8_t*>(shmem_input),
            reinterpret_cast<uint8_t*>(src1_dev),
            reinterpret_cast<uint8_t*>(tile_flag_shmem),
            computeStream,
            COMPUTE_BLOCK_NUM);
        aclrtSynchronizeStream(computeStream);

        if (rank_id == 0) std::cout << "  Compute warmup " << i << " done" << std::endl;
        HcclHostBarrier(hcclTestCtx.comm, hcclTestCtx.stream);
    }

    // ------------------------------------------------------------------
    // Warmup Phase 3: Sequential (comm then compute)
    // ------------------------------------------------------------------
    if (rank_id == 0) {
        std::cout << "[PERF] Starting sequential warmup (" << WARMUP_ITERS << " iterations)..." << std::endl;
    }

    for (int i = 0; i < WARMUP_ITERS; ++i) {
        resetStreamingState();
        HcclHostBarrier(hcclTestCtx.comm, hcclTestCtx.stream);

        launchRingCommStreaming(
            reinterpret_cast<uint8_t*>(shmem_input),
            reinterpret_cast<uint8_t*>(tile_flag_shmem),
            reinterpret_cast<uint8_t*>(hcclTestCtx.deviceCtx),
            n_ranks,
            commStream);
        aclrtSynchronizeStream(commStream);
        HcclHostBarrier(hcclTestCtx.comm, hcclTestCtx.stream);

        launchAllGatherGemmComputeStreaming(
            reinterpret_cast<uint8_t*>(output_dev),
            reinterpret_cast<uint8_t*>(shmem_input),
            reinterpret_cast<uint8_t*>(src1_dev),
            reinterpret_cast<uint8_t*>(tile_flag_shmem),
            computeStream,
            COMPUTE_BLOCK_NUM);
        aclrtSynchronizeStream(computeStream);

        if (rank_id == 0) std::cout << "  Sequential warmup " << i << " done" << std::endl;
        HcclHostBarrier(hcclTestCtx.comm, hcclTestCtx.stream);
    }

    // ------------------------------------------------------------------
    // Warmup Phase 4: Streaming pipelined (concurrent comm + compute)
    // ------------------------------------------------------------------
    if (rank_id == 0) {
        std::cout << "[PERF] Starting streaming pipelined warmup (" << WARMUP_ITERS << " iterations)..." << std::endl;
    }

    for (int i = 0; i < WARMUP_ITERS; ++i) {
        resetStreamingState();
        HcclHostBarrier(hcclTestCtx.comm, hcclTestCtx.stream);

        launchRingCommStreaming(
            reinterpret_cast<uint8_t*>(shmem_input),
            reinterpret_cast<uint8_t*>(tile_flag_shmem),
            reinterpret_cast<uint8_t*>(hcclTestCtx.deviceCtx),
            n_ranks,
            commStream);
        launchAllGatherGemmComputeStreaming(
            reinterpret_cast<uint8_t*>(output_dev),
            reinterpret_cast<uint8_t*>(shmem_input),
            reinterpret_cast<uint8_t*>(src1_dev),
            reinterpret_cast<uint8_t*>(tile_flag_shmem),
            computeStream,
            COMPUTE_BLOCK_NUM);
        aclrtSynchronizeStream(computeStream);
        aclrtSynchronizeStream(commStream);

        if (rank_id == 0) std::cout << "  Streaming pipelined warmup " << i << " done" << std::endl;
        HcclHostBarrier(hcclTestCtx.comm, hcclTestCtx.stream);
    }

    // ------------------------------------------------------------------
    // Functional verification after warmup
    // ------------------------------------------------------------------
    auto verifyAgainstGolden = [&](const char* phase_name) -> bool {
        float* verify_host = nullptr;
        aclrtMallocHost(reinterpret_cast<void**>(&verify_host), outputSize);
        aclrtMemcpy(verify_host, outputSize, output_dev, outputSize, ACL_MEMCPY_DEVICE_TO_HOST);

        std::string output_dir = std::string("../output/") + SIZE_NAME;
        std::string output_file = output_dir + "/output_rank" + std::to_string(rank_id) + ".bin";
        PtoTestCommon::WriteFile(output_file, verify_host, outputSize);

        std::string golden_file = output_dir + "/golden.bin";
        size_t goldenSize = static_cast<size_t>(ORIG_M) * ORIG_N * sizeof(float);
        std::vector<float> golden(goldenSize / sizeof(float));
        size_t golden_file_size = 0;
        bool rank_ok = false;
        if (PtoTestCommon::ReadFile(golden_file, golden_file_size, golden.data(), goldenSize) && golden_file_size == goldenSize) {
            if (ORIG_M == G_M && ORIG_N == G_N) {
                rank_ok = PtoTestCommon::ResultCmp(golden, verify_host, 0.001f);
            } else {
                std::vector<float> valid_output(static_cast<size_t>(ORIG_M) * ORIG_N);
                for (uint32_t row = 0; row < ORIG_M; ++row) {
                    std::memcpy(valid_output.data() + row * ORIG_N,
                                verify_host + row * G_N,
                                ORIG_N * sizeof(float));
                }
                rank_ok = PtoTestCommon::ResultCmp(golden, valid_output.data(), 0.001f);
            }
            if (rank_ok) {
                std::cout << "[INFO] Rank " << rank_id << " " << phase_name << " verification passed!" << std::endl;
            } else {
                std::cerr << "[ERROR] Rank " << rank_id << " " << phase_name << " verification FAILED!" << std::endl;
            }
        } else {
            std::cerr << "[ERROR] Rank " << rank_id << ": golden file not available or size mismatch: "
                      << golden_file << " (expected " << goldenSize << " bytes, got " << golden_file_size << ")" << std::endl;
            rank_ok = false;
        }
        aclrtFreeHost(verify_host);
        return rank_ok;
    };

    {
        if (rank_id == 0) {
            std::cout << "\n[VERIFY] Running functional verification (all ranks)..." << std::endl;
        }
        resetStreamingState();
        HcclHostBarrier(hcclTestCtx.comm, hcclTestCtx.stream);

        launchRingCommStreaming(
            reinterpret_cast<uint8_t*>(shmem_input),
            reinterpret_cast<uint8_t*>(tile_flag_shmem),
            reinterpret_cast<uint8_t*>(hcclTestCtx.deviceCtx),
            n_ranks,
            commStream);
        launchAllGatherGemmComputeStreaming(
            reinterpret_cast<uint8_t*>(output_dev),
            reinterpret_cast<uint8_t*>(shmem_input),
            reinterpret_cast<uint8_t*>(src1_dev),
            reinterpret_cast<uint8_t*>(tile_flag_shmem),
            computeStream,
            COMPUTE_BLOCK_NUM);
        aclrtSynchronizeStream(computeStream);
        aclrtSynchronizeStream(commStream);
        HcclHostBarrier(hcclTestCtx.comm, hcclTestCtx.stream);

        is_ok = verifyAgainstGolden("post-warmup");
        HcclHostBarrier(hcclTestCtx.comm, hcclTestCtx.stream);

        if (!is_ok) {
            std::cerr << "[ERROR] Rank " << rank_id << ": Functional verification failed, aborting performance measurement." << std::endl;
        }
    }

    // ------------------------------------------------------------------
    // Performance measurement
    // ------------------------------------------------------------------
    if (rank_id == 0) {
        std::cout << "[PERF] Starting measurement (" << MEASURE_ITERS << " iterations)..." << std::endl;
    }

    HcclHostBarrier(hcclTestCtx.comm, hcclTestCtx.stream);

    std::vector<double> comm_times_us;
    for (int iter = 0; iter < MEASURE_ITERS; ++iter) {
        resetStreamingState();
        aclrtSynchronizeStream(commStream);
        HcclHostBarrier(hcclTestCtx.comm, hcclTestCtx.stream);

        auto comm_start = std::chrono::high_resolution_clock::now();
        launchRingCommStreaming(
            reinterpret_cast<uint8_t*>(shmem_input),
            reinterpret_cast<uint8_t*>(tile_flag_shmem),
            reinterpret_cast<uint8_t*>(hcclTestCtx.deviceCtx),
            n_ranks,
            commStream);
        aclrtSynchronizeStream(commStream);
        auto comm_end = std::chrono::high_resolution_clock::now();
        double comm_time = std::chrono::duration<double, std::micro>(comm_end - comm_start).count();
        comm_times_us.push_back(comm_time);
        HcclHostBarrier(hcclTestCtx.comm, hcclTestCtx.stream);
    }

    std::vector<double> compute_times_us;
    for (int iter = 0; iter < MEASURE_ITERS; ++iter) {
        resetStreamingState();
        prePopulateTileFlags();
        aclrtSynchronizeStream(computeStream);
        HcclHostBarrier(hcclTestCtx.comm, hcclTestCtx.stream);

        auto compute_start = std::chrono::high_resolution_clock::now();
        launchAllGatherGemmComputeStreaming(
            reinterpret_cast<uint8_t*>(output_dev),
            reinterpret_cast<uint8_t*>(shmem_input),
            reinterpret_cast<uint8_t*>(src1_dev),
            reinterpret_cast<uint8_t*>(tile_flag_shmem),
            computeStream,
            COMPUTE_BLOCK_NUM);
        aclrtSynchronizeStream(computeStream);
        auto compute_end = std::chrono::high_resolution_clock::now();
        double compute_time = std::chrono::duration<double, std::micro>(compute_end - compute_start).count();
        compute_times_us.push_back(compute_time);
        HcclHostBarrier(hcclTestCtx.comm, hcclTestCtx.stream);
    }

    std::vector<double> sequential_times_us;
    for (int iter = 0; iter < MEASURE_ITERS; ++iter) {
        resetStreamingState();
        aclrtSynchronizeStream(commStream);
        aclrtSynchronizeStream(computeStream);
        HcclHostBarrier(hcclTestCtx.comm, hcclTestCtx.stream);

        auto seq_start = std::chrono::high_resolution_clock::now();
        launchRingCommStreaming(
            reinterpret_cast<uint8_t*>(shmem_input),
            reinterpret_cast<uint8_t*>(tile_flag_shmem),
            reinterpret_cast<uint8_t*>(hcclTestCtx.deviceCtx),
            n_ranks,
            commStream);
        aclrtSynchronizeStream(commStream);
        launchAllGatherGemmComputeStreaming(
            reinterpret_cast<uint8_t*>(output_dev),
            reinterpret_cast<uint8_t*>(shmem_input),
            reinterpret_cast<uint8_t*>(src1_dev),
            reinterpret_cast<uint8_t*>(tile_flag_shmem),
            computeStream,
            COMPUTE_BLOCK_NUM);
        aclrtSynchronizeStream(computeStream);
        auto seq_end = std::chrono::high_resolution_clock::now();
        double seq_time_us = std::chrono::duration<double, std::micro>(seq_end - seq_start).count();
        sequential_times_us.push_back(seq_time_us);
        HcclHostBarrier(hcclTestCtx.comm, hcclTestCtx.stream);
    }

    std::vector<double> streaming_times_us;
    for (int iter = 0; iter < MEASURE_ITERS; ++iter) {
        resetStreamingState();
        aclrtSynchronizeStream(commStream);
        aclrtSynchronizeStream(computeStream);
        HcclHostBarrier(hcclTestCtx.comm, hcclTestCtx.stream);

        auto stream_start = std::chrono::high_resolution_clock::now();
        launchRingCommStreaming(
            reinterpret_cast<uint8_t*>(shmem_input),
            reinterpret_cast<uint8_t*>(tile_flag_shmem),
            reinterpret_cast<uint8_t*>(hcclTestCtx.deviceCtx),
            n_ranks,
            commStream);
        launchAllGatherGemmComputeStreaming(
            reinterpret_cast<uint8_t*>(output_dev),
            reinterpret_cast<uint8_t*>(shmem_input),
            reinterpret_cast<uint8_t*>(src1_dev),
            reinterpret_cast<uint8_t*>(tile_flag_shmem),
            computeStream,
            COMPUTE_BLOCK_NUM);
        aclrtSynchronizeStream(computeStream);
        aclrtSynchronizeStream(commStream);
        auto stream_end = std::chrono::high_resolution_clock::now();
        double stream_time_us = std::chrono::duration<double, std::micro>(stream_end - stream_start).count();
        streaming_times_us.push_back(stream_time_us);
        HcclHostBarrier(hcclTestCtx.comm, hcclTestCtx.stream);
    }

    // ------------------------------------------------------------------
    // Final verification
    // ------------------------------------------------------------------
    HcclHostBarrier(hcclTestCtx.comm, hcclTestCtx.stream);

    is_ok = verifyAgainstGolden("final");

    // ------------------------------------------------------------------
    // Performance statistics (rank 0 only)
    // ------------------------------------------------------------------
    if (rank_id == 0) {
        auto calc_stats = [](const std::vector<double>& times) {
            double sum = 0.0, min_val = times[0], max_val = times[0];
            for (double t : times) {
                sum += t;
                min_val = std::min(min_val, t);
                max_val = std::max(max_val, t);
            }
            double avg = sum / times.size();
            double variance = 0.0;
            for (double t : times) {
                variance += (t - avg) * (t - avg);
            }
            double std_dev = std::sqrt(variance / times.size());
            return std::make_tuple(avg, min_val, max_val, std_dev);
        };

        auto [seq_avg, seq_min, seq_max, seq_std] = calc_stats(sequential_times_us);
        auto [stream_avg, stream_min, stream_max, stream_std] = calc_stats(streaming_times_us);
        auto [compute_avg, compute_min, compute_max, compute_std] = calc_stats(compute_times_us);
        auto [comm_avg, comm_min, comm_max, comm_std] = calc_stats(comm_times_us);

        double gemm_flops = 2.0 * static_cast<double>(G_M) * static_cast<double>(G_K) * static_cast<double>(G_N);
        double compute_tflops = (compute_avg > 0.0) ? (gemm_flops / (compute_avg * 1e-6) / 1e12) : 0.0;
        double seq_tflops = (seq_avg > 0.0) ? (gemm_flops / (seq_avg * 1e-6) / 1e12) : 0.0;
        double stream_tflops = (stream_avg > 0.0) ? (gemm_flops / (stream_avg * 1e-6) / 1e12) : 0.0;

#ifndef CONFIG_PEAK_TFLOPS_FP16
#define CONFIG_PEAK_TFLOPS_FP16 320.0
#endif
        constexpr double PEAK_TFLOPS_FP16 = CONFIG_PEAK_TFLOPS_FP16;
        double compute_mfu = compute_tflops / PEAK_TFLOPS_FP16 * 100.0;
        double seq_mfu = seq_tflops / PEAK_TFLOPS_FP16 * 100.0;
        double stream_mfu = stream_tflops / PEAK_TFLOPS_FP16 * 100.0;

        double m_local_d = static_cast<double>(G_M) / n_ranks;
        double comm_bytes = m_local_d * static_cast<double>(G_K) * sizeof(uint16_t) * (n_ranks - 1);
        double comm_gbps = (comm_avg > 0.0) ? (comm_bytes / (comm_avg * 1e-6) / 1e9) : 0.0;

        std::cout << "\n================================================================" << std::endl;
        if (is_ok) {
            std::cout << "[SUCCESS] AllGather GEMM (M-slice) test PASSED!" << std::endl;
        } else {
            std::cout << "[FAILED] AllGather GEMM (M-slice) test FAILED!" << std::endl;
        }
        std::cout << "  Matrix dimensions (padded): M=" << G_M << ", K=" << G_K << ", N=" << G_N << std::endl;
        if (ORIG_M != G_M || ORIG_K != G_K || ORIG_N != G_N) {
            std::cout << "  Matrix dimensions (original): M=" << ORIG_M << ", K=" << ORIG_K << ", N=" << ORIG_N << std::endl;
        }
        std::cout << "  Size configuration: " << SIZE_NAME << std::endl;
        std::cout << "  Ranks: " << n_ranks << ", M_local per rank: " << static_cast<int>(m_local_d) << std::endl;
        std::cout << "  Signaling: TileFlagMatrix (M-slice streaming)" << std::endl;

        std::cout << "\n  Workload Analysis:" << std::endl;
        std::cout << "    GEMM FLOPs (per rank, full M): " << std::scientific << std::setprecision(2) << gemm_flops << std::endl;
        std::cout << "    Input A size (per rank M-slice):  " << std::fixed << std::setprecision(2)
                  << (m_local_d * static_cast<double>(G_K) * sizeof(uint16_t) / (1024.0 * 1024.0)) << " MB" << std::endl;
        std::cout << "    Input A size (global):            "
                  << (static_cast<double>(G_M) * G_K * sizeof(uint16_t) / (1024.0 * 1024.0)) << " MB" << std::endl;
        std::cout << "    Comm bytes (per rank):            " << (comm_bytes / (1024.0 * 1024.0))
                  << " MB (local_M_slice * (n_ranks-1))" << std::endl;
        std::cout << "    Output C size:                    "
                  << (static_cast<double>(G_M) * G_N * sizeof(float) / (1024.0 * 1024.0)) << " MB" << std::endl;

        std::cout << "\n[PERF] Performance Results (n_iters=" << MEASURE_ITERS << "):" << std::endl;
        std::cout << std::fixed << std::setprecision(3);

        std::cout << "\n  Comm Kernel (AllGather M-slice, streaming):" << std::endl;
        std::cout << "    Avg Time:  " << comm_avg << " us" << std::endl;
        std::cout << "    Min Time:  " << comm_min << " us" << std::endl;
        std::cout << "    Max Time:  " << comm_max << " us" << std::endl;
        std::cout << "    Std Dev:   " << comm_std << " us" << std::endl;
        std::cout << "    Bandwidth: " << comm_gbps << " GB/s" << std::endl;

        std::cout << "\n  Compute Kernel (GEMM, standalone with pre-populated tile flags):" << std::endl;
        std::cout << "    Avg Time:  " << compute_avg << " us" << std::endl;
        std::cout << "    Min Time:  " << compute_min << " us" << std::endl;
        std::cout << "    Max Time:  " << compute_max << " us" << std::endl;
        std::cout << "    Std Dev:   " << compute_std << " us" << std::endl;
        std::cout << "    Throughput: " << compute_tflops << " TFLOPS (per rank)" << std::endl;
        std::cout << "    MFU:       " << compute_mfu << "% (vs " << PEAK_TFLOPS_FP16 << " TFLOPS peak)" << std::endl;

        std::cout << "\n  Component Summary:" << std::endl;
        std::cout << "    Comm Avg:          " << comm_avg << " us" << std::endl;
        std::cout << "    Compute Avg:       " << compute_avg << " us" << std::endl;
        std::cout << "    Sequential Sum:    " << (comm_avg + compute_avg) << " us" << std::endl;

        std::cout << "\n  Sequential Execution (Comm -> Compute, no overlap):" << std::endl;
        std::cout << "    Avg Time:  " << seq_avg << " us" << std::endl;
        std::cout << "    Min Time:  " << seq_min << " us" << std::endl;
        std::cout << "    Max Time:  " << seq_max << " us" << std::endl;
        std::cout << "    Std Dev:   " << seq_std << " us" << std::endl;
        std::cout << "    Throughput: " << seq_tflops << " TFLOPS" << std::endl;
        std::cout << "    MFU:       " << seq_mfu << "%" << std::endl;

        std::cout << "\n  *** STREAMING Pipelined (Tile-based, fine-grained overlap): ***" << std::endl;
        std::cout << "    Tile size: " << optimal_tile_size << " blocks (dynamic)" << std::endl;
        std::cout << "    Actual tiles per src: " << tile_flag_host->num_tiles_per_src << std::endl;
        std::cout << "    Avg Time:  " << stream_avg << " us" << std::endl;
        std::cout << "    Min Time:  " << stream_min << " us" << std::endl;
        std::cout << "    Max Time:  " << stream_max << " us" << std::endl;
        std::cout << "    Std Dev:   " << stream_std << " us" << std::endl;
        std::cout << "    Throughput: " << stream_tflops << " TFLOPS" << std::endl;
        std::cout << "    MFU:       " << stream_mfu << "%" << std::endl;

        double speedup_stream = seq_avg / stream_avg;
        std::cout << "\n  Performance Comparison:" << std::endl;
        std::cout << "    Speedup (Streaming vs Sequential):  " << speedup_stream << "x" << std::endl;
        std::cout << "    Time Saved (Streaming): " << (seq_avg - stream_avg) << " us ("
                  << ((seq_avg - stream_avg) / seq_avg * 100.0) << "%)" << std::endl;

        std::cout << "\n  MFU Summary (vs " << PEAK_TFLOPS_FP16 << " TFLOPS FP16 peak):" << std::endl;
        std::cout << "    Compute-only MFU:  " << compute_mfu << "%" << std::endl;
        std::cout << "    Sequential MFU:    " << seq_mfu << "%" << std::endl;
        std::cout << "    Streaming MFU:     " << stream_mfu << "%" << std::endl;

        double theoretical_min = std::max(compute_avg, comm_avg);
        double max_possible_saved = seq_avg - theoretical_min;
        double stream_saved = seq_avg - stream_avg;
        double stream_overhead = stream_avg - theoretical_min;
        double stream_efficiency = 0.0;
        if (max_possible_saved > 0.001) {
            stream_efficiency = (stream_saved / max_possible_saved) * 100.0;
            stream_efficiency = std::max(0.0, std::min(100.0, stream_efficiency));
        } else {
            stream_efficiency = 100.0;
        }

        std::cout << "\n  Overlap Analysis (Streaming Pipelined):" << std::endl;
        std::cout << "    Sequential time:                      " << seq_avg << " us" << std::endl;
        std::cout << "    Theoretical min (max(compute, comm)): " << theoretical_min << " us" << std::endl;
        std::cout << "    Actual streaming time:                " << stream_avg << " us" << std::endl;
        std::cout << "    Time saved by streaming overlap:      " << stream_saved << " us" << std::endl;
        std::cout << "    Overhead (vs theoretical):            " << stream_overhead << " us" << std::endl;
        std::cout << "    Streaming overlap efficiency:         " << stream_efficiency << "%" << std::endl;

        if (stream_avg > compute_avg) {
            double comm_remaining = stream_avg - compute_avg;
            std::cout << "\n  Streaming analysis:" << std::endl;
            std::cout << "    - Streaming time (" << stream_avg << " us) > Compute time ("
                      << compute_avg << " us)" << std::endl;
            std::cout << "    - Communication continues for " << comm_remaining << " us after compute finishes" << std::endl;
        } else {
            std::cout << "\n  Streaming compute successfully hides communication!" << std::endl;
            std::cout << "    Streaming time (" << stream_avg << " us) ~ Compute time ("
                      << compute_avg << " us)" << std::endl;
        }

        std::cout << "================================================================\n" << std::endl;
    }

    aclrtFree(src1_dev);
    aclrtFree(output_dev);
    aclrtFreeHost(tile_flag_host);

    status |= aclrtDestroyStream(computeStream);
    status |= aclrtDestroyStream(commStream);

    bool hcclOk = hcclTestCtx.Finalize();

    return (status == 0) && is_ok && hcclOk;
}

// ============================================================================
// RunAllGatherGemm: MPI init, parameter printing, per-rank dispatch
// ============================================================================
static bool RunAllGatherGemm()
{
    int n_ranks = 2;
    const char* n_ranks_env = std::getenv("N_RANKS");
    if (n_ranks_env != nullptr) {
        n_ranks = std::atoi(n_ranks_env);
    }

    if (n_ranks > MAX_RING_RANKS) {
        std::cerr << "[ERROR] n_ranks exceeds MAX_RING_RANKS=" << MAX_RING_RANKS << std::endl;
        return false;
    }

    int mpiRank = CommMpiRank();
    int mpiSize = CommMpiSize();

    if (mpiSize != n_ranks) {
        if (mpiRank == 0) {
            std::cerr << "[ERROR] MPI world size (" << mpiSize << ") != expected N_RANKS (" << n_ranks
                      << "). Launch with: mpirun -n " << n_ranks << " ./allgather_gemm" << std::endl;
        }
        return false;
    }

    int rank_id = mpiRank;
    int device_id = rank_id;

    constexpr int kAclRepeatInit = 100002;
    aclError aRet = aclInit(nullptr);
    if (aRet != ACL_SUCCESS && static_cast<int>(aRet) != kAclRepeatInit) {
        std::cerr << "[ERROR] Rank " << rank_id << ": aclInit failed: " << static_cast<int>(aRet) << std::endl;
        return false;
    }

    if (rank_id == 0) {
        rtSetDevice(device_id);
    }

    aRet = aclrtSetDevice(device_id);
    if (aRet != ACL_SUCCESS) {
        std::cerr << "[ERROR] Rank " << rank_id << ": aclrtSetDevice(" << device_id
                  << ") failed: " << static_cast<int>(aRet) << std::endl;
        return false;
    }

    HcclRootInfo rootInfo{};
    if (rank_id == 0) {
        HcclResult hret = HcclGetRootInfo(&rootInfo);
        if (hret != HCCL_SUCCESS) {
            std::cerr << "[ERROR] HcclGetRootInfo failed: " << hret << std::endl;
            return false;
        }
    }

    CommMpiBcast(&rootInfo, HCCL_ROOT_INFO_BYTES, COMM_MPI_CHAR, 0);
    CommMpiBarrier();

    int m_tiles = static_cast<int>(G_M / G_BASE_M);
    int n_tiles = static_cast<int>(G_N / G_BASE_N);
    int m_local = static_cast<int>(G_M) / n_ranks;
    int k_chunks = static_cast<int>(G_K / G_BASE_N);
    int num_blocks_per_src = (m_tiles / n_ranks) * k_chunks;
    int num_tiles = m_tiles * n_tiles;

    if (rank_id == 0) {
        std::cout << "\n================================================================" << std::endl;
        std::cout << "  AllGather GEMM (M-slice) Performance Test Suite (HCCL backend)" << std::endl;
        std::cout << "  (AllGather=producer, GEMM=consumer, M-dimension partitioning)" << std::endl;
        std::cout << "  Signaling: TileFlagMatrix + TNOTIFY AtomicAdd + Polling (M-slice streaming)" << std::endl;
        std::cout << "================================================================" << std::endl;

        std::cout << "  Current compiled configuration:" << std::endl;
        std::cout << "    M=" << G_M << ", K=" << G_K << ", N=" << G_N;
        if (ORIG_M != G_M || ORIG_K != G_K || ORIG_N != G_N) {
            std::cout << " (padded from " << ORIG_M << "x" << ORIG_K << "x" << ORIG_N << ")";
        }
        std::cout << std::endl;
        std::cout << "    n_ranks=" << n_ranks << ", M_local=" << m_local << " per rank" << std::endl;
        std::cout << "    Base tile: " << G_BASE_M << "x" << G_BASE_K << "x" << G_BASE_N << std::endl;
        std::cout << "    Total blocks per source rank: " << num_blocks_per_src
                  << " (" << (m_tiles / n_ranks) << " M-tiles x " << k_chunks << " K-blocks)" << std::endl;
        std::cout << "    Total compute tiles (per rank): " << num_tiles
                  << " (" << m_tiles << "x" << n_tiles << ")" << std::endl;

        int num_remote_ranks = n_ranks - 1;
        int blocks_per_dest = (num_remote_ranks > 0) ? (COMM_BLOCK_NUM / num_remote_ranks) : 0;
        if (blocks_per_dest < 1) blocks_per_dest = 1;
        int actual_comm_blocks = num_remote_ranks * blocks_per_dest;
        std::cout << "  Parallelization:" << std::endl;
        std::cout << "    Comm kernel blocks: " << actual_comm_blocks << " (streaming AllGather M-slice, CONFIG=" << COMM_BLOCK_NUM << ")" << std::endl;
        std::cout << "    Compute kernel blocks: " << COMPUTE_BLOCK_NUM << " (GEMM consumer)" << std::endl;
        std::cout << "  Performance test config:" << std::endl;
        std::cout << "    Warmup iterations: " << WARMUP_ITERS << std::endl;
        std::cout << "    Measurement iterations: " << MEASURE_ITERS << std::endl;
        std::cout << "================================================================" << std::endl;
    }

    bool ok = RunAllGatherGemmPerRank(rank_id, n_ranks, device_id, &rootInfo);
    return ok;
}

// ============================================================================
// main
// ============================================================================
int main(int argc, char** argv)
{
    if (!CommMpiInit(&argc, &argv)) {
        fprintf(stderr, "[FATAL] CommMpiInit failed. Launch with: mpirun -n <N> ./allgather_gemm\n");
        return 1;
    }

    bool ok = RunAllGatherGemm();

    int rank = CommMpiRank();
    if (rank == 0) {
        if (ok) {
            printf("AllGather GEMM demo completed successfully.\n");
        } else {
            printf("AllGather GEMM demo FAILED.\n");
        }
    }

    CommMpiFinalize();
    return ok ? 0 : 1;
}
