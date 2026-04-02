/**
 * AllGather GEMM Demo - Main Entry Point (HCCL backend)
 *
 * Multi-card AllGather then GEMM: each card first gathers all portions of A
 * via AllGather, then computes the full C = A * B.
 *
 * Tile-level pipelining: comm kernel and compute kernel run on separate
 * streams; the compute kernel starts processing each row group as soon as
 * the comm kernel signals its availability via TileFlagMatrix.
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

static constexpr int WARMUP_ITERS = 5;
static constexpr int MEASURE_ITERS = 100;

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
// Statistics helper
// ============================================================================
static double SimpleSqrt(double x)
{
    if (x <= 0.0) return 0.0;
    double r = x;
    for (int i = 0; i < 64; ++i) {
        r = 0.5 * (r + x / r);
    }
    return r;
}

static auto CalcStats(const std::vector<double> &times)
{
    double sum = 0.0, min_val = times[0], max_val = times[0];
    for (double t : times) {
        sum += t;
        min_val = std::min(min_val, t);
        max_val = std::max(max_val, t);
    }
    double avg = sum / static_cast<double>(times.size());
    double variance = 0.0;
    for (double t : times) {
        variance += (t - avg) * (t - avg);
    }
    double std_dev = SimpleSqrt(variance / static_cast<double>(times.size()));
    return std::make_tuple(avg, min_val, max_val, std_dev);
}

// ============================================================================
// RunAllGatherGemmPerRank: per-rank test harness
// ============================================================================
static bool RunAllGatherGemmPerRank(int rank_id, int n_ranks, int device_id,
                                    const HcclRootInfo* rootInfo,
                                    bool perfMode, const std::string& dataDir)
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

    // Read input data (following reference code naming: pe_*_a.bin, pe_*_b.bin)
    std::string a_file = dataDir + "/pe_" + std::to_string(rank_id) + "_a.bin";
    std::string b_file = dataDir + "/pe_" + std::to_string(rank_id) + "_b.bin";

    size_t m_local = static_cast<size_t>(G_M) / n_ranks;
    size_t aLocalSize = m_local * G_K * sizeof(uint16_t);
    uint16_t* a_local_host = nullptr;
    uint16_t* b_host = nullptr;
    aclrtMallocHost(reinterpret_cast<void**>(&a_local_host), aLocalSize);
    aclrtMallocHost(reinterpret_cast<void**>(&b_host), bSize);

    size_t a_file_size = 0;
    size_t b_file_size = 0;
    if (!PtoTestCommon::ReadFile(a_file, a_file_size, a_local_host, aLocalSize) || a_file_size != aLocalSize) {
        std::cerr << "[ERROR] Rank " << rank_id << ": A file mismatch: " << a_file << std::endl;
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
    // Functional run (always executed once)
    // ------------------------------------------------------------------
    {
        if (rank_id == 0) {
            std::cout << "\n[INFO] Running functional verification..." << std::endl;
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
    }

    // Write output
    {
        float* output_host = nullptr;
        aclrtMallocHost(reinterpret_cast<void**>(&output_host), outputSize);
        aclrtMemcpy(output_host, outputSize, output_dev, outputSize, ACL_MEMCPY_DEVICE_TO_HOST);

        std::string output_file = dataDir + "/output_rank" + std::to_string(rank_id) + ".bin";
        PtoTestCommon::WriteFile(output_file, output_host, outputSize);

        // Verify against golden
        std::string golden_file = dataDir + "/golden.bin";
        size_t goldenSize = static_cast<size_t>(ORIG_M) * ORIG_N * sizeof(float);
        std::vector<float> golden(goldenSize / sizeof(float));
        size_t golden_file_size = 0;
        if (PtoTestCommon::ReadFile(golden_file, golden_file_size, golden.data(), goldenSize) && golden_file_size == goldenSize) {
            if (ORIG_M == G_M && ORIG_N == G_N) {
                is_ok = PtoTestCommon::ResultCmp(golden, output_host, 0.001f);
            } else {
                std::vector<float> valid_output(static_cast<size_t>(ORIG_M) * ORIG_N);
                for (uint32_t row = 0; row < ORIG_M; ++row) {
                    std::memcpy(valid_output.data() + row * ORIG_N,
                                output_host + row * G_N,
                                ORIG_N * sizeof(float));
                }
                is_ok = PtoTestCommon::ResultCmp(golden, valid_output.data(), 0.001f);
            }
        } else {
            std::cerr << "[WARN] Rank " << rank_id << ": golden file not available, skipping verification" << std::endl;
        }
        aclrtFreeHost(output_host);
    }

    if (rank_id == 0) {
        if (is_ok) {
            std::cout << "[INFO] Functional run completed. Verification PASSED." << std::endl;
        } else {
            std::cerr << "[ERROR] Functional run completed. Verification FAILED!" << std::endl;
        }
    }

    // ------------------------------------------------------------------
    // Performance measurement (only when --perf is specified)
    // ------------------------------------------------------------------
    if (perfMode) {
        // Warmup
        if (rank_id == 0) {
            std::cout << "\n[PERF] Warmup (" << WARMUP_ITERS << " iterations)..." << std::endl;
        }

        for (int phase = 0; phase < 4; ++phase) {
            for (int i = 0; i < WARMUP_ITERS; ++i) {
                resetStreamingState();
                if (phase == 1) prePopulateTileFlags();
                HcclHostBarrier(hcclTestCtx.comm, hcclTestCtx.stream);

                if (phase == 0 || phase == 2 || phase == 3) {
                    launchRingCommStreaming(
                        reinterpret_cast<uint8_t*>(shmem_input),
                        reinterpret_cast<uint8_t*>(tile_flag_shmem),
                        reinterpret_cast<uint8_t*>(hcclTestCtx.deviceCtx),
                        n_ranks, commStream);
                    if (phase == 0 || phase == 2) aclrtSynchronizeStream(commStream);
                }
                if (phase == 1 || phase == 2 || phase == 3) {
                    launchAllGatherGemmComputeStreaming(
                        reinterpret_cast<uint8_t*>(output_dev),
                        reinterpret_cast<uint8_t*>(shmem_input),
                        reinterpret_cast<uint8_t*>(src1_dev),
                        reinterpret_cast<uint8_t*>(tile_flag_shmem),
                        computeStream, COMPUTE_BLOCK_NUM);
                    aclrtSynchronizeStream(computeStream);
                }
                if (phase == 3) aclrtSynchronizeStream(commStream);
                HcclHostBarrier(hcclTestCtx.comm, hcclTestCtx.stream);
            }
        }

        if (rank_id == 0) {
            std::cout << "[PERF] Measuring (" << MEASURE_ITERS << " iterations)..." << std::endl;
        }

        // Measure streaming pipelined (the primary metric)
        std::vector<double> times_us;
        times_us.reserve(MEASURE_ITERS);

        for (int iter = 0; iter < MEASURE_ITERS; ++iter) {
            resetStreamingState();
            aclrtSynchronizeStream(commStream);
            aclrtSynchronizeStream(computeStream);
            HcclHostBarrier(hcclTestCtx.comm, hcclTestCtx.stream);

            auto t_start = std::chrono::high_resolution_clock::now();

            launchRingCommStreaming(
                reinterpret_cast<uint8_t*>(shmem_input),
                reinterpret_cast<uint8_t*>(tile_flag_shmem),
                reinterpret_cast<uint8_t*>(hcclTestCtx.deviceCtx),
                n_ranks, commStream);
            launchAllGatherGemmComputeStreaming(
                reinterpret_cast<uint8_t*>(output_dev),
                reinterpret_cast<uint8_t*>(shmem_input),
                reinterpret_cast<uint8_t*>(src1_dev),
                reinterpret_cast<uint8_t*>(tile_flag_shmem),
                computeStream, COMPUTE_BLOCK_NUM);
            aclrtSynchronizeStream(computeStream);
            aclrtSynchronizeStream(commStream);

            auto t_end = std::chrono::high_resolution_clock::now();
            double elapsed_us = std::chrono::duration<double, std::micro>(t_end - t_start).count();
            times_us.push_back(elapsed_us);
            HcclHostBarrier(hcclTestCtx.comm, hcclTestCtx.stream);
        }

        // Print performance results (rank 0 only, matching reference format)
        if (rank_id == 0) {
            auto [avg, min_val, max_val, std_dev] = CalcStats(times_us);

            double gemm_flops = 2.0 * static_cast<double>(G_M) * static_cast<double>(G_K) * static_cast<double>(G_N);
            double tflops = (avg > 0.0) ? (gemm_flops / (avg * 1e-6) / 1e12) : 0.0;

            double m_local_d = static_cast<double>(G_M) / n_ranks;
            double comm_bytes = m_local_d * static_cast<double>(G_K) * sizeof(uint16_t) * (n_ranks - 1);
            double comm_gbps = (avg > 0.0) ? (comm_bytes / (avg * 1e-6) / 1e9) : 0.0;

            double total_read_bytes = static_cast<double>(G_M) * static_cast<double>(G_K) * sizeof(uint16_t)
                                    + static_cast<double>(G_K) * static_cast<double>(G_N) * sizeof(uint16_t);
            double total_write_bytes = static_cast<double>(G_M) * static_cast<double>(G_N) * sizeof(float);
            double mem_bw_gbps = (avg > 0.0) ? ((total_read_bytes + total_write_bytes) / (avg * 1e-6) / 1e9) : 0.0;

#ifndef CONFIG_PEAK_TFLOPS_FP16
#define CONFIG_PEAK_TFLOPS_FP16 320.0
#endif
            constexpr double PEAK_TFLOPS_FP16 = CONFIG_PEAK_TFLOPS_FP16;
            double mfu = tflops / PEAK_TFLOPS_FP16 * 100.0;

            std::cout << std::fixed << std::setprecision(3);
            std::cout << "\n================================================================" << std::endl;
            std::cout << "[PERF] AllGather GEMM Performance Results" << std::endl;
            std::cout << "================================================================" << std::endl;

            std::cout << "\n  Configuration:" << std::endl;
            std::cout << "    M (global):    " << G_M << std::endl;
            if (ORIG_M != G_M || ORIG_K != G_K || ORIG_N != G_N) {
                std::cout << "    M (original):  " << ORIG_M << std::endl;
            }
            std::cout << "    M (per rank):  " << static_cast<int>(m_local_d) << std::endl;
            std::cout << "    K:             " << G_K << std::endl;
            std::cout << "    N:             " << G_N << std::endl;
            std::cout << "    pe_size:       " << n_ranks << std::endl;
            std::cout << "    compute_blocks:" << COMPUTE_BLOCK_NUM << std::endl;
            std::cout << "    comm_blocks:   " << COMM_BLOCK_NUM << std::endl;

            std::cout << "\n  Workload:" << std::endl;
            std::cout << "    GEMM FLOPs:           " << std::scientific << std::setprecision(2)
                      << gemm_flops << std::endl;
            std::cout << std::fixed << std::setprecision(2);
            std::cout << "    Input A (per rank):   "
                      << (m_local_d * G_K * sizeof(uint16_t) / (1024.0 * 1024.0)) << " MB" << std::endl;
            std::cout << "    Input A (global):     "
                      << (static_cast<double>(G_M) * G_K * sizeof(uint16_t) / (1024.0 * 1024.0)) << " MB" << std::endl;
            std::cout << "    Input B:              "
                      << (static_cast<double>(G_K) * G_N * sizeof(uint16_t) / (1024.0 * 1024.0)) << " MB" << std::endl;
            std::cout << "    Output C:             "
                      << (static_cast<double>(G_M) * G_N * sizeof(float) / (1024.0 * 1024.0)) << " MB" << std::endl;
            std::cout << "    Comm (per rank):      "
                      << (comm_bytes / (1024.0 * 1024.0)) << " MB" << std::endl;

            std::cout << std::fixed << std::setprecision(3);
            std::cout << "\n  End-to-End (AllGather + GEMM fused):" << std::endl;
            std::cout << "    Avg Time:      " << avg << " us  (" << (avg / 1000.0) << " ms)" << std::endl;
            std::cout << "    Min Time:      " << min_val << " us" << std::endl;
            std::cout << "    Max Time:      " << max_val << " us" << std::endl;
            std::cout << "    Std Dev:       " << std_dev << " us" << std::endl;

            std::cout << "\n  Throughput:" << std::endl;
            std::cout << "    TFLOPS:        " << tflops << std::endl;
            std::cout << "    MFU:           " << mfu << "% (vs " << PEAK_TFLOPS_FP16 << " TFLOPS peak)" << std::endl;
            std::cout << "    Comm BW:       " << comm_gbps << " GB/s  (AllGather payload / e2e time)" << std::endl;
            std::cout << "    Mem BW:        " << mem_bw_gbps << " GB/s  (total read+write / e2e time)" << std::endl;

            std::cout << "================================================================\n" << std::endl;
        }
    }

    // Cleanup
    aclrtFree(src1_dev);
    aclrtFree(output_dev);
    aclrtFreeHost(tile_flag_host);

    status |= aclrtDestroyStream(computeStream);
    status |= aclrtDestroyStream(commStream);

    bool hcclOk = hcclTestCtx.Finalize();

    return (status == 0) && is_ok && hcclOk;
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

    int n_ranks = 2;
    const char* n_ranks_env = std::getenv("N_RANKS");
    if (n_ranks_env != nullptr) {
        n_ranks = std::atoi(n_ranks_env);
    }

    bool perfMode = false;
    const char* perf_env = std::getenv("ALLGATHER_GEMM_PERF_MODE");
    if (perf_env != nullptr && std::string(perf_env) == "1") {
        perfMode = true;
    }

    std::string dataDir = "../out";
    const char* data_dir_env = std::getenv("ALLGATHER_GEMM_DATA_DIR");
    if (data_dir_env != nullptr) {
        dataDir = data_dir_env;
    }

    if (n_ranks > MAX_RING_RANKS) {
        std::cerr << "[ERROR] n_ranks exceeds MAX_RING_RANKS=" << MAX_RING_RANKS << std::endl;
        CommMpiFinalize();
        return 1;
    }

    int mpiRank = CommMpiRank();
    int mpiSize = CommMpiSize();

    if (mpiSize != n_ranks) {
        if (mpiRank == 0) {
            std::cerr << "[ERROR] MPI world size (" << mpiSize << ") != expected N_RANKS (" << n_ranks
                      << "). Launch with: mpirun -n " << n_ranks << " ./allgather_gemm" << std::endl;
        }
        CommMpiFinalize();
        return 1;
    }

    int rank_id = mpiRank;
    int device_id = rank_id;

    constexpr int kAclRepeatInit = 100002;
    aclError aRet = aclInit(nullptr);
    if (aRet != ACL_SUCCESS && static_cast<int>(aRet) != kAclRepeatInit) {
        std::cerr << "[ERROR] Rank " << rank_id << ": aclInit failed: " << static_cast<int>(aRet) << std::endl;
        CommMpiFinalize();
        return 1;
    }

    if (rank_id == 0) {
        rtSetDevice(device_id);
    }

    aRet = aclrtSetDevice(device_id);
    if (aRet != ACL_SUCCESS) {
        std::cerr << "[ERROR] Rank " << rank_id << ": aclrtSetDevice(" << device_id
                  << ") failed: " << static_cast<int>(aRet) << std::endl;
        CommMpiFinalize();
        return 1;
    }

    HcclRootInfo rootInfo{};
    if (rank_id == 0) {
        HcclResult hret = HcclGetRootInfo(&rootInfo);
        if (hret != HCCL_SUCCESS) {
            std::cerr << "[ERROR] HcclGetRootInfo failed: " << hret << std::endl;
            CommMpiFinalize();
            return 1;
        }
    }

    CommMpiBcast(&rootInfo, HCCL_ROOT_INFO_BYTES, COMM_MPI_CHAR, 0);
    CommMpiBarrier();

    if (rank_id == 0) {
        std::cout << "\n================================================================" << std::endl;
        std::cout << "  AllGather GEMM (HCCL backend)" << std::endl;
        std::cout << "  M=" << G_M << ", K=" << G_K << ", N=" << G_N
                  << ", pe_size=" << n_ranks << std::endl;
        if (ORIG_M != G_M || ORIG_K != G_K || ORIG_N != G_N) {
            std::cout << "  (original: M=" << ORIG_M << ", K=" << ORIG_K << ", N=" << ORIG_N << ")" << std::endl;
        }
        if (perfMode) {
            std::cout << "  Mode: PERFORMANCE (warmup=" << WARMUP_ITERS
                      << ", measure=" << MEASURE_ITERS << ")" << std::endl;
        } else {
            std::cout << "  Mode: FUNCTIONAL VERIFICATION" << std::endl;
        }
        std::cout << "================================================================" << std::endl;
    }

    bool ok = RunAllGatherGemmPerRank(rank_id, n_ranks, device_id, &rootInfo, perfMode, dataDir);

    if (rank_id == 0) {
        if (ok) {
            std::cout << "[SUCCESS] AllGather GEMM demo completed successfully." << std::endl;
        } else {
            std::cout << "[FAILED] AllGather GEMM demo FAILED." << std::endl;
        }
    }

    CommMpiFinalize();
    return ok ? 0 : 1;
}
