/**
 * Benchmark: Single Compute Core Performance
 * 
 * This program tests the GEMM computation efficiency of a single compute core.
 * It measures how much computation a single compute core can perform per second.
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <vector>

// ACL runtime headers  
#include "acl/acl.h"
#include "acl/acl_rt.h"

// Forward declaration from gemm_compute_kernel.cpp
extern void launchGemmCompute(uint8_t *shmem_output, uint8_t *src0, uint8_t *src1,
                              uint8_t *queue_set, int rank, void *stream, int block_num, uint32_t k_per_rank);

// Use same parameters as main program
// Note: For single core test, use full K dimension (not split)
constexpr uint32_t G_M = 16384;
constexpr uint32_t G_K = 4096;  // Full K dimension for single core test
constexpr uint32_t G_N = 4096;
constexpr uint32_t G_BASE_M = 128;
constexpr uint32_t G_BASE_N = 256;
constexpr uint32_t G_M_TILES = G_M / G_BASE_M;
constexpr uint32_t G_N_TILES = G_N / G_BASE_N;
constexpr uint32_t G_NUM_TILES = G_M_TILES * G_N_TILES;

// ============================================================================
// Host-side benchmark runner
// Uses existing gemm_compute_kernel with block_num=1 to test single core
// ============================================================================
static double benchmarkSingleComputeCore(int device_id)
{
    // Initialize
    int status = 0;
    aclrtStream stream = nullptr;
    
    status |= aclInit(nullptr);
    status |= aclrtSetDevice(device_id);
    status |= aclrtCreateStream(&stream);
    
    if (status != 0) {
        std::cerr << "[ERROR] Failed to initialize ACL\n";
        return -1.0;
    }
    
    // Allocate buffers (same as main program)
    size_t outputSize = static_cast<size_t>(G_M) * G_N * sizeof(float);
    size_t aSize = G_M * G_K * sizeof(uint16_t);
    size_t bSize = G_K * G_N * sizeof(uint16_t);
    
    void *shmem_output = nullptr;
    void *src0_dev = nullptr;
    void *src1_dev = nullptr;
    
    // Use regular device memory instead of shmem for simplicity
    aclrtMalloc(&shmem_output, outputSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&src0_dev, aSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&src1_dev, bSize, ACL_MEM_MALLOC_HUGE_FIRST);
    
    // Allocate minimal queue set (we don't actually use it, but launchGemmCompute requires it)
    // Just allocate enough for 1 block
    size_t queueSetSize = 1024;  // Minimal size
    void *queueSet_dev = nullptr;
    aclrtMalloc(&queueSet_dev, queueSetSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemset(queueSet_dev, queueSetSize, 0, queueSetSize);
    
    // Initialize with test data
    aclrtMemset(src0_dev, aSize, 0x42, aSize);
    aclrtMemset(src1_dev, bSize, 0x43, bSize);
    aclrtMemset(shmem_output, outputSize, 0, outputSize);
    
    // Warmup
    const int warmup_iters = 3;
    for (int i = 0; i < warmup_iters; ++i) {
        // Use block_num=1 to test single compute core
        // For single core test, use full K dimension
        launchGemmCompute(
            reinterpret_cast<uint8_t *>(shmem_output),
            reinterpret_cast<uint8_t *>(src0_dev),
            reinterpret_cast<uint8_t *>(src1_dev),
            reinterpret_cast<uint8_t *>(queueSet_dev),
            0,  // rank
            stream,
            1,  // block_num=1 for single core
            G_K);  // Use full K dimension for single core test
        aclrtSynchronizeStream(stream);
    }
    
    // Benchmark
    const int measure_iters = 100;
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < measure_iters; ++i) {
        // Reset output for each iteration
        aclrtMemset(shmem_output, outputSize, 0, outputSize);
        
        launchGemmCompute(
            reinterpret_cast<uint8_t *>(shmem_output),
            reinterpret_cast<uint8_t *>(src0_dev),
            reinterpret_cast<uint8_t *>(src1_dev),
            reinterpret_cast<uint8_t *>(queueSet_dev),
            0,  // rank
            stream,
            1,  // block_num=1 for single core
            G_K);  // Use full K dimension for single core test
        aclrtSynchronizeStream(stream);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double avg_time_us = duration.count() / (double)measure_iters;
    
    // Calculate GFLOPS
    // Total computation: M * K * N * 2 operations
    double total_ops = (double)G_M * G_K * G_N * 2.0;
    double total_gflop = total_ops / 1e9;
    double gflops = (total_gflop / (avg_time_us / 1e6));
    
    std::cout << "[BENCHMARK] Single Compute Core Performance:\n";
    std::cout << "  Matrix: M=" << G_M << ", K=" << G_K << ", N=" << G_N << "\n";
    std::cout << "  Total Tiles: " << G_NUM_TILES << "\n";
    std::cout << "  Avg Time: " << std::fixed << std::setprecision(2) << avg_time_us << " us\n";
    std::cout << "  Total Ops: " << std::fixed << std::setprecision(2) << total_gflop << " GFLOP\n";
    std::cout << "  Performance: " << std::fixed << std::setprecision(2) << gflops << " GFLOPS\n";
    
    // Cleanup
    aclrtFree(shmem_output);
    aclrtFree(src0_dev);
    aclrtFree(src1_dev);
    aclrtFree(queueSet_dev);
    aclrtDestroyStream(stream);
    
    return gflops;
}

int main(int argc, char* argv[])
{
    int device_id = 0;
    
    // Parse arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--device") == 0 && i + 1 < argc) {
            device_id = atoi(argv[i + 1]);
            i++;
        }
    }
    
    double gflops = benchmarkSingleComputeCore(device_id);
    
    if (gflops < 0) {
        return 1;
    }
    
    return 0;
}
