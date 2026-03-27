/**
 * Benchmark: Single Communication Core Performance
 * 
 * This program tests the TPUT transmission efficiency of a single communication core.
 * It measures how much data a single comm core can transfer per second.
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <vector>
#include <unistd.h>
#include <sys/wait.h>

// ACL runtime headers
#include "acl/acl.h"
#include "acl/acl_rt.h"

// Shmem headers (host-side only)
#if defined(CANN_SHMEM)
#include "shmem.h"
#include "host/shmem_host_def.h"
#else
#include "shmem_api.h"
#endif

// Forward declaration from benchmark_comm_kernel.cpp
extern void launchSingleCommCoreBenchmark(
    uint8_t *src_buffer,
    uint8_t *dst_buffer,
    int nranks,
    int num_tiles,
    void *stream);

// Shmem host-side wrapper functions
struct ShmemEnv {
    int rank;
    int size;
    const char *ipPort;
    size_t heapBytes;
};

static bool ShmemInitFromEnv(const ShmemEnv &env)
{
#if defined(CANN_SHMEM)
    aclshmemx_init_attr_t attributes;
    attributes.my_pe = env.rank;
    attributes.n_pes = env.size;
    attributes.local_mem_size = env.heapBytes;
    
    size_t ipLen = 0;
    if (env.ipPort != nullptr) {
        for (; ipLen < ACLSHMEM_MAX_IP_PORT_LEN - 1 && env.ipPort[ipLen] != '\0'; ++ipLen) {
            attributes.ip_port[ipLen] = env.ipPort[ipLen];
        }
    }
    attributes.ip_port[ipLen] = '\0';
    
    constexpr int attrVersion = (1 << 16) + sizeof(aclshmemx_init_attr_t);
    constexpr int DEFAULT_TIMEOUT = 120;
    attributes.option_attr = {attrVersion, ACLSHMEM_DATA_OP_MTE,
                              DEFAULT_TIMEOUT, DEFAULT_TIMEOUT, DEFAULT_TIMEOUT, -1};
    
    int ret = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes);
    return (ret == 0);
#else
    // ASCEND_SHMEM implementation would go here
    return false;
#endif
}

static void* ShmemMalloc(size_t bytes)
{
#if defined(CANN_SHMEM)
    return aclshmem_malloc(bytes);
#else
    return shmem_malloc(bytes);
#endif
}

static void ShmemFree(void *ptr)
{
#if defined(CANN_SHMEM)
    aclshmem_free(ptr);
#else
    shmem_free(ptr);
#endif
}

static void ShmemBarrierAll()
{
#if defined(CANN_SHMEM)
    aclshmem_barrier_all();
#else
    shmem_barrier_all();
#endif
}

// Use same tile size as main program
constexpr uint32_t G_BASE_M = 128;
constexpr uint32_t G_BASE_N = 256;
constexpr uint32_t TILE_SIZE = G_BASE_M * G_BASE_N * sizeof(float);  // bytes per tile

// ============================================================================
// Host-side benchmark runner
// ============================================================================
static double benchmarkSingleCommCore(int rank_id, int n_ranks, int device_id, int num_tiles)
{
    // Initialize
    int status = 0;
    aclrtStream stream = nullptr;
    
    status |= aclInit(nullptr);
    status |= aclrtSetDevice(device_id);
    status |= aclrtCreateStream(&stream);
    
    if (status != 0) {
        std::cerr << "[ERROR] Rank " << rank_id << ": Failed to initialize ACL\n";
        return -1.0;
    }
    
    // Initialize shmem
    ShmemEnv env;
    env.rank = rank_id;
    env.size = n_ranks;
    env.ipPort = "tcp://127.0.0.1:8790";
    env.heapBytes = 512 * 1024 * 1024;  // 512MB
    
    if (!ShmemInitFromEnv(env)) {
        std::cerr << "[ERROR] Rank " << rank_id << ": ShmemInitFromEnv failed!\n";
        return -1.0;
    }
    
    // Allocate buffers
    size_t buffer_size = (size_t)num_tiles * G_BASE_M * G_BASE_N * sizeof(float);
    void *src_buffer = ShmemMalloc(buffer_size);
    void *dst_buffer = ShmemMalloc(buffer_size);
    
    if (src_buffer == nullptr || dst_buffer == nullptr) {
        std::cerr << "[ERROR] Rank " << rank_id << ": ShmemMalloc failed!\n";
        return -1.0;
    }
    
    // Initialize source buffer with test data
    aclrtMemset(src_buffer, buffer_size, 0x42, buffer_size);
    aclrtMemset(dst_buffer, buffer_size, 0, buffer_size);
    
    // Synchronize all ranks
    ShmemBarrierAll();
    
    // Warmup
    const int warmup_iters = 3;
    for (int i = 0; i < warmup_iters; ++i) {
        launchSingleCommCoreBenchmark(
            reinterpret_cast<uint8_t *>(src_buffer),
            reinterpret_cast<uint8_t *>(dst_buffer),
            n_ranks,
            num_tiles,
            stream);
        aclrtSynchronizeStream(stream);
        ShmemBarrierAll();
    }
    
    // Benchmark
    const int measure_iters = 100;
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < measure_iters; ++i) {
        launchSingleCommCoreBenchmark(
            reinterpret_cast<uint8_t *>(src_buffer),
            reinterpret_cast<uint8_t *>(dst_buffer),
            n_ranks,
            num_tiles,
            stream);
        aclrtSynchronizeStream(stream);
        ShmemBarrierAll();
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double avg_time_us = duration.count() / (double)measure_iters;
    
    // Calculate bandwidth
    // Each tile is transferred to all n_ranks
    double total_bytes = (double)num_tiles * TILE_SIZE * n_ranks;
    double bandwidth_gb_s = (total_bytes / (avg_time_us / 1e6)) / (1024.0 * 1024.0 * 1024.0);
    
    if (rank_id == 0) {
        std::cout << "[BENCHMARK] Single Comm Core Performance:\n";
        std::cout << "  Tiles: " << num_tiles << "\n";
        std::cout << "  Ranks: " << n_ranks << "\n";
        std::cout << "  Avg Time: " << std::fixed << std::setprecision(2) << avg_time_us << " us\n";
        std::cout << "  Total Data: " << std::fixed << std::setprecision(2) 
                  << (total_bytes / (1024.0 * 1024.0)) << " MB\n";
        std::cout << "  Bandwidth: " << std::fixed << std::setprecision(2) 
                  << bandwidth_gb_s << " GB/s\n";
    }
    
    // Cleanup
    ShmemFree(src_buffer);
    ShmemFree(dst_buffer);
    aclrtDestroyStream(stream);
    
    return bandwidth_gb_s;
}

int main(int argc, char* argv[])
{
    int n_ranks = 8;
    int first_device = 0;
    int num_tiles = 100;  // Number of tiles to transfer
    
    // Parse arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--nranks") == 0 && i + 1 < argc) {
            n_ranks = atoi(argv[i + 1]);
            i++;
        } else if (strcmp(argv[i], "--first-device") == 0 && i + 1 < argc) {
            first_device = atoi(argv[i + 1]);
            i++;
        } else if (strcmp(argv[i], "--num-tiles") == 0 && i + 1 < argc) {
            num_tiles = atoi(argv[i + 1]);
            i++;
        }
    }
    
    // Fork processes for multi-rank
    std::vector<pid_t> pids;
    for (int rank = 0; rank < n_ranks; ++rank) {
        pid_t pid = fork();
        if (pid == 0) {
            // Child process
            double bandwidth = benchmarkSingleCommCore(rank, n_ranks, first_device + rank, num_tiles);
            exit(0);
        } else if (pid > 0) {
            pids.push_back(pid);
        } else {
            std::cerr << "[ERROR] Failed to fork process for rank " << rank << std::endl;
            return 1;
        }
    }
    
    // Wait for all processes
    for (pid_t pid : pids) {
        int status;
        waitpid(pid, &status, 0);
    }
    
    return 0;
}
