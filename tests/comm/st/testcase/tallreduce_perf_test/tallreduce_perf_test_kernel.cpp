#include "tallreduce_perf_test.h"

#include "pto/comm/pto_comm_inst.hpp"
#include "pto/common/pto_tile.hpp"
#include "../common.hpp"
#include <pto/pto-inst.hpp>


// ============================================================================
// Device-side Kernel: TAllReduce with cycle counting
// ============================================================================
template <typename T, int kTRows_, int kTCols_, int vRows, int vCols>
__global__ AICORE void TAllReducePerfKernelImpl(
    __gm__ T *input, __gm__ T *output, __gm__ T *shmem,
    __gm__ int64_t *cycle_results,  // Store cycle counts
    int iteration)
{
    using ShapeDyn  = Shape<1, 1, 1, vRows, vCols>;
    using StrideDyn = Stride<1, 1, 1, kTCols_, 1>;
    using Global = pto::GlobalTensor<T, ShapeDyn, StrideDyn, pto::Layout::ND>;
    using TileData = Tile<TileType::Vec, T, kTRows_, kTCols_, BLayout::RowMajor, -1, -1>;


    TileData src0Tile(vRows, vCols);
    TileData src1Tile(vRows, vCols);
    TileData dstTile(vRows, vCols);
    
    constexpr size_t tileBytes = kTRows_ * kTCols_ * sizeof(T);
    constexpr size_t alignedTileBytes = ((tileBytes + 31) / 32) * 32;
    
    TASSIGN(src0Tile, 0);
    TASSIGN(src1Tile, 0 + alignedTileBytes);
    TASSIGN(dstTile,  0 + alignedTileBytes * 2);


    int my_rank = shmem_my_pe();
    int nranks = shmem_n_pes();

    ShapeDyn shape(1, 1, 1, vRows, vCols);
    StrideDyn stride(1, 1, 1, kTCols_, 1);
    
    Global srcGlobal(input, shape, stride);
    Global dstGlobal(output, shape, stride);

    Global srcShmemGlobal(shmem, shape, stride);

    // Load the src out of the shmem region to the shmem region
    TLOAD(src0Tile, srcGlobal);
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    TSTORE(srcShmemGlobal, src0Tile);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);


    Global *tensorPtrs[16];
    Global tensors[16];

    for (int i = 0; i < nranks; ++i) {
        Global srcShmemPtr((__gm__ T*)shmem_ptr(shmem, i), shape, stride);
        tensors[i] = srcShmemPtr;
        tensors[i].SetRank(i);
        tensorPtrs[i] = &tensors[i];
    }
    
    pto::comm::ParallelGroup<Global> pg(tensorPtrs, nranks, my_rank);
    

    int64_t start_cycle = AscendC::GetSystemCycle();

    
    pto::comm::TALLREDUCE(pg, dstGlobal, src0Tile, src1Tile, dstTile);
    pto::comm::TQUIET();
    
    int64_t end_cycle = AscendC::GetSystemCycle();
    
    if (my_rank == 0 && cycle_results != nullptr) {
        cycle_results[iteration] = end_cycle - start_cycle;
    }

}

// ============================================================================
// Host-side Performance Test Runner
// ============================================================================
template<typename T, int kTRows_, int kTCols_, int vRows, int vCols>
bool RunAllReducePerfKernel(
    int rank_id, 
    int n_ranks, 
    int n_devices, 
    int first_device_id,
    const PerfTestConfig &config)
{
    int32_t ret = shmem_set_conf_store_tls(false, nullptr, 0);
    if (ret != 0) {
        std::cerr << "[ERROR] Failed to init shmem tls\n";
        return false;
    }

    if (n_devices <= 0 || n_ranks <= 0) {
        std::cerr << "[ERROR] n_devices and n_ranks must be > 0\n";
        return false;
    }
    
    const int32_t device_id = rank_id % n_devices + first_device_id;
    int status = 0;
    aclrtStream stream = nullptr;

    status |= aclInit(nullptr);
    status |= aclrtSetDevice(device_id);
    status |= aclrtCreateStream(&stream);

    ShmemEnv env;
    const char *ip = "tcp://127.0.0.1:8766";
    env.rank = rank_id;
    env.size = n_ranks;
    env.ipPort = ip;
    
    if (!ShmemInitFromEnv(env)) {
        return false;
    }

    size_t count = kTRows_ * kTCols_;
    size_t fileSize = kTRows_ * kTCols_ * sizeof(T);
    T *srcHost, *resHost;
    aclrtMallocHost((void **)(&srcHost), fileSize);
    aclrtMallocHost((void **)(&resHost), fileSize);
    for (int i = 0; i < count; ++i) {
        srcHost[i] = static_cast<T>((rank_id + 1) * 100 + i);
        resHost[i] = static_cast<T>(-1);
    }
    T *srcDevice, *resDevice;
    aclrtMalloc((void **)(&srcDevice), fileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)(&resDevice), fileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemcpy(srcDevice, fileSize, srcHost, fileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(resDevice, fileSize, resHost, fileSize, ACL_MEMCPY_HOST_TO_DEVICE);

    void *shmemBufferDevice = pto::comm::ContextManager::SymmetricAlloc(4 * fileSize);

    int64_t *perfHost;
    aclrtMallocHost(reinterpret_cast<void**>(&perfHost), config.measure_iters * sizeof(int64_t));
    
    int64_t *perfDevice;
    aclrtMalloc(reinterpret_cast<void**>(&perfDevice), config.measure_iters * sizeof(int64_t), ACL_MEM_MALLOC_HUGE_FIRST);

    if (rank_id == 0 && config.verbose) {
        std::cout << "\n[PERF] Starting warmup (" << config.warmup_iters << " iterations)..." << std::endl;
    }

    for(int i = 0; i < config.warmup_iters; ++i) {
        TAllReducePerfKernelImpl<T, kTRows_, kTCols_, vRows, vCols><<<config.block_num, nullptr, stream>>>(
            srcDevice, resDevice, (T*) shmemBufferDevice, nullptr, 0);
        aclrtSynchronizeStream(stream);
    }

    // Barrier to ensure all ranks finished warmup
    #if defined(CANN_SHMEM)
        aclshmem_barrier_all();
    #elif defined(ASCEND_SHMEM)
        shmem_barrier_all();
    #endif

    if (rank_id == 0 && config.verbose) {
        std::cout << "[PERF] Starting measurement (" << config.measure_iters << " iterations)..." << std::endl;
    }
    
    // Clear cycle results
    aclrtMemset(perfDevice, config.measure_iters * sizeof(int64_t), 0, config.measure_iters * sizeof(int64_t));
    
    auto wall_start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < config.measure_iters; ++i) {
        TAllReducePerfKernelImpl<T, kTRows_, kTCols_, vRows, vCols><<<config.block_num, nullptr, stream>>>(
            srcDevice, resDevice, (T*) shmemBufferDevice, perfDevice, i);
        aclrtSynchronizeStream(stream);
    }
    auto wall_end = std::chrono::high_resolution_clock::now();
    double wall_time_ms = std::chrono::duration<double, std::milli>(wall_end - wall_start).count();

    // Copy cycle results back
    aclrtMemcpy(perfHost, config.measure_iters * sizeof(int64_t), 
                perfDevice, config.measure_iters * sizeof(int64_t),
                ACL_MEMCPY_DEVICE_TO_HOST);

    // Copy result data back for verification
    aclrtMemcpy(resHost, fileSize, resDevice, fileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    // ========================================================================
    // Results Analysis (only rank 0)
    // ========================================================================
    if (rank_id == 0) {
        // check the correctness
        for(int i = 0; i < count; ++i) {
            if (config.verbose) {
                if(i < 5) std::cout << "resHost[" << i << "] = " << resHost[i] << std::endl;
            }

            T expected = n_ranks * (n_ranks + 1) / 2 * 100 + i * n_ranks;
            T actual = resHost[i];
            if (actual != expected) {
                std::cerr << "[ERROR] TALLREDUCE failed\n";
                return false;
            }
        }

        // Convert cycles to microseconds
        // Based on rdma_perftest: cycles / 50.0 = us
        constexpr double CYCLES_PER_US = 50.0;
        std::vector<double> latencies_us;
        
        for (int i = 0; i < config.measure_iters; ++i) {
            double cycles = static_cast<double>(perfHost[i]);
            double us = cycles / CYCLES_PER_US;
            latencies_us.push_back(us);
            
            if (config.verbose) {
                std::cout << "  Iter " << std::setw(3) << i 
                          << ": " << std::fixed << std::setprecision(2) << us << " us"
                          << " (" << perfHost[i] << " cycles)" << std::endl;
            }
        }
        
        PerfStats stats = CalculateStats(latencies_us, fileSize);
        
        // Print summary
        std::cout << "\n================================================================" << std::endl;
        std::cout << "  TALLREDUCE Performance Test Results" << std::endl;
        std::cout << "================================================================" << std::endl;
        std::cout << "  Configuration:" << std::endl;
        std::cout << "    - Data type:      " << typeid(T).name() << std::endl;
        std::cout << "    - Element count:  " << count << std::endl;
        std::cout << "    - Data size:      " << fileSize << " bytes (" 
                  << (fileSize / 1024.0) << " KB)" << std::endl;
        std::cout << "    - Rank count:     " << n_ranks << std::endl;
        std::cout << "    - Warmup iters:   " << config.warmup_iters << std::endl;
        std::cout << "    - Measure iters:  " << config.measure_iters << std::endl;
        std::cout << std::endl;
        std::cout << "  Latency Statistics:" << std::endl;
        std::cout << "    - Min:            " << std::fixed << std::setprecision(2) 
                  << stats.min_us << " us" << std::endl;
        std::cout << "    - Max:            " << stats.max_us << " us" << std::endl;
        std::cout << "    - Average:        " << stats.avg_us << " us" << std::endl;
        std::cout << "    - Median:         " << stats.median_us << " us" << std::endl;
        std::cout << "    - Std Dev:        " << stats.std_dev_us << " us" << std::endl;
        std::cout << std::endl;
        std::cout << "  Throughput:" << std::endl;
        std::cout << "    - Bandwidth:      " << std::setprecision(3) 
                  << stats.bandwidth_gbps << " GB/s" << std::endl;
        std::cout << "    - Message Rate:   " << stats.msg_rate_mops << " Mops/s" << std::endl;
        std::cout << std::endl;
        std::cout << "  Wall Clock Time:    " << std::setprecision(2) 
                  << wall_time_ms << " ms (total)" << std::endl;
        std::cout << "================================================================\n" << std::endl;
    }

    // Cleanup
    aclrtFreeHost(srcHost);
    aclrtFreeHost(resHost);
    aclrtFree(srcDevice);
    aclrtFree(resDevice);
    aclrtFreeHost(perfHost);
    aclrtFree(perfDevice);
    pto::comm::ContextManager::SymmetricFree(shmemBufferDevice);

    status |= aclrtDestroyStream(stream);
    status |= aclrtResetDevice(device_id);
    status |= aclFinalize();

    return (status == 0);
}

// ============================================================================
// Multi-process Launcher
// ============================================================================
template <typename T, int kTRows_, int kTCols_, int vRows, int vCols>
bool RunAllReducePerf(
    int n_ranks, 
    int n_devices, 
    int first_rank_id, 
    int first_device_id,
    const PerfTestConfig &config)
{
    std::vector<pid_t> pids;
    
    for (int r = 0; r < n_ranks; ++r) {
        pid_t pid = fork();
        if (pid == 0) { // child
            const bool ok = RunAllReducePerfKernel<T, kTRows_, kTCols_, vRows, vCols>(
                first_rank_id + r, n_ranks, n_devices, first_device_id, config);
            _exit(ok ? 0 : 1);
        } else if (pid > 0) {
            pids.push_back(pid);
        } else {
            return false;
        }
    }
    
    bool success = true;
    for (pid_t p : pids) {
        int status = 0;
        waitpid(p, &status, 0);
        if (!(WIFEXITED(status) && WEXITSTATUS(status) == 0)) {
            success = false;
        }
    }
    return success;
}

// ============================================================================
// Explicit Instantiations for Common Configurations
// ============================================================================

// Small size tests
template bool RunAllReducePerf<float, 64, 64, 64, 64>(int, int, int, int, const PerfTestConfig&);
template bool RunAllReducePerf<float, 16, 256, 16, 256>(int, int, int, int, const PerfTestConfig&);

// Large size tests (max ~64KB per tile to fit 3 tiles in UB)
template bool RunAllReducePerf<int32_t, 128, 128, 128, 128>(int, int, int, int, const PerfTestConfig&);
template bool RunAllReducePerf<int32_t, 64, 256, 64, 256>(int, int, int, int, const PerfTestConfig&);


