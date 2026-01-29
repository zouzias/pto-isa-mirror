#include "tallreduce_mc_perf_test.h"

#include "pto/comm/pto_comm_inst.hpp"
#include "pto/common/pto_tile.hpp"
#include "../common.hpp"
#include <pto/pto-inst.hpp>
#include "pto/npu/a2a3/custom/TSyncCVID.hpp"


// ============================================================================
// Device-side Kernel: TAllReduce with cycle counting
// Different cores handle different reduce partitions based on block_id
// ============================================================================
template <typename T, int kTRows_, int kTCols_, int vRows, int vCols>
__global__ AICORE void TAllReducePerfKernelImpl(
    __gm__ T *input, __gm__ T *output, __gm__ T *shmem,
    __gm__ int64_t *cycle_results,  // Store cycle counts
    int iteration,
    int total_blocks,
    uint64_t fftsConfig)  // Total number of blocks launched
{
    util_set_ffts_config(fftsConfig);
    ShmemDeviceBarrierAll();

    // Get block id to determine which partition this core handles
    const int block_idx = get_block_idx();
    // if(block_idx == 0 && shmem_my_pe() == 0)
    //     cce::printf("block_idx: %d, kTRows_: %d, kTCols_: %d, vRows: %d, vCols: %d\n", block_idx, kTRows_, kTCols_, vRows, vCols);
    
    // // Partition data by rows: each block handles a subset of rows
    // // If total_blocks is not provided (0), fall back to single block per rank
    const int n_blocks = (total_blocks > 0) ? total_blocks : 1;
    // const int rows_per_block = (vRows + n_blocks - 1) / n_blocks;  // Ceiling division
    // const int my_start_row = block_idx * rows_per_block;
    // const int my_end_row = (block_idx + 1) * rows_per_block;
    // const int my_rows = (my_end_row > vRows) ? (vRows - my_start_row) : rows_per_block;
    

    const int my_rows = vRows;

    const int my_start_row = block_idx * kTRows_;

    using ShapeDyn  = Shape<1, 1, 1, vRows, vCols>;
    using StrideDyn = Stride<1, 1, 1, kTCols_, 1>;
    using Global = pto::GlobalTensor<T, ShapeDyn, StrideDyn, pto::Layout::ND>;
    using TileData = Tile<TileType::Vec, T, kTRows_, kTCols_, BLayout::RowMajor, -1, -1>;


    const int total_rows = vRows * n_blocks;
    const bool has_work = (my_rows > 0) && (my_start_row < total_rows);
    int64_t start_cycle = 0;
    int64_t end_cycle = 0;

    int my_rank = shmem_my_pe();
    int nranks = shmem_n_pes();

    if (has_work) {
        TileData src0Tile(my_rows, vCols);
        TileData src1Tile(my_rows, vCols);
        TileData dstTile(my_rows, vCols);
        
        constexpr size_t tileBytes = kTRows_ * kTCols_ * sizeof(T);
        constexpr size_t alignedTileBytes = ((tileBytes + 31) / 32) * 32;
        
        TASSIGN(src0Tile, 0);
        TASSIGN(src1Tile, 0 + alignedTileBytes);
        TASSIGN(dstTile,  0 + alignedTileBytes * 2);

        // Create shape and stride for this block's partition
        // cce::printf("my_rows: %d, vCols: %d\n", my_rows, vCols);
        ShapeDyn shape(1, 1, 1, my_rows, vCols);
        StrideDyn stride(1, 1, 1, kTCols_, 1);
        
        // Calculate offset for this block's partition
        // Each row has kTCols_ elements, so offset = my_start_row * kTCols_
        const int64_t row_offset = static_cast<int64_t>(my_start_row) * static_cast<int64_t>(kTCols_);
        
        Global srcGlobal(input + row_offset, shape, stride);
        Global dstGlobal(output + row_offset, shape, stride);

        // For shmem, we need to account for the row offset as well
        // shmem is organized as: [rank0_data, rank1_data, ...]
        // Each rank's data has vRows * vCols elements
        // We need to offset by row_offset within each rank's region
        const int64_t shmem_row_offset = static_cast<int64_t>(my_start_row) * static_cast<int64_t>(kTCols_);
        Global srcShmemGlobal((__gm__ T*)shmem + shmem_row_offset, shape, stride);

        // Load the src out of the shmem region to the shmem region
        TLOAD(src0Tile, srcGlobal);
        set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        TSTORE(srcShmemGlobal, src0Tile);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);

        Global *tensorPtrs[16];
        Global tensors[16];
        
        // Create ParallelGroup for this block's partition
        // Each rank's tensor points to the corresponding partition in shmem
        // shmem_ptr(shmem, i) returns pointer to rank i's data region
        // We add shmem_row_offset to access the specific row range for this block
        for (int i = 0; i < nranks; ++i) {
            int rank_i = (i + (n_blocks % nranks)) % nranks;
            // First get the base pointer for rank i, then add the row offset
            __gm__ T* rank_i_base = (__gm__ T*)shmem_ptr(shmem, rank_i);
            __gm__ T* rank_i_partition = rank_i_base + shmem_row_offset;
            Global srcShmemPtr(rank_i_partition, shape, stride);
            tensors[i] = srcShmemPtr;
            tensors[i].SetRank(rank_i);
            tensorPtrs[i] = &tensors[i];
        }
        
        pto::comm::ParallelGroup<Global> pg(tensorPtrs, nranks, my_rank);
        
        // ====================================================================
        // CRITICAL: Multi-stage synchronization for multi-block + multi-PE
        // Stage 1: Ensure all local DMA operations are complete (PipeBarrier)
        // Stage 2: Sync all cores on this device (barrier_core_soft)
        // Stage 3: Ensure all shmem operations are globally visible (quiet)
        // Stage 4: Sync across all PEs (barrier_all)
        // ====================================================================
        AscendC::PipeBarrier<PIPE_ALL>();
        aclshmemi_barrier_core_soft();   // Sync all cores on this device
        AscendC::PipeBarrier<PIPE_ALL>();

        start_cycle = AscendC::GetSystemCycle();
        pto::comm::TALLREDUCE(pg, dstGlobal, src0Tile, src1Tile, dstTile);
        AscendC::PipeBarrier<PIPE_ALL>();
        aclshmemi_barrier_core_soft();
        end_cycle = AscendC::GetSystemCycle();

        ShmemDeviceBarrierAll();
    } else {
        // Still participate in barriers to avoid deadlock with other blocks.
        // Even blocks without work must participate in ALL sync points!
        AscendC::PipeBarrier<PIPE_ALL>();
        aclshmemi_barrier_core_soft();

        // AscendC::PipeBarrier<PIPE_ALL>();

        // start_cycle = AscendC::GetSystemCycle();
        // AscendC::PipeBarrier<PIPE_ALL>();
        // aclshmemi_barrier_core_soft();
        // end_cycle = AscendC::GetSystemCycle();
        
        ShmemDeviceBarrierAll();
    }
    
    // ========================================================================
    // Block synchronization point 2: Wait for all blocks to finish reduce
    // Using FFTS cross-core sync mechanism
    // ========================================================================
    // if (n_blocks > 1) {
    //     constexpr uint16_t SYNC_FLAG_BASE_REDUCE = 4;  // Base flag for reduce sync
        
    //     // Ensure all previous operations are complete
    //     AscendC::PipeBarrier<PIPE_ALL>();
        
    //     // Each block signals completion using its own flag ID
    //     const uint16_t my_flag_id = SYNC_FLAG_BASE_REDUCE + static_cast<uint16_t>(block_idx);
    //     ffts_cross_core_sync(PIPE_FIX, pto::_getFFTSMsg(pto::CV_CORE_SYNC, my_flag_id));
        
    //     // Block 0 waits for all other blocks to complete
    //     if (block_idx == 0) {
    //         // Wait for all blocks (1 to n_blocks-1) to signal completion
    //         for (int b = 1; b < n_blocks; ++b) {
    //             const uint16_t wait_flag_id = SYNC_FLAG_BASE_REDUCE + static_cast<uint16_t>(b);
    //             wait_flag_dev(wait_flag_id);
    //         }
    //         // Signal that all blocks are ready (use flag 11 as completion signal for reduce)
    //         ffts_cross_core_sync(PIPE_FIX, pto::_getFFTSMsg(pto::CV_CORE_SYNC, static_cast<uint16_t>(11)));
    //     } else {
    //         // Other blocks wait for block 0's final signal
    //         wait_flag_dev(11);
    //     }
        
    //     AscendC::PipeBarrier<PIPE_ALL>();
    // }
    
    // Only record cycle count from block 0 of rank 0 to avoid duplicate entries
    if (my_rank == 0 && block_idx == 0 && cycle_results != nullptr) {
        cycle_results[iteration] = end_cycle - start_cycle;
    }

    ShmemDeviceBarrierAll();
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
    const char *ip = "tcp://127.0.0.1:8776";
    env.rank = rank_id;
    env.size = n_ranks;
    env.ipPort = ip;
    
    if (!ShmemInitFromEnv(env)) {
        return false;
    }

    const size_t total_rows = static_cast<size_t>(kTRows_) * static_cast<size_t>(config.block_num);
    size_t count = total_rows * static_cast<size_t>(kTCols_);
    size_t fileSize = count * sizeof(T);
    T *srcHost, *resHost;
    aclrtMallocHost((void **)(&srcHost), fileSize);
    aclrtMallocHost((void **)(&resHost), fileSize);
    T seed = 1280;
    for (int i = 0; i < count; ++i) {
        srcHost[i] = static_cast<T>((rank_id + 1) * 100 + i + seed);
        resHost[i] = static_cast<T>(-1);
    }
    T *srcDevice, *resDevice;
    aclrtMalloc((void **)(&srcDevice), fileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)(&resDevice), fileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemcpy(srcDevice, fileSize, srcHost, fileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(resDevice, fileSize, resHost, fileSize, ACL_MEMCPY_HOST_TO_DEVICE);

    uint64_t fftsConfig = util_get_ffts_config();
    void *shmemBufferDevice = ShmemMalloc(4 * fileSize);

    if(shmemBufferDevice == nullptr){
        std::cerr << "[ERROR] ShmemMalloc failed!" << std::endl;
        ShmemFinalize();
        return false;
    }

    int64_t *perfHost;
    aclrtMallocHost(reinterpret_cast<void**>(&perfHost), config.measure_iters * sizeof(int64_t));
    
    int64_t *perfDevice;
    aclrtMalloc(reinterpret_cast<void**>(&perfDevice), config.measure_iters * sizeof(int64_t), ACL_MEM_MALLOC_HUGE_FIRST);

    if (rank_id == 0 && config.verbose) {
        std::cout << "\n[PERF] Starting warmup (" << config.warmup_iters << " iterations)..." << std::endl;
    }

    for(int i = 0; i < config.warmup_iters; ++i) {
        TAllReducePerfKernelImpl<T, kTRows_, kTCols_, vRows, vCols><<<config.block_num, nullptr, stream>>>(
            srcDevice, resDevice, (T*) shmemBufferDevice, nullptr, 0, config.block_num, fftsConfig);
        aclrtSynchronizeStream(stream);
    }

    // Barrier to ensure all ranks finished warmup
    ShmemBarrierAll();

    if (rank_id == 0 && config.verbose) {
        std::cout << "[PERF] Starting measurement (" << config.measure_iters << " iterations)..." << std::endl;
    }
    
    aclrtMemcpy(resDevice, fileSize, resHost, fileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemset(perfDevice, config.measure_iters * sizeof(int64_t), 0, config.measure_iters * sizeof(int64_t));
    
    aclrtSynchronizeStream(stream);

    ShmemBarrierAll();

    auto wall_start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < config.measure_iters; ++i) {
        TAllReducePerfKernelImpl<T, kTRows_, kTCols_, vRows, vCols><<<config.block_num, nullptr, stream>>>(
            srcDevice, resDevice, (T*) shmemBufferDevice, perfDevice, i, config.block_num, fftsConfig);
        aclrtSynchronizeStream(stream);
    }
    auto wall_end = std::chrono::high_resolution_clock::now();
    double wall_time_ms = std::chrono::duration<double, std::milli>(wall_end - wall_start).count();

    ShmemBarrierAll();

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
        bool flag = true;
        for(int i = 0; i < count; ++i) {
            if (config.verbose) {
                if(i < 5) std::cout << "resHost[" << i << "] = " << resHost[i] << std::endl;
            }

            T expected = n_ranks * (n_ranks + 1) / 2 * 100 + i * n_ranks + seed * n_ranks;
            T actual = resHost[i];
            if (actual != expected) {
                std::cerr << "[ERROR] TALLREDUCE failed at index: " << i << "\n";
                flag = false;
                status = 1;
                break;
            }
        }
        if(flag){
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
            std::cout << "  TALLREDUCE w/ Multi-core Performance Test Results" << std::endl;
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
    }

    ShmemBarrierAll();

    // Cleanup
    aclrtFreeHost(srcHost);
    aclrtFreeHost(resHost);
    aclrtFree(srcDevice);
    aclrtFree(resDevice);
    aclrtFreeHost(perfHost);
    aclrtFree(perfDevice);
    ShmemFree(shmemBufferDevice);

    ShmemFinalize();

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
template bool RunAllReducePerf<float, 64 / AR_BLOCK_NUM, 64, 64 / AR_BLOCK_NUM, 64>(int, int, int, int, const PerfTestConfig&);
template bool RunAllReducePerf<float, 16 / AR_BLOCK_NUM, 256, 16 / AR_BLOCK_NUM, 256>(int, int, int, int, const PerfTestConfig&);

// Large size tests (max ~64KB per tile to fit 3 tiles in UB)
template bool RunAllReducePerf<int32_t, 128 / AR_BLOCK_NUM, 128, 128 / AR_BLOCK_NUM, 128>(int, int, int, int, const PerfTestConfig&);
template bool RunAllReducePerf<int32_t, 64 / AR_BLOCK_NUM, 256, 64 / AR_BLOCK_NUM, 256>(int, int, int, int, const PerfTestConfig&);


