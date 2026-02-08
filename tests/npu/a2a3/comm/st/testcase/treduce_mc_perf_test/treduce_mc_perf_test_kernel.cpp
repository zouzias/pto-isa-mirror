#include "treduce_mc_perf_test.h"

#include "pto/comm/pto_comm_inst.hpp"
#include "pto/common/pto_tile.hpp"
#include "../common.hpp"
#include <pto/pto-inst.hpp>
#include "pto/npu/a2a3/custom/TSyncCVID.hpp"


// ============================================================================
// Device-side Kernel: TReduce with Tiling
// 
// Each core processes a portion of a LARGE tensor by tiling:
// - Total tensor: total_rows x total_cols
// - Each core handles: (total_rows / n_blocks) rows
// - Tile size: kTRows_ x kTCols_ (fits in UB)
// - Loop iterations per core: (rows_per_block / kTRows_)
// ============================================================================
template <typename T, int kTRows_, int kTCols_>
__global__ AICORE void TReduceTilingKernelImpl(
    __gm__ T *input, __gm__ T *output, __gm__ T *shmem,
    __gm__ int64_t *cycle_results,
    int total_rows,          // Total rows in the tensor
    int total_cols,          // Total cols in the tensor (= kTCols_ for simplicity)
    int total_blocks,
    uint64_t fftsConfig)
{
    util_set_ffts_config(fftsConfig);
    ShmemDeviceBarrierAll();

    const int block_idx = get_block_idx();
    const int n_blocks = (total_blocks > 0) ? total_blocks : 1;
    
    int my_rank = shmem_my_pe();
    int nranks = shmem_n_pes();

    // Calculate this block's workload
    const int rows_per_block = (total_rows + n_blocks - 1) / n_blocks;
    const int my_start_row = block_idx * rows_per_block;
    const int my_end_row = (my_start_row + rows_per_block > total_rows) ? total_rows : (my_start_row + rows_per_block);
    const int my_total_rows = my_end_row - my_start_row;
    
    // Number of tile iterations this block needs
    const int tiles_per_block = (my_total_rows + kTRows_ - 1) / kTRows_;
    
    const bool has_work = (my_total_rows > 0);
    int64_t start_cycle = 0;
    int64_t end_cycle = 0;
    int actual_tiles = 0;

    // Tile types
    using TileShape = pto::Shape<1, 1, 1, kTRows_, kTCols_>;
    using TileStride = pto::Stride<1, 1, 1, kTCols_, 1>;
    using Global = pto::GlobalTensor<T, TileShape, TileStride, pto::Layout::ND>;
    using TileData = pto::Tile<pto::TileType::Vec, T, kTRows_, kTCols_, pto::BLayout::RowMajor, -1, -1>;

    if (has_work) {
        // Allocate UB tiles
        TileData accTile(kTRows_, kTCols_);
        TileData recvTile(kTRows_, kTCols_);
        
        constexpr size_t tileBytes = kTRows_ * kTCols_ * sizeof(T);
        constexpr size_t alignedTileBytes = ((tileBytes + 31) / 32) * 32;
        
        TASSIGN(accTile, 0);
        TASSIGN(recvTile, alignedTileBytes);
        
        TileShape tileShape(1, 1, 1, kTRows_, kTCols_);
        TileStride tileStride(1, 1, 1, kTCols_, 1);

        // ====================================================================
        // Phase 1: Initialize all tiles in shmem (NOT timed)
        // ====================================================================
        for (int tile_idx = 0; tile_idx < tiles_per_block; ++tile_idx) {
            const int tile_start_row = my_start_row + tile_idx * kTRows_;
            if (tile_start_row >= total_rows) break;
            
            const int64_t row_offset = static_cast<int64_t>(tile_start_row) * static_cast<int64_t>(total_cols);
            
            Global srcGlobal(input + row_offset, tileShape, tileStride);
            Global shmemGlobal((__gm__ T*)shmem + row_offset, tileShape, tileStride);
            
            // Load input to shmem (so other ranks can access it)
            TLOAD(accTile, srcGlobal);
            set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            TSTORE(shmemGlobal, accTile);
            set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
            wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        }
        
        // Sync all cores and ranks after initialization
        AscendC::PipeBarrier<PIPE_ALL>();
        aclshmemi_barrier_core_soft();
        ShmemDeviceBarrierAll();
        AscendC::PipeBarrier<PIPE_ALL>();

        // ====================================================================
        // Phase 2: TREDUCE loop (TIMED - only communication, no data init)
        // ====================================================================
        start_cycle = AscendC::GetSystemCycle();
        
        for (int tile_idx = 0; tile_idx < tiles_per_block; ++tile_idx) {
            const int tile_start_row = my_start_row + tile_idx * kTRows_;
            
            // Skip if this tile is out of bounds
            if (tile_start_row >= total_rows) break;
            actual_tiles++;
            
            const int64_t row_offset = static_cast<int64_t>(tile_start_row) * static_cast<int64_t>(total_cols);
            
            // Create GlobalTensors for this tile
            Global dstGlobal(output + row_offset, tileShape, tileStride);
            Global shmemGlobal((__gm__ T*)shmem + row_offset, tileShape, tileStride);
            
            // Reload data from shmem to accTile for reduction
            TLOAD(accTile, shmemGlobal);
            set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            
            // Build ParallelGroup for this tile
            Global tensors[16];
            for (int i = 0; i < nranks; ++i) {
                __gm__ T* rank_i_base = (__gm__ T*)shmem_ptr(shmem, i);
                tensors[i] = Global(rank_i_base + row_offset, tileShape, tileStride);
            }
            pto::comm::ParallelGroup<Global> pg(tensors, nranks, my_rank);
            
            // Perform TREDUCE for this tile (this is what we're measuring)
            pto::comm::TREDUCE(pg, dstGlobal, accTile, recvTile, pto::comm::ReduceOp::Sum);
            
            AscendC::PipeBarrier<PIPE_ALL>();
        }
        
        end_cycle = AscendC::GetSystemCycle();
        ShmemDeviceBarrierAll();
    } else {
        // Participate in all barriers (matching the main branch structure)
        // Phase 1 init barrier
        AscendC::PipeBarrier<PIPE_ALL>();
        aclshmemi_barrier_core_soft();
        ShmemDeviceBarrierAll();
        AscendC::PipeBarrier<PIPE_ALL>();
        
        // Phase 2 timed loop barriers
        for (int tile_idx = 0; tile_idx < tiles_per_block; ++tile_idx) {
            AscendC::PipeBarrier<PIPE_ALL>();
        }
        
        ShmemDeviceBarrierAll();
    }
    
    // Record results (only from block 0 of rank 0)
    if (my_rank == 0 && block_idx == 0 && cycle_results != nullptr) {
        cycle_results[0] = end_cycle - start_cycle;
        cycle_results[1] = actual_tiles;           // Tiles processed by this block
        cycle_results[2] = tiles_per_block;        // Expected tiles per block
        cycle_results[3] = my_total_rows;          // Rows assigned to this block
    }

    ShmemDeviceBarrierAll();
}

// ============================================================================
// Host-side Performance Test Runner with Tiling
// ============================================================================
template<typename T, int kTRows_, int kTCols_>
bool RunReduceTilingPerfKernel(
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

    // Calculate sizes based on config
    const int total_rows = config.total_rows;
    const int total_cols = config.total_cols;
    const size_t count = static_cast<size_t>(total_rows) * total_cols;
    const size_t fileSize = count * sizeof(T);
    
    // Calculate tiling info
    const int rows_per_block = (total_rows + config.block_num - 1) / config.block_num;
    const int tiles_per_block = (rows_per_block + kTRows_ - 1) / kTRows_;
    const size_t tile_size = static_cast<size_t>(kTRows_) * kTCols_ * sizeof(T);
    
    T *srcHost, *resHost;
    aclrtMallocHost((void **)(&srcHost), fileSize);
    aclrtMallocHost((void **)(&resHost), fileSize);
    T seed = 1280;
    for (size_t i = 0; i < count; ++i) {
        srcHost[i] = static_cast<T>((rank_id + 1) * 100 + (i % 1000) + seed);
        resHost[i] = static_cast<T>(-1);
    }
    
    T *srcDevice, *resDevice;
    aclrtMalloc((void **)(&srcDevice), fileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)(&resDevice), fileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemcpy(srcDevice, fileSize, srcHost, fileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(resDevice, fileSize, resHost, fileSize, ACL_MEMCPY_HOST_TO_DEVICE);

    uint64_t fftsConfig = util_get_ffts_config();
    
    // Only need 1x tensor size in shmem (local rank's data for others to read)
    void *shmemBufferDevice = ShmemMalloc(fileSize);

    if(shmemBufferDevice == nullptr){
        std::cerr << "[ERROR] ShmemMalloc failed! Requested " << (fileSize / (1024*1024)) << " MB" << std::endl;
        ShmemFinalize();
        return false;
    }

    // Allocate space for results: [cycles, actual_tiles, expected_tiles, rows_per_block]
    int64_t *perfHost;
    aclrtMallocHost(reinterpret_cast<void**>(&perfHost), 4 * sizeof(int64_t));
    
    int64_t *perfDevice;
    aclrtMalloc(reinterpret_cast<void**>(&perfDevice), 4 * sizeof(int64_t), ACL_MEM_MALLOC_HUGE_FIRST);

    if (rank_id == 0 && config.verbose) {
        std::cout << "\n[PERF] TREDUCE Tiling Bandwidth Test" << std::endl;
        std::cout << "================================================================" << std::endl;
        std::cout << "  Configuration:" << std::endl;
        std::cout << "    - Total tensor:     " << total_rows << " x " << total_cols << std::endl;
        std::cout << "    - Total size:       " << fileSize << " bytes (" << (fileSize / 1024.0) << " KB, " 
                  << (fileSize / (1024.0 * 1024.0)) << " MB)" << std::endl;
        std::cout << "    - Tile size:        " << kTRows_ << " x " << kTCols_ << " = " << tile_size << " bytes" << std::endl;
        std::cout << "    - Blocks (cores):   " << config.block_num << std::endl;
        std::cout << "    - Rows per block:   " << rows_per_block << std::endl;
        std::cout << "    - Tiles per block:  " << tiles_per_block << std::endl;
        std::cout << "\n[PERF] Starting warmup (" << config.warmup_iters << " kernel launches)..." << std::endl;
    }

    // Warmup
    for(int i = 0; i < config.warmup_iters; ++i) {
        TReduceTilingKernelImpl<T, kTRows_, kTCols_><<<config.block_num, nullptr, stream>>>(
            srcDevice, resDevice, (T*) shmemBufferDevice, nullptr, 
            total_rows, total_cols, config.block_num, fftsConfig);
        aclrtSynchronizeStream(stream);
    }

    ShmemBarrierAll();

    if (rank_id == 0 && config.verbose) {
        std::cout << "[PERF] Starting measurement..." << std::endl;
    }
    
    aclrtMemcpy(resDevice, fileSize, resHost, fileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemset(perfDevice, 4 * sizeof(int64_t), 0, 4 * sizeof(int64_t));
    
    aclrtSynchronizeStream(stream);
    ShmemBarrierAll();

    auto wall_start = std::chrono::high_resolution_clock::now();
    
    TReduceTilingKernelImpl<T, kTRows_, kTCols_><<<config.block_num, nullptr, stream>>>(
        srcDevice, resDevice, (T*) shmemBufferDevice, perfDevice,
        total_rows, total_cols, config.block_num, fftsConfig);
    aclrtSynchronizeStream(stream);
    
    auto wall_end = std::chrono::high_resolution_clock::now();
    double wall_time_ms = std::chrono::duration<double, std::milli>(wall_end - wall_start).count();

    ShmemBarrierAll();

    // Copy results back
    aclrtMemcpy(perfHost, 4 * sizeof(int64_t), perfDevice, 4 * sizeof(int64_t), ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(resHost, fileSize, resDevice, fileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    // ========================================================================
    // Results Analysis (only rank 0)
    // ========================================================================
    if (rank_id == 0) {
        int64_t total_cycles = perfHost[0];
        int64_t actual_tiles = perfHost[1];
        int64_t expected_tiles = perfHost[2];
        int64_t block0_rows = perfHost[3];
        
        // Total tiles processed by all blocks
        int64_t total_tiles = actual_tiles * config.block_num;
        
        // Convert cycles to microseconds (cycles / 50.0 = us)
        constexpr double CYCLES_PER_US = 50.0;
        double total_time_us = static_cast<double>(total_cycles) / CYCLES_PER_US;
        double avg_tile_latency_us = total_time_us / actual_tiles;
        
        // Calculate total data transferred:
        // Each tile: reads tile_size from all ranks
        size_t data_per_tile = tile_size * n_ranks;
        size_t total_data_transferred = data_per_tile * total_tiles;
        
        // Bandwidth = total data / total time
        double bandwidth_gbps = (static_cast<double>(total_data_transferred) / (total_time_us * 1e-6)) / (1024.0 * 1024.0 * 1024.0);
        double bandwidth_per_block = bandwidth_gbps / config.block_num;
        
        // Print summary
        std::cout << "\n================================================================" << std::endl;
        std::cout << "  TREDUCE Tiling Bandwidth Test Results" << std::endl;
        std::cout << "================================================================" << std::endl;
        std::cout << "  Tensor Info:" << std::endl;
        std::cout << "    - Total tensor:    " << total_rows << " x " << total_cols << " = " 
                  << (fileSize / 1024.0) << " KB" << std::endl;
        std::cout << "    - Tile size:       " << kTRows_ << " x " << kTCols_ << " = " 
                  << (tile_size / 1024.0) << " KB" << std::endl;
        std::cout << "    - Block count:     " << config.block_num << std::endl;
        std::cout << "    - Rank count:      " << n_ranks << std::endl;
        std::cout << std::endl;
        std::cout << "  Tiling Info:" << std::endl;
        std::cout << "    - Tiles/block:     " << actual_tiles << " (expected: " << expected_tiles << ")" << std::endl;
        std::cout << "    - Total tiles:     " << total_tiles << std::endl;
        std::cout << "    - Rows for block0: " << block0_rows << std::endl;
        std::cout << std::endl;
        std::cout << "  Timing:" << std::endl;
        std::cout << "    - Total cycles:    " << total_cycles << std::endl;
        std::cout << "    - Total time:      " << std::fixed << std::setprecision(2) 
                  << total_time_us << " us (" << (total_time_us / 1000.0) << " ms)" << std::endl;
        std::cout << "    - Avg tile time:   " << avg_tile_latency_us << " us/tile" << std::endl;
        std::cout << std::endl;
        std::cout << "  Data Transfer:" << std::endl;
        std::cout << "    - Per tile:        " << data_per_tile << " bytes (read from " << n_ranks << " ranks)" << std::endl;
        std::cout << "    - Total transfer:  " << total_data_transferred << " bytes (" 
                  << (total_data_transferred / (1024.0 * 1024.0)) << " MB)" << std::endl;
        std::cout << std::endl;
        std::cout << "  Throughput:" << std::endl;
        std::cout << "    - Aggregate BW:    " << std::setprecision(3) << bandwidth_gbps << " GB/s (all " << config.block_num << " cores)" << std::endl;
        std::cout << "    - BW per core:     " << bandwidth_per_block << " GB/s" << std::endl;
        std::cout << std::endl;
        std::cout << "  Wall Clock Time:     " << std::setprecision(2) << wall_time_ms << " ms" << std::endl;
        std::cout << "================================================================\n" << std::endl;
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
template <typename T, int kTRows_, int kTCols_>
bool RunReduceTilingPerf(
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
            const bool ok = RunReduceTilingPerfKernel<T, kTRows_, kTCols_>(
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
// Explicit Instantiations
// Tile size is constrained by UB size (need 2 tiles in UB)
// Typical UB size: ~192KB, so max tile ~64KB each
// ============================================================================

// Tile 64x64 = 16KB per tile (float) - for 32MB tensor test
template bool RunReduceTilingPerf<float, 64, 64>(int, int, int, int, const PerfTestConfig&);


