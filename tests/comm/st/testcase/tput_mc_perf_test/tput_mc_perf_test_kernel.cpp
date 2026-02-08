#include "tput_mc_perf_test.h"

#include "pto/comm/pto_comm_inst.hpp"
#include "pto/common/pto_tile.hpp"
#include "../common.hpp"
#include <pto/pto-inst.hpp>
#include "pto/npu/a2a3/custom/TSyncCVID.hpp"

// ============================================================================
// Helper: Compute destination rank for a given block index.
//
// Evenly distributes blocks across all (nranks - 1) peer ranks, skipping self.
// ============================================================================
AICORE inline int GetDestRank(int block_idx, int my_rank, int nranks)
{
    const int peer_count = nranks - 1;
    const int peer_idx = block_idx % peer_count;
    return (my_rank + 1 + peer_idx) % nranks;
}

// ============================================================================
// Kernel 1: AutoChunk with inner loop for steady-state measurement
//
// The kernel executes TPUT `repeat_iters` times inside a loop.
// This eliminates cold-start overhead from the measurement and lets the
// DMA engines + HCCS links reach steady-state throughput.
// ============================================================================
template <typename T, int kTRows_, int kTCols_>
__global__ AICORE void TPutAutoChunkPerfKernel(
    __gm__ T *input, __gm__ T *output, __gm__ T *shmem,
    int total_rows, int total_cols, int total_blocks,
    int repeat_iters, uint64_t fftsConfig)
{
    util_set_ffts_config(fftsConfig);
    ShmemDeviceBarrierAll();

    const int block_idx = get_block_idx();
    const int n_blocks = (total_blocks > 0) ? total_blocks : 1;
    int my_rank = shmem_my_pe();
    int nranks = shmem_n_pes();

    const int rows_per_block = (total_rows + n_blocks - 1) / n_blocks;
    const int my_start_row = block_idx * rows_per_block;
    const int my_end_row = (my_start_row + rows_per_block > total_rows)
                               ? total_rows : (my_start_row + rows_per_block);
    const int my_total_rows = my_end_row - my_start_row;

    const bool has_work = (my_total_rows > 0);

    using DynShape  = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using DynStride = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using Global    = pto::GlobalTensor<T, DynShape, DynStride, pto::Layout::ND>;
    using TileData  = Tile<TileType::Vec, T, kTRows_, kTCols_, BLayout::RowMajor, -1, -1>;

    if (has_work) {
        TileData stagingTile(kTRows_, kTCols_);
        TASSIGN(stagingTile, 0);

        const int64_t row_offset   = static_cast<int64_t>(my_start_row) * total_cols;
        const int64_t region_elems = static_cast<int64_t>(my_total_rows) * total_cols;

        DynShape  regionShape(1, 1, 1, my_total_rows, total_cols);
        DynStride regionStride(region_elems, region_elems, region_elems, total_cols, 1);

        int effective_dest_rank = GetDestRank(block_idx, my_rank, nranks);
        __gm__ T *dest_shmem = (__gm__ T *)shmem_ptr(shmem, effective_dest_rank);

        Global srcGlobal(input + row_offset, regionShape, regionStride);
        Global dstGlobal(dest_shmem + row_offset, regionShape, regionStride);

        // Sync before timed loop
        AscendC::PipeBarrier<PIPE_ALL>();
        aclshmemi_barrier_core_soft();
        ShmemDeviceBarrierAll();
        AscendC::PipeBarrier<PIPE_ALL>();

        // ====== TIMED: Loop TPUT for steady-state bandwidth ======
        for (int iter = 0; iter < repeat_iters; ++iter) {
            pto::comm::TPUT(dstGlobal, srcGlobal, stagingTile);
        }

        // Ensure all remote writes are visible
        AscendC::PipeBarrier<PIPE_ALL>();
        ShmemDeviceBarrierAll();
    } else {
        AscendC::PipeBarrier<PIPE_ALL>();
        aclshmemi_barrier_core_soft();
        ShmemDeviceBarrierAll();
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::PipeBarrier<PIPE_ALL>();
        ShmemDeviceBarrierAll();
    }

    ShmemDeviceBarrierAll();
}

// ============================================================================
// Kernel 2: PingPong with inner loop for steady-state measurement
// ============================================================================
template <typename T, int kTRows_, int kTCols_>
__global__ AICORE void TPutPingPongPerfKernel(
    __gm__ T *input, __gm__ T *output, __gm__ T *shmem,
    int total_rows, int total_cols, int total_blocks,
    int repeat_iters, uint64_t fftsConfig)
{
    util_set_ffts_config(fftsConfig);
    ShmemDeviceBarrierAll();

    const int block_idx = get_block_idx();
    const int n_blocks = (total_blocks > 0) ? total_blocks : 1;
    int my_rank = shmem_my_pe();
    int nranks = shmem_n_pes();

    const int rows_per_block = (total_rows + n_blocks - 1) / n_blocks;
    const int my_start_row = block_idx * rows_per_block;
    const int my_end_row = (my_start_row + rows_per_block > total_rows)
                               ? total_rows : (my_start_row + rows_per_block);
    const int my_total_rows = my_end_row - my_start_row;

    const bool has_work = (my_total_rows > 0);

    using DynShape  = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using DynStride = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using Global    = pto::GlobalTensor<T, DynShape, DynStride, pto::Layout::ND>;
    using TileData  = Tile<TileType::Vec, T, kTRows_, kTCols_, BLayout::RowMajor, -1, -1>;

    if (has_work) {
        constexpr size_t tileUBBytes =
            ((kTRows_ * kTCols_ * sizeof(T) + 1023) / 1024) * 1024;
        TileData pingTile(kTRows_, kTCols_);
        TileData pongTile(kTRows_, kTCols_);
        TASSIGN(pingTile, 0);
        TASSIGN(pongTile, tileUBBytes);

        const int64_t row_offset   = static_cast<int64_t>(my_start_row) * total_cols;
        const int64_t region_elems = static_cast<int64_t>(my_total_rows) * total_cols;

        DynShape  regionShape(1, 1, 1, my_total_rows, total_cols);
        DynStride regionStride(region_elems, region_elems, region_elems, total_cols, 1);

        int effective_dest_rank = GetDestRank(block_idx, my_rank, nranks);
        __gm__ T *dest_shmem = (__gm__ T *)shmem_ptr(shmem, effective_dest_rank);

        Global srcGlobal(input + row_offset, regionShape, regionStride);
        Global dstGlobal(dest_shmem + row_offset, regionShape, regionStride);

        AscendC::PipeBarrier<PIPE_ALL>();
        aclshmemi_barrier_core_soft();
        ShmemDeviceBarrierAll();
        AscendC::PipeBarrier<PIPE_ALL>();

        // ====== TIMED: Loop TPUT for steady-state bandwidth ======
        for (int iter = 0; iter < repeat_iters; ++iter) {
            pto::comm::TPUT(dstGlobal, srcGlobal, pingTile, pongTile);
        }

        AscendC::PipeBarrier<PIPE_ALL>();
        ShmemDeviceBarrierAll();
    } else {
        AscendC::PipeBarrier<PIPE_ALL>();
        aclshmemi_barrier_core_soft();
        ShmemDeviceBarrierAll();
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::PipeBarrier<PIPE_ALL>();
        ShmemDeviceBarrierAll();
    }

    ShmemDeviceBarrierAll();
}

// ============================================================================
// Host-side Performance Test Runner
//
// Timing strategy:
//   - Warmup: launch kernel warmup_iters times (each with repeat_iters loops)
//   - Measured: launch kernel once with repeat_iters loops, measure wall clock
//   - BW = (fileSize * repeat_iters) / wall_time
//
// This gives steady-state bandwidth by:
//   1. Warmup eliminates cold-start (TLB, connection setup, etc.)
//   2. Inner loop eliminates per-kernel-launch overhead from BW calculation
//   3. Wall clock captures the actual end-to-end time reliably
// ============================================================================
template <typename T, int kTRows_, int kTCols_, bool usePingPong>
bool RunPutPerfKernel(
    int rank_id, int n_ranks, int n_devices, int first_device_id,
    const PerfTestConfig &config)
{
    int32_t ret = ShmemSetConfStoreTls(false, nullptr, 0);
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

    const int total_rows = config.total_rows;
    const int total_cols = config.total_cols;
    const size_t count = static_cast<size_t>(total_rows) * total_cols;
    const size_t fileSize = count * sizeof(T);

    const int rows_per_block = (total_rows + config.block_num - 1) / config.block_num;
    const int row_chunks = (rows_per_block + kTRows_ - 1) / kTRows_;
    const int col_chunks = (total_cols + kTCols_ - 1) / kTCols_;
    const size_t tile_bytes = static_cast<size_t>(kTRows_) * kTCols_ * sizeof(T);

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

    uint64_t fftsConfig_val = util_get_ffts_config();

    void *shmemBufferDevice = ShmemMalloc(fileSize);
    if (shmemBufferDevice == nullptr) {
        std::cerr << "[ERROR] ShmemMalloc failed! Requested "
                  << (fileSize / (1024 * 1024)) << " MB" << std::endl;
        ShmemFinalize();
        return false;
    }

    const char *mode_name = usePingPong ? "PingPong" : "AutoChunk";
    const int peer_count = n_ranks - 1;
    const int blocks_per_peer_min = config.block_num / peer_count;
    const int blocks_per_peer_max = blocks_per_peer_min + (config.block_num % peer_count != 0 ? 1 : 0);

    // Total data moved in measured run = fileSize * repeat_iters
    const size_t total_measured_bytes =
        static_cast<size_t>(fileSize) * config.repeat_iters;

    if (rank_id == 0 && config.verbose) {
        std::cout << "\n[PERF] TPUT " << mode_name << " Bandwidth Test" << std::endl;
        std::cout << "================================================================" << std::endl;
        std::cout << "  Configuration:" << std::endl;
        std::cout << "    - Total tensor:     " << total_rows << " x " << total_cols
                  << " (" << (fileSize / (1024.0 * 1024.0)) << " MB)" << std::endl;
        std::cout << "    - Tile size:        " << kTRows_ << " x " << kTCols_
                  << " (" << (tile_bytes / 1024.0) << " KB)" << std::endl;
        std::cout << "    - Blocks (cores):   " << config.block_num << std::endl;
        std::cout << "    - Chunks per block: " << row_chunks << " x " << col_chunks
                  << " = " << (row_chunks * col_chunks) << std::endl;
        std::cout << "    - Mode:             " << mode_name << std::endl;
        std::cout << "    - Warmup iters:     " << config.warmup_iters << std::endl;
        std::cout << "    - Repeat iters:     " << config.repeat_iters << std::endl;
        std::cout << "    - Total measured:   "
                  << (total_measured_bytes / (1024.0 * 1024.0)) << " MB ("
                  << config.repeat_iters << " x " << (fileSize / (1024.0 * 1024.0)) << " MB)" << std::endl;
        std::cout << "    - Dest distribution: " << config.block_num << " blocks -> "
                  << peer_count << " peers (" << blocks_per_peer_min
                  << "~" << blocks_per_peer_max << " blocks/peer)" << std::endl;
    }

    // ====== Warmup: multiple kernel launches with inner loops ======
    if (rank_id == 0 && config.verbose) {
        std::cout << "\n[PERF] Warmup (" << config.warmup_iters << " kernel launches, "
                  << config.repeat_iters << " iters each)..." << std::endl;
    }
    for (int i = 0; i < config.warmup_iters; ++i) {
        if constexpr (usePingPong) {
            TPutPingPongPerfKernel<T, kTRows_, kTCols_><<<config.block_num, nullptr, stream>>>(
                srcDevice, resDevice, (T *)shmemBufferDevice,
                total_rows, total_cols, config.block_num,
                config.repeat_iters, fftsConfig_val);
        } else {
            TPutAutoChunkPerfKernel<T, kTRows_, kTCols_><<<config.block_num, nullptr, stream>>>(
                srcDevice, resDevice, (T *)shmemBufferDevice,
                total_rows, total_cols, config.block_num,
                config.repeat_iters, fftsConfig_val);
        }
        aclrtSynchronizeStream(stream);
    }

    ShmemBarrierAll();

    if (rank_id == 0 && config.verbose) {
        std::cout << "[PERF] Measuring..." << std::endl;
    }

    ShmemBarrierAll();

    // ====== Measured run ======
    auto wall_start = std::chrono::high_resolution_clock::now();

    if constexpr (usePingPong) {
        TPutPingPongPerfKernel<T, kTRows_, kTCols_><<<config.block_num, nullptr, stream>>>(
            srcDevice, resDevice, (T *)shmemBufferDevice,
            total_rows, total_cols, config.block_num,
            config.repeat_iters, fftsConfig_val);
    } else {
        TPutAutoChunkPerfKernel<T, kTRows_, kTCols_><<<config.block_num, nullptr, stream>>>(
            srcDevice, resDevice, (T *)shmemBufferDevice,
            total_rows, total_cols, config.block_num,
            config.repeat_iters, fftsConfig_val);
    }
    aclrtSynchronizeStream(stream);

    auto wall_end = std::chrono::high_resolution_clock::now();
    double wall_time_ms = std::chrono::duration<double, std::milli>(wall_end - wall_start).count();

    ShmemBarrierAll();

    // ====================================================================
    // Results
    // ====================================================================
    if (rank_id == 0) {
        double wall_time_us = wall_time_ms * 1000.0;

        // Steady-state bandwidth = total_data_moved / wall_time
        double bandwidth_gbps = (wall_time_us > 0)
            ? (static_cast<double>(total_measured_bytes) / (wall_time_us * 1e-6))
              / (1024.0 * 1024.0 * 1024.0)
            : 0.0;

        double per_iter_us = wall_time_us / config.repeat_iters;
        double per_link_bw = (peer_count > 0) ? bandwidth_gbps / peer_count : 0.0;
        double per_link_data_mb =
            (fileSize / (1024.0 * 1024.0)) / peer_count;

        constexpr double THEORETICAL_BW_GBPS = 192.0;  // 910B single-direction (392 bidirectional / 2)
        double efficiency_pct = (bandwidth_gbps / THEORETICAL_BW_GBPS) * 100.0;

        std::cout << "\n================================================================" << std::endl;
        std::cout << "  TPUT " << mode_name << " Steady-State Bandwidth" << std::endl;
        std::cout << "================================================================" << std::endl;
        std::cout << "  Tensor:          " << total_rows << " x " << total_cols
                  << " = " << std::fixed << std::setprecision(0)
                  << (fileSize / (1024.0 * 1024.0)) << " MB" << std::endl;
        std::cout << "  Tile:            " << kTRows_ << " x " << kTCols_
                  << " = " << std::setprecision(0) << (tile_bytes / 1024.0) << " KB" << std::endl;
        std::cout << "  Blocks:          " << config.block_num
                  << " -> " << peer_count << " peers ("
                  << std::setprecision(2) << per_link_data_mb << " MB/link/iter)" << std::endl;
        std::cout << "  Repeat iters:    " << config.repeat_iters << std::endl;
        std::cout << "  Total data:      " << std::setprecision(0)
                  << (total_measured_bytes / (1024.0 * 1024.0)) << " MB" << std::endl;
        std::cout << std::endl;
        std::cout << "  Wall time:       " << std::setprecision(2) << wall_time_ms << " ms"
                  << " (" << std::setprecision(2) << per_iter_us << " us/iter)" << std::endl;
        std::cout << std::endl;
        std::cout << "  *** Aggregate BW:  " << std::setprecision(3)
                  << bandwidth_gbps << " GB/s ***" << std::endl;
        std::cout << "  BW per link:     " << per_link_bw << " GB/s" << std::endl;
        std::cout << "  Efficiency:      " << std::setprecision(1) << efficiency_pct
                  << "% of " << std::setprecision(0) << THEORETICAL_BW_GBPS << " GB/s" << std::endl;
        std::cout << "================================================================\n" << std::endl;
    }

    ShmemBarrierAll();

    aclrtFreeHost(srcHost);
    aclrtFreeHost(resHost);
    aclrtFree(srcDevice);
    aclrtFree(resDevice);
    ShmemFree(shmemBufferDevice);

    ShmemFinalize();

    status |= aclrtDestroyStream(stream);
    status |= aclrtResetDevice(device_id);
    status |= aclFinalize();

    return (status == 0);
}

// ============================================================================
// Multi-process Launchers
// ============================================================================
template <typename T, int kTRows_, int kTCols_>
bool RunPutAutoChunkPerf(
    int n_ranks, int n_devices, int first_rank_id, int first_device_id,
    const PerfTestConfig &config)
{
    std::vector<pid_t> pids;
    for (int r = 0; r < n_ranks; ++r) {
        pid_t pid = fork();
        if (pid == 0) {
            const bool ok = RunPutPerfKernel<T, kTRows_, kTCols_, false>(
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
        int s = 0;
        waitpid(p, &s, 0);
        if (!(WIFEXITED(s) && WEXITSTATUS(s) == 0)) success = false;
    }
    return success;
}

template <typename T, int kTRows_, int kTCols_>
bool RunPutPingPongPerf(
    int n_ranks, int n_devices, int first_rank_id, int first_device_id,
    const PerfTestConfig &config)
{
    std::vector<pid_t> pids;
    for (int r = 0; r < n_ranks; ++r) {
        pid_t pid = fork();
        if (pid == 0) {
            const bool ok = RunPutPerfKernel<T, kTRows_, kTCols_, true>(
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
        int s = 0;
        waitpid(p, &s, 0);
        if (!(WIFEXITED(s) && WEXITSTATUS(s) == 0)) success = false;
    }
    return success;
}

// ============================================================================
// Explicit Instantiations
// ============================================================================
template bool RunPutAutoChunkPerf<float, 64, 64>(int, int, int, int, const PerfTestConfig &);
template bool RunPutPingPongPerf<float, 64, 64>(int, int, int, int, const PerfTestConfig &);

template bool RunPutAutoChunkPerf<float, 64, 128>(int, int, int, int, const PerfTestConfig &);
template bool RunPutPingPongPerf<float, 64, 128>(int, int, int, int, const PerfTestConfig &);

template bool RunPutAutoChunkPerf<float, 32, 64>(int, int, int, int, const PerfTestConfig &);
template bool RunPutPingPongPerf<float, 32, 64>(int, int, int, int, const PerfTestConfig &);
