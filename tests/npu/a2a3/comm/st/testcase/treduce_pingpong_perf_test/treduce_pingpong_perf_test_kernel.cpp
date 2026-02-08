#include "treduce_pingpong_perf_test.h"

#include "pto/comm/pto_comm_inst.hpp"
#include "pto/common/pto_tile.hpp"
#include "../common.hpp"
#include <pto/pto-inst.hpp>

// ============================================================================
// Device-side Kernel: TReduce with ping-pong double buffering and cycle counting
// ============================================================================
template <typename T, int kTRows_, int kTCols_, int vRows, int vCols>
__global__ AICORE void TReducePingPongPerfKernelImpl(
    __gm__ T *input, __gm__ T *output,
    __gm__ int64_t *cycle_results,  // Store cycle counts
    int iteration)
{
    using ShapeDyn  = pto::Shape<1, 1, 1, vRows, vCols>;
    using StrideDyn = pto::Stride<1, 1, 1, kTCols_, 1>;
    using Global = pto::GlobalTensor<T, ShapeDyn, StrideDyn, pto::Layout::ND>;
    using TileData = pto::Tile<pto::TileType::Vec, T, kTRows_, kTCols_, pto::BLayout::RowMajor, -1, -1>;

    TileData accTile(vRows, vCols);
    TileData pingTile(vRows, vCols);
    TileData pongTile(vRows, vCols);
    
    constexpr size_t tileBytes = kTRows_ * kTCols_ * sizeof(T);
    constexpr size_t alignedTileBytes = ((tileBytes + 31) / 32) * 32;
    
    TASSIGN(accTile, 0);
    TASSIGN(pingTile, 0 + alignedTileBytes);
    TASSIGN(pongTile, 0 + alignedTileBytes * 2);

    int my_rank = shmem_my_pe();
    int nranks = shmem_n_pes();

    ShapeDyn shape(1, 1, 1, vRows, vCols);
    StrideDyn stride(1, 1, 1, kTCols_, 1);
    
    Global dstGlobal(output, shape, stride);

    Global tensors[16];
    int actual_nranks = (nranks > 16) ? 16 : nranks;
    for (int i = 0; i < actual_nranks; ++i) {
        __gm__ T *remoteInput = ShmemPtr(input, i);
        tensors[i] = Global(remoteInput, shape, stride);
    }
    
    pto::comm::ParallelGroup<Global> pg(tensors, actual_nranks, my_rank);
    
    ShmemDeviceBarrierAll();

    AscendC::PipeBarrier<PIPE_ALL>();

    int64_t start_cycle = AscendC::GetSystemCycle();

    if (my_rank == 0) {
        pto::comm::TREDUCE_PINGPONG(pg, dstGlobal, accTile, pingTile, pongTile, pto::comm::ReduceOp::Sum);
    }

    AscendC::PipeBarrier<PIPE_ALL>();
    
    int64_t end_cycle = AscendC::GetSystemCycle();
    
    if (my_rank == 0 && cycle_results != nullptr) {
        cycle_results[iteration] = end_cycle - start_cycle;
    }
}

// ============================================================================
// Host-side Performance Test Runner
// ============================================================================
template<typename T, int kTRows_, int kTCols_, int vRows, int vCols>
bool RunReducePingPongPerfKernel(
    int rank_id, 
    int n_ranks, 
    int n_devices, 
    int first_device_id,
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
    const char *ip = "tcp://127.0.0.1:8767";
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
    T seed = 3367;
    for (int i = 0; i < count; ++i) {
        srcHost[i] = static_cast<T>((rank_id + 1) * 100 + i + seed);
        resHost[i] = static_cast<T>(-1);
    }
    T *srcDevice, *resDevice;
    aclrtMalloc((void **)(&srcDevice), fileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)(&resDevice), fileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    void *shmemInputDevice = ShmemMalloc(fileSize);
    if (shmemInputDevice == nullptr) {
        std::cerr << "[ERROR] ShmemMalloc failed!" << std::endl;
        ShmemFinalize();
        return false;
    }

    aclrtMemcpy(srcDevice, fileSize, srcHost, fileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(shmemInputDevice, fileSize, srcHost, fileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(resDevice, fileSize, resHost, fileSize, ACL_MEMCPY_HOST_TO_DEVICE);

    int64_t *perfHost;
    aclrtMallocHost(reinterpret_cast<void**>(&perfHost), config.measure_iters * sizeof(int64_t));
    
    int64_t *perfDevice;
    aclrtMalloc(reinterpret_cast<void**>(&perfDevice), config.measure_iters * sizeof(int64_t), ACL_MEM_MALLOC_HUGE_FIRST);

    if (rank_id == 0 && config.verbose) {
        std::cout << "\n[PERF] Starting warmup (" << config.warmup_iters << " iterations)..." << std::endl;
    }

    for(int i = 0; i < config.warmup_iters; ++i) {
        TReducePingPongPerfKernelImpl<T, kTRows_, kTCols_, vRows, vCols><<<1, nullptr, stream>>>(
            (T*)shmemInputDevice, resDevice, nullptr, 0);
        aclrtSynchronizeStream(stream);
    }

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
        TReducePingPongPerfKernelImpl<T, kTRows_, kTCols_, vRows, vCols><<<1, nullptr, stream>>>(
            (T*)shmemInputDevice, resDevice, perfDevice, i);
        aclrtSynchronizeStream(stream);
    }
    auto wall_end = std::chrono::high_resolution_clock::now();
    double wall_time_ms = std::chrono::duration<double, std::milli>(wall_end - wall_start).count();

    ShmemBarrierAll();

    aclrtMemcpy(perfHost, config.measure_iters * sizeof(int64_t), 
                perfDevice, config.measure_iters * sizeof(int64_t),
                ACL_MEMCPY_DEVICE_TO_HOST);

    if (rank_id == 0) {
        aclrtMemcpy(resHost, fileSize, resDevice, fileSize, ACL_MEMCPY_DEVICE_TO_HOST);

        bool flag = true;
        for(int i = 0; i < count; ++i) {
            if (config.verbose) {
                if(i < 5) std::cout << "resHost[" << i << "] = " << resHost[i] << std::endl;
            }

            T expected = static_cast<T>(n_ranks * (i + seed) + 100 * (n_ranks * (n_ranks + 1) / 2));
            T actual = resHost[i];
            if (actual != expected) {
                std::cerr << "[ERROR] TREDUCE_PINGPONG failed\n";
                flag = false;
                status = 1;
                break;
            }
        }

        if(flag){
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
            
            std::cout << "\n[PERF] Results Summary (n_ranks=" << n_ranks << ")" << std::endl;
            std::cout << "  Min Latency:    " << stats.min_us << " us" << std::endl;
            std::cout << "  Max Latency:    " << stats.max_us << " us" << std::endl;
            std::cout << "  Avg Latency:    " << stats.avg_us << " us" << std::endl;
            std::cout << "  Median Latency: " << stats.median_us << " us" << std::endl;
            std::cout << "  Std Dev:        " << stats.std_dev_us << " us" << std::endl;
            std::cout << "  Bandwidth:      " << stats.bandwidth_gbps << " GB/s" << std::endl;
            std::cout << "  Msg Rate:       " << stats.msg_rate_mops << " Mops/s" << std::endl;
            std::cout << "  Wall Time:      " << wall_time_ms << " ms" << std::endl;
        }
    }

    aclrtFreeHost(srcHost);
    aclrtFreeHost(resHost);
    aclrtFreeHost(perfHost);
    aclrtFree(srcDevice);
    aclrtFree(resDevice);
    aclrtFree(perfDevice);
    ShmemFree(shmemInputDevice);

    ShmemFinalize();

    status |= aclrtDestroyStream(stream);
    status |= aclrtResetDevice(device_id);
    status |= aclFinalize();

    return (status == 0);
}

// ============================================================================
// Multi-process Launcher
// ============================================================================
template<typename T, int kTRows_, int kTCols_, int vRows, int vCols>
bool RunReducePingPongPerf(
    int n_ranks, 
    int n_devices, 
    int first_rank_id, 
    int first_device_id,
    const PerfTestConfig &config)
{
    std::vector<pid_t> pids;
    for (int r = 0; r < n_ranks; ++r) {
        pid_t pid = fork();
        if (pid == 0) {
            const bool ok = RunReducePingPongPerfKernel<T, kTRows_, kTCols_, vRows, vCols>(
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

// Explicit instantiations
template bool RunReducePingPongPerf<float, 64, 64, 64, 64>(
    int n_ranks, int n_devices, int first_rank_id, int first_device_id, const PerfTestConfig &config);
template bool RunReducePingPongPerf<float, 16, 256, 16, 256>(
    int n_ranks, int n_devices, int first_rank_id, int first_device_id, const PerfTestConfig &config);
template bool RunReducePingPongPerf<int32_t, 128, 128, 128, 128>(
    int n_ranks, int n_devices, int first_rank_id, int first_device_id, const PerfTestConfig &config);
template bool RunReducePingPongPerf<int32_t, 64, 256, 64, 256>(
    int n_ranks, int n_devices, int first_rank_id, int first_device_id, const PerfTestConfig &config);
