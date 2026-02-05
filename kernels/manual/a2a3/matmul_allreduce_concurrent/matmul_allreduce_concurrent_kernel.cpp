/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include "matmul_allreduce_concurrent.h"


#include "pto/common/pto_tile.hpp"
#include "common.hpp"
#include <pto/pto-inst.hpp>

using namespace pto;

// Detect build-time macros and expose as constexpr flags for clearer conditionals
#ifdef __DAV_CUBE__
constexpr bool DAV_CUBE = true;
#else
constexpr bool DAV_CUBE = false;
#endif

#ifdef __DAV_VEC__
constexpr bool DAV_VEC = true;
#include "pto/comm/pto_comm_inst.hpp"
#else
constexpr bool DAV_VEC = false;
#endif


// ============================================================================
// GEMM Computation Implementation (for compute cores)
// Simple matrix multiplication: C = A * B
// ============================================================================

#ifdef __DAV_C220_CUBE__

template <typename T, typename U, typename S, 
          uint32_t baseM, uint32_t baseK, uint32_t baseN>
AICORE inline void RunGEMMOnCore(
    __gm__ U *srcA, 
    __gm__ S *srcB, 
    __gm__ T *dstC,
    int core_idx_in_gemm_group)
{
    using NDValidShapeA = TileShape2D<U, baseM, baseK>;
    using NDsingleCoreShapeA = BaseShape2D<U, baseM, baseK>;
    using GlobalDataSrcA = GlobalTensor<U, NDValidShapeA, NDsingleCoreShapeA>;

    using NDValidShapeB = TileShape2D<U, baseK, baseN, Layout::DN>;
    using NDsingleCoreShapeB = BaseShape2D<U, baseK, baseN, Layout::DN>;
    using GlobalDataSrcB = GlobalTensor<U, NDValidShapeB, NDsingleCoreShapeB, Layout::DN>;

    using NDValidShapeC = TileShape2D<T, baseM, baseN>;
    using NDWholeShapeC = BaseShape2D<T, baseM, baseN>;
    using GlobalDataOut = GlobalTensor<T, NDValidShapeC, NDWholeShapeC>;

    // Each GEMM core handles a different partition (offset by core_idx_in_gemm_group)
    uint64_t currentGmOffsetA = static_cast<uint64_t>(core_idx_in_gemm_group) * baseM * baseK;
    uint64_t currentGmOffsetB = static_cast<uint64_t>(core_idx_in_gemm_group) * baseK * baseN;
    uint64_t currentGmOffsetC = static_cast<uint64_t>(core_idx_in_gemm_group) * baseM * baseN;

    __gm__ U *currentSrc0 = srcA + currentGmOffsetA;
    __gm__ S *currentSrc1 = srcB + currentGmOffsetB;
    __gm__ T *currentDst = dstC + currentGmOffsetC;

    using TileMatAData = Tile<TileType::Mat, U, baseM, baseK, BLayout::ColMajor, baseM, baseK, SLayout::RowMajor>;
    using TileMatBData = Tile<TileType::Mat, S, baseK, baseN, BLayout::RowMajor, baseK, baseN, SLayout::ColMajor>;

    TileMatAData aMatTile;
    TileMatBData bMatTile;
    TASSIGN(aMatTile, 0x0);
    TASSIGN(bMatTile, 0x0 + baseM * baseK * sizeof(U));

    using LeftTile = TileLeft<U, baseM, baseK, baseM, baseK>;
    using RightTile = TileRight<S, baseK, baseN, baseK, baseN>;
    using ResTile = TileAcc<T, baseM, baseN, baseM, baseN>;

    LeftTile aTile;
    RightTile bTile;
    ResTile cTile;
    TASSIGN(aTile, 0x0);
    TASSIGN(bTile, 0x0);
    TASSIGN(cTile, 0x0);

    GlobalDataSrcA gmA(currentSrc0);
    GlobalDataSrcB gmB(currentSrc1);
    GlobalDataOut dstGlobal(currentDst);

    // Load A matrix
    TLOAD(aMatTile, gmA);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    
    // Load B matrix
    TLOAD(bMatTile, gmB);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
    
    // Move to L0
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    TMOV(aTile, aMatTile);
    
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
    TMOV(bTile, bMatTile);
    
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    
    // Matrix multiplication
    TMATMUL(cTile, aTile, bTile);
    
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    
    // Store result
    TSTORE(dstGlobal, cTile);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
}

#endif 

// ============================================================================
// AllReduce Communication Implementation (for communication cores)
// ============================================================================

#ifdef __DAV_C220_VEC__

template <typename T, int kTRows, int kTCols>
AICORE inline void RunAllReduceOnCore(
    __gm__ T *input, 
    __gm__ T *output, 
    __gm__ T *shmem,
    int core_idx_in_allreduce_group,
    int nranks,
    int my_rank)
{
    constexpr int vRows = kTRows;
    constexpr int vCols = kTCols;
    
    using ShapeDyn  = Shape<1, 1, 1, vRows, vCols>;
    using StrideDyn = Stride<1, 1, 1, kTCols, 1>;
    using Global = GlobalTensor<T, ShapeDyn, StrideDyn, Layout::ND>;
    using TileData = Tile<TileType::Vec, T, kTRows, kTCols, BLayout::RowMajor, -1, -1>;

    // Each AllReduce core handles a different partition
    const int64_t row_offset = static_cast<int64_t>(core_idx_in_allreduce_group) * static_cast<int64_t>(kTRows) * static_cast<int64_t>(kTCols);
    
    TileData src0Tile(vRows, vCols);
    TileData src1Tile(vRows, vCols);
    TileData dstTile(vRows, vCols);
    
    constexpr size_t tileBytes = kTRows * kTCols * sizeof(T);
    constexpr size_t alignedTileBytes = ((tileBytes + 31) / 32) * 32;
    
    TASSIGN(src0Tile, 0);
    TASSIGN(src1Tile, alignedTileBytes);
    TASSIGN(dstTile, alignedTileBytes * 2);

    ShapeDyn shape(1, 1, 1, vRows, vCols);
    StrideDyn stride(1, 1, 1, kTCols, 1);
    
    Global srcGlobal(input + row_offset, shape, stride);
    Global dstGlobal(output + row_offset, shape, stride);
    Global srcShmemGlobal((__gm__ T*)shmem + row_offset, shape, stride);

    // Load input data to shmem region
    TLOAD(src0Tile, srcGlobal);
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    TSTORE(srcShmemGlobal, src0Tile);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);

    // Global barrier: ensure ALL ranks have written to shmem before any rank reads
    // ShmemDeviceBarrierAll();

    // Create ParallelGroup for this core's partition
    Global tensors[16];
    
    for (int i = 0; i < nranks; ++i) {
        __gm__ T* rank_i_base = (__gm__ T*)shmem_ptr(shmem, i);
        __gm__ T* rank_i_partition = rank_i_base + row_offset;
        tensors[i] = Global(rank_i_partition, shape, stride);
        tensors[i].SetRank(i);
    }
    
    pto::comm::ParallelGroup<Global> pg(tensors, nranks, my_rank);
    ShmemDeviceBarrierAll();
    // Perform Reduce (Sum) - uses ping-pong double buffering for better performance
    pto::comm::TREDUCE_PINGPONG(pg, dstGlobal, src0Tile, src1Tile, dstTile, comm::ReduceOp::Sum);
}

#endif


// ============================================================================
// Main Kernel: Partitions cores between GEMM and AllReduce
// ============================================================================
template <typename T, 
          uint32_t gemmBaseM, uint32_t gemmBaseK, uint32_t gemmBaseN,
          int arRows, int arCols>
__global__ AICORE void MatmulAllReduceConcurrentKernel(
    // GEMM inputs/outputs
    __gm__ half *gemmSrcA,
    __gm__ half *gemmSrcB,
    __gm__ float *gemmDstC,
    // AllReduce inputs/outputs
    __gm__ T *arInput,
    __gm__ T *arOutput,
    __gm__ T *arShmem,
    // Configuration
    int total_blocks,
    int gemm_blocks,
    int nranks,
    uint64_t fftsConfig,
    // Performance counters
    __gm__ int64_t *gemm_cycles,
    __gm__ int64_t *allreduce_cycles,
    int iteration)
{
    util_set_ffts_config(fftsConfig);
    ShmemDeviceBarrierAll();
    
    const int block_idx = get_block_idx();
    const int my_rank = shmem_my_pe();
    
    int64_t start_cycle = 0;
    int64_t end_cycle = 0;
    
    // Determine if this core does GEMM or AllReduce
    const bool is_gemm_core = (block_idx < gemm_blocks);
    
    // Debug: Record which blocks are running CUBE vs VEC
    // Use barrier to serialize printf output
#ifdef _DEBUG
    for (int b = 0; b < total_blocks; ++b) {
        if (block_idx == b && my_rank == 0) {
            if constexpr (DAV_CUBE) {
                cce::printf("[CUBE] Block %d executing on Cube unit\n", block_idx);
            }
            if constexpr (DAV_VEC) {
                cce::printf("[VEC]  Block %d, SubBlock %d executing on Vec unit\n", block_idx, get_subblockid());
            }
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }
#endif
    if constexpr (DAV_CUBE) {
        ShmemDeviceBarrierAll();
        // ================================================================
        // GEMM Computation Path
        // ================================================================
        int gemm_core_idx = block_idx;
        
        start_cycle = AscendC::GetSystemCycle();
        
        RunGEMMOnCore<float, half, half, gemmBaseM, gemmBaseK, gemmBaseN>(
            gemmSrcA, gemmSrcB, gemmDstC, gemm_core_idx);
        
        AscendC::PipeBarrier<PIPE_ALL>();
        end_cycle = AscendC::GetSystemCycle();
        
        // Record cycle count (only from first GEMM core of rank 0)
        if (gemm_core_idx == 0 && my_rank == 0 && gemm_cycles != nullptr) {
            gemm_cycles[iteration] = end_cycle - start_cycle;
        }
        
    }
    if constexpr (DAV_VEC) {

        // ================================================================
        // AllReduce Communication Path
        // ================================================================
        int allreduce_core_idx = block_idx * get_subblockdim() + get_subblockid();
        
        // Sync all AllReduce cores before starting
        AscendC::PipeBarrier<PIPE_ALL>();
        aclshmemi_barrier_core_soft();
        AscendC::PipeBarrier<PIPE_ALL>();
        
        start_cycle = AscendC::GetSystemCycle();
        
        RunAllReduceOnCore<T, arRows, arCols>(
            arInput, arOutput, arShmem, allreduce_core_idx, nranks, my_rank);
        
        AscendC::PipeBarrier<PIPE_ALL>();
        aclshmemi_barrier_core_soft();
        end_cycle = AscendC::GetSystemCycle();
        
        // Record cycle count (only from first AllReduce core of rank 0)
        if (allreduce_core_idx == 0 && my_rank == 0 && allreduce_cycles != nullptr) {
            allreduce_cycles[iteration] = end_cycle - start_cycle;
        }
    }
    
    ShmemDeviceBarrierAll();
}

// ============================================================================
// Host-side Test Runner
// ============================================================================
template<typename T, 
         uint32_t gemmBaseM, uint32_t gemmBaseK, uint32_t gemmBaseN,
         int arRows, int arCols>
bool RunConcurrentKernel(
    int rank_id, 
    int n_ranks, 
    int n_devices, 
    int first_device_id,
    const ConcurrentTestConfig &config)
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

    // Initialize shmem
    ShmemEnv env;
    const char *ip = "tcp://127.0.0.1:8778";
    env.rank = rank_id;
    env.size = n_ranks;
    env.ipPort = ip;
    
    if (!ShmemInitFromEnv(env)) {
        return false;
    }

    // ================================================================
    // Allocate GEMM buffers
    // ================================================================
    size_t gemmASize = GEMM_BLOCK_NUM * gemmBaseM * gemmBaseK * sizeof(half);
    size_t gemmBSize = GEMM_BLOCK_NUM * gemmBaseK * gemmBaseN * sizeof(half);
    size_t gemmCSize = GEMM_BLOCK_NUM * gemmBaseM * gemmBaseN * sizeof(float);
    
    half *gemmAHost, *gemmBHost;
    float *gemmCHost;
    aclrtMallocHost((void **)(&gemmAHost), gemmASize);
    aclrtMallocHost((void **)(&gemmBHost), gemmBSize);
    aclrtMallocHost((void **)(&gemmCHost), gemmCSize);
    
    // Initialize GEMM inputs
    for (size_t i = 0; i < GEMM_BLOCK_NUM * gemmBaseM * gemmBaseK; ++i) {
        gemmAHost[i] = static_cast<half>(1.0f);
    }
    for (size_t i = 0; i < GEMM_BLOCK_NUM * gemmBaseK * gemmBaseN; ++i) {
        gemmBHost[i] = static_cast<half>(1.0f);
    }
    
    half *gemmADevice, *gemmBDevice;
    float *gemmCDevice;
    aclrtMalloc((void **)(&gemmADevice), gemmASize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)(&gemmBDevice), gemmBSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)(&gemmCDevice), gemmCSize, ACL_MEM_MALLOC_HUGE_FIRST);
    
    aclrtMemcpy(gemmADevice, gemmASize, gemmAHost, gemmASize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(gemmBDevice, gemmBSize, gemmBHost, gemmBSize, ACL_MEMCPY_HOST_TO_DEVICE);

    // ================================================================
    // Allocate AllReduce buffers
    // ================================================================
    size_t arCount = ALLREDUCE_BLOCK_NUM * arRows * arCols;
    size_t arSize = arCount * sizeof(T);
    
    T *arInputHost, *arOutputHost;
    aclrtMallocHost((void **)(&arInputHost), arSize);
    aclrtMallocHost((void **)(&arOutputHost), arSize);
    
    // Initialize AllReduce input
    for (size_t i = 0; i < arCount; ++i) {
        arInputHost[i] = static_cast<T>((rank_id + 1) * 100 + i);
    }
    
    T *arInputDevice, *arOutputDevice;
    aclrtMalloc((void **)(&arInputDevice), arSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)(&arOutputDevice), arSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemcpy(arInputDevice, arSize, arInputHost, arSize, ACL_MEMCPY_HOST_TO_DEVICE);

    // Allocate shmem buffer for AllReduce
    uint64_t fftsConfig = util_get_ffts_config();
    void *shmemBuffer = ShmemMalloc(4 * arSize);
    
    if (shmemBuffer == nullptr) {
        std::cerr << "[ERROR] ShmemMalloc failed!" << std::endl;
        ShmemFinalize();
        return false;
    }

    // ================================================================
    // Allocate performance counters
    // ================================================================
    int64_t *gemmCyclesHost, *arCyclesHost;
    aclrtMallocHost(reinterpret_cast<void**>(&gemmCyclesHost), config.measure_iters * sizeof(int64_t));
    aclrtMallocHost(reinterpret_cast<void**>(&arCyclesHost), config.measure_iters * sizeof(int64_t));
    
    int64_t *gemmCyclesDevice, *arCyclesDevice;
    aclrtMalloc(reinterpret_cast<void**>(&gemmCyclesDevice), config.measure_iters * sizeof(int64_t), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(reinterpret_cast<void**>(&arCyclesDevice), config.measure_iters * sizeof(int64_t), ACL_MEM_MALLOC_HUGE_FIRST);

    // ================================================================
    // Warmup
    // ================================================================
    if (rank_id == 0 && config.verbose) {
        std::cout << "\n[INFO] Starting warmup (" << config.warmup_iters << " iterations)..." << std::endl;
    }

    for (int i = 0; i < config.warmup_iters; ++i) {
        MatmulAllReduceConcurrentKernel<T, gemmBaseM, gemmBaseK, gemmBaseN, arRows, arCols>
            <<<TOTAL_BLOCK_NUM, nullptr, stream>>>(
                gemmADevice, gemmBDevice, gemmCDevice,
                arInputDevice, arOutputDevice, (T*)shmemBuffer,
                TOTAL_BLOCK_NUM, GEMM_BLOCK_NUM, n_ranks, fftsConfig,
                nullptr, nullptr, 0);
        aclrtSynchronizeStream(stream);
    }

    ShmemBarrierAll();

    // ================================================================
    // Measurement
    // ================================================================
    if (rank_id == 0 && config.verbose) {
        std::cout << "[INFO] Starting measurement (" << config.measure_iters << " iterations)..." << std::endl;
    }
    
    aclrtMemset(gemmCyclesDevice, config.measure_iters * sizeof(int64_t), 0, config.measure_iters * sizeof(int64_t));
    aclrtMemset(arCyclesDevice, config.measure_iters * sizeof(int64_t), 0, config.measure_iters * sizeof(int64_t));
    
    ShmemBarrierAll();

    auto wall_start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < config.measure_iters; ++i) {
        // Reset AllReduce output
        aclrtMemcpy(arInputDevice, arSize, arInputHost, arSize, ACL_MEMCPY_HOST_TO_DEVICE);
        
        MatmulAllReduceConcurrentKernel<T, gemmBaseM, gemmBaseK, gemmBaseN, arRows, arCols>
            <<<TOTAL_BLOCK_NUM, nullptr, stream>>>(
                gemmADevice, gemmBDevice, gemmCDevice,
                arInputDevice, arOutputDevice, (T*)shmemBuffer,
                TOTAL_BLOCK_NUM, GEMM_BLOCK_NUM, n_ranks, fftsConfig,
                gemmCyclesDevice, arCyclesDevice, i);
        aclrtSynchronizeStream(stream);
    }
    
    auto wall_end = std::chrono::high_resolution_clock::now();
    double wall_time_ms = std::chrono::duration<double, std::milli>(wall_end - wall_start).count();

    ShmemBarrierAll();

    // ================================================================
    // Copy results back
    // ================================================================
    aclrtMemcpy(gemmCyclesHost, config.measure_iters * sizeof(int64_t),
                gemmCyclesDevice, config.measure_iters * sizeof(int64_t),
                ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(arCyclesHost, config.measure_iters * sizeof(int64_t),
                arCyclesDevice, config.measure_iters * sizeof(int64_t),
                ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(gemmCHost, gemmCSize, gemmCDevice, gemmCSize, ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(arOutputHost, arSize, arOutputDevice, arSize, ACL_MEMCPY_DEVICE_TO_HOST);

    // ================================================================
    // Verify and Report Results (only rank 0)
    // ================================================================
    if (rank_id == 0) {
        bool gemm_ok = true;
        bool allreduce_ok = true;
        
        // Verify GEMM result (C = A * B, where A and B are all 1s)
        // Expected: each element should be gemmBaseK (sum of 1*1 for K times)
        float expected_gemm = static_cast<float>(gemmBaseK);
        for (size_t i = 0; i < GEMM_BLOCK_NUM * gemmBaseM * gemmBaseN && gemm_ok; ++i) {
            if (std::abs(gemmCHost[i] - expected_gemm) > 0.1f) {
                std::cerr << "[ERROR] GEMM verification failed at index " << i 
                          << ": expected " << expected_gemm << ", got " << gemmCHost[i] << std::endl;
                gemm_ok = false;
            }
        }
        
        // Verify AllReduce result
        for (size_t i = 0; i < arCount && allreduce_ok; ++i) {
            T expected = 0;
            for (int r = 0; r < n_ranks; ++r) {
                expected += static_cast<T>((r + 1) * 100 + i);
            }
            if (arOutputHost[i] != expected) {
                std::cerr << "[ERROR] AllReduce verification failed at index " << i 
                          << ": expected " << (float)expected << ", got " << (float)arOutputHost[i] << std::endl;
                allreduce_ok = false;
            }
        }

        if (gemm_ok && allreduce_ok) {
            // Calculate and display performance statistics
            constexpr double CYCLES_PER_US = 50.0;
            std::vector<double> gemm_latencies, ar_latencies;
            
            for (int i = 0; i < config.measure_iters; ++i) {
                gemm_latencies.push_back(static_cast<double>(gemmCyclesHost[i]) / CYCLES_PER_US);
                ar_latencies.push_back(static_cast<double>(arCyclesHost[i]) / CYCLES_PER_US);
            }
            
            ConcurrentPerfStats stats = CalculateConcurrentStats(gemm_latencies, ar_latencies);
            
            std::cout << "\n================================================================" << std::endl;
            std::cout << "  MATMUL + ALLREDUCE Concurrent Demo Results" << std::endl;
            std::cout << "================================================================" << std::endl;
            std::cout << "  Configuration:" << std::endl;
            std::cout << "    - Total cores:      " << TOTAL_BLOCK_NUM << std::endl;
            std::cout << "    - GEMM cores:       " << GEMM_BLOCK_NUM << std::endl;
            std::cout << "    - AllReduce cores:  " << ALLREDUCE_BLOCK_NUM << std::endl;
            std::cout << "    - Ranks:            " << n_ranks << std::endl;
            std::cout << std::endl;
            std::cout << "  GEMM Performance:" << std::endl;
            std::cout << "    - Matrix size:      " << gemmBaseM << "x" << gemmBaseK << " * " 
                      << gemmBaseK << "x" << gemmBaseN << std::endl;
            std::cout << "    - Min latency:      " << std::fixed << std::setprecision(2) 
                      << stats.gemm_min_us << " us" << std::endl;
            std::cout << "    - Avg latency:      " << stats.gemm_avg_us << " us" << std::endl;
            std::cout << "    - Max latency:      " << stats.gemm_max_us << " us" << std::endl;
            std::cout << std::endl;
            std::cout << "  AllReduce Performance:" << std::endl;
            std::cout << "    - Data size:        " << arSize << " bytes (" 
                      << (arSize / 1024.0) << " KB)" << std::endl;
            std::cout << "    - Min latency:      " << stats.allreduce_min_us << " us" << std::endl;
            std::cout << "    - Avg latency:      " << stats.allreduce_avg_us << " us" << std::endl;
            std::cout << "    - Max latency:      " << stats.allreduce_max_us << " us" << std::endl;
            std::cout << std::endl;
            std::cout << "  Concurrent Analysis:" << std::endl;
            std::cout << "    - Effective time:   " << stats.total_avg_us << " us" << std::endl;
            std::cout << "    - Sequential time:  " << (stats.gemm_avg_us + stats.allreduce_avg_us) << " us" << std::endl;
            std::cout << "    - Concurrent savings:  " << stats.concurrent_efficiency << "%" << std::endl;
            std::cout << std::endl;
            std::cout << "  Wall Clock Time:      " << wall_time_ms << " ms (total)" << std::endl;
            std::cout << "  Verification:         PASSED" << std::endl;
            std::cout << "================================================================\n" << std::endl;
        } else {
            status = 1;
        }
    }

    ShmemBarrierAll();

    // ================================================================
    // Cleanup
    // ================================================================
    aclrtFreeHost(gemmAHost);
    aclrtFreeHost(gemmBHost);
    aclrtFreeHost(gemmCHost);
    aclrtFree(gemmADevice);
    aclrtFree(gemmBDevice);
    aclrtFree(gemmCDevice);
    
    aclrtFreeHost(arInputHost);
    aclrtFreeHost(arOutputHost);
    aclrtFree(arInputDevice);
    aclrtFree(arOutputDevice);
    
    aclrtFreeHost(gemmCyclesHost);
    aclrtFreeHost(arCyclesHost);
    aclrtFree(gemmCyclesDevice);
    aclrtFree(arCyclesDevice);
    
    ShmemFree(shmemBuffer);
    ShmemFinalize();

    status |= aclrtDestroyStream(stream);
    status |= aclrtResetDevice(device_id);
    status |= aclFinalize();

    return (status == 0);
}

// ============================================================================
// Multi-process Launcher
// ============================================================================
template <typename T, 
          uint32_t gemmBaseM, uint32_t gemmBaseK, uint32_t gemmBaseN,
          int arRows, int arCols>
bool RunConcurrent(
    int n_ranks, 
    int n_devices, 
    int first_rank_id, 
    int first_device_id,
    const ConcurrentTestConfig &config)
{
    std::vector<pid_t> pids;
    
    for (int r = 0; r < n_ranks; ++r) {
        pid_t pid = fork();
        if (pid == 0) { // child
            const bool ok = RunConcurrentKernel<T, gemmBaseM, gemmBaseK, gemmBaseN, arRows, arCols>(
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
// ============================================================================

// Default configuration: 
// - GEMM: 128x64x128 (moderate size)
// - AllReduce: 16x256 float elements (16KB per core)
template bool RunConcurrent<float, 128, 64, 128, 16, 256>(
    int n_ranks, int n_devices, int first_rank_id, int first_device_id,
    const ConcurrentTestConfig &config);

// Wrapper function
bool RunMatmulAllReduceConcurrent(
    int n_ranks, 
    int n_devices, 
    int first_rank_id, 
    int first_device_id,
    const ConcurrentTestConfig &config)
{
    return RunConcurrent<float, 128, 64, 128, 16, 256>(
        n_ranks, n_devices, first_rank_id, first_device_id, config);
}
