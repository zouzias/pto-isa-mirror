/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// ============================================================================
// Communication Kernel (Vec Arch) for GEMM + AllReduce Demo
//
// Optimization: Replaced TPUT<AtomicAdd> with plain TPUT + local reduce.
// This avoids atomic operation overhead by writing to separate per-rank
// receive buffers, then performing a local TADD reduction.
//
// Phase 1 - Poll & Send (per tile):
//   - Poll compute kernel's queue using TTEST/TWAIT
//   - Once ready, plain TPUT tile data to recv_buffers[my_rank] on ALL ranks
//   - No atomic needed: each rank writes to its own dedicated slot
//
// Phase 2 - Barrier:
//   - ShmemDeviceQuiet() + ShmemDeviceBarrierAll() ensure all data visible
//
// Phase 3 - Local Reduce:
//   - Each comm block reduces its assigned tiles:
//     reduced_output[tile] = sum(recv_buffers[r][tile] for r in 0..nranks-1)
//   - Uses TLOAD + TADD + TSTORE on Vec tiles
//
// Also contains all host-side logic: per-rank initialization, kernel launch,
// synchronization, verification, and multi-process (fork) launcher.
// ============================================================================

#include <cstddef>
#include <cstdint>

#include <sys/wait.h>
#include <signal.h>
#include <unistd.h>
#include <vector>
#include <string>
#include <iostream>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <algorithm>

#include "pto/comm/pto_comm_inst.hpp"
#include "pto/common/pto_tile.hpp"
#include "common.hpp"
#include "ready_queue.hpp"

// Save and undefine DT_UNDEFINED macro before including test_common.h
// to avoid conflict with enum DT_UNDEFINED in test_common.h
#ifdef DT_UNDEFINED
#define DT_UNDEFINED_SAVED DT_UNDEFINED
#undef DT_UNDEFINED
#endif
#include "test_common.h"
// Restore DT_UNDEFINED macro if it was previously defined
#ifdef DT_UNDEFINED_SAVED
#define DT_UNDEFINED DT_UNDEFINED_SAVED
#undef DT_UNDEFINED_SAVED
#endif

#include <pto/pto-inst.hpp>

// ============================================================================
// Global GEMM parameters (must match compute kernel)
// ============================================================================
#ifndef CONFIG_G_M
#define CONFIG_G_M 16384
#endif
#ifndef CONFIG_G_K
#define CONFIG_G_K 4096
#endif
#ifndef CONFIG_G_N
#define CONFIG_G_N 4096
#endif

constexpr uint32_t G_M = CONFIG_G_M;
constexpr uint32_t G_K = CONFIG_G_K;
constexpr uint32_t G_N = CONFIG_G_N;
constexpr uint32_t G_BASE_M = 128;
constexpr uint32_t G_BASE_K = 64;
constexpr uint32_t G_BASE_N = 256;
constexpr uint32_t G_M_TILES = G_M / G_BASE_M;
constexpr uint32_t G_N_TILES = G_N / G_BASE_N;
constexpr uint32_t G_NUM_TILES = G_M_TILES * G_N_TILES;

#ifndef CONFIG_SIZE_NAME
#define CONFIG_SIZE_NAME "default"
#endif
constexpr const char* SIZE_NAME = CONFIG_SIZE_NAME;

// ============================================================================
// Performance test configuration
// ============================================================================
constexpr int WARMUP_ITERS = 5;
constexpr int MEASURE_ITERS = 20;

// ============================================================================
// Multi-block configuration
// ============================================================================
#ifndef CONFIG_COMPUTE_BLOCK_NUM
#define CONFIG_COMPUTE_BLOCK_NUM 32
#endif
#ifndef CONFIG_COMM_BLOCK_NUM
#define CONFIG_COMM_BLOCK_NUM 40
#endif
constexpr int COMPUTE_BLOCK_NUM = CONFIG_COMPUTE_BLOCK_NUM;
constexpr int COMM_BLOCK_NUM = CONFIG_COMM_BLOCK_NUM;

// ============================================================================
// GemmCommImpl: Core communication logic for AllReduce
//
// AllReduce Implementation Strategy (TPUT + Temporary Buffer):
// Each rank computes C_rank = A_rank * B with DIFFERENT input data.
// The final result is C_final = sum(C_rank for all ranks).
//
// Memory layout on symmetric heap (per rank):
//   shmem_output:   Local computation result C_rank [M, N]
//   recv_buffers:   Temporary buffers to receive data from other ranks
//                   Layout: recv_buffers[r * M * N] = buffer for rank r's data
//   reduced_output: Final AllReduce result [M, N]
//
// Algorithm:
// Phase 1: Wait for local computation to complete (poll queues)
// Phase 2: TPUT local data to all remote ranks' recv_buffers
// Phase 3: Barrier to ensure all data transfers complete
// Phase 4: Local reduce: sum local data + all received data
//
// Parameters:
//   shmem_output:   Symmetric heap buffer for C_rank [M, N] (float)
//   recv_buffers:   Symmetric heap buffer for receiving data from other ranks
//                   Size: nranks * M * N * sizeof(float)
//   reduced_output: Buffer for final reduced result [M, N]
//   queue_set:      MultiBlockQueueSet* - per-block queues from compute kernel
//   rank:           Current rank
//   nranks:         Total number of ranks
//   num_compute_blocks: Number of compute blocks
//   comm_block_idx: Index of this comm block
//   num_comm_blocks: Number of comm blocks
// ============================================================================
AICORE inline void GemmCommImpl(
    __gm__ float *shmem_output,
    __gm__ float *recv_buffers,
    __gm__ float *reduced_output,
    __gm__ MultiBlockQueueSet *queue_set,
    int rank,
    int nranks,
    int num_compute_blocks,
    int comm_block_idx,
    int num_comm_blocks)
{
    int my_rank = shmem_my_pe();
    const uint64_t output_size = (uint64_t)G_M * G_N;  // Elements per output matrix

    // recv_buffers layout: [nranks × output_size] on symmetric heap
    // recv_buffers[r * output_size ... (r+1) * output_size - 1] = slot for rank r's data
    // Each rank writes its data to slot[my_rank] on all remote ranks via plain TPUT (no atomic).
    // After barrier, each rank locally reduces all slots → reduced_output.

    // GlobalTensor types
    using ShapeDyn  = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using Global    = pto::GlobalTensor<float, ShapeDyn, StrideDyn, pto::Layout::ND>;

    // UB staging tiles for TPUT and reduction
    using TileData = pto::Tile<pto::TileType::Vec, float, G_BASE_M, G_BASE_N,
                               pto::BLayout::RowMajor, -1, -1>;

    constexpr size_t tileUBBytes = ((G_BASE_M * G_BASE_N * sizeof(float) + 1023) / 1024) * 1024;
    
    // Ping-pong tiles for TPUT
    TileData pingTile(G_BASE_M, G_BASE_N);
    TileData pongTile(G_BASE_M, G_BASE_N);
    TASSIGN(pingTile, 0x0);
    TASSIGN(pongTile, tileUBBytes);
    
    // Note: pingTile and pongTile are reused for the reduce phase after TPUT is done.
    // During TPUT: used as staging tiles for ping-pong double-buffered transfer.
    // During reduce: pingTile = accumulator, pongTile = recv tile for TADD.

    // Shape/stride for a tile region within the M x N output matrix
    ShapeDyn tileShape(1, 1, 1, G_BASE_M, G_BASE_N);
    StrideDyn tileStride(G_BASE_M * G_N, G_BASE_M * G_N, G_BASE_M * G_N, G_N, 1);

    volatile __gm__ MultiBlockQueueSet *qset = (volatile __gm__ MultiBlockQueueSet *)queue_set;
    
    // Local head positions for queues (each comm block tracks its own progress)
    int32_t heads[MAX_COMPUTE_BLOCKS];
    for (int b = 0; b < MAX_COMPUTE_BLOCKS; b++) {
        heads[b] = 0;
    }

    // Track which tiles have been processed by this comm block
    int32_t tiles_sent = 0;
    
    // Total tiles to be processed by ALL comm blocks
    const int total_tiles = G_NUM_TILES;

    // ================================================================
    // Phase 1 & 2: Poll for ready tiles and TPUT to remote ranks
    // 
    // Queue assignment strategy (exclusive assignment):
    //   - Each compute block queue is assigned to EXACTLY ONE comm block
    //   - This avoids race conditions between comm blocks
    //   - When num_comm_blocks > num_compute_blocks, extra comm blocks are idle
    //   - When num_comm_blocks < num_compute_blocks, some comm blocks handle multiple queues
    //
    // Assignment formula:
    //   - Comm block c handles queues where (queue_idx % num_comm_blocks == c)
    //   - This ensures even distribution and no overlap
    // ================================================================
    
    // Calculate which queues this comm block is responsible for
    // Using modulo assignment: queue q is handled by comm block (q % num_comm_blocks)
    int my_queue_indices[MAX_COMPUTE_BLOCKS];
    int my_queue_count = 0;
    for (int q = 0; q < num_compute_blocks; q++) {
        if (q % num_comm_blocks == comm_block_idx) {
            my_queue_indices[my_queue_count++] = q;
        }
    }
    
    // Note: Blocks with no queues assigned (num_comm_blocks > num_compute_blocks)
    // skip the TPUT loop naturally since my_expected_tiles = 0.
    // They still participate in the barrier and reduce phase.
    
    // Calculate how many tiles our assigned compute blocks will produce
    const int tiles_per_compute_block = (total_tiles + num_compute_blocks - 1) / num_compute_blocks;
    
    // Track max tiles per queue and calculate total expected tiles
    int32_t queue_max_tiles[MAX_COMPUTE_BLOCKS];
    int my_expected_tiles = 0;
    for (int i = 0; i < my_queue_count; i++) {
        int q = my_queue_indices[i];
        int block_start_tile = q * tiles_per_compute_block;
        int block_end_tile = (q + 1) * tiles_per_compute_block;
        if (block_end_tile > total_tiles) block_end_tile = total_tiles;
        int block_tiles = block_end_tile - block_start_tile;
        if (block_tiles < 0) block_tiles = 0;
        queue_max_tiles[q] = block_tiles;
        my_expected_tiles += block_tiles;
    }
    
    // Round-robin start position for polling within our assigned queues
    int next_queue_offset = 0;
    
    while (tiles_sent < my_expected_tiles) {
        // Poll only our assigned queues for ready tiles
        int32_t tile_idx = -1;
        int32_t found_queue_local_idx = -1;
        for (int i = 0; i < my_queue_count; i++) {
            int local_idx = (next_queue_offset + i) % my_queue_count;
            int32_t q = my_queue_indices[local_idx];
            
            // Skip exhausted queues
            if (heads[q] >= queue_max_tiles[q]) {
                continue;
            }
            
            volatile __gm__ PerBlockQueue* pq = GetMyBlockQueue(qset, q);
            int32_t tile = PerBlockQueueTryDequeue(pq, heads[q]);
            
            if (tile >= 0) {
                heads[q]++;
                next_queue_offset = (local_idx + 1) % my_queue_count;
                tile_idx = tile;
                found_queue_local_idx = local_idx;
                break;
            }
        }
        
        if (tile_idx >= 0) {
            // Tile is ready - plain TPUT to all ranks' recv_buffers (including self)
            // Each rank writes to slot[my_rank] on each destination rank.
            // No atomic needed: different ranks write to different slots.
            uint32_t mi = tile_idx / G_N_TILES;
            uint32_t ni = tile_idx % G_N_TILES;
            uint64_t tile_offset = (uint64_t)(mi * G_BASE_M) * G_N + ni * G_BASE_N;

            // Source: local shmem_output (our GEMM result)
            Global srcG(shmem_output + tile_offset, tileShape, tileStride);

            // Plain TPUT to REMOTE ranks' recv_buffers at slot[my_rank]
            // Skip self (r == my_rank): local data is read directly from shmem_output
            // during reduce phase, avoiding unnecessary self-TPUT overhead.
            for (int r = 0; r < nranks; ++r) {
                if (r == my_rank) continue;  // Skip self - read from shmem_output instead
                
                // Destination: rank r's recv_buffers at slot my_rank
                __gm__ float *dst_ptr = ShmemPtr(recv_buffers, r)
                                      + (uint64_t)my_rank * output_size + tile_offset;
                Global dstG(dst_ptr, tileShape, tileStride);
                
                // Plain TPUT (AtomicNone) - lower overhead than AtomicAdd
                pto::comm::TPUT(dstG, srcG, pingTile, pongTile);
            }
            
            tiles_sent++;
        } else {
            // No tile available from any of our queues, wait on a non-exhausted queue
            // Find a queue that still has pending tiles
            int32_t wait_queue = -1;
            for (int i = 0; i < my_queue_count; i++) {
                int local_idx = (next_queue_offset + i) % my_queue_count;
                int32_t q = my_queue_indices[local_idx];
                if (heads[q] < queue_max_tiles[q]) {
                    wait_queue = q;
                    break;
                }
            }
            
            if (wait_queue >= 0) {
                // TWAIT optimization: directly use TWAIT for blocking wait
                // TWAIT will return immediately if data is already available (hardware-level check)
                // This reduces the overhead of TTEST + TWAIT pattern
                volatile __gm__ PerBlockQueue* pq = GetMyBlockQueue(qset, wait_queue);
                pto::comm::Signal sig(const_cast<__gm__ int32_t*>(&pq->count));
                pto::comm::TWAIT(sig, heads[wait_queue] + 1, pto::comm::WaitCmp::GE);
            }
            // If all our queues are exhausted, loop will exit (tiles_sent == my_expected_tiles)
        }
    }

    // Ensure all TPUT writes are complete
    ShmemDeviceQuiet();
    
    // ================================================================
    // Phase 3: Global barrier - ensure all ranks have sent their data
    // ================================================================
    ShmemDeviceBarrierAll();

    // ================================================================
    // Phase 4: Local reduce (merged into comm kernel)
    //
    // After the device barrier, all TPUT data from all ranks is visible.
    // Each comm block reduces its assigned tile range:
    //   reduced_output[tile] = shmem_output[tile] + sum(recv_buffers[r][tile])
    //
    // Implementation: TPUT (plain copy) for local data, then
    // TPUT<AtomicAdd> for remote ranks' data.
    // Uses MTE2/MTE3 pipelines only (no V-pipeline event counter issues).
    // No contention: each block processes disjoint tile ranges.
    // ================================================================
    {
        int reduce_tiles_per_block = (total_tiles + num_comm_blocks - 1) / num_comm_blocks;
        int rt_start = comm_block_idx * reduce_tiles_per_block;
        int rt_end = (comm_block_idx + 1) * reduce_tiles_per_block;
        if (rt_end > total_tiles) rt_end = total_tiles;

        // Phase 4a: Copy local GEMM result → reduced_output (plain TPUT, non-atomic)
        for (int t = rt_start; t < rt_end; t++) {
            uint32_t mi = t / G_N_TILES;
            uint32_t ni = t % G_N_TILES;
            uint64_t toff = (uint64_t)(mi * G_BASE_M) * G_N + ni * G_BASE_N;

            Global srcG(shmem_output + toff, tileShape, tileStride);
            Global dstG(reduced_output + toff, tileShape, tileStride);
            pto::comm::TPUT(dstG, srcG, pingTile, pongTile);
        }

        // Phase 4b: Accumulate remote ranks' data via TPUT<AtomicAdd>
        // LOCAL atomic add: same-chip DMA, no network overhead.
        for (int r = 0; r < nranks; ++r) {
            if (r == my_rank) continue;  // Already copied in Phase 4a

            for (int t = rt_start; t < rt_end; t++) {
                uint32_t mi = t / G_N_TILES;
                uint32_t ni = t % G_N_TILES;
                uint64_t toff = (uint64_t)(mi * G_BASE_M) * G_N + ni * G_BASE_N;

                Global srcG(recv_buffers + (uint64_t)r * output_size + toff,
                            tileShape, tileStride);
                Global dstG(reduced_output + toff, tileShape, tileStride);
                pto::comm::TPUT<pto::AtomicType::AtomicAdd>(dstG, srcG, pingTile, pongTile);
            }
        }
    }
    pipe_barrier(PIPE_ALL);
}

// ============================================================================
// Reduce kernel: sum all ranks' data from recv_buffers → reduced_output
//
// Launched as a SEPARATE kernel after host-side ShmemBarrierAll() ensures
// all TPUT data is globally visible.
//
// Algorithm:
//   Phase 1: reduced_output[tile] = shmem_output[tile]  (copy local rank data)
//   Phase 2: For each remote rank r:
//              reduced_output[tile] += recv_buffers[r][tile]  (atomic add)
//
// Implementation: Uses TPUT (MTE2→MTE3 pipeline) instead of TADD (V pipeline).
//
// Why not TADD? The V-pipeline has a hard event counter limit (~16 events
// per kernel invocation across all set_flag/wait_flag calls involving PIPE_V).
// With 8 ranks, a single tile needs 16 V-pipeline events (1 + 7×2 + 1),
// leaving zero headroom for multi-tile processing. pipe_barrier(PIPE_ALL)
// does NOT reset V-pipeline event counters.
//
// TPUT<AtomicAdd> uses MTE3's DMA-level atomic read-modify-write, which:
//   - Only involves MTE2/MTE3 pipelines (no V-pipeline events)
//   - MTE2↔MTE3 counters can handle 2048+ tiles per kernel
//   - LOCAL atomic add has no network overhead (same-chip DMA)
//   - No contention: each block operates on its own tile range
// ============================================================================
AICORE inline void ReduceImpl(
    __gm__ float *shmem_output,
    __gm__ float *recv_buffers,
    __gm__ float *reduced_output,
    int rank,
    int nranks,
    int block_idx,
    int num_blocks)
{
    int my_rank = rank;
    const uint64_t output_size = (uint64_t)G_M * G_N;
    constexpr int total_tiles = G_NUM_TILES;

    // Same type aliases as GemmCommImpl
    using ShapeDyn  = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using Global    = pto::GlobalTensor<float, ShapeDyn, StrideDyn, pto::Layout::ND>;

    using TileData = pto::Tile<pto::TileType::Vec, float, G_BASE_M, G_BASE_N,
                               pto::BLayout::RowMajor, -1, -1>;
    constexpr size_t tileUBBytes = ((G_BASE_M * G_BASE_N * sizeof(float) + 1023) / 1024) * 1024;

    // Ping-pong tiles for TPUT double-buffered transfer
    TileData pingTile(G_BASE_M, G_BASE_N);
    TileData pongTile(G_BASE_M, G_BASE_N);
    TASSIGN(pingTile, 0x0);
    TASSIGN(pongTile, tileUBBytes);

    ShapeDyn tileShape(1, 1, 1, G_BASE_M, G_BASE_N);
    StrideDyn tileStride(G_BASE_M * G_N, G_BASE_M * G_N, G_BASE_M * G_N, G_N, 1);

    // Distribute tiles evenly across blocks
    int tiles_per_block = (total_tiles + num_blocks - 1) / num_blocks;
    int t_start = block_idx * tiles_per_block;
    int t_end = (block_idx + 1) * tiles_per_block;
    if (t_end > total_tiles) t_end = total_tiles;

    // ================================================================
    // Phase 1: Copy local GEMM result → reduced_output (plain TPUT)
    //   reduced_output[tile] = shmem_output[tile]
    // Uses MTE2 (TLOAD) + MTE3 (TSTORE) only, no V pipeline.
    // ================================================================
    for (int t = t_start; t < t_end; t++) {
        uint32_t mi = t / G_N_TILES;
        uint32_t ni = t % G_N_TILES;
        uint64_t toff = (uint64_t)(mi * G_BASE_M) * G_N + ni * G_BASE_N;

        Global srcG(shmem_output + toff, tileShape, tileStride);
        Global dstG(reduced_output + toff, tileShape, tileStride);
        pto::comm::TPUT(dstG, srcG, pingTile, pongTile);
    }

    // ================================================================
    // Phase 2: Accumulate remote ranks' data → reduced_output
    //   For each remote rank r:
    //     reduced_output[tile] += recv_buffers[r][tile]
    //
    // Uses TPUT<AtomicAdd> which does DMA-level read-modify-write on MTE3.
    // LOCAL atomic add: no network, just same-chip DMA with atomic semantics.
    // No contention between blocks (each block processes disjoint tile ranges).
    // ================================================================
    for (int r = 0; r < nranks; ++r) {
        if (r == my_rank) continue;  // Skip self (already copied in Phase 1)

        for (int t = t_start; t < t_end; t++) {
            uint32_t mi = t / G_N_TILES;
            uint32_t ni = t % G_N_TILES;
            uint64_t toff = (uint64_t)(mi * G_BASE_M) * G_N + ni * G_BASE_N;

            Global srcG(recv_buffers + (uint64_t)r * output_size + toff,
                        tileShape, tileStride);
            Global dstG(reduced_output + toff, tileShape, tileStride);
            pto::comm::TPUT<pto::AtomicType::AtomicAdd>(dstG, srcG, pingTile, pongTile);
        }
    }

    pipe_barrier(PIPE_ALL);
}

// ============================================================================
// Entry point kernels
// ============================================================================
__global__ AICORE void ReduceKernel(
    __gm__ uint8_t *shmem_output,
    __gm__ uint8_t *recv_buffers,
    __gm__ uint8_t *reduced_output,
    int rank,
    int nranks,
    int num_blocks)
{
    ReduceImpl(
        reinterpret_cast<__gm__ float *>(shmem_output),
        reinterpret_cast<__gm__ float *>(recv_buffers),
        reinterpret_cast<__gm__ float *>(reduced_output),
        rank, nranks, get_block_idx(), num_blocks);
}

__global__ AICORE void GemmCommKernel(
    __gm__ uint8_t *shmem_output,
    __gm__ uint8_t *recv_buffers,
    __gm__ uint8_t *reduced_output,
    __gm__ uint8_t *queue_set,
    int rank,
    int nranks,
    int num_compute_blocks,
    int num_comm_blocks)
{
    const int comm_block_idx = get_block_idx();
    GemmCommImpl(
        reinterpret_cast<__gm__ float *>(shmem_output),
        reinterpret_cast<__gm__ float *>(recv_buffers),
        reinterpret_cast<__gm__ float *>(reduced_output),
        reinterpret_cast<__gm__ MultiBlockQueueSet *>(queue_set),
        rank,
        nranks,
        num_compute_blocks,
        comm_block_idx,
        num_comm_blocks);
}

// ============================================================================
// Host-side launch functions
// ============================================================================
static void launchGemmComm(uint8_t *shmem_output, uint8_t *recv_buffers, uint8_t *reduced_output,
                           uint8_t *queue_set, int rank, int nranks, void *stream, int num_compute_blocks)
{
    GemmCommKernel<<<COMM_BLOCK_NUM, nullptr, stream>>>(
        shmem_output, recv_buffers, reduced_output, queue_set, rank, nranks, num_compute_blocks, COMM_BLOCK_NUM);
}

constexpr int REDUCE_BLOCK_NUM = COMM_BLOCK_NUM;  // Use same block count as comm

static void launchReduce(uint8_t *shmem_output, uint8_t *recv_buffers, uint8_t *reduced_output,
                         int rank, int nranks, void *stream)
{
    ReduceKernel<<<REDUCE_BLOCK_NUM, nullptr, stream>>>(
        shmem_output, recv_buffers, reduced_output, rank, nranks, REDUCE_BLOCK_NUM);
}

// ============================================================================
// Extern: compute kernel launch function (from gemm_compute_kernel.cpp)
// ============================================================================
extern void launchGemmCompute(uint8_t *shmem_output, uint8_t *src0, uint8_t *src1,
                              uint8_t *queue_set, int rank, void *stream, int block_num, uint32_t k_per_rank);

// ============================================================================
// Host-side per-rank logic
// ============================================================================
static bool RunGemmAllReducePerRank(int rank_id, int n_ranks, int device_id)
{
    // Initialize shmem TLS
    int32_t ret = ShmemSetConfStoreTls(false, nullptr, 0);
    if (ret != 0) {
        std::cerr << "[ERROR] Rank " << rank_id << ": Failed to init shmem tls\n";
        return false;
    }

    int status = 0;
    aclrtStream computeStream = nullptr;
    aclrtStream commStream = nullptr;

    status |= aclInit(nullptr);
    status |= aclrtSetDevice(device_id);
    status |= aclrtCreateStream(&computeStream);
    status |= aclrtCreateStream(&commStream);

    // Initialize shmem symmetric heap
    size_t outputSize = static_cast<size_t>(G_M) * G_N * sizeof(float);
    // Need space for:
    //   - shmem_output: local computation result [M, N]
    //   - recv_buffers: per-rank receive slots [nranks × M × N]
    //     Plain TPUT writes each rank's data to a dedicated slot.
    //     After barrier, local reduce sums all slots → reduced_output.
    //   - reduced_output: final result [M, N]
    size_t recvBuffersSize = static_cast<size_t>(n_ranks) * outputSize;
    size_t requiredHeapBytes = outputSize + recvBuffersSize + outputSize;
    // No 2x margin: actual allocations are exactly output + recv + reduced
    
    // Calculate heap size with upper limit to avoid shmem initialization failures
    size_t heapBytes = 64ULL * 1024 * 1024;
    const size_t maxHeapBytes = 3ULL * 1024 * 1024 * 1024;  // 3GB max (recv_buffers: nranks × outputSize)
    
    while (heapBytes < requiredHeapBytes && heapBytes < maxHeapBytes) {
        heapBytes *= 2;
    }
    
    // If still not enough, use maxHeapBytes (may fail if truly insufficient)
    if (heapBytes < requiredHeapBytes) {
        heapBytes = maxHeapBytes;
        if (rank_id == 0) {
            std::cerr << "[WARNING] Required heap size (" << (requiredHeapBytes / (1024 * 1024)) 
                      << " MB) exceeds maximum (" << (maxHeapBytes / (1024 * 1024)) 
                      << " MB). Using maximum, may fail if insufficient.\n";
        }
    }

    ShmemEnv env;
    env.rank = rank_id;
    env.size = n_ranks;
    env.ipPort = "tcp://127.0.0.1:8790";
    env.heapBytes = heapBytes;

    if (rank_id == 0) {
        std::cout << "[INFO] Shmem heap size: " << (heapBytes / (1024 * 1024)) << " MB"
                  << " (output: " << (outputSize / (1024 * 1024)) << " MB, "
                  << "recv_buffers: " << (recvBuffersSize / (1024 * 1024)) << " MB ["
                  << n_ranks << " ranks × " << (outputSize / (1024 * 1024)) << " MB])" << std::endl;
    }

    if (!ShmemInitFromEnv(env)) {
        std::cerr << "[ERROR] Rank " << rank_id << ": ShmemInitFromEnv failed!\n";
        return false;
    }

    // ------------------------------------------------------------------
    // Allocate memory
    // ------------------------------------------------------------------
    // shmem output: C_rank[M, N] on symmetric heap - each rank's partial GEMM result
    void *shmem_output = ShmemMalloc(outputSize);
    if (shmem_output == nullptr) {
        std::cerr << "[ERROR] Rank " << rank_id << ": ShmemMalloc for output failed!\n";
        return false;
    }
    aclrtMemset(shmem_output, outputSize, 0, outputSize);

    // recv_buffers: Per-rank receive slots [nranks × outputSize]
    // Each rank writes to slot[my_rank] via plain TPUT (no atomic).
    // After barrier, local reduce sums all nranks slots → reduced_output.
    void *recv_buffers = ShmemMalloc(recvBuffersSize);
    if (recv_buffers == nullptr) {
        std::cerr << "[ERROR] Rank " << rank_id << ": ShmemMalloc for recv_buffers failed!\n";
        return false;
    }
    aclrtMemset(recv_buffers, recvBuffersSize, 0, recvBuffersSize);

    // reduced_output: Final reduced result (on symmetric heap for consistency)
    void *reduced_output = ShmemMalloc(outputSize);
    if (reduced_output == nullptr) {
        std::cerr << "[ERROR] Rank " << rank_id << ": ShmemMalloc for reduced_output failed!\n";
        return false;
    }
    aclrtMemset(reduced_output, outputSize, 0, outputSize);

    // Data Parallel Mode: A and B matrices are split along K dimension
    // Each rank gets: A_part[M, K/r] and B_part[K/r, N]
    uint32_t k_per_rank = G_K / n_ranks;
    if (G_K % n_ranks != 0) {
        std::cerr << "[ERROR] Rank " << rank_id << ": K dimension (" << G_K 
                  << ") must be divisible by number of ranks (" << n_ranks << ")\n";
        return false;
    }
    
    // A_part[M, K/r] in half
    size_t aSize = G_M * k_per_rank * sizeof(uint16_t);
    void *src0_dev = nullptr;
    aclrtMalloc(&src0_dev, aSize, ACL_MEM_MALLOC_HUGE_FIRST);

    // B_part[K/r, N] in half
    size_t bSize = k_per_rank * G_N * sizeof(uint16_t);
    void *src1_dev = nullptr;
    aclrtMalloc(&src1_dev, bSize, ACL_MEM_MALLOC_HUGE_FIRST);

    // Multi-block queue set
    int tiles_per_block = (G_NUM_TILES + COMPUTE_BLOCK_NUM - 1) / COMPUTE_BLOCK_NUM;
    size_t queueSetSize = MultiBlockQueueSetSize(COMPUTE_BLOCK_NUM, tiles_per_block);
    void *queueSet_dev = nullptr;
    aclrtMalloc(&queueSet_dev, queueSetSize, ACL_MEM_MALLOC_HUGE_FIRST);
    
    // Initialize queue set
    MultiBlockQueueSet *queueSet_host = nullptr;
    aclrtMallocHost(reinterpret_cast<void **>(&queueSet_host), queueSetSize);
    MultiBlockQueueSetInit(queueSet_host, COMPUTE_BLOCK_NUM, G_NUM_TILES);
    aclrtMemcpy(queueSet_dev, queueSetSize, queueSet_host, queueSetSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtFreeHost(queueSet_host);

    // ------------------------------------------------------------------
    // Initialize input data (Data Parallel Mode)
    // Each rank loads A_part[M, K/r] and B_part[K/r, N]
    // This is data parallel: C_final = sum(A_part_i * B_part_i for all ranks)
    // ------------------------------------------------------------------
    uint16_t *a_host = nullptr, *b_host = nullptr;
    aclrtMallocHost(reinterpret_cast<void **>(&a_host), aSize);
    aclrtMallocHost(reinterpret_cast<void **>(&b_host), bSize);

    // Read input files - each rank has its own A_part and B_part matrices
    std::string input_dir = "../input";
    std::string a_file = input_dir + "/x1_gm_rank" + std::to_string(rank_id) + ".bin";
    std::string b_file = input_dir + "/x2_gm_rank" + std::to_string(rank_id) + ".bin";
    
    if (rank_id == 0) {
        std::cout << "[INFO] Data Parallel mode: A and B matrices split along K dimension" << std::endl;
        std::cout << "[INFO] Each rank: A_part[" << G_M << "×" << k_per_rank 
                  << "], B_part[" << k_per_rank << "×" << G_N << "]" << std::endl;
    }
    std::cout << "[INFO] Rank " << rank_id << ": Loading A_part from " << a_file << std::endl;
    std::cout << "[INFO] Rank " << rank_id << ": Loading B_part from " << b_file << std::endl;
    
    size_t a_file_size = 0, b_file_size = 0;
    if (!PtoTestCommon::ReadFile(a_file, a_file_size, a_host, aSize)) {
        std::cerr << "[ERROR] Rank " << rank_id << ": Failed to read " << a_file << std::endl;
        return false;
    }
    if (!PtoTestCommon::ReadFile(b_file, b_file_size, b_host, bSize)) {
        std::cerr << "[ERROR] Rank " << rank_id << ": Failed to read " << b_file << std::endl;
        return false;
    }

    aclrtMemcpy(src0_dev, aSize, a_host, aSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src1_dev, bSize, b_host, bSize, ACL_MEMCPY_HOST_TO_DEVICE);

    aclrtFreeHost(a_host);
    aclrtFreeHost(b_host);

    // ------------------------------------------------------------------
    // Synchronize all ranks before launching kernels
    // ------------------------------------------------------------------
    ShmemBarrierAll();

    // ------------------------------------------------------------------
    // Helper lambda to reset state
    // Optimization: Pre-allocate host buffer for queue set to avoid
    // repeated allocation/deallocation in performance measurement loop
    // ------------------------------------------------------------------
    MultiBlockQueueSet *queueSet_reset_host = nullptr;
    aclrtMallocHost(reinterpret_cast<void **>(&queueSet_reset_host), queueSetSize);
    
    auto resetState = [&]() {
        // Reset queue set (reuse pre-allocated host buffer)
        MultiBlockQueueSetInit(queueSet_reset_host, COMPUTE_BLOCK_NUM, G_NUM_TILES);
        aclrtMemcpy(queueSet_dev, queueSetSize, queueSet_reset_host, queueSetSize, ACL_MEMCPY_HOST_TO_DEVICE);
        
        // Reset shmem_output (partial results)
        aclrtMemset(shmem_output, outputSize, 0, outputSize);
        
        // Note: recv_buffers does NOT need zeroing between iterations.
        // TPUT fully overwrites all slots during the communication phase.
        // Skipping the nranks×outputSize memset saves significant time.
        
        // Reset reduced_output (final result)
        aclrtMemset(reduced_output, outputSize, 0, outputSize);
    };

    // ------------------------------------------------------------------
    // Warmup iterations
    // ------------------------------------------------------------------
    if (rank_id == 0) {
        std::cout << "\n[PERF] Starting warmup (" << WARMUP_ITERS << " iterations)..." << std::endl;
    }

    for (int i = 0; i < WARMUP_ITERS; ++i) {
        resetState();
        aclrtSynchronizeStream(computeStream);
        aclrtSynchronizeStream(commStream);
        ShmemBarrierAll();

        // Launch both kernels in parallel
        launchGemmCompute(
            reinterpret_cast<uint8_t *>(shmem_output),
            reinterpret_cast<uint8_t *>(src0_dev),
            reinterpret_cast<uint8_t *>(src1_dev),
            reinterpret_cast<uint8_t *>(queueSet_dev),
            rank_id, computeStream, COMPUTE_BLOCK_NUM, k_per_rank);

        launchGemmComm(
            reinterpret_cast<uint8_t *>(shmem_output),
            reinterpret_cast<uint8_t *>(recv_buffers),
            reinterpret_cast<uint8_t *>(reduced_output),
            reinterpret_cast<uint8_t *>(queueSet_dev),
            rank_id, n_ranks, commStream, COMPUTE_BLOCK_NUM);

        aclrtSynchronizeStream(computeStream);
        aclrtSynchronizeStream(commStream);
        ShmemBarrierAll();
    }

    // ------------------------------------------------------------------
    // Performance measurement iterations
    // ------------------------------------------------------------------
    if (rank_id == 0) {
        std::cout << "[PERF] Starting measurement (" << MEASURE_ITERS << " iterations)..." << std::endl;
    }

    ShmemBarrierAll();

    // Measure compute-only time (reference)
    // Run multiple iterations and take average for stable results (similar to gemm_performance_2)
    // Note: ShmemBarrierAll() is called AFTER timing to avoid including
    // synchronization overhead in pure compute performance measurement
    std::vector<double> compute_times_us;
    constexpr int COMPUTE_ONLY_ITERS = 10;  // Run 10 iterations for compute-only test
    
    if (rank_id == 0) {
        std::cout << "[PERF] Measuring compute-only performance (" << COMPUTE_ONLY_ITERS << " iterations)..." << std::endl;
    }
    
    for (int iter = 0; iter < COMPUTE_ONLY_ITERS; ++iter) {
        resetState();
        aclrtSynchronizeStream(computeStream);
        ShmemBarrierAll();

        auto compute_start = std::chrono::high_resolution_clock::now();
        launchGemmCompute(
            reinterpret_cast<uint8_t *>(shmem_output),
            reinterpret_cast<uint8_t *>(src0_dev),
            reinterpret_cast<uint8_t *>(src1_dev),
            reinterpret_cast<uint8_t *>(queueSet_dev),
            rank_id, computeStream, COMPUTE_BLOCK_NUM, k_per_rank);
        aclrtSynchronizeStream(computeStream);
        auto compute_end = std::chrono::high_resolution_clock::now();
        double iter_time_us = std::chrono::duration<double, std::micro>(compute_end - compute_start).count();
        compute_times_us.push_back(iter_time_us);
        
        // Synchronize after timing to ensure all ranks complete, but don't include in measurement
        ShmemBarrierAll();
    }
    
    // Calculate average compute time
    double compute_time_us = 0.0;
    if (!compute_times_us.empty()) {
        double sum = 0.0;
        for (double t : compute_times_us) {
            sum += t;
        }
        compute_time_us = sum / compute_times_us.size();
    }

    // Measure sequential execution (compute then comm)
    std::vector<double> sequential_times_us;
    std::vector<double> seq_compute_times_us;
    std::vector<double> seq_comm_times_us;
    for (int iter = 0; iter < MEASURE_ITERS; ++iter) {
        resetState();
        aclrtSynchronizeStream(computeStream);
        aclrtSynchronizeStream(commStream);
        ShmemBarrierAll();

        auto seq_start = std::chrono::high_resolution_clock::now();
        
        // Compute first
        launchGemmCompute(
            reinterpret_cast<uint8_t *>(shmem_output),
            reinterpret_cast<uint8_t *>(src0_dev),
            reinterpret_cast<uint8_t *>(src1_dev),
            reinterpret_cast<uint8_t *>(queueSet_dev),
            rank_id, computeStream, COMPUTE_BLOCK_NUM, k_per_rank);
        aclrtSynchronizeStream(computeStream);
        
        auto seq_compute_end = std::chrono::high_resolution_clock::now();
        double seq_compute_time = std::chrono::duration<double, std::micro>(seq_compute_end - seq_start).count();
        seq_compute_times_us.push_back(seq_compute_time);
        
        // Then comm + reduce (merged: TPUT scatter → device barrier → local reduce)
        launchGemmComm(
            reinterpret_cast<uint8_t *>(shmem_output),
            reinterpret_cast<uint8_t *>(recv_buffers),
            reinterpret_cast<uint8_t *>(reduced_output),
            reinterpret_cast<uint8_t *>(queueSet_dev),
            rank_id, n_ranks, commStream, COMPUTE_BLOCK_NUM);
        aclrtSynchronizeStream(commStream);
        
        auto seq_end = std::chrono::high_resolution_clock::now();
        double seq_comm_time = std::chrono::duration<double, std::micro>(seq_end - seq_compute_end).count();
        seq_comm_times_us.push_back(seq_comm_time);
        
        double seq_time_us = std::chrono::duration<double, std::micro>(seq_end - seq_start).count();
        sequential_times_us.push_back(seq_time_us);
        
        ShmemBarrierAll();
    }

    // Measure pipelined execution (overlapping compute and comm)
    std::vector<double> pipelined_times_us;
    std::vector<double> pipe_compute_times_us;
    std::vector<double> pipe_comm_times_us;
    for (int iter = 0; iter < MEASURE_ITERS; ++iter) {
        resetState();
        aclrtSynchronizeStream(computeStream);
        aclrtSynchronizeStream(commStream);
        ShmemBarrierAll();

        auto pipe_start = std::chrono::high_resolution_clock::now();
        
        // Launch both in parallel
        launchGemmCompute(
            reinterpret_cast<uint8_t *>(shmem_output),
            reinterpret_cast<uint8_t *>(src0_dev),
            reinterpret_cast<uint8_t *>(src1_dev),
            reinterpret_cast<uint8_t *>(queueSet_dev),
            rank_id, computeStream, COMPUTE_BLOCK_NUM, k_per_rank);
        launchGemmComm(
            reinterpret_cast<uint8_t *>(shmem_output),
            reinterpret_cast<uint8_t *>(recv_buffers),
            reinterpret_cast<uint8_t *>(reduced_output),
            reinterpret_cast<uint8_t *>(queueSet_dev),
            rank_id, n_ranks, commStream, COMPUTE_BLOCK_NUM);
        
        // Wait for compute to finish and record time
        aclrtSynchronizeStream(computeStream);
        auto pipe_compute_end = std::chrono::high_resolution_clock::now();
        double pipe_compute_time = std::chrono::duration<double, std::micro>(pipe_compute_end - pipe_start).count();
        pipe_compute_times_us.push_back(pipe_compute_time);
        
        // Wait for comm + reduce (merged) to finish
        aclrtSynchronizeStream(commStream);
        auto pipe_comm_end = std::chrono::high_resolution_clock::now();
        double pipe_comm_time = std::chrono::duration<double, std::micro>(pipe_comm_end - pipe_start).count();
        pipe_comm_times_us.push_back(pipe_comm_time);
        
        // Total pipelined time = max(compute, comm+reduce)
        double pipe_time_us = std::chrono::duration<double, std::micro>(pipe_comm_end - pipe_start).count();
        pipelined_times_us.push_back(pipe_time_us);
        
        ShmemBarrierAll();
    }

    // ------------------------------------------------------------------
    // Verification - run one more iteration to get final result
    // ------------------------------------------------------------------
    resetState();
    aclrtSynchronizeStream(computeStream);
    aclrtSynchronizeStream(commStream);
    ShmemBarrierAll();

    // Run final iteration for verification
    launchGemmCompute(
        reinterpret_cast<uint8_t *>(shmem_output),
        reinterpret_cast<uint8_t *>(src0_dev),
        reinterpret_cast<uint8_t *>(src1_dev),
        reinterpret_cast<uint8_t *>(queueSet_dev),
        rank_id, computeStream, COMPUTE_BLOCK_NUM, k_per_rank);
    launchGemmComm(
        reinterpret_cast<uint8_t *>(shmem_output),
        reinterpret_cast<uint8_t *>(recv_buffers),
        reinterpret_cast<uint8_t *>(reduced_output),
        reinterpret_cast<uint8_t *>(queueSet_dev),
        rank_id, n_ranks, commStream, COMPUTE_BLOCK_NUM);
    aclrtSynchronizeStream(computeStream);
    aclrtSynchronizeStream(commStream);
    ShmemBarrierAll();
    
    // Read final result from reduced_output
    // Plain TPUT wrote per-rank data to recv_buffers, then reduce kernel summed → reduced_output
    float *output_host = nullptr;
    aclrtMallocHost(reinterpret_cast<void **>(&output_host), outputSize);
    aclrtMemcpy(output_host, outputSize, reduced_output, outputSize, ACL_MEMCPY_DEVICE_TO_HOST);
    
    // Read first recv_buffers slot for debugging (slot 0 = rank 0's data)
    float *recv_host = nullptr;
    size_t debugBufferSize = std::min(recvBuffersSize, static_cast<size_t>(1024 * sizeof(float)));
    aclrtMallocHost(reinterpret_cast<void **>(&recv_host), debugBufferSize);
    aclrtMemcpy(recv_host, debugBufferSize, recv_buffers, debugBufferSize, ACL_MEMCPY_DEVICE_TO_HOST);

    // Also read shmem_output for diagnostics
    float *shmem_host = nullptr;
    size_t shmem_debug_size = 10 * sizeof(float);
    aclrtMallocHost(reinterpret_cast<void **>(&shmem_host), shmem_debug_size);
    aclrtMemcpy(shmem_host, shmem_debug_size, shmem_output, shmem_debug_size, ACL_MEMCPY_DEVICE_TO_HOST);
    
    // Read recv_buffers slot 1 (data received from rank 1 via TPUT)
    float *recv_slot1_host = nullptr;
    aclrtMallocHost(reinterpret_cast<void **>(&recv_slot1_host), shmem_debug_size);
    // slot 1 offset = 1 * output_size * sizeof(float)
    size_t slot1_byte_offset = static_cast<size_t>(1) * outputSize;
    aclrtMemcpy(recv_slot1_host, shmem_debug_size, 
                reinterpret_cast<uint8_t*>(recv_buffers) + slot1_byte_offset, 
                shmem_debug_size, ACL_MEMCPY_DEVICE_TO_HOST);
    
    // Save output (only rank 0)
    bool is_ok = true;
    if (rank_id == 0) {
        // Debug: shmem_output (local GEMM result)
        std::cout << "[DEBUG] First 10 values of shmem_output (local GEMM): ";
        for (int i = 0; i < 10; ++i) std::cout << shmem_host[i] << " ";
        std::cout << std::endl;
        
        // Debug: recv_buffers slot 0 (rank 0's own slot - should be zero since we skip self-TPUT)
        std::cout << "[DEBUG] First 10 values of recv_buffers[slot 0]: ";
        size_t max_debug_elements = debugBufferSize / sizeof(float);
        size_t debug_count = std::min(10UL, max_debug_elements);
        for (size_t i = 0; i < debug_count; ++i) {
            std::cout << recv_host[i] << " ";
        }
        std::cout << std::endl;
        
        // Debug: recv_buffers slot 1 (rank 1's data received via TPUT)
        std::cout << "[DEBUG] First 10 values of recv_buffers[slot 1] (rank 1 data): ";
        for (int i = 0; i < 10; ++i) std::cout << recv_slot1_host[i] << " ";
        std::cout << std::endl;
        
        std::cout << "[DEBUG] Communication: plain TPUT to per-rank recv_buffers + local TPUT<AtomicAdd> reduce (merged in comm kernel)."
                  << std::endl;
        
        // Debug: print first few values of output_host (device-side reduced result)
        std::cout << "[DEBUG] First 10 values of output_host (device reduced): ";
        for (int i = 0; i < 10; ++i) {
            std::cout << output_host[i] << " ";
        }
        std::cout << std::endl;
        // Diagnostic: check specific positions across tile boundaries
        size_t total_elements = G_M * G_N;
        std::cout << "[DEBUG] output[256]=" << output_host[256]  // row 0, col 256 (tile 0,1)
                  << " output[512]=" << output_host[512]  // row 0, col 512 (tile 0,2)
                  << " output[598]=" << output_host[598]  // 0x256
                  << " output[4096]=" << output_host[4096] // row 1, col 0 (tile 0,0)
                  << " output[4352]=" << output_host[4352] // row 1, col 256 (tile 0,1)
                  << std::endl;
        // Count total zeros
        size_t zero_count = 0;
        size_t first_zero = total_elements;
        for (size_t i = 0; i < total_elements; ++i) {
            if (output_host[i] == 0.0f) {
                if (zero_count == 0) first_zero = i;
                zero_count++;
            }
        }
        std::cout << "[DEBUG] Total zeros: " << zero_count << "/" << total_elements
                  << " first_zero_at=" << first_zero << std::endl;
        
        std::string output_dir = "../output";
        std::string output_file = output_dir + "/output_z.bin";
        if (!PtoTestCommon::WriteFile(output_file, output_host, outputSize)) {
            std::cerr << "[ERROR] Rank " << rank_id << ": Failed to write output file" << std::endl;
        }

        // Read golden data and compare
        std::string golden_file = output_dir + "/golden.bin";
        std::vector<float> golden(outputSize / sizeof(float));
        size_t golden_file_size = 0;
        if (PtoTestCommon::ReadFile(golden_file, golden_file_size, golden.data(), outputSize)) {
            // Debug: print first few values of golden
            std::cout << "[DEBUG] First 10 values of golden: ";
            for (int i = 0; i < 10; ++i) {
                std::cout << golden[i] << " ";
            }
            std::cout << std::endl;
            
            is_ok = PtoTestCommon::ResultCmp(golden, output_host, 0.001f);
            if (is_ok) {
                std::cout << "[INFO] Result comparison passed!" << std::endl;
            } else {
                std::cout << "[ERROR] Result comparison failed!" << std::endl;
            }
        } else {
            std::cout << "[WARNING] Golden file not found, skipping verification" << std::endl;
        }
    }

    aclrtFreeHost(output_host);
    aclrtFreeHost(recv_host);
    
    // Free pre-allocated queue set reset buffer
    if (queueSet_reset_host != nullptr) {
        aclrtFreeHost(queueSet_reset_host);
    }

    // ------------------------------------------------------------------
    // Performance statistics (only rank 0)
    // ------------------------------------------------------------------
    if (rank_id == 0) {
        struct Stats { double avg; double min_val; double max_val; double std_dev; };
        auto calc_stats = [](const std::vector<double>& times) -> Stats {
            double sum = 0.0, mn = times[0], mx = times[0];
            for (double t : times) {
                sum += t;
                if (t < mn) mn = t;
                if (t > mx) mx = t;
            }
            double avg = sum / times.size();
            double variance = 0.0;
            for (double t : times) {
                variance += (t - avg) * (t - avg);
            }
            double sd = std::sqrt(variance / times.size());
            return Stats{avg, mn, mx, sd};
        };

        Stats seq_s = calc_stats(sequential_times_us);
        double seq_avg = seq_s.avg, seq_min = seq_s.min_val, seq_max = seq_s.max_val, seq_std = seq_s.std_dev;
        Stats pipe_s = calc_stats(pipelined_times_us);
        double pipe_avg = pipe_s.avg, pipe_min = pipe_s.min_val, pipe_max = pipe_s.max_val, pipe_std = pipe_s.std_dev;
        
        // Calculate stats for sequential compute and comm times
        Stats seq_compute_s = calc_stats(seq_compute_times_us);
        double seq_compute_avg = seq_compute_s.avg, seq_compute_min = seq_compute_s.min_val, seq_compute_max = seq_compute_s.max_val, seq_compute_std = seq_compute_s.std_dev;
        Stats seq_comm_s = calc_stats(seq_comm_times_us);
        double seq_comm_avg = seq_comm_s.avg, seq_comm_min = seq_comm_s.min_val, seq_comm_max = seq_comm_s.max_val, seq_comm_std = seq_comm_s.std_dev;
        
        // Calculate stats for pipelined compute and comm times
        Stats pipe_compute_s = calc_stats(pipe_compute_times_us);
        double pipe_compute_avg = pipe_compute_s.avg, pipe_compute_min = pipe_compute_s.min_val, pipe_compute_max = pipe_compute_s.max_val, pipe_compute_std = pipe_compute_s.std_dev;
        Stats pipe_comm_s = calc_stats(pipe_comm_times_us);
        double pipe_comm_avg = pipe_comm_s.avg, pipe_comm_min = pipe_comm_s.min_val, pipe_comm_max = pipe_comm_s.max_val, pipe_comm_std = pipe_comm_s.std_dev;

        // Calculate FLOPS: GEMM requires 2*M*K*N operations
        // IMPORTANT: In data parallel mode, each rank only computes part of K dimension
        // So we use k_per_rank instead of G_K for per-rank computation
        // For total system throughput, we can use G_K (all ranks combined)
        uint32_t k_per_rank = G_K / n_ranks;
        double gemm_flops_per_rank = 2.0 * static_cast<double>(G_M) * static_cast<double>(k_per_rank) * static_cast<double>(G_N);
        double gemm_flops_total = 2.0 * static_cast<double>(G_M) * static_cast<double>(G_K) * static_cast<double>(G_N);
        
        // Per-rank GFLOPS (what each rank actually computes)
        double compute_gflops = (compute_time_us > 0.0) ? (gemm_flops_per_rank / (compute_time_us * 1e-6) / 1e9) : 0.0;
        
        // Total system GFLOPS (all ranks combined, for sequential/pipelined total time)
        double seq_gflops = (seq_avg > 0.0) ? (gemm_flops_total / (seq_avg * 1e-6) / 1e9) : 0.0;
        double pipe_gflops = (pipe_avg > 0.0) ? (gemm_flops_total / (pipe_avg * 1e-6) / 1e9) : 0.0;
        
        // Calculate compute-only GFLOPS for Sequential and Pipelined modes (per-rank)
        // This helps verify if the compute kernel itself is slower when running with comm
        double seq_compute_gflops = (seq_compute_avg > 0.0) ? (gemm_flops_per_rank / (seq_compute_avg * 1e-6) / 1e9) : 0.0;
        double pipe_compute_gflops = (pipe_compute_avg > 0.0) ? (gemm_flops_per_rank / (pipe_compute_avg * 1e-6) / 1e9) : 0.0;
        
        // Calculate communication bandwidth
        // For AllReduce: each rank sends its data (outputSize bytes) to all n_ranks (including self)
        // Total data sent per rank = outputSize * n_ranks
        size_t outputSize = static_cast<size_t>(G_M) * G_N * sizeof(float);
        double data_per_rank_bytes = static_cast<double>(outputSize) * static_cast<double>(n_ranks);
        double data_per_rank_gb = data_per_rank_bytes / (1024.0 * 1024.0 * 1024.0);
        
        // Sequential communication bandwidth (GB/s)
        double seq_comm_bandwidth_gbs = (seq_comm_avg > 0.0) 
            ? (data_per_rank_bytes / (seq_comm_avg * 1e-6) / (1024.0 * 1024.0 * 1024.0)) 
            : 0.0;
        
        // Pipelined communication bandwidth (GB/s)
        double pipe_comm_bandwidth_gbs = (pipe_comm_avg > 0.0) 
            ? (data_per_rank_bytes / (pipe_comm_avg * 1e-6) / (1024.0 * 1024.0 * 1024.0)) 
            : 0.0;

        std::cout << "\n================================================================" << std::endl;
        if (is_ok) {
            std::cout << "[SUCCESS] GEMM AllReduce test PASSED!" << std::endl;
        } else {
            std::cout << "[FAILED] GEMM AllReduce test FAILED!" << std::endl;
        }
        std::cout << "  Matrix dimensions: M=" << G_M << ", K=" << G_K << ", N=" << G_N << std::endl;
        std::cout << "  Ranks: " << n_ranks << std::endl;
        std::cout << "  Compute blocks: " << COMPUTE_BLOCK_NUM << ", Comm blocks: " << COMM_BLOCK_NUM << std::endl;
        std::cout << "  Total tiles: " << G_NUM_TILES << " (" << G_M_TILES << "x" << G_N_TILES << ")" << std::endl;
        
        std::cout << "\n[PERF] Performance Results (n_iters=" << MEASURE_ITERS << "):" << std::endl;
        std::cout << std::fixed << std::setprecision(3);
        
        // Calculate compute-only statistics
        double compute_min = *std::min_element(compute_times_us.begin(), compute_times_us.end());
        double compute_max = *std::max_element(compute_times_us.begin(), compute_times_us.end());
        double compute_variance = 0.0;
        for (double t : compute_times_us) {
            compute_variance += (t - compute_time_us) * (t - compute_time_us);
        }
        double compute_std = std::sqrt(compute_variance / compute_times_us.size());
        
        // Use per-rank FLOPS for compute-only statistics (k_per_rank and gemm_flops_per_rank already defined above)
        double compute_gflops_min = (compute_min > 0.0) ? (gemm_flops_per_rank / (compute_min * 1e-6) / 1e9) : 0.0;
        double compute_gflops_max = (compute_max > 0.0) ? (gemm_flops_per_rank / (compute_max * 1e-6) / 1e9) : 0.0;
        
        std::cout << "\n  Component Breakdown (Reference - Per Rank):" << std::endl;
        std::cout << "    Compute Kernel:    " << std::fixed << std::setprecision(3) 
                  << compute_time_us << " us (min: " << compute_min << ", max: " << compute_max 
                  << ", std: " << compute_std << ")" << std::endl;
        std::cout << "    Compute GFLOPS:    " << compute_gflops << " GFLOPS (avg), " 
                  << compute_gflops_max << " GFLOPS (best) [per rank, M×K/r×N]" << std::endl;
        
        std::cout << "\n  Sequential Execution (Compute -> Comm, no overlap):" << std::endl;
        std::cout << "    Total Time:      " << seq_avg << " us (min: " << seq_min << ", max: " << seq_max << ", std: " << seq_std << ")" << std::endl;
        std::cout << "    Compute Time:     " << seq_compute_avg << " us (min: " << seq_compute_min << ", max: " << seq_compute_max << ", std: " << seq_compute_std << ")" << std::endl;
        std::cout << "    Compute GFLOPS:   " << seq_compute_gflops << " GFLOPS (compute-only efficiency)" << std::endl;
        std::cout << "    Comm Time:        " << seq_comm_avg << " us (min: " << seq_comm_min << ", max: " << seq_comm_max << ", std: " << seq_comm_std << ")" << std::endl;
        std::cout << "    Comm Data:        " << std::setprecision(4) << data_per_rank_gb << " GB/rank (per rank sends to all " << n_ranks << " ranks)" << std::endl;
        std::cout << "    Comm Bandwidth:   " << std::setprecision(3) << seq_comm_bandwidth_gbs << " GB/s" << std::endl;
        std::cout << "    Throughput:       " << seq_gflops << " GFLOPS (total efficiency)" << std::endl;
        
        std::cout << "\n  Pipelined Execution (Compute || Comm, with overlap):" << std::endl;
        std::cout << "    Total Time:      " << pipe_avg << " us (min: " << pipe_min << ", max: " << pipe_max << ", std: " << pipe_std << ")" << std::endl;
        std::cout << "    Compute Done:    " << pipe_compute_avg << " us (min: " << pipe_compute_min << ", max: " << pipe_compute_max << ", std: " << pipe_compute_std << ")" << std::endl;
        std::cout << "    Compute GFLOPS:  " << pipe_compute_gflops << " GFLOPS (compute-only efficiency)" << std::endl;
        std::cout << "    Comm Done:       " << pipe_comm_avg << " us (min: " << pipe_comm_min << ", max: " << pipe_comm_max << ", std: " << pipe_comm_std << ")" << std::endl;
        std::cout << "    Comm Data:       " << std::setprecision(4) << data_per_rank_gb << " GB/rank (per rank sends to all " << n_ranks << " ranks)" << std::endl;
        std::cout << "    Comm Bandwidth:  " << std::setprecision(3) << pipe_comm_bandwidth_gbs << " GB/s" << std::endl;
        std::cout << "    Throughput:      " << pipe_gflops << " GFLOPS (total efficiency)" << std::endl;
        
        // Calculate overlap efficiency
        double overlap_time = (seq_compute_avg + seq_comm_avg) - pipe_avg;
        double overlap_efficiency = (overlap_time > 0) ? (overlap_time / std::min(seq_compute_avg, seq_comm_avg) * 100.0) : 0.0;
        
        double speedup = seq_avg / pipe_avg;
        std::cout << "\n  Performance Comparison:" << std::endl;
        std::cout << "    Speedup:            " << speedup << "x (Pipelined vs Sequential)" << std::endl;
        std::cout << "    Time Saved:         " << (seq_avg - pipe_avg) << " us (" 
                  << ((seq_avg - pipe_avg) / seq_avg * 100.0) << "%)" << std::endl;
        std::cout << "    Overlap Time:       " << overlap_time << " us" << std::endl;
        std::cout << "    Overlap Efficiency: " << overlap_efficiency << "%" << std::endl;
        
        // Compare compute efficiency
        std::cout << "\n  Compute Efficiency Analysis:" << std::endl;
        std::cout << "    Pure Compute:       " << compute_gflops << " GFLOPS" << std::endl;
        std::cout << "    Sequential Compute: " << seq_compute_gflops << " GFLOPS (" 
                  << (seq_compute_gflops / compute_gflops * 100.0) << "% of pure)" << std::endl;
        std::cout << "    Pipelined Compute:  " << pipe_compute_gflops << " GFLOPS (" 
                  << (pipe_compute_gflops / compute_gflops * 100.0) << "% of pure)" << std::endl;
        if (seq_compute_gflops < compute_gflops * 0.95) {
            std::cout << "    [WARNING] Sequential compute efficiency is significantly lower than pure compute!" << std::endl;
        }
        if (pipe_compute_gflops < compute_gflops * 0.95) {
            std::cout << "    [WARNING] Pipelined compute efficiency is significantly lower than pure compute!" << std::endl;
            std::cout << "              This suggests resource competition between compute and comm kernels." << std::endl;
        }
        
        std::cout << "================================================================\n" << std::endl;
    }

    // ------------------------------------------------------------------
    // Cleanup
    // ------------------------------------------------------------------
    ShmemFree(shmem_output);
    ShmemFree(recv_buffers);
    ShmemFree(reduced_output);
    aclrtFree(src0_dev);
    aclrtFree(src1_dev);
    aclrtFree(queueSet_dev);

    ShmemFinalize();

    status |= aclrtDestroyStream(computeStream);
    status |= aclrtDestroyStream(commStream);
    status |= aclrtResetDevice(device_id);
    status |= aclFinalize();

    return (status == 0) && is_ok;
}

// ============================================================================
// Multi-process launcher
// ============================================================================
bool RunGemmAllReduce(int n_ranks, int first_device_id)
{
    if (n_ranks <= 0 || n_ranks > 8) {
        std::cerr << "[ERROR] Invalid n_ranks: " << n_ranks << " (must be 1-8)\n";
        return false;
    }

    std::cout << "\n================================================================" << std::endl;
    std::cout << "  GEMM AllReduce Performance Test" << std::endl;
    std::cout << "================================================================" << std::endl;
    std::cout << "  Matrix: M=" << G_M << ", K=" << G_K << ", N=" << G_N << std::endl;
    std::cout << "  Base tile: " << G_BASE_M << "x" << G_BASE_K << "x" << G_BASE_N << std::endl;
    std::cout << "  Total tiles: " << G_NUM_TILES << " (" << G_M_TILES << "x" << G_N_TILES << ")" << std::endl;
    std::cout << "  Compute blocks: " << COMPUTE_BLOCK_NUM << std::endl;
    std::cout << "  Comm blocks: " << COMM_BLOCK_NUM << std::endl;
    std::cout << "  Number of ranks: " << n_ranks << std::endl;
    std::cout << "  Device range: [" << first_device_id << ", " << (first_device_id + n_ranks) << ")" << std::endl;
    std::cout << "  Warmup iterations: " << WARMUP_ITERS << std::endl;
    std::cout << "  Measurement iterations: " << MEASURE_ITERS << std::endl;
    std::cout << "================================================================" << std::endl;

    std::vector<pid_t> pids;
    for (int r = 0; r < n_ranks; ++r) {
        pid_t pid = fork();
        if (pid == 0) {
            int device_id = first_device_id + r;
            const bool ok = RunGemmAllReducePerRank(r, n_ranks, device_id);
            _exit(ok ? 0 : 1);
        } else if (pid > 0) {
            pids.push_back(pid);
        } else {
            std::cerr << "[ERROR] fork() failed for rank " << r << "\n";
            // Kill already forked processes
            for (pid_t p : pids) {
                kill(p, SIGTERM);
            }
            return false;
        }
    }

    bool success = true;
    for (pid_t p : pids) {
        int wstatus = 0;
        waitpid(p, &wstatus, 0);
        if (!(WIFEXITED(wstatus) && WEXITSTATUS(wstatus) == 0)) {
            success = false;
            std::cerr << "[ERROR] Rank process " << p << " exited with status " << WEXITSTATUS(wstatus) << "\n";
        }
    }
    return success;
}
