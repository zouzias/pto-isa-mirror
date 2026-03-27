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
// GEMM Compute Kernel (Cube Arch) for GEMM + AllReduce Demo
//
// This kernel computes C_local = A_local * B for all output tiles.
// After each tile is written to shmem output, the tile index is enqueued to
// a ready queue to signal the communication kernel for AllReduce transfer.
//
// Key Features:
//   - Uses lock-free ready queue for signaling (from gemm_allgather)
//   - Supports multi-block parallel execution with arbitrary completion order
//   - Comm kernel polls queues and starts AllReduce as soon as tiles are ready
//
// Parameters:
//   M=512, K=2048, N=1536 (global)
//   baseM=128, baseK=64, baseN=256
//   M_tiles=4, N_tiles=6 => 24 tiles per rank
//   kLoop = K/baseK = 32
// ============================================================================

#include <pto/common/constants.hpp>
#include <pto/pto-inst.hpp>
#include "ready_queue.hpp"

using namespace pto;

// ============================================================================
// ProcessKIteration: Optimized K-loop with L1 caching (stepK=4)
// 
// Optimization: Every stepK iterations, load a larger panel into L1 for reuse.
// This reduces GM->L1 DMA frequency by 4x.
// ============================================================================
template <typename T, typename U, typename S, int M, int K, int N,
          uint32_t baseM, uint32_t baseK, uint32_t baseN,
          uint32_t stepKa, uint32_t stepKb>
AICORE inline void ProcessKIteration(
    uint32_t kIter, __gm__ U *currentSrc0, __gm__ S *currentSrc1,
    Tile<TileType::Mat, U, baseM, baseK * stepKa,
         BLayout::ColMajor, baseM, baseK * stepKa, SLayout::RowMajor> aMatTile[2],
    Tile<TileType::Mat, S, baseK * stepKb, baseN,
         BLayout::RowMajor, baseK * stepKb, baseN, SLayout::ColMajor> bMatTile[2],
    TileLeft<U, baseM, baseK, baseM, baseK> aTile[2],
    TileRight<S, baseK, baseN, baseK, baseN> bTile[2],
    TileAcc<T, baseM, baseN, baseM, baseN> &cTile,
    uint8_t &mte2DBFlag, uint8_t &mte1DBFlag,
    uint32_t k_stride)  // Runtime K stride (k_per_rank for data parallel mode)
{
    // GlobalTensor shapes for TLOAD (valid shape includes stepK caching)
    // Use DYNAMIC BaseShape to support runtime K stride (k_per_rank)
    using NDValidShapeA = TileShape2D<U, baseM, baseK * stepKa, Layout::ND>;
    using NDsingleCoreShapeA = BaseShape2D<U, M, DYNAMIC, Layout::ND>;  // DYNAMIC for K dimension
    using GlobalDataSrcA = GlobalTensor<U, NDValidShapeA, NDsingleCoreShapeA, Layout::ND>;

    using NDValidShapeB = TileShape2D<U, baseK * stepKb, baseN, Layout::DN>;
    using NDsingleCoreShapeB = BaseShape2D<U, DYNAMIC, N, Layout::DN>;  // DYNAMIC for K dimension
    using GlobalDataSrcB = GlobalTensor<U, NDValidShapeB, NDsingleCoreShapeB, Layout::DN>;

    const uint32_t kModStepKa = kIter % stepKa;

    // ---- TLOAD stage ----
    // Phase 1.1 + 1.2: Implement stepK loading with TEXTRACT
    // Every stepKa iterations, load a larger panel (stepKa K-slices) into L1
    // Then use TEXTRACT to extract individual slices from cached panel
    if (kModStepKa == 0) {
        // Create GlobalTensor with runtime stride
        // For ND layout: stride dim3 (column stride) = k_stride
        // For DN layout: stride dim4 (row stride) = k_stride
        NDsingleCoreShapeA aStride(M, k_stride);  // (rows, cols) -> stride dim3 = cols = k_stride for ND layout
        NDsingleCoreShapeB bStride(k_stride, N);  // (rows, cols) -> stride dim4 = rows = k_stride for DN layout
        // GlobalTensor constructor: (data, shape, stride)
        // Use default Shape (static), pass runtime Stride as second parameter
        NDValidShapeA aShape;  // Default constructor for static shape
        NDValidShapeB bShape;  // Default constructor for static shape
        GlobalDataSrcA gmA(currentSrc0 + kIter * baseK, aShape, aStride);
        GlobalDataSrcB gmB(currentSrc1 + kIter * baseK, bShape, bStride);

        // Wait until TEXTRACT is done with this L1 buffer before reusing it
        wait_flag(PIPE_MTE1, PIPE_MTE2, (event_t)mte2DBFlag);
        TLOAD(aMatTile[mte2DBFlag], gmA);
        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
        TLOAD(bMatTile[mte2DBFlag], gmB);
        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
        mte2DBFlag = (mte2DBFlag == 0) ? 1 : 0;
    }

    const uint32_t currMte2Idx = (mte2DBFlag == 0) ? 1 : 0;

    // ---- TEXTRACT stage ----
    // Phase 1.2: Use TEXTRACT to extract current K-slice from cached L1 panel
    // Wait until TMATMUL is done with current L0 buffer before overwriting
    wait_flag(PIPE_M, PIPE_MTE1, (event_t)mte1DBFlag);
    
    // Extract current K-slice from the cached L1 panel
    if (kModStepKa == 0)
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    // For A matrix (ColMajor): extract from row 0, column kModStepKa * baseK
    TEXTRACT(aTile[mte1DBFlag], aMatTile[currMte2Idx], 0, kModStepKa * baseK);
    
    if (kModStepKa == 0)
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
    // For B matrix (RowMajor): extract from row (kIter % stepKb) * baseK, column 0
    TEXTRACT(bTile[mte1DBFlag], bMatTile[currMte2Idx], (kIter % stepKb) * baseK, 0);
    
    if ((kIter + 1) % stepKa == 0) {
        // Allow the next TLOAD to reuse this L1 slot
        set_flag(PIPE_MTE1, PIPE_MTE2, (event_t)currMte2Idx);
    }

    // ---- TMATMUL stage ----
    set_flag(PIPE_MTE1, PIPE_M, (event_t)mte1DBFlag);
    wait_flag(PIPE_MTE1, PIPE_M, (event_t)mte1DBFlag);
    if (kIter == 0) {
        TMATMUL(cTile, aTile[mte1DBFlag], bTile[mte1DBFlag]);
    } else {
        TMATMUL_ACC(cTile, cTile, aTile[mte1DBFlag], bTile[mte1DBFlag]);
    }
    // Signal that TMATMUL is done, so next iteration may TEXTRACT into the other slot
    set_flag(PIPE_M, PIPE_MTE1, (event_t)mte1DBFlag);
    mte1DBFlag = (mte1DBFlag == 0) ? 1 : 0;
}

// ============================================================================
// Global GEMM parameters
// ============================================================================
// Size configurations for AllReduce demo
#ifndef CONFIG_G_M
#define CONFIG_G_M 16384
#endif
#ifndef CONFIG_G_K
#define CONFIG_G_K 16384
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
constexpr uint32_t G_M_TILES = G_M / G_BASE_M;        // Tiles in M dimension
constexpr uint32_t G_N_TILES = G_N / G_BASE_N;        // Tiles in N dimension
constexpr uint32_t G_K_LOOP = G_K / G_BASE_K;         // K loop iterations
constexpr uint32_t G_NUM_TILES = G_M_TILES * G_N_TILES; // Total tiles per rank

// L1 caching: load stepK K-slices per TLOAD to improve bandwidth utilization
// Optimization: stepK=8 reduces GM→L1 DMA frequency by 8x (vs 4x with stepK=4)
// L1 usage: 2×128KB(A) + 2×256KB(B) = 768KB ≤ 1024KB L1 capacity
constexpr uint32_t G_STEP_KA = 4;    // A: cache 4 K-slices → TLOAD [128, 256] = 64KB
constexpr uint32_t G_STEP_KB = 4;    // B: cache 4 K-slices → TLOAD [256, 256] = 128KB
static_assert(G_K_LOOP % G_STEP_KA == 0, "G_K_LOOP must be divisible by G_STEP_KA");
static_assert(G_K_LOOP % G_STEP_KB == 0, "G_K_LOOP must be divisible by G_STEP_KB");
static_assert(G_STEP_KA == G_STEP_KB, "Current implementation assumes stepKa == stepKb");
static_assert(G_K_LOOP >= G_STEP_KA, "K_LOOP must be >= stepKa for L1 caching");

// ============================================================================
// Multi-block configuration for compute kernel
// ============================================================================
#ifndef CONFIG_COMPUTE_BLOCK_NUM
#define CONFIG_COMPUTE_BLOCK_NUM 32
#endif
constexpr int COMPUTE_BLOCK_NUM = CONFIG_COMPUTE_BLOCK_NUM;

// ============================================================================
// GemmComputeImpl: Core compute logic (AICORE inline, callable from entry)
//
// Multi-block parallel version (NO CONTENTION):
// Each block handles a subset of tiles and writes to its OWN queue.
// No atomic operations needed - each block is the sole producer for its queue.
//
// shmem_output: Symmetric heap buffer for C[M, N] (float)
// src0:         A[M, K] (half) - local portion of A
// src1:         B[K, N] (half) - full B matrix
// queue_set:    MultiBlockQueueSet* - per-block queues for signaling comm kernel
// rank:         Current rank
// block_num:    Total number of blocks launched
// ============================================================================
AICORE inline void GemmComputeImpl(
    __gm__ float *shmem_output,
    __gm__ half *src0,
    __gm__ half *src1,
    __gm__ MultiBlockQueueSet *queue_set,
    int rank,
    int block_num,
    uint32_t k_per_rank)  // K dimension per rank (for data parallel mode)
{
    // Get block index to determine which tiles this block handles
    const int block_idx = get_block_idx();

    // Output GlobalTensor: tile shape = baseM x baseN, base shape = M x N (for stride)
    using NDValidShapeC = TileShape2D<float, G_BASE_M, G_BASE_N>;
    using NDWholeShapeC = BaseShape2D<float, G_M, G_N>;
    using GlobalDataOut = GlobalTensor<float, NDValidShapeC, NDWholeShapeC>;

    // L1 Mat tiles with stepK caching (double buffered)
    // A: [baseM, baseK*stepKa] = [128, 256], 64KB per buffer
    using TileMatAData = Tile<TileType::Mat, half, G_BASE_M, G_BASE_K * G_STEP_KA,
                              BLayout::ColMajor, G_BASE_M, G_BASE_K * G_STEP_KA, SLayout::RowMajor>;
    // B: [baseK*stepKb, baseN] = [256, 256], 128KB per buffer
    using TileMatBData = Tile<TileType::Mat, half, G_BASE_K * G_STEP_KB, G_BASE_N,
                              BLayout::RowMajor, G_BASE_K * G_STEP_KB, G_BASE_N, SLayout::ColMajor>;

    TileMatAData aMatTile[2];
    TileMatBData bMatTile[2];
    // L1 layout: [pingA(64KB) | pongA(64KB) | pingB(128KB) | pongB(128KB)] = 384KB total
    constexpr size_t l1ASize = G_BASE_M * G_BASE_K * G_STEP_KA * sizeof(half);   // 64KB
    constexpr size_t l1BSize = G_BASE_K * G_STEP_KB * G_BASE_N * sizeof(half);   // 128KB
    TASSIGN(aMatTile[0], 0x0);
    TASSIGN(aMatTile[1], 0x0 + l1ASize);
    TASSIGN(bMatTile[0], 0x0 + 2 * l1ASize);
    TASSIGN(bMatTile[1], 0x0 + 2 * l1ASize + l1BSize);

    // L0A/L0B/L0C tiles
    using LeftTile  = TileLeft<half, G_BASE_M, G_BASE_K, G_BASE_M, G_BASE_K>;
    using RightTile = TileRight<half, G_BASE_K, G_BASE_N, G_BASE_K, G_BASE_N>;
    using ResTile   = TileAcc<float, G_BASE_M, G_BASE_N, G_BASE_M, G_BASE_N>;

    LeftTile  aTile[2];
    RightTile bTile[2];
    ResTile   cTile;

    TASSIGN(aTile[0], 0x0);
    TASSIGN(aTile[1], 0x0 + G_BASE_M * G_BASE_K * sizeof(half));
    TASSIGN(bTile[0], 0x0);
    TASSIGN(bTile[1], 0x0 + G_BASE_K * G_BASE_N * sizeof(half));
    TASSIGN(cTile, 0x0);

    // Calculate how many tiles this block should handle
    // Distribute tiles evenly among blocks
    const int total_tiles = G_NUM_TILES;
    const int tiles_per_block = (total_tiles + block_num - 1) / block_num;  // Ceiling division
    const int my_start_tile = block_idx * tiles_per_block;
    const int my_end_tile = (block_idx + 1) * tiles_per_block;

    // Cache the queue pointer once — avoid repeated GetMyBlockQueue dcci per tile
    volatile __gm__ PerBlockQueue* my_queue = GetMyBlockQueue(
        (volatile __gm__ MultiBlockQueueSet*)queue_set, block_idx);
    int32_t enqueue_slot = 0;  // Caller-tracked slot position for fast enqueue

    // Process tiles assigned to this block
    for (int tile_idx = my_start_tile; tile_idx < my_end_tile && tile_idx < total_tiles; tile_idx++) {
        // Convert linear tile_idx back to (mi, ni)
        uint32_t mi = tile_idx / G_N_TILES;
        uint32_t ni = tile_idx % G_N_TILES;

        // A pointer: row block mi of A (ND layout, stride = k_per_rank)
        __gm__ half *currentSrc0 = src0 + mi * G_BASE_M * k_per_rank;
        // B pointer: column block ni of B (DN layout, stride = k_per_rank per column)
        __gm__ half *currentSrc1 = src1 + ni * G_BASE_N * k_per_rank;

        // K-loop with two-level double-buffered pipeline
        // Initialize double-buffering flags for this tile
        uint8_t mte2DBFlag = 0, mte1DBFlag = 0;
        
        // Calculate K loop iterations for this rank
        uint32_t k_loop_per_rank = k_per_rank / G_BASE_K;
        static_assert(G_BASE_K == 64, "G_BASE_K must be 64 for this implementation");
        
        // Initial sync flags (consumed by reverse sync in ProcessKIteration)
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);

        for (uint32_t kIter = 0; kIter < k_loop_per_rank; kIter++) {
            // Pass k_per_rank as runtime stride to ProcessKIteration
            // This ensures GlobalTensor uses correct stride for data parallel mode
            ProcessKIteration<float, half, half, G_M, G_K, G_N,
                              G_BASE_M, G_BASE_K, G_BASE_N, G_STEP_KA, G_STEP_KB>(
                kIter, currentSrc0, currentSrc1,
                aMatTile, bMatTile, aTile, bTile, cTile,
                mte2DBFlag, mte1DBFlag,
                k_per_rank);  // Pass runtime K stride
        }

        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);

        // ----------------------------------------------------------------
        // Write result tile to shmem output
        // Position in C[M, N]:
        //   row = mi * baseM
        //   col = ni * baseN
        // ----------------------------------------------------------------
        uint64_t outOffset = (uint64_t)(mi * G_BASE_M) * G_N + ni * G_BASE_N;
        __gm__ float *tileDst = shmem_output + outOffset;

        set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
        wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
        GlobalDataOut dstGlobal(tileDst);
        TSTORE(dstGlobal, cTile);

        // Ensure tile data is fully written to GM before signaling comm kernel.
        // Using PIPE_ALL to drain all pipes and ensure GM write completion.
        // This is critical for data integrity before signaling the comm kernel.
        pipe_barrier(PIPE_ALL);

        // ----------------------------------------------------------------
        // Signal comm kernel: tile is ready
        // Fast enqueue: uses cached queue pointer and caller-tracked slot,
        // reducing from 5 dcci calls to 2 (data write + count signal).
        // ----------------------------------------------------------------
        MultiBlockEnqueueFast(my_queue, tile_idx, enqueue_slot);
        enqueue_slot++;
        
        // Note: No pipe_barrier needed after enqueue.
        // The dcci inside PerBlockQueueEnqueueFast ensures the enqueue data
        // is visible to the comm kernel. Removing this barrier allows the
        // next iteration's L1 loads (PIPE_MTE2) to overlap with cache ops.
    }
}

// ============================================================================
// Entry point kernel: takes __gm__ uint8_t* and casts inside
// Multi-block version: block_num blocks run in parallel
// ============================================================================
__global__ AICORE void GemmComputeKernel(
    __gm__ uint8_t *shmem_output,
    __gm__ uint8_t *src0,
    __gm__ uint8_t *src1,
    __gm__ uint8_t *queue_set,
    int rank,
    int block_num,
    uint32_t k_per_rank)
{
    GemmComputeImpl(
        reinterpret_cast<__gm__ float *>(shmem_output),
        reinterpret_cast<__gm__ half *>(src0),
        reinterpret_cast<__gm__ half *>(src1),
        reinterpret_cast<__gm__ MultiBlockQueueSet *>(queue_set),
        rank,
        block_num,
        k_per_rank);
}

// ============================================================================
// Host-side launch function (multi-block version)
// block_num: number of parallel blocks (default: COMPUTE_BLOCK_NUM)
// ============================================================================
void launchGemmCompute(uint8_t *shmem_output, uint8_t *src0, uint8_t *src1,
                       uint8_t *queue_set, int rank, void *stream, int block_num, uint32_t k_per_rank)
{
    GemmComputeKernel<<<block_num, nullptr, stream>>>(
        shmem_output, src0, src1, queue_set, rank, block_num, k_per_rank);
}
