/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// Communication Kernel (Vec Arch) for GEMM + AllReduce — HCCL backend
//
// AllReduce via ReduceScatter + AllGather, split into 3 kernel launches
// with host-side HcclHostBarrier between phases:
//
//   Kernel 1 (RS):  Poll compute queues, plain TPUT each tile to its owner rank
//   Host barrier:   HcclHostBarrier + aclrtSynchronizeStream
//   Kernel 2 (RED): Local TREDUCE_PINGPONG on owned tiles (half-height 64x256)
//   Host barrier:   HcclHostBarrier + aclrtSynchronizeStream
//   Kernel 3 (AG):  AllGather TPUT reduced tiles to all ranks
//
// TREDUCE_PINGPONG alternates EVENT_ID1/EVENT_ID2 to avoid V-pipeline
// event counter overflow (2-bit counters max at 1 per event).

#ifndef PIPE_FIX
#define PIPE_FIX static_cast<pipe_t>(10)
#endif

#include <cstddef>
#include <cstdint>

// PTO headers first (define AICORE macro and PTO types)
#include "pto/comm/pto_comm_inst.hpp"
#include "pto/common/pto_tile.hpp"
#include <pto/pto-inst.hpp>

// HCCL wrappers: MUST come before <iostream> to avoid kernel_operator.h <-> std::dec conflict
#include "common.hpp"
#include "ready_queue.hpp"

#include "gemm_ar_config.h"

// ============================================================================
// Phase 1 Kernel: ReduceScatter — TPUT each tile to its owner rank
// ============================================================================
AICORE inline void GemmCommRSImpl(
    __gm__ float *gemm_output,
    __gm__ float *recv_buffers,
    __gm__ MultiBlockQueueSet *queue_set,
    __gm__ HcclDeviceContext *hcclCtx,
    int rank,
    int nranks,
    int num_compute_blocks,
    int comm_block_idx,
    int num_comm_blocks)
{
    int my_rank = hcclCtx->rankId;
    const uint64_t output_size = (uint64_t)G_M * G_N;

    using ShapeDyn  = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using Global    = pto::GlobalTensor<float, ShapeDyn, StrideDyn, pto::Layout::ND>;

    using TileData = pto::Tile<pto::TileType::Vec, float, G_BASE_M, G_BASE_N,
                               pto::BLayout::RowMajor, -1, -1>;

    constexpr size_t tileUBBytes = ((G_BASE_M * G_BASE_N * sizeof(float) + 1023) / 1024) * 1024;

    TileData pingTile(G_BASE_M, G_BASE_N);
    TileData pongTile(G_BASE_M, G_BASE_N);
    TASSIGN(pingTile, 0x0);
    TASSIGN(pongTile, tileUBBytes);

    ShapeDyn tileShape(1, 1, 1, G_BASE_M, G_BASE_N);
    StrideDyn tileStride(G_BASE_M * G_N, G_BASE_M * G_N, G_BASE_M * G_N, G_N, 1);

    volatile __gm__ MultiBlockQueueSet *qset = (volatile __gm__ MultiBlockQueueSet *)queue_set;

    int32_t heads[MAX_COMPUTE_BLOCKS];
    for (int b = 0; b < MAX_COMPUTE_BLOCKS; b++) heads[b] = 0;

    int32_t tiles_sent = 0;
    const int total_tiles = G_NUM_TILES;

    int my_queue_indices[MAX_COMPUTE_BLOCKS];
    int my_queue_count = 0;
    for (int q = 0; q < num_compute_blocks; q++) {
        if (q % num_comm_blocks == comm_block_idx) {
            my_queue_indices[my_queue_count++] = q;
        }
    }

    const int tiles_per_compute_block = (total_tiles + num_compute_blocks - 1) / num_compute_blocks;

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

    int next_queue_offset = 0;

    while (tiles_sent < my_expected_tiles) {
        int32_t tile_idx = -1;
        for (int i = 0; i < my_queue_count; i++) {
            int local_idx = (next_queue_offset + i) % my_queue_count;
            int32_t q = my_queue_indices[local_idx];

            if (heads[q] >= queue_max_tiles[q]) continue;

            volatile __gm__ PerBlockQueue* pq = GetMyBlockQueue(qset, q);
            int32_t tile = PerBlockQueueTryDequeue(pq, heads[q]);

            if (tile >= 0) {
                heads[q]++;
                next_queue_offset = (local_idx + 1) % my_queue_count;
                tile_idx = tile;
                break;
            }
        }

        if (tile_idx >= 0) {
            int owner = tile_idx % nranks;

            if (owner != my_rank) {
                uint32_t mi = tile_idx / G_N_TILES;
                uint32_t ni = tile_idx % G_N_TILES;
                uint64_t tile_offset = (uint64_t)(mi * G_BASE_M) * G_N + ni * G_BASE_N;

                Global srcG(gemm_output + tile_offset, tileShape, tileStride);
                __gm__ float *dst_ptr = HcclRemotePtr(hcclCtx, recv_buffers, owner)
                                      + (uint64_t)my_rank * output_size + tile_offset;
                Global dstG(dst_ptr, tileShape, tileStride);
                pto::comm::TPUT(dstG, srcG, pingTile, pongTile);
            }

            tiles_sent++;
        } else {
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
                volatile __gm__ PerBlockQueue* pq = GetMyBlockQueue(qset, wait_queue);
                pto::comm::Signal sig(const_cast<__gm__ int32_t*>(&pq->count));
                pto::comm::TWAIT(sig, heads[wait_queue] + 1, pto::comm::WaitCmp::GE);
            }
        }
    }

    pipe_barrier(PIPE_ALL);
}

// ============================================================================
// Phase 2 Kernel: Local TREDUCE for owned tiles
// ============================================================================
AICORE inline void GemmCommReduceImpl(
    __gm__ float *gemm_output,
    __gm__ float *recv_buffers,
    __gm__ float *reduced_output,
    __gm__ HcclDeviceContext *hcclCtx,
    int rank,
    int nranks,
    int comm_block_idx,
    int num_comm_blocks)
{
    int my_rank = hcclCtx->rankId;
    const uint64_t output_size = (uint64_t)G_M * G_N;

    using ShapeDyn  = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using Global    = pto::GlobalTensor<float, ShapeDyn, StrideDyn, pto::Layout::ND>;

    StrideDyn tileStride(G_BASE_M * G_N, G_BASE_M * G_N, G_BASE_M * G_N, G_N, 1);

    const int tiles_per_owner = G_NUM_TILES / nranks;
    {
        using HalfTile = pto::Tile<pto::TileType::Vec, float, G_BASE_M / 2, G_BASE_N,
                                   pto::BLayout::RowMajor, -1, -1>;
        constexpr size_t halfTileBytes = ((G_BASE_M / 2 * G_BASE_N * sizeof(float) + 1023) / 1024) * 1024;

        HalfTile accTile(G_BASE_M / 2, G_BASE_N);
        HalfTile pingReduceTile(G_BASE_M / 2, G_BASE_N);
        HalfTile pongReduceTile(G_BASE_M / 2, G_BASE_N);
        TASSIGN(accTile,         0x0);
        TASSIGN(pingReduceTile,  halfTileBytes);
        TASSIGN(pongReduceTile,  halfTileBytes * 2);

        ShapeDyn halfShape(1, 1, 1, G_BASE_M / 2, G_BASE_N);

        int reduce_per_block = (tiles_per_owner + num_comm_blocks - 1) / num_comm_blocks;
        int oi_start = comm_block_idx * reduce_per_block;
        int oi_end = (comm_block_idx + 1) * reduce_per_block;
        if (oi_end > tiles_per_owner) oi_end = tiles_per_owner;

        for (int oi = oi_start; oi < oi_end; oi++) {
            int t = my_rank + oi * nranks;
            uint32_t mi = t / G_N_TILES;
            uint32_t ni = t % G_N_TILES;

            for (int half = 0; half < 2; half++) {
                uint64_t toff = (uint64_t)(mi * G_BASE_M + half * (G_BASE_M / 2)) * G_N
                              + ni * G_BASE_N;

                Global tensors[MAX_RANKS];
                for (int r = 0; r < nranks; ++r) {
                    if (r == my_rank) {
                        tensors[r] = Global(gemm_output + toff, halfShape, tileStride);
                    } else {
                        tensors[r] = Global(recv_buffers + (uint64_t)r * output_size + toff,
                                            halfShape, tileStride);
                    }
                }
                pto::comm::ParallelGroup<Global> pg(tensors, nranks, my_rank);

                Global dstG(reduced_output + toff, halfShape, tileStride);
                pto::comm::TREDUCE(pg, dstG, accTile, pingReduceTile, pongReduceTile,
                                   pto::comm::ReduceOp::Sum);
            }
        }
    }
    pipe_barrier(PIPE_ALL);
}

// ============================================================================
// Phase 3 Kernel: AllGather TPUT reduced tiles to all ranks
// ============================================================================
AICORE inline void GemmCommAGImpl(
    __gm__ float *reduced_output,
    __gm__ HcclDeviceContext *hcclCtx,
    int rank,
    int nranks,
    int comm_block_idx,
    int num_comm_blocks)
{
    int my_rank = hcclCtx->rankId;

    using ShapeDyn  = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using Global    = pto::GlobalTensor<float, ShapeDyn, StrideDyn, pto::Layout::ND>;

    using TileData = pto::Tile<pto::TileType::Vec, float, G_BASE_M, G_BASE_N,
                               pto::BLayout::RowMajor, -1, -1>;

    constexpr size_t tileUBBytes = ((G_BASE_M * G_BASE_N * sizeof(float) + 1023) / 1024) * 1024;

    TileData pingTile(G_BASE_M, G_BASE_N);
    TileData pongTile(G_BASE_M, G_BASE_N);
    TASSIGN(pingTile, 0x0);
    TASSIGN(pongTile, tileUBBytes);

    ShapeDyn tileShape(1, 1, 1, G_BASE_M, G_BASE_N);
    StrideDyn tileStride(G_BASE_M * G_N, G_BASE_M * G_N, G_BASE_M * G_N, G_N, 1);

    const int tiles_per_owner = G_NUM_TILES / nranks;
    {
        int reduce_per_block = (tiles_per_owner + num_comm_blocks - 1) / num_comm_blocks;
        int oi_start = comm_block_idx * reduce_per_block;
        int oi_end = (comm_block_idx + 1) * reduce_per_block;
        if (oi_end > tiles_per_owner) oi_end = tiles_per_owner;

        for (int oi = oi_start; oi < oi_end; oi++) {
            int t = my_rank + oi * nranks;
            uint32_t mi = t / G_N_TILES;
            uint32_t ni = t % G_N_TILES;
            uint64_t tile_offset = (uint64_t)(mi * G_BASE_M) * G_N + ni * G_BASE_N;

            Global srcG(reduced_output + tile_offset, tileShape, tileStride);

            for (int r = 0; r < nranks; r++) {
                if (r == my_rank) continue;
                __gm__ float *dst_ptr = HcclRemotePtr(hcclCtx, reduced_output, r) + tile_offset;
                Global dstG(dst_ptr, tileShape, tileStride);
                pto::comm::TPUT(dstG, srcG, pingTile, pongTile);
            }
        }
    }

    pipe_barrier(PIPE_ALL);
}

// ============================================================================
// Kernel entry points
// ============================================================================
__global__ AICORE void GemmCommRSKernel(
    __gm__ uint8_t *gemm_output,
    __gm__ uint8_t *recv_buffers,
    __gm__ uint8_t *queue_set,
    __gm__ uint8_t *hcclCtx,
    int rank,
    int nranks,
    int num_compute_blocks,
    int num_comm_blocks)
{
    GemmCommRSImpl(
        reinterpret_cast<__gm__ float *>(gemm_output),
        reinterpret_cast<__gm__ float *>(recv_buffers),
        reinterpret_cast<__gm__ MultiBlockQueueSet *>(queue_set),
        reinterpret_cast<__gm__ HcclDeviceContext *>(hcclCtx),
        rank, nranks, num_compute_blocks, get_block_idx(), num_comm_blocks);
}

__global__ AICORE void GemmCommReduceKernel(
    __gm__ uint8_t *gemm_output,
    __gm__ uint8_t *recv_buffers,
    __gm__ uint8_t *reduced_output,
    __gm__ uint8_t *hcclCtx,
    int rank,
    int nranks,
    int num_comm_blocks)
{
    GemmCommReduceImpl(
        reinterpret_cast<__gm__ float *>(gemm_output),
        reinterpret_cast<__gm__ float *>(recv_buffers),
        reinterpret_cast<__gm__ float *>(reduced_output),
        reinterpret_cast<__gm__ HcclDeviceContext *>(hcclCtx),
        rank, nranks, get_block_idx(), num_comm_blocks);
}

__global__ AICORE void GemmCommAGKernel(
    __gm__ uint8_t *reduced_output,
    __gm__ uint8_t *hcclCtx,
    int rank,
    int nranks,
    int num_comm_blocks)
{
    GemmCommAGImpl(
        reinterpret_cast<__gm__ float *>(reduced_output),
        reinterpret_cast<__gm__ HcclDeviceContext *>(hcclCtx),
        rank, nranks, get_block_idx(), num_comm_blocks);
}

// ============================================================================
// Host-side kernel launchers (called from main.cpp via extern linkage)
// ============================================================================
void launchGemmCommRS(uint8_t *gemm_output, uint8_t *recv_buffers,
                      uint8_t *queue_set, uint8_t *hcclCtx,
                      int rank, int nranks, void *stream, int num_compute_blocks)
{
    GemmCommRSKernel<<<COMM_BLOCK_NUM, nullptr, stream>>>(
        gemm_output, recv_buffers, queue_set, hcclCtx, rank, nranks, num_compute_blocks, COMM_BLOCK_NUM);
}

void launchGemmCommReduce(uint8_t *gemm_output, uint8_t *recv_buffers, uint8_t *reduced_output,
                          uint8_t *hcclCtx, int rank, int nranks, void *stream)
{
    GemmCommReduceKernel<<<COMM_BLOCK_NUM, nullptr, stream>>>(
        gemm_output, recv_buffers, reduced_output, hcclCtx, rank, nranks, COMM_BLOCK_NUM);
}

void launchGemmCommAG(uint8_t *reduced_output, uint8_t *hcclCtx,
                      int rank, int nranks, void *stream)
{
    GemmCommAGKernel<<<COMM_BLOCK_NUM, nullptr, stream>>>(
        reduced_output, hcclCtx, rank, nranks, COMM_BLOCK_NUM);
}

