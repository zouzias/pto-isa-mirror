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
// Two-phase kernel: RS (AtomicAdd) → device barrier → AG
// RS uses TPUT<AtomicAdd> to accumulate directly at the owner's reduced_output,
// eliminating the separate Reduce phase and its barrier.
//
// Signal matrix layout in HCCL window (per rank):
//   [0 .. MAX_RANKS-1]   Phase 0 cross-rank counters (RS done)
//   [MAX_RANKS]           Phase 0 local broadcast flag
//
// Only block_idx==0 performs cross-rank TNOTIFY/TWAIT signaling.

#ifndef PIPE_FIX
#define PIPE_FIX static_cast<pipe_t>(10)
#endif

#include <cstddef>
#include <cstdint>

#include "pto/comm/pto_comm_inst.hpp"
#include "pto/common/pto_tile.hpp"
#include <pto/pto-inst.hpp>

#include "common.hpp"
#include "ready_queue.hpp"

#include "gemm_ar_config.h"

// Signal matrix layout (per rank, in HCCL RDMA window):
//   [0 .. MAX_RANKS-1]              Phase 0 cross-rank counters (RS done)
//   [MAX_RANKS]                     Phase 0 local broadcast flag (block 0 -> all blocks)
static constexpr int SIGNAL_LOCAL_FLAG_OFFSET = MAX_RANKS;

// ============================================================================
// Device-side cross-rank barrier using TNOTIFY/TWAIT
//
// Block 0: performs cross-rank signaling then sets a local broadcast flag.
// Other blocks: wait on the local broadcast flag via TWAIT.
// ============================================================================
AICORE inline void DeviceBarrier(
    __gm__ HcclDeviceContext *hcclCtx,
    __gm__ int32_t *signal_base,
    int phase,
    int my_rank,
    int nranks,
    int block_idx,
    int32_t expected = 1)
{
    pipe_barrier(PIPE_ALL);

    if (block_idx == 0) {
        __gm__ int32_t *phase_base = signal_base + phase * MAX_RANKS;

        for (int r = 0; r < nranks; r++) {
            if (r == my_rank) continue;
            __gm__ int32_t *remote_sig = HcclRemotePtr(hcclCtx, phase_base + my_rank, r);
            pto::comm::Signal sig(remote_sig);
            pto::comm::TNOTIFY(sig, (int32_t)1, pto::comm::NotifyOp::AtomicAdd);
        }

        for (int r = 0; r < nranks; r++) {
            if (r == my_rank) continue;
            pto::comm::Signal sig(phase_base + r);
            pto::comm::TWAIT(sig, expected, pto::comm::WaitCmp::GE);
        }

        __gm__ int32_t *local_flag = signal_base + SIGNAL_LOCAL_FLAG_OFFSET + phase;
        pto::comm::Signal localSig(local_flag);
        pto::comm::TNOTIFY(localSig, expected, pto::comm::NotifyOp::Set);
    } else {
        __gm__ int32_t *local_flag = signal_base + SIGNAL_LOCAL_FLAG_OFFSET + phase;
        pto::comm::Signal localSig(local_flag);
        pto::comm::TWAIT(localSig, expected, pto::comm::WaitCmp::GE);
    }

    pipe_barrier(PIPE_ALL);
}

// ============================================================================
// Two-phase AllReduce kernel: RS (AtomicAdd) + AG in a single kernel launch
// ============================================================================
AICORE inline void GemmCommAllImpl(
    __gm__ half *gemm_output,
    __gm__ half *reduced_output,
    __gm__ int32_t *signal_matrix,
    __gm__ MultiBlockQueueSet *queue_set,
    __gm__ HcclDeviceContext *hcclCtx,
    int rank,
    int nranks,
    int num_compute_blocks,
    int block_idx,
    int num_comm_blocks)
{
    int my_rank = hcclCtx->rankId;

    using ShapeDyn  = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using Global    = pto::GlobalTensor<half, ShapeDyn, StrideDyn, pto::Layout::ND>;

    using TileData = pto::Tile<pto::TileType::Vec, half, G_BASE_M, G_BASE_N,
                               pto::BLayout::RowMajor, -1, -1>;

    constexpr size_t tileUBBytes = ((G_BASE_M * G_BASE_N * sizeof(half) + 1023) / 1024) * 1024;

    TileData pingTile(G_BASE_M, G_BASE_N);
    TileData pongTile(G_BASE_M, G_BASE_N);
    TASSIGN(pingTile, 0x0);
    TASSIGN(pongTile, tileUBBytes);

    ShapeDyn tileShape(1, 1, 1, G_BASE_M, G_BASE_N);
    StrideDyn tileStride(G_BASE_M * G_N, G_BASE_M * G_N, G_BASE_M * G_N, G_N, 1);

    // RS uses only the first num_compute_blocks AIVs (1:1 with compute queues).
    // Remaining AIVs skip RS to avoid HBM bandwidth contention with AIC MTE2.
    const int rs_active_blocks = num_compute_blocks;

    // ========================================================================
    // Phase 1: ReduceScatter — TPUT with AtomicAdd to owner's reduced_output
    //
    // Only block 0..(num_compute_blocks-1) participate.
    // Blocks >= num_compute_blocks skip straight to the barrier.
    // ========================================================================
    if (block_idx < rs_active_blocks) {
        volatile __gm__ MultiBlockQueueSet *qset = (volatile __gm__ MultiBlockQueueSet *)queue_set;

        int32_t heads[MAX_COMPUTE_BLOCKS];
        for (int b = 0; b < MAX_COMPUTE_BLOCKS; b++) heads[b] = 0;

        int32_t tiles_sent = 0;
        const int total_tiles = G_NUM_TILES;

        int my_queue_indices[MAX_COMPUTE_BLOCKS];
        int my_queue_count = 0;
        for (int q = 0; q < num_compute_blocks; q++) {
            if (q % rs_active_blocks == block_idx) {
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

        int pp_count = 0;
        Global pp_pending_dst(gemm_output, tileShape, tileStride);

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
                uint32_t mi = tile_idx / G_N_TILES;
                uint32_t ni = tile_idx % G_N_TILES;
                uint64_t tile_offset = (uint64_t)(mi * G_BASE_M) * G_N + ni * G_BASE_N;

                Global srcG(gemm_output + tile_offset, tileShape, tileStride);

                __gm__ half *dst_ptr;
                if (owner == my_rank) {
                    dst_ptr = reduced_output + tile_offset;
                } else {
                    dst_ptr = HcclRemotePtr(hcclCtx, reduced_output, owner) + tile_offset;
                }
                Global dstG(dst_ptr, tileShape, tileStride);

                bool use_ping = (pp_count % 2 == 0);
                TileData &curTile = use_ping ? pingTile : pongTile;
                event_t curEv = use_ping ? EVENT_ID0 : EVENT_ID1;

                if (pp_count == 0) {
                    TLOAD(curTile, srcG);
                    set_flag(PIPE_MTE2, PIPE_MTE3, curEv);
                } else {
                    TileData &prevTile = use_ping ? pongTile : pingTile;
                    event_t prevEv = use_ping ? EVENT_ID1 : EVENT_ID0;

                    wait_flag(PIPE_MTE2, PIPE_MTE3, prevEv);
                    TSTORE_IMPL<TileData, Global, pto::AtomicType::AtomicAdd>(pp_pending_dst, prevTile);
                    TLOAD(curTile, srcG);
                    set_flag(PIPE_MTE3, PIPE_MTE2, prevEv);
                    set_flag(PIPE_MTE2, PIPE_MTE3, curEv);
                    wait_flag(PIPE_MTE3, PIPE_MTE2, prevEv);
                }

                pp_pending_dst = dstG;
                pp_count++;

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

        if (pp_count > 0) {
            bool last_was_ping = ((pp_count - 1) % 2 == 0);
            TileData &lastTile = last_was_ping ? pingTile : pongTile;
            event_t lastEv = last_was_ping ? EVENT_ID0 : EVENT_ID1;
            wait_flag(PIPE_MTE2, PIPE_MTE3, lastEv);
            TSTORE_IMPL<TileData, Global, pto::AtomicType::AtomicAdd>(pp_pending_dst, lastTile);
            set_flag(PIPE_MTE3, PIPE_MTE2, lastEv);
            wait_flag(PIPE_MTE3, PIPE_MTE2, lastEv);
        }

        pipe_barrier(PIPE_ALL);
    }

    // ========================================================================
    // Device-side barrier: wait for all ranks to complete RS
    // ========================================================================
    DeviceBarrier(hcclCtx, signal_matrix, 0, my_rank, nranks, block_idx);

    // ========================================================================
    // Phase 2: AllGather — row-level flattened decomposition
    //
    // Flatten all AG work into rows:
    //   total_rows = my_tile_count * (nranks-1) * G_BASE_M
    // Each AIV block handles an equal-sized contiguous slice of rows.
    // Within each work row we recover (tile_owner_idx, remote_rank, row_in_tile)
    // and issue sub-tile TLOAD/TSTORE covering only the assigned rows.
    //
    // This ensures every AIV transfers the exact same amount of data,
    // eliminating the ±1 work-item imbalance of tile-level decomposition.
    // ========================================================================
    {
        const int total_tiles = G_NUM_TILES;
        const int tiles_per_owner = (total_tiles + nranks - 1) / nranks;
        const int my_tile_count = (my_rank < total_tiles % nranks || total_tiles % nranks == 0)
                                  ? tiles_per_owner
                                  : (total_tiles / nranks);

        const int remotes = nranks - 1;
        constexpr int ROWS_PER_TILE = G_BASE_M;
        const int rows_per_transfer = my_tile_count * ROWS_PER_TILE;
        const int total_rows = rows_per_transfer * remotes;

        if (total_rows > 0) {
            const int rows_per_block = (total_rows + num_comm_blocks - 1) / num_comm_blocks;
            int row_start = block_idx * rows_per_block;
            int row_end = (block_idx + 1) * rows_per_block;
            if (row_end > total_rows) row_end = total_rows;

            int cur_row = row_start;
            while (cur_row < row_end) {
                int flat_transfer = cur_row / ROWS_PER_TILE;
                int row_in_tile = cur_row % ROWS_PER_TILE;

                int oi = flat_transfer / remotes;
                int remote_idx = flat_transfer % remotes;

                int t = my_rank + oi * nranks;
                if (t >= total_tiles) break;

                int r = remote_idx;
                if (r >= my_rank) r++;

                int rows_left_in_transfer = ROWS_PER_TILE - row_in_tile;
                int rows_left_for_me = row_end - cur_row;
                int nrows = rows_left_in_transfer;
                if (nrows > rows_left_for_me) nrows = rows_left_for_me;

                uint32_t mi = t / G_N_TILES;
                uint32_t ni = t % G_N_TILES;
                uint64_t tile_base = (uint64_t)(mi * G_BASE_M) * G_N + ni * G_BASE_N;
                uint64_t row_offset = tile_base + (uint64_t)row_in_tile * G_N;

                ShapeDyn subShape(1, 1, 1, nrows, G_BASE_N);

                Global srcG(reduced_output + row_offset, subShape, tileStride);

                using SubTile = pto::Tile<pto::TileType::Vec, half, G_BASE_M, G_BASE_N,
                                          pto::BLayout::RowMajor, -1, -1>;
                SubTile subTile(nrows, G_BASE_N);
                TASSIGN(subTile, 0x0);

                TLOAD(subTile, srcG);
                set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
                wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);

                __gm__ half *dst_ptr = HcclRemotePtr(hcclCtx, reduced_output, r) + row_offset;
                Global dstG(dst_ptr, subShape, tileStride);
                TSTORE_IMPL<SubTile, Global, pto::AtomicType::AtomicNone>(dstG, subTile);
                set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
                wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);

                cur_row += nrows;
            }
        }

        pipe_barrier(PIPE_ALL);
    }
}

// ============================================================================
// Kernel entry point
// ============================================================================
__global__ AICORE void GemmCommAllKernel(
    __gm__ uint8_t *gemm_output,
    __gm__ uint8_t *reduced_output,
    __gm__ uint8_t *signal_matrix,
    __gm__ uint8_t *queue_set,
    __gm__ uint8_t *hcclCtx,
    int rank,
    int nranks,
    int num_compute_blocks,
    int num_comm_blocks)
{
    GemmCommAllImpl(
        reinterpret_cast<__gm__ half *>(gemm_output),
        reinterpret_cast<__gm__ half *>(reduced_output),
        reinterpret_cast<__gm__ int32_t *>(signal_matrix),
        reinterpret_cast<__gm__ MultiBlockQueueSet *>(queue_set),
        reinterpret_cast<__gm__ HcclDeviceContext *>(hcclCtx),
        rank, nranks, num_compute_blocks, get_block_idx(), num_comm_blocks);
}

// ============================================================================
// Host-side kernel launcher
// ============================================================================
void launchGemmCommAll(uint8_t *gemm_output,
                       uint8_t *reduced_output, uint8_t *signal_matrix,
                       uint8_t *queue_set, uint8_t *hcclCtx,
                       int rank, int nranks, void *stream, int num_compute_blocks)
{
    GemmCommAllKernel<<<COMM_BLOCK_NUM, nullptr, stream>>>(
        gemm_output, reduced_output, signal_matrix,
        queue_set, hcclCtx, rank, nranks, num_compute_blocks, COMM_BLOCK_NUM);
}
