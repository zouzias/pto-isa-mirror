#include <cstddef>
#include <cstdint>

#include <pto/pto-inst.hpp>

#ifdef __CCE_AICORE__
#include "pto/comm/pto_comm_inst.hpp"
#include "pto/common/pto_tile.hpp"
#endif
#include "common.hpp"
#include "ready_queue.hpp"

#ifndef CONFIG_G_M
#define CONFIG_G_M 2048
#endif
#ifndef CONFIG_G_K
#define CONFIG_G_K 2048
#endif
#ifndef CONFIG_G_N
#define CONFIG_G_N 1024
#endif

constexpr uint32_t G_M = CONFIG_G_M;
constexpr uint32_t G_K = CONFIG_G_K;
constexpr uint32_t G_N = CONFIG_G_N;

constexpr uint32_t G_BASE_M = 128;
constexpr uint32_t G_BASE_K = 64;
constexpr uint32_t G_BASE_N = 256;

#ifndef CONFIG_COMM_BLOCK_NUM
#define CONFIG_COMM_BLOCK_NUM 4
#endif
constexpr int COMM_BLOCK_NUM = CONFIG_COMM_BLOCK_NUM;

// ============================================================================
// CommAIVRoleStreamingParallel（M-slice）
//
// 每个 src(rank) 仅持有其本地 row-group 区间 [rank*m_tiles_local, (rank+1)*m_tiles_local)。
// 对每个 src，block 编码：block_idx = mi_local * k_chunks + kb。
// ============================================================================
AICORE inline void CommAIVRoleStreamingParallel(
    __gm__ half* shmem_input,
    __gm__ TileFlagMatrix* tile_flags,
    __gm__ HcclDeviceContext* hcclCtx,
    int block_idx,
    int num_blocks)
{
    int my_rank = static_cast<int>(hcclCtx->rankId);
    int n_ranks = static_cast<int>(hcclCtx->rankNum);
    int num_remote_ranks = n_ranks - 1;

    int m_tiles = static_cast<int>(G_M / G_BASE_M);
    int m_tiles_local = m_tiles / n_ranks;
    int k_chunks = static_cast<int>(G_K / G_BASE_N);
    int num_blocks_per_src = m_tiles_local * k_chunks;

    if (num_remote_ranks <= 0 || num_blocks <= 0) {
        return;
    }

    volatile __gm__ TileFlagMatrix* flags = reinterpret_cast<volatile __gm__ TileFlagMatrix*>(tile_flags);
    // 从 tile_flags 结构体读取动态计算的 tile_size 和 num_tiles
    int tile_size = flags->tile_size;
    int num_tiles = flags->num_tiles_per_src;
    volatile __gm__ int32_t* summary_base = GetSummaryBase(flags);

    // Block 0 负责置本 rank 的 local tile 就绪，并写本地 summary[my_rank]=num_tiles 供 AIC 先轮询 summary
    if (block_idx == 0) {
        for (int c = 0; c < num_tiles; ++c) {
            SetTileFlagReady(flags, my_rank, c);
        }
        SetLocalSummaryReady(summary_base, my_rank, num_tiles);
    }

    using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using Global = pto::GlobalTensor<half, ShapeDyn, StrideDyn, pto::Layout::ND>;
    using TileData = pto::Tile<pto::TileType::Vec, half, G_BASE_M, G_BASE_N,
                               pto::BLayout::RowMajor, -1, -1>;

    constexpr size_t tileUBBytes = ((G_BASE_M * G_BASE_N * sizeof(half) + 1023) / 1024) * 1024;
    TileData pingTile(G_BASE_M, G_BASE_N);
    TileData pongTile(G_BASE_M, G_BASE_N);
    TASSIGN(pingTile, 0x0);
    TASSIGN(pongTile, tileUBBytes);

    ShapeDyn tileShape(1, 1, 1, G_BASE_M, G_BASE_N);
    StrideDyn tileStride(G_BASE_M * G_K, G_BASE_M * G_K, G_BASE_M * G_K, G_K, 1);

    if (num_blocks < num_remote_ranks) {
        // block 数少于 dest 数：按 work_id 均分，work_id = dest_idx * num_tiles + tile_idx
        int total_work = num_remote_ranks * num_tiles;
        for (int work_id = block_idx; work_id < total_work; work_id += num_blocks) {
            int dest_idx = work_id / num_tiles;
            int tile_idx = work_id % num_tiles;

            int dest_rank = (dest_idx < my_rank) ? dest_idx : (dest_idx + 1);
            __gm__ TileFlagMatrix* remote_tile_flags =
                reinterpret_cast<__gm__ TileFlagMatrix*>(HcclRemotePtr(hcclCtx, tile_flags, dest_rank));
            __gm__ int32_t* remote_summary_src =
                reinterpret_cast<__gm__ int32_t*>(reinterpret_cast<__gm__ uint8_t*>(HcclRemotePtr(hcclCtx, tile_flags, dest_rank)) + TileFlagMatrixBytes(flags))
                + my_rank;

            int tile_start_blk = tile_idx * tile_size;
            int tile_end_blk = tile_start_blk + tile_size;
            if (tile_end_blk > num_blocks_per_src) {
                tile_end_blk = num_blocks_per_src;
            }
            for (int b = tile_start_blk; b < tile_end_blk; ++b) {
                int mi_local = b / k_chunks;
                int kb = b % k_chunks;
                int mi_global = my_rank * m_tiles_local + mi_local;
                uint64_t col_off = static_cast<uint64_t>(kb * static_cast<int>(G_BASE_N));
                uint64_t row_off = static_cast<uint64_t>(mi_global) * G_BASE_M;
                uint64_t offset = row_off * G_K + col_off;
                Global srcG(shmem_input + offset, tileShape, tileStride);
                __gm__ half* remote_input = HcclRemotePtr(hcclCtx, shmem_input, dest_rank);
                Global dstG(remote_input + offset, tileShape, tileStride);
                pto::comm::TPUT(dstG, srcG, pingTile, pongTile);
            }
            if (tile_idx < num_tiles) {
                SetRemoteTileFlagReady(remote_tile_flags, my_rank, tile_idx, remote_summary_src);
            }
        }
        return;
    }

    // num_blocks >= num_remote_ranks：每 dest 分配相同数量 block，按连续区间划分
    int blocks_per_dest = num_blocks / num_remote_ranks;
    if (blocks_per_dest <= 0) {
        blocks_per_dest = 1;
    }
    int dest_idx = block_idx / blocks_per_dest;
    int local_idx = block_idx % blocks_per_dest;
    if (dest_idx >= num_remote_ranks) {
        return;
    }

    int dest_rank = (dest_idx < my_rank) ? dest_idx : (dest_idx + 1);
    __gm__ TileFlagMatrix* remote_tile_flags =
        reinterpret_cast<__gm__ TileFlagMatrix*>(HcclRemotePtr(hcclCtx, tile_flags, dest_rank));
    __gm__ int32_t* remote_summary_base =
        reinterpret_cast<__gm__ int32_t*>(reinterpret_cast<__gm__ uint8_t*>(HcclRemotePtr(hcclCtx, tile_flags, dest_rank)) + TileFlagMatrixBytes(flags));
    __gm__ int32_t* remote_summary_src = remote_summary_base + my_rank;

    int tiles_per_block = (num_tiles + blocks_per_dest - 1) / blocks_per_dest;
    int tile_start = local_idx * tiles_per_block;
    int tile_end = tile_start + tiles_per_block;
    if (tile_end > num_tiles) {
        tile_end = num_tiles;
    }

    for (int tile_idx = tile_start; tile_idx < tile_end; ++tile_idx) {
        int blk_start = tile_idx * tile_size;
        int blk_end = blk_start + tile_size;
        if (blk_end > num_blocks_per_src) {
            blk_end = num_blocks_per_src;
        }
        for (int b = blk_start; b < blk_end; ++b) {
            int mi_local = b / k_chunks;
            int kb = b % k_chunks;
            int mi_global = my_rank * m_tiles_local + mi_local;
            uint64_t col_off = static_cast<uint64_t>(kb * static_cast<int>(G_BASE_N));
            uint64_t row_off = static_cast<uint64_t>(mi_global) * G_BASE_M;
            uint64_t offset = row_off * G_K + col_off;
            Global srcG(shmem_input + offset, tileShape, tileStride);
            __gm__ half* remote_input = HcclRemotePtr(hcclCtx, shmem_input, dest_rank);
            Global dstG(remote_input + offset, tileShape, tileStride);
            pto::comm::TPUT(dstG, srcG, pingTile, pongTile);
        }
        SetRemoteTileFlagReady(remote_tile_flags, my_rank, tile_idx, remote_summary_src);
    }
}

// ============================================================================
// STREAMING Kernel: AIV 并行，每个 dest rank 分配相同数量 block，总 block 数尽量用满 48
// ============================================================================
__global__ AICORE void RingCommStreamingKernel(
    __gm__ uint8_t* shmem_input,
    __gm__ uint8_t* tile_flags,
    __gm__ uint8_t* hccl_ctx_raw,
    int block_num)
{
    int block_idx = get_block_idx();
    __gm__ HcclDeviceContext* hcclCtx = reinterpret_cast<__gm__ HcclDeviceContext*>(hccl_ctx_raw);
    int n_ranks = static_cast<int>(hcclCtx->rankNum);
    int num_remote_ranks = n_ranks - 1;

    if (num_remote_ranks <= 0) {
        return;
    }

    CommAIVRoleStreamingParallel(
        reinterpret_cast<__gm__ half*>(shmem_input),
        reinterpret_cast<__gm__ TileFlagMatrix*>(tile_flags),
        hcclCtx,
        block_idx,
        block_num);
}

void launchRingCommStreaming(
    uint8_t* shmem_input,
    uint8_t* tile_flags,
    uint8_t* hccl_ctx,
    int n_ranks,
    void* stream)
{
    int num_remote_ranks = n_ranks - 1;
    if (num_remote_ranks <= 0) {
        return;
    }
    int blocks_per_dest = COMM_BLOCK_NUM / num_remote_ranks;
    if (blocks_per_dest < 1) {
        blocks_per_dest = 1;
    }
    int total_blocks = num_remote_ranks * blocks_per_dest;
    RingCommStreamingKernel<<<total_blocks, nullptr, stream>>>(
        shmem_input, tile_flags, hccl_ctx, total_blocks);
}
