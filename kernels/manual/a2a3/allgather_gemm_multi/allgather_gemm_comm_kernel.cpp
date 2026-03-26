#include <cstddef>
#include <cstdint>

#include <sys/wait.h>
#include <unistd.h>
#include <vector>
#include <string>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <tuple>

#include "pto/comm/pto_comm_inst.hpp"
#include "pto/common/pto_tile.hpp"
#include "common.hpp"
#include "ready_queue.hpp"

#ifdef DT_UNDEFINED
#define DT_UNDEFINED_SAVED DT_UNDEFINED
#undef DT_UNDEFINED
#endif
#include "test_common.h"
#ifdef DT_UNDEFINED_SAVED
#define DT_UNDEFINED DT_UNDEFINED_SAVED
#undef DT_UNDEFINED_SAVED
#endif

#include <pto/pto-inst.hpp>

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

#ifndef CONFIG_SIZE_NAME
#define CONFIG_SIZE_NAME "large"
#endif
constexpr const char* SIZE_NAME = CONFIG_SIZE_NAME;

constexpr uint32_t G_BASE_M = 128;
constexpr uint32_t G_BASE_K = 64;
constexpr uint32_t G_BASE_N = 256;

#ifndef CONFIG_COMPUTE_BLOCK_NUM
#define CONFIG_COMPUTE_BLOCK_NUM 20
#endif
#ifndef CONFIG_COMM_BLOCK_NUM
#define CONFIG_COMM_BLOCK_NUM 4
#endif
constexpr int COMPUTE_BLOCK_NUM = CONFIG_COMPUTE_BLOCK_NUM;
constexpr int COMM_BLOCK_NUM = CONFIG_COMM_BLOCK_NUM;

// ============================================================================
// Performance test configuration
// ============================================================================
constexpr int WARMUP_ITERS = 5;
constexpr int MEASURE_ITERS = 100;

// 多 AIV 时 ShmemDeviceQuiet() 易设备级死锁，可改为不调 Quiet，仅依赖 TPUT 与后续写 flag 的序
// （若运行时保证同 block 内 TPUT 先于远程写完成则正确；否则可能偶发 AIC 读到未写完的数据）
#ifndef AIV_SKIP_DEVICE_QUIET
#define AIV_SKIP_DEVICE_QUIET 1
#endif

// ============================================================================
// CommAIVRoleStreamingParallel（M-slice）
//
// 每个 src(rank) 仅持有其本地 row-group 区间 [rank*m_tiles_local, (rank+1)*m_tiles_local)。
// 对每个 src，block 编码：block_idx = mi_local * k_chunks + kb。
// ============================================================================
AICORE inline void CommAIVRoleStreamingParallel(
    __gm__ half* shmem_input,
    __gm__ TileFlagMatrix* tile_flags,
    int block_idx,
    int num_blocks)
{
    int my_rank = shmem_my_pe();
    int n_ranks = shmem_n_pes();
    int num_remote_ranks = n_ranks - 1;

    int m_tiles = static_cast<int>(G_M / G_BASE_M);
    int m_tiles_local = m_tiles / n_ranks;
    int k_chunks = static_cast<int>(G_K / G_BASE_N);
    int num_blocks_per_src = m_tiles_local * k_chunks;
    int num_tiles = (num_blocks_per_src + TILE_SIZE - 1) / TILE_SIZE;

    if (num_remote_ranks <= 0 || num_blocks <= 0) {
        return;
    }

    volatile __gm__ TileFlagMatrix* flags = reinterpret_cast<volatile __gm__ TileFlagMatrix*>(tile_flags);
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
                reinterpret_cast<__gm__ TileFlagMatrix*>(ShmemPtr(tile_flags, dest_rank));
            __gm__ int32_t* remote_summary_src =
                reinterpret_cast<__gm__ int32_t*>(reinterpret_cast<__gm__ uint8_t*>(ShmemPtr(tile_flags, dest_rank)) + TileFlagMatrixBytes(flags))
                + my_rank;

            int tile_start = tile_idx * TILE_SIZE;
            int tile_end = tile_start + TILE_SIZE;
            if (tile_end > num_blocks_per_src) {
                tile_end = num_blocks_per_src;
            }
            for (int b = tile_start; b < tile_end; ++b) {
                int mi_local = b / k_chunks;
                int kb = b % k_chunks;
                int mi_global = my_rank * m_tiles_local + mi_local;
                uint64_t col_off = static_cast<uint64_t>(kb * static_cast<int>(G_BASE_N));
                uint64_t row_off = static_cast<uint64_t>(mi_global) * G_BASE_M;
                uint64_t offset = row_off * G_K + col_off;
                Global srcG(shmem_input + offset, tileShape, tileStride);
                __gm__ half* remote_input = ShmemPtr(shmem_input, dest_rank);
                Global dstG(remote_input + offset, tileShape, tileStride);
                pto::comm::TPUT(dstG, srcG, pingTile, pongTile);
            }
#if !AIV_SKIP_DEVICE_QUIET
            ShmemDeviceQuiet();
#endif
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
        reinterpret_cast<__gm__ TileFlagMatrix*>(ShmemPtr(tile_flags, dest_rank));
    __gm__ int32_t* remote_summary_base =
        reinterpret_cast<__gm__ int32_t*>(reinterpret_cast<__gm__ uint8_t*>(ShmemPtr(tile_flags, dest_rank)) + TileFlagMatrixBytes(flags));
    __gm__ int32_t* remote_summary_src = remote_summary_base + my_rank;

    int tiles_per_block = (num_tiles + blocks_per_dest - 1) / blocks_per_dest;
    int tile_start = local_idx * tiles_per_block;
    int tile_end = tile_start + tiles_per_block;
    if (tile_end > num_tiles) {
        tile_end = num_tiles;
    }

    for (int tile_idx = tile_start; tile_idx < tile_end; ++tile_idx) {
        int blk_start = tile_idx * TILE_SIZE;
        int blk_end = blk_start + TILE_SIZE;
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
            __gm__ half* remote_input = ShmemPtr(shmem_input, dest_rank);
            Global dstG(remote_input + offset, tileShape, tileStride);
            pto::comm::TPUT(dstG, srcG, pingTile, pongTile);
        }
#if !AIV_SKIP_DEVICE_QUIET
        ShmemDeviceQuiet();
#endif
        SetRemoteTileFlagReady(remote_tile_flags, my_rank, tile_idx, remote_summary_src);
    }
}

// ============================================================================
// STREAMING Kernel: AIV 并行，每个 dest rank 分配相同数量 block，总 block 数尽量用满 48
// ============================================================================
__global__ AICORE void RingCommStreamingKernel(
    __gm__ uint8_t* shmem_input,
    __gm__ uint8_t* tile_flags,
    int block_num)
{
    int block_idx = get_block_idx();
    int n_ranks = shmem_n_pes();
    int num_remote_ranks = n_ranks - 1;

    if (num_remote_ranks <= 0) {
        return;
    }

    CommAIVRoleStreamingParallel(
        reinterpret_cast<__gm__ half*>(shmem_input),
        reinterpret_cast<__gm__ TileFlagMatrix*>(tile_flags),
        block_idx,
        block_num);
}

static void launchRingCommStreaming(
    uint8_t* shmem_input,
    uint8_t* tile_flags,
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
        shmem_input, tile_flags, total_blocks);
}

extern void launchAllGatherGemmComputeStreaming(
    uint8_t* output,
    uint8_t* shmem_input,
    uint8_t* src1,
    uint8_t* tile_flags,
    void* stream,
    int block_num);

static bool RunAllGatherGemmPerRank(int rank_id, int n_ranks, int device_id)
{
    if (G_M % G_BASE_M != 0 || G_N % G_BASE_N != 0 ||
        G_M % n_ranks != 0 || (G_M / n_ranks) % G_BASE_M != 0 ||
        G_K % G_BASE_N != 0) {
        if (rank_id == 0) {
            std::cerr << "[ERROR] Shape not aligned for ring-streaming path." << std::endl;
        }
        return false;
    }

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

    size_t inputShmemBytes = static_cast<size_t>(G_M) * G_K * sizeof(uint16_t);
    size_t heapBytes = 64ULL * 1024 * 1024;
    while (heapBytes < inputShmemBytes * 2) {
        heapBytes *= 2;
    }

    ShmemEnv env;
    env.rank = rank_id;
    env.size = n_ranks;
    env.ipPort = "tcp://127.0.0.1:8780";
    env.heapBytes = heapBytes;
    if (!ShmemInitFromEnv(env)) {
        std::cerr << "[ERROR] Rank " << rank_id << ": ShmemInitFromEnv failed\n";
        return false;
    }

    void* shmem_input = ShmemMalloc(inputShmemBytes);
    if (shmem_input == nullptr) {
        std::cerr << "[ERROR] Rank " << rank_id << ": ShmemMalloc input failed\n";
        return false;
    }
    aclrtMemset(shmem_input, inputShmemBytes, 0, inputShmemBytes);

    int m_tiles = static_cast<int>(G_M / G_BASE_M);
    int m_tiles_local = m_tiles / n_ranks;
    int k_chunks = static_cast<int>(G_K / G_BASE_N);
    int num_blocks_per_src = m_tiles_local * k_chunks;

    // Tile-based flag matrix + per-src summary (doorbell) for two-level polling
    size_t tileFlagMatrixSize = TileFlagMatrixSize(n_ranks, num_blocks_per_src, TILE_SIZE);
    size_t tileFlagWithSummarySize = TileFlagMatrixWithSummarySize(n_ranks, num_blocks_per_src, TILE_SIZE);
    void* tile_flag_shmem = ShmemMalloc(tileFlagWithSummarySize);
    TileFlagMatrix* tile_flag_host = nullptr;
    aclrtMallocHost(reinterpret_cast<void**>(&tile_flag_host), tileFlagWithSummarySize);
    TileFlagMatrixInit(tile_flag_host, n_ranks, num_blocks_per_src, TILE_SIZE);
    int32_t* summary_host = reinterpret_cast<int32_t*>(reinterpret_cast<uint8_t*>(tile_flag_host) + tileFlagMatrixSize);
    TileFlagMatrixSummaryInit(summary_host, n_ranks);
    aclrtMemcpy(tile_flag_shmem, tileFlagWithSummarySize, tile_flag_host, tileFlagWithSummarySize, ACL_MEMCPY_HOST_TO_DEVICE);

    size_t bSize = static_cast<size_t>(G_K) * G_N * sizeof(uint16_t);
    void* src1_dev = nullptr;
    aclrtMalloc(&src1_dev, bSize, ACL_MEM_MALLOC_HUGE_FIRST);

    size_t outputSize = static_cast<size_t>(G_M) * G_N * sizeof(float);
    void* output_dev = nullptr;
    aclrtMalloc(&output_dev, outputSize, ACL_MEM_MALLOC_HUGE_FIRST);

    std::string input_dir = std::string("../input/") + SIZE_NAME;
    std::string a_file = input_dir + "/a_rank" + std::to_string(rank_id) + ".bin";
    std::string b_file = input_dir + "/b.bin";

    size_t m_local = static_cast<size_t>(G_M) / n_ranks;
    size_t aLocalSize = m_local * G_K * sizeof(uint16_t);
    uint16_t* a_local_host = nullptr;
    uint16_t* b_host = nullptr;
    aclrtMallocHost(reinterpret_cast<void**>(&a_local_host), aLocalSize);
    aclrtMallocHost(reinterpret_cast<void**>(&b_host), bSize);

    size_t a_file_size = 0;
    size_t b_file_size = 0;
    if (!PtoTestCommon::ReadFile(a_file, a_file_size, a_local_host, aLocalSize) || a_file_size != aLocalSize) {
        std::cerr << "[ERROR] Rank " << rank_id << ": A file mismatch for M-slice: " << a_file << std::endl;
        return false;
    }
    if (!PtoTestCommon::ReadFile(b_file, b_file_size, b_host, bSize) || b_file_size != bSize) {
        std::cerr << "[ERROR] Rank " << rank_id << ": B file mismatch: " << b_file << std::endl;
        return false;
    }

    aclrtMemcpy(reinterpret_cast<uint8_t*>(shmem_input) + rank_id * aLocalSize,
                aLocalSize, a_local_host, aLocalSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src1_dev, bSize, b_host, bSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtFreeHost(a_local_host);
    aclrtFreeHost(b_host);

    ShmemBarrierAll();

    // ------------------------------------------------------------------
    // Helper lambdas (streaming only)
    // ------------------------------------------------------------------
    bool is_ok = true;

    auto resetStreamingState = [&]() {
        TileFlagMatrixReset(tile_flag_host);
        TileFlagMatrixSummaryInit(summary_host, n_ranks);
        aclrtMemcpy(tile_flag_shmem, tileFlagWithSummarySize, tile_flag_host, tileFlagWithSummarySize, ACL_MEMCPY_HOST_TO_DEVICE);
        aclrtMemset(output_dev, outputSize, 0, outputSize);
    };

    auto prePopulateTileFlags = [&]() {
        for (int src = 0; src < n_ranks; ++src) {
            TileFlagMatrixSetLocalReady(tile_flag_host, src);
        }
        int num_tiles = (num_blocks_per_src + TILE_SIZE - 1) / TILE_SIZE;
        for (int src = 0; src < n_ranks; ++src) {
            summary_host[src] = num_tiles;
        }
        aclrtMemcpy(tile_flag_shmem, tileFlagWithSummarySize, tile_flag_host, tileFlagWithSummarySize, ACL_MEMCPY_HOST_TO_DEVICE);
    };

    // ------------------------------------------------------------------
    // Warmup Phase 1: Comm kernel only (diagnose TPUT communication)
    // ------------------------------------------------------------------
    if (rank_id == 0) {
        std::cout << "\n[PERF] Starting comm-only warmup (" << WARMUP_ITERS << " iterations)..." << std::endl;
    }

    for (int i = 0; i < WARMUP_ITERS; ++i) {
        resetStreamingState();
        ShmemBarrierAll();

        launchRingCommStreaming(
            reinterpret_cast<uint8_t*>(shmem_input),
            reinterpret_cast<uint8_t*>(tile_flag_shmem),
            n_ranks,
            commStream);
        aclrtSynchronizeStream(commStream);

        if (rank_id == 0) std::cout << "  Comm warmup " << i << " done" << std::endl;
        ShmemBarrierAll();
    }

    // ------------------------------------------------------------------
    // Warmup Phase 2: Compute kernel only (pre-populate tile flags)
    // ------------------------------------------------------------------
    if (rank_id == 0) {
        std::cout << "[PERF] Starting compute-only warmup (" << WARMUP_ITERS << " iterations)..." << std::endl;
    }

    for (int i = 0; i < WARMUP_ITERS; ++i) {
        resetStreamingState();
        prePopulateTileFlags();
        ShmemBarrierAll();

        launchAllGatherGemmComputeStreaming(
            reinterpret_cast<uint8_t*>(output_dev),
            reinterpret_cast<uint8_t*>(shmem_input),
            reinterpret_cast<uint8_t*>(src1_dev),
            reinterpret_cast<uint8_t*>(tile_flag_shmem),
            computeStream,
            COMPUTE_BLOCK_NUM);
        aclrtSynchronizeStream(computeStream);

        if (rank_id == 0) std::cout << "  Compute warmup " << i << " done" << std::endl;
        ShmemBarrierAll();
    }

    // ------------------------------------------------------------------
    // Warmup Phase 3: Sequential (streaming comm then streaming compute)
    // ------------------------------------------------------------------
    if (rank_id == 0) {
        std::cout << "[PERF] Starting sequential warmup (" << WARMUP_ITERS << " iterations)..." << std::endl;
    }

    for (int i = 0; i < WARMUP_ITERS; ++i) {
        resetStreamingState();
        ShmemBarrierAll();

        launchRingCommStreaming(
            reinterpret_cast<uint8_t*>(shmem_input),
            reinterpret_cast<uint8_t*>(tile_flag_shmem),
            n_ranks,
            commStream);
        aclrtSynchronizeStream(commStream);
        ShmemBarrierAll();

        launchAllGatherGemmComputeStreaming(
            reinterpret_cast<uint8_t*>(output_dev),
            reinterpret_cast<uint8_t*>(shmem_input),
            reinterpret_cast<uint8_t*>(src1_dev),
            reinterpret_cast<uint8_t*>(tile_flag_shmem),
            computeStream,
            COMPUTE_BLOCK_NUM);
        aclrtSynchronizeStream(computeStream);

        if (rank_id == 0) std::cout << "  Sequential warmup " << i << " done" << std::endl;
        ShmemBarrierAll();
    }

    // ------------------------------------------------------------------
    // Warmup Phase 4: Streaming pipelined (concurrent comm + compute)
    // ------------------------------------------------------------------
    if (rank_id == 0) {
        std::cout << "[PERF] Starting streaming pipelined warmup (" << WARMUP_ITERS << " iterations)..." << std::endl;
    }

    for (int i = 0; i < WARMUP_ITERS; ++i) {
        resetStreamingState();
        ShmemBarrierAll();

        launchRingCommStreaming(
            reinterpret_cast<uint8_t*>(shmem_input),
            reinterpret_cast<uint8_t*>(tile_flag_shmem),
            n_ranks,
            commStream);
        launchAllGatherGemmComputeStreaming(
            reinterpret_cast<uint8_t*>(output_dev),
            reinterpret_cast<uint8_t*>(shmem_input),
            reinterpret_cast<uint8_t*>(src1_dev),
            reinterpret_cast<uint8_t*>(tile_flag_shmem),
            computeStream,
            COMPUTE_BLOCK_NUM);
        aclrtSynchronizeStream(computeStream);
        aclrtSynchronizeStream(commStream);

        if (rank_id == 0) std::cout << "  Streaming pipelined warmup " << i << " done" << std::endl;
        ShmemBarrierAll();
    }

    // ------------------------------------------------------------------
    // Functional verification after warmup (all ranks participate)
    // ------------------------------------------------------------------
    auto verifyAgainstGolden = [&](const char* phase_name) -> bool {
        float* verify_host = nullptr;
        aclrtMallocHost(reinterpret_cast<void**>(&verify_host), outputSize);
        aclrtMemcpy(verify_host, outputSize, output_dev, outputSize, ACL_MEMCPY_DEVICE_TO_HOST);

        std::string output_dir = std::string("../output/") + SIZE_NAME;
        std::string output_file = output_dir + "/output_rank" + std::to_string(rank_id) + ".bin";
        PtoTestCommon::WriteFile(output_file, verify_host, outputSize);

        std::string golden_file = output_dir + "/golden.bin";
        std::vector<float> golden(outputSize / sizeof(float));
        size_t golden_file_size = 0;
        bool rank_ok = false;
        if (PtoTestCommon::ReadFile(golden_file, golden_file_size, golden.data(), outputSize) && golden_file_size == outputSize) {
            rank_ok = PtoTestCommon::ResultCmp(golden, verify_host, 0.001f);
            if (rank_ok) {
                std::cout << "[INFO] Rank " << rank_id << " " << phase_name << " verification passed!" << std::endl;
            } else {
                std::cerr << "[ERROR] Rank " << rank_id << " " << phase_name << " verification FAILED!" << std::endl;
            }
        } else {
            std::cerr << "[ERROR] Rank " << rank_id << ": golden file not available or size mismatch: "
                      << golden_file << std::endl;
            rank_ok = false;
        }
        aclrtFreeHost(verify_host);
        return rank_ok;
    };

    {
        if (rank_id == 0) {
            std::cout << "\n[VERIFY] Running functional verification (all ranks)..." << std::endl;
        }
        resetStreamingState();
        ShmemBarrierAll();

        launchRingCommStreaming(
            reinterpret_cast<uint8_t*>(shmem_input),
            reinterpret_cast<uint8_t*>(tile_flag_shmem),
            n_ranks,
            commStream);
        launchAllGatherGemmComputeStreaming(
            reinterpret_cast<uint8_t*>(output_dev),
            reinterpret_cast<uint8_t*>(shmem_input),
            reinterpret_cast<uint8_t*>(src1_dev),
            reinterpret_cast<uint8_t*>(tile_flag_shmem),
            computeStream,
            COMPUTE_BLOCK_NUM);
        aclrtSynchronizeStream(computeStream);
        aclrtSynchronizeStream(commStream);
        ShmemBarrierAll();

        is_ok = verifyAgainstGolden("post-warmup");
        ShmemBarrierAll();

        if (!is_ok) {
            std::cerr << "[ERROR] Rank " << rank_id << ": Functional verification failed, aborting performance measurement." << std::endl;
        }
    }

    // ------------------------------------------------------------------
    // Performance measurement
    // ------------------------------------------------------------------
    if (rank_id == 0) {
        std::cout << "[PERF] Starting measurement (" << MEASURE_ITERS << " iterations)..." << std::endl;
    }

    ShmemBarrierAll();

    // Measure comm kernel time (streaming, standalone)
    std::vector<double> comm_times_us;
    for (int iter = 0; iter < MEASURE_ITERS; ++iter) {
        resetStreamingState();
        aclrtSynchronizeStream(commStream);
        ShmemBarrierAll();

        auto comm_start = std::chrono::high_resolution_clock::now();
        launchRingCommStreaming(
            reinterpret_cast<uint8_t*>(shmem_input),
            reinterpret_cast<uint8_t*>(tile_flag_shmem),
            n_ranks,
            commStream);
        aclrtSynchronizeStream(commStream);
        auto comm_end = std::chrono::high_resolution_clock::now();
        double comm_time = std::chrono::duration<double, std::micro>(comm_end - comm_start).count();
        comm_times_us.push_back(comm_time);
        ShmemBarrierAll();
    }

    // Measure compute kernel time (standalone, pre-populated tile flags)
    std::vector<double> compute_times_us;
    for (int iter = 0; iter < MEASURE_ITERS; ++iter) {
        resetStreamingState();
        prePopulateTileFlags();
        aclrtSynchronizeStream(computeStream);
        ShmemBarrierAll();

        auto compute_start = std::chrono::high_resolution_clock::now();
        launchAllGatherGemmComputeStreaming(
            reinterpret_cast<uint8_t*>(output_dev),
            reinterpret_cast<uint8_t*>(shmem_input),
            reinterpret_cast<uint8_t*>(src1_dev),
            reinterpret_cast<uint8_t*>(tile_flag_shmem),
            computeStream,
            COMPUTE_BLOCK_NUM);
        aclrtSynchronizeStream(computeStream);
        auto compute_end = std::chrono::high_resolution_clock::now();
        double compute_time = std::chrono::duration<double, std::micro>(compute_end - compute_start).count();
        compute_times_us.push_back(compute_time);
        ShmemBarrierAll();
    }

    // Measure sequential execution (streaming comm then streaming compute)
    std::vector<double> sequential_times_us;
    for (int iter = 0; iter < MEASURE_ITERS; ++iter) {
        resetStreamingState();
        aclrtSynchronizeStream(commStream);
        aclrtSynchronizeStream(computeStream);
        ShmemBarrierAll();

        auto seq_start = std::chrono::high_resolution_clock::now();
        launchRingCommStreaming(
            reinterpret_cast<uint8_t*>(shmem_input),
            reinterpret_cast<uint8_t*>(tile_flag_shmem),
            n_ranks,
            commStream);
        aclrtSynchronizeStream(commStream);
        launchAllGatherGemmComputeStreaming(
            reinterpret_cast<uint8_t*>(output_dev),
            reinterpret_cast<uint8_t*>(shmem_input),
            reinterpret_cast<uint8_t*>(src1_dev),
            reinterpret_cast<uint8_t*>(tile_flag_shmem),
            computeStream,
            COMPUTE_BLOCK_NUM);
        aclrtSynchronizeStream(computeStream);
        auto seq_end = std::chrono::high_resolution_clock::now();
        double seq_time_us = std::chrono::duration<double, std::micro>(seq_end - seq_start).count();
        sequential_times_us.push_back(seq_time_us);
        ShmemBarrierAll();
    }

    // ------------------------------------------------------------------
    // Measure STREAMING pipelined execution (fine-grained tile-based overlap)
    // ------------------------------------------------------------------
    std::vector<double> streaming_times_us;
    for (int iter = 0; iter < MEASURE_ITERS; ++iter) {
        resetStreamingState();
        aclrtSynchronizeStream(commStream);
        aclrtSynchronizeStream(computeStream);
        ShmemBarrierAll();

        auto stream_start = std::chrono::high_resolution_clock::now();
        launchRingCommStreaming(
            reinterpret_cast<uint8_t*>(shmem_input),
            reinterpret_cast<uint8_t*>(tile_flag_shmem),
            n_ranks,
            commStream);
        launchAllGatherGemmComputeStreaming(
            reinterpret_cast<uint8_t*>(output_dev),
            reinterpret_cast<uint8_t*>(shmem_input),
            reinterpret_cast<uint8_t*>(src1_dev),
            reinterpret_cast<uint8_t*>(tile_flag_shmem),
            computeStream,
            COMPUTE_BLOCK_NUM);
        aclrtSynchronizeStream(computeStream);
        aclrtSynchronizeStream(commStream);
        auto stream_end = std::chrono::high_resolution_clock::now();
        double stream_time_us = std::chrono::duration<double, std::micro>(stream_end - stream_start).count();
        streaming_times_us.push_back(stream_time_us);
        ShmemBarrierAll();
    }

    // ------------------------------------------------------------------
    // Final verification against golden data (all ranks)
    // ------------------------------------------------------------------
    ShmemBarrierAll();

    is_ok = verifyAgainstGolden("final");

    // ------------------------------------------------------------------
    // Performance statistics (rank 0 only)
    // ------------------------------------------------------------------
    if (rank_id == 0) {
        auto calc_stats = [](const std::vector<double>& times) {
            double sum = 0.0, min_val = times[0], max_val = times[0];
            for (double t : times) {
                sum += t;
                min_val = std::min(min_val, t);
                max_val = std::max(max_val, t);
            }
            double avg = sum / times.size();
            double variance = 0.0;
            for (double t : times) {
                variance += (t - avg) * (t - avg);
            }
            double std_dev = std::sqrt(variance / times.size());
            return std::make_tuple(avg, min_val, max_val, std_dev);
        };

        auto [seq_avg, seq_min, seq_max, seq_std] = calc_stats(sequential_times_us);
        auto [stream_avg, stream_min, stream_max, stream_std] = calc_stats(streaming_times_us);
        auto [compute_avg, compute_min, compute_max, compute_std] = calc_stats(compute_times_us);
        auto [comm_avg, comm_min, comm_max, comm_std] = calc_stats(comm_times_us);

        // GEMM FLOPs and TFLOPS calculation (TFLOPS = FLOPs / time_sec / 1e12)
        double gemm_flops = 2.0 * static_cast<double>(G_M) * static_cast<double>(G_K) * static_cast<double>(G_N);
        double compute_tflops = (compute_avg > 0.0) ? (gemm_flops / (compute_avg * 1e-6) / 1e12) : 0.0;
        double seq_tflops = (seq_avg > 0.0) ? (gemm_flops / (seq_avg * 1e-6) / 1e12) : 0.0;
        double stream_tflops = (stream_avg > 0.0) ? (gemm_flops / (stream_avg * 1e-6) / 1e12) : 0.0;

        // Communication bandwidth calculation (M-slice AllGather)
        double m_local = static_cast<double>(G_M) / n_ranks;
        double comm_bytes = m_local * static_cast<double>(G_K) * sizeof(uint16_t) * (n_ranks - 1);
        double comm_gbps = (comm_avg > 0.0) ? (comm_bytes / (comm_avg * 1e-6) / 1e9) : 0.0;

        std::cout << "\n================================================================" << std::endl;
        if (is_ok) {
            std::cout << "[SUCCESS] AllGather GEMM (M-slice) test PASSED!" << std::endl;
        } else {
            std::cout << "[FAILED] AllGather GEMM (M-slice) test FAILED!" << std::endl;
        }
        std::cout << "  Matrix dimensions: M=" << G_M << ", K=" << G_K << ", N=" << G_N << std::endl;
        std::cout << "  Size configuration: " << SIZE_NAME << std::endl;
        std::cout << "  Ranks: " << n_ranks << ", M_local per rank: " << static_cast<int>(m_local) << std::endl;
        std::cout << "  Signaling: TileFlagMatrix (M-slice streaming)" << std::endl;

        std::cout << "\n  Workload Analysis:" << std::endl;
        std::cout << "    GEMM FLOPs (per rank, full M): " << std::scientific << std::setprecision(2) << gemm_flops << std::endl;
        std::cout << "    Input A size (per rank M-slice):  " << std::fixed << std::setprecision(2)
                  << (m_local * static_cast<double>(G_K) * sizeof(uint16_t) / (1024.0 * 1024.0)) << " MB" << std::endl;
        std::cout << "    Input A size (global):            "
                  << (static_cast<double>(G_M) * G_K * sizeof(uint16_t) / (1024.0 * 1024.0)) << " MB" << std::endl;
        std::cout << "    Comm bytes (per rank):            " << (comm_bytes / (1024.0 * 1024.0))
                  << " MB (local_M_slice * (n_ranks-1))" << std::endl;
        std::cout << "    Output C size:                    "
                  << (static_cast<double>(G_M) * G_N * sizeof(float) / (1024.0 * 1024.0)) << " MB" << std::endl;

        std::cout << "\n[PERF] Performance Results (n_iters=" << MEASURE_ITERS << "):" << std::endl;
        std::cout << std::fixed << std::setprecision(3);

        std::cout << "\n  Comm Kernel (AllGather M-slice, streaming):" << std::endl;
        std::cout << "    Avg Time:  " << comm_avg << " us" << std::endl;
        std::cout << "    Min Time:  " << comm_min << " us" << std::endl;
        std::cout << "    Max Time:  " << comm_max << " us" << std::endl;
        std::cout << "    Std Dev:   " << comm_std << " us" << std::endl;
        std::cout << "    Bandwidth: " << comm_gbps << " GB/s" << std::endl;

        std::cout << "\n  Compute Kernel (GEMM, standalone with pre-populated tile flags):" << std::endl;
        std::cout << "    Avg Time:  " << compute_avg << " us" << std::endl;
        std::cout << "    Min Time:  " << compute_min << " us" << std::endl;
        std::cout << "    Max Time:  " << compute_max << " us" << std::endl;
        std::cout << "    Std Dev:   " << compute_std << " us" << std::endl;
        std::cout << "    Throughput: " << compute_tflops << " TFLOPS (per rank)" << std::endl;

        std::cout << "\n  Component Summary:" << std::endl;
        std::cout << "    Comm Avg:          " << comm_avg << " us" << std::endl;
        std::cout << "    Compute Avg:       " << compute_avg << " us" << std::endl;
        std::cout << "    Sequential Sum:    " << (comm_avg + compute_avg) << " us" << std::endl;

        std::cout << "\n  Sequential Execution (Comm -> Compute, no overlap):" << std::endl;
        std::cout << "    Avg Time:  " << seq_avg << " us" << std::endl;
        std::cout << "    Min Time:  " << seq_min << " us" << std::endl;
        std::cout << "    Max Time:  " << seq_max << " us" << std::endl;
        std::cout << "    Std Dev:   " << seq_std << " us" << std::endl;
        std::cout << "    Throughput: " << seq_tflops << " TFLOPS" << std::endl;

        std::cout << "\n  *** STREAMING Pipelined (Tile-based, fine-grained overlap): ***" << std::endl;
        std::cout << "    Tile size: " << TILE_SIZE << " blocks" << std::endl;
        std::cout << "    Avg Time:  " << stream_avg << " us" << std::endl;
        std::cout << "    Min Time:  " << stream_min << " us" << std::endl;
        std::cout << "    Max Time:  " << stream_max << " us" << std::endl;
        std::cout << "    Std Dev:   " << stream_std << " us" << std::endl;
        std::cout << "    Throughput: " << stream_tflops << " TFLOPS" << std::endl;

        double speedup_stream = seq_avg / stream_avg;
        std::cout << "\n  Performance Comparison:" << std::endl;
        std::cout << "    Speedup (Streaming vs Sequential):  " << speedup_stream << "x" << std::endl;
        std::cout << "    Time Saved (Streaming): " << (seq_avg - stream_avg) << " us ("
                  << ((seq_avg - stream_avg) / seq_avg * 100.0) << "%)" << std::endl;

        double theoretical_min = std::max(compute_avg, comm_avg);
        double max_possible_saved = seq_avg - theoretical_min;
        double stream_saved = seq_avg - stream_avg;
        double stream_overhead = stream_avg - theoretical_min;
        double stream_efficiency = 0.0;
        if (max_possible_saved > 0.001) {
            stream_efficiency = (stream_saved / max_possible_saved) * 100.0;
            stream_efficiency = std::max(0.0, std::min(100.0, stream_efficiency));
        } else {
            stream_efficiency = 100.0;
        }

        std::cout << "\n  Overlap Analysis (Streaming Pipelined):" << std::endl;
        std::cout << "    Sequential time:                      " << seq_avg << " us" << std::endl;
        std::cout << "    Theoretical min (max(compute, comm)): " << theoretical_min << " us" << std::endl;
        std::cout << "    Actual streaming time:                " << stream_avg << " us" << std::endl;
        std::cout << "    Time saved by streaming overlap:      " << stream_saved << " us" << std::endl;
        std::cout << "    Overhead (vs theoretical):            " << stream_overhead << " us" << std::endl;
        std::cout << "    Streaming overlap efficiency:         " << stream_efficiency << "%" << std::endl;

        if (stream_avg > compute_avg) {
            double comm_remaining = stream_avg - compute_avg;
            std::cout << "\n  Streaming analysis:" << std::endl;
            std::cout << "    - Streaming time (" << stream_avg << " us) > Compute time ("
                      << compute_avg << " us)" << std::endl;
            std::cout << "    - Communication continues for " << comm_remaining << " us after compute finishes" << std::endl;
        } else {
            std::cout << "\n  Streaming compute successfully hides communication!" << std::endl;
            std::cout << "    Streaming time (" << stream_avg << " us) ~ Compute time ("
                      << compute_avg << " us)" << std::endl;
        }

        std::cout << "================================================================\n" << std::endl;
    }

    ShmemFree(shmem_input);
    ShmemFree(tile_flag_shmem);
    aclrtFree(src1_dev);
    aclrtFree(output_dev);
    aclrtFreeHost(tile_flag_host);

    ShmemFinalize();

    status |= aclrtDestroyStream(computeStream);
    status |= aclrtDestroyStream(commStream);
    status |= aclrtResetDevice(device_id);
    status |= aclFinalize();

    return (status == 0) && is_ok;
}

bool RunAllGatherGemm()
{
    int n_ranks = 2;
    const char* n_ranks_env = std::getenv("N_RANKS");
    if (n_ranks_env != nullptr) {
        n_ranks = std::atoi(n_ranks_env);
    }

    if (n_ranks > MAX_RING_RANKS) {
        std::cerr << "[ERROR] n_ranks exceeds MAX_RING_RANKS=" << MAX_RING_RANKS << std::endl;
        return false;
    }

    int m_tiles = static_cast<int>(G_M / G_BASE_M);
    int n_tiles = static_cast<int>(G_N / G_BASE_N);
    int m_local = static_cast<int>(G_M) / n_ranks;
    int k_chunks = static_cast<int>(G_K / G_BASE_N);
    int num_blocks_per_src = (m_tiles / n_ranks) * k_chunks;
    int num_tiles = m_tiles * n_tiles;

    std::cout << "\n================================================================" << std::endl;
    std::cout << "  AllGather GEMM (M-slice) Performance Test Suite" << std::endl;
    std::cout << "  (AllGather=producer, GEMM=consumer, M-dimension partitioning)" << std::endl;
    std::cout << "  Signaling: TileFlagMatrix + TNOTIFY AtomicAdd + Polling (M-slice streaming)" << std::endl;
    std::cout << "================================================================" << std::endl;

    std::cout << "  Current compiled configuration:" << std::endl;
    std::cout << "    M=" << G_M << ", K=" << G_K << ", N=" << G_N << std::endl;
    std::cout << "    n_ranks=" << n_ranks << ", M_local=" << m_local << " per rank" << std::endl;
    std::cout << "    Base tile: " << G_BASE_M << "x" << G_BASE_K << "x" << G_BASE_N << std::endl;
    std::cout << "    Total blocks per source rank: " << num_blocks_per_src
              << " (" << (m_tiles / n_ranks) << " M-tiles x " << k_chunks << " K-blocks)" << std::endl;
    std::cout << "    Total compute tiles (per rank): " << num_tiles
              << " (" << m_tiles << "x" << n_tiles << ")" << std::endl;

    int num_remote_ranks = n_ranks - 1;
    int blocks_per_dest = (num_remote_ranks > 0) ? (COMM_BLOCK_NUM / num_remote_ranks) : 0;
    if (blocks_per_dest < 1) blocks_per_dest = 1;
    int actual_comm_blocks = num_remote_ranks * blocks_per_dest;
    std::cout << "  Parallelization:" << std::endl;
    std::cout << "    Comm kernel blocks: " << actual_comm_blocks << " (streaming AllGather M-slice, CONFIG=" << COMM_BLOCK_NUM << ")" << std::endl;
    std::cout << "    Compute kernel blocks: " << COMPUTE_BLOCK_NUM << " (GEMM consumer)" << std::endl;
    std::cout << "  Performance test config:" << std::endl;
    std::cout << "    Warmup iterations: " << WARMUP_ITERS << std::endl;
    std::cout << "    Measurement iterations: " << MEASURE_ITERS << std::endl;
    std::cout << "================================================================" << std::endl;

    int n_devices = n_ranks;
    constexpr int first_device_id = 0;

    std::vector<pid_t> pids;
    for (int r = 0; r < n_ranks; ++r) {
        pid_t pid = fork();
        if (pid == 0) {
            int device_id = r % n_devices + first_device_id;
            const bool ok = RunAllGatherGemmPerRank(r, n_ranks, device_id);
            _exit(ok ? 0 : 1);
        } else if (pid > 0) {
            pids.push_back(pid);
        } else {
            std::cerr << "[ERROR] fork() failed\n";
            return false;
        }
    }

    bool success = true;
    for (pid_t p : pids) {
        int wstatus = 0;
        waitpid(p, &wstatus, 0);
        if (!(WIFEXITED(wstatus) && WEXITSTATUS(wstatus) == 0)) {
            success = false;
        }
    }
    return success;
}
