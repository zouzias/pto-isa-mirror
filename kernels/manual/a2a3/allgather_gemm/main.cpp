/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include "securec.h"
#include <unistd.h>
#include <vector>
#include <string>
#include <iostream>

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

#ifdef AICORE
#undef AICORE
#endif
#define AICORE

#ifndef __gm__
#define __gm__
#endif

#include "common.hpp"

// ============================================================================
// Compile-time configuration
// ============================================================================
#include "gemm_config.hpp"

#ifndef CONFIG_ORIG_M
#define CONFIG_ORIG_M CONFIG_G_M
#endif
#ifndef CONFIG_ORIG_K
#define CONFIG_ORIG_K CONFIG_G_K
#endif
#ifndef CONFIG_ORIG_N
#define CONFIG_ORIG_N CONFIG_G_N
#endif

constexpr uint32_t ORIG_M = CONFIG_ORIG_M;
constexpr uint32_t ORIG_K = CONFIG_ORIG_K;
constexpr uint32_t ORIG_N = CONFIG_ORIG_N;

#ifndef CONFIG_COMPUTE_BLOCK_NUM
#define CONFIG_COMPUTE_BLOCK_NUM 20
#endif
#ifndef CONFIG_COMM_BLOCK_NUM
#define CONFIG_COMM_BLOCK_NUM 4
#endif
constexpr int COMPUTE_BLOCK_NUM = CONFIG_COMPUTE_BLOCK_NUM;
constexpr int COMM_BLOCK_NUM = CONFIG_COMM_BLOCK_NUM;

#include "kernel_launch.hpp"

// ============================================================================
// Per-rank resource context
// ============================================================================
struct RankResources {
    TestContext hcclTestCtx;
    aclrtStream computeStream = nullptr;
    aclrtStream commStream = nullptr;

    void *shmem_input = nullptr;
    void *chunk_flag_shmem = nullptr;
    ChunkFlagMatrix *chunk_flag_host = nullptr;
    int32_t *summary_host = nullptr;

    void *src1_dev = nullptr;
    void *output_dev = nullptr;

    size_t inputShmemBytes = 0;
    size_t chunkFlagWithSummarySize = 0;
    size_t chunkFlagMatrixSize = 0;
    size_t bSize = 0;
    size_t outputSize = 0;
    size_t aLocalSize = 0;

    int n_ranks = 0;
    int rank_id = 0;
};

// ============================================================================
// Helper functions
// ============================================================================
static void LaunchCommKernel(RankResources &r)
{
    launchRingCommStreaming(reinterpret_cast<uint8_t *>(r.shmem_input), reinterpret_cast<uint8_t *>(r.chunk_flag_shmem),
                            reinterpret_cast<uint8_t *>(r.hcclTestCtx.deviceCtx), r.n_ranks, r.commStream);
}

static void LaunchComputeKernel(RankResources &r)
{
    launchAllGatherGemmComputeStreaming(
        reinterpret_cast<uint8_t *>(r.output_dev), reinterpret_cast<uint8_t *>(r.shmem_input),
        reinterpret_cast<uint8_t *>(r.src1_dev), reinterpret_cast<uint8_t *>(r.chunk_flag_shmem), r.computeStream,
        COMPUTE_BLOCK_NUM);
}

// ============================================================================
// Sub-functions for RunAllGatherGemmPerRank
// ============================================================================
static bool AllocateResources(RankResources &r, int rank_id, int n_ranks, const HcclRootInfo *rootInfo)
{
    if (n_ranks <= 0) {
        std::cerr << "[ERROR] n_ranks must be positive, got " << n_ranks << std::endl;
        return false;
    }
    r.rank_id = rank_id;
    r.n_ranks = n_ranks;

    if (!r.hcclTestCtx.Init(rank_id, n_ranks, n_ranks, 0, rootInfo)) {
        std::cerr << "[ERROR] Rank " << rank_id << ": HCCL TestContext Init failed\n";
        return false;
    }

    int status = 0;
    status |= aclrtCreateStream(&r.computeStream);
    status |= aclrtCreateStream(&r.commStream);

    r.inputShmemBytes = static_cast<size_t>(G_M) * G_K * sizeof(uint16_t);
    int m_tiles = static_cast<int>(G_M / G_BASE_M);
    // NOLINTNEXTLINE - n_ranks > 0 guaranteed by guard above
    int m_tiles_local = (n_ranks > 0) ? (m_tiles / n_ranks) : 0;
    int k_tiles = static_cast<int>(G_K / G_BASE_N);
    int num_tiles_per_src = m_tiles_local * k_tiles;
    int optimal_chunk_size = ComputeOptimalChunkSize(num_tiles_per_src);

    r.chunkFlagMatrixSize = ChunkFlagMatrixSize(n_ranks, num_tiles_per_src, optimal_chunk_size);
    r.chunkFlagWithSummarySize = ChunkFlagMatrixWithSummarySize(n_ranks, num_tiles_per_src, optimal_chunk_size);

    uint64_t localWinBase = r.hcclTestCtx.hostCtx.windowsIn[rank_id];
    size_t winOffset = 0;
    r.shmem_input = WindowAlloc(localWinBase, winOffset, r.inputShmemBytes);
    aclrtMemset(r.shmem_input, r.inputShmemBytes, 0, r.inputShmemBytes);

    r.chunk_flag_shmem = WindowAlloc(localWinBase, winOffset, r.chunkFlagWithSummarySize);
    aclrtMallocHost(reinterpret_cast<void **>(&r.chunk_flag_host), r.chunkFlagWithSummarySize);
    ChunkFlagMatrixInit(r.chunk_flag_host, n_ranks, num_tiles_per_src, optimal_chunk_size);
    r.chunk_flag_host->my_rank = rank_id;
    r.summary_host =
        reinterpret_cast<int32_t *>(reinterpret_cast<uint8_t *>(r.chunk_flag_host) + r.chunkFlagMatrixSize);
    ChunkFlagMatrixSummaryInit(r.summary_host, n_ranks);
    aclrtMemcpy(r.chunk_flag_shmem, r.chunkFlagWithSummarySize, r.chunk_flag_host, r.chunkFlagWithSummarySize,
                ACL_MEMCPY_HOST_TO_DEVICE);

    r.bSize = static_cast<size_t>(G_K) * G_N * sizeof(uint16_t);
    aclrtMalloc(&r.src1_dev, r.bSize, ACL_MEM_MALLOC_HUGE_FIRST);

    r.outputSize = static_cast<size_t>(G_M) * G_N * sizeof(float);
    aclrtMalloc(&r.output_dev, r.outputSize, ACL_MEM_MALLOC_HUGE_FIRST);

    r.aLocalSize = (n_ranks > 0) ? ((static_cast<size_t>(G_M) / n_ranks) * G_K * sizeof(uint16_t)) : 0;
    return status == 0;
}

static bool LoadInputData(RankResources &r, const std::string &dataDir)
{
    std::string a_file = dataDir + "/pe_" + std::to_string(r.rank_id) + "_a.bin";
    std::string b_file = dataDir + "/pe_" + std::to_string(r.rank_id) + "_b.bin";

    uint16_t *a_local_host = nullptr;
    uint16_t *b_host = nullptr;
    aclrtMallocHost(reinterpret_cast<void **>(&a_local_host), r.aLocalSize);
    aclrtMallocHost(reinterpret_cast<void **>(&b_host), r.bSize);

    size_t a_file_size = 0;
    size_t b_file_size = 0;
    if (!PtoTestCommon::ReadFile(a_file, a_file_size, a_local_host, r.aLocalSize) || a_file_size != r.aLocalSize) {
        std::cerr << "[ERROR] Rank " << r.rank_id << ": A file mismatch: " << a_file << std::endl;
        return false;
    }
    if (!PtoTestCommon::ReadFile(b_file, b_file_size, b_host, r.bSize) || b_file_size != r.bSize) {
        std::cerr << "[ERROR] Rank " << r.rank_id << ": B file mismatch: " << b_file << std::endl;
        return false;
    }

    aclrtMemcpy(reinterpret_cast<uint8_t *>(r.shmem_input) + r.rank_id * r.aLocalSize, r.aLocalSize, a_local_host,
                r.aLocalSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(r.src1_dev, r.bSize, b_host, r.bSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtFreeHost(a_local_host);
    aclrtFreeHost(b_host);
    return true;
}

static void RunFunctionalTest(RankResources &r)
{
    if (r.rank_id == 0) {
        std::cout << "\n[INFO] Running functional verification..." << std::endl;
    }
    aclrtMemset(r.output_dev, r.outputSize, 0, r.outputSize);
    HcclHostBarrier(r.hcclTestCtx.comm, r.hcclTestCtx.stream);

    LaunchCommKernel(r);
    LaunchComputeKernel(r);
    aclrtSynchronizeStream(r.computeStream);
    aclrtSynchronizeStream(r.commStream);
    HcclHostBarrier(r.hcclTestCtx.comm, r.hcclTestCtx.stream);
}

static bool VerifyOutput(RankResources &r, const std::string &dataDir)
{
    float *output_host = nullptr;
    aclrtMallocHost(reinterpret_cast<void **>(&output_host), r.outputSize);
    aclrtMemcpy(output_host, r.outputSize, r.output_dev, r.outputSize, ACL_MEMCPY_DEVICE_TO_HOST);

    std::string output_file = dataDir + "/output_rank" + std::to_string(r.rank_id) + ".bin";
    PtoTestCommon::WriteFile(output_file, output_host, r.outputSize);

    bool is_ok = false;
    std::string golden_file = dataDir + "/golden.bin";
    size_t goldenSize = static_cast<size_t>(ORIG_M) * ORIG_N * sizeof(float);
    std::vector<float> golden(goldenSize / sizeof(float));
    size_t golden_file_size = 0;

    if (PtoTestCommon::ReadFile(golden_file, golden_file_size, golden.data(), goldenSize) &&
        golden_file_size == goldenSize) {
        if (ORIG_M == G_M && ORIG_N == G_N) {
            is_ok = PtoTestCommon::ResultCmp(golden, output_host, 0.001f);
        } else {
            std::vector<float> valid_output(static_cast<size_t>(ORIG_M) * ORIG_N);
            for (uint32_t row = 0; row < ORIG_M; ++row) {
                errno_t ret = memcpy_s(valid_output.data() + row * ORIG_N, ORIG_N * sizeof(float),
                                       output_host + row * G_N, ORIG_N * sizeof(float));
                if (ret != EOK) {
                    std::cerr << "[ERROR] memcpy_s failed at row " << row << " with errno " << ret << std::endl;
                    aclrtFreeHost(output_host);
                    return false;
                }
            }
            is_ok = PtoTestCommon::ResultCmp(golden, valid_output.data(), 0.001f);
        }
    } else {
        std::cerr << "[ERROR] Rank " << r.rank_id << ": golden.bin missing or size mismatch (expected " << goldenSize
                  << " bytes); numerical verification not performed, treated as FAILED" << std::endl;
    }
    aclrtFreeHost(output_host);
    return is_ok;
}

static void Cleanup(RankResources &r)
{
    aclrtFree(r.src1_dev);
    aclrtFree(r.output_dev);
    aclrtFreeHost(r.chunk_flag_host);
    if (r.computeStream)
        aclrtDestroyStream(r.computeStream);
    if (r.commStream)
        aclrtDestroyStream(r.commStream);
}

// ============================================================================
// RunAllGatherGemmPerRank: top-level per-rank orchestration
// ============================================================================
static bool RunAllGatherGemmPerRank(int rank_id, int n_ranks, int device_id, const HcclRootInfo *rootInfo,
                                    const std::string &dataDir)
{
    RankResources r;
    if (!AllocateResources(r, rank_id, n_ranks, rootInfo))
        return false;
    if (!LoadInputData(r, dataDir))
        return false;

    HcclHostBarrier(r.hcclTestCtx.comm, r.hcclTestCtx.stream);

    RunFunctionalTest(r);
    bool is_ok = VerifyOutput(r, dataDir);

    if (rank_id == 0) {
        std::cout << (is_ok ? "[INFO] Functional run completed. Verification PASSED." :
                              "[ERROR] Functional run completed. Verification FAILED!")
                  << std::endl;
    }

    aclrtSynchronizeStream(r.computeStream);
    aclrtSynchronizeStream(r.commStream);
    aclrtSynchronizeStream(r.hcclTestCtx.stream);

    CommMpiBarrier();
    Cleanup(r);

    if (rank_id == 0) {
        std::cout << (is_ok ? "[SUCCESS] AllGather GEMM demo completed successfully." :
                              "[FAILED] AllGather GEMM demo FAILED.")
                  << std::endl;
    }

    // HcclCommDestroy hangs on the internal barrier in some HCCL versions
    // when used with HCCL windows (shmem_input). Force-exit after MPI
    // barrier to avoid the hang; OS reclaims all resources.
    CommMpiBarrier();
    CommMpiFinalize();
    _exit(is_ok ? 0 : 1);
}

// ============================================================================
// Argument parsing and ACL initialization
// ============================================================================
struct AppArgs {
    int n_ranks = 2;
    std::string dataDir = "../out";
};

static AppArgs ParseArgs()
{
    AppArgs args;
    if (const char *env = std::getenv("N_RANKS")) {
        args.n_ranks = std::atoi(env);
    }
    if (const char *env = std::getenv("ALLGATHER_GEMM_DATA_DIR")) {
        args.dataDir = env;
    }
    return args;
}

static bool InitAcl(int rank_id, int device_id)
{
    constexpr int kAclRepeatInit = 100002;
    aclError aRet = aclInit(nullptr);
    if (aRet != ACL_SUCCESS && static_cast<int>(aRet) != kAclRepeatInit) {
        std::cerr << "[ERROR] Rank " << rank_id << ": aclInit failed: " << static_cast<int>(aRet) << std::endl;
        return false;
    }

    if (rank_id == 0)
        rtSetDevice(device_id);

    aRet = aclrtSetDevice(device_id);
    if (aRet != ACL_SUCCESS) {
        std::cerr << "[ERROR] Rank " << rank_id << ": aclrtSetDevice(" << device_id
                  << ") failed: " << static_cast<int>(aRet) << std::endl;
        return false;
    }
    return true;
}

static void PrintBanner(int n_ranks)
{
    std::cout << "\n================================================================" << std::endl;
    std::cout << "  AllGather GEMM (HCCL backend)" << std::endl;
    std::cout << "  M=" << G_M << ", K=" << G_K << ", N=" << G_N << ", pe_size=" << n_ranks << std::endl;
    if (ORIG_M != G_M || ORIG_K != G_K || ORIG_N != G_N) {
        std::cout << "  (original: M=" << ORIG_M << ", K=" << ORIG_K << ", N=" << ORIG_N << ")" << std::endl;
    }
    std::cout << "  Mode: FUNCTIONAL VERIFICATION" << std::endl;
    std::cout << "================================================================" << std::endl;
}

// ============================================================================
// main
// ============================================================================
int main(int argc, char **argv)
{
    if (!CommMpiInit(&argc, &argv)) {
        fprintf(stderr, "[FATAL] CommMpiInit failed. Launch with: mpirun -n <N> ./allgather_gemm\n");
        return 1;
    }

    AppArgs args = ParseArgs();
    int n_ranks = args.n_ranks;

    if (n_ranks > MAX_RING_RANKS) {
        std::cerr << "[ERROR] n_ranks exceeds MAX_RING_RANKS=" << MAX_RING_RANKS << std::endl;
        CommMpiFinalize();
        return 1;
    }

    int mpiRank = CommMpiRank();
    int mpiSize = CommMpiSize();
    if (mpiSize != n_ranks) {
        if (mpiRank == 0) {
            std::cerr << "[ERROR] MPI world size (" << mpiSize << ") != expected N_RANKS (" << n_ranks
                      << "). Launch with: mpirun -n " << n_ranks << " ./allgather_gemm" << std::endl;
        }
        CommMpiFinalize();
        return 1;
    }

    int rank_id = mpiRank;
    int device_id = rank_id;

    if (!InitAcl(rank_id, device_id)) {
        CommMpiFinalize();
        return 1;
    }

    HcclRootInfo rootInfo{};
    if (rank_id == 0) {
        HcclResult hret = HcclGetRootInfo(&rootInfo);
        if (hret != HCCL_SUCCESS) {
            std::cerr << "[ERROR] HcclGetRootInfo failed: " << hret << std::endl;
            CommMpiFinalize();
            return 1;
        }
    }

    CommMpiBcast(&rootInfo, HCCL_ROOT_INFO_BYTES, COMM_MPI_CHAR, 0);
    CommMpiBarrier();

    if (rank_id == 0)
        PrintBanner(n_ranks);

    bool ok = RunAllGatherGemmPerRank(rank_id, n_ranks, device_id, &rootInfo, args.dataDir);

    if (rank_id == 0) {
        std::cerr << "[FAILED] AllGather GEMM early init failure." << std::endl;
    }

    CommMpiFinalize();
    return ok ? 0 : 1;
}
