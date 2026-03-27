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
//
// Also contains host-side logic: per-rank init, kernel launch, verification,
// performance measurement, and MPI-based multi-process launcher.

#include <cstddef>
#include <cstdint>

// PTO headers first (define AICORE macro and PTO types)
#include "pto/comm/pto_comm_inst.hpp"
#include "pto/common/pto_tile.hpp"
#include <pto/pto-inst.hpp>

// HCCL wrappers: MUST come before <iostream> to avoid kernel_operator.h <-> std::dec conflict
#include "common.hpp"
#include "ready_queue.hpp"

// Host-only headers (safe now: kernel_operator.h already parsed)
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

#include "acl/acl.h"
#include "hccl/hccl.h"
#include "hccl/hccl_comm.h"
#include "hccl/hccl_types.h"
#include "comm_mpi.h"

// Internal HCCL APIs
extern "C" HcclResult HcclAllocComResourceByTiling(HcclComm comm, void *stream, void *mc2Tiling, void **commContext);
extern "C" HcclResult HcomGetCommHandleByGroup(const char *group, HcclComm *commHandle);

using CommTopo = uint32_t;
extern "C" HcclResult HcomGetL0TopoTypeEx(const char *group, CommTopo *topoType, uint32_t isSetDevice);
static constexpr uint32_t COMM_IS_NOT_SET_DEVICE = 0;
static constexpr uint32_t COMM_TOPO_MESH = 0b1u;

using rtError_t = int32_t;
using rtStream_t = void *;
static constexpr int32_t RT_STREAM_PRIORITY_DEFAULT = 0;
extern "C" rtError_t rtSetDevice(int32_t device);
extern "C" rtError_t rtStreamCreate(rtStream_t *stream, int32_t priority);
extern "C" rtError_t rtStreamDestroy(rtStream_t stream);

// ============================================================================
// V2 tiling structures for HcclAllocComResourceByTiling
// ============================================================================
namespace gemm_ar_tiling {

static constexpr uint32_t TILING_MAX_CC_NUM = 8U;
static constexpr uint32_t TILING_GROUP_NAME_SIZE = 128U;
static constexpr uint32_t TILING_ALG_CONFIG_SIZE = 128U;

struct Mc2InitTilingInner {
    uint32_t version;
    uint32_t mc2HcommCnt;
    uint32_t offset[TILING_MAX_CC_NUM];
    uint8_t debugMode;
    uint8_t preparePosition;
    uint16_t queueNum;
    uint16_t commBlockNum;
    uint8_t devType;
    char reserved[17];
};

struct Mc2cCTilingInner {
    uint8_t skipLocalRankCopy;
    uint8_t skipBufferWindowCopy;
    uint8_t stepSize;
    uint8_t version;
    char reserved[9];
    uint8_t commEngine;
    uint8_t srcDataType;
    uint8_t dstDataType;
    char groupName[TILING_GROUP_NAME_SIZE];
    char algConfig[TILING_ALG_CONFIG_SIZE];
    uint32_t opType;
    uint32_t reduceType;
};

struct Mc2CommConfigV2 {
    Mc2InitTilingInner init;
    Mc2cCTilingInner inner;
};

} // namespace gemm_ar_tiling

// ============================================================================
// HcclOpResParam compat structs for RING topology
// ============================================================================
namespace hccl_compat {

struct HcclSignalInfo {
    uint64_t resId;
    uint64_t addr;
    uint32_t devId;
    uint32_t tsId;
    uint32_t rankId;
    uint32_t flag;
};

struct HcclStreamInfo {
    int32_t streamIds;
    uint32_t sqIds;
    uint32_t cqIds;
    uint32_t logicCqids;
};

struct ListCommon {
    uint64_t nextHost;
    uint64_t preHost;
    uint64_t nextDevice;
    uint64_t preDevice;
};

static constexpr uint32_t COMPAT_LOCAL_NOTIFY_MAX_NUM = 64;
static constexpr uint32_t COMPAT_LOCAL_STREAM_MAX_NUM = 19;
static constexpr uint32_t COMPAT_AICPU_OP_NOTIFY_MAX_NUM = 2;

struct LocalResInfoV2 {
    uint32_t streamNum;
    uint32_t signalNum;
    HcclSignalInfo localSignals[COMPAT_LOCAL_NOTIFY_MAX_NUM];
    HcclStreamInfo streamInfo[COMPAT_LOCAL_STREAM_MAX_NUM];
    HcclStreamInfo mainStreamInfo;
    HcclSignalInfo aicpuOpNotify[COMPAT_AICPU_OP_NOTIFY_MAX_NUM];
    ListCommon nextTagRes;
};

struct AlgoTopoInfo {
    uint32_t userRank;
    uint32_t userRankSize;
    int32_t deviceLogicId;
    bool isSingleMeshAggregation;
    uint32_t deviceNumPerAggregation;
    uint32_t superPodNum;
    uint32_t devicePhyId;
    uint32_t topoType;
    uint32_t deviceType;
    uint32_t serverNum;
    uint32_t meshAggregationRankSize;
    uint32_t multiModuleDiffDeviceNumMode;
    uint32_t multiSuperPodDiffServerNumMode;
    uint32_t realUserRank;
    bool isDiffDeviceModule;
    bool isDiffDeviceType;
    uint32_t gcdDeviceNumPerAggregation;
    uint32_t moduleNum;
    uint32_t isUsedRdmaRankPairNum;
    uint64_t isUsedRdmaRankPair;
    uint32_t pairLinkCounterNum;
    uint64_t pairLinkCounter;
    uint32_t nicNum;
    uint64_t nicList;
    uint64_t complanRankLength;
    uint64_t complanRank;
    uint64_t bridgeRankNum;
    uint64_t bridgeRank;
    uint64_t serverAndsuperPodRankLength;
    uint64_t serverAndsuperPodRank;
};

struct HcclOpConfig {
    uint8_t deterministic;
    uint8_t retryEnable;
    uint8_t highPerfEnable;
    uint8_t padding[5];
    uint8_t linkTimeOut[8];
    uint64_t notifyWaitTime;
    uint32_t retryHoldTime;
    uint32_t retryIntervalTime;
    bool interXLinkDisable;
    uint32_t floatOverflowMode;
    uint32_t multiQpThreshold;
};

struct HDCommunicateParams {
    uint64_t hostAddr;
    uint64_t deviceAddr;
    uint64_t readCacheAddr;
    uint32_t devMemSize;
    uint32_t buffLen;
    uint32_t flag;
};

struct RemoteResPtr {
    uint64_t nextHostPtr;
    uint64_t nextDevicePtr;
};

struct HcclMC2WorkSpace {
    uint64_t workspace;
    uint64_t workspaceSize;
};

struct HcclRankRelationResV2 {
    uint32_t remoteUsrRankId;
    uint32_t remoteWorldRank;
    uint64_t windowsIn;
    uint64_t windowsOut;
    uint64_t windowsExp;
    ListCommon nextTagRes;
};

struct HcclOpResParamHead {
    uint32_t localUsrRankId;
    uint32_t rankSize;
    uint64_t winSize;
    uint64_t localWindowsIn;
    uint64_t localWindowsOut;
    char hcomId[128];
    uint64_t winExpSize;
    uint64_t localWindowsExp;
};

struct HcclOpResParam {
    HcclMC2WorkSpace mc2WorkSpace;
    uint32_t localUsrRankId;
    uint32_t rankSize;
    uint64_t winSize;
    uint64_t localWindowsIn;
    uint64_t localWindowsOut;
    char hcomId[128];
    uint64_t winExpSize;
    uint64_t localWindowsExp;
    uint32_t rWinStart;
    uint32_t rWinOffset;
    uint64_t version;
    LocalResInfoV2 localRes;
    AlgoTopoInfo topoInfo;
    HcclOpConfig config;
    uint64_t hostStateInfo;
    uint64_t aicpuStateInfo;
    uint64_t lockAddr;
    uint32_t rsv[16];
    uint32_t notifysize;
    uint32_t remoteResNum;
    RemoteResPtr remoteRes[1];
};

} // namespace hccl_compat

// ============================================================================
// Host-side helpers
// ============================================================================
inline void HcclHostBarrier(HcclComm comm, aclrtStream stream)
{
    HcclBarrier(comm, stream);
    aclrtSynchronizeStream(stream);
}

inline void *WindowAlloc(uint64_t windowBase, size_t &offset, size_t bytes)
{
    void *ptr = reinterpret_cast<void *>(windowBase + offset);
    offset += bytes;
    return ptr;
}

// ============================================================================
// Global GEMM parameters (must match compute kernel)
// ============================================================================
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
constexpr uint32_t G_M_TILES = G_M / G_BASE_M;
constexpr uint32_t G_N_TILES = G_N / G_BASE_N;
constexpr uint32_t G_NUM_TILES = G_M_TILES * G_N_TILES;

constexpr int WARMUP_ITERS = 5;
constexpr int MEASURE_ITERS = 10;
constexpr int COMPUTE_ONLY_ITERS = 5;

#ifndef CONFIG_COMPUTE_BLOCK_NUM
#define CONFIG_COMPUTE_BLOCK_NUM 24
#endif
#ifndef CONFIG_COMM_BLOCK_NUM
#define CONFIG_COMM_BLOCK_NUM 24
#endif
constexpr int COMPUTE_BLOCK_NUM = CONFIG_COMPUTE_BLOCK_NUM;
constexpr int COMM_BLOCK_NUM = CONFIG_COMM_BLOCK_NUM;
constexpr int MAX_RANKS = 8;

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
// Host-side kernel launchers
// ============================================================================
static void launchGemmCommRS(uint8_t *gemm_output, uint8_t *recv_buffers,
                             uint8_t *queue_set, uint8_t *hcclCtx,
                             int rank, int nranks, void *stream, int num_compute_blocks)
{
    GemmCommRSKernel<<<COMM_BLOCK_NUM, nullptr, stream>>>(
        gemm_output, recv_buffers, queue_set, hcclCtx, rank, nranks, num_compute_blocks, COMM_BLOCK_NUM);
}

static void launchGemmCommReduce(uint8_t *gemm_output, uint8_t *recv_buffers, uint8_t *reduced_output,
                                 uint8_t *hcclCtx, int rank, int nranks, void *stream)
{
    GemmCommReduceKernel<<<COMM_BLOCK_NUM, nullptr, stream>>>(
        gemm_output, recv_buffers, reduced_output, hcclCtx, rank, nranks, COMM_BLOCK_NUM);
}

static void launchGemmCommAG(uint8_t *reduced_output, uint8_t *hcclCtx,
                             int rank, int nranks, void *stream)
{
    GemmCommAGKernel<<<COMM_BLOCK_NUM, nullptr, stream>>>(
        reduced_output, hcclCtx, rank, nranks, COMM_BLOCK_NUM);
}

extern void launchGemmCompute(uint8_t *gemm_output, uint8_t *src0, uint8_t *src1,
                              uint8_t *queue_set, int rank, void *stream, int block_num, uint32_t k_per_rank);

// ============================================================================
// Helpers
// ============================================================================

struct PerfStats {
    double avg, min_val, max_val, std_dev;
};

static PerfStats calcStats(const std::vector<double>& times)
{
    double sum = 0.0, mn = times[0], mx = times[0];
    for (double t : times) {
        sum += t;
        if (t < mn) mn = t;
        if (t > mx) mx = t;
    }
    double avg = sum / times.size();
    double var = 0.0;
    for (double t : times) var += (t - avg) * (t - avg);
    return {avg, mn, mx, std::sqrt(var / times.size())};
}

// ============================================================================
// HCCL context initialization (MESH or RING)
// ============================================================================
struct GemmHcclContext {
    HcclComm comm{nullptr};
    HcclDeviceContext *deviceCtx{nullptr};
    HcclDeviceContext hostCtx{};
    bool ownsDeviceCtx{false};

    bool Init(int rankId, int nRanks, int deviceId, const HcclRootInfo *rootInfo, rtStream_t hcclStream)
    {
        constexpr int kMaxRetries = 3;
        HcclResult hret = HCCL_SUCCESS;
        for (int attempt = 0; attempt < kMaxRetries; ++attempt) {
            hret = HcclCommInitRootInfo(static_cast<uint32_t>(nRanks), rootInfo,
                                        static_cast<uint32_t>(rankId), &comm);
            if (hret == HCCL_SUCCESS) break;
            std::cerr << "[WARN] Rank " << rankId << ": HcclCommInitRootInfo failed: " << hret
                      << " (attempt " << (attempt + 1) << "/" << kMaxRetries
                      << "), retrying in 5s..." << std::endl;
            sleep(5);
        }
        if (hret != HCCL_SUCCESS) {
            std::cerr << "[ERROR] Rank " << rankId << ": HcclCommInitRootInfo failed after "
                      << kMaxRetries << " attempts: " << hret << std::endl;
            return false;
        }

        char group[128] = {};
        hret = HcclGetCommName(comm, group);
        if (hret != HCCL_SUCCESS) {
            std::cerr << "[ERROR] Rank " << rankId << ": HcclGetCommName failed: " << hret << std::endl;
            return false;
        }

        CommTopo topoRet = 0;
        hret = HcomGetL0TopoTypeEx(group, &topoRet, COMM_IS_NOT_SET_DEVICE);
        if (hret != HCCL_SUCCESS) {
            std::cerr << "[ERROR] Rank " << rankId << ": HcomGetL0TopoTypeEx failed: " << hret << std::endl;
            return false;
        }

        HcclComm commHandle = nullptr;
        hret = HcomGetCommHandleByGroup(group, &commHandle);
        if (hret != HCCL_SUCCESS) {
            std::cerr << "[ERROR] Rank " << rankId << ": HcomGetCommHandleByGroup failed: " << hret << std::endl;
            return false;
        }

        CommMpiBarrier();

        gemm_ar_tiling::Mc2CommConfigV2 tiling{};
        memset(&tiling, 0, sizeof(tiling));

        tiling.init.version = 100U;
        tiling.init.mc2HcommCnt = 1U;
        tiling.init.commBlockNum = 48U;
        tiling.init.devType = 4U;
        tiling.init.offset[0] =
            static_cast<uint32_t>(reinterpret_cast<uint64_t>(&tiling.inner) - reinterpret_cast<uint64_t>(&tiling.init));

        tiling.inner.opType = 18U;
        tiling.inner.commEngine = 3U;
        tiling.inner.version = 1U;
        strncpy(tiling.inner.groupName, group, gemm_ar_tiling::TILING_GROUP_NAME_SIZE - 1);
        strncpy(tiling.inner.algConfig, "BatchWrite=level0:fullmesh", gemm_ar_tiling::TILING_ALG_CONFIG_SIZE - 1);

        void *ctxPtr = nullptr;
        hret = HcclAllocComResourceByTiling(commHandle, hcclStream, &tiling, &ctxPtr);
        if (hret != HCCL_SUCCESS || ctxPtr == nullptr) {
            std::cerr << "[ERROR] Rank " << rankId << ": HcclAllocComResourceByTiling failed: " << hret << std::endl;
            return false;
        }

        if (topoRet == COMM_TOPO_MESH) {
            return InitMeshPath(rankId, ctxPtr);
        }
        return InitRingPath(rankId, nRanks, ctxPtr);
    }

    void Finalize()
    {
        if (ownsDeviceCtx && deviceCtx != nullptr) {
            aclrtFree(deviceCtx);
            deviceCtx = nullptr;
        }
        if (comm != nullptr) {
            HcclCommDestroy(comm);
            comm = nullptr;
        }
    }

private:
    bool InitMeshPath(int rankId, void *ctxPtr)
    {
        deviceCtx = reinterpret_cast<HcclDeviceContext *>(ctxPtr);
        aclError aRet = aclrtMemcpy(&hostCtx, sizeof(hostCtx), deviceCtx, sizeof(hostCtx), ACL_MEMCPY_DEVICE_TO_HOST);
        if (aRet != ACL_SUCCESS) {
            std::cerr << "[ERROR] Rank " << rankId << ": aclrtMemcpy(deviceCtx) failed: " << (int)aRet << std::endl;
            return false;
        }
        if (rankId == 0) {
            std::cout << "[INFO] HCCL MESH init OK"
                      << " rankId=" << hostCtx.rankId << " rankNum=" << hostCtx.rankNum
                      << " winSize=" << hostCtx.winSize << std::endl;
        }
        return true;
    }

    bool InitRingPath(int rankId, int nRanks, void *ctxPtr)
    {
        using namespace hccl_compat;
        auto *rawCtx = reinterpret_cast<uint8_t *>(ctxPtr);

        HcclOpResParamHead head{};
        const size_t headOff = offsetof(HcclOpResParam, localUsrRankId);
        aclError aRet = aclrtMemcpy(&head, sizeof(head), rawCtx + headOff, sizeof(head), ACL_MEMCPY_DEVICE_TO_HOST);
        if (aRet != ACL_SUCCESS) {
            std::cerr << "[ERROR] Rank " << rankId << ": read HcclOpResParam head failed\n";
            return false;
        }

        if (head.rankSize == 0 || head.rankSize > HCCL_MAX_RANK_NUM) {
            std::cerr << "[ERROR] Rank " << rankId << ": invalid rankSize=" << head.rankSize << std::endl;
            return false;
        }

        const size_t remoteResOff = offsetof(HcclOpResParam, remoteRes);
        const size_t remoteResBytes = head.rankSize * sizeof(RemoteResPtr);
        std::vector<RemoteResPtr> remoteResArr(head.rankSize);

        aRet = aclrtMemcpy(remoteResArr.data(), remoteResBytes, rawCtx + remoteResOff, remoteResBytes,
                           ACL_MEMCPY_DEVICE_TO_HOST);
        if (aRet != ACL_SUCCESS) {
            std::cerr << "[ERROR] Rank " << rankId << ": read remoteRes failed\n";
            return false;
        }

        memset(&hostCtx, 0, sizeof(hostCtx));

        uint64_t wsFields[2] = {0, 0};
        aRet = aclrtMemcpy(wsFields, sizeof(wsFields), rawCtx, sizeof(wsFields), ACL_MEMCPY_DEVICE_TO_HOST);
        if (aRet == ACL_SUCCESS) {
            hostCtx.workSpace = wsFields[0];
            hostCtx.workSpaceSize = wsFields[1];
        }

        hostCtx.rankId = head.localUsrRankId;
        hostCtx.rankNum = head.rankSize;
        hostCtx.winSize = head.winSize;

        for (uint32_t i = 0; i < head.rankSize; ++i) {
            if (i == head.localUsrRankId) {
                hostCtx.windowsIn[i] = head.localWindowsIn;
                continue;
            }

            uint64_t devPtr = remoteResArr[i].nextDevicePtr;
            if (devPtr == 0) {
                std::cerr << "[ERROR] Rank " << rankId << ": remoteRes[" << i << "].nextDevicePtr is null\n";
                return false;
            }

            HcclRankRelationResV2 remoteInfo{};
            aRet = aclrtMemcpy(&remoteInfo, sizeof(remoteInfo), reinterpret_cast<void *>(devPtr), sizeof(remoteInfo),
                               ACL_MEMCPY_DEVICE_TO_HOST);
            if (aRet != ACL_SUCCESS) {
                std::cerr << "[ERROR] Rank " << rankId << ": read remote rank " << i << " info failed\n";
                return false;
            }

            hostCtx.windowsIn[i] = remoteInfo.windowsIn;
        }

        void *newDevMem = nullptr;
        aRet = aclrtMalloc(&newDevMem, sizeof(HcclDeviceContext), ACL_MEM_MALLOC_HUGE_FIRST);
        if (aRet != ACL_SUCCESS || newDevMem == nullptr) {
            std::cerr << "[ERROR] Rank " << rankId << ": aclrtMalloc for RING deviceCtx failed\n";
            return false;
        }

        aRet = aclrtMemcpy(newDevMem, sizeof(HcclDeviceContext), &hostCtx, sizeof(HcclDeviceContext),
                           ACL_MEMCPY_HOST_TO_DEVICE);
        if (aRet != ACL_SUCCESS) {
            aclrtFree(newDevMem);
            std::cerr << "[ERROR] Rank " << rankId << ": copy RING deviceCtx to device failed\n";
            return false;
        }

        deviceCtx = reinterpret_cast<HcclDeviceContext *>(newDevMem);
        ownsDeviceCtx = true;

        if (rankId == 0) {
            std::cout << "[INFO] HCCL RING init OK"
                      << " rankId=" << hostCtx.rankId << " rankNum=" << hostCtx.rankNum
                      << " winSize=" << hostCtx.winSize << std::endl;
        }
        return true;
    }
};

// ============================================================================
// Per-rank execution logic
// ============================================================================
static bool RunGemmAllReducePerRank(int rank_id, int n_ranks, int device_id,
                                    const uint16_t *a_data, size_t a_bytes,
                                    const uint16_t *b_data, size_t b_bytes,
                                    const float *golden, size_t golden_bytes,
                                    const HcclRootInfo *rootInfo)
{
    int status = 0;
    aclrtStream computeStream = nullptr;
    aclrtStream commStream = nullptr;

    status |= aclrtCreateStream(&computeStream);
    status |= aclrtCreateStream(&commStream);

    // ------ HCCL init ------
    rtStream_t hcclStream = nullptr;
    rtStreamCreate(&hcclStream, RT_STREAM_PRIORITY_DEFAULT);

    GemmHcclContext hctx;
    if (!hctx.Init(rank_id, n_ranks, device_id, rootInfo, hcclStream)) {
        std::cerr << "[ERROR] Rank " << rank_id << ": HCCL init failed!\n";
        return false;
    }

    // ------ Allocate memory ------
    // Only buffers that are written remotely (via HcclRemotePtr / TPUT) must
    // reside in the HCCL RDMA window.  gemm_output is only read locally by the
    // comm kernel, so it can live in normal device memory.
    //
    //   recv_buffers   — remote-written in Phase 1 (ReduceScatter)
    //   reduced_output — remote-written in Phase 3 (AllGather)
    //   gemm_output    — local-only (compute kernel writes, comm kernel reads)
    size_t outputSize = static_cast<size_t>(G_M) * G_N * sizeof(float);
    size_t recvBuffersSize = static_cast<size_t>(n_ranks) * outputSize;

    void *gemm_output = nullptr;
    aclrtMalloc(&gemm_output, outputSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (!gemm_output) {
        std::cerr << "[ERROR] Rank " << rank_id << ": aclrtMalloc gemm_output failed\n";
        return false;
    }

    uint64_t windowBase = hctx.hostCtx.windowsIn[hctx.hostCtx.rankId];
    size_t winOffset = 0;
    void *recv_buffers = WindowAlloc(windowBase, winOffset, recvBuffersSize);
    void *reduced_output = WindowAlloc(windowBase, winOffset, outputSize);

    if (rank_id == 0) {
        std::cout << "[INFO] HCCL window: winSize=" << hctx.hostCtx.winSize
                  << " used=" << winOffset
                  << " (recv=" << (recvBuffersSize / (1024 * 1024)) << "MB"
                  << ", reduced=" << (outputSize / (1024 * 1024)) << "MB)"
                  << "  gemm_output=" << (outputSize / (1024 * 1024)) << "MB (device mem)" << std::endl;
    }

    if (winOffset > hctx.hostCtx.winSize) {
        std::cerr << "[ERROR] Rank " << rank_id << ": HCCL window too small! need=" << winOffset
                  << " have=" << hctx.hostCtx.winSize << std::endl;
        aclrtFree(gemm_output);
        return false;
    }

    aclrtMemset(gemm_output, outputSize, 0, outputSize);
    aclrtMemset(recv_buffers, recvBuffersSize, 0, recvBuffersSize);
    aclrtMemset(reduced_output, outputSize, 0, outputSize);

    uint32_t k_per_rank = G_K / n_ranks;
    if (G_K % n_ranks != 0) {
        std::cerr << "[ERROR] K=" << G_K << " not divisible by nranks=" << n_ranks << "\n";
        return false;
    }

    size_t aSize = G_M * k_per_rank * sizeof(uint16_t);
    size_t bSize = k_per_rank * G_N * sizeof(uint16_t);
    void *src0_dev = nullptr, *src1_dev = nullptr;
    aclrtMalloc(&src0_dev, aSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&src1_dev, bSize, ACL_MEM_MALLOC_HUGE_FIRST);

    int tiles_per_block = (G_NUM_TILES + COMPUTE_BLOCK_NUM - 1) / COMPUTE_BLOCK_NUM;
    size_t queueSetSize = MultiBlockQueueSetSize(COMPUTE_BLOCK_NUM, tiles_per_block);
    void *queueSet_dev = nullptr;
    aclrtMalloc(&queueSet_dev, queueSetSize, ACL_MEM_MALLOC_HUGE_FIRST);

    MultiBlockQueueSet *queueSet_host = nullptr;
    aclrtMallocHost(reinterpret_cast<void **>(&queueSet_host), queueSetSize);
    MultiBlockQueueSetInit(queueSet_host, COMPUTE_BLOCK_NUM, G_NUM_TILES);
    aclrtMemcpy(queueSet_dev, queueSetSize, queueSet_host, queueSetSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtFreeHost(queueSet_host);

    // ------ Upload input data to device ------
    if (rank_id == 0) {
        std::cout << "[INFO] Data Parallel: A_part[" << G_M << "x" << k_per_rank
                  << "], B_part[" << k_per_rank << "x" << G_N << "]" << std::endl;
    }

    aclrtMemcpy(src0_dev, aSize, a_data, a_bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src1_dev, bSize, b_data, b_bytes, ACL_MEMCPY_HOST_TO_DEVICE);

    HcclHostBarrier(hctx.comm, commStream);

    // ------ Helpers ------
    MultiBlockQueueSet *queueSet_reset_host = nullptr;
    aclrtMallocHost(reinterpret_cast<void **>(&queueSet_reset_host), queueSetSize);

    auto resetState = [&]() {
        MultiBlockQueueSetInit(queueSet_reset_host, COMPUTE_BLOCK_NUM, G_NUM_TILES);
        aclrtMemcpy(queueSet_dev, queueSetSize, queueSet_reset_host, queueSetSize, ACL_MEMCPY_HOST_TO_DEVICE);
        aclrtMemset(gemm_output, outputSize, 0, outputSize);
        aclrtMemset(recv_buffers, recvBuffersSize, 0, recvBuffersSize);
        aclrtMemset(reduced_output, outputSize, 0, outputSize);
    };

    auto launchCompute = [&](aclrtStream s) {
        launchGemmCompute(
            reinterpret_cast<uint8_t *>(gemm_output),
            reinterpret_cast<uint8_t *>(src0_dev),
            reinterpret_cast<uint8_t *>(src1_dev),
            reinterpret_cast<uint8_t *>(queueSet_dev),
            rank_id, s, COMPUTE_BLOCK_NUM, k_per_rank);
    };

    uint8_t *hcclCtxPtr = reinterpret_cast<uint8_t *>(hctx.deviceCtx);

    auto launchComm = [&](aclrtStream s) {
        launchGemmCommRS(
            reinterpret_cast<uint8_t *>(gemm_output),
            reinterpret_cast<uint8_t *>(recv_buffers),
            reinterpret_cast<uint8_t *>(queueSet_dev),
            hcclCtxPtr,
            rank_id, n_ranks, s, COMPUTE_BLOCK_NUM);
        HcclHostBarrier(hctx.comm, s);

        launchGemmCommReduce(
            reinterpret_cast<uint8_t *>(gemm_output),
            reinterpret_cast<uint8_t *>(recv_buffers),
            reinterpret_cast<uint8_t *>(reduced_output),
            hcclCtxPtr,
            rank_id, n_ranks, s);
        HcclHostBarrier(hctx.comm, s);

        launchGemmCommAG(
            reinterpret_cast<uint8_t *>(reduced_output),
            hcclCtxPtr,
            rank_id, n_ranks, s);
        HcclHostBarrier(hctx.comm, s);
    };

    auto syncAll = [&]() {
        aclrtSynchronizeStream(computeStream);
        aclrtSynchronizeStream(commStream);
        HcclHostBarrier(hctx.comm, commStream);
    };

    // ------ Warmup ------
    for (int i = 0; i < WARMUP_ITERS; ++i) {
        resetState();
        syncAll();
        launchCompute(computeStream);
        launchComm(commStream);
        syncAll();
    }

    // ------ Compute-only measurement ------
    std::vector<double> compute_times_us;
    for (int iter = 0; iter < COMPUTE_ONLY_ITERS; ++iter) {
        resetState();
        aclrtSynchronizeStream(computeStream);
        HcclHostBarrier(hctx.comm, commStream);

        auto t0 = std::chrono::high_resolution_clock::now();
        launchCompute(computeStream);
        aclrtSynchronizeStream(computeStream);
        auto t1 = std::chrono::high_resolution_clock::now();
        compute_times_us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
        HcclHostBarrier(hctx.comm, commStream);
    }

    // ------ Sequential measurement (compute -> comm, no overlap) ------
    std::vector<double> sequential_times_us, seq_compute_us, seq_comm_us;
    for (int iter = 0; iter < MEASURE_ITERS; ++iter) {
        resetState();
        syncAll();

        auto t0 = std::chrono::high_resolution_clock::now();
        launchCompute(computeStream);
        aclrtSynchronizeStream(computeStream);
        auto t1 = std::chrono::high_resolution_clock::now();
        launchComm(commStream);
        auto t2 = std::chrono::high_resolution_clock::now();

        seq_compute_us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
        seq_comm_us.push_back(std::chrono::duration<double, std::micro>(t2 - t1).count());
        sequential_times_us.push_back(std::chrono::duration<double, std::micro>(t2 - t0).count());
        HcclHostBarrier(hctx.comm, commStream);
    }

    // ------ Pipelined measurement (compute || comm, with overlap) ------
    // Compute on computeStream, comm on commStream — NPU overlaps them.

    aclrtEvent evComputeStart = nullptr, evComputeEnd = nullptr;
    aclrtCreateEvent(&evComputeStart);
    aclrtCreateEvent(&evComputeEnd);

    std::vector<double> pipelined_times_us, pipe_compute_us, pipe_comm_us;
    for (int iter = 0; iter < MEASURE_ITERS; ++iter) {
        resetState();
        syncAll();

        auto t0 = std::chrono::high_resolution_clock::now();

        aclrtRecordEvent(evComputeStart, computeStream);
        launchCompute(computeStream);
        aclrtRecordEvent(evComputeEnd, computeStream);

        launchComm(commStream);
        aclrtSynchronizeStream(computeStream);

        auto t1 = std::chrono::high_resolution_clock::now();

        float compute_ms = 0.0f;
        aclrtEventElapsedTime(&compute_ms, evComputeStart, evComputeEnd);

        double total_us = std::chrono::duration<double, std::micro>(t1 - t0).count();

        pipe_compute_us.push_back((double)compute_ms * 1000.0);
        pipe_comm_us.push_back(total_us);
        pipelined_times_us.push_back(total_us);
    }

    aclrtDestroyEvent(evComputeStart);
    aclrtDestroyEvent(evComputeEnd);

    // ------ Final run for verification ------
    resetState();
    syncAll();
    launchCompute(computeStream);
    launchComm(commStream);
    syncAll();

    // ------ Verification ------
    float *output_host = nullptr;
    aclrtMallocHost(reinterpret_cast<void **>(&output_host), outputSize);
    aclrtMemcpy(output_host, outputSize, reduced_output, outputSize, ACL_MEMCPY_DEVICE_TO_HOST);

    bool is_ok = true;
    if (rank_id == 0) {
        const float eps = 0.001f;
        const size_t num_elements = outputSize / sizeof(float);
        float max_diff = 0.0f, max_diff_ratio = 0.0f;
        size_t err_count = 0, zero_count = 0;
        const size_t err_threshold = static_cast<size_t>(num_elements * eps);
        const size_t zero_threshold = 0x1000;

        for (size_t i = 0; i < num_elements; ++i) {
            float exp_val = golden[i], act_val = output_host[i];
            float diff = std::abs(exp_val - act_val);
            float rel = (std::abs(exp_val) > 1e-6f) ? (diff / std::abs(exp_val)) : diff;
            if (diff > max_diff) max_diff = diff;
            if (rel > max_diff_ratio) max_diff_ratio = rel;
            if (std::abs(act_val) <= 1e-6f && std::abs(exp_val) > 1e-6f) zero_count++;
            if ((diff > eps && rel > eps) || zero_count > zero_threshold) err_count++;
        }

        is_ok = (err_count <= err_threshold) && (zero_count <= zero_threshold);
        std::cout << "[VERIFY] max_diff=" << max_diff << " max_ratio=" << max_diff_ratio
                  << " err=" << err_count << "/" << err_threshold
                  << " zeros=0x" << std::hex << zero_count << std::dec
                  << " -> " << (is_ok ? "PASS" : "FAIL") << std::endl;
    }

    aclrtFreeHost(output_host);
    aclrtFreeHost(queueSet_reset_host);

    // ------ Performance report ------
    if (rank_id == 0) {
        PerfStats comp_s = calcStats(compute_times_us);
        PerfStats seq_s  = calcStats(sequential_times_us);
        PerfStats pipe_s = calcStats(pipelined_times_us);
        PerfStats seq_comp_s = calcStats(seq_compute_us);
        PerfStats seq_comm_s = calcStats(seq_comm_us);
        PerfStats pipe_comp_s = calcStats(pipe_compute_us);
        PerfStats pipe_comm_s = calcStats(pipe_comm_us);

        double flops_per_rank = 2.0 * G_M * (double)k_per_rank * G_N;
        double flops_total    = 2.0 * G_M * (double)G_K * G_N;
        auto gflops = [](double flops, double us) { return (us > 0) ? (flops / (us * 1e-6) / 1e9) : 0.0; };

        size_t tileBytes = static_cast<size_t>(G_BASE_M) * G_BASE_N * sizeof(float);
        int tiles_per_owner = G_NUM_TILES / n_ranks;
        double rs_bytes = static_cast<double>(G_NUM_TILES - tiles_per_owner) * tileBytes;
        double ag_bytes = static_cast<double>(tiles_per_owner) * (n_ranks - 1) * tileBytes;
        double data_gb = (rs_bytes + ag_bytes) / (1024.0 * 1024.0 * 1024.0);
        auto bw_gbs = [&](double us) { return (us > 0) ? ((rs_bytes + ag_bytes) / (us * 1e-6) / (1024.0*1024.0*1024.0)) : 0.0; };

        double speedup = seq_s.avg / pipe_s.avg;
        double overlap_time = (seq_comp_s.avg + seq_comm_s.avg) - pipe_s.avg;
        double overlap_eff = (overlap_time > 0) ? (overlap_time / std::min(seq_comp_s.avg, seq_comm_s.avg) * 100.0) : 0.0;

        std::cout << std::fixed << std::setprecision(1);
        std::cout << "\n================================================================" << std::endl;
        std::cout << (is_ok ? "[SUCCESS]" : "[FAILED]") << " GEMM AllReduce (HCCL)" << std::endl;
        std::cout << "  M=" << G_M << " K=" << G_K << " N=" << G_N
                  << "  ranks=" << n_ranks
                  << "  compute_blocks=" << COMPUTE_BLOCK_NUM
                  << "  comm_blocks=" << COMM_BLOCK_NUM << std::endl;
        std::cout << "  tiles=" << G_NUM_TILES << " (" << G_M_TILES << "x" << G_N_TILES << ")"
                  << "  comm_data=" << std::setprecision(3) << data_gb << " GB/rank" << std::endl;

        std::cout << "\n  Compute-only:   " << std::setprecision(1) << comp_s.avg << " us"
                  << "  (" << std::setprecision(0) << gflops(flops_per_rank, comp_s.avg) << " GFLOPS)" << std::endl;

        std::cout << "\n  Sequential:     " << std::setprecision(1) << seq_s.avg << " us" << std::endl;
        std::cout << "    compute:      " << seq_comp_s.avg << " us"
                  << "  (" << std::setprecision(0) << gflops(flops_per_rank, seq_comp_s.avg) << " GFLOPS)" << std::endl;
        std::cout << "    comm:         " << std::setprecision(1) << seq_comm_s.avg << " us"
                  << "  (" << std::setprecision(1) << bw_gbs(seq_comm_s.avg) << " GB/s)" << std::endl;

        std::cout << "\n  Pipelined:      " << std::setprecision(1) << pipe_s.avg << " us" << std::endl;
        std::cout << "    compute done: " << pipe_comp_s.avg << " us"
                  << "  (" << std::setprecision(0) << gflops(flops_per_rank, pipe_comp_s.avg) << " GFLOPS, "
                  << std::setprecision(1) << (gflops(flops_per_rank, pipe_comp_s.avg) / gflops(flops_per_rank, comp_s.avg) * 100.0)
                  << "% of pure)" << std::endl;
        std::cout << "    comm done:    " << std::setprecision(1) << pipe_comm_s.avg << " us"
                  << "  (" << std::setprecision(1) << bw_gbs(pipe_comm_s.avg) << " GB/s)" << std::endl;

        std::cout << "\n  Speedup:        " << std::setprecision(3) << speedup << "x" << std::endl;
        std::cout << "  Time saved:     " << std::setprecision(1) << (seq_s.avg - pipe_s.avg) << " us"
                  << " (" << std::setprecision(1) << ((seq_s.avg - pipe_s.avg) / seq_s.avg * 100.0) << "%)" << std::endl;
        std::cout << "  Overlap eff:    " << std::setprecision(1) << overlap_eff << "%" << std::endl;
        std::cout << "  Throughput:     " << std::setprecision(0) << gflops(flops_total, pipe_s.avg) << " GFLOPS (total)" << std::endl;
        std::cout << "================================================================\n" << std::endl;
    }

    // ------ Cleanup ------
    aclrtFree(gemm_output);
    aclrtFree(src0_dev);
    aclrtFree(src1_dev);
    aclrtFree(queueSet_dev);

    hctx.Finalize();
    if (hcclStream) rtStreamDestroy(hcclStream);

    status |= aclrtDestroyStream(computeStream);
    status |= aclrtDestroyStream(commStream);

    return (status == 0) && is_ok;
}

// ============================================================================
// MPI-based multi-process launcher
// ============================================================================
bool RunGemmAllReduce(int n_ranks, int first_device_id,
                      const uint16_t *a_parts,
                      const uint16_t *b_parts,
                      const float *golden)
{
    if (n_ranks <= 0 || n_ranks > 8) {
        std::cerr << "[ERROR] Invalid n_ranks: " << n_ranks << " (must be 1-8)\n";
        return false;
    }

    int mpiRank = CommMpiRank();

    uint32_t k_per_rank = G_K / n_ranks;
    size_t a_rank_bytes = (size_t)G_M * k_per_rank * sizeof(uint16_t);
    size_t b_rank_bytes = (size_t)G_N * k_per_rank * sizeof(uint16_t);
    size_t golden_bytes = (size_t)G_M * G_N * sizeof(float);
    size_t a_rank_elems = (size_t)G_M * k_per_rank;
    size_t b_rank_elems = (size_t)G_N * k_per_rank;

    if (mpiRank == 0) {
        std::cout << "\n================================================================" << std::endl;
        std::cout << "  GEMM AllReduce (ReduceScatter + AllGather) — HCCL backend" << std::endl;
        std::cout << "  M=" << G_M << " K=" << G_K << " N=" << G_N
                  << "  tile=" << G_BASE_M << "x" << G_BASE_K << "x" << G_BASE_N
                  << "  tiles=" << G_NUM_TILES << std::endl;
        std::cout << "  ranks=" << n_ranks
                  << "  devices=[" << first_device_id << "," << (first_device_id + n_ranks) << ")"
                  << "  compute_blocks=" << COMPUTE_BLOCK_NUM
                  << "  comm_blocks=" << COMM_BLOCK_NUM << std::endl;
        std::cout << "================================================================" << std::endl;
    }

    int device_id = mpiRank % n_ranks + first_device_id;

    constexpr int kAclRepeatInit = 100002;
    aclError aRet = aclInit(nullptr);
    if (aRet != ACL_SUCCESS && static_cast<int>(aRet) != kAclRepeatInit) {
        std::cerr << "[ERROR] Rank " << mpiRank << ": aclInit failed: " << (int)aRet << std::endl;
        return false;
    }

    if (mpiRank == 0) {
        rtSetDevice(device_id);
    }

    aRet = aclrtSetDevice(device_id);
    if (aRet != ACL_SUCCESS) {
        std::cerr << "[ERROR] Rank " << mpiRank << ": aclrtSetDevice(" << device_id << ") failed\n";
        return false;
    }

    HcclRootInfo rootInfo{};
    if (mpiRank == 0) {
        constexpr int kMaxRetries = 3;
        HcclResult hret = HCCL_SUCCESS;
        for (int attempt = 0; attempt < kMaxRetries; ++attempt) {
            hret = HcclGetRootInfo(&rootInfo);
            if (hret == HCCL_SUCCESS) break;
            std::cerr << "[WARN] HcclGetRootInfo failed: " << hret
                      << " (attempt " << (attempt + 1) << "/" << kMaxRetries
                      << "), retrying in 5s..." << std::endl;
            sleep(5);
        }
        if (hret != HCCL_SUCCESS) {
            std::cerr << "[ERROR] HcclGetRootInfo failed after " << kMaxRetries
                      << " attempts: " << hret << std::endl;
            return false;
        }
    }

    CommMpiBcast(&rootInfo, HCCL_ROOT_INFO_BYTES, COMM_MPI_CHAR, 0);
    CommMpiBarrier();

    const uint16_t *a_rank = a_parts + (size_t)mpiRank * a_rank_elems;
    const uint16_t *b_rank = b_parts + (size_t)mpiRank * b_rank_elems;

    bool ok = RunGemmAllReducePerRank(mpiRank, n_ranks, device_id,
                                      a_rank, a_rank_bytes,
                                      b_rank, b_rank_bytes,
                                      golden, golden_bytes,
                                      &rootInfo);

    CommMpiBarrier();

    aclrtResetDevice(device_id);
    aclFinalize();

    return ok;
}
