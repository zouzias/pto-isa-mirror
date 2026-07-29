/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#pragma once

#include <cstdint>

#if defined(AICORE)
#define CCU_GEMM_AR_RUNTIME_INLINE AICORE inline
#else
#define CCU_GEMM_AR_RUNTIME_INLINE static inline
#endif

#ifndef CONFIG_G_M
#define CONFIG_G_M 5416
#endif
#ifndef CONFIG_G_K
#define CONFIG_G_K 6144
#endif
#ifndef CONFIG_G_N
#define CONFIG_G_N 1408
#endif

static constexpr uint32_t G_ORIG_M = CONFIG_G_M;
static constexpr uint32_t G_ORIG_K = CONFIG_G_K;
static constexpr uint32_t G_ORIG_N = CONFIG_G_N;

#ifndef CONFIG_G_BASE_M
#define CONFIG_G_BASE_M 128
#endif
#ifndef CONFIG_G_BASE_K
#define CONFIG_G_BASE_K 64
#endif
#ifndef CONFIG_G_BASE_N
#define CONFIG_G_BASE_N 256
#endif

static constexpr uint32_t G_BASE_M = CONFIG_G_BASE_M;
static constexpr uint32_t G_BASE_K = CONFIG_G_BASE_K;
static constexpr uint32_t G_BASE_N = CONFIG_G_BASE_N;

static constexpr uint32_t CeilDiv(uint32_t a, uint32_t b) { return (b == 0) ? 0 : (a + b - 1) / b; }
static constexpr uint32_t AlignUp(uint32_t a, uint32_t b) { return CeilDiv(a, b) * b; }

static constexpr uint32_t G_M = AlignUp(G_ORIG_M, G_BASE_M);
static constexpr uint32_t G_K = G_ORIG_K;
static constexpr uint32_t G_N = AlignUp(G_ORIG_N, G_BASE_N);
static constexpr uint32_t G_M_TILES = G_M / G_BASE_M;
static constexpr uint32_t G_N_TILES = G_N / G_BASE_N;
static constexpr uint32_t G_NUM_TILES = G_M_TILES * G_N_TILES;

#ifndef CONFIG_COMPUTE_BLOCK_NUM
#define CONFIG_COMPUTE_BLOCK_NUM 24
#endif
#ifndef CONFIG_COMM_BLOCK_NUM
#define CONFIG_COMM_BLOCK_NUM 24
#endif
static constexpr int COMPUTE_BLOCK_NUM = CONFIG_COMPUTE_BLOCK_NUM;
static constexpr int COMM_BLOCK_NUM = CONFIG_COMM_BLOCK_NUM;
static constexpr int MAX_RANKS = 8;

#ifndef CONFIG_COMM_SUB_M
#define CONFIG_COMM_SUB_M 128
#endif
static constexpr uint32_t G_COMM_SUB_M = CONFIG_COMM_SUB_M;
static_assert(G_COMM_SUB_M > 0, "CONFIG_COMM_SUB_M must be positive");
static_assert(G_BASE_M % G_COMM_SUB_M == 0, "CONFIG_COMM_SUB_M must divide G_BASE_M");
static constexpr uint32_t G_COMM_SUBTILES_PER_TILE = G_BASE_M / G_COMM_SUB_M;

#ifndef CONFIG_COMM_GROUP_TILES
// Default 16: good 2-rank balance. For 4-rank Pipelined, prefer
// --comm-group-tiles 8 (see docs sweep). Residual last groups OK: owner shard
// padded to a multiple of G_COMM_GROUP_TILES (zero-fill).
#define CONFIG_COMM_GROUP_TILES 16
#endif
static constexpr uint32_t G_COMM_GROUP_TILES = CONFIG_COMM_GROUP_TILES;
static_assert(G_COMM_GROUP_TILES > 0, "CONFIG_COMM_GROUP_TILES must be positive");
static_assert(
    G_COMM_SUBTILES_PER_TILE == 1, "Split RS/AG scheduler requires CONFIG_COMM_SUB_M == G_BASE_M (subtiles == 1)");

static constexpr uint32_t G_NUM_GROUPS = (G_NUM_TILES + G_COMM_GROUP_TILES - 1) / G_COMM_GROUP_TILES;
static constexpr uint32_t G_PADDED_TILES = G_NUM_GROUPS * G_COMM_GROUP_TILES;
static constexpr uint64_t G_TILE_BYTES = static_cast<uint64_t>(G_BASE_M) * G_BASE_N * sizeof(uint16_t);
static constexpr uint64_t G_GROUP_BYTES = static_cast<uint64_t>(G_COMM_GROUP_TILES) * G_TILE_BYTES;
static constexpr uint64_t G_PADDED_OUTPUT_BYTES =
    static_cast<uint64_t>(G_PADDED_TILES) * G_BASE_M * G_BASE_N * sizeof(uint16_t);

CCU_GEMM_AR_RUNTIME_INLINE uint32_t CcuOwnerTileCount(uint32_t owner, uint32_t numTiles, uint32_t rankSize)
{
    const uint32_t safe = (rankSize > 0) ? rankSize : 1;
    return (owner < numTiles) ? ((numTiles - 1 - owner) / safe + 1) : 0;
}

// Owner shard capacity in the packed buffer, rounded up so every CCU group
// (including a residual last group) can use a fixed G_GROUP_BYTES payload.
CCU_GEMM_AR_RUNTIME_INLINE uint32_t CcuOwnerPaddedTileCount(uint32_t owner, uint32_t numTiles, uint32_t rankSize)
{
    const uint32_t tiles = CcuOwnerTileCount(owner, numTiles, rankSize);
    if (tiles == 0 || G_COMM_GROUP_TILES == 0)
        return 0;
    return ((tiles + G_COMM_GROUP_TILES - 1) / G_COMM_GROUP_TILES) * G_COMM_GROUP_TILES;
}

CCU_GEMM_AR_RUNTIME_INLINE uint32_t CcuOwnerTilePrefix(uint32_t owner, uint32_t numTiles, uint32_t rankSize)
{
    const uint32_t safe = (rankSize > 0) ? rankSize : 1;
    uint32_t prefix = 0;
    for (uint32_t r = 0; r < owner && r < safe; ++r) {
        prefix += CcuOwnerPaddedTileCount(r, numTiles, safe);
    }
    return prefix;
}

CCU_GEMM_AR_RUNTIME_INLINE uint32_t CcuPackedTileIndex(uint32_t tile, uint32_t numTiles, uint32_t rankSize)
{
    const uint32_t safe = (rankSize > 0) ? rankSize : 1;
    const uint32_t owner = tile % safe;
    return CcuOwnerTilePrefix(owner, numTiles, safe) + tile / safe;
}

CCU_GEMM_AR_RUNTIME_INLINE uint32_t CcuTotalPaddedTiles(uint32_t numTiles, uint32_t rankSize)
{
    const uint32_t safe = (rankSize > 0) ? rankSize : 1;
    uint32_t total = 0;
    for (uint32_t r = 0; r < safe; ++r) {
        total += CcuOwnerPaddedTileCount(r, numTiles, safe);
    }
    return total;
}

CCU_GEMM_AR_RUNTIME_INLINE uint32_t CcuOwnerGroupCount(uint32_t owner, uint32_t numTiles, uint32_t rankSize)
{
    const uint32_t tiles = CcuOwnerTileCount(owner, numTiles, rankSize);
    return (tiles + G_COMM_GROUP_TILES - 1) / G_COMM_GROUP_TILES;
}

CCU_GEMM_AR_RUNTIME_INLINE uint32_t CcuOwnerGroupPrefix(uint32_t owner, uint32_t numTiles, uint32_t rankSize)
{
    const uint32_t safe = (rankSize > 0) ? rankSize : 1;
    uint32_t prefix = 0;
    for (uint32_t r = 0; r < owner && r < safe; ++r) {
        prefix += CcuOwnerGroupCount(r, numTiles, safe);
    }
    return prefix;
}

// Flat groupDone index for owner-local group g.
CCU_GEMM_AR_RUNTIME_INLINE uint32_t
CcuOwnerGroupFlatIndex(uint32_t owner, uint32_t ownerLocalGroup, uint32_t numTiles, uint32_t rankSize)
{
    return CcuOwnerGroupPrefix(owner, numTiles, rankSize) + ownerLocalGroup;
}

CCU_GEMM_AR_RUNTIME_INLINE uint32_t
CcuOwnerGroupTilesInGroup(uint32_t owner, uint32_t ownerLocalGroup, uint32_t numTiles, uint32_t rankSize)
{
    const uint32_t ownerTiles = CcuOwnerTileCount(owner, numTiles, rankSize);
    const uint32_t start = ownerLocalGroup * G_COMM_GROUP_TILES;
    if (start >= ownerTiles)
        return 0;
    const uint32_t remain = ownerTiles - start;
    return (remain < G_COMM_GROUP_TILES) ? remain : G_COMM_GROUP_TILES;
}

// Per-rank signal_matrix layout (HCCL window — CommRemotePtr-visible):
//   [0 .. GROUP_DONE)                 small reserved prefix
//   [GROUP_DONE .. GROUP_READY)       per-owner-group tile-done (AIC AtomicAdd, local AIV poll)
//   [GROUP_READY .. MATRIX_SLOTS)     per-owner-group peer-ready (peer TNOTIFY, owner TTEST)
static constexpr uint32_t G_SIGNAL_RESERVED_PREFIX_SLOTS = MAX_RANKS + 2;
// Progress sync:
//   groupDone[flat]:  AIC per-tile AtomicAdd on the producing rank's window (local poll).
//   groupReady[flat]: peer AIV TNOTIFY(+1) on the owner window when that peer's local
//                     groupDone reaches expected; owner waits >= (rankSize-1).
static constexpr uint32_t G_SIGNAL_GROUP_DONE_OFFSET = G_SIGNAL_RESERVED_PREFIX_SLOTS;
static constexpr uint32_t G_SIGNAL_GROUP_DONE_STRIDE = 16; // ints per counter (= 64B line)
static constexpr uint32_t G_SIGNAL_GROUP_DONE_MAX_GROUPS = 512;
static constexpr uint32_t G_SIGNAL_GROUP_DONE_SLOTS = G_SIGNAL_GROUP_DONE_MAX_GROUPS * G_SIGNAL_GROUP_DONE_STRIDE;
static_assert(G_SIGNAL_GROUP_DONE_STRIDE * sizeof(int32_t) >= 64, "groupDone slot must be at least one cache line");
static constexpr uint32_t G_SIGNAL_GROUP_READY_OFFSET = G_SIGNAL_GROUP_DONE_OFFSET + G_SIGNAL_GROUP_DONE_SLOTS;
static constexpr uint32_t G_SIGNAL_GROUP_READY_STRIDE = G_SIGNAL_GROUP_DONE_STRIDE;
static constexpr uint32_t G_SIGNAL_GROUP_READY_MAX_GROUPS = G_SIGNAL_GROUP_DONE_MAX_GROUPS;
static constexpr uint32_t G_SIGNAL_GROUP_READY_SLOTS = G_SIGNAL_GROUP_READY_MAX_GROUPS * G_SIGNAL_GROUP_READY_STRIDE;
static_assert(G_SIGNAL_GROUP_READY_STRIDE * sizeof(int32_t) >= 64, "groupReady slot must be at least one cache line");
static constexpr uint32_t G_SIGNAL_MATRIX_SLOTS = G_SIGNAL_GROUP_READY_OFFSET + G_SIGNAL_GROUP_READY_SLOTS;

#ifndef CONFIG_CCU_MISSION_PARALLEL
#define CONFIG_CCU_MISSION_PARALLEL 1
#endif
static constexpr uint32_t CCU_MISSION_PARALLEL = CONFIG_CCU_MISSION_PARALLEL;
// Stable path: one CCU mission per rank. The current gate registry publishes a
// single gate descriptor per rank, so K>1 can leave earlier missions stuck at
// WaitEvent(gate). Re-enable K>1 only after per-mission gate descriptors exist.
static_assert(CCU_MISSION_PARALLEL == 1, "ccu_gemm_ar stable path requires CONFIG_CCU_MISSION_PARALLEL=1");

// In-flight depth of the AIV -> CCU progress handshake, i.e. how many groups the
// AIV may have poked but the CCU not yet retired.
//   1 = stable path. One channel/peer; Reduce then Broadcast+wait (serial).
//   2 = dual progress CKE + Reduce(g)||Broadcast(g-1). Requires a second CCU
//       channel per peer (bcastChannels) so Read and Write do not share one SQ.
#ifndef CONFIG_CCU_PIPE_DEPTH
#define CONFIG_CCU_PIPE_DEPTH 1
#endif
static constexpr uint32_t CCU_PIPE_DEPTH = CONFIG_CCU_PIPE_DEPTH;
static_assert(CCU_PIPE_DEPTH == 1 || CCU_PIPE_DEPTH == 2, "ccu_gemm_ar supports CONFIG_CCU_PIPE_DEPTH of 1 or 2 only");

// One progress CKE slot per in-flight group, so consecutive pokes never collide.
static constexpr uint32_t CCU_PROGRESS_SLOTS = CCU_PIPE_DEPTH;
static constexpr uint32_t CCU_MAX_PROGRESS_SLOTS = 2;

static constexpr int WARMUP_ITERS = 5;
static constexpr int COMPUTE_ONLY_ITERS = 5;
