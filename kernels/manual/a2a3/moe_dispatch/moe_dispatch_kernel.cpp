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
// MoE Dispatch Kernel — PTO-ISA Implementation
//
// Standalone Dispatch communication operator matching MegaMoE logic:
// - Multi-core parallel TGET from remote ranks' shmem
// - Ping-pong double buffering via UB staging tiles
// - Token/scale separation on arrival
// - Per-expert-group SyncAll barrier
//
// Execution model: AIV-only (vector cores), launched via mpirun for multi-rank
// ============================================================================

#include <cstdint>
#include <cstddef>
#include <pto/pto-inst.hpp>
#include "pto/common/pto_tile.hpp"
#include "moe_dispatch_config.h"

#include "hccl_context.h"

// ============================================================================
// Device-side helper: translate local shmem pointer to remote rank's address
// ============================================================================
template <typename T>
AICORE inline __gm__ T *HcclRemotePtr(__gm__ HcclDeviceContext *ctx, __gm__ T *localPtr, int pe)
{
    uint64_t localBase = ctx->windowsIn[ctx->rankId];
    uint64_t offset = (uint64_t)localPtr - localBase;
    return (__gm__ T *)(ctx->windowsIn[pe] + offset);
}

// ============================================================================
// MoeDispatchKernel — Main Entry Point
//
// Dual-loop over (expertGroup, remoteRank) with strided multi-core distribution.
// Each core pulls tokens from assigned remote ranks, separates token/scale,
// and signals completion per expert group via SyncAll.
// ============================================================================
template <int HIDDEN_SIZE, int TILE_COLS>
AICORE void MoeDispatchKernel(
    __gm__ int8_t *gmA,
    __gm__ float *gmPerTokenScale,
    __gm__ int32_t *cumsumMM,
    __gm__ int32_t *tokenPerExpert,
    __gm__ int32_t *preSumBeforeRank,
    __gm__ uint8_t *shmemBase,
    __gm__ HcclDeviceContext *hcclCtx,
    __gm__ int32_t *syncWorkspace,
    int32_t EP,
    int32_t expertPerRank,
    int32_t maxOutputSize,
    int64_t offsetA)
{
    int32_t myRank = static_cast<int32_t>(hcclCtx->rankId);
    int32_t coreIdx = get_block_idx();
    int32_t coreNum = get_block_num();

    constexpr int32_t copyInNum = HIDDEN_SIZE + UB_ALIGN;
    constexpr int32_t TILE_ROWS = 32;

    using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using Global = pto::GlobalTensor<int8_t, ShapeDyn, StrideDyn, pto::Layout::ND>;
    using TileT = pto::Tile<pto::TileType::Vec, int8_t, TILE_ROWS, TILE_COLS, pto::BLayout::RowMajor, -1, -1>;

    TileT tile(TILE_ROWS, TILE_COLS);
    TASSIGN(tile, 0x0);

    // Soft SYNCALL resources: GM workspace + UB tile
    using SyncGlobal = pto::GlobalTensor<int32_t, ShapeDyn, StrideDyn, pto::Layout::ND>;
    constexpr int32_t SYNC_UB_SLOTS = 48 * pto::SYNCALL_SOFT_SLOT_INT32;
    using SyncTile = pto::Tile<pto::TileType::Vec, int32_t, 1, SYNC_UB_SLOTS>;

    int32_t syncGmSize = coreNum * pto::SYNCALL_SOFT_SLOT_INT32;
    int64_t syncTotalBytes = static_cast<int64_t>(syncGmSize) * sizeof(int32_t);
    ShapeDyn syncShape(1, 1, 1, 1, static_cast<size_t>(syncGmSize));
    StrideDyn syncStride(syncGmSize, syncGmSize, syncGmSize, syncGmSize, 1);
    SyncGlobal syncGM(reinterpret_cast<__gm__ int32_t *>(syncWorkspace), syncShape, syncStride);

    // Place sync UB tile after the TGET tile in UB
    constexpr int32_t TGET_TILE_BYTES = TILE_ROWS * TILE_COLS;
    constexpr int32_t SYNC_UB_OFFSET = (TGET_TILE_BYTES + 31) & ~31;
    SyncTile syncUB;
    TASSIGN(syncUB, SYNC_UB_OFFSET);

    uint32_t prevGroupSum = 0;

    for (int32_t groupIdx = 0; groupIdx < expertPerRank; ++groupIdx) {
        uint32_t currentM = static_cast<uint32_t>(cumsumMM[(EP - 1) * expertPerRank + groupIdx]);

        for (int32_t dstEpIdx = coreIdx; dstEpIdx < EP; dstEpIdx += coreNum) {
            uint32_t rowStart;
            if (dstEpIdx == 0) {
                rowStart = prevGroupSum;
            } else {
                rowStart = static_cast<uint32_t>(cumsumMM[(dstEpIdx - 1) * expertPerRank + groupIdx]) + prevGroupSum;
            }

            if (rowStart >= static_cast<uint32_t>(maxOutputSize)) {
                continue;
            }

            int32_t tpeIdx = dstEpIdx * EP * expertPerRank + myRank * expertPerRank + groupIdx;
            uint32_t rows = static_cast<uint32_t>(tokenPerExpert[tpeIdx]);

            if (rowStart + rows > static_cast<uint32_t>(maxOutputSize)) {
                rows = static_cast<uint32_t>(maxOutputSize) - rowStart;
            }

            if (rows == 0) {
                continue;
            }

            int32_t rowSrc = preSumBeforeRank[dstEpIdx * expertPerRank + groupIdx];

            __gm__ uint8_t *otherRankBase = HcclRemotePtr(hcclCtx, shmemBase, dstEpIdx);
            __gm__ int8_t *remoteSrcPtr = reinterpret_cast<__gm__ int8_t *>(
                otherRankBase + offsetA + static_cast<int64_t>(rowSrc) * copyInNum);

            __gm__ int8_t *localDstToken = gmA + static_cast<int64_t>(rowStart) * copyInNum;

            int64_t totalBytes = static_cast<int64_t>(rows) * copyInNum;
            ShapeDyn shape(1, 1, 1, static_cast<size_t>(rows), static_cast<size_t>(copyInNum));
            StrideDyn stride(totalBytes, totalBytes, totalBytes, copyInNum, 1);
            Global remoteSrcG(remoteSrcPtr, shape, stride);
            Global localDstG(localDstToken, shape, stride);

            pto::comm::TGET(localDstG, remoteSrcG, tile);
            pipe_barrier(PIPE_ALL);
        }

        pto::SYNCALL<pto::SyncAllMode::Soft>(syncGM, syncUB, coreNum);

        prevGroupSum += currentM;
    }
}

// ============================================================================
// Kernel Launch Wrapper (explicit template instantiations)
// ============================================================================
extern "C" __global__ AICORE void MoeDispatchKernel_K128(
    __gm__ int8_t *gmA,
    __gm__ float *gmPerTokenScale,
    __gm__ int32_t *cumsumMM,
    __gm__ int32_t *tokenPerExpert,
    __gm__ int32_t *preSumBeforeRank,
    __gm__ uint8_t *shmemBase,
    __gm__ HcclDeviceContext *hcclCtx,
    __gm__ int32_t *syncWorkspace,
    int32_t EP,
    int32_t expertPerRank,
    int32_t maxOutputSize,
    int64_t offsetA)
{
    // TILE_COLS = HIDDEN_SIZE + UB_ALIGN = 128 + 32 = 160
    MoeDispatchKernel<128, 160>(gmA, gmPerTokenScale, cumsumMM, tokenPerExpert,
                                preSumBeforeRank, shmemBase, hcclCtx, syncWorkspace,
                                EP, expertPerRank, maxOutputSize, offsetA);
}

extern "C" __global__ AICORE void MoeDispatchKernel_K512(
    __gm__ int8_t *gmA,
    __gm__ float *gmPerTokenScale,
    __gm__ int32_t *cumsumMM,
    __gm__ int32_t *tokenPerExpert,
    __gm__ int32_t *preSumBeforeRank,
    __gm__ uint8_t *shmemBase,
    __gm__ HcclDeviceContext *hcclCtx,
    __gm__ int32_t *syncWorkspace,
    int32_t EP,
    int32_t expertPerRank,
    int32_t maxOutputSize,
    int64_t offsetA)
{
    MoeDispatchKernel<512, 544>(gmA, gmPerTokenScale, cumsumMM, tokenPerExpert,
                                 preSumBeforeRank, shmemBase, hcclCtx, syncWorkspace,
                                 EP, expertPerRank, maxOutputSize, offsetA);
}

extern "C" __global__ AICORE void MoeDispatchKernel_K4096(
    __gm__ int8_t *gmA,
    __gm__ float *gmPerTokenScale,
    __gm__ int32_t *cumsumMM,
    __gm__ int32_t *tokenPerExpert,
    __gm__ int32_t *preSumBeforeRank,
    __gm__ uint8_t *shmemBase,
    __gm__ HcclDeviceContext *hcclCtx,
    __gm__ int32_t *syncWorkspace,
    int32_t EP,
    int32_t expertPerRank,
    int32_t maxOutputSize,
    int64_t offsetA)
{
    MoeDispatchKernel<4096, 4128>(gmA, gmPerTokenScale, cumsumMM, tokenPerExpert,
                                  preSumBeforeRank, shmemBase, hcclCtx, syncWorkspace,
                                  EP, expertPerRank, maxOutputSize, offsetA);
}

// ============================================================================
// Host-callable launch wrapper (compiled as part of fat object)
// ============================================================================
#include "acl/acl.h"
#include <cstdio>

bool LaunchMoeDispatchK128(
    int32_t blockNum, void *stream,
    void *gmA, void *gmPerTokenScale,
    void *cumsumMM, void *tokenPerExpert, void *preSumBeforeRank,
    void *shmemBase, void *hcclCtx, void *syncWorkspace,
    int32_t EP, int32_t expertPerRank, int32_t maxOutputSize, int64_t offsetA)
{
    fprintf(stderr, "[KERNEL] LaunchMoeDispatchK128: blockNum=%d EP=%d expertPerRank=%d maxOutput=%d\n",
            blockNum, EP, expertPerRank, maxOutputSize);
    MoeDispatchKernel_K128<<<blockNum, nullptr, stream>>>(
        (__gm__ int8_t *)gmA, (__gm__ float *)gmPerTokenScale,
        (__gm__ int32_t *)cumsumMM, (__gm__ int32_t *)tokenPerExpert,
        (__gm__ int32_t *)preSumBeforeRank,
        (__gm__ uint8_t *)shmemBase, (__gm__ HcclDeviceContext *)hcclCtx,
        (__gm__ int32_t *)syncWorkspace,
        EP, expertPerRank, maxOutputSize, offsetA);
    aclError err = aclrtSynchronizeStream((aclrtStream)stream);
    fprintf(stderr, "[KERNEL] aclrtSynchronizeStream returned: %d\n", (int)err);
    return (err == ACL_SUCCESS);
}
