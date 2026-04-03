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

#ifdef __CCE_AICORE__

using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
using Global = pto::GlobalTensor<half, ShapeDyn, StrideDyn, pto::Layout::ND>;
using TileData = pto::Tile<pto::TileType::Vec, half, G_BASE_M, G_BASE_N, pto::BLayout::RowMajor, -1, -1>;

constexpr size_t TILE_UB_BYTES = ((G_BASE_M * G_BASE_N * sizeof(half) + 1023) / 1024) * 1024;

struct CommParams {
    int myRank;
    int nRanks;
    int numRemoteRanks;
    int mTilesLocal;
    int kChunks;
    int numBlocksPerSrc;
    int tileSize;
    int numTiles;
};

AICORE inline CommParams BuildCommParams(
    __gm__ HcclDeviceContext* hcclCtx,
    volatile __gm__ TileFlagMatrix* flags)
{
    CommParams p;
    p.myRank = static_cast<int>(hcclCtx->rankId);
    p.nRanks = static_cast<int>(hcclCtx->rankNum);
    p.numRemoteRanks = p.nRanks - 1;
    int mTiles = static_cast<int>(G_M / G_BASE_M);
    p.mTilesLocal = mTiles / p.nRanks;
    p.kChunks = static_cast<int>(G_K / G_BASE_N);
    p.numBlocksPerSrc = p.mTilesLocal * p.kChunks;
    p.tileSize = flags->tile_size;
    p.numTiles = flags->num_tiles_per_src;
    return p;
}

AICORE inline void SetupLocalFlags(
    volatile __gm__ TileFlagMatrix* flags,
    volatile __gm__ int32_t* summaryBase,
    int myRank,
    int numTiles)
{
    for (int c = 0; c < numTiles; ++c) {
        SetTileFlagReady(flags, myRank, c);
    }
    SetLocalSummaryReady(summaryBase, myRank, numTiles);
}

AICORE inline void InitTileBuffers(TileData& pingTile, TileData& pongTile)
{
    TASSIGN(pingTile, 0x0);
    TASSIGN(pongTile, TILE_UB_BYTES);
}

AICORE inline int DestIdxToRank(int destIdx, int myRank)
{
    return (destIdx < myRank) ? destIdx : (destIdx + 1);
}

// 将一个 tile 对应的所有 block 通过 TPUT 传输到远端，完成后设置远端 flag
AICORE inline void TransferTileToRemote(
    __gm__ half* shmemInput,
    __gm__ HcclDeviceContext* hcclCtx,
    __gm__ TileFlagMatrix* remoteTileFlags,
    __gm__ int32_t* remoteSummarySrc,
    const ShapeDyn& tileShape,
    const StrideDyn& tileStride,
    TileData& pingTile,
    TileData& pongTile,
    const CommParams& p,
    int destRank,
    int tileIdx)
{
    int blkStart = tileIdx * p.tileSize;
    int blkEnd = blkStart + p.tileSize;
    if (blkEnd > p.numBlocksPerSrc) {
        blkEnd = p.numBlocksPerSrc;
    }

    __gm__ half* remoteInput = HcclRemotePtr(hcclCtx, shmemInput, destRank);
    for (int b = blkStart; b < blkEnd; ++b) {
        int miLocal = b / p.kChunks;
        int kb = b % p.kChunks;
        int miGlobal = p.myRank * p.mTilesLocal + miLocal;
        uint64_t colOff = static_cast<uint64_t>(kb * static_cast<int>(G_BASE_N));
        uint64_t rowOff = static_cast<uint64_t>(miGlobal) * G_BASE_M;
        uint64_t offset = rowOff * G_K + colOff;
        Global srcG(shmemInput + offset, tileShape, tileStride);
        Global dstG(remoteInput + offset, tileShape, tileStride);
        pto::comm::TPUT(dstG, srcG, pingTile, pongTile);
    }

    pipe_barrier(PIPE_ALL);
    dsb(DSB_DDR);
    SetRemoteTileFlagReady(remoteTileFlags, p.myRank, tileIdx, remoteSummarySrc);
}

// block 数少于 dest 数：按 work_id 均分
AICORE inline void DispatchFewBlocks(
    __gm__ half* shmemInput,
    __gm__ HcclDeviceContext* hcclCtx,
    volatile __gm__ TileFlagMatrix* flags,
    const ShapeDyn& tileShape,
    const StrideDyn& tileStride,
    TileData& pingTile,
    TileData& pongTile,
    const CommParams& p,
    int blockIdx,
    int numBlocks)
{
    __gm__ TileFlagMatrix* flagsMut = const_cast<__gm__ TileFlagMatrix*>(flags);
    int totalWork = p.numRemoteRanks * p.numTiles;

    for (int workId = blockIdx; workId < totalWork; workId += numBlocks) {
        int destIdx = workId / p.numTiles;
        int tileIdx = workId % p.numTiles;
        int destRank = DestIdxToRank(destIdx, p.myRank);

        __gm__ TileFlagMatrix* remoteTileFlags =
            reinterpret_cast<__gm__ TileFlagMatrix*>(HcclRemotePtr(hcclCtx, flagsMut, destRank));
        __gm__ int32_t* remoteSummarySrc =
            reinterpret_cast<__gm__ int32_t*>(
                reinterpret_cast<__gm__ uint8_t*>(HcclRemotePtr(hcclCtx, flagsMut, destRank))
                + TileFlagMatrixBytes(flags))
            + p.myRank;

        TransferTileToRemote(shmemInput, hcclCtx, remoteTileFlags, remoteSummarySrc,
                             tileShape, tileStride, pingTile, pongTile, p, destRank, tileIdx);
    }
}

// block 数 >= dest 数：每 dest 分配连续 block 区间
AICORE inline void DispatchManyBlocks(
    __gm__ half* shmemInput,
    __gm__ HcclDeviceContext* hcclCtx,
    volatile __gm__ TileFlagMatrix* flags,
    const ShapeDyn& tileShape,
    const StrideDyn& tileStride,
    TileData& pingTile,
    TileData& pongTile,
    const CommParams& p,
    int blockIdx,
    int numBlocks)
{
    __gm__ TileFlagMatrix* flagsMut = const_cast<__gm__ TileFlagMatrix*>(flags);
    int blocksPerDest = numBlocks / p.numRemoteRanks;
    if (blocksPerDest <= 0) {
        blocksPerDest = 1;
    }
    int destIdx = blockIdx / blocksPerDest;
    int localIdx = blockIdx % blocksPerDest;
    if (destIdx >= p.numRemoteRanks) {
        return;
    }

    int destRank = DestIdxToRank(destIdx, p.myRank);
    __gm__ TileFlagMatrix* remoteTileFlags =
        reinterpret_cast<__gm__ TileFlagMatrix*>(HcclRemotePtr(hcclCtx, flagsMut, destRank));
    __gm__ int32_t* remoteSummarySrc =
        reinterpret_cast<__gm__ int32_t*>(
            reinterpret_cast<__gm__ uint8_t*>(HcclRemotePtr(hcclCtx, flagsMut, destRank))
            + TileFlagMatrixBytes(flags))
        + p.myRank;

    int tilesPerBlock = (p.numTiles + blocksPerDest - 1) / blocksPerDest;
    int tileStart = localIdx * tilesPerBlock;
    int tileEnd = tileStart + tilesPerBlock;
    if (tileEnd > p.numTiles) {
        tileEnd = p.numTiles;
    }

    for (int tileIdx = tileStart; tileIdx < tileEnd; ++tileIdx) {
        TransferTileToRemote(shmemInput, hcclCtx, remoteTileFlags, remoteSummarySrc,
                             tileShape, tileStride, pingTile, pongTile, p, destRank, tileIdx);
    }
}

AICORE inline void CommAIVRoleStreamingParallel(
    __gm__ half* shmemInput,
    __gm__ TileFlagMatrix* tileFlags,
    __gm__ HcclDeviceContext* hcclCtx,
    int blockIdx,
    int numBlocks)
{
    volatile __gm__ TileFlagMatrix* flags = reinterpret_cast<volatile __gm__ TileFlagMatrix*>(tileFlags);
    CommParams p = BuildCommParams(hcclCtx, flags);

    if (p.numRemoteRanks <= 0 || numBlocks <= 0) {
        return;
    }

    volatile __gm__ int32_t* summaryBase = GetSummaryBase(flags);
    if (blockIdx == 0) {
        SetupLocalFlags(flags, summaryBase, p.myRank, p.numTiles);
    }

    TileData pingTile(G_BASE_M, G_BASE_N);
    TileData pongTile(G_BASE_M, G_BASE_N);
    InitTileBuffers(pingTile, pongTile);

    ShapeDyn tileShape(1, 1, 1, G_BASE_M, G_BASE_N);
    StrideDyn tileStride(G_BASE_M * G_K, G_BASE_M * G_K, G_BASE_M * G_K, G_K, 1);

    if (numBlocks < p.numRemoteRanks) {
        DispatchFewBlocks(shmemInput, hcclCtx, flags, tileShape, tileStride,
                          pingTile, pongTile, p, blockIdx, numBlocks);
    } else {
        DispatchManyBlocks(shmemInput, hcclCtx, flags, tileShape, tileStride,
                           pingTile, pongTile, p, blockIdx, numBlocks);
    }
}

#endif // __CCE_AICORE__

__global__ AICORE void RingCommStreamingKernel(
    __gm__ uint8_t* shmemInput,
    __gm__ uint8_t* tileFlags,
    __gm__ uint8_t* hcclCtxRaw,
    int blockNum)
{
#ifdef __CCE_AICORE__
    int blockIdx = get_block_idx();
    __gm__ HcclDeviceContext* hcclCtx = reinterpret_cast<__gm__ HcclDeviceContext*>(hcclCtxRaw);
    int nRanks = static_cast<int>(hcclCtx->rankNum);

    if (nRanks <= 1) {
        return;
    }

    CommAIVRoleStreamingParallel(
        reinterpret_cast<__gm__ half*>(shmemInput),
        reinterpret_cast<__gm__ TileFlagMatrix*>(tileFlags),
        hcclCtx, blockIdx, blockNum);
#endif
}

void launchRingCommStreaming(
    uint8_t* shmemInput,
    uint8_t* tileFlags,
    uint8_t* hcclCtx,
    int nRanks,
    void* stream)
{
    int numRemoteRanks = nRanks - 1;
    if (numRemoteRanks <= 0) {
        return;
    }
    int blocksPerDest = COMM_BLOCK_NUM / numRemoteRanks;
    if (blocksPerDest < 1) {
        blocksPerDest = 1;
    }
    int totalBlocks = numRemoteRanks * blocksPerDest;
    RingCommStreamingKernel<<<totalBlocks, nullptr, stream>>>(shmemInput, tileFlags, hcclCtx, totalBlocks);
}
