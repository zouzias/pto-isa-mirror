/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <pto/pto-inst.hpp>
#include "acl/acl.h"

using namespace pto;

#define ALIGN_32B_ELEM(T) (32 / sizeof(T))
#define ALIGN_UP_32B(x, T) (((x) + ALIGN_32B_ELEM(T) - 1) / ALIGN_32B_ELEM(T) * ALIGN_32B_ELEM(T))

/**
 * Pingpong (double-buffer) fused weight kernel: bf16 x bf16 -> fp32
 * Pipeline: load buffer N+1 while computing buffer N for overlap optimization.
 * Each core shares UB space, TASSIGN uses block_idx offset.
 * GM/UB addresses and shape parameters must be 32B aligned.
 */
template <typename T, typename U, int Rows, int Cols, int TileChunkSize = 256>
__global__ AICORE void runEngramFusedWeightPingPong(__gm__ T *out, __gm__ U *src0, __gm__ U *src1)
{
    constexpr uint32_t BUFFER_NUM = 2;

    uint32_t blockNum = get_block_num();
    uint32_t blockIdx = get_block_idx();

    uint32_t totalElems = Rows * Cols;
    uint32_t alignedPerCore = ALIGN_UP_32B(totalElems / blockNum, U);
    uint32_t numChunks = alignedPerCore / TileChunkSize;
    uint32_t baseOffset = alignedPerCore * blockIdx;

    using DynShape = pto::Shape<1, 1, 1, 1, -1>;
    using DynStride = pto::Stride<-1, -1, -1, -1, 1>;
    using GlobalDataSrc = GlobalTensor<U, DynShape, DynStride>;
    using GlobalDataDst = GlobalTensor<T, DynShape, DynStride>;

    using TileDataSrc = Tile<TileType::Vec, U, 1, TileChunkSize, BLayout::RowMajor, 1, -1>;
    using TileDataFp32 = Tile<TileType::Vec, T, 1, TileChunkSize, BLayout::RowMajor, 1, -1>;

    uint32_t alignedChunkSize = ALIGN_UP_32B(TileChunkSize, U);
    TileDataSrc src0Tile[BUFFER_NUM] = {TileDataSrc(alignedChunkSize), TileDataSrc(alignedChunkSize)};
    TileDataSrc src1Tile[BUFFER_NUM] = {TileDataSrc(alignedChunkSize), TileDataSrc(alignedChunkSize)};
    TileDataFp32 src0Fp32Tile[BUFFER_NUM] = {TileDataFp32(alignedChunkSize), TileDataFp32(alignedChunkSize)};
    TileDataFp32 src1Fp32Tile[BUFFER_NUM] = {TileDataFp32(alignedChunkSize), TileDataFp32(alignedChunkSize)};
    TileDataFp32 dstTile[BUFFER_NUM] = {TileDataFp32(alignedChunkSize), TileDataFp32(alignedChunkSize)};

    uint32_t srcTileSize = TileChunkSize * sizeof(U);
    uint32_t fp32TileSize = TileChunkSize * sizeof(T);
    uint32_t singleBufferSize = 2 * srcTileSize + 3 * fp32TileSize;
    uint32_t alignedUBPerCore = ALIGN_UP_32B(BUFFER_NUM * singleBufferSize, uint8_t);
    uint32_t ubBaseOffset = alignedUBPerCore * blockIdx;

    for (uint32_t p = 0; p < BUFFER_NUM; p++) {
        uint32_t offset = ubBaseOffset + p * singleBufferSize;
        TASSIGN(src0Tile[p], offset);
        TASSIGN(src1Tile[p], offset + srcTileSize);
        TASSIGN(src0Fp32Tile[p], offset + 2 * srcTileSize);
        TASSIGN(src1Fp32Tile[p], offset + 2 * srcTileSize + fp32TileSize);
        TASSIGN(dstTile[p], offset + 2 * srcTileSize + 2 * fp32TileSize);
    }

    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);

    GlobalDataSrc src0Global0(src0 + baseOffset, DynShape(alignedChunkSize), DynStride(alignedChunkSize, alignedChunkSize, alignedChunkSize, alignedChunkSize));
    GlobalDataSrc src1Global0(src1 + baseOffset, DynShape(alignedChunkSize), DynStride(alignedChunkSize, alignedChunkSize, alignedChunkSize, alignedChunkSize));

    int8_t pingpongFlag = 0;

    wait_flag(PIPE_V, PIPE_MTE2, (event_t)pingpongFlag);

    TLOAD(src0Tile[pingpongFlag], src0Global0);
    TLOAD(src1Tile[pingpongFlag], src1Global0);

    set_flag(PIPE_MTE2, PIPE_V, (event_t)pingpongFlag);

    for (uint32_t chunkIdx = 0; chunkIdx < numChunks; chunkIdx++) {
        uint32_t currOffset = baseOffset + chunkIdx * TileChunkSize;
        uint32_t rawCurrSize = alignedPerCore - chunkIdx * TileChunkSize;
        uint32_t currSize = (chunkIdx == numChunks - 1) ? ALIGN_UP_32B(rawCurrSize, U) : TileChunkSize;
        if (rawCurrSize <= 0) break;

        if (chunkIdx + 1 < numChunks) {
            int8_t nextFlag = (pingpongFlag == 0) ? 1 : 0;
            uint32_t nextOffset = baseOffset + (chunkIdx + 1) * TileChunkSize;
            uint32_t rawNextSize = alignedPerCore - (chunkIdx + 1) * TileChunkSize;
            uint32_t nextSize = (chunkIdx + 1 == numChunks - 1) ? ALIGN_UP_32B(rawNextSize, U) : TileChunkSize;
            GlobalDataSrc src0GlobalNext(src0 + nextOffset, DynShape(nextSize), DynStride(nextSize, nextSize, nextSize, nextSize));
            GlobalDataSrc src1GlobalNext(src1 + nextOffset, DynShape(nextSize), DynStride(nextSize, nextSize, nextSize, nextSize));

            wait_flag(PIPE_V, PIPE_MTE2, (event_t)nextFlag);

            TLOAD(src0Tile[nextFlag], src0GlobalNext);
            TLOAD(src1Tile[nextFlag], src1GlobalNext);

            set_flag(PIPE_MTE2, PIPE_V, (event_t)nextFlag);
        }

        wait_flag(PIPE_MTE2, PIPE_V, (event_t)pingpongFlag);
        wait_flag(PIPE_MTE3, PIPE_V, (event_t)pingpongFlag);

        TCVT(src0Fp32Tile[pingpongFlag], src0Tile[pingpongFlag], RoundMode::CAST_NONE);
        TCVT(src1Fp32Tile[pingpongFlag], src1Tile[pingpongFlag], RoundMode::CAST_NONE);
        TMUL(dstTile[pingpongFlag], src0Fp32Tile[pingpongFlag], src1Fp32Tile[pingpongFlag]);

        set_flag(PIPE_V, PIPE_MTE2, (event_t)pingpongFlag);
        set_flag(PIPE_V, PIPE_MTE3, (event_t)pingpongFlag);

        uint32_t alignedDstSize = ALIGN_UP_32B(currSize, T);
        GlobalDataDst dstGlobalCurr(out + currOffset, DynShape(alignedDstSize), DynStride(alignedDstSize, alignedDstSize, alignedDstSize, alignedDstSize));

        wait_flag(PIPE_V, PIPE_MTE3, (event_t)pingpongFlag);

        TSTORE(dstGlobalCurr, dstTile[pingpongFlag]);

        set_flag(PIPE_MTE3, PIPE_V, (event_t)pingpongFlag);
        pingpongFlag = (pingpongFlag == 0) ? 1 : 0;
    }

    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
}

template <typename T, bool isBf16, int Rows, int Cols>
void LaunchEngramFusedWeight(void *out, void *src0, void *src1, void *stream)
{
    if constexpr (isBf16) {
        runEngramFusedWeightPingPong<T, bfloat16_t, Rows, Cols><<<32, nullptr, stream>>>((T *)out, (bfloat16_t *)src0, (bfloat16_t *)src1);
    } else {
        runEngramFusedWeightPingPong<T, T, Rows, Cols><<<32, nullptr, stream>>>((T *)out, (T *)src0, (T *)src1);
    }
}

template void LaunchEngramFusedWeight<float, true, 4, 4096>(void *out, void *src0, void *src1, void *stream);