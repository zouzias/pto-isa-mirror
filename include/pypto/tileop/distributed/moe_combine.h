/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_combine.h
 * \brief
*/

#ifndef __DISTRIBUTED_COMBINE__
#define __DISTRIBUTED_COMBINE__

#include "common.h"
#include <type_traits>

namespace TileOp::Distributed {
template <typename T, uint32_t topK, uint16_t rowShape, uint16_t colShape, uint16_t paddedColShape>
TILEOP void MoeDistributedCombineSend(
    __ubuf__ T* dataBuffer,
    __ubuf__ int32_t* assistInfoForCombineBuffer,
    __ubuf__ int32_t* signalBuffer,
    __gm__ T* expandX,
    __gm__ int32_t* assistInfoForCombine,
    __ubuf__ int32_t* recvCounts,
    __gm__ T* shmemDataBaseAddr,
    __gm__ int32_t* shmemSignalBaseAddr,
    uint64_t expandXOffset0,
    uint64_t expandXOffset1,
    __gm__ int64_t* hcclContext)
{
    signalBuffer[0] = 1;
    uint64_t maxRow = ((expandXOffset0 + rowShape) < recvCounts[0]) ? (expandXOffset0 + rowShape) : recvCounts[0];
    for (uint64_t row = expandXOffset0; row < maxRow; row++) {
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        TileOp::UBCopyIn<int32_t, 1, 3, 8, 3>(assistInfoForCombineBuffer, assistInfoForCombine + MOE_COMBINE_INFO_NUM * row);

        set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
        int32_t rankId = assistInfoForCombineBuffer[0];
        int32_t tokenId = assistInfoForCombineBuffer[1];
        int32_t kOffset = assistInfoForCombineBuffer[2];

        TileOp::UBCopyIn<T, 1, colShape, paddedColShape, colShape>(dataBuffer, expandX + colShape * row);

        __gm__ T* winDataAddr = MapVirtualAddr<T>(hcclContext, shmemDataBaseAddr, rankId) +
            colShape * (topK * tokenId + kOffset);
        set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        TileOp::UBCopyOut<T, 1, colShape, colShape, paddedColShape>(winDataAddr, dataBuffer);

        __gm__ int32_t* winSignalAddr = MapVirtualAddr<int32_t>(hcclContext, shmemSignalBaseAddr, rankId) +
            MOE_COMBINE_SIGNAL_OFFSET * tokenId;
        set_atomic_add();
        set_atomic_s32();
        pipe_barrier(PIPE_MTE3);
        copy_ubuf_to_gm(winSignalAddr, signalBuffer, 0, 1, 1, 0, 0);
        set_atomic_none();
    }
}

TILEOP void MoeDistributedCombineWaitSignal(
    __gm__ int32_t* winSignalAddr,
    __ubuf__ int32_t* signalBuffer,
    int32_t expectedValue)
{
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    do {
        copy_gm_to_ubuf(signalBuffer, winSignalAddr, 0, 1, 1, 0, 0);
        set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
    } while (signalBuffer[0] != expectedValue);
}

template <typename T, uint32_t topK, uint16_t colShape, uint16_t paddedColShape>
TILEOP void MoeDistributedCombineCompute(
    __ubuf__ T* out,
    __ubuf__ float* mulFp32Buffer,
    __ubuf__ float* sumFp32Buffer,
    __ubuf__ float* expertScales,
    __gm__ T* winDataAddr)
{
    uint8_t repeat = static_cast<uint8_t>(
        AlignUp<uint16_t>(sizeof(float) * paddedColShape, VECTOR_INSTRUCTION_BYTE_SIZE) / VECTOR_INSTRUCTION_BYTE_SIZE
    );

    vector_dup(sumFp32Buffer, 0.0f, repeat, 1, 1, 8, 8);

    for (int kOffset = 0; kOffset < topK; kOffset++) {
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        TileOp::UBCopyIn<T, 1, colShape, paddedColShape, colShape>(out, winDataAddr);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        vconv_bf162f32(mulFp32Buffer, out, repeat, 1, 1, 8, 4);
        pipe_barrier(PIPE_V);
        vmuls(mulFp32Buffer, mulFp32Buffer, static_cast<float>(expertScales[kOffset]), repeat, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        vadd(sumFp32Buffer, mulFp32Buffer, sumFp32Buffer, repeat, 1, 1, 1, 8, 8, 8);
        winDataAddr += colShape;
    }

    pipe_barrier(PIPE_V);
    vconv_f322bf16a(out, sumFp32Buffer, repeat, 1, 1, 4, 8);
}

template <typename T, uint32_t topK, uint16_t rowShape, uint16_t colShape, uint16_t paddedColShape>
TILEOP void MoeDistributedCombineReceive(
    __gm__ T* out,
    __ubuf__ float* mulFp32Buffer,
    __ubuf__ float* sumFp32Buffer,
    __ubuf__ T* outBuffer,
    __ubuf__ float* expertScales,
    __gm__ T* shmemDataBaseAddr,
    __gm__ int32_t* shmemSignalBaseAddr,
    uint64_t shmemDataOffset0,
    uint64_t shmemDataOffset1,
    uint64_t shmemDataOffset2,
    uint64_t shmemDataOffset3,
    int64_t rowOffset,
    __gm__ int64_t* hcclContext)
{
    uint64_t thisRankId = shmemDataOffset0;

    for (uint64_t tokenId = rowOffset; tokenId < rowOffset + rowShape; tokenId++) {
        __gm__ int32_t* winSignalAddr = MapVirtualAddr<int32_t>(hcclContext, shmemSignalBaseAddr, thisRankId) +
            MOE_COMBINE_SIGNAL_OFFSET * tokenId;
        __ubuf__ int32_t* signalBuffer = reinterpret_cast<__ubuf__ int32_t*>(outBuffer);
        MoeDistributedCombineWaitSignal(winSignalAddr, signalBuffer, topK);

        constexpr uint32_t expertScalesColShape = AlignUp<uint32_t>(sizeof(float) * topK, COPY_BLOCK_BYTE_SIZE) /
            sizeof(float);
        __gm__ T* winDataAddr = MapVirtualAddr<T>(hcclContext, shmemDataBaseAddr, thisRankId) +
            colShape * topK * tokenId;
        MoeDistributedCombineCompute<T, topK, colShape, paddedColShape>(outBuffer, mulFp32Buffer,
            sumFp32Buffer, expertScales + expertScalesColShape * tokenId, winDataAddr);

        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        TileOp::UBCopyOut<T, 1, colShape, colShape, paddedColShape>(out + colShape * tokenId, outBuffer);
    }
}
} // namespace TileOp::Distributed

#endif