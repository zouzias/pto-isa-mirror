/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file tileop_shmem_reduce.h
 * \brief Shmem reduce helpers and implementation.
 */

#ifndef __DISTRIBUTED_TILEOP_SHMEM_REDUCE__
#define __DISTRIBUTED_TILEOP_SHMEM_REDUCE__

#include "common.h"
#include <type_traits>

namespace TileOp::Distributed {

// UB 类型转换：half/bf16 <-> float，仅 reduce FP32 模式使用
template<typename T>
TILEOP void Conv2FP32(__ubuf__ float* dst, __ubuf__ T* src, uint8_t repeat, uint16_t dstBlockStride,
    uint16_t srcBlockStride, uint8_t dstRepeatStride, uint8_t srcRepeatStride)
{
    if constexpr (std::is_same_v<T, half>) {
        vconv_f162f32(dst, src, repeat, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
    } else if constexpr (std::is_same_v<T, bfloat16_t>) {
        vconv_bf162f32(dst, src, repeat, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
    }
}

template<typename T>
TILEOP void DeConvFP32(__ubuf__ T* dst, __ubuf__ float* src, uint8_t repeat, uint16_t dstBlockStride,
    uint16_t srcBlockStride, uint8_t dstRepeatStride, uint8_t srcRepeatStride)
{
    if constexpr (std::is_same_v<T, half>) {
        vconv_f322f16(dst, src, repeat, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
    } else if constexpr (std::is_same_v<T, bfloat16_t>) {
        vconv_f322bf16r(dst, src, repeat, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
    }
}

template<typename T>
TILEOP void ReduceTLoad(__gm__ T* gmAddr, __ubuf__ T* ubAddr, CopyParams params)
{
    copy_gm_to_ubuf(ubAddr, gmAddr, 0, params.nBurst, params.lenBurst, params.srcStride, params.dstStride);
    PIPE_SYNC_EVENT(PIPE_MTE2, PIPE_S, EVENT_ID0);
}

template<typename T>
TILEOP void ReduceTStore(__gm__ T* gmAddr, __ubuf__ T* ubAddr, CopyParams params)
{
    copy_ubuf_to_gm(gmAddr, ubAddr, 0, params.nBurst, params.lenBurst, params.srcStride, params.dstStride);
    PIPE_SYNC_EVENT(PIPE_MTE3, PIPE_S, EVENT_ID0);
}

template<typename T, bool FP32Mode>
struct ShmemReduceProcess {};

constexpr uint64_t REDUCE_CHUNK_BYTES = 256;
constexpr uint64_t REDUCE_FP32_REPEAT_STRIDE = REDUCE_CHUNK_BYTES / sizeof(float);  // 64

template<typename T>
struct ShmemReduceProcess<T, true> {
    TILEOP void ShmemReduceCopyIn(__gm__ T* x, __ubuf__ T* ubTensor, int64_t row, int64_t col, CopyParams params)
    {
        __ubuf__ T* copyUb = ubTensor;
        __ubuf__ float* sumUb = (__ubuf__ float*)(ubTensor + row * col);
        ReduceTLoad<T>(x, copyUb, params);
        const uint64_t repeat = static_cast<uint64_t>(row) * static_cast<uint64_t>(col) * sizeof(float) / REDUCE_CHUNK_BYTES;
        for (uint64_t i = 0; i < repeat; i++) {
            Conv2FP32<T>(sumUb + i * REDUCE_FP32_REPEAT_STRIDE, copyUb + i * REDUCE_FP32_REPEAT_STRIDE, 1, 1, 1, 8, 8);
            PIPE_SYNC_EVENT(PIPE_V, PIPE_S, EVENT_ID0);
        }
    }

    TILEOP void ShmemReduceCopyOut(__gm__ T* out, __ubuf__ T* ubTensor, int64_t row, int64_t col, CopyParams params)
    {
        __ubuf__ T* copyUb = ubTensor;
        __ubuf__ float* sumUb = (__ubuf__ float*)(ubTensor + row * col);
        const uint64_t repeat = static_cast<uint64_t>(row) * static_cast<uint64_t>(col) * sizeof(float) / REDUCE_CHUNK_BYTES;
        for (uint64_t i = 0; i < repeat; i++) {
            DeConvFP32<T>(copyUb + i * REDUCE_FP32_REPEAT_STRIDE, sumUb + i * REDUCE_FP32_REPEAT_STRIDE, 1, 1, 1, 8, 8);
            PIPE_SYNC_EVENT(PIPE_V, PIPE_S, EVENT_ID0);
        }
        ReduceTStore<T>(out, copyUb, params);
    }

    TILEOP void ShmemReduceAdd(__gm__ T* x, __ubuf__ T* ubTensor, int64_t row, int64_t col, CopyParams params)
    {
        __ubuf__ T* copyUb = ubTensor;
        __ubuf__ float* sumUb = (__ubuf__ float*)(ubTensor + row * col);
        __ubuf__ float* tempUb = sumUb + row * col;
        ReduceTLoad<T>(x, copyUb, params);
        const uint64_t repeat = static_cast<uint64_t>(row) * static_cast<uint64_t>(col) * sizeof(float) / REDUCE_CHUNK_BYTES;
        for (uint64_t i = 0; i < repeat; i++) {
            Conv2FP32<T>(tempUb + i * REDUCE_FP32_REPEAT_STRIDE, copyUb + i * REDUCE_FP32_REPEAT_STRIDE, 1, 1, 1, 8, 8);
            PIPE_SYNC_EVENT(PIPE_V, PIPE_S, EVENT_ID0);
            vadd(sumUb + i * REDUCE_FP32_REPEAT_STRIDE, tempUb + i * REDUCE_FP32_REPEAT_STRIDE, sumUb + i * REDUCE_FP32_REPEAT_STRIDE, 1, 1, 1, 1, 8, 8, 8);
            PIPE_SYNC_EVENT(PIPE_V, PIPE_S, EVENT_ID0);
        }
    }
};

template<typename T>
struct ShmemReduceProcess<T, false> {
    TILEOP void ShmemReduceCopyIn(__gm__ T* x, __ubuf__ T* ubTensor, int64_t row, int64_t col, CopyParams params)
    {
        ReduceTLoad<T>(x, ubTensor, params);
    }

    TILEOP void ShmemReduceCopyOut(__gm__ T* out, __ubuf__ T* ubTensor, int64_t row, int64_t col, CopyParams params)
    {
        ReduceTStore<T>(out, ubTensor, params);
    }

    TILEOP void ShmemReduceAdd(__gm__ T* x, __ubuf__ T* ubTensor, int64_t row, int64_t col, CopyParams params)
    {
        __ubuf__ T* sumUb = ubTensor;
        __ubuf__ T* copyUb = ubTensor + row * col;
        ReduceTLoad<T>(x, copyUb, params);
        const uint64_t repeat = static_cast<uint64_t>(row) * static_cast<uint64_t>(col) * sizeof(T) / REDUCE_CHUNK_BYTES;
        const uint64_t stride = REDUCE_CHUNK_BYTES / sizeof(T);
        for (uint64_t i = 0; i < repeat; i++) {
            vadd(sumUb + i * stride, copyUb + i * stride, sumUb + i * stride, 1, 1, 1, 1, 8, 8, 8);
            PIPE_SYNC_EVENT(PIPE_V, PIPE_S, EVENT_ID0);
        }
    }
};

template<typename T, bool FP32Mode, int64_t row, int64_t col>
TILEOP void ShmemReduce(__gm__ T* out, __ubuf__ T* ubTensor, __gm__ T* in, __gm__ T* shmData,
    int64_t rowOffset, int64_t colOffset, int64_t rowPerRank, int64_t colPerRank, __gm__ int64_t *hcclContext)
{
    // 暂时只支持二维的in和out
    __gm__ CommContext *winContext = (__gm__ CommContext *)(hcclContext[0]);    // 需要 hcclGroupIndex
    uint32_t localRankId = winContext->rankId;
    uint32_t rankSize = winContext->rankNum;
    int64_t offset = rowOffset * colPerRank + colOffset;
    out += offset;
    in += offset;
    shmData += offset;

    uint16_t nBurst = (uint16_t)row;
    uint16_t lenBurst = (uint16_t)(col * sizeof(T) / 32);
    uint16_t stride = (uint16_t)((colPerRank - col) * sizeof(T) / 32);
    CopyParams copyInParams{nBurst, lenBurst, stride, 0};
    CopyParams copyOutParams{nBurst, lenBurst, 0, stride};

    ShmemReduceProcess<T, FP32Mode> proc;
    for (uint32_t rankId = 0; rankId < rankSize; rankId++) {
        __gm__ T* x = (rankId == localRankId ? in : shmData) + (uint64_t)rankId * rowPerRank * colPerRank;
        if (rankId == 0) {
            proc.ShmemReduceCopyIn(x, ubTensor, row, col, copyInParams);
        } else {
            proc.ShmemReduceAdd(x, ubTensor, row, col, copyInParams);
        }
    }
    proc.ShmemReduceCopyOut(out, ubTensor, row, col, copyOutParams);
}

} // namespace TileOp::Distributed

#endif
