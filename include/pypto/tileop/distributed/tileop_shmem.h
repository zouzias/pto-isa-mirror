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
 * \file mte.h
 * \brief
*/

#ifndef __DISTRIBUTED_SHMEM__
#define __DISTRIBUTED_SHMEM__

#include "common.h"
#include "hccl_context.h"

#include <type_traits>

namespace TileOp::Distributed {
template<typename T>
TILEOP void Conv2FP32(__ubuf__ float* dst, __ubuf__ T* src, uint8_t repeat, uint16_t dstBlockStride,
    uint16_t srcBlocakStride, uint8_t dstRepeatStride, uint8_t srcRepeatStride)
{
    if constexpr(std::is_same_v<T, half>) {
        vconv_f162f32(dst, src, repeat, dstBlockStride, srcBlocakStride, dstRepeatStride, srcRepeatStride);
    } else if constexpr(std::is_same_v<T, bfloat16_t>) {
        vconv_bf162f32(dst, src, repeat, dstBlockStride, srcBlocakStride, dstRepeatStride, srcRepeatStride);
    }
}

template<typename T>
TILEOP void DeConvFP32(__ubuf__ T* dst, __ubuf__ float* src, uint8_t repeat, uint16_t dstBlockStride,
    uint16_t srcBlocakStride, uint8_t dstRepeatStride, uint8_t srcRepeatStride)
{
    if constexpr(std::is_same_v<T, half>) {
        vconv_f322f16(dst, src, repeat, dstBlockStride, srcBlocakStride, dstRepeatStride, srcRepeatStride);
    } else if constexpr(std::is_same_v<T, bfloat16_t>) {
        vconv_f322bf16r(dst, src, repeat, dstBlockStride, srcBlocakStride, dstRepeatStride, srcRepeatStride);
    }
}

template<typename T, uint16_t sid, uint16_t nBurst, uint16_t lenBurst, uint16_t srcStride, uint16_t dstStride>
TILEOP void CopyGmToGmCore(__gm__ T* target, __ubuf__ T* buffer, __gm__ T* source)
{
    copy_gm_to_ubuf(buffer, source, sid, nBurst, lenBurst, srcStride, dstStride);
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    copy_ubuf_to_gm(target, buffer, sid, nBurst, lenBurst, dstStride, srcStride);
    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
}

template<typename TargetType, typename UBType, typename SourceType, uint32_t rowShape, uint32_t colShape,
    uint32_t srcStride, uint32_t bufferStride, uint32_t dstStride, AtomicType atomicType>
TILEOP void CopyGmToGmBlock(__gm__ TargetType* target, __ubuf__ UBType* buffer, __gm__ SourceType* source, uint32_t eventId = EVENT_ID0) {
    wait_flag(PIPE_MTE3, PIPE_S, eventId);
    set_flag(PIPE_S, PIPE_MTE2, eventId);
    wait_flag(PIPE_S, PIPE_MTE2, eventId);
    if constexpr (std::is_same_v<TargetType, SourceType>) {
        TileOp::UBCopyIn<SourceType, rowShape, colShape, bufferStride, srcStride>(buffer, source);
        set_flag(PIPE_MTE2, PIPE_MTE3, eventId);
        wait_flag(PIPE_MTE2, PIPE_MTE3, eventId);
        TileOp::UBCopyOut<TargetType, rowShape, colShape, dstStride, bufferStride>(target, buffer);
    } else {
        uint64_t castAddr = AlignUp<uint64_t>(rowShape * colShape * sizeof(UBType), 32) / sizeof(UBType);
        __ubuf__ float* castUb = (__ubuf__ float*)(buffer + castAddr);
        uint64_t repeat = AlignUp<uint64_t>(rowShape * colShape * sizeof(float), 256) / 256;
        if constexpr (atomicType == AtomicType::ADD) {
            TileOp::UBCopyIn<SourceType, rowShape, colShape, bufferStride, srcStride>(buffer, source);
            set_flag(PIPE_MTE2, PIPE_V, eventId);
            wait_flag(PIPE_MTE2, PIPE_V, eventId);
            Conv2FP32<UBType>(castUb, buffer, repeat, 1, 1, 8, 4);
            set_flag(PIPE_V, PIPE_MTE3, eventId);
            wait_flag(PIPE_V, PIPE_MTE3, eventId);
            TileOp::UBCopyOut<TargetType, rowShape, colShape, dstStride, bufferStride>(target, castUb);
        } else if constexpr (atomicType == AtomicType::SET) {
            TileOp::UBCopyIn<SourceType, rowShape, colShape, bufferStride, srcStride>(castUb, source);
            set_flag(PIPE_MTE2, PIPE_V, eventId);
            wait_flag(PIPE_MTE2, PIPE_V, eventId);
            DeConvFP32<UBType>(buffer, castUb, repeat, 1, 1, 4, 8);
            set_flag(PIPE_V, PIPE_MTE3, eventId);
            wait_flag(PIPE_V, PIPE_MTE3, eventId);
            TileOp::UBCopyOut<TargetType, rowShape, colShape, dstStride, bufferStride>(target, buffer);
        }
    }
    set_flag(PIPE_MTE3, PIPE_S, eventId);
}

template<typename TargetType, typename SourceType, uint32_t rowShape, uint32_t colShape,
    uint32_t srcStride, uint32_t bufferStride, uint32_t dstStride, AtomicType atomicType>
TILEOP void CopyUbToGmBlock(__gm__ TargetType* target, __ubuf__ SourceType* source) {
    set_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
    if constexpr (atomicType == AtomicType::ADD) {
        SetAttomicType<TargetType>();
        set_atomic_add();
    }
    TileOp::UBCopyOut<TargetType, rowShape, colShape, dstStride, bufferStride>(target, source);
    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    if constexpr (atomicType == AtomicType::ADD) {
        set_atomic_none();
    }
}

template<typename TargetType, typename SourceType, uint32_t rowShape, uint32_t colShape,
    uint32_t srcStride, uint32_t bufferStride, uint32_t dstStride, AtomicType atomicType>
TILEOP void CopyGmToUbBlock(__ubuf__ TargetType* target, __gm__ SourceType* source) {
    set_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
    TileOp::UBCopyIn<SourceType, rowShape, colShape, bufferStride, srcStride>(target, source);
    set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
}

template<typename TargetType, typename UBType, typename SourceType, uint32_t colFullBlockCount, uint32_t bufferRowShape,
    uint32_t bufferColShape, uint32_t colTailShape, uint32_t srcStride, uint32_t dstStride, AtomicType atomicType>
TILEOP void CopyGmToGmRow(__gm__ TargetType* target, __ubuf__ UBType* bufferA, __ubuf__ UBType* bufferB, __gm__ SourceType* source, uint32_t eventId = EVENT_ID0) {
    uint32_t offset = 0;
    __ubuf__ UBType* useBuffer = eventId == EVENT_ID0 ? bufferA : bufferB;
    for (uint32_t colIndex = 0; colIndex < colFullBlockCount; ++colIndex, offset += bufferColShape) {
        CopyGmToGmBlock<TargetType, UBType, SourceType, bufferRowShape, bufferColShape, srcStride, bufferColShape, dstStride, atomicType>(target + offset, useBuffer, source + offset, eventId);
        eventId = eventId == EVENT_ID0 ? EVENT_ID1 : EVENT_ID0;
        useBuffer = eventId == EVENT_ID0 ? bufferA : bufferB;
    }
    if (colTailShape > 0) {
        CopyGmToGmBlock<TargetType, UBType, SourceType, bufferRowShape, colTailShape, srcStride, bufferColShape, dstStride, atomicType>(target + offset, useBuffer, source + offset, eventId);
    }
}

template<typename TargetType, typename SourceType, uint32_t colFullBlockCount, uint32_t bufferRowShape,
    uint32_t bufferColShape, uint32_t colTailShape, uint32_t srcStride, uint32_t dstStride, AtomicType atomicType>
TILEOP void CopyUbToGmRow(__gm__ TargetType* target, __ubuf__ SourceType* source) {
    uint32_t offset = 0;
    for (uint32_t colIndex = 0; colIndex < colFullBlockCount; ++colIndex, offset += bufferColShape) {
        CopyUbToGmBlock<TargetType, SourceType, bufferRowShape, bufferColShape, srcStride, bufferColShape, dstStride, atomicType>(target + offset, source + offset);
    }
    if (colTailShape > 0) {
        CopyUbToGmBlock<TargetType, SourceType, bufferRowShape, colTailShape, srcStride, bufferColShape, dstStride, atomicType>(target + offset, source + offset);
    }
}

template<typename TargetType, typename SourceType, uint32_t colFullBlockCount, uint32_t bufferRowShape,
    uint32_t bufferColShape, uint32_t colTailShape, uint32_t srcStride, uint32_t dstStride, AtomicType atomicType>
TILEOP void CopyGmToUbRow(__ubuf__ TargetType* target, __gm__ SourceType* source) {
    uint32_t offset = 0;
    for (uint32_t colIndex = 0; colIndex < colFullBlockCount; ++colIndex, offset += bufferColShape) {
        CopyGmToUbBlock<TargetType, SourceType, bufferRowShape, bufferColShape, srcStride, bufferColShape, dstStride, atomicType>(target + offset, source + offset);
    }
    if (colTailShape > 0) {
        CopyGmToUbBlock<TargetType, SourceType, bufferRowShape, colTailShape, srcStride, bufferColShape, dstStride, atomicType>(target + offset, source + offset);
    }
}

template<typename TargetType, typename UBType, typename SourceType, uint32_t tileRowShape, uint32_t tileColShape, uint32_t bufferRowShape, uint32_t bufferColShape,
    uint32_t srcStride, uint32_t dstStride, AtomicType atomicType>
TILEOP void CopyGmToGm(__gm__ TargetType* target, __ubuf__ UBType* buffer, __gm__ SourceType* source)
{
    constexpr uint32_t rowFullBlockCount = tileRowShape / bufferRowShape;
    constexpr uint32_t colFullBlockCount = tileColShape / bufferColShape;
    constexpr uint32_t rowTailShape = tileRowShape % bufferRowShape;
    constexpr uint32_t colTailShape = tileColShape % bufferColShape;
    constexpr uint32_t srcRowStride = bufferRowShape * srcStride;
    constexpr uint32_t dstRowStride = bufferRowShape * dstStride;
    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID1);
    uint32_t eventId = EVENT_ID0;
    uint32_t colLoopNum = colFullBlockCount + colTailShape > 0 ? 1 : 0;
    uint32_t copyLen = AlignUp<uint32_t>(bufferRowShape * bufferColShape * sizeof(UBType), 32) / sizeof(UBType);
    __ubuf__ UBType* bufferA = buffer;
    __ubuf__ UBType* bufferB = buffer + copyLen;
    if constexpr (!std::is_same_v<TargetType, SourceType>) { 
        uint64_t castSize = AlignUp<uint64_t>(bufferRowShape * bufferColShape * sizeof(float), 256);
        bufferB = bufferB + castSize / sizeof(UBType);
    }
    for (uint32_t rowIndex = 0; rowIndex < rowFullBlockCount; ++rowIndex, source += srcRowStride, target += dstRowStride) {
        CopyGmToGmRow<TargetType, UBType, SourceType, colFullBlockCount, bufferRowShape, bufferColShape, colTailShape, srcStride, dstStride, atomicType>(target, bufferA, bufferB, source, eventId);
        if (colLoopNum % 2 == 1) {
            eventId = eventId == EVENT_ID0 ? EVENT_ID1 : EVENT_ID0;
        }
    }
    if (rowTailShape > 0) {
        CopyGmToGmRow<TargetType, UBType, SourceType, colFullBlockCount, rowTailShape, bufferColShape, colTailShape, srcStride, dstStride, atomicType>(target, bufferA, bufferB, source, eventId);
    }
    wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID1);
}

template<typename TargetType, typename SourceType, uint32_t tileRowShape, uint32_t tileColShape, uint32_t bufferRowShape, uint32_t bufferColShape,
    uint32_t srcStride, uint32_t dstStride, AtomicType atomicType>
TILEOP void CopyUbToGm(__gm__ TargetType* target, __ubuf__ SourceType* source)
{
    constexpr uint32_t rowFullBlockCount = tileRowShape / bufferRowShape;
    constexpr uint32_t colFullBlockCount = tileColShape / bufferColShape;
    constexpr uint32_t rowTailShape = tileRowShape % bufferRowShape;
    constexpr uint32_t colTailShape = tileColShape % bufferColShape;
    constexpr uint32_t srcRowStride = bufferRowShape * srcStride;
    constexpr uint32_t dstRowStride = bufferRowShape * dstStride;
    for (uint32_t rowIndex = 0; rowIndex < rowFullBlockCount; ++rowIndex, source += srcRowStride, target += dstRowStride) {
        CopyUbToGmRow<TargetType, SourceType, colFullBlockCount, bufferRowShape, bufferColShape, colTailShape, srcStride, dstStride, atomicType>(target, source);
    }
    if (rowTailShape > 0) {
        CopyUbToGmRow<TargetType, SourceType, colFullBlockCount, rowTailShape, bufferColShape, colTailShape, srcStride, dstStride, atomicType>(target, source);
    }
}

template<typename TargetType, typename SourceType, uint32_t tileRowShape, uint32_t tileColShape, uint32_t bufferRowShape, uint32_t bufferColShape,
    uint32_t srcStride, uint32_t dstStride, AtomicType atomicType>
TILEOP void CopyGmToUb(__ubuf__ TargetType* target, __gm__ SourceType* source)
{
    constexpr uint32_t rowFullBlockCount = tileRowShape / bufferRowShape;
    constexpr uint32_t colFullBlockCount = tileColShape / bufferColShape;
    constexpr uint32_t rowTailShape = tileRowShape % bufferRowShape;
    constexpr uint32_t colTailShape = tileColShape % bufferColShape;
    constexpr uint32_t srcRowStride = bufferRowShape * srcStride;
    constexpr uint32_t dstRowStride = bufferRowShape * dstStride;
    for (uint32_t rowIndex = 0; rowIndex < rowFullBlockCount; ++rowIndex, source += srcRowStride, target += dstRowStride) {
        CopyGmToUbRow<TargetType, SourceType, colFullBlockCount, bufferRowShape, bufferColShape, colTailShape, srcStride, dstStride, atomicType>(target, source);
    }
    if (rowTailShape > 0) {
        CopyGmToUbRow<TargetType, SourceType, colFullBlockCount, rowTailShape, bufferColShape, colTailShape, srcStride, dstStride, atomicType>(target, source);
    }
}

template<typename T, uint32_t bufferEleNum, uint32_t shmemTensorRawShape1, uint32_t shmemTensorRawShape2, uint32_t shmemTensorRawShape3>
TILEOP void ShmemClear(__ubuf__ T* buffer, __gm__ T* shmemTensorAddr)
{
    constexpr uint8_t repeat = sizeof(T) * bufferEleNum / VECTOR_INSTRUCTION_BYTE_SIZE;
    vector_dup(buffer, static_cast<T>(0), repeat, 1, 0, 8, 0);

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    constexpr uint32_t shmemTensorEleNum = shmemTensorRawShape1 * shmemTensorRawShape2 * shmemTensorRawShape3;
    constexpr uint32_t fullChunkCount = shmemTensorEleNum / bufferEleNum;
    for (int32_t i = 0; i < fullChunkCount; i++) {
        UBCopyOut<T, 1, bufferEleNum, bufferEleNum, bufferEleNum>(shmemTensorAddr + bufferEleNum * i, buffer);
    }
    constexpr uint32_t tailEleNum = shmemTensorEleNum % bufferEleNum;
    if constexpr (tailEleNum != 0) {
        UBCopyOut<T, 1, tailEleNum, tailEleNum, tailEleNum>(shmemTensorAddr + bufferEleNum * fullChunkCount, buffer);
    }
}

template<typename T, uint32_t shmemTensorRawShape1, uint32_t shmemTensorRawShape2, uint32_t shmemTensorRawShape3,
    uint32_t bufferEleNum>
TILEOP void ShmemSet(__ubuf__ T* buffer, __gm__ T* shmemTensorBaseAddr, uint32_t shmemTensorOffset0,
    uint32_t shmemTensorOffset1, uint32_t shmemTensorOffset2, uint32_t shmemTensorOffset3, __gm__ int64_t *hcclContext)
{
    __gm__ T* shmemTensorAddr = MapVirtualAddr<T>(hcclContext, shmemTensorBaseAddr, shmemTensorOffset0) + 
        shmemTensorRawShape3 * shmemTensorRawShape2 * shmemTensorOffset1 + shmemTensorRawShape3 * shmemTensorOffset2 +
        shmemTensorOffset3;
    ShmemClear<T, bufferEleNum, shmemTensorRawShape1, shmemTensorRawShape2, shmemTensorRawShape3>(buffer, shmemTensorAddr);
}

template<typename T, uint32_t worldSize, uint32_t stride, uint32_t signalMaxTileNum,
    uint32_t bufferEleNum>
TILEOP void ShmemSet(__ubuf__ T* buffer, __gm__ T* shmemTensorBaseAddr, uint32_t shmemTensorOffset0,
    uint32_t shmemTensorOffset1, uint32_t shmemTensorOffset2, uint32_t shmemTensorOffset3, uint32_t shmemTensorOffset4, 
    uint32_t shmemTensorRawShape0, uint32_t shmemTensorRawShape1, uint32_t shmemTensorRawShape2,
    uint32_t shmemTensorRawShape3, uint32_t shmemTensorRawShape4, uint32_t shmemTensorShape0, uint32_t shmemTensorShape1,
    uint32_t shmemTensorShape2, uint32_t shmemTensorShape3, uint32_t shmemTensorShape4, __gm__ int64_t *hcclContext)
{
    int32_t tileIndex = (shmemTensorOffset3 / shmemTensorShape3) *
        (shmemTensorRawShape4 / shmemTensorShape4 + (shmemTensorRawShape4 % shmemTensorShape4 == 0 ? 0 : 1)) +
        (shmemTensorOffset4 / shmemTensorShape4);
    int32_t rowTileNum = shmemTensorRawShape3 / shmemTensorShape3 + (shmemTensorRawShape3 % shmemTensorShape3 == 0 ? 0 : 1);
    int32_t colTileNum = shmemTensorRawShape4 / shmemTensorShape4 + (shmemTensorRawShape4 % shmemTensorShape4 == 0 ? 0 : 1);
    int32_t totalTileNum = rowTileNum * colTileNum;

    __gm__ T* shmemTensorAddr = MapVirtualAddr<T>(hcclContext, shmemTensorBaseAddr, shmemTensorOffset0) + 
        shmemTensorOffset0 * shmemTensorRawShape2 * totalTileNum * stride + (shmemTensorOffset2 * totalTileNum + tileIndex) * stride;

    ShmemClear<T, bufferEleNum, worldSize, signalMaxTileNum, stride>(buffer, shmemTensorAddr);
}

template<typename NonShmemType, typename ShmemType, uint32_t tileRowShape, uint32_t tileColShape, uint32_t bufferRowShape,
    uint32_t bufferColShape, uint32_t srcStride, uint32_t dstStride, AtomicType atomicType>
TILEOP void ShmemPut(__ubuf__ NonShmemType* buffer, __gm__ NonShmemType* nonShmemDataBaseAddr, __gm__ ShmemType* shmemDataBaseAddr,
    uint32_t nonShmemDataOffset0, uint32_t nonShmemDataOffset1, uint32_t nonShmemDataRawShape0,
    uint32_t nonShmemDataRawShape1, uint32_t shmemDataOffset0, uint32_t shmemDataOffset1, uint32_t shmemDataOffset2, uint32_t shmemDataOffset3,
    uint32_t shmemDataRawShape0, uint32_t shmemDataRawShape1, uint32_t shmemDataRawShape2, uint32_t shmemDataRawShape3, __gm__ int64_t *hcclContext)
{
    (void)nonShmemDataRawShape0;
    (void)shmemDataRawShape0;
    __gm__ NonShmemType* nonShmemDataAddr = nonShmemDataBaseAddr + nonShmemDataOffset0 * nonShmemDataRawShape1 + nonShmemDataOffset1;
    __gm__ ShmemType* shmemDataAddr = MapVirtualAddr<ShmemType>(hcclContext, shmemDataBaseAddr, shmemDataOffset0) +
        shmemDataOffset1 * shmemDataRawShape2 * shmemDataRawShape3 + shmemDataOffset2 * shmemDataRawShape3 + shmemDataOffset3;
    if constexpr (atomicType == AtomicType::ADD) {
        SetAttomicType<ShmemType>();
        set_atomic_add();
    }
    CopyGmToGm<ShmemType, NonShmemType, NonShmemType, tileRowShape, tileColShape, bufferRowShape, bufferColShape, srcStride, dstStride, atomicType>(shmemDataAddr, buffer, nonShmemDataAddr);
    if constexpr (atomicType == AtomicType::ADD) {
        set_atomic_none();
    }
}

template<typename InShmemType, typename OutShmemType, uint32_t tileRowShape, uint32_t tileColShape, uint32_t bufferRowShape,
    uint32_t bufferColShape, uint32_t srcStride, uint32_t dstStride, AtomicType atomicType>
TILEOP void ShmemPut(__ubuf__ InShmemType* buffer, __gm__ InShmemType* inShmemDataBaseAddr, __gm__ OutShmemType* shmemDataBaseAddr,
    uint32_t inShmemDataOffset0, uint32_t inShmemDataOffset1, uint32_t inShmemDataOffset2, uint32_t inShmemDataOffset3,
    uint32_t inShmemDataRawShape0, uint32_t inShmemDataRawShape1, uint32_t inShmemDataRawShape2, uint32_t inShmemDataRawShape3,
    uint32_t shmemDataOffset0, uint32_t shmemDataOffset1, uint32_t shmemDataOffset2, uint32_t shmemDataOffset3,
    uint32_t shmemDataRawShape0, uint32_t shmemDataRawShape1, uint32_t shmemDataRawShape2, uint32_t shmemDataRawShape3, __gm__ int64_t *hcclContext)
{
    (void)inShmemDataRawShape0;
    (void)shmemDataRawShape0;
    __gm__ InShmemType* inShmemDataAddr = MapVirtualAddr<InShmemType>(hcclContext, inShmemDataBaseAddr, inShmemDataOffset0) + inShmemDataOffset1 * inShmemDataRawShape2 * inShmemDataRawShape3 +
        inShmemDataOffset2 * inShmemDataRawShape3 + inShmemDataOffset3;
    __gm__ OutShmemType* shmemDataAddr = MapVirtualAddr<OutShmemType>(hcclContext, shmemDataBaseAddr, shmemDataOffset0) + shmemDataOffset1 * shmemDataRawShape2 * shmemDataRawShape3 +
        shmemDataOffset2 * shmemDataRawShape3 + shmemDataOffset3;
    CopyGmToGm<OutShmemType, InShmemType, InShmemType, tileRowShape, tileColShape, bufferRowShape, bufferColShape, srcStride, dstStride, atomicType>(shmemDataAddr, buffer, inShmemDataAddr);
}

template<typename UBType, typename ShmemType, uint32_t tileRowShape, uint32_t tileColShape, uint32_t bufferRowShape,
    uint32_t bufferColShape, uint32_t srcStride, uint32_t dstStride, AtomicType atomicType>
TILEOP void ShmemPutUb2Gm(__ubuf__ UBType* UBDataBaseAddr, __gm__ ShmemType* shmemDataBaseAddr, uint32_t UBDataOffset0, uint32_t UBDataOffset1, uint32_t UBDataRawShape0,
    uint32_t UBDataRawShape1, uint32_t shmemDataOffset0, uint32_t shmemDataOffset1, uint32_t shmemDataOffset2, uint32_t shmemDataOffset3,
    uint32_t shmemDataRawShape0, uint32_t shmemDataRawShape1, uint32_t shmemDataRawShape2, uint32_t shmemDataRawShape3, __gm__ int64_t *hcclContext)
{
    (void)UBDataRawShape0;
    (void)shmemDataRawShape0;
    __ubuf__ UBType* UBDataAddr = UBDataBaseAddr + UBDataOffset0 * UBDataRawShape1 + UBDataOffset1;
    __gm__ ShmemType* shmemDataAddr = MapVirtualAddr<ShmemType>(hcclContext, shmemDataBaseAddr, shmemDataOffset0) + shmemDataOffset1 * shmemDataRawShape2 * shmemDataRawShape3 +
        shmemDataOffset2 * shmemDataRawShape3 + shmemDataOffset3;
    CopyUbToGm<ShmemType, UBType, tileRowShape, tileColShape, bufferRowShape, bufferColShape, srcStride, dstStride, atomicType>(shmemDataAddr, UBDataAddr);
}

template<int64_t value, int32_t stride, AtomicType atomicType>
TILEOP void ShmemSignal(__ubuf__ int32_t* buffer, __gm__ int32_t* shmemSignalBaseAddr,
    uint32_t shmemSignalOffset0, uint32_t shmemSignalOffset1, uint32_t shmemSignalOffset2, uint32_t shmemSignalOffset3, uint32_t shmemSignalOffset4,
    uint32_t shmemSignalRawShape0, uint32_t shmemSignalRawShape1, uint32_t shmemSignalRawShape2, uint32_t shmemSignalRawShape3, uint32_t shmemSignalRawShape4,
    uint32_t shmemSignalShape0, uint32_t shmemSignalShape1, uint32_t shmemSignalShape2, uint32_t shmemSignalShape3, uint32_t shmemSignalShape4, __gm__ int64_t *hcclContext)
{
    (void)shmemSignalRawShape0;
    (void)shmemSignalRawShape1;
    (void)shmemSignalShape1;
    int32_t tileIndex = (shmemSignalOffset3 / shmemSignalShape3) *
        (shmemSignalRawShape4 / shmemSignalShape4 + (shmemSignalRawShape4 % shmemSignalShape4 == 0 ? 0 : 1)) +
        (shmemSignalOffset4 / shmemSignalShape4);
    int32_t totalTileNum = (shmemSignalRawShape3 / shmemSignalShape3 + (shmemSignalRawShape3 % shmemSignalShape3 == 0 ? 0 : 1)) *
        (shmemSignalRawShape4 / shmemSignalShape4 + (shmemSignalRawShape4 % shmemSignalShape4 == 0 ? 0 : 1));

    for (uint32_t rankId = shmemSignalOffset0; rankId < shmemSignalOffset0 + shmemSignalShape0; rankId++) {
        __gm__ int32_t* shmemSignalAddr = MapVirtualAddr<int32_t>(hcclContext, shmemSignalBaseAddr, rankId) +
            static_cast<int32_t>(shmemSignalOffset1) * shmemSignalRawShape2 * totalTileNum * stride +
            (static_cast<int32_t>(shmemSignalOffset2) * totalTileNum + tileIndex) * stride;
        constexpr uint16_t sid = 0;
        constexpr uint16_t nBurst = 1;
        constexpr uint16_t lenBurst = 1;
        constexpr uint16_t srcStride = 0;
        constexpr uint16_t dstStride = 0;
        buffer[0] = static_cast<int32_t>(value);
        if constexpr (atomicType == AtomicType::ADD) {
            set_atomic_s32();
            set_atomic_add();
        }
        set_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
        copy_ubuf_to_gm(shmemSignalAddr, buffer, sid, nBurst, lenBurst, dstStride, srcStride);
        set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
        if constexpr (atomicType == AtomicType::ADD) {
            set_atomic_none();
        }
    }
}

template<typename NonShmemType, typename ShmemType, uint32_t tileRowShape, uint32_t tileColShape, uint32_t bufferRowShape,
    uint32_t bufferColShape, uint32_t srcStride, uint32_t dstStride, AtomicType atomicType>
TILEOP void ShmemGet(__gm__ NonShmemType* nonShmemDataBaseAddr, __ubuf__ NonShmemType* buffer, __gm__ ShmemType* shmemDataBaseAddr,
    uint32_t nonShmemDataOffset0, uint32_t nonShmemDataOffset1, uint32_t nonShmemDataRawShape0,
    uint32_t nonShmemDataRawShape1, uint32_t shmemDataOffset0, uint32_t shmemDataOffset1, uint32_t shmemDataOffset2, uint32_t shmemDataOffset3,
    uint32_t shmemDataRawShape0, uint32_t shmemDataRawShape1, uint32_t shmemDataRawShape2, uint32_t shmemDataRawShape3, __gm__ int64_t *hcclContext)
{
    (void)nonShmemDataRawShape0;
    (void)shmemDataRawShape0;
    __gm__ NonShmemType* nonShmemDataAddr = nonShmemDataBaseAddr + nonShmemDataOffset0 * nonShmemDataRawShape1 + nonShmemDataOffset1;
    __gm__ ShmemType* shmemDataAddr = MapVirtualAddr<ShmemType>(hcclContext, shmemDataBaseAddr, shmemDataOffset0) +
        shmemDataOffset1 * shmemDataRawShape2 * shmemDataRawShape3 + shmemDataOffset2 * shmemDataRawShape3 + shmemDataOffset3;
    CopyGmToGm<NonShmemType, NonShmemType, ShmemType, tileRowShape, tileColShape, bufferRowShape, bufferColShape, srcStride, dstStride, atomicType>(nonShmemDataAddr, buffer, shmemDataAddr);
}

template<typename UBType, typename ShmemType, uint32_t tileRowShape, uint32_t tileColShape, uint32_t bufferRowShape,
    uint32_t bufferColShape, uint32_t srcStride, uint32_t dstStride, AtomicType atomicType>
TILEOP void ShmemGetGm2Ub(__ubuf__ UBType* UBDataBaseAddr, __gm__ ShmemType* shmemDataBaseAddr,
    uint32_t UBDataOffset0, uint32_t UBDataOffset1, uint32_t UBDataRawShape0, uint32_t UBDataRawShape1,
    uint32_t shmemDataOffset0, uint32_t shmemDataOffset1, uint32_t shmemDataOffset2, uint32_t shmemDataOffset3,
    uint32_t shmemDataRawShape0, uint32_t shmemDataRawShape1, uint32_t shmemDataRawShape2, uint32_t shmemDataRawShape3, __gm__ int64_t *hcclContext)
{
    (void)UBDataRawShape0;
    (void)shmemDataRawShape0;
    __ubuf__ UBType* UBDataAddr = UBDataBaseAddr + UBDataOffset0 * UBDataRawShape1 + UBDataOffset1;
    __gm__ ShmemType* shmemDataAddr = MapVirtualAddr<ShmemType>(hcclContext, shmemDataBaseAddr, shmemDataOffset0) +
        shmemDataOffset1 * shmemDataRawShape2 * shmemDataRawShape3 + shmemDataOffset2 * shmemDataRawShape3 + shmemDataOffset3;
    CopyGmToUb<UBType, ShmemType, tileRowShape, tileColShape, bufferRowShape, bufferColShape, srcStride, dstStride, atomicType>(UBDataAddr, shmemDataAddr);
}

template<typename T, bool FP32Mode>
struct ShmemReduceProcess {};

// 类模板部分特化
template<typename T>
struct ShmemReduceProcess<T, true> {
    TILEOP void ShmemReduceCopyIn(__gm__ T* x, __ubuf__ T* ubTensor, int64_t row, int64_t col, CopyParams params)
    {
        __ubuf__ T* copyUb = ubTensor;
        __ubuf__ float* sumUb = (__ubuf__ float*)(ubTensor + row * col);

        copy_gm_to_ubuf(copyUb, x, 0, params.nBurst, params.lenBurst, params.srcStride, params.dstStride);
        set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);

        uint64_t repeat = row * col * sizeof(float) / 256;
        uint64_t offset = 256 / sizeof(float);
        __ubuf__ T* src = copyUb;
        __ubuf__ float* dst = sumUb;
        for (uint64_t i = 0; i < repeat; i++) {
            Conv2FP32<T>(dst, src, 1, 1, 1, 8, 8);
            set_flag(PIPE_V, PIPE_S, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
            src += offset;
            dst += offset;
        }
    }

    TILEOP void ShmemReduceCopyOut(__gm__ T* out, __ubuf__ T* ubTensor, int64_t row, int64_t col, CopyParams params)
    {
        __ubuf__ T* copyUb = ubTensor;
        __ubuf__ float* sumUb = (__ubuf__ float*)(ubTensor + row * col);

        uint64_t repeat = row * col * sizeof(float) / 256;
        uint64_t offset = 256 / sizeof(float);
        __ubuf__ float* src = sumUb;
        __ubuf__ T* dst = copyUb;
        for (uint64_t i = 0; i < repeat; i++) {
            DeConvFP32<T>(dst, src, 1, 1, 1, 8, 8);
            set_flag(PIPE_V, PIPE_S, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
            src += offset;
            dst += offset;
        }

        copy_ubuf_to_gm(out, copyUb, 0, params.nBurst, params.lenBurst, params.srcStride, params.dstStride);
        set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    }

    TILEOP void ShmemReduceAdd(__gm__ T* x, __ubuf__ T* ubTensor, int64_t row, int64_t col, CopyParams params)
    {
        __ubuf__ T* copyUb = ubTensor;
        __ubuf__ float* sumUb = (__ubuf__ float*)(ubTensor + row * col);
        __ubuf__ float* tempUb = sumUb + row * col;

        copy_gm_to_ubuf(copyUb, x, 0, params.nBurst, params.lenBurst, params.srcStride, params.dstStride);
        set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);

        uint64_t repeat = row * col * sizeof(float) / 256;
        uint64_t offset = 256 / sizeof(float);
        __ubuf__ float* src = copyUb;
        __ubuf__ T* dst = sumUb;
        for (uint64_t i = 0; i < repeat; i++) {
            Conv2FP32<T>(tempUb, src, 1, 1, 1, 8, 8);
            pipe_barrier(PIPE_V);
            vadd(dst, tempUb, dst, 1, 1, 1, 1, 8, 8, 8);
            set_flag(PIPE_V, PIPE_S, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
            src += offset;
            dst += offset;
        }
    }
};

template<typename T>
struct ShmemReduceProcess<T, false> {
    TILEOP void ShmemReduceCopyIn(__gm__ T* x, __ubuf__ T* ubTensor, int64_t row, int64_t col, CopyParams params)
    {
        copy_gm_to_ubuf(ubTensor, x, 0, params.nBurst, params.lenBurst, params.srcStride, params.dstStride);
        set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
    }

    TILEOP void ShmemReduceCopyOut(__gm__ T* out, __ubuf__ T* ubTensor, int64_t row, int64_t col, CopyParams params)
    {
        copy_ubuf_to_gm(out, ubTensor, 0, params.nBurst, params.lenBurst, params.srcStride, params.dstStride);
        set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    }

    TILEOP void ShmemReduceAdd(__gm__ T* x, __ubuf__ T* ubTensor, int64_t row, int64_t col, CopyParams params)
    {
        __ubuf__ T* sumUb = ubTensor;
        __ubuf__ T* copyUb = ubTensor + row * col;

        copy_gm_to_ubuf(copyUb, x, 0, params.nBurst, params.lenBurst, params.srcStride, params.dstStride);
        set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);

        uint64_t repeat = row * col * sizeof(float) / 256;
        uint64_t offset = 256 / sizeof(float);
        __ubuf__ T* src = copyUb;
        __ubuf__ T* dst = sumUb;
        for (uint64_t i = 0; i < repeat; i++) {
            vadd(dst, src, dst, 1, 1, 1, 1, 8, 8, 8);
            set_flag(PIPE_V, PIPE_S, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
            src += offset;
            dst += offset;
        }
    }
};

template<typename T, bool FP32Mode, int64_t row, int64_t col>
TILEOP void ShmemReduce(__gm__ T* out, __ubuf__ T* ubTensor, __gm__ T* in, __gm__ T* shmData,
    int64_t rowOffset, int64_t colOffset, int64_t rowPerRank, int64_t colPerRank, __gm__ int64_t *hcclContext)
{
    // 暂时只支持二维的in和out
    __gm__ HcclCombinOpParam *winContext = (__gm__ HcclCombinOpParam *)(hcclContext[0]);    // 需要 hcclGroupIndex
    uint32_t localRankId = winContext->rankId;
    uint32_t rankSize = winContext->rankNum;

    // 先进行gm地址的偏移计算
    int64_t offset = rowOffset * colPerRank + colOffset;
    out += offset;
    in += offset;
    shmData += offset;

    uint16_t nBurst = (uint16_t)row;                                // 暂不考虑超过uint16大小的场景
    uint16_t lenBurst = (uint16_t)(col * sizeof(T) / 32);
    uint16_t stride = (uint16_t)((colPerRank - col) * sizeof(T) / 32);
    CopyParams copyInParams{nBurst, lenBurst, stride, 0};           // ub是连续的，gm是stride间隔的
    CopyParams copyOutParams{nBurst, lenBurst, 0, stride};

    __gm__ T* x;
    for (uint32_t rankId = 0; rankId < rankSize; rankId++) {
        // 根据rankId进行perRank偏移计算
        if (rankId == localRankId) {
            x = in + (uint64_t)rankId * rowPerRank * colPerRank;
        } else {
            x = shmData + (uint64_t)rankId * rowPerRank * colPerRank;
        }
        if (rankId == 0) {
            ShmemReduceProcess<T, FP32Mode>::ShmemReduceCopyIn(x, ubTensor, row, col, copyInParams);
        } else {
            ShmemReduceProcess<T, FP32Mode>::ShmemReduceAdd(x, ubTensor, row, col, copyInParams);
        }
    }
    ShmemReduceProcess<T, FP32Mode>::ShmemReduceCopyOut(out, ubTensor, row, col, copyOutParams);
}

} // namespace TileOp::Distributed
#endif