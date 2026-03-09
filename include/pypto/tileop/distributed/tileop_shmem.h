/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file tileop_shmem.h
 * \brief Shmem (shared memory) tileops: clear/set, GM/UB copy, Put/Get/Signal, Reduce.
 */

#ifndef __DISTRIBUTED_SHMEM__
#define __DISTRIBUTED_SHMEM__

#include "common.h"
#include "tileop_shmem_reduce.h"
#include <type_traits>

#ifdef SUPPORT_TILE_TENSOR
#include "pto/comm/pto_comm_inst.hpp"
#endif

namespace TileOp::Distributed {

// ---------------------------------------------------------------------------
// Shmem tensor/tile type aliases
// ---------------------------------------------------------------------------
using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;

TILEOP inline ShapeDyn MakeShape(uint32_t row, uint32_t col) { return ShapeDyn(1, 1, 1, row, col); }
TILEOP inline StrideDyn MakeStride(uint32_t row, uint32_t stride) { return StrideDyn(row, row, row, stride, 1); }
TILEOP inline uint32_t ToggleEvent(uint32_t eventId) { return eventId == EVENT_ID0 ? EVENT_ID1 : EVENT_ID0; }

template<typename T>
TILEOP constexpr T CeilDiv(T x, T y) { return (x + y - 1) / y; }

template<AtomicType atomicType, typename TileType, typename GlobalType>
TILEOP void AtomicStore(GlobalType& global, TileType& tile)
{
    if constexpr (atomicType == AtomicType::ADD) {
        pto::TSTORE<TileType, GlobalType, pto::AtomicType::AtomicAdd>(global, tile);
    } else {
        pto::TSTORE<TileType, GlobalType, pto::AtomicType::AtomicNone>(global, tile);
    }
}

template<typename T, uint32_t RowShape, uint32_t ColShape>
using ShmemGlobalTensor = pto::GlobalTensor<T, ShapeDyn, StrideDyn, pto::Layout::ND>;

template<typename T, uint32_t RowShape, uint32_t ColShape>
using ShmemUbTile = pto::Tile<pto::TileType::Vec, T,
    RowShape,
    AlignUp<uint32_t>(ColShape * sizeof(T), COPY_BLOCK_BYTE_SIZE) / sizeof(T),
    pto::BLayout::RowMajor,
    pto::DYNAMIC,
    pto::DYNAMIC>;

// ---------------------------------------------------------------------------
// Shmem clear / set
// ---------------------------------------------------------------------------
// Zero a shmem region; use PTO TEXPANDS instead of vector_dup for forward compatibility.
// V→MTE3 sync ensures fill completes before TSTORE.
template<typename T, uint32_t bufferEleNum, uint32_t shmemTensorRawShape1, uint32_t shmemTensorRawShape2, uint32_t shmemTensorRawShape3>
TILEOP void ShmemClear(__ubuf__ T* buffer, __gm__ T* shmemTensorAddr)
{
    ShmemUbTile<T, 1, bufferEleNum> ubTile(1, bufferEleNum);
    pto::TASSIGN(ubTile, reinterpret_cast<uintptr_t>(buffer));
    pto::TEXPANDS(ubTile, static_cast<T>(0));
    PIPE_SYNC_EVENT(PIPE_V, PIPE_MTE3, EVENT_ID0);

    constexpr uint32_t shmemTensorEleNum = shmemTensorRawShape1 * shmemTensorRawShape2 * shmemTensorRawShape3;
    constexpr uint32_t fullChunkCount = shmemTensorEleNum / bufferEleNum;

    for (int32_t i = 0; i < fullChunkCount; i++) {
        __gm__ T* dstAddr = shmemTensorAddr + bufferEleNum * i;
        ShapeDyn shape = MakeShape(1, bufferEleNum);
        StrideDyn strideDyn = MakeStride(1, bufferEleNum);
        ShmemGlobalTensor<T, 1, bufferEleNum> gmTensor(dstAddr, shape, strideDyn);
        pto::TSTORE<decltype(ubTile), decltype(gmTensor), pto::AtomicType::AtomicNone>(gmTensor, ubTile);
    }

    constexpr uint32_t tailEleNum = shmemTensorEleNum % bufferEleNum;
    if constexpr (tailEleNum != 0) {
        __gm__ T* tailDstAddr = shmemTensorAddr + bufferEleNum * fullChunkCount;
        ShapeDyn tailShape = MakeShape(1, tailEleNum);
        StrideDyn tailStrideDyn = MakeStride(1, tailEleNum);
        ShmemGlobalTensor<T, 1, tailEleNum> tailGmTensor(tailDstAddr, tailShape, tailStrideDyn);
        ShmemUbTile<T, 1, tailEleNum> tailUbTile(1, tailEleNum);
        pto::TASSIGN(tailUbTile, reinterpret_cast<uintptr_t>(buffer));
        pto::TSTORE<decltype(tailUbTile), decltype(tailGmTensor), pto::AtomicType::AtomicNone>(tailGmTensor, tailUbTile);
    }
}

template<typename T, uint32_t shmemTensorRawShape1, uint32_t shmemTensorRawShape2, uint32_t shmemTensorRawShape3,
    uint32_t bufferEleNum>
TILEOP void ShmemSet(__ubuf__ T* buffer, __gm__ T* shmemTensorBaseAddr, uint32_t shmemTensorOffset0,
    uint32_t shmemTensorOffset1, uint32_t shmemTensorOffset2, uint32_t shmemTensorOffset3, __gm__ int64_t *hcclContext)
{
    __gm__ T* shmemTensorAddr = MapAndOffsetShmem(
        hcclContext, shmemTensorBaseAddr, shmemTensorOffset0, shmemTensorOffset1, shmemTensorOffset2, shmemTensorOffset3,
        shmemTensorRawShape2, shmemTensorRawShape3);
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
    uint32_t colTileNum = CeilDiv(shmemTensorRawShape4, shmemTensorShape4);
    uint32_t rowTileNum = CeilDiv(shmemTensorRawShape3, shmemTensorShape3);
    uint32_t tileIndex = (shmemTensorOffset3 / shmemTensorShape3) * colTileNum + (shmemTensorOffset4 / shmemTensorShape4);
    int32_t totalTileNum = rowTileNum * colTileNum;

    __gm__ T* shmemTensorAddr = MapVirtualAddr<T>(hcclContext, shmemTensorBaseAddr, shmemTensorOffset0) +
        shmemTensorOffset1 * shmemTensorRawShape2 * totalTileNum * stride + (shmemTensorOffset2 * totalTileNum + tileIndex) * stride;

    ShmemClear<T, bufferEleNum, worldSize, signalMaxTileNum, stride>(buffer, shmemTensorAddr);
}

// ---------------------------------------------------------------------------
// Copy: GM↔GM (via UB, with optional type conversion and ping-pong)
// ---------------------------------------------------------------------------
template<typename TargetType, typename UBType, typename SourceType, uint32_t rowShape, uint32_t colShape,
    uint32_t srcStride, uint32_t dstStride, AtomicType atomicType>
TILEOP void CopyGmToGmBlockSameType(__gm__ TargetType* target, __ubuf__ UBType* buffer, __gm__ SourceType* source,
    uint32_t eventId)
{
    static_assert(std::is_same_v<TargetType, SourceType>, "SameType path requires identical source/target types.");
    static_assert(std::is_same_v<UBType, SourceType>, "SameType path requires UB type to match GM element type.");
    ShapeDyn shape = MakeShape(rowShape, colShape);
    StrideDyn srcStrideDyn = MakeStride(rowShape, srcStride);
    StrideDyn dstStrideDyn = MakeStride(rowShape, dstStride);
    ShmemGlobalTensor<SourceType, rowShape, colShape> srcGlobal(source, shape, srcStrideDyn);
    ShmemGlobalTensor<TargetType, rowShape, colShape> dstGlobal(target, shape, dstStrideDyn);
    ShmemUbTile<UBType, rowShape, colShape> ubTile(rowShape, colShape);
    pto::TASSIGN(ubTile, reinterpret_cast<uintptr_t>(buffer));
    pto::TLOAD(ubTile, srcGlobal);
    PIPE_SYNC_EVENT(PIPE_MTE2, PIPE_MTE3, eventId);
    AtomicStore<atomicType>(dstGlobal, ubTile);
}

template<typename TargetType, typename UBType, typename SourceType, uint32_t rowShape, uint32_t colShape,
    uint32_t srcStride, uint32_t dstStride, AtomicType atomicType>
TILEOP void CopyGmToGmBlockWithCast(__gm__ TargetType* target, __ubuf__ UBType* buffer, __gm__ SourceType* source,
    uint32_t eventId)
{
    ShapeDyn shape = MakeShape(rowShape, colShape);
    StrideDyn srcStrideDyn = MakeStride(rowShape, srcStride);
    StrideDyn dstStrideDyn = MakeStride(rowShape, dstStride);
    ShmemGlobalTensor<SourceType, rowShape, colShape> srcGlobal(source, shape, srcStrideDyn);
    ShmemGlobalTensor<TargetType, rowShape, colShape> dstGlobal(target, shape, dstStrideDyn);
    constexpr uint64_t copyLen = rowShape * AlignUp<uint64_t>(colShape * sizeof(UBType), 32) / sizeof(UBType);
    __ubuf__ float* castUb = (__ubuf__ float*)(buffer + copyLen);
    constexpr bool kAtomicAdd = (atomicType == AtomicType::ADD);
    using SrcElemType = std::conditional_t<kAtomicAdd, UBType, float>;
    using DstElemType = std::conditional_t<kAtomicAdd, float, UBType>;
    ShmemUbTile<SrcElemType, rowShape, colShape> srcTile(rowShape, colShape);
    ShmemUbTile<DstElemType, rowShape, colShape> dstTile(rowShape, colShape);
    if constexpr (kAtomicAdd) {
        pto::TASSIGN(srcTile, reinterpret_cast<uintptr_t>(buffer));
        pto::TASSIGN(dstTile, reinterpret_cast<uintptr_t>(castUb));
    } else {
        pto::TASSIGN(srcTile, reinterpret_cast<uintptr_t>(castUb));
        pto::TASSIGN(dstTile, reinterpret_cast<uintptr_t>(buffer));
    }
    pto::TLOAD(srcTile, srcGlobal);
    PIPE_SYNC_EVENT(PIPE_MTE2, PIPE_V, eventId);
    pto::TCVT(dstTile, srcTile, pto::RoundMode::CAST_NONE);
    PIPE_SYNC_EVENT(PIPE_V, PIPE_MTE3, eventId);
    AtomicStore<atomicType>(dstGlobal, dstTile);
}

// Single block GM→UB→GM. With conversion: buffer[0..copyLen-1]=UBType, buffer[copyLen..]=float.
template<typename TargetType, typename UBType, typename SourceType, uint32_t rowShape, uint32_t colShape,
    uint32_t srcStride, uint32_t dstStride, AtomicType atomicType>
TILEOP void CopyGmToGmBlock(__gm__ TargetType* target, __ubuf__ UBType* buffer, __gm__ SourceType* source,
    uint32_t eventId = EVENT_ID0) {
    wait_flag(PIPE_MTE3, PIPE_S, eventId);
    PIPE_SYNC_EVENT(PIPE_S, PIPE_MTE2, eventId);
    if constexpr (std::is_same_v<TargetType, SourceType>) {
        CopyGmToGmBlockSameType<TargetType, UBType, SourceType, rowShape, colShape, srcStride, dstStride, atomicType>(
            target, buffer, source, eventId);
    } else {
        CopyGmToGmBlockWithCast<TargetType, UBType, SourceType, rowShape, colShape, srcStride, dstStride, atomicType>(
            target, buffer, source, eventId);
    }
    set_flag(PIPE_MTE3, PIPE_S, eventId);
}

template<typename TargetType, typename UBType, typename SourceType, uint32_t rowShape, uint32_t colFullBlockCount,
    uint32_t colTailShape, uint32_t bufferColShape, uint32_t srcStride, uint32_t dstStride, AtomicType atomicType>
TILEOP void CopyGmToGmRow(__gm__ TargetType* dstPtr, __gm__ SourceType* srcPtr,
    __ubuf__ UBType* bufferA, __ubuf__ UBType* bufferB, uint32_t& eventId)
{
    uint32_t colOffset = 0;
    for (uint32_t colIndex = 0; colIndex < colFullBlockCount; ++colIndex, colOffset += bufferColShape) {
        __ubuf__ UBType* useBuffer = eventId == EVENT_ID0 ? bufferA : bufferB;
        CopyGmToGmBlock<TargetType, UBType, SourceType, rowShape, bufferColShape, srcStride, dstStride, atomicType>(
            dstPtr + colOffset, useBuffer, srcPtr + colOffset, eventId);
        eventId = eventId == EVENT_ID0 ? EVENT_ID1 : EVENT_ID0;
    }
    if constexpr (colTailShape > 0) {
        __ubuf__ UBType* useBuffer = eventId == EVENT_ID0 ? bufferA : bufferB;
        CopyGmToGmBlock<TargetType, UBType, SourceType, rowShape, colTailShape, srcStride, dstStride, atomicType>(
            dstPtr + colOffset, useBuffer, srcPtr + colOffset, eventId);
        eventId = eventId == EVENT_ID0 ? EVENT_ID1 : EVENT_ID0;
    }
}

template<bool useTPut, typename DataType, uint32_t tileRowShape, uint32_t tileColShape, uint32_t bufferRowShape,
    uint32_t srcStride, uint32_t dstStride, AtomicType atomicType = AtomicType::SET>
TILEOP void CopyGmToGmByTRowSliced(__gm__ DataType* target, __ubuf__ DataType* buffer, __gm__ DataType* source)
{
    static_assert(bufferRowShape > 0, "bufferRowShape must be greater than 0.");
    constexpr uint32_t kMaxTileRows = 4095;
    constexpr uint32_t kEffectiveRows = bufferRowShape < kMaxTileRows ? bufferRowShape : kMaxTileRows;
    constexpr uint32_t kChunkRows = tileRowShape < kEffectiveRows ? tileRowShape : kEffectiveRows;

    constexpr uint32_t kAlignedCols =
        AlignUp<uint32_t>(tileColShape * sizeof(DataType), COPY_BLOCK_BYTE_SIZE) / sizeof(DataType);
    constexpr uint32_t kHalfBufferEleCount = bufferRowShape * kAlignedCols;

    ShapeDyn shape = MakeShape(tileRowShape, tileColShape);
    StrideDyn srcStrideDyn = MakeStride(tileRowShape, srcStride);
    StrideDyn dstStrideDyn = MakeStride(tileRowShape, dstStride);
    ShmemGlobalTensor<DataType, kChunkRows, tileColShape> srcGlobal(source, shape, srcStrideDyn);
    ShmemGlobalTensor<DataType, kChunkRows, tileColShape> dstGlobal(target, shape, dstStrideDyn);

    ShmemUbTile<DataType, kChunkRows, tileColShape> pingTile(kChunkRows, tileColShape);
    ShmemUbTile<DataType, kChunkRows, tileColShape> pongTile(kChunkRows, tileColShape);
    pto::TASSIGN(pingTile, reinterpret_cast<uintptr_t>(buffer));
    pto::TASSIGN(pongTile, reinterpret_cast<uintptr_t>(buffer + kHalfBufferEleCount));

    if constexpr (useTPut) {
        if constexpr (atomicType == AtomicType::ADD) {
            pto::comm::TPUT<pto::AtomicType::AtomicAdd>(dstGlobal, srcGlobal, pingTile, pongTile);
        } else {
            pto::comm::TPUT<pto::AtomicType::AtomicNone>(dstGlobal, srcGlobal, pingTile, pongTile);
        }
    } else {
        pto::comm::TGET(dstGlobal, srcGlobal, pingTile, pongTile);
    }
    PIPE_SYNC_EVENT(PIPE_MTE3, PIPE_S, EVENT_ID0);
}

// Full tile copy with row/column chunking. Ping-pong layout: same type bufferA|bufferB; with conversion bufferA|castUbA|bufferB|castUbB.
template<typename TargetType, typename UBType, typename SourceType, uint32_t tileRowShape, uint32_t tileColShape, 
    uint32_t bufferRowShape, uint32_t bufferColShape, uint32_t srcStride, uint32_t dstStride, AtomicType atomicType>
TILEOP void CopyGmToGm(__gm__ TargetType* target, __ubuf__ UBType* buffer, __gm__ SourceType* source)
{
    if constexpr (std::is_same_v<TargetType, SourceType> && std::is_same_v<UBType, SourceType> &&
        (bufferColShape >= tileColShape)) {
        CopyGmToGmByTRowSliced<true, TargetType, tileRowShape, tileColShape, bufferRowShape, srcStride, dstStride,
            atomicType>(
            target, buffer, source);
        return;
    }

    constexpr uint32_t rowFullBlockCount = tileRowShape / bufferRowShape;
    constexpr uint32_t colFullBlockCount = tileColShape / bufferColShape;
    constexpr uint32_t rowTailShape = tileRowShape % bufferRowShape;
    constexpr uint32_t colTailShape = tileColShape % bufferColShape;
    constexpr uint32_t srcRowStride = bufferRowShape * srcStride;
    constexpr uint32_t dstRowStride = bufferRowShape * dstStride;
    
    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID1);
    
    uint32_t eventId = EVENT_ID0;
    
    constexpr uint32_t copyLen = bufferRowShape * AlignUp<uint32_t>(bufferColShape * sizeof(UBType), 32) / sizeof(UBType);
    __ubuf__ UBType* bufferA = buffer;
    __ubuf__ UBType* bufferB = buffer + copyLen;
    
    if constexpr (!std::is_same_v<TargetType, SourceType>) {
        constexpr uint64_t castSize = AlignUp<uint64_t>(copyLen * sizeof(float), 256);
        bufferB = buffer + copyLen + castSize / sizeof(UBType);
    }
    
    __gm__ SourceType* srcPtr = source;
    __gm__ TargetType* dstPtr = target;
    for (uint32_t rowIndex = 0; rowIndex < rowFullBlockCount; ++rowIndex, srcPtr += srcRowStride, dstPtr += dstRowStride) {
        CopyGmToGmRow<TargetType, UBType, SourceType, bufferRowShape, colFullBlockCount, colTailShape,
            bufferColShape, srcStride, dstStride, atomicType>(dstPtr, srcPtr, bufferA, bufferB, eventId);
    }
    
    if constexpr (rowTailShape > 0) {
        CopyGmToGmRow<TargetType, UBType, SourceType, rowTailShape, colFullBlockCount, colTailShape,
            bufferColShape, srcStride, dstStride, atomicType>(dstPtr, srcPtr, bufferA, bufferB, eventId);
    }
    
    wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID1);
}

// ---------------------------------------------------------------------------
// Copy: UB → GM
// ---------------------------------------------------------------------------
template<typename TargetType, typename SourceType, uint32_t rowShape, uint32_t colShape,
    uint32_t dstStride, AtomicType atomicType>
TILEOP void CopyUbToGmBlock(__gm__ TargetType* target, __ubuf__ SourceType* source) {
    PIPE_SYNC_EVENT(PIPE_S, PIPE_MTE3, EVENT_ID0);
    
    ShapeDyn shape = MakeShape(rowShape, colShape);
    StrideDyn dstStrideDyn = MakeStride(rowShape, dstStride);
    
    ShmemGlobalTensor<TargetType, rowShape, colShape> dstGlobal(target, shape, dstStrideDyn);
    ShmemUbTile<SourceType, rowShape, colShape> ubTile(rowShape, colShape);
    pto::TASSIGN(ubTile, reinterpret_cast<uintptr_t>(source));
    
    AtomicStore<atomicType>(dstGlobal, ubTile);
    
    PIPE_SYNC_EVENT(PIPE_MTE3, PIPE_S, EVENT_ID0);
}

template<typename TargetType, typename SourceType, uint32_t rowShape, uint32_t colFullBlockCount,
    uint32_t colTailShape, uint32_t bufferColShape, uint32_t dstStride, AtomicType atomicType>
TILEOP void CopyUbToGmRow(__gm__ TargetType* dstPtr, __ubuf__ SourceType* srcPtr)
{
    uint32_t colOffset = 0;
    for (uint32_t colIndex = 0; colIndex < colFullBlockCount; ++colIndex, colOffset += bufferColShape) {
        CopyUbToGmBlock<TargetType, SourceType, rowShape, bufferColShape, dstStride, atomicType>(
            dstPtr + colOffset, srcPtr + colOffset);
    }
    if constexpr (colTailShape > 0) {
        CopyUbToGmBlock<TargetType, SourceType, rowShape, colTailShape, dstStride, atomicType>(
            dstPtr + colOffset, srcPtr + colOffset);
    }
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
    
    __ubuf__ SourceType* srcPtr = source;
    __gm__ TargetType* dstPtr = target;
    for (uint32_t rowIndex = 0; rowIndex < rowFullBlockCount; ++rowIndex, srcPtr += srcRowStride, dstPtr += dstRowStride) {
        CopyUbToGmRow<TargetType, SourceType, bufferRowShape, colFullBlockCount, colTailShape, bufferColShape, dstStride, atomicType>(
            dstPtr, srcPtr);
    }
    if constexpr (rowTailShape > 0) {
        CopyUbToGmRow<TargetType, SourceType, rowTailShape, colFullBlockCount, colTailShape, bufferColShape, dstStride, atomicType>(
            dstPtr, srcPtr);
    }
}

// ---------------------------------------------------------------------------
// Copy: GM → UB (single block, optional type conversion)
// ---------------------------------------------------------------------------
template<typename TargetType, typename SourceType, uint32_t rowShape, uint32_t colShape,
    uint32_t srcStride, uint32_t dstStride>
TILEOP void CopyGmToUbBlock(__ubuf__ TargetType* target, __ubuf__ TargetType* buffer, __gm__ SourceType* source) {
    ShapeDyn shape = MakeShape(rowShape, colShape);
    StrideDyn srcStrideDyn = MakeStride(rowShape, srcStride);
    ShmemGlobalTensor<SourceType, rowShape, colShape> srcGlobal(source, shape, srcStrideDyn);
    if constexpr (std::is_same_v<TargetType, SourceType>) {
        PIPE_SYNC_EVENT(PIPE_S, PIPE_MTE2, EVENT_ID0);
        ShmemUbTile<TargetType, rowShape, colShape> ubTile(rowShape, colShape);
        pto::TASSIGN(ubTile, reinterpret_cast<uintptr_t>(target));
        pto::TLOAD(ubTile, srcGlobal);
        PIPE_SYNC_EVENT(PIPE_MTE2, PIPE_S, EVENT_ID0);
    } else {
        __ubuf__ float* castUb = (__ubuf__ float*)buffer;
        ShmemUbTile<float, rowShape, colShape> srcTile(rowShape, colShape);
        ShmemUbTile<TargetType, rowShape, colShape> dstTile(rowShape, colShape);
        pto::TASSIGN(srcTile, reinterpret_cast<uintptr_t>(castUb));
        pto::TASSIGN(dstTile, reinterpret_cast<uintptr_t>(target));
        pto::TLOAD(srcTile, srcGlobal);
        PIPE_SYNC_EVENT(PIPE_MTE2, PIPE_V, EVENT_ID0);
        pto::TCVT(dstTile, srcTile, pto::RoundMode::CAST_NONE);
        PIPE_SYNC_EVENT(PIPE_V, PIPE_S, EVENT_ID0);
    }
}

// ---------------------------------------------------------------------------
// Shmem Put / Get / Signal
// ---------------------------------------------------------------------------
// Put: local GM (or inShmem GM) → remote shmem GM.
template<typename NonShmemType, typename ShmemType, uint32_t tileRowShape, uint32_t tileColShape, uint32_t bufferRowShape,
    uint32_t bufferColShape, uint32_t srcStride, uint32_t dstStride, AtomicType atomicType>
TILEOP void ShmemPut(__ubuf__ NonShmemType* buffer, __gm__ NonShmemType* nonShmemDataBaseAddr, __gm__ ShmemType* shmemDataBaseAddr,
    uint32_t nonShmemDataOffset0, uint32_t nonShmemDataOffset1, uint32_t nonShmemDataRawShape0,
    uint32_t nonShmemDataRawShape1, uint32_t shmemDataOffset0, uint32_t shmemDataOffset1, uint32_t shmemDataOffset2, uint32_t shmemDataOffset3,
    uint32_t shmemDataRawShape0, uint32_t shmemDataRawShape1, uint32_t shmemDataRawShape2, uint32_t shmemDataRawShape3, uint32_t shmemGetTensorDataOffset, __gm__ int64_t *hcclContext)
{
    (void)nonShmemDataRawShape0;
    (void)shmemDataRawShape0;
    if (shmemGetTensorDataOffset != -1) {
        shmemDataOffset2 = shmemGetTensorDataOffset;
    }
    __gm__ NonShmemType* srcAddr = nonShmemDataBaseAddr + TileOp::CalcLinearOffset(nonShmemDataRawShape1,
        nonShmemDataOffset0, nonShmemDataOffset1);
    __gm__ ShmemType* dstAddr = MapAndOffsetShmem(
        hcclContext, shmemDataBaseAddr, shmemDataOffset0, shmemDataOffset1, shmemDataOffset2, shmemDataOffset3,
        shmemDataRawShape2, shmemDataRawShape3);
    
    if constexpr (atomicType == AtomicType::ADD) {
        SetAttomicType<ShmemType>();
        set_atomic_add();
    }
    CopyGmToGm<ShmemType, NonShmemType, NonShmemType, tileRowShape, tileColShape, bufferRowShape, bufferColShape, srcStride, dstStride, atomicType>(
        dstAddr, buffer, srcAddr);
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
    
    __gm__ InShmemType* inShmemDataAddr = MapAndOffsetShmem(
        hcclContext, inShmemDataBaseAddr, inShmemDataOffset0, inShmemDataOffset1, inShmemDataOffset2, inShmemDataOffset3,
        inShmemDataRawShape2, inShmemDataRawShape3);
    __gm__ OutShmemType* shmemDataAddr = MapAndOffsetShmem(
        hcclContext, shmemDataBaseAddr, shmemDataOffset0, shmemDataOffset1, shmemDataOffset2, shmemDataOffset3,
        shmemDataRawShape2, shmemDataRawShape3);
    
    CopyGmToGm<OutShmemType, InShmemType, InShmemType, tileRowShape, tileColShape, bufferRowShape, bufferColShape, srcStride, dstStride, atomicType>(
        shmemDataAddr, buffer, inShmemDataAddr);
}

// Put UB directly to remote shmem GM.
template<typename UBType, typename ShmemType, uint32_t tileRowShape, uint32_t tileColShape, uint32_t bufferRowShape,
    uint32_t bufferColShape, uint32_t srcStride, uint32_t dstStride, AtomicType atomicType>
TILEOP void ShmemPutUb2Gm(__ubuf__ UBType* UBDataBaseAddr, __gm__ ShmemType* shmemDataBaseAddr, uint32_t UBDataOffset0, uint32_t UBDataOffset1, uint32_t UBDataRawShape0,
    uint32_t UBDataRawShape1, uint32_t shmemDataOffset0, uint32_t shmemDataOffset1, uint32_t shmemDataOffset2, uint32_t shmemDataOffset3,
    uint32_t shmemDataRawShape0, uint32_t shmemDataRawShape1, uint32_t shmemDataRawShape2, uint32_t shmemDataRawShape3, __gm__ int64_t *hcclContext)
{
    (void)UBDataRawShape0;
    (void)shmemDataRawShape0;
    
    __ubuf__ UBType* UBDataAddr = UBDataBaseAddr + TileOp::CalcLinearOffset(UBDataRawShape1, UBDataOffset0, UBDataOffset1);
    __gm__ ShmemType* shmemDataAddr = MapAndOffsetShmem(
        hcclContext, shmemDataBaseAddr, shmemDataOffset0, shmemDataOffset1, shmemDataOffset2, shmemDataOffset3,
        shmemDataRawShape2, shmemDataRawShape3);
    
    CopyUbToGm<ShmemType, UBType, tileRowShape, tileColShape, bufferRowShape, bufferColShape, srcStride, dstStride, atomicType>(shmemDataAddr, UBDataAddr);
}

// Signal: write value to remote ranks; S→MTE3 sync so scalar write is visible to TSTORE.
template<int64_t value, int32_t stride, int32_t tileRowShape, int32_t tileColShape, AtomicType atomicType>
TILEOP void ShmemSignal(__ubuf__ int32_t* buffer, __gm__ int32_t* shmemSignalBaseAddr,
    uint32_t shmemSignalOffset0, uint32_t shmemSignalOffset1, uint32_t shmemSignalOffset2, uint32_t shmemSignalOffset3, uint32_t shmemSignalOffset4,
    uint32_t shmemSignalRawShape0, uint32_t shmemSignalRawShape1, uint32_t shmemSignalRawShape2, uint32_t shmemSignalRawShape3, uint32_t shmemSignalRawShape4,
    uint32_t shmemSignalShape0, uint32_t shmemSignalShape1, uint32_t shmemSignalShape2, uint32_t shmemSignalShape3, uint32_t shmemSignalShape4, __gm__ int64_t *hcclContext)
{
    (void)shmemSignalRawShape0;
    (void)shmemSignalRawShape1;
    (void)shmemSignalShape1;
    (void)shmemSignalShape2;
    
    int32_t tileCols = CeilDiv(static_cast<int32_t>(shmemSignalRawShape4), tileColShape);
    int32_t tileRows = CeilDiv(static_cast<int32_t>(shmemSignalRawShape3), tileRowShape);
    int32_t tileRow = static_cast<int32_t>(shmemSignalOffset3) / tileRowShape;
    int32_t tileCol = static_cast<int32_t>(shmemSignalOffset4) / tileColShape;
    int32_t tileIndex = tileRow * tileCols + tileCol;
    int32_t totalTileNum = tileRows * tileCols;

    buffer[0] = static_cast<int32_t>(value);
    constexpr uint32_t signalColShape = 8;  // 8*4=32B alignment
    ShmemUbTile<int32_t, 1, signalColShape> signalTile(1, 1);
    pto::TASSIGN(signalTile, reinterpret_cast<uintptr_t>(buffer));
    
    PIPE_SYNC_EVENT(PIPE_S, PIPE_MTE3, EVENT_ID0);
    
    ShapeDyn signalShape = MakeShape(1, 1);
    StrideDyn signalStride = MakeStride(1, 1);
    for (uint32_t rankId = shmemSignalOffset0; rankId < shmemSignalOffset0 + shmemSignalShape0; rankId++) {
        __gm__ int32_t* shmemSignalAddr = MapVirtualAddr<int32_t>(hcclContext, shmemSignalBaseAddr, rankId) +
            static_cast<int32_t>(shmemSignalOffset1) * static_cast<int32_t>(shmemSignalRawShape2) * totalTileNum * stride +
            (static_cast<int32_t>(shmemSignalOffset2) * totalTileNum + tileIndex) * stride;
        ShmemGlobalTensor<int32_t, 1, signalColShape> signalGlobal(shmemSignalAddr, signalShape, signalStride);
        
        AtomicStore<atomicType>(signalGlobal, signalTile);
    }
}

// Get: remote shmem GM → local GM.
template<typename NonShmemType, typename ShmemType, uint32_t tileRowShape, uint32_t tileColShape, uint32_t bufferRowShape,
    uint32_t bufferColShape, uint32_t srcStride, uint32_t dstStride, AtomicType atomicType>
TILEOP void ShmemGet(__gm__ NonShmemType* nonShmemDataBaseAddr, __ubuf__ NonShmemType* buffer, __gm__ ShmemType* shmemDataBaseAddr,
    uint32_t nonShmemDataOffset0, uint32_t nonShmemDataOffset1, uint32_t nonShmemDataRawShape0,
    uint32_t nonShmemDataRawShape1, uint32_t shmemDataOffset0, uint32_t shmemDataOffset1, uint32_t shmemDataOffset2, uint32_t shmemDataOffset3,
    uint32_t shmemDataRawShape0, uint32_t shmemDataRawShape1, uint32_t shmemDataRawShape2, uint32_t shmemDataRawShape3, __gm__ int64_t *hcclContext)
{
    (void)nonShmemDataRawShape0;
    (void)shmemDataRawShape0;
    
    __gm__ NonShmemType* nonShmemDataAddr = nonShmemDataBaseAddr + TileOp::CalcLinearOffset(nonShmemDataRawShape1,
        nonShmemDataOffset0, nonShmemDataOffset1);
    __gm__ ShmemType* shmemDataAddr = MapAndOffsetShmem(
        hcclContext, shmemDataBaseAddr, shmemDataOffset0, shmemDataOffset1, shmemDataOffset2, shmemDataOffset3,
        shmemDataRawShape2, shmemDataRawShape3);

    if constexpr (std::is_same_v<NonShmemType, ShmemType> && atomicType == AtomicType::SET &&
        (bufferColShape >= tileColShape)) {
        CopyGmToGmByTRowSliced<false, NonShmemType, tileRowShape, tileColShape, bufferRowShape, srcStride, dstStride>(
            nonShmemDataAddr, buffer, shmemDataAddr);
        return;
    }

    CopyGmToGm<NonShmemType, NonShmemType, ShmemType, tileRowShape, tileColShape, bufferRowShape, bufferColShape,
        srcStride, dstStride, atomicType>(nonShmemDataAddr, buffer, shmemDataAddr);
}

// Get: remote shmem GM → UB (single block, optional type conversion).
template<typename UBType, typename ShmemType, uint32_t tileRowShape, uint32_t tileColShape, uint32_t bufferRowShape,
    uint32_t bufferColShape, uint32_t srcStride, uint32_t dstStride, AtomicType atomicType>
TILEOP void ShmemGetGm2Ub(__ubuf__ UBType* UBDataBaseAddr, __ubuf__ UBType* buffer, __gm__ ShmemType* shmemDataBaseAddr,
    uint32_t UBDataOffset0, uint32_t UBDataOffset1, uint32_t UBDataRawShape0, uint32_t UBDataRawShape1,
    uint32_t shmemDataOffset0, uint32_t shmemDataOffset1, uint32_t shmemDataOffset2, uint32_t shmemDataOffset3,
    uint32_t shmemDataRawShape0, uint32_t shmemDataRawShape1, uint32_t shmemDataRawShape2, uint32_t shmemDataRawShape3, __gm__ int64_t *hcclContext)
{
    (void)tileRowShape;
    (void)tileColShape;
    (void)UBDataRawShape0;
    (void)shmemDataRawShape0;
    
    __ubuf__ UBType* UBDataAddr = UBDataBaseAddr + TileOp::CalcLinearOffset(UBDataRawShape1, UBDataOffset0, UBDataOffset1);
    __gm__ ShmemType* shmemDataAddr = MapAndOffsetShmem(
        hcclContext, shmemDataBaseAddr, shmemDataOffset0, shmemDataOffset1, shmemDataOffset2, shmemDataOffset3,
        shmemDataRawShape2, shmemDataRawShape3);
    
    CopyGmToUbBlock<UBType, ShmemType, bufferRowShape, bufferColShape, srcStride, dstStride>(UBDataAddr, buffer, shmemDataAddr);
}

} // namespace TileOp::Distributed

#endif