/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMM_A2A3_TREDUCE_SCATTER_HPP
#define PTO_COMM_A2A3_TREDUCE_SCATTER_HPP

#include <type_traits>

#include "pto/common/debug.h"
#include "pto/common/type.hpp"
#include "pto/common/constants.hpp"
#include "pto/common/pto_instr.hpp"
#include "pto/comm/comm_types.hpp"

namespace pto {
namespace comm {

// Simple ReduceScatter: each rank's slice fits in one tile.
// This rank reads its designated slice from all peers, reduces, and stores.
template <typename ParallelGroupType, typename GlobalDstData, typename TileData>
PTO_INTERNAL void TreduceScatterSimple(ParallelGroupType &parallelGroup, GlobalDstData &dstGlobalData,
                                       TileData &accTileData, TileData &recvTileData, ReduceOp op, int selfIdx,
                                       int nranks, int64_t sliceOffset)
{
    using GlobalSrcData = typename ParallelGroupTraits<ParallelGroupType>::GlobalDataType;
    using T = typename GlobalSrcData::RawDType;
    using DynShape = Shape<1, 1, 1, DYNAMIC, DYNAMIC>;
    using DynStride = Stride<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    using SrcViewT = GlobalTensor<T, DynShape, DynStride, GlobalSrcData::layout>;

    auto &refTensor = parallelGroup[0];
    const int sliceRows = dstGlobalData.GetShape(GlobalTensorDim::DIM_3);
    const int sliceCols = dstGlobalData.GetShape(GlobalTensorDim::DIM_4);
    DynShape sliceShape(1, 1, 1, sliceRows, sliceCols);
    DynStride srcStride(refTensor.GetStride(GlobalTensorDim::DIM_0), refTensor.GetStride(GlobalTensorDim::DIM_1),
                        refTensor.GetStride(GlobalTensorDim::DIM_2), refTensor.GetStride(GlobalTensorDim::DIM_3),
                        refTensor.GetStride(GlobalTensorDim::DIM_4));

    // Load first rank's slice into accumulator
    SrcViewT firstView(parallelGroup[0].data() + sliceOffset, sliceShape, srcStride);
    TLOAD(accTileData, firstView);

    if (nranks == 1) {
        set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        TSTORE(dstGlobalData, accTileData);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        return;
    }

    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    for (int r = 1; r < nranks; ++r) {
        SrcViewT remoteView(parallelGroup[r].data() + sliceOffset, sliceShape, srcStride);
        TLOAD(recvTileData, remoteView);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
        detail::ReduceTiles(accTileData, recvTileData, op);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    }

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(dstGlobalData, accTileData);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
}

// Process one chunk of ReduceScatter (single buffer)
template <typename ParallelGroupType, typename GlobalDstData, typename TileData, typename DynStrideT>
PTO_INTERNAL void TreduceScatterProcessChunkSingle(ParallelGroupType &parallelGroup, GlobalDstData &dstGlobalData,
                                                   TileData &accTileData, TileData &recvTileData, ReduceOp op,
                                                   int64_t srcOffset, int64_t dstOffset, int currentRows,
                                                   int currentCols, const DynStrideT &srcChunkStride,
                                                   const DynStrideT &dstChunkStride, int nranks)
{
    using GlobalSrcData = typename ParallelGroupTraits<ParallelGroupType>::GlobalDataType;
    using T = typename GlobalSrcData::RawDType;
    using DynShape = Shape<1, 1, 1, DYNAMIC, DYNAMIC>;
    using SrcViewT = GlobalTensor<T, DynShape, DynStrideT, GlobalSrcData::layout>;
    using DstViewT = GlobalTensor<T, DynShape, DynStrideT, GlobalDstData::layout>;
    constexpr bool isDynamicRow = (TileData::ValidRow == DYNAMIC);
    constexpr bool isDynamicCol = (TileData::ValidCol == DYNAMIC);

    if constexpr (isDynamicRow) {
        accTileData.RowMaskInternal = currentRows;
        recvTileData.RowMaskInternal = currentRows;
    }
    if constexpr (isDynamicCol) {
        accTileData.ColMaskInternal = currentCols;
        recvTileData.ColMaskInternal = currentCols;
    }

    DynShape chunkShape(1, 1, 1, currentRows, currentCols);
    SrcViewT firstView(parallelGroup[0].data() + srcOffset, chunkShape, srcChunkStride);
    TLOAD(accTileData, firstView);

    if (nranks == 1) {
        set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    } else {
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        for (int r = 1; r < nranks; ++r) {
            SrcViewT remoteView(parallelGroup[r].data() + srcOffset, chunkShape, srcChunkStride);
            TLOAD(recvTileData, remoteView);
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
            detail::ReduceTiles(accTileData, recvTileData, op);
            set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        }
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    }

    DstViewT dstView(dstGlobalData.data() + dstOffset, chunkShape, dstChunkStride);
    TSTORE(dstView, accTileData);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
}

// 2D sliding chunked ReduceScatter with single buffer
template <typename ParallelGroupType, typename GlobalDstData, typename TileData>
PTO_INTERNAL void TreduceScatterChunkedSingle(ParallelGroupType &parallelGroup, GlobalDstData &dstGlobalData,
                                              TileData &accTileData, TileData &recvTileData, ReduceOp op, int gShape0,
                                              int gShape1, int gShape2, int sliceRows, int sliceCols, int tileValidRow,
                                              int tileValidCol, int nranks, int64_t sliceBaseOffset)
{
    using GlobalSrcData = typename ParallelGroupTraits<ParallelGroupType>::GlobalDataType;
    auto &refTensor = parallelGroup[0];
    const int srcStride3 = refTensor.GetStride(GlobalTensorDim::DIM_3);
    const int srcStride4 = refTensor.GetStride(GlobalTensorDim::DIM_4);
    const int srcStride0 = refTensor.GetStride(GlobalTensorDim::DIM_0);
    const int srcStride1 = refTensor.GetStride(GlobalTensorDim::DIM_1);
    const int srcStride2 = refTensor.GetStride(GlobalTensorDim::DIM_2);
    const int dstStride0 = dstGlobalData.GetStride(GlobalTensorDim::DIM_0);
    const int dstStride1 = dstGlobalData.GetStride(GlobalTensorDim::DIM_1);
    const int dstStride2 = dstGlobalData.GetStride(GlobalTensorDim::DIM_2);
    const int dstStride3 = dstGlobalData.GetStride(GlobalTensorDim::DIM_3);
    const int dstStride4 = dstGlobalData.GetStride(GlobalTensorDim::DIM_4);

    using DynStride = Stride<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    DynStride srcChunkStride(srcStride0, srcStride1, srcStride2, srcStride3, srcStride4);
    DynStride dstChunkStride(dstStride0, dstStride1, dstStride2, dstStride3, dstStride4);

    for (int i0 = 0; i0 < gShape0; ++i0) {
        for (int i1 = 0; i1 < gShape1; ++i1) {
            for (int i2 = 0; i2 < gShape2; ++i2) {
                int64_t srcBase = sliceBaseOffset + static_cast<int64_t>(i0) * srcStride0 +
                                  static_cast<int64_t>(i1) * srcStride1 + static_cast<int64_t>(i2) * srcStride2;
                int64_t dstBase = static_cast<int64_t>(i0) * dstStride0 + static_cast<int64_t>(i1) * dstStride1 +
                                  static_cast<int64_t>(i2) * dstStride2;
                for (int rowOff = 0; rowOff < sliceRows; rowOff += tileValidRow) {
                    int curRows = (rowOff + tileValidRow <= sliceRows) ? tileValidRow : (sliceRows - rowOff);
                    for (int colOff = 0; colOff < sliceCols; colOff += tileValidCol) {
                        int curCols = (colOff + tileValidCol <= sliceCols) ? tileValidCol : (sliceCols - colOff);
                        int64_t srcOff = srcBase + static_cast<int64_t>(rowOff) * srcStride3 +
                                         static_cast<int64_t>(colOff) * srcStride4;
                        int64_t dstOff = dstBase + static_cast<int64_t>(rowOff) * dstStride3 +
                                         static_cast<int64_t>(colOff) * dstStride4;
                        TreduceScatterProcessChunkSingle<ParallelGroupType, GlobalDstData, TileData>(
                            parallelGroup, dstGlobalData, accTileData, recvTileData, op, srcOff, dstOff, curRows,
                            curCols, srcChunkStride, dstChunkStride, nranks);
                    }
                }
            }
        }
    }
}

// Ping-pong loop for ReduceScatter (reduce across ranks for one chunk)
template <typename ParallelGroupType, typename TileData, typename DynStrideT>
PTO_INTERNAL void TreduceScatterPingPongLoop(ParallelGroupType &parallelGroup, TileData &accTileData,
                                             TileData &pingTile, TileData &pongTile, ReduceOp op, int64_t srcOffset,
                                             int currentRows, int currentCols, const DynStrideT &srcChunkStride,
                                             int nranks)
{
    using GlobalSrcData = typename ParallelGroupTraits<ParallelGroupType>::GlobalDataType;
    using T = typename GlobalSrcData::RawDType;
    using DynShape = Shape<1, 1, 1, DYNAMIC, DYNAMIC>;
    using SrcViewT = GlobalTensor<T, DynShape, DynStrideT, GlobalSrcData::layout>;
    DynShape chunkShape(1, 1, 1, currentRows, currentCols);

    // Prefetch rank 1 into pingTile
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    SrcViewT firstRemoteView(parallelGroup[1].data() + srcOffset, chunkShape, srcChunkStride);
    TLOAD(pingTile, firstRemoteView);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    const int numRemote = nranks - 1;
    for (int i = 0; i < numRemote; ++i) {
        const bool scheduleNext = (i + 1 < numRemote);
        const bool currentIsPing = ((i & 1) == 0);
        TileData &currentTile = currentIsPing ? pingTile : pongTile;
        TileData &nextTile = currentIsPing ? pongTile : pingTile;
        const event_t currentReady = currentIsPing ? EVENT_ID1 : EVENT_ID2;
        const event_t nextReady = currentIsPing ? EVENT_ID2 : EVENT_ID1;
        if (scheduleNext) {
            SrcViewT nextView(parallelGroup[i + 2].data() + srcOffset, chunkShape, srcChunkStride);
            TLOAD(nextTile, nextView);
            set_flag(PIPE_MTE2, PIPE_V, nextReady);
        }
        wait_flag(PIPE_MTE2, PIPE_V, currentReady);
        detail::ReduceTiles(accTileData, currentTile, op);
        if (!scheduleNext) {
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            continue;
        }
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    }
}

// Process one chunk with ping-pong for ReduceScatter
template <typename ParallelGroupType, typename GlobalDstData, typename TileData, typename DynStrideT>
PTO_INTERNAL void TreduceScatterProcessChunkPingPong(ParallelGroupType &parallelGroup, GlobalDstData &dstGlobalData,
                                                     TileData &accTileData, TileData &pingTile, TileData &pongTile,
                                                     ReduceOp op, int64_t srcOffset, int64_t dstOffset, int currentRows,
                                                     int currentCols, const DynStrideT &srcChunkStride,
                                                     const DynStrideT &dstChunkStride, int nranks)
{
    using GlobalSrcData = typename ParallelGroupTraits<ParallelGroupType>::GlobalDataType;
    using T = typename GlobalSrcData::RawDType;
    using DynShape = Shape<1, 1, 1, DYNAMIC, DYNAMIC>;
    using SrcViewT = GlobalTensor<T, DynShape, DynStrideT, GlobalSrcData::layout>;
    using DstViewT = GlobalTensor<T, DynShape, DynStrideT, GlobalDstData::layout>;
    constexpr bool isDynamicRow = (TileData::ValidRow == DYNAMIC);
    constexpr bool isDynamicCol = (TileData::ValidCol == DYNAMIC);

    if constexpr (isDynamicRow) {
        accTileData.RowMaskInternal = currentRows;
        pingTile.RowMaskInternal = currentRows;
        pongTile.RowMaskInternal = currentRows;
    }
    if constexpr (isDynamicCol) {
        accTileData.ColMaskInternal = currentCols;
        pingTile.ColMaskInternal = currentCols;
        pongTile.ColMaskInternal = currentCols;
    }

    DynShape chunkShape(1, 1, 1, currentRows, currentCols);
    SrcViewT rootView(parallelGroup[0].data() + srcOffset, chunkShape, srcChunkStride);
    TLOAD(accTileData, rootView);

    if (nranks == 1) {
        set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    } else {
        TreduceScatterPingPongLoop<ParallelGroupType, TileData>(parallelGroup, accTileData, pingTile, pongTile, op,
                                                                srcOffset, currentRows, currentCols, srcChunkStride,
                                                                nranks);
    }

    DstViewT dstView(dstGlobalData.data() + dstOffset, chunkShape, dstChunkStride);
    TSTORE(dstView, accTileData);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
}

// 2D sliding chunked ReduceScatter with ping-pong
template <typename ParallelGroupType, typename GlobalDstData, typename TileData>
PTO_INTERNAL void TreduceScatterChunkedPingPong(ParallelGroupType &parallelGroup, GlobalDstData &dstGlobalData,
                                                TileData &accTileData, TileData &pingTile, TileData &pongTile,
                                                ReduceOp op, int gShape0, int gShape1, int gShape2, int sliceRows,
                                                int sliceCols, int tileValidRow, int tileValidCol, int nranks,
                                                int64_t sliceBaseOffset)
{
    using GlobalSrcData = typename ParallelGroupTraits<ParallelGroupType>::GlobalDataType;
    auto &refTensor = parallelGroup[0];
    const int srcStride[5] = {static_cast<int>(refTensor.GetStride(GlobalTensorDim::DIM_0)),
                              static_cast<int>(refTensor.GetStride(GlobalTensorDim::DIM_1)),
                              static_cast<int>(refTensor.GetStride(GlobalTensorDim::DIM_2)),
                              static_cast<int>(refTensor.GetStride(GlobalTensorDim::DIM_3)),
                              static_cast<int>(refTensor.GetStride(GlobalTensorDim::DIM_4))};
    const int dstStride[5] = {static_cast<int>(dstGlobalData.GetStride(GlobalTensorDim::DIM_0)),
                              static_cast<int>(dstGlobalData.GetStride(GlobalTensorDim::DIM_1)),
                              static_cast<int>(dstGlobalData.GetStride(GlobalTensorDim::DIM_2)),
                              static_cast<int>(dstGlobalData.GetStride(GlobalTensorDim::DIM_3)),
                              static_cast<int>(dstGlobalData.GetStride(GlobalTensorDim::DIM_4))};

    using DynStride = Stride<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    DynStride srcChunkStride(srcStride[0], srcStride[1], srcStride[2], srcStride[3], srcStride[4]);
    DynStride dstChunkStride(dstStride[0], dstStride[1], dstStride[2], dstStride[3], dstStride[4]);

    for (int i0 = 0; i0 < gShape0; ++i0) {
        for (int i1 = 0; i1 < gShape1; ++i1) {
            for (int i2 = 0; i2 < gShape2; ++i2) {
                int64_t srcBase = sliceBaseOffset + static_cast<int64_t>(i0) * srcStride[0] +
                                  static_cast<int64_t>(i1) * srcStride[1] + static_cast<int64_t>(i2) * srcStride[2];
                int64_t dstBase = static_cast<int64_t>(i0) * dstStride[0] + static_cast<int64_t>(i1) * dstStride[1] +
                                  static_cast<int64_t>(i2) * dstStride[2];
                for (int rowOff = 0; rowOff < sliceRows; rowOff += tileValidRow) {
                    int curRows = (rowOff + tileValidRow <= sliceRows) ? tileValidRow : (sliceRows - rowOff);
                    for (int colOff = 0; colOff < sliceCols; colOff += tileValidCol) {
                        int curCols = (colOff + tileValidCol <= sliceCols) ? tileValidCol : (sliceCols - colOff);
                        int64_t srcOff = srcBase + static_cast<int64_t>(rowOff) * srcStride[3] +
                                         static_cast<int64_t>(colOff) * srcStride[4];
                        int64_t dstOff = dstBase + static_cast<int64_t>(rowOff) * dstStride[3] +
                                         static_cast<int64_t>(colOff) * dstStride[4];
                        TreduceScatterProcessChunkPingPong<ParallelGroupType, GlobalDstData, TileData>(
                            parallelGroup, dstGlobalData, accTileData, pingTile, pongTile, op, srcOff, dstOff, curRows,
                            curCols, srcChunkStride, dstChunkStride, nranks);
                    }
                }
            }
        }
    }
}

// ============================================================================
// TREDUCE_SCATTER_IMPL: ReduceScatter — all ranks reduce, each keeps one slice.
//
// All ranks call this. Each rank reduces elements across all peers for its own
// slice (determined by selfIdx) and stores the result locally.
//
// Input layout: parallelGroup[r] has full data of size N * sliceSize.
// Output: dstGlobalData receives only this rank's slice (sliceSize elements).
// ============================================================================

template <typename ParallelGroupType, typename GlobalDstData, typename TileData>
PTO_INTERNAL void TREDUCE_SCATTER_IMPL(ParallelGroupType &parallelGroup, GlobalDstData &dstGlobalData,
                                       TileData &accTileData, TileData &recvTileData, ReduceOp op)
{
    using GlobalSrcData = typename ParallelGroupTraits<ParallelGroupType>::GlobalDataType;
    using T = typename GlobalSrcData::RawDType;

    static_assert(std::is_same_v<T, typename GlobalDstData::RawDType>, "TREDUCE_SCATTER: type mismatch!");
    static_assert(std::is_same_v<T, typename TileData::DType>, "TREDUCE_SCATTER: tile type must match GlobalData");

    const int selfIdx = parallelGroup.GetRootIdx();
    const int nranks = parallelGroup.GetSize();

    PTO_ASSERT(nranks > 0, "TREDUCE_SCATTER: nranks must be > 0");
    PTO_ASSERT(selfIdx >= 0 && selfIdx < nranks, "TREDUCE_SCATTER: selfIdx out of range");

    auto &refTensor = parallelGroup[0];
    const int gShape0 = refTensor.GetShape(GlobalTensorDim::DIM_0);
    const int gShape1 = refTensor.GetShape(GlobalTensorDim::DIM_1);
    const int gShape2 = refTensor.GetShape(GlobalTensorDim::DIM_2);
    const int totalSrcRows = refTensor.GetShape(GlobalTensorDim::DIM_3);
    const int gShape4 = refTensor.GetShape(GlobalTensorDim::DIM_4);

    const int sliceRows = totalSrcRows / nranks;
    PTO_ASSERT(totalSrcRows == sliceRows * nranks, "TREDUCE_SCATTER: src rows must be divisible by nranks");

    const int srcStride3 = refTensor.GetStride(GlobalTensorDim::DIM_3);
    const int64_t sliceBaseOffset = static_cast<int64_t>(selfIdx) * sliceRows * srcStride3;

    const int tileValidRow = accTileData.GetValidRow();
    const int tileValidCol = accTileData.GetValidCol();
    PTO_ASSERT(tileValidRow > 0 && tileValidCol > 0, "TREDUCE_SCATTER: tile dims must be > 0");

    const int64_t totalSliceRows = static_cast<int64_t>(gShape0) * gShape1 * gShape2 * sliceRows;
    if (totalSliceRows == 0 || gShape4 == 0) {
        return;
    }

    if (totalSliceRows <= tileValidRow && gShape4 <= tileValidCol) {
        TreduceScatterSimple<ParallelGroupType, GlobalDstData, TileData>(
            parallelGroup, dstGlobalData, accTileData, recvTileData, op, selfIdx, nranks, sliceBaseOffset);
        return;
    }

    constexpr bool isDynamicRow = (TileData::ValidRow == DYNAMIC);
    constexpr bool isDynamicCol = (TileData::ValidCol == DYNAMIC);
    if constexpr (!isDynamicRow) {
        PTO_ASSERT(sliceRows % tileValidRow == 0, "TREDUCE_SCATTER: sliceRows must be divisible by tile ValidRow");
    }
    if constexpr (!isDynamicCol) {
        PTO_ASSERT(gShape4 % tileValidCol == 0, "TREDUCE_SCATTER: cols must be divisible by tile ValidCol");
    }

    TreduceScatterChunkedSingle<ParallelGroupType, GlobalDstData, TileData>(
        parallelGroup, dstGlobalData, accTileData, recvTileData, op, gShape0, gShape1, gShape2, sliceRows, gShape4,
        tileValidRow, tileValidCol, nranks, sliceBaseOffset);
}

// Ping-pong overload
template <typename ParallelGroupType, typename GlobalDstData, typename TileData>
PTO_INTERNAL void TREDUCE_SCATTER_IMPL(ParallelGroupType &parallelGroup, GlobalDstData &dstGlobalData,
                                       TileData &accTileData, TileData &pingTile, TileData &pongTile, ReduceOp op)
{
    using GlobalSrcData = typename ParallelGroupTraits<ParallelGroupType>::GlobalDataType;
    using T = typename GlobalSrcData::RawDType;

    static_assert(std::is_same_v<T, typename GlobalDstData::RawDType>, "TREDUCE_SCATTER: type mismatch!");
    static_assert(std::is_same_v<T, typename TileData::DType>, "TREDUCE_SCATTER: tile type must match GlobalData");

    const int selfIdx = parallelGroup.GetRootIdx();
    const int nranks = parallelGroup.GetSize();

    PTO_ASSERT(nranks > 0, "TREDUCE_SCATTER: nranks must be > 0");
    PTO_ASSERT(selfIdx >= 0 && selfIdx < nranks, "TREDUCE_SCATTER: selfIdx out of range");

    auto &refTensor = parallelGroup[0];
    const int gShape0 = refTensor.GetShape(GlobalTensorDim::DIM_0);
    const int gShape1 = refTensor.GetShape(GlobalTensorDim::DIM_1);
    const int gShape2 = refTensor.GetShape(GlobalTensorDim::DIM_2);
    const int totalSrcRows = refTensor.GetShape(GlobalTensorDim::DIM_3);
    const int gShape4 = refTensor.GetShape(GlobalTensorDim::DIM_4);

    const int sliceRows = totalSrcRows / nranks;
    PTO_ASSERT(totalSrcRows == sliceRows * nranks, "TREDUCE_SCATTER: src rows must be divisible by nranks");

    const int srcStride3 = refTensor.GetStride(GlobalTensorDim::DIM_3);
    const int64_t sliceBaseOffset = static_cast<int64_t>(selfIdx) * sliceRows * srcStride3;

    const int tileValidRow = accTileData.GetValidRow();
    const int tileValidCol = accTileData.GetValidCol();
    PTO_ASSERT(tileValidRow > 0 && tileValidCol > 0, "TREDUCE_SCATTER: tile dims must be > 0");

    const int64_t totalSliceRows = static_cast<int64_t>(gShape0) * gShape1 * gShape2 * sliceRows;
    if (totalSliceRows == 0 || gShape4 == 0) {
        return;
    }

    if (totalSliceRows <= tileValidRow && gShape4 <= tileValidCol) {
        // For simple case with ping-pong, reuse single-buffer simple path
        TreduceScatterSimple<ParallelGroupType, GlobalDstData, TileData>(
            parallelGroup, dstGlobalData, accTileData, pingTile, op, selfIdx, nranks, sliceBaseOffset);
        return;
    }

    constexpr bool isDynamicRow = (TileData::ValidRow == DYNAMIC);
    constexpr bool isDynamicCol = (TileData::ValidCol == DYNAMIC);
    if constexpr (!isDynamicRow) {
        PTO_ASSERT(sliceRows % tileValidRow == 0, "TREDUCE_SCATTER: sliceRows must be divisible by tile ValidRow");
    }
    if constexpr (!isDynamicCol) {
        PTO_ASSERT(gShape4 % tileValidCol == 0, "TREDUCE_SCATTER: cols must be divisible by tile ValidCol");
    }

    TreduceScatterChunkedPingPong<ParallelGroupType, GlobalDstData, TileData>(
        parallelGroup, dstGlobalData, accTileData, pingTile, pongTile, op, gShape0, gShape1, gShape2, sliceRows,
        gShape4, tileValidRow, tileValidCol, nranks, sliceBaseOffset);
}

#ifndef PTO_COMM_A5_TREDUCE_SCATTER_PROVIDED
template <CollEngine engine = CollEngine::CCU, typename... Args>
PTO_INTERNAL void TREDUCE_SCATTER_CCU_IMPL(Args &&...)
{
    static_assert(engine != CollEngine::CCU,
                  "TREDUCE_SCATTER<CollEngine::CCU> requires A5 hardware; "
                  "CCU engine is not available on A2/A3.");
}
#endif

} // namespace comm
} // namespace pto

#endif // PTO_COMM_A2A3_TREDUCE_SCATTER_HPP
