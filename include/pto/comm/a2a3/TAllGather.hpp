/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMM_A2A3_TALL_GATHER_HPP
#define PTO_COMM_A2A3_TALL_GATHER_HPP

#include <type_traits>

#include "pto/common/debug.h"
#include "pto/common/type.hpp"
#include "pto/common/constants.hpp"
#include "pto/common/pto_instr.hpp"
#include "pto/comm/comm_types.hpp"

namespace pto {
namespace comm {

// Simple AllGather: per-rank data fits in a single tile.
// Each rank reads all peers' data and writes to dst at proper offset.
template <typename ParallelGroupType, typename GlobalDstData, typename TileData>
PTO_INTERNAL void TallGatherSimple(ParallelGroupType &parallelGroup, GlobalDstData &dstGlobalData,
                                   TileData &stagingTileData, int gShape0, int gShape1, int gShape2, int gShape3,
                                   int gShape4, int nranks, int perRankRows)
{
    using GlobalSrcData = typename ParallelGroupTraits<ParallelGroupType>::GlobalDataType;
    using T = typename GlobalSrcData::RawDType;
    const int dstStride3 = dstGlobalData.GetStride(GlobalTensorDim::DIM_3);

    using DynShape5D = Shape<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    using DynStride = Stride<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    using DstViewT = GlobalTensor<T, DynShape5D, DynStride, GlobalDstData::layout>;

    DynShape5D perRankShape(gShape0, gShape1, gShape2, gShape3, gShape4);
    DynStride dstViewStride(
        dstGlobalData.GetStride(GlobalTensorDim::DIM_0), dstGlobalData.GetStride(GlobalTensorDim::DIM_1),
        dstGlobalData.GetStride(GlobalTensorDim::DIM_2), dstStride3, dstGlobalData.GetStride(GlobalTensorDim::DIM_4));

    for (int r = 0; r < nranks; ++r) {
        TLOAD(stagingTileData, parallelGroup[r]);
        set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        int64_t dstOffset = static_cast<int64_t>(r) * perRankRows * dstStride3;
        DstViewT dstView(dstGlobalData.data() + dstOffset, perRankShape, dstViewStride);
        TSTORE(dstView, stagingTileData);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    }
}

// Process one 2D tile chunk for AllGather: load from src, store to dst.
template <typename ParallelGroupType, typename GlobalDstData, typename TileData, typename DynStrideT>
PTO_INTERNAL void TallGatherProcessChunk(ParallelGroupType &parallelGroup, GlobalDstData &dstGlobalData,
                                         TileData &stagingTileData, int rankIdx, int64_t srcOffset, int64_t dstOffset,
                                         int curRows, int curCols, const DynStrideT &srcChunkStride,
                                         const DynStrideT &dstChunkStride)
{
    using GlobalSrcData = typename ParallelGroupTraits<ParallelGroupType>::GlobalDataType;
    using T = typename GlobalSrcData::RawDType;
    using ChunkShape = Shape<1, 1, 1, DYNAMIC, DYNAMIC>;
    using SrcChunkView = GlobalTensor<T, ChunkShape, DynStrideT, GlobalSrcData::layout>;
    using DstChunkView = GlobalTensor<T, ChunkShape, DynStrideT, GlobalDstData::layout>;
    constexpr bool isDynamicRow = (TileData::ValidRow == DYNAMIC);
    constexpr bool isDynamicCol = (TileData::ValidCol == DYNAMIC);

    if constexpr (isDynamicRow)
        stagingTileData.RowMaskInternal = curRows;
    if constexpr (isDynamicCol)
        stagingTileData.ColMaskInternal = curCols;

    ChunkShape chunkShape(1, 1, 1, curRows, curCols);
    SrcChunkView srcView(parallelGroup[rankIdx].data() + srcOffset, chunkShape, srcChunkStride);
    TLOAD(stagingTileData, srcView);
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    DstChunkView dstChunk(dstGlobalData.data() + dstOffset, chunkShape, dstChunkStride);
    TSTORE(dstChunk, stagingTileData);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
}

// 2D tile sliding for one rank's data block at given base offsets.
template <typename ParallelGroupType, typename GlobalDstData, typename TileData, typename DynStrideT>
PTO_INTERNAL void TallGatherSlide2D(ParallelGroupType &parallelGroup, GlobalDstData &dstGlobalData,
                                    TileData &stagingTileData, int rankIdx, int64_t srcBase, int64_t dstBase, int rows,
                                    int cols, int tileValidRow, int tileValidCol, const DynStrideT &srcChunkStride,
                                    const DynStrideT &dstChunkStride, int srcStride3, int srcStride4, int dstStride3,
                                    int dstStride4)
{
    for (int rowOff = 0; rowOff < rows; rowOff += tileValidRow) {
        int curRows = (rows - rowOff < tileValidRow) ? (rows - rowOff) : tileValidRow;
        for (int colOff = 0; colOff < cols; colOff += tileValidCol) {
            int curCols = (cols - colOff < tileValidCol) ? (cols - colOff) : tileValidCol;
            int64_t srcOff =
                srcBase + static_cast<int64_t>(rowOff) * srcStride3 + static_cast<int64_t>(colOff) * srcStride4;
            int64_t dstOff =
                dstBase + static_cast<int64_t>(rowOff) * dstStride3 + static_cast<int64_t>(colOff) * dstStride4;
            TallGatherProcessChunk(parallelGroup, dstGlobalData, stagingTileData, rankIdx, srcOff, dstOff, curRows,
                                   curCols, srcChunkStride, dstChunkStride);
        }
    }
}

// 2D sliding chunked AllGather with single buffer
template <typename ParallelGroupType, typename GlobalDstData, typename TileData>
PTO_INTERNAL void TallGatherChunkedSingle(ParallelGroupType &parallelGroup, GlobalDstData &dstGlobalData,
                                          TileData &stagingTileData, int gShape0, int gShape1, int gShape2, int gShape3,
                                          int gShape4, int tileValidRow, int tileValidCol, int nranks, int perRankRows)
{
    auto &refSrc = parallelGroup[0];
    const int srcS0 = refSrc.GetStride(GlobalTensorDim::DIM_0);
    const int srcS1 = refSrc.GetStride(GlobalTensorDim::DIM_1);
    const int srcS2 = refSrc.GetStride(GlobalTensorDim::DIM_2);
    const int srcS3 = refSrc.GetStride(GlobalTensorDim::DIM_3);
    const int srcS4 = refSrc.GetStride(GlobalTensorDim::DIM_4);
    const int dstS0 = dstGlobalData.GetStride(GlobalTensorDim::DIM_0);
    const int dstS1 = dstGlobalData.GetStride(GlobalTensorDim::DIM_1);
    const int dstS2 = dstGlobalData.GetStride(GlobalTensorDim::DIM_2);
    const int dstS3 = dstGlobalData.GetStride(GlobalTensorDim::DIM_3);
    const int dstS4 = dstGlobalData.GetStride(GlobalTensorDim::DIM_4);

    using DynStride = Stride<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    DynStride srcChunkStride(srcS0, srcS1, srcS2, srcS3, srcS4);
    DynStride dstChunkStride(dstS0, dstS1, dstS2, dstS3, dstS4);
    const int outerIters = gShape0 * gShape1 * gShape2;

    for (int r = 0; r < nranks; ++r) {
        const int64_t rankDstBase = static_cast<int64_t>(r) * perRankRows * dstS3;
        for (int idx = 0; idx < outerIters; ++idx) {
            int i2 = idx % gShape2;
            int rem = idx / gShape2;
            int i1 = rem % gShape1;
            int i0 = rem / gShape1;
            int64_t srcBase =
                static_cast<int64_t>(i0) * srcS0 + static_cast<int64_t>(i1) * srcS1 + static_cast<int64_t>(i2) * srcS2;
            int64_t dstBase = rankDstBase + static_cast<int64_t>(i0) * dstS0 + static_cast<int64_t>(i1) * dstS1 +
                              static_cast<int64_t>(i2) * dstS2;
            TallGatherSlide2D(parallelGroup, dstGlobalData, stagingTileData, r, srcBase, dstBase, gShape3, gShape4,
                              tileValidRow, tileValidCol, srcChunkStride, dstChunkStride, srcS3, srcS4, dstS3, dstS4);
        }
    }
}

// ============================================================================
// TALL_GATHER_IMPL: AllGather — each rank contributes a chunk, all get the
// full concatenated result.
//
// All ranks call this. Each rank reads all peers' source data via
// ParallelGroup and assembles the full output locally.
//
// parallelGroup[r] = rank r's source buffer (shape: per-rank data).
// dstGlobalData = this rank's output (shape: nranks * per-rank rows).
// ============================================================================

template <typename ParallelGroupType, typename GlobalDstData, typename TileData>
PTO_INTERNAL void TALL_GATHER_IMPL(ParallelGroupType &parallelGroup, GlobalDstData &dstGlobalData,
                                   TileData &stagingTileData)
{
    using GlobalSrcData = typename ParallelGroupTraits<ParallelGroupType>::GlobalDataType;
    using T = typename GlobalSrcData::RawDType;

    static_assert(std::is_same_v<T, typename GlobalDstData::RawDType>, "TALL_GATHER: type mismatch!");
    static_assert(std::is_same_v<T, typename TileData::DType>, "TALL_GATHER: tile type must match GlobalData");

    const int nranks = parallelGroup.GetSize();
    PTO_ASSERT(nranks > 0, "TALL_GATHER: nranks must be > 0");

    auto &srcRef = parallelGroup[0];
    const int gShape0 = srcRef.GetShape(GlobalTensorDim::DIM_0);
    const int gShape1 = srcRef.GetShape(GlobalTensorDim::DIM_1);
    const int gShape2 = srcRef.GetShape(GlobalTensorDim::DIM_2);
    const int gShape3 = srcRef.GetShape(GlobalTensorDim::DIM_3);
    const int gShape4 = srcRef.GetShape(GlobalTensorDim::DIM_4);

    const int perRankRows = gShape3;
    const int tileValidRow = stagingTileData.GetValidRow();
    const int tileValidCol = stagingTileData.GetValidCol();
    const int64_t totalRows = static_cast<int64_t>(gShape0) * gShape1 * gShape2 * gShape3;

    PTO_ASSERT(tileValidRow > 0 && tileValidCol > 0, "TALL_GATHER: tile dims must be > 0");

    if (totalRows == 0 || gShape4 == 0) {
        return;
    }

    if (totalRows <= tileValidRow && gShape4 <= tileValidCol) {
        TallGatherSimple<ParallelGroupType, GlobalDstData, TileData>(parallelGroup, dstGlobalData, stagingTileData,
                                                                     gShape0, gShape1, gShape2, gShape3, gShape4,
                                                                     nranks, perRankRows);
        return;
    }

    constexpr bool isDynamicRow = (TileData::ValidRow == DYNAMIC);
    constexpr bool isDynamicCol = (TileData::ValidCol == DYNAMIC);
    if constexpr (!isDynamicRow) {
        PTO_ASSERT(gShape3 % tileValidRow == 0, "TALL_GATHER: per-rank rows must be divisible by tile ValidRow");
    }
    if constexpr (!isDynamicCol) {
        PTO_ASSERT(gShape4 % tileValidCol == 0, "TALL_GATHER: cols must be divisible by tile ValidCol");
    }

    TallGatherChunkedSingle<ParallelGroupType, GlobalDstData, TileData>(
        parallelGroup, dstGlobalData, stagingTileData, gShape0, gShape1, gShape2, gShape3, gShape4, tileValidRow,
        tileValidCol, nranks, perRankRows);
}

// Ping-pong overload
template <typename ParallelGroupType, typename GlobalDstData, typename TileData>
PTO_INTERNAL void TALL_GATHER_IMPL(ParallelGroupType &parallelGroup, GlobalDstData &dstGlobalData,
                                   TileData &pingTileData, TileData &pongTileData)
{
    // For AllGather the ping-pong benefit is in overlapping TLOAD[next] with TSTORE[current].
    // Simplified: use single-buffer path with pingTileData as staging.
    TALL_GATHER_IMPL(parallelGroup, dstGlobalData, pingTileData);
    (void)pongTileData;
}

#ifndef PTO_COMM_A5_TALL_GATHER_PROVIDED
template <CollEngine engine = CollEngine::CCU, typename... Args>
PTO_INTERNAL void TALL_GATHER_CCU_IMPL(Args &&...)
{
    static_assert(engine != CollEngine::CCU,
                  "TALL_GATHER<CollEngine::CCU> requires A5 hardware; "
                  "CCU engine is not available on A2/A3.");
}
#endif

} // namespace comm
} // namespace pto

#endif // PTO_COMM_A2A3_TALL_GATHER_HPP
