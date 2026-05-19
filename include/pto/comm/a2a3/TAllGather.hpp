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

// 2D sliding chunked AllGather with single buffer
template <typename ParallelGroupType, typename GlobalDstData, typename TileData>
PTO_INTERNAL void TallGatherChunkedSingle(ParallelGroupType &parallelGroup, GlobalDstData &dstGlobalData,
                                          TileData &stagingTileData, int gShape0, int gShape1, int gShape2, int gShape3,
                                          int gShape4, int tileValidRow, int tileValidCol, int nranks, int perRankRows)
{
    using GlobalSrcData = typename ParallelGroupTraits<ParallelGroupType>::GlobalDataType;
    using T = typename GlobalSrcData::RawDType;
    constexpr bool isDynamicRow = (TileData::ValidRow == DYNAMIC);
    constexpr bool isDynamicCol = (TileData::ValidCol == DYNAMIC);

    auto &refSrc = parallelGroup[0];
    const int srcStep[5] = {static_cast<int>(refSrc.GetStride(GlobalTensorDim::DIM_0)),
                            static_cast<int>(refSrc.GetStride(GlobalTensorDim::DIM_1)),
                            static_cast<int>(refSrc.GetStride(GlobalTensorDim::DIM_2)),
                            static_cast<int>(refSrc.GetStride(GlobalTensorDim::DIM_3)),
                            static_cast<int>(refSrc.GetStride(GlobalTensorDim::DIM_4))};
    const int dstStep[5] = {static_cast<int>(dstGlobalData.GetStride(GlobalTensorDim::DIM_0)),
                            static_cast<int>(dstGlobalData.GetStride(GlobalTensorDim::DIM_1)),
                            static_cast<int>(dstGlobalData.GetStride(GlobalTensorDim::DIM_2)),
                            static_cast<int>(dstGlobalData.GetStride(GlobalTensorDim::DIM_3)),
                            static_cast<int>(dstGlobalData.GetStride(GlobalTensorDim::DIM_4))};

    using ChunkShape = Shape<1, 1, 1, DYNAMIC, DYNAMIC>;
    using DynStride = Stride<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    using SrcChunkView = GlobalTensor<T, ChunkShape, DynStride, GlobalSrcData::layout>;
    using DstChunkView = GlobalTensor<T, ChunkShape, DynStride, GlobalDstData::layout>;
    DynStride srcChunkStride(srcStep[0], srcStep[1], srcStep[2], srcStep[3], srcStep[4]);
    DynStride dstChunkStride(dstStep[0], dstStep[1], dstStep[2], dstStep[3], dstStep[4]);

    for (int r = 0; r < nranks; ++r) {
        const int64_t rankDstBase = static_cast<int64_t>(r) * perRankRows * dstStep[3];
        for (int i0 = 0; i0 < gShape0; ++i0) {
            for (int i1 = 0; i1 < gShape1; ++i1) {
                for (int i2 = 0; i2 < gShape2; ++i2) {
                    const int64_t srcBase = static_cast<int64_t>(i0) * srcStep[0] +
                                            static_cast<int64_t>(i1) * srcStep[1] +
                                            static_cast<int64_t>(i2) * srcStep[2];
                    const int64_t dstBase = rankDstBase + static_cast<int64_t>(i0) * dstStep[0] +
                                            static_cast<int64_t>(i1) * dstStep[1] +
                                            static_cast<int64_t>(i2) * dstStep[2];
                    for (int rowOff = 0; rowOff < gShape3; rowOff += tileValidRow) {
                        int curRows = (gShape3 - rowOff < tileValidRow) ? (gShape3 - rowOff) : tileValidRow;
                        if constexpr (isDynamicRow)
                            stagingTileData.RowMaskInternal = curRows;
                        for (int colOff = 0; colOff < gShape4; colOff += tileValidCol) {
                            int curCols = (gShape4 - colOff < tileValidCol) ? (gShape4 - colOff) : tileValidCol;
                            if constexpr (isDynamicCol)
                                stagingTileData.ColMaskInternal = curCols;
                            const int64_t srcOff = srcBase + static_cast<int64_t>(rowOff) * srcStep[3] +
                                                   static_cast<int64_t>(colOff) * srcStep[4];
                            const int64_t dstOff = dstBase + static_cast<int64_t>(rowOff) * dstStep[3] +
                                                   static_cast<int64_t>(colOff) * dstStep[4];
                            ChunkShape chunkShape(1, 1, 1, curRows, curCols);
                            SrcChunkView srcView(parallelGroup[r].data() + srcOff, chunkShape, srcChunkStride);
                            TLOAD(stagingTileData, srcView);
                            set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
                            wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
                            DstChunkView dstChunk(dstGlobalData.data() + dstOff, chunkShape, dstChunkStride);
                            TSTORE(dstChunk, stagingTileData);
                            set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
                            wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
                        }
                    }
                }
            }
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
