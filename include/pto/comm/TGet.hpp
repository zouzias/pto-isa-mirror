/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMM_TGET_HPP
#define PTO_COMM_TGET_HPP

#include <type_traits>

#include "pto/common/debug.h"
#include "pto/common/type.hpp"
#include "pto/common/constants.hpp"
#include "pto/common/pto_instr.hpp"
#include "pto/comm/comm_types.hpp"

namespace pto {
namespace comm {

// Ping-pong double-buffering state for chunked TGET transfers
struct TgetPingPongState {
    bool usePing = true;
    bool hasPending = false;
    int64_t pendingDstOffset = 0;
    int pendingRows = 0;
    int pendingCols = 0;
};

// Single synchronous transfer: TLOAD from src → sync → TSTORE to dst → sync
template <typename TileData, typename DstGT, typename SrcGT>
PTO_INTERNAL void TgetTransferOnce(DstGT &dst, SrcGT &src, TileData &tile)
{
    TLOAD(tile, src);
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    TSTORE(dst, tile);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
}

// Process one (dim0, dim1, dim2) slice with 2D row/col sliding for chunked TGET
template <typename GlobalDstData, typename GlobalSrcData, typename TileData, typename SrcViewT, typename DstViewT,
          typename DynShape, typename DynStride>
PTO_INTERNAL void TgetChunked2DSlice(GlobalDstData &dstGlobalData, GlobalSrcData &srcGlobalData,
                                     TileData &stagingTileData, int64_t srcBase, int64_t dstBase, int gShape3,
                                     int gShape4, int tileValidRow, int tileValidCol, const int (&srcStep)[5],
                                     const int (&dstStep)[5], const DynStride &srcStride, const DynStride &dstStride)
{
    constexpr bool isDynamicRow = (TileData::ValidRow == DYNAMIC);
    constexpr bool isDynamicCol = (TileData::ValidCol == DYNAMIC);
    int rowCursor = 0;
    while (rowCursor < gShape3) {
        const int curRows = (gShape3 - rowCursor < tileValidRow) ? (gShape3 - rowCursor) : tileValidRow;
        if constexpr (isDynamicRow)
            stagingTileData.RowMaskInternal = curRows;
        int colCursor = 0;
        while (colCursor < gShape4) {
            const int curCols = (gShape4 - colCursor < tileValidCol) ? (gShape4 - colCursor) : tileValidCol;
            if constexpr (isDynamicCol)
                stagingTileData.ColMaskInternal = curCols;
            const int64_t srcOff =
                srcBase + static_cast<int64_t>(rowCursor) * srcStep[3] + static_cast<int64_t>(colCursor) * srcStep[4];
            const int64_t dstOff =
                dstBase + static_cast<int64_t>(rowCursor) * dstStep[3] + static_cast<int64_t>(colCursor) * dstStep[4];
            const DynShape chunkShape(1, 1, 1, curRows, curCols);
            SrcViewT srcView(srcGlobalData.data() + srcOff, chunkShape, srcStride);
            DstViewT dstView(dstGlobalData.data() + dstOff, chunkShape, dstStride);
            TgetTransferOnce<TileData, DstViewT, SrcViewT>(dstView, srcView, stagingTileData);
            colCursor += tileValidCol;
        }
        rowCursor += tileValidRow;
    }
}

// 2D sliding chunked transfer with single buffer
template <typename GlobalDstData, typename GlobalSrcData, typename TileData>
PTO_INTERNAL void TgetChunkedSingle(GlobalDstData &dstGlobalData, GlobalSrcData &srcGlobalData,
                                    TileData &stagingTileData, int gShape0, int gShape1, int gShape2, int gShape3,
                                    int gShape4, int tileValidRow, int tileValidCol)
{
    using T = typename GlobalSrcData::RawDType;
    const int remoteStep[5] = {static_cast<int>(srcGlobalData.GetStride(GlobalTensorDim::DIM_0)),
                               static_cast<int>(srcGlobalData.GetStride(GlobalTensorDim::DIM_1)),
                               static_cast<int>(srcGlobalData.GetStride(GlobalTensorDim::DIM_2)),
                               static_cast<int>(srcGlobalData.GetStride(GlobalTensorDim::DIM_3)),
                               static_cast<int>(srcGlobalData.GetStride(GlobalTensorDim::DIM_4))};
    const int localStep[5] = {static_cast<int>(dstGlobalData.GetStride(GlobalTensorDim::DIM_0)),
                              static_cast<int>(dstGlobalData.GetStride(GlobalTensorDim::DIM_1)),
                              static_cast<int>(dstGlobalData.GetStride(GlobalTensorDim::DIM_2)),
                              static_cast<int>(dstGlobalData.GetStride(GlobalTensorDim::DIM_3)),
                              static_cast<int>(dstGlobalData.GetStride(GlobalTensorDim::DIM_4))};

    using DynShape = Shape<1, 1, 1, DYNAMIC, DYNAMIC>;
    using DynStride = Stride<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    using SrcViewT = GlobalTensor<T, DynShape, DynStride, GlobalSrcData::layout>;
    using DstViewT = GlobalTensor<T, DynShape, DynStride, GlobalDstData::layout>;
    DynStride localChunkStride(localStep[0], localStep[1], localStep[2], localStep[3], localStep[4]);
    DynStride remoteChunkStride(remoteStep[0], remoteStep[1], remoteStep[2], remoteStep[3], remoteStep[4]);

    for (int i0 = 0; i0 < gShape0; ++i0) {
        for (int i1 = 0; i1 < gShape1; ++i1) {
            for (int i2 = 0; i2 < gShape2; ++i2) {
                const int64_t remoteBase = static_cast<int64_t>(i0) * remoteStep[0] +
                                           static_cast<int64_t>(i1) * remoteStep[1] +
                                           static_cast<int64_t>(i2) * remoteStep[2];
                const int64_t localBase = static_cast<int64_t>(i0) * localStep[0] +
                                          static_cast<int64_t>(i1) * localStep[1] +
                                          static_cast<int64_t>(i2) * localStep[2];
                TgetChunked2DSlice<GlobalDstData, GlobalSrcData, TileData, SrcViewT, DstViewT, DynShape>(
                    dstGlobalData, srcGlobalData, stagingTileData, remoteBase, localBase, gShape3, gShape4,
                    tileValidRow, tileValidCol, remoteStep, localStep, remoteChunkStride, localChunkStride);
            }
        }
    }
}

// ============================================================================
// TGET_IMPL: Remote read operation implementation
//
// Data flow: srcGlobalData (remote GM) → stagingTileData (UB) → dstGlobalData (local GM)
//
// When the GlobalTensor exceeds the UB tile capacity in rows and/or columns,
// the transfer is automatically chunked via 2D sliding:
//   - Outer dimensions (DIM_0, DIM_1, DIM_2) are iterated explicitly.
//   - DIM_3 (rows) is split into tileValidRow-sized chunks.
//   - DIM_4 (cols) is split into tileValidCol-sized chunks.
//
// Constraints for chunked mode:
//   - If TileData has static ValidRow, shape3 must be divisible by ValidRow.
//     Use DYNAMIC ValidRow for partial row chunk support.
//   - If TileData has static ValidCol, shape4 must be divisible by ValidCol.
//     Use DYNAMIC ValidCol for partial column chunk support.
// ============================================================================

template <typename GlobalDstData, typename GlobalSrcData, typename TileData>
PTO_INTERNAL void TGET_IMPL(GlobalDstData &dstGlobalData, GlobalSrcData &srcGlobalData, TileData &stagingTileData)
{
    using T = typename GlobalSrcData::RawDType;

    static_assert(std::is_same_v<typename GlobalDstData::RawDType, T>, "TGET: src/dst element type mismatch");
    static_assert(std::is_same_v<T, typename TileData::DType>,
                  "TGET: TileData element type must match GlobalData element type");
    constexpr bool layoutMatched = (GlobalSrcData::layout == GlobalDstData::layout);
    static_assert(layoutMatched, "TGET: src/dst layout mismatch");

    const int remoteDims[5] = {static_cast<int>(srcGlobalData.GetShape(GlobalTensorDim::DIM_0)),
                               static_cast<int>(srcGlobalData.GetShape(GlobalTensorDim::DIM_1)),
                               static_cast<int>(srcGlobalData.GetShape(GlobalTensorDim::DIM_2)),
                               static_cast<int>(srcGlobalData.GetShape(GlobalTensorDim::DIM_3)),
                               static_cast<int>(srcGlobalData.GetShape(GlobalTensorDim::DIM_4))};

    const int64_t totalRemoteRows = static_cast<int64_t>(remoteDims[0]) * remoteDims[1] * remoteDims[2] * remoteDims[3];
    const int singleTileRows = stagingTileData.GetValidRow();
    const int singleTileCols = stagingTileData.GetValidCol();

    PTO_ASSERT(singleTileRows > 0, "TGET: tileValidRow must be greater than 0");
    PTO_ASSERT(singleTileCols > 0, "TGET: tileValidCol must be greater than 0");

    if (totalRemoteRows == 0 || remoteDims[4] == 0) {
        return;
    }

    // Simple path: data fits in UB tile in both dimensions
    if (totalRemoteRows <= singleTileRows && remoteDims[4] <= singleTileCols) {
        TgetTransferOnce<TileData, GlobalDstData, GlobalSrcData>(dstGlobalData, srcGlobalData, stagingTileData);
        return;
    }

    // 2D sliding chunked path
    PTO_ASSERT(singleTileRows > 0, "TGET: tile ValidRow must be greater than 0 for chunked transfer");
    PTO_ASSERT(singleTileCols > 0, "TGET: tile ValidCol must be greater than 0 for chunked transfer");

    constexpr bool isDynamicRow = (TileData::ValidRow == DYNAMIC);
    constexpr bool isDynamicCol = (TileData::ValidCol == DYNAMIC);

    if constexpr (!isDynamicRow) {
        PTO_ASSERT(remoteDims[3] % singleTileRows == 0,
                   "TGET chunked: shape3 must be divisible by tile ValidRow when ValidRow is static. "
                   "Use a Tile with DYNAMIC ValidRow for partial row chunk support.");
    }
    if constexpr (!isDynamicCol) {
        PTO_ASSERT(remoteDims[4] % singleTileCols == 0,
                   "TGET chunked: shape4 must be divisible by tile ValidCol when ValidCol is static. "
                   "Use a Tile with DYNAMIC ValidCol for partial column chunk support.");
    }

    TgetChunkedSingle<GlobalDstData, GlobalSrcData, TileData>(
        dstGlobalData, srcGlobalData, stagingTileData, remoteDims[0], remoteDims[1], remoteDims[2], remoteDims[3],
        remoteDims[4], singleTileRows, singleTileCols);
}

// ============================================================================
// TGET_IMPL (ping-pong): Remote read with double buffering
//
// Uses two staging tiles (pingTile, pongTile) to overlap TLOAD (MTE2) and
// TSTORE (MTE3) for adjacent chunks, effectively hiding one DMA transfer
// behind the other.
//
// Timeline without ping-pong:
//   [TLOAD chunk0] -> [TSTORE chunk0] -> [TLOAD chunk1] -> [TSTORE chunk1] -> ...
//
// Timeline with ping-pong (overlap TSTORE[i] with TLOAD[i+1]):
//   [TLOAD chunk0] -> [TSTORE chunk0 | TLOAD chunk1] -> [TSTORE chunk1 | TLOAD chunk2] -> ...
//
// Requirements:
//   - pingTile and pongTile must have the same type and dimensions.
//   - Uses EVENT_ID0 (pingTile) and EVENT_ID1 (pongTile) for synchronization.
// ============================================================================

template <typename GlobalDstData, typename GlobalSrcData, typename TileData>
PTO_INTERNAL void TGET_IMPL(GlobalDstData &dstGlobalData, GlobalSrcData &srcGlobalData, TileData &pingTile,
                            TileData &pongTile)
{
    using T = typename GlobalSrcData::RawDType;

    static_assert(std::is_same_v<T, typename GlobalDstData::RawDType>, "TGET: src/dst element type mismatch");
    static_assert(GlobalSrcData::layout == GlobalDstData::layout, "TGET: src/dst layout mismatch");
    static_assert(std::is_same_v<T, typename TileData::DType>,
                  "TGET: TileData element type must match GlobalData element type");

    const int remoteDim0 = srcGlobalData.GetShape(GlobalTensorDim::DIM_0);
    const int remoteDim1 = srcGlobalData.GetShape(GlobalTensorDim::DIM_1);
    const int remoteDim2 = srcGlobalData.GetShape(GlobalTensorDim::DIM_2);
    const int remoteDim3 = srcGlobalData.GetShape(GlobalTensorDim::DIM_3);
    const int remoteDim4 = srcGlobalData.GetShape(GlobalTensorDim::DIM_4);

    const int64_t totalRemoteRows = static_cast<int64_t>(remoteDim0) * remoteDim1 * remoteDim2 * remoteDim3;
    const int pingRows = pingTile.GetValidRow();
    const int pingCols = pingTile.GetValidCol();

    PTO_ASSERT(pingRows > 0, "TGET: tileValidRow must be greater than 0");
    PTO_ASSERT(pingCols > 0, "TGET: tileValidCol must be greater than 0");

    if (totalRemoteRows == 0 || remoteDim4 == 0) {
        return;
    }

    // ---- Simple path: single chunk, no ping-pong benefit ----
    if (totalRemoteRows <= pingRows && remoteDim4 <= pingCols) {
        TLOAD(pingTile, srcGlobalData);
        set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        TSTORE(dstGlobalData, pingTile);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        return;
    }

    // 2D sliding chunked path with ping-pong double buffering
    constexpr bool isDynamicRow = (TileData::ValidRow == DYNAMIC);
    constexpr bool isDynamicCol = (TileData::ValidCol == DYNAMIC);

    if constexpr (!isDynamicRow) {
        PTO_ASSERT(remoteDim3 % pingRows == 0,
                   "TGET chunked: shape3 must be divisible by tile ValidRow when ValidRow is static. "
                   "Use a Tile with DYNAMIC ValidRow for partial row chunk support.");
    }
    if constexpr (!isDynamicCol) {
        PTO_ASSERT(remoteDim4 % pingCols == 0,
                   "TGET chunked: shape4 must be divisible by tile ValidCol when ValidCol is static. "
                   "Use a Tile with DYNAMIC ValidCol for partial column chunk support.");
    }

    const int remotePitch0 = srcGlobalData.GetStride(GlobalTensorDim::DIM_0);
    const int remotePitch1 = srcGlobalData.GetStride(GlobalTensorDim::DIM_1);
    const int remotePitch2 = srcGlobalData.GetStride(GlobalTensorDim::DIM_2);
    const int remotePitch3 = srcGlobalData.GetStride(GlobalTensorDim::DIM_3);
    const int remotePitch4 = srcGlobalData.GetStride(GlobalTensorDim::DIM_4);
    const int localPitch0 = dstGlobalData.GetStride(GlobalTensorDim::DIM_0);
    const int localPitch1 = dstGlobalData.GetStride(GlobalTensorDim::DIM_1);
    const int localPitch2 = dstGlobalData.GetStride(GlobalTensorDim::DIM_2);
    const int localPitch3 = dstGlobalData.GetStride(GlobalTensorDim::DIM_3);
    const int localPitch4 = dstGlobalData.GetStride(GlobalTensorDim::DIM_4);

    using DynShape = Shape<1, 1, 1, DYNAMIC, DYNAMIC>;
    using DynStride = Stride<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    using SrcViewT = GlobalTensor<T, DynShape, DynStride, GlobalSrcData::layout>;
    using DstViewT = GlobalTensor<T, DynShape, DynStride, GlobalDstData::layout>;

    // Precompute strides (identical for all chunk views)
    DynStride srcChunkStride(remotePitch0, remotePitch1, remotePitch2, remotePitch3, remotePitch4);
    DynStride dstChunkStride(localPitch0, localPitch1, localPitch2, localPitch3, localPitch4);

    // Ping-pong state (same as TPUT_IMPL ping-pong)
    // See TPUT_IMPL comments for detailed pipeline analysis.
    bool usePing = true;
    bool hasPending = false;
    int64_t pendingDstOffset = 0;
    int pendingRows = 0;
    int pendingCols = 0;

    for (int outer0 = 0; outer0 < remoteDim0; ++outer0) {
        for (int outer1 = 0; outer1 < remoteDim1; ++outer1) {
            for (int outer2 = 0; outer2 < remoteDim2; ++outer2) {
                int64_t srcBaseOffset = static_cast<int64_t>(outer0) * remotePitch0 +
                                        static_cast<int64_t>(outer1) * remotePitch1 +
                                        static_cast<int64_t>(outer2) * remotePitch2;
                int64_t dstBaseOffset = static_cast<int64_t>(outer0) * localPitch0 +
                                        static_cast<int64_t>(outer1) * localPitch1 +
                                        static_cast<int64_t>(outer2) * localPitch2;

                for (int rowOffset = 0; rowOffset < remoteDim3; rowOffset += pingRows) {
                    int chunkRows = (rowOffset + pingRows <= remoteDim3) ? pingRows : (remoteDim3 - rowOffset);

                    for (int colOffset = 0; colOffset < remoteDim4; colOffset += pingCols) {
                        int chunkCols = (colOffset + pingCols <= remoteDim4) ? pingCols : (remoteDim4 - colOffset);

                        int64_t srcOffset = srcBaseOffset + static_cast<int64_t>(rowOffset) * remotePitch3 +
                                            static_cast<int64_t>(colOffset) * remotePitch4;
                        int64_t dstOffset = dstBaseOffset + static_cast<int64_t>(rowOffset) * localPitch3 +
                                            static_cast<int64_t>(colOffset) * localPitch4;

                        TileData &loadTile = usePing ? pingTile : pongTile;
                        event_t curEvent = usePing ? EVENT_ID0 : EVENT_ID1;

                        if constexpr (isDynamicRow)
                            loadTile.RowMaskInternal = chunkRows;
                        if constexpr (isDynamicCol)
                            loadTile.ColMaskInternal = chunkCols;

                        DynShape chunkShape(1, 1, 1, chunkRows, chunkCols);
                        SrcViewT srcView(srcGlobalData.data() + srcOffset, chunkShape, srcChunkStride);

                        if (hasPending) {
                            TileData &storeTile = usePing ? pongTile : pingTile;
                            event_t prevEvent = usePing ? EVENT_ID1 : EVENT_ID0;

                            // Wait for previous TLOAD to finish (data in storeTile is ready)
                            wait_flag(PIPE_MTE2, PIPE_MTE3, prevEvent);

                            DynShape pendShape(1, 1, 1, pendingRows, pendingCols);
                            DstViewT pendView(dstGlobalData.data() + pendingDstOffset, pendShape, dstChunkStride);

                            // Issue TSTORE + TLOAD concurrently (MTE3 and MTE2 in parallel)
                            TSTORE(pendView, storeTile);
                            TLOAD(loadTile, srcView);

                            set_flag(PIPE_MTE3, PIPE_MTE2, prevEvent);
                            set_flag(PIPE_MTE2, PIPE_MTE3, curEvent);

                            // Ensure storeTile's UB has been fully read by MTE3
                            wait_flag(PIPE_MTE3, PIPE_MTE2, prevEvent);
                        } else {
                            TLOAD(loadTile, srcView);
                            set_flag(PIPE_MTE2, PIPE_MTE3, curEvent);
                        }

                        pendingDstOffset = dstOffset;
                        pendingRows = chunkRows;
                        pendingCols = chunkCols;
                        hasPending = true;
                        usePing = !usePing;
                    }
                }
            }
        }
    }

    // Epilogue: drain the last pending TSTORE
    if (hasPending) {
        const bool finalChunkInPong = usePing;
        TileData &finalTile = finalChunkInPong ? pongTile : pingTile;
        const event_t finalEvent = finalChunkInPong ? EVENT_ID1 : EVENT_ID0;

        wait_flag(PIPE_MTE2, PIPE_MTE3, finalEvent);

        const DynShape finalChunkShape(1, 1, 1, pendingRows, pendingCols);
        DstViewT finalDstView(dstGlobalData.data() + pendingDstOffset, finalChunkShape, dstChunkStride);

        TSTORE(finalDstView, finalTile);
        set_flag(PIPE_MTE3, PIPE_MTE2, finalEvent);
        wait_flag(PIPE_MTE3, PIPE_MTE2, finalEvent);
    }
}

} // namespace comm
} // namespace pto

#endif // PTO_COMM_TGET_HPP
