/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef DISPATCH_MEGA_COMBINE_FRONT_REORDER_TILING_IMPL_HPP
#define DISPATCH_MEGA_COMBINE_FRONT_REORDER_TILING_IMPL_HPP

AICORE inline FrontCoreRowLoopView FrontGetCoreRowLoopView(const FrontReorderCommonState &op,
                                                           const FrontRowSplitCoreTiling &tiling, uint32_t perLoopRows,
                                                           uint32_t lastLoopRows, uint32_t explicitRowLoops = 0U)
{
    FrontCoreRowLoopView view;
    if (tiling.needCoreNum == 0U || op.coreIdx_ >= tiling.needCoreNum || tiling.perCoreRows == 0U) {
        return view;
    }
    view.active = true;
    const bool isLastCore = op.coreIdx_ == tiling.needCoreNum - 1U;
    view.coreRows = isLastCore ? tiling.lastCoreRows : tiling.perCoreRows;
    view.perLoopRows = perLoopRows;
    view.lastLoopRows = lastLoopRows;
    view.rowLoops = explicitRowLoops != 0U ?
                        explicitRowLoops :
                        (view.perLoopRows == 0U ? 0U : static_cast<uint32_t>(ceilDiv(view.coreRows, view.perLoopRows)));
    view.coreBase = op.coreIdx_ * tiling.perCoreRows;
    return view;
}

AICORE inline FrontCoreRowLoopView FrontGetCoreRowLoopView(const FrontReorderCommonState &op,
                                                           const FrontRowSplitLoopTiling &tiling)
{
    if (op.coreIdx_ >= tiling.needCoreNum) {
        return FrontCoreRowLoopView{};
    }
    const bool isLastCore = op.coreIdx_ == tiling.needCoreNum - 1U;
    return FrontGetCoreRowLoopView(op, tiling, isLastCore ? tiling.lastCorePerLoopRows : tiling.perCorePerLoopRows,
                                   isLastCore ? tiling.lastCoreLastLoopRows : tiling.perCoreLastLoopRows, 0U);
}

AICORE inline FrontCoreRowLoopView FrontGetCoreRowLoopView(const FrontReorderCommonState &op,
                                                           const FrontGatherQuantTiling &tiling)
{
    if (op.coreIdx_ >= tiling.needCoreNum) {
        return FrontCoreRowLoopView{};
    }
    const bool isLastCore = op.coreIdx_ == tiling.needCoreNum - 1U;
    return FrontGetCoreRowLoopView(op, tiling, isLastCore ? tiling.lastCorePerLoopRows : tiling.perCorePerLoopRows,
                                   isLastCore ? tiling.lastCoreLastLoopRows : tiling.perCoreLastLoopRows,
                                   isLastCore ? tiling.lastCoreLoops : tiling.perCoreLoops);
}

AICORE inline uint32_t FrontSrcToDstPerLoopMaxRows(uint32_t coreNum)
{
    const uint64_t reservedBytes = static_cast<uint64_t>(kFrontSrcToDstAssistElems) * sizeof(float) +
                                   static_cast<uint64_t>(coreNum) * kFrontSortAlignElement;
    if (AtlasA2::UB_SIZE <= reservedBytes) {
        return 0U;
    }
    return static_cast<uint32_t>((AtlasA2::UB_SIZE - reservedBytes) / (kFrontSortAlignElement * 2U) / 2U);
}

AICORE inline FrontSrcToDstTiling FrontBuildSrcToDstTiling(uint32_t routeElems, uint32_t coreNum)
{
    FrontSrcToDstTiling tiling;
    static_cast<FrontRowSplitCoreTiling &>(tiling) = FrontBuildRouteCoreTiling(routeElems, coreNum);
    if (tiling.perCoreRows == 0U) {
        return tiling;
    }
    const uint32_t perLoopMaxRows = FrontSrcToDstPerLoopMaxRows(coreNum);
    FrontApplyPerCoreRowLoopSplit(tiling, perLoopMaxRows);
    return tiling;
}

AICORE inline bool FrontGatherQuantPreferSingleColOneLoop(uint32_t problemK, uint64_t colSize, uint64_t scaleSize)
{
    return AtlasA2::UB_SIZE > colSize + scaleSize + 32U * 4U * 4U && problemK == kFrontFullLoadHLimit;
}

AICORE inline void FrontApplyGatherQuantOneLoop(FrontGatherQuantTiling &tiling, uint32_t perCoreRows,
                                                uint32_t onceRowSize, uint32_t problemK, bool forceSplit)
{
    const uint32_t perCoreOnceRows = forceSplit ? (onceRowSize < perCoreRows ? onceRowSize : perCoreRows) : perCoreRows;
    const uint32_t lastCoreOnceRows =
        forceSplit ? (onceRowSize < tiling.lastCoreRows ? onceRowSize : tiling.lastCoreRows) : tiling.lastCoreRows;
    tiling.perCorePerLoopRows = perCoreOnceRows;
    tiling.perCoreLastLoopRows = forceSplit ? FrontGetPerOrLastValue(perCoreRows, perCoreOnceRows) : perCoreRows;
    tiling.lastCorePerLoopRows = lastCoreOnceRows;
    tiling.lastCoreLastLoopRows =
        forceSplit ? FrontGetPerOrLastValue(tiling.lastCoreRows, lastCoreOnceRows) : tiling.lastCoreRows;
    tiling.perCoreLoops = forceSplit ? static_cast<uint32_t>(ceilDiv(perCoreRows, perCoreOnceRows)) : 1U;
    tiling.lastCoreLoops = forceSplit ? static_cast<uint32_t>(ceilDiv(tiling.lastCoreRows, lastCoreOnceRows)) : 1U;
    tiling.perLoopCols = problemK;
    tiling.lastLoopCols = problemK;
    tiling.colLoops = 1U;
}

AICORE inline void FrontApplyGatherQuantColumnSplit(FrontGatherQuantTiling &tiling, uint32_t perCoreRows,
                                                    uint32_t problemK, uint64_t rowSize, uint64_t colSize,
                                                    uint64_t scaleSize)
{
    uint32_t baseMaxCols = 6144U;
    const uint64_t totalColSize =
        AlignBytes<int8_t>(static_cast<uint64_t>(baseMaxCols) * sizeof(int8_t)) * kFrontDynamicQuantColsBuffer;
    uint32_t basePerLoopMaxRows =
        static_cast<uint32_t>(AlignBytes<int32_t>((AtlasA2::UB_SIZE - totalColSize - scaleSize) / sizeof(int32_t))) /
        4U;
    if (problemK < 6144U) {
        basePerLoopMaxRows =
            static_cast<uint32_t>(AlignBytes<int32_t>((AtlasA2::UB_SIZE - colSize - scaleSize) / sizeof(int32_t))) / 4U;
    } else if (perCoreRows < basePerLoopMaxRows) {
        baseMaxCols = static_cast<uint32_t>(AlignBytes<int32_t>(AtlasA2::UB_SIZE - rowSize - scaleSize)) /
                      kFrontDynamicQuantColsBuffer;
    }
    tiling.perLoopCols = baseMaxCols < problemK ? baseMaxCols : problemK;
    tiling.lastLoopCols = FrontGetPerOrLastValue(problemK, baseMaxCols);
    tiling.colLoops = baseMaxCols == 0U ? 0U : static_cast<uint32_t>(ceilDiv(problemK, baseMaxCols));
    tiling.perCorePerLoopRows = perCoreRows < basePerLoopMaxRows ? perCoreRows : basePerLoopMaxRows;
    tiling.perCoreLastLoopRows = FrontGetPerOrLastValue(perCoreRows, basePerLoopMaxRows);
    tiling.perCoreLoops =
        basePerLoopMaxRows == 0U ? 0U : static_cast<uint32_t>(ceilDiv(perCoreRows, basePerLoopMaxRows));
    tiling.lastCorePerLoopRows = tiling.lastCoreRows < basePerLoopMaxRows ? tiling.lastCoreRows : basePerLoopMaxRows;
    tiling.lastCoreLastLoopRows = FrontGetPerOrLastValue(tiling.lastCoreRows, basePerLoopMaxRows);
    tiling.lastCoreLoops =
        basePerLoopMaxRows == 0U ? 0U : static_cast<uint32_t>(ceilDiv(tiling.lastCoreRows, basePerLoopMaxRows));
}

AICORE inline FrontGatherQuantTiling FrontBuildGatherQuantTiling(uint32_t routeElems, uint32_t coreNum,
                                                                 uint32_t problemK)
{
    FrontGatherQuantTiling tiling;
    tiling.activateRows = routeElems;
    static_cast<FrontRowSplitCoreTiling &>(tiling) = FrontBuildRouteCoreTiling(routeElems, coreNum);
    if (tiling.perCoreRows == 0U) {
        return tiling;
    }

    const uint32_t perCoreRows = tiling.perCoreRows;
    const uint64_t rowSize = AlignBytes<int32_t>(static_cast<uint64_t>(perCoreRows) * sizeof(int32_t)) * 4U;
    const uint64_t colSize =
        AlignBytes<int8_t>(static_cast<uint64_t>(problemK) * sizeof(int8_t)) * kFrontDynamicQuantColsBuffer;
    const uint64_t scaleSize = kFrontDynamicQuantScaleBytes;
    uint32_t onceRowSize = 0U;
    if (AtlasA2::UB_SIZE > colSize + scaleSize + 32U * 4U * 3U) {
        onceRowSize =
            static_cast<uint32_t>((AtlasA2::UB_SIZE - colSize - scaleSize - 32U * 4U * 3U) / (sizeof(int32_t) * 4U));
        onceRowSize = onceRowSize / kFrontInt32OneBlockElems * kFrontInt32OneBlockElems;
    }
    const bool ifOneLoop = FrontGatherQuantPreferSingleColOneLoop(problemK, colSize, scaleSize);
    if (rowSize + colSize + scaleSize < AtlasA2::UB_SIZE || ifOneLoop) {
        FrontApplyGatherQuantOneLoop(tiling, perCoreRows, onceRowSize, problemK, ifOneLoop);
        return tiling;
    }

    FrontApplyGatherQuantColumnSplit(tiling, perCoreRows, problemK, rowSize, colSize, scaleSize);
    return tiling;
}

AICORE inline bool FrontRowLoopUbFits(const FrontCoreRowLoopView &coreLoop, uint64_t requiredUbBytes)
{
    return coreLoop.coreRows != 0U && coreLoop.perLoopRows != 0U && requiredUbBytes <= AtlasA2::UB_SIZE;
}

AICORE inline bool FrontGatherQuantCoreLoopReady(const FrontCoreRowLoopView &coreLoop, uint64_t requiredUbBytes)
{
    return FrontRowLoopUbFits(coreLoop, requiredUbBytes) && coreLoop.rowLoops != 0U;
}

AICORE inline bool FrontPostSortGatherQuantEnabled(const FrontGatherQuantTiling &tiling, uint32_t problemK)
{
    // Post-sort gather only implements single full-K quant tiles (colLoops == 1).
    return problemK <= kLargeFullRowMaxK && tiling.needCoreNum != 0U && tiling.colLoops == 1U;
}

AICORE inline bool FrontSrcToDstInputsValid(uint32_t routeElems, const FrontSrcToDstTiling &tiling)
{
    return routeElems != 0U && tiling.needCoreNum != 0U && tiling.perCoreRows != 0U;
}

AICORE inline bool FrontExpertTokenOutInputsValid(uint32_t routeElems, uint32_t expertNumUbAlign)
{
    return routeElems != 0U && expertNumUbAlign != 0U;
}

template <typename InputElement>
AICORE inline float FrontDynamicQuantRowToUb(__gm__ InputElement *xRow, uint32_t k, uint64_t rawUb, uint64_t fp32Ub,
                                             uint64_t tmpUb, uint64_t outUb, uint64_t scaleUb)
{
    PtoLoadVector<InputElement>(rawUb, xRow, k);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    if constexpr (std::is_same_v<InputElement, float>) {
        PtoMoveUb<float>(fp32Ub, rawUb, k);
    } else {
        CastDynamicQuantInputToFp32<InputElement>(fp32Ub, rawUb, k);
    }
    pipe_barrier(PIPE_V);

    BuildDynamicQuantAbs(tmpUb, fp32Ub, k);
    pipe_barrier(PIPE_V);
    ReduceDynamicQuantAbsMax(scaleUb, tmpUb, k);
    pipe_barrier(PIPE_V);
    set_flag(PIPE_V, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
    const float scaleValue = PtoGetValue<float>(scaleUb, 0U) / 127.0f;
    PtoFillUb<float>(scaleUb, scaleValue, 8U);
    PtoFillUb<float>(tmpUb, scaleValue, k);
    pipe_barrier(PIPE_V);

    DivideDynamicQuantInputByScale(tmpUb, fp32Ub, tmpUb, k);
    pipe_barrier(PIPE_V);
    PtoCastUb<half, float>(tmpUb, tmpUb, k, pto::RoundMode::CAST_TRUNC);
    pipe_barrier(PIPE_V);
    PtoCastUb<int8_t, half>(outUb, tmpUb, k, pto::RoundMode::CAST_ROUND);
    pto::PtoSetWaitFlag<PIPE_V, PIPE_MTE3>();
    return scaleValue;
}

template <typename InputElement>
AICORE inline void FrontQuantAndScatterPackedRows(const FrontReorderCommonState &op, uint32_t routeRowStart,
                                                  uint32_t routeRowCount, uint64_t dstRowIdxUb, uint32_t maxDstRow,
                                                  bool relativeDstIdx, uint64_t rawUb, uint64_t fp32Ub, uint64_t tmpUb,
                                                  uint64_t outUb, uint64_t scaleUb)
{
    if (routeRowCount == 0U || op.topK_ == 0U) {
        return;
    }

    uint32_t curRouteRow = routeRowStart;
    const uint32_t routeRowEnd = routeRowStart + routeRowCount - 1U;
    const uint32_t startTokenRow = routeRowStart / op.topK_;
    const uint32_t endTokenRow = routeRowEnd / op.topK_;

    for (uint32_t tokenRow = startTokenRow; tokenRow <= endTokenRow && tokenRow < op.problemM_; ++tokenRow) {
        (void)FrontDynamicQuantRowToUb<InputElement>(
            reinterpret_cast<__gm__ InputElement *>(op.xPtr_) + static_cast<uint64_t>(tokenRow) * op.problemK_,
            op.problemK_, rawUb, fp32Ub, tmpUb, outUb, scaleUb);

        bool rowStored = false;
        while (curRouteRow <= routeRowEnd && curRouteRow / op.topK_ == tokenRow) {
            const uint32_t ubIdx = relativeDstIdx ? (curRouteRow - routeRowStart) : curRouteRow;
            const int32_t dstRow = PtoGetValue<int32_t>(dstRowIdxUb, ubIdx);
            ++curRouteRow;
            if (dstRow < 0 || static_cast<uint32_t>(dstRow) >= maxDstRow) {
                continue;
            }
            PtoStoreVector<int8_t>(op.offsetAPtr_ + FrontPackedRowOffset(static_cast<uint32_t>(dstRow), op.problemK_),
                                   outUb, FrontPackedRowStride(op.problemK_));
            rowStored = true;
        }
        if (rowStored) {
            pto::PtoSetWaitFlag<PIPE_MTE3, PIPE_V>();
        }
    }
}

constexpr uint32_t kFrontEndMaxExpertNumAligned = 384U;

using FrontEndShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
using FrontEndStrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;

AICORE inline uint64_t FrontEndCountRowBytes(const FrontReorderCommonState &op)
{
    return AlignBytes<int32_t>(static_cast<uint64_t>(op.expertNumAligned_) * sizeof(int32_t));
}

AICORE inline uint64_t FrontEndCountRowUb()
{
    return 0U;
}

AICORE inline uint64_t FrontEndPrefixUb(const FrontReorderCommonState &op)
{
    return FrontEndCountRowUb() + FrontEndCountRowBytes(op);
}

AICORE inline uint64_t FrontEndTputUb(const FrontReorderCommonState &op)
{
    return FrontEndPrefixUb(op) + FrontEndCountRowBytes(op);
}

AICORE inline uint64_t FrontEndCumsumUb(const FrontReorderCommonState &op)
{
    return FrontEndTputUb(op) + FrontEndCountRowBytes(op);
}

template <uint32_t ExpertPerRank>
AICORE inline uint64_t FrontEndCountExchangeRequiredUbBytes(const FrontReorderCommonState &op)
{
    return FrontEndCumsumUb(op) + static_cast<uint64_t>(op.rankSize_) * ExpertPerRank * sizeof(int32_t);
}

AICORE inline __gm__ int32_t *FrontEndTokenPerExpertRow(const FrontReorderCommonState &op, uint32_t srcRank)
{
    return op.tokenPerExpertPtr_ + tokenPerExpertOffset(static_cast<int32_t>(srcRank), 0, 0,
                                                        static_cast<int32_t>(op.expertNumAligned_),
                                                        static_cast<int32_t>(op.expertPerRank_));
}

template <uint32_t ExpertPerRank>
AICORE inline bool FrontEndPostprocessEnabled(const FrontReorderCommonState &op)
{
    return op.rankSize_ != 0U && op.expertNumAligned_ != 0U && op.expertPerRank_ != 0U &&
           op.expertPerRank_ == ExpertPerRank && op.expertNumAligned_ <= kFrontEndMaxExpertNumAligned &&
           FrontEndCountExchangeRequiredUbBytes<ExpertPerRank>(op) <= AtlasA2::UB_SIZE;
}

/* 本 rank 把自己的整行 localTokenPerExpert[expertNumAligned]
写到每个 peer 的 tokenPerExpert[rank, :]
 */
AICORE inline void FrontEndPublishCountRowsToPeers(const FrontReorderCommonState &op)
{
    for (uint32_t dstRank = op.coreIdx_; dstRank < op.rankSize_; dstRank += op.coreNum_) {
        if (dstRank == op.rank_) {
            continue;
        }
        PtoLoadVector<int32_t>(FrontEndCountRowUb(), op.localTokenPerExpertPtr_, op.expertNumAligned_);
        pto::PtoSetWaitFlag<PIPE_MTE2, PIPE_V>();
        PtoAddScalarUb<int32_t>(FrontEndCountRowUb(), FrontEndCountRowUb(), op.expertNumAligned_,
                                kFrontCountMarkerBias);
        pipe_barrier(PIPE_V);
        pto::PtoSetWaitFlag<PIPE_V, PIPE_MTE3>();
        __gm__ int32_t *remoteRow =
            op.remoteWindow_.RemotePtr(FrontEndTokenPerExpertRow(op, op.rank_), static_cast<int32_t>(dstRank));
        PtoStoreVector<int32_t>(remoteRow, FrontEndCountRowUb(), op.expertNumAligned_);
        pto::PtoSetWaitFlag<PIPE_MTE3, PIPE_S>();
    }
}

AICORE inline void FrontEndWaitCountRowMarker(const FrontReorderCommonState &op, uint32_t srcRank)
{
    if (srcRank == op.rank_) {
        return;
    }

    using MarkerGlobal = pto::GlobalTensor<int32_t, FrontEndShapeDyn, FrontEndStrideDyn, pto::Layout::ND>;
    const uint32_t markerNum = static_cast<uint32_t>(ceilDiv(op.expertNumAligned_, kFrontCountMarkerStrideElems));
    FrontEndShapeDyn markerShape(1, 1, 1, 1, markerNum);
    FrontEndStrideDyn markerStride(kFrontCountMarkerStrideElems, kFrontCountMarkerStrideElems,
                                   kFrontCountMarkerStrideElems, kFrontCountMarkerStrideElems,
                                   kFrontCountMarkerStrideElems);
    MarkerGlobal marker(FrontEndTokenPerExpertRow(op, srcRank), markerShape, markerStride);
    pto::comm::TWAIT(marker, 0, pto::comm::WaitCmp::NE);
    V5DcciGmRange(static_cast<__gm__ void *>(FrontEndTokenPerExpertRow(op, srcRank)), FrontEndCountRowBytes(op));
}

AICORE inline void FrontEndRestoreCountRowAndBuildPreSum(const FrontReorderCommonState &op, uint32_t srcRank)
{
    __gm__ int32_t *srcRow = FrontEndTokenPerExpertRow(op, srcRank);
    FrontEndWaitCountRowMarker(op, srcRank);

    if (srcRank != op.rank_) {
        PtoLoadVector<int32_t>(FrontEndCountRowUb(), srcRow, op.expertNumAligned_);
        pto::PtoSetWaitFlag<PIPE_MTE2, PIPE_V>();
        PtoAddScalarUb<int32_t>(FrontEndCountRowUb(), FrontEndCountRowUb(), op.expertNumAligned_,
                                -kFrontCountMarkerBias);
        pipe_barrier(PIPE_V);
        pto::PtoSetWaitFlag<PIPE_V, PIPE_MTE3>();
        PtoStoreVector<int32_t>(srcRow, FrontEndCountRowUb(), op.expertNumAligned_);
        pto::PtoSetWaitFlag<PIPE_MTE3, PIPE_S>();
    } else {
        PtoLoadVector<int32_t>(FrontEndCountRowUb(), op.localTokenPerExpertPtr_, op.expertNumAligned_);
        pto::PtoSetWaitFlag<PIPE_MTE2, PIPE_MTE3>();
        PtoStoreVector<int32_t>(srcRow, FrontEndCountRowUb(), op.expertNumAligned_);
        pto::PtoSetWaitFlag<PIPE_MTE3, PIPE_S>();
    }
    PtoFillUb<int32_t>(FrontEndPrefixUb(op), 0, op.expertPerRank_);
    pipe_barrier(PIPE_V);
    pto::PtoSetWaitFlag<PIPE_V, PIPE_S>();

    int32_t prevSum = 0;
    uint32_t localExpert = 0U;
    const uint32_t localBegin = op.rank_ * op.expertPerRank_;
    const uint32_t prefixEnd = localBegin + op.expertPerRank_;
    for (uint32_t expert = 0U; expert < prefixEnd && expert < op.expertNumAligned_; ++expert) {
        if (expert >= localBegin && localExpert < op.expertPerRank_) {
            PtoSetValue<int32_t>(FrontEndPrefixUb(op), localExpert, prevSum);
            ++localExpert;
        }
        if (expert < op.expertNum_) {
            prevSum += PtoGetValue<int32_t>(FrontEndCountRowUb(), expert);
        }
    }
    pto::PtoSetWaitFlag<PIPE_S, PIPE_MTE3>();
    PtoStoreVector<int32_t>(op.preSumBeforeRankPtr_ + static_cast<uint64_t>(srcRank) * op.expertPerRank_,
                            FrontEndPrefixUb(op), op.expertPerRank_);
    pto::PtoSetWaitFlag<PIPE_MTE3, PIPE_S>();
}

/* 每个 rank 把自己的全局 expert 计数行发给所有 peer；
每个 rank 收到每个 srcRank 的计数行后，
对这行从 globalExpert=0 起求前缀，
取出“本 rank 的 local expert 在 srcRank.offsetA 里的起始行”，
写成 preSumBeforeRank[srcRank, localExpert]。
 */
AICORE inline void FrontEndBuildCountExchangeAndPreSum(const FrontReorderCommonState &op)
{
    FrontEndPublishCountRowsToPeers(op);

    for (uint32_t srcRank = op.coreIdx_; srcRank < op.rankSize_; srcRank += op.coreNum_) {
        FrontEndRestoreCountRowAndBuildPreSum(op, srcRank);
    }

    pto::SYNCALL<pto::SyncCoreType::AIVOnly>();
}

template <uint32_t Pitch>
AICORE inline void FrontEndBuildCumsumForPitch(const FrontReorderCommonState &op)
{
    static_assert(Pitch == 8U || Pitch == 16U, "front cumsum supports expertPerRank 8 or 16");
    using CumsumTile = pto::Tile<pto::TileType::Vec, int32_t, 16, Pitch, pto::BLayout::RowMajor, -1, -1>;
    using CumsumGlobal = pto::GlobalTensor<int32_t, FrontEndShapeDyn, FrontEndStrideDyn, pto::Layout::ND>;
    const uint32_t localBegin = op.rank_ * Pitch;
    CumsumTile cumsumTile(op.rankSize_, Pitch);
    pto::TASSIGN(cumsumTile, FrontEndCumsumUb(op));

    FrontEndShapeDyn loadShape(1, 1, 1, op.rankSize_, Pitch);
    FrontEndStrideDyn loadStride(op.rankSize_ * op.expertNumAligned_, op.rankSize_ * op.expertNumAligned_,
                                 op.rankSize_ * op.expertNumAligned_, op.expertNumAligned_, 1);
    CumsumGlobal srcGlobal(op.tokenPerExpertPtr_ + localBegin, loadShape, loadStride);
    pto::TLOAD(cumsumTile, srcGlobal);
    pto::PtoSetWaitFlag<PIPE_MTE2, PIPE_V>();

    for (uint32_t srcRank = 1U; srcRank < op.rankSize_; ++srcRank) {
        PtoAddUb<int32_t>(FrontEndCumsumUb(op) + static_cast<uint64_t>(srcRank) * Pitch * sizeof(int32_t),
                          FrontEndCumsumUb(op) + static_cast<uint64_t>(srcRank) * Pitch * sizeof(int32_t),
                          FrontEndCumsumUb(op) + static_cast<uint64_t>(srcRank - 1U) * Pitch * sizeof(int32_t), Pitch);
        pipe_barrier(PIPE_V);
    }

    pto::PtoSetWaitFlag<PIPE_V, PIPE_MTE3>();
    FrontEndShapeDyn storeShape(1, 1, 1, op.rankSize_, Pitch);
    FrontEndStrideDyn storeStride(op.rankSize_ * Pitch, op.rankSize_ * Pitch, op.rankSize_ * Pitch, Pitch, 1);
    CumsumGlobal dstGlobal(op.cumsumMMPtr_, storeShape, storeStride);
    pto::TSTORE(dstGlobal, cumsumTile);
    pto::PtoSetWaitFlag<PIPE_MTE3, PIPE_S>();
}

/* 输入：tokenPerExpert[srcRank, globalExpert]
输出：
  cumsumMM[srcRank, localExpert]
  expertTokenNums[localExpert]
   */
template <uint32_t ExpertPerRank>
AICORE inline void FrontEndBuildCumsumAndExpertTokenNums(const FrontReorderCommonState &op)
{
    static_assert(ExpertPerRank == 8U || ExpertPerRank == 16U, "front cumsum supports expertPerRank 8 or 16");
    if (op.coreIdx_ == 0U) {
        FrontEndBuildCumsumForPitch<ExpertPerRank>(op);
    }

    if (op.coreIdx_ == 0U) {
        PtoLoadVector<int32_t>(FrontEndPrefixUb(op),
                               op.cumsumMMPtr_ + static_cast<uint64_t>(op.rankSize_ - 1U) * ExpertPerRank,
                               ExpertPerRank);
        pto::PtoSetWaitFlag<PIPE_MTE2, PIPE_MTE3>();
        PtoStoreVector<int32_t>(op.expertTokenNumsPtr_, FrontEndPrefixUb(op), ExpertPerRank);
        pto::PtoSetWaitFlag<PIPE_MTE3, PIPE_S>();
    }
}

template <uint32_t ExpertPerRank>
AICORE inline void FrontFinalizeRankMetadata(const FrontReorderCommonState &op)
{
    if (!FrontEndPostprocessEnabled<ExpertPerRank>(op)) {
        return;
    }
    FrontEndBuildCountExchangeAndPreSum(op);

    FrontEndBuildCumsumAndExpertTokenNums<ExpertPerRank>(op);

    pto::SYNCALL<pto::SyncCoreType::AIVOnly>();
}

#endif // DISPATCH_MEGA_COMBINE_FRONT_REORDER_TILING_IMPL_HPP
