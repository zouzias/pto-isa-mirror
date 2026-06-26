/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef DISPATCH_MEGA_COMBINE_FRONT_REORDER_H
#define DISPATCH_MEGA_COMBINE_FRONT_REORDER_H

#include "kernel_operator.h"

#include <type_traits>

#include <pto/comm/pto_comm_inst.hpp>
#include <pto/pto-inst.hpp>

#include "dispatch_mega_combine_tiling.h"
#include "utils/common_helpers.hpp"
#include "utils/const_args.hpp"
#include "utils/hccl_window.hpp"
#include "utils/pto_sync_substrate.hpp"
#include "utils/pto_vector.hpp"

constexpr uint32_t kFrontSortBlockElems = 32;
constexpr uint32_t kFrontPackedSortBlockElems = 64;
constexpr uint32_t kFrontSortMaxElems = 8192;
constexpr float kFrontSortNegInf = -3.4028235e38F;

using FrontSortKeyTile = pto::Tile<pto::TileType::Vec, float, 1, kFrontSortMaxElems, pto::BLayout::RowMajor, -1, -1>;
using FrontSortPayloadTile =
    pto::Tile<pto::TileType::Vec, uint32_t, 1, kFrontSortMaxElems, pto::BLayout::RowMajor, -1, -1>;
using FrontPackedSortTile =
    pto::Tile<pto::TileType::Vec, float, 1, kFrontSortMaxElems * 2, pto::BLayout::RowMajor, -1, -1>;
using FrontPackedPayloadTile =
    pto::Tile<pto::TileType::Vec, uint32_t, 1, kFrontSortMaxElems * 2, pto::BLayout::RowMajor, -1, -1>;

AICORE inline uint32_t FrontAlignSortBlock(uint32_t elemNum)
{
    return ((elemNum + kFrontSortBlockElems - 1U) / kFrontSortBlockElems) * kFrontSortBlockElems;
}

AICORE inline int32_t FrontFillTailMergeArray(int32_t *mergePlan, int32_t validCols, int32_t blockLen)
{
    int32_t planCount = 0;
    int32_t remainCols = validCols;
    for (int32_t curBlockLen = blockLen; curBlockLen >= static_cast<int32_t>(kFrontPackedSortBlockElems);
         curBlockLen /= 4) {
        int32_t count = 0;
        for (; count < remainCols / curBlockLen; ++count) {
            mergePlan[planCount++] = curBlockLen;
        }
        remainCols -= count * curBlockLen;
    }
    return planCount;
}

AICORE inline void FrontMergeTailPackedSortRecords(FrontPackedSortTile &packedSortTile,
                                                   FrontPackedSortTile &mergeTmpTile, uint32_t validCols,
                                                   uint32_t blockLen)
{
    int32_t mergePlan[15] = {0};
    const int32_t mergePlanCount =
        FrontFillTailMergeArray(mergePlan, static_cast<int32_t>(validCols), static_cast<int32_t>(blockLen));
    if (mergePlanCount <= 1) {
        return;
    }

    pto::MrgSortExecutedNumList executedNumList{};
    uint16_t mergedCols = 0;
    const uint64_t packedAddr = reinterpret_cast<uint64_t>(packedSortTile.data());
    const uint64_t tmpAddr = reinterpret_cast<uint64_t>(mergeTmpTile.data());
    for (int32_t i = 0; i < mergePlanCount - 1; ++i) {
        mergedCols += static_cast<uint16_t>(mergePlan[i]);
        FrontPackedSortTile src0Tile(1, mergedCols);
        FrontPackedSortTile src1Tile(1, static_cast<uint16_t>(mergePlan[i + 1]));
        FrontPackedSortTile dstTile(1, mergedCols + static_cast<uint16_t>(mergePlan[i + 1]));
        FrontPackedSortTile tmpTile(1, mergedCols + static_cast<uint16_t>(mergePlan[i + 1]));
        pto::TASSIGN(src0Tile, packedAddr);
        pto::TASSIGN(src1Tile, packedAddr + static_cast<uint64_t>(mergedCols) * sizeof(float));
        pto::TASSIGN(dstTile, packedAddr);
        pto::TASSIGN(tmpTile, tmpAddr);
        pto::TMRGSORT<FrontPackedSortTile, FrontPackedSortTile, FrontPackedSortTile, FrontPackedSortTile, false>(
            dstTile, executedNumList, tmpTile, src0Tile, src1Tile);
        pipe_barrier(PIPE_V);
    }
}

AICORE inline void FrontMergePackedSortRecords(FrontPackedSortTile &packedSortTile, FrontPackedSortTile &mergeTmpTile,
                                               uint32_t validCols)
{
    uint32_t blockLen = kFrontPackedSortBlockElems; // 32个排序单元，每个单元是[key,payload] 2个元素，因此是64
    const uint64_t packedAddr = reinterpret_cast<uint64_t>(packedSortTile.data());
    const uint64_t tmpAddr = reinterpret_cast<uint64_t>(mergeTmpTile.data());
    for (; blockLen * 4U <= validCols; blockLen *= 4U) { /* 按照4批递归，128->512->.... */
        const uint16_t cols = static_cast<uint16_t>(validCols / (blockLen * 4U) * (blockLen * 4U));
        FrontPackedSortTile srcTile(1, cols);
        FrontPackedSortTile tmpTile(1, cols);
        pto::TASSIGN(srcTile, packedAddr);
        pto::TASSIGN(tmpTile, tmpAddr);
        pto::TMRGSORT(tmpTile, srcTile, blockLen); // 4路归并排序
        pipe_barrier(PIPE_V);
        pto::TMOV(srcTile, tmpTile);
        pipe_barrier(PIPE_V);
    }

    if (blockLen < validCols) {
        FrontPackedSortTile tailTile(1, validCols);
        FrontPackedSortTile tailTmpTile(1, validCols);
        pto::TASSIGN(tailTile, packedAddr);
        pto::TASSIGN(tailTmpTile, tmpAddr);
        FrontMergeTailPackedSortRecords(tailTile, tailTmpTile, validCols, blockLen); // 最后不足4路的归并排序
    }
}

AICORE inline void FrontBuildAscendingSortKey(uint64_t sortKeyUb, uint64_t inputValueUb, uint32_t elemNum)
{
    PtoCastUb<float, int32_t>(sortKeyUb, inputValueUb, elemNum, pto::RoundMode::CAST_ROUND);
    pipe_barrier(PIPE_V);
    PtoMulScalarUb<float>(sortKeyUb, sortKeyUb, elemNum, -1.0F);
    pipe_barrier(PIPE_V);
}

AICORE inline void FrontRestoreAscendingSortValue(uint64_t sortedValueUb, uint64_t sortedValueScratchUb,
                                                  uint32_t elemNum)
{
    PtoMulScalarUb<float>(sortedValueScratchUb, sortedValueScratchUb, elemNum, -1.0F);
    pipe_barrier(PIPE_V);
    PtoCastUb<int32_t, float>(sortedValueUb, sortedValueScratchUb, elemNum, pto::RoundMode::CAST_ROUND);
    pipe_barrier(PIPE_V);
}

/* 把一组 int32 expertId 和对应 payload routeIdx 在 UB 里排成 packed record，并按 expert 升序排好
inputValueUb    原始 expertId 数组，int32[elemNum]
inputPayloadUb  原始 routeIdx 数组，uint32[alignedElemNum]，通常是 0,1,2... 或 loopStart...
packedSortUb    输出 packed records 的 UB 地址
mergeTmpUb      TMRGSORT 临时 UB
sortKeyUb       float sort key 临时 UB, float[alignedElemNum]
elemNum         有效元素数
alignedElemNum  按 32 对齐后的排序元素数
 */
AICORE inline void FrontSortInt32ToPackedUb(uint64_t inputValueUb, uint64_t inputPayloadUb, uint64_t packedSortUb,
                                            uint64_t mergeTmpUb, uint64_t sortKeyUb, uint32_t elemNum,
                                            uint32_t alignedElemNum)
{
    if (elemNum == 0) {
        return;
    }
    if (alignedElemNum > kFrontSortMaxElems) {
        return;
    }
    if (alignedElemNum < FrontAlignSortBlock(elemNum)) {
        return;
    }

    // 把expertid 从int转float,再乘以-1，便于TSORT32最终按 expertId 升序排序
    FrontBuildAscendingSortKey(sortKeyUb, inputValueUb, elemNum);
    pto::PtoSetWaitFlag<PIPE_V, PIPE_S>();
    for (uint32_t i = elemNum; i < alignedElemNum; ++i) { // padding
        PtoSetValue<float>(sortKeyUb, i, kFrontSortNegInf);
        PtoSetValue<uint32_t>(inputPayloadUb, i, 0U);
    }
    if (alignedElemNum > elemNum) {
        pto::PtoSetWaitFlag<PIPE_S, PIPE_V>();
    }

    // 准备排序用的UB片上数据
    FrontSortKeyTile srcTile(1, alignedElemNum);
    FrontSortPayloadTile payloadTile(1, alignedElemNum);
    FrontPackedSortTile packedTile(1, alignedElemNum * 2U);
    FrontPackedSortTile mergeTmpTile(1, alignedElemNum * 2U);
    pto::TASSIGN(srcTile, sortKeyUb);
    pto::TASSIGN(payloadTile, inputPayloadUb);
    pto::TASSIGN(packedTile, packedSortUb);
    pto::TASSIGN(mergeTmpTile, mergeTmpUb);

    pto::TSORT32(packedTile, srcTile, payloadTile); // 完成按照32切分的排序，排序完毕后是每32个元素内部顺序排列
    pipe_barrier(PIPE_V);
    FrontMergePackedSortRecords(packedTile, mergeTmpTile, alignedElemNum * 2U); // 将所有的32元素block做归并排序
}

AICORE inline void FrontExtractPackedSortResult(uint64_t sortedValueUb, uint64_t sortedPayloadUb,
                                                uint64_t sortedValueScratchUb, uint64_t packedSortUb, uint32_t elemNum)
{
    if (elemNum == 0) {
        return;
    }

    FrontPackedPayloadTile packedPayloadTile(1, elemNum * 2U);
    FrontSortPayloadTile sortedPayloadTile(1, elemNum);
    pto::TASSIGN(packedPayloadTile, packedSortUb);
    pto::TASSIGN(sortedPayloadTile, sortedPayloadUb);
    pto::TGATHER<FrontSortPayloadTile, FrontPackedPayloadTile, pto::MaskPattern::P1010>(sortedPayloadTile,
                                                                                        packedPayloadTile);
    pipe_barrier(PIPE_V);

    FrontSortKeyTile sortedKeyTile(1, elemNum);
    FrontPackedSortTile packedTile(1, elemNum * 2U);
    pto::TASSIGN(sortedKeyTile, sortedValueScratchUb);
    pto::TASSIGN(packedTile, packedSortUb);
    pto::TGATHER<FrontSortKeyTile, FrontPackedSortTile, pto::MaskPattern::P0101>(sortedKeyTile, packedTile);
    pipe_barrier(PIPE_V);

    FrontRestoreAscendingSortValue(sortedValueUb, sortedValueScratchUb, elemNum);
}

constexpr uint32_t kLargeFullRowMaxK = 8192;

template <typename T>
AICORE inline uint64_t AlignBytes(uint64_t value)
{
    return (value + 31U) / 32U * 32U;
}

constexpr uint32_t kFrontMergeOutMaxFanIn = 4U;
constexpr uint32_t kFrontMetadataChunkElems = 4096U;
constexpr uint32_t kFrontCountMarkerStrideElems = 128U;
constexpr int32_t kFrontCountMarkerBias = 0x800000;
constexpr uint32_t kFrontSrcToDstAssistElems = 256U;
constexpr uint32_t kFrontSrcToDstAssistIndexElems = 32U;
constexpr uint32_t kFrontInt32OneBlockElems = 8U;
constexpr uint32_t kFrontSortAlignElement = 32U;
constexpr uint32_t kFrontMaxExpertNum = 5120U;
constexpr uint32_t kFrontDynamicQuantColsBuffer = 21U;
constexpr uint32_t kFrontDynamicQuantScaleBytes = 64U;
constexpr uint32_t kFrontFullLoadHLimit = 7168U;

struct FrontRowSplitCoreTiling {
    uint32_t needCoreNum = 0;
    uint32_t perCoreRows = 0;
    uint32_t lastCoreRows = 0;
};

struct FrontRowSplitLoopTiling : FrontRowSplitCoreTiling {
    uint32_t perCorePerLoopRows = 0;
    uint32_t perCoreLastLoopRows = 0;
    uint32_t lastCorePerLoopRows = 0;
    uint32_t lastCoreLastLoopRows = 0;
};

using FrontSrcToDstTiling = FrontRowSplitLoopTiling;

struct FrontGatherQuantTiling : FrontRowSplitLoopTiling {
    uint32_t activateRows = 0;
    uint32_t perCoreLoops = 0;
    uint32_t lastCoreLoops = 0;
    uint32_t perLoopCols = 0;
    uint32_t lastLoopCols = 0;
    uint32_t colLoops = 0;
};

AICORE inline uint32_t FrontGetPerOrLastValue(uint32_t value, uint32_t divisor)
{
    if (divisor == 0U) {
        return 0U;
    }
    return value <= divisor ? value : value % divisor;
}

AICORE inline FrontRowSplitCoreTiling FrontBuildRouteCoreTiling(uint32_t routeElems, uint32_t coreNum)
{
    FrontRowSplitCoreTiling tiling;
    if (routeElems == 0U || coreNum == 0U) {
        return tiling;
    }
    tiling.perCoreRows = static_cast<uint32_t>(ceilDiv(routeElems, coreNum));
    if (tiling.perCoreRows == 0U) {
        return tiling;
    }
    tiling.needCoreNum = static_cast<uint32_t>(ceilDiv(routeElems, tiling.perCoreRows));
    tiling.lastCoreRows = routeElems - tiling.perCoreRows * (tiling.needCoreNum - 1U);
    return tiling;
}

AICORE inline void FrontApplyPerCoreRowLoopSplit(FrontRowSplitLoopTiling &tiling, uint32_t perLoopMaxRows)
{
    if (tiling.perCoreRows == 0U || perLoopMaxRows == 0U) {
        tiling = FrontRowSplitLoopTiling{};
        return;
    }
    if (perLoopMaxRows >= tiling.perCoreRows) {
        tiling.perCorePerLoopRows = tiling.perCoreRows;
        tiling.perCoreLastLoopRows = tiling.perCoreRows;
    } else {
        tiling.perCorePerLoopRows = perLoopMaxRows;
        const uint32_t loops = static_cast<uint32_t>(ceilDiv(tiling.perCoreRows, perLoopMaxRows));
        tiling.perCoreLastLoopRows = tiling.perCoreRows - (loops - 1U) * perLoopMaxRows;
    }
    if (perLoopMaxRows >= tiling.lastCoreRows) {
        tiling.lastCorePerLoopRows = tiling.lastCoreRows;
        tiling.lastCoreLastLoopRows = tiling.lastCoreRows;
    } else {
        tiling.lastCorePerLoopRows = perLoopMaxRows;
        const uint32_t loops = static_cast<uint32_t>(ceilDiv(tiling.lastCoreRows, perLoopMaxRows));
        tiling.lastCoreLastLoopRows = tiling.lastCoreRows - (loops - 1U) * perLoopMaxRows;
    }
}

struct FrontCoreRowLoopView {
    bool active = false;
    uint32_t coreRows = 0;
    uint32_t perLoopRows = 0;
    uint32_t lastLoopRows = 0;
    uint32_t rowLoops = 0;
    uint32_t coreBase = 0;
};

struct FrontUbStack {
    uint64_t cursor = 0;

    AICORE inline uint64_t Push(uint64_t bytes)
    {
        const uint64_t base = cursor;
        cursor += bytes;
        return base;
    }

    AICORE inline uint64_t End() const
    {
        return cursor;
    }
};

AICORE inline uint32_t FrontExpertTokenOutExpertNumUbAlign(uint32_t expertNum)
{
    const uint32_t alignedExpertNum = static_cast<uint32_t>(alignUp(expertNum, kFrontInt32OneBlockElems));
    return alignedExpertNum > kFrontMaxExpertNum ? kFrontMaxExpertNum : alignedExpertNum;
}

struct FrontExpertTokenOutUbPlan {
    uint64_t countUb = 0;
    uint64_t chunkUb = 0;
    uint64_t scratchUb = 0;
    uint64_t requiredBytes = 0;

    AICORE inline explicit FrontExpertTokenOutUbPlan(uint32_t expertNum, uint32_t perLoopRows = 0U)
    {
        const uint32_t expertAlign = FrontExpertTokenOutExpertNumUbAlign(expertNum);
        const uint64_t countBytes = AlignBytes<int32_t>(static_cast<uint64_t>(expertAlign) * sizeof(int32_t));
        const uint64_t chunkBytes =
            AlignBytes<int32_t>(static_cast<uint64_t>(kFrontMetadataChunkElems) * sizeof(int32_t));
        countUb = 0U;
        chunkUb = countBytes;
        requiredBytes = chunkUb + chunkBytes;
        if (perLoopRows == 0U) {
            return;
        }
        const uint64_t inputBytes = AlignBytes<int32_t>(static_cast<uint64_t>(perLoopRows) * sizeof(int32_t));
        const uint64_t scratchBytes =
            AlignBytes<int32_t>(static_cast<uint64_t>(expertAlign + kFrontInt32OneBlockElems) * sizeof(int32_t));
        const uint64_t inputEnd = chunkUb + inputBytes;
        scratchUb = inputEnd;
        if (inputEnd > requiredBytes) {
            requiredBytes = inputEnd;
        }
        const uint64_t scratchEnd = inputEnd + scratchBytes;
        if (scratchEnd > requiredBytes) {
            requiredBytes = scratchEnd;
        }
    }
};

struct FrontSrcToDstUbPlan {
    uint64_t inputUb = 0;
    uint64_t outputUb = 0;
    uint64_t assistUb = 0;
    uint64_t requiredBytes = 0;

    AICORE inline explicit FrontSrcToDstUbPlan(uint32_t perLoopRows)
    {
        const uint64_t inputBytes = AlignBytes<int32_t>(static_cast<uint64_t>(perLoopRows) * sizeof(int32_t));
        const uint64_t assistGroups =
            (static_cast<uint64_t>(perLoopRows) + kFrontSrcToDstAssistElems - 1U) / kFrontSrcToDstAssistElems;
        const uint64_t outputBytes =
            AlignBytes<int32_t>(assistGroups * kFrontSrcToDstAssistElems * kFrontInt32OneBlockElems * sizeof(int32_t));
        const uint64_t assistBytes =
            AlignBytes<int32_t>(static_cast<uint64_t>(kFrontSrcToDstAssistElems) * sizeof(int32_t));
        FrontUbStack stack;
        inputUb = stack.Push(inputBytes);
        outputUb = stack.Push(outputBytes);
        assistUb = stack.Push(assistBytes);
        requiredBytes = stack.End();
    }
};

template <typename InputElement>
struct FrontGatherQuantUbPlan {
    uint64_t rawUb = 0;
    uint64_t fp32Ub = 0;
    uint64_t tmpUb = 0;
    uint64_t outUb = 0;
    uint64_t scaleUb = 0;
    uint64_t indexUb = 0;
    uint64_t requiredBytes = 0;

    AICORE inline FrontGatherQuantUbPlan(uint32_t problemK, uint32_t perLoopRows)
    {
        FrontUbStack stack;
        rawUb = stack.Push(AlignBytes<InputElement>(static_cast<uint64_t>(problemK) * sizeof(InputElement)));
        fp32Ub = stack.Push(AlignBytes<float>(static_cast<uint64_t>(problemK) * sizeof(float)));
        tmpUb = stack.Push(AlignBytes<float>(static_cast<uint64_t>(problemK) * sizeof(float)));
        outUb = stack.Push(AlignBytes<int8_t>(static_cast<uint64_t>(problemK) * sizeof(int8_t)));
        scaleUb = stack.Push(AlignBytes<float>(8U * sizeof(float)));
        indexUb = stack.Push(AlignBytes<int32_t>(static_cast<uint64_t>(perLoopRows) * sizeof(int32_t)));
        requiredBytes = stack.End();
    }
};

template <typename T>
AICORE inline uint32_t FrontPtoGetSortLen(uint32_t elemCount)
{
    static_assert(std::is_same_v<T, float>, "front sort packed records use float slots");
    return elemCount * 2U;
}

template <typename T>
AICORE inline uint32_t FrontPtoGetSortOffset(uint32_t elemOffset)
{
    static_assert(std::is_same_v<T, float>, "front sort packed records use float slots");
    return elemOffset * 2U;
}

AICORE inline uint32_t FrontPackedRowStride(uint32_t k)
{
    return k + static_cast<uint32_t>(UB_ALIGN);
}

AICORE inline uint64_t FrontPackedRowOffset(uint32_t row, uint32_t k)
{
    return static_cast<uint64_t>(row) * FrontPackedRowStride(k);
}

AICORE inline void FrontMergePackedSortRecordsChecked(uint64_t dstUb, uint64_t tmpUb, uint64_t src0Ub, uint64_t src1Ub,
                                                      uint64_t src2Ub, uint64_t src3Ub,
                                                      const uint16_t *elementCountList, uint32_t remainListNum,
                                                      uint32_t *listSortedNums)
{
    const uint32_t src0Cols = FrontPtoGetSortLen<float>(elementCountList[0]);
    const uint32_t src1Cols = (remainListNum >= 2U) ? FrontPtoGetSortLen<float>(elementCountList[1]) : 0U;
    const uint32_t src2Cols = (remainListNum >= 3U) ? FrontPtoGetSortLen<float>(elementCountList[2]) : 0U;
    const uint32_t src3Cols = (remainListNum >= 4U) ? FrontPtoGetSortLen<float>(elementCountList[3]) : 0U;
    const uint32_t dstCols = src0Cols + src1Cols + src2Cols + src3Cols;
    if (dstCols > kFrontSortMaxElems * 2U) {
        return;
    }

    FrontPackedSortTile dstTile(1, dstCols);
    FrontPackedSortTile tmpTile(1, dstCols);
    FrontPackedSortTile src0Tile(1, src0Cols);
    FrontPackedSortTile src1Tile(1, src1Cols);
    FrontPackedSortTile src2Tile(1, src2Cols);
    FrontPackedSortTile src3Tile(1, src3Cols);
    pto::TASSIGN(dstTile, dstUb);
    pto::TASSIGN(tmpTile, tmpUb);
    pto::TASSIGN(src0Tile, src0Ub);
    pto::TASSIGN(src1Tile, src1Ub);
    if (src2Cols > 0U) {
        pto::TASSIGN(src2Tile, src2Ub);
    }
    if (src3Cols > 0U) {
        pto::TASSIGN(src3Tile, src3Ub);
    }

    pto::MrgSortExecutedNumList executedNumList{};
    if (remainListNum == 2U) {
        pto::TMRGSORT<FrontPackedSortTile, FrontPackedSortTile, FrontPackedSortTile, FrontPackedSortTile, true>(
            dstTile, executedNumList, tmpTile, src0Tile, src1Tile);
    } else if (remainListNum == 3U) {
        pto::TMRGSORT<FrontPackedSortTile, FrontPackedSortTile, FrontPackedSortTile, FrontPackedSortTile,
                      FrontPackedSortTile, true>(dstTile, executedNumList, tmpTile, src0Tile, src1Tile, src2Tile);
    } else {
        pto::TMRGSORT<FrontPackedSortTile, FrontPackedSortTile, FrontPackedSortTile, FrontPackedSortTile,
                      FrontPackedSortTile, FrontPackedSortTile, true>(dstTile, executedNumList, tmpTile, src0Tile,
                                                                      src1Tile, src2Tile, src3Tile);
    }
    pipe_barrier(PIPE_V);

    listSortedNums[0] = executedNumList.mrgSortList0;
    listSortedNums[1] = executedNumList.mrgSortList1;
    listSortedNums[2] = executedNumList.mrgSortList2;
    listSortedNums[3] = executedNumList.mrgSortList3;
}

struct FrontReorderCommonState {
    AICORE inline void InitCommonInputs(GM_ADDR xGM, GM_ADDR expertIdGM, GM_ADDR expertTokenNumsGM, GM_ADDR workspaceGM,
                                        const __gm__ MegaMoeTilingData *tilingData)
    {
        xPtr_ = xGM;
        expertIdPtr_ = reinterpret_cast<__gm__ int32_t *>(expertIdGM);
        expertTokenNumsPtr_ = reinterpret_cast<__gm__ int32_t *>(expertTokenNumsGM);
        workspaceGM_ = workspaceGM;
        tilingData_ = tilingData;

        const auto &info = tilingData_->megaMoeInfo;
        problemM_ = info.M;
        problemK_ = info.K;
        topK_ = info.topK;
        expertPerRank_ = info.expertPerRank;
        rank_ = tilingData_->runtimeInfo.rank;
        rankSize_ = tilingData_->runtimeInfo.rankSize;
    }

    AICORE inline void InitCommonCoreIdx()
    {
        coreIdx_ = get_block_idx();
        coreNum_ = get_block_num();
        if ASCEND_IS_AIV {
            coreIdx_ = get_block_idx() + get_subblockid() * get_block_num();
            coreNum_ = get_block_num() * get_subblockdim();
        }
    }

    AICORE inline void InitCommonPeerWindow()
    {
        remoteWindow_.Init(reinterpret_cast<GM_ADDR>(tilingData_->runtimeInfo.remoteWindowContext));
        peerMemoryLayout_.Init(remoteWindow_);
        offsetAPtr_ = reinterpret_cast<__gm__ int8_t *>(remoteWindow_.LocalBase() + peerMemoryLayout_.offsetA);
        tokenPerExpertPtr_ =
            reinterpret_cast<__gm__ int32_t *>(remoteWindow_.LocalBase() + peerMemoryLayout_.offsetPeerTokenPerExpert);
    }

    AICORE inline void InitCommonWorkspacePtrs(const __gm__ MegaMoeFrontReorderTiling &front, bool extended)
    {
        expandedRowIdxPtr_ = reinterpret_cast<__gm__ int32_t *>(workspaceGM_ + front.expandedRowIdxOffset);
        localTokenPerExpertPtr_ = reinterpret_cast<__gm__ int32_t *>(workspaceGM_ + front.localTokenPerExpertOffset);
        cumsumMMPtr_ = reinterpret_cast<__gm__ int32_t *>(workspaceGM_ + front.cumsumMMOffset);
        preSumBeforeRankPtr_ = reinterpret_cast<__gm__ int32_t *>(workspaceGM_ + front.preSumBeforeRankOffset);
        if (extended) {
            frontExpandedExpertPtr_ =
                reinterpret_cast<__gm__ int32_t *>(workspaceGM_ + front.frontExpandedExpertOffset);
            frontExpandDstToSrcPtr_ =
                reinterpret_cast<__gm__ int32_t *>(workspaceGM_ + front.frontExpandDstToSrcOffset);
        }
    }

    AICORE inline void InitCommonSortTiling(const __gm__ MegaMoeFrontReorderTiling &front)
    {
        expertNum_ = front.expertNum;
        expertNumAligned_ = front.expertNumAligned;
        routeElems_ = front.routeElems;
        alignedRouteElems_ = front.alignedRouteElems;
        sortLoopMaxElement_ = front.sortLoopMaxElement;
        sortNeedCoreNum_ = front.sortNeedCoreNum;
        sortPerCoreElems_ = front.sortPerCoreElems;
        sortLastCoreElems_ = front.sortLastCoreElems;
        sortPerCoreLoops_ = front.sortPerCoreLoops;
        sortPerCorePerLoopElems_ = front.sortPerCorePerLoopElems;
        sortPerCoreLastLoopElems_ = front.sortPerCoreLastLoopElems;
        sortLastCoreLoops_ = front.sortLastCoreLoops;
        sortLastCorePerLoopElems_ = front.sortLastCorePerLoopElems;
        sortLastCoreLastLoopElems_ = front.sortLastCoreLastLoopElems;
        sortOutLoopMaxElems_ = front.sortOutLoopMaxElems;
    }

    AICORE inline void InitMinimalSortTiling(const __gm__ MegaMoeFrontReorderTiling &front)
    {
        expertNum_ = front.expertNum;
        expertNumAligned_ = front.expertNumAligned;
        routeElems_ = front.routeElems;
        sortLoopMaxElement_ = front.sortLoopMaxElement;
        sortLastCorePerLoopElems_ = front.sortLastCorePerLoopElems;
    }

    GM_ADDR xPtr_ = nullptr;
    __gm__ int32_t *expertIdPtr_ = nullptr;
    __gm__ int32_t *expertTokenNumsPtr_ = nullptr;
    __gm__ int32_t *expandedRowIdxPtr_ = nullptr;
    __gm__ int32_t *frontExpandedExpertPtr_ = nullptr;
    __gm__ int32_t *frontExpandDstToSrcPtr_ = nullptr;
    __gm__ int32_t *localTokenPerExpertPtr_ = nullptr;
    __gm__ int32_t *tokenPerExpertPtr_ = nullptr;
    __gm__ int32_t *cumsumMMPtr_ = nullptr;
    __gm__ int32_t *preSumBeforeRankPtr_ = nullptr;
    __gm__ int8_t *offsetAPtr_ = nullptr;
    GM_ADDR workspaceGM_ = nullptr;
    const __gm__ MegaMoeTilingData *tilingData_ = nullptr;
    PtoRemoteWindow remoteWindow_;
    MegaMoePeerMemoryLayout peerMemoryLayout_;

    uint32_t problemM_ = 0;
    uint32_t problemK_ = 0;
    uint32_t topK_ = 0;
    uint32_t expertPerRank_ = 0;
    uint32_t rank_ = 0;
    uint32_t rankSize_ = 0;
    uint32_t expertNum_ = 0;
    uint32_t expertNumAligned_ = 0;
    uint32_t routeElems_ = 0;
    uint32_t alignedRouteElems_ = 0;
    uint32_t sortLoopMaxElement_ = 0;
    uint32_t sortNeedCoreNum_ = 0;
    uint32_t sortPerCoreElems_ = 0;
    uint32_t sortLastCoreElems_ = 0;
    uint32_t sortPerCoreLoops_ = 0;
    uint32_t sortPerCorePerLoopElems_ = 0;
    uint32_t sortPerCoreLastLoopElems_ = 0;
    uint32_t sortLastCoreLoops_ = 0;
    uint32_t sortLastCorePerLoopElems_ = 0;
    uint32_t sortLastCoreLastLoopElems_ = 0;
    uint32_t sortOutLoopMaxElems_ = 0;
    uint32_t coreIdx_ = 0;
    uint32_t coreNum_ = 0;
};

class FrontReorderPathBase {
public:
    AICORE inline explicit FrontReorderPathBase(FrontReorderCommonState &op) : op_(op)
    {}

    AICORE inline FrontReorderCommonState &common()
    {
        return op_;
    }
    AICORE inline const FrontReorderCommonState &common() const
    {
        return op_;
    }

protected:
    FrontReorderCommonState &op_;
};

#include "front_reorder_tiling_impl.hpp"

#include "front_reorder_postsort_impl.hpp"

template <typename InputElement, uint32_t ExpertPerRank>
AICORE inline void FrontRunPostSortPipeline(const FrontReorderCommonState &op)
{
    FrontBuildExpertTokenOut(op); // 生成 localTokenPerExpert[global_expert] = count

    FrontBuildSrcToDst(op); // 生成 expandedRowIdx[srcRoute] = dstRow

    FrontRunGatherQuantLoop<InputElement>(op);

    FrontFinalizeRankMetadata<ExpertPerRank>(op);
}

#endif // DISPATCH_MEGA_COMBINE_FRONT_REORDER_H
