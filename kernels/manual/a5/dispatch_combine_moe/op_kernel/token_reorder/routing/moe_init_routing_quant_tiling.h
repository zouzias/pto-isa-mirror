#pragma once
#ifndef ASCENDC_DISPATCH_FFN_COMBINE_INIT_ROUTING_QUANT_TILING_H
#define ASCENDC_DISPATCH_FFN_COMBINE_INIT_ROUTING_QUANT_TILING_H

#include <algorithm>

#include "moe_init_routing_tiling_common.h"

namespace optiling {

const static int64_t ATTR_QUANT_MODE = 6;
const static int64_t TILING_KEY_BASE = 10000;
const static int64_t TILING_KEY_PERF_BASE = 20000;
const static int64_t TILING_KEY_QUANT_BASE = 1000;
const static int64_t TILING_KEY_DROP_MODE_BASE = 100;
const static int64_t TILING_KEY_SORT_BASE = 10;
const static int64_t FOUR_BLOCK_BYTE = 128;
const static int64_t MAX_COLS_ONE_LOOP_QUANT = 8192;
const static int64_t INDEX_SCALE = 2;
const static int64_t INDEX_OFFSET = 3;
const static int64_t SMOOTH_NONE = 0;
const static int64_t SMOOTH_1H = 1;
const static int64_t SMOOTH_EH = 2;
const static int64_t MAX_COLS_DYNAMIC_QUANT = 6144;
const static int64_t DYNAMIC_QUANT_SRC_TO_DST_BUFFER = 15;
const static int64_t DYNAMIC_QUANT_COLS_BUFFER = 21;
const static int64_t DYNAMIC_QUANT_FULLLOAD_COLS_BUFFER = 13;
const static int64_t DYNAMIC_QUANT_SCALE_SIZE_64 = 64;
const static int64_t DYNAMIC_QUANT_SCALE_SIZE_128 = 128;
const static int64_t OUTOUT_DYNAMIC_QUANT_SCALE = 4;
const static int64_t FULLLOAD_H_LIMIT = 7168;

struct MoeInitRoutingQuantTilingData {
    int64_t coreNum;
    int64_t n;
    int64_t cols;
    int64_t k;
    int64_t expertCapacity;
    int64_t expertNum;
    int64_t dropPadMode;
    int64_t expertTokensCountOrCumsumFlag;
    int64_t expertTokensBeforeCapacityFlag;
    int64_t smoothType;
    InnerMoeVBSComputeTilingData vbsComputeParamsOp;
    InnerMoeVMSMiddleComputeTilingData vmsMiddleComputeParamsOp;
    InnerMoeSortOutComputeTilingData sortOutComputeParamsOp;
    InnerMoeGatherOutComputeTilingData srcToDstComputeParamsOp;
    InnerMoeGatherOutComputeTilingData srcToDstCapacityComputeParamsOp;
    InnerMoeGatherOutComputeTilingData gatherOutComputeParamsOp;
};

class MoeInitRoutingQuantTilingBase : public InnerMoeInitRoutingTilingBase {
public:
protected:
    bool GetShapeAttrsInfo(int64_t m, int64_t cols, int64_t topK, int64_t expertCapacity, int64_t expertNum,
                           int64_t activeNum, int64_t dropPadMode, int64_t expertTokensCountOrCumsumFlag,
                           bool expertTokensBeforeCapacityFlag, int64_t inuptXDtypeSize, int64_t quantMode,
                           int64_t scaleDim0) override;
    uint64_t GetTilingKey() const override;
    bool GetWorkspaceSize() override;
    bool PostTiling() override;

public:
    // bool CheckOutShape() override;
    bool IsFullLoadQuant(int64_t space);
    bool IsFullLoadDynamicQuant(int64_t space);
    bool IsFullLoad() override;
    void SetGatherTilingData(InnerMoeGatherOutComputeTilingData *tilingData, int64_t perCoreRows, int64_t lastCoreRows,
                             int64_t cols);
    void SetGatherTilingDataCols(InnerMoeGatherOutComputeTilingData *tilingData, int64_t baseMaxCols, int64_t cols);
    void SetGatherTilingDataRows(InnerMoeGatherOutComputeTilingData *tilingData, int64_t perCoreRows,
                                 int64_t lastCoreRows, int64_t basePerLoopMaxRows);
    void Tiling4GatherQuant();
    void Tiling4GatherDynamicQuant();
    void Tiling4SrcToDstCapacityCompute() override;
    void Tiling4GatherOutCompute() override;
    void CopyGatherOutTiling(InnerMoeGatherOutComputeTilingData &dst, InnerMoeGatherOutComputeTilingData &src);
    void CopyTilingData();

    int64_t quantMode;
    MoeInitRoutingQuantTilingData quantTilingData;
};

inline bool MoeInitRoutingQuantTilingBase::IsFullLoadQuant(int64_t space)
{
    int64_t perCoreXRows = moeInitRoutingTilingData.n / aivNum;
    int64_t remainder = moeInitRoutingTilingData.n % aivNum;
    // NUM_TWO is Max xRows need add 2 becauseof the left and right row may be another row.
    perCoreXRows = remainder <= 1 ? perCoreXRows + 1 : perCoreXRows + NUM_TWO;
    int64_t quantBaseSpace = CeilDiv(moeInitRoutingTilingData.cols, ONE_BLOCK_BYTE) * ONE_BLOCK_BYTE;
    int64_t quantSpace =
        quantBaseSpace * (inuptXDtypeSize_ + sizeof(int8_t) + sizeof(float) + sizeof(int16_t)) * perCoreXRows;
    int64_t remainUbAfterSort = aicoreParams_.ubSize - space - quantSpace;
    return remainUbAfterSort > 0;
}

inline bool MoeInitRoutingQuantTilingBase::IsFullLoadDynamicQuant(int64_t space)
{
    int64_t quantSpace =
        CeilDiv(moeInitRoutingTilingData.cols, ONE_BLOCK_BYTE) * ONE_BLOCK_BYTE * DYNAMIC_QUANT_FULLLOAD_COLS_BUFFER;
    int64_t scaleOutSpace = 64;
    int64_t remainUbAfterSort = aicoreParams_.ubSize - space - scaleOutSpace - quantSpace;
    return remainUbAfterSort > 0;
}

inline bool MoeInitRoutingQuantTilingBase::IsFullLoad()
{
    if (totalLength > sortLoopMaxElement || moeInitRoutingTilingData.cols > MAX_COLS_ONE_LOOP_QUANT ||
        this->dropPadMode == 1) {
        return false;
    }
    int64_t sortSpace =
        CeilDiv(this->totalLength, ONE_BLOCK_BYTE) * ONE_BLOCK_BYTE * sizeof(int32_t) * ONE_CORE_SORT_BUFFER;
    int64_t otherSpace = CeilDiv(this->totalLength, ONE_BLOCK_BYTE) * ONE_BLOCK_BYTE * sizeof(int32_t) * NUM_THREE;
    int64_t expertSpace =
        CeilDiv(this->expertNum * static_cast<int64_t>(sizeof(int32_t)), ONE_BLOCK_BYTE) * ONE_BLOCK_BYTE;
    if (quantMode == 0) {
        return IsFullLoadQuant(sortSpace + otherSpace + expertSpace);
    } else {
        return IsFullLoadDynamicQuant(sortSpace + otherSpace + expertSpace);
    }
}

inline bool MoeInitRoutingQuantTilingBase::GetShapeAttrsInfo(
    int64_t m, int64_t cols, int64_t topK, int64_t expertCapacity, int64_t expertNum, int64_t activeNum,
    int64_t dropPadMode, int64_t expertTokensCountOrCumsumFlag, bool expertTokensBeforeCapacityFlag,
    int64_t inuptXDtypeSize, int64_t quantMode, int64_t scaleDim0)
{
    InnerMoeInitRoutingTilingBase::GetShapeAttrsInfo(m, cols, topK, expertCapacity, expertNum, activeNum, dropPadMode,
                                                     expertTokensCountOrCumsumFlag, expertTokensBeforeCapacityFlag,
                                                     inuptXDtypeSize, quantMode, scaleDim0);
    this->quantMode = quantMode;
    if (quantMode == 0) {
    } else {
        if (scaleDim0 > 0) {
            quantTilingData.smoothType = ((scaleDim0 == 1) ? SMOOTH_1H : SMOOTH_EH);
        } else {
            quantTilingData.smoothType = SMOOTH_NONE;
        }
    }
    return true;
}

inline uint64_t MoeInitRoutingQuantTilingBase::GetTilingKey() const
{
    if (isFullLoad) {
        return TILING_KEY_PERF_BASE + quantMode * TILING_KEY_QUANT_BASE;
    }
    return TILING_KEY_BASE + quantMode * TILING_KEY_QUANT_BASE + dropPadMode * TILING_KEY_DROP_MODE_BASE +
           (totalLength > sortLoopMaxElement) * TILING_KEY_SORT_BASE;
}

inline bool MoeInitRoutingQuantTilingBase::PostTiling()
{
    CopyTilingData();
    return true;
}
inline void MoeInitRoutingQuantTilingBase::CopyGatherOutTiling(InnerMoeGatherOutComputeTilingData &dst,
                                                               InnerMoeGatherOutComputeTilingData &src)
{
    dst.needCoreNum = (src.needCoreNum);
    dst.activateRows = (src.activateRows);
    dst.perCoreRows = (src.perCoreRows);
    dst.perCorePerLoopRows = (src.perCorePerLoopRows);
    dst.perCoreLastLoopRows = (src.perCoreLastLoopRows);
    dst.lastCoreRows = (src.lastCoreRows);
    dst.lastCorePerLoopRows = (src.lastCorePerLoopRows);
    dst.lastCoreLastLoopRows = (src.lastCoreLastLoopRows);
    dst.perCoreLoops = (src.perCoreLoops);
    dst.lastCoreLoops = (src.lastCoreLoops);
    dst.perLoopCols = (src.perLoopCols);
    dst.lastLoopCols = (src.lastLoopCols);
    dst.colLoops = (src.colLoops);
}

inline void MoeInitRoutingQuantTilingBase::CopyTilingData()
{
    quantTilingData.coreNum = (InnerMoeInitRoutingTilingBase::moeInitRoutingTilingData.coreNum);
    quantTilingData.n = (InnerMoeInitRoutingTilingBase::moeInitRoutingTilingData.n);
    quantTilingData.cols = (InnerMoeInitRoutingTilingBase::moeInitRoutingTilingData.cols);
    quantTilingData.k = (InnerMoeInitRoutingTilingBase::moeInitRoutingTilingData.k);
    quantTilingData.expertCapacity = (InnerMoeInitRoutingTilingBase::moeInitRoutingTilingData.expertCapacity);
    quantTilingData.expertNum = (InnerMoeInitRoutingTilingBase::moeInitRoutingTilingData.expertNum);
    quantTilingData.dropPadMode = (InnerMoeInitRoutingTilingBase::moeInitRoutingTilingData.dropPadMode);
    quantTilingData.expertTokensCountOrCumsumFlag =
        (InnerMoeInitRoutingTilingBase::moeInitRoutingTilingData.expertTokensCountOrCumsumFlag);
    quantTilingData.expertTokensBeforeCapacityFlag =
        (InnerMoeInitRoutingTilingBase::moeInitRoutingTilingData.expertTokensBeforeCapacityFlag);

    auto vbsTilingData = &InnerMoeInitRoutingTilingBase::moeInitRoutingTilingData.vbsComputeParamsOp;
    quantTilingData.vbsComputeParamsOp.needCoreNum = (vbsTilingData->needCoreNum);
    quantTilingData.vbsComputeParamsOp.perCoreElements = (vbsTilingData->perCoreElements);
    quantTilingData.vbsComputeParamsOp.perCoreLoops = (vbsTilingData->perCoreLoops);
    quantTilingData.vbsComputeParamsOp.perCorePerLoopElements = (vbsTilingData->perCorePerLoopElements);
    quantTilingData.vbsComputeParamsOp.perCoreLastLoopElements = (vbsTilingData->perCoreLastLoopElements);
    quantTilingData.vbsComputeParamsOp.lastCoreElements = (vbsTilingData->lastCoreElements);
    quantTilingData.vbsComputeParamsOp.lastCoreLoops = (vbsTilingData->lastCoreLoops);
    quantTilingData.vbsComputeParamsOp.lastCorePerLoopElements = (vbsTilingData->lastCorePerLoopElements);
    quantTilingData.vbsComputeParamsOp.lastCoreLastLoopElements = (vbsTilingData->lastCoreLastLoopElements);
    quantTilingData.vbsComputeParamsOp.oneLoopMaxElements = (vbsTilingData->oneLoopMaxElements);

    quantTilingData.vmsMiddleComputeParamsOp.needCoreNum =
        (InnerMoeInitRoutingTilingBase::moeInitRoutingTilingData.vmsMiddleComputeParamsOp.needCoreNum);
    quantTilingData.sortOutComputeParamsOp.oneLoopMaxElements =
        (InnerMoeInitRoutingTilingBase::moeInitRoutingTilingData.sortOutComputeParamsOp.oneLoopMaxElements);

    CopyGatherOutTiling(quantTilingData.srcToDstComputeParamsOp,
                        InnerMoeInitRoutingTilingBase::moeInitRoutingTilingData.srcToDstComputeParamsOp);
    CopyGatherOutTiling(quantTilingData.srcToDstCapacityComputeParamsOp,
                        InnerMoeInitRoutingTilingBase::moeInitRoutingTilingData.srcToDstCapacityComputeParamsOp);
}

inline bool MoeInitRoutingQuantTilingBase::GetWorkspaceSize()
{
    InnerMoeInitRoutingTilingBase::GetWorkspaceSize();
    bool useCols =
        (dropPadMode == 0 && quantTilingData.gatherOutComputeParamsOp.colLoops > 1) ||
        (dropPadMode == 1 &&
         InnerMoeInitRoutingTilingBase::moeInitRoutingTilingData.srcToDstCapacityComputeParamsOp.colLoops > 1);
    if (quantMode == 1 && useCols) {
        workspaceSize_ += aivNum * InnerMoeInitRoutingTilingBase::moeInitRoutingTilingData.cols * sizeof(float);
    }
    return true;
}

inline void MoeInitRoutingQuantTilingBase::SetGatherTilingData(InnerMoeGatherOutComputeTilingData *tilingData,
                                                               int64_t perCoreRows, int64_t lastCoreRows, int64_t cols)
{
    tilingData->perCorePerLoopRows = perCoreRows;
    tilingData->perCoreLastLoopRows = perCoreRows;
    tilingData->lastCorePerLoopRows = lastCoreRows;
    tilingData->lastCoreLastLoopRows = lastCoreRows;
    tilingData->perCoreLoops = 1;
    tilingData->lastCoreLoops = 1;
    tilingData->perLoopCols = cols;
    tilingData->lastLoopCols = cols;
    tilingData->colLoops = 1;
}

inline void MoeInitRoutingQuantTilingBase::SetGatherTilingDataCols(InnerMoeGatherOutComputeTilingData *tilingData,
                                                                   int64_t baseMaxCols, int64_t cols)
{
    tilingData->perLoopCols = (std::min(baseMaxCols, cols));
    tilingData->lastLoopCols = (GetPerOrLastValue(cols, baseMaxCols));
    tilingData->colLoops = (baseMaxCols == 0 ? 0 : (cols + baseMaxCols - 1) / baseMaxCols);
}

inline void MoeInitRoutingQuantTilingBase::SetGatherTilingDataRows(InnerMoeGatherOutComputeTilingData *tilingData,
                                                                   int64_t perCoreRows, int64_t lastCoreRows,
                                                                   int64_t basePerLoopMaxRows)
{
    tilingData->perCorePerLoopRows = (std::min(perCoreRows, basePerLoopMaxRows));
    tilingData->perCoreLastLoopRows = (GetPerOrLastValue(perCoreRows, basePerLoopMaxRows));
    tilingData->perCoreLoops =
        (basePerLoopMaxRows == 0 ? 0 : (perCoreRows + basePerLoopMaxRows - 1) / basePerLoopMaxRows);
    tilingData->lastCorePerLoopRows = (std::min(lastCoreRows, basePerLoopMaxRows));
    tilingData->lastCoreLastLoopRows = (GetPerOrLastValue(lastCoreRows, basePerLoopMaxRows));
    tilingData->lastCoreLoops =
        (basePerLoopMaxRows == 0 ? 0 : (lastCoreRows + basePerLoopMaxRows - 1) / basePerLoopMaxRows);
}

inline void MoeInitRoutingQuantTilingBase::Tiling4SrcToDstCapacityCompute()
{
    if (quantMode == 0 || dropPadMode == 0) {
        InnerMoeInitRoutingTilingBase::Tiling4SrcToDstCapacityCompute();
        return;
    }

    auto tilingData = &moeInitRoutingTilingData.srcToDstCapacityComputeParamsOp;
    int64_t perCoreRows = CeilDiv(totalLength, aivNum);
    if (perCoreRows <= 0) {
        tilingData->needCoreNum = 0;
        return;
    }

    tilingData->needCoreNum = CeilDiv(totalLength, perCoreRows);
    int64_t cols = moeInitRoutingTilingData.cols;
    tilingData->perCoreRows = perCoreRows;
    int64_t lastCoreRows = totalLength - perCoreRows * (tilingData->needCoreNum - 1);
    tilingData->lastCoreRows = lastCoreRows;

    int64_t rowSize =
        CeilDiv(perCoreRows * static_cast<int64_t>(sizeof(int32_t)), ONE_BLOCK_BYTE) * ONE_BLOCK_BYTE * NUM_FOUR;
    int64_t colSize = CeilDiv(cols * static_cast<int64_t>(sizeof(int8_t)), ONE_BLOCK_BYTE) * ONE_BLOCK_BYTE *
                      DYNAMIC_QUANT_SRC_TO_DST_BUFFER;
    int64_t scaleSize = DYNAMIC_QUANT_SCALE_SIZE_64;
    if (rowSize + colSize + scaleSize < static_cast<int64_t>(aicoreParams_.ubSize)) {
        SetGatherTilingData(tilingData, perCoreRows, lastCoreRows, cols);
    } else {
        int64_t baseMaxCols = MAX_COLS_DYNAMIC_QUANT;
        int64_t totalColSize = CeilDiv(baseMaxCols * static_cast<int64_t>(sizeof(int8_t)), ONE_BLOCK_BYTE) *
                               ONE_BLOCK_BYTE * DYNAMIC_QUANT_SRC_TO_DST_BUFFER;
        int64_t ubSize = static_cast<int64_t>(aicoreParams_.ubSize);
        int64_t basePerLoopMaxRows =
            (((ubSize - totalColSize - scaleSize) / sizeof(int32_t)) / ONE_BLOCK_BYTE * ONE_BLOCK_BYTE) / NUM_FOUR;
        if (cols < MAX_COLS_DYNAMIC_QUANT) {
            basePerLoopMaxRows =
                (((ubSize - colSize - scaleSize) / sizeof(int32_t)) / ONE_BLOCK_BYTE * ONE_BLOCK_BYTE) / NUM_FOUR;
        } else if (perCoreRows < basePerLoopMaxRows) {
            baseMaxCols =
                ((ubSize - rowSize - scaleSize) / ONE_BLOCK_BYTE * ONE_BLOCK_BYTE) / DYNAMIC_QUANT_SRC_TO_DST_BUFFER;
        }
        SetGatherTilingDataCols(tilingData, baseMaxCols, cols);
        SetGatherTilingDataRows(tilingData, perCoreRows, lastCoreRows, basePerLoopMaxRows);
    }
}

inline void MoeInitRoutingQuantTilingBase::Tiling4GatherQuant()
{
    auto tilingData = &quantTilingData.gatherOutComputeParamsOp;
    tilingData->activateRows = totalLength;
    if (dropPadMode == 0 && activateNum > 0) {
        tilingData->activateRows = (std::min(activateNum, totalLength));
    }
    int64_t perCoreRows = CeilDiv(totalLength, aivNum);

    if (perCoreRows <= 0) {
        tilingData->needCoreNum = 0;
        return;
    }

    tilingData->needCoreNum = (CeilDiv(totalLength, perCoreRows));
    int64_t cols = moeInitRoutingTilingData.cols;
    tilingData->perCoreRows = perCoreRows;
    int64_t lastCoreRows = totalLength - perCoreRows * (tilingData->needCoreNum - 1);
    tilingData->lastCoreRows = lastCoreRows;
    int64_t sizeOfCol = sizeof(int8_t) * NUM_TWO + sizeof(float) + sizeof(int16_t) + inuptXDtypeSize_ * NUM_TWO;
    int64_t rowSize =
        CeilDiv(perCoreRows * static_cast<int64_t>(sizeof(int32_t)) * NUM_TWO, ONE_BLOCK_BYTE) * ONE_BLOCK_BYTE;
    int64_t colSize = CeilDiv(cols * sizeOfCol, ONE_BLOCK_BYTE) * ONE_BLOCK_BYTE;
    if (rowSize + colSize < static_cast<int64_t>(aicoreParams_.ubSize) / NUM_TWO) {
        SetGatherTilingData(tilingData, perCoreRows, lastCoreRows, cols);
    } else {
        int64_t baseMaxCols = MAX_COLS_ONE_LOOP_QUANT;
        int64_t baseMaxColsSize = CeilDiv(baseMaxCols * sizeOfCol, ONE_BLOCK_BYTE) * ONE_BLOCK_BYTE;
        int64_t ubSize = static_cast<int64_t>(aicoreParams_.ubSize);
        int64_t basePerLoopMaxRows =
            (((ubSize - baseMaxColsSize) / NUM_TWO / sizeof(int32_t)) / ONE_BLOCK_BYTE * ONE_BLOCK_BYTE);
        if (cols < MAX_COLS_ONE_LOOP_QUANT) {
            basePerLoopMaxRows = (((ubSize - colSize) / NUM_TWO / sizeof(int32_t)) / ONE_BLOCK_BYTE * ONE_BLOCK_BYTE);
        } else if (perCoreRows < basePerLoopMaxRows) {
            baseMaxCols = (((ubSize - rowSize) / sizeOfCol) / ONE_BLOCK_BYTE * ONE_BLOCK_BYTE);
        }
        SetGatherTilingDataCols(tilingData, baseMaxCols, cols);
        SetGatherTilingDataRows(tilingData, perCoreRows, lastCoreRows, basePerLoopMaxRows);
    }
}

inline void SetGatherTilingDatawithloop(InnerMoeGatherOutComputeTilingData *tilingData, int64_t perCorePerLoopRows,
                                        int64_t lastCorePerLoopRows, int64_t cols, int64_t perCoreLastLoopRows = 1,
                                        int64_t lastCoreLastLoopRows = 1, int64_t perCoreLoops = 1,
                                        int64_t lastCoreLoops = 1)
{
    tilingData->perCorePerLoopRows = perCorePerLoopRows;
    tilingData->perCoreLastLoopRows = perCoreLastLoopRows;
    tilingData->lastCorePerLoopRows = lastCorePerLoopRows;
    tilingData->lastCoreLastLoopRows = lastCoreLastLoopRows;
    tilingData->perCoreLoops = perCoreLoops;
    tilingData->lastCoreLoops = lastCoreLoops;
    tilingData->perLoopCols = cols;
    tilingData->lastLoopCols = cols;
    tilingData->colLoops = 1;
}

inline void MoeInitRoutingQuantTilingBase::Tiling4GatherDynamicQuant()
{
    auto tilingData = &quantTilingData.gatherOutComputeParamsOp;
    tilingData->activateRows = totalLength;
    if (dropPadMode == 0 && activateNum > 0) {
        tilingData->activateRows = (std::min(activateNum, totalLength));
    }
    int64_t perCoreRows = CeilDiv(totalLength, aivNum);

    if (perCoreRows <= 0) {
        tilingData->needCoreNum = 0;
        return;
    }

    tilingData->needCoreNum = (CeilDiv(totalLength, perCoreRows));

    int64_t cols = InnerMoeInitRoutingTilingBase::moeInitRoutingTilingData.cols;

    tilingData->perCoreRows = perCoreRows;
    int64_t lastCoreRows = totalLength - perCoreRows * (tilingData->needCoreNum - 1);
    tilingData->lastCoreRows = lastCoreRows;

    int64_t rowSize =
        CeilDiv(perCoreRows * static_cast<int64_t>(sizeof(int32_t)), ONE_BLOCK_BYTE) * ONE_BLOCK_BYTE * NUM_FOUR;
    int64_t colSize = CeilDiv(cols * static_cast<int64_t>(sizeof(int8_t)), ONE_BLOCK_BYTE) * ONE_BLOCK_BYTE *
                      DYNAMIC_QUANT_COLS_BUFFER;
    int64_t scaleSize = DYNAMIC_QUANT_SCALE_SIZE_64;
    int64_t onceRowSize =
        (static_cast<int64_t>(aicoreParams_.ubSize) - colSize - scaleSize - ONE_BLOCK_BYTE * NUM_FOUR * NUM_THREE) /
        (sizeof(int32_t) * NUM_FOUR);
    int64_t oneBlockNumInt = static_cast<int64_t>(ONE_BLOCK_BYTE) / static_cast<int64_t>(sizeof(int32_t));
    onceRowSize = onceRowSize / oneBlockNumInt * oneBlockNumInt;
    bool ifOneLoop =
        ((static_cast<int64_t>(aicoreParams_.ubSize) > colSize + scaleSize + ONE_BLOCK_BYTE * NUM_FOUR * NUM_FOUR) &&
         quantTilingData.smoothType == SMOOTH_NONE && cols == FULLLOAD_H_LIMIT);

    int64_t perCoreOnceRowSize = ifOneLoop ? std::min(onceRowSize, perCoreRows) : perCoreRows;
    int64_t lastCoreOnceRowSize = ifOneLoop ? std::min(onceRowSize, lastCoreRows) : lastCoreRows;
    int64_t perCoreLoops = ifOneLoop ? CeilDiv(perCoreRows, perCoreOnceRowSize) : 1;
    int64_t lastCoreLoops = ifOneLoop ? CeilDiv(lastCoreRows, lastCoreOnceRowSize) : 1;
    int64_t perCoreLastLoopRows = ifOneLoop ? GetPerOrLastValue(perCoreRows, perCoreOnceRowSize) : perCoreRows;
    int64_t lastCoreLastLoopRows = ifOneLoop ? GetPerOrLastValue(lastCoreRows, lastCoreOnceRowSize) : lastCoreRows;

    if (rowSize + colSize + scaleSize < static_cast<int64_t>(aicoreParams_.ubSize) || ifOneLoop) {
        SetGatherTilingDatawithloop(tilingData, perCoreOnceRowSize, lastCoreOnceRowSize, cols, perCoreLastLoopRows,
                                    lastCoreLastLoopRows, perCoreLoops, lastCoreLoops);
    } else {
        int64_t baseMaxCols = MAX_COLS_DYNAMIC_QUANT;
        int64_t totalColSize = CeilDiv(baseMaxCols * static_cast<int64_t>(sizeof(int8_t)), ONE_BLOCK_BYTE) *
                               ONE_BLOCK_BYTE * DYNAMIC_QUANT_COLS_BUFFER;
        int64_t ubSize = static_cast<int64_t>(aicoreParams_.ubSize);
        int64_t basePerLoopMaxRows =
            (((ubSize - totalColSize - scaleSize) / sizeof(int32_t)) / ONE_BLOCK_BYTE * ONE_BLOCK_BYTE) / NUM_FOUR;
        if (cols < MAX_COLS_DYNAMIC_QUANT) {
            basePerLoopMaxRows =
                (((ubSize - colSize - scaleSize) / sizeof(int32_t)) / ONE_BLOCK_BYTE * ONE_BLOCK_BYTE) / NUM_FOUR;
        } else if (perCoreRows < basePerLoopMaxRows) {
            baseMaxCols =
                ((ubSize - rowSize - scaleSize) / ONE_BLOCK_BYTE * ONE_BLOCK_BYTE) / DYNAMIC_QUANT_COLS_BUFFER;
        }
        SetGatherTilingDataCols(tilingData, baseMaxCols, cols);
        SetGatherTilingDataRows(tilingData, perCoreRows, lastCoreRows, basePerLoopMaxRows);
    }
}

inline void MoeInitRoutingQuantTilingBase::Tiling4GatherOutCompute()
{
    if (quantMode == 0) {
        Tiling4GatherQuant();
    } else {
        Tiling4GatherDynamicQuant();
    }
}

} // namespace optiling

#endif