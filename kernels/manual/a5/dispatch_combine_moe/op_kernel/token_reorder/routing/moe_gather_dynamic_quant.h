/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_gather_dynamic_quant.h
 * \brief
 */
#ifndef MOE_GATHER_DYNAMIC_QUANT_H
#define MOE_GATHER_DYNAMIC_QUANT_H

#include "moe_common.h"
#include "moe_pto_sort.h"

namespace MoeInitRoutingQuant {
using namespace AscendC;
using namespace optiling;
template <typename T>
class MoeGatherDynamicQuant {
public:
    __aicore__ inline MoeGatherDynamicQuant(){};
    __aicore__ inline void Init(GM_ADDR inputX, GM_ADDR quantSmooth, GM_ADDR expandedRowIdx, GM_ADDR expandedX,
                                GM_ADDR dynamicQuantScale, GM_ADDR workspace,
                                const MoeInitRoutingQuantTilingData *tilingData, AscendC::TPipe *tPipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyInExpandedRowIdx(int64_t progress);
    __aicore__ inline void CopyInExpandedExpertIdx(int64_t progress);
    __aicore__ inline void CopyOutXQuant1H(int64_t progress);
    __aicore__ inline void CopyOutXQuantEH(int64_t progress);
    __aicore__ inline void Compute(uint64_t smoothUb);
    __aicore__ inline void LoadInputTile(uint64_t inUb, int64_t srcOffset, int64_t elemNum);
    __aicore__ inline void StoreExpandedXTile(int64_t dstOffset, uint64_t outUb, int64_t elemNum);
    __aicore__ inline void CopyOutPartialXQuantEH(int64_t progress);
    __aicore__ inline void CopyOutPartialXQuant1H(int64_t progress);
    __aicore__ inline float ComputeMax(uint64_t inUb, uint64_t tempUb, uint64_t dynamicQuantScaleUb, int32_t srcIdx,
                                       int32_t expertIdx, int64_t j);
    __aicore__ inline void ComputeScale(uint64_t inUb, uint64_t tempUb, float scaleTemp, int64_t dstIndex, int64_t j);

private:
    uint64_t indicesUb;
    uint64_t inputXUb;
    uint64_t smoothUb_;
    uint64_t tempUb;
    uint64_t outputXUb;
    uint64_t scaleUb;

    __gm__ T *inputXGm;
    __gm__ int8_t *expandedXGm;
    __gm__ int32_t *expandedRowIdxGm;
    __gm__ float *quantSmoothGm;
    __gm__ float *dynamicQuantScaleGm;
    __gm__ float *quantSrcGm;
    __gm__ int32_t *expandedExpertIdxGm;
    __gm__ int32_t *sortedRowIdxGm;

    const InnerMoeGatherOutComputeTilingData *gatherOutTilingData;

    int64_t needCoreNum;
    int64_t blockIdx;
    int64_t cols;
    int64_t cols_scale_;
    int64_t n;
    int64_t k;
    int64_t totalLength;
    int64_t activateRows;
    int64_t currentLoopRows;
    int64_t currentLoopRowsAlign;
    int64_t coreRows;
    int64_t perLoopRows;
    int64_t lastLoopRows;
    int64_t rowLoops;
    int64_t colsTileLength;
    int64_t perLoopCols;
    int64_t perLoopColsAlign;
    int64_t lastLoopCols;
    int64_t colLoops;
    int64_t dropPadMode;
    int64_t smoothType;

    int64_t indicesOffset;
    int64_t inputOffset;
    int64_t outOffset;
};

template <typename T>
__aicore__ inline void MoeGatherDynamicQuant<T>::CopyInExpandedRowIdx(int64_t progress)
{
    this->indicesOffset = progress * this->perLoopRows;
    pto_detail::PtoLoadVector<int32_t>(this->indicesUb, expandedRowIdxGm + indicesOffset, this->currentLoopRows);
}

template <typename T>
__aicore__ inline void MoeGatherDynamicQuant<T>::CopyInExpandedExpertIdx(int64_t progress)
{
    this->indicesOffset = progress * this->perLoopRows;
    pto_detail::PtoLoadVector<int32_t>(this->indicesUb, sortedRowIdxGm + indicesOffset, this->currentLoopRows);
    pto_detail::PtoLoadVector<int32_t>(this->indicesUb + static_cast<uint64_t>(currentLoopRowsAlign) * sizeof(int32_t),
                                       expandedExpertIdxGm + indicesOffset, this->currentLoopRows);
}

template <typename T>
__aicore__ inline void MoeGatherDynamicQuant<T>::Compute(uint64_t smoothUb)
{
    const uint64_t inUb = this->inputXUb;
    const uint64_t tempUb = this->tempUb;
    const uint64_t outputPayloadUb = this->outputXUb;
    const uint64_t dynamicQuantScaleUb = outputPayloadUb + static_cast<uint64_t>(this->cols) * sizeof(int8_t);

    if constexpr (!IsSameType<T, float>::value) {
        const uint64_t rawInputUb = inUb + static_cast<uint64_t>(perLoopColsAlign) * sizeof(T);
        pto_detail::PtoCastVector<float, T>(inUb, rawInputUb, this->cols, pto::RoundMode::CAST_NONE);
    }

    if (smoothType != 0) {
        pto_detail::PtoMulElementwiseVector<float>(inUb, inUb, smoothUb, this->cols);
    }

    pto_detail::PtoAbsVector<float>(tempUb, inUb, this->cols);

    pto_detail::PtoReduceMaxVector(dynamicQuantScaleUb, tempUb, tempUb, this->cols);
    pto_detail::PtoPipeBarrier<PIPE_V>();

    float maxValue = pto_detail::PtoGetValue<float>(dynamicQuantScaleUb, 0) / 127.0f;

    pto_detail::PtoFillVector<float>(dynamicQuantScaleUb, maxValue, 8);
    pto_detail::PtoFillVector<float>(tempUb, maxValue, this->cols);
    pto_detail::PtoPipeBarrier<PIPE_V>();

    pto_detail::PtoDivVector<float>(tempUb, inUb, tempUb, this->cols);

    pto_detail::PtoCastVector<half, float>(tempUb, tempUb, this->cols, pto::RoundMode::CAST_TRUNC);
    pto_detail::PtoCastVector<int8_t, half>(outputPayloadUb, tempUb, this->cols, pto::RoundMode::CAST_ROUND);
}

template <typename T>
__aicore__ inline void MoeGatherDynamicQuant<T>::LoadInputTile(uint64_t inUb, int64_t srcOffset, int64_t elemNum)
{
    if constexpr (IsSameType<T, float>::value) {
        pto_detail::PtoLoadVector<float>(inUb, inputXGm + srcOffset, elemNum);
    } else {
        pto_detail::PtoLoadVector<T>(inUb + static_cast<uint64_t>(perLoopColsAlign) * sizeof(T), inputXGm + srcOffset,
                                     elemNum);
    }
}

template <typename T>
__aicore__ inline void MoeGatherDynamicQuant<T>::StoreExpandedXTile(int64_t dstOffset, uint64_t outUb,
                                                                      int64_t elemNum)
{
    pto_detail::PtoStoreVector(expandedXGm + dstOffset, outUb, elemNum);
}

template <typename T>
__aicore__ inline void MoeGatherDynamicQuant<T>::CopyOutXQuant1H(int64_t progress)
{
    int64_t initialRow = this->gatherOutTilingData->perCoreRows * this->blockIdx + this->perLoopRows * progress;
    int64_t curLoopRow = 0;
    int64_t currentLoopStartRow = initialRow / this->k;
    int64_t currentLoopLastRow = (initialRow + this->currentLoopRows - 1) / this->k;
    if (smoothType == 1) {
        pto_detail::PtoLoadVector<float>(this->smoothUb_, quantSmoothGm, this->cols);
        pto_detail::PtoSetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
    }

    for (int64_t row = currentLoopStartRow; row <= currentLoopLastRow; row++) {
        LoadInputTile(this->inputXUb, row * this->cols, this->cols);
        pto_detail::PtoSetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
        Compute(this->smoothUb_);
        pto_detail::PtoSetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);

        while (curLoopRow < this->currentLoopRows && initialRow / this->k == row) {
            int32_t outIndex = pto_detail::PtoGetValue<int32_t>(this->indicesUb, curLoopRow);
            curLoopRow++;
            initialRow++;
            if (outIndex == -1 || (this->dropPadMode == DROPLESS_MODE && outIndex >= this->activateRows)) {
                continue;
            }
            StoreExpandedXTile(outIndex * cols_scale_, this->outputXUb, cols_scale_);
        }
        pto_detail::PtoSetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
    }
}

template <typename T>
__aicore__ inline void MoeGatherDynamicQuant<T>::CopyOutXQuantEH(int64_t progress)
{
    pto_detail::PtoSetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);

    int32_t lastExpertIdx = -1;
    for (int64_t i = 0; i < this->currentLoopRows; i++) {
        int64_t rowOffset = this->gatherOutTilingData->perCoreRows * this->blockIdx + this->perLoopRows * progress;
        if (this->dropPadMode == DROPLESS_MODE && rowOffset + i >= this->activateRows) {
            break;
        }
        int32_t srcIdx = pto_detail::PtoGetValue<int32_t>(this->indicesUb, i);
        int32_t expertIdx = pto_detail::PtoGetValue<int32_t>(this->indicesUb, currentLoopRowsAlign + i);

        LoadInputTile(this->inputXUb, srcIdx / this->k * this->cols, this->perLoopCols);
        pto_detail::PtoSetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
        if (expertIdx != lastExpertIdx) {
            pto_detail::PtoLoadVector<float>(this->smoothUb_, quantSmoothGm + expertIdx * this->cols,
                                             this->perLoopCols);
            pto_detail::PtoSetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
            lastExpertIdx = expertIdx;
        }

        Compute(this->smoothUb_);
        pto_detail::PtoSetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);

        pto_detail::PtoStoreVector<float>(dynamicQuantScaleGm + (rowOffset + i),
                                          this->outputXUb + static_cast<uint64_t>(this->cols) * sizeof(int8_t), 1);
        StoreExpandedXTile((rowOffset + i) * this->cols, this->outputXUb, this->perLoopCols);
        pto_detail::PtoSetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
    }
}

template <typename T>
__aicore__ inline float MoeGatherDynamicQuant<T>::ComputeMax(uint64_t inUb, uint64_t tempUb,
                                                               uint64_t dynamicQuantScaleUb, int32_t srcIdx,
                                                               int32_t expertIdx, int64_t j)
{
    LoadInputTile(inUb, srcIdx * this->cols + j * this->perLoopCols, colsTileLength);
    pto_detail::PtoSetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);

    if (smoothType != 0) {
        pto_detail::PtoLoadVector<float>(
            this->smoothUb_, quantSmoothGm + expertIdx * this->cols + j * this->perLoopCols, colsTileLength);
        pto_detail::PtoSetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
    }

    if constexpr (!IsSameType<T, float>::value) {
        pto_detail::PtoCastVector<float, T>(inUb, inUb + static_cast<uint64_t>(perLoopColsAlign) * sizeof(T),
                                            colsTileLength, pto::RoundMode::CAST_NONE);
    }

    if (smoothType != 0) {
        pto_detail::PtoMulElementwiseVector<float>(inUb, inUb, this->smoothUb_, colsTileLength);
    }

    pto_detail::PtoAbsVector<float>(tempUb, inUb, colsTileLength);

    pto_detail::PtoReduceMaxVector(dynamicQuantScaleUb + 8 * sizeof(float), tempUb, tempUb, colsTileLength);

    pto_detail::PtoStoreVector<float>(quantSrcGm + j * this->perLoopCols, inUb, colsTileLength);
    pto_detail::PtoSetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);

    return pto_detail::PtoGetValue<float>(dynamicQuantScaleUb, 8);
}

template <typename T>
__aicore__ inline void MoeGatherDynamicQuant<T>::ComputeScale(uint64_t inUb, uint64_t tempUb, float scaleTemp,
                                                                int64_t dstIndex, int64_t j)
{
    pto_detail::PtoLoadVector<float>(inUb, quantSrcGm + j * this->perLoopCols, colsTileLength);
    pto_detail::PtoSetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);

    pto_detail::PtoFillVector<float>(tempUb, scaleTemp, colsTileLength);
    pto_detail::PtoPipeBarrier<PIPE_V>();

    pto_detail::PtoDivVector<float>(tempUb, inUb, tempUb, colsTileLength);

    pto_detail::PtoCastVector<half, float>(tempUb, tempUb, colsTileLength, pto::RoundMode::CAST_TRUNC);
    pto_detail::PtoCastVector<int8_t, half>(this->outputXUb, tempUb, colsTileLength, pto::RoundMode::CAST_ROUND);

    pto_detail::PtoSetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
    StoreExpandedXTile(dstIndex * this->cols + j * this->perLoopCols, this->outputXUb, colsTileLength);
    pto_detail::PtoSetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
}

template <typename T>
__aicore__ inline void MoeGatherDynamicQuant<T>::CopyOutPartialXQuantEH(int64_t progress)
{
    pto_detail::PtoSetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);

    for (int64_t i = 0; i < this->currentLoopRows; i++) {
        int64_t rowOffset = this->gatherOutTilingData->perCoreRows * this->blockIdx + this->perLoopRows * progress;
        if (this->dropPadMode == DROPLESS_MODE && rowOffset + i >= this->activateRows) {
            break;
        }
        int32_t srcIdx = pto_detail::PtoGetValue<int32_t>(this->indicesUb, i);
        int32_t expertIdx = pto_detail::PtoGetValue<int32_t>(this->indicesUb, currentLoopRowsAlign + i);

        uint32_t tmp = 0xFF7FFFFF;
        float reduceMax = *((float *)&tmp);
        for (int64_t j = 0; j < this->colLoops; j++) {
            colsTileLength = this->perLoopCols;
            if (j == this->colLoops - 1) {
                colsTileLength = this->lastLoopCols;
            }
            float tileMax = ComputeMax(this->inputXUb, this->tempUb, this->scaleUb, srcIdx / this->k, expertIdx, j);
            reduceMax = (reduceMax > tileMax) ? reduceMax : tileMax;
        }

        float scaleTemp = reduceMax / 127.0f;
        pto_detail::PtoFillVector<float>(this->scaleUb, scaleTemp, 8);
        pto_detail::PtoSetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
        pto_detail::PtoStoreVector<float>(dynamicQuantScaleGm + (rowOffset + i), this->scaleUb, 1);

        for (int64_t j = 0; j < this->colLoops; j++) {
            colsTileLength = this->perLoopCols;
            if (j == this->colLoops - 1) {
                colsTileLength = this->lastLoopCols;
            }

            ComputeScale(this->inputXUb, this->tempUb, scaleTemp, rowOffset + i, j);
        }
    }
}

template <typename T>
__aicore__ inline void MoeGatherDynamicQuant<T>::CopyOutPartialXQuant1H(int64_t progress)
{
    int64_t initialRow = this->gatherOutTilingData->perCoreRows * this->blockIdx + this->perLoopRows * progress;
    int64_t curLoopRow = 0;

    int64_t currentLoopStartRow = initialRow / this->k;
    int64_t currentLoopLastRow = (initialRow + this->currentLoopRows - 1) / this->k;

    for (int64_t row = currentLoopStartRow; row <= currentLoopLastRow; row++) {
        uint32_t tmp = 0xFF7FFFFF;
        float reduceMax = *((float *)&tmp);
        for (int64_t j = 0; j < this->colLoops; j++) {
            colsTileLength = this->perLoopCols;
            if (j == this->colLoops - 1) {
                colsTileLength = this->lastLoopCols;
            }

            float tileMax = ComputeMax(this->inputXUb, this->tempUb, this->scaleUb, row, 0, j);
            reduceMax = (reduceMax > tileMax) ? reduceMax : tileMax;
        }

        float scaleTemp = reduceMax / 127.0f;
        pto_detail::PtoFillVector<float>(this->scaleUb, scaleTemp, 8);
        pto_detail::PtoSetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);

        while (curLoopRow < this->currentLoopRows && initialRow / this->k == row) {
            int32_t outIndex = pto_detail::PtoGetValue<int32_t>(this->indicesUb, curLoopRow);
            curLoopRow++;
            initialRow++;
            if (outIndex == -1 || (this->dropPadMode == DROPLESS_MODE && outIndex >= this->activateRows)) {
                continue;
            }
            pto_detail::PtoStoreVector<float>(dynamicQuantScaleGm + outIndex, this->scaleUb, 1);
            for (int64_t j = 0; j < this->colLoops; j++) {
                colsTileLength = this->perLoopCols;
                if (j == this->colLoops - 1) {
                    colsTileLength = this->lastLoopCols;
                }

                ComputeScale(this->inputXUb, this->tempUb, scaleTemp, outIndex, j);
            }
        }
    }
}

template <typename T>
__aicore__ inline void MoeGatherDynamicQuant<T>::Init(GM_ADDR inputX, GM_ADDR quantSmooth, GM_ADDR expandedRowIdx,
                                                        GM_ADDR expandedX, GM_ADDR dynamicQuantScale, GM_ADDR workspace,
                                                        const MoeInitRoutingQuantTilingData *tilingData,
                                                        AscendC::TPipe *tPipe)
{
    (void)tPipe;
    this->blockIdx = get_block_idx() + get_subblockid() * get_block_num();
    this->gatherOutTilingData = &(tilingData->gatherOutComputeParamsOp);

    this->needCoreNum = this->gatherOutTilingData->needCoreNum;
    this->activateRows = this->gatherOutTilingData->activateRows;
    this->cols = tilingData->cols;
    this->cols_scale_ = this->cols + UB_ALIGN;
    this->n = tilingData->n;
    this->k = tilingData->k;
    this->totalLength = tilingData->n * tilingData->k;
    this->dropPadMode = tilingData->dropPadMode;
    this->smoothType = tilingData->smoothType;

    if (this->blockIdx == this->gatherOutTilingData->needCoreNum - 1) {
        this->coreRows = this->gatherOutTilingData->lastCoreRows;
        this->perLoopRows = this->gatherOutTilingData->lastCorePerLoopRows;
        this->lastLoopRows = this->gatherOutTilingData->lastCoreLastLoopRows;
        this->rowLoops = this->gatherOutTilingData->lastCoreLoops;
    } else {
        this->coreRows = this->gatherOutTilingData->perCoreRows;
        this->perLoopRows = this->gatherOutTilingData->perCorePerLoopRows;
        this->lastLoopRows = this->gatherOutTilingData->perCoreLastLoopRows;
        this->rowLoops = this->gatherOutTilingData->perCoreLoops;
    }
    this->perLoopCols = this->gatherOutTilingData->perLoopCols;
    this->lastLoopCols = this->gatherOutTilingData->lastLoopCols;
    this->colLoops = this->gatherOutTilingData->colLoops;
    this->perLoopColsAlign = Align(this->perLoopCols, sizeof(T));

    inputXGm = (__gm__ T *)inputX;
    expandedXGm = (__gm__ int8_t *)expandedX;

    expandedRowIdxGm = (__gm__ int32_t *)expandedRowIdx + this->blockIdx * this->gatherOutTilingData->perCoreRows;

    quantSmoothGm = (__gm__ float *)quantSmooth;
    dynamicQuantScaleGm = (__gm__ float *)dynamicQuantScale;

    expandedExpertIdxGm = (__gm__ int32_t *)workspace + this->blockIdx * this->gatherOutTilingData->perCoreRows;
    sortedRowIdxGm = (__gm__ int32_t *)workspace + Align(this->totalLength, sizeof(int32_t)) +
                     this->blockIdx * this->gatherOutTilingData->perCoreRows;
    if (this->cols > 1) {
        quantSrcGm =
            (__gm__ float *)workspace + Align(this->totalLength, sizeof(int32_t)) * 2 + this->blockIdx * this->cols;
    }

    this->currentLoopRowsAlign = Align(this->perLoopRows, sizeof(int32_t));

    int64_t perLoopColsAlignBytes = AlignBytes(this->perLoopCols, sizeof(T));
    perLoopColsAlignBytes =
        Max(int64_t(perLoopColsAlignBytes * sizeof(float) / sizeof(T)), int64_t(BLOCK_BYTES + BLOCK_BYTES));

    this->indicesUb = 0;
    this->inputXUb = this->indicesUb + 2 * AlignBytes(this->perLoopRows, sizeof(int32_t));
    this->smoothUb_ = this->inputXUb + perLoopColsAlignBytes;
    this->tempUb = this->smoothUb_ + AlignBytes(this->perLoopCols, sizeof(float));
    this->outputXUb = this->tempUb + AlignBytes(this->perLoopCols, sizeof(float));
    this->scaleUb = this->outputXUb + AlignBytes(this->cols_scale_, sizeof(int8_t));
}

template <typename T>
__aicore__ inline void MoeGatherDynamicQuant<T>::Process()
{
    if (this->blockIdx < this->needCoreNum) {
        currentLoopRows = perLoopRows;
        if (colLoops > 1) { // Cannot fit all data in one row, workspace is required
            trap();         // Not supported
        } else {            // All data can fit in one row
            if (smoothType == 2) {
                for (int64_t loop = 0; loop < this->rowLoops - 1; loop++) {
                    CopyInExpandedExpertIdx(loop);
                    CopyOutXQuantEH(loop);
                }
                currentLoopRows = lastLoopRows;
                CopyInExpandedExpertIdx(this->rowLoops - 1);
                CopyOutXQuantEH(this->rowLoops - 1);
            } else {
                for (int64_t loop = 0; loop < this->rowLoops - 1; loop++) {
                    CopyInExpandedRowIdx(loop);
                    CopyOutXQuant1H(loop);
                }
                currentLoopRows = lastLoopRows;
                CopyInExpandedRowIdx(this->rowLoops - 1);
                CopyOutXQuant1H(this->rowLoops - 1);
            }
        }
    }
}
} // namespace MoeInitRoutingQuant
#endif // MOE_GATHER_DYNAMIC_QUANT_H
