/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file moe_v2_fullload_quant.h
 * \brief
 */
#ifndef MOE_V2_FULL_LOAD_QUANT_H
#define MOE_V2_FULL_LOAD_QUANT_H

#include "moe_v2_fullload_quant_base.h"

namespace MoeInitRoutingQuantV2 {
using namespace AscendC;
using namespace optiling;
template <typename T>
class MoeV2FullLoadQuant : public MoeV2FullLoadQuantBase {
public:
    __aicore__ inline MoeV2FullLoadQuant(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR expertIdx, GM_ADDR scale, GM_ADDR offset, GM_ADDR expandedX,
                                GM_ADDR expandedRowIdx, GM_ADDR expertTokensCountOrCumsum, GM_ADDR workspace,
                                const MoeInitRoutingQuantV2TilingData *tilingData, TPipe *tPipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void Compute(int64_t xLocalLength);
    __aicore__ inline void LoadXRows(uint64_t xUb, int64_t startXRow, int64_t rowCount, int64_t inFactor);
    __aicore__ inline void StoreExpandedXRow(int32_t outIndex, uint64_t outUb, int64_t localOffset);
    __aicore__ inline void CopyOutX();

private:
    uint64_t inputXUb;
    uint64_t outputXUb;
    uint64_t floatUb;
    uint64_t halfUb;

    __gm__ T *xGm;
    __gm__ float *scaleGm;
    __gm__ float *offsetGm;

    float scale;
    float offset;
};

template <typename T>
__aicore__ inline void MoeV2FullLoadQuant<T>::Compute(int64_t xLocalLength)
{
    uint32_t elements = Align(this->cols, sizeof(int8_t)) * xLocalLength;
    if constexpr (IsSameType<T, bfloat16_t>::value) {
        pto_detail::PtoCastVector<float, T>(this->floatUb, this->inputXUb, elements, pto::RoundMode::CAST_NONE);
        pto_detail::PtoCastVector<half, float>(this->halfUb, this->floatUb, elements, pto::RoundMode::CAST_NONE);
        pto_detail::PtoMulVector<half>(this->halfUb, this->halfUb, elements, static_cast<half>(this->scale));
        pto_detail::PtoAddScalarVector<half>(this->halfUb, this->halfUb, elements, static_cast<half>(this->offset));
        pto_detail::PtoCastVector<int32_t, half>(this->floatUb, this->halfUb, elements, pto::RoundMode::CAST_RINT);
        SetDeqScale((half)1.000000e+00f);
        pto_detail::PtoPipeBarrier<PIPE_V>();
        pto_detail::PtoCastVector<half, int32_t>(this->halfUb, this->floatUb, elements, pto::RoundMode::CAST_RINT);
        pto_detail::PtoCastVector<int8_t, half>(this->outputXUb, this->halfUb, elements, pto::RoundMode::CAST_RINT);
    } else if constexpr (IsSameType<T, float>::value) {
        pto_detail::PtoCastVector<half, T>(this->halfUb, this->inputXUb, elements, pto::RoundMode::CAST_NONE);
        pto_detail::PtoMulVector<half>(this->halfUb, this->halfUb, elements, static_cast<half>(this->scale));
        pto_detail::PtoAddScalarVector<half>(this->halfUb, this->halfUb, elements, static_cast<half>(this->offset));
        pto_detail::PtoCastVector<int8_t, half>(this->outputXUb, this->halfUb, elements, pto::RoundMode::CAST_RINT);
    } else {
        pto_detail::PtoMulVector<T>(this->inputXUb, this->inputXUb, elements, static_cast<T>(this->scale));
        pto_detail::PtoAddScalarVector<T>(this->inputXUb, this->inputXUb, elements, static_cast<T>(this->offset));
        pto_detail::PtoCastVector<int8_t, T>(this->outputXUb, this->inputXUb, elements, pto::RoundMode::CAST_RINT);
    }
}

template <typename T>
__aicore__ inline void MoeV2FullLoadQuant<T>::LoadXRows(uint64_t xUb, int64_t startXRow, int64_t rowCount,
                                                        int64_t inFactor)
{
    for (int64_t row = 0; row < rowCount; ++row) {
        pto_detail::PtoLoadVector<T>(xUb + static_cast<uint64_t>(row * inFactor) * sizeof(T),
                                     xGm + (startXRow + row) * this->cols, this->cols);
    }
}

template <typename T>
__aicore__ inline void MoeV2FullLoadQuant<T>::StoreExpandedXRow(int32_t outIndex, uint64_t outUb, int64_t localOffset)
{
    pto_detail::PtoStoreVector(expandedXGm + outIndex * this->cols,
                               outUb + static_cast<uint64_t>(localOffset) * sizeof(int8_t), this->cols);
}

template <typename T>
__aicore__ inline void MoeV2FullLoadQuant<T>::CopyOutX()
{
    int64_t inFactor = Align(this->cols, sizeof(int8_t));
    int64_t curRowsStart = this->blockIdx * this->perCoreRows;
    int64_t startXRow = curRowsStart / this->k;
    int64_t endXRow = (curRowsStart + this->coreRows - 1) / this->k;

    LoadXRows(this->inputXUb, startXRow, endXRow - startXRow + 1, inFactor);
    pto_detail::PtoSetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
    Compute(endXRow - startXRow + 1);
    pto_detail::PtoSetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
    int64_t k = 0;
    for (int64_t i = startXRow; i <= endXRow; i++) {
        for (; k < this->perCoreRows && curRowsStart / this->k == i; curRowsStart++, k++) {
            int32_t outIndex = pto_detail::PtoGetValue<int32_t>(this->expandedRowIdxUb, curRowsStart);
            if (outIndex < this->activateRows) {
                StoreExpandedXRow(outIndex, this->outputXUb, (i - startXRow) * inFactor);
            }
        }
    }
    pto_detail::PtoSetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
}

template <typename T>
__aicore__ inline void MoeV2FullLoadQuant<T>::Init(GM_ADDR x, GM_ADDR expertIdx, GM_ADDR scale, GM_ADDR offset,
                                                   GM_ADDR expandedX, GM_ADDR expandedRowIdx,
                                                   GM_ADDR expertTokensCountOrCumsum, GM_ADDR workspace,
                                                   const MoeInitRoutingQuantV2TilingData *tilingData, TPipe *tPipe)
{
    this->InitBase(x, expertIdx, expandedX, expandedRowIdx, expertTokensCountOrCumsum, workspace, tilingData, tPipe);
    xGm = (__gm__ T *)x;
    scaleGm = (__gm__ float *)scale;
    offsetGm = (__gm__ float *)offset;
    this->scale = scaleGm[0];
    this->offset = offsetGm[0];

    int64_t curRowsStart = this->blockIdx * this->perCoreRows;
    int64_t rowLength = (curRowsStart + this->coreRows - 1) / this->k - curRowsStart / this->k + 1;
    int64_t xAlignedCount = Align(this->cols, sizeof(int8_t));
    uint64_t baseUb = this->mergeTmpUb + GetSortLen<float>(this->sortNum) * sizeof(float);
    this->inputXUb = baseUb;
    this->outputXUb = this->inputXUb + AlignBytes(xAlignedCount * rowLength, sizeof(T));
    this->floatUb = this->outputXUb + AlignBytes(xAlignedCount * rowLength, sizeof(int8_t));
    this->halfUb = this->floatUb + AlignBytes(xAlignedCount * rowLength, sizeof(float));
}

template <typename T>
__aicore__ inline void MoeV2FullLoadQuant<T>::Process()
{
    if (this->blockIdx < this->needCoreNum) {
        this->ProcessBase();
        CopyOutX();
    }
}
} // namespace MoeInitRoutingQuantV2
#endif