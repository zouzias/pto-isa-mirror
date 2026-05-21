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
 * \file moe_v2_fullload_dynamic_quant.h
 * \brief
 */
#ifndef MOE_V2_FULL_LOAD_DYNAMIC_QUANT_H
#define MOE_V2_FULL_LOAD_DYNAMIC_QUANT_H

#include "moe_v2_mrgsort.h"
#include "moe_v2_pto_sort.h"
#include "moe_v2_sort_base.h"
namespace MoeInitRoutingQuantV2 {
using namespace AscendC;
using namespace optiling;
template <typename T>
class MoeV2FullLoadDynamicQuant : public MoeV2SortBase {
public:
    __aicore__ inline MoeV2FullLoadDynamicQuant(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR expertIdx, GM_ADDR expandedX, GM_ADDR expandedRowIdx,
                                GM_ADDR expertTokensCountOrCumsum, GM_ADDR quantSmooth, GM_ADDR dynamicQuantScale,
                                GM_ADDR workspace, const MoeInitRoutingQuantV2TilingData *tilingData,
                                AscendC::TPipe *tPipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn();
    __aicore__ inline void SortCompute();
    __aicore__ inline void CopyOutIdx();
    __aicore__ inline void CopyOutEmpty();
    __aicore__ inline void CopyOutXQuant1H();
    __aicore__ inline void ComputeExpertTokenCountOrCumsum();
    __aicore__ inline void Compute(uint64_t smoothUb);
    __aicore__ inline void LoadInputRow(uint64_t inUb, int64_t row);
    __aicore__ inline void StoreExpandedXRow(int64_t outIndex, uint64_t outUb);
    __aicore__ inline void QuantizeTile(uint64_t inUb, uint64_t tempUb, uint64_t outputPayloadUb,
                                        uint64_t dynamicQuantScaleUb, uint64_t smoothUb);

private:
    int64_t sortNum_;
    const InnerMoeV2GatherOutComputeTilingData *gatherOutTilingData_;
    int64_t blockIdx_;
    int64_t needCoreNum_;
    int64_t coreRows_;
    int64_t perCoreRows_;
    int64_t k_;
    int64_t n_;
    int64_t cols_;
    int64_t cols_scale_;
    int64_t activateRows_;
    int64_t expertNum;
    int64_t smoothType;
    int64_t colsAlign;

    uint64_t expandedRowIdxUb_;
    uint64_t expandedExpertIdxUb_;
    uint64_t expandDstToSrcRowUb_;
    uint64_t expertTokensUb_;
    uint64_t inputXUb_;
    uint64_t smoothUb_;
    uint64_t tempUb_;
    uint64_t outputXUb_;

    __gm__ T *xGm_;
    __gm__ int32_t *expertIdxGm_;
    __gm__ float *quantSmoothGm;

    __gm__ int8_t *expandedXGm_;
    __gm__ int32_t *expandedRowIdxGm_;
    __gm__ int32_t *expertTokensCountOrCumsumGm;

    int64_t expertTokensCountOrCumsumFlag = 0;
    int64_t dropPadMode = 0;
};

template <typename T>
__aicore__ inline void MoeV2FullLoadDynamicQuant<T>::CopyIn()
{
    pto_detail::PtoLoadVector<int32_t>(this->sortInputUb, expertIdxGm_, this->totalLength);
    pto_detail::PtoSetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);
    pto_detail::PtoFillArithProgressionInt32(
        this->sortInputUb + static_cast<uint64_t>(this->sortNum_) * sizeof(int32_t), 0, 1, this->totalLength);
}

template <typename T>
__aicore__ inline void MoeV2FullLoadDynamicQuant<T>::SortCompute()
{
    const uint64_t expertIdxUb = this->sortInputUb;
    const uint64_t rowIdxUb = this->sortInputUb + static_cast<uint64_t>(this->sortNum_) * sizeof(int32_t);

    pto_detail::PtoSortInt32AscendingUB(expertIdxUb, rowIdxUb, this->expandedExpertIdxUb_, this->expandDstToSrcRowUb_,
                                        this->sortTempUb, this->sortMergeTmpUb, this->totalLength);

    pto_detail::PtoFillArithProgressionInt32(rowIdxUb, 0, 1, this->totalLength);
    pto_detail::PtoPipeBarrier<PIPE_V>();
    pto_detail::PtoSortInt32AscendingUB(this->expandDstToSrcRowUb_, rowIdxUb, expertIdxUb, this->expandedRowIdxUb_,
                                        this->sortTempUb, this->sortMergeTmpUb, this->totalLength);
}

template <typename T>
__aicore__ inline void MoeV2FullLoadDynamicQuant<T>::CopyOutIdx()
{
    pto_detail::PtoStoreVector<int32_t>(expandedRowIdxGm_, this->expandedRowIdxUb_, this->totalLength);
}

template <typename T>
__aicore__ inline void MoeV2FullLoadDynamicQuant<T>::ComputeExpertTokenCountOrCumsum()
{
    int64_t expertNumAlign = Align(this->expertNum, sizeof(int32_t));
    pto_detail::PtoFillVector<int32_t>(this->expertTokensUb_, static_cast<int32_t>(0), expertNumAlign);
    pto_detail::PtoSetWaitFlag<HardEvent::V_S>(HardEvent::V_S);

    int32_t lastExpertId = pto_detail::PtoGetValue<int32_t>(this->expandedExpertIdxUb_, 0);
    int64_t tokenCount = 0;
    for (int64_t i = 0; i < this->totalLength; i++) {
        int32_t curExpertId = pto_detail::PtoGetValue<int32_t>(this->expandedExpertIdxUb_, i);
        tokenCount++;
        while (lastExpertId < curExpertId) {
            pto_detail::PtoSetValue<int32_t>(this->expertTokensUb_, lastExpertId, tokenCount - 1);
            if (this->expertTokensCountOrCumsumFlag == EXERPT_TOKENS_COUNT) {
                tokenCount = 1;
            }
            lastExpertId++;
        }
    }
    pto_detail::PtoSetValue<int32_t>(this->expertTokensUb_, lastExpertId, tokenCount);
    if (this->expertTokensCountOrCumsumFlag == EXERPT_TOKENS_CUMSUM) {
        lastExpertId++;
        while (lastExpertId < this->expertNum) {
            pto_detail::PtoSetValue<int32_t>(this->expertTokensUb_, lastExpertId, tokenCount);
            lastExpertId++;
        }
    }
    if (this->expertTokensCountOrCumsumFlag > 0) {
        pto_detail::PtoStoreVector<int32_t>(expertTokensCountOrCumsumGm, this->expertTokensUb_, this->expertNum);
    }
}

template <typename T>
__aicore__ inline void MoeV2FullLoadDynamicQuant<T>::CopyOutEmpty()
{}

template <typename T>
__aicore__ inline void MoeV2FullLoadDynamicQuant<T>::LoadInputRow(uint64_t inUb, int64_t row)
{
    if constexpr (IsSameType<T, float>::value) {
        pto_detail::PtoLoadVector<float>(inUb, xGm_ + row * this->cols_, this->cols_);
    } else {
        pto_detail::PtoLoadVector<T>(inUb + static_cast<uint64_t>(colsAlign) * sizeof(T), xGm_ + row * this->cols_,
                                     this->cols_);
    }
}

template <typename T>
__aicore__ inline void MoeV2FullLoadDynamicQuant<T>::StoreExpandedXRow(int64_t outIndex, uint64_t outUb)
{
    pto_detail::PtoStoreVector(expandedXGm_ + outIndex * this->cols_scale_, outUb, this->cols_scale_);
}

template <typename T>
__aicore__ inline void MoeV2FullLoadDynamicQuant<T>::QuantizeTile(uint64_t inUb, uint64_t tempUb,
                                                                  uint64_t outputPayloadUb,
                                                                  uint64_t dynamicQuantScaleUb, uint64_t smoothUb)
{
    if constexpr (!IsSameType<T, float>::value) {
        const uint64_t rawInputUb = inUb + static_cast<uint64_t>(colsAlign) * sizeof(T);
        pto_detail::PtoCastVector<float, T>(inUb, rawInputUb, this->cols_, pto::RoundMode::CAST_NONE);
    }

    if (smoothType != 0) {
        pto_detail::PtoMulElementwiseVector<float>(inUb, inUb, smoothUb, this->cols_);
    }

    pto_detail::PtoAbsVector<float>(tempUb, inUb, this->cols_);

    pto_detail::PtoReduceMaxVector(dynamicQuantScaleUb, tempUb, tempUb, this->cols_);
    pto_detail::PtoPipeBarrier<PIPE_V>();

    float maxValue = pto_detail::PtoGetValue<float>(dynamicQuantScaleUb, 0) / 127.0f;

    pto_detail::PtoFillVector<float>(dynamicQuantScaleUb, maxValue, 8);
    pto_detail::PtoFillVector<float>(tempUb, maxValue, this->cols_);
    pto_detail::PtoPipeBarrier<PIPE_V>();

    pto_detail::PtoDivVector<float>(tempUb, inUb, tempUb, this->cols_);

    pto_detail::PtoCastVector<half, float>(tempUb, tempUb, this->cols_, pto::RoundMode::CAST_TRUNC);
    pto_detail::PtoCastVector<int8_t, half>(outputPayloadUb, tempUb, this->cols_, pto::RoundMode::CAST_ROUND);
}

template <typename T>
__aicore__ inline void MoeV2FullLoadDynamicQuant<T>::Compute(uint64_t smoothUb)
{
    const uint64_t dynamicQuantScaleUb = this->outputXUb_ + static_cast<uint64_t>(this->cols_) * sizeof(int8_t);
    QuantizeTile(this->inputXUb_, this->tempUb_, this->outputXUb_, dynamicQuantScaleUb, smoothUb);
}

template <typename T>
__aicore__ inline void MoeV2FullLoadDynamicQuant<T>::CopyOutXQuant1H()
{
    int64_t curRowsStart = this->blockIdx_ * this->perCoreRows_;
    int64_t curRowsEnd = curRowsStart + this->coreRows_ - 1;
    int64_t startXRow = curRowsStart / this->k_;
    int64_t endXRow = curRowsEnd / this->k_;

    if (smoothType == 1) {
        pto_detail::PtoLoadVector<float>(this->smoothUb_, quantSmoothGm, this->cols_);
        pto_detail::PtoSetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
    }
    for (int64_t row = startXRow; row <= endXRow; row++) {
        LoadInputRow(this->inputXUb_, row);
        pto_detail::PtoSetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
        Compute(this->smoothUb_);
        pto_detail::PtoSetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);

        while (curRowsStart <= curRowsEnd && curRowsStart / this->k_ == row) {
            int32_t outIndex = pto_detail::PtoGetValue<int32_t>(this->expandedRowIdxUb_, curRowsStart);
            curRowsStart++;
            if (outIndex == -1 || (this->dropPadMode == DROPLESS_MODE && outIndex >= this->activateRows_)) {
                continue;
            }
            StoreExpandedXRow(outIndex, this->outputXUb_);
        }
        pto_detail::PtoSetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
    }
}

template <typename T>
__aicore__ inline void MoeV2FullLoadDynamicQuant<T>::Init(GM_ADDR x, GM_ADDR expertIdx, GM_ADDR expandedX,
                                                          GM_ADDR expandedRowIdx, GM_ADDR expertTokensCountOrCumsum,
                                                          GM_ADDR quantSmooth, GM_ADDR dynamicQuantScale,
                                                          GM_ADDR workspace,
                                                          const MoeInitRoutingQuantV2TilingData *tilingData,
                                                          AscendC::TPipe *tPipe)
{
    this->gatherOutTilingData_ = &(tilingData->gatherOutComputeParamsOp);
    // this->blockIdx_ = GetBlockIdx();
    this->blockIdx_ = get_block_idx() + get_subblockid() * get_block_num();
    this->k_ = tilingData->k;
    this->n_ = tilingData->n;
    this->cols_ = tilingData->cols;
    this->cols_scale_ = this->cols_ + UB_ALIGN;
    this->needCoreNum_ = this->gatherOutTilingData_->needCoreNum;
    this->perCoreRows_ = this->gatherOutTilingData_->perCoreRows;
    this->activateRows_ = this->gatherOutTilingData_->activateRows;
    if (this->blockIdx_ == this->gatherOutTilingData_->needCoreNum - 1) {
        this->coreRows_ = this->gatherOutTilingData_->lastCoreRows;
    } else {
        this->coreRows_ = this->gatherOutTilingData_->perCoreRows;
    }
    this->expertNum = tilingData->expertNum;
    this->dropPadMode = tilingData->dropPadMode;
    this->expertTokensCountOrCumsumFlag = tilingData->expertTokensCountOrCumsumFlag;

    this->tileLength = Align(tilingData->vbsComputeParamsOp.lastCorePerLoopElements, sizeof(int32_t));
    this->sortNum_ = Ceil(this->tileLength, ONE_REPEAT_SORT_NUM) * ONE_REPEAT_SORT_NUM;
    this->totalLength = tilingData->n * tilingData->k;
    this->smoothType = tilingData->smoothType;
    this->colsAlign = Align(this->cols_, sizeof(T));

    xGm_ = (__gm__ T *)x;
    expertIdxGm_ = (__gm__ int32_t *)expertIdx;

    expandedXGm_ = (__gm__ int8_t *)expandedX;
    expandedRowIdxGm_ = (__gm__ int32_t *)expandedRowIdx;
    if (this->expertTokensCountOrCumsumFlag > 0) {
        // dropless
        expertTokensCountOrCumsumGm = (__gm__ int32_t *)expertTokensCountOrCumsum;
    }
    quantSmoothGm = (__gm__ float *)quantSmooth;

    int64_t kvFactor = 2;
    int64_t sortBytes = this->sortNum_ * sizeof(int32_t) * kvFactor;
    int64_t sortScratchBytes = GetSortLen<float>(this->sortNum_) * sizeof(float);
    this->sortInputUb = 0;
    this->expandedExpertIdxUb_ = this->sortInputUb + sortBytes;
    this->expandDstToSrcRowUb_ = this->expandedExpertIdxUb_ + AlignBytes(this->sortNum_, sizeof(int32_t));
    this->expandedRowIdxUb_ = this->expandDstToSrcRowUb_ + AlignBytes(this->sortNum_, sizeof(int32_t));
    this->expertTokensUb_ = this->expandedRowIdxUb_ + AlignBytes(this->sortNum_, sizeof(int32_t));
    this->sortTempUb = this->expertTokensUb_ + AlignBytes(this->expertNum, sizeof(int32_t));
    this->sortMergeTmpUb = this->sortTempUb + sortScratchBytes;

    uint64_t baseUb = this->sortMergeTmpUb + sortScratchBytes;
    this->inputXUb_ = baseUb;
    if constexpr (IsSameType<T, float>::value) {
        this->smoothUb_ = this->inputXUb_ + AlignBytes(this->cols_, sizeof(float));
    } else {
        this->smoothUb_ = this->inputXUb_ + 2 * AlignBytes(this->cols_, sizeof(T));
    }
    this->tempUb_ = this->smoothUb_ + AlignBytes(this->cols_, sizeof(float));
    this->outputXUb_ = this->tempUb_ + AlignBytes(this->cols_, sizeof(float));
}

template <typename T>
__aicore__ inline void MoeV2FullLoadDynamicQuant<T>::Process()
{
    if (this->blockIdx_ < this->needCoreNum_) {
        CopyIn();
        SortCompute();
        if (this->blockIdx_ == 0) {
            CopyOutIdx();
        }
        if (this->blockIdx_ == this->needCoreNum_ - 1 && this->expertTokensCountOrCumsumFlag > EXERPT_TOKENS_NONE) {
            ComputeExpertTokenCountOrCumsum();
        } else {
            CopyOutEmpty();
        }
        CopyOutXQuant1H();
    }
}
} // namespace MoeInitRoutingQuantV2
#endif // MOE_V2_DYNAMIC_QUANT_FULL_LOAD_H