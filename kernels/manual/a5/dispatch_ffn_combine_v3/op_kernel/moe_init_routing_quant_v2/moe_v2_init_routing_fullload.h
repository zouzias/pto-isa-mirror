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
 * \file moe_v2_init_routing_fullload.h
 * \brief
 */
#ifndef INNER_MOE_V2_FULL_LOAD_H
#define INNER_MOE_V2_FULL_LOAD_H

#include "moe_v2_mrgsort.h"
#include "moe_v2_pto_sort.h"

namespace MoeInitRoutingQuantV2 {
using namespace AscendC;
using namespace optiling;
template <typename T>
class MoeV2FullLoad : public MoeV2SortBase {
 public:
  __aicore__ inline MoeV2FullLoad(){};
  __aicore__ inline void Init(GM_ADDR x, GM_ADDR expertIdx, GM_ADDR expandedX, GM_ADDR expandedRowIdx,
                              GM_ADDR expertTokensCountOrCumsum, GM_ADDR workspace,
                              const InnerMoeInitRoutingV2TilingData* tilingData, AscendC::TPipe* tPipe);
  __aicore__ inline void Process();

 private:
  __aicore__ inline void CopyIn();
  __aicore__ inline void SortCompute();
  __aicore__ inline void CopyOutIdx();
  __aicore__ inline void CopyOutEmpty();
  __aicore__ inline void CopyOutX();
  __aicore__ inline void ComputeExpertTokenCountOrCumsum();

 private:
  int64_t sortNum_;
  const InnerMoeV2GatherOutComputeTilingData* gatherOutTilingData_;
  int64_t blockIdx_;
  int64_t needCoreNum_;
  int64_t coreRows_;
  int64_t perCoreRows_;
  int64_t k_;
  int64_t n_;
  int64_t cols_;
  int64_t activateRows_;
  int64_t expertNum;

  uint64_t expandedRowIdxUb_;
  uint64_t expandedExpertIdxUb_;
  uint64_t expandDstToSrcRowUb_;
  uint64_t expertTokensUb_;
  uint64_t inputXUb_;

  __gm__ T *xGm_;
  __gm__ int32_t *expertIdxGm_;

  __gm__ T *expandedXGm_;
  __gm__ int32_t *expandedRowIdxGm_;
  __gm__ int32_t *expertTokensCountOrCumsumGm;

  int64_t expertTokensCountOrCumsumFlag = 0;
  int64_t dropPadMode = 0;
};

template <typename T>
__aicore__ inline void MoeV2FullLoad<T>::CopyIn() {
  pto_detail::PtoLoadVector<int32_t>(this->sortInputUb, expertIdxGm_, this->totalLength);
  pto_detail::PtoSetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);
  pto_detail::PtoFillArithProgressionInt32(this->sortInputUb + static_cast<uint64_t>(this->sortNum_) * sizeof(int32_t),
                                           0,
                                           1,
                                           this->totalLength);
}

template <typename T>
__aicore__ inline void MoeV2FullLoad<T>::SortCompute() {
  const uint64_t expertIdxUb = this->sortInputUb;
  const uint64_t rowIdxUb = this->sortInputUb + static_cast<uint64_t>(this->sortNum_) * sizeof(int32_t);

  pto_detail::PtoSortInt32AscendingUB(expertIdxUb,
                                      rowIdxUb,
                                      this->expandedExpertIdxUb_,
                                      this->expandDstToSrcRowUb_,
                                      this->sortTempUb,
                                      this->sortMergeTmpUb,
                                      this->totalLength);

  pto_detail::PtoFillArithProgressionInt32(rowIdxUb, 0, 1, this->totalLength);
  pto_detail::PtoPipeBarrier<PIPE_V>();
  pto_detail::PtoSortInt32AscendingUB(this->expandDstToSrcRowUb_,
                                      rowIdxUb,
                                      expertIdxUb,
                                      this->expandedRowIdxUb_,
                                      this->sortTempUb,
                                      this->sortMergeTmpUb,
                                      this->totalLength);
}

template <typename T>
__aicore__ inline void MoeV2FullLoad<T>::CopyOutIdx() {
  pto_detail::PtoStoreVector<int32_t>(expandedRowIdxGm_, this->expandedRowIdxUb_, this->totalLength);
}

template <typename T>
__aicore__ inline void MoeV2FullLoad<T>::ComputeExpertTokenCountOrCumsum() {
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
__aicore__ inline void MoeV2FullLoad<T>::CopyOutX() {
  int64_t inFactor = Align(this->cols_, sizeof(T));
  int64_t curRowsStart = this->blockIdx_ * this->perCoreRows_;
  int64_t startXRow = curRowsStart / this->k_;
  int64_t endXRow = (curRowsStart + this->coreRows_ - 1) / this->k_;

  for (int64_t row = startXRow; row <= endXRow; row++) {
    pto_detail::PtoLoadVector<T>(this->inputXUb_ + static_cast<uint64_t>((row - startXRow) * inFactor) * sizeof(T),
                                 xGm_ + row * this->cols_,
                                 this->cols_);
  }
  pto_detail::PtoSetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);

  int64_t k = 0;
  for (int64_t i = startXRow; i <= endXRow; i++) {
    for (; k < this->perCoreRows_ && curRowsStart / this->k_ == i; curRowsStart++, k++) {
      int32_t outIndex = pto_detail::PtoGetValue<int32_t>(this->expandedRowIdxUb_, curRowsStart);
      if (outIndex < this->activateRows_) {
        pto_detail::PtoStoreVector<T>(expandedXGm_ + outIndex * this->cols_,
                                      this->inputXUb_ + static_cast<uint64_t>((i - startXRow) * inFactor) * sizeof(T),
                                      this->cols_);
      }
    }
  }
  pto_detail::PtoSetWaitFlag<HardEvent::MTE3_S>(HardEvent::MTE3_S);
}

template <typename T>
__aicore__ inline void MoeV2FullLoad<T>::CopyOutEmpty() {}

template <typename T>
__aicore__ inline void MoeV2FullLoad<T>::Init(GM_ADDR x, GM_ADDR expertIdx, GM_ADDR expandedX, GM_ADDR expandedRowIdx,
                                              GM_ADDR expertTokensCountOrCumsum, GM_ADDR workspace,
                                              const InnerMoeInitRoutingV2TilingData* tilingData, AscendC::TPipe* tPipe) {
  this->gatherOutTilingData_ = &(tilingData->gatherOutComputeParamsOp);
  this->blockIdx_ = get_block_idx() + get_subblockid() * get_block_num();
  this->k_ = tilingData->k;
  this->n_ = tilingData->n;
  this->cols_ = tilingData->cols;
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

  xGm_ = (__gm__ T*)x;
  expertIdxGm_ = (__gm__ int32_t*)expertIdx;

  expandedXGm_ = (__gm__ T*)expandedX;
  expandedRowIdxGm_ = (__gm__ int32_t*)expandedRowIdx;
  if (this->expertTokensCountOrCumsumFlag > 0) {
    // dropless
    expertTokensCountOrCumsumGm = (__gm__ int32_t*)expertTokensCountOrCumsum;
  }

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

  int64_t curRowsStart = this->blockIdx_ * this->perCoreRows_;
  int64_t startXRow = curRowsStart / this->k_;
  int64_t endXRow = (curRowsStart + this->coreRows_ - 1) / this->k_;
  this->inputXUb_ = this->sortMergeTmpUb + sortScratchBytes;
}

template <typename T>
__aicore__ inline void MoeV2FullLoad<T>::Process() {
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
    CopyOutX();
  }
}
}  // namespace MoeInitRoutingQuantV2
#endif  // INNER_MOE_V2_FULL_LOAD_H