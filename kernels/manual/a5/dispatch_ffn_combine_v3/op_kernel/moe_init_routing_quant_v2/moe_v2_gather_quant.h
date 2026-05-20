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
 * \file moe_v2_gather_quant.h
 * \brief
 */
#ifndef MOE_V2_GATHER_QUANT_H
#define MOE_V2_GATHER_QUANT_H

#include "moe_v2_common.h"
#include "kernel_operator.h"
#include "moe_v2_pto_sort.h"

namespace MoeInitRoutingQuantV2 {
using namespace AscendC;
using namespace optiling;
constexpr int64_t BUFFER_NUM = 2;

template <typename T>
class MoeV2GatherQuant {
 public:
  __aicore__ inline MoeV2GatherQuant(){};
  __aicore__ inline void Init(GM_ADDR inputX, GM_ADDR scale, GM_ADDR offset, GM_ADDR expandedRowIdx, GM_ADDR expandedX,
                              GM_ADDR workspace, const MoeInitRoutingQuantV2TilingData* tilingData, AscendC::TPipe* tPipe);
  __aicore__ inline void Process();

 private:
  __aicore__ inline void CopyInIndices(int64_t progress);
  __aicore__ inline void Compute();
  __aicore__ inline void CopyOut(int64_t progress);

 private:
  uint64_t indicesUbOffset;
  uint64_t inputXUbOffset;
  uint64_t outputXUbOffset;
  uint64_t floatUbOffset;
  uint64_t halfUbOffset;

  __gm__ T *inputXGm;
  __gm__ int8_t *expandedXGm;
  __gm__ int32_t *expandedRowIdxGm;
  __gm__ float *scaleGm;
  __gm__ float *offsetGm;

  const InnerMoeV2GatherOutComputeTilingData* gatherOutTilingData;

  int64_t needCoreNum;
  int64_t blockIdx;
  int64_t cols;
  int64_t n;
  int64_t k;
  int64_t activateRows;
  int64_t currentLoopRows;
  int64_t coreRows;
  int64_t perLoopRows;
  int64_t lastLoopRows;
  int64_t rowLoops;
  int64_t colsTileLength;
  int64_t perLoopCols;
  int64_t lastLoopCols;
  int64_t colLoops;
  int64_t dropPadMode;
  float scale;
  float offset;

  int64_t indicesOffset;
  int64_t inputOffset;
  int64_t outOffset;
};

template <typename T>
__aicore__ inline void MoeV2GatherQuant<T>::CopyInIndices(int64_t progress) {
  this->indicesOffset = progress * this->perLoopRows;
  pto_detail::PtoLoadVector<int32_t>(this->indicesUbOffset,
                                     expandedRowIdxGm + indicesOffset,
                                     this->currentLoopRows);
}

template <typename T>
__aicore__ inline void MoeV2GatherQuant<T>::Compute() {
  uint32_t elements = Align(this->colsTileLength, sizeof(T));
  if constexpr (IsSameType<T, bfloat16_t>::value) {
    pto_detail::PtoCastVector<float, T>(this->floatUbOffset, this->inputXUbOffset, elements, pto::RoundMode::CAST_NONE);
    pto_detail::PtoCastVector<half, float>(this->halfUbOffset, this->floatUbOffset, elements, pto::RoundMode::CAST_NONE);
    pto_detail::PtoMulVector<half>(this->halfUbOffset, this->halfUbOffset, elements, static_cast<half>(this->scale));
    pto_detail::PtoAddScalarVector<half>(this->halfUbOffset, this->halfUbOffset, elements, static_cast<half>(this->offset));
    pto_detail::PtoCastVector<int32_t, half>(this->floatUbOffset, this->halfUbOffset, elements, pto::RoundMode::CAST_RINT);
    SetDeqScale((half)1.000000e+00f);
    pto_detail::PtoPipeBarrier<PIPE_V>();
    pto_detail::PtoCastVector<half, int32_t>(this->halfUbOffset, this->floatUbOffset, elements, pto::RoundMode::CAST_RINT);
    pto_detail::PtoCastVector<int8_t, half>(this->outputXUbOffset, this->halfUbOffset, elements, pto::RoundMode::CAST_RINT);
  } else if constexpr (IsSameType<T, float>::value) {
    pto_detail::PtoCastVector<half, T>(this->halfUbOffset, this->inputXUbOffset, elements, pto::RoundMode::CAST_NONE);
    pto_detail::PtoMulVector<half>(this->halfUbOffset, this->halfUbOffset, elements, static_cast<half>(this->scale));
    pto_detail::PtoAddScalarVector<half>(this->halfUbOffset, this->halfUbOffset, elements, static_cast<half>(this->offset));
    pto_detail::PtoCastVector<int8_t, half>(this->outputXUbOffset, this->halfUbOffset, elements, pto::RoundMode::CAST_RINT);
  } else {
    pto_detail::PtoMulVector<T>(this->inputXUbOffset, this->inputXUbOffset, elements, static_cast<T>(this->scale));
    pto_detail::PtoAddScalarVector<T>(this->inputXUbOffset, this->inputXUbOffset, elements, static_cast<T>(this->offset));
    pto_detail::PtoCastVector<int8_t, T>(this->outputXUbOffset, this->inputXUbOffset, elements, pto::RoundMode::CAST_RINT);
  }
}

template <typename T>
__aicore__ inline void MoeV2GatherQuant<T>::CopyOut(int64_t progress) {
  pto_detail::PtoSetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);
  colsTileLength = this->perLoopCols;
  for (int64_t colsLoop = 0; colsLoop < this->colLoops; colsLoop++) {
    int64_t initialRow = this->gatherOutTilingData->perCoreRows * this->blockIdx + this->perLoopRows * progress;
    int64_t curLoopRow = 0;
    if (colsLoop == this->colLoops - 1) {
      colsTileLength = this->lastLoopCols;
    }
    int64_t currentLoopStartRow = initialRow / this->k;
    int64_t currentLoopLastRow = (initialRow + this->currentLoopRows - 1) / this->k;
    for (int64_t row = currentLoopStartRow; row <= currentLoopLastRow; row++) {
      // input row position
      inputOffset = row * this->cols + colsLoop * this->perLoopCols;
      pto_detail::PtoLoadVector<T>(this->inputXUbOffset, inputXGm + inputOffset, this->colsTileLength);
      pto_detail::PtoSetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
      Compute();
      pto_detail::PtoSetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
      while (curLoopRow < this->currentLoopRows && initialRow / this->k == row) {
        int32_t outIndex = pto_detail::PtoGetValue<int32_t>(this->indicesUbOffset, curLoopRow);
        curLoopRow++;
        initialRow++;
        if (outIndex == -1 || (this->dropPadMode == DROPLESS_MODE && outIndex >= this->activateRows)) {
          continue;
        }
        outOffset = outIndex * cols + colsLoop * this->perLoopCols;
        pto_detail::PtoStoreVector<int8_t>(expandedXGm + outOffset, this->outputXUbOffset, this->colsTileLength);
      }
      pto_detail::PtoSetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
    }
  }
}

template <typename T>
__aicore__ inline void MoeV2GatherQuant<T>::Init(GM_ADDR inputX, GM_ADDR scale, GM_ADDR offset, GM_ADDR expandedRowIdx,
                                                 GM_ADDR expandedX, GM_ADDR workspace,
                                                 const MoeInitRoutingQuantV2TilingData* tilingData, AscendC::TPipe* tPipe) {
  this->blockIdx = get_block_idx() + get_subblockid() * get_block_num();
  this->gatherOutTilingData = &(tilingData->gatherOutComputeParamsOp);

  this->needCoreNum = this->gatherOutTilingData->needCoreNum;
  this->activateRows = this->gatherOutTilingData->activateRows;
  this->cols = tilingData->cols;
  this->n = tilingData->n;
  this->k = tilingData->k;
  this->dropPadMode = tilingData->dropPadMode;

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

  inputXGm = (__gm__ T*)inputX;
  expandedXGm = (__gm__ int8_t*)expandedX;
  expandedRowIdxGm = (__gm__ int32_t*)expandedRowIdx + this->blockIdx * this->gatherOutTilingData->perCoreRows;
  scaleGm = (__gm__ float*)scale;
  offsetGm = (__gm__ float*)offset;
  this->scale = scaleGm[0];
  this->offset = offsetGm[0];

  this->indicesUbOffset = 0;
  this->inputXUbOffset = AlignBytes(this->perLoopRows, sizeof(int32_t));
  this->outputXUbOffset = this->inputXUbOffset + AlignBytes(this->perLoopCols, sizeof(T));
  this->floatUbOffset = this->outputXUbOffset + AlignBytes(this->perLoopCols, sizeof(int8_t));
  this->halfUbOffset = this->floatUbOffset + AlignBytes(this->perLoopCols, sizeof(float));
}

template <typename T>
__aicore__ inline void MoeV2GatherQuant<T>::Process() {
  if (this->blockIdx < this->needCoreNum) {
    currentLoopRows = perLoopRows;
    for (int64_t loop = 0; loop < this->rowLoops; loop++) {
      if (loop == this->rowLoops - 1) {
        currentLoopRows = lastLoopRows;
      }
      CopyInIndices(loop);
      CopyOut(loop);
    }
  }
}
}  // namespace MoeInitRoutingQuantV2
#endif  // MOE_V2_GATHER_QUANT_H
