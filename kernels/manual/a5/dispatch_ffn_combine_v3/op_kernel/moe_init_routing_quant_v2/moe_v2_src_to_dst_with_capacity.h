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
 * \file moe_v2_src_to_dst_with_capacity.h
 * \brief
 */
#ifndef INNER_MOE_V2_SRC_TO_DST_WITH_CAPACITY_H
#define INNER_MOE_V2_SRC_TO_DST_WITH_CAPACITY_H

#include "moe_v2_common.h"
#include "moe_v2_pto_sort.h"

namespace MoeInitRoutingQuantV2 {
using namespace AscendC;
using namespace optiling;
template <typename T, typename TilingData>
class MoeV2SrcToDstWithCapacity {
 public:
  __aicore__ inline MoeV2SrcToDstWithCapacity(){};
  __aicore__ inline void Init(GM_ADDR expandedRowIdx, GM_ADDR expandedX, GM_ADDR workspace,
                              const TilingData* tilingData, AscendC::TPipe* tPipe);
  __aicore__ inline void Process();

 private:
  __aicore__ inline void CopyIn(int64_t progress);
  __aicore__ inline void CopyOut(int64_t progress);
  __aicore__ inline void CopyOutRemain();
  __aicore__ inline void SyncAll();
  __aicore__ inline void AssistInit();

 private:
  uint64_t inputDstToSrcUb;
  uint64_t inputExpertIdxUb;
  uint64_t outputIndexUb;
  uint64_t zeroOutputUb;

  __gm__ int32_t *expandDstToSrcRowGm;
  __gm__ int32_t *expandedRowIdxGm;
  __gm__ int32_t *expertIdxValueGm;
  __gm__ int32_t *expandedExpertIdxGm;
  __gm__ T *expandedXGm;

  const InnerMoeV2GatherOutComputeTilingData* srcToDstTilingData;

  int64_t coreNum;
  int64_t blockIdx;
  int64_t totalLength;
  int64_t currentLoopRows;
  int64_t coreRows;
  int64_t perLoopRows;
  int64_t lastLoopRows;
  int64_t rowLoops;
  int64_t expertCapacity;
  int64_t expertNum;
  int64_t cols;
  int64_t perLoopCols;
  int64_t lastLoopCols;
  int64_t colLoops;

  int64_t tokenCount = 0;
  int32_t lastExpertId = -1;
  int32_t lastCoreExpertId = 0;
  int32_t lastCoreExpertIdNum = 0;
};

template <typename T, typename TilingData>
__aicore__ inline void MoeV2SrcToDstWithCapacity<T, TilingData>::AssistInit() {
  if constexpr (IsSameType<T, int8_t>::value) {
    pto_detail::PtoFillVector<int16_t>(this->zeroOutputUb, static_cast<int16_t>(0), this->perLoopCols);
  } else {
    pto_detail::PtoFillVector<T>(this->zeroOutputUb, static_cast<T>(0), this->perLoopCols);
  }

  if (this->blockIdx != 0) {
    this->lastCoreExpertId = expertIdxValueGm[(this->blockIdx - 1) * 2];
    this->lastCoreExpertIdNum = expertIdxValueGm[(this->blockIdx - 1) * 2 + 1];
    for (int64_t i = this->blockIdx - 2; i >= 0; i--) {
      int32_t lastExpertIdx = expertIdxValueGm[i * 2];
      if (lastExpertIdx < this->lastCoreExpertId) {
        break;
      }
      int32_t lastExpertNum = expertIdxValueGm[i * 2 + 1];
      this->lastCoreExpertIdNum += lastExpertNum;
    }
  }
}

template <typename T, typename TilingData>
__aicore__ inline void MoeV2SrcToDstWithCapacity<T, TilingData>::CopyIn(int64_t progress) {
  int64_t length = Align(currentLoopRows, sizeof(int32_t));
  pto_detail::PtoLoadVector<int32_t>(this->inputDstToSrcUb, expandDstToSrcRowGm + progress * perLoopRows, length);
  pto_detail::PtoLoadVector<int32_t>(this->inputExpertIdxUb, expandedExpertIdxGm + progress * perLoopRows, length);
}

template <typename T, typename TilingData>
__aicore__ inline void MoeV2SrcToDstWithCapacity<T, TilingData>::CopyOut(int64_t progress) {
  pto_detail::PtoSetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);
  if (this->lastExpertId == -1) {
    this->lastExpertId = this->lastCoreExpertId;
    this->tokenCount = this->lastCoreExpertIdNum;
  }
  for (int64_t idx = 0; idx < currentLoopRows; idx++) {
    int32_t expertIdx = pto_detail::PtoGetValue<int32_t>(this->inputExpertIdxUb, idx);
    pto_detail::PtoSetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
    int32_t index = 0;
    while (this->lastExpertId < expertIdx) {
      while (this->tokenCount < this->expertCapacity) {
        index = this->lastExpertId * this->expertCapacity + this->tokenCount;
        int64_t col = this->perLoopCols;
        for (int64_t i = 0; i < this->colLoops; i++) {
          if (i == this->colLoops - 1) {
            col = this->lastLoopCols;
          }
          pto_detail::PtoStoreVector<T>(expandedXGm + index * this->cols + i * this->perLoopCols, this->zeroOutputUb, col);
          pto_detail::PtoSetWaitFlag<HardEvent::MTE3_S>(HardEvent::MTE3_S);
        }
        this->tokenCount++;
      }
      this->tokenCount = 0;
      this->lastExpertId++;
    }

    if (this->tokenCount < this->expertCapacity) {
      int32_t outOffset = pto_detail::PtoGetValue<int32_t>(this->inputDstToSrcUb, idx);
      index = expertIdx * this->expertCapacity + this->tokenCount;
      pto_detail::PtoSetValue<int32_t>(this->outputIndexUb, 0, index);
      pto_detail::PtoSetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
      pto_detail::PtoStoreVector<int32_t>(expandedRowIdxGm + outOffset, this->outputIndexUb, 1);
      pto_detail::PtoSetWaitFlag<HardEvent::MTE3_S>(HardEvent::MTE3_S);
      this->tokenCount++;
    }
  }
}

template <typename T, typename TilingData>
__aicore__ inline void MoeV2SrcToDstWithCapacity<T, TilingData>::CopyOutRemain() {
  if (this->blockIdx != this->srcToDstTilingData->needCoreNum - 1) {
    return;
  }
  while (this->lastExpertId < this->expertNum) {
    while (this->tokenCount < this->expertCapacity) {
      int32_t index = this->lastExpertId * this->expertCapacity + this->tokenCount;
      int64_t col = this->perLoopCols;
      for (int64_t i = 0; i < this->colLoops; i++) {
        if (i == this->colLoops - 1) {
          col = this->lastLoopCols;
        }
        pto_detail::PtoStoreVector<T>(expandedXGm + index * this->cols + i * this->perLoopCols, this->zeroOutputUb, col);
        pto_detail::PtoSetWaitFlag<HardEvent::MTE3_S>(HardEvent::MTE3_S);
      }
      this->tokenCount++;
    }
    this->tokenCount = 0;
    this->lastExpertId++;
  }
}

template <typename T, typename TilingData>
__aicore__ inline void MoeV2SrcToDstWithCapacity<T, TilingData>::SyncAll() {
  if (coreNum == 1) {
    return;
  }
  pto_detail::PtoSyncAll();
}

template <typename T, typename TilingData>
__aicore__ inline void MoeV2SrcToDstWithCapacity<T, TilingData>::Init(GM_ADDR expandedRowIdx, GM_ADDR expandedX,
                                                                      GM_ADDR workspace, const TilingData* tilingData,
                                                                      AscendC::TPipe* tPipe) {
  this->blockIdx = get_block_idx() + get_subblockid() * get_block_num();

  this->coreNum = tilingData->coreNum;
  this->totalLength = tilingData->n * tilingData->k;
  this->srcToDstTilingData = &(tilingData->srcToDstCapacityComputeParamsOp);
  this->expertNum = tilingData->expertNum;
  this->expertCapacity = tilingData->expertCapacity;
  this->cols = tilingData->cols;

  if (this->blockIdx == this->srcToDstTilingData->needCoreNum - 1) {
    this->coreRows = this->srcToDstTilingData->lastCoreRows;
    this->perLoopRows = this->srcToDstTilingData->lastCorePerLoopRows;
    this->lastLoopRows = this->srcToDstTilingData->lastCoreLastLoopRows;
    this->rowLoops = this->srcToDstTilingData->lastCoreLoops;
  } else {
    this->coreRows = this->srcToDstTilingData->perCoreRows;
    this->perLoopRows = this->srcToDstTilingData->perCorePerLoopRows;
    this->lastLoopRows = this->srcToDstTilingData->perCoreLastLoopRows;
    this->rowLoops = this->srcToDstTilingData->perCoreLoops;
  }
  this->perLoopCols = this->srcToDstTilingData->perLoopCols;
  this->lastLoopCols = this->srcToDstTilingData->lastLoopCols;
  this->colLoops = this->srcToDstTilingData->colLoops;

  int64_t length = Align(this->totalLength, sizeof(int32_t));
  expandedRowIdxGm = (__gm__ int32_t*)expandedRowIdx;
  expandedXGm = (__gm__ T*)expandedX;

  expandedExpertIdxGm = (__gm__ int32_t*)workspace + this->blockIdx * this->srcToDstTilingData->perCoreRows;
  expandDstToSrcRowGm = (__gm__ int32_t*)workspace + length + this->blockIdx * this->srcToDstTilingData->perCoreRows;
  expertIdxValueGm = (__gm__ int32_t*)workspace + length * 2;

  int64_t inputBytes = AlignBytes(this->perLoopRows, sizeof(int32_t));
  this->inputDstToSrcUb = 0;
  this->inputExpertIdxUb = inputBytes;
  this->outputIndexUb = this->inputExpertIdxUb + inputBytes;
  this->zeroOutputUb = this->outputIndexUb + AlignBytes(INT32_ONE_BLOCK_NUM, sizeof(int32_t));
}

template <typename T, typename TilingData>
__aicore__ inline void MoeV2SrcToDstWithCapacity<T, TilingData>::Process() {
  if (this->blockIdx < this->srcToDstTilingData->needCoreNum) {
    AssistInit();
    currentLoopRows = perLoopRows;
    for (int64_t loop = 0; loop < this->rowLoops; loop++) {
      if (loop == this->rowLoops - 1) {
        currentLoopRows = lastLoopRows;
      }
      CopyIn(loop);
      CopyOut(loop);
    }
    CopyOutRemain();
  }
  this->SyncAll();
}
}  // namespace MoeInitRoutingQuantV2
#endif  // INNER_MOE_V2_SRC_TO_DST_WITH_CAPACITY_H
