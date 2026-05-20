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
 * \file moe_v2_src_to_dst_and_gather.h
 * \brief
 */
#ifndef MOE_V2_SRC_TO_DST_AND_GATHER_H
#define MOE_V2_SRC_TO_DST_AND_GATHER_H

#include "moe_v2_common.h"
#include "moe_v2_pto_sort.h"

namespace MoeInitRoutingQuantV2 {
using namespace AscendC;
using namespace optiling;
template <typename T, typename TilingData>
class MoeV2SrcToDstAndGather {
 public:
  __aicore__ inline MoeV2SrcToDstAndGather(){};
  __aicore__ inline void Init(GM_ADDR x, GM_ADDR scale, GM_ADDR expandedRowIdx, GM_ADDR expandedX,
                              GM_ADDR dynamicQuantScale, GM_ADDR workspace, const TilingData* tilingData, AscendC::TPipe* tPipe);
  __aicore__ inline void Process();

 private:
  __aicore__ inline void CopyIn(int64_t progress);
  __aicore__ inline void LoadInputTile(uint64_t inUb, int64_t srcOffset, int64_t elemNum);
  __aicore__ inline void StoreExpandedXTile(int64_t dstOffset, uint64_t outUb, int64_t elemNum);
  __aicore__ inline void CopyOut(int64_t progress);
  __aicore__ inline void CopyOutLoops(int64_t progress);
  __aicore__ inline void Compute(int32_t srcIdx, int32_t dstIdx, int32_t expertIdx);
  __aicore__ inline float ComputeMax(uint64_t inUb, uint64_t tempUb, uint64_t dynamicQuantScaleUb,
                                     int32_t srcIdx, int32_t expertIdx, int64_t j);
  __aicore__ inline void ComputeScale(uint64_t inUb, uint64_t tempUb, float scaleTemp,
                                      int64_t dstIndex, int64_t j);
  __aicore__ inline void ComputeLoops(int32_t srcIdx, int32_t dstIdx, int32_t expertIdx);

  __aicore__ inline void CopyOutRemain();
  __aicore__ inline void SyncAll();
  __aicore__ inline void AssistInit();
  __aicore__ inline void InitZeroBuffers();
  __aicore__ inline void InitPreviousCoreExpertState();
  __aicore__ inline void BindZeroBuffers();
  __aicore__ inline void ProcessRowLoops();

 private:
  uint64_t inputIdxUb;
  uint64_t outputIdxUb;
  uint64_t outTmpUb;
  uint64_t scaleOutTmpUb;
  uint64_t inputXUb;
  uint64_t smoothUb;
  uint64_t tempUb;
  uint64_t outputXUb;
  uint64_t scaleUb;

  __gm__ int32_t *expandDstToSrcRowGm;
  __gm__ int32_t *expandedRowIdxGm;
  __gm__ int32_t *expertIdxValueGm;
  __gm__ int32_t *expandedExpertIdxGm;
  __gm__ int8_t *expandedXGm;

  __gm__ T *inputXGm;
  __gm__ float *quantSmoothGm;
  __gm__ float *dynamicQuantScaleGm;
  __gm__ float *quantSrcGm;

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
  int64_t perLoopColsAlign;
  int64_t k;
  int64_t colsTileLength;
  int64_t smoothType;

  int64_t tokenCount = 0;
  int32_t lastExpertId = -1;
  int32_t lastCoreExpertId = 0;
  int32_t lastCoreExpertIdNum = 0;
};

template <typename T, typename TilingData>
__aicore__ inline void MoeV2SrcToDstAndGather<T, TilingData>::InitZeroBuffers() {
  pto_detail::PtoFillVector<int16_t>(this->outTmpUb, static_cast<int16_t>(0), this->perLoopCols);
  pto_detail::PtoFillVector<float>(this->scaleOutTmpUb, 0.0f, 8);
  pto_detail::PtoSetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
}

template <typename T, typename TilingData>
__aicore__ inline void MoeV2SrcToDstAndGather<T, TilingData>::InitPreviousCoreExpertState() {
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
__aicore__ inline void MoeV2SrcToDstAndGather<T, TilingData>::AssistInit() {
  InitZeroBuffers();
  InitPreviousCoreExpertState();
}

template <typename T, typename TilingData>
__aicore__ inline void MoeV2SrcToDstAndGather<T, TilingData>::CopyIn(int64_t progress) {
  pto_detail::PtoLoadVector<int32_t>(this->inputIdxUb, expandDstToSrcRowGm + progress * perLoopRows, currentLoopRows);
  pto_detail::PtoLoadVector<int32_t>(this->inputIdxUb + static_cast<uint64_t>(Align(currentLoopRows, sizeof(int32_t))) * sizeof(int32_t),
                                    expandedExpertIdxGm + progress * perLoopRows,
                                    currentLoopRows);
}

template <typename T, typename TilingData>
__aicore__ inline void MoeV2SrcToDstAndGather<T, TilingData>::LoadInputTile(uint64_t inUb,
                                                                            int64_t srcOffset,
                                                                            int64_t elemNum) {
  if constexpr (IsSameType<T, float>::value) {
    pto_detail::PtoLoadVector<float>(inUb, inputXGm + srcOffset, elemNum);
  } else {
    pto_detail::PtoLoadVector<T>(inUb + static_cast<uint64_t>(perLoopColsAlign) * sizeof(T),
                                 inputXGm + srcOffset,
                                 elemNum);
  }
}

template <typename T, typename TilingData>
__aicore__ inline void MoeV2SrcToDstAndGather<T, TilingData>::StoreExpandedXTile(int64_t dstOffset,
                                                                                 uint64_t outUb,
                                                                                 int64_t elemNum) {
  pto_detail::PtoStoreVector(expandedXGm + dstOffset, outUb, elemNum);
}

template <typename T, typename TilingData>
__aicore__ inline void MoeV2SrcToDstAndGather<T, TilingData>::Compute(int32_t srcIdx, int32_t dstIdx,
                                                                      int32_t expertIdx) {
  LoadInputTile(this->inputXUb, srcIdx / this->k * this->cols, this->cols);
  pto_detail::PtoSetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);

  if (smoothType == 2) {
    pto_detail::PtoLoadVector<float>(this->smoothUb, quantSmoothGm + expertIdx * this->cols, this->cols);
    pto_detail::PtoSetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
  }

  const uint64_t inUb = this->inputXUb;
  const uint64_t tempUb = this->tempUb;
  const uint64_t outputPayloadUb = this->outputXUb;
  const uint64_t dynamicQuantScaleUb = this->scaleUb;

  if constexpr (!IsSameType<T, float>::value) {
    const uint64_t rawInputUb = inUb + static_cast<uint64_t>(perLoopColsAlign) * sizeof(T);
    pto_detail::PtoCastVector<float, T>(inUb, rawInputUb, this->cols, pto::RoundMode::CAST_NONE);
    pto_detail::PtoPipeBarrier<PIPE_V>();
  }

  if (smoothType != 0) {
    pto_detail::PtoMulElementwiseVector<float>(inUb, inUb, this->smoothUb, this->cols);
    pto_detail::PtoPipeBarrier<PIPE_V>();
  }

  pto_detail::PtoAbsVector<float>(tempUb, inUb, this->cols);
  pto_detail::PtoPipeBarrier<PIPE_V>();

  pto_detail::PtoReduceMaxVector(dynamicQuantScaleUb, tempUb, tempUb, this->cols);
  pto_detail::PtoPipeBarrier<PIPE_V>();

  float maxValue = pto_detail::PtoGetValue<float>(dynamicQuantScaleUb, 0) / 127.0f;

  pto_detail::PtoFillVector<float>(dynamicQuantScaleUb, maxValue, 8);
  pto_detail::PtoFillVector<float>(tempUb, maxValue, this->cols);
  pto_detail::PtoPipeBarrier<PIPE_V>();

  pto_detail::PtoDivVector<float>(tempUb, inUb, tempUb, this->cols);
  pto_detail::PtoPipeBarrier<PIPE_V>();

  pto_detail::PtoCastVector<half, float>(tempUb, tempUb, this->cols, pto::RoundMode::CAST_TRUNC);
  pto_detail::PtoPipeBarrier<PIPE_V>();

  pto_detail::PtoCastVector<int8_t, half>(outputPayloadUb, tempUb, this->cols, pto::RoundMode::CAST_ROUND);
  pto_detail::PtoSetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);

  pto_detail::PtoStoreVector<float>(dynamicQuantScaleGm + dstIdx, dynamicQuantScaleUb, 1);
  StoreExpandedXTile(dstIdx * this->cols, outputPayloadUb, this->cols);
}

template <typename T, typename TilingData>
__aicore__ inline void MoeV2SrcToDstAndGather<T, TilingData>::CopyOut(int64_t progress) {
  int64_t length = Align(currentLoopRows, sizeof(int32_t));

  pto_detail::PtoSetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);
  if (this->lastExpertId == -1) {
    this->lastExpertId = this->lastCoreExpertId;
    this->tokenCount = this->lastCoreExpertIdNum;
  }
  for (int64_t idx = 0; idx < currentLoopRows; idx++) {
    int32_t expertIdx = pto_detail::PtoGetValue<int32_t>(this->inputIdxUb + static_cast<uint64_t>(length) * sizeof(int32_t), idx);
    pto_detail::PtoSetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
    int32_t index = 0;
    while (this->lastExpertId < expertIdx) {
      while (this->tokenCount < this->expertCapacity) {
        index = this->lastExpertId * this->expertCapacity + this->tokenCount;
        pto_detail::PtoStoreVector(expandedXGm + index * this->cols, this->outTmpUb, this->cols);
        pto_detail::PtoStoreVector(dynamicQuantScaleGm + index, this->scaleOutTmpUb, 1);
        this->tokenCount++;
      }
      this->tokenCount = 0;
      this->lastExpertId++;
    }

    if (this->tokenCount < this->expertCapacity) {
      int32_t outOffset = pto_detail::PtoGetValue<int32_t>(this->inputIdxUb, idx);
      index = expertIdx * this->expertCapacity + this->tokenCount;
      pto_detail::PtoSetValue<int32_t>(this->outputIdxUb, 0, index);
      pto_detail::PtoSetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
      pto_detail::PtoStoreVector<int32_t>(expandedRowIdxGm + outOffset, this->outputIdxUb, 1);
      Compute(outOffset, index, expertIdx);
      pto_detail::PtoSetWaitFlag<HardEvent::MTE3_S>(HardEvent::MTE3_S);
      this->tokenCount++;
    }
  }
}

template <typename T, typename TilingData>
__aicore__ inline float MoeV2SrcToDstAndGather<T, TilingData>::ComputeMax(uint64_t inUb,
                                                                          uint64_t tempUb,
                                                                          uint64_t dynamicQuantScaleUb,
                                                                          int32_t srcIdx, int32_t expertIdx,
                                                                          int64_t j) {
  LoadInputTile(inUb, srcIdx * this->cols + j * this->perLoopCols, colsTileLength);
  pto_detail::PtoSetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);

  if constexpr (!IsSameType<T, float>::value) {
    pto_detail::PtoCastVector<float, T>(inUb,
                                        inUb + static_cast<uint64_t>(perLoopColsAlign) * sizeof(T),
                                        colsTileLength,
                                        pto::RoundMode::CAST_NONE);
    pto_detail::PtoPipeBarrier<PIPE_V>();
  }

  if (smoothType != 0) {
    pto_detail::PtoLoadVector<float>(this->smoothUb,
                                     quantSmoothGm + expertIdx * this->cols + j * this->perLoopCols,
                                     colsTileLength);
    pto_detail::PtoSetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
    pto_detail::PtoMulElementwiseVector<float>(inUb, inUb, this->smoothUb, colsTileLength);
    pto_detail::PtoPipeBarrier<PIPE_V>();
  }

  pto_detail::PtoAbsVector<float>(tempUb, inUb, colsTileLength);
  pto_detail::PtoPipeBarrier<PIPE_V>();

  pto_detail::PtoReduceMaxVector(dynamicQuantScaleUb + 8 * sizeof(float), tempUb, tempUb, colsTileLength);

  pto_detail::PtoStoreVector<float>(quantSrcGm + j * this->perLoopCols, inUb, colsTileLength);
  pto_detail::PtoSetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);

  return pto_detail::PtoGetValue<float>(dynamicQuantScaleUb, 8);
}

template <typename T, typename TilingData>
__aicore__ inline void MoeV2SrcToDstAndGather<T, TilingData>::ComputeScale(uint64_t inUb,
                                                                           uint64_t tempUb,
                                                                           float scaleTemp, int64_t dstIndex,
                                                                           int64_t j) {
  pto_detail::PtoLoadVector<float>(inUb, quantSrcGm + j * this->perLoopCols, colsTileLength);
  pto_detail::PtoSetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);

  pto_detail::PtoFillVector<float>(tempUb, scaleTemp, colsTileLength);
  pto_detail::PtoPipeBarrier<PIPE_V>();

  pto_detail::PtoDivVector<float>(tempUb, inUb, tempUb, colsTileLength);
  pto_detail::PtoPipeBarrier<PIPE_V>();

  pto_detail::PtoCastVector<half, float>(tempUb, tempUb, colsTileLength, pto::RoundMode::CAST_TRUNC);
  pto_detail::PtoPipeBarrier<PIPE_V>();

  pto_detail::PtoCastVector<int8_t, half>(this->outputXUb, tempUb, colsTileLength, pto::RoundMode::CAST_ROUND);
  pto_detail::PtoSetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
  StoreExpandedXTile(dstIndex * this->cols + j * this->perLoopCols, this->outputXUb, colsTileLength);
  pto_detail::PtoSetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
}

template <typename T, typename TilingData>
__aicore__ inline void MoeV2SrcToDstAndGather<T, TilingData>::ComputeLoops(int32_t srcIdx, int32_t dstIdx,
                                                                           int32_t expertIdx) {
  uint32_t tmp = 0xFF7FFFFF;
  float reduceMax = *((float*)&tmp);
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
  pto_detail::PtoStoreVector<float>(dynamicQuantScaleGm + dstIdx, this->scaleUb, 1);

  for (int64_t j = 0; j < this->colLoops; j++) {
    colsTileLength = this->perLoopCols;
    if (j == this->colLoops - 1) {
      colsTileLength = this->lastLoopCols;
    }
    ComputeScale(this->inputXUb, this->tempUb, scaleTemp, dstIdx, j);
  }
}

template <typename T, typename TilingData>
__aicore__ inline void MoeV2SrcToDstAndGather<T, TilingData>::CopyOutLoops(int64_t progress) {
  int64_t length = Align(currentLoopRows, sizeof(int32_t));

  pto_detail::PtoSetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);
  if (this->lastExpertId == -1) {
    this->lastExpertId = this->lastCoreExpertId;
    this->tokenCount = this->lastCoreExpertIdNum;
  }
  for (int64_t idx = 0; idx < currentLoopRows; idx++) {
    int32_t expertIdx = pto_detail::PtoGetValue<int32_t>(this->inputIdxUb + static_cast<uint64_t>(length) * sizeof(int32_t), idx);
    pto_detail::PtoSetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
    int32_t index = 0;
    while (this->lastExpertId < expertIdx) {
      while (this->tokenCount < this->expertCapacity) {
        index = this->lastExpertId * this->expertCapacity + this->tokenCount;
        int64_t col = this->perLoopCols;
        pto_detail::PtoStoreVector(dynamicQuantScaleGm + index, this->scaleOutTmpUb, 1);
        for (int64_t i = 0; i < this->colLoops; i++) {
          if (i == this->colLoops - 1) {
            col = this->lastLoopCols;
          }
          pto_detail::PtoStoreVector(expandedXGm + index * this->cols + i * this->perLoopCols, this->outTmpUb, col);
          pto_detail::PtoSetWaitFlag<HardEvent::MTE3_S>(HardEvent::MTE3_S);
        }
        this->tokenCount++;
      }
      this->tokenCount = 0;
      this->lastExpertId++;
    }

    if (this->tokenCount < this->expertCapacity) {
      int32_t outOffset = pto_detail::PtoGetValue<int32_t>(this->inputIdxUb, idx);
      index = expertIdx * this->expertCapacity + this->tokenCount;
      pto_detail::PtoSetValue<int32_t>(this->outputIdxUb, 0, index);
      pto_detail::PtoSetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
      pto_detail::PtoStoreVector<int32_t>(expandedRowIdxGm + outOffset, this->outputIdxUb, 1);
      if (smoothType == 2) {
        ComputeLoops(outOffset, index, expertIdx);
      } else {
        ComputeLoops(outOffset, index, 0);
      }
      pto_detail::PtoSetWaitFlag<HardEvent::MTE3_S>(HardEvent::MTE3_S);
      this->tokenCount++;
    }
  }
}

template <typename T, typename TilingData>
__aicore__ inline void MoeV2SrcToDstAndGather<T, TilingData>::CopyOutRemain() {
  if (this->blockIdx != this->srcToDstTilingData->needCoreNum - 1) {
    return;
  }
  while (this->lastExpertId < this->expertNum) {
    while (this->tokenCount < this->expertCapacity) {
      int32_t index = this->lastExpertId * this->expertCapacity + this->tokenCount;
      int64_t col = this->perLoopCols;
      pto_detail::PtoStoreVector(dynamicQuantScaleGm + index, this->scaleOutTmpUb, 1);
      for (int64_t i = 0; i < this->colLoops; i++) {
        if (i == this->colLoops - 1) {
          col = this->lastLoopCols;
        }
        pto_detail::PtoStoreVector(expandedXGm + index * this->cols + i * this->perLoopCols, this->outTmpUb, col);
        pto_detail::PtoSetWaitFlag<HardEvent::MTE3_S>(HardEvent::MTE3_S);
      }
      this->tokenCount++;
    }
    this->tokenCount = 0;
    this->lastExpertId++;
  }
}

template <typename T, typename TilingData>
__aicore__ inline void MoeV2SrcToDstAndGather<T, TilingData>::Init(GM_ADDR x, GM_ADDR scale, GM_ADDR expandedRowIdx,
                                                                   GM_ADDR expandedX, GM_ADDR dynamicQuantScale,
                                                                   GM_ADDR workspace, const TilingData* tilingData,
                                                                   AscendC::TPipe* tPipe) {
  int64_t blockNum = GetBlockNum();
  (void)tPipe;
  this->blockIdx = get_block_idx() + get_subblockid() * get_block_num();

  this->coreNum = tilingData->coreNum;
  this->totalLength = tilingData->n * tilingData->k;
  this->srcToDstTilingData = &(tilingData->srcToDstCapacityComputeParamsOp);
  this->expertNum = tilingData->expertNum;
  this->expertCapacity = tilingData->expertCapacity;
  this->cols = tilingData->cols;
  this->k = tilingData->k;
  this->smoothType = tilingData->smoothType;

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
  this->perLoopColsAlign = Align(this->perLoopCols, sizeof(T));

  inputXGm = (__gm__ T*)x;
  quantSmoothGm = (__gm__ float*)scale;
  dynamicQuantScaleGm = (__gm__ float*)dynamicQuantScale;

  int64_t length = Align(this->totalLength, sizeof(int32_t));
  expandedRowIdxGm = (__gm__ int32_t*)expandedRowIdx;
  expandedXGm = (__gm__ int8_t*)expandedX;

  expandedExpertIdxGm = (__gm__ int32_t*)workspace + this->blockIdx * this->srcToDstTilingData->perCoreRows;
  expandDstToSrcRowGm = (__gm__ int32_t*)workspace + length + this->blockIdx * this->srcToDstTilingData->perCoreRows;
  expertIdxValueGm = (__gm__ int32_t*)workspace + length * 2;
  if (this->colLoops > 1) {
    quantSrcGm = (__gm__ float*)workspace + length * 2 + this->coreNum * 2 + this->blockIdx * this->cols;
  }

  int64_t perLoopColsAlignBytes = AlignBytes(this->perLoopCols, sizeof(T));
  perLoopColsAlignBytes =
      Max(int64_t(perLoopColsAlignBytes * sizeof(float) / sizeof(T)), int64_t(BLOCK_BYTES + BLOCK_BYTES));

  this->inputIdxUb = 0;
  this->outputIdxUb = this->inputIdxUb + AlignBytes(this->perLoopRows, sizeof(int32_t)) * 2;
  this->outTmpUb = this->outputIdxUb + AlignBytes(INT32_ONE_BLOCK_NUM, sizeof(int32_t));
  this->scaleOutTmpUb = this->outTmpUb + AlignBytes(this->perLoopCols, sizeof(int16_t));
  this->inputXUb = this->scaleOutTmpUb + BLOCK_BYTES;
  this->smoothUb = this->inputXUb + perLoopColsAlignBytes;
  this->tempUb = this->smoothUb + AlignBytes(this->perLoopCols, sizeof(float));
  this->outputXUb = this->tempUb + AlignBytes(this->perLoopCols, sizeof(float));
  this->scaleUb = this->outputXUb + AlignBytes(this->perLoopCols, sizeof(int8_t));
}

template <typename T, typename TilingData>
__aicore__ inline void MoeV2SrcToDstAndGather<T, TilingData>::BindZeroBuffers() {}

template <typename T, typename TilingData>
__aicore__ inline void MoeV2SrcToDstAndGather<T, TilingData>::ProcessRowLoops() {
  currentLoopRows = perLoopRows;
  if (colLoops > 1) {
    for (int64_t loop = 0; loop < this->rowLoops; loop++) {
      if (loop == this->rowLoops - 1) {
        currentLoopRows = lastLoopRows;
      }
      CopyIn(loop);
      CopyOutLoops(loop);
    }
  } else {
    if (smoothType == 1) {
      pto_detail::PtoLoadVector<float>(this->smoothUb, quantSmoothGm, this->cols);
      pto_detail::PtoSetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
    }
    for (int64_t loop = 0; loop < this->rowLoops; loop++) {
      if (loop == this->rowLoops - 1) {
        currentLoopRows = lastLoopRows;
      }
      CopyIn(loop);
      CopyOut(loop);
    }
  }
}

template <typename T, typename TilingData>
__aicore__ inline void MoeV2SrcToDstAndGather<T, TilingData>::Process() {
  if (this->blockIdx < this->srcToDstTilingData->needCoreNum) {
    AssistInit();
    BindZeroBuffers();
    ProcessRowLoops();
    CopyOutRemain();
  }
}
}  // namespace MoeInitRoutingQuantV2
#endif  // MOE_V2_SRC_TO_DST_AND_GATHER_H