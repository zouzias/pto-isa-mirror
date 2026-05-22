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
 * \file moe_init_routing_expert_tokens.h
 * \brief
 */
#ifndef INNER_MOE_INIT_ROUTING_EXPERT_TOKENS_H
#define INNER_MOE_INIT_ROUTING_EXPERT_TOKENS_H

#include "moe_common.h"
#include "moe_pto_sort.h"

namespace MoeInitRoutingQuant {
using namespace AscendC;
using namespace optiling;
constexpr int64_t EXPERT_ID_VALUE_NUM = 2;

class MoeExpertTokenOut {
public:
    __aicore__ inline MoeExpertTokenOut(){};
    template <typename TilingData>
    __aicore__ inline void Init(GM_ADDR expertTokensCountOrCumsum, GM_ADDR expertTokensBeforeCapacity,
                                GM_ADDR expandedRowIdx, GM_ADDR workspace, const TilingData *tilingData,
                                AscendC::TPipe *tPipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t progress);
    __aicore__ inline void Compute(int64_t progress);
    __aicore__ inline void SyncAll();
    __aicore__ inline void InitLocal();
    __aicore__ inline void GetExpertTokenCount(int32_t curExpertId);
    __aicore__ inline void AtomicStoreCountSlice(__gm__ int32_t *dstGm, int64_t offset, int64_t copyLength);
    __aicore__ inline void CopyOutTokenGm();
    __aicore__ inline void CopyOutExpertTokensCumsum(bool isTail);
    __aicore__ inline void CopyOutExpertTokensCount(bool isTail);

private:
    uint64_t inputExpertIdxUb;
    uint64_t expertTokenIdxOutUb;
    uint64_t expandedRowIdxInitUb;

    __gm__ int32_t *expertTokensCountOrCumsumGm;
    __gm__ int32_t *expertTokensBeforeCapacityGm;
    __gm__ int32_t *expandedExpertIdxGm;
    __gm__ int32_t *expertIdxValueGm;
    __gm__ int32_t *expandedRowIdxGm;

    const InnerMoeGatherOutComputeTilingData *srcToDstTilingData;

    int64_t coreNum;
    int64_t blockIdx;
    int64_t totalLength;
    int64_t currentLoopRows;
    int64_t coreRows;
    int64_t perLoopRows;
    int64_t lastLoopRows;
    int64_t expertNum;
    int64_t expertNumUbAlign;
    int64_t dropPadMode = 0;
    int64_t expertTokensCountOrCumsumFlag = 0;
    int64_t expertTokensBeforeCapacityFlag = 0;

    int64_t tokenCount = 0;
    int64_t expertIdx = 0;
    int32_t lastExpertId = -1;
    int32_t firstExpertId = -1;

    int32_t expertTokenValue = 0;
};

__aicore__ inline void MoeExpertTokenOut::InitLocal()
{
    pto_detail::PtoFillVector(this->expertTokenIdxOutUb, static_cast<int32_t>(0), this->expertNumUbAlign);

    // expandedRowIdx initialized to -1, which is used in the src_to_dst_with_capacity step.
    // use this step SyncAll to synchronize every core data
    if (this->dropPadMode == 0) {
        return;
    }
    int64_t loops = (coreRows + perLoopRows - 1) / perLoopRows;
    pto_detail::PtoFillVector(this->expandedRowIdxInitUb, static_cast<int32_t>(-1), perLoopRows);
    pto_detail::PtoSetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
    for (int64_t loop = 0; loop < loops; loop++) {
        int64_t copyLength = perLoopRows;
        if (loop == loops - 1) {
            copyLength = lastLoopRows;
        }
        pto_detail::PtoStoreVector(
            expandedRowIdxGm + this->blockIdx * this->srcToDstTilingData->perCoreRows + loop * perLoopRows,
            this->expandedRowIdxInitUb, copyLength);
    }
    pto_detail::PtoSetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
}

__aicore__ inline void MoeExpertTokenOut::CopyIn(int64_t progress)
{
    pto_detail::PtoLoadVector<int32_t>(this->inputExpertIdxUb, expandedExpertIdxGm + progress * perLoopRows,
                                       currentLoopRows);
}

__aicore__ inline void MoeExpertTokenOut::GetExpertTokenCount(int32_t curExpertId)
{
    this->tokenCount++;
    if (this->lastExpertId < curExpertId) {
        pto_detail::PtoSetValue<int32_t>(this->expertTokenIdxOutUb, this->expertIdx, this->tokenCount - 1);
        this->tokenCount = 1;
        this->expertIdx += (curExpertId - this->lastExpertId);
        while (curExpertId - this->firstExpertId + 1 > this->expertNumUbAlign) {
            pto_detail::PtoSetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
            CopyOutExpertTokensCumsum(false);
            CopyOutExpertTokensCount(false);
            pto_detail::PtoSetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
            pto_detail::PtoFillVector(this->expertTokenIdxOutUb, static_cast<int32_t>(0), this->expertNumUbAlign);
            pto_detail::PtoSetWaitFlag<HardEvent::V_S>(HardEvent::V_S);
            this->firstExpertId += this->expertNumUbAlign;
            this->expertIdx = curExpertId - this->firstExpertId;
        }
        this->lastExpertId = curExpertId;
    }
}

__aicore__ inline void MoeExpertTokenOut::Compute(int64_t progress)
{
    pto_detail::PtoSetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);
    if (this->lastExpertId == -1) {
        this->lastExpertId = pto_detail::PtoGetValue<int32_t>(this->inputExpertIdxUb, 0);
        this->firstExpertId = this->lastExpertId;
    }
    for (int64_t i = 0; i < currentLoopRows; i++) {
        int32_t expertId = pto_detail::PtoGetValue<int32_t>(this->inputExpertIdxUb, i);
        GetExpertTokenCount(expertId);
    }
    pto_detail::PtoSetValue<int32_t>(this->expertTokenIdxOutUb, this->expertIdx, this->tokenCount);
}

__aicore__ inline void MoeExpertTokenOut::AtomicStoreCountSlice(__gm__ int32_t *dstGm, int64_t offset,
                                                                int64_t copyLength)
{
    pto_detail::PtoStoreAtomicAddVector(dstGm + offset, this->expertTokenIdxOutUb, static_cast<uint32_t>(copyLength));
}

__aicore__ inline void MoeExpertTokenOut::CopyOutExpertTokensCumsum(bool isTail)
{
    if (this->dropPadMode != DROPLESS_MODE || expertTokensCountOrCumsumFlag != EXERPT_TOKENS_CUMSUM) {
        return;
    }
    int64_t copyLength = isTail ? this->lastExpertId - this->firstExpertId + 1 : this->expertNumUbAlign;
    int64_t end = this->expertNum - this->firstExpertId;
    for (int64_t i = 0; i < copyLength; i++) {
        this->expertTokenValue += pto_detail::PtoGetValue<int32_t>(this->expertTokenIdxOutUb, i);
        pto_detail::PtoSetValue<int32_t>(this->expertTokenIdxOutUb, i, this->expertTokenValue);
    }
    // if the remaining UB is sufficient, use the UB space to copy
    // otherwise, copy the calculated data first, and then copy the last tokenValue to remaining expert position
    if (isTail && end <= this->expertNumUbAlign) {
        int64_t startAlign = Min(Align(copyLength, sizeof(int32_t)), end);
        for (int64_t i = copyLength; i < startAlign; i++) {
            pto_detail::PtoSetValue<int32_t>(this->expertTokenIdxOutUb, i, this->expertTokenValue);
        }
        if (startAlign < end) {
            pto_detail::PtoFillVector(this->expertTokenIdxOutUb + static_cast<uint64_t>(startAlign) * sizeof(int32_t),
                                      this->expertTokenValue, end - startAlign);
        }
        copyLength = end;
        pto_detail::PtoSetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
    }
    AtomicStoreCountSlice(expertTokensCountOrCumsumGm, this->firstExpertId, copyLength);
    if (isTail && end > this->expertNumUbAlign) {
        int64_t remainderLength = end - copyLength;
        pto_detail::PtoSetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
        pto_detail::PtoFillVector(this->expertTokenIdxOutUb, this->expertTokenValue, this->expertNumUbAlign);
        pto_detail::PtoSetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
        int64_t loopTimes = remainderLength / this->expertNumUbAlign + 1;
        for (int64_t i = 0; i < loopTimes; i++) {
            copyLength = i == loopTimes - 1 ? remainderLength - this->expertNumUbAlign * i : this->expertNumUbAlign;
            AtomicStoreCountSlice(expertTokensCountOrCumsumGm, this->lastExpertId + 1 + this->expertNumUbAlign * i,
                                  copyLength);
        }
    }
}

__aicore__ inline void MoeExpertTokenOut::CopyOutExpertTokensCount(bool isTail)
{
    int64_t copyLength = isTail ? this->lastExpertId - this->firstExpertId + 1 : this->expertNumUbAlign;
    if (this->dropPadMode == DROP_PAD_MODE && expertTokensBeforeCapacityFlag > EXERPT_TOKENS_NONE) {
        AtomicStoreCountSlice(expertTokensBeforeCapacityGm, this->firstExpertId, copyLength);
    }
    if (this->dropPadMode == DROPLESS_MODE && expertTokensCountOrCumsumFlag == EXERPT_TOKENS_COUNT) {
        AtomicStoreCountSlice(expertTokensCountOrCumsumGm, this->firstExpertId, copyLength);
    }
}

__aicore__ inline void MoeExpertTokenOut::CopyOutTokenGm()
{
    if (this->dropPadMode == DROPLESS_MODE) {
        pto_detail::PtoSetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
        CopyOutExpertTokensCumsum(true);
        CopyOutExpertTokensCount(true);
        return;
    }
    pto_detail::PtoSetValue<int32_t>(this->expertTokenIdxOutUb, this->expertNumUbAlign, this->lastExpertId);
    pto_detail::PtoSetValue<int32_t>(this->expertTokenIdxOutUb, this->expertNumUbAlign + 1, this->tokenCount);
    pto_detail::PtoSetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
    pto_detail::PtoStoreVector(
        expertIdxValueGm + this->blockIdx * EXPERT_ID_VALUE_NUM,
        this->expertTokenIdxOutUb + static_cast<uint64_t>(this->expertNumUbAlign) * sizeof(int32_t),
        EXPERT_ID_VALUE_NUM);
    CopyOutExpertTokensCount(true);
}

__aicore__ inline void MoeExpertTokenOut::SyncAll()
{
    if (coreNum == 1) {
        return;
    }
    pto_detail::PtoSyncAll();
}

template <typename TilingData>
__aicore__ inline void MoeExpertTokenOut::Init(GM_ADDR expertTokensCountOrCumsum, GM_ADDR expertTokensBeforeCapacity,
                                               GM_ADDR expandedRowIdx, GM_ADDR workspace, const TilingData *tilingData,
                                               AscendC::TPipe *tPipe)
{
    this->blockIdx = get_block_idx() + get_subblockid() * get_block_num();
    this->coreNum = tilingData->coreNum;
    this->totalLength = tilingData->n * tilingData->k;
    this->srcToDstTilingData = &(tilingData->srcToDstComputeParamsOp);
    this->expertNum = tilingData->expertNum;
    this->dropPadMode = tilingData->dropPadMode;
    this->expertTokensCountOrCumsumFlag = tilingData->expertTokensCountOrCumsumFlag;
    this->expertTokensBeforeCapacityFlag = tilingData->expertTokensBeforeCapacityFlag;

    if (this->blockIdx == this->srcToDstTilingData->needCoreNum - 1) {
        this->coreRows = this->srcToDstTilingData->lastCoreRows;
        this->perLoopRows = this->srcToDstTilingData->lastCorePerLoopRows;
        this->lastLoopRows = this->srcToDstTilingData->lastCoreLastLoopRows;
    } else {
        this->coreRows = this->srcToDstTilingData->perCoreRows;
        this->perLoopRows = this->srcToDstTilingData->perCorePerLoopRows;
        this->lastLoopRows = this->srcToDstTilingData->perCoreLastLoopRows;
    }

    expandedRowIdxGm = (__gm__ int32_t *)expandedRowIdx;
    if (this->dropPadMode == DROPLESS_MODE && this->expertTokensCountOrCumsumFlag > EXERPT_TOKENS_NONE) {
        expertTokensCountOrCumsumGm = (__gm__ int32_t *)expertTokensCountOrCumsum;
    }
    if (this->dropPadMode == DROP_PAD_MODE && this->expertTokensBeforeCapacityFlag == EXERPT_TOKENS_BEFORE_CAPACITY) {
        expertTokensBeforeCapacityGm = (__gm__ int32_t *)expertTokensBeforeCapacity;
    }

    expandedExpertIdxGm = (__gm__ int32_t *)workspace + this->blockIdx * this->srcToDstTilingData->perCoreRows;
    expertIdxValueGm = (__gm__ int32_t *)workspace + Align(this->totalLength, sizeof(int32_t)) * 2;

    this->expertNumUbAlign = Min(Align(this->expertNum, sizeof(int32_t)), MAX_EXPERT_NUM);
    this->inputExpertIdxUb = 0;
    this->expertTokenIdxOutUb = AlignBytes(this->perLoopRows, sizeof(int32_t));
    this->expandedRowIdxInitUb =
        this->expertTokenIdxOutUb + AlignBytes(this->expertNumUbAlign + EXPERT_ID_VALUE_NUM, sizeof(int32_t));
}

__aicore__ inline void MoeExpertTokenOut::Process()
{
    if (this->blockIdx < this->srcToDstTilingData->needCoreNum) {
        int64_t loops = (coreRows + perLoopRows - 1) / perLoopRows;
        currentLoopRows = perLoopRows;
        InitLocal();
        for (int64_t loop = 0; loop < loops - 1; loop++) {
            CopyIn(loop);
            Compute(loop);
        }
        currentLoopRows = lastLoopRows;
        CopyIn(loops - 1);
        Compute(loops - 1);
        CopyOutTokenGm();
    }
    this->SyncAll();
}

class MoeSrcToDstOp {
public:
    __aicore__ inline MoeSrcToDstOp(){};
    template <typename TilingData>
    __aicore__ inline void Init(GM_ADDR expandSrcToDstRow, GM_ADDR workspace, const TilingData *tilingData,
                                AscendC::TPipe *tPipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t progress);
    __aicore__ inline void Compute(int64_t progress);
    __aicore__ inline void CopyOut();
    __aicore__ inline void SyncAll();
    __aicore__ inline void AssistInit();

private:
    uint64_t inputDstToSrcUb;
    uint64_t outputSrcToDstUb;
    uint64_t assistUb;

    __gm__ int32_t *expandDstToSrcRowGm;
    __gm__ int32_t *expandSrcToDstRowGm;
    __gm__ int32_t *assistGm;

    const InnerMoeGatherOutComputeTilingData *srcToDstTilingData;

    int64_t coreNum;
    int64_t blockIdx;
    int64_t totalLength;
    int64_t currentLoopRows;
    int64_t coreRows;
    int64_t perLoopRows;
    int64_t lastLoopRows;
};

__aicore__ inline void MoeSrcToDstOp::AssistInit()
{
#if defined(ASCENDC_OOM) && ASCENDC_OOM == 1
    OOMCheckAddrRange(assistGm, ASSIST_NUM * sizeof(int32_t));
#endif
    pto_detail::PtoLoadVector<int32_t>(this->assistUb, assistGm, ASSIST_NUM);
    pto_detail::PtoSetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
    pto_detail::PtoAddScalarVector<int32_t>(
        this->assistUb, this->assistUb, ASSIST_NUM,
        static_cast<int32_t>(this->blockIdx * this->srcToDstTilingData->perCoreRows));
}

__aicore__ inline void MoeSrcToDstOp::CopyIn(int64_t progress)
{
    pto_detail::PtoLoadVector<int32_t>(this->inputDstToSrcUb, expandDstToSrcRowGm + progress * perLoopRows,
                                       currentLoopRows);
}

__aicore__ inline void MoeSrcToDstOp::Compute(int64_t progress)
{
    pto_detail::PtoWaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    pto_detail::PtoPipeBarrier<PIPE_V>();
    int64_t loops = Ceil(currentLoopRows, ASSIST_INDEX_NUM);
    for (int64_t i = 0; i < loops; i++) {
        pto_detail::PtoAddScalarVector<int32_t>(
            this->outputSrcToDstUb + static_cast<uint64_t>(i * ASSIST_NUM) * sizeof(int32_t), this->assistUb,
            ASSIST_NUM, static_cast<int32_t>(this->perLoopRows * progress + i * ASSIST_INDEX_NUM));
    }
    pto_detail::PtoPipeBarrier<PIPE_V>();
    pto_detail::PtoSetFlag<HardEvent::V_MTE3>(EVENT_ID0);
}

__aicore__ inline void MoeSrcToDstOp::CopyOut()
{
    pto_detail::PtoWaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    pto_detail::PtoSetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);
    uint32_t outOffset;
    for (int64_t idx = 0; idx < currentLoopRows; idx++) {
        outOffset = pto_detail::PtoGetValue<int32_t>(this->inputDstToSrcUb, idx);
        pto_detail::PtoStoreVector<int32_t>(
            expandSrcToDstRowGm + outOffset,
            this->outputSrcToDstUb + static_cast<uint64_t>(idx * INT32_ONE_BLOCK_NUM) * sizeof(int32_t), 1);
    }
    pto_detail::PtoSetFlag<HardEvent::MTE3_V>(EVENT_ID0);
}

__aicore__ inline void MoeSrcToDstOp::SyncAll()
{
    if (coreNum == 1) {
        return;
    }
    pto_detail::PtoSyncAll();
}

template <typename TilingData>
__aicore__ inline void MoeSrcToDstOp::Init(GM_ADDR expandSrcToDstRow, GM_ADDR workspace, const TilingData *tilingData,
                                           AscendC::TPipe *tPipe)
{
    this->blockIdx = get_block_idx() + get_subblockid() * get_block_num();

    this->coreNum = tilingData->coreNum;
    this->totalLength = tilingData->n * tilingData->k;
    this->srcToDstTilingData = &(tilingData->srcToDstComputeParamsOp);

    if (this->blockIdx == this->srcToDstTilingData->needCoreNum - 1) {
        this->coreRows = this->srcToDstTilingData->lastCoreRows;
        this->perLoopRows = this->srcToDstTilingData->lastCorePerLoopRows;
        this->lastLoopRows = this->srcToDstTilingData->lastCoreLastLoopRows;
    } else {
        this->coreRows = this->srcToDstTilingData->perCoreRows;
        this->perLoopRows = this->srcToDstTilingData->perCorePerLoopRows;
        this->lastLoopRows = this->srcToDstTilingData->perCoreLastLoopRows;
    }

    expandSrcToDstRowGm = (__gm__ int32_t *)expandSrcToDstRow;
    expandDstToSrcRowGm = (__gm__ int32_t *)workspace + Align(this->totalLength, sizeof(int32_t)) +
                          this->blockIdx * this->srcToDstTilingData->perCoreRows;
    assistGm = (__gm__ int32_t *)assist;

    this->inputDstToSrcUb = 0;
    this->outputSrcToDstUb = AlignBytes(this->perLoopRows, sizeof(int32_t));
    this->assistUb = this->outputSrcToDstUb + Ceil(this->perLoopRows, ASSIST_NUM) * ASSIST_NUM * BLOCK_BYTES;
    pto_detail::PtoSetFlag<HardEvent::MTE3_V>(EVENT_ID0);
}

__aicore__ inline void MoeSrcToDstOp::Process()
{
    if (this->blockIdx < this->srcToDstTilingData->needCoreNum) {
        int64_t loops = (coreRows + perLoopRows - 1) / perLoopRows;
        currentLoopRows = perLoopRows;
        AssistInit();
        for (int64_t loop = 0; loop < loops - 1; loop++) {
            CopyIn(loop);
            Compute(loop);
            CopyOut();
        }
        currentLoopRows = lastLoopRows;
        CopyIn(loops - 1);
        Compute(loops - 1);
        CopyOut();
    }
    this->SyncAll();
}

} // namespace MoeInitRoutingQuant
#endif // INNER_MOE_INIT_ROUTING_EXPERT_TOKENS_H
