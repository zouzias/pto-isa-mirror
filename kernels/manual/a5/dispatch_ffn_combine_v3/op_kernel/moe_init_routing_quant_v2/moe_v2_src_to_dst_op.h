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
 * \file moe_v2_src_to_dst_op.h
 * \brief
 */
#ifndef INNER_MOE_V2_SRC_TO_DST_H
#define INNER_MOE_V2_SRC_TO_DST_H

#include "moe_v2_common.h"
#include "moe_v2_pto_sort.h"

namespace MoeInitRoutingQuantV2 {
using namespace AscendC;
using namespace optiling;
class MoeV2SrcToDstOp {
public:
    __aicore__ inline MoeV2SrcToDstOp(){};
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

    const InnerMoeV2GatherOutComputeTilingData *srcToDstTilingData;

    int64_t coreNum;
    int64_t blockIdx;
    int64_t totalLength;
    int64_t currentLoopRows;
    int64_t coreRows;
    int64_t perLoopRows;
    int64_t lastLoopRows;
};

__aicore__ inline void MoeV2SrcToDstOp::AssistInit()
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

__aicore__ inline void MoeV2SrcToDstOp::CopyIn(int64_t progress)
{
    pto_detail::PtoLoadVector<int32_t>(this->inputDstToSrcUb, expandDstToSrcRowGm + progress * perLoopRows,
                                       currentLoopRows);
}

__aicore__ inline void MoeV2SrcToDstOp::Compute(int64_t progress)
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

__aicore__ inline void MoeV2SrcToDstOp::CopyOut()
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

__aicore__ inline void MoeV2SrcToDstOp::SyncAll()
{
    if (coreNum == 1) {
        return;
    }
    pto_detail::PtoSyncAll();
}

template <typename TilingData>
__aicore__ inline void MoeV2SrcToDstOp::Init(GM_ADDR expandSrcToDstRow, GM_ADDR workspace, const TilingData *tilingData,
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

__aicore__ inline void MoeV2SrcToDstOp::Process()
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
} // namespace MoeInitRoutingQuantV2
#endif // INNER_MOE_V2_SRC_TO_DST_H