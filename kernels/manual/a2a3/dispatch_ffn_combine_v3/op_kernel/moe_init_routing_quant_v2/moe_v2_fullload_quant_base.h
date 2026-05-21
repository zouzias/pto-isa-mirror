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
 * \file moe_v2_fullload_quant_base.h
 * \brief
 */
#ifndef MOE_V2_FULL_LOAD_QUANT_BASE_H
#define MOE_V2_FULL_LOAD_QUANT_BASE_H

#include "kernel_operator.h"

#include "moe_v2_pto_sort.h"

namespace MoeInitRoutingQuantV2 {
using namespace AscendC;
using namespace optiling;
class MoeV2FullLoadQuantBase {
public:
    __aicore__ inline MoeV2FullLoadQuantBase(){};

protected:
    __aicore__ inline void InitBase(GM_ADDR x, GM_ADDR expertIdx, GM_ADDR expandedX, GM_ADDR expandedRowIdx,
                                    GM_ADDR expertTokensCountOrCumsum, GM_ADDR workspace,
                                    const MoeInitRoutingQuantV2TilingData *tilingData, TPipe *tPipe);
    __aicore__ inline void ProcessBase();
    __aicore__ inline void CopyIn();
    __aicore__ inline void SortCompute();
    __aicore__ inline void CopyOutIdx();
    __aicore__ inline void CopyOutEmpty();
    __aicore__ inline void ComputeExpertTokenCountOrCumsum();

protected:
    const InnerMoeV2GatherOutComputeTilingData *gatherOutTilingData;

    int64_t tileLength;
    int64_t totalLength;
    int64_t sortNum;
    int64_t blockIdx;
    int64_t needCoreNum;
    int64_t coreRows;
    int64_t perCoreRows;
    int64_t k;
    int64_t n;
    int64_t cols;
    int64_t activateRows;
    int64_t expertNum;

    uint64_t sortInputExpertIdxUb;
    uint64_t sortInputRowIdxUb;
    uint64_t expandedExpertIdxUb;
    uint64_t expandDstToSrcRowUb;
    uint64_t expandedRowIdxUb;
    uint64_t expertTokensUb;
    uint64_t packedSortUb;
    uint64_t mergeTmpUb;

    __gm__ int32_t *expertIdxGm;
    __gm__ int8_t *expandedXGm;
    __gm__ int32_t *expandedRowIdxGm;
    __gm__ int32_t *expertTokensCountOrCumsumGm;

    int64_t expertTokensCountOrCumsumFlag = 0;
    int64_t dropPadMode = 0;
    static constexpr int64_t DST_BLK_STRIDE = 1;
    static constexpr int64_t DST_REP_STRIDE = 8;
    static constexpr int64_t FOUR_BLOCK_BYTES = 128;
};

__aicore__ inline void MoeV2FullLoadQuantBase::CopyIn()
{
    pto_detail::PtoLoadVector<int32_t>(this->sortInputExpertIdxUb, expertIdxGm, this->totalLength);
    pto_detail::PtoSetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);
    pto_detail::PtoFillArithProgressionInt32(this->sortInputRowIdxUb, 0, 1, this->totalLength);
}

__aicore__ inline void MoeV2FullLoadQuantBase::SortCompute()
{
    pto_detail::PtoSortInt32AscendingUB(this->sortInputExpertIdxUb, this->sortInputRowIdxUb, this->expandedExpertIdxUb,
                                        this->expandDstToSrcRowUb, this->packedSortUb, this->mergeTmpUb,
                                        this->totalLength);

    pto_detail::PtoFillArithProgressionInt32(this->sortInputRowIdxUb, 0, 1, this->totalLength);
    pto_detail::PtoPipeBarrier<PIPE_V>();
    pto_detail::PtoSortInt32AscendingUB(this->expandDstToSrcRowUb, this->sortInputRowIdxUb, this->sortInputExpertIdxUb,
                                        this->expandedRowIdxUb, this->packedSortUb, this->mergeTmpUb,
                                        this->totalLength);
}

__aicore__ inline void MoeV2FullLoadQuantBase::CopyOutIdx()
{
    pto_detail::PtoStoreVector<int32_t>(expandedRowIdxGm, this->expandedRowIdxUb, this->totalLength);
}

__aicore__ inline void MoeV2FullLoadQuantBase::ComputeExpertTokenCountOrCumsum()
{
    int64_t expertNumAlign = Align(this->expertNum, sizeof(int32_t));
    pto_detail::PtoFillVector<int32_t>(this->expertTokensUb, static_cast<int32_t>(0), expertNumAlign);
    pto_detail::PtoSetWaitFlag<HardEvent::V_S>(HardEvent::V_S);

    int32_t lastExpertId = pto_detail::PtoGetValue<int32_t>(this->expandedExpertIdxUb, 0);
    int64_t tokenCount = 0;
    for (int64_t i = 0; i < this->totalLength; i++) {
        int32_t curExpertId = pto_detail::PtoGetValue<int32_t>(this->expandedExpertIdxUb, i);
        tokenCount++;
        while (lastExpertId < curExpertId) {
            pto_detail::PtoSetValue<int32_t>(this->expertTokensUb, lastExpertId, tokenCount - 1);
            if (this->expertTokensCountOrCumsumFlag == EXERPT_TOKENS_COUNT) {
                tokenCount = 1;
            }
            lastExpertId++;
        }
    }
    pto_detail::PtoSetValue<int32_t>(this->expertTokensUb, lastExpertId, tokenCount);
    if (this->expertTokensCountOrCumsumFlag == EXERPT_TOKENS_CUMSUM) {
        lastExpertId++;
        while (lastExpertId < this->expertNum) {
            pto_detail::PtoSetValue<int32_t>(this->expertTokensUb, lastExpertId, tokenCount);
            lastExpertId++;
        }
    }
    if (this->expertTokensCountOrCumsumFlag > 0) {
        pto_detail::PtoStoreVector<int32_t>(expertTokensCountOrCumsumGm, this->expertTokensUb, this->expertNum);
    }
}

__aicore__ inline void MoeV2FullLoadQuantBase::CopyOutEmpty()
{}

__aicore__ inline void MoeV2FullLoadQuantBase::InitBase(GM_ADDR x, GM_ADDR expertIdx, GM_ADDR expandedX,
                                                        GM_ADDR expandedRowIdx, GM_ADDR expertTokensCountOrCumsum,
                                                        GM_ADDR workspace,
                                                        const MoeInitRoutingQuantV2TilingData *tilingData, TPipe *tPipe)
{
    this->gatherOutTilingData = &(tilingData->gatherOutComputeParamsOp);
    this->blockIdx = get_block_idx() + get_subblockid() * get_block_num();
    this->k = tilingData->k;
    this->n = tilingData->n;
    this->cols = tilingData->cols;
    this->needCoreNum = this->gatherOutTilingData->needCoreNum;
    this->perCoreRows = this->gatherOutTilingData->perCoreRows;
    this->activateRows = this->gatherOutTilingData->activateRows;
    if (this->blockIdx == this->gatherOutTilingData->needCoreNum - 1) {
        this->coreRows = this->gatherOutTilingData->lastCoreRows;
    } else {
        this->coreRows = this->gatherOutTilingData->perCoreRows;
    }
    this->expertNum = tilingData->expertNum;
    this->dropPadMode = tilingData->dropPadMode;
    this->expertTokensCountOrCumsumFlag = tilingData->expertTokensCountOrCumsumFlag;

    this->tileLength = Align(tilingData->vbsComputeParamsOp.lastCorePerLoopElements, sizeof(int32_t));
    this->sortNum = Ceil(this->tileLength, ONE_REPEAT_SORT_NUM) * ONE_REPEAT_SORT_NUM;
    this->totalLength = tilingData->n * tilingData->k;

    expertIdxGm = (__gm__ int32_t *)expertIdx;

    expandedXGm = (__gm__ int8_t *)expandedX;
    expandedRowIdxGm = (__gm__ int32_t *)expandedRowIdx;
    if (this->expertTokensCountOrCumsumFlag > 0) {
        // dropless
        expertTokensCountOrCumsumGm = (__gm__ int32_t *)expertTokensCountOrCumsum;
    }

    int64_t sortBytes = AlignBytes(this->sortNum, sizeof(int32_t));
    int64_t sortScratchBytes = GetSortLen<float>(this->sortNum) * sizeof(float);
    this->sortInputExpertIdxUb = 0;
    this->sortInputRowIdxUb = this->sortInputExpertIdxUb + sortBytes;
    this->expandedExpertIdxUb = this->sortInputRowIdxUb + sortBytes;
    this->expandDstToSrcRowUb = this->expandedExpertIdxUb + sortBytes;
    this->expandedRowIdxUb = this->expandDstToSrcRowUb + sortBytes;
    this->expertTokensUb = this->expandedRowIdxUb + sortBytes;
    this->packedSortUb = this->expertTokensUb + AlignBytes(this->expertNum, sizeof(int32_t));
    this->mergeTmpUb = this->packedSortUb + sortScratchBytes;
}

__aicore__ inline void MoeV2FullLoadQuantBase::ProcessBase()
{
    if (this->blockIdx < this->needCoreNum) {
        CopyIn();
        SortCompute();
        if (this->blockIdx == 0) {
            CopyOutIdx();
        }
        if (this->blockIdx == this->needCoreNum - 1 && this->expertTokensCountOrCumsumFlag > EXERPT_TOKENS_NONE) {
            ComputeExpertTokenCountOrCumsum();
        } else {
            CopyOutEmpty();
        }
    }
}

} // namespace MoeInitRoutingQuantV2
#endif // MOE_V2_FULL_LOAD_QUANT_BASE_H