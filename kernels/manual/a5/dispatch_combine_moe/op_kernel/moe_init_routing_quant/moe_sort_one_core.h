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
 * \file moe_sort_one_core.h
 * \brief
 */
#ifndef INNER_MOE_SORT_ONE_CORE_H
#define INNER_MOE_SORT_ONE_CORE_H

#include "moe_mrgsort.h"
#include "moe_pto_sort.h"
#include "moe_sort_base.h"

namespace MoeInitRoutingQuant {
using namespace AscendC;
using namespace optiling;
class MoeSortOneCore : public MoeSortBase {
public:
    __aicore__ inline MoeSortOneCore(){};
    template <typename TilingData>
    __aicore__ inline void Init(GM_ADDR expertIdx, GM_ADDR expertTokensCountOrCumsum,
                                GM_ADDR expertTokensBeforeCapacity, GM_ADDR workspace, const TilingData *tilingData,
                                AscendC::TPipe *tPipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn();
    __aicore__ inline void SortCompute();
    __aicore__ inline void CopyOut();

private:
    int64_t sortNum;
    int64_t blockIdx;
};

__aicore__ inline void MoeSortOneCore::CopyIn()
{
    pto_detail::PtoLoadVector<int32_t>(this->sortInputUb, expertIdxGm, this->totalLength);
    pto_detail::PtoSetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);
    pto_detail::PtoFillArithProgressionInt32(this->sortInputUb + static_cast<uint64_t>(this->sortNum) * sizeof(int32_t),
                                             0, 1, this->sortNum);
}

__aicore__ inline void MoeSortOneCore::SortCompute()
{
    const uint64_t expertForSourceRowUb = this->sortInputUb;
    const uint64_t sourceRowUb = this->sortInputUb + static_cast<uint64_t>(this->sortNum) * sizeof(int32_t);
    const uint64_t sortedExpertUb = this->sortOutputUb;
    const uint64_t sortedRowUb = this->sortOutputUb + static_cast<uint64_t>(this->sortNum) * sizeof(int32_t);

    pto_detail::PtoSortInt32AscendingUB(expertForSourceRowUb, sourceRowUb, sortedExpertUb, sortedRowUb,
                                        this->sortTempUb, this->sortMergeTmpUb, this->totalLength);
}

__aicore__ inline void MoeSortOneCore::CopyOut()
{
    pto_detail::PtoStoreVector<int32_t>(sortedexpertIdxGm, this->sortOutputUb, this->totalLength);
    pto_detail::PtoStoreVector<int32_t>(expandDstToSrcRowGm,
                                        this->sortOutputUb + static_cast<uint64_t>(this->sortNum) * sizeof(int32_t),
                                        this->totalLength);
}

template <typename TilingData>
__aicore__ inline void MoeSortOneCore::Init(GM_ADDR expertIdx, GM_ADDR expertTokensCountOrCumsum,
                                              GM_ADDR expertTokensBeforeCapacity, GM_ADDR workspace,
                                              const TilingData *tilingData, AscendC::TPipe *tPipe)
{
    this->blockIdx = get_block_idx() + get_subblockid() * get_block_num();
    this->tileLength = Align(tilingData->vbsComputeParamsOp.lastCorePerLoopElements, sizeof(int32_t));
    this->sortNum = Ceil(this->tileLength, ONE_REPEAT_SORT_NUM) * ONE_REPEAT_SORT_NUM;
    this->totalLength = tilingData->n * tilingData->k;
    this->coreNum = tilingData->coreNum;
    this->n = tilingData->n;
    this->k = tilingData->k;
    this->expertNum = tilingData->expertNum;
    this->expertTokensCountOrCumsumFlag = tilingData->expertTokensCountOrCumsumFlag;
    this->expertTokensBeforeCapacityFlag = tilingData->expertTokensBeforeCapacityFlag;

    expertIdxGm = (__gm__ int32_t *)expertIdx;
    sortedexpertIdxGm = reinterpret_cast<__gm__ int32_t *>(workspace);
    expandDstToSrcRowGm = reinterpret_cast<__gm__ int32_t *>(workspace) + this->tileLength;

    if (this->blockIdx == this->coreNum - 1) {
        if (this->expertTokensCountOrCumsumFlag > 0) {
            expertTokensCountOrCumsumGm = (__gm__ int32_t *)expertTokensCountOrCumsum;
            InitGlobalMemory(expertTokensCountOrCumsumGm, this->expertNum, 0);
        }
        if (this->expertTokensBeforeCapacityFlag == 1) {
            expertTokensBeforeCapacityGm = (__gm__ int32_t *)expertTokensBeforeCapacity;
            InitGlobalMemory(expertTokensBeforeCapacityGm, this->expertNum, 0);
        }
    }
    int64_t kvFactor = 2;
    int64_t sortBytes = this->sortNum * sizeof(int32_t) * kvFactor;
    int64_t scratchBytes = GetSortLen<float>(this->sortNum) * sizeof(float);
    this->sortInputUb = 0;
    this->sortOutputUb = this->sortInputUb + sortBytes;
    this->sortTempUb = this->sortOutputUb + sortBytes;
    this->sortMergeTmpUb = this->sortTempUb + scratchBytes;
}

__aicore__ inline void MoeSortOneCore::Process()
{
    if (get_block_idx() + get_subblockid() * get_block_num() < 1) {
        CopyIn();
        SortCompute();
        CopyOut();
    }
    this->SyncAll();
}
} // namespace MoeInitRoutingQuant
#endif // INNER_MOE_SORT_ONE_CORE_H