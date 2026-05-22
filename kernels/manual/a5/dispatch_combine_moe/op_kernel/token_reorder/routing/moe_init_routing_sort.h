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
 * \file moe_init_routing_sort.h
 * \brief
 */
#ifndef INNER_MOE_INIT_ROUTING_SORT_H
#define INNER_MOE_INIT_ROUTING_SORT_H

#include "kernel_operator.h"
#include "moe_packed_sort_merge.h"
#include "moe_pto_sort.h"

namespace MoeInitRoutingQuant {
using namespace AscendC;
using namespace optiling;

class MoeSortBase {
public:
    __aicore__ inline MoeSortBase(){};

protected:
    __aicore__ inline void SyncAll();

protected:
    uint64_t sortInputUb;
    uint64_t sortOutputUb;
    uint64_t sortTempUb;
    uint64_t sortMergeTmpUb;

    __gm__ int32_t *expertIdxGm;
    __gm__ int32_t *sortedexpertIdxGm;
    __gm__ int32_t *expandDstToSrcRowGm;
    __gm__ int32_t *expertTokensCountOrCumsumGm;
    __gm__ int32_t *expertTokensBeforeCapacityGm;

    int64_t tileLength;
    int64_t totalLength;
    int64_t coreNum;
    int64_t n;
    int64_t k;
    int64_t existRowIdx;
    int64_t expertNum;
    int64_t expertTokensCountOrCumsumFlag = 0;
    int64_t expertTokensBeforeCapacityFlag = 0;

    static constexpr int64_t SYNC_GM_NUM = 2;
    static constexpr int64_t WORK_GM_NUM = 2;
    static constexpr int64_t DST_BLK_STRIDE = 1;
    static constexpr int64_t DST_REP_STRIDE = 8;
};

__aicore__ inline void MoeSortBase::SyncAll()
{
    if (coreNum == 1) {
        return;
    }
    pto_detail::PtoSyncAll();
}

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
    PtoFillArithProgressionInt32(this->sortInputUb + static_cast<uint64_t>(this->sortNum) * sizeof(int32_t), 0, 1,
                                 this->sortNum);
}

__aicore__ inline void MoeSortOneCore::SortCompute()
{
    const uint64_t expertForSourceRowUb = this->sortInputUb;
    const uint64_t sourceRowUb = this->sortInputUb + static_cast<uint64_t>(this->sortNum) * sizeof(int32_t);
    const uint64_t sortedExpertUb = this->sortOutputUb;
    const uint64_t sortedRowUb = this->sortOutputUb + static_cast<uint64_t>(this->sortNum) * sizeof(int32_t);

    PtoSortInt32AscendingUB(expertForSourceRowUb, sourceRowUb, sortedExpertUb, sortedRowUb, this->sortTempUb,
                            this->sortMergeTmpUb, this->totalLength);
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

class MoeSortMultiCore : public MoeSortBase {
public:
    __aicore__ inline MoeSortMultiCore(){};
    template <typename TilingData>
    __aicore__ inline void Init(GM_ADDR expertIdx, GM_ADDR expertTokensCountOrCumsum,
                                GM_ADDR expertTokensBeforeCapacity, GM_ADDR workspace, const TilingData *tilingData,
                                AscendC::TPipe *tPipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void VBSProcess();
    __aicore__ inline void UBSortProcess(int64_t progress, int64_t size, int64_t sortNum);
    __aicore__ inline void OneCoreVMSProcess(int64_t listNum, int64_t perListElements, int64_t lastListElements);
    __aicore__ inline void VMSProcess();
    __aicore__ inline void SortOutProcess();
    __aicore__ inline void VBSCopyIn(int64_t progress, int64_t size, int64_t sortNum);
    __aicore__ inline void UBSortCompute(int64_t progress, int64_t size, int64_t sortNum);
    __aicore__ inline void VBSCopyOut(int64_t progress, int64_t size, int64_t sortNum);
    __aicore__ inline void RunMoeMrgSort(MoeMrgsort *sorter, int64_t listNum, int64_t coreOffset, int64_t loopOffset);
    __aicore__ inline void RunMoeMrgSortOut(MoeMrgsortOut *sorter, int64_t listNum, int64_t coreOffset);
    __aicore__ inline void InitExpertTokensGlobalMemory();

private:
    __gm__ float *workspaceGms[2];

    const InnerMoeVBSComputeTilingData *vbsTilingData;
    const InnerMoeVMSMiddleComputeTilingData *vmsTilingData;
    const InnerMoeSortOutComputeTilingData *sortOutTilingData;

    // for MoeMrgsort
    MoeMrgsort mrgsorter;
    MoeMrgsortParam mrgsortParam;

    int64_t coreNum;
    int64_t blockIdx;
    int64_t srcWsIndex = 0;

    int64_t listNum;
    int64_t perListElements;
    int64_t lastListElements;

    int64_t sortTotalLength;
    int64_t sortCoreLoops;
    int64_t sortCoreLoopElements;
    int64_t sortCoreLastLoopElements;

    int64_t perCoreExpert;
    int64_t needInitExpertCore;
    int64_t currentCoreExpert;

    static constexpr int64_t MAX_MRGSORT_LIST = 4;
};

__aicore__ inline void MoeSortMultiCore::InitExpertTokensGlobalMemory()
{
    if (this->blockIdx < this->needInitExpertCore) {
        if (this->expertTokensCountOrCumsumFlag > EXERPT_TOKENS_NONE) {
            InitGlobalMemory(expertTokensCountOrCumsumGm, currentCoreExpert, 0);
        }
        if (this->expertTokensBeforeCapacityFlag == EXERPT_TOKENS_BEFORE_CAPACITY) {
            InitGlobalMemory(expertTokensBeforeCapacityGm, currentCoreExpert, 0);
        }
    }
}

__aicore__ inline void MoeSortMultiCore::VBSCopyIn(int64_t progress, int64_t size, int64_t sortNum)
{
    int64_t inOffset = progress * sortCoreLoopElements;
    pto_detail::PtoLoadVector<int32_t>(this->sortInputUb, expertIdxGm + inOffset, size);

    int64_t startValue = this->blockIdx * this->vbsTilingData->perCoreElements + inOffset;
    pto_detail::PtoSetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);
    PtoFillArithProgressionInt32(this->sortInputUb + static_cast<uint64_t>(sortNum) * sizeof(int32_t),
                                 static_cast<int32_t>(startValue), 1, size);
}

__aicore__ inline void MoeSortMultiCore::UBSortCompute(int64_t progress, int64_t size, int64_t sortNum)
{
    const uint64_t expertForSourceRowUb = this->sortInputUb;
    const uint64_t sourceRowUb = this->sortInputUb + static_cast<uint64_t>(sortNum) * sizeof(int32_t);

    PtoSortInt32ToPackedUB(expertForSourceRowUb, sourceRowUb, this->sortOutputUb, this->sortMergeTmpUb, size);
}

__aicore__ inline void MoeSortMultiCore::VBSCopyOut(int64_t progress, int64_t size, int64_t sortNum)
{
    pto_detail::PtoStoreVector<float>(workspaceGms[0] +
                                          this->blockIdx * GetSortLen<float>(this->vbsTilingData->perCoreElements) +
                                          GetSortLen<float>(progress * sortCoreLoopElements),
                                      this->sortOutputUb, GetSortLen<float>(size));
}

__aicore__ inline void MoeSortMultiCore::RunMoeMrgSort(MoeMrgsort *sorter, int64_t listNum, int64_t coreOffset,
                                                       int64_t loopOffset)
{
    __gm__ float *srcWsGm = workspaceGms[srcWsIndex] + blockIdx * coreOffset + loopOffset;
    const uint64_t listStrideBytes =
        static_cast<uint64_t>(GetSortLen<float>(this->sortOutTilingData->oneLoopMaxElements)) * sizeof(float);
    for (int64_t i = 0; i < listNum; i++) {
        sorter->SetInput(srcWsGm, this->sortInputUb + listStrideBytes * i);
    }
    __gm__ float *dstWsGm = workspaceGms[1 - srcWsIndex] + blockIdx * coreOffset + loopOffset;
    sorter->SetOutput(dstWsGm, this->sortOutputUb);
    sorter->SetBuffer(this->sortMergeTmpUb);
    sorter->Init(&mrgsortParam);
    sorter->Process();
}

__aicore__ inline void MoeSortMultiCore::RunMoeMrgSortOut(MoeMrgsortOut *sorter, int64_t listNum, int64_t coreOffset)
{
    __gm__ float *srcWsGm = workspaceGms[srcWsIndex];

    const uint64_t listStrideBytes =
        static_cast<uint64_t>(GetSortLen<float>(this->sortOutTilingData->oneLoopMaxElements)) * sizeof(float);
    for (int64_t i = 0; i < listNum; i++) {
        sorter->SetInput(srcWsGm, this->sortInputUb + listStrideBytes * i);
    }

    const uint64_t outPayloadUb =
        this->sortOutputUb +
        static_cast<uint64_t>(this->sortOutTilingData->oneLoopMaxElements * MAX_MRGSORT_LIST) * sizeof(float);
    sorter->SetOutput(this->sortedexpertIdxGm, this->expandDstToSrcRowGm, this->sortOutputUb, outPayloadUb);

    sorter->SetBuffer(this->sortTempUb, this->sortOutputUb);
    sorter->Init(&mrgsortParam);
    sorter->Process();
}

__aicore__ inline void MoeSortMultiCore::OneCoreVMSProcess(int64_t listNum, int64_t perListElements,
                                                           int64_t lastListElements)
{
    int64_t coreOffset = GetSortLen<float>(this->vbsTilingData->perCoreElements);
    mrgsortParam.oneLoopMaxElements = this->sortOutTilingData->oneLoopMaxElements;

    for (int64_t i = 0; listNum >= 1; i++) {
        int64_t loops = (listNum + MAX_MRGSORT_LIST - 1) / MAX_MRGSORT_LIST;
        int64_t remainListNum = listNum - (loops - 1) * MAX_MRGSORT_LIST;

        mrgsortParam.perListElements = perListElements;
        mrgsortParam.lastListElements = perListElements;

        int64_t loopOffset = GetSortLen<float>(mrgsortParam.perListElements * MAX_MRGSORT_LIST);
        for (int64_t loop = 0; loop < loops - 1; loop++) {
            RunMoeMrgSort(&mrgsorter, MAX_MRGSORT_LIST, coreOffset, loop * loopOffset);
        }

        mrgsortParam.perListElements = perListElements;
        mrgsortParam.lastListElements = lastListElements;
        RunMoeMrgSort(&mrgsorter, remainListNum, coreOffset, (loops - 1) * loopOffset);

        listNum = loops;
        lastListElements = perListElements * (remainListNum - 1) + lastListElements;
        perListElements = perListElements * MAX_MRGSORT_LIST;
        srcWsIndex = (srcWsIndex + 1) % WORK_GM_NUM;

        if (loops == 1) {
            break;
        }
    }
}

__aicore__ inline void MoeSortMultiCore::UBSortProcess(int64_t progress, int64_t size, int64_t sortNum)
{
    VBSCopyIn(progress, size, sortNum);
    UBSortCompute(progress, size, sortNum);
    VBSCopyOut(progress, size, sortNum);
}

__aicore__ inline void MoeSortMultiCore::VBSProcess()
{
    if (this->blockIdx < this->vbsTilingData->needCoreNum) {
        int64_t sortNum = Ceil(sortCoreLoopElements, ONE_REPEAT_SORT_NUM) * ONE_REPEAT_SORT_NUM;
        for (int64_t loop = 0; loop < sortCoreLoops - 1; loop++) {
            UBSortProcess(loop, sortCoreLoopElements, sortNum);
        }

        sortNum = Ceil(sortCoreLastLoopElements, ONE_REPEAT_SORT_NUM) * ONE_REPEAT_SORT_NUM;
        UBSortProcess(sortCoreLoops - 1, sortCoreLastLoopElements, sortNum);
        if (sortCoreLoops > 1) {
            OneCoreVMSProcess(sortCoreLoops, sortCoreLoopElements, sortCoreLastLoopElements);
        }
    }
    pto_detail::PtoSyncAll();
}

__aicore__ inline void MoeSortMultiCore::VMSProcess()
{
    int64_t currentStageNeedCoreNum = this->vmsTilingData->needCoreNum;
    perListElements = this->vbsTilingData->perCoreElements;
    lastListElements = this->vbsTilingData->lastCoreElements;
    listNum = this->vbsTilingData->needCoreNum;

    for (; listNum > MAX_MRGSORT_LIST;) {
        currentStageNeedCoreNum = Ceil(listNum, MAX_MRGSORT_LIST);
        int64_t coreOffset = GetSortLen<float>(perListElements * MAX_MRGSORT_LIST);
        int64_t remainListNum = listNum - (currentStageNeedCoreNum - 1) * MAX_MRGSORT_LIST;

        if (this->blockIdx < currentStageNeedCoreNum - 1) {
            mrgsortParam.perListElements = perListElements;
            mrgsortParam.lastListElements = perListElements;
            mrgsortParam.oneLoopMaxElements = this->sortOutTilingData->oneLoopMaxElements;
            RunMoeMrgSort(&mrgsorter, MAX_MRGSORT_LIST, coreOffset, 0);
        } else if (this->blockIdx == currentStageNeedCoreNum - 1) {
            mrgsortParam.perListElements = perListElements;
            mrgsortParam.lastListElements = lastListElements;
            mrgsortParam.oneLoopMaxElements = this->sortOutTilingData->oneLoopMaxElements;
            RunMoeMrgSort(&mrgsorter, remainListNum, coreOffset, 0);
        }
        listNum = currentStageNeedCoreNum;
        currentStageNeedCoreNum = Ceil(listNum, MAX_MRGSORT_LIST);
        srcWsIndex = (srcWsIndex + 1) % WORK_GM_NUM;

        lastListElements = perListElements * (remainListNum - 1) + lastListElements;
        perListElements = perListElements * MAX_MRGSORT_LIST;
        pto_detail::PtoSyncAll();
    }
}

__aicore__ inline void MoeSortMultiCore::SortOutProcess()
{
    if (this->blockIdx < 1) {
        mrgsortParam.perListElements = perListElements;
        mrgsortParam.lastListElements = lastListElements;
        mrgsortParam.oneLoopMaxElements = this->sortOutTilingData->oneLoopMaxElements;

        MoeMrgsortOut sorter;
        RunMoeMrgSortOut(&sorter, listNum, GetSortLen<float>(perListElements));
    }
    pto_detail::PtoSyncAll();
}

template <typename TilingData>
__aicore__ inline void MoeSortMultiCore::Init(GM_ADDR expertIdx, GM_ADDR expertTokensCountOrCumsum,
                                              GM_ADDR expertTokensBeforeCapacity, GM_ADDR workspace,
                                              const TilingData *tilingData, AscendC::TPipe *tPipe)
{
    this->totalLength = tilingData->n * tilingData->k;
    this->coreNum = tilingData->coreNum;
    this->vbsTilingData = &(tilingData->vbsComputeParamsOp);
    this->vmsTilingData = &(tilingData->vmsMiddleComputeParamsOp);
    this->sortOutTilingData = &(tilingData->sortOutComputeParamsOp);

    this->blockIdx = get_block_idx() + get_subblockid() * get_block_num();
    this->tileLength = this->vbsTilingData->perCorePerLoopElements;
    this->sortTotalLength = this->vbsTilingData->perCoreElements;
    if (this->blockIdx == tilingData->vbsComputeParamsOp.needCoreNum - 1) {
        this->tileLength = this->vbsTilingData->lastCorePerLoopElements;
        this->sortTotalLength = this->vbsTilingData->lastCoreElements;
    }
    this->n = tilingData->n;
    this->k = tilingData->k;
    this->expertNum = tilingData->expertNum;
    this->expertTokensCountOrCumsumFlag = tilingData->expertTokensCountOrCumsumFlag;
    this->expertTokensBeforeCapacityFlag = tilingData->expertTokensBeforeCapacityFlag;

    // VBS param init
    if (this->blockIdx == this->vbsTilingData->needCoreNum - 1) {
        sortCoreLoops = this->vbsTilingData->lastCoreLoops;
        sortCoreLoopElements = this->vbsTilingData->lastCorePerLoopElements;
        sortCoreLastLoopElements = this->vbsTilingData->lastCoreLastLoopElements;
    } else {
        sortCoreLoops = this->vbsTilingData->perCoreLoops;
        sortCoreLoopElements = this->vbsTilingData->perCorePerLoopElements;
        sortCoreLastLoopElements = this->vbsTilingData->perCoreLastLoopElements;
    }

    expertIdxGm = (__gm__ int32_t *)expertIdx + this->blockIdx * tilingData->vbsComputeParamsOp.perCoreElements;
    sortedexpertIdxGm = reinterpret_cast<__gm__ int32_t *>(workspace);
    expandDstToSrcRowGm = reinterpret_cast<__gm__ int32_t *>(workspace) + Align(this->totalLength, sizeof(int32_t));

    this->perCoreExpert = Align((this->expertNum + this->coreNum - 1) / this->coreNum, sizeof(int32_t));
    this->needInitExpertCore = (this->expertNum + this->perCoreExpert - 1) / this->perCoreExpert;
    this->currentCoreExpert = this->perCoreExpert;
    if (this->blockIdx == needInitExpertCore - 1) {
        this->currentCoreExpert = this->expertNum - (this->needInitExpertCore - 1) * this->perCoreExpert;
    }
    if (this->expertTokensCountOrCumsumFlag > EXERPT_TOKENS_NONE) {
        expertTokensCountOrCumsumGm =
            (__gm__ int32_t *)expertTokensCountOrCumsum + this->blockIdx * this->perCoreExpert;
    }
    if (this->expertTokensBeforeCapacityFlag == EXERPT_TOKENS_BEFORE_CAPACITY) {
        expertTokensBeforeCapacityGm =
            (__gm__ int32_t *)expertTokensBeforeCapacity + this->blockIdx * this->perCoreExpert;
    }
    // key and value
    int64_t kvFactor = 2;
    workspaceGms[0] = (__gm__ float *)workspace + Align(this->totalLength, sizeof(int32_t)) * 2;
    workspaceGms[1] = (__gm__ float *)workspace + Align(this->totalLength, sizeof(int32_t)) * (kvFactor + 2);

    int64_t sortElems = Ceil(Max(this->sortOutTilingData->oneLoopMaxElements * MAX_MRGSORT_LIST, sortCoreLoopElements),
                             ONE_REPEAT_SORT_NUM) *
                        ONE_REPEAT_SORT_NUM;
    int64_t sortBytes = sortElems * sizeof(int32_t) * kvFactor;
    this->sortInputUb = 0;
    this->sortOutputUb = this->sortInputUb + sortBytes;
    this->sortTempUb = this->sortOutputUb + sortBytes;
    this->sortMergeTmpUb = this->sortTempUb;
}

__aicore__ inline void MoeSortMultiCore::Process()
{
    InitExpertTokensGlobalMemory();
    VBSProcess();
    VMSProcess();
    SortOutProcess();
}
} // namespace MoeInitRoutingQuant
#endif // INNER_MOE_INIT_ROUTING_SORT_H
