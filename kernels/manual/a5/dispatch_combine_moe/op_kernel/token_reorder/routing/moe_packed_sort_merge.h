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
 * \file moe_packed_sort_merge.h
 * \brief
 */
#ifndef INNER_MOE_PACKED_SORT_MERGE_H
#define INNER_MOE_PACKED_SORT_MERGE_H

#include "moe_common.h"
#include "moe_pto_sort.h"
#include "kernel_operator.h"

namespace MoeInitRoutingQuant {
using namespace AscendC;
using namespace optiling;
struct MoeMrgsortParam {
    int64_t perListElements;
    int64_t lastListElements;
    int64_t oneLoopMaxElements;
};

class MoeMrgsort {
public:
    __aicore__ inline MoeMrgsort(){};
    __aicore__ inline void Init(MoeMrgsortParam *param);
    __aicore__ inline void Process();
    __aicore__ inline void SetInput(__gm__ float *gmInput, uint64_t ubInput);
    __aicore__ inline void SetOutput(__gm__ float *gmOutput, uint64_t ubOutput);
    __aicore__ inline void SetBuffer(uint64_t tempBuffer);

private:
    __aicore__ inline void CopyIn();
    __aicore__ inline void UpdateMrgParam();
    __aicore__ inline void MrgsortCompute();
    __aicore__ inline void UpdateSortInfo();
    __aicore__ inline void CopyOut();
    __aicore__ inline void ClearCache();

private:
    MoeMrgsortParam *param = nullptr;

    __gm__ float *gmInputs[4];
    __gm__ float *gmOutput;

    uint64_t ubInputs[4];
    uint64_t ubOutput;
    uint64_t tempBuffer;

    int64_t listNum{0};
    int64_t remainListNum{0};
    int64_t outOffset{0};
    int64_t offsets[4];
    int64_t listRemainElements[4];
    int64_t lengths[4];
    int64_t allRemainElements{0};
    int64_t curLoopSortedNum{0};

    // for MrgSort
    uint16_t validBitTail{0};
    uint16_t elementCountListTail[4];
    uint32_t listSortedNums[4];
    uint64_t tmpUbInputs[4];
};

__aicore__ inline void MoeMrgsort::ClearCache()
{
    this->listNum = 0;
    this->allRemainElements = 0;
    this->outOffset = 0;
}

__aicore__ inline void MoeMrgsort::SetInput(__gm__ float *gmInput, uint64_t ubInput)
{
    this->gmInputs[listNum] = gmInput;
    this->ubInputs[listNum] = ubInput;
    this->listNum += 1;
}

__aicore__ inline void MoeMrgsort::SetOutput(__gm__ float *gmOutput, uint64_t ubOutput)
{
    this->gmOutput = gmOutput;
    this->ubOutput = ubOutput;
}

__aicore__ inline void MoeMrgsort::SetBuffer(uint64_t tempBuffer)
{
    this->tempBuffer = tempBuffer;
}

__aicore__ inline void MoeMrgsort::UpdateMrgParam()
{
    if (this->remainListNum == MERGE_LIST_TWO) {
        elementCountListTail[MERGE_LIST_IDX_TWO] = 0;
        elementCountListTail[MERGE_LIST_IDX_THREE] = 0;
        validBitTail = 0b0011;
    } else if (this->remainListNum == MERGE_LIST_THREE) {
        elementCountListTail[MERGE_LIST_IDX_THREE] = 0;
        validBitTail = 0b0111;
    } else if (this->remainListNum == MERGE_LIST_FOUR) {
        validBitTail = 0b1111;
    } else {
        validBitTail = 0b0001;
    }
}

__aicore__ inline void MoeMrgsort::CopyIn()
{
    this->remainListNum = 0;
    pto_detail::PtoSetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
    for (int64_t i = 0, j = 0; i < listNum; i++) {
        lengths[i] = Min(param->oneLoopMaxElements, listRemainElements[i]);
        if (lengths[i] > 0) {
            pto_detail::PtoLoadVector(this->ubInputs[i], this->gmInputs[i] + offsets[i], GetSortLen<float>(lengths[i]));
            tmpUbInputs[j] = this->ubInputs[i];
            elementCountListTail[j] = lengths[i];
            this->remainListNum += 1;
            j++;
        }
    }
}

__aicore__ inline void MoeMrgsort::MrgsortCompute()
{
    pto_detail::PtoSetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
    if (this->remainListNum > 1) {
        PtoMergePackedSortRecords(this->ubOutput, this->tempBuffer, this->tmpUbInputs[0], this->tmpUbInputs[1],
                                  this->remainListNum >= MERGE_LIST_THREE ? this->tmpUbInputs[MERGE_LIST_IDX_TWO] : 0,
                                  this->remainListNum >= MERGE_LIST_FOUR ? this->tmpUbInputs[MERGE_LIST_IDX_THREE] : 0,
                                  this->elementCountListTail, this->remainListNum, this->listSortedNums);
    } else {
        pto_detail::PtoMoveVector<float>(this->ubOutput, this->tmpUbInputs[0],
                                         GetSortLen<float>(elementCountListTail[0]));
        listSortedNums[0] = elementCountListTail[0];
    }
}

__aicore__ inline void MoeMrgsort::UpdateSortInfo()
{
    curLoopSortedNum = 0;
    for (int64_t i = 0, j = 0; i < listNum; i++) {
        if (lengths[i] > 0) {
            // update remain size
            listRemainElements[i] -= listSortedNums[j];
            allRemainElements -= listSortedNums[j];
            // update offset
            offsets[i] += GetSortOffset<float>(listSortedNums[j]);
            // update current loop sorted nums
            curLoopSortedNum += listSortedNums[j];
            j += 1;
        }
    }
}

__aicore__ inline void MoeMrgsort::CopyOut()
{
    pto_detail::PtoSetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
    pto_detail::PtoStoreVector(this->gmOutput + outOffset, this->ubOutput, GetSortLen<float>(curLoopSortedNum));
    outOffset += GetSortLen<float>(curLoopSortedNum);
}

__aicore__ inline void MoeMrgsort::Init(MoeMrgsortParam *param)
{
    this->param = param;
    this->remainListNum = listNum;

    for (int64_t i = 0; i < listNum; i++) {
        offsets[i] = GetSortOffset<float>(param->perListElements * i);
        if (i == listNum - 1) {
            listRemainElements[i] = param->lastListElements;
        } else {
            listRemainElements[i] = param->perListElements;
        }
        allRemainElements += listRemainElements[i];
    }
}

__aicore__ inline void MoeMrgsort::Process()
{
    for (; allRemainElements > 0;) {
        CopyIn();
        UpdateMrgParam();
        MrgsortCompute();
        UpdateSortInfo();
        CopyOut();
    }

    ClearCache();
}

class MoeMrgsortOut {
public:
    __aicore__ inline MoeMrgsortOut(){};
    __aicore__ inline void Init(MoeMrgsortParam *param);
    __aicore__ inline void Process();
    __aicore__ inline void SetInput(__gm__ float *gmInput, uint64_t ubInput);
    __aicore__ inline void SetOutput(__gm__ int32_t *gmOutput1, __gm__ int32_t *gmOutput2, uint64_t ubOutput1,
                                     uint64_t ubOutput2);
    __aicore__ inline void SetBuffer(uint64_t tempBuffer, uint64_t mergeTmpBuffer);

private:
    __aicore__ inline void CopyIn();
    __aicore__ inline void UpdateMrgParam();
    __aicore__ inline void MrgsortCompute();
    __aicore__ inline void UpdateSortInfo();
    __aicore__ inline void Extract();
    __aicore__ inline void CopyOut();
    __aicore__ inline void ClearCache();

private:
    MoeMrgsortParam *param = nullptr;

    __gm__ float *gmInputs[4];
    __gm__ int32_t *gmOutput1;
    __gm__ int32_t *gmOutput2;

    uint64_t ubInputs[4];
    uint64_t tempBuffer;

    // for extract
    uint64_t ubOutput1;
    uint64_t ubOutput2;
    uint64_t mergeTmpBuffer;

    // for copy out
    uint64_t ubOutputInt1;
    uint64_t ubOutputInt2;

    int64_t listNum{0};
    int64_t remainListNum{0};
    int64_t outOffset{0};
    int64_t offsets[4];
    int64_t listRemainElements[4];
    int64_t lengths[4];
    int64_t allRemainElements{0};
    int64_t curLoopSortedNum{0};

    // for MrgSort
    uint16_t validBitTail;
    uint16_t elementCountListTail[4];
    uint32_t listSortedNums[4];
    uint64_t tmpUbInputs[4];
};

__aicore__ inline void MoeMrgsortOut::ClearCache()
{
    this->listNum = 0;
    this->allRemainElements = 0;
    this->outOffset = 0;
}

__aicore__ inline void MoeMrgsortOut::SetInput(__gm__ float *gmInput, uint64_t ubInput)
{
    this->gmInputs[listNum] = gmInput;
    this->ubInputs[listNum] = ubInput;
    this->listNum += 1;
}

__aicore__ inline void MoeMrgsortOut::SetOutput(__gm__ int32_t *gmOutput1, __gm__ int32_t *gmOutput2,
                                                uint64_t ubOutput1, uint64_t ubOutput2)
{
    this->gmOutput1 = gmOutput1;
    this->ubOutput1 = ubOutput1;
    this->ubOutputInt1 = ubOutput1;

    this->gmOutput2 = gmOutput2;
    this->ubOutput2 = ubOutput2;
    this->ubOutputInt2 = ubOutput2;
}

__aicore__ inline void MoeMrgsortOut::SetBuffer(uint64_t tempBuffer, uint64_t mergeTmpBuffer)
{
    this->tempBuffer = tempBuffer;
    this->mergeTmpBuffer = mergeTmpBuffer;
}

__aicore__ inline void MoeMrgsortOut::UpdateMrgParam()
{
    if (this->remainListNum == MERGE_LIST_TWO) {
        elementCountListTail[MERGE_LIST_IDX_TWO] = 0;
        elementCountListTail[MERGE_LIST_IDX_THREE] = 0;
        validBitTail = 0b0011;
    } else if (this->remainListNum == MERGE_LIST_THREE) {
        elementCountListTail[MERGE_LIST_IDX_THREE] = 0;
        validBitTail = 0b0111;
    } else if (this->remainListNum == MERGE_LIST_FOUR) {
        validBitTail = 0b1111;
    } else {
        validBitTail = 0b0001;
    }
}

__aicore__ inline void MoeMrgsortOut::CopyIn()
{
    this->remainListNum = 0;
    pto_detail::PtoSetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
    for (int64_t i = 0, j = 0; i < listNum; i++) {
        lengths[i] = Min(param->oneLoopMaxElements, listRemainElements[i]);
        if (lengths[i] > 0) {
            pto_detail::PtoLoadVector(this->ubInputs[i], this->gmInputs[i] + offsets[i], GetSortLen<float>(lengths[i]));
            tmpUbInputs[j] = this->ubInputs[i];
            elementCountListTail[j] = lengths[i];
            this->remainListNum += 1;
            j++;
        }
    }
}

__aicore__ inline void MoeMrgsortOut::MrgsortCompute()
{
    pto_detail::PtoSetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
    if (this->remainListNum > 1) {
        PtoMergePackedSortRecords(this->tempBuffer, this->mergeTmpBuffer, this->tmpUbInputs[0], this->tmpUbInputs[1],
                                  this->remainListNum >= MERGE_LIST_THREE ? this->tmpUbInputs[MERGE_LIST_IDX_TWO] : 0,
                                  this->remainListNum >= MERGE_LIST_FOUR ? this->tmpUbInputs[MERGE_LIST_IDX_THREE] : 0,
                                  this->elementCountListTail, this->remainListNum, this->listSortedNums);
    } else {
        pto_detail::PtoMoveVector<float>(this->tempBuffer, this->tmpUbInputs[0],
                                         GetSortLen<float>(elementCountListTail[0]));
        listSortedNums[0] = elementCountListTail[0];
    }
}

__aicore__ inline void MoeMrgsortOut::UpdateSortInfo()
{
    curLoopSortedNum = 0;
    for (int64_t i = 0, j = 0; i < listNum; i++) {
        if (lengths[i] > 0) {
            // update remain size
            listRemainElements[i] -= listSortedNums[j];
            allRemainElements -= listSortedNums[j];
            // update offset
            offsets[i] += GetSortOffset<float>(listSortedNums[j]);
            // update current loop sorted nums
            curLoopSortedNum += listSortedNums[j];
            j += 1;
        }
    }
}

__aicore__ inline void MoeMrgsortOut::Extract()
{
    PtoExtractPackedSortResult(this->ubOutputInt1, this->ubOutput2, this->tempBuffer, curLoopSortedNum);
}

__aicore__ inline void MoeMrgsortOut::CopyOut()
{
    pto_detail::PtoSetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
    pto_detail::PtoStoreVector(this->gmOutput1 + outOffset, this->ubOutputInt1, curLoopSortedNum);
    pto_detail::PtoStoreVector(this->gmOutput2 + outOffset, this->ubOutputInt2, curLoopSortedNum);
    outOffset += curLoopSortedNum;
}

__aicore__ inline void MoeMrgsortOut::Init(MoeMrgsortParam *param)
{
    this->param = param;
    this->allRemainElements = 0;
    for (int64_t i = 0; i < listNum; i++) {
        offsets[i] = GetSortOffset<float>(param->perListElements * i);
        if (i == listNum - 1) {
            listRemainElements[i] = param->lastListElements;
        } else {
            listRemainElements[i] = param->perListElements;
        }
        allRemainElements += listRemainElements[i];
    }
}

__aicore__ inline void MoeMrgsortOut::Process()
{
    for (; allRemainElements > 0;) {
        CopyIn();
        UpdateMrgParam();
        MrgsortCompute();
        UpdateSortInfo();
        Extract();
        CopyOut();
    }
    ClearCache();
}
} // namespace MoeInitRoutingQuant
#endif // INNER_MOE_PACKED_SORT_MERGE_H