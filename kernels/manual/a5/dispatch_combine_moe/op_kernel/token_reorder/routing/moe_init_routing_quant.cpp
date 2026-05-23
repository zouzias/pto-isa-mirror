/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

/*!
 * \file moe_init_routing_quant.cpp
 * \brief
 */
#include "moe_init_routing_expert_tokens.h"
#include "moe_init_routing_fullload_dynamic_quant.h"
#include "moe_init_routing_gather_dynamic_quant.h"
#include "moe_init_routing_sort.h"

using namespace MoeInitRoutingQuant;
using namespace optiling;

template <class DTYPE_X>
__aicore__ inline void RunFullLoadDynamicQuant(GM_ADDR x, GM_ADDR expertIdx, GM_ADDR scale, GM_ADDR expandedX,
                                               GM_ADDR expandedRowIdx, GM_ADDR expertTokensCountOrCumsum,
                                               GM_ADDR dynamicQuantScale, GM_ADDR workspace,
                                               const MoeInitRoutingQuantTilingData *tilingData)
{
        MoeFullLoadDynamicQuant<DTYPE_X> op;
    op.Init(x, expertIdx, expandedX, expandedRowIdx, expertTokensCountOrCumsum, scale, dynamicQuantScale, workspace,
            tilingData);
    op.Process();
}

template <typename SortOp>
__aicore__ inline void RunSortStage(GM_ADDR expertIdx, GM_ADDR expertTokensCountOrCumsum,
                                    GM_ADDR expertTokensBeforeCapacity, GM_ADDR workspace,
                                    const MoeInitRoutingQuantTilingData *tilingData)
{
        SortOp op;
    op.template Init<MoeInitRoutingQuantTilingData>(expertIdx, expertTokensCountOrCumsum, expertTokensBeforeCapacity,
                                                    workspace, tilingData);
    op.Process();
}

__aicore__ inline void RunExpertTokenOut(GM_ADDR expertTokensCountOrCumsum, GM_ADDR expertTokensBeforeCapacity,
                                         GM_ADDR expandedRowIdx, GM_ADDR workspace,
                                         const MoeInitRoutingQuantTilingData *tilingData)
{
    if (tilingData->expertTokensCountOrCumsumFlag == EXERPT_TOKENS_NONE) {
        return;
    }
        MoeExpertTokenOut expertTokenOutOp;
    expertTokenOutOp.Init<MoeInitRoutingQuantTilingData>(expertTokensCountOrCumsum, expertTokensBeforeCapacity,
                                                         expandedRowIdx, workspace, tilingData);
    expertTokenOutOp.Process();
}

__aicore__ inline void RunSrcToDst(GM_ADDR expandedRowIdx, GM_ADDR workspace,
                                   const MoeInitRoutingQuantTilingData *tilingData)
{
        MoeSrcToDstOp srcToDstOp;
    srcToDstOp.Init<MoeInitRoutingQuantTilingData>(expandedRowIdx, workspace, tilingData);
    srcToDstOp.Process();
}

template <class DTYPE_X>
__aicore__ inline void RunGatherDynamicQuant(GM_ADDR x, GM_ADDR scale, GM_ADDR expandedRowIdx, GM_ADDR expandedX,
                                             GM_ADDR dynamicQuantScale, GM_ADDR workspace,
                                             const MoeInitRoutingQuantTilingData *tilingData)
{
        MoeGatherDynamicQuant<DTYPE_X> gatherDynamicQuantOp;
    gatherDynamicQuantOp.Init(x, scale, expandedRowIdx, expandedX, dynamicQuantScale, workspace, tilingData);
    gatherDynamicQuantOp.Process();
}

template <class DTYPE_X = bfloat16_t>
__aicore__ inline void moe_init_routing_quant(GM_ADDR x, GM_ADDR expertIdx, GM_ADDR scale, GM_ADDR offset,
                                              GM_ADDR expandedX, GM_ADDR expandedRowIdx,
                                              GM_ADDR expertTokensCountOrCumsum, GM_ADDR expertTokensBeforeCapacity,
                                              GM_ADDR dynamicQuantScale, GM_ADDR workspace,
                                              const MoeInitRoutingQuantTilingData *tilingData, uint64_t tilingKey)
{
    if (g_coreType == pto_ext::PTO_AIC || workspace == nullptr) {
        return;
    }

    if (tilingKey == 21000) {
        RunFullLoadDynamicQuant<DTYPE_X>(x, expertIdx, scale, expandedX, expandedRowIdx, expertTokensCountOrCumsum,
                                         dynamicQuantScale, workspace, tilingData);
        return;
    }

    if (tilingKey == 11000) {
        RunSortStage<MoeSortOneCore>(expertIdx, expertTokensCountOrCumsum, expertTokensBeforeCapacity, workspace,
                                     tilingData);
    } else if (tilingKey == 11010) {
        RunSortStage<MoeSortMultiCore>(expertIdx, expertTokensCountOrCumsum, expertTokensBeforeCapacity, workspace,
                                       tilingData);
    } else {
        return;
    }

    RunExpertTokenOut(expertTokensCountOrCumsum, expertTokensBeforeCapacity, expandedRowIdx, workspace, tilingData);
    RunSrcToDst(expandedRowIdx, workspace, tilingData);
    RunGatherDynamicQuant<DTYPE_X>(x, scale, expandedRowIdx, expandedX, dynamicQuantScale, workspace, tilingData);
}
