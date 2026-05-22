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

using namespace AscendC;
using namespace MoeInitRoutingQuant;
using namespace optiling;

template <class DTYPE_X = bfloat16_t>
__aicore__ inline void moe_init_routing_quant(GM_ADDR x, GM_ADDR expertIdx, GM_ADDR scale, GM_ADDR offset,
                                              GM_ADDR expandedX, GM_ADDR expandedRowIdx,
                                              GM_ADDR expertTokensCountOrCumsum, GM_ADDR expertTokensBeforeCapacity,
                                              GM_ADDR dynamicQuantScale, GM_ADDR workspace,
                                              const MoeInitRoutingQuantTilingData *tilingData, uint64_t tilingKey)
{
    if (g_coreType == AIC) {
        return;
    }

    if (workspace == nullptr) {
        return;
    }

    if (tilingKey == 21000) {
        AscendC::TPipe sortPipe;
        MoeFullLoadDynamicQuant<DTYPE_X> op;
        op.Init(x, expertIdx, expandedX, expandedRowIdx, expertTokensCountOrCumsum, scale, dynamicQuantScale, workspace,
                tilingData, &sortPipe);
        op.Process();
        sortPipe.Destroy();
        return;
    }

    if (tilingKey == 11000) {
        AscendC::TPipe sortPipe;
        MoeSortOneCore op;
        op.Init<MoeInitRoutingQuantTilingData>(expertIdx, expertTokensCountOrCumsum, expertTokensBeforeCapacity,
                                               workspace, tilingData, &sortPipe);
        op.Process();
        sortPipe.Destroy();
    } else if (tilingKey == 11010) {
        AscendC::TPipe sortPipe;
        MoeSortMultiCore op;
        op.Init<MoeInitRoutingQuantTilingData>(expertIdx, expertTokensCountOrCumsum, expertTokensBeforeCapacity,
                                               workspace, tilingData, &sortPipe);
        op.Process();
        sortPipe.Destroy();
    } else {
        return;
    }

    if (tilingData->expertTokensCountOrCumsumFlag != EXERPT_TOKENS_NONE) {
        AscendC::TPipe expertTokenOutPipe;
        MoeExpertTokenOut expertTokenOutOp;
        expertTokenOutOp.Init<MoeInitRoutingQuantTilingData>(expertTokensCountOrCumsum, expertTokensBeforeCapacity,
                                                             expandedRowIdx, workspace, tilingData,
                                                             &expertTokenOutPipe);
        expertTokenOutOp.Process();
        expertTokenOutPipe.Destroy();
    }

    AscendC::TPipe srcToDstPipe;
    MoeSrcToDstOp srcToDstOp;
    srcToDstOp.Init<MoeInitRoutingQuantTilingData>(expandedRowIdx, workspace, tilingData, &srcToDstPipe);
    srcToDstOp.Process();
    srcToDstPipe.Destroy();

    AscendC::TPipe gatherPipe;
    MoeGatherDynamicQuant<DTYPE_X> gatherDynamicQuantOp;
    gatherDynamicQuantOp.Init(x, scale, expandedRowIdx, expandedX, dynamicQuantScale, workspace, tilingData,
                              &gatherPipe);
    gatherDynamicQuantOp.Process();
    gatherPipe.Destroy();
}
