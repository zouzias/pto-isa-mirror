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
 * \file moe_init_routing_quant_v2.cpp
 * \brief
 */
#include "moe_v2_expert_token_out.h"
#include "moe_v2_fullload_dynamic_quant.h"
#include "moe_v2_gather_dynamic_quant.h"
#include "moe_v2_sort_multi_core.h"
#include "moe_v2_sort_one_core.h"
#include "moe_v2_src_to_dst_op.h"

using namespace AscendC;
using namespace MoeInitRoutingQuantV2;
using namespace optiling;

template <class DTYPE_X = bfloat16_t>
__aicore__ inline void moe_init_routing_quant_v2(GM_ADDR x, GM_ADDR expertIdx, GM_ADDR scale, GM_ADDR offset,
                                                 GM_ADDR expandedX, GM_ADDR expandedRowIdx,
                                                 GM_ADDR expertTokensCountOrCumsum, GM_ADDR expertTokensBeforeCapacity,
                                                 GM_ADDR dynamicQuantScale, GM_ADDR workspace,
                                                 const MoeInitRoutingQuantV2TilingData *tilingData, uint64_t tilingKey)
{
    if (g_coreType == AIC) {
        return;
    }

    if (workspace == nullptr) {
        return;
    }

    if (tilingKey == 21000) {
        AscendC::TPipe sortPipe;
        MoeV2FullLoadDynamicQuant<DTYPE_X> op;
        op.Init(x, expertIdx, expandedX, expandedRowIdx, expertTokensCountOrCumsum, scale, dynamicQuantScale, workspace,
                tilingData, &sortPipe);
        op.Process();
        sortPipe.Destroy();
        return;
    }

    if (tilingKey == 11000) {
        AscendC::TPipe sortPipe;
        MoeV2SortOneCore op;
        op.Init<MoeInitRoutingQuantV2TilingData>(expertIdx, expertTokensCountOrCumsum, expertTokensBeforeCapacity,
                                                 workspace, tilingData, &sortPipe);
        op.Process();
        sortPipe.Destroy();
    } else if (tilingKey == 11010) {
        AscendC::TPipe sortPipe;
        MoeV2SortMultiCore op;
        op.Init<MoeInitRoutingQuantV2TilingData>(expertIdx, expertTokensCountOrCumsum, expertTokensBeforeCapacity,
                                                 workspace, tilingData, &sortPipe);
        op.Process();
        sortPipe.Destroy();
    } else {
        return;
    }

    if (tilingData->expertTokensCountOrCumsumFlag != EXERPT_TOKENS_NONE) {
        AscendC::TPipe expertTokenOutPipe;
        MoeV2ExpertTokenOut expertTokenOutOp;
        expertTokenOutOp.Init<MoeInitRoutingQuantV2TilingData>(expertTokensCountOrCumsum, expertTokensBeforeCapacity,
                                                               expandedRowIdx, workspace, tilingData,
                                                               &expertTokenOutPipe);
        expertTokenOutOp.Process();
        expertTokenOutPipe.Destroy();
    }

    AscendC::TPipe srcToDstPipe;
    MoeV2SrcToDstOp srcToDstOp;
    srcToDstOp.Init<MoeInitRoutingQuantV2TilingData>(expandedRowIdx, workspace, tilingData, &srcToDstPipe);
    srcToDstOp.Process();
    srcToDstPipe.Destroy();

    AscendC::TPipe gatherPipe;
    MoeV2GatherDynamicQuant<DTYPE_X> gatherDynamicQuantOp;
    gatherDynamicQuantOp.Init(x, scale, expandedRowIdx, expandedX, dynamicQuantScale, workspace, tilingData, &gatherPipe);
    gatherDynamicQuantOp.Process();
    gatherPipe.Destroy();
}
