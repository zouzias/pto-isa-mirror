/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef MOE_DISPATCH_COMBINE_A8W8_KERNEL_PROTOCOL_CORE_HPP_
#define MOE_DISPATCH_COMBINE_A8W8_KERNEL_PROTOCOL_CORE_HPP_

#include <pto/common/type.hpp>
#include <pto/pto-inst.hpp>

#include "moe_dispatch_combine_a8w8_types.hpp"

#ifndef GM_ADDR
#define GM_ADDR __gm__ uint8_t *
#endif

namespace moe_dispatch_combine_a8w8 {

struct StageContext {
    ShapeConfig shape;
    RankConfig rank;
    GM_ADDR workspace;
    GM_ADDR peerWindow;
    GM_ADDR hcclCtx;
};

// M0 only fixes the stage names and file ownership. M1 will fill these stages
// with direct PTO primitive calls at the call sites described in DESIGN.md.
PTO_INTERNAL void RouteLocalTokens(StageContext &ctx)
{
    (void)ctx;
}

PTO_INTERNAL void RoutePackQuantLocal(StageContext &ctx)
{
    (void)ctx;
}

PTO_INTERNAL void PublishCounts(StageContext &ctx)
{
    (void)ctx;
}

PTO_INTERNAL void WaitCounts(StageContext &ctx)
{
    (void)ctx;
}

PTO_INTERNAL void BuildCumsumAndPreSumBeforeRank(StageContext &ctx)
{
    (void)ctx;
}

PTO_INTERNAL void GatherDispatchToGmm1Input(StageContext &ctx)
{
    (void)ctx;
}

PTO_INTERNAL void RestoreOutput(StageContext &ctx)
{
    (void)ctx;
}

} // namespace moe_dispatch_combine_a8w8

#endif // MOE_DISPATCH_COMBINE_A8W8_KERNEL_PROTOCOL_CORE_HPP_
