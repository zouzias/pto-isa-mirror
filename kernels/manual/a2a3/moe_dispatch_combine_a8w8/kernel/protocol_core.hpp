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

#ifndef PIPE_FIX
#define PIPE_FIX static_cast<pipe_t>(10)
#endif

#include <pto/comm/pto_comm_inst.hpp>
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

PTO_INTERNAL void RouteLocalTokens(StageContext &ctx)
{
    // M1 direct stage body: metadata route/count uses scalar GM views; payload movement is not hidden here.
    (void)ctx;
}

PTO_INTERNAL void RoutePackQuantLocal(StageContext &ctx)
{
    // M1 mock/non-quant pack call site:
    //   TLOAD(input tile) -> TSTORE(peerWindow.dispatchPayload)
    // M2 upgrades the same stage to TQUANT + TSTORE int8 payload and per-token scale.
    (void)ctx;
}

PTO_INTERNAL void PublishCounts(StageContext &ctx)
{
    // Count allgather publish call site:
    //   pto::comm::TNOTIFY(remote countReadySignal[tokenOwnerRank], ..., AtomicAdd)
    (void)ctx;
}

PTO_INTERNAL void WaitCounts(StageContext &ctx)
{
    // Count wait call site:
    //   pto::comm::TTEST/TWAIT(local countReadySignal[tokenOwnerRank], ..., GE)
    (void)ctx;
}

PTO_INTERNAL void BuildCumsumAndPreSumBeforeRank(StageContext &ctx)
{
    // Builds cumsumMM/preSumBeforeRank/expertTokenNums from tokenPerExpertMatrix.
    (void)ctx;
}

PTO_INTERNAL void GatherDispatchToGmm1Input(StageContext &ctx)
{
    // M1 mock sink call site:
    //   pto::comm::TGET(workspace.dispatchedA, peerWindow.dispatchPayload, stagingTile)
    // M2 keeps this stage and writes gmm1InputInt8 + routingPerTokenScale directly.
    (void)ctx;
}

PTO_INTERNAL void MockExpertOutput(StageContext &ctx)
{
    // M1 deterministic mock is the only allowed GMM substitute; protocol stages remain final.
    (void)ctx;
}

PTO_INTERNAL void RunGmm2EpilogueAndReturn(StageContext &ctx)
{
    // M1 mock return call site:
    //   TLOAD(mockExpertOutput) -> pto::comm::TPUT(peerWindow.returnPayload/offsetD)
    //   pto::comm::TNOTIFY(remote combineDoneSignal[expertOwnerRank], ..., AtomicAdd)
    // M2 replaces mock input with GMM2 epilogue/cast, not with a separate ReturnCombine stage.
    (void)ctx;
}

PTO_INTERNAL void RestoreOutput(StageContext &ctx)
{
    // Restore call site:
    //   pto::comm::TWAIT(combineDoneSignal[expertOwnerRank], ..., GE)
    //   TLOAD(returnPayload) -> TMUL(probs) -> TADD(accum) -> TSTORE(output)
    (void)ctx;
}

} // namespace moe_dispatch_combine_a8w8

#endif // MOE_DISPATCH_COMBINE_A8W8_KERNEL_PROTOCOL_CORE_HPP_
