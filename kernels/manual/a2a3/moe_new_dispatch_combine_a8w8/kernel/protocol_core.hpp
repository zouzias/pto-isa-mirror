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
#include <pto/npu/a2a3/TSync.hpp>
#include <pto/pto-inst.hpp>

#include "moe_new_dispatch_combine_a8w8_types.hpp"

#ifndef GM_ADDR
#define GM_ADDR __gm__ uint8_t *
#endif

namespace moe_new_dispatch_combine_a8w8 {

struct StageContext {
    ShapeConfig shape;
    RankConfig rank;
    GM_ADDR workspace;
    GM_ADDR peerWindow;
    GM_ADDR hcclCtx;
};

constexpr uint8_t kM3N4MaxUserCrossCoreId = 10;
constexpr uint8_t kM3N4DispatchToGmm1Flag = 0;
constexpr uint8_t kM3N4Gmm1ToEpilogueFlag = 1;
constexpr uint8_t kM3N4Gmm1EpilogueToActivationFlag = 2;
constexpr uint8_t kM3N4ActivationToGmm2Flag = 3;
constexpr uint8_t kM3N4Gmm2ToCombineFlag = 4;
constexpr uint8_t kM3N4CombineToRestoreFlag = 5;
constexpr uint32_t kM3N4StreamHandshakeSites = 6;
constexpr uint32_t kM3N4FullOpenCvWaitCount = 6;
constexpr uint32_t kM3N4CoarseMixSyncallCount = 16;
constexpr uint32_t kM3N4AicOnlySyncCount = 2;
constexpr uint32_t kM3N4AivOnlySyncCount = 2;
constexpr uint8_t kM3N5DispatchExpertFlagBase = 6;
constexpr uint32_t kM3N5DispatchExpertFlagCapacity = 5;
constexpr uint8_t kM3N6Gmm1SyncGroupFlagBase = kM3N5DispatchExpertFlagBase;
constexpr uint32_t kM3N6Gmm1SyncGroupFlagCapacity = kM3N5DispatchExpertFlagCapacity;

using M3N4V2CEvent = pto::Event<pto::Op::TSTORE_VEC, pto::Op::TLOAD, false, EVENT_ID0>;
using M3N4C2VEvent = pto::Event<pto::Op::TSTORE_ACC, pto::Op::TLOAD, false, EVENT_ID0>;

template <uint8_t CrossCoreId>
PTO_INTERNAL void M3N4SignalV2C()
{
    PTO_STATIC_ASSERT(CrossCoreId <= kM3N4MaxUserCrossCoreId, "M3N4 cross-core flag id exceeds user range");
    M3N4V2CEvent event;
    event.Init<CrossCoreId>();
}

template <uint8_t CrossCoreId>
PTO_INTERNAL void M3N4WaitV2C()
{
    PTO_STATIC_ASSERT(CrossCoreId <= kM3N4MaxUserCrossCoreId, "M3N4 cross-core flag id exceeds user range");
    M3N4V2CEvent event;
    event.Wait<CrossCoreId>();
}

template <uint8_t CrossCoreId>
PTO_INTERNAL void M3N4SignalC2V()
{
    PTO_STATIC_ASSERT(CrossCoreId <= kM3N4MaxUserCrossCoreId, "M3N4 cross-core flag id exceeds user range");
    M3N4C2VEvent event;
    event.Init<CrossCoreId>();
}

template <uint8_t CrossCoreId>
PTO_INTERNAL void M3N4WaitC2V()
{
    PTO_STATIC_ASSERT(CrossCoreId <= kM3N4MaxUserCrossCoreId, "M3N4 cross-core flag id exceeds user range");
    M3N4C2VEvent event;
    event.Wait<CrossCoreId>();
}

PTO_INTERNAL void M3N4AicAllDoneCoarseSync()
{
    pipe_barrier(PIPE_ALL);
    dsb(DSB_DDR);
    pto::SYNCALL<pto::SyncCoreType::AICOnly>();
}

PTO_INTERNAL void M3N4AivAllDoneCoarseSync()
{
    pipe_barrier(PIPE_ALL);
    dsb(DSB_DDR);
    pto::SYNCALL<pto::SyncCoreType::AIVOnly>();
}

PTO_INTERNAL bool M3N5DispatchExpertFlagsSupported(const ShapeConfig &shape)
{
    return shape.expertPerRank > 0U && shape.expertPerRank <= kM3N5DispatchExpertFlagCapacity;
}

PTO_INTERNAL uint32_t M3N6NextSwigluGroupSize(uint32_t remainingExperts)
{
    if (remainingExperts <= 1U) {
        return 1U;
    }
    return remainingExperts / 2U;
}

PTO_INTERNAL uint32_t M3N6SwigluSyncGroupCount(const ShapeConfig &shape)
{
    uint32_t groupCount = 0;
    uint32_t expert = 0;
    while (expert < shape.expertPerRank) {
        uint32_t groupSize = M3N6NextSwigluGroupSize(shape.expertPerRank - expert);
        expert += groupSize;
        ++groupCount;
    }
    return groupCount;
}

PTO_INTERNAL bool M3N6Gmm1SyncGroupFlagsSupported(const ShapeConfig &shape)
{
    uint32_t groupCount = M3N6SwigluSyncGroupCount(shape);
    return groupCount > 0U && groupCount <= kM3N6Gmm1SyncGroupFlagCapacity;
}

PTO_INTERNAL void M3N5SignalDispatchExpertReady(uint32_t localExpert)
{
    switch (localExpert) {
        case 0U:
            M3N4SignalV2C<kM3N5DispatchExpertFlagBase + 0U>();
            break;
        case 1U:
            M3N4SignalV2C<kM3N5DispatchExpertFlagBase + 1U>();
            break;
        case 2U:
            M3N4SignalV2C<kM3N5DispatchExpertFlagBase + 2U>();
            break;
        case 3U:
            M3N4SignalV2C<kM3N5DispatchExpertFlagBase + 3U>();
            break;
        case 4U:
            M3N4SignalV2C<kM3N5DispatchExpertFlagBase + 4U>();
            break;
        default:
            break;
    }
}

PTO_INTERNAL void M3N5WaitDispatchExpertReady(uint32_t localExpert)
{
    switch (localExpert) {
        case 0U:
            M3N4WaitV2C<kM3N5DispatchExpertFlagBase + 0U>();
            break;
        case 1U:
            M3N4WaitV2C<kM3N5DispatchExpertFlagBase + 1U>();
            break;
        case 2U:
            M3N4WaitV2C<kM3N5DispatchExpertFlagBase + 2U>();
            break;
        case 3U:
            M3N4WaitV2C<kM3N5DispatchExpertFlagBase + 3U>();
            break;
        case 4U:
            M3N4WaitV2C<kM3N5DispatchExpertFlagBase + 4U>();
            break;
        default:
            break;
    }
}

PTO_INTERNAL void M3N6SignalGmm1SyncGroupReady(uint32_t syncIdx)
{
    switch (syncIdx) {
        case 0U:
            M3N4SignalC2V<kM3N6Gmm1SyncGroupFlagBase + 0U>();
            break;
        case 1U:
            M3N4SignalC2V<kM3N6Gmm1SyncGroupFlagBase + 1U>();
            break;
        case 2U:
            M3N4SignalC2V<kM3N6Gmm1SyncGroupFlagBase + 2U>();
            break;
        case 3U:
            M3N4SignalC2V<kM3N6Gmm1SyncGroupFlagBase + 3U>();
            break;
        case 4U:
            M3N4SignalC2V<kM3N6Gmm1SyncGroupFlagBase + 4U>();
            break;
        default:
            break;
    }
}

PTO_INTERNAL void M3N6WaitGmm1SyncGroupReady(uint32_t syncIdx)
{
    switch (syncIdx) {
        case 0U:
            M3N4WaitC2V<kM3N6Gmm1SyncGroupFlagBase + 0U>();
            break;
        case 1U:
            M3N4WaitC2V<kM3N6Gmm1SyncGroupFlagBase + 1U>();
            break;
        case 2U:
            M3N4WaitC2V<kM3N6Gmm1SyncGroupFlagBase + 2U>();
            break;
        case 3U:
            M3N4WaitC2V<kM3N6Gmm1SyncGroupFlagBase + 3U>();
            break;
        case 4U:
            M3N4WaitC2V<kM3N6Gmm1SyncGroupFlagBase + 4U>();
            break;
        default:
            break;
    }
}

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

} // namespace moe_new_dispatch_combine_a8w8

#endif // MOE_DISPATCH_COMBINE_A8W8_KERNEL_PROTOCOL_CORE_HPP_
