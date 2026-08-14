/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef DISPATCH_MEGA_COMBINE_H
#define DISPATCH_MEGA_COMBINE_H

#include "kernel_operator.h"

#include "dispatch_mega_combine_tiling.h"
#include "utils/hccl_window.hpp"
#include "utils/mega_expert_sync.hpp"

#if defined(__DAV_VEC__)
#include "combine.h"
#include "deferred_route_metadata.h"
#include "dispatch.h"
#include "front_reorder.h"
#include "swiglu.h"
#include "unpermute.h"
#endif

#if defined(__DAV_CUBE__)
#include "gmm1.h"
#include "gmm2.h"
#endif

template <typename CType_, uint32_t ExpertPerRank>
class MegaMoe {
public:
    __aicore__ inline void Init(GM_ADDR xGM, GM_ADDR weight1GM, GM_ADDR weight2GM, GM_ADDR expertIdGM,
                                GM_ADDR weightScale1GM, GM_ADDR weightScale2GM, GM_ADDR probs, GM_ADDR outGM,
                                GM_ADDR expertTokenNums, GM_ADDR workspaceGM,
                                const __gm__ MegaMoeTilingData *tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ProcessFixedGroups();
#if defined(__DAV_CUBE__)
    __aicore__ inline void ProcessFixedGmm1(uint32_t physicalBlockId);
#endif

    GM_ADDR xGM_ = nullptr;
    GM_ADDR weight1GM_ = nullptr;
    GM_ADDR weight2GM_ = nullptr;
    GM_ADDR weightScale1GM_ = nullptr;
    GM_ADDR weightScale2GM_ = nullptr;
    GM_ADDR expertIdGM_ = nullptr;
    GM_ADDR expertTokenNumsGM_ = nullptr;
    GM_ADDR workspaceGM_ = nullptr;
    GM_ADDR probsGM_ = nullptr;
    GM_ADDR outGM_ = nullptr;
    const __gm__ MegaMoeTilingData *tilingData_ = nullptr;
};

template <typename CType_, uint32_t ExpertPerRank>
__aicore__ inline void MegaMoe<CType_, ExpertPerRank>::Init(GM_ADDR xGM, GM_ADDR weight1GM, GM_ADDR weight2GM,
                                                            GM_ADDR expertIdGM, GM_ADDR weightScale1GM,
                                                            GM_ADDR weightScale2GM, GM_ADDR probs, GM_ADDR outGM,
                                                            GM_ADDR expertTokenNums, GM_ADDR workspaceGM,
                                                            const __gm__ MegaMoeTilingData *tilingData)
{
    xGM_ = xGM;
    weight1GM_ = weight1GM;
    weight2GM_ = weight2GM;
    weightScale1GM_ = weightScale1GM;
    weightScale2GM_ = weightScale2GM;
    expertIdGM_ = expertIdGM;
    expertTokenNumsGM_ = expertTokenNums;
    workspaceGM_ = workspaceGM;
    probsGM_ = probs;
    outGM_ = outGM;
    tilingData_ = tilingData;
}

#if defined(__DAV_CUBE__)
template <typename CType_, uint32_t ExpertPerRank>
__aicore__ inline void MegaMoe<CType_, ExpertPerRank>::ProcessFixedGmm1(uint32_t physicalBlockId)
{
    const __gm__ MegaMoeFixedGroupTiling &fixed = tilingData_->fixedGroupTiling;
    Gmm1<CType_> gmm1;
    gmm1.Init(weight1GM_, weightScale1GM_, expertTokenNumsGM_, workspaceGM_, tilingData_);
    gmm1.ProcessFixed(physicalBlockId, fixed.physicalAicNum);
}
#endif

template <typename CType_, uint32_t ExpertPerRank>
__aicore__ inline void MegaMoe<CType_, ExpertPerRank>::ProcessFixedGroups()
{
    const MegaMoeFixedCoreRoleInfo role =
        FixedCoreRole(tilingData_);
    const __gm__ MegaMoeFixedGroupTiling &fixed = tilingData_->fixedGroupTiling;
    const uint32_t dispatchRankSize = tilingData_->runtimeInfo.rankSize;
    const uint32_t dispatchLanesPerRank = dispatchRankSize == 0U ? 0U : fixed.dispatchGroupSize / dispatchRankSize;
    const uint32_t dispatchActiveWorkerCount = dispatchRankSize * dispatchLanesPerRank;
    const bool cvSwigluWorker = role.subblockId == 1U;
#if defined(__DAV_CUBE__)
    if (role.role == kMegaMoeFixedRoleGmm1 || role.role == kMegaMoeFixedRoleGmm2) {
        ProcessFixedGmm1(role.physicalBlockId);
        if (role.role == kMegaMoeFixedRoleGmm1 && role.groupLocalId == 0U) {
            pipe_barrier(PIPE_ALL);
            dsb(DSB_DDR);
            PublishScalarEpoch(
                FixedSyncSlot(workspaceGM_, tilingData_, kMegaMoeFixedSyncGmm1DoneSlot),
                kMegaMoeFixedGmm1DoneMarker);
        }
        WaitGmm2EntryReady(workspaceGM_, tilingData_);
        WaitCombineConsumerArmed(workspaceGM_, tilingData_, role.physicalBlockId);
        Gmm2<CType_> gmm2;
        gmm2.Init(weight2GM_, weightScale2GM_, expertTokenNumsGM_, workspaceGM_, tilingData_);
        if (role.role == kMegaMoeFixedRoleGmm2) {
            gmm2.ProcessFixed(role.groupLocalId, role.groupSize);
        } else {
            gmm2.ProcessFixedHelper(role.groupLocalId);
        }
    }
#elif defined(__DAV_VEC__)
    // AIV0s outside the independent Dispatch group build direct-route
    // metadata. The post-Dispatch GMM2 group can be wider than this pool.
    if (role.subblockId == 0U && role.physicalBlockId >= fixed.dispatchGroupSize) {
        const uint32_t metadataWorkerIdx = role.physicalBlockId - fixed.dispatchGroupSize;
        const uint32_t metadataWorkerCount = fixed.physicalAicNum - fixed.dispatchGroupSize;
        deferred_route_metadata::DeferredRouteMetadata<ExpertPerRank> metadata;
        metadata.Init(expertIdGM_, workspaceGM_, tilingData_, metadataWorkerIdx, metadataWorkerCount);
        metadata.Run();
        if (metadataWorkerIdx == 0U) {
            // Only this local Group2 AIV0 coordinates the common handoff. Its
            // own metadata run has completed expandedRowIdx. Publish entry-ready
            // after wave 0, local Dispatch, and every source-rank preSum are ready.
            WaitEpochAcquire(
                FixedSyncSlot(
                    workspaceGM_, tilingData_, kMegaMoeFixedSyncGmm2ReadyBase),
                1);
            WaitEpochAcquire(
                FixedSyncSlot(
                    workspaceGM_, tilingData_, kMegaMoeFixedSyncDispatchDoneSlot),
                kMegaMoeFixedDispatchDoneMarker);
            PtoRemoteWindow remoteWindow;
            remoteWindow.Init(reinterpret_cast<GM_ADDR>(tilingData_->runtimeInfo.remoteWindowContext));
            const int32_t preSumEpoch = remoteWindow.FrontReadyEpoch();
            for (uint32_t srcRank = 0U; srcRank < tilingData_->runtimeInfo.rankSize; ++srcRank) {
                remoteWindow.WaitPreSumReady(static_cast<int32_t>(srcRank), preSumEpoch);
            }
            PublishGmm2EntryReady(workspaceGM_, tilingData_);
        }
    }
    if (role.role == kMegaMoeFixedRoleDispatch && role.groupLocalId < dispatchActiveWorkerCount) {
        DispatchGather<CType_> dispatchGather;
        dispatchGather.Init(expertTokenNumsGM_, workspaceGM_, tilingData_);
        dispatchGather.ProcessFixed(role.groupLocalId, role.groupSize);
        pipe_barrier(PIPE_ALL);
        dsb(DSB_DDR);
        PtoRemoteWindow remoteWindow;
        remoteWindow.Init(reinterpret_cast<GM_ADDR>(tilingData_->runtimeInfo.remoteWindowContext));
        const int32_t epoch = remoteWindow.DataReadyEpoch();
        remoteWindow.PublishDispatchRelease(role.groupLocalId, epoch);
        if (role.groupLocalId == 0U) {
            // The 32-core topology has one inactive Dispatch role after rank-balanced lane assignment.
            remoteWindow.WaitDispatchReleaseMte(dispatchActiveWorkerCount, epoch);
            PublishScalarEpoch(
                FixedSyncSlot(workspaceGM_, tilingData_, kMegaMoeFixedSyncDispatchDoneSlot),
                kMegaMoeFixedDispatchDoneMarker);
        }
    }
    if (cvSwigluWorker) {
        Swiglu<CType_> swiglu;
        swiglu.Init(expertTokenNumsGM_, workspaceGM_, tilingData_);
        swiglu.ProcessFixed(role.physicalBlockId, fixed.physicalAicNum);
    }
    // GMM2 can eventually use every physical AIC after the GMM1 group joins,
    // so every paired AIV1 becomes a direct Combine consumer after finishing
    // its SwiGLU work. Arm first, then wait for the local Dispatch + deferred
    // metadata gate before either side starts the direct pipeline.
    if (role.subblockId == 1U) {
        pipe_barrier(PIPE_ALL);
        dsb(DSB_DDR);
        PublishCombineConsumerArmed(workspaceGM_, tilingData_, role.physicalBlockId);
        WaitGmm2EntryReady(workspaceGM_, tilingData_);
        Combine<bfloat16_t> combine;
        combine.Init(workspaceGM_, tilingData_);
        combine.ProcessFixed(role.physicalBlockId);
    }
#endif

#if defined(__DAV_VEC__)
    {
        // Rank-streaming uses stable logical ordinals rather than fixed-role
        // group-local ids. AIV1 occupies [0, P), and AIV0 occupies [P, 2P).
        const uint32_t rankStreamingWorkerIdx = role.subblockId == 1U ? role.physicalBlockId :
                                                                         fixed.physicalAicNum + role.physicalBlockId;
        if (rankStreamingWorkerIdx == fixed.physicalAicNum) {
            WaitArrivalMinMte(
                workspaceGM_, tilingData_, kMegaMoeFixedSyncGmm1ArrivalBase, fixed.gmm1GroupSize,
                static_cast<int32_t>(fixed.totalWaveCount + kMegaMoeFixedGmm1GroupDoneEpochOffset));

            PtoRemoteWindow remoteWindow;
            remoteWindow.Init(reinterpret_cast<GM_ADDR>(tilingData_->runtimeInfo.remoteWindowContext));
            const int32_t epoch = remoteWindow.DataReadyEpoch();
            const uint32_t rankCount = tilingData_->runtimeInfo.rankSize;
            const uint32_t expertPerRank = tilingData_->megaMoeInfo.expertPerRank;
            uint32_t readyExpertCounts[COMBINE_EXPERT_PROGRESS_MAX_RANKS] = {0U};
            (void)remoteWindow.ReadExpertProgressMte(epoch, expertPerRank, readyExpertCounts);
            remoteWindow.AcquireDataReady();
            uint32_t readyRankMask = 0U;
            for (uint32_t producerRank = 0U; producerRank < rankCount; ++producerRank) {
                if (readyExpertCounts[producerRank] != 0U) {
                    readyRankMask |= 1U << producerRank;
                }
            }
            remoteWindow.PublishUnpermutePhase1Progress(readyExpertCounts, rankCount, readyRankMask, epoch);
        }
        Unpermute<bfloat16_t> unpermute;
        const uint32_t workerCount = fixed.physicalAicNum * kMegaMoeFixedAivSubblocksPerPhysicalBlock;
        unpermute.Init(workspaceGM_, expertIdGM_, probsGM_, outGM_, tilingData_, rankStreamingWorkerIdx, workerCount);
        unpermute.Process();
    }
#endif
}

template <typename CType_, uint32_t ExpertPerRank>
__aicore__ inline void MegaMoe<CType_, ExpertPerRank>::Process()
{
#if defined(__DAV_VEC__)
    FrontReorderProcess<CType_, ExpertPerRank>(xGM_, expertIdGM_, expertTokenNumsGM_, workspaceGM_, tilingData_);
#endif

    ProcessFixedGroups();
}

#endif // DISPATCH_MEGA_COMBINE_H
