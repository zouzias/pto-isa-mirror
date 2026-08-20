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
#include "gmm_expert_progress.h"
#include "gmm_task_producer.h"
#include "swiglu.h"
#include "unpermute.h"
#endif

#if defined(__DAV_CUBE__)
#include "gmm1.h"
#include "gmm2.h"
#endif

template <typename CType_>
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
    __aicore__ inline GmmClaimedTask ProcessFixedGmm1(uint32_t physicalBlockId,
                                                      GmmCommonPipeline *gmmPipeline);
    __aicore__ inline void ProcessFixedGmm2(uint32_t physicalBlockId, GmmCommonPipeline &gmmPipeline,
                                            const GmmClaimedTask &initialTask);
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

template <typename CType_>
__aicore__ inline void MegaMoe<CType_>::Init(GM_ADDR xGM, GM_ADDR weight1GM, GM_ADDR weight2GM, GM_ADDR expertIdGM,
                                             GM_ADDR weightScale1GM, GM_ADDR weightScale2GM, GM_ADDR probs,
                                             GM_ADDR outGM, GM_ADDR expertTokenNums, GM_ADDR workspaceGM,
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
template <typename CType_>
__aicore__ inline GmmClaimedTask MegaMoe<CType_>::ProcessFixedGmm1(uint32_t physicalBlockId,
                                                                  GmmCommonPipeline *gmmPipeline)
{
    const __gm__ MegaMoeFixedGroupTiling &fixed = tilingData_->fixedGroupTiling;
    Gmm1 gmm1;
    gmm1.Init(weight1GM_, weightScale1GM_, workspaceGM_, tilingData_);
    return gmm1.ProcessFixed(physicalBlockId, fixed.physicalAicNum, gmmPipeline);
}

template <typename CType_>
__aicore__ inline void MegaMoe<CType_>::ProcessFixedGmm2(uint32_t physicalBlockId, GmmCommonPipeline &gmmPipeline,
                                                         const GmmClaimedTask &initialTask)
{
    const __gm__ MegaMoeFixedGroupTiling &fixed = tilingData_->fixedGroupTiling;
    WaitGmm2ConsumerReady(workspaceGM_, tilingData_, physicalBlockId);
    Gmm2 gmm2;
    gmm2.Init(weight2GM_, weightScale2GM_, workspaceGM_, tilingData_);
    gmm2.ProcessFixed(physicalBlockId, gmmPipeline, initialTask);
}
#endif

template <typename CType_>
__aicore__ inline void MegaMoe<CType_>::ProcessFixedGroups()
{
    const MegaMoeFixedCoreRoleInfo role = FixedCoreRole(tilingData_);
    const __gm__ MegaMoeFixedGroupTiling &fixed = tilingData_->fixedGroupTiling;

#if defined(__DAV_CUBE__)
    if (role.role == kMegaMoeFixedRoleGmm1 || role.role == kMegaMoeFixedRoleGmm2) {
        const bool fixedWaveGmm1 = tilingData_->gmmSchedulerTiling.gmm1ScheduleMode == kMegaMoeGmm1ScheduleFixedWave;
        if (fixedWaveGmm1) {
            // The fixed-wave implementation owns its local pipeline. GMM2
            // starts with a separate pipeline after the wave-0 marker.
            ProcessFixedGmm1(role.physicalBlockId, nullptr);
            if (role.role == kMegaMoeFixedRoleGmm1) {
                pipe_barrier(PIPE_ALL);
                dsb(DSB_DDR);
                PublishGmmMailboxProgress(workspaceGM_, tilingData_->gmmSchedulerTiling.mailbox,
                                          role.physicalBlockId, kGmmMailboxGmm1LocalDoneTicket);
            }
            GmmCommonPipeline gmm2Pipeline;
            const GmmClaimedTask noInitialTask;
            ProcessFixedGmm2(role.physicalBlockId, gmm2Pipeline, noInitialTask);
        } else {
            // In mailbox mode the GMM1-to-GMM2 transition carries the first
            // GMM2 task, so both stages must share one pipeline instance.
            GmmCommonPipeline sharedPipeline;
            const GmmClaimedTask initialGmm2 = ProcessFixedGmm1(role.physicalBlockId, &sharedPipeline);
            ProcessFixedGmm2(role.physicalBlockId, sharedPipeline, initialGmm2);
        }
    }
#elif defined(__DAV_VEC__)
    const uint32_t dispatchRankSize = tilingData_->runtimeInfo.rankSize;
    const uint32_t dispatchLanesPerRank = dispatchRankSize == 0U ? 0U : fixed.dispatchGroupSize / dispatchRankSize;
    const uint32_t dispatchActiveWorkerCount = dispatchRankSize * dispatchLanesPerRank;
    const bool cvSwigluWorker = role.subblockId == 1U && role.physicalBlockId < fixed.swigluActiveGroupSize;
    const bool gmm1Mailbox =
        tilingData_->gmmSchedulerTiling.gmm1ScheduleMode == kMegaMoeGmm1ScheduleWave0MailboxSuffix;

    const uint32_t rankStreamingWorkerCount = tilingData_->unpermuteTiling.rankStreamingWorkerCount;
    const uint32_t producerPhysicalBlockId = tilingData_->gmmSchedulerTiling.producerPhysicalBlockId;
    const bool gmmTaskProducerRole = role.subblockId == 0U && role.physicalBlockId == producerPhysicalBlockId;
    const bool hasDedicatedProgressWorker = producerPhysicalBlockId > fixed.dispatchGroupSize;
    const uint32_t progressPhysicalBlockId = hasDedicatedProgressWorker ? producerPhysicalBlockId - 1U : 0U;

    // GMM2's mailbox is always active. The producer waits for the front
    // metadata marker, then schedules both direct-wave suffixes.
    if (gmmTaskProducerRole) {
        MegaMoeGmmTaskProducer producer;
        producer.Init(workspaceGM_, tilingData_);
        producer.Process();
        if (!hasDedicatedProgressWorker) {
            Gmm2ExpertProgressCoordinator coordinator;
            coordinator.Init(workspaceGM_, tilingData_);
            coordinator.Process();
        }
    }

    // Reserve the AIV0s between Dispatch and the producer for route metadata.
    // The producer itself is deliberately excluded from this pool.
    const bool hasMetadataWorkers = producerPhysicalBlockId > fixed.dispatchGroupSize;
    if (hasMetadataWorkers && role.subblockId == 0U && role.physicalBlockId >= fixed.dispatchGroupSize &&
        role.physicalBlockId < producerPhysicalBlockId) {
        const uint32_t metadataWorkerIdx = role.physicalBlockId - fixed.dispatchGroupSize;
        const uint32_t metadataWorkerCount = producerPhysicalBlockId - fixed.dispatchGroupSize;
        deferred_route_metadata::DeferredRouteMetadata metadata;
        metadata.Init(expertIdGM_, workspaceGM_, tilingData_, metadataWorkerIdx, metadataWorkerCount);
        metadata.Run();
        if (metadataWorkerIdx == 0U) {
            // Expanded metadata, local Dispatch and every source-rank preSum
            // must be visible before any GMM2 consumer enters the mailbox.
            WaitEpochAcquire(
                FixedSyncSlot(workspaceGM_, tilingData_, kMegaMoeFixedSyncDispatchDoneSlot),
                kMegaMoeFixedDispatchDoneMarker);
            PtoRemoteWindow remoteWindow;
            remoteWindow.Init(reinterpret_cast<GM_ADDR>(tilingData_->runtimeInfo.remoteWindowContext));
            const int32_t preSumEpoch = remoteWindow.FrontReadyEpoch();
            for (uint32_t srcRank = 0U; srcRank < tilingData_->runtimeInfo.rankSize; ++srcRank) {
                remoteWindow.WaitPreSumReady(static_cast<int32_t>(srcRank), preSumEpoch);
            }
            PublishGmm2EntryReady(workspaceGM_, tilingData_);
        }
        if (role.physicalBlockId == progressPhysicalBlockId) {
            Gmm2ExpertProgressCoordinator coordinator;
            coordinator.Init(workspaceGM_, tilingData_);
            coordinator.Process();
        }
    }

    if (role.role == kMegaMoeFixedRoleDispatch && role.groupLocalId < dispatchActiveWorkerCount) {
        DispatchGather dispatchGather;
        dispatchGather.Init(workspaceGM_, tilingData_);
        dispatchGather.ProcessFixed(role.groupLocalId, role.groupSize);
        pipe_barrier(PIPE_ALL);
        dsb(DSB_DDR);
        PtoRemoteWindow remoteWindow;
        remoteWindow.Init(reinterpret_cast<GM_ADDR>(tilingData_->runtimeInfo.remoteWindowContext));
        const int32_t epoch = remoteWindow.DataReadyEpoch();
        remoteWindow.PublishDispatchRelease(role.groupLocalId, epoch);
        if (role.groupLocalId == 0U) {
            // Some topologies leave a trailing Dispatch role unused after
            // rank-balanced lane assignment.
            remoteWindow.WaitDispatchReleaseMte(dispatchActiveWorkerCount, epoch);
            PublishScalarEpoch(
                FixedSyncSlot(workspaceGM_, tilingData_, kMegaMoeFixedSyncDispatchDoneSlot),
                kMegaMoeFixedDispatchDoneMarker);
        }
    }

    if (cvSwigluWorker) {
        Swiglu<CType_> swiglu;
        swiglu.Init(workspaceGM_, tilingData_);
        swiglu.ProcessFixed(role.physicalBlockId, fixed.physicalAicNum);
    }

    // Every AIV1 becomes a direct Combine consumer after its paired SwiGLU
    // work. Arm each local consumer before opening the common GMM2 gate.
    if (role.subblockId == 1U) {
        pipe_barrier(PIPE_ALL);
        dsb(DSB_DDR);
        PublishCombineConsumerArmed(workspaceGM_, tilingData_, role.physicalBlockId);
        WaitGmm2EntryReady(workspaceGM_, tilingData_);
        Combine<bfloat16_t> combine;
        combine.Init(workspaceGM_, tilingData_);
        combine.ProcessFixed(role.physicalBlockId);
    }

    // Rank-streaming uses stable logical ordinals: AIV1 is [0, P), AIV0 is
    // [P, 2P). The dedicated coordinator publishes live expert progress;
    // the phase-1 coordinator only snapshots progress for two-phase cases.
    const uint32_t rankStreamingWorkerIdx =
        role.subblockId == 1U ? role.physicalBlockId : fixed.physicalAicNum + role.physicalBlockId;
    const bool twoPhaseUnpermute = tilingData_->unpermuteTiling.rankStreamingInitialWorkerCount != 0U;
    const bool rankStreamingCoordinator =
        rankStreamingWorkerIdx == tilingData_->unpermuteTiling.rankStreamingCoordinatorWorker;
    if (twoPhaseUnpermute && rankStreamingCoordinator) {
        // Group1 handoff is owned by each AIC. Fixed-wave GMM1 publishes its
        // final wave epoch; mailbox GMM1 publishes the done marker only after
        // the AIC has claimed its GMM2 transition ticket.
        const int32_t gmm1HandoffEpoch =
            gmm1Mailbox ? kMegaMoeFixedGmm1DoneMarker :
                          static_cast<int32_t>(fixed.totalWaveCount + kMegaMoeFixedGmm1GroupDoneEpochOffset);
        WaitArrivalMinMte(workspaceGM_, tilingData_, kMegaMoeFixedSyncGmm1ArrivalBase, fixed.gmm1GroupSize,
                          gmm1HandoffEpoch);
        PtoRemoteWindow remoteWindow;
        remoteWindow.Init(reinterpret_cast<GM_ADDR>(tilingData_->runtimeInfo.remoteWindowContext));
        const int32_t epoch = remoteWindow.DataReadyEpoch();
        const uint32_t rankCount = tilingData_->runtimeInfo.rankSize;
        const uint32_t expertPerRank = tilingData_->megaMoeInfo.expertPerRank;
        uint32_t readyExpertCounts[COMBINE_EXPERT_PROGRESS_MAX_RANKS] = {0U};
        remoteWindow.ReadExpertProgressMte(epoch, expertPerRank, readyExpertCounts);
        uint32_t readyRankMask = 0U;
        for (uint32_t producerRank = 0U; producerRank < rankCount; ++producerRank) {
            if (readyExpertCounts[producerRank] != 0U) {
                readyRankMask |= 1U << producerRank;
            }
        }
        remoteWindow.AcquireDataReady();
        remoteWindow.PublishUnpermutePhase1Progress(readyExpertCounts, rankCount, readyRankMask, epoch);
    }

    const bool unpermuteWorker = rankStreamingWorkerIdx < rankStreamingWorkerCount;
    if (unpermuteWorker) {
        Unpermute<bfloat16_t> unpermute;
        if (!unpermute.Init(workspaceGM_, expertIdGM_, probsGM_, outGM_, tilingData_, rankStreamingWorkerIdx,
                            rankStreamingWorkerCount)) {
            return;
        }
        unpermute.Process();
    }
#endif
}

template <typename CType_>
__aicore__ inline void MegaMoe<CType_>::Process()
{
#if defined(__DAV_VEC__)
    FrontReorderProcess<CType_>(xGM_, expertIdGM_, expertTokenNumsGM_, workspaceGM_, tilingData_);
#endif
    ProcessFixedGroups();
}

#endif // DISPATCH_MEGA_COMBINE_H
