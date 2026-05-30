/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef MOE_DISPATCH_COMBINE_A8W8_KERNEL_CONTROL_METADATA_HPP_
#define MOE_DISPATCH_COMBINE_A8W8_KERNEL_CONTROL_METADATA_HPP_

#include <pto/common/type.hpp>

#include "moe_dispatch_combine_a8w8_types.hpp"

#ifndef GM_ADDR
#define GM_ADDR __gm__ uint8_t *
#endif

namespace moe_dispatch_combine_a8w8 {

constexpr uint32_t kM3N9TimeoutDumpStride = 16U;
constexpr uint32_t kM3N9TimeoutDumpPresentSlot = 0U;
constexpr uint32_t kM3N9TimeoutDumpRankSlot = 1U;
constexpr uint32_t kM3N9TimeoutDumpExpertSlot = 2U;
constexpr uint32_t kM3N9TimeoutDumpTokenOwnerRankSlot = 3U;
constexpr uint32_t kM3N9TimeoutDumpExpertOwnerRankSlot = 4U;
constexpr uint32_t kM3N9TimeoutDumpStageSlot = 5U;
constexpr uint32_t kM3N9TimeoutDumpSignalIdSlot = 6U;
constexpr uint32_t kM3N9TimeoutDumpDebugStopStageSlot = 7U;
constexpr uint32_t kM3N10TimeoutDumpProducerStatusBaseSlot = 8U;
constexpr uint32_t kM3N10TimeoutDumpScoreboardMinStatusSlot = 14U;
constexpr uint32_t kM3N10TimeoutDumpScoreboardDomainSlot = 15U;
constexpr uint32_t kM3N9TimeoutStageDispatchToGmm1 = 1U;

template <typename T>
struct GmArrayView {
    __gm__ T *ptr = nullptr;

    AICORE inline __gm__ T *At(uint64_t index) const
    {
        return ptr + index;
    }
};

struct TokenPerExpertView {
    __gm__ int32_t *ptr = nullptr;
    uint32_t rankNum = 0;
    uint32_t expertPerRank = 0;
    uint32_t rowStride = 0;

    AICORE inline uint64_t Index(uint32_t tokenOwnerRank, uint32_t expertOwnerRank, uint32_t localExpert) const
    {
        return static_cast<uint64_t>(tokenOwnerRank) * rowStride +
               static_cast<uint64_t>(expertOwnerRank) * expertPerRank + localExpert;
    }

    AICORE inline __gm__ int32_t *At(uint32_t tokenOwnerRank, uint32_t expertOwnerRank, uint32_t localExpert) const
    {
        return ptr + Index(tokenOwnerRank, expertOwnerRank, localExpert);
    }
};

struct RankExpertView {
    __gm__ int32_t *ptr = nullptr;
    uint32_t expertPerRank = 0;

    AICORE inline uint64_t Index(uint32_t tokenOwnerRank, uint32_t localExpert) const
    {
        return static_cast<uint64_t>(tokenOwnerRank) * expertPerRank + localExpert;
    }

    AICORE inline __gm__ int32_t *At(uint32_t tokenOwnerRank, uint32_t localExpert) const
    {
        return ptr + Index(tokenOwnerRank, localExpert);
    }
};

struct ControlMetadataView {
    TokenPerExpertView tokenPerExpert;
    GmArrayView<int32_t> blockTokenPerExpert;
    GmArrayView<int32_t> blockPrefixPerExpert;
    GmArrayView<int32_t> expandedRowIdx;
    GmArrayView<int32_t> dispatchOffset;
    RankExpertView cumsumMM;
    RankExpertView preSumBeforeRank;
    GmArrayView<int32_t> expertTokenNums;
    GmArrayView<int32_t> countReadySignal;
    GmArrayView<int32_t> combineDoneSignal;
    GmArrayView<int32_t> debugCounters;
    GmArrayView<int32_t> dispatchGroupReady;
    GmArrayView<int32_t> gmm1SyncGroupReady;
    GmArrayView<int32_t> activationSyncGroupReady;
    GmArrayView<int32_t> gmm2GroupReady;
    GmArrayView<int32_t> swigluSyncGroups;
    GmArrayView<int32_t> dequantSum;
    GmArrayView<int32_t> swigluGroupDesc;
    GmArrayView<int32_t> gmm1TileTaskPlan;
    GmArrayView<int32_t> gmm2TileTaskPlan;
    GmArrayView<int32_t> scoreboardTaskMap;
    GmArrayView<int32_t> producerStatus;
    GmArrayView<int32_t> scoreboardMinStatus;
    GmArrayView<int32_t> workerWaitCounters;
    GmArrayView<int32_t> scoreboardTimeoutCounters;
    GmArrayView<int32_t> subTileReturnPlan;
    GmArrayView<int32_t> subTileOwnerSegments;
    GmArrayView<int32_t> subTileReady;
    GmArrayView<uint64_t> timelineScratch;
    GmArrayView<uint64_t> timeline;
};

AICORE inline __gm__ int32_t *I32Field(GM_ADDR base, const FieldLayout &field)
{
    return reinterpret_cast<__gm__ int32_t *>(base + field.offset);
}

template <typename T>
AICORE inline __gm__ T *TypedField(GM_ADDR base, const FieldLayout &field)
{
    return reinterpret_cast<__gm__ T *>(base + field.offset);
}

struct M2DataPathView {
    GmArrayView<int8_t> gmm1InputInt8;
    GmArrayView<float> routingPerTokenScale;
    GmArrayView<int8_t> gmm1WeightInt8;
    GmArrayView<uint64_t> scale1Uint64;
    GmArrayView<int32_t> gmm1AccInt32;
    GmArrayView<float> gmm1Out;
    GmArrayView<float> swigluOut;
    GmArrayView<int8_t> gmm2InputInt8;
    GmArrayView<float> gmm2PerTokenScale;
    GmArrayView<int8_t> gmm2WeightInt8;
    GmArrayView<uint64_t> scale2Uint64;
    GmArrayView<int32_t> gmm2AccInt32;
    GmArrayView<uint8_t> gmm2OutDebug;
    GmArrayView<uint8_t> returnSegmentStaging;
    GmArrayView<int8_t> peerDispatchPayload;
    GmArrayView<float> peerDispatchScale;
    GmArrayView<uint8_t> peerReturnPayload;
};

AICORE inline ControlMetadataView MakeControlMetadataView(GM_ADDR workspace, GM_ADDR peerWindow,
                                                          const WorkspaceLayout &workspaceLayout,
                                                          const PeerWindowLayout &peerWindowLayout,
                                                          const ShapeConfig &shape)
{
    ControlMetadataView view;
    uint32_t tokenMatrixRowStride = ((shape.rankNum * shape.expertPerRank + 15U) / 16U) * 16U;
    view.tokenPerExpert = TokenPerExpertView{I32Field(peerWindow, peerWindowLayout.tokenPerExpertMatrix), shape.rankNum,
                                             shape.expertPerRank, tokenMatrixRowStride};
    view.blockTokenPerExpert.ptr = I32Field(workspace, workspaceLayout.blockTokenPerExpert);
    view.blockPrefixPerExpert.ptr = I32Field(workspace, workspaceLayout.blockPrefixPerExpert);
    view.expandedRowIdx.ptr = I32Field(workspace, workspaceLayout.expandedRowIdx);
    view.dispatchOffset.ptr = I32Field(workspace, workspaceLayout.dispatchOffset);
    view.cumsumMM = RankExpertView{I32Field(workspace, workspaceLayout.cumsumMM), shape.expertPerRank};
    view.preSumBeforeRank = RankExpertView{I32Field(workspace, workspaceLayout.preSumBeforeRank), shape.expertPerRank};
    view.expertTokenNums.ptr = I32Field(workspace, workspaceLayout.expertTokenNums);
    view.countReadySignal.ptr = I32Field(peerWindow, peerWindowLayout.countReadySignal);
    view.combineDoneSignal.ptr = I32Field(peerWindow, peerWindowLayout.combineDoneSignal);
    view.debugCounters.ptr = I32Field(peerWindow, peerWindowLayout.debugCounters);
    view.dispatchGroupReady.ptr = I32Field(workspace, workspaceLayout.dispatchGroupReady);
    view.gmm1SyncGroupReady.ptr = I32Field(workspace, workspaceLayout.gmm1SyncGroupReady);
    view.activationSyncGroupReady.ptr = I32Field(workspace, workspaceLayout.activationSyncGroupReady);
    view.gmm2GroupReady.ptr = I32Field(workspace, workspaceLayout.gmm2GroupReady);
    view.swigluSyncGroups.ptr = I32Field(workspace, workspaceLayout.swigluSyncGroups);
    view.dequantSum.ptr = I32Field(workspace, workspaceLayout.dequantSum);
    view.swigluGroupDesc.ptr = I32Field(workspace, workspaceLayout.swigluGroupDesc);
    view.gmm1TileTaskPlan.ptr = I32Field(workspace, workspaceLayout.gmm1TileTaskPlan);
    view.gmm2TileTaskPlan.ptr = I32Field(workspace, workspaceLayout.gmm2TileTaskPlan);
    view.scoreboardTaskMap.ptr = I32Field(workspace, workspaceLayout.scoreboardTaskMap);
    view.producerStatus.ptr = I32Field(workspace, workspaceLayout.producerStatus);
    view.scoreboardMinStatus.ptr = I32Field(workspace, workspaceLayout.scoreboardMinStatus);
    view.workerWaitCounters.ptr = I32Field(workspace, workspaceLayout.workerWaitCounters);
    view.scoreboardTimeoutCounters.ptr = I32Field(workspace, workspaceLayout.scoreboardTimeoutCounters);
    view.subTileReturnPlan.ptr = I32Field(workspace, workspaceLayout.subTileReturnPlan);
    view.subTileOwnerSegments.ptr = I32Field(workspace, workspaceLayout.subTileOwnerSegments);
    view.subTileReady.ptr = I32Field(workspace, workspaceLayout.subTileReady);
    view.timelineScratch.ptr = TypedField<uint64_t>(workspace, workspaceLayout.timelineScratch);
    view.timeline.ptr = TypedField<uint64_t>(peerWindow, peerWindowLayout.timeline);
    return view;
}

AICORE inline M2DataPathView MakeM2DataPathView(GM_ADDR workspace, GM_ADDR peerWindow,
                                                const WorkspaceLayout &workspaceLayout,
                                                const PeerWindowLayout &peerWindowLayout)
{
    M2DataPathView view;
    view.gmm1InputInt8.ptr = TypedField<int8_t>(workspace, workspaceLayout.gmm1InputInt8);
    view.routingPerTokenScale.ptr = TypedField<float>(workspace, workspaceLayout.routingPerTokenScale);
    view.gmm1WeightInt8.ptr = TypedField<int8_t>(workspace, workspaceLayout.gmm1WeightInt8);
    view.scale1Uint64.ptr = TypedField<uint64_t>(workspace, workspaceLayout.scale1Uint64);
    view.gmm1AccInt32.ptr = TypedField<int32_t>(workspace, workspaceLayout.gmm1AccInt32);
    view.gmm1Out.ptr = TypedField<float>(workspace, workspaceLayout.gmm1Out);
    view.swigluOut.ptr = TypedField<float>(workspace, workspaceLayout.swigluOut);
    view.gmm2InputInt8.ptr = TypedField<int8_t>(workspace, workspaceLayout.gmm2InputInt8);
    view.gmm2PerTokenScale.ptr = TypedField<float>(workspace, workspaceLayout.gmm2PerTokenScale);
    view.gmm2WeightInt8.ptr = TypedField<int8_t>(workspace, workspaceLayout.gmm2WeightInt8);
    view.scale2Uint64.ptr = TypedField<uint64_t>(workspace, workspaceLayout.scale2Uint64);
    view.gmm2AccInt32.ptr = TypedField<int32_t>(workspace, workspaceLayout.gmm2AccInt32);
    view.gmm2OutDebug.ptr = TypedField<uint8_t>(workspace, workspaceLayout.gmm2Out);
    view.returnSegmentStaging.ptr = TypedField<uint8_t>(workspace, workspaceLayout.returnSegmentStaging);
    view.peerDispatchPayload.ptr = TypedField<int8_t>(peerWindow, peerWindowLayout.dispatchPayload);
    view.peerDispatchScale.ptr = TypedField<float>(peerWindow, peerWindowLayout.dispatchScale);
    view.peerReturnPayload.ptr = TypedField<uint8_t>(peerWindow, peerWindowLayout.returnPayload);
    return view;
}

} // namespace moe_dispatch_combine_a8w8

#endif // MOE_DISPATCH_COMBINE_A8W8_KERNEL_CONTROL_METADATA_HPP_
