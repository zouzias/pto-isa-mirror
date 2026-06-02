/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef MOE_DISPATCH_COMBINE_A8W8_TYPES_HPP_
#define MOE_DISPATCH_COMBINE_A8W8_TYPES_HPP_

#include <cstdint>

namespace moe_new_dispatch_combine_a8w8 {

constexpr uint32_t kPeerWindowMagic = 0x4D443841U; // "MD8A"
constexpr uint32_t kPeerWindowVersion = 1;
constexpr uint32_t kCacheLineBytes = 64;
constexpr uint32_t kPeerWindowHeaderBytes = 256;
constexpr uint32_t kWarmupIters = 0;
constexpr uint32_t kMeasureIters = 1;
constexpr uint32_t kMaxRankNum = 64;
constexpr uint32_t kGmmBaseM = 128;
constexpr uint32_t kGmmBaseN = 256;
constexpr uint32_t kGmmBaseK = 64;
constexpr uint32_t kGmmStepK = 4;
constexpr uint32_t kGmmL1Stages = 2;
constexpr uint32_t kGmmL0AStages = 2;
constexpr uint32_t kGmmL0BStages = 2;
constexpr uint32_t kGmmL0CStages = 1;
constexpr uint32_t kReturnTileRows = 128;
constexpr uint32_t kM2FusedFullConfigMagic = 0x4D328CFU;
constexpr uint32_t kM2FusedFullConfigWords = 64;
constexpr uint32_t kM2FusedFullConfigMagicSlot = 0;
constexpr uint32_t kM2FusedFullShapeRankNumSlot = 1;
constexpr uint32_t kM2FusedFullShapeExpertPerRankSlot = 2;
constexpr uint32_t kM2FusedFullShapeTopKSlot = 3;
constexpr uint32_t kM2FusedFullShapeMSlot = 4;
constexpr uint32_t kM2FusedFullShapeHiddenSizeSlot = 5;
constexpr uint32_t kM2FusedFullShapeIntermediateSizeSlot = 6;
constexpr uint32_t kM2FusedFullShapeMaxTokensPerExpertSlot = 7;
constexpr uint32_t kM2FusedFullShapePayloadTileColsSlot = 8;
constexpr uint32_t kM2FusedFullShapeGmmBlockMSlot = 9;
constexpr uint32_t kM2FusedFullShapeGmmBlockNSlot = 10;
constexpr uint32_t kM2FusedFullShapeGmmBlockKSlot = 11;
constexpr uint32_t kM2FusedFullShapeDtypeInSlot = 12;
constexpr uint32_t kM2FusedFullShapeDtypeOutSlot = 13;
constexpr uint32_t kM2FusedFullRankRankNumSlot = 14;
constexpr uint32_t kM2FusedFullRankRankIdSlot = 15;
constexpr uint32_t kM2FusedFullRankFromMpiSlot = 16;
constexpr uint32_t kM2FusedFullRankDeviceBaseSlot = 17;
constexpr uint32_t kM2FusedFullRankNdevicesSlot = 18;
constexpr uint32_t kM2FusedFullPtrInputASlot = 19;
constexpr uint32_t kM2FusedFullPtrExpertIdxSlot = 21;
constexpr uint32_t kM2FusedFullPtrProbsSlot = 23;
constexpr uint32_t kM2FusedFullPtrOutputCSlot = 25;
constexpr uint32_t kM2FusedFullPtrPeerWindowSlot = 27;
constexpr uint32_t kM2FusedFullPtrHcclCtxSlot = 29;
constexpr uint32_t kM2FusedFullPtrWorkspaceSlot = 31;
constexpr uint32_t kM2FusedFullPtrXActiveMaskSlot = 33;
constexpr uint32_t kM2FusedFullTimelineEnableSlot = 35;
constexpr uint32_t kM2FusedFullOverlapModeSlot = 36;
constexpr uint32_t kM3N11SubtileRows = 16U;
constexpr uint32_t kInitQuantMaxDispatchWorkers = 40U;
constexpr uint32_t kM3N11SubtileCounterBase = 24U * 16U + 128U;
constexpr uint32_t kM3N11SubtileCounterSlots = 16U;
constexpr uint32_t kM3OCombineProbeCounterBase = 24U * 16U + 304U;
constexpr uint32_t kM3OCombineProbeCounterWords = 16U;
constexpr uint32_t kM3ORestoreCounterBase = 160U * 16U;
constexpr uint32_t kM3ORestoreCounterWords = 144U;
constexpr uint32_t kM3ORestoreWorkerCount = 8U;
constexpr uint32_t kM3ORestoreWorkerBase = 16U;
constexpr uint32_t kM3ORestoreWorkerWords = 16U;
constexpr uint32_t kM3N12TimelineRecordWords = 4U;
constexpr uint32_t kM3N12TimelineRecordCount = 64U;
constexpr uint32_t kM3N12TimelineSlotRoute = 0U;
constexpr uint32_t kM3N12TimelineSlotCountSync = 1U;
constexpr uint32_t kM3N12TimelineSlotDispatchGather = 2U;
constexpr uint32_t kM3N12TimelineSlotGmm1 = 3U;
constexpr uint32_t kM3N12TimelineSlotGmm2 = 4U;
constexpr uint32_t kM3N12TimelineSlotCombine = 5U;
constexpr uint32_t kM3N12TimelineSlotRestore = 6U;
constexpr uint32_t kM3N12TimelineSlotRouteCount = 7U;
constexpr uint32_t kM3N12TimelineSlotSwigluBase = 8U;
constexpr uint32_t kM3N12TimelineSwigluSlotCount = 8U;
constexpr uint32_t kM3N12TimelineSlotPrefix = 16U;
constexpr uint32_t kM3N12TimelineSlotInitQuantE2E = 17U;
constexpr uint32_t kM3N12TimelineSlotInitQuantPrepare = 18U;
constexpr uint32_t kM3N12TimelineSlotInitQuantCount = 19U;
constexpr uint32_t kM3N12TimelineSlotInitQuantPrefix = 20U;
constexpr uint32_t kM3N12TimelineSlotInitQuantRoutePackQuant = 21U;

enum class M3N12TimelineKind : uint32_t
{
    kNone = 0,
    kRoute = 1,
    kCountSync = 2,
    kDispatchGather = 3,
    kGmm1Tile = 4,
    kSwigluGroup = 5,
    kGmm2Tile = 6,
    kCombineOwnerSegment = 7,
    kRestore = 8,
    kRouteCount = 9,
    kPrefix = 10,
    kInitQuantE2E = 11,
    kInitQuantPrepare = 12,
    kInitQuantCount = 13,
    kInitQuantPrefix = 14,
    kInitQuantRoutePackQuant = 15,
};

enum class M3N12TimelineCoreType : uint32_t
{
    kUnknown = 0,
    kAic = 1,
    kAiv = 2,
};

enum class M3N12TimelineStatus : uint32_t
{
    kEmpty = 0,
    kProcessed = 1,
    kSkipped = 2,
};

enum class M3N12TimelineWaitSource : uint32_t
{
    kNone = 0,
    kPtoEvent = 1,
    kGmPoll = 2,
    kSyncAll = 4,
};

enum class DType : uint32_t
{
    kFp16 = 1,
    kBf16 = 2,
    kInt8 = 3,
    kFloat = 4,
};

struct ShapeConfig {
    uint32_t rankNum = 2;
    uint32_t expertPerRank = 2;
    uint32_t topK = 2;
    uint32_t m = 16;
    uint32_t hiddenSize = 64;
    uint32_t intermediateSize = 32;
    uint32_t maxTokensPerExpert = 64;
    uint32_t payloadTileCols = 64;
    uint32_t gmmBlockM = kGmmBaseM;
    uint32_t gmmBlockN = kGmmBaseN;
    uint32_t gmmBlockK = kGmmBaseK;
    uint32_t dtypeIn = static_cast<uint32_t>(DType::kFp16);
    uint32_t dtypeOut = static_cast<uint32_t>(DType::kFp16);
};

struct RankConfig {
    uint32_t rankNum = 2;
    uint32_t rankId = 0;
    uint32_t rankFromMpi = 1;
    uint32_t deviceBase = 0;
    uint32_t ndevices = 2;
};

struct M2FusedFullParams {
    ShapeConfig shape;
    RankConfig rank;
    uint32_t debugStopStage = 0;
};

struct M2FusedFullLaunchArgs {
    M2FusedFullParams params;
    uint32_t debugStopStage = 0;
    uint32_t reserved0 = 0;
    uint64_t stageStatusAddr = 0;
    uint32_t shapeRankNum = 0;
    uint32_t shapeExpertPerRank = 0;
    uint32_t shapeTopK = 0;
    uint32_t shapeM = 0;
    uint32_t shapeHiddenSize = 0;
    uint32_t shapeIntermediateSize = 0;
    uint32_t shapeMaxTokensPerExpert = 0;
    uint32_t shapePayloadTileCols = 0;
    uint32_t shapeGmmBlockM = 0;
    uint32_t shapeGmmBlockN = 0;
    uint32_t shapeGmmBlockK = 0;
    uint32_t shapeDtypeIn = 0;
    uint32_t shapeDtypeOut = 0;
    uint32_t rankRankNum = 0;
    uint32_t rankRankId = 0;
    uint32_t rankFromMpi = 0;
    uint32_t rankDeviceBase = 0;
    uint32_t rankNdevices = 0;
    uint64_t inputA = 0;
    uint64_t expertIdx = 0;
    uint64_t xActiveMask = 0;
    uint64_t probs = 0;
    uint64_t outputC = 0;
    uint64_t peerWindow = 0;
    uint64_t hcclCtx = 0;
    uint64_t workspace = 0;
    uint32_t timelineEnable = 0;
    uint32_t overlapMode = 0;
};

struct FieldLayout {
    uint64_t offset = 0;
    uint64_t bytes = 0;
    uint64_t alignment = kCacheLineBytes;
};

struct WorkspaceLayout {
    FieldLayout tokenPerExpertMatrix;
    FieldLayout blockTokenPerExpert;
    FieldLayout blockPrefixPerExpert;
    FieldLayout initQuantWorkerTokenPerExpert;
    FieldLayout initQuantWorkerPrefixPerExpert;
    FieldLayout expandedRowIdx;
    FieldLayout packedRowToRouteIndex;
    FieldLayout dispatchOffset;
    FieldLayout cumsumMM;
    FieldLayout preSumBeforeRank;
    FieldLayout expertTokenNums;
    FieldLayout tokenOwnerRankOffsets;
    FieldLayout dispatchedA;
    FieldLayout dispatchedScale;
    FieldLayout gmm1InputInt8;
    FieldLayout routingPerTokenScale;
    FieldLayout gmm1WeightInt8;
    FieldLayout scale1Uint64;
    FieldLayout gmm1AccInt32;
    FieldLayout gmm1Out;
    FieldLayout swigluOut;
    FieldLayout gmm2InputInt8;
    FieldLayout gmm2PerTokenScale;
    FieldLayout gmm2WeightInt8;
    FieldLayout scale2Uint64;
    FieldLayout gmm2AccInt32;
    FieldLayout gmm2Out;
    FieldLayout returnSegmentStaging;
    FieldLayout readyCounters;
    FieldLayout dispatchGroupReady;
    FieldLayout gmm1SyncGroupReady;
    FieldLayout activationSyncGroupReady;
    FieldLayout gmm2GroupReady;
    FieldLayout stageStatus;
    FieldLayout swigluSyncGroups;
    FieldLayout dequantSum;
    FieldLayout swigluGroupDesc;
    FieldLayout gmm1TileTaskPlan;
    FieldLayout gmm2TileTaskPlan;
    FieldLayout timeoutDump;
    FieldLayout subTileReturnPlan;
    FieldLayout subTileOwnerSegments;
    FieldLayout subTileReady;
    FieldLayout timelineScratch;
    uint64_t totalBytes = 0;
};

struct PeerWindowLayout {
    FieldLayout header;
    FieldLayout tokenPerExpertMatrix;
    FieldLayout countReadySignal;
    FieldLayout dispatchPayload;
    FieldLayout dispatchScale;
    FieldLayout returnPayload;
    FieldLayout combineDoneSignal;
    FieldLayout returnSegmentCounters;
    FieldLayout debugCounters;
    FieldLayout timeline;
    uint64_t dispatchPayloadRowBytes = 0;
    uint64_t returnPayloadRowBytes = 0;
    uint64_t totalBytes = 0;
};

struct PeerWindowHeader {
    uint32_t magic = kPeerWindowMagic;
    uint32_t version = kPeerWindowVersion;
    uint32_t rankNum = 0;
    uint32_t rankId = 0;
    uint32_t expertPerRank = 0;
    uint32_t topK = 0;
    uint32_t dtypeIn = 0;
    uint32_t dtypeOut = 0;
    uint64_t dispatchPayloadRowBytes = 0;
    uint64_t returnPayloadRowBytes = 0;
};

} // namespace moe_new_dispatch_combine_a8w8

#endif // MOE_DISPATCH_COMBINE_A8W8_TYPES_HPP_
