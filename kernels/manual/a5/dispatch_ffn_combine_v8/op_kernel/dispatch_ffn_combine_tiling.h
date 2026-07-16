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
 * \file dispatch_ffn_combine_tiling.h
 * \brief
 */

#pragma once

#include <stddef.h>
#include <stdint.h>

struct DispatchFFNCombineInfo {
    uint32_t M;
    uint32_t K;
    uint32_t N;
    uint32_t expertPerRank;
    uint32_t maxOutputSize;
    uint32_t aivNum;
    uint32_t topK;
    uint32_t worldSize;
    uint32_t listLen;
    uint32_t ubMoveNum;
};

struct DispatchFFNCombineRuntimeInfo {
    uint64_t remoteWindowContext = 0;
    uint32_t rank = 0;
    uint32_t rankSize = 0;
};

// Keep uint16_t control fields paired or explicitly padded before uint32_t offset/size fields.
// Other tiling/debug structs with uint64_t fields keep their natural 8-byte alignment.
struct DispatchFFNCombineFrontReorderTiling {
    // Common runtime fields, read by front path selection and downstream stages.
    uint32_t routeElems = 0;
    uint16_t frontCase = 0;
    uint16_t stageNum = 0;
    uint32_t expertNum = 0;
    uint32_t expertNumAligned = 0;

    // Full-load hot fields.
    uint32_t sortLoopMaxElement = 0;
    uint32_t sortLastCorePerLoopElems = 0;
    uint32_t fullLoadMaxRouteElems = 0;

    // Common front/postprocess workspace offsets.
    uint32_t expandedRowIdxOffset = 0;
    uint32_t localTokenPerExpertOffset = 0;
    uint32_t cumsumMMOffset = 0;
    uint32_t preSumBeforeRankOffset = 0;
    uint32_t frontCountScratchBytes = 0;

    // Single/multi sort split fields.
    uint32_t alignedRouteElems = 0;
    uint32_t sortPerCoreElems = 0;
    uint32_t sortLastCoreElems = 0;
    uint32_t sortPerCoreLoops = 0;
    uint32_t sortPerCorePerLoopElems = 0;
    uint32_t sortPerCoreLastLoopElems = 0;
    uint32_t sortLastCoreLoops = 0;
    uint32_t sortLastCoreLastLoopElems = 0;
    uint16_t sortNeedCoreNum = 0;
    uint16_t sortVmsMiddleNeedCoreNum = 0;
    uint16_t sortOutLoopMaxElems = 0;
    uint16_t reservedFrontSortPad = 0;
    uint32_t frontExpandedExpertOffset = 0;
    uint32_t frontExpandDstToSrcOffset = 0;
    uint32_t frontSortWs0Offset = 0;
    uint32_t frontSortWs1Offset = 0;
    uint32_t frontQuantTmpOffset = 0;
    uint32_t frontQuantTmpBytes = 0;

    // Front-end count/check workspace fields.
    uint32_t coreCountOffset = 0;
    uint32_t expertBaseOffset = 0;
    uint32_t coreBaseOffset = 0;
    uint32_t frontWorkspaceBytes = 0;
    uint32_t frontCountScratchOffset = 0;

    // Small-front host/debug fields.
    uint32_t smallRouteElems = 0;
    uint32_t smallFrontUbBytes = 0;
    uint32_t smallFrontRouteElems = 0;
    uint32_t smallFrontAlignedRouteElems = 0;
    uint32_t smallFrontPerCoreRoutes = 0;
    uint32_t smallFrontLastCoreRoutes = 0;
    uint32_t smallFrontSortUbBytes = 0;
    uint32_t smallRouteLimit = 0;
    uint16_t frontPath = 0;
};

// Debug/check controls are kept out of DispatchFFNCombineFrontReorderTiling so the normal front path only reads
// execution fields. uint16_t controls stay paired before uint32_t offset/size fields.
struct DispatchFFNCombineFrontDebugTiling {
    uint16_t smallFrontMode = 0;
    uint16_t smallFrontDebugMode = 0;
    uint16_t smallFrontNeedCoreNum = 0;
    uint16_t reservedSmallFrontPad = 0;
    uint32_t smallFrontDebugOffset = 0;
    uint32_t smallFrontDebugBytes = 0;
    uint32_t smallFrontDebugBytesPerWorker = 0;

    uint16_t frontMode = 0;
    uint16_t frontStopStep = 0;
    uint16_t frontDebugMode = 0;
    uint16_t frontSortDebugStep = 0;
    uint16_t frontCheckMode = 0;
    uint16_t frontCheckDebugMode = 0;
    uint16_t frontCheckStopStep = 0;
    uint16_t frontSortCheckDebugStep = 0;
    uint16_t frontCheckDebugSync = 0;
    uint16_t reservedFrontCheckPad = 0;
    uint32_t frontSortCheckOffset = 0;
    uint32_t frontSortCheckBytes = 0;
    uint32_t frontMergeCheckOffset = 0;
    uint32_t frontMergeCheckBytes = 0;
    uint32_t frontCheckDebugOffset = 0;
    uint32_t frontCheckDebugBytes = 0;
    uint32_t frontCheckDebugBytesPerWorker = 0;
    uint32_t frontDebugOffset = 0;
    uint32_t frontDebugBytes = 0;
    uint32_t frontDebugBytesPerWorker = 0;
};

struct DispatchFFNCombineFrontCheckLayoutDebug {
    uint64_t magic = 0;
    uint64_t workspaceBase = 0;
    uint64_t expandedRowIdxOffset = 0;
    uint64_t coreCountOffset = 0;
    uint64_t localTokenPerExpertOffset = 0;
    uint64_t expertBaseOffset = 0;
    uint64_t coreBaseOffset = 0;
    uint64_t cumsumMMOffset = 0;
    uint64_t preSumBeforeRankOffset = 0;
    uint64_t frontWorkspaceBytes = 0;
    uint64_t frontSortCheckOffset = 0;
    uint64_t frontSortCheckBytes = 0;
    uint64_t frontMergeCheckOffset = 0;
    uint64_t frontMergeCheckBytes = 0;
    uint64_t frontCountScratchOffset = 0;
    uint64_t frontCountScratchBytes = 0;
    uint64_t frontCheckDebugOffset = 0;
    uint64_t frontCheckDebugBytes = 0;
    uint64_t frontCheckDebugBytesPerWorker = 0;
    uint64_t peerOffsetA = 0;
    uint64_t peerOffsetPeerTokenPerExpert = 0;
    uint32_t rank = 0;
    uint32_t rankSize = 0;
    uint32_t coreIdx = 0;
    uint32_t coreNum = 0;
    uint32_t stageNum = 0;
    uint32_t problemM = 0;
    uint32_t problemK = 0;
    uint32_t topK = 0;
    uint32_t expertPerRank = 0;
    uint32_t expertNum = 0;
    uint32_t expertNumAligned = 0;
    uint32_t frontPath = 0;
    uint32_t frontCheckMode = 0;
    uint32_t frontCase = 0;
    uint32_t frontCheckDebugMode = 0;
    uint32_t frontCheckStopStep = 0;
    uint32_t frontSortCheckDebugStep = 0;
    uint32_t frontCheckDebugSync = 0;
    uint32_t routeElems = 0;
    uint32_t alignedRouteElems = 0;
    uint32_t sortLoopMaxElement = 0;
    uint32_t fullLoadMaxRouteElems = 0;
    uint32_t sortNeedCoreNum = 0;
    uint32_t sortPerCoreElems = 0;
    uint32_t sortLastCoreElems = 0;
    uint32_t sortPerCoreLoops = 0;
    uint32_t sortPerCorePerLoopElems = 0;
    uint32_t sortPerCoreLastLoopElems = 0;
    uint32_t sortLastCoreLoops = 0;
    uint32_t sortLastCorePerLoopElems = 0;
    uint32_t sortLastCoreLastLoopElems = 0;
    uint32_t sortVmsMiddleNeedCoreNum = 0;
    uint32_t sortOutLoopMaxElems = 0;
    uint32_t frontMode = 0;
    uint32_t helperDebugMask = 0;
    uint32_t helperDebugFailures = 0;
    uint32_t helperSortLenFloat33 = 0;
    uint32_t helperSortOffsetFloat33 = 0;
    uint32_t helperPackedStride = 0;
    uint32_t helperPackedScaleOffset = 0;
    int32_t quantExpectedRows = 0;
    int32_t quantActualRows = 0;
    int32_t quantExpectedStores = 0;
    int32_t quantActualStores = 0;
    int32_t quantActiveCores = 0;
    int32_t quantFullRowMode = 0;
    int32_t quantColumnTiledMode = 0;
    int32_t quantMarker = 0;
    uint32_t marker = 0;
    uint64_t reserved1[12] = {0};
};

struct DispatchFFNCombineFrontLayoutDebug {
    uint64_t magic = 0;
    uint64_t workspaceBase = 0;
    uint64_t expandedRowIdxOffset = 0;
    uint64_t frontExpandedExpertOffset = 0;
    uint64_t frontExpandDstToSrcOffset = 0;
    uint64_t frontSortWs0Offset = 0;
    uint64_t frontSortWs1Offset = 0;
    uint64_t frontQuantTmpOffset = 0;
    uint64_t frontQuantTmpBytes = 0;
    uint64_t frontDebugOffset = 0;
    uint64_t frontDebugBytes = 0;
    uint64_t frontWorkspaceBytes = 0;
    uint64_t peerOffsetA = 0;
    uint64_t peerOffsetPeerTokenPerExpert = 0;
    uint32_t rank = 0;
    uint32_t rankSize = 0;
    uint32_t coreIdx = 0;
    uint32_t coreNum = 0;
    uint32_t stageNum = 0;
    uint32_t problemM = 0;
    uint32_t problemK = 0;
    uint32_t topK = 0;
    uint32_t expertPerRank = 0;
    uint32_t expertNum = 0;
    uint32_t expertNumAligned = 0;
    uint32_t frontCase = 0;
    uint32_t frontMode = 0;
    uint32_t frontDebugMode = 0;
    uint32_t frontStopStep = 0;
    uint32_t frontSortDebugStep = 0;
    uint32_t routeElems = 0;
    uint32_t alignedRouteElems = 0;
    uint32_t sortNeedCoreNum = 0;
    uint32_t sortPerCoreElems = 0;
    uint32_t sortLastCoreElems = 0;
    uint32_t sortPerCoreLoops = 0;
    uint32_t sortPerCorePerLoopElems = 0;
    uint32_t sortPerCoreLastLoopElems = 0;
    uint32_t sortLastCoreLoops = 0;
    uint32_t sortLastCorePerLoopElems = 0;
    uint32_t sortLastCoreLastLoopElems = 0;
    uint32_t sortVmsMiddleNeedCoreNum = 0;
    uint32_t sortOutLoopMaxElems = 0;
    uint32_t activeCopyCores = 0;
    uint32_t oneCoreVmsActive = 0;
    uint32_t middleMergeActive = 0;
    uint32_t sortOutSrcWsIndex = 0;
    uint32_t noOldNotifyWait = 0;
    uint32_t producerClass = 0;
    uint32_t sortOwnerCore = 0;
    uint32_t activeAivNum = 0;
    uint32_t totalLength = 0;
    uint32_t sortLoopMaxElement = 0;
    uint32_t oneCoreCondition = 0;
    uint32_t routeElemsAlias = 0;
    uint32_t tileLength = 0;
    uint32_t sortNum = 0;
    uint32_t singleSortUbFits = 0;
    uint64_t requiredUbBytesSingleSort = 0;
    uint64_t fullLoadRequiredUbBytes = 0;
    uint64_t fullLoadRemainUbBytes = 0;
    uint32_t fullLoadCondition = 0;
    uint32_t fullLoadNeedCoreNum = 0;
    uint32_t fullLoadPerCoreRows = 0;
    uint32_t fullLoadLastCoreRows = 0;
    uint32_t fullLoadActivateRows = 0;
    uint32_t fullLoadCoreRows = 0;
    uint32_t fullLoadColsScale = 0;
    uint32_t fullLoadRequiredUbFits = 0;
    uint32_t fullLoadSortInputLoaded = 0;
    uint32_t fullLoadPayloadArangeTotalLength = 0;
    uint32_t fullLoadAllActiveCoresSort = 0;
    uint32_t fullLoadSecondSortUsed = 0;
    uint32_t fullLoadExpandedRowIdxLocalReady = 0;
    uint32_t fullLoadOldSmallPathBusinessUsed = 0;
    uint32_t fullLoadCopyOutIdxBlock0Only = 0;
    uint32_t fullLoadExpandedRowIdxQueueReEnqueued = 0;
    uint32_t fullLoadNonOwnerCopyOutIdxStores = 0;
    uint32_t fullLoadCountOwnerCore = 0;
    uint32_t fullLoadCopyOutEmptyCores = 0;
    uint32_t fullLoadExpertTokensFlag = 0;
    uint32_t sortRepeat = 0;
    uint32_t core0Only = 0;
    uint32_t sortInputLoaded = 0;
    uint32_t payloadArangeToSortNum = 0;
    uint32_t keyNegExpert = 0;
    uint32_t tailRuleReference = 0;
    uint32_t extractMaskOk = 0;
    uint32_t restoreExpertCastRound = 0;
    uint32_t sortOutputStored = 0;
    uint32_t expertTokenNeedCoreNum = 0;
    uint32_t expertTokenPerCoreRows = 0;
    uint32_t expertTokenLastCoreRows = 0;
    uint32_t expertTokenPerCorePerLoopRows = 0;
    uint32_t expertTokenPerCoreLastLoopRows = 0;
    uint32_t expertTokenLastCorePerLoopRows = 0;
    uint32_t expertTokenLastCoreLastLoopRows = 0;
    uint32_t expertTokenExpertNumUbAlign = 0;
    uint32_t sortedScanUsed = 0;
    uint32_t atomicCopyoutOnly = 0;
    uint32_t histogramMainPathUsed = 0;
    uint32_t expertTokenOutputStored = 0;
    uint32_t srcToDstAssistLayoutOk = 0;
    uint32_t srcToDstSingleIntStore = 0;
    uint32_t srcToDstInputLoaded = 0;
    uint32_t srcToDstOutputComputed = 0;
    uint32_t srcToDstOutputStored = 0;
    uint32_t srcToDstStageSync = 0;
    uint32_t gatherSmoothType = 0;
    uint32_t gatherColLoops = 0;
    uint32_t gatherNeedCoreNum = 0;
    uint32_t gatherPerCoreRows = 0;
    uint32_t gatherLastCoreRows = 0;
    uint32_t gatherPerCorePerLoopRows = 0;
    uint32_t gatherPerCoreLastLoopRows = 0;
    uint32_t gatherLastCorePerLoopRows = 0;
    uint32_t gatherLastCoreLastLoopRows = 0;
    uint32_t gatherRowLoops = 0;
    uint32_t gatherFullRowQuant = 0;
    uint32_t gatherColLoopsSinglePass = 0;
    uint32_t gatherDynamicQuantScaleSeparateStore = 0;
    uint32_t gatherV5PackedContract = 0;
    uint32_t gatherInlineScaleBytes = 0;
    uint32_t gatherPackedStride = 0;
    uint32_t gatherZeroScaleGuardUsed = 0;
    uint32_t gatherOutputStored = 0;
    uint32_t gatherBoundarySync = 0;
    uint32_t countMarkerBias = 0;
    uint32_t tokenPerExpertRowBytes = 0;
    uint32_t countPublishedRanks = 0;
    uint32_t countWaitedRanks = 0;
    uint32_t countMarkerRestored = 0;
    uint32_t countPreSumBuilt = 0;
    uint32_t countOldNotifyWaitUsed = 0;
    uint32_t cumsumCore0Only = 0;
    uint32_t cumsumRows = 0;
    uint32_t cumsumLocalExpertCount = 0;
    uint32_t cumsumFormulaReference = 0;
    uint32_t cumsumExpertTokenNumsWritten = 0;
    uint32_t offsetAPublished = 0;
    uint32_t tokenPerExpertPublished = 0;
    uint32_t preSumPublished = 0;
    uint32_t cumsumPublished = 0;
    uint32_t expertTokenNumsPublished = 0;
    uint32_t dispatchContractReady = 0;
    uint32_t marker = 0;
};

struct DispatchFFNCombineDispatchTiling {
    uint64_t gmAOffset = 0;
    uint64_t perTokenScaleOffset = 0;
    uint64_t dispatchScratchOffset = 0;
    uint64_t dispatchScratchBytes = 0;
    uint64_t dispatchScratchBytesPerAiv = 0;
    uint64_t dispatchDebugOffset = 0;
    uint64_t dispatchDebugBytes = 0;
    uint64_t dispatchTileBytes = 0;
    uint32_t dispatchDebugMode = 0;
    uint32_t dispatchGatherMode = 0;
    uint32_t dispatchGatherDebugMode = 0;
    uint32_t dispatchGatherStopStep = 0;
    uint32_t dispatchGatherProfileMode = 0;
    uint32_t dispatchGatherDebugMaxGroup = 0xFFFFFFFFU;
    uint32_t dispatchGatherDebugMaxSrcRank = 0xFFFFFFFFU;
    uint32_t dispatchGatherDebugMaxBatches = 0xFFFFFFFFU;
    uint32_t dispatchGatherDebugCutPoint = 0;
    uint64_t dispatchGatherScratchOffset = 0;
    uint64_t dispatchGatherScratchBytes = 0;
    uint64_t dispatchGatherScratchBytesPerAiv = 0;
    uint64_t dispatchGatherDebugOffset = 0;
    uint64_t dispatchGatherDebugBytes = 0;
    uint64_t dispatchGatherTileBytes = 0;
};

struct DispatchFFNCombineDispatchLayoutDebug {
    uint64_t magic = 0;
    uint64_t workspaceBase = 0;
    uint64_t gmAOffset = 0;
    uint64_t perTokenScaleOffset = 0;
    uint64_t dispatchScratchOffset = 0;
    uint64_t dispatchScratchBytes = 0;
    uint64_t dispatchScratchBytesPerAiv = 0;
    uint64_t dispatchScratchCoreOffset = 0;
    uint64_t dispatchDebugOffset = 0;
    uint64_t dispatchDebugBytes = 0;
    uint64_t dispatchTileBytes = 0;
    uint64_t cumsumMMOffset = 0;
    uint64_t preSumBeforeRankOffset = 0;
    uint64_t frontWorkspaceBytes = 0;
    uint64_t peerOffsetA = 0;
    uint64_t peerOffsetPeerTokenPerExpert = 0;
    uint32_t rank = 0;
    uint32_t rankSize = 0;
    uint32_t coreIdx = 0;
    uint32_t coreNum = 0;
    uint32_t stageNum = 0;
    uint32_t problemK = 0;
    uint32_t maxOutputSize = 0;
    uint32_t packedStride = 0;
};

struct DispatchFFNCombineDispatchGatherLayoutDebug {
    uint64_t magic = 0;
    uint64_t workspaceBase = 0;
    uint64_t gmAOffset = 0;
    uint64_t perTokenScaleOffset = 0;
    uint64_t dispatchGatherScratchOffset = 0;
    uint64_t dispatchGatherScratchBytes = 0;
    uint64_t dispatchGatherScratchBytesPerAiv = 0;
    uint64_t dispatchGatherScratchCoreOffset = 0;
    uint64_t dispatchGatherDebugOffset = 0;
    uint64_t dispatchGatherDebugBytes = 0;
    uint64_t dispatchGatherTileBytes = 0;
    uint64_t cumsumMMOffset = 0;
    uint64_t preSumBeforeRankOffset = 0;
    uint64_t peerOffsetA = 0;
    uint64_t peerOffsetPeerTokenPerExpert = 0;
    uint32_t rank = 0;
    uint32_t rankSize = 0;
    uint32_t coreIdx = 0;
    uint32_t coreNum = 0;
    uint32_t stageNum = 0;
    uint32_t problemK = 0;
    uint32_t maxOutputSize = 0;
    uint32_t packedStride = 0;
    uint32_t activeCopyCores = 0;
    uint32_t dispatchGatherMode = 0;
    uint32_t dispatchGatherDebugMode = 0;
    uint32_t dispatchGatherStopStep = 0;
    uint32_t marker = 0;
    uint64_t reserved1[16] = {0};
};

struct DispatchFFNCombineDispatchMetadataDebug {
    uint32_t groupIdx = 0;
    uint32_t srcRank = 0;
    uint32_t groupBase = 0;
    uint32_t currentM = 0;
    uint32_t rawRows = 0;
    uint32_t rows = 0;
    uint32_t srcRowBase = 0;
    uint32_t dstRowBase = 0;
};

struct DispatchFFNCombineDispatchTaskStatsDebug {
    uint32_t groupIdx = 0;
    uint32_t coreIdx = 0;
    uint32_t rankSize = 0;
    uint32_t assignedBlockCount = 0;
    uint32_t coreRowCount = 0;
    uint32_t groupBase = 0;
    uint32_t currentM = 0;
    uint32_t firstSrcRank = 0;
    uint32_t firstRows = 0;
    uint32_t firstSrcRowBase = 0;
    uint32_t firstDstRowBase = 0;
    uint32_t lastSrcRank = 0;
    uint32_t lastRows = 0;
    uint32_t lastSrcRowBase = 0;
    uint32_t lastDstRowBase = 0;
    uint32_t rowBlockRows = 0;
    uint32_t maxBlocksPerGroup = 0;
};

struct DispatchFFNCombineDispatchCopyChunkDebug {
    uint64_t magic = 0;
    uint64_t remoteOffsetBytes = 0;
    uint64_t gmAOffsetBytes = 0;
    uint64_t scaleOffsetBytes = 0;
    uint32_t groupIdx = 0;
    uint32_t srcRank = 0;
    uint32_t coreIdx = 0;
    uint32_t coreNum = 0;
    uint32_t groupBase = 0;
    uint32_t currentM = 0;
    uint32_t rawRows = 0;
    uint32_t rows = 0;
    uint32_t chunkRows = 0;
    uint32_t tileRows = 0;
    uint32_t srcRowBase = 0;
    uint32_t dstRowBase = 0;
    uint32_t packedStride = 0;
    uint32_t marker = 0;
};

struct DispatchFFNCombineDispatchGroupDoneDebug {
    uint64_t magic = 0;
    uint32_t groupIdx = 0;
    uint32_t rank = 0;
    uint32_t rankSize = 0;
    uint32_t groupBase = 0;
    uint32_t currentM = 0;
    uint32_t marker = 0;
    uint32_t reserved0 = 0;
};

struct DispatchFFNCombineGmm1Tiling {
    uint64_t gmCOffset = 0;
    uint64_t gmm1DebugOffset = 0;
    uint64_t gmm1DebugBytes = 0;
    uint64_t reserved0 = 0;
    uint32_t gmm1DebugMode = 0;
    uint32_t l1TileM = 128;
    uint32_t l1TileN = 256;
    uint32_t l1TileK = 512;
    uint32_t l0TileM = 128;
    uint32_t l0TileN = 256;
    uint32_t l0TileK = 128;
    uint32_t reserved1 = 0;
};

struct DispatchFFNCombineGmm1LayoutDebug {
    uint64_t magic = 0;
    uint64_t workspaceBase = 0;
    uint64_t gmAOffset = 0;
    uint64_t gmCOffset = 0;
    uint64_t perTokenScaleOffset = 0;
    uint64_t cumsumMMOffset = 0;
    uint64_t expertTokenNumsBase = 0;
    uint64_t weight1Base = 0;
    uint64_t scale1Base = 0;
    uint64_t gmm1DebugOffset = 0;
    uint64_t gmm1DebugBytes = 0;
    uint32_t rank = 0;
    uint32_t rankSize = 0;
    uint32_t coreIdx = 0;
    uint32_t coreNum = 0;
    uint32_t stageNum = 0;
    uint32_t problemK = 0;
    uint32_t problemN = 0;
    uint32_t maxOutputSize = 0;
    uint32_t expertPerRank = 0;
    uint32_t l1TileM = 0;
    uint32_t l1TileN = 0;
    uint32_t l1TileK = 0;
    uint32_t l0TileM = 0;
    uint32_t l0TileN = 0;
    uint32_t l0TileK = 0;
    uint32_t debugMode = 0;
    uint64_t reserved1[45] = {0};
};

struct DispatchFFNCombineGmm1SyncDebug {
    uint64_t magic = 0;
    uint32_t groupIdx = 0;
    uint32_t coreIdx = 0;
    uint32_t flagId = 0;
    uint32_t rank = 0;
    uint32_t rankSize = 0;
    uint32_t groupBase = 0;
    uint32_t currentM = 0;
    uint32_t marker = 0;
    uint32_t reserved0[6] = {0};
};

struct DispatchFFNCombineGmm1TaskDebug {
    uint64_t magic = 0;
    uint32_t groupIdx = 0;
    uint32_t coreIdx = 0;
    uint32_t rank = 0;
    uint32_t rankSize = 0;
    uint32_t groupBase = 0;
    uint32_t currentMRaw = 0;
    uint32_t currentM = 0;
    uint32_t expertTokenNums = 0;
    uint32_t tileM = 0;
    uint32_t tileN = 0;
    uint32_t coreLoops = 0;
    uint32_t startCoreIdx = 0;
    uint32_t startLoopIdx = 0;
    uint32_t assignedTileCount = 0;
    uint32_t firstLoop = 0;
    uint32_t lastLoop = 0;
    uint32_t firstBlockM = 0;
    uint32_t firstBlockN = 0;
    uint32_t lastBlockM = 0;
    uint32_t lastBlockN = 0;
    uint32_t firstActualM = 0;
    uint32_t firstActualN = 0;
    uint32_t lastActualM = 0;
    uint32_t lastActualN = 0;
    uint32_t marker = 0;
    uint32_t reserved0[5] = {0};
};

struct DispatchFFNCombineGmm1DoneDebug {
    uint64_t magic = 0;
    uint32_t segmentIdx = 0;
    uint32_t coreIdx = 0;
    uint32_t rank = 0;
    uint32_t rankSize = 0;
    uint32_t segmentStartExpert = 0;
    uint32_t segmentEndExpert = 0;
    uint32_t segmentRowBase = 0;
    uint32_t segmentRows = 0;
    uint32_t c2vFlagId = 0;
    uint32_t c2vEnabled = 0;
    uint32_t marker = 0;
    uint32_t segmentPolicy = 0;
    uint32_t reserved0[2] = {0};
};

struct DispatchFFNCombineSwigluTiling {
    uint64_t gmPermutedTokenOffset = 0;
    uint64_t perTokenScale2Offset = 0;
    uint64_t swigluDebugOffset = 0;
    uint64_t swigluDebugBytes = 0;
    uint64_t swigluSegmentMetaOffset = 0;
    uint64_t swigluSegmentMetaBytes = 0;
    uint32_t swigluDebugMode = 0;
    uint32_t swigluTileElems = 1024;
    uint32_t swigluSegmentNum = 0;
    uint32_t swigluEpilogueGranularity = 0;
    uint32_t swigluUbStages = 2;
    uint32_t swigluMetadataMode = 0;
    uint32_t reserved0[2] = {0};
};

struct DispatchFFNCombineSwigluSegmentRuntimeMeta {
    uint32_t segmentIdx = 0;
    uint32_t segmentStartExpert = 0;
    uint32_t segmentEndExpert = 0;
    uint32_t segmentRowBase = 0;
    uint32_t segmentRows = 0;
    uint32_t cumsumRows = 0;
    uint32_t expertTokenRows = 0;
    uint32_t rowSplitBase = 0;
    uint32_t rowSplitRem = 0;
    uint32_t valid = 0;
    uint32_t generation = 0;
    uint32_t producerCoreIdx = 0;
    uint32_t metadataMode = 0;
    uint32_t segmentNum = 0;
    uint32_t epilogueGranularity = 0;
    uint32_t marker = 0;
};

struct DispatchFFNCombineSwigluLayoutDebug {
    uint64_t magic = 0;
    uint64_t workspaceBase = 0;
    uint64_t gmCOffset = 0;
    uint64_t perTokenScaleOffset = 0;
    uint64_t gmPermutedTokenOffset = 0;
    uint64_t perTokenScale2Offset = 0;
    uint64_t cumsumMMOffset = 0;
    uint64_t expertTokenNumsBase = 0;
    uint64_t swigluDebugOffset = 0;
    uint64_t swigluDebugBytes = 0;
    uint64_t swigluSegmentMetaOffset = 0;
    uint64_t swigluSegmentMetaBytes = 0;
    uint64_t gmCBytes = 0;
    uint64_t gmPermutedTokenBytes = 0;
    uint64_t perTokenScaleBytes = 0;
    uint64_t perTokenScale2Bytes = 0;
    uint64_t frontWorkspaceBytes = 0;
    uint64_t dispatchScratchOffset = 0;
    uint64_t gmm1DebugOffset = 0;
    uint32_t rank = 0;
    uint32_t rankSize = 0;
    uint32_t coreIdx = 0;
    uint32_t coreNum = 0;
    uint32_t stageNum = 0;
    uint32_t problemN = 0;
    uint32_t outputN = 0;
    uint32_t maxOutputSize = 0;
    uint32_t expertPerRank = 0;
    uint32_t segmentNum = 0;
    uint32_t epilogueGranularity = 0;
    uint32_t tileElems = 0;
    uint32_t ubStages = 0;
    uint32_t debugMode = 0;
    uint32_t pipelineMode = 0;
    uint32_t scale2BufferNum = 0;
    uint32_t metadataMode = 0;
    uint32_t gmCRowBytes = 0;
    uint32_t gmPermutedTokenRowBytes = 0;
    uint32_t perTokenScaleBytesPerRow = 0;
    uint32_t perTokenScale2BytesPerRow = 0;
    uint32_t layoutVersion = 0;
    uint32_t marker = 0;
    uint64_t reserved1[33] = {0};
};

struct DispatchFFNCombineSwigluC2VDebug {
    uint64_t magic = 0;
    uint32_t segmentIdx = 0;
    uint32_t coreIdx = 0;
    uint32_t rank = 0;
    uint32_t rankSize = 0;
    uint32_t stageNum = 0;
    uint32_t waitSource = 0;
    uint32_t c2vFlagId = 0;
    uint32_t c2vEnabled = 0;
    uint32_t gmm1DoneConsumed = 0;
    uint32_t aivSyncAfterWait = 0;
    uint32_t marker = 0;
    uint32_t reserved0[3] = {0};
};

struct DispatchFFNCombineSwigluSegmentDebug {
    uint64_t magic = 0;
    uint32_t segmentIdx = 0;
    uint32_t rank = 0;
    uint32_t rankSize = 0;
    uint32_t segmentStartExpert = 0;
    uint32_t segmentEndExpert = 0;
    uint32_t segmentRowBase = 0;
    uint32_t segmentRows = 0;
    uint32_t cumsumRows = 0;
    uint32_t expertTokenRows = 0;
    uint32_t maxOutputSize = 0;
    uint32_t segmentNum = 0;
    uint32_t epilogueGranularity = 0;
    uint32_t marker = 0;
    uint32_t reserved0 = 0;
};

struct DispatchFFNCombineSwigluTaskDebug {
    uint64_t magic = 0;
    uint32_t segmentIdx = 0;
    uint32_t coreIdx = 0;
    uint32_t rank = 0;
    uint32_t rankSize = 0;
    uint32_t segmentStartExpert = 0;
    uint32_t segmentEndExpert = 0;
    uint32_t segmentRowBase = 0;
    uint32_t segmentRows = 0;
    uint32_t localRowStart = 0;
    uint32_t localRows = 0;
    uint32_t rowSplitBase = 0;
    uint32_t rowSplitRem = 0;
    uint32_t marker = 0;
    uint32_t reserved0 = 0;
};

struct DispatchFFNCombineSwigluDoneDebug {
    uint64_t magic = 0;
    uint32_t segmentIdx = 0;
    uint32_t coreIdx = 0;
    uint32_t rank = 0;
    uint32_t rankSize = 0;
    uint32_t stageNum = 0;
    uint32_t segmentRowBase = 0;
    uint32_t segmentRows = 0;
    uint32_t localRowStart = 0;
    uint32_t localRows = 0;
    uint32_t aivSyncBeforeDone = 0;
    uint32_t v2cFlagId = 0;
    uint32_t v2cEnabled = 0;
    uint32_t marker = 0;
    uint32_t reserved0 = 0;
};

struct DispatchFFNCombineSwigluFinalSyncDebug {
    uint64_t magic = 0;
    uint32_t coreIdx = 0;
    uint32_t rank = 0;
    uint32_t rankSize = 0;
    uint32_t stageNum = 0;
    uint32_t segmentNum = 0;
    uint32_t finalSyncEnabled = 0;
    uint32_t finalSyncAfterAllSegments = 0;
    uint32_t v2cEnabled = 0;
    uint32_t marker = 0;
    uint32_t reserved0[5] = {0};
};

struct DispatchFFNCombineGmm2Tiling {
    uint64_t gmm2OutputOffset = 0;
    uint64_t gmm2DebugOffset = 0;
    uint64_t gmm2DebugBytes = 0;
    uint32_t gmm2DebugMode = 0;
    uint32_t l1TileM = 128;
    uint32_t l1TileN = 256;
    uint32_t l1TileK = 512;
    uint32_t l0TileM = 128;
    uint32_t l0TileN = 256;
    uint32_t l0TileK = 128;
    uint32_t reserved0[3] = {0};
};

struct DispatchFFNCombineGmm2LayoutDebug {
    uint64_t magic = 0;
    uint64_t workspaceBase = 0;
    uint64_t gmPermutedTokenOffset = 0;
    uint64_t perTokenScale2Offset = 0;
    uint64_t gmm2OutputOffset = 0;
    uint64_t weight2Base = 0;
    uint64_t scale2Base = 0;
    uint64_t cumsumMMOffset = 0;
    uint64_t expertTokenNumsBase = 0;
    uint64_t gmm2DebugOffset = 0;
    uint64_t gmm2DebugBytes = 0;
    uint64_t frontWorkspaceBytes = 0;
    uint64_t dispatchScratchOffset = 0;
    uint64_t gmm1DebugOffset = 0;
    uint64_t swigluDebugOffset = 0;
    uint64_t gmPermutedTokenBytes = 0;
    uint64_t perTokenScale2Bytes = 0;
    uint64_t gmm2OutputBytes = 0;
    uint32_t rank = 0;
    uint32_t rankSize = 0;
    uint32_t coreIdx = 0;
    uint32_t coreNum = 0;
    uint32_t stageNum = 0;
    uint32_t problemK = 0;
    uint32_t problemN = 0;
    uint32_t inputK = 0;
    uint32_t outputN = 0;
    uint32_t maxOutputSize = 0;
    uint32_t expertPerRank = 0;
    uint32_t segmentNum = 0;
    uint32_t epilogueGranularity = 0;
    uint32_t l1TileM = 0;
    uint32_t l1TileN = 0;
    uint32_t l1TileK = 0;
    uint32_t l0TileM = 0;
    uint32_t l0TileN = 0;
    uint32_t l0TileK = 0;
    uint32_t debugMode = 0;
    uint32_t commonReuseMode = 0;
    uint32_t gmPermutedTokenRowBytes = 0;
    uint32_t gmm2OutputRowBytes = 0;
    uint32_t perTokenScale2BytesPerRow = 0;
    uint32_t layoutVersion = 0;
    uint32_t marker = 0;
    uint64_t reserved1[33] = {0};
};

struct DispatchFFNCombineGmm2V2CDebug {
    uint64_t magic = 0;
    uint32_t segmentIdx = 0;
    uint32_t coreIdx = 0;
    uint32_t rank = 0;
    uint32_t rankSize = 0;
    uint32_t stageNum = 0;
    uint32_t waitSource = 0;
    uint32_t v2cFlagId = 0;
    uint32_t v2cEnabled = 0;
    uint32_t swigluDoneConsumed = 0;
    uint32_t finalSyncConsumed = 0;
    uint32_t marker = 0;
    uint32_t reserved0[3] = {0};
};

struct DispatchFFNCombineGmm2SegmentDebug {
    uint64_t magic = 0;
    uint32_t segmentIdx = 0;
    uint32_t rank = 0;
    uint32_t rankSize = 0;
    uint32_t segmentStartExpert = 0;
    uint32_t segmentEndExpert = 0;
    uint32_t segmentRowBase = 0;
    uint32_t segmentRows = 0;
    uint32_t cumsumRows = 0;
    uint32_t expertTokenRows = 0;
    uint32_t maxOutputSize = 0;
    uint32_t segmentNum = 0;
    uint32_t epilogueGranularity = 0;
    uint32_t marker = 0;
    uint32_t reserved0 = 0;
};

struct DispatchFFNCombineGmm2TaskDebug {
    uint64_t magic = 0;
    uint32_t segmentIdx = 0;
    uint32_t groupIdx = 0;
    uint32_t coreIdx = 0;
    uint32_t rank = 0;
    uint32_t rankSize = 0;
    uint32_t groupBase = 0;
    uint32_t currentMRaw = 0;
    uint32_t currentM = 0;
    uint32_t expertTokenNums = 0;
    uint32_t tileM = 0;
    uint32_t tileN = 0;
    uint32_t coreLoops = 0;
    uint32_t startCoreIdx = 0;
    uint32_t startLoopIdx = 0;
    uint32_t assignedTileCount = 0;
    uint32_t firstLoop = 0;
    uint32_t lastLoop = 0;
    uint32_t firstBlockM = 0;
    uint32_t firstBlockN = 0;
    uint32_t lastBlockM = 0;
    uint32_t lastBlockN = 0;
    uint32_t firstActualM = 0;
    uint32_t firstActualN = 0;
    uint32_t lastActualM = 0;
    uint32_t lastActualN = 0;
    uint32_t marker = 0;
    uint32_t reserved0[4] = {0};
};

struct DispatchFFNCombineGmm2DoneDebug {
    uint64_t magic = 0;
    uint32_t coreIdx = 0;
    uint32_t rank = 0;
    uint32_t rankSize = 0;
    uint32_t stageNum = 0;
    uint32_t segmentNum = 0;
    uint32_t expertPerRank = 0;
    uint32_t finalSyncEnabled = 0;
    uint32_t combineReadyEnabled = 0;
    uint32_t pipelineDrained = 0;
    uint32_t marker = 0;
    uint32_t reserved0[4] = {0};
};

struct DispatchFFNCombineGmm2FinalSyncDebug {
    uint64_t magic = 0;
    uint32_t coreIdx = 0;
    uint32_t rank = 0;
    uint32_t rankSize = 0;
    uint32_t stageNum = 0;
    uint32_t segmentNum = 0;
    uint32_t expertPerRank = 0;
    uint32_t finalSyncEnabled = 0;
    uint32_t finalSyncAfterAllGroups = 0;
    uint32_t combineReadyEnabled = 0;
    uint32_t marker = 0;
    uint32_t reserved0[4] = {0};
};

struct DispatchFFNCombineCombineTiling {
    uint64_t gmm2OutputOffset = 0;
    uint64_t perTokenScale2Offset = 0;
    uint64_t combineScratchOffset = 0;
    uint64_t combineScratchBytes = 0;
    uint64_t combineScratchBytesPerAiv = 0;
    uint64_t combineDebugOffset = 0;
    uint64_t combineDebugBytes = 0;
    uint32_t combineTileCols = 1024;
    uint32_t combineDebugMode = 0;
    uint32_t combineImplMode = 0;
    uint32_t combineStopStep = 0;
    uint32_t gmm2CombineCvMode = 0;
    uint32_t gmm2CombineCvDebugMode = 0;
};

struct DispatchFFNCombineCombineLayoutDebug {
    uint64_t magic = 0;
    uint64_t workspaceBase = 0;
    uint64_t gmm2OutputOffset = 0;
    uint64_t perTokenScale2Offset = 0;
    uint64_t cumsumMMOffset = 0;
    uint64_t preSumBeforeRankOffset = 0;
    uint64_t combineScratchOffset = 0;
    uint64_t combineScratchBytes = 0;
    uint64_t combineScratchBytesPerAiv = 0;
    uint64_t combineScratchCoreOffset = 0;
    uint64_t combineDebugOffset = 0;
    uint64_t combineDebugBytes = 0;
    uint64_t peerOffsetD = 0;
    uint64_t peerOffsetPeerTokenPerExpert = 0;
    uint64_t peerSignalBaseOffset = 0;
    uint64_t gmm2OutputBytes = 0;
    uint64_t perTokenScale2Bytes = 0;
    uint64_t offsetDBytes = 0;
    uint32_t rank = 0;
    uint32_t rankSize = 0;
    uint32_t coreIdx = 0;
    uint32_t coreNum = 0;
    uint32_t stageNum = 0;
    uint32_t problemK = 0;
    uint32_t maxOutputSize = 0;
    uint32_t expertPerRank = 0;
    uint32_t combineTileCols = 0;
    uint32_t combineTileRows = 0;
    uint32_t debugMode = 0;
    uint32_t outputElementBytes = 0;
    uint32_t gmm2OutputRowBytes = 0;
    uint32_t offsetDRowBytes = 0;
    uint32_t tileBytes = 0;
    uint32_t scratchBytesPerBuffer = 0;
    uint32_t layoutVersion = 0;
    uint32_t marker = 0;
    uint64_t directLargeRequiredUbBytes = 0;
    uint64_t directLargeUbCOffset0 = 0;
    uint64_t directLargeUbDOffset0 = 0;
    uint64_t directLargeUbFp32Offset0 = 0;
    uint64_t directLargeUbCOffset1 = 0;
    uint64_t directLargeUbDOffset1 = 0;
    uint64_t directLargeUbFp32Offset1 = 0;
    uint32_t tokenVolume = 0;
    uint32_t isLargePath = 0;
    uint32_t combineImplMode = 0;
    uint32_t combineStopStep = 0;
    uint32_t directLargeEnabled = 0;
    uint32_t directLargeTileCols = 0;
    uint32_t directLargeUbStages = 0;
    uint32_t directLargeRequiredUbBytesOk = 0;
    uint32_t usesLaneSplit = 0;
    uint32_t usesColumnFallback = 0;
    uint32_t usesOldBusinessHelper = 0;
    uint64_t directSmallRequiredUbBytes = 0;
    uint64_t directSmallUbCOffset0 = 0;
    uint64_t directSmallUbDOffset0 = 0;
    uint64_t directSmallUbFp32Offset0 = 0;
    uint64_t directSmallUbScaleOffset0 = 0;
    uint64_t directSmallUbCOffset1 = 0;
    uint64_t directSmallUbDOffset1 = 0;
    uint64_t directSmallUbFp32Offset1 = 0;
    uint64_t directSmallUbScaleOffset1 = 0;
    uint32_t directSmallEnabled = 0;
    uint32_t directSmallTileRows = 0;
    uint32_t directSmallTileCols = 0;
    uint32_t directSmallN0 = 0;
    uint32_t directSmallUbStages = 0;
    uint32_t directSmallMaxElems = 0;
    uint32_t directSmallScaleElems = 0;
    uint32_t directSmallRequiredUbBytesOk = 0;
    uint32_t usesFrontCaseBranch = 0;
    uint32_t offsetScale2CapacityOk = 0;
    uint32_t offsetScale2DNoOverlap = 0;
    uint32_t offsetDIsRawHalf = 0;
    uint32_t reserved2 = 0;
    uint64_t peerOffsetScale2 = 0;
    uint64_t offsetScale2Bytes = 0;
    uint64_t offsetScale2CapacityBytes = 0;
    uint32_t gmm2CombineCvMode = 0;
    uint32_t gmm2CombineCvDebugMode = 0;
    uint32_t gmm2CombineCvReadyFlag = 0;
    uint32_t gmm2CombineCvFreeFlag = 0;
    uint64_t gmm2CombineCvSlotOffset = 0;
    uint64_t gmm2CombineCvTileScaleOffset = 0;
    uint64_t gmm2CombineCvRequiredUbBytes = 0;
};

struct DispatchFFNCombineCombineGmm2ReadyDebug {
    uint64_t magic = 0;
    uint64_t waitStartSyscnt = 0;
    uint64_t waitEndSyscnt = 0;
    uint64_t syncAllStartSyscnt = 0;
    uint64_t syncAllEndSyscnt = 0;
    uint32_t groupIdx = 0;
    uint32_t coreIdx = 0;
    uint32_t flagId = 0;
    uint32_t aivSyncAfterWait = 0;
    uint32_t initialEventMask = 0;
    uint32_t marker = 0;
};

struct DispatchFFNCombineCombineMetadataDebug {
    uint64_t magic = 0;
    uint32_t groupIdx = 0;
    uint32_t srcRank = 0;
    uint32_t coreIdx = 0;
    uint32_t coreNum = 0;
    uint32_t groupBase = 0;
    uint32_t groupBaseAfter = 0;
    uint32_t rowsRaw = 0;
    uint32_t rows = 0;
    uint32_t srcRowOffset = 0;
    uint32_t dstRowOffset = 0;
    uint32_t cumsumBeforeSrc = 0;
    uint32_t clipped = 0;
    uint32_t skipReason = 0;
    uint32_t marker = 0;
};

struct DispatchFFNCombineCombineLoadDebug {
    uint64_t magic = 0;
    uint32_t groupIdx = 0;
    uint32_t srcRank = 0;
    uint32_t coreIdx = 0;
    uint32_t coreNum = 0;
    uint32_t srcRow = 0;
    uint32_t dstRow = 0;
    uint32_t rows = 0;
    uint32_t cLoadBytes = 0;
    uint32_t scaleOffsetBytes = 0;
    uint32_t scaleBits = 0;
    uint32_t cFirstBits = 0;
    uint32_t cLastBits = 0;
    uint32_t scaleReadMode = 0;
    uint32_t marker = 0;
};

struct DispatchFFNCombineCombineDequantDebug {
    uint64_t magic = 0;
    uint32_t groupIdx = 0;
    uint32_t srcRank = 0;
    uint32_t coreIdx = 0;
    uint32_t coreNum = 0;
    uint32_t srcRow = 0;
    uint32_t rows = 0;
    uint32_t scaleBits = 0;
    uint32_t fp32BeforeFirstBits = 0;
    uint32_t fp32AfterFirstBits = 0;
    uint32_t fp32AfterLastBits = 0;
    uint32_t dFirstBits = 0;
    uint32_t dLastBits = 0;
    uint32_t opCounts = 0;
    uint32_t marker = 0;
};

struct DispatchFFNCombineCombineStoreDebug {
    uint64_t magic = 0;
    uint64_t dstGmOffsetBytes = 0;
    uint32_t groupIdx = 0;
    uint32_t srcRank = 0;
    uint32_t coreIdx = 0;
    uint32_t srcRow = 0;
    uint32_t dstRow = 0;
    uint32_t rows = 0;
    uint32_t remoteRank = 0;
    uint32_t remoteBaseValid = 0;
    uint32_t storeBytes = 0;
    uint32_t storedFirstBits = 0;
    uint32_t storedLastBits = 0;
    uint32_t localOrRemote = 0;
};

struct DispatchFFNCombineCombineFinalizeDebug {
    uint64_t magic = 0;
    uint64_t finalizeWaitStartSyscnt = 0;
    uint64_t finalizeWaitEndSyscnt = 0;
    uint64_t syncAllStartSyscnt = 0;
    uint64_t syncAllEndSyscnt = 0;
    uint64_t resetStartSyscnt = 0;
    uint64_t resetEndSyscnt = 0;
    uint64_t crossRankSyncStartSyscnt = 0;
    uint64_t crossRankSyncEndSyscnt = 0;
    uint64_t tokenPerExpertBaseOffsetBytes = 0;
    uint32_t coreIdx = 0;
    uint32_t coreNum = 0;
    uint32_t resetElems = 0;
    uint32_t resetWriter = 0;
    uint32_t tokenPerExpertResetFirstSample = 0;
    uint32_t tokenPerExpertResetLastSample = 0;
    uint32_t boundaryDoneMarker = 0;
    uint32_t crossRankSyncAfterReset = 0;
    uint32_t marker = 0;
    uint32_t reserved0 = 0;
    uint32_t reserved1 = 0;
    uint32_t reserved2 = 0;
};

struct DispatchFFNCombineCombineTaskStatsDebug {
    uint64_t magic = 0;
    uint32_t groupIdx = 0;
    uint32_t coreIdx = 0;
    uint32_t rankSize = 0;
    uint32_t assignedRankCount = 0;
    uint32_t coreRowCount = 0;
    uint32_t groupBase = 0;
    uint32_t currentM = 0;
    uint32_t firstSrcRank = 0;
    uint32_t firstRows = 0;
    uint32_t firstSrcRowOffset = 0;
    uint32_t firstDstRowOffset = 0;
    uint32_t lastSrcRank = 0;
    uint32_t lastRows = 0;
    uint32_t lastSrcRowOffset = 0;
    uint32_t lastDstRowOffset = 0;
    uint32_t marker = 0;
    uint32_t currentMRaw = 0;
    uint32_t coreLoops = 0;
    uint32_t startCoreIdx = 0;
    uint32_t startLoopIdx = 0;
    uint32_t aicCoreIdx = 0;
    uint32_t aicCoreNum = 0;
    uint32_t aivSubCoreIdx = 0;
    uint32_t tileM = 0;
    uint32_t tileN = 0;
    uint32_t firstActualM = 0;
    uint32_t firstActualN = 0;
    uint32_t lastActualM = 0;
    uint32_t lastActualN = 0;
    uint32_t reserved0 = 0;
};

struct DispatchFFNCombineCombineDoneDebug {
    uint64_t magic = 0;
    uint32_t coreIdx = 0;
    uint32_t rank = 0;
    uint32_t rankSize = 0;
    uint32_t stageNum = 0;
    uint32_t expertPerRank = 0;
    uint32_t syncBeforeDone = 0;
    uint32_t crossRankSync = 0;
    uint32_t marker = 0;
    uint32_t localSegmentCount = 0;
    uint32_t remoteSegmentCount = 0;
    uint32_t localRows = 0;
    uint32_t remoteRows = 0;
    uint32_t localBytes = 0;
    uint32_t remoteBytes = 0;
};

struct DispatchFFNCombineUnpermuteTiling {
    uint64_t unpermuteDebugOffset = 0;
    uint64_t unpermuteDebugBytes = 0;
    uint32_t unpermuteTileCols = 1024;
    uint32_t unpermuteTokenBatch = 256;
    uint32_t unpermuteDebugMode = 0;
    uint32_t unpermuteLayoutVersion = 1;
    uint32_t reserved0[8] = {0};
};

struct DispatchFFNCombineUnpermuteLayoutDebug {
    uint64_t magic = 0;
    uint64_t workspaceBase = 0;
    uint64_t offsetDBase = 0;
    uint64_t expandedRowIdxBase = 0;
    uint64_t probsBase = 0;
    uint64_t outBase = 0;
    uint64_t unpermuteDebugOffset = 0;
    uint64_t unpermuteDebugBytes = 0;
    uint64_t peerOffsetD = 0;
    uint64_t offsetDBytes = 0;
    uint64_t outBytes = 0;
    uint64_t expandedRowIdxBytes = 0;
    uint64_t probsBytes = 0;
    uint64_t frontWorkspaceBytes = 0;
    uint32_t rank = 0;
    uint32_t rankSize = 0;
    uint32_t coreIdx = 0;
    uint32_t coreNum = 0;
    uint32_t stageNum = 0;
    uint32_t problemM = 0;
    uint32_t problemK = 0;
    uint32_t topK = 0;
    uint32_t maxOutputSize = 0;
    uint32_t expandedRowsValid = 0;
    uint32_t tileCols = 0;
    uint32_t tokenBatch = 0;
    uint32_t debugMode = 0;
    uint32_t layoutVersion = 0;
    uint32_t outputElementBytes = 0;
    uint32_t offsetDRowBytes = 0;
    uint32_t outRowBytes = 0;
    uint32_t tileBytes = 0;
    uint32_t metadataBytesPerBatch = 0;
    uint32_t ubMainBytes = 0;
    uint32_t metadataBufferNum = 0;
    uint32_t tokenBufferNum = 0;
    uint32_t syncBoundaryInCombine = 0;
    uint32_t crossRankSyncInUnpermute = 0;
    uint32_t taskSplitMode = 0;
    uint32_t kTileMode = 0;
    uint32_t probsDtypeBytes = 0;
    uint32_t indexDtypeBytes = 0;
    uint32_t marker = 0;
    uint32_t offsetScale2CapacityOk = 0;
    uint32_t offsetScale2DNoOverlap = 0;
    uint32_t offsetDIsRawHalf = 0;
    uint64_t offsetScale2Base = 0;
    uint64_t peerOffsetScale2 = 0;
    uint64_t offsetScale2Bytes = 0;
    uint64_t offsetScale2CapacityBytes = 0;
    uint64_t reserved1[30] = {0};
};

struct DispatchFFNCombineUnpermuteTaskDebug {
    uint64_t magic = 0;
    uint32_t coreIdx = 0;
    uint32_t rank = 0;
    uint32_t rankSize = 0;
    uint32_t stageNum = 0;
    uint32_t tokenStart = 0;
    uint32_t tokenCount = 0;
    uint32_t expandedStart = 0;
    uint32_t expandedCount = 0;
    uint32_t splitBase = 0;
    uint32_t splitRem = 0;
    uint32_t problemM = 0;
    uint32_t topK = 0;
    uint32_t marker = 0;
    uint32_t reserved0 = 0;
};

struct DispatchFFNCombineUnpermuteMetaDebug {
    uint64_t magic = 0;
    uint32_t coreIdx = 0;
    uint32_t token = 0;
    uint32_t topkIdx = 0;
    uint32_t expandedRow = 0;
    uint32_t valid = 0;
    uint32_t probBits = 0;
    uint32_t batchStart = 0;
    uint32_t localToken = 0;
    uint32_t expandedRowsValid = 0;
    uint32_t marker = 0;
    uint32_t reserved0[4] = {0};
};

struct DispatchFFNCombineUnpermuteAccumDebug {
    uint64_t magic = 0;
    uint32_t coreIdx = 0;
    uint32_t token = 0;
    uint32_t col = 0;
    uint32_t cols = 0;
    uint32_t validTopk = 0;
    uint32_t topkProcessed = 0;
    uint32_t firstExpandedRow = 0;
    uint32_t firstProbBits = 0;
    float accSample = 0.0f;
    uint32_t marker = 0;
    uint32_t reserved0[4] = {0};
};

struct DispatchFFNCombineUnpermuteOutputDebug {
    uint64_t magic = 0;
    uint32_t coreIdx = 0;
    uint32_t token = 0;
    uint32_t col = 0;
    uint32_t cols = 0;
    uint32_t outputValue = 0;
    uint32_t validTopk = 0;
    uint32_t topkProcessed = 0;
    uint32_t marker = 0;
    uint32_t reserved0[6] = {0};
};

struct DispatchFFNCombineUnpermuteDoneDebug {
    uint64_t magic = 0;
    uint32_t coreIdx = 0;
    uint32_t rank = 0;
    uint32_t rankSize = 0;
    uint32_t stageNum = 0;
    uint32_t tokenStart = 0;
    uint32_t tokenCount = 0;
    uint32_t syncBeforeDone = 0;
    uint32_t marker = 0;
    uint32_t rows = 0;
    uint32_t values = 0;
    uint32_t bytes = 0;
    uint32_t reserved0[3] = {0};
};

static_assert(sizeof(DispatchFFNCombineGmm1LayoutDebug) == 512);
static_assert(sizeof(DispatchFFNCombineGmm1SyncDebug) == 64);
static_assert(sizeof(DispatchFFNCombineGmm1TaskDebug) == 128);
static_assert(sizeof(DispatchFFNCombineGmm1DoneDebug) == 64);
static_assert(sizeof(DispatchFFNCombineSwigluTiling) == 80);
static_assert(sizeof(DispatchFFNCombineSwigluSegmentRuntimeMeta) == 64);
static_assert(sizeof(DispatchFFNCombineSwigluLayoutDebug) == 512);
static_assert(sizeof(DispatchFFNCombineSwigluC2VDebug) == 64);
static_assert(sizeof(DispatchFFNCombineSwigluSegmentDebug) == 64);
static_assert(sizeof(DispatchFFNCombineSwigluTaskDebug) == 64);
static_assert(sizeof(DispatchFFNCombineSwigluDoneDebug) == 64);
static_assert(sizeof(DispatchFFNCombineSwigluFinalSyncDebug) == 64);
static_assert(sizeof(DispatchFFNCombineGmm2Tiling) == 64);
static_assert(sizeof(DispatchFFNCombineGmm2LayoutDebug) == 512);
static_assert(sizeof(DispatchFFNCombineGmm2V2CDebug) == 64);
static_assert(sizeof(DispatchFFNCombineGmm2SegmentDebug) == 64);
static_assert(sizeof(DispatchFFNCombineGmm2TaskDebug) == 128);
static_assert(sizeof(DispatchFFNCombineGmm2DoneDebug) == 64);
static_assert(sizeof(DispatchFFNCombineGmm2FinalSyncDebug) == 64);
static_assert(sizeof(DispatchFFNCombineCombineTiling) == 80);
static_assert(sizeof(DispatchFFNCombineCombineLayoutDebug) == 512);
static_assert(sizeof(DispatchFFNCombineCombineGmm2ReadyDebug) == 64);
static_assert(sizeof(DispatchFFNCombineCombineMetadataDebug) == 64);
static_assert(sizeof(DispatchFFNCombineCombineLoadDebug) == 64);
static_assert(sizeof(DispatchFFNCombineCombineDequantDebug) == 64);
static_assert(sizeof(DispatchFFNCombineCombineStoreDebug) == 64);
static_assert(sizeof(DispatchFFNCombineCombineFinalizeDebug) == 128);
static_assert(sizeof(DispatchFFNCombineCombineTaskStatsDebug) == 128);
static_assert(sizeof(DispatchFFNCombineCombineDoneDebug) == 64);
static_assert(sizeof(DispatchFFNCombineUnpermuteTiling) == 64);
static_assert(sizeof(DispatchFFNCombineUnpermuteLayoutDebug) == 512);
static_assert(sizeof(DispatchFFNCombineUnpermuteTaskDebug) == 64);
static_assert(sizeof(DispatchFFNCombineUnpermuteMetaDebug) == 64);
static_assert(sizeof(DispatchFFNCombineUnpermuteAccumDebug) == 64);
static_assert(sizeof(DispatchFFNCombineUnpermuteOutputDebug) == 64);
static_assert(sizeof(DispatchFFNCombineUnpermuteDoneDebug) == 64);

struct DispatchFFNCombineTilingData {
    DispatchFFNCombineInfo dispatchFFNCombineInfo;
    DispatchFFNCombineRuntimeInfo runtimeInfo;
    DispatchFFNCombineFrontReorderTiling frontReorderTiling;
    DispatchFFNCombineDispatchTiling dispatchTiling;
    DispatchFFNCombineGmm1Tiling gmm1Tiling;
    DispatchFFNCombineSwigluTiling swigluTiling;
    DispatchFFNCombineGmm2Tiling gmm2Tiling;
    DispatchFFNCombineCombineTiling combineTiling;
    DispatchFFNCombineUnpermuteTiling unpermuteTiling;
    DispatchFFNCombineFrontDebugTiling frontDebugTiling;
};
