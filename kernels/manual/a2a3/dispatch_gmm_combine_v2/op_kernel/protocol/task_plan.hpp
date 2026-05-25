#pragma once

#include <cstdint>


static constexpr uint32_t kCombineTileAlignBytes = 32;
static constexpr uint32_t kDispatchScaleSlotBytes = 128;
static constexpr uint32_t kCombineTaskWaitGmm2Ready = 1U << 0;
static constexpr uint32_t kCombineTaskPublishOwnerCompletion = 1U << 1;

struct DispatchPullTask {
    uint32_t srcRank = 0;
    uint32_t dstRank = 0;
    uint32_t expertGroupId = 0;
    uint32_t expertId = 0;
    uint32_t localExpertSlot = 0;
    uint32_t srcRowBegin = 0;
    uint32_t dstRowBegin = 0;
    uint32_t rowCount = 0;
    uint32_t hiddenBytes = 0;
    uint32_t scaleBytes = 0;
    uint64_t srcPayloadOffsetBytes = 0;
    uint64_t dstPayloadOffsetBytes = 0;
    uint64_t srcScaleOffsetBytes = 0;
    uint64_t dstScaleOffsetBytes = 0;
    uint32_t readyEpoch = 0;
    uint32_t taskFlags = 0;
};

struct CombinePushTask {
    uint32_t srcRank = 0;
    uint32_t dstRank = 0;
    uint32_t expertGroupId = 0;
    uint32_t expertId = 0;
    uint32_t ownerRow = 0;
    uint32_t tileId = 0;
    uint32_t srcRowBegin = 0;
    uint32_t dstRowBegin = 0;
    uint32_t rowCount = 0;
    uint32_t outputBytes = 0;
    uint64_t srcPayloadOffsetBytes = 0;
    uint64_t dstPayloadOffsetBytes = 0;
    uint32_t completionEpoch = 0;
    uint32_t gmm2TileReadyIndex = 0;
    uint32_t gmm2ReadyRows = 0;
    uint32_t ownerCompletionIndex = 0;
    uint32_t taskFlags = 0;
};

struct CombineRowRange {
    uint32_t begin = 0;
    uint32_t end = 0;
};

struct ExpertSourceSegmentMeta {
    uint32_t localExpert = 0;
    uint32_t srcRank = 0;
    uint32_t rowBegin = 0;
    uint32_t rowEnd = 0;
    uint32_t dispatchRangeStatusIndex = 0;
    uint32_t dispatchRangeCompletionIndex = 0;
    uint32_t doneSignalIndex = 0;
};

struct ComputeTileMeta {
    uint32_t rowBegin = 0;
    uint32_t rowEnd = 0;
    uint32_t sourceSegmentBegin = 0;
    uint32_t sourceSegmentEnd = 0;
    uint32_t computeTileStatusIndex = 0;
    uint32_t gmm2TileReadyIndex = 0;
    uint32_t localExpert = 0;
    uint32_t expertRangeIndex = 0;
    uint32_t kBegin = 0;
    uint32_t kEnd = 0;
    uint32_t nBegin = 0;
    uint32_t nEnd = 0;
    uint32_t expandedRowBegin = 0;
    uint32_t scaleOffsetBegin = 0;
};

struct ExpertRangeMeta {
    uint32_t localExpert = 0;
    uint32_t rowBegin = 0;
    uint32_t rowEnd = 0;
    uint32_t sourceSegmentBegin = 0;
    uint32_t sourceSegmentEnd = 0;
    uint32_t computeTileBegin = 0;
    uint32_t computeTileEnd = 0;
    uint32_t expertSafeRowsIndex = 0;
};

struct ScoreboardMeta {
    uint32_t dispatchRangeStatusBase = 0;
    uint32_t dispatchRangeStatusCount = 0;
    uint32_t expertSafeRowsBase = 0;
    uint32_t expertSafeRowsCount = 0;
    uint32_t dispatchRangeCompletionBase = 0;
    uint32_t dispatchRangeCompletionCount = 0;
    uint32_t computeTileStatusBase = 0;
    uint32_t computeTileStatusCount = 0;
    uint32_t gmm2TileReadyBase = 0;
    uint32_t gmm2TileReadyCount = 0;
    uint32_t ownerCompletionBase = 0;
    uint32_t ownerCompletionCount = 0;
    uint32_t totalWords = 0;
};

inline uint32_t DispatchRangeStatusIndex(const ScoreboardMeta& meta,
                                         uint32_t localExpert,
                                         uint32_t srcRank,
                                         uint32_t worldSize)
{
    return meta.dispatchRangeStatusBase + localExpert * worldSize + srcRank;
}

inline uint32_t ExpertSafeRowsIndex(const ScoreboardMeta& meta, uint32_t localExpert)
{
    return meta.expertSafeRowsBase + localExpert;
}

inline uint32_t DispatchRangeCompletionIndex(const ScoreboardMeta& meta, uint32_t rangeOrdinal)
{
    constexpr uint32_t doneSlotElems = kCombineTileAlignBytes / sizeof(int32_t);
    return meta.dispatchRangeCompletionBase + rangeOrdinal * doneSlotElems;
}

inline uint32_t ComputeTileStatusIndex(const ScoreboardMeta& meta, uint32_t tileIndex)
{
    return meta.computeTileStatusBase + tileIndex;
}

inline uint32_t Gmm2TileReadyIndex(const ScoreboardMeta& meta, uint32_t tileIndex)
{
    return meta.gmm2TileReadyBase + tileIndex;
}

inline uint32_t OwnerCompletionIndex(const ScoreboardMeta& meta, uint32_t ownerOrdinal)
{
    return meta.ownerCompletionBase + ownerOrdinal;
}

inline uint32_t OwnerCompletionSummaryIndex(uint32_t ownerRank, uint32_t ownerOrdinal)
{
    constexpr uint32_t doneSlotElems = kCombineTileAlignBytes / sizeof(int32_t);
    constexpr uint32_t ownerSummaryBase = 20480U + 32U * 64U;
    return ownerSummaryBase + ownerRank * 64U * doneSlotElems + ownerOrdinal * doneSlotElems;
}

struct DispatchRangeTask {
    uint32_t srcRank = 0;
    uint32_t dstRank = 0;
    uint32_t localExpert = 0;
    uint32_t rowCount = 0;
    uint32_t hiddenBytes = 0;
    uint32_t scaleBytes = 0;
    uint64_t srcPayloadOffsetBytes = 0;
    uint64_t dstExpertMajorOffsetBytes = 0;
    uint64_t srcScaleOffsetBytes = 0;
    uint64_t dstScaleOffsetBytes = 0;
    uint32_t expandedRowBegin = 0;
    uint32_t expandedRowEnd = 0;
    uint32_t kBegin = 0;
    uint32_t kEnd = 0;
    uint32_t scaleOffsetBegin = 0;
    uint32_t readySignalIndex = 0;
    uint32_t doneSignalIndex = 0;
    uint32_t rangeStatusIndex = 0;
    uint32_t readyRows = 0;
    uint32_t rangeStatusBase = 0;
    uint32_t expertSafeRowsIndex = 0;
    uint32_t dispatchRangeCompletionIndex = 0;
    uint32_t dispatchRangeCompletionBase = 0;
    uint32_t expectedRangeCount = 0;
    uint32_t localExpertRowBegin = 0;
    uint32_t localExpertRowEnd = 0;
    uint32_t worldSize = 0;
    uint32_t sourceExpectedMask = 0;
};

struct CombineRangeTask {
    uint32_t srcRank = 0;
    uint32_t ownerRank = 0;
    uint32_t localExpert = 0;
    uint32_t rowCount = 0;
    uint32_t outputBytes = 0;
    uint64_t srcGmm2OffsetBytes = 0;
    uint64_t dstCombineOffsetBytes = 0;
    uint32_t probOffset = 0;
    uint32_t readySignalIndex = 0;
    uint32_t gmm2TileReadyIndex = 0;
    uint32_t gmm2ReadyRows = 0;
    uint32_t ownerCompletionIndex = 0;
    uint32_t taskFlags = 0;
};

struct ExpertGroupTaskRange {
    uint32_t expertGroupId = 0;
    uint32_t dispatchTaskBegin = 0;
    uint32_t dispatchTaskEnd = 0;
    uint32_t combineTaskBegin = 0;
    uint32_t combineTaskEnd = 0;
    uint32_t dispatchRangeBegin = 0;
    uint32_t dispatchRangeEnd = 0;
    uint32_t combineRangeBegin = 0;
    uint32_t combineRangeEnd = 0;
    uint32_t computeTileBegin = 0;
    uint32_t computeTileEnd = 0;
};

struct RuntimeRoutingMeta {
    uint32_t tokenPerExpertOffset = 0;
    uint32_t tokenPerExpertCount = 0;
    uint32_t cumsumMMOffset = 0;
    uint32_t cumsumMMCount = 0;
    uint32_t cumsumSendOffset = 0;
    uint32_t cumsumSendCount = 0;
    uint32_t expertTokensBeforeCapacityOffset = 0;
    uint32_t expertTokensBeforeCapacityCount = 0;
    uint32_t droppedByCapacityCount = 0;
    uint32_t sentinelDropCount = 0;
    uint32_t groupCount = 0;
};

