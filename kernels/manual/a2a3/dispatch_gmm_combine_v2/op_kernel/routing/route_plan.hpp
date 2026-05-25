#pragma once

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

#include "../protocol/signal_protocol.hpp"
#include "../protocol/task_plan.hpp"


struct RoutingFixture {
    uint32_t worldSize = 0;
    uint32_t expertsPerRank = 0;
    uint32_t topk = 0;
    uint32_t maxOutputSize = 0;
    uint32_t hiddenBytes = 256;
    uint32_t outputBytes = sizeof(float);
    uint32_t gmm1N = 0;
    std::vector<uint32_t> rowsPerRank;
    std::vector<std::vector<uint8_t>> xActiveMaskPerRank;
    std::vector<std::vector<uint32_t>> topkIdsPerRank;
    std::vector<std::vector<float>> topkProbsPerRank;
};



template <typename T>
inline std::vector<T> ExclusiveScan(const std::vector<T>& input)
{
    std::vector<T> output(input.size(), 0);
    T running = 0;
    for (std::size_t i = 0; i < input.size(); ++i) {
        output[i] = running;
        running += input[i];
    }
    return output;
}

struct RawExpandedEntry {
    uint32_t ownerRank = 0;
    uint32_t srcRank = 0;
    uint32_t rowIdx = 0;
    uint32_t topkOrdinal = 0;
    uint32_t expertId = 0;
    uint32_t dstRank = 0;
    uint32_t localExpertSlot = 0;
    float prob = 0.0f;
    bool isSentinel = false;
};

struct DispatchTruthView {
    std::vector<uint32_t> expandedRowIdx;
    std::vector<uint32_t> expandedExpertIdx;
    std::vector<float> expandedProb;
    std::vector<uint32_t> expandedSrcRank;
    std::vector<uint32_t> expandedLocalExpertSlot;
    std::vector<uint32_t> tokenPerExpert;
    std::vector<uint32_t> gatheredExpertCount;
    std::vector<uint32_t> localExpertPrefix;
    std::vector<uint32_t> cumsumMM;
    std::vector<uint32_t> groupRowCount;
    uint32_t packedRowCount = 0;
    std::vector<uint32_t> srcOffset;
    std::vector<uint32_t> dstOffset;
};

struct CombineTruthView {
    std::vector<uint32_t> expandedRowIdx;
    std::vector<uint32_t> expandedTopkOrdinal;
    std::vector<uint32_t> expandedSrcRank;
    std::vector<uint32_t> expandedExpertIdx;
    std::vector<uint32_t> expandedDstRank;
    std::vector<uint32_t> expandedComputeRow;
    std::vector<float> expandedProb;
    std::vector<std::pair<uint32_t, uint32_t>> rowToExpandedRange;
    std::vector<uint32_t> ownerRowExpandedOrdinal;
    std::vector<uint32_t> combineDstOffset;
};

struct RoutingMetadataView {
    DispatchTruthView dispatch;
    CombineTruthView combine;
};

struct RoutingSemanticMeta {
    uint32_t beforeCapacityCount = 0;
    uint32_t executableCount = 0;
    uint32_t droppedByCapacityCount = 0;
    uint32_t sentinelDropCount = 0;
    uint32_t maxOutputSize = 0;
    uint32_t dropPadEnabled = 1;
};

struct SemanticLedgerEntry {
    uint32_t beforeCapacityOrdinal = 0;
    uint32_t executableOrdinal = 0xFFFFFFFFU;
    uint32_t ownerRank = 0;
    uint32_t srcRank = 0;
    uint32_t rowIdx = 0;
    uint32_t topkOrdinal = 0;
    uint32_t expertId = 0;
    uint32_t dstRank = 0;
    uint32_t localExpertSlot = 0;
    float prob = 0.0f;
    uint32_t accepted = 0;
    uint32_t droppedByCapacity = 0;
    uint32_t sentinel = 0;
};

struct RoutingPlanBundle {
    RoutingMetadataView view;
    std::vector<RawExpandedEntry> beforeCapacity;
    std::vector<RawExpandedEntry> executable;
    std::vector<SemanticLedgerEntry> semanticLedger;
    std::vector<uint32_t> runtimeRoutingWords;
    RuntimeRoutingMeta runtimeRoutingMeta;
    std::vector<DispatchPullTask> dispatchTasks;
    std::vector<CombinePushTask> combineTasks;
    std::vector<DispatchRangeTask> dispatchRanges;
    std::vector<CombineRangeTask> combineRanges;
    std::vector<ExpertRangeMeta> expertRanges;
    std::vector<ExpertGroupTaskRange> expertGroupRanges;
    std::vector<ExpertSourceSegmentMeta> expertSourceSegments;
    std::vector<ComputeTileMeta> computeTiles;
    RoutingSemanticMeta semanticMeta;
    ScoreboardMeta scoreboardMeta;
};

inline uint32_t SentinelExpertId(const RoutingFixture& fixture)
{
    return fixture.worldSize * fixture.expertsPerRank;
}

inline std::vector<RawExpandedEntry> ExpandBeforeCapacity(const RoutingFixture& fixture)
{
    std::vector<RawExpandedEntry> out;
    const uint32_t sentinelExpert = SentinelExpertId(fixture);
    for (uint32_t srcRank = 0; srcRank < fixture.worldSize; ++srcRank) {
        const uint32_t rows = fixture.rowsPerRank.at(srcRank);
        for (uint32_t row = 0; row < rows; ++row) {
            for (uint32_t k = 0; k < fixture.topk; ++k) {
                const uint32_t idx = row * fixture.topk + k;
                const bool active = fixture.xActiveMaskPerRank.at(srcRank).at(row) != 0;
                const uint32_t expertId = active ? fixture.topkIdsPerRank.at(srcRank).at(idx) : sentinelExpert;
                out.push_back({
                    .ownerRank = srcRank,
                    .srcRank = srcRank,
                    .rowIdx = row,
                    .topkOrdinal = k,
                    .expertId = expertId,
                    .dstRank = active ? (expertId / fixture.expertsPerRank) : fixture.worldSize,
                    .localExpertSlot = active ? (expertId % fixture.expertsPerRank) : fixture.expertsPerRank,
                    .prob = fixture.topkProbsPerRank.at(srcRank).at(idx),
                    .isSentinel = !active,
                });
            }
        }
    }
    return out;
}

inline std::vector<RawExpandedEntry> KeepExecutableEntries(const std::vector<RawExpandedEntry>& raw,
                                                           const RoutingFixture& fixture)
{
    std::vector<uint32_t> keptPerDst(fixture.worldSize, 0);
    std::vector<RawExpandedEntry> out;
    out.reserve(raw.size());
    for (const auto& entry : raw) {
        if (entry.isSentinel || entry.dstRank >= fixture.worldSize) {
            continue;
        }
        if (keptPerDst[entry.dstRank] >= fixture.maxOutputSize) {
            continue;
        }
        ++keptPerDst[entry.dstRank];
        out.push_back(entry);
    }
    return out;
}

inline RoutingSemanticMeta BuildRoutingSemanticMeta(const std::vector<RawExpandedEntry>& beforeCapacity,
                                                    const std::vector<RawExpandedEntry>& executable,
                                                    const RoutingFixture& fixture)
{
    RoutingSemanticMeta meta{};
    meta.beforeCapacityCount = static_cast<uint32_t>(beforeCapacity.size());
    meta.executableCount = static_cast<uint32_t>(executable.size());
    meta.maxOutputSize = fixture.maxOutputSize;
    meta.dropPadEnabled = 1;
    for (const auto& entry : beforeCapacity) {
        if (entry.isSentinel || entry.dstRank >= fixture.worldSize) {
            ++meta.sentinelDropCount;
        }
    }
    const uint32_t nonSentinelCount = meta.beforeCapacityCount - meta.sentinelDropCount;
    meta.droppedByCapacityCount = nonSentinelCount > meta.executableCount ? nonSentinelCount - meta.executableCount : 0;
    return meta;
}

inline std::vector<SemanticLedgerEntry> BuildSemanticLedger(const std::vector<RawExpandedEntry>& beforeCapacity,
                                                            const RoutingFixture& fixture)
{
    std::vector<SemanticLedgerEntry> ledger;
    ledger.reserve(beforeCapacity.size());
    std::vector<uint32_t> keptPerDst(fixture.worldSize, 0);
    uint32_t executableOrdinal = 0;
    for (uint32_t ordinal = 0; ordinal < beforeCapacity.size(); ++ordinal) {
        const auto& entry = beforeCapacity[ordinal];
        const bool sentinel = entry.isSentinel || entry.dstRank >= fixture.worldSize;
        const bool accepted = !sentinel && keptPerDst[entry.dstRank] < fixture.maxOutputSize;
        if (accepted) {
            ++keptPerDst[entry.dstRank];
        }
        ledger.push_back({
            .beforeCapacityOrdinal = ordinal,
            .executableOrdinal = accepted ? executableOrdinal++ : 0xFFFFFFFFU,
            .ownerRank = entry.ownerRank,
            .srcRank = entry.srcRank,
            .rowIdx = entry.rowIdx,
            .topkOrdinal = entry.topkOrdinal,
            .expertId = entry.expertId,
            .dstRank = entry.dstRank,
            .localExpertSlot = entry.localExpertSlot,
            .prob = entry.prob,
            .accepted = accepted ? 1U : 0U,
            .droppedByCapacity = (!sentinel && !accepted) ? 1U : 0U,
            .sentinel = sentinel ? 1U : 0U,
        });
    }
    return ledger;
}

inline void BuildDispatchTruthForLocalRank(const std::vector<RawExpandedEntry>& executable,
                                           const RoutingFixture& fixture,
                                           uint32_t localRank,
                                           DispatchTruthView& view)
{
    view = {};
    view.tokenPerExpert.assign(fixture.worldSize * fixture.expertsPerRank, 0);
    view.gatheredExpertCount.assign(fixture.expertsPerRank, 0);
    view.localExpertPrefix.assign(fixture.expertsPerRank, 0);
    view.cumsumMM.assign(fixture.expertsPerRank * fixture.worldSize, 0);
    view.groupRowCount.assign(fixture.expertsPerRank, 0);

    std::vector<RawExpandedEntry> localEntries;
    localEntries.reserve(executable.size());
    for (const auto& entry : executable) {
        if (entry.dstRank != localRank) {
            continue;
        }
        localEntries.push_back(entry);
        view.expandedRowIdx.push_back(entry.rowIdx);
        view.expandedExpertIdx.push_back(entry.expertId);
        view.expandedProb.push_back(entry.prob);
        view.expandedSrcRank.push_back(entry.srcRank);
        view.expandedLocalExpertSlot.push_back(entry.localExpertSlot);
        ++view.tokenPerExpert[entry.srcRank * fixture.expertsPerRank + entry.localExpertSlot];
        ++view.groupRowCount[entry.localExpertSlot];
    }

    for (uint32_t localExpert = 0; localExpert < fixture.expertsPerRank; ++localExpert) {
        for (uint32_t srcRank = 0; srcRank < fixture.worldSize; ++srcRank) {
            const uint32_t count = view.tokenPerExpert[srcRank * fixture.expertsPerRank + localExpert];
            view.gatheredExpertCount[localExpert] += count;
        }
    }
    view.localExpertPrefix = ExclusiveScan(view.gatheredExpertCount);
    for (uint32_t localExpert = 0; localExpert < fixture.expertsPerRank; ++localExpert) {
        uint32_t prefix = 0;
        for (uint32_t srcRank = 0; srcRank < fixture.worldSize; ++srcRank) {
            const size_t index = static_cast<size_t>(localExpert) * fixture.worldSize + srcRank;
            view.cumsumMM[index] = prefix;
            prefix += view.tokenPerExpert[srcRank * fixture.expertsPerRank + localExpert];
        }
    }

    std::vector<uint32_t> srcOrdinalPerRank(fixture.worldSize, 0);
    std::vector<uint32_t> dstOrdinalPerSrcLocalExpert(fixture.worldSize * fixture.expertsPerRank, 0);
    for (const auto& entry : localEntries) {
        view.srcOffset.push_back(srcOrdinalPerRank[entry.srcRank]++);
        const uint32_t localKey = entry.srcRank * fixture.expertsPerRank + entry.localExpertSlot;
        const uint32_t localOrdinal = dstOrdinalPerSrcLocalExpert[localKey]++;
        view.dstOffset.push_back(
            view.localExpertPrefix[entry.localExpertSlot] +
            view.cumsumMM[entry.localExpertSlot * fixture.worldSize + entry.srcRank] +
            localOrdinal);
    }
    view.packedRowCount = static_cast<uint32_t>(localEntries.size());
}

inline std::vector<uint32_t> BuildComputeRowOrdinalForExecutable(const std::vector<RawExpandedEntry>& executable,
                                                                  const RoutingFixture& fixture)
{
    std::vector<uint32_t> computeRows(executable.size(), 0);
    std::vector<std::vector<uint32_t>> tokenPerExpert(fixture.worldSize);
    std::vector<std::vector<uint32_t>> gatheredExpertCount(fixture.worldSize);
    std::vector<std::vector<uint32_t>> localExpertPrefix(fixture.worldSize);
    std::vector<std::vector<uint32_t>> cumsumMM(fixture.worldSize);
    std::vector<std::vector<uint32_t>> ordinalPerSrcLocalExpert(fixture.worldSize);
    for (uint32_t rank = 0; rank < fixture.worldSize; ++rank) {
        tokenPerExpert[rank].assign(fixture.worldSize * fixture.expertsPerRank, 0);
        gatheredExpertCount[rank].assign(fixture.expertsPerRank, 0);
        localExpertPrefix[rank].assign(fixture.expertsPerRank, 0);
        cumsumMM[rank].assign(fixture.expertsPerRank * fixture.worldSize, 0);
        ordinalPerSrcLocalExpert[rank].assign(fixture.worldSize * fixture.expertsPerRank, 0);
    }
    for (const auto& entry : executable) {
        if (entry.dstRank >= fixture.worldSize || entry.localExpertSlot >= fixture.expertsPerRank) {
            continue;
        }
        ++tokenPerExpert[entry.dstRank][entry.srcRank * fixture.expertsPerRank + entry.localExpertSlot];
    }
    for (uint32_t rank = 0; rank < fixture.worldSize; ++rank) {
        for (uint32_t localExpert = 0; localExpert < fixture.expertsPerRank; ++localExpert) {
            for (uint32_t srcRank = 0; srcRank < fixture.worldSize; ++srcRank) {
                gatheredExpertCount[rank][localExpert] +=
                    tokenPerExpert[rank][srcRank * fixture.expertsPerRank + localExpert];
            }
        }
        localExpertPrefix[rank] = ExclusiveScan(gatheredExpertCount[rank]);
        for (uint32_t localExpert = 0; localExpert < fixture.expertsPerRank; ++localExpert) {
            uint32_t prefix = 0;
            for (uint32_t srcRank = 0; srcRank < fixture.worldSize; ++srcRank) {
                cumsumMM[rank][localExpert * fixture.worldSize + srcRank] = prefix;
                prefix += tokenPerExpert[rank][srcRank * fixture.expertsPerRank + localExpert];
            }
        }
    }
    for (uint32_t entryIdx = 0; entryIdx < executable.size(); ++entryIdx) {
        const auto& entry = executable[entryIdx];
        if (entry.dstRank >= fixture.worldSize || entry.localExpertSlot >= fixture.expertsPerRank) {
            continue;
        }
        const uint32_t key = entry.srcRank * fixture.expertsPerRank + entry.localExpertSlot;
        const uint32_t localOrdinal = ordinalPerSrcLocalExpert[entry.dstRank][key]++;
        computeRows[entryIdx] = localExpertPrefix[entry.dstRank][entry.localExpertSlot] +
            cumsumMM[entry.dstRank][entry.localExpertSlot * fixture.worldSize + entry.srcRank] + localOrdinal;
    }
    return computeRows;
}

inline void BuildCombineTruthForLocalOwner(const std::vector<RawExpandedEntry>& executable,
                                           const RoutingFixture& fixture,
                                           uint32_t localRank,
                                           CombineTruthView& view)
{
    view = {};
    const uint32_t localRows = fixture.rowsPerRank.at(localRank);
    const auto computeRows = BuildComputeRowOrdinalForExecutable(executable, fixture);
    std::vector<std::vector<uint32_t>> rowBuckets(localRows);
    for (uint32_t entryIdx = 0; entryIdx < executable.size(); ++entryIdx) {
        const auto& entry = executable[entryIdx];
        if (entry.ownerRank != localRank) {
            continue;
        }
        rowBuckets.at(entry.rowIdx).push_back(entryIdx);
    }

    uint32_t running = 0;
    for (uint32_t row = 0; row < localRows; ++row) {
        const uint32_t begin = running;
        for (const uint32_t entryIdx : rowBuckets[row]) {
            const auto& entry = executable[entryIdx];
            view.expandedRowIdx.push_back(entry.rowIdx);
            view.expandedTopkOrdinal.push_back(entry.topkOrdinal);
            view.expandedSrcRank.push_back(entry.srcRank);
            view.expandedExpertIdx.push_back(entry.expertId);
            view.expandedDstRank.push_back(entry.dstRank);
            view.expandedComputeRow.push_back(computeRows[entryIdx]);
            view.expandedProb.push_back(entry.prob);
            view.ownerRowExpandedOrdinal.push_back(running);
            view.combineDstOffset.push_back(running);
            ++running;
        }
        view.rowToExpandedRange.push_back({begin, running});
    }
}

inline std::vector<DispatchRangeTask> CoalesceDispatchRanges(
    const std::vector<DispatchPullTask>& tasks)
{
    std::vector<DispatchRangeTask> ranges;
    for (const auto& task : tasks) {
        const bool canMerge = !ranges.empty() &&
            ranges.back().srcRank == task.srcRank &&
            ranges.back().dstRank == task.dstRank &&
            ranges.back().localExpert == task.localExpertSlot &&
            ranges.back().hiddenBytes == task.hiddenBytes &&
            ranges.back().scaleBytes == task.scaleBytes &&
            ranges.back().srcPayloadOffsetBytes + static_cast<uint64_t>(ranges.back().rowCount) * task.hiddenBytes == task.srcPayloadOffsetBytes &&
            ranges.back().dstExpertMajorOffsetBytes + static_cast<uint64_t>(ranges.back().rowCount) * task.hiddenBytes == task.dstPayloadOffsetBytes &&
            ranges.back().srcScaleOffsetBytes + static_cast<uint64_t>(ranges.back().rowCount) * task.scaleBytes == task.srcScaleOffsetBytes &&
            ranges.back().dstScaleOffsetBytes + static_cast<uint64_t>(ranges.back().rowCount) * task.scaleBytes == task.dstScaleOffsetBytes;
        if (canMerge) {
            ++ranges.back().rowCount;
            continue;
        }
        ranges.push_back({
            .srcRank = task.srcRank,
            .dstRank = task.dstRank,
            .localExpert = task.localExpertSlot,
            .rowCount = task.rowCount,
            .hiddenBytes = task.hiddenBytes,
            .scaleBytes = task.scaleBytes,
            .srcPayloadOffsetBytes = task.srcPayloadOffsetBytes,
            .dstExpertMajorOffsetBytes = task.dstPayloadOffsetBytes,
            .srcScaleOffsetBytes = task.srcScaleOffsetBytes,
            .dstScaleOffsetBytes = task.dstScaleOffsetBytes,
            .expandedRowBegin = task.dstRowBegin,
            .expandedRowEnd = task.dstRowBegin + task.rowCount,
            .kBegin = 0,
            .kEnd = task.hiddenBytes,
            .scaleOffsetBegin = task.dstRowBegin,
            .readySignalIndex = DispatchReadyIndex(task.srcRank, task.srcRowBegin),
            .doneSignalIndex = DispatchDoneIndex(task.srcRank, task.srcRowBegin),
        });
    }
    return ranges;
}

inline std::vector<CombineRangeTask> CoalesceCombineRanges(
    const std::vector<CombinePushTask>& tasks)
{
    std::vector<CombineRangeTask> ranges;
    for (const auto& task : tasks) {
        const bool canMerge = !ranges.empty() &&
            ranges.back().srcRank == task.srcRank &&
            ranges.back().ownerRank == task.dstRank &&
            ranges.back().localExpert == task.expertGroupId &&
            ranges.back().outputBytes == task.outputBytes &&
            ranges.back().gmm2TileReadyIndex == task.gmm2TileReadyIndex &&
            ranges.back().taskFlags == task.taskFlags &&
            ranges.back().srcGmm2OffsetBytes + static_cast<uint64_t>(ranges.back().rowCount) * task.outputBytes == task.srcPayloadOffsetBytes &&
            ranges.back().dstCombineOffsetBytes + static_cast<uint64_t>(ranges.back().rowCount) * task.outputBytes == task.dstPayloadOffsetBytes;
        if (canMerge) {
            ++ranges.back().rowCount;
            ranges.back().gmm2ReadyRows = task.gmm2ReadyRows;
            continue;
        }
        ranges.push_back({
            .srcRank = task.srcRank,
            .ownerRank = task.dstRank,
            .localExpert = task.expertGroupId,
            .rowCount = task.rowCount,
            .outputBytes = task.outputBytes,
            .srcGmm2OffsetBytes = task.srcPayloadOffsetBytes,
            .dstCombineOffsetBytes = task.dstPayloadOffsetBytes,
            .probOffset = task.srcRowBegin,
            .readySignalIndex = CombineReadyIndex(task.dstRank, task.dstRowBegin),
            .gmm2TileReadyIndex = task.gmm2TileReadyIndex,
            .gmm2ReadyRows = task.gmm2ReadyRows,
            .ownerCompletionIndex = task.ownerCompletionIndex,
            .taskFlags = task.taskFlags,
        });
    }
    return ranges;
}

inline std::vector<uint32_t> BuildOwnerCombineDstOffsetForExecutable(const std::vector<RawExpandedEntry>& executable,
                                                                  const RoutingFixture& fixture)
{
    std::vector<std::vector<uint32_t>> rowCounts(fixture.worldSize);
    std::vector<std::vector<uint32_t>> rowBases(fixture.worldSize);
    std::vector<std::vector<uint32_t>> rowOrdinals(fixture.worldSize);
    for (uint32_t ownerRank = 0; ownerRank < fixture.worldSize; ++ownerRank) {
        const uint32_t rows = fixture.rowsPerRank.at(ownerRank);
        rowCounts[ownerRank].assign(rows, 0);
        rowBases[ownerRank].assign(rows, 0);
        rowOrdinals[ownerRank].assign(rows, 0);
    }

    for (const auto& entry : executable) {
        ++rowCounts.at(entry.ownerRank).at(entry.rowIdx);
    }
    for (uint32_t ownerRank = 0; ownerRank < fixture.worldSize; ++ownerRank) {
        uint32_t running = 0;
        for (uint32_t row = 0; row < rowCounts[ownerRank].size(); ++row) {
            rowBases[ownerRank][row] = running;
            running += rowCounts[ownerRank][row];
        }
    }

    std::vector<uint32_t> dstOffsets;
    dstOffsets.reserve(executable.size());
    for (const auto& entry : executable) {
        auto& ownerRowOrdinal = rowOrdinals.at(entry.ownerRank).at(entry.rowIdx);
        dstOffsets.push_back(rowBases.at(entry.ownerRank).at(entry.rowIdx) + ownerRowOrdinal);
        ++ownerRowOrdinal;
    }
    return dstOffsets;
}

inline ScoreboardMeta BuildScoreboardMeta(uint32_t expertsPerRank,
                                                             uint32_t worldSize,
                                                             uint32_t dispatchRangeCount,
                                                             uint32_t computeTileCount,
                                                             uint32_t ownerCompletionCount)
{
    ScoreboardMeta meta{};
    meta.dispatchRangeStatusBase = SummaryIndex(0, 0);
    meta.dispatchRangeStatusCount = expertsPerRank * worldSize;
    meta.expertSafeRowsBase = meta.dispatchRangeStatusBase + meta.dispatchRangeStatusCount;
    meta.expertSafeRowsCount = expertsPerRank;
    constexpr uint32_t signalSlotElems = kCombineTileAlignBytes / sizeof(int32_t);
    meta.dispatchRangeCompletionBase =
        ((meta.expertSafeRowsBase + meta.expertSafeRowsCount + signalSlotElems - 1U) / signalSlotElems) * signalSlotElems;
    meta.dispatchRangeCompletionCount = dispatchRangeCount * (kCombineTileAlignBytes / sizeof(int32_t));
    meta.computeTileStatusBase = meta.dispatchRangeCompletionBase + meta.dispatchRangeCompletionCount;
    meta.computeTileStatusCount = computeTileCount;
    meta.gmm2TileReadyBase = meta.computeTileStatusBase + meta.computeTileStatusCount;
    meta.gmm2TileReadyCount = computeTileCount;
    meta.ownerCompletionBase = meta.gmm2TileReadyBase + meta.gmm2TileReadyCount;
    meta.ownerCompletionCount = ownerCompletionCount;
    meta.totalWords = meta.ownerCompletionBase + meta.ownerCompletionCount;
    return meta;
}

inline RuntimeRoutingMeta MaterializeRuntimeRoutingMetadata(const RoutingFixture& fixture,
                                                            RoutingPlanBundle& bundle)
{
    bundle.runtimeRoutingWords.clear();
    RuntimeRoutingMeta meta{};

    auto appendVector = [&](const std::vector<uint32_t>& values, uint32_t& offset, uint32_t& count) {
        offset = static_cast<uint32_t>(bundle.runtimeRoutingWords.size());
        count = static_cast<uint32_t>(values.size());
        bundle.runtimeRoutingWords.insert(bundle.runtimeRoutingWords.end(), values.begin(), values.end());
    };

    appendVector(bundle.view.dispatch.tokenPerExpert, meta.tokenPerExpertOffset, meta.tokenPerExpertCount);
    appendVector(bundle.view.dispatch.cumsumMM, meta.cumsumMMOffset, meta.cumsumMMCount);

    std::vector<uint32_t> cumsumSend(fixture.worldSize * fixture.expertsPerRank, 0);
    std::vector<uint32_t> runningSend(fixture.worldSize, 0);
    for (const auto& entry : bundle.executable) {
        if (entry.srcRank >= fixture.worldSize || entry.localExpertSlot >= fixture.expertsPerRank) {
            continue;
        }
        const size_t index = static_cast<size_t>(entry.srcRank) * fixture.expertsPerRank + entry.localExpertSlot;
        cumsumSend[index] = runningSend[entry.srcRank];
        ++runningSend[entry.srcRank];
    }
    appendVector(cumsumSend, meta.cumsumSendOffset, meta.cumsumSendCount);

    std::vector<uint32_t> expertTokensBeforeCapacity(fixture.worldSize * fixture.expertsPerRank, 0);
    for (const auto& entry : bundle.beforeCapacity) {
        if (entry.isSentinel || entry.srcRank >= fixture.worldSize || entry.localExpertSlot >= fixture.expertsPerRank) {
            continue;
        }
        ++expertTokensBeforeCapacity[entry.srcRank * fixture.expertsPerRank + entry.localExpertSlot];
    }
    appendVector(expertTokensBeforeCapacity,
                 meta.expertTokensBeforeCapacityOffset,
                 meta.expertTokensBeforeCapacityCount);

    meta.droppedByCapacityCount = bundle.semanticMeta.droppedByCapacityCount;
    meta.sentinelDropCount = bundle.semanticMeta.sentinelDropCount;
    meta.groupCount = fixture.expertsPerRank;
    bundle.runtimeRoutingMeta = meta;
    return meta;
}

inline void MaterializeExpertAndScoreboardMetadata(const RoutingFixture& fixture,
                                                   RoutingPlanBundle& bundle)
{
    constexpr uint32_t kDefaultComputeTileRows = 8;
    bundle.expertRanges.clear();
    bundle.expertGroupRanges.clear();
    bundle.expertSourceSegments.clear();
    bundle.computeTiles.clear();

    for (uint32_t tileRowBegin = 0; tileRowBegin < bundle.view.dispatch.packedRowCount; tileRowBegin += kDefaultComputeTileRows) {
        const uint32_t tileRowEnd = std::min(bundle.view.dispatch.packedRowCount, tileRowBegin + kDefaultComputeTileRows);
        bundle.computeTiles.push_back({
            .rowBegin = tileRowBegin,
            .rowEnd = tileRowEnd,
            .sourceSegmentBegin = 0,
            .sourceSegmentEnd = 0,
            .kBegin = 0,
            .kEnd = fixture.hiddenBytes,
            .nBegin = 0,
            .nEnd = fixture.gmm1N,
            .expandedRowBegin = tileRowBegin,
            .scaleOffsetBegin = tileRowBegin,
        });
    }

    bundle.scoreboardMeta = BuildScoreboardMeta(fixture.expertsPerRank,
                                                fixture.worldSize,
                                                static_cast<uint32_t>(bundle.dispatchRanges.size()),
                                                static_cast<uint32_t>(bundle.computeTiles.size()),
                                                static_cast<uint32_t>(bundle.view.combine.combineDstOffset.size()));
    for (uint32_t tileIndex = 0; tileIndex < bundle.computeTiles.size(); ++tileIndex) {
        bundle.computeTiles[tileIndex].computeTileStatusIndex =
            ComputeTileStatusIndex(bundle.scoreboardMeta, tileIndex);
        bundle.computeTiles[tileIndex].gmm2TileReadyIndex =
            Gmm2TileReadyIndex(bundle.scoreboardMeta, tileIndex);
    }

    bundle.expertRanges.reserve(fixture.expertsPerRank);
    bundle.expertGroupRanges.reserve(fixture.expertsPerRank);
    bundle.expertSourceSegments.reserve(bundle.dispatchRanges.size());
    for (uint32_t localExpert = 0; localExpert < fixture.expertsPerRank; ++localExpert) {
        const uint32_t rowBegin = bundle.view.dispatch.localExpertPrefix[localExpert];
        const uint32_t rowEnd = rowBegin + bundle.view.dispatch.gatheredExpertCount[localExpert];
        uint32_t computeTileBegin = static_cast<uint32_t>(bundle.computeTiles.size());
        uint32_t computeTileEnd = computeTileBegin;
        for (uint32_t tileIndex = 0; tileIndex < bundle.computeTiles.size(); ++tileIndex) {
            auto& tile = bundle.computeTiles[tileIndex];
            if (tile.rowBegin >= rowEnd || tile.rowEnd <= rowBegin) {
                continue;
            }
            if (computeTileBegin == bundle.computeTiles.size()) {
                computeTileBegin = tileIndex;
            }
            computeTileEnd = tileIndex + 1;
            tile.localExpert = localExpert;
            tile.expertRangeIndex = localExpert;
        }
        uint32_t dispatchRangeBegin = static_cast<uint32_t>(bundle.dispatchRanges.size());
        uint32_t dispatchRangeEnd = dispatchRangeBegin;
        for (uint32_t rangeIndex = 0; rangeIndex < bundle.dispatchRanges.size(); ++rangeIndex) {
            if (bundle.dispatchRanges[rangeIndex].localExpert != localExpert) {
                continue;
            }
            if (dispatchRangeBegin == bundle.dispatchRanges.size()) {
                dispatchRangeBegin = rangeIndex;
            }
            dispatchRangeEnd = rangeIndex + 1;
        }
        bundle.expertRanges.push_back({
            .localExpert = localExpert,
            .rowBegin = rowBegin,
            .rowEnd = rowEnd,
            .sourceSegmentBegin = 0,
            .sourceSegmentEnd = 0,
            .computeTileBegin = computeTileBegin,
            .computeTileEnd = computeTileEnd,
            .expertSafeRowsIndex = ExpertSafeRowsIndex(bundle.scoreboardMeta, localExpert),
        });
        bundle.expertGroupRanges.push_back({
            .expertGroupId = localExpert,
            .dispatchRangeBegin = dispatchRangeBegin,
            .dispatchRangeEnd = dispatchRangeEnd,
            .computeTileBegin = computeTileBegin,
            .computeTileEnd = computeTileEnd,
        });
    }
}

inline RoutingPlanBundle BuildRoutePlanForRank(const RoutingFixture& fixture,
                                               uint32_t localRank)
{
    RoutingPlanBundle bundle;
    bundle.beforeCapacity = ExpandBeforeCapacity(fixture);
    bundle.executable = KeepExecutableEntries(bundle.beforeCapacity, fixture);
    bundle.semanticLedger = BuildSemanticLedger(bundle.beforeCapacity, fixture);
    bundle.semanticMeta = BuildRoutingSemanticMeta(bundle.beforeCapacity, bundle.executable, fixture);
    BuildDispatchTruthForLocalRank(bundle.executable, fixture, localRank, bundle.view.dispatch);
    BuildCombineTruthForLocalOwner(bundle.executable, fixture, localRank, bundle.view.combine);
    MaterializeRuntimeRoutingMetadata(fixture, bundle);
    const auto ownerCombineDstOffset = BuildOwnerCombineDstOffsetForExecutable(bundle.executable, fixture);

    const uint64_t dispatchSegmentBytes = static_cast<uint64_t>(fixture.maxOutputSize) * fixture.hiddenBytes;
    const uint64_t dispatchScaleBaseBytes = static_cast<uint64_t>(fixture.worldSize) * dispatchSegmentBytes;
    const uint64_t computeScaleBaseBytes = dispatchSegmentBytes;
    const uint64_t dispatchScaleSegmentBytes = static_cast<uint64_t>(fixture.maxOutputSize) * kDispatchScaleSlotBytes;
    for (size_t i = 0; i < bundle.view.dispatch.srcOffset.size(); ++i) {
        bundle.dispatchTasks.push_back({
            .srcRank = bundle.view.dispatch.expandedSrcRank[i],
            .dstRank = localRank,
            .expertGroupId = bundle.view.dispatch.expandedLocalExpertSlot[i],
            .expertId = bundle.view.dispatch.expandedExpertIdx[i],
            .localExpertSlot = bundle.view.dispatch.expandedLocalExpertSlot[i],
            .srcRowBegin = bundle.view.dispatch.srcOffset[i],
            .dstRowBegin = bundle.view.dispatch.dstOffset[i],
            .rowCount = 1,
            .hiddenBytes = fixture.hiddenBytes,
            .scaleBytes = kDispatchScaleSlotBytes,
            .srcPayloadOffsetBytes = dispatchSegmentBytes * localRank +
                                     static_cast<uint64_t>(bundle.view.dispatch.srcOffset[i]) * fixture.hiddenBytes,
            .dstPayloadOffsetBytes = static_cast<uint64_t>(bundle.view.dispatch.dstOffset[i]) * fixture.hiddenBytes,
            .srcScaleOffsetBytes = dispatchScaleBaseBytes +
                                   dispatchScaleSegmentBytes * localRank +
                                   static_cast<uint64_t>(bundle.view.dispatch.srcOffset[i]) * kDispatchScaleSlotBytes,
            .dstScaleOffsetBytes = computeScaleBaseBytes +
                                   static_cast<uint64_t>(bundle.view.dispatch.dstOffset[i]) * kDispatchScaleSlotBytes,
            .readyEpoch = 1,
            .taskFlags = 0,
        });
    }

    uint32_t tileId = 0;
    uint32_t localDispatchOrdinal = 0;
    for (size_t entryIdx = 0; entryIdx < bundle.executable.size(); ++entryIdx) {
        const auto& entry = bundle.executable[entryIdx];
        if (entry.dstRank != localRank) {
            continue;
        }
        const uint32_t computeRowOrdinal = bundle.view.dispatch.dstOffset.at(localDispatchOrdinal++);
        const uint32_t ownerOrdinal = ownerCombineDstOffset[entryIdx];
        bundle.combineTasks.push_back({
            .srcRank = localRank,
            .dstRank = entry.ownerRank,
            .expertGroupId = entry.localExpertSlot,
            .expertId = entry.expertId,
            .ownerRow = entry.rowIdx,
            .tileId = tileId++,
            .srcRowBegin = computeRowOrdinal,
            .dstRowBegin = ownerOrdinal,
            .rowCount = 1,
            .outputBytes = fixture.outputBytes,
            .srcPayloadOffsetBytes = static_cast<uint64_t>(computeRowOrdinal) * fixture.outputBytes,
            .dstPayloadOffsetBytes = static_cast<uint64_t>(ownerOrdinal) * fixture.outputBytes,
            .completionEpoch = 1,
            .taskFlags = kCombineTaskWaitGmm2Ready,
        });
    }

    bundle.dispatchRanges = CoalesceDispatchRanges(bundle.dispatchTasks);
    MaterializeExpertAndScoreboardMetadata(fixture, bundle);
    for (auto& task : bundle.combineTasks) {
        const uint32_t computeRowOrdinal = task.srcRowBegin;
        for (uint32_t tileIndex = 0; tileIndex < bundle.computeTiles.size(); ++tileIndex) {
            const auto& tile = bundle.computeTiles[tileIndex];
            if (computeRowOrdinal >= tile.rowBegin && computeRowOrdinal < tile.rowEnd) {
                task.gmm2TileReadyIndex = tile.gmm2TileReadyIndex;
                break;
            }
        }
        task.ownerCompletionIndex = OwnerCompletionSummaryIndex(task.dstRank, task.dstRowBegin);
    }
    bundle.combineRanges = CoalesceCombineRanges(bundle.combineTasks);
    for (auto& group : bundle.expertGroupRanges) {
        uint32_t combineRangeBegin = static_cast<uint32_t>(bundle.combineRanges.size());
        uint32_t combineRangeEnd = combineRangeBegin;
        for (uint32_t rangeIndex = 0; rangeIndex < bundle.combineRanges.size(); ++rangeIndex) {
            if (bundle.combineRanges[rangeIndex].localExpert != group.expertGroupId) {
                continue;
            }
            if (combineRangeBegin == bundle.combineRanges.size()) {
                combineRangeBegin = rangeIndex;
            }
            combineRangeEnd = rangeIndex + 1;
        }
        group.combineRangeBegin = combineRangeBegin;
        group.combineRangeEnd = combineRangeEnd;
    }
    std::vector<uint32_t> rangeBeginPerLocalExpert(fixture.expertsPerRank, 0);
    std::vector<uint32_t> rangeCountPerLocalExpert(fixture.expertsPerRank, 0);
    for (const auto& range : bundle.dispatchRanges) {
        ++rangeCountPerLocalExpert[range.localExpert];
    }
    uint32_t rangeCompletionCursor = 0;
    for (uint32_t localExpert = 0; localExpert < fixture.expertsPerRank; ++localExpert) {
        rangeBeginPerLocalExpert[localExpert] = rangeCompletionCursor;
        rangeCompletionCursor += rangeCountPerLocalExpert[localExpert];
    }
    std::vector<uint32_t> rangeOrdinalPerLocalExpert(fixture.expertsPerRank, 0);
    struct PendingExpertSourceSegment {
        uint32_t tileIndex = 0;
        ExpertSourceSegmentMeta segment;
    };
    std::vector<PendingExpertSourceSegment> pendingSourceSegments;
    for (auto& range : bundle.dispatchRanges) {
        range.rangeStatusIndex = DispatchRangeStatusIndex(
            bundle.scoreboardMeta, range.localExpert, range.srcRank, fixture.worldSize);
        const uint32_t cumsum = bundle.view.dispatch.cumsumMM[range.localExpert * fixture.worldSize + range.srcRank];
        range.readyRows = cumsum + range.rowCount;
        range.rangeStatusBase = bundle.scoreboardMeta.dispatchRangeStatusBase + range.localExpert * fixture.worldSize;
        range.expertSafeRowsIndex = ExpertSafeRowsIndex(bundle.scoreboardMeta, range.localExpert);
        range.dispatchRangeCompletionBase = DispatchRangeCompletionIndex(
            bundle.scoreboardMeta, rangeBeginPerLocalExpert[range.localExpert]);
        range.dispatchRangeCompletionIndex = DispatchRangeCompletionIndex(
            bundle.scoreboardMeta, rangeBeginPerLocalExpert[range.localExpert] + rangeOrdinalPerLocalExpert[range.localExpert]++);
        range.expectedRangeCount = rangeCountPerLocalExpert[range.localExpert];
        range.localExpertRowBegin = bundle.view.dispatch.localExpertPrefix[range.localExpert];
        range.localExpertRowEnd = range.localExpertRowBegin + bundle.view.dispatch.gatheredExpertCount[range.localExpert];
        range.worldSize = fixture.worldSize;
        const ExpertSourceSegmentMeta segment{
            .localExpert = range.localExpert,
            .srcRank = range.srcRank,
            .rowBegin = range.localExpertRowBegin + cumsum,
            .rowEnd = range.localExpertRowBegin + range.readyRows,
            .dispatchRangeStatusIndex = range.rangeStatusIndex,
            .dispatchRangeCompletionIndex = range.dispatchRangeCompletionIndex,
            .doneSignalIndex = range.doneSignalIndex,
        };
        for (uint32_t tileIndex = 0; tileIndex < bundle.computeTiles.size(); ++tileIndex) {
            const auto& tile = bundle.computeTiles[tileIndex];
            if (tile.rowBegin >= range.localExpertRowBegin + range.readyRows ||
                tile.rowEnd <= range.localExpertRowBegin + cumsum) {
                continue;
            }
            pendingSourceSegments.push_back({
                .tileIndex = tileIndex,
                .segment = segment,
            });
        }
        uint32_t sourceMask = 0;
        for (uint32_t srcRank = 0; srcRank < fixture.worldSize; ++srcRank) {
            if (bundle.view.dispatch.tokenPerExpert[srcRank * fixture.expertsPerRank + range.localExpert] != 0) {
                sourceMask |= (1U << srcRank);
            }
        }
        range.sourceExpectedMask = sourceMask;
    }
    bundle.expertSourceSegments.clear();
    for (uint32_t tileIndex = 0; tileIndex < bundle.computeTiles.size(); ++tileIndex) {
        auto& tile = bundle.computeTiles[tileIndex];
        tile.sourceSegmentBegin = static_cast<uint32_t>(bundle.expertSourceSegments.size());
        for (const auto& pending : pendingSourceSegments) {
            if (pending.tileIndex == tileIndex) {
                bundle.expertSourceSegments.push_back(pending.segment);
            }
        }
        tile.sourceSegmentEnd = static_cast<uint32_t>(bundle.expertSourceSegments.size());
    }
    return bundle;
}

