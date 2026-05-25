#pragma once

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>


struct ReadyQueueEntry {
    uint32_t groupId = 0;
    uint32_t readyEpoch = 0;
    uint32_t expectedSourceCount = 0;
    uint32_t completedSourceCount = 0;
};

struct ReadyQueueSummary {
    uint32_t dispatchReady = 0;
    uint32_t computeReady = 0;
    uint32_t combineReady = 0;
    uint32_t completed = 0;
};

struct DispatchSummaryView {
    uint32_t minReadyEpoch = 0;
    uint32_t visibleSourceCount = 0;
};

struct ExpertGroupSchedule {
    std::vector<uint32_t> widths;

    uint32_t WidthForStep(uint32_t step) const
    {
        return step < widths.size() ? widths[step] : 1;
    }
};

struct ExpertSafeRowsView {
    std::vector<uint32_t> safeRows;
};

struct ComputeTileReadyView {
    std::vector<uint8_t> ready;
};

struct ReadyQueue {
    std::vector<ReadyQueueEntry> dispatchEntries;
    std::vector<ReadyQueueEntry> computeEntries;
    std::vector<ReadyQueueEntry> combineEntries;

    void PushDispatchReady(uint32_t groupId, uint32_t readyEpoch, uint32_t expectedSourceCount)
    {
        dispatchEntries.push_back({groupId, readyEpoch, expectedSourceCount, 0});
    }

    void PushComputeReady(uint32_t groupId, uint32_t readyEpoch)
    {
        computeEntries.push_back({groupId, readyEpoch, 0, 0});
    }

    void PushCombineReady(uint32_t groupId, uint32_t readyEpoch)
    {
        combineEntries.push_back({groupId, readyEpoch, 0, 0});
    }
};

inline ExpertSafeRowsView BuildExpertSafeRows(uint32_t expertsPerRank,
                                               uint32_t worldSize,
                                               const std::vector<uint32_t>& dispatchRangeStatus,
                                               const std::vector<uint32_t>& tokenPerExpert,
                                               const std::vector<uint32_t>& localExpertPrefix,
                                               const std::vector<uint32_t>& gatheredExpertCount)
{
    ExpertSafeRowsView view;
    view.safeRows.assign(expertsPerRank, 0);
    for (uint32_t expert = 0; expert < expertsPerRank; ++expert) {
        uint32_t safeRowsWithinExpert = gatheredExpertCount.at(expert);
        bool hasRequiredSource = false;
        for (uint32_t srcRank = 0; srcRank < worldSize; ++srcRank) {
            const uint32_t expectedRows = tokenPerExpert.at(srcRank * expertsPerRank + expert);
            if (expectedRows == 0) {
                continue;
            }
            hasRequiredSource = true;
            if (dispatchRangeStatus.at(expert * worldSize + srcRank) == 0) {
                safeRowsWithinExpert = 0;
                break;
            }
        }
        view.safeRows[expert] = localExpertPrefix.at(expert) + (hasRequiredSource ? safeRowsWithinExpert : 0);
    }
    return view;
}

inline ComputeTileReadyView BuildComputeTileReady(const ExpertSafeRowsView& safeRows,
                                                  const std::vector<uint32_t>& tileExperts,
                                                  const std::vector<uint32_t>& tileRowEnds)
{
    ComputeTileReadyView view;
    view.ready.assign(tileExperts.size(), 0);
    for (size_t tile = 0; tile < tileExperts.size(); ++tile) {
        const uint32_t expert = tileExperts.at(tile);
        view.ready[tile] = tileRowEnds.at(tile) <= safeRows.safeRows.at(expert) ? 1 : 0;
    }
    return view;
}

inline ExpertGroupSchedule BuildExpertGroupSchedule(uint32_t experts)
{
    ExpertGroupSchedule schedule;
    if (experts == 0) {
        return schedule;
    }
    const uint32_t seed[] = {8, 4, 2, 1, 1};
    uint32_t remaining = experts;
    for (uint32_t width : seed) {
        if (remaining == 0) {
            break;
        }
        const uint32_t clamped = width < remaining ? width : remaining;
        schedule.widths.push_back(clamped);
        remaining -= clamped;
    }
    while (remaining > 0) {
        schedule.widths.push_back(1);
        --remaining;
    }
    return schedule;
}

