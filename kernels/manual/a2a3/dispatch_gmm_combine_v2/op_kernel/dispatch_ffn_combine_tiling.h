#pragma once

#include <cstdint>


enum class KernelMode : uint32_t {
    DispatchRangeOnly = 1,
    CombineOnly = 2,
};

struct KernelLaunchConfig {
    uint32_t m = 16;
    uint32_t k = 128;
    uint32_t n = 128;
    uint32_t topk = 2;
    uint32_t numExperts = 2;
    uint32_t expertsPerRank = 1;
    uint32_t worldSize = 2;
    uint32_t numExpertGroups = 1;
    uint32_t rowsPerGroup = 16;
    uint32_t maxOutputSize = 32;
};

struct WorkspaceLayoutConfig {
    uint64_t controlBytes = 0x200000;
    uint64_t dispatchBytes = 0x300000;
    uint64_t computeBytes = 0x200000;
    uint64_t combineBytes = 0x200000;
    uint64_t signalBytes = 0x20000;

    uint64_t TotalBytes() const {
        return controlBytes + dispatchBytes + computeBytes + combineBytes + signalBytes;
    }
};

struct WorkspaceRegionDesc {
    uint64_t offset = 0;
    uint64_t bytes = 0;
    uint64_t alignment = 64;
};

struct MegaMoeParams {
    uint64_t remoteWindow = 0;
    uint64_t workspace = 0;
    uint64_t dispatchTasks = 0;
    uint64_t combineTasks = 0;
    uint64_t input = 0;
    uint64_t weight1 = 0;
    uint64_t weight2 = 0;
    uint64_t scale1 = 0;
    uint64_t scale2 = 0;
    uint64_t probs = 0;
    uint64_t xActiveMask = 0;
    uint64_t output = 0;
};

struct MegaMoeTilingData {
    uint32_t m = 0;
    uint32_t k = 0;
    uint32_t n = 0;
    uint32_t topk = 0;
    uint32_t expertsPerRank = 0;
    uint32_t worldSize = 0;
    uint32_t maxOutputSize = 0;
    uint32_t blockDim = 1;
    uint32_t dispatchTaskCount = 0;
    uint32_t combineTaskCount = 0;
    uint32_t hiddenBytes = 0;
    uint32_t outputBytes = 0;
    WorkspaceRegionDesc control;
    WorkspaceRegionDesc dispatch;
    WorkspaceRegionDesc compute;
    WorkspaceRegionDesc combine;
    WorkspaceRegionDesc signal;
};

struct StandaloneKernelTilingData {
    uint32_t mode = 0;
    uint32_t taskCount = 0;
    uint32_t hiddenBytes = 0;
    uint32_t outputBytes = 0;
    uint32_t groupId = 0;
    uint32_t groupWidth = 0;
    uint32_t maxOutputSize = 0;
    uint32_t reserved = 0;
};

struct CombineOnlyParams {
    uint64_t localPartial = 0;
    uint64_t restoredRows = 0;
    uint64_t combineTasks = 0;
    uint64_t combineRangeTasks = 0;
    uint64_t expandedProb = 0;
    uint64_t rowRanges = 0;
    uint64_t incomingCombineTasks = 0;
    uint64_t doneValue = 0;
    uint64_t perTokenScale2 = 0;
    uint32_t taskCount = 0;
    uint32_t rangeTaskCount = 0;
    uint32_t localRowCount = 0;
    uint32_t phase = 0;
    uint32_t incomingTaskCount = 0;
    uint32_t outputElems = 0;
};

