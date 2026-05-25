#include "tiling_builder.hpp"

#include "op_kernel/protocol/task_plan.hpp"


WorkspaceLayoutConfig BuildDefaultWorkspaceLayout() {
    return WorkspaceLayoutConfig{};
}

KernelLaunchConfig BuildDefaultLaunchConfig() {
    KernelLaunchConfig cfg{};
    cfg.expertsPerRank = 2;
    cfg.numExperts = 2;
    cfg.worldSize = 2;
    cfg.numExpertGroups = 2;
    cfg.maxOutputSize = 8;
    return cfg;
}

namespace {

WorkspaceRegionDesc Region(uint64_t offset, uint64_t bytes) {
    return WorkspaceRegionDesc{offset, bytes, 64};
}

bool IsAligned(uint64_t value, uint64_t alignment) {
    return alignment == 0 || (value % alignment) == 0;
}

}  // namespace

MegaMoeTilingData BuildMegaMoeTilingData(const ModeConfig& mode,
                                         const WorkspaceLayoutConfig& layout,
                                         uint32_t dispatchTaskCount,
                                         uint32_t combineTaskCount,
                                         uint32_t blockDim,
                                         uint32_t hiddenBytes,
                                         uint32_t outputBytes) {
    MegaMoeTilingData tiling{};
    tiling.m = mode.m;
    tiling.k = mode.k;
    tiling.n = mode.n;
    tiling.topk = mode.topk;
    tiling.expertsPerRank = mode.expertsPerRank;
    tiling.worldSize = mode.worldSize;
    tiling.maxOutputSize = mode.maxOutputSize;
    tiling.blockDim = blockDim;
    tiling.dispatchTaskCount = dispatchTaskCount;
    tiling.combineTaskCount = combineTaskCount;
    tiling.hiddenBytes = hiddenBytes;
    tiling.outputBytes = outputBytes;

    uint64_t offset = 0;
    tiling.control = Region(offset, layout.controlBytes);
    offset += layout.controlBytes;
    tiling.dispatch = Region(offset, layout.dispatchBytes);
    offset += layout.dispatchBytes;
    tiling.compute = Region(offset, layout.computeBytes);
    offset += layout.computeBytes;
    tiling.combine = Region(offset, layout.combineBytes);
    offset += layout.combineBytes;
    tiling.signal = Region(offset, layout.signalBytes);
    return tiling;
}

bool ValidateMegaMoeWorkspace(const MegaMoeTilingData& tiling) {
    const WorkspaceRegionDesc regions[] = {
        tiling.control,
        tiling.dispatch,
        tiling.compute,
        tiling.combine,
        tiling.signal,
    };
    for (const auto& region : regions) {
        if (!IsAligned(region.offset, region.alignment) || !IsAligned(region.bytes, region.alignment)) {
            return false;
        }
        if (region.bytes == 0) {
            return false;
        }
    }
    const uint64_t dispatchBytes = static_cast<uint64_t>(tiling.worldSize) * tiling.maxOutputSize *
                                   (tiling.hiddenBytes + kDispatchScaleSlotBytes);
    const uint64_t computeBytes = static_cast<uint64_t>(tiling.maxOutputSize) *
                                  (tiling.hiddenBytes + kDispatchScaleSlotBytes);
    const uint64_t combineBytes = static_cast<uint64_t>(tiling.maxOutputSize) * tiling.outputBytes;
    return dispatchBytes <= tiling.dispatch.bytes &&
           computeBytes <= tiling.compute.bytes &&
           combineBytes <= tiling.combine.bytes;
}

