#pragma once

#include "op_kernel/dispatch_ffn_combine_tiling.h"
#include "runtime_context.hpp"


WorkspaceLayoutConfig BuildDefaultWorkspaceLayout();
KernelLaunchConfig BuildDefaultLaunchConfig();
MegaMoeTilingData BuildMegaMoeTilingData(const ModeConfig& mode,
                                         const WorkspaceLayoutConfig& layout,
                                         uint32_t dispatchTaskCount,
                                         uint32_t combineTaskCount,
                                         uint32_t blockDim,
                                         uint32_t hiddenBytes,
                                         uint32_t outputBytes);
bool ValidateMegaMoeWorkspace(const MegaMoeTilingData& tiling);

