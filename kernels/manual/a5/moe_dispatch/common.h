/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef MOE_DISPATCH_COMMON_H_
#define MOE_DISPATCH_COMMON_H_

#include <cstdint>

namespace moe_dispatch {

constexpr uint32_t kMaxMoeDispatchRanks = 64;
constexpr uint64_t kMoeDispatchKiB = 1024ULL;
constexpr uint64_t kMoeDispatchMiB = 1024ULL * 1024ULL;
constexpr uint64_t kMoeDispatchWindowHeadGuardBytes = 4ULL * kMoeDispatchKiB;
constexpr uint64_t kMoeDispatchTailControlBytes = 2ULL * kMoeDispatchMiB;
constexpr uint64_t kMoeDispatchSignalBytes = 1ULL * kMoeDispatchMiB;
constexpr uint64_t kMoeDispatchWindowAlignBytes = 512ULL;
constexpr uint64_t kMoeDispatchSignalStrideI32 = 16ULL;

enum class MoeDispatchWindowLayoutMode : uint32_t
{
    GuardedDispatchOnly = 0,
    FusionCompatible = 1,
};

struct MoeDispatchShape {
    uint32_t ep;
    uint32_t m;
    uint32_t k;
    uint32_t topK;
    uint32_t expertPerRank;
    uint32_t expertNum;
    uint32_t maxOutputSize;
    uint32_t aivBlocks;
    uint32_t tileCols;
    uint32_t metadataPad;
    uint32_t signalValue;
};

struct MoeDispatchRuntimeConfig {
    uint32_t runMode;
    uint32_t socVersion;
    uint32_t deviceBase;
    uint32_t ndevices;
    uint32_t rankFromMpi;
    uint32_t rank;
    uint32_t nranks;
    uint32_t debug;
    uint32_t iters;
    uint32_t warmup;
    uint32_t seed;
    uint32_t genData;
    uint32_t verify;
    uint32_t skipRun;
    uint32_t skipBuild;
    uint32_t cleanBuild;
    uint32_t hostGoldenOnly;
    uint32_t keepHcclShm;
    uint64_t hcclBuffSizeMb;
};

struct MoeDispatchResourceConfig {
    uint32_t defaultAicBlocks;
    uint32_t defaultAivRatio;
    uint32_t defaultAivBlocks;
    uint32_t maxAivBlocks;
    uint32_t blockDim;
};

struct MoeDispatchWorkspaceLayout {
    uint64_t localTokenPerExpert;
    uint64_t blockTokenPerExpert;
    uint64_t blockPrefixPerExpert;
    uint64_t routeExpert;
    uint64_t routePackedRow;
    uint64_t localSync;
    uint64_t tileScratch;
    uint64_t totalBytes;
};

struct MoeDispatchWindowLayout {
    MoeDispatchWindowLayoutMode mode;
    uint64_t headGuard;
    uint64_t packedA;
    uint64_t expandedRowIdx;
    uint64_t reservedScratch;
    uint64_t tokenPerExpert;
    uint64_t signal;
    uint64_t totalVisibleBytes;
};

struct PtoRemoteWindowContext {
    uint64_t workspaceBase;
    uint64_t workspaceBytes;
    uint32_t rank;
    uint32_t rankSize;
    uint64_t windowBytes;
    uint64_t windowIn[kMaxMoeDispatchRanks];
    uint64_t windowOut[kMaxMoeDispatchRanks];
};

} // namespace moe_dispatch

#endif // MOE_DISPATCH_COMMON_H_
