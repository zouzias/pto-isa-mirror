/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef DISPATCH_COMBINE_TILE_COMMON_H_
#define DISPATCH_COMBINE_TILE_COMMON_H_

#include <cstdint>

namespace dispatch_combine_tile {

constexpr uint32_t kMaxDispatchCombineTileRanks = 64;

// Shared ABI names. Later tasks may add helper methods, but these struct names
// and field names are the host/kernel contract for the first PTO version.
struct DispatchCombineTileShape {
    uint32_t ep;
    uint32_t m;
    uint32_t k;
    uint32_t topK;
    uint32_t expertPerRank;
    uint32_t expertNum;
    uint32_t maxOutputSize;
    uint32_t aivBlocks;
    uint32_t tileCols;
    uint32_t rowChunk;
    uint32_t metadataPad;
    uint32_t signalValue;
};

struct WorkspaceLayout {
    uint64_t localTokenPerExpert;
    uint64_t blockTokenPerExpert;
    uint64_t blockPrefixPerExpert;
    uint64_t cumsumPerExpert;
    uint64_t dispatchOffset;
    uint64_t prevSumBeforeRank;
    uint64_t localSync;
    uint64_t floatScratch;
    uint64_t dispatchedA;
    uint64_t ptrDLocal;
    uint64_t totalBytes;
};

struct PeerWindowLayout {
    uint64_t peerTokenPerExpert;
    uint64_t expandedRowIdx;
    uint64_t packedA;
    uint64_t ptrD;
    uint64_t countReadySignal;
    uint64_t combineDoneSignal;
    uint64_t totalBytes;
};

struct HcclDeviceContext {
    uint64_t workSpace;
    uint64_t workSpaceSize;
    uint32_t rankId;
    uint32_t rankNum;
    uint64_t winSize;
    uint64_t windowsIn[kMaxDispatchCombineTileRanks];
    uint64_t windowsOut[kMaxDispatchCombineTileRanks];
};

struct DispatchCombineTileRuntimeConfig {
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
    uint32_t skipKernels;
    uint32_t hostGoldenOnly;
    uint32_t dispatchMetadataOnly;
    uint32_t dispatchOnly;
    uint32_t combineReturnOnly;
    uint32_t keepHcclShm;
    uint64_t hcclBuffSizeMb;
};

} // namespace dispatch_combine_tile

#endif // DISPATCH_COMBINE_TILE_COMMON_H_
