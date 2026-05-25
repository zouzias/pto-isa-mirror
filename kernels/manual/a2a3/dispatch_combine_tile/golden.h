/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef DISPATCH_COMBINE_TILE_GOLDEN_H_
#define DISPATCH_COMBINE_TILE_GOLDEN_H_

#include "args.h"

#include <cstdint>
#include <vector>

namespace dispatch_combine_tile {

struct HostInputData {
    std::vector<float> inputA;
    std::vector<int32_t> expertIdx;
    std::vector<float> probs;
};

struct CpuGoldenData {
    std::vector<float> dispatchedA;
    std::vector<float> expertOutput;
    std::vector<float> outputC;
    std::vector<int32_t> peerTokenPerExpert;
    std::vector<int32_t> expandedRowIdx;
};

struct CompareResult {
    uint64_t elementCount;
    uint64_t mismatchCount;
    uint64_t firstMismatchIndex;
    float actual;
    float expected;
};

// Task 4 implements deterministic input generation/loading, CPU reference
// dispatch/combine, and rank-scoped output comparison.
HostInputData GenerateOrLoadInputs(const DispatchCombineTileArgs &args, uint32_t myRank);
CpuGoldenData ComputeCpuGolden(const DispatchCombineTileArgs &args, const HostInputData &inputs, uint32_t myRank);
CompareResult CompareOutputs(const DispatchCombineTileArgs &args, const CpuGoldenData &golden,
                             const std::vector<float> &actualOutputC, uint32_t myRank);

} // namespace dispatch_combine_tile

#endif // DISPATCH_COMBINE_TILE_GOLDEN_H_
