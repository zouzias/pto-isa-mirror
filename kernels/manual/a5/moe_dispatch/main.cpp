/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include "args.h"
#include "golden.h"
#include "layout.h"

#include <cstdint>
#include <exception>
#include <iostream>
#include <numeric>

namespace {

int RunHostGolden(const moe_dispatch::MoeDispatchArgs &args)
{
    const moe_dispatch::MoeDispatchShape &shape = args.shape;
    auto data = moe_dispatch::GenerateHostData(shape, args.runtime.rank, args.runtime.seed);
    auto golden = moe_dispatch::BuildDispatchGolden(shape, data);
    const int64_t routedRows =
        std::accumulate(golden.tokenPerExpert.begin(), golden.tokenPerExpert.end(), static_cast<int64_t>(0));
    const int64_t expectedRows = static_cast<int64_t>(shape.m) * shape.topK;
    if (routedRows != expectedRows) {
        std::cerr << "[ERROR] golden routed rows mismatch, got " << routedRows << " expected " << expectedRows << '\n';
        return 1;
    }
    std::cout << "[HOST_GOLDEN] routed_rows=" << routedRows << " packed_elements=" << golden.packedA.size()
              << " expanded_row_idx=" << golden.expandedRowIdx.size() << '\n';
    return 0;
}

void DumpLayouts(const moe_dispatch::MoeDispatchArgs &args)
{
    const moe_dispatch::MoeDispatchShape &shape = args.shape;
    const auto workspace = moe_dispatch::ComputeWorkspaceLayout(shape);
    uint64_t hcclMb = args.runtime.hcclBuffSizeMb;
    if (hcclMb == 0) {
        hcclMb = moe_dispatch::EstimateHcclBuffSizeMb(shape);
    }
    const uint64_t windowBytes = hcclMb * moe_dispatch::kMoeDispatchMiB;
    const auto window = moe_dispatch::ComputeGuardedWindowLayout(shape, windowBytes);
    moe_dispatch::ValidateWindowLayout(shape, window);

    std::cout << "[LAYOUT] workspaceBytes=" << workspace.totalBytes << " hcclBuffSizeMb=" << hcclMb << '\n';
    std::cout << "[WINDOW] mode=GuardedDispatchOnly headGuardBytes=" << moe_dispatch::kMoeDispatchWindowHeadGuardBytes
              << " packedA=" << window.packedA << " expandedRowIdx=" << window.expandedRowIdx
              << " tokenPerExpert=" << window.tokenPerExpert << " signal=" << window.signal
              << " totalBytes=" << window.totalVisibleBytes << '\n';
}

} // namespace

int main(int argc, char **argv)
{
    try {
        moe_dispatch::MoeDispatchArgs args = moe_dispatch::ParseArgs(argc, argv);
        moe_dispatch::ValidateArgs(args);
        moe_dispatch::DumpArgs(args);
        DumpLayouts(args);
        const int goldenStatus = RunHostGolden(args);
        if (goldenStatus != 0) {
            return goldenStatus;
        }
        if (args.runtime.hostGoldenOnly != 0 || args.runtime.skipRun != 0) {
            std::cout << "[INFO] host-only path complete; A5 device runtime was not launched\n";
            return 0;
        }
        std::cout << "[INFO] A5 device runtime launch is scaffolded in this version; use --host-golden-only 1 for "
                     "local validation on A3 machines\n";
        return 0;
    } catch (const std::exception &ex) {
        std::cerr << "[ERROR] " << ex.what() << '\n';
        return 1;
    }
}
