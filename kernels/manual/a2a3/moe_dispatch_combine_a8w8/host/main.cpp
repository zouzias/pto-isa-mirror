/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include "args.hpp"
#include "hccl_window.hpp"
#include "reference.hpp"
#include "workspace_layout.hpp"

#include <cstdlib>
#include <exception>
#include <iostream>
#include <string>

namespace moe_dispatch_combine_a8w8 {
namespace {

bool ReadEnvU32(const char *name, uint32_t *out)
{
    const char *value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return false;
    }
    try {
        *out = ParseU32(value, name);
        return true;
    } catch (const std::exception &) {
        return false;
    }
}

uint32_t DetectMpiRankFromEnv()
{
    uint32_t rank = 0;
    if (ReadEnvU32("OMPI_COMM_WORLD_RANK", &rank) || ReadEnvU32("PMI_RANK", &rank) ||
        ReadEnvU32("PMIX_RANK", &rank) || ReadEnvU32("MPI_RANKID", &rank)) {
        return rank;
    }
    return 0;
}

uint32_t DetectMpiSizeFromEnv(uint32_t fallback)
{
    uint32_t size = fallback;
    if (ReadEnvU32("OMPI_COMM_WORLD_SIZE", &size) || ReadEnvU32("PMI_SIZE", &size) ||
        ReadEnvU32("PMIX_SIZE", &size) || ReadEnvU32("MPI_LOCALNRANKS", &size)) {
        return size;
    }
    return fallback;
}

void ApplyRankSource(ProgramArgs *args)
{
    if (args->rank.rankFromMpi != 0) {
        args->rank.rankId = DetectMpiRankFromEnv();
        uint32_t mpiSize = DetectMpiSizeFromEnv(args->shape.rankNum);
        if (mpiSize != args->shape.rankNum) {
            std::cerr << "[WARN] MPI size " << mpiSize << " differs from shape.rankNum " << args->shape.rankNum
                      << "; shape.rankNum remains authoritative.\n";
        }
    }
}

} // namespace
} // namespace moe_dispatch_combine_a8w8

int main(int argc, char **argv)
{
    using namespace moe_dispatch_combine_a8w8;
    try {
        ProgramArgs args = ParseArgs(argc, argv);
        ApplyRankSource(&args);
        ValidateArgs(args);

        WorkspaceLayout workspaceLayout = MakeWorkspaceLayout(args.shape);
        PeerWindowLayout peerWindowLayout = MakePeerWindowLayout(args.shape);
        HcclWindowPlan windowPlan = MakeHcclWindowPlan(peerWindowLayout, args.runtime.hcclBuffSizeMb);
        CheckWindowSize(windowPlan);
        RankWindowBootstrap windowBootstrap = MakeRankWindowBootstrap(args.rank, windowPlan);

        if (args.rank.rankId == 0 || args.runtime.debug != 0) {
            PrintArgSummary(std::cout, args);
            PrintLayoutDump(std::cout, args.shape, args.rank, workspaceLayout, peerWindowLayout);
            PrintHcclWindowPlan(std::cout, windowPlan);
            PrintRankWindowBootstrap(std::cout, windowBootstrap);
            PrintM1MetadataDump(std::cout, args.runtime.caseName, args.runtime.seed, args.shape, args.rank);
        }

        CorrectnessReport correctness =
            BuildCorrectnessReport(args.runtime.caseName, args.runtime.seed, args.shape, args.rank);
        if (args.rank.rankId == 0 || args.runtime.debug != 0) {
            PrintCorrectnessReport(std::cout, correctness);
            PrintPerfReportSkeleton(std::cout, args.runtime.caseName, args.shape, args.rank, correctness.pass);
        }

        if (args.runtime.dryRun != 0 || args.runtime.skipKernelLaunch != 0) {
            std::cout << "[M1Smoke] rank=" << args.rank.rankId << " mode=dry_run kernel_launch=skipped\n";
            std::cout.flush();
            return correctness.pass ? 0 : 1;
        }

        std::cout << "[M1Smoke] rank=" << args.rank.rankId << " mode=runtime kernel_launch=not_implemented\n";
        std::cout.flush();
        return 0;
    } catch (const std::exception &ex) {
        std::cerr << "[ERROR] " << ex.what() << "\n";
        return 1;
    }
}
