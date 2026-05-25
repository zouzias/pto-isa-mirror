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
#include "comm_mpi.h"
#include "golden.h"
#include "hccl_context.h"
#include "kernel_launchers.h"
#include "layout.h"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

#include "acl/acl.h"

namespace dispatch_combine_tile {

struct RuntimeState {
    MpiContext mpi;
    bool mpiActive = false;
    bool aclActive = false;
    uint32_t rank = 0;
    uint32_t size = 1;
    uint32_t device = 0;
    aclrtStream computeStream = nullptr;
    rtStream_t hcclStream = nullptr;
    HcclWindowContext hccl;
    bool hcclActive = false;
};

void PrintStage(uint32_t rank, const char *stage, const char *state)
{
    std::cout << "rank=" << rank << " stage=" << stage << " " << state << std::endl;
}

void InitRankInfo(const DispatchCombineTileArgs &args, int *argc, char ***argv, RuntimeState *state)
{
    if (args.runtime.rankFromMpi != 0) {
        state->mpi = InitMpiAndRank(argc, argv);
        state->mpiActive = true;
        state->rank = static_cast<uint32_t>(state->mpi.rank);
        state->size = static_cast<uint32_t>(state->mpi.size);
    } else {
        state->rank = args.runtime.rank;
        state->size = args.runtime.nranks;
    }

    if (state->size != args.shape.ep) {
        throw std::runtime_error("rank size mismatch: runtime size=" + std::to_string(state->size) +
                                 " shape EP=" + std::to_string(args.shape.ep));
    }
    if (state->rank >= state->size) {
        throw std::runtime_error("rank is outside rank size");
    }
    state->device = args.runtime.deviceBase + state->rank;
}

void RunHostGoldenOnly(const DispatchCombineTileArgs &args, RuntimeState *state)
{
    PrintStage(state->rank, "host_golden", "begin");
    if (args.runtime.genData != 0 && state->rank == 0) {
        GenerateAllInputFiles(args);
    }
    MpiBarrier(&state->mpi);

    HostInputData inputs = GenerateOrLoadInputs(args, state->rank);
    CpuGoldenData golden = ComputeCpuGolden(args, inputs, state->rank);
    CompareResult compare = CompareOutputs(args, golden, golden.outputC, state->rank);

    std::cout << "rank=" << state->rank << " golden_owner_rows=";
    for (uint32_t rank = 0; rank < golden.ownerRows.size(); ++rank) {
        if (rank != 0) {
            std::cout << ",";
        }
        std::cout << rank << ":" << golden.ownerRows[rank];
    }
    std::cout << "\n";
    std::cout << "rank=" << state->rank << " golden_total_routes=" << golden.totalRoutes
              << " golden_invalid_routes=" << golden.invalidRoutes << " compare_elements=" << compare.elementCount
              << " compare_mismatches=" << compare.mismatchCount << "\n";
    std::cout << "rank=" << state->rank << " golden_rank_done" << std::endl;
    MpiBarrier(&state->mpi);
}

void BindDeviceContinuous(const DispatchCombineTileArgs &args, RuntimeState *state)
{
    (void)args;
    PrintStage(state->rank, "bind_device", "begin");
    if (!InitAclAndBindDevice(state->rank, state->device)) {
        throw std::runtime_error("rank " + std::to_string(state->rank) + " failed to bind device " +
                                 std::to_string(state->device));
    }
    state->aclActive = true;
    PrintStage(state->rank, "bind_device", "done");
}

void CreateStreams(RuntimeState *state)
{
    PrintStage(state->rank, "create_streams", "begin");
    if (aclrtCreateStream(&state->computeStream) != ACL_SUCCESS) {
        throw std::runtime_error("rank " + std::to_string(state->rank) + " aclrtCreateStream failed");
    }
    if (rtStreamCreate(&state->hcclStream, kRtStreamPriorityDefault) != 0) {
        throw std::runtime_error("rank " + std::to_string(state->rank) + " rtStreamCreate failed");
    }
    PrintStage(state->rank, "create_streams", "done");
}

void InitHccl(RuntimeState *state, const DispatchCombineTileArgs &args, const PeerWindowLayout &peerWindowLayout)
{
    PrintStage(state->rank, "hccl_root_info", "begin");
    HcclRootInfo rootInfo{};
    if (state->rank == 0 && !InitHcclRootInfo(&rootInfo)) {
        throw std::runtime_error("rank 0 failed to get HCCL root info");
    }
    MpiBroadcast(&state->mpi, &rootInfo, HCCL_ROOT_INFO_BYTES, 0);
    MpiBarrier(&state->mpi);
    PrintStage(state->rank, "hccl_root_info", "done");

    PrintStage(state->rank, "hccl_window", "begin");
    state->hccl =
        InitHcclWindowContext(args.shape, peerWindowLayout, state->rank, state->size, &rootInfo, state->hcclStream);
    state->hcclActive = true;
    std::cout << "rank=" << state->rank << " size=" << state->size << " device=" << state->device
              << " window_base=" << state->hccl.hostDeviceContext.windowsIn[state->rank]
              << " peer_window=" << reinterpret_cast<uint64_t>(state->hccl.peerWindow)
              << " win_size=" << state->hccl.hostDeviceContext.winSize << std::endl;
    PrintStage(state->rank, "hccl_window", "done");
}

void RunDispatch(const DispatchCombineTileArgs &args, uint32_t myRank)
{
    (void)args;
    PrintStage(myRank, "dispatch", "skipped");
}

void PrepareExpertOutputIdentity(const DispatchCombineTileArgs &args, uint32_t myRank)
{
    (void)args;
    PrintStage(myRank, "prepare_expert_output", "skipped");
}

void RunCombine(const DispatchCombineTileArgs &args, uint32_t myRank)
{
    (void)args;
    PrintStage(myRank, "combine", "skipped");
}

void VerifyAndDump(const DispatchCombineTileArgs &args, uint32_t myRank)
{
    (void)args;
    PrintStage(myRank, "verify", "skipped");
}

void Cleanup(RuntimeState *state)
{
    if (state == nullptr) {
        return;
    }
    if (state->mpiActive) {
        try {
            MpiBarrier(&state->mpi);
        } catch (const std::exception &ex) {
            std::cerr << "[WARN] rank=" << state->rank << " cleanup MPI barrier failed: " << ex.what() << "\n";
        }
    }
    if (state->hcclActive) {
        DestroyHcclWindowContext(&state->hccl);
        state->hcclActive = false;
    }
    if (state->hcclStream != nullptr) {
        rtStreamDestroy(state->hcclStream);
        state->hcclStream = nullptr;
    }
    if (state->computeStream != nullptr) {
        aclrtDestroyStream(state->computeStream);
        state->computeStream = nullptr;
    }
    if (state->aclActive) {
        aclrtResetDevice(static_cast<int32_t>(state->device));
        aclFinalize();
        state->aclActive = false;
    }
    if (state->mpiActive) {
        FinalizeMpi(&state->mpi);
        state->mpiActive = false;
    }
}

} // namespace dispatch_combine_tile

int main(int argc, char **argv)
{
    dispatch_combine_tile::RuntimeState state;
    try {
        dispatch_combine_tile::DispatchCombineTileArgs args = dispatch_combine_tile::ParseArgs(argc, argv);
        dispatch_combine_tile::ValidateArgs(args);
        dispatch_combine_tile::WorkspaceLayout workspaceLayout =
            dispatch_combine_tile::ComputeWorkspaceLayout(args.shape);
        dispatch_combine_tile::PeerWindowLayout peerWindowLayout =
            dispatch_combine_tile::ComputePeerWindowLayout(args.shape);
        uint64_t hcclBuffSizeMb = args.runtime.hcclBuffSizeMb == 0 ?
                                      dispatch_combine_tile::EstimateHcclBuffSizeMb(args.shape, peerWindowLayout) :
                                      args.runtime.hcclBuffSizeMb;
        dispatch_combine_tile::PrintRunSummary(args);
        std::cout << "workspace_bytes=" << workspaceLayout.totalBytes << "\n";
        std::cout << "peer_window_bytes=" << peerWindowLayout.totalBytes << "\n";
        std::cout << "HCCL_BUFFSIZE=" << hcclBuffSizeMb << std::endl;

        dispatch_combine_tile::InitRankInfo(args, &argc, &argv, &state);
        std::cout << "rank=" << state.rank << " size=" << state.size << " device=" << state.device << " start"
                  << std::endl;

        if (args.runtime.hostGoldenOnly != 0) {
            dispatch_combine_tile::RunHostGoldenOnly(args, &state);
            dispatch_combine_tile::Cleanup(&state);
            return 0;
        }

        dispatch_combine_tile::BindDeviceContinuous(args, &state);
        dispatch_combine_tile::CreateStreams(&state);
        dispatch_combine_tile::InitHccl(&state, args, peerWindowLayout);

        if (args.runtime.skipKernels != 0) {
            dispatch_combine_tile::MpiBarrier(&state.mpi);
            std::cout << "rank=" << state.rank << " skip_kernels_done" << std::endl;
            dispatch_combine_tile::Cleanup(&state);
            return 0;
        }

        dispatch_combine_tile::RunDispatch(args, state.rank);
        dispatch_combine_tile::PrepareExpertOutputIdentity(args, state.rank);
        dispatch_combine_tile::RunCombine(args, state.rank);
        dispatch_combine_tile::VerifyAndDump(args, state.rank);
        dispatch_combine_tile::MpiBarrier(&state.mpi);
        std::cout << "rank=" << state.rank << " run_done" << std::endl;
        dispatch_combine_tile::Cleanup(&state);
        return 0;
    } catch (const std::exception &ex) {
        std::cerr << "[ERROR] rank=" << state.rank << " " << ex.what() << std::endl;
        dispatch_combine_tile::Cleanup(&state);
        return 1;
    }
}
