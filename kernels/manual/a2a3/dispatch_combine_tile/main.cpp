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

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "acl/acl.h"

namespace dispatch_combine_tile {

namespace {

void CheckAcl(aclError ret, const std::string &where)
{
    if (ret != ACL_SUCCESS) {
        throw std::runtime_error(where + " failed: " + std::to_string(static_cast<int>(ret)));
    }
}

size_t BytesOfHalfVector(size_t elements)
{
    return elements * sizeof(uint16_t);
}

size_t BytesOfFloatVector(size_t elements)
{
    return elements * sizeof(float);
}

size_t BytesOfI32Vector(size_t elements)
{
    return elements * sizeof(int32_t);
}

double UsSince(std::chrono::steady_clock::time_point start, std::chrono::steady_clock::time_point end)
{
    return std::chrono::duration<double, std::micro>(end - start).count();
}

} // namespace

struct DeviceBuffers {
    void *inputA = nullptr;
    void *expertIdx = nullptr;
    void *probs = nullptr;
    void *outputC = nullptr;
    void *workspace = nullptr;
    void *expertOutput = nullptr;
};

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
    HostInputData inputs;
    CpuGoldenData golden;
    bool dataReady = false;
    DeviceBuffers buffers;
    bool buffersAllocated = false;
    double dispatchE2eUs = 0.0;
    double prepareHostUs = 0.0;
    double combineE2eUs = 0.0;
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

void PrepareHostData(const DispatchCombineTileArgs &args, RuntimeState *state)
{
    PrintStage(state->rank, "host_data", "begin");
    if (args.runtime.genData != 0 && state->rank == 0) {
        GenerateAllInputFiles(args);
    }
    MpiBarrier(&state->mpi);
    state->inputs = GenerateOrLoadInputs(args, state->rank);
    state->golden = ComputeCpuGolden(args, state->inputs, state->rank);
    state->dataReady = true;
    std::cout << "rank=" << state->rank << " golden_total_routes=" << state->golden.totalRoutes
              << " golden_invalid_routes=" << state->golden.invalidRoutes << std::endl;
    PrintStage(state->rank, "host_data", "done");
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

void AllocateLocalBuffers(const DispatchCombineTileArgs &args, const WorkspaceLayout &workspaceLayout,
                          RuntimeState *state)
{
    PrintStage(state->rank, "allocate_buffers", "begin");
    const DispatchCombineTileShape &shape = args.shape;
    size_t inputBytes = BytesOfHalfVector(static_cast<size_t>(shape.m) * shape.k);
    size_t expertIdxBytes = BytesOfI32Vector(static_cast<size_t>(shape.m) * shape.topK);
    size_t probsBytes = BytesOfFloatVector(static_cast<size_t>(shape.m) * shape.topK);
    size_t outputBytes = BytesOfHalfVector(static_cast<size_t>(shape.m) * shape.k);
    size_t expertOutputBytes = BytesOfHalfVector(static_cast<size_t>(shape.maxOutputSize) * shape.k);
    CheckAcl(aclrtMalloc(&state->buffers.inputA, inputBytes, ACL_MEM_MALLOC_HUGE_FIRST),
             "rank " + std::to_string(state->rank) + " aclrtMalloc inputA");
    CheckAcl(aclrtMalloc(&state->buffers.expertIdx, expertIdxBytes, ACL_MEM_MALLOC_HUGE_FIRST),
             "rank " + std::to_string(state->rank) + " aclrtMalloc expertIdx");
    CheckAcl(aclrtMalloc(&state->buffers.probs, probsBytes, ACL_MEM_MALLOC_HUGE_FIRST),
             "rank " + std::to_string(state->rank) + " aclrtMalloc probs");
    CheckAcl(aclrtMalloc(&state->buffers.outputC, outputBytes, ACL_MEM_MALLOC_HUGE_FIRST),
             "rank " + std::to_string(state->rank) + " aclrtMalloc outputC");
    CheckAcl(aclrtMalloc(&state->buffers.workspace, workspaceLayout.totalBytes, ACL_MEM_MALLOC_HUGE_FIRST),
             "rank " + std::to_string(state->rank) + " aclrtMalloc workspace");
    CheckAcl(aclrtMalloc(&state->buffers.expertOutput, expertOutputBytes, ACL_MEM_MALLOC_HUGE_FIRST),
             "rank " + std::to_string(state->rank) + " aclrtMalloc expertOutput");
    state->buffersAllocated = true;
    PrintStage(state->rank, "allocate_buffers", "done");
}

void CopyInputsToDevice(const DispatchCombineTileArgs &args, RuntimeState *state)
{
    if (!state->dataReady) {
        throw std::runtime_error("host data is not ready before device copy");
    }
    PrintStage(state->rank, "copy_inputs", "begin");
    const DispatchCombineTileShape &shape = args.shape;
    std::vector<uint16_t> inputHalf = FloatVectorToHalfBits(state->inputs.inputA);
    size_t inputBytes = BytesOfHalfVector(inputHalf.size());
    size_t expertIdxBytes = BytesOfI32Vector(state->inputs.expertIdx.size());
    size_t probsBytes = BytesOfFloatVector(state->inputs.probs.size());
    CheckAcl(aclrtMemcpy(state->buffers.inputA, inputBytes, inputHalf.data(), inputBytes, ACL_MEMCPY_HOST_TO_DEVICE),
             "rank " + std::to_string(state->rank) + " copy inputA");
    CheckAcl(aclrtMemcpy(state->buffers.expertIdx, expertIdxBytes, state->inputs.expertIdx.data(), expertIdxBytes,
                         ACL_MEMCPY_HOST_TO_DEVICE),
             "rank " + std::to_string(state->rank) + " copy expertIdx");
    CheckAcl(aclrtMemcpy(state->buffers.probs, probsBytes, state->inputs.probs.data(), probsBytes,
                         ACL_MEMCPY_HOST_TO_DEVICE),
             "rank " + std::to_string(state->rank) + " copy probs");
    CheckAcl(aclrtMemset(state->buffers.outputC, BytesOfHalfVector(static_cast<size_t>(shape.m) * shape.k), 0,
                         BytesOfHalfVector(static_cast<size_t>(shape.m) * shape.k)),
             "rank " + std::to_string(state->rank) + " clear outputC");
    PrintStage(state->rank, "copy_inputs", "done");
}

void ClearDeviceState(const DispatchCombineTileArgs &args, const WorkspaceLayout &workspaceLayout,
                      const PeerWindowLayout &peerWindowLayout, RuntimeState *state)
{
    (void)args;
    PrintStage(state->rank, "clear_device_state", "begin");
    CheckAcl(aclrtMemset(state->buffers.workspace, workspaceLayout.totalBytes, 0, workspaceLayout.totalBytes),
             "rank " + std::to_string(state->rank) + " clear workspace");
    CheckAcl(aclrtMemset(state->hccl.peerWindow, peerWindowLayout.totalBytes, 0, peerWindowLayout.totalBytes),
             "rank " + std::to_string(state->rank) + " clear peerWindow");
    CheckAcl(aclrtSynchronizeStream(state->computeStream), "rank " + std::to_string(state->rank) + " sync clear");
    PrintStage(state->rank, "clear_device_state", "done");
}

struct DispatchMetadataDump {
    std::vector<int32_t> localTokenPerExpert;
    std::vector<int32_t> peerTokenPerExpert;
    std::vector<int32_t> cumsumPerExpert;
    std::vector<int32_t> dispatchOffset;
    std::vector<int32_t> prevSumBeforeRank;
    std::vector<int32_t> countReadySignal;
    std::vector<int32_t> expandedRowIdx;
    std::vector<float> packedA;
    std::vector<float> dispatchedA;
};

void CopyDispatchMetadataToHost(const DispatchCombineTileArgs &args, const WorkspaceLayout &workspaceLayout,
                                const PeerWindowLayout &peerWindowLayout, RuntimeState *state,
                                DispatchMetadataDump *dump)
{
    const DispatchCombineTileShape &shape = args.shape;
    size_t expertNumPadded = static_cast<size_t>(ExpertNumPadded(shape));
    dump->localTokenPerExpert.assign(expertNumPadded, 0);
    dump->peerTokenPerExpert.assign(static_cast<size_t>(shape.ep) * expertNumPadded, 0);
    dump->cumsumPerExpert.assign(static_cast<size_t>(shape.ep) * expertNumPadded, 0);
    dump->dispatchOffset.assign(shape.expertPerRank, 0);
    dump->prevSumBeforeRank.assign(static_cast<size_t>(shape.ep) * shape.expertPerRank, 0);
    dump->countReadySignal.assign(shape.ep, 0);
    dump->expandedRowIdx.assign(static_cast<size_t>(shape.m) * shape.topK, -1);
    dump->packedA.assign(static_cast<size_t>(shape.m) * shape.topK * shape.k, 0.0f);
    dump->dispatchedA.assign(static_cast<size_t>(shape.maxOutputSize) * shape.k, 0.0f);

    auto *workspaceBase = reinterpret_cast<uint8_t *>(state->buffers.workspace);
    auto *peerBase = reinterpret_cast<uint8_t *>(state->hccl.peerWindow);
    CheckAcl(aclrtMemcpy(dump->localTokenPerExpert.data(), BytesOfI32Vector(dump->localTokenPerExpert.size()),
                         workspaceBase + workspaceLayout.localTokenPerExpert,
                         BytesOfI32Vector(dump->localTokenPerExpert.size()), ACL_MEMCPY_DEVICE_TO_HOST),
             "rank " + std::to_string(state->rank) + " copy localTokenPerExpert");
    CheckAcl(aclrtMemcpy(dump->peerTokenPerExpert.data(), BytesOfI32Vector(dump->peerTokenPerExpert.size()),
                         peerBase + peerWindowLayout.peerTokenPerExpert,
                         BytesOfI32Vector(dump->peerTokenPerExpert.size()), ACL_MEMCPY_DEVICE_TO_HOST),
             "rank " + std::to_string(state->rank) + " copy peerTokenPerExpert");
    CheckAcl(aclrtMemcpy(dump->cumsumPerExpert.data(), BytesOfI32Vector(dump->cumsumPerExpert.size()),
                         workspaceBase + workspaceLayout.cumsumPerExpert,
                         BytesOfI32Vector(dump->cumsumPerExpert.size()), ACL_MEMCPY_DEVICE_TO_HOST),
             "rank " + std::to_string(state->rank) + " copy cumsumPerExpert");
    CheckAcl(aclrtMemcpy(dump->dispatchOffset.data(), BytesOfI32Vector(dump->dispatchOffset.size()),
                         workspaceBase + workspaceLayout.dispatchOffset,
                         BytesOfI32Vector(dump->dispatchOffset.size()), ACL_MEMCPY_DEVICE_TO_HOST),
             "rank " + std::to_string(state->rank) + " copy dispatchOffset");
    CheckAcl(aclrtMemcpy(dump->prevSumBeforeRank.data(), BytesOfI32Vector(dump->prevSumBeforeRank.size()),
                         workspaceBase + workspaceLayout.prevSumBeforeRank,
                         BytesOfI32Vector(dump->prevSumBeforeRank.size()), ACL_MEMCPY_DEVICE_TO_HOST),
             "rank " + std::to_string(state->rank) + " copy prevSumBeforeRank");
    CheckAcl(aclrtMemcpy(dump->countReadySignal.data(), BytesOfI32Vector(dump->countReadySignal.size()),
                         peerBase + peerWindowLayout.countReadySignal, BytesOfI32Vector(dump->countReadySignal.size()),
                         ACL_MEMCPY_DEVICE_TO_HOST),
             "rank " + std::to_string(state->rank) + " copy countReadySignal");
    CheckAcl(aclrtMemcpy(dump->expandedRowIdx.data(), BytesOfI32Vector(dump->expandedRowIdx.size()),
                         peerBase + peerWindowLayout.expandedRowIdx, BytesOfI32Vector(dump->expandedRowIdx.size()),
                         ACL_MEMCPY_DEVICE_TO_HOST),
             "rank " + std::to_string(state->rank) + " copy expandedRowIdx");
    std::vector<uint16_t> packedHalf(dump->packedA.size());
    std::vector<uint16_t> dispatchedHalf(dump->dispatchedA.size());
    CheckAcl(aclrtMemcpy(packedHalf.data(), BytesOfHalfVector(packedHalf.size()), peerBase + peerWindowLayout.packedA,
                         BytesOfHalfVector(packedHalf.size()), ACL_MEMCPY_DEVICE_TO_HOST),
             "rank " + std::to_string(state->rank) + " copy packedA");
    CheckAcl(aclrtMemcpy(dispatchedHalf.data(), BytesOfHalfVector(dispatchedHalf.size()),
                         workspaceBase + workspaceLayout.dispatchedA, BytesOfHalfVector(dispatchedHalf.size()),
                         ACL_MEMCPY_DEVICE_TO_HOST),
             "rank " + std::to_string(state->rank) + " copy dispatchedA");
    dump->packedA = HalfBitsToFloatVector(packedHalf);
    dump->dispatchedA = HalfBitsToFloatVector(dispatchedHalf);
}

std::vector<float> CopyDeviceHalfToFloat(void *devicePtr, size_t elementCount, uint32_t rank, const std::string &name)
{
    std::vector<uint16_t> halfData(elementCount);
    CheckAcl(aclrtMemcpy(halfData.data(), BytesOfHalfVector(halfData.size()), devicePtr,
                         BytesOfHalfVector(halfData.size()), ACL_MEMCPY_DEVICE_TO_HOST),
             "rank " + std::to_string(rank) + " copy " + name);
    return HalfBitsToFloatVector(halfData);
}

uint64_t CompareI32Buffer(const std::string &name, const std::vector<int32_t> &actual,
                          const std::vector<int32_t> &expected, uint32_t rank)
{
    uint64_t mismatches = 0;
    size_t elementCount = actual.size() < expected.size() ? actual.size() : expected.size();
    size_t firstMismatch = elementCount;
    int32_t firstActual = 0;
    int32_t firstExpected = 0;
    if (actual.size() != expected.size()) {
        mismatches = actual.size() > expected.size() ? actual.size() - expected.size() : expected.size() - actual.size();
        firstMismatch = elementCount;
    }
    for (size_t i = 0; i < elementCount; ++i) {
        if (actual[i] != expected[i]) {
            if (firstMismatch == elementCount) {
                firstMismatch = i;
                firstActual = actual[i];
                firstExpected = expected[i];
            }
            ++mismatches;
        }
    }
    std::cout << "rank=" << rank << " buffer=" << name << " elements=" << actual.size()
              << " mismatches=" << mismatches;
    if (mismatches != 0) {
        std::cout << " first_index=" << firstMismatch << " actual=" << firstActual << " expected=" << firstExpected;
    }
    std::cout << std::endl;
    return mismatches;
}

uint64_t CompareFloatBuffer(const DispatchCombineTileArgs &args, const std::string &name,
                            const std::vector<float> &actual, const std::vector<float> &expected, uint32_t rank)
{
    uint64_t mismatches = 0;
    size_t elementCount = actual.size() < expected.size() ? actual.size() : expected.size();
    size_t firstMismatch = elementCount;
    float firstActual = 0.0f;
    float firstExpected = 0.0f;
    if (actual.size() != expected.size()) {
        mismatches = actual.size() > expected.size() ? actual.size() - expected.size() : expected.size() - actual.size();
        firstMismatch = elementCount;
    }
    for (size_t i = 0; i < elementCount; ++i) {
        float actualValue = actual[i];
        float expectedValue = expected[i];
        float diff = std::fabs(actualValue - expectedValue);
        float tol = static_cast<float>(args.atol + args.rtol * std::fabs(expectedValue));
        if (diff > tol) {
            if (firstMismatch == elementCount) {
                firstMismatch = i;
                firstActual = actualValue;
                firstExpected = expectedValue;
            }
            ++mismatches;
        }
    }
    std::cout << "rank=" << rank << " buffer=" << name << " elements=" << actual.size()
              << " mismatches=" << mismatches;
    if (mismatches != 0) {
        std::cout << " first_index=" << firstMismatch << " actual=" << firstActual << " expected=" << firstExpected;
    }
    std::cout << std::endl;
    return mismatches;
}

uint64_t VerifyDispatchMetadata(const DispatchCombineTileArgs &args, RuntimeState *state,
                                const DispatchMetadataDump &dump)
{
    uint64_t mismatches = 0;
    mismatches += CompareI32Buffer("localTokenPerExpert", dump.localTokenPerExpert,
                                   state->golden.localTokenPerExpert, state->rank);
    mismatches += CompareI32Buffer("peerTokenPerExpert", dump.peerTokenPerExpert,
                                   state->golden.peerTokenPerExpert, state->rank);
    mismatches += CompareI32Buffer("cumsumPerExpert", dump.cumsumPerExpert, state->golden.cumsumPerExpert,
                                   state->rank);
    mismatches += CompareI32Buffer("dispatchOffset", dump.dispatchOffset, state->golden.dispatchOffset, state->rank);
    mismatches += CompareI32Buffer("prevSumBeforeRank", dump.prevSumBeforeRank, state->golden.prevSumBeforeRank,
                                   state->rank);
    if (args.runtime.debug != 0) {
        std::cout << "rank=" << state->rank << " count_ready_signal=";
        for (size_t i = 0; i < dump.countReadySignal.size(); ++i) {
            if (i != 0) {
                std::cout << ",";
            }
            std::cout << dump.countReadySignal[i];
        }
        std::cout << std::endl;
        WriteBinaryFile(RankBinaryFile(args, state->rank, "actual_localTokenPerExpert"), dump.localTokenPerExpert);
        WriteBinaryFile(RankBinaryFile(args, state->rank, "actual_peerTokenPerExpert"), dump.peerTokenPerExpert);
        WriteBinaryFile(RankBinaryFile(args, state->rank, "actual_cumsumPerExpert"), dump.cumsumPerExpert);
        WriteBinaryFile(RankBinaryFile(args, state->rank, "actual_dispatchOffset"), dump.dispatchOffset);
        WriteBinaryFile(RankBinaryFile(args, state->rank, "actual_prevSumBeforeRank"), dump.prevSumBeforeRank);
        WriteBinaryFile(RankBinaryFile(args, state->rank, "actual_expandedRowIdx"), dump.expandedRowIdx);
        WriteBinaryFile(RankBinaryFile(args, state->rank, "actual_packedA_head"), FloatVectorToHalfBits(dump.packedA));
        WriteBinaryFile(RankBinaryFile(args, state->rank, "actual_dispatchedA_head"),
                        FloatVectorToHalfBits(dump.dispatchedA));
    }
    return mismatches;
}

uint64_t VerifyDispatchPayload(const DispatchCombineTileArgs &args, RuntimeState *state,
                               const DispatchMetadataDump &dump)
{
    uint64_t mismatches = 0;
    mismatches += CompareI32Buffer("expandedRowIdx", dump.expandedRowIdx, state->golden.expandedRowIdx, state->rank);
    mismatches += CompareFloatBuffer(args, "packedA", dump.packedA, state->golden.packedA, state->rank);
    mismatches += CompareFloatBuffer(args, "dispatchedA", dump.dispatchedA, state->golden.dispatchedA, state->rank);
    return mismatches;
}

void PrintDispatchGatherSegments(const DispatchCombineTileArgs &args, RuntimeState *state)
{
    if (args.runtime.debug < 2) {
        return;
    }
    const DispatchCombineTileShape &shape = args.shape;
    size_t expertNumPadded = ExpertNumPadded(shape);
    for (uint32_t localExpert = 0; localExpert < shape.expertPerRank; ++localExpert) {
        uint32_t globalExpert = state->rank * shape.expertPerRank + localExpert;
        for (uint32_t src = 0; src < shape.ep; ++src) {
            int32_t rows =
                state->golden.peerTokenPerExpert[static_cast<size_t>(src) * expertNumPadded + globalExpert];
            if (rows <= 0) {
                continue;
            }
            int32_t dstStart =
                state->golden.dispatchOffset[localExpert] +
                state->golden.prevSumBeforeRank[static_cast<size_t>(src) * shape.expertPerRank + localExpert];
            std::cout << "rank=" << state->rank << " dispatch_gather local_expert=" << localExpert
                      << " src=" << src << " rows=" << rows << " dst_start=" << dstStart << std::endl;
        }
    }
}

void RunDispatch(const DispatchCombineTileArgs &args, const WorkspaceLayout &workspaceLayout,
                 const PeerWindowLayout &peerWindowLayout, RuntimeState *state)
{
    PrintStage(state->rank, "dispatch", "begin");
    uint32_t launchBlocks = args.shape.aivBlocks == 0 ? 1 : args.shape.aivBlocks;
    MpiBarrier(&state->mpi);
    auto dispatchStart = std::chrono::steady_clock::now();
    LaunchDispatchCombineTileDispatch(args.shape, state->rank, reinterpret_cast<uint8_t *>(state->buffers.inputA),
                                      reinterpret_cast<uint8_t *>(state->buffers.expertIdx),
                                      reinterpret_cast<uint8_t *>(state->hccl.peerWindow),
                                      reinterpret_cast<uint8_t *>(state->hccl.deviceContext),
                                      reinterpret_cast<uint8_t *>(state->buffers.workspace), state->computeStream,
                                      launchBlocks);
    CheckAcl(aclrtSynchronizeStream(state->computeStream),
             "rank " + std::to_string(state->rank) + " dispatch stream sync");
    MpiBarrier(&state->mpi);
    auto dispatchEnd = std::chrono::steady_clock::now();
    state->dispatchE2eUs = UsSince(dispatchStart, dispatchEnd);
    std::cout << "rank=" << state->rank << " dispatch_e2e_us=" << state->dispatchE2eUs << std::endl;

    DispatchMetadataDump dump;
    CopyDispatchMetadataToHost(args, workspaceLayout, peerWindowLayout, state, &dump);
    if (args.runtime.debug >= 2 && !dump.packedA.empty() && !state->golden.packedA.empty()) {
        std::cout << "rank=" << state->rank << " packedA_sample actual=" << dump.packedA[0]
                  << "," << dump.packedA[1] << "," << dump.packedA[2] << "," << dump.packedA[3]
                  << " expected=" << state->golden.packedA[0] << "," << state->golden.packedA[1] << ","
                  << state->golden.packedA[2] << "," << state->golden.packedA[3] << std::endl;
    }
    uint64_t localTotal = 0;
    for (int32_t count : dump.localTokenPerExpert) {
        localTotal += static_cast<uint32_t>(count);
    }
    uint32_t ownerRows = state->rank < state->golden.ownerRows.size() ? state->golden.ownerRows[state->rank] : 0;
    std::cout << "rank=" << state->rank << " dispatch_counts local_total=" << localTotal
              << " owner_rows=" << ownerRows << std::endl;
    std::cout << "rank=" << state->rank << " count_wait_done peers=" << args.shape.ep << std::endl;
    uint64_t metadataMismatches = VerifyDispatchMetadata(args, state, dump);
    std::cout << "rank=" << state->rank << " dispatch_metadata_mismatches=" << metadataMismatches << std::endl;
    if (metadataMismatches != 0) {
        throw std::runtime_error("rank " + std::to_string(state->rank) + " dispatch metadata mismatch");
    }
    std::cout << "rank=" << state->rank << " dispatch_metadata_done" << std::endl;
    if (args.runtime.dispatchOnly != 0) {
        PrintDispatchGatherSegments(args, state);
        uint64_t payloadMismatches = VerifyDispatchPayload(args, state, dump);
        std::cout << "rank=" << state->rank << " dispatch_payload_mismatches=" << payloadMismatches << std::endl;
        if (payloadMismatches != 0) {
            throw std::runtime_error("rank " + std::to_string(state->rank) + " dispatch payload mismatch");
        }
        std::cout << "rank=" << state->rank << " dispatch_pack_done" << std::endl;
        std::cout << "rank=" << state->rank << " dispatch_gather_done" << std::endl;
    }
    PrintStage(state->rank, "dispatch", "done");
}

void PrepareExpertOutputIdentity(const DispatchCombineTileArgs &args, const WorkspaceLayout &workspaceLayout,
                                 RuntimeState *state)
{
    PrintStage(state->rank, "prepare_expert_output", "begin");
    const DispatchCombineTileShape &shape = args.shape;
    size_t elements = static_cast<size_t>(shape.maxOutputSize) * shape.k;
    size_t bytes = BytesOfHalfVector(elements);
    auto *workspaceBase = reinterpret_cast<uint8_t *>(state->buffers.workspace);
    auto prepareStart = std::chrono::steady_clock::now();
    CheckAcl(aclrtMemcpy(state->buffers.expertOutput, bytes, workspaceBase + workspaceLayout.dispatchedA, bytes,
                         ACL_MEMCPY_DEVICE_TO_DEVICE),
             "rank " + std::to_string(state->rank) + " prepare expertOutput");
    auto prepareEnd = std::chrono::steady_clock::now();
    state->prepareHostUs = UsSince(prepareStart, prepareEnd);
    std::cout << "rank=" << state->rank << " prepare_host_us=" << state->prepareHostUs << std::endl;

    if (args.runtime.debug >= 2) {
        std::vector<float> dispatched =
            CopyDeviceHalfToFloat(workspaceBase + workspaceLayout.dispatchedA, elements, state->rank, "dispatchedA");
        std::vector<float> expertOutput =
            CopyDeviceHalfToFloat(state->buffers.expertOutput, elements, state->rank, "expertOutput");
        uint64_t copyMismatches =
            CompareFloatBuffer(args, "expertOutput_vs_dispatchedA", expertOutput, dispatched, state->rank);
        uint64_t goldenMismatches =
            CompareFloatBuffer(args, "expertOutput", expertOutput, state->golden.dispatchedA, state->rank);
        if (copyMismatches != 0 || goldenMismatches != 0) {
            throw std::runtime_error("rank " + std::to_string(state->rank) + " expertOutput prepare mismatch");
        }
    }
    MpiBarrier(&state->mpi);
    PrintStage(state->rank, "prepare_expert_output", "done");
}

struct CombineReturnDump {
    std::vector<float> ptrD;
    std::vector<int32_t> combineDoneSignal;
};

void CopyCombineReturnToHost(const DispatchCombineTileArgs &args, const PeerWindowLayout &peerWindowLayout,
                             RuntimeState *state, CombineReturnDump *dump)
{
    const DispatchCombineTileShape &shape = args.shape;
    size_t expandedRows = static_cast<size_t>(shape.m) * shape.topK;
    dump->ptrD.assign(expandedRows * shape.k, 0.0f);
    dump->combineDoneSignal.assign(shape.ep, 0);
    auto *peerBase = reinterpret_cast<uint8_t *>(state->hccl.peerWindow);
    std::vector<uint16_t> ptrDHalf(dump->ptrD.size());
    CheckAcl(aclrtMemcpy(ptrDHalf.data(), BytesOfHalfVector(ptrDHalf.size()), peerBase + peerWindowLayout.ptrD,
                         BytesOfHalfVector(ptrDHalf.size()), ACL_MEMCPY_DEVICE_TO_HOST),
             "rank " + std::to_string(state->rank) + " copy ptrD");
    CheckAcl(aclrtMemcpy(dump->combineDoneSignal.data(), BytesOfI32Vector(dump->combineDoneSignal.size()),
                         peerBase + peerWindowLayout.combineDoneSignal,
                         BytesOfI32Vector(dump->combineDoneSignal.size()), ACL_MEMCPY_DEVICE_TO_HOST),
             "rank " + std::to_string(state->rank) + " copy combineDoneSignal");
    dump->ptrD = HalfBitsToFloatVector(ptrDHalf);
}

void ClearCombineReturnState(const DispatchCombineTileArgs &args, const PeerWindowLayout &peerWindowLayout,
                             RuntimeState *state)
{
    PrintStage(state->rank, "clear_combine_state", "begin");
    const DispatchCombineTileShape &shape = args.shape;
    size_t expandedRows = static_cast<size_t>(shape.m) * shape.topK;
    auto *peerBase = reinterpret_cast<uint8_t *>(state->hccl.peerWindow);
    CheckAcl(aclrtMemset(peerBase + peerWindowLayout.ptrD, BytesOfHalfVector(expandedRows * shape.k), 0,
                         BytesOfHalfVector(expandedRows * shape.k)),
             "rank " + std::to_string(state->rank) + " clear ptrD");
    CheckAcl(aclrtMemset(peerBase + peerWindowLayout.combineDoneSignal, BytesOfI32Vector(shape.ep), 0,
                         BytesOfI32Vector(shape.ep)),
             "rank " + std::to_string(state->rank) + " clear combineDoneSignal");
    CheckAcl(aclrtMemset(state->buffers.outputC, BytesOfHalfVector(static_cast<size_t>(shape.m) * shape.k), 0,
                         BytesOfHalfVector(static_cast<size_t>(shape.m) * shape.k)),
             "rank " + std::to_string(state->rank) + " clear combine outputC");
    PrintStage(state->rank, "clear_combine_state", "done");
}

void PrintCombineReturnSegments(const DispatchCombineTileArgs &args, RuntimeState *state)
{
    if (args.runtime.debug < 2) {
        return;
    }
    const DispatchCombineTileShape &shape = args.shape;
    size_t expertNumPadded = ExpertNumPadded(shape);
    for (uint32_t dst = 0; dst < shape.ep; ++dst) {
        for (uint32_t localExpert = 0; localExpert < shape.expertPerRank; ++localExpert) {
            uint32_t globalExpert = state->rank * shape.expertPerRank + localExpert;
            int32_t rows =
                state->golden.peerTokenPerExpert[static_cast<size_t>(dst) * expertNumPadded + globalExpert];
            if (rows <= 0) {
                continue;
            }
            int32_t srcStart =
                state->golden.dispatchOffset[localExpert] +
                state->golden.prevSumBeforeRank[static_cast<size_t>(dst) * shape.expertPerRank + localExpert];
            int32_t dstStart = globalExpert == 0 ?
                                   0 :
                                   state->golden.cumsumPerExpert[static_cast<size_t>(dst) * expertNumPadded +
                                                                 globalExpert - 1];
            std::cout << "rank=" << state->rank << " combine_return dst=" << dst
                      << " local_expert=" << localExpert << " rows=" << rows << " src_start=" << srcStart
                      << " dst_start=" << dstStart << std::endl;
        }
    }
}

void RunCombine(const DispatchCombineTileArgs &args, const PeerWindowLayout &peerWindowLayout, RuntimeState *state)
{
    PrintStage(state->rank, "combine", "begin");
    ClearCombineReturnState(args, peerWindowLayout, state);
    PrintCombineReturnSegments(args, state);
    uint32_t launchBlocks = args.shape.aivBlocks == 0 ? 1 : args.shape.aivBlocks;
    MpiBarrier(&state->mpi);
    auto combineStart = std::chrono::steady_clock::now();
    LaunchDispatchCombineTileCombine(args.shape, state->rank, reinterpret_cast<uint8_t *>(state->buffers.expertOutput),
                                     reinterpret_cast<uint8_t *>(state->buffers.probs),
                                     reinterpret_cast<uint8_t *>(state->buffers.outputC),
                                     reinterpret_cast<uint8_t *>(state->hccl.peerWindow),
                                     reinterpret_cast<uint8_t *>(state->hccl.deviceContext),
                                     reinterpret_cast<uint8_t *>(state->buffers.workspace), state->computeStream,
                                     launchBlocks);
    CheckAcl(aclrtSynchronizeStream(state->computeStream),
             "rank " + std::to_string(state->rank) + " combine stream sync");
    MpiBarrier(&state->mpi);
    auto combineEnd = std::chrono::steady_clock::now();
    state->combineE2eUs = UsSince(combineStart, combineEnd);

    if (args.runtime.combineReturnOnly != 0) {
        CombineReturnDump dump;
        CopyCombineReturnToHost(args, peerWindowLayout, state, &dump);
        if (args.runtime.debug != 0) {
            std::cout << "rank=" << state->rank << " combine_done_signal=";
            for (size_t i = 0; i < dump.combineDoneSignal.size(); ++i) {
                if (i != 0) {
                    std::cout << ",";
                }
                std::cout << dump.combineDoneSignal[i];
            }
            std::cout << std::endl;
            WriteBinaryFile(RankBinaryFile(args, state->rank, "actual_ptrD_head"), FloatVectorToHalfBits(dump.ptrD));
        }
        uint64_t ptrDMismatches = CompareFloatBuffer(args, "ptrD", dump.ptrD, state->golden.ptrD, state->rank);
        if (ptrDMismatches != 0) {
            throw std::runtime_error("rank " + std::to_string(state->rank) + " combine return mismatch");
        }
        std::cout << "rank=" << state->rank << " combine_return_done" << std::endl;
        std::cout << "rank=" << state->rank << " combine_wait_done peers=" << args.shape.ep << std::endl;
    }
    PrintStage(state->rank, "combine", "done");
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
    if (state->buffersAllocated) {
        if (state->buffers.inputA != nullptr) {
            aclrtFree(state->buffers.inputA);
        }
        if (state->buffers.expertIdx != nullptr) {
            aclrtFree(state->buffers.expertIdx);
        }
        if (state->buffers.probs != nullptr) {
            aclrtFree(state->buffers.probs);
        }
        if (state->buffers.outputC != nullptr) {
            aclrtFree(state->buffers.outputC);
        }
        if (state->buffers.workspace != nullptr) {
            aclrtFree(state->buffers.workspace);
        }
        if (state->buffers.expertOutput != nullptr) {
            aclrtFree(state->buffers.expertOutput);
        }
        state->buffers = DeviceBuffers{};
        state->buffersAllocated = false;
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

        dispatch_combine_tile::PrepareHostData(args, &state);

        dispatch_combine_tile::BindDeviceContinuous(args, &state);
        dispatch_combine_tile::CreateStreams(&state);
        dispatch_combine_tile::InitHccl(&state, args, peerWindowLayout);
        dispatch_combine_tile::AllocateLocalBuffers(args, workspaceLayout, &state);
        dispatch_combine_tile::CopyInputsToDevice(args, &state);

        if (args.runtime.skipKernels != 0) {
            dispatch_combine_tile::MpiBarrier(&state.mpi);
            std::cout << "rank=" << state.rank << " skip_kernels_done" << std::endl;
            dispatch_combine_tile::Cleanup(&state);
            return 0;
        }

        dispatch_combine_tile::ClearDeviceState(args, workspaceLayout, peerWindowLayout, &state);
        dispatch_combine_tile::RunDispatch(args, workspaceLayout, peerWindowLayout, &state);
        if (args.runtime.dispatchMetadataOnly != 0) {
            dispatch_combine_tile::MpiBarrier(&state.mpi);
            std::cout << "rank=" << state.rank << " run_done" << std::endl;
            dispatch_combine_tile::Cleanup(&state);
            return 0;
        }
        dispatch_combine_tile::PrepareExpertOutputIdentity(args, workspaceLayout, &state);
        if (args.runtime.dispatchOnly != 0) {
            dispatch_combine_tile::MpiBarrier(&state.mpi);
            std::cout << "rank=" << state.rank << " run_done" << std::endl;
            dispatch_combine_tile::Cleanup(&state);
            return 0;
        }
        dispatch_combine_tile::RunCombine(args, peerWindowLayout, &state);
        if (args.runtime.combineReturnOnly != 0) {
            dispatch_combine_tile::MpiBarrier(&state.mpi);
            std::cout << "rank=" << state.rank << " run_done" << std::endl;
            dispatch_combine_tile::Cleanup(&state);
            return 0;
        }
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
