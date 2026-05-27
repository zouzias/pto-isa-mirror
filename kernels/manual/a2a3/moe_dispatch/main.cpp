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

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "acl/acl.h"

namespace moe_dispatch {

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

size_t BytesOfI32Vector(size_t elements)
{
    return elements * sizeof(int32_t);
}

} // namespace

struct PerfStats {
    double avg = 0.0;
    double min = 0.0;
    double max = 0.0;
    double stddev = 0.0;
};

struct IterationTiming {
    double dispatchKernelUs = 0.0;
};

PerfStats CalcStats(const std::vector<double> &samples)
{
    PerfStats stats;
    if (samples.empty()) {
        return stats;
    }

    stats.min = *std::min_element(samples.begin(), samples.end());
    stats.max = *std::max_element(samples.begin(), samples.end());
    stats.avg = std::accumulate(samples.begin(), samples.end(), 0.0) / static_cast<double>(samples.size());
    double variance = 0.0;
    for (double sample : samples) {
        double delta = sample - stats.avg;
        variance += delta * delta;
    }
    stats.stddev = std::sqrt(variance / static_cast<double>(samples.size()));
    return stats;
}

std::vector<double> ExtractTimingSamples(const std::vector<IterationTiming> &timings, double IterationTiming::*field)
{
    std::vector<double> samples;
    samples.reserve(timings.size());
    for (const IterationTiming &timing : timings) {
        samples.push_back(timing.*field);
    }
    return samples;
}

void PrintOneTimingStats(const char *label, const std::vector<IterationTiming> &timings, double IterationTiming::*field)
{
    PerfStats stats = CalcStats(ExtractTimingSamples(timings, field));
    std::cout << "  " << label << ": avg=" << stats.avg << " us"
              << " max=" << stats.max << " us" << std::endl;
}

bool VerboseRuntimeLogs(const MoeDispatchArgs &args)
{
    return args.runtime.debug != 0 || args.runtime.hostGoldenOnly != 0 || args.runtime.skipKernels != 0 ||
           args.runtime.dispatchMetadataOnly != 0 || args.runtime.dispatchOnly != 0;
}

struct DeviceBuffers {
    void *inputA = nullptr;
    void *expertIdx = nullptr;
    void *workspace = nullptr;
};

struct RuntimeState {
    MpiContext mpi;
    bool mpiActive = false;
    bool aclActive = false;
    uint32_t rank = 0;
    uint32_t size = 1;
    uint32_t device = 0;
    aclrtStream computeStream = nullptr;
    aclrtEvent dispatchStartEvent = nullptr;
    aclrtEvent dispatchEndEvent = nullptr;
    rtStream_t hcclStream = nullptr;
    HcclWindowContext hccl;
    bool hcclActive = false;
    HostInputData inputs;
    CpuGoldenData golden;
    bool dataReady = false;
    DeviceBuffers buffers;
    bool buffersAllocated = false;
    double dispatchKernelUs = 0.0;
    uint32_t currentSignalValue = 1;
};

void PrintProfileSummary(const MoeDispatchArgs &args, RuntimeState *state,
                         const std::vector<IterationTiming> &localTimings)
{
    if (args.runtime.iters == 0) {
        return;
    }

    std::vector<IterationTiming> allTimings;
    size_t measuredIters = localTimings.size();
    if (state->rank == 0) {
        allTimings.resize(static_cast<size_t>(state->size) * localTimings.size());
    }
    MpiGatherBytes(&state->mpi, localTimings.data(), localTimings.size() * sizeof(IterationTiming),
                   state->rank == 0 ? allTimings.data() : nullptr, 0);

    if (state->rank != 0) {
        return;
    }

    std::vector<IterationTiming> globalTimings(measuredIters);
    for (size_t iter = 0; iter < measuredIters; ++iter) {
        IterationTiming timing;
        for (uint32_t rank = 0; rank < state->size; ++rank) {
            const IterationTiming &rankTiming = allTimings[static_cast<size_t>(rank) * measuredIters + iter];
            timing.dispatchKernelUs = std::max(timing.dispatchKernelUs, rankTiming.dispatchKernelUs);
        }
        globalTimings[iter] = timing;
    }

    std::cout << std::fixed << std::setprecision(1);
    std::cout << "\n================================================================" << std::endl;
    std::cout << "[PROFILE] MoeDispatch" << std::endl;
    std::cout << "  M=" << args.shape.m << " K=" << args.shape.k << " ranks=" << args.shape.ep
              << " topK=" << args.shape.topK << " expertPerPe=" << args.shape.expertPerRank
              << " warmup=" << args.runtime.warmup << " measured=" << args.runtime.iters
              << " samples=" << globalTimings.size() << std::endl;
    std::cout << "  logical work: input tokens(all ranks)=" << static_cast<uint64_t>(args.shape.ep) * args.shape.m
              << " routed tokens(all ranks)=" << static_cast<uint64_t>(args.shape.ep) * args.shape.m * args.shape.topK
              << std::endl;
    PrintOneTimingStats("dispatch_kernel", globalTimings, &IterationTiming::dispatchKernelUs);
    std::cout << "  verify=" << (args.runtime.verify == 0 ? "SKIP" : "PASS") << std::endl;
    std::cout << "  note: warmup iterations are excluded; kernel samples use device events and max across ranks."
              << std::endl;
    std::cout << "================================================================\n" << std::endl;
}

void PrintStage(uint32_t rank, const char *stage, const char *state)
{
    std::cout << "rank=" << rank << " stage=" << stage << " " << state << std::endl;
}

void InitRankInfo(const MoeDispatchArgs &args, int *argc, char ***argv, RuntimeState *state)
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

void RunHostGoldenOnly(const MoeDispatchArgs &args, RuntimeState *state)
{
    PrintStage(state->rank, "host_golden", "begin");
    if (args.runtime.genData != 0 && state->rank == 0) {
        GenerateAllInputFiles(args);
    }
    MpiBarrier(&state->mpi);

    HostInputData inputs = GenerateOrLoadInputs(args, state->rank);
    CpuGoldenData golden = ComputeCpuGolden(args, inputs, state->rank);

    std::cout << "rank=" << state->rank << " golden_owner_rows=";
    for (uint32_t rank = 0; rank < golden.ownerRows.size(); ++rank) {
        if (rank != 0) {
            std::cout << ",";
        }
        std::cout << rank << ":" << golden.ownerRows[rank];
    }
    std::cout << "\n";
    std::cout << "rank=" << state->rank << " golden_total_routes=" << golden.totalRoutes
              << " golden_invalid_routes=" << golden.invalidRoutes << "\n";
    std::cout << "rank=" << state->rank << " golden_rank_done" << std::endl;
    MpiBarrier(&state->mpi);
}

void PrepareHostData(const MoeDispatchArgs &args, RuntimeState *state)
{
    bool verbose = VerboseRuntimeLogs(args);
    if (verbose) {
        PrintStage(state->rank, "host_data", "begin");
    }
    if (args.runtime.genData != 0 && state->rank == 0) {
        GenerateAllInputFiles(args);
    }
    MpiBarrier(&state->mpi);
    state->inputs = GenerateOrLoadInputs(args, state->rank);
    state->golden = ComputeCpuGolden(args, state->inputs, state->rank);
    state->dataReady = true;
    if (verbose) {
        std::cout << "rank=" << state->rank << " golden_total_routes=" << state->golden.totalRoutes
                  << " golden_invalid_routes=" << state->golden.invalidRoutes << std::endl;
        PrintStage(state->rank, "host_data", "done");
    }
}

void BindDeviceContinuous(const MoeDispatchArgs &args, RuntimeState *state)
{
    bool verbose = VerboseRuntimeLogs(args);
    if (verbose) {
        PrintStage(state->rank, "bind_device", "begin");
    }
    if (!InitAclAndBindDevice(state->rank, state->device)) {
        throw std::runtime_error("rank " + std::to_string(state->rank) + " failed to bind device " +
                                 std::to_string(state->device));
    }
    state->aclActive = true;
    if (verbose) {
        PrintStage(state->rank, "bind_device", "done");
    }
}

void CreateStreams(const MoeDispatchArgs &args, RuntimeState *state)
{
    bool verbose = VerboseRuntimeLogs(args);
    if (verbose) {
        PrintStage(state->rank, "create_streams", "begin");
    }
    if (aclrtCreateStream(&state->computeStream) != ACL_SUCCESS) {
        throw std::runtime_error("rank " + std::to_string(state->rank) + " aclrtCreateStream failed");
    }
    CheckAcl(aclrtCreateEvent(&state->dispatchStartEvent),
             "rank " + std::to_string(state->rank) + " aclrtCreateEvent dispatch start");
    CheckAcl(aclrtCreateEvent(&state->dispatchEndEvent),
             "rank " + std::to_string(state->rank) + " aclrtCreateEvent dispatch end");
    if (rtStreamCreate(&state->hcclStream, kRtStreamPriorityDefault) != 0) {
        throw std::runtime_error("rank " + std::to_string(state->rank) + " rtStreamCreate failed");
    }
    if (verbose) {
        PrintStage(state->rank, "create_streams", "done");
    }
}

void InitHccl(RuntimeState *state, const MoeDispatchArgs &args, const PeerWindowLayout &peerWindowLayout)
{
    bool verbose = VerboseRuntimeLogs(args);
    if (verbose) {
        PrintStage(state->rank, "hccl_root_info", "begin");
    }
    HcclRootInfo rootInfo{};
    if (state->rank == 0 && !InitHcclRootInfo(&rootInfo)) {
        throw std::runtime_error("rank 0 failed to get HCCL root info");
    }
    MpiBroadcast(&state->mpi, &rootInfo, HCCL_ROOT_INFO_BYTES, 0);
    MpiBarrier(&state->mpi);
    if (verbose) {
        PrintStage(state->rank, "hccl_root_info", "done");
    }

    if (verbose) {
        PrintStage(state->rank, "hccl_window", "begin");
    }
    state->hccl =
        InitHcclWindowContext(args.shape, peerWindowLayout, state->rank, state->size, &rootInfo, state->hcclStream);
    state->hcclActive = true;
    if (verbose) {
        std::cout << "rank=" << state->rank << " size=" << state->size << " device=" << state->device
                  << " window_base=" << state->hccl.hostDeviceContext.windowsIn[state->rank]
                  << " peer_window=" << reinterpret_cast<uint64_t>(state->hccl.peerWindow)
                  << " win_size=" << state->hccl.hostDeviceContext.winSize << std::endl;
        PrintStage(state->rank, "hccl_window", "done");
    }
}

void AllocateLocalBuffers(const MoeDispatchArgs &args, const WorkspaceLayout &workspaceLayout, RuntimeState *state)
{
    bool verbose = VerboseRuntimeLogs(args);
    if (verbose) {
        PrintStage(state->rank, "allocate_buffers", "begin");
    }
    const MoeDispatchShape &shape = args.shape;
    size_t inputBytes = BytesOfHalfVector(static_cast<size_t>(shape.m) * shape.k);
    size_t expertIdxBytes = BytesOfI32Vector(static_cast<size_t>(shape.m) * shape.topK);
    CheckAcl(aclrtMalloc(&state->buffers.inputA, inputBytes, ACL_MEM_MALLOC_HUGE_FIRST),
             "rank " + std::to_string(state->rank) + " aclrtMalloc inputA");
    CheckAcl(aclrtMalloc(&state->buffers.expertIdx, expertIdxBytes, ACL_MEM_MALLOC_HUGE_FIRST),
             "rank " + std::to_string(state->rank) + " aclrtMalloc expertIdx");
    CheckAcl(aclrtMalloc(&state->buffers.workspace, workspaceLayout.totalBytes, ACL_MEM_MALLOC_HUGE_FIRST),
             "rank " + std::to_string(state->rank) + " aclrtMalloc workspace");
    state->buffersAllocated = true;
    if (verbose) {
        PrintStage(state->rank, "allocate_buffers", "done");
    }
}

void CopyInputsToDevice(const MoeDispatchArgs &args, RuntimeState *state)
{
    if (!state->dataReady) {
        throw std::runtime_error("host data is not ready before device copy");
    }
    bool verbose = VerboseRuntimeLogs(args);
    if (verbose) {
        PrintStage(state->rank, "copy_inputs", "begin");
    }
    std::vector<uint16_t> inputHalf = FloatVectorToHalfBits(state->inputs.inputA);
    size_t inputBytes = BytesOfHalfVector(inputHalf.size());
    size_t expertIdxBytes = BytesOfI32Vector(state->inputs.expertIdx.size());
    CheckAcl(aclrtMemcpy(state->buffers.inputA, inputBytes, inputHalf.data(), inputBytes, ACL_MEMCPY_HOST_TO_DEVICE),
             "rank " + std::to_string(state->rank) + " copy inputA");
    CheckAcl(aclrtMemcpy(state->buffers.expertIdx, expertIdxBytes, state->inputs.expertIdx.data(), expertIdxBytes,
                         ACL_MEMCPY_HOST_TO_DEVICE),
             "rank " + std::to_string(state->rank) + " copy expertIdx");
    if (verbose) {
        PrintStage(state->rank, "copy_inputs", "done");
    }
}

void ClearDeviceState(const MoeDispatchArgs &args, const WorkspaceLayout &workspaceLayout,
                      const PeerWindowLayout &peerWindowLayout, RuntimeState *state)
{
    bool verbose = VerboseRuntimeLogs(args);
    if (verbose) {
        PrintStage(state->rank, "clear_device_state", "begin");
    }
    CheckAcl(aclrtMemset(state->buffers.workspace, workspaceLayout.totalBytes, 0, workspaceLayout.totalBytes),
             "rank " + std::to_string(state->rank) + " clear workspace");
    size_t peerWindowDataBytes = state->currentSignalValue == 1 ?
                                     static_cast<size_t>(peerWindowLayout.totalBytes) :
                                     static_cast<size_t>(peerWindowLayout.countReadySignal);
    CheckAcl(aclrtMemset(state->hccl.peerWindow, peerWindowDataBytes, 0, peerWindowDataBytes),
             "rank " + std::to_string(state->rank) + " clear peerWindow data");
    CheckAcl(aclrtSynchronizeStream(state->computeStream), "rank " + std::to_string(state->rank) + " sync clear");
    if (verbose) {
        PrintStage(state->rank, "clear_device_state", "done");
    }
}

struct DispatchMetadataDump {
    std::vector<int32_t> localTokenPerExpert;
    std::vector<int32_t> blockTokenPerExpert;
    std::vector<int32_t> blockPrefixPerExpert;
    std::vector<int32_t> peerTokenPerExpert;
    std::vector<int32_t> cumsumPerExpert;
    std::vector<int32_t> dispatchOffset;
    std::vector<int32_t> prevSumBeforeRank;
    std::vector<int32_t> countReadySignal;
    std::vector<int32_t> expandedRowIdx;
    std::vector<float> packedA;
    std::vector<float> dispatchedA;
};

void CopyDispatchMetadataToHost(const MoeDispatchArgs &args, const WorkspaceLayout &workspaceLayout,
                                const PeerWindowLayout &peerWindowLayout, RuntimeState *state,
                                DispatchMetadataDump *dump)
{
    const MoeDispatchShape &shape = args.shape;
    size_t expertNumPadded = static_cast<size_t>(ExpertNumPadded(shape));
    size_t aivBlocks = static_cast<size_t>(EffectiveAivBlocks(shape));
    dump->localTokenPerExpert.assign(expertNumPadded, 0);
    dump->blockTokenPerExpert.assign(aivBlocks * expertNumPadded, 0);
    dump->blockPrefixPerExpert.assign(aivBlocks * expertNumPadded, 0);
    dump->peerTokenPerExpert.assign(static_cast<size_t>(shape.ep) * expertNumPadded, 0);
    dump->cumsumPerExpert.assign(static_cast<size_t>(shape.ep) * expertNumPadded, 0);
    dump->dispatchOffset.assign(shape.expertPerRank, 0);
    dump->prevSumBeforeRank.assign(static_cast<size_t>(shape.ep) * shape.expertPerRank, 0);
    dump->countReadySignal.assign(shape.ep, 0);
    dump->expandedRowIdx.assign(static_cast<size_t>(shape.m) * shape.topK, -1);
    if (args.runtime.debug >= 2 || args.runtime.verify != 0 || args.runtime.dispatchOnly != 0) {
        dump->packedA.assign(static_cast<size_t>(shape.m) * shape.topK * shape.k, 0.0f);
        dump->dispatchedA.assign(static_cast<size_t>(shape.maxOutputSize) * shape.k, 0.0f);
    }

    auto *workspaceBase = reinterpret_cast<uint8_t *>(state->buffers.workspace);
    auto *peerBase = reinterpret_cast<uint8_t *>(state->hccl.peerWindow);
    CheckAcl(aclrtMemcpy(dump->localTokenPerExpert.data(), BytesOfI32Vector(dump->localTokenPerExpert.size()),
                         workspaceBase + workspaceLayout.localTokenPerExpert,
                         BytesOfI32Vector(dump->localTokenPerExpert.size()), ACL_MEMCPY_DEVICE_TO_HOST),
             "rank " + std::to_string(state->rank) + " copy localTokenPerExpert");
    CheckAcl(aclrtMemcpy(dump->blockTokenPerExpert.data(), BytesOfI32Vector(dump->blockTokenPerExpert.size()),
                         workspaceBase + workspaceLayout.blockTokenPerExpert,
                         BytesOfI32Vector(dump->blockTokenPerExpert.size()), ACL_MEMCPY_DEVICE_TO_HOST),
             "rank " + std::to_string(state->rank) + " copy blockTokenPerExpert");
    CheckAcl(aclrtMemcpy(dump->blockPrefixPerExpert.data(), BytesOfI32Vector(dump->blockPrefixPerExpert.size()),
                         workspaceBase + workspaceLayout.blockPrefixPerExpert,
                         BytesOfI32Vector(dump->blockPrefixPerExpert.size()), ACL_MEMCPY_DEVICE_TO_HOST),
             "rank " + std::to_string(state->rank) + " copy blockPrefixPerExpert");
    CheckAcl(aclrtMemcpy(dump->peerTokenPerExpert.data(), BytesOfI32Vector(dump->peerTokenPerExpert.size()),
                         peerBase + peerWindowLayout.peerTokenPerExpert,
                         BytesOfI32Vector(dump->peerTokenPerExpert.size()), ACL_MEMCPY_DEVICE_TO_HOST),
             "rank " + std::to_string(state->rank) + " copy peerTokenPerExpert");
    CheckAcl(aclrtMemcpy(dump->cumsumPerExpert.data(), BytesOfI32Vector(dump->cumsumPerExpert.size()),
                         workspaceBase + workspaceLayout.cumsumPerExpert,
                         BytesOfI32Vector(dump->cumsumPerExpert.size()), ACL_MEMCPY_DEVICE_TO_HOST),
             "rank " + std::to_string(state->rank) + " copy cumsumPerExpert");
    CheckAcl(aclrtMemcpy(dump->dispatchOffset.data(), BytesOfI32Vector(dump->dispatchOffset.size()),
                         workspaceBase + workspaceLayout.dispatchOffset, BytesOfI32Vector(dump->dispatchOffset.size()),
                         ACL_MEMCPY_DEVICE_TO_HOST),
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
    if (!dump->packedA.empty()) {
        std::vector<uint16_t> packedHalf(dump->packedA.size());
        CheckAcl(
            aclrtMemcpy(packedHalf.data(), BytesOfHalfVector(packedHalf.size()), peerBase + peerWindowLayout.packedA,
                        BytesOfHalfVector(packedHalf.size()), ACL_MEMCPY_DEVICE_TO_HOST),
            "rank " + std::to_string(state->rank) + " copy packedA");
        dump->packedA = HalfBitsToFloatVector(packedHalf);
    }
    if (!dump->dispatchedA.empty()) {
        std::vector<uint16_t> dispatchedHalf(dump->dispatchedA.size());
        CheckAcl(aclrtMemcpy(dispatchedHalf.data(), BytesOfHalfVector(dispatchedHalf.size()),
                             workspaceBase + workspaceLayout.dispatchedA, BytesOfHalfVector(dispatchedHalf.size()),
                             ACL_MEMCPY_DEVICE_TO_HOST),
                 "rank " + std::to_string(state->rank) + " copy dispatchedA");
        dump->dispatchedA = HalfBitsToFloatVector(dispatchedHalf);
    }
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
        mismatches =
            actual.size() > expected.size() ? actual.size() - expected.size() : expected.size() - actual.size();
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
    std::cout << "rank=" << rank << " buffer=" << name << " elements=" << actual.size() << " mismatches=" << mismatches;
    if (mismatches != 0) {
        std::cout << " first_index=" << firstMismatch << " actual=" << firstActual << " expected=" << firstExpected;
    }
    std::cout << std::endl;
    return mismatches;
}

uint64_t CompareFloatBuffer(const MoeDispatchArgs &args, const std::string &name, const std::vector<float> &actual,
                            const std::vector<float> &expected, uint32_t rank)
{
    uint64_t mismatches = 0;
    size_t elementCount = actual.size() < expected.size() ? actual.size() : expected.size();
    size_t firstMismatch = elementCount;
    float firstActual = 0.0f;
    float firstExpected = 0.0f;
    if (actual.size() != expected.size()) {
        mismatches =
            actual.size() > expected.size() ? actual.size() - expected.size() : expected.size() - actual.size();
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
    std::cout << "rank=" << rank << " buffer=" << name << " elements=" << actual.size() << " mismatches=" << mismatches;
    if (mismatches != 0) {
        std::cout << " first_index=" << firstMismatch << " actual=" << firstActual << " expected=" << firstExpected;
    }
    std::cout << std::endl;
    return mismatches;
}

uint64_t VerifyDispatchMetadata(const MoeDispatchArgs &args, RuntimeState *state, const DispatchMetadataDump &dump)
{
    uint64_t mismatches = 0;
    mismatches += CompareI32Buffer("localTokenPerExpert", dump.localTokenPerExpert, state->golden.localTokenPerExpert,
                                   state->rank);
    if (args.runtime.debug >= 2) {
        const MoeDispatchShape &shape = args.shape;
        size_t expertNumPadded = static_cast<size_t>(ExpertNumPadded(shape));
        size_t aivBlocks = static_cast<size_t>(EffectiveAivBlocks(shape));
        std::vector<int32_t> expectedBlockToken(aivBlocks * expertNumPadded, 0);
        std::vector<int32_t> expectedBlockPrefix(aivBlocks * expertNumPadded, 0);
        constexpr uint32_t kI32PerCacheLine = 16;
        size_t routeCount = static_cast<size_t>(shape.m) * shape.topK;
        size_t routeLineCount = (routeCount + kI32PerCacheLine - 1) / kI32PerCacheLine;
        auto shardBegin = [](size_t totalItems, size_t block, size_t blockCount) {
            size_t base = totalItems / blockCount;
            size_t rem = totalItems % blockCount;
            return block * base + (block < rem ? block : rem);
        };
        for (uint32_t block = 0; block < aivBlocks; ++block) {
            size_t lineBegin = shardBegin(routeLineCount, block, aivBlocks);
            size_t lineEnd = shardBegin(routeLineCount, block + 1, aivBlocks);
            size_t routeBegin = lineBegin * kI32PerCacheLine;
            size_t routeEnd = lineEnd * kI32PerCacheLine;
            if (routeEnd > routeCount) {
                routeEnd = routeCount;
            }
            for (size_t routeIndex = routeBegin; routeIndex < routeEnd; ++routeIndex) {
                int32_t expert = state->inputs.expertIdx[routeIndex];
                if (expert < 0 || static_cast<uint32_t>(expert) >= shape.expertNum) {
                    continue;
                }
                ++expectedBlockToken[static_cast<size_t>(block) * expertNumPadded + static_cast<uint32_t>(expert)];
            }
        }
        for (uint32_t expert = 0; expert < shape.expertNum; ++expert) {
            int32_t sum = 0;
            for (uint32_t block = 0; block < aivBlocks; ++block) {
                size_t index = static_cast<size_t>(block) * expertNumPadded + expert;
                expectedBlockPrefix[index] = sum;
                sum += expectedBlockToken[index];
            }
        }
        mismatches +=
            CompareI32Buffer("blockTokenPerExpert", dump.blockTokenPerExpert, expectedBlockToken, state->rank);
        mismatches +=
            CompareI32Buffer("blockPrefixPerExpert", dump.blockPrefixPerExpert, expectedBlockPrefix, state->rank);
    }
    mismatches +=
        CompareI32Buffer("peerTokenPerExpert", dump.peerTokenPerExpert, state->golden.peerTokenPerExpert, state->rank);
    mismatches += CompareI32Buffer("cumsumPerExpert", dump.cumsumPerExpert, state->golden.cumsumPerExpert, state->rank);
    mismatches += CompareI32Buffer("dispatchOffset", dump.dispatchOffset, state->golden.dispatchOffset, state->rank);
    mismatches +=
        CompareI32Buffer("prevSumBeforeRank", dump.prevSumBeforeRank, state->golden.prevSumBeforeRank, state->rank);
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
        if (args.runtime.debug >= 2) {
            WriteBinaryFile(RankBinaryFile(args, state->rank, "actual_blockTokenPerExpert"), dump.blockTokenPerExpert);
            WriteBinaryFile(RankBinaryFile(args, state->rank, "actual_blockPrefixPerExpert"),
                            dump.blockPrefixPerExpert);
        }
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

uint64_t VerifyDispatchPayload(const MoeDispatchArgs &args, RuntimeState *state, const DispatchMetadataDump &dump)
{
    uint64_t mismatches = 0;
    mismatches += CompareI32Buffer("expandedRowIdx", dump.expandedRowIdx, state->golden.expandedRowIdx, state->rank);
    mismatches += CompareFloatBuffer(args, "packedA", dump.packedA, state->golden.packedA, state->rank);
    mismatches += CompareFloatBuffer(args, "dispatchedA", dump.dispatchedA, state->golden.dispatchedA, state->rank);
    return mismatches;
}

void PrintDispatchGatherSegments(const MoeDispatchArgs &args, RuntimeState *state)
{
    if (args.runtime.debug < 2) {
        return;
    }
    const MoeDispatchShape &shape = args.shape;
    size_t expertNumPadded = ExpertNumPadded(shape);
    for (uint32_t localExpert = 0; localExpert < shape.expertPerRank; ++localExpert) {
        uint32_t globalExpert = state->rank * shape.expertPerRank + localExpert;
        for (uint32_t src = 0; src < shape.ep; ++src) {
            int32_t rows = state->golden.peerTokenPerExpert[static_cast<size_t>(src) * expertNumPadded + globalExpert];
            if (rows <= 0) {
                continue;
            }
            int32_t dstStart =
                state->golden.dispatchOffset[localExpert] +
                state->golden.prevSumBeforeRank[static_cast<size_t>(src) * shape.expertPerRank + localExpert];
            std::cout << "rank=" << state->rank << " dispatch_gather local_expert=" << localExpert << " src=" << src
                      << " rows=" << rows << " dst_start=" << dstStart << std::endl;
        }
    }
}

void RunDispatch(const MoeDispatchArgs &args, const WorkspaceLayout &workspaceLayout,
                 const PeerWindowLayout &peerWindowLayout, RuntimeState *state)
{
    bool verbose = VerboseRuntimeLogs(args);
    if (verbose) {
        PrintStage(state->rank, "dispatch", "begin");
    }
    uint32_t launchBlocks = args.shape.aivBlocks == 0 ? 1 : args.shape.aivBlocks;
    MoeDispatchShape launchShape = args.shape;
    launchShape.signalValue = state->currentSignalValue;
    MpiBarrier(&state->mpi);
    CheckAcl(aclrtRecordEvent(state->dispatchStartEvent, state->computeStream),
             "rank " + std::to_string(state->rank) + " record dispatch start");
    LaunchMoeDispatchKernel(launchShape, state->rank, reinterpret_cast<uint8_t *>(state->buffers.inputA),
                            reinterpret_cast<uint8_t *>(state->buffers.expertIdx),
                            reinterpret_cast<uint8_t *>(state->hccl.peerWindow),
                            reinterpret_cast<uint8_t *>(state->hccl.deviceContext),
                            reinterpret_cast<uint8_t *>(state->buffers.workspace), state->computeStream, launchBlocks);
    CheckAcl(aclrtRecordEvent(state->dispatchEndEvent, state->computeStream),
             "rank " + std::to_string(state->rank) + " record dispatch end");
    CheckAcl(aclrtSynchronizeStream(state->computeStream),
             "rank " + std::to_string(state->rank) + " dispatch stream sync");
    MpiBarrier(&state->mpi);
    float kernelMs = 0.0f;
    CheckAcl(aclrtEventElapsedTime(&kernelMs, state->dispatchStartEvent, state->dispatchEndEvent),
             "rank " + std::to_string(state->rank) + " dispatch kernel elapsed time");
    state->dispatchKernelUs = static_cast<double>(kernelMs) * 1000.0;

    bool inspectDispatch = args.runtime.verify != 0 || args.runtime.debug != 0 ||
                           args.runtime.dispatchMetadataOnly != 0 || args.runtime.dispatchOnly != 0;
    if (inspectDispatch) {
        DispatchMetadataDump dump;
        CopyDispatchMetadataToHost(args, workspaceLayout, peerWindowLayout, state, &dump);
        if (args.runtime.debug >= 2 && !dump.packedA.empty() && !state->golden.packedA.empty()) {
            std::cout << "rank=" << state->rank << " packedA_sample actual=" << dump.packedA[0] << ","
                      << dump.packedA[1] << "," << dump.packedA[2] << "," << dump.packedA[3]
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
        if (args.runtime.dispatchMetadataOnly != 0) {
            if (verbose) {
                PrintStage(state->rank, "dispatch", "done");
            }
            return;
        }
        if (args.runtime.verify != 0 || args.runtime.debug >= 2 || args.runtime.dispatchOnly != 0) {
            PrintDispatchGatherSegments(args, state);
            uint64_t payloadMismatches = VerifyDispatchPayload(args, state, dump);
            std::cout << "rank=" << state->rank << " dispatch_payload_mismatches=" << payloadMismatches << std::endl;
            if (payloadMismatches != 0) {
                throw std::runtime_error("rank " + std::to_string(state->rank) + " dispatch payload mismatch");
            }
            std::cout << "rank=" << state->rank << " dispatch_pack_done" << std::endl;
            std::cout << "rank=" << state->rank << " dispatch_gather_done" << std::endl;
        }
    }
    if (verbose) {
        PrintStage(state->rank, "dispatch", "done");
    }
}

void PrintPerformanceConfig(const MoeDispatchArgs &args, const WorkspaceLayout &workspaceLayout,
                            const PeerWindowLayout &peerWindowLayout, uint32_t rank)
{
    if (!VerboseRuntimeLogs(args)) {
        return;
    }
    uint32_t aivBlocks = args.shape.aivBlocks == 0 ? 1 : args.shape.aivBlocks;
    uint32_t peerShards = args.shape.ep < aivBlocks ? args.shape.ep : aivBlocks;
    std::cout << "rank=" << rank << " aiv_blocks=" << aivBlocks << " peer_window_bytes=" << peerWindowLayout.totalBytes
              << " workspace_bytes=" << workspaceLayout.totalBytes << " dispatch_peer_shards=" << peerShards
              << std::endl;
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
        if (state->buffers.workspace != nullptr) {
            aclrtFree(state->buffers.workspace);
        }
        state->buffers = DeviceBuffers{};
        state->buffersAllocated = false;
    }
    if (state->hcclStream != nullptr) {
        rtStreamDestroy(state->hcclStream);
        state->hcclStream = nullptr;
    }
    if (state->dispatchEndEvent != nullptr) {
        aclrtDestroyEvent(state->dispatchEndEvent);
        state->dispatchEndEvent = nullptr;
    }
    if (state->dispatchStartEvent != nullptr) {
        aclrtDestroyEvent(state->dispatchStartEvent);
        state->dispatchStartEvent = nullptr;
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

} // namespace moe_dispatch

int main(int argc, char **argv)
{
    moe_dispatch::RuntimeState state;
    try {
        moe_dispatch::MoeDispatchArgs args = moe_dispatch::ParseArgs(argc, argv);
        moe_dispatch::ValidateArgs(args);
        moe_dispatch::WorkspaceLayout workspaceLayout = moe_dispatch::ComputeWorkspaceLayout(args.shape);
        moe_dispatch::PeerWindowLayout peerWindowLayout = moe_dispatch::ComputePeerWindowLayout(args.shape);
        uint64_t hcclBuffSizeMb = args.runtime.hcclBuffSizeMb == 0 ?
                                      moe_dispatch::EstimateHcclBuffSizeMb(args.shape, peerWindowLayout) :
                                      args.runtime.hcclBuffSizeMb;
        moe_dispatch::PrintRunSummary(args);
        std::cout << "workspace_bytes=" << workspaceLayout.totalBytes << "\n";
        std::cout << "peer_window_bytes=" << peerWindowLayout.totalBytes << "\n";
        std::cout << "HCCL_BUFFSIZE=" << hcclBuffSizeMb << std::endl;

        moe_dispatch::InitRankInfo(args, &argc, &argv, &state);
        bool verbose = moe_dispatch::VerboseRuntimeLogs(args);
        if (verbose) {
            std::cout << "rank=" << state.rank << " size=" << state.size << " device=" << state.device << " start"
                      << std::endl;
        }

        if (args.runtime.hostGoldenOnly != 0) {
            moe_dispatch::RunHostGoldenOnly(args, &state);
            moe_dispatch::Cleanup(&state);
            return 0;
        }

        moe_dispatch::PrepareHostData(args, &state);

        moe_dispatch::BindDeviceContinuous(args, &state);
        moe_dispatch::CreateStreams(args, &state);
        moe_dispatch::InitHccl(&state, args, peerWindowLayout);
        moe_dispatch::AllocateLocalBuffers(args, workspaceLayout, &state);
        moe_dispatch::CopyInputsToDevice(args, &state);

        if (args.runtime.skipKernels != 0) {
            moe_dispatch::MpiBarrier(&state.mpi);
            std::cout << "rank=" << state.rank << " skip_kernels_done" << std::endl;
            moe_dispatch::Cleanup(&state);
            return 0;
        }

        moe_dispatch::PrintPerformanceConfig(args, workspaceLayout, peerWindowLayout, state.rank);
        uint32_t totalIterations = args.runtime.warmup + args.runtime.iters;
        std::vector<moe_dispatch::IterationTiming> measureTimings;
        measureTimings.reserve(args.runtime.iters);
        for (uint32_t iter = 0; iter < totalIterations; ++iter) {
            bool isWarmup = iter < args.runtime.warmup;
            state.currentSignalValue = iter + 1;
            if (verbose && !isWarmup) {
                std::cout << "rank=" << state.rank << " iteration=" << (iter - args.runtime.warmup)
                          << " phase=measure begin" << std::endl;
            }
            moe_dispatch::ClearDeviceState(args, workspaceLayout, peerWindowLayout, &state);
            moe_dispatch::RunDispatch(args, workspaceLayout, peerWindowLayout, &state);
            if (args.runtime.dispatchMetadataOnly != 0) {
                moe_dispatch::MpiBarrier(&state.mpi);
                std::cout << "rank=" << state.rank << " run_done" << std::endl;
                moe_dispatch::Cleanup(&state);
                return 0;
            }
            if (!isWarmup) {
                moe_dispatch::IterationTiming timing;
                timing.dispatchKernelUs = state.dispatchKernelUs;
                measureTimings.push_back(timing);
                if (verbose) {
                    std::cout << "rank=" << state.rank << " iteration=" << (iter - args.runtime.warmup)
                              << " phase=measure done" << std::endl;
                }
            }
        }
        moe_dispatch::PrintProfileSummary(args, &state, measureTimings);
        moe_dispatch::MpiBarrier(&state.mpi);
        if (verbose) {
            std::cout << "rank=" << state.rank << " run_done" << std::endl;
        }
        moe_dispatch::Cleanup(&state);
        return 0;
    } catch (const std::exception &ex) {
        std::cerr << "[ERROR] rank=" << state.rank << " " << ex.what() << std::endl;
        moe_dispatch::Cleanup(&state);
        return 1;
    }
}
