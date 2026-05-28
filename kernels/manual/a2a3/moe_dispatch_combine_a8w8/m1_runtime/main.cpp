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
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
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

uint64_t Fnv1aAppend(uint64_t hash, const void *data, size_t bytes)
{
    constexpr uint64_t kFnvPrime = 1099511628211ULL;
    const auto *ptr = static_cast<const uint8_t *>(data);
    for (size_t i = 0; i < bytes; ++i) {
        hash ^= ptr[i];
        hash *= kFnvPrime;
    }
    return hash;
}

uint64_t MixChecksum(uint64_t hash, uint64_t value)
{
    return Fnv1aAppend(hash, &value, sizeof(value));
}

template <typename T>
uint64_t ChecksumVector(const std::vector<T> &values)
{
    constexpr uint64_t kFnvOffset = 1469598103934665603ULL;
    uint64_t hash = MixChecksum(kFnvOffset, static_cast<uint64_t>(values.size()));
    if (!values.empty()) {
        hash = Fnv1aAppend(hash, values.data(), values.size() * sizeof(T));
    }
    return hash;
}

} // namespace

struct PerfStats {
    double avg = 0.0;
    double min = 0.0;
    double max = 0.0;
    double stddev = 0.0;
};

struct IterationTiming {
    double dispatchE2eUs = 0.0;
    double prepareHostUs = 0.0;
    double combineE2eUs = 0.0;
    double totalE2eUs = 0.0;
};

struct RankCorrectnessSummary {
    uint64_t elementCount = 0;
    uint64_t errCount = 0;
    uint64_t firstMismatchIndex = 0;
    uint64_t checksumActual = 0;
    uint64_t checksumExpected = 0;
    uint64_t tokenPerExpertMatrixChecksum = 0;
    uint64_t cumsumMMChecksum = 0;
    uint64_t preSumBeforeRankChecksum = 0;
    uint64_t expandedRowIdxChecksum = 0;
    uint64_t dispatchedAChecksum = 0;
    uint64_t returnPayloadChecksum = 0;
    double maxAbsDiff = 0.0;
    double maxRelDiff = 0.0;
    uint32_t pass = 0;
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

std::vector<double> ExtractTimingSamples(const std::vector<IterationTiming> &timings,
                                         double IterationTiming::*field)
{
    std::vector<double> samples;
    samples.reserve(timings.size());
    for (const IterationTiming &timing : timings) {
        samples.push_back(timing.*field);
    }
    return samples;
}

void PrintOneTimingStats(const char *label, const std::vector<IterationTiming> &timings,
                         double IterationTiming::*field)
{
    PerfStats stats = CalcStats(ExtractTimingSamples(timings, field));
    std::cout << "  " << label << ": avg=" << stats.avg << " us"
              << " max=" << stats.max << " us" << std::endl;
}

bool VerboseRuntimeLogs(const DispatchCombineTileArgs &args)
{
    return args.runtime.debug != 0 || args.runtime.hostGoldenOnly != 0 || args.runtime.skipKernels != 0 ||
           args.runtime.dispatchMetadataOnly != 0 || args.runtime.dispatchOnly != 0 ||
           args.runtime.combineReturnOnly != 0;
}

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
    RankCorrectnessSummary correctness;
    bool correctnessReady = false;
    double dispatchE2eUs = 0.0;
    double prepareHostUs = 0.0;
    double combineE2eUs = 0.0;
    double totalE2eUs = 0.0;
    uint32_t currentSignalValue = 1;
};

void PrintProfileSummary(const DispatchCombineTileArgs &args, RuntimeState *state,
                         const std::vector<IterationTiming> &localTimings,
                         const RankCorrectnessSummary &localCorrectness)
{
    if (args.runtime.iters == 0) {
        return;
    }

    std::vector<IterationTiming> allTimings;
    std::vector<RankCorrectnessSummary> allCorrectness;
    size_t measuredIters = localTimings.size();
    if (state->rank == 0) {
        allTimings.resize(static_cast<size_t>(state->size) * localTimings.size());
        allCorrectness.resize(state->size);
    }
    MpiGatherBytes(&state->mpi, localTimings.data(), localTimings.size() * sizeof(IterationTiming),
                   state->rank == 0 ? allTimings.data() : nullptr, 0);
    MpiGatherBytes(&state->mpi, &localCorrectness, sizeof(RankCorrectnessSummary),
                   state->rank == 0 ? allCorrectness.data() : nullptr, 0);

    if (state->rank != 0) {
        return;
    }

    std::vector<IterationTiming> globalTimings(measuredIters);
    for (size_t iter = 0; iter < measuredIters; ++iter) {
        IterationTiming timing;
        for (uint32_t rank = 0; rank < state->size; ++rank) {
            const IterationTiming &rankTiming = allTimings[static_cast<size_t>(rank) * measuredIters + iter];
            timing.dispatchE2eUs = std::max(timing.dispatchE2eUs, rankTiming.dispatchE2eUs);
            timing.prepareHostUs = std::max(timing.prepareHostUs, rankTiming.prepareHostUs);
            timing.combineE2eUs = std::max(timing.combineE2eUs, rankTiming.combineE2eUs);
            timing.totalE2eUs = std::max(timing.totalE2eUs, rankTiming.totalE2eUs);
        }
        globalTimings[iter] = timing;
    }

    PerfStats dispatchStats = CalcStats(ExtractTimingSamples(globalTimings, &IterationTiming::dispatchE2eUs));
    PerfStats prepareStats = CalcStats(ExtractTimingSamples(globalTimings, &IterationTiming::prepareHostUs));
    PerfStats combineStats = CalcStats(ExtractTimingSamples(globalTimings, &IterationTiming::combineE2eUs));
    PerfStats totalStats = CalcStats(ExtractTimingSamples(globalTimings, &IterationTiming::totalE2eUs));

    RankCorrectnessSummary globalCorrectness;
    globalCorrectness.pass = 1;
    constexpr uint64_t kFnvOffset = 1469598103934665603ULL;
    globalCorrectness.checksumActual = kFnvOffset;
    globalCorrectness.checksumExpected = kFnvOffset;
    globalCorrectness.tokenPerExpertMatrixChecksum = kFnvOffset;
    globalCorrectness.cumsumMMChecksum = kFnvOffset;
    globalCorrectness.preSumBeforeRankChecksum = kFnvOffset;
    globalCorrectness.expandedRowIdxChecksum = kFnvOffset;
    globalCorrectness.dispatchedAChecksum = kFnvOffset;
    globalCorrectness.returnPayloadChecksum = kFnvOffset;
    for (const RankCorrectnessSummary &rankSummary : allCorrectness) {
        globalCorrectness.elementCount += rankSummary.elementCount;
        globalCorrectness.errCount += rankSummary.errCount;
        globalCorrectness.maxAbsDiff = std::max(globalCorrectness.maxAbsDiff, rankSummary.maxAbsDiff);
        globalCorrectness.maxRelDiff = std::max(globalCorrectness.maxRelDiff, rankSummary.maxRelDiff);
        globalCorrectness.pass &= rankSummary.pass;
        globalCorrectness.checksumActual = MixChecksum(globalCorrectness.checksumActual, rankSummary.checksumActual);
        globalCorrectness.checksumExpected =
            MixChecksum(globalCorrectness.checksumExpected, rankSummary.checksumExpected);
        globalCorrectness.tokenPerExpertMatrixChecksum =
            MixChecksum(globalCorrectness.tokenPerExpertMatrixChecksum, rankSummary.tokenPerExpertMatrixChecksum);
        globalCorrectness.cumsumMMChecksum =
            MixChecksum(globalCorrectness.cumsumMMChecksum, rankSummary.cumsumMMChecksum);
        globalCorrectness.preSumBeforeRankChecksum =
            MixChecksum(globalCorrectness.preSumBeforeRankChecksum, rankSummary.preSumBeforeRankChecksum);
        globalCorrectness.expandedRowIdxChecksum =
            MixChecksum(globalCorrectness.expandedRowIdxChecksum, rankSummary.expandedRowIdxChecksum);
        globalCorrectness.dispatchedAChecksum =
            MixChecksum(globalCorrectness.dispatchedAChecksum, rankSummary.dispatchedAChecksum);
        globalCorrectness.returnPayloadChecksum =
            MixChecksum(globalCorrectness.returnPayloadChecksum, rankSummary.returnPayloadChecksum);
        if (rankSummary.errCount != 0 && globalCorrectness.firstMismatchIndex == 0) {
            globalCorrectness.firstMismatchIndex = rankSummary.firstMismatchIndex;
        }
    }

    std::cout << std::fixed << std::setprecision(1);
    std::cout << "\n================================================================" << std::endl;
    std::cout << "[PROFILE] M1RealDispatchCombineMockGmm" << std::endl;
    std::cout << "  M=" << args.shape.m << " K=" << args.shape.k << " ranks=" << args.shape.ep
              << " topK=" << args.shape.topK << " expertPerPe=" << args.shape.expertPerRank
              << " warmup=" << args.runtime.warmup << " measured=" << args.runtime.iters
              << " samples=" << globalTimings.size() << std::endl;
    std::cout << "  logical work: input tokens(all ranks)=" << static_cast<uint64_t>(args.shape.ep) * args.shape.m
              << " routed tokens(all ranks)="
              << static_cast<uint64_t>(args.shape.ep) * args.shape.m * args.shape.topK << std::endl;
    std::cout << "  dispatch_e2e: avg=" << dispatchStats.avg << " us max=" << dispatchStats.max << " us"
              << std::endl;
    std::cout << "  prepare_host: avg=" << prepareStats.avg << " us max=" << prepareStats.max << " us" << std::endl;
    std::cout << "  combine_e2e: avg=" << combineStats.avg << " us max=" << combineStats.max << " us" << std::endl;
    std::cout << "  total_e2e: avg=" << totalStats.avg << " us max=" << totalStats.max << " us" << std::endl;
    std::cout << "  verify=" << (globalCorrectness.pass == 0 ? "FAIL" : "PASS") << std::endl;
    std::cout << "  note: warmup iterations are excluded; each measured sample is the max across ranks." << std::endl;
    std::cout << "================================================================\n" << std::endl;

    std::cout << std::setprecision(6);
    std::cout << "[CorrectnessReport]\n";
    std::cout << "  case_name=" << args.caseName << "\n";
    std::cout << "  seed=" << args.runtime.seed << "\n";
    std::cout << "  rankNum=" << state->size << " rankId=0"
              << " expertPerRank=" << args.shape.expertPerRank << " topK=" << args.shape.topK
              << " M=" << args.shape.m << " hiddenSize=" << args.shape.k
              << " intermediateSize=" << args.intermediateSize
              << " maxTokensPerExpert=" << args.shape.maxOutputSize / args.shape.expertPerRank << "\n";
    std::cout << "  dtype_in=fp16 dtype_out=fp16\n";
    std::cout << "  reference_source=m1-real-dispatch-combine-host-reference\n";
    std::cout << "  protocol_stage=m1_real_dispatch_combine_mock_gmm\n";
    std::cout << "  final_output.max_abs_diff=" << globalCorrectness.maxAbsDiff << "\n";
    std::cout << "  final_output.max_rel_diff=" << globalCorrectness.maxRelDiff << "\n";
    std::cout << "  final_output.err_count=" << globalCorrectness.errCount << "\n";
    std::cout << "  final_output.err_threshold=0\n";
    std::cout << "  final_output.tolerance_atol=" << args.atol << "\n";
    std::cout << "  final_output.tolerance_rtol=" << args.rtol << "\n";
    std::cout << "  final_output.checksum_actual=" << globalCorrectness.checksumActual << "\n";
    std::cout << "  final_output.checksum_expected=" << globalCorrectness.checksumExpected << "\n";
    std::cout << "  final_output.pass=" << (globalCorrectness.pass == 0 ? "false" : "true") << "\n";
    std::cout << "  intermediate.tokenPerExpertMatrix_checksum="
              << globalCorrectness.tokenPerExpertMatrixChecksum << "\n";
    std::cout << "  intermediate.cumsumMM_checksum=" << globalCorrectness.cumsumMMChecksum << "\n";
    std::cout << "  intermediate.preSumBeforeRank_checksum=" << globalCorrectness.preSumBeforeRankChecksum << "\n";
    std::cout << "  intermediate.expandedRowIdx_checksum=" << globalCorrectness.expandedRowIdxChecksum << "\n";
    std::cout << "  intermediate.dispatchedA_checksum=" << globalCorrectness.dispatchedAChecksum << "\n";
    std::cout << "  intermediate.returnPayload_checksum=" << globalCorrectness.returnPayloadChecksum << "\n";
    std::cout << "  intermediate.gmm1_accumulator_checksum=null\n";
    std::cout << "  intermediate.gmm2_accumulator_checksum=null\n";
    std::cout << "  intermediate.scale_dequant_checksum=null\n";
    std::cout << "  dispatch_tget_real=true\n";
    std::cout << "  combine_tput_real=true\n";
    std::cout << "  count_wait_real=true\n";
    std::cout << "  notify_wait_real=true\n";
    std::cout << "  ttest_wait_real=true\n";
    std::cout << "  mock_payload_target=workspace.dispatchedA\n";
    std::cout << "  gmm_block_mock=true\n";
    std::cout << "  two_rank_npu_run=" << (state->size >= 2 ? "true" : "false") << "\n";
    std::cout << "  pass=" << (globalCorrectness.pass == 0 ? "false" : "true") << "\n";

    std::cout << std::fixed << std::setprecision(1);
    std::cout << "[PerfReport]\n";
    std::cout << "  case_name=" << args.caseName << "\n";
    std::cout << "  warmup_iters=" << args.runtime.warmup << "\n";
    std::cout << "  measure_iters=" << args.runtime.iters << "\n";
    std::cout << "  rankNum=" << state->size << " rankId=0 shape=M" << args.shape.m << "xH" << args.shape.k
              << "xI" << args.intermediateSize << "\n";
    std::cout << "  correctness_pass=" << (globalCorrectness.pass == 0 ? "false" : "true") << "\n";
    std::cout << "  e2e_us.samples=" << globalTimings.size() << "\n";
    std::cout << "  e2e_us.avg=" << totalStats.avg << "\n";
    std::cout << "  e2e_us.min=" << totalStats.min << "\n";
    std::cout << "  e2e_us.max=" << totalStats.max << "\n";
    std::cout << "  e2e_us.stddev=" << totalStats.stddev << "\n";
    std::cout << "  stage_us.dispatch_gather.avg=" << dispatchStats.avg << "\n";
    std::cout << "  stage_us.mock_gmm_bridge.avg=" << prepareStats.avg << "\n";
    std::cout << "  stage_us.fused_combine_return.avg=" << combineStats.avg << "\n";
    std::cout << "  stage_us.total.avg=" << totalStats.avg << "\n";
    std::cout << "  pass=" << (globalCorrectness.pass == 0 ? "false" : "true") << "\n";
}

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

void BindDeviceContinuous(const DispatchCombineTileArgs &args, RuntimeState *state)
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

void CreateStreams(const DispatchCombineTileArgs &args, RuntimeState *state)
{
    bool verbose = VerboseRuntimeLogs(args);
    if (verbose) {
        PrintStage(state->rank, "create_streams", "begin");
    }
    if (aclrtCreateStream(&state->computeStream) != ACL_SUCCESS) {
        throw std::runtime_error("rank " + std::to_string(state->rank) + " aclrtCreateStream failed");
    }
    if (rtStreamCreate(&state->hcclStream, kRtStreamPriorityDefault) != 0) {
        throw std::runtime_error("rank " + std::to_string(state->rank) + " rtStreamCreate failed");
    }
    if (verbose) {
        PrintStage(state->rank, "create_streams", "done");
    }
}

void InitHccl(RuntimeState *state, const DispatchCombineTileArgs &args, const PeerWindowLayout &peerWindowLayout)
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

void AllocateLocalBuffers(const DispatchCombineTileArgs &args, const WorkspaceLayout &workspaceLayout,
                          RuntimeState *state)
{
    bool verbose = VerboseRuntimeLogs(args);
    if (verbose) {
        PrintStage(state->rank, "allocate_buffers", "begin");
    }
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
    if (verbose) {
        PrintStage(state->rank, "allocate_buffers", "done");
    }
}

void CopyInputsToDevice(const DispatchCombineTileArgs &args, RuntimeState *state)
{
    if (!state->dataReady) {
        throw std::runtime_error("host data is not ready before device copy");
    }
    bool verbose = VerboseRuntimeLogs(args);
    if (verbose) {
        PrintStage(state->rank, "copy_inputs", "begin");
    }
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
    if (verbose) {
        PrintStage(state->rank, "copy_inputs", "done");
    }
}

void ClearDeviceState(const DispatchCombineTileArgs &args, const WorkspaceLayout &workspaceLayout,
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

void CopyDispatchMetadataToHost(const DispatchCombineTileArgs &args, const WorkspaceLayout &workspaceLayout,
                                const PeerWindowLayout &peerWindowLayout, RuntimeState *state,
                                DispatchMetadataDump *dump)
{
    const DispatchCombineTileShape &shape = args.shape;
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
    if (args.runtime.debug >= 2 || args.runtime.dispatchOnly != 0) {
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

uint64_t CompareFloatBuffer(const DispatchCombineTileArgs &args, const std::string &name,
                            const std::vector<float> &actual, const std::vector<float> &expected, uint32_t rank)
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

uint64_t VerifyDispatchMetadata(const DispatchCombineTileArgs &args, RuntimeState *state,
                                const DispatchMetadataDump &dump)
{
    uint64_t mismatches = 0;
    mismatches += CompareI32Buffer("localTokenPerExpert", dump.localTokenPerExpert, state->golden.localTokenPerExpert,
                                   state->rank);
    if (args.runtime.debug >= 2) {
        const DispatchCombineTileShape &shape = args.shape;
        size_t expertNumPadded = static_cast<size_t>(ExpertNumPadded(shape));
        size_t aivBlocks = static_cast<size_t>(EffectiveAivBlocks(shape));
        std::vector<int32_t> expectedBlockToken(aivBlocks * expertNumPadded, 0);
        std::vector<int32_t> expectedBlockPrefix(aivBlocks * expertNumPadded, 0);
        for (uint32_t token = 0; token < shape.m; ++token) {
            uint32_t block = static_cast<uint32_t>((static_cast<uint64_t>(token) * aivBlocks) / shape.m);
            if (block >= aivBlocks) {
                block = static_cast<uint32_t>(aivBlocks - 1);
            }
            for (uint32_t slot = 0; slot < shape.topK; ++slot) {
                size_t routeIndex = static_cast<size_t>(token) * shape.topK + slot;
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

void RunDispatch(const DispatchCombineTileArgs &args, const WorkspaceLayout &workspaceLayout,
                 const PeerWindowLayout &peerWindowLayout, RuntimeState *state)
{
    bool verbose = VerboseRuntimeLogs(args);
    if (verbose) {
        PrintStage(state->rank, "dispatch", "begin");
    }
    uint32_t launchBlocks = args.shape.aivBlocks == 0 ? 1 : args.shape.aivBlocks;
    DispatchCombineTileShape launchShape = args.shape;
    launchShape.signalValue = state->currentSignalValue;
    MpiBarrier(&state->mpi);
    auto dispatchStart = std::chrono::steady_clock::now();
    LaunchDispatchCombineTileDispatch(
        launchShape, state->rank, reinterpret_cast<uint8_t *>(state->buffers.inputA),
        reinterpret_cast<uint8_t *>(state->buffers.expertIdx), reinterpret_cast<uint8_t *>(state->hccl.peerWindow),
        reinterpret_cast<uint8_t *>(state->hccl.deviceContext), reinterpret_cast<uint8_t *>(state->buffers.workspace),
        state->computeStream, launchBlocks);
    CheckAcl(aclrtSynchronizeStream(state->computeStream),
             "rank " + std::to_string(state->rank) + " dispatch stream sync");
    MpiBarrier(&state->mpi);
    auto dispatchEnd = std::chrono::steady_clock::now();
    state->dispatchE2eUs = UsSince(dispatchStart, dispatchEnd);

    if (args.runtime.debug != 0 || args.runtime.dispatchMetadataOnly != 0 || args.runtime.dispatchOnly != 0) {
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
        if (args.runtime.dispatchOnly == 0) {
            if (verbose) {
                PrintStage(state->rank, "dispatch", "done");
            }
            return;
        }
        PrintDispatchGatherSegments(args, state);
        uint64_t payloadMismatches = VerifyDispatchPayload(args, state, dump);
        std::cout << "rank=" << state->rank << " dispatch_payload_mismatches=" << payloadMismatches << std::endl;
        if (payloadMismatches != 0) {
            throw std::runtime_error("rank " + std::to_string(state->rank) + " dispatch payload mismatch");
        }
        std::cout << "rank=" << state->rank << " dispatch_pack_done" << std::endl;
        std::cout << "rank=" << state->rank << " dispatch_gather_done" << std::endl;
    }
    if (verbose) {
        PrintStage(state->rank, "dispatch", "done");
    }
}

void PrepareExpertOutputIdentity(const DispatchCombineTileArgs &args, const WorkspaceLayout &workspaceLayout,
                                 RuntimeState *state)
{
    bool verbose = VerboseRuntimeLogs(args);
    if (verbose) {
        PrintStage(state->rank, "prepare_expert_output", "begin");
    }
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
    if (verbose) {
        PrintStage(state->rank, "prepare_expert_output", "done");
    }
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
    bool verbose = VerboseRuntimeLogs(args);
    if (verbose) {
        PrintStage(state->rank, "clear_combine_state", "begin");
    }
    const DispatchCombineTileShape &shape = args.shape;
    size_t expandedRows = static_cast<size_t>(shape.m) * shape.topK;
    auto *peerBase = reinterpret_cast<uint8_t *>(state->hccl.peerWindow);
    CheckAcl(aclrtMemset(peerBase + peerWindowLayout.ptrD, BytesOfHalfVector(expandedRows * shape.k), 0,
                         BytesOfHalfVector(expandedRows * shape.k)),
             "rank " + std::to_string(state->rank) + " clear ptrD");
    CheckAcl(aclrtMemset(state->buffers.outputC, BytesOfHalfVector(static_cast<size_t>(shape.m) * shape.k), 0,
                         BytesOfHalfVector(static_cast<size_t>(shape.m) * shape.k)),
             "rank " + std::to_string(state->rank) + " clear combine outputC");
    CheckAcl(aclrtSynchronizeStream(state->computeStream),
             "rank " + std::to_string(state->rank) + " sync clear combine state");
    if (verbose) {
        PrintStage(state->rank, "clear_combine_state", "done");
    }
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
            int32_t rows = state->golden.peerTokenPerExpert[static_cast<size_t>(dst) * expertNumPadded + globalExpert];
            if (rows <= 0) {
                continue;
            }
            int32_t srcStart =
                state->golden.dispatchOffset[localExpert] +
                state->golden.prevSumBeforeRank[static_cast<size_t>(dst) * shape.expertPerRank + localExpert];
            int32_t dstStart =
                globalExpert == 0 ?
                    0 :
                    state->golden.cumsumPerExpert[static_cast<size_t>(dst) * expertNumPadded + globalExpert - 1];
            std::cout << "rank=" << state->rank << " combine_return dst=" << dst << " local_expert=" << localExpert
                      << " rows=" << rows << " src_start=" << srcStart << " dst_start=" << dstStart << std::endl;
        }
    }
}

void RunCombine(const DispatchCombineTileArgs &args, const PeerWindowLayout &peerWindowLayout, RuntimeState *state)
{
    bool verbose = VerboseRuntimeLogs(args);
    if (verbose) {
        PrintStage(state->rank, "combine", "begin");
    }
    ClearCombineReturnState(args, peerWindowLayout, state);
    PrintCombineReturnSegments(args, state);
    uint32_t launchBlocks = args.shape.aivBlocks == 0 ? 1 : args.shape.aivBlocks;
    DispatchCombineTileShape launchShape = args.shape;
    launchShape.signalValue = state->currentSignalValue;
    MpiBarrier(&state->mpi);
    auto combineStart = std::chrono::steady_clock::now();
    LaunchDispatchCombineTileCombine(
        launchShape, state->rank, reinterpret_cast<uint8_t *>(state->buffers.expertOutput),
        reinterpret_cast<uint8_t *>(state->buffers.probs), reinterpret_cast<uint8_t *>(state->buffers.outputC),
        reinterpret_cast<uint8_t *>(state->hccl.peerWindow), reinterpret_cast<uint8_t *>(state->hccl.deviceContext),
        reinterpret_cast<uint8_t *>(state->buffers.workspace), state->computeStream, launchBlocks);
    CheckAcl(aclrtSynchronizeStream(state->computeStream),
             "rank " + std::to_string(state->rank) + " combine stream sync");
    MpiBarrier(&state->mpi);
    auto combineEnd = std::chrono::steady_clock::now();
    state->combineE2eUs = UsSince(combineStart, combineEnd);

    if (args.runtime.combineReturnOnly != 0 || args.runtime.debug >= 2) {
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
    }
    if (verbose && args.runtime.combineReturnOnly != 0) {
        std::cout << "rank=" << state->rank << " combine_return_done" << std::endl;
        std::cout << "rank=" << state->rank << " combine_wait_done peers=" << args.shape.ep << std::endl;
    } else if (verbose) {
        std::cout << "rank=" << state->rank << " combine_return_done" << std::endl;
        std::cout << "rank=" << state->rank << " combine_wait_done peers=" << args.shape.ep << std::endl;
        std::cout << "rank=" << state->rank << " restore_done" << std::endl;
    }
    if (verbose) {
        PrintStage(state->rank, "combine", "done");
    }
}

RankCorrectnessSummary VerifyAndDump(const DispatchCombineTileArgs &args, RuntimeState *state)
{
    RankCorrectnessSummary summary;
    bool verbose = VerboseRuntimeLogs(args);
    if (verbose) {
        PrintStage(state->rank, "verify", "begin");
    }
    size_t elements = static_cast<size_t>(args.shape.m) * args.shape.k;
    std::vector<float> actualOutputC = CopyDeviceHalfToFloat(state->buffers.outputC, elements, state->rank, "outputC");
    std::vector<float> expectedOutputC = HalfBitsToFloatVector(FloatVectorToHalfBits(state->golden.outputC));
    summary.elementCount = static_cast<uint64_t>(state->golden.outputC.size());
    summary.checksumActual = ChecksumVector(actualOutputC);
    summary.checksumExpected = ChecksumVector(expectedOutputC);
    summary.tokenPerExpertMatrixChecksum = ChecksumVector(state->golden.peerTokenPerExpert);
    summary.cumsumMMChecksum = ChecksumVector(state->golden.cumsumPerExpert);
    summary.preSumBeforeRankChecksum = ChecksumVector(state->golden.prevSumBeforeRank);
    summary.expandedRowIdxChecksum = ChecksumVector(state->golden.expandedRowIdx);
    summary.dispatchedAChecksum = ChecksumVector(state->golden.dispatchedA);
    summary.returnPayloadChecksum = ChecksumVector(state->golden.ptrD);
    for (size_t i = 0; i < actualOutputC.size() && i < expectedOutputC.size(); ++i) {
        double actual = actualOutputC[i];
        double expected = expectedOutputC[i];
        double absDiff = std::fabs(actual - expected);
        double denom = std::fabs(expected) > std::numeric_limits<double>::epsilon() ? std::fabs(expected) : 1.0;
        summary.maxAbsDiff = std::max(summary.maxAbsDiff, absDiff);
        summary.maxRelDiff = std::max(summary.maxRelDiff, absDiff / denom);
    }
    if (args.runtime.verify == 0) {
        if (verbose) {
            std::cout << "rank=" << state->rank << " verify=SKIP mismatch_count=0" << std::endl;
            PrintStage(state->rank, "verify", "done");
        }
        summary.pass = 0;
        state->correctness = summary;
        state->correctnessReady = true;
        return summary;
    }
    if (args.runtime.debug != 0) {
        WriteBinaryFile(RankBinaryFile(args, state->rank, "outputC"), FloatVectorToHalfBits(actualOutputC));
    }
    CompareResult compare = CompareOutputs(args, state->golden, actualOutputC, state->rank);
    summary.errCount = compare.mismatchCount;
    summary.firstMismatchIndex = compare.firstMismatchIndex;
    summary.pass = compare.mismatchCount == 0 ? 1 : 0;
    if (verbose || compare.mismatchCount != 0) {
        std::cout << "rank=" << state->rank << " buffer=outputC elements=" << compare.elementCount
                  << " mismatch_count=" << compare.mismatchCount;
    }
    if (compare.mismatchCount != 0) {
        uint64_t row = compare.firstMismatchIndex / args.shape.k;
        uint64_t col = compare.firstMismatchIndex % args.shape.k;
        std::cout << " first_index=" << compare.firstMismatchIndex << " row=" << row << " col=" << col
                  << " actual=" << compare.actual << " expected=" << compare.expected;
    }
    if (verbose || compare.mismatchCount != 0) {
        std::cout << std::endl;
    }
    if (compare.mismatchCount != 0) {
        std::cout << "rank=" << state->rank << " verify=FAIL mismatch_count=" << compare.mismatchCount << std::endl;
        throw std::runtime_error("rank " + std::to_string(state->rank) + " outputC mismatch");
    }
    if (verbose) {
        std::cout << "rank=" << state->rank << " verify=PASS mismatch_count=0" << std::endl;
        PrintStage(state->rank, "verify", "done");
    }
    state->correctness = summary;
    state->correctnessReady = true;
    return summary;
}

void PrintPerformanceConfig(const DispatchCombineTileArgs &args, const WorkspaceLayout &workspaceLayout,
                            const PeerWindowLayout &peerWindowLayout, uint32_t rank)
{
    if (!VerboseRuntimeLogs(args)) {
        return;
    }
    uint32_t aivBlocks = args.shape.aivBlocks == 0 ? 1 : args.shape.aivBlocks;
    uint32_t peerShards = args.shape.ep < aivBlocks ? args.shape.ep : aivBlocks;
    std::cout << "rank=" << rank << " aiv_blocks=" << aivBlocks << " tile_cols=" << args.shape.tileCols
              << " peer_window_bytes=" << peerWindowLayout.totalBytes
              << " workspace_bytes=" << workspaceLayout.totalBytes << " dispatch_peer_shards=" << peerShards
              << " combine_peer_shards=" << peerShards << std::endl;
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
        bool verbose = dispatch_combine_tile::VerboseRuntimeLogs(args);
        if (verbose) {
            std::cout << "rank=" << state.rank << " size=" << state.size << " device=" << state.device << " start"
                      << std::endl;
        }

        if (args.runtime.hostGoldenOnly != 0) {
            dispatch_combine_tile::RunHostGoldenOnly(args, &state);
            dispatch_combine_tile::Cleanup(&state);
            return 0;
        }

        dispatch_combine_tile::PrepareHostData(args, &state);

        dispatch_combine_tile::BindDeviceContinuous(args, &state);
        dispatch_combine_tile::CreateStreams(args, &state);
        dispatch_combine_tile::InitHccl(&state, args, peerWindowLayout);
        dispatch_combine_tile::AllocateLocalBuffers(args, workspaceLayout, &state);
        dispatch_combine_tile::CopyInputsToDevice(args, &state);

        if (args.runtime.skipKernels != 0) {
            dispatch_combine_tile::MpiBarrier(&state.mpi);
            std::cout << "rank=" << state.rank << " skip_kernels_done" << std::endl;
            dispatch_combine_tile::Cleanup(&state);
            return 0;
        }

        dispatch_combine_tile::PrintPerformanceConfig(args, workspaceLayout, peerWindowLayout, state.rank);
        uint32_t totalIterations = args.runtime.warmup + args.runtime.iters;
        std::vector<dispatch_combine_tile::IterationTiming> measureTimings;
        measureTimings.reserve(args.runtime.iters);
        for (uint32_t iter = 0; iter < totalIterations; ++iter) {
            bool isWarmup = iter < args.runtime.warmup;
            state.currentSignalValue = iter + 1;
            if (verbose && !isWarmup) {
                std::cout << "rank=" << state.rank << " iteration=" << (iter - args.runtime.warmup)
                          << " phase=measure begin" << std::endl;
            }
            auto totalStart = std::chrono::steady_clock::now();
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
            state.correctness = dispatch_combine_tile::VerifyAndDump(args, &state);
            state.correctnessReady = true;
            auto totalEnd = std::chrono::steady_clock::now();
            state.totalE2eUs = dispatch_combine_tile::UsSince(totalStart, totalEnd);
            if (!isWarmup) {
                dispatch_combine_tile::IterationTiming timing;
                timing.dispatchE2eUs = state.dispatchE2eUs;
                timing.prepareHostUs = state.prepareHostUs;
                timing.combineE2eUs = state.combineE2eUs;
                timing.totalE2eUs = state.totalE2eUs;
                measureTimings.push_back(timing);
                if (verbose) {
                    std::cout << "rank=" << state.rank << " iteration=" << (iter - args.runtime.warmup)
                              << " phase=measure done" << std::endl;
                }
            }
        }
        dispatch_combine_tile::PrintProfileSummary(args, &state, measureTimings, state.correctness);
        dispatch_combine_tile::MpiBarrier(&state.mpi);
        if (verbose) {
            std::cout << "rank=" << state.rank << " run_done" << std::endl;
        }
        dispatch_combine_tile::Cleanup(&state);
        return 0;
    } catch (const std::exception &ex) {
        std::cerr << "[ERROR] rank=" << state.rank << " " << ex.what() << std::endl;
        dispatch_combine_tile::Cleanup(&state);
        return 1;
    }
}
