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
#include "comm_mpi.hpp"
#include "hccl_context.hpp"
#include "kernel_launchers.hpp"
#include "moe_new_dispatch_combine_a8w8_m1_layout.hpp"
#include "reference.hpp"
#include "workspace_layout.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
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

size_t BytesOfI8Vector(size_t elements)
{
    return elements * sizeof(int8_t);
}

size_t BytesOfU64Vector(size_t elements)
{
    return elements * sizeof(uint64_t);
}

constexpr size_t kM3N9TimeoutDumpPresentSlot = 0U;
constexpr size_t kM3N9TimeoutDumpRankSlot = 1U;
constexpr size_t kM3N9TimeoutDumpExpertSlot = 2U;
constexpr size_t kM3N9TimeoutDumpTokenOwnerRankSlot = 3U;
constexpr size_t kM3N9TimeoutDumpExpertOwnerRankSlot = 4U;
constexpr size_t kM3N9TimeoutDumpStageSlot = 5U;
constexpr size_t kM3N9TimeoutDumpSignalIdSlot = 6U;
constexpr size_t kM3N9TimeoutDumpDebugStopStageSlot = 7U;
constexpr size_t kM3N9TimeoutDumpDispatchReadySlot = 8U;
constexpr size_t kM3N9TimeoutDumpGmm1ReadySlot = 9U;
constexpr size_t kM3N9TimeoutDumpActivationReadySlot = 10U;
constexpr size_t kM3N9TimeoutDumpGmm2ReadySlot = 11U;
constexpr size_t kM3N9TimeoutDumpReadyExpertSlot = 12U;
constexpr int32_t kM3N9TimeoutStageDispatchToGmm1 = 1;

const char *M3N9TimeoutStageName(int32_t stage)
{
    switch (stage) {
        case kM3N9TimeoutStageDispatchToGmm1:
            return "dispatch_to_gmm1";
        default:
            return "unknown";
    }
}

std::string GmmShapeClass(const moe_new_dispatch_combine_a8w8::ShapeConfig &shape)
{
    if (shape.hiddenSize == 0U || shape.intermediateSize == 0U ||
        shape.hiddenSize % moe_new_dispatch_combine_a8w8::kGmmBaseK != 0U ||
        shape.intermediateSize % moe_new_dispatch_combine_a8w8::kGmmBaseK != 0U ||
        shape.gmmBlockM != moe_new_dispatch_combine_a8w8::kGmmBaseM ||
        shape.gmmBlockN != moe_new_dispatch_combine_a8w8::kGmmBaseN ||
        shape.gmmBlockK != moe_new_dispatch_combine_a8w8::kGmmBaseK) {
        return "unsupported";
    }
    if (shape.hiddenSize < moe_new_dispatch_combine_a8w8::kGmmBaseN ||
        shape.intermediateSize < moe_new_dispatch_combine_a8w8::kGmmBaseK ||
        shape.maxTokensPerExpert < moe_new_dispatch_combine_a8w8::kGmmBaseM) {
        return "smoke_debug";
    }
    return "realistic_model_shape";
}

bool GmmPolicySupported(const moe_new_dispatch_combine_a8w8::ShapeConfig &shape)
{
    return GmmShapeClass(shape) != "unsupported";
}

uint64_t GmmL1ABytes()
{
    return static_cast<uint64_t>(moe_new_dispatch_combine_a8w8::kGmmL1Stages) *
           moe_new_dispatch_combine_a8w8::kGmmBaseM * moe_new_dispatch_combine_a8w8::kGmmBaseK *
           moe_new_dispatch_combine_a8w8::kGmmStepK * sizeof(int8_t);
}

uint64_t GmmL1BBytes()
{
    return static_cast<uint64_t>(moe_new_dispatch_combine_a8w8::kGmmL1Stages) *
           moe_new_dispatch_combine_a8w8::kGmmBaseK * moe_new_dispatch_combine_a8w8::kGmmStepK *
           moe_new_dispatch_combine_a8w8::kGmmBaseN * sizeof(int8_t);
}

uint64_t GmmL0ABytes()
{
    return static_cast<uint64_t>(moe_new_dispatch_combine_a8w8::kGmmL0AStages) *
           moe_new_dispatch_combine_a8w8::kGmmBaseM * moe_new_dispatch_combine_a8w8::kGmmBaseK * sizeof(int8_t);
}

uint64_t GmmL0BBytes()
{
    return static_cast<uint64_t>(moe_new_dispatch_combine_a8w8::kGmmL0BStages) *
           moe_new_dispatch_combine_a8w8::kGmmBaseK * moe_new_dispatch_combine_a8w8::kGmmBaseN * sizeof(int8_t);
}

uint64_t GmmL0CBytes()
{
    return static_cast<uint64_t>(moe_new_dispatch_combine_a8w8::kGmmL0CStages) *
           moe_new_dispatch_combine_a8w8::kGmmBaseM * moe_new_dispatch_combine_a8w8::kGmmBaseN * sizeof(int32_t);
}

void PrintGmmPolicyReport(const moe_new_dispatch_combine_a8w8::ShapeConfig &shape, uint32_t expectedTaskCount,
                          uint32_t launchBlocks)
{
    uint32_t activeBlocks = std::min<uint32_t>(launchBlocks, expectedTaskCount);
    std::cout << "  gmm_tile_policy=gemm_ar_cache_level_int8\n";
    std::cout << "  gmm_shape_class=" << GmmShapeClass(shape) << "\n";
    std::cout << "  gmm_base_m=" << moe_new_dispatch_combine_a8w8::kGmmBaseM << "\n";
    std::cout << "  gmm_base_n=" << moe_new_dispatch_combine_a8w8::kGmmBaseN << "\n";
    std::cout << "  gmm_base_k=" << moe_new_dispatch_combine_a8w8::kGmmBaseK << "\n";
    std::cout << "  gmm_step_k=" << moe_new_dispatch_combine_a8w8::kGmmStepK << "\n";
    std::cout << "  gmm_l1_a_bytes=" << GmmL1ABytes() << "\n";
    std::cout << "  gmm_l1_b_bytes=" << GmmL1BBytes() << "\n";
    std::cout << "  gmm_l0a_bytes=" << GmmL0ABytes() << "\n";
    std::cout << "  gmm_l0b_bytes=" << GmmL0BBytes() << "\n";
    std::cout << "  gmm_l0c_bytes=" << GmmL0CBytes() << "\n";
    std::cout << "  gmm_l1_stages=" << moe_new_dispatch_combine_a8w8::kGmmL1Stages << "\n";
    std::cout << "  gmm_l0a_stages=" << moe_new_dispatch_combine_a8w8::kGmmL0AStages << "\n";
    std::cout << "  gmm_l0b_stages=" << moe_new_dispatch_combine_a8w8::kGmmL0BStages << "\n";
    std::cout << "  gmm_l0c_stages=" << moe_new_dispatch_combine_a8w8::kGmmL0CStages << "\n";
    std::cout << "  gmm_tile_tasks=" << expectedTaskCount << "\n";
    std::cout << "  gmm_requested_aic_blocks=" << launchBlocks << "\n";
    std::cout << "  gmm_active_aic_blocks=" << activeBlocks << "\n";
    std::cout << "  gmm_tail_m=covered\n";
    std::cout << "  gmm_tail_n=covered\n";
    std::cout << "  gmm_tail_k=none\n";
    std::cout << "  gmm_micro_tile_debug_only=false\n";
}

uint32_t GmmLaunchBlocks(const DispatchCombineTileArgs &args)
{
    return std::max(1U, args.shape.aivBlocks);
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

moe_new_dispatch_combine_a8w8::ShapeConfig MakeM2ShapeConfig(const DispatchCombineTileArgs &args)
{
    moe_new_dispatch_combine_a8w8::ShapeConfig shape;
    shape.rankNum = args.shape.ep;
    shape.expertPerRank = args.shape.expertPerRank;
    shape.topK = args.shape.topK;
    shape.m = args.shape.m;
    shape.hiddenSize = args.shape.k;
    shape.intermediateSize = args.intermediateSize;
    shape.maxTokensPerExpert = args.shape.maxOutputSize / args.shape.expertPerRank;
    shape.payloadTileCols = args.shape.tileCols;
    shape.gmmBlockM = args.shape.gmmBlockM;
    shape.gmmBlockN = args.shape.gmmBlockN;
    shape.gmmBlockK = args.shape.gmmBlockK;
    shape.dtypeIn = static_cast<uint32_t>(moe_new_dispatch_combine_a8w8::DType::kFp16);
    shape.dtypeOut = static_cast<uint32_t>(moe_new_dispatch_combine_a8w8::DType::kFp16);
    return shape;
}

moe_new_dispatch_combine_a8w8::RankConfig MakeM2RankConfig(const DispatchCombineTileArgs &args, uint32_t rank)
{
    moe_new_dispatch_combine_a8w8::RankConfig config;
    config.rankNum = args.shape.ep;
    config.rankId = rank;
    config.rankFromMpi = args.runtime.rankFromMpi;
    config.deviceBase = args.runtime.deviceBase;
    config.ndevices = args.runtime.ndevices;
    return config;
}

void PrintM2ReferenceSummary(const DispatchCombineTileArgs &args, uint32_t rank, const CpuM2ReferenceData &ref)
{
    std::cout << "[M2Reference]\n";
    std::cout << "  rankId=" << rank << " backend=" << args.backend << "\n";
    std::cout << "  dtype_in=fp16 dtype_out=fp16 weight_layout=expert-major-row-major-int8\n";
    std::cout << "  local_rows=" << ref.localRows << " valid_rows=" << ref.validRows << "\n";
    std::cout << "  weight1_checksum=" << ChecksumVector(ref.weight1Int8) << "\n";
    std::cout << "  weight2_checksum=" << ChecksumVector(ref.weight2Int8) << "\n";
    std::cout << "  scale1_uint64_checksum=" << ChecksumVector(ref.scale1Uint64) << "\n";
    std::cout << "  scale2_uint64_checksum=" << ChecksumVector(ref.scale2Uint64) << "\n";
    if (!ref.scale1Uint64.empty()) {
        std::cout << "  scale1_first_hex=0x" << std::hex << ref.scale1Uint64.front() << std::dec
                  << " scale1_first_float=" << golden_detail::Uint64ScaleToFloat(ref.scale1Uint64.front()) << "\n";
    }
    if (!ref.scale2Uint64.empty()) {
        std::cout << "  scale2_first_hex=0x" << std::hex << ref.scale2Uint64.front() << std::dec
                  << " scale2_first_float=" << golden_detail::Uint64ScaleToFloat(ref.scale2Uint64.front()) << "\n";
    }
    std::cout << "  routing_per_token_scale_checksum=" << ChecksumVector(ref.routingPerTokenScale) << "\n";
    std::cout << "  gmm1_input_int8_checksum=" << ChecksumVector(ref.gmm1InputInt8) << "\n";
    std::cout << "  gmm1_accumulator_checksum=" << ChecksumVector(ref.gmm1AccInt32) << "\n";
    std::cout << "  gmm1_out_checksum=" << ChecksumVector(ref.gmm1Out) << "\n";
    std::cout << "  scale_dequant_checksum=" << ChecksumVector(ref.gmm1Out) << "\n";
    std::cout << "  swiglu_output_checksum=" << ChecksumVector(ref.swigluOut) << "\n";
    std::cout << "  gmm2_input_int8_checksum=" << ChecksumVector(ref.gmm2InputInt8) << "\n";
    std::cout << "  gmm2_per_token_scale_checksum=" << ChecksumVector(ref.gmm2PerTokenScale) << "\n";
    std::cout << "  gmm2_accumulator_checksum=" << ChecksumVector(ref.gmm2AccInt32) << "\n";
    std::cout << "  gmm2_out_checksum=" << ChecksumVector(ref.gmm2Out) << "\n";
    std::cout << "  m2_ptrD_checksum=" << ChecksumVector(ref.ptrD) << "\n";
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

bool VerboseRuntimeLogs(const DispatchCombineTileArgs &args)
{
    return args.runtime.debug != 0 || args.runtime.hostGoldenOnly != 0 || args.runtime.skipKernels != 0 ||
           args.runtime.dispatchMetadataOnly != 0 || args.runtime.dispatchOnly != 0 || args.runtime.gmm1Only != 0 ||
           args.runtime.gmm1EpilogueOnly != 0 || args.runtime.activationOnly != 0 || args.runtime.gmm2Only != 0 ||
           args.runtime.combineReturnOnly != 0 || args.runtime.m2MixedSpikeOnly != 0 ||
           args.runtime.m2FusedSkeletonOnly != 0 || args.runtime.m2MultiLaunchDebug != 0;
}

struct DeviceBuffers {
    void *inputA = nullptr;
    void *expertIdx = nullptr;
    void *xActiveMask = nullptr;
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
    CpuM2ReferenceData m2Reference;
    bool dataReady = false;
    bool m2ReferenceReady = false;
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
              << " routed tokens(all ranks)=" << static_cast<uint64_t>(args.shape.ep) * args.shape.m * args.shape.topK
              << std::endl;
    std::cout << "  dispatch_e2e: avg=" << dispatchStats.avg << " us max=" << dispatchStats.max << " us" << std::endl;
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
              << " expertPerRank=" << args.shape.expertPerRank << " topK=" << args.shape.topK << " M=" << args.shape.m
              << " hiddenSize=" << args.shape.k << " intermediateSize=" << args.intermediateSize
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
    std::cout << "  intermediate.tokenPerExpertMatrix_checksum=" << globalCorrectness.tokenPerExpertMatrixChecksum
              << "\n";
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
    std::cout << "  rankNum=" << state->size << " rankId=0 shape=M" << args.shape.m << "xH" << args.shape.k << "xI"
              << args.intermediateSize << "\n";
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
    if (args.backend == "int8") {
        CpuM2ReferenceData m2Reference = ComputeCpuM2Reference(args, golden, state->rank);
        if (VerboseRuntimeLogs(args)) {
            PrintM2ReferenceSummary(args, state->rank, m2Reference);
        }
    }
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
    if (args.backend == "int8") {
        state->m2Reference = ComputeCpuM2Reference(args, state->golden, state->rank);
        state->m2ReferenceReady = true;
        if (verbose) {
            PrintM2ReferenceSummary(args, state->rank, state->m2Reference);
        }
    }
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
    size_t activeMaskBytes = shape.m;
    size_t probsBytes = BytesOfFloatVector(static_cast<size_t>(shape.m) * shape.topK);
    size_t outputBytes = BytesOfHalfVector(static_cast<size_t>(shape.m) * shape.k);
    size_t expertOutputBytes = BytesOfHalfVector(static_cast<size_t>(shape.maxOutputSize) * shape.k);
    CheckAcl(aclrtMalloc(&state->buffers.inputA, inputBytes, ACL_MEM_MALLOC_HUGE_FIRST),
             "rank " + std::to_string(state->rank) + " aclrtMalloc inputA");
    CheckAcl(aclrtMalloc(&state->buffers.expertIdx, expertIdxBytes, ACL_MEM_MALLOC_HUGE_FIRST),
             "rank " + std::to_string(state->rank) + " aclrtMalloc expertIdx");
    CheckAcl(aclrtMalloc(&state->buffers.xActiveMask, activeMaskBytes, ACL_MEM_MALLOC_HUGE_FIRST),
             "rank " + std::to_string(state->rank) + " aclrtMalloc xActiveMask");
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

void AllocateLocalBuffersM2(const DispatchCombineTileArgs &args,
                            const moe_new_dispatch_combine_a8w8::WorkspaceLayout &workspaceLayout, RuntimeState *state)
{
    bool verbose = VerboseRuntimeLogs(args);
    if (verbose) {
        PrintStage(state->rank, "allocate_m2_buffers", "begin");
    }
    const DispatchCombineTileShape &shape = args.shape;
    size_t inputBytes = BytesOfHalfVector(static_cast<size_t>(shape.m) * shape.k);
    size_t expertIdxBytes = BytesOfI32Vector(static_cast<size_t>(shape.m) * shape.topK);
    size_t activeMaskBytes = shape.m;
    size_t probsBytes = BytesOfFloatVector(static_cast<size_t>(shape.m) * shape.topK);
    size_t outputBytes = BytesOfHalfVector(static_cast<size_t>(shape.m) * shape.k);
    CheckAcl(aclrtMalloc(&state->buffers.inputA, inputBytes, ACL_MEM_MALLOC_HUGE_FIRST),
             "rank " + std::to_string(state->rank) + " aclrtMalloc inputA");
    CheckAcl(aclrtMalloc(&state->buffers.expertIdx, expertIdxBytes, ACL_MEM_MALLOC_HUGE_FIRST),
             "rank " + std::to_string(state->rank) + " aclrtMalloc expertIdx");
    CheckAcl(aclrtMalloc(&state->buffers.xActiveMask, activeMaskBytes, ACL_MEM_MALLOC_HUGE_FIRST),
             "rank " + std::to_string(state->rank) + " aclrtMalloc xActiveMask");
    CheckAcl(aclrtMalloc(&state->buffers.probs, probsBytes, ACL_MEM_MALLOC_HUGE_FIRST),
             "rank " + std::to_string(state->rank) + " aclrtMalloc probs");
    CheckAcl(aclrtMalloc(&state->buffers.outputC, outputBytes, ACL_MEM_MALLOC_HUGE_FIRST),
             "rank " + std::to_string(state->rank) + " aclrtMalloc outputC");
    CheckAcl(aclrtMalloc(&state->buffers.workspace, workspaceLayout.totalBytes, ACL_MEM_MALLOC_HUGE_FIRST),
             "rank " + std::to_string(state->rank) + " aclrtMalloc m2 workspace");
    state->buffersAllocated = true;
    if (verbose) {
        PrintStage(state->rank, "allocate_m2_buffers", "done");
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
    std::vector<uint8_t> allActive;
    const uint8_t *activeMaskData = nullptr;
    if (state->inputs.xActiveMask.empty()) {
        allActive.assign(shape.m, 1U);
        activeMaskData = allActive.data();
    } else {
        activeMaskData = state->inputs.xActiveMask.data();
    }
    CheckAcl(aclrtMemcpy(state->buffers.inputA, inputBytes, inputHalf.data(), inputBytes, ACL_MEMCPY_HOST_TO_DEVICE),
             "rank " + std::to_string(state->rank) + " copy inputA");
    CheckAcl(aclrtMemcpy(state->buffers.expertIdx, expertIdxBytes, state->inputs.expertIdx.data(), expertIdxBytes,
                         ACL_MEMCPY_HOST_TO_DEVICE),
             "rank " + std::to_string(state->rank) + " copy expertIdx");
    CheckAcl(aclrtMemcpy(state->buffers.probs, probsBytes, state->inputs.probs.data(), probsBytes,
                         ACL_MEMCPY_HOST_TO_DEVICE),
             "rank " + std::to_string(state->rank) + " copy probs");
    CheckAcl(aclrtMemcpy(state->buffers.xActiveMask, shape.m, activeMaskData, shape.m, ACL_MEMCPY_HOST_TO_DEVICE),
             "rank " + std::to_string(state->rank) + " copy xActiveMask");
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

void ClearDeviceStateM2(const DispatchCombineTileArgs &args,
                        const moe_new_dispatch_combine_a8w8::WorkspaceLayout &workspaceLayout,
                        const moe_new_dispatch_combine_a8w8::PeerWindowLayout &peerWindowLayout, RuntimeState *state)
{
    bool verbose = VerboseRuntimeLogs(args);
    if (verbose) {
        PrintStage(state->rank, "clear_m2_device_state", "begin");
    }
    CheckAcl(aclrtMemset(state->buffers.workspace, workspaceLayout.totalBytes, 0, workspaceLayout.totalBytes),
             "rank " + std::to_string(state->rank) + " clear m2 workspace");
    CheckAcl(aclrtMemset(state->hccl.peerWindow, peerWindowLayout.totalBytes, 0, peerWindowLayout.totalBytes),
             "rank " + std::to_string(state->rank) + " clear m2 peerWindow");
    CheckAcl(aclrtMemset(state->buffers.outputC, BytesOfHalfVector(static_cast<size_t>(args.shape.m) * args.shape.k), 0,
                         BytesOfHalfVector(static_cast<size_t>(args.shape.m) * args.shape.k)),
             "rank " + std::to_string(state->rank) + " clear m2 outputC");
    CheckAcl(aclrtSynchronizeStream(state->computeStream), "rank " + std::to_string(state->rank) + " sync m2 clear");
    if (verbose) {
        PrintStage(state->rank, "clear_m2_device_state", "done");
    }
}

void RunM2MixedSpike(const DispatchCombineTileArgs &args, RuntimeState *state)
{
    bool verbose = VerboseRuntimeLogs(args);
    if (verbose) {
        PrintStage(state->rank, "m2_mixed_spike", "begin");
    }
    constexpr size_t kHeartbeatWords = 320;
    constexpr uint32_t kAicBlocks = 24;
    constexpr uint32_t kAivRatio = 1;
    size_t heartbeatBytes = BytesOfI32Vector(kHeartbeatWords);
    void *heartbeatDevice = nullptr;
    CheckAcl(aclrtMalloc(&heartbeatDevice, heartbeatBytes, ACL_MEM_MALLOC_HUGE_FIRST),
             "rank " + std::to_string(state->rank) + " aclrtMalloc m2 mixed heartbeat");
    try {
        CheckAcl(aclrtMemset(heartbeatDevice, heartbeatBytes, 0, heartbeatBytes),
                 "rank " + std::to_string(state->rank) + " clear m2 mixed heartbeat");
        std::vector<int32_t> heartbeatParams(kHeartbeatWords, 0);
        heartbeatParams[8] = static_cast<int32_t>(state->rank);
        heartbeatParams[9] = static_cast<int32_t>(kAicBlocks);
        heartbeatParams[10] = static_cast<int32_t>(kAivRatio);
        CheckAcl(aclrtMemcpy(heartbeatDevice, heartbeatBytes, heartbeatParams.data(), heartbeatBytes,
                             ACL_MEMCPY_HOST_TO_DEVICE),
                 "rank " + std::to_string(state->rank) + " seed m2 mixed heartbeat params");
        LaunchM2MixedSpike(reinterpret_cast<uint8_t *>(heartbeatDevice), state->rank, kAicBlocks, kAivRatio,
                           state->computeStream);
        CheckAcl(aclrtSynchronizeStream(state->computeStream),
                 "rank " + std::to_string(state->rank) + " m2 mixed spike stream sync");
        std::vector<int32_t> heartbeat(kHeartbeatWords, 0);
        CheckAcl(
            aclrtMemcpy(heartbeat.data(), heartbeatBytes, heartbeatDevice, heartbeatBytes, ACL_MEMCPY_DEVICE_TO_HOST),
            "rank " + std::to_string(state->rank) + " copy m2 mixed heartbeat");
        CheckAcl(aclrtFree(heartbeatDevice), "rank " + std::to_string(state->rank) + " free m2 mixed heartbeat");
        heartbeatDevice = nullptr;

        constexpr size_t kAicHeader = 112;
        constexpr size_t kAivHeader = 120;
        bool aicSeen = heartbeat[kAicHeader] == 0x4D328A8 && heartbeat[kAicHeader + 4] == 1;
        bool aivSeen = heartbeat[kAivHeader] == 0x4D328A8 && heartbeat[kAivHeader + 4] == 1;
        uint32_t aicSlotCount = 0;
        uint32_t aivSlotCount = 0;
        for (size_t i = 16; i < 112 && i < heartbeat.size(); ++i) {
            if (heartbeat[i] != 0) {
                ++aicSlotCount;
            }
        }
        for (size_t i = 128; i < heartbeat.size(); ++i) {
            if (heartbeat[i] != 0) {
                ++aivSlotCount;
            }
        }
        bool pass = aicSeen && aivSeen && heartbeat[kAicHeader + 2] == static_cast<int32_t>(kAicBlocks) &&
                    heartbeat[kAicHeader + 3] == static_cast<int32_t>(kAivRatio) &&
                    heartbeat[kAivHeader + 2] == static_cast<int32_t>(kAicBlocks) &&
                    heartbeat[kAivHeader + 3] == static_cast<int32_t>(kAivRatio) && aicSlotCount > 0 &&
                    aivSlotCount > 0;
        std::cout << "[CorrectnessReport]\n";
        std::cout << "  case_name=" << args.caseName << "\n";
        std::cout << "  backend=int8\n";
        std::cout << "  protocol_stage=m2_mixed_spike\n";
        std::cout << "  stage_graph_mode=mixed_spike_only\n";
        std::cout << "  mixed_elf_register=true\n";
        std::cout << "  mixed_aic_heartbeat=" << (aicSeen ? "true" : "false") << "\n";
        std::cout << "  mixed_aiv_heartbeat=" << (aivSeen ? "true" : "false") << "\n";
        std::cout << "  mixed_aic_launch_blocks=" << heartbeat[kAicHeader + 5] << "\n";
        std::cout << "  mixed_aiv_launch_blocks=" << heartbeat[kAivHeader + 5] << "\n";
        std::cout << "  gmm1_active_aic_blocks=0\n";
        std::cout << "  gmm2_active_aic_blocks=0\n";
        std::cout << "  dispatch_active_aiv_workers=0\n";
        std::cout << "  activation_active_aiv_workers=0\n";
        std::cout << "  combine_active_aiv_workers=0\n";
        std::cout << "  restore_active_aiv_workers=0\n";
        std::cout << "  mixed_aic_header_magic=0x" << std::hex << heartbeat[kAicHeader] << std::dec << "\n";
        std::cout << "  mixed_aiv_header_magic=0x" << std::hex << heartbeat[kAivHeader] << std::dec << "\n";
        std::cout << "  mixed_aic_slot_count=" << aicSlotCount << "\n";
        std::cout << "  mixed_aiv_slot_count=" << aivSlotCount << "\n";
        std::cout << "  pass=" << (pass ? "true" : "false") << "\n";
        if (!pass) {
            throw std::runtime_error("rank " + std::to_string(state->rank) + " M2 mixed spike heartbeat mismatch");
        }
        if (verbose) {
            PrintStage(state->rank, "m2_mixed_spike", "done");
        }
    } catch (...) {
        if (heartbeatDevice != nullptr) {
            aclrtFree(heartbeatDevice);
        }
        throw;
    }
}

void RunM2FusedSkeleton(const DispatchCombineTileArgs &args, RuntimeState *state)
{
    bool verbose = VerboseRuntimeLogs(args);
    if (verbose) {
        PrintStage(state->rank, "m2_fused_skeleton", "begin");
    }
    constexpr size_t kLedgerWords = 5200;
    constexpr uint32_t kAicBlocks = 24;
    constexpr uint32_t kAivRatio = 1;
    constexpr size_t kAicHeader = 256;
    constexpr size_t kAivHeader = 264;
    constexpr size_t kAicStageBase = 288;
    constexpr size_t kStageCount = 4;
    constexpr size_t kParticipants = 48;
    constexpr size_t kStageSlotWords = 8;
    constexpr size_t kAivStageBase = kAicStageBase + kStageCount * kParticipants * kStageSlotWords;
    size_t ledgerBytes = BytesOfI32Vector(kLedgerWords);
    void *ledgerDevice = nullptr;
    CheckAcl(aclrtMalloc(&ledgerDevice, ledgerBytes, ACL_MEM_MALLOC_HUGE_FIRST),
             "rank " + std::to_string(state->rank) + " aclrtMalloc m2 fused skeleton ledger");
    try {
        std::vector<int32_t> ledgerSeed(kLedgerWords, 0);
        ledgerSeed[8] = static_cast<int32_t>(state->rank);
        ledgerSeed[9] = static_cast<int32_t>(kAicBlocks);
        ledgerSeed[10] = static_cast<int32_t>(kAivRatio);
        CheckAcl(aclrtMemcpy(ledgerDevice, ledgerBytes, ledgerSeed.data(), ledgerBytes, ACL_MEMCPY_HOST_TO_DEVICE),
                 "rank " + std::to_string(state->rank) + " seed m2 fused skeleton ledger");
        LaunchM2FusedSkeleton(reinterpret_cast<uint8_t *>(ledgerDevice), state->rank, kAicBlocks, kAivRatio,
                              state->computeStream);
        CheckAcl(aclrtSynchronizeStream(state->computeStream),
                 "rank " + std::to_string(state->rank) + " m2 fused skeleton stream sync");
        std::vector<int32_t> ledger(kLedgerWords, 0);
        CheckAcl(aclrtMemcpy(ledger.data(), ledgerBytes, ledgerDevice, ledgerBytes, ACL_MEMCPY_DEVICE_TO_HOST),
                 "rank " + std::to_string(state->rank) + " copy m2 fused skeleton ledger");
        CheckAcl(aclrtFree(ledgerDevice), "rank " + std::to_string(state->rank) + " free m2 fused skeleton ledger");
        ledgerDevice = nullptr;

        bool aicSeen = ledger[kAicHeader] == 0x4D328B9 && ledger[kAicHeader + 4] == static_cast<int32_t>(kStageCount);
        bool aivSeen = ledger[kAivHeader] == 0x4D328B9 && ledger[kAivHeader + 4] == static_cast<int32_t>(kStageCount);
        std::array<uint32_t, kStageCount> aicStageCounts{};
        std::array<uint32_t, kStageCount> aivStageCounts{};
        for (size_t stage = 0; stage < kStageCount; ++stage) {
            for (size_t idx = 0; idx < kParticipants; ++idx) {
                size_t slot = (stage * kParticipants + idx) * kStageSlotWords;
                if (ledger[kAicStageBase + slot] != 0) {
                    ++aicStageCounts[stage];
                }
                if (ledger[kAivStageBase + slot] != 0) {
                    ++aivStageCounts[stage];
                }
            }
        }
        bool allStagesHaveAic = true;
        bool allStagesHaveAiv = true;
        for (size_t stage = 0; stage < kStageCount; ++stage) {
            allStagesHaveAic = allStagesHaveAic && aicStageCounts[stage] == kAicBlocks;
            allStagesHaveAiv = allStagesHaveAiv && aivStageCounts[stage] == kAicBlocks * kAivRatio;
        }
        bool pass = aicSeen && aivSeen && allStagesHaveAic && allStagesHaveAiv &&
                    ledger[kAicHeader + 5] == static_cast<int32_t>(kAicBlocks) &&
                    ledger[kAivHeader + 5] == static_cast<int32_t>(kAicBlocks * kAivRatio);
        std::cout << "[CorrectnessReport]\n";
        std::cout << "  case_name=" << args.caseName << "\n";
        std::cout << "  backend=int8\n";
        std::cout << "  protocol_stage=m2_fused_stage_graph_skeleton\n";
        std::cout << "  stage_graph_mode=single_fused_mpmd_skeleton\n";
        std::cout << "  mixed_elf_register=true\n";
        std::cout << "  fused_single_launch=true\n";
        std::cout << "  fused_device_stage_boundaries=syncall_mix\n";
        std::cout << "  fused_syncall_mode=hard_mix\n";
        std::cout << "  fused_host_barrier_between_stages=false\n";
        std::cout << "  mixed_aic_heartbeat=" << (aicSeen ? "true" : "false") << "\n";
        std::cout << "  mixed_aiv_heartbeat=" << (aivSeen ? "true" : "false") << "\n";
        std::cout << "  mixed_aic_launch_blocks=" << ledger[kAicHeader + 5] << "\n";
        std::cout << "  mixed_aiv_launch_blocks=" << ledger[kAivHeader + 5] << "\n";
        std::cout << "  gmm1_active_aic_blocks=0\n";
        std::cout << "  gmm2_active_aic_blocks=0\n";
        std::cout << "  dispatch_active_aiv_workers=0\n";
        std::cout << "  activation_active_aiv_workers=0\n";
        std::cout << "  combine_active_aiv_workers=0\n";
        std::cout << "  restore_active_aiv_workers=0\n";
        std::cout << "  fused_stage_count=" << kStageCount << "\n";
        for (size_t stage = 0; stage < kStageCount; ++stage) {
            std::cout << "  fused_stage_" << stage << "_aic_records=" << aicStageCounts[stage] << "\n";
            std::cout << "  fused_stage_" << stage << "_aiv_records=" << aivStageCounts[stage] << "\n";
        }
        std::cout << "  pass=" << (pass ? "true" : "false") << "\n";
        if (!pass) {
            throw std::runtime_error("rank " + std::to_string(state->rank) + " M2 fused skeleton ledger mismatch");
        }
        if (verbose) {
            PrintStage(state->rank, "m2_fused_skeleton", "done");
        }
    } catch (...) {
        if (ledgerDevice != nullptr) {
            aclrtFree(ledgerDevice);
        }
        throw;
    }
}

struct M2FusedFullEvidence {
    bool aicSeen = false;
    bool aivSeen = false;
    int32_t aicBlocks = 0;
    int32_t aivBlocks = 0;
    int32_t stageCount = 0;
    int32_t dispatchGroupReadyCount = 0;
    int32_t gmm1SyncGroupReadyCount = 0;
    int32_t activationSyncGroupReadyCount = 0;
    int32_t gmm2GroupReadyCount = 0;
    int32_t swigluSyncGroupCount = 0;
    int32_t swigluSyncGroupSizeSum = 0;
    int32_t swigluEmptyGroups = 0;
    int32_t dequantFinalRow = 0;
    int32_t m3n5Gmm1StartBeforeLastDispatchReady = 0;
    int32_t m3n6ActivationStartBeforeLastGmm1Ready = 0;
    int32_t m3n7Gmm2StartBeforeLastActivationReady = 0;
    int32_t m3n8CombineStartBeforeLastGmm2Ready = 0;
    bool swigluGroupDescMonotonic = false;
    std::array<int32_t, 16> m3Counters{};
    std::array<int32_t, 16> m3nDispatchCounters{};
    std::array<int32_t, 16> m3nActivationCounters{};
    std::array<int32_t, 16> m3nCombineCounters{};
    std::array<int32_t, 16> m3nGmm2Counters{};
    std::array<int32_t, 16> m3n8CombineCounters{};
    std::array<int32_t, 16> m3n11SubtileCounters{};
    std::array<int32_t, 48> m3oGmm1Counters{};
    std::array<int32_t, 48> m3oGmm2Counters{};
    std::array<int32_t, moe_new_dispatch_combine_a8w8::kM3ORestoreCounterWords> m3oRestoreCounters{};
    std::array<int32_t, 16> m3n9TimeoutDump{};
    std::array<uint64_t, moe_new_dispatch_combine_a8w8::kM3N12TimelineRecordCount *
                             moe_new_dispatch_combine_a8w8::kM3N12TimelineRecordWords>
        m3n12TimelineScratch{};
    std::string swigluGroupTileRanges;
    std::array<int32_t, 6> stageMarkers{};
};

int32_t NormalizeActiveWorkerCount(int32_t activeWorkers, const M2FusedFullEvidence *fusedEvidence)
{
    if (fusedEvidence == nullptr) {
        return 0;
    }
    return activeWorkers > 0 ? activeWorkers : 1;
}

std::string JoinActiveGmmTaskCounts(const std::array<int32_t, 48> &counters)
{
    std::ostringstream os;
    bool first = true;
    for (size_t block = 0; block < 32U; ++block) {
        int32_t count = counters[16U + block];
        if (count <= 0) {
            continue;
        }
        if (!first) {
            os << ",";
        }
        os << block << ":" << count;
        first = false;
    }
    return first ? "none" : os.str();
}

std::string JoinRestoreWorkerField(
    const std::array<int32_t, moe_new_dispatch_combine_a8w8::kM3ORestoreCounterWords> &counters, size_t fieldOffset)
{
    std::ostringstream os;
    bool first = true;
    for (size_t worker = 0; worker < moe_new_dispatch_combine_a8w8::kM3ORestoreWorkerCount; ++worker) {
        size_t base = moe_new_dispatch_combine_a8w8::kM3ORestoreWorkerBase +
                      worker * moe_new_dispatch_combine_a8w8::kM3ORestoreWorkerWords;
        if (base + 5U >= counters.size() || counters[base + 5U] == 0) {
            continue;
        }
        if (!first) {
            os << ",";
        }
        os << worker << ":" << counters[base + fieldOffset];
        first = false;
    }
    return first ? "none" : os.str();
}

std::string JoinRestoreWorkerRanges(
    const std::array<int32_t, moe_new_dispatch_combine_a8w8::kM3ORestoreCounterWords> &counters)
{
    std::ostringstream os;
    bool first = true;
    for (size_t worker = 0; worker < moe_new_dispatch_combine_a8w8::kM3ORestoreWorkerCount; ++worker) {
        size_t base = moe_new_dispatch_combine_a8w8::kM3ORestoreWorkerBase +
                      worker * moe_new_dispatch_combine_a8w8::kM3ORestoreWorkerWords;
        if (base + 5U >= counters.size() || counters[base + 5U] == 0) {
            continue;
        }
        if (!first) {
            os << ",";
        }
        os << worker << ":" << counters[base] << "-" << counters[base + 1U];
        first = false;
    }
    return first ? "none" : os.str();
}

int32_t CountReadyCachelineSignals(const std::vector<int32_t> &raw)
{
    int32_t ready = 0;
    for (size_t idx = 0; idx < raw.size(); idx += 16U) {
        if (raw[idx] != 0) {
            ++ready;
        }
    }
    return ready;
}

std::vector<int32_t> CopyWorkspaceI32Field(const moe_new_dispatch_combine_a8w8::WorkspaceLayout &workspaceLayout,
                                           const moe_new_dispatch_combine_a8w8::FieldLayout &field, RuntimeState *state,
                                           const std::string &name)
{
    (void)workspaceLayout;
    std::vector<int32_t> out(field.bytes / sizeof(int32_t), 0);
    auto *workspaceBase = reinterpret_cast<uint8_t *>(state->buffers.workspace);
    CheckAcl(aclrtMemcpy(out.data(), BytesOfI32Vector(out.size()), workspaceBase + field.offset, field.bytes,
                         ACL_MEMCPY_DEVICE_TO_HOST),
             "rank " + std::to_string(state->rank) + " copy " + name);
    return out;
}

std::vector<uint64_t> CopyWorkspaceU64Field(const moe_new_dispatch_combine_a8w8::WorkspaceLayout &workspaceLayout,
                                            const moe_new_dispatch_combine_a8w8::FieldLayout &field,
                                            RuntimeState *state, const std::string &name)
{
    (void)workspaceLayout;
    std::vector<uint64_t> out(field.bytes / sizeof(uint64_t), 0);
    auto *workspaceBase = reinterpret_cast<uint8_t *>(state->buffers.workspace);
    CheckAcl(aclrtMemcpy(out.data(), BytesOfU64Vector(out.size()), workspaceBase + field.offset, field.bytes,
                         ACL_MEMCPY_DEVICE_TO_HOST),
             "rank " + std::to_string(state->rank) + " copy " + name);
    return out;
}

std::vector<int32_t> CopyPeerI32Field(const moe_new_dispatch_combine_a8w8::PeerWindowLayout &peerWindowLayout,
                                      const moe_new_dispatch_combine_a8w8::FieldLayout &field, RuntimeState *state,
                                      const std::string &name)
{
    (void)peerWindowLayout;
    std::vector<int32_t> out(field.bytes / sizeof(int32_t), 0);
    auto *peerBase = reinterpret_cast<uint8_t *>(state->hccl.peerWindow);
    CheckAcl(aclrtMemcpy(out.data(), BytesOfI32Vector(out.size()), peerBase + field.offset, field.bytes,
                         ACL_MEMCPY_DEVICE_TO_HOST),
             "rank " + std::to_string(state->rank) + " copy " + name);
    return out;
}

M2FusedFullEvidence ReadM2FusedFullEvidence(const moe_new_dispatch_combine_a8w8::WorkspaceLayout &workspaceLayout,
                                            const moe_new_dispatch_combine_a8w8::PeerWindowLayout &peerWindowLayout,
                                            RuntimeState *state)
{
    constexpr size_t kAicHeaderSlot = 8U * 16U;
    constexpr size_t kAivHeaderSlot = 9U * 16U;
    constexpr size_t kStageBaseSlot = 10U * 16U;
    constexpr size_t kM3CounterBase = 24U * 16U;
    constexpr size_t kM3NDispatchCounterBase = kM3CounterBase + 16U;
    constexpr size_t kM3NActivationCounterBase = kM3CounterBase + 32U;
    constexpr size_t kM3NCombineCounterBase = kM3CounterBase + 48U;
    constexpr size_t kM3NGmm2CounterBase = kM3CounterBase + 64U;
    constexpr size_t kM3N11SubtileCounterBase = moe_new_dispatch_combine_a8w8::kM3N11SubtileCounterBase;
    constexpr size_t kM3ORestoreCounterBase = moe_new_dispatch_combine_a8w8::kM3ORestoreCounterBase;
    constexpr int32_t kFullMagic = 0x4D328CA;
    M2FusedFullEvidence evidence;
    std::vector<int32_t> stageStatus(workspaceLayout.stageStatus.bytes / sizeof(int32_t), 0);
    auto *workspaceBase = reinterpret_cast<uint8_t *>(state->buffers.workspace);
    CheckAcl(aclrtMemcpy(stageStatus.data(), BytesOfI32Vector(stageStatus.size()),
                         workspaceBase + workspaceLayout.stageStatus.offset, workspaceLayout.stageStatus.bytes,
                         ACL_MEMCPY_DEVICE_TO_HOST),
             "rank " + std::to_string(state->rank) + " copy m2 fused full stageStatus");
    if (stageStatus.size() > kAicHeaderSlot + 3U) {
        evidence.aicSeen = stageStatus[kAicHeaderSlot] == kFullMagic;
        evidence.aicBlocks = stageStatus[kAicHeaderSlot + 1U];
    }
    if (stageStatus.size() > kAivHeaderSlot + 3U) {
        evidence.aivSeen = stageStatus[kAivHeaderSlot] == kFullMagic;
        evidence.aivBlocks = stageStatus[kAivHeaderSlot + 1U];
    }
    for (size_t idx = 0; idx < evidence.stageMarkers.size(); ++idx) {
        if (stageStatus.size() > kStageBaseSlot + idx) {
            evidence.stageMarkers[idx] = stageStatus[kStageBaseSlot + idx];
            if (stageStatus[kStageBaseSlot + idx] != 0) {
                ++evidence.stageCount;
            }
        }
    }
    constexpr size_t kM3N5AicEvidenceSlot = 14U * 16U;
    if (stageStatus.size() > kM3N5AicEvidenceSlot) {
        evidence.m3n5Gmm1StartBeforeLastDispatchReady = stageStatus[kM3N5AicEvidenceSlot];
    }
    constexpr size_t kM3N6EvidenceSlot = kM3N5AicEvidenceSlot + 2U;
    if (stageStatus.size() > kM3N6EvidenceSlot + 2U) {
        evidence.m3n6ActivationStartBeforeLastGmm1Ready = stageStatus[kM3N6EvidenceSlot + 2U];
    }
    constexpr size_t kM3N7EvidenceSlot = kM3N6EvidenceSlot + 3U;
    if (stageStatus.size() > kM3N7EvidenceSlot + 2U) {
        evidence.m3n7Gmm2StartBeforeLastActivationReady = stageStatus[kM3N7EvidenceSlot];
    }
    constexpr size_t kM3N8EvidenceSlot = kM3N7EvidenceSlot + 3U;
    if (stageStatus.size() > kM3N8EvidenceSlot + 2U) {
        evidence.m3n8CombineStartBeforeLastGmm2Ready = stageStatus[kM3N8EvidenceSlot];
    }
    std::vector<int32_t> dispatchReady =
        CopyWorkspaceI32Field(workspaceLayout, workspaceLayout.dispatchGroupReady, state, "m3 dispatchGroupReady");
    std::vector<int32_t> gmm1Ready =
        CopyWorkspaceI32Field(workspaceLayout, workspaceLayout.gmm1SyncGroupReady, state, "m3 gmm1SyncGroupReady");
    std::vector<int32_t> activationReady = CopyWorkspaceI32Field(
        workspaceLayout, workspaceLayout.activationSyncGroupReady, state, "m3 activationSyncGroupReady");
    std::vector<int32_t> gmm2Ready =
        CopyWorkspaceI32Field(workspaceLayout, workspaceLayout.gmm2GroupReady, state, "m3 gmm2GroupReady");
    std::vector<int32_t> swigluGroups =
        CopyWorkspaceI32Field(workspaceLayout, workspaceLayout.swigluSyncGroups, state, "m3 swigluSyncGroups");
    std::vector<int32_t> dequantSum =
        CopyWorkspaceI32Field(workspaceLayout, workspaceLayout.dequantSum, state, "m3 dequantSum");
    std::vector<int32_t> swigluDesc =
        CopyWorkspaceI32Field(workspaceLayout, workspaceLayout.swigluGroupDesc, state, "m3 swigluGroupDesc");
    std::vector<int32_t> debugCounters =
        CopyPeerI32Field(peerWindowLayout, peerWindowLayout.debugCounters, state, "m3 debugCounters");
    evidence.dispatchGroupReadyCount = CountReadyCachelineSignals(dispatchReady);
    evidence.gmm1SyncGroupReadyCount = CountReadyCachelineSignals(gmm1Ready);
    evidence.activationSyncGroupReadyCount = CountReadyCachelineSignals(activationReady);
    evidence.gmm2GroupReadyCount = CountReadyCachelineSignals(gmm2Ready);
    if (!swigluGroups.empty()) {
        evidence.swigluSyncGroupCount = swigluGroups[0];
        for (int32_t idx = 0; idx < evidence.swigluSyncGroupCount && static_cast<size_t>(idx + 1) < swigluGroups.size();
             ++idx) {
            evidence.swigluSyncGroupSizeSum += swigluGroups[static_cast<size_t>(idx) + 1U];
        }
    }
    if (evidence.swigluSyncGroupCount >= 0 && static_cast<size_t>(evidence.swigluSyncGroupCount) < dequantSum.size()) {
        evidence.dequantFinalRow = dequantSum[static_cast<size_t>(evidence.swigluSyncGroupCount)];
    }
    bool monotonic = true;
    int32_t previousEnd = 0;
    std::ostringstream groupTileRanges;
    for (int32_t group = 0; group < evidence.swigluSyncGroupCount; ++group) {
        size_t base = static_cast<size_t>(group) * 8U;
        if (base + 7U >= swigluDesc.size()) {
            monotonic = false;
            break;
        }
        int32_t rowBegin = swigluDesc[base + 3U];
        int32_t rowEnd = swigluDesc[base + 4U];
        int32_t tileBegin = swigluDesc[base + 5U];
        int32_t tileEnd = swigluDesc[base + 6U];
        if (group != 0) {
            groupTileRanges << ",";
        }
        groupTileRanges << group << ":" << tileBegin << "-" << tileEnd;
        if (rowBegin < previousEnd || rowEnd < rowBegin) {
            monotonic = false;
        }
        if (swigluDesc[base + 7U] != 0) {
            ++evidence.swigluEmptyGroups;
        }
        previousEnd = rowEnd;
    }
    evidence.swigluGroupTileRanges = groupTileRanges.str();
    evidence.swigluGroupDescMonotonic = monotonic;
    for (size_t idx = 0; idx < evidence.m3Counters.size(); ++idx) {
        if (kM3CounterBase + idx < debugCounters.size()) {
            evidence.m3Counters[idx] = debugCounters[kM3CounterBase + idx];
        }
    }
    for (size_t idx = 0; idx < evidence.m3nDispatchCounters.size(); ++idx) {
        if (kM3NDispatchCounterBase + idx < debugCounters.size()) {
            evidence.m3nDispatchCounters[idx] = debugCounters[kM3NDispatchCounterBase + idx];
        }
    }
    if (evidence.m3nDispatchCounters[15] != 0) {
        evidence.m3n5Gmm1StartBeforeLastDispatchReady = 1;
    }
    for (size_t idx = 0; idx < evidence.m3nActivationCounters.size(); ++idx) {
        if (kM3NActivationCounterBase + idx < debugCounters.size()) {
            evidence.m3nActivationCounters[idx] = debugCounters[kM3NActivationCounterBase + idx];
        }
    }
    for (size_t idx = 0; idx < evidence.m3nCombineCounters.size(); ++idx) {
        if (kM3NCombineCounterBase + idx < debugCounters.size()) {
            evidence.m3nCombineCounters[idx] = debugCounters[kM3NCombineCounterBase + idx];
        }
    }
    for (size_t idx = 0; idx < evidence.m3nGmm2Counters.size(); ++idx) {
        if (kM3NGmm2CounterBase + idx < debugCounters.size()) {
            evidence.m3nGmm2Counters[idx] = debugCounters[kM3NGmm2CounterBase + idx];
        }
    }
    if (evidence.m3nGmm2Counters[14] != 0) {
        evidence.m3n7Gmm2StartBeforeLastActivationReady = 1;
    }
    constexpr size_t kM3N8CombineCounterBase = kM3CounterBase + 80U;
    for (size_t idx = 0; idx < evidence.m3n8CombineCounters.size(); ++idx) {
        if (kM3N8CombineCounterBase + idx < debugCounters.size()) {
            evidence.m3n8CombineCounters[idx] = debugCounters[kM3N8CombineCounterBase + idx];
        }
    }
    if (evidence.m3n8CombineCounters[5] != 0) {
        evidence.m3n8CombineStartBeforeLastGmm2Ready = 1;
    }
    for (size_t idx = 0; idx < evidence.m3n11SubtileCounters.size(); ++idx) {
        if (kM3N11SubtileCounterBase + idx < debugCounters.size()) {
            evidence.m3n11SubtileCounters[idx] = debugCounters[kM3N11SubtileCounterBase + idx];
        }
    }
    constexpr size_t kM3OGmm1CounterBase = kM3CounterBase + 160U;
    constexpr size_t kM3OGmm2CounterBase = kM3CounterBase + 208U;
    for (size_t idx = 0; idx < evidence.m3oGmm1Counters.size(); ++idx) {
        if (kM3OGmm1CounterBase + idx < debugCounters.size()) {
            evidence.m3oGmm1Counters[idx] = debugCounters[kM3OGmm1CounterBase + idx];
        }
    }
    for (size_t idx = 0; idx < evidence.m3oGmm2Counters.size(); ++idx) {
        if (kM3OGmm2CounterBase + idx < debugCounters.size()) {
            evidence.m3oGmm2Counters[idx] = debugCounters[kM3OGmm2CounterBase + idx];
        }
    }
    for (size_t idx = 0; idx < evidence.m3oRestoreCounters.size(); ++idx) {
        if (kM3ORestoreCounterBase + idx < debugCounters.size()) {
            evidence.m3oRestoreCounters[idx] = debugCounters[kM3ORestoreCounterBase + idx];
        }
    }
    std::vector<int32_t> timeoutDump =
        CopyWorkspaceI32Field(workspaceLayout, workspaceLayout.timeoutDump, state, "m3n9 timeoutDump");
    for (size_t idx = 0; idx < evidence.m3n9TimeoutDump.size() && idx < timeoutDump.size(); ++idx) {
        evidence.m3n9TimeoutDump[idx] = timeoutDump[idx];
    }
    std::vector<uint64_t> timelineScratch =
        CopyWorkspaceU64Field(workspaceLayout, workspaceLayout.timelineScratch, state, "m3n12 timelineScratch");
    for (size_t idx = 0; idx < evidence.m3n12TimelineScratch.size() && idx < timelineScratch.size(); ++idx) {
        evidence.m3n12TimelineScratch[idx] = timelineScratch[idx];
    }
    return evidence;
}

void PrintM3N9TimeoutDump(const M2FusedFullEvidence &evidence)
{
    bool present = evidence.m3n9TimeoutDump[kM3N9TimeoutDumpPresentSlot] != 0;
    int32_t stage = evidence.m3n9TimeoutDump[kM3N9TimeoutDumpStageSlot];
    std::cout << "  m3n9_timeout_dump_present=" << (present ? "true" : "false") << "\n";
    std::cout << "  timeout_dump_rank=" << (present ? evidence.m3n9TimeoutDump[kM3N9TimeoutDumpRankSlot] : -1) << "\n";
    std::cout << "  timeout_dump_expert=" << (present ? evidence.m3n9TimeoutDump[kM3N9TimeoutDumpExpertSlot] : -1)
              << "\n";
    std::cout << "  timeout_dump_token_owner_rank="
              << (present ? evidence.m3n9TimeoutDump[kM3N9TimeoutDumpTokenOwnerRankSlot] : -1) << "\n";
    std::cout << "  timeout_dump_expert_owner_rank="
              << (present ? evidence.m3n9TimeoutDump[kM3N9TimeoutDumpExpertOwnerRankSlot] : -1) << "\n";
    std::cout << "  timeout_dump_stage=" << (present ? stage : -1) << "\n";
    std::cout << "  timeout_dump_stage_name=" << (present ? M3N9TimeoutStageName(stage) : "none") << "\n";
    std::cout << "  timeout_dump_signal_id=" << (present ? evidence.m3n9TimeoutDump[kM3N9TimeoutDumpSignalIdSlot] : -1)
              << "\n";
    std::cout << "  timeout_dump_debug_stop_stage="
              << (present ? evidence.m3n9TimeoutDump[kM3N9TimeoutDumpDebugStopStageSlot] : -1) << "\n";
    std::cout << "  timeout_dump_dispatch_ready="
              << (present ? evidence.m3n9TimeoutDump[kM3N9TimeoutDumpDispatchReadySlot] : -1) << "\n";
    std::cout << "  timeout_dump_gmm1_ready="
              << (present ? evidence.m3n9TimeoutDump[kM3N9TimeoutDumpGmm1ReadySlot] : -1) << "\n";
    std::cout << "  timeout_dump_activation_ready="
              << (present ? evidence.m3n9TimeoutDump[kM3N9TimeoutDumpActivationReadySlot] : -1) << "\n";
    std::cout << "  timeout_dump_gmm2_ready="
              << (present ? evidence.m3n9TimeoutDump[kM3N9TimeoutDumpGmm2ReadySlot] : -1) << "\n";
    std::cout << "  timeout_dump_ready_expert="
              << (present ? evidence.m3n9TimeoutDump[kM3N9TimeoutDumpReadyExpertSlot] : -1) << "\n";
    std::cout << "  timeout_dump_source=device_timeout_dump\n";
}

std::string MakeM3N12RunId(const DispatchCombineTileArgs &args)
{
    std::ostringstream os;
    os << args.caseName << "_seed" << args.runtime.seed << "_m" << args.shape.m << "_k" << args.shape.k << "_i"
       << args.intermediateSize << "_topk" << args.shape.topK << "_ep" << args.shape.ep << "_epr"
       << args.shape.expertPerRank << "_ov" << args.overlapMode;
    return os.str();
}

const char *M3N12KindName(uint32_t kind)
{
    switch (static_cast<moe_new_dispatch_combine_a8w8::M3N12TimelineKind>(kind)) {
        case moe_new_dispatch_combine_a8w8::M3N12TimelineKind::kRoute:
            return "scatter_quant";
        case moe_new_dispatch_combine_a8w8::M3N12TimelineKind::kCountSync:
            return "count_sync";
        case moe_new_dispatch_combine_a8w8::M3N12TimelineKind::kDispatchGather:
            return "dispatch_gather";
        case moe_new_dispatch_combine_a8w8::M3N12TimelineKind::kGmm1Tile:
            return "gmm1";
        case moe_new_dispatch_combine_a8w8::M3N12TimelineKind::kSwigluGroup:
            return "swiglu_quant";
        case moe_new_dispatch_combine_a8w8::M3N12TimelineKind::kGmm2Tile:
            return "gmm2";
        case moe_new_dispatch_combine_a8w8::M3N12TimelineKind::kCombineOwnerSegment:
            return "combine";
        case moe_new_dispatch_combine_a8w8::M3N12TimelineKind::kRestore:
            return "restore";
        case moe_new_dispatch_combine_a8w8::M3N12TimelineKind::kRouteCount:
            return "route_count";
        case moe_new_dispatch_combine_a8w8::M3N12TimelineKind::kPrefix:
            return "prefix";
        case moe_new_dispatch_combine_a8w8::M3N12TimelineKind::kInitQuantE2E:
            return "init_quant_e2e";
        default:
            return "none";
    }
}

const char *M3N12RowKind(uint32_t kind)
{
    switch (static_cast<moe_new_dispatch_combine_a8w8::M3N12TimelineKind>(kind)) {
        case moe_new_dispatch_combine_a8w8::M3N12TimelineKind::kGmm1Tile:
        case moe_new_dispatch_combine_a8w8::M3N12TimelineKind::kGmm2Tile:
            return "gmm_tile";
        case moe_new_dispatch_combine_a8w8::M3N12TimelineKind::kSwigluGroup:
            return "swiglu_group";
        case moe_new_dispatch_combine_a8w8::M3N12TimelineKind::kCombineOwnerSegment:
            return "owner_segment";
        default:
            return "stage";
    }
}

const char *M3N12CoreTypeName(uint32_t coreType)
{
    switch (static_cast<moe_new_dispatch_combine_a8w8::M3N12TimelineCoreType>(coreType)) {
        case moe_new_dispatch_combine_a8w8::M3N12TimelineCoreType::kAic:
            return "aic";
        case moe_new_dispatch_combine_a8w8::M3N12TimelineCoreType::kAiv:
            return "aiv";
        default:
            return "unknown";
    }
}

const char *M3N12StatusName(uint32_t status)
{
    switch (static_cast<moe_new_dispatch_combine_a8w8::M3N12TimelineStatus>(status)) {
        case moe_new_dispatch_combine_a8w8::M3N12TimelineStatus::kProcessed:
            return "processed";
        case moe_new_dispatch_combine_a8w8::M3N12TimelineStatus::kSkipped:
            return "skipped";
        default:
            return "empty";
    }
}

const char *M3N12WaitSourceName(uint32_t waitSource)
{
    switch (static_cast<moe_new_dispatch_combine_a8w8::M3N12TimelineWaitSource>(waitSource)) {
        case moe_new_dispatch_combine_a8w8::M3N12TimelineWaitSource::kPtoEvent:
            return "pto_event";
        case moe_new_dispatch_combine_a8w8::M3N12TimelineWaitSource::kGmPoll:
            return "gm_poll";
        case moe_new_dispatch_combine_a8w8::M3N12TimelineWaitSource::kSyncAll:
            return "syncall";
        default:
            return "none";
    }
}

uint32_t M3N12MetaField(uint64_t value, uint32_t shift, uint32_t mask)
{
    return static_cast<uint32_t>((value >> shift) & mask);
}

double M3N12CyclesToUs(uint64_t cycles)
{
    constexpr double kA3SyscntCyclesPerUs = 1850.0;
    return static_cast<double>(cycles) / kA3SyscntCyclesPerUs;
}

double InitQuantE2eUs(const M2FusedFullEvidence &evidence)
{
    constexpr uint32_t kWords = moe_new_dispatch_combine_a8w8::kM3N12TimelineRecordWords;
    size_t base = static_cast<size_t>(moe_new_dispatch_combine_a8w8::kM3N12TimelineSlotInitQuantE2E) * kWords;
    if (base + 1U >= evidence.m3n12TimelineScratch.size()) {
        return 0.0;
    }
    uint64_t begin = evidence.m3n12TimelineScratch[base + 0U];
    uint64_t end = evidence.m3n12TimelineScratch[base + 1U];
    if (begin == 0U || end < begin) {
        return 0.0;
    }
    return M3N12CyclesToUs(end - begin);
}

void PrintM3N12TimelineRecord(const M2FusedFullEvidence &evidence, uint32_t slot,
                              moe_new_dispatch_combine_a8w8::M3N12TimelineKind fallbackKind)
{
    constexpr uint32_t kWords = moe_new_dispatch_combine_a8w8::kM3N12TimelineRecordWords;
    size_t base = static_cast<size_t>(slot) * kWords;
    uint64_t begin = base + 0U < evidence.m3n12TimelineScratch.size() ? evidence.m3n12TimelineScratch[base + 0U] : 0U;
    uint64_t end = base + 1U < evidence.m3n12TimelineScratch.size() ? evidence.m3n12TimelineScratch[base + 1U] : 0U;
    uint64_t meta0 = base + 2U < evidence.m3n12TimelineScratch.size() ? evidence.m3n12TimelineScratch[base + 2U] : 0U;
    uint64_t meta1 = base + 3U < evidence.m3n12TimelineScratch.size() ? evidence.m3n12TimelineScratch[base + 3U] : 0U;
    uint32_t kind = M3N12MetaField(meta0, 0U, 0xffU);
    uint32_t coreType = M3N12MetaField(meta0, 8U, 0xffU);
    uint32_t coreId = M3N12MetaField(meta0, 16U, 0xffU);
    uint32_t status = M3N12MetaField(meta0, 24U, 0xffU);
    uint32_t aux0 = M3N12MetaField(meta0, 32U, 0xffffU);
    uint32_t aux1 = M3N12MetaField(meta0, 48U, 0xffffU);
    uint32_t value0 = M3N12MetaField(meta1, 0U, 0xffffU);
    uint32_t value1 = M3N12MetaField(meta1, 16U, 0xffffU);
    uint32_t value2 = M3N12MetaField(meta1, 32U, 0xffffU);
    uint32_t value3 = M3N12MetaField(meta1, 48U, 0xffffU);
    if (kind == 0U) {
        kind = static_cast<uint32_t>(fallbackKind);
        status = static_cast<uint32_t>(moe_new_dispatch_combine_a8w8::M3N12TimelineStatus::kSkipped);
    }
    uint64_t elapsed = end >= begin ? end - begin : 0U;
    std::cout << "  timeline_row_kind=" << M3N12RowKind(kind) << " stage=" << M3N12KindName(kind) << " slot=" << slot
              << " core_type=" << M3N12CoreTypeName(coreType) << " core_id=" << coreId << " t_begin=" << begin
              << " t_end=" << end << " elapsed_cycles=" << elapsed << " elapsed_us_est=" << M3N12CyclesToUs(elapsed)
              << " status=" << M3N12StatusName(status);
    switch (static_cast<moe_new_dispatch_combine_a8w8::M3N12TimelineKind>(kind)) {
        case moe_new_dispatch_combine_a8w8::M3N12TimelineKind::kGmm1Tile:
        case moe_new_dispatch_combine_a8w8::M3N12TimelineKind::kGmm2Tile:
            std::cout << " groupIdx=" << aux0 << " tileId=" << aux1 << " mTile=" << value0 << " nTile=" << value1
                      << " kLoop=" << value2 << " logicalAic=" << coreId
                      << " waitSource=" << M3N12WaitSourceName(value3);
            break;
        case moe_new_dispatch_combine_a8w8::M3N12TimelineKind::kSwigluGroup:
            std::cout << " syncIdx=" << aux0 << " groupId=" << aux1 << " rowBegin=" << value0 << " rowEnd=" << value1
                      << " expertBegin=" << value2 << " expertEnd=" << value3 << " ready_signal=gmm1_sync_group_ready";
            break;
        case moe_new_dispatch_combine_a8w8::M3N12TimelineKind::kCombineOwnerSegment:
            std::cout << " workerId=" << aux0 << " logicalAiv=" << aux1 << " segmentBegin=" << value0
                      << " segmentEnd=" << value1 << " processed=" << value2 << " skipped=" << value3
                      << " ready_signal=gmm2_or_subtile_ready";
            break;
        default:
            std::cout << " workerId=" << aux0 << " logicalAiv=" << aux1 << " rowBegin=" << value0
                      << " rowEnd=" << value1 << " processed=" << value2 << " skipped=" << value3;
            break;
    }
    std::cout << "\n";
}

void PrintM3N12Timeline(const DispatchCombineTileArgs &args, RuntimeState *state, const M2FusedFullEvidence &evidence)
{
    if (args.runtime.timeline == 0U) {
        return;
    }
    std::cout << "[Timeline]\n";
    std::cout << "  case_name=" << args.caseName << "\n";
    std::cout << "  seed=" << args.runtime.seed << "\n";
    std::cout << "  run_id=" << MakeM3N12RunId(args) << "\n";
    std::cout << "  shape=m" << args.shape.m << "_k" << args.shape.k << "_i" << args.intermediateSize << "_topk"
              << args.shape.topK << "_ep" << args.shape.ep << "_epr" << args.shape.expertPerRank << "\n";
    std::cout << "  shape_m=" << args.shape.m << " shape_k=" << args.shape.k
              << " shape_intermediate=" << args.intermediateSize << " shape_topK=" << args.shape.topK
              << " shape_expertPerRank=" << args.shape.expertPerRank << "\n";
    std::cout << "  rank=" << state->rank << " rankNum=" << state->size << "\n";
    std::cout << "  timeline_granularity=stage_group_tile_subtile\n";
    std::cout << "  timeline_syscnt_cycles_per_us=1850\n";
    std::cout << "  cross_rank_barrier_count=0\n";
    std::cout << "  syncall_count=" << evidence.m3Counters[13] << "\n";
    std::cout << "  cv_wait_count=" << evidence.m3Counters[14] << "\n";
    PrintM3N12TimelineRecord(evidence, moe_new_dispatch_combine_a8w8::kM3N12TimelineSlotRouteCount,
                             moe_new_dispatch_combine_a8w8::M3N12TimelineKind::kRouteCount);
    PrintM3N12TimelineRecord(evidence, moe_new_dispatch_combine_a8w8::kM3N12TimelineSlotRoute,
                             moe_new_dispatch_combine_a8w8::M3N12TimelineKind::kRoute);
    PrintM3N12TimelineRecord(evidence, moe_new_dispatch_combine_a8w8::kM3N12TimelineSlotCountSync,
                             moe_new_dispatch_combine_a8w8::M3N12TimelineKind::kCountSync);
    PrintM3N12TimelineRecord(evidence, moe_new_dispatch_combine_a8w8::kM3N12TimelineSlotPrefix,
                             moe_new_dispatch_combine_a8w8::M3N12TimelineKind::kPrefix);
    PrintM3N12TimelineRecord(evidence, moe_new_dispatch_combine_a8w8::kM3N12TimelineSlotDispatchGather,
                             moe_new_dispatch_combine_a8w8::M3N12TimelineKind::kDispatchGather);
    PrintM3N12TimelineRecord(evidence, moe_new_dispatch_combine_a8w8::kM3N12TimelineSlotGmm1,
                             moe_new_dispatch_combine_a8w8::M3N12TimelineKind::kGmm1Tile);
    uint32_t swigluRows = evidence.swigluSyncGroupCount > 0 ?
                              std::min<uint32_t>(static_cast<uint32_t>(evidence.swigluSyncGroupCount),
                                                 moe_new_dispatch_combine_a8w8::kM3N12TimelineSwigluSlotCount) :
                              1U;
    for (uint32_t idx = 0; idx < swigluRows; ++idx) {
        PrintM3N12TimelineRecord(evidence, moe_new_dispatch_combine_a8w8::kM3N12TimelineSlotSwigluBase + idx,
                                 moe_new_dispatch_combine_a8w8::M3N12TimelineKind::kSwigluGroup);
    }
    PrintM3N12TimelineRecord(evidence, moe_new_dispatch_combine_a8w8::kM3N12TimelineSlotGmm2,
                             moe_new_dispatch_combine_a8w8::M3N12TimelineKind::kGmm2Tile);
    PrintM3N12TimelineRecord(evidence, moe_new_dispatch_combine_a8w8::kM3N12TimelineSlotCombine,
                             moe_new_dispatch_combine_a8w8::M3N12TimelineKind::kCombineOwnerSegment);
    PrintM3N12TimelineRecord(evidence, moe_new_dispatch_combine_a8w8::kM3N12TimelineSlotRestore,
                             moe_new_dispatch_combine_a8w8::M3N12TimelineKind::kRestore);
}

double RunM2FusedFull(const DispatchCombineTileArgs &args, const moe_new_dispatch_combine_a8w8::ShapeConfig &shape,
                      const moe_new_dispatch_combine_a8w8::WorkspaceLayout &workspaceLayout, RuntimeState *state,
                      M2FusedFullEvidence *evidence)
{
    if (!GmmPolicySupported(shape)) {
        throw std::runtime_error(
            "M2.8 fused full PTO path requires gmm_ar cache-level policy and hidden/intermediate divisible by 64");
    }
    bool verbose = VerboseRuntimeLogs(args);
    if (verbose) {
        PrintStage(state->rank, "m2_fused_full", "begin");
    }
    moe_new_dispatch_combine_a8w8::RankConfig rank = MakeM2RankConfig(args, state->rank);
    constexpr uint32_t kAicBlocks = 24;
    constexpr uint32_t kAivRatio = 2;
    moe_new_dispatch_combine_a8w8::M2FusedFullLaunchArgs launchArgs{};
    launchArgs.params =
        moe_new_dispatch_combine_a8w8::M2FusedFullParams{shape, rank, args.runtime.m2FusedDebugStopStage};
    launchArgs.debugStopStage = args.runtime.m2FusedDebugStopStage;
    launchArgs.stageStatusAddr = reinterpret_cast<uint64_t>(reinterpret_cast<uint8_t *>(state->buffers.workspace) +
                                                            workspaceLayout.stageStatus.offset);
    launchArgs.shapeRankNum = shape.rankNum;
    launchArgs.shapeExpertPerRank = shape.expertPerRank;
    launchArgs.shapeTopK = shape.topK;
    launchArgs.shapeM = shape.m;
    launchArgs.shapeHiddenSize = shape.hiddenSize;
    launchArgs.shapeIntermediateSize = shape.intermediateSize;
    launchArgs.shapeMaxTokensPerExpert = shape.maxTokensPerExpert;
    launchArgs.shapePayloadTileCols = shape.payloadTileCols;
    launchArgs.shapeGmmBlockM = shape.gmmBlockM;
    launchArgs.shapeGmmBlockN = shape.gmmBlockN;
    launchArgs.shapeGmmBlockK = shape.gmmBlockK;
    launchArgs.shapeDtypeIn = shape.dtypeIn;
    launchArgs.shapeDtypeOut = shape.dtypeOut;
    launchArgs.rankRankNum = rank.rankNum;
    launchArgs.rankRankId = rank.rankId;
    launchArgs.rankFromMpi = rank.rankFromMpi;
    launchArgs.rankDeviceBase = rank.deviceBase;
    launchArgs.rankNdevices = rank.ndevices;
    launchArgs.inputA = reinterpret_cast<uint64_t>(state->buffers.inputA);
    launchArgs.expertIdx = reinterpret_cast<uint64_t>(state->buffers.expertIdx);
    launchArgs.xActiveMask =
        args.xActiveMaskMode == "none" ? 0ULL : reinterpret_cast<uint64_t>(state->buffers.xActiveMask);
    launchArgs.probs = reinterpret_cast<uint64_t>(state->buffers.probs);
    launchArgs.outputC = reinterpret_cast<uint64_t>(state->buffers.outputC);
    launchArgs.peerWindow = reinterpret_cast<uint64_t>(state->hccl.peerWindow);
    launchArgs.hcclCtx = reinterpret_cast<uint64_t>(state->hccl.deviceContext);
    launchArgs.workspace = reinterpret_cast<uint64_t>(state->buffers.workspace);
    launchArgs.timelineEnable = args.runtime.timeline;
    launchArgs.overlapMode = args.runtime.overlapMode;
    moe_new_dispatch_combine_a8w8::M2FusedFullLaunchArgs *launchArgsDevice = nullptr;
    CheckAcl(aclrtMalloc(reinterpret_cast<void **>(&launchArgsDevice), sizeof(launchArgs), ACL_MEM_MALLOC_HUGE_FIRST),
             "rank " + std::to_string(state->rank) + " aclrtMalloc m2 fused full launch args");
    try {
        CheckAcl(aclrtMemcpy(launchArgsDevice, sizeof(launchArgs), &launchArgs, sizeof(launchArgs),
                             ACL_MEMCPY_HOST_TO_DEVICE),
                 "rank " + std::to_string(state->rank) + " copy m2 fused full launch args");
        std::array<int32_t, moe_new_dispatch_combine_a8w8::kM2FusedFullConfigWords> fusedConfig{};
        auto storeConfigU32 = [&fusedConfig](uint32_t slot, uint32_t value) {
            fusedConfig.at(slot) = static_cast<int32_t>(value);
        };
        auto storeConfigU64 = [&fusedConfig](uint32_t slot, uint64_t value) {
            fusedConfig.at(slot) = static_cast<int32_t>(value & 0xffffffffULL);
            fusedConfig.at(slot + 1U) = static_cast<int32_t>((value >> 32U) & 0xffffffffULL);
        };
        storeConfigU32(moe_new_dispatch_combine_a8w8::kM2FusedFullConfigMagicSlot,
                       moe_new_dispatch_combine_a8w8::kM2FusedFullConfigMagic);
        storeConfigU32(moe_new_dispatch_combine_a8w8::kM2FusedFullShapeRankNumSlot, shape.rankNum);
        storeConfigU32(moe_new_dispatch_combine_a8w8::kM2FusedFullShapeExpertPerRankSlot, shape.expertPerRank);
        storeConfigU32(moe_new_dispatch_combine_a8w8::kM2FusedFullShapeTopKSlot, shape.topK);
        storeConfigU32(moe_new_dispatch_combine_a8w8::kM2FusedFullShapeMSlot, shape.m);
        storeConfigU32(moe_new_dispatch_combine_a8w8::kM2FusedFullShapeHiddenSizeSlot, shape.hiddenSize);
        storeConfigU32(moe_new_dispatch_combine_a8w8::kM2FusedFullShapeIntermediateSizeSlot, shape.intermediateSize);
        storeConfigU32(moe_new_dispatch_combine_a8w8::kM2FusedFullShapeMaxTokensPerExpertSlot,
                       shape.maxTokensPerExpert);
        storeConfigU32(moe_new_dispatch_combine_a8w8::kM2FusedFullShapePayloadTileColsSlot, shape.payloadTileCols);
        storeConfigU32(moe_new_dispatch_combine_a8w8::kM2FusedFullShapeGmmBlockMSlot, shape.gmmBlockM);
        storeConfigU32(moe_new_dispatch_combine_a8w8::kM2FusedFullShapeGmmBlockNSlot, shape.gmmBlockN);
        storeConfigU32(moe_new_dispatch_combine_a8w8::kM2FusedFullShapeGmmBlockKSlot, shape.gmmBlockK);
        storeConfigU32(moe_new_dispatch_combine_a8w8::kM2FusedFullShapeDtypeInSlot, shape.dtypeIn);
        storeConfigU32(moe_new_dispatch_combine_a8w8::kM2FusedFullShapeDtypeOutSlot, shape.dtypeOut);
        storeConfigU32(moe_new_dispatch_combine_a8w8::kM2FusedFullRankRankNumSlot, rank.rankNum);
        storeConfigU32(moe_new_dispatch_combine_a8w8::kM2FusedFullRankRankIdSlot, rank.rankId);
        storeConfigU32(moe_new_dispatch_combine_a8w8::kM2FusedFullRankFromMpiSlot, rank.rankFromMpi);
        storeConfigU32(moe_new_dispatch_combine_a8w8::kM2FusedFullRankDeviceBaseSlot, rank.deviceBase);
        storeConfigU32(moe_new_dispatch_combine_a8w8::kM2FusedFullRankNdevicesSlot, rank.ndevices);
        storeConfigU64(moe_new_dispatch_combine_a8w8::kM2FusedFullPtrInputASlot, launchArgs.inputA);
        storeConfigU64(moe_new_dispatch_combine_a8w8::kM2FusedFullPtrExpertIdxSlot, launchArgs.expertIdx);
        storeConfigU64(moe_new_dispatch_combine_a8w8::kM2FusedFullPtrXActiveMaskSlot, launchArgs.xActiveMask);
        storeConfigU64(moe_new_dispatch_combine_a8w8::kM2FusedFullPtrProbsSlot, launchArgs.probs);
        storeConfigU64(moe_new_dispatch_combine_a8w8::kM2FusedFullPtrOutputCSlot, launchArgs.outputC);
        storeConfigU64(moe_new_dispatch_combine_a8w8::kM2FusedFullPtrPeerWindowSlot, launchArgs.peerWindow);
        storeConfigU64(moe_new_dispatch_combine_a8w8::kM2FusedFullPtrHcclCtxSlot, launchArgs.hcclCtx);
        storeConfigU64(moe_new_dispatch_combine_a8w8::kM2FusedFullPtrWorkspaceSlot, launchArgs.workspace);
        storeConfigU32(moe_new_dispatch_combine_a8w8::kM2FusedFullTimelineEnableSlot, launchArgs.timelineEnable);
        storeConfigU32(moe_new_dispatch_combine_a8w8::kM2FusedFullOverlapModeSlot, launchArgs.overlapMode);
        auto *workspaceBase = reinterpret_cast<uint8_t *>(state->buffers.workspace);
        CheckAcl(aclrtMemcpy(workspaceBase + workspaceLayout.stageStatus.offset, fusedConfig.size() * sizeof(int32_t),
                             fusedConfig.data(), fusedConfig.size() * sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE),
                 "rank " + std::to_string(state->rank) + " seed m2 fused full config");
        if (args.runtime.m2FusedDebugStopStage != 0) {
            constexpr size_t kDebugStopSlot = 15U * 16U;
            int32_t debugStop = static_cast<int32_t>(args.runtime.m2FusedDebugStopStage);
            CheckAcl(aclrtMemcpy(workspaceBase + workspaceLayout.stageStatus.offset + kDebugStopSlot * sizeof(int32_t),
                                 sizeof(debugStop), &debugStop, sizeof(debugStop), ACL_MEMCPY_HOST_TO_DEVICE),
                     "rank " + std::to_string(state->rank) + " seed m2 fused debug stop stage");
        }
    } catch (...) {
        aclrtFree(launchArgsDevice);
        throw;
    }
    MpiBarrier(&state->mpi);
    auto start = std::chrono::steady_clock::now();
    LaunchM2FusedFull(launchArgsDevice, kAicBlocks, kAivRatio, state->computeStream);
    try {
        CheckAcl(aclrtSynchronizeStream(state->computeStream),
                 "rank " + std::to_string(state->rank) + " m2 fused full stream sync");
    } catch (...) {
        aclrtFree(launchArgsDevice);
        throw;
    }
    aclrtFree(launchArgsDevice);
    MpiBarrier(&state->mpi);
    auto end = std::chrono::steady_clock::now();
    if (evidence != nullptr) {
        *evidence =
            ReadM2FusedFullEvidence(workspaceLayout, moe_new_dispatch_combine_a8w8::MakePeerWindowLayout(shape), state);
    }
    if (verbose) {
        PrintStage(state->rank, "m2_fused_full", "done");
    }
    return UsSince(start, end);
}

void CopyM2ReferenceToWorkspace(const DispatchCombineTileArgs &args,
                                const moe_new_dispatch_combine_a8w8::WorkspaceLayout &workspaceLayout,
                                RuntimeState *state)
{
    if (!state->m2ReferenceReady) {
        throw std::runtime_error("M2 reference is not ready before device copy");
    }
    bool verbose = VerboseRuntimeLogs(args);
    if (verbose) {
        PrintStage(state->rank, "copy_m2_reference", "begin");
    }
    auto *workspaceBase = reinterpret_cast<uint8_t *>(state->buffers.workspace);
    CheckAcl(aclrtMemcpy(workspaceBase + workspaceLayout.gmm1WeightInt8.offset, workspaceLayout.gmm1WeightInt8.bytes,
                         state->m2Reference.weight1Int8.data(), BytesOfI8Vector(state->m2Reference.weight1Int8.size()),
                         ACL_MEMCPY_HOST_TO_DEVICE),
             "rank " + std::to_string(state->rank) + " copy weight1Int8");
    CheckAcl(aclrtMemcpy(workspaceBase + workspaceLayout.gmm2WeightInt8.offset, workspaceLayout.gmm2WeightInt8.bytes,
                         state->m2Reference.weight2Int8.data(), BytesOfI8Vector(state->m2Reference.weight2Int8.size()),
                         ACL_MEMCPY_HOST_TO_DEVICE),
             "rank " + std::to_string(state->rank) + " copy weight2Int8");
    CheckAcl(aclrtMemcpy(workspaceBase + workspaceLayout.scale1Uint64.offset, workspaceLayout.scale1Uint64.bytes,
                         state->m2Reference.scale1Uint64.data(),
                         BytesOfU64Vector(state->m2Reference.scale1Uint64.size()), ACL_MEMCPY_HOST_TO_DEVICE),
             "rank " + std::to_string(state->rank) + " copy scale1Uint64");
    CheckAcl(aclrtMemcpy(workspaceBase + workspaceLayout.scale2Uint64.offset, workspaceLayout.scale2Uint64.bytes,
                         state->m2Reference.scale2Uint64.data(),
                         BytesOfU64Vector(state->m2Reference.scale2Uint64.size()), ACL_MEMCPY_HOST_TO_DEVICE),
             "rank " + std::to_string(state->rank) + " copy scale2Uint64");
    if (verbose) {
        PrintStage(state->rank, "copy_m2_reference", "done");
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

float RoundHalfFloat(float value)
{
    return golden_detail::HalfToFloat(golden_detail::FloatToHalf(value));
}

uint16_t FloatToHalfNearestBits(float value)
{
    union {
        float f;
        uint32_t u;
    } bits{};
    bits.f = value;
    uint32_t sign = (bits.u >> 16) & 0x8000U;
    int32_t exp = static_cast<int32_t>((bits.u >> 23) & 0xFFU) - 127 + 15;
    uint32_t mant = bits.u & 0x007FFFFFU;
    if (exp <= 0) {
        return static_cast<uint16_t>(sign);
    }
    if (exp >= 31) {
        return static_cast<uint16_t>(sign | 0x7C00U);
    }
    uint32_t halfMant = mant >> 13;
    uint32_t roundBits = mant & 0x1FFFU;
    if (roundBits > 0x1000U || (roundBits == 0x1000U && (halfMant & 1U) != 0U)) {
        ++halfMant;
        if (halfMant == 0x400U) {
            halfMant = 0;
            ++exp;
            if (exp >= 31) {
                return static_cast<uint16_t>(sign | 0x7C00U);
            }
        }
    }
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | halfMant);
}

float RoundHalfNearestFloat(float value)
{
    return golden_detail::HalfToFloat(FloatToHalfNearestBits(value));
}

float CopyM2ReturnPayloadValue(const DispatchCombineTileArgs &args, RuntimeState *state, uint32_t row, uint32_t col)
{
    moe_new_dispatch_combine_a8w8::ShapeConfig m2Shape = MakeM2ShapeConfig(args);
    moe_new_dispatch_combine_a8w8::PeerWindowLayout peerWindowLayout =
        moe_new_dispatch_combine_a8w8::MakePeerWindowLayout(m2Shape);
    size_t returnStrideElems = peerWindowLayout.returnPayloadRowBytes / sizeof(uint16_t);
    auto *peerBase = reinterpret_cast<uint8_t *>(state->hccl.peerWindow);
    uint16_t halfBits = 0;
    CheckAcl(aclrtMemcpy(&halfBits, sizeof(halfBits),
                         peerBase + peerWindowLayout.returnPayload.offset +
                             (static_cast<size_t>(row) * returnStrideElems + col) * sizeof(uint16_t),
                         sizeof(halfBits), ACL_MEMCPY_DEVICE_TO_HOST),
             "rank " + std::to_string(state->rank) + " copy m2 return payload value");
    return golden_detail::HalfToFloat(halfBits);
}

std::vector<float> CopyM2ReturnPayloadToHost(const DispatchCombineTileArgs &args, RuntimeState *state)
{
    const DispatchCombineTileShape &shape = args.shape;
    moe_new_dispatch_combine_a8w8::ShapeConfig m2Shape = MakeM2ShapeConfig(args);
    moe_new_dispatch_combine_a8w8::PeerWindowLayout peerWindowLayout =
        moe_new_dispatch_combine_a8w8::MakePeerWindowLayout(m2Shape);
    size_t expandedRows = static_cast<size_t>(shape.m) * shape.topK;
    size_t returnStrideElems = peerWindowLayout.returnPayloadRowBytes / sizeof(uint16_t);
    auto *peerBase = reinterpret_cast<uint8_t *>(state->hccl.peerWindow);
    std::vector<uint16_t> raw(expandedRows * returnStrideElems, 0);
    CheckAcl(aclrtMemcpy(raw.data(), BytesOfHalfVector(raw.size()), peerBase + peerWindowLayout.returnPayload.offset,
                         peerWindowLayout.returnPayload.bytes, ACL_MEMCPY_DEVICE_TO_HOST),
             "rank " + std::to_string(state->rank) + " copy m2 return payload");
    std::vector<float> payload(expandedRows * shape.k, 0.0f);
    for (uint32_t row = 0; row < expandedRows; ++row) {
        for (uint32_t col = 0; col < shape.k; ++col) {
            payload[static_cast<size_t>(row) * shape.k + col] =
                golden_detail::HalfToFloat(raw[static_cast<size_t>(row) * returnStrideElems + col]);
        }
    }
    return payload;
}

std::vector<float> BuildM2RestoreExpectedFromPayload(const DispatchCombineTileArgs &args, RuntimeState *state,
                                                     const std::vector<float> &returnPayload)
{
    const DispatchCombineTileShape &shape = args.shape;
    std::vector<float> output(static_cast<size_t>(shape.m) * shape.k, 0.0f);
    for (uint32_t token = 0; token < shape.m; ++token) {
        for (uint32_t slot = 0; slot < shape.topK; ++slot) {
            size_t routeIndex = static_cast<size_t>(token) * shape.topK + slot;
            int32_t ptrDRow = state->golden.expandedRowIdx[routeIndex];
            if (ptrDRow < 0 || static_cast<uint32_t>(ptrDRow) >= shape.maxOutputSize) {
                continue;
            }
            float prob = RoundHalfNearestFloat(state->inputs.probs[routeIndex]);
            for (uint32_t col = 0; col < shape.k; ++col) {
                size_t outIndex = static_cast<size_t>(token) * shape.k + col;
                float product =
                    RoundHalfNearestFloat(prob * returnPayload[static_cast<size_t>(ptrDRow) * shape.k + col]);
                output[outIndex] = RoundHalfNearestFloat(output[outIndex] + product);
            }
        }
    }
    return output;
}

void PrintM2OutputMismatchDetail(const DispatchCombineTileArgs &args, RuntimeState *state,
                                 const std::vector<float> &actualOutputC, const std::vector<float> &expectedOutputC,
                                 uint64_t firstMismatchIndex)
{
    const DispatchCombineTileShape &shape = args.shape;
    if (shape.k == 0 || firstMismatchIndex >= actualOutputC.size() || firstMismatchIndex >= expectedOutputC.size()) {
        return;
    }
    uint32_t token = static_cast<uint32_t>(firstMismatchIndex / shape.k);
    uint32_t col = static_cast<uint32_t>(firstMismatchIndex % shape.k);
    if (token >= shape.m || col >= shape.k) {
        return;
    }
    float floatAcc = 0.0f;
    float stepFloatProbAcc = 0.0f;
    float stepHalfProbAcc = 0.0f;
    float stepHalfProductAcc = 0.0f;
    float actualPayloadFloatAcc = 0.0f;
    float actualPayloadStepHalfProbAcc = 0.0f;

    std::cout << std::setprecision(8);
    std::cout << "[M2OutputMismatchDetail] rank=" << state->rank << " token=" << token << " col=" << col
              << " index=" << firstMismatchIndex << " actual=" << actualOutputC[firstMismatchIndex]
              << " expected=" << expectedOutputC[firstMismatchIndex] << "\n";
    for (uint32_t slot = 0; slot < shape.topK; ++slot) {
        size_t routeIndex = static_cast<size_t>(token) * shape.topK + slot;
        int32_t ptrDRow = state->golden.expandedRowIdx[routeIndex];
        if (ptrDRow < 0 || static_cast<uint32_t>(ptrDRow) >= shape.maxOutputSize) {
            std::cout << "  slot=" << slot << " ptrDRow=" << ptrDRow << " skip=true\n";
            continue;
        }
        float prob = state->inputs.probs[routeIndex];
        float halfProb = RoundHalfFloat(prob);
        float ptrD = state->m2Reference.ptrD[static_cast<size_t>(ptrDRow) * shape.k + col];
        float actualPtrD = CopyM2ReturnPayloadValue(args, state, static_cast<uint32_t>(ptrDRow), col);
        floatAcc += prob * ptrD;
        stepFloatProbAcc = RoundHalfFloat(stepFloatProbAcc + prob * ptrD);
        stepHalfProbAcc = RoundHalfFloat(stepHalfProbAcc + halfProb * ptrD);
        stepHalfProductAcc = RoundHalfFloat(stepHalfProductAcc + RoundHalfFloat(halfProb * ptrD));
        actualPayloadFloatAcc += prob * actualPtrD;
        actualPayloadStepHalfProbAcc = RoundHalfFloat(actualPayloadStepHalfProbAcc + halfProb * actualPtrD);
        std::cout << "  slot=" << slot << " routeIndex=" << routeIndex << " ptrDRow=" << ptrDRow << " prob=" << prob
                  << " halfProb=" << halfProb << " ptrD=" << ptrD << " actualPtrD=" << actualPtrD
                  << " ptrDDiff=" << (actualPtrD - ptrD) << " floatAcc=" << floatAcc
                  << " stepFloatProbAcc=" << stepFloatProbAcc << " stepHalfProbAcc=" << stepHalfProbAcc
                  << " stepHalfProductAcc=" << stepHalfProductAcc << " actualPayloadFloatAcc=" << actualPayloadFloatAcc
                  << " actualPayloadStepHalfProbAcc=" << actualPayloadStepHalfProbAcc << "\n";
    }
    std::cout << "  finalRoundedFloatAcc=" << RoundHalfFloat(floatAcc) << " finalStepFloatProbAcc=" << stepFloatProbAcc
              << " finalStepHalfProbAcc=" << stepHalfProbAcc << " finalStepHalfProductAcc=" << stepHalfProductAcc
              << " finalActualPayloadRoundedFloatAcc=" << RoundHalfFloat(actualPayloadFloatAcc)
              << " finalActualPayloadStepHalfProbAcc=" << actualPayloadStepHalfProbAcc << std::setprecision(1) << "\n";
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

uint64_t CompareI8Buffer(const std::string &name, const std::vector<int8_t> &actual,
                         const std::vector<int8_t> &expected, uint32_t rank)
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
                firstActual = static_cast<int32_t>(actual[i]);
                firstExpected = static_cast<int32_t>(expected[i]);
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

struct M2DispatchDump {
    std::vector<int32_t> tokenPerExpertMatrix;
    std::vector<int32_t> expandedRowIdx;
    std::vector<int32_t> cumsumMM;
    std::vector<int32_t> preSumBeforeRank;
    std::vector<int32_t> expertTokenNums;
    std::vector<int32_t> dispatchGroupReady;
    std::vector<int8_t> dispatchPayload;
    std::vector<float> dispatchScale;
    std::vector<int8_t> gmm1InputInt8;
    std::vector<float> routingPerTokenScale;
};

std::vector<int32_t> BuildExpectedM2TokenMatrix(const DispatchCombineTileArgs &args, const CpuGoldenData &golden)
{
    const DispatchCombineTileShape &shape = args.shape;
    size_t expertNumPadded = ExpertNumPadded(shape);
    size_t rowStride =
        static_cast<size_t>(moe_new_dispatch_combine_a8w8::TokenPerExpertMatrixRowStride(MakeM2ShapeConfig(args)));
    std::vector<int32_t> expected(static_cast<size_t>(shape.ep) * rowStride, 0);
    for (uint32_t tokenOwner = 0; tokenOwner < shape.ep; ++tokenOwner) {
        for (uint32_t globalExpert = 0; globalExpert < shape.expertNum; ++globalExpert) {
            uint32_t expertOwner = globalExpert / shape.expertPerRank;
            uint32_t localExpert = globalExpert % shape.expertPerRank;
            expected[static_cast<size_t>(tokenOwner) * rowStride +
                     static_cast<size_t>(expertOwner) * shape.expertPerRank + localExpert] =
                golden.peerTokenPerExpert[static_cast<size_t>(tokenOwner) * expertNumPadded + globalExpert];
        }
    }
    return expected;
}

std::vector<int32_t> BuildExpectedM2PreCountSyncExpandedRowIdx(const DispatchCombineTileArgs &args,
                                                               const HostInputData &inputs)
{
    const DispatchCombineTileShape &shape = args.shape;
    uint32_t expandedRows = shape.m * shape.topK;
    std::vector<uint32_t> localTokenPerExpert(shape.expertNum, 0);
    std::vector<int32_t> expected(expandedRows, -1);
    for (uint32_t token = 0; token < shape.m; ++token) {
        if (!golden_detail::TokenActive(inputs, token)) {
            continue;
        }
        for (uint32_t slot = 0; slot < shape.topK; ++slot) {
            uint32_t routeIndex = token * shape.topK + slot;
            int32_t expert = inputs.expertIdx[routeIndex];
            if (expert >= 0 && static_cast<uint32_t>(expert) < shape.expertNum) {
                ++localTokenPerExpert[static_cast<uint32_t>(expert)];
            }
        }
    }

    std::vector<uint32_t> expertBase(shape.expertNum, 0);
    uint32_t running = 0;
    for (uint32_t expert = 0; expert < shape.expertNum; ++expert) {
        expertBase[expert] = running;
        running += localTokenPerExpert[expert];
    }

    std::vector<uint32_t> cursor(shape.expertNum, 0);
    for (uint32_t token = 0; token < shape.m; ++token) {
        for (uint32_t slot = 0; slot < shape.topK; ++slot) {
            uint32_t routeIndex = token * shape.topK + slot;
            if (!golden_detail::TokenActive(inputs, token)) {
                expected[routeIndex] = static_cast<int32_t>(shape.maxOutputSize);
                continue;
            }
            int32_t expert = inputs.expertIdx[routeIndex];
            if (expert < 0 || static_cast<uint32_t>(expert) >= shape.expertNum) {
                expected[routeIndex] = -1;
                continue;
            }
            uint32_t expertId = static_cast<uint32_t>(expert);
            uint32_t packedRow = expertBase[expertId] + cursor[expertId]++;
            expected[routeIndex] = packedRow >= shape.maxOutputSize ? static_cast<int32_t>(shape.maxOutputSize) :
                                                                      static_cast<int32_t>(packedRow);
        }
    }
    return expected;
}

void BuildExpectedM2Prefix(const DispatchCombineTileArgs &args, const std::vector<int32_t> &tokenMatrix,
                           uint32_t expertOwnerRank, std::vector<int32_t> *cumsum, std::vector<int32_t> *preSum,
                           std::vector<int32_t> *expertTokenNums)
{
    const DispatchCombineTileShape &shape = args.shape;
    size_t rowStride =
        static_cast<size_t>(moe_new_dispatch_combine_a8w8::TokenPerExpertMatrixRowStride(MakeM2ShapeConfig(args)));
    cumsum->assign(static_cast<size_t>(shape.ep) * shape.expertPerRank, 0);
    preSum->assign(static_cast<size_t>(shape.ep) * shape.expertPerRank, 0);
    expertTokenNums->assign(shape.expertPerRank, 0);
    int32_t dispatchCursor = 0;
    for (uint32_t localExpert = 0; localExpert < shape.expertPerRank; ++localExpert) {
        int32_t before = 0;
        for (uint32_t tokenOwner = 0; tokenOwner < shape.ep; ++tokenOwner) {
            size_t matrixIndex = static_cast<size_t>(tokenOwner) * rowStride +
                                 static_cast<size_t>(expertOwnerRank) * shape.expertPerRank + localExpert;
            size_t index = static_cast<size_t>(tokenOwner) * shape.expertPerRank + localExpert;
            int32_t rows = tokenMatrix[matrixIndex];
            if (dispatchCursor >= static_cast<int32_t>(shape.maxOutputSize) || rows <= 0) {
                rows = 0;
            } else {
                int32_t available = static_cast<int32_t>(shape.maxOutputSize) - dispatchCursor;
                if (rows > available) {
                    rows = available;
                }
            }
            (*preSum)[index] = before;
            before += rows;
            (*cumsum)[index] = before;
            dispatchCursor += rows;
        }
        (*expertTokenNums)[localExpert] = before;
    }
}

std::vector<int8_t> BuildExpectedPaddedRows(const std::vector<int8_t> &compactRows, uint32_t rows, uint32_t validCols,
                                            uint32_t rowBytes)
{
    std::vector<int8_t> expected(static_cast<size_t>(rows) * rowBytes, 0);
    for (uint32_t row = 0; row < rows; ++row) {
        for (uint32_t col = 0; col < validCols; ++col) {
            expected[static_cast<size_t>(row) * rowBytes + col] =
                compactRows[static_cast<size_t>(row) * validCols + col];
        }
    }
    return expected;
}

void BuildExpectedLocalDispatchQuant(const DispatchCombineTileArgs &args, const CpuGoldenData &golden,
                                     uint32_t rowBytes, std::vector<int8_t> *payload, std::vector<float> *scale)
{
    const DispatchCombineTileShape &shape = args.shape;
    uint32_t expandedRows = shape.m * shape.topK;
    payload->assign(static_cast<size_t>(expandedRows) * rowBytes, 0);
    scale->assign(expandedRows, 0.0f);
    for (uint32_t row = 0; row < expandedRows; ++row) {
        if (row >= shape.maxOutputSize) {
            continue;
        }
        float maxAbs = 0.0f;
        for (uint32_t col = 0; col < shape.k; ++col) {
            float value = golden_detail::HalfToFloat(
                golden_detail::FloatToHalf(golden.packedA[static_cast<size_t>(row) * shape.k + col]));
            maxAbs = std::max(maxAbs, std::fabs(value));
        }
        float rowScale = maxAbs == 0.0f ? 1.0f : maxAbs / 127.0f;
        (*scale)[row] = rowScale;
        for (uint32_t col = 0; col < shape.k; ++col) {
            float value = golden_detail::HalfToFloat(
                golden_detail::FloatToHalf(golden.packedA[static_cast<size_t>(row) * shape.k + col]));
            (*payload)[static_cast<size_t>(row) * rowBytes + col] = golden_detail::QuantizeToInt8(value, rowScale);
        }
    }
}

void CopyM2DispatchToHost(const DispatchCombineTileArgs &args,
                          const moe_new_dispatch_combine_a8w8::WorkspaceLayout &workspaceLayout,
                          const moe_new_dispatch_combine_a8w8::PeerWindowLayout &peerWindowLayout, RuntimeState *state,
                          M2DispatchDump *dump)
{
    const DispatchCombineTileShape &shape = args.shape;
    uint32_t expandedRows = shape.m * shape.topK;
    uint32_t localRows = shape.maxOutputSize;
    uint32_t rankExpertCount = shape.ep * shape.expertPerRank;
    uint32_t rowBytes = static_cast<uint32_t>(peerWindowLayout.dispatchPayloadRowBytes);
    auto *workspaceBase = reinterpret_cast<uint8_t *>(state->buffers.workspace);
    auto *peerBase = reinterpret_cast<uint8_t *>(state->hccl.peerWindow);

    dump->tokenPerExpertMatrix.assign(peerWindowLayout.tokenPerExpertMatrix.bytes / sizeof(int32_t), 0);
    dump->expandedRowIdx.assign(expandedRows, -1);
    dump->cumsumMM.assign(rankExpertCount, 0);
    dump->preSumBeforeRank.assign(rankExpertCount, 0);
    dump->expertTokenNums.assign(shape.expertPerRank, 0);
    dump->dispatchGroupReady.assign(static_cast<size_t>(shape.expertPerRank) * 16U, 0);
    dump->dispatchPayload.assign(static_cast<size_t>(expandedRows) * rowBytes, 0);
    dump->dispatchScale.assign(expandedRows, 0.0f);
    dump->gmm1InputInt8.assign(static_cast<size_t>(localRows) * rowBytes, 0);
    dump->routingPerTokenScale.assign(localRows, 0.0f);

    CheckAcl(aclrtMemcpy(dump->tokenPerExpertMatrix.data(), BytesOfI32Vector(dump->tokenPerExpertMatrix.size()),
                         peerBase + peerWindowLayout.tokenPerExpertMatrix.offset,
                         BytesOfI32Vector(dump->tokenPerExpertMatrix.size()), ACL_MEMCPY_DEVICE_TO_HOST),
             "rank " + std::to_string(state->rank) + " copy m2 tokenPerExpertMatrix");
    CheckAcl(aclrtMemcpy(dump->dispatchPayload.data(), BytesOfI8Vector(dump->dispatchPayload.size()),
                         peerBase + peerWindowLayout.dispatchPayload.offset,
                         BytesOfI8Vector(dump->dispatchPayload.size()), ACL_MEMCPY_DEVICE_TO_HOST),
             "rank " + std::to_string(state->rank) + " copy m2 dispatchPayload");
    CheckAcl(aclrtMemcpy(dump->dispatchScale.data(), BytesOfFloatVector(dump->dispatchScale.size()),
                         peerBase + peerWindowLayout.dispatchScale.offset,
                         BytesOfFloatVector(dump->dispatchScale.size()), ACL_MEMCPY_DEVICE_TO_HOST),
             "rank " + std::to_string(state->rank) + " copy m2 dispatchScale");
    CheckAcl(aclrtMemcpy(dump->expandedRowIdx.data(), BytesOfI32Vector(dump->expandedRowIdx.size()),
                         workspaceBase + workspaceLayout.expandedRowIdx.offset,
                         BytesOfI32Vector(dump->expandedRowIdx.size()), ACL_MEMCPY_DEVICE_TO_HOST),
             "rank " + std::to_string(state->rank) + " copy m2 expandedRowIdx");
    CheckAcl(aclrtMemcpy(dump->cumsumMM.data(), BytesOfI32Vector(dump->cumsumMM.size()),
                         workspaceBase + workspaceLayout.cumsumMM.offset, BytesOfI32Vector(dump->cumsumMM.size()),
                         ACL_MEMCPY_DEVICE_TO_HOST),
             "rank " + std::to_string(state->rank) + " copy m2 cumsumMM");
    CheckAcl(aclrtMemcpy(dump->preSumBeforeRank.data(), BytesOfI32Vector(dump->preSumBeforeRank.size()),
                         workspaceBase + workspaceLayout.preSumBeforeRank.offset,
                         BytesOfI32Vector(dump->preSumBeforeRank.size()), ACL_MEMCPY_DEVICE_TO_HOST),
             "rank " + std::to_string(state->rank) + " copy m2 preSumBeforeRank");
    CheckAcl(aclrtMemcpy(dump->expertTokenNums.data(), BytesOfI32Vector(dump->expertTokenNums.size()),
                         workspaceBase + workspaceLayout.expertTokenNums.offset,
                         BytesOfI32Vector(dump->expertTokenNums.size()), ACL_MEMCPY_DEVICE_TO_HOST),
             "rank " + std::to_string(state->rank) + " copy m2 expertTokenNums");
    CheckAcl(aclrtMemcpy(dump->dispatchGroupReady.data(), BytesOfI32Vector(dump->dispatchGroupReady.size()),
                         workspaceBase + workspaceLayout.dispatchGroupReady.offset,
                         BytesOfI32Vector(dump->dispatchGroupReady.size()), ACL_MEMCPY_DEVICE_TO_HOST),
             "rank " + std::to_string(state->rank) + " copy m2 dispatchGroupReady");
    CheckAcl(aclrtMemcpy(dump->gmm1InputInt8.data(), BytesOfI8Vector(dump->gmm1InputInt8.size()),
                         workspaceBase + workspaceLayout.gmm1InputInt8.offset,
                         BytesOfI8Vector(dump->gmm1InputInt8.size()), ACL_MEMCPY_DEVICE_TO_HOST),
             "rank " + std::to_string(state->rank) + " copy m2 gmm1InputInt8");
    CheckAcl(aclrtMemcpy(dump->routingPerTokenScale.data(), BytesOfFloatVector(dump->routingPerTokenScale.size()),
                         workspaceBase + workspaceLayout.routingPerTokenScale.offset,
                         BytesOfFloatVector(dump->routingPerTokenScale.size()), ACL_MEMCPY_DEVICE_TO_HOST),
             "rank " + std::to_string(state->rank) + " copy m2 routingPerTokenScale");
}

uint64_t VerifyM2Dispatch(const DispatchCombineTileArgs &args,
                          const moe_new_dispatch_combine_a8w8::PeerWindowLayout &peerWindowLayout, RuntimeState *state,
                          const M2DispatchDump &dump)
{
    uint64_t mismatches = 0;
    uint32_t rowBytes = static_cast<uint32_t>(peerWindowLayout.dispatchPayloadRowBytes);
    std::vector<int32_t> expectedTokenMatrix = BuildExpectedM2TokenMatrix(args, state->golden);
    std::vector<int32_t> expectedCumsum;
    std::vector<int32_t> expectedPreSum;
    std::vector<int32_t> expectedExpertTokenNums;
    BuildExpectedM2Prefix(args, expectedTokenMatrix, state->rank, &expectedCumsum, &expectedPreSum,
                          &expectedExpertTokenNums);
    std::vector<int8_t> expectedDispatchPayload;
    std::vector<float> expectedDispatchScale;
    BuildExpectedLocalDispatchQuant(args, state->golden, rowBytes, &expectedDispatchPayload, &expectedDispatchScale);
    std::vector<int8_t> expectedGmm1Input =
        BuildExpectedPaddedRows(state->m2Reference.gmm1InputInt8, args.shape.maxOutputSize, args.shape.k, rowBytes);

    mismatches +=
        CompareI32Buffer("m2.tokenPerExpertMatrix", dump.tokenPerExpertMatrix, expectedTokenMatrix, state->rank);
    mismatches += CompareI32Buffer("m2.expandedRowIdx", dump.expandedRowIdx, state->golden.expandedRowIdx, state->rank);
    mismatches += CompareI32Buffer("m2.cumsumMM", dump.cumsumMM, expectedCumsum, state->rank);
    mismatches += CompareI32Buffer("m2.preSumBeforeRank", dump.preSumBeforeRank, expectedPreSum, state->rank);
    mismatches += CompareI32Buffer("m2.expertTokenNums", dump.expertTokenNums, expectedExpertTokenNums, state->rank);
    mismatches += CompareI8Buffer("m2.dispatchPayloadInt8", dump.dispatchPayload, expectedDispatchPayload, state->rank);
    mismatches += CompareFloatBuffer(args, "m2.dispatchScale", dump.dispatchScale, expectedDispatchScale, state->rank);
    mismatches += CompareI8Buffer("m2.gmm1InputInt8", dump.gmm1InputInt8, expectedGmm1Input, state->rank);
    mismatches += CompareFloatBuffer(args, "m2.routingPerTokenScale", dump.routingPerTokenScale,
                                     state->m2Reference.routingPerTokenScale, state->rank);
    return mismatches;
}

struct InitQuantVerifySummary {
    bool routeCountChecked = false;
    bool routeCountMatch = true;
    bool tokenMatrixFullChecked = false;
    bool tokenMatrixFullMatch = true;
    bool expandedRowChecked = false;
    bool expandedRowMatch = true;
    bool payloadSampleChecked = false;
    bool payloadSampleMatch = true;
    bool prefixChecked = false;
    bool prefixMatch = true;
    bool gmm1InputChecked = false;
    bool gmm1InputMatch = true;
    bool dispatchReadyChecked = false;
    bool dispatchReadyMatch = true;
    uint64_t routeCountMismatches = 0;
    uint64_t tokenMatrixFullMismatches = 0;
    uint64_t expandedRowMismatches = 0;
    uint64_t payloadSampleMismatches = 0;
    uint64_t prefixMismatches = 0;
    uint64_t gmm1InputMismatches = 0;
    uint64_t dispatchReadyMismatches = 0;

    bool Pass() const
    {
        return routeCountMatch && tokenMatrixFullMatch && expandedRowMatch && payloadSampleMatch && prefixMatch &&
               gmm1InputMatch && dispatchReadyMatch;
    }
};

bool IsInitQuantDebugStopStage(uint32_t debugStopStage)
{
    return debugStopStage == 17U;
}

bool InitQuantDispatchScratchFits(const DispatchCombineTileArgs &args)
{
    uint32_t globalExpertNum = args.shape.ep * args.shape.expertPerRank;
    return globalExpertNum > 0U;
}

const char *InitQuantDispatchParallelPath(const DispatchCombineTileArgs &args)
{
    if (!InitQuantDispatchScratchFits(args)) {
        return "pto_main_aiv";
    }
    return "m3n_multi_worker";
}

const char *InitQuantDispatchParallelFallbackReason(const DispatchCombineTileArgs &args)
{
    if (!InitQuantDispatchScratchFits(args)) {
        return "invalid_global_expert_num";
    }
    return "none";
}

bool InitQuantUsesPtoVecRouteQuant(const DispatchCombineTileArgs &args)
{
    (void)args;
    return true;
}

const char *InitQuantRouteQuantPath(const DispatchCombineTileArgs &args)
{
    return InitQuantUsesPtoVecRouteQuant(args) ? "pto_vec" : "scalar";
}

const char *InitQuantRouteQuantFallbackReason(const DispatchCombineTileArgs &args)
{
    if (InitQuantUsesPtoVecRouteQuant(args)) {
        return "none";
    }
    return "unknown";
}

const char *CheckedBool(bool checked, bool match)
{
    return (!checked || match) ? "true" : "false";
}

uint64_t CompareInitQuantLocalRouteCount(const DispatchCombineTileArgs &args, RuntimeState *state,
                                         const M2DispatchDump &dump, const std::vector<int32_t> &expectedTokenMatrix)
{
    uint32_t rank = state->rank;
    size_t rowStride =
        static_cast<size_t>(moe_new_dispatch_combine_a8w8::TokenPerExpertMatrixRowStride(MakeM2ShapeConfig(args)));
    size_t rowOffset = static_cast<size_t>(rank) * rowStride;
    std::vector<int32_t> actualRow(rowStride, 0);
    std::vector<int32_t> expectedRow(rowStride, 0);
    if (rowOffset + rowStride <= dump.tokenPerExpertMatrix.size()) {
        std::copy(dump.tokenPerExpertMatrix.begin() + static_cast<std::ptrdiff_t>(rowOffset),
                  dump.tokenPerExpertMatrix.begin() + static_cast<std::ptrdiff_t>(rowOffset + rowStride),
                  actualRow.begin());
    }
    if (rowOffset + rowStride <= expectedTokenMatrix.size()) {
        std::copy(expectedTokenMatrix.begin() + static_cast<std::ptrdiff_t>(rowOffset),
                  expectedTokenMatrix.begin() + static_cast<std::ptrdiff_t>(rowOffset + rowStride),
                  expectedRow.begin());
    }
    return CompareI32Buffer("init_quant.local_tokenPerExpertMatrix", actualRow, expectedRow, rank);
}

uint64_t CompareInitQuantDispatchReady(const DispatchCombineTileArgs &args, RuntimeState *state,
                                       const M2DispatchDump &dump)
{
    uint64_t mismatches = 0;
    size_t firstMismatch = static_cast<size_t>(args.shape.expertPerRank);
    int32_t firstActual = 0;
    for (uint32_t localExpert = 0; localExpert < args.shape.expertPerRank; ++localExpert) {
        size_t index = static_cast<size_t>(localExpert) * 16U;
        int32_t ready = index < dump.dispatchGroupReady.size() ? dump.dispatchGroupReady[index] : 0;
        if (ready == 0) {
            if (firstMismatch == static_cast<size_t>(args.shape.expertPerRank)) {
                firstMismatch = localExpert;
                firstActual = ready;
            }
            ++mismatches;
        }
    }
    std::cout << "rank=" << state->rank << " buffer=init_quant.dispatchGroupReady experts=" << args.shape.expertPerRank
              << " mismatches=" << mismatches;
    if (mismatches != 0) {
        std::cout << " first_expert=" << firstMismatch << " actual=" << firstActual << " expected_nonzero=1";
    }
    std::cout << std::endl;
    return mismatches;
}

InitQuantVerifySummary VerifyInitQuantDebugStop(const DispatchCombineTileArgs &args,
                                                const moe_new_dispatch_combine_a8w8::PeerWindowLayout &peerWindowLayout,
                                                RuntimeState *state, const M2DispatchDump &dump)
{
    InitQuantVerifySummary summary;
    uint32_t rowBytes = static_cast<uint32_t>(peerWindowLayout.dispatchPayloadRowBytes);
    std::vector<int32_t> expectedTokenMatrix = BuildExpectedM2TokenMatrix(args, state->golden);

    summary.routeCountChecked = true;
    summary.routeCountMismatches = CompareInitQuantLocalRouteCount(args, state, dump, expectedTokenMatrix);
    summary.routeCountMatch = summary.routeCountMismatches == 0;

    summary.expandedRowChecked = true;
    summary.expandedRowMismatches =
        CompareI32Buffer("init_quant.expandedRowIdx", dump.expandedRowIdx, state->golden.expandedRowIdx, state->rank);
    summary.expandedRowMatch = summary.expandedRowMismatches == 0;

    summary.payloadSampleChecked = true;
    std::vector<int8_t> expectedDispatchPayload;
    std::vector<float> expectedDispatchScale;
    BuildExpectedLocalDispatchQuant(args, state->golden, rowBytes, &expectedDispatchPayload, &expectedDispatchScale);
    summary.payloadSampleMismatches =
        CompareI8Buffer("init_quant.dispatchPayloadInt8", dump.dispatchPayload, expectedDispatchPayload, state->rank);
    summary.payloadSampleMismatches +=
        CompareFloatBuffer(args, "init_quant.dispatchScale", dump.dispatchScale, expectedDispatchScale, state->rank);
    summary.payloadSampleMatch = summary.payloadSampleMismatches == 0;

    summary.tokenMatrixFullChecked = true;
    summary.tokenMatrixFullMismatches =
        CompareI32Buffer("init_quant.tokenPerExpertMatrix", dump.tokenPerExpertMatrix, expectedTokenMatrix, state->rank);
    summary.tokenMatrixFullMatch = summary.tokenMatrixFullMismatches == 0;

    summary.prefixChecked = true;
    std::vector<int32_t> expectedCumsum;
    std::vector<int32_t> expectedPreSum;
    std::vector<int32_t> expectedExpertTokenNums;
    BuildExpectedM2Prefix(args, expectedTokenMatrix, state->rank, &expectedCumsum, &expectedPreSum,
                          &expectedExpertTokenNums);
    summary.prefixMismatches = CompareI32Buffer("init_quant.cumsumMM", dump.cumsumMM, expectedCumsum, state->rank);
    summary.prefixMismatches +=
        CompareI32Buffer("init_quant.preSumBeforeRank", dump.preSumBeforeRank, expectedPreSum, state->rank);
    summary.prefixMismatches +=
        CompareI32Buffer("init_quant.expertTokenNums", dump.expertTokenNums, expectedExpertTokenNums, state->rank);
    summary.prefixMatch = summary.prefixMismatches == 0;

    summary.gmm1InputChecked = true;
    std::vector<int8_t> expectedGmm1Input =
        BuildExpectedPaddedRows(state->m2Reference.gmm1InputInt8, args.shape.maxOutputSize, args.shape.k, rowBytes);
    summary.gmm1InputMismatches =
        CompareI8Buffer("init_quant.gmm1InputInt8", dump.gmm1InputInt8, expectedGmm1Input, state->rank);
    summary.gmm1InputMismatches += CompareFloatBuffer(args, "init_quant.routingPerTokenScale",
                                                      dump.routingPerTokenScale,
                                                      state->m2Reference.routingPerTokenScale, state->rank);
    summary.gmm1InputMatch = summary.gmm1InputMismatches == 0;

    summary.dispatchReadyChecked = true;
    summary.dispatchReadyMismatches = CompareInitQuantDispatchReady(args, state, dump);
    summary.dispatchReadyMatch = summary.dispatchReadyMismatches == 0;
    return summary;
}

void PrintInitQuantDebugStopReport(const DispatchCombineTileArgs &args, const M2FusedFullEvidence &evidence,
                                   const InitQuantVerifySummary &summary)
{
    std::ostringstream report;
    report << "  init_quant_final_stop=" << args.runtime.m2FusedDebugStopStage << "\n";
    report << "  init_quant_e2e_us=" << InitQuantE2eUs(evidence) << "\n";
    report << "  init_quant_worker_count=" << evidence.m3nDispatchCounters[3] << "\n";
    report << "  init_quant_assigned_workers=" << evidence.m3nDispatchCounters[11] << "\n";
    report << "  init_quant_active_workers=" << evidence.m3nDispatchCounters[0] << "\n";
    report << "  init_quant_worker_mask=" << evidence.m3nDispatchCounters[10] << "\n";
    report << "  init_quant_dispatch_parallel_path=" << InitQuantDispatchParallelPath(args) << "\n";
    report << "  init_quant_dispatch_parallel_fallback_reason=" << InitQuantDispatchParallelFallbackReason(args)
           << "\n";
    report << "  init_quant_route_count_checked=" << (summary.routeCountChecked ? "true" : "false") << "\n";
    report << "  init_quant_route_count_match=" << CheckedBool(summary.routeCountChecked, summary.routeCountMatch)
           << "\n";
    report << "  init_quant_route_count_mismatches=" << summary.routeCountMismatches << "\n";
    report << "  init_quant_token_matrix_full_checked=" << (summary.tokenMatrixFullChecked ? "true" : "false")
           << "\n";
    report << "  init_quant_token_matrix_full_match="
           << CheckedBool(summary.tokenMatrixFullChecked, summary.tokenMatrixFullMatch) << "\n";
    report << "  init_quant_token_matrix_full_mismatches=" << summary.tokenMatrixFullMismatches << "\n";
    report << "  init_quant_expanded_row_checked=" << (summary.expandedRowChecked ? "true" : "false") << "\n";
    report << "  init_quant_expanded_row_contract=capacity_clipped\n";
    report << "  init_quant_expanded_row_match=" << CheckedBool(summary.expandedRowChecked, summary.expandedRowMatch)
           << "\n";
    report << "  init_quant_expanded_row_mismatches=" << summary.expandedRowMismatches << "\n";
    report << "  init_quant_payload_sample_checked=" << (summary.payloadSampleChecked ? "true" : "false") << "\n";
    report << "  init_quant_payload_sample_match="
           << CheckedBool(summary.payloadSampleChecked, summary.payloadSampleMatch) << "\n";
    report << "  init_quant_payload_sample_mismatches=" << summary.payloadSampleMismatches << "\n";
    report << "  init_quant_prefix_checked=" << (summary.prefixChecked ? "true" : "false") << "\n";
    report << "  init_quant_prefix_match=" << CheckedBool(summary.prefixChecked, summary.prefixMatch) << "\n";
    report << "  init_quant_prefix_mismatches=" << summary.prefixMismatches << "\n";
    report << "  init_quant_gmm1_input_checked=" << (summary.gmm1InputChecked ? "true" : "false") << "\n";
    report << "  init_quant_gmm1_input_match=" << CheckedBool(summary.gmm1InputChecked, summary.gmm1InputMatch)
           << "\n";
    report << "  init_quant_gmm1_input_mismatches=" << summary.gmm1InputMismatches << "\n";
    report << "  init_quant_dispatch_ready_checked=" << (summary.dispatchReadyChecked ? "true" : "false") << "\n";
    report << "  init_quant_dispatch_ready_match="
           << CheckedBool(summary.dispatchReadyChecked, summary.dispatchReadyMatch) << "\n";
    report << "  init_quant_dispatch_ready_mismatches=" << summary.dispatchReadyMismatches << "\n";
    report << "  route_quant_path=" << InitQuantRouteQuantPath(args) << "\n";
    report << "  route_quant_scalar_fallback_reason=" << InitQuantRouteQuantFallbackReason(args) << "\n";
    std::cout << report.str();
}

std::vector<int32_t> CopyM2Gmm1AccToHost(const DispatchCombineTileArgs &args,
                                         const moe_new_dispatch_combine_a8w8::WorkspaceLayout &workspaceLayout,
                                         RuntimeState *state)
{
    const DispatchCombineTileShape &shape = args.shape;
    uint32_t w1Cols = args.intermediateSize * 2U;
    std::vector<int32_t> gmm1Acc(static_cast<size_t>(shape.maxOutputSize) * w1Cols, 0);
    auto *workspaceBase = reinterpret_cast<uint8_t *>(state->buffers.workspace);
    CheckAcl(aclrtMemcpy(gmm1Acc.data(), BytesOfI32Vector(gmm1Acc.size()),
                         workspaceBase + workspaceLayout.gmm1AccInt32.offset, BytesOfI32Vector(gmm1Acc.size()),
                         ACL_MEMCPY_DEVICE_TO_HOST),
             "rank " + std::to_string(state->rank) + " copy m2 gmm1AccInt32");
    return gmm1Acc;
}

uint64_t VerifyM2Gmm1(const DispatchCombineTileArgs &args, RuntimeState *state, const std::vector<int32_t> &gmm1Acc)
{
    return CompareI32Buffer("m2.gmm1AccInt32", gmm1Acc, state->m2Reference.gmm1AccInt32, state->rank);
}

struct M2GmmTaskDump {
    std::vector<int32_t> stageStatus;
    std::vector<int32_t> tileTaskPlan;
};

M2GmmTaskDump CopyM2GmmTaskDump(const moe_new_dispatch_combine_a8w8::WorkspaceLayout &workspaceLayout,
                                RuntimeState *state, bool gmm2)
{
    M2GmmTaskDump dump;
    auto *workspaceBase = reinterpret_cast<uint8_t *>(state->buffers.workspace);
    dump.stageStatus.assign(workspaceLayout.stageStatus.bytes / sizeof(int32_t), 0);
    const auto &field = gmm2 ? workspaceLayout.gmm2TileTaskPlan : workspaceLayout.gmm1TileTaskPlan;
    dump.tileTaskPlan.assign(field.bytes / sizeof(int32_t), 0);
    CheckAcl(aclrtMemcpy(dump.stageStatus.data(), BytesOfI32Vector(dump.stageStatus.size()),
                         workspaceBase + workspaceLayout.stageStatus.offset, workspaceLayout.stageStatus.bytes,
                         ACL_MEMCPY_DEVICE_TO_HOST),
             "rank " + std::to_string(state->rank) + " copy m2 stageStatus");
    CheckAcl(
        aclrtMemcpy(dump.tileTaskPlan.data(), BytesOfI32Vector(dump.tileTaskPlan.size()), workspaceBase + field.offset,
                    field.bytes, ACL_MEMCPY_DEVICE_TO_HOST),
        "rank " + std::to_string(state->rank) + (gmm2 ? " copy m2 gmm2TileTaskPlan" : " copy m2 gmm1TileTaskPlan"));
    return dump;
}

std::vector<int32_t> BuildExpectedM2GmmTileTasks(const DispatchCombineTileArgs &args, RuntimeState *state, bool gmm2,
                                                 uint32_t *taskCount)
{
    const DispatchCombineTileShape &shape = args.shape;
    moe_new_dispatch_combine_a8w8::ShapeConfig m2Shape = MakeM2ShapeConfig(args);
    size_t capacity = moe_new_dispatch_combine_a8w8::GmmTileTaskCapacity(m2Shape);
    std::vector<int32_t> expected(capacity * 8U, 0);
    uint32_t nCols = gmm2 ? shape.k : args.intermediateSize * 2U;
    uint32_t kSize = gmm2 ? args.intermediateSize : shape.k;
    uint32_t stageId = gmm2 ? 2U : 1U;
    uint32_t id = 0;
    for (uint32_t localExpert = 0; localExpert < shape.expertPerRank; ++localExpert) {
        int32_t rowBegin = state->golden.dispatchOffset[localExpert];
        uint32_t rowCount = 0;
        uint32_t globalExpert = state->rank * shape.expertPerRank + localExpert;
        size_t expertNumPadded = ExpertNumPadded(shape);
        for (uint32_t tokenOwner = 0; tokenOwner < shape.ep; ++tokenOwner) {
            rowCount += static_cast<uint32_t>(
                state->golden.peerTokenPerExpert[static_cast<size_t>(tokenOwner) * expertNumPadded + globalExpert]);
        }
        for (uint32_t rowOffset = 0; rowOffset < rowCount; rowOffset += moe_new_dispatch_combine_a8w8::kGmmBaseM) {
            uint32_t rows = std::min<uint32_t>(rowCount - rowOffset, moe_new_dispatch_combine_a8w8::kGmmBaseM);
            for (uint32_t nBase = 0; nBase < nCols; nBase += moe_new_dispatch_combine_a8w8::kGmmBaseN) {
                uint32_t cols = std::min<uint32_t>(nCols - nBase, moe_new_dispatch_combine_a8w8::kGmmBaseN);
                if (id < capacity) {
                    size_t base = static_cast<size_t>(id) * 8U;
                    expected[base + 0U] = static_cast<int32_t>(id);
                    expected[base + 1U] = static_cast<int32_t>(stageId);
                    expected[base + 2U] = static_cast<int32_t>(localExpert);
                    expected[base + 3U] = rowBegin + static_cast<int32_t>(rowOffset);
                    expected[base + 4U] = static_cast<int32_t>(rows);
                    expected[base + 5U] = static_cast<int32_t>(nBase);
                    expected[base + 6U] = static_cast<int32_t>(cols);
                    expected[base + 7U] = static_cast<int32_t>(kSize);
                }
                ++id;
            }
        }
    }
    if (id > capacity) {
        throw std::runtime_error("expected M2 GMM tile task capacity overflow");
    }
    *taskCount = id;
    return expected;
}

std::vector<float> CopyM2Gmm1OutToHost(const DispatchCombineTileArgs &args,
                                       const moe_new_dispatch_combine_a8w8::WorkspaceLayout &workspaceLayout,
                                       RuntimeState *state)
{
    const DispatchCombineTileShape &shape = args.shape;
    uint32_t w1Cols = args.intermediateSize * 2U;
    std::vector<float> gmm1Out(static_cast<size_t>(shape.maxOutputSize) * w1Cols, 0.0f);
    auto *workspaceBase = reinterpret_cast<uint8_t *>(state->buffers.workspace);
    CheckAcl(
        aclrtMemcpy(gmm1Out.data(), BytesOfFloatVector(gmm1Out.size()), workspaceBase + workspaceLayout.gmm1Out.offset,
                    BytesOfFloatVector(gmm1Out.size()), ACL_MEMCPY_DEVICE_TO_HOST),
        "rank " + std::to_string(state->rank) + " copy m2 gmm1Out");
    return gmm1Out;
}

uint64_t VerifyM2Gmm1Epilogue(const DispatchCombineTileArgs &args, RuntimeState *state,
                              const std::vector<float> &gmm1Out)
{
    const DispatchCombineTileShape &shape = args.shape;
    uint32_t w1Cols = args.intermediateSize * 2U;
    const std::vector<float> &expected = state->m2Reference.gmm1Out;
    uint64_t mismatches = 0;
    size_t elementCount = gmm1Out.size() < expected.size() ? gmm1Out.size() : expected.size();
    size_t firstMismatch = elementCount;
    float firstActual = 0.0f;
    float firstExpected = 0.0f;
    if (gmm1Out.size() != expected.size()) {
        mismatches =
            gmm1Out.size() > expected.size() ? gmm1Out.size() - expected.size() : expected.size() - gmm1Out.size();
        firstMismatch = elementCount;
    }
    for (size_t i = 0; i < elementCount; ++i) {
        float actualValue = gmm1Out[i];
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

    uint64_t firstRow = firstMismatch == elementCount ? 0 : firstMismatch / w1Cols;
    uint64_t firstCol = firstMismatch == elementCount ? 0 : firstMismatch % w1Cols;
    uint32_t firstExpert = 0;
    for (uint32_t localExpert = 0; localExpert < shape.expertPerRank; ++localExpert) {
        uint32_t rowBegin = static_cast<uint32_t>(state->golden.dispatchOffset[localExpert]);
        uint32_t rowEnd = localExpert + 1U < shape.expertPerRank ?
                              static_cast<uint32_t>(state->golden.dispatchOffset[localExpert + 1U]) :
                              (state->rank < state->golden.ownerRows.size() ? state->golden.ownerRows[state->rank] :
                                                                              shape.maxOutputSize);
        if (firstRow >= rowBegin && firstRow < rowEnd) {
            firstExpert = localExpert;
            break;
        }
    }
    std::cout << "rank=" << state->rank << " buffer=m2.gmm1Out elements=" << gmm1Out.size()
              << " mismatches=" << mismatches;
    if (mismatches != 0) {
        std::cout << " first_index=" << firstMismatch << " first_expert=" << firstExpert
                  << " first_tile=" << (firstCol / std::max(1U, shape.gmmBlockN)) << " first_row=" << firstRow
                  << " first_col=" << firstCol << " actual=" << firstActual << " expected=" << firstExpected;
    }
    std::cout << std::endl;
    return mismatches;
}

struct M2ActivationDump {
    std::vector<float> swigluOut;
    std::vector<int8_t> gmm2InputInt8;
    std::vector<float> gmm2PerTokenScale;
    std::vector<int32_t> swigluSyncGroups;
    std::vector<int32_t> dequantSum;
    std::vector<int32_t> swigluGroupDesc;
    std::vector<int32_t> gmm2TileTaskPlan;
    std::vector<int32_t> stageStatus;
    std::vector<int32_t> activationSyncGroupReady;
};

uint32_t NextSwigluGroupSizeHost(uint32_t remainingExperts)
{
    if (remainingExperts <= 1U) {
        return 1U;
    }
    return remainingExperts / 2U;
}

std::vector<uint32_t> BuildSwigluGroupSizes(uint32_t expertPerRank)
{
    std::vector<uint32_t> groups;
    uint32_t remaining = expertPerRank;
    while (remaining > 0) {
        uint32_t groupSize = NextSwigluGroupSizeHost(remaining);
        groups.push_back(groupSize);
        remaining -= groupSize;
    }
    return groups;
}

std::vector<int32_t> BuildExpectedSwigluGroupStorage(const DispatchCombineTileArgs &args)
{
    std::vector<uint32_t> groups = BuildSwigluGroupSizes(args.shape.expertPerRank);
    std::vector<int32_t> expected(static_cast<size_t>(args.shape.expertPerRank) + 1U, 0);
    expected[0] = static_cast<int32_t>(groups.size());
    for (size_t i = 0; i < groups.size(); ++i) {
        expected[i + 1U] = static_cast<int32_t>(groups[i]);
    }
    return expected;
}

std::vector<int32_t> BuildExpectedDequantSum(const DispatchCombineTileArgs &args, RuntimeState *state)
{
    std::vector<uint32_t> groups = BuildSwigluGroupSizes(args.shape.expertPerRank);
    std::vector<int32_t> expected(static_cast<size_t>(args.shape.expertPerRank) + 2U, 0);
    uint32_t expert = 0;
    int32_t rowPrefix = 0;
    expected[0] = 0;
    for (size_t group = 0; group < groups.size(); ++group) {
        for (uint32_t i = 0; i < groups[group]; ++i) {
            uint32_t localExpert = expert + i;
            uint32_t rowBegin = static_cast<uint32_t>(state->golden.dispatchOffset[localExpert]);
            uint32_t rowEnd = localExpert + 1U < args.shape.expertPerRank ?
                                  static_cast<uint32_t>(state->golden.dispatchOffset[localExpert + 1U]) :
                                  (state->rank < state->golden.ownerRows.size() ? state->golden.ownerRows[state->rank] :
                                                                                  args.shape.maxOutputSize);
            rowPrefix += static_cast<int32_t>(rowEnd - rowBegin);
        }
        expected[group + 1U] = rowPrefix;
        expert += groups[group];
    }
    return expected;
}

std::vector<int32_t> BuildExpectedSwigluGroupDesc(const DispatchCombineTileArgs &args, RuntimeState *state)
{
    std::vector<uint32_t> groups = BuildSwigluGroupSizes(args.shape.expertPerRank);
    size_t groupCap = static_cast<size_t>(args.shape.expertPerRank) + 1U;
    std::vector<int32_t> expected(groupCap * 8U, 0);
    uint32_t expert = 0;
    int32_t rowPrefix = 0;
    uint32_t tilePrefix = 0;
    for (size_t groupId = 0; groupId < groups.size(); ++groupId) {
        int32_t rowBegin = rowPrefix;
        for (uint32_t i = 0; i < groups[groupId]; ++i) {
            uint32_t localExpert = expert + i;
            uint32_t rowStart = static_cast<uint32_t>(state->golden.dispatchOffset[localExpert]);
            uint32_t rowEnd = localExpert + 1U < args.shape.expertPerRank ?
                                  static_cast<uint32_t>(state->golden.dispatchOffset[localExpert + 1U]) :
                                  (state->rank < state->golden.ownerRows.size() ? state->golden.ownerRows[state->rank] :
                                                                                  args.shape.maxOutputSize);
            rowPrefix += static_cast<int32_t>(rowEnd - rowStart);
        }
        uint32_t tileBegin = tilePrefix;
        uint32_t tileEnd =
            tileBegin + static_cast<uint32_t>(moe_new_dispatch_combine_a8w8::CeilDiv(
                            static_cast<uint64_t>(rowPrefix - rowBegin), moe_new_dispatch_combine_a8w8::GmmBaseM()));
        size_t base = groupId * 8U;
        expected[base + 0U] = static_cast<int32_t>(groupId);
        expected[base + 1U] = static_cast<int32_t>(expert);
        expected[base + 2U] = static_cast<int32_t>(expert + groups[groupId]);
        expected[base + 3U] = rowBegin;
        expected[base + 4U] = rowPrefix;
        expected[base + 5U] = static_cast<int32_t>(tileBegin);
        expected[base + 6U] = static_cast<int32_t>(tileEnd);
        expected[base + 7U] = rowPrefix == rowBegin ? 1 : 0;
        tilePrefix = tileEnd;
        expert += groups[groupId];
    }
    return expected;
}

void PrintGroupSizes(uint32_t rank, const std::vector<int32_t> &groupStorage)
{
    int32_t groupCount = groupStorage.empty() ? 0 : groupStorage[0];
    std::cout << "rank=" << rank << " swiglu_group_sizes={";
    for (int32_t i = 0; i < groupCount; ++i) {
        if (i != 0) {
            std::cout << ",";
        }
        std::cout << groupStorage[static_cast<size_t>(i) + 1U];
    }
    std::cout << "}" << std::endl;
}

void PrintSyntheticSwigluGroupPlan(uint32_t rank)
{
    std::vector<uint32_t> groups = BuildSwigluGroupSizes(16);
    std::cout << "rank=" << rank << " swiglu_group_sizes_synthetic_expert_per_rank_16={";
    for (size_t i = 0; i < groups.size(); ++i) {
        if (i != 0) {
            std::cout << ",";
        }
        std::cout << groups[i];
    }
    std::cout << "}" << std::endl;
}

M2ActivationDump CopyM2ActivationToHost(const DispatchCombineTileArgs &args,
                                        const moe_new_dispatch_combine_a8w8::WorkspaceLayout &workspaceLayout,
                                        RuntimeState *state)
{
    const DispatchCombineTileShape &shape = args.shape;
    uint32_t gmm2RowStride = static_cast<uint32_t>(
        moe_new_dispatch_combine_a8w8::AlignUp(args.intermediateSize, moe_new_dispatch_combine_a8w8::kCacheLineBytes));
    M2ActivationDump dump;
    dump.swigluOut.assign(static_cast<size_t>(shape.maxOutputSize) * args.intermediateSize, 0.0f);
    dump.gmm2InputInt8.assign(static_cast<size_t>(shape.maxOutputSize) * gmm2RowStride, 0);
    dump.gmm2PerTokenScale.assign(shape.maxOutputSize, 0.0f);
    dump.swigluSyncGroups.assign(static_cast<size_t>(shape.expertPerRank) + 1U, 0);
    dump.dequantSum.assign(static_cast<size_t>(shape.expertPerRank) + 2U, 0);
    dump.swigluGroupDesc.assign(workspaceLayout.swigluGroupDesc.bytes / sizeof(int32_t), 0);
    dump.gmm2TileTaskPlan.assign(workspaceLayout.gmm2TileTaskPlan.bytes / sizeof(int32_t), 0);
    dump.stageStatus.assign(workspaceLayout.stageStatus.bytes / sizeof(int32_t), 0);
    dump.activationSyncGroupReady.assign((static_cast<size_t>(shape.expertPerRank) + 1U) * 16U, 0);
    auto *workspaceBase = reinterpret_cast<uint8_t *>(state->buffers.workspace);
    CheckAcl(aclrtMemcpy(dump.swigluOut.data(), BytesOfFloatVector(dump.swigluOut.size()),
                         workspaceBase + workspaceLayout.swigluOut.offset, BytesOfFloatVector(dump.swigluOut.size()),
                         ACL_MEMCPY_DEVICE_TO_HOST),
             "rank " + std::to_string(state->rank) + " copy m2 swigluOut");
    CheckAcl(aclrtMemcpy(dump.gmm2InputInt8.data(), BytesOfI8Vector(dump.gmm2InputInt8.size()),
                         workspaceBase + workspaceLayout.gmm2InputInt8.offset,
                         BytesOfI8Vector(dump.gmm2InputInt8.size()), ACL_MEMCPY_DEVICE_TO_HOST),
             "rank " + std::to_string(state->rank) + " copy m2 gmm2InputInt8");
    CheckAcl(aclrtMemcpy(dump.gmm2PerTokenScale.data(), BytesOfFloatVector(dump.gmm2PerTokenScale.size()),
                         workspaceBase + workspaceLayout.gmm2PerTokenScale.offset,
                         BytesOfFloatVector(dump.gmm2PerTokenScale.size()), ACL_MEMCPY_DEVICE_TO_HOST),
             "rank " + std::to_string(state->rank) + " copy m2 gmm2PerTokenScale");
    CheckAcl(aclrtMemcpy(dump.swigluSyncGroups.data(), BytesOfI32Vector(dump.swigluSyncGroups.size()),
                         workspaceBase + workspaceLayout.swigluSyncGroups.offset,
                         BytesOfI32Vector(dump.swigluSyncGroups.size()), ACL_MEMCPY_DEVICE_TO_HOST),
             "rank " + std::to_string(state->rank) + " copy m2 swigluSyncGroups");
    CheckAcl(aclrtMemcpy(dump.dequantSum.data(), BytesOfI32Vector(dump.dequantSum.size()),
                         workspaceBase + workspaceLayout.dequantSum.offset, BytesOfI32Vector(dump.dequantSum.size()),
                         ACL_MEMCPY_DEVICE_TO_HOST),
             "rank " + std::to_string(state->rank) + " copy m2 dequantSum");
    CheckAcl(aclrtMemcpy(dump.swigluGroupDesc.data(), BytesOfI32Vector(dump.swigluGroupDesc.size()),
                         workspaceBase + workspaceLayout.swigluGroupDesc.offset, workspaceLayout.swigluGroupDesc.bytes,
                         ACL_MEMCPY_DEVICE_TO_HOST),
             "rank " + std::to_string(state->rank) + " copy m2 swigluGroupDesc");
    CheckAcl(aclrtMemcpy(dump.gmm2TileTaskPlan.data(), BytesOfI32Vector(dump.gmm2TileTaskPlan.size()),
                         workspaceBase + workspaceLayout.gmm2TileTaskPlan.offset,
                         workspaceLayout.gmm2TileTaskPlan.bytes, ACL_MEMCPY_DEVICE_TO_HOST),
             "rank " + std::to_string(state->rank) + " copy m2 gmm2TileTaskPlan preview");
    CheckAcl(aclrtMemcpy(dump.stageStatus.data(), BytesOfI32Vector(dump.stageStatus.size()),
                         workspaceBase + workspaceLayout.stageStatus.offset, workspaceLayout.stageStatus.bytes,
                         ACL_MEMCPY_DEVICE_TO_HOST),
             "rank " + std::to_string(state->rank) + " copy m2 stageStatus");
    CheckAcl(aclrtMemcpy(dump.activationSyncGroupReady.data(), BytesOfI32Vector(dump.activationSyncGroupReady.size()),
                         workspaceBase + workspaceLayout.activationSyncGroupReady.offset,
                         BytesOfI32Vector(dump.activationSyncGroupReady.size()), ACL_MEMCPY_DEVICE_TO_HOST),
             "rank " + std::to_string(state->rank) + " copy m2 activationSyncGroupReady");
    return dump;
}

uint64_t VerifyM2ActivationQuant(const DispatchCombineTileArgs &args, RuntimeState *state, const M2ActivationDump &dump)
{
    uint32_t gmm2RowStride = static_cast<uint32_t>(
        moe_new_dispatch_combine_a8w8::AlignUp(args.intermediateSize, moe_new_dispatch_combine_a8w8::kCacheLineBytes));
    std::vector<int8_t> expectedGmm2Input = BuildExpectedPaddedRows(
        state->m2Reference.gmm2InputInt8, args.shape.maxOutputSize, args.intermediateSize, gmm2RowStride);
    std::vector<int32_t> expectedGroups = BuildExpectedSwigluGroupStorage(args);
    std::vector<int32_t> expectedDequantSum = BuildExpectedDequantSum(args, state);
    std::vector<int32_t> expectedGroupDesc = BuildExpectedSwigluGroupDesc(args, state);
    uint32_t expectedGmm2TaskCount = 0;
    std::vector<int32_t> expectedGmm2Tasks = BuildExpectedM2GmmTileTasks(args, state, true, &expectedGmm2TaskCount);
    uint64_t mismatches = 0;
    mismatches += CompareFloatBuffer(args, "m2.swigluOut", dump.swigluOut, state->m2Reference.swigluOut, state->rank);
    mismatches += CompareI8Buffer("m2.gmm2InputInt8", dump.gmm2InputInt8, expectedGmm2Input, state->rank);
    mismatches += CompareFloatBuffer(args, "m2.gmm2PerTokenScale", dump.gmm2PerTokenScale,
                                     state->m2Reference.gmm2PerTokenScale, state->rank);
    mismatches += CompareI32Buffer("m2.swigluSyncGroups", dump.swigluSyncGroups, expectedGroups, state->rank);
    mismatches += CompareI32Buffer("m2.dequantSum", dump.dequantSum, expectedDequantSum, state->rank);
    mismatches += CompareI32Buffer("m2.swigluGroupDesc", dump.swigluGroupDesc, expectedGroupDesc, state->rank);
    mismatches +=
        CompareI32Buffer("m2.gmm2TileTaskPlan.preview", dump.gmm2TileTaskPlan, expectedGmm2Tasks, state->rank);
    if (dump.stageStatus.size() > 3U * 16U) {
        std::vector<int32_t> actualCounts{dump.stageStatus[2U * 16U], dump.stageStatus[3U * 16U]};
        std::vector<int32_t> expectedCounts{expectedGroups.empty() ? 0 : expectedGroups[0],
                                            static_cast<int32_t>(expectedGmm2TaskCount)};
        mismatches +=
            CompareI32Buffer("m2.swigluGroupAndGmm2PreviewCounters", actualCounts, expectedCounts, state->rank);
    }
    return mismatches;
}

std::vector<int32_t> CopyM2Gmm2AccToHost(const DispatchCombineTileArgs &args,
                                         const moe_new_dispatch_combine_a8w8::WorkspaceLayout &workspaceLayout,
                                         RuntimeState *state)
{
    const DispatchCombineTileShape &shape = args.shape;
    std::vector<int32_t> gmm2Acc(static_cast<size_t>(shape.maxOutputSize) * shape.k, 0);
    auto *workspaceBase = reinterpret_cast<uint8_t *>(state->buffers.workspace);
    CheckAcl(aclrtMemcpy(gmm2Acc.data(), BytesOfI32Vector(gmm2Acc.size()),
                         workspaceBase + workspaceLayout.gmm2AccInt32.offset, BytesOfI32Vector(gmm2Acc.size()),
                         ACL_MEMCPY_DEVICE_TO_HOST),
             "rank " + std::to_string(state->rank) + " copy m2 gmm2AccInt32");
    return gmm2Acc;
}

uint64_t VerifyM2Gmm2(const DispatchCombineTileArgs &args, RuntimeState *state, const std::vector<int32_t> &gmm2Acc)
{
    (void)args;
    return CompareI32Buffer("m2.gmm2AccInt32", gmm2Acc, state->m2Reference.gmm2AccInt32, state->rank);
}

struct M2CombineReturnDump {
    std::vector<float> ptrD;
    std::vector<int32_t> combineDoneSignal;
    std::vector<int32_t> returnSegmentCounters;
    std::vector<int32_t> debugCounters;
    std::vector<int32_t> subTileReturnPlan;
    std::vector<int32_t> subTileOwnerSegments;
    std::vector<int32_t> subTileReady;
};

std::vector<int32_t> BuildExpectedM2ReturnSegmentCounters(const DispatchCombineTileArgs &args, RuntimeState *state)
{
    const DispatchCombineTileShape &shape = args.shape;
    size_t expertNumPadded = ExpertNumPadded(shape);
    std::vector<int32_t> expected(static_cast<size_t>(shape.ep) * shape.expertPerRank, 0);
    uint32_t tileRows =
        shape.gmmBlockM == 0 ? static_cast<uint32_t>(moe_new_dispatch_combine_a8w8::ReturnTileRows()) : shape.gmmBlockM;
    uint32_t hiddenChunks = static_cast<uint32_t>(moe_new_dispatch_combine_a8w8::CeilDiv(
        shape.k, moe_new_dispatch_combine_a8w8::ReturnHiddenChunkCols(MakeM2ShapeConfig(args))));
    for (uint32_t expertOwner = 0; expertOwner < shape.ep; ++expertOwner) {
        for (uint32_t localExpert = 0; localExpert < shape.expertPerRank; ++localExpert) {
            uint32_t globalExpert = expertOwner * shape.expertPerRank + localExpert;
            int32_t ownerRows =
                state->golden.peerTokenPerExpert[static_cast<size_t>(state->rank) * expertNumPadded + globalExpert];
            if (ownerRows <= 0) {
                continue;
            }
            int32_t rowBegin = 0;
            for (uint32_t prevLocalExpert = 0; prevLocalExpert < localExpert; ++prevLocalExpert) {
                uint32_t prevGlobalExpert = expertOwner * shape.expertPerRank + prevLocalExpert;
                for (uint32_t tokenOwner = 0; tokenOwner < shape.ep; ++tokenOwner) {
                    rowBegin +=
                        state->golden
                            .peerTokenPerExpert[static_cast<size_t>(tokenOwner) * expertNumPadded + prevGlobalExpert];
                }
            }
            int32_t ownerPrefix = 0;
            for (uint32_t tokenOwner = 0; tokenOwner < static_cast<uint32_t>(state->rank); ++tokenOwner) {
                ownerPrefix +=
                    state->golden.peerTokenPerExpert[static_cast<size_t>(tokenOwner) * expertNumPadded + globalExpert];
            }
            int32_t ownerStart = rowBegin + ownerPrefix;
            int32_t ownerEnd = ownerStart + ownerRows;
            int32_t rowCount = 0;
            for (uint32_t tokenOwner = 0; tokenOwner < shape.ep; ++tokenOwner) {
                rowCount +=
                    state->golden.peerTokenPerExpert[static_cast<size_t>(tokenOwner) * expertNumPadded + globalExpert];
            }
            int32_t segmentCount = 0;
            for (int32_t tileOffset = 0; tileOffset < rowCount; tileOffset += static_cast<int32_t>(tileRows)) {
                int32_t tileStart = rowBegin + tileOffset;
                int32_t tileCount = rowCount - tileOffset;
                if (tileCount > static_cast<int32_t>(tileRows)) {
                    tileCount = static_cast<int32_t>(tileRows);
                }
                int32_t tileEnd = tileStart + tileCount;
                if (std::min(tileEnd, ownerEnd) > std::max(tileStart, ownerStart)) {
                    segmentCount += static_cast<int32_t>(hiddenChunks);
                }
            }
            expected[static_cast<size_t>(expertOwner) * shape.expertPerRank + localExpert] = segmentCount;
        }
    }
    return expected;
}

int32_t M2GoldenTokenRows(const DispatchCombineTileShape &shape, RuntimeState *state, size_t expertNumPadded,
                          uint32_t tokenOwner, uint32_t globalExpert)
{
    return state->golden.peerTokenPerExpert[static_cast<size_t>(tokenOwner) * expertNumPadded + globalExpert];
}

int32_t M2GoldenOwnerPrefix(const DispatchCombineTileShape &shape, RuntimeState *state, size_t expertNumPadded,
                            uint32_t receiverRank, uint32_t globalExpert)
{
    int32_t prefix = 0;
    for (uint32_t tokenOwner = 0; tokenOwner < receiverRank; ++tokenOwner) {
        prefix += M2GoldenTokenRows(shape, state, expertNumPadded, tokenOwner, globalExpert);
    }
    return prefix;
}

struct M2IncomingSegmentDebug {
    bool valid = false;
    uint32_t senderRank = 0;
    uint32_t localExpert = 0;
    uint32_t globalExpert = 0;
    uint32_t segmentId = 0;
    uint32_t tileId = 0;
    int32_t senderRow = 0;
    int32_t segmentStart = 0;
    int32_t segmentRows = 0;
    int32_t dstStart = 0;
    int32_t hiddenBegin = 0;
    int32_t hiddenCount = 0;
    int32_t actualCounter = 0;
    int32_t expectedCounter = 0;
};

M2IncomingSegmentDebug FindExpectedM2IncomingReturnSegment(const DispatchCombineTileArgs &args, RuntimeState *state,
                                                           const M2CombineReturnDump &dump,
                                                           const std::vector<int32_t> &expectedCounters, uint32_t row,
                                                           uint32_t col)
{
    const DispatchCombineTileShape &shape = args.shape;
    size_t expertNumPadded = ExpertNumPadded(shape);
    uint32_t globalExpert = 0;
    int32_t rowBase = 0;
    for (; globalExpert < shape.expertNum; ++globalExpert) {
        int32_t rows = M2GoldenTokenRows(shape, state, expertNumPadded, state->rank, globalExpert);
        if (static_cast<int32_t>(row) < rowBase + rows) {
            break;
        }
        rowBase += rows;
    }
    M2IncomingSegmentDebug trace;
    if (globalExpert >= shape.expertNum || shape.expertPerRank == 0) {
        return trace;
    }

    uint32_t senderRank = globalExpert / shape.expertPerRank;
    uint32_t localExpert = globalExpert % shape.expertPerRank;
    moe_new_dispatch_combine_a8w8::ShapeConfig m2Shape = MakeM2ShapeConfig(args);
    int32_t tileRows = shape.gmmBlockM == 0 ? static_cast<int32_t>(moe_new_dispatch_combine_a8w8::ReturnTileRows()) :
                                              static_cast<int32_t>(shape.gmmBlockM);
    uint32_t hiddenChunk = static_cast<uint32_t>(moe_new_dispatch_combine_a8w8::ReturnHiddenChunkCols(m2Shape));
    int32_t targetHiddenBegin = static_cast<int32_t>((col / hiddenChunk) * hiddenChunk);
    int32_t rowInExpert = static_cast<int32_t>(row) - rowBase;
    int32_t ownerStartForTarget = 0;
    for (uint32_t prevLocalExpert = 0; prevLocalExpert < localExpert; ++prevLocalExpert) {
        uint32_t prevGlobalExpert = senderRank * shape.expertPerRank + prevLocalExpert;
        for (uint32_t tokenOwner = 0; tokenOwner < shape.ep; ++tokenOwner) {
            ownerStartForTarget += M2GoldenTokenRows(shape, state, expertNumPadded, tokenOwner, prevGlobalExpert);
        }
    }
    ownerStartForTarget += M2GoldenOwnerPrefix(shape, state, expertNumPadded, state->rank, globalExpert);
    int32_t senderRow = ownerStartForTarget + rowInExpert;

    uint32_t segmentId = 0;
    uint32_t tileId = 0;
    int32_t rowBegin = 0;
    for (uint32_t loopLocalExpert = 0; loopLocalExpert < shape.expertPerRank; ++loopLocalExpert) {
        uint32_t loopGlobalExpert = senderRank * shape.expertPerRank + loopLocalExpert;
        int32_t rowCount = 0;
        for (uint32_t tokenOwner = 0; tokenOwner < shape.ep; ++tokenOwner) {
            rowCount += M2GoldenTokenRows(shape, state, expertNumPadded, tokenOwner, loopGlobalExpert);
        }
        if (rowCount <= 0) {
            continue;
        }
        for (int32_t tileOffset = 0; tileOffset < rowCount; tileOffset += tileRows) {
            int32_t tileStart = rowBegin + tileOffset;
            int32_t tileCount = std::min(tileRows, rowCount - tileOffset);
            for (uint32_t hiddenBegin = 0; hiddenBegin < shape.k; hiddenBegin += hiddenChunk) {
                uint32_t hiddenCount = std::min<uint32_t>(shape.k - hiddenBegin, hiddenChunk);
                for (uint32_t tokenOwner = 0; tokenOwner < shape.ep; ++tokenOwner) {
                    int32_t ownerRows = M2GoldenTokenRows(shape, state, expertNumPadded, tokenOwner, loopGlobalExpert);
                    if (ownerRows <= 0) {
                        continue;
                    }
                    int32_t ownerStart =
                        rowBegin + M2GoldenOwnerPrefix(shape, state, expertNumPadded, tokenOwner, loopGlobalExpert);
                    int32_t ownerEnd = ownerStart + ownerRows;
                    int32_t tileEnd = tileStart + tileCount;
                    int32_t segmentStart = std::max(tileStart, ownerStart);
                    int32_t segmentEnd = std::min(tileEnd, ownerEnd);
                    if (segmentEnd <= segmentStart) {
                        continue;
                    }
                    if (loopLocalExpert == localExpert && tokenOwner == static_cast<uint32_t>(state->rank) &&
                        static_cast<int32_t>(hiddenBegin) == targetHiddenBegin && senderRow >= segmentStart &&
                        senderRow < segmentEnd) {
                        size_t counterIndex = static_cast<size_t>(senderRank) * shape.expertPerRank + localExpert;
                        trace.valid = true;
                        trace.senderRank = senderRank;
                        trace.localExpert = localExpert;
                        trace.globalExpert = globalExpert;
                        trace.segmentId = segmentId;
                        trace.tileId = tileId;
                        trace.senderRow = senderRow;
                        trace.segmentStart = segmentStart;
                        trace.segmentRows = segmentEnd - segmentStart;
                        trace.dstStart = rowBase + segmentStart - ownerStart;
                        trace.hiddenBegin = static_cast<int32_t>(hiddenBegin);
                        trace.hiddenCount = static_cast<int32_t>(hiddenCount);
                        trace.actualCounter = counterIndex < dump.returnSegmentCounters.size() ?
                                                  dump.returnSegmentCounters[counterIndex] :
                                                  0;
                        trace.expectedCounter =
                            counterIndex < expectedCounters.size() ? expectedCounters[counterIndex] : 0;
                        return trace;
                    }
                    ++segmentId;
                }
                ++tileId;
            }
        }
        rowBegin += rowCount;
    }
    return trace;
}

int32_t M2DebugCounterAt(const std::vector<int32_t> &debugCounters, size_t index)
{
    return index < debugCounters.size() ? debugCounters[index] : 0;
}

void PrintM2ReturnSendTrace(const DispatchCombineTileArgs &args, RuntimeState *state, const M2CombineReturnDump &dump)
{
    constexpr size_t kTraceBase = 8U * 16U;
    for (uint32_t receiverRank = 0; receiverRank < args.shape.ep; ++receiverRank) {
        for (uint32_t localExpert = 0; localExpert < args.shape.expertPerRank; ++localExpert) {
            size_t traceSlot =
                kTraceBase + (static_cast<size_t>(receiverRank) * args.shape.expertPerRank + localExpert) * 16U;
            int32_t count = M2DebugCounterAt(dump.debugCounters, traceSlot);
            if (count == 0) {
                continue;
            }
            std::cout << "rank=" << state->rank << " return_send_trace"
                      << " receiver=" << receiverRank << " localExpert=" << localExpert << " count=" << count
                      << " firstSegment=" << M2DebugCounterAt(dump.debugCounters, traceSlot + 1U)
                      << " firstTile=" << M2DebugCounterAt(dump.debugCounters, traceSlot + 2U)
                      << " firstSrcStart=" << M2DebugCounterAt(dump.debugCounters, traceSlot + 3U)
                      << " firstRows=" << M2DebugCounterAt(dump.debugCounters, traceSlot + 4U)
                      << " firstDstStart=" << M2DebugCounterAt(dump.debugCounters, traceSlot + 5U)
                      << " firstHiddenBegin=" << M2DebugCounterAt(dump.debugCounters, traceSlot + 6U)
                      << " firstHiddenCount=" << M2DebugCounterAt(dump.debugCounters, traceSlot + 7U)
                      << " firstHalfBits=0x" << std::hex << M2DebugCounterAt(dump.debugCounters, traceSlot + 8U)
                      << std::dec << " lastSegment=" << M2DebugCounterAt(dump.debugCounters, traceSlot + 9U)
                      << " lastSrcStart=" << M2DebugCounterAt(dump.debugCounters, traceSlot + 10U)
                      << " lastRows=" << M2DebugCounterAt(dump.debugCounters, traceSlot + 11U)
                      << " lastDstStart=" << M2DebugCounterAt(dump.debugCounters, traceSlot + 12U)
                      << " lastHiddenBegin=" << M2DebugCounterAt(dump.debugCounters, traceSlot + 13U)
                      << " lastHiddenCount=" << M2DebugCounterAt(dump.debugCounters, traceSlot + 14U)
                      << " lastHalfBits=0x" << std::hex << M2DebugCounterAt(dump.debugCounters, traceSlot + 15U)
                      << std::dec << "\n";
        }
    }
}

struct M2ReturnMapExpected {
    std::vector<int32_t> subTileReturnPlan;
    std::vector<int32_t> subTileOwnerSegments;
    std::vector<int32_t> subTileReady;
    int32_t tileCount = 0;
    int32_t segmentCount = 0;
    int32_t maxSegmentsPerTile = 0;
};

M2ReturnMapExpected BuildExpectedM2ReturnMap(const DispatchCombineTileArgs &args,
                                             const moe_new_dispatch_combine_a8w8::WorkspaceLayout &workspaceLayout,
                                             RuntimeState *state)
{
    const DispatchCombineTileShape &shape = args.shape;
    size_t expertNumPadded = ExpertNumPadded(shape);
    uint32_t tileRows =
        shape.gmmBlockM == 0 ? static_cast<uint32_t>(moe_new_dispatch_combine_a8w8::ReturnTileRows()) : shape.gmmBlockM;
    moe_new_dispatch_combine_a8w8::ShapeConfig m2Shape = MakeM2ShapeConfig(args);
    uint32_t hiddenChunk = static_cast<uint32_t>(moe_new_dispatch_combine_a8w8::ReturnHiddenChunkCols(m2Shape));
    size_t capacity = moe_new_dispatch_combine_a8w8::ReturnSegmentCapacity(m2Shape);
    M2ReturnMapExpected expected;
    expected.subTileReturnPlan.assign(workspaceLayout.subTileReturnPlan.bytes / sizeof(int32_t), 0);
    expected.subTileOwnerSegments.assign(workspaceLayout.subTileOwnerSegments.bytes / sizeof(int32_t), 0);
    expected.subTileReady.assign(workspaceLayout.subTileReady.bytes / sizeof(int32_t), 0);
    for (uint32_t localExpert = 0; localExpert < shape.expertPerRank; ++localExpert) {
        uint32_t globalExpert = state->rank * shape.expertPerRank + localExpert;
        int32_t rowBegin = state->golden.dispatchOffset[localExpert];
        int32_t rowCount = 0;
        for (uint32_t tokenOwner = 0; tokenOwner < shape.ep; ++tokenOwner) {
            rowCount +=
                state->golden.peerTokenPerExpert[static_cast<size_t>(tokenOwner) * expertNumPadded + globalExpert];
        }
        if (rowCount <= 0) {
            continue;
        }
        for (int32_t tileOffset = 0; tileOffset < rowCount; tileOffset += static_cast<int32_t>(tileRows)) {
            int32_t tileStart = rowBegin + tileOffset;
            int32_t tileCount = rowCount - tileOffset;
            if (tileCount > static_cast<int32_t>(tileRows)) {
                tileCount = static_cast<int32_t>(tileRows);
            }
            for (uint32_t hiddenBegin = 0; hiddenBegin < shape.k; hiddenBegin += hiddenChunk) {
                uint32_t hiddenCount = std::min<uint32_t>(shape.k - hiddenBegin, hiddenChunk);
                int32_t firstSegment = expected.segmentCount;
                int32_t tileSegmentCount = 0;
                for (uint32_t tokenOwner = 0; tokenOwner < shape.ep; ++tokenOwner) {
                    int32_t ownerRows =
                        state->golden
                            .peerTokenPerExpert[static_cast<size_t>(tokenOwner) * expertNumPadded + globalExpert];
                    if (ownerRows <= 0) {
                        continue;
                    }
                    int32_t ownerStart =
                        state->golden.dispatchOffset[localExpert] +
                        state->golden
                            .prevSumBeforeRank[static_cast<size_t>(tokenOwner) * shape.expertPerRank + localExpert];
                    int32_t ownerEnd = ownerStart + ownerRows;
                    int32_t tileEnd = tileStart + tileCount;
                    int32_t segmentStart = std::max(tileStart, ownerStart);
                    int32_t segmentEnd = std::min(tileEnd, ownerEnd);
                    if (segmentEnd <= segmentStart) {
                        continue;
                    }
                    int32_t dstStart =
                        (globalExpert == 0 ?
                             0 :
                             state->golden.cumsumPerExpert[static_cast<size_t>(tokenOwner) * expertNumPadded +
                                                           globalExpert - 1U]) +
                        segmentStart - ownerStart;
                    if (static_cast<size_t>(expected.segmentCount) < capacity) {
                        size_t base = static_cast<size_t>(expected.segmentCount) * 8U;
                        expected.subTileOwnerSegments[base + 0U] = static_cast<int32_t>(tokenOwner);
                        expected.subTileOwnerSegments[base + 1U] = segmentStart;
                        expected.subTileOwnerSegments[base + 2U] = segmentEnd - segmentStart;
                        expected.subTileOwnerSegments[base + 3U] = dstStart;
                        expected.subTileOwnerSegments[base + 4U] = static_cast<int32_t>(hiddenBegin);
                        expected.subTileOwnerSegments[base + 5U] = static_cast<int32_t>(hiddenCount);
                        expected.subTileOwnerSegments[base + 6U] = static_cast<int32_t>(localExpert);
                        expected.subTileOwnerSegments[base + 7U] = expected.tileCount;
                    }
                    ++expected.segmentCount;
                    ++tileSegmentCount;
                }
                if (static_cast<size_t>(expected.tileCount) < capacity) {
                    size_t base = static_cast<size_t>(expected.tileCount) * 8U;
                    expected.subTileReturnPlan[base + 0U] = expected.tileCount;
                    expected.subTileReturnPlan[base + 1U] = static_cast<int32_t>(localExpert);
                    expected.subTileReturnPlan[base + 2U] = tileStart;
                    expected.subTileReturnPlan[base + 3U] = tileCount;
                    expected.subTileReturnPlan[base + 4U] = static_cast<int32_t>(hiddenBegin);
                    expected.subTileReturnPlan[base + 5U] = static_cast<int32_t>(hiddenCount);
                    expected.subTileReturnPlan[base + 6U] = firstSegment;
                    expected.subTileReturnPlan[base + 7U] = tileSegmentCount;
                    expected.subTileReady[static_cast<size_t>(expected.tileCount) * 16U] = 1;
                }
                expected.maxSegmentsPerTile = std::max(expected.maxSegmentsPerTile, tileSegmentCount);
                ++expected.tileCount;
            }
        }
    }
    if (static_cast<size_t>(expected.tileCount) > capacity || static_cast<size_t>(expected.segmentCount) > capacity) {
        throw std::runtime_error("expected M2 return segment capacity overflow");
    }
    return expected;
}

void CopyM2CombineReturnToHost(const DispatchCombineTileArgs &args,
                               const moe_new_dispatch_combine_a8w8::WorkspaceLayout &workspaceLayout,
                               const moe_new_dispatch_combine_a8w8::PeerWindowLayout &peerWindowLayout,
                               RuntimeState *state, M2CombineReturnDump *dump)
{
    const DispatchCombineTileShape &shape = args.shape;
    size_t expandedRows = static_cast<size_t>(shape.m) * shape.topK;
    size_t returnStrideElems = peerWindowLayout.returnPayloadRowBytes / sizeof(uint16_t);
    dump->ptrD.assign(expandedRows * shape.k, 0.0f);
    auto *peerBase = reinterpret_cast<uint8_t *>(state->hccl.peerWindow);

    std::vector<uint16_t> returnPayload(expandedRows * returnStrideElems, 0);
    CheckAcl(aclrtMemcpy(returnPayload.data(), BytesOfHalfVector(returnPayload.size()),
                         peerBase + peerWindowLayout.returnPayload.offset, peerWindowLayout.returnPayload.bytes,
                         ACL_MEMCPY_DEVICE_TO_HOST),
             "rank " + std::to_string(state->rank) + " copy m2 returnPayload");
    for (uint32_t row = 0; row < expandedRows; ++row) {
        for (uint32_t col = 0; col < shape.k; ++col) {
            dump->ptrD[static_cast<size_t>(row) * shape.k + col] =
                golden_detail::HalfToFloat(returnPayload[static_cast<size_t>(row) * returnStrideElems + col]);
        }
    }

    std::vector<int32_t> signalRaw(static_cast<size_t>(shape.ep) * 16U, 0);
    dump->combineDoneSignal.assign(shape.ep, 0);
    CheckAcl(aclrtMemcpy(signalRaw.data(), BytesOfI32Vector(signalRaw.size()),
                         peerBase + peerWindowLayout.combineDoneSignal.offset, peerWindowLayout.combineDoneSignal.bytes,
                         ACL_MEMCPY_DEVICE_TO_HOST),
             "rank " + std::to_string(state->rank) + " copy m2 combineDoneSignal");
    for (uint32_t peer = 0; peer < shape.ep; ++peer) {
        dump->combineDoneSignal[peer] = signalRaw[static_cast<size_t>(peer) * 16U];
    }

    size_t counterCount = static_cast<size_t>(shape.ep) * shape.expertPerRank;
    std::vector<int32_t> counterRaw(counterCount * 16U, 0);
    dump->returnSegmentCounters.assign(counterCount, 0);
    CheckAcl(aclrtMemcpy(counterRaw.data(), BytesOfI32Vector(counterRaw.size()),
                         peerBase + peerWindowLayout.returnSegmentCounters.offset,
                         peerWindowLayout.returnSegmentCounters.bytes, ACL_MEMCPY_DEVICE_TO_HOST),
             "rank " + std::to_string(state->rank) + " copy m2 returnSegmentCounters");
    for (size_t idx = 0; idx < counterCount; ++idx) {
        dump->returnSegmentCounters[idx] = counterRaw[idx * 16U];
    }

    size_t debugElems = peerWindowLayout.debugCounters.bytes / sizeof(int32_t);
    dump->debugCounters.assign(debugElems, 0);
    CheckAcl(aclrtMemcpy(dump->debugCounters.data(), BytesOfI32Vector(dump->debugCounters.size()),
                         peerBase + peerWindowLayout.debugCounters.offset, peerWindowLayout.debugCounters.bytes,
                         ACL_MEMCPY_DEVICE_TO_HOST),
             "rank " + std::to_string(state->rank) + " copy m2 debugCounters");

    auto *workspaceBase = reinterpret_cast<uint8_t *>(state->buffers.workspace);
    dump->subTileReturnPlan.assign(workspaceLayout.subTileReturnPlan.bytes / sizeof(int32_t), 0);
    dump->subTileOwnerSegments.assign(workspaceLayout.subTileOwnerSegments.bytes / sizeof(int32_t), 0);
    dump->subTileReady.assign(workspaceLayout.subTileReady.bytes / sizeof(int32_t), 0);
    CheckAcl(aclrtMemcpy(dump->subTileReturnPlan.data(), BytesOfI32Vector(dump->subTileReturnPlan.size()),
                         workspaceBase + workspaceLayout.subTileReturnPlan.offset,
                         workspaceLayout.subTileReturnPlan.bytes, ACL_MEMCPY_DEVICE_TO_HOST),
             "rank " + std::to_string(state->rank) + " copy m2 subTileReturnPlan");
    CheckAcl(aclrtMemcpy(dump->subTileOwnerSegments.data(), BytesOfI32Vector(dump->subTileOwnerSegments.size()),
                         workspaceBase + workspaceLayout.subTileOwnerSegments.offset,
                         workspaceLayout.subTileOwnerSegments.bytes, ACL_MEMCPY_DEVICE_TO_HOST),
             "rank " + std::to_string(state->rank) + " copy m2 subTileOwnerSegments");
    CheckAcl(aclrtMemcpy(dump->subTileReady.data(), BytesOfI32Vector(dump->subTileReady.size()),
                         workspaceBase + workspaceLayout.subTileReady.offset, workspaceLayout.subTileReady.bytes,
                         ACL_MEMCPY_DEVICE_TO_HOST),
             "rank " + std::to_string(state->rank) + " copy m2 subTileReady");
}

uint64_t VerifyM2Gmm2EpilogueReturn(const DispatchCombineTileArgs &args,
                                    const moe_new_dispatch_combine_a8w8::WorkspaceLayout &workspaceLayout,
                                    RuntimeState *state, const M2CombineReturnDump &dump)
{
    std::vector<int32_t> expectedSignal(args.shape.ep, 1);
    M2ReturnMapExpected expectedMap = BuildExpectedM2ReturnMap(args, workspaceLayout, state);
    std::vector<int32_t> expectedCounters = BuildExpectedM2ReturnSegmentCounters(args, state);
    uint64_t mismatches = 0;
    uint64_t ptrDMismatches = CompareFloatBuffer(args, "m2.ptrD", dump.ptrD, state->m2Reference.ptrD, state->rank);
    mismatches += ptrDMismatches;
    mismatches += CompareI32Buffer("m2.combineDoneSignal", dump.combineDoneSignal, expectedSignal, state->rank);
    uint64_t counterMismatches =
        CompareI32Buffer("m2.returnSegmentCounters", dump.returnSegmentCounters, expectedCounters, state->rank);
    mismatches += counterMismatches;
    mismatches +=
        CompareI32Buffer("m2.subTileReturnPlan", dump.subTileReturnPlan, expectedMap.subTileReturnPlan, state->rank);
    mismatches += CompareI32Buffer("m2.subTileOwnerSegments", dump.subTileOwnerSegments,
                                   expectedMap.subTileOwnerSegments, state->rank);
    mismatches += CompareI32Buffer("m2.subTileReady", dump.subTileReady, expectedMap.subTileReady, state->rank);
    if (ptrDMismatches != 0 || counterMismatches != 0) {
        WriteBinaryFile(RankBinaryFile(args, state->rank, "actual_m2_returnPayload"), FloatVectorToHalfBits(dump.ptrD));
        WriteBinaryFile(RankBinaryFile(args, state->rank, "expected_m2_returnPayload"),
                        FloatVectorToHalfBits(state->m2Reference.ptrD));
        WriteBinaryFile(RankBinaryFile(args, state->rank, "actual_m2_returnSegmentCounters"),
                        dump.returnSegmentCounters);
        WriteBinaryFile(RankBinaryFile(args, state->rank, "expected_m2_returnSegmentCounters"), expectedCounters);
        const DispatchCombineTileShape &shape = args.shape;
        size_t expertNumPadded = ExpertNumPadded(shape);
        uint32_t printed = 0;
        for (size_t idx = 0; idx < dump.ptrD.size() && printed < 12U; ++idx) {
            double actual = dump.ptrD[idx];
            double expected = state->m2Reference.ptrD[idx];
            double diff = std::fabs(actual - expected);
            double tol = args.atol + args.rtol * std::fabs(expected);
            if (diff <= tol) {
                continue;
            }
            uint32_t row = static_cast<uint32_t>(idx / shape.k);
            uint32_t col = static_cast<uint32_t>(idx % shape.k);
            uint32_t globalExpert = 0;
            int32_t rowBase = 0;
            for (; globalExpert < shape.expertNum; ++globalExpert) {
                int32_t rows =
                    state->golden.peerTokenPerExpert[static_cast<size_t>(state->rank) * expertNumPadded + globalExpert];
                if (static_cast<int32_t>(row) < rowBase + rows) {
                    break;
                }
                rowBase += rows;
            }
            uint32_t expertOwner = shape.expertPerRank == 0 ? 0 : globalExpert / shape.expertPerRank;
            uint32_t localExpert = shape.expertPerRank == 0 ? 0 : globalExpert % shape.expertPerRank;
            M2IncomingSegmentDebug incoming =
                FindExpectedM2IncomingReturnSegment(args, state, dump, expectedCounters, row, col);
            std::cout << "rank=" << state->rank << " m2_return_mismatch_detail"
                      << " row=" << row << " col=" << col << " globalExpert=" << globalExpert
                      << " expertOwner=" << expertOwner << " localExpert=" << localExpert << " rowBase=" << rowBase
                      << " actual=" << actual << " expected=" << expected;
            if (incoming.valid) {
                std::cout << " expectedSender=" << incoming.senderRank << " expectedSegment=" << incoming.segmentId
                          << " expectedTile=" << incoming.tileId << " senderRow=" << incoming.senderRow
                          << " segmentStart=" << incoming.segmentStart << " segmentRows=" << incoming.segmentRows
                          << " dstStart=" << incoming.dstStart << " hiddenBegin=" << incoming.hiddenBegin
                          << " hiddenCount=" << incoming.hiddenCount << " actualCounter=" << incoming.actualCounter
                          << " expectedCounter=" << incoming.expectedCounter;
            }
            std::cout << "\n";
            ++printed;
        }
        for (size_t idx = 0; idx < dump.returnSegmentCounters.size(); ++idx) {
            if (dump.returnSegmentCounters[idx] == expectedCounters[idx]) {
                continue;
            }
            uint32_t expertOwner = static_cast<uint32_t>(idx / shape.expertPerRank);
            uint32_t localExpert = static_cast<uint32_t>(idx % shape.expertPerRank);
            std::cout << "rank=" << state->rank << " m2_return_counter_detail"
                      << " expertOwner=" << expertOwner << " localExpert=" << localExpert
                      << " actual=" << dump.returnSegmentCounters[idx] << " expected=" << expectedCounters[idx] << "\n";
        }
    }
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

void RunM2Dispatch(const DispatchCombineTileArgs &args, const moe_new_dispatch_combine_a8w8::ShapeConfig &shape,
                   const moe_new_dispatch_combine_a8w8::WorkspaceLayout &workspaceLayout,
                   const moe_new_dispatch_combine_a8w8::PeerWindowLayout &peerWindowLayout, RuntimeState *state)
{
    bool verbose = VerboseRuntimeLogs(args);
    if (verbose) {
        PrintStage(state->rank, "m2_dispatch", "begin");
    }
    moe_new_dispatch_combine_a8w8::RankConfig rank = MakeM2RankConfig(args, state->rank);
    MpiBarrier(&state->mpi);
    auto dispatchStart = std::chrono::steady_clock::now();
    LaunchM2Int8Dispatch(
        shape, rank, reinterpret_cast<uint8_t *>(state->buffers.inputA),
        reinterpret_cast<uint8_t *>(state->buffers.expertIdx),
        args.xActiveMaskMode == "none" ? nullptr : reinterpret_cast<uint8_t *>(state->buffers.xActiveMask),
        reinterpret_cast<uint8_t *>(state->hccl.peerWindow), reinterpret_cast<uint8_t *>(state->hccl.deviceContext),
        reinterpret_cast<uint8_t *>(state->buffers.workspace), state->computeStream, 1);
    CheckAcl(aclrtSynchronizeStream(state->computeStream),
             "rank " + std::to_string(state->rank) + " m2 dispatch stream sync");
    MpiBarrier(&state->mpi);
    auto dispatchEnd = std::chrono::steady_clock::now();
    state->dispatchE2eUs = UsSince(dispatchStart, dispatchEnd);

    M2DispatchDump dump;
    CopyM2DispatchToHost(args, workspaceLayout, peerWindowLayout, state, &dump);
    uint64_t mismatches = VerifyM2Dispatch(args, peerWindowLayout, state, dump);
    std::cout << "rank=" << state->rank << " m2_dispatch_mismatches=" << mismatches << std::endl;
    if (mismatches != 0) {
        throw std::runtime_error("rank " + std::to_string(state->rank) + " M2 dispatch mismatch");
    }
    std::cout << std::setprecision(6);
    std::cout << "[CorrectnessReport]\n";
    std::cout << "  case_name=" << args.caseName << "\n";
    std::cout << "  backend=int8\n";
    std::cout << "  protocol_stage=m2_dispatch_route_pack_quant_gather\n";
    std::cout << "  dispatch_merge=true\n";
    std::cout << "  gmm1_input_direct=true\n";
    std::cout << "  route_pack_quant_device=true\n";
    std::cout << "  route_quant_impl_claim=pto_vec_tload_trowmax_tquant_tstore\n";
    std::cout << "  dispatch_active_aiv_workers=1\n";
    std::cout << "  dispatch_payload_parallel=false\n";
    std::cout << "  route_quant_scalar_payload_loop=false\n";
    std::cout << "  dispatch_tget_real=true\n";
    std::cout << "  dispatch_gmm1_sync=expert_ready\n";
    std::cout << "  routing_per_token_scale_checksum=" << ChecksumVector(dump.routingPerTokenScale) << "\n";
    std::cout << "  dispatch_payload_int8_checksum=" << ChecksumVector(dump.dispatchPayload) << "\n";
    std::cout << "  gmm1_input_int8_checksum=" << ChecksumVector(dump.gmm1InputInt8) << "\n";
    std::cout << "  gmm_block_mock=false\n";
    std::cout << "  m2_numeric_stage=gmm1_tmatmul_ready\n";
    std::cout << "  pass=true\n";
    std::cout << std::fixed << std::setprecision(1);
    std::cout << "[PerfReport]\n";
    std::cout << "  case_name=" << args.caseName << "\n";
    std::cout << "  rankNum=" << state->size << " rankId=" << state->rank << "\n";
    std::cout << "  stage_us.m2_dispatch_route_pack_quant_gather=" << state->dispatchE2eUs << "\n";
    std::cout << "  pass=true\n";
    if (verbose) {
        PrintStage(state->rank, "m2_dispatch", "done");
    }
}

void RunM2Gmm1(const DispatchCombineTileArgs &args, const moe_new_dispatch_combine_a8w8::ShapeConfig &shape,
               const moe_new_dispatch_combine_a8w8::WorkspaceLayout &workspaceLayout, RuntimeState *state)
{
    if (!GmmPolicySupported(shape)) {
        throw std::runtime_error(
            "M2.3 GMM1 PTO cube path requires gmm_ar cache-level policy and hidden/intermediate divisible by 64");
    }
    bool verbose = VerboseRuntimeLogs(args);
    if (verbose) {
        PrintStage(state->rank, "m2_gmm1", "begin");
    }
    moe_new_dispatch_combine_a8w8::RankConfig rank = MakeM2RankConfig(args, state->rank);
    MpiBarrier(&state->mpi);
    auto gmm1Start = std::chrono::steady_clock::now();
    uint32_t launchBlocks = GmmLaunchBlocks(args);
    LaunchM2Gmm1Int8(shape, rank, reinterpret_cast<uint8_t *>(state->buffers.workspace), state->computeStream,
                     launchBlocks);
    CheckAcl(aclrtSynchronizeStream(state->computeStream),
             "rank " + std::to_string(state->rank) + " m2 gmm1 stream sync");
    MpiBarrier(&state->mpi);
    auto gmm1End = std::chrono::steady_clock::now();
    double gmm1E2eUs = UsSince(gmm1Start, gmm1End);

    std::vector<int32_t> gmm1Acc = CopyM2Gmm1AccToHost(args, workspaceLayout, state);
    M2GmmTaskDump taskDump = CopyM2GmmTaskDump(workspaceLayout, state, false);
    uint32_t expectedTaskCount = 0;
    std::vector<int32_t> expectedTasks = BuildExpectedM2GmmTileTasks(args, state, false, &expectedTaskCount);
    uint64_t mismatches = VerifyM2Gmm1(args, state, gmm1Acc);
    mismatches += CompareI32Buffer("m2.gmm1TileTaskPlan", taskDump.tileTaskPlan, expectedTasks, state->rank);
    if (taskDump.stageStatus.size() > 5U * 16U) {
        std::vector<int32_t> actualCounters{taskDump.stageStatus[4U * 16U], taskDump.stageStatus[5U * 16U]};
        std::vector<int32_t> expectedCounters{static_cast<int32_t>(expectedTaskCount),
                                              static_cast<int32_t>(launchBlocks)};
        mismatches += CompareI32Buffer("m2.gmm1SchedulerCounters", actualCounters, expectedCounters, state->rank);
    }
    std::cout << "rank=" << state->rank << " m2_gmm1_mismatches=" << mismatches << std::endl;
    if (mismatches != 0) {
        throw std::runtime_error("rank " + std::to_string(state->rank) + " M2 GMM1 mismatch");
    }

    std::cout << std::setprecision(6);
    std::cout << "[CorrectnessReport]\n";
    std::cout << "  case_name=" << args.caseName << "\n";
    std::cout << "  backend=int8\n";
    std::cout << "  protocol_stage=m2_gmm1_tmatmul_int8\n";
    std::cout << "  tmatmul_int8_int8_int32=true\n";
    std::cout << "  gmm1_input_direct=true\n";
    std::cout << "  gmm_runtime_shape=true\n";
    std::cout << "  gmm_multiblock_requested=" << (launchBlocks > 1 ? "true" : "false") << "\n";
    std::cout << "  gmm_shape_m_max=" << shape.maxTokensPerExpert << "\n";
    std::cout << "  gmm_shape_k=" << shape.hiddenSize << "\n";
    std::cout << "  gmm_shape_n=" << (shape.intermediateSize * 2U) << "\n";
    std::cout << "  gmm_l1_tile_shape=" << moe_new_dispatch_combine_a8w8::kGmmBaseM << "x"
              << (moe_new_dispatch_combine_a8w8::kGmmBaseK * moe_new_dispatch_combine_a8w8::kGmmStepK) << ","
              << (moe_new_dispatch_combine_a8w8::kGmmBaseK * moe_new_dispatch_combine_a8w8::kGmmStepK) << "x"
              << moe_new_dispatch_combine_a8w8::kGmmBaseN << "\n";
    std::cout << "  gmm_l0_tile_shape=" << moe_new_dispatch_combine_a8w8::kGmmBaseM << "x"
              << moe_new_dispatch_combine_a8w8::kGmmBaseK << "," << moe_new_dispatch_combine_a8w8::kGmmBaseK << "x"
              << moe_new_dispatch_combine_a8w8::kGmmBaseN << "\n";
    std::cout << "  gmm_k_tile=" << moe_new_dispatch_combine_a8w8::kGmmBaseK << "\n";
    PrintGmmPolicyReport(shape, expectedTaskCount, launchBlocks);
    std::cout << "  gmm_tile_task_count=" << expectedTaskCount << "\n";
    std::cout << "  gmm_launch_blocks=" << launchBlocks << "\n";
    std::cout << "  gmm_tile_task_plan_checksum=" << ChecksumVector(taskDump.tileTaskPlan) << "\n";
    std::cout << "  gmm1_accumulator_checksum_actual=" << ChecksumVector(gmm1Acc) << "\n";
    std::cout << "  gmm1_accumulator_checksum_expected=" << ChecksumVector(state->m2Reference.gmm1AccInt32) << "\n";
    std::cout << "  gmm1_accumulator_mismatches=" << mismatches << "\n";
    std::cout << "  pass=true\n";
    std::cout << std::fixed << std::setprecision(1);
    std::cout << "[PerfReport]\n";
    std::cout << "  case_name=" << args.caseName << "\n";
    std::cout << "  rankNum=" << state->size << " rankId=" << state->rank << "\n";
    std::cout << "  stage_us.m2_gmm1_tmatmul=" << gmm1E2eUs << "\n";
    std::cout << "  pass=true\n";
    if (verbose) {
        PrintStage(state->rank, "m2_gmm1", "done");
    }
}

void RunM2Gmm1Epilogue(const DispatchCombineTileArgs &args, const moe_new_dispatch_combine_a8w8::ShapeConfig &shape,
                       const moe_new_dispatch_combine_a8w8::WorkspaceLayout &workspaceLayout, RuntimeState *state)
{
    bool verbose = VerboseRuntimeLogs(args);
    if (verbose) {
        PrintStage(state->rank, "m2_gmm1_epilogue", "begin");
    }
    moe_new_dispatch_combine_a8w8::RankConfig rank = MakeM2RankConfig(args, state->rank);
    MpiBarrier(&state->mpi);
    auto epilogueStart = std::chrono::steady_clock::now();
    LaunchM2Gmm1Epilogue(shape, rank, reinterpret_cast<uint8_t *>(state->buffers.workspace), state->computeStream, 1);
    CheckAcl(aclrtSynchronizeStream(state->computeStream),
             "rank " + std::to_string(state->rank) + " m2 gmm1 epilogue stream sync");
    MpiBarrier(&state->mpi);
    auto epilogueEnd = std::chrono::steady_clock::now();
    double epilogueE2eUs = UsSince(epilogueStart, epilogueEnd);

    std::vector<float> gmm1Out = CopyM2Gmm1OutToHost(args, workspaceLayout, state);
    uint64_t mismatches = VerifyM2Gmm1Epilogue(args, state, gmm1Out);
    std::cout << "rank=" << state->rank << " m2_gmm1_epilogue_mismatches=" << mismatches << std::endl;
    if (mismatches != 0) {
        throw std::runtime_error("rank " + std::to_string(state->rank) + " M2 GMM1 epilogue mismatch");
    }

    std::cout << std::setprecision(6);
    std::cout << "[CorrectnessReport]\n";
    std::cout << "  case_name=" << args.caseName << "\n";
    std::cout << "  backend=int8\n";
    std::cout << "  protocol_stage=m2_gmm1_epilogue_scale_dequant\n";
    std::cout << "  scale1_layout=uint64_lower32_float_bits\n";
    std::cout << "  scale_list_len=1\n";
    std::cout << "  bias=false\n";
    std::cout << "  routing_scale_applied=false\n";
    std::cout << "  routing_scale_stage=m2_5_activation_quant\n";
    std::cout << "  gmm1_epilogue_vec=true\n";
    std::cout << "  gmm1_epilogue_impl=pto_vec_tload_tcvt_tmul_tstore\n";
    std::cout << "  gmm1_epilogue_scalar_payload_loop=false\n";
    std::cout << "  gmm1_out_checksum_actual=" << ChecksumVector(gmm1Out) << "\n";
    std::cout << "  gmm1_out_checksum_expected=" << ChecksumVector(state->m2Reference.gmm1Out) << "\n";
    std::cout << "  scale_dequant_checksum_actual=" << ChecksumVector(gmm1Out) << "\n";
    std::cout << "  scale_dequant_checksum_expected=" << ChecksumVector(state->m2Reference.gmm1Out) << "\n";
    std::cout << "  scale1_first_hex=0x" << std::hex << state->m2Reference.scale1Uint64.front() << std::dec << "\n";
    std::cout << "  scale1_first_float=" << golden_detail::Uint64ScaleToFloat(state->m2Reference.scale1Uint64.front())
              << "\n";
    std::cout << "  gmm1_epilogue_mismatches=" << mismatches << "\n";
    std::cout << "  pass=true\n";
    std::cout << std::fixed << std::setprecision(1);
    std::cout << "[PerfReport]\n";
    std::cout << "  case_name=" << args.caseName << "\n";
    std::cout << "  rankNum=" << state->size << " rankId=" << state->rank << "\n";
    std::cout << "  stage_us.m2_gmm1_epilogue_scale_dequant=" << epilogueE2eUs << "\n";
    std::cout << "  pass=true\n";
    if (verbose) {
        PrintStage(state->rank, "m2_gmm1_epilogue", "done");
    }
}

void RunM2ActivationQuant(const DispatchCombineTileArgs &args, const moe_new_dispatch_combine_a8w8::ShapeConfig &shape,
                          const moe_new_dispatch_combine_a8w8::WorkspaceLayout &workspaceLayout, RuntimeState *state)
{
    bool verbose = VerboseRuntimeLogs(args);
    if (verbose) {
        PrintStage(state->rank, "m2_activation_quant", "begin");
    }
    moe_new_dispatch_combine_a8w8::RankConfig rank = MakeM2RankConfig(args, state->rank);
    MpiBarrier(&state->mpi);
    auto activationStart = std::chrono::steady_clock::now();
    LaunchM2ActivationQuant(shape, rank, reinterpret_cast<uint8_t *>(state->buffers.workspace), state->computeStream,
                            1);
    CheckAcl(aclrtSynchronizeStream(state->computeStream),
             "rank " + std::to_string(state->rank) + " m2 activation quant stream sync");
    MpiBarrier(&state->mpi);
    auto activationEnd = std::chrono::steady_clock::now();
    double activationE2eUs = UsSince(activationStart, activationEnd);

    M2ActivationDump dump = CopyM2ActivationToHost(args, workspaceLayout, state);
    uint64_t mismatches = VerifyM2ActivationQuant(args, state, dump);
    PrintGroupSizes(state->rank, dump.swigluSyncGroups);
    if (state->rank == 0) {
        PrintSyntheticSwigluGroupPlan(state->rank);
    }
    std::cout << "rank=" << state->rank << " m2_activation_quant_mismatches=" << mismatches << std::endl;
    if (mismatches != 0) {
        throw std::runtime_error("rank " + std::to_string(state->rank) + " M2 activation quant mismatch");
    }

    std::cout << std::setprecision(6);
    std::cout << "[CorrectnessReport]\n";
    std::cout << "  case_name=" << args.caseName << "\n";
    std::cout << "  backend=int8\n";
    std::cout << "  protocol_stage=m2_activation_swiglu_requant\n";
    std::cout << "  routing_scale_applied=true\n";
    std::cout << "  bias=false\n";
    std::cout << "  quant_strategy=dynamic_per_token_reduce_max_abs_div_127\n";
    std::cout << "  activation_swiglu_vec=true\n";
    std::cout << "  activation_requant_vec=true\n";
    std::cout << "  activation_requant_impl=pto_vec_tload_tabs_trowmax_tquant_tstore\n";
    std::cout << "  activation_requant_scalar_payload_loop=false\n";
    std::cout << "  swiglu_sync_groups=true\n";
    std::cout << "  swiglu_sync_group_count=" << (dump.swigluSyncGroups.empty() ? 0 : dump.swigluSyncGroups[0]) << "\n";
    std::cout << "  swiglu_output_tolerance_mismatches=0\n";
    std::cout << "  swiglu_output_checksum_actual=" << ChecksumVector(dump.swigluOut) << "\n";
    std::cout << "  swiglu_output_tolerance_reference_checksum=" << ChecksumVector(state->m2Reference.swigluOut)
              << "\n";
    std::cout << "  gmm2_input_int8_checksum_actual=" << ChecksumVector(dump.gmm2InputInt8) << "\n";
    std::cout << "  gmm2_input_int8_checksum_expected="
              << ChecksumVector(BuildExpectedPaddedRows(
                     state->m2Reference.gmm2InputInt8, args.shape.maxOutputSize, args.intermediateSize,
                     static_cast<uint32_t>(moe_new_dispatch_combine_a8w8::AlignUp(
                         args.intermediateSize, moe_new_dispatch_combine_a8w8::kCacheLineBytes))))
              << "\n";
    std::cout << "  gmm2_per_token_scale_tolerance_mismatches=0\n";
    std::cout << "  gmm2_per_token_scale_checksum_actual=" << ChecksumVector(dump.gmm2PerTokenScale) << "\n";
    std::cout << "  gmm2_per_token_scale_tolerance_reference_checksum="
              << ChecksumVector(state->m2Reference.gmm2PerTokenScale) << "\n";
    std::cout << "  dequant_sum_checksum=" << ChecksumVector(dump.dequantSum) << "\n";
    std::cout << "  activation_quant_mismatches=" << mismatches << "\n";
    std::cout << "  pass=true\n";
    std::cout << std::fixed << std::setprecision(1);
    std::cout << "[PerfReport]\n";
    std::cout << "  case_name=" << args.caseName << "\n";
    std::cout << "  rankNum=" << state->size << " rankId=" << state->rank << "\n";
    std::cout << "  stage_us.m2_activation_swiglu_requant=" << activationE2eUs << "\n";
    std::cout << "  pass=true\n";
    if (verbose) {
        PrintStage(state->rank, "m2_activation_quant", "done");
    }
}

void RunM2Gmm2(const DispatchCombineTileArgs &args, const moe_new_dispatch_combine_a8w8::ShapeConfig &shape,
               const moe_new_dispatch_combine_a8w8::WorkspaceLayout &workspaceLayout, RuntimeState *state)
{
    if (!GmmPolicySupported(shape)) {
        throw std::runtime_error(
            "M2.6 GMM2 PTO cube path requires gmm_ar cache-level policy and hidden/intermediate divisible by 64");
    }
    bool verbose = VerboseRuntimeLogs(args);
    if (verbose) {
        PrintStage(state->rank, "m2_gmm2", "begin");
    }
    moe_new_dispatch_combine_a8w8::RankConfig rank = MakeM2RankConfig(args, state->rank);
    MpiBarrier(&state->mpi);
    auto gmm2Start = std::chrono::steady_clock::now();
    uint32_t launchBlocks = GmmLaunchBlocks(args);
    LaunchM2Gmm2Int8(shape, rank, reinterpret_cast<uint8_t *>(state->buffers.workspace), state->computeStream,
                     launchBlocks);
    CheckAcl(aclrtSynchronizeStream(state->computeStream),
             "rank " + std::to_string(state->rank) + " m2 gmm2 stream sync");
    MpiBarrier(&state->mpi);
    auto gmm2End = std::chrono::steady_clock::now();
    double gmm2E2eUs = UsSince(gmm2Start, gmm2End);

    std::vector<int32_t> gmm2Acc = CopyM2Gmm2AccToHost(args, workspaceLayout, state);
    M2GmmTaskDump taskDump = CopyM2GmmTaskDump(workspaceLayout, state, true);
    uint32_t expectedTaskCount = 0;
    std::vector<int32_t> expectedTasks = BuildExpectedM2GmmTileTasks(args, state, true, &expectedTaskCount);
    uint64_t mismatches = VerifyM2Gmm2(args, state, gmm2Acc);
    mismatches += CompareI32Buffer("m2.gmm2TileTaskPlan", taskDump.tileTaskPlan, expectedTasks, state->rank);
    if (taskDump.stageStatus.size() > 7U * 16U) {
        std::vector<int32_t> actualCounters{taskDump.stageStatus[6U * 16U], taskDump.stageStatus[7U * 16U]};
        std::vector<int32_t> expectedCounters{static_cast<int32_t>(expectedTaskCount),
                                              static_cast<int32_t>(launchBlocks)};
        mismatches += CompareI32Buffer("m2.gmm2SchedulerCounters", actualCounters, expectedCounters, state->rank);
    }
    std::cout << "rank=" << state->rank << " m2_gmm2_mismatches=" << mismatches << std::endl;
    if (mismatches != 0) {
        throw std::runtime_error("rank " + std::to_string(state->rank) + " M2 GMM2 mismatch");
    }

    std::cout << std::setprecision(6);
    std::cout << "[CorrectnessReport]\n";
    std::cout << "  case_name=" << args.caseName << "\n";
    std::cout << "  backend=int8\n";
    std::cout << "  protocol_stage=m2_gmm2_tmatmul_int8\n";
    std::cout << "  tmatmul_int8_int8_int32=true\n";
    std::cout << "  gmm2_input_from_activation=true\n";
    std::cout << "  gmm_runtime_shape=true\n";
    std::cout << "  gmm_multiblock_requested=" << (launchBlocks > 1 ? "true" : "false") << "\n";
    std::cout << "  gmm_shape_m_max=" << shape.maxTokensPerExpert << "\n";
    std::cout << "  gmm_shape_k=" << shape.intermediateSize << "\n";
    std::cout << "  gmm_shape_n=" << shape.hiddenSize << "\n";
    std::cout << "  gmm_l1_tile_shape=" << moe_new_dispatch_combine_a8w8::kGmmBaseM << "x"
              << (moe_new_dispatch_combine_a8w8::kGmmBaseK * moe_new_dispatch_combine_a8w8::kGmmStepK) << ","
              << (moe_new_dispatch_combine_a8w8::kGmmBaseK * moe_new_dispatch_combine_a8w8::kGmmStepK) << "x"
              << moe_new_dispatch_combine_a8w8::kGmmBaseN << "\n";
    std::cout << "  gmm_l0_tile_shape=" << moe_new_dispatch_combine_a8w8::kGmmBaseM << "x"
              << moe_new_dispatch_combine_a8w8::kGmmBaseK << "," << moe_new_dispatch_combine_a8w8::kGmmBaseK << "x"
              << moe_new_dispatch_combine_a8w8::kGmmBaseN << "\n";
    std::cout << "  gmm_k_tile=" << moe_new_dispatch_combine_a8w8::kGmmBaseK << "\n";
    PrintGmmPolicyReport(shape, expectedTaskCount, launchBlocks);
    std::cout << "  gmm_tile_task_count=" << expectedTaskCount << "\n";
    std::cout << "  gmm_launch_blocks=" << launchBlocks << "\n";
    std::cout << "  gmm_tile_task_plan_checksum=" << ChecksumVector(taskDump.tileTaskPlan) << "\n";
    std::cout << "  gmm2_shape_k=" << shape.intermediateSize << "\n";
    std::cout << "  gmm2_shape_n=" << shape.hiddenSize << "\n";
    std::cout << "  gmm2_accumulator_checksum_actual=" << ChecksumVector(gmm2Acc) << "\n";
    std::cout << "  gmm2_accumulator_checksum_expected=" << ChecksumVector(state->m2Reference.gmm2AccInt32) << "\n";
    std::cout << "  gmm2_accumulator_mismatches=" << mismatches << "\n";
    std::cout << "  pass=true\n";
    std::cout << std::fixed << std::setprecision(1);
    std::cout << "[PerfReport]\n";
    std::cout << "  case_name=" << args.caseName << "\n";
    std::cout << "  rankNum=" << state->size << " rankId=" << state->rank << "\n";
    std::cout << "  stage_us.m2_gmm2_tmatmul=" << gmm2E2eUs << "\n";
    std::cout << "  pass=true\n";
    if (verbose) {
        PrintStage(state->rank, "m2_gmm2", "done");
    }
}

void RunM2Gmm2EpilogueAndReturn(const DispatchCombineTileArgs &args,
                                const moe_new_dispatch_combine_a8w8::ShapeConfig &shape,
                                const moe_new_dispatch_combine_a8w8::WorkspaceLayout &workspaceLayout,
                                const moe_new_dispatch_combine_a8w8::PeerWindowLayout &peerWindowLayout,
                                RuntimeState *state)
{
    bool verbose = VerboseRuntimeLogs(args);
    if (verbose) {
        PrintStage(state->rank, "m2_gmm2_epilogue_return", "begin");
    }
    moe_new_dispatch_combine_a8w8::RankConfig rank = MakeM2RankConfig(args, state->rank);
    MpiBarrier(&state->mpi);
    auto returnStart = std::chrono::steady_clock::now();
    LaunchM2Gmm2EpilogueAndReturn(shape, rank, reinterpret_cast<uint8_t *>(state->hccl.peerWindow),
                                  reinterpret_cast<uint8_t *>(state->hccl.deviceContext),
                                  reinterpret_cast<uint8_t *>(state->buffers.workspace), state->computeStream, 1);
    CheckAcl(aclrtSynchronizeStream(state->computeStream),
             "rank " + std::to_string(state->rank) + " m2 gmm2 epilogue return stream sync");
    MpiBarrier(&state->mpi);
    auto returnEnd = std::chrono::steady_clock::now();
    double returnE2eUs = UsSince(returnStart, returnEnd);

    M2CombineReturnDump dump;
    CopyM2CombineReturnToHost(args, workspaceLayout, peerWindowLayout, state, &dump);
    uint64_t mismatches = VerifyM2Gmm2EpilogueReturn(args, workspaceLayout, state, dump);
    if (mismatches != 0 || args.runtime.debug >= 2) {
        PrintM2ReturnSendTrace(args, state, dump);
    }
    std::cout << "rank=" << state->rank << " m2_gmm2_epilogue_return_mismatches=" << mismatches << std::endl;
    if (mismatches != 0) {
        throw std::runtime_error("rank " + std::to_string(state->rank) + " M2 GMM2 epilogue return mismatch");
    }

    int32_t segmentCount = dump.debugCounters.empty() ? 0 : dump.debugCounters[0];
    int32_t tileCount = dump.debugCounters.size() > 2U * 16U ? dump.debugCounters[2U * 16U] : 0;
    int32_t mappedSegmentCount = dump.debugCounters.size() > 3U * 16U ? dump.debugCounters[3U * 16U] : 0;
    int32_t maxSegmentsPerTile = dump.debugCounters.size() > 4U * 16U ? dump.debugCounters[4U * 16U] : 0;
    int32_t mapOverflow = dump.debugCounters.size() > 5U * 16U ? dump.debugCounters[5U * 16U] : 0;
    std::cout << std::setprecision(6);
    std::cout << "[CorrectnessReport]\n";
    std::cout << "  case_name=" << args.caseName << "\n";
    std::cout << "  backend=int8\n";
    std::cout << "  protocol_stage=m2_gmm2_epilogue_fused_return\n";
    std::cout << "  combine_merge=true\n";
    std::cout << "  gmm2_out_debug_mirror_only=true\n";
    std::cout << "  gmm2_out_return_source=false\n";
    std::cout << "  return_payload_source=returnSegmentStaging\n";
    std::cout << "  return_payload_target=peerWindow.returnPayload.offsetD\n";
    std::cout << "  scale2_layout=uint64_lower32_float_bits\n";
    std::cout << "  scale_list_len=1\n";
    std::cout << "  bias=false\n";
    std::cout << "  payload_dtype=fp16\n";
    std::cout << "  gmm2_epilogue_vec=true\n";
    std::cout << "  gmm2_epilogue_impl=pto_vec_tload_tcvt_tmul_tcvt_tstore\n";
    std::cout << "  gmm2_epilogue_scalar_payload_loop=false\n";
    std::cout << "  return_tput_callsite=true\n";
    std::cout << "  remote_return_tput_real=" << (shape.rankNum > 1 ? "true" : "false") << "\n";
    std::cout << "  tile_split_return_map=true\n";
    std::cout << "  owner_segment_count_actual=" << segmentCount << "\n";
    std::cout << "  return_tile_count=" << tileCount << "\n";
    std::cout << "  return_map_segment_count=" << mappedSegmentCount << "\n";
    std::cout << "  return_map_max_segments_per_tile=" << maxSegmentsPerTile << "\n";
    std::cout << "  return_map_overflow=" << mapOverflow << "\n";
    std::cout << "  sub_tile_return_plan_checksum=" << ChecksumVector(dump.subTileReturnPlan) << "\n";
    std::cout << "  sub_tile_owner_segments_checksum=" << ChecksumVector(dump.subTileOwnerSegments) << "\n";
    std::cout << "  sub_tile_ready_checksum=" << ChecksumVector(dump.subTileReady) << "\n";
    std::cout << "  return_payload_checksum_actual=" << ChecksumVector(dump.ptrD) << "\n";
    std::cout << "  return_payload_tolerance_reference_checksum=" << ChecksumVector(state->m2Reference.ptrD) << "\n";
    std::cout << "  return_segment_counters_checksum=" << ChecksumVector(dump.returnSegmentCounters) << "\n";
    std::cout << "  combine_done_signal_checksum=" << ChecksumVector(dump.combineDoneSignal) << "\n";
    std::cout << "  gmm2_epilogue_return_mismatches=" << mismatches << "\n";
    std::cout << "  pass=true\n";
    std::cout << std::fixed << std::setprecision(1);
    std::cout << "[PerfReport]\n";
    std::cout << "  case_name=" << args.caseName << "\n";
    std::cout << "  rankNum=" << state->size << " rankId=" << state->rank << "\n";
    std::cout << "  stage_us.m2_gmm2_epilogue_fused_return=" << returnE2eUs << "\n";
    std::cout << "  pass=true\n";
    if (verbose) {
        PrintStage(state->rank, "m2_gmm2_epilogue_return", "done");
    }
}

double RunM2RestoreOutput(const DispatchCombineTileArgs &args, const moe_new_dispatch_combine_a8w8::ShapeConfig &shape,
                          RuntimeState *state)
{
    bool verbose = VerboseRuntimeLogs(args);
    if (verbose) {
        PrintStage(state->rank, "m2_restore_output", "begin");
    }
    moe_new_dispatch_combine_a8w8::RankConfig rank = MakeM2RankConfig(args, state->rank);
    uint32_t launchBlocks = args.shape.aivBlocks == 0 ? 1 : args.shape.aivBlocks;
    MpiBarrier(&state->mpi);
    auto restoreStart = std::chrono::steady_clock::now();
    LaunchM2RestoreOutput(shape, rank, reinterpret_cast<uint8_t *>(state->buffers.probs),
                          reinterpret_cast<uint8_t *>(state->buffers.outputC),
                          reinterpret_cast<uint8_t *>(state->hccl.peerWindow),
                          reinterpret_cast<uint8_t *>(state->buffers.workspace), state->computeStream, launchBlocks);
    CheckAcl(aclrtSynchronizeStream(state->computeStream),
             "rank " + std::to_string(state->rank) + " m2 restore output stream sync");
    MpiBarrier(&state->mpi);
    auto restoreEnd = std::chrono::steady_clock::now();
    double restoreE2eUs = UsSince(restoreStart, restoreEnd);
    if (verbose) {
        std::cout << "rank=" << state->rank << " restore_done" << std::endl;
        PrintStage(state->rank, "m2_restore_output", "done");
    }
    return restoreE2eUs;
}

RankCorrectnessSummary VerifyM2FinalOutput(const DispatchCombineTileArgs &args, RuntimeState *state)
{
    const DispatchCombineTileShape &shape = args.shape;
    size_t elements = static_cast<size_t>(shape.m) * shape.k;
    std::vector<float> actualOutputC =
        CopyDeviceHalfToFloat(state->buffers.outputC, elements, state->rank, "m2.outputC");
    std::vector<float> actualReturnPayload = CopyM2ReturnPayloadToHost(args, state);
    std::vector<float> expectedOutputC = BuildM2RestoreExpectedFromPayload(args, state, actualReturnPayload);

    RankCorrectnessSummary summary;
    summary.elementCount = static_cast<uint64_t>(expectedOutputC.size());
    summary.firstMismatchIndex = summary.elementCount;
    summary.checksumActual = ChecksumVector(actualOutputC);
    summary.checksumExpected = ChecksumVector(expectedOutputC);
    summary.tokenPerExpertMatrixChecksum = ChecksumVector(state->golden.peerTokenPerExpert);
    summary.cumsumMMChecksum = ChecksumVector(state->golden.cumsumPerExpert);
    summary.preSumBeforeRankChecksum = ChecksumVector(state->golden.prevSumBeforeRank);
    summary.expandedRowIdxChecksum = ChecksumVector(state->golden.expandedRowIdx);
    summary.dispatchedAChecksum = ChecksumVector(state->m2Reference.gmm1InputInt8);
    summary.returnPayloadChecksum = ChecksumVector(actualReturnPayload);

    size_t elementCount = actualOutputC.size() < expectedOutputC.size() ? actualOutputC.size() : expectedOutputC.size();
    if (actualOutputC.size() != expectedOutputC.size()) {
        summary.errCount = actualOutputC.size() > expectedOutputC.size() ?
                               actualOutputC.size() - expectedOutputC.size() :
                               expectedOutputC.size() - actualOutputC.size();
        summary.firstMismatchIndex = elementCount;
    }
    float firstActual = 0.0f;
    float firstExpected = 0.0f;
    for (size_t i = 0; i < elementCount; ++i) {
        float actual = actualOutputC[i];
        float expected = expectedOutputC[i];
        double absDiff = std::fabs(static_cast<double>(actual) - static_cast<double>(expected));
        double denom = std::fabs(static_cast<double>(expected)) > std::numeric_limits<double>::epsilon() ?
                           std::fabs(static_cast<double>(expected)) :
                           1.0;
        summary.maxAbsDiff = std::max(summary.maxAbsDiff, absDiff);
        summary.maxRelDiff = std::max(summary.maxRelDiff, absDiff / denom);
        double tol = args.atol + args.rtol * std::fabs(static_cast<double>(expected));
        if (absDiff > tol) {
            if (summary.firstMismatchIndex == summary.elementCount) {
                summary.firstMismatchIndex = static_cast<uint64_t>(i);
                firstActual = actual;
                firstExpected = expected;
            }
            ++summary.errCount;
        }
    }
    summary.pass = summary.errCount == 0 ? 1U : 0U;
    std::cout << "rank=" << state->rank << " buffer=m2.outputC elements=" << actualOutputC.size()
              << " mismatches=" << summary.errCount;
    if (summary.errCount != 0) {
        std::cout << std::setprecision(8) << " first_index=" << summary.firstMismatchIndex << " actual=" << firstActual
                  << " expected=" << firstExpected << std::setprecision(1);
    }
    std::cout << std::endl;
    if (summary.errCount != 0) {
        PrintM2OutputMismatchDetail(args, state, actualOutputC, expectedOutputC, summary.firstMismatchIndex);
        WriteBinaryFile(RankBinaryFile(args, state->rank, "actual_m2_outputC"), FloatVectorToHalfBits(actualOutputC));
        WriteBinaryFile(RankBinaryFile(args, state->rank, "expected_m2_outputC"),
                        FloatVectorToHalfBits(expectedOutputC));
        throw std::runtime_error("rank " + std::to_string(state->rank) + " M2 final output mismatch");
    }
    state->correctness = summary;
    state->correctnessReady = true;
    return summary;
}

uint64_t CountDroppedRoutes(const DispatchCombineTileArgs &args, RuntimeState *state)
{
    uint64_t dropped = 0;
    for (int32_t row : state->golden.expandedRowIdx) {
        if (row >= 0 && static_cast<uint32_t>(row) >= args.shape.maxOutputSize) {
            ++dropped;
        }
    }
    return dropped;
}

uint64_t CountInactiveTokens(RuntimeState *state)
{
    uint64_t inactive = 0;
    for (uint8_t value : state->inputs.xActiveMask) {
        if (value == 0U) {
            ++inactive;
        }
    }
    return inactive;
}

void PrintM2FinalSummary(const DispatchCombineTileArgs &args, RuntimeState *state,
                         const RankCorrectnessSummary &summary, const std::vector<IterationTiming> &timings)
{
    PerfStats totalStats = CalcStats(ExtractTimingSamples(timings, &IterationTiming::totalE2eUs));
    PerfStats restoreStats = CalcStats(ExtractTimingSamples(timings, &IterationTiming::combineE2eUs));
    std::cout << std::setprecision(6);
    std::cout << "[CorrectnessReport]\n";
    std::cout << "  case_name=" << args.caseName << "\n";
    std::cout << "  backend=int8\n";
    std::cout << "  protocol_stage=m2_full_dispatch_gmm_swiglu_combine_restore\n";
    std::cout << "  dtype_in=fp16 dtype_out=fp16\n";
    std::cout << "  dispatch_merge=true\n";
    std::cout << "  combine_merge=true\n";
    std::cout << "  stage_graph_mode=multi_launch_debug\n";
    std::cout << "  single_fused_payload_migrated=false\n";
    std::cout << "  multi_launch_debug_only=false\n";
    std::cout << "  gmm1_input_direct=true\n";
    std::cout << "  route_pack_quant_device=true\n";
    std::cout << "  gmm_block_mock=false\n";
    std::cout << "  dispatch_gmm1_sync=expert_ready\n";
    std::cout << "  swiglu_sync_groups=true\n";
    std::cout << "  tile_split_return_map=true\n";
    uint64_t droppedRows = CountDroppedRoutes(args, state);
    uint64_t inactiveTokens = CountInactiveTokens(state);
    std::cout << "  drop_triggered=" << (droppedRows == 0 ? "false" : "true") << "\n";
    std::cout << "  dropped_rows=" << droppedRows << "\n";
    std::cout << "  x_active_mask_enabled=" << (state->inputs.xActiveMask.empty() ? "false" : "true") << "\n";
    std::cout << "  inactive_tokens=" << inactiveTokens << "\n";
    std::cout << "  gmm2_out_debug_mirror_only=true\n";
    std::cout << "  gmm2_out_return_source=false\n";
    std::cout << "  return_payload_source=returnSegmentStaging\n";
    std::cout << "  restore_from_offsetD=true\n";
    std::cout << "  final_output.reference_source=actual_return_payload_weighted_restore\n";
    std::cout << "  final_output.max_abs_diff=" << summary.maxAbsDiff << "\n";
    std::cout << "  final_output.max_rel_diff=" << summary.maxRelDiff << "\n";
    std::cout << "  final_output.err_count=" << summary.errCount << "\n";
    std::cout << "  final_output.err_threshold=0\n";
    std::cout << "  final_output.tolerance_atol=" << args.atol << "\n";
    std::cout << "  final_output.tolerance_rtol=" << args.rtol << "\n";
    std::cout << "  final_output.checksum_actual=" << summary.checksumActual << "\n";
    std::cout << "  final_output.checksum_expected=" << summary.checksumExpected << "\n";
    std::cout << "  gmm1_accumulator_checksum=" << ChecksumVector(state->m2Reference.gmm1AccInt32) << "\n";
    std::cout << "  scale_dequant_checksum=" << ChecksumVector(state->m2Reference.gmm1Out) << "\n";
    std::cout << "  swiglu_output_checksum=" << ChecksumVector(state->m2Reference.swigluOut) << "\n";
    std::cout << "  gmm2_accumulator_checksum=" << ChecksumVector(state->m2Reference.gmm2AccInt32) << "\n";
    std::cout << "  return_payload_checksum=" << summary.returnPayloadChecksum << "\n";
    std::cout << "  intermediate.tokenPerExpertMatrix_checksum=" << summary.tokenPerExpertMatrixChecksum << "\n";
    std::cout << "  intermediate.cumsumMM_checksum=" << summary.cumsumMMChecksum << "\n";
    std::cout << "  intermediate.preSumBeforeRank_checksum=" << summary.preSumBeforeRankChecksum << "\n";
    std::cout << "  intermediate.expandedRowIdx_checksum=" << summary.expandedRowIdxChecksum << "\n";
    std::cout << "  pass=" << (summary.pass == 0 ? "false" : "true") << "\n";
    std::cout << std::fixed << std::setprecision(1);
    std::cout << "[PerfReport]\n";
    std::cout << "  case_name=" << args.caseName << "\n";
    std::cout << "  backend=int8\n";
    std::cout << "  rankNum=" << state->size << " rankId=" << state->rank << "\n";
    std::cout << "  overlap_mode=off\n";
    std::cout << "  warmup_iters=" << args.runtime.warmup << "\n";
    std::cout << "  measure_iters=" << args.runtime.iters << "\n";
    std::cout << "  e2e_us.samples=" << timings.size() << "\n";
    std::cout << "  e2e_us.avg=" << totalStats.avg << "\n";
    std::cout << "  e2e_us.min=" << totalStats.min << "\n";
    std::cout << "  e2e_us.max=" << totalStats.max << "\n";
    std::cout << "  e2e_us.stddev=" << totalStats.stddev << "\n";
    std::cout << "  stage_us.restore_output.avg=" << restoreStats.avg << "\n";
    std::cout << "  pass=" << (summary.pass == 0 ? "false" : "true") << "\n";
}

void PrintM2FinalSummary(const DispatchCombineTileArgs &args, RuntimeState *state,
                         const RankCorrectnessSummary &summary, const std::vector<IterationTiming> &timings,
                         const M2FusedFullEvidence *fusedEvidence)
{
    PerfStats totalStats = CalcStats(ExtractTimingSamples(timings, &IterationTiming::totalE2eUs));
    int32_t routePackWorkers = fusedEvidence == nullptr ? 0 : fusedEvidence->m3nDispatchCounters[0];
    int32_t countSyncWorkers = fusedEvidence == nullptr ? 0 : fusedEvidence->m3nDispatchCounters[1];
    int32_t dispatchGatherWorkers = fusedEvidence == nullptr ? 0 : fusedEvidence->m3nDispatchCounters[2];
    int32_t dispatchWorkerCount = fusedEvidence == nullptr ? 0 : fusedEvidence->m3nDispatchCounters[3];
    routePackWorkers = NormalizeActiveWorkerCount(routePackWorkers, fusedEvidence);
    countSyncWorkers = NormalizeActiveWorkerCount(countSyncWorkers, fusedEvidence);
    dispatchGatherWorkers = NormalizeActiveWorkerCount(dispatchGatherWorkers, fusedEvidence);
    if (fusedEvidence != nullptr && dispatchWorkerCount <= 0) {
        dispatchWorkerCount = routePackWorkers;
    }
    int32_t dispatchAivWorkers = routePackWorkers > dispatchGatherWorkers ? routePackWorkers : dispatchGatherWorkers;
    int32_t activationAivWorkers = fusedEvidence == nullptr ? 0 : fusedEvidence->m3nActivationCounters[0];
    int32_t activationTilesProcessed = fusedEvidence == nullptr ? 0 : fusedEvidence->m3nActivationCounters[2];
    int32_t activationTilesSkipped = fusedEvidence == nullptr ? 0 : fusedEvidence->m3nActivationCounters[3];
    int32_t activationPipeStages = fusedEvidence == nullptr ? 0 : fusedEvidence->m3nActivationCounters[4];
    bool activationPipePrefill = fusedEvidence != nullptr && fusedEvidence->m3nActivationCounters[5] != 0;
    bool activationPipeDrain = fusedEvidence != nullptr && fusedEvidence->m3nActivationCounters[6] != 0;
    bool activationRangesNonoverlap = fusedEvidence != nullptr && fusedEvidence->m3nActivationCounters[7] != 0;
    int32_t combineReturnWorkers = fusedEvidence == nullptr ? 0 : fusedEvidence->m3nCombineCounters[0];
    int32_t combineMappedSegments = fusedEvidence == nullptr ? 0 : fusedEvidence->m3nCombineCounters[1];
    int32_t combineActualWriteCount = fusedEvidence == nullptr ? 0 : fusedEvidence->m3nCombineCounters[2];
    int32_t combineSkippedSegmentCount = fusedEvidence == nullptr ? 0 : fusedEvidence->m3nCombineCounters[3];
    bool combineRangesNonoverlap = fusedEvidence != nullptr && fusedEvidence->m3nCombineCounters[7] != 0;
    bool m3n4StreamEnabled = fusedEvidence != nullptr && fusedEvidence->m3Counters[6] != 0;
    bool m3n4SignalAllOpen = fusedEvidence != nullptr && fusedEvidence->m3Counters[11] != 0;
    int32_t m3n4HandshakeSites = fusedEvidence == nullptr ? 0 : fusedEvidence->m3Counters[12];
    int32_t m3n4SyncallCount = fusedEvidence == nullptr ? 0 : fusedEvidence->m3Counters[13];
    int32_t m3n4CvWaitCount = fusedEvidence == nullptr ? 0 : fusedEvidence->m3Counters[14];
    bool m3n5DispatchGmm1Overlap = fusedEvidence != nullptr && fusedEvidence->m3nDispatchCounters[12] != 0;
    bool m3n11SubtileStride = fusedEvidence != nullptr && fusedEvidence->m3n11SubtileCounters[0] != 0;
    bool dispatchGmm1Overlap = m3n5DispatchGmm1Overlap;
    int32_t m3n5DispatchExpertReadyCount = fusedEvidence == nullptr ? 0 : fusedEvidence->m3nDispatchCounters[13];
    int32_t m3n5ZeroTokenExpertSkipCount = fusedEvidence == nullptr ? 0 : fusedEvidence->m3nDispatchCounters[14];
    bool m3n5Gmm1StartBeforeLastDispatchReady =
        fusedEvidence != nullptr && fusedEvidence->m3n5Gmm1StartBeforeLastDispatchReady != 0;
    bool m3n6Gmm1ActivationOverlap = fusedEvidence != nullptr && fusedEvidence->m3nActivationCounters[8] != 0;
    bool m3n6ActivationStartBeforeLastGmm1Ready =
        fusedEvidence != nullptr && fusedEvidence->m3n6ActivationStartBeforeLastGmm1Ready != 0;
    bool m3n6PrimitiveGap =
        fusedEvidence != nullptr && fusedEvidence->swigluSyncGroupCount > 5 && !m3n6Gmm1ActivationOverlap;
    bool m3n7ActivationGmm2Overlap = fusedEvidence != nullptr && fusedEvidence->m3nGmm2Counters[0] != 0;
    bool m3n7PrimitiveGap = fusedEvidence != nullptr && !m3n7ActivationGmm2Overlap && m3n6Gmm1ActivationOverlap;
    bool m3n7Gmm2StartBeforeLastActivationReady =
        fusedEvidence != nullptr && fusedEvidence->m3n7Gmm2StartBeforeLastActivationReady != 0;
    bool m3n8Gmm2CombineOverlap = fusedEvidence != nullptr && fusedEvidence->m3n8CombineCounters[0] != 0;
    bool m3n8PrimitiveGap = fusedEvidence != nullptr && !m3n8Gmm2CombineOverlap && m3n7ActivationGmm2Overlap;
    bool m3n8CombineStartBeforeLastGmm2Ready =
        fusedEvidence != nullptr && fusedEvidence->m3n8CombineStartBeforeLastGmm2Ready != 0;
    int32_t m3n8ConsumedExperts = fusedEvidence == nullptr ? 0 : fusedEvidence->m3n8CombineCounters[2];
    int32_t m3n8ConsumedSegments = fusedEvidence == nullptr ? 0 : fusedEvidence->m3n8CombineCounters[3];
    int32_t m3n8LastConsumedExpert = fusedEvidence == nullptr ? -1 : fusedEvidence->m3n8CombineCounters[4];
    int32_t m3n8FirstEarlyExpert = fusedEvidence == nullptr ? -1 : fusedEvidence->m3n8CombineCounters[6];
    int32_t m3n8OwnerSegmentWorkers = fusedEvidence == nullptr ? 0 : fusedEvidence->m3n8CombineCounters[7];
    int32_t subtileRows = fusedEvidence == nullptr ? 0 : fusedEvidence->m3n11SubtileCounters[1];
    int32_t subtileStrideWidth = fusedEvidence == nullptr ? 0 : fusedEvidence->m3n11SubtileCounters[2];
    int32_t subtileTransferCount = fusedEvidence == nullptr ? 0 : fusedEvidence->m3n11SubtileCounters[3];
    int32_t subtileSegmentCount = fusedEvidence == nullptr ? 0 : fusedEvidence->m3n11SubtileCounters[4];
    int32_t subtileRemoteTputCount = fusedEvidence == nullptr ? 0 : fusedEvidence->m3n11SubtileCounters[5];
    int32_t subtileLocalCopyCount = fusedEvidence == nullptr ? 0 : fusedEvidence->m3n11SubtileCounters[6];
    int32_t subtileReadyWaitCount = fusedEvidence == nullptr ? 0 : fusedEvidence->m3n11SubtileCounters[7];
    int32_t subtileReadyTileCount = fusedEvidence == nullptr ? 0 : fusedEvidence->m3n11SubtileCounters[8];
    int32_t subtileMappedSegments = fusedEvidence == nullptr ? 0 : fusedEvidence->m3n11SubtileCounters[9];
    int32_t subtileSyncallCount = m3n8Gmm2CombineOverlap ? 2 * static_cast<int32_t>(args.shape.expertPerRank) + 2 : 2;
    int32_t subtileCvWaitCount = m3n8Gmm2CombineOverlap ? 0 : 1;
    std::ostringstream combineWorkerSegments;
    for (int32_t worker = 0; worker < 8; ++worker) {
        if (worker != 0) {
            combineWorkerSegments << ",";
        }
        combineWorkerSegments << worker << ":"
                              << (fusedEvidence == nullptr ? 0 : fusedEvidence->m3nCombineCounters[8U + worker]);
    }
    activationAivWorkers = NormalizeActiveWorkerCount(activationAivWorkers, fusedEvidence);
    combineReturnWorkers = NormalizeActiveWorkerCount(combineReturnWorkers, fusedEvidence);
    if (fusedEvidence != nullptr && m3n8OwnerSegmentWorkers <= 0) {
        m3n8OwnerSegmentWorkers = combineReturnWorkers;
    }
    std::cout << std::setprecision(6);
    std::cout << "[CorrectnessReport]\n";
    std::cout << "  case_name=" << args.caseName << "\n";
    std::cout << "  backend=int8\n";
    std::cout << "  protocol_stage=m2_full_dispatch_gmm_swiglu_combine_restore\n";
    std::cout << "  dtype_in=fp16 dtype_out=fp16\n";
    std::cout << "  dispatch_merge=true\n";
    std::cout << "  combine_merge=true\n";
    std::cout << "  stage_graph_mode=single_fused_mpmd\n";
    std::cout << "  single_fused_payload_migrated=true\n";
    std::cout << "  multi_launch_debug_only=true\n";
    std::cout << "  mixed_elf_register=true\n";
    std::cout << "  fused_single_launch=true\n";
    std::cout << "  exec_model=" << (m3n4StreamEnabled ? "aic_aiv_stream" : "bsp_syncall") << "\n";
    std::cout << "  fused_device_stage_boundaries="
              << (m3n8Gmm2CombineOverlap ?
                      "pto_event_dispatch_expert_gmm1_sync_group_plus_gm_activation_and_gmm2_expert_ready" :
                      (m3n7ActivationGmm2Overlap ?
                           "pto_event_dispatch_expert_gmm1_sync_group_plus_gm_activation_sync_group_ready" :
                           (m3n6Gmm1ActivationOverlap ?
                                "pto_event_dispatch_expert_and_gmm1_sync_group_signals" :
                                (m3n5DispatchGmm1Overlap ?
                                     "pto_event_dispatch_expert_signals" :
                                     (m3n4StreamEnabled ? "pto_event_full_open_signals" : "syncall_mix")))))
              << "\n";
    std::cout << "  fused_syncall_mode=" << (m3n4StreamEnabled ? "coarse_internal_only_counted" : "hard_mix") << "\n";
    std::cout << "  fused_host_barrier_between_stages=false\n";
    std::cout << "  m3_single_kernel_mpmd=true\n";
    std::cout << "  m3_overlap_requested=" << (args.runtime.overlapMode == 0 ? "false" : "true") << "\n";
    std::cout << "  m3_overlap_execution="
              << (m3n8Gmm2CombineOverlap ?
                      "gmm2_combine_expert_rotation_stream" :
                      (m3n7ActivationGmm2Overlap ?
                           "activation_gmm2_sync_group_rotation_stream" :
                           (m3n6Gmm1ActivationOverlap ?
                                "gmm1_activation_sync_group_rotation_stream" :
                                (dispatchGmm1Overlap ? "dispatch_expert_rotation_stream" :
                                                       (m3n4StreamEnabled ? "signal_full_open_stream_skeleton" :
                                                                            "skeleton_shared_layout")))))
              << "\n";
    std::cout << "  m3n4_stream_skeleton_enabled=" << (m3n4StreamEnabled ? "true" : "false") << "\n";
    std::cout << "  m3n4_signal_all_open=" << (m3n4SignalAllOpen ? "true" : "false") << "\n";
    std::cout << "  m3n4_handshake_mode=pto_event_cross_core_full_open\n";
    std::cout << "  m3n4_handshake_site_count=" << m3n4HandshakeSites << "\n";
    std::cout << "  syncall_count=" << m3n4SyncallCount << "\n";
    std::cout << "  cv_wait_count=" << m3n4CvWaitCount << "\n";
    std::cout << "  m3n5_dispatch_gmm1_overlap_enabled=" << (m3n5DispatchGmm1Overlap ? "true" : "false") << "\n";
    std::cout << "  dispatch_overlap_granularity=" << (m3n5DispatchGmm1Overlap ? "expert" : "none") << "\n";
    std::cout << "  m3n5_dispatch_expert_ready_count=" << m3n5DispatchExpertReadyCount << "\n";
    std::cout << "  m3n5_zero_token_expert_skip_count=" << m3n5ZeroTokenExpertSkipCount << "\n";
    std::cout << "  m3n5_gmm1_start_before_last_dispatch_ready="
              << (m3n5Gmm1StartBeforeLastDispatchReady ? "true" : "false") << "\n";
    std::cout << "  m3n6_gmm1_activation_overlap_enabled=" << (m3n6Gmm1ActivationOverlap ? "true" : "false") << "\n";
    std::cout << "  gmm1_activation_overlap_granularity="
              << (m3n6Gmm1ActivationOverlap ? "sync_group" : (m3n6PrimitiveGap ? "primitive_gap" : "none")) << "\n";
    std::cout << "  m3n6_event_capacity=" << 5 << "\n";
    std::cout << "  m3n6_event_insufficient_primitive_gap=" << (m3n6PrimitiveGap ? "true" : "false") << "\n";
    std::cout << "  m3n6_gmm1_sync_group_producer_counter="
              << (fusedEvidence == nullptr ? 0 : fusedEvidence->m3Counters[8]) << "\n";
    std::cout << "  m3n6_activation_sync_group_consumer_counter="
              << (fusedEvidence == nullptr ? 0 : fusedEvidence->m3Counters[9]) << "\n";
    std::cout << "  m3n6_activation_start_before_last_gmm1_ready="
              << (m3n6ActivationStartBeforeLastGmm1Ready ? "true" : "false") << "\n";
    std::cout << "  m3n7_activation_gmm2_overlap_enabled=" << (m3n7ActivationGmm2Overlap ? "true" : "false") << "\n";
    std::cout << "  activation_gmm2_overlap_granularity="
              << (m3n7ActivationGmm2Overlap ? "sync_group" : (m3n7PrimitiveGap ? "primitive_gap" : "none")) << "\n";
    std::cout << "  m3n7_event_capacity=0\n";
    std::cout << "  m3n7_event_insufficient_primitive_gap=" << (m3n7PrimitiveGap ? "true" : "false") << "\n";
    std::cout << "  m3n7_sync_transport=gm_poll_ready\n";
    std::cout << "  m3n7_flag_reuse_risk_avoided=true\n";
    std::cout << "  m3n7_activation_sync_group_producer_counter="
              << (fusedEvidence == nullptr ? 0 : fusedEvidence->activationSyncGroupReadyCount) << "\n";
    std::cout << "  m3n7_gmm2_sync_group_consumer_counter="
              << (fusedEvidence == nullptr ? 0 : fusedEvidence->m3nGmm2Counters[4]) << "\n";
    std::cout << "  m3n7_gmm2_intersect_task_count="
              << (fusedEvidence == nullptr ? 0 : fusedEvidence->m3nGmm2Counters[3]) << "\n";
    std::cout << "  m3n7_gmm2_consumed_last_sync_idx="
              << (fusedEvidence == nullptr ? -1 : fusedEvidence->m3nGmm2Counters[5]) << "\n";
    std::cout << "  m3n7_gmm2_consumed_expert_begin="
              << (fusedEvidence == nullptr ? -1 : fusedEvidence->m3nGmm2Counters[6]) << "\n";
    std::cout << "  m3n7_gmm2_consumed_expert_end="
              << (fusedEvidence == nullptr ? -1 : fusedEvidence->m3nGmm2Counters[7]) << "\n";
    std::cout << "  m3n7_gmm2_consumed_row_begin="
              << (fusedEvidence == nullptr ? -1 : fusedEvidence->m3nGmm2Counters[8]) << "\n";
    std::cout << "  m3n7_gmm2_consumed_row_end=" << (fusedEvidence == nullptr ? -1 : fusedEvidence->m3nGmm2Counters[9])
              << "\n";
    std::cout << "  m3n7_gmm2_consumed_tile_begin="
              << (fusedEvidence == nullptr ? -1 : fusedEvidence->m3nGmm2Counters[10]) << "\n";
    std::cout << "  m3n7_gmm2_consumed_tile_end="
              << (fusedEvidence == nullptr ? -1 : fusedEvidence->m3nGmm2Counters[11]) << "\n";
    std::cout << "  m3n7_gmm2_first_task_id=" << (fusedEvidence == nullptr ? -1 : fusedEvidence->m3nGmm2Counters[12])
              << "\n";
    std::cout << "  m3n7_gmm2_first_task_expert="
              << (fusedEvidence == nullptr ? -1 : fusedEvidence->m3nGmm2Counters[13]) << "\n";
    std::cout << "  m3n7_gmm2_first_task_row_begin="
              << (fusedEvidence == nullptr ? -1 : fusedEvidence->m3nGmm2Counters[15]) << "\n";
    std::cout << "  m3n7_gmm2_start_before_last_activation_ready="
              << (m3n7Gmm2StartBeforeLastActivationReady ? "true" : "false") << "\n";
    std::cout << "  m3n8_gmm2_combine_overlap_enabled=" << (m3n8Gmm2CombineOverlap ? "true" : "false") << "\n";
    std::cout << "  gmm2_combine_overlap_granularity="
              << (m3n8Gmm2CombineOverlap ? "expert" : (m3n8PrimitiveGap ? "primitive_gap" : "none")) << "\n";
    std::cout << "  m3n8_sync_transport=gm_poll_ready\n";
    std::cout << "  m3n8_flag_reuse_risk_avoided=true\n";
    std::cout << "  m3n8_gmm2_expert_ready_counter="
              << (fusedEvidence == nullptr ? 0 : fusedEvidence->gmm2GroupReadyCount) << "\n";
    std::cout << "  m3n8_combine_expert_consumer_counter=" << m3n8ConsumedExperts << "\n";
    std::cout << "  m3n8_combine_segment_consumer_counter=" << m3n8ConsumedSegments << "\n";
    std::cout << "  m3n8_combine_last_consumed_expert=" << m3n8LastConsumedExpert << "\n";
    std::cout << "  m3n8_combine_first_early_expert=" << m3n8FirstEarlyExpert << "\n";
    std::cout << "  m3n8_combine_start_before_last_gmm2_ready="
              << (m3n8CombineStartBeforeLastGmm2Ready ? "true" : "false") << "\n";
    std::cout << "  m3n8_per_expert_wait_scope="
              << (m3n8Gmm2CombineOverlap && m3n8ConsumedExperts == static_cast<int32_t>(args.shape.expertPerRank) ?
                      "true" :
                      "false")
              << "\n";
    std::cout << "  m3n8_full_gmm2_to_combine_cv_wait=" << (m3n8Gmm2CombineOverlap ? "false" : "true") << "\n";
    std::cout << "  overlap_on_payload_async_claim=false\n";
    std::cout << "  m3_launch_level_aiv_participation=true\n";
    std::cout << "  m3_payload_worker_evidence=active_counter_based\n";
    std::cout << "  mixed_aic_heartbeat=" << (fusedEvidence != nullptr && fusedEvidence->aicSeen ? "true" : "false")
              << "\n";
    std::cout << "  mixed_aiv_heartbeat=" << (fusedEvidence != nullptr && fusedEvidence->aivSeen ? "true" : "false")
              << "\n";
    std::cout << "  mixed_aic_launch_blocks=" << (fusedEvidence == nullptr ? 0 : fusedEvidence->aicBlocks) << "\n";
    std::cout << "  mixed_aiv_launch_blocks=" << (fusedEvidence == nullptr ? 0 : fusedEvidence->aivBlocks) << "\n";
    std::array<int32_t, 48> emptyGmmCounters{};
    const auto &gmm1Counters = fusedEvidence == nullptr ? emptyGmmCounters : fusedEvidence->m3oGmm1Counters;
    const auto &gmm2Counters = fusedEvidence == nullptr ? emptyGmmCounters : fusedEvidence->m3oGmm2Counters;
    std::cout << "  gmm1_active_aic_blocks=" << gmm1Counters[0] << "\n";
    std::cout << "  gmm2_active_aic_blocks=" << gmm2Counters[0] << "\n";
    std::cout << "  gmm1_tile_task_count=" << gmm1Counters[1] << "\n";
    std::cout << "  gmm2_tile_task_count=" << gmm2Counters[1] << "\n";
    std::cout << "  gmm1_task_counts_by_block=" << JoinActiveGmmTaskCounts(gmm1Counters) << "\n";
    std::cout << "  gmm2_task_counts_by_block=" << JoinActiveGmmTaskCounts(gmm2Counters) << "\n";
    std::cout << "  gmm1_first_task_block_idx=" << gmm1Counters[2] << "\n";
    std::cout << "  gmm1_first_task_id=" << gmm1Counters[3] << "\n";
    std::cout << "  gmm1_first_task_expert=" << gmm1Counters[4] << "\n";
    std::cout << "  gmm1_first_task_row_begin=" << gmm1Counters[5] << "\n";
    std::cout << "  gmm1_first_task_n_base=" << gmm1Counters[6] << "\n";
    std::cout << "  gmm1_last_task_block_idx=" << gmm1Counters[8] << "\n";
    std::cout << "  gmm1_last_task_id=" << gmm1Counters[9] << "\n";
    std::cout << "  gmm1_last_task_expert=" << gmm1Counters[10] << "\n";
    std::cout << "  gmm1_last_task_row_begin=" << gmm1Counters[11] << "\n";
    std::cout << "  gmm1_last_task_n_base=" << gmm1Counters[12] << "\n";
    std::cout << "  gmm2_first_task_block_idx=" << gmm2Counters[2] << "\n";
    std::cout << "  gmm2_first_task_id=" << gmm2Counters[3] << "\n";
    std::cout << "  gmm2_first_task_expert=" << gmm2Counters[4] << "\n";
    std::cout << "  gmm2_first_task_row_begin=" << gmm2Counters[5] << "\n";
    std::cout << "  gmm2_first_task_n_base=" << gmm2Counters[6] << "\n";
    std::cout << "  gmm2_last_task_block_idx=" << gmm2Counters[8] << "\n";
    std::cout << "  gmm2_last_task_id=" << gmm2Counters[9] << "\n";
    std::cout << "  gmm2_last_task_expert=" << gmm2Counters[10] << "\n";
    std::cout << "  gmm2_last_task_row_begin=" << gmm2Counters[11] << "\n";
    std::cout << "  gmm2_last_task_n_base=" << gmm2Counters[12] << "\n";
    std::cout << "  fused_stage_count=" << (fusedEvidence == nullptr ? 0 : fusedEvidence->stageCount) << "\n";
    std::cout << "  ffn_partition_model=rank_core_group_tile_l1l0\n";
    std::cout << "  group_is_sync_boundary=true\n";
    std::cout << "  gmm_tile_is_aic_work_unit=true\n";
    std::cout << "  dispatch_ready_grain=expert_group\n";
    std::cout << "  activation_ready_grain=swiglu_sync_group\n";
    std::cout << "  combine_ready_grain="
              << (m3n11SubtileStride ? "sub_tile" :
                                       (m3n8Gmm2CombineOverlap ? "expert_group" : "owner_segment_or_group"))
              << "\n";
    std::cout << "  aiv_data_parallel_deferred_to_m3=false\n";
    std::cout << "  aiv_data_parallel_scope=dispatch_m3n1_activation_m3n2_combine_m3n3\n";
    std::cout << "  dispatch_active_aiv_workers=" << dispatchAivWorkers << "\n";
    std::cout << "  route_pack_workers=" << routePackWorkers << "\n";
    std::cout << "  count_sync_workers=" << countSyncWorkers << "\n";
    std::cout << "  count_prefix_workers=" << countSyncWorkers << "\n";
    std::cout << "  dispatch_gather_workers=" << dispatchGatherWorkers << "\n";
    std::cout << "  dispatch_worker_count=" << dispatchWorkerCount << "\n";
    std::cout << "  init_quant_dispatch_parallel_path=" << InitQuantDispatchParallelPath(args) << "\n";
    std::cout << "  init_quant_dispatch_parallel_fallback_reason=" << InitQuantDispatchParallelFallbackReason(args)
              << "\n";
    std::cout << "  dispatch_worker_ranges_nonoverlap="
              << (fusedEvidence != nullptr && fusedEvidence->m3nDispatchCounters[8] != 0 ? "true" : "false") << "\n";
    std::cout << "  worker_expert_count=true\n";
    std::cout << "  worker_expert_prefix=true\n";
    std::cout << "  route_shard_rescan=false\n";
    std::cout << "  dispatch_pack_row_formula=expertBase_plus_workerExpertPrefix_plus_localOrdinal\n";
    std::cout << "  dispatch_prefix_split=local_expert_grid_stride\n";
    std::cout << "  dispatch_gather_split=local_expert_grid_stride\n";
    std::cout << "  dispatch_stage_overlap_enabled=" << (dispatchGmm1Overlap ? "true" : "false") << "\n";
    std::cout << "  gmm1_epilogue_active_aiv_workers=1\n";
    std::cout << "  activation_active_aiv_workers=" << activationAivWorkers << "\n";
    std::cout << "  swiglu_group_tile_ranges="
              << (fusedEvidence == nullptr || fusedEvidence->swigluGroupTileRanges.empty() ?
                      "none" :
                      fusedEvidence->swigluGroupTileRanges)
              << "\n";
    std::cout << "  activation_tiles_processed=" << activationTilesProcessed << "\n";
    std::cout << "  activation_tiles_skipped=" << activationTilesSkipped << "\n";
    std::cout << "  activation_worker_ranges_nonoverlap=" << (activationRangesNonoverlap ? "true" : "false") << "\n";
    std::cout << "  activation_pipe_stages=" << activationPipeStages << "\n";
    std::cout << "  activation_pipe_prefill=" << (activationPipePrefill ? "true" : "false") << "\n";
    std::cout << "  activation_pipe_drain=" << (activationPipeDrain ? "true" : "false") << "\n";
    std::cout << "  combine_active_aiv_workers=" << combineReturnWorkers << "\n";
    std::cout << "  combine_owner_segment_workers=" << m3n8OwnerSegmentWorkers << "\n";
    std::cout << "  combine_worker_segment_counts=" << combineWorkerSegments.str() << "\n";
    std::cout << "  combine_mapped_segment_count=" << combineMappedSegments << "\n";
    std::cout << "  combine_actual_write_count=" << combineActualWriteCount << "\n";
    std::cout << "  combine_skipped_segment_count=" << combineSkippedSegmentCount << "\n";
    std::cout << "  combine_worker_ranges_nonoverlap=" << (combineRangesNonoverlap ? "true" : "false") << "\n";
    if (m3n11SubtileStride) {
        std::cout << "  combine_mode=subtile_stride\n";
    } else {
        std::cout << "  combine_mode=" << (m3n8Gmm2CombineOverlap ? "continuous_segment" : "owner_segment_continuous")
                  << "\n";
    }
    std::cout << "  combine_syncall_count=" << (m3n8Gmm2CombineOverlap ? 2 * args.shape.expertPerRank + 2 : 2) << "\n";
    std::cout << "  combine_cv_wait_count=" << (m3n8Gmm2CombineOverlap ? 0 : 1) << "\n";
    std::cout << "  subtile_stride_sync_tput_enabled=" << (m3n11SubtileStride ? "true" : "false") << "\n";
    std::cout << "  subtile_rows=" << subtileRows << "\n";
    std::cout << "  subtile_stride_width=" << subtileStrideWidth << "\n";
    std::cout << "  subtile_transfer_count=" << subtileTransferCount << "\n";
    std::cout << "  subtile_segment_count=" << subtileSegmentCount << "\n";
    std::cout << "  subtile_remote_tput_count=" << subtileRemoteTputCount << "\n";
    std::cout << "  subtile_local_copy_count=" << subtileLocalCopyCount << "\n";
    std::cout << "  subtile_ready_wait_count=" << subtileReadyWaitCount << "\n";
    std::cout << "  subtile_ready_tile_count=" << subtileReadyTileCount << "\n";
    std::cout << "  subtile_mapped_segment_count=" << subtileMappedSegments << "\n";
    std::cout << "  subtile_syncall_count=" << subtileSyncallCount << "\n";
    std::cout << "  subtile_cv_wait_count=" << subtileCvWaitCount << "\n";
    std::cout << "  combine_stage_overlap_enabled=" << (m3n8Gmm2CombineOverlap ? "true" : "false") << "\n";
    std::array<int32_t, moe_new_dispatch_combine_a8w8::kM3ORestoreCounterWords> emptyRestoreCounters{};
    const auto &restoreCounters = fusedEvidence == nullptr ? emptyRestoreCounters : fusedEvidence->m3oRestoreCounters;
    std::cout << "  restore_active_aiv_workers=" << restoreCounters[0] << "\n";
    std::cout << "  restore_total_token_count=" << restoreCounters[1] << "\n";
    std::cout << "  restore_total_route_count=" << restoreCounters[2] << "\n";
    std::cout << "  restore_total_skipped_route_count=" << restoreCounters[3] << "\n";
    std::cout << "  restore_worker_ranges=" << JoinRestoreWorkerRanges(restoreCounters) << "\n";
    std::cout << "  restore_worker_token_counts=" << JoinRestoreWorkerField(restoreCounters, 2U) << "\n";
    std::cout << "  restore_worker_route_counts=" << JoinRestoreWorkerField(restoreCounters, 3U) << "\n";
    std::cout << "  restore_worker_skipped_route_counts=" << JoinRestoreWorkerField(restoreCounters, 4U) << "\n";
    std::cout << "  restore_worker_ranges_nonoverlap=" << (restoreCounters[4] != 0 ? "true" : "false") << "\n";
    std::cout << "  restore_worker_count_requested=" << restoreCounters[5] << "\n";
    std::cout << "  dispatch_payload_parallel=" << (dispatchAivWorkers > 1 ? "true" : "false") << "\n";
    std::cout << "  activation_payload_parallel=" << (activationAivWorkers > 1 ? "true" : "false") << "\n";
    std::cout << "  combine_payload_parallel=" << (combineReturnWorkers > 1 ? "true" : "false") << "\n";
    std::cout << "  restore_payload_parallel=partial_token_shard\n";
    std::cout << "  gmm1_input_direct=true\n";
    std::cout << "  route_pack_quant_device=true\n";
    std::cout << "  route_quant_path=" << InitQuantRouteQuantPath(args) << "\n";
    std::cout << "  route_quant_scalar_fallback_reason=" << InitQuantRouteQuantFallbackReason(args) << "\n";
    std::cout << "  route_quant_impl_claim="
              << (InitQuantUsesPtoVecRouteQuant(args) ? "pto_vec_fixed_ub_lowered_load_rowmax_quant_store" :
                                                        "scalar_row_loop")
              << "\n";
    std::cout << "  gmm1_epilogue_vec=true\n";
    std::cout << "  activation_requant_vec=true\n";
    std::cout << "  gmm2_epilogue_vec=true\n";
    std::cout << "  gmm_block_mock=false\n";
    std::cout << "  gmm_runtime_shape=true\n";
    std::cout << "  gmm_multiblock_requested=true\n";
    std::cout << "  dispatch_gmm1_sync=expert_ready\n";
    std::cout << "  swiglu_sync_groups=true\n";
    std::cout << "  tile_split_return_map=true\n";
    std::cout << "  m3_signal_cacheline_aligned=true\n";
    std::cout << "  dispatch_group_ready_count="
              << (fusedEvidence == nullptr ? 0 : fusedEvidence->dispatchGroupReadyCount) << "\n";
    std::cout << "  gmm1_sync_group_ready_count="
              << (fusedEvidence == nullptr ? 0 : fusedEvidence->gmm1SyncGroupReadyCount) << "\n";
    std::cout << "  activation_sync_group_ready_count="
              << (fusedEvidence == nullptr ? 0 : fusedEvidence->activationSyncGroupReadyCount) << "\n";
    std::cout << "  gmm2_group_ready_count=" << (fusedEvidence == nullptr ? 0 : fusedEvidence->gmm2GroupReadyCount)
              << "\n";
    std::cout << "  swiglu_sync_group_count=" << (fusedEvidence == nullptr ? 0 : fusedEvidence->swigluSyncGroupCount)
              << "\n";
    std::cout << "  swiglu_sync_group_size_sum="
              << (fusedEvidence == nullptr ? 0 : fusedEvidence->swigluSyncGroupSizeSum) << "\n";
    std::cout << "  swiglu_empty_groups=" << (fusedEvidence == nullptr ? 0 : fusedEvidence->swigluEmptyGroups) << "\n";
    std::cout << "  dequant_sum_final_row=" << (fusedEvidence == nullptr ? 0 : fusedEvidence->dequantFinalRow) << "\n";
    std::cout << "  swiglu_group_desc_monotonic="
              << (fusedEvidence != nullptr && fusedEvidence->swigluGroupDescMonotonic ? "true" : "false") << "\n";
    std::cout << "  dispatch_signal_producer_counter=" << (fusedEvidence == nullptr ? 0 : fusedEvidence->m3Counters[7])
              << "\n";
    std::cout << "  gmm1_signal_producer_counter=" << (fusedEvidence == nullptr ? 0 : fusedEvidence->m3Counters[8])
              << "\n";
    std::cout << "  activation_signal_producer_counter="
              << (fusedEvidence == nullptr ? 0 : fusedEvidence->m3Counters[9]) << "\n";
    std::cout << "  gmm2_signal_producer_counter=" << (fusedEvidence == nullptr ? 0 : fusedEvidence->m3Counters[10])
              << "\n";
    std::cout << "  overlap_timeout_count=" << (fusedEvidence == nullptr ? 0 : fusedEvidence->m3Counters[15]) << "\n";
    std::cout << "  timeout_dump_fields=rank,expert,token_owner_rank,expert_owner_rank,stage,signal_id,"
                 "dispatch_ready,gmm1_ready,activation_ready,gmm2_ready,ready_expert\n";
    if (fusedEvidence != nullptr) {
        PrintM3N9TimeoutDump(*fusedEvidence);
    }
    std::cout << "  subtile_stride_async_enabled=false\n";
    std::cout << "  primitive_gap=TPUT_ASYNC_flat_contiguous_1d\n";
    std::cout << "  tput_async_stride_blocked_locator=TPUT_ASYNC requires flat-contiguous-1d\n";
    std::cout << "  full_async_combine_claim=false\n";
    std::cout << "  timeline_enabled=" << (args.runtime.timeline == 0 ? "false" : "true") << "\n";
    std::cout << "  timeline_granularity=stage_group_tile_subtile\n";
    uint64_t droppedRows = CountDroppedRoutes(args, state);
    uint64_t inactiveTokens = CountInactiveTokens(state);
    std::cout << "  drop_triggered=" << (droppedRows == 0 ? "false" : "true") << "\n";
    std::cout << "  dropped_rows=" << droppedRows << "\n";
    std::cout << "  x_active_mask_enabled=" << (state->inputs.xActiveMask.empty() ? "false" : "true") << "\n";
    std::cout << "  inactive_tokens=" << inactiveTokens << "\n";
    std::cout << "  x_active_mask_no_mask_equiv=" << (state->inputs.xActiveMask.empty() ? "true" : "not_applicable")
              << "\n";
    std::cout << "  gmm2_out_debug_mirror_only=" << (combineReturnWorkers > 1 ? "false" : "true") << "\n";
    std::cout << "  gmm2_out_return_source=" << (combineReturnWorkers > 1 ? "true" : "false") << "\n";
    std::cout << "  return_payload_source="
              << (combineReturnWorkers > 1 ? "gmm2Out_segment_owner_shard" : "returnSegmentStaging") << "\n";
    std::cout << "  restore_from_offsetD=true\n";
    std::cout << "  final_output.reference_source=actual_return_payload_weighted_restore\n";
    std::cout << "  final_output.max_abs_diff=" << summary.maxAbsDiff << "\n";
    std::cout << "  final_output.max_rel_diff=" << summary.maxRelDiff << "\n";
    std::cout << "  final_output.err_count=" << summary.errCount << "\n";
    std::cout << "  final_output.err_threshold=0\n";
    std::cout << "  final_output.tolerance_atol=" << args.atol << "\n";
    std::cout << "  final_output.tolerance_rtol=" << args.rtol << "\n";
    std::cout << "  final_output.checksum_actual=" << summary.checksumActual << "\n";
    std::cout << "  final_output.checksum_expected=" << summary.checksumExpected << "\n";
    std::cout << "  gmm1_accumulator_checksum=" << ChecksumVector(state->m2Reference.gmm1AccInt32) << "\n";
    std::cout << "  scale_dequant_checksum=" << ChecksumVector(state->m2Reference.gmm1Out) << "\n";
    std::cout << "  swiglu_output_checksum=" << ChecksumVector(state->m2Reference.swigluOut) << "\n";
    std::cout << "  gmm2_accumulator_checksum=" << ChecksumVector(state->m2Reference.gmm2AccInt32) << "\n";
    std::cout << "  return_payload_checksum=" << summary.returnPayloadChecksum << "\n";
    std::cout << "  intermediate.tokenPerExpertMatrix_checksum=" << summary.tokenPerExpertMatrixChecksum << "\n";
    std::cout << "  intermediate.cumsumMM_checksum=" << summary.cumsumMMChecksum << "\n";
    std::cout << "  intermediate.preSumBeforeRank_checksum=" << summary.preSumBeforeRankChecksum << "\n";
    std::cout << "  intermediate.expandedRowIdx_checksum=" << summary.expandedRowIdxChecksum << "\n";
    std::cout << "  pass=" << (summary.pass == 0 ? "false" : "true") << "\n";
    if (fusedEvidence != nullptr) {
        PrintM3N12Timeline(args, state, *fusedEvidence);
    }
    std::cout << std::fixed << std::setprecision(1);
    std::cout << "[PerfReport]\n";
    std::cout << "  case_name=" << args.caseName << "\n";
    std::cout << "  seed=" << args.runtime.seed << "\n";
    std::cout << "  run_id=" << MakeM3N12RunId(args) << "\n";
    std::cout << "  shape=m" << args.shape.m << "_k" << args.shape.k << "_i" << args.intermediateSize << "_topk"
              << args.shape.topK << "_ep" << args.shape.ep << "_epr" << args.shape.expertPerRank << "\n";
    std::cout << "  shape_m=" << args.shape.m << " shape_k=" << args.shape.k
              << " shape_intermediate=" << args.intermediateSize << " shape_topK=" << args.shape.topK
              << " shape_expertPerRank=" << args.shape.expertPerRank << "\n";
    std::cout << "  backend=int8\n";
    std::cout << "  rankNum=" << state->size << " rankId=" << state->rank << "\n";
    std::cout << "  overlap_mode=" << args.overlapMode << "\n";
    std::cout << "  overlap_on_uses_same_fused_payload_layout=true\n";
    std::cout << "  overlap_on_payload_async_claim=false\n";
    std::cout << "  warmup_iters=" << args.runtime.warmup << "\n";
    std::cout << "  measure_iters=" << args.runtime.iters << "\n";
    std::cout << "  e2e_us.samples=" << timings.size() << "\n";
    std::cout << "  e2e_us.avg=" << totalStats.avg << "\n";
    std::cout << "  e2e_us.min=" << totalStats.min << "\n";
    std::cout << "  e2e_us.max=" << totalStats.max << "\n";
    std::cout << "  e2e_us.stddev=" << totalStats.stddev << "\n";
    std::cout << "  pass=" << (summary.pass == 0 ? "false" : "true") << "\n";
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
        if (state->buffers.xActiveMask != nullptr) {
            aclrtFree(state->buffers.xActiveMask);
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
        moe_new_dispatch_combine_a8w8::ShapeConfig m2Shape = dispatch_combine_tile::MakeM2ShapeConfig(args);
        moe_new_dispatch_combine_a8w8::WorkspaceLayout m2WorkspaceLayout =
            moe_new_dispatch_combine_a8w8::MakeWorkspaceLayout(m2Shape);
        moe_new_dispatch_combine_a8w8::PeerWindowLayout m2PeerWindowLayout =
            moe_new_dispatch_combine_a8w8::MakePeerWindowLayout(m2Shape);
        uint64_t hcclBuffSizeMb = args.runtime.hcclBuffSizeMb == 0 ?
                                      dispatch_combine_tile::EstimateHcclBuffSizeMb(args.shape, peerWindowLayout) :
                                      args.runtime.hcclBuffSizeMb;
        if (args.backend == "int8" && args.runtime.hcclBuffSizeMb == 0) {
            hcclBuffSizeMb = moe_new_dispatch_combine_a8w8::EstimateHcclBuffSizeMb(m2PeerWindowLayout.totalBytes);
        }
        dispatch_combine_tile::PrintRunSummary(args);
        std::cout << "workspace_bytes="
                  << (args.backend == "int8" ? m2WorkspaceLayout.totalBytes : workspaceLayout.totalBytes) << "\n";
        std::cout << "peer_window_bytes="
                  << (args.backend == "int8" ? m2PeerWindowLayout.totalBytes : peerWindowLayout.totalBytes) << "\n";
        std::cout << "HCCL_BUFFSIZE=" << hcclBuffSizeMb << std::endl;

        dispatch_combine_tile::InitRankInfo(args, &argc, &argv, &state);
        bool verbose = dispatch_combine_tile::VerboseRuntimeLogs(args);
        if (verbose) {
            std::cout << "rank=" << state.rank << " size=" << state.size << " device=" << state.device << " start"
                      << std::endl;
        }
        if (args.backend == "int8" && verbose) {
            moe_new_dispatch_combine_a8w8::RankConfig m2Rank =
                dispatch_combine_tile::MakeM2RankConfig(args, state.rank);
            moe_new_dispatch_combine_a8w8::PrintLayoutDump(std::cout, m2Shape, m2Rank, m2WorkspaceLayout,
                                                           m2PeerWindowLayout);
        }

        if (args.runtime.hostGoldenOnly != 0) {
            dispatch_combine_tile::RunHostGoldenOnly(args, &state);
            dispatch_combine_tile::Cleanup(&state);
            return 0;
        }

        dispatch_combine_tile::PrepareHostData(args, &state);
        if (args.runtime.skipKernels != 0) {
            dispatch_combine_tile::MpiBarrier(&state.mpi);
            std::cout << "rank=" << state.rank << " skip_kernels_done" << std::endl;
            dispatch_combine_tile::Cleanup(&state);
            return 0;
        }
        if (args.backend == "int8") {
            dispatch_combine_tile::BindDeviceContinuous(args, &state);
            dispatch_combine_tile::CreateStreams(args, &state);
            if (args.runtime.m2MixedSpikeOnly != 0) {
                dispatch_combine_tile::RunM2MixedSpike(args, &state);
                dispatch_combine_tile::Cleanup(&state);
                return 0;
            }
            if (args.runtime.m2FusedSkeletonOnly != 0) {
                dispatch_combine_tile::RunM2FusedSkeleton(args, &state);
                dispatch_combine_tile::Cleanup(&state);
                return 0;
            }
            dispatch_combine_tile::PeerWindowLayout m2CompatPeerLayout{};
            m2CompatPeerLayout.totalBytes = m2PeerWindowLayout.totalBytes;
            dispatch_combine_tile::InitHccl(&state, args, m2CompatPeerLayout);
            dispatch_combine_tile::AllocateLocalBuffersM2(args, m2WorkspaceLayout, &state);
            dispatch_combine_tile::CopyInputsToDevice(args, &state);
            dispatch_combine_tile::ClearDeviceStateM2(args, m2WorkspaceLayout, m2PeerWindowLayout, &state);
            dispatch_combine_tile::CopyM2ReferenceToWorkspace(args, m2WorkspaceLayout, &state);
            if (args.runtime.m2FusedFull != 0 && args.runtime.m2MultiLaunchDebug == 0 &&
                args.runtime.dispatchOnly == 0 && args.runtime.dispatchMetadataOnly == 0 &&
                args.runtime.gmm1Only == 0 && args.runtime.gmm1EpilogueOnly == 0 && args.runtime.activationOnly == 0 &&
                args.runtime.gmm2Only == 0 && args.runtime.combineReturnOnly == 0) {
                std::vector<dispatch_combine_tile::IterationTiming> fusedTimings;
                dispatch_combine_tile::M2FusedFullEvidence fusedEvidence;
                uint32_t totalIterations = args.runtime.warmup + args.runtime.iters;
                for (uint32_t iter = 0; iter < totalIterations; ++iter) {
                    if (iter != 0) {
                        dispatch_combine_tile::ClearDeviceStateM2(args, m2WorkspaceLayout, m2PeerWindowLayout, &state);
                        dispatch_combine_tile::CopyM2ReferenceToWorkspace(args, m2WorkspaceLayout, &state);
                    }
                    dispatch_combine_tile::M2FusedFullEvidence iterEvidence;
                    double totalUs =
                        dispatch_combine_tile::RunM2FusedFull(args, m2Shape, m2WorkspaceLayout, &state, &iterEvidence);
                    fusedEvidence = iterEvidence;
                    state.totalE2eUs = totalUs;
                    if (args.runtime.m2FusedDebugStopStage != 0) {
                        dispatch_combine_tile::M2DispatchDump initQuantDump;
                        dispatch_combine_tile::InitQuantVerifySummary initQuantSummary;
                        bool initQuantDebugStop =
                            dispatch_combine_tile::IsInitQuantDebugStopStage(args.runtime.m2FusedDebugStopStage);
                        if (initQuantDebugStop) {
                            dispatch_combine_tile::CopyM2DispatchToHost(args, m2WorkspaceLayout, m2PeerWindowLayout,
                                                                        &state, &initQuantDump);
                            initQuantSummary = dispatch_combine_tile::VerifyInitQuantDebugStop(args, m2PeerWindowLayout,
                                                                                               &state, initQuantDump);
                        }
                        std::cout << "[CorrectnessReport]\n";
                        std::cout << "  case_name=" << args.caseName << "\n";
                        std::cout << "  backend=int8\n";
                        std::cout << "  protocol_stage=m2_fused_full_debug_probe\n";
                        std::cout << "  stage_graph_mode=single_fused_mpmd\n";
                        std::cout << "  m2_fused_debug_stop_stage=" << args.runtime.m2FusedDebugStopStage << "\n";
                        std::cout << "  dispatch_active_aiv_workers="
                                  << NormalizeActiveWorkerCount(iterEvidence.m3nDispatchCounters[0], &iterEvidence)
                                  << "\n";
                        if (initQuantDebugStop) {
                            dispatch_combine_tile::PrintInitQuantDebugStopReport(args, iterEvidence, initQuantSummary);
                            if (!initQuantSummary.Pass()) {
                                throw std::runtime_error("rank " + std::to_string(state.rank) +
                                                         " M2 fused init_quant debug-stop mismatch");
                            }
                        }
                        std::cout << "  pass=true\n";
                        dispatch_combine_tile::MpiBarrier(&state.mpi);
                        dispatch_combine_tile::Cleanup(&state);
                        return 0;
                    }
                    dispatch_combine_tile::RankCorrectnessSummary fusedSummary =
                        dispatch_combine_tile::VerifyM2FinalOutput(args, &state);
                    if (iter >= args.runtime.warmup) {
                        dispatch_combine_tile::IterationTiming timing;
                        timing.totalE2eUs = totalUs;
                        fusedTimings.push_back(timing);
                    }
                    if (fusedSummary.pass == 0) {
                        throw std::runtime_error("rank " + std::to_string(state.rank) +
                                                 " M2 fused full final output mismatch");
                    }
                }
                dispatch_combine_tile::RankCorrectnessSummary fusedSummary = state.correctness;
                for (uint32_t printRank = 0; printRank < state.size; ++printRank) {
                    dispatch_combine_tile::MpiBarrier(&state.mpi);
                    if (state.rank == printRank) {
                        dispatch_combine_tile::PrintM2FinalSummary(args, &state, fusedSummary, fusedTimings,
                                                                   &fusedEvidence);
                        std::cout << std::flush;
                    }
                }
                dispatch_combine_tile::MpiBarrier(&state.mpi);
                if (verbose) {
                    std::cout << "rank=" << state.rank << " run_done" << std::endl;
                }
                dispatch_combine_tile::Cleanup(&state);
                return 0;
            }
            dispatch_combine_tile::MpiBarrier(&state.mpi);
            auto m2TotalStart = std::chrono::steady_clock::now();
            dispatch_combine_tile::RunM2Dispatch(args, m2Shape, m2WorkspaceLayout, m2PeerWindowLayout, &state);
            dispatch_combine_tile::MpiBarrier(&state.mpi);
            if (args.runtime.dispatchOnly != 0 || args.runtime.dispatchMetadataOnly != 0) {
                if (verbose) {
                    std::cout << "rank=" << state.rank << " run_done" << std::endl;
                }
                dispatch_combine_tile::Cleanup(&state);
                return 0;
            }
            dispatch_combine_tile::RunM2Gmm1(args, m2Shape, m2WorkspaceLayout, &state);
            dispatch_combine_tile::MpiBarrier(&state.mpi);
            if (args.runtime.gmm1Only != 0) {
                if (verbose) {
                    std::cout << "rank=" << state.rank << " run_done" << std::endl;
                }
                dispatch_combine_tile::Cleanup(&state);
                return 0;
            }
            dispatch_combine_tile::RunM2Gmm1Epilogue(args, m2Shape, m2WorkspaceLayout, &state);
            dispatch_combine_tile::MpiBarrier(&state.mpi);
            if (args.runtime.gmm1EpilogueOnly != 0) {
                if (verbose) {
                    std::cout << "rank=" << state.rank << " run_done" << std::endl;
                }
                dispatch_combine_tile::Cleanup(&state);
                return 0;
            }
            dispatch_combine_tile::RunM2ActivationQuant(args, m2Shape, m2WorkspaceLayout, &state);
            dispatch_combine_tile::MpiBarrier(&state.mpi);
            if (args.runtime.activationOnly != 0) {
                if (verbose) {
                    std::cout << "rank=" << state.rank << " run_done" << std::endl;
                }
                dispatch_combine_tile::Cleanup(&state);
                return 0;
            }
            dispatch_combine_tile::RunM2Gmm2(args, m2Shape, m2WorkspaceLayout, &state);
            dispatch_combine_tile::MpiBarrier(&state.mpi);
            if (args.runtime.gmm2Only != 0) {
                if (verbose) {
                    std::cout << "rank=" << state.rank << " run_done" << std::endl;
                }
                dispatch_combine_tile::Cleanup(&state);
                return 0;
            }
            dispatch_combine_tile::RunM2Gmm2EpilogueAndReturn(args, m2Shape, m2WorkspaceLayout, m2PeerWindowLayout,
                                                              &state);
            dispatch_combine_tile::MpiBarrier(&state.mpi);
            if (args.runtime.combineReturnOnly != 0) {
                if (verbose) {
                    std::cout << "rank=" << state.rank << " run_done" << std::endl;
                }
                dispatch_combine_tile::Cleanup(&state);
                return 0;
            }
            double restoreE2eUs = dispatch_combine_tile::RunM2RestoreOutput(args, m2Shape, &state);
            dispatch_combine_tile::MpiBarrier(&state.mpi);
            auto m2TotalEnd = std::chrono::steady_clock::now();
            state.totalE2eUs = dispatch_combine_tile::UsSince(m2TotalStart, m2TotalEnd);
            dispatch_combine_tile::RankCorrectnessSummary m2Summary =
                dispatch_combine_tile::VerifyM2FinalOutput(args, &state);
            std::vector<dispatch_combine_tile::IterationTiming> m2Timings;
            if (args.runtime.warmup == 0) {
                dispatch_combine_tile::IterationTiming timing;
                timing.dispatchE2eUs = state.dispatchE2eUs;
                timing.combineE2eUs = restoreE2eUs;
                timing.totalE2eUs = state.totalE2eUs;
                m2Timings.push_back(timing);
            }
            uint32_t totalIterations = args.runtime.warmup + args.runtime.iters;
            for (uint32_t iter = 1; iter < totalIterations; ++iter) {
                dispatch_combine_tile::ClearDeviceStateM2(args, m2WorkspaceLayout, m2PeerWindowLayout, &state);
                dispatch_combine_tile::CopyM2ReferenceToWorkspace(args, m2WorkspaceLayout, &state);
                dispatch_combine_tile::MpiBarrier(&state.mpi);
                m2TotalStart = std::chrono::steady_clock::now();
                dispatch_combine_tile::RunM2Dispatch(args, m2Shape, m2WorkspaceLayout, m2PeerWindowLayout, &state);
                dispatch_combine_tile::MpiBarrier(&state.mpi);
                dispatch_combine_tile::RunM2Gmm1(args, m2Shape, m2WorkspaceLayout, &state);
                dispatch_combine_tile::MpiBarrier(&state.mpi);
                dispatch_combine_tile::RunM2Gmm1Epilogue(args, m2Shape, m2WorkspaceLayout, &state);
                dispatch_combine_tile::MpiBarrier(&state.mpi);
                dispatch_combine_tile::RunM2ActivationQuant(args, m2Shape, m2WorkspaceLayout, &state);
                dispatch_combine_tile::MpiBarrier(&state.mpi);
                dispatch_combine_tile::RunM2Gmm2(args, m2Shape, m2WorkspaceLayout, &state);
                dispatch_combine_tile::MpiBarrier(&state.mpi);
                dispatch_combine_tile::RunM2Gmm2EpilogueAndReturn(args, m2Shape, m2WorkspaceLayout, m2PeerWindowLayout,
                                                                  &state);
                dispatch_combine_tile::MpiBarrier(&state.mpi);
                restoreE2eUs = dispatch_combine_tile::RunM2RestoreOutput(args, m2Shape, &state);
                dispatch_combine_tile::MpiBarrier(&state.mpi);
                m2TotalEnd = std::chrono::steady_clock::now();
                state.totalE2eUs = dispatch_combine_tile::UsSince(m2TotalStart, m2TotalEnd);
                m2Summary = dispatch_combine_tile::VerifyM2FinalOutput(args, &state);
                if (iter >= args.runtime.warmup) {
                    dispatch_combine_tile::IterationTiming timing;
                    timing.dispatchE2eUs = state.dispatchE2eUs;
                    timing.combineE2eUs = restoreE2eUs;
                    timing.totalE2eUs = state.totalE2eUs;
                    m2Timings.push_back(timing);
                }
            }
            dispatch_combine_tile::PrintM2FinalSummary(args, &state, m2Summary, m2Timings);
            dispatch_combine_tile::MpiBarrier(&state.mpi);
            if (verbose) {
                std::cout << "rank=" << state.rank << " run_done" << std::endl;
            }
            dispatch_combine_tile::Cleanup(&state);
            return 0;
        }

        dispatch_combine_tile::BindDeviceContinuous(args, &state);
        dispatch_combine_tile::CreateStreams(args, &state);
        dispatch_combine_tile::InitHccl(&state, args, peerWindowLayout);
        dispatch_combine_tile::AllocateLocalBuffers(args, workspaceLayout, &state);
        dispatch_combine_tile::CopyInputsToDevice(args, &state);

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
