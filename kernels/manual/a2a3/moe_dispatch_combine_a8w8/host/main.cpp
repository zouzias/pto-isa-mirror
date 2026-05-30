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
#include "moe_dispatch_combine_a8w8_m1_layout.hpp"
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

std::string GmmShapeClass(const moe_dispatch_combine_a8w8::ShapeConfig &shape)
{
    if (shape.hiddenSize == 0U || shape.intermediateSize == 0U ||
        shape.hiddenSize % moe_dispatch_combine_a8w8::kGmmBaseK != 0U ||
        shape.intermediateSize % moe_dispatch_combine_a8w8::kGmmBaseK != 0U ||
        shape.gmmBlockM != moe_dispatch_combine_a8w8::kGmmBaseM ||
        shape.gmmBlockN != moe_dispatch_combine_a8w8::kGmmBaseN ||
        shape.gmmBlockK != moe_dispatch_combine_a8w8::kGmmBaseK) {
        return "unsupported";
    }
    if (shape.hiddenSize < moe_dispatch_combine_a8w8::kGmmBaseN ||
        shape.intermediateSize < moe_dispatch_combine_a8w8::kGmmBaseK ||
        shape.maxTokensPerExpert < moe_dispatch_combine_a8w8::kGmmBaseM) {
        return "smoke_debug";
    }
    return "realistic_model_shape";
}

bool GmmPolicySupported(const moe_dispatch_combine_a8w8::ShapeConfig &shape)
{
    return GmmShapeClass(shape) != "unsupported";
}

uint64_t GmmL1ABytes()
{
    return static_cast<uint64_t>(moe_dispatch_combine_a8w8::kGmmL1Stages) * moe_dispatch_combine_a8w8::kGmmBaseM *
           moe_dispatch_combine_a8w8::kGmmBaseK * moe_dispatch_combine_a8w8::kGmmStepK * sizeof(int8_t);
}

uint64_t GmmL1BBytes()
{
    return static_cast<uint64_t>(moe_dispatch_combine_a8w8::kGmmL1Stages) * moe_dispatch_combine_a8w8::kGmmBaseK *
           moe_dispatch_combine_a8w8::kGmmStepK * moe_dispatch_combine_a8w8::kGmmBaseN * sizeof(int8_t);
}

uint64_t GmmL0ABytes()
{
    return static_cast<uint64_t>(moe_dispatch_combine_a8w8::kGmmL0AStages) * moe_dispatch_combine_a8w8::kGmmBaseM *
           moe_dispatch_combine_a8w8::kGmmBaseK * sizeof(int8_t);
}

uint64_t GmmL0BBytes()
{
    return static_cast<uint64_t>(moe_dispatch_combine_a8w8::kGmmL0BStages) * moe_dispatch_combine_a8w8::kGmmBaseK *
           moe_dispatch_combine_a8w8::kGmmBaseN * sizeof(int8_t);
}

uint64_t GmmL0CBytes()
{
    return static_cast<uint64_t>(moe_dispatch_combine_a8w8::kGmmL0CStages) * moe_dispatch_combine_a8w8::kGmmBaseM *
           moe_dispatch_combine_a8w8::kGmmBaseN * sizeof(int32_t);
}

void PrintGmmPolicyReport(const moe_dispatch_combine_a8w8::ShapeConfig &shape, uint32_t expectedTaskCount,
                          uint32_t launchBlocks)
{
    uint32_t activeBlocks = std::min<uint32_t>(launchBlocks, expectedTaskCount);
    std::cout << "  gmm_tile_policy=gemm_ar_cache_level_int8\n";
    std::cout << "  gmm_shape_class=" << GmmShapeClass(shape) << "\n";
    std::cout << "  gmm_base_m=" << moe_dispatch_combine_a8w8::kGmmBaseM << "\n";
    std::cout << "  gmm_base_n=" << moe_dispatch_combine_a8w8::kGmmBaseN << "\n";
    std::cout << "  gmm_base_k=" << moe_dispatch_combine_a8w8::kGmmBaseK << "\n";
    std::cout << "  gmm_step_k=" << moe_dispatch_combine_a8w8::kGmmStepK << "\n";
    std::cout << "  gmm_l1_a_bytes=" << GmmL1ABytes() << "\n";
    std::cout << "  gmm_l1_b_bytes=" << GmmL1BBytes() << "\n";
    std::cout << "  gmm_l0a_bytes=" << GmmL0ABytes() << "\n";
    std::cout << "  gmm_l0b_bytes=" << GmmL0BBytes() << "\n";
    std::cout << "  gmm_l0c_bytes=" << GmmL0CBytes() << "\n";
    std::cout << "  gmm_l1_stages=" << moe_dispatch_combine_a8w8::kGmmL1Stages << "\n";
    std::cout << "  gmm_l0a_stages=" << moe_dispatch_combine_a8w8::kGmmL0AStages << "\n";
    std::cout << "  gmm_l0b_stages=" << moe_dispatch_combine_a8w8::kGmmL0BStages << "\n";
    std::cout << "  gmm_l0c_stages=" << moe_dispatch_combine_a8w8::kGmmL0CStages << "\n";
    std::cout << "  gmm_tile_tasks=" << expectedTaskCount << "\n";
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

moe_dispatch_combine_a8w8::ShapeConfig MakeM2ShapeConfig(const DispatchCombineTileArgs &args)
{
    moe_dispatch_combine_a8w8::ShapeConfig shape;
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
    shape.dtypeIn = static_cast<uint32_t>(moe_dispatch_combine_a8w8::DType::kFp16);
    shape.dtypeOut = static_cast<uint32_t>(moe_dispatch_combine_a8w8::DType::kFp16);
    return shape;
}

moe_dispatch_combine_a8w8::RankConfig MakeM2RankConfig(const DispatchCombineTileArgs &args, uint32_t rank)
{
    moe_dispatch_combine_a8w8::RankConfig config;
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
        PrintM2ReferenceSummary(args, state->rank, m2Reference);
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
        PrintM2ReferenceSummary(args, state->rank, state->m2Reference);
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
                            const moe_dispatch_combine_a8w8::WorkspaceLayout &workspaceLayout, RuntimeState *state)
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
                        const moe_dispatch_combine_a8w8::WorkspaceLayout &workspaceLayout,
                        const moe_dispatch_combine_a8w8::PeerWindowLayout &peerWindowLayout, RuntimeState *state)
{
    bool verbose = VerboseRuntimeLogs(args);
    if (verbose) {
        PrintStage(state->rank, "clear_m2_device_state", "begin");
    }
    CheckAcl(aclrtMemset(state->buffers.workspace, workspaceLayout.totalBytes, 0, workspaceLayout.totalBytes),
             "rank " + std::to_string(state->rank) + " clear m2 workspace");
    CheckAcl(aclrtMemset(state->hccl.peerWindow, peerWindowLayout.totalBytes, 0, peerWindowLayout.totalBytes),
             "rank " + std::to_string(state->rank) + " clear m2 peerWindow");
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
        std::cout << "  mixed_aic_blocks=" << heartbeat[kAicHeader + 5] << "\n";
        std::cout << "  mixed_aiv_blocks=" << heartbeat[kAivHeader + 5] << "\n";
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
        std::cout << "  mixed_aic_blocks=" << ledger[kAicHeader + 5] << "\n";
        std::cout << "  mixed_aiv_blocks=" << ledger[kAivHeader + 5] << "\n";
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
    bool swigluGroupDescMonotonic = false;
    std::array<int32_t, 16> m3Counters{};
    std::array<int32_t, 16> m3nDispatchCounters{};
    std::array<int32_t, 16> m3nActivationCounters{};
    std::array<int32_t, 16> m3nCombineCounters{};
    std::string swigluGroupTileRanges;
    std::array<int32_t, 6> stageMarkers{};
};

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

std::vector<int32_t> CopyWorkspaceI32Field(const moe_dispatch_combine_a8w8::WorkspaceLayout &workspaceLayout,
                                           const moe_dispatch_combine_a8w8::FieldLayout &field, RuntimeState *state,
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

std::vector<int32_t> CopyPeerI32Field(const moe_dispatch_combine_a8w8::PeerWindowLayout &peerWindowLayout,
                                      const moe_dispatch_combine_a8w8::FieldLayout &field, RuntimeState *state,
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

M2FusedFullEvidence ReadM2FusedFullEvidence(const moe_dispatch_combine_a8w8::WorkspaceLayout &workspaceLayout,
                                            const moe_dispatch_combine_a8w8::PeerWindowLayout &peerWindowLayout,
                                            RuntimeState *state)
{
    constexpr size_t kAicHeaderSlot = 8U * 16U;
    constexpr size_t kAivHeaderSlot = 9U * 16U;
    constexpr size_t kStageBaseSlot = 10U * 16U;
    constexpr size_t kM3CounterBase = 24U * 16U;
    constexpr size_t kM3NDispatchCounterBase = kM3CounterBase + 16U;
    constexpr size_t kM3NActivationCounterBase = kM3CounterBase + 32U;
    constexpr size_t kM3NCombineCounterBase = kM3CounterBase + 48U;
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
    return evidence;
}

double RunM2FusedFull(const DispatchCombineTileArgs &args, const moe_dispatch_combine_a8w8::ShapeConfig &shape,
                      const moe_dispatch_combine_a8w8::WorkspaceLayout &workspaceLayout, RuntimeState *state,
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
    moe_dispatch_combine_a8w8::RankConfig rank = MakeM2RankConfig(args, state->rank);
    constexpr uint32_t kAicBlocks = 24;
    constexpr uint32_t kAivRatio = 2;
    moe_dispatch_combine_a8w8::M2FusedFullLaunchArgs launchArgs{};
    launchArgs.params = moe_dispatch_combine_a8w8::M2FusedFullParams{shape, rank, args.runtime.m2FusedDebugStopStage};
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
    moe_dispatch_combine_a8w8::M2FusedFullLaunchArgs *launchArgsDevice = nullptr;
    CheckAcl(aclrtMalloc(reinterpret_cast<void **>(&launchArgsDevice), sizeof(launchArgs), ACL_MEM_MALLOC_HUGE_FIRST),
             "rank " + std::to_string(state->rank) + " aclrtMalloc m2 fused full launch args");
    try {
        CheckAcl(aclrtMemcpy(launchArgsDevice, sizeof(launchArgs), &launchArgs, sizeof(launchArgs),
                             ACL_MEMCPY_HOST_TO_DEVICE),
                 "rank " + std::to_string(state->rank) + " copy m2 fused full launch args");
        std::array<int32_t, moe_dispatch_combine_a8w8::kM2FusedFullConfigWords> fusedConfig{};
        auto storeConfigU32 = [&fusedConfig](uint32_t slot, uint32_t value) {
            fusedConfig.at(slot) = static_cast<int32_t>(value);
        };
        auto storeConfigU64 = [&fusedConfig](uint32_t slot, uint64_t value) {
            fusedConfig.at(slot) = static_cast<int32_t>(value & 0xffffffffULL);
            fusedConfig.at(slot + 1U) = static_cast<int32_t>((value >> 32U) & 0xffffffffULL);
        };
        storeConfigU32(moe_dispatch_combine_a8w8::kM2FusedFullConfigMagicSlot,
                       moe_dispatch_combine_a8w8::kM2FusedFullConfigMagic);
        storeConfigU32(moe_dispatch_combine_a8w8::kM2FusedFullShapeRankNumSlot, shape.rankNum);
        storeConfigU32(moe_dispatch_combine_a8w8::kM2FusedFullShapeExpertPerRankSlot, shape.expertPerRank);
        storeConfigU32(moe_dispatch_combine_a8w8::kM2FusedFullShapeTopKSlot, shape.topK);
        storeConfigU32(moe_dispatch_combine_a8w8::kM2FusedFullShapeMSlot, shape.m);
        storeConfigU32(moe_dispatch_combine_a8w8::kM2FusedFullShapeHiddenSizeSlot, shape.hiddenSize);
        storeConfigU32(moe_dispatch_combine_a8w8::kM2FusedFullShapeIntermediateSizeSlot, shape.intermediateSize);
        storeConfigU32(moe_dispatch_combine_a8w8::kM2FusedFullShapeMaxTokensPerExpertSlot, shape.maxTokensPerExpert);
        storeConfigU32(moe_dispatch_combine_a8w8::kM2FusedFullShapePayloadTileColsSlot, shape.payloadTileCols);
        storeConfigU32(moe_dispatch_combine_a8w8::kM2FusedFullShapeGmmBlockMSlot, shape.gmmBlockM);
        storeConfigU32(moe_dispatch_combine_a8w8::kM2FusedFullShapeGmmBlockNSlot, shape.gmmBlockN);
        storeConfigU32(moe_dispatch_combine_a8w8::kM2FusedFullShapeGmmBlockKSlot, shape.gmmBlockK);
        storeConfigU32(moe_dispatch_combine_a8w8::kM2FusedFullShapeDtypeInSlot, shape.dtypeIn);
        storeConfigU32(moe_dispatch_combine_a8w8::kM2FusedFullShapeDtypeOutSlot, shape.dtypeOut);
        storeConfigU32(moe_dispatch_combine_a8w8::kM2FusedFullRankRankNumSlot, rank.rankNum);
        storeConfigU32(moe_dispatch_combine_a8w8::kM2FusedFullRankRankIdSlot, rank.rankId);
        storeConfigU32(moe_dispatch_combine_a8w8::kM2FusedFullRankFromMpiSlot, rank.rankFromMpi);
        storeConfigU32(moe_dispatch_combine_a8w8::kM2FusedFullRankDeviceBaseSlot, rank.deviceBase);
        storeConfigU32(moe_dispatch_combine_a8w8::kM2FusedFullRankNdevicesSlot, rank.ndevices);
        storeConfigU64(moe_dispatch_combine_a8w8::kM2FusedFullPtrInputASlot, launchArgs.inputA);
        storeConfigU64(moe_dispatch_combine_a8w8::kM2FusedFullPtrExpertIdxSlot, launchArgs.expertIdx);
        storeConfigU64(moe_dispatch_combine_a8w8::kM2FusedFullPtrXActiveMaskSlot, launchArgs.xActiveMask);
        storeConfigU64(moe_dispatch_combine_a8w8::kM2FusedFullPtrProbsSlot, launchArgs.probs);
        storeConfigU64(moe_dispatch_combine_a8w8::kM2FusedFullPtrOutputCSlot, launchArgs.outputC);
        storeConfigU64(moe_dispatch_combine_a8w8::kM2FusedFullPtrPeerWindowSlot, launchArgs.peerWindow);
        storeConfigU64(moe_dispatch_combine_a8w8::kM2FusedFullPtrHcclCtxSlot, launchArgs.hcclCtx);
        storeConfigU64(moe_dispatch_combine_a8w8::kM2FusedFullPtrWorkspaceSlot, launchArgs.workspace);
        storeConfigU32(moe_dispatch_combine_a8w8::kM2FusedFullTimelineEnableSlot, launchArgs.timelineEnable);
        storeConfigU32(moe_dispatch_combine_a8w8::kM2FusedFullOverlapModeSlot, launchArgs.overlapMode);
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
            ReadM2FusedFullEvidence(workspaceLayout, moe_dispatch_combine_a8w8::MakePeerWindowLayout(shape), state);
    }
    if (verbose) {
        PrintStage(state->rank, "m2_fused_full", "done");
    }
    return UsSince(start, end);
}

void CopyM2ReferenceToWorkspace(const DispatchCombineTileArgs &args,
                                const moe_dispatch_combine_a8w8::WorkspaceLayout &workspaceLayout, RuntimeState *state)
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
    moe_dispatch_combine_a8w8::ShapeConfig m2Shape = MakeM2ShapeConfig(args);
    moe_dispatch_combine_a8w8::PeerWindowLayout peerWindowLayout =
        moe_dispatch_combine_a8w8::MakePeerWindowLayout(m2Shape);
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
    moe_dispatch_combine_a8w8::ShapeConfig m2Shape = MakeM2ShapeConfig(args);
    moe_dispatch_combine_a8w8::PeerWindowLayout peerWindowLayout =
        moe_dispatch_combine_a8w8::MakePeerWindowLayout(m2Shape);
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
    std::vector<int32_t> scoreboardTaskMap;
    std::vector<int32_t> producerStatus;
    std::vector<int32_t> scoreboardMinStatus;
    std::vector<int32_t> workerWaitCounters;
    std::vector<int32_t> scoreboardTimeoutCounters;
    std::vector<int8_t> dispatchPayload;
    std::vector<float> dispatchScale;
    std::vector<int8_t> gmm1InputInt8;
    std::vector<float> routingPerTokenScale;
};

constexpr int32_t kM2ScoreboardStatusCopyDone = 2;
constexpr int32_t kM2ScoreboardStatusSkipDone = 3;

std::vector<int32_t> BuildExpectedM2TokenMatrix(const DispatchCombineTileArgs &args, const CpuGoldenData &golden)
{
    const DispatchCombineTileShape &shape = args.shape;
    size_t expertNumPadded = ExpertNumPadded(shape);
    size_t rowStride =
        static_cast<size_t>(moe_dispatch_combine_a8w8::TokenPerExpertMatrixRowStride(MakeM2ShapeConfig(args)));
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

void BuildExpectedM2Prefix(const DispatchCombineTileArgs &args, const std::vector<int32_t> &tokenMatrix,
                           uint32_t expertOwnerRank, std::vector<int32_t> *cumsum, std::vector<int32_t> *preSum,
                           std::vector<int32_t> *expertTokenNums)
{
    const DispatchCombineTileShape &shape = args.shape;
    size_t rowStride =
        static_cast<size_t>(moe_dispatch_combine_a8w8::TokenPerExpertMatrixRowStride(MakeM2ShapeConfig(args)));
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

void BuildExpectedM2Scoreboard(const DispatchCombineTileArgs &args, const std::vector<int32_t> &tokenMatrix,
                               const std::vector<int32_t> &preSum, const std::vector<int32_t> &expertTokenNums,
                               uint32_t expertOwnerRank, std::vector<int32_t> *taskMap,
                               std::vector<int32_t> *producerStatus, std::vector<int32_t> *scoreboardMinStatus,
                               std::vector<int32_t> *workerWaitCounters,
                               std::vector<int32_t> *scoreboardTimeoutCounters)
{
    const DispatchCombineTileShape &shape = args.shape;
    size_t rowStride =
        static_cast<size_t>(moe_dispatch_combine_a8w8::TokenPerExpertMatrixRowStride(MakeM2ShapeConfig(args)));
    uint32_t rankExpertCount = shape.ep * shape.expertPerRank;
    taskMap->assign(static_cast<size_t>(rankExpertCount) * 4U, 0);
    producerStatus->assign(static_cast<size_t>(rankExpertCount) * 16U, 0);
    scoreboardMinStatus->assign(static_cast<size_t>(rankExpertCount) * 16U, 0);
    workerWaitCounters->assign(static_cast<size_t>(rankExpertCount) * 16U, 0);
    scoreboardTimeoutCounters->assign(static_cast<size_t>(rankExpertCount) * 16U, 0);

    int32_t dispatchOffset = 0;
    for (uint32_t localExpert = 0; localExpert < shape.expertPerRank; ++localExpert) {
        int32_t expertOffset = dispatchOffset;
        int32_t activeSegments = 0;
        int32_t skippedSegments = 0;
        int32_t rowsTotal = 0;
        for (uint32_t tokenOwner = 0; tokenOwner < shape.ep; ++tokenOwner) {
            size_t matrixIndex = static_cast<size_t>(tokenOwner) * rowStride +
                                 static_cast<size_t>(expertOwnerRank) * shape.expertPerRank + localExpert;
            size_t prefixIndex = static_cast<size_t>(tokenOwner) * shape.expertPerRank + localExpert;
            uint32_t taskId = tokenOwner * shape.expertPerRank + localExpert;
            int32_t current = preSum[prefixIndex] + 0;
            if (tokenOwner + 1U < shape.ep) {
                size_t nextPrefixIndex = static_cast<size_t>(tokenOwner + 1U) * shape.expertPerRank + localExpert;
                current = preSum[nextPrefixIndex];
            } else {
                current = expertTokenNums[localExpert];
            }
            int32_t rows = current - preSum[prefixIndex];
            int32_t dstStart = expertOffset + preSum[prefixIndex];
            size_t mapBase = static_cast<size_t>(taskId) * 4U;
            (*taskMap)[mapBase + 0U] = static_cast<int32_t>(tokenOwner);
            (*taskMap)[mapBase + 1U] = static_cast<int32_t>(localExpert);
            (*taskMap)[mapBase + 2U] = dstStart;
            (*taskMap)[mapBase + 3U] = rows;
            int32_t finalStatus = rows > 0 ? kM2ScoreboardStatusCopyDone : kM2ScoreboardStatusSkipDone;
            (*producerStatus)[static_cast<size_t>(taskId) * 16U] = finalStatus;
            if (rows > 0) {
                ++activeSegments;
                rowsTotal += rows;
            } else {
                ++skippedSegments;
            }
        }
        size_t domainBase = static_cast<size_t>(localExpert) * 16U;
        int32_t domainStatus = activeSegments == 0 ? kM2ScoreboardStatusSkipDone : kM2ScoreboardStatusCopyDone;
        (*scoreboardMinStatus)[domainBase + 0U] = domainStatus;
        (*scoreboardMinStatus)[domainBase + 1U] = 1;
        (*scoreboardMinStatus)[domainBase + 2U] = static_cast<int32_t>(localExpert);
        (*scoreboardMinStatus)[domainBase + 3U] = static_cast<int32_t>(localExpert);
        (*scoreboardMinStatus)[domainBase + 4U] = static_cast<int32_t>(shape.expertPerRank);
        (*scoreboardMinStatus)[domainBase + 5U] = static_cast<int32_t>(shape.ep);
        (*scoreboardMinStatus)[domainBase + 6U] = rowsTotal;
        (*scoreboardMinStatus)[domainBase + 7U] = activeSegments;
        (*scoreboardMinStatus)[domainBase + 8U] = skippedSegments;
        (*scoreboardMinStatus)[domainBase + 9U] = activeSegments;
        (*scoreboardMinStatus)[domainBase + 10U] = activeSegments + skippedSegments;
        (*scoreboardMinStatus)[domainBase + 11U] = static_cast<int32_t>(localExpert);
        (*scoreboardMinStatus)[domainBase + 12U] =
            static_cast<int32_t>((shape.ep - 1U) * shape.expertPerRank + localExpert);
        (*scoreboardMinStatus)[domainBase + 13U] = skippedSegments;
        (*scoreboardMinStatus)[domainBase + 14U] = 0;
        (*scoreboardMinStatus)[domainBase + 15U] = 1;

        (*workerWaitCounters)[domainBase + 0U] = 1;
        (*workerWaitCounters)[domainBase + 1U] = static_cast<int32_t>(localExpert);
        (*workerWaitCounters)[domainBase + 2U] = static_cast<int32_t>(shape.ep);
        (*workerWaitCounters)[domainBase + 3U] = activeSegments;
        (*workerWaitCounters)[domainBase + 4U] = activeSegments;
        (*workerWaitCounters)[domainBase + 5U] = skippedSegments;
        (*workerWaitCounters)[domainBase + 6U] = rowsTotal;
        (*workerWaitCounters)[domainBase + 7U] = static_cast<int32_t>(localExpert);
        (*workerWaitCounters)[domainBase + 8U] = static_cast<int32_t>(shape.expertPerRank);
        (*workerWaitCounters)[domainBase + 9U] = domainStatus;
        (*workerWaitCounters)[domainBase + 13U] = 1;
        (*workerWaitCounters)[domainBase + 14U] = 1;
        dispatchOffset += expertTokenNums[localExpert];
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
                          const moe_dispatch_combine_a8w8::WorkspaceLayout &workspaceLayout,
                          const moe_dispatch_combine_a8w8::PeerWindowLayout &peerWindowLayout, RuntimeState *state,
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
    dump->scoreboardTaskMap.assign(static_cast<size_t>(rankExpertCount) * 4U, 0);
    dump->producerStatus.assign(static_cast<size_t>(rankExpertCount) * 16U, 0);
    dump->scoreboardMinStatus.assign(static_cast<size_t>(rankExpertCount) * 16U, 0);
    dump->workerWaitCounters.assign(static_cast<size_t>(rankExpertCount) * 16U, 0);
    dump->scoreboardTimeoutCounters.assign(static_cast<size_t>(rankExpertCount) * 16U, 0);
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
    CheckAcl(aclrtMemcpy(dump->scoreboardTaskMap.data(), BytesOfI32Vector(dump->scoreboardTaskMap.size()),
                         workspaceBase + workspaceLayout.scoreboardTaskMap.offset,
                         BytesOfI32Vector(dump->scoreboardTaskMap.size()), ACL_MEMCPY_DEVICE_TO_HOST),
             "rank " + std::to_string(state->rank) + " copy m2 scoreboardTaskMap");
    CheckAcl(aclrtMemcpy(dump->producerStatus.data(), BytesOfI32Vector(dump->producerStatus.size()),
                         workspaceBase + workspaceLayout.producerStatus.offset,
                         BytesOfI32Vector(dump->producerStatus.size()), ACL_MEMCPY_DEVICE_TO_HOST),
             "rank " + std::to_string(state->rank) + " copy m2 producerStatus");
    CheckAcl(aclrtMemcpy(dump->scoreboardMinStatus.data(), BytesOfI32Vector(dump->scoreboardMinStatus.size()),
                         workspaceBase + workspaceLayout.scoreboardMinStatus.offset,
                         BytesOfI32Vector(dump->scoreboardMinStatus.size()), ACL_MEMCPY_DEVICE_TO_HOST),
             "rank " + std::to_string(state->rank) + " copy m2 scoreboardMinStatus");
    CheckAcl(aclrtMemcpy(dump->workerWaitCounters.data(), BytesOfI32Vector(dump->workerWaitCounters.size()),
                         workspaceBase + workspaceLayout.workerWaitCounters.offset,
                         BytesOfI32Vector(dump->workerWaitCounters.size()), ACL_MEMCPY_DEVICE_TO_HOST),
             "rank " + std::to_string(state->rank) + " copy m2 workerWaitCounters");
    CheckAcl(
        aclrtMemcpy(dump->scoreboardTimeoutCounters.data(), BytesOfI32Vector(dump->scoreboardTimeoutCounters.size()),
                    workspaceBase + workspaceLayout.scoreboardTimeoutCounters.offset,
                    BytesOfI32Vector(dump->scoreboardTimeoutCounters.size()), ACL_MEMCPY_DEVICE_TO_HOST),
        "rank " + std::to_string(state->rank) + " copy m2 scoreboardTimeoutCounters");
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
                          const moe_dispatch_combine_a8w8::PeerWindowLayout &peerWindowLayout, RuntimeState *state,
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
    std::vector<int32_t> expectedScoreboardTaskMap;
    std::vector<int32_t> expectedProducerStatus;
    std::vector<int32_t> expectedScoreboardMinStatus;
    std::vector<int32_t> expectedWorkerWaitCounters;
    std::vector<int32_t> expectedScoreboardTimeoutCounters;
    BuildExpectedM2Scoreboard(args, expectedTokenMatrix, expectedPreSum, expectedExpertTokenNums, state->rank,
                              &expectedScoreboardTaskMap, &expectedProducerStatus, &expectedScoreboardMinStatus,
                              &expectedWorkerWaitCounters, &expectedScoreboardTimeoutCounters);
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
    mismatches +=
        CompareI32Buffer("m2.scoreboardTaskMap", dump.scoreboardTaskMap, expectedScoreboardTaskMap, state->rank);
    mismatches += CompareI32Buffer("m2.producerStatus", dump.producerStatus, expectedProducerStatus, state->rank);
    mismatches +=
        CompareI32Buffer("m2.scoreboardMinStatus", dump.scoreboardMinStatus, expectedScoreboardMinStatus, state->rank);
    mismatches +=
        CompareI32Buffer("m2.workerWaitCounters", dump.workerWaitCounters, expectedWorkerWaitCounters, state->rank);
    mismatches += CompareI32Buffer("m2.scoreboardTimeoutCounters", dump.scoreboardTimeoutCounters,
                                   expectedScoreboardTimeoutCounters, state->rank);
    mismatches += CompareI8Buffer("m2.dispatchPayloadInt8", dump.dispatchPayload, expectedDispatchPayload, state->rank);
    mismatches += CompareFloatBuffer(args, "m2.dispatchScale", dump.dispatchScale, expectedDispatchScale, state->rank);
    mismatches += CompareI8Buffer("m2.gmm1InputInt8", dump.gmm1InputInt8, expectedGmm1Input, state->rank);
    mismatches += CompareFloatBuffer(args, "m2.routingPerTokenScale", dump.routingPerTokenScale,
                                     state->m2Reference.routingPerTokenScale, state->rank);
    return mismatches;
}

std::vector<int32_t> CopyM2Gmm1AccToHost(const DispatchCombineTileArgs &args,
                                         const moe_dispatch_combine_a8w8::WorkspaceLayout &workspaceLayout,
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

M2GmmTaskDump CopyM2GmmTaskDump(const moe_dispatch_combine_a8w8::WorkspaceLayout &workspaceLayout, RuntimeState *state,
                                bool gmm2)
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
    moe_dispatch_combine_a8w8::ShapeConfig m2Shape = MakeM2ShapeConfig(args);
    size_t capacity = moe_dispatch_combine_a8w8::GmmTileTaskCapacity(m2Shape);
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
        for (uint32_t rowOffset = 0; rowOffset < rowCount; rowOffset += moe_dispatch_combine_a8w8::kGmmBaseM) {
            uint32_t rows = std::min<uint32_t>(rowCount - rowOffset, moe_dispatch_combine_a8w8::kGmmBaseM);
            for (uint32_t nBase = 0; nBase < nCols; nBase += moe_dispatch_combine_a8w8::kGmmBaseN) {
                uint32_t cols = std::min<uint32_t>(nCols - nBase, moe_dispatch_combine_a8w8::kGmmBaseN);
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
                                       const moe_dispatch_combine_a8w8::WorkspaceLayout &workspaceLayout,
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
            tileBegin + static_cast<uint32_t>(moe_dispatch_combine_a8w8::CeilDiv(
                            static_cast<uint64_t>(rowPrefix - rowBegin), moe_dispatch_combine_a8w8::GmmBaseM()));
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
                                        const moe_dispatch_combine_a8w8::WorkspaceLayout &workspaceLayout,
                                        RuntimeState *state)
{
    const DispatchCombineTileShape &shape = args.shape;
    uint32_t gmm2RowStride = static_cast<uint32_t>(
        moe_dispatch_combine_a8w8::AlignUp(args.intermediateSize, moe_dispatch_combine_a8w8::kCacheLineBytes));
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
        moe_dispatch_combine_a8w8::AlignUp(args.intermediateSize, moe_dispatch_combine_a8w8::kCacheLineBytes));
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
                                         const moe_dispatch_combine_a8w8::WorkspaceLayout &workspaceLayout,
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
        shape.gmmBlockM == 0 ? static_cast<uint32_t>(moe_dispatch_combine_a8w8::ReturnTileRows()) : shape.gmmBlockM;
    uint32_t hiddenChunks = static_cast<uint32_t>(moe_dispatch_combine_a8w8::CeilDiv(
        shape.k, moe_dispatch_combine_a8w8::ReturnHiddenChunkCols(MakeM2ShapeConfig(args))));
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
    moe_dispatch_combine_a8w8::ShapeConfig m2Shape = MakeM2ShapeConfig(args);
    int32_t tileRows = shape.gmmBlockM == 0 ? static_cast<int32_t>(moe_dispatch_combine_a8w8::ReturnTileRows()) :
                                              static_cast<int32_t>(shape.gmmBlockM);
    uint32_t hiddenChunk = static_cast<uint32_t>(moe_dispatch_combine_a8w8::ReturnHiddenChunkCols(m2Shape));
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
                                             const moe_dispatch_combine_a8w8::WorkspaceLayout &workspaceLayout,
                                             RuntimeState *state)
{
    const DispatchCombineTileShape &shape = args.shape;
    size_t expertNumPadded = ExpertNumPadded(shape);
    uint32_t tileRows =
        shape.gmmBlockM == 0 ? static_cast<uint32_t>(moe_dispatch_combine_a8w8::ReturnTileRows()) : shape.gmmBlockM;
    moe_dispatch_combine_a8w8::ShapeConfig m2Shape = MakeM2ShapeConfig(args);
    uint32_t hiddenChunk = static_cast<uint32_t>(moe_dispatch_combine_a8w8::ReturnHiddenChunkCols(m2Shape));
    size_t capacity = moe_dispatch_combine_a8w8::ReturnSegmentCapacity(m2Shape);
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
                               const moe_dispatch_combine_a8w8::WorkspaceLayout &workspaceLayout,
                               const moe_dispatch_combine_a8w8::PeerWindowLayout &peerWindowLayout, RuntimeState *state,
                               M2CombineReturnDump *dump)
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
                                    const moe_dispatch_combine_a8w8::WorkspaceLayout &workspaceLayout,
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

void RunM2Dispatch(const DispatchCombineTileArgs &args, const moe_dispatch_combine_a8w8::ShapeConfig &shape,
                   const moe_dispatch_combine_a8w8::WorkspaceLayout &workspaceLayout,
                   const moe_dispatch_combine_a8w8::PeerWindowLayout &peerWindowLayout, RuntimeState *state)
{
    bool verbose = VerboseRuntimeLogs(args);
    if (verbose) {
        PrintStage(state->rank, "m2_dispatch", "begin");
    }
    moe_dispatch_combine_a8w8::RankConfig rank = MakeM2RankConfig(args, state->rank);
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
    std::cout << "  route_quant_impl=pto_vec_tload_trowmax_tquant_tstore\n";
    std::cout << "  route_quant_scalar_payload_loop=false\n";
    std::cout << "  dispatch_tget_real=true\n";
    std::cout << "  scoreboard_ledger=true\n";
    std::cout << "  scoreboard_publish_after_tget=true\n";
    std::cout << "  producer_status_domain=token_owner_local_expert\n";
    std::cout << "  scoreboard_dependency_domain=local_expert_consumer_domain\n";
    std::cout << "  scoreboard_dependency_domain_aggregation=true\n";
    std::cout << "  scoreboard_global_min_task_id=false\n";
    std::cout << "  scoreboard_worker_wait_plan=true\n";
    std::cout << "  scoreboard_worker_wait_plan_domain=local_expert\n";
    std::cout << "  scoreboard_domain_count=" << shape.expertPerRank << "\n";
    std::cout << "  scoreboard_domain_stride_i32=16\n";
    std::cout << "  scoreboard_zero_row_skip_semantics=true\n";
    std::cout << "  soft_sync_mode=ledger_only\n";
    std::cout << "  scoreboard_copy_done_status=" << kM2ScoreboardStatusCopyDone << "\n";
    std::cout << "  scoreboard_skip_done_status=" << kM2ScoreboardStatusSkipDone << "\n";
    std::cout << "  routing_per_token_scale_checksum=" << ChecksumVector(dump.routingPerTokenScale) << "\n";
    std::cout << "  dispatch_payload_int8_checksum=" << ChecksumVector(dump.dispatchPayload) << "\n";
    std::cout << "  gmm1_input_int8_checksum=" << ChecksumVector(dump.gmm1InputInt8) << "\n";
    std::cout << "  scoreboard_task_map_checksum=" << ChecksumVector(dump.scoreboardTaskMap) << "\n";
    std::cout << "  producer_status_checksum=" << ChecksumVector(dump.producerStatus) << "\n";
    std::cout << "  scoreboard_min_status_checksum=" << ChecksumVector(dump.scoreboardMinStatus) << "\n";
    std::cout << "  worker_wait_counters_checksum=" << ChecksumVector(dump.workerWaitCounters) << "\n";
    std::cout << "  scoreboard_timeout_counters_checksum=" << ChecksumVector(dump.scoreboardTimeoutCounters) << "\n";
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

void RunM2Gmm1(const DispatchCombineTileArgs &args, const moe_dispatch_combine_a8w8::ShapeConfig &shape,
               const moe_dispatch_combine_a8w8::WorkspaceLayout &workspaceLayout, RuntimeState *state)
{
    if (!GmmPolicySupported(shape)) {
        throw std::runtime_error(
            "M2.3 GMM1 PTO cube path requires gmm_ar cache-level policy and hidden/intermediate divisible by 64");
    }
    bool verbose = VerboseRuntimeLogs(args);
    if (verbose) {
        PrintStage(state->rank, "m2_gmm1", "begin");
    }
    moe_dispatch_combine_a8w8::RankConfig rank = MakeM2RankConfig(args, state->rank);
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
    std::cout << "  gmm_multiblock=" << (launchBlocks > 1 ? "true" : "false") << "\n";
    std::cout << "  gmm_shape_m_max=" << shape.maxTokensPerExpert << "\n";
    std::cout << "  gmm_shape_k=" << shape.hiddenSize << "\n";
    std::cout << "  gmm_shape_n=" << (shape.intermediateSize * 2U) << "\n";
    std::cout << "  gmm_l1_tile_shape=" << moe_dispatch_combine_a8w8::kGmmBaseM << "x"
              << (moe_dispatch_combine_a8w8::kGmmBaseK * moe_dispatch_combine_a8w8::kGmmStepK) << ","
              << (moe_dispatch_combine_a8w8::kGmmBaseK * moe_dispatch_combine_a8w8::kGmmStepK) << "x"
              << moe_dispatch_combine_a8w8::kGmmBaseN << "\n";
    std::cout << "  gmm_l0_tile_shape=" << moe_dispatch_combine_a8w8::kGmmBaseM << "x"
              << moe_dispatch_combine_a8w8::kGmmBaseK << "," << moe_dispatch_combine_a8w8::kGmmBaseK << "x"
              << moe_dispatch_combine_a8w8::kGmmBaseN << "\n";
    std::cout << "  gmm_k_tile=" << moe_dispatch_combine_a8w8::kGmmBaseK << "\n";
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

void RunM2Gmm1Epilogue(const DispatchCombineTileArgs &args, const moe_dispatch_combine_a8w8::ShapeConfig &shape,
                       const moe_dispatch_combine_a8w8::WorkspaceLayout &workspaceLayout, RuntimeState *state)
{
    bool verbose = VerboseRuntimeLogs(args);
    if (verbose) {
        PrintStage(state->rank, "m2_gmm1_epilogue", "begin");
    }
    moe_dispatch_combine_a8w8::RankConfig rank = MakeM2RankConfig(args, state->rank);
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

void RunM2ActivationQuant(const DispatchCombineTileArgs &args, const moe_dispatch_combine_a8w8::ShapeConfig &shape,
                          const moe_dispatch_combine_a8w8::WorkspaceLayout &workspaceLayout, RuntimeState *state)
{
    bool verbose = VerboseRuntimeLogs(args);
    if (verbose) {
        PrintStage(state->rank, "m2_activation_quant", "begin");
    }
    moe_dispatch_combine_a8w8::RankConfig rank = MakeM2RankConfig(args, state->rank);
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
                     static_cast<uint32_t>(moe_dispatch_combine_a8w8::AlignUp(
                         args.intermediateSize, moe_dispatch_combine_a8w8::kCacheLineBytes))))
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

void RunM2Gmm2(const DispatchCombineTileArgs &args, const moe_dispatch_combine_a8w8::ShapeConfig &shape,
               const moe_dispatch_combine_a8w8::WorkspaceLayout &workspaceLayout, RuntimeState *state)
{
    if (!GmmPolicySupported(shape)) {
        throw std::runtime_error(
            "M2.6 GMM2 PTO cube path requires gmm_ar cache-level policy and hidden/intermediate divisible by 64");
    }
    bool verbose = VerboseRuntimeLogs(args);
    if (verbose) {
        PrintStage(state->rank, "m2_gmm2", "begin");
    }
    moe_dispatch_combine_a8w8::RankConfig rank = MakeM2RankConfig(args, state->rank);
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
    std::cout << "  gmm_multiblock=" << (launchBlocks > 1 ? "true" : "false") << "\n";
    std::cout << "  gmm_shape_m_max=" << shape.maxTokensPerExpert << "\n";
    std::cout << "  gmm_shape_k=" << shape.intermediateSize << "\n";
    std::cout << "  gmm_shape_n=" << shape.hiddenSize << "\n";
    std::cout << "  gmm_l1_tile_shape=" << moe_dispatch_combine_a8w8::kGmmBaseM << "x"
              << (moe_dispatch_combine_a8w8::kGmmBaseK * moe_dispatch_combine_a8w8::kGmmStepK) << ","
              << (moe_dispatch_combine_a8w8::kGmmBaseK * moe_dispatch_combine_a8w8::kGmmStepK) << "x"
              << moe_dispatch_combine_a8w8::kGmmBaseN << "\n";
    std::cout << "  gmm_l0_tile_shape=" << moe_dispatch_combine_a8w8::kGmmBaseM << "x"
              << moe_dispatch_combine_a8w8::kGmmBaseK << "," << moe_dispatch_combine_a8w8::kGmmBaseK << "x"
              << moe_dispatch_combine_a8w8::kGmmBaseN << "\n";
    std::cout << "  gmm_k_tile=" << moe_dispatch_combine_a8w8::kGmmBaseK << "\n";
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
                                const moe_dispatch_combine_a8w8::ShapeConfig &shape,
                                const moe_dispatch_combine_a8w8::WorkspaceLayout &workspaceLayout,
                                const moe_dispatch_combine_a8w8::PeerWindowLayout &peerWindowLayout,
                                RuntimeState *state)
{
    bool verbose = VerboseRuntimeLogs(args);
    if (verbose) {
        PrintStage(state->rank, "m2_gmm2_epilogue_return", "begin");
    }
    moe_dispatch_combine_a8w8::RankConfig rank = MakeM2RankConfig(args, state->rank);
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

double RunM2RestoreOutput(const DispatchCombineTileArgs &args, const moe_dispatch_combine_a8w8::ShapeConfig &shape,
                          RuntimeState *state)
{
    bool verbose = VerboseRuntimeLogs(args);
    if (verbose) {
        PrintStage(state->rank, "m2_restore_output", "begin");
    }
    moe_dispatch_combine_a8w8::RankConfig rank = MakeM2RankConfig(args, state->rank);
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
    std::cout << "  soft_sync_ledger=true\n";
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
    if (routePackWorkers <= 0) {
        routePackWorkers = 1;
    }
    if (countSyncWorkers <= 0) {
        countSyncWorkers = 1;
    }
    if (dispatchGatherWorkers <= 0) {
        dispatchGatherWorkers = 1;
    }
    if (dispatchWorkerCount <= 0) {
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
    std::ostringstream combineWorkerSegments;
    for (int32_t worker = 0; worker < 8; ++worker) {
        if (worker != 0) {
            combineWorkerSegments << ",";
        }
        combineWorkerSegments << worker << ":"
                              << (fusedEvidence == nullptr ? 0 : fusedEvidence->m3nCombineCounters[8U + worker]);
    }
    if (activationAivWorkers <= 0) {
        activationAivWorkers = 1;
    }
    if (activationPipeStages <= 0) {
        activationPipeStages = 1;
    }
    if (combineReturnWorkers <= 0) {
        combineReturnWorkers = 1;
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
              << (m3n4StreamEnabled ? "pto_event_full_open_signals" : "syncall_mix") << "\n";
    std::cout << "  fused_syncall_mode=" << (m3n4StreamEnabled ? "coarse_internal_only_counted" : "hard_mix") << "\n";
    std::cout << "  fused_host_barrier_between_stages=false\n";
    std::cout << "  m3_single_kernel_mpmd=true\n";
    std::cout << "  m3_overlap_requested=" << (args.runtime.overlapMode == 0 ? "false" : "true") << "\n";
    std::cout << "  m3_overlap_execution="
              << (m3n4StreamEnabled ? "signal_full_open_stream_skeleton" : "skeleton_shared_layout") << "\n";
    std::cout << "  m3n4_stream_skeleton_enabled=" << (m3n4StreamEnabled ? "true" : "false") << "\n";
    std::cout << "  m3n4_signal_all_open=" << (m3n4SignalAllOpen ? "true" : "false") << "\n";
    std::cout << "  m3n4_handshake_mode=pto_event_cross_core_full_open\n";
    std::cout << "  m3n4_handshake_site_count=" << m3n4HandshakeSites << "\n";
    std::cout << "  syncall_count=" << m3n4SyncallCount << "\n";
    std::cout << "  cv_wait_count=" << m3n4CvWaitCount << "\n";
    std::cout << "  overlap_on_payload_async_claim=false\n";
    std::cout << "  m3_launch_level_aiv_participation=true\n";
    std::cout << "  m3_payload_worker_evidence=partial\n";
    std::cout << "  mixed_aic_heartbeat=" << (fusedEvidence != nullptr && fusedEvidence->aicSeen ? "true" : "false")
              << "\n";
    std::cout << "  mixed_aiv_heartbeat=" << (fusedEvidence != nullptr && fusedEvidence->aivSeen ? "true" : "false")
              << "\n";
    std::cout << "  mixed_aic_blocks=" << (fusedEvidence == nullptr ? 0 : fusedEvidence->aicBlocks) << "\n";
    std::cout << "  mixed_aiv_blocks=" << (fusedEvidence == nullptr ? 0 : fusedEvidence->aivBlocks) << "\n";
    std::cout << "  fused_stage_count=" << (fusedEvidence == nullptr ? 0 : fusedEvidence->stageCount) << "\n";
    std::cout << "  ffn_partition_model=rank_core_group_tile_l1l0\n";
    std::cout << "  group_is_sync_boundary=true\n";
    std::cout << "  gmm_tile_is_aic_work_unit=true\n";
    std::cout << "  dispatch_ready_grain=expert_group\n";
    std::cout << "  activation_ready_grain=swiglu_sync_group\n";
    std::cout << "  combine_ready_grain=owner_segment_or_group\n";
    std::cout << "  aiv_data_parallel_deferred_to_m3=false\n";
    std::cout << "  aiv_data_parallel_scope=dispatch_m3n1_activation_m3n2_combine_m3n3\n";
    std::cout << "  dispatch_aiv_workers=" << dispatchAivWorkers << "\n";
    std::cout << "  route_pack_workers=" << routePackWorkers << "\n";
    std::cout << "  count_sync_workers=" << countSyncWorkers << "\n";
    std::cout << "  dispatch_gather_workers=" << dispatchGatherWorkers << "\n";
    std::cout << "  dispatch_worker_count=" << dispatchWorkerCount << "\n";
    std::cout << "  dispatch_worker_ranges_nonoverlap="
              << (fusedEvidence != nullptr && fusedEvidence->m3nDispatchCounters[8] != 0 ? "true" : "false") << "\n";
    std::cout << "  dispatch_gather_split=local_expert_grid_stride\n";
    std::cout << "  dispatch_stage_overlap_enabled=false\n";
    std::cout << "  gmm1_epilogue_aiv_workers=1\n";
    std::cout << "  activation_aiv_workers=" << activationAivWorkers << "\n";
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
    std::cout << "  combine_return_aiv_workers=" << combineReturnWorkers << "\n";
    std::cout << "  combine_owner_segment_workers=" << combineReturnWorkers << "\n";
    std::cout << "  combine_worker_segment_counts=" << combineWorkerSegments.str() << "\n";
    std::cout << "  combine_mapped_segment_count=" << combineMappedSegments << "\n";
    std::cout << "  combine_actual_write_count=" << combineActualWriteCount << "\n";
    std::cout << "  combine_skipped_segment_count=" << combineSkippedSegmentCount << "\n";
    std::cout << "  combine_worker_ranges_nonoverlap=" << (combineRangesNonoverlap ? "true" : "false") << "\n";
    std::cout << "  combine_mode=owner_segment_continuous\n";
    std::cout << "  combine_stage_overlap_enabled=false\n";
    std::cout << "  restore_aiv_workers=8\n";
    std::cout << "  dispatch_payload_parallel=" << (dispatchAivWorkers > 1 ? "true" : "false") << "\n";
    std::cout << "  activation_payload_parallel=" << (activationAivWorkers > 1 ? "true" : "false") << "\n";
    std::cout << "  combine_payload_parallel=" << (combineReturnWorkers > 1 ? "true" : "false") << "\n";
    std::cout << "  restore_payload_parallel=partial_token_shard\n";
    std::cout << "  gmm1_input_direct=true\n";
    std::cout << "  route_pack_quant_device=true\n";
    std::cout << "  route_quant_impl=pto_vec_tload_trowmax_tquant_tstore\n";
    std::cout << "  gmm1_epilogue_vec=true\n";
    std::cout << "  activation_requant_vec=true\n";
    std::cout << "  gmm2_epilogue_vec=true\n";
    std::cout << "  gmm_block_mock=false\n";
    std::cout << "  gmm_runtime_shape=true\n";
    std::cout << "  gmm_multiblock=true\n";
    std::cout << "  soft_sync_ledger=true\n";
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
    std::cout << "  timeout_dump_fields=rank,expert,token_owner_rank,expert_owner_rank,stage,signal_id\n";
    std::cout << "  scoreboard_async_enabled=false\n";
    std::cout << "  subtile_stride_async_enabled=false\n";
    std::cout << "  timeline_enabled=" << (args.runtime.timeline == 0 ? "false" : "true") << "\n";
    std::cout << "  timeline_granularity=stage_signal_counter\n";
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
    std::cout << std::fixed << std::setprecision(1);
    std::cout << "[PerfReport]\n";
    std::cout << "  case_name=" << args.caseName << "\n";
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
        moe_dispatch_combine_a8w8::ShapeConfig m2Shape = dispatch_combine_tile::MakeM2ShapeConfig(args);
        moe_dispatch_combine_a8w8::WorkspaceLayout m2WorkspaceLayout =
            moe_dispatch_combine_a8w8::MakeWorkspaceLayout(m2Shape);
        moe_dispatch_combine_a8w8::PeerWindowLayout m2PeerWindowLayout =
            moe_dispatch_combine_a8w8::MakePeerWindowLayout(m2Shape);
        uint64_t hcclBuffSizeMb = args.runtime.hcclBuffSizeMb == 0 ?
                                      dispatch_combine_tile::EstimateHcclBuffSizeMb(args.shape, peerWindowLayout) :
                                      args.runtime.hcclBuffSizeMb;
        if (args.backend == "int8" && args.runtime.hcclBuffSizeMb == 0) {
            hcclBuffSizeMb = moe_dispatch_combine_a8w8::EstimateHcclBuffSizeMb(m2PeerWindowLayout.totalBytes);
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
        if (args.backend == "int8") {
            moe_dispatch_combine_a8w8::RankConfig m2Rank = dispatch_combine_tile::MakeM2RankConfig(args, state.rank);
            moe_dispatch_combine_a8w8::PrintLayoutDump(std::cout, m2Shape, m2Rank, m2WorkspaceLayout,
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
                        std::cout << "[CorrectnessReport]\n";
                        std::cout << "  case_name=" << args.caseName << "\n";
                        std::cout << "  backend=int8\n";
                        std::cout << "  protocol_stage=m2_fused_full_debug_probe\n";
                        std::cout << "  stage_graph_mode=single_fused_mpmd\n";
                        std::cout << "  m2_fused_debug_stop_stage=" << args.runtime.m2FusedDebugStopStage << "\n";
                        std::cout << "  mixed_aic_heartbeat=" << (iterEvidence.aicSeen ? "true" : "false") << "\n";
                        std::cout << "  mixed_aiv_heartbeat=" << (iterEvidence.aivSeen ? "true" : "false") << "\n";
                        std::cout << "  mixed_aic_blocks=" << iterEvidence.aicBlocks << "\n";
                        std::cout << "  mixed_aiv_blocks=" << iterEvidence.aivBlocks << "\n";
                        for (size_t markerIdx = 0; markerIdx < iterEvidence.stageMarkers.size(); ++markerIdx) {
                            std::cout << "  fused_stage_marker_" << markerIdx << "="
                                      << iterEvidence.stageMarkers[markerIdx] << "\n";
                        }
                        if (args.runtime.m2FusedDebugStopStage == 1U) {
                            std::cout << "  fused_m3n_route_pack_workers=" << iterEvidence.m3nDispatchCounters[0]
                                      << "\n";
                            std::cout << "  fused_m3n_count_sync_workers=" << iterEvidence.m3nDispatchCounters[1]
                                      << "\n";
                            std::cout << "  fused_m3n_gather_workers=" << iterEvidence.m3nDispatchCounters[2] << "\n";
                            std::cout << "  fused_m3n_worker_count=" << iterEvidence.m3nDispatchCounters[3] << "\n";
                            std::cout << "  fused_m3n_total_counted_rows=" << iterEvidence.m3nDispatchCounters[4]
                                      << "\n";
                            std::cout << "  fused_m3n_active_worker_mask=" << iterEvidence.m3nDispatchCounters[10]
                                      << "\n";
                            dispatch_combine_tile::M2DispatchDump dispatchDump;
                            dispatch_combine_tile::CopyM2DispatchToHost(args, m2WorkspaceLayout, m2PeerWindowLayout,
                                                                        &state, &dispatchDump);
                            uint64_t dispatchMismatches =
                                dispatch_combine_tile::VerifyM2Dispatch(args, m2PeerWindowLayout, &state, dispatchDump);
                            std::cout << "  fused_dispatch_mismatches=" << dispatchMismatches << "\n";
                            if (dispatchMismatches != 0) {
                                throw std::runtime_error("rank " + std::to_string(state.rank) +
                                                         " M2 fused dispatch mismatch");
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
                dispatch_combine_tile::PrintM2FinalSummary(args, &state, fusedSummary, fusedTimings, &fusedEvidence);
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
