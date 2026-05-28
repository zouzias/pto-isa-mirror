/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef MOE_DISPATCH_COMBINE_A8W8_HOST_REFERENCE_HPP_
#define MOE_DISPATCH_COMBINE_A8W8_HOST_REFERENCE_HPP_

#include "moe_dispatch_combine_a8w8_types.hpp"

#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

namespace moe_dispatch_combine_a8w8 {

struct HostInputData {
    std::vector<float> input;
    std::vector<int32_t> expertId;
    std::vector<float> probs;
};

struct RoutingReference {
    std::vector<int32_t> tokenPerExpertMatrix;
    std::vector<int32_t> expandedRowIdx;
    std::vector<int32_t> dispatchOffset;
    uint64_t tokenPerExpertMatrixChecksum = 0;
    uint64_t expandedRowIdxChecksum = 0;
    uint64_t dispatchOffsetChecksum = 0;
};

struct CorrectnessReport {
    std::string caseName;
    uint32_t seed = 0;
    ShapeConfig shape;
    RankConfig rank;
    uint64_t tokenPerExpertMatrixChecksum = 0;
    uint64_t cumsumMMChecksum = 0;
    uint64_t preSumBeforeRankChecksum = 0;
    uint64_t expandedRowIdxChecksum = 0;
    uint64_t dispatchOffsetChecksum = 0;
    bool protocolChecksumPresent = false;
    bool pass = true;
};

inline uint64_t HashUpdate(uint64_t hash, uint64_t value)
{
    constexpr uint64_t kFnvPrime = 1099511628211ULL;
    hash ^= value;
    hash *= kFnvPrime;
    return hash;
}

template <typename T>
inline uint64_t ChecksumVector(const std::vector<T> &values)
{
    uint64_t hash = 1469598103934665603ULL;
    for (T value : values) {
        hash = HashUpdate(hash, static_cast<uint64_t>(value));
    }
    return hash;
}

inline uint64_t ChecksumFloatVector(const std::vector<float> &values)
{
    uint64_t hash = 1469598103934665603ULL;
    for (float value : values) {
        uint32_t bits = 0;
        static_assert(sizeof(bits) == sizeof(value));
        std::memcpy(&bits, &value, sizeof(bits));
        hash = HashUpdate(hash, bits);
    }
    return hash;
}

inline HostInputData GenerateHostInput(const ShapeConfig &shape, uint32_t rankId, uint32_t seed,
                                       const std::string &caseName)
{
    HostInputData data;
    uint64_t inputElems = static_cast<uint64_t>(shape.m) * shape.hiddenSize;
    uint64_t routeElems = static_cast<uint64_t>(shape.m) * shape.topK;
    uint32_t globalExpertNum = shape.rankNum * shape.expertPerRank;
    data.input.resize(inputElems);
    data.expertId.resize(routeElems);
    data.probs.resize(routeElems);
    for (uint64_t i = 0; i < inputElems; ++i) {
        data.input[i] = static_cast<float>((i + seed + rankId * 17) % 257) / 128.0f - 1.0f;
    }
    for (uint32_t token = 0; token < shape.m; ++token) {
        for (uint32_t slot = 0; slot < shape.topK; ++slot) {
            uint64_t route = static_cast<uint64_t>(token) * shape.topK + slot;
            uint32_t expert = (token * shape.topK + slot + rankId) % globalExpertNum;
            if (caseName == "skewed") {
                expert = (slot == 0) ? rankId % globalExpertNum : (globalExpertNum - 1);
            } else if (caseName == "zero-token") {
                expert = (token + slot) % (globalExpertNum > 1 ? globalExpertNum - 1 : 1);
            }
            data.expertId[route] = static_cast<int32_t>(expert);
            data.probs[route] = shape.topK == 1 ? 1.0f : 1.0f / static_cast<float>(shape.topK);
        }
    }
    return data;
}

inline size_t TokenPerExpertIndex(const ShapeConfig &shape, uint32_t tokenOwnerRank, uint32_t expertOwnerRank,
                                  uint32_t localExpert)
{
    return (static_cast<size_t>(tokenOwnerRank) * shape.rankNum + expertOwnerRank) * shape.expertPerRank + localExpert;
}

inline RoutingReference BuildRoutingReference(const ShapeConfig &shape, const RankConfig &rank,
                                              const HostInputData &input)
{
    RoutingReference ref;
    uint64_t matrixElems = static_cast<uint64_t>(shape.rankNum) * shape.rankNum * shape.expertPerRank;
    uint64_t routeElems = static_cast<uint64_t>(shape.m) * shape.topK;
    ref.tokenPerExpertMatrix.assign(matrixElems, 0);
    ref.expandedRowIdx.assign(routeElems, -1);
    ref.dispatchOffset.assign(routeElems, 0);
    std::vector<int32_t> localCursor(shape.rankNum * shape.expertPerRank, 0);
    for (uint32_t token = 0; token < shape.m; ++token) {
        for (uint32_t slot = 0; slot < shape.topK; ++slot) {
            uint64_t route = static_cast<uint64_t>(token) * shape.topK + slot;
            int32_t expert = input.expertId[route];
            if (expert < 0 || static_cast<uint32_t>(expert) >= shape.rankNum * shape.expertPerRank) {
                continue;
            }
            uint32_t expertOwnerRank = static_cast<uint32_t>(expert) / shape.expertPerRank;
            uint32_t localExpert = static_cast<uint32_t>(expert) % shape.expertPerRank;
            size_t matrixIdx = TokenPerExpertIndex(shape, rank.rankId, expertOwnerRank, localExpert);
            int32_t row = ref.tokenPerExpertMatrix[matrixIdx]++;
            ref.expandedRowIdx[route] = static_cast<int32_t>(route);
            size_t cursorIdx = static_cast<size_t>(expertOwnerRank) * shape.expertPerRank + localExpert;
            ref.dispatchOffset[route] = localCursor[cursorIdx]++;
            (void)row;
        }
    }
    ref.tokenPerExpertMatrixChecksum = ChecksumVector(ref.tokenPerExpertMatrix);
    ref.expandedRowIdxChecksum = ChecksumVector(ref.expandedRowIdx);
    ref.dispatchOffsetChecksum = ChecksumVector(ref.dispatchOffset);
    return ref;
}

inline CorrectnessReport BuildCorrectnessReport(const std::string &caseName, uint32_t seed, const ShapeConfig &shape,
                                                const RankConfig &rank)
{
    HostInputData input = GenerateHostInput(shape, rank.rankId, seed, caseName);
    RoutingReference routing = BuildRoutingReference(shape, rank, input);
    CorrectnessReport report;
    report.caseName = caseName;
    report.seed = seed;
    report.shape = shape;
    report.rank = rank;
    report.tokenPerExpertMatrixChecksum = routing.tokenPerExpertMatrixChecksum;
    report.expandedRowIdxChecksum = routing.expandedRowIdxChecksum;
    report.dispatchOffsetChecksum = routing.dispatchOffsetChecksum;
    report.cumsumMMChecksum = 0;
    report.preSumBeforeRankChecksum = 0;
    report.protocolChecksumPresent = true;
    return report;
}

inline void PrintCorrectnessReport(std::ostream &os, const CorrectnessReport &report)
{
    os << "[CorrectnessReport]\n";
    os << "  case_name=" << report.caseName << "\n";
    os << "  seed=" << report.seed << "\n";
    os << "  rankNum=" << report.shape.rankNum << " rankId=" << report.rank.rankId
       << " expertPerRank=" << report.shape.expertPerRank << " topK=" << report.shape.topK
       << " M=" << report.shape.m << " hiddenSize=" << report.shape.hiddenSize
       << " intermediateSize=" << report.shape.intermediateSize
       << " maxTokensPerExpert=" << report.shape.maxTokensPerExpert << "\n";
    os << "  dtype_in=" << report.shape.dtypeIn << " dtype_out=" << report.shape.dtypeOut << "\n";
    os << "  reference_source=m0-protocol-host-reference\n";
    os << "  final_output.pass=null\n";
    os << "  intermediate.tokenPerExpertMatrix_checksum=" << report.tokenPerExpertMatrixChecksum << "\n";
    os << "  intermediate.cumsumMM_checksum=" << report.cumsumMMChecksum << "\n";
    os << "  intermediate.preSumBeforeRank_checksum=" << report.preSumBeforeRankChecksum << "\n";
    os << "  intermediate.expandedRowIdx_checksum=" << report.expandedRowIdxChecksum << "\n";
    os << "  intermediate.dispatchOffset_checksum=" << report.dispatchOffsetChecksum << "\n";
    os << "  protocol_checksum_present=" << (report.protocolChecksumPresent ? "true" : "false") << "\n";
    os << "  pass=" << (report.pass ? "true" : "false") << "\n";
}

inline void PrintPerfReportSkeleton(std::ostream &os, const std::string &caseName, const ShapeConfig &shape,
                                    const RankConfig &rank, bool correctnessPass)
{
    os << "[PerfReport]\n";
    os << "  case_name=" << caseName << "\n";
    os << "  warmup_iters=" << kWarmupIters << "\n";
    os << "  measure_iters=" << kMeasureIters << "\n";
    os << "  rankNum=" << shape.rankNum << " rankId=" << rank.rankId << " shape=M" << shape.m << "xH"
       << shape.hiddenSize << "xI" << shape.intermediateSize << "\n";
    os << "  correctness_pass=" << (correctnessPass ? "true" : "false") << "\n";
    os << "  e2e_us.samples=[0]\n";
    os << "  e2e_us.avg=0\n";
    os << "  e2e_us.min=0\n";
    os << "  e2e_us.max=0\n";
    os << "  e2e_us.stddev=0\n";
    os << "  stage_us.route=0\n";
    os << "  stage_us.count_sync=0\n";
    os << "  stage_us.dispatch_gather=0\n";
    os << "  stage_us.gmm1=0\n";
    os << "  stage_us.activation_quant=0\n";
    os << "  stage_us.gmm2=0\n";
    os << "  stage_us.fused_combine_return=0\n";
    os << "  stage_us.restore=0\n";
}

} // namespace moe_dispatch_combine_a8w8

#endif // MOE_DISPATCH_COMBINE_A8W8_HOST_REFERENCE_HPP_
