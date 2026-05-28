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

#include <algorithm>
#include <cmath>
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
    std::vector<int32_t> cumsumMM;
    std::vector<int32_t> preSumBeforeRank;
    std::vector<int32_t> expertTokenNums;
    std::vector<int32_t> expandedRowIdx;
    std::vector<int32_t> dispatchOffset;
    uint64_t tokenPerExpertMatrixChecksum = 0;
    uint64_t cumsumMMChecksum = 0;
    uint64_t preSumBeforeRankChecksum = 0;
    uint64_t expertTokenNumsChecksum = 0;
    uint64_t expandedRowIdxChecksum = 0;
    uint64_t dispatchOffsetChecksum = 0;
};

struct M1ProtocolReference {
    RoutingReference routing;
    std::vector<float> dispatchedA;
    std::vector<float> mockExpertOutput;
    std::vector<float> returnPayload;
    std::vector<float> finalOutput;
    std::vector<float> expectedOutput;
    uint64_t dispatchedAChecksum = 0;
    uint64_t mockExpertOutputChecksum = 0;
    uint64_t returnPayloadChecksum = 0;
    uint64_t finalOutputChecksum = 0;
    bool capacityOk = true;
    bool topKRestoreOk = true;
};

struct CorrectnessReport {
    std::string caseName;
    uint32_t seed = 0;
    ShapeConfig shape;
    RankConfig rank;
    uint64_t tokenPerExpertMatrixChecksum = 0;
    uint64_t cumsumMMChecksum = 0;
    uint64_t preSumBeforeRankChecksum = 0;
    uint64_t expertTokenNumsChecksum = 0;
    uint64_t expandedRowIdxChecksum = 0;
    uint64_t dispatchOffsetChecksum = 0;
    uint64_t dispatchedAChecksum = 0;
    uint64_t mockExpertOutputChecksum = 0;
    uint64_t returnPayloadChecksum = 0;
    uint64_t finalOutputChecksum = 0;
    bool protocolChecksumPresent = false;
    bool m1ProtocolMock = false;
    bool capacityOk = true;
    bool topKRestoreOk = true;
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

inline size_t RankExpertIndex(const ShapeConfig &shape, uint32_t tokenOwnerRank, uint32_t localExpert)
{
    return static_cast<size_t>(tokenOwnerRank) * shape.expertPerRank + localExpert;
}

inline uint32_t GlobalExpert(const ShapeConfig &shape, uint32_t expertOwnerRank, uint32_t localExpert)
{
    return expertOwnerRank * shape.expertPerRank + localExpert;
}

inline int32_t TokenPerExpertAt(const ShapeConfig &shape, const std::vector<int32_t> &matrix,
                                uint32_t tokenOwnerRank, uint32_t expertOwnerRank, uint32_t localExpert)
{
    return matrix[TokenPerExpertIndex(shape, tokenOwnerRank, expertOwnerRank, localExpert)];
}

inline uint32_t PreSumBeforeGlobalExpert(const ShapeConfig &shape, const std::vector<int32_t> &matrix,
                                         uint32_t tokenOwnerRank, uint32_t globalExpert)
{
    uint32_t sum = 0;
    for (uint32_t expert = 0; expert < globalExpert; ++expert) {
        uint32_t expertOwnerRank = expert / shape.expertPerRank;
        uint32_t localExpert = expert % shape.expertPerRank;
        sum += static_cast<uint32_t>(TokenPerExpertAt(shape, matrix, tokenOwnerRank, expertOwnerRank, localExpert));
    }
    return sum;
}

inline RoutingReference BuildRoutingReferenceForRank(const ShapeConfig &shape, uint32_t rankId,
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
            size_t matrixIdx = TokenPerExpertIndex(shape, rankId, expertOwnerRank, localExpert);
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

inline M1ProtocolReference BuildM1ProtocolReference(const ShapeConfig &shape, const RankConfig &rank, uint32_t seed,
                                                    const std::string &caseName)
{
    M1ProtocolReference ref;
    uint64_t matrixElems = static_cast<uint64_t>(shape.rankNum) * shape.rankNum * shape.expertPerRank;
    uint64_t routeElems = static_cast<uint64_t>(shape.m) * shape.topK;
    uint64_t inputElems = static_cast<uint64_t>(shape.m) * shape.hiddenSize;
    ref.routing.tokenPerExpertMatrix.assign(matrixElems, 0);
    ref.routing.cumsumMM.assign(static_cast<uint64_t>(shape.rankNum) * shape.expertPerRank, 0);
    ref.routing.preSumBeforeRank.assign(static_cast<uint64_t>(shape.rankNum) * shape.expertPerRank, 0);
    ref.routing.expertTokenNums.assign(shape.expertPerRank, 0);
    ref.routing.expandedRowIdx.assign(routeElems, -1);
    ref.routing.dispatchOffset.assign(routeElems, 0);
    ref.expectedOutput.assign(inputElems, 0.0f);
    ref.finalOutput.assign(inputElems, 0.0f);
    ref.returnPayload.assign(routeElems * shape.hiddenSize, 0.0f);

    std::vector<HostInputData> perRankInput;
    std::vector<std::vector<int32_t>> perRankDispatchOffset(shape.rankNum);
    std::vector<std::vector<int32_t>> perRankExpandedRowIdx(shape.rankNum);
    perRankInput.reserve(shape.rankNum);

    for (uint32_t tokenOwnerRank = 0; tokenOwnerRank < shape.rankNum; ++tokenOwnerRank) {
        perRankInput.push_back(GenerateHostInput(shape, tokenOwnerRank, seed, caseName));
        RoutingReference rankRouting = BuildRoutingReferenceForRank(shape, tokenOwnerRank, perRankInput.back());
        perRankDispatchOffset[tokenOwnerRank] = rankRouting.dispatchOffset;
        perRankExpandedRowIdx[tokenOwnerRank] = rankRouting.expandedRowIdx;
        for (size_t i = 0; i < ref.routing.tokenPerExpertMatrix.size(); ++i) {
            ref.routing.tokenPerExpertMatrix[i] += rankRouting.tokenPerExpertMatrix[i];
        }
        if (tokenOwnerRank == rank.rankId) {
            ref.routing.dispatchOffset = rankRouting.dispatchOffset;
            ref.routing.expandedRowIdx = rankRouting.expandedRowIdx;
            ref.expectedOutput = perRankInput.back().input;
        }
    }

    for (uint32_t localExpert = 0; localExpert < shape.expertPerRank; ++localExpert) {
        uint32_t cumulative = 0;
        for (uint32_t tokenOwnerRank = 0; tokenOwnerRank < shape.rankNum; ++tokenOwnerRank) {
            cumulative += static_cast<uint32_t>(
                TokenPerExpertAt(shape, ref.routing.tokenPerExpertMatrix, tokenOwnerRank, rank.rankId, localExpert));
            ref.routing.cumsumMM[RankExpertIndex(shape, tokenOwnerRank, localExpert)] =
                static_cast<int32_t>(cumulative);
            uint32_t globalExpert = GlobalExpert(shape, rank.rankId, localExpert);
            ref.routing.preSumBeforeRank[RankExpertIndex(shape, tokenOwnerRank, localExpert)] =
                static_cast<int32_t>(
                    PreSumBeforeGlobalExpert(shape, ref.routing.tokenPerExpertMatrix, tokenOwnerRank, globalExpert));
        }
        ref.routing.expertTokenNums[localExpert] = static_cast<int32_t>(cumulative);
        if (cumulative > shape.maxTokensPerExpert) {
            ref.capacityOk = false;
        }
    }

    for (uint32_t localExpert = 0; localExpert < shape.expertPerRank; ++localExpert) {
        uint32_t globalExpert = GlobalExpert(shape, rank.rankId, localExpert);
        for (uint32_t tokenOwnerRank = 0; tokenOwnerRank < shape.rankNum; ++tokenOwnerRank) {
            const HostInputData &ownerInput = perRankInput[tokenOwnerRank];
            for (uint32_t route = 0; route < routeElems; ++route) {
                if (static_cast<uint32_t>(ownerInput.expertId[route]) != globalExpert) {
                    continue;
                }
                uint32_t token = route / shape.topK;
                uint64_t srcBase = static_cast<uint64_t>(token) * shape.hiddenSize;
                ref.dispatchedA.insert(ref.dispatchedA.end(), ownerInput.input.begin() + srcBase,
                                       ownerInput.input.begin() + srcBase + shape.hiddenSize);
            }
        }
    }
    ref.mockExpertOutput = ref.dispatchedA;

    const HostInputData &selfInput = perRankInput[rank.rankId];
    for (uint32_t route = 0; route < routeElems; ++route) {
        uint32_t globalExpert = static_cast<uint32_t>(selfInput.expertId[route]);
        uint32_t localExpert = globalExpert % shape.expertPerRank;
        uint32_t dstRow = PreSumBeforeGlobalExpert(shape, ref.routing.tokenPerExpertMatrix, rank.rankId, globalExpert) +
                          static_cast<uint32_t>(perRankDispatchOffset[rank.rankId][route]);
        uint32_t token = route / shape.topK;
        uint64_t srcBase = static_cast<uint64_t>(token) * shape.hiddenSize;
        uint64_t dstBase = static_cast<uint64_t>(dstRow) * shape.hiddenSize;
        (void)localExpert;
        for (uint32_t col = 0; col < shape.hiddenSize; ++col) {
            float value = selfInput.input[srcBase + col];
            ref.returnPayload[dstBase + col] = value;
            ref.finalOutput[srcBase + col] += value * selfInput.probs[route];
        }
    }

    for (size_t i = 0; i < ref.finalOutput.size(); ++i) {
        if (std::fabs(ref.finalOutput[i] - ref.expectedOutput[i]) > 1e-5f) {
            ref.topKRestoreOk = false;
            break;
        }
    }

    ref.routing.tokenPerExpertMatrixChecksum = ChecksumVector(ref.routing.tokenPerExpertMatrix);
    ref.routing.cumsumMMChecksum = ChecksumVector(ref.routing.cumsumMM);
    ref.routing.preSumBeforeRankChecksum = ChecksumVector(ref.routing.preSumBeforeRank);
    ref.routing.expertTokenNumsChecksum = ChecksumVector(ref.routing.expertTokenNums);
    ref.routing.expandedRowIdxChecksum = ChecksumVector(ref.routing.expandedRowIdx);
    ref.routing.dispatchOffsetChecksum = ChecksumVector(ref.routing.dispatchOffset);
    ref.dispatchedAChecksum = ChecksumFloatVector(ref.dispatchedA);
    ref.mockExpertOutputChecksum = ChecksumFloatVector(ref.mockExpertOutput);
    ref.returnPayloadChecksum = ChecksumFloatVector(ref.returnPayload);
    ref.finalOutputChecksum = ChecksumFloatVector(ref.finalOutput);
    return ref;
}

inline CorrectnessReport BuildCorrectnessReport(const std::string &caseName, uint32_t seed, const ShapeConfig &shape,
                                                const RankConfig &rank)
{
    M1ProtocolReference m1 = BuildM1ProtocolReference(shape, rank, seed, caseName);
    CorrectnessReport report;
    report.caseName = caseName;
    report.seed = seed;
    report.shape = shape;
    report.rank = rank;
    report.tokenPerExpertMatrixChecksum = m1.routing.tokenPerExpertMatrixChecksum;
    report.cumsumMMChecksum = m1.routing.cumsumMMChecksum;
    report.preSumBeforeRankChecksum = m1.routing.preSumBeforeRankChecksum;
    report.expertTokenNumsChecksum = m1.routing.expertTokenNumsChecksum;
    report.expandedRowIdxChecksum = m1.routing.expandedRowIdxChecksum;
    report.dispatchOffsetChecksum = m1.routing.dispatchOffsetChecksum;
    report.dispatchedAChecksum = m1.dispatchedAChecksum;
    report.mockExpertOutputChecksum = m1.mockExpertOutputChecksum;
    report.returnPayloadChecksum = m1.returnPayloadChecksum;
    report.finalOutputChecksum = m1.finalOutputChecksum;
    report.protocolChecksumPresent = true;
    report.m1ProtocolMock = true;
    report.capacityOk = m1.capacityOk;
    report.topKRestoreOk = m1.topKRestoreOk;
    report.pass = report.capacityOk && report.topKRestoreOk;
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
    os << "  reference_source=m1-protocol-host-reference\n";
    os << "  protocol_stage=" << (report.m1ProtocolMock ? "m1_mock" : "m0_skeleton") << "\n";
    os << "  final_output.pass=" << (report.topKRestoreOk ? "true" : "false") << "\n";
    os << "  intermediate.tokenPerExpertMatrix_checksum=" << report.tokenPerExpertMatrixChecksum << "\n";
    os << "  intermediate.cumsumMM_checksum=" << report.cumsumMMChecksum << "\n";
    os << "  intermediate.preSumBeforeRank_checksum=" << report.preSumBeforeRankChecksum << "\n";
    os << "  intermediate.expertTokenNums_checksum=" << report.expertTokenNumsChecksum << "\n";
    os << "  intermediate.expandedRowIdx_checksum=" << report.expandedRowIdxChecksum << "\n";
    os << "  intermediate.dispatchOffset_checksum=" << report.dispatchOffsetChecksum << "\n";
    os << "  intermediate.dispatchedA_checksum=" << report.dispatchedAChecksum << "\n";
    os << "  intermediate.mockExpertOutput_checksum=" << report.mockExpertOutputChecksum << "\n";
    os << "  intermediate.returnPayload_checksum=" << report.returnPayloadChecksum << "\n";
    os << "  final_output.checksum=" << report.finalOutputChecksum << "\n";
    os << "  protocol_checksum_present=" << (report.protocolChecksumPresent ? "true" : "false") << "\n";
    os << "  capacity_ok=" << (report.capacityOk ? "true" : "false") << "\n";
    os << "  pass=" << (report.pass ? "true" : "false") << "\n";
}

inline void PrintI32Vector(std::ostream &os, const char *name, const std::vector<int32_t> &values, size_t limit = 64)
{
    os << "  " << name << "=[";
    size_t count = std::min(values.size(), limit);
    for (size_t i = 0; i < count; ++i) {
        if (i != 0) {
            os << ",";
        }
        os << values[i];
    }
    if (values.size() > limit) {
        os << ",...";
    }
    os << "]\n";
}

inline void PrintM1MetadataDump(std::ostream &os, const std::string &caseName, uint32_t seed,
                                const ShapeConfig &shape, const RankConfig &rank)
{
    M1ProtocolReference ref = BuildM1ProtocolReference(shape, rank, seed, caseName);
    os << "[M1Metadata]\n";
    os << "  stage_graph=RouteLocalTokens->RoutePackQuantLocal->PublishCounts->WaitCounts->"
          "BuildCumsumAndPreSumBeforeRank->GatherDispatchToGmm1Input->MockExpertOutput->"
          "RunGmm2EpilogueAndReturn->RestoreOutput\n";
    PrintI32Vector(os, "tokenPerExpertMatrix", ref.routing.tokenPerExpertMatrix);
    PrintI32Vector(os, "cumsumMM", ref.routing.cumsumMM);
    PrintI32Vector(os, "preSumBeforeRank", ref.routing.preSumBeforeRank);
    PrintI32Vector(os, "expertTokenNums", ref.routing.expertTokenNums);
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
