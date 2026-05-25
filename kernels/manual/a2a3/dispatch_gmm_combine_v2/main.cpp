#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "comm_mpi.hpp"
#include "kernel_launch.hpp"
#include "op_kernel/combine/combine_progress.hpp"
#include "op_kernel/dispatch/dispatch_progress.hpp"
#include "op_kernel/protocol/pipeline_queue.hpp"
#include "op_kernel/protocol/pto_tpush_tsync_pilot.hpp"
#include "op_kernel/protocol/ready_queue.hpp"
#include "op_kernel/protocol/signal_protocol.hpp"
#include "op_kernel/protocol/task_plan.hpp"
#include "op_kernel/routing/route_plan.hpp"
#include "runtime_context.hpp"
#include "tiling_builder.hpp"

namespace {

constexpr int kDefaultPerfWarmupIters = 3;
constexpr int kDefaultPerfMeasureIters = 5;

struct EventHandle {
    aclrtEvent event = nullptr;

    ~EventHandle()
    {
        if (event != nullptr) {
            aclrtDestroyEvent(event);
        }
    }
};

struct PerfStats {
    double avg = 0.0;
    double min = 0.0;
    double max = 0.0;
    double stddev = 0.0;
};

struct PerfLogicalMetrics {
    uint64_t inputTokens = 0;
    uint64_t routedTokens = 0;
    uint64_t remoteRoutedTokens = 0;
    uint64_t dispatchBytes = 0;
    uint64_t outputBytes = 0;
    uint64_t beforeCapacityTokens = 0;
    uint64_t droppedByCapacityTokens = 0;
    uint64_t sentinelDropTokens = 0;
    uint32_t dispatchProducerCount = 1;
    uint32_t swigluProducerCount = 1;
    uint32_t cubeProducerCount = 1;
    uint32_t tpushPilotActivePath = 0;
    uint32_t tpushPilotRequiresFusedKernel = 1;
    uint32_t tpushPilotBoundary = 0;
    RuntimeRoutingMeta runtimeRoutingMeta;
    uint64_t runtimeRoutingChecksum = 0;
};

struct SemanticOracleSummary {
    uint32_t ledgerEntries = 0;
    uint32_t acceptedEntries = 0;
    uint32_t droppedByCapacityEntries = 0;
    uint32_t sentinelEntries = 0;
    uint32_t tokenPerExpertWords = 0;
    uint32_t cumsumMMWords = 0;
    uint32_t cumsumSendWords = 0;
    uint32_t ownerRows = 0;
    uint32_t dispatchLayoutRanges = 0;
    uint32_t computeLayoutTiles = 0;
    uint64_t dispatchLayoutChecksum = 0;
    uint64_t computeLayoutChecksum = 0;
    uint32_t gmm1SkeletonTiles = 0;
    uint32_t gmm1SkeletonTrueShapeReady = 0;
    uint32_t gmm2SkeletonTiles = 0;
    uint32_t gmm2SkeletonTrueShapeReady = 0;
    uint32_t restoreTraceEntries = 0;
    uint32_t restoreTraceRows = 0;
    uint64_t restoreTraceChecksum = 0;
    uint64_t ledgerChecksum = 0;
    uint64_t routeChecksum = 0;
    uint64_t gmm1Checksum = 0;
    uint64_t swigluChecksum = 0;
    uint64_t quantChecksum = 0;
    uint64_t gmm2Checksum = 0;
    uint64_t finalChecksum = 0;
};

uint64_t MixFloatChecksum(uint64_t seed, float value)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return seed * 131U + bits;
}

SemanticOracleSummary BuildSemanticOracleSummary(const RoutingFixture& fixture,
                                                 const RoutingPlanBundle& plan,
                                                 uint32_t localRank,
                                                 uint32_t outputElems)
{
    SemanticOracleSummary summary{};
    summary.ledgerEntries = static_cast<uint32_t>(plan.semanticLedger.size());
    summary.tokenPerExpertWords = static_cast<uint32_t>(plan.view.dispatch.tokenPerExpert.size());
    summary.cumsumMMWords = static_cast<uint32_t>(plan.view.dispatch.cumsumMM.size());
    summary.cumsumSendWords = plan.runtimeRoutingMeta.cumsumSendCount;
    summary.ownerRows = static_cast<uint32_t>(plan.view.combine.rowToExpandedRange.size());
    summary.dispatchLayoutRanges = static_cast<uint32_t>(plan.dispatchRanges.size());
    summary.computeLayoutTiles = static_cast<uint32_t>(plan.computeTiles.size());
    for (const auto& range : plan.dispatchRanges) {
        summary.dispatchLayoutChecksum = summary.dispatchLayoutChecksum * 131U + range.expandedRowBegin;
        summary.dispatchLayoutChecksum = summary.dispatchLayoutChecksum * 131U + range.expandedRowEnd;
        summary.dispatchLayoutChecksum = summary.dispatchLayoutChecksum * 131U + range.kBegin;
        summary.dispatchLayoutChecksum = summary.dispatchLayoutChecksum * 131U + range.kEnd;
        summary.dispatchLayoutChecksum = summary.dispatchLayoutChecksum * 131U + range.scaleOffsetBegin;
    }
    for (const auto& tile : plan.computeTiles) {
        summary.computeLayoutChecksum = summary.computeLayoutChecksum * 131U + tile.rowBegin;
        summary.computeLayoutChecksum = summary.computeLayoutChecksum * 131U + tile.rowEnd;
        summary.computeLayoutChecksum = summary.computeLayoutChecksum * 131U + tile.kBegin;
        summary.computeLayoutChecksum = summary.computeLayoutChecksum * 131U + tile.kEnd;
        summary.computeLayoutChecksum = summary.computeLayoutChecksum * 131U + tile.nBegin;
        summary.computeLayoutChecksum = summary.computeLayoutChecksum * 131U + tile.nEnd;
        summary.computeLayoutChecksum = summary.computeLayoutChecksum * 131U + tile.expandedRowBegin;
        summary.computeLayoutChecksum = summary.computeLayoutChecksum * 131U + tile.scaleOffsetBegin;
        if (tile.kEnd > tile.kBegin && tile.nEnd > tile.nBegin && tile.rowEnd > tile.rowBegin) {
            ++summary.gmm1SkeletonTiles;
        }
        if ((tile.kEnd - tile.kBegin) > 2 || (tile.nEnd - tile.nBegin) > 4) {
            summary.gmm1SkeletonTrueShapeReady = 1;
        }
        if (tile.nEnd > tile.nBegin && outputElems != 0) {
            summary.gmm2SkeletonTiles += (outputElems + 3U) / 4U;
        }
        if ((tile.nEnd - tile.nBegin) > 2 || outputElems > 4) {
            summary.gmm2SkeletonTrueShapeReady = 1;
        }
    }
    summary.restoreTraceEntries = static_cast<uint32_t>(plan.view.combine.expandedProb.size());
    summary.restoreTraceRows = static_cast<uint32_t>(plan.view.combine.rowToExpandedRange.size());
    for (uint32_t row = 0; row < plan.view.combine.rowToExpandedRange.size(); ++row) {
        const auto& range = plan.view.combine.rowToExpandedRange[row];
        summary.restoreTraceChecksum = summary.restoreTraceChecksum * 131U + row;
        summary.restoreTraceChecksum = summary.restoreTraceChecksum * 131U + range.first;
        summary.restoreTraceChecksum = summary.restoreTraceChecksum * 131U + range.second;
    }
    for (uint32_t idx = 0; idx < plan.view.combine.expandedProb.size(); ++idx) {
        summary.restoreTraceChecksum = summary.restoreTraceChecksum * 131U + plan.view.combine.expandedRowIdx[idx];
        summary.restoreTraceChecksum = summary.restoreTraceChecksum * 131U + plan.view.combine.expandedTopkOrdinal[idx];
        summary.restoreTraceChecksum = summary.restoreTraceChecksum * 131U + plan.view.combine.expandedSrcRank[idx];
        summary.restoreTraceChecksum = summary.restoreTraceChecksum * 131U + plan.view.combine.expandedExpertIdx[idx];
        summary.restoreTraceChecksum = summary.restoreTraceChecksum * 131U + plan.view.combine.expandedDstRank[idx];
        summary.restoreTraceChecksum = summary.restoreTraceChecksum * 131U + plan.view.combine.expandedComputeRow[idx];
        summary.restoreTraceChecksum = MixFloatChecksum(summary.restoreTraceChecksum, plan.view.combine.expandedProb[idx]);
    }
    std::vector<float> finalRows(static_cast<size_t>(fixture.rowsPerRank.at(localRank)) * outputElems, 0.0f);
    for (const auto& entry : plan.semanticLedger) {
        summary.acceptedEntries += entry.accepted;
        summary.droppedByCapacityEntries += entry.droppedByCapacity;
        summary.sentinelEntries += entry.sentinel;
        summary.ledgerChecksum = summary.ledgerChecksum * 131U + entry.beforeCapacityOrdinal;
        summary.ledgerChecksum = summary.ledgerChecksum * 131U + entry.executableOrdinal;
        summary.ledgerChecksum = summary.ledgerChecksum * 131U + entry.ownerRank;
        summary.ledgerChecksum = summary.ledgerChecksum * 131U + entry.srcRank;
        summary.ledgerChecksum = summary.ledgerChecksum * 131U + entry.rowIdx;
        summary.ledgerChecksum = summary.ledgerChecksum * 131U + entry.topkOrdinal;
        summary.ledgerChecksum = summary.ledgerChecksum * 131U + entry.expertId;
        summary.ledgerChecksum = summary.ledgerChecksum * 131U + entry.dstRank;
        summary.ledgerChecksum = summary.ledgerChecksum * 131U + entry.localExpertSlot;
        summary.ledgerChecksum = summary.ledgerChecksum * 131U + entry.accepted;
        summary.ledgerChecksum = summary.ledgerChecksum * 131U + entry.droppedByCapacity;
        summary.ledgerChecksum = summary.ledgerChecksum * 131U + entry.sentinel;
        if (entry.accepted == 0) {
            continue;
        }
        const float tokenBase = static_cast<float>((entry.srcRank + 1U) * 17U + entry.rowIdx * 3U + entry.topkOrdinal);
        const float scale1 = 1.0f + 0.01f * static_cast<float>(entry.expertId + entry.topkOrdinal);
        const float gmm1Gate = tokenBase * scale1 * (1.0f + static_cast<float>(entry.localExpertSlot));
        const float gmm1Up = tokenBase * scale1 * (2.0f + static_cast<float>(entry.expertId % 5U));
        const float sigmoid = 1.0f / (1.0f + std::exp(-gmm1Gate));
        const float swiglu = gmm1Gate * sigmoid * gmm1Up;
        const float quantScale = std::max(std::fabs(swiglu) / 127.0f, 1.0e-6f);
        const float quantized = std::round(swiglu / quantScale) * quantScale;
        summary.routeChecksum = MixFloatChecksum(summary.routeChecksum, tokenBase);
        summary.gmm1Checksum = MixFloatChecksum(summary.gmm1Checksum, gmm1Gate + gmm1Up);
        summary.swigluChecksum = MixFloatChecksum(summary.swigluChecksum, swiglu);
        summary.quantChecksum = MixFloatChecksum(summary.quantChecksum, quantized + quantScale);
        for (uint32_t col = 0; col < outputElems; ++col) {
            const float gmm2 = quantized * (1.0f + 0.001f * static_cast<float>(col + entry.expertId));
            summary.gmm2Checksum = MixFloatChecksum(summary.gmm2Checksum, gmm2);
            const size_t outIndex = static_cast<size_t>(entry.rowIdx) * outputElems + col;
            if (outIndex < finalRows.size()) {
                finalRows[outIndex] += entry.prob * gmm2;
            }
        }
    }
    for (float value : finalRows) {
        summary.finalChecksum = MixFloatChecksum(summary.finalChecksum, value);
    }
    return summary;
}

int ParseEnvInt(const char* name, int defaultValue)
{
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return defaultValue;
    }
    return std::stoi(value);
}

int ParsePerfEnvInt(const char* primaryName, const char* fallbackName, int defaultValue)
{
    const char* value = std::getenv(primaryName);
    if (value != nullptr && value[0] != '\0') {
        return std::stoi(value);
    }
    return ParseEnvInt(fallbackName, defaultValue);
}

PerfStats CalcStats(const std::vector<double>& samples)
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
        const double delta = sample - stats.avg;
        variance += delta * delta;
    }
    stats.stddev = std::sqrt(variance / static_cast<double>(samples.size()));
    return stats;
}

std::vector<double> GatherMaxSamplesToRoot(const std::vector<double>& localSamples, int rank, int worldSize)
{
    if (localSamples.empty()) {
        return {};
    }
    const size_t sampleCount = localSamples.size();
    const int bytesPerRank = static_cast<int>(sampleCount * sizeof(double));
    std::vector<double> gathered;
    if (rank == 0) {
        gathered.resize(sampleCount * static_cast<size_t>(worldSize));
    }
    CommMpiGather(localSamples.data(),
                  bytesPerRank,
                  COMM_MPI_CHAR,
                  rank == 0 ? static_cast<void*>(gathered.data()) : nullptr,
                  bytesPerRank,
                  COMM_MPI_CHAR,
                  0);
    if (rank != 0) {
        return {};
    }

    std::vector<double> maxSamples(sampleCount, 0.0);
    for (size_t sampleIdx = 0; sampleIdx < sampleCount; ++sampleIdx) {
        double maxValue = gathered[sampleIdx];
        for (int r = 1; r < worldSize; ++r) {
            maxValue = std::max(maxValue, gathered[static_cast<size_t>(r) * sampleCount + sampleIdx]);
        }
        maxSamples[sampleIdx] = maxValue;
    }
    return maxSamples;
}


std::vector<uint8_t> MakeDispatchRowPayload(const RoutingFixture& fixture,
                                            uint32_t srcRank,
                                            uint32_t rowIdx)
{
    std::vector<uint8_t> row(fixture.hiddenBytes, 0);
    const uint8_t value = static_cast<uint8_t>(srcRank * 16 + rowIdx + 1);
    std::fill(row.begin(), row.end(), value);
    return row;
}

RoutingFixture BuildBalancedRoutingFixture(uint32_t worldSize,
                                                    uint32_t rowsPerRank,
                                                    uint32_t topk,
                                                    uint32_t expertsPerRank,
                                                    uint32_t maxOutputSize,
                                                    uint32_t hiddenBytes,
                                                    uint32_t outputBytes,
                                                    uint32_t gmm1N)
{
    RoutingFixture fixture;
    fixture.worldSize = worldSize;
    fixture.expertsPerRank = expertsPerRank;
    fixture.topk = topk;
    fixture.maxOutputSize = maxOutputSize;
    fixture.hiddenBytes = hiddenBytes;
    fixture.outputBytes = outputBytes;
    fixture.gmm1N = gmm1N;
    fixture.rowsPerRank.assign(worldSize, rowsPerRank);
    fixture.xActiveMaskPerRank.assign(worldSize, std::vector<uint8_t>(rowsPerRank, 0));
    for (uint32_t rank = 0; rank < worldSize; ++rank) {
        std::fill(fixture.xActiveMaskPerRank[rank].begin(), fixture.xActiveMaskPerRank[rank].end(), topk == 0 ? 0 : 1);
    }
    fixture.topkIdsPerRank.assign(worldSize, std::vector<uint32_t>(static_cast<size_t>(rowsPerRank) * topk, 0));
    fixture.topkProbsPerRank.assign(worldSize, std::vector<float>(static_cast<size_t>(rowsPerRank) * topk, 0.0f));

    const uint32_t totalExperts = worldSize * expertsPerRank;
    for (uint32_t rank = 0; rank < worldSize; ++rank) {
        for (uint32_t row = 0; row < rowsPerRank; ++row) {
            for (uint32_t k = 0; k < topk; ++k) {
                const size_t index = static_cast<size_t>(row) * topk + k;
                fixture.topkIdsPerRank[rank][index] = (rank * expertsPerRank + row + k) % totalExperts;
                fixture.topkProbsPerRank[rank][index] =
                    topk == 0 ? 0.0f : static_cast<float>(k + 1U) * 2.0f / static_cast<float>(topk * (topk + 1U));
            }
        }
    }
    return fixture;
}

std::vector<uint8_t> BuildDispatchPublicationForRank(const RoutingFixture& fixture, uint32_t srcRank)
{
    const auto raw = ExpandBeforeCapacity(fixture);
    const auto executable = KeepExecutableEntries(raw, fixture);
    const uint64_t segmentBytes = static_cast<uint64_t>(fixture.maxOutputSize) * fixture.hiddenBytes;
    const uint64_t scaleBaseBytes = static_cast<uint64_t>(fixture.worldSize) * segmentBytes;
    const uint64_t scaleSegmentBytes = static_cast<uint64_t>(fixture.maxOutputSize) * kDispatchScaleSlotBytes;
    std::vector<uint8_t> publication(static_cast<size_t>(scaleBaseBytes + fixture.worldSize * scaleSegmentBytes), 0);
    std::vector<uint32_t> ordinalPerDst(fixture.worldSize, 0);

    for (const auto& entry : executable) {
        if (entry.srcRank != srcRank) {
            continue;
        }
        const uint32_t ordinal = ordinalPerDst[entry.dstRank]++;
        const uint64_t offset = static_cast<uint64_t>(entry.dstRank) * segmentBytes +
                                static_cast<uint64_t>(ordinal) * fixture.hiddenBytes;
        const uint64_t scaleOffset = scaleBaseBytes +
                                     static_cast<uint64_t>(entry.dstRank) * scaleSegmentBytes +
                                     static_cast<uint64_t>(ordinal) * kDispatchScaleSlotBytes;
        const auto row = MakeDispatchRowPayload(fixture, srcRank, entry.rowIdx);
        std::memcpy(publication.data() + offset, row.data(), row.size());
        std::memcpy(publication.data() + scaleOffset, &entry.prob, sizeof(float));
    }
    return publication;
}

std::vector<uint8_t> BuildHostDispatchOracle(const RoutingFixture& fixture,
                                             uint32_t localRank,
                                             const RoutingPlanBundle& plan)
{
    std::vector<std::vector<uint8_t>> publications;
    publications.reserve(fixture.worldSize);
    for (uint32_t srcRank = 0; srcRank < fixture.worldSize; ++srcRank) {
        publications.push_back(BuildDispatchPublicationForRank(fixture, srcRank));
    }

    uint32_t totalRows = 0;
    for (uint32_t value : plan.view.dispatch.gatheredExpertCount) {
        totalRows += value;
    }
    const uint64_t scaleBaseBytes = static_cast<uint64_t>(fixture.maxOutputSize) * fixture.hiddenBytes;
    std::vector<uint8_t> oracle(static_cast<size_t>(scaleBaseBytes + totalRows * kDispatchScaleSlotBytes), 0);
    for (const auto& task : plan.dispatchTasks) {
        const auto& publication = publications.at(task.srcRank);
        std::memcpy(oracle.data() + task.dstPayloadOffsetBytes,
                    publication.data() + task.srcPayloadOffsetBytes,
                    fixture.hiddenBytes);
        std::memcpy(oracle.data() + task.dstScaleOffsetBytes,
                    publication.data() + task.srcScaleOffsetBytes,
                    task.scaleBytes);
    }
    return oracle;
}

struct ComputeWeights {
    std::vector<float> weight1;
    std::vector<int8_t> weight1Q;
    std::vector<uint64_t> scale1Channel;
    std::vector<float> weight2;
    std::vector<int8_t> weight2Q;
    std::vector<uint64_t> scale2Channel;
    uint32_t weight2StrideElems = 0;
};

struct HostComputeOracle {
    ComputeWeights fixture;
};

uint32_t AlignUpUint32(uint32_t value, uint32_t align)
{
    return align == 0 ? value : ((value + align - 1U) / align) * align;
}

HostComputeOracle BuildHostComputeOracle(uint32_t hiddenK, uint32_t gmm1N, uint32_t outputElems, uint32_t localRank = 0)
{
    HostComputeOracle oracle;
    oracle.fixture.weight1.resize(static_cast<size_t>(hiddenK) * gmm1N, 0.0f);
    oracle.fixture.weight1Q.resize(static_cast<size_t>(hiddenK) * gmm1N, 0);
    oracle.fixture.scale1Channel.resize(gmm1N, 0);
    for (uint32_t k = 0; k < hiddenK; ++k) {
        for (uint32_t n = 0; n < gmm1N; ++n) {
            const float fval = ((n % 2U) == (k % 2U)) ? static_cast<float>(2U + localRank) : 0.0f;
            oracle.fixture.weight1[static_cast<size_t>(k) * gmm1N + n] = fval;
            float clamped = std::max(-127.0f, std::min(127.0f, fval));
            oracle.fixture.weight1Q[static_cast<size_t>(k) * gmm1N + n] = static_cast<int8_t>(std::round(clamped));
        }
    }
    for (uint32_t n = 0; n < gmm1N; ++n) {
        float colScale = 1.0f;
        uint64_t scaleBits = 0;
        std::memcpy(&scaleBits, &colScale, sizeof(float));
        oracle.fixture.scale1Channel[n] = scaleBits;
    }
    const uint32_t k2Total = gmm1N / 2;
    const uint32_t weight2StrideElems = AlignUpUint32(outputElems, 32);
    oracle.fixture.weight2StrideElems = weight2StrideElems;
    oracle.fixture.weight2.resize(static_cast<size_t>(k2Total) * weight2StrideElems, 0.0f);
    oracle.fixture.weight2Q.resize(static_cast<size_t>(k2Total) * weight2StrideElems, 0);
    oracle.fixture.scale2Channel.resize(weight2StrideElems, 0);
    for (uint32_t k2 = 0; k2 < k2Total; ++k2) {
        for (uint32_t col = 0; col < outputElems; ++col) {
            const int8_t value = static_cast<int8_t>(1 + ((col + k2 * 3 + localRank) % 7));
            oracle.fixture.weight2[static_cast<size_t>(k2) * weight2StrideElems + col] = static_cast<float>(value);
            oracle.fixture.weight2Q[static_cast<size_t>(k2) * weight2StrideElems + col] = value;
        }
    }
    for (uint32_t col = 0; col < weight2StrideElems; ++col) {
        float colScale = 1.0f;
        uint64_t scaleBits = 0;
        std::memcpy(&scaleBits, &colScale, sizeof(float));
        oracle.fixture.scale2Channel[col] = scaleBits;
    }
    return oracle;
}

bool CompareFloats(const std::vector<float>& lhs, const std::vector<float>& rhs, float atol, float rtol = 1e-3f)
{
    if (lhs.size() != rhs.size()) {
        return false;
    }
    for (size_t i = 0; i < lhs.size(); ++i) {
        const float tolerance = atol + rtol * std::fabs(rhs[i]);
        if (std::fabs(lhs[i] - rhs[i]) > tolerance) {
            return false;
        }
    }
    return true;
}

uint32_t RowsPerRankFromMode(const ModeConfig& mode)
{
    return mode.m;
}

RoutingFixture BuildPerfFixtureFromMode(const ModeConfig& mode)
{
    const uint32_t rowsPerRank = RowsPerRankFromMode(mode);
    const uint32_t hiddenBytes = mode.k * sizeof(int8_t);
    const uint32_t outputBytes = mode.k * sizeof(uint16_t);
    return BuildBalancedRoutingFixture(mode.worldSize,
                                                rowsPerRank,
                                                mode.topk,
                                                mode.expertsPerRank,
                                                mode.maxOutputSize,
                                                hiddenBytes,
                                                outputBytes,
                                                mode.n);
}

PerfLogicalMetrics BuildPerfLogicalMetrics(const RoutingFixture& fixture)
{
    PerfLogicalMetrics metrics;
    for (uint32_t rows : fixture.rowsPerRank) {
        metrics.inputTokens += rows;
    }
    const auto raw = ExpandBeforeCapacity(fixture);
    const auto executable = KeepExecutableEntries(raw, fixture);
    const RoutingSemanticMeta semanticMeta = BuildRoutingSemanticMeta(raw, executable, fixture);
    metrics.beforeCapacityTokens = semanticMeta.beforeCapacityCount;
    metrics.droppedByCapacityTokens = semanticMeta.droppedByCapacityCount;
    metrics.sentinelDropTokens = semanticMeta.sentinelDropCount;
    metrics.routedTokens = executable.size();
    for (const auto& entry : executable) {
        if (entry.srcRank != entry.dstRank) {
            ++metrics.remoteRoutedTokens;
        }
    }
    metrics.dispatchBytes = metrics.routedTokens * fixture.hiddenBytes;
    metrics.outputBytes = metrics.routedTokens * fixture.outputBytes;
    return metrics;
}

void PrintPerfMainSummary(const ModeConfig& mode,
                          int warmupIters,
                          int measureIters,
                          const PerfLogicalMetrics& metrics,
                          const std::string& scheduleTag,
                          const std::vector<double>& kernelSamplesUs,
                          const std::vector<double>& e2eSamplesUs)
{
    if (kernelSamplesUs.empty() || e2eSamplesUs.empty()) {
        return;
    }
    const PerfStats kernelStats = CalcStats(kernelSamplesUs);
    const PerfStats e2eStats = CalcStats(e2eSamplesUs);
    auto tokensPerSecond = [](uint64_t tokens, double us) -> double {
        return us > 0.0 ? static_cast<double>(tokens) * 1.0e6 / us : 0.0;
    };
    auto gbPerSecond = [](uint64_t bytes, double us) -> double {
        return us > 0.0 ? static_cast<double>(bytes) / 1.0e9 / (us / 1.0e6) : 0.0;
    };

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "\n===============================================================\n";
    std::cout << "[PROFILE] dispatch_combine_moe_v2 perf-main\n";
    std::cout << "  shape: m=" << mode.m
              << " k=" << mode.k
              << " n=" << mode.n
              << " topk=" << mode.topk
              << " experts_per_rank=" << mode.expertsPerRank
              << " world_size=" << mode.worldSize
              << " max_output_size=" << mode.maxOutputSize << '\n';
    std::cout << "  iters: warmup=" << warmupIters << " measure=" << measureIters
              << " schedule_tag=" << scheduleTag << '\n';
    std::cout << "  workload: generated_balanced_case input_tokens=" << metrics.inputTokens
              << " before_capacity_tokens=" << metrics.beforeCapacityTokens
              << " routed_tokens=" << metrics.routedTokens
              << " dropped_by_capacity=" << metrics.droppedByCapacityTokens
              << " sentinel_dropped=" << metrics.sentinelDropTokens
              << " remote_routed_tokens=" << metrics.remoteRoutedTokens
              << " dispatch_bytes=" << metrics.dispatchBytes
              << " output_bytes=" << metrics.outputBytes << '\n';
    std::cout << "  producers: dispatch_dequant=" << metrics.dispatchProducerCount
              << " swiglu=" << metrics.swigluProducerCount
              << " cube=" << metrics.cubeProducerCount << '\n';
    std::cout << "  tpush_tsync_pilot: active_path=" << metrics.tpushPilotActivePath
              << " requires_fused_cube_vec_kernel=" << metrics.tpushPilotRequiresFusedKernel
              << " boundary=" << metrics.tpushPilotBoundary << '\n';
    std::cout << "  runtime_routing_meta: token_per_expert_count=" << metrics.runtimeRoutingMeta.tokenPerExpertCount
              << " cumsum_mm_count=" << metrics.runtimeRoutingMeta.cumsumMMCount
              << " cumsum_send_count=" << metrics.runtimeRoutingMeta.cumsumSendCount
              << " expert_tokens_before_capacity_count=" << metrics.runtimeRoutingMeta.expertTokensBeforeCapacityCount
              << " group_count=" << metrics.runtimeRoutingMeta.groupCount
              << " checksum=" << metrics.runtimeRoutingChecksum << '\n';
    std::cout << "  kernel(max rank per iter): avg=" << kernelStats.avg << " us"
              << " min=" << kernelStats.min << " us"
              << " max=" << kernelStats.max << " us"
              << " std=" << kernelStats.stddev << " us\n";
    std::cout << "    input_tokens/s=" << tokensPerSecond(metrics.inputTokens, kernelStats.avg)
              << " routed_tokens/s=" << tokensPerSecond(metrics.routedTokens, kernelStats.avg)
              << " eq_comm=" << gbPerSecond(metrics.dispatchBytes, kernelStats.avg) << " GB/s\n";
    std::cout << "  e2e(max rank per iter):    avg=" << e2eStats.avg << " us"
              << " min=" << e2eStats.min << " us"
              << " max=" << e2eStats.max << " us"
              << " std=" << e2eStats.stddev << " us\n";
    std::cout << "    input_tokens/s=" << tokensPerSecond(metrics.inputTokens, e2eStats.avg)
              << " routed_tokens/s=" << tokensPerSecond(metrics.routedTokens, e2eStats.avg)
              << " eq_comm=" << gbPerSecond(metrics.dispatchBytes, e2eStats.avg) << " GB/s\n";
    std::cout << "  note: V2-T2 generates an in-memory balanced routing case from CLI shape; file-backed v2 case IO comes next.\n";
    std::cout << "===============================================================\n" << std::endl;
}

ModeConfig ParseMode(int argc, char** argv)
{
    ModeConfig cfg;
    auto parseUint = [](const char* text) -> uint32_t {
        return static_cast<uint32_t>(std::strtoul(text, nullptr, 10));
    };
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--mode" && (i + 1) < argc) {
            cfg.mode = argv[++i];
        } else if (arg == "--world-size" && (i + 1) < argc) {
            cfg.worldSize = parseUint(argv[++i]);
        } else if (arg == "--m" && (i + 1) < argc) {
            cfg.m = parseUint(argv[++i]);
        } else if (arg == "--k" && (i + 1) < argc) {
            cfg.k = parseUint(argv[++i]);
        } else if (arg == "--n" && (i + 1) < argc) {
            cfg.n = parseUint(argv[++i]);
        } else if (arg == "--topk" && (i + 1) < argc) {
            cfg.topk = parseUint(argv[++i]);
        } else if (arg == "--experts" && (i + 1) < argc) {
            cfg.expertsPerRank = parseUint(argv[++i]);
        } else if (arg == "--max-output-size" && (i + 1) < argc) {
            cfg.maxOutputSize = parseUint(argv[++i]);
        } else if (arg == "--check-golden") {
            cfg.checkGolden = true;
        } else if (!arg.empty() && arg[0] != '-') {
            cfg.mode = arg;
        }
    }
    return cfg;
}

void* AddBytes(void* base, uint64_t offset)
{
    return reinterpret_cast<void*>(reinterpret_cast<uint8_t*>(base) + offset);
}

const void* AddBytes(const void* base, uint64_t offset)
{
    return reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(base) + offset);
}

std::vector<CombineRowRange> BuildCombineRowRanges(
    const CombineTruthView& combine)
{
    std::vector<CombineRowRange> rowRanges;
    rowRanges.reserve(combine.rowToExpandedRange.size());
    for (const auto& range : combine.rowToExpandedRange) {
        rowRanges.push_back({range.first, range.second});
    }
    return rowRanges;
}

std::string BuildScheduleTag(const ExpertGroupSchedule& schedule)
{
    std::string tag;
    for (size_t i = 0; i < schedule.widths.size(); ++i) {
        if (!tag.empty()) {
            tag += 'x';
        }
        tag += std::to_string(schedule.widths[i]);
    }
    return tag.empty() ? std::string("none") : tag;
}

bool CopyHostToDeviceRegion(void* base, uint64_t offset, const void* src, size_t bytes)
{
    if (bytes == 0) {
        return true;
    }
    return aclrtMemcpy(AddBytes(base, offset), bytes, src, bytes, ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS;
}

bool AllocateAndCopyDeviceBuffer(void*& dst, const void* src, size_t bytes)
{
    dst = nullptr;
    if (bytes == 0) {
        return true;
    }
    if (aclrtMalloc(&dst, bytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS || dst == nullptr) {
        return false;
    }
    return aclrtMemcpy(dst, bytes, src, bytes, ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS;
}

bool AllocateDeviceBuffer(void*& dst, size_t bytes)
{
    dst = nullptr;
    if (bytes == 0) {
        return true;
    }
    return aclrtMalloc(&dst, bytes, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS && dst != nullptr;
}

bool CopyDeviceToHostFloats(std::vector<float>& dst, const void* src, size_t count)
{
    dst.resize(count);
    if (count == 0) {
        return true;
    }
    const size_t bytes = count * sizeof(float);
    return aclrtMemcpy(dst.data(), bytes, src, bytes, ACL_MEMCPY_DEVICE_TO_HOST) == ACL_SUCCESS;
}

int RunMegaMoePerfMain(int argc, char** argv)
{
    if (!CommMpiInit(&argc, &argv)) {
        return 1;
    }

    const int rank = CommMpiRank();
    const int worldSize = CommMpiSize();
    bool aclReady = false;
    bool pass = false;
    bool ok = worldSize == 2;
    const char* failStage = "world-size-check";
    if (!ok && rank == 0) {
        std::cerr << "perf-main requires worldSize=2, got " << worldSize << '\n';
    }

    ModeConfig mode = ParseMode(argc, argv);
    mode.mode = "perf-main";
    mode.localRank = static_cast<uint32_t>(rank);
    mode.worldSize = static_cast<uint32_t>(worldSize);

    StandaloneRankRuntime runtime{};
    HcclRootInfo rootInfo{};
    void* dispatchTaskDev = nullptr;
    void* dispatchRangeDev = nullptr;
    void* dispatchTilingDev = nullptr;
    void* computeTileDev = nullptr;
    void* expertSourceSegmentDev = nullptr;
    void* expertGroupRangeDev = nullptr;
    void* weight1Dev = nullptr;
    void* scale1ChannelDev = nullptr;
    void* gmm1OutDev = nullptr;
    void* swigluQDev = nullptr;
    void* scale2Dev = nullptr;
    void* weight2Dev = nullptr;
    void* scale2ChannelDev = nullptr;
    void* localPartialDev = nullptr;
    void* restoredRowsDev = nullptr;
    void* combineTaskDev = nullptr;
    void* combineRangeTaskDev = nullptr;
    void* incomingCombineTaskDev = nullptr;
    void* combineDoneValueDev = nullptr;
    void* combineProbDev = nullptr;
    void* rowRangesDev = nullptr;
    void* combineParamsPushDev = nullptr;
    void* combineParamsRestoreDev = nullptr;
    void* combineTilingDev = nullptr;
    void* dispatchToGmm1QueueDev = nullptr;
    void* gmm1ToSwiGluQueueDev = nullptr;
    void* swiGluToGmm2QueueDev = nullptr;
    void* gmm2ToCombineQueueDev = nullptr;
    EventHandle kernelStart;
    EventHandle kernelEnd;
    EventHandle dispatchReadyEvent;

    const int warmupIters = ParsePerfEnvInt("DISPATCH_COMBINE_MOE_V2_WARMUP_ITERS",
                                            "DISPATCH_FFN_COMBINE_V2_WARMUP_ITERS",
                                            kDefaultPerfWarmupIters);
    const int measureIters = ParsePerfEnvInt("DISPATCH_COMBINE_MOE_V2_MEASURE_ITERS",
                                             "DISPATCH_FFN_COMBINE_V2_MEASURE_ITERS",
                                             kDefaultPerfMeasureIters);
    if (warmupIters < 0 || measureIters < 0) {
        ok = false;
        failStage = "iteration-count-check";
    }
    bool invalidOutputBytes = false;
    const uint32_t requestedOutputBytes = mode.k * sizeof(float);
    if (ok && (requestedOutputBytes == 0 || (requestedOutputBytes % kCombineTileAlignBytes) != 0)) {
        ok = false;
        invalidOutputBytes = true;
        failStage = "output-bytes-alignment-check";
        if (rank == 0) {
            std::cerr << "perf-main requires outputBytes to be a positive multiple of "
                      << kCombineTileAlignBytes << " bytes, got " << requestedOutputBytes << '\n';
        }
    }
    if (invalidOutputBytes) {
        std::cerr << "[rank " << rank << "] perf-main failed at stage: " << failStage << '\n';
        CommMpiFinalize();
        return 1;
    }

    if (ok) {
        failStage = "acl-init";
        ok = aclInit(nullptr) == ACL_SUCCESS;
        aclReady = ok;
    }
    if (ok) {
        failStage = "set-device";
        ok = aclrtSetDevice(rank) == ACL_SUCCESS;
    }
    if (rank == 0 && ok) {
        failStage = "hccl-root-info";
        ok = HcclGetRootInfo(&rootInfo) == HCCL_SUCCESS;
    }
    CommMpiBcast(&rootInfo, HCCL_ROOT_INFO_BYTES, COMM_MPI_CHAR, 0);

    if (ok) {
        failStage = "runtime-init";
        ok = InitStandaloneRankRuntime(runtime, mode, rootInfo);
        if (!ok) {
            std::cerr << "[rank " << rank << "] runtime-init recentErrMsg=" << aclGetRecentErrMsg() << '\n';
        }
    }

    const auto fixture = BuildPerfFixtureFromMode(mode);
    const auto plan = BuildRoutePlanForRank(fixture, static_cast<uint32_t>(rank));
    const uint32_t outputElems = fixture.outputBytes / sizeof(uint16_t);
    const uint32_t gmm2PhysicalStrideElems = AlignUpUint32(outputElems, 32);
    const uint32_t currentPhysicalComputeK = mode.k;
    const uint32_t currentPhysicalGmm1N = mode.n;
    const auto semanticOracleSummary = BuildSemanticOracleSummary(fixture, plan, static_cast<uint32_t>(rank), outputElems);
    const auto semanticMeta = BuildRoutingSemanticMeta(plan.beforeCapacity, plan.executable, fixture);
    if (ok && (semanticMeta.beforeCapacityCount != plan.semanticMeta.beforeCapacityCount ||
               semanticMeta.executableCount != plan.semanticMeta.executableCount ||
               semanticMeta.droppedByCapacityCount != plan.semanticMeta.droppedByCapacityCount ||
               semanticMeta.sentinelDropCount != plan.semanticMeta.sentinelDropCount ||
               semanticOracleSummary.ledgerEntries != semanticMeta.beforeCapacityCount ||
               semanticOracleSummary.acceptedEntries != semanticMeta.executableCount ||
               semanticOracleSummary.droppedByCapacityEntries != semanticMeta.droppedByCapacityCount ||
               semanticOracleSummary.sentinelEntries != semanticMeta.sentinelDropCount)) {
        ok = false;
        failStage = "routing-semantic-ledger-check";
    }
    const auto publication = BuildDispatchPublicationForRank(fixture, static_cast<uint32_t>(rank));
    const auto dispatchOracle = BuildHostDispatchOracle(fixture, static_cast<uint32_t>(rank), plan);
    const auto rowRanges = BuildCombineRowRanges(plan.view.combine);
    const auto activationSchedule = BuildExpertGroupSchedule(mode.m >= 4097 ? 16U : mode.expertsPerRank);
    const std::string scheduleTag = BuildScheduleTag(activationSchedule);
    std::vector<CombinePushTask> incomingCombineTasks;
    for (uint32_t producer = 0; producer < static_cast<uint32_t>(worldSize); ++producer) {
        const auto producerPlan = BuildRoutePlanForRank(fixture, producer);
        for (const auto& task : producerPlan.combineTasks) {
            if (task.dstRank == static_cast<uint32_t>(rank)) {
                auto incoming = task;
                incoming.ownerCompletionIndex =
                    OwnerCompletionSummaryIndex(static_cast<uint32_t>(rank), incoming.dstRowBegin);
                incoming.taskFlags |= kCombineTaskPublishOwnerCompletion;
                incomingCombineTasks.push_back(incoming);
            }
        }
    }
    auto logicalMetrics = BuildPerfLogicalMetrics(fixture);
    const PtoTpushTsyncPilotPlan tpushPilotPlan = BuildGmm1ToSwiGluPilotPlanHost();
    logicalMetrics.tpushPilotActivePath = tpushPilotPlan.activePathEnabled;
    logicalMetrics.tpushPilotRequiresFusedKernel = tpushPilotPlan.requiresFusedCubeVecKernel;
    logicalMetrics.tpushPilotBoundary = tpushPilotPlan.candidateBoundary;
    logicalMetrics.runtimeRoutingMeta = plan.runtimeRoutingMeta;
    for (const uint32_t word : plan.runtimeRoutingWords) {
        logicalMetrics.runtimeRoutingChecksum = logicalMetrics.runtimeRoutingChecksum * 131U + word;
    }
    auto hostComputeRowsFromDispatch = [&](uint32_t dstRank) -> std::vector<float> {
        const auto dstPlan = BuildRoutePlanForRank(fixture, dstRank);
        const auto dstDispatch = BuildHostDispatchOracle(fixture, dstRank, dstPlan);
        const auto compute = BuildHostComputeOracle(currentPhysicalComputeK, currentPhysicalGmm1N, outputElems, dstRank);
        const uint64_t scaleBaseBytes = static_cast<uint64_t>(fixture.maxOutputSize) * fixture.hiddenBytes;
        const uint32_t rowCount = dstPlan.view.dispatch.packedRowCount;
        std::vector<float> rows(static_cast<size_t>(rowCount) * outputElems, 0.0f);
        for (uint32_t row = 0; row < rowCount; ++row) {
            float scale = 1.0f;
            const uint64_t payloadOffset = static_cast<uint64_t>(row) * fixture.hiddenBytes;
            const uint64_t scaleOffset = scaleBaseBytes +
                                         static_cast<uint64_t>(row) * kDispatchScaleSlotBytes;
            if (scaleOffset + sizeof(float) <= dstDispatch.size()) {
                std::memcpy(&scale, dstDispatch.data() + scaleOffset, sizeof(scale));
            }
            std::vector<float> dequant(currentPhysicalComputeK, 0.0f);
            for (uint32_t k = 0; k < currentPhysicalComputeK; ++k) {
                int8_t q = 0;
                if (payloadOffset + k < dstDispatch.size()) {
                    std::memcpy(&q, dstDispatch.data() + payloadOffset + k, sizeof(q));
                }
                dequant[k] = static_cast<float>(q) * scale;
            }
            std::vector<float> gmm1(currentPhysicalGmm1N, 0.0f);
            for (uint32_t n = 0; n < currentPhysicalGmm1N; ++n) {
                for (uint32_t k = 0; k < currentPhysicalComputeK; ++k) {
                    gmm1[n] += dequant[k] * compute.fixture.weight1[static_cast<size_t>(k) * currentPhysicalGmm1N + n];
                }
            }
            const uint32_t k2Total = currentPhysicalGmm1N / 2;
            std::vector<float> swigluFp(k2Total, 0.0f);
            float absMax = 0.0f;
            for (uint32_t k2 = 0; k2 < k2Total; ++k2) {
                const float gate = gmm1[k2];
                const float up = gmm1[k2Total + k2];
                const float sigmoid = 1.0f / (1.0f + std::exp(-gate));
                const float value = gate * sigmoid * up;
                swigluFp[k2] = value;
                absMax = std::max(absMax, std::fabs(value));
            }
            const float scale2 = std::max(absMax / 127.0f, 1.0e-8f);
            std::vector<int8_t> swigluQ(k2Total, 0);
            for (uint32_t k2 = 0; k2 < k2Total; ++k2) {
                const float quantized = std::round(swigluFp[k2] / scale2);
                const float clamped = std::max(-127.0f, std::min(127.0f, quantized));
                swigluQ[k2] = static_cast<int8_t>(clamped);
            }
            for (uint32_t col = 0; col < outputElems; ++col) {
                int32_t gmm2Acc = 0;
                for (uint32_t k2 = 0; k2 < k2Total; ++k2) {
                    gmm2Acc += static_cast<int32_t>(swigluQ[k2]) *
                               static_cast<int32_t>(compute.fixture.weight2Q[static_cast<size_t>(k2) * compute.fixture.weight2StrideElems + col]);
                }
                rows[static_cast<size_t>(row) * outputElems + col] = static_cast<float>(gmm2Acc) * scale2;
            }
        }
        return rows;
    };

    std::vector<RoutingPlanBundle> producerPlans;
    producerPlans.reserve(static_cast<size_t>(worldSize));
    std::vector<std::vector<float>> producerGmm2(static_cast<size_t>(worldSize));
    for (int producer = 0; producer < worldSize; ++producer) {
        producerPlans.push_back(BuildRoutePlanForRank(fixture, static_cast<uint32_t>(producer)));
        producerGmm2[static_cast<size_t>(producer)] = hostComputeRowsFromDispatch(static_cast<uint32_t>(producer));
    }

    auto partialValue = [&](uint32_t producerRank, uint32_t computeRowOrdinal, uint32_t col) -> float {
        const auto& rows = producerGmm2.at(producerRank);
        const size_t index = static_cast<size_t>(computeRowOrdinal) * outputElems + col;
        return index < rows.size() ? rows[index] : 0.0f;
    };

    std::vector<uint32_t> sourceOrdinalForEntry(plan.executable.size(), 0);
    std::vector<uint32_t> nextLocalOrdinal(static_cast<size_t>(worldSize), 0);
    for (size_t entryIdx = 0; entryIdx < plan.executable.size(); ++entryIdx) {
        const auto& entry = plan.executable[entryIdx];
        const auto& dstOffsets = producerPlans.at(entry.dstRank).view.dispatch.dstOffset;
        const uint32_t localOrdinal = nextLocalOrdinal.at(entry.dstRank)++;
        sourceOrdinalForEntry[entryIdx] = localOrdinal < dstOffsets.size() ? dstOffsets[localOrdinal] : 0U;
    }

    std::vector<float> localPartialHost(static_cast<size_t>(plan.view.dispatch.packedRowCount) * outputElems, 0.0f);
    if (localPartialHost.empty()) {
        localPartialHost.resize(outputElems == 0 ? 1 : outputElems, 0.0f);
    }
    const uint32_t localComputeRowCount = static_cast<uint32_t>(localPartialHost.size() / outputElems);
    const uint32_t dequantReadySignalIndex = static_cast<uint32_t>(runtime.layout.signalBytes / sizeof(int32_t) - 1);
    const uint32_t gmm2TileReadyIndex = plan.scoreboardMeta.gmm2TileReadyBase;

    std::vector<CombinePushTask> productionCombineTasks = plan.combineTasks;
    for (auto& task : productionCombineTasks) {
        task.gmm2ReadyRows = task.srcRowBegin + task.rowCount;
        task.taskFlags |= kCombineTaskPublishOwnerCompletion;
    }
    std::vector<CombineRangeTask> productionCombineRanges =
        CoalesceCombineRanges(productionCombineTasks);

    uint32_t activeRestoreRows = 0;
    for (uint32_t row = 0; row < rowRanges.size(); ++row) {
        if (rowRanges[row].begin != rowRanges[row].end) {
            activeRestoreRows = row + 1;
        }
    }
    std::vector<float> finalOracle(static_cast<size_t>(fixture.rowsPerRank.at(static_cast<uint32_t>(rank))) * outputElems,
                                   0.0f);
    for (size_t entryIdx = 0; entryIdx < plan.executable.size(); ++entryIdx) {
        const auto& entry = plan.executable[entryIdx];
        if (entry.ownerRank != static_cast<uint32_t>(rank)) {
            continue;
        }
        const uint32_t sourceOrdinal = sourceOrdinalForEntry[entryIdx];
        for (uint32_t col = 0; col < outputElems; ++col) {
            finalOracle[static_cast<size_t>(entry.rowIdx) * outputElems + col] +=
                entry.prob * partialValue(entry.dstRank, sourceOrdinal, col);
        }
    }

    if (ok) {
        const uint64_t dispatchSegmentBytes = static_cast<uint64_t>(fixture.maxOutputSize) * fixture.hiddenBytes;
        const uint64_t dispatchScaleSegmentBytes = static_cast<uint64_t>(fixture.maxOutputSize) *
                                                  kDispatchScaleSlotBytes;
        const uint64_t publicationBytes = static_cast<uint64_t>(fixture.worldSize) *
                                          (dispatchSegmentBytes + dispatchScaleSegmentBytes);
        const uint64_t computeBytes = dispatchSegmentBytes + dispatchScaleSegmentBytes;
        const uint64_t combineBytes = static_cast<uint64_t>(plan.view.combine.expandedProb.size()) * fixture.outputBytes;
        if (publicationBytes > runtime.layout.dispatchBytes ||
            computeBytes > runtime.layout.computeBytes ||
            combineBytes > runtime.layout.combineBytes) {
            ok = false;
            failStage = "workspace-capacity-check";
            if (rank == 0) {
                std::cerr << "perf-main workspace too small for generated case: dispatch=" << publicationBytes
                          << "/" << runtime.layout.dispatchBytes
                          << " compute=" << computeBytes << "/" << runtime.layout.computeBytes
                          << " combine=" << combineBytes << "/" << runtime.layout.combineBytes << '\n';
            }
        }
    }

    std::vector<int32_t> signalWords;
    if (ok) {
        signalWords.assign(static_cast<size_t>(runtime.layout.signalBytes / sizeof(int32_t)), 0);
        for (const auto& task : plan.dispatchTasks) {
            const uint32_t readyIndex = DispatchReadyIndex(task.srcRank, task.srcRowBegin);
            if (readyIndex < signalWords.size()) {
                signalWords[readyIndex] = static_cast<int32_t>(task.readyEpoch);
            }
        }
    }

    void* localWindow = nullptr;
    if (ok) {
        failStage = "publication-seed";
        localWindow = runtime.hccl.WindowIn(static_cast<uint32_t>(rank));
        ok = localWindow != nullptr &&
             CopyHostToDeviceRegion(localWindow,
                                    runtime.hccl.host_remote_window_ctx.dispatchRegionOffset,
                                    publication.data(),
                                    publication.size()) &&
             CopyHostToDeviceRegion(localWindow,
                                    runtime.hccl.host_remote_window_ctx.signalRegionOffset,
                                    signalWords.data(),
                                    signalWords.size() * sizeof(int32_t));
    }

    if (ok) {
        failStage = "dispatch-range-copy";
        ok = AllocateAndCopyDeviceBuffer(dispatchRangeDev,
                                         plan.dispatchRanges.data(),
                                         plan.dispatchRanges.size() * sizeof(DispatchRangeTask));
    }

    StandaloneKernelTilingData dispatchTiling{};
    dispatchTiling.mode = static_cast<uint32_t>(KernelMode::DispatchRangeOnly);
    dispatchTiling.taskCount = static_cast<uint32_t>(plan.dispatchRanges.size());
    dispatchTiling.hiddenBytes = fixture.hiddenBytes;
    dispatchTiling.outputBytes = fixture.outputBytes;
    dispatchTiling.groupWidth = fixture.expertsPerRank;
    dispatchTiling.maxOutputSize = fixture.maxOutputSize;
    if (ok) {
        failStage = "dispatch-tiling-copy";
        ok = AllocateAndCopyDeviceBuffer(dispatchTilingDev, &dispatchTiling, sizeof(dispatchTiling));
    }
    if (ok) {
        failStage = "compute-tile-meta-copy";
        ok = AllocateAndCopyDeviceBuffer(computeTileDev,
                                         plan.computeTiles.data(),
                                         plan.computeTiles.size() * sizeof(ComputeTileMeta));
    }
    if (ok) {
        failStage = "expert-source-segment-copy";
        ok = AllocateAndCopyDeviceBuffer(expertSourceSegmentDev,
                                         plan.expertSourceSegments.data(),
                                         plan.expertSourceSegments.size() * sizeof(ExpertSourceSegmentMeta));
    }
    if (ok) {
        failStage = "expert-group-range-copy";
        ok = AllocateAndCopyDeviceBuffer(expertGroupRangeDev,
                                         plan.expertGroupRanges.data(),
                                         plan.expertGroupRanges.size() * sizeof(ExpertGroupTaskRange));
    }

    const auto megaTiling = BuildMegaMoeTilingData(mode,
                                                           runtime.layout,
                                                           static_cast<uint32_t>(plan.dispatchRanges.size()),
                                                           static_cast<uint32_t>(plan.combineTasks.size()),
                                                           plan.dispatchRanges.empty() ? 1U : static_cast<uint32_t>(plan.dispatchRanges.size()),
                                                           fixture.hiddenBytes,
                                                           fixture.outputBytes);
    if (ok) {
        failStage = "megamoe-workspace-contract-check";
        ok = ValidateMegaMoeWorkspace(megaTiling);
    }

    const auto computeOracle = BuildHostComputeOracle(currentPhysicalComputeK, currentPhysicalGmm1N, outputElems, static_cast<uint32_t>(rank));
    const size_t computeTileRowCapacity = ((static_cast<size_t>(localComputeRowCount) + 15) / 16) * 16;
    if (ok) {
        failStage = "compute-weight1-int8-copy";
        ok = AllocateAndCopyDeviceBuffer(weight1Dev,
                                         computeOracle.fixture.weight1Q.data(),
                                         computeOracle.fixture.weight1Q.size() * sizeof(int8_t));
    }
    if (ok) {
        failStage = "compute-scale1-channel-copy";
        ok = AllocateAndCopyDeviceBuffer(scale1ChannelDev,
                                         computeOracle.fixture.scale1Channel.data(),
                                         computeOracle.fixture.scale1Channel.size() * sizeof(uint64_t));
    }
    if (ok) {
        failStage = "compute-gmm1-alloc";
        ok = AllocateDeviceBuffer(gmm1OutDev, computeTileRowCapacity * currentPhysicalGmm1N * sizeof(uint16_t));
    }
    if (ok) {
        failStage = "compute-swiglu-quant-alloc";
        ok = AllocateDeviceBuffer(swigluQDev, computeTileRowCapacity * (currentPhysicalGmm1N / 2) * sizeof(int8_t));
    }
    if (ok) {
        failStage = "compute-scale2-alloc";
        ok = AllocateDeviceBuffer(scale2Dev, computeTileRowCapacity * sizeof(float));
    }
    if (ok) {
        failStage = "compute-weight2-copy";
        ok = AllocateAndCopyDeviceBuffer(weight2Dev,
                                         computeOracle.fixture.weight2Q.data(),
                                         computeOracle.fixture.weight2Q.size() * sizeof(int8_t));
    }
    if (ok) {
        failStage = "compute-scale2-channel-copy";
        ok = AllocateAndCopyDeviceBuffer(scale2ChannelDev,
                                         computeOracle.fixture.scale2Channel.data(),
                                         computeOracle.fixture.scale2Channel.size() * sizeof(uint64_t));
    }
    if (ok) {
        failStage = "local-partial-alloc";
        ok = AllocateDeviceBuffer(localPartialDev,
                                  (localPartialHost.size() + computeTileRowCapacity * outputElems) * sizeof(uint16_t));
    }
    if (ok) {
        failStage = "restored-rows-alloc";
        ok = AllocateDeviceBuffer(restoredRowsDev, finalOracle.size() * sizeof(float));
    }
    if (ok) {
        failStage = "combine-task-copy";
        ok = AllocateAndCopyDeviceBuffer(combineTaskDev,
                                         productionCombineTasks.data(),
                                         productionCombineTasks.size() * sizeof(CombinePushTask));
    }
    if (ok) {
        failStage = "combine-range-task-copy";
        ok = AllocateAndCopyDeviceBuffer(combineRangeTaskDev,
                                         productionCombineRanges.data(),
                                         productionCombineRanges.size() * sizeof(CombineRangeTask));
    }
    if (ok) {
        failStage = "incoming-combine-task-copy";
        ok = AllocateAndCopyDeviceBuffer(incomingCombineTaskDev,
                                         incomingCombineTasks.data(),
                                         incomingCombineTasks.size() * sizeof(CombinePushTask));
    }
    const int32_t combineDoneValue[kCombineTileAlignBytes / sizeof(int32_t)] = {1};
    if (ok) {
        failStage = "combine-done-value-copy";
        ok = AllocateAndCopyDeviceBuffer(combineDoneValueDev, combineDoneValue, sizeof(combineDoneValue));
    }
    if (ok) {
        failStage = "combine-prob-copy";
        ok = AllocateAndCopyDeviceBuffer(combineProbDev,
                                         plan.view.combine.expandedProb.data(),
                                         plan.view.combine.expandedProb.size() * sizeof(float));
    }
    if (ok) {
        failStage = "row-range-copy";
        ok = AllocateAndCopyDeviceBuffer(rowRangesDev,
                                         rowRanges.data(),
                                         rowRanges.size() * sizeof(CombineRowRange));
    }

    CombineOnlyParams combinePushParams{};
    combinePushParams.localPartial = reinterpret_cast<uint64_t>(localPartialDev);
    combinePushParams.restoredRows = reinterpret_cast<uint64_t>(restoredRowsDev);
    combinePushParams.combineTasks = reinterpret_cast<uint64_t>(combineTaskDev);
    combinePushParams.combineRangeTasks = reinterpret_cast<uint64_t>(combineRangeTaskDev);
    combinePushParams.expandedProb = reinterpret_cast<uint64_t>(combineProbDev);
    combinePushParams.rowRanges = reinterpret_cast<uint64_t>(rowRangesDev);
    combinePushParams.incomingCombineTasks = reinterpret_cast<uint64_t>(incomingCombineTaskDev);
    combinePushParams.doneValue = reinterpret_cast<uint64_t>(combineDoneValueDev);
    combinePushParams.taskCount = static_cast<uint32_t>(productionCombineTasks.size());
    combinePushParams.rangeTaskCount = static_cast<uint32_t>(productionCombineRanges.size());
    combinePushParams.localRowCount = activeRestoreRows;
    combinePushParams.phase = 1;
    combinePushParams.incomingTaskCount = static_cast<uint32_t>(incomingCombineTasks.size());
    combinePushParams.perTokenScale2 = reinterpret_cast<uint64_t>(scale2Dev);
    combinePushParams.outputElems = outputElems;
    if (ok) {
        failStage = "combine-push-params-copy";
        ok = AllocateAndCopyDeviceBuffer(combineParamsPushDev, &combinePushParams, sizeof(combinePushParams));
    }

    CombineOnlyParams combineRestoreParams = combinePushParams;
    combineRestoreParams.rangeTaskCount = 0;
    combineRestoreParams.phase = 2;
    if (ok) {
        failStage = "combine-restore-params-copy";
        ok = AllocateAndCopyDeviceBuffer(combineParamsRestoreDev, &combineRestoreParams, sizeof(combineRestoreParams));
    }

    StandaloneKernelTilingData combineTiling{};
    combineTiling.mode = static_cast<uint32_t>(KernelMode::CombineOnly);
    combineTiling.outputBytes = fixture.outputBytes;
    if (ok) {
        failStage = "combine-tiling-copy";
        ok = AllocateAndCopyDeviceBuffer(combineTilingDev, &combineTiling, sizeof(combineTiling));
    }

    const uint32_t pipelineTileCount = static_cast<uint32_t>(plan.computeTiles.size());
    const uint32_t cubeProducerCount = pipelineTileCount == 0 ? 1U :
        (pipelineTileCount > kPipelineQueueMaxProducers ? kPipelineQueueMaxProducers : pipelineTileCount);
    const uint32_t vecDispatchProducerCount = pipelineTileCount == 0 ? 1U :
        (pipelineTileCount > 2U ? 2U : pipelineTileCount);
    const uint32_t vecSwiGluProducerCount = pipelineTileCount == 0 ? 1U :
        (pipelineTileCount > 2U ? 2U : pipelineTileCount);
    const PipelineQueuePlan dispatchProducerQueuePlan = BuildPipelineQueuePlanHost(pipelineTileCount, vecDispatchProducerCount);
    const PipelineQueuePlan swigluProducerQueuePlan = BuildPipelineQueuePlanHost(pipelineTileCount, vecSwiGluProducerCount);
    const PipelineQueuePlan cubeProducerQueuePlan = BuildPipelineQueuePlanHost(pipelineTileCount, cubeProducerCount);
    const auto dispatchToGmm1QueueHost = MakePipelineQueueSetHost(dispatchProducerQueuePlan.producerCount,
                                                                  dispatchProducerQueuePlan.dispatchToGmm1Expected,
                                                                  dispatchProducerQueuePlan.capacity);
    const auto gmm1ToSwiGluQueueHost = MakePipelineQueueSetHost(cubeProducerQueuePlan.producerCount,
                                                                cubeProducerQueuePlan.gmm1ToSwiGluExpected,
                                                                cubeProducerQueuePlan.capacity);
    const auto swiGluToGmm2QueueHost = MakePipelineQueueSetHost(swigluProducerQueuePlan.producerCount,
                                                                swigluProducerQueuePlan.swiGluToGmm2Expected,
                                                                swigluProducerQueuePlan.capacity);
    const auto gmm2ToCombineQueueHost = MakePipelineQueueSetHost(cubeProducerQueuePlan.producerCount,
                                                                 cubeProducerQueuePlan.gmm2ToCombineExpected,
                                                                 cubeProducerQueuePlan.capacity);
    logicalMetrics.dispatchProducerCount = dispatchProducerQueuePlan.producerCount;
    logicalMetrics.swigluProducerCount = swigluProducerQueuePlan.producerCount;
    logicalMetrics.cubeProducerCount = cubeProducerQueuePlan.producerCount;
    if (ok) {
        failStage = "dispatch-to-gmm1-queue-copy";
        ok = AllocateAndCopyDeviceBuffer(dispatchToGmm1QueueDev,
                                         dispatchToGmm1QueueHost.data(),
                                         dispatchToGmm1QueueHost.size());
    }
    if (ok) {
        failStage = "gmm1-to-swiglu-queue-copy";
        ok = AllocateAndCopyDeviceBuffer(gmm1ToSwiGluQueueDev,
                                         gmm1ToSwiGluQueueHost.data(),
                                         gmm1ToSwiGluQueueHost.size());
    }
    if (ok) {
        failStage = "swiglu-to-gmm2-queue-copy";
        ok = AllocateAndCopyDeviceBuffer(swiGluToGmm2QueueDev,
                                         swiGluToGmm2QueueHost.data(),
                                         swiGluToGmm2QueueHost.size());
    }
    if (ok) {
        failStage = "gmm2-to-combine-queue-copy";
        ok = AllocateAndCopyDeviceBuffer(gmm2ToCombineQueueDev,
                                         gmm2ToCombineQueueHost.data(),
                                         gmm2ToCombineQueueHost.size());
    }

    if (ok) {
        failStage = "event-create";
        if (measureIters > 0) {
            ok = aclrtCreateEvent(&kernelStart.event) == ACL_SUCCESS &&
                 aclrtCreateEvent(&kernelEnd.event) == ACL_SUCCESS;
        }
    }

    DispatchRangeLaunchArgs dispatchLaunchArgs;
    dispatchLaunchArgs.remoteWindow = runtime.hccl.RemoteWindowContextPtr();
    dispatchLaunchArgs.dispatchRanges = dispatchRangeDev;
    dispatchLaunchArgs.tiling = dispatchTilingDev;
    dispatchLaunchArgs.blockDim = 1;

    const uint64_t computeScaleBaseBytes = static_cast<uint64_t>(fixture.maxOutputSize) * fixture.hiddenBytes;
    void* quantInputDev = AddBytes(localWindow, runtime.hccl.host_remote_window_ctx.computeRegionOffset);
    void* scale1Dev = AddBytes(localWindow, runtime.hccl.host_remote_window_ctx.computeRegionOffset + computeScaleBaseBytes);
    void* signalBaseDev = AddBytes(localWindow, runtime.hccl.host_remote_window_ctx.signalRegionOffset);


    CombineOnlyParams combinePipelineParams = combinePushParams;
    combinePipelineParams.phase = 3;
    void* combineParamsPipelineDev = nullptr;
    if (ok) {
        failStage = "combine-pipeline-params-copy";
        ok = AllocateAndCopyDeviceBuffer(combineParamsPipelineDev, &combinePipelineParams, sizeof(combinePipelineParams));
    }

    CombineOnlyLaunchArgs combinePushLaunchArgs;
    combinePushLaunchArgs.remoteWindow = runtime.hccl.RemoteWindowContextPtr();
    combinePushLaunchArgs.params = combineParamsPushDev;
    combinePushLaunchArgs.tiling = combineTilingDev;
    combinePushLaunchArgs.blockDim = combinePushParams.rangeTaskCount != 0 ? combinePushParams.rangeTaskCount :
                                      (combinePushParams.taskCount == 0 ? 1 : combinePushParams.taskCount);

    CombineOnlyLaunchArgs combinePipelineLaunchArgs;
    combinePipelineLaunchArgs.remoteWindow = runtime.hccl.RemoteWindowContextPtr();
    combinePipelineLaunchArgs.params = combineParamsPipelineDev;
    combinePipelineLaunchArgs.tiling = combineTilingDev;
    combinePipelineLaunchArgs.blockDim = combinePushLaunchArgs.blockDim + 2;

    CommVecQueueLaunchArgs commVecQueueLaunchArgs;
    commVecQueueLaunchArgs.remoteWindow = runtime.hccl.RemoteWindowContextPtr();
    commVecQueueLaunchArgs.quantInput = quantInputDev;
    commVecQueueLaunchArgs.scale1 = scale1Dev;
    commVecQueueLaunchArgs.gmm1Out = gmm1OutDev;
    commVecQueueLaunchArgs.swigluQ = swigluQDev;
    commVecQueueLaunchArgs.scale2 = scale2Dev;
    commVecQueueLaunchArgs.gmm2Out = localPartialDev;
    commVecQueueLaunchArgs.signalBase = signalBaseDev;
    commVecQueueLaunchArgs.computeTiles = computeTileDev;
    commVecQueueLaunchArgs.expertSourceSegments = expertSourceSegmentDev;
    commVecQueueLaunchArgs.expertGroupRanges = expertGroupRangeDev;
    commVecQueueLaunchArgs.combineParams = combineParamsPipelineDev;
    commVecQueueLaunchArgs.combineTiling = combineTilingDev;
    commVecQueueLaunchArgs.dispatchRanges = dispatchRangeDev;
    commVecQueueLaunchArgs.dispatchTiling = dispatchTilingDev;
    commVecQueueLaunchArgs.dispatchToGmm1Queue = dispatchToGmm1QueueDev;
    commVecQueueLaunchArgs.gmm1ToSwiGluQueue = gmm1ToSwiGluQueueDev;
    commVecQueueLaunchArgs.swiGluToGmm2Queue = swiGluToGmm2QueueDev;
    commVecQueueLaunchArgs.gmm2ToCombineQueue = gmm2ToCombineQueueDev;
    commVecQueueLaunchArgs.expertSafeRowsIndex = dequantReadySignalIndex;
    commVecQueueLaunchArgs.requiredSafeRows = localComputeRowCount;
    commVecQueueLaunchArgs.expertGroupCount = static_cast<uint32_t>(plan.expertGroupRanges.size());
    commVecQueueLaunchArgs.vecDispatchProducerCount = dispatchProducerQueuePlan.producerCount;
    commVecQueueLaunchArgs.vecSwiGluProducerCount = swigluProducerQueuePlan.producerCount;
    commVecQueueLaunchArgs.computeTileCount = static_cast<uint32_t>(plan.computeTiles.size());
    commVecQueueLaunchArgs.rowCount = localComputeRowCount;
    commVecQueueLaunchArgs.inputStrideBytes = fixture.hiddenBytes;
    commVecQueueLaunchArgs.scaleStrideBytes = kDispatchScaleSlotBytes;
    commVecQueueLaunchArgs.outputElems = outputElems;
    commVecQueueLaunchArgs.outputStrideElems = gmm2PhysicalStrideElems;
    commVecQueueLaunchArgs.hiddenK = currentPhysicalComputeK;
    commVecQueueLaunchArgs.gmm1N = currentPhysicalGmm1N;
    commVecQueueLaunchArgs.blockDim = combinePipelineLaunchArgs.blockDim +
        dispatchProducerQueuePlan.producerCount + swigluProducerQueuePlan.producerCount;

    ComputeCubeQueueLaunchArgs computeCubeQueueLaunchArgs;
    computeCubeQueueLaunchArgs.quantInput = quantInputDev;
    computeCubeQueueLaunchArgs.weight1 = weight1Dev;
    computeCubeQueueLaunchArgs.gmm1Out = gmm1OutDev;
    computeCubeQueueLaunchArgs.scale1Channel = scale1ChannelDev;
    computeCubeQueueLaunchArgs.swigluQ = swigluQDev;
    computeCubeQueueLaunchArgs.weight2 = weight2Dev;
    computeCubeQueueLaunchArgs.gmm2Out = localPartialDev;
    computeCubeQueueLaunchArgs.scale2Channel = scale2ChannelDev;
    computeCubeQueueLaunchArgs.signalBase = signalBaseDev;
    computeCubeQueueLaunchArgs.dispatchToGmm1Queue = dispatchToGmm1QueueDev;
    computeCubeQueueLaunchArgs.gmm1ToSwiGluQueue = gmm1ToSwiGluQueueDev;
    computeCubeQueueLaunchArgs.swiGluToGmm2Queue = swiGluToGmm2QueueDev;
    computeCubeQueueLaunchArgs.gmm2ToCombineQueue = gmm2ToCombineQueueDev;
    computeCubeQueueLaunchArgs.gmm2TileReadyIndex = gmm2TileReadyIndex;
    computeCubeQueueLaunchArgs.gmm2TileReadyCount = static_cast<uint32_t>(plan.computeTiles.size());
    computeCubeQueueLaunchArgs.rowCount = localComputeRowCount;
    computeCubeQueueLaunchArgs.outputElems = outputElems;
    computeCubeQueueLaunchArgs.outputStrideElems = gmm2PhysicalStrideElems;
    computeCubeQueueLaunchArgs.weight2StrideElems = computeOracle.fixture.weight2StrideElems;
    computeCubeQueueLaunchArgs.hiddenK = currentPhysicalComputeK;
    computeCubeQueueLaunchArgs.gmm1N = currentPhysicalGmm1N;
    computeCubeQueueLaunchArgs.blockDim = cubeProducerCount;


    auto resetFullPipeline = [&]() -> bool {
        void* window = runtime.hccl.WindowIn(static_cast<uint32_t>(rank));
        if (window == nullptr) {
            return false;
        }
        constexpr uint64_t dispatchDoneOffsetBytes = static_cast<uint64_t>(4096) * sizeof(int32_t);
        constexpr uint64_t dispatchDoneBytes = static_cast<uint64_t>(64 * 64) * sizeof(int32_t);
        constexpr uint64_t combineDoneOffsetBytes = static_cast<uint64_t>(16384) * sizeof(int32_t);
        constexpr uint64_t combineDoneBytes = static_cast<uint64_t>(64 * 64) * kCombineTileAlignBytes;
        constexpr uint64_t summaryOffsetBytes = static_cast<uint64_t>(20480) * sizeof(int32_t);
        constexpr uint64_t summaryBytes = static_cast<uint64_t>(64 * 64) * sizeof(int32_t);
        const int32_t dequantReadyRows = static_cast<int32_t>(localComputeRowCount);
        return aclrtMemset(dispatchToGmm1QueueDev,
                           dispatchToGmm1QueueHost.size(),
                           0,
                           dispatchToGmm1QueueHost.size()) == ACL_SUCCESS &&
               aclrtMemcpy(dispatchToGmm1QueueDev,
                           dispatchToGmm1QueueHost.size(),
                           dispatchToGmm1QueueHost.data(),
                           dispatchToGmm1QueueHost.size(),
                           ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS &&
               aclrtMemset(gmm1ToSwiGluQueueDev,
                           gmm1ToSwiGluQueueHost.size(),
                           0,
                           gmm1ToSwiGluQueueHost.size()) == ACL_SUCCESS &&
               aclrtMemcpy(gmm1ToSwiGluQueueDev,
                           gmm1ToSwiGluQueueHost.size(),
                           gmm1ToSwiGluQueueHost.data(),
                           gmm1ToSwiGluQueueHost.size(),
                           ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS &&
               aclrtMemset(swiGluToGmm2QueueDev,
                           swiGluToGmm2QueueHost.size(),
                           0,
                           swiGluToGmm2QueueHost.size()) == ACL_SUCCESS &&
               aclrtMemcpy(swiGluToGmm2QueueDev,
                           swiGluToGmm2QueueHost.size(),
                           swiGluToGmm2QueueHost.data(),
                           swiGluToGmm2QueueHost.size(),
                           ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS &&
               aclrtMemset(gmm2ToCombineQueueDev,
                           gmm2ToCombineQueueHost.size(),
                           0,
                           gmm2ToCombineQueueHost.size()) == ACL_SUCCESS &&
               aclrtMemcpy(gmm2ToCombineQueueDev,
                           gmm2ToCombineQueueHost.size(),
                           gmm2ToCombineQueueHost.data(),
                           gmm2ToCombineQueueHost.size(),
                           ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS &&
               aclrtMemset(AddBytes(window, runtime.hccl.host_remote_window_ctx.computeRegionOffset),
                           runtime.layout.computeBytes,
                           0,
                           runtime.layout.computeBytes) == ACL_SUCCESS &&
               aclrtMemset(AddBytes(window, runtime.hccl.host_remote_window_ctx.combineRegionOffset),
                           runtime.layout.combineBytes,
                           0,
                           runtime.layout.combineBytes) == ACL_SUCCESS &&
               aclrtMemset(AddBytes(window, runtime.hccl.host_remote_window_ctx.signalRegionOffset + dispatchDoneOffsetBytes),
                           dispatchDoneBytes,
                           0,
                           dispatchDoneBytes) == ACL_SUCCESS &&
               aclrtMemset(AddBytes(window, runtime.hccl.host_remote_window_ctx.signalRegionOffset + combineDoneOffsetBytes),
                           combineDoneBytes,
                           0,
                           combineDoneBytes) == ACL_SUCCESS &&
               aclrtMemset(AddBytes(window, runtime.hccl.host_remote_window_ctx.signalRegionOffset + summaryOffsetBytes),
                           summaryBytes,
                           0,
                           summaryBytes) == ACL_SUCCESS &&
               aclrtMemcpy(AddBytes(window,
                                    runtime.hccl.host_remote_window_ctx.signalRegionOffset +
                                        static_cast<uint64_t>(dequantReadySignalIndex) * sizeof(int32_t)),
                           sizeof(dequantReadyRows),
                           &dequantReadyRows,
                           sizeof(dequantReadyRows),
                           ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS &&
               aclrtMemset(localPartialDev,
                           localPartialHost.size() * sizeof(uint16_t),
                           0,
                           localPartialHost.size() * sizeof(uint16_t)) == ACL_SUCCESS &&
               aclrtMemset(restoredRowsDev,
                           finalOracle.size() * sizeof(float),
                           0,
                           finalOracle.size() * sizeof(float)) == ACL_SUCCESS;
    };

    CombineOnlyLaunchArgs combineRestoreLaunchArgs;
    combineRestoreLaunchArgs.remoteWindow = runtime.hccl.RemoteWindowContextPtr();
    combineRestoreLaunchArgs.params = combineParamsRestoreDev;
    combineRestoreLaunchArgs.tiling = combineTilingDev;
    combineRestoreLaunchArgs.blockDim = 1;

    auto enqueuePtoPipeline = [&]() -> bool {
        launchCommVecQueuePipeline(commVecQueueLaunchArgs, runtime.dispatch_stream);
        launchComputeCubeQueuePipeline(computeCubeQueueLaunchArgs, runtime.compute_stream);
        return true;
    };

    auto synchronizePtoPipeline = [&]() -> bool {
        const int syncTimeoutMs = ParseEnvInt("DISPATCH_COMBINE_MOE_V2_SYNC_TIMEOUT_MS", 0);
        bool commOk = false;
        bool computeOk = false;
        if (syncTimeoutMs > 0) {
            commOk = aclrtSynchronizeStreamWithTimeout(runtime.dispatch_stream, syncTimeoutMs) == ACL_SUCCESS;
            computeOk = aclrtSynchronizeStreamWithTimeout(runtime.compute_stream, syncTimeoutMs) == ACL_SUCCESS;
        } else {
            commOk = aclrtSynchronizeStream(runtime.dispatch_stream) == ACL_SUCCESS;
            computeOk = aclrtSynchronizeStream(runtime.compute_stream) == ACL_SUCCESS;
        }
        return computeOk && commOk;
    };

    auto launchOnce = [&]() -> bool {
        return enqueuePtoPipeline() && synchronizePtoPipeline();
    };

    std::vector<double> kernelTimesUs;
    std::vector<double> e2eTimesUs;
    bool hasPipelineOutput = false;
    if (ok) {
        kernelTimesUs.reserve(static_cast<size_t>(measureIters));
        e2eTimesUs.reserve(static_cast<size_t>(measureIters));
        CommMpiBarrier();
        for (int iter = 0; ok && iter < warmupIters; ++iter) {
            failStage = "warmup-reset";
            ok = resetFullPipeline();
            CommMpiBarrier();
            if (ok) {
                failStage = "warmup-launch";
                ok = launchOnce();
                hasPipelineOutput = ok;
            }
            CommMpiBarrier();
        }

        for (int iter = 0; ok && iter < measureIters; ++iter) {
            failStage = "measure-reset";
            ok = resetFullPipeline();
            CommMpiBarrier();
            const auto hostStart = std::chrono::high_resolution_clock::now();
            if (ok) {
                failStage = "measure-event-start";
                ok = aclrtRecordEvent(kernelStart.event, runtime.compute_stream) == ACL_SUCCESS;
            }
            if (ok) {
                failStage = "measure-launch";
                ok = enqueuePtoPipeline() &&
                     synchronizePtoPipeline() &&
                     aclrtRecordEvent(kernelEnd.event, runtime.compute_stream) == ACL_SUCCESS &&
                     aclrtSynchronizeStream(runtime.compute_stream) == ACL_SUCCESS;
            }
            const auto hostEnd = std::chrono::high_resolution_clock::now();
            CommMpiBarrier();
            if (ok) {
                float kernelMs = 0.0f;
                failStage = "measure-event-elapsed";
                ok = aclrtEventElapsedTime(&kernelMs, kernelStart.event, kernelEnd.event) == ACL_SUCCESS;
                if (ok) {
                    kernelTimesUs.push_back(static_cast<double>(kernelMs) * 1000.0);
                    e2eTimesUs.push_back(std::chrono::duration<double, std::micro>(hostEnd - hostStart).count());
                    hasPipelineOutput = true;
                }
            }
        }
    }

    if (ok) {
        const auto kernelMaxSamples = GatherMaxSamplesToRoot(kernelTimesUs, rank, worldSize);
        const auto e2eMaxSamples = GatherMaxSamplesToRoot(e2eTimesUs, rank, worldSize);
        if (rank == 0) {
            PrintPerfMainSummary(mode, warmupIters, measureIters, logicalMetrics, scheduleTag, kernelMaxSamples, e2eMaxSamples);
        }
    }

    for (int turn = 0; turn < worldSize; ++turn) {
        CommMpiBarrier();
        if (turn == rank) {
            std::cout << "[rank " << rank << "] semantic-ledger"
                      << " entries=" << semanticOracleSummary.ledgerEntries
                      << " accepted=" << semanticOracleSummary.acceptedEntries
                      << " dropped_by_capacity=" << semanticOracleSummary.droppedByCapacityEntries
                      << " sentinel=" << semanticOracleSummary.sentinelEntries
                      << " token_per_expert_words=" << semanticOracleSummary.tokenPerExpertWords
                      << " cumsum_mm_words=" << semanticOracleSummary.cumsumMMWords
                      << " cumsum_send_words=" << semanticOracleSummary.cumsumSendWords
                      << " owner_rows=" << semanticOracleSummary.ownerRows
                      << " dispatch_layout_ranges=" << semanticOracleSummary.dispatchLayoutRanges
                      << " compute_layout_tiles=" << semanticOracleSummary.computeLayoutTiles
                      << " dispatch_layout_checksum=" << semanticOracleSummary.dispatchLayoutChecksum
                      << " compute_layout_checksum=" << semanticOracleSummary.computeLayoutChecksum
                      << " gmm1_skeleton_tiles=" << semanticOracleSummary.gmm1SkeletonTiles
                      << " gmm1_skeleton_true_shape_ready=" << semanticOracleSummary.gmm1SkeletonTrueShapeReady
                      << " gmm2_skeleton_tiles=" << semanticOracleSummary.gmm2SkeletonTiles
                      << " gmm2_skeleton_true_shape_ready=" << semanticOracleSummary.gmm2SkeletonTrueShapeReady
                      << " restore_trace_entries=" << semanticOracleSummary.restoreTraceEntries
                      << " restore_trace_rows=" << semanticOracleSummary.restoreTraceRows
                      << " restore_trace_checksum=" << semanticOracleSummary.restoreTraceChecksum
                      << " ledger_checksum=" << semanticOracleSummary.ledgerChecksum
                      << " route_checksum=" << semanticOracleSummary.routeChecksum
                      << " gmm1_checksum=" << semanticOracleSummary.gmm1Checksum
                      << " swiglu_checksum=" << semanticOracleSummary.swigluChecksum
                      << " quant_checksum=" << semanticOracleSummary.quantChecksum
                      << " gmm2_checksum=" << semanticOracleSummary.gmm2Checksum
                      << " final_checksum=" << semanticOracleSummary.finalChecksum
                      << " note=semantic_oracle_chain_not_active_toy_compute_golden" << '\n';
        }
    }

    std::vector<float> actual;
    if (ok && !hasPipelineOutput) {
        failStage = "accuracy-reset";
        ok = resetFullPipeline();
        CommMpiBarrier();
        if (ok) {
            failStage = "accuracy-launch";
            ok = launchOnce();
            hasPipelineOutput = ok;
            CommMpiBarrier();
        }
    }
    if (ok) {
        failStage = "accuracy-copy-back";
        ok = CopyDeviceToHostFloats(actual, restoredRowsDev, finalOracle.size());
    }
    if (ok) {
        pass = CompareFloats(actual, finalOracle, 1e-3f);
        if (!pass) {
            size_t mismatch = 0;
            while (mismatch < actual.size() && mismatch < finalOracle.size()) {
                const float tolerance = 1e-3f + 1e-3f * std::fabs(finalOracle[mismatch]);
                if (std::fabs(actual[mismatch] - finalOracle[mismatch]) > tolerance) {
                    break;
                }
                ++mismatch;
            }
            std::cerr << "[rank " << rank << "] perf-main full-pipeline mismatch at float " << mismatch;
            if (mismatch < actual.size() && mismatch < finalOracle.size()) {
                std::cerr << " actual=" << actual[mismatch] << " oracle=" << finalOracle[mismatch];
            }
            std::cerr << '\n';
        }
    }

    for (int turn = 0; turn < worldSize; ++turn) {
        CommMpiBarrier();
        if (turn == rank) {
            std::cout << "[rank " << rank << "] perf-main full-pipeline accuracy "
                      << ((ok && pass) ? "PASS" : "FAIL") << '\n';
        }
    }
    CommMpiBarrier();

    if (!ok) {
        std::cerr << "[rank " << rank << "] perf-main failed at stage: " << failStage << '\n';
    }
    if (gmm2ToCombineQueueDev != nullptr) {
        aclrtFree(gmm2ToCombineQueueDev);
    }
    if (swiGluToGmm2QueueDev != nullptr) {
        aclrtFree(swiGluToGmm2QueueDev);
    }
    if (gmm1ToSwiGluQueueDev != nullptr) {
        aclrtFree(gmm1ToSwiGluQueueDev);
    }
    if (dispatchToGmm1QueueDev != nullptr) {
        aclrtFree(dispatchToGmm1QueueDev);
    }
    if (combineTilingDev != nullptr) {
        aclrtFree(combineTilingDev);
    }
    if (combineParamsPipelineDev != nullptr) {
        aclrtFree(combineParamsPipelineDev);
    }
    if (combineParamsRestoreDev != nullptr) {
        aclrtFree(combineParamsRestoreDev);
    }
    if (combineParamsPushDev != nullptr) {
        aclrtFree(combineParamsPushDev);
    }
    if (rowRangesDev != nullptr) {
        aclrtFree(rowRangesDev);
    }
    if (combineProbDev != nullptr) {
        aclrtFree(combineProbDev);
    }
    if (combineDoneValueDev != nullptr) {
        aclrtFree(combineDoneValueDev);
    }
    if (incomingCombineTaskDev != nullptr) {
        aclrtFree(incomingCombineTaskDev);
    }
    if (combineRangeTaskDev != nullptr) {
        aclrtFree(combineRangeTaskDev);
    }
    if (combineTaskDev != nullptr) {
        aclrtFree(combineTaskDev);
    }
    if (restoredRowsDev != nullptr) {
        aclrtFree(restoredRowsDev);
    }
    if (localPartialDev != nullptr) {
        aclrtFree(localPartialDev);
    }
    if (scale2ChannelDev != nullptr) {
        aclrtFree(scale2ChannelDev);
    }
    if (weight2Dev != nullptr) {
        aclrtFree(weight2Dev);
    }
    if (scale2Dev != nullptr) {
        aclrtFree(scale2Dev);
    }
    if (swigluQDev != nullptr) {
        aclrtFree(swigluQDev);
    }
    if (gmm1OutDev != nullptr) {
        aclrtFree(gmm1OutDev);
    }
    if (scale1ChannelDev != nullptr) {
        aclrtFree(scale1ChannelDev);
    }
    if (weight1Dev != nullptr) {
        aclrtFree(weight1Dev);
    }
    if (expertGroupRangeDev != nullptr) {
        aclrtFree(expertGroupRangeDev);
    }
    if (expertSourceSegmentDev != nullptr) {
        aclrtFree(expertSourceSegmentDev);
    }
    if (computeTileDev != nullptr) {
        aclrtFree(computeTileDev);
    }
    if (dispatchTilingDev != nullptr) {
        aclrtFree(dispatchTilingDev);
    }
    if (dispatchRangeDev != nullptr) {
        aclrtFree(dispatchRangeDev);
    }
    if (dispatchTaskDev != nullptr) {
        aclrtFree(dispatchTaskDev);
    }
    DestroyStandaloneRankRuntime(runtime);
    if (aclReady) {
        aclrtResetDevice(rank);
        aclFinalize();
    }
    CommMpiFinalize();
    return (ok && pass) ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv)
{
    const auto mode = ParseMode(argc, argv);

    if (mode.mode == "default" || mode.mode == "perf-main") {
        return RunMegaMoePerfMain(argc, argv);
    }

    std::cerr << "mode not implemented yet: " << mode.mode << '\n';
    return 2;
}
