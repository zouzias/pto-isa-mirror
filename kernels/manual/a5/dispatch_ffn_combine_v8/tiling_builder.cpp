#include "tiling_builder.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>

#include "op_kernel/utils/const_args.hpp"

namespace {
constexpr uint32_t kFrontDefaultQuantFullRowMaxK = 8192U;
constexpr uint64_t kA5UbMainReserveBytes = 32U * 1024U;
constexpr uint32_t kDispatchGatherRowsPerUbBatch = 8U;

uint64_t A5MainUbBudgetBytes()
{
    return AtlasA5::UB_SIZE > kA5UbMainReserveBytes ? AtlasA5::UB_SIZE - kA5UbMainReserveBytes : AtlasA5::UB_SIZE;
}

void RequirePositive(const char *name, uint32_t value)
{
    if (value == 0) {
        throw std::runtime_error(std::string(name) + " must be positive");
    }
}

void RequireInt8RowAligned(uint32_t k)
{
    constexpr uint32_t kDataBlockBytes = 32;
    if (k % kDataBlockBytes != 0) {
        throw std::runtime_error("K must be 32-byte aligned for int8 offsetA rows");
    }
}

void RequirePackedOffsetACapacity(const CaseConfig &cfg, const StandaloneRankRuntime &runtime)
{
    constexpr uint64_t kPackedScalePadBytes = 32;
    const uint64_t rawOffsetABytes = runtime.hccl.WindowBytes() / 3U;
    const uint64_t offsetABytes = (rawOffsetABytes + 511U) / 512U * 512U;
    const uint64_t requiredBytes = static_cast<uint64_t>(cfg.max_output_size) * (cfg.k + kPackedScalePadBytes);
    if (requiredBytes > offsetABytes) {
        throw std::runtime_error(
            "HCCL window offsetA section is too small for packed rows: windowBytes=" +
            std::to_string(runtime.hccl.WindowBytes()) + " offsetABytes=" + std::to_string(offsetABytes) +
            " requiredBytes=" + std::to_string(requiredBytes) +
            " maxOutputSize=" + std::to_string(cfg.max_output_size) + " K=" + std::to_string(cfg.k));
    }
}

void RequireDispatchTileCapacity(uint64_t dispatchTileBytes)
{
    const uint64_t maxDispatchPingPongBytes = A5MainUbBudgetBytes();
    if (dispatchTileBytes * 2U > maxDispatchPingPongBytes) {
        throw std::runtime_error("dispatch tile ping-pong buffers exceed UB budget");
    }
}

uint64_t CombineUbBytes(uint32_t tileCols)
{
    auto alignUpBytes = [](uint64_t value) { return (value + UB_ALIGN - 1U) / UB_ALIGN * UB_ALIGN; };
    const uint64_t computeStageBytes = alignUpBytes(static_cast<uint64_t>(tileCols) * sizeof(uint16_t)) +
                                       alignUpBytes(static_cast<uint64_t>(tileCols) * sizeof(float)) +
                                       alignUpBytes(static_cast<uint64_t>(tileCols) * sizeof(uint16_t));
    return computeStageBytes * 2U + UB_ALIGN;
}

void RequireCombineUbCapacity(uint32_t tileCols)
{
    if (CombineUbBytes(tileCols) > A5MainUbBudgetBytes()) {
        throw std::runtime_error("combine UB buffers exceed A5 main UB budget");
    }
}

uint32_t ChooseCombineTileCols(uint32_t k)
{
    constexpr uint32_t kMaxCombineVecTileElems = 8192U;
    constexpr uint32_t kHalfDataBlockElems = 16U;
    const uint64_t maxCombineMainUbBytes = A5MainUbBudgetBytes();
    uint32_t tileCols = std::min<uint32_t>(k, kMaxCombineVecTileElems);
    if (CombineUbBytes(tileCols) <= maxCombineMainUbBytes) {
        return tileCols;
    }
    tileCols = tileCols / kHalfDataBlockElems * kHalfDataBlockElems;
    while (tileCols > kHalfDataBlockElems && CombineUbBytes(tileCols) > maxCombineMainUbBytes) {
        tileCols -= kHalfDataBlockElems;
    }
    if (CombineUbBytes(tileCols) > maxCombineMainUbBytes) {
        throw std::runtime_error("unable to choose combine tile cols within UB budget");
    }
    return tileCols;
}

uint64_t UnpermuteUbBytes(uint32_t tokenBatch, uint32_t topk, uint32_t tileCols)
{
    auto alignUpBytes = [](uint64_t value) { return (value + UB_ALIGN - 1U) / UB_ALIGN * UB_ALIGN; };
    uint64_t ubBytes = 0;
    for (uint32_t i = 0; i < 2U; ++i) {
        ubBytes += alignUpBytes(static_cast<uint64_t>(tokenBatch) * topk * sizeof(int32_t));
        ubBytes += alignUpBytes(static_cast<uint64_t>(tokenBatch) * topk * sizeof(float));
    }
    ubBytes += alignUpBytes(static_cast<uint64_t>(tileCols) * sizeof(float));
    for (uint32_t i = 0; i < 2U; ++i) {
        ubBytes += alignUpBytes(static_cast<uint64_t>(tileCols) * sizeof(uint16_t));
        ubBytes += alignUpBytes(static_cast<uint64_t>(tileCols) * sizeof(float));
    }
    ubBytes += alignUpBytes(static_cast<uint64_t>(tileCols) * sizeof(uint16_t));
    return ubBytes;
}

void RequireUnpermuteUbCapacity(uint32_t tokenBatch, uint32_t topk, uint32_t tileCols)
{
    if (UnpermuteUbBytes(tokenBatch, topk, tileCols) > A5MainUbBudgetBytes()) {
        throw std::runtime_error("unpermute UB buffers exceed A5 main UB budget");
    }
}

uint32_t ChooseUnpermuteTileCols(uint32_t k, uint32_t tokenBatch, uint32_t topk)
{
    constexpr uint32_t kMaxUnpermuteVecTileElems = 8192U;
    constexpr uint32_t kHalfDataBlockElems = 16U;
    const uint64_t maxUnpermuteMainUbBytes = A5MainUbBudgetBytes();
    uint32_t tileCols = std::min<uint32_t>(k, kMaxUnpermuteVecTileElems);
    tileCols = tileCols / kHalfDataBlockElems * kHalfDataBlockElems;
    while (tileCols > kHalfDataBlockElems &&
           UnpermuteUbBytes(tokenBatch, topk, tileCols) > maxUnpermuteMainUbBytes) {
        tileCols -= kHalfDataBlockElems;
    }
    if (UnpermuteUbBytes(tokenBatch, topk, tileCols) > maxUnpermuteMainUbBytes) {
        throw std::runtime_error("unable to choose unpermute tile cols within UB budget");
    }
    return tileCols;
}

uint32_t CalcMixAic1To2BlockDim(uint32_t aic_num, uint32_t aiv_num)
{
    if (aiv_num > aic_num * 2U) {
        throw std::runtime_error("direct mixed launch expects aiv_num <= aic_num * 2");
    }
    return aic_num;
}

uint32_t CalcInitRoutingAivNum(uint32_t aiv_num)
{
    return aiv_num;
}

uint64_t AlignUp(uint64_t value, uint64_t align)
{
    return (value + align - 1) / align * align;
}

uint32_t DivCeil(uint32_t value, uint32_t divisor)
{
    return divisor == 0U ? 0U : (value + divisor - 1U) / divisor;
}

uint32_t Pow4Ceil(uint32_t value)
{
    if (value <= 1U) {
        return 1U;
    }
    uint32_t out = 1U;
    while (out < value && out <= UINT32_MAX / 4U) {
        out *= 4U;
    }
    return out;
}

constexpr uint32_t kFrontCaseFullLoadDynamic = 21000U;
constexpr uint32_t kFrontCaseOneCoreDynamic = 11000U;
constexpr uint32_t kFrontCaseMultiCoreDynamic = 11010U;
constexpr uint32_t kFrontMaxColsOneLoopQuant = 8192U;
constexpr uint32_t kFrontSortAlignElems = 32U;
constexpr uint32_t kFrontMrgListNum = 4U;
constexpr uint32_t kFrontSortOutLoopMaxElems = 2040U;

void PopulateFrontSortLoopFields(DispatchFFNCombineFrontReorderTiling &front)
{
    if (front.sortNeedCoreNum == 0U || front.sortPerCoreElems == 0U || front.sortLastCoreElems == 0U ||
        front.sortLoopMaxElement == 0U) {
        return;
    }
    front.sortPerCoreLoops = DivCeil(front.sortPerCoreElems, front.sortLoopMaxElement);
    front.sortPerCorePerLoopElems = std::min(front.sortPerCoreElems, front.sortLoopMaxElement);
    front.sortPerCoreLastLoopElems =
        front.sortPerCoreElems - (front.sortPerCoreLoops - 1U) * front.sortPerCorePerLoopElems;
    front.sortLastCoreLoops = front.sortPerCoreLoops;
    const uint32_t lastCoreAvgElems = DivCeil(front.sortLastCoreElems, front.sortLastCoreLoops);
    front.sortLastCorePerLoopElems = DivCeil(lastCoreAvgElems, kFrontSortAlignElems) * kFrontSortAlignElems;
    const uint64_t lastCoreLoopConsumed =
        static_cast<uint64_t>(front.sortLastCoreLoops - 1U) * front.sortLastCorePerLoopElems;
    front.sortLastCoreLastLoopElems = lastCoreLoopConsumed >= front.sortLastCoreElems ?
                                          0U :
                                          static_cast<uint32_t>(front.sortLastCoreElems - lastCoreLoopConsumed);
}

uint32_t SmallRouteLimitFromReference()
{
    constexpr uint32_t kSort32AlignElement = 32U;
    constexpr uint32_t kFrontSortBytesPerRouteElem = sizeof(int32_t) * 2U * 4U;
    return static_cast<uint32_t>(AtlasA5::UB_SIZE / kFrontSortBytesPerRouteElem / kSort32AlignElement *
                                 kSort32AlignElement);
}

uint32_t SmallFrontAlignedRouteElems(uint32_t routeElems)
{
    constexpr uint64_t kFrontRouteAlign = 128U;
    return static_cast<uint32_t>(AlignUp(routeElems, kFrontRouteAlign));
}

uint64_t FullLoadDynamicUbBudgetBytes(uint32_t routeElems, uint32_t k, uint32_t expertNumAligned)
{
    const uint64_t alignedRouteElems = SmallFrontAlignedRouteElems(routeElems);
    const uint64_t routeBytes = AlignUp(alignedRouteElems * sizeof(int32_t), UB_ALIGN);
    const uint64_t packedSortBytes = AlignUp(alignedRouteElems * 2U * sizeof(float), UB_ALIGN);
    const uint64_t tableBytes = AlignUp(static_cast<uint64_t>(expertNumAligned) * sizeof(int32_t), UB_ALIGN);
    const uint64_t rawBytes = AlignUp(static_cast<uint64_t>(k) * sizeof(uint16_t), UB_ALIGN);
    const uint64_t fp32Bytes = AlignUp(static_cast<uint64_t>(k) * sizeof(float), UB_ALIGN);
    const uint64_t tmpBytes = AlignUp(static_cast<uint64_t>(k) * sizeof(float), UB_ALIGN);
    const uint64_t outBytes = AlignUp(static_cast<uint64_t>(k) * sizeof(int8_t), UB_ALIGN);
    const uint64_t scaleBytes = AlignUp(sizeof(float), UB_ALIGN);
    constexpr uint64_t kFrontRouteBufferCount = 6U; // expert, payload, dstToSrc, expanded, sortedKey, keyScratch.
    return kFrontRouteBufferCount * routeBytes + 2U * packedSortBytes + 2U * tableBytes + rawBytes + fp32Bytes +
           tmpBytes + outBytes + scaleBytes;
}

uint64_t Mc2FullLoadDynamicUbBudgetBytes(uint32_t routeElems, uint32_t k, uint32_t expertNum)
{
    constexpr uint64_t kOneCoreSortBuffer = 6U;
    constexpr uint64_t kOtherRouteBuffer = 3U;
    constexpr uint64_t kDynamicQuantFullLoadColsBuffer = 13U;
    constexpr uint64_t kScaleOutBytes = 64U;
    const uint64_t alignedRouteElems = AlignUp(routeElems, UB_ALIGN);
    const uint64_t sortSpace = alignedRouteElems * sizeof(int32_t) * kOneCoreSortBuffer;
    const uint64_t otherSpace = alignedRouteElems * sizeof(int32_t) * kOtherRouteBuffer;
    const uint64_t expertSpace = AlignUp(static_cast<uint64_t>(expertNum) * sizeof(int32_t), UB_ALIGN);
    const uint64_t quantSpace = AlignUp(static_cast<uint64_t>(k), UB_ALIGN) * kDynamicQuantFullLoadColsBuffer;
    return sortSpace + otherSpace + expertSpace + quantSpace + kScaleOutBytes;
}

uint64_t SmallFrontUbBytes(const CaseConfig &cfg, uint32_t expertNumAligned)
{
    const uint64_t tableBytes = AlignUp(static_cast<uint64_t>(expertNumAligned) * sizeof(int32_t), UB_ALIGN);
    const uint64_t routeElems = static_cast<uint64_t>(cfg.m) * cfg.topk;
    const uint64_t routeBytes = AlignUp(routeElems * sizeof(int32_t), UB_ALIGN);
    const uint64_t routeStageBytes = 5U * tableBytes + 2U * routeBytes;

    const uint64_t inputElemBytes = sizeof(uint16_t);
    const uint64_t rawBytes = AlignUp(static_cast<uint64_t>(cfg.k) * inputElemBytes, UB_ALIGN);
    const uint64_t fp32Bytes = AlignUp(static_cast<uint64_t>(cfg.k) * sizeof(float), UB_ALIGN);
    const uint64_t tmpBytes = AlignUp(static_cast<uint64_t>(cfg.k) * sizeof(float), UB_ALIGN);
    const uint64_t outBytes = AlignUp(static_cast<uint64_t>(cfg.k) * sizeof(int8_t), UB_ALIGN);
    const uint64_t scaleBytes = AlignUp(sizeof(float), UB_ALIGN);
    const uint64_t scatterStageBytes = 5U * tableBytes + rawBytes + fp32Bytes + tmpBytes + outBytes + scaleBytes;

    return std::max(routeStageBytes, scatterStageBytes);
}

uint64_t SmallFrontUbBudgetBytesForPackedSort(const CaseConfig &cfg, uint32_t expertNumAligned)
{
    return FullLoadDynamicUbBudgetBytes(cfg.m * cfg.topk, cfg.k, expertNumAligned);
}

uint64_t SmallFrontDebugBytesPerWorker(uint32_t alignedRouteElems, uint32_t expertNumAligned)
{
    const uint64_t debugElems = 3U * static_cast<uint64_t>(alignedRouteElems) + 2U * expertNumAligned;
    return AlignUp(debugElems * sizeof(int32_t), 512U);
}

uint64_t FrontCheckDebugBytesPerWorker()
{
    return AlignUp(sizeof(DispatchFFNCombineFrontCheckLayoutDebug), 512U);
}

uint64_t FrontDebugBytesPerWorker()
{
    return AlignUp(sizeof(DispatchFFNCombineFrontLayoutDebug), 512U);
}

uint64_t FrontDebugBytes()
{
    return AlignUp(sizeof(DispatchFFNCombineFrontLayoutDebug), 512U);
}

uint64_t FrontRouteWorkspaceBytes(uint32_t alignedRouteElems)
{
    const uint64_t sortedIntBytes = AlignUp(static_cast<uint64_t>(alignedRouteElems) * 2U * sizeof(int32_t), 512U);
    const uint64_t packedRunBytes = AlignUp(static_cast<uint64_t>(alignedRouteElems) * 2U * sizeof(float), 512U);
    return std::max(sortedIntBytes, packedRunBytes);
}

uint64_t FrontQuantTmpBytes(const CaseConfig &cfg)
{
    const uint64_t rawBytes = AlignUp(static_cast<uint64_t>(cfg.k) * sizeof(uint16_t), 512U);
    const uint64_t fp32Bytes = AlignUp(static_cast<uint64_t>(cfg.k) * sizeof(float), 512U);
    const uint64_t tmpBytes = AlignUp(static_cast<uint64_t>(cfg.k) * sizeof(float), 512U);
    const uint64_t outBytes = AlignUp(static_cast<uint64_t>(cfg.k) * sizeof(int8_t), 512U);
    const uint64_t scaleBytes = AlignUp(8U * sizeof(float), 512U);
    return rawBytes + fp32Bytes + tmpBytes + outBytes + scaleBytes;
}

bool IsFrontFullLoadDynamic(const CaseConfig &cfg, uint32_t expertNumAligned, uint32_t routeElems,
                            uint32_t sortLoopMaxElement)
{
    (void)expertNumAligned;
    if (routeElems == 0U || routeElems > sortLoopMaxElement || cfg.k > kFrontMaxColsOneLoopQuant ||
        cfg.k % UB_ALIGN != 0U || cfg.topk > 8U) {
        return false;
    }
    const uint32_t expertNum = cfg.world_size * cfg.expert_per_rank;
    return Mc2FullLoadDynamicUbBudgetBytes(routeElems, cfg.k, expertNum) <= AtlasA5::UB_SIZE;
}

uint32_t CalcFrontFullLoadMaxRouteElems(const CaseConfig &cfg, uint32_t expertNumAligned, uint32_t sortLoopMaxElement)
{
    uint32_t maxRouteElems = 0;
    for (uint32_t routeElems = 32U; routeElems <= sortLoopMaxElement; routeElems += 32U) {
        if (IsFrontFullLoadDynamic(cfg, expertNumAligned, routeElems, sortLoopMaxElement)) {
            maxRouteElems = routeElems;
        }
    }
    return maxRouteElems;
}

uint32_t SelectFrontCase(const CaseConfig &cfg, uint32_t expertNumAligned, uint32_t routeElems,
                         uint32_t sortLoopMaxElement)
{
    if (routeElems == 0U) {
        return 0U;
    }
    if (IsFrontFullLoadDynamic(cfg, expertNumAligned, routeElems, sortLoopMaxElement)) {
        return kFrontCaseFullLoadDynamic;
    }
    if (routeElems <= sortLoopMaxElement) {
        return kFrontCaseOneCoreDynamic;
    }
    return kFrontCaseMultiCoreDynamic;
}

void PopulateFrontSortSplit(DispatchFFNCombineFrontReorderTiling &front, DispatchFFNCombineFrontDebugTiling &frontDebug,
                            uint32_t aivNum)
{
    frontDebug.frontMode = static_cast<uint16_t>(1U);
    front.sortOutLoopMaxElems = static_cast<uint16_t>(kFrontSortOutLoopMaxElems);
    if (front.routeElems == 0U) {
        return;
    }
    if (front.frontCase != kFrontCaseMultiCoreDynamic || aivNum == 0U) {
        front.sortNeedCoreNum = static_cast<uint16_t>(1U);
        front.sortPerCoreElems = front.routeElems;
        front.sortLastCoreElems = front.routeElems;
        PopulateFrontSortLoopFields(front);
        front.sortVmsMiddleNeedCoreNum = 0U;
        return;
    }

    uint32_t needCoreNum = DivCeil(front.routeElems, front.sortLoopMaxElement);
    needCoreNum = std::min<uint32_t>(Pow4Ceil(needCoreNum), aivNum);
    if (needCoreNum == 0U) {
        return;
    }

    uint32_t perCoreElems = front.routeElems / needCoreNum;
    uint32_t floorPerCoreElems = perCoreElems - perCoreElems % kFrontSortAlignElems;
    if (floorPerCoreElems == 0U) {
        floorPerCoreElems = kFrontSortAlignElems;
    }
    const uint32_t lastCoreElemsWithFloor = front.routeElems - (needCoreNum - 1U) * floorPerCoreElems;
    uint32_t ceilPerCoreElems = perCoreElems + kFrontSortAlignElems - perCoreElems % kFrontSortAlignElems;
    if (perCoreElems % kFrontSortAlignElems == 0U) {
        // reference uses the next aligned block for the ceil-side capacity probe.
        ceilPerCoreElems = perCoreElems + kFrontSortAlignElems;
    }
    if (lastCoreElemsWithFloor > ceilPerCoreElems) {
        perCoreElems = ceilPerCoreElems;
        needCoreNum = DivCeil(front.routeElems, perCoreElems);
    } else {
        perCoreElems = floorPerCoreElems;
    }

    do {
        front.sortNeedCoreNum = static_cast<uint16_t>(needCoreNum);
        front.sortPerCoreElems = perCoreElems;
        front.sortLastCoreElems = front.routeElems - (front.sortNeedCoreNum - 1U) * front.sortPerCoreElems;
        PopulateFrontSortLoopFields(front);
        if (front.sortLastCoreLastLoopElems != 0U || perCoreElems <= kFrontSortAlignElems) {
            break;
        }
        perCoreElems -= kFrontSortAlignElems;
    } while (true);

    front.sortVmsMiddleNeedCoreNum = static_cast<uint16_t>(
        front.sortNeedCoreNum <= kFrontMrgListNum ? 0U : DivCeil(front.sortNeedCoreNum, kFrontMrgListNum));
}

void RequireAlignedRange(const char *name, uint64_t offset, uint64_t bytes)
{
    if (offset % 512U != 0U || bytes % 512U != 0U) {
        throw std::runtime_error(std::string(name) + " must be 512-byte aligned");
    }
}

bool IsSmallFrontPath(const CaseConfig &cfg, uint32_t expertNumAligned, uint32_t &routeLimit, uint64_t &smallUbBytes)
{
    constexpr uint32_t kSmallFullRowMaxK = 8192U;
    routeLimit = SmallRouteLimitFromReference();
    const uint64_t routeElems = static_cast<uint64_t>(cfg.m) * cfg.topk;
    smallUbBytes = SmallFrontUbBytes(cfg, expertNumAligned);
    if (routeElems == 0 || routeElems > routeLimit) {
        return false;
    }
    if (cfg.k > kSmallFullRowMaxK || cfg.k % UB_ALIGN != 0 || cfg.topk > 8U) {
        return false;
    }
    if (smallUbBytes > AtlasA5::UB_SIZE) {
        return false;
    }
    if (SmallFrontUbBudgetBytesForPackedSort(cfg, expertNumAligned) > AtlasA5::UB_SIZE) {
        return false;
    }
    return true;
}

uint32_t SwigluSegmentNum(uint32_t expert_per_rank)
{
    return expert_per_rank <= 1U ? expert_per_rank : 2U;
}

uint32_t SwigluEpilogueGranularity(uint32_t expert_per_rank)
{
    if (expert_per_rank <= 1U) {
        return expert_per_rank;
    }
    return expert_per_rank <= 4U ? expert_per_rank - 1U : expert_per_rank - 3U;
}

} // namespace

DispatchFFNCombineBuildResult BuildDispatchFFNCombineTiling(const CaseConfig &cfg, const StandaloneRankRuntime &runtime)
{
    RequirePositive("aic_num", cfg.aic_num);
    RequirePositive("aiv_num", cfg.aiv_num);
    RequireInt8RowAligned(cfg.k);
    RequirePackedOffsetACapacity(cfg, runtime);

    DispatchFFNCombineBuildResult result;
    result.block_dim = CalcMixAic1To2BlockDim(cfg.aic_num, cfg.aiv_num);

    auto &info = result.tiling.dispatchFFNCombineInfo;
    info.M = cfg.m;
    info.K = cfg.k;
    info.N = cfg.n;
    info.topK = cfg.topk;
    info.expertPerRank = cfg.expert_per_rank;
    info.worldSize = cfg.world_size;
    info.maxOutputSize = cfg.max_output_size;
    info.listLen = cfg.list_len;
    info.aivNum = cfg.aiv_num;
    info.ubMoveNum = 16 * 1024;

    result.tiling.runtimeInfo.remoteWindowContext = reinterpret_cast<uint64_t>(runtime.hccl.RemoteWindowContextPtr());
    result.tiling.runtimeInfo.rank = static_cast<uint32_t>(runtime.hccl.rank_id);
    result.tiling.runtimeInfo.rankSize = static_cast<uint32_t>(runtime.hccl.world_size);

    auto &front = result.tiling.frontReorderTiling;
    auto &frontDebug = result.tiling.frontDebugTiling;
    const uint64_t expertNum = static_cast<uint64_t>(cfg.world_size) * cfg.expert_per_rank;
    front.expertNum = static_cast<uint32_t>(expertNum);
    front.expertNumAligned = static_cast<uint32_t>(AlignUp(expertNum + 1U, 128U));
    front.stageNum = static_cast<uint16_t>(14U);
    uint32_t smallRouteLimit = 0;
    uint64_t smallFrontUbBytes = 0;
    const bool smallFrontPath = IsSmallFrontPath(cfg, front.expertNumAligned, smallRouteLimit, smallFrontUbBytes);
    front.frontPath = static_cast<uint16_t>(smallFrontPath ? 1U : 0U);
    front.smallRouteElems = static_cast<uint32_t>(static_cast<uint64_t>(cfg.m) * cfg.topk);
    front.smallRouteLimit = smallRouteLimit;
    front.smallFrontUbBytes = static_cast<uint32_t>(smallFrontUbBytes);
    front.routeElems = static_cast<uint32_t>(static_cast<uint64_t>(cfg.m) * cfg.topk);
    front.alignedRouteElems = SmallFrontAlignedRouteElems(front.routeElems);
    front.sortLoopMaxElement = smallRouteLimit;
    front.fullLoadMaxRouteElems = CalcFrontFullLoadMaxRouteElems(cfg, front.expertNumAligned, front.sortLoopMaxElement);
    front.frontCase =
        static_cast<uint16_t>(SelectFrontCase(cfg, front.expertNumAligned, front.routeElems, front.sortLoopMaxElement));
    PopulateFrontSortSplit(front, frontDebug, cfg.aiv_num);
    frontDebug.frontCheckDebugBytesPerWorker = static_cast<uint32_t>(FrontCheckDebugBytesPerWorker());
    frontDebug.frontDebugBytesPerWorker = static_cast<uint32_t>(FrontDebugBytesPerWorker());
    if (smallFrontPath && front.smallRouteElems != 0U) {
        front.smallFrontRouteElems = front.smallRouteElems;
        front.smallFrontAlignedRouteElems = SmallFrontAlignedRouteElems(front.smallFrontRouteElems);
        front.smallFrontSortUbBytes =
            static_cast<uint32_t>(SmallFrontUbBudgetBytesForPackedSort(cfg, front.expertNumAligned));
        if (cfg.aiv_num != 0U) {
            front.smallFrontPerCoreRoutes = DivCeil(front.smallFrontRouteElems, cfg.aiv_num);
            frontDebug.smallFrontNeedCoreNum =
                static_cast<uint16_t>(DivCeil(front.smallFrontRouteElems, front.smallFrontPerCoreRoutes));
            front.smallFrontLastCoreRoutes =
                front.smallFrontRouteElems - front.smallFrontPerCoreRoutes * (frontDebug.smallFrontNeedCoreNum - 1U);
        }
        frontDebug.smallFrontDebugBytesPerWorker = static_cast<uint32_t>(
            SmallFrontDebugBytesPerWorker(front.smallFrontAlignedRouteElems, front.expertNumAligned));
    }
    front.expandedRowIdxOffset = 0;

    const uint32_t n2 = cfg.k;
    const uint32_t k2 = cfg.n / 2;
    const uint64_t expandedRowIdxBytes = ((cfg.m + 255) / 256) * 256 * cfg.topk * sizeof(int32_t);
    uint64_t frontWorkspaceOffset = expandedRowIdxBytes;
    front.coreCountOffset = static_cast<uint32_t>(frontWorkspaceOffset);
    frontWorkspaceOffset += static_cast<uint64_t>(cfg.aiv_num) * front.expertNumAligned * sizeof(int32_t);
    front.localTokenPerExpertOffset = static_cast<uint32_t>(frontWorkspaceOffset);
    frontWorkspaceOffset += static_cast<uint64_t>(front.expertNumAligned) * sizeof(int32_t);
    front.expertBaseOffset = static_cast<uint32_t>(frontWorkspaceOffset);
    frontWorkspaceOffset += static_cast<uint64_t>(front.expertNumAligned) * sizeof(int32_t);
    front.coreBaseOffset = static_cast<uint32_t>(frontWorkspaceOffset);
    frontWorkspaceOffset += static_cast<uint64_t>(cfg.aiv_num) * front.expertNumAligned * sizeof(int32_t);
    front.cumsumMMOffset = static_cast<uint32_t>(frontWorkspaceOffset);
    frontWorkspaceOffset += static_cast<uint64_t>(cfg.world_size) * cfg.expert_per_rank * sizeof(int32_t);
    front.preSumBeforeRankOffset = static_cast<uint32_t>(frontWorkspaceOffset);
    frontWorkspaceOffset += static_cast<uint64_t>(cfg.world_size) * cfg.expert_per_rank * sizeof(int32_t);
    if (front.routeElems != 0U) {
        frontWorkspaceOffset = AlignUp(frontWorkspaceOffset, 512U);
        front.frontExpandedExpertOffset = static_cast<uint32_t>(frontWorkspaceOffset);
        frontWorkspaceOffset += AlignUp(static_cast<uint64_t>(front.alignedRouteElems) * sizeof(int32_t), 512U);

        frontWorkspaceOffset = AlignUp(frontWorkspaceOffset, 512U);
        front.frontExpandDstToSrcOffset = static_cast<uint32_t>(frontWorkspaceOffset);
        frontWorkspaceOffset += AlignUp(static_cast<uint64_t>(front.alignedRouteElems) * sizeof(int32_t), 512U);

        frontWorkspaceOffset = AlignUp(frontWorkspaceOffset, 512U);
        front.frontSortWs0Offset = static_cast<uint32_t>(frontWorkspaceOffset);
        frontWorkspaceOffset += FrontRouteWorkspaceBytes(front.alignedRouteElems);

        frontWorkspaceOffset = AlignUp(frontWorkspaceOffset, 512U);
        front.frontSortWs1Offset = static_cast<uint32_t>(frontWorkspaceOffset);
        frontWorkspaceOffset += FrontRouteWorkspaceBytes(front.alignedRouteElems);

        frontWorkspaceOffset = AlignUp(frontWorkspaceOffset, 512U);
        front.frontQuantTmpOffset = static_cast<uint32_t>(frontWorkspaceOffset);
        front.frontQuantTmpBytes = static_cast<uint32_t>(FrontQuantTmpBytes(cfg));
        frontWorkspaceOffset += front.frontQuantTmpBytes;

        frontWorkspaceOffset = AlignUp(frontWorkspaceOffset, 512U);
        frontDebug.frontDebugOffset = static_cast<uint32_t>(frontWorkspaceOffset);
        frontDebug.frontDebugBytes = static_cast<uint32_t>(FrontDebugBytes());
        frontWorkspaceOffset += frontDebug.frontDebugBytes;

        RequireAlignedRange("frontExpandedExpert", front.frontExpandedExpertOffset,
                            AlignUp(static_cast<uint64_t>(front.alignedRouteElems) * sizeof(int32_t), 512U));
        RequireAlignedRange("frontExpandDstToSrc", front.frontExpandDstToSrcOffset,
                            AlignUp(static_cast<uint64_t>(front.alignedRouteElems) * sizeof(int32_t), 512U));
        RequireAlignedRange("frontSortWs0", front.frontSortWs0Offset,
                            FrontRouteWorkspaceBytes(front.alignedRouteElems));
        RequireAlignedRange("frontSortWs1", front.frontSortWs1Offset,
                            FrontRouteWorkspaceBytes(front.alignedRouteElems));
        RequireAlignedRange("frontQuantTmp", front.frontQuantTmpOffset, front.frontQuantTmpBytes);
        RequireAlignedRange("frontDebug", frontDebug.frontDebugOffset, frontDebug.frontDebugBytes);
    }
    if (smallFrontPath && frontDebug.smallFrontNeedCoreNum != 0U && frontDebug.smallFrontDebugBytesPerWorker != 0U) {
        frontWorkspaceOffset = AlignUp(frontWorkspaceOffset, 512U);
        frontDebug.smallFrontDebugOffset = static_cast<uint32_t>(frontWorkspaceOffset);
        frontDebug.smallFrontDebugBytes = static_cast<uint32_t>(AlignUp(
            static_cast<uint64_t>(frontDebug.smallFrontDebugBytesPerWorker) * frontDebug.smallFrontNeedCoreNum, 512U));
        frontWorkspaceOffset += frontDebug.smallFrontDebugBytes;
    }
    if (front.routeElems != 0U) {
        frontWorkspaceOffset = AlignUp(frontWorkspaceOffset, 512U);
        frontDebug.frontSortCheckOffset = static_cast<uint32_t>(frontWorkspaceOffset);
        frontDebug.frontSortCheckBytes = static_cast<uint32_t>(FrontRouteWorkspaceBytes(front.alignedRouteElems));
        RequireAlignedRange("frontSortCheck", frontDebug.frontSortCheckOffset, frontDebug.frontSortCheckBytes);
        frontWorkspaceOffset += frontDebug.frontSortCheckBytes;

        frontWorkspaceOffset = AlignUp(frontWorkspaceOffset, 512U);
        frontDebug.frontMergeCheckOffset = static_cast<uint32_t>(frontWorkspaceOffset);
        frontDebug.frontMergeCheckBytes = static_cast<uint32_t>(FrontRouteWorkspaceBytes(front.alignedRouteElems));
        RequireAlignedRange("frontMergeCheck", frontDebug.frontMergeCheckOffset, frontDebug.frontMergeCheckBytes);
        frontWorkspaceOffset += frontDebug.frontMergeCheckBytes;

        frontWorkspaceOffset = AlignUp(frontWorkspaceOffset, 512U);
        front.frontCountScratchOffset = static_cast<uint32_t>(frontWorkspaceOffset);
        const uint64_t frontEndCountScratchRows = std::min<uint64_t>(cfg.aiv_num, cfg.world_size);
        front.frontCountScratchBytes = static_cast<uint32_t>(
            AlignUp(static_cast<uint64_t>(front.expertNumAligned) * sizeof(int32_t), 512U) * frontEndCountScratchRows);
        RequireAlignedRange("frontCountScratch", front.frontCountScratchOffset, front.frontCountScratchBytes);
        frontWorkspaceOffset += front.frontCountScratchBytes;

        frontWorkspaceOffset = AlignUp(frontWorkspaceOffset, 512U);
        frontDebug.frontCheckDebugOffset = static_cast<uint32_t>(frontWorkspaceOffset);
        frontDebug.frontCheckDebugBytes = frontDebug.frontCheckDebugBytesPerWorker;
        RequireAlignedRange("frontCheckDebug", frontDebug.frontCheckDebugOffset, frontDebug.frontCheckDebugBytes);
        frontWorkspaceOffset += frontDebug.frontCheckDebugBytes;
    }
    front.frontWorkspaceBytes = static_cast<uint32_t>(AlignUp(frontWorkspaceOffset, 512U));

    auto &dispatch = result.tiling.dispatchTiling;
    uint64_t dispatchWorkspaceOffset = front.frontWorkspaceBytes;
    dispatch.perTokenScaleOffset = dispatchWorkspaceOffset;
    dispatchWorkspaceOffset += static_cast<uint64_t>(cfg.max_output_size) * sizeof(float);
    dispatchWorkspaceOffset = AlignUp(dispatchWorkspaceOffset, 512U);

    auto &swiglu = result.tiling.swigluTiling;
    swiglu.perTokenScale2Offset = dispatchWorkspaceOffset;
    dispatchWorkspaceOffset += static_cast<uint64_t>(cfg.max_output_size) * sizeof(float);
    dispatchWorkspaceOffset = AlignUp(dispatchWorkspaceOffset, 512U);

    dispatch.gmAOffset = dispatchWorkspaceOffset;
    dispatchWorkspaceOffset += static_cast<uint64_t>(cfg.max_output_size) * cfg.k * sizeof(int8_t);
    dispatchWorkspaceOffset = AlignUp(dispatchWorkspaceOffset, 512U);

    swiglu.gmPermutedTokenOffset = dispatchWorkspaceOffset;
    dispatchWorkspaceOffset += static_cast<uint64_t>(cfg.max_output_size) * k2 * sizeof(int8_t);
    dispatchWorkspaceOffset = AlignUp(dispatchWorkspaceOffset, 512U);
    swiglu.swigluTileElems = 1024U;
    swiglu.swigluSegmentNum = SwigluSegmentNum(cfg.expert_per_rank);
    swiglu.swigluEpilogueGranularity = SwigluEpilogueGranularity(cfg.expert_per_rank);
    swiglu.swigluUbStages = 2U;
    swiglu.swigluMetadataMode = 1U;

    auto &gmm1 = result.tiling.gmm1Tiling;
    gmm1.gmCOffset = dispatchWorkspaceOffset;
    dispatchWorkspaceOffset += static_cast<uint64_t>(cfg.max_output_size) * cfg.n * sizeof(int16_t);
    dispatchWorkspaceOffset = AlignUp(dispatchWorkspaceOffset, 512U);

    auto &gmm2 = result.tiling.gmm2Tiling;
    gmm2.gmm2OutputOffset = dispatchWorkspaceOffset;
    dispatchWorkspaceOffset += static_cast<uint64_t>(cfg.max_output_size) * n2 * sizeof(int16_t);
    dispatchWorkspaceOffset = AlignUp(dispatchWorkspaceOffset, 512U);
    gmm2.l1TileM = gmm1.l1TileM;
    gmm2.l1TileN = gmm1.l1TileN;
    gmm2.l1TileK = gmm1.l1TileK;
    gmm2.l0TileM = gmm1.l0TileM;
    gmm2.l0TileN = gmm1.l0TileN;
    gmm2.l0TileK = gmm1.l0TileK;

    auto &combine = result.tiling.combineTiling;
    combine.gmm2OutputOffset = gmm2.gmm2OutputOffset;
    combine.perTokenScale2Offset = swiglu.perTokenScale2Offset;
    combine.combineTileCols = ChooseCombineTileCols(cfg.k);
    combine.combineScratchOffset = 0U;
    combine.combineScratchBytes = 0U;
    combine.combineScratchBytesPerAiv = 0U;
    RequireCombineUbCapacity(combine.combineTileCols);

    dispatch.dispatchTileBytes =
        std::max<uint64_t>(32U * 1024U,
                           AlignUp(static_cast<uint64_t>(kDispatchGatherRowsPerUbBatch) * (cfg.k + 32U), 32U));
    RequireDispatchTileCapacity(dispatch.dispatchTileBytes);
    dispatch.dispatchScratchBytesPerAiv = AlignUp(dispatch.dispatchTileBytes * 2U, 512U);
    dispatch.dispatchScratchOffset = dispatchWorkspaceOffset;
    dispatch.dispatchScratchBytes = static_cast<uint64_t>(cfg.aiv_num) * dispatch.dispatchScratchBytesPerAiv;
    dispatchWorkspaceOffset += dispatch.dispatchScratchBytes;
    dispatchWorkspaceOffset = AlignUp(dispatchWorkspaceOffset, 512U);

    dispatch.dispatchDebugOffset = dispatchWorkspaceOffset;
    const uint64_t dispatchMetadataDebugBytes =
        static_cast<uint64_t>(cfg.world_size) * cfg.expert_per_rank * sizeof(DispatchFFNCombineDispatchMetadataDebug);
    const uint64_t dispatchTaskStatsDebugBytes =
        static_cast<uint64_t>(cfg.aiv_num) * cfg.expert_per_rank * sizeof(DispatchFFNCombineDispatchTaskStatsDebug);
    const uint64_t dispatchGroupDoneDebugBytes =
        static_cast<uint64_t>(cfg.expert_per_rank) * sizeof(DispatchFFNCombineDispatchGroupDoneDebug);
    dispatch.dispatchDebugBytes = AlignUp(sizeof(DispatchFFNCombineDispatchLayoutDebug) + dispatchMetadataDebugBytes +
                                              dispatchTaskStatsDebugBytes + dispatchGroupDoneDebugBytes,
                                          512U);
    dispatchWorkspaceOffset += dispatch.dispatchDebugBytes;
    dispatchWorkspaceOffset = AlignUp(dispatchWorkspaceOffset, 512U);

    dispatch.dispatchGatherTileBytes = dispatch.dispatchTileBytes;
    dispatch.dispatchGatherScratchBytesPerAiv = dispatch.dispatchScratchBytesPerAiv;
    dispatch.dispatchGatherScratchOffset = dispatchWorkspaceOffset;
    dispatch.dispatchGatherScratchBytes =
        static_cast<uint64_t>(cfg.aiv_num) * dispatch.dispatchGatherScratchBytesPerAiv;
    RequireAlignedRange("dispatchGatherScratch", dispatch.dispatchGatherScratchOffset,
                        dispatch.dispatchGatherScratchBytes);
    dispatchWorkspaceOffset += dispatch.dispatchGatherScratchBytes;
    dispatchWorkspaceOffset = AlignUp(dispatchWorkspaceOffset, 512U);

    dispatch.dispatchGatherDebugOffset = dispatchWorkspaceOffset;
    const uint64_t dispatchGatherMetadataDebugBytes =
        static_cast<uint64_t>(cfg.world_size) * cfg.expert_per_rank * sizeof(DispatchFFNCombineDispatchMetadataDebug);
    dispatch.dispatchGatherDebugBytes = AlignUp(
        AlignUp(sizeof(DispatchFFNCombineDispatchGatherLayoutDebug), 512U) + dispatchGatherMetadataDebugBytes, 512U);
    RequireAlignedRange("dispatchGatherDebug", dispatch.dispatchGatherDebugOffset, dispatch.dispatchGatherDebugBytes);
    dispatchWorkspaceOffset += dispatch.dispatchGatherDebugBytes;
    dispatchWorkspaceOffset = AlignUp(dispatchWorkspaceOffset, 512U);

    gmm1.gmm1DebugOffset = dispatchWorkspaceOffset;
    const uint64_t gmm1SyncDebugBytes =
        static_cast<uint64_t>(cfg.expert_per_rank) * cfg.aic_num * sizeof(DispatchFFNCombineGmm1SyncDebug);
    const uint64_t gmm1TaskDebugBytes =
        static_cast<uint64_t>(cfg.expert_per_rank) * cfg.aic_num * sizeof(DispatchFFNCombineGmm1TaskDebug);
    const uint32_t gmm1SegmentNum = SwigluSegmentNum(cfg.expert_per_rank);
    const uint64_t gmm1DoneDebugBytes =
        static_cast<uint64_t>(gmm1SegmentNum) * cfg.aic_num * sizeof(DispatchFFNCombineGmm1DoneDebug);
    gmm1.gmm1DebugBytes = AlignUp(
        sizeof(DispatchFFNCombineGmm1LayoutDebug) + gmm1SyncDebugBytes + gmm1TaskDebugBytes + gmm1DoneDebugBytes, 512U);
    dispatchWorkspaceOffset += gmm1.gmm1DebugBytes;
    dispatchWorkspaceOffset = AlignUp(dispatchWorkspaceOffset, 512U);

    swiglu.swigluSegmentMetaOffset = dispatchWorkspaceOffset;
    swiglu.swigluSegmentMetaBytes = AlignUp(
        static_cast<uint64_t>(swiglu.swigluSegmentNum) * sizeof(DispatchFFNCombineSwigluSegmentRuntimeMeta), 512U);
    dispatchWorkspaceOffset += swiglu.swigluSegmentMetaBytes;
    dispatchWorkspaceOffset = AlignUp(dispatchWorkspaceOffset, 512U);

    swiglu.swigluDebugOffset = dispatchWorkspaceOffset;
    const uint64_t swigluC2VDebugBytes =
        static_cast<uint64_t>(swiglu.swigluSegmentNum) * cfg.aiv_num * sizeof(DispatchFFNCombineSwigluC2VDebug);
    const uint64_t swigluSegmentDebugBytes =
        static_cast<uint64_t>(swiglu.swigluSegmentNum) * sizeof(DispatchFFNCombineSwigluSegmentDebug);
    const uint64_t swigluTaskDebugBytes =
        static_cast<uint64_t>(swiglu.swigluSegmentNum) * cfg.aiv_num * sizeof(DispatchFFNCombineSwigluTaskDebug);
    const uint64_t swigluDoneDebugBytes =
        static_cast<uint64_t>(swiglu.swigluSegmentNum) * cfg.aiv_num * sizeof(DispatchFFNCombineSwigluDoneDebug);
    const uint64_t swigluFinalSyncDebugBytes =
        static_cast<uint64_t>(cfg.aiv_num) * sizeof(DispatchFFNCombineSwigluFinalSyncDebug);
    swiglu.swigluDebugBytes =
        AlignUp(sizeof(DispatchFFNCombineSwigluLayoutDebug) + swigluC2VDebugBytes + swigluSegmentDebugBytes +
                    swigluTaskDebugBytes + swigluDoneDebugBytes + swigluFinalSyncDebugBytes,
                512U);
    dispatchWorkspaceOffset += swiglu.swigluDebugBytes;
    dispatchWorkspaceOffset = AlignUp(dispatchWorkspaceOffset, 512U);

    gmm2.gmm2DebugOffset = dispatchWorkspaceOffset;
    const uint64_t gmm2V2CDebugBytes =
        static_cast<uint64_t>(swiglu.swigluSegmentNum) * cfg.aic_num * sizeof(DispatchFFNCombineGmm2V2CDebug);
    const uint64_t gmm2SegmentDebugBytes =
        static_cast<uint64_t>(swiglu.swigluSegmentNum) * sizeof(DispatchFFNCombineGmm2SegmentDebug);
    const uint64_t gmm2TaskDebugBytes =
        static_cast<uint64_t>(cfg.expert_per_rank) * cfg.aic_num * sizeof(DispatchFFNCombineGmm2TaskDebug);
    const uint64_t gmm2DoneDebugBytes = static_cast<uint64_t>(cfg.aic_num) * sizeof(DispatchFFNCombineGmm2DoneDebug);
    const uint64_t gmm2FinalSyncDebugBytes =
        static_cast<uint64_t>(cfg.aic_num) * sizeof(DispatchFFNCombineGmm2FinalSyncDebug);
    gmm2.gmm2DebugBytes =
        AlignUp(sizeof(DispatchFFNCombineGmm2LayoutDebug) + gmm2V2CDebugBytes + gmm2SegmentDebugBytes +
                    gmm2TaskDebugBytes + gmm2DoneDebugBytes + gmm2FinalSyncDebugBytes,
                512U);
    dispatchWorkspaceOffset += gmm2.gmm2DebugBytes;
    dispatchWorkspaceOffset = AlignUp(dispatchWorkspaceOffset, 512U);

    combine.combineDebugOffset = dispatchWorkspaceOffset;
    const uint64_t combineGmm2ReadyDebugBytes =
        static_cast<uint64_t>(cfg.expert_per_rank) * cfg.aiv_num * sizeof(DispatchFFNCombineCombineGmm2ReadyDebug);
    const uint64_t combineMetadataDebugBytes =
        static_cast<uint64_t>(cfg.expert_per_rank) * cfg.world_size * sizeof(DispatchFFNCombineCombineMetadataDebug);
    const uint32_t combineLoadDebugSlots = std::max(cfg.world_size, cfg.aiv_num);
    const uint64_t combineLoadDebugBytes =
        static_cast<uint64_t>(cfg.expert_per_rank) * combineLoadDebugSlots * sizeof(DispatchFFNCombineCombineLoadDebug);
    const uint32_t combineDequantDebugSlots = std::max(cfg.world_size, cfg.aiv_num);
    const uint64_t combineDequantDebugBytes = static_cast<uint64_t>(cfg.expert_per_rank) * combineDequantDebugSlots *
                                              sizeof(DispatchFFNCombineCombineDequantDebug);
    const uint32_t combineStoreDebugSlots = std::max(cfg.world_size, cfg.aiv_num);
    const uint64_t combineStoreDebugBytes = static_cast<uint64_t>(cfg.expert_per_rank) * combineStoreDebugSlots *
                                            sizeof(DispatchFFNCombineCombineStoreDebug);
    const uint64_t combineFinalizeDebugBytes =
        static_cast<uint64_t>(cfg.aiv_num) * sizeof(DispatchFFNCombineCombineFinalizeDebug);
    const uint64_t combineTaskStatsDebugBytes =
        static_cast<uint64_t>(cfg.expert_per_rank) * cfg.aiv_num * sizeof(DispatchFFNCombineCombineTaskStatsDebug);
    const uint64_t combineDoneDebugBytes =
        static_cast<uint64_t>(cfg.aiv_num) * sizeof(DispatchFFNCombineCombineDoneDebug);
    combine.combineDebugBytes =
        AlignUp(sizeof(DispatchFFNCombineCombineLayoutDebug) + combineGmm2ReadyDebugBytes + combineMetadataDebugBytes +
                    combineLoadDebugBytes + combineDequantDebugBytes + combineStoreDebugBytes +
                    combineFinalizeDebugBytes + combineTaskStatsDebugBytes + combineDoneDebugBytes,
                512U);
    dispatchWorkspaceOffset += combine.combineDebugBytes;
    dispatchWorkspaceOffset = AlignUp(dispatchWorkspaceOffset, 512U);

    auto &unpermute = result.tiling.unpermuteTiling;
    unpermute.unpermuteTokenBatch = 256U;
    unpermute.unpermuteTileCols = ChooseUnpermuteTileCols(cfg.k, unpermute.unpermuteTokenBatch, cfg.topk);
    unpermute.unpermuteLayoutVersion = 1U;
    RequireUnpermuteUbCapacity(unpermute.unpermuteTokenBatch, cfg.topk, unpermute.unpermuteTileCols);
    unpermute.unpermuteDebugOffset = dispatchWorkspaceOffset;
    const uint64_t unpermuteTaskDebugBytes =
        static_cast<uint64_t>(cfg.aiv_num) * sizeof(DispatchFFNCombineUnpermuteTaskDebug);
    const uint64_t unpermuteMetaDebugBytes =
        static_cast<uint64_t>(cfg.aiv_num) * sizeof(DispatchFFNCombineUnpermuteMetaDebug);
    const uint64_t unpermuteAccumDebugBytes =
        static_cast<uint64_t>(cfg.aiv_num) * sizeof(DispatchFFNCombineUnpermuteAccumDebug);
    const uint64_t unpermuteOutputDebugBytes =
        static_cast<uint64_t>(cfg.aiv_num) * sizeof(DispatchFFNCombineUnpermuteOutputDebug);
    const uint64_t unpermuteDoneDebugBytes =
        static_cast<uint64_t>(cfg.aiv_num) * sizeof(DispatchFFNCombineUnpermuteDoneDebug);
    unpermute.unpermuteDebugBytes =
        AlignUp(sizeof(DispatchFFNCombineUnpermuteLayoutDebug) + unpermuteTaskDebugBytes + unpermuteMetaDebugBytes +
                    unpermuteAccumDebugBytes + unpermuteOutputDebugBytes + unpermuteDoneDebugBytes,
                512U);
    dispatchWorkspaceOffset += unpermute.unpermuteDebugBytes;
    dispatchWorkspaceOffset = AlignUp(dispatchWorkspaceOffset, 512U);

    const uint64_t kernelWorkspace = dispatchWorkspaceOffset;

    result.workspace_bytes = std::max(kernelWorkspace, static_cast<uint64_t>(front.frontWorkspaceBytes));
    return result;
}
