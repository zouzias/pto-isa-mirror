/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef DISPATCH_MEGA_COMBINE_UNPERMUTE_H
#define DISPATCH_MEGA_COMBINE_UNPERMUTE_H

#include <type_traits>

#include <pto/pto-inst.hpp>

#include "dispatch_mega_combine_tiling.h"
#include "kernel_operator.h"
#include "utils/common_helpers.hpp"
#include "utils/const_args.hpp"
#include "utils/hccl_window.hpp"
#include "utils/mega_expert_sync.hpp"
#include "utils/mega_wave_schedule.hpp"
#include "utils/pto_vector.hpp"

constexpr uint32_t kUnpermuteVecTileElems = 8192U;
constexpr uint32_t kUnpermuteMetadataBufferNum = 2U;
constexpr uint32_t kUnpermuteTokenBufferNum = 2U;
constexpr uint32_t kUnpermuteAccumulatorBufferNum = 2U;

template <typename OutputElement>
class Unpermute {
public:
    AICORE inline void Init(GM_ADDR workspaceGM, GM_ADDR expertIdGM, GM_ADDR probsGM, GM_ADDR outGM,
                            const __gm__ MegaMoeTilingData *tilingData, uint32_t workerIdx, uint32_t workerCount);
    AICORE inline void Process();

private:
    static_assert(std::is_same_v<OutputElement, bfloat16_t>, "MXFP8 unpermute output must be BF16");

    using VectorShape = pto::Shape<1, 1, 1, 1, pto::DYNAMIC>;
    using VectorStride = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, 1>;
    using DGlobal = pto::GlobalTensor<OutputElement, VectorShape, VectorStride, pto::Layout::ND>;
    using OutputElementType = OutputElement;
    using TileD =
        pto::Tile<pto::TileType::Vec, OutputElement, 1, kUnpermuteVecTileElems, pto::BLayout::RowMajor, -1, -1>;
    using TileFp32 = pto::Tile<pto::TileType::Vec, float, 1, kUnpermuteVecTileElems, pto::BLayout::RowMajor, -1, -1>;
    AICORE inline uint32_t TileCols() const
    {
        uint32_t tileCols = tilingData_->unpermuteTiling.unpermuteTileCols;
        if (tileCols == 0U || tileCols > kUnpermuteVecTileElems) {
            tileCols = kUnpermuteVecTileElems;
        }
        return tileCols;
    }
    AICORE inline uint32_t TokenBatch() const
    {
        uint32_t batch = tilingData_->unpermuteTiling.unpermuteTokenBatch;
        return batch == 0U ? 1U : batch;
    }
    AICORE inline event_t LoadFreeEvent(uint32_t bufferId) const
    {
        return static_cast<event_t>(bufferId);
    }
    AICORE inline event_t LoadReadyEvent(uint32_t bufferId) const
    {
        return static_cast<event_t>(bufferId + 2U);
    }
    AICORE inline event_t AccumulatorStoreEvent(uint32_t bufferId) const
    {
        return static_cast<event_t>(bufferId == 0U ? EVENT_ID4 : EVENT_ID5);
    }
    AICORE inline event_t OutputStoreFreeEvent() const
    {
        return EVENT_ID6;
    }
    AICORE inline event_t OutputStoreReadyEvent() const
    {
        return EVENT_ID7;
    }
    AICORE inline event_t MetadataReadyEvent(uint32_t bufferId) const
    {
        return static_cast<event_t>(bufferId == 0U ? EVENT_ID6 : EVENT_ID7);
    }
    AICORE inline void InitUbLayout();
    AICORE inline void SetInitialFlags() const;
    AICORE inline void FinalizeLocalPipe() const;
    AICORE inline void PrefetchMetadata(uint32_t bufferId, uint32_t batchStart, uint32_t batchTokens) const;
    AICORE inline void WaitMetadata(uint32_t bufferId) const;
    AICORE inline int32_t ReadExpandedRow(uint32_t metaBufferId, uint32_t localToken, uint32_t topkIdx) const;
    AICORE inline float ReadProb(uint32_t metaBufferId, uint32_t localToken, uint32_t topkIdx) const;
    AICORE inline uint32_t AcquireAccumulatorBuffer();
    AICORE inline void LoadRouteChunk(uint32_t bufferId, uint32_t compactRow, uint32_t col, uint32_t cols) const;
    AICORE inline void AccumulateChunk(uint32_t bufferId, uint32_t accumulatorBufferId, float prob, uint32_t cols);
    AICORE inline void StoreOutputChunk(uint32_t accumulatorBufferId, uint32_t token, uint32_t col, uint32_t cols);
    AICORE inline void ProcessToken(uint32_t metaBufferId, uint32_t batchStart, uint32_t localToken);
    AICORE inline bool TokenReadyForExpertProgress(uint32_t batchStart, uint32_t localToken,
                                                   const uint32_t *readyExpertCounts) const;
    AICORE inline void BuildTokenRange(uint32_t workerIdx, uint32_t workerCount, uint32_t &tokenStart,
                                       uint32_t &tokenCount) const;
    AICORE inline void ProcessRankStreamingRange(uint32_t tokenStart, uint32_t tokenCount,
                                                 const uint32_t *phase1ReadyExpertCounts, bool processPhase1);
    AICORE inline void ProcessRankStreaming();
    AICORE inline uint32_t CombineCompletionWorkerCount() const;

    GM_ADDR workspaceGM_ = nullptr;
    const __gm__ MegaMoeTilingData *tilingData_ = nullptr;
    PtoRemoteWindow remoteWindow_;
    MegaMoePeerMemoryLayout peerMemoryLayout_;
    __gm__ OutputElement *combineOutputPtr_ = nullptr;
    __gm__ int32_t *expertIdPtr_ = nullptr;
    __gm__ int32_t *expandedRowIdxPtr_ = nullptr;
    __gm__ float *probsPtr_ = nullptr;
    __gm__ OutputElement *outPtr_ = nullptr;

    uint32_t problemM_ = 0;
    uint32_t problemK_ = 0;
    uint32_t topK_ = 0;
    uint32_t expertPerRank_ = 0;
    uint32_t rankSize_ = 0;
    int32_t dataReadyEpoch_ = 0;
    uint32_t coreIdx_ = 0;
    uint32_t coreNum_ = 1;
    uint32_t pingpongId_ = 0;
    uint32_t accumulatorCursor_ = 0;
    uint64_t ubIndexOffset_[kUnpermuteMetadataBufferNum] = {0, 0};
    uint64_t ubProbOffset_[kUnpermuteMetadataBufferNum] = {0, 0};
    uint64_t ubAccOffset_[kUnpermuteAccumulatorBufferNum] = {0, 0};
    uint64_t ubTokenOffset_[kUnpermuteTokenBufferNum] = {0, 0};
    uint64_t ubTokenFp32Offset_[kUnpermuteTokenBufferNum] = {0, 0};
    uint64_t ubOutOffset_ = 0;
    bool accumulatorStorePending_[kUnpermuteAccumulatorBufferNum] = {false, false};
};

template <typename OutputElement>
AICORE inline void Unpermute<OutputElement>::Init(GM_ADDR workspaceGM, GM_ADDR expertIdGM, GM_ADDR probsGM,
                                                  GM_ADDR outGM, const __gm__ MegaMoeTilingData *tilingData,
                                                  uint32_t workerIdx, uint32_t workerCount)
{
    workspaceGM_ = workspaceGM;
    tilingData_ = tilingData;

    problemM_ = tilingData_->megaMoeInfo.M;
    problemK_ = tilingData_->megaMoeInfo.K;
    topK_ = tilingData_->megaMoeInfo.topK;
    expertPerRank_ = tilingData_->megaMoeInfo.expertPerRank;
    rankSize_ = tilingData_->runtimeInfo.rankSize;
    coreIdx_ = workerIdx;
    coreNum_ = workerCount;

    remoteWindow_.Init(reinterpret_cast<GM_ADDR>(tilingData_->runtimeInfo.remoteWindowContext));
    peerMemoryLayout_.Init(remoteWindow_, tilingData_->frontReorderTiling);
    combineOutputPtr_ = reinterpret_cast<__gm__ OutputElement *>(
        remoteWindow_.LocalBase() + peerMemoryLayout_.combineOutputByRouteSlot);
    expertIdPtr_ = reinterpret_cast<__gm__ int32_t *>(expertIdGM);
    expandedRowIdxPtr_ = reinterpret_cast<__gm__ int32_t *>(
        workspaceGM_ + tilingData_->frontReorderTiling.expandedRowIdxOffset);
    probsPtr_ = reinterpret_cast<__gm__ float *>(probsGM);
    outPtr_ = reinterpret_cast<__gm__ OutputElement *>(outGM);
    dataReadyEpoch_ = remoteWindow_.DataReadyEpoch();

    InitUbLayout();
}

template <typename OutputElement>
AICORE inline void Unpermute<OutputElement>::InitUbLayout()
{
    const uint32_t tileCols = TileCols();
    const uint32_t metaElems = TokenBatch() * topK_;
    uint64_t ubOffset = 0;
    for (uint32_t i = 0; i < kUnpermuteMetadataBufferNum; ++i) {
        ubIndexOffset_[i] = ubOffset;
        ubOffset += alignUp(static_cast<uint64_t>(metaElems) * sizeof(int32_t), UB_ALIGN);
        ubProbOffset_[i] = ubOffset;
        ubOffset += alignUp(static_cast<uint64_t>(metaElems) * sizeof(float), UB_ALIGN);
    }
    for (uint32_t i = 0; i < kUnpermuteAccumulatorBufferNum; ++i) {
        ubAccOffset_[i] = ubOffset;
        ubOffset += alignUp(static_cast<uint64_t>(tileCols) * sizeof(float), UB_ALIGN);
    }
    for (uint32_t i = 0; i < kUnpermuteTokenBufferNum; ++i) {
        ubTokenOffset_[i] = ubOffset;
        ubOffset += alignUp(static_cast<uint64_t>(tileCols) * sizeof(OutputElement), UB_ALIGN);
        ubTokenFp32Offset_[i] = ubOffset;
        ubOffset += alignUp(static_cast<uint64_t>(tileCols) * sizeof(float), UB_ALIGN);
    }
    ubOutOffset_ = ubOffset;
    ubOffset += alignUp(static_cast<uint64_t>(tileCols) * sizeof(OutputElement), UB_ALIGN);
}

template <typename OutputElement>
AICORE inline void Unpermute<OutputElement>::SetInitialFlags() const
{
    for (uint32_t i = 0; i < kUnpermuteTokenBufferNum; ++i) {
        set_flag(PIPE_V, PIPE_MTE2, LoadFreeEvent(i));
    }
    set_flag(PIPE_MTE3, PIPE_V, OutputStoreFreeEvent());
}

template <typename OutputElement>
AICORE inline void Unpermute<OutputElement>::FinalizeLocalPipe() const
{
    for (uint32_t i = 0; i < kUnpermuteTokenBufferNum; ++i) {
        wait_flag(PIPE_V, PIPE_MTE2, LoadFreeEvent(i));
    }
    for (uint32_t i = 0; i < kUnpermuteAccumulatorBufferNum; ++i) {
        if (accumulatorStorePending_[i]) {
            wait_flag(PIPE_MTE3, PIPE_V, AccumulatorStoreEvent(i));
        }
    }
    wait_flag(PIPE_MTE3, PIPE_V, OutputStoreFreeEvent());
}

template <typename OutputElement>
AICORE inline void Unpermute<OutputElement>::PrefetchMetadata(uint32_t bufferId, uint32_t batchStart,
                                                              uint32_t batchTokens) const
{
    const uint32_t metaElems = batchTokens * topK_;
    PtoLoadVector<int32_t, kUnpermuteVecTileElems>(ubIndexOffset_[bufferId],
                                                   expandedRowIdxPtr_ + batchStart * topK_, metaElems);
    PtoLoadVector<float, kUnpermuteVecTileElems>(ubProbOffset_[bufferId], probsPtr_ + batchStart * topK_, metaElems);
    set_flag(PIPE_MTE2, PIPE_S, MetadataReadyEvent(bufferId));
}

template <typename OutputElement>
AICORE inline void Unpermute<OutputElement>::WaitMetadata(uint32_t bufferId) const
{
    wait_flag(PIPE_MTE2, PIPE_S, MetadataReadyEvent(bufferId));
}

template <typename OutputElement>
AICORE inline int32_t Unpermute<OutputElement>::ReadExpandedRow(uint32_t metaBufferId, uint32_t localToken,
                                                                uint32_t topkIdx) const
{
    return PtoGetValue<int32_t, kUnpermuteVecTileElems>(ubIndexOffset_[metaBufferId],
                                                        localToken * topK_ + topkIdx);
}

template <typename OutputElement>
AICORE inline float Unpermute<OutputElement>::ReadProb(uint32_t metaBufferId, uint32_t localToken,
                                                       uint32_t topkIdx) const
{
    return PtoGetValue<float, kUnpermuteVecTileElems>(ubProbOffset_[metaBufferId], localToken * topK_ + topkIdx);
}

template <typename OutputElement>
AICORE inline uint32_t Unpermute<OutputElement>::AcquireAccumulatorBuffer()
{
    const uint32_t accumulatorBufferId = accumulatorCursor_;
    accumulatorCursor_ = (accumulatorCursor_ + 1U) % kUnpermuteAccumulatorBufferNum;
    if (accumulatorStorePending_[accumulatorBufferId]) {
        wait_flag(PIPE_MTE3, PIPE_V, AccumulatorStoreEvent(accumulatorBufferId));
        accumulatorStorePending_[accumulatorBufferId] = false;
    }
    return accumulatorBufferId;
}

template <typename OutputElement>
AICORE inline void Unpermute<OutputElement>::LoadRouteChunk(uint32_t bufferId, uint32_t compactRow, uint32_t col,
                                                            uint32_t cols) const
{
    wait_flag(PIPE_V, PIPE_MTE2, LoadFreeEvent(bufferId));
    TileD dTile(1, cols);
    pto::TASSIGN(dTile, ubTokenOffset_[bufferId]);
    VectorShape dShape(cols);
    VectorStride dStride(cols, cols, cols, cols);
    DGlobal dGlobal(combineOutputPtr_ + static_cast<uint64_t>(compactRow) * problemK_ + col, dShape, dStride);
    pto::TLOAD(dTile, dGlobal);
    set_flag(PIPE_MTE2, PIPE_V, LoadReadyEvent(bufferId));
}

template <typename OutputElement>
AICORE inline void Unpermute<OutputElement>::AccumulateChunk(uint32_t bufferId, uint32_t accumulatorBufferId,
                                                             float prob, uint32_t cols)
{
    wait_flag(PIPE_MTE2, PIPE_V, LoadReadyEvent(bufferId));
    TileD dTile(1, cols);
    TileFp32 fp32Tile(1, cols);
    pto::TASSIGN(dTile, ubTokenOffset_[bufferId]);
    pto::TASSIGN(fp32Tile, ubTokenFp32Offset_[bufferId]);
    pto::TCVT(fp32Tile, dTile, pto::RoundMode::CAST_NONE); // token转fp32
    set_flag(PIPE_V, PIPE_MTE2, LoadFreeEvent(bufferId));
    pto::TMULS(fp32Tile, fp32Tile, prob); // token乘以权重
    TileFp32 accTile(1, cols);
    pto::TASSIGN(accTile, ubAccOffset_[accumulatorBufferId]);
    pto::TADD(accTile, accTile, fp32Tile); // 累加
}

template <typename OutputElement>
AICORE inline void Unpermute<OutputElement>::StoreOutputChunk(uint32_t accumulatorBufferId, uint32_t token,
                                                              uint32_t col, uint32_t cols)
{
    wait_flag(PIPE_MTE3, PIPE_V, OutputStoreFreeEvent());
    TileFp32 accTile(1, cols);
    TileD outTile(1, cols);
    pto::TASSIGN(accTile, ubAccOffset_[accumulatorBufferId]);
    pto::TASSIGN(outTile, ubOutOffset_);
    pto::TCVT(outTile, accTile, pto::RoundMode::CAST_RINT);
    set_flag(PIPE_V, PIPE_MTE3, OutputStoreReadyEvent());
    wait_flag(PIPE_V, PIPE_MTE3, OutputStoreReadyEvent());
    VectorShape outShape(cols);
    VectorStride outStride(cols, cols, cols, cols);
    DGlobal outGlobal(outPtr_ + static_cast<uint64_t>(token) * problemK_ + col, outShape, outStride);
    pto::TSTORE(outGlobal, outTile);
    set_flag(PIPE_MTE3, PIPE_V, OutputStoreFreeEvent());
}

template <typename OutputElement>
AICORE inline void Unpermute<OutputElement>::ProcessToken(uint32_t metaBufferId, uint32_t batchStart,
                                                          uint32_t localToken)
{
    const uint32_t token = batchStart + localToken;
    for (uint32_t col = 0; col < problemK_; col += TileCols()) { // K按照1024切分
        const uint32_t cols = (problemK_ - col > TileCols()) ? TileCols() : (problemK_ - col);
        const uint32_t accumulatorBufferId = AcquireAccumulatorBuffer();
        PtoFillUb<float, kUnpermuteVecTileElems>(ubAccOffset_[accumulatorBufferId], 0.0f, cols); // 初始化输出UB

        bool hasPending = false;
        uint32_t pendingBuffer = 0;
        float pendingProb = 0.0f;

        for (uint32_t topkIdx = 0; topkIdx < topK_; ++topkIdx) {
            const int32_t expandedRow = ReadExpandedRow(metaBufferId, localToken, topkIdx);
            if (expandedRow < 0) {
                continue;
            }
            const float prob = ReadProb(metaBufferId, localToken, topkIdx);
            pto::PtoSetWaitFlag<PIPE_S, PIPE_V>();
            const uint32_t bufferId = pingpongId_;
            pingpongId_ = (pingpongId_ + 1U) % kUnpermuteTokenBufferNum;
            LoadRouteChunk(bufferId, static_cast<uint32_t>(expandedRow), col, cols);
            if (hasPending) {
                AccumulateChunk(pendingBuffer, accumulatorBufferId, pendingProb, cols);
            }
            pendingBuffer = bufferId;
            pendingProb = prob;
            hasPending = true;
        }
        if (hasPending) {
            AccumulateChunk(pendingBuffer, accumulatorBufferId, pendingProb, cols);
        }
        StoreOutputChunk(accumulatorBufferId, token, col, cols); // 写回GM
    }
}

template <typename OutputElement>
AICORE inline bool Unpermute<OutputElement>::TokenReadyForExpertProgress(uint32_t batchStart, uint32_t localToken,
                                                                         const uint32_t *readyExpertCounts) const
{
    if (readyExpertCounts == nullptr || expertPerRank_ == 0U || rankSize_ == 0U ||
        rankSize_ > COMBINE_EXPERT_PROGRESS_MAX_RANKS) {
        return false;
    }
    bool allRoutesReady = true;
    for (uint32_t topkIdx = 0U; topkIdx < topK_; ++topkIdx) {
        const int32_t expert = expertIdPtr_[static_cast<uint64_t>(batchStart + localToken) * topK_ + topkIdx];
        if (expert < 0) {
            allRoutesReady = false;
            continue;
        }
        const uint32_t globalExpert = static_cast<uint32_t>(expert);
        const uint32_t producerRank = globalExpert / expertPerRank_;
        const uint32_t localExpert = globalExpert - producerRank * expertPerRank_;
        if (producerRank >= rankSize_ || localExpert >= readyExpertCounts[producerRank]) {
            allRoutesReady = false;
        }
    }
    return topK_ != 0U && allRoutesReady;
}

template <typename OutputElement>
AICORE inline void Unpermute<OutputElement>::BuildTokenRange(uint32_t workerIdx, uint32_t workerCount,
                                                             uint32_t &tokenStart, uint32_t &tokenCount) const
{
    tokenStart = 0U;
    tokenCount = 0U;
    if (workerCount == 0U || workerIdx >= workerCount) {
        return;
    }
    const uint32_t splitBase = problemM_ / workerCount;
    const uint32_t splitRem = problemM_ % workerCount;
    tokenStart = workerIdx * splitBase + (workerIdx < splitRem ? workerIdx : splitRem);
    tokenCount = splitBase + (workerIdx < splitRem ? 1U : 0U);
}

template <typename OutputElement>
AICORE inline void Unpermute<OutputElement>::ProcessRankStreamingRange(uint32_t tokenStart, uint32_t tokenCount,
                                                                       const uint32_t *phase1ReadyExpertCounts,
                                                                       bool processPhase1)
{
    if (tokenCount == 0U) {
        return;
    }

    const uint32_t batchLimit = TokenBatch();
    uint32_t currentBatchStart = tokenStart;
    uint32_t currentBatchTokens = tokenCount < batchLimit ? tokenCount : batchLimit;
    uint32_t currentBuffer = 0U;
    PrefetchMetadata(currentBuffer, currentBatchStart, currentBatchTokens);

    uint32_t consumedTokens = 0U;
    while (consumedTokens < tokenCount) {
        WaitMetadata(currentBuffer);
        const uint32_t nextConsumed = consumedTokens + currentBatchTokens;
        const bool hasNext = nextConsumed < tokenCount;
        const uint32_t nextBuffer = (currentBuffer + 1U) % kUnpermuteMetadataBufferNum;
        uint32_t nextBatchStart = 0U;
        uint32_t nextBatchTokens = 0U;
        if (hasNext) {
            nextBatchStart = tokenStart + nextConsumed;
            const uint32_t remaining = tokenCount - nextConsumed;
            nextBatchTokens = remaining < batchLimit ? remaining : batchLimit;
            PrefetchMetadata(nextBuffer, nextBatchStart, nextBatchTokens);
        }

        for (uint32_t localToken = 0U; localToken < currentBatchTokens; ++localToken) {
            const bool phase1Task =
                TokenReadyForExpertProgress(currentBatchStart, localToken, phase1ReadyExpertCounts);
            if (phase1Task == processPhase1) {
                ProcessToken(currentBuffer, currentBatchStart, localToken);
            }
        }

        consumedTokens = nextConsumed;
        currentBatchStart = nextBatchStart;
        currentBatchTokens = nextBatchTokens;
        currentBuffer = nextBuffer;
    }
}

template <typename OutputElement>
AICORE inline uint32_t Unpermute<OutputElement>::CombineCompletionWorkerCount() const
{
    const __gm__ MegaMoeFixedGroupTiling &fixed = tilingData_->fixedGroupTiling;
    if (rankSize_ == 0U || fixed.gmm2GroupSize == 0U) {
        return 0U;
    }
    return fixed.gmm2GroupSize < rankSize_ ? fixed.gmm2GroupSize : rankSize_;
}

template <typename OutputElement>
AICORE inline void Unpermute<OutputElement>::ProcessRankStreaming()
{
    const uint32_t rankCount = rankSize_;
    const uint32_t initialWorkerStart = tilingData_->fixedGroupTiling.physicalAicNum;
    const uint32_t initialWorkerCount =
        initialWorkerStart < kMegaMoeFixedInitialUnpermuteAiv0WorkerCount ?
            initialWorkerStart :
            kMegaMoeFixedInitialUnpermuteAiv0WorkerCount;
    const uint32_t initialWorkerEnd = initialWorkerStart + initialWorkerCount;
    const bool phase1Worker = coreIdx_ >= initialWorkerStart && coreIdx_ < initialWorkerEnd;
    const uint32_t coordinatorWorker = initialWorkerStart;

    WaitEpochAcquire(remoteWindow_.LocalUnpermutePhase1MaskEpochSlot(), dataReadyEpoch_);
    uint32_t phase1ReadyExpertCounts[COMBINE_EXPERT_PROGRESS_MAX_RANKS] = {0U};
    remoteWindow_.ReadUnpermutePhase1Progress(phase1ReadyExpertCounts, rankCount);

    if (phase1Worker) {
        const uint32_t phase1WorkerIdx = coreIdx_ - initialWorkerStart;
        uint32_t phase1TokenStart = 0U;
        uint32_t phase1TokenCount = 0U;
        BuildTokenRange(phase1WorkerIdx, initialWorkerCount, phase1TokenStart, phase1TokenCount);
        ProcessRankStreamingRange(phase1TokenStart, phase1TokenCount, phase1ReadyExpertCounts, true);
        remoteWindow_.PublishPhase1Done(phase1WorkerIdx, dataReadyEpoch_);
    }

    if (coreIdx_ == coordinatorWorker) {
        remoteWindow_.WaitPhase1DoneMte(initialWorkerCount, dataReadyEpoch_);
        remoteWindow_.WaitLocalCombineDoneMte(CombineCompletionWorkerCount(), dataReadyEpoch_);
        uint32_t liveReadyExpertCounts[COMBINE_EXPERT_PROGRESS_MAX_RANKS] = {0U};
        uint32_t minimumReady = 0U;
        uint32_t missCount = 0U;
        while (minimumReady < expertPerRank_) {
            const uint32_t observedMinimum =
                remoteWindow_.ReadExpertProgressMte(dataReadyEpoch_, expertPerRank_, liveReadyExpertCounts);
            if (observedMinimum > minimumReady) {
                remoteWindow_.AcquireDataReady();
                minimumReady = observedMinimum;
                missCount = 0U;
            } else {
                ++missCount;
                EpochPollBackoff(missCount);
            }
        }
        remoteWindow_.PublishUnpermuteAllReady(coreNum_, dataReadyEpoch_);
    } else {
        WaitEpochAcquire(remoteWindow_.LocalUnpermuteAllReadySlot(coreIdx_), dataReadyEpoch_);
    }

    uint32_t phase2TokenStart = 0U;
    uint32_t phase2TokenCount = 0U;
    BuildTokenRange(coreIdx_, coreNum_, phase2TokenStart, phase2TokenCount);
    ProcessRankStreamingRange(phase2TokenStart, phase2TokenCount, phase1ReadyExpertCounts, false);
}

template <typename OutputElement>
AICORE inline void Unpermute<OutputElement>::Process()
{
    if ASCEND_IS_AIC {
        return;
    }
    WaitEpochAcquire(
        FixedSyncSlot(
            workspaceGM_, tilingData_, kMegaMoeFixedSyncDeferredExpandedReadySlot),
        kMegaMoeFixedDeferredExpandedReadyMarker);
    SetInitialFlags();
    ProcessRankStreaming();
    FinalizeLocalPipe();
}

#endif // DISPATCH_MEGA_COMBINE_UNPERMUTE_H
