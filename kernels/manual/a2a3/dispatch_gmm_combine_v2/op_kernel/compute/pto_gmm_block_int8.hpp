#pragma once

#if defined(__CCE_AICORE__)
#include "kernel_operator.h"
#include <pto/pto-inst.hpp>
#include "gmm_tile_graph.hpp"

#ifndef V4_FORCE_INLINE_AICORE
#define V4_FORCE_INLINE_AICORE inline __attribute__((always_inline)) __aicore__
#endif

template <typename T>
V4_FORCE_INLINE_AICORE constexpr T Int8CeilAlign(T value, T align)
{
    return align == 0 ? 0 : (value + align - 1) / align * align;
}

template <typename ElementDst,
          typename ElementAccumulator,
          uint32_t ValidM,
          uint32_t ValidN>
V4_FORCE_INLINE_AICORE void StoreAccWithFixpipe(__gm__ ElementDst* dst,
                                                __gm__ uint64_t* channelScale,
                                                uint32_t dstStrideN,
                                                uint32_t validCol)
{
    using AccTile = pto::TileAccCompact<ElementAccumulator, Int8CeilAlign<int>(ValidM, 16),
                                        Int8CeilAlign<int>(ValidN, 16),
                                        ValidM, ValidN>;
    constexpr uint32_t AlignedScaleN = ((ValidN * 8 + 127) / 128) * 128 / 8;
    using ScaleMatTile = pto::Tile<pto::TileType::Mat, uint64_t, 1, AlignedScaleN,
                                   pto::BLayout::RowMajor, 1, pto::DYNAMIC, pto::SLayout::NoneBox>;
    using ScalingTile = pto::Tile<pto::TileType::Scaling, uint64_t, 1, AlignedScaleN,
                                  pto::BLayout::RowMajor, 1, pto::DYNAMIC, pto::SLayout::NoneBox>;

    using DstShape = pto::Shape<1, 1, 1, pto::DYNAMIC, pto::DYNAMIC>;
    using DstStride = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, 1>;
    using DstGlobal = pto::GlobalTensor<ElementDst, DstShape, DstStride, pto::Layout::ND>;
    using ScaleShape = pto::Shape<1, 1, 1, 1, pto::DYNAMIC>;
    using ScaleStride = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, 1>;
    using ScaleGlobal = pto::GlobalTensor<uint64_t, ScaleShape, ScaleStride, pto::Layout::ND>;

    DstShape dstShape(ValidM, validCol);
    DstStride dstStride(static_cast<int64_t>(dstStrideN) * ValidM,
                        static_cast<int64_t>(dstStrideN) * ValidM,
                        static_cast<int64_t>(dstStrideN) * ValidM,
                        dstStrideN, 1);
    DstGlobal dstGlobal(dst, dstShape, dstStride);

    ScaleShape scaleShape(validCol);
    ScaleStride scaleStride(validCol, validCol, validCol, validCol);
    ScaleGlobal scaleGlobal(channelScale, scaleShape, scaleStride);

    ScaleMatTile scaleMatTile(validCol);
    ScalingTile scalingTile(validCol);
    AccTile accTile;

    constexpr uint64_t kL1ScaleOffset = 0x60000;
    constexpr uint64_t kFixpipeOffset = 0x0;

    pto::TASSIGN(scaleMatTile, kL1ScaleOffset);
    pto::TASSIGN(scalingTile, kFixpipeOffset);
    pto::TASSIGN(accTile, 0x0);

    pto::TLOAD(scaleMatTile, scaleGlobal);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_FIX>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_FIX>(0);
    pto::TMOV(scalingTile, scaleMatTile);

    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);

    pto::TSTORE_FP<AccTile, DstGlobal, ScalingTile>(dstGlobal, accTile, scalingTile);

    AscendC::SetFlag<AscendC::HardEvent::FIX_MTE2>(0);
    AscendC::WaitFlag<AscendC::HardEvent::FIX_MTE2>(0);
}

template <uint32_t ValidM,
          uint32_t ValidK,
          uint32_t ValidN,
          uint32_t BaseK,
          typename L1A_T, typename L1B_T, typename GlobalA_T, typename GlobalB_T>
V4_FORCE_INLINE_AICORE void LoadKTileInt8(L1A_T& lMat, L1B_T& rMat,
                                          __gm__ int8_t* lhs,
                                          __gm__ int8_t* rhs,
                                          uint32_t lhsStrideK,
                                          uint32_t rhsStrideN,
                                          uint32_t kIter)
{
    typename GlobalA_T::Stride lhsStride(
        static_cast<int64_t>(lhsStrideK) * ValidM,
        static_cast<int64_t>(lhsStrideK) * ValidM,
        static_cast<int64_t>(lhsStrideK) * ValidM,
        lhsStrideK);
    typename GlobalB_T::Stride rhsStride(
        static_cast<int64_t>(rhsStrideN) * BaseK,
        static_cast<int64_t>(rhsStrideN) * BaseK,
        static_cast<int64_t>(rhsStrideN) * BaseK,
        rhsStrideN);
    GlobalA_T lhsGlobal(lhs + static_cast<uint64_t>(kIter) * BaseK, typename GlobalA_T::Shape{}, lhsStride);
    GlobalB_T rhsGlobal(rhs + static_cast<uint64_t>(kIter) * BaseK * rhsStrideN, typename GlobalB_T::Shape{}, rhsStride);
    pto::TLOAD(lMat, lhsGlobal);
    pto::TLOAD(rMat, rhsGlobal);
}

template <uint32_t ValidM,
          uint32_t ValidK,
          uint32_t ValidN,
          uint32_t BaseK>
V4_FORCE_INLINE_AICORE void RunPtoGmmBlockInt8(__gm__ half* dst,
                                               __gm__ int8_t* lhs,
                                               __gm__ int8_t* rhs,
                                               __gm__ uint64_t* channelScale,
                                               uint32_t lhsStrideK,
                                               uint32_t rhsStrideN,
                                               uint32_t dstStrideN)
{
    constexpr int AlignedM = Int8CeilAlign<int>(static_cast<int>(ValidM), 32);
    constexpr int AlignedK = Int8CeilAlign<int>(static_cast<int>(BaseK), 32);
    constexpr int AlignedN = Int8CeilAlign<int>(static_cast<int>(ValidN), 32);
    constexpr uint32_t kLoop = (ValidK + BaseK - 1U) / BaseK;

    using GlobalDataA = pto::GlobalTensor<int8_t,
                                          pto::Shape<1, 1, 1, ValidM, BaseK>,
                                          pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, 1>>;
    using GlobalDataB = pto::GlobalTensor<int8_t,
                                          pto::Shape<1, 1, 1, BaseK, ValidN>,
                                          pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, 1>>;

    using L1A = pto::Tile<pto::TileType::Mat,
                          int8_t,
                          AlignedM,
                          AlignedK,
                          pto::BLayout::ColMajor,
                          pto::DYNAMIC,
                          pto::DYNAMIC,
                          pto::SLayout::RowMajor>;
    using L1B = pto::Tile<pto::TileType::Mat,
                          int8_t,
                          AlignedK,
                          AlignedN,
                          pto::BLayout::ColMajor,
                          pto::DYNAMIC,
                          pto::DYNAMIC,
                          pto::SLayout::RowMajor>;
    using L0A = pto::TileLeft<int8_t, AlignedM, AlignedK, pto::DYNAMIC, pto::DYNAMIC>;
    using L0B = pto::TileRight<int8_t, AlignedK, AlignedN, pto::DYNAMIC, pto::DYNAMIC>;
    using L0C = pto::TileAcc<int32_t, AlignedM, AlignedN, pto::DYNAMIC, pto::DYNAMIC>;

    constexpr uint64_t kL1ASize = static_cast<uint64_t>(AlignedM) * AlignedK;
    constexpr uint64_t kL1BSize = static_cast<uint64_t>(AlignedK) * AlignedN;
    constexpr uint64_t kL1APing = 0x0;
    constexpr uint64_t kL1APong = kL1APing + kL1ASize;
    constexpr uint64_t kL1BPing = 0x20000;
    constexpr uint64_t kL1BPong = kL1BPing + kL1BSize;

    constexpr uint64_t kL0ASize = static_cast<uint64_t>(AlignedM) * AlignedK;
    constexpr uint64_t kL0BSize = static_cast<uint64_t>(AlignedK) * AlignedN;
    constexpr uint64_t kL0APing = 0x0;
    constexpr uint64_t kL0APong = kL0APing + kL0ASize;
    constexpr uint64_t kL0BPing = 0x0;
    constexpr uint64_t kL0BPong = kL0BPing + kL0BSize;

    L1A lhsMatPing(ValidM, BaseK);
    L1A lhsMatPong(ValidM, BaseK);
    L1B rhsMatPing(BaseK, ValidN);
    L1B rhsMatPong(BaseK, ValidN);
    L0A lhsTilePing(ValidM, BaseK);
    L0A lhsTilePong(ValidM, BaseK);
    L0B rhsTilePing(BaseK, ValidN);
    L0B rhsTilePong(BaseK, ValidN);
    L0C accTile(ValidM, ValidN);

    pto::TASSIGN(lhsMatPing, kL1APing);
    pto::TASSIGN(lhsMatPong, kL1APong);
    pto::TASSIGN(rhsMatPing, kL1BPing);
    pto::TASSIGN(rhsMatPong, kL1BPong);
    pto::TASSIGN(lhsTilePing, kL0APing);
    pto::TASSIGN(lhsTilePong, kL0APong);
    pto::TASSIGN(rhsTilePing, kL0BPing);
    pto::TASSIGN(rhsTilePong, kL0BPong);
    pto::TASSIGN(accTile, 0x0);

    LoadKTileInt8<ValidM, ValidK, ValidN, BaseK, L1A, L1B, GlobalDataA, GlobalDataB>(
        lhsMatPing, rhsMatPing, lhs, rhs, lhsStrideK, rhsStrideN, 0);

    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    pto::TMOV(lhsTilePing, lhsMatPing);
    pto::TMOV(rhsTilePing, rhsMatPing);

    if (kLoop > 1) {
        LoadKTileInt8<ValidM, ValidK, ValidN, BaseK, L1A, L1B, GlobalDataA, GlobalDataB>(
            lhsMatPong, rhsMatPong, lhs, rhs, lhsStrideK, rhsStrideN, 1);
    }

    for (uint32_t kIter = 0; kIter < kLoop; ++kIter) {
        const bool isPing = (kIter & 1U) == 0;
        L0A& curL0A = isPing ? lhsTilePing : lhsTilePong;
        L0B& curL0B = isPing ? rhsTilePing : rhsTilePong;
        L0A& nextL0A = isPing ? lhsTilePong : lhsTilePing;
        L0B& nextL0B = isPing ? rhsTilePong : rhsTilePing;
        L1A& nextL1A = isPing ? lhsMatPing : lhsMatPong;
        L1B& nextL1B = isPing ? rhsMatPing : rhsMatPong;

        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);

        if (kIter == 0) {
            pto::TMATMUL(accTile, curL0A, curL0B);
        } else {
            pto::TMATMUL_ACC(accTile, accTile, curL0A, curL0B);
        }

        if (kIter + 1 < kLoop) {
            set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
            wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
            pto::TMOV(nextL0A, isPing ? lhsMatPong : lhsMatPing);
            pto::TMOV(nextL0B, isPing ? rhsMatPong : rhsMatPing);
        }

        set_flag(PIPE_M, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_M, PIPE_MTE2, EVENT_ID0);

        if (kIter + 2 < kLoop) {
            LoadKTileInt8<ValidM, ValidK, ValidN, BaseK, L1A, L1B, GlobalDataA, GlobalDataB>(
                nextL1A, nextL1B, lhs, rhs, lhsStrideK, rhsStrideN, kIter + 2);
        }
    }

    StoreAccWithFixpipe<half, int32_t, ValidM, ValidN>(dst, channelScale, dstStrideN, ValidN);
    pipe_barrier(PIPE_ALL);
}

template <uint32_t ValidK, uint32_t BaseK>
V4_FORCE_INLINE_AICORE void RunGmm1Int8NRange(__gm__ int8_t* input,
                                               __gm__ int8_t* weight1,
                                               __gm__ half* gmm1Out,
                                               __gm__ uint64_t* channelScale,
                                               uint32_t hiddenK,
                                               uint32_t gmm1N,
                                               uint32_t outputStrideN,
                                               uint32_t nBegin,
                                               uint32_t nEnd)
{
    for (uint32_t nBase = nBegin; nBase < nEnd;) {
        const uint32_t remaining = nEnd - nBase;
        auto* tileWeight = weight1 + static_cast<uint64_t>(nBase);
        auto* tileOut = gmm1Out + nBase;
        auto* tileScale = channelScale + nBase;
        if (remaining >= 32) {
            RunPtoGmmBlockInt8<8, ValidK, 32, BaseK>(
                tileOut, input, tileWeight, tileScale, hiddenK, gmm1N, outputStrideN);
            nBase += 32;
        } else if (remaining >= 16) {
            RunPtoGmmBlockInt8<8, ValidK, 16, BaseK>(
                tileOut, input, tileWeight, tileScale, hiddenK, gmm1N, outputStrideN);
            nBase += 16;
        } else {
            return;
        }
    }
}

V4_FORCE_INLINE_AICORE void RunGmm1Int8Tile(__gm__ int8_t* input,
                                             __gm__ int8_t* weight1,
                                             __gm__ half* gmm1Out,
                                             __gm__ uint64_t* channelScale,
                                             const GmmTileRuntimeMeta& tileMeta,
                                             uint32_t hiddenK,
                                             uint32_t gmm1N,
                                             uint32_t outputStrideN)
{
    const uint32_t rows = tileMeta.rowEnd - tileMeta.rowBegin;
    const uint32_t kWidth = tileMeta.kEnd - tileMeta.kBegin;
    const uint32_t nWidth = tileMeta.nEnd - tileMeta.nBegin;
    if (rows == 0 || kWidth == 0 || nWidth == 0 || hiddenK == 0 || gmm1N < 16) {
        return;
    }
    if (hiddenK >= 128) {
        RunGmm1Int8NRange<128, 16>(input, weight1, gmm1Out, channelScale, hiddenK, gmm1N, outputStrideN, tileMeta.nBegin, tileMeta.nEnd);
    } else if (hiddenK >= 64) {
        RunGmm1Int8NRange<64, 16>(input, weight1, gmm1Out, channelScale, hiddenK, gmm1N, outputStrideN, tileMeta.nBegin, tileMeta.nEnd);
    } else if (hiddenK >= 32) {
        RunGmm1Int8NRange<32, 16>(input, weight1, gmm1Out, channelScale, hiddenK, gmm1N, outputStrideN, tileMeta.nBegin, tileMeta.nEnd);
    } else if (hiddenK >= 16) {
        RunGmm1Int8NRange<16, 16>(input, weight1, gmm1Out, channelScale, hiddenK, gmm1N, outputStrideN, tileMeta.nBegin, tileMeta.nEnd);
    }
}

template <uint32_t ValidK2, uint32_t BaseK2>
V4_FORCE_INLINE_AICORE void RunGmm2Int8NRange(__gm__ int8_t* swigluQ,
                                               __gm__ int8_t* weight2,
                                               __gm__ half* gmm2Out,
                                               __gm__ uint64_t* channelScale2,
                                               uint32_t k2Total,
                                               uint32_t weight2StrideN,
                                               uint32_t outputStrideN,
                                               uint32_t n2Begin,
                                               uint32_t n2End)
{
    for (uint32_t n2Base = n2Begin; n2Base < n2End;) {
        const uint32_t remaining = n2End - n2Base;
        auto* tileWeight = weight2 + static_cast<uint64_t>(n2Base);
        auto* tileOut = gmm2Out + n2Base;
        auto* tileScale = channelScale2 + n2Base;
        if (remaining >= 32) {
            RunPtoGmmBlockInt8<8, ValidK2, 32, BaseK2>(
                tileOut, swigluQ, tileWeight, tileScale, k2Total, weight2StrideN, outputStrideN);
            n2Base += 32;
        } else if (remaining >= 16) {
            RunPtoGmmBlockInt8<8, ValidK2, 16, BaseK2>(
                tileOut, swigluQ, tileWeight, tileScale, k2Total, weight2StrideN, outputStrideN);
            n2Base += 16;
        } else {
            return;
        }
    }
}

V4_FORCE_INLINE_AICORE void RunGmm2Int8Tile(__gm__ int8_t* swigluQ,
                                             __gm__ int8_t* weight2,
                                             __gm__ half* gmm2Out,
                                             __gm__ uint64_t* channelScale2,
                                             const Gmm2TileRuntimeMeta& tileMeta,
                                             uint32_t k2Total,
                                             uint32_t outputElems,
                                             uint32_t weight2StrideN,
                                             uint32_t outputStrideN)
{
    const uint32_t rows = tileMeta.rowEnd - tileMeta.rowBegin;
    const uint32_t k2Width = tileMeta.k2End - tileMeta.k2Begin;
    const uint32_t n2Width = tileMeta.n2End - tileMeta.n2Begin;
    if (rows == 0 || k2Width == 0 || n2Width == 0 || k2Total == 0 || outputElems < 16) {
        return;
    }

    const uint32_t n2LogicalEnd = tileMeta.n2End > outputElems ? outputElems : tileMeta.n2End;
    for (uint32_t n2Base = tileMeta.n2Begin; n2Base < n2LogicalEnd;) {
        const uint32_t n2Next = n2Base + 32U <= outputStrideN ? n2Base + 32U : outputStrideN;
        if (k2Total >= 128) {
            RunGmm2Int8NRange<128, 64>(swigluQ, weight2, gmm2Out, channelScale2, k2Total, weight2StrideN, outputStrideN, n2Base, n2Next);
        } else if (k2Total >= 64) {
            RunGmm2Int8NRange<64, 64>(swigluQ, weight2, gmm2Out, channelScale2, k2Total, weight2StrideN, outputStrideN, n2Base, n2Next);
        } else if (k2Total >= 32) {
            RunGmm2Int8NRange<32, 32>(swigluQ, weight2, gmm2Out, channelScale2, k2Total, weight2StrideN, outputStrideN, n2Base, n2Next);
        }
        n2Base = n2Next;
    }
}

#endif
