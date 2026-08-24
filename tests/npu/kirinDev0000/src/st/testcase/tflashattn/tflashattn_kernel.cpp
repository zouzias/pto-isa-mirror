/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <pto/pto-inst.hpp>
#include <pto/common/pto_tile.hpp>
#include <pto/common/constants.hpp>
#include "acl/acl.h"

using namespace pto;

namespace {

constexpr int kSeqLen = 64;
constexpr int kHeadDim = 32;
constexpr float kScale = 0.17677669529f;

template <typename T>
AICORE constexpr inline T CeilAlign(T num1, T num2)
{
    if (num2 == 0) {
        return 0;
    }
    return (num1 + num2 - 1) / num2 * num2;
}

} // namespace

template <typename OutType, typename AType, typename BType, int M, int K, int N>
AICORE inline void RunMatmul(__gm__ OutType* out, __gm__ AType* src0, __gm__ BType* src1)
{
    using GlobalSrc0 = GlobalTensor<AType, pto::Shape<1, 1, 1, M, K>, pto::Stride<M * K, M * K, M * K, K, 1>>;
    using GlobalSrc1 = GlobalTensor<BType, pto::Shape<1, 1, 1, K, N>, pto::Stride<K * N, K * N, K * N, N, 1>>;
    using GlobalDst = GlobalTensor<OutType, pto::Shape<1, 1, 1, M, N>, pto::Stride<M * N, M * N, M * N, N, 1>>;
    GlobalSrc0 src0Global(src0);
    GlobalSrc1 src1Global(src1);
    GlobalDst dstGlobal(out);

    using AMat = Tile<TileType::Mat, AType, M, K, BLayout::ColMajor, M, K, SLayout::RowMajor, 512>;
    using BMat = Tile<TileType::Mat, BType, K, N, BLayout::ColMajor, K, N, SLayout::RowMajor, 512>;
    using BRight = TileRight<BType, K, N, K, N>;
    using AccTile = TileAcc<OutType, M, N, M, N>;
    using OutVec = Tile<TileType::Vec, OutType, M, N, BLayout::RowMajor, M, N, SLayout::NoneBox>;

    AMat aMat;
    BMat bMat;
    TASSIGN<0x0>(aMat);
    TASSIGN<M * K * sizeof(AType)>(bMat);

    BRight bRight;
    TASSIGN<0x0>(bRight);

    AccTile accTile;
    TASSIGN<0x0>(accTile);

    OutVec outVec;
    TASSIGN<0x0>(outVec);

    TLOAD(aMat, src0Global);
    TLOAD(bMat, src1Global);

    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

    TMOV(bRight, bMat);

    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);

    TMATMUL(accTile, aMat, bRight);

    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);

    TMOV(outVec, accTile);

    set_flag(PIPE_FIX, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_FIX, PIPE_MTE3, EVENT_ID0);

    TSTORE(dstGlobal, outVec);

    out = dstGlobal.data();
}

template <typename T, int M, int N>
AICORE inline void RunSoftmax(__gm__ T* probsOut, __gm__ T* scoresIn, T scale)
{
    using GlobalData = GlobalTensor<T, pto::Shape<1, 1, 1, M, N>, pto::Stride<M * N, M * N, M * N, N, 1>>;
    GlobalData scoresGlobal(scoresIn);
    GlobalData probsGlobal(probsOut);

    using ScoresVec = Tile<TileType::Vec, T, M, N, BLayout::RowMajor, M, N>;
    using RowReduce = Tile<TileType::Vec, T, M, 1, BLayout::ColMajor, M, 1>;
    using TmpVec = Tile<TileType::Vec, T, M, N, BLayout::RowMajor, M, N>;

    ScoresVec scoresVec;
    RowReduce rowMax;
    TmpVec tmpVec;
    ScoresVec centeredVec;
    ScoresVec expVec;
    RowReduce rowSum;
    ScoresVec probsVec;

    TASSIGN<0x0>(scoresVec);
    TASSIGN<M * N * sizeof(T)>(rowMax);
    TASSIGN<M * N * sizeof(T) + M * sizeof(T)>(tmpVec);
    TASSIGN<2 * M * N * sizeof(T) + M * sizeof(T)>(centeredVec);
    TASSIGN<3 * M * N * sizeof(T) + M * sizeof(T)>(expVec);
    TASSIGN<4 * M * N * sizeof(T) + M * sizeof(T)>(rowSum);
    TASSIGN<5 * M * N * sizeof(T) + 2 * M * sizeof(T)>(probsVec);

    TLOAD(scoresVec, scoresGlobal);

    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    TMULS(scoresVec, scoresVec, scale);
    TROWMAX(rowMax, scoresVec, tmpVec);
    TROWEXPANDSUB(centeredVec, scoresVec, rowMax);
    TEXP(expVec, centeredVec);
    TROWSUM(rowSum, expVec, tmpVec);
    TROWEXPANDDIV(probsVec, expVec, rowSum);

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    TSTORE(probsGlobal, probsVec);
}

__global__ AICORE void RunTFLASHATTN_HALF(
    __gm__ half* out, __gm__ half* q, __gm__ half* kT, __gm__ half* v,
    __gm__ half* scoresBuf, __gm__ half* probsBuf)
{
    constexpr int M = CeilAlign(kSeqLen, 16);
    constexpr int K1 = CeilAlign(kHeadDim, 16);
    constexpr int N1 = CeilAlign(kSeqLen, 16);
    constexpr int K2 = CeilAlign(kSeqLen, 16);
    constexpr int N2 = CeilAlign(kHeadDim, 16);

    RunMatmul<half, half, half, M, K1, N1>(scoresBuf, q, kT);

    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);

    const half scale = static_cast<half>(kScale);
    RunSoftmax<half, M, N1>(probsBuf, scoresBuf, scale);

    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);

    RunMatmul<half, half, half, M, K2, N2>(out, probsBuf, v);
}

template <typename IntType, int M, int K, int N>
AICORE inline void RunQKQuantizedSoftmax(
    __gm__ half* probsBuf, __gm__ IntType* q, __gm__ IntType* kT, __gm__ int32_t* scoresBuf)
{
    RunMatmul<int32_t, IntType, IntType, M, K, N>(scoresBuf, q, kT);

    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);

    using GlobalInt = GlobalTensor<int32_t, pto::Shape<1, 1, 1, M, N>, pto::Stride<M * N, M * N, M * N, N, 1>>;
    using GlobalHalf = GlobalTensor<half, pto::Shape<1, 1, 1, M, N>, pto::Stride<M * N, M * N, M * N, N, 1>>;
    GlobalInt intGlobal(scoresBuf);
    GlobalHalf halfGlobal(probsBuf);

    using IntVec = Tile<TileType::Vec, int32_t, M, N, BLayout::RowMajor, M, N>;
    using FltVec = Tile<TileType::Vec, float, M, N, BLayout::RowMajor, M, N>;
    using RowReduce = Tile<TileType::Vec, float, M, 1, BLayout::ColMajor, M, 1>;
    using TmpVec = Tile<TileType::Vec, float, M, N, BLayout::RowMajor, M, N>;
    using HalfVec = Tile<TileType::Vec, half, M, N, BLayout::RowMajor, M, N>;

    IntVec intVec;
    FltVec fltVec;
    RowReduce rowMax;
    TmpVec tmpVec;
    FltVec centeredVec;
    FltVec expVec;
    RowReduce rowSum;
    FltVec probsFltVec;
    HalfVec probsHalfVec;

    TASSIGN<0x0>(intVec);
    TASSIGN<0x0>(fltVec);
    TASSIGN<0x0>(rowMax);
    TASSIGN<M * sizeof(float)>(tmpVec);
    TASSIGN<M * N * sizeof(float) + M * sizeof(float)>(centeredVec);
    TASSIGN<2 * M * N * sizeof(float) + M * sizeof(float)>(expVec);
    TASSIGN<3 * M * N * sizeof(float) + M * sizeof(float)>(rowSum);
    TASSIGN<0x0>(probsFltVec);
    TASSIGN<M * N * sizeof(float)>(probsHalfVec);

    TLOAD(intVec, intGlobal);

    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    TCVT(fltVec, intVec, RoundMode::CAST_RINT);
    TMULS(fltVec, fltVec, kScale);
    TROWMAX(rowMax, fltVec, tmpVec);
    TROWEXPANDSUB(centeredVec, fltVec, rowMax);
    TEXP(expVec, centeredVec);
    TROWSUM(rowSum, expVec, tmpVec);
    TROWEXPANDDIV(probsFltVec, expVec, rowSum);
    TCVT(probsHalfVec, probsFltVec, RoundMode::CAST_RINT);

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    TSTORE(halfGlobal, probsHalfVec);
}

__global__ AICORE void RunTFLASHATTN_S8(
    __gm__ half* out, __gm__ int8_t* q, __gm__ int8_t* kT, __gm__ half* v,
    __gm__ int32_t* scoresBuf, __gm__ half* probsBuf)
{
    constexpr int M = CeilAlign(kSeqLen, 16);
    constexpr int K1 = CeilAlign(kHeadDim, 16);
    constexpr int N1 = CeilAlign(kSeqLen, 16);
    constexpr int K2 = CeilAlign(kSeqLen, 16);
    constexpr int N2 = CeilAlign(kHeadDim, 16);

    RunQKQuantizedSoftmax<int8_t, M, K1, N1>(probsBuf, q, kT, scoresBuf);

    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);

    RunMatmul<half, half, half, M, K2, N2>(out, probsBuf, v);
}

__global__ AICORE void RunTFLASHATTN_S16(
    __gm__ half* out, __gm__ int16_t* q, __gm__ int16_t* kT, __gm__ half* v,
    __gm__ half* scoresBuf, __gm__ half* probsBuf,
    __gm__ half* qHalfBuf, __gm__ half* ktHalfBuf)
{
    constexpr int M = CeilAlign(kSeqLen, 16);
    constexpr int K1 = CeilAlign(kHeadDim, 16);
    constexpr int N1 = CeilAlign(kSeqLen, 16);
    constexpr int K2 = CeilAlign(kSeqLen, 16);
    constexpr int N2 = CeilAlign(kHeadDim, 16);

    using GlobalQInt = GlobalTensor<int16_t, pto::Shape<1, 1, 1, M, K1>, pto::Stride<M * K1, M * K1, M * K1, K1, 1>>;
    using GlobalKTInt = GlobalTensor<int16_t, pto::Shape<1, 1, 1, K1, N1>, pto::Stride<K1 * N1, K1 * N1, K1 * N1, N1, 1>>;
    using GlobalQHalf = GlobalTensor<half, pto::Shape<1, 1, 1, M, K1>, pto::Stride<M * K1, M * K1, M * K1, K1, 1>>;
    using GlobalKTHalf = GlobalTensor<half, pto::Shape<1, 1, 1, K1, N1>, pto::Stride<K1 * N1, K1 * N1, K1 * N1, N1, 1>>;
    GlobalQInt qIntGlobal(q);
    GlobalKTInt ktIntGlobal(kT);
    GlobalQHalf qHalfGlobal(qHalfBuf);
    GlobalKTHalf ktHalfGlobal(ktHalfBuf);

    using QIntVec = Tile<TileType::Vec, int16_t, M, K1, BLayout::RowMajor, M, K1>;
    using KTIntVec = Tile<TileType::Vec, int16_t, K1, N1, BLayout::RowMajor, K1, N1>;
    using QHalfVec = Tile<TileType::Vec, half, M, K1, BLayout::RowMajor, M, K1>;
    using KTHalfVec = Tile<TileType::Vec, half, K1, N1, BLayout::RowMajor, K1, N1>;

    QIntVec qIntVec;
    KTIntVec ktIntVec;
    QHalfVec qHalfVec;
    KTHalfVec ktHalfVec;
    TASSIGN<0x0>(qIntVec);
    TASSIGN<M * K1 * sizeof(int16_t)>(ktIntVec);
    TASSIGN<0x0>(qHalfVec);
    TASSIGN<M * K1 * sizeof(half)>(ktHalfVec);

    TLOAD(qIntVec, qIntGlobal);
    TLOAD(ktIntVec, ktIntGlobal);

    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    TCVT(qHalfVec, qIntVec, RoundMode::CAST_RINT);
    TCVT(ktHalfVec, ktIntVec, RoundMode::CAST_RINT);

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    TSTORE(qHalfGlobal, qHalfVec);
    TSTORE(ktHalfGlobal, ktHalfVec);

    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);

    RunMatmul<half, half, half, M, K1, N1>(scoresBuf, qHalfBuf, ktHalfBuf);

    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);

    const half scale = static_cast<half>(kScale);
    RunSoftmax<half, M, N1>(probsBuf, scoresBuf, scale);

    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);

    RunMatmul<half, half, half, M, K2, N2>(out, probsBuf, v);
}

void LaunchTFLASHATTNHalf(
    aclFloat16* out, aclFloat16* q, aclFloat16* kT, aclFloat16* v,
    aclFloat16* scoresBuf, aclFloat16* probsBuf, void* stream)
{
    RunTFLASHATTN_HALF<<<1, nullptr, stream>>>(
        reinterpret_cast<half*>(out), reinterpret_cast<half*>(q), reinterpret_cast<half*>(kT),
        reinterpret_cast<half*>(v), reinterpret_cast<half*>(scoresBuf), reinterpret_cast<half*>(probsBuf));
}

void LaunchTFLASHATTNS8(
    aclFloat16* out, int8_t* q, int8_t* kT, aclFloat16* v,
    aclFloat16* scoresBuf, aclFloat16* probsBuf, void* stream)
{
    RunTFLASHATTN_S8<<<1, nullptr, stream>>>(
        reinterpret_cast<half*>(out), q, kT, reinterpret_cast<half*>(v),
        reinterpret_cast<int32_t*>(scoresBuf), reinterpret_cast<half*>(probsBuf));
}

void LaunchTFLASHATTNS16(
    aclFloat16* out, int16_t* q, int16_t* kT, aclFloat16* v,
    aclFloat16* scoresBuf, aclFloat16* probsBuf,
    aclFloat16* qHalfBuf, aclFloat16* ktHalfBuf, void* stream)
{
    RunTFLASHATTN_S16<<<1, nullptr, stream>>>(
        reinterpret_cast<half*>(out), q, kT, reinterpret_cast<half*>(v),
        reinterpret_cast<half*>(scoresBuf), reinterpret_cast<half*>(probsBuf),
        reinterpret_cast<half*>(qHalfBuf), reinterpret_cast<half*>(ktHalfBuf));
}
