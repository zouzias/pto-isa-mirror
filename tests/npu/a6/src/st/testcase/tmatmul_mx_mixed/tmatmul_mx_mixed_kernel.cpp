/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software; you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <pto/pto-inst.hpp>
#include <pto/common/constants.hpp>

using namespace pto;

constexpr uint32_t MX_SCALE_GROUP = 32;
constexpr uint32_t HIF4_SCALE_GROUP = 64;
constexpr uint64_t L0A_BUF0 = 0x0u;
constexpr uint64_t L0B_BUF0 = 0x0u;

template <typename T>
AICORE constexpr inline T CeilAlign(T num1, T num2)
{
    return (num1 + num2 - 1) / num2 * num2;
}

template <typename LeftT, int validM, int validK, int validN>
AICORE inline void RunMxE2m1Impl(
    __gm__ bfloat16_t* out, __gm__ LeftT* aData, __gm__ uint8_t* aScale, __gm__ float4_e2m1x2_t* bData,
    __gm__ uint8_t* bScale)
{
    constexpr int M = CeilAlign<int>(validM, 16);
    constexpr int N = CeilAlign<int>(validN, 16);
    constexpr int K = CeilAlign<int>(validK, 16);
    constexpr int scaleK = validK / MX_SCALE_GROUP;

    using TileMatA = Tile<TileType::Mat, LeftT, M, K, BLayout::ColMajor, validM, validK, SLayout::RowMajor, TileConfig::fractalABSize>;
    using TileMatB = Tile<TileType::Mat, float4_e2m1x2_t, K, N, BLayout::ColMajor, validK, validN, SLayout::RowMajor, TileConfig::fractalABSize>;
    using TileScaleA = Tile<TileType::Mat, uint8_t, M, scaleK, BLayout::RowMajor, validM, scaleK, SLayout::RowMajor, 32>;
    using TileScaleB = Tile<TileType::Mat, uint8_t, scaleK, N, BLayout::ColMajor, scaleK, validN, SLayout::ColMajor, 32>;

    using GlobalDataA = GlobalTensor<LeftT, pto::Shape<1, 1, 1, validM, validK>, pto::Stride<validM * validK, validM * validK, validM * validK, validK, 1>>;
    using GlobalDataB = GlobalTensor<float4_e2m1x2_t, pto::Shape<1, 1, 1, validK, validN>, pto::Stride<validK * validN, validK * validN, validK * validN, validN, 1>>;
    using MxShapeA = TileShape2D<uint8_t, M, scaleK, Layout::MX_A_ZZ>;
    using MxStrideA = BaseShape2D<uint8_t, M, scaleK, Layout::MX_A_ZZ>;
    using GlobalScaleA = GlobalTensor<uint8_t, MxShapeA, MxStrideA, Layout::MX_A_ZZ>;
    using MxShapeB = TileShape2D<uint8_t, scaleK, N, Layout::MX_B_NN>;
    using MxStrideB = BaseShape2D<uint8_t, scaleK, N, Layout::MX_B_NN>;
    using GlobalScaleB = GlobalTensor<uint8_t, MxShapeB, MxStrideB, Layout::MX_B_NN>;
    using GlobalDataOut = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, validM, validN>, pto::Stride<validM * validN, validM * validN, validM * validN, validN, 1>>;

    GlobalDataA aDataGm(aData);
    GlobalDataB bDataGm(bData);
    GlobalScaleA aScaleGm(aScale);
    GlobalScaleB bScaleGm(bScale);
    GlobalDataOut outGm(out);

    using LeftTile = TileLeft<LeftT, M, K, validM, validK>;
    using RightTile = TileRight<float4_e2m1x2_t, K, N, validK, validN>;
    using LeftScaleTile = TileLeftScale<uint8_t, M, scaleK, validM, scaleK>;
    using RightScaleTile = TileRightScale<uint8_t, scaleK, N, scaleK, validN>;
    using AccTile = TileAcc<float, M, N, validM, validN>;

    TileMatA aMatTile;
    TileMatB bMatTile;
    TileScaleA aScaleTile;
    TileScaleB bScaleTile;
    TASSIGN(aMatTile, 0x0u);
    TASSIGN(bMatTile, 0x20000u);
    TASSIGN(aScaleTile, 0x40000u);
    TASSIGN(bScaleTile, 0x60000u);

    LeftTile al0;
    RightTile bl0;
    LeftScaleTile aScaleL0;
    RightScaleTile bScaleL0;
    AccTile cTile;

    TASSIGN(al0, L0A_BUF0);
    TASSIGN(bl0, L0B_BUF0);
    TASSIGN(aScaleL0, GetScaleAddr(al0.data()));
    TASSIGN(bScaleL0, GetScaleAddr(bl0.data()));
    TASSIGN(cTile, 0x0u);

    TLOAD(aMatTile, aDataGm);
    TLOAD(bMatTile, bDataGm);
    TLOAD<TileScaleA, GlobalScaleA>(aScaleTile, aScaleGm);
    TLOAD<TileScaleB, GlobalScaleB>(bScaleTile, bScaleGm);

#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
#endif

    TEXTRACT(al0, aMatTile, 0, 0);
    TEXTRACT(aScaleL0, aScaleTile, 0, 0);
    TEXTRACT(bl0, bMatTile, 0, 0);
    TEXTRACT(bScaleL0, bScaleTile, 0, 0);

#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
#endif

    TMATMUL_MX(cTile, al0, aScaleL0, bl0, bScaleL0);

#ifndef __PTO_AUTO__
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
#endif

    TSTORE(outGm, cTile);
}

template <typename LeftT, int validM, int validK, int validN>
AICORE inline void RunMxHif4Impl(
    __gm__ bfloat16_t* out, __gm__ LeftT* aData, __gm__ uint8_t* aScale, __gm__ hifloat4x2_t* bData,
    __gm__ uint8_t* bScale)
{
    constexpr int M = CeilAlign<int>(validM, 16);
    constexpr int N = CeilAlign<int>(validN, 16);
    constexpr int K = CeilAlign<int>(validK, 16);
    constexpr int scaleKLeft = validK / MX_SCALE_GROUP;
    constexpr int scaleKRight = validK / HIF4_SCALE_GROUP;
    constexpr int scaleKColsRight = scaleKRight * HIF4_COL_LEN;

    using TileMatA = Tile<TileType::Mat, LeftT, M, K, BLayout::ColMajor, validM, validK, SLayout::RowMajor, TileConfig::fractalABSize>;
    using TileMatB = Tile<TileType::Mat, hifloat4x2_t, K, N, BLayout::ColMajor, validK, validN, SLayout::RowMajor, TileConfig::fractalABSize>;
    using TileScaleA = Tile<TileType::Mat, uint8_t, M, scaleKLeft, BLayout::RowMajor, validM, scaleKLeft, SLayout::RowMajor, 32>;
    using TileScaleB = Tile<TileType::Mat, uint8_t, scaleKColsRight, N, BLayout::ColMajor, scaleKColsRight, validN, SLayout::ColMajor, 32>;

    using GlobalDataA = GlobalTensor<LeftT, pto::Shape<1, 1, 1, validM, validK>, pto::Stride<validM * validK, validM * validK, validM * validK, validK, 1>>;
    using GlobalDataB = GlobalTensor<hifloat4x2_t, pto::Shape<1, 1, 1, validK, validN>, pto::Stride<validK * validN, validK * validN, validK * validN, validN, 1>>;
    using MxShapeA = TileShape2D<uint8_t, M, scaleKLeft, Layout::MX_A_ZZ>;
    using MxStrideA = BaseShape2D<uint8_t, M, scaleKLeft, Layout::MX_A_ZZ>;
    using GlobalScaleA = GlobalTensor<uint8_t, MxShapeA, MxStrideA, Layout::MX_A_ZZ>;
    using MxShapeB = TileShape2D<uint8_t, scaleKRight, N, Layout::HIF4_B_NN>;
    using MxStrideB = BaseShape2D<uint8_t, scaleKRight, N, Layout::HIF4_B_NN>;
    using GlobalScaleB = GlobalTensor<uint8_t, MxShapeB, MxStrideB, Layout::HIF4_B_NN>;
    using GlobalDataOut = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, validM, validN>, pto::Stride<validM * validN, validM * validN, validM * validN, validN, 1>>;

    GlobalDataA aDataGm(aData);
    GlobalDataB bDataGm(bData);
    GlobalScaleA aScaleGm(aScale);
    GlobalScaleB bScaleGm(bScale);
    GlobalDataOut outGm(out);

    using LeftTile = TileLeft<LeftT, M, K, validM, validK>;
    using RightTile = TileRight<hifloat4x2_t, K, N, validK, validN>;
    using LeftScaleTile = TileLeftScale<uint8_t, M, scaleKLeft, validM, scaleKLeft>;
    using RightScaleTile = TileRightScale<uint8_t, scaleKColsRight, N, scaleKColsRight, validN>;
    using AccTile = TileAcc<float, M, N, validM, validN>;

    TileMatA aMatTile;
    TileMatB bMatTile;
    TileScaleA aScaleTile;
    TileScaleB bScaleTile;
    TASSIGN(aMatTile, 0x0u);
    TASSIGN(bMatTile, 0x20000u);
    TASSIGN(aScaleTile, 0x40000u);
    TASSIGN(bScaleTile, 0x60000u);

    LeftTile al0;
    RightTile bl0;
    LeftScaleTile aScaleL0;
    RightScaleTile bScaleL0;
    AccTile cTile;

    TASSIGN(al0, L0A_BUF0);
    TASSIGN(bl0, L0B_BUF0);
    TASSIGN(aScaleL0, GetScaleAddr(al0.data()));
    TASSIGN(bScaleL0, GetScaleAddr(bl0.data()));
    TASSIGN(cTile, 0x0u);

    TLOAD(aMatTile, aDataGm);
    TLOAD(bMatTile, bDataGm);
    TLOAD<TileScaleA, GlobalScaleA>(aScaleTile, aScaleGm);
    TLOAD<TileScaleB, GlobalScaleB>(bScaleTile, bScaleGm);

#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
#endif

    TEXTRACT(al0, aMatTile, 0, 0);
    TEXTRACT(aScaleL0, aScaleTile, 0, 0);
    TEXTRACT(bl0, bMatTile, 0, 0);
    TEXTRACT(bScaleL0, bScaleTile, 0, 0);

#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
#endif

    TMATMUL_MX(cTile, al0, aScaleL0, bl0, bScaleL0);

#ifndef __PTO_AUTO__
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
#endif

    TSTORE(outGm, cTile);
}

template <typename LeftT, int validM, int validK, int validN>
__global__ AICORE void RunMxE2m1Matmul(
    __gm__ bfloat16_t* out, __gm__ LeftT* aData, __gm__ uint8_t* aScale, __gm__ float4_e2m1x2_t* bData,
    __gm__ uint8_t* bScale)
{
    RunMxE2m1Impl<LeftT, validM, validK, validN>(out, aData, aScale, bData, bScale);
}

template <typename LeftT, int validM, int validK, int validN>
__global__ AICORE void RunMxHif4Matmul(
    __gm__ bfloat16_t* out, __gm__ LeftT* aData, __gm__ uint8_t* aScale, __gm__ hifloat4x2_t* bData,
    __gm__ uint8_t* bScale)
{
    RunMxHif4Impl<LeftT, validM, validK, validN>(out, aData, aScale, bData, bScale);
}

namespace TmatmulMxMixedA6 {
template <int caseId>
void Launch(uint8_t* out, uint8_t* aData, uint8_t* aScale, uint8_t* bData, uint8_t* bScale, void* stream)
{
    if constexpr (caseId == 1) {
        RunMxHif4Matmul<float8_e4m3_t, 128, 128, 128><<<1, nullptr, stream>>>(
            reinterpret_cast<bfloat16_t*>(out), reinterpret_cast<float8_e4m3_t*>(aData), reinterpret_cast<uint8_t*>(aScale),
            reinterpret_cast<hifloat4x2_t*>(bData), reinterpret_cast<uint8_t*>(bScale));
    } else if constexpr (caseId == 2) {
        RunMxHif4Matmul<float8_e4m3_t, 64, 128, 64><<<1, nullptr, stream>>>(
            reinterpret_cast<bfloat16_t*>(out), reinterpret_cast<float8_e4m3_t*>(aData), reinterpret_cast<uint8_t*>(aScale),
            reinterpret_cast<hifloat4x2_t*>(bData), reinterpret_cast<uint8_t*>(bScale));
    } else if constexpr (caseId == 3) {
        RunMxE2m1Matmul<half, 128, 128, 128><<<1, nullptr, stream>>>(
            reinterpret_cast<bfloat16_t*>(out), reinterpret_cast<half*>(aData), reinterpret_cast<uint8_t*>(aScale),
            reinterpret_cast<float4_e2m1x2_t*>(bData), reinterpret_cast<uint8_t*>(bScale));
    } else if constexpr (caseId == 4) {
        RunMxE2m1Matmul<half, 64, 128, 64><<<1, nullptr, stream>>>(
            reinterpret_cast<bfloat16_t*>(out), reinterpret_cast<half*>(aData), reinterpret_cast<uint8_t*>(aScale),
            reinterpret_cast<float4_e2m1x2_t*>(bData), reinterpret_cast<uint8_t*>(bScale));
    } else if constexpr (caseId == 5) {
        RunMxE2m1Matmul<bfloat16_t, 128, 128, 128><<<1, nullptr, stream>>>(
            reinterpret_cast<bfloat16_t*>(out), reinterpret_cast<bfloat16_t*>(aData), reinterpret_cast<uint8_t*>(aScale),
            reinterpret_cast<float4_e2m1x2_t*>(bData), reinterpret_cast<uint8_t*>(bScale));
    } else if constexpr (caseId == 6) {
        RunMxE2m1Matmul<bfloat16_t, 64, 128, 64><<<1, nullptr, stream>>>(
            reinterpret_cast<bfloat16_t*>(out), reinterpret_cast<bfloat16_t*>(aData), reinterpret_cast<uint8_t*>(aScale),
            reinterpret_cast<float4_e2m1x2_t*>(bData), reinterpret_cast<uint8_t*>(bScale));
    } else if constexpr (caseId == 7) {
        RunMxHif4Matmul<half, 128, 128, 128><<<1, nullptr, stream>>>(
            reinterpret_cast<bfloat16_t*>(out), reinterpret_cast<half*>(aData), reinterpret_cast<uint8_t*>(aScale),
            reinterpret_cast<hifloat4x2_t*>(bData), reinterpret_cast<uint8_t*>(bScale));
    } else if constexpr (caseId == 8) {
        RunMxHif4Matmul<half, 64, 128, 64><<<1, nullptr, stream>>>(
            reinterpret_cast<bfloat16_t*>(out), reinterpret_cast<half*>(aData), reinterpret_cast<uint8_t*>(aScale),
            reinterpret_cast<hifloat4x2_t*>(bData), reinterpret_cast<uint8_t*>(bScale));
    } else if constexpr (caseId == 9) {
        RunMxHif4Matmul<bfloat16_t, 128, 128, 128><<<1, nullptr, stream>>>(
            reinterpret_cast<bfloat16_t*>(out), reinterpret_cast<bfloat16_t*>(aData), reinterpret_cast<uint8_t*>(aScale),
            reinterpret_cast<hifloat4x2_t*>(bData), reinterpret_cast<uint8_t*>(bScale));
    } else if constexpr (caseId == 10) {
        RunMxHif4Matmul<bfloat16_t, 64, 128, 64><<<1, nullptr, stream>>>(
            reinterpret_cast<bfloat16_t*>(out), reinterpret_cast<bfloat16_t*>(aData), reinterpret_cast<uint8_t*>(aScale),
            reinterpret_cast<hifloat4x2_t*>(bData), reinterpret_cast<uint8_t*>(bScale));
    }
}
} // namespace TmatmulMxMixedA6

template void TmatmulMxMixedA6::Launch<1>(uint8_t*, uint8_t*, uint8_t*, uint8_t*, uint8_t*, void*);
template void TmatmulMxMixedA6::Launch<2>(uint8_t*, uint8_t*, uint8_t*, uint8_t*, uint8_t*, void*);
template void TmatmulMxMixedA6::Launch<3>(uint8_t*, uint8_t*, uint8_t*, uint8_t*, uint8_t*, void*);
template void TmatmulMxMixedA6::Launch<4>(uint8_t*, uint8_t*, uint8_t*, uint8_t*, uint8_t*, void*);
template void TmatmulMxMixedA6::Launch<5>(uint8_t*, uint8_t*, uint8_t*, uint8_t*, uint8_t*, void*);
template void TmatmulMxMixedA6::Launch<6>(uint8_t*, uint8_t*, uint8_t*, uint8_t*, uint8_t*, void*);
template void TmatmulMxMixedA6::Launch<7>(uint8_t*, uint8_t*, uint8_t*, uint8_t*, uint8_t*, void*);
template void TmatmulMxMixedA6::Launch<8>(uint8_t*, uint8_t*, uint8_t*, uint8_t*, uint8_t*, void*);
template void TmatmulMxMixedA6::Launch<9>(uint8_t*, uint8_t*, uint8_t*, uint8_t*, uint8_t*, void*);
template void TmatmulMxMixedA6::Launch<10>(uint8_t*, uint8_t*, uint8_t*, uint8_t*, uint8_t*, void*);
