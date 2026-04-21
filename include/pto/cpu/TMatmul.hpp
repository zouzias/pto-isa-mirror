/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TMATMUL_HPP
#define TMATMUL_HPP

#include "pto/cpu/tile_offsets.hpp"
#include "pto/cpu/parallel.hpp"

namespace pto {
template <typename LeftSrcType, typename RightSrcType, typename DType>
void Gemm(DType* dst, const DType* acc,
          const DType src0, const DType src1,
          uint16_t M, uint16_t N, uint16_t K)
{
    // Placeholder implementation - simple triple loop
    for (uint16_t i = 0; i < M; i++) {
        for (uint16_t j = 0; j < N; j++) {
            DType sum = acc ? acc[i * N + j] : static_cast<DType>(0);

            for (uint16_t k = 0; k < K; k++) {
                sum += static_cast<DType>(src0[i * K + k] * src1[k * N + j]);
            }

            dst[i * N + j] = sum;
        }
    }
}

template <typename TileAcc, typename TileLeft, typename TileRight>
void TMatmulNzZn(typename TileAcc::TileDType dst, typename TileAcc::TileDType acc, typename TileLeft::TileDType src0,
                 typename TileRight::TileDType src1, uint16_t M, uint16_t N, uint16_t K)
{
    using LeftSrcType = typename TileLeft::TileDType;
    using RightSrcType = typename TileRight::TileDType;
    using DType = typename TileAcc::DType;
    cpu::Gemm<LeftSrcType, RightSrcType, DType>(dst, acc, src0, src1, M, N, K);
}

template <typename TileAcc, typename TileLeft, typename TileRight>
PTO_INTERNAL void CheckMadValid()
{
    using AType = typename TileLeft::DType;
    using BType = typename TileRight::DType;
    using CType = typename TileAcc::DType;
    static_assert(
        (std::is_same_v<AType, int8_t> && std::is_same_v<BType, int8_t> && std::is_same_v<CType, int32_t>) || // s8
            (std::is_same_v<AType, half> && std::is_same_v<BType, half> && std::is_same_v<CType, float>) ||   // f162f32
            (std::is_same_v<AType, bfloat16_t> && std::is_same_v<BType, bfloat16_t> &&
             std::is_same_v<CType, float>) ||                                                              // bf162f32
            (std::is_same_v<AType, float> && std::is_same_v<BType, float> && std::is_same_v<CType, float>) // f322f32
        ,
        "Not supported data type");
    static_assert(
        (TileLeft::Rows == TileAcc::Rows) && (TileLeft::Cols == TileRight::Rows) && (TileRight::Cols == TileAcc::Cols),
        "Inconsistent number of m, k, n");
    // CPU simulation can see two equivalent Left-tile encodings:
    // - TileLeft<...> aliases use ColMajor B-layout in CPU builds
    // - PTOAS-generated kernels materialize explicit Tile<Left,...,RowMajor,...>
    //   declarations that still produce correct offsets through GetTileElementOffset.
    static_assert(
        ((TileLeft::Loc == TileType::Left) && (TileLeft::SFractal == SLayout::RowMajor)) &&
            ((TileRight::Loc == TileType::Right) && (TileRight::isRowMajor) &&
             (TileRight::SFractal == SLayout::ColMajor)) &&
            ((TileAcc::Loc == TileType::Acc) && (!TileAcc::isRowMajor) && (TileAcc::SFractal == SLayout::RowMajor)),
        "Non-conforming matrix fractal");
}

template <typename TileAcc, typename TileBias>
PTO_INTERNAL void CheckBiasValid()
{
    using CType = typename TileAcc::DType;
    using BiasType = typename TileBias::DType;
    static_assert(std::is_same_v<CType, BiasType>, "No supported bias data type");
    static_assert((TileBias::Loc == TileType::Bias) && (TileBias::Rows == 1) && (TileBias::isRowMajor),
                  "Non-conforming bias fractal");
}

template <typename TileAcc, typename TileLeft, typename TileRight>
PTO_INTERNAL void TMATMUL_IMPL(TileAcc &cMatrix, TileLeft &aMatrix, TileRight &bMatrix)
{
    CheckMadValid<TileAcc, TileLeft, TileRight>();

    uint16_t m = aMatrix.GetValidRow();
    uint16_t k = aMatrix.GetValidCol();
    uint16_t n = bMatrix.GetValidCol();

    TMatmulNzZn<TileAcc, TileLeft, TileRight>(cMatrix.data(), nullptr, aMatrix.data(), bMatrix.data(), m, n, k);
}

template <typename TileAcc, typename TileLeft, typename TileRight>
PTO_INTERNAL void TMATMUL_ACC_IMPL(TileAcc &cOutMatrix, TileAcc &cInMatrix, TileLeft &aMatrix, TileRight &bMatrix)
{
    CheckMadValid<TileAcc, TileLeft, TileRight>();

    uint16_t m = aMatrix.GetValidRow();
    uint16_t k = aMatrix.GetValidCol();
    uint16_t n = bMatrix.GetValidCol();

    TMatmulNzZn<TileAcc, TileLeft, TileRight>(cOutMatrix.data(), cInMatrix.data(), aMatrix.data(), bMatrix.data(), m, n,
                                              k);
}

template <typename TileAcc, typename TileLeft, typename TileRight, typename TileBias>
PTO_INTERNAL void TMATMUL_BIAS_IMPL(TileAcc &cMatrix, TileLeft &aMatrix, TileRight &bMatrix, TileBias &biasMatrix)
{
    CheckMadValid<TileAcc, TileLeft, TileRight>();
    CheckBiasValid<TileAcc, TileBias>();

    uint16_t m = aMatrix.GetValidRow();
    uint16_t k = aMatrix.GetValidCol();
    uint16_t n = bMatrix.GetValidCol();

    TMatmulNzZn<TileAcc, TileLeft, TileRight>(cMatrix.data(), nullptr, aMatrix.data(), bMatrix.data(), m, n, k);
    for (size_t c = 0; c < n; c++) {
        for (size_t r = 0; r < m; r++) {
            size_t out_idx = GetTileElementOffset<TileAcc>(r, c);
            size_t bias_idx = GetTileElementOffset<TileBias>(0, c);
            cMatrix.data()[out_idx] += biasMatrix.data()[bias_idx];
        }
    }
}

template <AccPhase Phase = AccPhase::Unspecified, typename TileRes, typename TileLeft, typename TileRight>
PTO_INTERNAL void TGEMV_IMPL(TileRes &cMatrix, TileLeft &aMatrix, TileRight &bMatrix)
{
    (void)Phase;
    TMATMUL_IMPL(cMatrix, aMatrix, bMatrix);
}

template <AccPhase Phase = AccPhase::Unspecified, typename TileRes, typename TileLeft, typename TileRight>
PTO_INTERNAL void TGEMV_ACC_IMPL(TileRes &cOutMatrix, TileRes &cInMatrix, TileLeft &aMatrix, TileRight &bMatrix)
{
    (void)Phase;
    TMATMUL_ACC_IMPL(cOutMatrix, cInMatrix, aMatrix, bMatrix);
}

template <AccPhase Phase = AccPhase::Unspecified, typename TileRes, typename TileLeft, typename TileRight,
          typename TileBias>
PTO_INTERNAL void TGEMV_BIAS_IMPL(TileRes &cMatrix, TileLeft &aMatrix, TileRight &bMatrix, TileBias &biasData)
{
    (void)Phase;
    TMATMUL_BIAS_IMPL(cMatrix, aMatrix, bMatrix, biasData);
}

template <AccPhase Phase = AccPhase::Unspecified, typename TileRes, typename TileLeft, typename TileLeftScale,
          typename TileRight, typename TileRightScale>
PTO_INTERNAL void TMATMUL_MX_IMPL(TileRes &cMatrix, TileLeft &aMatrix, TileLeftScale &aScaleMatrix, TileRight &bMatrix,
                                  TileRightScale &bScaleMatrix)
{
    (void)Phase;
    (void)aScaleMatrix;
    (void)bScaleMatrix;
    TMATMUL_IMPL(cMatrix, aMatrix, bMatrix);
}

template <AccPhase Phase = AccPhase::Unspecified, typename TileRes, typename TileLeft, typename TileLeftScale,
          typename TileRight, typename TileRightScale>
PTO_INTERNAL void TMATMUL_MX_IMPL(TileRes &cOutMatrix, TileRes &cInMatrix, TileLeft &aMatrix,
                                  TileLeftScale &aScaleMatrix, TileRight &bMatrix, TileRightScale &bScaleMatrix)
{
    (void)Phase;
    (void)aScaleMatrix;
    (void)bScaleMatrix;
    TMATMUL_ACC_IMPL(cOutMatrix, cInMatrix, aMatrix, bMatrix);
}

template <AccPhase Phase = AccPhase::Unspecified, typename TileRes, typename TileLeft, typename TileLeftScale,
          typename TileRight, typename TileRightScale, typename TileBias>
PTO_INTERNAL void TMATMUL_MX_IMPL(TileRes &cMatrix, TileLeft &aMatrix, TileLeftScale &aScaleMatrix, TileRight &bMatrix,
                                  TileRightScale &bScaleMatrix, TileBias &biasData)
{
    (void)Phase;
    (void)aScaleMatrix;
    (void)bScaleMatrix;
    TMATMUL_BIAS_IMPL(cMatrix, aMatrix, bMatrix, biasData);
}

template <AccPhase Phase = AccPhase::Unspecified, typename TileRes, typename TileLeft, typename TileLeftScale,
          typename TileRight, typename TileRightScale>
PTO_INTERNAL void TGEMV_MX_IMPL(TileRes &cMatrix, TileLeft &aMatrix, TileLeftScale &aScaleMatrix, TileRight &bMatrix,
                                TileRightScale &bScaleMatrix)
{
    (void)Phase;
    (void)aScaleMatrix;
    (void)bScaleMatrix;
    TGEMV_IMPL(cMatrix, aMatrix, bMatrix);
}

template <AccPhase Phase = AccPhase::Unspecified, typename TileRes, typename TileLeft, typename TileLeftScale,
          typename TileRight, typename TileRightScale>
PTO_INTERNAL void TGEMV_MX_IMPL(TileRes &cOutMatrix, TileRes &cInMatrix, TileLeft &aMatrix, TileLeftScale &aScaleMatrix,
                                TileRight &bMatrix, TileRightScale &bScaleMatrix)
{
    (void)Phase;
    (void)aScaleMatrix;
    (void)bScaleMatrix;
    TGEMV_ACC_IMPL(cOutMatrix, cInMatrix, aMatrix, bMatrix);
}

template <AccPhase Phase = AccPhase::Unspecified, typename TileRes, typename TileLeft, typename TileLeftScale,
          typename TileRight, typename TileRightScale, typename TileBias>
PTO_INTERNAL void TGEMV_MX_IMPL(TileRes &cMatrix, TileLeft &aMatrix, TileLeftScale &aScaleMatrix, TileRight &bMatrix,
                                TileRightScale &bScaleMatrix, TileBias &biasData)
{
    (void)Phase;
    (void)aScaleMatrix;
    (void)bScaleMatrix;
    TGEMV_BIAS_IMPL(cMatrix, aMatrix, bMatrix, biasData);
}
} // namespace pto
#endif
