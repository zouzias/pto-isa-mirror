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

#include <cstdint>

namespace pto {

// Accumulate phase for unit-flag aware TMATMUL paths; Unknown is kept as an alias for compatibility.
enum class AccPhase : uint8_t {
    Unspecified = 0x0,
    Unknown = Unspecified,
    Partial = 0x2,
    Final = 0x3,
};

template <AccPhase Phase = AccPhase::Unspecified, typename TileRes, typename TileLeft, typename TileRight,
    bool cmatrixSource, bool cmatrixInitVal>
__tf__ AICORE void TMatmul(typename TileRes::TileDType __out__ cMatrix, typename TileLeft::TileDType __in__ aMatrix,
    typename TileRight::TileDType __in__ bMatrix, uint16_t m, uint16_t k, uint16_t n) {
    __cc__ typename TileRes::DType *c = (__cc__ typename TileRes::DType *)__cce_get_tile_ptr(cMatrix);
    __ca__ typename TileLeft::DType *a = (__ca__ typename TileLeft::DType *)__cce_get_tile_ptr(aMatrix);
    __cb__ typename TileRight::DType *b = (__cb__ typename TileRight::DType *)__cce_get_tile_ptr(bMatrix);

    mad(c, a, b, m, k, n, 0, 0, static_cast<uint8_t>(Phase), 0, cmatrixSource, cmatrixInitVal);
}

template <AccPhase Phase = AccPhase::Unspecified, typename TileRes, typename TileLeft, typename TileRight,
    bool cmatrixSource, bool cmatrixInitVal>
__tf__ AICORE void TMatmulBias(typename TileRes::TileDType __out__ cMatrix,
    typename TileLeft::TileDType __in__ aMatrix, typename TileRight::TileDType __in__ bMatrix, uint64_t bias,
    uint16_t m, uint16_t k, uint16_t n) {
    __cc__ typename TileRes::DType *c = (__cc__ typename TileRes::DType *)__cce_get_tile_ptr(cMatrix);
    __ca__ typename TileLeft::DType *a = (__ca__ typename TileLeft::DType *)__cce_get_tile_ptr(aMatrix);
    __cb__ typename TileRight::DType *b = (__cb__ typename TileRight::DType *)__cce_get_tile_ptr(bMatrix);
    uint64_t xd = ((uint64_t)c) & 0xffffffffULL | ((bias & 0xffffffffULL) << 32);
    c = (__cc__ typename TileRes::DType *)xd;

    mad(c, a, b, m, k, n, 0, 0, static_cast<uint8_t>(Phase), 0, cmatrixSource, cmatrixInitVal);
}

PTO_INTERNAL void CheckDynamicMmad(uint16_t aMatrixRow, uint16_t aMatrixCol, uint16_t bMatrixCol) {
    constexpr uint16_t elementSize = 4095;
    PTO_ASSERT(aMatrixRow >= 1 && aMatrixRow <= elementSize, "ERROR: The range of valid aMatrixRow is [1, 4095].");
    PTO_ASSERT(aMatrixCol >= 1 && aMatrixCol <= elementSize, "ERROR: The range of valid aMatrixCol is [1, 4095].");
    PTO_ASSERT(bMatrixCol >= 1 && bMatrixCol <= elementSize, "ERROR: The range of valid bMatrixCol is [1, 4095].");
}


template <typename TileRes, typename TileLeft, typename TileRight>
PTO_INTERNAL void CheckMadValid() {
    using AType = typename TileLeft::DType;
    using BType = typename TileRight::DType;
    using CType = typename TileRes::DType;
    static_assert(std::is_same_v<CType, int32_t> || std::is_same_v<CType, uint32_t> || std::is_same_v<CType, half>,
        "TMATMUL: Acc Type support int32_t/uint32_t/half.");
    if constexpr (std::is_same_v<CType, half>) {
        static_assert(std::is_same_v<AType, half> && std::is_same_v<BType, half>,
            "TMATMUL: Left Type and Rigth Type must be half when Acc Type is half.");
    } else if constexpr (std::is_same_v<CType, uint32_t>) {
        static_assert(std::is_same_v<AType, uint8_t> && std::is_same_v<BType, uint8_t>,
            "TMATMUL: Left Type and Rigth Type must be uint8_t when Acc Type is uint32_t.");
    } else if constexpr (std::is_same_v<CType, int32_t>) {
        static_assert((std::is_same_v<AType, int8_t> && std::is_same_v<BType, int8_t>) ||
            (std::is_same_v<AType, uint8_t> && std::is_same_v<BType, int8_t>) ||
            (std::is_same_v<AType, uint8_t> && std::is_same_v<BType, uint8_t>) ||
            (std::is_same_v<AType, int16_t> && std::is_same_v<BType, int8_t>),
            "TMATMUL: [LeftType, RigthType] must be [uint8_t, int8_t] / [uint8_t, uint8_t] / [int16_t, int8_t]"
            " when Acc Type is int32_t.");
    }
    static_assert((TileLeft::Loc == TileType::Left) && (TileRight::Loc == TileType::Right) &&
        (TileRes::Loc == TileType::Acc) && (!TileLeft::isRowMajor) && (TileRight::isRowMajor) &&
        (!TileRes::isRowMajor) && (TileLeft::SFractal == SLayout::RowMajor) &&
        (TileRight::SFractal == SLayout::ColMajor) && (TileRes::SFractal == SLayout::RowMajor),
        "TMATMUL: Non-conforming matrix fractal.");
}

template <AccPhase Phase = AccPhase::Unspecified, typename TileRes, typename TileLeft, typename TileRight>
PTO_INTERNAL void TMATMUL_IMPL(TileRes &cMatrix, TileLeft &aMatrix, TileRight &bMatrix)
{
    // cmatrixInitVal Indicates the initial matrix, 1: the number in C matrix is 0, 0：use the real number in C matrix
    CheckMadValid<TileRes, TileLeft, TileRight>();

    uint16_t m = aMatrix.GetValidRow();
    uint16_t k = aMatrix.GetValidCol();
    uint16_t n = bMatrix.GetValidCol();
    CheckDynamicMmad(m, k, n);

    TMatmul<Phase, TileRes, TileLeft, TileRight, false, true>(cMatrix.data(), aMatrix.data(), bMatrix.data(), m, k, n);
}

template <AccPhase Phase = AccPhase::Unspecified, typename TileRes, typename TileLeft, typename TileRight>
PTO_INTERNAL void TMATMUL_ACC_IMPL(TileRes &cOutMatrix, TileRes &cInMatrix, TileLeft &aMatrix, TileRight &bMatrix)
{
    // cmatrixInitVal Indicates the initial matrix, 1: the number in C matrix is 0, 0：use the real number in C matrix
    CheckMadValid<TileRes, TileLeft, TileRight>();

    uint16_t m = aMatrix.GetValidRow();
    uint16_t k = aMatrix.GetValidCol();
    uint16_t n = bMatrix.GetValidCol();
    CheckDynamicMmad(m, k, n);

    TMatmul<Phase, TileRes, TileLeft, TileRight, false, false>(cOutMatrix.data(), aMatrix.data(), bMatrix.data(), m, k, n);
}

// Convenience overload where the accumulator tile is both the input and output.
template <AccPhase Phase = AccPhase::Unspecified, typename TileRes, typename TileLeft, typename TileRight>
PTO_INTERNAL void TMATMUL_ACC_IMPL(TileRes &cMatrix, TileLeft &aMatrix, TileRight &bMatrix)
{
    TMATMUL_ACC_IMPL<Phase>(cMatrix, cMatrix, aMatrix, bMatrix);
}

template <AccPhase Phase = AccPhase::Unspecified, typename TileRes, typename TileLeft, typename TileRight, typename TileBias>
PTO_INTERNAL void TMATMUL_BIAS_IMPL(TileRes &cMatrix, TileLeft &aMatrix, TileRight &bMatrix, TileBias &biasData)
{
    // cmatrixSource control matrix source, 0: C matrix is in L0C, 1: C matrix is in C2
    // cmatrixInitVal Indicates the initial matrix, 1: the number in C matrix is 0, 0：use the real number in C matrix
    CheckMadValid<TileRes, TileLeft, TileRight>();
    static_assert(std::is_same_v<typename TileRes::DType, typename TileBias::DType>, "No supported bias data type.");
    static_assert((TileBias::Loc == TileType::Bias) && (TileBias::Rows == 1) && (TileBias::isRowMajor),
        "Non-conforming bias fractal.");

    uint16_t m = aMatrix.GetValidRow();
    uint16_t k = aMatrix.GetValidCol();
    uint16_t n = bMatrix.GetValidCol();
    CheckDynamicMmad(m, k, n);

    TMatmulBias<Phase, TileRes, TileLeft, TileRight, true, false>(
        cMatrix.data(), aMatrix.data(), bMatrix.data(), biasData.data(), m, k, n);
}

template <typename TileRes, typename TileLeft, typename TileLeftScale, typename TileRight, typename TileRightScale>
PTO_INTERNAL void TMATMUL_MX_IMPL(TileRes &cMatrix, TileLeft &aMatrix, TileLeftScale &aScaleMatrix, TileRight &bMatrix,
    TileRightScale &bScaleMatrix) {
    // static_assert(false, "no support instruction.");
}

template <typename TileRes, typename TileLeft, typename TileLeftScale, typename TileRight, typename TileRightScale>
PTO_INTERNAL void TMATMUL_MX_IMPL(TileRes &cOutMatrix, TileRes &cInMatrix, TileLeft &aMatrix,
    TileLeftScale &aScaleMatrix, TileRight &bMatrix, TileRightScale &bScaleMatrix) {
    // static_assert(false, "no support instruction.");
}

template <typename TileRes, typename TileLeft, typename TileLeftScale, typename TileRight, typename TileRightScale,
    typename TileBias>
PTO_INTERNAL void TMATMUL_MX_IMPL(TileRes &cMatrix, TileLeft &aMatrix, TileLeftScale &aScaleMatrix, TileRight &bMatrix,
    TileRightScale &bScaleMatrix, TileBias &biasData) {
    // static_assert(false, "no support instruction.");
}
} // namespace pto
#endif