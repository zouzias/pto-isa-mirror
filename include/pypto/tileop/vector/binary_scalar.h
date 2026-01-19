/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file binary_scalar.h
 * \brief
 */

#ifndef TILEOP_TILE_OPERATOR_BINARY_SCALAR__H
#define TILEOP_TILE_OPERATOR_BINARY_SCALAR__H
#include "pto_tile.h"
#include "../utils/layout.h"
#include "../utils/tile_tensor.h"

template <BinaryScalarOp op, typename T0, typename T1, typename Scalar>
TILEOP void BinaryScalarComputeImpl(T0 dst, T1 src0, Scalar src1) {
    if constexpr (op == BinaryScalarOp::ADD) {
        pto::TADDS(dst, src0, src1);
        return;
    }

    if constexpr (op == BinaryScalarOp::SUB) {
        pto::TADDS(dst, src0, -src1);
        return;
    }

    if constexpr (op == BinaryScalarOp::MUL) {
        pto::TMULS(dst, src0, src1);
    }

    if constexpr (op == BinaryScalarOp::DIV) {
        pto::TDIVS(dst, src0, src1);
    }
}

template <BinaryScalarOp op, typename T0, typename T1, typename Scalar>
TILEOP void BinaryScalarCompute(T0 dst, T1 src0, Scalar src1) {
    const auto dstLayout = dst.GetLayout();
    auto shape0 = dstLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
    auto shape1 = dstLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto shape2 = dstLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();

    auto dstTile = PtoTile<T0>(dst);
    auto src0Tile = PtoTile<T1>(src0);
    for (size_t n0Index = 0; n0Index < shape0; ++n0Index) {
        for (size_t n1Index = 0; n1Index < shape1; ++n1Index) {
            for (size_t n2Index = 0; n2Index < shape2; ++n2Index) {
                auto tileOffsets = TileOffset(n0Index, n1Index, n2Index);
                dstTile.Assign(dst, tileOffsets);
                src0Tile.Assign(src0, tileOffsets);
                BinaryScalarComputeImpl<op>(dstTile.Data(), src0Tile.Data(), src1);
            }
        }
    }
}
#define OP_TILE_OP_ADDS TAddS
template <typename Scalar, typename T0, typename T1>
TILEOP void TAddS(T0 dst, T1 src0, Scalar src1) {
    BinaryScalarCompute<BinaryScalarOp::ADD>(dst, src0, src1);
}

#define OP_TILE_OP_SUBS TSubS
template <typename Scalar, typename T0, typename T1>
TILEOP void TSubS(T0 dst, T1 src0, Scalar src1) {
    BinaryScalarCompute<BinaryScalarOp::SUB>(dst, src0, src1);
}

#define OP_TILE_OP_MULS TMulS
template <typename Scalar, typename T0, typename T1>
TILEOP void TMulS(T0 dst, T1 src0, Scalar src1) {
    BinaryScalarCompute<BinaryScalarOp::MUL>(dst, src0, src1);
}

#define OP_TILE_OP_DivS TDivS
template <typename Scalar, typename T0, typename T1>
TILEOP void TDivS(T0 dst, T1 src0, Scalar src1) {
    BinaryScalarCompute<BinaryScalarOp::DIV>(dst, src0, src1);
}
#endif