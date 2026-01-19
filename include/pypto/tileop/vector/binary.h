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
 * \file binary.h
 * \brief
 */

#ifndef TILEOP_TILE_OPERATOR_BINARY__H
#define TILEOP_TILE_OPERATOR_BINARY__H
#include "pto_tile.h"
#include "../utils/layout.h"
#include "../utils/tile_tensor.h"
#include <pto/common/pto_tile.hpp>
#include <pto/common/pto_instr.hpp>

template <BinaryOp op, TileOp::BroadcastOperand operand, typename T0, typename T1, typename T2>
TILEOP void BinaryComputeImpl(T0 dst, T1 src0, T2 src1) {
    if constexpr (op == BinaryOp::ADD) {
        if constexpr (operand == TileOp::BroadcastOperand::NONE) {
            pto::TADD(dst, src0, src1);
        } else {
            pto::TROWEXPANDADD(dst, src0, src1);
        }
        return;
    }
    if constexpr (op == BinaryOp::SUB) {
        if constexpr (operand == TileOp::BroadcastOperand::NONE) {
            pto::TSUB(dst, src0, src1);
        } else {
            pto::TROWEXPANDSUB(dst, src0, src1);
        }
    }

    if constexpr (op == BinaryOp::MUL) {
        if constexpr (operand == TileOp::BroadcastOperand::NONE) {
            pto::TMUL(dst, src0, src1);
        } else {
            pto::TROWEXPANDMUL(dst, src0, src1);
        }
    }

    if constexpr (op == BinaryOp::DIV) {
        if constexpr (operand == TileOp::BroadcastOperand::NONE) {
            pto::TDIV(dst, src0, src1);
        } else {
            pto::TROWEXPANDDIV(dst, src0, src1);
        }
    }

    if constexpr (op == BinaryOp::MAX) {
        if constexpr (operand == TileOp::BroadcastOperand::NONE) {
            pto::TMAX(dst, src0, src1);
        } else {
            pto::TROWEXPANDMAX(dst, src0, src1);
        }
    }

    if constexpr (op == BinaryOp::MIN) {
        if constexpr (operand == TileOp::BroadcastOperand::NONE) {
            pto::TMIN(dst, src0, src1);
        } else {
            pto::TROWEXPANDMIN(dst, src0, src1);
        }
    }
}

template <BinaryOp op, TileOp::BroadcastOperand operand, typename T0, typename T1, typename T2>
TILEOP void BinaryCompute(T0 dst, T1 src0, T2 src1) {
    constexpr auto shapeSize = Std::tuple_size<typename T0::Shape>::value;
    if constexpr (TileOp::IsConstContinous<T0, T1, T2>() == true) {
        auto dstTile = PtoTile<T0, pto::BLayout::RowMajor, true>().Data();
        auto src0Tile = PtoTile<T1, pto::BLayout::RowMajor, true>().Data();
        auto src1Tile = PtoTile<T2, pto::BLayout::RowMajor, true>().Data();
        pto::TASSIGN(dstTile, (uint64_t)dst.GetAddr());
        pto::TASSIGN(src0Tile, (uint64_t)src0.GetAddr());
        pto::TASSIGN(src1Tile, (uint64_t)src1.GetAddr());
        BinaryComputeImpl<op, operand>(dstTile, src0Tile, src1Tile);
        return;
    }
    const auto dstLayout = dst.GetLayout();
    auto shape0 = dstLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
    auto shape1 = dstLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto shape2 = dstLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();

    auto dstTile = PtoTile<T0>(dst);
    auto src0Tile = PtoTile<T1>(src0);
    auto src1Tile = PtoTile<T2>(src1);
    for (size_t n0Index = 0; n0Index < shape0; ++n0Index) {
        for (size_t n1Index = 0; n1Index < shape1; ++n1Index) {
            for (size_t n2Index = 0; n2Index < shape2; ++n2Index) {
                auto tileOffsets = TileOffset(n0Index, n1Index, n2Index);
                dstTile.Assign(dst, tileOffsets);
                src0Tile.Assign(src0, tileOffsets);
                src1Tile.Assign(src1, tileOffsets);
                BinaryComputeImpl<op, operand>(dstTile.Data(), src0Tile.Data(), src1Tile.Data());
            }
        }
    }
}

#define OP_TILE_OP_ADD TAdd
template <TileOp::BroadcastOperand operand = TileOp::BroadcastOperand::NONE, typename T0, typename T1, typename T2>
TILEOP void TAdd(T0 dst, T1 src0, T2 src1) {
    BinaryCompute<BinaryOp::ADD, operand>(dst, src0, src1);
}

#define OP_TILE_OP_SUB TSub
template <TileOp::BroadcastOperand operand = TileOp::BroadcastOperand::NONE, typename T0, typename T1, typename T2>
TILEOP void TSub(T0 dst, T1 src0, T2 src1) {
    BinaryCompute<BinaryOp::SUB, operand>(dst, src0, src1);
}

#define OP_TILE_OP_MUL TMul
template <TileOp::BroadcastOperand operand = TileOp::BroadcastOperand::NONE, typename T0, typename T1, typename T2>
TILEOP void TMul(T0 dst, T1 src0, T2 src1) {
    BinaryCompute<BinaryOp::MUL, operand>(dst, src0, src1);
}

#define OP_TILE_OP_DIV TDiv
template <TileOp::BroadcastOperand operand = TileOp::BroadcastOperand::NONE, typename T0, typename T1, typename T2>
TILEOP void TDiv(T0 dst, T1 src0, T2 src1) {
    BinaryCompute<BinaryOp::DIV, operand>(dst, src0, src1);
}

#define OP_TILE_OP_MAX TMax
template <TileOp::BroadcastOperand operand = TileOp::BroadcastOperand::NONE, typename T0, typename T1, typename T2>
TILEOP void TMax(T0 dst, T1 src0, T2 src1) {
    BinaryCompute<BinaryOp::MAX, operand>(dst, src0, src1);
}

#define OP_TILE_OP_MIN TMin
template <TileOp::BroadcastOperand operand = TileOp::BroadcastOperand::NONE, typename T0, typename T1, typename T2>
TILEOP void TMin(T0 dst, T1 src0, T2 src1) {
    BinaryCompute<BinaryOp::MIN, operand>(dst, src0, src1);
}
#endif