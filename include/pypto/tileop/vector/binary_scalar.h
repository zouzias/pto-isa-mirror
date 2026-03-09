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
#include "utils/layout.h"
#include "utils/tile_tensor.h"

template <BinaryScalarOp op, typename LastUse, typename T0, typename T1, typename Scalar>
TILEOP void BinaryScalarComputeImpl(T0 dst, T1 src0, Scalar src1) {
    constexpr auto n1 = Std::tuple_element<DIM_1ST, LastUse>::type::value;
    constexpr auto n2 = Std::tuple_element<DIM_2ND, LastUse>::type::value;
    if constexpr (op == BinaryScalarOp::ADD) {
        PTO_WITH_LAST_USE(pto::TADDS(dst, src0, src1), n1, n2);
        return;
    }

    if constexpr (op == BinaryScalarOp::SUB) {
        PTO_WITH_LAST_USE(pto::TADDS(dst, src0, -src1), n1, n2);
        return;
    }

    if constexpr (op == BinaryScalarOp::MUL) {
        PTO_WITH_LAST_USE(pto::TMULS(dst, src0, src1), n1, n2);
        return;
    }

    if constexpr (op == BinaryScalarOp::DIV) {
        PTO_WITH_LAST_USE(pto::TDIVS(dst, src0, src1), n1, n2);
        return;
    }

    if constexpr (op == BinaryScalarOp::MAX) {
        PTO_WITH_LAST_USE(pto::TMAXS(dst, src0, src1), n1, n2);
        return;
    }

    if constexpr (op == BinaryScalarOp::MIN) {
        PTO_WITH_LAST_USE(pto::TMINS(dst, src0, src1), n1, n2);
        return;
    }

    if constexpr (op == BinaryScalarOp::BITWISEAND) {
        pto::TANDS(dst, src0, src1);
        return;
    }

    if constexpr (op == BinaryScalarOp::BITWISEOR) {
        pto::TORS(dst, src0, src1);
        return;
    }

    if constexpr (op == BinaryScalarOp::MOD) {
        pto::TFMODS(dst, src0, src1);
        return;
    }

    if constexpr (op == BinaryScalarOp::LRELU) {
        pto::TLRELU(dst, src0, src1);
        return;
    }

    if constexpr (op == BinaryScalarOp::REM) {
        pto::TREMS(dst, src0, src1);
        return;
    }
}

template <BinaryScalarOp op, typename LastUse, typename T0, typename T1, typename Scalar>
TILEOP void BinaryScalarCompute(T0 dst, T1 src0, Scalar src1) {
    const auto dstLayout = dst.GetLayout();
    auto shape0 = dstLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
    auto shape1 = dstLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto shape2 = dstLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();

    auto dstTile = PtoTile<T0>(dst);
    auto src0Tile = PtoTile<T1>(src0);
    for (LoopVar n0Index = 0; n0Index < shape0; ++n0Index) {
        for (LoopVar n1Index = 0; n1Index < shape1; ++n1Index) {
            for (LoopVar n2Index = 0; n2Index < shape2; ++n2Index) {
                auto tileOffsets = TileOffset(n0Index, n1Index, n2Index);
                dstTile.Assign(dst, tileOffsets);
                src0Tile.Assign(src0, tileOffsets);
                BinaryScalarComputeImpl<op, LastUse>(dstTile.Data(), src0Tile.Data(), src1);
            }
        }
    }
}
#define OP_TILE_OP_ADDS TAddS
template <typename LastUse = LastUse2Dim<0, 0>, typename Scalar, typename T0, typename T1>
TILEOP void TAddS(T0 dst, T1 src0, Scalar src1) {
    BinaryScalarCompute<BinaryScalarOp::ADD, LastUse>(dst, src0, src1);
}

#define OP_TILE_OP_SUBS TSubS
template <typename LastUse = LastUse2Dim<0, 0>, typename Scalar, typename T0, typename T1>
TILEOP void TSubS(T0 dst, T1 src0, Scalar src1) {
    BinaryScalarCompute<BinaryScalarOp::SUB, LastUse>(dst, src0, src1);
}

#define OP_TILE_OP_MULS TMulS
template <typename LastUse = LastUse2Dim<0, 0>, typename Scalar, typename T0, typename T1>
TILEOP void TMulS(T0 dst, T1 src0, Scalar src1) {
    BinaryScalarCompute<BinaryScalarOp::MUL, LastUse>(dst, src0, src1);
}

#define OP_TILE_OP_DIVS TDivS
template <typename LastUse = LastUse2Dim<0, 0>, typename Scalar, typename T0, typename T1>
TILEOP void TDivS(T0 dst, T1 src0, Scalar src1) {
    BinaryScalarCompute<BinaryScalarOp::DIV, LastUse>(dst, src0, src1);
}

#define OP_TILE_OP_MAXS TMaxS
template <typename LastUse = LastUse2Dim<0, 0>, typename Scalar, typename T0, typename T1>
TILEOP void TMaxS(T0 dst, T1 src0, Scalar src1) {
    BinaryScalarCompute<BinaryScalarOp::MAX, LastUse>(dst, src0, src1);
}

#define OP_TILE_OP_MINS TMinS
template <typename LastUse = LastUse2Dim<0, 0>, typename Scalar, typename T0, typename T1>
TILEOP void TMinS(T0 dst, T1 src0, Scalar src1) {
    BinaryScalarCompute<BinaryScalarOp::MIN, LastUse>(dst, src0, src1);
}

#define OP_TILE_OP_LRELU TLReLU
template <typename LastUse = LastUse2Dim<0, 0>, typename Scalar, typename T0, typename T1>
TILEOP void TLReLU(T0 dst, T1 src0, Scalar src1) {
    BinaryScalarCompute<BinaryScalarOp::LRELU, LastUse>(dst, src0, src1);
}

#define OP_TILE_OP_REMS TRemainderS
template <typename Scalar, typename T0, typename T1>
TILEOP void TRemainderS(T0 dst, T1 src0, Scalar src1) {
    BinaryScalarCompute<BinaryScalarOp::REM, LastUse2Dim<0, 0>>(dst, src0, src1);
}

#define OP_TILE_OP_BITWISEANDS TBitwiseAndS
template <typename LastUse = LastUse2Dim<0, 0>, typename Scalar, typename T0, typename T1>
TILEOP void TBitwiseAndS(T0 dst, T1 src0, Scalar src1) {
    BinaryScalarCompute<BinaryScalarOp::BITWISEAND, LastUse>(dst, src0, src1);
}

#define OP_TILE_OP_BITWISEORS TBitwiseOrS
template <typename LastUse = LastUse2Dim<0, 0>, typename Scalar, typename T0, typename T1>
TILEOP void TBitwiseOrS(T0 dst, T1 src0, Scalar src1) {
    BinaryScalarCompute<BinaryScalarOp::BITWISEOR, LastUse>(dst, src0, src1);
}

TILEOP int gcds(int a, int b) {
    if (a < 0) {
        a = 0 - a;
    }
    if (b < 0) {
        b = 0 - b;
    }
    while (a % b != 0) {
        int c = a % b;
        a = b;
        b = c;
    }
    return b;
}

#define OP_TILE_OP_GCDS TGcdS
template <typename Scalar, typename T0, typename T1>
TILEOP void TGcdS(T0 dst, T1 src0, Scalar src1) {
    const auto dstLayout = dst.GetLayout();
    auto shape0 = dstLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
    auto shape1 = dstLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto shape2 = dstLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();
    auto shape3 = dstLayout.template GetShapeDim<DIM_4TH, MAX_DIMS>();
    auto shape4 = dstLayout.template GetShapeDim<DIM_5TH, MAX_DIMS>();
    auto dstStride0 = dstLayout.template GetStrideDim<DIM_1ST, MAX_DIMS>();
    auto dstStride1 = dstLayout.template GetStrideDim<DIM_2ND, MAX_DIMS>();
    auto dstStride2 = dstLayout.template GetStrideDim<DIM_3RD, MAX_DIMS>();
    auto dstStride3 = dstLayout.template GetStrideDim<DIM_4TH, MAX_DIMS>();
    constexpr auto dstTileH = TileOp::GetTensorTileShapeDim<T0, 3, 5>();
    constexpr auto dstTileW = TileOp::GetTensorTileShapeDim<T0, 4, 5>();
    auto src0Addr = (__ubuf__ typename T1::Type *)((uint64_t)(src0.GetAddr()));
    auto dstAddr = (__ubuf__ typename T0::Type *)((uint64_t)(dst.GetAddr()));

    set_flag(PIPE_V, PIPE_S, EVENT_ID7);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID7);
    for (LoopVar n = 0; n < shape0; n++) {
        for (LoopVar j = 0; j < shape1; j++) {
            for (LoopVar k = 0; k < shape2; k++) {
                for (LoopVar m = 0; m < shape3; m++) {
                    for (LoopVar i = 0; i < shape4; i++) {
                        int tmpStride = n * dstStride0 + j * dstStride1 + k * dstStride2 + m * dstStride3 + i;
                        dstAddr[tmpStride] = gcds(src0Addr[tmpStride], src1);
                    }
                }
            }
        }
    }
    set_flag(PIPE_S, PIPE_V, EVENT_ID7);
    wait_flag(PIPE_S, PIPE_V, EVENT_ID7);
}

#define OP_TILE_OP_MODS TModS
template <typename LastUse = LastUse2Dim<0, 0>, typename Scalar, typename T0, typename T1>
TILEOP void TModS(T0 dst, T1 src0, Scalar src1) {
    BinaryScalarCompute<BinaryScalarOp::MOD, LastUse>(dst, src0, src1);
}

template <BinaryScalarOp op, typename T0, typename T1, typename Scalar, typename T2>
TILEOP void BinaryScalarTmpComputeImpl(T0 dst, T1 src0, Scalar src1, T2 tmp) {
    if constexpr (op == BinaryScalarOp::BITWISEXOR) {
        pto::TXORS(dst, src0, src1, tmp);
        return;
    }
}

template <BinaryScalarOp op, typename T0, typename T1, typename Scalar, typename T2>
TILEOP void BinaryScalarTmpCompute(T0 dst, T1 src0, Scalar src1, T2 tmp) {
    const auto dstLayout = dst.GetLayout();
    auto shape0 = dstLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
    auto shape1 = dstLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto shape2 = dstLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();

    auto dstTile = PtoTile<T0>(dst);
    auto src0Tile = PtoTile<T1>(src0);
    auto tmpTile = PtoTile<T2>(tmp);
    for (LoopVar n0Index = 0; n0Index < shape0; ++n0Index) {
        for (LoopVar n1Index = 0; n1Index < shape1; ++n1Index) {
            for (LoopVar n2Index = 0; n2Index < shape2; ++n2Index) {
                auto tileOffsets = TileOffset(n0Index, n1Index, n2Index);
                dstTile.Assign(dst, tileOffsets);
                src0Tile.Assign(src0, tileOffsets);
                tmpTile.Assign(tmp, tileOffsets);
                BinaryScalarTmpComputeImpl<op>(dstTile.Data(), src0Tile.Data(), src1, tmpTile.Data());
            }
        }
    }
}

#define OP_TILE_OP_BITWISEXORS TBitwiseXorS
template <typename Scalar, typename T0, typename T1, typename T2>
TILEOP void TBitwiseXorS(T0 dst, T1 src0, Scalar src1, T2 tmp) {
    BinaryScalarTmpCompute<BinaryScalarOp::BITWISEXOR>(dst, src0, src1, tmp);
}

#define OP_TILE_OP_REMRS TRemainderRS
template <typename Scalar, typename T0, typename T1, typename T2>
TILEOP void TRemainderRS(T0 dst, T1 src0, Scalar src1, T2 tmp) {
    const auto dstLayout = dst.GetLayout();
    auto shape0 = dstLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
    auto shape1 = dstLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto shape2 = dstLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();
    auto dstTile = PtoTile<T0>(dst);
    auto src0Tile = PtoTile<T1>(src0);
    auto tmpTile = PtoTile<T2>(tmp);
    for (LoopVar n0Index = 0; n0Index < shape0; ++n0Index) {
        for (LoopVar n1Index = 0; n1Index < shape1; ++n1Index) {
            for (LoopVar n2Index = 0; n2Index < shape2; ++n2Index) {
                auto tileOffsets = TileOffset(n0Index, n1Index, n2Index);
                dstTile.Assign(dst, tileOffsets);
                src0Tile.Assign(src0, tileOffsets);
                tmpTile.Assign(tmp, tileOffsets);
                pto::TEXPANDS(tmpTile.Data(), src1);
                #ifdef __DAV_V220
                pipe_barrier(PIPE_V);
                #endif
                pto::TREM(dstTile.Data(), tmpTile.Data(), src0Tile.Data());
            }
        }
    }
}
#endif