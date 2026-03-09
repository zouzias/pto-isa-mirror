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
#include "utils/layout.h"
#include "utils/tile_tensor.h"

template <BinaryOp op, TileOp::BroadcastOperand operand, typename LastUse, typename T0, typename T1, typename T2>
TILEOP void BinaryComputeImpl(T0 dst, T1 src0, T2 src1) {
    constexpr auto n1 = Std::tuple_element<DIM_1ST, LastUse>::type::value;
    constexpr auto n2 = Std::tuple_element<DIM_2ND, LastUse>::type::value;
    constexpr auto n3 = Std::tuple_element<DIM_3RD, LastUse>::type::value;
    if constexpr (op == BinaryOp::ADD) {
        if constexpr (operand == TileOp::BroadcastOperand::NONE) {
            PTO_WITH_LAST_USE(pto::TADD(dst, src0, src1), n1, n2, n3);
        } else {
            PTO_WITH_LAST_USE(pto::TROWEXPANDADD(dst, src0, src1), n1, n2, n3);
        }
        return;
    }

    if constexpr (op == BinaryOp::SUB) {
        if constexpr (operand == TileOp::BroadcastOperand::NONE) {
            PTO_WITH_LAST_USE(pto::TSUB(dst, src0, src1), n1, n2, n3);
        } else {
            PTO_WITH_LAST_USE(pto::TROWEXPANDSUB(dst, src0, src1), n1, n2, n3);
        }
    }

    if constexpr (op == BinaryOp::MUL) {
        if constexpr (operand == TileOp::BroadcastOperand::NONE) {
            PTO_WITH_LAST_USE(pto::TMUL(dst, src0, src1), n1, n2, n3);
        } else {
            PTO_WITH_LAST_USE(pto::TROWEXPANDMUL(dst, src0, src1), n1, n2, n3);
        }
    }

    if constexpr (op == BinaryOp::DIV) {
        if constexpr (operand == TileOp::BroadcastOperand::NONE) {
            PTO_WITH_LAST_USE(pto::TDIV(dst, src0, src1), n1, n2, n3);
        } else {
            PTO_WITH_LAST_USE(pto::TROWEXPANDDIV(dst, src0, src1), n1, n2, n3);
        }
    }

    if constexpr (op == BinaryOp::MAX) {
        if constexpr (operand == TileOp::BroadcastOperand::NONE) {
            PTO_WITH_LAST_USE(pto::TMAX(dst, src0, src1), n1, n2, n3);
        } else {
            PTO_WITH_LAST_USE(pto::TROWEXPANDMAX(dst, src0, src1), n1, n2, n3);
        }
    }

    if constexpr (op == BinaryOp::MIN) {
        if constexpr (operand == TileOp::BroadcastOperand::NONE) {
            PTO_WITH_LAST_USE(pto::TMIN(dst, src0, src1), n1, n2, n3);
        } else {
            PTO_WITH_LAST_USE(pto::TROWEXPANDMIN(dst, src0, src1), n1, n2, n3);
        }
    }

    if constexpr (op == BinaryOp::BITWISEAND) {
        pto::TAND(dst, src0, src1);
        return;
    }

    if constexpr (op == BinaryOp::BITWISEOR) {
        pto::TOR(dst, src0, src1);
        return;
    }
    
    if constexpr (op == BinaryOp::REM) {
        pto::TREM(dst, src0, src1);
        return;
    }  

    if constexpr (op == BinaryOp::EXPANDEXPDIF) {
        if constexpr (operand == TileOp::BroadcastOperand::NONE) {
            pto::TCOLEXPANDEXPDIF(dst, src0, src1);
        } else {
            pto::TROWEXPANDEXPDIF(dst, src0, src1);
        }
    }

    if constexpr (op == BinaryOp::MOD) {
        if constexpr (operand == TileOp::BroadcastOperand::NONE) {
            pto::TFMOD(dst, src0, src1);
        } else {
            pto::TROWEXPANDDIV(dst, src0, src1);
            #ifdef __DAV_V220
            pipe_barrier(PIPE_V);
            #endif
            pto::TCVT(dst, dst, pto::RoundMode::CAST_TRUNC);
            #ifdef __DAV_V220
            pipe_barrier(PIPE_V);
            #endif
            pto::TROWEXPANDMUL(dst, dst, src1);
            #ifdef __DAV_V220
            pipe_barrier(PIPE_V);
            #endif
            pto::TROWEXPANDSUB(dst, src0, dst);
        }
        return;
    }
}

template <BinaryOp op, TileOp::BroadcastOperand operand, typename LastUse, typename T0, typename T1, typename T2>
TILEOP void BinaryCompute(T0 dst, T1 src0, T2 src1) {
    constexpr auto shapeSize = Std::tuple_size<typename T0::Shape>::value;
    constexpr auto src0TileW = TileOp::GetTensorTileShapeDim<T1, DIM_5TH, MAX_DIMS>();
    constexpr auto src1TileW = TileOp::GetTensorTileShapeDim<T2, DIM_5TH, MAX_DIMS>();
    if constexpr (TileOp::IsConstContinous<T0, T1, T2>() == true) {
        auto dstTile = PtoTile<T0, pto::BLayout::RowMajor, true>().Data();
        using Src0PtoTile = typename std::conditional<(src0TileW == 1 && operand == TileOp::BroadcastOperand::LEFT_OPERAND),
            PtoTile<T1, pto::BLayout::ColMajor, true>, PtoTile<T1, pto::BLayout::RowMajor, true>>::type;
        using Src1PtoTile = typename std::conditional<(src1TileW == 1 && operand == TileOp::BroadcastOperand::RIGHT_OPERAND),
            PtoTile<T2, pto::BLayout::ColMajor, true>, PtoTile<T2, pto::BLayout::RowMajor, true>>::type;
        auto src0Tile = Src0PtoTile().Data();
        auto src1Tile = Src1PtoTile().Data();
        pto::TASSIGN(dstTile, (uint64_t)dst.GetAddr());
        pto::TASSIGN(src0Tile, (uint64_t)src0.GetAddr());
        pto::TASSIGN(src1Tile, (uint64_t)src1.GetAddr());
        BinaryComputeImpl<op, operand, LastUse>(dstTile, src0Tile, src1Tile);
        return;
    }
    const auto dstLayout = dst.GetLayout();
    auto shape0 = dstLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
    auto shape1 = dstLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto shape2 = dstLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();

    using Src0PtoTile = typename std::conditional<(src0TileW == 1 && operand == TileOp::BroadcastOperand::LEFT_OPERAND),
        PtoTile<T1, pto::BLayout::ColMajor>, PtoTile<T1>>::type;
    using Src1PtoTile = typename std::conditional<(src1TileW == 1 && operand == TileOp::BroadcastOperand::RIGHT_OPERAND),
        PtoTile<T2, pto::BLayout::ColMajor>, PtoTile<T2>>::type;

    auto dstTile = PtoTile<T0>(dst);
    auto src0Tile = Src0PtoTile(src0);
    auto src1Tile = Src1PtoTile(src1);
    for (LoopVar n0Index = 0; n0Index < shape0; ++n0Index) {
        for (LoopVar n1Index = 0; n1Index < shape1; ++n1Index) {
            for (LoopVar n2Index = 0; n2Index < shape2; ++n2Index) {
                auto tileOffsets = TileOffset(n0Index, n1Index, n2Index);
                dstTile.Assign(dst, tileOffsets);
                src0Tile.Assign(src0, tileOffsets);
                src1Tile.Assign(src1, tileOffsets);
                BinaryComputeImpl<op, operand, LastUse>(dstTile.Data(), src0Tile.Data(), src1Tile.Data());
            }
        }
    }
}

#define OP_TILE_OP_ADD TAdd
template <typename LastUse = LastUse3Dim<0, 0, 0>, TileOp::BroadcastOperand operand = TileOp::BroadcastOperand::NONE, typename T0, typename T1, typename T2>
TILEOP void TAdd(T0 dst, T1 src0, T2 src1) {
    BinaryCompute<BinaryOp::ADD, operand, LastUse>(dst, src0, src1);
}

#define OP_TILE_OP_SUB TSub
template <typename LastUse = LastUse3Dim<0, 0, 0>, TileOp::BroadcastOperand operand = TileOp::BroadcastOperand::NONE, typename T0, typename T1, typename T2>
TILEOP void TSub(T0 dst, T1 src0, T2 src1) {
    BinaryCompute<BinaryOp::SUB, operand, LastUse>(dst, src0, src1);
}

#define OP_TILE_OP_MUL TMul
template <typename LastUse = LastUse3Dim<0, 0, 0>, TileOp::BroadcastOperand operand = TileOp::BroadcastOperand::NONE, typename T0, typename T1, typename T2>
TILEOP void TMul(T0 dst, T1 src0, T2 src1) {
    BinaryCompute<BinaryOp::MUL, operand, LastUse>(dst, src0, src1);
}

#define OP_TILE_OP_DIV TDiv
template <typename LastUse = LastUse3Dim<0, 0, 0>, TileOp::BroadcastOperand operand = TileOp::BroadcastOperand::NONE, typename T0, typename T1, typename T2>
TILEOP void TDiv(T0 dst, T1 src0, T2 src1) {
    BinaryCompute<BinaryOp::DIV, operand, LastUse>(dst, src0, src1);
}

#define OP_TILE_OP_MAX TMax
template <typename LastUse = LastUse3Dim<0, 0, 0>, TileOp::BroadcastOperand operand = TileOp::BroadcastOperand::NONE, typename T0, typename T1, typename T2>
TILEOP void TMax(T0 dst, T1 src0, T2 src1) {
    BinaryCompute<BinaryOp::MAX, operand, LastUse>(dst, src0, src1);
}

#define OP_TILE_OP_MIN TMin
template <typename LastUse = LastUse3Dim<0, 0, 0>, TileOp::BroadcastOperand operand = TileOp::BroadcastOperand::NONE, typename T0, typename T1, typename T2>
TILEOP void TMin(T0 dst, T1 src0, T2 src1) {
    BinaryCompute<BinaryOp::MIN, operand, LastUse>(dst, src0, src1);
}

#define OP_TILE_OP_REM TRem
template <typename LastUse = LastUse3Dim<0, 0, 0>, TileOp::BroadcastOperand operand = TileOp::BroadcastOperand::NONE, typename T0, typename T1, typename T2>
TILEOP void TRemainder(T0 dst, T1 src0, T2 src1) {
    BinaryCompute<BinaryOp::REM, operand, LastUse>(dst, src0, src1);
}

#define OP_TILE_OP_BITWISEAND TBitwiseAnd
template <typename LastUse = LastUse3Dim<0, 0, 0>, TileOp::BroadcastOperand operand = TileOp::BroadcastOperand::NONE, typename T0, typename T1, typename T2>
TILEOP void TBitwiseAnd(T0 dst, T1 src0, T2 src1) {
    BinaryCompute<BinaryOp::BITWISEAND, operand, LastUse>(dst, src0, src1);
}

#define OP_TILE_OP_BITWISEOR TBitwiseOr
template <typename LastUse = LastUse3Dim<0, 0, 0>, TileOp::BroadcastOperand operand = TileOp::BroadcastOperand::NONE, typename T0, typename T1, typename T2>
TILEOP void TBitwiseOr(T0 dst, T1 src0, T2 src1) {
    BinaryCompute<BinaryOp::BITWISEOR, operand, LastUse>(dst, src0, src1);
}

#define OP_TILE_OP_EXPANDEXPDIF TExpandExpDif
template <TileOp::BroadcastOperand operand = TileOp::BroadcastOperand::NONE, typename T0, typename T1, typename T2>
TILEOP void TExpandExpDif(T0 dst, T1 src0, T2 src1) {
    BinaryCompute<BinaryOp::EXPANDEXPDIF, operand, LastUse3Dim<0, 0, 0>>(dst, src0, src1);
}

TILEOP int gcd(int a, int b) {
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

#define OP_TILE_OP_GCD TGcd
template <TileOp::BroadcastOperand operand = TileOp::BroadcastOperand::NONE, typename T0, typename T1, typename T2>
TILEOP void TGcd(T0 dst, T1 src0, T2 src1) {
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
    auto src1Addr = (__ubuf__ typename T2::Type *)((uint64_t)(src1.GetAddr()));
    auto dstAddr = (__ubuf__ typename T0::Type *)((uint64_t)(dst.GetAddr()));

    set_flag(PIPE_V, PIPE_S, EVENT_ID7);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID7);
    for (LoopVar n = 0; n < shape0; n++) {
        for (LoopVar j = 0; j < shape1; j++) {
            for (LoopVar k = 0; k < shape2; k++) {
                for (LoopVar m = 0; m < shape3; m++) {
                    for (LoopVar i = 0; i < shape4; i++) {
                        int tmpStride = n * dstStride0 + j * dstStride1 + k * dstStride2 + m * dstStride3 + i;
                        dstAddr[tmpStride] = gcd(src0Addr[tmpStride], src1Addr[tmpStride]);
                    }
                }
            }
        }
    }
    set_flag(PIPE_S, PIPE_V, EVENT_ID7);
    wait_flag(PIPE_S, PIPE_V, EVENT_ID7);
}

#define OP_TILE_OP_Mod TMod
template <typename LastUse = LastUse3Dim<0, 0, 0>, TileOp::BroadcastOperand operand = TileOp::BroadcastOperand::NONE, typename T0, typename T1, typename T2>
TILEOP void TMod(T0 dst, T1 src0, T2 src1) {
    BinaryCompute<BinaryOp::MOD, operand, LastUse>(dst, src0, src1);
}

template <typename T0, typename T1, typename T2, typename T3>
TILEOP void SwitchPowResult(const size_t offset, T0 dst, T1 src0, T2 src1, T3 tmp) {
    float a = src0.GetValue(offset);
    float b = src1.GetValue(offset);
    if (b == 0) {
        dst.SetValue(offset, static_cast<float>(1));
        return;
    }
    if (a == 0) {
        if (b > 0) {
            dst.SetValue(offset, static_cast<float>(0));
            return;
        }
        dst.SetValue(offset, static_cast<float>(1) / static_cast<float>(0));
        return;
    }
    if (TileOp::IsInteger(b)) {
        if (static_cast<int>(b) % 2 != 0 && a < 0) {
            dst.SetValue(offset, -tmp.GetValue(offset));
        } else {
            dst.SetValue(offset, tmp.GetValue(offset));
        }
    }
}

template <typename T0, typename T1, typename T2, typename T3>
TILEOP void SwitchPowResult(T0 dst, T1 src0, T2 src1, T3 tmp) {
    constexpr size_t rowStride = src0.RowStride;
    constexpr size_t colStride = src0.ColStride;
    for (size_t n0 = 0, validRow = src0.GetValidRow(); n0 < validRow; ++n0) {
        for (size_t n1 = 0, validCol = src0.GetValidCol(); n1 < validCol; ++n1) {
            auto offset = n0 * rowStride + n1 * colStride;
            SwitchPowResult(offset, dst, src0, src1, tmp);
        }
    }
}

TILEOP int QuickPow(int a, int b) {
    if (b == 0) {
        return 1;
    }
    if (a == 0) {
        return 0;
    }
    if (a == 1) {
        return 1;
    }
    if (a == -1) {
        if ((b & 1) == 0) {
            return 1;
        }
        return -1;
    }
    if (b < 0) {
        return 0;
    }
    int result = 1;
    int tmp = a;
    while (b != 0) {
        if ((b & 1) != 0) {
            result *= tmp;
        }
        b /= 2;
        tmp *= tmp;
    }
    return result;
}

template <typename T0, typename T1, typename T2>
TILEOP void IntegerPow(T0 dst, T1 src0, T2 src1) {
    constexpr size_t rowStride = src0.RowStride;
    constexpr size_t colStride = src0.ColStride;
    for (size_t n0 = 0, validRow = src0.GetValidRow(); n0 < validRow; ++n0) {
        for (size_t n1 = 0, validCol = src0.GetValidCol(); n1 < validCol; ++n1) {
            auto offset = n0 * rowStride + n1 * colStride;
            int a = src0.GetValue(offset);
            int b = src1.GetValue(offset);
            dst.SetValue(offset, QuickPow(a, b));
        }
    }
}

template <typename T0, typename T1, typename T2, typename T3>
TILEOP void CalcPow(T0 dst, T1 src0, T2 src1, T3 tmp) {
    if constexpr (std::is_same_v<typename T1::DType, int32_t>) {
        set_flag(PIPE_V, PIPE_S, EVENT_ID7);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID7);
        IntegerPow(dst, src0, src1);
        set_flag(PIPE_S, PIPE_V, EVENT_ID7);
        wait_flag(PIPE_S, PIPE_V, EVENT_ID7);
    } else {
        pto::TABS(tmp, src0);
        #ifdef __DAV_V220
            pipe_barrier(PIPE_V);
        #endif
        pto::TLOG(tmp, tmp);
        pto::TLOG(dst, src0);
        #ifdef __DAV_V220
            pipe_barrier(PIPE_V);
        #endif
        pto::TMUL(tmp, tmp, src1);
        pto::TMUL(dst, dst, src1);
        #ifdef __DAV_V220
            pipe_barrier(PIPE_V);
        #endif
        pto::TEXP(tmp, tmp);
        pto::TEXP(dst, dst);
        #ifdef __DAV_V220
            pipe_barrier(PIPE_V);
        #endif
        set_flag(PIPE_V, PIPE_S, EVENT_ID7);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID7);
        SwitchPowResult(dst, src0, src1, tmp);
        set_flag(PIPE_S, PIPE_V, EVENT_ID7);
        wait_flag(PIPE_S, PIPE_V, EVENT_ID7);
    }
}

template <BinaryOp op, TileOp::BroadcastOperand operand, typename T0, typename T1, typename T2, typename T3>
TILEOP void BinaryTmpComputeImpl(T0 dst, T1 src0, T2 src1, T3 tmp) {
    if constexpr (op == BinaryOp::BITWISEXOR) {
        pto::TXOR(dst, src0, src1, tmp);
        return;
    }
    if constexpr (op == BinaryOp::POW) {
        CalcPow(dst, src0, src1, tmp);
        return;
    }
}

template <BinaryOp op, TileOp::BroadcastOperand operand, typename T0, typename T1, typename T2, typename T3>
TILEOP void BinaryTmpCompute(T0 dst, T1 src0, T2 src1, T3 tmp) {
    constexpr auto shapeSize = Std::tuple_size<typename T0::Shape>::value;
    if constexpr (TileOp::IsConstContinous<T0, T1, T2, T3>() == true) {
        auto dstTile = PtoTile<T0, pto::BLayout::RowMajor, true>().Data();
        auto src0Tile = PtoTile<T1, pto::BLayout::RowMajor, true>().Data();
        auto src1Tile = PtoTile<T2, pto::BLayout::RowMajor, true>().Data();
        auto tmpTile = PtoTile<T3, pto::BLayout::RowMajor, true>().Data();
        pto::TASSIGN(dstTile, (uint64_t)dst.GetAddr());
        pto::TASSIGN(src0Tile, (uint64_t)src0.GetAddr());
        pto::TASSIGN(src1Tile, (uint64_t)src1.GetAddr());
        pto::TASSIGN(tmpTile, (uint64_t)tmp.GetAddr());
        BinaryTmpComputeImpl<op, operand>(dstTile, src0Tile, src1Tile, tmpTile);
        return;
    }
    const auto dstLayout = dst.GetLayout();
    auto shape0 = dstLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
    auto shape1 = dstLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto shape2 = dstLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();

    auto dstTile = PtoTile<T0>(dst);
    auto src0Tile = PtoTile<T1>(src0);
    auto src1Tile = PtoTile<T2>(src1);
    auto tmpTile = PtoTile<T3>(tmp);
    for (LoopVar n0Index = 0; n0Index < shape0; ++n0Index) {
        for (LoopVar n1Index = 0; n1Index < shape1; ++n1Index) {
            for (LoopVar n2Index = 0; n2Index < shape2; ++n2Index) {
                auto tileOffsets = TileOffset(n0Index, n1Index, n2Index);
                dstTile.Assign(dst, tileOffsets);
                src0Tile.Assign(src0, tileOffsets);
                src1Tile.Assign(src1, tileOffsets);
                tmpTile.Assign(tmp, tileOffsets);
                BinaryTmpComputeImpl<op, operand>(dstTile.Data(), src0Tile.Data(), src1Tile.Data(), tmpTile.Data());
            }
        }
    }
}

#define OP_TILE_OP_BITWISEXOR TBitwiseXor
template <TileOp::BroadcastOperand operand = TileOp::BroadcastOperand::NONE, typename T0, typename T1, typename T2, typename T3>
TILEOP void TBitwiseXor(T0 dst, T1 src0, T2 src1, T3 tmp) {
    BinaryTmpCompute<BinaryOp::BITWISEXOR, operand>(dst, src0, src1, tmp);
}

#define OP_TILE_OP_POW TPow
template <TileOp::BroadcastOperand operand = TileOp::BroadcastOperand::NONE, typename T0, typename T1, typename T2,  typename T3>
TILEOP void TPow(T0 dst, T1 src0, T2 src1, T3 tmp) {
    static_assert(std::is_same_v<typename T1::Type, float> || std::is_same_v<typename T1::Type, int32_t>);
    BinaryTmpCompute<BinaryOp::POW, operand>(dst, src0, src1, tmp);
}

#endif