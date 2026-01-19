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
 * \file vec_unary.h
 * \brief
 */

#ifndef TILEOP_TILE_OPERATOR_VEC_UNARY__H
#define TILEOP_TILE_OPERATOR_VEC_UNARY__H
#include "pto_tile.h"
#include "utils/layout.h"
#include "utils/tile_tensor.h"

template <UnaryOp op, typename T0, typename T1>
TILEOP void UnaryComputeImpl(T0 dst, T1 src) {
    if constexpr (op == UnaryOp::EXP) {
        pto::TEXP(dst, src);
        return;
    }
    if constexpr (op == UnaryOp::RSQRT) {
        pto::TRSQRT(dst, src);
        return;
    }
    if constexpr (op == UnaryOp::SQRT) {
        pto::TSQRT(dst, src);
        return;
    }
    if constexpr (op == UnaryOp::BRCB) {
        pto::TROWEXPAND(dst, src);
        return;
    }
    if constexpr (op == UnaryOp::ABS) {
        pto::TABS(dst, src);
        return;
    }
}

template <UnaryOp op, typename T0, typename T1>
TILEOP void UnaryCompute(T0 dst, T1 src) {
    if constexpr (TileOp::IsConstContinous<T0, T1>() == true) {
        auto dstTile = PtoTile<T0, pto::BLayout::RowMajor, true>().Data();
        auto srcTile = PtoTile<T1, pto::BLayout::RowMajor, true>().Data();
        pto::TASSIGN(dstTile, (uint64_t)dst.GetAddr());
        pto::TASSIGN(srcTile, (uint64_t)src.GetAddr());
        UnaryComputeImpl<op>(dstTile, srcTile);
        return;
    }
    const auto dstLayout = dst.GetLayout();
    auto shape0 = dstLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
    auto shape1 = dstLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto shape2 = dstLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();

    auto dstTile = PtoTile<T0>(dst);
    auto srcTile = PtoTile<T1>(src);
    for (size_t n0Index = 0; n0Index < shape0; ++n0Index) {
        for (size_t n1Index = 0; n1Index < shape1; ++n1Index) {
            for (size_t n2Index = 0; n2Index < shape2; ++n2Index) {
                auto tileOffsets = TileOffset(n0Index, n1Index, n2Index);
                dstTile.Assign(dst, tileOffsets);
                srcTile.Assign(src, tileOffsets);
                UnaryComputeImpl<op>(dstTile.Data(), srcTile.Data());
            }
        }
    }
}

#define OP_TILE_OP_EXP TExp
template <typename T0, typename T1>
TILEOP void BrcbCompute(T0 dst, T1 src) {
    const auto dstLayout = dst.GetLayout();
    auto shape0 = dstLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
    auto shape1 = dstLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto shape2 = dstLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();
    auto dstStride0 = dstLayout.template GetStrideDim<DIM_1ST, MAX_DIMS>();
    auto dstStride1 = dstLayout.template GetStrideDim<DIM_2ND, MAX_DIMS>();
    auto dstStride2 = dstLayout.template GetStrideDim<DIM_3RD, MAX_DIMS>();
    const auto srcLayout = src.GetLayout();
    auto srcStride0 = srcLayout.template GetStrideDim<DIM_1ST, MAX_DIMS>();
    auto srcStride1 = srcLayout.template GetStrideDim<DIM_2ND, MAX_DIMS>();
    auto srcStride2 = srcLayout.template GetStrideDim<DIM_3RD, MAX_DIMS>();

    constexpr auto srcTileH = TileOp::GetTensorTileShapeDim<T1, DIM_4TH, MAX_DIMS>();
    constexpr auto srcTileW = TileOp::GetTensorTileShapeDim<T1, DIM_5TH, MAX_DIMS>();
    constexpr auto dstTileH = TileOp::GetTensorTileShapeDim<T0, DIM_4TH, MAX_DIMS>();
    constexpr auto dstTileW = TileOp::GetTensorTileShapeDim<T0, DIM_5TH, MAX_DIMS>();

    using DstTileDefine =pto::Tile<pto::TileType::Vec, typename T0::Type, dstTileH, dstTileW, pto::BLayout::RowMajor>;
    using SrcTileDefine = typename std::conditional<(srcTileW == 1), 
        pto::Tile<pto::TileType::Vec, typename T1::Type, srcTileH, srcTileW, pto::BLayout::ColMajor>,
        pto::Tile<pto::TileType::Vec, typename T1::Type, srcTileW, srcTileH, pto::BLayout::ColMajor>>::type;

    SrcTileDefine srcTile;
    DstTileDefine dstTile;
    for (size_t n0Index = 0; n0Index < shape0; ++n0Index) {
        for (size_t n1Index = 0; n1Index < shape1; ++n1Index) {
            for (size_t n2Index = 0; n2Index < shape2; ++n2Index) {
                auto dstTileOffsets = n0Index * dstStride0 + n1Index * dstStride1 + n2Index * dstStride2;
                auto srcTileOffsets = n0Index * srcStride0 + n1Index * srcStride1 + n2Index * srcStride2;
                pto::TASSIGN(dstTile, (uint64_t)(dst.GetAddr() + dstTileOffsets * sizeof(typename T0::Type)));
                pto::TASSIGN(srcTile, (uint64_t)(src.GetAddr() + srcTileOffsets * sizeof(typename T1::Type)));
                UnaryComputeImpl<UnaryOp::BRCB>(dstTile, srcTile);
            }
        }
    }
}

template <typename T0, typename T1>
TILEOP void TExp(T0 dst, T1 src) {
    UnaryCompute<UnaryOp::EXP>(dst, src);
}

#define OP_TILE_OP_RSQRT TRsqrt
template <typename T0, typename T1>
TILEOP void TRsqrt(T0 dst, T1 src) {
    UnaryCompute<UnaryOp::RSQRT>(dst, src);
}

#define OP_TILE_OP_SQRT TSqrt
template <typename T0, typename T1>
TILEOP void TSqrt(T0 dst, T1 src) {
    UnaryCompute<UnaryOp::SQRT>(dst, src);
}

template <typename T0, typename T1>
TILEOP void Tbrcb(T0 dst, T1 src) {
    BrcbCompute(dst, src);
}

template <typename T0, typename T1>
TILEOP void TAbs(T0 dst, T1 src) {
    UnaryCompute<UnaryOp::ABS>(dst, src);
}
#endif