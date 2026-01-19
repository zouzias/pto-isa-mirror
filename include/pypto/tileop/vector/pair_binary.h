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
 * \file pair_binary.h
 * \brief
 */
#ifndef TILEOP_TILE_OPERATOR_PAIR_BINARY_H
#define TILEOP_TILE_OPERATOR_PAIR_BINARY_H

#include "utils/layout.h"
#include "utils/tile_tensor.h"

template <typename T0, typename T1, size_t index = 0, size_t shapeSize = 0>
TILEOP int GetReduceAxisIndex(T0 src0, T1 src1) {
    // 使用validshape进行比较，如果index大于等于shapesize，仍然没有找到reduce轴，则默认R轴为首轴，合并成RA进行处理
    if constexpr (index < shapeSize) {
        const auto src0Layout = src0.GetLayout();
        const auto src1Layout = src1.GetLayout();
        auto src0CurShape = src0Layout.template GetShapeDim<index, shapeSize>();
        auto src1CurShape = src1Layout.template GetShapeDim<index, shapeSize>();
        if (src0CurShape != src1CurShape) {
            return index;
        }
        return GetReduceAxisIndex<T0, T1, index + 1, shapeSize>(src0, src1);
    } else {
        return 0;
    }
}

// 左闭右开
template <int startIndex, int endIndex, typename TileShape>
TILEOP constexpr int GetTileShapeMergeResult() {
    if constexpr (startIndex >= endIndex) {
        return 1;
    } else {
        constexpr auto curShape = Std::tuple_element<startIndex, TileShape>::type::value;
        return curShape * GetTileShapeMergeResult<startIndex + 1, endIndex, TileShape>();
    }
}

template <int reduceAxisIndex, size_t shapeSize, bool isHeight, typename T0>
TILEOP constexpr int GetOneDimensionTileSize() {
    // R轴非尾轴
    if constexpr (reduceAxisIndex < shapeSize - 1) {
        if constexpr (isHeight) {
            return Std::tuple_element<reduceAxisIndex, typename T0::TileShape>::type::value;
        } else {
            return GetTileShapeMergeResult<reduceAxisIndex + 1, shapeSize, typename T0::TileShape>();
        }
    } else {
        if constexpr (isHeight) {
            return TileOp::GetTensorTileShapeDim<T0, 3, 5>();
        } else {
            return TileOp::GetTensorTileShapeDim<T0, 4, 5>();
        }
    }
}

template <int reduceAxisIndex, size_t shapeSize, typename T0>
TILEOP int GetHeightValidSize(T0 tensor) {
    const auto tensorLayout = tensor.GetLayout();
    if constexpr (reduceAxisIndex == shapeSize - 1) {
        return tensorLayout.template GetShapeDim<reduceAxisIndex - 1, shapeSize>();
    } else {
        return tensorLayout.template GetShapeDim<reduceAxisIndex, shapeSize>();
    }
}

template <int reduceAxisIndex, size_t shapeSize, typename T0>
TILEOP int GetWidthValidSize(T0 tensor) {
    const auto tensorLayout = tensor.GetLayout();
    if constexpr (reduceAxisIndex == shapeSize - 1) {
        return tensorLayout.template GetShapeDim<reduceAxisIndex, shapeSize>();
    } else {
        return GetTileShapeMergeResult<reduceAxisIndex + 1, shapeSize, typename T0::TileShape>();
    }
}

template <PairBinaryOp op, typename T0, typename T1, typename T2>
TILEOP void PairBinaryComputeImpl(T0 dst, T1 src0, T2 src1) {
    if constexpr (op == PairBinaryOp::ADD) {
        pto::TPARTADD(dst, src0, src1);
        return;
    }
    if constexpr (op == PairBinaryOp::MAX) {
        pto::TPARTMAX(dst, src0, src1);
        return;
    }
    if constexpr (op == PairBinaryOp::MIN) {
        pto::TPARTMIN(dst, src0, src1);
        return;
    }
}

template <PairBinaryOp op, int reduceAxisIndex, typename T0, typename T1, typename T2>
TILEOP void InnerPairBinaryCompute(T0 dst, T1 src0, T2 src1) {
    constexpr auto shapeSize = Std::tuple_size<typename T0::Shape>::value;
    if constexpr (reduceAxisIndex < shapeSize) {
        const auto dstLayout = dst.GetLayout();
        const auto src0Layout = src0.GetLayout();
        const auto src1Layout = src1.GetLayout();
        constexpr auto dstTypeSize = sizeof(typename T0::Type);
        constexpr auto src0TypeSize = sizeof(typename T1::Type);
        constexpr auto src1TypeSize = sizeof(typename T2::Type);

        size_t dstShape0 = 1;
        size_t dstShape1 = 1;
        size_t dstShape2 = 1;
        size_t dstStride0 = 0;
        size_t dstStride1 = 0;
        size_t dstStride2 = 0;
        size_t src0Stride0 = 0;
        size_t src0Stride1 = 0;
        size_t src0Stride2 = 0;
        size_t src1Stride0 = 0;
        size_t src1Stride1 = 0;
        size_t src1Stride2 = 0;

        if constexpr ((reduceAxisIndex == shapeSize - 1 && reduceAxisIndex >= 2) ||
                      (reduceAxisIndex < shapeSize - 1 && reduceAxisIndex >= 1)) {
            dstShape0 = dstLayout.template GetShapeDim<0, shapeSize>();
            dstStride0 = dstLayout.template GetStrideDim<0, shapeSize>();
            src0Stride0 = src0Layout.template GetStrideDim<0, shapeSize>();
            src1Stride0 = src1Layout.template GetStrideDim<0, shapeSize>();
        }
        if constexpr ((reduceAxisIndex == shapeSize - 1 && reduceAxisIndex >= 3) ||
                      (reduceAxisIndex < shapeSize - 1 && reduceAxisIndex >= 2)) {
            dstShape1 = dstLayout.template GetShapeDim<1, shapeSize>();
            dstStride1 = dstLayout.template GetStrideDim<1, shapeSize>();
            src0Stride1 = src0Layout.template GetStrideDim<1, shapeSize>();
            src1Stride1 = src1Layout.template GetStrideDim<1, shapeSize>();
        }
        if constexpr ((reduceAxisIndex == shapeSize - 1 && reduceAxisIndex >= 4) ||
                      (reduceAxisIndex < shapeSize - 1 && reduceAxisIndex >= 3)) {
            dstShape2 = dstLayout.template GetShapeDim<2, shapeSize>();
            dstStride2 = dstLayout.template GetStrideDim<2, shapeSize>();
            src0Stride2 = src0Layout.template GetStrideDim<2, shapeSize>();
            src1Stride2 = src1Layout.template GetStrideDim<2, shapeSize>();
        }

        constexpr auto dstTileH = GetOneDimensionTileSize<reduceAxisIndex, shapeSize, true, T0>();
        constexpr auto dstTileW = GetOneDimensionTileSize<reduceAxisIndex, shapeSize, false, T0>();
        constexpr auto src0TileH = GetOneDimensionTileSize<reduceAxisIndex, shapeSize, true, T1>();
        constexpr auto src0TileW = GetOneDimensionTileSize<reduceAxisIndex, shapeSize, false, T1>();
        constexpr auto src1TileH = GetOneDimensionTileSize<reduceAxisIndex, shapeSize, true, T2>();
        constexpr auto src1TileW = GetOneDimensionTileSize<reduceAxisIndex, shapeSize, false, T2>();
        int dstShape3;
        int src0Shape3;
        int src1Shape3;
        if constexpr (shapeSize !=1 ) {
            dstShape3 = GetHeightValidSize<reduceAxisIndex, shapeSize, T0>(dst);
            src0Shape3 = GetHeightValidSize<reduceAxisIndex, shapeSize, T1>(src0);
            src1Shape3 = GetHeightValidSize<reduceAxisIndex, shapeSize, T2>(src1);
        }else {
            dstShape3 = 1;
            src0Shape3 = 1;
            src1Shape3 = 1;
        }

        auto dstShape4 = GetWidthValidSize<reduceAxisIndex, shapeSize, T0>(dst);
        auto src0Shape4 = GetWidthValidSize<reduceAxisIndex, shapeSize, T1>(src0);
        auto src1Shape4 = GetWidthValidSize<reduceAxisIndex, shapeSize, T2>(src1);

        using DstTileDefine =
            pto::Tile<pto::TileType::Vec, typename T0::Type, dstTileH, dstTileW, pto::BLayout::RowMajor, -1, -1>;
        using Src0TileDefine =
            pto::Tile<pto::TileType::Vec, typename T0::Type, src0TileH, src0TileW, pto::BLayout::RowMajor, -1, -1>;
        using Src1TileDefine =
            pto::Tile<pto::TileType::Vec, typename T0::Type, src1TileH, src1TileW, pto::BLayout::RowMajor, -1, -1>;

        for (size_t n0Index = 0; n0Index < dstShape0; ++n0Index) {
            for (size_t n1Index = 0; n1Index < dstShape1; ++n1Index) {
                for (size_t n2Index = 0; n2Index < dstShape2; ++n2Index) {
                    DstTileDefine dstTile(dstShape3, dstShape4);
                    Src0TileDefine src0Tile(src0Shape3, src0Shape4);
                    Src1TileDefine src1Tile(src1Shape3, src1Shape4);
                    auto dstOffset = n0Index * dstStride0 + n1Index * dstStride1 + n2Index * dstStride2;
                    auto src0Offset = n0Index * src0Stride0 + n1Index * src0Stride1 + n2Index * src0Stride2;
                    auto src1Offset = n0Index * src1Stride0 + n1Index * src1Stride1 + n2Index * src1Stride2;
                    pto::TASSIGN(dstTile, (uint64_t)(dst.GetAddr() + dstOffset * dstTypeSize));
                    pto::TASSIGN(src0Tile, (uint64_t)(src0.GetAddr() + src0Offset * src0TypeSize));
                    pto::TASSIGN(src1Tile, (uint64_t)(src1.GetAddr() + src1Offset * src1TypeSize));
                    PairBinaryComputeImpl<op>(dstTile, src0Tile, src1Tile);
                }
            }
        }
    }
}

template <PairBinaryOp op, typename T0, typename T1, typename T2>
TILEOP void PairBinaryCompute(T0 dst, T1 src0, T2 src1) {
    constexpr auto shapeSize = Std::tuple_size<typename T0::Shape>::value;
    size_t reduceAxisIndex = GetReduceAxisIndex<T1, T2, 0, shapeSize>(src0, src1);
    if (reduceAxisIndex == 0) {
        InnerPairBinaryCompute<op, 0>(dst, src0, src1);
    } else if (reduceAxisIndex == 1) {
        InnerPairBinaryCompute<op, 1>(dst, src0, src1);
    } else if (reduceAxisIndex == 2) {
        InnerPairBinaryCompute<op, 2>(dst, src0, src1);
    } else if (reduceAxisIndex == 3) {
        InnerPairBinaryCompute<op, 3>(dst, src0, src1);
    } else if (reduceAxisIndex == 4) {
        InnerPairBinaryCompute<op, 4>(dst, src0, src1);
    }
}

#define OP_TILE_OP_PAIRSUM TPairSum
template <typename T0, typename T1, typename T2>
TILEOP void TPairSum(T0 dst, T1 src0, T2 src1) {
    PairBinaryCompute<PairBinaryOp::ADD>(dst, src0, src1);
}

#define OP_TILE_OP_PAIRMAX TPairMax
template <typename T0, typename T1, typename T2>
TILEOP void TPairMax(T0 dst, T1 src0, T2 src1) {
    PairBinaryCompute<PairBinaryOp::MAX>(dst, src0, src1);
}

#define OP_TILE_OP_PAIRMIN TPairMin
template <typename T0, typename T1, typename T2>
TILEOP void TPairMin(T0 dst, T1 src0, T2 src1) {
    PairBinaryCompute<PairBinaryOp::MIN>(dst, src0, src1);
}

#endif