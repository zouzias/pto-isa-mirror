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
 * \file gather.h
 * \brief
 */
#ifndef TILEOP_TILE_OPERATOR_GATHER__H
#define TILEOP_TILE_OPERATOR_GATHER__H
#include "utils/layout.h"
#include "utils/tile_tensor.h"

#define OP_TILE_OP_GATHER_ELEMENT TgatherElement
template <int axis, typename T0, typename T1, typename T2, typename T3>
TILEOP void TgatherElement(T0 dst, T1 src0, T2 src1, T3 tmp) {
    constexpr auto shapeSize = Std::tuple_size<typename T0::Shape>::value;
    constexpr size_t expectSize = 5;
    const auto srcLayout = src0.GetLayout();
    auto n0SrcStride = srcLayout.template GetStrideDim<0, expectSize>();
    auto n1SrcStride = srcLayout.template GetStrideDim<1, expectSize>();
    auto n2SrcStride = srcLayout.template GetStrideDim<2, expectSize>();
    auto n3SrcStride = srcLayout.template GetStrideDim<3, expectSize>();

    const auto idxLayout = src1.GetLayout();
    auto n0IdxShape = idxLayout.template GetShapeDim<0, expectSize>();
    auto n1IdxShape = idxLayout.template GetShapeDim<1, expectSize>();
    auto n2IdxShape = idxLayout.template GetShapeDim<2, expectSize>();
    auto n3IdxShape = idxLayout.template GetShapeDim<3, expectSize>();
    auto n4IdxShape = idxLayout.template GetShapeDim<4, expectSize>();
    auto n0IdxStride = idxLayout.template GetStrideDim<0, expectSize>();
    auto n1IdxStride = idxLayout.template GetStrideDim<1, expectSize>();
    auto n2IdxStride = idxLayout.template GetStrideDim<2, expectSize>();
    auto n3IdxStride = idxLayout.template GetStrideDim<3, expectSize>();

    const auto dstLayout = dst.GetLayout();
    auto n0DstStride = dstLayout.template GetStrideDim<0, expectSize>();
    auto n1DstStride = dstLayout.template GetStrideDim<1, expectSize>();
    auto n2DstStride = dstLayout.template GetStrideDim<2, expectSize>();
    auto n3DstStride = dstLayout.template GetStrideDim<3, expectSize>();

    constexpr auto dstTileW = TileOp::GetTensorTileShapeDim<T0, 4, 5>();
    constexpr auto srcTileW = TileOp::GetTensorTileShapeDim<T1, 4, 5>();
    constexpr auto idxTileW = TileOp::GetTensorTileShapeDim<T2, 4, 5>();

    constexpr bool scalarFlag = (sizeof(typename T2::Type) == 8) ? true : false;
    constexpr auto dstTypeSize = sizeof(typename T0::Type);
    constexpr auto srcTileShape1 = TileOp::GetOutterAxisMergeResult<shapeSize, typename T1::TileShape>();
    using srcTileDefine = pto::Tile<pto::TileType::Vec, typename T1::Type, srcTileShape1, srcTileW, pto::BLayout::RowMajor>;
    using idxTileDefine = pto::Tile<pto::TileType::Vec, typename T2::Type, 1, idxTileW, pto::BLayout::RowMajor, -1, -1>;
    using dstTileDefine = pto::Tile<pto::TileType::Vec, typename T0::Type, 1, dstTileW, pto::BLayout::RowMajor, -1, -1>;
    srcTileDefine srcTile;
    idxTileDefine idxTile(1, n4IdxShape);
    dstTileDefine dstTile(1, n4IdxShape);
    if constexpr (scalarFlag) {
        set_flag(PIPE_V, PIPE_S, EVENT_ID7);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID7);
    }
    auto srcAddr = (__ubuf__ typename T1::Type*)((uint64_t)(src0.GetAddr()));
    auto idxAddr = (__ubuf__ typename T2::Type*)((uint64_t)(src1.GetAddr()));
    auto dstAddr = (__ubuf__ typename T0::Type*)((uint64_t)(dst.GetAddr()));
    auto tmpAddr = (__ubuf__ typename T2::Type*)((uint64_t)(tmp.GetAddr()));
    auto newIdxValue = 0;
    for (int i = 0; i < n0IdxShape; ++i) {
        for (int j = 0; j < n1IdxShape; ++j) {
            for (int k = 0; k < n2IdxShape; ++k) {
                for (int l = 0; l < n3IdxShape; ++l) {
                    if constexpr (scalarFlag == false) {
                        set_flag(PIPE_V, PIPE_S, EVENT_ID7);
                        wait_flag(PIPE_V, PIPE_S, EVENT_ID7);
                    }
                    for (int m = 0; m < n4IdxShape; ++m) {
                        auto dstOffset = i * n0DstStride + j * n1DstStride + k * n2DstStride + l * n3DstStride + m;
                        auto orgIdxValue = 
                            *(idxAddr + i * n0IdxStride + j * n1IdxStride + k * n2IdxStride + l * n3IdxStride + m);
                        if constexpr (axis == 0) {
                            newIdxValue =
                                orgIdxValue * n0SrcStride  + j * n1SrcStride + k * n2SrcStride + l * n3SrcStride + m;
                        } else if constexpr (axis == 1) {
                            newIdxValue =
                                i * n0SrcStride  + orgIdxValue * n1SrcStride + k * n2SrcStride + l * n3SrcStride + m;
                        } else if constexpr (axis == 2) {
                            newIdxValue =
                                i * n0SrcStride  + j * n1SrcStride + orgIdxValue * n2SrcStride + l * n3SrcStride + m;
                        } else if constexpr (axis == 3) {
                            newIdxValue =
                                i * n0SrcStride  + j * n1SrcStride + k * n2SrcStride + orgIdxValue * n3SrcStride + m;
                        } else {
                            newIdxValue =
                                i * n0SrcStride  + j * n1SrcStride + k * n2SrcStride + l * n3SrcStride + orgIdxValue;
                        }
                        if constexpr (scalarFlag) {
                            dstAddr[dstOffset] = srcAddr[newIdxValue];
                        } else {
                            *(tmpAddr + m) = newIdxValue;
                        }
                    }
                    if constexpr (scalarFlag == false) {
                        set_flag(PIPE_S, PIPE_V, EVENT_ID7);
                        wait_flag(PIPE_S, PIPE_V, EVENT_ID7);
                        auto dstAddrOffset = i * n0DstStride + j * n1DstStride + k * n2DstStride + l * n3DstStride;
                        pto::TASSIGN(dstTile, (uint64_t)(dst.GetAddr() + dstAddrOffset * dstTypeSize));
                        pto::TASSIGN(srcTile, (uint64_t)(src0.GetAddr()));
                        pto::TASSIGN(idxTile, (uint64_t)(tmp.GetAddr()));
                        pto::TGATHER(dstTile, srcTile, idxTile);
                    }
                }
            }
        }
    }
    if constexpr (scalarFlag) {
        set_flag(PIPE_S, PIPE_V, EVENT_ID7);
        wait_flag(PIPE_S, PIPE_V, EVENT_ID7);
    }
}

#endif
