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
 * \file cast.h
 * \brief
 */

#ifndef TILEOP_TILE_OPERATOR_CAST__H
#define TILEOP_TILE_OPERATOR_CAST__H
#include "pto_tile.h"
#include "utils/layout.h"
#include "utils/tile_tensor.h"

#define OP_TILE_OP_CAST TCast
template <typename LastUse = LastUse2Dim<0, 0>, unsigned Mode, typename T0, typename T1>
TILEOP void TCast(T0 dst, T1 src) {
    constexpr auto shapeSize = Std::tuple_size<typename T0::Shape>::value;
    constexpr size_t expectSize = 5;
    const auto dstLayout = dst.GetLayout();
    const auto srcLayout = src.GetLayout();
    auto shape0 = dstLayout.template GetShapeDim<0, expectSize>();
    auto shape1 = dstLayout.template GetShapeDim<1, expectSize>();
    auto shape2 = dstLayout.template GetShapeDim<2, expectSize>();
    auto shape3 = dstLayout.template GetShapeDim<3, expectSize>();
    auto shape4 = dstLayout.template GetShapeDim<4, expectSize>();
    auto dstStride0 = dstLayout.template GetStrideDim<0, expectSize>();
    auto dstStride1 = dstLayout.template GetStrideDim<1, expectSize>();
    auto dstStride2 = dstLayout.template GetStrideDim<2, expectSize>();
    auto srcStride0 = srcLayout.template GetStrideDim<0, expectSize>();
    auto srcStride1 = srcLayout.template GetStrideDim<1, expectSize>();
    auto srcStride2 = srcLayout.template GetStrideDim<2, expectSize>();

    constexpr auto dstTileH = TileOp::GetTensorTileShapeDim<T0, 3, 5>();
    constexpr auto dstTileW = TileOp::GetTensorTileShapeDim<T0, 4, 5>();
    constexpr auto srcTileH = TileOp::GetTensorTileShapeDim<T1, 3, 5>();
    constexpr auto srcTileW = TileOp::GetTensorTileShapeDim<T1, 4, 5>();
    constexpr auto dstTypeSize = sizeof(typename T0::Type);
    constexpr auto srcTypeSize = sizeof(typename T1::Type);
    constexpr auto n1 = Std::tuple_element<DIM_1ST, LastUse>::type::value;
    constexpr auto n2 = Std::tuple_element<DIM_2ND, LastUse>::type::value;
    for (LoopVar n0Index = 0; n0Index < shape0; ++n0Index) {
        for (LoopVar n1Index = 0; n1Index < shape1; ++n1Index) {
            for (LoopVar n2Index = 0; n2Index < shape2; ++n2Index) {
                using TileDefineDst =
                    pto::Tile<pto::TileType::Vec, typename T0::Type, dstTileH, dstTileW, pto::BLayout::RowMajor, -1, -1>;
                using TileDefineSrc =
                    pto::Tile<pto::TileType::Vec, typename T1::Type, srcTileH, srcTileW, pto::BLayout::RowMajor, -1, -1>;
                TileDefineDst dstTile(shape3, shape4);
                TileDefineSrc srcTile(shape3, shape4);
                auto dstOffset = n0Index * dstStride0 + n1Index * dstStride1 + n2Index * dstStride2;
                auto srcOffset = n0Index * srcStride0 + n1Index * srcStride1 + n2Index * srcStride2;
                pto::TASSIGN(dstTile, (uint64_t)(dst.GetAddr() + dstOffset * dstTypeSize));
                pto::TASSIGN(srcTile, (uint64_t)(src.GetAddr() + srcOffset * srcTypeSize));
                PTO_WITH_LAST_USE(pto::TCVT(dstTile, srcTile, static_cast<pto::RoundMode>(Mode)), n1, n2);
            }
        }
    }
}
#endif