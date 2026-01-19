/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file one_hot.h
 * \brief
 */
#ifndef TILEOP_TILE_OPERATOR_ONEHOT__H
#define TILEOP_TILE_OPERATOR_ONEHOT__H

#include "utils/layout.h"
#include "utils/tile_tensor.h"

template <typename TileData>
PTO_INST void TZEROS(TileData &dst) {
    static_assert(TileData::isRowMajor,"layout of dst must be pto::BLayout::RowMajor.");
    if constexpr (std::is_same_v<typename TileData::DType, int64_t>) {
        using DstTileType = pto::Tile<TileData::Loc,
                                      int32_t,
                                      TileData::Rows,
                                      TileData::Cols * 2,
                                      pto::BLayout::RowMajor,
                                      -1, -1>;
        DstTileType dstTile(dst.RowMaskInternal,dst.ColMaskInternal * 2);
        pto::TASSIGN(dstTile, (uint64_t)dst.data());
        pto::TEXPANDS(dstTile, 0);
    } else {
        pto::TEXPANDS(dst, 0);
    }
}

#define OP_TILE_OP_ONEHOT TOneHot
template <typename DstTensor, typename SrcTensor>
TILEOP void TOneHot(DstTensor dst,SrcTensor src) {
    constexpr auto dstShapeSize = Std::tuple_size<typename DstTensor::Shape>::value;
    constexpr auto srcShapeSize = Std::tuple_size<typename SrcTensor::Shape>::value;
    static_assert(srcShapeSize + 1 == dstShapeSize);

    constexpr size_t expectSize = 5;
    const auto dstLayout = dst.GetLayout();
    auto dstShape0 = dstLayout.template GetShapeDim<0, expectSize>();
    auto dstShape1 = dstLayout.template GetShapeDim<1, expectSize>();
    auto dstShape2 = dstLayout.template GetShapeDim<2, expectSize>();
    auto dstShape3 = dstLayout.template GetShapeDim<3, expectSize>();
    auto dstShape4 = dstLayout.template GetShapeDim<4, expectSize>();
    auto dstStride0 = dstLayout.template GetStrideDim<0, expectSize>();
    auto dstStride1 = dstLayout.template GetStrideDim<1, expectSize>();
    auto dstStride2 = dstLayout.template GetStrideDim<2, expectSize>();

    const auto  srcLayout = src.GetLayout();
    auto srcStride0 = srcLayout.template GetStrideDim<0, expectSize - 1>();
    auto srcStride1 = srcLayout.template GetStrideDim<1, expectSize - 1>();
    auto srcStride2 = srcLayout.template GetStrideDim<2, expectSize - 1>();

    using SrcDtype = typename SrcTensor::Type;
    using DstDtype = typename DstTensor::Type;
    using DstTileShape = typename DstTensor::TileShape;

    constexpr auto dstTileH = TileOp::GetTensorTileShapeDim<DstTensor, 3, 5>();
    constexpr auto dstTileW = TileOp::GetTensorTileShapeDim<DstTensor, 4, 5>();

    using DstTileType = pto::Tile<pto::TileType::Vec, DstDtype, dstTileH, dstTileW, pto::BLayout::RowMajor, -1, -1>;

    if(dstShape3 == 0 || dstShape4 == 0) {
        return;
    }

    for (size_t n0Index = 0; n0Index < dstShape0; ++n0Index) {
        for (size_t n1Index = 0; n1Index < dstShape1; ++n1Index) {
            for (size_t n2Index = 0; n2Index < dstShape2; ++n2Index) {
                DstTileType dstTile(dstShape3, dstShape4);

                auto dstOffset = n0Index * dstStride0 + n1Index * dstStride1 + n2Index * dstStride2;
                auto dstAddr = dst.GetAddr() +  dstOffset * sizeof(DstDtype); // 32B align
                pto::TASSIGN(dstTile, dstAddr);
                TZEROS(dstTile);

                set_flag(PIPE_V, PIPE_S, EVENT_ID7);
                wait_flag(PIPE_V, PIPE_S, EVENT_ID7);

                auto srcOffset = n0Index * srcStride0 + n1Index * srcStride1 + n2Index * srcStride2;
                auto srcPtr = (__ubuf__ SrcDtype *)(src.GetAddr() + srcOffset * sizeof(SrcDtype));
                for(size_t n3Index = 0; n3Index < dstShape3; ++n3Index) {
                    auto dstPtr = (__ubuf__ DstDtype *)(dstAddr + n3Index * dstTileW * sizeof(DstDtype));
                    SrcDtype onePos = srcPtr[n3Index];
                    dstPtr[onePos] = 1;
                }
            }
        }
    }
    set_flag(PIPE_S, PIPE_V, EVENT_ID7);
    wait_flag(PIPE_S, PIPE_V, EVENT_ID7);
}
#endif