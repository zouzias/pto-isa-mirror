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
 * \file cumsum.h
 * \brief
 */

#ifndef TILEOP_TILE_OPERATOR_CUMSUM__H
#define TILEOP_TILE_OPERATOR_CUMSUM__H
#include "utils/layout.h"
#include "utils/tile_tensor.h"
#include <array>

template <typename T0, typename T1, unsigned tileW, unsigned dstTypeSize>
TILEOP void CumSumTool(T0 dst, T1 src, uint64_t tileH, uint64_t tmpStride) {
    using tmpTileDefine = pto::Tile<pto::TileType::Vec, typename T0::Type, 1, tileW, pto::BLayout::RowMajor, 1, tileW>;
    tmpTileDefine tmpDstTile, tmpSrcTile;
    pto::TASSIGN(tmpDstTile, (uint64_t)(dst.GetAddr() + tmpStride));
    pto::TASSIGN(tmpSrcTile, (uint64_t)(src.GetAddr() + tmpStride));
    pto::TMOV(tmpDstTile, tmpSrcTile);

    for (LoopVar i = 1; i < tileH; i++) {
        #ifdef __DAV_V220
        pipe_barrier(PIPE_V);
        #endif
        using TileDefine = pto::Tile<pto::TileType::Vec, typename T0::Type, 1, tileW, pto::BLayout::RowMajor, 1, tileW>;
        TileDefine dst0Tile, dst1Tile, src1Tile;
        pto::TASSIGN(dst0Tile, (uint64_t)(dst.GetAddr() + tmpStride + (i - 1) * tileW * dstTypeSize));
        pto::TASSIGN(dst1Tile, (uint64_t)(dst.GetAddr() + tmpStride + i * tileW * dstTypeSize));
        pto::TASSIGN(src1Tile, (uint64_t)(src.GetAddr() + tmpStride + i * tileW * dstTypeSize));
        pto::TADD(dst1Tile, dst0Tile, src1Tile);
    }
}

template <unsigned axis, typename T0, typename T1>
TILEOP void TCumSum(T0 dst, T1 src) {
    constexpr size_t expectSize = 5;
    constexpr auto shapeSize = Std::tuple_size<typename T0::Shape>::value;
    constexpr auto dstTypeSize = sizeof(typename T0::Type);
    const auto dstLayout = dst.GetLayout();
    auto n0DstStride = dstLayout.template GetStrideDim<0, expectSize>();
    auto n1DstStride = dstLayout.template GetStrideDim<1, expectSize>();
    auto n2DstStride = dstLayout.template GetStrideDim<2, expectSize>();
    auto n3DstStride = dstLayout.template GetStrideDim<3, expectSize>();
    auto n0DstShape = dstLayout.template GetShapeDim<0, expectSize>();
    auto n1DstShape = dstLayout.template GetShapeDim<1, expectSize>();
    auto n2DstShape = dstLayout.template GetShapeDim<2, expectSize>();
    auto n3DstShape = dstLayout.template GetShapeDim<3, expectSize>();
    auto n4DstShape = dstLayout.template GetShapeDim<4, expectSize>();
    constexpr auto dst1RawShape = Std::tuple_element<shapeSize - 1, typename T0::TileShape>::type::value;

    if constexpr (axis == 0) {
        constexpr auto tileH = Std::tuple_element<DIM_1ST, typename T0::TileShape>::type::value;
        constexpr auto tileW = TileOp::GetNonFirstAxisMergeResult<shapeSize, typename T0::TileShape>();
        uint64_t tmpStride = 0;
        CumSumTool<T0, T1, tileW, dstTypeSize>(dst, src, tileH, tmpStride);
        return;
    } else if constexpr (axis == 1) {
        int loops = n0DstShape;
        int tileH = n1DstShape;
        constexpr auto dst2RawShape = Std::tuple_element<shapeSize - 2, typename T0::TileShape>::type::value;
        constexpr auto dst3RawShape = Std::tuple_element<shapeSize - 3, typename T0::TileShape>::type::value;
        constexpr int tileW = dst3RawShape * dst2RawShape * dst1RawShape;

        for (LoopVar loop = 0; loop < loops; loop++) {
            uint64_t tmpStride = loop * n0DstStride * dstTypeSize;
            CumSumTool<T0, T1, tileW, dstTypeSize>(dst, src, tileH, tmpStride);
        }
        return;
    } else if constexpr (axis == 2) {
        constexpr auto dst2RawShape = Std::tuple_element<shapeSize - 2, typename T0::TileShape>::type::value;
        constexpr auto dst3RawShape = Std::tuple_element<shapeSize - 3, typename T0::TileShape>::type::value;
        constexpr int tileH = dst3RawShape;
        constexpr int tileW = dst2RawShape * dst1RawShape;

        for (LoopVar j = 0; j < n0DstShape; j++) {
            for (LoopVar k = 0; k < n1DstShape; k++) {
                uint64_t tmpStride = (j * n0DstStride + k * n1DstStride) * dstTypeSize;
                CumSumTool<T0, T1, tileW, dstTypeSize>(dst, src, tileH, tmpStride);
            }
        }
        return;
    } else if constexpr (axis == 3) {
        constexpr auto dst2RawShape = Std::tuple_element<shapeSize - 2, typename T0::TileShape>::type::value;
        constexpr int tileH = dst2RawShape;
        constexpr int tileW = dst1RawShape;

        for (LoopVar m = 0; m < n0DstShape; m++) {
            for (LoopVar j = 0; j < n1DstShape; j++) {
                for (LoopVar k = 0; k < n2DstShape; k++) {
                    uint64_t tmpStride = (m * n0DstStride + j * n1DstStride + k * n2DstStride) * dstTypeSize;
                    CumSumTool<T0, T1, tileW, dstTypeSize>(dst, src, tileH, tmpStride);
                }
            }
        }
        return;
    } else {
        set_flag(PIPE_V, PIPE_S, EVENT_ID7);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID7);
        auto srcAddr = (__ubuf__ typename T1::Type *)((uint64_t)(src.GetAddr()));
        auto dstAddr = (__ubuf__ typename T0::Type *)((uint64_t)(dst.GetAddr()));

        for (LoopVar n = 0; n < n0DstShape; n++) {
            for (LoopVar j = 0; j < n1DstShape; j++) {
                for (LoopVar k = 0; k < n2DstShape; k++) {
                    for (LoopVar m = 0; m < n3DstShape; m++) {
                        int tmpStride = n * n0DstStride + j * n1DstStride + k * n2DstStride + m * n3DstStride;
                        dstAddr[tmpStride] = srcAddr[tmpStride];
                        for (LoopVar i = 1; i < n4DstShape; i++) {
                            dstAddr[tmpStride + i] = srcAddr[tmpStride + i] + dstAddr[tmpStride + i - 1];
                        }
                    }
                }
            }
        }
        set_flag(PIPE_S, PIPE_V, EVENT_ID7);
        wait_flag(PIPE_S, PIPE_V, EVENT_ID7);
    }
}
#endif