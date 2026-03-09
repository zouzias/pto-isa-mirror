/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file copysign.h
 * \brief
 */

#ifndef TILEOP_TILE_OPERATOR_COPYSIGN__H
#define TILEOP_TILE_OPERATOR_COPYSIGN__H
#include "pto_tile.h"
#include "utils/layout.h"
#include "utils/tile_tensor.h"
#include "tileop_common.h"

template <typename U>
using select_bit = std::conditional_t<sizeof(typename U::Type) == 4, uint32_t, uint16_t>;

template <size_t STRIDE, typename T0, typename T1>
TILEOP void TAND16B(T0 &src, T1 &tmp) {
    static_assert(T0::isRowMajor && T1::isRowMajor,"layout of src and tmp must be pto::BLayout::RowMajor.");
    if constexpr (std::is_same_v<typename T0::DType, uint16_t>) {
        pto::TAND(src, src, tmp);
    } else {
        using T0Uint16 = pto::Tile<T0::Loc, uint16_t, T0::Rows, T0::Cols * STRIDE, pto::BLayout::RowMajor, -1, -1>;
        using T1Uint16 = pto::Tile<T1::Loc, uint16_t, T1::Rows, T1::Cols * STRIDE, pto::BLayout::RowMajor, -1, -1>;
        T0Uint16 srcTileUint16(src.RowMaskInternal, src.ColMaskInternal * STRIDE);
        T1Uint16 tmpTileUint16(tmp.RowMaskInternal, tmp.ColMaskInternal * STRIDE);
        pto::TASSIGN(srcTileUint16, (uint64_t)src.data());
        pto::TASSIGN(tmpTileUint16, (uint64_t)tmp.data());
        pto::TAND(srcTileUint16, srcTileUint16, tmpTileUint16);
    }
}

template <size_t STRIDE, typename T0, typename T1, typename T2>
TILEOP void TOR16B(T0 &dst, T1 &src0, T2 &src1) {
    static_assert(T0::isRowMajor && T1::isRowMajor,"layout of src and tmp must be pto::BLayout::RowMajor.");
    if constexpr (std::is_same_v<typename T0::DType, uint16_t>) {
        pto::TOR(dst, src0, src1);
    } else {
        using T0Uint16 = pto::Tile<T0::Loc, uint16_t, T0::Rows, T0::Cols * STRIDE, pto::BLayout::RowMajor, -1, -1>;
        using T1Uint16 = pto::Tile<T1::Loc, uint16_t, T1::Rows, T1::Cols * STRIDE, pto::BLayout::RowMajor, -1, -1>;
        using T2Uint16 = pto::Tile<T2::Loc, uint16_t, T2::Rows, T2::Cols * STRIDE, pto::BLayout::RowMajor, -1, -1>;
        T0Uint16 dstTileUint16(dst.RowMaskInternal, dst.ColMaskInternal * STRIDE);
        T1Uint16 src0TileUint16(src0.RowMaskInternal, src0.ColMaskInternal * STRIDE);
        T2Uint16 src1TileUint16(src1.RowMaskInternal, src1.ColMaskInternal * STRIDE);
        pto::TASSIGN(dstTileUint16, (uint64_t)dst.data());
        pto::TASSIGN(src0TileUint16, (uint64_t)src0.data());
        pto::TASSIGN(src1TileUint16, (uint64_t)src1.data());
        pto::TOR(dstTileUint16, src0TileUint16, src1TileUint16);
    }
}


template <uint16_t MASK_16B, uint32_t MASK_32B, typename T>
TILEOP void TMASKS(T &tmpMask) {
    if constexpr (std::is_same_v<typename T::DType, uint32_t>){
        pto::TEXPANDS(tmpMask, MASK_32B);
    } else {
        pto::TEXPANDS(tmpMask, MASK_16B);
    }
}

#define OP_TILE_OP_COPYSIGN TCopysign
template <typename T0, typename T1, typename T2, typename T3>
TILEOP void TCopysign(T0 dst, T1 src0, T2 src1, T3 tmp) {
    const auto dstLayout = dst.GetLayout();
    auto shape0 = dstLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
    auto shape1 = dstLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto shape2 = dstLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();
    auto shape3 = dstLayout.template GetShapeDim<DIM_4TH, MAX_DIMS>();
    auto shape4 = dstLayout.template GetShapeDim<DIM_5TH, MAX_DIMS>();
    auto dstStride0 = dstLayout.template GetStrideDim<DIM_1ST, MAX_DIMS>();
    auto dstStride1 = dstLayout.template GetStrideDim<DIM_2ND, MAX_DIMS>();
    auto dstStride2 = dstLayout.template GetStrideDim<DIM_3RD, MAX_DIMS>();
    constexpr auto dstTileH = TileOp::GetTensorTileShapeDim<T0, DIM_4TH, MAX_DIMS>();
    constexpr auto dstTileW = TileOp::GetTensorTileShapeDim<T0, DIM_5TH, MAX_DIMS>();

    constexpr uint32_t VALUEMASK32B = 0x7FFFFFFF;
    constexpr uint16_t VALUEMASK16B = 0x7FFF;
    constexpr uint32_t SIGNMASK32B = 0x80000000;
    constexpr uint16_t SIGNMASK16B = 0x8000;
    using T = select_bit<T0>;
    constexpr auto STRIDE = sizeof(typename T0::Type) / 2;


    using dstTileDefine = pto::Tile<pto::TileType::Vec, T, dstTileH, dstTileW, pto::BLayout::RowMajor, -1, -1>;
    using src0TileDefine = pto::Tile<pto::TileType::Vec, T, dstTileH, dstTileW, pto::BLayout::RowMajor, -1, -1>;
    using src1TileDefine = pto::Tile<pto::TileType::Vec, T, dstTileH, dstTileW, pto::BLayout::RowMajor, -1, -1>;
    using TmpMaskDefine = pto::Tile<pto::TileType::Vec, T, dstTileH, dstTileW, pto::BLayout::RowMajor, -1, -1>;

    auto dstAddr = (__ubuf__ typename T0::Type *)((uint64_t)(dst.GetAddr()));
    auto src0Addr = (__ubuf__ typename T1::Type *)((uint64_t)(src0.GetAddr()));
    auto src1Addr = (__ubuf__ typename T2::Type *)((uint64_t)(src1.GetAddr()));
    auto tmpAddr = (__ubuf__ typename T3::Type *)((uint64_t)(tmp.GetAddr()));

    dstTileDefine dstTile(shape3, shape4);
    src0TileDefine src0Tile(shape3, shape4);
    src1TileDefine src1Tile(shape3, shape4);
    TmpMaskDefine tmpTile(shape3, shape4);

    if(shape3 == 0 || shape4 == 0) {
        return;
    }

    for (LoopVar n0Index = 0; n0Index < shape0; ++n0Index) {
        for (LoopVar n1Index = 0; n1Index < shape1; ++n1Index) {
            for (LoopVar n2Index = 0; n2Index < shape2; ++n2Index) {
                auto tileOffsets = n0Index * dstStride0 + n1Index * dstStride1 + n2Index * dstStride2;
                pto::TASSIGN(dstTile, (uint64_t)(dstAddr + tileOffsets));
                pto::TASSIGN(src0Tile, (uint64_t)(src0Addr + tileOffsets));
                pto::TASSIGN(src1Tile, (uint64_t)(src1Addr + tileOffsets));
                pto::TASSIGN(tmpTile, (uint64_t)(tmpAddr + tileOffsets));
                TMASKS<VALUEMASK16B, VALUEMASK32B>(tmpTile);
                #ifdef __DAV_V220
                    pipe_barrier(PIPE_V);
                #endif
                TAND16B<STRIDE>(src0Tile, tmpTile);
                #ifdef __DAV_V220
                    pipe_barrier(PIPE_V);
                #endif
                TMASKS<SIGNMASK16B, SIGNMASK32B>(tmpTile);
                #ifdef __DAV_V220
                    pipe_barrier(PIPE_V);
                #endif
                TAND16B<STRIDE>(src1Tile, tmpTile);
                #ifdef __DAV_V220
                    pipe_barrier(PIPE_V);
                #endif
                TOR16B<STRIDE>(dstTile, src0Tile, src1Tile);
            }
        }
    }
}
#endif // TILEOP_TILE_OPERATOR_COPYSIGN__H