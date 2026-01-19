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
 * \file cube_pto.h
 * \brief
 */

#ifndef TILEOP_TILE_OPERATOR_CUBE_PTO__H
#define TILEOP_TILE_OPERATOR_CUBE_PTO__H
#include "../utils/layout.h"
#include "../utils/tile_tensor.h"

namespace TileOp {
constexpr int16_t SHAPE_DIM2 = 2;

template <CopyOutMode mode, bool isAcc, uint8_t reluMode>
struct TStoreConfig {
    static constexpr CopyOutMode kMode = mode;
    static constexpr bool kIsAcc = isAcc;
    static constexpr uint8_t kReluMode = reluMode;
};

template <int16_t idx, typename U>
INLINE int64_t GetShape(const U &tileTensor) {
    static_assert(idx < SHAPE_DIM2, "Idx should be less than 2");
    const auto tileLayout = tileTensor.GetLayout();
    return tileLayout.template GetShapeDim<idx>();
}

template <int16_t idx, typename U>
INLINE int64_t GetStride(const U &tileTensor) {
    static_assert(idx < SHAPE_DIM2, "Idx should be less than 2");
    const auto tileLayout = tileTensor.GetLayout();
    return tileLayout.template GetStrideDim<idx>();
}

INLINE int64_t CalNZOffset(const int64_t &srcShape0, const int64_t &srcShape1, const int64_t &offset0,
    const int64_t &offset1, const int64_t &c0Size) {
    int64_t batchSize = srcShape0 * srcShape1;
    int64_t offsetElem = offset1 + offset0 * srcShape1;
    int64_t batchIndex = offsetElem / batchSize;
    int64_t gmOffset = batchIndex * batchSize + (offset1 * srcShape0) + (offset0 - batchIndex * srcShape0) * c0Size;
    return gmOffset;
}

// Copy data from DDR to L1
template <CopyInMode mode, typename Coord, typename T, typename U>
TILEOP void TLoad(T &dst, U &src, const Coord &coord, const int64_t &curH, const int64_t &curW) {
    constexpr auto shapeSize = Std::tuple_size<typename T::Shape>::value;
    static_assert(shapeSize == SHAPE_DIM2 && Std::tuple_size<Coord>::value == SHAPE_DIM2, "Shape Size should be 2 Dim");
    uint16_t offset0 = coord.GetValue();
    uint16_t offset1 = static_cast<const Std::tuple<size_t> &>(coord).GetValue();

    static_assert(T::FORMAT == Hardware::L1 && U::FORMAT == Hardware::GM,
        "[TLoad Error]: Dst format shoulde be L1 and Src format shoulde be GM");
    if constexpr (mode == CopyInMode::ND2NZ) {
        TLoadND2NZ(dst, src, offset0, offset1);
    } else if constexpr (mode == CopyInMode::NZ2NZ) {
        TLoadNZ2NZ(dst, src, offset0, offset1, curH, curW);
    } else if constexpr (mode == CopyInMode::ND2ND) {
        TLoadND2ND(dst, src, offset0, offset1);
    }
    return;
}

// Copy data from DDR to L1 with ND -> NZ format
template <typename T, typename U>
INLINE void TLoadND2NZ(T &dst, U &src, const int64_t &offset0, const int64_t &offset1) {
    constexpr auto shapeSize = Std::tuple_size<typename T::Shape>::value;
    int64_t dstShape0 = GetShape<0>(dst);
    int64_t dstShape1 = GetShape<1>(dst);
    int64_t srcShape0 = GetShape<0>(src);
    int64_t srcShape1 = GetShape<1>(src);
    int64_t srcStride0 = GetStride<0>(src);
    int64_t srcStride1 = GetStride<1>(src);
    constexpr auto staticL1H = Std::tuple_element<shapeSize - SHAPE_DIM2, typename T::TileShape>::type::value;
    constexpr auto staticL1W = Std::tuple_element<shapeSize - 1, typename T::TileShape>::type::value;
    using shapeDim2 = pto::Shape<1, 1, 1, -1, -1>;
    using strideDim2 = pto::Stride<1, 1, 1, -1, -1>;
    using globalData = pto::GlobalTensor<typename U::Type, shapeDim2, strideDim2, pto::Layout::ND>;
    int64_t gmOffset = offset1 + offset0 * srcShape1;
    globalData src0Global((__gm__ typename U::Type *)(src.GetAddr() + gmOffset),
        pto::Shape<1, 1, 1, -1, -1>(staticL1H, staticL1W), pto::Stride<1, 1, 1, -1, -1>(srcStride0, srcStride1));
    using tileData = pto::Tile<pto::TileType::Mat, typename T::Type, staticL1H, staticL1W, pto::BLayout::ColMajor, -1,
        -1, SLayout::RowMajor>;
    tileData dstL1(dstShape0, dstShape1);
    pto::TASSIGN(dstL1, (uint64_t)dst.GetAddr());
    pto::TLOAD(dstL1, src0Global);
    return;
}

// Copy data from DDR to L1 with NZ -> NZ format
template <typename T, typename U>
INLINE void TLoadNZ2NZ(T &dst, U &src, const int64_t &offset0, const int64_t &offset1, const int64_t &curH,
    const int64_t &curW) {
    constexpr int64_t c0Size = BLOCK_ALIGN_BYTE / sizeof(typename U::Type);
    constexpr auto shapeSize = Std::tuple_size<typename T::Shape>::value;
    int64_t srcShape0 = curH;
    int64_t srcShape1 = curW;
    int64_t dstShape0 = GetShape<0>(dst);
    int64_t dstShape1 = GetShape<1>(dst);
    constexpr auto staticL1H = Std::tuple_element<shapeSize - SHAPE_DIM2, typename T::TileShape>::type::value;
    constexpr auto staticL1W = Std::tuple_element<shapeSize - 1, typename T::TileShape>::type::value;
    using shapeDim2 = pto::Shape<1, -1, -1, BLOCK_CUBE_M_N, c0Size>;
    using strideDim2 = pto::Stride<-1, -1, -1, c0Size, 1>;
    using globalData = pto::GlobalTensor<typename U::Type, shapeDim2, strideDim2, pto::Layout::NZ>;
    int64_t gmOffset = CalNZOffset(srcShape0, srcShape1, offset0, offset1, c0Size);
    globalData src0Global((__gm__ typename U::Type *)(src.GetAddr() + gmOffset),
        shapeDim2(dstShape1 / c0Size, dstShape0 / BLOCK_CUBE_M_N),
        strideDim2(srcShape0 * srcShape1, srcShape0 * c0Size, BLOCK_CUBE_M_N * c0Size));
    using tileData = pto::Tile<pto::TileType::Mat, typename T::Type, staticL1H, staticL1W, pto::BLayout::ColMajor, -1,
        -1, SLayout::RowMajor>;
    tileData dstL1(dstShape0, dstShape1);
    pto::TASSIGN(dstL1, (uint64_t)dst.GetAddr());
    pto::TLOAD(dstL1, src0Global);
    return;
}

// Copy data from DDR to L1 with ND -> ND format
template <typename T, typename U>
INLINE void TLoadND2ND(T &dst, U &src, const int64_t &offset0, const int64_t &offset1) {
    constexpr auto shapeSize = Std::tuple_size<typename T::Shape>::value;
    constexpr auto staticL1H = Std::tuple_element<shapeSize - SHAPE_DIM2, typename T::TileShape>::type::value;
    constexpr auto staticL1W = Std::tuple_element<shapeSize - 1, typename T::TileShape>::type::value;
    int64_t dstShape0 = GetShape<0>(dst);
    int64_t dstShape1 = GetShape<1>(dst);
    int64_t srcStride0 = GetStride<0>(src);
    int64_t srcStride1 = GetStride<1>(src);
    int64_t srcShape0 = GetShape<0>(src);
    int64_t srcShape1 = GetShape<1>(src);
    using shapeDim2 = pto::Shape<1, 1, 1, -1, -1>;
    using strideDim2 = pto::Stride<1, 1, 1, -1, -1>;
    using globalData = pto::GlobalTensor<typename U::Type, shapeDim2, strideDim2, pto::Layout::ND>;
    //目前场景,ND2ND只搬运bias和fixpipe，大小均为1 * N，offset0默认均为0
    int64_t gmOffset = offset1 + offset0 * srcShape1;
    globalData src0Global((__gm__ typename U::Type *)(src.GetAddr() + gmOffset),
        pto::Shape<1, 1, 1, -1, -1>(staticL1H, staticL1W), pto::Stride<1, 1, 1, -1, -1>(srcStride0, srcStride1));
    using tileData =
        pto::Tile<pto::TileType::Mat, typename T::Type, staticL1H, staticL1W, pto::BLayout::RowMajor, -1, -1>;
    tileData dstL1(dstShape0, dstShape1);
    pto::TASSIGN(dstL1, (uint64_t)dst.GetAddr());
    pto::TLOAD(dstL1, src0Global);
    return;
}

// Copy data from UB to UB with ND -> NZ format
template <typename T, typename U>
TILEOP void TMoveND2NZ(T &dst, U &src) {
    constexpr int64_t shapeSize = Std::tuple_size<typename T::Shape>::value;
    static_assert(shapeSize == SHAPE_DIM2, "Shape Size should be 2 Dim");
    static_assert(T::FORMAT == Hardware::UB && U::FORMAT == Hardware::UB);
    constexpr int64_t staticVecH = Std::tuple_element<shapeSize - SHAPE_DIM2, typename U::TileShape>::type::value;
    constexpr int64_t staticVecW = Std::tuple_element<shapeSize - 1, typename U::TileShape>::type::value;
    int64_t dstShape0 = GetShape<0>(dst);
    int64_t dstShape1 = GetShape<1>(dst);
    int64_t srcShape0 = GetShape<0>(src);
    int64_t srcShape1 = GetShape<1>(src);
    using tileNDTensor =
        pto::Tile<pto::TileType::Vec, typename U::Type, staticVecH, staticVecW, BLayout::RowMajor, -1, -1>;
    using tileNZTensor = pto::Tile<pto::TileType::Vec, typename T::Type, staticVecH, staticVecW, BLayout::ColMajor, -1,
        -1, SLayout::RowMajor>;
    tileNDTensor srcTile(srcShape0, srcShape1);
    tileNZTensor dstTile(dstShape0, dstShape1);
    pto::TASSIGN(srcTile, (uint64_t)src.GetAddr());
    pto::TASSIGN(dstTile, (uint64_t)dst.GetAddr());
    pto::TMOV(dstTile, srcTile);
    return;
}

// Copy data from UB to L1 with NZ -> NZ format
template <typename Coord, typename T, typename U>
TILEOP void TExtract(T &dst, U &src, const Coord &coord) {
    constexpr int64_t shapeSize = Std::tuple_size<typename T::Shape>::value;
    constexpr int64_t c0Size = BLOCK_ALIGN_BYTE / sizeof(typename U::Type);
    static_assert(shapeSize == SHAPE_DIM2 && Std::tuple_size<Coord>::value == SHAPE_DIM2, "Shape Size should be 2 Dim");
    static_assert(T::FORMAT == Hardware::L1 && U::FORMAT == Hardware::UB);
    int64_t offset0 = coord.GetValue();
    int64_t offset1 = static_cast<const Std::tuple<size_t> &>(coord).GetValue();
    constexpr int64_t staticUBH = Std::tuple_element<shapeSize - SHAPE_DIM2, typename U::TileShape>::type::value;
    constexpr int64_t staticUBW = Std::tuple_element<shapeSize - 1, typename U::TileShape>::type::value;
    constexpr int64_t staticL1H = Std::tuple_element<shapeSize - SHAPE_DIM2, typename T::TileShape>::type::value;
    constexpr int64_t staticL1W = Std::tuple_element<shapeSize - 1, typename T::TileShape>::type::value;
    int64_t srcShape0 = GetShape<0>(src);
    int64_t srcShape1 = GetShape<1>(src);
    int64_t dstShape0 = GetShape<0>(dst);
    int64_t dstShape1 = GetShape<1>(dst);

    int64_t UBOffset = CalNZOffset(srcShape0, srcShape1, offset0, offset1, c0Size);
    using tileUBTensor = pto::Tile<pto::TileType::Vec, typename U::Type, staticUBH, staticUBW, pto::BLayout::ColMajor,
        -1, -1, pto::SLayout::RowMajor>;
    using tileL1Tensor = pto::Tile<pto::TileType::Mat, typename T::Type, staticL1H, staticL1W, pto::BLayout::ColMajor,
        -1, -1, pto::SLayout::RowMajor>;
    tileUBTensor UBTile(srcShape0, srcShape1);
    tileL1Tensor l1Tile(dstShape0, dstShape1);
    pto::TASSIGN(UBTile, (uint64_t)src.GetAddr() + UBOffset);
    pto::TASSIGN(l1Tile, (uint64_t)dst.GetAddr());
    pto::TEXTRACT(l1Tile, UBTile);
}

// Copy data from L0C to L1 with quantization ability
template <typename config, typename Coord, typename T, typename U, typename V>
TILEOP void TExtract(T &dst, U &src, V &fixbuf, const Coord &coord, uint64_t scaleValue = 0) {
    constexpr int64_t shapeSize = Std::tuple_size<typename T::Shape>::value;
    static_assert(shapeSize == SHAPE_DIM2 && Std::tuple_size<Coord>::value == SHAPE_DIM2, "Shape Size should be 2 Dim");
    static_assert(U::FORMAT == Hardware::L0C && T::FORMAT == Hardware::L1);
    int64_t offset0 = coord.GetValue();
    int64_t offset1 = static_cast<const Std::tuple<size_t> &>(coord).GetValue();
    constexpr int64_t c0Size = BLOCK_ALIGN_BYTE / sizeof(typename T::Type);
    int64_t dstShape0 = GetShape<0>(dst);
    int64_t dstShape1 = GetShape<1>(dst);
    int64_t srcShape0 = GetShape<0>(src);
    int64_t srcShape1 = GetShape<1>(src);

    constexpr int64_t tileH = Std::tuple_element<shapeSize - SHAPE_DIM2, typename U::TileShape>::type::value;
    constexpr int64_t tileW = Std::tuple_element<shapeSize - 1, typename U::TileShape>::type::value;

    int64_t l0cOffset = CalNZOffset(srcShape0, srcShape1, offset0, offset1, c0Size);
    using l1TileData = pto::Tile<pto::TileType::Mat, typename T::Type, tileH, tileW,
        config::kMode == CopyOutMode::NZ2ND ? pto::BLayout::RowMajor : pto::BLayout::ColMajor, -1, -1,
        config::kMode == CopyOutMode::NZ2ND ? pto::SLayout::NoneBox : pto::SLayout::RowMajor>;
    using l0cTileData = pto::Tile<pto::TileType::Acc, typename U::Type, tileH, tileW, pto::BLayout::ColMajor, -1, -1,
        SLayout::RowMajor>;
    l0cTileData srcL0C(srcShape0, srcShape1);
    l1TileData dstL1(dstShape0, dstShape1);
    pto::TASSIGN(srcL0C, (uint64_t)src.GetAddr() + l0cOffset);
    pto::TASSIGN(dstL1, (uint64_t)dst.GetAddr());
    if constexpr (std::is_same<typename U::Type, int32_t>::value && std::is_same<typename T::Type, half>::value) {
        if (scaleValue != 0) {
            pto::TMOV(dstL1, srcL0C, scaleValue);
        } else {
            constexpr int64_t scaleTileH =
                Std::tuple_element<shapeSize - SHAPE_DIM2, typename V::TileShape>::type::value;
            constexpr int64_t scaleTileW = Std::tuple_element<shapeSize - 1, typename V::TileShape>::type::value;
            int64_t scaleShape0 = GetShape<0>(fixbuf);
            int64_t scaleShape1 = GetShape<1>(fixbuf);
            using scaleTileData =
                pto::Tile<pto::TileType::Scaling, uint64_t, scaleTileH, scaleTileW, pto::BLayout::RowMajor, -1, -1>;
            scaleTileData scaleData(scaleShape0, scaleShape1);
            pto::TMOV_FP(dstL1, srcL0C, scaleData);
        }
    } else {
        pto::TMOV(dstL1, srcL0C);
    }
    return;
}

template <bool isTrans, typename T, typename U>
INLINE void TExtractL1ToL0(T &dst, U &src, const int64_t &offset0, const int64_t &offset1) {
    constexpr auto shapeSize = Std::tuple_size<typename T::Shape>::value;
    constexpr auto staticL1H = Std::tuple_element<shapeSize - SHAPE_DIM2, typename U::TileShape>::type::value;
    constexpr auto staticL1W = Std::tuple_element<shapeSize - 1, typename U::TileShape>::type::value;
    constexpr int64_t c0Size = BLOCK_ALIGN_BYTE / sizeof(typename U::Type);
    constexpr auto staticL0H = Std::tuple_element<shapeSize - SHAPE_DIM2, typename T::TileShape>::type::value;
    constexpr auto staticL0W = Std::tuple_element<shapeSize - 1, typename T::TileShape>::type::value;
    int64_t dstShape0 = GetShape<0>(dst);
    int64_t dstShape1 = GetShape<1>(dst);
    int64_t srcShape0 = GetShape<0>(src);
    int64_t srcShape1 = GetShape<1>(src);
    // L1 Tile内模板参数对应含义：
    // Tile类型为Cube用于Matmul，矩阵数据类型，TileShape0，TileShape1，大分型RowMajor表明Z，ColMajor表明N,
    // validShape0, validShape1, 小分型
    using tileL1Tensor = pto::Tile<pto::TileType::Mat, typename U::Type, isTrans ? staticL1W : staticL1H,
        isTrans ? staticL1H : staticL1W, isTrans ? pto::BLayout::RowMajor : pto::BLayout::ColMajor, -1, -1,
        isTrans ? pto::SLayout::ColMajor : pto::SLayout::RowMajor>;
    // L0 TileLeft为L0A的Tile，TileRight为L0B的Tile，传入的值分别为：
    // 矩阵数据类型，tileShape0，tileShape1，validShape0，validShape0（-1表明传递动态值，在声明时传入）
    using tileL0Tensor =
        std::conditional_t<T::FORMAT == Hardware::L0A, TileLeftCompact<typename T::Type, staticL0H, staticL0W, -1, -1>,
            TileRightCompact<typename T::Type, staticL0H, staticL0W, -1, -1>>;
    tileL1Tensor l1Tile(srcShape0, srcShape1);
    tileL0Tensor l0Tile(dstShape0, dstShape1);
    pto::TASSIGN(l1Tile, (uint64_t)src.GetAddr());
    pto::TASSIGN(l0Tile, (uint64_t)dst.GetAddr());
    pto::TEXTRACT(l0Tile, l1Tile, isTrans ? offset1 : offset0, isTrans ? offset0 : offset1);
}

template <bool isTrans, typename T, typename U>
INLINE void TExtractL1ToBTOrFB(T &dst, U &src) {
    constexpr auto shapeSize = Std::tuple_size<typename T::Shape>::value;
    constexpr auto staticL1W = Std::tuple_element<shapeSize - 1, typename U::TileShape>::type::value;
    constexpr auto staticL0BW = Std::tuple_element<shapeSize - 1, typename T::TileShape>::type::value;
    int64_t nL1 = GetShape<1>(src);
    int64_t nL0 = GetShape<1>(dst);
    using tileL1Tensor = pto::Tile<pto::TileType::Mat, typename U::Type, 1, staticL1W, pto::BLayout::RowMajor, -1, -1>;
    using tileBiasOrFbTensor = pto::Tile<T::FORMAT == Hardware::BIAS ? TileType::Bias : TileType::Scaling,
        typename T::Type, 1, staticL0BW, BLayout::RowMajor, -1, -1>;
    tileL1Tensor l1Tensor(1, nL1);
    tileBiasOrFbTensor biasOrFbTensor(1, nL0);
    pto::TASSIGN<tileL1Tensor>(l1Tensor, (uint64_t)src.GetAddr());
    pto::TASSIGN<tileBiasOrFbTensor>(biasOrFbTensor, (uint64_t)dst.GetAddr());
    pto::TMOV(biasOrFbTensor, l1Tensor);
}

// Copy data from L1 to L0A/L0B
template <bool isTrans, typename Coord, typename T, typename U>
TILEOP void TExtract(T &dst, U &src, const Coord &coord) {
    constexpr auto shapeSize = Std::tuple_size<typename T::Shape>::value;
    static_assert(shapeSize == SHAPE_DIM2 && Std::tuple_size<Coord>::value == SHAPE_DIM2, "Shape Size should be 2 Dim");
    uint16_t offset0 = coord.GetValue();
    uint16_t offset1 = static_cast<const Std::tuple<size_t> &>(coord).GetValue();
    if constexpr ((T::FORMAT == Hardware::L0A || T::FORMAT == Hardware::L0B) && U::FORMAT == Hardware::L1) {
        TExtractL1ToL0<isTrans>(dst, src, offset0, offset1);
    }
    if constexpr ((T::FORMAT == Hardware::BIAS || T::FORMAT == Hardware::FIXBUF) && U::FORMAT == Hardware::L1) {
        TExtractL1ToBTOrFB<isTrans>(dst, src);
    }
    return;
}

// Copy data from L0C to UB
template <CopyOutMode mode, typename Coord, typename T, typename U>
TILEOP void TExtract(T &dst, U &src, const Coord &coord, int16_t subblockId) {
    constexpr auto shapeSize = Std::tuple_size<typename T::Shape>::value;
    constexpr int64_t c0Size = BLOCK_ALIGN_BYTE / sizeof(typename U::Type);
    static_assert(shapeSize == SHAPE_DIM2 && Std::tuple_size<Coord>::value == SHAPE_DIM2, "Shape Size should be 2 Dim");
    if constexpr (T::FORMAT == Hardware::UB && U::FORMAT == Hardware::L0C) {
        uint16_t offset0 = coord.GetValue();
        uint16_t offset1 = static_cast<const Std::tuple<size_t> &>(coord).GetValue();
        constexpr auto staticUBH = Std::tuple_element<shapeSize - SHAPE_DIM2, typename T::TileShape>::type::value;
        constexpr auto staticUBW = Std::tuple_element<shapeSize - 1, typename T::TileShape>::type::value;
        constexpr auto staticL0CH = Std::tuple_element<shapeSize - SHAPE_DIM2, typename U::TileShape>::type::value;
        constexpr auto staticL0CW = Std::tuple_element<shapeSize - 1, typename U::TileShape>::type::value;
        int64_t srcShape0 = GetShape<0>(src);
        int64_t srcShape1 = GetShape<1>(src);
        int64_t l0cOffset = CalNZOffset(srcShape0, srcShape1, offset0, offset1, c0Size);
        using tileUBTensor = pto::Tile<pto::TileType::Vec, typename T::Type, staticUBH, staticUBW,
            mode == CopyOutMode::NZ2ND ? pto::BLayout::RowMajor : pto::BLayout::ColMajor, staticUBH, staticUBW,
            mode == CopyOutMode::NZ2ND ? pto::SLayout::NoneBox : pto::SLayout::RowMajor>;
        using tileL0CTensor = pto::TileAcc<typename U::Type, staticL0CH, staticL0CW>;
        tileUBTensor UBTile;
        tileL0CTensor l0cTile;
        pto::TASSIGN(UBTile, (uint64_t)dst.GetAddr() + l0cOffset);
        pto::TASSIGN(l0cTile, (uint64_t)src.GetAddr());
        if (subblockId == 0) {
            pto::TMOV<tileUBTensor, tileL0CTensor, AccToVecMode::SingleModeVec0>(UBTile, l0cTile);
        } else {
            pto::TMOV<tileUBTensor, tileL0CTensor, AccToVecMode::SingleModeVec1>(UBTile, l0cTile);
        }
    }
}

template <bool isZeroC, typename T, typename U, typename V>
TILEOP void Matmul(T &c, U &a, V &b) {
    constexpr auto shapeSizeA = Std::tuple_size<typename U::Shape>::value;
    constexpr auto shapeSizeB = Std::tuple_size<typename V::Shape>::value;
    constexpr auto shapeSizeC = Std::tuple_size<typename T::Shape>::value;
    static_assert(shapeSizeA == SHAPE_DIM2 && shapeSizeB == SHAPE_DIM2 && shapeSizeC == SHAPE_DIM2,
        "[Matmul ERROR]: Shape dim size shoulde be 2");

    constexpr auto staticL0AH = Std::tuple_element<shapeSizeA - SHAPE_DIM2, typename U::TileShape>::type::value;
    constexpr auto staticL0AW = Std::tuple_element<shapeSizeA - 1, typename U::TileShape>::type::value;
    constexpr auto staticL0BH = Std::tuple_element<shapeSizeB - SHAPE_DIM2, typename V::TileShape>::type::value;
    constexpr auto staticL0BW = Std::tuple_element<shapeSizeB - 1, typename V::TileShape>::type::value;
    constexpr auto staticL0CH = Std::tuple_element<shapeSizeC - SHAPE_DIM2, typename T::TileShape>::type::value;
    constexpr auto staticL0CW = Std::tuple_element<shapeSizeC - 1, typename T::TileShape>::type::value;

    int64_t validM = GetShape<0>(a);
    int64_t validK = GetShape<1>(a);
    int64_t validN = GetShape<1>(b);

    using tileL0ATensor = pto::TileLeft<typename U::Type, staticL0AH, staticL0AW, -1, -1>;
    using tileL0BTensor = pto::TileRight<typename V::Type, staticL0BH, staticL0BW, -1, -1>;
    using tileL0CTensor = pto::TileAcc<typename T::Type, staticL0CH, staticL0CW, -1, -1>;

    validM = (validM + BLOCK_CUBE_M_N - 1) / BLOCK_CUBE_M_N * BLOCK_CUBE_M_N;
    tileL0ATensor l0a(validM, validK);
    tileL0BTensor l0b(validK, validN);
    tileL0CTensor l0c(validM, validN);
    if (std::is_same<typename tileL0ATensor::DType, float>::value){
        l0a.SetKAligned(true);
    }

    pto::TASSIGN(l0a, (uint64_t)a.GetAddr());
    pto::TASSIGN(l0b, (uint64_t)b.GetAddr());
    pto::TASSIGN(l0c, (uint64_t)c.GetAddr());

    if constexpr (!isZeroC) {
        pto::TMATMUL(l0c, l0a, l0b);
    } else {
        pto::TMATMUL_ACC(l0c, l0c, l0a, l0b);
    }
}

template <typename T0, typename T1, typename T2, typename T3>
TILEOP void Matmul(T0 &c, T1 &a, T2 &b, T3 &bias) {
    constexpr auto shapeSizeA = Std::tuple_size<typename T1::Shape>::value;
    constexpr auto shapeSizeB = Std::tuple_size<typename T2::Shape>::value;
    constexpr auto shapeSizeC = Std::tuple_size<typename T0::Shape>::value;
    static_assert(shapeSizeA == SHAPE_DIM2 && shapeSizeB == SHAPE_DIM2 && shapeSizeC == SHAPE_DIM2,
        "[Matmul ERROR]: Shape dim size shoulde be 2");

    constexpr auto staticL0AH = Std::tuple_element<shapeSizeA - SHAPE_DIM2, typename T1::TileShape>::type::value;
    constexpr auto staticL0AW = Std::tuple_element<shapeSizeA - 1, typename T1::TileShape>::type::value;
    constexpr auto staticL0BH = Std::tuple_element<shapeSizeB - SHAPE_DIM2, typename T2::TileShape>::type::value;
    constexpr auto staticL0BW = Std::tuple_element<shapeSizeB - 1, typename T2::TileShape>::type::value;
    constexpr auto staticL0CH = Std::tuple_element<shapeSizeC - SHAPE_DIM2, typename T0::TileShape>::type::value;
    constexpr auto staticL0CW = Std::tuple_element<shapeSizeC - 1, typename T0::TileShape>::type::value;

    using tileL0ATensor = pto::TileLeft<typename T1::Type, staticL0AH, staticL0AW>;
    using tileL0BTensor = pto::TileRight<typename T2::Type, staticL0BH, staticL0BW>;
    using tileL0CTensor = pto::TileAcc<typename T0::Type, staticL0CH, staticL0CW>;
    using tileBiasTensor =
        pto::Tile<TileType::Bias, typename T3::Type, 1, staticL0BW, BLayout::RowMajor, 1, staticL0BW>;

    tileL0ATensor l0a;
    tileL0BTensor l0b;
    tileL0CTensor l0c;
    tileBiasTensor biasT;

    pto::TASSIGN(l0a, (uint64_t)a.GetAddr());
    pto::TASSIGN(l0b, (uint64_t)b.GetAddr());
    pto::TASSIGN(l0c, (uint64_t)c.GetAddr());
    pto::TASSIGN(biasT, (uint64_t)bias.GetAddr());
    pto::TMATMUL_BIAS(l0c, l0a, l0b, biasT);
}

template <typename config, typename globalData, typename tileData>
INLINE void TStoreExecute(globalData dstGlobal, tileData srcL0C, uint64_t scaleValue) {
    if constexpr (std::is_same<typename tileData::DType, int32_t>::value &&
                  std::is_same<typename globalData::DType, __gm__ half>::value) {
        if (scaleValue != 0) {
            pto::TSTORE<tileData, globalData, config::kIsAcc ? AtomicType::AtomicAdd : AtomicType::AtomicNone>(
                dstGlobal, srcL0C, scaleValue);
        } else {
            // fixpipe perchannle 临时方案
            // 通过传入validShape(0,0)，让TSTORE规避set_fpc()，TileShape不会使用，传入不被拦截的值即可
            using fpTileData = pto::Tile<pto::TileType::Scaling, uint64_t, 32, 32, pto::BLayout::RowMajor, 0, 0>;
            fpTileData fpData;
            pto::TSTORE<tileData, globalData, fpTileData,
                config::kIsAcc ? AtomicType::AtomicAdd : AtomicType::AtomicNone>(dstGlobal, srcL0C, fpData);
        }
    } else {
        pto::TSTORE<tileData, globalData, config::kIsAcc ? AtomicType::AtomicAdd : AtomicType::AtomicNone>(dstGlobal, srcL0C);
    }
}

// Copy data from L0C to DDR with NZ -> ND format
template <typename config, typename T, typename U>
INLINE void TStoreNZ2ND(T &dst, U &src, const int64_t &offset0, const int64_t &offset1, uint64_t scaleValue = 0) {
    constexpr auto shapeSize = Std::tuple_size<typename T::Shape>::value;
    int64_t srcShape0 = GetShape<0>(src);
    int64_t srcShape1 = GetShape<1>(src);
    int64_t dstShape0 = GetShape<0>(dst);
    int64_t dstShape1 = GetShape<1>(dst);
    int64_t dstStride0 = GetStride<0>(dst);
    int64_t dstStride1 = GetStride<1>(dst);

    constexpr auto tileH = Std::tuple_element<shapeSize - SHAPE_DIM2, typename U::TileShape>::type::value;
    constexpr auto tileW = Std::tuple_element<shapeSize - 1, typename U::TileShape>::type::value;

    using shapeDim2 = pto::Shape<1, 1, 1, -1, -1>;
    using strideDim2 = pto::Stride<1, 1, 1, -1, -1>;
    int64_t gmOffset = offset1 + offset0 * dstShape1;
    using globalData = pto::GlobalTensor<typename T::Type, shapeDim2, strideDim2, pto::Layout::ND>;
    using tileData = pto::Tile<pto::TileType::Acc, typename U::Type, tileH, tileW, pto::BLayout::ColMajor, -1, -1,
        SLayout::RowMajor, TileConfig::fractalCSize, PadValue::Null, CompactMode::Normal>;
    globalData dstGlobal((__gm__ typename T::Type *)(dst.GetAddr() + gmOffset),
        pto::Shape<1, 1, 1, -1, -1>(srcShape0, srcShape1), pto::Stride<1, 1, 1, -1, -1>(dstStride0, dstStride1));
    tileData srcL0C(srcShape0, srcShape1);
    pto::TASSIGN(srcL0C, (uint64_t)src.GetAddr());
    TStoreExecute<config, globalData, tileData>(dstGlobal, srcL0C, scaleValue);
    return;
}

// Copy data from L0C to DDR with NZ -> NZ format
template <typename config, typename T, typename U>
INLINE void TStoreNZ2NZ(T &dst, U &src, const int64_t &offset0, const int64_t &offset1, const int64_t &curH,
    const int64_t &curW, uint64_t scaleValue = 0) {
    constexpr auto shapeSize = Std::tuple_size<typename T::Shape>::value;
    constexpr int64_t c0Size =
        std::is_same<typename U::Type, int32_t>::value ? BLOCK_CUBE_M_N : BLOCK_ALIGN_BYTE / sizeof(typename T::Type);
    int64_t dstShape0 = curH;
    int64_t dstShape1 = curW;
    int64_t srcShape0 = GetShape<0>(src);
    int64_t srcShape1 = GetShape<1>(src);

    constexpr auto tileH = Std::tuple_element<shapeSize - SHAPE_DIM2, typename U::TileShape>::type::value;
    constexpr auto tileW = Std::tuple_element<shapeSize - 1, typename U::TileShape>::type::value;

    int64_t gmOffset = CalNZOffset(dstShape0, dstShape1, offset0, offset1, c0Size);
    using shapeDim2 = pto::Shape<1, -1, -1, BLOCK_CUBE_M_N, c0Size>;
    using strideDim2 = pto::Stride<-1, -1, -1, c0Size, 1>;
    using globalData = pto::GlobalTensor<typename T::Type, shapeDim2, strideDim2, pto::Layout::NZ>;
    globalData dstGlobal((__gm__ typename T::Type *)(dst.GetAddr() + gmOffset),
        shapeDim2(dstShape1 / c0Size, dstShape0 / BLOCK_CUBE_M_N),
        strideDim2(dstShape0 * dstShape1, dstShape0 * c0Size, BLOCK_CUBE_M_N * c0Size));
    using tileData = pto::Tile<pto::TileType::Acc, typename U::Type, tileH, tileW, pto::BLayout::ColMajor, -1, -1,
        SLayout::RowMajor, TileConfig::fractalCSize, PadValue::Null, CompactMode::Normal>;
    tileData srcL0C(srcShape0, srcShape1);
    pto::TASSIGN(srcL0C, (uint64_t)src.GetAddr());
    TStoreExecute<config, globalData, tileData>(dstGlobal, srcL0C, scaleValue);
    return;
}

// Copy data from L0C to DDR with quantization ability
template <typename config, typename Coord, typename T, typename U, typename V>
TILEOP void TStore(T &dst, U &src, V &fixbuf, const Coord &coord, const int64_t &curH, const int64_t &curW,
    uint64_t scaleValue = 0) {
    constexpr auto shapeSize = Std::tuple_size<typename T::Shape>::value;
    static_assert(shapeSize == SHAPE_DIM2 && Std::tuple_size<Coord>::value == SHAPE_DIM2, "Shape Size should be 2 Dim");
    uint16_t offset0 = coord.GetValue();
    uint16_t offset1 = static_cast<const Std::tuple<size_t> &>(coord).GetValue();
    if constexpr (U::FORMAT == Hardware::L0C && T::FORMAT == Hardware::GM) {
        if constexpr (config::kMode == CopyOutMode::NZ2ND) {
            TStoreNZ2ND<config>(dst, src, offset0, offset1, scaleValue);
        } else {
            TStoreNZ2NZ<config>(dst, src, offset0, offset1, curH, curW, scaleValue);
        }
    }
}
} // namespace TileOp
#endif // TILEOP_TILE_OPERATOR_CUBE_PTO__H
