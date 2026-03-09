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
#include "utils/layout.h"
#include "utils/tile_tensor.h"

#if __NPU_ARCH__ == 3101
#define PTO_NPU_ARCH_A5
#elif __NPU_ARCH__ == 3510
#define PTO_NPU_ARCH_A5
#endif

constexpr int16_t SHAPE_DIM2 = 2;
constexpr int16_t SHAPE_DIM3 = 3;
constexpr uint16_t BLOCK_CUBE_M_N = 16;
constexpr uint16_t BLOCK_ALIGN_BYTE = 32;

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

template <typename T, typename U>
INLINE bool CheckShapeValid(const T &dst, const U &src) {
    int64_t dstShape0 = GetShape<0>(dst);
    int64_t dstShape1 = GetShape<1>(dst);
    int64_t srcShape0 = GetShape<0>(src);
    int64_t srcShape1 = GetShape<1>(src);
    if (dstShape0 == 0 || dstShape1 == 0 || srcShape0 == 0 || srcShape1 == 0) {
        return false;
    }
    return true;
}

// Copy data from DDR to L1 with ND -> NZ format
template <PaddingMode padMode, typename T, typename U>
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
    using tileData = pto::Tile<pto::TileType::Mat, typename T::Type, staticL1H, staticL1W, pto::BLayout::ColMajor, -1,
        -1, pto::SLayout::RowMajor>;
    int64_t gmOffset = offset1 + offset0 * srcShape1;
    globalData src0Global((__gm__ typename U::Type *)(src.GetAddr() + gmOffset), shapeDim2(staticL1H, staticL1W),
        strideDim2(srcStride0, srcStride1));
    tileData dstL1(dstShape0, dstShape1);
    pto::TASSIGN(dstL1, (uint64_t)dst.GetAddr());
    pto::TLOAD(dstL1, src0Global);
    if constexpr (padMode != PaddingMode::NO_PADDING) {
        pto::TFILLPAD(dstL1, dstL1);
    }
    return;
}

// Copy data from DDR to L1 with NZ -> NZ format
template <PaddingMode padMode, typename T, typename U>
INLINE void TLoadNZ2NZ(
    T &dst, U &src, const int64_t &offset0, const int64_t &offset1, const int64_t &curH, const int64_t &curW) {
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
    using tileData = pto::Tile<pto::TileType::Mat, typename T::Type, staticL1H, staticL1W, pto::BLayout::ColMajor, -1,
        -1, pto::SLayout::RowMajor>;
    int64_t gmOffset = CalNZOffset(srcShape0, srcShape1, offset0, offset1, c0Size);
    globalData src0Global((__gm__ typename U::Type *)(src.GetAddr() + gmOffset),
        shapeDim2(dstShape1 / c0Size, dstShape0 / BLOCK_CUBE_M_N),
        strideDim2(srcShape0 * srcShape1, srcShape0 * c0Size, BLOCK_CUBE_M_N * c0Size));
    tileData dstL1(dstShape0, dstShape1);
    pto::TASSIGN(dstL1, (uint64_t)dst.GetAddr());
    pto::TLOAD(dstL1, src0Global);
    if constexpr (padMode != PaddingMode::NO_PADDING) {
        pto::TFILLPAD(dstL1, dstL1);
    }
    return;
}

// Copy data from DDR to L1
template <CopyInMode copyMode, PaddingMode padMode, typename Coord, typename T, typename U>
TILEOP void TLoad(T &dst, U &src, const Coord &coord, const int64_t &curH, const int64_t &curW) {
    constexpr auto shapeSize = Std::tuple_size<typename T::Shape>::value;
    static_assert(shapeSize == SHAPE_DIM2 && Std::tuple_size<Coord>::value == SHAPE_DIM2, "Shape Size should be 2 Dim");
    if (!CheckShapeValid(dst, src)) {
        return;
    }
    int64_t offset0 = coord.GetValue();
    int64_t offset1 = static_cast<const Std::tuple<size_t> &>(coord).GetValue();

    static_assert(T::FORMAT == Hardware::L1 && U::FORMAT == Hardware::GM,
        "[TLoad Error]: Dst format shoulde be L1 and Src format shoulde be GM");
    if constexpr (copyMode == CopyInMode::ND2NZ) {
        TLoadND2NZ<padMode>(dst, src, offset0, offset1);
    } else if constexpr (copyMode == CopyInMode::NZ2NZ) {
        TLoadNZ2NZ<padMode>(dst, src, offset0, offset1, curH, curW);
    } else if constexpr (copyMode == CopyInMode::ND2ND) {
        TLoadND2ND(dst, src, offset0, offset1);
    }
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
    // 目前场景,ND2ND只搬运bias和fixpipe，大小均为1 * N，offset0默认均为0
    int64_t gmOffset = offset1 + offset0 * srcShape1;
    globalData src0Global((__gm__ typename U::Type *)(src.GetAddr() + gmOffset), shapeDim2(staticL1H, staticL1W),
        strideDim2(srcStride0, srcStride1));
    using tileData =
        pto::Tile<pto::TileType::Mat, typename T::Type, staticL1H, staticL1W, pto::BLayout::RowMajor, -1, -1>;
    tileData dstL1(dstShape0, dstShape1);
    pto::TASSIGN(dstL1, (uint64_t)dst.GetAddr());
    pto::TLOAD(dstL1, src0Global);
    return;
}

// Copy Scale A data from DDR to L1 for MX matmul
template <CopyInMode mode, typename Coord, typename T, typename U>
TILEOP void TLoadAMX(T &dst, U &src, const Coord &coord) {
    constexpr auto shapeSize = Std::tuple_size<typename T::Shape>::value;
    static_assert(shapeSize == SHAPE_DIM3 && Std::tuple_size<Coord>::value == SHAPE_DIM3,
        "[TLoadAMX Error]: MXMatmul A Scale Shape Size should be 3 Dim");
    int64_t offset0 = TileOp::GetTupleElement<Coord, DIM_1ST, SHAPE_DIM3, 0>(coord);
    int64_t offset1 = TileOp::GetTupleElement<Coord, DIM_2ND, SHAPE_DIM3, 0>(coord);

    static_assert(T::FORMAT == Hardware::L1 && U::FORMAT == Hardware::GM,
        "[TLoadAMX Error]: Dst format shoulde be L1 and Src format shoulde be GM");

    int64_t dstShape0 = GetShape<0>(dst);
    int64_t dstShape1 = GetShape<1>(dst);
    int64_t srcShape0 = GetShape<0>(src);
    int64_t srcShape1 = GetShape<1>(src);
    constexpr auto staticL1H = Std::tuple_element<shapeSize - SHAPE_DIM3, typename T::TileShape>::type::value;
    constexpr auto staticL1W =
        Std::tuple_element<shapeSize - SHAPE_DIM2, typename T::TileShape>::type::value * SHAPE_DIM2;

    using shapeDim2 = pto::Shape<1, 1, -1, -1, SHAPE_DIM2>;
    using strideDim3 = pto::Stride<-1, -1, -1, SHAPE_DIM2, 1>;
    using tileData = pto::Tile<pto::TileType::Mat, typename T::Type, staticL1H, staticL1W, pto::BLayout::RowMajor, -1,
        -1, pto::SLayout::RowMajor, pto::TileConfig::alignedSize>;
    if constexpr (mode == CopyInMode::ND2NZ) {
        int64_t gmOffset = offset1 * SHAPE_DIM2 + offset0 * srcShape1 * SHAPE_DIM2;
        using globalData = pto::GlobalTensor<typename U::Type, shapeDim2, strideDim3, pto::Layout::MX_A_ND>;
        globalData src0Global((__gm__ typename U::Type *)(src.GetAddr() + gmOffset),
            shapeDim2(dstShape0, dstShape1),
            strideDim3(srcShape0 * srcShape1 * SHAPE_DIM2, srcShape0 * srcShape1 * SHAPE_DIM2, srcShape1 * SHAPE_DIM2));
        tileData dstL1(dstShape0, dstShape1 * SHAPE_DIM2);
        pto::TASSIGN(dstL1, (uint64_t)dst.GetAddr());
        pto::TLOAD(dstL1, src0Global);
    } else if constexpr (mode == CopyInMode::DN2NZ) {
        int64_t gmOffset = offset1 * SHAPE_DIM2  + offset0 * SHAPE_DIM2 * srcShape1;
        using globalData = pto::GlobalTensor<typename U::Type, shapeDim2, strideDim3, pto::Layout::MX_A_DN>;
        globalData src0Global((__gm__ typename U::Type *)(src.GetAddr() + gmOffset),
            shapeDim2(dstShape0, dstShape1),
            strideDim3(srcShape0 * srcShape1 * SHAPE_DIM2, srcShape0 * srcShape1 * SHAPE_DIM2, srcShape1 * SHAPE_DIM2));
        tileData dstL1(dstShape0, dstShape1 * SHAPE_DIM2);
        pto::TASSIGN(dstL1, (uint64_t)dst.GetAddr());
        pto::TLOAD(dstL1, src0Global);
    }
}

// Copy Scale B data from DDR to L1 for MX matmul
template <CopyInMode mode, typename Coord, typename T, typename U>
TILEOP void TLoadBMX(T &dst, U &src, const Coord &coord) {
    constexpr auto shapeSize = Std::tuple_size<typename T::Shape>::value;
    static_assert(shapeSize == SHAPE_DIM3 && Std::tuple_size<Coord>::value == SHAPE_DIM3,
        "[TLoadBMX Error]: MXMatmul B Scale Shape Size should be 3 Dim");
    int64_t offset0 = TileOp::GetTupleElement<Coord, DIM_1ST, SHAPE_DIM3, 0>(coord);
    int64_t offset1 = TileOp::GetTupleElement<Coord, DIM_2ND, SHAPE_DIM3, 0>(coord);

    static_assert(T::FORMAT == Hardware::L1 && U::FORMAT == Hardware::GM,
        "[TLoadBMX Error]: Dst format shoulde be L1 and Src format shoulde be GM");

    int64_t dstShape0 = GetShape<0>(dst);
    int64_t dstShape1 = GetShape<1>(dst);
    int64_t srcShape0 = GetShape<0>(src);
    int64_t srcShape1 = GetShape<1>(src);
    constexpr auto staticL1H =
        Std::tuple_element<shapeSize - SHAPE_DIM3, typename T::TileShape>::type::value * SHAPE_DIM2;
    constexpr auto staticL1W = Std::tuple_element<shapeSize - SHAPE_DIM2, typename T::TileShape>::type::value;

    using shapeDim2 = pto::Shape<1, 1, -1, -1, SHAPE_DIM2>;
    using strideDim3 = pto::Stride<-1, -1, -1, SHAPE_DIM2, 1>;
    using tileData = pto::Tile<pto::TileType::Mat, typename T::Type, staticL1H, staticL1W, pto::BLayout::ColMajor, -1,
        -1, pto::SLayout::ColMajor, pto::TileConfig::alignedSize>;
    if constexpr (mode == CopyInMode::ND2NZ) {
        int64_t gmOffset = offset1 * SHAPE_DIM2 + offset0 * SHAPE_DIM2 * srcShape1;
        using globalData = pto::GlobalTensor<typename U::Type, shapeDim2, strideDim3, pto::Layout::MX_B_ND>;
        globalData src0Global((__gm__ typename U::Type *)(src.GetAddr() + gmOffset),
            shapeDim2(dstShape0, dstShape1),
            strideDim3(srcShape0 * srcShape1 * SHAPE_DIM2, srcShape0 * srcShape1 * SHAPE_DIM2, srcShape1 * SHAPE_DIM2));
        tileData dstL1(dstShape0 * SHAPE_DIM2, dstShape1);
        pto::TASSIGN(dstL1, (uint64_t)dst.GetAddr());
        pto::TLOAD(dstL1, src0Global);
    } else if constexpr (mode == CopyInMode::DN2NZ) {
        int64_t gmOffset = offset1 * SHAPE_DIM2 + offset0 * srcShape1 * SHAPE_DIM2;
        using globalData = pto::GlobalTensor<typename U::Type, shapeDim2, strideDim3, pto::Layout::MX_B_DN>;
        globalData src0Global((__gm__ typename U::Type *)(src.GetAddr() + gmOffset),
            shapeDim2(dstShape0, dstShape1),
            strideDim3(srcShape0 * srcShape1 * SHAPE_DIM2, srcShape0 * srcShape1 * SHAPE_DIM2, srcShape1 * SHAPE_DIM2));
        tileData dstL1(dstShape0 * SHAPE_DIM2, dstShape1);
        pto::TASSIGN(dstL1, (uint64_t)dst.GetAddr());
        pto::TLOAD(dstL1, src0Global);
    }
}

// Copy data from UB to UB with ND -> NZ format
template <typename T, typename U>
TILEOP void TMoveND2NZ(T &dst, U &src) {
    if (!CheckShapeValid(dst, src)) {
        return;
    }
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
        pto::Tile<pto::TileType::Vec, typename U::Type, staticVecH, staticVecW, pto::BLayout::RowMajor, -1, -1>;
    using tileNZTensor = pto::Tile<pto::TileType::Vec, typename T::Type, staticVecH, staticVecW, pto::BLayout::ColMajor,
        -1, -1, pto::SLayout::RowMajor>;
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
    if (!CheckShapeValid(dst, src)) {
        return;
    }
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

template <typename V>
INLINE auto CreateScaleTileData(V &fixbuf) {
    constexpr int64_t shapeSize = Std::tuple_size<typename V::Shape>::value;
    constexpr int64_t scaleTileH = Std::tuple_element<shapeSize - SHAPE_DIM2, typename V::TileShape>::type::value;
    constexpr int64_t scaleTileW = Std::tuple_element<shapeSize - 1, typename V::TileShape>::type::value;
    int64_t scaleShape0 = GetShape<0>(fixbuf);
    int64_t scaleShape1 = GetShape<1>(fixbuf);
    using scaleTileData =
        pto::Tile<pto::TileType::Scaling, uint64_t, scaleTileH, scaleTileW, pto::BLayout::RowMajor, -1, -1>;
    return scaleTileData(scaleShape0, scaleShape1);
}

template <typename config, typename l1Data, typename l0cData, typename V>
TILEOP void TExtractL0CToL1(
    l1Data &dstL1, l0cData &srcL0C, V &fixbuf, uint16_t l0cOffset0, uint16_t l0cOffset1, uint64_t scaleValue = 0) {
    if constexpr (std::is_same<typename l0cData::DType, int32_t>::value &&
                  std::is_same<typename l1Data::DType, half>::value) {
        if (scaleValue != 0) {
            constexpr pto::ReluPreMode relu_mode =
                (config::kReluMode == 0) ? pto::ReluPreMode::NoRelu : pto::ReluPreMode::NormalRelu;
            pto::TEXTRACT<l1Data, l0cData, relu_mode>(dstL1, srcL0C, scaleValue, l0cOffset0, l0cOffset1);
        } else {
            auto scaleData = CreateScaleTileData(fixbuf);
            pto::TASSIGN(scaleData, (uint64_t)fixbuf.GetAddr());
            pto::TEXTRACT_FP<l1Data, l0cData, decltype(scaleData),
                config::kReluMode == 0 ? pto::ReluPreMode::NoRelu : pto::ReluPreMode::NormalRelu>(
                dstL1, srcL0C, scaleData, l0cOffset0, l0cOffset1);
        }
    } else {
        pto::TEXTRACT<l1Data, l0cData,
            config::kReluMode == 0 ? pto::ReluPreMode::NoRelu : pto::ReluPreMode::NormalRelu>(
            dstL1, srcL0C, l0cOffset0, l0cOffset1);
    }
}

template <typename config, typename l1Data, typename l0cData, typename V>
TILEOP void TInsertL0CToL1(
    l1Data &dstL1, l0cData &srcL0C, V &fixbuf, uint16_t l1Offset0, uint16_t l1Offset1, uint64_t scaleValue = 0) {
    if constexpr (std::is_same<typename l0cData::DType, int32_t>::value &&
                  std::is_same<typename l1Data::DType, half>::value) {
        if (scaleValue != 0) {
            constexpr pto::ReluPreMode relu_mode =
                (config::kReluMode == 0) ? pto::ReluPreMode::NoRelu : pto::ReluPreMode::NormalRelu;
            pto::TINSERT<l1Data, l0cData, relu_mode>(dstL1, srcL0C, scaleValue, l1Offset0, l1Offset1);
        } else {
            auto scaleData = CreateScaleTileData(fixbuf);
            pto::TASSIGN(scaleData, (uint64_t)fixbuf.GetAddr());
            pto::TINSERT_FP<l1Data, l0cData, decltype(scaleData),
                config::kReluMode == 0 ? pto::ReluPreMode::NoRelu : pto::ReluPreMode::NormalRelu>(
                dstL1, srcL0C, scaleData, l1Offset0, l1Offset1);
        }
    } else {
        pto::TINSERT<l1Data, l0cData, config::kReluMode == 0 ? pto::ReluPreMode::NoRelu : pto::ReluPreMode::NormalRelu>(
            dstL1, srcL0C, l1Offset0, l1Offset1);
    }
}

// Copy data from L0C to L1 with quantization ability
template <typename config, typename Coord, typename T, typename U, typename V>
TILEOP void TExtract(T &dst, U &src, V &fixbuf, const Coord &l1Coord, const Coord &l0cCoord, uint64_t scaleValue = 0) {
    if (!CheckShapeValid(dst, src) || !CheckShapeValid(dst, fixbuf)) {
        return;
    }
    constexpr int64_t shapeSize = Std::tuple_size<typename T::Shape>::value;
    static_assert(shapeSize == SHAPE_DIM2 && Std::tuple_size<Coord>::value == SHAPE_DIM2, "Shape Size should be 2 Dim");
    static_assert(U::FORMAT == Hardware::L0C && T::FORMAT == Hardware::L1);

    uint16_t l1Offset0 = l1Coord.GetValue();
    uint16_t l1Offset1 = static_cast<const Std::tuple<size_t> &>(l1Coord).GetValue();
    uint16_t l0cOffset0 = l0cCoord.GetValue();
    uint16_t l0cOffset1 = static_cast<const Std::tuple<size_t> &>(l0cCoord).GetValue();

    constexpr int64_t c0Size = BLOCK_ALIGN_BYTE / sizeof(typename T::Type);
    int64_t dstShape0 = GetShape<0>(dst);
    int64_t dstShape1 = GetShape<1>(dst);
    int64_t srcShape0 = GetShape<0>(src);
    int64_t srcShape1 = GetShape<1>(src);

    constexpr int64_t tileL1H = Std::tuple_element<shapeSize - SHAPE_DIM2, typename T::TileShape>::type::value;
    constexpr int64_t tileL1W = Std::tuple_element<shapeSize - 1, typename T::TileShape>::type::value;
    constexpr int64_t tileL0CH = Std::tuple_element<shapeSize - SHAPE_DIM2, typename U::TileShape>::type::value;
    constexpr int64_t tileL0CW = Std::tuple_element<shapeSize - 1, typename U::TileShape>::type::value;

    using l1TileData = pto::Tile<pto::TileType::Mat, typename T::Type, tileL1H, tileL1W,
        config::kMode == CopyOutMode::NZ2ND ? pto::BLayout::RowMajor : pto::BLayout::ColMajor, -1, -1,
        config::kMode == CopyOutMode::NZ2ND ? pto::SLayout::NoneBox : pto::SLayout::RowMajor>;
    using l0cTileData = pto::Tile<pto::TileType::Acc, typename U::Type, tileL0CH, tileL0CW, pto::BLayout::ColMajor, -1,
        -1, pto::SLayout::RowMajor>;

    l1TileData dstL1(dstShape0, dstShape1);
    l0cTileData srcL0C(srcShape0, srcShape1);
    pto::TASSIGN(srcL0C, (uint64_t)src.GetAddr());
    pto::TASSIGN(dstL1, (uint64_t)dst.GetAddr());

    if (dstShape0 < srcShape0 || dstShape1 < srcShape1) {
        TExtractL0CToL1<config, l1TileData, l0cTileData, V>(dstL1, srcL0C, fixbuf, l0cOffset0, l0cOffset1, scaleValue);
    } else {
        TInsertL0CToL1<config, l1TileData, l0cTileData, V>(dstL1, srcL0C, fixbuf, l1Offset0, l1Offset1, scaleValue);
    }
    return;
}

template <bool isTrans, typename T, typename U>
INLINE void TExtractL1ToL0(T &dst, U &src, const int64_t &offset0, const int64_t &offset1) {
    constexpr auto shapeSize = Std::tuple_size<typename T::Shape>::value;
    constexpr auto staticL1H = Std::tuple_element<shapeSize - SHAPE_DIM2, typename U::TileShape>::type::value;
    constexpr auto staticL1W = Std::tuple_element<shapeSize - 1, typename U::TileShape>::type::value;
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
    using tileL0Tensor = std::conditional_t<T::FORMAT == Hardware::L0A,
        pto::TileLeftCompact<typename T::Type, staticL0H, staticL0W, -1, -1>,
        pto::TileRightCompact<typename T::Type, staticL0H, staticL0W, -1, -1>>;
    tileL1Tensor l1Tile(srcShape0, srcShape1);
    tileL0Tensor l0Tile(dstShape0, dstShape1);
    if (std::is_same<typename tileL0Tensor::DType, float>::value && T::FORMAT == Hardware::L0A) {
        l0Tile.SetKAligned(true);
    }
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
    using tileBiasOrFbTensor = pto::Tile<T::FORMAT == Hardware::BIAS ? pto::TileType::Bias : pto::TileType::Scaling,
        typename T::Type, 1, staticL0BW, pto::BLayout::RowMajor, -1, -1>;
    tileL1Tensor l1Tensor(1, nL1);
    tileBiasOrFbTensor biasOrFbTensor(1, nL0);
    pto::TASSIGN<tileL1Tensor>(l1Tensor, (uint64_t)src.GetAddr());
    pto::TASSIGN<tileBiasOrFbTensor>(biasOrFbTensor, (uint64_t)dst.GetAddr());
    pto::TMOV(biasOrFbTensor, l1Tensor);
}

// Copy data from L1 to L0A/L0B
template <bool isTrans, typename Coord, typename T, typename U>
TILEOP void TExtract(T &dst, U &src, const Coord &coord) {
    if (!CheckShapeValid(dst, src)) {
        return;
    }
    constexpr auto shapeSize = Std::tuple_size<typename T::Shape>::value;
    static_assert(shapeSize == SHAPE_DIM2 && Std::tuple_size<Coord>::value == SHAPE_DIM2, "Shape Size should be 2 Dim");
    int64_t offset0 = coord.GetValue();
    int64_t offset1 = static_cast<const Std::tuple<size_t> &>(coord).GetValue();
    if constexpr ((T::FORMAT == Hardware::L0A || T::FORMAT == Hardware::L0B) && U::FORMAT == Hardware::L1) {
        TExtractL1ToL0<isTrans>(dst, src, offset0, offset1);
    }
    if constexpr ((T::FORMAT == Hardware::BIAS || T::FORMAT == Hardware::FIXBUF) && U::FORMAT == Hardware::L1) {
        TExtractL1ToBTOrFB<isTrans>(dst, src);
    }
    return;
}

// Copy data from L1 to L0A_MX scale or L0B_MX scale
template <typename Coord, typename T, typename U>
TILEOP void TExtractMX(T &dst, U &src, const Coord &coord)
{
    constexpr auto shapeSize = Std::tuple_size<typename T::Shape>::value;
    static_assert(shapeSize == SHAPE_DIM3 && Std::tuple_size<Coord>::value == SHAPE_DIM3,
                  "[TExtractMX Error]: L0A_MX scale or L0B_MX scale Shape Size should be 3 Dim");
    static_assert((T::FORMAT == Hardware::L0A_MX || T::FORMAT == Hardware::L0B_MX) && U::FORMAT == Hardware::L1);
    int64_t offset0 = TileOp::GetTupleElement<Coord, DIM_1ST, SHAPE_DIM3, 0>(coord);
    int64_t offset1 = TileOp::GetTupleElement<Coord, DIM_2ND, SHAPE_DIM3, 0>(coord);
    constexpr auto staticL1H = Std::tuple_element<shapeSize - SHAPE_DIM3, typename U::TileShape>::type::value;
    constexpr auto staticL1W = Std::tuple_element<shapeSize - SHAPE_DIM2, typename U::TileShape>::type::value;
    constexpr auto staticL0H = Std::tuple_element<shapeSize - SHAPE_DIM3, typename T::TileShape>::type::value;
    constexpr auto staticL0W = Std::tuple_element<shapeSize - SHAPE_DIM2, typename T::TileShape>::type::value;
    int64_t dstShape0 = GetShape<0>(dst);
    int64_t dstShape1 = GetShape<1>(dst);
    int64_t srcShape0 = GetShape<0>(src);
    int64_t srcShape1 = GetShape<1>(src);
    using tileL1Tensor = std::conditional_t<T::FORMAT == Hardware::L0A_MX,
          pto::Tile<pto::TileType::Mat, typename U::Type, staticL1H, staticL1W * SHAPE_DIM2, pto::BLayout::RowMajor, -1,
              -1, pto::SLayout::RowMajor, pto::TileConfig::alignedSize>,
          pto::Tile<pto::TileType::Mat, typename U::Type, staticL1H * SHAPE_DIM2, staticL1W, pto::BLayout::ColMajor, -1,
              -1, pto::SLayout::ColMajor, pto::TileConfig::alignedSize>>;
    using tileL0MXTensor = std::conditional_t<T::FORMAT == Hardware::L0A_MX,
          pto::TileLeftScaleCompact<typename T::Type, staticL0H, staticL0W * SHAPE_DIM2, -1, -1>,
          pto::TileRightScaleCompact<typename T::Type, staticL0H * SHAPE_DIM2, staticL0W, -1, -1>>;
    if constexpr (T::FORMAT == Hardware::L0A_MX) {
        tileL1Tensor l1Tile(srcShape0, srcShape1 * SHAPE_DIM2);
        tileL0MXTensor l0MXTile(dstShape0, dstShape1 * SHAPE_DIM2);
        pto::TASSIGN(l1Tile, (uint64_t)src.GetAddr());
        pto::TASSIGN(l0MXTile, (uint64_t)dst.GetAddr());
        pto::TEXTRACT(l0MXTile, l1Tile, offset0, offset1 * SHAPE_DIM2);
    } else {
        tileL1Tensor l1Tile(srcShape0 * SHAPE_DIM2, srcShape1);
        tileL0MXTensor l0MXTile(dstShape0 * SHAPE_DIM2, dstShape1);
        pto::TASSIGN(l1Tile, (uint64_t)src.GetAddr());
        pto::TASSIGN(l0MXTile, (uint64_t)dst.GetAddr());
        pto::TEXTRACT(l0MXTile, l1Tile, offset0 * SHAPE_DIM2, offset1);
    }
}

// Copy data from L0C to UB
template <CopyOutMode mode, typename Coord, typename T, typename U>
TILEOP void TExtract(T &dst, U &src, const Coord &coord, int16_t subblockId) {
    if (!CheckShapeValid(dst, src)) {
        return;
    }
    constexpr auto shapeSize = Std::tuple_size<typename T::Shape>::value;
    constexpr int64_t c0Size = BLOCK_ALIGN_BYTE / sizeof(typename U::Type);
    static_assert(shapeSize == SHAPE_DIM2 && Std::tuple_size<Coord>::value == SHAPE_DIM2, "Shape Size should be 2 Dim");
    if constexpr (T::FORMAT == Hardware::UB && U::FORMAT == Hardware::L0C) {
        int64_t offset0 = coord.GetValue();
        int64_t offset1 = static_cast<const Std::tuple<size_t> &>(coord).GetValue();
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
            pto::TMOV<tileUBTensor, tileL0CTensor, pto::AccToVecMode::SingleModeVec0>(UBTile, l0cTile);
        } else {
            pto::TMOV<tileUBTensor, tileL0CTensor, pto::AccToVecMode::SingleModeVec1>(UBTile, l0cTile);
        }
    }
}

template <bool isZeroC, TransMode transMode, typename T, typename U, typename V>
TILEOP void TMatmul(T &c, U &a, V &b) {
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
    if (validM == 0 || validK == 0 || validN == 0) {
        return;
    }

    using tileL0ATensor = pto::TileLeft<typename U::Type, staticL0AH, staticL0AW, -1, -1>;
    using tileL0BTensor = pto::TileRight<typename V::Type, staticL0BH, staticL0BW, -1, -1>;
    using tileL0CTensor = pto::TileAcc<typename T::Type, staticL0CH, staticL0CW, -1, -1>;

    validM = (validM + BLOCK_CUBE_M_N - 1) / BLOCK_CUBE_M_N * BLOCK_CUBE_M_N;
    tileL0ATensor l0a(validM, validK);
    if constexpr (transMode != TransMode::CAST_NONE) {
        l0a.SetMadTF32Mode(static_cast<pto::RoundMode>(transMode));
    }
    tileL0BTensor l0b(validK, validN);
    tileL0CTensor l0c(validM, validN);
    if (std::is_same<typename tileL0ATensor::DType, float>::value) {
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
    if constexpr (transMode != TransMode::CAST_NONE) {
        l0a.ResetMadMode();
    }
}

template <TransMode transMode, typename T0, typename T1, typename T2, typename T3>
TILEOP void TMatmul(T0 &c, T1 &a, T2 &b, T3 &bias) {
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

    using tileL0ATensor = pto::TileLeft<typename T1::Type, staticL0AH, staticL0AW, -1, -1>;
    using tileL0BTensor = pto::TileRight<typename T2::Type, staticL0BH, staticL0BW, -1, -1>;
    using tileL0CTensor = pto::TileAcc<typename T0::Type, staticL0CH, staticL0CW, -1, -1>;
    using tileBiasTensor =
        pto::Tile<pto::TileType::Bias, typename T3::Type, 1, staticL0BW, pto::BLayout::RowMajor, -1, -1>;

    int64_t validM = GetShape<0>(a);
    int64_t validK = GetShape<1>(a);
    int64_t validN = GetShape<1>(b);
    if (validM == 0 || validK == 0 || validN == 0) {
        return;
    }

    validM = (validM + BLOCK_CUBE_M_N - 1) / BLOCK_CUBE_M_N * BLOCK_CUBE_M_N;
    tileL0ATensor l0a(validM, validK);
    if constexpr (transMode != TransMode::CAST_NONE) {
        l0a.SetMadTF32Mode(static_cast<pto::RoundMode>(transMode));
    }
    tileL0BTensor l0b(validK, validN);
    tileL0CTensor l0c(validM, validN);
    tileBiasTensor biasT(1, validN);

    pto::TASSIGN(l0a, (uint64_t)a.GetAddr());
    pto::TASSIGN(l0b, (uint64_t)b.GetAddr());
    pto::TASSIGN(l0c, (uint64_t)c.GetAddr());
    pto::TASSIGN(biasT, (uint64_t)bias.GetAddr());
    pto::TMATMUL_BIAS(l0c, l0a, l0b, biasT);
    if constexpr (transMode != TransMode::CAST_NONE) {
        l0a.ResetMadMode();
    }
}

#if defined PTO_NPU_ARCH_A5
template <bool isZeroC, typename T0, typename T1, typename T2, typename T3, typename T4>
TILEOP void MatmulMX(T0 &c, T1 &a, T2 &aScale, T3 &b, T4 &bScale)
{
    constexpr auto shapeSizeC = Std::tuple_size<typename T0::Shape>::value;
    constexpr auto shapeSizeA = Std::tuple_size<typename T1::Shape>::value;
    constexpr auto shapeSizeAScale = Std::tuple_size<typename T2::Shape>::value;
    constexpr auto shapeSizeB = Std::tuple_size<typename T3::Shape>::value;
    constexpr auto shapeSizeBScale = Std::tuple_size<typename T4::Shape>::value;
    static_assert(shapeSizeC == SHAPE_DIM2 && shapeSizeA == SHAPE_DIM2 && shapeSizeAScale == SHAPE_DIM3 &&
                      shapeSizeB == SHAPE_DIM2 && shapeSizeBScale == SHAPE_DIM3,
        "[MatmulMX ERROR]: Tensor Shape dim size shoulde be 2 and Scale Shape dim size shoulde be 3");

    constexpr auto staticL0AH = Std::tuple_element<shapeSizeA - SHAPE_DIM2, typename T1::TileShape>::type::value;
    constexpr auto staticL0AW = Std::tuple_element<shapeSizeA - 1, typename T1::TileShape>::type::value;
    constexpr auto staticL0AScaleW =
        Std::tuple_element<shapeSizeAScale - SHAPE_DIM2, typename T2::TileShape>::type::value;
    constexpr auto staticL0BH = Std::tuple_element<shapeSizeB - SHAPE_DIM2, typename T3::TileShape>::type::value;
    constexpr auto staticL0BW = Std::tuple_element<shapeSizeB - 1, typename T3::TileShape>::type::value;
    constexpr auto staticL0BScaleH =
        Std::tuple_element<shapeSizeBScale - SHAPE_DIM3, typename T4::TileShape>::type::value;
    constexpr auto staticL0CH = Std::tuple_element<shapeSizeC - SHAPE_DIM2, typename T0::TileShape>::type::value;
    constexpr auto staticL0CW = Std::tuple_element<shapeSizeC - 1, typename T0::TileShape>::type::value;

    int64_t validM = GetShape<0>(a);
    int64_t validK = GetShape<1>(a);
    int64_t validN = GetShape<1>(b);
    int64_t validScaleK = GetShape<1>(aScale) * SHAPE_DIM2;

    using tileL0CTensor = pto::TileAcc<typename T0::Type, staticL0CH, staticL0CW, -1, -1>;
    using tileL0ATensor = pto::TileLeft<typename T1::Type, staticL0AH, staticL0AW, -1, -1>;
    using tileL0AScaleTensor = pto::TileLeftScale<typename T1::Type, staticL0AH, staticL0AScaleW * SHAPE_DIM2, -1, -1>;
    using tileL0BTensor = pto::TileRight<typename T3::Type, staticL0BH, staticL0BW, -1, -1>;
    using tileL0BScaleTensor = pto::TileRightScale<typename T3::Type, staticL0BScaleH * SHAPE_DIM2, staticL0BW, -1, -1>;

    validM = (validM + BLOCK_CUBE_M_N - 1) / BLOCK_CUBE_M_N * BLOCK_CUBE_M_N;
    tileL0ATensor l0a(validM, validK);
    tileL0AScaleTensor l0aScale(validM, validScaleK);
    tileL0BTensor l0b(validK, validN);
    tileL0BScaleTensor l0bScale(validScaleK, validN);
    tileL0CTensor l0c(validM, validN);

    pto::TASSIGN(l0a, (uint64_t)a.GetAddr());
    pto::TASSIGN(l0aScale, (uint64_t)aScale.GetAddr());
    pto::TASSIGN(l0b, (uint64_t)b.GetAddr());
    pto::TASSIGN(l0bScale, (uint64_t)bScale.GetAddr());
    pto::TASSIGN(l0c, (uint64_t)c.GetAddr());

    if constexpr (!isZeroC) {
        pto::TMATMUL_MX(l0c, l0a, l0aScale, l0b, l0bScale);
    } else {
        pto::TMATMUL_MX(l0c, l0c, l0a, l0aScale, l0b, l0bScale);
    }
}

template <typename T0, typename T1, typename T2, typename T3, typename T4, typename T5>
TILEOP void MatmulMX(T0 &c, T1 &a, T2 &aScale, T3 &b, T4 &bScale, T5 &bias)
{
    constexpr auto shapeSizeC = Std::tuple_size<typename T0::Shape>::value;
    constexpr auto shapeSizeA = Std::tuple_size<typename T1::Shape>::value;
    constexpr auto shapeSizeAScale = Std::tuple_size<typename T2::Shape>::value;
    constexpr auto shapeSizeB = Std::tuple_size<typename T3::Shape>::value;
    constexpr auto shapeSizeBScale = Std::tuple_size<typename T4::Shape>::value;
    static_assert(shapeSizeC == SHAPE_DIM2 && shapeSizeA == SHAPE_DIM2 && shapeSizeAScale == SHAPE_DIM3 &&
                      shapeSizeB == SHAPE_DIM2 && shapeSizeBScale == SHAPE_DIM3,
                  "[MatmulMX ERROR]: Shape dim size shoulde be 2 and Scale Shape dim size shoulde be 3");

    constexpr auto staticL0CH = Std::tuple_element<shapeSizeC - SHAPE_DIM2, typename T0::TileShape>::type::value;
    constexpr auto staticL0CW = Std::tuple_element<shapeSizeC - 1, typename T0::TileShape>::type::value;
    constexpr auto staticL0AH = Std::tuple_element<shapeSizeA - SHAPE_DIM2, typename T1::TileShape>::type::value;
    constexpr auto staticL0AW = Std::tuple_element<shapeSizeA - 1, typename T1::TileShape>::type::value;
    constexpr auto staticL0AScaleW =
        Std::tuple_element<shapeSizeAScale - SHAPE_DIM2, typename T2::TileShape>::type::value;
    constexpr auto staticL0BH = Std::tuple_element<shapeSizeB - SHAPE_DIM2, typename T3::TileShape>::type::value;
    constexpr auto staticL0BW = Std::tuple_element<shapeSizeB - 1, typename T3::TileShape>::type::value;
    constexpr auto staticL0BScaleH =
        Std::tuple_element<shapeSizeBScale - SHAPE_DIM3, typename T4::TileShape>::type::value;

    int64_t validM = GetShape<0>(a);
    int64_t validK = GetShape<1>(a);
    int64_t validScaleK = GetShape<1>(aScale) * SHAPE_DIM2;
    int64_t validN = GetShape<1>(b);

    using tileL0CTensor = pto::TileAcc<typename T0::Type, staticL0CH, staticL0CW, -1, -1>;
    using tileL0ATensor = pto::TileLeft<typename T1::Type, staticL0AH, staticL0AW, -1, -1>;
    using tileL0AScaleTensor = pto::TileLeftScale<typename T1::Type, staticL0AH, staticL0AScaleW * SHAPE_DIM2, -1, -1>;
    using tileL0BTensor = pto::TileRight<typename T3::Type, staticL0BH, staticL0BW, -1, -1>;
    using tileL0BScaleTensor = pto::TileRightScale<typename T3::Type, staticL0BScaleH * SHAPE_DIM2, staticL0BW, -1, -1>;
    using tileBiasTensor =
        pto::Tile<pto::TileType::Bias, typename T5::Type, 1, staticL0BW, pto::BLayout::RowMajor, -1, -1>;

    validM = (validM + BLOCK_CUBE_M_N - 1) / BLOCK_CUBE_M_N * BLOCK_CUBE_M_N;
    tileL0ATensor l0a(validM, validK);
    tileL0AScaleTensor l0aScale(validM, validScaleK);
    tileL0BTensor l0b(validK, validN);
    tileL0BScaleTensor l0bScale(validScaleK, validN);
    tileL0CTensor l0c(validM, validN);
    tileBiasTensor biasT(1, validM);

    pto::TASSIGN(l0a, (uint64_t)a.GetAddr());
    pto::TASSIGN(l0aScale, (uint64_t)aScale.GetAddr());
    pto::TASSIGN(l0b, (uint64_t)b.GetAddr());
    pto::TASSIGN(l0bScale, (uint64_t)bScale.GetAddr());
    pto::TASSIGN(l0c, (uint64_t)c.GetAddr());
    pto::TASSIGN(biasT, (uint64_t)bias.GetAddr());
    pto::TMATMUL_MX(l0c, l0a, l0aScale, l0b, l0bScale, biasT);
}
#endif

template <typename config, typename globalData, typename tileData, typename V>
INLINE void TStoreExecute(globalData dstGlobal, tileData srcL0C, V &fixbuf, uint64_t scaleValue) {
    if constexpr (std::is_same<typename tileData::DType, int32_t>::value &&
                  std::is_same<typename globalData::DType, __gm__ half>::value) {
        if (scaleValue != 0) {
            pto::TSTORE<tileData, globalData, config::kIsAcc ? pto::AtomicType::AtomicAdd : pto::AtomicType::AtomicNone,
                config::kReluMode == 0 ? pto::ReluPreMode::NoRelu : pto::ReluPreMode::NormalRelu>(
                dstGlobal, srcL0C, scaleValue);
        } else {
            constexpr auto shapeSize = Std::tuple_size<typename V::Shape>::value;
            constexpr auto tileH = Std::tuple_element<shapeSize - SHAPE_DIM2, typename V::TileShape>::type::value;
            constexpr auto tileW = Std::tuple_element<shapeSize - 1, typename V::TileShape>::type::value;
            using fpTileData =
                pto::Tile<pto::TileType::Scaling, uint64_t, tileH, tileW, pto::BLayout::RowMajor, -1, -1>;
            int64_t fixShape0 = GetShape<0>(fixbuf);
            int64_t fixShape1 = GetShape<1>(fixbuf);
            if (fixShape0 == 0 || fixShape1 == 0) {
                return;
            }
            fpTileData fpData(fixShape0, fixShape1);
            pto::TASSIGN(fpData, (uint64_t)fixbuf.GetAddr());
            pto::TSTORE_FP<tileData, globalData, fpTileData,
                config::kIsAcc ? pto::AtomicType::AtomicAdd : pto::AtomicType::AtomicNone,
                config::kReluMode == 0 ? pto::ReluPreMode::NoRelu : pto::ReluPreMode::NormalRelu>(
                dstGlobal, srcL0C, fpData);
        }
    } else {
        pto::TSTORE<tileData, globalData, config::kIsAcc ? pto::AtomicType::AtomicAdd : pto::AtomicType::AtomicNone,
            config::kReluMode == 0 ? pto::ReluPreMode::NoRelu : pto::ReluPreMode::NormalRelu>(
            dstGlobal, srcL0C);
    }
}

// Copy data from L0C to DDR with NZ -> ND format
template <typename config, typename T, typename U, typename V>
INLINE void TStoreNZ2ND(
    T &dst, U &src, V &fixbuf, const int64_t &offset0, const int64_t &offset1, uint64_t scaleValue = 0) {
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
        pto::SLayout::RowMajor, pto::TileConfig::fractalCSize, pto::PadValue::Null, pto::CompactMode::Normal>;
    globalData dstGlobal((__gm__ typename T::Type *)(dst.GetAddr() + gmOffset),
        pto::Shape<1, 1, 1, -1, -1>(srcShape0, srcShape1), pto::Stride<1, 1, 1, -1, -1>(dstStride0, dstStride1));
    tileData srcL0C(srcShape0, srcShape1);
    pto::TASSIGN(srcL0C, (uint64_t)src.GetAddr());
    TStoreExecute<config, globalData, tileData>(dstGlobal, srcL0C, fixbuf, scaleValue);
    return;
}

// Copy data from L0C to DDR with NZ -> NZ format
template <typename config, typename T, typename U, typename V>
INLINE void TStoreNZ2NZ(T &dst, U &src, V &fixbuf, const int64_t &offset0, const int64_t &offset1, const int64_t &curH,
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
        pto::SLayout::RowMajor, pto::TileConfig::fractalCSize, pto::PadValue::Null, pto::CompactMode::Normal>;
    tileData srcL0C(srcShape0, srcShape1);
    pto::TASSIGN(srcL0C, (uint64_t)src.GetAddr());
    TStoreExecute<config, globalData, tileData>(dstGlobal, srcL0C, fixbuf, scaleValue);
    return;
}

// Copy data from L0C to DDR with quantization ability
template <typename config, typename Coord, typename T, typename U, typename V>
TILEOP void TStore(
    T &dst, U &src, V &fixbuf, const Coord &coord, const int64_t &curH, const int64_t &curW, uint64_t scaleValue = 0) {
    if (!CheckShapeValid(dst, src)) {
        return;
    }
    constexpr auto shapeSize = Std::tuple_size<typename T::Shape>::value;
    static_assert(shapeSize == SHAPE_DIM2 && Std::tuple_size<Coord>::value == SHAPE_DIM2, "Shape Size should be 2 Dim");
    int64_t offset0 = coord.GetValue();
    int64_t offset1 = static_cast<const Std::tuple<size_t> &>(coord).GetValue();
    if constexpr (U::FORMAT == Hardware::L0C && T::FORMAT == Hardware::GM) {
        if constexpr (config::kMode == CopyOutMode::NZ2ND) {
            TStoreNZ2ND<config>(dst, src, fixbuf, offset0, offset1, scaleValue);
        } else {
            TStoreNZ2NZ<config>(dst, src, fixbuf, offset0, offset1, curH, curW, scaleValue);
        }
    }
}
// L1 spill
// When L1 space is insufficient, spill to GM. (Supported on A2/A3 only.)
template <typename config, typename Coord, typename T, typename U>
TILEOP void TStore(T &dst, U &src, const Coord &coord)
{
    constexpr auto shapeSize = Std::tuple_size<typename U::Shape>::value;
    constexpr auto staticL1H = Std::tuple_element<shapeSize - SHAPE_DIM2, typename U::TileShape>::type::value;
    constexpr auto staticL1W = Std::tuple_element<shapeSize - 1, typename U::TileShape>::type::value;
    int64_t dstShape0 = GetShape<0>(dst);
    int64_t dstShape1 = GetShape<1>(dst);
    int64_t dstStride0 = GetStride<0>(dst);
    int64_t dstStride1 = GetStride<1>(dst);
    int64_t srcShape0 = GetShape<0>(src);
    int64_t srcShape1 = GetShape<1>(src);
    using shapeDim2 = pto::Shape<1, 1, 1, -1, -1>;
    using strideDim2 = pto::Stride<1, 1, 1, -1, -1>;
    if constexpr (config::kMode == CopyOutMode::ND2ND) {
        using globalData = pto::GlobalTensor<typename T::Type, shapeDim2, strideDim2, pto::Layout::ND>;
        using tileData =
            pto::Tile<pto::TileType::Mat, typename U::Type, staticL1H, staticL1W, pto::BLayout::RowMajor, -1, -1>;
        globalData dstGlobal((__gm__ typename T::Type *)(dst.GetAddr()),
            shapeDim2(srcShape0, srcShape1), strideDim2(dstStride0, dstStride1));
        tileData srcL1(srcShape0, srcShape1);
        pto::TASSIGN(srcL1, (uint64_t)src.GetAddr());
        pto::TSTORE<tileData, globalData>(dstGlobal, srcL1);
    }
}

template <int64_t blockSize, typename globalData, typename tileData, typename DstT, typename SrcT, typename BlockT, typename OffsetT>
INLINE void GatherExecute(DstT dst, SrcT src, BlockT block, OffsetT offset, uint64_t offsetsStartOffset,
    uint64_t srcColumnStartOffset, uint64_t GMBlockTableOffset, int64_t srcShape0, int64_t srcShape1,
    int64_t srcStride0, int64_t srcStride1, int64_t dstShape0, int64_t dstShape1, int64_t srcCol, int64_t loop,
    int64_t c0Size) {
    for (int64_t i = 0; i < loop; i++) {
        uint64_t gatherOffset = offset.GetAddr()[i + offsetsStartOffset];
        gatherOffset = CalaOffset2PageAttention<uint64_t, typename BlockT::Type, blockSize>(
            block.GetAddr() + GMBlockTableOffset, gatherOffset);
        globalData src0Global(
            (__gm__ typename SrcT::Type *)(src.GetAddr() + gatherOffset * srcCol + srcColumnStartOffset),
            pto::Shape<1, 1, 1, -1, -1>(srcShape0, srcShape1), pto::Stride<1, 1, 1, -1, -1>(srcStride0, srcStride1));
        tileData dstL1(dstShape0, dstShape1);
        pto::TASSIGN(dstL1, (uint64_t)((__cbuf__ typename DstT::Type *)dst.GetAddr() + i * c0Size));
        pto::TLOAD(dstL1, src0Global);
    }
}

template <int64_t blockSize, typename DstT, typename SrcT, typename BlockT, typename OffsetT, typename SrcCoord,
    typename OffsetCoord, typename BlockCoord>
TILEOP void TGatherInL1(DstT dst, SrcT src, BlockT block, OffsetT offset, SrcCoord srcCoord, OffsetCoord offsetCoord,
    BlockCoord blockCoord) {
    if (!CheckShapeValid(dst, src)) {
        return;
    }
    constexpr auto shapeSize = Std::tuple_size<typename DstT::Shape>::value;
    constexpr int64_t c0Size = BLOCK_ALIGN_BYTE / sizeof(typename DstT::Type);
    int64_t srcShape1 = GetShape<1>(src);
    int64_t srcStride0 = GetStride<0>(src);
    int64_t srcStride1 = GetStride<1>(src);
    int64_t dstShape0 = GetShape<0>(dst);
    int64_t dstShape1 = GetShape<1>(dst);
    int64_t blockShape1 = GetShape<1>(block);
    int64_t offsetShape1 = GetShape<1>(offset);
    uint64_t srcColumnStartOffset = srcCoord.GetValue();
    uint64_t offsetsRowStartOffset = offsetCoord.GetValue();
    uint64_t offsetsColumnStartOffset = static_cast<const Std::tuple<size_t> &>(offsetCoord).GetValue();
    uint64_t offsetsStartOffset = offsetsRowStartOffset * offsetShape1 + offsetsColumnStartOffset;
    uint64_t GMBlockTableOffset0 = blockCoord.GetValue();
    uint64_t GMBlockTableOffset1 = static_cast<const Std::tuple<size_t> &>(blockCoord).GetValue();
    uint64_t GMBlockTableOffset = GMBlockTableOffset0 * blockShape1 + GMBlockTableOffset1;
    constexpr auto staticL1H = Std::tuple_element<shapeSize - SHAPE_DIM2, typename DstT::TileShape>::type::value;
    constexpr auto staticL1W = Std::tuple_element<shapeSize - 1, typename DstT::TileShape>::type::value;
    using shapeDim2 = pto::Shape<1, 1, 1, -1, -1>;
    using strideDim2 = pto::Stride<1, 1, 1, -1, -1>;
    using globalData = pto::GlobalTensor<typename SrcT::Type, shapeDim2, strideDim2, pto::Layout::ND>;
    if (dstShape1 % c0Size > 0) {
        using tileData = pto::Tile<pto::TileType::Mat, typename DstT::Type, staticL1H, staticL1W,
            pto::BLayout::ColMajor, -1, -1, pto::SLayout::RowMajor>;
        GatherExecute<blockSize, globalData, tileData>(dst, src, block, offset, offsetsStartOffset, srcColumnStartOffset, GMBlockTableOffset,
            1, staticL1W, srcStride0, srcStride1, dstShape0, dstShape1, srcShape1, dstShape0, c0Size);
    } else {
        using tileData = pto::Tile<pto::TileType::Mat, typename DstT::Type, staticL1W / c0Size, staticL1H * c0Size,
            pto::BLayout::RowMajor, -1, -1>;
        // 这里需要采用性能更高的ND2ND的搬运方式。
        // GM上将(1, dstShape1)的数据，转变为(dstShape1 / c0Size, c0Size)。
        // L1上将(staticL1W / c0Size, staticL1H / 16, 16, c0Size)的NZ数据，对每c0Size列做展平为一行，得到(staticL1W / c0Size, staticL1H * c0Size)的ND格式。
        // 那么L1上有效数据就是(dstShape1 / c0Size, c0Size)。
        GatherExecute<blockSize, globalData, tileData>(dst, src, block, offset, offsetsStartOffset, srcColumnStartOffset, GMBlockTableOffset,
            dstShape1 / c0Size, c0Size, c0Size, 1, dstShape1 / c0Size, c0Size, srcShape1, dstShape0, c0Size);
    }
}
#endif // TILEOP_TILE_OPERATOR_CUBE_PTO__H