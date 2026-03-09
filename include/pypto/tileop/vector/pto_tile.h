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
 * \file pto_tile.h
 * \brief
 */

#ifndef TILEOP_TILE_OPERATOR_PTO_TILE__H
#define TILEOP_TILE_OPERATOR_PTO_TILE__H
#include <cstddef>

#include "utils/layout.h"
#include "utils/tile_tensor.h"

#ifdef __DAV_V220
#define PTO_WITH_LAST_USE(OP, ...) OP
#else
#define PTO_WITH_LAST_USE(OP, ...) [[pto::last_use(__VA_ARGS__)]]OP
#endif

template <typename Tuple, size_t index, size_t default_value = 1, bool use_default = false>
__aicore__ inline constexpr size_t GetTupleElement(const Tuple &t) {
    static_assert(index < MAX_DIMS, "The index of tuple is out of range.");
    constexpr auto size = Std::tuple_size<Tuple>::value;
    if constexpr (use_default || (size < MAX_DIMS && index < (MAX_DIMS - size))) {
        return default_value;
    } else {
        return Std::get<index + size - MAX_DIMS>(t);
    }
}

template <typename T, typename Shape, typename Stride, bool need_mask = false>
class PtoGlobal {
public:
    using Dtype = std::conditional_t<std::is_same_v<typename T::Type, bool>, uint8_t, typename T::Type>;
    using Type = pto::GlobalTensor<Dtype, pto::Shape<-1, -1, -1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>>;

    __aicore__ inline PtoGlobal(__gm__ typename T::Type *addr, const Shape &shape, const Stride &stride)
        : data_((__gm__ Dtype *)(addr),
              pto::Shape(GetTupleElement<Shape, DIM_1ST, 1, need_mask>(shape),
                  GetTupleElement<Shape, DIM_2ND, 1, need_mask>(shape),
                  GetTupleElement<Shape, DIM_3RD, 1, need_mask>(shape), GetTupleElement<Shape, DIM_4TH>(shape),
                  GetTupleElement<Shape, DIM_5TH>(shape)),
              pto::Stride(GetTupleElement<Stride, DIM_1ST, 0, need_mask>(stride),
                  GetTupleElement<Stride, DIM_2ND, 0, need_mask>(stride),
                  GetTupleElement<Stride, DIM_3RD, 0, need_mask>(stride), GetTupleElement<Stride, DIM_4TH, 0>(stride),
                  GetTupleElement<Stride, DIM_5TH, 0>(stride))) {}

    __aicore__ inline PtoGlobal(const Shape &shape, const Stride &stride) : PtoGlobal(0x0, shape, stride) {}

    __aicore__ inline void Assign(__gm__ typename T::Type *addr) { pto::TASSIGN(data_, (__gm__ Dtype *)addr); }

    inline Type &Data() { return data_; }

private:
    Type data_;
};

template <typename T>
__aicore__ inline size_t GenTileOffset(const T &tensor, const TileOffset &offsets) {
    const auto layout = tensor.GetLayout();
    size_t offset = Std::get<DIM_1ST>(offsets) * layout.template GetStrideDim<DIM_1ST, MAX_DIMS>();
    offset += Std::get<DIM_2ND>(offsets) * layout.template GetStrideDim<DIM_2ND, MAX_DIMS>();
    offset += Std::get<DIM_3RD>(offsets) * layout.template GetStrideDim<DIM_3RD, MAX_DIMS>();
    return offset;
}

template <typename T>
__aicore__ inline size_t GenTileOffset(const T &tensor, const TileOffset4Dim &offsets) {
    const auto layout = tensor.GetLayout();
    size_t offset = Std::get<DIM_1ST>(offsets) * layout.template GetStrideDim<DIM_1ST, MAX_DIMS>();
    offset += Std::get<DIM_2ND>(offsets) * layout.template GetStrideDim<DIM_2ND, MAX_DIMS>();
    offset += Std::get<DIM_3RD>(offsets) * layout.template GetStrideDim<DIM_3RD, MAX_DIMS>();
    offset += Std::get<DIM_4TH>(offsets) * layout.template GetStrideDim<DIM_4TH, MAX_DIMS>();
    return offset;
}

template <typename T, bool Mergeable = false>
__aicore__ inline constexpr size_t GetMergedAxisIfNeed() {
    if constexpr (Mergeable) {
        constexpr auto size = Std::tuple_size<typename T::TileShape>::value;
        return TileOp::GetOutterAxisMergeResult<size, typename T::TileShape>();
    } else {
        return TileOp::GetTensorTileShapeDim<T, DIM_4TH, MAX_DIMS>();
    }
}

template <typename T, bool Mergeable = false>
__aicore__ inline constexpr int GetValidHeight() {
    if constexpr (Mergeable) {
        constexpr auto size = Std::tuple_size<typename T::Shape>::value;
        return TileOp::GetOutterAxisMergeResult<size, typename T::Shape>();
    } else if constexpr (T::IsStaticLayout()) {
        return TileOp::GetTensorShapeDim<T, DIM_4TH, MAX_DIMS>();
    } else {
        return -1;
    }
}

template <typename T>
__aicore__ inline constexpr int GetValidWidth() {
    if constexpr (T::IsStaticLayout()) {
        return TileOp::GetTensorShapeDim<T, DIM_5TH, MAX_DIMS>();
    } else {
        return -1;
    }
}

template <typename T, pto::BLayout Layout = pto::BLayout::RowMajor, bool Mergeable = false>
class PtoTile {
private:
    static constexpr auto size = Std::tuple_size<typename T::Shape>::value;
    static constexpr auto tileH = GetMergedAxisIfNeed<T, Mergeable>();
    static constexpr auto tileW = TileOp::GetTensorTileShapeDim<T, DIM_5TH, MAX_DIMS>();
    static constexpr auto validH = GetValidHeight<T, Mergeable>();
    static constexpr auto validW = GetValidWidth<T>();

public:
    using Dtype = std::conditional_t<std::is_same_v<typename T::Type, bool>, uint8_t, typename T::Type>;
    using Type = pto::Tile<pto::TileType::Vec, Dtype, tileH, tileW, Layout, validH, validW>;

    __aicore__ inline PtoTile() : data_() {
        static_assert(T::IsStaticLayout(), "Only valild for static layout tile tensor.");
    }

    __aicore__ inline PtoTile(const uint64_t &addr) : PtoTile() { pto::TASSIGN(data_, addr); }

    __aicore__ inline PtoTile(const int &h, const int &w) {
        if constexpr (!T::IsStaticLayout()) {
            Type tile(h, w);
            data_ = tile;
        }
    }

    __aicore__ inline PtoTile(const int &h, const int &w, const uint64_t addr) : PtoTile(h, w) {
        pto::TASSIGN(data_, addr);
    }

    __aicore__ inline PtoTile(const T &tensor)
        : PtoTile(tensor.GetLayout().template GetShapeDim<DIM_4TH, MAX_DIMS>(),
              tensor.GetLayout().template GetShapeDim<DIM_5TH, MAX_DIMS>()) {}

    __aicore__ inline Type &Data() { return data_; }

    __aicore__ inline const Type &Data() const { return data_; }

    __aicore__ inline void Assign(uint64_t addr) { pto::TASSIGN(data_, addr); }

    __aicore__ inline void Assign(uint64_t addr, uint64_t element_cnt) {
        pto::TASSIGN(data_, addr + (element_cnt * sizeof(typename T::Type)));
    }

    __aicore__ inline void Assign(T &tensor) { Assign((uint64_t)(tensor.GetAddr())); }

    __aicore__ inline void Assign(T &tensor, const TileOffset &offsets) {
        pto::TASSIGN(data_, (uint64_t)(tensor.GetAddr() + GenTileOffset(tensor, offsets) * sizeof(typename T::Type)));
    }

private:
    Type data_;
};
#endif // TILEOP_TILE_OPERATOR_PTO_TILE__H