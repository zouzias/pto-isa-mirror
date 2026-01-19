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
 * \file tile_tensor.h
 * \brief
 */

#ifndef TILEOP_UTILS_TILE_TENSOR_H
#define TILEOP_UTILS_TILE_TENSOR_H

#include "common_type.h"

template <typename T, typename LA, Hardware FMT = Hardware::UB>
struct TileTensor {
    using Type = T;
    using LayoutType = LA;
    using Shape = typename LA::Shape;
    using Stride = typename LA::Stride;
    using TileShape = typename LA::TileShape;
    static constexpr Hardware FORMAT = FMT;

    __aicore__ inline TileTensor(uint64_t addr, LA layout) : addr_(addr), layout_(layout) {}
    __aicore__ inline TileTensor(uint64_t addr, Shape shape) : addr_(addr), layout_(LA(shape)) {}

    __aicore__ inline TileTensor(uint64_t addr) : addr_(addr) {}
    __aicore__ inline uint64_t GetAddr() { return addr_; }
    __aicore__ inline LA GetLayout() { return layout_; }
    __aicore__ inline const LA GetLayout() const { return layout_; }
    __aicore__ inline static constexpr bool IsStaticLayout() { return LayoutType::IsStaticLayout(); }
    __aicore__ inline constexpr Hardware GetPhyType() { return FORMAT; }
    __aicore__ inline Shape GetShape() { return layout_.GetShape(); }
    __aicore__ inline Stride GetStride() { return layout_.GetStride(); }

private:
    uint64_t addr_;
    LA layout_;
};

template <typename T, typename LA>
struct TileTensor<T, LA, Hardware::GM> {
    using Type = T;
    using LayoutType = LA;
    using Shape = typename LA::Shape;
    using Stride = typename LA::Stride;
    using TileShape = typename LA::TileShape;
    static constexpr Hardware FORMAT = Hardware::GM;

    __aicore__ inline TileTensor(T *addr, LA layout) : addr_(addr), layout_(layout) {}
    __aicore__ inline TileTensor(T *addr, Shape shape) : addr_(addr), layout_(LA(shape)) {}

    __aicore__ inline TileTensor(T *addr) : addr_(addr) {}
    __aicore__ inline T *GetAddr() { return addr_; }
    __aicore__ inline LA GetLayout() { return layout_; }
    __aicore__ inline const LA GetLayout() const { return layout_; }
    __aicore__ inline static constexpr bool IsStaticLayout() { return LayoutType::IsStaticLayout(); }
    __aicore__ inline constexpr Hardware GetPhyType() { return FORMAT; }
    __aicore__ inline Shape GetShape() { return layout_.GetShape(); }
    __aicore__ inline Stride GetStride() { return layout_.GetStride(); }

private:
    T *addr_;
    LA layout_;
};

template <typename T, typename LA, Hardware FMT = Hardware::UB>
TILEOP TileTensor<T, LA, FMT> MakeTensor(__ubuf__ T *addr, LA layout) {
    return TileTensor<T, LA, FMT>(addr, layout);
}
#endif // TILEOP_UTILS_TILE_TENSOR_H
