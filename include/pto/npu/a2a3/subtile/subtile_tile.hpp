/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_SUBTILE_TILE_HPP
#define PTO_SUBTILE_TILE_HPP

#include <pto/common/constants.hpp>
#include <pto/common/utils.hpp>

namespace pto {

// Subtile 1D: runtime length, can be > VL, uses counter mode in ops
// Data is assumed contiguous

template <typename T>
struct Subtile1D {
    using DType = T;
    __ubuf__ T *ptr = nullptr;
    uint32_t len = 0; // number of elements

    PTO_INTERNAL Subtile1D(__ubuf__ T *p, uint32_t n) : ptr(p), len(n) {}

    PTO_INTERNAL __ubuf__ T *data() const { return ptr; }
    PTO_INTERNAL uint32_t length() const { return len; }
};

// Subtile 2D: runtime rows/cols (cols <= VL, rows <= 255), uses mask + hw repeat
// Row-major with configurable stride

template <typename T>
struct Subtile2D {
    using DType = T;
    __ubuf__ T *ptr = nullptr;
    uint16_t rows = 0;
    uint16_t cols = 0;
    uint16_t rowStride = 0; // in elements

    PTO_INTERNAL Subtile2D(__ubuf__ T *p, uint16_t r, uint16_t c, uint16_t stride)
        : ptr(p), rows(r), cols(c), rowStride(stride) {}

    PTO_INTERNAL __ubuf__ T *data() const { return ptr; }
    PTO_INTERNAL uint16_t getRows() const { return rows; }
    PTO_INTERNAL uint16_t getCols() const { return cols; }
    PTO_INTERNAL uint16_t getRowStride() const { return rowStride; }
};

} // namespace pto

#endif // PTO_SUBTILE_TILE_HPP
