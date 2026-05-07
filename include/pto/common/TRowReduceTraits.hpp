/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_TROW_REDUCE_TRAITS_HPP
#define PTO_TROW_REDUCE_TRAITS_HPP

#include <type_traits>

namespace pto {

template <typename T, typename TileOut, typename TileIn>
struct TRowReduceFP32OptTraits {
    static constexpr bool IsTargetLayout = !TileOut::isBoxedLayout && !TileOut::isRowMajor && TileOut::ValidCol == 1;
    static constexpr bool IsFP32 = std::is_same_v<T, float>;

    static constexpr bool ShapeOf64x128 =
        TileIn::Rows == 64 && TileIn::ValidRow == 64 && TileIn::Cols == 128 && TileIn::ValidCol == 128;
    static constexpr bool ShapeOf32x256 =
        TileIn::Rows == 32 && TileIn::ValidRow == 32 && TileIn::Cols == 256 && TileIn::ValidCol == 256;
    static constexpr bool ShapeOf16x512 =
        TileIn::Rows == 16 && TileIn::ValidRow == 16 && TileIn::Cols == 512 && TileIn::ValidCol == 512;
    static constexpr bool ShapeOf8x1024 =
        TileIn::Rows == 8 && TileIn::ValidRow == 8 && TileIn::Cols == 1024 && TileIn::ValidCol == 1024;

    static constexpr bool CanOptimize = IsTargetLayout && IsFP32;
};

} // namespace pto

#endif // PTO_TROW_REDUCE_TRAITS_HPP
