/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef TPRINT_CPU_HPP
#define TPRINT_CPU_HPP

#include <iomanip>
#include <iostream>
#include <pto/common/pto_tile.hpp>
#include "pto/cpu/tile_offsets.hpp"

namespace pto {

template <typename T>
constexpr bool is_half_type_v = std::is_same_v<T, half> || std::is_same_v<T, bfloat16_t>;

template <PrintFormat Format, typename T>
PTO_INTERNAL void PrintValue(std::ostream &os, T val)
{
    if constexpr (std::is_floating_point_v<T> || is_half_type_v<T>) {
        if constexpr (Format == PrintFormat::Width8_Precision4) {
            os << std::setw(8) << std::fixed << std::setprecision(4) << static_cast<float>(val);
        } else if constexpr (Format == PrintFormat::Width8_Precision2) {
            os << std::setw(8) << std::fixed << std::setprecision(2) << static_cast<float>(val);
        } else if constexpr (Format == PrintFormat::Width10_Precision6) {
            os << std::setw(10) << std::fixed << std::setprecision(6) << static_cast<float>(val);
        }
    } else if constexpr (std::is_signed_v<T>) {
        if constexpr (Format == PrintFormat::Width10_Precision6) {
            os << std::setw(10) << static_cast<int>(val);
        } else {
            os << std::setw(8) << static_cast<int>(val);
        }
    } else if constexpr (std::is_unsigned_v<T>) {
        if constexpr (Format == PrintFormat::Width10_Precision6) {
            os << std::setw(10) << static_cast<unsigned int>(val);
        } else {
            os << std::setw(8) << static_cast<unsigned int>(val);
        }
    } else {
        os << val;
    }
}

template <PrintFormat Format = PrintFormat::Width8_Precision4, typename T>
PTO_INTERNAL void TPRINT_IMPL(T &src)
{
    using DType = typename T::DType;
    std::cout << "TPRINT " << src.GetValidRow() << "x" << src.GetValidCol() << '\n';
    for (unsigned r = 0; r < src.GetValidRow(); ++r) {
        for (unsigned c = 0; c < src.GetValidCol(); ++c) {
            if (c != 0) {
                std::cout << ' ';
            }
            PrintValue<Format>(std::cout, static_cast<DType>(src.data()[GetTileElementOffset<T>(r, c)]));
        }
        std::cout << '\n';
    }
}

template <PrintFormat Format = PrintFormat::Width8_Precision4, typename TileData, typename GlobalData>
PTO_INTERNAL void TPRINT_IMPL(TileData &src, GlobalData &tmp)
{
    TPRINT_IMPL<Format>(src);
}

} // namespace pto

#endif
