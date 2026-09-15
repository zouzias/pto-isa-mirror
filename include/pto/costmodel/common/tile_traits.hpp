/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#pragma once

#include <concepts>
#include <cstdint>
#include <type_traits>
#include <pto/common/type.hpp>
#include <pto/cpu/Hifloat8.hpp>
#include <pto/cpu/MXTypes.hpp>

namespace pto::mocker {

template <typename T>
struct TileTraits {
    static constexpr int rows = 0;
    static constexpr int cols = 0;
    static constexpr const char* dtype_str() { return "unknown"; }
};

template <typename T>
concept HasTileDims = requires {
    typename T::DType;
    { T::Rows } -> std::convertible_to<int>;
    { T::Cols } -> std::convertible_to<int>;
};

template <HasTileDims T>
struct TileTraits<T> {
    using DType = typename T::DType;
    static constexpr int rows = T::Rows;
    static constexpr int cols = T::Cols;

    static constexpr const char* dtype_str()
    {
        if constexpr (std::is_same_v<DType, float>)
            return "fp32";
        else if constexpr (std::is_same_v<DType, int32_t>)
            return "int32";
        else if constexpr (std::is_same_v<DType, uint32_t>)
            return "uint32";
        else if constexpr (std::is_same_v<DType, int16_t>)
            return "int16";
        else if constexpr (std::is_same_v<DType, uint16_t>)
            return "uint16";
        else if constexpr (std::is_same_v<DType, int8_t>)
            return "int8";
        else if constexpr (std::is_same_v<DType, uint8_t>)
            return "uint8";
        else if constexpr (std::is_same_v<DType, float8_e4m3_t>)
            return "fp8_e4m3";
        else if constexpr (std::is_same_v<DType, float8_e5m2_t>)
            return "fp8_e5m2";
        else if constexpr (std::is_same_v<DType, hifloat8_t>)
            return "hif8";
        else if constexpr (std::is_same_v<DType, float4_e1m2x2_t>)
            return "fp4_e1m2";
        else if constexpr (std::is_same_v<DType, float4_e2m1x2_t>)
            return "fp4_e2m1";
        else if constexpr (std::is_same_v<DType, bfloat16_t> && !std::is_same_v<bfloat16_t, half>)
            return "bf16";
        else if constexpr (std::is_same_v<DType, half>)
            return "fp16";
        else
            return "unknown";
    }
};

} // namespace pto::mocker
