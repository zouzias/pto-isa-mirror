/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TPARTIALBINOPS_HPP
#define TPARTIALBINOPS_HPP

#include <pto/common/constants.hpp>
#include <pto/common/utils.hpp>

namespace pto {

template <typename T>
struct Padding {
    using Type = std::conditional_t<sizeof(T) == sizeof(uint32_t), uint32_t,
                                    std::conditional_t<sizeof(T) == sizeof(uint16_t), uint16_t, uint8_t>>;

    PTO_INTERNAL static constexpr Type GetPaddingMin()
    {
        if constexpr (std::is_same_v<T, float>) {
            return (Type)0xff800000UL;
        } else if constexpr (std::is_same_v<T, half>) {
            return (Type)0xfc00UL;
        } else if constexpr (std::is_same_v<T, bfloat16_t>) {
            return (Type)0xff80UL;
        } else if constexpr (std::is_same_v<T, int32_t>) {
            return (Type)0x80000000UL;
        } else if constexpr (std::is_same_v<T, int16_t>) {
            return (Type)0x8000UL;
        } else if constexpr (std::is_same_v<T, int8_t>) {
            return (Type)0x80UL;
        } else {
            return (Type)0;
        }
    }

    PTO_INTERNAL static constexpr Type GetPaddingMax()
    {
        if constexpr (std::is_same_v<T, float>) {
            return (Type)0x7f800000UL;
        } else if constexpr (std::is_same_v<T, half>) {
            return (Type)0x7c00UL;
        } else if constexpr (std::is_same_v<T, bfloat16_t>) {
            return (Type)0x7f80UL;
        } else if constexpr (std::is_same_v<T, int32_t>) {
            return (Type)0x7fffffffUL;
        } else if constexpr (std::is_same_v<T, int16_t>) {
            return (Type)0x7fffUL;
        } else if constexpr (std::is_same_v<T, int8_t>) {
            return (Type)0x7fUL;
        } else {
            return (Type)(~(Type)0);
        }
    }

    static constexpr Type Null = (Type)0;
    static constexpr Type Zero = (Type)0;
    static constexpr Type Min = GetPaddingMin();
    static constexpr Type Max = GetPaddingMax();
};

} // namespace pto

#endif
