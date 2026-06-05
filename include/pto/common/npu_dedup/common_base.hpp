/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMMON_NPU_DEDUP_COMMON_BASE_HPP
#define PTO_COMMON_NPU_DEDUP_COMMON_BASE_HPP

#include <cstdint>
#include <pto/common/type.hpp>

namespace pto {

template <typename T, int U, int... Args>
AICORE constexpr bool SupportBytes()
{
    if constexpr (sizeof...(Args) > 0) {
        return sizeof(T) == U || SupportBytes<T, Args...>();
    }
    return sizeof(T) == U;
}

using MaskReg = vector_bool;
using UnalignReg = vector_align;
using AddrReg = vector_address;

template <typename T>
PTO_INTERNAL MaskReg CreatePredicate(uint32_t &scalar)
{
    MaskReg reg;
    if constexpr (sizeof(T) == 1) {
        reg = plt_b8(scalar, POST_UPDATE);
    } else if constexpr (sizeof(T) == 2) {
        reg = plt_b16(scalar, POST_UPDATE);
    } else if constexpr (sizeof(T) == 4) {
        reg = plt_b32(scalar, POST_UPDATE);
    }
    return reg;
}

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

template <typename DstTile, typename SrcTile, AccToVecMode mode, QuantMode_t quantPre>
PTO_INTERNAL constexpr uint8_t GetDualDstCtl()
{
    if constexpr (mode == AccToVecMode::DualModeSplitM || mode == AccToVecMode::DualModeSplitN) {
        static_assert(quantPre == QuantMode_t::NoQuant, "Quant is not support in dual Dst Mode.");
        static_assert((!(!DstTile::isRowMajor && DstTile::SFractal == SLayout::NoneBox)),
                      "Dual Dst Mode is not support in nz2dn.");
        return ((mode == AccToVecMode::DualModeSplitM) ? 1 : 2);
    }
    return 0;
}

} // namespace pto

#endif
