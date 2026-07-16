/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COMMON_HELPERS_HPP
#define COMMON_HELPERS_HPP

#include "kernel_operator.h"

#include <type_traits>

template <typename T>
AICORE inline __gm__ T *tensorListEntry(GM_ADDR tensorList, uint32_t index)
{
    __gm__ uint64_t *tensorListBase = reinterpret_cast<__gm__ uint64_t *>(tensorList);
    uint64_t entryTableOffset = *tensorListBase;
    __gm__ uint64_t *entryTable = tensorListBase + (entryTableOffset >> 3);
    return reinterpret_cast<__gm__ T *>(*(entryTable + index));
}

template <uint32_t Align, class T>
AICORE inline constexpr T ceilDiv(T value)
{
    return (value + static_cast<T>(Align) - 1) / static_cast<T>(Align);
}

template <class T, class U>
AICORE inline constexpr auto ceilDiv(T lhs, U rhs)
{
    using Common = std::common_type_t<T, U>;
    Common lhsValue = static_cast<Common>(lhs);
    Common rhsValue = static_cast<Common>(rhs);
    return (lhsValue + rhsValue - 1) / rhsValue;
}

template <uint32_t Align, class T>
AICORE inline constexpr T roundUp(T value)
{
    return ceilDiv<Align>(value) * static_cast<T>(Align);
}

template <class T, class U>
AICORE inline constexpr auto alignUp(T value, U align)
{
    using Common = std::common_type_t<T, U>;
    Common alignValue = static_cast<Common>(align);
    return ceilDiv(static_cast<Common>(value), alignValue) * alignValue;
}

AICORE inline int64_t tokenPerExpertOffset(int32_t epIdx, int32_t rank, int32_t groupIdx,
                                           int32_t paddedExpertNumAligned, int32_t expertPerRank)
{
    return static_cast<int64_t>(epIdx) * paddedExpertNumAligned + static_cast<int64_t>(rank) * expertPerRank + groupIdx;
}

AICORE inline void V5DcciGmRangeNoFence(__gm__ void *ptr, uint64_t bytes)
{
    if (bytes == 0) {
        return;
    }
    constexpr uint64_t cacheLineBytes = 64U;
    const uint64_t start = reinterpret_cast<uint64_t>(ptr) & ~(cacheLineBytes - 1U);
    const uint64_t end = (reinterpret_cast<uint64_t>(ptr) + bytes + cacheLineBytes - 1U) & ~(cacheLineBytes - 1U);
    for (uint64_t addr = start; addr < end; addr += cacheLineBytes) {
        __asm__ __volatile__("");
        dcci(reinterpret_cast<__gm__ void *>(addr), SINGLE_CACHE_LINE);
        __asm__ __volatile__("");
    }
}

AICORE inline void V5DcciGmRange(__gm__ void *ptr, uint64_t bytes)
{
    V5DcciGmRangeNoFence(ptr, bytes);
    dsb(DSB_DDR);
}

#endif // COMMON_HELPERS_HPP
