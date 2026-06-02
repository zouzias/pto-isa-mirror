/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TCI_HPP
#define TCI_HPP

#include <pto/common/constants.hpp>
#include <pto/common/utils.hpp>
#include "common.hpp"
#include "utils.hpp"

namespace pto {
template <typename TileData, typename T>
PTO_INTERNAL void CheckValid()
{
    static_assert((std::is_same<typename TileData::DType, T>::value), "Fix: TCI expect src and dst same datatype");
    static_assert((sizeof(typename TileData::DType) == 4 || (sizeof(typename TileData::DType) == 2)),
                  "Fix: TCI expect b32 or b16");
    static_assert((TileData::Rows == 1), "Fix: TCI expect row is 1");
}

template <typename TileData, typename T, int descending = 0>
__tf__ AICORE void Tci(typename TileData::TileDType __out__ dst, T start, unsigned validCol)
{
    using Tdst = typename TileData::DType;
    __ubuf__ Tdst *dstPtr = (__ubuf__ Tdst *)__cce_get_tile_ptr(dst);
    // scalar
    if constexpr (descending) {
        for (int32_t j = 0; j < validCol; j++) {
            *(dstPtr + j) = start - j;
        }
    } else {
        for (int32_t j = 0; j < validCol; j++) {
            *(dstPtr + j) = start + j;
        }
    }
}

template <typename TileData, typename T, int descending>
PTO_INTERNAL void TCI_IMPL(TileData &dst, T start)
{
    CheckValid<TileData, T>();
    unsigned validCol = dst.GetValidCol();
    Tci<TileData, T, descending>(dst.data(), start, validCol);
}

template <typename TileData, typename TileDataTmp, typename T, int descending = 0>
__tf__ AICORE void Tci_b32(typename TileData::TileDType __out__ dst, typename TileDataTmp::TileDType __in__ tmp, T S,
                           unsigned validCol)
{
    using Tdst = typename TileData::DType;
    __ubuf__ Tdst *dstPtr = (__ubuf__ Tdst *)__cce_get_tile_ptr(dst);
    constexpr uint16_t vl_size = REPEAT_BYTE / static_cast<uint16_t>(sizeof(typename TileData::DType)); // vl size in elem
    uint16_t loop_cnt = (validCol + vl_size - 1) / vl_size;  // number of loop_cnt in terms of VLs
    int32_t s = S; // starting value for TCI sequence
    uint32_t remain = (validCol % vl_size == 0)? vl_size : (validCol % vl_size); 
    MaskReg preg;
    if constexpr (descending == 0) {
        __VEC_SCOPE__
        {
            vector_s32 index;
            MaskReg preg_b32_all = pset_b32(PAT_ALL);
            uint16_t i;
            for (i = 0; i < (uint16_t)(loop_cnt-1); ++i) {
                vci(index, s + i * vl_size);
                vsts(index, (__ubuf__ int32_t *)dstPtr, (i * vl_size), NORM_B32, preg_b32_all);
            }
            preg = CreatePredicate<Tdst>(remain);
            vci(index, s + i * vl_size);
            vsts(index, (__ubuf__ int32_t *)dstPtr, (i * vl_size), NORM_B32, preg);    
        }
    } else if constexpr (descending == 1) {
        s = S - vl_size + 1;
        __VEC_SCOPE__
        {
            vector_s32 index;
            MaskReg preg_b32_all = pset_b32(PAT_ALL);
            uint16_t i;
            for (i = 0; i < (uint16_t)(loop_cnt-1); ++i) {
                vci(index, s - i * vl_size, DEC_ORDER);
                vsts(index, (__ubuf__ int32_t *)dstPtr, (i * vl_size), NORM_B32, preg_b32_all);
            }
            preg = CreatePredicate<Tdst>(remain);
            vci(index, s - i * vl_size, DEC_ORDER);
            vsts(index, (__ubuf__ int32_t *)dstPtr, (i * vl_size), NORM_B32, preg);
        }
    }
}

template <typename TileData, typename TileDataTmp, typename T, int descending = 0>
__tf__ AICORE void Tci_b16(typename TileData::TileDType __out__ dst, typename TileDataTmp::TileDType __in__ tmp, T S,
                           unsigned validCol)
{
    using Tdst = typename TileData::DType;
    __ubuf__ Tdst *dstPtr = (__ubuf__ Tdst *)__cce_get_tile_ptr(dst);
    constexpr uint16_t vl_size = REPEAT_BYTE / static_cast<uint16_t>(sizeof(typename TileData::DType));
    uint16_t loop_cnt = (validCol + vl_size - 1) / vl_size;
    int32_t s = S;
    uint32_t remain = (validCol % vl_size == 0)? vl_size : (validCol % vl_size); 
    MaskReg preg;
    if constexpr (descending == 0) {
        __VEC_SCOPE__
        {
            vector_s16 index;
            MaskReg preg_b16_all = pset_b16(PAT_ALL);
            uint16_t i = 0;
            for (i = 0; i < (uint16_t)(loop_cnt-1); ++i) {
                vci(index, s + i*vl_size, INC_ORDER);
                vsts(index, dstPtr, (i * vl_size), NORM_B16, preg_b16_all);
            }
            preg = CreatePredicate<Tdst>(remain);
            vci(index, s+i*vl_size);
            vsts(index, dstPtr, (i * vl_size), NORM_B16, preg);
        }

    } else if constexpr (descending == 1) {
        s = S - vl_size + 1;
        __VEC_SCOPE__
        {
            vector_s16 index;
            MaskReg preg_b16_all = pset_b16(PAT_ALL);
            uint16_t i = 0;
            for (i = 0; i < (uint16_t)(loop_cnt-1); ++i) {
                vci(index, s - i*vl_size, DEC_ORDER);
                vsts(index, dstPtr, (i * vl_size), NORM_B16, preg_b16_all);
            }
            preg = CreatePredicate<Tdst>(remain);
            vci(index, s - i*vl_size, DEC_ORDER);
            vsts(index, dstPtr, (i * vl_size), NORM_B16, preg);
        }
    }
}

template <typename TileData, typename TileDataTmp, typename T, int descending>
PTO_INTERNAL void TCI_IMPL(TileData &dst, T start, TileDataTmp &tmp)
{
    CheckValid<TileData, T>();
    unsigned validCol = dst.GetValidCol();
    if constexpr (std::is_same_v<typename TileData::DType, int32_t> ||
                  std::is_same_v<typename TileData::DType, uint32_t>) {
        Tci_b32<TileData, TileDataTmp, T, descending>(dst.data(), tmp.data(), start, validCol);
    } else {
        Tci_b16<TileData, TileDataTmp, T, descending>(dst.data(), tmp.data(), start, validCol);
    }
}
} // namespace pto
#endif
