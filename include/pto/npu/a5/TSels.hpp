/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TSELS_HPP
#define TSELS_HPP

#include <pto/common/constants.hpp>
#include <pto/common/utils.hpp>
#include "utils.hpp"

namespace pto {
template <typename T, unsigned elementsPerRepeat, uint16_t unRollConstant, unsigned dstStride, unsigned maskStride, unsigned srcStride>
PTO_INTERNAL void TSelsHead(__ubuf__ T *dstPtr, __ubuf__ uint32_t *maskPtr, __ubuf__ T *srcPtr, T scalar,
    uint16_t pairedRepeatTimes, unsigned validRow, unsigned validCol) {
    MaskReg preg, selMask0, selMask1, tmpMask0;
    MaskReg tmpMask1 = pset_b16(PAT_ALL);
    RegTensor<T> vreg0, vreg1, vreg2, vreg3, dreg0, dreg1, dreg2;

    constexpr auto distValue =
        std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<T, DistVST::DIST_NORM>())>();

    for (uint16_t i = 0; i < (uint16_t)(validRow); ++i) {
        for (uint16_t j = 0; j < (uint16_t)(pairedRepeatTimes); ++j) {
            uint16_t repeatIdx = j * unRollConstant;
            uint32_t colOffset0 = repeatIdx * elementsPerRepeat;
            uint32_t colOffset1 = colOffset0 + elementsPerRepeat;
            uint32_t count0 =
                ((colOffset0 + elementsPerRepeat) >= validCol ? validCol - colOffset0 : elementsPerRepeat);
            preg = CreatePredicate<T>(count0);

            vlds(vreg0, srcPtr, (int32_t)(i * srcStride + colOffset0), NORM);
            vdup(vreg1, scalar, preg, MODE_ZEROING);
            plds(tmpMask0, maskPtr, i * maskStride + repeatIdx * 8, US);
            pintlv_b16(selMask0, selMask1, tmpMask0, tmpMask1);

            vsel(dreg0, vreg0, vreg1, selMask0);

            vsts(dreg0, dstPtr, (int32_t)(i * dstStride + colOffset0), distValue, preg);

            vlds(vreg2, srcPtr, (int32_t)(i * srcStride + colOffset1), NORM);
            vsel(dreg1, vreg2, vreg1, selMask1);
            uint32_t count1 =
                ((colOffset1 + elementsPerRepeat) >= validCol ? validCol - colOffset1 : elementsPerRepeat);
            preg = CreatePredicate<T>(count1);
            vsts(dreg1, dstPtr, (int32_t)(i * dstStride + colOffset1), distValue, preg);
        }
    }
}

template <typename T, unsigned elementsPerRepeat, unsigned dstStride, unsigned maskStride, unsigned srcStride>
PTO_INTERNAL void TSelsTail(__ubuf__ T *dstPtr, __ubuf__ uint32_t *maskPtr, __ubuf__ T *srcPtr, T scalar,
    uint16_t repeatIdx, uint16_t remainRepeat, unsigned validRow, unsigned validCol) {
    MaskReg preg, selMask2;
    MaskReg tmpMask1 = pset_b16(PAT_ALL);
    RegTensor<T> vreg4, vreg5, dreg2;
    constexpr auto distValue =
        std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<T, DistVST::DIST_NORM>())>();
    for (uint16_t i = 0; i < (uint16_t)(validRow); ++i) {
        for (uint16_t j = 0; j < (uint16_t)(remainRepeat); ++j) {
            uint32_t colOffset = (repeatIdx + j) * elementsPerRepeat;
            uint32_t count = (validCol > colOffset) ? (validCol - colOffset) : 0;
            preg = CreatePredicate<T>(count);

            plds(selMask2, maskPtr, i * maskStride + (repeatIdx + j) * 8, US);
            punpack(selMask2, selMask2, LOWER);

            vlds(vreg4, srcPtr, (int32_t)(i * srcStride + colOffset), NORM);
            vdup(vreg5, scalar, preg, MODE_ZEROING);
            vsel(dreg2, vreg4, vreg5, selMask2);
            vsts(dreg2, dstPtr, (int32_t)(i * dstStride + colOffset), distValue, preg);
        }
    }
}

template <typename TileDataDst, typename TileDataMask, typename TileDataSrc, unsigned elementsPerRepeat>
__tf__ PTO_INTERNAL void TSels_b32(typename TileDataDst::TileDType __out__ dst, typename TileDataMask::TileDType __in__ mask,
    typename TileDataSrc::TileDType __in__ src, typename TileDataSrc::DType __in__ scalar, unsigned validRow,
    unsigned validCol) {
    using T = typename TileDataSrc::DType;
    __ubuf__ T *dstPtr = (__ubuf__ T *)__cce_get_tile_ptr(dst);
    __ubuf__ uint32_t *maskPtr = (__ubuf__ uint32_t *)__cce_get_tile_ptr(mask);
    __ubuf__ T *srcPtr = (__ubuf__ T *)__cce_get_tile_ptr(src);

    uint16_t repeatTimes = CeilDivision(validCol, elementsPerRepeat);
    constexpr uint32_t unRollConstant = 2;
    uint16_t pairedRepeatTimes = repeatTimes / unRollConstant;
    uint16_t remainRepeat = repeatTimes % unRollConstant;
    uint16_t repeatIdx = pairedRepeatTimes * unRollConstant;

    __VEC_SCOPE__ {
        TSelsHead<T, elementsPerRepeat, unRollConstant, TileDataDst::RowStride, TileDataMask::RowStride, TileDataSrc::RowStride>(
            dstPtr, maskPtr, srcPtr, scalar, pairedRepeatTimes, validRow, validCol);
        TSelsTail<T, elementsPerRepeat, TileDataDst::RowStride, TileDataMask::RowStride, TileDataSrc::RowStride>(
            dstPtr, maskPtr, srcPtr, scalar, repeatIdx, remainRepeat, validRow, validCol);
    } // end of vf
}

template <typename TileDataDst, typename TileDataMask, typename TileDataSrc, unsigned elementsPerRepeat>
__tf__ PTO_INTERNAL void TSels_b16_8(typename TileDataDst::TileDType __out__ dst, typename TileDataMask::TileDType __in__ mask,
    typename TileDataSrc::TileDType __in__ src, typename TileDataSrc::DType __in__ scalar, unsigned validRow,
    unsigned validCol) {
    using T = typename TileDataSrc::DType;
    __ubuf__ T *dstPtr = (__ubuf__ T *)__cce_get_tile_ptr(dst);
    __ubuf__ typename TileDataMask::DType *maskPtr = (__ubuf__ typename TileDataMask::DType *)__cce_get_tile_ptr(mask);
    __ubuf__ T *srcPtr = (__ubuf__ T *)__cce_get_tile_ptr(src);
    uint16_t repeatTimes = CeilDivision(validCol, elementsPerRepeat);
    __VEC_SCOPE__ {
        MaskReg preg, maskreg;
        RegTensor<T> vreg0, vreg1, vreg2;
        constexpr auto distValue =
            std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<T, DistVST::DIST_NORM>())>();
        for (uint16_t i = 0; i < (uint16_t)validRow; ++i) {
            for (uint16_t j = 0; j < (uint16_t)repeatTimes; ++j) {
                uint32_t count = ((j + 1) * elementsPerRepeat >= validCol ? validCol - j * elementsPerRepeat : elementsPerRepeat);
                vlds(vreg0, srcPtr, i * TileDataSrc::RowStride + j * elementsPerRepeat, NORM);
                preg = CreatePredicate<T>(count);
                vdup(vreg1, scalar, preg, MODE_ZEROING);
                if (sizeof(T) == 2) {
                    plds(maskreg, (__ubuf__ uint32_t *)maskPtr, i * TileDataMask::RowStride + j * 16, US);
                } else {
                    plds(maskreg, (__ubuf__ uint32_t *)maskPtr, i * TileDataMask::RowStride + j * 16, NORM);
                }
                vsel(vreg2, vreg0, vreg1, maskreg);
                count =
                    ((j + 1) * elementsPerRepeat >= validCol ? validCol - j * elementsPerRepeat : elementsPerRepeat);
                preg = CreatePredicate<T>(count);
                vsts(vreg2, dstPtr, i * TileDataDst::RowStride + j * elementsPerRepeat, distValue, preg);
            }
        }
    } // end of vf
}

template <typename TileDataDst, typename TileDataMask, typename TileDataSrc>
PTO_INTERNAL void TSELS_IMPL(TileDataDst &dst, TileDataMask &mask, TileDataSrc &src, typename TileDataSrc::DType scalar)
{
    using T = typename TileDataDst::DType;
    static_assert(std::is_same_v<typename TileDataSrc::DType, typename TileDataDst::DType>, "TileType of dst and src must be the same.");
    static_assert(std::is_same<T, int8_t>::value || std::is_same<T, int16_t>::value ||
                  std::is_same<T, int32_t>::value || std::is_same<T, half>::value ||
                  std::is_same<T, float32_t>::value || std::is_same<T, uint8_t>::value ||
                  std::is_same<T, uint16_t>::value || std::is_same<T, uint32_t>::value,
                  "TSELS: Invalid data type");
    static_assert(TileDataDst::isRowMajor && TileDataMask::isRowMajor && TileDataSrc::isRowMajor,
        "TSELS: not supported Layout type");
    static_assert(sizeof(T) == 4 || sizeof(T) == 2 || sizeof(T) == 1, "TSELS: Invalid data type.");

    constexpr unsigned blockSizeElem = BLOCK_BYTE_SIZE / sizeof(T);
    constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(T);
    unsigned validRow = dst.GetValidRow();
    unsigned validCol = dst.GetValidCol();

    PTO_ASSERT(src.GetValidCol() == dst.GetValidCol(), "Number of columns of src1 and dst must be the same.");
    PTO_ASSERT(src.GetValidRow() == dst.GetValidRow(), "Number of rows of src1 and dst must be the same.");

    if (sizeof(T) == 4) {
        TSels_b32<TileDataDst, TileDataMask, TileDataSrc, elementsPerRepeat>(
            dst.data(), mask.data(), src.data(), scalar, validRow, validCol);
    } else {
        TSels_b16_8<TileDataDst, TileDataMask, TileDataSrc, elementsPerRepeat>(
            dst.data(), mask.data(), src.data(), scalar, validRow, validCol);
    }
}
}  // namespace pto
#endif
