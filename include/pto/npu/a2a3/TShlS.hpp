/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TADDS_HPP
#define TADDS_HPP

#include <pto/common/constants.hpp>
#include "pto/npu/a2a3/TBinSOp.hpp"
#include "pto/npu/a2a3/TBinSPlusOp.hpp"

namespace pto
{
    template<typename T>
    struct ShlSOp {
        PTO_INTERNAL static void BinSInstr(__ubuf__ T* dst, __ubuf__ T* src0, T src1, uint8_t repeats) {
            vshl(dst, src0, src1, repeats, 1, 1, 8, 8);
        }
        PTO_INTERNAL static void BinSInstr(__ubuf__ T* dst, __ubuf__ T* src0, T src1, uint8_t repeats, uint8_t dstRepeatStride, uint8_t srcRepeatStride) {
            vshl(dst, src0, src1, repeats, 1, 1, dstRepeatStride, srcRepeatStride);
        }
    };

    template <typename TileData, unsigned elementsPerRepeat, unsigned blockSizeElem, unsigned stride>
    __tf__ PTO_INTERNAL void TShlS(typename TileData::TileDType __out__ dst,
                                typename TileData::TileDType __in__ src0,
                                typename TileData::DType __in__ src1,
                                unsigned validRow,
                                unsigned validCol) 
    {
        using T = typename TileData::DType;
        __ubuf__ T *dstPtr = (__ubuf__ T *)__cce_get_tile_ptr(dst);
        __ubuf__ T *src0Ptr = (__ubuf__ T *)__cce_get_tile_ptr(src0);

        TBinSInstr<ShlSOp<typename TileData::DType>, TileData, elementsPerRepeat, blockSizeElem, stride>(
                dstPtr, src0Ptr, src1, validRow, validCol);
    }

    template <typename TileDataDst, typename TileDataSrc>
    PTO_INTERNAL void TSHLS_IMPL(TileDataDst &dst, TileDataSrc &src, typename TileDataSrc::DType scalar)
    {
        using T = typename TileDataSrc::DType;
        static_assert(std::is_same_v<T, typename TileDataDst::DType>,
            "TADDS: The data type of dst must be consistent with src.");
        static_assert(std::is_same<T, int32_t>::value || std::is_same<T, int>::value ||
                      std::is_same<T, int16_t>::value || std::is_same<T, uint32_t>::value ||
                      std::is_same<T, uint16_t>::value || std::is_same<T, unsigned int>::value,
                      "TADDS: Invalid data type");

        static_assert(TileDataSrc::Loc == TileType::Vec, "TileType of src and dst tiles must be TileType::Vec.");

        PTO_ASSERT(src.GetValidCol() == dst.GetValidCol(), "Number of cols of src and dst must be the same.");
        PTO_ASSERT(src.GetValidRow() == dst.GetValidRow(), "Number of rows of src and dst must be the same.");

        unsigned dstValidRow = dst.GetValidRow();
        unsigned dstValidCol = dst.GetValidCol();
        if ((dstValidRow != 0 && dstValidCol != 0) &&
            (dstValidRow == src.GetValidRow() && dstValidCol == src.GetValidCol())) {
            TShlS<T, TileDataDst, TileDataSrc>(dst.data(), src.data(), scalar, dstValidRow, dstValidCol);
        } else {
            PTO_ASSERT(false, "TADDS: dstTile validRow/validCol must be consistent with of src.");
        }
    }
}

#endif