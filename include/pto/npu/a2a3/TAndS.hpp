/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TANDS_HPP
#define TANDS_HPP

#include <pto/common/constants.hpp>
#include "pto/npu/a2a3/TBinSOp.hpp"
#include "pto/npu/a2a3/TBinSPlusOp.hpp"

namespace pto
{
    template <typename TileData>
    PTO_INTERNAL void TANDS_IMPL(TileData &dst, TileData &src0, typename TileData::DType scalar)
    {
        static_assert(std::is_same<typename TileData::DType, int32_t>::value ||
                      std::is_same<typename TileData::DType, int>::value ||
                      std::is_same<typename TileData::DType, int16_t>::value ||
                      std::is_same<typename TileData::DType, uint32_t>::value ||
                      std::is_same<typename TileData::DType, unsigned int>::value ||
                      std::is_same<typename TileData::DType, uint16_t>::value ||
                      "TANDS: Invalid data type");

        static_assert(TileData::isRowMajor, "TADDS: not supported Layout type.");
        static_assert(TileData::Loc == TileType::Vec, "TileType of src and dst tiles must be TileType::Vec.");
        static_assert(TileData::ValidCol <= TileData::Cols, "Number of valid columns must not be greater than number of tile columns.");
        static_assert(TileData::ValidRow <= TileData::Rows, "Number of valid rows must not be greater than number of tile rows.");
        
        PTO_ASSERT(src0.GetValidCol() == dst.GetValidCol(), "Number of columns of src and dst must be the same.");
        PTO_ASSERT(src0.GetValidRow() == dst.GetValidRow(), "Number of rows of src and dst must be the same.");

        constexpr unsigned blockSizeElem = BLOCK_BYTE_SIZE / sizeof(typename TileData::DType);
        constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(typename TileData::DType);
        unsigned numRepeatPerLine = dst.GetValidCol() / elementsPerRepeat;
        unsigned numRemainPerLine = dst.GetValidCol() % elementsPerRepeat;
        constexpr unsigned stride = TileData::RowStride;
        unsigned validRow = dst.GetValidRow();
        unsigned validCol = dst.GetValidCol();
        TEXPANDS_IMPL<TileData>(dst, scalar);
        TAND_IMPL<TileData>(dst, src0, dst);
    }

    template <typename TileDataDst, typename TileDataSrc>
    PTO_INTERNAL void TANDS_IMPL(TileDataDst &dst, TileDataSrc &src, typename TileDataSrc::DType scalar)
    {
        using T = typename TileDataSrc::DType;
        static_assert(std::is_same_v<T, typename TileDataDst::DType>,
            "TADDS: The data type of dst must be consistent with src.");
        static_assert(std::is_same<T, int32_t>::value || std::is_same<T, int>::value ||
                      std::is_same<T, int16_t>::value || std::is_same<T, half>::value ||
                      std::is_same<T, float16_t>::value || std::is_same<T, float>::value ||
                      std::is_same<T, float32_t>::value, "TADDS: Invalid data type");

        static_assert(TileDataSrc::Loc == TileType::Vec, "TileType of src and dst tiles must be TileType::Vec.");

        PTO_ASSERT(src.GetValidCol() == dst.GetValidCol(), "Number of cols of src and dst must be the same.");
        PTO_ASSERT(src.GetValidRow() == dst.GetValidRow(), "Number of rows of src and dst must be the same.");

        unsigned dstValidRow = dst.GetValidRow();
        unsigned dstValidCol = dst.GetValidCol();
        if ((dstValidRow != 0 && dstValidCol != 0) &&
            (dstValidRow == src.GetValidRow() && dstValidCol == src.GetValidCol())) {
            TEXPANDS_IMPL<TileDataDst>(dst, scalar);
            TAND_IMPL<T, TileDataDst, TileDataSrc>(dst, src, dst);
        } else {
            PTO_ASSERT(false, "TADDS: dstTile validRow/validCol must be consistent with of src.");
        }
    }
}

#endif