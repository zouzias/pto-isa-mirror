/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TEXPANDS_HPP
#define TEXPANDS_HPP

#include <pto/common/constants.hpp>
#include <pto/common/utils.hpp>
#include "common.hpp"
#include "utils.hpp"
#include "TBinSOp.hpp"

namespace pto {

template <typename T>
struct ExpandSOp {
    PTO_INTERNAL static void BinSInstr(RegTensor<T> &reg_dst, RegTensor<T> &reg_src0, T scalar, MaskReg &preg)
    {
        vdup(reg_dst, scalar, preg, MODE_ZEROING);
    }
};

template <typename TileDataDst, unsigned elementsPerRepeat, unsigned blockSizeElem, unsigned rowStride>
__tf__ PTO_INTERNAL void TExpandS(typename TileDataDst::TileDType __out__ dst, typename TileDataDst::DType scalar,
                                  unsigned kValidRows, unsigned kValidCols,
                                  VFImplKind version = VFImplKind::VFIMPL_DEFAULT)
{
    using T = typename TileDataDst::DType;
    __ubuf__ T *dstPtr = (__ubuf__ T *)__cce_get_tile_ptr(dst);
    BinaryInstr<ExpandSOp<T>, TileDataDst, TileDataDst, T, elementsPerRepeat, blockSizeElem, rowStride, rowStride>(
        dstPtr, nullptr, scalar, kValidRows, kValidCols, version);
}

template <typename TileData>
PTO_INTERNAL void TSetValueInstrL1(__cbuf__ typename TileData::DType *dstPtr, int64_t repeatConfig,
                                   typename TileData::DType value)
{
    if constexpr (sizeof(typename TileData::DType) == 1) {
        auto dstCast = reinterpret_cast<__cbuf__ uint32_t *>(dstPtr);
        uint16_t expanded16 = static_cast<uint16_t>(value) | (static_cast<uint16_t>(value) << 8);
        uint32_t expanded32 = static_cast<uint32_t>(expanded16) | (static_cast<uint32_t>(expanded16) << 16);
        create_cbuf_matrix(dstCast, repeatConfig, expanded32);
    } else if constexpr (sizeof(typename TileData::DType) == 2) {
        if constexpr (std::is_same<typename TileData::DType, bfloat16_t>::value) {
            create_cbuf_matrix_bf16(reinterpret_cast<__cbuf__ bfloat16_t *>(dstPtr), repeatConfig, value);
        } else {
            create_cbuf_matrix(dstPtr, repeatConfig, value);
        }
    } else if constexpr (sizeof(typename TileData::DType) == 4) {
        if constexpr (std::is_same_v<typename TileData::DType, float>) {
            uint32_t bits;
            __builtin_memcpy(&bits, &value, sizeof(float));
            create_cbuf_matrix(dstPtr, repeatConfig, bits);
        } else {
            create_cbuf_matrix(dstPtr, repeatConfig, static_cast<uint32_t>(value));
        }
    }
}

template <typename TileData>
__tf__ PTO_INTERNAL void TSetValueTile(typename TileData::TileDType __out__ dst, typename TileData::DType value)
{
    using U = typename TileData::DType;
    __cbuf__ U *dstPtr = (__cbuf__ U *)__cce_get_tile_ptr(dst);
    uint16_t dstRow = TileData::Rows;
    uint16_t dstCol = TileData::Cols;
    // block uint is 32B
    uint16_t repeatTimes = GetByteSize<typename TileData::DType>(dstRow * dstCol) / 32;
    int64_t repeatConfig = 0;
    // [46:32] is the repeat gap between two consecutive repeats
    repeatConfig |= (static_cast<int64_t>(0) & 0x7FFF) << 32;
    repeatConfig |= (static_cast<int64_t>(1) & 0xFFFF) << 16;     // [30:16] is the block number of each repeat
    repeatConfig |= (static_cast<int64_t>(repeatTimes) & 0xFFFF); // [14:0] is the repeat times
    // create_cbuf_matrix(dstPtr, repeatConfig, static_cast<uint16_t>(value));
    TSetValueInstrL1<TileData>(dstPtr, repeatConfig, value);
}

template <typename TileData>
__tf__ PTO_INTERNAL void TSetValue5HD(typename TileData::TileDType __out__ dst, typename TileData::DType value,
                                      int shape0, int shape1, int shape2, int shape3)
{
    using U = typename TileData::DType;
    __cbuf__ U *dstPtr = (__cbuf__ U *)__cce_get_tile_ptr(dst);

    constexpr uint32_t c0ElemCount = C0_SIZE_BYTE / sizeof(typename TileData::DType);
    // 32B block
    uint16_t repeatTimes = shape0 * shape1 * shape2 * shape3;
    int64_t repeatConfig = 0;
    // [46:32] is the repeat gap between two consecutive repeats
    repeatConfig |= (static_cast<int64_t>(0) & 0x7FFF) << 32;
    repeatConfig |= (static_cast<int64_t>(1) & 0xFFFF) << 16;     // [30:16] is the block number of each repeat
    repeatConfig |= (static_cast<int64_t>(repeatTimes) & 0xFFFF); // [14:0] is the repeat times
    TSetValueInstrL1<TileData>(dstPtr, repeatConfig, value);
}

template <typename TileDataDst>
PTO_INTERNAL void TEXPANDS_IMPL(TileDataDst &dst, typename TileDataDst::DType scalar)
{
    using T = typename TileDataDst::DType;
    static_assert(
        std::is_same<T, int32_t>::value || std::is_same<T, uint32_t>::value || std::is_same<T, int>::value ||
            std::is_same<T, int16_t>::value || std::is_same<T, uint16_t>::value || std::is_same<T, int8_t>::value ||
            std::is_same<T, uint8_t>::value || std::is_same<T, half>::value || std::is_same<T, float16_t>::value ||
            std::is_same<T, float>::value || std::is_same<T, float32_t>::value || std::is_same<T, bfloat16_t>::value,
        "TEXPANDS: Invalid data type");
    static_assert(TileDataDst::Loc == TileType::Vec || TileDataDst::Loc == TileType::Mat,
                  "Location of src and dst tiles must be Location::Vec or Mat.");

    if constexpr (TileDataDst::Loc == TileType::Vec) {
        static_assert(TileDataDst::ValidCol <= TileDataDst::Cols,
                      "Number of valid columns must not be greater than number of tile columns.");
        static_assert(TileDataDst::ValidRow <= TileDataDst::Rows,
                      "Number of valid rows must not be greater than number of tile rows.");

        constexpr unsigned blockSizeElem = BLOCK_BYTE_SIZE / sizeof(T);
        constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(T);
        constexpr unsigned rowStride = TileDataDst::RowStride;
        unsigned validRow = dst.GetValidRow();
        unsigned validCol = dst.GetValidCol();

        TExpandS<TileDataDst, elementsPerRepeat, blockSizeElem, rowStride>(dst.data(), scalar, validRow, validCol);
    } else if constexpr (TileDataDst::Loc == TileType::Mat) {
        if constexpr (is_conv_tile_v<TileDataDst>) { // layout is NC1HWC0, dst dim4 is c0Size
            TSetValue5HD<TileDataDst>(dst.data(), scalar, dst.GetShape(0), dst.GetShape(1), dst.GetShape(2),
                                      dst.GetShape(3));
        } else {
            TSetValueTile<TileDataDst>(dst.data(), scalar);
        }
    }
}
} // namespace pto
#endif
