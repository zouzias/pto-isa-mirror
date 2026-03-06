/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TSETVALUE_HPP
#define TSETVALUE_HPP

namespace pto {
template <typename T>
PTO_INTERNAL int GetByteSize(const int value)
{
    return sizeof(T) * value;
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
            bfloat16_t bits;
            __builtin_memcpy(&bits, &value, sizeof(bfloat16_t));
            create_cbuf_matrix_bf16(reinterpret_cast<__cbuf__ bfloat16_t *>(dstPtr), repeatConfig, bits);
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
    // 多少个32B
    uint16_t repeatTimes = GetByteSize<typename TileData::DType>(dstRow * dstCol) / 32;
    int64_t repeatConfig = 0;
    // [46:32] is the repeat gap between two consecutive repeats
    repeatConfig |= (static_cast<int64_t>(0) & 0x7FFF) << 32;
    repeatConfig |= (static_cast<int64_t>(1) & 0xFFFF) << 16;     // [30:16] is the block number of each repeat
    repeatConfig |= (static_cast<int64_t>(repeatTimes) & 0xFFFF); // [14:0] is the repeat times
    // value类型仅仅支持uint32和half
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
    uint16_t repeatTimes = GetByteSize<typename TileData::DType>(shape0 * shape1 * shape2 * shape3) / 32;
    int64_t repeatConfig = 0;
    // [46:32] is the repeat gap between two consecutive repeats
    repeatConfig |= (static_cast<int64_t>(0) & 0x7FFF) << 32;
    repeatConfig |= (static_cast<int64_t>(1) & 0xFFFF) << 16;     // [30:16] is the block number of each repeat
    repeatConfig |= (static_cast<int64_t>(repeatTimes) & 0xFFFF); // [14:0] is the repeat times
    TSetValueInstrL1<TileData>(dstPtr, repeatConfig, value);
}

template <typename TileData>
PTO_INTERNAL void TSET_VALUE_IMPL(TileData &dst, typename TileData::DType value)
{
    static_assert(TileData::Loc == pto::TileType::Mat, "Only support Mat!");

    if constexpr (is_conv_tile_v<TileData>) { // layout is NC1HWC0, dst dim4 is c0Size
        TSetValue5HD<TileData>(dst.data(), value, dst.GetShape(0), dst.GetShape(1), dst.GetShape(2), dst.GetShape(3));

    } else {
        TSetValueTile<TileData>(dst.data(), value);
    }
}
} // namespace pto

#endif