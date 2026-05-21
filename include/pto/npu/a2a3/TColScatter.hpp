/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TCOLSCATTER_HPP
#define TCOLSCATTER_HPP

#include <pto/common/constants.hpp>

namespace pto {
template <typename DstTile, typename T>
PTO_INTERNAL void InitDstUBBuffer(__ubuf__ T *dstPtr)
{
    using U = std::conditional_t<sizeof(T) == sizeof(uint32_t), uint32_t, uint16_t>;
    __ubuf__ U *dst = (__ubuf__ U *)dstPtr;

    // tile must be 32B align, the result is no remainder
    constexpr uint32_t numel = DstTile::Numel * sizeof(T) / sizeof(U);
    set_mask_count();
    set_vector_mask(0, numel);
    vector_dup(dst, (U)0, 0, 1, 1, 8, 8);
    set_mask_norm();
    set_vector_mask(-1, -1);
    PtoSetWaitFlag<PIPE_V, PIPE_S>();
}

template <MaskPattern mask>
PTO_INTERNAL constexpr int GetRowTimesByMask()
{
    switch (mask) {
        case MaskPattern::P1111:
            return PTO_TSCATTER_TIME_1;
        case MaskPattern::P1010:
        case MaskPattern::P0101:
            return PTO_TSCATTER_TIME_2;
        default:
            return PTO_TSCATTER_TIME_4;
    }
}

template <MaskPattern mask, int RowStride>
PTO_INTERNAL int GetStrideByMask(int i)
{
    switch (mask) {
        case MaskPattern::P0101:
            return PTO_TSCATTER_TIME_2 * i * RowStride;
        case MaskPattern::P1010:
            return (PTO_TSCATTER_TIME_2 * i + PTO_TSCATTER_IDX_1) * RowStride;
        case MaskPattern::P0001:
            return PTO_TSCATTER_TIME_4 * i * RowStride;
        case MaskPattern::P0010:
            return (PTO_TSCATTER_TIME_4 * i + PTO_TSCATTER_IDX_1) * RowStride;
        case MaskPattern::P0100:
            return (PTO_TSCATTER_TIME_4 * i + PTO_TSCATTER_IDX_2) * RowStride;
        case MaskPattern::P1000:
            return (PTO_TSCATTER_TIME_4 * i + PTO_TSCATTER_IDX_3) * RowStride;
        default:
            return i * RowStride;
    }
}

template <MaskPattern mask, typename DstTile, typename SrcTile>
__tf__ PTO_INTERNAL void TColScatterMaskImpl(typename DstTile::TileDType __out__ dst,
                                             typename SrcTile::TileDType __in__ src, unsigned validRow,
                                             unsigned validCol)
{
    using T = typename DstTile::DType;
    using copyType = std::conditional_t<sizeof(T) == sizeof(int32_t), __ubuf__ int32_t *, __ubuf__ int16_t *>;
    __ubuf__ T *dstPtr = (__ubuf__ T *)__cce_get_tile_ptr(dst);
    __ubuf__ T *srcPtr = (__ubuf__ T *)__cce_get_tile_ptr(src);

    constexpr unsigned dstStride = DstTile::RowStride;
    constexpr unsigned srcStride = SrcTile::RowStride;

    // Initialize dst UB buffer
    InitDstUBBuffer<DstTile>(dstPtr);

    uint16_t stride = 0;
    set_mask_count();
    set_vector_mask(0, validCol);
    for (int i = 0; i < validRow; i++) {
        stride = GetStrideByMask<mask, DstTile::RowStride>(i);
        vcopy((copyType)(dstPtr + stride), (copyType)(srcPtr + i * srcStride), 1, 1, 1, 0, 0);
    }
    set_mask_norm();
    set_vector_mask(-1, -1);
    PtoSetWaitFlag<PIPE_S, PIPE_V>();
}

template <MaskPattern mask, typename TileDst, typename TileSrc>
PTO_INTERNAL void TCOLSCATTER_IMPL(TileDst &dst, TileSrc &src)
{
    unsigned validRow = src.GetValidRow();
    unsigned validCol = src.GetValidCol();
    if constexpr (mask == MaskPattern::P1111) {
        PTO_ASSERT(validRow == dst.GetValidRow(), "TCOLSCATTER: validRow of src must match dst.");
        PTO_ASSERT(validCol == dst.GetValidCol(), "TCOLSCATTER: validRow of src must match dst.");
        return TMOV_IMPL(dst, src);
    } else {
        using T = typename TileDst::DType;
        static_assert(std::is_same<T, int32_t>::value || std::is_same<T, int16_t>::value ||
                          std::is_same<T, int8_t>::value || std::is_same<T, uint32_t>::value ||
                          std::is_same<T, uint16_t>::value || std::is_same<T, uint8_t>::value ||
                          std::is_same<T, half>::value || std::is_same<T, float16_t>::value ||
                          std::is_same<T, float32_t>::value || std::is_same<T, bfloat16_t>::value,
                      "TCOLSCATTER: Invalid data type.");
        static_assert(std::is_same_v<T, typename TileSrc::DType>,
                      "TCOLSCATTER: Data type of dst and src must be the same.");
        static_assert(TileDst::Loc == TileType::Vec && TileSrc::Loc == TileType::Vec,
                      "TCOLSCATTER: TileType of src and dst tiles must be TileType::Vec.");
        static_assert(TileDst::ValidCol <= TileDst::Cols && TileSrc::ValidCol <= TileSrc::Cols,
                      "TCOLSCATTER: Number of valid columns must not be greater than number of tile columns.");
        static_assert(TileDst::ValidRow <= TileDst::Rows && TileSrc::ValidRow <= TileSrc::Rows,
                      "TCOLSCATTER: Number of valid rows must not be greater than number of tile rows.");
        static_assert(mask >= MaskPattern::P0101 && mask <= MaskPattern::P1111,
                      "TCOLSCATTER: MaskPattern parameter value out of range: must be P0101...P1111 inclusive.");
        PTO_ASSERT(dst.GetValidRow() == validRow * GetRowTimesByMask<mask>(),
                   "TCOLSCATTER: validRow of src must match dst.");
        TColScatterMaskImpl<mask, TileDst, TileSrc>(dst.data(), src.data(), validRow, validCol);
    }
}
} // namespace pto

#endif