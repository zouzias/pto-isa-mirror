/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_NPU_A2A3_TCOLEXPAND_HPP
#define PTO_NPU_A2A3_TCOLEXPAND_HPP

#include <type_traits>

#include <pto/common/constants.hpp>
#include <pto/common/utils.hpp>
#include <pto/common/type.hpp>

namespace pto {
    template <typename T, typename TileDataDst, typename TileDataSrc, unsigned dstStride>
    __tf__ PTO_INTERNAL void TColExpand(typename TileDataDst::TileDType __out__ dst,
                                     typename TileDataSrc::TileDType __in__ src,
                                     int validRow, int validCol)
    {
        __ubuf__ T *dstPtr = (__ubuf__ T *)__cce_get_tile_ptr(dst);
        __ubuf__ T *srcPtr = (__ubuf__ T *)__cce_get_tile_ptr(src);

        int lenBurst = (validCol * sizeof(T) + BLOCK_BYTE_SIZE - 1) / BLOCK_BYTE_SIZE;

        for (int i = 0; i < validRow; i++) {
            copy_ubuf_to_ubuf(dstPtr + i * dstStride, srcPtr, 0, 1, lenBurst, 0, 0);
        }
    }

    template <typename T, typename TileDataOut, typename TileDataIn>
    PTO_INTERNAL void TColExpandCheck(int SrcValidRow, int SrcValidCol, int DstValidCol) {
        static_assert(TileDataOut::Loc == pto::TileType::Vec && TileDataIn::Loc == pto::TileType::Vec,
                      "Fix: TCOLEXPAND only support Vec Tile");
        static_assert(TileDataIn::isRowMajor && TileDataIn::SFractal == SLayout::NoneBox,
                      "Fix: TCOLEXPAND input tile only support Nd fractal Tile");
        static_assert(TileDataOut::isRowMajor && TileDataOut::SFractal == SLayout::NoneBox,
                      "Fix: TCOLEXPAND output tile only support Nd fractal Tile");
        static_assert(std::is_same_v<T, half> || std::is_same_v<T, float> ||
                      std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t> ||
                      std::is_same_v<T, int16_t> || std::is_same_v<T, uint16_t> ||
                      std::is_same_v<T, int32_t> || std::is_same_v<T, uint32_t>,
                      "Fix: TCOLEXPAND input data type is not supported by this instruction.");
        static_assert(std::is_same_v<typename TileDataOut::DType, T>,
                      "Fix: TCOLEXPAND input data type must be consistent with the output data type.");
        PTO_ASSERT(SrcValidCol == DstValidCol,
                   "Fix: TCOLEXPAND input valid col must be consistent with the output valid row.");
    }

    template <typename TileDataOut, typename TileDataIn>
    PTO_INTERNAL void TCOLEXPAND_IMPL(TileDataOut &dst, TileDataIn &src) {
        using T = typename TileDataIn::DType;
        int validRow = dst.GetValidRow();
        int validCol = dst.GetValidCol();
        TColExpandCheck<T, TileDataOut, TileDataIn>(validRow, validCol, dst.GetValidCol());
        if (validRow == 0 || validCol == 0 || src.GetValidRow() == 0 || src.GetValidCol() == 0) {
            return;
        }
        constexpr int dstStride = TileDataOut::RowStride;
        TColExpand<T, TileDataOut, TileDataIn, dstStride>(dst.data(), src.data(), validRow, validCol);
    }

    // ================================================================
    // Column-expand binary ops (dst = op(src0, src1_broadcast))
    //
    // `src1` is a row-vector (shape `1 x validCol`) and is broadcast across rows.
    // ================================================================

    template <typename Op, typename T, unsigned dstRowStride>
    PTO_INTERNAL void TColExpandBinaryCountMode(
        __ubuf__ T *dstPtr, __ubuf__ T *src0Ptr, __ubuf__ T *src1Ptr, unsigned validRow, unsigned validCol)
    {
        set_mask_count();
        SetVectorCount(validCol);
        for (unsigned i = 0; i < validRow; ++i) {
            unsigned offset = i * dstRowStride;
            Op::BinInstr(dstPtr + offset, src0Ptr + offset, src1Ptr, 0);
        }
        set_mask_norm();
        SetFullVecMaskByDType<T>();
    }

    template <typename Op, typename TileDataDst, typename TileDataSrc1, unsigned dstRowStride>
    __tf__ PTO_INTERNAL void TColExpandBinary(
        typename TileDataDst::TileDType __out__ dst,
        typename TileDataDst::TileDType __in__ src0,
        typename TileDataSrc1::TileDType __in__ src1,
        unsigned validRow,
        unsigned validCol)
    {
        using T = typename TileDataDst::DType;
        __ubuf__ T *dstPtr = (__ubuf__ T *)__cce_get_tile_ptr(dst);
        __ubuf__ T *src0Ptr = (__ubuf__ T *)__cce_get_tile_ptr(src0);
        __ubuf__ T *src1Ptr = (__ubuf__ T *)__cce_get_tile_ptr(src1);
        TColExpandBinaryCountMode<Op, T, dstRowStride>(dstPtr, src0Ptr, src1Ptr, validRow, validCol);
    }

    template <typename T>
    struct ColExpandDivOp {
        PTO_INTERNAL static void BinInstr(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1, uint8_t repeats)
        {
            vdiv(dst, src0, src1, repeats, 1, 1, 1, 8, 8, 8);
        }
        PTO_INTERNAL static void BinInstr(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1, uint8_t repeats,
            uint8_t dstRepeatStride, uint8_t src0RepeatStride, uint8_t src1RepeatStride)
        {
            vdiv(dst, src0, src1, repeats, 1, 1, 1, dstRepeatStride, src0RepeatStride, src1RepeatStride);
        }
    };

    template <typename T>
    struct ColExpandMulOp {
        PTO_INTERNAL static void BinInstr(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1, uint8_t repeats)
        {
            vmul(dst, src0, src1, repeats, 1, 1, 1, 8, 8, 8);
        }
        PTO_INTERNAL static void BinInstr(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1, uint8_t repeats,
            uint8_t dstRepeatStride, uint8_t src0RepeatStride, uint8_t src1RepeatStride)
        {
            vmul(dst, src0, src1, repeats, 1, 1, 1, dstRepeatStride, src0RepeatStride, src1RepeatStride);
        }
    };

    template <typename T>
    struct ColExpandSubOp {
        PTO_INTERNAL static void BinInstr(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1, uint8_t repeats)
        {
            vsub(dst, src0, src1, repeats, 1, 1, 1, 8, 8, 8);
        }
        PTO_INTERNAL static void BinInstr(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1, uint8_t repeats,
            uint8_t dstRepeatStride, uint8_t src0RepeatStride, uint8_t src1RepeatStride)
        {
            vsub(dst, src0, src1, repeats, 1, 1, 1, dstRepeatStride, src0RepeatStride, src1RepeatStride);
        }
    };

    template <typename TileDataDst, typename TileDataSrc1>
    PTO_INTERNAL void TCOLEXPANDDIV_IMPL(TileDataDst &dst, TileDataDst &src0, TileDataSrc1 &src1)
    {
        using T = typename TileDataDst::DType;
        static_assert(std::is_same_v<T, half> || std::is_same_v<T, float>, "TCOLEXPANDDIV: Invalid data type.");
        static_assert(TileDataDst::Loc == pto::TileType::Vec, "TCOLEXPANDDIV: dst/src0 must be Vec tiles.");
        static_assert(TileDataSrc1::Loc == pto::TileType::Vec, "TCOLEXPANDDIV: src1 must be a Vec tile.");
        static_assert(TileDataDst::isRowMajor && TileDataSrc1::isRowMajor, "TCOLEXPANDDIV: Only RowMajor is supported.");

        unsigned validRow = dst.GetValidRow();
        unsigned validCol = dst.GetValidCol();
        if (validRow == 0 || validCol == 0) {
            return;
        }

        PTO_ASSERT(src0.GetValidRow() == validRow && src0.GetValidCol() == validCol, "TCOLEXPANDDIV: invalid src0 shape.");
        PTO_ASSERT(src1.GetValidRow() == 1 && src1.GetValidCol() == validCol, "TCOLEXPANDDIV: invalid src1 shape.");

        constexpr unsigned rowStride = TileDataDst::RowStride;
        TColExpandBinary<ColExpandDivOp<T>, TileDataDst, TileDataSrc1, rowStride>(
            dst.data(), src0.data(), src1.data(), validRow, validCol);
    }

    template <typename TileDataDst, typename TileDataSrc1>
    PTO_INTERNAL void TCOLEXPANDMUL_IMPL(TileDataDst &dst, TileDataDst &src0, TileDataSrc1 &src1)
    {
        using T = typename TileDataDst::DType;
        static_assert(std::is_same_v<T, half> || std::is_same_v<T, float>, "TCOLEXPANDMUL: Invalid data type.");
        static_assert(TileDataDst::Loc == pto::TileType::Vec, "TCOLEXPANDMUL: dst/src0 must be Vec tiles.");
        static_assert(TileDataSrc1::Loc == pto::TileType::Vec, "TCOLEXPANDMUL: src1 must be a Vec tile.");
        static_assert(TileDataDst::isRowMajor && TileDataSrc1::isRowMajor, "TCOLEXPANDMUL: Only RowMajor is supported.");

        unsigned validRow = dst.GetValidRow();
        unsigned validCol = dst.GetValidCol();
        if (validRow == 0 || validCol == 0) {
            return;
        }

        PTO_ASSERT(src0.GetValidRow() == validRow && src0.GetValidCol() == validCol, "TCOLEXPANDMUL: invalid src0 shape.");
        PTO_ASSERT(src1.GetValidRow() == 1 && src1.GetValidCol() == validCol, "TCOLEXPANDMUL: invalid src1 shape.");

        constexpr unsigned rowStride = TileDataDst::RowStride;
        TColExpandBinary<ColExpandMulOp<T>, TileDataDst, TileDataSrc1, rowStride>(
            dst.data(), src0.data(), src1.data(), validRow, validCol);
    }

    template <typename TileDataDst, typename TileDataSrc1>
    PTO_INTERNAL void TCOLEXPANDSUB_IMPL(TileDataDst &dst, TileDataDst &src0, TileDataSrc1 &src1)
    {
        using T = typename TileDataDst::DType;
        static_assert(std::is_same_v<T, half> || std::is_same_v<T, float>, "TCOLEXPANDSUB: Invalid data type.");
        static_assert(TileDataDst::Loc == pto::TileType::Vec, "TCOLEXPANDSUB: dst/src0 must be Vec tiles.");
        static_assert(TileDataSrc1::Loc == pto::TileType::Vec, "TCOLEXPANDSUB: src1 must be a Vec tile.");
        static_assert(TileDataDst::isRowMajor && TileDataSrc1::isRowMajor, "TCOLEXPANDSUB: Only RowMajor is supported.");

        unsigned validRow = dst.GetValidRow();
        unsigned validCol = dst.GetValidCol();
        if (validRow == 0 || validCol == 0) {
            return;
        }

        PTO_ASSERT(src0.GetValidRow() == validRow && src0.GetValidCol() == validCol, "TCOLEXPANDSUB: invalid src0 shape.");
        PTO_ASSERT(src1.GetValidRow() == 1 && src1.GetValidCol() == validCol, "TCOLEXPANDSUB: invalid src1 shape.");

        constexpr unsigned rowStride = TileDataDst::RowStride;
        TColExpandBinary<ColExpandSubOp<T>, TileDataDst, TileDataSrc1, rowStride>(
            dst.data(), src0.data(), src1.data(), validRow, validCol);
    }
}
#endif
