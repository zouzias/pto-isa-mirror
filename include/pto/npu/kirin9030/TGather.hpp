/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TGATHER_HPP
#define TGATHER_HPP

#include <pto/common/constants.hpp>
#include "common.hpp"

namespace pto {
PTO_INTERNAL int CEIL(int a, int b)
{
    return (a + (b - 1)) / (b);
}

template <typename DstTileData, typename Src0TileData, typename Src1TileData>
PTO_INTERNAL void CheckValid()
{
    using T = typename DstTileData::DType;
    using U = typename Src0TileData::DType;
    using V = typename Src1TileData::DType;
    static_assert(std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t> || std::is_same_v<T, int16_t> ||
                      std::is_same_v<T, uint16_t> || std::is_same_v<T, int32_t> || std::is_same_v<T, uint32_t> ||
                      std::is_same_v<T, half> || std::is_same_v<T, float>,
                  "Fix: TGATHER Dst data type must be int8_t/uint8_t/int16_t/uint16_t/int32_t/uint32_t/half/float.");
    static_assert(std::is_same_v<V, int16_t> || std::is_same_v<V, uint16_t> || std::is_same_v<V, int32_t> ||
                      std::is_same_v<V, uint32_t> || std::is_same_v<V, half> || std::is_same_v<V, float>,
                  "Fix: TGATHER Src1 data type must be int16_t/uint16_t/int32_t/uint32_t/half/float.");
    static_assert(std::is_same_v<T, U>, "Fix: TGATHER expect same size for indice and dst");
}

template <typename TileDataD, typename TileDataS0, typename TileDataS1>
__tf__ AICORE void TGather_b32(typename TileDataD::TileDType __out__ dst, typename TileDataS0::TileDType __in__ src0,
                               typename TileDataS1::TileDType __in__ src1, unsigned validCol, unsigned validRow)
{
    __ubuf__ typename TileDataD::DType *dstPtr = (__ubuf__ typename TileDataD::DType *)__cce_get_tile_ptr(dst);
    __ubuf__ typename TileDataS0::DType *src0Ptr = (__ubuf__ typename TileDataS0::DType *)__cce_get_tile_ptr(src0);
    __ubuf__ typename TileDataS1::DType *src1Ptr = (__ubuf__ typename TileDataS1::DType *)__cce_get_tile_ptr(src1);
    unsigned TShape1 = TileDataD::Cols;
    __VEC_SCOPE__
    {
        uint16_t batchSize = CCE_VL / static_cast<uint16_t>(sizeof(typename TileDataS1::DType));
        uint16_t innerLoopNum = CEIL(validCol, batchSize);
        for (uint16_t i = 0; i < (uint16_t)validRow; ++i) {
            for (uint16_t j = 0; j < innerLoopNum; ++j) {
                RegTensor<typename TileDataS1::DType> index;
                vlds(index, src1Ptr, (i * TileDataS1::Cols + j * batchSize), NORM);

                uint32_t count = ((j + 1) * batchSize >= validCol ? validCol - j * batchSize : batchSize);
                vector_bool preg = CreatePredicate<typename TileDataS1::DType>(count);

                RegTensor<typename TileDataD::DType> v_output;
                vgather2(v_output, src0Ptr, (vector_u32 &)index, preg);
                vsts(v_output, dstPtr, (i * TShape1 + j * batchSize), NORM_B32, preg);
            }
        }
    }
}

template <typename TileDataD, typename TileDataS0, typename TileDataS1>
__tf__ AICORE void TGather_b16(typename TileDataD::TileDType __out__ dst, typename TileDataS0::TileDType __in__ src0,
                               typename TileDataS1::TileDType __in__ src1, unsigned validCol, unsigned validRow)
{
    __ubuf__ typename TileDataS0::DType *src0Ptr = (__ubuf__ typename TileDataS0::DType *)__cce_get_tile_ptr(src0);
    __ubuf__ typename TileDataS1::DType *src1Ptr = (__ubuf__ typename TileDataS1::DType *)__cce_get_tile_ptr(src1);
    __ubuf__ typename TileDataD::DType *dstPtr = (__ubuf__ typename TileDataD::DType *)__cce_get_tile_ptr(dst);
    unsigned TShape1 = TileDataD::Cols;
    __VEC_SCOPE__
    {
        uint16_t batchSize = CCE_VL / static_cast<uint16_t>(sizeof(typename TileDataS1::DType));
        uint16_t loop_num = CEIL(validCol, batchSize);
        for (uint16_t i = 0; i < (uint16_t)validRow; ++i) {
            for (uint16_t j = 0; j < loop_num; ++j) {
                RegTensor<typename TileDataS1::DType> index;
                vlds(index, src1Ptr, (i * TileDataS1::Cols + j * batchSize), NORM);

                uint32_t count = ((j + 1) * batchSize >= validCol ? validCol - j * batchSize : batchSize);
                vector_bool preg = CreatePredicate<typename TileDataS1::DType>(count);

                RegTensor<typename TileDataD::DType> vOutput;
                vgather2(vOutput, src0Ptr, (vector_u16 &)index, preg);
                vsts(vOutput, dstPtr, (i * TShape1 + j * batchSize), NORM_B16, preg);
            }
        }
    }
}

template <typename TileDataD, typename TileDataS0, typename TileDataS1>
__tf__ AICORE void TGather_b16_bc(typename TileDataD::TileDType __out__ dst, typename TileDataS0::TileDType __in__ src0,
                                  typename TileDataS1::TileDType __in__ src1, unsigned validCol, unsigned validRow)
{
    __ubuf__ typename TileDataD::DType *dstPtr = (__ubuf__ typename TileDataD::DType *)__cce_get_tile_ptr(dst);
    __ubuf__ typename TileDataS0::DType *src0Ptr = (__ubuf__ typename TileDataS0::DType *)__cce_get_tile_ptr(src0);
    __ubuf__ typename TileDataS1::DType *src1Ptr = (__ubuf__ typename TileDataS1::DType *)__cce_get_tile_ptr(src1);
    unsigned TShapeDst = TileDataD::Cols;
    unsigned TShapeIdx = TileDataS1::Cols;
    __VEC_SCOPE__
    {
        uint16_t batchSize = CCE_VL / static_cast<uint16_t>(sizeof(typename TileDataS1::DType));
        uint16_t loop_num = CEIL(validCol, batchSize);
        for (uint16_t i = 0; i < (uint16_t)validRow; ++i) {
            for (uint16_t j = 0; j < loop_num; ++j) {
                RegTensor<typename TileDataS1::DType> index;
                vlds(index, src1Ptr, (i * TShapeIdx + j * batchSize), NORM);

                uint32_t count = ((j + 1) * batchSize >= validCol ? validCol - j * batchSize : batchSize);
                vector_bool preg = CreatePredicate<typename TileDataS1::DType>(count);

                RegTensor<typename TileDataD::DType> v_output;
                vgather2_bc(v_output, src0Ptr, (vector_u32 &)index, preg);
                vsts(v_output, dstPtr, (i * TShapeDst + j * batchSize), PK_B32, preg);
            }
        }
    }
}

template <typename TileDataD, typename TileDataS0, typename TileDataS1>
PTO_INTERNAL void TGather(__ubuf__ typename TileDataD::DType *dst, __ubuf__ typename TileDataS0::DType *src0,
                          __ubuf__ typename TileDataS1::DType *src1, unsigned validCol, unsigned validRow)
{
    if constexpr (sizeof(typename TileDataS0::DType) == 4) {
        TGather_b32<TileDataD, TileDataS0, TileDataS1>(dst, src0, src1, validCol, validRow);
    } else if constexpr (sizeof(typename TileDataS0::DType) == 2 && sizeof(typename TileDataS1::DType) == 2) {
        TGather_b16<TileDataD, TileDataS0, TileDataS1>(dst, src0, src1, validCol, validRow);
    } else if constexpr (sizeof(typename TileDataS0::DType) == 2 && sizeof(typename TileDataS1::DType) == 4) {
        TGather_b16_bc<TileDataD, TileDataS0, TileDataS1>(dst, src0, src1, validCol, validRow);
    }
}

template <typename T>
PTO_INTERNAL void PIntlvWithType(MaskReg &dst0_, MaskReg &dst1_, MaskReg src0_, MaskReg src1_)
{
    if constexpr (sizeof(T) == sizeof(float)) {
        pintlv_b32(dst0_, dst1_, src0_, src1_);
    } else if constexpr (sizeof(T) == sizeof(half)) {
        pintlv_b16(dst0_, dst1_, src0_, src1_);
    } else if constexpr (sizeof(T) == sizeof(uint8_t)) {
        pintlv_b8(dst0_, dst1_, src0_, src1_);
    }
}

template <typename TileDataD, typename TileDataS0, typename TileDataS1, typename TileDataTmp>
PTO_INTERNAL void TGATHER_IMPL(TileDataD &dst, TileDataS0 &src0, TileDataS1 &src1, TileDataTmp &tmp)
{
    CheckValid<TileDataD, TileDataS0, TileDataS1>();

    unsigned kValidCols = dst.GetValidCol();
    unsigned kValidRows = dst.GetValidRow();

    TGather<TileDataD, TileDataS0, TileDataS1>(dst.data(), src0.data(), src1.data(), kValidCols, kValidRows);
}

template <typename T, MaskPattern maskPattern>
PTO_INTERNAL MaskReg GetMaskVal()
{
    MaskReg pg0_;
    MaskReg pg1_;
    MaskReg dstPg0;
    MaskReg dstPg1;
    if constexpr (maskPattern == MaskPattern::P0101) {
        pg0_ = PSetWithType<T>(PAT_ALL);
        pg1_ = PSetWithType<T>(PAT_ALLF);
        PIntlvWithType<T>(dstPg0, dstPg1, pg0_, pg1_);
    } else if constexpr (maskPattern == MaskPattern::P1010) {
        pg0_ = PSetWithType<T>(PAT_ALL);
        pg1_ = PSetWithType<T>(PAT_ALLF);
        PIntlvWithType<T>(dstPg0, dstPg1, pg1_, pg0_);
    } else if constexpr (maskPattern == MaskPattern::P0001) {
        pg0_ = PSetWithType<T>(PAT_ALL);
        pg1_ = PSetWithType<T>(PAT_ALLF);
        PIntlvWithType<T>(dstPg0, dstPg1, pg0_, pg1_);
        PIntlvWithType<T>(dstPg0, dstPg1, dstPg0, pg1_);
    } else if constexpr (maskPattern == MaskPattern::P0010) {
        pg0_ = PSetWithType<T>(PAT_ALL);
        pg1_ = PSetWithType<T>(PAT_ALLF);
        PIntlvWithType<T>(dstPg0, dstPg1, pg0_, pg1_);
        PIntlvWithType<T>(dstPg0, dstPg1, pg1_, dstPg0);
    } else if constexpr (maskPattern == MaskPattern::P0100) {
        pg0_ = PSetWithType<T>(PAT_ALL);
        pg1_ = PSetWithType<T>(PAT_ALLF);
        PIntlvWithType<T>(dstPg0, dstPg1, pg1_, pg0_);
        PIntlvWithType<T>(dstPg0, dstPg1, dstPg0, pg1_);
    } else if constexpr (maskPattern == MaskPattern::P1000) {
        pg0_ = PSetWithType<T>(PAT_ALL);
        pg1_ = PSetWithType<T>(PAT_ALLF);
        PIntlvWithType<T>(dstPg0, dstPg1, pg1_, pg0_);
        PIntlvWithType<T>(dstPg0, dstPg1, pg1_, dstPg0);
    } else if constexpr (maskPattern == MaskPattern::P1111) {
        dstPg0 = PSetWithType<T>(PAT_ALL);
    }
    return dstPg0;
}

template <typename DstTileData, typename SrcTileData, MaskPattern maskPattern>
__tf__ AICORE void TGather(typename DstTileData::TileDType __out__ dst_, typename SrcTileData::TileDType __in__ src_,
                           uint16_t validRow, uint16_t validCol)
{
    using T = typename DstTileData::DType;
    constexpr unsigned rowStride = SrcTileData::RowStride;
    __ubuf__ typename DstTileData::DType *dstPtr_ = (__ubuf__ typename DstTileData::DType *)__cce_get_tile_ptr(dst_);
    __ubuf__ typename DstTileData::DType *srcPtr_ = (__ubuf__ typename DstTileData::DType *)__cce_get_tile_ptr(src_);
    __VEC_SCOPE__
    {
        constexpr uint8_t SPR_AR_VALUE = 74;
        constexpr auto sprValue = std::integral_constant<::Spr, static_cast<::Spr>(SPR_AR_VALUE)>();
        sprclr(sprValue);

        RegTensor<T> dstReg_;
        RegTensor<T> srcReg_;
        MaskReg loadMask_;
        MaskReg executeMask_;
        UnalignReg ureg_;
        MaskReg dstPg0_ = GetMaskVal<T, maskPattern>();

        constexpr unsigned elementsPerRepeat = CCE_VL / sizeof(T);
        uint16_t innerRepeatTimes = CeilDivision(validCol, elementsPerRepeat);

        for (uint16_t i = 0; i < validRow; ++i) {
            uint32_t maskValue = validCol;
            for (uint16_t j = 0; j < innerRepeatTimes; ++j) {
                loadMask_ = CreatePredicate<T>(maskValue);
                vlds(srcReg_, srcPtr_ + i * rowStride, j * elementsPerRepeat, NORM);
                pand(executeMask_, dstPg0_, loadMask_, loadMask_);
                vsqz(dstReg_, srcReg_, executeMask_, MODE_STORED);
                vstur(ureg_, dstReg_, dstPtr_, POST_UPDATE);
            }
        }
        vstar(ureg_, dstPtr_);
    }
}

template <typename DstTileData, typename SrcTileData, MaskPattern maskPattern>
PTO_INTERNAL void TGATHER_IMPL(DstTileData &dst, SrcTileData &src)
{
    using U = typename DstTileData::DType;
    using T = typename SrcTileData::DType;
    static_assert(std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t> || std::is_same_v<T, int16_t> ||
                      std::is_same_v<T, uint16_t> || std::is_same_v<T, int32_t> || std::is_same_v<T, uint32_t> ||
                      std::is_same_v<T, half> || std::is_same_v<T, float>,
                  "Fix: TGATHER Src data type must be int8_t/uint8_t/int16_t/uint16_t/int32_t/uint32_t/half/float.");
    static_assert(std::is_same_v<U, int8_t> || std::is_same_v<U, uint8_t> || std::is_same_v<U, int16_t> ||
                      std::is_same_v<U, uint16_t> || std::is_same_v<U, int32_t> || std::is_same_v<U, uint32_t> ||
                      std::is_same_v<U, half> || std::is_same_v<U, float>,
                  "Fix: TGATHER Dst data type must be int8_t/uint8_t/int16_t/uint16_t/int32_t/uint32_t/half/float.");
    static_assert((sizeof(U) == sizeof(T)), "Fix: TGATHER expect same type size for dst and src");
    static_assert((DstTileData::Loc == TileType::Vec) && (SrcTileData::Loc == TileType::Vec),
                  "Fix: TGATHER expect vec TileType");
    static_assert((DstTileData::isRowMajor && SrcTileData::isRowMajor), "Fix: TGATHER expect row major");
    uint16_t rows = src.GetValidRow();
    uint16_t cols = src.GetValidCol();
    TGather<DstTileData, SrcTileData, maskPattern>(dst.data(), src.data(), rows, cols);
}
} // namespace pto
#endif
