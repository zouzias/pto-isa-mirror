/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
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
template <typename DstTileData, typename Src0TileData, typename Src1TileData>
PTO_INTERNAL void CheckValid()
{
    static_assert((sizeof(typename DstTileData::DType) == 1) || (sizeof(typename DstTileData::DType) == 2) ||
                      (sizeof(typename DstTileData::DType) == 4),
                  "Fix: TGATHER expect b8/b16/b32");
    static_assert((sizeof(typename Src1TileData::DType) == 2) || (sizeof(typename Src1TileData::DType) == 4),
                  "Fix: TGATHER expect b16/b32");
    static_assert((std::is_same<typename DstTileData::DType, typename Src0TileData::DType>::value),
                  "Fix: TGATHER expect same datatype for src and dst");
}

template <typename TileDataD, typename TileDataS0, typename TileDataS1>
__tf__ AICORE void TGather_b32(typename TileDataD::TileDType __out__ dst, typename TileDataS0::TileDType __in__ src0,
                               typename TileDataS1::TileDType __in__ src1, unsigned validCol, unsigned validRow)
{
    __ubuf__ typename TileDataD::DType *dstPtr = (__ubuf__ typename TileDataD::DType *)__cce_get_tile_ptr(dst);
    __ubuf__ typename TileDataS0::DType *src0Ptr = (__ubuf__ typename TileDataS0::DType *)__cce_get_tile_ptr(src0);
    __ubuf__ typename TileDataS1::DType *src1Ptr = (__ubuf__ typename TileDataS1::DType *)__cce_get_tile_ptr(src1);
    unsigned elementsPerRepeat = CCE_VL / sizeof(typename TileDataS1::DType);
    uint16_t innerLoopNum = CeilDivision(validCol, elementsPerRepeat);
    __VEC_SCOPE__
    {
        RegTensor<typename TileDataS1::DType> index;
        RegTensor<typename TileDataD::DType> v_output;
        MaskReg preg;
        for (uint16_t i = 0; i < (uint16_t)validRow; ++i) {
            uint32_t sreg = (uint32_t)(validCol);
            for (uint16_t j = 0; j < innerLoopNum; ++j) {
                preg = CreatePredicate<typename TileDataS1::DType>(sreg);
                vlds(index, src1Ptr, (i * TileDataS1::Cols + j * elementsPerRepeat), NORM);
                vgather2(v_output, src0Ptr, (vector_u32 &)index, preg);
                vsts(v_output, dstPtr, (i * TileDataD::Cols + j * elementsPerRepeat), NORM_B32, preg);
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
    uint16_t elementsPerRepeat = CCE_VL / sizeof(typename TileDataS1::DType);
    uint16_t innerLoopNum = CeilDivision(validCol, elementsPerRepeat);
    __VEC_SCOPE__
    {
        MaskReg preg;
        RegTensor<typename TileDataS1::DType> index;
        RegTensor<typename TileDataD::DType> v_output;
        for (uint16_t i = 0; i < (uint16_t)validRow; ++i) {
            uint32_t sreg = (uint32_t)(validCol);
            for (uint16_t j = 0; j < innerLoopNum; ++j) {
                preg = CreatePredicate<typename TileDataS1::DType>(sreg);
                vlds(index, src1Ptr, (i * TileDataS1::Cols + j * elementsPerRepeat), NORM);
                vgather2(v_output, src0Ptr, (vector_u16 &)index, preg);
                vsts(v_output, dstPtr, (i * TileDataD::Cols + j * elementsPerRepeat), NORM_B16, preg);
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
    uint16_t elementsPerRepeat = CCE_VL / sizeof(typename TileDataS1::DType);
    uint16_t innerLoopNum = CeilDivision(validCol, elementsPerRepeat);
    __VEC_SCOPE__
    {
        MaskReg preg;
        RegTensor<typename TileDataS1::DType> index;
        RegTensor<typename TileDataD::DType> v_output;
        for (uint16_t i = 0; i < (uint16_t)validRow; ++i) {
            uint32_t sreg = (uint32_t)(validCol);
            for (uint16_t j = 0; j < innerLoopNum; ++j) {
                preg = CreatePredicate<typename TileDataS1::DType>(sreg);
                vlds(index, src1Ptr, (i * TileDataS1::Cols + j * elementsPerRepeat), NORM);
                vgather2_bc(v_output, src0Ptr, (vector_u32 &)index, preg);
                vsts(v_output, dstPtr, (i * TileDataD::Cols + j * elementsPerRepeat), PK_B32, preg);
            }
        }
    }
}

template <typename TileDataD, typename TileDataS0, typename TileDataS1>
__tf__ AICORE void TGather_fp8_e4m3(typename TileDataD::TileDType __out__ dst,
                                    typename TileDataS0::TileDType __in__ src0,
                                    typename TileDataS1::TileDType __in__ src1, unsigned validCol, unsigned validRow)
{
    __ubuf__ typename TileDataS0::DType *src0Ptr = (__ubuf__ typename TileDataS0::DType *)__cce_get_tile_ptr(src0);
    __ubuf__ typename TileDataS1::DType *src1Ptr = (__ubuf__ typename TileDataS1::DType *)__cce_get_tile_ptr(src1);
    __ubuf__ typename TileDataD::DType *dstPtr = (__ubuf__ typename TileDataD::DType *)__cce_get_tile_ptr(dst);
    uint16_t elementsPerRepeat = CCE_VL / sizeof(typename TileDataS1::DType);
    uint16_t innerLoopNum = CeilDivision(validCol, elementsPerRepeat);
    __VEC_SCOPE__
    {
        MaskReg preg;
        RegTensor<typename TileDataS1::DType> index;
        RegTensor<typename TileDataD::DType> v_output;
        for (uint16_t i = 0; i < (uint16_t)validRow; ++i) {
            uint32_t sreg = (uint32_t)(validCol);
            for (uint16_t j = 0; j < innerLoopNum; ++j) {
                preg = CreatePredicate<typename TileDataS1::DType>(sreg);
                vlds(index, src1Ptr, (i * TileDataS1::Cols + j * elementsPerRepeat), NORM);
                vgather2(v_output, src0Ptr, (vector_u16 &)index, preg);
                vsts(v_output, dstPtr, (i * TileDataD::Cols + j * elementsPerRepeat), PK_B16, preg);
            }
        }
    }
}

template <typename TileDataD, typename TileDataS0, typename TileDataS1>
__tf__ AICORE void TGather_fp8_e5m2(typename TileDataD::TileDType __out__ dst,
                                    typename TileDataS0::TileDType __in__ src0,
                                    typename TileDataS1::TileDType __in__ src1, unsigned validCol, unsigned validRow)
{
    __ubuf__ typename TileDataD::DType *dstPtr = (__ubuf__ typename TileDataD::DType *)__cce_get_tile_ptr(dst);
    __ubuf__ typename TileDataS0::DType *src0Ptr = (__ubuf__ typename TileDataS0::DType *)__cce_get_tile_ptr(src0);
    __ubuf__ typename TileDataS1::DType *src1Ptr = (__ubuf__ typename TileDataS1::DType *)__cce_get_tile_ptr(src1);
    uint16_t elementsPerRepeat = CCE_VL / sizeof(typename TileDataS1::DType);
    uint16_t innerLoopNum = CeilDivision(validCol, elementsPerRepeat);
    __VEC_SCOPE__
    {
        MaskReg preg;
        RegTensor<typename TileDataS1::DType> index;
        RegTensor<typename TileDataD::DType> v_output;
        for (uint16_t i = 0; i < (uint16_t)validRow; ++i) {
            uint32_t sreg = (uint32_t)(validCol);
            for (uint16_t j = 0; j < innerLoopNum; ++j) {
                preg = CreatePredicate<typename TileDataS1::DType>(sreg);
                vlds(index, src1Ptr, (i * TileDataS1::Cols + j * elementsPerRepeat), NORM);
                vgather2(v_output, src0Ptr, (vector_u16 &)index, preg);
                vsts(v_output, dstPtr, (i * TileDataD::Cols + j * elementsPerRepeat), PK_B16, preg);
            }
        }
    }
}

template <typename TileDataD, typename TileDataS0, typename TileDataS1, typename TileDataTmp>
PTO_INTERNAL void TGATHER_IMPL(TileDataD &dst, TileDataS0 &src0, TileDataS1 &src1, TileDataTmp &tmp)
{
    CheckValid<TileDataD, TileDataS0, TileDataS1>();

    unsigned kValidCols = dst.GetValidCol();
    unsigned kValidRows = dst.GetValidRow();

    if constexpr (sizeof(typename TileDataS0::DType) == 4) {
        TGather_b32<TileDataD, TileDataS0, TileDataS1>(dst.data(), src0.data(), src1.data(), kValidCols, kValidRows);
    } else if constexpr (sizeof(typename TileDataS0::DType) == 2 && sizeof(typename TileDataS1::DType) == 2) {
        TGather_b16<TileDataD, TileDataS0, TileDataS1>(dst.data(), src0.data(), src1.data(), kValidCols, kValidRows);
    } else if constexpr (sizeof(typename TileDataS0::DType) == 2 && sizeof(typename TileDataS1::DType) == 4) {
        TGather_b16_bc<TileDataD, TileDataS0, TileDataS1>(dst.data(), src0.data(), src1.data(), kValidCols, kValidRows);
    } else if constexpr (std::is_same<typename TileDataS0::DType, float8_e4m3_t>::value) {
        TGather_fp8_e4m3<TileDataD, TileDataS0, TileDataS1>(dst.data(), src0.data(), src1.data(), kValidCols,
                                                            kValidRows);
    } else {
        TGather_fp8_e5m2<TileDataD, TileDataS0, TileDataS1>(dst, src0, src1, kValidCols, kValidRows);
    }
}

template <typename T>
PTO_INTERNAL void PIntlvWithType(MaskReg &dst0, MaskReg &dst1, MaskReg src0, MaskReg src1)
{
    if constexpr (sizeof(T) == sizeof(float)) {
        pintlv_b32(dst0, dst1, src0, src1);
    } else if constexpr (sizeof(T) == sizeof(half)) {
        pintlv_b16(dst0, dst1, src0, src1);
    } else if constexpr (sizeof(T) == sizeof(uint8_t)) {
        pintlv_b8(dst0, dst1, src0, src1);
    }
}

template <typename T, MaskPattern maskPattern>
PTO_INTERNAL MaskReg GetMaskVal()
{
    MaskReg pg0;
    MaskReg pg1;
    MaskReg dstPg0;
    MaskReg dstPg1;
    if constexpr (maskPattern == MaskPattern::P0101) {
        pg0 = PSetWithType<T>(PAT_ALL);
        pg1 = PSetWithType<T>(PAT_ALLF);
        PIntlvWithType<T>(dstPg0, dstPg1, pg0, pg1);
    } else if constexpr (maskPattern == MaskPattern::P1010) {
        pg0 = PSetWithType<T>(PAT_ALL);
        pg1 = PSetWithType<T>(PAT_ALLF);
        PIntlvWithType<T>(dstPg0, dstPg1, pg1, pg0);
    } else if constexpr (maskPattern == MaskPattern::P0001) {
        pg0 = PSetWithType<T>(PAT_ALL);
        pg1 = PSetWithType<T>(PAT_ALLF);
        PIntlvWithType<T>(dstPg0, dstPg1, pg0, pg1);
        PIntlvWithType<T>(dstPg0, dstPg1, dstPg0, pg1);
    } else if constexpr (maskPattern == MaskPattern::P0010) {
        pg0 = PSetWithType<T>(PAT_ALL);
        pg1 = PSetWithType<T>(PAT_ALLF);
        PIntlvWithType<T>(dstPg0, dstPg1, pg0, pg1);
        PIntlvWithType<T>(dstPg0, dstPg1, pg1, dstPg0);
    } else if constexpr (maskPattern == MaskPattern::P0100) {
        pg0 = PSetWithType<T>(PAT_ALL);
        pg1 = PSetWithType<T>(PAT_ALLF);
        PIntlvWithType<T>(dstPg0, dstPg1, pg1, pg0);
        PIntlvWithType<T>(dstPg0, dstPg1, dstPg0, pg1);
    } else if constexpr (maskPattern == MaskPattern::P1000) {
        pg0 = PSetWithType<T>(PAT_ALL);
        pg1 = PSetWithType<T>(PAT_ALLF);
        PIntlvWithType<T>(dstPg0, dstPg1, pg1, pg0);
        PIntlvWithType<T>(dstPg0, dstPg1, pg1, dstPg0);
    } else if constexpr (maskPattern == MaskPattern::P1111) {
        dstPg0 = PSetWithType<T>(PAT_ALL);
    }
    return dstPg0;
}

template <typename DstTileData, typename SrcTileData, MaskPattern maskPattern, auto gatherType = GatherAxis::GATHER_ROW>
__tf__ AICORE void TGather(typename DstTileData::TileDType __out__ dst, typename SrcTileData::TileDType __in__ src,
                           unsigned validRow, unsigned validCol)
{
    using T = typename DstTileData::DType;
    constexpr unsigned srcStride = SrcTileData::RowStride;
    constexpr unsigned dstStride = DstTileData::RowStride;
    __ubuf__ typename DstTileData::DType *dstPtr = (__ubuf__ typename DstTileData::DType *)__cce_get_tile_ptr(dst);
    __ubuf__ typename DstTileData::DType *srcPtr = (__ubuf__ typename DstTileData::DType *)__cce_get_tile_ptr(src);
    constexpr uint16_t nElemPerVL = CCE_VL / sizeof(T);
    uint16_t repeatTimes = CeilDivision(validCol, nElemPerVL);

    __VEC_SCOPE__
    {
        RegTensor<T> srcReg;
        MaskReg loadMask;
        uint32_t maskValue = 0;
        if constexpr (gatherType == GatherAxis::GATHER_COL) {
            uint16_t stride = 0;
            constexpr auto distValue =
                std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<T, DistVST::DIST_NORM>())>();
            for (uint16_t i = 0; i < (uint16_t)validRow; ++i) {
                stride = GetStrideByMask<maskPattern, srcStride>(i);
                maskValue = validCol;
                for (uint16_t j = 0; j < repeatTimes; ++j) {
                    loadMask = CreatePredicate<T>(maskValue);
                    vlds(srcReg, srcPtr, stride + j * nElemPerVL, NORM);
                    vsts(srcReg, dstPtr, i * dstStride + j * nElemPerVL, distValue, loadMask);
                }
            }
        } else {
            constexpr uint8_t SPR_AR_VALUE = 74;
            constexpr auto sprValue = std::integral_constant<::Spr, static_cast<::Spr>(SPR_AR_VALUE)>();
            sprclr(sprValue);

            MaskReg dstPg0 = GetMaskVal<T, maskPattern>();
            RegTensor<T> dstReg;
            MaskReg executeMask;
            UnalignReg ureg;

            for (uint16_t i = 0; i < (uint16_t)validRow; ++i) {
                maskValue = validCol;
                for (uint16_t j = 0; j < repeatTimes; ++j) {
                    loadMask = CreatePredicate<T>(maskValue);
                    vlds(srcReg, srcPtr + i * srcStride, j * nElemPerVL, NORM);
                    pand(executeMask, dstPg0, loadMask, loadMask);
                    vsqz(dstReg, srcReg, executeMask, MODE_STORED);
                    vstur(ureg, dstReg, dstPtr, POST_UPDATE);
                }
            }
            vstar(ureg, dstPtr);
        }
    }
}

template <typename DstTileData, typename SrcTileData, MaskPattern maskPattern, auto gatherType = GatherAxis::GATHER_ROW>
PTO_INTERNAL void TGATHER_IMPL(DstTileData &dst, SrcTileData &src)
{
    using T = typename SrcTileData::DType;
    using U = typename DstTileData::DType;
    static_assert(std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t> || std::is_same_v<T, int16_t> ||
                      std::is_same_v<T, uint16_t> || std::is_same_v<T, int32_t> || std::is_same_v<T, uint32_t> ||
                      std::is_same_v<T, half> || std::is_same_v<T, bfloat16_t> || std::is_same_v<T, float> ||
                      std::is_same_v<T, float8_e4m3_t> || std::is_same_v<T, float8_e5m2_t> ||
                      std::is_same_v<T, hifloat8_t>,
                  "Fix: TGATHER Src data type must be int8_t/uint8_t/int16_t/uint16_t/int32_t/uint32_t/"
                  "half/bfloat16_t/float/float8_e4m3_t/float8_e5m2_t/hifloat8_t.");
    static_assert(std::is_same_v<U, int8_t> || std::is_same_v<U, uint8_t> || std::is_same_v<U, int16_t> ||
                      std::is_same_v<U, uint16_t> || std::is_same_v<U, int32_t> || std::is_same_v<U, uint32_t> ||
                      std::is_same_v<U, half> || std::is_same_v<U, bfloat16_t> || std::is_same_v<U, float> ||
                      std::is_same_v<U, float8_e4m3_t> || std::is_same_v<U, float8_e5m2_t> ||
                      std::is_same_v<U, hifloat8_t>,
                  "Fix: TGATHER Dst data type must be int8_t/uint8_t/int16_t/uint16_t/int32_t/uint32_t/"
                  "half/bfloat16_t/float/float8_e4m3_t/float8_e5m2_t/hifloat8_t.");
    static_assert((sizeof(U) == sizeof(T)), "Fix: TGATHER expect same type size for dst and src");
    static_assert((DstTileData::Loc == TileType::Vec) && (SrcTileData::Loc == TileType::Vec),
                  "Fix: TGATHER expect vec TileType");
    static_assert((DstTileData::isRowMajor && SrcTileData::isRowMajor), "Fix: TGATHER expect row major");
    unsigned rows = src.GetValidRow();
    unsigned cols = src.GetValidCol();
    TGather<DstTileData, SrcTileData, maskPattern, gatherType>(dst.data(), src.data(), rows, cols);
}

template <typename TileDataD, typename TileDataS, typename TileDataS1, typename TileDataC, CmpMode cmpMode>
__tf__ AICORE void TGather_float_gt(typename TileDataD::TileDType __out__ dst,
                                    typename TileDataS::TileDType __in__ src0,
                                    typename TileDataS1::TileDType __in__ k_value, uint32_t offset,
                                    typename TileDataC::TileDType __in__ cdst, unsigned srcValidCol,
                                    unsigned srcValidRow, unsigned dstValidCol, unsigned dstValidRow)
{
    __ubuf__ typename TileDataS::DType *src0Ptr = (__ubuf__ typename TileDataS::DType *)__cce_get_tile_ptr(src0);
    __ubuf__ typename TileDataC::DType *cdstPtr = (__ubuf__ typename TileDataC::DType *)__cce_get_tile_ptr(cdst);
    __ubuf__ typename TileDataD::DType *dstPtr = (__ubuf__ typename TileDataD::DType *)__cce_get_tile_ptr(dst);
    __ubuf__ typename TileDataS1::DType *kvaluePtr = (__ubuf__ typename TileDataS1::DType *)__cce_get_tile_ptr(k_value);

    constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(typename TileDataS::DType);
    uint16_t repeatTimes = CeilDivision(srcValidCol, elementsPerRepeat);
    unsigned srcRowStride = TileDataS::RowStride;
    unsigned dstRowStride = TileDataD::RowStride;

    __VEC_SCOPE__
    {
        vector_bool preg_b32 = pset_b32(PAT_ALL);
        vector_align align_index;
        vector_s32 add_offset;
        vector_s32 index;
        vci(index, offset, INC_ORDER);
        sprclr(SPR_AR);
        vbr(add_offset, 0x00000040);
        MaskReg preg;

        for (uint16_t i = 0; i < (uint16_t)srcValidRow; ++i) {
            uint32_t cols = (uint32_t)(srcValidCol);
            typename TileDataS1::DType k_scalar1 = *(kvaluePtr + i);
            for (uint16_t j = 0; j < repeatTimes; ++j) {
                RegTensor<typename TileDataS::DType> score;
                vector_s32 score_s32;
                vlds(score, src0Ptr, (i * TileDataS::Cols + j * elementsPerRepeat), NORM);
                // f32->s32
                vcvt(score_s32, score, preg_b32, ROUND_R, RS_DISABLE);

                uint32_t count = (cols > elementsPerRepeat) ? elementsPerRepeat : cols;
                preg = CreatePredicate<typename TileDataS::DType>(count);

                vector_bool pout_ge;
                vector_s32 sqz_index_out;
                vcmps_gt(pout_ge, (vector_u32)score_s32, k_scalar1, preg);
                vsqz(sqz_index_out, index, pout_ge, MODE_STORED);
                vstur(align_index, (vector_u32)sqz_index_out, (__ubuf__ uint32_t *)dstPtr, POST_UPDATE);
                vadd(index, index, add_offset, preg_b32, MODE_ZEROING);
                cols -= elementsPerRepeat;
            }
            vstar(align_index, (__ubuf__ uint32_t *)dstPtr);
            sprsts(SPR_AR, cdstPtr, i * sizeof(typename TileDataC::DType));
            sprclr(SPR_AR);
            dstPtr = dstPtr + TileDataD::Cols;
        }
    }
}

template <typename TileDataD, typename TileDataS, typename TileDataS1, typename TileDataC, CmpMode cmpMode>
__tf__ AICORE void TGather_float_eq(typename TileDataD::TileDType __out__ dst,
                                    typename TileDataS::TileDType __in__ src0,
                                    typename TileDataS1::TileDType __in__ k_value, uint32_t offset,
                                    typename TileDataC::TileDType __in__ cdst, unsigned srcValidCol,
                                    unsigned srcValidRow, unsigned dstValidCol, unsigned dstValidRow)
{
    __ubuf__ typename TileDataD::DType *dstPtr = (__ubuf__ typename TileDataD::DType *)__cce_get_tile_ptr(dst);
    __ubuf__ typename TileDataS::DType *src0Ptr = (__ubuf__ typename TileDataS::DType *)__cce_get_tile_ptr(src0);
    __ubuf__ typename TileDataC::DType *cdstPtr = (__ubuf__ typename TileDataC::DType *)__cce_get_tile_ptr(cdst);
    __ubuf__ typename TileDataS1::DType *kvaluePtr = (__ubuf__ typename TileDataS1::DType *)__cce_get_tile_ptr(k_value);

    using T = typename TileDataD::DType;
    constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(typename TileDataS::DType);
    uint16_t repeatTimes = CeilDivision(srcValidCol, elementsPerRepeat);

    __VEC_SCOPE__
    {
        vector_bool preg_b32 = pset_b32(PAT_ALL);
        vector_align align_index;
        vector_s32 index;
        vci(index, offset, INC_ORDER);
        vector_s32 add_offset;
        MaskReg preg;
        vbr(add_offset, 0x00000040);
        sprclr(SPR_AR);

        for (uint16_t i = 0; i < (uint16_t)srcValidRow; ++i) {
            uint32_t cols1 = (uint32_t)(srcValidCol);
            typename TileDataS1::DType k_scalar1 = *(kvaluePtr + i);
            for (uint16_t j = 0; j < repeatTimes; ++j) {
                vector_s32 score_s32;
                RegTensor<typename TileDataS::DType> score;
                vlds(score, src0Ptr, (i * TileDataS::Cols + j * elementsPerRepeat), NORM);
                vcvt(score_s32, score, preg_b32, ROUND_R, RS_DISABLE);

                uint32_t count = (cols1 > elementsPerRepeat) ? elementsPerRepeat : cols1;
                preg = CreatePredicate<typename TileDataS::DType>(count);

                vector_bool pout_eq;
                vector_s32 sqz_index_out;
                vcmps_eq(pout_eq, (vector_u32)score_s32, k_scalar1, preg);
                vsqz(sqz_index_out, index, pout_eq, MODE_STORED);
                vadd(index, index, add_offset, preg_b32, MODE_ZEROING);
                vstur(align_index, (vector_u32)sqz_index_out, (__ubuf__ uint32_t *)dstPtr, POST_UPDATE);
                cols1 -= elementsPerRepeat;
            }
            vstar(align_index, (__ubuf__ uint32_t *)dstPtr);
            sprsts(SPR_AR, cdstPtr, i * sizeof(typename TileDataC::DType));
            sprclr(SPR_AR);
            dstPtr = dstPtr + TileDataD::Cols;
        }
    }
}

template <typename TileDataD, typename TileDataS, typename TileDataS1, typename TileDataC, CmpMode cmpMode>
__tf__ AICORE void TGather_b32_gt(typename TileDataD::TileDType __out__ dst, typename TileDataS::TileDType __in__ src0,
                                  typename TileDataS1::TileDType __in__ k_value, uint32_t offset,
                                  typename TileDataC::TileDType __in__ cdst, unsigned srcValidCol, unsigned srcValidRow,
                                  unsigned dstValidCol, unsigned dstValidRow)
{
    __ubuf__ typename TileDataD::DType *dstPtr = (__ubuf__ typename TileDataD::DType *)__cce_get_tile_ptr(dst);
    __ubuf__ typename TileDataS::DType *src0Ptr = (__ubuf__ typename TileDataS::DType *)__cce_get_tile_ptr(src0);
    __ubuf__ typename TileDataC::DType *cdstPtr = (__ubuf__ typename TileDataC::DType *)__cce_get_tile_ptr(cdst);
    __ubuf__ typename TileDataS1::DType *kvaluePtr = (__ubuf__ typename TileDataS1::DType *)__cce_get_tile_ptr(k_value);

    constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(typename TileDataS::DType);
    uint16_t repeatTimes = CeilDivision(srcValidCol, elementsPerRepeat);
    unsigned srcRowStride = TileDataS::RowStride;
    unsigned dstRowStride = TileDataD::RowStride;

    __VEC_SCOPE__
    {
        vector_bool preg_b32 = pset_b32(PAT_ALL);
        vector_s32 add_offset;
        vector_align align_index1;
        vector_s32 index1;
        MaskReg preg;
        vci(index1, offset, INC_ORDER);
        sprclr(SPR_AR);
        vbr(add_offset, 0x00000040);

        for (uint16_t i = 0; i < (uint16_t)srcValidRow; ++i) {
            uint32_t cols = (uint32_t)(srcValidCol);
            typename TileDataS1::DType k_scalar = *(kvaluePtr + i);
            for (uint16_t j = 0; j < repeatTimes; ++j) {
                RegTensor<typename TileDataS::DType> score;
                vlds(score, src0Ptr, (i * TileDataS::Cols + j * elementsPerRepeat), NORM);

                uint32_t count = (cols > elementsPerRepeat) ? elementsPerRepeat : cols;
                preg = CreatePredicate<typename TileDataS::DType>(count);

                vector_bool pout_ge;
                vector_s32 sqz_index_out;
                vcmps_gt(pout_ge, (vector_u32)score, k_scalar, preg);
                vsqz(sqz_index_out, index1, pout_ge, MODE_STORED);
                vstur(align_index1, (vector_u32)sqz_index_out, (__ubuf__ uint32_t *)dstPtr, POST_UPDATE);
                vadd(index1, index1, add_offset, preg_b32, MODE_ZEROING);
                cols -= elementsPerRepeat;
            }
            vstar(align_index1, (__ubuf__ uint32_t *)dstPtr);
            sprsts(SPR_AR, cdstPtr, i * sizeof(typename TileDataC::DType));
            sprclr(SPR_AR);
            dstPtr = dstPtr + TileDataD::Cols;
        }
    }
}

template <typename TileDataD, typename TileDataS, typename TileDataS1, typename TileDataC, CmpMode cmpMode>
__tf__ AICORE void TGather_b32_eq(typename TileDataD::TileDType __out__ dst, typename TileDataS::TileDType __in__ src0,
                                  typename TileDataS1::TileDType __in__ k_value, uint32_t offset,
                                  typename TileDataC::TileDType __in__ cdst, unsigned srcValidCol, unsigned srcValidRow,
                                  unsigned dstValidCol, unsigned dstValidRow)
{
    __ubuf__ typename TileDataD::DType *dstPtr = (__ubuf__ typename TileDataD::DType *)__cce_get_tile_ptr(dst);
    __ubuf__ typename TileDataS::DType *src0Ptr = (__ubuf__ typename TileDataS::DType *)__cce_get_tile_ptr(src0);
    __ubuf__ typename TileDataS1::DType *kvaluePtr = (__ubuf__ typename TileDataS1::DType *)__cce_get_tile_ptr(k_value);
    __ubuf__ typename TileDataC::DType *cdstPtr = (__ubuf__ typename TileDataC::DType *)__cce_get_tile_ptr(cdst);

    using T = typename TileDataD::DType;
    constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(typename TileDataS::DType);
    uint16_t repeatTimes = CeilDivision(srcValidCol, elementsPerRepeat);

    __VEC_SCOPE__
    {
        vector_bool preg_b32 = pset_b32(PAT_ALL);
        vector_align align_index2;
        vector_s32 index2;
        vci(index2, offset, INC_ORDER);
        vector_s32 add_offset;
        vbr(add_offset, 0x00000040);
        sprclr(SPR_AR);
        MaskReg preg;

        for (uint16_t i = 0; i < (uint16_t)srcValidRow; ++i) {
            uint32_t cols = (uint32_t)(srcValidCol);
            typename TileDataS1::DType k_scalar = *(kvaluePtr + i);
            for (uint16_t j = 0; j < repeatTimes; ++j) {
                RegTensor<typename TileDataS::DType> score;
                vlds(score, src0Ptr, (i * TileDataS::Cols + j * elementsPerRepeat), NORM);

                uint32_t count = (cols > elementsPerRepeat) ? elementsPerRepeat : cols;
                preg = CreatePredicate<typename TileDataS::DType>(count);

                vector_bool pout_eq;
                vector_s32 sqz_index_out;
                vcmps_eq(pout_eq, (vector_u32)score, k_scalar, preg);
                vsqz(sqz_index_out, index2, pout_eq, MODE_STORED);
                vadd(index2, index2, add_offset, preg_b32, MODE_ZEROING);
                vstur(align_index2, (vector_u32)sqz_index_out, (__ubuf__ uint32_t *)dstPtr, POST_UPDATE);
                cols -= elementsPerRepeat;
            }
            vstar(align_index2, (__ubuf__ uint32_t *)dstPtr);
            sprsts(SPR_AR, cdstPtr, i * sizeof(typename TileDataC::DType));
            sprclr(SPR_AR);
            dstPtr = dstPtr + TileDataD::Cols;
        }
    }
}

template <typename TileDataD, typename TileDataS, typename TileDataS1, typename TileDataC, CmpMode cmpMode>
__tf__ AICORE void TGather_b16_gt(typename TileDataD::TileDType __out__ dst, typename TileDataS::TileDType __in__ src0,
                                  typename TileDataS1::TileDType __in__ k_value, uint32_t offset,
                                  typename TileDataC::TileDType __in__ cdst, unsigned srcValidCol, unsigned srcValidRow,
                                  unsigned dstValidCol, unsigned dstValidRow)
{
    __ubuf__ typename TileDataD::DType *dstPtr = (__ubuf__ typename TileDataD::DType *)__cce_get_tile_ptr(dst);
    __ubuf__ typename TileDataS::DType *src0Ptr = (__ubuf__ typename TileDataS::DType *)__cce_get_tile_ptr(src0);
    __ubuf__ typename TileDataC::DType *cdstPtr = (__ubuf__ typename TileDataC::DType *)__cce_get_tile_ptr(cdst);
    __ubuf__ typename TileDataS1::DType *kvaluePtr = (__ubuf__ typename TileDataS1::DType *)__cce_get_tile_ptr(k_value);

    unsigned dstStride = TileDataD::Cols;
    constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(typename TileDataD::DType);
    uint16_t repeatTimes = CeilDivision(srcValidCol, elementsPerRepeat);

    __VEC_SCOPE__
    {
        vector_bool preg_b32 = pset_b32(PAT_ALL);
        vector_bool preg_b16 = pset_b16(PAT_ALL);
        vector_s32 add_offset;
        vector_align align_index;
        vector_u32 mask_k;
        vector_s32 index;

        vci(index, offset, INC_ORDER);
        vbr(add_offset, 0x00000040);
        sprclr(SPR_AR);

        for (uint16_t i = 0; i < (uint16_t)srcValidRow; ++i) {
            typename TileDataS1::DType k_scalar = *(kvaluePtr + i);
            float k_value_f32 = (float)k_scalar;
            vbr(mask_k, k_value_f32);
            for (uint16_t j = 0; j < repeatTimes; ++j) {
                vector_u16 score;
                vlds(score, (__ubuf__ uint16_t *)src0Ptr, (i * TileDataS::Cols + j * elementsPerRepeat), UNPK_B16);

                vector_bool pout_ge;
                vector_s32 sqz_index_out;
                vcmp_gt(pout_ge, (vector_u32)score, mask_k, preg_b32);
                vsqz(sqz_index_out, index, pout_ge, MODE_STORED);
                vstur(align_index, (vector_u32)sqz_index_out, (__ubuf__ uint32_t *)dstPtr, POST_UPDATE);
                vadd(index, index, add_offset, preg_b32, MODE_ZEROING);
            }
            vstar(align_index, (__ubuf__ uint32_t *)dstPtr);
            sprsts(SPR_AR, cdstPtr, i * sizeof(typename TileDataC::DType));
            sprclr(SPR_AR);
            dstPtr = dstPtr + dstStride;
        }
    }
}

// =============================================================================
// TGather_b16_eq — bounded-K stream compaction (vsqz NO_STORED + vscatter)
// -----------------------------------------------------------------------------
// Per row, write the global flat indices of the first K positions whose b16
// value bit-exactly equals k_scalar[i] into dst[i][0..K-1] (golden: gen_data.py
// EQ mode). Per b16 VL (UNPK_B16 -> 64 b32 lanes):
//
//   1) vlds UNPK_B16         load 64 b16 elems into b32 lanes
//   2) vcmps_eq(p_eq, ...)   bit-level EQ under auto-decrementing tail predicate
//   3) vsqz NO_STORED ×2     squeeze (v_in_idx, v_one) under p_eq:
//                              v_sqz_idx = [I_0..I_{N-1}, 0..]   (N = popcount)
//                              v_sqz_one = [1..1, 0..]
//   4) vcmps_eq -> p_sqz     predicate covering the first N lanes
//   5) vcmps_lt -> p_bnd     keep only lanes with v_out_pos < K
//   6) vscatter              dst_row[v_out_pos[t]] = v_sqz_idx[t]
//   7) vcadd + vdup LOWEST   N broadcast to all lanes
//   8) vadd                  v_out_pos += N ; v_in_idx += 64
//
// Pros vs. the legacy VSQZ STORED + VSTUR + SPR_AR impl:
//   + Generic.  No use of the SPR_AR scalar pointer register or sprclr/sprsts
//     scratch protocol; same code shape works for b8/b16/b32 by swapping the
//     load mode, k zero-extension width, and step constant. Naturally extends
//     to GT/LT/GE/LE/NE by swapping vcmps_eq->vcmps_*.
//   + K-bounded.  The vcmps_lt against dstValidCol guards every scatter, so
//     dst[i] is never overwritten past K, even on rows whose match count
//     exceeds K. Legacy impl relied on AR-pointer post-increment never tripping
//     past the row stride, which is only safe when caller guarantees M_i <= K.
//   + Per-row reset is a single vci, not an SPR clear+restore round-trip.
//   + Valid tail handled by CreatePredicate's POST_UPDATE auto-decrement; no
//     manual remaining-count bookkeeping inside the loop.
//
// Cons / tradeoffs:
//   - Higher static instruction count per block (~10 vec ops vs ~5 in legacy).
//   - Adds a SLIDE-pipe round-trip (vcadd -> vdup POS_LOWEST -> vadd) on the
//     critical path; legacy keeps the running count entirely in the scalar SPR.
//   - Two VSQZ issues per block instead of one (the second one just produces
//     the bit-vector for popcount + first-N-lanes predicate).
//   - Uses GSU (vscatter) instead of LSU+POST_UPDATE; on workloads that issue
//     other GSU traffic this can become the bottleneck.
//
// Numerical stability: identical. Both impls produce bit-exact matches against
// the b16 bit pattern of k_scalar; no float arithmetic is performed.
//
// Hardware-friendliness: spreads work across more pipes (LSU+VEC+SLIDE+GSU)
// rather than serializing through the SPR_AR scalar resource, which tends to
// hide better behind unrelated scalar work and avoids the SPR clear/restore
// fence that legacy needs around row boundaries. On the other hand the SLIDE
// dependency chain (vcadd->vdup->vadd) is fully serial and bounds the per-block
// latency.
//
// Correctness sketch: with INC_ORDER, v_out_pos = [base..base+63] before the
// scatter; p_sqz selects exactly the N populated lanes after vsqz; p_bnd then
// drops any lane with position >= K. So writes are dst_row[base+t] = I_t for
// t in [0, min(N, K-base)). Induction over blocks (with v_out_pos += N) yields
// dst_row[k] = k-th matching global flat index for k in [0, min(M_i, K)). ∎
// =============================================================================
template <typename TileDataD, typename TileDataS, typename TileDataS1, typename TileDataC, CmpMode cmpMode>
__tf__ AICORE void TGather_b16_eq(typename TileDataD::TileDType __out__ dst, typename TileDataS::TileDType __in__ src0,
                                  typename TileDataS1::TileDType __in__ k_value, uint32_t offset,
                                  typename TileDataC::TileDType __in__ cdst, unsigned srcValidCol, unsigned srcValidRow,
                                  unsigned dstValidCol, unsigned dstValidRow)
{
    __ubuf__ typename TileDataD::DType *dstPtr = (__ubuf__ typename TileDataD::DType *)__cce_get_tile_ptr(dst);
    __ubuf__ typename TileDataS::DType *src0Ptr = (__ubuf__ typename TileDataS::DType *)__cce_get_tile_ptr(src0);
    __ubuf__ typename TileDataC::DType *cdstPtr = (__ubuf__ typename TileDataC::DType *)__cce_get_tile_ptr(cdst);
    __ubuf__ typename TileDataS1::DType *kvaluePtr = (__ubuf__ typename TileDataS1::DType *)__cce_get_tile_ptr(k_value);
    (void)cdstPtr;  // intermediate UB tile, never compared against golden

    // After UNPK_B16, every repeat consumes 64 b16 elements -> 64 b32 lanes.
    constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(typename TileDataD::DType);
    uint16_t repeatTimes = CeilDivision(srcValidCol, elementsPerRepeat);

    __VEC_SCOPE__
    {
        vector_bool preg_b32_all = pset_b32(PAT_ALL);

        vector_u32 v_one;
        vbr(v_one, (uint32_t)1);

        vector_s32 v_step_vl;
        vbr(v_step_vl, (int32_t)elementsPerRepeat);
        vector_u32 v_out_col_size;
        vbr(v_out_col_size, 0);

        // Global flat-index counter; continuous across rows and blocks.
        vector_s32 v_in_idx;
        vci(v_in_idx, (int32_t)offset, INC_ORDER);

        for (uint16_t i = 0; i < (uint16_t)srcValidRow; ++i) {
            uint32_t k_u32 = (uint32_t)(*(kvaluePtr + i));  // bit-pattern EQ on b16, zero-extended

            // Per-row reset of the per-lane output positions: lane k starts at k.
            vector_s32 v_out_pos;
            vci(v_out_pos, 0, INC_ORDER);

            // POST_UPDATE inside CreatePredicate auto-decrements sreg by VL (=64
            // for b32) on each call, so no manual remaining-count bookkeeping.
            uint32_t sreg = (uint32_t)srcValidCol;

            for (uint16_t j = 0; j < repeatTimes; ++j) {
                vector_bool p_valid = CreatePredicate<uint32_t>(sreg);

                vector_u16 score;
                vlds(score, (__ubuf__ uint16_t *)src0Ptr, (i * TileDataS::Cols + j * elementsPerRepeat), UNPK_B16);

                vector_bool p_eq;
                vcmps_eq(p_eq, (vector_u32)score, k_u32, p_valid);

                vector_u32 v_sqz_idx;
                vector_u32 v_sqz_one;
                vsqz(v_sqz_idx, (vector_u32 &)v_in_idx, p_eq, MODE_NO_STORED);
                vsqz(v_sqz_one, v_one, p_eq, MODE_NO_STORED);

                vector_bool p_sqz;
                vcmps_eq(p_sqz, v_sqz_one, (uint32_t)1, preg_b32_all);

                vector_bool p_bnd;
                vcmps_lt(p_bnd, (vector_u32 &)v_out_pos, (uint32_t)dstValidCol, p_sqz);

                vscatter(v_sqz_idx, (__ubuf__ uint32_t *)dstPtr, (vector_u32 &)v_out_pos, p_bnd);

                vector_u32 v_n;
                vcadd(v_n, v_sqz_one, p_bnd, MODE_ZEROING);
                vadd(v_out_col_size, v_out_col_size, v_n, preg_b32_all, MODE_ZEROING);

                vector_u32 v_n_brc;
                vdup(v_n_brc, v_n, preg_b32_all, POS_LOWEST, MODE_ZEROING);

                vadd(v_out_pos, v_out_pos, (vector_s32 &)v_n_brc, preg_b32_all, MODE_ZEROING);
                vadd(v_in_idx, v_in_idx, v_step_vl, preg_b32_all, MODE_ZEROING);
            }
            vmuls(v_out_col_size, v_out_col_size, (uint32_t)sizeof(typename TileDataD::DType), preg_b32_all, MODE_ZEROING);
            vsts(v_out_col_size, (__ubuf__ uint32_t *)cdstPtr, i, ONEPT_B32, preg_b32_all);

            dstPtr = dstPtr + TileDataD::Cols;
        }
    }
}

template <typename TileDataD, typename TileDataS, typename TileDataS1, typename TileDataC, CmpMode cmpMode>
__tf__ AICORE void TGather_half_gt(typename TileDataD::TileDType __out__ dst, typename TileDataS::TileDType __in__ src0,
                                   typename TileDataS1::TileDType __in__ k_value, uint32_t offset,
                                   typename TileDataC::TileDType __in__ cdst, unsigned srcValidCol,
                                   unsigned srcValidRow, unsigned dstValidCol, unsigned dstValidRow)
{
    __ubuf__ typename TileDataD::DType *dstPtr = (__ubuf__ typename TileDataD::DType *)__cce_get_tile_ptr(dst);
    __ubuf__ typename TileDataS::DType *src0Ptr = (__ubuf__ typename TileDataS::DType *)__cce_get_tile_ptr(src0);
    __ubuf__ typename TileDataC::DType *cdstPtr = (__ubuf__ typename TileDataC::DType *)__cce_get_tile_ptr(cdst);
    __ubuf__ typename TileDataS1::DType *kvaluePtr = (__ubuf__ typename TileDataS1::DType *)__cce_get_tile_ptr(k_value);

    constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(typename TileDataD::DType);
    uint16_t repeatTimes = CeilDivision(srcValidCol, elementsPerRepeat);
    unsigned srcRowStride = TileDataS::RowStride;
    unsigned dstRowStride = TileDataD::RowStride;

    __VEC_SCOPE__
    {
        vector_bool preg_b32 = pset_b32(PAT_ALL);
        vector_bool preg_b16 = pset_b16(PAT_ALL);
        vector_align align_index;
        vector_f32 mask_k;
        vector_s32 index;
        vector_s16 mask_k_b16;

        vci(index, offset, INC_ORDER);
        vector_s32 index_offset;
        vbr(index_offset, 0x00000040);
        sprclr(SPR_AR);

        for (uint16_t i = 0; i < (uint16_t)srcValidRow; ++i) {
            typename TileDataS1::DType k_scalar = *(kvaluePtr + i);
            vbr(mask_k_b16, k_scalar);
            vcvt(mask_k, mask_k_b16, preg_b16, PART_EVEN);

            for (uint16_t j = 0; j < repeatTimes; ++j) {
                vector_f16 score;
                vector_f32 score_f32;
                vlds(score, (__ubuf__ half *)src0Ptr, (i * TileDataS::Cols + j * elementsPerRepeat), UNPK_B16);
                vcvt(score_f32, score, preg_b16, PART_EVEN);

                vector_bool pout_ge;
                vector_s32 sqz_index_out;

                vcmp_gt(pout_ge, score_f32, mask_k, preg_b32);
                vsqz(sqz_index_out, index, pout_ge, MODE_STORED);

                vstur(align_index, (vector_u32)sqz_index_out, (__ubuf__ uint32_t *)dstPtr, POST_UPDATE);
                vadd(index, index, index_offset, preg_b32, MODE_ZEROING);
            }
            vstar(align_index, (__ubuf__ uint32_t *)dstPtr);
            sprsts(SPR_AR, cdstPtr, i * sizeof(typename TileDataC::DType));
            sprclr(SPR_AR);
            dstPtr = dstPtr + TileDataD::Cols;
        }
    }
}

template <typename TileDataD, typename TileDataS, typename TileDataS1, typename TileDataC, CmpMode cmpMode>
__tf__ AICORE void TGather_half_eq(typename TileDataD::TileDType __out__ dst, typename TileDataS::TileDType __in__ src0,
                                   typename TileDataS1::TileDType __in__ k_value, uint32_t offset,
                                   typename TileDataC::TileDType __in__ cdst, unsigned srcValidCol,
                                   unsigned srcValidRow, unsigned dstValidCol, unsigned dstValidRow)
{
    __ubuf__ typename TileDataD::DType *dstPtr = (__ubuf__ typename TileDataD::DType *)__cce_get_tile_ptr(dst);
    __ubuf__ typename TileDataS::DType *src0Ptr = (__ubuf__ typename TileDataS::DType *)__cce_get_tile_ptr(src0);
    __ubuf__ typename TileDataC::DType *cdstPtr = (__ubuf__ typename TileDataC::DType *)__cce_get_tile_ptr(cdst);
    __ubuf__ typename TileDataS1::DType *kvaluePtr = (__ubuf__ typename TileDataS1::DType *)__cce_get_tile_ptr(k_value);

    unsigned TShape1 = TileDataD::Cols;
    using T = typename TileDataD::DType;
    constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(typename TileDataD::DType);
    uint16_t repeatTimes = CeilDivision(srcValidCol, elementsPerRepeat);

    __VEC_SCOPE__
    {
        vector_bool preg_b32 = pset_b32(PAT_ALL);
        vector_bool preg_b32_half = pset_b32(PAT_H);
        vector_bool preg_b16 = pset_b16(PAT_ALL);
        vector_s32 add_offset;
        vector_align align_index;
        vector_f32 mask_k;
        vector_s16 mask_k_b16;

        vector_s32 index;
        vci(index, offset, INC_ORDER);
        vbr(add_offset, 0x00000040);
        sprclr(SPR_AR);

        for (uint16_t i = 0; i < (uint16_t)srcValidRow; ++i) {
            typename TileDataS1::DType k_scalar = *(kvaluePtr + i);
            vbr(mask_k_b16, k_scalar);
            vcvt(mask_k, mask_k_b16, preg_b16, PART_EVEN);

            for (uint16_t j = 0; j < repeatTimes; ++j) {
                vector_f16 score;
                vlds(score, (__ubuf__ half *)src0Ptr, (i * TileDataS::Cols + j * elementsPerRepeat), UNPK_B16);
                vector_f32 score_f32;
                vcvt(score_f32, score, preg_b16, PART_EVEN);

                vector_bool pout_eq;
                vector_s32 sqz_index_out;

                vcmp_eq(pout_eq, score_f32, mask_k, preg_b32);
                vsqz(sqz_index_out, index, pout_eq, MODE_STORED);

                vstur(align_index, (vector_u32)sqz_index_out, (__ubuf__ uint32_t *)dstPtr, POST_UPDATE);
                vadd(index, index, add_offset, preg_b32, MODE_ZEROING);
            }
            vstar(align_index, (__ubuf__ uint32_t *)dstPtr);
            sprsts(SPR_AR, cdstPtr, i * sizeof(typename TileDataC::DType));
            sprclr(SPR_AR);
            dstPtr = dstPtr + TShape1;
        }
    }
}

template <typename TileDataD, typename TileDataS, typename TileDataS1, typename TileDataC, CmpMode cmpMode>
AICORE void TGather_cmp(typename TileDataD::TileDType dst, typename TileDataS::TileDType src0,
                        typename TileDataC::TileDType cdst, typename TileDataS1::TileDType k_value, uint32_t offset,
                        unsigned srcValidCol, unsigned srcValidRow, unsigned dstValidCol, unsigned dstValidRow)
{
    if constexpr (std::is_same_v<typename TileDataS::DType, float> && cmpMode == CmpMode::GT) {
        TGather_float_gt<TileDataD, TileDataS, TileDataS1, TileDataC, cmpMode>(
            dst, src0, k_value, offset, cdst, srcValidCol, srcValidRow, dstValidCol, dstValidRow);
    } else if constexpr (std::is_same_v<typename TileDataS::DType, float> && cmpMode == CmpMode::EQ) {
        TGather_float_eq<TileDataD, TileDataS, TileDataS1, TileDataC, cmpMode>(
            dst, src0, k_value, offset, cdst, srcValidCol, srcValidRow, dstValidCol, dstValidRow);
    } else if constexpr (sizeof(typename TileDataS::DType) == 4 && cmpMode == CmpMode::GT) {
        TGather_b32_gt<TileDataD, TileDataS, TileDataS1, TileDataC, cmpMode>(
            dst, src0, k_value, offset, cdst, srcValidCol, srcValidRow, dstValidCol, dstValidRow);
    } else if constexpr (sizeof(typename TileDataS::DType) == 4 && cmpMode == CmpMode::EQ) {
        TGather_b32_eq<TileDataD, TileDataS, TileDataS1, TileDataC, cmpMode>(
            dst, src0, k_value, offset, cdst, srcValidCol, srcValidRow, dstValidCol, dstValidRow);
    } else if constexpr (std::is_same_v<typename TileDataS::DType, half> && cmpMode == CmpMode::GT) {
        TGather_half_gt<TileDataD, TileDataS, TileDataS1, TileDataC, cmpMode>(
            dst, src0, k_value, offset, cdst, srcValidCol, srcValidRow, dstValidCol, dstValidRow);
    } else if constexpr (std::is_same_v<typename TileDataS::DType, half> && cmpMode == CmpMode::EQ) {
        TGather_half_eq<TileDataD, TileDataS, TileDataS1, TileDataC, cmpMode>(
            dst, src0, k_value, offset, cdst, srcValidCol, srcValidRow, dstValidCol, dstValidRow);
    } else if constexpr (std::is_same_v<typename TileDataS::DType, int16_t> && cmpMode == CmpMode::GT) {
        TGather_b16_gt<TileDataD, TileDataS, TileDataS1, TileDataC, cmpMode>(
            dst, src0, k_value, offset, cdst, srcValidCol, srcValidRow, dstValidCol, dstValidRow);
    } else {
        TGather_b16_eq<TileDataD, TileDataS, TileDataS1, TileDataC, cmpMode>(
            dst, src0, k_value, offset, cdst, srcValidCol, srcValidRow, dstValidCol, dstValidRow);
    }
}

template <typename TileDataD, typename TileDataS, typename TileDataS1, typename TileDataC, typename TileDataTmp,
          CmpMode cmpMode>
PTO_INTERNAL void TGATHER_IMPL(TileDataD &dst, TileDataS &src0, TileDataS1 &k_value, TileDataC &cdst, TileDataTmp &tmp,
                               uint32_t offset)
{
    static_assert(
        std::is_same_v<typename TileDataD::DType, uint32_t> || std::is_same_v<typename TileDataD::DType, int32_t>,
        "Fix: TGATHER Dst data type must be int32_t/uint32_t.");
    static_assert(
        std::is_same_v<typename TileDataS::DType, float> || std::is_same_v<typename TileDataS::DType, uint32_t> ||
            std::is_same_v<typename TileDataS::DType, int32_t> || std::is_same_v<typename TileDataS::DType, uint16_t> ||
            std::is_same_v<typename TileDataS::DType, int16_t> || std::is_same_v<typename TileDataS::DType, half>,
        "Fix: TGATHER Src data type must be int16_t/uint16_t/int32_t/uint32_t/half/float.");
    static_assert((cmpMode == CmpMode::GT || cmpMode == CmpMode::EQ), "Fix: TGATHER only support GT or EQ mode");
    static_assert(
        (std::is_same_v<typename TileDataS1::DType, uint16_t> || std::is_same_v<typename TileDataS1::DType, uint32_t>),
        "Fix: TGATHER k_value tile must be uint16_t/uint32_t");

    unsigned sValidCols = src0.GetValidCol();
    unsigned sValidRows = src0.GetValidRow();
    unsigned dValidCols = dst.GetValidCol();
    unsigned dValidRows = dst.GetValidRow();

    TGather_cmp<TileDataD, TileDataS, TileDataS1, TileDataC, cmpMode>(
        dst.data(), src0.data(), cdst.data(), k_value.data(), offset, sValidCols, sValidRows, dValidCols, dValidRows);
}

} // namespace pto
#endif
