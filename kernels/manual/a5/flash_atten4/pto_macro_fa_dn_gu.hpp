/*
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_MACRO_FA_GU_HPP
#define PTO_MACRO_FA_GU_HPP

#include <pto/common/constants.hpp>
#include <pto/common/utils.hpp>
#include <pto/common/pto_tile.hpp>

namespace pto {

// -----------------------------------------------------------------------------
// FlashAttention "GU" (running update) macro
//
// This implements the numerically-stable streaming update:
//   O = O * exp(max_prev - max_new) + PV_tile
// and on the last tile:
//   O = O / global_sum
//
// Performance notes:
// - Keep O resident in UB across tiles to avoid extra TLOAD/TSTORE.
// - exp_max and global_sum are per-row reduced tiles (shape [S0, 1]) that get broadcast over columns.
// -----------------------------------------------------------------------------

#define USE_MANUAL 1

template <typename reducedTileData, typename svTileData>
__tf__ AICORE inline void pto_macro_fa_gu(svTileData __out__ prev_sv_tile, svTileData __in__ est_sv_tile,
                                          reducedTileData __in__ exp_max)
{
    if constexpr (reducedTileData::BFractal == BLayout::ColMajor) {
        #if USE_MANUAL
            __ubuf__ typename svTileData::DType *prev_sv_tile_Ptr =
                (__ubuf__ typename svTileData::DType *)__cce_get_tile_ptr(prev_sv_tile.data());
            __ubuf__ typename svTileData::DType *est_sv_tile_Ptr =
                (__ubuf__ typename svTileData::DType *)__cce_get_tile_ptr(est_sv_tile.data());
            __ubuf__ typename reducedTileData::DType *exp_max_Ptr =
                (__ubuf__ typename reducedTileData::DType *)__cce_get_tile_ptr(exp_max.data());

            using T = typename svTileData::DType;
            unsigned ubM = svTileData::Rows;
            unsigned ubN = svTileData::Cols;
            unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(T);
            uint16_t repeatTimes = CeilDivision(ubN, elementsPerRepeat);
            constexpr unsigned stride = reducedTileData::Cols;
            constexpr unsigned rowStride = svTileData::RowStride;

            __VEC_SCOPE__ {
                RegTensor<T> vreg0;
                RegTensor<T> vreg1;
                RegTensor<T> vreg2;
                RegTensor<T> vreg3;
                RegTensor<T> vreg_uld;
                MaskReg preg;
                vector_bool preg_b8_all = pset_b8(PAT_ALL);
                vector_align ureg_1;
                constexpr auto distValue =
                    std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<T, DistVST::DIST_NORM>())>();
                for (uint16_t i = 0; i < (uint16_t)(ubM); ++i) {
                    vlds(vreg1, (__ubuf__ T *)(exp_max_Ptr), i * stride, BRC_B32);
                    uint32_t sreg = (uint32_t)(ubN);
                    for (uint16_t j = 0; j < (uint16_t)repeatTimes; ++j) {
                        preg = CreatePredicate<T>(sreg);
                        vlds(vreg0, prev_sv_tile_Ptr, 0, NORM, POST_UPDATE);
                        vlds(vreg3, est_sv_tile_Ptr, elementsPerRepeat, NORM, POST_UPDATE);
                        vmul(vreg2, vreg0, vreg1, preg, MODE_ZEROING);
                        vadd(vreg3, vreg2, vreg3, preg, MODE_ZEROING);
                        vsts(vreg3, prev_sv_tile_Ptr, elementsPerRepeat, distValue, preg, POST_UPDATE);
                    }
                }
            }
        #else
            pto::TROWEXPANDMUL(prev_sv_tile, prev_sv_tile, exp_max);
            pto::TADD(prev_sv_tile, prev_sv_tile, est_sv_tile);
        #endif
    } else {
        using reducedTileData_Col =
            Tile<TileType::Vec, float, reducedTileData::Cols, 1, BLayout::ColMajor, reducedTileData::Cols, 1>;
        reducedTileData_Col exp_max_col;
        TRESHAPE(exp_max_col, exp_max);
        #if USE_MANUAL
            __ubuf__ typename svTileData::DType *prev_sv_tile_Ptr =
                (__ubuf__ typename svTileData::DType *)__cce_get_tile_ptr(prev_sv_tile.data());
            __ubuf__ typename svTileData::DType *est_sv_tile_Ptr =
                (__ubuf__ typename svTileData::DType *)__cce_get_tile_ptr(est_sv_tile.data());
            __ubuf__ typename reducedTileData_Col::DType *exp_max_Ptr =
                (__ubuf__ typename reducedTileData_Col::DType *)__cce_get_tile_ptr(exp_max_col.data());

            using T = typename svTileData::DType;
            unsigned ubM = svTileData::Rows;
            unsigned ubN = svTileData::Cols;
            unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(T);
            uint16_t repeatTimes = CeilDivision(ubN, elementsPerRepeat);
            constexpr unsigned stride = reducedTileData_Col::Cols;
            constexpr unsigned rowStride = svTileData::RowStride;

            __VEC_SCOPE__ {
                RegTensor<T> vreg0;
                RegTensor<T> vreg1;
                RegTensor<T> vreg2;
                RegTensor<T> vreg3;
                RegTensor<T> vreg_uld;
                MaskReg preg;
                vector_bool preg_b8_all = pset_b8(PAT_ALL);
                vector_align ureg_1;
                constexpr auto distValue =
                    std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<T, DistVST::DIST_NORM>())>();
                for (uint16_t i = 0; i < (uint16_t)(ubM); ++i) {
                    vlds(vreg1, (__ubuf__ T *)(exp_max_Ptr), i * stride, BRC_B32);
                    uint32_t sreg = (uint32_t)(ubN);
                    for (uint16_t j = 0; j < (uint16_t)repeatTimes; ++j) {
                        preg = CreatePredicate<T>(sreg);
                        vlds(vreg0, prev_sv_tile_Ptr, 0, NORM, POST_UPDATE);
                        vlds(vreg3, est_sv_tile_Ptr, elementsPerRepeat, NORM, POST_UPDATE);
                        vmul(vreg2, vreg0, vreg1, preg, MODE_ZEROING);
                        vadd(vreg3, vreg2, vreg3, preg, MODE_ZEROING);
                        vsts(vreg3, prev_sv_tile_Ptr, elementsPerRepeat, distValue, preg, POST_UPDATE);
                    }
                }
            }
        #else
            pto::TROWEXPANDMUL(prev_sv_tile, prev_sv_tile, exp_max_col);
            pto::TADD(prev_sv_tile, prev_sv_tile, est_sv_tile);
        #endif
    }
}

template <typename reducedTileData, typename svTileData>
__tf__ AICORE inline void pto_macro_fa_gu_last(svTileData __out__ prev_sv_tile, svTileData __in__ est_sv_tile,
                                               reducedTileData __in__ exp_max, reducedTileData __in__ new_global_sum)
{
    if constexpr (reducedTileData::BFractal == BLayout::ColMajor) {
        #if USE_MANUAL
            __ubuf__ typename svTileData::DType *prev_sv_tile_Ptr =
                (__ubuf__ typename svTileData::DType *)__cce_get_tile_ptr(prev_sv_tile.data());
            __ubuf__ typename svTileData::DType *est_sv_tile_Ptr =
                (__ubuf__ typename svTileData::DType *)__cce_get_tile_ptr(est_sv_tile.data());
            __ubuf__ typename reducedTileData::DType *exp_max_Ptr =
                (__ubuf__ typename reducedTileData::DType *)__cce_get_tile_ptr(exp_max.data());
            __ubuf__ typename reducedTileData::DType *new_global_sum_Ptr =
                (__ubuf__ typename reducedTileData::DType *)__cce_get_tile_ptr(new_global_sum.data());

            using T = typename svTileData::DType;
            unsigned ubM = svTileData::Rows;
            unsigned ubN = svTileData::Cols;
            unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(T);
            uint16_t repeatTimes = CeilDivision(ubN, elementsPerRepeat);
            constexpr unsigned stride = reducedTileData::Cols;
            constexpr unsigned rowStride = svTileData::RowStride;

            __VEC_SCOPE__ {
                RegTensor<T> vreg0;
                RegTensor<T> vreg1;
                RegTensor<T> vreg2;
                RegTensor<T> vreg3;
                RegTensor<T> vreg4;
                RegTensor<T> vreg_uld1;
                RegTensor<T> vreg_uld2;
                MaskReg preg;
                vector_bool preg_b8_all = pset_b8(PAT_ALL);
                vector_align ureg_1;
                vector_align ureg_2;
                constexpr auto distValue =
                    std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<T, DistVST::DIST_NORM>())>();
                for (uint16_t i = 0; i < (uint16_t)(ubM); ++i) {
                    vlds(vreg1, (__ubuf__ T *)(exp_max_Ptr), i * stride, BRC_B32);

                    vlds(vreg4, (__ubuf__ T *)(new_global_sum_Ptr), i * stride, BRC_B32);
                    uint32_t sreg = (uint32_t)(ubN);
                    for (uint16_t j = 0; j < (uint16_t)repeatTimes; ++j) {
                        preg = CreatePredicate<T>(sreg);
                        vlds(vreg0, prev_sv_tile_Ptr, 0, NORM, POST_UPDATE);

                        vlds(vreg3, est_sv_tile_Ptr, elementsPerRepeat, NORM, POST_UPDATE);

                        vmul(vreg2, vreg0, vreg1, preg, MODE_ZEROING);
                        vadd(vreg3, vreg2, vreg3, preg, MODE_ZEROING);

                        vdiv(vreg3, vreg3, vreg4, preg, MODE_ZEROING);
                        vsts(vreg3, prev_sv_tile_Ptr, elementsPerRepeat, distValue, preg, POST_UPDATE);
                    }
                }
            }
        #else
            pto::TROWEXPANDMUL(prev_sv_tile, prev_sv_tile, exp_max);
            pto::TADD(prev_sv_tile, prev_sv_tile, est_sv_tile);
            pto::TROWEXPANDDIV(prev_sv_tile, prev_sv_tile, new_global_sum);
            // pto::TCVT(prev_sv_nd_tile, prev_sv_tile, RoundMode::CAST_RINT);
        #endif
    } else {
        using reducedTileData_Col =
            Tile<TileType::Vec, float, reducedTileData::Cols, 1, BLayout::ColMajor, reducedTileData::Cols, 1>;
        reducedTileData_Col exp_max_col;
        TRESHAPE(exp_max_col, exp_max);
        reducedTileData_Col new_global_sum_col;
        TRESHAPE(new_global_sum_col, new_global_sum);
        #if USE_MANUAL
            __ubuf__ typename svTileData::DType *prev_sv_tile_Ptr =
                (__ubuf__ typename svTileData::DType *)__cce_get_tile_ptr(prev_sv_tile.data());
            __ubuf__ typename svTileData::DType *est_sv_tile_Ptr =
                (__ubuf__ typename svTileData::DType *)__cce_get_tile_ptr(est_sv_tile.data());
            __ubuf__ typename reducedTileData_Col::DType *exp_max_Ptr =
                (__ubuf__ typename reducedTileData_Col::DType *)__cce_get_tile_ptr(exp_max_col.data());
            __ubuf__ typename reducedTileData_Col::DType *new_global_sum_Ptr =
                (__ubuf__ typename reducedTileData_Col::DType *)__cce_get_tile_ptr(new_global_sum_col.data());

            using T = typename svTileData::DType;
            unsigned ubM = svTileData::Rows;
            unsigned ubN = svTileData::Cols;
            unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(T);
            uint16_t repeatTimes = CeilDivision(ubN, elementsPerRepeat);
            constexpr unsigned stride = reducedTileData_Col::Cols;
            constexpr unsigned rowStride = svTileData::RowStride;

            __VEC_SCOPE__ {
                RegTensor<T> vreg0;
                RegTensor<T> vreg1;
                RegTensor<T> vreg2;
                RegTensor<T> vreg3;
                RegTensor<T> vreg4;
                RegTensor<T> vreg_uld1;
                RegTensor<T> vreg_uld2;
                MaskReg preg;
                vector_bool preg_b8_all = pset_b8(PAT_ALL);
                vector_align ureg_1;
                vector_align ureg_2;
                constexpr auto distValue =
                    std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<T, DistVST::DIST_NORM>())>();
                for (uint16_t i = 0; i < (uint16_t)(ubM); ++i) {
                    vlds(vreg1, (__ubuf__ T *)(exp_max_Ptr), i * stride, BRC_B32);

                    vlds(vreg4, (__ubuf__ T *)(new_global_sum_Ptr), i * stride, BRC_B32);
                    uint32_t sreg = (uint32_t)(ubN);
                    for (uint16_t j = 0; j < (uint16_t)repeatTimes; ++j) {
                        preg = CreatePredicate<T>(sreg);
                        vlds(vreg0, prev_sv_tile_Ptr, 0, NORM, POST_UPDATE);

                        vlds(vreg3, est_sv_tile_Ptr, elementsPerRepeat, NORM, POST_UPDATE);

                        vmul(vreg2, vreg0, vreg1, preg, MODE_ZEROING);
                        vadd(vreg3, vreg2, vreg3, preg, MODE_ZEROING);

                        vdiv(vreg3, vreg3, vreg4, preg, MODE_ZEROING);
                        vsts(vreg3, prev_sv_tile_Ptr, elementsPerRepeat, distValue, preg, POST_UPDATE);
                    }
                }
            }
        #else
            pto::TROWEXPANDMUL(prev_sv_tile, prev_sv_tile, exp_max_col);
            pto::TADD(prev_sv_tile, prev_sv_tile, est_sv_tile);
            pto::TROWEXPANDDIV(prev_sv_tile, prev_sv_tile, new_global_sum_col);
        #endif
    }
}

template <typename reducedTileData, typename svTileData>
AICORE inline void pto_macro_fa_gu_single_and_last_tile(svTileData __out__ sv_tile,
                                                        reducedTileData __in__ new_global_sum)
{
    if constexpr (reducedTileData::BFractal == BLayout::ColMajor) {
        pto::TROWEXPANDDIV(sv_tile, sv_tile, new_global_sum);
    } else {
        using reducedTileData_Col =
            Tile<TileType::Vec, float, reducedTileData::Cols, 1, BLayout::ColMajor, reducedTileData::Cols, 1>;
        reducedTileData_Col new_global_sum_col;
        TRESHAPE(new_global_sum_col, new_global_sum);
        pto::TROWEXPANDDIV(sv_tile, sv_tile, new_global_sum_col);
    }
}

} // namespace pto
#endif // TGU_PTO_H
