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

#define USE_MANUAL 0

template <typename reducedTileData, typename svTileData>
__tf__ AICORE inline void pto_macro_fa_gu(
    svTileData __out__ prev_sv_tile, svTileData __in__ est_sv_tile, reducedTileData __in__ exp_max) {

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

        __VEC_SCOPE__
        {
            RegTensor<T> vreg0;
            RegTensor<T> vreg1;
            RegTensor<T> vreg2;
            RegTensor<T> vreg3;
            RegTensor<T> vreg_uld;
            MaskReg preg;
            vector_bool preg_b8_all = pset_b8(PAT_ALL);
            vector_align ureg_1;
            constexpr auto distValue = std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<T, DistVST::DIST_NORM>())>();
            for (uint16_t i = 0; i < (uint16_t)(ubM); ++i) {
                vldas(ureg_1, (__ubuf__ T*)(exp_max_Ptr + i*stride));
                vldus(vreg_uld, ureg_1, (__ubuf__ T*)(exp_max_Ptr + i*stride));
                vdup(vreg1, vreg_uld, preg_b8_all, POS_LOWEST, MODE_ZEROING);
                uint32_t sreg = (uint32_t)(ubN);
                for (uint16_t j = 0; j < (uint16_t)repeatTimes; ++j) {
                    preg = CreatePredicate<T>(sreg);
                    vlds(vreg0, prev_sv_tile_Ptr,  0, NORM, POST_UPDATE);
                    vmul(vreg2, vreg0, vreg1, preg, MODE_ZEROING);
                    vlds(vreg3, est_sv_tile_Ptr, elementsPerRepeat, NORM, POST_UPDATE);
                    vadd(vreg3, vreg2, vreg3, preg, MODE_ZEROING);
                    vsts(vreg3, prev_sv_tile_Ptr, elementsPerRepeat, distValue, preg, POST_UPDATE);
                }
            }
        }
    #else
        pto::TROWEXPANDMUL(prev_sv_tile, prev_sv_tile, exp_max);
        pto::TADD(prev_sv_tile, prev_sv_tile, est_sv_tile);
    #endif
}



template <typename reducedTileData, typename svTileData>
__tf__ AICORE inline void pto_macro_fa_gu_last(svTileData __out__ prev_sv_tile, svTileData __in__ est_sv_tile,
    reducedTileData __in__ exp_max, reducedTileData __in__ new_global_sum) {

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

        __VEC_SCOPE__
        {
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
            constexpr auto distValue = std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<T, DistVST::DIST_NORM>())>();
            for (uint16_t i = 0; i < (uint16_t)(ubM); ++i) {
                vldas(ureg_1, (__ubuf__ T*)(exp_max_Ptr + i*stride));
                vldus(vreg_uld1, ureg_1, (__ubuf__ T*)(exp_max_Ptr + i*stride));
                vdup(vreg1, vreg_uld1, preg_b8_all, POS_LOWEST, MODE_ZEROING);
                vldas(ureg_2, (__ubuf__ T*)(new_global_sum_Ptr + i*stride));
                vldus(vreg_uld2, ureg_2, (__ubuf__ T*)(new_global_sum_Ptr + i*stride));
                vdup(vreg4, vreg_uld2, preg_b8_all, POS_LOWEST, MODE_ZEROING);
                uint32_t sreg = (uint32_t)(ubN);
                for (uint16_t j = 0; j < (uint16_t)repeatTimes; ++j) {
                    preg = CreatePredicate<T>(sreg);
                    vlds(vreg0, prev_sv_tile_Ptr,  0, NORM, POST_UPDATE);
                    vmul(vreg2, vreg0, vreg1, preg, MODE_ZEROING);
                    vlds(vreg3, est_sv_tile_Ptr,  elementsPerRepeat, NORM, POST_UPDATE);
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
}

} // namespace pto
#endif // TGU_PTO_H