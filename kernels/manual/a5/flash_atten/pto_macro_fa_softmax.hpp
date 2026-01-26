/*
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_MACRO_FA_SOFTMAX_HPP
#define PTO_MACRO_FA_SOFTMAX_HPP

#include <pto/pto-inst.hpp>

namespace pto {

// -----------------------------------------------------------------------------
// FlashAttention streaming softmax (tile-level)
//
// Given one QK tile X (fp32), compute x_exp = exp(scale * (X - new_global_max)).
// This function maintains per-row running state (global_max, global_sum) so that we can
// stream over S1 tiles without materializing the full attention matrix.
//
// Performance notes:
// - Keep intermediate computations in fp32 for numerical stability.
// - The `init` specialization initializes running state for the first S1 tile.
// - The 2D->1D reshape for TCVT is used to avoid layout constraints and keep the cast fast.
// -----------------------------------------------------------------------------

#define USE_MANUAL 0

constexpr PTO_INTERNAL float constexpr_sqrt(float x) {
    if (x <= 0.0f)
        return 0.0f;
    float guess = x;
    for (int i = 0; i < 8; ++i) {
        guess = 0.5f * (guess + x / guess);
    }
    return guess;
}

constexpr AICORE inline float constexpr_inv_sqrt(float x) {
    return 1.0f / constexpr_sqrt(x);
}

template <int HEAD_SIZE, typename ReduceTileD1, typename TileDataD2, typename TileDataS1>
__tf__ AICORE inline void softmax_opt_fa_init_impl(TileDataD2 __out__ x_exp, TileDataS1 __in__ input_x,
    ReduceTileD1 __out__ local_max, ReduceTileD1 __out__ local_sum, ReduceTileD1 __out__ new_global_max,
    ReduceTileD1 __out__ new_global_sum, ReduceTileD1 __out__ exp_max, TileDataS1 __out__ tmp_float,
    TileDataS1 __out__ p_tile_f32) {
    
    #if USE_MANUAL
        __ubuf__ typename TileDataD2::DType *x_exp_Ptr = (__ubuf__ typename TileDataD2::DType *)__cce_get_tile_ptr(x_exp.data());
        __ubuf__ typename TileDataS1::DType *input_x_Ptr = (__ubuf__ typename TileDataS1::DType *)__cce_get_tile_ptr(input_x.data());
        __ubuf__ typename ReduceTileD1::DType *local_max_Ptr = (__ubuf__ typename ReduceTileD1::DType *)__cce_get_tile_ptr(local_max.data());
        __ubuf__ typename ReduceTileD1::DType *local_sum_Ptr = (__ubuf__ typename ReduceTileD1::DType *)__cce_get_tile_ptr(local_sum.data());
        __ubuf__ typename ReduceTileD1::DType *new_global_max_Ptr = (__ubuf__ typename ReduceTileD1::DType *)__cce_get_tile_ptr(new_global_max.data());
        __ubuf__ typename ReduceTileD1::DType *new_global_sum_Ptr = (__ubuf__ typename ReduceTileD1::DType *)__cce_get_tile_ptr(new_global_sum.data());
        __ubuf__ typename ReduceTileD1::DType *exp_max_Ptr = (__ubuf__ typename ReduceTileD1::DType *)__cce_get_tile_ptr(exp_max.data());

        constexpr float scale = constexpr_inv_sqrt(HEAD_SIZE);
        
        unsigned ubM = TileDataD2::Rows;
        unsigned ubN = TileDataD2::Cols;
        unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(typename TileDataS1::DType);
        uint16_t repeatTimes = CeilDivision(ubN, elementsPerRepeat);

        __VEC_SCOPE__{
            vector_f32 src_in1, src_in2;
            vector_f32 max_1a, max_2a;
            vector_f32 sum_1a, sum_1b, sum_2a, sum_2b;
            vector_f32 vreg_exp_max;

            __ubuf__ float *src0_ub = (__ubuf__ float *)input_x_Ptr;

            vector_bool preg_src = pset_b32(PAT_ALL);
            vector_bool preg_b16_all = pset_b16(PAT_ALL);
            vector_bool preg_b8_all = pset_b8(PAT_ALL);
            uint32_t destItems = 1;
            vector_bool preg_reduce = plt_b32(destItems, POST_UPDATE);
            constexpr auto distValue =std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<typename TileDataS1::DType, DistVST::DIST_ONEPT>())>();

            for (uint16_t i = 0; i < uint16_t(ubM) ; ++i) {
                vbr(max_2a, -INFINITY);
                __ubuf__ float *row_ptr = src0_ub + i * TileDataS1::RowStride;
                for (uint16_t iter_n = 0; iter_n < uint16_t(repeatTimes) ; ++iter_n) {
                    vlds(src_in1, row_ptr, elementsPerRepeat, NORM, POST_UPDATE);
                    vcmax(max_1a, src_in1, preg_src, MODE_ZEROING);
                    vmax(max_2a, max_2a, max_1a, preg_src, MODE_ZEROING);
                }
                vsts(max_2a, new_global_max_Ptr+i, 0, distValue, preg_reduce);
            }
            
            mem_bar(VST_VLD);
            unsigned stride = 1;
            vector_align ureg_1;
            vector_f32 vreg_uld;
            vector_f32 vreg0;
            vector_f32 vreg1;
            vector_f32 vreg2;
            vector_f32 vreg3;
            vector_f32 vreg4;
            vector_f16 vreg2_f16;
            vector_f16 vreg4_f16;
            vector_f16 vreg_x_exp_f16_pack, vreg_x_exp_f16_packa;
            unsigned remains = repeatTimes % 2;
            for (uint16_t i = 0; i < uint16_t(ubM) ; ++i) {
                if(remains){
                    vldas(ureg_1, (__ubuf__ float*)(new_global_max_Ptr + i*stride));
                    vldus(vreg_uld, ureg_1, (__ubuf__ float*)(new_global_max_Ptr + i*stride));
                    vdup(vreg1, vreg_uld, preg_b8_all, POS_LOWEST, MODE_ZEROING);
                    __ubuf__ float *row_ptr = src0_ub + i * TileDataS1::RowStride;
                    vbr(sum_2a, 0);

                    for (uint16_t j = 0; j < (uint16_t)(repeatTimes); ++j) {
                        vlds(vreg0, row_ptr,  j * elementsPerRepeat, NORM);

                        vsub(vreg2, vreg0, vreg1, preg_src, MODE_ZEROING);

                        vmuls(vreg2, vreg2, scale, preg_src, MODE_ZEROING);

                        vexp(vreg2, vreg2, preg_src, MODE_ZEROING);

                        vcadd(sum_1a, vreg2, preg_src, MODE_ZEROING);
                        vadd(sum_2a, sum_2a, sum_1a, preg_src, MODE_ZEROING);

                        vcvt(vreg2_f16, vreg2, preg_src, ROUND_R, RS_ENABLE, PART_EVEN, MODE_ZEROING);

                        vsts(vreg2_f16, ((__ubuf__ half *) x_exp_Ptr + i*ubN + j*elementsPerRepeat), 0, PK_B32, preg_b16_all);
                    }
                    vsts(sum_2a, new_global_sum_Ptr+i, 0, distValue, preg_reduce);
                }
                else {
                    vldas(ureg_1, (__ubuf__ float*)(new_global_max_Ptr + i*stride));
                    vldus(vreg_uld, ureg_1, (__ubuf__ float*)(new_global_max_Ptr + i*stride));
                    vdup(vreg1, vreg_uld, preg_b8_all, POS_LOWEST, MODE_ZEROING);
                    __ubuf__ float *row_ptr = src0_ub + i * TileDataS1::RowStride;
                    vbr(sum_2a, 0);
                    vbr(sum_2b, 0);

                    for (uint16_t j = 0; j < (uint16_t)(repeatTimes / 2); ++j) {
                        vlds(vreg0, row_ptr,  (j*2) * elementsPerRepeat, NORM);
                        vlds(vreg3, row_ptr,  (j*2+1) * elementsPerRepeat, NORM);

                        vsub(vreg2, vreg0, vreg1, preg_src, MODE_ZEROING);
                        vsub(vreg4, vreg3, vreg1, preg_src, MODE_ZEROING);

                        vmuls(vreg2, vreg2, scale, preg_src, MODE_ZEROING);
                        vmuls(vreg4, vreg4, scale, preg_src, MODE_ZEROING);

                        vexp(vreg2, vreg2, preg_src, MODE_ZEROING);
                        vexp(vreg4, vreg4, preg_src, MODE_ZEROING);

                        vcadd(sum_1a, vreg2, preg_src, MODE_ZEROING);
                        vcadd(sum_1b, vreg4, preg_src, MODE_ZEROING);
                        vadd(sum_2a, sum_2a, sum_1a, preg_src, MODE_ZEROING);
                        vadd(sum_2b, sum_2b, sum_1b, preg_src, MODE_ZEROING);

                        vcvt(vreg2_f16, vreg2, preg_src, ROUND_R, RS_ENABLE, PART_EVEN, MODE_ZEROING);
                        vcvt(vreg4_f16, vreg4, preg_src, ROUND_R, RS_ENABLE, PART_EVEN, MODE_ZEROING);

                        vdintlv(vreg_x_exp_f16_pack, vreg_x_exp_f16_packa, vreg2_f16, vreg4_f16);
                        vsts(vreg_x_exp_f16_pack, ((__ubuf__ half *) x_exp_Ptr + i*ubN + j*2*elementsPerRepeat), 0, NORM_B32, preg_src);
                    }
                    vadd(sum_2a, sum_2a, sum_2b, preg_src, MODE_ZEROING);
                    vsts(sum_2a, new_global_sum_Ptr+i, 0, distValue, preg_reduce);
                }
            }
        }
    #else
        (void)local_max;
        (void)exp_max;
        (void)local_sum;

        constexpr float scale = constexpr_inv_sqrt(HEAD_SIZE);
        using Tile1D_fp32 = Tile<TileType::Vec, float, 1, TileDataS1::Rows*TileDataS1::Cols, BLayout::RowMajor, 1, TileDataS1::Rows*TileDataS1::Cols>;
        using Tile1D_out = Tile<TileType::Vec, typename TileDataD2::DType, 1, TileDataS1::Rows*TileDataS1::Cols, BLayout::RowMajor, 1, TileDataS1::Rows*TileDataS1::Cols>;
        Tile1D_fp32 p_tile_f32_1d;
        Tile1D_out x_exp_1d;

        // FA2.0 init mode
        TROWMAX(new_global_max, input_x, tmp_float);
        TROWEXPANDSUB(tmp_float, input_x, new_global_max);
        TMULS(tmp_float, tmp_float, scale);
        TEXP(p_tile_f32, tmp_float);
        TROWSUM(new_global_sum, p_tile_f32, tmp_float);
        //TCVT(x_exp, p_tile_f32, RoundMode::CAST_ROUND);
        TRESHAPE(p_tile_f32_1d, p_tile_f32);
        TRESHAPE(x_exp_1d, x_exp);   
        TCVT(x_exp_1d, p_tile_f32_1d, RoundMode::CAST_ROUND);
    #endif
}

template <int HEAD_SIZE, typename ReduceTileD1, typename TileDataD2, typename TileDataS1>
__tf__ AICORE inline void softmax_opt_fa_not_init_impl(TileDataD2 __out__ x_exp, TileDataS1 __in__ input_x,
    ReduceTileD1 __out__ local_max, ReduceTileD1 __out__ local_sum, ReduceTileD1 __out__ new_global_max,
    ReduceTileD1 __out__ new_global_sum, ReduceTileD1 __out__ exp_max, TileDataS1 __out__ tmp_float,
    TileDataS1 __out__ p_tile_f32) {

    #ifndef USE_MANUAL
        __ubuf__ typename TileDataD2::DType *x_exp_Ptr = (__ubuf__ typename TileDataD2::DType *)__cce_get_tile_ptr(x_exp.data());
        __ubuf__ typename TileDataS1::DType *input_x_Ptr = (__ubuf__ typename TileDataS1::DType *)__cce_get_tile_ptr(input_x.data());
        __ubuf__ typename ReduceTileD1::DType *local_max_Ptr = (__ubuf__ typename ReduceTileD1::DType *)__cce_get_tile_ptr(local_max.data());
        __ubuf__ typename ReduceTileD1::DType *local_sum_Ptr = (__ubuf__ typename ReduceTileD1::DType *)__cce_get_tile_ptr(local_sum.data());
        __ubuf__ typename ReduceTileD1::DType *new_global_max_Ptr = (__ubuf__ typename ReduceTileD1::DType *)__cce_get_tile_ptr(new_global_max.data());
        __ubuf__ typename ReduceTileD1::DType *new_global_sum_Ptr = (__ubuf__ typename ReduceTileD1::DType *)__cce_get_tile_ptr(new_global_sum.data());
        __ubuf__ typename ReduceTileD1::DType *exp_max_Ptr = (__ubuf__ typename ReduceTileD1::DType *)__cce_get_tile_ptr(exp_max.data());

        constexpr float scale  = constexpr_inv_sqrt(HEAD_SIZE);

        unsigned ubM = TileDataD2::Rows;
        unsigned ubN = TileDataD2::Cols;
        unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(typename TileDataS1::DType);
        uint16_t repeatTimes = CeilDivision(ubN, elementsPerRepeat);

        __VEC_SCOPE__{
            vector_f32 src_in1, src_in2;
            vector_f32 max_1a, max_2a;
            vector_f32 sum_1a, sum_1b, sum_2a, sum_2b;
            vector_f32 vreg_exp_max;

            __ubuf__ float *src0_ub = (__ubuf__ float *)input_x_Ptr;

            vector_bool preg_src = pset_b32(PAT_ALL);
            vector_bool preg_b16_all = pset_b16(PAT_ALL);
            vector_bool preg_b16_half = pset_b16(PAT_VL64);
            vector_bool preg_b8_all = pset_b8(PAT_ALL);
            uint32_t destItems = 1;
            vector_bool preg_reduce = plt_b32(destItems, POST_UPDATE);
            constexpr auto distValue =std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<typename TileDataS1::DType, DistVST::DIST_ONEPT>())>();

            for (uint16_t i = 0; i < uint16_t(ubM) ; ++i) {
                vbr(max_2a, -INFINITY);
                __ubuf__ float *row_ptr = src0_ub + i * TileDataS1::RowStride;
                for (uint16_t iter_n = 0; iter_n < uint16_t(repeatTimes) ; ++iter_n) {
                    vlds(src_in1, row_ptr, elementsPerRepeat, NORM, POST_UPDATE);
                    vcmax(max_1a, src_in1, preg_src, MODE_ZEROING);
                    vmax(max_2a, max_2a, max_1a, preg_src, MODE_ZEROING);
                }
                vsts(max_2a, local_max_Ptr+i, 0, distValue, preg_reduce);
            }

            mem_bar(VST_VLD);
            unsigned stride = 1;
            vector_align ureg_1;
            vector_f32 vreg_uld;
            vector_f32 vreg0;
            vector_f32 vreg1;
            vector_f32 vreg2;
            vector_f32 vreg3;
            vector_f32 vreg4;
            vector_f16 vreg2_f16;
            vector_f16 vreg4_f16;
            vector_f16 vreg_x_exp_f16_pack, vreg_x_exp_f16_packa;
            unsigned remains = repeatTimes % 2;
            
            vlds(max_2a, local_max_Ptr, 0, NORM);
            vlds(src_in1, new_global_max_Ptr, 0, NORM);
            vmax(max_2a, max_2a, src_in1, preg_src, MODE_ZEROING);
            vsub(vreg_exp_max, src_in1, max_2a, preg_b16_half, MODE_ZEROING);
            vsts(max_2a, new_global_max_Ptr, 0, NORM_B32, preg_b16_half);

            vmuls(vreg_exp_max, vreg_exp_max, scale, preg_b16_half, MODE_ZEROING);
            vexp(vreg_exp_max, vreg_exp_max, preg_b16_half, MODE_ZEROING);
            vsts(vreg_exp_max, exp_max_Ptr, 0, NORM_B32, preg_src);
            mem_bar(VST_VLD);

            for (uint16_t i = 0; i < uint16_t(ubM) ; ++i) {
                if(remains){
                    vldas(ureg_1, (__ubuf__ float*)(new_global_max_Ptr + i*stride));
                    vldus(vreg_uld, ureg_1, (__ubuf__ float*)(new_global_max_Ptr + i*stride));
                    vdup(vreg1, vreg_uld, preg_b8_all, POS_LOWEST, MODE_ZEROING);
                    __ubuf__ float *row_ptr = src0_ub + i * TileDataS1::RowStride;
                    vbr(sum_2a, 0);

                    for (uint16_t j = 0; j < (uint16_t)(repeatTimes); ++j) {
                        vlds(vreg0, row_ptr,  j * elementsPerRepeat, NORM);

                        vsub(vreg2, vreg0, vreg1, preg_src, MODE_ZEROING);

                        vmuls(vreg2, vreg2, scale, preg_src, MODE_ZEROING);
                        vexp(vreg2, vreg2, preg_src, MODE_ZEROING);

                        vcadd(sum_1a, vreg2, preg_src, MODE_ZEROING);
                        vadd(sum_2a, sum_2a, sum_1a, preg_src, MODE_ZEROING);

                        vcvt(vreg2_f16, vreg2, preg_src, ROUND_R, RS_ENABLE, PART_EVEN, MODE_ZEROING);

                        vsts(vreg2_f16, ((__ubuf__ half *) x_exp_Ptr + i*ubN + j*elementsPerRepeat), 0, PK_B32, preg_b16_all);
                    }
                    vsts(sum_2a, local_sum_Ptr+i, 0, distValue, preg_reduce);
                }
                else {
                    vldas(ureg_1, (__ubuf__ float*)(new_global_max_Ptr + i*stride));
                    vldus(vreg_uld, ureg_1, (__ubuf__ float*)(new_global_max_Ptr + i*stride));
                    vdup(vreg1, vreg_uld, preg_b8_all, POS_LOWEST, MODE_ZEROING);
                    __ubuf__ float *row_ptr = src0_ub + i * TileDataS1::RowStride;
                    vbr(sum_2a, 0);
                    vbr(sum_2b, 0);

                    for (uint16_t j = 0; j < (uint16_t)(repeatTimes / 2); ++j) {
                        vlds(vreg0, row_ptr,  (j*2) * elementsPerRepeat, NORM);
                        vlds(vreg3, row_ptr,  (j*2+1) * elementsPerRepeat, NORM);

                        vsub(vreg2, vreg0, vreg1, preg_src, MODE_ZEROING);
                        vsub(vreg4, vreg3, vreg1, preg_src, MODE_ZEROING);

                        vmuls(vreg2, vreg2, scale, preg_src, MODE_ZEROING);
                        vmuls(vreg4, vreg4, scale, preg_src, MODE_ZEROING);
                        vexp(vreg2, vreg2, preg_src, MODE_ZEROING);
                        vexp(vreg4, vreg4, preg_src, MODE_ZEROING);

                        vcadd(sum_1a, vreg2, preg_src, MODE_ZEROING);
                        vcadd(sum_1b, vreg4, preg_src, MODE_ZEROING);
                        vadd(sum_2a, sum_2a, sum_1a, preg_src, MODE_ZEROING);
                        vadd(sum_2b, sum_2b, sum_1b, preg_src, MODE_ZEROING);

                        vcvt(vreg2_f16, vreg2, preg_src, ROUND_R, RS_ENABLE, PART_EVEN, MODE_ZEROING);
                        vcvt(vreg4_f16, vreg4, preg_src, ROUND_R, RS_ENABLE, PART_EVEN, MODE_ZEROING);

                        vdintlv(vreg_x_exp_f16_pack, vreg_x_exp_f16_packa, vreg2_f16, vreg4_f16);
                        vsts(vreg_x_exp_f16_pack, ((__ubuf__ half *) x_exp_Ptr + i*ubN + j*2*elementsPerRepeat), 0, NORM_B32, preg_src);
                    }
                    vadd(sum_2a, sum_2a, sum_2b, preg_src, MODE_ZEROING);
                    vsts(sum_2a, local_sum_Ptr+i, 0, distValue, preg_reduce);
                }
            }
            mem_bar(VST_VLD);
            vlds(src_in1, local_sum_Ptr, 0, NORM);
            vlds(src_in2, new_global_sum_Ptr, 0, NORM);
            vmul(src_in2, vreg_exp_max, src_in2, preg_src, MODE_ZEROING);
            vadd(src_in2, src_in2, src_in1, preg_src, MODE_ZEROING);
            vsts(src_in2, new_global_sum_Ptr, 0, NORM_B32, preg_src);
        }
    #else
        constexpr float scale  = constexpr_inv_sqrt(HEAD_SIZE);

        using ReduceTileD2 = Tile<TileType::Vec, float, 1, ReduceTileD1::Rows, BLayout::RowMajor, 1, ReduceTileD1::Rows>;
        using Tile1D_fp32 = Tile<TileType::Vec, float, 1, TileDataS1::Rows*TileDataS1::Cols, BLayout::RowMajor, 1, TileDataS1::Rows*TileDataS1::Cols>;
        using Tile1D_out = Tile<TileType::Vec, typename TileDataD2::DType, 1, TileDataS1::Rows*TileDataS1::Cols, BLayout::RowMajor, 1, TileDataS1::Rows*TileDataS1::Cols>;
        
        ReduceTileD2 tmp_shw_local_max;
        ReduceTileD2 tmp_shw_new_global_max;
        ReduceTileD2 tmp_shw_exp_max;
        ReduceTileD2 tmp_shw_new_global_sum;
        ReduceTileD2 tmp_shw_local_sum;
        Tile1D_fp32 p_tile_f32_1d;
        Tile1D_out x_exp_1d;

        // FA2.0 streaming mode (not first tile): update (global_max, global_sum) and rescale old sums.
        TROWMAX(local_max, input_x, tmp_float);
        TRESHAPE(tmp_shw_local_max, local_max);
        TRESHAPE(tmp_shw_new_global_max, new_global_max);
        TMAX(tmp_shw_local_max, tmp_shw_local_max, tmp_shw_new_global_max);
        TRESHAPE(tmp_shw_exp_max, exp_max);
        TSUB(tmp_shw_exp_max, tmp_shw_new_global_max, tmp_shw_local_max);

        TMULS(tmp_shw_new_global_max, tmp_shw_local_max, 1.0f); // just copy
        TROWEXPANDSUB(tmp_float, input_x, local_max);
        TMULS(tmp_shw_exp_max, tmp_shw_exp_max, scale);
        TMULS(tmp_float, tmp_float, scale);
        TEXP(tmp_shw_exp_max, tmp_shw_exp_max);
        TRESHAPE(tmp_shw_exp_max, exp_max);
        TEXP(p_tile_f32, tmp_float);
        TRESHAPE(tmp_shw_exp_max, exp_max);
        //TCVT(x_exp, p_tile_f32, RoundMode::CAST_ROUND);
        TRESHAPE(p_tile_f32_1d, p_tile_f32);
        TRESHAPE(x_exp_1d, x_exp);    
        TCVT(x_exp_1d, p_tile_f32_1d, RoundMode::CAST_ROUND);
        TRESHAPE(tmp_shw_new_global_sum, new_global_sum);
        TMUL(tmp_shw_new_global_sum, tmp_shw_exp_max, tmp_shw_new_global_sum);
        TROWSUM(local_sum, p_tile_f32, tmp_float);
        TRESHAPE(tmp_shw_local_sum, local_sum);
        TADD(tmp_shw_new_global_sum, tmp_shw_new_global_sum, tmp_shw_local_sum);
    #endif
}

template <bool init = false, int HEAD_SIZE, typename ReduceTileD1, typename TileDataD2, typename TileDataS1>
AICORE inline void pto_macro_fa_softmax(TileDataD2 __out__ x_exp, TileDataS1 __in__ input_x,
    ReduceTileD1 __out__ local_max, ReduceTileD1 __out__ local_sum, ReduceTileD1 __in__ new_global_max,
        ReduceTileD1 __out__ new_global_sum, ReduceTileD1 __out__ exp_max, TileDataS1 __out__ input_reduce_tmp,
        TileDataS1 __out__ p_tile_fp32) {
    if constexpr (init) {
        softmax_opt_fa_init_impl<HEAD_SIZE, ReduceTileD1, TileDataD2, TileDataS1>(
                x_exp, input_x, local_max, local_sum, new_global_max, new_global_sum, exp_max, input_reduce_tmp, p_tile_fp32);
    } else {
        softmax_opt_fa_not_init_impl<HEAD_SIZE, ReduceTileD1, TileDataD2, TileDataS1>(
                x_exp, input_x, local_max, local_sum, new_global_max, new_global_sum, exp_max, input_reduce_tmp, p_tile_fp32);
    }
}

} // namespace pto

#endif
