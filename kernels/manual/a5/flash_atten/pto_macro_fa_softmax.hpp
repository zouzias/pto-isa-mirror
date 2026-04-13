/*
Copyright (c) 2026 Huawei Technologies Co., Ltd.
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

constexpr PTO_INTERNAL float constexpr_sqrt(float x)
{
    if (x <= 0.0f)
        return 0.0f;
    float guess = x;
    for (int i = 0; i < 8; ++i) {
        guess = 0.5f * (guess + x / guess);
    }
    return guess;
}

constexpr AICORE inline float constexpr_inv_sqrt(float x)
{
    return 1.0f / constexpr_sqrt(x);
}

#define USE_MANUAL 0

template <int HEAD_SIZE, bool CAUSAL_MASK, typename ReduceTileD1, typename TileDataD2, typename TileDataS1>
__tf__ AICORE inline void softmax_opt_fa_init_impl(TileDataD2 __out__ x_exp, TileDataS1 __in__ input_x,
                                            ReduceTileD1 __out__ local_max, ReduceTileD1 __out__ local_sum,
                                            ReduceTileD1 __out__ new_global_max, ReduceTileD1 __out__ new_global_sum,
                                            ReduceTileD1 __out__ exp_max, TileDataS1 __out__ tmp_float,
                                            TileDataS1 __out__ p_tile_f32, TileDataS1 triu, int s0_index, int s1_index)
{
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
        __ubuf__ float *src0_ub      = (__ubuf__ float *)input_x_Ptr;
        __ubuf__ float *src0_ub_bkup = (__ubuf__ float *)input_x_Ptr;
        __ubuf__ float *max_ptr_bkup = (__ubuf__ float *)new_global_max_Ptr;
        __ubuf__ float *sum_ptr_bkup = (__ubuf__ float *)new_global_sum_Ptr;
        __ubuf__ half *x_exp_bkup = (__ubuf__ half *)x_exp_Ptr;
    
        vector_f32 vb32_in_even, vb32_in_odd, vb32_in_even_unroll, vb32_in_odd_unroll;
        vector_f32 vb32_max, vb32_max_unroll, vb32_row_max, vb32_row_max_unroll;
        vector_bool preg_b32_all = pset_b32(PAT_ALL);
        vector_bool preg_b16_all = pset_b16(PAT_ALL);
        vector_bool preg_b8_all = pset_b8(PAT_ALL);
        constexpr auto distValue =std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<typename TileDataS1::DType, DistVST::DIST_ONEPT>())>();
        vector_align ureg_max, ureg_max_unroll;

        for (uint16_t i = 0; i < uint16_t(ubM/2) ; ++i) {
            vlds(vb32_in_even,        vb32_in_odd,        src0_ub, 128, DINTLV_B32, POST_UPDATE);
            vlds(vb32_in_even_unroll, vb32_in_odd_unroll, src0_ub, 128, DINTLV_B32, POST_UPDATE);
            vmax(vb32_max,        vb32_in_even,        vb32_in_odd,         preg_b32_all);
            vmax(vb32_max_unroll, vb32_in_even_unroll, vb32_in_odd_unroll,  preg_b32_all);
            vcmax(vb32_row_max,        vb32_max,        preg_b32_all, MODE_ZEROING);
            vcmax(vb32_row_max_unroll, vb32_max_unroll, preg_b32_all, MODE_ZEROING);
            vstus(ureg_max, 1, vb32_row_max,        new_global_max_Ptr, POST_UPDATE);
            vstus(ureg_max, 1, vb32_row_max_unroll, new_global_max_Ptr, POST_UPDATE);
        }
        vstas(ureg_max,  new_global_max_Ptr, 0, POST_UPDATE);
        mem_bar(VST_VLD);

        src0_ub = src0_ub_bkup;
        new_global_max_Ptr = max_ptr_bkup;
        vector_align ureg_sum;
        vector_f32 vb32_max_0, vb32_max_1, vreg0, vreg1, vreg2, vreg3, vb32_add0, vb32_add1, vb32_sum0, vb32_sum1;
        vector_f16 vb16_x_exp_0, vb16_x_exp_1, vb16_x_exp_2, vb16_x_exp_3;
        for (uint16_t j = 0; j < (uint16_t)(ubM / 2); ++j) {
            vlds(vb32_max_0, new_global_max_Ptr, 1, BRC_B32, POST_UPDATE);
            vmuls(vb32_max_0, vb32_max_0, scale, preg_b32_all);

            vlds(vreg0, vreg1, src0_ub, 128, DINTLV_B32, POST_UPDATE);        
            vmuls(vreg0, vreg0, scale, preg_b32_all);
            vmuls(vreg1, vreg1, scale, preg_b32_all);
            
            vlds(vreg2, vreg3, src0_ub, 128, DINTLV_B32, POST_UPDATE);
            vmuls(vreg2, vreg2, scale, preg_b32_all);
            vmuls(vreg3, vreg3, scale, preg_b32_all);

            vlds(vb32_max_1, new_global_max_Ptr, 1, BRC_B32, POST_UPDATE);
            vmuls(vb32_max_1, vb32_max_1, scale, preg_b32_all);

            vexpdif(vreg0, vreg0, vb32_max_0, preg_b32_all, PART_EVEN); // 0 2 4 6 8 10
            vexpdif(vreg1, vreg1, vb32_max_0, preg_b32_all, PART_EVEN); // 1 3 5 7 9 11
            vexpdif(vreg2, vreg2, vb32_max_1, preg_b32_all, PART_EVEN); // 64 66 68 70 72 
            vexpdif(vreg3, vreg3, vb32_max_1, preg_b32_all, PART_EVEN); // 65 67 69 71 73

            vadd(vb32_add0, vreg0, vreg1, preg_b32_all, MODE_ZEROING); 
            vadd(vb32_add1, vreg2, vreg3, preg_b32_all, MODE_ZEROING);

            vcadd(vb32_sum0, vb32_add0, preg_b32_all, MODE_ZEROING);
            vcadd(vb32_sum1, vb32_add1, preg_b32_all, MODE_ZEROING);
            vstus(ureg_sum, 1, vb32_sum0, new_global_sum_Ptr, POST_UPDATE);
            vstus(ureg_sum, 1, vb32_sum1, new_global_sum_Ptr, POST_UPDATE);
            
            vcvt(vb16_x_exp_0, vreg0, preg_b32_all, ROUND_R, RS_ENABLE, PART_EVEN, MODE_ZEROING); 
            vcvt(vb16_x_exp_1, vreg1, preg_b32_all, ROUND_R, RS_ENABLE, PART_ODD, MODE_ZEROING);
            vor(vb16_x_exp_0, vb16_x_exp_0, vb16_x_exp_1, preg_b16_all, MODE_ZEROING);

            vcvt(vb16_x_exp_2, vreg2, preg_b32_all, ROUND_R, RS_ENABLE, PART_EVEN, MODE_ZEROING);
            vcvt(vb16_x_exp_3, vreg3, preg_b32_all, ROUND_R, RS_ENABLE, PART_ODD, MODE_ZEROING);
            vor(vb16_x_exp_2, vb16_x_exp_2, vb16_x_exp_3, preg_b16_all, MODE_ZEROING);
            
            vsts(vb16_x_exp_0, (__ubuf__ half *&) x_exp_Ptr, 128, NORM_B16, preg_b16_all, POST_UPDATE);
            vsts(vb16_x_exp_2, (__ubuf__ half *&) x_exp_Ptr, 128, NORM_B16, preg_b16_all, POST_UPDATE);
        }
        vstas(ureg_sum, new_global_sum_Ptr, 0, POST_UPDATE);
    }
#else
    (void)local_max;
    (void)exp_max;
    (void)local_sum;

    constexpr float scale = constexpr_inv_sqrt(HEAD_SIZE);
    using Tile1D_fp32 = Tile<TileType::Vec, float, 1, TileDataS1::Rows * TileDataS1::Cols, BLayout::RowMajor, 1,
                             TileDataS1::Rows * TileDataS1::Cols>;
    using Tile1D_out = Tile<TileType::Vec, typename TileDataD2::DType, 1, TileDataS1::Rows * TileDataS1::Cols,
                            BLayout::RowMajor, 1, TileDataS1::Rows * TileDataS1::Cols>;
    Tile1D_fp32 p_tile_f32_1d;
    Tile1D_out x_exp_1d;
    if constexpr (CAUSAL_MASK) {
        if (s0_index / TileDataS1::Cols == s1_index / TileDataS1::Cols) {
            constexpr float negInf = -3.40282e+38;
            TTRI<TileDataS1, 1>(triu, 1 + (s0_index % TileDataS1::Cols));
            TMULS(triu, triu, negInf);
            TADD(input_x, input_x, triu);
        }
    }
    // FA2.0 init mode
    TRESHAPE(p_tile_f32_1d, p_tile_f32);
    TRESHAPE(x_exp_1d, x_exp);

    TROWMAX(new_global_max, input_x, tmp_float);
    TROWEXPANDSUB(p_tile_f32, input_x, new_global_max);
    TMULS(p_tile_f32, p_tile_f32, scale);
    TEXP(p_tile_f32, p_tile_f32);
    TROWSUM(new_global_sum, p_tile_f32, tmp_float);
    TCVT(x_exp_1d, p_tile_f32_1d, RoundMode::CAST_ROUND);
#endif
}

template <int HEAD_SIZE, bool CAUSAL_MASK, typename ReduceTileD1, typename TileDataD2, typename TileDataS1>
__tf__ AICORE inline void softmax_opt_fa_not_init_impl(TileDataD2 __out__ x_exp, TileDataS1 __in__ input_x,
                                                ReduceTileD1 __out__ local_max, ReduceTileD1 __out__ local_sum,
                                                ReduceTileD1 __out__ new_global_max,
                                                ReduceTileD1 __out__ new_global_sum, ReduceTileD1 __out__ exp_max,
                                                TileDataS1 __out__ tmp_float, TileDataS1 __out__ p_tile_f32,
                                                TileDataS1 triu, int s0_index, int s1_index)
{
#if USE_MANUAL
    __ubuf__ typename TileDataD2::DType *x_exp_Ptr = (__ubuf__ typename TileDataD2::DType *)__cce_get_tile_ptr(x_exp.data());
    __ubuf__ typename TileDataS1::DType *input_x_Ptr = (__ubuf__ typename TileDataS1::DType *)__cce_get_tile_ptr(input_x.data());
    __ubuf__ typename ReduceTileD1::DType *local_max_Ptr = (__ubuf__ typename ReduceTileD1::DType *)__cce_get_tile_ptr(local_max.data());
    __ubuf__ typename ReduceTileD1::DType *local_sum_Ptr = (__ubuf__ typename ReduceTileD1::DType *)__cce_get_tile_ptr(local_sum.data());
    __ubuf__ typename ReduceTileD1::DType *new_global_max_Ptr = (__ubuf__ typename ReduceTileD1::DType *)__cce_get_tile_ptr(new_global_max.data());
    __ubuf__ typename ReduceTileD1::DType *new_global_sum_Ptr = (__ubuf__ typename ReduceTileD1::DType *)__cce_get_tile_ptr(new_global_sum.data());
    __ubuf__ typename ReduceTileD1::DType *exp_max_Ptr = (__ubuf__ typename ReduceTileD1::DType *)__cce_get_tile_ptr(exp_max.data());

    constexpr float scale = constexpr_inv_sqrt(HEAD_SIZE);
    float keepProb = 1.0;

    unsigned ubM = TileDataD2::Rows;
    unsigned ubN = TileDataD2::Cols;
    unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(typename TileDataS1::DType);

    uint16_t repeatTimes = CeilDivision(ubN, elementsPerRepeat);

    __VEC_SCOPE__ {
        __ubuf__ float *src0_ub         = (__ubuf__ float *)input_x_Ptr;
        __ubuf__ float *src0_ub_bkup    = (__ubuf__ float *)input_x_Ptr;
        __ubuf__ float *local_max_bkup  = (__ubuf__ float *)local_max_Ptr;
        __ubuf__ float *local_sum_bkup  = (__ubuf__ float *)local_sum_Ptr;
        __ubuf__ float *global_max_bkup = (__ubuf__ float *)new_global_max_Ptr;
        __ubuf__ float *global_sum_bkup = (__ubuf__ float *)new_global_sum_Ptr;
        __ubuf__ half *x_exp_bkup       = (__ubuf__ half *)x_exp_Ptr;
    
        vector_f32 vb32_in_even, vb32_in_odd, vb32_in_even_unroll, vb32_in_odd_unroll;
        vector_f32 vb32_max, vb32_max_unroll, vb32_row_max, vb32_row_max_unroll;
        vector_bool preg_b32_all = pset_b32(PAT_ALL);
        vector_bool preg_b16_all = pset_b16(PAT_ALL);
        vector_bool preg_b8_all  = pset_b8(PAT_ALL);
        vector_align ureg_max;

        for (uint16_t i = 0; i < uint16_t(ubM/2) ; ++i) {
            vlds(vb32_in_even,        vb32_in_odd,        src0_ub, 128, DINTLV_B32, POST_UPDATE);
            vlds(vb32_in_even_unroll, vb32_in_odd_unroll, src0_ub, 128, DINTLV_B32, POST_UPDATE);
            vmax(vb32_max,        vb32_in_even,        vb32_in_odd,         preg_b32_all);
            vmax(vb32_max_unroll, vb32_in_even_unroll, vb32_in_odd_unroll,  preg_b32_all);
            vcmax(vb32_row_max,        vb32_max,        preg_b32_all, MODE_ZEROING);
            vcmax(vb32_row_max_unroll, vb32_max_unroll, preg_b32_all, MODE_ZEROING);
            vstus(ureg_max, 1, vb32_row_max,        local_max_Ptr, POST_UPDATE);
            vstus(ureg_max, 1, vb32_row_max_unroll, local_max_Ptr, POST_UPDATE);
        }
        vstas(ureg_max,  local_max_Ptr, 0, POST_UPDATE);
        mem_bar(VST_VLD);
        local_max_Ptr = local_max_bkup; // reset local_max pointer
        vector_f32 vb32_local_max, vb32_global_max, vreg_exp_max;
        
        vlds(vb32_local_max,  local_max_Ptr,      0, NORM);
        vlds(vb32_global_max, new_global_max_Ptr, 0, NORM);
        vmax(vb32_local_max, vb32_local_max,  vb32_global_max, preg_b32_all, MODE_ZEROING);
        vsub(vreg_exp_max,   vb32_global_max, vb32_local_max,  preg_b32_all, MODE_ZEROING);
        vsts(vb32_local_max, new_global_max_Ptr, 0, NORM_B32, preg_b32_all);
        
        vmuls(vreg_exp_max, vreg_exp_max, scale, preg_b32_all, MODE_ZEROING);
        vexp(vreg_exp_max, vreg_exp_max, preg_b32_all, MODE_ZEROING);
        vsts(vreg_exp_max, exp_max_Ptr, 0, NORM_B32, preg_b32_all);
        mem_bar(VST_VLD);
        //-----------------------Calculating local sum and exp--------------------------
        src0_ub = src0_ub_bkup; // reset input pointer
        new_global_max_Ptr = global_max_bkup; // reset new_global_max pointer
        vector_align ureg_sum;
        vector_f32 vb32_max_0, vb32_max_1, vreg0, vreg1, vreg2, vreg3, vb32_add0, vb32_add1, vb32_sum0, vb32_sum1;
        vector_f16 vb16_x_exp_0, vb16_x_exp_1, vb16_x_exp_2, vb16_x_exp_3;
        vector_f32 vb32_local_sum, vb32_global_sum;
        for (uint16_t j = 0; j < (uint16_t)(ubM / 2); ++j) {
            vlds(vb32_max_0, new_global_max_Ptr, 1, BRC_B32, POST_UPDATE);
            vmuls(vb32_max_0, vb32_max_0, scale, preg_b32_all);

            vlds(vreg0, vreg1, src0_ub, 128, DINTLV_B32, POST_UPDATE);        
            vmuls(vreg0, vreg0, scale, preg_b32_all);
            vmuls(vreg1, vreg1, scale, preg_b32_all);
            
            vlds(vreg2, vreg3, src0_ub, 128, DINTLV_B32, POST_UPDATE);
            vmuls(vreg2, vreg2, scale, preg_b32_all);
            vmuls(vreg3, vreg3, scale, preg_b32_all);

            vlds(vb32_max_1, new_global_max_Ptr, 1, BRC_B32, POST_UPDATE);
            vmuls(vb32_max_1, vb32_max_1, scale, preg_b32_all);

            vexpdif(vreg0, vreg0, vb32_max_0, preg_b32_all, PART_EVEN); // 0 2 4 6 8 10
            vexpdif(vreg1, vreg1, vb32_max_0, preg_b32_all, PART_EVEN); // 1 3 5 7 9 11
            vexpdif(vreg2, vreg2, vb32_max_1, preg_b32_all, PART_EVEN); // 64 66 68 70 72 
            vexpdif(vreg3, vreg3, vb32_max_1, preg_b32_all, PART_EVEN); // 65 67 69 71 73
            vadd(vb32_add0, vreg0, vreg1, preg_b32_all, MODE_ZEROING); 
            vadd(vb32_add1, vreg2, vreg3, preg_b32_all, MODE_ZEROING);

            vcadd(vb32_sum0, vb32_add0, preg_b32_all, MODE_ZEROING);
            vcadd(vb32_sum1, vb32_add1, preg_b32_all, MODE_ZEROING);
            vstus(ureg_sum, 1, vb32_sum0, local_sum_Ptr, POST_UPDATE);
            vstus(ureg_sum, 1, vb32_sum1, local_sum_Ptr, POST_UPDATE);
            
            vcvt(vb16_x_exp_0, vreg0, preg_b32_all, ROUND_R, RS_ENABLE, PART_EVEN, MODE_ZEROING); 
            vcvt(vb16_x_exp_1, vreg1, preg_b32_all, ROUND_R, RS_ENABLE, PART_ODD,  MODE_ZEROING);
            vor(vb16_x_exp_0, vb16_x_exp_0, vb16_x_exp_1, preg_b16_all, MODE_ZEROING);

            vcvt(vb16_x_exp_2, vreg2, preg_b32_all, ROUND_R, RS_ENABLE, PART_EVEN, MODE_ZEROING);
            vcvt(vb16_x_exp_3, vreg3, preg_b32_all, ROUND_R, RS_ENABLE, PART_ODD,  MODE_ZEROING);
            vor(vb16_x_exp_2, vb16_x_exp_2, vb16_x_exp_3, preg_b16_all, MODE_ZEROING);
            
            vsts(vb16_x_exp_0, (__ubuf__ half *&) x_exp_Ptr, 128, NORM_B16, preg_b16_all, POST_UPDATE);
            vsts(vb16_x_exp_2, (__ubuf__ half *&) x_exp_Ptr, 128, NORM_B16, preg_b16_all, POST_UPDATE);
        }
        vstas(ureg_sum, local_sum_Ptr, 0, POST_UPDATE);
        mem_bar(VST_VLD);
        //-----------------------Updating Global Sum--------------------------
        local_sum_Ptr = local_sum_bkup; // reset local_sum pointer
        vlds(vb32_local_sum,  local_sum_Ptr,      0, NORM);
        vlds(vb32_global_sum, new_global_sum_Ptr, 0, NORM);
        vmul(vb32_global_sum, vreg_exp_max,    vb32_global_sum, preg_b32_all, MODE_ZEROING);
        vadd(vb32_global_sum, vb32_global_sum, vb32_local_sum,  preg_b32_all, MODE_ZEROING);
        vsts(vb32_global_sum, new_global_sum_Ptr, 0, NORM_B32,  preg_b32_all);
    }
#else
    constexpr float scale = constexpr_inv_sqrt(HEAD_SIZE);

    using ReduceTileD2 = Tile<TileType::Vec, float, 1, ReduceTileD1::Rows, BLayout::RowMajor, 1, ReduceTileD1::Rows>;
    using Tile1D_fp32 = Tile<TileType::Vec, float, 1, TileDataS1::Rows * TileDataS1::Cols, BLayout::RowMajor, 1,
                             TileDataS1::Rows * TileDataS1::Cols>;
    using Tile1D_out = Tile<TileType::Vec, typename TileDataD2::DType, 1, TileDataS1::Rows * TileDataS1::Cols,
                            BLayout::RowMajor, 1, TileDataS1::Rows * TileDataS1::Cols>;

    ReduceTileD2 tmp_shw_local_max;
    ReduceTileD2 tmp_shw_new_global_max;
    ReduceTileD2 tmp_shw_exp_max;
    ReduceTileD2 tmp_shw_new_global_sum;
    ReduceTileD2 tmp_shw_local_sum;
    Tile1D_fp32 p_tile_f32_1d;
    Tile1D_out x_exp_1d;

    if constexpr (CAUSAL_MASK) {
        if (s0_index / TileDataS1::Cols == s1_index / TileDataS1::Cols) {
            constexpr float negInf = -3.40282e+38;
            TTRI<TileDataS1, 1>(triu, 1 + (s0_index % TileDataS1::Cols));
            TMULS(triu, triu, negInf);
            TADD(input_x, input_x, triu);
        }
    }
    // FA2.0 streaming mode (not first tile): update (global_max, global_sum) and rescale old sums.
    TRESHAPE(tmp_shw_local_max, local_max);
    TRESHAPE(tmp_shw_new_global_max, new_global_max);
    TRESHAPE(tmp_shw_exp_max, exp_max);
    TRESHAPE(p_tile_f32_1d, p_tile_f32);
    TRESHAPE(x_exp_1d, x_exp);
    TRESHAPE(tmp_shw_new_global_sum, new_global_sum);
    TRESHAPE(tmp_shw_local_sum, local_sum);

    TROWMAX(local_max, input_x, tmp_float);
    TMAX(tmp_shw_local_max, tmp_shw_local_max, tmp_shw_new_global_max);
    TSUB(tmp_shw_exp_max, tmp_shw_new_global_max, tmp_shw_local_max);
    TMULS(tmp_shw_new_global_max, tmp_shw_local_max, 1.0f); // just copy
    TMULS(tmp_shw_exp_max, tmp_shw_exp_max, scale);
    TEXP(tmp_shw_exp_max, tmp_shw_exp_max);
    TROWEXPANDSUB(p_tile_f32, input_x, local_max);
    TMULS(p_tile_f32, p_tile_f32, scale);
    TEXP(p_tile_f32, p_tile_f32);
    TROWSUM(local_sum, p_tile_f32, tmp_float);
    TCVT(x_exp_1d, p_tile_f32_1d, RoundMode::CAST_ROUND);
    TMUL(tmp_shw_new_global_sum, tmp_shw_exp_max, tmp_shw_new_global_sum);
    TADD(tmp_shw_new_global_sum, tmp_shw_new_global_sum, tmp_shw_local_sum);
#endif
}

template <bool init = false, int HEAD_SIZE, bool CAUSAL_MASK, typename ReduceTileD1, typename TileDataD2,
          typename TileDataS1>
AICORE inline void pto_macro_fa_softmax(TileDataD2 __out__ x_exp, TileDataS1 __in__ input_x,
                                        ReduceTileD1 __out__ local_max, ReduceTileD1 __out__ local_sum,
                                        ReduceTileD1 __in__ new_global_max, ReduceTileD1 __out__ new_global_sum,
                                        ReduceTileD1 __out__ exp_max, TileDataS1 __out__ input_reduce_tmp,
                                        TileDataS1 __out__ p_tile_fp32, TileDataS1 triu, int s0_index, int s1_index)
{
    if (s1_index <= s0_index || !CAUSAL_MASK) {
        if constexpr (init) {
            softmax_opt_fa_init_impl<HEAD_SIZE, CAUSAL_MASK, ReduceTileD1, TileDataD2, TileDataS1>(
                x_exp, input_x, local_max, local_sum, new_global_max, new_global_sum, exp_max, input_reduce_tmp,
                p_tile_fp32, triu, s0_index, s1_index);
        } else {
            softmax_opt_fa_not_init_impl<HEAD_SIZE, CAUSAL_MASK, ReduceTileD1, TileDataD2, TileDataS1>(
                x_exp, input_x, local_max, local_sum, new_global_max, new_global_sum, exp_max, input_reduce_tmp,
                p_tile_fp32, triu, s0_index, s1_index);
        }
    } else if constexpr (CAUSAL_MASK) {
        TMULS(x_exp, x_exp, 0.0);
        TMULS(exp_max, exp_max, 0.0);
        TADDS(exp_max, exp_max, 1.0);
    }
}

} // namespace pto

#endif
