/*
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_MACRO_FA_SOFTMAX_DN_HPP
#define PTO_MACRO_FA_SOFTMAX_DN_HPP

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
__tf__ AICORE inline void softmax_opt_fa_dn_init_impl(TileDataD2 __out__ x_exp, TileDataS1 __in__ input_x,
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

    unsigned ubM = TileDataD2::Cols;
    unsigned ubN = TileDataD2::Rows;

    __VEC_SCOPE__{
        vector_f32 vreg_x_even;
        vector_f32 vreg_x_odd;
        vector_f32 vreg_x;
        vector_f32 vreg_max;
        vector_f16 vreg_x_exp_f16;
        vector_f32 vreg_x_sum_even;
        vector_f32 vreg_x_sum_odd;
        vector_f32 vreg_x_sum_1_even;
        vector_f32 vreg_x_sum_1_odd;
        vector_f32 vreg_x_sum0;
        vector_f32 vreg_x_sum1;
        vector_align ureg_198;
        int32_t sreg_197 = 0;
        vector_f32 vreg_x_sub_even;
        vector_f32 vreg_x_sub_odd;
        vector_f16 vreg_x_exp_even_f16;
        vector_f16 vreg_x_exp_odd_f16;
        vector_f32 vreg_x_exp_even;
        vector_f32 vreg_x_exp_odd;
        vector_f32 vreg_x_exp_1_even;
        vector_f32 vreg_x_exp_1_odd;
        vector_f32 vreg_x_exp_even_a;
        vector_f32 vreg_x_exp_odd_a;
        vector_f32 vreg_x_f32_a;
        vector_f32 vreg_x_f32_b;
        vector_f32 vreg_x_exp_even_b;
        vector_f32 vreg_x_exp_odd_b;
        vector_f32 vreg_x_exp_even_1;
        vector_f32 vreg_x_exp_odd_1;
        vector_f16 vreg_x_exp_even_f16_1;
        vector_f16 vreg_x_exp_odd_f16_1;
        vector_f32 vreg_x_1;
        vector_f32 vreg_x_f32_1_a;
        vector_f32 vreg_x_f32_1_b;
        vector_f16 vreg_x_exp_f16_1;
        vector_f32 vreg_x_sum_even_1;
        vector_f32 vreg_x_sum_odd_1;
        vector_f32 vreg_x_sum2;
        vector_f32 vreg_x_sum3;
        vector_f32 vreg_x_max_f32_a;
        vector_f32 vreg_x_max_f32_b;
        vector_f16 vreg_x_exp_f16_pack;
        vector_f16 vreg_x_exp_f16_1_pack;
        vector_f16 vreg_x_exp_f16_packa;
        vector_f16 vreg_x_exp_f16_1_packa;
        vector_u16 vreg_x_exp_u16_pack;
        vector_u16 vreg_x_exp_u16_1_pack;
        vector_u16 vreg_x_exp_u16_packa;
        vector_u16 vreg_x_exp_u16_1_packa;
        vector_bool preg_100;
        vector_bool preg_101;
        vector_bool preg_134;
        vector_bool preg_135;
        vector_bool preg_136;
        preg_134 = pset_b8(PAT_ALL);
        preg_135 = pset_b32(PAT_ALL);
        uint32_t sreg_92 = 0;
        sreg_92 = (uint32_t) 128ULL;
        preg_136 = plt_b16(sreg_92, POST_UPDATE);
        vector_bool p_sel;
        p_sel = pset_b32(PAT_ALL);
        vector_bool preg_108;
        vector_f32 src_00a, src_01a, src_02a, src_03a;
        vector_f32 src_00b, src_01b, src_02b, src_03b;
        vector_f32 src_10a, src_11a, src_12a, src_13a;
        vector_f32 src_10b, src_11b, src_12b, src_13b;
        vector_f32 max_0a, max_1a, max_2a, max_3a;
        vector_f32 max_0b, max_1b, max_2b, max_3b;

        __ubuf__ float *src0_ub = (__ubuf__ float *)input_x_Ptr;
        __ubuf__ half *x_exp_1 = (__ubuf__ half *)x_exp_Ptr + (ubN/2 *16 / 2);
        __ubuf__ float *src0_ub1 = src0_ub + 128;
        __ubuf__ float *src0_ub2 = src0_ub + 256;
        __ubuf__ float *src0_ub3 = src0_ub + 384;

        __ubuf__ float *src0_ub_unroll = src0_ub + 64;
        __ubuf__ float *src0_ub1_unroll = src0_ub1 + 64;
        __ubuf__ float *src0_ub2_unroll = src0_ub2 + 64;
        __ubuf__ float *src0_ub3_unroll = src0_ub3 + 64;

        vbr(max_0a, 0);
        vbr(max_0b, 0);
        vbr(max_1a, 0);
        vbr(max_1b, 0);
        vbr(max_2a, 0);
        vbr(max_2b, 0);
        vbr(max_3a, 0);
        vbr(max_3b, 0);

        preg_108 = pset_b16(PAT_ALL);
        // 128,64 -> 1,64

        for (uint16_t iter_m = 0; iter_m < uint16_t(ubN / 8) ; ++iter_m) {
            vlds(src_00a, src0_ub,        512, NORM, POST_UPDATE);
            vlds(src_00b, src0_ub_unroll, 512, NORM, POST_UPDATE);
            vmax(max_0a, max_0a, src_00a, preg_108);
            vmax(max_0b, max_0b, src_00b, preg_108);

            vlds(src_01a, src0_ub1,      512, NORM, POST_UPDATE);
            vlds(src_01b, src0_ub1_unroll, 512, NORM, POST_UPDATE);
            vmax(max_1a, max_1a, src_01a, preg_108);
            vmax(max_1b, max_1b, src_01b, preg_108);
            
            vlds(src_02a, src0_ub2,      512, NORM, POST_UPDATE);
            vlds(src_02b, src0_ub2_unroll, 512, NORM, POST_UPDATE);
            vmax(max_2a, max_2a, src_02a, preg_108);
            vmax(max_2b, max_2b, src_02b, preg_108);
            
            vlds(src_03a, src0_ub3,      512, NORM, POST_UPDATE);
            vlds(src_03b, src0_ub3_unroll, 512, NORM, POST_UPDATE);
            vmax(max_3a, max_3a, src_03a, preg_108);
            vmax(max_3b, max_3b, src_03b, preg_108);
        }

        vmax(max_0a, max_0a, max_1a, preg_108);
        vmax(max_0b, max_0b, max_1b, preg_108);
        vmax(max_2a, max_2a, max_3a, preg_108);
        vmax(max_2b, max_2b, max_3b, preg_108);
        vmax(max_0a, max_0a, max_2a, preg_108);
        vmax(max_0b, max_0b, max_2b, preg_108);
        vmax(max_0a, max_0a, max_0b, preg_108);
        vsts(max_0a, (__ubuf__ float *)new_global_max_Ptr, 0, NORM_B16, preg_108);
        vmuls(max_0a, max_0a, scale, preg_108);

        vdup(vreg_x_sum_even, 0, preg_134, MODE_ZEROING);
        vdup(vreg_x_sum_odd, 0, preg_134, MODE_ZEROING);
        vdup(vreg_x_sum_1_even, 0, preg_134, MODE_ZEROING);
        vdup(vreg_x_sum_1_odd, 0, preg_134, MODE_ZEROING);

        for (uint16_t i0 = 0; i0 < uint16_t(ubN / 4) ; ++i0) { //128,64
            // vector_address areg_x_1 = vag_b32(128);
            vld(vreg_x_f32_a, input_x_Ptr, vag_b32(128), NORM);
            vld(vreg_x_f32_b, ((__ubuf__ float *) input_x_Ptr + 64), vag_b32(128), NORM);
            vld(vreg_x_f32_1_a, ((__ubuf__ float *) input_x_Ptr + ubN*ubM/2), vag_b32(128), NORM);
            vld(vreg_x_f32_1_b, ((__ubuf__ float *) input_x_Ptr + ubN*ubM/2 + 64), vag_b32(128), NORM);

            vmuls(vreg_x_f32_a, vreg_x_f32_a, scale, preg_108);
            vmuls(vreg_x_f32_b, vreg_x_f32_b, scale, preg_108);
            vexpdif(vreg_x_exp_even, vreg_x_f32_a, max_0a, preg_134, PART_ODD);
            vexpdif(vreg_x_exp_odd, vreg_x_f32_b, max_0a, preg_134, PART_ODD);

            vmuls(vreg_x_f32_1_a, vreg_x_f32_1_a, scale, preg_108);
            vmuls(vreg_x_f32_1_b, vreg_x_f32_1_b, scale, preg_108);
            vexpdif(vreg_x_exp_even_1, vreg_x_f32_1_a, max_0a, preg_134, PART_ODD);
            vexpdif(vreg_x_exp_odd_1, vreg_x_f32_1_b, max_0a, preg_134, PART_ODD);

            vmulscvt(vreg_x_exp_even_f16, vreg_x_exp_even, 1.0f, preg_100, PART_EVEN);
            vmulscvt(vreg_x_exp_odd_f16, vreg_x_exp_odd, 1.0f, preg_101, PART_EVEN);
            vmulscvt(vreg_x_exp_even_f16_1, vreg_x_exp_even_1, 1.0f, preg_135, PART_EVEN);
            vmulscvt(vreg_x_exp_odd_f16_1, vreg_x_exp_odd_1, 1.0f, preg_136, PART_EVEN);

            vdintlv(vreg_x_exp_f16_pack, vreg_x_exp_f16_packa, vreg_x_exp_even_f16, vreg_x_exp_odd_f16);
            vdintlv(vreg_x_exp_f16_1_pack, vreg_x_exp_f16_1_packa, vreg_x_exp_even_f16_1, vreg_x_exp_odd_f16_1);

            vsts(vreg_x_exp_f16_pack, ((__ubuf__ half *) x_exp_Ptr + i0*128), 0, NORM_B16, preg_108);
            vsts(vreg_x_exp_f16_1_pack, ((__ubuf__ half *) x_exp_Ptr + ubN*ubM/2 + i0*128), 0, NORM_B16, preg_108);

            vadd(vreg_x_sum_even, vreg_x_exp_even, vreg_x_sum_even, preg_134, MODE_ZEROING);
            vadd(vreg_x_sum_odd, vreg_x_exp_odd, vreg_x_sum_odd, preg_134, MODE_ZEROING);

            vadd(vreg_x_sum_1_even, vreg_x_exp_even_1, vreg_x_sum_1_even, preg_134, MODE_ZEROING);
            vadd(vreg_x_sum_1_odd, vreg_x_exp_odd_1, vreg_x_sum_1_odd, preg_134, MODE_ZEROING);
        }

        vadd(vreg_x_sum0, vreg_x_sum_odd, vreg_x_sum_even, preg_134, MODE_ZEROING);
        vadd(vreg_x_sum1, vreg_x_sum_1_odd, vreg_x_sum_1_even, preg_134, MODE_ZEROING);
        vadd(vreg_x_sum0, vreg_x_sum0, vreg_x_sum1, preg_134, MODE_ZEROING);
        vsts(vreg_x_sum0, ((__ubuf__ float *&) local_sum_Ptr), 0, NORM_B32, preg_134);
        
        vsts(vreg_x_sum0, ((__ubuf__ float *&) new_global_sum_Ptr), 0, NORM_B32, preg_134);
    }
#else
    (void)local_max;
    (void)exp_max;
    (void)local_sum;

    constexpr float scale = constexpr_inv_sqrt(HEAD_SIZE);

    if constexpr (CAUSAL_MASK) {
        if (s0_index / TileDataS1::Rows == s1_index / TileDataS1::Rows) {
            constexpr float negInf = -3.40282e+38;
            TTRI<TileDataS1, 0>(triu, (s0_index % TileDataS1::Rows));
            TMULS(triu, triu, negInf);
            TADD(input_x, input_x, triu);
        }
    }

    TCOLMAX(new_global_max, input_x);
    TCOLEXPANDSUB(input_x, input_x, new_global_max);
    TMULS(input_x, input_x, scale);
    TEXP(input_x, input_x);
    TCOLSUM(new_global_sum, input_x, tmp_float, false);
    TCVT(x_exp, input_x, RoundMode::CAST_ROUND);
#endif
}

template <int HEAD_SIZE, bool CAUSAL_MASK, typename ReduceTileD1, typename TileDataD2, typename TileDataS1>
__tf__ AICORE inline void softmax_opt_fa_dn_not_init_impl(
    TileDataD2 __out__ x_exp, TileDataS1 __in__ input_x, ReduceTileD1 __out__ local_max, ReduceTileD1 __out__ local_sum,
    ReduceTileD1 __out__ new_global_max, ReduceTileD1 __out__ new_global_sum, ReduceTileD1 __out__ exp_max,
    TileDataS1 __out__ tmp_float, TileDataS1 __out__ p_tile_f32, TileDataS1 triu, int s0_index, int s1_index)
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

    unsigned ubM = TileDataD2::Cols;
    unsigned ubN = TileDataD2::Rows;

    __VEC_SCOPE__{
        vector_f32 vreg_x_even;
        vector_f32 vreg_x_odd;
        vector_f32 vreg_x;
        vector_f32 vreg_max;
        vector_f16 vreg_x_exp_f16;
        vector_f32 vreg_x_sum_even;
        vector_f32 vreg_x_sum_odd;
        vector_f32 vreg_x_sum_1_even;
        vector_f32 vreg_x_sum_1_odd;
        vector_f32 vreg_x_sum0;
        vector_f32 vreg_x_sum1;
        vector_align ureg_198;
        int32_t sreg_197 = 0;
        vector_f32 vreg_x_sub_even;
        vector_f32 vreg_x_sub_odd;
        vector_f16 vreg_x_exp_even_f16;
        vector_f16 vreg_x_exp_odd_f16;
        vector_f32 vreg_x_exp_even;
        vector_f32 vreg_x_exp_odd;
        vector_f32 vreg_x_exp_1_even;
        vector_f32 vreg_x_exp_1_odd;
        vector_f32 vreg_x_exp_even_a;
        vector_f32 vreg_x_exp_odd_a;
        vector_f32 vreg_x_f32_a;
        vector_f32 vreg_x_f32_b;
        vector_f32 vreg_x_exp_even_b;
        vector_f32 vreg_x_exp_odd_b;
        vector_f32 vreg_x_exp_even_1;
        vector_f32 vreg_x_exp_odd_1;
        vector_f16 vreg_x_exp_even_f16_1;
        vector_f16 vreg_x_exp_odd_f16_1;
        vector_f32 vreg_x_1;
        vector_f32 vreg_x_f32_1_a;
        vector_f32 vreg_x_f32_1_b;
        vector_f16 vreg_x_exp_f16_1;
        vector_f32 vreg_x_sum_even_1;
        vector_f32 vreg_x_sum_odd_1;
        vector_f32 vreg_x_sum2;
        vector_f32 vreg_x_sum3;
        vector_f32 vreg_x_max_f32_a;
        vector_f32 vreg_x_max_f32_b;
        vector_f16 vreg_x_exp_f16_pack;
        vector_f16 vreg_x_exp_f16_1_pack;
        vector_f16 vreg_x_exp_f16_packa;
        vector_f16 vreg_x_exp_f16_1_packa;
        vector_u16 vreg_x_exp_u16_pack;
        vector_u16 vreg_x_exp_u16_1_pack;
        vector_u16 vreg_x_exp_u16_packa;
        vector_u16 vreg_x_exp_u16_1_packa;
        vector_bool preg_100;
        vector_bool preg_101;
        vector_bool preg_134;
        vector_bool preg_135;
        vector_bool preg_136;
        preg_134 = pset_b8(PAT_ALL);
        preg_135 = pset_b32(PAT_ALL);
        uint32_t sreg_92 = 0;
        sreg_92 = (uint32_t) 128ULL;
        preg_136 = plt_b16(sreg_92, POST_UPDATE);
        vector_bool p_sel;
        p_sel = pset_b32(PAT_ALL);
        vector_bool preg_108;
        vector_f32 src_00a, src_01a, src_02a, src_03a;
        vector_f32 src_00b, src_01b, src_02b, src_03b;
        vector_f32 src_10a, src_11a, src_12a, src_13a;
        vector_f32 src_10b, src_11b, src_12b, src_13b;
        vector_f32 max_0a, max_1a, max_2a, max_3a;
        vector_f32 max_0b, max_1b, max_2b, max_3b;

        __ubuf__ float *src0_ub = (__ubuf__ float *)input_x_Ptr;
        __ubuf__ half *x_exp_1 = (__ubuf__ half *)x_exp_Ptr + (ubN/2 *16 / 2);
        __ubuf__ float *src0_ub1 = src0_ub + 128;
        __ubuf__ float *src0_ub2 = src0_ub + 256;
        __ubuf__ float *src0_ub3 = src0_ub + 384;

        __ubuf__ float *src0_ub_unroll = src0_ub + 64;
        __ubuf__ float *src0_ub1_unroll = src0_ub1 + 64;
        __ubuf__ float *src0_ub2_unroll = src0_ub2 + 64;
        __ubuf__ float *src0_ub3_unroll = src0_ub3 + 64;

        vbr(max_0a, 0);
        vbr(max_0b, 0);
        vbr(max_1a, 0);
        vbr(max_1b, 0);
        vbr(max_2a, 0);
        vbr(max_2b, 0);
        vbr(max_3a, 0);
        vbr(max_3b, 0);

        preg_108 = pset_b16(PAT_ALL);
        // 128,64 -> 1,64

        for (uint16_t iter_m = 0; iter_m < uint16_t(ubN / 8) ; ++iter_m) {
            vlds(src_00a, src0_ub,        512, NORM, POST_UPDATE);
            vlds(src_00b, src0_ub_unroll, 512, NORM, POST_UPDATE);
            vmax(max_0a, max_0a, src_00a, preg_108);
            vmax(max_0b, max_0b, src_00b, preg_108);

            vlds(src_01a, src0_ub1,      512, NORM, POST_UPDATE);
            vlds(src_01b, src0_ub1_unroll, 512, NORM, POST_UPDATE);
            vmax(max_1a, max_1a, src_01a, preg_108);
            vmax(max_1b, max_1b, src_01b, preg_108);
            
            vlds(src_02a, src0_ub2,      512, NORM, POST_UPDATE);
            vlds(src_02b, src0_ub2_unroll, 512, NORM, POST_UPDATE);
            vmax(max_2a, max_2a, src_02a, preg_108);
            vmax(max_2b, max_2b, src_02b, preg_108);
            
            vlds(src_03a, src0_ub3,      512, NORM, POST_UPDATE);
            vlds(src_03b, src0_ub3_unroll, 512, NORM, POST_UPDATE);
            vmax(max_3a, max_3a, src_03a, preg_108);
            vmax(max_3b, max_3b, src_03b, preg_108);
        }

        vlds(vreg_x_max_f32_b, new_global_max_Ptr, 0, NORM);
        vmax(max_0a, max_0a, max_1a, preg_108);
        vmax(max_0b, max_0b, max_1b, preg_108);
        vmax(max_2a, max_2a, max_3a, preg_108);
        vmax(max_2b, max_2b, max_3b, preg_108);
        vmax(max_0a, max_0a, max_2a, preg_108);
        vmax(max_0b, max_0b, max_2b, preg_108);
        vmax(max_0a, max_0a, max_0b, preg_108);

        vmax(max_0a, max_0a, vreg_x_max_f32_b, preg_108);

        vsts(max_0a, new_global_max_Ptr, 0, NORM_B16, preg_108);  //just copy

        vmuls(max_0a, max_0a, scale, preg_108);
        vmuls(vreg_x_max_f32_b, vreg_x_max_f32_b, scale, preg_108);
        vexpdif(vreg_x_max_f32_b, vreg_x_max_f32_b, max_0a, preg_134, PART_ODD);

        vsts(vreg_x_max_f32_b, exp_max_Ptr, 0, NORM_B16, preg_108);     //Store input max in local_max
        
        vdup(vreg_x_sum_even, 0, preg_134, MODE_ZEROING);
        vdup(vreg_x_sum_odd, 0, preg_134, MODE_ZEROING);
        vdup(vreg_x_sum_1_even, 0, preg_134, MODE_ZEROING);
        vdup(vreg_x_sum_1_odd, 0, preg_134, MODE_ZEROING);

        for (uint16_t i0 = 0; i0 < uint16_t(ubN / 4) ; ++i0) { //128,64
            vld(vreg_x_f32_a, input_x_Ptr, vag_b32(128), NORM);
            vld(vreg_x_f32_b, ((__ubuf__ float *) input_x_Ptr + 64), vag_b32(128), NORM);
            vld(vreg_x_f32_1_a, ((__ubuf__ float *) input_x_Ptr + ubN*ubM/2), vag_b32(128), NORM);
            vld(vreg_x_f32_1_b, ((__ubuf__ float *) input_x_Ptr + ubN*ubM/2 + 64), vag_b32(128), NORM);

            vmuls(vreg_x_f32_a, vreg_x_f32_a, scale, preg_108);
            vmuls(vreg_x_f32_b, vreg_x_f32_b, scale, preg_108);
            vexpdif(vreg_x_exp_even, vreg_x_f32_a, max_0a, preg_134, PART_ODD);
            vexpdif(vreg_x_exp_odd, vreg_x_f32_b, max_0a, preg_134, PART_ODD);

            vmuls(vreg_x_f32_1_a, vreg_x_f32_1_a, scale, preg_108);
            vmuls(vreg_x_f32_1_b, vreg_x_f32_1_b, scale, preg_108);
            vexpdif(vreg_x_exp_even_1, vreg_x_f32_1_a, max_0a, preg_134, PART_ODD);
            vexpdif(vreg_x_exp_odd_1, vreg_x_f32_1_b, max_0a, preg_134, PART_ODD);

            vmulscvt(vreg_x_exp_even_f16, vreg_x_exp_even, 1.0f, preg_100, PART_EVEN);
            vmulscvt(vreg_x_exp_odd_f16, vreg_x_exp_odd, 1.0f, preg_101, PART_EVEN);
            vmulscvt(vreg_x_exp_even_f16_1, vreg_x_exp_even_1, 1.0f, preg_135, PART_EVEN);
            vmulscvt(vreg_x_exp_odd_f16_1, vreg_x_exp_odd_1, 1.0f, preg_136, PART_EVEN);

            vdintlv(vreg_x_exp_f16_pack, vreg_x_exp_f16_packa, vreg_x_exp_even_f16, vreg_x_exp_odd_f16);
            vdintlv(vreg_x_exp_f16_1_pack, vreg_x_exp_f16_1_packa, vreg_x_exp_even_f16_1, vreg_x_exp_odd_f16_1);

            vsts(vreg_x_exp_f16_pack, ((__ubuf__ half *) x_exp_Ptr + i0*128), 0, NORM_B16, preg_108);
            vsts(vreg_x_exp_f16_1_pack, ((__ubuf__ half *) x_exp_Ptr + ubN*ubM/2 + i0*128), 0, NORM_B16, preg_108);

            vadd(vreg_x_sum_even, vreg_x_exp_even, vreg_x_sum_even, preg_134, MODE_ZEROING);
            vadd(vreg_x_sum_odd, vreg_x_exp_odd, vreg_x_sum_odd, preg_134, MODE_ZEROING);

            vadd(vreg_x_sum_1_even, vreg_x_exp_even_1, vreg_x_sum_1_even, preg_134, MODE_ZEROING);
            vadd(vreg_x_sum_1_odd, vreg_x_exp_odd_1, vreg_x_sum_1_odd, preg_134, MODE_ZEROING);
        }

        vadd(vreg_x_sum0, vreg_x_sum_odd, vreg_x_sum_even, preg_134, MODE_ZEROING);
        vadd(vreg_x_sum1, vreg_x_sum_1_odd, vreg_x_sum_1_even, preg_134, MODE_ZEROING);
        vadd(vreg_x_sum0, vreg_x_sum0, vreg_x_sum1, preg_134, MODE_ZEROING);
        vsts(vreg_x_sum0, ((__ubuf__ float *&) local_sum_Ptr), 0, NORM_B32, preg_134);  //no need?

        // mem_bar(VST_VLD);
        vector_f32 vreg_max0;
        vector_f32 vreg_max1;
        vector_f32 vreg_exp_max;
        vector_f32 vreg_exp;
        vector_f32 vreg_l0;
        vector_f32 vreg_l1;
        for (uint16_t ii = 0; ii < 1; ++ii) { // 128/64=2
            vlds(vreg_exp_max, ((__ubuf__ float *) exp_max_Ptr + ii*64), 0, NORM);
            vlds(vreg_l0, ((__ubuf__ float *) new_global_sum_Ptr + ii*64), 0, NORM);
            vmul(vreg_l0, vreg_exp_max, vreg_l0, preg_134, MODE_ZEROING);
            vadd(vreg_l0, vreg_l0, vreg_x_sum0, preg_134, MODE_ZEROING);
            vsts(vreg_l0, ((__ubuf__ float *) new_global_sum_Ptr+ ii*64), 0, NORM_B32, preg_134);
        }
    }
#else
    constexpr float scale = constexpr_inv_sqrt(HEAD_SIZE);

    if constexpr (CAUSAL_MASK) {
        if (s0_index / TileDataS1::Rows == s1_index / TileDataS1::Rows) {
            constexpr float negInf = -3.40282e+38;
            TTRI<TileDataS1, 0>(triu, (s0_index % TileDataS1::Rows));
            TMULS(triu, triu, negInf);
            TADD(input_x, input_x, triu);
        }
    }

    // FA2.0 streaming mode (not first tile): update (global_max, global_sum) and rescale old sums.

    TCOLMAX(local_max, input_x);
    TMAX(local_max, local_max, new_global_max);
    TSUB(exp_max, new_global_max, local_max);
    TMULS(new_global_max, local_max, 1.0f); // just copy
    TMULS(exp_max, exp_max, scale);
    TEXP(exp_max, exp_max);
    TCOLEXPANDSUB(input_x, input_x, local_max);
    TMULS(input_x, input_x, scale);
    TEXP(input_x, input_x);
    TCOLSUM(local_sum, input_x, tmp_float, false);
    TCVT(x_exp, input_x, RoundMode::CAST_ROUND);
    TMUL(new_global_sum, exp_max, new_global_sum);
    TADD(new_global_sum, new_global_sum, local_sum);
#endif
}

template <bool init = false, int HEAD_SIZE, bool CAUSAL_MASK, typename ReduceTileD1, typename TileDataD2,
          typename TileDataS1>
AICORE inline void pto_macro_fa_softmax_dn(TileDataD2 __out__ x_exp, TileDataS1 __in__ input_x,
                                           ReduceTileD1 __out__ local_max, ReduceTileD1 __out__ local_sum,
                                           ReduceTileD1 __in__ new_global_max, ReduceTileD1 __out__ new_global_sum,
                                           ReduceTileD1 __out__ exp_max, TileDataS1 __out__ input_reduce_tmp,
                                           TileDataS1 __out__ p_tile_fp32, TileDataS1 triu, int s0_index, int s1_index)
{
    if (s1_index <= s0_index || !CAUSAL_MASK) {
        if constexpr (init) {
            softmax_opt_fa_dn_init_impl<HEAD_SIZE, CAUSAL_MASK, ReduceTileD1, TileDataD2, TileDataS1>(
                x_exp, input_x, local_max, local_sum, new_global_max, new_global_sum, exp_max, input_reduce_tmp,
                p_tile_fp32, triu, s0_index, s1_index);
        } else {
            softmax_opt_fa_dn_not_init_impl<HEAD_SIZE, CAUSAL_MASK, ReduceTileD1, TileDataD2, TileDataS1>(
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