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

template <int HEAD_SIZE, typename ReduceTileD1, typename TileDataD2, typename TileDataS1>
AICORE inline void softmax_opt_fa_dn_init_impl(TileDataD2 __out__ x_exp, TileDataS1 __in__ input_x,
    ReduceTileD1 __out__ local_max, ReduceTileD1 __out__ local_sum, ReduceTileD1 __out__ new_global_max,
    ReduceTileD1 __out__ new_global_sum, ReduceTileD1 __out__ exp_max, TileDataS1 __out__ tmp_float,
    TileDataS1 __out__ p_tile_f32) {
    (void)local_max;
    (void)exp_max;
    (void)local_sum;

    constexpr float scale = constexpr_inv_sqrt(HEAD_SIZE);

    TCOLMAX(new_global_max, input_x);
    TCOLEXPANDSUB(input_x, input_x, new_global_max);
    TMULS(input_x, input_x, scale);
    TEXP(input_x, input_x);
    TCOLSUM(new_global_sum, input_x, tmp_float, false);
    // TMULS(input_x, input_x, 1.0f);
    TCVT(x_exp, input_x, RoundMode::CAST_ROUND);
}

template <int HEAD_SIZE, typename ReduceTileD1, typename TileDataD2, typename TileDataS1>
AICORE inline void softmax_opt_fa_dn_not_init_impl(TileDataD2 __out__ x_exp, TileDataS1 __in__ input_x,
    ReduceTileD1 __out__ local_max, ReduceTileD1 __out__ local_sum, ReduceTileD1 __out__ new_global_max,
    ReduceTileD1 __out__ new_global_sum, ReduceTileD1 __out__ exp_max, TileDataS1 __out__ tmp_float,
    TileDataS1 __out__ p_tile_f32) {

    constexpr float scale  = constexpr_inv_sqrt(HEAD_SIZE);

    // FA2.0 streaming mode (not first tile): update (global_max, global_sum) and rescale old sums.

    TCOLMAX(local_max, input_x);
    TMAX(local_max, local_max, new_global_max);
    TSUB(exp_max, new_global_max, local_max);
    TMULS(new_global_max, local_max, 1.0f);  //just copy
    TMULS(exp_max, exp_max, scale);
    TEXP(exp_max, exp_max);
    TCOLEXPANDSUB(input_x, input_x, local_max);
    TMULS(input_x, input_x, scale);
    TEXP(input_x, input_x);
    TCOLSUM(local_sum, input_x, tmp_float, false);
    TCVT(x_exp, input_x, RoundMode::CAST_ROUND);
    TMUL(new_global_sum, exp_max, new_global_sum);
    TADD(new_global_sum, new_global_sum, local_sum);
}

/*
// p_out from 4096 all 0, but maybe not code problem
template <int HEAD_SIZE, typename ReduceTileD1, typename TileDataD2, typename TileDataS1>
__tf__ AICORE inline void softmax_opt_fa_dn_init_impl(TileDataD2 __out__ x_exp, TileDataS1 __in__ input_x,
    ReduceTileD1 __out__ local_max, ReduceTileD1 __out__ local_sum, ReduceTileD1 __out__ new_global_max,
    ReduceTileD1 __out__ new_global_sum, ReduceTileD1 __out__ exp_max, TileDataS1 __out__ tmp_float,
    TileDataS1 __out__ p_tile_f32) {

    __ubuf__ typename TileDataD2::DType *x_exp_Ptr = (__ubuf__ typename TileDataD2::DType *)__cce_get_tile_ptr(x_exp.data());
    __ubuf__ typename TileDataS1::DType *input_x_Ptr = (__ubuf__ typename TileDataS1::DType *)__cce_get_tile_ptr(input_x.data());
    __ubuf__ typename ReduceTileD1::DType *local_max_Ptr = (__ubuf__ typename ReduceTileD1::DType *)__cce_get_tile_ptr(local_max.data());
    __ubuf__ typename ReduceTileD1::DType *local_sum_Ptr = (__ubuf__ typename ReduceTileD1::DType *)__cce_get_tile_ptr(local_sum.data());
    __ubuf__ typename ReduceTileD1::DType *new_global_max_Ptr = (__ubuf__ typename ReduceTileD1::DType *)__cce_get_tile_ptr(new_global_max.data());
    __ubuf__ typename ReduceTileD1::DType *new_global_sum_Ptr = (__ubuf__ typename ReduceTileD1::DType *)__cce_get_tile_ptr(new_global_sum.data());
    __ubuf__ typename ReduceTileD1::DType *exp_max_Ptr = (__ubuf__ typename ReduceTileD1::DType *)__cce_get_tile_ptr(exp_max.data());

    constexpr float scale  = constexpr_inv_sqrt(HEAD_SIZE);
    unsigned ubM = TileDataD2::Rows;    //Cube_S0 / VEC_CORES / kTileFactor = 64
    unsigned ubN = TileDataD2::Cols;    //64

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

        // vector_address areg_0_1, areg_x_1;
        // vector_address areg_0_1 = vag_b16(512);
        // vector_address areg_x_1 = vag_b32(128);

        __ubuf__ float *src0_ub = (__ubuf__ float *)input_x_Ptr;
        __ubuf__ half *x_exp_1 = (__ubuf__ half *)x_exp_Ptr + (ubN/2 *16 / 2);
        __ubuf__ float *src0_ub1 = src0_ub + 128;//input x
        __ubuf__ float *src0_ub2 = src0_ub + 256;
        __ubuf__ float *src0_ub3 = src0_ub + 384;
        vbr(max_0a, 0);
        vbr(max_0b, 0);
        vbr(max_1a, 0);
        vbr(max_1b, 0);
        vbr(max_2a, 0);
        vbr(max_2b, 0);
        vbr(max_3a, 0);
        vbr(max_3b, 0);

        preg_108 = pset_b16(PAT_ALL);

        for (uint16_t iter_m = 0; iter_m < uint16_t(ubN / 8) ; ++iter_m) {
            // vector_address areg_0_1 = vag_b16(512);
            vld(src_00a, src0_ub + iter_m * 512 / 2, vag_b16(512), NORM);//
            vmax(max_0a, max_0a, src_00a, preg_108);
            vld(src_00b, src0_ub + iter_m * 512 / 2 + 64, vag_b16(512), NORM);//
            vmax(max_0b, max_0b, src_00b, preg_108);
            vld(src_01a, src0_ub1 + iter_m * 512 / 2, vag_b16(512), NORM);
            vmax(max_1a, max_1a, src_01a, preg_108);
            vld(src_01b, src0_ub1 + iter_m * 512 / 2 + 64, vag_b16(512), NORM);
            vmax(max_1b, max_1b, src_01b, preg_108);
            vld(src_02a, src0_ub2 + iter_m * 512 / 2, vag_b16(512), NORM);
            vmax(max_2a, max_2a, src_02a, preg_108);
            vld(src_02b, src0_ub2 + iter_m * 512 / 2 + 64, vag_b16(512), NORM);
            vmax(max_2b, max_2b, src_02b, preg_108);
            vld(src_03a, src0_ub3 + iter_m * 512 / 2, vag_b16(512), NORM);
            vmax(max_3a, max_3a, src_03a, preg_108);
            vld(src_03b, src0_ub3 + iter_m * 512 / 2 + 64, vag_b16(512), NORM);
            vmax(max_3b, max_3b, src_03b, preg_108);
        }

        vmax(max_0a, max_0a, max_1a, preg_108);
        vmax(max_0b, max_0b, max_1b, preg_108);
        vmax(max_2a, max_2a, max_3a, preg_108);
        vmax(max_2b, max_2b, max_3b, preg_108);
        vmax(max_0a, max_0a, max_2a, preg_108);
        vmax(max_0b, max_0b, max_2b, preg_108);
        vmax(max_0a, max_0a, max_0b, preg_108);
        // vsts(max_0a, (__ubuf__ float *)local_max_Ptr, 0, NORM_B16, preg_108);     //no need?
        vsts(max_0a, (__ubuf__ float *)new_global_max_Ptr, 0, NORM_B16, preg_108);
        vmuls(max_0a, max_0a, scale, preg_108);

        mem_bar(VST_VLD);
        
        vdup(vreg_x_sum_even, 0, preg_134, MODE_ZEROING);
        vdup(vreg_x_sum_odd, 0, preg_134, MODE_ZEROING);
        vdup(vreg_x_sum_1_even, 0, preg_134, MODE_ZEROING);
        vdup(vreg_x_sum_1_odd, 0, preg_134, MODE_ZEROING);

        for (uint16_t i0 = 0; i0 < uint16_t(ubN / 4) ; ++i0) {
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
        // vsts(vreg_x_sum0, ((__ubuf__ float *&) local_sum_Ptr), 0, NORM_B32, preg_134);    //no need?
        vsts(vreg_x_sum0, ((__ubuf__ float *&) new_global_sum_Ptr), 0, NORM_B32, preg_134);

    }
}
*/

/*
template <int HEAD_SIZE, typename ReduceTileD1, typename TileDataD2, typename TileDataS1>
__tf__ AICORE inline void softmax_opt_fa_dn_not_init_impl(TileDataD2 __out__ x_exp, TileDataS1 __in__ input_x,
    ReduceTileD1 __out__ local_max, ReduceTileD1 __out__ local_sum, ReduceTileD1 __out__ new_global_max,
    ReduceTileD1 __out__ new_global_sum, ReduceTileD1 __out__ exp_max, TileDataS1 __out__ tmp_float,
    TileDataS1 __out__ p_tile_f32) {

    __ubuf__ typename TileDataD2::DType *x_exp_Ptr = (__ubuf__ typename TileDataD2::DType *)__cce_get_tile_ptr(x_exp.data());
    __ubuf__ typename TileDataS1::DType *input_x_Ptr = (__ubuf__ typename TileDataS1::DType *)__cce_get_tile_ptr(input_x.data());
    __ubuf__ typename ReduceTileD1::DType *local_max_Ptr = (__ubuf__ typename ReduceTileD1::DType *)__cce_get_tile_ptr(local_max.data());
    __ubuf__ typename ReduceTileD1::DType *local_sum_Ptr = (__ubuf__ typename ReduceTileD1::DType *)__cce_get_tile_ptr(local_sum.data());
    __ubuf__ typename ReduceTileD1::DType *new_global_max_Ptr = (__ubuf__ typename ReduceTileD1::DType *)__cce_get_tile_ptr(new_global_max.data());
    __ubuf__ typename ReduceTileD1::DType *new_global_sum_Ptr = (__ubuf__ typename ReduceTileD1::DType *)__cce_get_tile_ptr(new_global_sum.data());
    __ubuf__ typename ReduceTileD1::DType *exp_max_Ptr = (__ubuf__ typename ReduceTileD1::DType *)__cce_get_tile_ptr(exp_max.data());

    constexpr float scale  = constexpr_inv_sqrt(HEAD_SIZE);
    unsigned ubM = TileDataD2::Rows;    //Cube_S0 / VEC_CORES / kTileFactor = 64
    unsigned ubN = TileDataD2::Cols;    //64

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
        __ubuf__ float *src0_ub1 = src0_ub + 128;//input x
        __ubuf__ float *src0_ub2 = src0_ub + 256;
        __ubuf__ float *src0_ub3 = src0_ub + 384;
        vbr(max_0a, 0);
        vbr(max_0b, 0);
        vbr(max_1a, 0);
        vbr(max_1b, 0);
        vbr(max_2a, 0);
        vbr(max_2b, 0);
        vbr(max_3a, 0);
        vbr(max_3b, 0);

        preg_108 = pset_b16(PAT_ALL);

        for (uint16_t iter_m = 0; iter_m < uint16_t(ubN / 8) ; ++iter_m) {
            vld(src_00a, src0_ub + iter_m * 512 / 2, vag_b16(512), NORM);
            vmax(max_0a, max_0a, src_00a, preg_108);
            vld(src_00b, src0_ub + iter_m * 512 / 2 + 64, vag_b16(512), NORM);
            vmax(max_0b, max_0b, src_00b, preg_108);
            vld(src_01a, src0_ub1 + iter_m * 512 / 2, vag_b16(512), NORM);
            vmax(max_1a, max_1a, src_01a, preg_108);
            vld(src_01b, src0_ub1 + iter_m * 512 / 2 + 64, vag_b16(512), NORM);
            vmax(max_1b, max_1b, src_01b, preg_108);
            vld(src_02a, src0_ub2 + iter_m * 512 / 2, vag_b16(512), NORM);
            vmax(max_2a, max_2a, src_02a, preg_108);
            vld(src_02b, src0_ub2 + iter_m * 512 / 2 + 64, vag_b16(512), NORM);
            vmax(max_2b, max_2b, src_02b, preg_108);
            vld(src_03a, src0_ub3 + iter_m * 512 / 2, vag_b16(512), NORM);
            vmax(max_3a, max_3a, src_03a, preg_108);
            vld(src_03b, src0_ub3 + iter_m * 512 / 2 + 64, vag_b16(512), NORM);
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
        vsts(max_0a, local_max_Ptr, 0, NORM_B16, preg_108);
        vmuls(max_0a, max_0a, scale, preg_108);
        vmax(max_0a, max_0a, vreg_x_max_f32_b, preg_108);
        vexpdif(vreg_x_max_f32_b, vreg_x_max_f32_b, max_0a, preg_134, PART_ODD);
        vsts(vreg_x_max_f32_b, exp_max_Ptr, 0, NORM_B16, preg_108);
        

        mem_bar(VST_VLD);
        vdup(vreg_x_sum_even, 0, preg_134, MODE_ZEROING);
        vdup(vreg_x_sum_odd, 0, preg_134, MODE_ZEROING);
        vdup(vreg_x_sum_1_even, 0, preg_134, MODE_ZEROING);
        vdup(vreg_x_sum_1_odd, 0, preg_134, MODE_ZEROING);

        for (uint16_t i0 = 0; i0 < uint16_t(ubN / 4) ; ++i0) {
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
}
*/

template <bool init = false, int HEAD_SIZE, typename ReduceTileD1, typename TileDataD2, typename TileDataS1>
AICORE inline void pto_macro_fa_softmax_dn(TileDataD2 __out__ x_exp, TileDataS1 __in__ input_x,
    ReduceTileD1 __out__ local_max, ReduceTileD1 __out__ local_sum, ReduceTileD1 __in__ new_global_max,
        ReduceTileD1 __out__ new_global_sum, ReduceTileD1 __out__ exp_max, TileDataS1 __out__ input_reduce_tmp,
        TileDataS1 __out__ p_tile_fp32) {
    if constexpr (init) {
        softmax_opt_fa_dn_init_impl<HEAD_SIZE, ReduceTileD1, TileDataD2, TileDataS1>(
                x_exp, input_x, local_max, local_sum, new_global_max, new_global_sum, exp_max, input_reduce_tmp, p_tile_fp32);
    } else {
        softmax_opt_fa_dn_not_init_impl<HEAD_SIZE, ReduceTileD1, TileDataD2, TileDataS1>(
                x_exp, input_x, local_max, local_sum, new_global_max, new_global_sum, exp_max, input_reduce_tmp, p_tile_fp32);
    }
}

} // namespace pto

#endif
