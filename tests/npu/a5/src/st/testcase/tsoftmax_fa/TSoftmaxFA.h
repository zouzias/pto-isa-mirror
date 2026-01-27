/*
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TSOFTMAXFA_HPP
#define TSOFTMAXFA_HPP

#include <pto/common/constants.hpp>
#include <pto/common/utils.hpp>
#include <pto/common/pto_tile.hpp>

namespace pto {

template <typename TileDataD1, typename TileDataD2, typename TileDataS1, typename TileDataS2, int init>
 __tf__ AICORE void TSOFTMAX_DN_FUSION(TileDataD2 &x_exp, TileDataS1 &input_x, TileDataS2 &bit_mask, 
                             TileDataD1 &local_max, TileDataD1 &local_sum,
                             TileDataD1 &new_global_max, TileDataD1 &new_global_sum,
                             TileDataD1 &exp_max) {

    __ubuf__ typename TileDataD2::DType *x_exp_Ptr = (__ubuf__ typename TileDataD2::DType *)__cce_get_tile_ptr(x_exp.data());
    __ubuf__ typename TileDataS1::DType *input_x_Ptr = (__ubuf__ typename TileDataS1::DType *)__cce_get_tile_ptr(input_x.data());
    __ubuf__ typename TileDataS2::DType *bit_mask_Ptr = (__ubuf__ typename TileDataS2::DType *)__cce_get_tile_ptr(bit_mask.data());
    __ubuf__ typename TileDataD1::DType *local_max_Ptr = (__ubuf__ typename TileDataD1::DType *)__cce_get_tile_ptr(local_max.data());
    __ubuf__ typename TileDataD1::DType *local_sum_Ptr = (__ubuf__ typename TileDataD1::DType *)__cce_get_tile_ptr(local_sum.data());
    __ubuf__ typename TileDataD1::DType *new_global_max_Ptr = (__ubuf__ typename TileDataD1::DType *)__cce_get_tile_ptr(new_global_max.data());
    __ubuf__ typename TileDataD1::DType *new_global_sum_Ptr = (__ubuf__ typename TileDataD1::DType *)__cce_get_tile_ptr(new_global_sum.data());
    __ubuf__ typename TileDataD1::DType *exp_max_Ptr = (__ubuf__ typename TileDataD1::DType *)__cce_get_tile_ptr(exp_max.data());

    float scale = 0.8;
    float keepProb = 1.0;

    unsigned ubM = TileDataD2::Rows;
    unsigned ubN = TileDataD2::Cols;

    __VEC_SCOPE__{
        // --------- flash softmax -------------
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
        // 128,64 -> 1,64

        for (uint16_t iter_m = 0; iter_m < uint16_t(ubN / 8) ; ++iter_m) {
            vector_address areg_0_1 = vag_b16(512);
            vld(src_00a, src0_ub + iter_m * 512 / 2, areg_0_1, NORM);//
            vmax(max_0a, max_0a, src_00a, preg_108);                    //64 b
            vld(src_00b, src0_ub + iter_m * 512 / 2 + 64, areg_0_1, NORM);//
            vmax(max_0b, max_0b, src_00b, preg_108);//64 b
            vld(src_01a, src0_ub1 + iter_m * 512 / 2, areg_0_1, NORM);
            vmax(max_1a, max_1a, src_01a, preg_108);
            vld(src_01b, src0_ub1 + iter_m * 512 / 2 + 64, areg_0_1, NORM);
            vmax(max_1b, max_1b, src_01b, preg_108);
            vld(src_02a, src0_ub2 + iter_m * 512 / 2, areg_0_1, NORM);
            vmax(max_2a, max_2a, src_02a, preg_108);
            vld(src_02b, src0_ub2 + iter_m * 512 / 2 + 64, areg_0_1, NORM);
            vmax(max_2b, max_2b, src_02b, preg_108);
            vld(src_03a, src0_ub3 + iter_m * 512 / 2, areg_0_1, NORM);
            vmax(max_3a, max_3a, src_03a, preg_108);
            vld(src_03b, src0_ub3 + iter_m * 512 / 2 + 64, areg_0_1, NORM);
            vmax(max_3b, max_3b, src_03b, preg_108);
        }

        if (not init)
        {
            vlds(vreg_x_max_f32_b, new_global_max_Ptr, 0, NORM);
            vmax(max_0a, max_0a, max_1a, preg_108);
            vmax(max_0b, max_0b, max_1b, preg_108);
            vmax(max_2a, max_2a, max_3a, preg_108);
            vmax(max_2b, max_2b, max_3b, preg_108);
            vmax(max_0a, max_0a, max_2a, preg_108);
            vmax(max_0b, max_0b, max_2b, preg_108);
            vmax(max_0a, max_0a, max_0b, preg_108);
            vsts(max_0a, local_max_Ptr, 0, NORM_B16, preg_108);     //Store input max in local_max
            vmuls(max_0a, max_0a, scale, preg_108);
            vmax(max_0a, max_0a, vreg_x_max_f32_b, preg_108);
            vexpdif(vreg_x_max_f32_b, vreg_x_max_f32_b, max_0a, preg_134, PART_ODD);
            vsts(vreg_x_max_f32_b, exp_max_Ptr, 0, NORM_B16, preg_108);     //Store input max in local_max
        }
        else
        {
            vmax(max_0a, max_0a, max_1a, preg_108);
            vmax(max_0b, max_0b, max_1b, preg_108);
            vmax(max_2a, max_2a, max_3a, preg_108);
            vmax(max_2b, max_2b, max_3b, preg_108);
            vmax(max_0a, max_0a, max_2a, preg_108);
            vmax(max_0b, max_0b, max_2b, preg_108);
            vmax(max_0a, max_0a, max_0b, preg_108);
            vsts(max_0a, (__ubuf__ float *)local_max_Ptr, 0, NORM_B16, preg_108);     //Store input max in local_max
            vmuls(max_0a, max_0a, scale, preg_108);
        }

        // mem_bar(VST_VLD);  //TODO: remove?
        //
        vdup(vreg_x_sum_even, 0, preg_134, MODE_ZEROING);
        vdup(vreg_x_sum_odd, 0, preg_134, MODE_ZEROING);
        vdup(vreg_x_sum_1_even, 0, preg_134, MODE_ZEROING);
        vdup(vreg_x_sum_1_odd, 0, preg_134, MODE_ZEROING);

        for (uint16_t i0 = 0; i0 < uint16_t(ubN / 4) ; ++i0) { //128,64
            vector_address areg_x_1 = vag_b32(128);
            vld(vreg_x_f32_a, input_x_Ptr, areg_x_1, NORM);
            vld(vreg_x_f32_b, ((__ubuf__ float *) input_x_Ptr + 64), areg_x_1, NORM);
            vld(vreg_x_f32_1_a, ((__ubuf__ float *) input_x_Ptr + ubN*ubM/2), areg_x_1, NORM);
            vld(vreg_x_f32_1_b, ((__ubuf__ float *) input_x_Ptr + ubN*ubM/2 + 64), areg_x_1, NORM);

            //NZ+1 output
            // vector_address areg_x_1 = vag_b32(64);
            // vld(vreg_x_f32_a, input_x_Ptr, areg_x_1, NORM);
            // vld(vreg_x_f32_b, ((__ubuf__ float *) input_x_Ptr + ubN*ubM/2), areg_x_1, NORM);
            // vld(vreg_x_f32_1_a, ((__ubuf__ float *) input_x_Ptr + ubN*ubM/4), areg_x_1, NORM);
            // vld(vreg_x_f32_1_b, ((__ubuf__ float *) input_x_Ptr + ubN*ubM/2 + ubN*ubM/4), areg_x_1, NORM);

            vmuls(vreg_x_f32_a, vreg_x_f32_a, scale, preg_108);
            vmuls(vreg_x_f32_b, vreg_x_f32_b, scale, preg_108);
            vexpdif(vreg_x_exp_even, vreg_x_f32_a, max_0a, preg_134, PART_ODD);// 0 ²»¸ÐÖªpart_xx
            vexpdif(vreg_x_exp_odd, vreg_x_f32_b, max_0a, preg_134, PART_ODD); // 1

            vmuls(vreg_x_f32_1_a, vreg_x_f32_1_a, scale, preg_108);
            vmuls(vreg_x_f32_1_b, vreg_x_f32_1_b, scale, preg_108);
            vexpdif(vreg_x_exp_even_1, vreg_x_f32_1_a, max_0a, preg_134, PART_ODD);// 0 ²»¸ÐÖªpart_xx
            vexpdif(vreg_x_exp_odd_1, vreg_x_f32_1_b, max_0a, preg_134, PART_ODD); // 1

            // vmuls(vreg_x_exp_even, vreg_x_exp_even, 1.0f, preg_108);
            // vmuls(vreg_x_exp_odd, vreg_x_exp_odd, 1.0f, preg_108);
            // vmuls(vreg_x_exp_even_1, vreg_x_exp_even_1, 1.0f, preg_108);
            // vmuls(vreg_x_exp_odd_1, vreg_x_exp_odd_1, 1.0f, preg_108);

            // vcvt(vreg_x_exp_even_f16, vreg_x_exp_even, preg_108, ROUND_R, RS_ENABLE, PART_EVEN, MODE_ZEROING);
            // vcvt(vreg_x_exp_odd_f16, vreg_x_exp_odd, preg_108, ROUND_R, RS_ENABLE, PART_EVEN, MODE_ZEROING);
            // vcvt(vreg_x_exp_even_f16_1, vreg_x_exp_even_1, preg_108, ROUND_R, RS_ENABLE, PART_EVEN, MODE_ZEROING);
            // vcvt(vreg_x_exp_odd_f16_1, vreg_x_exp_odd_1, preg_108, ROUND_R, RS_ENABLE, PART_EVEN, MODE_ZEROING);

            vmulscvt(vreg_x_exp_even_f16, vreg_x_exp_even, 1.0f, preg_100, PART_EVEN);
            vmulscvt(vreg_x_exp_odd_f16, vreg_x_exp_odd, 1.0f, preg_101, PART_EVEN);
            vmulscvt(vreg_x_exp_even_f16_1, vreg_x_exp_even_1, 1.0f, preg_135, PART_EVEN);
            vmulscvt(vreg_x_exp_odd_f16_1, vreg_x_exp_odd_1, 1.0f, preg_136, PART_EVEN);

            vdintlv(vreg_x_exp_f16_pack, vreg_x_exp_f16_packa, vreg_x_exp_even_f16, vreg_x_exp_odd_f16);
            vdintlv(vreg_x_exp_f16_1_pack, vreg_x_exp_f16_1_packa, vreg_x_exp_even_f16_1, vreg_x_exp_odd_f16_1);

            vsts(vreg_x_exp_f16_pack, ((__ubuf__ half *) x_exp_Ptr + i0*128), 0, NORM_B16, preg_108);
            vsts(vreg_x_exp_f16_1_pack, ((__ubuf__ half *) x_exp_Ptr + ubN*ubM/2 + i0*128), 0, NORM_B16, preg_108);

            //NZ+1 output
            // vsstb(vreg_x_exp_f16_pack, ((__ubuf__ half *&) x_exp_Ptr),
            //         0x810001,                                                             
            //         preg_136,                                                                 
            //         POST_UPDATE);                                                             
            // vsstb(vreg_x_exp_f16_1_pack, ((__ubuf__ half *&) x_exp_1),
            //         0x810001,                                                                 
            //         preg_136,                                                                
            //         POST_UPDATE); 

            vadd(vreg_x_sum_even, vreg_x_exp_even, vreg_x_sum_even, preg_134, MODE_ZEROING);//np.sum(x_exp, axis=-2, keepdims=True)
            vadd(vreg_x_sum_odd, vreg_x_exp_odd, vreg_x_sum_odd, preg_134, MODE_ZEROING);   //np.sum(x_exp, axis=-2, keepdims=True)

            vadd(vreg_x_sum_1_even, vreg_x_exp_even_1, vreg_x_sum_1_even, preg_134, MODE_ZEROING);//np.sum(x_exp, axis=-2, keepdims=True)
            vadd(vreg_x_sum_1_odd, vreg_x_exp_odd_1, vreg_x_sum_1_odd, preg_134, MODE_ZEROING);   //np.sum(x_exp, axis=-2, keepdims=True)
        }

        vadd(vreg_x_sum0, vreg_x_sum_odd, vreg_x_sum_even, preg_134, MODE_ZEROING);
        vadd(vreg_x_sum1, vreg_x_sum_1_odd, vreg_x_sum_1_even, preg_134, MODE_ZEROING);
        vadd(vreg_x_sum0, vreg_x_sum0, vreg_x_sum1, preg_134, MODE_ZEROING);
        vsts(vreg_x_sum0, ((__ubuf__ float *&) local_sum_Ptr), 0, NORM_B32, preg_134);
        if (init){
            vsts(vreg_x_sum0, ((__ubuf__ float *&) new_global_sum_Ptr), 0, NORM_B32, preg_134);
        }

        if (not init){
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
                vmul(vreg_l0, vreg_exp_max, vreg_l0, preg_134, MODE_ZEROING); // exp_max * insum
                vadd(vreg_l0, vreg_l0, vreg_x_sum0, preg_134, MODE_ZEROING); // + x_sum
                vsts(vreg_l0, ((__ubuf__ float *) new_global_sum_Ptr+ ii*64), 0, NORM_B32, preg_134);
            }
        }

    }
}

//compiler team pto-optimal version
template <typename TileDataD1, typename TileDataD2, typename TileDataS1, typename TileDataS2, int init>
__tf__ AICORE void TSOFTMAX_DN_FUSION2(TileDataD2 &x_exp, TileDataS1 &input_x, TileDataS2 &bit_mask, 
                             TileDataD1 &local_max, TileDataD1 &local_sum,
                             TileDataD1 &new_global_max, TileDataD1 &new_global_sum,
                             TileDataD1 &exp_max) {

    __ubuf__ typename TileDataD2::DType *x_exp_Ptr = (__ubuf__ typename TileDataD2::DType *)__cce_get_tile_ptr(x_exp.data());
    __ubuf__ typename TileDataS1::DType *input_x_Ptr = (__ubuf__ typename TileDataS1::DType *)__cce_get_tile_ptr(input_x.data());
    __ubuf__ typename TileDataS2::DType *bit_mask_Ptr = (__ubuf__ typename TileDataS2::DType *)__cce_get_tile_ptr(bit_mask.data());
    __ubuf__ typename TileDataD1::DType *local_max_Ptr = (__ubuf__ typename TileDataD1::DType *)__cce_get_tile_ptr(local_max.data());
    __ubuf__ typename TileDataD1::DType *local_sum_Ptr = (__ubuf__ typename TileDataD1::DType *)__cce_get_tile_ptr(local_sum.data());
    __ubuf__ typename TileDataD1::DType *new_global_max_Ptr = (__ubuf__ typename TileDataD1::DType *)__cce_get_tile_ptr(new_global_max.data());
    __ubuf__ typename TileDataD1::DType *new_global_sum_Ptr = (__ubuf__ typename TileDataD1::DType *)__cce_get_tile_ptr(new_global_sum.data());
    __ubuf__ typename TileDataD1::DType *exp_max_Ptr = (__ubuf__ typename TileDataD1::DType *)__cce_get_tile_ptr(exp_max.data());
    __ubuf__ typename TileDataD2::DType *x_exp_Ptr1 = (__ubuf__ typename TileDataD2::DType *)__cce_get_tile_ptr(x_exp.data());

    float scale = 0.8;
    float keepProb = 1.0;
    unsigned ubM_ = TileDataD2::Rows;
    // unsigned ubM_ = 128;
    unsigned ubN_ = TileDataD2::Cols;
    using T = typename TileDataS1::DType;
    using T1 = typename TileDataS1::DType;
    using T2 = typename TileDataD2::DType;
    constexpr unsigned nElmPerRepeat = CCE_VL / sizeof(float);
    constexpr unsigned elementsPerRepeat = CCE_VL / sizeof(float);

    uint16_t repeatTimes = CeilDivision(ubM_, nElmPerRepeat);
    unsigned reduceRow = 1;
    
    __VEC_SCOPE__{
        // --------- flash softmax -------------
        uint32_t sreg_92 = (uint32_t) 128ULL;
        uint32_t sreg_93 = (uint32_t) 64ULL;
        // vector_bool preg_108 = pset_b16(PAT_ALL);
        vector_bool preg_135 = pset_b32(PAT_ALL);
        vector_bool preg_136 = plt_b16(sreg_92, POST_UPDATE);
        vector_bool preg_137 = plt_b16(sreg_93, POST_UPDATE);


        // pto::TCOLMAX(maxaddrTiles[0], inputxTilesDN[pingpongQK_UB - 3]);
        {    
            RegTensor<T> input_even_reg;
            RegTensor<T> input_odd_reg;
            RegTensor<T> tmpVReg;
            RegTensor<T> input_dst_reg;
            RegTensor<T> input_max;
            __ubuf__ typename TileDataD1::DType *local_max_Ptr_cp = local_max_Ptr;
            __ubuf__ typename TileDataS1::DType *input_x_Ptr_cp = input_x_Ptr;

            for (uint16_t i = 0; i < repeatTimes; ++i) {
                vlds(input_max, input_x_Ptr_cp, 0, NORM);
                for (uint16_t j = 0; j < (ubN_ - 1) / 2; ++j) { // 127
                    vlds(input_even_reg, input_x_Ptr_cp, (2 * j + 1) * ubM_, NORM);
                    vlds(input_odd_reg, input_x_Ptr_cp, (2 * j + 2) * ubM_, NORM);
                    vmax(tmpVReg, input_even_reg, input_odd_reg, preg_135, MODE_ZEROING);
                    vmax(input_max, input_max, tmpVReg, preg_135, MODE_ZEROING);
                }
                if ((ubN_ - 1) % 2) { // 1
                    vlds(input_even_reg, input_x_Ptr_cp, (ubN_ - 1) * ubM_, NORM);
                    vmax(input_max, input_max, input_even_reg, preg_135, MODE_ZEROING);
                }
                vsts(input_max, local_max_Ptr_cp, 0, NORM_B32, preg_135);
                input_x_Ptr_cp += nElmPerRepeat;
                local_max_Ptr_cp += nElmPerRepeat;
            }
        }

        {
            if (not init)
            // pto::TMULS(maxaddrTiles[0], maxaddrTiles[0], 0.8f);
            // pto::TPARTMAX(maxaddrTiles[0], maxaddrTiles[0], maxaddrTiles[1]);
            // pto::TEXPDIF(expmaxTiles[qk_n_index % 3], maxaddrTiles[1], maxaddrTiles[0]);
            {
                RegTensor<T> v_local_max, v_global_max, v_expdif;
                for (uint16_t i = 0; i < (uint16_t)(reduceRow); ++i) {
                    for (uint16_t j = 0; j < (uint16_t)repeatTimes; ++j) {
                        // 这里多个ld
                        vlds(v_local_max,  local_max_Ptr + i * ubM_, j * elementsPerRepeat, NORM);
                        vlds(v_global_max, new_global_max_Ptr + i * ubM_, j * elementsPerRepeat, NORM);

                        vmuls(v_local_max, v_local_max, scale, preg_135);
                        vmax(v_local_max, v_local_max, v_global_max, preg_135);
                        vexpdif(v_expdif, v_global_max, v_local_max, preg_135, PART_ODD);
                        
                        vsts(v_local_max, local_max_Ptr + i * ubM_, j * elementsPerRepeat, NORM_B32, preg_135);
                        vsts(v_expdif, exp_max_Ptr + i * ubM_, j * elementsPerRepeat, NORM_B32, preg_135);
                    }
                }
            }
            else {
            // pto::TMULS(maxaddrTiles[1], maxaddrTiles[1], 0.8f);
                RegTensor<T> v_local_max;
                for (uint16_t i = 0; i < (uint16_t)(reduceRow); ++i)
                {
                    for (uint16_t j = 0; j < (uint16_t)repeatTimes; ++j)
                    {
                        vlds(v_local_max, local_max_Ptr + i * ubN_, j * elementsPerRepeat, NORM);
                        // vmuls(v_local_max, v_local_max, scale, preg_135);
                        vsts(v_local_max, local_max_Ptr + i * ubN_, j * elementsPerRepeat, NORM_B32, preg_135);
                    }
                }
            }
        }

        // mem_bar(VST_VLD);

        {
            // if we dont schedule manually
            RegTensor<T> vreg_x_sum, input_reg, vreg_x_exp, vreg_x_max_f32;
            RegTensor<T2> vreg_x_exp_f16;
            MaskReg preg1;
            MaskReg preg2;
            vdup(vreg_x_sum, 0, preg_135, MODE_ZEROING);
            vlds(vreg_x_max_f32, ((__ubuf__ float *) local_max_Ptr), 0, NORM); 
            vmuls(vreg_x_max_f32, vreg_x_max_f32, scale, preg_135);
            for (uint16_t i = 0; i < uint16_t(ubN_); ++i) {
                uint32_t sreg = (uint32_t)(ubM_);
                uint32_t sreg2 = (uint32_t)(128ULL);
                for (uint16_t j = 0; j < uint16_t(1); j++) {
                    preg1 = CreatePredicate<T1>(sreg);
                    preg2 = CreatePredicate<T2>(sreg2);
                    vlds(input_reg,  input_x_Ptr,  i * ubM_ + j * elementsPerRepeat, NORM);
                    vmuls(input_reg, input_reg, scale, preg1);
                    vexpdif(vreg_x_exp, input_reg, vreg_x_max_f32, preg1, PART_ODD);
                    vmulscvt(vreg_x_exp_f16, vreg_x_exp, keepProb, preg2, PART_EVEN);
                    vsts(vreg_x_exp_f16, ((__ubuf__ half *&)input_x_Ptr), i * ubM_*2 + j * elementsPerRepeat*2, NORM_B16, preg2);
                    vadd(vreg_x_sum, vreg_x_exp, vreg_x_sum, preg1, MODE_ZEROING);
                }
            }

            if (init)   
            // pto::TCOLSUM(sumaddrTiles[0], inputxTilesDN[pingpongQK_UB - 3]);
                vsts(vreg_x_sum, ((__ubuf__ float *&) new_global_sum_Ptr), 0, NORM_B32, preg_135);
            else 
            // pto::TCOLSUM(sumaddrTiles[1], inputxTilesDN[pingpongQK_UB - 3]);
                vsts(vreg_x_sum, ((__ubuf__ float *&) local_sum_Ptr), 0, NORM_B32, preg_135);
        }

/*
        //4 block fusion version
        {
            // if we dont schedule manually
            RegTensor<T> vreg_x_sum, input_reg, vreg_x_exp, vreg_x_max_f32;
            RegTensor<T2> vreg_x_exp_f16;

            vector_f16 vreg_dummy;
            vector_f16 vreg_x_exp_f16_pack;
            vector_f16 vreg_x_exp_f16_packa;

            MaskReg preg1;
            MaskReg preg2;
            vdup(vreg_x_sum, 0, preg_135, MODE_ZEROING);
            vlds(vreg_x_max_f32, ((__ubuf__ float *) local_max_Ptr), 0, NORM); 
            vmuls(vreg_x_max_f32, vreg_x_max_f32, scale, preg_135);
            for (uint16_t i = 0; i < uint16_t(ubN_); ++i) {
                uint32_t sreg = (uint32_t)(ubM_);
                uint32_t sreg2 = (uint32_t)(128ULL);
                for (uint16_t j = 0; j < uint16_t(1); j++) {
                    preg1 = CreatePredicate<T1>(sreg);
                    preg2 = CreatePredicate<T2>(sreg2);
                    vlds(input_reg,  input_x_Ptr,  i * ubM_ + j * elementsPerRepeat, NORM);
                    vmuls(input_reg, input_reg, scale, preg1);
                    vexpdif(vreg_x_exp, input_reg, vreg_x_max_f32, preg1, PART_ODD);
                    vmulscvt(vreg_x_exp_f16, vreg_x_exp, keepProb, preg2, PART_EVEN);

                    vdintlv(vreg_x_exp_f16_pack, vreg_x_exp_f16_packa, vreg_x_exp_f16, vreg_dummy);
                    vsstb(vreg_x_exp_f16_pack, ((__ubuf__ half *&) x_exp_Ptr1),
                        // 0x1000001,   //NZ
                        0x1010001,   //NZ+1                                                    
                        preg_137);
                    x_exp_Ptr1 += 8;

                    vsts(vreg_x_exp_f16, ((__ubuf__ half *&)input_x_Ptr), i * ubM_*2 + j * elementsPerRepeat*2, NORM_B16, preg2);
                    vadd(vreg_x_sum, vreg_x_exp, vreg_x_sum, preg1, MODE_ZEROING);
                }
            }

            if (init)   
            // pto::TCOLSUM(sumaddrTiles[0], inputxTilesDN[pingpongQK_UB - 3]);
                vsts(vreg_x_sum, ((__ubuf__ float *&) new_global_sum_Ptr), 0, NORM_B32, preg_135);
            else 
            // pto::TCOLSUM(sumaddrTiles[1], inputxTilesDN[pingpongQK_UB - 3]);
                vsts(vreg_x_sum, ((__ubuf__ float *&) local_sum_Ptr), 0, NORM_B32, preg_135);
        }
*/
/*
        {
            // if we dont schedule manually
            RegTensor<T> vreg_x_sum, input_reg, vreg_x_exp, vreg_x_max_f32;
            RegTensor<T2> vreg_x_exp_f16;

            vector_f16 vreg_dummy;
            vector_f16 vreg_x_exp_f16_pack;
            vector_f16 vreg_x_exp_f16_packa;

            MaskReg preg1;
            MaskReg preg2;

            UnalignReg ureg;

            vdup(vreg_x_sum, 0, preg_135, MODE_ZEROING);
            vlds(vreg_x_max_f32, ((__ubuf__ float *) local_max_Ptr), 0, NORM); 
            vmuls(vreg_x_max_f32, vreg_x_max_f32, scale, preg_135);
            for (uint16_t i = 0; i < uint16_t(ubN_); ++i) {
                uint32_t sreg = (uint32_t)(ubM_);
                uint32_t sreg2 = (uint32_t)(128ULL);
                uint32_t sreg3 = (uint32_t)(64ULL);
                for (uint16_t j = 0; j < uint16_t(1); j++) {
                    preg1 = CreatePredicate<T1>(sreg);
                    preg2 = CreatePredicate<T2>(sreg2);
                    vlds(input_reg,  input_x_Ptr,  i * ubM_ + j * elementsPerRepeat, NORM);
                    vmuls(input_reg, input_reg, scale, preg1);
                    vexpdif(vreg_x_exp, input_reg, vreg_x_max_f32, preg1, PART_ODD);
                    vmulscvt(vreg_x_exp_f16, vreg_x_exp, keepProb, preg2, PART_EVEN);

                    vdintlv(vreg_x_exp_f16_pack, vreg_x_exp_f16_packa, vreg_x_exp_f16, vreg_dummy);
                    // vsstb(vreg_x_exp_f16_pack, ((__ubuf__ half *&) x_exp_Ptr1),
                    //     0x810001,                                                            
                    //     preg_137);
                    // x_exp_Ptr1 += 8;
                    //USTd register, Sm, 
                    vstus(ureg, sreg3, vreg_x_exp_f16_pack, x_exp_Ptr1);

                    vsts(vreg_x_exp_f16, ((__ubuf__ half *&)input_x_Ptr), i * ubM_*2 + j * elementsPerRepeat*2, NORM_B16, preg2);
                    vadd(vreg_x_sum, vreg_x_exp, vreg_x_sum, preg1, MODE_ZEROING);
                }
                // vstas(ureg, x_exp_Ptr1, 0);
                x_exp_Ptr1 += 64;
            }
            vstas(ureg, x_exp_Ptr1, 0);    //able to fuse?
            

            if (init)   
            // pto::TCOLSUM(sumaddrTiles[0], inputxTilesDN[pingpongQK_UB - 3]);
                vsts(vreg_x_sum, ((__ubuf__ float *&) new_global_sum_Ptr), 0, NORM_B32, preg_135);
            else 
            // pto::TCOLSUM(sumaddrTiles[1], inputxTilesDN[pingpongQK_UB - 3]);
                vsts(vreg_x_sum, ((__ubuf__ float *&) local_sum_Ptr), 0, NORM_B32, preg_135);
        }
*/
/*
        {
            // unaligned test
            RegTensor<T> vreg_x_sum, input_reg, vreg_x_exp, vreg_x_max_f32;
            RegTensor<T2> vreg_x_exp_f16;

            vector_f16 vreg_dummy;
            vector_f16 vreg_x_exp_f16_pack;
            vector_f16 vreg_x_exp_f16_packa;

            MaskReg preg1;
            MaskReg preg2;
            MaskReg preg3;

            UnalignReg ureg;
            // uint16_t padRepeatTimes = CeilDivision(64, elementsPerRepeat);
            uint16_t padRepeatTimes = CeilDivision(66, elementsPerRepeat);

            vdup(vreg_x_sum, 0, preg_135, MODE_ZEROING);
            vlds(vreg_x_max_f32, ((__ubuf__ float *) local_max_Ptr), 0, NORM); 
            vmuls(vreg_x_max_f32, vreg_x_max_f32, scale, preg_135);
            for (uint16_t i = 0; i < uint16_t(ubN_); ++i) {
                uint32_t sreg = (uint32_t)(ubM_);
                uint32_t sreg2 = (uint32_t)(128ULL);
                uint32_t sreg3 = (uint32_t)(64ULL);
                uint32_t cols = (uint32_t)(66);
                for (uint16_t j = 0; j < uint16_t(padRepeatTimes); j++) {
                    preg1 = CreatePredicate<T1>(sreg);
                    preg2 = CreatePredicate<T2>(sreg2);
                    // preg3 = CreatePredicate<T2>(sreg3);
                    vlds(input_reg,  input_x_Ptr,  i * ubM_ + j * elementsPerRepeat, NORM);
                    vmuls(input_reg, input_reg, scale, preg1);
                    vexpdif(vreg_x_exp, input_reg, vreg_x_max_f32, preg1, PART_ODD);
                    vmulscvt(vreg_x_exp_f16, vreg_x_exp, keepProb, preg2, PART_EVEN);

                    vdintlv(vreg_x_exp_f16_pack, vreg_x_exp_f16_packa, vreg_x_exp_f16, vreg_dummy);
                    
                    // uint32_t sreg4 = cols > elementsPerRepeat ? elementsPerRepeat : cols;
                    // vstus(ureg, sreg4, vreg_x_exp_f16_pack, x_exp_Ptr1);
                    // cols -= elementsPerRepeat;

                    // vsts(vreg_x_exp_f16_pack, ((__ubuf__ half *&)x_exp_Ptr), i * ubM_ + j * elementsPerRepeat, NORM_B16, preg3);
                    uint32_t sreg4 = cols > elementsPerRepeat ? elementsPerRepeat : cols;
                    preg3 = CreatePredicate<T2>(sreg4);
                    vsts(vreg_x_exp_f16_pack, ((__ubuf__ half *&)x_exp_Ptr), i * ubM_ + j * elementsPerRepeat, NORM_B16, preg3);
                    cols -= elementsPerRepeat;

                    vsts(vreg_x_exp_f16, ((__ubuf__ half *&)input_x_Ptr), i * ubM_*2 + j * elementsPerRepeat*2, NORM_B16, preg2);
                    vadd(vreg_x_sum, vreg_x_exp, vreg_x_sum, preg1, MODE_ZEROING);
                }
                // x_exp_Ptr1 += 64;
            }
            // vstas(ureg, x_exp_Ptr1, 0);
            

            if (init)   
            // pto::TCOLSUM(sumaddrTiles[0], inputxTilesDN[pingpongQK_UB - 3]);
                vsts(vreg_x_sum, ((__ubuf__ float *&) new_global_sum_Ptr), 0, NORM_B32, preg_135);
            else 
            // pto::TCOLSUM(sumaddrTiles[1], inputxTilesDN[pingpongQK_UB - 3]);
                vsts(vreg_x_sum, ((__ubuf__ float *&) local_sum_Ptr), 0, NORM_B32, preg_135);
        }
*/
/*
        {
            // 128+128 test
            RegTensor<T> vreg_x_sum, input_reg, vreg_x_exp, vreg_x_max_f32;
            RegTensor<T2> vreg_x_exp_f16;

            vector_f16 vreg_dummy;
            vector_f16 vreg_x_exp_f16_pack;
            vector_f16 vreg_x_exp_f16_packa;

            MaskReg preg1;
            MaskReg preg2;
            MaskReg preg3;

            UnalignReg ureg;
            // uint16_t padRepeatTimes = CeilDivision(64, elementsPerRepeat);
            uint16_t padRepeatTimes = CeilDivision(128, elementsPerRepeat);

            vdup(vreg_x_sum, 0, preg_135, MODE_ZEROING);
            vlds(vreg_x_max_f32, ((__ubuf__ float *) local_max_Ptr), 0, NORM); 
            vmuls(vreg_x_max_f32, vreg_x_max_f32, scale, preg_135);
            for (uint16_t i = 0; i < uint16_t(ubN_); ++i) {
                uint32_t sreg = (uint32_t)(ubM_);
                uint32_t sreg2 = (uint32_t)(128ULL);
                uint32_t sreg3 = (uint32_t)(64ULL);
                preg3 = CreatePredicate<T2>(sreg3);
                for (uint16_t j = 0; j < uint16_t(padRepeatTimes); j++) {
                    preg1 = CreatePredicate<T1>(sreg);
                    preg2 = CreatePredicate<T2>(sreg2);
                    vlds(input_reg,  input_x_Ptr,  i * ubM_ + j * elementsPerRepeat, NORM);
                    vmuls(input_reg, input_reg, scale, preg1);
                    vexpdif(vreg_x_exp, input_reg, vreg_x_max_f32, preg1, PART_ODD);
                    vmulscvt(vreg_x_exp_f16, vreg_x_exp, keepProb, preg2, PART_EVEN);

                    vdintlv(vreg_x_exp_f16_pack, vreg_x_exp_f16_packa, vreg_x_exp_f16, vreg_dummy);

                    vsts(vreg_x_exp_f16_pack, ((__ubuf__ half *&)x_exp_Ptr), i * ubM_ + j * elementsPerRepeat, NORM_B16, preg3);
                    pnot(preg3, preg3, preg2);

                    vsts(vreg_x_exp_f16, ((__ubuf__ half *&)input_x_Ptr), i * ubM_*2 + j * elementsPerRepeat*2, NORM_B16, preg2);
                    vadd(vreg_x_sum, vreg_x_exp, vreg_x_sum, preg1, MODE_ZEROING);
                }
                // vstus(ureg, sreg2, vreg_x_exp_f16_pack, x_exp_Ptr1);
                // x_exp_Ptr1 += 128;
            }
            // vstas(ureg, x_exp_Ptr1, 0);
            

            if (init)   
            // pto::TCOLSUM(sumaddrTiles[0], inputxTilesDN[pingpongQK_UB - 3]);
                vsts(vreg_x_sum, ((__ubuf__ float *&) new_global_sum_Ptr), 0, NORM_B32, preg_135);
            else 
            // pto::TCOLSUM(sumaddrTiles[1], inputxTilesDN[pingpongQK_UB - 3]);
                vsts(vreg_x_sum, ((__ubuf__ float *&) local_sum_Ptr), 0, NORM_B32, preg_135);
        }
*/

        {
            // pto::TMULS(inputxTiles[pingpongQK_UB - 3], inputxTiles[pingpongQK_UB - 3], 0.8f);
            // pto::TEXPDIFBC(inputxTiles[pingpongQK_UB - 3], inputxTiles[pingpongQK_UB - 3], maxaddrTiles[0]);
            // pto::TCOLSUM(sumaddrTiles[0], inputxTilesDN[pingpongQK_UB - 3]);
            // pto::TMULSCVT(tmpintlvTile2, inputxTiles[pingpongQK_UB - 3], 1.0f);
            __ubuf__ half *x_exp_1 = (__ubuf__ half *)x_exp_Ptr + (ubN_/2 *16 / 2);
            vector_f16 vreg_x_exp_even_f16;
            vector_f16 vreg_x_exp_odd_f16;
            vector_f16 vreg_x_exp_even_f16_1;
            vector_f16 vreg_x_exp_odd_f16_1;

            vector_f16 vreg_x_exp_f16_pack;
            vector_f16 vreg_x_exp_f16_packa;
            vector_f16 vreg_x_exp_f16_1_pack;
            vector_f16 vreg_x_exp_f16_1_packa;
            //NZ output
            for (uint16_t i0 = 0; i0 < uint16_t(ubN_ / 4) ; ++i0) {
                vector_address areg_x_2 = vag_b16(128);
                vld(vreg_x_exp_even_f16, ((__ubuf__ half *)input_x_Ptr), areg_x_2, NORM);
                vld(vreg_x_exp_odd_f16,  ((__ubuf__ half *)input_x_Ptr+ ubN_*ubM_), areg_x_2, NORM);
                vld(vreg_x_exp_even_f16_1,((__ubuf__ half *)input_x_Ptr + ubN_*ubM_/2), areg_x_2, NORM);
                vld(vreg_x_exp_odd_f16_1, ((__ubuf__ half *)input_x_Ptr+ ubN_*ubM_ + ubN_*ubM_/2), areg_x_2, NORM);
                
                vdintlv(vreg_x_exp_f16_pack, vreg_x_exp_f16_packa, vreg_x_exp_even_f16, vreg_x_exp_odd_f16);
                vdintlv(vreg_x_exp_f16_1_pack, vreg_x_exp_f16_1_packa, vreg_x_exp_even_f16_1, vreg_x_exp_odd_f16_1);
                
                vsstb(vreg_x_exp_f16_pack, ((__ubuf__ half *&) x_exp_Ptr),
                        0x810001,                                                             
                        preg_136,                                                                 
                        POST_UPDATE);                                                             
                vsstb(vreg_x_exp_f16_1_pack, ((__ubuf__ half *&) x_exp_1),
                        0x810001,                                                                 
                        preg_136,                                                                
                        POST_UPDATE); 
                
                // vsstb(vreg_x_exp_f16_pack, ((__ubuf__ half *&) x_exp_Ptr),
                //         0x810001,
                //         // 0x810000,                                                         
                //         preg_136);                                                             
                // vsstb(vreg_x_exp_f16_1_pack, ((__ubuf__ half *&) x_exp_1),
                //         0x810001,  
                //         // 0x810000,                                                                  
                //         preg_136); 
                // // x_exp_Ptr += i0*129*32+32;
                // // x_exp_1 += i0*129*32+32;
                // // x_exp_Ptr += i0*129*32+16;
                // // x_exp_1 += i0*129*32+16;
                // // x_exp_Ptr += i0*32;
                // x_exp_Ptr += 8;  //32B for half, should be 16, why 8?
                // x_exp_1 += 8;
            }

            //ND output
            // for (uint16_t i0 = 0; i0 < uint16_t(ubN_ / 2) ; ++i0) {
            //     vector_address areg_x_2 = vag_b16(256);
            //     vld(vreg_x_exp_even_f16, ((__ubuf__ half *)input_x_Ptr), areg_x_2, NORM);
            //     vld(vreg_x_exp_odd_f16,  ((__ubuf__ half *)input_x_Ptr + 128), areg_x_2, NORM);

            //     vdintlv(vreg_x_exp_f16_pack, vreg_x_exp_f16_packa, vreg_x_exp_even_f16, vreg_x_exp_odd_f16);

            //     vsts(vreg_x_exp_f16_pack, ((__ubuf__ half *) x_exp_Ptr + i0*128), 0, NORM_B16, preg_136);
            // }
        }

        
        if (not init) {
            // pto::TMUL(sumaddrTiles[1], expmaxTiles[qk_n_index % 3], sumaddrTiles[1]);
            // pto::TADD(sumaddrTiles[1], sumaddrTiles[1], sumaddrTiles[0]);
            {
                RegTensor<T> v_exp_max, v_global_sum, v_local_sum;
                for (uint16_t i = 0; i < reduceRow; ++i) {
                    for (uint16_t j = 0; j < repeatTimes; ++j) {
                        vlds(v_exp_max, exp_max_Ptr + i * ubN_, j * elementsPerRepeat, NORM);
                        vlds(v_global_sum, new_global_sum_Ptr + i * ubN_, j * elementsPerRepeat, NORM);
                        vlds(v_local_sum, local_sum_Ptr + i * ubN_, j * elementsPerRepeat, NORM);

                        vmul(v_global_sum, v_exp_max, v_global_sum, preg_135, MODE_ZEROING);
                        vadd(v_global_sum, v_global_sum, v_local_sum, preg_135, MODE_ZEROING);

                        vsts(v_global_sum, new_global_sum_Ptr + i * ubN_, j * elementsPerRepeat, NORM_B32, preg_135);
                    }
                }
            }   
        }

    }
}

template <typename TileDataD1, typename TileDataD2, typename TileDataS1, typename TileDataS2, int init>
 __tf__ AICORE void TSOFTMAX_ND_FUSION(TileDataD2 &x_exp, TileDataS1 &input_x, TileDataS2 &bit_mask, 
                             TileDataD1 &local_max, TileDataD1 &local_sum,
                             TileDataD1 &new_global_max, TileDataD1 &new_global_sum,
                             TileDataD1 &exp_max) {

    __ubuf__ typename TileDataD2::DType *x_exp_Ptr = (__ubuf__ typename TileDataD2::DType *)__cce_get_tile_ptr(x_exp.data());
    __ubuf__ typename TileDataS1::DType *input_x_Ptr = (__ubuf__ typename TileDataS1::DType *)__cce_get_tile_ptr(input_x.data());
    __ubuf__ typename TileDataS2::DType *bit_mask_Ptr = (__ubuf__ typename TileDataS2::DType *)__cce_get_tile_ptr(bit_mask.data());
    __ubuf__ typename TileDataD1::DType *local_max_Ptr = (__ubuf__ typename TileDataD1::DType *)__cce_get_tile_ptr(local_max.data());
    __ubuf__ typename TileDataD1::DType *local_sum_Ptr = (__ubuf__ typename TileDataD1::DType *)__cce_get_tile_ptr(local_sum.data());
    __ubuf__ typename TileDataD1::DType *new_global_max_Ptr = (__ubuf__ typename TileDataD1::DType *)__cce_get_tile_ptr(new_global_max.data());
    __ubuf__ typename TileDataD1::DType *new_global_sum_Ptr = (__ubuf__ typename TileDataD1::DType *)__cce_get_tile_ptr(new_global_sum.data());
    __ubuf__ typename TileDataD1::DType *exp_max_Ptr = (__ubuf__ typename TileDataD1::DType *)__cce_get_tile_ptr(exp_max.data());

    float scale = 0.8;
    float keepProb = 1.0;

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

        if(init){
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
        }
        else {
            vlds(src_in2, new_global_max_Ptr, 0, NORM);
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
        if(init){
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

                        vmuls(vreg2, vreg2, 0.8f, preg_src, MODE_ZEROING);

                        vexp(vreg2, vreg2, preg_src, MODE_ZEROING);

                        vcadd(sum_1a, vreg2, preg_src, MODE_ZEROING);
                        vadd(sum_2a, sum_2a, sum_1a, preg_src, MODE_ZEROING);

                        // vmuls(vreg2, vreg2, 1.0f, preg_src, MODE_ZEROING);

                        // vcvt(vreg2_f16, vreg2, preg_src, ROUND_R, RS_ENABLE, PART_EVEN, MODE_ZEROING);
                        vmulscvt(vreg2_f16, vreg2, 1.0f, preg_src, PART_EVEN);

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

                    //unroll 2 to make use of vcvt
                    for (uint16_t j = 0; j < (uint16_t)(repeatTimes / 2); ++j) {
                        vlds(vreg0, row_ptr,  (j*2) * elementsPerRepeat, NORM);
                        vlds(vreg3, row_ptr,  (j*2+1) * elementsPerRepeat, NORM);

                        vsub(vreg2, vreg0, vreg1, preg_src, MODE_ZEROING);
                        vsub(vreg4, vreg3, vreg1, preg_src, MODE_ZEROING);

                        vmuls(vreg2, vreg2, 0.8f, preg_src, MODE_ZEROING);
                        vmuls(vreg4, vreg4, 0.8f, preg_src, MODE_ZEROING);

                        vexp(vreg2, vreg2, preg_src, MODE_ZEROING);
                        vexp(vreg4, vreg4, preg_src, MODE_ZEROING);

                        vcadd(sum_1a, vreg2, preg_src, MODE_ZEROING);
                        vcadd(sum_1b, vreg4, preg_src, MODE_ZEROING);
                        vadd(sum_2a, sum_2a, sum_1a, preg_src, MODE_ZEROING);
                        vadd(sum_2b, sum_2b, sum_1b, preg_src, MODE_ZEROING);

                        // vmuls(vreg2, vreg2, 1.0f, preg_src, MODE_ZEROING);
                        // vmuls(vreg4, vreg4, 1.0f, preg_src, MODE_ZEROING);

                        // vcvt(vreg2_f16, vreg2, preg_src, ROUND_R, RS_ENABLE, PART_EVEN, MODE_ZEROING);
                        // vcvt(vreg4_f16, vreg4, preg_src, ROUND_R, RS_ENABLE, PART_EVEN, MODE_ZEROING);
                        vmulscvt(vreg2_f16, vreg2, 1.0f, preg_src, PART_EVEN);
                        vmulscvt(vreg4_f16, vreg4, 1.0f, preg_src, PART_EVEN);

                        vdintlv(vreg_x_exp_f16_pack, vreg_x_exp_f16_packa, vreg2_f16, vreg4_f16);
                        vsts(vreg_x_exp_f16_pack, ((__ubuf__ half *) x_exp_Ptr + i*ubN + j*2*elementsPerRepeat), 0, NORM_B32, preg_src);
                    }
                    vadd(sum_2a, sum_2a, sum_2b, preg_src, MODE_ZEROING);
                    vsts(sum_2a, new_global_sum_Ptr+i, 0, distValue, preg_reduce);
                }
            }
        }
        else {
            vlds(max_2a, local_max_Ptr, 0, NORM);
            vlds(src_in1, new_global_max_Ptr, 0, NORM);
            vmax(max_2a, max_2a, src_in1, preg_src, MODE_ZEROING);
            vsub(vreg_exp_max, src_in1, max_2a, preg_b16_half, MODE_ZEROING);
            vsts(max_2a, new_global_max_Ptr, 0, NORM_B32, preg_b16_half);

            vmuls(vreg_exp_max, vreg_exp_max, 0.8f, preg_b16_half, MODE_ZEROING);
            vexp(vreg_exp_max, vreg_exp_max, preg_b16_half, MODE_ZEROING);
            vsts(vreg_exp_max, exp_max_Ptr, 0, NORM_B32, preg_src);
            // mem_bar(VST_VLD);

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

                        vmuls(vreg2, vreg2, 0.8f, preg_src, MODE_ZEROING);
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

                    //unroll 2 to make use of vcvt
                    for (uint16_t j = 0; j < (uint16_t)(repeatTimes / 2); ++j) {
                        vlds(vreg0, row_ptr,  (j*2) * elementsPerRepeat, NORM);
                        vlds(vreg3, row_ptr,  (j*2+1) * elementsPerRepeat, NORM);

                        vsub(vreg2, vreg0, vreg1, preg_src, MODE_ZEROING);
                        vsub(vreg4, vreg3, vreg1, preg_src, MODE_ZEROING);

                        vmuls(vreg2, vreg2, 0.8f, preg_src, MODE_ZEROING);
                        vmuls(vreg4, vreg4, 0.8f, preg_src, MODE_ZEROING);
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
    }
}

template <typename ReduceTileD1, typename TileDataD1, typename TileDataS1, typename TileDataS2, int init>
    AICORE void TSOFTMAX_DN_NOFUSION(TileDataD1 &x_exp, TileDataS1 &input_x, TileDataS2 &bit_mask, 
                             ReduceTileD1 &local_max, ReduceTileD1 &local_sum,
                             ReduceTileD1 &new_global_max, ReduceTileD1 &new_global_sum,
                             ReduceTileD1 &exp_max,
                             TileDataS1 &tmp0,
                             ReduceTileD1 &tmp1,
                             ReduceTileD1 &tmp2) {

            if(!init){
                /*
                TCOLMAX(local_max, input_x);
                TMULS(tmp2, local_max, 0.8f);
                TMAX(tmp1, tmp2, new_global_max);
                TSUB(tmp1, new_global_max, tmp1);
                TEXP(exp_max, tmp1);
                TMULS(input_x, input_x, 0.8f);
                TCOLEXPAND(tmp0, tmp2);
                TSUB(input_x, input_x, tmp0);
                TEXP(input_x, input_x);
                TCOLSUM(local_sum, input_x, tmp0, false);
                TMULS(input_x, input_x, 1.0f);
                TCVT(x_exp, input_x, RoundMode::CAST_ROUND);
                TMUL(new_global_sum, exp_max, new_global_sum);
                TADD(new_global_sum, new_global_sum, local_sum);
                */

                TCOLMAX(local_max, input_x);
                TMAX(local_max, local_max, new_global_max);
                TSUB(exp_max, new_global_max, local_max);
                TMULS(new_global_max, local_max, 1.0f);
                TMULS(exp_max, exp_max, 0.8f);
                TEXP(exp_max, exp_max);
                TCOLEXPANDSUB(input_x, input_x, local_max);
                TMULS(input_x, input_x, 0.8f);
                TEXP(input_x, input_x);
                TCOLSUM(local_sum, input_x, tmp1, false);
                TCVT(x_exp, input_x, RoundMode::CAST_ROUND);
                TMUL(new_global_sum, exp_max, new_global_sum);
                TADD(new_global_sum, new_global_sum, local_sum);
            }
            else {
                /*
                TCOLMAX(local_max, input_x);
                TCOLEXPAND(tmp0, local_max);
                TSUB(input_x, input_x, tmp0);
                TMULS(input_x, input_x, 0.8f);
                // TMULS(local_max, local_max, 0.8f);
                TEXP(input_x, input_x);
                TCOLSUM(local_sum, input_x, tmp0, false);
                TCOLSUM(new_global_sum, input_x, tmp0, false);
                TMULS(input_x, input_x, 1.0f);  //KeepProb, compiler cannot optimize
                TCVT(x_exp, input_x, RoundMode::CAST_ROUND);
                */

                TCOLMAX(new_global_max, input_x);
                TCOLEXPANDSUB(input_x, input_x, new_global_max);
                TMULS(input_x, input_x, 0.8f);
                TEXP(input_x, input_x);
                TCOLSUM(new_global_sum, input_x, tmp1, false);
                TMULS(input_x, input_x, 1.0f);
                TCVT(x_exp, input_x, RoundMode::CAST_ROUND);    //1111 ND output
            }

    }

template <typename ReduceTileD1, typename TileDataD1, typename TileDataS1, typename TileDataS2, int init>
    AICORE void TSOFTMAX_ND_NOFUSION(TileDataD1 &x_exp, TileDataS1 &input_x, TileDataS2 &bit_mask, 
                             ReduceTileD1 &local_max, ReduceTileD1 &local_sum,
                             ReduceTileD1 &new_global_max, ReduceTileD1 &new_global_sum,
                             ReduceTileD1 &exp_max,
                             TileDataS1 &tmp_float,
                             TileDataS1 &p_tile_f32) {

            if(!init){
                using ReduceTileD2 = Tile<TileType::Vec, float, 1, ReduceTileD1::Rows, BLayout::RowMajor, 1, ReduceTileD1::Rows>;
                using Tile1D_fp32 = Tile<TileType::Vec, float, 1, TileDataS1::Rows*TileDataS1::Cols, BLayout::RowMajor, 1, TileDataS1::Rows*TileDataS1::Cols>;
                using Tile1D_out = Tile<TileType::Vec, typename TileDataD1::DType, 1, TileDataS1::Rows*TileDataS1::Cols, BLayout::RowMajor, 1, TileDataS1::Rows*TileDataS1::Cols>;

                ReduceTileD2 tmp_shw_local_max;
                ReduceTileD2 tmp_shw_new_global_max;
                ReduceTileD2 tmp_shw_exp_max;
                ReduceTileD2 tmp_shw_new_global_sum;
                ReduceTileD2 tmp_shw_local_sum;
                Tile1D_fp32 p_tile_f32_1d;
                Tile1D_out x_exp_1d;

                TROWMAX(local_max, input_x, tmp_float);
                TRESHAPE(tmp_shw_local_max, local_max);
                TRESHAPE(tmp_shw_new_global_max, new_global_max);
                TMAX(tmp_shw_local_max, tmp_shw_local_max, tmp_shw_new_global_max);
                TRESHAPE(tmp_shw_exp_max, exp_max);
                TSUB(tmp_shw_exp_max, tmp_shw_new_global_max, tmp_shw_local_max);
                TMULS(tmp_shw_new_global_max, tmp_shw_local_max, 1.0f); // just copy
                TROWEXPANDSUB(tmp_float, input_x, local_max);
                TMULS(tmp_shw_exp_max, tmp_shw_exp_max, 0.8f);
                TMULS(tmp_float, tmp_float, 0.8f);
                TEXP(tmp_shw_exp_max, tmp_shw_exp_max);
                TRESHAPE(tmp_shw_exp_max, exp_max);
                TEXP(p_tile_f32, tmp_float);
                TRESHAPE(tmp_shw_exp_max, exp_max);
                TRESHAPE(p_tile_f32_1d, p_tile_f32);
                TRESHAPE(x_exp_1d, x_exp);    
                TCVT(x_exp_1d, p_tile_f32_1d, RoundMode::CAST_ROUND);  //TODO: check if 1d block the vf fusion?
                TRESHAPE(tmp_shw_new_global_sum, new_global_sum);
                TMUL(tmp_shw_new_global_sum, tmp_shw_exp_max, tmp_shw_new_global_sum);
                TROWSUM(local_sum, p_tile_f32, tmp_float);
                TRESHAPE(tmp_shw_local_sum, local_sum);
                TADD(tmp_shw_new_global_sum, tmp_shw_new_global_sum, tmp_shw_local_sum);

            }
            else {
                using Tile1D_fp32 = Tile<TileType::Vec, float, 1, TileDataS1::Rows*TileDataS1::Cols, BLayout::RowMajor, 1, TileDataS1::Rows*TileDataS1::Cols>;
                using Tile1D_out = Tile<TileType::Vec, typename TileDataD1::DType, 1, TileDataS1::Rows*TileDataS1::Cols, BLayout::RowMajor, 1, TileDataS1::Rows*TileDataS1::Cols>;
                Tile1D_fp32 p_tile_f32_1d;
                Tile1D_out x_exp_1d;

                TROWMAX(new_global_max, input_x, tmp_float);
                TROWEXPANDSUB(tmp_float, input_x, new_global_max);
                TMULS(tmp_float, tmp_float, 0.8f);
                TEXP(p_tile_f32, tmp_float);
                TROWSUM(new_global_sum, input_x, tmp_float);
                TMULS(input_x, input_x, 1.0f);
                TRESHAPE(p_tile_f32_1d, p_tile_f32);
                TRESHAPE(x_exp_1d, x_exp);
                TCVT(x_exp_1d, p_tile_f32_1d, RoundMode::CAST_ROUND);
            }

    }
}

#endif
