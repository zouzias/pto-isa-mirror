/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TQUANT_HPP
#define TQUANT_HPP

#include <pto/common/constants.hpp>
#include <pto/common/utils.hpp>
#include <pto/npu/a5/common.hpp>
#include <pto/npu/a5/utils.hpp>

namespace pto {

PTO_INTERNAL void fp32_to_bf16(__ubuf__ float* srcPtr,
                                __ubuf__ bfloat16_t* dstPtr,
                                unsigned vl_count_b32,
                                unsigned total_elements_count) {
    static constexpr auto distValue =
            std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<bfloat16_t, DistVST::DIST_NORM>())>();
    vector_f32 vreg_b32_even, vreg_b32_odd;
    vector_bf16 vreg_b16_even, vreg_b16_odd, vreg_dst;
    uint32_t element_count_b16 = total_elements_count;
    for (uint16_t i = 0; i < (uint16_t) CeilDivision(vl_count_b32, 2); ++i){
        vector_bool preg_b32_0 = CreatePredicate<float>(total_elements_count);
        vector_bool preg_b32_1 = CreatePredicate<float>(total_elements_count);
        vector_bool preg_b16   = CreatePredicate<bfloat16_t>(total_elements_count);
        vlds(vreg_b32_even, vreg_b32_odd, srcPtr, i*128, DINTLV_B32);
        vcvt(vreg_b16_even, vreg_b32_even, preg_b32_0, ROUND_Z, RS_DISABLE, PART_EVEN, MODE_ZEROING);
        vcvt(vreg_b16_odd,  vreg_b32_odd,  preg_b32_1, ROUND_Z, RS_DISABLE, PART_ODD,  MODE_ZEROING);
        vor(vreg_dst, vreg_b16_even, vreg_b16_odd, preg_b16);
        vsts(vreg_dst, dstPtr, i*128, distValue, preg_b16);
    }
}

PTO_INTERNAL void AbsReduceMax_Naive(__ubuf__ float* srcPtr,
                                    __ubuf__ float* maxPtr,
                                    unsigned total_elements_count,
                                    unsigned vl_count,
                                    unsigned elementsPerRepeat,
                                    MaskReg &preg_lower32,
                                    MaskReg &preg_upper32) {

    RegTensor<float> vreg_b32;
    uint32_t elem_count = total_elements_count;
    //assumption: input total num of elements is a multiple of 64
    for (uint16_t i = 0; i < (uint16_t) vl_count; ++i){
        MaskReg preg = CreatePredicate<float>(elem_count);
        RegTensor<float> vreg_max_0, vreg_max_1;
        vlds(vreg_b32, srcPtr, i*elementsPerRepeat, NORM);
        vabs(vreg_b32, vreg_b32, preg);
        vcmax(vreg_max_0, vreg_b32, preg_lower32);
        vcmax(vreg_max_1, vreg_b32, preg_upper32);
        vsts(vreg_max_0, maxPtr,   2*i, ONEPT_B32, preg);
        vsts(vreg_max_1, maxPtr+1, 2*i, ONEPT_B32, preg);
    }
}

//Assumption: input total size is a multiple of 256 elements
PTO_INTERNAL void AbsReduceMax_f32_opt(__ubuf__ float* srcPtr,
                                    __ubuf__ float* maxPtr,
                                    unsigned vl_count,
                                    unsigned elementsPerRepeat,
                                    unsigned total_elements_count) {
    
    vector_f32 vreg_in_1, vreg_in_2, vreg_in_3, vreg_in_4, vreg_max_0, vreg_max_1, vreg_max;
    vector_f32 vreg_dintlv_1, vreg_dintlv_2, vreg_dintlv_3, vreg_dintlv_4, vreg_gp_max;
    vector_f32 vreg_dintlv_out_1, vreg_dintlv_out_2, vreg_dintlv_out_3, vreg_dintlv_out_4;
    vector_align ureg_max;
    uint32_t total_count = total_elements_count;
    MaskReg preg_lower8 = pset_b32(PAT_VL8);
    static constexpr auto distValue =
            std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<float, DistVST::DIST_NORM>())>();
    for (uint16_t i = 0; i < (uint16_t) vl_count/4; ++i){
        MaskReg preg_vl0 = CreatePredicate<float>(total_count);
        MaskReg preg_vl1 = CreatePredicate<float>(total_count);
        MaskReg preg_vl2 = CreatePredicate<float>(total_count);
        MaskReg preg_vl3 = CreatePredicate<float>(total_count);
        vlds(vreg_in_1, vreg_in_2, srcPtr,       i*4*elementsPerRepeat, DINTLV_B32); 
        vlds(vreg_in_3, vreg_in_4, srcPtr + 128, i*4*elementsPerRepeat, DINTLV_B32);
        vabs(vreg_in_1, vreg_in_1, preg_vl0);
        vabs(vreg_in_3, vreg_in_3, preg_vl2);
        vdintlv(vreg_dintlv_out_1, vreg_dintlv_out_2, vreg_in_1, vreg_in_3);
        vabs(vreg_in_2, vreg_in_2, preg_vl1);
        vabs(vreg_in_4, vreg_in_4, preg_vl3);
        vdintlv(vreg_dintlv_out_3, vreg_dintlv_out_4, vreg_in_2, vreg_in_4);
        vmax(vreg_max_0, vreg_dintlv_out_1, vreg_dintlv_out_2, preg_vl0);
        vmax(vreg_max_1, vreg_dintlv_out_3, vreg_dintlv_out_4, preg_vl1);
        vmax(vreg_max,   vreg_max_0,        vreg_max_1,        preg_vl0);
        vcgmax(vreg_gp_max, vreg_max, preg_vl0);
        vsts(vreg_gp_max, maxPtr, i*8, distValue, preg_lower8);
    }
}

// Finds maximum exponent directly and stores in bf16 format
PTO_INTERNAL void AbsReduceMax_bf16(__ubuf__ bfloat16_t* srcPtr,
                                   __ubuf__ bfloat16_t* maxPtr,
                                   unsigned vl_count_b16, 
                                   unsigned total_elements_count) {

    RegTensor<bfloat16_t> vb16_in_0, vb16_in_1, vb16_in_2, vb16_in_3, vb16_exponent_mask, vb16_max_0, vb16_max_1;
    vbr(vb16_exponent_mask, 0x7F80);
    constexpr unsigned elementsPerRepeat = 128; // b16
    MaskReg preg_all_b16 = pset_b16(PAT_ALL);
    vector_align ureg_max;
    //assumption: input total num of elements is a multiple of 128
    for (uint16_t i = 0; i < (uint16_t) vl_count_b16/4; ++i){
        vlds(vb16_in_0, vb16_in_1, srcPtr, i*elementsPerRepeat, DINTLV_B16);
        vand(vb16_in_0, vb16_in_0, vb16_exponent_mask, preg_all_b16, MODE_ZEROING);
        vand(vb16_in_1, vb16_in_1, vb16_exponent_mask, preg_all_b16, MODE_ZEROING);
        vlds(vb16_in_2, vb16_in_3, srcPtr + 256, i*elementsPerRepeat, DINTLV_B16);
        vand(vb16_in_2, vb16_in_2, vb16_exponent_mask, preg_all_b16, MODE_ZEROING);
        vand(vb16_in_3, vb16_in_3, vb16_exponent_mask, preg_all_b16, MODE_ZEROING);
        vmax(vb16_in_0, vb16_in_0, vb16_in_1, preg_all_b16);
        vmax(vb16_in_2, vb16_in_2, vb16_in_3, preg_all_b16);
        vcgmax((vector_f16 &)vb16_max_0, (vector_f16 &)vb16_in_0, preg_all_b16);
        vcgmax((vector_f16 &)vb16_max_1, (vector_f16 &)vb16_in_2, preg_all_b16);
        vstus(ureg_max, 8, vb16_max_0, maxPtr, POST_UPDATE);
        vstus(ureg_max, 8, vb16_max_1, maxPtr, POST_UPDATE);
    }
}



//Assumption: input total size is a multiple of 2K elements
PTO_INTERNAL void AbsReduceMax_f32_opt_largesizes(__ubuf__ float* srcPtr,
                                                __ubuf__ float* maxPtr,
                                                unsigned vl_count,
                                                unsigned elementsPerRepeat,
                                                unsigned total_elements_count) {
    
    vector_f32 vreg_in_1, vreg_in_2, vreg_in_3, vreg_in_4, vreg_max_0, vreg_max_1, vreg_max;
    vector_f32 vreg_dintlv_1, vreg_dintlv_2, vreg_dintlv_3, vreg_dintlv_4, vreg_gp_max;
    vector_f32 vreg_dintlv_out_1, vreg_dintlv_out_2, vreg_dintlv_out_3, vreg_dintlv_out_4;
    vector_bf16 vreg_max_bf16;
    vector_align ureg_max;
    uint32_t total_count = total_elements_count;
    MaskReg preg_lower16 = pset_b16(PAT_VL16);
    static constexpr auto distValue =
            std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<float, DistVST::DIST_NORM>())>();
    for (uint16_t i = 0; i < (uint16_t) vl_count/32; ++i){
        for (uint16_t j = 0; j < 8; ++j){ //handling 4 VLs per loop, each VL is 256 B (64 fp32)
            MaskReg preg_vl0 = CreatePredicate<float>(total_count);
            MaskReg preg_vl1 = CreatePredicate<float>(total_count);
            MaskReg preg_vl2 = CreatePredicate<float>(total_count);
            MaskReg preg_vl3 = CreatePredicate<float>(total_count);
            vlds(vreg_in_1, vreg_in_2, srcPtr,       (i*32+j*4)*elementsPerRepeat, DINTLV_B32); 
            vlds(vreg_in_3, vreg_in_4, srcPtr + 128, (i*32+j*4)*elementsPerRepeat, DINTLV_B32);
            vabs(vreg_in_1, vreg_in_1, preg_vl0);
            vabs(vreg_in_3, vreg_in_3, preg_vl2);
            vdintlv(vreg_dintlv_out_1, vreg_dintlv_out_2, vreg_in_1, vreg_in_3);
            vabs(vreg_in_2, vreg_in_2, preg_vl1);
            vabs(vreg_in_4, vreg_in_4, preg_vl3);
            vdintlv(vreg_dintlv_out_3, vreg_dintlv_out_4, vreg_in_2, vreg_in_4);
            vmax(vreg_max_0, vreg_dintlv_out_1, vreg_dintlv_out_2, preg_vl0);
            vmax(vreg_max_1, vreg_dintlv_out_3, vreg_dintlv_out_4, preg_vl1);
            vmax(vreg_max,   vreg_max_0,        vreg_max_1,        preg_vl0);
            vcgmax(vreg_gp_max, vreg_max, preg_vl0);
            // vcvt(vreg_max_bf16, vreg_gp_max, preg_vl0, ROUND_R, RS_ENABLE, PART_EVEN, MODE_ZEROING);
            // vsts(vreg_max_bf16, maxPtr, i*8, PK_B32, preg_lower16);
            vstus(ureg_max, 8, vreg_gp_max, maxPtr, POST_UPDATE);
        }
        vstas(ureg_max, maxPtr, 0, POST_UPDATE);
    }
}


// Computing scalar focus and exponent for F32 -> b8 e4m3 quantization
PTO_INTERNAL void ExtractB8ExponentAndScaling(__ubuf__ float* maxPtr,
                                            __ubuf__ uint8_t* expPtr,
                                            __ubuf__ float* scalingPtr,
                                            unsigned exp_max_loop_count,
                                            unsigned total_elements_count,
                                            unsigned elementsPerRepeat) {
    static constexpr auto distValue =
            std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<float, DistVST::DIST_NORM>())>();
    vector_f32 vb32_max;
    vector_s32 vb32_exponent, vb32_shared_exp, vb32_scaling, vb32_nan, vb32_subnorm;
    vector_s32 vb32_b8_shared_exp, vb32_b8_nan, vb32_b8_emax, vb32_exp_mask, vb32_exp_max;
    constexpr int shr = 23;
    vbr(vb32_exp_mask, 0x7F800000);
    vbr(vb32_b8_nan, 0xFF);
    vbr(vb32_subnorm, 0x7F800000);
    vbr(vb32_exp_max, 0xFE);
    vbr(vb32_exponent, 0x7F800000);
    vbr(vb32_b8_emax, 8);           // Max exponent for e4m3 is 8   
    vector_bool preg_inf;    
    uint32_t total_count = total_elements_count;
    for (uint16_t i = 0; i < (uint16_t) exp_max_loop_count; ++i){
        vector_bool preg_b32 = CreatePredicate<float>(total_count);
        vlds((vector_s32&)vb32_max, (__ubuf__ int32_t*)maxPtr, i*elementsPerRepeat, NORM);
        // Getting biased exponent
        vand((vector_s32&)vb32_exponent, (vector_s32&)vb32_max, vb32_exp_mask, preg_b32, MODE_ZEROING);
        vshrs((vector_s32&)vb32_exponent, (vector_s32&)vb32_exponent, shr, preg_b32, MODE_ZEROING);
        // Offsetting exponent to get shared exponent for fp8 e4m3
        vsub((vector_u32&)vb32_shared_exp, (vector_u32&)vb32_exponent, (vector_u32&)vb32_b8_emax, preg_b32);

        // calculating scaling factor = 1/shared_exponent
        vsub((vector_s32&)vb32_scaling, (vector_s32&)vb32_exp_max, (vector_s32&)vb32_shared_exp, preg_b32);
        vshls((vector_u32&)vb32_scaling, (vector_u32&)vb32_scaling, shr, preg_b32, MODE_ZEROING);

        // Handling special cases for NaN and Subnormal
        vcmps_ne(preg_inf, (vector_s32&)vb32_exponent, 0xFF, preg_b32);
        vsel(vb32_scaling,    vb32_scaling,    vb32_b8_nan, preg_inf);
        vsel(vb32_shared_exp, vb32_shared_exp, vb32_b8_nan, preg_inf);
        
        // clamp to scale_bits range
        vcmps_ge(preg_inf, (vector_s32&)vb32_scaling, -127, preg_b32);
        vsel(vb32_scaling,    vb32_scaling,    vb32_subnorm, preg_inf);
        vsel(vb32_shared_exp, vb32_shared_exp, vb32_subnorm, preg_inf);
        
        vsts((vector_s32 &)vb32_shared_exp, ((__ubuf__ int32_t *)expPtr),     i*elementsPerRepeat/4, PK4_B32,   preg_b32);
        vsts((vector_s32 &)vb32_scaling,    ((__ubuf__ int32_t *)scalingPtr), i*elementsPerRepeat,   distValue, preg_b32);
    }
}

// Computing scalar focus and exponent for Bf16 -> b8 e4m3 quantization
PTO_INTERNAL void ExtractB8ExponentAndScalingFromBF16(__ubuf__ bfloat16_t* maxPtr, // input is already the exponent but in bf16, needs shifting only
                                                    __ubuf__ uint8_t* expPtr,
                                                    __ubuf__ bfloat16_t* scalingPtr,
                                                    unsigned exp_max_loop_count,
                                                    unsigned total_elements_count,
                                                    unsigned elementsPerRepeat) {
    static constexpr auto distValue =
            std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<bfloat16_t, DistVST::DIST_NORM>())>();
    vector_bf16 vb16_max;
    vector_s16  vb16_exponent, vb16_shared_exp, vb16_scaling, vb16_nan, vb16_subnorm;
    vector_s16  vb16_b8_shared_exp, vb16_b8_nan, vb16_b8_emax, vb16_exp_mask, vb16_exp_max;
    constexpr int shr = 7;
    vbr(vb16_exp_mask, 0x7F80);
    vbr(vb16_b8_nan, 0xFF);
    vbr(vb16_subnorm, 0x7F80);
    vbr(vb16_exp_max, 0xFE);
    vbr(vb16_exponent, 0x7F80);
    vbr(vb16_b8_emax, 8);           // Max exponent for e4m3 is 8   
    vector_bool preg_inf;    
    uint32_t total_count = total_elements_count;
    for (uint16_t i = 0; i < (uint16_t) exp_max_loop_count; ++i){
        vector_bool preg_b16 = CreatePredicate<bfloat16_t>(total_count);
        vlds((vector_s16&)vb16_max, (__ubuf__ int16_t*)maxPtr, i*128, NORM);
        vshrs((vector_s16&)vb16_exponent, (vector_s16&)vb16_exponent, shr, preg_b16, MODE_ZEROING);
        // Offsetting exponent to get shared exponent for fp8 e4m3
        vsub((vector_u16&)vb16_shared_exp, (vector_u16&)vb16_exponent, (vector_u16&)vb16_b8_emax, preg_b16);

        // calculating scaling factor = 1/shared_exponent
        vsub((vector_s16&)vb16_scaling, (vector_s16&)vb16_exp_max, (vector_s16&)vb16_shared_exp, preg_b16);
        vshls((vector_u16&)vb16_scaling, (vector_u16&)vb16_scaling, shr, preg_b16, MODE_ZEROING);

        // Handling special cases for NaN and Subnormal
        vcmps_ne(preg_inf, (vector_s16&)vb16_exponent, 0xFF, preg_b16);
        vsel(vb16_scaling,    vb16_scaling,    vb16_b8_nan, preg_inf);
        vsel(vb16_shared_exp, vb16_shared_exp, vb16_b8_nan, preg_inf);
        
        // clamp to scale_bits range
        vcmps_ge(preg_inf, (vector_s16&)vb16_scaling, -127, preg_b16);
        vsel(vb16_scaling,    vb16_scaling,    vb16_subnorm, preg_inf);
        vsel(vb16_shared_exp, vb16_shared_exp, vb16_subnorm, preg_inf);
        
        vsts((vector_s16 &)vb16_shared_exp, ((__ubuf__ int16_t *)expPtr),     i*64,  PK_B16,    preg_b16);
        vsts((vector_s16 &)vb16_scaling,    ((__ubuf__ int16_t *)scalingPtr), i*128, distValue, preg_b16);
    }
}


PTO_INTERNAL void CalcQuantizedFP8Values(__ubuf__ float* srcPtr,
                                        __ubuf__ float* scalingPtr,
                                        __ubuf__ uint8_t* dstPtr,
                                        unsigned vl_count,
                                        unsigned elementsPerRepeat,
                                        unsigned total_elements_count,
                                        MaskReg &preg_lower32,
                                        MaskReg &preg_upper32) {
    vector_f32 vb32_scaling_0, vb32_scaling_1, vb32_in, vb32_out_1, vb32_out_2, vb32_out; 
    vector_f8e4m3 vb8_out;
    uint32_t elem_count = total_elements_count;
    MaskReg preg_ALL = pset_b32(PAT_ALL);
    for(uint16_t i = 0; i < (uint16_t) vl_count; ++i){
            MaskReg preg = CreatePredicate<float>(elem_count);
            vlds(vb32_scaling_0, scalingPtr,   2*i, BRC_B32);
            vlds(vb32_scaling_1, scalingPtr+1, 2*i, BRC_B32);
            vlds(vb32_in, srcPtr, i*elementsPerRepeat, NORM);
            vmul(vb32_out_1, vb32_in, vb32_scaling_0, preg_lower32, MODE_ZEROING);
            vmul(vb32_out_2, vb32_in, vb32_scaling_1, preg_upper32, MODE_ZEROING);
            vor(vb32_out, vb32_out_1, vb32_out_2, preg_ALL);
            vcvt((vector_f8e4m3 &)vb8_out, (vector_f32 &)vb32_out, preg_ALL, ROUND_R, RS_ENABLE, PART_P0);
            vsts((vector_u8 &) vb8_out, (__ubuf__ uint8_t* )dstPtr, i*elementsPerRepeat, PK4_B32, preg_ALL);
    }
}

PTO_INTERNAL void CalcQuantizedFP8Values_Unroll4(__ubuf__ float* srcPtr,
                                        __ubuf__ float* scalingPtr,
                                        __ubuf__ uint8_t* dstPtr,
                                        unsigned vl_count,
                                        unsigned elementsPerRepeat,
                                        unsigned total_elements_count) {
    vector_f32 vb32_scaling, vb32_in_0, vb32_in_1, vb32_in_2, vb32_in_3, vb32_out_1, vb32_out_2, vb32_out_3, vb32_out_4, vb32_out;
    vector_f32 vb32_in_even_0, vb32_in_odd_0, vb32_in_even_1, vb32_in_odd_1; 
    vector_f8e4m3 vb8_out, vb8_out_0, vb8_out_1, vb8_out_2, vb8_out_3, vb8_or, vb8_or_0, vb8_or_1;
    uint32_t elem_count = total_elements_count;
    MaskReg preg_ALL    = pset_b32(PAT_ALL);
    MaskReg preg_ALL_b8 = pset_b8(PAT_ALL);
        static constexpr auto distValue =
        std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<float, DistVST::DIST_NORM>())>();
    for(uint16_t i = 0; i < (uint16_t) vl_count/4; ++i){
            MaskReg preg = CreatePredicate<float>(elem_count);
            vlds(vb32_scaling, scalingPtr, 8*i, E2B_B32);
            vlds(vb32_in_0, vb32_in_1, srcPtr,       4*i*elementsPerRepeat, DINTLV_B32);
            vlds(vb32_in_2, vb32_in_3, srcPtr + 128, 4*i*elementsPerRepeat, DINTLV_B32);
            vdintlv(vb32_in_even_0, vb32_in_even_1, vb32_in_0, vb32_in_2);
            vmul(vb32_out_1, vb32_in_even_0, vb32_scaling, preg_ALL, MODE_ZEROING); // 0 4 8 12 ...
            vmul(vb32_out_3, vb32_in_even_1, vb32_scaling, preg_ALL, MODE_ZEROING); // 2 6 10 14 ...
            vdintlv(vb32_in_odd_0,  vb32_in_odd_1, vb32_in_1, vb32_in_3);
            vmul(vb32_out_2, vb32_in_odd_0, vb32_scaling, preg_ALL, MODE_ZEROING); // 1 5 9 13 ...
            vmul(vb32_out_4, vb32_in_odd_1, vb32_scaling, preg_ALL, MODE_ZEROING); // 3 7 11 15 ...
            vcvt((vector_f8e4m3 &)vb8_out_0, (vector_f32 &)vb32_out_1, preg_ALL, ROUND_R, RS_ENABLE, PART_P0, MODE_ZEROING);
            vcvt((vector_f8e4m3 &)vb8_out_1, (vector_f32 &)vb32_out_2, preg_ALL, ROUND_R, RS_ENABLE, PART_P1, MODE_ZEROING);
            vcvt((vector_f8e4m3 &)vb8_out_2, (vector_f32 &)vb32_out_3, preg_ALL, ROUND_R, RS_ENABLE, PART_P2, MODE_ZEROING);
            vcvt((vector_f8e4m3 &)vb8_out_3, (vector_f32 &)vb32_out_4, preg_ALL, ROUND_R, RS_ENABLE, PART_P3, MODE_ZEROING);
            vor(vb8_out_0, vb8_out_0, vb8_out_1, preg_ALL_b8);
            vor(vb8_out_2, vb8_out_2, vb8_out_3, preg_ALL_b8);
            vor(vb8_out, vb8_out_0, vb8_out_2, preg_ALL_b8);
            vsts((vector_u8 &) vb8_out, (__ubuf__ uint8_t* )dstPtr, 4*i*elementsPerRepeat, distValue, preg_ALL_b8);
    }
}

PTO_INTERNAL void CalcQuantizedFp8_BF16in(__ubuf__ bfloat16_t* srcPtr,
                                        __ubuf__ bfloat16_t* scalingPtr,
                                        __ubuf__ uint8_t* dstPtr,
                                        unsigned vl_count_b16,
                                        unsigned elementsPerRepeat_b16,
                                        unsigned total_elements_count) {
    vector_bf16 vb16_scaling, vb16_in_0, vb16_in_1, vb16_out_0, vb16_out_1, vb16_out; 
    vector_f8e4m3 vb8_out, vb8_out_0, vb8_out_1;
    uint32_t elem_count = total_elements_count;
    MaskReg preg_ALL = pset_b16(PAT_ALL);
    for(uint16_t i = 0; i < (uint16_t) vl_count_b16; ++i){
            MaskReg preg = CreatePredicate<bfloat16_t>(elem_count);
            vlds(vb16_scaling, scalingPtr, 8, E2B_B16);
            vlds(vb16_in_0, vb16_in_1, srcPtr, 2*i*elementsPerRepeat_b16, DINTLV_B16);
            vmul(vb16_out_0, vb16_in_0, vb16_scaling, preg_ALL, MODE_ZEROING);
            vmul(vb16_out_1, vb16_in_1, vb16_scaling, preg_ALL, MODE_ZEROING);
            vcvt(vb8_out_0, vb16_out_0, preg_ALL, ROUND_R, RS_DISABLE, PART_EVEN);
            vcvt(vb8_out_1, vb16_out_1, preg_ALL, ROUND_R, RS_DISABLE, PART_ODD);
            vor((vector_u8 &)vb8_out, (vector_u8 &)vb8_out_0, (vector_u8 &)vb8_out_1, preg_ALL);
            vsts((vector_u8 &) vb8_out, (__ubuf__ uint8_t* )dstPtr, i*elementsPerRepeat_b16, PK4_B32, preg_ALL);
    }

}


template <typename TileDataSrc, typename TileDataExp, typename TileDataOut, typename TileDataMax, unsigned SrcStride>
__tf__ PTO_INTERNAL void TQuant(typename TileDataSrc::TileDType __in__  src,
                                typename TileDataExp::TileDType __out__ exp,
                                typename TileDataOut::TileDType __out__ dst,
                                typename TileDataMax::TileDType __out__ max,
                                typename TileDataMax::TileDType __out__ scaling,
                                unsigned validRows,
                                unsigned validCols) {
    
    using T = typename TileDataSrc::DType;  // fp32
    using U = typename TileDataExp::DType;  // f8e8m0
    using V = typename TileDataOut::DType;  // f8e4m3
    __ubuf__ T *srcPtr = (__ubuf__ T *)__cce_get_tile_ptr(src);
    __ubuf__ U *expPtr = (__ubuf__ U *)__cce_get_tile_ptr(exp);
    __ubuf__ V *dstPtr = (__ubuf__ V *)__cce_get_tile_ptr(dst);
    __ubuf__ T *maxPtr = (__ubuf__ T *)__cce_get_tile_ptr(max);
    __ubuf__ T *maxPtr_backup = (__ubuf__ T *)__cce_get_tile_ptr(max);
    __ubuf__ T *scalingPtr = (__ubuf__ T *)__cce_get_tile_ptr(scaling);
    set_ctrl(1<<50); // set SPR.CTRL[50] to 1, to allow data clipping into MAX_NORM range for VCVTf32->f8 conversion
    __VEC_SCOPE__ {
        constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(T);
        unsigned numRepeatPerRow = CeilDivision(validCols, elementsPerRepeat);
        unsigned exp_max_loop_count = CeilDivision(validRows * TileDataSrc::Cols, 32*elementsPerRepeat);
        unsigned exp_max_loop_count_bf16 = CeilDivision(validRows * TileDataSrc::Cols, 32*128);
        MaskReg preg_lower32 = pset_b32(PAT_VL32);
        MaskReg preg_upper32;
        MaskReg preg_ALL = pset_b32(PAT_ALL);
        pxor(preg_upper32, preg_ALL, preg_lower32, preg_ALL);
        uint16_t vl_count = CeilDivision(validRows * TileDataSrc::Cols, elementsPerRepeat);
        uint32_t total_elements_count = validRows * TileDataSrc::Cols;
        // if total valid size less than 1024, don't unroll
        if (validRows * validCols <= 1024) {
            AbsReduceMax_Naive(srcPtr, maxPtr, total_elements_count, vl_count, elementsPerRepeat, preg_lower32, preg_upper32);
        } else if ((validRows * validCols)%2048 == 0) {
            AbsReduceMax_f32_opt_largesizes(srcPtr, maxPtr, vl_count, elementsPerRepeat, total_elements_count);
        } else { // unroll by 4
            AbsReduceMax_f32_opt(srcPtr, maxPtr, vl_count, elementsPerRepeat, total_elements_count);
        }
        mem_bar(VST_VLD);
        maxPtr = maxPtr_backup; // reset maxPtr
        ExtractB8ExponentAndScaling(maxPtr, expPtr, scalingPtr, exp_max_loop_count, 
                                            total_elements_count, elementsPerRepeat);
        mem_bar(VST_VLD);
        
        // use unrolled way if size is large 
        if (validRows * validCols > 1024 && (validRows * validCols)%256==0) {
            CalcQuantizedFP8Values_Unroll4(srcPtr, scalingPtr, (__ubuf__ uint8_t*)dstPtr, vl_count, 
                    elementsPerRepeat, total_elements_count);
        }
        else {
            CalcQuantizedFP8Values(srcPtr, scalingPtr, (__ubuf__ uint8_t*)dstPtr, vl_count, 
                    elementsPerRepeat, total_elements_count, preg_lower32, preg_upper32);
        }
    }
}


template <typename TileDataSrc, typename TileDataExp, typename TileDataOut, typename TileDataMax, unsigned SrcStride>
__tf__ PTO_INTERNAL void TQuant_via_bf16(typename TileDataSrc::TileDType __in__  src,
                                typename TileDataExp::TileDType __out__ exp,
                                typename TileDataOut::TileDType __out__ dst,
                                typename TileDataMax::TileDType __out__ max,
                                typename TileDataMax::TileDType __out__ scaling,
                                unsigned validRows,
                                unsigned validCols) {
    
    using T = typename TileDataSrc::DType;  // fp32
    using U = typename TileDataExp::DType;  // f8e8m0
    using V = typename TileDataOut::DType;  // f8e4m3
    __ubuf__ T *srcPtr = (__ubuf__ T *)__cce_get_tile_ptr(src);
    __ubuf__ U *expPtr = (__ubuf__ U *)__cce_get_tile_ptr(exp);
    __ubuf__ V *dstPtr = (__ubuf__ V *)__cce_get_tile_ptr(dst);
    __ubuf__ T *maxPtr = (__ubuf__ T *)__cce_get_tile_ptr(max);
    __ubuf__ T *maxPtr_backup = (__ubuf__ T *)__cce_get_tile_ptr(max);
    __ubuf__ T *scalingPtr = (__ubuf__ T *)__cce_get_tile_ptr(scaling);
    set_ctrl(1<<50); // set SPR.CTRL[50] to 1, to allow data clipping into MAX_NORM range for VCVTf32->f8 conversion
    __VEC_SCOPE__ {
        constexpr unsigned elementsPerRepeat_b32 = REPEAT_BYTE / sizeof(float);
        constexpr unsigned elementsPerRepeat_b16 = REPEAT_BYTE / sizeof(uint16_t);
        unsigned exp_max_loop_count_bf16 = CeilDivision(validRows * TileDataSrc::Cols, 32*elementsPerRepeat_b16);
        uint16_t vl_count_b32 = CeilDivision(validRows * TileDataSrc::Cols, elementsPerRepeat_b32);
        uint16_t vl_count_b16 = CeilDivision(validRows * TileDataSrc::Cols, elementsPerRepeat_b16);
        uint32_t total_elements_count = validRows * TileDataSrc::Cols;
        // Only designed for large size input
        fp32_to_bf16(srcPtr, (__ubuf__ bfloat16_t*)srcPtr, vl_count_b32, total_elements_count);
        mem_bar(VST_VLD);
        AbsReduceMax_bf16((__ubuf__ bfloat16_t*)srcPtr, (__ubuf__ bfloat16_t*)maxPtr, vl_count_b16, total_elements_count);
        mem_bar(VST_VLD);
        maxPtr = maxPtr_backup; // reset maxPtr
        ExtractB8ExponentAndScalingFromBF16((__ubuf__ bfloat16_t*)maxPtr, expPtr, (__ubuf__ bfloat16_t*)scalingPtr, 
                                            exp_max_loop_count_bf16, total_elements_count, elementsPerRepeat_b16);
        mem_bar(VST_VLD);
        
        CalcQuantizedFp8_BF16in((__ubuf__ bfloat16_t*)srcPtr, (__ubuf__ bfloat16_t*)scalingPtr, (__ubuf__ uint8_t*)dstPtr, 
                                vl_count_b16, elementsPerRepeat_b16, total_elements_count);
    }
}


template <typename TileDataSrc, typename TileDataExp, typename TileDataOut, typename TileDataMax, int mode>
PTO_INTERNAL void TQUANT_IMPL(TileDataSrc &src,
                            TileDataExp &exp,
                            TileDataOut &dst,
                            TileDataMax &max,
                            TileDataSrc &scaling) {
    using T = typename TileDataSrc::DType;
    static_assert(std::is_same<T, float32_t>::value, "Fix: Input has to be float 32");

    // TQuant<TileDataSrc, TileDataExp, TileDataOut, TileDataMax, mode>
    //     (src.data(), exp.data(), dst.data(), max.data(), scaling.data(),
    //      src.GetValidRow(), src.GetValidCol());

    TQuant_via_bf16<TileDataSrc, TileDataExp, TileDataOut, TileDataMax, mode>
        (src.data(), exp.data(), dst.data(), max.data(), scaling.data(),
         src.GetValidRow(), src.GetValidCol());
}
} // namespace pto
#endif // TQUANT_HPP