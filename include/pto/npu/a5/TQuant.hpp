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
        constexpr auto distValue =
            std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<T, DistVST::DIST_NORM>())>();
        constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(T);
        unsigned numRepeatPerRow = CeilDivision(validCols, elementsPerRepeat);
        unsigned exp_max_loop_count = CeilDivision(validRows * TileDataSrc::Cols, 32*elementsPerRepeat);
//----------------------Computing Absolute Max Value Per Groups of 32---------------------------------------
         // uint32_t group_num_per_row = CeilDivision(validCols, 32);
        vector_bool preg_lower32 = pset_b32(PAT_VL32);
        vector_bool preg_upper32;
        vector_bool preg_ALL = pset_b32(PAT_ALL);
        pxor(preg_upper32, preg_ALL, preg_lower32, preg_ALL);
        uint16_t vl_count = CeilDivision(validRows * TileDataSrc::Cols, elementsPerRepeat);
        uint32_t total_elements_count = validRows * TileDataSrc::Cols;
        // if total valid size less than 1024, don't unroll
        if (validRows * validCols <= 1024) {
            // No unrolling case
            // Finding max per 32 elements
            // assumption: row size is larger than 32
            vector_f32 vreg_b32;
            vector_align ureg_max;
            for (uint16_t i = 0; i < (uint16_t) vl_count; ++i){   // handles grouping within each row
                vector_bool preg = CreatePredicate<T>(total_elements_count);
                RegTensor<T> vreg_max_0, vreg_max_1;
                vlds(vreg_b32, srcPtr, i*elementsPerRepeat, NORM);
                vabs(vreg_b32, vreg_b32, preg);
                vcmax(vreg_max_0, vreg_b32, preg_lower32);
                vcmax(vreg_max_1, vreg_b32, preg_upper32);
                vstus(ureg_max, 1, vreg_max_0, maxPtr, POST_UPDATE);
                vstus(ureg_max, 1, vreg_max_1, maxPtr, POST_UPDATE);
            }
            vstas(ureg_max, maxPtr, 0, POST_UPDATE); // need a vstas every 64 vstus
        } else { // unroll by 4
            vector_f32 vreg_in_1, vreg_in_2, vreg_in_3, vreg_in_4, vreg_max_0, vreg_max_1, vreg_max;
            vector_f32 vreg_dintlv_1, vreg_dintlv_2, vreg_dintlv_3, vreg_dintlv_4, vreg_gp_max;
            vector_f32 vreg_dintlv_out_1, vreg_dintlv_out_2, vreg_dintlv_out_3, vreg_dintlv_out_4;
            vector_align ureg_max;
            vector_bool preg_ALL = pset_b32(PAT_ALL);
            for (uint16_t i = 0; i < (uint16_t) vl_count/4; ++i){
                vlds(vreg_in_1, vreg_in_2, srcPtr,       i*4*elementsPerRepeat, DINTLV_B32); 
                vlds(vreg_in_3, vreg_in_4, srcPtr + 128, i*4*elementsPerRepeat, DINTLV_B32);
                vabs(vreg_in_1, vreg_in_1, preg_ALL);
                vabs(vreg_in_3, vreg_in_3, preg_ALL);
                vdintlv(vreg_dintlv_out_1, vreg_dintlv_out_2, vreg_in_1, vreg_in_3);
                vabs(vreg_in_2, vreg_in_2, preg_ALL);
                vabs(vreg_in_4, vreg_in_4, preg_ALL);
                vdintlv(vreg_dintlv_out_3, vreg_dintlv_out_4, vreg_in_2, vreg_in_4);
                vmax(vreg_max_0, vreg_dintlv_out_1, vreg_dintlv_out_2, preg_ALL);
                vmax(vreg_max_1, vreg_dintlv_out_3, vreg_dintlv_out_4, preg_ALL);
                vmax(vreg_max,   vreg_max_0,        vreg_max_1,        preg_ALL);
                vcgmax(vreg_gp_max, vreg_max, preg_ALL);
                vstus(ureg_max, 8, vreg_gp_max, maxPtr, POST_UPDATE);
                vstas(ureg_max, maxPtr, 0, POST_UPDATE);
            }
        }
        mem_bar(VST_VLD);
//----------------------Computing Scaling Factor and Exponent------------------------------------------------
        maxPtr = maxPtr_backup; // reset maxPtr
        vector_f32 vb32_max, vb32_e4m3_max, vb32_e4m3_min;
        vector_s32 vb32_exponent, vb32_shared_exp, vb32_scaling, vb32_nan, vb32_subnorm;
        vector_s32 vb32_b8_shared_exp, vb32_b8_nan, vb32_b8_emax, vb32_exp_mask, vb32_exp_max;
        vector_f32 vb32_scaling_0, vb32_scaling_1, vb32_in, vb32_out_1, vb32_out_2, vb32_out; 
        vector_f8e4m3 vreg_f8_out;
        vector_bool preg_inf;
        constexpr int shr = 23;
        vbr(vb32_exp_mask, 0x7F800000);
        vbr(vb32_b8_nan, 0xFF);
        vbr(vb32_subnorm, 0x7F800000);
        vbr(vb32_exp_max, 0xFE);
        vbr(vb32_exponent, 0x7F800000);
        vbr(vb32_b8_emax, 8);           // Max exponent for e4m3 is 8   
        for (uint16_t i = 0; i < (uint16_t) exp_max_loop_count; ++i){
            vector_bool preg_b32 = CreatePredicate<T>(total_elements_count);
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
            
            // current format is 00XX, need to only store XX
            vsts((vector_s32 &)vb32_shared_exp, ((__ubuf__ int32_t *)expPtr),     i*elementsPerRepeat/4, PK4_B32,   preg_b32);
            vsts((vector_s32 &)vb32_scaling,    ((__ubuf__ int32_t *)scalingPtr), i*elementsPerRepeat,   distValue, preg_b32);
        }

        mem_bar(VST_VLD);
//-----------------------------Calculating Quantized FP8 values--------------------------------------------------
        for(uint16_t i = 0; i < (uint16_t) vl_count; ++i){
            vlds(vb32_scaling_0, scalingPtr,   2*i, BRC_B32);
            vlds(vb32_scaling_1, scalingPtr+1, 2*i, BRC_B32);
            vlds(vb32_in, srcPtr, i*elementsPerRepeat, NORM);
            vmul(vb32_out_1, vb32_in, vb32_scaling_0, preg_lower32, MODE_ZEROING);
            vmul(vb32_out_2, vb32_in, vb32_scaling_1, preg_upper32, MODE_ZEROING);
            vor(vb32_out, vb32_out_1, vb32_out_2, preg_ALL);
            vcvt((vector_f8e4m3 &)vreg_f8_out, (vector_f32 &)vb32_out, preg_ALL, ROUND_R, RS_ENABLE, PART_P0);
            vsts((vector_u8 &) vreg_f8_out, (__ubuf__ uint8_t* )dstPtr, i*elementsPerRepeat, PK4_B32, preg_ALL);
        }
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

    TQuant<TileDataSrc, TileDataExp, TileDataOut, TileDataMax, mode>
        (src.data(), exp.data(), dst.data(), max.data(), scaling.data(),
         src.GetValidRow(), src.GetValidCol());
}
} // namespace pto

#endif // TQUANT_HPP