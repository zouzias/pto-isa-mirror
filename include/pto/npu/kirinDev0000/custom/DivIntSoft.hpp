/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TDIV_INT_SOFT_HPP
#define TDIV_INT_SOFT_HPP

/**
 * @file DivIntSoft.hpp
 * @brief Soft integer division implementations for kirinDev0000 (__NPU_ARCH__ == 5101).
 *
 * kirinDev0000 hardware does not support integer division. The bisheng compiler provides
 * soft vdiv overloads for integer types only for __NPU_ARCH__ == 5102/5161 (a5/a6 variants),
 * guarded by macros that exclude 5101 even though __DAV_L510__ is defined.
 *
 * This file re-implements the same soft-division algorithms (adapted from the bisheng
 * compiler's __clang_cce_vector_intrinsics.h) so that TDIV can support int16_t/uint16_t/
 * int32_t/uint32_t on kirinDev0000. All required building-block intrinsics (vmull via
 * __DAV_L510__, vfcvt, pintlv_b32, vmul, vcvt, etc.) are available on 5101.
 */

#include <pto/common/constants.hpp>
#include <pto/common/utils.hpp>

namespace pto {

// ============================================================================
// vdiv soft implement for uint16_t
// Algorithm: convert u16 to f32, estimate 1/y via hardware f32 vdiv, scale by
// 2^16, convert back to s32, multiply by x (32-bit), deinterleave to u16
// quotient, then 2 iterations of quotient/remainder refinement.
// ============================================================================
AICORE inline void DivSoftIntImpl(vector_u16& dst, vector_u16 src0, vector_u16 src1, vector_bool mask)
{
    // x/0 = 0xFFFF
    vector_bool zero_mask, non_zero_mask, ori_mask;
    ori_mask = mask;
    vcmps_eq(zero_mask, src1, 0, mask);
    vector_u16 q_zero;
    vdup(q_zero, 0xFFFF, mask, MODE_ZEROING);
    pnot(non_zero_mask, zero_mask, mask);
    mask = non_zero_mask;

    // Since we need to do interleave, we temply use PAT_ALL.
    vector_bool pg = pset_b16(PAT_ALL);
    vector_f32 v_one_f32;
    vector_u16 v_zero_u16;
    vector_f32 v_zero_f32;
    vdup(v_one_f32, (float)1.0, pg, MODE_ZEROING);
    vdup(v_zero_u16, (uint16_t)0, pg, MODE_ZEROING);
    vdup(v_zero_f32, (float)0.0, pg, MODE_ZEROING);

    // convert u16 to f32
    vector_f32 vy_lower_f32;
    vector_f32 vy_higher_f32;
    vector_u16 vy_lower_u16;
    vector_u16 vy_higher_u16;
    vintlv(vy_lower_u16, vy_higher_u16, src1, v_zero_u16);
    vcvt(vy_lower_f32, (vector_s32&)vy_lower_u16, pg, ROUND_F);
    vcvt(vy_higher_f32, (vector_s32&)vy_higher_u16, pg, ROUND_F);

    // Initial estimate of inv(y).
    vector_f32 vy_rec_lower;
    vector_f32 vy_rec_higher;
    vdiv(vy_rec_lower, v_one_f32, vy_lower_f32, pg, MODE_ZEROING);
    vdiv(vy_rec_higher, v_one_f32, vy_higher_f32, pg, MODE_ZEROING);
    vector_f32 vy_scale_lower;
    vector_f32 vy_scale_higher;
    vector_u32 v_const;
    // Since f32 can fully cover the numerical range of u16,
    // we use 2^16 for the initial value calculation.
    vdup(v_const, 0x47800000U, pg, MODE_ZEROING);
    vmul(vy_scale_lower, vy_rec_lower, (vector_f32&)v_const, pg, MODE_ZEROING);
    vmul(vy_scale_higher, vy_rec_higher, (vector_f32&)v_const, pg, MODE_ZEROING);

    // trick: no need to do f32->u16, we can fully reuse s32 to do vmull(u16).
    vector_s32 v_lower_s32;
    vector_s32 v_higher_s32;
    vcvt(v_lower_s32, vy_scale_lower, pg, ROUND_F, RS_DISABLE);
    vcvt(v_higher_s32, vy_scale_higher, pg, ROUND_F, RS_DISABLE);

    // Quotient/remainder estimate.
    // vmull(u16)
    vector_u32 q_tmp_lower;
    vector_u32 q_tmp_higher;
    vector_u16 vx_lower_u16;
    vector_u16 vx_higher_u16;
    vector_u16 q_tmp;
    vector_u16 q_lower;
    vintlv(vx_lower_u16, vx_higher_u16, src0, v_zero_u16);
    vmul(q_tmp_lower, (vector_u32&)v_lower_s32, (vector_u32&)vx_lower_u16, pg, MODE_ZEROING);
    vmul(q_tmp_higher, (vector_u32&)v_higher_s32, (vector_u32&)vx_higher_u16, pg, MODE_ZEROING);
    vdintlv(q_lower, q_tmp, (vector_u16&)q_tmp_lower, (vector_u16&)q_tmp_higher);

    vector_u16 yq_tmp;
    vmul(yq_tmp, q_tmp, src1, mask, MODE_ZEROING);
    vector_u16 r_tmp;
    vsub(r_tmp, src0, yq_tmp, mask, MODE_ZEROING);

    // Two times of quotient/remainder refinement.
    vector_bool preg_1;
    vcmp_ge(preg_1, r_tmp, src1, mask);
    vsub(r_tmp, r_tmp, src1, preg_1, MODE_MERGING);
    vadds(q_tmp, q_tmp, 1, preg_1, MODE_MERGING);
    vcmp_ge(preg_1, r_tmp, src1, mask);
    vsub(r_tmp, r_tmp, src1, preg_1, MODE_MERGING);
    vadds(q_tmp, q_tmp, 1, preg_1, MODE_MERGING);

    vsel(q_tmp, q_zero, q_tmp, zero_mask);
    // MODE_ZEROING: dst = q_tmp
    dst = q_tmp;
}

// ============================================================================
// vdiv soft implement for int16_t
// Algorithm: strip sign, compute |x|/|y| via u16 soft div, then restore sign.
// ============================================================================
AICORE inline void DivSoftIntImpl(vector_s16& dst, vector_s16 src0, vector_s16 src1, vector_bool mask)
{
    // x/0 = -1
    vector_bool zero_mask, non_zero_mask, ori_mask, neg_x_mask;
    ori_mask = mask;
    vcmps_eq(zero_mask, src1, 0, mask);
    vector_s16 q_zero, neg_q_zero, pos_q_zero;
    vdup(q_zero, 0x7FFF, mask, MODE_ZEROING);
    vdup(neg_q_zero, 0x8000, mask, MODE_ZEROING);
    vcmps_lt(neg_x_mask, src0, 0, mask);
    vsel(q_zero, neg_q_zero, q_zero, neg_x_mask);
    pnot(non_zero_mask, zero_mask, mask);
    mask = non_zero_mask;

    // strip sign
    vector_u16 abs_x;
    vector_u16 abs_y;
    vabs((vector_s16&)abs_x, src0, mask, MODE_ZEROING);
    vabs((vector_s16&)abs_y, src1, mask, MODE_ZEROING);
    vector_s16 x_xor_y;
    vxor(x_xor_y, src0, src1, mask, MODE_ZEROING);
    vector_bool p_pos;
    vcmps_ge(p_pos, x_xor_y, 0, mask);

    // reuse vdiv u16
    vector_u16 dst_tmp;
    DivSoftIntImpl(dst_tmp, abs_x, abs_y, mask);

    // handle sign
    vector_s16 neg_q;
    vneg(neg_q, (vector_s16)dst_tmp, mask, MODE_ZEROING);
    vector_s16 q;
    vsel(q, (vector_s16)dst_tmp, neg_q, p_pos);

    vsel(q, q_zero, q, zero_mask);
    // MODE_ZEROING: dst = q
    dst = q;
}

} // namespace pto
#endif // TDIV_INT_SOFT_HPP
