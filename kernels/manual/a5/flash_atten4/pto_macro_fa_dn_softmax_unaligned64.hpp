/*
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_MACRO_FA_SOFTMAX_DN_UNALIGNED64_HPP
#define PTO_MACRO_FA_SOFTMAX_DN_UNALIGNED64_HPP

#include <pto/pto-inst.hpp>
#include "fa_performance_kernel.h"

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

[aicore] static inline uint64_t get_skip_status_slot_base(int tile_id)
{
    int slot_idx = tile_id % SKIP_STATUS_FIFO_SIZE;
    return SKIP_STATUS_SSBUF_BASE + static_cast<uint64_t>(slot_idx * SKIP_STATUS_SLOT_BYTES);
}

[aicore] static inline uint64_t get_skip_status_vec0_addr(int tile_id)
{
    return get_skip_status_slot_base(tile_id);
}

[aicore] static inline uint64_t get_skip_status_vec1_addr(int tile_id)
{
    return get_skip_status_slot_base(tile_id) + sizeof(uint64_t);
}

[aicore] static inline void write_skip_status_slot(int tile_id, bool is_vec1, uint32_t value)
{
    uint64_t addr = is_vec1 ? get_skip_status_vec1_addr(tile_id) : get_skip_status_vec0_addr(tile_id);
    volatile __ssbuf__ uint64_t *ptr = (volatile __ssbuf__ uint64_t *)addr;
    *ptr = static_cast<uint64_t>(value);
}

[aicore] static inline uint32_t read_skip_status_slot(int tile_id, bool is_vec1)
{
    uint64_t addr = is_vec1 ? get_skip_status_vec1_addr(tile_id) : get_skip_status_vec0_addr(tile_id);
    volatile __ssbuf__ uint64_t *ptr = (volatile __ssbuf__ uint64_t *)addr;
    return static_cast<uint32_t>(*ptr);
}

template <int FifoSize, int SyncPeriod>
AICORE inline bool aiv_should_wait_consumption(int sync_iter)
{
    static_assert(FifoSize >= 1, "FIFO size must be >= 1");
    constexpr int period = (SyncPeriod > 0) ? SyncPeriod : 1;
    static_assert(period >= 1, "Sync period must be >= 1");
    if (sync_iter < static_cast<int>(FifoSize))
        return false;
    return (sync_iter % period) == 0;
}



template <int HEAD_SIZE, bool CAUSAL_MASK, typename ReduceTileD1, typename TileDataD2, typename TileDataS1, typename TileDataH_NZ_T, int MODE>
__tf__ AICORE inline void softmax_opt_fa_dn_init_impl(int tile_id, int sync_iter, TileDataD2 __out__ x_exp,
                                                      TileDataS1 __in__ input_x,
                                                      ReduceTileD1 __out__ local_max, ReduceTileD1 __out__ local_sum,
                                                      ReduceTileD1 __out__ new_global_max,
                                                      ReduceTileD1 __out__ new_global_sum, ReduceTileD1 __out__ exp_max,
                                                      TileDataS1 __out__ tmp_float, TileDataS1 __out__ p_tile_f32,
                                                      TileDataS1 triu, TileDataH_NZ_T nzConvBuffer, int s0_index, int s1_index, bool last_tile)
{
#if USE_MANUAL
    __ubuf__ typename TileDataD2::DType *x_exp_Ptr = (__ubuf__ typename TileDataD2::DType *)__cce_get_tile_ptr(x_exp.data());
    __ubuf__ typename TileDataS1::DType *input_x_Ptr = (__ubuf__ typename TileDataS1::DType *)__cce_get_tile_ptr(input_x.data());
    __ubuf__ typename ReduceTileD1::DType *local_max_Ptr = (__ubuf__ typename ReduceTileD1::DType *)__cce_get_tile_ptr(local_max.data());
    __ubuf__ typename ReduceTileD1::DType *local_sum_Ptr = (__ubuf__ typename ReduceTileD1::DType *)__cce_get_tile_ptr(local_sum.data());
    __ubuf__ typename ReduceTileD1::DType *new_global_max_Ptr = (__ubuf__ typename ReduceTileD1::DType *)__cce_get_tile_ptr(new_global_max.data());
    __ubuf__ typename ReduceTileD1::DType *new_global_sum_Ptr = (__ubuf__ typename ReduceTileD1::DType *)__cce_get_tile_ptr(new_global_sum.data());
    __ubuf__ typename ReduceTileD1::DType *exp_max_Ptr = (__ubuf__ typename ReduceTileD1::DType *)__cce_get_tile_ptr(exp_max.data());
    __ubuf__ typename TileDataH_NZ_T::DType *nz_buffer_Ptr = (__ubuf__ typename TileDataH_NZ_T::DType *)__cce_get_tile_ptr(nzConvBuffer.data());

    constexpr float scale = constexpr_inv_sqrt(HEAD_SIZE);
    const bool is_vec1 = static_cast<size_t>(get_subblockid());

    unsigned ubM = TileDataD2::Cols;
    unsigned ubN = TileDataD2::Rows;

        __ubuf__ half *nz_buffer_Ptr2 = nz_buffer_Ptr + 16;
        __ubuf__ half *nz_buffer_Ptr3 = nz_buffer_Ptr + ubN/2*16;
        __ubuf__ half *nz_buffer_Ptr4 = nz_buffer_Ptr + ubN/2*16 + 16;
        uint64_t VSSTB_CONFIG = ((ubN + 1) << 16) | 2u;

    __VEC_SCOPE__{
        vector_f32 vreg_x_sum_even;
        vector_f32 vreg_x_sum_odd;
        vector_f32 vreg_x_sum0;
        vector_f16 vreg_x_exp_even_f16;
        vector_f16 vreg_x_exp_odd_f16;
        vector_f32 vreg_x_exp_even;
        vector_f32 vreg_x_exp_odd;
        vector_f32 vreg_x_f32_a;
        vector_f32 vreg_x_f32_b;
        vector_f32 vreg_x_exp_even_1;
        vector_f32 vreg_x_exp_odd_1;
        vector_f16 vreg_x_exp_even_f16_1;
        vector_f16 vreg_x_exp_odd_f16_1;
        vector_f32 vreg_x_f32_1_a;
        vector_f32 vreg_x_f32_1_b;
        vector_f16 vreg_x_exp_f16_pack;
        vector_f16 vreg_x_exp_f16_1_pack;
        vector_f16 vreg_x_exp_f16_packa;
        vector_f16 vreg_x_exp_f16_1_packa;
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
        vector_f32 max_0a, max_1a, max_2a, max_3a;

        __ubuf__ float *src0_ub = (__ubuf__ float *)input_x_Ptr;

        preg_108 = pset_b16(PAT_ALL);
        vector_bool preg_low_half = pset_b16(PAT_VL64);
        // 128,64 -> 1,64

        __ubuf__ float *p0 = src0_ub + 4 * 64;
        __ubuf__ float *p1 = src0_ub + 5 * 64;
        __ubuf__ float *p2 = src0_ub + 6 * 64;
        __ubuf__ float *p3 = src0_ub + 7 * 64;

        vlds(max_0a, src0_ub, 0 * 64, NORM);
        vlds(max_1a, src0_ub, 1 * 64, NORM);
        vlds(max_2a, src0_ub, 2 * 64, NORM);
        vlds(max_3a, src0_ub, 3 * 64, NORM);

        RegTensor<float> v_row;
        for (uint16_t row = 4; row < uint16_t(ubN); row += 4) {
            vlds(v_row, p0, 4 * 64, NORM, POST_UPDATE);
            vmax(max_0a, max_0a, v_row, preg_108, MODE_ZEROING);
            vlds(v_row, p1, 4 * 64, NORM, POST_UPDATE);
            vmax(max_1a, max_1a, v_row, preg_108, MODE_ZEROING);
            vlds(v_row, p2, 4 * 64, NORM, POST_UPDATE);
            vmax(max_2a, max_2a, v_row, preg_108, MODE_ZEROING);
            vlds(v_row, p3, 4 * 64, NORM, POST_UPDATE);
            vmax(max_3a, max_3a, v_row, preg_108, MODE_ZEROING);
        }

        vmax(max_0a, max_0a, max_1a, preg_108, MODE_ZEROING);
        vmax(max_2a, max_2a, max_3a, preg_108, MODE_ZEROING);
        vmax(max_0a, max_0a, max_2a, preg_108, MODE_ZEROING);

        vsts(max_0a, (__ubuf__ float *)new_global_max_Ptr, 0, NORM_B16, preg_108);
        vmuls(max_0a, max_0a, scale, preg_108);

        vdup(vreg_x_sum_even, 0, preg_134, MODE_ZEROING);
        vdup(vreg_x_sum_odd, 0, preg_134, MODE_ZEROING);

        for (uint16_t i0 = 0; i0 < uint16_t(ubN / 4) ; ++i0) { //128,64
            vld(vreg_x_f32_a, input_x_Ptr, vag_b32(128), NORM);
            vld(vreg_x_f32_b, ((__ubuf__ float *) input_x_Ptr + 64), vag_b32(128), NORM);
            vld(vreg_x_f32_1_a, ((__ubuf__ float *) input_x_Ptr + ubN*ubM/2), vag_b32(128), NORM);
            vld(vreg_x_f32_1_b, ((__ubuf__ float *) input_x_Ptr + ubN*ubM/2 + 64), vag_b32(128), NORM);

            vmuls(vreg_x_f32_a, vreg_x_f32_a, scale, preg_108);
            vmuls(vreg_x_f32_b, vreg_x_f32_b, scale, preg_108);
            vexpdif(vreg_x_exp_even, vreg_x_f32_a, max_0a, preg_134, PART_ODD);
            vexpdif(vreg_x_exp_odd, vreg_x_f32_b, max_0a, preg_134, PART_ODD);
            vadd(vreg_x_sum_even, vreg_x_exp_even, vreg_x_sum_even, preg_134, MODE_ZEROING);
            vadd(vreg_x_sum_odd, vreg_x_exp_odd, vreg_x_sum_odd, preg_134, MODE_ZEROING);

            vmuls(vreg_x_f32_1_a, vreg_x_f32_1_a, scale, preg_108);
            vmuls(vreg_x_f32_1_b, vreg_x_f32_1_b, scale, preg_108);
            vexpdif(vreg_x_exp_even_1, vreg_x_f32_1_a, max_0a, preg_134, PART_ODD);
            vexpdif(vreg_x_exp_odd_1, vreg_x_f32_1_b, max_0a, preg_134, PART_ODD);
            vadd(vreg_x_sum_even, vreg_x_exp_even_1, vreg_x_sum_even, preg_134, MODE_ZEROING);
            vadd(vreg_x_sum_odd, vreg_x_exp_odd_1, vreg_x_sum_odd, preg_134, MODE_ZEROING);

            // change from vmulscvt to vcvt performance not better in this case
            vmulscvt(vreg_x_exp_even_f16, vreg_x_exp_even, 1.0f, preg_100, PART_EVEN);
            vmulscvt(vreg_x_exp_odd_f16, vreg_x_exp_odd, 1.0f, preg_101, PART_EVEN);
            vmulscvt(vreg_x_exp_even_f16_1, vreg_x_exp_even_1, 1.0f, preg_135, PART_EVEN);
            vmulscvt(vreg_x_exp_odd_f16_1, vreg_x_exp_odd_1, 1.0f, preg_136, PART_EVEN);
            // vcvt(vreg_x_exp_even_f16, vreg_x_exp_even, preg_108, ROUND_R, RS_ENABLE, PART_EVEN, MODE_ZEROING);
            // vcvt(vreg_x_exp_odd_f16, vreg_x_exp_odd, preg_108, ROUND_R, RS_ENABLE, PART_EVEN, MODE_ZEROING);
            // vcvt(vreg_x_exp_even_f16_1, vreg_x_exp_even_1, preg_108, ROUND_R, RS_ENABLE, PART_EVEN, MODE_ZEROING);
            // vcvt(vreg_x_exp_odd_f16_1, vreg_x_exp_odd_1, preg_108, ROUND_R, RS_ENABLE, PART_EVEN, MODE_ZEROING);
        
            vpack((vector_u16&)vreg_x_exp_even_f16, (vector_u32&)vreg_x_exp_even_f16, LOWER);
            vsstb(vreg_x_exp_even_f16, ((__ubuf__ half *&) nz_buffer_Ptr), VSSTB_CONFIG, preg_low_half, POST_UPDATE);
            vpack((vector_u16&)vreg_x_exp_odd_f16, (vector_u32&)vreg_x_exp_odd_f16, LOWER);
            vsstb(vreg_x_exp_odd_f16, ((__ubuf__ half *&) nz_buffer_Ptr2), VSSTB_CONFIG, preg_low_half, POST_UPDATE);
            vpack((vector_u16&)vreg_x_exp_even_f16_1, (vector_u32&)vreg_x_exp_even_f16_1, LOWER);
            vsstb(vreg_x_exp_even_f16_1, ((__ubuf__ half *&) nz_buffer_Ptr3), VSSTB_CONFIG, preg_low_half, POST_UPDATE);
            vpack((vector_u16&)vreg_x_exp_odd_f16_1, (vector_u32&)vreg_x_exp_odd_f16_1, LOWER);
            vsstb(vreg_x_exp_odd_f16_1, ((__ubuf__ half *&) nz_buffer_Ptr4), VSSTB_CONFIG, preg_low_half, POST_UPDATE);
        }

        vadd(vreg_x_sum0, vreg_x_sum_odd, vreg_x_sum_even, preg_134, MODE_ZEROING);
        vsts(vreg_x_sum0, ((__ubuf__ float *&) local_sum_Ptr), 0, NORM_B32, preg_134);
        vsts(vreg_x_sum0, ((__ubuf__ float *&) new_global_sum_Ptr), 0, NORM_B32, preg_134);
    }
#if skip_rescale
    constexpr int SYNC_PERIOD = kFaCvFifoConsSyncPeriod;
    const bool should_wait_consume = aiv_should_wait_consumption<SKIP_STATUS_FIFO_SIZE, SYNC_PERIOD>(sync_iter);
    if (should_wait_consume) {
        wait_intra_block(PIPE_S, FftsBufferFlag::SS_BUF_READY + 1);
    }
    write_skip_status_slot(tile_id, is_vec1, 0);
    set_intra_block(PIPE_S, FftsBufferFlag::SS_BUF_READY);
#else
    (void)tile_id;
    (void)sync_iter;
    (void)is_vec1;
#endif
    (void)last_tile;
#else
    (void)local_max;
    (void)exp_max;
    (void)local_sum;
    (void)last_tile;

    constexpr float scale = constexpr_inv_sqrt(HEAD_SIZE);
    const bool is_vec1 = static_cast<size_t>(get_subblockid());

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
#if skip_rescale
    constexpr int SYNC_PERIOD = kFaCvFifoConsSyncPeriod;
    const bool should_wait_consume = aiv_should_wait_consumption<SKIP_STATUS_FIFO_SIZE, SYNC_PERIOD>(sync_iter);
    if (should_wait_consume) {
        wait_intra_block(PIPE_S, FftsBufferFlag::SS_BUF_READY + 1);
    }
    write_skip_status_slot(tile_id, is_vec1, 0);
    set_intra_block(PIPE_S, FftsBufferFlag::SS_BUF_READY);
#else
    (void)tile_id;
    (void)sync_iter;
    (void)is_vec1;
#endif
#endif
}

template <int HEAD_SIZE, bool CAUSAL_MASK, typename ReduceTileD1, typename TileDataD2, typename TileDataS1, typename TileDataH_NZ_T, int MODE>
__tf__ AICORE inline void softmax_opt_fa_dn_init_asc_impl(int tile_id, int sync_iter, TileDataD2 __out__ x_exp,
                                                      TileDataS1 __in__ input_x,
                                                      ReduceTileD1 __out__ local_max, ReduceTileD1 __out__ local_sum,
                                                      ReduceTileD1 __out__ new_global_max,
                                                      ReduceTileD1 __out__ new_global_sum, ReduceTileD1 __out__ exp_max,
                                                      TileDataS1 __out__ tmp_float, TileDataS1 __out__ p_tile_f32,
                                                      TileDataS1 triu, TileDataH_NZ_T nzConvBuffer, int s0_index, int s1_index, bool last_tile)
{
#if USE_MANUAL
    __ubuf__ typename TileDataD2::DType *x_exp_Ptr = (__ubuf__ typename TileDataD2::DType *)__cce_get_tile_ptr(x_exp.data());
    __ubuf__ typename TileDataS1::DType *input_x_Ptr = (__ubuf__ typename TileDataS1::DType *)__cce_get_tile_ptr(input_x.data());
    __ubuf__ typename ReduceTileD1::DType *local_max_Ptr = (__ubuf__ typename ReduceTileD1::DType *)__cce_get_tile_ptr(local_max.data());
    __ubuf__ typename ReduceTileD1::DType *local_sum_Ptr = (__ubuf__ typename ReduceTileD1::DType *)__cce_get_tile_ptr(local_sum.data());
    __ubuf__ typename ReduceTileD1::DType *new_global_max_Ptr = (__ubuf__ typename ReduceTileD1::DType *)__cce_get_tile_ptr(new_global_max.data());
    __ubuf__ typename ReduceTileD1::DType *new_global_sum_Ptr = (__ubuf__ typename ReduceTileD1::DType *)__cce_get_tile_ptr(new_global_sum.data());
    __ubuf__ typename ReduceTileD1::DType *exp_max_Ptr = (__ubuf__ typename ReduceTileD1::DType *)__cce_get_tile_ptr(exp_max.data());
    __ubuf__ typename TileDataH_NZ_T::DType *nz_buffer_Ptr = (__ubuf__ typename TileDataH_NZ_T::DType *)__cce_get_tile_ptr(nzConvBuffer.data());

    constexpr float scale = constexpr_inv_sqrt(HEAD_SIZE);
    const bool is_vec1 = static_cast<size_t>(get_subblockid());

    unsigned ubM = TileDataD2::Cols;
    unsigned ubN = TileDataD2::Rows;

    __ubuf__ half *nz_buffer_Ptr2 = nz_buffer_Ptr + ubN*4;

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
        __ubuf__ float *src0_ub1 = src0_ub + 64;
        __ubuf__ float *src0_ub2 = src0_ub + 128;
        __ubuf__ float *src0_ub3 = src0_ub + 192;

        vbr(max_0a, 0);
        vbr(max_1a, 0);
        vbr(max_2a, 0);
        vbr(max_3a, 0);

        preg_108 = pset_b16(PAT_ALL);
        // 128,64 -> 1,64

        for (uint16_t iter_m = 0; iter_m < uint16_t(ubN / 4) ; ++iter_m) {
            vlds(src_00a, src0_ub,       256, NORM, POST_UPDATE);
            vlds(src_01a, src0_ub1,      256, NORM, POST_UPDATE);
            vlds(src_02a, src0_ub2,      256, NORM, POST_UPDATE);
            vlds(src_03a, src0_ub3,      256, NORM, POST_UPDATE);

            vmax(max_0a, max_0a, src_00a, preg_108);
            vmax(max_1a, max_1a, src_01a, preg_108);            
            vmax(max_2a, max_2a, src_02a, preg_108);
            vmax(max_3a, max_3a, src_03a, preg_108);
        }

        vmax(max_0a, max_0a, max_2a, preg_108);
        vmax(max_1a, max_1a, max_3a, preg_108);
        vmax(max_0a, max_0a, max_1a, preg_108);
        vsts(max_0a, (__ubuf__ float *)new_global_max_Ptr, 0, NORM_B16, preg_108);  //TODO: ASC do the muls and then store
        vmuls(max_0a, max_0a, scale, preg_108);

        vdup(vreg_x_sum_even, 0, preg_134, MODE_ZEROING);
        vdup(vreg_x_sum_odd, 0, preg_134, MODE_ZEROING);
        vdup(vreg_x_sum_1_even, 0, preg_134, MODE_ZEROING);
        vdup(vreg_x_sum_1_odd, 0, preg_134, MODE_ZEROING);

        // __ubuf__ float *input_x_Ptr1 = input_x_Ptr + ubN*ubM/2;
        // __ubuf__ float *input_x_Ptr2 = input_x_Ptr + ubN*ubM/4;
        // __ubuf__ float *input_x_Ptr3 = input_x_Ptr + ubN*ubM/2 + ubN*ubM/4;

        for (uint16_t i0 = 0; i0 < uint16_t(ubN / 4) ; ++i0) { //128,64
            vld(vreg_x_f32_a, input_x_Ptr, vag_b32(64), NORM);
            vld(vreg_x_f32_b, ((__ubuf__ float *) input_x_Ptr + ubN*ubM/2), vag_b32(64), NORM);
            vld(vreg_x_f32_1_a, ((__ubuf__ float *) input_x_Ptr + ubN*ubM/4), vag_b32(64), NORM);
            vld(vreg_x_f32_1_b, ((__ubuf__ float *) input_x_Ptr + ubN*ubM/2 + ubN*ubM/4), vag_b32(64), NORM);
            // worse performance
            // vlds(vreg_x_f32_a,   input_x_Ptr,    64, NORM, POST_UPDATE);
            // vlds(vreg_x_f32_b,   input_x_Ptr1,   64, NORM, POST_UPDATE);
            // vlds(vreg_x_f32_1_a, input_x_Ptr2,   64, NORM, POST_UPDATE);
            // vlds(vreg_x_f32_1_b, input_x_Ptr3,   64, NORM, POST_UPDATE);

            vmuls(vreg_x_f32_a, vreg_x_f32_a, scale, preg_108);
            vmuls(vreg_x_f32_b, vreg_x_f32_b, scale, preg_108);
            vmuls(vreg_x_f32_1_a, vreg_x_f32_1_a, scale, preg_108);
            vmuls(vreg_x_f32_1_b, vreg_x_f32_1_b, scale, preg_108);
            // vexpdif(vreg_x_exp_even, vreg_x_f32_a, max_0a, preg_134, PART_ODD);
            // vexpdif(vreg_x_exp_odd, vreg_x_f32_b, max_0a, preg_134, PART_ODD);

            // vmuls(vreg_x_f32_1_a, vreg_x_f32_1_a, scale, preg_108);
            // vmuls(vreg_x_f32_1_b, vreg_x_f32_1_b, scale, preg_108);
            vexpdif(vreg_x_exp_even, vreg_x_f32_a, max_0a, preg_134, PART_ODD);
            vexpdif(vreg_x_exp_odd, vreg_x_f32_b, max_0a, preg_134, PART_ODD);
            vexpdif(vreg_x_exp_even_1, vreg_x_f32_1_a, max_0a, preg_134, PART_ODD);
            vexpdif(vreg_x_exp_odd_1, vreg_x_f32_1_b, max_0a, preg_134, PART_ODD);

            // vmulscvt change to vcvt has performance gain
            // vmulscvt(vreg_x_exp_even_f16, vreg_x_exp_even, 1.0f, preg_100, PART_EVEN);
            // vmulscvt(vreg_x_exp_odd_f16, vreg_x_exp_odd, 1.0f, preg_101, PART_EVEN);
            // vmulscvt(vreg_x_exp_even_f16_1, vreg_x_exp_even_1, 1.0f, preg_135, PART_EVEN);
            // vmulscvt(vreg_x_exp_odd_f16_1, vreg_x_exp_odd_1, 1.0f, preg_136, PART_EVEN);
            vcvt(vreg_x_exp_even_f16, vreg_x_exp_even, preg_108, ROUND_R, RS_ENABLE, PART_EVEN, MODE_ZEROING);
            vcvt(vreg_x_exp_odd_f16, vreg_x_exp_odd, preg_108, ROUND_R, RS_ENABLE, PART_EVEN, MODE_ZEROING);
            vcvt(vreg_x_exp_even_f16_1, vreg_x_exp_even_1, preg_108, ROUND_R, RS_ENABLE, PART_EVEN, MODE_ZEROING);
            vcvt(vreg_x_exp_odd_f16_1, vreg_x_exp_odd_1, preg_108, ROUND_R, RS_ENABLE, PART_EVEN, MODE_ZEROING);
            
            // change to ASC position performance even worse
            vdintlv(vreg_x_exp_f16_pack, vreg_x_exp_f16_packa, vreg_x_exp_even_f16, vreg_x_exp_odd_f16);
            vdintlv(vreg_x_exp_f16_1_pack, vreg_x_exp_f16_1_packa, vreg_x_exp_even_f16_1, vreg_x_exp_odd_f16_1);
            vsstb(vreg_x_exp_f16_pack, ((__ubuf__ half *&) nz_buffer_Ptr), 0x410001, preg_108, POST_UPDATE);
            vsstb(vreg_x_exp_f16_1_pack, ((__ubuf__ half *&) nz_buffer_Ptr2), 0x410001, preg_108, POST_UPDATE);

            vadd(vreg_x_sum_even, vreg_x_exp_even, vreg_x_sum_even, preg_134, MODE_ZEROING);
            vadd(vreg_x_sum_odd, vreg_x_exp_odd, vreg_x_sum_odd, preg_134, MODE_ZEROING);

            vadd(vreg_x_sum_1_even, vreg_x_exp_even_1, vreg_x_sum_1_even, preg_134, MODE_ZEROING);
            vadd(vreg_x_sum_1_odd, vreg_x_exp_odd_1, vreg_x_sum_1_odd, preg_134, MODE_ZEROING);
        }

        vadd(vreg_x_sum0, vreg_x_sum_odd, vreg_x_sum_even, preg_134, MODE_ZEROING);
        vadd(vreg_x_sum1, vreg_x_sum_1_odd, vreg_x_sum_1_even, preg_134, MODE_ZEROING);
        vadd(vreg_x_sum0, vreg_x_sum0, vreg_x_sum1, preg_134, MODE_ZEROING);
        vsts(vreg_x_sum0, ((__ubuf__ float *&) local_sum_Ptr), 0, NORM_B32, preg_134);  //ASC no write out local_sum, just 1 cycle increase
        
        vsts(vreg_x_sum0, ((__ubuf__ float *&) new_global_sum_Ptr), 0, NORM_B32, preg_134);
    }
#if skip_rescale
    constexpr int SYNC_PERIOD = kFaCvFifoConsSyncPeriod;
    const bool should_wait_consume = aiv_should_wait_consumption<SKIP_STATUS_FIFO_SIZE, SYNC_PERIOD>(sync_iter);
    if (should_wait_consume) {
        wait_intra_block(PIPE_S, FftsBufferFlag::SS_BUF_READY + 1);
    }
    write_skip_status_slot(tile_id, is_vec1, 0);
    set_intra_block(PIPE_S, FftsBufferFlag::SS_BUF_READY);
#else
    (void)tile_id;
    (void)sync_iter;
    (void)is_vec1;
#endif
    (void)last_tile;
#else
    (void)local_max;
    (void)exp_max;
    (void)local_sum;
    (void)last_tile;

    constexpr float scale = constexpr_inv_sqrt(HEAD_SIZE);
    const bool is_vec1 = static_cast<size_t>(get_subblockid());

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
#if skip_rescale
    constexpr int SYNC_PERIOD = kFaCvFifoConsSyncPeriod;
    const bool should_wait_consume = aiv_should_wait_consumption<SKIP_STATUS_FIFO_SIZE, SYNC_PERIOD>(sync_iter);
    if (should_wait_consume) {
        wait_intra_block(PIPE_S, FftsBufferFlag::SS_BUF_READY + 1);
    }
    write_skip_status_slot(tile_id, is_vec1, 0);
    set_intra_block(PIPE_S, FftsBufferFlag::SS_BUF_READY);
#else
    (void)tile_id;
    (void)sync_iter;
    (void)is_vec1;
#endif
#endif
}

template <int HEAD_SIZE, bool CAUSAL_MASK, typename ReduceTileD1, typename TileDataD2, typename TileDataS1, typename TileDataH_NZ_T, int MODE>
__tf__ AICORE inline void softmax_opt_fa_dn_not_init_impl(
    int tile_id, int sync_iter, TileDataD2 __out__ x_exp, TileDataS1 __in__ input_x,
    ReduceTileD1 __out__ local_max, ReduceTileD1 __out__ local_sum, ReduceTileD1 __out__ new_global_max,
    ReduceTileD1 __out__ new_global_sum, ReduceTileD1 __out__ exp_max, TileDataS1 __out__ tmp_float,
    TileDataS1 __out__ p_tile_f32, TileDataS1 triu, TileDataH_NZ_T nzConvBuffer, int s0_index, int s1_index, bool last_tile)
{
#if USE_MANUAL
    __ubuf__ typename TileDataD2::DType *x_exp_Ptr = (__ubuf__ typename TileDataD2::DType *)__cce_get_tile_ptr(x_exp.data());
    __ubuf__ typename TileDataS1::DType *input_x_Ptr = (__ubuf__ typename TileDataS1::DType *)__cce_get_tile_ptr(input_x.data());
    __ubuf__ typename ReduceTileD1::DType *local_max_Ptr = (__ubuf__ typename ReduceTileD1::DType *)__cce_get_tile_ptr(local_max.data());
    __ubuf__ typename ReduceTileD1::DType *local_sum_Ptr = (__ubuf__ typename ReduceTileD1::DType *)__cce_get_tile_ptr(local_sum.data());
    __ubuf__ typename ReduceTileD1::DType *new_global_max_Ptr = (__ubuf__ typename ReduceTileD1::DType *)__cce_get_tile_ptr(new_global_max.data());
    __ubuf__ typename ReduceTileD1::DType *new_global_sum_Ptr = (__ubuf__ typename ReduceTileD1::DType *)__cce_get_tile_ptr(new_global_sum.data());
    __ubuf__ typename ReduceTileD1::DType *exp_max_Ptr = (__ubuf__ typename ReduceTileD1::DType *)__cce_get_tile_ptr(exp_max.data());
    __ubuf__ typename TileDataH_NZ_T::DType *nz_buffer_Ptr = (__ubuf__ typename TileDataH_NZ_T::DType *)__cce_get_tile_ptr(nzConvBuffer.data());

    constexpr float scale = constexpr_inv_sqrt(HEAD_SIZE);
    const bool is_vec1 = static_cast<size_t>(get_subblockid());
#if skip_rescale
    constexpr float threshold = 8.0f / scale;
    ReduceTileD1 deltaMaxTile;
    const uint64_t delta_max_offset = 254U * 1024U;
    TASSIGN(deltaMaxTile, delta_max_offset);
    __ubuf__ typename ReduceTileD1::DType *delta_max_Ptr =
        (__ubuf__ typename ReduceTileD1::DType *)__cce_get_tile_ptr(deltaMaxTile.data());
#endif

    unsigned ubM = TileDataD2::Cols;
    unsigned ubN = TileDataD2::Rows;

        __ubuf__ half *nz_buffer_Ptr2 = nz_buffer_Ptr + 16;
        __ubuf__ half *nz_buffer_Ptr3 = nz_buffer_Ptr + ubN/2*16;
        __ubuf__ half *nz_buffer_Ptr4 = nz_buffer_Ptr + ubN/2*16 + 16;
        uint64_t VSSTB_CONFIG = ((ubN + 1) << 16) | 2u;

    __VEC_SCOPE__{
        vector_f32 vreg_x_sum_even;
        vector_f32 vreg_x_sum_odd;
        vector_f32 vreg_x_sum0;
        vector_f16 vreg_x_exp_even_f16;
        vector_f16 vreg_x_exp_odd_f16;
        vector_f32 vreg_x_exp_even;
        vector_f32 vreg_x_exp_odd;
        vector_f32 vreg_x_f32_a;
        vector_f32 vreg_x_f32_b;
        vector_f32 vreg_x_exp_even_1;
        vector_f32 vreg_x_exp_odd_1;
        vector_f16 vreg_x_exp_even_f16_1;
        vector_f16 vreg_x_exp_odd_f16_1;
        vector_f32 vreg_x_f32_1_a;
        vector_f32 vreg_x_f32_1_b;
        vector_f32 vreg_x_max_f32_b;
        vector_f16 vreg_x_exp_f16_pack;
        vector_f16 vreg_x_exp_f16_1_pack;
        vector_f16 vreg_x_exp_f16_packa;
        vector_f16 vreg_x_exp_f16_1_packa;
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
        vector_bool preg_108;
        vector_f32 max_0a, max_1a, max_2a, max_3a;
#if skip_rescale
        vector_f32 vreg_delta_max;
#endif

        __ubuf__ float *src0_ub = (__ubuf__ float *)input_x_Ptr;

        preg_108 = pset_b16(PAT_ALL);
        vector_bool preg_low_half = pset_b16(PAT_VL64);
        // 128,64 -> 1,64

        vlds(vreg_x_max_f32_b, new_global_max_Ptr, 0, NORM);

        __ubuf__ float *p0 = src0_ub + 4 * 64;
        __ubuf__ float *p1 = src0_ub + 5 * 64;
        __ubuf__ float *p2 = src0_ub + 6 * 64;
        __ubuf__ float *p3 = src0_ub + 7 * 64;

        vlds(max_0a, src0_ub, 0 * 64, NORM);
        vlds(max_1a, src0_ub, 1 * 64, NORM);
        vlds(max_2a, src0_ub, 2 * 64, NORM);
        vlds(max_3a, src0_ub, 3 * 64, NORM);

        RegTensor<float> v_row;
        for (uint16_t row = 4; row < uint16_t(ubN); row += 4) {
            vlds(v_row, p0, 4 * 64, NORM, POST_UPDATE);
            vmax(max_0a, max_0a, v_row, preg_108, MODE_ZEROING);
            vlds(v_row, p1, 4 * 64, NORM, POST_UPDATE);
            vmax(max_1a, max_1a, v_row, preg_108, MODE_ZEROING);
            vlds(v_row, p2, 4 * 64, NORM, POST_UPDATE);
            vmax(max_2a, max_2a, v_row, preg_108, MODE_ZEROING);
            vlds(v_row, p3, 4 * 64, NORM, POST_UPDATE);
            vmax(max_3a, max_3a, v_row, preg_108, MODE_ZEROING);
        }

        vmax(max_0a, max_0a, max_1a, preg_108, MODE_ZEROING);
        vmax(max_2a, max_2a, max_3a, preg_108, MODE_ZEROING);
        vmax(max_0a, max_0a, max_2a, preg_108, MODE_ZEROING);

        vmax(max_0a, max_0a, vreg_x_max_f32_b, preg_108);    //for FA4 skip logic, cannot optimize compared with ASC

#if skip_rescale
        vsub(vreg_delta_max, max_0a, vreg_x_max_f32_b, preg_108);
        vsts(vreg_delta_max, delta_max_Ptr, 0, NORM_B32, preg_108);
        vsts(max_0a, local_max_Ptr, 0, NORM_B16, preg_108);
#else
        vsts(max_0a, new_global_max_Ptr, 0, NORM_B16, preg_108);  //just copy
#endif

        vmuls(max_0a, max_0a, scale, preg_108);
        vmuls(vreg_x_max_f32_b, vreg_x_max_f32_b, scale, preg_108);
        vexpdif(vreg_x_max_f32_b, vreg_x_max_f32_b, max_0a, preg_134, PART_ODD);

        vsts(vreg_x_max_f32_b, exp_max_Ptr, 0, NORM_B16, preg_108);     //Store input max in local_max
        
        vdup(vreg_x_sum_even, 0, preg_134, MODE_ZEROING);
        vdup(vreg_x_sum_odd, 0, preg_134, MODE_ZEROING);

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

            vadd(vreg_x_sum_even, vreg_x_exp_even, vreg_x_sum_even, preg_134, MODE_ZEROING);
            vadd(vreg_x_sum_odd, vreg_x_exp_odd, vreg_x_sum_odd, preg_134, MODE_ZEROING);
            vadd(vreg_x_sum_even, vreg_x_exp_even_1, vreg_x_sum_even, preg_134, MODE_ZEROING);
            vadd(vreg_x_sum_odd, vreg_x_exp_odd_1, vreg_x_sum_odd, preg_134, MODE_ZEROING);

            vmulscvt(vreg_x_exp_even_f16, vreg_x_exp_even, 1.0f, preg_100, PART_EVEN);
            vpack((vector_u16&)vreg_x_exp_even_f16, (vector_u32&)vreg_x_exp_even_f16, LOWER);
            vsstb(vreg_x_exp_even_f16, ((__ubuf__ half *&) nz_buffer_Ptr), VSSTB_CONFIG, preg_low_half, POST_UPDATE);
            vmulscvt(vreg_x_exp_odd_f16, vreg_x_exp_odd, 1.0f, preg_101, PART_EVEN);
            vpack((vector_u16&)vreg_x_exp_odd_f16, (vector_u32&)vreg_x_exp_odd_f16, LOWER);
            vsstb(vreg_x_exp_odd_f16, ((__ubuf__ half *&) nz_buffer_Ptr2), VSSTB_CONFIG, preg_low_half, POST_UPDATE);
            vmulscvt(vreg_x_exp_even_f16_1, vreg_x_exp_even_1, 1.0f, preg_135, PART_EVEN);
            vpack((vector_u16&)vreg_x_exp_even_f16_1, (vector_u32&)vreg_x_exp_even_f16_1, LOWER);
            vsstb(vreg_x_exp_even_f16_1, ((__ubuf__ half *&) nz_buffer_Ptr3), VSSTB_CONFIG, preg_low_half, POST_UPDATE);
            vmulscvt(vreg_x_exp_odd_f16_1, vreg_x_exp_odd_1, 1.0f, preg_136, PART_EVEN);
            vpack((vector_u16&)vreg_x_exp_odd_f16_1, (vector_u32&)vreg_x_exp_odd_f16_1, LOWER);
            vsstb(vreg_x_exp_odd_f16_1, ((__ubuf__ half *&) nz_buffer_Ptr4), VSSTB_CONFIG, preg_low_half, POST_UPDATE);
            
        }

        vadd(vreg_x_sum0, vreg_x_sum_odd, vreg_x_sum_even, preg_134, MODE_ZEROING);
        vsts(vreg_x_sum0, ((__ubuf__ float *&) local_sum_Ptr), 0, NORM_B32, preg_134);  //no need?

        // mem_bar(VST_VLD);
        // vector_f32 vreg_max0;
        // vector_f32 vreg_max1;
        // vector_f32 vreg_exp_max;
        // vector_f32 vreg_exp;
        vector_f32 vreg_l0;
        // vector_f32 vreg_l1;
        
        // vlds(vreg_exp_max, ((__ubuf__ float *) exp_max_Ptr), 0, NORM);
        vlds(vreg_l0, ((__ubuf__ float *) new_global_sum_Ptr), 0, NORM);
        // vmul(vreg_l0, vreg_exp_max, vreg_l0, preg_134, MODE_ZEROING);
        vmul(vreg_l0, vreg_x_max_f32_b, vreg_l0, preg_134, MODE_ZEROING);
        vadd(vreg_l0, vreg_l0, vreg_x_sum0, preg_134, MODE_ZEROING);
        vsts(vreg_l0, ((__ubuf__ float *) new_global_sum_Ptr), 0, NORM_B32, preg_134);
        
    }
#if skip_rescale
    constexpr int SYNC_PERIOD = kFaCvFifoConsSyncPeriod;
    const bool should_wait_consume = aiv_should_wait_consumption<SKIP_STATUS_FIFO_SIZE, SYNC_PERIOD>(sync_iter);

    if (last_tile) {
        if (should_wait_consume) {
            wait_intra_block(PIPE_S, FftsBufferFlag::SS_BUF_READY + 1);
        }
        write_skip_status_slot(tile_id, is_vec1, 0);
        set_intra_block(PIPE_S, FftsBufferFlag::SS_BUF_READY);
        TMULS(new_global_max, local_max, 1.0f);
    } else {
        using TileDataScalar = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, 1, 1>;
        TileDataScalar ctrlTile;
        const uint64_t ctrl_tile_offset = 255U * 1024U;
        TASSIGN(ctrlTile, ctrl_tile_offset);
        TROWMAX(ctrlTile, deltaMaxTile, local_sum);
        const uint64_t ss_buf_pingpong = tile_id % 2;
        set_flag(PIPE_V, PIPE_S, EVENT_ID7 + ss_buf_pingpong);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID7 + ss_buf_pingpong);

        if (should_wait_consume) {
            wait_intra_block(PIPE_S, FftsBufferFlag::SS_BUF_READY + 1);
        }

        if (*(ctrlTile.data()) < threshold) {
            write_skip_status_slot(tile_id, is_vec1, 1);
        } else {
            write_skip_status_slot(tile_id, is_vec1, 0);
        }

        set_intra_block(PIPE_S, FftsBufferFlag::SS_BUF_READY);

        if (*(ctrlTile.data()) > threshold) {
            TMULS(new_global_max, local_max, 1.0f);
        }
    }
#else
    (void)tile_id;
    (void)sync_iter;
    (void)is_vec1;
    (void)last_tile;
#endif
#else
    constexpr float scale = constexpr_inv_sqrt(HEAD_SIZE);
    const bool is_vec1 = static_cast<size_t>(get_subblockid());
    constexpr float threshold = 8.0f / scale;

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
#if skip_rescale
    constexpr int SYNC_PERIOD = kFaCvFifoConsSyncPeriod;
    const bool should_wait_consume = aiv_should_wait_consumption<SKIP_STATUS_FIFO_SIZE, SYNC_PERIOD>(sync_iter);

    if (last_tile) {
        if (should_wait_consume) {
            wait_intra_block(PIPE_S, FftsBufferFlag::SS_BUF_READY + 1);
        }
        write_skip_status_slot(tile_id, is_vec1, 0);
        set_intra_block(PIPE_S, FftsBufferFlag::SS_BUF_READY);
        TMULS(new_global_max, local_max, 1.0f);
    } else {
        using TileDataScalar = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, 1, 1>;
        TileDataScalar ctrlTile;
        const uint64_t ctrl_tile_offset = 255U * 1024U;
        TASSIGN(ctrlTile, ctrl_tile_offset);
        const uint64_t delta_max_offset = 254U * 1024U;
        ReduceTileD1 deltaMaxTile;
        TASSIGN(deltaMaxTile, delta_max_offset);
        TSUB(deltaMaxTile, local_max, new_global_max);
        TROWMAX(ctrlTile, deltaMaxTile, local_sum);
        const uint64_t ss_buf_pingpong = tile_id % 2;
        set_flag(PIPE_V, PIPE_S, EVENT_ID7 + ss_buf_pingpong);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID7 + ss_buf_pingpong);

        if (should_wait_consume) {
            wait_intra_block(PIPE_S, FftsBufferFlag::SS_BUF_READY + 1);
        }

        if (*(ctrlTile.data()) < threshold) {
            write_skip_status_slot(tile_id, is_vec1, 1);
        } else {
            write_skip_status_slot(tile_id, is_vec1, 0);
        }

        set_intra_block(PIPE_S, FftsBufferFlag::SS_BUF_READY);

        if (*(ctrlTile.data()) > threshold) {
            TMULS(new_global_max, local_max, 1.0f);
        }
    }
#else
    TMULS(new_global_max, local_max, 1.0f); // just copy
    (void)tile_id;
    (void)sync_iter;
    (void)is_vec1;
    (void)last_tile;
#endif
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

template <int HEAD_SIZE, bool CAUSAL_MASK, typename ReduceTileD1, typename TileDataD2, typename TileDataS1, typename TileDataH_NZ_T, int MODE>
__tf__ AICORE inline void softmax_opt_fa_dn_not_init_asc_impl(
    int tile_id, int sync_iter, TileDataD2 __out__ x_exp, TileDataS1 __in__ input_x,
    ReduceTileD1 __out__ local_max, ReduceTileD1 __out__ local_sum, ReduceTileD1 __out__ new_global_max,
    ReduceTileD1 __out__ new_global_sum, ReduceTileD1 __out__ exp_max, TileDataS1 __out__ tmp_float,
    TileDataS1 __out__ p_tile_f32, TileDataS1 triu, TileDataH_NZ_T nzConvBuffer, int s0_index, int s1_index, bool last_tile)
{
#if USE_MANUAL
    __ubuf__ typename TileDataD2::DType *x_exp_Ptr = (__ubuf__ typename TileDataD2::DType *)__cce_get_tile_ptr(x_exp.data());
    __ubuf__ typename TileDataS1::DType *input_x_Ptr = (__ubuf__ typename TileDataS1::DType *)__cce_get_tile_ptr(input_x.data());
    __ubuf__ typename ReduceTileD1::DType *local_max_Ptr = (__ubuf__ typename ReduceTileD1::DType *)__cce_get_tile_ptr(local_max.data());
    __ubuf__ typename ReduceTileD1::DType *local_sum_Ptr = (__ubuf__ typename ReduceTileD1::DType *)__cce_get_tile_ptr(local_sum.data());
    __ubuf__ typename ReduceTileD1::DType *new_global_max_Ptr = (__ubuf__ typename ReduceTileD1::DType *)__cce_get_tile_ptr(new_global_max.data());
    __ubuf__ typename ReduceTileD1::DType *new_global_sum_Ptr = (__ubuf__ typename ReduceTileD1::DType *)__cce_get_tile_ptr(new_global_sum.data());
    __ubuf__ typename ReduceTileD1::DType *exp_max_Ptr = (__ubuf__ typename ReduceTileD1::DType *)__cce_get_tile_ptr(exp_max.data());
    __ubuf__ typename TileDataH_NZ_T::DType *nz_buffer_Ptr = (__ubuf__ typename TileDataH_NZ_T::DType *)__cce_get_tile_ptr(nzConvBuffer.data());

    constexpr float scale = constexpr_inv_sqrt(HEAD_SIZE);
    const bool is_vec1 = static_cast<size_t>(get_subblockid());
#if skip_rescale
    constexpr float threshold = 8.0f / scale;
    ReduceTileD1 deltaMaxTile;
    const uint64_t delta_max_offset = 254U * 1024U;
    TASSIGN(deltaMaxTile, delta_max_offset);
    __ubuf__ typename ReduceTileD1::DType *delta_max_Ptr =
        (__ubuf__ typename ReduceTileD1::DType *)__cce_get_tile_ptr(deltaMaxTile.data());
#endif

    unsigned ubM = TileDataD2::Cols;
    unsigned ubN = TileDataD2::Rows;

    __ubuf__ half *nz_buffer_Ptr2 = nz_buffer_Ptr + ubN*4;

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
#if skip_rescale
        vector_f32 vreg_delta_max;
#endif

        __ubuf__ float *src0_ub = (__ubuf__ float *)input_x_Ptr;
        __ubuf__ float *src0_ub1 = src0_ub + 64;
        __ubuf__ float *src0_ub2 = src0_ub + 128;
        __ubuf__ float *src0_ub3 = src0_ub + 192;

        vbr(max_0a, 0);
        vbr(max_1a, 0);
        vbr(max_2a, 0);
        vbr(max_3a, 0);

        preg_108 = pset_b16(PAT_ALL);
        // 128,64 -> 1,64

        for (uint16_t iter_m = 0; iter_m < uint16_t(ubN / 4) ; ++iter_m) {
            vlds(src_00a, src0_ub,        256, NORM, POST_UPDATE);
            vlds(src_01a, src0_ub1,      256, NORM, POST_UPDATE);
            vlds(src_02a, src0_ub2,      256, NORM, POST_UPDATE);
            vlds(src_03a, src0_ub3,      256, NORM, POST_UPDATE);

            vmax(max_0a, max_0a, src_00a, preg_108);
            vmax(max_1a, max_1a, src_01a, preg_108);
            vmax(max_2a, max_2a, src_02a, preg_108);
            vmax(max_3a, max_3a, src_03a, preg_108);
        }

        vlds(vreg_x_max_f32_b, new_global_max_Ptr, 0, NORM);
        vmax(max_0a, max_0a, max_2a, preg_108);
        vmax(max_1a, max_1a, max_3a, preg_108);
        vmax(max_0a, max_0a, max_1a, preg_108);

        vmax(max_0a, max_0a, vreg_x_max_f32_b, preg_108);

#if skip_rescale
        vsub(vreg_delta_max, max_0a, vreg_x_max_f32_b, preg_108);
        vsts(vreg_delta_max, delta_max_Ptr, 0, NORM_B32, preg_108);
        vsts(max_0a, local_max_Ptr, 0, NORM_B16, preg_108);
#else
        vsts(max_0a, new_global_max_Ptr, 0, NORM_B16, preg_108);  //just copy
#endif

        vmuls(max_0a, max_0a, scale, preg_108);
        vmuls(vreg_x_max_f32_b, vreg_x_max_f32_b, scale, preg_108);
        vexpdif(vreg_x_max_f32_b, vreg_x_max_f32_b, max_0a, preg_134, PART_ODD);

        vsts(vreg_x_max_f32_b, exp_max_Ptr, 0, NORM_B16, preg_108);     //Store input max in local_max
        
        vdup(vreg_x_sum_even, 0, preg_134, MODE_ZEROING);
        vdup(vreg_x_sum_odd, 0, preg_134, MODE_ZEROING);
        vdup(vreg_x_sum_1_even, 0, preg_134, MODE_ZEROING);
        vdup(vreg_x_sum_1_odd, 0, preg_134, MODE_ZEROING);

        for (uint16_t i0 = 0; i0 < uint16_t(ubN / 4) ; ++i0) { //128,64
            vld(vreg_x_f32_a, input_x_Ptr, vag_b32(64), NORM);
            vld(vreg_x_f32_b, ((__ubuf__ float *) input_x_Ptr + ubN*ubM/2), vag_b32(64), NORM);
            vld(vreg_x_f32_1_a, ((__ubuf__ float *) input_x_Ptr + ubN*ubM/4), vag_b32(64), NORM);
            vld(vreg_x_f32_1_b, ((__ubuf__ float *) input_x_Ptr + ubN*ubM/2 + ubN*ubM/4), vag_b32(64), NORM);

            vmuls(vreg_x_f32_a, vreg_x_f32_a, scale, preg_108);
            vmuls(vreg_x_f32_b, vreg_x_f32_b, scale, preg_108);
            vmuls(vreg_x_f32_1_a, vreg_x_f32_1_a, scale, preg_108);
            vmuls(vreg_x_f32_1_b, vreg_x_f32_1_b, scale, preg_108);
            // vexpdif(vreg_x_exp_even, vreg_x_f32_a, max_0a, preg_134, PART_ODD);
            // vexpdif(vreg_x_exp_odd, vreg_x_f32_b, max_0a, preg_134, PART_ODD);

            // vmuls(vreg_x_f32_1_a, vreg_x_f32_1_a, scale, preg_108);
            // vmuls(vreg_x_f32_1_b, vreg_x_f32_1_b, scale, preg_108);
            vexpdif(vreg_x_exp_even, vreg_x_f32_a, max_0a, preg_134, PART_ODD);
            vexpdif(vreg_x_exp_odd, vreg_x_f32_b, max_0a, preg_134, PART_ODD);
            vexpdif(vreg_x_exp_even_1, vreg_x_f32_1_a, max_0a, preg_134, PART_ODD);
            vexpdif(vreg_x_exp_odd_1, vreg_x_f32_1_b, max_0a, preg_134, PART_ODD);

            // vmulscvt change to vcvt has performance gain
            // vmulscvt(vreg_x_exp_even_f16, vreg_x_exp_even, 1.0f, preg_100, PART_EVEN);
            // vmulscvt(vreg_x_exp_odd_f16, vreg_x_exp_odd, 1.0f, preg_101, PART_EVEN);
            // vmulscvt(vreg_x_exp_even_f16_1, vreg_x_exp_even_1, 1.0f, preg_135, PART_EVEN);
            // vmulscvt(vreg_x_exp_odd_f16_1, vreg_x_exp_odd_1, 1.0f, preg_136, PART_EVEN);
            vcvt(vreg_x_exp_even_f16, vreg_x_exp_even, preg_108, ROUND_R, RS_ENABLE, PART_EVEN, MODE_ZEROING);
            vcvt(vreg_x_exp_odd_f16, vreg_x_exp_odd, preg_108, ROUND_R, RS_ENABLE, PART_EVEN, MODE_ZEROING);
            vcvt(vreg_x_exp_even_f16_1, vreg_x_exp_even_1, preg_108, ROUND_R, RS_ENABLE, PART_EVEN, MODE_ZEROING);
            vcvt(vreg_x_exp_odd_f16_1, vreg_x_exp_odd_1, preg_108, ROUND_R, RS_ENABLE, PART_EVEN, MODE_ZEROING);
        
            vdintlv(vreg_x_exp_f16_pack, vreg_x_exp_f16_packa, vreg_x_exp_even_f16, vreg_x_exp_odd_f16);
            vdintlv(vreg_x_exp_f16_1_pack, vreg_x_exp_f16_1_packa, vreg_x_exp_even_f16_1, vreg_x_exp_odd_f16_1);
            vsstb(vreg_x_exp_f16_pack, ((__ubuf__ half *&) nz_buffer_Ptr), 0x410001, preg_108, POST_UPDATE);
            vsstb(vreg_x_exp_f16_1_pack, ((__ubuf__ half *&) nz_buffer_Ptr2), 0x410001, preg_108, POST_UPDATE);

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
        
        vlds(vreg_exp_max, ((__ubuf__ float *) exp_max_Ptr), 0, NORM);    //ASC directly use vreg_x_max_f32_b, worse performance because of long dependency
        vlds(vreg_l0, ((__ubuf__ float *) new_global_sum_Ptr), 0, NORM);
        vmul(vreg_l0, vreg_exp_max, vreg_l0, preg_134, MODE_ZEROING);
        // vmul(vreg_l0, vreg_x_max_f32_b, vreg_l0, preg_134, MODE_ZEROING);
        vadd(vreg_l0, vreg_l0, vreg_x_sum0, preg_134, MODE_ZEROING);
        vsts(vreg_l0, ((__ubuf__ float *) new_global_sum_Ptr), 0, NORM_B32, preg_134);
    }
#if skip_rescale
    constexpr int SYNC_PERIOD = kFaCvFifoConsSyncPeriod;
    const bool should_wait_consume = aiv_should_wait_consumption<SKIP_STATUS_FIFO_SIZE, SYNC_PERIOD>(sync_iter);

    if (last_tile) {
        if (should_wait_consume) {
            wait_intra_block(PIPE_S, FftsBufferFlag::SS_BUF_READY + 1);
        }
        write_skip_status_slot(tile_id, is_vec1, 0);
        set_intra_block(PIPE_S, FftsBufferFlag::SS_BUF_READY);
        TMULS(new_global_max, local_max, 1.0f);
    } else {
        using TileDataScalar = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, 1, 1>;
        TileDataScalar ctrlTile;
        const uint64_t ctrl_tile_offset = 255U * 1024U;
        TASSIGN(ctrlTile, ctrl_tile_offset);
        TROWMAX(ctrlTile, deltaMaxTile, local_sum);
        const uint64_t ss_buf_pingpong = tile_id % 2;
        set_flag(PIPE_V, PIPE_S, EVENT_ID7 + ss_buf_pingpong);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID7 + ss_buf_pingpong);

        if (should_wait_consume) {
            wait_intra_block(PIPE_S, FftsBufferFlag::SS_BUF_READY + 1);
        }

        if (*(ctrlTile.data()) < threshold) {
            write_skip_status_slot(tile_id, is_vec1, 1);
        } else {
            write_skip_status_slot(tile_id, is_vec1, 0);
        }

        set_intra_block(PIPE_S, FftsBufferFlag::SS_BUF_READY);

        if (*(ctrlTile.data()) > threshold) {
            TMULS(new_global_max, local_max, 1.0f);
        }
    }
#else
    (void)tile_id;
    (void)sync_iter;
    (void)is_vec1;
    (void)last_tile;
#endif
#else
    constexpr float scale = constexpr_inv_sqrt(HEAD_SIZE);
    const bool is_vec1 = static_cast<size_t>(get_subblockid());
    constexpr float threshold = 8.0f / scale;

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
#if skip_rescale
    constexpr int SYNC_PERIOD = kFaCvFifoConsSyncPeriod;
    const bool should_wait_consume = aiv_should_wait_consumption<SKIP_STATUS_FIFO_SIZE, SYNC_PERIOD>(sync_iter);

    if (last_tile) {
        if (should_wait_consume) {
            wait_intra_block(PIPE_S, FftsBufferFlag::SS_BUF_READY + 1);
        }
        write_skip_status_slot(tile_id, is_vec1, 0);
        set_intra_block(PIPE_S, FftsBufferFlag::SS_BUF_READY);
        TMULS(new_global_max, local_max, 1.0f);
    } else {
        using TileDataScalar = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, 1, 1>;
        TileDataScalar ctrlTile;
        const uint64_t ctrl_tile_offset = 255U * 1024U;
        TASSIGN(ctrlTile, ctrl_tile_offset);
        const uint64_t delta_max_offset = 254U * 1024U;
        ReduceTileD1 deltaMaxTile;
        TASSIGN(deltaMaxTile, delta_max_offset);
        TSUB(deltaMaxTile, local_max, new_global_max);
        TROWMAX(ctrlTile, deltaMaxTile, local_sum);
        const uint64_t ss_buf_pingpong = tile_id % 2;
        set_flag(PIPE_V, PIPE_S, EVENT_ID7 + ss_buf_pingpong);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID7 + ss_buf_pingpong);

        if (should_wait_consume) {
            wait_intra_block(PIPE_S, FftsBufferFlag::SS_BUF_READY + 1);
        }

        if (*(ctrlTile.data()) < threshold) {
            write_skip_status_slot(tile_id, is_vec1, 1);
        } else {
            write_skip_status_slot(tile_id, is_vec1, 0);
        }

        set_intra_block(PIPE_S, FftsBufferFlag::SS_BUF_READY);

        if (*(ctrlTile.data()) > threshold) {
            TMULS(new_global_max, local_max, 1.0f);
        }
    }
#else
    TMULS(new_global_max, local_max, 1.0f); // just copy
    (void)tile_id;
    (void)sync_iter;
    (void)is_vec1;
    (void)last_tile;
#endif
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
          typename TileDataS1, typename TileDataH_NZ_T, int MODE>
AICORE inline void pto_macro_fa_softmax_dn(TileDataD2 __out__ x_exp, TileDataS1 __in__ input_x,
                                           ReduceTileD1 __out__ local_max, ReduceTileD1 __out__ local_sum,
                                           ReduceTileD1 __in__ new_global_max, ReduceTileD1 __out__ new_global_sum,
                                           ReduceTileD1 __out__ exp_max, TileDataS1 __out__ input_reduce_tmp,
                                           TileDataS1 __out__ p_tile_fp32, TileDataS1 triu, TileDataH_NZ_T nzConvBuffer, int s0_index, int s1_index,
                                           int tile_id, int sync_iter, bool last_tile)
{
    if (s1_index <= s0_index || !CAUSAL_MASK) {
        if constexpr (init) {
            softmax_opt_fa_dn_init_impl<HEAD_SIZE, CAUSAL_MASK, ReduceTileD1, TileDataD2, TileDataS1, TileDataH_NZ_T, MODE>(
                tile_id, sync_iter, x_exp, input_x, local_max, local_sum, new_global_max, new_global_sum, exp_max,
                input_reduce_tmp, p_tile_fp32, triu, nzConvBuffer, s0_index, s1_index, last_tile);
        } else {
            softmax_opt_fa_dn_not_init_impl<HEAD_SIZE, CAUSAL_MASK, ReduceTileD1, TileDataD2, TileDataS1, TileDataH_NZ_T, MODE>(
                tile_id, sync_iter, x_exp, input_x, local_max, local_sum, new_global_max, new_global_sum, exp_max,
                input_reduce_tmp, p_tile_fp32, triu, nzConvBuffer, s0_index, s1_index, last_tile);
        }
    } else if constexpr (CAUSAL_MASK) {
        TMULS(x_exp, x_exp, 0.0);
        TMULS(exp_max, exp_max, 0.0);
        TADDS(exp_max, exp_max, 1.0);
#if skip_rescale
        write_skip_status_slot(tile_id, static_cast<size_t>(get_subblockid()), 0);
        set_intra_block(PIPE_S, FftsBufferFlag::SS_BUF_READY);
#endif
    }
}

} // namespace pto

#endif
