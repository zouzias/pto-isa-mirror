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

    unsigned ubM = TileDataD2::Rows;
    unsigned ubN = TileDataD2::Cols;
    unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(typename TileDataS1::DType);
    uint16_t repeatTimes = CeilDivision(ubN, elementsPerRepeat);
    uint16_t nLoop = (ubM - 1) / 2;
    bool remain = (ubM - 1) % 2;
    uint16_t rowRepeat = CeilDivision(ubM, elementsPerRepeat);
    uint64_t VSSTB_CONFIG = ((ubN + 1) << 16) | 1u;

    __VEC_SCOPE__{
        vector_f32 src_in1, src_in2, src_in3, tmp_in;
        vector_f32 max_1a;
        vector_f32 sum_1a, sum_1b;
        vector_f32 vreg_exp_max;

        __ubuf__ float *src0_ub = (__ubuf__ float *)input_x_Ptr;

        vector_bool preg_src;
        vector_bool preg_b32_all = pset_b32(PAT_ALL);
        vector_bool preg_b16_all = pset_b16(PAT_ALL);
        vector_bool preg_b8_all = pset_b8(PAT_ALL);
        uint32_t destItems = 1;
        constexpr auto distValue = std::integral_constant<::DistVST,static_cast<::DistVST>(GetDistVst<typename TileDataS1::DType, DistVST::DIST_NORM>())>();
        uint32_t sReg = ubN;

        for (uint16_t i = 0; i < uint16_t(repeatTimes) ; ++i) {
            preg_src = CreatePredicate<typename TileDataS1::DType>(sReg);
            vlds(max_1a, src0_ub, i * elementsPerRepeat, NORM);
            for (uint16_t j = 0; j < uint16_t(nLoop) ; ++j) {
                vlds(src_in1, src0_ub, i * elementsPerRepeat + (2 * j + 1) * TileDataS1::Cols, NORM);
                vlds(src_in2, src0_ub, i * elementsPerRepeat + (2 * j + 2) * TileDataS1::Cols, NORM);
                vmax(tmp_in, src_in1, src_in2, preg_src, MODE_ZEROING);
                vmax(max_1a, max_1a, tmp_in, preg_src, MODE_ZEROING);
            }
            if (remain) {
                vlds(src_in1, src0_ub, i * elementsPerRepeat + (2 * nLoop + 1) * TileDataS1::Cols, NORM);
                vmax(max_1a, max_1a, src_in1, preg_src, MODE_ZEROING);
            }
            vsts(max_1a, new_global_max_Ptr, i * elementsPerRepeat, distValue, preg_src);
        }
        
        mem_bar(VST_VLD);
        vector_f32 vreg0;
        vector_f32 vreg1;
        vector_f32 vreg2;
        vector_f32 vreg3;
        vector_f16 vreg0_f16;
        vector_f16 vreg2_f16;
        vector_f16 vreg_x_exp_f16_pack, vreg_x_exp_f16_packa;
        unsigned remains = repeatTimes % 2;
        if(remains){
            for (uint16_t i = 0; i < uint16_t(repeatTimes) ; ++i) {
                vbr(sum_1a, 0);

                for (uint16_t j = 0; j < (uint16_t)(ubM); ++j) {
                    vlds(vreg0, src0_ub,  i * elementsPerRepeat + j * TileDataS1::RowStride, NORM);
                    vlds(vreg1, new_global_max_Ptr, i * elementsPerRepeat, NORM);

                    vsub(vreg2, vreg0, vreg1, preg_b32_all, MODE_ZEROING);

                    vmuls(vreg2, vreg2, scale, preg_b32_all, MODE_ZEROING);

                    vexp(vreg2, vreg2, preg_b32_all, MODE_ZEROING);

                    vadd(sum_1a, sum_1a, vreg2, preg_b32_all, MODE_ZEROING);

                    vmulscvt(vreg2_f16, vreg2, 1.0f, preg_src, PART_EVEN);

                    // ND output
                    // vsts(vreg2_f16, ((__ubuf__ half *) nz_buffer_Ptr + i*elementsPerRepeat + j*ubN), 0, PK_B32, preg_b16_all);
                    // NZ+1 output
                    vsstb(vreg2_f16, ((__ubuf__ half *&) nz_buffer_Ptr), VSSTB_CONFIG, preg_b16_all, POST_UPDATE);
                }
                vsts(sum_1a, new_global_sum_Ptr, i * elementsPerRepeat, distValue, preg_b32_all);
            }
        } else {
            for (uint16_t i = 0; i < uint16_t(repeatTimes/2) ; ++i) {
                vbr(sum_1a, 0);
                vbr(sum_1b, 0);

                for (uint16_t j = 0; j < (uint16_t)(ubM); ++j) {
                    vlds(vreg0, src0_ub,  (i*2) * elementsPerRepeat + j * TileDataS1::RowStride, NORM);
                    vlds(vreg2, src0_ub,  (i*2+1) * elementsPerRepeat + j * TileDataS1::RowStride, NORM);
                    vlds(vreg1, new_global_max_Ptr, (i*2) * elementsPerRepeat, NORM);
                    vlds(vreg3, new_global_max_Ptr, (i*2+1) * elementsPerRepeat, NORM);

                    vsub(vreg0, vreg0, vreg1, preg_b32_all, MODE_ZEROING);
                    vsub(vreg2, vreg2, vreg3, preg_b32_all, MODE_ZEROING);

                    vmuls(vreg0, vreg0, scale, preg_b32_all, MODE_ZEROING);
                    vmuls(vreg2, vreg2, scale, preg_b32_all, MODE_ZEROING);

                    vexp(vreg0, vreg0, preg_b32_all, MODE_ZEROING);
                    vexp(vreg2, vreg2, preg_b32_all, MODE_ZEROING);

                    vadd(sum_1a, sum_1a, vreg0, preg_b32_all, MODE_ZEROING);
                    vadd(sum_1b, sum_1b, vreg2, preg_b32_all, MODE_ZEROING);

                    vmulscvt(vreg0_f16, vreg0, 1.0f, preg_b32_all, PART_EVEN);
                    vmulscvt(vreg2_f16, vreg2, 1.0f, preg_b32_all, PART_EVEN);

                    vdintlv(vreg_x_exp_f16_pack, vreg_x_exp_f16_packa, vreg0_f16, vreg2_f16);
                    // ND output
                    // vsts(vreg_x_exp_f16_pack, ((__ubuf__ half *) nz_buffer_Ptr + i*2*elementsPerRepeat + j*ubN), 0, NORM_B32, preg_b32_all);
                    // NZ+1 output
                    vsstb(vreg_x_exp_f16_pack, ((__ubuf__ half *&) nz_buffer_Ptr), VSSTB_CONFIG, preg_b16_all, POST_UPDATE);
                }
                vsts(sum_1a, new_global_sum_Ptr, (i*2) * elementsPerRepeat, distValue, preg_b32_all);
                vsts(sum_1b, new_global_sum_Ptr, (i*2+1) * elementsPerRepeat, distValue, preg_b32_all);
            }
        }
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

    unsigned ubM = TileDataD2::Rows;
    unsigned ubN = TileDataD2::Cols;
    unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(typename TileDataS1::DType);
    uint16_t repeatTimes = CeilDivision(ubN, elementsPerRepeat);
    uint16_t nLoop = (ubM - 1) / 2;
    bool remain = (ubM - 1) % 2;
    uint16_t rowRepeat = CeilDivision(ubM, elementsPerRepeat);
    uint64_t VSSTB_CONFIG = ((ubN + 1) << 16) | 1u;

    __VEC_SCOPE__{
        vector_f32 src_in1, src_in2, src_in3, tmp_in;
        vector_f32 max_1a;
        vector_f32 sum_1a, sum_1b;
        vector_f32 vreg_exp_max;
#if skip_rescale
        vector_f32 vreg_delta_max;
#endif

        __ubuf__ float *src0_ub = (__ubuf__ float *)input_x_Ptr;

        vector_bool preg_src;
        vector_bool preg_b32_all = pset_b32(PAT_ALL);
        vector_bool preg_b16_all = pset_b16(PAT_ALL);
        vector_bool preg_b8_all = pset_b8(PAT_ALL);
        uint32_t destItems = 1;
        constexpr auto distValue = std::integral_constant<::DistVST,static_cast<::DistVST>(GetDistVst<typename TileDataS1::DType, DistVST::DIST_NORM>())>();
        uint32_t sReg = ubN;

        for (uint16_t i = 0; i < uint16_t(repeatTimes) ; ++i) {
            preg_src = CreatePredicate<typename TileDataS1::DType>(sReg);
            vlds(max_1a, src0_ub, i * elementsPerRepeat, NORM);
            for (uint16_t j = 0; j < uint16_t(nLoop) ; ++j) {
                vlds(src_in1, src0_ub, i * elementsPerRepeat + (2 * j + 1) * TileDataS1::Cols, NORM);
                vlds(src_in2, src0_ub, i * elementsPerRepeat + (2 * j + 2) * TileDataS1::Cols, NORM);
                vmax(tmp_in, src_in1, src_in2, preg_src, MODE_ZEROING);
                vmax(max_1a, max_1a, tmp_in, preg_src, MODE_ZEROING);
            }
            if (remain) {
                vlds(src_in1, src0_ub, i * elementsPerRepeat + (2 * nLoop + 1) * TileDataS1::Cols, NORM);
                vmax(max_1a, max_1a, src_in1, preg_src, MODE_ZEROING);
            }
            vsts(max_1a, local_max_Ptr, i * elementsPerRepeat, distValue, preg_src);
        }

        mem_bar(VST_VLD);
        vector_f32 vreg0;
        vector_f32 vreg1;
        vector_f32 vreg2;
        vector_f32 vreg3;
        vector_f16 vreg0_f16;
        vector_f16 vreg2_f16;
        vector_f16 vreg_x_exp_f16_pack, vreg_x_exp_f16_packa;
        unsigned remains = repeatTimes % 2;
        
        for (uint16_t j = 0; j < (uint16_t)(repeatTimes); ++j) {
            vlds(max_1a, local_max_Ptr, j * elementsPerRepeat, NORM);
            vlds(src_in1, new_global_max_Ptr, j * elementsPerRepeat, NORM);
            vmax(max_1a, max_1a, src_in1, preg_b32_all, MODE_ZEROING);
            vsub(vreg_exp_max, src_in1, max_1a, preg_b32_all, MODE_ZEROING);
            // vsts(max_1a, new_global_max_Ptr, j * elementsPerRepeat, NORM_B32, preg_b32_all);

            vmuls(vreg_exp_max, vreg_exp_max, scale, preg_b32_all, MODE_ZEROING);
            vexp(vreg_exp_max, vreg_exp_max, preg_b32_all, MODE_ZEROING);
            vsts(vreg_exp_max, exp_max_Ptr, j * elementsPerRepeat, NORM_B32, preg_b32_all);
        }
        mem_bar(VST_VLD);

#if skip_rescale
        vsub(vreg_delta_max, max_1a, vreg_x_max_f32_b, preg_b16_all);
        vsts(vreg_delta_max, delta_max_Ptr, 0, NORM_B32, preg_b16_all);
        vsts(max_1a, local_max_Ptr, 0, NORM_B16, preg_b16_all);
#else
        vsts(max_1a, new_global_max_Ptr, 0, NORM_B16, preg_b16_all);  //just copy
#endif
        
        if(remains){
            for (uint16_t i = 0; i < uint16_t(repeatTimes) ; ++i) {
                vbr(sum_1a, 0);

                for (uint16_t j = 0; j < (uint16_t)(ubM); ++j) {
                    vlds(vreg0, src0_ub,  i * elementsPerRepeat + j * TileDataS1::RowStride, NORM);
                    vlds(vreg1, new_global_max_Ptr, i * elementsPerRepeat, NORM);

                    vsub(vreg2, vreg0, vreg1, preg_b32_all, MODE_ZEROING);

                    vmuls(vreg2, vreg2, scale, preg_b32_all, MODE_ZEROING);

                    vexp(vreg2, vreg2, preg_b32_all, MODE_ZEROING);

                    vadd(sum_1a, sum_1a, vreg2, preg_b32_all, MODE_ZEROING);

                    vmulscvt(vreg2_f16, vreg2, 1.0f, preg_src, PART_EVEN);

                    // ND output
                    // vsts(vreg2_f16, ((__ubuf__ half *) nz_buffer_Ptr + i*elementsPerRepeat + j*ubN), 0, PK_B32, preg_b16_all);
                    // NZ+1 output
                    vsstb(vreg2_f16, ((__ubuf__ half *&) nz_buffer_Ptr), VSSTB_CONFIG, preg_b16_all, POST_UPDATE);
                }
                vsts(sum_1a, local_sum_Ptr, i * elementsPerRepeat, distValue, preg_b32_all);
            }
        } else {
            for (uint16_t i = 0; i < uint16_t(repeatTimes/2) ; ++i) {
                vbr(sum_1a, 0);
                vbr(sum_1b, 0);

                for (uint16_t j = 0; j < (uint16_t)(ubM); ++j) {
                    vlds(vreg0, src0_ub,  (i*2) * elementsPerRepeat + j * TileDataS1::RowStride, NORM);
                    vlds(vreg2, src0_ub,  (i*2+1) * elementsPerRepeat + j * TileDataS1::RowStride, NORM);
                    vlds(vreg1, new_global_max_Ptr, (i*2) * elementsPerRepeat, NORM);
                    vlds(vreg3, new_global_max_Ptr, (i*2+1) * elementsPerRepeat, NORM);

                    vsub(vreg0, vreg0, vreg1, preg_b32_all, MODE_ZEROING);
                    vsub(vreg2, vreg2, vreg3, preg_b32_all, MODE_ZEROING);

                    vmuls(vreg0, vreg0, scale, preg_b32_all, MODE_ZEROING);
                    vmuls(vreg2, vreg2, scale, preg_b32_all, MODE_ZEROING);

                    vexp(vreg0, vreg0, preg_b32_all, MODE_ZEROING);
                    vexp(vreg2, vreg2, preg_b32_all, MODE_ZEROING);

                    vadd(sum_1a, sum_1a, vreg0, preg_b32_all, MODE_ZEROING);
                    vadd(sum_1b, sum_1b, vreg2, preg_b32_all, MODE_ZEROING);

                    vmulscvt(vreg0_f16, vreg0, 1.0f, preg_b32_all, PART_EVEN);
                    vmulscvt(vreg2_f16, vreg2, 1.0f, preg_b32_all, PART_EVEN);

                    vdintlv(vreg_x_exp_f16_pack, vreg_x_exp_f16_packa, vreg0_f16, vreg2_f16);
                    
                    // ND output
                    // vsts(vreg_x_exp_f16_pack, ((__ubuf__ half *) nz_buffer_Ptr + i*2*elementsPerRepeat + j*ubN), 0, NORM_B32, preg_b32_all);
                    // NZ+1 output
                    vsstb(vreg_x_exp_f16_pack, ((__ubuf__ half *&) nz_buffer_Ptr), VSSTB_CONFIG, preg_b16_all, POST_UPDATE);
                }
                vsts(sum_1a, local_sum_Ptr, (i*2) * elementsPerRepeat, distValue, preg_b32_all);
                vsts(sum_1b, local_sum_Ptr, (i*2+1) * elementsPerRepeat, distValue, preg_b32_all);
            }
        }
        mem_bar(VST_VLD);
        for(uint16_t j = 0; j < (uint16_t)(repeatTimes); ++j)
        {
            vlds(src_in1, local_sum_Ptr, j * elementsPerRepeat, NORM);
            vlds(src_in2, new_global_sum_Ptr, j * elementsPerRepeat, NORM);
            vmul(src_in2, vreg_exp_max, src_in2, preg_b32_all, MODE_ZEROING);
            vadd(src_in2, src_in2, src_in1, preg_b32_all, MODE_ZEROING);
            vsts(src_in2, new_global_sum_Ptr, j * elementsPerRepeat, NORM_B32, preg_b32_all);
        }
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