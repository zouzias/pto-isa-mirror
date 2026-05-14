/*
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_MACRO_FA_SOFTMAX_DN_OPUS_HPP
#define PTO_MACRO_FA_SOFTMAX_DN_OPUS_HPP

#include <pto/pto-inst.hpp>

namespace pto {

// =============================================================================
// FlashAttention v4 (FP16, DN-layout) softmax — Opus FUSED variant.
//
// FUSION TARGETS (vs the baseline pto_macro_fa_dn_softmax.hpp):
//
//   1. Direct ND→NZ+1 store using VSSTB.
//      The baseline writes Phase-2 results (FP16 x_exp) into x_expT in ND
//      layout, then a separate TMOV VF re-loads ND and stores NZ+1 into
//      nzConvBuffer.  Here Phase-2 writes the NZ+1 buffer DIRECTLY using
//      vsstb with block_stride = Cube_S1 + 1 = 129 (4 active blocks per row
//      for the [128,64] tile).  This eliminates the entire UB→UB TMOV VF
//      and its store-load round-trip cost.
//
//   2. vcvt replaces vmulscvt(scale=1.0).
//      Baseline used vmulscvt(_,_,1.0f,preg,PART_EVEN) — an LNEXP-class FMA
//      that occupies both MUL and ADD pipes, with throughput 1/cyc.  When
//      the scalar is 1.0 the multiply is a no-op and a plain vcvt(half,
//      float, preg, R(), RS_DISABLE, PART_EVEN) does the same conversion
//      on the ADD pipe alone, freeing the MUL pipe for the per-row scaling
//      vmuls — increasing MUL/ADD dual-issue parallelism.
//
//   3. Per-row processing (one row per vexpdif/vsstb) replaces the
//      2-rows-interleaved pack pattern of the baseline.  Required because
//      direct vsstb for [128,64] needs 4 active blocks (128B) per
//      instruction; the interleave pattern packed two rows (256B) into a
//      single VL register that we can't scatter to two different NZ+1 row
//      slots in one vsstb.
//
// Tile geometry (per vec core, for the FA4 [128,64] config):
//   - input_x  : Tile_S1=128 rows × Vec_S0=64 cols FP32, ND row-major UB.
//   - x_exp_NZ : nzConvBuffer, shape [Cube_S1+1=129, Vec_S0=64] half, NZ+1
//                (BLayout::ColMajor + CompactMode::RowPlusOne, 4 blocks/row).
//   - local_max, local_sum, new_global_max, new_global_sum: ReduceTileF_T,
//                shape [1, Vec_S0=64] FP32.
//
// Roofline (Phase 2, [128, 64] FP16 path):
//   LNEXP FP32 vexpdif throughput = 4 cyc / 64-elem VL  (dual-issuable on
//   EXQ0/EXQ1, but already at peak for one stream).
//   128 rows × 1 vexpdif/row = 128 vexpdif × 4 cyc/issue / 2 (dual-issue
//   with paired stream) ≈ 256 LNEXP-cycles best-case.
//   MUL pipe: 128 vmuls = 128 cyc.
//   ADD pipe: 128 vadd + 128 vcvt = 256 cyc (will dual-issue across slots).
//   LSU: 128 vld + 128 vsstb = 256 LSU cyc.
//   Phase-2 floor ≈ max(256 LNEXP, 256 LSU) ≈ 256 cycles assuming perfect
//   pipelining.  Baseline Phase-2 + TMOV ND→NZ runs ≈ 833 vec ticks.
// =============================================================================

constexpr PTO_INTERNAL float opus_constexpr_sqrt(float x)
{
    if (x <= 0.0f) return 0.0f;
    float guess = x;
    for (int i = 0; i < 8; ++i) {
        guess = 0.5f * (guess + x / guess);
    }
    return guess;
}

constexpr AICORE inline float opus_constexpr_inv_sqrt(float x)
{
    return 1.0f / opus_constexpr_sqrt(x);
}

// -----------------------------------------------------------------------------
// init variant: first S1 tile, initializes running state.
//
// Writes:
//   - new_global_max   ← local col-max(x)        (FP32, 64 elems)
//   - new_global_sum   ← Σ exp(scale*(x - max))   (FP32, 64 elems)
//   - nzConvBuffer     ← FP16 exp(scale*(x-max))  (NZ+1, [128+1, 64])
//
// Skips: x_expT (no ND buffer needed — VSSTB writes NZ+1 directly).
// -----------------------------------------------------------------------------
template <int HEAD_SIZE, bool CAUSAL_MASK,
          typename ReduceTileD1, typename TileDataNZ, typename TileDataS1>
__tf__ AICORE inline void
softmax_opus_fused_fa_dn_init_impl(TileDataNZ __out__ nz_x_exp,
                                   TileDataS1 __in__ input_x,
                                   ReduceTileD1 __out__ local_max,
                                   ReduceTileD1 __out__ local_sum,
                                   ReduceTileD1 __out__ new_global_max,
                                   ReduceTileD1 __out__ new_global_sum,
                                   ReduceTileD1 __out__ exp_max,
                                   int s0_index, int s1_index)
{
    (void)local_max;
    (void)exp_max;
    (void)local_sum;
    (void)s0_index;
    (void)s1_index;

    __ubuf__ half  *nz_dst_base       = (__ubuf__ half *)__cce_get_tile_ptr(nz_x_exp.data());
    __ubuf__ float *input_x_Ptr       = (__ubuf__ float *)__cce_get_tile_ptr(input_x.data());
    __ubuf__ float *new_global_max_Ptr = (__ubuf__ float *)__cce_get_tile_ptr(new_global_max.data());
    __ubuf__ float *new_global_sum_Ptr = (__ubuf__ float *)__cce_get_tile_ptr(new_global_sum.data());

    constexpr float scale  = opus_constexpr_inv_sqrt(HEAD_SIZE);
    constexpr unsigned ubN = TileDataS1::Rows;        // = Tile_S1 = 128
    constexpr unsigned ubM = TileDataS1::Cols;        // = Vec_S0  = 64

    // VSSTB block-stride for NZ+1:  (Rows + 1) × C0_SIZE_BYTE / BLOCK_BYTE_SIZE
    //                            = (128 + 1) × 32 / 32 = 129 blocks.
    // repeat_stride = 1 block (advance to next row's slot in the column-group).
    // cfgVsstb = (blockStride << 16) | repeatStride.
    constexpr uint32_t cfgVsstb     = (129u << 16) | 1u;        // = 0x00810001
    // Last-row repeatStride pulls the destination pointer back to the
    // beginning of the buffer so a subsequent call can re-use the same
    // pointer.  Not strictly required since the caller re-derives the base
    // each invocation, but matches PTO's TMov ND2Nz convention.
    constexpr uint32_t cfgVsstbLast = (129u << 16) | 1u;

    __VEC_SCOPE__
    {
        // -----------------------------------------------------------------
        // Predicates
        // -----------------------------------------------------------------
        vector_bool preg_b32 = pset_b32(PAT_ALL);     // 64 f32 lanes active
        vector_bool preg_b16 = pset_b16(PAT_ALL);     // 128 f16 lanes active
        // 64-half predicate = lower 128 B = 4 blocks active.  This is what
        // VSSTB uses to scatter 4 blocks to the NZ+1 column-group.
        uint32_t sreg_count  = (uint32_t)(ubM);       // 64 halves
        vector_bool preg_4blk = plt_b16(sreg_count, POST_UPDATE);

        // -----------------------------------------------------------------
        // Phase 1: column max over all 128 rows.
        // 8 parallel max accumulators (max_0a..max_3b) covering rows {0..7
        // (mod 8)}; outer iter advances each by 8 rows (stride=512 elements).
        // After the loop, tree-reduce to max_0a (1 VL of 64 f32 col-maxes).
        // -----------------------------------------------------------------
        vector_f32 src_00a, src_01a, src_02a, src_03a;
        vector_f32 src_00b, src_01b, src_02b, src_03b;
        vector_f32 max_0a, max_1a, max_2a, max_3a;
        vector_f32 max_0b, max_1b, max_2b, max_3b;

        __ubuf__ float *src0_ub        = input_x_Ptr;
        __ubuf__ float *src0_ub1       = input_x_Ptr + 128;
        __ubuf__ float *src0_ub2       = input_x_Ptr + 256;
        __ubuf__ float *src0_ub3       = input_x_Ptr + 384;
        __ubuf__ float *src0_ub_un     = input_x_Ptr +  64;
        __ubuf__ float *src0_ub1_un    = src0_ub1   +  64;
        __ubuf__ float *src0_ub2_un    = src0_ub2   +  64;
        __ubuf__ float *src0_ub3_un    = src0_ub3   +  64;

        vbr(max_0a, 0); vbr(max_0b, 0);
        vbr(max_1a, 0); vbr(max_1b, 0);
        vbr(max_2a, 0); vbr(max_2b, 0);
        vbr(max_3a, 0); vbr(max_3b, 0);

        for (uint16_t iter_m = 0; iter_m < uint16_t(ubN / 8); ++iter_m) {
            vlds(src_00a, src0_ub,     512, NORM, POST_UPDATE);
            vlds(src_00b, src0_ub_un,  512, NORM, POST_UPDATE);
            vmax(max_0a, max_0a, src_00a, preg_b32);
            vmax(max_0b, max_0b, src_00b, preg_b32);

            vlds(src_01a, src0_ub1,    512, NORM, POST_UPDATE);
            vlds(src_01b, src0_ub1_un, 512, NORM, POST_UPDATE);
            vmax(max_1a, max_1a, src_01a, preg_b32);
            vmax(max_1b, max_1b, src_01b, preg_b32);

            vlds(src_02a, src0_ub2,    512, NORM, POST_UPDATE);
            vlds(src_02b, src0_ub2_un, 512, NORM, POST_UPDATE);
            vmax(max_2a, max_2a, src_02a, preg_b32);
            vmax(max_2b, max_2b, src_02b, preg_b32);

            vlds(src_03a, src0_ub3,    512, NORM, POST_UPDATE);
            vlds(src_03b, src0_ub3_un, 512, NORM, POST_UPDATE);
            vmax(max_3a, max_3a, src_03a, preg_b32);
            vmax(max_3b, max_3b, src_03b, preg_b32);
        }

        // Tree reduce: 8 → 1
        vmax(max_0a, max_0a, max_1a, preg_b32);
        vmax(max_0b, max_0b, max_1b, preg_b32);
        vmax(max_2a, max_2a, max_3a, preg_b32);
        vmax(max_2b, max_2b, max_3b, preg_b32);
        vmax(max_0a, max_0a, max_2a, preg_b32);
        vmax(max_0b, max_0b, max_2b, preg_b32);
        vmax(max_0a, max_0a, max_0b, preg_b32);

        // Store col-max as new_global_max (FP32).
        vsts(max_0a, new_global_max_Ptr, 0, NORM_B32, preg_b32);

        // Scale the max for vexpdif: pre-scale max so vexpdif computes
        // exp(scale*x - scale*max).  The baseline does this with vmuls.
        vmuls(max_0a, max_0a, scale, preg_b32);

        // -----------------------------------------------------------------
        // Phase 2: per-row exp + sum + direct NZ+1 store via vsstb.
        //
        // Loop body processes 2 rows per iter (paired for dual-issue):
        //   vld row0/row1 (FP32)         (LSU)
        //   vmuls scale                  (MUL or ADD)
        //   vexpdif against scaled-max   (LNEXP, FP32, thpt=4, dual-iss)
        //   vadd sum accumulator         (ADD)
        //   vcvt to FP16  PART_EVEN       (ADD)
        //   vsstb to NZ+1                (LSU, 4 active blocks)
        //
        // After 128/2 = 64 iters: tree-reduce two sum accumulators, store.
        // -----------------------------------------------------------------
        vector_f32 v_sum0, v_sum1;
        vbr(v_sum0, 0);
        vbr(v_sum1, 0);

        __ubuf__ float *src_p0 = input_x_Ptr;
        __ubuf__ half  *nz_p0  = nz_dst_base;

        // We post-update src by 64 elements per row (= 1 row of 64 f32),
        // and post-update nz_p0 by repeatStride = 1 block (= 32 bytes,
        // half-typed = 16 halves) per vsstb.
        // After 128 vsstb calls, nz_p0 has advanced by 128 blocks (= 4096 B
        // = 128 × 32 B).  Combined with block_stride = 129 inside each
        // vsstb, this lays out the [128, 64] data into the [4, 129] block
        // NZ+1 column-group layout (4 column-groups × 128 rows each
        // physically present, with 1 row of padding per column-group).

        for (uint16_t r = 0; r < uint16_t(ubN / 2); ++r) {
            vector_f32 v_x0, v_x1;
            vector_f32 v_exp0, v_exp1;
            vector_f16 v_h0, v_h1;
            vector_f16 v_pack0, v_pack1;
            vector_f16 v_packa;    // garbage sink for vdintlv's second output

            vlds(v_x0, src_p0, 64, NORM, POST_UPDATE);   // load row 2r
            vlds(v_x1, src_p0, 64, NORM, POST_UPDATE);   // load row 2r+1

            vmuls(v_x0, v_x0, scale, preg_b32);
            vmuls(v_x1, v_x1, scale, preg_b32);

            // FP32 vexpdif canonically uses PART_ODD (see TRowExpandExpdif.hpp).
            vexpdif(v_exp0, v_x0, max_0a, preg_b32, PART_ODD);
            vexpdif(v_exp1, v_x1, max_0a, preg_b32, PART_ODD);

            // Sum accumulator (post-exp f32 values).
            vadd(v_sum0, v_sum0, v_exp0, preg_b32, MODE_ZEROING);
            vadd(v_sum1, v_sum1, v_exp1, preg_b32, MODE_ZEROING);

            // FP32 → FP16, lower 4 blocks (PART_EVEN).  ROUND_A == CAST_ROUND.
            // After vcvt PART_EVEN the 64 valid halves live in the EVEN lanes
            // of the 128-wide vreg (lanes 0,2,4,…,126), interleaved with zeros.
            vcvt(v_h0, v_exp0, preg_b16, ROUND_A, RS_DISABLE, PART_EVEN);
            vcvt(v_h1, v_exp1, preg_b16, ROUND_A, RS_DISABLE, PART_EVEN);

            // Pack the even-lane data into CONTIGUOUS lower lanes via vdintlv.
            // vdintlv(out0, out1, in0, in1) is a 256-half deinterleave over
            // in0||in1: out0 receives the even-indexed lanes, out1 the odd
            // ones.  Calling it with the same f16 vreg as both inputs gives:
            //   out0[0..63]    = even lanes of in    = packed 64 valid halves
            //   out0[64..127]  = even lanes of in (duplicate)
            //   out1           = odd lanes (= zeros, since PART_EVEN gaps)
            // Only the lower 4 blocks of v_pack* are later stored (predicate),
            // so the duplicate in the upper lanes is harmless.
            vdintlv(v_pack0, v_packa, v_h0, v_h0);
            vdintlv(v_pack1, v_packa, v_h1, v_h1);

            // Scatter the 4 valid blocks of each packed vreg to NZ+1.
            // block_stride = 129 blocks, repeat_stride = 1 block.
            vsstb((vector_u16 &)v_pack0, (__ubuf__ uint16_t *&)nz_p0,
                  cfgVsstb, preg_4blk, POST_UPDATE);
            vsstb((vector_u16 &)v_pack1, (__ubuf__ uint16_t *&)nz_p0,
                  cfgVsstb, preg_4blk, POST_UPDATE);
        }

        // Sum reduce + write new_global_sum.
        vadd(v_sum0, v_sum0, v_sum1, preg_b32, MODE_ZEROING);
        vsts(v_sum0, new_global_sum_Ptr, 0, NORM_B32, preg_b32);
    }
}

// -----------------------------------------------------------------------------
// not_init variant: subsequent S1 tiles, maintains running global_max/global_sum.
//
// Same fusion as init (direct NZ+1 store, vcvt replaces vmulscvt(1.0)) PLUS:
//   - Rescales running global_sum by exp(scale * (prev_global_max - new_local_max))
//     (the FlashAttention streaming rescale).
//   - Updates global_max ← max(prev_global_max, new_local_max).
// -----------------------------------------------------------------------------
template <int HEAD_SIZE, bool CAUSAL_MASK,
          typename ReduceTileD1, typename TileDataNZ, typename TileDataS1>
__tf__ AICORE inline void
softmax_opus_fused_fa_dn_not_init_impl(TileDataNZ __out__ nz_x_exp,
                                       TileDataS1 __in__ input_x,
                                       ReduceTileD1 __out__ local_max,
                                       ReduceTileD1 __out__ local_sum,
                                       ReduceTileD1 __out__ new_global_max,
                                       ReduceTileD1 __out__ new_global_sum,
                                       ReduceTileD1 __out__ exp_max,
                                       int s0_index, int s1_index)
{
    (void)local_max;
    (void)local_sum;
    (void)s0_index;
    (void)s1_index;

    __ubuf__ half  *nz_dst_base        = (__ubuf__ half *)__cce_get_tile_ptr(nz_x_exp.data());
    __ubuf__ float *input_x_Ptr        = (__ubuf__ float *)__cce_get_tile_ptr(input_x.data());
    __ubuf__ float *new_global_max_Ptr = (__ubuf__ float *)__cce_get_tile_ptr(new_global_max.data());
    __ubuf__ float *new_global_sum_Ptr = (__ubuf__ float *)__cce_get_tile_ptr(new_global_sum.data());
    __ubuf__ float *exp_max_Ptr        = (__ubuf__ float *)__cce_get_tile_ptr(exp_max.data());

    constexpr float scale  = opus_constexpr_inv_sqrt(HEAD_SIZE);
    constexpr unsigned ubN = TileDataS1::Rows;
    constexpr unsigned ubM = TileDataS1::Cols;

    constexpr uint32_t cfgVsstb = (129u << 16) | 1u;

    __VEC_SCOPE__
    {
        vector_bool preg_b32 = pset_b32(PAT_ALL);
        vector_bool preg_b16 = pset_b16(PAT_ALL);
        uint32_t sreg_count  = (uint32_t)(ubM);
        vector_bool preg_4blk = plt_b16(sreg_count, POST_UPDATE);

        vector_f32 src_00a, src_01a, src_02a, src_03a;
        vector_f32 src_00b, src_01b, src_02b, src_03b;
        vector_f32 max_0a, max_1a, max_2a, max_3a;
        vector_f32 max_0b, max_1b, max_2b, max_3b;

        __ubuf__ float *src0_ub     = input_x_Ptr;
        __ubuf__ float *src0_ub1    = input_x_Ptr + 128;
        __ubuf__ float *src0_ub2    = input_x_Ptr + 256;
        __ubuf__ float *src0_ub3    = input_x_Ptr + 384;
        __ubuf__ float *src0_ub_un  = input_x_Ptr +  64;
        __ubuf__ float *src0_ub1_un = src0_ub1   +  64;
        __ubuf__ float *src0_ub2_un = src0_ub2   +  64;
        __ubuf__ float *src0_ub3_un = src0_ub3   +  64;

        vbr(max_0a, 0); vbr(max_0b, 0);
        vbr(max_1a, 0); vbr(max_1b, 0);
        vbr(max_2a, 0); vbr(max_2b, 0);
        vbr(max_3a, 0); vbr(max_3b, 0);

        for (uint16_t iter_m = 0; iter_m < uint16_t(ubN / 8); ++iter_m) {
            vlds(src_00a, src0_ub,     512, NORM, POST_UPDATE);
            vlds(src_00b, src0_ub_un,  512, NORM, POST_UPDATE);
            vmax(max_0a, max_0a, src_00a, preg_b32);
            vmax(max_0b, max_0b, src_00b, preg_b32);

            vlds(src_01a, src0_ub1,    512, NORM, POST_UPDATE);
            vlds(src_01b, src0_ub1_un, 512, NORM, POST_UPDATE);
            vmax(max_1a, max_1a, src_01a, preg_b32);
            vmax(max_1b, max_1b, src_01b, preg_b32);

            vlds(src_02a, src0_ub2,    512, NORM, POST_UPDATE);
            vlds(src_02b, src0_ub2_un, 512, NORM, POST_UPDATE);
            vmax(max_2a, max_2a, src_02a, preg_b32);
            vmax(max_2b, max_2b, src_02b, preg_b32);

            vlds(src_03a, src0_ub3,    512, NORM, POST_UPDATE);
            vlds(src_03b, src0_ub3_un, 512, NORM, POST_UPDATE);
            vmax(max_3a, max_3a, src_03a, preg_b32);
            vmax(max_3b, max_3b, src_03b, preg_b32);
        }

        // Load prev_global_max while local max is reducing.
        vector_f32 v_prev_max;
        vlds(v_prev_max, new_global_max_Ptr, 0, NORM);

        vmax(max_0a, max_0a, max_1a, preg_b32);
        vmax(max_0b, max_0b, max_1b, preg_b32);
        vmax(max_2a, max_2a, max_3a, preg_b32);
        vmax(max_2b, max_2b, max_3b, preg_b32);
        vmax(max_0a, max_0a, max_2a, preg_b32);
        vmax(max_0b, max_0b, max_2b, preg_b32);
        vmax(max_0a, max_0a, max_0b, preg_b32);

        // new_global_max = max(prev, local)
        vmax(max_0a, max_0a, v_prev_max, preg_b32);
        vsts(max_0a, new_global_max_Ptr, 0, NORM_B32, preg_b32);

        // exp_max = exp(scale * (prev_max - new_max))
        vmuls(max_0a,     max_0a,     scale, preg_b32);
        vmuls(v_prev_max, v_prev_max, scale, preg_b32);
        vexpdif(v_prev_max, v_prev_max, max_0a, preg_b32, PART_ODD);
        vsts(v_prev_max, exp_max_Ptr, 0, NORM_B32, preg_b32);

        vector_f32 v_sum0, v_sum1;
        vbr(v_sum0, 0);
        vbr(v_sum1, 0);

        __ubuf__ float *src_p0 = input_x_Ptr;
        __ubuf__ half  *nz_p0  = nz_dst_base;

        for (uint16_t r = 0; r < uint16_t(ubN / 2); ++r) {
            vector_f32 v_x0, v_x1;
            vector_f32 v_exp0, v_exp1;
            vector_f16 v_h0, v_h1;
            vector_f16 v_pack0, v_pack1;
            vector_f16 v_packa;

            vlds(v_x0, src_p0, 64, NORM, POST_UPDATE);
            vlds(v_x1, src_p0, 64, NORM, POST_UPDATE);

            vmuls(v_x0, v_x0, scale, preg_b32);
            vmuls(v_x1, v_x1, scale, preg_b32);

            vexpdif(v_exp0, v_x0, max_0a, preg_b32, PART_ODD);
            vexpdif(v_exp1, v_x1, max_0a, preg_b32, PART_ODD);

            vadd(v_sum0, v_sum0, v_exp0, preg_b32, MODE_ZEROING);
            vadd(v_sum1, v_sum1, v_exp1, preg_b32, MODE_ZEROING);

            vcvt(v_h0, v_exp0, preg_b16, ROUND_A, RS_DISABLE, PART_EVEN);
            vcvt(v_h1, v_exp1, preg_b16, ROUND_A, RS_DISABLE, PART_EVEN);

            // Pack even-lane halves into contiguous lower 64 lanes.
            vdintlv(v_pack0, v_packa, v_h0, v_h0);
            vdintlv(v_pack1, v_packa, v_h1, v_h1);

            vsstb((vector_u16 &)v_pack0, (__ubuf__ uint16_t *&)nz_p0,
                  cfgVsstb, preg_4blk, POST_UPDATE);
            vsstb((vector_u16 &)v_pack1, (__ubuf__ uint16_t *&)nz_p0,
                  cfgVsstb, preg_4blk, POST_UPDATE);
        }

        vadd(v_sum0, v_sum0, v_sum1, preg_b32, MODE_ZEROING);

        // Rescale running global_sum: new_sum = exp_max * prev_sum + local_sum
        vector_f32 v_prev_sum;
        vlds(v_prev_sum, new_global_sum_Ptr, 0, NORM);
        vmul(v_prev_sum, v_prev_sum, v_prev_max, preg_b32, MODE_ZEROING);   // re-use scaled exp_max in v_prev_max
        vadd(v_sum0, v_sum0, v_prev_sum, preg_b32, MODE_ZEROING);
        vsts(v_sum0, new_global_sum_Ptr, 0, NORM_B32, preg_b32);
    }
}

// -----------------------------------------------------------------------------
// Macro dispatch — identical signature shape to the baseline so the kernel
// can call this with the same call site, but writes nzConvBuffer (NZ+1)
// instead of x_expT (ND).  The caller must skip the TMOV ND→NZ step.
// -----------------------------------------------------------------------------
template <bool init = false, int HEAD_SIZE, bool CAUSAL_MASK,
          typename ReduceTileD1, typename TileDataNZ, typename TileDataS1>
AICORE inline void
pto_macro_fa_softmax_dn_opus(TileDataNZ __out__ nz_x_exp,
                             TileDataS1 __in__ input_x,
                             ReduceTileD1 __out__ local_max,
                             ReduceTileD1 __out__ local_sum,
                             ReduceTileD1 __in__  new_global_max,
                             ReduceTileD1 __out__ new_global_sum,
                             ReduceTileD1 __out__ exp_max,
                             int s0_index, int s1_index)
{
    if constexpr (init) {
        softmax_opus_fused_fa_dn_init_impl<HEAD_SIZE, CAUSAL_MASK,
                                           ReduceTileD1, TileDataNZ, TileDataS1>(
            nz_x_exp, input_x, local_max, local_sum,
            new_global_max, new_global_sum, exp_max,
            s0_index, s1_index);
    } else {
        softmax_opus_fused_fa_dn_not_init_impl<HEAD_SIZE, CAUSAL_MASK,
                                               ReduceTileD1, TileDataNZ, TileDataS1>(
            nz_x_exp, input_x, local_max, local_sum,
            new_global_max, new_global_sum, exp_max,
            s0_index, s1_index);
    }
}

} // namespace pto

#endif // PTO_MACRO_FA_SOFTMAX_DN_OPUS_HPP
