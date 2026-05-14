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
//   ITERATION 2  (this file).
//
//   Iteration 1 (saved as pto_macro_fa_dn_softmax_opus_iter1.hpp) hit a
//   measured 729-tick fused VF (down from 833 baseline = -12.5 %) but
//   exposed three PMU-confirmed bottlenecks all at ~65 % of the VF window:
//     - c_cycle_phy_vreg_stall        = 464   (rename file saturated)
//     - c_cycle_shq_st_src_not_ready  = 490   (vsstb waits on vdintlv)
//     - c_cycle_idu_not_disp.vec_stall= 480   (backend back-pressure)
//
//   ITERATION 2 STRATEGY — break the producer→consumer RAW barrier by
//   increasing inner-loop ILP from 2 rows/iter to 4 rows/iter, and split
//   the sum accumulator from 2-way to 4-way to remove the vadd RAW chain.
//
// FUSION TARGETS (vs the baseline pto_macro_fa_dn_softmax.hpp):
//
//   1. Direct ND→NZ+1 store using VSSTB.   (kept from iter 1)
//      The baseline writes Phase-2 results (FP16 x_exp) into x_expT in ND
//      layout, then a separate TMOV VF re-loads ND and stores NZ+1 into
//      nzConvBuffer.  Here Phase-2 writes the NZ+1 buffer DIRECTLY using
//      vsstb with block_stride = Cube_S1 + 1 = 129 (4 active blocks per row
//      for the [128,64] tile).  This eliminates the entire UB→UB TMOV VF
//      and its store-load round-trip cost.
//
//   2. vcvt replaces vmulscvt(scale=1.0).   (kept from iter 1)
//      Baseline used vmulscvt(_,_,1.0f,preg,PART_EVEN) — an LNEXP-class FMA
//      that occupies both MUL and ADD pipes, with throughput 1/cyc.  When
//      the scalar is 1.0 the multiply is a no-op and a plain vcvt(half,
//      float, preg, ROUND_A, RS_DISABLE, PART_EVEN) does the same conversion
//      on the ADD pipe alone, freeing the MUL pipe for the per-row scaling
//      vmuls — increasing MUL/ADD dual-issue parallelism.
//
//   3. **NEW — 4-row inner unroll.**   The Phase-2 loop processes 4 rows
//      per iter (down from 2 in iter 1, from 1 in baseline).  This lets the
//      hardware overlap 4 independent vdintlv→vsstb RAW chains, replacing
//      the iter-1 single-chain stall with a 4-deep pipeline.  In the
//      theoretical schedule the steady-state per-row cost falls from
//      ~5.7 cyc (iter 1) to ~2 cyc (LSU-port bound), matching the floor.
//
//   4. **NEW — 4-way split sum accumulator.**   v_sum0/1/2/3 each accumulate
//      32 rows' worth of exp values, then a final 3-stage tree-reduce
//      produces the single new_global_sum.  Breaks the back-to-back
//      RAW between successive vadds (iter 1 had only 2 sum streams ⇒
//      RAW distance 2; iter 2 has 4 ⇒ RAW distance 4, comfortably above
//      vadd FP32 latency of ~5 cyc).
//
//   5. **NEW — vdintlv input duplication.**   We continue to use
//      `vdintlv(v_pack, _, v_h, v_h)` to pack PART_EVEN-sparse vcvt output
//      into the lower 64 lanes of v_pack.  The "duplicate same vreg into
//      both inputs" pattern is now documented in ND→NZ Patterns §4.7.
//
// Tile geometry (per vec core, for the FA4 [128,64] config):
//   - input_x   : Tile_S1=128 rows × Vec_S0=64 cols FP32, ND row-major UB.
//   - x_exp_NZ  : nzConvBuffer, shape [Cube_S1+1=129, Vec_S0=64] half, NZ+1
//                 (BLayout::ColMajor + CompactMode::RowPlusOne, 4 blocks/row).
//   - new_global_max, new_global_sum: ReduceTileF_T [1, 64] FP32.
//
// Roofline (Phase 2, [128, 64] FP16 path) — same as iter 1:
//   LNEXP FP32 vexpdif: thpt=4 cyc, lat=15.  Dual-issuable across EXQ0/EXQ1
//     ⇒ 128 vexpdif × 4 cyc / 2 = 256 LNEXP-cycles.
//   LSU: 128 vlds + 128 vsstb = 256 LSU cyc.
//   ADD: 128 vadd + 128 vcvt   = 256 ADD cyc.
//   MUL: 128 vmuls             = 128 cyc (slack 128).
//   SLIDE: 128 vdintlv          = 128 cyc (slack 128).
//   Phase-2 floor ≈ max(256, 256, 256) = 256 cyc.
//
//   Why iter 1 missed this floor:
//     - 2-way unroll left only 2 vdintlv→vsstb chains in flight.  With
//       vdintlv latency ≈ 5 cyc and vsstb back-to-back, each iter served
//       2 vsstb in ~5 cyc + 1 cyc = ~6 cyc, so 64 iters × 6 ≈ 384 cyc on
//       the SLIDE/LSU critical path (vs the 256-cyc roofline).
//     - Iter 2 with 4-way unroll: each iter overlaps 4 vdintlv (4 SLIDE
//       cycles) with the wait, then 4 vsstb back-to-back (4 LSU cyc).
//       Per-iter critical path ≈ max(4 SLIDE + lat-overlap, 4 LSU) ≈ 5 cyc,
//       so 32 iters × 5 ≈ 160 cyc on the SLIDE/LSU path.  Now LNEXP/ADD/LSU
//       all converge near 256 cyc — the roofline is genuinely reachable.
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
        // Phase 2 — 4-row inner unroll with 4-way split sum accumulator.
        //
        // Loop body processes 4 rows per iter; each row issues:
        //   vlds   row r         (LSU)
        //   vmuls  v_x *= scale  (MUL)
        //   vexpdif v_exp = exp(v_x - max) PART_ODD  (LNEXP, FP32, thpt=4)
        //   vadd   v_sum[r%4] += v_exp               (ADD; 4-way split)
        //   vcvt   v_h = (half)v_exp PART_EVEN        (ADD)
        //   vdintlv v_pack, _, v_h, v_h               (SLIDE, packs even lanes)
        //   vsstb  v_pack → NZ+1, 4 active blocks     (LSU)
        //
        // 32 outer iters × 4 rows = 128 rows.  After the loop a 2-stage tree
        // reduction collapses v_sum[0..3] into v_sum0.
        // -----------------------------------------------------------------
        vector_f32 v_sum0, v_sum1, v_sum2, v_sum3;
        vbr(v_sum0, 0);
        vbr(v_sum1, 0);
        vbr(v_sum2, 0);
        vbr(v_sum3, 0);

        __ubuf__ float *src_p0 = input_x_Ptr;
        __ubuf__ half  *nz_p0  = nz_dst_base;

        // Per 4-row iter: src_p0 advances by 4 × 64 = 256 f32 (= 1024 B);
        //                 nz_p0 advances by 4 blocks (= 128 B) via the four
        //                 vsstb POST_UPDATE * repeat_stride = 1.

        for (uint16_t r4 = 0; r4 < uint16_t(ubN / 4); ++r4) {
            vector_f32 v_x0, v_x1, v_x2, v_x3;
            vector_f32 v_exp0, v_exp1, v_exp2, v_exp3;
            vector_f16 v_h0, v_h1, v_h2, v_h3;
            vector_f16 v_pack0, v_pack1, v_pack2, v_pack3;
            vector_f16 v_packa;    // garbage sink for vdintlv's second output

            // ---- Stage 1 : 4 vlds, fan-out on LSU ----------------------
            vlds(v_x0, src_p0, 64, NORM, POST_UPDATE);   // row 4r+0
            vlds(v_x1, src_p0, 64, NORM, POST_UPDATE);   // row 4r+1
            vlds(v_x2, src_p0, 64, NORM, POST_UPDATE);   // row 4r+2
            vlds(v_x3, src_p0, 64, NORM, POST_UPDATE);   // row 4r+3

            // ---- Stage 2 : 4 vmuls, MUL pipe (back-to-back ok) ---------
            vmuls(v_x0, v_x0, scale, preg_b32);
            vmuls(v_x1, v_x1, scale, preg_b32);
            vmuls(v_x2, v_x2, scale, preg_b32);
            vmuls(v_x3, v_x3, scale, preg_b32);

            // ---- Stage 3 : 4 vexpdif, LNEXP (thpt=4) -------------------
            vexpdif(v_exp0, v_x0, max_0a, preg_b32, PART_ODD);
            vexpdif(v_exp1, v_x1, max_0a, preg_b32, PART_ODD);
            vexpdif(v_exp2, v_x2, max_0a, preg_b32, PART_ODD);
            vexpdif(v_exp3, v_x3, max_0a, preg_b32, PART_ODD);

            // ---- Stage 4 : sum accumulators (4 independent streams) ----
            // 4-way split removes the back-to-back RAW chain that limited
            // iter 1 to 2 sum streams.  Each stream accumulates 32 rows.
            vadd(v_sum0, v_sum0, v_exp0, preg_b32, MODE_ZEROING);
            vadd(v_sum1, v_sum1, v_exp1, preg_b32, MODE_ZEROING);
            vadd(v_sum2, v_sum2, v_exp2, preg_b32, MODE_ZEROING);
            vadd(v_sum3, v_sum3, v_exp3, preg_b32, MODE_ZEROING);

            // ---- Stage 5 : narrowing vcvt FP32→FP16 (PART_EVEN) --------
            // ROUND_A = round-to-nearest-ties-away (matches baseline TCVT).
            // Output is sparse: 64 valid halves live in even lanes 0..126.
            vcvt(v_h0, v_exp0, preg_b16, ROUND_A, RS_DISABLE, PART_EVEN);
            vcvt(v_h1, v_exp1, preg_b16, ROUND_A, RS_DISABLE, PART_EVEN);
            vcvt(v_h2, v_exp2, preg_b16, ROUND_A, RS_DISABLE, PART_EVEN);
            vcvt(v_h3, v_exp3, preg_b16, ROUND_A, RS_DISABLE, PART_EVEN);

            // ---- Stage 6 : pack even-lane gaps with vdintlv (SLIDE) ----
            // vdintlv(out0, out1, in0, in1) returns out0 = concat[0::2],
            // out1 = concat[1::2] over the 256-half concatenation in0||in1.
            // With in0=in1=v_h (PART_EVEN sparse), out0[0..63] = packed 64
            // valid halves, out0[64..127] = duplicate (later masked by
            // preg_4blk), out1 = zeros (garbage sink).
            // See ND→NZ Patterns §4.7 for the formal pack identity proof.
            vdintlv(v_pack0, v_packa, v_h0, v_h0);
            vdintlv(v_pack1, v_packa, v_h1, v_h1);
            vdintlv(v_pack2, v_packa, v_h2, v_h2);
            vdintlv(v_pack3, v_packa, v_h3, v_h3);

            // ---- Stage 7 : 4 vsstb to NZ+1, 4 active blocks each ------
            // block_stride=129 (avoids UB bank conflicts, NZ+1 theorem 2),
            // repeat_stride=1 advances nz_p0 to next row's slot.  Four
            // back-to-back vsstb fill row slots 4r+0 .. 4r+3.  By the time
            // vsstb #0 fires, vdintlv #0 has had 4 SLIDE cycles to retire
            // (in steady state), so the RAW gap is hidden by the unroll.
            vsstb((vector_u16 &)v_pack0, (__ubuf__ uint16_t *&)nz_p0,
                  cfgVsstb, preg_4blk, POST_UPDATE);
            vsstb((vector_u16 &)v_pack1, (__ubuf__ uint16_t *&)nz_p0,
                  cfgVsstb, preg_4blk, POST_UPDATE);
            vsstb((vector_u16 &)v_pack2, (__ubuf__ uint16_t *&)nz_p0,
                  cfgVsstb, preg_4blk, POST_UPDATE);
            vsstb((vector_u16 &)v_pack3, (__ubuf__ uint16_t *&)nz_p0,
                  cfgVsstb, preg_4blk, POST_UPDATE);
        }

        // Tree reduce v_sum[0..3] → v_sum0 and write new_global_sum.
        vadd(v_sum0, v_sum0, v_sum1, preg_b32, MODE_ZEROING);
        vadd(v_sum2, v_sum2, v_sum3, preg_b32, MODE_ZEROING);
        vadd(v_sum0, v_sum0, v_sum2, preg_b32, MODE_ZEROING);
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

        // ---- Phase 2 (not_init): same 4-row unroll + 4-split sum as init.
        vector_f32 v_sum0, v_sum1, v_sum2, v_sum3;
        vbr(v_sum0, 0);
        vbr(v_sum1, 0);
        vbr(v_sum2, 0);
        vbr(v_sum3, 0);

        __ubuf__ float *src_p0 = input_x_Ptr;
        __ubuf__ half  *nz_p0  = nz_dst_base;

        for (uint16_t r4 = 0; r4 < uint16_t(ubN / 4); ++r4) {
            vector_f32 v_x0, v_x1, v_x2, v_x3;
            vector_f32 v_exp0, v_exp1, v_exp2, v_exp3;
            vector_f16 v_h0, v_h1, v_h2, v_h3;
            vector_f16 v_pack0, v_pack1, v_pack2, v_pack3;
            vector_f16 v_packa;

            vlds(v_x0, src_p0, 64, NORM, POST_UPDATE);
            vlds(v_x1, src_p0, 64, NORM, POST_UPDATE);
            vlds(v_x2, src_p0, 64, NORM, POST_UPDATE);
            vlds(v_x3, src_p0, 64, NORM, POST_UPDATE);

            vmuls(v_x0, v_x0, scale, preg_b32);
            vmuls(v_x1, v_x1, scale, preg_b32);
            vmuls(v_x2, v_x2, scale, preg_b32);
            vmuls(v_x3, v_x3, scale, preg_b32);

            vexpdif(v_exp0, v_x0, max_0a, preg_b32, PART_ODD);
            vexpdif(v_exp1, v_x1, max_0a, preg_b32, PART_ODD);
            vexpdif(v_exp2, v_x2, max_0a, preg_b32, PART_ODD);
            vexpdif(v_exp3, v_x3, max_0a, preg_b32, PART_ODD);

            vadd(v_sum0, v_sum0, v_exp0, preg_b32, MODE_ZEROING);
            vadd(v_sum1, v_sum1, v_exp1, preg_b32, MODE_ZEROING);
            vadd(v_sum2, v_sum2, v_exp2, preg_b32, MODE_ZEROING);
            vadd(v_sum3, v_sum3, v_exp3, preg_b32, MODE_ZEROING);

            vcvt(v_h0, v_exp0, preg_b16, ROUND_A, RS_DISABLE, PART_EVEN);
            vcvt(v_h1, v_exp1, preg_b16, ROUND_A, RS_DISABLE, PART_EVEN);
            vcvt(v_h2, v_exp2, preg_b16, ROUND_A, RS_DISABLE, PART_EVEN);
            vcvt(v_h3, v_exp3, preg_b16, ROUND_A, RS_DISABLE, PART_EVEN);

            vdintlv(v_pack0, v_packa, v_h0, v_h0);
            vdintlv(v_pack1, v_packa, v_h1, v_h1);
            vdintlv(v_pack2, v_packa, v_h2, v_h2);
            vdintlv(v_pack3, v_packa, v_h3, v_h3);

            vsstb((vector_u16 &)v_pack0, (__ubuf__ uint16_t *&)nz_p0,
                  cfgVsstb, preg_4blk, POST_UPDATE);
            vsstb((vector_u16 &)v_pack1, (__ubuf__ uint16_t *&)nz_p0,
                  cfgVsstb, preg_4blk, POST_UPDATE);
            vsstb((vector_u16 &)v_pack2, (__ubuf__ uint16_t *&)nz_p0,
                  cfgVsstb, preg_4blk, POST_UPDATE);
            vsstb((vector_u16 &)v_pack3, (__ubuf__ uint16_t *&)nz_p0,
                  cfgVsstb, preg_4blk, POST_UPDATE);
        }

        // Tree reduce v_sum[0..3] -> v_sum0.
        vadd(v_sum0, v_sum0, v_sum1, preg_b32, MODE_ZEROING);
        vadd(v_sum2, v_sum2, v_sum3, preg_b32, MODE_ZEROING);
        vadd(v_sum0, v_sum0, v_sum2, preg_b32, MODE_ZEROING);

        // Rescale running global_sum: new_sum = exp_max * prev_sum + local_sum
        vector_f32 v_prev_sum;
        vlds(v_prev_sum, new_global_sum_Ptr, 0, NORM);
        vmul(v_prev_sum, v_prev_sum, v_prev_max, preg_b32, MODE_ZEROING);
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
