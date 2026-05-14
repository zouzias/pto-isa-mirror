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
//   ITERATION 3  (this file).
//
//   Iteration history:
//     iter 1 (pto_macro_..._iter1.hpp): 2-row Phase-2 unroll, single shared
//                         src/nz pointer with POST_UPDATE chain.
//                         VF1=729 ticks, whole-prog cube=9416, vec=10215/10114,
//                         idu_ports_stall.asu ≈ 480 cyc (AGS RAW chain).
//     iter 2 (pto_macro_..._iter2.hpp): 4-row Phase-2 unroll, SAME single
//                         POST_UPDATE chain — VF1=716 (−13) but whole-prog
//                         regressed +200 across all cores.
//     iter 2.5 (pto_macro_..._iter2p5.hpp): isolation control — iter 1's
//                         2-row unroll + iter 2's 4-way sum split.
//     iter 3-U2 (pto_macro_..._iter3_u2.hpp): U=2 split-pointer variant.
//                         VF1=714, cube=9399, vec=10097/10198.  asu=10,
//                         phy_vreg_stall=524.  Higher VF and slightly
//                         higher whole-prog tick than U=4.
//     iter 3-U4 (this file, ALSO pto_macro_..._iter3_u4.hpp):
//                         WINNER of the unroll sweep.  VF1=697 (−32 vs
//                         iter 1, −19 vs iter 2), cube=9374 (BEST cube
//                         across all variants, −42 vs iter 1, −94 vs
//                         unfused baseline).  asu=14 (vs ~480 in iters
//                         1 & 2 — AGS RAW chain ELIMINATED).
//                         Five independent wins:
//                         (a) 4-row Phase-2 unroll;
//                         (b) **4 INDEPENDENT input/output pointers**, each
//                             pre-offset by k×row and POST_UPDATEd by its OWN
//                             full-iter stride — no shared AGS chain;
//                         (c) **fully static pregs** hoisted once at the top
//                             of `__VEC_SCOPE__`;
//                         (d) **4-way split sum accumulator**, one stream per
//                             unrolled row, with a final 3-stage tree-reduce —
//                             **no `if` inside the vec loop**;
//                         (e) **vbr-free initialisation** — iter 0 is PEELED
//                             so the first 8 vlds (Phase 1) and the first 4
//                             vexpdif outputs (Phase 2) write directly into
//                             the max_* / v_sum* accumulators.  Removes 12
//                             vbr cycles AND the iter-0 vmax/vadd that those
//                             vbrs forced (16 + 4 cyc total).
//
//   ROOT CAUSE OF ITER 1/2 BOTTLENECKS — the AGS pointer chain.
//
//   Iter 1 and iter 2 issued 4 vlds back-to-back through one src pointer
//   with POST_UPDATE.  Each vlds's address generator (AGS) reads the
//   scalar pointer, computes the load address, and writes back
//   `ptr += stride` — a per-instruction scalar RAW dependency.  Successive
//   vlds therefore *serialise on the pointer*, not on the LSU port.  Same
//   pattern poisons the vsstb side.  The PMU c_cycle_idu_not_disp.vec_stall
//   counter (480 cyc in iter 1) is the visible signature.
//
//   FIX (iter 3): declare 4 independent pointers — one per row of the
//   unrolled iter — each pre-offset by k * VL_in_elements from the base.
//   Each vlds/vsstb POST_UPDATEs its OWN pointer by the loop step.  The
//   AGS sees 4 totally independent scalar registers; no inter-instr RAW.
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
//   3. **4-row inner unroll.**   The Phase-2 loop processes 4 rows
//      per iter (down from 2 in iter 1, from 1 in baseline).
//
//   4. **4-way split sum accumulator.**   v_sum0..3 each accumulate
//      32 rows' worth of exp values; round-robin per pair of rows (so
//      each iter touches exactly 2 of the 4 streams), then a final 3-stage
//      tree-reduce produces the single new_global_sum.  Breaks the
//      back-to-back vadd RAW chain of iter 1.
//
//   5. **vdintlv pack of PART_EVEN gaps.**   We continue to use
//      `vdintlv(v_pack, _, v_h, v_h)` to pack PART_EVEN-sparse vcvt output
//      into the lower 64 lanes of v_pack.  See ND→NZ Patterns §4.7.
//
//   6. **NEW — 4 independent input/output pointers (no shared POST_UPDATE
//      chain).**   src_p0..3 = input_x_Ptr + {0,1,2,3} × 64 fp32,
//      nz_p0..3 = nz_dst_base + {0,1,2,3} blocks.  Each vlds and each
//      vsstb POST_UPDATEs its OWN pointer once per iter (by 4×row stride
//      = 256 fp32 for src, by 4 blocks = 64 halves for nz).  See VF Fusion
//      Guide §14 Step 10 + load-dist-modes.md for the formal pattern.
//
//   7. **NEW — fully static pregs.**   preg_b32 / preg_b16 / preg_4blk are
//      created exactly once at the top of `__VEC_SCOPE__` from PAT_ALL
//      or a constexpr count and are read-only thereafter.  No predicate
//      construction inside the inner loop.
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
        // Static predicates — created once, read-only inside both phases.
        // No predicate construction inside the inner loops.
        // -----------------------------------------------------------------
        vector_bool preg_b32 = pset_b32(PAT_ALL);     // 64 f32 lanes active
        vector_bool preg_b16 = pset_b16(PAT_ALL);     // 128 f16 lanes active
        // 64-half predicate = lower 128 B = 4 blocks active.  This is what
        // VSSTB uses to scatter 4 blocks to the NZ+1 column-group.
        // POST_UPDATE here is unavoidable (the intrinsic requires it) but
        // harmless — sreg_count is dead after this single call.
        uint32_t sreg_count  = (uint32_t)(ubM);       // 64 halves
        vector_bool preg_4blk = plt_b16(sreg_count, POST_UPDATE);

        // -----------------------------------------------------------------
        // Phase 1: column max over all 128 rows.
        //
        // 8 parallel max accumulators (max_*) on 8 INDEPENDENT pointers
        // (one per row-mod-8).  Each pointer is advanced by 8 rows × 64
        // = 512 fp32 per iter.
        //
        // **No vbr.**  Iter 0 is PEELED: the first 8 vlds write directly
        // into the max_* accumulators.  Subsequent 15 iters do vlds + vmax.
        // This saves (8 vbr) + (8 vmax in iter 0) = 16 cyc, removes the
        // initial vmax-against-zero work, and also gives a tighter max
        // (we no longer clamp negative inputs to 0 via max(0, x)).
        // -----------------------------------------------------------------
        vector_f32 src_00a, src_01a, src_02a, src_03a;
        vector_f32 src_00b, src_01b, src_02b, src_03b;
        vector_f32 max_0a, max_1a, max_2a, max_3a;
        vector_f32 max_0b, max_1b, max_2b, max_3b;

        // 8 INDEPENDENT pointers — no shared AGS chain.
        __ubuf__ float *p_00a = input_x_Ptr + 0 * 64;
        __ubuf__ float *p_00b = input_x_Ptr + 1 * 64;
        __ubuf__ float *p_01a = input_x_Ptr + 2 * 64;
        __ubuf__ float *p_01b = input_x_Ptr + 3 * 64;
        __ubuf__ float *p_02a = input_x_Ptr + 4 * 64;
        __ubuf__ float *p_02b = input_x_Ptr + 5 * 64;
        __ubuf__ float *p_03a = input_x_Ptr + 6 * 64;
        __ubuf__ float *p_03b = input_x_Ptr + 7 * 64;

        // Iter 0 PEELED — directly initialise accumulators (no vbr).
        vlds(max_0a, p_00a, 512, NORM, POST_UPDATE);
        vlds(max_0b, p_00b, 512, NORM, POST_UPDATE);
        vlds(max_1a, p_01a, 512, NORM, POST_UPDATE);
        vlds(max_1b, p_01b, 512, NORM, POST_UPDATE);
        vlds(max_2a, p_02a, 512, NORM, POST_UPDATE);
        vlds(max_2b, p_02b, 512, NORM, POST_UPDATE);
        vlds(max_3a, p_03a, 512, NORM, POST_UPDATE);
        vlds(max_3b, p_03b, 512, NORM, POST_UPDATE);

        // Remaining 15 iters — vlds + vmax.
        for (uint16_t iter_m = 0; iter_m < uint16_t(ubN / 8 - 1); ++iter_m) {
            vlds(src_00a, p_00a, 512, NORM, POST_UPDATE);
            vlds(src_00b, p_00b, 512, NORM, POST_UPDATE);
            vmax(max_0a, max_0a, src_00a, preg_b32);
            vmax(max_0b, max_0b, src_00b, preg_b32);

            vlds(src_01a, p_01a, 512, NORM, POST_UPDATE);
            vlds(src_01b, p_01b, 512, NORM, POST_UPDATE);
            vmax(max_1a, max_1a, src_01a, preg_b32);
            vmax(max_1b, max_1b, src_01b, preg_b32);

            vlds(src_02a, p_02a, 512, NORM, POST_UPDATE);
            vlds(src_02b, p_02b, 512, NORM, POST_UPDATE);
            vmax(max_2a, max_2a, src_02a, preg_b32);
            vmax(max_2b, max_2b, src_02b, preg_b32);

            vlds(src_03a, p_03a, 512, NORM, POST_UPDATE);
            vlds(src_03b, p_03b, 512, NORM, POST_UPDATE);
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
        // Phase 2 — ITER 3: 4-row unroll, 4 independent pointers, 4 sums.
        //
        // The key fix vs iter 1/2: the 4 vlds and 4 vsstb of each iter use
        // **4 totally independent scalar pointers** (one per unrolled row).
        // Each pointer is POST_UPDATEd by 4 × row stride per iter — that's
        // one AGS post-update per pointer per iter, no inter-instruction
        // RAW chain at the AGS.
        //
        // src layout (ND, row-major fp32, [128 rows × 64 elems]):
        //   row r lives at input_x_Ptr + r * 64.
        //   src_pk = input_x_Ptr + k * 64  (k = 0..3) covers row k each iter.
        //   Each iter advances src_pk by 4 × 64 = 256 fp32 (= 4 rows).
        //
        // nz_p layout (NZ+1, [4 col-groups × 129 rows], half):
        //   row r's column-group-c block is at nz_dst_base
        //     + (c * 129 + r) * 32 bytes = + (c * 129 + r) * 16 halves.
        //   The vsstb with block_stride=129 scatters one source vreg's
        //   4 active blocks to col-groups 0..3 for the SAME row.  So
        //   nz_pk starts at row k (= k blocks) and advances by 4 blocks
        //   (= 4 rows) per iter.
        //   cfgVsstbIter encodes (block_stride=129, repeat_stride=4).
        // -----------------------------------------------------------------
        constexpr uint32_t cfgVsstbIter = (129u << 16) | 4u;   // rep_stride=4 blocks

        // 4 independent src pointers — pre-offset by row.
        __ubuf__ float *src_p0 = input_x_Ptr + 0 * 64;
        __ubuf__ float *src_p1 = input_x_Ptr + 1 * 64;
        __ubuf__ float *src_p2 = input_x_Ptr + 2 * 64;
        __ubuf__ float *src_p3 = input_x_Ptr + 3 * 64;

        // 4 independent nz pointers — pre-offset by 1 block (= 16 halves) per row.
        __ubuf__ half  *nz_p0  = nz_dst_base + 0 * 16;
        __ubuf__ half  *nz_p1  = nz_dst_base + 1 * 16;
        __ubuf__ half  *nz_p2  = nz_dst_base + 2 * 16;
        __ubuf__ half  *nz_p3  = nz_dst_base + 3 * 16;

        vector_f32 v_x0, v_x1, v_x2, v_x3;
        vector_f32 v_exp0, v_exp1, v_exp2, v_exp3;
        vector_f32 v_sum0, v_sum1, v_sum2, v_sum3;
        vector_f16 v_h0, v_h1, v_h2, v_h3;
        vector_f16 v_pack0, v_pack1, v_pack2, v_pack3;
        vector_f16 v_packa;

        // -----------------------------------------------------------------
        // Phase-2 ITER 0 PEELED  (replaces 4 × vbr + 4 × vadd-against-zero).
        //
        // Writes the first exp directly into v_sum0..3 instead of using
        // vbr to zero them.  The vcvt and vsstb chains read from v_sum*
        // (which IS the first exp for iter 0) — semantically identical
        // to (vbr(v_sum, 0); vadd(v_sum, v_sum, v_exp); use(v_exp)).
        // -----------------------------------------------------------------
        vlds(v_x0, src_p0, 256, NORM, POST_UPDATE);
        vlds(v_x1, src_p1, 256, NORM, POST_UPDATE);
        vlds(v_x2, src_p2, 256, NORM, POST_UPDATE);
        vlds(v_x3, src_p3, 256, NORM, POST_UPDATE);

        vmuls(v_x0, v_x0, scale, preg_b32);
        vmuls(v_x1, v_x1, scale, preg_b32);
        vmuls(v_x2, v_x2, scale, preg_b32);
        vmuls(v_x3, v_x3, scale, preg_b32);

        // v_sum* receive the first exp (NO vbr, NO vadd-against-zero).
        vexpdif(v_sum0, v_x0, max_0a, preg_b32, PART_ODD);
        vexpdif(v_sum1, v_x1, max_0a, preg_b32, PART_ODD);
        vexpdif(v_sum2, v_x2, max_0a, preg_b32, PART_ODD);
        vexpdif(v_sum3, v_x3, max_0a, preg_b32, PART_ODD);

        vcvt(v_h0, v_sum0, preg_b16, ROUND_A, RS_DISABLE, PART_EVEN);
        vcvt(v_h1, v_sum1, preg_b16, ROUND_A, RS_DISABLE, PART_EVEN);
        vcvt(v_h2, v_sum2, preg_b16, ROUND_A, RS_DISABLE, PART_EVEN);
        vcvt(v_h3, v_sum3, preg_b16, ROUND_A, RS_DISABLE, PART_EVEN);

        vdintlv(v_pack0, v_packa, v_h0, v_h0);
        vdintlv(v_pack1, v_packa, v_h1, v_h1);
        vdintlv(v_pack2, v_packa, v_h2, v_h2);
        vdintlv(v_pack3, v_packa, v_h3, v_h3);

        vsstb((vector_u16 &)v_pack0, (__ubuf__ uint16_t *&)nz_p0,
              cfgVsstbIter, preg_4blk, POST_UPDATE);
        vsstb((vector_u16 &)v_pack1, (__ubuf__ uint16_t *&)nz_p1,
              cfgVsstbIter, preg_4blk, POST_UPDATE);
        vsstb((vector_u16 &)v_pack2, (__ubuf__ uint16_t *&)nz_p2,
              cfgVsstbIter, preg_4blk, POST_UPDATE);
        vsstb((vector_u16 &)v_pack3, (__ubuf__ uint16_t *&)nz_p3,
              cfgVsstbIter, preg_4blk, POST_UPDATE);

        // -----------------------------------------------------------------
        // Phase-2 loop iters 1..31 — standard 4-row body with vadd into v_sum*.
        // -----------------------------------------------------------------
        for (uint16_t r4 = 0; r4 < uint16_t(ubN / 4 - 1); ++r4) {
            // 4 vlds, INDEPENDENT pointers, each advances 4 rows / iter.
            vlds(v_x0, src_p0, 256, NORM, POST_UPDATE);
            vlds(v_x1, src_p1, 256, NORM, POST_UPDATE);
            vlds(v_x2, src_p2, 256, NORM, POST_UPDATE);
            vlds(v_x3, src_p3, 256, NORM, POST_UPDATE);

            vmuls(v_x0, v_x0, scale, preg_b32);
            vmuls(v_x1, v_x1, scale, preg_b32);
            vmuls(v_x2, v_x2, scale, preg_b32);
            vmuls(v_x3, v_x3, scale, preg_b32);

            vexpdif(v_exp0, v_x0, max_0a, preg_b32, PART_ODD);
            vexpdif(v_exp1, v_x1, max_0a, preg_b32, PART_ODD);
            vexpdif(v_exp2, v_x2, max_0a, preg_b32, PART_ODD);
            vexpdif(v_exp3, v_x3, max_0a, preg_b32, PART_ODD);

            // 4 independent sum streams — each used 31 more times here.
            vadd(v_sum0, v_sum0, v_exp0, preg_b32, MODE_ZEROING);
            vadd(v_sum1, v_sum1, v_exp1, preg_b32, MODE_ZEROING);
            vadd(v_sum2, v_sum2, v_exp2, preg_b32, MODE_ZEROING);
            vadd(v_sum3, v_sum3, v_exp3, preg_b32, MODE_ZEROING);

            vcvt(v_h0, v_exp0, preg_b16, ROUND_A, RS_DISABLE, PART_EVEN);
            vcvt(v_h1, v_exp1, preg_b16, ROUND_A, RS_DISABLE, PART_EVEN);
            vcvt(v_h2, v_exp2, preg_b16, ROUND_A, RS_DISABLE, PART_EVEN);
            vcvt(v_h3, v_exp3, preg_b16, ROUND_A, RS_DISABLE, PART_EVEN);

            // Pack PART_EVEN-sparse halves into contiguous lower 64 lanes.
            vdintlv(v_pack0, v_packa, v_h0, v_h0);
            vdintlv(v_pack1, v_packa, v_h1, v_h1);
            vdintlv(v_pack2, v_packa, v_h2, v_h2);
            vdintlv(v_pack3, v_packa, v_h3, v_h3);

            // 4 vsstb, INDEPENDENT pointers, each advances 4 blocks / iter.
            vsstb((vector_u16 &)v_pack0, (__ubuf__ uint16_t *&)nz_p0,
                  cfgVsstbIter, preg_4blk, POST_UPDATE);
            vsstb((vector_u16 &)v_pack1, (__ubuf__ uint16_t *&)nz_p1,
                  cfgVsstbIter, preg_4blk, POST_UPDATE);
            vsstb((vector_u16 &)v_pack2, (__ubuf__ uint16_t *&)nz_p2,
                  cfgVsstbIter, preg_4blk, POST_UPDATE);
            vsstb((vector_u16 &)v_pack3, (__ubuf__ uint16_t *&)nz_p3,
                  cfgVsstbIter, preg_4blk, POST_UPDATE);
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

    // VSSTB iter-stride config: block_stride=129 (NZ+1), repeat_stride=4 blocks.
    constexpr uint32_t cfgVsstbIter = (129u << 16) | 4u;

    __VEC_SCOPE__
    {
        // ---- Static predicates (once) ------------------------------------
        vector_bool preg_b32 = pset_b32(PAT_ALL);
        vector_bool preg_b16 = pset_b16(PAT_ALL);
        uint32_t sreg_count  = (uint32_t)(ubM);
        vector_bool preg_4blk = plt_b16(sreg_count, POST_UPDATE);

        // ---- Phase 1: column max — 8-way unroll, INDEPENDENT pointers,
        //              peeled iter 0 (no vbr).
        vector_f32 src_00a, src_01a, src_02a, src_03a;
        vector_f32 src_00b, src_01b, src_02b, src_03b;
        vector_f32 max_0a, max_1a, max_2a, max_3a;
        vector_f32 max_0b, max_1b, max_2b, max_3b;

        __ubuf__ float *p_00a = input_x_Ptr + 0 * 64;
        __ubuf__ float *p_00b = input_x_Ptr + 1 * 64;
        __ubuf__ float *p_01a = input_x_Ptr + 2 * 64;
        __ubuf__ float *p_01b = input_x_Ptr + 3 * 64;
        __ubuf__ float *p_02a = input_x_Ptr + 4 * 64;
        __ubuf__ float *p_02b = input_x_Ptr + 5 * 64;
        __ubuf__ float *p_03a = input_x_Ptr + 6 * 64;
        __ubuf__ float *p_03b = input_x_Ptr + 7 * 64;

        // Peeled iter 0 — direct load into accumulators (no vbr).
        vlds(max_0a, p_00a, 512, NORM, POST_UPDATE);
        vlds(max_0b, p_00b, 512, NORM, POST_UPDATE);
        vlds(max_1a, p_01a, 512, NORM, POST_UPDATE);
        vlds(max_1b, p_01b, 512, NORM, POST_UPDATE);
        vlds(max_2a, p_02a, 512, NORM, POST_UPDATE);
        vlds(max_2b, p_02b, 512, NORM, POST_UPDATE);
        vlds(max_3a, p_03a, 512, NORM, POST_UPDATE);
        vlds(max_3b, p_03b, 512, NORM, POST_UPDATE);

        for (uint16_t iter_m = 0; iter_m < uint16_t(ubN / 8 - 1); ++iter_m) {
            vlds(src_00a, p_00a, 512, NORM, POST_UPDATE);
            vlds(src_00b, p_00b, 512, NORM, POST_UPDATE);
            vmax(max_0a, max_0a, src_00a, preg_b32);
            vmax(max_0b, max_0b, src_00b, preg_b32);

            vlds(src_01a, p_01a, 512, NORM, POST_UPDATE);
            vlds(src_01b, p_01b, 512, NORM, POST_UPDATE);
            vmax(max_1a, max_1a, src_01a, preg_b32);
            vmax(max_1b, max_1b, src_01b, preg_b32);

            vlds(src_02a, p_02a, 512, NORM, POST_UPDATE);
            vlds(src_02b, p_02b, 512, NORM, POST_UPDATE);
            vmax(max_2a, max_2a, src_02a, preg_b32);
            vmax(max_2b, max_2b, src_02b, preg_b32);

            vlds(src_03a, p_03a, 512, NORM, POST_UPDATE);
            vlds(src_03b, p_03b, 512, NORM, POST_UPDATE);
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

        // new_global_max = max(prev, local).
        vmax(max_0a, max_0a, v_prev_max, preg_b32);
        vsts(max_0a, new_global_max_Ptr, 0, NORM_B32, preg_b32);

        // exp_max = exp(scale * (prev_max - new_max)).
        vmuls(max_0a,     max_0a,     scale, preg_b32);
        vmuls(v_prev_max, v_prev_max, scale, preg_b32);
        vexpdif(v_prev_max, v_prev_max, max_0a, preg_b32, PART_ODD);
        vsts(v_prev_max, exp_max_Ptr, 0, NORM_B32, preg_b32);

        // ---- Phase 2 (not_init): 4-row unroll, 4 INDEPENDENT pointers,
        //              4 sum streams, peeled iter 0 (no vbr, no `if`).
        __ubuf__ float *src_p0 = input_x_Ptr + 0 * 64;
        __ubuf__ float *src_p1 = input_x_Ptr + 1 * 64;
        __ubuf__ float *src_p2 = input_x_Ptr + 2 * 64;
        __ubuf__ float *src_p3 = input_x_Ptr + 3 * 64;
        __ubuf__ half  *nz_p0  = nz_dst_base + 0 * 16;
        __ubuf__ half  *nz_p1  = nz_dst_base + 1 * 16;
        __ubuf__ half  *nz_p2  = nz_dst_base + 2 * 16;
        __ubuf__ half  *nz_p3  = nz_dst_base + 3 * 16;

        vector_f32 v_x0, v_x1, v_x2, v_x3;
        vector_f32 v_exp0, v_exp1, v_exp2, v_exp3;
        vector_f32 v_sum0, v_sum1, v_sum2, v_sum3;
        vector_f16 v_h0, v_h1, v_h2, v_h3;
        vector_f16 v_pack0, v_pack1, v_pack2, v_pack3;
        vector_f16 v_packa;

        // Peeled iter 0 — v_sum* receive the first exp (NO vbr).
        vlds(v_x0, src_p0, 256, NORM, POST_UPDATE);
        vlds(v_x1, src_p1, 256, NORM, POST_UPDATE);
        vlds(v_x2, src_p2, 256, NORM, POST_UPDATE);
        vlds(v_x3, src_p3, 256, NORM, POST_UPDATE);

        vmuls(v_x0, v_x0, scale, preg_b32);
        vmuls(v_x1, v_x1, scale, preg_b32);
        vmuls(v_x2, v_x2, scale, preg_b32);
        vmuls(v_x3, v_x3, scale, preg_b32);

        vexpdif(v_sum0, v_x0, max_0a, preg_b32, PART_ODD);
        vexpdif(v_sum1, v_x1, max_0a, preg_b32, PART_ODD);
        vexpdif(v_sum2, v_x2, max_0a, preg_b32, PART_ODD);
        vexpdif(v_sum3, v_x3, max_0a, preg_b32, PART_ODD);

        vcvt(v_h0, v_sum0, preg_b16, ROUND_A, RS_DISABLE, PART_EVEN);
        vcvt(v_h1, v_sum1, preg_b16, ROUND_A, RS_DISABLE, PART_EVEN);
        vcvt(v_h2, v_sum2, preg_b16, ROUND_A, RS_DISABLE, PART_EVEN);
        vcvt(v_h3, v_sum3, preg_b16, ROUND_A, RS_DISABLE, PART_EVEN);

        vdintlv(v_pack0, v_packa, v_h0, v_h0);
        vdintlv(v_pack1, v_packa, v_h1, v_h1);
        vdintlv(v_pack2, v_packa, v_h2, v_h2);
        vdintlv(v_pack3, v_packa, v_h3, v_h3);

        vsstb((vector_u16 &)v_pack0, (__ubuf__ uint16_t *&)nz_p0,
              cfgVsstbIter, preg_4blk, POST_UPDATE);
        vsstb((vector_u16 &)v_pack1, (__ubuf__ uint16_t *&)nz_p1,
              cfgVsstbIter, preg_4blk, POST_UPDATE);
        vsstb((vector_u16 &)v_pack2, (__ubuf__ uint16_t *&)nz_p2,
              cfgVsstbIter, preg_4blk, POST_UPDATE);
        vsstb((vector_u16 &)v_pack3, (__ubuf__ uint16_t *&)nz_p3,
              cfgVsstbIter, preg_4blk, POST_UPDATE);

        // Loop iters 1..31.
        for (uint16_t r4 = 0; r4 < uint16_t(ubN / 4 - 1); ++r4) {
            vlds(v_x0, src_p0, 256, NORM, POST_UPDATE);
            vlds(v_x1, src_p1, 256, NORM, POST_UPDATE);
            vlds(v_x2, src_p2, 256, NORM, POST_UPDATE);
            vlds(v_x3, src_p3, 256, NORM, POST_UPDATE);

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
                  cfgVsstbIter, preg_4blk, POST_UPDATE);
            vsstb((vector_u16 &)v_pack1, (__ubuf__ uint16_t *&)nz_p1,
                  cfgVsstbIter, preg_4blk, POST_UPDATE);
            vsstb((vector_u16 &)v_pack2, (__ubuf__ uint16_t *&)nz_p2,
                  cfgVsstbIter, preg_4blk, POST_UPDATE);
            vsstb((vector_u16 &)v_pack3, (__ubuf__ uint16_t *&)nz_p3,
                  cfgVsstbIter, preg_4blk, POST_UPDATE);
        }

        // Tree reduce v_sum[0..3] → v_sum0.
        vadd(v_sum0, v_sum0, v_sum1, preg_b32, MODE_ZEROING);
        vadd(v_sum2, v_sum2, v_sum3, preg_b32, MODE_ZEROING);
        vadd(v_sum0, v_sum0, v_sum2, preg_b32, MODE_ZEROING);

        // Rescale running global_sum: new_sum = exp_max * prev_sum + local_sum.
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
