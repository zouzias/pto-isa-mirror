#!/usr/bin/python3
# coding=utf-8
# --------------------------------------------------------------------------------
# moe_segmented_ffn_top1 - compare_outputs.py
#
# Same first-mismatch breakdown as the §A16 / §A17 compare scripts. The
# kernel composes GEMM1 + ReLU + GEMM2, so when something is wrong the
# diagnosis question becomes: did GEMM1 misbehave (scratch wrong), did the
# fused ReLU + downcast TSTORE misbehave (scratch wrong on a specific
# clip/pass-through partition), or did GEMM2 misbehave (scratch fine but
# final output wrong)? We can't see device scratch (kernel-managed) so the
# script reports only what is observable from packed_output, plus the golden
# scratch/pre-ReLU GEMM1 values to help reason about which stage drifted.
#
# Run after `bash run.sh -r npu -v Ascend910B*`:
#   python ./scripts/compare_outputs.py
#
# Reads:
#   ./output/golden_packed_output.bin     (T_PADDED * kH float32; final FFN)
#   ./output/output_packed_output.bin     (T_PADDED * kH float32; device)
#   ./output/golden_scratch.bin           (T_PADDED * kF float16; debug)
#   ./output/golden_gemm1_output.bin      (T_PADDED * kF float32; PRE-ReLU; debug)
#   ./output/t_padded.txt
#   ./input/input_expert_count.bin        (PADDED counts)
#   ./input/input_expert_start.bin        (PADDED starts)
#   ./output/expert_count_real.bin        (REAL counts; for padded-vs-real labels)
# --------------------------------------------------------------------------------

import os
import sys
import numpy as np


def main():
    kH     = 64
    kF     = 64
    kE     = 4
    kTileM = 128

    if not os.path.exists("./output/t_padded.txt"):
        print("[compare] missing ./output/t_padded.txt (run scripts/gen_data.py first)")
        sys.exit(2)
    with open("./output/t_padded.txt") as f:
        T_padded = int(f.read().strip())

    gold_path    = "./output/golden_packed_output.bin"
    got_path     = "./output/output_packed_output.bin"
    scratch_path = "./output/golden_scratch.bin"
    gemm1_path   = "./output/golden_gemm1_output.bin"
    for p in (gold_path, got_path):
        if not os.path.exists(p):
            print(f"[compare] missing {p}")
            sys.exit(2)

    gold = np.fromfile(gold_path, dtype=np.float32).reshape(T_padded, kH)
    got  = np.fromfile(got_path,  dtype=np.float32).reshape(T_padded, kH)

    have_scratch = os.path.exists(scratch_path)
    if have_scratch:
        scratch_gold = np.fromfile(scratch_path, dtype=np.float16).reshape(T_padded, kF)
    else:
        scratch_gold = None
        print("[compare] note: golden_scratch.bin not present; "
              "scratch-side hints will say 'Unknown'.")

    have_gemm1 = os.path.exists(gemm1_path)
    if have_gemm1:
        gemm1_gold = np.fromfile(gemm1_path, dtype=np.float32).reshape(T_padded, kF)
    else:
        gemm1_gold = None
        print("[compare] note: golden_gemm1_output.bin not present; "
              "pre-ReLU GEMM1 hints will say 'Unknown'.")

    expert_count_padded = np.fromfile("./input/input_expert_count.bin", dtype=np.int32)
    expert_start_padded = np.fromfile("./input/input_expert_start.bin", dtype=np.int32)

    real_path = "./output/expert_count_real.bin"
    if os.path.exists(real_path):
        expert_count_real = np.fromfile(real_path, dtype=np.int32)
        have_real = True
    else:
        expert_count_real = np.zeros(kE, dtype=np.int32)
        have_real = False
        print("[compare] note: expert_count_real.bin not present; "
              "padded-vs-real labels will say 'Unknown'.")

    diff = got - gold
    abs_diff = np.abs(diff)
    max_abs = float(abs_diff.max())

    row_eq = np.all(diff == 0, axis=1)
    n_match = int(row_eq.sum())
    n_mismatch = T_padded - n_match

    print(f"[compare] shape:        ({T_padded}, {kH}) float32 (final FFN output)")
    print(f"[compare] max abs diff: {max_abs:.6g}")
    print(f"[compare] matching rows:    {n_match}/{T_padded}")
    print(f"[compare] mismatching rows: {n_mismatch}/{T_padded}")

    if have_gemm1:
        n_neg_g1   = int((gemm1_gold < 0.0).sum())
        n_pos_g1   = int((gemm1_gold > 0.0).sum())
        print(f"[compare] PRE-ReLU GEMM1 negatives (clipped in scratch): "
              f"{n_neg_g1}/{int(gemm1_gold.size)} "
              f"({100.0 * n_neg_g1 / gemm1_gold.size:.1f}%)")
        print(f"[compare] PRE-ReLU GEMM1 positives (passed through):    "
              f"{n_pos_g1}/{int(gemm1_gold.size)} "
              f"({100.0 * n_pos_g1 / gemm1_gold.size:.1f}%)")

    if n_mismatch == 0 and max_abs == 0.0:
        print("[compare] PASS (bit-exact)")
        sys.exit(0)

    if np.allclose(gold, got, atol=1e-3, rtol=0.0):
        print("[compare] PASS (within atol=1e-3)")
        sys.exit(0)

    flat_gold = gold.reshape(-1)
    flat_got  = got.reshape(-1)
    flat_neq  = flat_gold != flat_got
    if not flat_neq.any():
        print("[compare] FAIL — no flat-index mismatch found despite tolerance check failing.")
        sys.exit(1)

    first_flat = int(np.argmax(flat_neq))
    first_row  = first_flat // kH
    first_col  = first_flat %  kH

    def segment_of(row):
        for e in range(kE):
            s = int(expert_start_padded[e])
            c = int(expert_count_padded[e])
            if s <= row < s + c:
                return e, s, c
        return None, None, None

    e0, s0, c0 = segment_of(first_row)

    print()
    print(f"[compare] FAIL  total mismatching rows: {n_mismatch}/{T_padded}")
    print(f"[compare] ---- first mismatch ----")
    print(f"  flat_index        = {first_flat}  (0x{first_flat:x})")
    print(f"  row               = {first_row}  (flat // kH={kH})")
    print(f"  col               = {first_col}  (flat %  kH={kH})")
    print(f"  golden            = {float(gold[first_row, first_col])}")
    print(f"  got    (device)   = {float(got [first_row, first_col])}")
    print(f"  abs_diff          = {float(abs_diff[first_row, first_col])}")
    if e0 is None:
        print(f"  segment           = OUT-OF-RANGE (row >= T_PADDED)")
    else:
        offset_in_seg = first_row - s0
        tile_m0       = (offset_in_seg // kTileM) * kTileM
        tile_idx      = offset_in_seg // kTileM
        n_inner_iters = c0 // kTileM
        real_c        = int(expert_count_real[e0]) if have_real else None
        is_padded     = None if not have_real else (offset_in_seg >= real_c)

        print(f"  expert            = {e0}")
        print(f"  expert_start[e]   = {s0}  (PADDED)")
        print(f"  expert_count[e]   = {c0}  (PADDED, multiple of {kTileM})")
        if have_real:
            print(f"  expert_count_real = {real_c}  (REAL)")
        print(f"  offset_in_segment = {offset_in_seg}")
        print(f"  tile_m0           = {tile_m0}")
        print(f"  tile_idx          = {tile_idx}  (0-based inner-loop iter)")
        print(f"  inner iters total = {n_inner_iters}  (= padded_count / {kTileM})")
        if have_real:
            print(f"  is_padded_row     = {is_padded}")

        cues = []
        if tile_idx == 0:
            cues.append("FIRST tile of expert (m0 = 0)")
        else:
            cues.append(f"NON-FIRST tile of expert (tile #{tile_idx})")
        if tile_idx == n_inner_iters - 1 and n_inner_iters > 1:
            cues.append("LAST tile of expert")
        if have_real and is_padded:
            cues.append("padded row (expected output: all zeros)")
        print(f"  cues              = {' | '.join(cues)}")

    # Stage-attribution heuristic. Without device-side scratch we can only
    # offer hints, not a definitive partition.
    if have_gemm1 and have_scratch and e0 is not None:
        # How many of this row's GEMM1 outputs were clipped (and therefore
        # contribute 0 to GEMM2 regardless of w2)?
        g1_row = gemm1_gold[first_row, :]
        sc_row = scratch_gold[first_row, :].astype(np.float32)
        n_clip_in_row = int((g1_row < 0.0).sum())
        n_pass_in_row = int((g1_row > 0.0).sum())
        print()
        print(f"[compare] ---- stage-attribution hints (golden, per row {first_row}) ----")
        print(f"  GEMM1 pre-ReLU min / max         = {float(g1_row.min())} / {float(g1_row.max())}")
        print(f"  GEMM1 pre-ReLU negatives in row  = {n_clip_in_row}/{kF}")
        print(f"  GEMM1 pre-ReLU positives in row  = {n_pass_in_row}/{kF}")
        print(f"  scratch (post-ReLU FP16) min/max = {float(sc_row.min())} / {float(sc_row.max())}")
        print(f"  scratch nonzeros                  = {int((sc_row != 0.0).sum())}/{kF}")
        if n_clip_in_row == kF:
            print("  -> all GEMM1 outputs negative; scratch row is entirely zero; "
                  "GEMM2 output for this row should be exactly zero. If device "
                  "shows nonzero here, ReLU did NOT clip on scratch write.")
        elif n_pass_in_row == kF:
            print("  -> no clipping for this row; ReLU is essentially identity here. "
                  "Mismatches in this row likely indicate a GEMM1 or GEMM2 numeric "
                  "bug, NOT a ReLU bug.")
        else:
            print("  -> mixed clip/pass-through row. Drift here is consistent with "
                  "either ReLU mis-applied or FP32->FP16 downcast not happening in "
                  "the fused TSTORE. Compare device output magnitude vs golden to "
                  "tell which.")

    mismatch_idx_rows = np.where(~row_eq)[0]
    show = mismatch_idx_rows[: min(5, mismatch_idx_rows.size)]
    print()
    print(f"[compare] ---- first {show.size} mismatching rows ----")
    for row in show:
        e, s, c = segment_of(int(row))
        seg_lbl = (f"e={e} s={s} c={c} off={int(row) - s} tile#{(int(row) - s)//kTileM}"
                   if e is not None else "out-of-range")
        if have_real and e is not None:
            offs = int(row) - s
            seg_lbl += f" {'PAD' if offs >= int(expert_count_real[e]) else 'REAL'}"
        print(f"  row {row} ({seg_lbl}):")
        print(f"    golden[:8]   = {gold[row, :8].tolist()}")
        print(f"    got[:8]      = {got[row, :8].tolist()}")
        print(f"    abs_diff[:8] = {abs_diff[row, :8].tolist()}")
        if have_scratch:
            print(f"    scratch_gold[:8]   = {scratch_gold[row, :8].tolist()}")
        if have_gemm1:
            print(f"    gemm1_pre_relu[:8] = {gemm1_gold[row, :8].tolist()}")

    sys.exit(1)


if __name__ == "__main__":
    main()
