#!/usr/bin/python3
# coding=utf-8
# --------------------------------------------------------------------------------
# moe_segmented_gemm_relu - compare_outputs.py
#
# Same first-mismatch breakdown as the §A16 GEMM compare script, with one
# addition: each mismatching row also reports whether the corresponding
# PRE-ReLU GEMM output (from ./output/golden_gemm_output.bin) was negative —
# i.e., whether the golden value was clipped to zero by ReLU. This helps
# distinguish "ReLU never fired" failures from "GEMM result already
# differed before ReLU" failures.
#
# Run after `bash run.sh -r npu -v Ascend910B*`:
#   python ./scripts/compare_outputs.py
#
# Reads:
#   ./output/golden_packed_output.bin    (T_PADDED * kO float32; POST-ReLU)
#   ./output/output_packed_output.bin    (T_PADDED * kO float32)
#   ./output/golden_gemm_output.bin      (T_PADDED * kO float32; PRE-ReLU)
#   ./output/t_padded.txt
#   ./input/input_expert_count.bin       (PADDED counts)
#   ./input/input_expert_start.bin       (PADDED starts)
#   ./output/expert_count_real.bin       (REAL counts, for padded-vs-real labels)
# --------------------------------------------------------------------------------

import os
import sys
import numpy as np


def main():
    kO     = 64
    kE     = 4
    kTileM = 128

    if not os.path.exists("./output/t_padded.txt"):
        print("[compare] missing ./output/t_padded.txt (run scripts/gen_data.py first)")
        sys.exit(2)
    with open("./output/t_padded.txt") as f:
        T_padded = int(f.read().strip())

    gold_path = "./output/golden_packed_output.bin"
    got_path  = "./output/output_packed_output.bin"
    gemm_path = "./output/golden_gemm_output.bin"
    for p in (gold_path, got_path):
        if not os.path.exists(p):
            print(f"[compare] missing {p}")
            sys.exit(2)

    gold = np.fromfile(gold_path, dtype=np.float32).reshape(T_padded, kO)
    got  = np.fromfile(got_path,  dtype=np.float32).reshape(T_padded, kO)

    have_gemm = os.path.exists(gemm_path)
    if have_gemm:
        gemm = np.fromfile(gemm_path, dtype=np.float32).reshape(T_padded, kO)
    else:
        gemm = None
        print("[compare] note: ./output/golden_gemm_output.bin not present; "
              "'clipped by ReLU' labels will say 'Unknown'.")

    expert_count_padded = np.fromfile("./input/input_expert_count.bin", dtype=np.int32)
    expert_start_padded = np.fromfile("./input/input_expert_start.bin", dtype=np.int32)

    real_path = "./output/expert_count_real.bin"
    if os.path.exists(real_path):
        expert_count_real = np.fromfile(real_path, dtype=np.int32)
        have_real = True
    else:
        expert_count_real = np.zeros(kE, dtype=np.int32)
        have_real = False
        print("[compare] note: ./output/expert_count_real.bin not present; "
              "padded-vs-real labels will say 'Unknown'.")

    diff = got - gold
    abs_diff = np.abs(diff)
    max_abs = float(abs_diff.max())

    row_eq = np.all(diff == 0, axis=1)
    n_match = int(row_eq.sum())
    n_mismatch = T_padded - n_match

    print(f"[compare] shape:        ({T_padded}, {kO}) float32 (POST-ReLU)")
    print(f"[compare] max abs diff: {max_abs:.6g}")
    print(f"[compare] matching rows:    {n_match}/{T_padded}")
    print(f"[compare] mismatching rows: {n_mismatch}/{T_padded}")

    if have_gemm:
        n_neg_in_gemm = int((gemm < 0).sum())
        print(f"[compare] PRE-ReLU negatives (clipped to 0 in golden): "
              f"{n_neg_in_gemm}/{int(gemm.size)} "
              f"({100.0 * n_neg_in_gemm / gemm.size:.1f}%)")

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
    first_row  = first_flat // kO
    first_col  = first_flat %  kO

    def segment_of(row):
        for e in range(kE):
            s = int(expert_start_padded[e])
            c = int(expert_count_padded[e])
            if s <= row < s + c:
                return e, s, c
        return None, None, None

    e0, s0, c0 = segment_of(first_row)
    pre_relu_val   = float(gemm[first_row, first_col]) if have_gemm else None
    clipped        = (pre_relu_val < 0.0) if have_gemm else None

    print()
    print(f"[compare] FAIL  total mismatching rows: {n_mismatch}/{T_padded}")
    print(f"[compare] ---- first mismatch ----")
    print(f"  flat_index        = {first_flat}  (0x{first_flat:x})")
    print(f"  row               = {first_row}  (flat // kO={kO})")
    print(f"  col               = {first_col}  (flat %  kO={kO})")
    print(f"  golden (post-ReLU)= {float(gold[first_row, first_col])}")
    print(f"  got    (device)   = {float(got [first_row, first_col])}")
    print(f"  abs_diff          = {float(abs_diff[first_row, first_col])}")
    if have_gemm:
        print(f"  pre-ReLU value    = {pre_relu_val}  "
              f"({'CLIPPED (negative -> 0 by ReLU)' if clipped else 'PASS-THROUGH (>=0)'})")
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
        if have_gemm and clipped:
            cues.append("golden was CLIPPED by ReLU here")
        print(f"  cues              = {' | '.join(cues)}")

    # Aggregate stat: how many mismatches fall on positions that the golden
    # clipped to zero vs positions that passed through. A "ReLU didn't fire"
    # failure mode will show n_bad_on_clip ~= n_total_clipped and
    # got > 0 at those positions.
    if have_gemm:
        clipped_mask = gemm < 0.0
        n_bad_on_clip   = int(((~row_eq).reshape(-1, 1) & clipped_mask & (got != gold)).sum())
        n_bad_on_passthr = int(((~row_eq).reshape(-1, 1) & (~clipped_mask) & (got != gold)).sum())
        # Reduce per-row OR'd mismatch over both subsets:
        clip_diff_any = (clipped_mask & (got != gold))
        passthr_diff_any = ((~clipped_mask) & (got != gold))
        print()
        print(f"[compare] ---- failure-mode breakdown ----")
        print(f"  mismatching ELEMENTS where golden was CLIPPED by ReLU       : {int(clip_diff_any.sum())}")
        print(f"  mismatching ELEMENTS where golden was PASS-THROUGH (>=0)    : {int(passthr_diff_any.sum())}")
        print(f"  interpretation:")
        print(f"    - many mismatches on CLIPPED  positions: device skipped ReLU")
        print(f"    - many mismatches on PASS-THR positions: GEMM itself is wrong")

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
        if have_gemm:
            print(f"    pre-ReLU[:8] = {gemm[row, :8].tolist()}")
        print(f"    abs_diff[:8] = {abs_diff[row, :8].tolist()}")

    sys.exit(1)


if __name__ == "__main__":
    main()
