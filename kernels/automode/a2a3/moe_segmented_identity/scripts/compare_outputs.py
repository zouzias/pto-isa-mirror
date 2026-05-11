#!/usr/bin/python3
# coding=utf-8
# --------------------------------------------------------------------------------
# moe_segmented_identity - compare_outputs.py
#
# Standalone diff utility. main.cpp already runs ResultCmp internally; this
# script is for richer post-mortem when the device output disagrees with
# the golden — and especially to map the first mismatching flat index back
# to its segment / tile context so we can tell whether the failure is in:
#
#   - a real token row,
#   - a padded row,
#   - the first inner-loop tile of an expert,
#   - a second-or-later inner-loop tile of an expert,
#   - outside any expert's padded segment (kernel walked off the end).
#
# Run after `bash run.sh -r npu -v Ascend910B*`:
#   python ./scripts/compare_outputs.py
#
# Reads:
#   ./output/golden_packed_output.bin
#   ./output/output_packed_output.bin
#   ./output/t_padded.txt
#   ./input/input_expert_count.bin    (PADDED counts)
#   ./input/input_expert_start.bin    (PADDED starts)
#   ./output/expert_count_real.bin    (REAL counts, for padded-vs-real labels)
# --------------------------------------------------------------------------------

import os
import sys
import numpy as np


def main():
    kH      = 64
    kE      = 4
    kTileM  = 128

    # ---- Load shape / metadata ----------------------------------------------
    if not os.path.exists("./output/t_padded.txt"):
        print("[compare] missing ./output/t_padded.txt (run scripts/gen_data.py first)")
        sys.exit(2)
    with open("./output/t_padded.txt") as f:
        T_padded = int(f.read().strip())

    gold_path = "./output/golden_packed_output.bin"
    got_path  = "./output/output_packed_output.bin"
    for p in (gold_path, got_path):
        if not os.path.exists(p):
            print(f"[compare] missing {p}")
            sys.exit(2)

    gold = np.fromfile(gold_path, dtype=np.float32).reshape(T_padded, kH)
    got  = np.fromfile(got_path,  dtype=np.float32).reshape(T_padded, kH)

    expert_count_padded = np.fromfile("./input/input_expert_count.bin", dtype=np.int32)
    expert_start_padded = np.fromfile("./input/input_expert_start.bin", dtype=np.int32)

    real_path = "./output/expert_count_real.bin"
    if os.path.exists(real_path):
        expert_count_real = np.fromfile(real_path, dtype=np.int32)
        have_real = True
    else:
        # Older gen_data.py revision; padded-vs-real labels degrade to Unknown.
        expert_count_real = np.zeros(kE, dtype=np.int32)
        have_real = False
        print("[compare] note: ./output/expert_count_real.bin not present; "
              "padded-vs-real labels will say 'Unknown'.")

    # ---- Aggregate stats ---------------------------------------------------
    diff = got - gold
    abs_diff = np.abs(diff)
    max_abs = float(abs_diff.max())

    row_eq = np.all(diff == 0, axis=1)
    n_match = int(row_eq.sum())
    n_mismatch = T_padded - n_match

    print(f"[compare] shape:        ({T_padded}, {kH}) float32")
    print(f"[compare] max abs diff: {max_abs:.6g}")
    print(f"[compare] matching rows:    {n_match}/{T_padded}")
    print(f"[compare] mismatching rows: {n_mismatch}/{T_padded}")

    if n_mismatch == 0 and max_abs == 0.0:
        print("[compare] PASS (bit-exact)")
        sys.exit(0)

    # Tight float32 tolerance (integer-valued + 1.0 should be exact).
    if np.allclose(gold, got, atol=1e-6, rtol=0.0):
        print("[compare] PASS (within atol=1e-6)")
        sys.exit(0)

    # =========================================================================
    # FAIL — produce a detailed first-mismatch report (per user's request).
    # =========================================================================

    flat_gold = gold.reshape(-1)
    flat_got  = got.reshape(-1)
    flat_neq  = flat_gold != flat_got
    if not flat_neq.any():
        # All abs diffs nonzero but the boolean compare says equal? Numerically
        # impossible with float32 if max_abs > 1e-6 — guard anyway.
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
    print(f"  row               = {first_row}  (flat // {kH})")
    print(f"  col               = {first_col}  (flat %  {kH})")
    if e0 is None:
        print(f"  segment           = OUT-OF-RANGE (row {first_row} >= T_PADDED {T_padded}"
              f" or beyond the last expert end {int(expert_start_padded[-1]) + int(expert_count_padded[-1])})")
    else:
        offset_in_seg = first_row - s0
        tile_m0       = (offset_in_seg // kTileM) * kTileM
        tile_idx      = offset_in_seg // kTileM
        n_inner_iters = c0 // kTileM
        real_c        = int(expert_count_real[e0]) if have_real else None
        is_padded     = None if not have_real else (offset_in_seg >= real_c)
        last_real_row_of_expert = (s0 + real_c - 1) if have_real else None

        print(f"  expert            = {e0}")
        print(f"  expert_start[e]   = {s0}  (PADDED)")
        print(f"  expert_count[e]   = {c0}  (PADDED; multiple of {kTileM})")
        if have_real:
            print(f"  expert_count_real = {real_c}  (REAL)")
        print(f"  offset_in_segment = {offset_in_seg}  (= row - expert_start[e])")
        print(f"  tile_m0           = {tile_m0}  (= floor(offset / {kTileM}) * {kTileM})")
        print(f"  tile_idx          = {tile_idx}  (0-based inner-loop iteration index)")
        print(f"  inner iters total = {n_inner_iters}  (= padded_count / {kTileM})")
        if have_real:
            print(f"  is_padded_row     = {is_padded}  "
                  f"({'this row is past the last real token of its expert' if is_padded else 'this row holds a real token'})")
            print(f"  last_real_row_e   = {last_real_row_of_expert}")
        else:
            print(f"  is_padded_row     = Unknown (expert_count_real.bin missing)")

        # Position cues
        cues = []
        if tile_idx == 0:
            cues.append("FIRST tile of expert (m0 = 0)")
        else:
            cues.append(f"NON-FIRST tile of expert (tile #{tile_idx})")
        if tile_idx == n_inner_iters - 1 and n_inner_iters > 1:
            cues.append("LAST tile of expert")
        if offset_in_seg == 0:
            cues.append("FIRST row of expert segment")
        if have_real and is_padded and offset_in_seg == real_c:
            cues.append("FIRST padded row (real/pad boundary)")
        print(f"  cues              = {' | '.join(cues)}")

    # Show first up to 5 mismatching rows with values
    mismatch_idx_rows = np.where(~row_eq)[0]
    show = mismatch_idx_rows[: min(5, mismatch_idx_rows.size)]
    print()
    print(f"[compare] ---- first {show.size} mismatching rows ----")
    for row in show:
        e, s, c = segment_of(int(row))
        seg_lbl = f"e={e} s={s} c={c} off={int(row) - s} tile#{(int(row) - s)//kTileM}" if e is not None else "out-of-range"
        if have_real and e is not None:
            offs = int(row) - s
            seg_lbl += f" {'PAD' if offs >= int(expert_count_real[e]) else 'REAL'}"
        print(f"  row {row} ({seg_lbl}):")
        print(f"    golden[:8]   = {gold[row, :8].tolist()}")
        print(f"    got[:8]      = {got[row, :8].tolist()}")
        print(f"    abs_diff[:8] = {abs_diff[row, :8].tolist()}")

    # Aggregate failure pattern by expert (helps see "all of expert 0 is wrong"
    # vs "only the second tile of expert 0 is wrong").
    print()
    print(f"[compare] ---- mismatch counts per expert segment ----")
    for e in range(kE):
        s = int(expert_start_padded[e])
        c = int(expert_count_padded[e])
        n_bad = int((~row_eq[s : s + c]).sum())
        per_tile = []
        n_tiles = c // kTileM
        for t in range(n_tiles):
            ts = s + t * kTileM
            te = ts + kTileM
            per_tile.append(int((~row_eq[ts:te]).sum()))
        print(f"  expert {e}: {n_bad}/{c} bad rows, per-tile = {per_tile}")
    # Trailing rows past the last expert (if any) — shouldn't exist by
    # construction but flag if so.
    last_end = int(expert_start_padded[-1]) + int(expert_count_padded[-1])
    if last_end < T_padded:
        trailing_bad = int((~row_eq[last_end:T_padded]).sum())
        print(f"  trailing rows [{last_end}..{T_padded}): {trailing_bad} bad")

    sys.exit(1)


if __name__ == "__main__":
    main()
