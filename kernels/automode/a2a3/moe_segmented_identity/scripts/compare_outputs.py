#!/usr/bin/python3
# coding=utf-8
# --------------------------------------------------------------------------------
# moe_segmented_identity - compare_outputs.py
#
# Standalone diff utility. main.cpp already runs ResultCmp internally; this
# script is for richer post-mortem when the device output disagrees with the
# golden — prints max abs diff, mismatching row count, and the first few
# differing rows side-by-side. Also flags whether the mismatches fall in
# padded rows (which should still be golden = 1.0 since v1 pads with zero
# and golden adds 1.0 to all rows including padded).
#
# Run after `bash run.sh -r npu -v Ascend910B*`:
#   python ./scripts/compare_outputs.py
#
# Reads:
#   ./output/golden_packed_output.bin
#   ./output/output_packed_output.bin
#   ./output/t_padded.txt
#   ./input/input_expert_count.bin   (padded counts)
#   ./input/input_expert_start.bin   (padded starts)
# --------------------------------------------------------------------------------

import os
import sys
import numpy as np


def main():
    kH = 64
    kE = 4

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

    # Failed — show segment context for the first few mismatching rows.
    mismatch_idx = np.where(~row_eq)[0]

    # Read padded segment metadata so we can label each mismatching row.
    expert_count_padded = np.fromfile("./input/input_expert_count.bin", dtype=np.int32)
    expert_start_padded = np.fromfile("./input/input_expert_start.bin", dtype=np.int32)

    def segment_of(row):
        for e in range(kE):
            s = int(expert_start_padded[e])
            c = int(expert_count_padded[e])
            if s <= row < s + c:
                return e, s, c, row - s
        return None, None, None, None

    show = mismatch_idx[: min(8, mismatch_idx.size)]
    print(f"[compare] FAIL — first {show.size} mismatching row indices: {show.tolist()}")
    for row in show:
        e, s, c, offset_in_seg = segment_of(int(row))
        seg_str = (f"expert={e} start={s} count={c} offset={offset_in_seg}"
                   if e is not None else "out-of-range")
        print(f"  row {row} ({seg_str}):")
        print(f"    golden[:8]   = {gold[row, :8].tolist()}")
        print(f"    got[:8]      = {got[row, :8].tolist()}")
        print(f"    abs_diff[:8] = {abs_diff[row, :8].tolist()}")
    sys.exit(1)


if __name__ == "__main__":
    main()
