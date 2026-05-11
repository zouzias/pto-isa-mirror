#!/usr/bin/python3
# coding=utf-8
# --------------------------------------------------------------------------------
# moe_top1_unpermute - compare_outputs.py
#
# Standalone diff utility. main.cpp already runs ResultCmp internally; this
# script is for richer post-mortem when the device output disagrees with the
# golden — prints max abs diff, mismatching row count, and the first few
# differing rows side-by-side.
#
# Run after `bash run.sh -r npu -v Ascend910B*`:
#   python ./scripts/compare_outputs.py
#
# Reads:
#   ./output/golden_output.bin
#   ./output/output_output.bin
# --------------------------------------------------------------------------------

import numpy as np
import sys


def main():
    kT = 256
    kH = 64

    gold_path = "./output/golden_output.bin"
    got_path  = "./output/output_output.bin"

    try:
        gold = np.fromfile(gold_path, dtype=np.float32).reshape(kT, kH)
    except FileNotFoundError:
        print(f"[compare] missing {gold_path} (run scripts/gen_data.py first)")
        sys.exit(2)
    try:
        got = np.fromfile(got_path,  dtype=np.float32).reshape(kT, kH)
    except FileNotFoundError:
        print(f"[compare] missing {got_path} (run the kernel first)")
        sys.exit(2)

    diff = got - gold
    abs_diff = np.abs(diff)
    max_abs = float(abs_diff.max())

    row_eq = np.all(diff == 0, axis=1)
    n_match = int(row_eq.sum())
    n_mismatch = kT - n_match

    print(f"[compare] shape:        ({kT}, {kH}) float32")
    print(f"[compare] max abs diff: {max_abs:.6g}")
    print(f"[compare] matching rows:    {n_match}/{kT}")
    print(f"[compare] mismatching rows: {n_mismatch}/{kT}")

    if n_mismatch == 0 and max_abs == 0.0:
        print("[compare] PASS (bit-exact)")
        sys.exit(0)

    # Tight float32 tolerance (integer-valued inputs + 1.0 should be exact).
    if np.allclose(gold, got, atol=1e-6, rtol=0.0):
        print("[compare] PASS (within atol=1e-6)")
        sys.exit(0)

    # Failed — print first few differing rows.
    mismatch_idx = np.where(~row_eq)[0]
    show = mismatch_idx[: min(5, mismatch_idx.size)]
    print(f"[compare] FAIL — first {show.size} mismatching row indices: {show.tolist()}")
    for t in show:
        print(f"  row t={t}:")
        print(f"    golden[:8]   = {gold[t, :8].tolist()}")
        print(f"    got[:8]      = {got[t, :8].tolist()}")
        print(f"    abs_diff[:8] = {abs_diff[t, :8].tolist()}")
    sys.exit(1)


if __name__ == "__main__":
    main()
