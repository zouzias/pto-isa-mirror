#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# THISTOGRAM (see tests/npu/a5/.../thistogram/gen_data.py) is ascending cumulative:
#   C[b] = number of keys with MSB byte in [0..b].
# draft.cpp FindWinnerBucketDescending / SumBinsStrictlyAbove use diffs / C[255]-C[winner].

import os
import sys

import numpy as np


def find_winner_from_asc_cumulative(C: np.ndarray, need: int) -> int:
    """Same as draft.cpp: raw[b] = C[b]-C[b-1], scan b=255..0, acc += raw[b]."""
    if need == 0:
        return 0
    acc = 0
    for b in range(255, -1, -1):
        raw_b = int(C[b]) - (int(C[b - 1]) if b > 0 else 0)
        acc += raw_b
        if acc >= need:
            return b
    return 0


def sum_strictly_above_from_asc_cumulative(C: np.ndarray, winner: int) -> int:
    return int(C[255]) - int(C[winner])


def main():
    topk = 512
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    keys_path = os.path.join(root, "input", "keys.bin")
    if not os.path.isfile(keys_path):
        print(f"Missing {keys_path}; run: python3 scripts/gen_data.py", file=sys.stderr)
        sys.exit(1)

    keys = np.fromfile(keys_path, dtype=np.dtype("<u2"))
    n = keys.size
    msb = (keys.astype(np.uint32) >> 8) & 0xFF
    counts = np.bincount(msb, minlength=256).astype(np.uint64)
    C = np.cumsum(counts).astype(np.uint64)

    assert C[255] == n, "C[255] must equal N"

    msb_winner = find_winner_from_asc_cumulative(C, topk)
    sum_above = sum_strictly_above_from_asc_cumulative(C, msb_winner)
    remain_k = topk - sum_above

    print("=== CPU reference (ascending cumulative C[b], same as THISTOGRAM golden) ===")
    print(f"N = {n}, TopK = {topk}")
    print(f"C[255] = {C[255]} (expect N)")
    print(f"msb_winner = {msb_winner}")
    print(f"sum_bins_strictly_above = C[255] - C[winner] = {sum_above}")
    print(f"remain_k = TopK - sum_above = {remain_k}")
    print()
    print("Per-bin counts [0..15] and [240..255] (for intuition; device holds C not these):")
    print("  [0..15] :", " ".join(f"{counts[i]:4d}" for i in range(16)))
    print("  [240..255]:", " ".join(f"{counts[i]:4d}" for i in range(240, 256)))
    print()
    print("C[b] samples [0..15] and [240..255]:")
    print("  [0..15] :", " ".join(f"{C[i]:4d}" for i in range(16)))
    print("  [240..255]:", " ".join(f"{C[i]:4d}" for i in range(240, 256)))


if __name__ == "__main__":
    main()
