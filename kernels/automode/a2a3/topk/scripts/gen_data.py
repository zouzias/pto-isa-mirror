#!/usr/bin/python3
# coding=utf-8
# --------------------------------------------------------------------------------
# topk (auto-mode A3 prototype, 2D full TopK) - gen_data.py
#
# Generates a (kRows, kCols) float32 input and computes the per-row top-K
# values and matching original-position indices via NumPy.
#
# Output files (raw little-endian, contiguous, no header):
#   ./input/input_src.bin     kRows * kCols  float32   (raw random unsorted, row-major)
#   ./input/input_idx.bin     1     * kCols  uint32_t  ([0..kCols-1]; shared across rows)
#   ./output/golden_val.bin   kRows * kTopK  float32   (per-row top-K values, descending)
#   ./output/golden_idx.bin   kRows * kTopK  uint32_t  (per-row matching original-position indices)
# --------------------------------------------------------------------------------

import os
import json
from pathlib import Path
import numpy as np

np.random.seed(19)


def load_generated_case():
    case_path = Path(__file__).resolve().parent.parent / "build" / "generated_cases.json"
    if not case_path.exists():
        return {"rows": 4, "cols": 1280, "topk": 512}
    with case_path.open("r", encoding="utf-8") as f:
        cases = json.load(f)
    return cases[0]


def gen_golden_data(kRows, kCols, kTopK):
    src = np.random.uniform(-1000.0, 1000.0, size=(kRows, kCols)).astype(np.float32)
    # idx is the same identity row for every input row; kernel TLOADs it once
    # per row from a shared (kCols,) GM region. Keep file size at kCols.
    idx = np.arange(kCols, dtype=np.uint32)

    # Per-row descending sort by value, stable on original order for ties.
    order = np.argsort(-src, axis=1, kind='stable')
    topk_idx = order[:, :kTopK].astype(np.uint32)
    topk_val = np.take_along_axis(src, topk_idx.astype(np.int64), axis=1).astype(np.float32)

    os.makedirs("input", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    src.tofile("./input/input_src.bin")
    idx.tofile("./input/input_idx.bin")
    topk_val.tofile("./output/golden_val.bin")
    topk_idx.tofile("./output/golden_idx.bin")


if __name__ == "__main__":
    case = load_generated_case()
    gen_golden_data(case["rows"], case["cols"], case["topk"])
