#!/usr/bin/python3
# coding=utf-8
# --------------------------------------------------------------------------------
# topk (auto-mode A3 prototype, full TopK) - gen_data.py
#
# Generates random unsorted inputs + identity index array, computes the true
# top-K values and indices via NumPy. Replaces the previous values-only
# pre-sort recipe.
#
# Output files (raw little-endian, contiguous, no header):
#   ./input/input_src.bin     1 * kCols  float32   (raw random unsorted)
#   ./input/input_idx.bin     1 * kCols  uint32_t  ([0..kCols-1])
#   ./output/golden_val.bin   1 * kTopK  float32   (top-K values, descending)
#   ./output/golden_idx.bin   1 * kTopK  uint32_t  (matching original-position indices)
# --------------------------------------------------------------------------------

import os
import numpy as np

np.random.seed(19)

kCols = 1280
kTopK = 512


def gen_golden_data():
    src = np.random.uniform(-1000.0, 1000.0, size=(kCols,)).astype(np.float32)
    idx = np.arange(kCols, dtype=np.uint32)

    # Descending sort by value, stable on original order for ties.
    order = np.argsort(-src, kind='stable')
    topk_idx = order[:kTopK].astype(np.uint32)
    topk_val = src[topk_idx].astype(np.float32)

    os.makedirs("input", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    src.tofile("./input/input_src.bin")
    idx.tofile("./input/input_idx.bin")
    topk_val.tofile("./output/golden_val.bin")
    topk_idx.tofile("./output/golden_idx.bin")


if __name__ == "__main__":
    gen_golden_data()
