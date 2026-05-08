#!/usr/bin/python3
# coding=utf-8
# --------------------------------------------------------------------------------
# topk (auto-mode A3 prototype) - gen_data.py
#
# v1 limitations:
#   - Single row only.
#   - Values-only top-K (no indices). v2 will add TSORT32 + TGATHER for indices.
#   - The kernel's merge sort assumes input is **pre-sorted in 64-element
#     blocks (descending)**. In the manual mode kernel, this pre-sort is
#     provided by an in-kernel TSORT32 step. For v1 we do that pre-sort here
#     in NumPy. v2 will move the pre-sort back into the kernel.
#
# Output files (raw little-endian float32, contiguous, no header):
#   ./input/input_src.bin    1 * COLS = 1280  floats, 64-block-descending
#   ./output/golden_val.bin  1 * TOPK = 512   floats, fully descending top-K
# --------------------------------------------------------------------------------

import os
import numpy as np
np.random.seed(19)

COLS = 1280
TOPK = 512
BLOCK = 64
DTYPE = np.float32


def gen_golden_data():
    assert COLS % BLOCK == 0, "COLS must be a multiple of BLOCK"

    # Random row.
    raw = np.random.uniform(-1000.0, 1000.0, size=(COLS,)).astype(DTYPE)

    # In-block descending pre-sort (TSORT32 surrogate for 64-element blocks).
    presorted = raw.copy().reshape(COLS // BLOCK, BLOCK)
    presorted = -np.sort(-presorted, axis=1)   # descending per block
    presorted = presorted.reshape(-1)

    # Golden top-K = fully sorted descending, then prefix.
    full_sorted = -np.sort(-raw)
    golden = full_sorted[:TOPK].astype(DTYPE)

    os.makedirs("input", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    presorted.astype(DTYPE).tofile("./input/input_src.bin")
    golden.tofile("./output/golden_val.bin")


if __name__ == "__main__":
    gen_golden_data()
