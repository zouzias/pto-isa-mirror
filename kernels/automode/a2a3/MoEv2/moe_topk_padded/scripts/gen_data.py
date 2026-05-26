#!/usr/bin/python3
# coding=utf-8
# --------------------------------------------------------------------------------
# moe_topk_padded - gen_data.py
#
# Generates a (kT, kE) float32 logits input and computes the per-row top-K
# expert values and indices via NumPy. kTopK is generic in {1, 2, 4, 8, 16}.
# Must match the constants in moe_topk_padded_kernel.cpp and main.cpp.
#
# Output files (raw little-endian, contiguous, no header):
#   ./input/input_logits.bin   kT * kE    float32  (router logits per token)
#   ./input/input_idx.bin      1  * kE    uint32   ([0..kE-1] identity row)
#   ./output/golden_val.bin    kT * kTopK float32  (descending top-K logit values)
#   ./output/golden_idx.bin    kT * kTopK uint32   (matching expert ids)
# --------------------------------------------------------------------------------

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from case_utils import load_moe_case

np.random.seed(19)

_case = load_moe_case()
kT = _case["t"]
kE = _case["e"]
kTopK = _case["topk"]


def gen_golden_data():
    logits = np.random.uniform(-100.0, 100.0, size=(kT, kE)).astype(np.float32)
    idx    = np.arange(kE, dtype=np.uint32)

    # Per-row descending sort by value, stable on original order for ties.
    order      = np.argsort(-logits, axis=1, kind="stable")
    topk_idx   = order[:, :kTopK].astype(np.uint32)
    topk_val   = np.take_along_axis(logits, topk_idx.astype(np.int64), axis=1).astype(np.float32)

    os.makedirs("input",  exist_ok=True)
    os.makedirs("output", exist_ok=True)
    logits.tofile  ("./input/input_logits.bin")
    idx.tofile     ("./input/input_idx.bin")
    topk_val.tofile("./output/golden_val.bin")
    topk_idx.tofile("./output/golden_idx.bin")

    print(f"[gen_data] kT={kT}  kE={kE}  kTopK={kTopK}")
    print(f"[gen_data] topk_idx[0] = {topk_idx[0].tolist()}")
    print(f"[gen_data] topk_val[0] = {topk_val[0].tolist()}")


if __name__ == "__main__":
    gen_golden_data()
