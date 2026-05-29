#!/usr/bin/python3
# coding=utf-8
# hc_pre_reduce - gen_data.py
#
# Generates input and golden data for hc_pre_reduce:
#   y[b, s, d] = sum_{h=0..hc-1} pre[b, s, h] * x[b, s, h, d]
#
# Reference: deepseek/model.py:681 (Block.hc_pre final line).
#
# Output files (raw little-endian, contiguous, no header):
#   ./input/input_pre.bin   B*S*HC_MULT          float32
#   ./input/input_x.bin     B*S*HC_MULT*DIM      float32
#   ./output/golden_y.bin   B*S*DIM              float32

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from case_utils import load_hc_case  # noqa: E402

np.random.seed(42)

_case = load_hc_case()
kB   = _case["b"]
kS   = _case["s"]
kDim = _case["dim"]
kHc  = _case["hc_mult"]


def gen_golden_data():
    # `pre` in the real pipeline is sigmoid output in (eps, 1+eps); mimic
    # that range so golden values stay numerically reasonable.
    pre = np.random.uniform(1e-6, 1.0, size=(kB, kS, kHc)).astype(np.float32)
    x   = np.random.uniform(-1.0, 1.0, size=(kB, kS, kHc, kDim)).astype(np.float32)

    # y[b, s, d] = sum_h pre[b, s, h] * x[b, s, h, d]
    y = np.einsum("bsh,bshd->bsd", pre, x).astype(np.float32)

    os.makedirs("input",  exist_ok=True)
    os.makedirs("output", exist_ok=True)

    pre.tofile("./input/input_pre.bin")
    x.tofile("./input/input_x.bin")
    y.tofile("./output/golden_y.bin")

    print(f"[gen_data] B={kB} S={kS} HC_MULT={kHc} DIM={kDim}")
    print(f"[gen_data] pre range: [{pre.min():.4f}, {pre.max():.4f}]")
    print(f"[gen_data] x   range: [{x.min():.4f}, {x.max():.4f}]")
    print(f"[gen_data] y   range: [{y.min():.4f}, {y.max():.4f}]")


if __name__ == "__main__":
    gen_golden_data()
