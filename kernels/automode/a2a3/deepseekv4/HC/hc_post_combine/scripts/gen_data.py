#!/usr/bin/python3
# coding=utf-8
# hc_post_combine - gen_data.py
#
# Generates input and golden data for hc_post_combine:
#   y[b,s,h,d] = post[b,s,h] * x[b,s,d]
#              + sum_{h2} comb[b,s,h2,h] * residual[b,s,h2,d]
#
# Reference: deepseek/model.py:685-686 (Block.hc_post).
#
# Output files (raw little-endian, contiguous, no header):
#   ./input/input_x.bin          B*S*DIM              float32
#   ./input/input_residual.bin   B*S*HC_MULT*DIM      float32
#   ./input/input_post.bin       B*S*HC_MULT          float32
#   ./input/input_comb.bin       B*S*HC_MULT*HC_MULT  float32
#   ./output/golden_y.bin        B*S*HC_MULT*DIM      float32

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
    x        = np.random.uniform(-1.0, 1.0, size=(kB, kS, kDim)).astype(np.float32)
    residual = np.random.uniform(-1.0, 1.0, size=(kB, kS, kHc, kDim)).astype(np.float32)
    # `post` and `comb` in the real pipeline come from hc_sinkhorn; mimic
    # their value ranges (post in roughly (0, 2), comb in (0, 1) and doubly
    # stochastic — but exact doubly-stochasticity is not required for the
    # numerical golden, so we just sample positive and normalize lightly).
    post = np.random.uniform(0.0, 2.0, size=(kB, kS, kHc)).astype(np.float32)
    comb = np.random.uniform(0.0, 1.0, size=(kB, kS, kHc, kHc)).astype(np.float32)
    comb = (comb / (comb.sum(axis=-1, keepdims=True) + 1e-6)).astype(np.float32)

    # y[b,s,h,d] = post[b,s,h] * x[b,s,d]
    #            + sum_{h2} comb[b,s,h2,h] * residual[b,s,h2,d]
    # model.py:686 uses comb.unsqueeze(-1) on dim 2 (so h2 axis is dim 2 of
    # comb) and residual.unsqueeze(-2) (broadcast over output h dim 3 of
    # residual.unsqueeze(-2)). The reduction is over dim=2 -> the h2 axis.
    y = (post[..., None] * x[..., None, :]
         + np.einsum("bshH,bshd->bsHd", comb, residual)).astype(np.float32)

    os.makedirs("input",  exist_ok=True)
    os.makedirs("output", exist_ok=True)

    x.tofile("./input/input_x.bin")
    residual.tofile("./input/input_residual.bin")
    post.tofile("./input/input_post.bin")
    comb.tofile("./input/input_comb.bin")
    y.tofile("./output/golden_y.bin")

    print(f"[gen_data] B={kB} S={kS} HC_MULT={kHc} DIM={kDim}")
    print(f"[gen_data] x        range: [{x.min():.4f}, {x.max():.4f}]")
    print(f"[gen_data] residual range: [{residual.min():.4f}, {residual.max():.4f}]")
    print(f"[gen_data] post     range: [{post.min():.4f}, {post.max():.4f}]")
    print(f"[gen_data] comb     range: [{comb.min():.4f}, {comb.max():.4f}]")
    print(f"[gen_data] y        range: [{y.min():.4f}, {y.max():.4f}]")


if __name__ == "__main__":
    gen_golden_data()
