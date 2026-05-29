#!/usr/bin/python3
# coding=utf-8
# hc_sinkhorn - gen_data.py
#
# Generates input and golden data for hc_split_sinkhorn_kernel
# (deepseek/kernel.py:371-427). Per-row of mixes [N, MIX_HC]:
#
#   pre[i, :hc]   = sigmoid(mixes[i, :hc]      * hc_scale[0] + hc_base[:hc]) + eps
#   post[i, :hc]  = 2 * sigmoid(mixes[i, hc:2*hc] * hc_scale[1] + hc_base[hc:2*hc])
#   comb[i, :, :] = reshape(mixes[i, 2*hc:], (hc, hc)) * hc_scale[2]
#                    + reshape(hc_base[2*hc:], (hc, hc))
#   comb -> row-softmax + eps -> col-normalize ->
#           (sinkhorn_iters - 1) * (row-normalize, col-normalize)
#
# Output files (raw little-endian, contiguous, no header):
#   ./input/input_mixes.bin       N*MIX_HC   float32
#   ./input/input_hc_scale.bin    3          float32
#   ./input/input_hc_base.bin     MIX_HC     float32
#   ./output/golden_pre.bin       N*HC_MULT          float32
#   ./output/golden_post.bin      N*HC_MULT          float32
#   ./output/golden_comb.bin      N*HC_MULT*HC_MULT  float32

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from case_utils import load_hc_case  # noqa: E402

np.random.seed(42)

_case = load_hc_case()
kB     = _case["b"]
kS     = _case["s"]
kHc    = _case["hc_mult"]
kMixHc = _case["mix_hc"]
kIters = _case["iters"]
kEps   = float(_case.get("eps", 1e-6))

kN = kB * kS


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _row_softmax(x: np.ndarray) -> np.ndarray:
    # x: [hc, hc]
    m = x.max(axis=-1, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=-1, keepdims=True)


def gen_golden_data():
    # mixes range chosen small so sigmoid / exp stay in numerically pleasant
    # regions (no exp overflow, no sigmoid saturation).
    mixes    = np.random.uniform(-1.0, 1.0,  size=(kN, kMixHc)).astype(np.float32)
    hc_scale = np.random.uniform( 0.5, 1.5,  size=(3,)).astype(np.float32)
    hc_base  = np.random.uniform(-0.5, 0.5,  size=(kMixHc,)).astype(np.float32)

    pre  = np.zeros((kN, kHc), dtype=np.float32)
    post = np.zeros((kN, kHc), dtype=np.float32)
    comb = np.zeros((kN, kHc, kHc), dtype=np.float32)

    base_pre  = hc_base[:kHc]
    base_post = hc_base[kHc:2 * kHc]
    base_comb = hc_base[2 * kHc:].reshape(kHc, kHc)

    for i in range(kN):
        m = mixes[i]
        pre[i]  = _sigmoid(m[:kHc]            * hc_scale[0] + base_pre)  + kEps
        post[i] = 2.0 * _sigmoid(m[kHc:2*kHc] * hc_scale[1] + base_post)

        c = m[2 * kHc:].reshape(kHc, kHc) * hc_scale[2] + base_comb

        # First normalization: row-softmax + eps, then col-normalize.
        c = _row_softmax(c) + kEps
        c = c / (c.sum(axis=0, keepdims=True) + kEps)

        # (sinkhorn_iters - 1) extra iterations of row, col normalize.
        for _ in range(kIters - 1):
            c = c / (c.sum(axis=1, keepdims=True) + kEps)
            c = c / (c.sum(axis=0, keepdims=True) + kEps)

        comb[i] = c.astype(np.float32)

    os.makedirs("input",  exist_ok=True)
    os.makedirs("output", exist_ok=True)

    mixes.tofile("./input/input_mixes.bin")
    hc_scale.tofile("./input/input_hc_scale.bin")
    hc_base.tofile("./input/input_hc_base.bin")
    pre.tofile("./output/golden_pre.bin")
    post.tofile("./output/golden_post.bin")
    comb.tofile("./output/golden_comb.bin")

    print(f"[gen_data] B={kB} S={kS} HC_MULT={kHc} MIX_HC={kMixHc} iters={kIters} -> N={kN}")
    print(f"[gen_data] pre  range: [{pre.min():.4f}, {pre.max():.4f}]")
    print(f"[gen_data] post range: [{post.min():.4f}, {post.max():.4f}]")
    print(f"[gen_data] comb range: [{comb.min():.4f}, {comb.max():.4f}]")


if __name__ == "__main__":
    gen_golden_data()
