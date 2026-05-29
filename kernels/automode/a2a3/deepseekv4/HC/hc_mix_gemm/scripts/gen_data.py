#!/usr/bin/python3
# coding=utf-8
# hc_mix_gemm - gen_data.py
#
# Generates input and golden data for the HC "mixes" projection:
#   mixes[M, N] = (x_flat[M, K] @ hc_fn[N, K]^T) * rsqrt[M]
#     x_flat : (M=B*S, K=HC_MULT*DIM)   float32
#     hc_fn  : (N=MIX_HC, K)            float32
#     rsqrt  : (M,)                     float32  (precomputed; see README)
#     mixes  : (M, N)                   float32
#
# Output files (raw little-endian, contiguous, no header):
#   ./input/input_x.bin         M*K  float32
#   ./input/input_hc_fn.bin     N*K  float32
#   ./input/input_rsqrt.bin     M    float32
#   ./output/golden_mixes.bin   M*N  float32

import os
import sys
from pathlib import Path

import numpy as np

# Shared family case loader.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from case_utils import load_hc_case  # noqa: E402

np.random.seed(42)

_case = load_hc_case()
kB     = _case["b"]
kS     = _case["s"]
kDim   = _case["dim"]
kHcM   = _case["hc_mult"]
kMixHc = _case["mix_hc"]
kEps   = float(_case.get("eps", 1e-6))

kM = kB * kS
kK = kHcM * kDim
kN = kMixHc


def gen_golden_data():
    # Small integer FP32 inputs - well within FP32 precision, easy to verify.
    x_flat = np.random.randint(-4, 5, size=(kM, kK)).astype(np.float32)
    hc_fn  = np.random.randint(-4, 5, size=(kN, kK)).astype(np.float32)

    # rsqrt of row-mean-square + eps, precomputed (model.py:678).
    row_ms = (x_flat ** 2).mean(axis=-1) + kEps           # [M]
    rsqrt  = (1.0 / np.sqrt(row_ms)).astype(np.float32)   # [M]

    # mixes = (x_flat @ hc_fn^T) * rsqrt[:, None]  (model.py:679)
    mixes = (x_flat @ hc_fn.T) * rsqrt[:, None]
    mixes = mixes.astype(np.float32)

    os.makedirs("input",  exist_ok=True)
    os.makedirs("output", exist_ok=True)

    x_flat.tofile("./input/input_x.bin")
    hc_fn.tofile("./input/input_hc_fn.bin")
    rsqrt.tofile("./input/input_rsqrt.bin")
    mixes.tofile("./output/golden_mixes.bin")

    print(f"[gen_data] B={kB} S={kS} DIM={kDim} HC_MULT={kHcM} MIX_HC={kMixHc} "
          f"-> M={kM} K={kK} N={kN}")
    print(f"[gen_data] x_flat range: [{x_flat.min():.2f}, {x_flat.max():.2f}]  "
          f"hc_fn range: [{hc_fn.min():.2f}, {hc_fn.max():.2f}]")
    print(f"[gen_data] rsqrt range: [{rsqrt.min():.4f}, {rsqrt.max():.4f}]")
    print(f"[gen_data] mixes range: [{mixes.min():.4f}, {mixes.max():.4f}]")


if __name__ == "__main__":
    gen_golden_data()
