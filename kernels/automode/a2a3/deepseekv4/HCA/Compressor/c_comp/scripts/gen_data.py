#!/usr/bin/python3
# coding=utf-8
# c_comp - gen_data.py
#
# Generates input and golden data for Compressor weighted-sum reduction:
#
#   kv_comp[b, sb, n] = sum_r kv[b, sb, r, n] * softmax_score[b, sb, r, n]
#
# Shapes:
#   kv, softmax_score : (B, SB, R, N)  FP32
#   kv_comp           : (B, SB, N)     FP32
#
# We generate `softmax_score` as a real softmax (random unnormalised
# logits → softmax along R) so the values look like what biased_softmax
# would emit. `kv` is small floats.
#
# Output files (raw little-endian, contiguous, no header):
#   ./input/input_kv.bin           B*SB*R*N  float32
#   ./input/input_softmax.bin      B*SB*R*N  float32
#   ./output/golden_kv_comp.bin    B*SB*N    float32

import os
import sys
from pathlib import Path

import numpy as np

# Shared family case loader.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from case_utils import load_compressor_case  # noqa: E402

np.random.seed(45)

_case = load_compressor_case()
kB              = _case["b"]
kS              = _case["s"]
kDim            = _case["dim"]
kHeadDim        = _case["head_dim"]
kRopeDim        = _case["rope_dim"]
kCompressRatio  = _case["compress_ratio"]
kOverlap        = _case["overlap"]
kCoff           = 2 if kOverlap else 1
kSB             = kS // kCompressRatio
kR              = (2 * kCompressRatio) if kOverlap else kCompressRatio
kN              = kCoff * kHeadDim


def _softmax(x: np.ndarray, axis: int) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def gen_golden_data():
    kv     = np.random.uniform(-2.0, 2.0, size=(kB, kSB, kR, kN)).astype(np.float32)
    logits = np.random.uniform(-1.0, 1.0, size=(kB, kSB, kR, kN)).astype(np.float32)
    softmax_score = _softmax(logits, axis=2).astype(np.float32)

    kv_comp = (kv * softmax_score).sum(axis=2).astype(np.float32)

    os.makedirs("input",  exist_ok=True)
    os.makedirs("output", exist_ok=True)

    kv.tofile("./input/input_kv.bin")
    softmax_score.tofile("./input/input_softmax.bin")
    kv_comp.tofile("./output/golden_kv_comp.bin")

    print(f"[gen_data] B={kB} S={kS} HeadDim={kHeadDim} "
          f"ratio={kCompressRatio} overlap={kOverlap} coff={kCoff}")
    print(f"[gen_data] -> SB={kSB} R={kR} N={kN}; out shape (B, SB, N)")
    print(f"[gen_data] kv_comp range: [{kv_comp.min():.4f}, {kv_comp.max():.4f}]")


if __name__ == "__main__":
    gen_golden_data()
