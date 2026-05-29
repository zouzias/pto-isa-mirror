#!/usr/bin/python3
# coding=utf-8
# biased_softmax - gen_data.py
#
# Generates input and golden data for Compressor APE-biased softmax:
#
#   tmp           = score + ape   (broadcast ape across (B, SB))
#   softmax_score = softmax(tmp, dim=2)         # axis 2 is R
#
# Shapes:
#   score, softmax_score : (B, SB, R, N)  FP32
#   ape                  : (R, N)         FP32
#
# Assumption (see README §Status): the overlap_transform `-inf` mask
# (model.py:342) is NOT applied here. The golden is plain softmax over the
# whole R axis. This keeps the kernel-side numerics trivial; the mask will
# be reintroduced when wiring the full Compressor pipeline.
#
# Output files (raw little-endian, contiguous, no header):
#   ./input/input_score.bin       B*SB*R*N  float32
#   ./input/input_ape.bin         R*N       float32
#   ./output/golden_softmax.bin   B*SB*R*N  float32

import os
import sys
from pathlib import Path

import numpy as np

# Shared family case loader.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from case_utils import load_compressor_case  # noqa: E402

np.random.seed(44)

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
    # Modest-magnitude inputs keep exp() well-conditioned.
    score = np.random.uniform(-1.0, 1.0, size=(kB, kSB, kR, kN)).astype(np.float32)
    ape   = np.random.uniform(-0.5, 0.5, size=(kR, kN)).astype(np.float32)

    biased = score + ape[None, None, :, :]     # broadcast over (B, SB)
    softmax_score = _softmax(biased, axis=2).astype(np.float32)

    os.makedirs("input",  exist_ok=True)
    os.makedirs("output", exist_ok=True)

    score.tofile("./input/input_score.bin")
    ape.tofile("./input/input_ape.bin")
    softmax_score.tofile("./output/golden_softmax.bin")

    print(f"[gen_data] B={kB} S={kS} HeadDim={kHeadDim} "
          f"ratio={kCompressRatio} overlap={kOverlap} coff={kCoff}")
    print(f"[gen_data] -> SB={kSB} R={kR} N={kN}")
    print(f"[gen_data] softmax range: [{softmax_score.min():.4f}, "
          f"{softmax_score.max():.4f}]; row-sum mean = "
          f"{softmax_score.sum(axis=2).mean():.4f}")


if __name__ == "__main__":
    gen_golden_data()
