#!/usr/bin/python3
# coding=utf-8
# rms_norm - gen_data.py
#
# Generates input and golden data for Compressor RMSNorm:
#
#   var   = mean_h(x^2)                          (over the last axis, D)
#   x_hat = x * rsqrt(var + eps)
#   out   = weight * x_hat
#
# Shapes:
#   x, out : (B, SB, D)   FP32
#   weight : (D,)         FP32
#   eps    = 1e-6                                (ModelArgs.norm_eps default)
#
# D    = kCompHeadDim
# SB   = kCompS / kCompRatio
#
# Output files (raw little-endian, contiguous, no header):
#   ./input/input_x.bin         B*SB*D    float32
#   ./input/input_weight.bin    D         float32
#   ./output/golden_out.bin     B*SB*D    float32

import os
import sys
from pathlib import Path

import numpy as np

# Shared family case loader.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from case_utils import load_compressor_case  # noqa: E402

np.random.seed(47)

_case = load_compressor_case()
kB              = _case["b"]
kS              = _case["s"]
kHeadDim        = _case["head_dim"]
kCompressRatio  = _case["compress_ratio"]
kSB             = kS // kCompressRatio
kD              = kHeadDim

NORM_EPS = 1e-6


def gen_golden_data():
    x      = np.random.uniform(-2.0, 2.0, size=(kB, kSB, kD)).astype(np.float32)
    # `RMSNorm.weight` is initialized to ones in model.py:190; perturb
    # mildly so we can detect any pointwise-mul regression in the kernel.
    weight = (1.0 + np.random.uniform(-0.1, 0.1, size=(kD,))).astype(np.float32)

    var   = (x.astype(np.float32) ** 2).mean(axis=-1, keepdims=True)
    inv   = 1.0 / np.sqrt(var + NORM_EPS)
    out   = (weight[None, None, :] * (x * inv)).astype(np.float32)

    os.makedirs("input",  exist_ok=True)
    os.makedirs("output", exist_ok=True)

    x.tofile("./input/input_x.bin")
    weight.tofile("./input/input_weight.bin")
    out.tofile("./output/golden_out.bin")

    print(f"[gen_data] B={kB} S={kS} ratio={kCompressRatio} HeadDim={kD} "
          f"-> SB={kSB} D={kD} eps={NORM_EPS}")
    print(f"[gen_data] out range: [{out.min():.4f}, {out.max():.4f}]")


if __name__ == "__main__":
    gen_golden_data()
