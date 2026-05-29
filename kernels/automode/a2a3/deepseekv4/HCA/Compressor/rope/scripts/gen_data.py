#!/usr/bin/python3
# coding=utf-8
# rope - gen_data.py
#
# Generates input and golden data for Compressor.apply_rotary_emb on
# kv_comp[..., -rd:].
#
# Real-arithmetic formulation (matches the kernel pseudocode):
#     re_in  = x[..., 0::2]
#     im_in  = x[..., 1::2]
#     re_out = re_in * cs - im_in * sn
#     im_out = re_in * sn + im_in * cs
#     out[..., 0::2] = re_out
#     out[..., 1::2] = im_out
#
# Shapes:
#   kv_tail, rotated     : (B, SB, RD)     FP32
#   freqs_cos, freqs_sin : (SB, RD/2)      FP32
#
# We mirror precompute_freqs_cis (model.py:200-230) with the simple base
# branch (original_seq_len = 0 → no YaRN scaling). That is sufficient for
# this prototype's numerical contract.
#
# Output files (raw little-endian, contiguous, no header):
#   ./input/input_kv_tail.bin       B*SB*RD     float32
#   ./input/input_freqs_cos.bin     SB*(RD/2)   float32
#   ./input/input_freqs_sin.bin     SB*(RD/2)   float32
#   ./output/golden_rotated.bin     B*SB*RD     float32

import os
import sys
from pathlib import Path

import numpy as np

# Shared family case loader.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from case_utils import load_compressor_case  # noqa: E402

np.random.seed(46)

_case = load_compressor_case()
kB              = _case["b"]
kS              = _case["s"]
kRopeDim        = _case["rope_dim"]
kCompressRatio  = _case["compress_ratio"]
kSB             = kS // kCompressRatio
kRD             = kRopeDim
kHalfRD         = kRD // 2

ROPE_BASE = 10000.0


def gen_golden_data():
    # kv_tail: small floats, shape (B, SB, RD)
    kv_tail = np.random.uniform(-1.0, 1.0, size=(kB, kSB, kRD)).astype(np.float32)

    # Simple freqs (no YaRN scaling).
    # freqs[k] = 1 / base ** (2k / RD)
    freqs = 1.0 / (ROPE_BASE ** (np.arange(0, kRD, 2, dtype=np.float32) / kRD))
    t = np.arange(kSB, dtype=np.float32)
    angles = np.outer(t, freqs)             # (SB, RD/2)
    freqs_cos = np.cos(angles).astype(np.float32)
    freqs_sin = np.sin(angles).astype(np.float32)

    re_in = kv_tail[..., 0::2]
    im_in = kv_tail[..., 1::2]
    cs = freqs_cos[None, :, :]
    sn = freqs_sin[None, :, :]
    re_out = re_in * cs - im_in * sn
    im_out = re_in * sn + im_in * cs

    rotated = np.empty_like(kv_tail)
    rotated[..., 0::2] = re_out
    rotated[..., 1::2] = im_out

    os.makedirs("input",  exist_ok=True)
    os.makedirs("output", exist_ok=True)

    kv_tail.tofile("./input/input_kv_tail.bin")
    freqs_cos.tofile("./input/input_freqs_cos.bin")
    freqs_sin.tofile("./input/input_freqs_sin.bin")
    rotated.tofile("./output/golden_rotated.bin")

    print(f"[gen_data] B={kB} S={kS} ratio={kCompressRatio} RD={kRD} -> "
          f"SB={kSB} half={kHalfRD}")
    print(f"[gen_data] rotated range: [{rotated.min():.4f}, {rotated.max():.4f}]")


if __name__ == "__main__":
    gen_golden_data()
