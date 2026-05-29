#!/usr/bin/python3
# coding=utf-8
# z_gemm - gen_data.py
#
# Generates input and golden data for Compressor.wkv linear projection.
#
# Kernel computes:
#   KV[M, N] = X[M, K] @ Wkv[N, K]^T
#     X   : (M=B*S, K=DIM)            bfloat16
#     Wkv : (N=COFF*HEAD_DIM, K=DIM)  bfloat16
#     KV  : (M, N)                    float32  (FP32 accumulator)
#
# Output files (raw little-endian, contiguous, no header):
#   ./input/input_x.bin     M*K  bfloat16
#   ./input/input_wkv.bin   N*K  bfloat16
#   ./output/golden_kv.bin  M*N  float32

import json
import os
import sys
from pathlib import Path

import numpy as np

# Shared family case loader.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from case_utils import load_compressor_case  # noqa: E402

np.random.seed(42)

_case = load_compressor_case()
kB              = _case["b"]
kS              = _case["s"]
kDim            = _case["dim"]
kHeadDim        = _case["head_dim"]
kRopeDim        = _case["rope_dim"]
kCompressRatio  = _case["compress_ratio"]
kOverlap        = _case["overlap"]
kCoff           = 2 if kOverlap else 1
kM              = kB * kS
kK              = kDim
kN              = kCoff * kHeadDim


def _bf16(x: np.ndarray) -> np.ndarray:
    """Cast FP32 -> BF16 by truncating the low 16 bits of the IEEE-754 word."""
    raw = x.astype(np.float32).view(np.uint32)
    raw = (raw + 0x8000) & 0xFFFF0000  # round-to-nearest-even approx
    return raw.view(np.float32).astype(np.float32)


def gen_golden_data():
    # Small integers ([-4, 4]) are bit-exact in BF16 — keeps the FP32 golden
    # within ~1e-3 of the BF16 GEMM output.
    X_f32   = np.random.randint(-4, 5, size=(kM, kK)).astype(np.float32)
    Wkv_f32 = np.random.randint(-4, 5, size=(kN, kK)).astype(np.float32)

    # Truncate to BF16 storage. We dump the BF16 bit pattern as uint16.
    def to_bf16_bytes(x_f32: np.ndarray) -> bytes:
        raw = x_f32.view(np.uint32)
        raw_bf16 = ((raw + 0x8000) >> 16).astype(np.uint16)
        return raw_bf16.tobytes()

    KV_f32 = X_f32 @ Wkv_f32.T  # FP32 accumulator, matches kernel contract

    os.makedirs("input",  exist_ok=True)
    os.makedirs("output", exist_ok=True)

    with open("./input/input_x.bin", "wb") as f:
        f.write(to_bf16_bytes(X_f32))
    with open("./input/input_wkv.bin", "wb") as f:
        f.write(to_bf16_bytes(Wkv_f32))
    KV_f32.tofile("./output/golden_kv.bin")

    print(f"[gen_data] B={kB} S={kS} Dim={kDim} HeadDim={kHeadDim} "
          f"Coff={kCoff} -> M={kM} N={kN} K={kK}")
    print(f"[gen_data] KV range: [{KV_f32.min():.2f}, {KV_f32.max():.2f}]")


if __name__ == "__main__":
    gen_golden_data()
