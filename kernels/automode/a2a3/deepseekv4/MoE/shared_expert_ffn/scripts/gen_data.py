#!/usr/bin/python3
# coding=utf-8
# shared_expert_ffn — gen_data.py
#
# Generates input and golden data for DeepSeek-V4 shared-expert SwiGLU FFN
# (deepseek/model.py:628, 644). Same FFN as expert_ffn, single-expert,
# applied to ALL tokens (no routing, no per-row weight).
#
# Output files (raw little-endian, contiguous, no header):
#   ./input/input_X.bin    T * DIM             bfloat16
#   ./input/input_W1.bin   INTER_DIM * DIM     bfloat16
#   ./input/input_W3.bin   INTER_DIM * DIM     bfloat16
#   ./input/input_W2.bin   DIM * INTER_DIM     bfloat16
#   ./output/golden_Y.bin  T * DIM             bfloat16

import os
import sys
from pathlib import Path

import numpy as np

# Shared family case loader.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from case_utils import load_dsmoe_case  # noqa: E402

np.random.seed(42)

_case = load_dsmoe_case()
kT     = _case["t"]
kDim   = _case["dim"]
kInter = _case["inter_dim"]

SWIGLU_LIMIT = float(os.environ.get("DSMOE_SWIGLU_LIMIT", "0.0"))


def _bf16_round(x_f32: np.ndarray) -> np.ndarray:
    raw = x_f32.astype(np.float32).view(np.uint32)
    raw = (raw + 0x8000) & 0xFFFF0000
    return raw.view(np.float32).copy()


def _to_bf16_bytes(x_f32: np.ndarray) -> bytes:
    raw = x_f32.astype(np.float32).view(np.uint32)
    raw_bf16 = ((raw + 0x8000) >> 16).astype(np.uint16)
    return raw_bf16.tobytes()


def _silu(x: np.ndarray) -> np.ndarray:
    return (x / (1.0 + np.exp(-x))).astype(np.float32)


def gen_golden_data():
    X  = _bf16_round(np.random.randn(kT,     kDim).astype(np.float32) * 0.1)
    W1 = _bf16_round(np.random.randn(kInter, kDim).astype(np.float32) * 0.05)
    W3 = _bf16_round(np.random.randn(kInter, kDim).astype(np.float32) * 0.05)
    W2 = _bf16_round(np.random.randn(kDim,   kInter).astype(np.float32) * 0.05)

    gate = X @ W1.T
    up   = X @ W3.T
    if SWIGLU_LIMIT > 0.0:
        up   = np.clip(up,  -SWIGLU_LIMIT, SWIGLU_LIMIT)
        gate = np.minimum(gate, SWIGLU_LIMIT)
    y = _silu(gate) * up

    Y_f32 = (y @ W2.T).astype(np.float32)
    Y = _bf16_round(Y_f32)

    os.makedirs("input",  exist_ok=True)
    os.makedirs("output", exist_ok=True)

    with open("./input/input_X.bin",  "wb") as f: f.write(_to_bf16_bytes(X))
    with open("./input/input_W1.bin", "wb") as f: f.write(_to_bf16_bytes(W1))
    with open("./input/input_W3.bin", "wb") as f: f.write(_to_bf16_bytes(W3))
    with open("./input/input_W2.bin", "wb") as f: f.write(_to_bf16_bytes(W2))
    with open("./output/golden_Y.bin", "wb") as f: f.write(_to_bf16_bytes(Y))

    print(f"[gen_data] T={kT} DIM={kDim} INTER_DIM={kInter}  SWIGLU_LIMIT={SWIGLU_LIMIT}")
    print(f"[gen_data] Y range: [{Y.min():.4f}, {Y.max():.4f}]")


if __name__ == "__main__":
    gen_golden_data()
