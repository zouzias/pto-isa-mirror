#!/usr/bin/python3
# coding=utf-8
# gate_logits — gen_data.py
#
# Generates input and golden data for DeepSeek-V4 router GEMM (FP32 path).
#
# Kernel computes:
#   scores[T, N_ROUTED] = X[T, DIM] @ W_gate[N_ROUTED, DIM]^T
#     X       : (T, DIM)       float32   (model.py:566 `x.float()`)
#     W_gate  : (N_ROUTED, DIM) float32  (model.py:566 `self.weight.float()`)
#     scores  : (T, N_ROUTED)  float32
#
# Output files (raw little-endian, contiguous, no header):
#   ./input/input_x.bin        T * DIM        float32
#   ./input/input_w_gate.bin   N_ROUTED * DIM float32
#   ./output/golden_scores.bin T * N_ROUTED   float32

import os
import sys
from pathlib import Path

import numpy as np

# Shared family case loader.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from case_utils import load_dsmoe_case  # noqa: E402

np.random.seed(42)

_case = load_dsmoe_case()
kT       = _case["t"]
kDim     = _case["dim"]
kNRouted = _case["n_routed"]


def gen_golden_data():
    # Small random FP32; no BF16 rounding needed since the kernel is full-FP32.
    X      = np.random.randn(kT,       kDim).astype(np.float32) * 0.1
    W_gate = np.random.randn(kNRouted, kDim).astype(np.float32) * 0.1

    scores = X @ W_gate.T  # FP32 reference

    os.makedirs("input",  exist_ok=True)
    os.makedirs("output", exist_ok=True)

    X.tofile     ("./input/input_x.bin")
    W_gate.tofile("./input/input_w_gate.bin")
    scores.tofile("./output/golden_scores.bin")

    print(f"[gen_data] T={kT} DIM={kDim} N_ROUTED={kNRouted}")
    print(f"[gen_data] scores range: [{scores.min():.3f}, {scores.max():.3f}]")


if __name__ == "__main__":
    gen_golden_data()
