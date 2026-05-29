#!/usr/bin/python3
# coding=utf-8
# index_score - gen_data.py
#
# Computes:
#   score[b, s, t] = sum_h ( relu(sum_d q[b, s, h, d] * kv[b, t, d])
#                            * weights[b, s, h] )
#
# Output files:
#   ./input/input_q.bin         B*S*H*D  bfloat16
#   ./input/input_kv.bin        B*T*D    bfloat16
#   ./input/input_weights.bin   B*S*H    float32
#   ./output/golden_score.bin   B*S*T    float32

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from case_utils import load_indexer_case  # noqa: E402

np.random.seed(42)

_case = load_indexer_case()
kB = _case["b"]
kS = _case["s"]
kT = _case["t"]
kH = _case["n_heads"]
kD = _case["head_dim"]


def to_bf16_bytes(x_f32: np.ndarray) -> bytes:
    raw = x_f32.view(np.uint32)
    raw_bf16 = ((raw + 0x8000) >> 16).astype(np.uint16)
    return raw_bf16.tobytes()


def gen_golden_data():
    # Small integers ([-2, 2]) keep BF16 storage bit-exact and bound drift.
    q_f32  = np.random.randint(-2, 3, size=(kB, kS, kH, kD)).astype(np.float32)
    kv_f32 = np.random.randint(-2, 3, size=(kB, kT, kD)).astype(np.float32)
    w_f32  = np.random.uniform(-1.0, 1.0,
                               size=(kB, kS, kH)).astype(np.float32)

    # einsum("bshd,btd->bsht")
    inner   = np.einsum("bshd,btd->bsht", q_f32, kv_f32, optimize=True)
    relu    = np.maximum(inner, 0.0).astype(np.float32)
    scaled  = relu * w_f32[..., None]
    score   = scaled.sum(axis=2).astype(np.float32)   # (B, S, T)

    os.makedirs("input",  exist_ok=True)
    os.makedirs("output", exist_ok=True)

    with open("./input/input_q.bin", "wb") as f:
        f.write(to_bf16_bytes(q_f32.reshape(-1)))
    with open("./input/input_kv.bin", "wb") as f:
        f.write(to_bf16_bytes(kv_f32.reshape(-1)))
    w_f32.astype(np.float32).tofile("./input/input_weights.bin")
    score.tofile("./output/golden_score.bin")

    print(f"[gen_data] B={kB} S={kS} T={kT} H={kH} D={kD}")
    print(f"[gen_data] score range: [{score.min():.4f}, {score.max():.4f}]")


if __name__ == "__main__":
    gen_golden_data()
