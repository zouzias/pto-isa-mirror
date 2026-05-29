#!/usr/bin/python3
# coding=utf-8
# gather — gen_data.py
#
# Generates input and golden data for DeepSeek-V4 gather (expert-output →
# token-position weighted recombine).
#
# Output files (raw little-endian, contiguous, no header):
#   ./input/input_B.bin        (T*N_ACTIVATED + 16) * DIM      bfloat16
#   ./input/input_A_id.bin     (T*N_ACTIVATED + 16)            int32
#   ./input/input_rank_id.bin  (T*N_ACTIVATED + 16)            int32
#   ./input/input_weights.bin  T * N_ACTIVATED                 float32
#   ./output/golden_Y.bin      T * DIM                         bfloat16

import os
import sys
from pathlib import Path

import numpy as np

# Shared family case loader.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from case_utils import load_dsmoe_case  # noqa: E402

np.random.seed(42)

_case = load_dsmoe_case()
kT          = _case["t"]
kDim        = _case["dim"]
kNRouted    = _case["n_routed"]
kNActivated = _case["n_activated"]
kPackedRows = kT * kNActivated
kAlloc      = kPackedRows + 16


def _bf16_round(x_f32: np.ndarray) -> np.ndarray:
    raw = x_f32.astype(np.float32).view(np.uint32)
    raw = (raw + 0x8000) & 0xFFFF0000
    return raw.view(np.float32).copy()


def _to_bf16_bytes(x_f32: np.ndarray) -> bytes:
    raw = x_f32.astype(np.float32).view(np.uint32)
    raw_bf16 = ((raw + 0x8000) >> 16).astype(np.uint16)
    return raw_bf16.tobytes()


def gen_golden_data():
    # Build a synthetic scatter assignment so A_id / rank_id are valid.
    expert_id = np.empty((kT, kNActivated), dtype=np.int32)
    for t in range(kT):
        expert_id[t] = np.random.choice(kNRouted, size=kNActivated, replace=False)

    count    = np.bincount(expert_id.flatten(), minlength=kNRouted).astype(np.int32)
    start    = np.zeros(kNRouted, dtype=np.int32)
    start[1:] = np.cumsum(count[:-1]).astype(np.int32)

    # Reconstruct A_id / rank_id in scatter order.
    A_id    = np.full((kAlloc,), -1, dtype=np.int32)
    rank_id = np.full((kAlloc,), -1, dtype=np.int32)
    counter = np.zeros(kNRouted, dtype=np.int32)
    for t in range(kT):
        for k in range(kNActivated):
            e = int(expert_id[t, k])
            r = int(start[e] + counter[e])
            counter[e] += 1
            A_id   [r] = t
            rank_id[r] = k

    # B: per-expert output values, BF16-roundable random.
    B = _bf16_round(np.random.randn(kAlloc, kDim).astype(np.float32) * 0.1)

    # Routing weights (post-normalize, post-route_scale). Make each row sum
    # to ~1 to mimic the upstream gate output.
    raw = np.random.rand(kT, kNActivated).astype(np.float32) + 0.1
    weights = raw / raw.sum(axis=-1, keepdims=True)

    # Reference gather.
    Y_f32 = np.zeros((kT, kDim), dtype=np.float32)
    for r in range(kPackedRows):
        t = int(A_id   [r])
        k = int(rank_id[r])
        w = float(weights[t, k])
        Y_f32[t, :] += w * B[r, :]

    Y = _bf16_round(Y_f32)

    os.makedirs("input",  exist_ok=True)
    os.makedirs("output", exist_ok=True)

    with open("./input/input_B.bin", "wb") as f:
        f.write(_to_bf16_bytes(B))
    A_id   .tofile("./input/input_A_id.bin")
    rank_id.tofile("./input/input_rank_id.bin")
    weights.tofile("./input/input_weights.bin")
    with open("./output/golden_Y.bin", "wb") as f:
        f.write(_to_bf16_bytes(Y))

    print(f"[gen_data] T={kT} DIM={kDim} N_ACTIVATED={kNActivated} kAlloc={kAlloc}")
    print(f"[gen_data] Y range: [{Y.min():.4f}, {Y.max():.4f}]")


if __name__ == "__main__":
    gen_golden_data()
