#!/usr/bin/python3
# coding=utf-8
# expert_ffn — gen_data.py
#
# Generates input and golden data for DeepSeek-V4 per-expert SwiGLU FFN
# over the packed-by-expert layout produced by scatter.
#
# Output files (raw little-endian, contiguous, no header):
#   ./input/input_A.bin              (T*N_ACTIVATED + 16) * DIM      bfloat16
#   ./input/input_W1.bin             N_ROUTED * INTER_DIM * DIM      bfloat16
#   ./input/input_W3.bin             N_ROUTED * INTER_DIM * DIM      bfloat16
#   ./input/input_W2.bin             N_ROUTED * DIM       * INTER_DIM bfloat16
#   ./input/input_weights.bin        (T*N_ACTIVATED + 16)            float32
#   ./input/input_expert_count.bin   N_ROUTED                        int32
#   ./input/input_expert_start.bin   N_ROUTED                        int32
#   ./output/golden_B.bin            (T*N_ACTIVATED + 16) * DIM      float32

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
kInter      = _case["inter_dim"]
kNRouted    = _case["n_routed"]
kNActivated = _case["n_activated"]
kPackedRows = kT * kNActivated
kAlloc      = kPackedRows + 16

SWIGLU_LIMIT          = float(os.environ.get("DSMOE_SWIGLU_LIMIT", "0.0"))
APPLY_ROUTING_WEIGHT  = os.environ.get("DSMOE_APPLY_ROUTING_WEIGHT", "0") == "1"


def _bf16_round(x_f32: np.ndarray) -> np.ndarray:
    raw = x_f32.astype(np.float32).view(np.uint32)
    raw = (raw + 0x8000) & 0xFFFF0000
    return raw.view(np.float32).copy()


def _to_bf16_bytes(x_f32: np.ndarray) -> bytes:
    raw = x_f32.astype(np.float32).view(np.uint32)
    raw_bf16 = ((raw + 0x8000) >> 16).astype(np.uint16)
    return raw_bf16.tobytes()


def _silu(x: np.ndarray) -> np.ndarray:
    # silu(x) = x / (1 + exp(-x)).
    return (x / (1.0 + np.exp(-x))).astype(np.float32)


def gen_golden_data():
    # Small random BF16-roundable inputs.
    A   = _bf16_round(np.random.randn(kAlloc, kDim).astype(np.float32) * 0.1)
    W1  = _bf16_round(np.random.randn(kNRouted, kInter, kDim).astype(np.float32) * 0.05)
    W3  = _bf16_round(np.random.randn(kNRouted, kInter, kDim).astype(np.float32) * 0.05)
    W2  = _bf16_round(np.random.randn(kNRouted, kDim,   kInter).astype(np.float32) * 0.05)
    w   = (np.random.rand(kAlloc).astype(np.float32) if APPLY_ROUTING_WEIGHT
           else np.zeros(kAlloc, dtype=np.float32))

    # Synthesize a realistic expert assignment: same scheme as scatter gen_data.
    expert_id = np.empty((kT, kNActivated), dtype=np.int32)
    for t in range(kT):
        expert_id[t] = np.random.choice(kNRouted, size=kNActivated, replace=False)
    count = np.bincount(expert_id.flatten(), minlength=kNRouted).astype(np.int32)
    start = np.zeros(kNRouted, dtype=np.int32)
    start[1:] = np.cumsum(count[:-1]).astype(np.int32)

    # Reference FFN: per expert, take the rows [start[e]:start[e]+count[e]].
    B = np.zeros((kAlloc, kDim), dtype=np.float32)
    for e in range(kNRouted):
        s = int(start[e]); c = int(count[e])
        if c == 0:
            continue
        x = A[s:s+c, :]                                # (c, DIM)
        gate = x @ W1[e].T                             # (c, INTER_DIM)
        up   = x @ W3[e].T                             # (c, INTER_DIM)
        if SWIGLU_LIMIT > 0.0:
            up   = np.clip(up,  -SWIGLU_LIMIT, SWIGLU_LIMIT)
            gate = np.minimum(gate, SWIGLU_LIMIT)
        y = _silu(gate) * up                           # (c, INTER_DIM)
        if APPLY_ROUTING_WEIGHT:
            y = y * w[s:s+c, None]
        B[s:s+c, :] = (y @ W2[e].T).astype(np.float32)

    os.makedirs("input",  exist_ok=True)
    os.makedirs("output", exist_ok=True)

    with open("./input/input_A.bin", "wb") as f:  f.write(_to_bf16_bytes(A))
    with open("./input/input_W1.bin", "wb") as f: f.write(_to_bf16_bytes(W1))
    with open("./input/input_W3.bin", "wb") as f: f.write(_to_bf16_bytes(W3))
    with open("./input/input_W2.bin", "wb") as f: f.write(_to_bf16_bytes(W2))
    w    .tofile("./input/input_weights.bin")
    count.tofile("./input/input_expert_count.bin")
    start.tofile("./input/input_expert_start.bin")
    B    .tofile("./output/golden_B.bin")

    print(f"[gen_data] T={kT} DIM={kDim} INTER_DIM={kInter} "
          f"N_ROUTED={kNRouted} N_ACTIVATED={kNActivated} kAlloc={kAlloc}")
    print(f"[gen_data] SWIGLU_LIMIT={SWIGLU_LIMIT}  APPLY_WEIGHT={APPLY_ROUTING_WEIGHT}")
    print(f"[gen_data] per-expert counts: {count.tolist()}")
    print(f"[gen_data] B range: [{B.min():.4f}, {B.max():.4f}]")


if __name__ == "__main__":
    gen_golden_data()
