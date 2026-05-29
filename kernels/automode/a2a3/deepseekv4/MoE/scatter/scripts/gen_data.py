#!/usr/bin/python3
# coding=utf-8
# scatter — gen_data.py
#
# Generates input and golden data for DeepSeek-V4 scatter (BF16 row dtype).
#
# Output files (raw little-endian, contiguous, no header):
#   ./input/input_X.bin              T * DIM                         bfloat16
#   ./input/input_expert_id.bin      T * N_ACTIVATED                 int32
#   ./output/golden_A.bin            (T*N_ACTIVATED + 16) * DIM      bfloat16
#   ./output/golden_A_id.bin         (T*N_ACTIVATED + 16)            int32
#   ./output/golden_rank_id.bin      (T*N_ACTIVATED + 16)            int32
#   ./output/golden_expert_count.bin N_ROUTED                        int32
#   ./output/golden_expert_start.bin N_ROUTED                        int32

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


def _f32_to_bf16_bytes(x_f32: np.ndarray) -> bytes:
    raw = x_f32.astype(np.float32).view(np.uint32)
    raw_bf16 = ((raw + 0x8000) >> 16).astype(np.uint16)
    return raw_bf16.tobytes()


def gen_golden_data():
    # X as BF16 (stored as the BF16 bit pattern). Small integers stay
    # bit-exact in BF16 — ideal for a byte-exact compare.
    X_f32 = np.random.randint(-4, 5, size=(kT, kDim)).astype(np.float32)

    # expert_id: each token picks N_ACTIVATED experts from [0, N_ROUTED).
    expert_id = np.empty((kT, kNActivated), dtype=np.int32)
    for t in range(kT):
        expert_id[t] = np.random.choice(kNRouted, size=kNActivated, replace=False)

    # Reference scatter — identical to MoE/scatter/scripts/gen_data.py logic.
    count    = np.bincount(expert_id.flatten(), minlength=kNRouted).astype(np.int32)
    start    = np.zeros(kNRouted, dtype=np.int32)
    start[1:] = np.cumsum(count[:-1]).astype(np.int32)

    # Packed buffers (size = kAlloc, only the first kPackedRows are written).
    A      = np.zeros((kAlloc, kDim), dtype=np.float32)   # FP32 view; reduces to BF16 bytes below
    A_id   = np.full((kAlloc,), -1, dtype=np.int32)        # -1 sentinel for pad
    rank_id = np.full((kAlloc,), -1, dtype=np.int32)

    counter = np.zeros(kNRouted, dtype=np.int32)
    for t in range(kT):
        for k in range(kNActivated):
            e = int(expert_id[t, k])
            r = int(start[e] + counter[e])
            counter[e] += 1
            A[r, :]    = X_f32[t, :]
            A_id[r]    = t
            rank_id[r] = k

    os.makedirs("input",  exist_ok=True)
    os.makedirs("output", exist_ok=True)

    with open("./input/input_X.bin", "wb") as f:
        f.write(_f32_to_bf16_bytes(X_f32))
    expert_id.tofile("./input/input_expert_id.bin")

    with open("./output/golden_A.bin", "wb") as f:
        f.write(_f32_to_bf16_bytes(A))
    A_id     .tofile("./output/golden_A_id.bin")
    rank_id  .tofile("./output/golden_rank_id.bin")
    count    .tofile("./output/golden_expert_count.bin")
    start    .tofile("./output/golden_expert_start.bin")

    print(f"[gen_data] T={kT} DIM={kDim} N_ROUTED={kNRouted} "
          f"N_ACTIVATED={kNActivated} kAlloc={kAlloc}")
    print(f"[gen_data] per-expert counts: {count.tolist()}")


if __name__ == "__main__":
    gen_golden_data()
