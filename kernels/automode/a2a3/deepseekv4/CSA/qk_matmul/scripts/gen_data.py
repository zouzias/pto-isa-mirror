#!/usr/bin/python3
# coding=utf-8
# qk_matmul — gen_data.py
#
# Generates input + golden data for the Q@K^T * scale stage of sparse_attn.
# Mirrors kernel.py:328-330 for ONE pipelined block.
#
# Output files (raw little-endian, contiguous, no header):
#   ./input/input_q.bin         H * D       bfloat16
#   ./input/input_kv_block.bin  BLOCK * D   bfloat16
#   ./input/softmax_scale.f32   1           float32
#   ./output/golden_acc_s.bin   H * BLOCK   float32

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from case_utils import load_csa_case  # noqa: E402

np.random.seed(42)

_case = load_csa_case()
kH     = _case["h"]
kD     = _case["d"]
kBlock = _case["block"]

# softmax_scale default at kernel.py:286: (1.0 / d) ** 0.5
kSoftmaxScale = (1.0 / kD) ** 0.5


def _f32_to_bf16_bytes(x_f32: np.ndarray) -> bytes:
    raw = x_f32.astype(np.float32).view(np.uint32)
    raw_bf16 = ((raw + 0x8000) >> 16).astype(np.uint16)
    return raw_bf16.tobytes()


def _bf16_round(x_f32: np.ndarray) -> np.ndarray:
    raw = x_f32.astype(np.float32).view(np.uint32)
    raw = (raw + 0x8000) & 0xFFFF0000
    return raw.view(np.float32).copy()


def gen_golden_data():
    # Small integers => bit-exact in BF16.
    Q_f32  = np.random.randint(-4, 5, size=(kH,     kD)).astype(np.float32)
    KV_f32 = np.random.randint(-4, 5, size=(kBlock, kD)).astype(np.float32)
    Q_f32  = _bf16_round(Q_f32)
    KV_f32 = _bf16_round(KV_f32)

    # FP32 accumulator: acc_s = Q @ KV^T * scale.
    acc_s = (Q_f32 @ KV_f32.T).astype(np.float32) * np.float32(kSoftmaxScale)

    os.makedirs("input",  exist_ok=True)
    os.makedirs("output", exist_ok=True)

    with open("./input/input_q.bin", "wb") as f:
        f.write(_f32_to_bf16_bytes(Q_f32))
    with open("./input/input_kv_block.bin", "wb") as f:
        f.write(_f32_to_bf16_bytes(KV_f32))
    with open("./input/softmax_scale.f32", "wb") as f:
        f.write(np.float32(kSoftmaxScale).tobytes())
    acc_s.tofile("./output/golden_acc_s.bin")

    print(f"[gen_data] H={kH} D={kD} BLOCK={kBlock}  scale={kSoftmaxScale:.6g}")
    print(f"[gen_data] acc_s range: [{acc_s.min():.4f}, {acc_s.max():.4f}]")


if __name__ == "__main__":
    gen_golden_data()
