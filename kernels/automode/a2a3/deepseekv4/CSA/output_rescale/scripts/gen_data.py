#!/usr/bin/python3
# coding=utf-8
# output_rescale — gen_data.py
#
# Generates input + golden data for the post-loop output_rescale stage.
# Mirrors kernel.py:345-350.
#
# Output files (raw little-endian, contiguous, no header):
#   ./input/input_acc_o.bin       H * D   float32
#   ./input/input_scores_max.bin  H       float32
#   ./input/input_sum_exp.bin     H       float32
#   ./input/input_attn_sink.bin   H       float32
#   ./output/golden_o.bin         H * D   bfloat16

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from case_utils import load_csa_case  # noqa: E402

np.random.seed(42)

_case = load_csa_case()
kH = _case["h"]
kD = _case["d"]


def _f32_to_bf16_bytes(x_f32: np.ndarray) -> bytes:
    raw = x_f32.astype(np.float32).view(np.uint32)
    raw_bf16 = ((raw + 0x8000) >> 16).astype(np.uint16)
    return raw_bf16.tobytes()


def _bf16_round(x_f32: np.ndarray) -> np.ndarray:
    raw = x_f32.astype(np.float32).view(np.uint32)
    raw = (raw + 0x8000) & 0xFFFF0000
    return raw.view(np.float32).copy()


def gen_golden_data():
    # Post-loop FP32 accumulator. Use small ints scaled to (-50, 50) to be a
    # realistic "sum of a few exp-weighted dot products".
    acc_o = np.random.uniform(-50.0, 50.0, size=(kH, kD)).astype(np.float32)

    # Final running stats.
    scores_max = np.random.uniform(0.0,  2.0,  size=(kH,)).astype(np.float32)
    sum_exp    = np.random.uniform(0.5, 10.0,  size=(kH,)).astype(np.float32)
    attn_sink  = np.random.uniform(-1.0, 1.0,  size=(kH,)).astype(np.float32)

    # Golden:
    sum_adj = sum_exp + np.exp(attn_sink - scores_max).astype(np.float32)
    o_f32   = (acc_o / sum_adj[:, None]).astype(np.float32)
    o_bf16  = _bf16_round(o_f32)

    os.makedirs("input",  exist_ok=True)
    os.makedirs("output", exist_ok=True)

    acc_o.tofile("./input/input_acc_o.bin")
    scores_max.tofile("./input/input_scores_max.bin")
    sum_exp.tofile("./input/input_sum_exp.bin")
    attn_sink.tofile("./input/input_attn_sink.bin")
    with open("./output/golden_o.bin", "wb") as f:
        f.write(_f32_to_bf16_bytes(o_bf16))

    print(f"[gen_data] H={kH} D={kD}")
    print(f"[gen_data] sum_adj: {sum_adj}")
    print(f"[gen_data] o range: [{o_f32.min():.4f}, {o_f32.max():.4f}]")


if __name__ == "__main__":
    gen_golden_data()
