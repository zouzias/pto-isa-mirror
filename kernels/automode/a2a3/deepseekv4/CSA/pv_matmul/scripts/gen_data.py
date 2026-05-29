#!/usr/bin/python3
# coding=utf-8
# pv_matmul — gen_data.py
#
# Generates input + golden data for the PV-matmul stage (acc_o += acc_s_cast @ V).
# Mirrors kernel.py:343 — note the vector-side rescale (acc_o *= scores_scale)
# at kernel.py:341-342 is APPLIED IN THE INPUT file by this script, so the
# kernel only needs to do the cube GEMM accumulate.
#
# Output files (raw little-endian, contiguous, no header):
#   ./input/input_acc_o.bin      H * D       float32  (pre-rescaled)
#   ./input/input_acc_s_cast.bin H * BLOCK   bfloat16
#   ./input/input_kv_block.bin   BLOCK * D   bfloat16
#   ./output/golden_acc_o.bin    H * D       float32

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


def _f32_to_bf16_bytes(x_f32: np.ndarray) -> bytes:
    raw = x_f32.astype(np.float32).view(np.uint32)
    raw_bf16 = ((raw + 0x8000) >> 16).astype(np.uint16)
    return raw_bf16.tobytes()


def _bf16_round(x_f32: np.ndarray) -> np.ndarray:
    raw = x_f32.astype(np.float32).view(np.uint32)
    raw = (raw + 0x8000) & 0xFFFF0000
    return raw.view(np.float32).copy()


def gen_golden_data():
    # Prior accumulator (already rescaled by the driver). Small ints.
    acc_o_in = np.random.randint(-4, 5, size=(kH, kD)).astype(np.float32)

    # acc_s_cast: post-exp soft probabilities cast to BF16. We synthesize
    # them as small positive BF16 values in (0, 1).
    sc_f32 = np.random.uniform(0.0, 1.0, size=(kH, kBlock)).astype(np.float32)
    sc_f32 = _bf16_round(sc_f32)

    # V block: small ints, BF16-bit-exact.
    v_f32 = np.random.randint(-4, 5, size=(kBlock, kD)).astype(np.float32)
    v_f32 = _bf16_round(v_f32)

    # Golden: acc_o += acc_s_cast @ V (FP32 accumulator).
    acc_o_out = acc_o_in + sc_f32 @ v_f32

    os.makedirs("input",  exist_ok=True)
    os.makedirs("output", exist_ok=True)

    acc_o_in.tofile("./input/input_acc_o.bin")
    with open("./input/input_acc_s_cast.bin", "wb") as f:
        f.write(_f32_to_bf16_bytes(sc_f32))
    with open("./input/input_kv_block.bin", "wb") as f:
        f.write(_f32_to_bf16_bytes(v_f32))
    acc_o_out.tofile("./output/golden_acc_o.bin")

    print(f"[gen_data] H={kH} D={kD} BLOCK={kBlock}")
    print(f"[gen_data] acc_o_in  range: [{acc_o_in.min():.2f}, {acc_o_in.max():.2f}]")
    print(f"[gen_data] acc_o_out range: [{acc_o_out.min():.2f}, {acc_o_out.max():.2f}]")


if __name__ == "__main__":
    gen_golden_data()
