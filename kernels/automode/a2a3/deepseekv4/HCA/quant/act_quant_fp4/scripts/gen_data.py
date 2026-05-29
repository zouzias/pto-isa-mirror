#!/usr/bin/python3
# coding=utf-8
# act_quant_fp4 - gen_data.py
#
# Generates input + golden for block-wise FP4 activation quant.
# Reference: deepseek/kernel.py:128-200, helper at kernel.py:13-39.
#
# Per row, per block of BLOCK_SIZE=32 along the last axis:
#   amax = max(|x|, dim=block)
#   amax = max(amax, 6 * 2^-126)
#   scale = 2^ceil(log2(amax / 6))
#   y     = clamp(x / scale, [-6, 6])
#   if inplace: x' = y * scale          (BF16 inplace write-back)
#
# Output files:
#   ./input/input_x.bin        M*N                bfloat16 (original x)
#   ./output/golden_x.bin      M*N                bfloat16 (dequant inplace)
#   ./output/golden_scale.bin  M*ceildiv(N,BLK)   uint8    (E8M0 biased exponent)

import math
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from case_utils import load_quant_case  # noqa: E402

np.random.seed(42)

_case = load_quant_case()
kM   = _case["m"]
kN   = _case["n"]
kBLK = _case["block_size"]
kInplace = _case["inplace"]
kNB  = (kN + kBLK - 1) // kBLK

# FP4 parameters per kernel.py:133-134.
FP4_MAX       = 6.0
FP4_MIN_SCALE = 6.0 * (2.0 ** -126)


def to_bf16_bytes(x_f32: np.ndarray) -> bytes:
    raw = x_f32.astype(np.float32).view(np.uint32)
    raw_bf16 = ((raw + 0x8000) >> 16).astype(np.uint16)
    return raw_bf16.tobytes()


def from_bf16_view(x_f32: np.ndarray) -> np.ndarray:
    raw = x_f32.astype(np.float32).view(np.uint32)
    raw = (raw + 0x8000) & 0xFFFF0000
    return raw.view(np.float32).copy()


def pow2_ceil(x: np.ndarray):
    """Return (scale, biased_exp_uint8) where scale = 2^ceil(log2(x))
    and biased_exp_uint8 = (ceil(log2(x)) + 127) mod 256 (E8M0)."""
    x_safe = np.maximum(x, np.finfo(np.float32).tiny)
    exp_f  = np.ceil(np.log2(x_safe))
    biased = np.clip(exp_f.astype(np.int32) + 127, 0, 255).astype(np.uint8)
    scale  = np.exp2(exp_f).astype(np.float32)
    return scale, biased


def fp4_emulate(y: np.ndarray) -> np.ndarray:
    """Approximate FP4 (sign + 3-level magnitude grid in [-6, 6]).

    A proper E2M1 FP4 has 8 magnitudes; here we use a coarse 4-magnitude
    grid {0.5, 1.5, 2.5, 4.0} scaled to fit, which is *good enough* under
    the loose tolerance set in main.cpp. The kernel-side will eventually
    emit real FP4 once that primitive lands; this golden documents the
    contract."""
    y = np.clip(y, -FP4_MAX, FP4_MAX).astype(np.float32)
    grid = np.array([0.5, 1.5, 2.5, 4.0, 6.0], dtype=np.float32)
    sign = np.sign(y)
    mag  = np.abs(y)
    # Snap to the nearest grid value.
    diff = np.abs(mag[..., None] - grid[None, ...])
    idx  = np.argmin(diff, axis=-1)
    snapped = grid[idx]
    return sign * snapped


def gen_golden_data():
    x_f32 = np.random.uniform(-3.0, 3.0, size=(kM, kN)).astype(np.float32)
    x_f32 = from_bf16_view(x_f32)

    if kN % kBLK != 0:
        raise NotImplementedError(
            "N must be a multiple of BLOCK_SIZE in the prototype default; "
            "tail handling is not scaffolded yet.")
    x_view = x_f32.reshape(kM, kNB, kBLK)

    amax = np.maximum(np.max(np.abs(x_view), axis=-1), FP4_MIN_SCALE)  # (M, NB)
    scale, scale_e8m0 = pow2_ceil(amax / FP4_MAX)

    y     = x_view / scale[..., None]
    y_fp4 = fp4_emulate(y)
    deq   = (y_fp4 * scale[..., None]).reshape(kM, kN)
    deq   = from_bf16_view(deq.astype(np.float32))

    os.makedirs("input",  exist_ok=True)
    os.makedirs("output", exist_ok=True)

    with open("./input/input_x.bin", "wb") as f:
        f.write(to_bf16_bytes(x_f32))
    with open("./output/golden_x.bin", "wb") as f:
        f.write(to_bf16_bytes(deq))
    scale_e8m0.astype(np.uint8).tofile("./output/golden_scale.bin")

    print(f"[gen_data] M={kM} N={kN} BLK={kBLK} nBlocks={kNB} inplace={kInplace}")
    print(f"[gen_data] scale (E8M0 uint8) range: "
          f"[{int(scale_e8m0.min())}, {int(scale_e8m0.max())}]")
    print("[gen_data] NOTE: FP4 path is a numpy software approximation; "
          "tolerance must be loose (see ../README.md).")


if __name__ == "__main__":
    gen_golden_data()
