#!/usr/bin/python3
# coding=utf-8
# act_quant_fp8 - gen_data.py
#
# Generates input + golden for block-wise FP8 (E4M3) activation quant.
# Reference: deepseek/kernel.py:41-102, helper at kernel.py:13-39.
#
# Per row, per block of BLOCK_SIZE along the last axis:
#   amax = max(|x|, dim=block)
#   amax = max(amax, 1e-4)
#   scale = round_to_pow2(amax / 448)
#   y     = clamp(x / scale, [-448, 448])
#   if inplace: x' = y * scale          (BF16 inplace write-back)
#
# Output files:
#   ./input/input_x.bin        M*N                bfloat16 (original x)
#   ./output/golden_x.bin      M*N                bfloat16 (dequant inplace)
#   ./output/golden_scale.bin  M*ceildiv(N,BLK)   float32

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

# FP8 (E4M3) parameters per kernel.py:46-47.
FP8_MAX = 448.0
FP8_MIN_SCALE = 1.0e-4


def to_bf16_bytes(x_f32: np.ndarray) -> bytes:
    raw = x_f32.astype(np.float32).view(np.uint32)
    raw_bf16 = ((raw + 0x8000) >> 16).astype(np.uint16)
    return raw_bf16.tobytes()


def from_bf16_view(x_f32: np.ndarray) -> np.ndarray:
    """Round x_f32 to BF16 storage and return as fp32 ndarray
    with the truncated low 16 bits. Lets us emulate the kernel's
    inplace write-back precision."""
    raw = x_f32.astype(np.float32).view(np.uint32)
    raw = (raw + 0x8000) & 0xFFFF0000
    return raw.view(np.float32).copy()


def round_to_pow2(x: np.ndarray) -> np.ndarray:
    """Round each element to the nearest power of 2 (ceil-exp variant
    matching fast_round_scale in deepseek/kernel.py:13-39).

    For x > 0: 2 ^ ceil(log2(x)). Per-row vectorized."""
    x_safe = np.maximum(x, np.finfo(np.float32).tiny)
    exp = np.ceil(np.log2(x_safe))
    return np.exp2(exp).astype(np.float32)


def fp8_emulate(y: np.ndarray) -> np.ndarray:
    """Software approximation of FP8 (E4M3) cast: clamp to +/- 448 and
    round to a coarse representable grid. We do not implement exact E4M3
    rounding here; instead we clamp and quantize to a 4-bit mantissa-ish
    grid in log space — *good enough* for a golden the kernel can compare
    against under a loose tolerance (5e-2 relative)."""
    # 1) clamp
    y = np.clip(y, -FP8_MAX, FP8_MAX).astype(np.float32)
    # 2) approximate E4M3 rounding: split sign / exponent / 4-bit mantissa.
    sign = np.sign(y).astype(np.float32)
    mag  = np.abs(y) + 1e-38
    exp  = np.floor(np.log2(mag)).astype(np.float32)
    frac = mag / np.exp2(exp)                       # in [1, 2)
    # 4-bit mantissa: 16 buckets in [1, 2)
    frac_q = np.round((frac - 1.0) * 16.0) / 16.0 + 1.0
    return sign * frac_q * np.exp2(exp)


def gen_golden_data():
    # Use a small range so BF16 quant + FP8 round-trip stays bounded.
    x_f32 = np.random.uniform(-3.0, 3.0, size=(kM, kN)).astype(np.float32)
    # Snap input to BF16 storage so the inplace round-trip is well defined.
    x_f32 = from_bf16_view(x_f32)

    # Per-row, per-block.
    x_view = x_f32.reshape(kM, kNB, kBLK) if kN % kBLK == 0 else None
    if x_view is None:
        raise NotImplementedError(
            "N must be a multiple of BLOCK_SIZE in the prototype default; "
            "tail handling is not scaffolded yet.")

    amax = np.maximum(np.max(np.abs(x_view), axis=-1), FP8_MIN_SCALE)  # (M, NB)
    scale = round_to_pow2(amax / FP8_MAX).astype(np.float32)            # (M, NB)

    # Quantize then dequantize for the inplace BF16 output.
    y      = x_view / scale[..., None]
    y_fp8  = fp8_emulate(y)
    deq    = (y_fp8 * scale[..., None]).reshape(kM, kN)
    deq    = from_bf16_view(deq.astype(np.float32))

    os.makedirs("input",  exist_ok=True)
    os.makedirs("output", exist_ok=True)

    with open("./input/input_x.bin", "wb") as f:
        f.write(to_bf16_bytes(x_f32))
    with open("./output/golden_x.bin", "wb") as f:
        f.write(to_bf16_bytes(deq))
    scale.astype(np.float32).tofile("./output/golden_scale.bin")

    print(f"[gen_data] M={kM} N={kN} BLK={kBLK} nBlocks={kNB} inplace={kInplace}")
    print(f"[gen_data] scale range: [{scale.min():.6f}, {scale.max():.6f}]")
    print("[gen_data] NOTE: FP8 path is a numpy software approximation; "
          "tolerance must be loose (see ../README.md).")


if __name__ == "__main__":
    gen_golden_data()
