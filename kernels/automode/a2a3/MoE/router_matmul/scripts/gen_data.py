#!/usr/bin/python3
# coding=utf-8
# router_matmul - gen_data.py
#
# Generates input and golden data for the router matmul kernel.
# Kernel computes: logits = X @ W_router
#   X          : (kT, kH) float16  (token features)
#   W_router   : (kH, kE) float16  (router projection)
#   logits     : (kT, kE) float32  (router logits per token per expert)
#
# Integer inputs in [-4, 4] are exactly representable in FP16. The golden is
# computed in float32 (matching the kernel's FP32 accumulator), so results
# should match to ~0.01f.
#
# Output files (raw little-endian, contiguous, no header):
#   ./input/input_x.bin           kT * kH  float16
#   ./input/input_w_router.bin    kH * kE  float16
#   ./output/golden_logits.bin    kT * kE  float32

import json
import os
from pathlib import Path

import numpy as np

np.random.seed(42)


def _load_case_from_json():
    # Mirror generated_cases.h: first JSON entry drives the build.
    # Path: <MoE>/build/generated_cases.json (parents[2] = <MoE>).
    json_path = Path(__file__).resolve().parents[2] / "build" / "generated_cases.json"
    if not json_path.exists():
        return None
    payload = json.loads(json_path.read_text())
    if not payload:
        return None
    c = payload[0]
    return int(c["t"]), int(c["h"]), int(c["f"]), int(c["e"]), int(c["topk"])


_case = _load_case_from_json()
if _case is not None:
    kT, kH, _kF, kE, _kTopK = _case
else:
    kT = 256  # num_tokens
    kH = 64   # d_model
    kE = 32   # num_experts


def gen_golden_data():
    # Small integer inputs: exactly representable in FP16.
    X        = np.random.randint(-4, 5, size=(kT, kH)).astype(np.float16)
    W_router = np.random.randint(-4, 5, size=(kH, kE)).astype(np.float16)

    # Golden: upcast to float32 before matmul (matches kernel FP32 accumulator).
    logits = X.astype(np.float32) @ W_router.astype(np.float32)

    os.makedirs("input",  exist_ok=True)
    os.makedirs("output", exist_ok=True)

    X.tofile("./input/input_x.bin")
    W_router.tofile("./input/input_w_router.bin")
    logits.tofile("./output/golden_logits.bin")

    print(f"[gen_data] kT={kT}  kH={kH}  kE={kE}")
    print(f"[gen_data] X.dtype={X.dtype}  W_router.dtype={W_router.dtype}  logits.dtype={logits.dtype}")
    print(f"[gen_data] logits range: [{logits.min():.2f}, {logits.max():.2f}]")
    print(f"[gen_data] logits[0, :8] = {logits[0, :8].tolist()}")


if __name__ == "__main__":
    gen_golden_data()
