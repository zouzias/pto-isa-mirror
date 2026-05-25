#!/usr/bin/python3
# coding=utf-8
# --------------------------------------------------------------------------------
# full_moe_separate / full_moe_combined - gen_data.py
#
# Full end-to-end MoE forward pass golden. Pipeline:
#     logits   = X @ W_router                              (fp32 acc)
#     top_idx  = argsort_desc(logits)[:, :kTopK]
#     top_val  = take_along(logits, top_idx)               (descending sorted)
#     weights  = softmax(top_val, axis=1)
#     for t in range(kT):
#         for k in range(kTopK):
#             e = top_idx[t, k]
#             Y = relu(X[t] @ W1[e]).astype(fp16)          (matches kernel Stage 1 fp32->fp16 fuse)
#             C[t] += weights[t, k] * (Y @ W2[e])
#
# Output files (raw little-endian, contiguous, no header):
#   ./input/input_X.bin           kT * kH            half (fp16)
#   ./input/input_W_router.bin    kH * kE            half
#   ./input/input_W1.bin          kE * kH * kF       half
#   ./input/input_W2.bin          kE * kF * kH       half
#   ./input/input_idx_init.bin    kE                 uint32   (identity row 0..kE-1; topk input)
#   ./output/golden_C.bin         kT * kH            float32  (final MoE output)
# --------------------------------------------------------------------------------

import json
import os
from pathlib import Path

import numpy as np

np.random.seed(101)


def _load_case_from_json():
    # Mirror generated_cases.h: first JSON entry drives the build.
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
    kT, kH, kF, kE, kTopK = _case
else:
    # v1 shape — must match the C++ side. Patched by sweep.sh.
    kT     = 256
    kH     = 64
    kF     = 64
    kE     = 32
    kTopK  = 1


def gen_golden_data():
    # FP16-representable integers (matches mani_moe / scatter / expert_ffn
    # gen_data distributions; small-int FP16 GEMM is bit-exact with FP32 acc).
    X        = (np.random.randint(-10, 11, size=(kT, kH)).astype(np.float16) / np.float16(3.0))
    W_router = (np.random.randint(-10, 11, size=(kH, kE)).astype(np.float16) / np.float16(3.0))
    W1       = (np.random.randint(-10, 11, size=(kE, kH, kF)).astype(np.float16) / np.float16(3.0))
    W2       = (np.random.randint(-10, 11, size=(kE, kF, kH)).astype(np.float16) / np.float16(3.0))

    idx_init = np.arange(kE, dtype=np.uint32)

    # Router matmul (fp16 inputs, fp32 accumulator, fp32 output).
    logits = X.astype(np.float32) @ W_router.astype(np.float32)

    # Top-K per row, descending.
    order   = np.argsort(-logits, axis=1, kind="stable")
    top_idx = order[:, :kTopK].astype(np.uint32)
    top_val = np.take_along_axis(logits, top_idx.astype(np.int64), axis=1).astype(np.float32)

    # Softmax over the kTopK values per row.
    if kTopK == 1:
        weights = np.ones((kT, 1), dtype=np.float32)
    else:
        shifted = top_val - top_val.max(axis=1, keepdims=True)
        exp_shifted = np.exp(shifted)
        weights = exp_shifted / exp_shifted.sum(axis=1, keepdims=True)

    # Per-token per-expert FFN: matches the kernel's Stage 1 fp32->fp16 cast
    # between matmuls (see kernels/automode/a2a3/MoE/expert_ffn).
    C = np.zeros((kT, kH), dtype=np.float32)
    for t in range(kT):
        for k in range(kTopK):
            e = int(top_idx[t, k])
            Y_pre = X[t].astype(np.float32) @ W1[e].astype(np.float32)
            Y_pre = np.maximum(Y_pre, 0.0)
            Y     = Y_pre.astype(np.float16)
            B_tk  = Y.astype(np.float32) @ W2[e].astype(np.float32)
            C[t] += weights[t, k] * B_tk

    os.makedirs("input",  exist_ok=True)
    os.makedirs("output", exist_ok=True)

    X       .tofile("./input/input_X.bin")
    W_router.tofile("./input/input_W_router.bin")
    W1      .tofile("./input/input_W1.bin")
    W2      .tofile("./input/input_W2.bin")
    idx_init.tofile("./input/input_idx_init.bin")
    C       .tofile("./output/golden_C.bin")

    print(f"[gen_data] kT={kT} kH={kH} kF={kF} kE={kE} kTopK={kTopK}")
    print(f"[gen_data] top_idx[0]   = {top_idx[0].tolist()}")
    print(f"[gen_data] top_val[0]   = {top_val[0].tolist()}")
    print(f"[gen_data] weights[0]   = {weights[0].tolist()}")
    print(f"[gen_data] C[0, :4]     = {C[0, :4].tolist()}")
    print(f"[gen_data] C range      = [{C.min():.3f}, {C.max():.3f}]")


if __name__ == "__main__":
    gen_golden_data()
