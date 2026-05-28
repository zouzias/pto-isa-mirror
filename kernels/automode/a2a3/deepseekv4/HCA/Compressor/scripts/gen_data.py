#!/usr/bin/env python3
# coding=utf-8
# --------------------------------------------------------------------------------
# HCA Compressor - gen_data.py
#
# Generates input tensors and PyTorch ground-truth output for the HCA Compressor.
#
# Model: DeepSeek-V4-Flash, compress_ratio = 128.
# Forward pass (mirrors model.py Compressor with ratio=128, no overlap):
#
#   C  = F.linear(H, W_kv)                            # [B, S, c]   GEMM
#   Z  = F.linear(H, W_z)                             # [B, S, c]   GEMM
#   scores = softmax(Z.view(B, nb, r, c) + B, dim=2)  # [B, nb, r, c]  Biased Softmax
#   out    = (C.view(B, nb, r, c) * scores).sum(dim=2) # [B, nb, c]  Weighted Sum
#
# where nb = S // r  (number of blocks), r = 128.
#
# Dtypes:
#   H, W_kv, W_z : float16  (kernel input)
#   B (bias)     : float32  (matches model.py's fp32 ape parameter)
#   Computation  : float32  (for numerical precision of golden)
#   Saved outputs: float16  (C, Z, scores, out — to match kernel output dtype)
#
# Output files (all raw little-endian, no header):
#   input/input_H.bin       [B*S*d  float16]
#   input/input_W_kv.bin    [c*d    float16]   (weight: [out=c, in=d])
#   input/input_W_z.bin     [c*d    float16]
#   input/input_B.bin       [r*c    float32]
#   output/golden_C.bin     [B*S*c  float16]   debug: GEMM output for KV
#   output/golden_Z.bin     [B*S*c  float16]   debug: GEMM output for weights
#   output/golden_scores.bin[B*nb*r*c float16] debug: softmax output
#   output/golden_output.bin[B*nb*c float16]   final ground truth
# --------------------------------------------------------------------------------

import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F

torch.manual_seed(42)


def load_generated_case():
    case_path = Path(__file__).resolve().parent.parent / "build" / "generated_cases.json"
    if not case_path.exists():
        return {
            "batch": 1,
            "seq_len": 128,
            "hidden_dim": 4096,
            "compress_dim": 512,
            "compress_ratio": 128,
        }
    with case_path.open("r", encoding="utf-8") as f:
        cases = json.load(f)
    return cases[0]


def save_fp16(tensor: torch.Tensor, path: str) -> None:
    tensor.contiguous().to(torch.float16).numpy().tofile(path)


def save_fp32(tensor: torch.Tensor, path: str) -> None:
    tensor.contiguous().to(torch.float32).numpy().tofile(path)


def gen_data(case: dict) -> None:
    B   = case["batch"]
    S   = case["seq_len"]
    d   = case["hidden_dim"]
    c   = case["compress_dim"]
    r   = case["compress_ratio"]
    nb  = S // r   # number of compressed blocks

    print(f"[INFO] Shapes: H=[{B},{S},{d}]  W_kv/W_z=[{c},{d}]  B=[{r},{c}]")
    print(f"[INFO] Output: [{B},{nb},{c}]  ({nb} block(s))")

    # Generate inputs
    H    = torch.randn(B, S, d,  dtype=torch.float16)
    W_kv = torch.randn(c, d,     dtype=torch.float16)
    W_z  = torch.randn(c, d,     dtype=torch.float16)
    B_bias = torch.randn(r, c,   dtype=torch.float32)

    # Forward pass in float32 for numerical precision
    H_f  = H.float()
    Wkv_f = W_kv.float()
    Wz_f  = W_z.float()

    # Stage 1: GEMM
    C = F.linear(H_f, Wkv_f)                 # [B, S, c]
    Z = F.linear(H_f, Wz_f)                  # [B, S, c]

    # Stage 2: Biased Softmax (reshape → add bias → softmax along ratio dim)
    C_blocks = C.unflatten(1, (nb, r))        # [B, nb, r, c]
    Z_blocks = Z.unflatten(1, (nb, r))        # [B, nb, r, c]
    scores   = (Z_blocks + B_bias).softmax(dim=2)  # [B, nb, r, c]

    # Stage 3: Weighted Sum
    output = (C_blocks * scores).sum(dim=2)   # [B, nb, c]

    # Write files
    os.makedirs("input",  exist_ok=True)
    os.makedirs("output", exist_ok=True)

    save_fp16(H,      "./input/input_H.bin")
    save_fp16(W_kv,   "./input/input_W_kv.bin")
    save_fp16(W_z,    "./input/input_W_z.bin")
    save_fp32(B_bias, "./input/input_B.bin")

    save_fp16(C,      "./output/golden_C.bin")
    save_fp16(Z,      "./output/golden_Z.bin")
    save_fp16(scores, "./output/golden_scores.bin")
    save_fp16(output, "./output/golden_output.bin")

    print("[INFO] Wrote input/input_H.bin       "
          f"({H.numel() * 2} bytes, float16)")
    print("[INFO] Wrote input/input_W_kv.bin    "
          f"({W_kv.numel() * 2} bytes, float16)")
    print("[INFO] Wrote input/input_W_z.bin     "
          f"({W_z.numel() * 2} bytes, float16)")
    print("[INFO] Wrote input/input_B.bin       "
          f"({B_bias.numel() * 4} bytes, float32)")
    print("[INFO] Wrote output/golden_C.bin     "
          f"({C.numel() * 2} bytes, float16)")
    print("[INFO] Wrote output/golden_Z.bin     "
          f"({Z.numel() * 2} bytes, float16)")
    print("[INFO] Wrote output/golden_scores.bin"
          f" ({scores.numel() * 2} bytes, float16)")
    print("[INFO] Wrote output/golden_output.bin"
          f" ({output.numel() * 2} bytes, float16)")
    print(f"[INFO] output min={output.min():.4f}  max={output.max():.4f}")


if __name__ == "__main__":
    case = load_generated_case()
    gen_data(case)
