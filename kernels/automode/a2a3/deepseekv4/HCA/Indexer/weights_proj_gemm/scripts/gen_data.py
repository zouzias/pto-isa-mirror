#!/usr/bin/python3
# coding=utf-8
# weights_proj_gemm - gen_data.py
#
# Kernel computes:
#   weights[M, N] = (X[M, K] @ W_wproj[N, K]^T) * scale
#     X       : (M=B*S, K=DIM)              bfloat16
#     W_wproj : (N=N_HEADS, K=DIM)          bfloat16
#     weights : (M, N)                      float32 (post-scale)
#     scale   = softmax_scale * n_heads^-0.5
#             = head_dim^-0.5 * n_heads^-0.5  (model.py:395, model.py:419)
#
# Output files:
#   ./input/input_x.bin       M*K  bfloat16
#   ./input/input_wproj.bin   N*K  bfloat16
#   ./output/golden_w.bin     M*N  float32

import math
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from case_utils import load_indexer_case  # noqa: E402

np.random.seed(42)

_case = load_indexer_case()
kB        = _case["b"]
kS        = _case["s"]
kDim      = _case["dim"]
kNHeads   = _case["n_heads"]
kHeadDim  = _case["head_dim"]
kM = kB * kS
kK = kDim
kN = kNHeads

kSoftmaxScale = 1.0 / math.sqrt(kHeadDim)
kHeadsScale   = 1.0 / math.sqrt(kNHeads)
kScale        = float(kSoftmaxScale * kHeadsScale)


def gen_golden_data():
    X_f32 = np.random.randint(-4, 5, size=(kM, kK)).astype(np.float32)
    W_f32 = np.random.randint(-4, 5, size=(kN, kK)).astype(np.float32)

    def to_bf16_bytes(x_f32: np.ndarray) -> bytes:
        raw = x_f32.view(np.uint32)
        raw_bf16 = ((raw + 0x8000) >> 16).astype(np.uint16)
        return raw_bf16.tobytes()

    W_f32_post = X_f32 @ W_f32.T               # (M, N), FP32 accumulator
    W_f32_post = W_f32_post * np.float32(kScale)

    os.makedirs("input",  exist_ok=True)
    os.makedirs("output", exist_ok=True)

    with open("./input/input_x.bin", "wb") as f:
        f.write(to_bf16_bytes(X_f32))
    with open("./input/input_wproj.bin", "wb") as f:
        f.write(to_bf16_bytes(W_f32))
    W_f32_post.astype(np.float32).tofile("./output/golden_w.bin")

    print(f"[gen_data] B={kB} S={kS} Dim={kDim} NHeads={kNHeads} "
          f"HeadDim={kHeadDim} -> M={kM} N={kN} K={kK}  scale={kScale:.6f}")
    print(f"[gen_data] weights range: "
          f"[{W_f32_post.min():.4f}, {W_f32_post.max():.4f}]")


if __name__ == "__main__":
    gen_golden_data()
