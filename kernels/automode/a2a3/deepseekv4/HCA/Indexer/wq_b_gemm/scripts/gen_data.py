#!/usr/bin/python3
# coding=utf-8
# wq_b_gemm - gen_data.py
#
# Generates input and golden data for Indexer.wq_b low-rank Q expansion.
#
# Kernel computes:
#   q[M, N] = qr[M, K] @ Wq_b[N, K]^T
#     qr   : (M=B*S, K=Q_LORA_RANK)            bfloat16
#     Wq_b : (N=N_HEADS*HEAD_DIM, K=Q_LORA_RANK) bfloat16
#     q    : (M, N)                             float32  (FP32 accumulator)
#
# Output files (raw little-endian, contiguous, no header):
#   ./input/input_qr.bin    M*K  bfloat16
#   ./input/input_wq_b.bin  N*K  bfloat16
#   ./output/golden_q.bin   M*N  float32

import os
import sys
from pathlib import Path

import numpy as np

# Shared family case loader.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from case_utils import load_indexer_case  # noqa: E402

np.random.seed(42)

_case = load_indexer_case()
kB          = _case["b"]
kS          = _case["s"]
kQLoraRank  = _case["q_lora_rank"]
kNHeads     = _case["n_heads"]
kHeadDim    = _case["head_dim"]
kM          = kB * kS
kK          = kQLoraRank
kN          = kNHeads * kHeadDim


def gen_golden_data():
    # Small integers ([-4, 4]) are bit-exact in BF16; keeps the FP32 golden
    # within ~1e-3 of the BF16 GEMM output.
    qr_f32   = np.random.randint(-4, 5, size=(kM, kK)).astype(np.float32)
    Wq_b_f32 = np.random.randint(-4, 5, size=(kN, kK)).astype(np.float32)

    def to_bf16_bytes(x_f32: np.ndarray) -> bytes:
        raw = x_f32.view(np.uint32)
        raw_bf16 = ((raw + 0x8000) >> 16).astype(np.uint16)
        return raw_bf16.tobytes()

    q_f32 = qr_f32 @ Wq_b_f32.T  # FP32 accumulator, matches kernel contract

    os.makedirs("input",  exist_ok=True)
    os.makedirs("output", exist_ok=True)

    with open("./input/input_qr.bin", "wb") as f:
        f.write(to_bf16_bytes(qr_f32))
    with open("./input/input_wq_b.bin", "wb") as f:
        f.write(to_bf16_bytes(Wq_b_f32))
    q_f32.tofile("./output/golden_q.bin")

    print(f"[gen_data] B={kB} S={kS} QLR={kQLoraRank} NHeads={kNHeads} "
          f"HeadDim={kHeadDim} -> M={kM} N={kN} K={kK}")
    print(f"[gen_data] q range: [{q_f32.min():.2f}, {q_f32.max():.2f}]")


if __name__ == "__main__":
    gen_golden_data()
