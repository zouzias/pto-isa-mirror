#!/usr/bin/python3
# coding=utf-8
# --------------------------------------------------------------------------------
# expert_ffn - gen_data.py
#
# Generates pre-packed inputs and reference golden for the two-stage per-expert
# FFN kernel. We synthesize a plausible (expert_count, expert_start, A) tuple
# directly (without running the scatter kernel — this folder is independent),
# then compute the per-expert FFN reference.
#
# Output files (raw little-endian, contiguous, no header):
#   ./input/input_A.bin             (kT*kTopK + 16) * kH        half (fp16)
#   ./input/input_expert_count.bin   kE                          int32
#   ./input/input_expert_start.bin   kE                          int32
#   ./input/input_W1.bin             kE * kH_aligned * kF_aligned half (zero-padded to blockAlign)
#   ./input/input_W2.bin             kE * kF_aligned * kH_aligned half (zero-padded to blockAlign)
#   ./output/golden_B.bin            (kT*kTopK + 16) * kH        float32  (trailing 16 rows = 0)
# --------------------------------------------------------------------------------

import os
import numpy as np

np.random.seed(31)

# v1 shape — must match expert_ffn_kernel.cpp and main.cpp.
kT     = 256
kH     = 64
kF     = 64
kE     = 32
kTopK  = 1
kTileM = 16

kPackedRows   = kT * kTopK
kOverspillPad = kTileM
kAlloc        = kPackedRows + kOverspillPad

# Cube blockAlign for fp16 = C0_SIZE_BYTE(32) / sizeof(fp16)(2) = 16.
# A and W must be padded to multiples of 16 so the GEMM MatTile K/N columns
# that exceed the logical dimension are zero, not uninitialized UB garbage.
kH_aligned = ((kH + 15) // 16) * 16
kF_aligned = ((kF + 15) // 16) * 16


def gen_golden_data():
    # Reproduce the (count, start, A_id-driven A) state that the scatter
    # kernel would emit. We don't need A_id here — only A, count, and start.
    expert_id = np.zeros((kT, kTopK), dtype=np.int32)
    for t in range(kT):
        expert_id[t] = np.random.choice(kE, size=kTopK, replace=False).astype(np.int32)
    expert_id_flat = expert_id.flatten()
    expert_count = np.bincount(expert_id_flat, minlength=kE).astype(np.int32)
    expert_start = np.zeros(kE, dtype=np.int32)
    expert_start[1:] = np.cumsum(expert_count[:-1])

    # X in [-10, 10] / 3 rounded to fp16 (same as scatter/gen_data).
    X = (np.random.randint(-10, 11, size=(kT, kH)).astype(np.float16) / np.float16(3.0))
    token_ids_per_pair = np.repeat(np.arange(kT, dtype=np.int32), kTopK)
    order = np.argsort(expert_id_flat, kind="stable")
    A_valid = X[token_ids_per_pair[order]]  # (kPackedRows, kH) fp16

    # A: (kAlloc, kH_aligned) — rows padded to blockAlign columns with zeros.
    A = np.zeros((kAlloc, kH_aligned), dtype=np.float16)
    A[:kPackedRows, :kH] = A_valid

    # Per-expert weights padded to aligned dimensions with zeros.
    # W1: (kE, kH_aligned, kF_aligned), W2: (kE, kF_aligned, kH_aligned).
    W1_raw = (np.random.randint(-10, 11, size=(kE, kH, kF)).astype(np.float16) / np.float16(3.0))
    W2_raw = (np.random.randint(-10, 11, size=(kE, kF, kH)).astype(np.float16) / np.float16(3.0))
    W1 = np.zeros((kE, kH_aligned, kF_aligned), dtype=np.float16)
    W2 = np.zeros((kE, kF_aligned, kH_aligned), dtype=np.float16)
    W1[:, :kH, :kF] = W1_raw
    W2[:, :kF, :kH] = W2_raw

    # Golden B (fp32): use unpadded raw weights for the reference computation.
    # The kernel computes the same result because zero-padding contributes 0.
    B = np.zeros((kAlloc, kH), dtype=np.float32)
    for e in range(kE):
        c = int(expert_count[e])
        if c == 0:
            continue
        s = int(expert_start[e])
        A_chunk = A[s:s + c, :kH]   # unpadded logical slice
        Y_pre   = A_chunk.astype(np.float32) @ W1_raw[e].astype(np.float32)
        Y_pre   = np.maximum(Y_pre, 0.0)
        Y       = Y_pre.astype(np.float16)
        B_chunk = Y.astype(np.float32) @ W2_raw[e].astype(np.float32)
        B[s:s + c] = B_chunk.astype(np.float32)

    # Trailing 16 rows of golden B are "don't care"; leave as zeros — the
    # validator only compares the first kPackedRows rows.

    os.makedirs("input",  exist_ok=True)
    os.makedirs("output", exist_ok=True)
    A.tofile           ("./input/input_A.bin")
    expert_count.tofile("./input/input_expert_count.bin")
    expert_start.tofile("./input/input_expert_start.bin")
    W1.tofile          ("./input/input_W1.bin")
    W2.tofile          ("./input/input_W2.bin")
    B.tofile           ("./output/golden_B.bin")

    print(f"[gen_data] kT={kT} kH={kH} kF={kF} kE={kE} kTopK={kTopK} "
          f"kTileM={kTileM} kAlloc={kAlloc}")
    print(f"[gen_data] expert_count = {expert_count.tolist()}")
    print(f"[gen_data] expert_start = {expert_start.tolist()}")
    n_zero = int((B[:kPackedRows] == 0.0).sum())
    n_pos  = int((B[:kPackedRows] >  0.0).sum())
    n_neg  = int((B[:kPackedRows] <  0.0).sum())
    print(f"[gen_data] B valid region: zero={n_zero} pos={n_pos} neg={n_neg}")


if __name__ == "__main__":
    gen_golden_data()
