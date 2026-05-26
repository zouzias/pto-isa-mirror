#!/usr/bin/python3
# coding=utf-8
# --------------------------------------------------------------------------------
# scatter - gen_data.py
#
# Generates inputs (X, expert_id) and reference goldens for the four outputs
# of the scatter kernel (A, A_id, expert_count, expert_start). Generic over
# kTopK in {1, 2, 4, 8, 16}. Must match the constants in scatter_kernel.cpp
# and main.cpp.
#
# Output files (raw little-endian, contiguous, no header):
#   ./input/input_X.bin                  kT * kH            half (fp16)
#   ./input/input_expert_id.bin          kT * kTopK         int32
#   ./output/golden_A.bin                (kT*kTopK + 16) * kH  half  (trailing 16 rows zero)
#   ./output/golden_A_id.bin             (kT*kTopK + 16)       int32 (trailing 16 = -1)
#   ./output/golden_rank_id.bin          (kT*kTopK + 16)       int32 (trailing 16 = -1)
#   ./output/golden_expert_count.bin     kE                    int32
#   ./output/golden_expert_start.bin     kE                    int32
# --------------------------------------------------------------------------------

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from case_utils import load_moe_case

np.random.seed(23)

_case = load_moe_case()
kT = _case["t"]
kH = _case["h"]
kE = _case["e"]
kTopK = _case["topk"]

kPackedRows   = kT * kTopK
kOverspillPad = 16
kAlloc        = kPackedRows + kOverspillPad


def gen_golden_data():
    # X in [-10, 10] / 3 rounded to fp16 (same distribution shape as mani_moe).
    X = (np.random.randint(-10, 11, size=(kT, kH)).astype(np.float16) / np.float16(3.0))

    # Each token's kTopK expert assignments. For kTopK=1, only one column.
    # For kTopK>1, each row needs kTopK *distinct* expert ids (typical top-k
    # output). Use random sampling without replacement per row.
    expert_id = np.zeros((kT, kTopK), dtype=np.int32)
    for t in range(kT):
        expert_id[t] = np.random.choice(kE, size=kTopK, replace=False).astype(np.int32)

    # Histogram + prefix sum.
    expert_id_flat = expert_id.flatten()
    expert_count = np.bincount(expert_id_flat, minlength=kE).astype(np.int32)
    expert_start = np.zeros(kE, dtype=np.int32)
    expert_start[1:] = np.cumsum(expert_count[:-1])

    # Pack: stable argsort over expert_id_flat groups (t, k) pairs by expert
    # in ascending order, preserving the original within-expert ordering —
    # same rule the kernel implements via counter[e]++ in pass 3.
    token_ids_per_pair = np.repeat(np.arange(kT, dtype=np.int32), kTopK)
    ranks_per_pair     = np.tile  (np.arange(kTopK, dtype=np.int32), kT)
    order = np.argsort(expert_id_flat, kind="stable")

    A_valid       = X[token_ids_per_pair[order]]              # (kPackedRows, kH)
    A_id_valid    = token_ids_per_pair[order].astype(np.int32) # (kPackedRows,)
    rank_id_valid = ranks_per_pair[order].astype(np.int32)     # (kPackedRows,)

    # Pad with kOverspillPad trailing rows: zeros for A, -1 for id arrays
    # (matches the host-side memset sentinel used as the "unwritten" marker).
    A       = np.zeros((kAlloc, kH), dtype=np.float16)
    A_id    = np.full((kAlloc,), -1, dtype=np.int32)
    rank_id = np.full((kAlloc,), -1, dtype=np.int32)
    A[:kPackedRows]       = A_valid
    A_id[:kPackedRows]    = A_id_valid
    rank_id[:kPackedRows] = rank_id_valid

    os.makedirs("input",  exist_ok=True)
    os.makedirs("output", exist_ok=True)

    X.tofile           ("./input/input_X.bin")
    expert_id.tofile   ("./input/input_expert_id.bin")
    A.tofile           ("./output/golden_A.bin")
    A_id.tofile        ("./output/golden_A_id.bin")
    rank_id.tofile     ("./output/golden_rank_id.bin")
    expert_count.tofile("./output/golden_expert_count.bin")
    expert_start.tofile("./output/golden_expert_start.bin")

    print(f"[gen_data] kT={kT} kH={kH} kE={kE} kTopK={kTopK} kAlloc={kAlloc}")
    print(f"[gen_data] expert_count = {expert_count.tolist()}")
    print(f"[gen_data] expert_start = {expert_start.tolist()}")


if __name__ == "__main__":
    gen_golden_data()
